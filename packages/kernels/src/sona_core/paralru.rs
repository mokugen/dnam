//! Decoupled `1D` scalar `ParaLRU` and Selective-Duhem stereo-linked `ParaLRU`
//! (`ref[dsp.op-n.core.stereo-simd]`), zero-DC smooth soft rectifier, exact `0-ULP` forward tile
//! recurrence, and `O(log2 N)` cross-tile linear adjoint scan (`ref[dsp.op-n.core.matryoshka-blocks]`).

pub mod adjoint;
pub mod duhem_adjoint;
pub mod duhem_scans;
pub mod scans;

use crate::simd::{GpuVec4, KernelVec4, MAX_PARALRU_R_F32, StereoVec8};

pub use adjoint::{
    ParaLruTileGrads, compose_paralru_adjoint_scan, paralru_adjoint_step_elem,
    run_paralru_adjoint_scan_partial_tile, run_paralru_adjoint_scan_tile,
};
pub use duhem_adjoint::{
    ParaLruDuhemAdjointTrace, ParaLruDuhemTileGrads, paralru_duhem_adjoint_step_elem,
    run_paralru_duhem_adjoint_scan_partial_tile, run_paralru_duhem_adjoint_scan_tile,
};
pub use duhem_scans::{
    MAX_PARALRU_TWO_SVF_G_F32, ParaLruDuhemForwardOut, ParaLruDuhemParams,
    run_paralru_duhem_forward_partial_tile, run_paralru_duhem_forward_tile,
};
pub use scans::{
    LinearScanElem, apply_linear_scan, compose_linear_scan, run_paralru_forward_partial_tile,
    run_paralru_forward_tile, scan_linear_prefix_tile,
};

/// Compile-time `1 / 3` constant for the Selective-Duhem odd cubic saturation `psi_1(z) = z_bar * (1 - z_bar^2 / 3)`.
pub const ONE_THIRD_F32: f32 = 1.0 / 3.0;

/// Maximum clamped Selective-Duhem derivative gate gain `gamma_duhem` (`[0.0, 64.0]`).
pub const MAX_PARALRU_DUHEM_GAMMA_F32: f32 = 64.0;

/// Smoothing radius `epsilon` for `soft_rect_zero_dc_*` (`phi(0) == +0.0`, `phi'(0) == 0.5`).
pub const SOFT_RECT_EPS: f32 = 1.0e-3;
const SOFT_RECT_EPS2: f32 = SOFT_RECT_EPS * SOFT_RECT_EPS;
const SOFT_RECT_HALF_EPS: f32 = 0.5 * SOFT_RECT_EPS;
/// Magnitude threshold above which `z^2 + eps^2` equals `z^2` in `binary32` without `f32` overflow.
const SOFT_RECT_LARGE_Z: f32 = 1.0e15;

const _: () = assert!(ONE_THIRD_F32.to_bits() == 0x3EAA_AAAB);
const _: () = assert!(MAX_PARALRU_DUHEM_GAMMA_F32 == 64.0);
const _: () = assert!(SOFT_RECT_EPS > 0.0 && SOFT_RECT_EPS2 > 0.0);
const _: () = assert!(SOFT_RECT_LARGE_Z > 1.0e6);

/// Clamped scalar `ParaLRU` parameters (`alpha_rel in [0, MAX_PARALRU_R_F32]`, `alpha_atk, kappa in [0, 1]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParaLruParams {
    pub alpha_rel: f32,
    pub alpha_atk: f32,
    pub kappa: f32,
}

#[inline]
pub(super) fn clamp_f32_safe(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() { lo } else { x.clamp(lo, hi) }
}

#[inline(always)]
fn clamp_nonneg_vec4_safe<V: KernelVec4>(x: V, hi: V) -> V {
    let zero = V::zero();
    V::select_le(
        x,
        zero,
        zero,
        V::select_le(x, hi, x, V::select_le(hi, x, hi, zero)),
    )
}

/// Vectorized `4`-lane Selective-Duhem stereo-linked `ParaLRU` coefficients
/// (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaParaLruDuhemCoeffs4<V: KernelVec4> {
    pub alpha_rel: V,
    pub alpha_atk: V,
    pub kappa: V,
    pub gamma_duhem: V,
    pub signed_mix: V,
    pub stereo_link: V,
}

impl<V: KernelVec4> SonaParaLruDuhemCoeffs4<V> {
    /// Clamps 4-lane Selective-Duhem `ParaLRU` coefficients into the unconditionally contractive
    /// domain (`alpha_rel in [0, MAX_PARALRU_R_F32]`, `alpha_atk, kappa, signed_mix, stereo_link in [0, 1]`,
    /// `gamma_duhem in [0, 64]`, mapping `NaN` -> `0.0`).
    #[inline]
    #[must_use]
    pub fn from_clamped(
        alpha_rel: V,
        alpha_atk: V,
        kappa: V,
        gamma_duhem: V,
        signed_mix: V,
        stereo_link: V,
    ) -> Self {
        let one = V::splat(1.0);
        Self {
            alpha_rel: clamp_nonneg_vec4_safe(alpha_rel, V::splat(MAX_PARALRU_R_F32)),
            alpha_atk: clamp_nonneg_vec4_safe(alpha_atk, one),
            kappa: clamp_nonneg_vec4_safe(kappa, one),
            gamma_duhem: clamp_nonneg_vec4_safe(gamma_duhem, V::splat(MAX_PARALRU_DUHEM_GAMMA_F32)),
            signed_mix: clamp_nonneg_vec4_safe(signed_mix, one),
            stereo_link: clamp_nonneg_vec4_safe(stereo_link, one),
        }
    }
}

impl ParaLruParams {
    /// Clamps `ParaLRU` ballistics into the unconditionally contractive domain
    /// (`J_i in (0, MAX_PARALRU_R_F32]`).
    #[inline]
    #[must_use]
    pub fn clamped(alpha_rel: f32, alpha_atk: f32, kappa: f32) -> Self {
        Self {
            alpha_rel: clamp_f32_safe(alpha_rel, 0.0, MAX_PARALRU_R_F32),
            alpha_atk: clamp_f32_safe(alpha_atk, 0.0, 1.0),
            kappa: clamp_f32_safe(kappa, 0.0, 1.0),
        }
    }
}

/// Evaluates the zero-DC smooth rectifier `phi(z) = 0.5 * (z + sqrt(z^2 + eps^2) - eps)` and its
/// analytical derivative `phi'(z) = 0.5 * (1 + z / sqrt(z^2 + eps^2))` on 4 SIMD lanes.
///
/// Guarantees bit-exact `+0.0` at `z == ±0.0`, `0 <= phi'(z) <= 1`, zero `FDIV`, zero `FSQRT`,
/// and cancellation-free evaluation across `[−f32::MAX, +f32::MAX]`.
#[inline(always)]
#[must_use]
pub fn soft_rect_zero_dc_vec4<V: KernelVec4>(z: V) -> (V, V) {
    let zero = V::zero();
    let half = V::splat(0.5);
    let one = V::splat(1.0);
    let eps = V::splat(SOFT_RECT_EPS);
    let eps2 = V::splat(SOFT_RECT_EPS2);
    let half_eps_neg = V::splat(-SOFT_RECT_HALF_EPS);
    let large_z = V::splat(SOFT_RECT_LARGE_Z);

    let z_abs = z.abs();
    let z_safe = z_abs.min(large_z);
    let z2 = z_safe.mul(z_safe);
    let q = z2.add(eps2);
    let inv_r = q.rsqrt_pos_nr3();
    let rad = q.mul(inv_r);

    let inv_rad_plus_eps = rad.add(eps).rcp_pos_nr3();
    let inv_rad_plus_abs = rad.add(z_safe).rcp_pos_nr3();

    let one_plus_frac_eps = one.fma(z_safe, inv_rad_plus_eps);
    let phi_pos_small = half.mul(z_safe.fma(z2, inv_rad_plus_eps));
    let phi_pos = V::select_le(z_abs, large_z, phi_pos_small, z_abs);

    let phi_neg_small = half
        .mul(eps)
        .mul(z_safe.mul(inv_rad_plus_abs))
        .mul(one_plus_frac_eps)
        .neg();
    let phi_neg = V::select_le(z_abs, large_z, phi_neg_small, half_eps_neg);
    let raw_phi = V::select_le(z, zero, phi_neg, phi_pos);
    let phi = V::select_le(z_abs, zero, zero, raw_phi);

    let prime_pos_small = half.mul(one.fma(z_safe, inv_r));
    let prime_pos = V::select_le(z_abs, large_z, prime_pos_small, one);
    let prime_neg_small = half.mul(eps2).mul(inv_r).mul(inv_rad_plus_abs);
    let prime_neg = V::select_le(z_abs, large_z, prime_neg_small, zero);
    let raw_prime = V::select_le(z, zero, prime_neg, prime_pos);
    let phi_prime = V::select_le(z_abs, zero, half, raw_prime).clamp(zero, one);

    (phi, phi_prime)
}

/// Evaluates [`soft_rect_zero_dc_vec4`] across a stereo `StereoVec8` register (`0-ULP` parity).
#[inline(always)]
#[must_use]
pub fn soft_rect_zero_dc_stereo8<S: StereoVec8>(z: S) -> (S, S) {
    let zero = S::zero();
    let half = S::splat(0.5);
    let one = S::splat(1.0);
    let eps = S::splat(SOFT_RECT_EPS);
    let eps2 = S::splat(SOFT_RECT_EPS2);
    let half_eps_neg = S::splat(-SOFT_RECT_HALF_EPS);
    let large_z = S::splat(SOFT_RECT_LARGE_Z);

    let z_abs = z.abs();
    let z_safe = z_abs.min(large_z);
    let z2 = z_safe.mul(z_safe);
    let q = z2.add(eps2);
    let inv_r = q.rsqrt_pos_nr3();
    let rad = q.mul(inv_r);

    let inv_rad_plus_eps = rad.add(eps).rcp_pos_nr3();
    let inv_rad_plus_abs = rad.add(z_safe).rcp_pos_nr3();

    let one_plus_frac_eps = one.fma(z_safe, inv_rad_plus_eps);
    let phi_pos_small = half.mul(z_safe.fma(z2, inv_rad_plus_eps));
    let phi_pos = S::select_le(z_abs, large_z, phi_pos_small, z_abs);

    let phi_neg_small = half
        .mul(eps)
        .mul(z_safe.mul(inv_rad_plus_abs))
        .mul(one_plus_frac_eps)
        .neg();
    let phi_neg = S::select_le(z_abs, large_z, phi_neg_small, half_eps_neg);
    let raw_phi = S::select_le(z, zero, phi_neg, phi_pos);
    let phi = S::select_le(z_abs, zero, zero, raw_phi);

    let prime_pos_small = half.mul(one.fma(z_safe, inv_r));
    let prime_pos = S::select_le(z_abs, large_z, prime_pos_small, one);
    let prime_neg_small = half.mul(eps2).mul(inv_r).mul(inv_rad_plus_abs);
    let prime_neg = S::select_le(z_abs, large_z, prime_neg_small, zero);
    let raw_prime = S::select_le(z, zero, prime_neg, prime_pos);
    let phi_prime = S::select_le(z_abs, zero, half, raw_prime).clamp(zero, one);

    (phi, phi_prime)
}

/// Scalar `f32` wrapper over [`soft_rect_zero_dc_vec4`] for CPU/WGSL scalar scans (`0 FDIV`, `0 FSQRT`).
#[inline(always)]
#[must_use]
pub fn soft_rect_zero_dc_f32(z: f32) -> (f32, f32) {
    let (phi, phi_prime) = soft_rect_zero_dc_vec4(GpuVec4::splat(z));
    (phi.to_array()[0], phi_prime.to_array()[0])
}

/// Steps 4 decoupled `1D` scalar `ParaLRU` lanes (`v_new = alpha_rel * v + (1 - alpha_atk) * phi(u - kappa * alpha_rel * v)`).
#[inline(always)]
pub fn step_paralru_vec4<V: KernelVec4>(
    u: V,
    v_state: &mut V,
    alpha_rel: V,
    alpha_atk: V,
    kappa: V,
) -> V {
    let decayed = alpha_rel.mul(*v_state);
    let arg = u.fms(kappa, decayed);
    let (phi, _) = soft_rect_zero_dc_vec4(arg);
    let one_minus_atk = V::splat(1.0).sub(alpha_atk);
    let v_new = decayed.fma(one_minus_atk, phi).flush_recursive_state();
    *v_state = v_new;
    v_new
}

/// Steps stereo `L+R` `4`-lane `ParaLRU` states in `StereoVec8` (`0-ULP` parity with [`step_paralru_vec4`]).
#[inline(always)]
pub fn step_paralru_stereo8<S: StereoVec8>(
    u: S,
    v_state: &mut S,
    alpha_rel: S::Half,
    alpha_atk: S::Half,
    kappa: S::Half,
) -> S {
    let a_rel = S::broadcast_half(alpha_rel);
    let a_atk = S::broadcast_half(alpha_atk);
    let kap = S::broadcast_half(kappa);
    let decayed = a_rel.mul(*v_state);
    let arg = u.fms(kap, decayed);
    let (phi, _) = soft_rect_zero_dc_stereo8(arg);
    let one_minus_atk = S::splat(1.0).sub(a_atk);
    let v_new = decayed.fma(one_minus_atk, phi).flush_recursive_state();
    *v_state = v_new;
    v_new
}

/// Steps 4 Selective-Duhem `1D` `ParaLRU` lanes (`ref[dsp.op-n.core.matryoshka-blocks]`).
///
/// Reduces bit-for-bit (`0-ULP`) to [`step_paralru_vec4`] when `gamma_duhem == 0.0` and
/// `signed_mix == 0.0`, and canonicalizes Zero-DC (`±0.0`) to exact `+0.0`.
#[inline(always)]
pub fn step_paralru_duhem_vec4<V: KernelVec4>(
    u_rect: V,
    flux_inc: V,
    v_state: &mut V,
    coeffs: &SonaParaLruDuhemCoeffs4<V>,
) -> V {
    let zero = V::zero();
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);

    let flux_abs = flux_inc.flush_recursive_state().abs();
    let duhem_mod = V::select_le(
        coeffs.gamma_duhem,
        zero,
        zero,
        coeffs.gamma_duhem.mul(flux_abs),
    );
    let rho = one.sub(coeffs.alpha_atk).add(duhem_mod).clamp(zero, one);
    let d = coeffs.alpha_rel.mul(*v_state);
    let one_minus_eta = one.sub(coeffs.signed_mix);
    let z = u_rect.fms(coeffs.kappa.mul(one_minus_eta), d);
    let z_bar = z.clamp(neg_one, one);
    let z_bar2 = z_bar.mul(z_bar);
    let psi_1 = z_bar.mul(one.fms(V::splat(ONE_THIRD_F32), z_bar2));
    let (phi_rect, _) = soft_rect_zero_dc_vec4(z);
    let phi_eta = V::select_le(
        coeffs.signed_mix,
        zero,
        phi_rect,
        one_minus_eta.mul(phi_rect).fma(coeffs.signed_mix, psi_1),
    );
    let keep = one.fms(coeffs.signed_mix, rho);
    let v_new = keep
        .mul(d)
        .fma(rho, phi_eta)
        .flush_recursive_state()
        .add(zero);
    *v_state = v_new;
    v_new
}

/// Steps stereo `L+R` `4`-lane Selective-Duhem `ParaLRU` states in `StereoVec8` (`0-ULP` parity
/// with [`step_paralru_duhem_vec4`] and [`step_paralru_stereo8`] at `gamma_duhem = 0, eta = 0`).
#[inline(always)]
pub fn step_paralru_duhem_stereo8<S: StereoVec8>(
    u_rect: S,
    flux_inc: S,
    v_state: &mut S,
    coeffs: &SonaParaLruDuhemCoeffs4<S::Half>,
) -> S {
    let zero = S::zero();
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let a_rel = S::broadcast_half(coeffs.alpha_rel);
    let a_atk = S::broadcast_half(coeffs.alpha_atk);
    let kap = S::broadcast_half(coeffs.kappa);
    let gam = S::broadcast_half(coeffs.gamma_duhem);
    let eta = S::broadcast_half(coeffs.signed_mix);

    let flux_abs = flux_inc.flush_recursive_state().abs();
    let duhem_mod = S::select_le(gam, zero, zero, gam.mul(flux_abs));
    let rho = one.sub(a_atk).add(duhem_mod).clamp(zero, one);
    let d = a_rel.mul(*v_state);
    let one_minus_eta = one.sub(eta);
    let z = u_rect.fms(kap.mul(one_minus_eta), d);
    let z_bar = z.clamp(neg_one, one);
    let z_bar2 = z_bar.mul(z_bar);
    let psi_1 = z_bar.mul(one.fms(S::splat(ONE_THIRD_F32), z_bar2));
    let (phi_rect, _) = soft_rect_zero_dc_stereo8(z);
    let phi_eta = S::select_le(
        eta,
        zero,
        phi_rect,
        one_minus_eta.mul(phi_rect).fma(eta, psi_1),
    );
    let keep = one.fms(eta, rho);
    let v_new = keep
        .mul(d)
        .fma(rho, phi_eta)
        .flush_recursive_state()
        .add(zero);
    *v_state = v_new;
    v_new
}

/// Applies audio-rate stereo linking `v_eff_c = v_c + lambda * (v_mid - v_c)` across a `(L, R)`
/// `KernelVec4` pair (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline(always)]
#[must_use]
pub fn apply_stereo_link_vec4_pair<V: KernelVec4>(v_l: V, v_r: V, lambda: V) -> (V, V) {
    let v_mid = V::splat(0.5).mul(v_l.add(v_r));
    (
        v_l.fma(lambda, v_mid.sub(v_l)),
        v_r.fma(lambda, v_mid.sub(v_r)),
    )
}

/// Applies audio-rate stereo linking `v_eff_c = v_c + lambda * (v_mid - v_c)` across a `StereoVec8`
/// register (`0-ULP` parity with [`apply_stereo_link_vec4_pair`]).
#[inline(always)]
#[must_use]
pub fn apply_stereo_link_stereo8<S: StereoVec8>(v: S, lambda: S::Half) -> S {
    let (v_l, v_r) = v.into_lr();
    let (eff_l, eff_r) = apply_stereo_link_vec4_pair(v_l, v_r, lambda);
    S::from_lr(eff_l, eff_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::InterleavedStereo8;
    use proptest::prelude::*;

    fn soft_rect_f64(z: f64) -> (f64, f64) {
        let eps = SOFT_RECT_EPS as f64;
        let r = (z * z + eps * eps).sqrt();
        let phi = 0.5 * (z + r - eps);
        let dphi = 0.5 * (1.0 + z / r);
        (phi, dphi)
    }

    #[test]
    fn test_soft_rect_zero_dc_exact_bits_and_extremes() {
        for z_zero in [0.0_f32, -0.0_f32] {
            let (phi, dphi) = soft_rect_zero_dc_f32(z_zero);
            assert_eq!(phi.to_bits(), 0, "expected bit-exact +0.0 at z={z_zero:?}");
            assert_eq!(dphi, 0.5);
        }

        let extremes = [
            -1.0e20_f32,
            -1.0e9,
            -100.0,
            -10.0,
            -1.0e-7,
            1.0e-7,
            10.0,
            100.0,
            1.0e9,
            1.0e20,
        ];
        let mut prev_phi = f32::NEG_INFINITY;
        for x in extremes {
            let (phi, dphi) = soft_rect_zero_dc_f32(x);
            assert!(phi.is_finite());
            assert!(dphi.is_finite());
            assert!(phi >= -0.5 * SOFT_RECT_EPS - 1.0e-7);
            assert!((0.0..=1.0).contains(&dphi));
            assert!(phi >= prev_phi);
            prev_phi = phi;
        }

        let v_in = GpuVec4::from_array([-1.0e20, -1.0e-4, 0.0, 1.0e20]);
        let (vp, vdp) = soft_rect_zero_dc_vec4(v_in);
        let st_in = InterleavedStereo8::from_lr(v_in, v_in.neg());
        let (sp, sdp) = soft_rect_zero_dc_stereo8(st_in);
        let (sp_l, _sp_r) = sp.into_lr();
        let (sdp_l, _sdp_r) = sdp.into_lr();
        assert_eq!(vp.to_array(), sp_l.to_array());
        assert_eq!(vdp.to_array(), sdp_l.to_array());
    }

    #[test]
    fn test_paralru_step_zero_equilibrium_and_subnormal_flush() {
        let clamped = ParaLruParams::clamped(2.0, -0.5, 1.5);
        assert_eq!(clamped.alpha_rel, MAX_PARALRU_R_F32);
        assert_eq!(clamped.alpha_atk, 0.0);
        assert_eq!(clamped.kappa, 1.0);
        let nan_clamped = ParaLruParams::clamped(f32::NAN, f32::NAN, f32::NAN);
        assert_eq!(nan_clamped, ParaLruParams::clamped(0.0, 0.0, 0.0));

        let mut v_vec = GpuVec4::zero();
        let y_vec = step_paralru_vec4(
            GpuVec4::zero(),
            &mut v_vec,
            GpuVec4::splat(0.99),
            GpuVec4::splat(0.5),
            GpuVec4::splat(1.0),
        );
        for lane in y_vec.to_array() {
            assert_eq!(lane.to_bits(), 0);
        }

        let mut v_st = InterleavedStereo8::<GpuVec4>::zero();
        let y_st = step_paralru_stereo8(
            InterleavedStereo8::zero(),
            &mut v_st,
            GpuVec4::splat(0.99),
            GpuVec4::splat(0.5),
            GpuVec4::splat(1.0),
        );
        let (yl, yr) = y_st.into_lr();
        assert_eq!(yl.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(yr.to_array().map(f32::to_bits), [0; 4]);

        // Subnormal state decays cleanly to zero without denormal traps.
        let mut v_sub = GpuVec4::splat(f32::from_bits(1));
        let y_sub = step_paralru_vec4(
            GpuVec4::zero(),
            &mut v_sub,
            GpuVec4::splat(0.5),
            GpuVec4::splat(1.0),
            GpuVec4::zero(),
        );
        assert_eq!(y_sub.to_array(), [0.0; 4]);
        assert_eq!(v_sub.to_array(), [0.0; 4]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_soft_rect_matches_f64_oracle_and_finite_diff(z in -10.0_f32..10.0_f32) {
            let (phi, dphi) = soft_rect_zero_dc_f32(z);
            let (phi64, dphi64) = soft_rect_f64(z as f64);
            prop_assert!((phi as f64 - phi64).abs() < 2.0e-6);
            prop_assert!((dphi as f64 - dphi64).abs() < 2.0e-6);

            let h = if z.abs() < 0.05 { 2.0e-5_f32 } else { 5.0e-3_f32 };
            let (p_plus, _) = soft_rect_zero_dc_f32(z + h);
            let (p_minus, _) = soft_rect_zero_dc_f32(z - h);
            let fd = (p_plus - p_minus) / (2.0 * h);
            prop_assert!((dphi - fd).abs() < 5.0e-4, "z={z}, dphi={dphi}, fd={fd}");
        }

        #[test]
        fn prop_duhem_step_neutral_zero_mode_and_stereo_link_parity(
            u_l in prop::array::uniform4(-4.0_f32..4.0_f32),
            u_r in prop::array::uniform4(-4.0_f32..4.0_f32),
            flux_l in prop::array::uniform4(-2.0_f32..2.0_f32),
            flux_r in prop::array::uniform4(-2.0_f32..2.0_f32),
            v0_l in prop::array::uniform4(-2.0_f32..2.0_f32),
            v0_r in prop::array::uniform4(-2.0_f32..2.0_f32),
            alpha_rel in 0.1_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.05_f32..0.95_f32,
            kappa in 0.0_f32..1.0_f32,
            gamma_duhem in 0.0_f32..16.0_f32,
            signed_mix in 0.0_f32..1.0_f32,
            stereo_link in 0.0_f32..1.0_f32,
        ) {
            let coeffs_neutral = SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(alpha_rel),
                GpuVec4::splat(alpha_atk),
                GpuVec4::splat(kappa),
                GpuVec4::zero(),
                GpuVec4::zero(),
                GpuVec4::zero(),
            );
            let mut st_neutral_v = GpuVec4::from_array(v0_l);
            let mut st_duhem_v = GpuVec4::from_array(v0_l);
            let y_neutral = step_paralru_vec4(
                GpuVec4::from_array(u_l),
                &mut st_neutral_v,
                coeffs_neutral.alpha_rel,
                coeffs_neutral.alpha_atk,
                coeffs_neutral.kappa,
            );
            let y_duhem_neutral = step_paralru_duhem_vec4(
                GpuVec4::from_array(u_l),
                GpuVec4::from_array(flux_l),
                &mut st_duhem_v,
                &coeffs_neutral,
            );
            prop_assert_eq!(y_duhem_neutral.to_array().map(f32::to_bits), y_neutral.to_array().map(f32::to_bits));

            let coeffs = SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(alpha_rel),
                GpuVec4::splat(alpha_atk),
                GpuVec4::splat(kappa),
                GpuVec4::splat(gamma_duhem),
                GpuVec4::splat(signed_mix),
                GpuVec4::splat(stereo_link),
            );
            let mut vl = GpuVec4::from_array(v0_l);
            let mut vr = GpuVec4::from_array(v0_r);
            let mut vs8 = InterleavedStereo8::from_lr(vl, vr);
            let rl = step_paralru_duhem_vec4(GpuVec4::from_array(u_l), GpuVec4::from_array(flux_l), &mut vl, &coeffs);
            let rr = step_paralru_duhem_vec4(GpuVec4::from_array(u_r), GpuVec4::from_array(flux_r), &mut vr, &coeffs);
            let rs8 = step_paralru_duhem_stereo8(
                InterleavedStereo8::load_lr(&u_l, &u_r),
                InterleavedStereo8::load_lr(&flux_l, &flux_r),
                &mut vs8,
                &coeffs,
            );
            let (rs8_l, rs8_r) = rs8.into_lr();
            prop_assert_eq!(rs8_l.to_array().map(f32::to_bits), rl.to_array().map(f32::to_bits));
            prop_assert_eq!(rs8_r.to_array().map(f32::to_bits), rr.to_array().map(f32::to_bits));

            let (el, er) = apply_stereo_link_vec4_pair(rl, rr, coeffs.stereo_link);
            let es8 = apply_stereo_link_stereo8(rs8, coeffs.stereo_link);
            let (es8_l, es8_r) = es8.into_lr();
            prop_assert_eq!(es8_l.to_array().map(f32::to_bits), el.to_array().map(f32::to_bits));
            prop_assert_eq!(es8_r.to_array().map(f32::to_bits), er.to_array().map(f32::to_bits));
        }
    }

    #[test]
    fn test_paralru_duhem_clamping_zero_dc_remanence_and_stereo_link() {
        let nan_c = SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(f32::NAN),
            GpuVec4::splat(-1.0),
            GpuVec4::splat(2.0),
            GpuVec4::splat(100.0),
            GpuVec4::splat(-0.0),
            GpuVec4::splat(f32::INFINITY),
        );
        assert_eq!(nan_c.alpha_rel.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(nan_c.alpha_atk.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(nan_c.kappa.to_array(), [1.0; 4]);
        assert_eq!(nan_c.gamma_duhem.to_array(), [64.0; 4]);
        assert_eq!(nan_c.signed_mix.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(nan_c.stereo_link.to_array(), [1.0; 4]);

        // Zero-DC canonical +0.0 even for -0.0 input/state and Inf flux_inc protection.
        let c_active = SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(0.95),
            GpuVec4::splat(1.0),
            GpuVec4::splat(0.7),
            GpuVec4::splat(0.0),
            GpuVec4::splat(1.0),
            GpuVec4::splat(0.5),
        );
        let mut st_z = InterleavedStereo8::<GpuVec4>::splat(-0.0);
        let out_z = step_paralru_duhem_stereo8(
            InterleavedStereo8::splat(-0.0),
            InterleavedStereo8::splat(f32::INFINITY),
            &mut st_z,
            &c_active,
        );
        let (zl, zr) = out_z.into_lr();
        assert_eq!(zl.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(zr.to_array().map(f32::to_bits), [0; 4]);

        // Coercive magnetic remanence: eta=1, alpha_atk=1, flux_inc=0 -> v_new = alpha_rel * v_prev.
        let mut st_rem = InterleavedStereo8::<GpuVec4>::splat_lr(0.75, -0.5);
        let out_rem = step_paralru_duhem_stereo8(
            InterleavedStereo8::splat(0.25),
            InterleavedStereo8::zero(),
            &mut st_rem,
            &c_active,
        );
        assert_eq!(out_rem.into_lr().0.to_array(), [0.95 * 0.75; 4]);
        assert_eq!(out_rem.into_lr().1.to_array(), [0.95 * -0.5; 4]);

        // Stereo-link symmetry when v_l == v_r and convergence at lambda = 1.0.
        let eq_v = InterleavedStereo8::<GpuVec4>::splat(0.42);
        let linked_eq = apply_stereo_link_stereo8(eq_v, GpuVec4::splat(0.73));
        assert_eq!(
            linked_eq.into_lr().0.to_array().map(f32::to_bits),
            [0.42_f32.to_bits(); 4]
        );
        assert_eq!(
            linked_eq.into_lr().1.to_array().map(f32::to_bits),
            [0.42_f32.to_bits(); 4]
        );

        let asym_v = InterleavedStereo8::<GpuVec4>::splat_lr(1.0, -0.5);
        let linked_1 = apply_stereo_link_stereo8(asym_v, GpuVec4::splat(1.0));
        let (l1, r1) = linked_1.into_lr();
        assert!((l1.to_array()[0] - r1.to_array()[0]).abs() < 1.0e-6);
        assert!((l1.to_array()[0] - 0.25).abs() < 1.0e-6);
    }
}
