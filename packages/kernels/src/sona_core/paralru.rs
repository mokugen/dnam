//! Decoupled `1D` scalar `ParaLRU` (`ref[dsp.op-n.stage2.conv6-simd]`), zero-DC smooth
//! soft rectifier, exact `0-ULP` forward tile recurrence, and `O(log2 N)` cross-tile linear
//! adjoint scan (`ref[dsp.op-n.stage2.sub-stages]`).

pub mod adjoint;
pub mod scans;

use crate::simd::{GpuVec4, KernelVec4, MAX_PARALRU_R_F32, StereoVec8};

pub use adjoint::{
    ParaLruTileGrads, compose_paralru_adjoint_scan, paralru_adjoint_step_elem,
    run_paralru_adjoint_scan_partial_tile, run_paralru_adjoint_scan_tile,
};
pub use scans::{
    LinearScanElem, apply_linear_scan, compose_linear_scan, run_paralru_forward_partial_tile,
    run_paralru_forward_tile, scan_linear_prefix_tile,
};

/// Smoothing radius `epsilon` for `soft_rect_zero_dc_*` (`phi(0) == +0.0`, `phi'(0) == 0.5`).
pub const SOFT_RECT_EPS: f32 = 1.0e-3;
const SOFT_RECT_EPS2: f32 = SOFT_RECT_EPS * SOFT_RECT_EPS;
const SOFT_RECT_HALF_EPS: f32 = 0.5 * SOFT_RECT_EPS;
/// Magnitude threshold above which `z^2 + eps^2` equals `z^2` in `binary32` without `f32` overflow.
const SOFT_RECT_LARGE_Z: f32 = 1.0e15;

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
fn clamp_f32_safe(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() { lo } else { x.clamp(lo, hi) }
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
    }
}
