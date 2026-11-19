//! Decoupled `1D` scalar `ParaLRU` (`ref[dsp.op-n.stage2.conv6-simd]`), zero-DC smooth
//! soft rectifier, `K`-step parallel Newton Blelloch scan, and backward linear adjoint scan.

use crate::simd::{GpuVec4, KernelVec4, MAX_PARALRU_R_F32, StereoVec8};

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
    let v_new = decayed.fma(one_minus_atk, phi);
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
    let v_new = decayed.fma(one_minus_atk, phi);
    *v_state = v_new;
    v_new
}

/// Scalar `1D` affine scan element representing `v_out = a * v_in + b`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearScanElem {
    pub a: f32,
    pub b: f32,
}

impl LinearScanElem {
    /// Identity linear scan element (`a = 1.0`, `b = 0.0`).
    #[inline(always)]
    #[must_use]
    pub const fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }
}

/// Associatively composes `first` followed by `second`: `(a2 * a1, a2 * b1 + b2)`.
#[inline(always)]
#[must_use]
pub fn compose_linear_scan(first: LinearScanElem, second: LinearScanElem) -> LinearScanElem {
    LinearScanElem {
        a: second.a * first.a,
        b: second.a * first.b + second.b,
    }
}

/// Applies `elem` to initial scalar state `v0`.
#[inline(always)]
#[must_use]
pub fn apply_linear_scan(elem: LinearScanElem, v0: f32) -> f32 {
    elem.a * v0 + elem.b
}

/// Forms the exact Newton linearization `(a_n, b_n)` of `ParaLRU` around previous-iterate state
/// `v_prev_est`, avoiding `alpha_rel * v_prev_est` subtraction cancellation algebraically:
/// `a = alpha_rel * (1 - kappa * (1 - alpha_atk) * phi'(z))`,
/// `b = (1 - alpha_atk) * (phi(z) + kappa * alpha_rel * phi'(z) * v_prev_est)`.
#[inline(always)]
#[must_use]
pub fn paralru_newton_step_elem(
    u: f32,
    v_prev_est: f32,
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
) -> LinearScanElem {
    let z = u - kappa * alpha_rel * v_prev_est;
    let (phi, phi_prime) = soft_rect_zero_dc_f32(z);
    let one_minus_atk = 1.0 - alpha_atk;
    let a = alpha_rel * (1.0 - kappa * one_minus_atk * phi_prime);
    let b = one_minus_atk * (phi + kappa * alpha_rel * phi_prime * v_prev_est);
    LinearScanElem { a, b }
}

/// Solves an `N`-sample `ParaLRU` tile in parallel-scan form via an analytical release-floor seed
/// followed by `K` Newton associative prefix scans (`K = 3` in `SONA` WebGPU shaders).
#[inline]
#[must_use]
pub fn run_paralru_newton_scan_tile<const N: usize, const K: usize>(
    u: &[f32; N],
    v_init: f32,
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
) -> [f32; N] {
    let mut v = [0.0_f32; N];
    if N == 0 {
        return v;
    }

    let one_minus_atk = 1.0 - alpha_atk;
    let floor_offset = SOFT_RECT_HALF_EPS * one_minus_atk;

    let mut slope_acc = LinearScanElem::identity();
    let mut floor_a = 1.0_f32;
    let mut floor_d = 0.0_f32;
    let mut floor_b = 0.0_f32;

    for i in 0..N {
        let prev_est = if i == 0 { v_init } else { v[i - 1] };
        let drive_hint = u[i] - kappa * alpha_rel * prev_est;
        let (phi, _) = soft_rect_zero_dc_f32(if drive_hint <= 0.0 { drive_hint } else { u[i] });
        let a_seed = if drive_hint > 0.0 {
            alpha_rel * (1.0 - kappa * one_minus_atk)
        } else {
            alpha_rel
        };
        let b_seed = one_minus_atk * phi;
        slope_acc = compose_linear_scan(
            slope_acc,
            LinearScanElem {
                a: a_seed,
                b: b_seed,
            },
        );
        let v_slope = apply_linear_scan(slope_acc, v_init);

        floor_a *= alpha_rel;
        floor_d = alpha_rel * floor_d - floor_offset;
        floor_b = if i == 0 {
            v_slope
        } else {
            (alpha_rel * floor_b - floor_offset).max(v_slope)
        };
        v[i] = (floor_a * v_init + floor_d).max(floor_b);
    }

    let mut v_next = [0.0_f32; N];
    for _ in 0..K {
        let mut scan_acc = LinearScanElem::identity();
        for i in 0..N {
            let v_prev_est = if i == 0 { v_init } else { v[i - 1] };
            let elem = paralru_newton_step_elem(u[i], v_prev_est, alpha_rel, alpha_atk, kappa);
            scan_acc = compose_linear_scan(scan_acc, elem);
            v_next[i] = apply_linear_scan(scan_acc, v_init);
        }
        v = v_next;
    }

    v
}

/// Forms the reverse-BPTT linear adjoint scan element `(J_n, J_n * adj_v_n)` at step `n`.
#[inline(always)]
#[must_use]
pub fn paralru_adjoint_step_elem(
    u: f32,
    v_prev: f32,
    params: ParaLruParams,
    adj_v: f32,
) -> LinearScanElem {
    let z = u - params.kappa * params.alpha_rel * v_prev;
    let (_, phi_prime) = soft_rect_zero_dc_f32(z);
    let one_minus_atk = 1.0 - params.alpha_atk;
    let jac = params.alpha_rel * (1.0 - params.kappa * one_minus_atk * phi_prime);
    LinearScanElem {
        a: jac,
        b: jac * adj_v,
    }
}

/// Composes two reverse-BPTT `ParaLRU` adjoint elements (`earlier_in_reverse` then `later_in_reverse`).
#[inline(always)]
#[must_use]
pub fn compose_paralru_adjoint_scan(
    later_in_reverse: LinearScanElem,
    earlier_in_reverse: LinearScanElem,
) -> LinearScanElem {
    compose_linear_scan(earlier_in_reverse, later_in_reverse)
}

/// Reverse-BPTT gradients computed over an `N`-sample `ParaLRU` tile.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParaLruTileGrads<const N: usize> {
    pub adj_u: [f32; N],
    pub adj_v_init: f32,
    pub adj_alpha_rel: f32,
    pub adj_alpha_atk: f32,
    pub adj_kappa: f32,
    pub lambda: [f32; N],
}

/// Executes the backward linear adjoint scan over an `N`-sample `ParaLRU` tile, returning gradients
/// with respect to `u[0..N]`, `v_init`, `alpha_rel`, `alpha_atk`, and `kappa`.
#[inline]
#[must_use]
pub fn run_paralru_adjoint_scan_tile<const N: usize>(
    u: &[f32; N],
    v_traj: &[f32; N],
    v_init: f32,
    params: ParaLruParams,
    adj_v: &[f32; N],
) -> ParaLruTileGrads<N> {
    let mut adj_u = [0.0_f32; N];
    let mut lambda = [0.0_f32; N];
    if N == 0 {
        return ParaLruTileGrads {
            adj_u,
            adj_v_init: 0.0,
            adj_alpha_rel: 0.0,
            adj_alpha_atk: 0.0,
            adj_kappa: 0.0,
            lambda,
        };
    }

    let one_minus_atk = 1.0 - params.alpha_atk;
    let mut rev_acc = LinearScanElem::identity();
    let mut adj_alpha_rel = 0.0_f32;
    let mut adj_alpha_atk = 0.0_f32;
    let mut adj_kappa = 0.0_f32;

    for rev_idx in 0..N {
        let i = N - 1 - rev_idx;
        let v_prev = if i == 0 { v_init } else { v_traj[i - 1] };
        let lam_i = adj_v[i] + rev_acc.b;
        lambda[i] = lam_i;

        let z = u[i] - params.kappa * params.alpha_rel * v_prev;
        let (phi, phi_prime) = soft_rect_zero_dc_f32(z);
        let jac = params.alpha_rel * (1.0 - params.kappa * one_minus_atk * phi_prime);

        adj_u[i] = lam_i * one_minus_atk * phi_prime;
        adj_alpha_rel += lam_i * v_prev * (1.0 - params.kappa * one_minus_atk * phi_prime);
        adj_alpha_atk -= lam_i * phi;
        adj_kappa -= lam_i * one_minus_atk * params.alpha_rel * v_prev * phi_prime;

        let step_elem = LinearScanElem {
            a: jac,
            b: jac * adj_v[i],
        };
        rev_acc = compose_paralru_adjoint_scan(step_elem, rev_acc);
    }

    ParaLruTileGrads {
        adj_u,
        adj_v_init: rev_acc.b,
        adj_alpha_rel,
        adj_alpha_atk,
        adj_kappa,
        lambda,
    }
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

    fn paralru_loss_f64(
        u: &[f64],
        v_init: f64,
        alpha_rel: f64,
        alpha_atk: f64,
        kappa: f64,
        adj_v: &[f64],
    ) -> f64 {
        let mut v = v_init;
        let mut loss = 0.0;
        for (idx, &ui) in u.iter().enumerate() {
            let z = ui - kappa * alpha_rel * v;
            let (phi, _) = soft_rect_f64(z);
            v = alpha_rel * v + (1.0 - alpha_atk) * phi;
            loss += adj_v[idx] * v;
        }
        loss
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
    fn test_paralru_zero_equilibrium_and_tile_edge_cases() {
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

        let empty_scan = run_paralru_newton_scan_tile::<0, 3>(&[], 0.0, 0.99, 0.5, 1.0);
        assert!(empty_scan.is_empty());
        let empty_grads = run_paralru_adjoint_scan_tile::<0>(&[], &[], 0.0, clamped, &[]);
        assert_eq!(empty_grads.adj_v_init, 0.0);

        // Impulse + decay from zero state, plus K=0 seed-only check and step-down peak-hold check.
        let mut impulse = [0.0_f32; 32];
        impulse[0] = 10.0;
        let k0 = run_paralru_newton_scan_tile::<32, 0>(&impulse, 0.0, 0.999, 0.0, 1.0);
        let k3 = run_paralru_newton_scan_tile::<32, 3>(&impulse, 0.0, 0.999, 0.0, 1.0);
        assert!((k0[31] - k3[31]).abs() < 1.0e-2);

        let mut step_drop = [2.0_f32; 32];
        for slot in step_drop.iter_mut().take(8) {
            *slot = 10.0;
        }
        let drop_scan = run_paralru_newton_scan_tile::<32, 3>(&step_drop, 0.0, 0.99, 0.0, 1.0);
        let mut v_seq = GpuVec4::zero();
        for (idx, &u_val) in step_drop.iter().enumerate() {
            let expected = step_paralru_vec4(
                GpuVec4::splat(u_val),
                &mut v_seq,
                GpuVec4::splat(0.99),
                GpuVec4::zero(),
                GpuVec4::splat(1.0),
            )
            .to_array()[0];
            assert!(
                (drop_scan[idx] - expected).abs() < 1.0e-4,
                "idx={idx}: scan={}, seq={expected}",
                drop_scan[idx]
            );
        }

        // Negative initial state recovery.
        let zeros = [0.0_f32; 16];
        let neg_init = run_paralru_newton_scan_tile::<16, 3>(&zeros, -2.0, 0.95, 0.2, 0.8);
        let mut v_neg_seq = GpuVec4::splat(-2.0);
        for (idx, &u_val) in zeros.iter().enumerate() {
            let expected = step_paralru_vec4(
                GpuVec4::splat(u_val),
                &mut v_neg_seq,
                GpuVec4::splat(0.95),
                GpuVec4::splat(0.2),
                GpuVec4::splat(0.8),
            )
            .to_array()[0];
            assert!((neg_init[idx] - expected).abs() < 1.0e-4);
        }

        // Verify standalone adjoint step helper and composition identity.
        let p = ParaLruParams::clamped(0.95, 0.4, 0.7);
        let step_adj = paralru_adjoint_step_elem(0.5, 0.2, p, 1.0);
        let id = LinearScanElem::identity();
        assert_eq!(compose_paralru_adjoint_scan(step_adj, id), step_adj);
        assert_eq!(compose_paralru_adjoint_scan(id, step_adj), step_adj);
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
        fn prop_paralru_newton_scan_matches_sequential_and_stereo8(
            u_vec in proptest::collection::vec(-10.0_f32..10.0_f32, 64),
            v_init in 0.0_f32..10.0_f32,
            alpha_rel in 0.05_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.05_f32..0.98_f32,
            kappa in 0.0_f32..1.0_f32,
        ) {
            let mut u = [0.0_f32; 64];
            u.copy_from_slice(&u_vec);

            let scanned = run_paralru_newton_scan_tile::<64, 3>(
                &u,
                v_init,
                alpha_rel,
                alpha_atk,
                kappa,
            );

            let mut v_vec = GpuVec4::splat(v_init);
            let mut v_st = InterleavedStereo8::<GpuVec4>::splat(v_init);
            let a_rel_v = GpuVec4::splat(alpha_rel);
            let a_atk_v = GpuVec4::splat(alpha_atk);
            let kap_v = GpuVec4::splat(kappa);

            for (idx, &ui) in u.iter().enumerate() {
                let y_v = step_paralru_vec4(
                    GpuVec4::splat(ui),
                    &mut v_vec,
                    a_rel_v,
                    a_atk_v,
                    kap_v,
                );
                let y_st = step_paralru_stereo8(
                    InterleavedStereo8::splat(ui),
                    &mut v_st,
                    a_rel_v,
                    a_atk_v,
                    kap_v,
                );
                let (yl, yr) = y_st.into_lr();
                prop_assert_eq!(yl.to_array(), y_v.to_array());
                prop_assert_eq!(yr.to_array(), y_v.to_array());

                let seq_val = y_v.to_array()[0];
                prop_assert!(
                    (scanned[idx] - seq_val).abs() < 1.0e-3,
                    "idx={idx}, scanned={}, seq={seq_val}",
                    scanned[idx]
                );
            }
        }

        #[test]
        fn prop_paralru_adjoint_scan_matches_finite_differences(
            u_vec in proptest::collection::vec(-4.0_f32..4.0_f32, 16),
            adj_vec in proptest::collection::vec(-1.0_f32..1.0_f32, 16),
            v_init in 0.0_f32..3.0_f32,
            alpha_rel in 0.1_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.05_f32..0.95_f32,
            kappa in 0.0_f32..0.95_f32,
        ) {
            let mut u = [0.0_f32; 16];
            let mut adj_v = [0.0_f32; 16];
            u.copy_from_slice(&u_vec);
            adj_v.copy_from_slice(&adj_vec);
            let params = ParaLruParams::clamped(alpha_rel, alpha_atk, kappa);

            let mut v_traj = [0.0_f32; 16];
            let mut v_state = GpuVec4::splat(v_init);
            for (idx, &ui) in u.iter().enumerate() {
                v_traj[idx] = step_paralru_vec4(
                    GpuVec4::splat(ui),
                    &mut v_state,
                    GpuVec4::splat(params.alpha_rel),
                    GpuVec4::splat(params.alpha_atk),
                    GpuVec4::splat(params.kappa),
                )
                .to_array()[0];
            }

            let grads = run_paralru_adjoint_scan_tile::<16>(&u, &v_traj, v_init, params, &adj_v);

            let u64: Vec<f64> = u.iter().map(|&v| v as f64).collect();
            let adj64: Vec<f64> = adj_v.iter().map(|&v| v as f64).collect();
            let v0_64 = v_init as f64;
            let ar_64 = params.alpha_rel as f64;
            let aa_64 = params.alpha_atk as f64;
            let kp_64 = params.kappa as f64;
            let eps = 1.0e-6_f64;

            let fd_v0 = (paralru_loss_f64(&u64, v0_64 + eps, ar_64, aa_64, kp_64, &adj64)
                - paralru_loss_f64(&u64, v0_64 - eps, ar_64, aa_64, kp_64, &adj64))
                / (2.0 * eps);
            let fd_ar = (paralru_loss_f64(&u64, v0_64, ar_64 + eps, aa_64, kp_64, &adj64)
                - paralru_loss_f64(&u64, v0_64, ar_64 - eps, aa_64, kp_64, &adj64))
                / (2.0 * eps);
            let fd_aa = (paralru_loss_f64(&u64, v0_64, ar_64, aa_64 + eps, kp_64, &adj64)
                - paralru_loss_f64(&u64, v0_64, ar_64, aa_64 - eps, kp_64, &adj64))
                / (2.0 * eps);
            let fd_kp = (paralru_loss_f64(&u64, v0_64, ar_64, aa_64, kp_64 + eps, &adj64)
                - paralru_loss_f64(&u64, v0_64, ar_64, aa_64, kp_64 - eps, &adj64))
                / (2.0 * eps);

            prop_assert!((grads.adj_v_init as f64 - fd_v0).abs() < 1.0e-3);
            prop_assert!((grads.adj_alpha_rel as f64 - fd_ar).abs() < 1.0e-3);
            prop_assert!((grads.adj_alpha_atk as f64 - fd_aa).abs() < 1.0e-3);
            prop_assert!((grads.adj_kappa as f64 - fd_kp).abs() < 1.0e-3);

            for t in 0..16 {
                let mut u_p = u64.clone();
                let mut u_m = u64.clone();
                u_p[t] += eps;
                u_m[t] -= eps;
                let fd_u = (paralru_loss_f64(&u_p, v0_64, ar_64, aa_64, kp_64, &adj64)
                    - paralru_loss_f64(&u_m, v0_64, ar_64, aa_64, kp_64, &adj64))
                    / (2.0 * eps);
                prop_assert!((grads.adj_u[t] as f64 - fd_u).abs() < 1.0e-3);
            }
        }
    }
}
