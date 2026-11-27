//! Zero-DC smooth soft rectifiers for `ParaLRU`: $C^2$ quartic spline (`soft_rect_quartic_*`,
//! SONA v4.0, `ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.op-n.commuted.dc-equilibrium]`)
//! and legacy `sqrt`-based v3 rectifier (`soft_rect_zero_dc_*`).

use crate::simd::{GpuVec4, KernelVec4, StereoVec8};

/// Smoothing radius `epsilon` (`1.0e-3`) for `soft_rect_quartic_*` and `soft_rect_zero_dc_*`.
pub const SOFT_RECT_EPS: f32 = 1.0e-3;

/// Transition width `2 * SOFT_RECT_EPS` (`0.002`) of the $C^2$ quartic spline `soft_rect_quartic_*`.
pub const SOFT_RECT_TWO_EPS: f32 = 2.0 * SOFT_RECT_EPS;

/// Precomputed reciprocal `1 / (2 * SOFT_RECT_EPS)` (`500.0`) eliminating runtime `FDIV`.
pub const SOFT_RECT_INV_TWO_EPS: f32 = 1.0 / SOFT_RECT_TWO_EPS;

const SOFT_RECT_EPS2: f32 = SOFT_RECT_EPS * SOFT_RECT_EPS;
const SOFT_RECT_HALF_EPS: f32 = 0.5 * SOFT_RECT_EPS;
/// Magnitude threshold above which `z^2 + eps^2` equals `z^2` in `binary32` without `f32` overflow.
const SOFT_RECT_LARGE_Z: f32 = 1.0e15;

const _: () = assert!(SOFT_RECT_EPS > 0.0 && SOFT_RECT_EPS2 > 0.0);
const _: () = assert!(SOFT_RECT_TWO_EPS == 2.0 * SOFT_RECT_EPS);
const _: () = assert!(crate::simd::const_abs_f32(SOFT_RECT_INV_TWO_EPS - 500.0) < 1.0e-4);
const _: () =
    assert!(crate::simd::const_abs_f32(SOFT_RECT_TWO_EPS * SOFT_RECT_INV_TWO_EPS - 1.0) < 1.0e-6);
const _: () = assert!(SOFT_RECT_LARGE_Z > 1.0e6);

/// Evaluates the $C^2$ quartic spline soft rectifier on `[0, 2*eps]` (`t = clamp(z / (2*eps), 0, 1)`,
/// `phi(z) = eps * t^3 * (2 - t) + max(z - 2*eps, 0)`, `phi'(z) = t^2 * (3 - 2*t)`) on 4 SIMD lanes.
///
/// Guarantees zero negative leak (`phi(z <= 0) == +0.0`, `phi'(z <= 0) == +0.0`), $C^2$ continuity
/// at `0` and `2*eps`, `NaN`/`±Inf` scrubbing via `.max(zero)` with `zero` on RHS, and zero `FDIV`/`FSQRT`
/// (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.op-n.commuted.dc-equilibrium]`).
#[inline(always)]
#[must_use]
pub fn soft_rect_quartic_vec4<V: KernelVec4>(z: V) -> (V, V) {
    let zero = V::zero();
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    let three = V::splat(3.0);
    let eps = V::splat(SOFT_RECT_EPS);
    let two_eps = V::splat(SOFT_RECT_TWO_EPS);
    let inv_two_eps = V::splat(SOFT_RECT_INV_TWO_EPS);

    let z_safe = V::select_le(z.abs(), V::splat(f32::MAX), z, zero);
    let t = z_safe.mul(inv_two_eps).max(zero).min(one);
    let t2 = t.mul(t);
    let t3 = t2.mul(t);
    let spline = eps.mul(t3.mul(two.sub(t)));
    let linear = z_safe.sub(two_eps).max(zero);
    let phi = spline.add(linear).add(zero);
    let phi_prime = t2.mul(three.fms(two, t)).add(zero);
    (phi, phi_prime)
}

/// Evaluates [`soft_rect_quartic_vec4`] across a stereo `StereoVec8` register (`0-ULP` parity).
#[inline(always)]
#[must_use]
pub fn soft_rect_quartic_stereo8<S: StereoVec8>(z: S) -> (S, S) {
    let zero = S::zero();
    let one = S::splat(1.0);
    let two = S::splat(2.0);
    let three = S::splat(3.0);
    let eps = S::splat(SOFT_RECT_EPS);
    let two_eps = S::splat(SOFT_RECT_TWO_EPS);
    let inv_two_eps = S::splat(SOFT_RECT_INV_TWO_EPS);

    let z_safe = S::select_le(z.abs(), S::splat(f32::MAX), z, zero);
    let t = z_safe.mul(inv_two_eps).max(zero).min(one);
    let t2 = t.mul(t);
    let t3 = t2.mul(t);
    let spline = eps.mul(t3.mul(two.sub(t)));
    let linear = z_safe.sub(two_eps).max(zero);
    let phi = spline.add(linear).add(zero);
    let phi_prime = t2.mul(three.fms(two, t)).add(zero);
    (phi, phi_prime)
}

/// Scalar `f32` wrapper over [`soft_rect_quartic_vec4`] for CPU/GPU scalar scans (`0 FDIV`, `0 FSQRT`).
#[inline(always)]
#[must_use]
pub fn soft_rect_quartic_f32(z: f32) -> (f32, f32) {
    let (phi, phi_prime) = soft_rect_quartic_vec4(GpuVec4::splat(z));
    (phi.to_array()[0], phi_prime.to_array()[0])
}

/// Evaluates the v3 zero-DC smooth rectifier `phi(z) = 0.5 * (z + sqrt(z^2 + eps^2) - eps)` and its
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

    fn soft_rect_quartic_f64(z: f64) -> (f64, f64) {
        let eps = f64::from(SOFT_RECT_EPS);
        let two_eps = 2.0 * eps;
        let t = (z / two_eps).clamp(0.0, 1.0);
        let phi = eps * t * t * t * (2.0 - t) + (z - two_eps).max(0.0);
        let dphi = t * t * (3.0 - 2.0 * t);
        (phi, dphi)
    }

    /// verify[dsp.op-n.commuted.dc-equilibrium]
    /// verify[dsp.op-n.core.matryoshka-blocks]
    /// verify[dsp.op-n.core.stereo-simd]
    #[test]
    fn test_soft_rect_quartic_c2_continuity_zero_leak_and_extremes() {
        for z in [-1.0e9_f32, -10.0, -1.0e-5, -0.0, 0.0] {
            let (phi, dphi) = soft_rect_quartic_f32(z);
            assert_eq!(phi.to_bits(), 0, "expected +0.0 at z={z:?}");
            assert_eq!(dphi.to_bits(), 0, "expected +0.0 derivative at z={z:?}");
        }

        // Midpoint and C^2 boundaries: z=eps -> phi'=0.5, z=2*eps -> phi=eps, phi'=1.0, phi''=0.
        let (phi_mid, dphi_mid) = soft_rect_quartic_f32(SOFT_RECT_EPS);
        assert!((phi_mid - 0.1875 * SOFT_RECT_EPS).abs() < 1.0e-8);
        assert!((dphi_mid - 0.5).abs() < 1.0e-7);

        let (phi_2e, dphi_2e) = soft_rect_quartic_f32(SOFT_RECT_TWO_EPS);
        assert!((phi_2e - SOFT_RECT_EPS).abs() < 1.0e-8);
        assert_eq!(dphi_2e, 1.0);

        // C^2 second derivative vanishes at z = 0+ and z = 2*eps-.
        let d2_zero_plus = soft_rect_quartic_f32(1.0e-6).1 / 1.0e-6;
        let d2_2e_minus = (1.0 - soft_rect_quartic_f32(SOFT_RECT_TWO_EPS - 1.0e-6).1) / 1.0e-6;
        assert!(d2_zero_plus.abs() < 2.0, "d2(0+)={d2_zero_plus}");
        assert!(d2_2e_minus.abs() < 2.0, "d2(2eps-)={d2_2e_minus}");

        // NaN / ±Inf / ±1e9 safety + 0-ULP vec4 vs stereo8 parity.
        let v_in = GpuVec4::from_array([f32::NAN, f32::INFINITY, -1.0e9, 1.0e9]);
        let (vp, vdp) = soft_rect_quartic_vec4(v_in);
        let st_in = InterleavedStereo8::from_lr(v_in, v_in.neg());
        let (sp, sdp) = soft_rect_quartic_stereo8(st_in);
        assert_eq!(
            vp.to_array().map(f32::to_bits),
            sp.into_lr().0.to_array().map(f32::to_bits)
        );
        assert_eq!(
            vdp.to_array().map(f32::to_bits),
            sdp.into_lr().0.to_array().map(f32::to_bits)
        );
        assert_eq!(vp.to_array()[0].to_bits(), 0);
        assert_eq!(vp.to_array()[1].to_bits(), 0);
        assert_eq!(vp.to_array()[2].to_bits(), 0);
        assert!(vp.to_array()[3].is_finite() && (vp.to_array()[3] - 1.0e9).abs() < 128.0);
        assert_eq!(vdp.to_array()[3], 1.0);
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_soft_rect_quartic_matches_f64_oracle_and_fd(
            z in prop_oneof![-10.0_f32..10.0_f32, -0.004_f32..0.006_f32],
        ) {
            let (phi, dphi) = soft_rect_quartic_f32(z);
            let (phi64, dphi64) = soft_rect_quartic_f64(f64::from(z));
            prop_assert!(phi >= 0.0);
            prop_assert!((0.0..=1.0).contains(&dphi));
            prop_assert!((f64::from(phi) - phi64).abs() < 1.0e-6);
            prop_assert!((f64::from(dphi) - dphi64).abs() < 1.0e-6);

            let h = 1.0e-6_f64;
            let fd64 = (soft_rect_quartic_f64(f64::from(z) + h).0
                - soft_rect_quartic_f64(f64::from(z) - h).0)
                / (2.0 * h);
            prop_assert!((f64::from(dphi) - fd64).abs() < 1.0e-6);
        }

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
