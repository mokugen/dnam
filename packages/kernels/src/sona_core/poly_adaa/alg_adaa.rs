//! Clamped `AlgADAA1(0.5)` (`MAX_ALG_R = 4096.0`, `NR3`) forward and fused $C^2$ wavefolder
//! kernels (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`,
//! `ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`, `ref[dsp.op-n.commuted.dc-equilibrium]`).

pub mod backward;

pub use backward::{AlgAdaa1BackwardOut, eval_alg_adaa1_backward_f32};

use crate::simd::{GpuVec4, KernelVec4, StereoVec8};

pub use crate::simd::MAX_ALG_R;

use super::{
    SegmentGeom4, SegmentGeom8, all_lanes_le_one_stereo8, all_lanes_le_one_vec4, wavefolder,
};

/// Scrubs non-finite lanes (`NaN`/`±Inf` -> `+0.0`), clamps pre-activation endpoints to
/// `[-MAX_ALG_R, +MAX_ALG_R]` (`[-4096.0, 4096.0]`), and canonicalizes `-0.0` to `+0.0`
/// across 4 SIMD lanes **before** storing `adaa_prev` (`ref[dsp.op-n.polyadaa.clamp]`).
#[inline(always)]
#[must_use]
pub fn scrub_clamped_alg_endpoint_vec4<V: KernelVec4>(r: V) -> V {
    let zero = V::zero();
    let max_abs_finite = V::splat(f32::MAX);
    let max_r = V::splat(MAX_ALG_R);
    let neg_max_r = V::splat(-MAX_ALG_R);
    V::select_le(r.abs(), max_abs_finite, r, zero)
        .clamp(neg_max_r, max_r)
        .add(zero)
}

/// Scrubs non-finite lanes (`NaN`/`±Inf` -> `+0.0`), clamps pre-activation endpoints to
/// `[-MAX_ALG_R, +MAX_ALG_R]` (`[-4096.0, 4096.0]`), and canonicalizes `-0.0` to `+0.0`
/// across an 8-lane stereo register **before** storing `adaa_prev` (`ref[dsp.op-n.polyadaa.clamp]`).
#[inline(always)]
#[must_use]
pub fn scrub_clamped_alg_endpoint_stereo8<S: StereoVec8>(r: S) -> S {
    let zero = S::zero();
    let max_abs_finite = S::splat(f32::MAX);
    let max_r = S::splat(MAX_ALG_R);
    let neg_max_r = S::splat(-MAX_ALG_R);
    S::select_le(r.abs(), max_abs_finite, r, zero)
        .clamp(neg_max_r, max_r)
        .add(zero)
}

/// Scalar `f32` wrapper over [`scrub_clamped_alg_endpoint_vec4`].
#[inline(always)]
#[must_use]
pub fn scrub_clamped_alg_endpoint_f32(r: f32) -> f32 {
    scrub_clamped_alg_endpoint_vec4(GpuVec4::splat(r)).to_array()[0]
}

/// Evaluates clamped `AlgADAA1(0.5)` (`f(r) = r / sqrt(1 + r^2)`, `MAX_ALG_R = 4096.0`, `NR3`)
/// on 4 SIMD lanes, returning `(odd, even, r_curr_safe)` with canonical `+0.0` at equilibrium,
/// `odd in [-1.0, 1.0]`, `even in [0.0, 1.0]`, zero `FDIV`, and zero `FSQRT`
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
#[inline]
#[must_use]
pub fn eval_alg_adaa1_vec4<V: KernelVec4>(r_prev: V, r_curr: V) -> (V, V, V) {
    let zero = V::zero();
    let half = V::splat(0.5);
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);
    let two = V::splat(2.0);

    let r_p = scrub_clamped_alg_endpoint_vec4(r_prev);
    let r_n = scrub_clamped_alg_endpoint_vec4(r_curr);

    let rn2 = r_n.mul(r_n);
    let rp2 = r_p.mul(r_p);
    let q_n = one.add(rn2);
    let q_p = one.add(rp2);

    let inv_s_n = q_n.rsqrt_pos_nr3();
    let inv_s_p = q_p.rsqrt_pos_nr3();
    let s_n = q_n.mul(inv_s_n);
    let s_p = q_p.mul(inv_s_p);

    let t_n = rn2.mul(s_n.add(one).rcp_ge1_nr3());
    let t_p = rp2.mul(s_p.add(one).rcp_ge1_nr3());
    let inv_sum = s_n.add(s_p).rcp_ge1_nr3();

    let raw_odd = r_n
        .add(r_p)
        .mul(inv_sum)
        .mul(one.fma(half, inv_s_n.mul(inv_s_p)));

    let t_sum = t_n.add(t_p);
    let t_prod = t_n.mul(t_p);
    let even_num = two
        .mul(t_sum)
        .add(t_sum.mul(t_sum))
        .add(t_prod.mul(one.add(t_sum)))
        .add(r_n.mul(r_p));
    let raw_even = even_num.mul(inv_sum).mul(inv_s_n).mul(inv_s_p);

    let max_abs = r_p.abs().max(r_n.abs());
    let odd = V::select_le(max_abs, zero, zero, raw_odd.clamp(neg_one, one)).add(zero);
    let even = V::select_le(max_abs, zero, zero, raw_even.clamp(zero, one)).add(zero);
    (odd, even, r_n)
}

/// Evaluates clamped `AlgADAA1(0.5)` across an 8-lane `StereoVec8` register, returning
/// `(odd, even, r_curr_safe)` with `0-ULP` parity against [`eval_alg_adaa1_vec4`].
#[inline]
#[must_use]
pub fn eval_alg_adaa1_stereo8<S: StereoVec8>(r_prev: S, r_curr: S) -> (S, S, S) {
    let zero = S::zero();
    let half = S::splat(0.5);
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let two = S::splat(2.0);

    let r_p = scrub_clamped_alg_endpoint_stereo8(r_prev);
    let r_n = scrub_clamped_alg_endpoint_stereo8(r_curr);

    let rn2 = r_n.mul(r_n);
    let rp2 = r_p.mul(r_p);
    let q_n = one.add(rn2);
    let q_p = one.add(rp2);

    let inv_s_n = q_n.rsqrt_pos_nr3();
    let inv_s_p = q_p.rsqrt_pos_nr3();
    let s_n = q_n.mul(inv_s_n);
    let s_p = q_p.mul(inv_s_p);

    let t_n = rn2.mul(s_n.add(one).rcp_ge1_nr3());
    let t_p = rp2.mul(s_p.add(one).rcp_ge1_nr3());
    let inv_sum = s_n.add(s_p).rcp_ge1_nr3();

    let raw_odd = r_n
        .add(r_p)
        .mul(inv_sum)
        .mul(one.fma(half, inv_s_n.mul(inv_s_p)));

    let t_sum = t_n.add(t_p);
    let t_prod = t_n.mul(t_p);
    let even_num = two
        .mul(t_sum)
        .add(t_sum.mul(t_sum))
        .add(t_prod.mul(one.add(t_sum)))
        .add(r_n.mul(r_p));
    let raw_even = even_num.mul(inv_sum).mul(inv_s_n).mul(inv_s_p);

    let max_abs = r_p.abs().max(r_n.abs());
    let odd = S::select_le(max_abs, zero, zero, raw_odd.clamp(neg_one, one)).add(zero);
    let even = S::select_le(max_abs, zero, zero, raw_even.clamp(zero, one)).add(zero);
    (odd, even, r_n)
}

/// Scalar `f32` wrapper over [`eval_alg_adaa1_vec4`] for CPU/GPU scalar scans (`0 FDIV`, `0 FSQRT`),
/// returning `(odd, even, r_curr_safe)`.
#[inline(always)]
#[must_use]
pub fn eval_alg_adaa1_f32(r_prev: f32, r_curr: f32) -> (f32, f32, f32) {
    let (odd, even, r_safe) = eval_alg_adaa1_vec4(GpuVec4::splat(r_prev), GpuVec4::splat(r_curr));
    (odd.to_array()[0], even.to_array()[0], r_safe.to_array()[0])
}

/// Fused evaluation of clamped `AlgADAA1(0.5)` and odd-only $C^2$ wavefolder blend on 4 SIMD lanes
/// (`all_lanes_le_one` fast path, dropping dead `_wf_even`, `ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`).
#[inline(always)]
#[must_use]
pub fn eval_alg_adaa1_with_wavefolder_vec4<V: KernelVec4>(
    r_prev: V,
    r_curr: V,
    wf_mix: V,
) -> (V, V, V) {
    let zero = V::zero();
    let one = V::splat(1.0);
    let wf_clamped = V::select_le(wf_mix.abs(), V::splat(f32::MAX), wf_mix, zero).clamp(zero, one);
    let (alg_odd, alg_even, r_curr_safe) = eval_alg_adaa1_vec4(r_prev, r_curr);
    if wf_clamped.all_le(zero) {
        return (alg_odd, alg_even, r_curr_safe);
    }

    let r_prev_safe = scrub_clamped_alg_endpoint_vec4(r_prev);
    let geom = SegmentGeom4::from_endpoints(r_prev_safe, r_curr_safe);
    let wf_odd_int =
        wavefolder::eval_c2_wf_odd_interior_vec4(geom.m, geom.m2, geom.d2, geom.max_abs);
    let wf_odd = if all_lanes_le_one_vec4(geom.max_abs) {
        wf_odd_int
    } else {
        let inv = geom.distance.rcp_pos_nr2();
        wavefolder::finish_c2_wf_odd_rail_crossing_vec4(&geom, wf_odd_int, inv)
    };
    let one_minus_wf = one.sub(wf_clamped);
    let mixed_odd = alg_odd.mul(one_minus_wf).fma(wf_clamped, wf_odd).add(zero);
    (mixed_odd, alg_even, r_curr_safe)
}

/// Fused evaluation of clamped `AlgADAA1(0.5)` and odd-only $C^2$ wavefolder blend across an
/// 8-lane `StereoVec8` register (`0-ULP` parity against [`eval_alg_adaa1_with_wavefolder_vec4`]).
#[inline(always)]
#[must_use]
pub fn eval_alg_adaa1_with_wavefolder_stereo8<S: StereoVec8>(
    r_prev: S,
    r_curr: S,
    wf_mix: S::Half,
) -> (S, S, S) {
    let zero_h = S::Half::zero();
    let one_h = S::Half::splat(1.0);
    let wf_clamped = S::Half::select_le(wf_mix.abs(), S::Half::splat(f32::MAX), wf_mix, zero_h)
        .clamp(zero_h, one_h);
    let (alg_odd, alg_even, r_curr_safe) = eval_alg_adaa1_stereo8(r_prev, r_curr);
    if wf_clamped.all_le(zero_h) {
        return (alg_odd, alg_even, r_curr_safe);
    }

    let r_prev_safe = scrub_clamped_alg_endpoint_stereo8(r_prev);
    let geom = SegmentGeom8::from_endpoints(r_prev_safe, r_curr_safe);
    let wf_odd_int =
        wavefolder::eval_c2_wf_odd_interior_stereo8(geom.m, geom.m2, geom.d2, geom.max_abs);
    let wf_odd = if all_lanes_le_one_stereo8(geom.max_abs) {
        wf_odd_int
    } else {
        let inv = geom.distance.rcp_pos_nr2();
        wavefolder::finish_c2_wf_odd_rail_crossing_stereo8(&geom, wf_odd_int, inv)
    };
    let wf_mix_s = S::broadcast_half(wf_clamped);
    let one_minus_wf = S::splat(1.0).sub(wf_mix_s);
    let mixed_odd = alg_odd
        .mul(one_minus_wf)
        .fma(wf_mix_s, wf_odd)
        .add(S::zero());
    (mixed_odd, alg_even, r_curr_safe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::InterleavedStereo8;
    use proptest::prelude::*;

    fn alg_antideriv_f64(r: f64) -> (f64, f64) {
        let s = (1.0 + r * r).sqrt();
        let f_odd = s - 0.5 / s;
        let f_even = r - r / s;
        (f_odd, f_even)
    }

    fn alg_adaa1_oracle_f64(rp_raw: f64, rn_raw: f64) -> (f64, f64) {
        let rp = rp_raw.clamp(-f64::from(MAX_ALG_R), f64::from(MAX_ALG_R));
        let rn = rn_raw.clamp(-f64::from(MAX_ALG_R), f64::from(MAX_ALG_R));
        if rp == 0.0 && rn == 0.0 {
            return (0.0, 0.0);
        }
        let sp = (1.0 + rp * rp).sqrt();
        let sn = (1.0 + rn * rn).sqrt();
        let odd = ((rn + rp) / (sn + sp)) * (1.0 + 0.5 / (sn * sp));
        let even = if (rn - rp).abs() > 1.0e-7 * (1.0 + rn.abs().max(rp.abs())) {
            let (fp_o, fp_e) = alg_antideriv_f64(rp);
            let (fn_o, fn_e) = alg_antideriv_f64(rn);
            let _ = (fp_o, fn_o);
            ((fn_e - fp_e) / (rn - rp)).clamp(0.0, 1.0)
        } else {
            let tp = (rp * rp) / (sp + 1.0);
            let tn = (rn * rn) / (sn + 1.0);
            let ts = tn + tp;
            let num = 2.0 * ts + ts * ts + tn * tp * (1.0 + ts) + rn * rp;
            (num / ((sn + sp) * sn * sp)).clamp(0.0, 1.0)
        };
        (odd, even)
    }

    /// verify[dsp.op-n.commuted.dc-equilibrium]
    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.op-n.polyadaa.stereo-and-rail-guard]
    #[test]
    fn test_alg_adaa1_zero_dc_clamp_nonfinite_and_nr3_accuracy_grid() {
        for z in [0.0_f32, -0.0_f32] {
            let (o, e, r_safe) = eval_alg_adaa1_f32(z, z);
            assert_eq!(o.to_bits(), 0);
            assert_eq!(e.to_bits(), 0);
            assert_eq!(r_safe.to_bits(), 0);

            let (wo, we, wr) = eval_alg_adaa1_with_wavefolder_stereo8(
                InterleavedStereo8::<GpuVec4>::splat(z),
                InterleavedStereo8::<GpuVec4>::splat(z),
                GpuVec4::splat(0.75),
            );
            let (wol, wor) = wo.into_lr();
            let (wel, wer) = we.into_lr();
            let (wrl, wrr) = wr.into_lr();
            assert_eq!(wol.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(wor.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(wel.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(wer.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(wrl.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(wrr.to_array().map(f32::to_bits), [0; 4]);
        }

        // Pre-store clamp and NaN/±Inf scrub at ±1e9 and ±f32::INFINITY.
        let bad_p = GpuVec4::from_array([f32::NAN, f32::INFINITY, -1.0e9, 1.0e9]);
        let bad_c = GpuVec4::from_array([f32::NEG_INFINITY, f32::NAN, 1.0e9, -1.0e9]);
        let (bo, be, bc) = eval_alg_adaa1_vec4(bad_p, bad_c);
        assert_eq!(bo.to_array()[0].to_bits(), 0);
        assert_eq!(be.to_array()[0].to_bits(), 0);
        assert_eq!(bc.to_array()[0].to_bits(), 0);
        assert_eq!(bo.to_array()[1].to_bits(), 0);
        assert_eq!(be.to_array()[1].to_bits(), 0);
        assert_eq!(bc.to_array()[1].to_bits(), 0);
        assert_eq!(bc.to_array()[2], MAX_ALG_R);
        assert_eq!(bc.to_array()[3], -MAX_ALG_R);
        assert!(bo.to_array().iter().all(|v| v.is_finite()));
        assert!(
            be.to_array()
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
        );

        // Grid verification across [-1e9, 1e9] verifying NR3 accuracy bounds on non-FMA GpuVec4
        // (odd <= 5.0e-7, even <= 1.0e-6).
        let pts: [f32; 29] = [
            -1.0e9, -1.0e5, -4096.0, -4000.0, -1000.0, -100.0, -10.0, -2.5, -2.0, -1.0, -0.5, -0.1,
            -1.0e-3, -1.0e-6, 0.0, 1.0e-6, 1.0e-3, 0.1, 0.5, 1.0, 2.0, 2.5, 10.0, 100.0, 1000.0,
            4000.0, 4096.0, 1.0e5, 1.0e9,
        ];
        let mut max_odd_err = 0.0_f64;
        let mut max_even_err = 0.0_f64;
        for &rp in &pts {
            for &rn in &pts {
                let (o, e, _) = eval_alg_adaa1_f32(rp, rn);
                assert!(o.is_finite() && e.is_finite());
                assert!(
                    (-1.0..=1.0).contains(&o),
                    "odd out of [-1, 1] at ({rp}, {rn}): {o}"
                );
                assert!((0.0..=1.0).contains(&e));
                let (eo, ee) = alg_adaa1_oracle_f64(f64::from(rp), f64::from(rn));
                max_odd_err = max_odd_err.max((f64::from(o) - eo).abs());
                max_even_err = max_even_err.max((f64::from(e) - ee).abs());
            }
        }
        assert!(max_odd_err <= 5.0e-7, "max_odd_err={max_odd_err}");
        assert!(max_even_err <= 1.0e-6, "max_even_err={max_even_err}");

        // Regression for NR3 overshoot past ±1.0 at high-amplitude same-sign pairs, and NaN wf_mix scrubbing.
        let reg_p = GpuVec4::from_array([3996.0, -3996.0, 3096.0, 96.0]);
        let reg_c = GpuVec4::from_array([4096.0, -4096.0, 4096.0, 4096.0]);
        let (ro, re, _) = eval_alg_adaa1_vec4(reg_p, reg_c);
        let (rso, rse, _) = eval_alg_adaa1_stereo8(
            InterleavedStereo8::from_lr(reg_p, reg_c),
            InterleavedStereo8::from_lr(reg_c, reg_p),
        );
        for lane in 0..4 {
            assert!(ro.to_array()[lane].abs() <= 1.0);
            assert!((0.0..=1.0).contains(&re.to_array()[lane]));
            assert_eq!(
                rso.into_lr().0.to_array()[lane].to_bits(),
                ro.to_array()[lane].to_bits()
            );
            assert_eq!(
                rse.into_lr().0.to_array()[lane].to_bits(),
                re.to_array()[lane].to_bits()
            );
        }
        assert!(rso.into_lr().1.to_array().iter().all(|v| v.abs() <= 1.0));

        let wf_nan = GpuVec4::from_array([f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 2.0]);
        let (wfo4, wfe4, _) = eval_alg_adaa1_with_wavefolder_vec4(reg_p, reg_c, wf_nan);
        let (wfo_ref, _, _) = eval_alg_adaa1_with_wavefolder_vec4(
            reg_p,
            reg_c,
            GpuVec4::from_array([0.0, 0.0, 0.0, 1.0]),
        );
        let (wfo8, _, _) = eval_alg_adaa1_with_wavefolder_stereo8(
            InterleavedStereo8::from_lr(reg_p, reg_c),
            InterleavedStereo8::from_lr(reg_c, reg_p),
            wf_nan,
        );
        assert_eq!(
            wfo4.to_array().map(f32::to_bits),
            wfo_ref.to_array().map(f32::to_bits)
        );
        assert_eq!(
            wfo8.into_lr().0.to_array().map(f32::to_bits),
            wfo_ref.to_array().map(f32::to_bits)
        );
        assert!(wfe4.to_array().iter().all(|v| (0.0..=1.0).contains(v)));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// verify[dsp.op-n.core.stereo-simd]
        /// verify[arch.fp.range-saturation]
        #[test]
        fn prop_alg_adaa1_and_wavefolder_0ulp_parity_and_req72_bounds(
            pl_a in prop::array::uniform4(prop_oneof![-1.0e9_f32..1.0e9_f32, -4096.0_f32..4096.0_f32, -1.0_f32..1.0_f32, Just(-0.0_f32)]),
            cl_a in prop::array::uniform4(prop_oneof![-1.0e9_f32..1.0e9_f32, -4096.0_f32..4096.0_f32, -1.0_f32..1.0_f32, Just(-0.0_f32)]),
            pr_a in prop::array::uniform4(-8.0_f32..8.0_f32),
            cr_a in prop::array::uniform4(-8.0_f32..8.0_f32),
            wf_a in prop::array::uniform4(prop_oneof![Just(0.0_f32), Just(1.0_f32), 0.01_f32..0.99_f32]),
        ) {
            let pl = GpuVec4::from_array(pl_a);
            let cl = GpuVec4::from_array(cl_a);
            let pr = GpuVec4::from_array(pr_a);
            let cr = GpuVec4::from_array(cr_a);
            let wf = GpuVec4::from_array(wf_a);

            let (ol, el, nl) = eval_alg_adaa1_vec4(pl, cl);
            let (or, er, nr) = eval_alg_adaa1_vec4(pr, cr);
            let (so, se, sn) = eval_alg_adaa1_stereo8(
                InterleavedStereo8::from_lr(pl, pr),
                InterleavedStereo8::from_lr(cl, cr),
            );
            let (sol, sor) = so.into_lr();
            let (sel, ser) = se.into_lr();
            let (snl, snr) = sn.into_lr();
            prop_assert_eq!(sol.to_array().map(f32::to_bits), ol.to_array().map(f32::to_bits));
            prop_assert_eq!(sor.to_array().map(f32::to_bits), or.to_array().map(f32::to_bits));
            prop_assert_eq!(sel.to_array().map(f32::to_bits), el.to_array().map(f32::to_bits));
            prop_assert_eq!(ser.to_array().map(f32::to_bits), er.to_array().map(f32::to_bits));
            prop_assert_eq!(snl.to_array().map(f32::to_bits), nl.to_array().map(f32::to_bits));
            prop_assert_eq!(snr.to_array().map(f32::to_bits), nr.to_array().map(f32::to_bits));

            for i in 0..4 {
                let (eo, ee) = alg_adaa1_oracle_f64(f64::from(pl_a[i]), f64::from(cl_a[i]));
                prop_assert!((f64::from(ol.to_array()[i]) - eo).abs() <= 5.0e-7);
                prop_assert!((f64::from(el.to_array()[i]) - ee).abs() <= 1.0e-6);
                prop_assert!(ol.to_array()[i].abs() <= 1.0);
                prop_assert!((0.0..=1.0).contains(&el.to_array()[i]));
                prop_assert!(nl.to_array()[i].abs() <= MAX_ALG_R);

                // Mechanism A2 odd/even sign-flip parity.
                let (mo, me, _) = eval_alg_adaa1_f32(-pl_a[i], -cl_a[i]);
                if ol.to_array()[i] == 0.0 {
                    prop_assert_eq!(mo.to_bits(), 0);
                } else {
                    prop_assert_eq!(mo.to_bits(), (-ol.to_array()[i]).to_bits());
                }
                prop_assert_eq!(me.to_bits(), el.to_array()[i].to_bits());
            }

            let (wol, wel, wnl) = eval_alg_adaa1_with_wavefolder_vec4(pl, cl, wf);
            let (wor, wer, wnr) = eval_alg_adaa1_with_wavefolder_vec4(pr, cr, wf);
            let (wso, wse, wsn) = eval_alg_adaa1_with_wavefolder_stereo8(
                InterleavedStereo8::from_lr(pl, pr),
                InterleavedStereo8::from_lr(cl, cr),
                wf,
            );
            let (wsol, wsor) = wso.into_lr();
            let (wsel, wser) = wse.into_lr();
            let (wsnl, wsnr) = wsn.into_lr();
            prop_assert_eq!(wsol.to_array().map(f32::to_bits), wol.to_array().map(f32::to_bits));
            prop_assert_eq!(wsor.to_array().map(f32::to_bits), wor.to_array().map(f32::to_bits));
            prop_assert_eq!(wsel.to_array().map(f32::to_bits), wel.to_array().map(f32::to_bits));
            prop_assert_eq!(wser.to_array().map(f32::to_bits), wer.to_array().map(f32::to_bits));
            prop_assert_eq!(wsnl.to_array().map(f32::to_bits), wnl.to_array().map(f32::to_bits));
            prop_assert_eq!(wsnr.to_array().map(f32::to_bits), wnr.to_array().map(f32::to_bits));
            prop_assert_eq!(wel.to_array().map(f32::to_bits), el.to_array().map(f32::to_bits));
            prop_assert_eq!(wer.to_array().map(f32::to_bits), er.to_array().map(f32::to_bits));
        }
    }
}
