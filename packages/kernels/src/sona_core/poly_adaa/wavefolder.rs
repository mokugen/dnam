//! Pure-`f32`, division-free $C^2$ compactly supported cubed-envelope wavefolder `ADAA1`
//! kernels (`w_odd(u) = u_bar * (1 - u_bar^2)^3`, `w_even(u) = u_bar^2 * (1 - u_bar^2)^3`,
//! `ref[dsp.op-n.wavefolder.cubic-c2]`).

use crate::simd::{KernelVec4, StereoVec8};

use super::{ONE_FIFTH, ONE_THIRD, SegmentGeom4, SegmentGeom8, TINY_CROSSING_EPS};

const ONE_SEVENTH: f32 = 1.0 / 7.0;
const ONE_NINTH: f32 = 1.0 / 9.0;
const TWENTY_EIGHT_THIRDS: f32 = 28.0 / 3.0;

/// 2D Quadrature Phase-Space Hysteresis state (`ref[dsp.op-n.hysteresis.quadrature]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadratureHysteresis2D {
    /// Previous real-axis quadrature sample `s_re[n - 1]`.
    pub prev_s_re: f32,
    /// Cross-axis memory coupling gain (default `0.15`).
    pub kappa_quad: f32,
}

impl Default for QuadratureHysteresis2D {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadratureHysteresis2D {
    /// Constructs a resting hysteresis cell initialized at zero (`kappa_quad = 0.15`).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev_s_re: 0.0,
            kappa_quad: 0.15,
        }
    }

    /// Resets lagged quadrature memory to canonical zero equilibrium.
    #[inline]
    pub fn reset(&mut self) {
        self.prev_s_re = 0.0;
    }

    /// Evaluates one frame of `h[n] = s_re[n] * (s_re[n] - s_re[n-1]) * diff_scale + kappa_quad * s_re[n] * s_im[n]`.
    #[inline(always)]
    pub fn step_with_rate(&mut self, s_re: f32, s_im: f32, diff_scale: f32) -> f32 {
        let s_re_c = crate::simd::scrub_and_flush_f32_bits(s_re);
        let s_im_c = crate::simd::scrub_and_flush_f32_bits(s_im);
        let d_c = crate::simd::scrub_and_flush_f32_bits(diff_scale);
        let delta_re = (s_re_c - self.prev_s_re) * d_c;
        let hyst = s_re_c * delta_re + self.kappa_quad * s_re_c * s_im_c;
        self.prev_s_re = s_re_c;
        crate::simd::scrub_and_flush_f32_bits(hyst)
    }

    /// Evaluates at native `diff_scale = 1.0`.
    #[inline(always)]
    pub fn step(&mut self, s_re: f32, s_im: f32) -> f32 {
        self.step_with_rate(s_re, s_im, 1.0)
    }
}

#[inline(always)]
pub(super) fn eval_c2_wf_odd_interior_vec4<V: KernelVec4>(m: V, m2: V, d2: V, max_abs: V) -> V {
    let zero = V::zero();
    let one = V::splat(1.0);
    let four = V::splat(4.0);

    let s = m2.add(d2);
    let oms = one.sub(s);
    let raw_odd = m.mul(oms).mul(oms.mul(oms).fma(four.mul(m2), d2));
    V::select_le(max_abs, zero, zero, raw_odd).add(zero)
}

#[inline(always)]
fn eval_c2_wf_interior_vec4<V: KernelVec4>(m: V, m2: V, d2: V, max_abs: V) -> (V, V) {
    let zero = V::zero();
    let two = V::splat(2.0);
    let three = V::splat(3.0);
    let four = V::splat(4.0);
    let five = V::splat(5.0);
    let fourteen = V::splat(14.0);
    let one_third = V::splat(ONE_THIRD);
    let one_fifth = V::splat(ONE_FIFTH);
    let one_seventh = V::splat(ONE_SEVENTH);
    let one_ninth = V::splat(ONE_NINTH);
    let twenty_eight_thirds = V::splat(TWENTY_EIGHT_THIRDS);

    let wf_odd = eval_c2_wf_odd_interior_vec4(m, m2, d2, max_abs);

    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let m6 = m4.mul(m2);
    let d6 = d4.mul(d2);
    let m8 = m4.mul(m4);
    let d8 = d4.mul(d4);

    let q2 = m2.fma(one_third, d2);
    let q4 = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let q6 = m6
        .fma(five.mul(m4), d2)
        .fma(three.mul(m2), d4)
        .fma(one_seventh, d6);
    let q8 = m8
        .fma(twenty_eight_thirds.mul(m6), d2)
        .fma(fourteen.mul(m4), d4)
        .fma(four.mul(m2), d6)
        .fma(one_ninth, d8);

    let raw_even = q2.sub(q8).sub(three.mul(q4.sub(q6)));
    let wf_even = V::select_le(max_abs, zero, zero, raw_even).add(zero);
    (wf_odd, wf_even)
}

#[inline(always)]
pub(super) fn eval_c2_wf_odd_interior_stereo8<S: StereoVec8>(m: S, m2: S, d2: S, max_abs: S) -> S {
    let zero = S::zero();
    let one = S::splat(1.0);
    let four = S::splat(4.0);

    let s = m2.add(d2);
    let oms = one.sub(s);
    let raw_odd = m.mul(oms).mul(oms.mul(oms).fma(four.mul(m2), d2));
    S::select_le(max_abs, zero, zero, raw_odd).add(zero)
}

#[inline(always)]
fn eval_c2_wf_interior_stereo8<S: StereoVec8>(m: S, m2: S, d2: S, max_abs: S) -> (S, S) {
    let zero = S::zero();
    let two = S::splat(2.0);
    let three = S::splat(3.0);
    let four = S::splat(4.0);
    let five = S::splat(5.0);
    let fourteen = S::splat(14.0);
    let one_third = S::splat(ONE_THIRD);
    let one_fifth = S::splat(ONE_FIFTH);
    let one_seventh = S::splat(ONE_SEVENTH);
    let one_ninth = S::splat(ONE_NINTH);
    let twenty_eight_thirds = S::splat(TWENTY_EIGHT_THIRDS);

    let wf_odd = eval_c2_wf_odd_interior_stereo8(m, m2, d2, max_abs);

    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let m6 = m4.mul(m2);
    let d6 = d4.mul(d2);
    let m8 = m4.mul(m4);
    let d8 = d4.mul(d4);

    let q2 = m2.fma(one_third, d2);
    let q4 = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let q6 = m6
        .fma(five.mul(m4), d2)
        .fma(three.mul(m2), d4)
        .fma(one_seventh, d6);
    let q8 = m8
        .fma(twenty_eight_thirds.mul(m6), d2)
        .fma(fourteen.mul(m4), d4)
        .fma(four.mul(m2), d6)
        .fma(one_ninth, d8);

    let raw_even = q2.sub(q8).sub(three.mul(q4.sub(q6)));
    let wf_even = S::select_le(max_abs, zero, zero, raw_even).add(zero);
    (wf_odd, wf_even)
}

#[inline(always)]
pub(super) fn finish_c2_wf_odd_rail_crossing_vec4<V: KernelVec4>(
    geom: &SegmentGeom4<V>,
    wf_odd_int: V,
    inv: V,
) -> V {
    let zero = V::zero();
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);
    let tiny = V::splat(TINY_CROSSING_EPS);

    let wf_odd_cross = geom.inside.mul(wf_odd_int).mul(inv);

    let is_int_lo = V::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = V::select_le(geom.upper, one, one, zero);
    let use_int = V::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let wf_odd_base = V::select_le(use_int, zero, wf_odd_cross, wf_odd_int);

    V::select_le(
        geom.upper,
        neg_one,
        zero,
        V::select_le(one, geom.lower, zero, wf_odd_base),
    )
    .add(zero)
}

#[inline(always)]
pub(super) fn finish_c2_wf_odd_rail_crossing_stereo8<S: StereoVec8>(
    geom: &SegmentGeom8<S>,
    wf_odd_int: S,
    inv: S,
) -> S {
    let zero = S::zero();
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let tiny = S::splat(TINY_CROSSING_EPS);

    let wf_odd_cross = geom.inside.mul(wf_odd_int).mul(inv);

    let is_int_lo = S::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = S::select_le(geom.upper, one, one, zero);
    let use_int = S::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let wf_odd_base = S::select_le(use_int, zero, wf_odd_cross, wf_odd_int);

    S::select_le(
        geom.upper,
        neg_one,
        zero,
        S::select_le(one, geom.lower, zero, wf_odd_base),
    )
    .add(zero)
}

/// Evaluates the $C^2$ compactly supported cubed-envelope wavefolder `ADAA1` odd and even responses
/// `w_odd(u) = u_bar * (1 - u_bar^2)^3`, `w_even(u) = u_bar^2 * (1 - u_bar^2)^3` on 4 SIMD lanes
/// (`ref[dsp.op-n.wavefolder.cubic-c2]`, zero outside `[-1, 1]`).
#[inline]
#[must_use]
pub fn eval_c2_wavefolder_adaa_vec4<V: KernelVec4>(u_prev: V, u_curr: V) -> (V, V) {
    let geom = SegmentGeom4::from_endpoints(u_prev, u_curr);
    let (wf_odd_int, wf_even_int) =
        eval_c2_wf_interior_vec4(geom.m, geom.m2, geom.d2, geom.max_abs);
    if super::all_lanes_le_one_vec4(geom.max_abs) {
        return (wf_odd_int, wf_even_int);
    }

    let zero = V::zero();
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);
    let tiny = V::splat(TINY_CROSSING_EPS);

    let inv = geom.distance.rcp_pos_nr2();
    let wf_odd = finish_c2_wf_odd_rail_crossing_vec4(&geom, wf_odd_int, inv);
    let wf_even_cross = geom.inside.mul(wf_even_int).mul(inv);

    let is_int_lo = V::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = V::select_le(geom.upper, one, one, zero);
    let use_int = V::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let wf_even_base = V::select_le(use_int, zero, wf_even_cross, wf_even_int);
    let wf_even = V::select_le(
        geom.upper,
        neg_one,
        zero,
        V::select_le(one, geom.lower, zero, wf_even_base),
    )
    .add(zero);
    (wf_odd, wf_even)
}

/// Evaluates [`eval_c2_wavefolder_adaa_vec4`] across a stereo `StereoVec8` register (`0-ULP` parity).
#[inline]
#[must_use]
pub fn eval_c2_wavefolder_adaa_stereo8<S: StereoVec8>(u_prev: S, u_curr: S) -> (S, S) {
    let geom = SegmentGeom8::from_endpoints(u_prev, u_curr);
    let (wf_odd_int, wf_even_int) =
        eval_c2_wf_interior_stereo8(geom.m, geom.m2, geom.d2, geom.max_abs);
    if super::all_lanes_le_one_stereo8(geom.max_abs) {
        return (wf_odd_int, wf_even_int);
    }

    let zero = S::zero();
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let tiny = S::splat(TINY_CROSSING_EPS);

    let inv = geom.distance.rcp_pos_nr2();
    let wf_odd = finish_c2_wf_odd_rail_crossing_stereo8(&geom, wf_odd_int, inv);
    let wf_even_cross = geom.inside.mul(wf_even_int).mul(inv);

    let is_int_lo = S::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = S::select_le(geom.upper, one, one, zero);
    let use_int = S::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let wf_even_base = S::select_le(use_int, zero, wf_even_cross, wf_even_int);
    let wf_even = S::select_le(
        geom.upper,
        neg_one,
        zero,
        S::select_le(one, geom.lower, zero, wf_even_base),
    )
    .add(zero);
    (wf_odd, wf_even)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    fn c2_wf_antideriv_f64(u: f64) -> (f64, f64) {
        let ub = u.clamp(-1.0, 1.0);
        let u2 = ub * ub;
        let om = 1.0 - u2;
        let odd_anti = -0.125 * om * om * om * om;
        let u3 = u2 * ub;
        let u5 = u3 * u2;
        let u7 = u5 * u2;
        let u9 = u7 * u2;
        let even_anti = u3 / 3.0 - 0.6 * u5 + (3.0 / 7.0) * u7 - u9 / 9.0;
        (odd_anti, even_anti)
    }

    fn c2_wf_oracle_f64(a_raw: f64, b_raw: f64) -> (f64, f64) {
        let lower = a_raw.min(b_raw);
        let upper = a_raw.max(b_raw);
        if lower >= 1.0 || upper <= -1.0 {
            return (0.0, 0.0);
        }
        let dist = upper - lower;
        if dist < 1.0e-9 {
            let ub = (0.5 * (lower + upper)).clamp(-1.0, 1.0);
            let u2 = ub * ub;
            let om = 1.0 - u2;
            let f = om * om * om;
            return (ub * f, u2 * f);
        }
        let (fl_o, fl_e) = c2_wf_antideriv_f64(lower);
        let (fu_o, fu_e) = c2_wf_antideriv_f64(upper);
        ((fu_o - fl_o) / dist, (fu_e - fl_e) / dist)
    }

    #[test]
    fn test_c2_wavefolder_zero_signed_zero_rails_and_nonfinite() {
        for z in [0.0_f32, -0.0_f32, -1.0_f32, 1.0_f32] {
            let (wf_o, wf_e) = eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(z), GpuVec4::splat(z));
            assert_eq!(wf_o.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(wf_e.to_array().map(f32::to_bits), [0; 4]);

            let st_z = InterleavedStereo8::<GpuVec4>::splat(z);
            let (st_wf_o, st_wf_e) = eval_c2_wavefolder_adaa_stereo8(st_z, st_z);
            let (st_wf_ol, st_wf_or) = st_wf_o.into_lr();
            let (st_wf_el, st_wf_er) = st_wf_e.into_lr();
            assert_eq!(st_wf_ol.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(st_wf_or.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(st_wf_el.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(st_wf_er.to_array().map(f32::to_bits), [0; 4]);
        }

        // Scrubbing NaN/±Inf and pre-clamping ±1e30 prevents Inf - Inf = NaN.
        let bad_p = GpuVec4::from_array([f32::NAN, f32::INFINITY, -1.0e30, 1.0e30]);
        let bad_c = GpuVec4::from_array([f32::NEG_INFINITY, f32::NAN, 1.0e30, -1.0e30]);
        let (bo, be) = eval_c2_wavefolder_adaa_vec4(bad_p, bad_c);
        assert_eq!(bo.to_array()[0].to_bits(), 0);
        assert_eq!(be.to_array()[0].to_bits(), 0);
        assert_eq!(bo.to_array()[1].to_bits(), 0);
        assert_eq!(be.to_array()[1].to_bits(), 0);
        assert!(bo.to_array().iter().all(|v| v.is_finite()));
        assert!(be.to_array().iter().all(|v| v.is_finite()));

        // C^2 wavefolder anchors: outside plateaus -> +0.0, [-1, 1] -> (0, 16/315), [-2, 2] -> (0, 8/315).
        let (wf_p_o, wf_p_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(2.0), GpuVec4::splat(4.0));
        assert_eq!(wf_p_o.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(wf_p_e.to_array().map(f32::to_bits), [0; 4]);

        let (wf_n_o, wf_n_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(-4.0), GpuVec4::splat(-2.0));
        assert_eq!(wf_n_o.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(wf_n_e.to_array().map(f32::to_bits), [0; 4]);

        let (wf_int_o, wf_int_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(-1.0), GpuVec4::splat(1.0));
        assert_eq!(wf_int_o.to_array().map(f32::to_bits), [0; 4]);
        assert!((wf_int_e.to_array()[0] as f64 - 16.0 / 315.0).abs() < 1.0e-6);

        let (wf_cr_o, wf_cr_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(-2.0), GpuVec4::splat(2.0));
        assert_eq!(wf_cr_o.to_array().map(f32::to_bits), [0; 4]);
        assert!((wf_cr_e.to_array()[0] as f64 - 8.0 / 315.0).abs() < 1.0e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_c2_wavefolder_adaa_matches_oracle_and_zero_ulp_symmetry(
            a0 in -20.0_f32..20.0_f32,
            a1 in -1.0_f32..1.0_f32,
            a2 in prop_oneof![Just(-0.0_f32), Just(0.0_f32), -20.0_f32..20.0_f32],
            a3 in -20.0_f32..20.0_f32,
            b0 in -20.0_f32..20.0_f32,
            b1 in -1.0_f32..1.0_f32,
            b2 in prop_oneof![Just(-0.0_f32), Just(0.0_f32), -20.0_f32..20.0_f32],
            b3 in -20.0_f32..20.0_f32,
        ) {
            let prev_l = GpuVec4::from_array([a0, a1, a2, a3]);
            let curr_l = GpuVec4::from_array([b0, b1, b2, b3]);
            let prev_r = prev_l.neg();
            let curr_r = curr_l.neg();

            let (o_l, e_l) = eval_c2_wavefolder_adaa_vec4(prev_l, curr_l);
            let (o_r, e_r) = eval_c2_wavefolder_adaa_vec4(prev_r, curr_r);

            let (st_o, st_e) = eval_c2_wavefolder_adaa_stereo8(
                InterleavedStereo8::from_lr(prev_l, prev_r),
                InterleavedStereo8::from_lr(curr_l, curr_r),
            );
            let (st_ol, st_or) = st_o.into_lr();
            let (st_el, st_er) = st_e.into_lr();
            prop_assert_eq!(st_ol.to_array(), o_l.to_array());
            prop_assert_eq!(st_or.to_array(), o_r.to_array());
            prop_assert_eq!(st_el.to_array(), e_l.to_array());
            prop_assert_eq!(st_er.to_array(), e_r.to_array());

            for idx in 0..4 {
                let pos_o = o_l.to_array()[idx];
                let neg_o = o_r.to_array()[idx];
                if pos_o == 0.0 {
                    prop_assert_eq!(pos_o.to_bits(), 0);
                    prop_assert_eq!(neg_o.to_bits(), 0);
                } else {
                    prop_assert_eq!(neg_o.to_bits(), (-pos_o).to_bits());
                }
                prop_assert_eq!(e_l.to_array()[idx].to_bits(), e_r.to_array()[idx].to_bits());

                let (exp_o, exp_e) = c2_wf_oracle_f64(
                    prev_l.to_array()[idx] as f64,
                    curr_l.to_array()[idx] as f64,
                );
                prop_assert!((pos_o as f64 - exp_o).abs() < 2.0e-5);
                prop_assert!((e_l.to_array()[idx] as f64 - exp_e).abs() < 2.0e-5);
            }
        }
    }

    /// verify[dsp.op-n.wavefolder.cubic-c2]
    #[test]
    fn test_c2_wavefolder_interior_0ulp_equivalence() {
        let pairs: [([f32; 4], [f32; 4]); 4] = [
            ([-1.0, -0.5, 0.0, 0.25], [-0.5, 0.5, 0.0, 0.75]),
            (
                [-0.0, 1.0, -0.999_9, 0.5],
                [0.0, 1.0, 0.999_9, 0.5 + 1.0e-7],
            ),
            ([-0.75, -0.25, 0.125, -0.1], [-0.75, 0.25, -0.125, 0.9]),
            ([-1.0, 1.0, -0.5, 0.0], [-1.0, 1.0, -0.5, 1.0]),
        ];
        for (idx, &(u_prev_arr, u_curr_arr)) in pairs.iter().enumerate() {
            let u_prev = GpuVec4::from_array(u_prev_arr);
            let u_curr = GpuVec4::from_array(u_curr_arr);
            let geom = SegmentGeom4::from_endpoints(u_prev, u_curr);
            let (int_odd, int_even) =
                eval_c2_wf_interior_vec4(geom.m, geom.m2, geom.d2, geom.max_abs);
            let (out_odd, out_even) = eval_c2_wavefolder_adaa_vec4(u_prev, u_curr);
            assert_eq!(
                out_odd.to_array().map(f32::to_bits),
                int_odd.to_array().map(f32::to_bits),
                "0-ULP odd mismatch at vec4 case {idx}"
            );
            assert_eq!(
                out_even.to_array().map(f32::to_bits),
                int_even.to_array().map(f32::to_bits),
                "0-ULP even mismatch at vec4 case {idx}"
            );

            let prev_st = InterleavedStereo8::from_lr(u_prev, u_curr);
            let curr_st = InterleavedStereo8::from_lr(u_curr, u_prev);
            let geom8 = SegmentGeom8::from_endpoints(prev_st, curr_st);
            let (int_odd8, int_even8) =
                eval_c2_wf_interior_stereo8(geom8.m, geom8.m2, geom8.d2, geom8.max_abs);
            let (out_odd8, out_even8) = eval_c2_wavefolder_adaa_stereo8(prev_st, curr_st);
            let (ol, or) = out_odd8.into_lr();
            let (iol, ior) = int_odd8.into_lr();
            let (el, er) = out_even8.into_lr();
            let (iel, ier) = int_even8.into_lr();
            assert_eq!(
                ol.to_array().map(f32::to_bits),
                iol.to_array().map(f32::to_bits)
            );
            assert_eq!(
                or.to_array().map(f32::to_bits),
                ior.to_array().map(f32::to_bits)
            );
            assert_eq!(
                el.to_array().map(f32::to_bits),
                iel.to_array().map(f32::to_bits)
            );
            assert_eq!(
                er.to_array().map(f32::to_bits),
                ier.to_array().map(f32::to_bits)
            );
        }
    }

    /// verify[dsp.op-n.hysteresis.quadrature]
    #[test]
    fn test_quadrature_hysteresis_2d_step_rate_and_zero_dc() {
        let mut hyst = QuadratureHysteresis2D::default();
        assert_eq!(hyst.kappa_quad, 0.15);
        assert_eq!(hyst.step(-0.0, -0.0).to_bits(), 0);
        let h1 = hyst.step(0.5, 0.2);
        assert!((h1 - (0.5 * 0.5 + 0.15 * 0.5 * 0.2)).abs() < 1.0e-6);
        let h2 = hyst.step(0.8, -0.1);
        assert!((h2 - (0.8 * 0.3 + 0.15 * 0.8 * (-0.1))).abs() < 1.0e-6);
        hyst.reset();
        let h3 = hyst.step_with_rate(0.5, 0.2, 2.0);
        assert!((h3 - (0.5 * 0.5 * 2.0 + 0.15 * 0.5 * 0.2)).abs() < 1.0e-6);
    }
}
