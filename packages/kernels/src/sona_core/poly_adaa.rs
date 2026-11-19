//! Pure-`f32`, division-free divided-difference `PolyADAA1` and $C^2$ cubed-envelope wavefolder
//! kernels (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`,
//! `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`,
//! `ref[dsp.op-n.wavefolder.cubic-c2]`).

use crate::simd::{KernelVec4, StereoVec8};

const ONE_THIRD: f32 = 1.0 / 3.0;
const ONE_FIFTH: f32 = 1.0 / 5.0;
const ONE_SEVENTH: f32 = 1.0 / 7.0;
const ONE_NINTH: f32 = 1.0 / 9.0;
const TWENTY_EIGHT_THIRDS: f32 = 28.0 / 3.0;
const TINY_CROSSING_EPS: f32 = 1.0e-6;

const _: () = assert!(TINY_CROSSING_EPS > 0.0);

#[derive(Copy, Clone, Debug)]
struct SegmentGeom4<V: KernelVec4> {
    lower: V,
    upper: V,
    m: V,
    m2: V,
    d2: V,
    inside: V,
    distance: V,
    below: V,
    above: V,
    max_abs: V,
}

impl<V: KernelVec4> SegmentGeom4<V> {
    #[inline(always)]
    fn from_endpoints(u_prev: V, u_curr: V) -> Self {
        let neg_one = V::splat(-1.0);
        let one = V::splat(1.0);
        let half = V::splat(0.5);
        let zero = V::zero();

        let lower = u_prev.min(u_curr);
        let upper = u_prev.max(u_curr);
        let clamped_lower = lower.clamp(neg_one, one);
        let clamped_upper = upper.clamp(neg_one, one);
        let m = half.mul(clamped_upper.add(clamped_lower));
        let d = half.mul(clamped_upper.sub(clamped_lower));
        let m2 = m.mul(m);
        let d2 = d.mul(d);
        let inside = clamped_upper.sub(clamped_lower).max(zero);
        let distance = upper.sub(lower);
        let below = upper.min(neg_one).sub(lower).max(zero);
        let above = upper.sub(lower.max(one)).max(zero);
        let max_abs = u_prev.abs().max(u_curr.abs());

        Self {
            lower,
            upper,
            m,
            m2,
            d2,
            inside,
            distance,
            below,
            above,
            max_abs,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct SegmentGeom8<S: StereoVec8> {
    lower: S,
    upper: S,
    m: S,
    m2: S,
    d2: S,
    inside: S,
    distance: S,
    below: S,
    above: S,
    max_abs: S,
}

impl<S: StereoVec8> SegmentGeom8<S> {
    #[inline(always)]
    fn from_endpoints(u_prev: S, u_curr: S) -> Self {
        let neg_one = S::splat(-1.0);
        let one = S::splat(1.0);
        let half = S::splat(0.5);
        let zero = S::zero();

        let lower = u_prev.min(u_curr);
        let upper = u_prev.max(u_curr);
        let clamped_lower = lower.clamp(neg_one, one);
        let clamped_upper = upper.clamp(neg_one, one);
        let m = half.mul(clamped_upper.add(clamped_lower));
        let d = half.mul(clamped_upper.sub(clamped_lower));
        let m2 = m.mul(m);
        let d2 = d.mul(d);
        let inside = clamped_upper.sub(clamped_lower).max(zero);
        let distance = upper.sub(lower);
        let below = upper.min(neg_one).sub(lower).max(zero);
        let above = upper.sub(lower.max(one)).max(zero);
        let max_abs = u_prev.abs().max(u_curr.abs());

        Self {
            lower,
            upper,
            m,
            m2,
            d2,
            inside,
            distance,
            below,
            above,
            max_abs,
        }
    }
}

#[inline(always)]
fn all_lanes_le_one_vec4<V: KernelVec4>(max_abs: V) -> bool {
    let arr = max_abs.to_array();
    arr[0] <= 1.0 && arr[1] <= 1.0 && arr[2] <= 1.0 && arr[3] <= 1.0
}

#[inline(always)]
fn all_lanes_le_one_stereo8<S: StereoVec8>(max_abs: S) -> bool {
    let (left, right) = max_abs.into_lr();
    all_lanes_le_one_vec4(left) && all_lanes_le_one_vec4(right)
}

#[inline(always)]
fn eval_interior_odd_even_vec4<V: KernelVec4>(m: V, m2: V, d2: V, max_abs: V) -> (V, V) {
    let zero = V::zero();
    let half = V::splat(0.5);
    let one_half = V::splat(1.5);
    let two = V::splat(2.0);
    let one_third = V::splat(ONE_THIRD);
    let one_fifth = V::splat(ONE_FIFTH);

    let raw_odd = m.mul(one_half.fms(half, m2.add(d2)));
    let odd = V::select_le(max_abs, zero, zero, raw_odd);

    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let linear_part = m2.fma(one_third, d2);
    let quartic_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let even = linear_part.sub(quartic_part);
    (odd, even)
}

#[inline(always)]
fn eval_interior_odd_even_stereo8<S: StereoVec8>(m: S, m2: S, d2: S, max_abs: S) -> (S, S) {
    let zero = S::zero();
    let half = S::splat(0.5);
    let one_half = S::splat(1.5);
    let two = S::splat(2.0);
    let one_third = S::splat(ONE_THIRD);
    let one_fifth = S::splat(ONE_FIFTH);

    let raw_odd = m.mul(one_half.fms(half, m2.add(d2)));
    let odd = S::select_le(max_abs, zero, zero, raw_odd);

    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let linear_part = m2.fma(one_third, d2);
    let quartic_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let even = linear_part.sub(quartic_part);
    (odd, even)
}

#[inline(always)]
fn finish_rail_crossing_odd_even_vec4<V: KernelVec4>(
    geom: &SegmentGeom4<V>,
    odd_int: V,
    even_int: V,
    inv: V,
) -> (V, V) {
    let zero = V::zero();
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);
    let tiny = V::splat(TINY_CROSSING_EPS);

    let odd_cross = geom
        .above
        .sub(geom.below)
        .fma(geom.inside, odd_int)
        .mul(inv);
    let even_cross = geom.inside.mul(even_int).mul(inv);

    let is_int_lo = V::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = V::select_le(geom.upper, one, one, zero);
    let use_int = V::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let odd_base = V::select_le(use_int, zero, odd_cross, odd_int);
    let even_base = V::select_le(use_int, zero, even_cross, even_int);

    let odd_1 = V::select_le(one, geom.lower, one, odd_base);
    let odd_out = V::select_le(geom.upper, neg_one, neg_one, odd_1);

    let even_1 = V::select_le(one, geom.lower, zero, even_base);
    let even_out = V::select_le(geom.upper, neg_one, zero, even_1);
    (odd_out, even_out)
}

#[inline(always)]
fn finish_rail_crossing_odd_even_stereo8<S: StereoVec8>(
    geom: &SegmentGeom8<S>,
    odd_int: S,
    even_int: S,
    inv: S,
) -> (S, S) {
    let zero = S::zero();
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let tiny = S::splat(TINY_CROSSING_EPS);

    let odd_cross = geom
        .above
        .sub(geom.below)
        .fma(geom.inside, odd_int)
        .mul(inv);
    let even_cross = geom.inside.mul(even_int).mul(inv);

    let is_int_lo = S::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = S::select_le(geom.upper, one, one, zero);
    let use_int = S::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let odd_base = S::select_le(use_int, zero, odd_cross, odd_int);
    let even_base = S::select_le(use_int, zero, even_cross, even_int);

    let odd_1 = S::select_le(one, geom.lower, one, odd_base);
    let odd_out = S::select_le(geom.upper, neg_one, neg_one, odd_1);

    let even_1 = S::select_le(one, geom.lower, zero, even_base);
    let even_out = S::select_le(geom.upper, neg_one, zero, even_1);
    (odd_out, even_out)
}

/// Evaluates `PolyADAA1` cubic-odd and quartic-even activations on 4 SIMD lanes, returning
/// `(odd, even, u_curr)` with an interior fast path (`|u| <= 1`) and pure-`f32` rail-crossing
/// correction (`0 FDIV`, exact `0-ULP` antisymmetry/symmetry, `f(+0.0) == +0.0`).
#[inline]
#[must_use]
pub fn eval_poly_adaa1_odd_even_vec4<V: KernelVec4>(u_prev: V, u_curr: V) -> (V, V, V) {
    let geom = SegmentGeom4::from_endpoints(u_prev, u_curr);
    let (odd_int, even_int) = eval_interior_odd_even_vec4(geom.m, geom.m2, geom.d2, geom.max_abs);
    if all_lanes_le_one_vec4(geom.max_abs) {
        return (odd_int, even_int, u_curr);
    }
    let inv = geom.distance.rcp_pos_nr2();
    let (odd, even) = finish_rail_crossing_odd_even_vec4(&geom, odd_int, even_int, inv);
    (odd, even, u_curr)
}

/// Evaluates two `KernelVec4` `PolyADAA1` pairs (`L + R` or dual block) simultaneously, batching
/// rail-crossing interval reciprocals via `KernelVec4::rcp_pos_quad_nr2`
/// (`ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`).
#[inline]
#[must_use]
pub fn eval_poly_adaa1_quad_batched<V: KernelVec4>(u_prev: [V; 2], u_curr: [V; 2]) -> [(V, V); 2] {
    let g0 = SegmentGeom4::from_endpoints(u_prev[0], u_curr[0]);
    let g1 = SegmentGeom4::from_endpoints(u_prev[1], u_curr[1]);
    let (odd0, even0) = eval_interior_odd_even_vec4(g0.m, g0.m2, g0.d2, g0.max_abs);
    let (odd1, even1) = eval_interior_odd_even_vec4(g1.m, g1.m2, g1.d2, g1.max_abs);

    if all_lanes_le_one_vec4(g0.max_abs) && all_lanes_le_one_vec4(g1.max_abs) {
        return [(odd0, even0), (odd1, even1)];
    }

    let one = V::splat(1.0);
    let [inv0, inv1, _, _] = V::rcp_pos_quad_nr2([g0.distance, g1.distance, one, one]);
    [
        finish_rail_crossing_odd_even_vec4(&g0, odd0, even0, inv0),
        finish_rail_crossing_odd_even_vec4(&g1, odd1, even1, inv1),
    ]
}

/// Evaluates `PolyADAA1` odd/even antiderivative quotients on an `8`-lane `StereoVec8` register,
/// matching [`eval_poly_adaa1_odd_even_vec4`] bit-for-bit (`0-ULP` stereo parity).
#[inline]
#[must_use]
pub fn eval_poly_adaa1_stereo8<S: StereoVec8>(u_prev: S, u_curr: S) -> (S, S) {
    let geom = SegmentGeom8::from_endpoints(u_prev, u_curr);
    let (odd_int, even_int) =
        eval_interior_odd_even_stereo8(geom.m, geom.m2, geom.d2, geom.max_abs);
    if all_lanes_le_one_stereo8(geom.max_abs) {
        return (odd_int, even_int);
    }
    let inv = geom.distance.rcp_pos_nr2();
    finish_rail_crossing_odd_even_stereo8(&geom, odd_int, even_int, inv)
}

#[inline(always)]
fn eval_c2_wf_interior_vec4<V: KernelVec4>(m: V, m2: V, d2: V, max_abs: V) -> (V, V) {
    let zero = V::zero();
    let one = V::splat(1.0);
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

    let s = m2.add(d2);
    let oms = one.sub(s);
    let raw_odd = m.mul(oms).mul(oms.mul(oms).fma(four.mul(m2), d2));
    let wf_odd = V::select_le(max_abs, zero, zero, raw_odd);

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

    let wf_even = q2.sub(q8).sub(three.mul(q4.sub(q6)));
    (wf_odd, wf_even)
}

#[inline(always)]
fn eval_c2_wf_interior_stereo8<S: StereoVec8>(m: S, m2: S, d2: S, max_abs: S) -> (S, S) {
    let zero = S::zero();
    let one = S::splat(1.0);
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

    let s = m2.add(d2);
    let oms = one.sub(s);
    let raw_odd = m.mul(oms).mul(oms.mul(oms).fma(four.mul(m2), d2));
    let wf_odd = S::select_le(max_abs, zero, zero, raw_odd);

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

    let wf_even = q2.sub(q8).sub(three.mul(q4.sub(q6)));
    (wf_odd, wf_even)
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
    if all_lanes_le_one_vec4(geom.max_abs) {
        return (wf_odd_int, wf_even_int);
    }

    let zero = V::zero();
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);
    let tiny = V::splat(TINY_CROSSING_EPS);

    let inv = geom.distance.rcp_pos_nr2();
    let wf_odd_cross = geom.inside.mul(wf_odd_int).mul(inv);
    let wf_even_cross = geom.inside.mul(wf_even_int).mul(inv);

    let is_int_lo = V::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = V::select_le(geom.upper, one, one, zero);
    let use_int = V::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let wf_odd_base = V::select_le(use_int, zero, wf_odd_cross, wf_odd_int);
    let wf_even_base = V::select_le(use_int, zero, wf_even_cross, wf_even_int);

    let wf_odd = V::select_le(
        geom.upper,
        neg_one,
        zero,
        V::select_le(one, geom.lower, zero, wf_odd_base),
    );
    let wf_even = V::select_le(
        geom.upper,
        neg_one,
        zero,
        V::select_le(one, geom.lower, zero, wf_even_base),
    );
    (wf_odd, wf_even)
}

/// Evaluates [`eval_c2_wavefolder_adaa_vec4`] across a stereo `StereoVec8` register (`0-ULP` parity).
#[inline]
#[must_use]
pub fn eval_c2_wavefolder_adaa_stereo8<S: StereoVec8>(u_prev: S, u_curr: S) -> (S, S) {
    let geom = SegmentGeom8::from_endpoints(u_prev, u_curr);
    let (wf_odd_int, wf_even_int) =
        eval_c2_wf_interior_stereo8(geom.m, geom.m2, geom.d2, geom.max_abs);
    if all_lanes_le_one_stereo8(geom.max_abs) {
        return (wf_odd_int, wf_even_int);
    }

    let zero = S::zero();
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let tiny = S::splat(TINY_CROSSING_EPS);

    let inv = geom.distance.rcp_pos_nr2();
    let wf_odd_cross = geom.inside.mul(wf_odd_int).mul(inv);
    let wf_even_cross = geom.inside.mul(wf_even_int).mul(inv);

    let is_int_lo = S::select_le(neg_one, geom.lower, one, zero);
    let is_int_hi = S::select_le(geom.upper, one, one, zero);
    let use_int = S::select_le(geom.distance, tiny, one, is_int_lo.mul(is_int_hi));

    let wf_odd_base = S::select_le(use_int, zero, wf_odd_cross, wf_odd_int);
    let wf_even_base = S::select_le(use_int, zero, wf_even_cross, wf_even_int);

    let wf_odd = S::select_le(
        geom.upper,
        neg_one,
        zero,
        S::select_le(one, geom.lower, zero, wf_odd_base),
    );
    let wf_even = S::select_le(
        geom.upper,
        neg_one,
        zero,
        S::select_le(one, geom.lower, zero, wf_even_base),
    );
    (wf_odd, wf_even)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    fn poly_antideriv_f64(u: f64) -> (f64, f64) {
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let u5 = u4 * u;
        (0.75 * u2 - 0.125 * u4, u3 / 3.0 - 0.2 * u5)
    }

    fn poly_adaa_sat_oracle_f64(a_raw: f64, b_raw: f64) -> (f64, f64) {
        let lower = a_raw.min(b_raw);
        let upper = a_raw.max(b_raw);
        if lower >= 1.0 {
            return (1.0, 0.0);
        }
        if upper <= -1.0 {
            return (-1.0, 0.0);
        }
        let cl = lower.max(-1.0);
        let cu = upper.min(1.0);
        let dist = upper - lower;
        if dist < 1.0e-9 {
            let m = 0.5 * (cl + cu);
            let m2 = m * m;
            return (m * (1.5 - 0.5 * m2), m2 - m2 * m2);
        }
        let (fl_o, fl_e) = poly_antideriv_f64(cl);
        let (fu_o, fu_e) = poly_antideriv_f64(cu);
        let below = (upper.min(-1.0) - lower).max(0.0);
        let above = (upper - lower.max(1.0)).max(0.0);
        (((fu_o - fl_o) + above - below) / dist, (fu_e - fl_e) / dist)
    }

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

    fn assert_zero_ulp_antisymmetric(pos: f32, neg: f32) {
        if pos == 0.0 {
            assert_eq!(neg, 0.0);
        } else {
            assert_eq!(neg.to_bits(), (-pos).to_bits());
        }
    }

    #[test]
    fn test_poly_adaa1_exact_zero_and_known_values() {
        for z in [0.0_f32, -0.0_f32] {
            let (odd, even, curr) =
                eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(z), GpuVec4::splat(z));
            assert_eq!(odd.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(curr.to_array()[0], 0.0);

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

        let cases = [
            (0.0_f32, 1.0_f32, 0.625_f64, 2.0_f64 / 15.0),
            (0.0, 2.0, 0.8125, 1.0 / 15.0),
            (-2.0, 0.0, -0.8125, 1.0 / 15.0),
            (-2.0, 2.0, 0.0, 1.0 / 15.0),
            (-1.0, 1.0, 0.0, 2.0 / 15.0),
            (2.0, 4.0, 1.0, 0.0),
            (-4.0, -2.0, -1.0, 0.0),
            (0.25, 0.25, 0.367_187_5, 0.058_593_75),
            (-1.0 - 1.0e-7, -1.0 + 1.0e-7, -1.0, 5.0e-8),
            (1.0 - 1.0e-7, 1.0 + 1.0e-7, 1.0, 5.0e-8),
        ];
        for (a, b, exp_o, exp_e) in cases {
            let (o, e, _) = eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(a), GpuVec4::splat(b));
            assert!(
                (o.to_array()[0] as f64 - exp_o).abs() < 1.0e-5,
                "a={a}, b={b}"
            );
            assert!(
                (e.to_array()[0] as f64 - exp_e).abs() < 1.0e-5,
                "a={a}, b={b}"
            );
        }

        // C^2 wavefolder anchors: outside plateaus -> 0, [-1, 1] -> (0, 16/315), [-2, 2] -> (0, 8/315).
        let (wf_p_o, wf_p_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(2.0), GpuVec4::splat(4.0));
        assert_eq!(wf_p_o.to_array()[0], 0.0);
        assert_eq!(wf_p_e.to_array()[0], 0.0);

        let (wf_n_o, wf_n_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(-4.0), GpuVec4::splat(-2.0));
        assert_eq!(wf_n_o.to_array()[0], 0.0);
        assert_eq!(wf_n_e.to_array()[0], 0.0);

        let (wf_int_o, wf_int_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(-1.0), GpuVec4::splat(1.0));
        assert_eq!(wf_int_o.to_array()[0], 0.0);
        assert!((wf_int_e.to_array()[0] as f64 - 16.0 / 315.0).abs() < 1.0e-6);

        let (wf_cr_o, wf_cr_e) =
            eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(-2.0), GpuVec4::splat(2.0));
        assert_eq!(wf_cr_o.to_array()[0], 0.0);
        assert!((wf_cr_e.to_array()[0] as f64 - 8.0 / 315.0).abs() < 1.0e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_poly_adaa1_interior_matches_oracle_and_zero_ulp_symmetry(
            a0 in -1.0_f32..1.0_f32,
            a1 in -1.0_f32..1.0_f32,
            a2 in -1.0_f32..1.0_f32,
            a3 in -1.0_f32..1.0_f32,
            b0 in -1.0_f32..1.0_f32,
            b1 in -1.0_f32..1.0_f32,
            b2 in -1.0_f32..1.0_f32,
            b3 in -1.0_f32..1.0_f32,
        ) {
            let prev_l = GpuVec4::from_array([a0, a1, a2, a3]);
            let curr_l = GpuVec4::from_array([b0, b1, b2, b3]);
            let prev_r = prev_l.neg();
            let curr_r = curr_l.neg();

            let (o_l, e_l, c_l) = eval_poly_adaa1_odd_even_vec4(prev_l, curr_l);
            let (o_r, e_r, _) = eval_poly_adaa1_odd_even_vec4(prev_r, curr_r);
            prop_assert_eq!(c_l.to_array(), curr_l.to_array());

            let batched = eval_poly_adaa1_quad_batched([prev_l, prev_r], [curr_l, curr_r]);
            prop_assert_eq!(batched[0].0.to_array(), o_l.to_array());
            prop_assert_eq!(batched[0].1.to_array(), e_l.to_array());
            prop_assert_eq!(batched[1].0.to_array(), o_r.to_array());
            prop_assert_eq!(batched[1].1.to_array(), e_r.to_array());

            let st = InterleavedStereo8::from_lr(prev_l, prev_r);
            let st_c = InterleavedStereo8::from_lr(curr_l, curr_r);
            let (st_o, st_e) = eval_poly_adaa1_stereo8(st, st_c);
            let (st_ol, st_or) = st_o.into_lr();
            let (st_el, st_er) = st_e.into_lr();
            prop_assert_eq!(st_ol.to_array(), o_l.to_array());
            prop_assert_eq!(st_or.to_array(), o_r.to_array());
            prop_assert_eq!(st_el.to_array(), e_l.to_array());
            prop_assert_eq!(st_er.to_array(), e_r.to_array());

            for idx in 0..4 {
                assert_zero_ulp_antisymmetric(o_l.to_array()[idx], o_r.to_array()[idx]);
                prop_assert_eq!(e_l.to_array()[idx].to_bits(), e_r.to_array()[idx].to_bits());
                let (exp_o, exp_e) = poly_adaa_sat_oracle_f64(
                    prev_l.to_array()[idx] as f64,
                    curr_l.to_array()[idx] as f64,
                );
                prop_assert!((o_l.to_array()[idx] as f64 - exp_o).abs() < 1.0e-6);
                prop_assert!((e_l.to_array()[idx] as f64 - exp_e).abs() < 1.0e-6);
            }
        }

        #[test]
        fn prop_poly_adaa1_extreme_rail_crossing_matches_oracle_and_zero_ulp_symmetry(
            a0 in -20.0_f32..20.0_f32,
            a1 in -1.0_f32..1.0_f32,
            a2 in -20.0_f32..20.0_f32,
            a3 in -20.0_f32..20.0_f32,
            b0 in -20.0_f32..20.0_f32,
            b1 in -1.0_f32..1.0_f32,
            b2 in -20.0_f32..20.0_f32,
            b3 in -20.0_f32..20.0_f32,
        ) {
            let prev_l = GpuVec4::from_array([a0, a1, a2, a3]);
            let curr_l = GpuVec4::from_array([b0, b1, b2, b3]);
            let prev_r = prev_l.neg();
            let curr_r = curr_l.neg();

            let (o_l, e_l, _) = eval_poly_adaa1_odd_even_vec4(prev_l, curr_l);
            let (o_r, e_r, _) = eval_poly_adaa1_odd_even_vec4(prev_r, curr_r);

            let batched = eval_poly_adaa1_quad_batched([prev_l, prev_r], [curr_l, curr_r]);
            prop_assert_eq!(batched[0].0.to_array(), o_l.to_array());
            prop_assert_eq!(batched[0].1.to_array(), e_l.to_array());
            prop_assert_eq!(batched[1].0.to_array(), o_r.to_array());
            prop_assert_eq!(batched[1].1.to_array(), e_r.to_array());

            let st_p = InterleavedStereo8::from_lr(prev_l, prev_r);
            let st_c = InterleavedStereo8::from_lr(curr_l, curr_r);
            let (st_o, st_e) = eval_poly_adaa1_stereo8(st_p, st_c);
            let (st_ol, st_or) = st_o.into_lr();
            let (st_el, st_er) = st_e.into_lr();
            prop_assert_eq!(st_ol.to_array(), o_l.to_array());
            prop_assert_eq!(st_or.to_array(), o_r.to_array());
            prop_assert_eq!(st_el.to_array(), e_l.to_array());
            prop_assert_eq!(st_er.to_array(), e_r.to_array());

            for idx in 0..4 {
                assert_zero_ulp_antisymmetric(o_l.to_array()[idx], o_r.to_array()[idx]);
                prop_assert_eq!(e_l.to_array()[idx].to_bits(), e_r.to_array()[idx].to_bits());
                let (exp_o, exp_e) = poly_adaa_sat_oracle_f64(
                    prev_l.to_array()[idx] as f64,
                    curr_l.to_array()[idx] as f64,
                );
                prop_assert!((o_l.to_array()[idx] as f64 - exp_o).abs() < 1.0e-5);
                prop_assert!((e_l.to_array()[idx] as f64 - exp_e).abs() < 1.0e-5);
            }
        }

        #[test]
        fn prop_c2_wavefolder_adaa_matches_oracle_and_zero_ulp_symmetry(
            a0 in -20.0_f32..20.0_f32,
            a1 in -1.0_f32..1.0_f32,
            a2 in -20.0_f32..20.0_f32,
            a3 in -20.0_f32..20.0_f32,
            b0 in -20.0_f32..20.0_f32,
            b1 in -1.0_f32..1.0_f32,
            b2 in -20.0_f32..20.0_f32,
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
                assert_zero_ulp_antisymmetric(o_l.to_array()[idx], o_r.to_array()[idx]);
                prop_assert_eq!(e_l.to_array()[idx].to_bits(), e_r.to_array()[idx].to_bits());
                let (exp_o, exp_e) = c2_wf_oracle_f64(
                    prev_l.to_array()[idx] as f64,
                    curr_l.to_array()[idx] as f64,
                );
                prop_assert!((o_l.to_array()[idx] as f64 - exp_o).abs() < 2.0e-5);
                prop_assert!((e_l.to_array()[idx] as f64 - exp_e).abs() < 2.0e-5);
            }
        }
    }
}
