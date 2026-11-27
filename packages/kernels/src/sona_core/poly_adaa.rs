//! Pure-`f32`, division-free divided-difference `PolyADAA1` and $C^2$ cubed-envelope wavefolder
//! kernels (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`,
//! `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`,
//! `ref[dsp.op-n.wavefolder.cubic-c2]`).

pub mod wavefolder;

use crate::simd::{KernelVec4, StereoVec8};

pub use wavefolder::{
    QuadratureHysteresis2D, eval_c2_wavefolder_adaa_stereo8, eval_c2_wavefolder_adaa_vec4,
};

pub(super) const ONE_THIRD: f32 = 1.0 / 3.0;
pub(super) const ONE_FIFTH: f32 = 1.0 / 5.0;
pub const TINY_CROSSING_EPS: f32 = 1.0e-6;
/// Pre-clamp endpoint bound ensuring `upper - lower <= 2 * MAX_ADAA_ENDPOINT` never overflows to `+Inf`.
pub const MAX_ADAA_ENDPOINT: f32 = 1.0e18;

const _: () = assert!(TINY_CROSSING_EPS > 0.0);
const _: () = assert!(MAX_ADAA_ENDPOINT > 1.0 && (2.0 * MAX_ADAA_ENDPOINT).is_finite());

/// Scrubs non-finite lanes (`NaN`/`±Inf` -> `+0.0`), clamps finite extremes to `[-1e18, 1e18]`,
/// and canonicalizes `-0.0` to `+0.0` across 4 SIMD lanes.
#[inline(always)]
pub(crate) fn scrub_endpoint_vec4<V: KernelVec4>(u: V) -> V {
    let zero = V::zero();
    let max_abs_finite = V::splat(f32::MAX);
    let max_ep = V::splat(MAX_ADAA_ENDPOINT);
    let neg_max_ep = V::splat(-MAX_ADAA_ENDPOINT);
    V::select_le(u.abs(), max_abs_finite, u, zero)
        .clamp(neg_max_ep, max_ep)
        .add(zero)
}

/// Scrubs non-finite lanes (`NaN`/`±Inf` -> `+0.0`), clamps finite extremes to `[-1e18, 1e18]`,
/// and canonicalizes `-0.0` to `+0.0` across an 8-lane stereo register.
#[inline(always)]
pub(crate) fn scrub_endpoint_stereo8<S: StereoVec8>(u: S) -> S {
    let zero = S::zero();
    let max_abs_finite = S::splat(f32::MAX);
    let max_ep = S::splat(MAX_ADAA_ENDPOINT);
    let neg_max_ep = S::splat(-MAX_ADAA_ENDPOINT);
    S::select_le(u.abs(), max_abs_finite, u, zero)
        .clamp(neg_max_ep, max_ep)
        .add(zero)
}

#[derive(Copy, Clone, Debug)]
pub(super) struct SegmentGeom4<V: KernelVec4> {
    pub(super) u_curr_safe: V,
    pub(super) lower: V,
    pub(super) upper: V,
    pub(super) m: V,
    pub(super) m2: V,
    pub(super) d2: V,
    pub(super) inside: V,
    pub(super) distance: V,
    pub(super) below: V,
    pub(super) above: V,
    pub(super) max_abs: V,
}

impl<V: KernelVec4> SegmentGeom4<V> {
    #[inline(always)]
    pub(super) fn from_endpoints(u_prev: V, u_curr: V) -> Self {
        let neg_one = V::splat(-1.0);
        let one = V::splat(1.0);
        let half = V::splat(0.5);
        let zero = V::zero();

        let u0 = scrub_endpoint_vec4(u_prev);
        let u1 = scrub_endpoint_vec4(u_curr);
        let lower = u0.min(u1);
        let upper = u0.max(u1);
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
        let max_abs = u0.abs().max(u1.abs());

        Self {
            u_curr_safe: u1,
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
pub(super) struct SegmentGeom8<S: StereoVec8> {
    pub(super) u_curr_safe: S,
    pub(super) lower: S,
    pub(super) upper: S,
    pub(super) m: S,
    pub(super) m2: S,
    pub(super) d2: S,
    pub(super) inside: S,
    pub(super) distance: S,
    pub(super) below: S,
    pub(super) above: S,
    pub(super) max_abs: S,
}

impl<S: StereoVec8> SegmentGeom8<S> {
    #[inline(always)]
    pub(super) fn from_endpoints(u_prev: S, u_curr: S) -> Self {
        let neg_one = S::splat(-1.0);
        let one = S::splat(1.0);
        let half = S::splat(0.5);
        let zero = S::zero();

        let u0 = scrub_endpoint_stereo8(u_prev);
        let u1 = scrub_endpoint_stereo8(u_curr);
        let lower = u0.min(u1);
        let upper = u0.max(u1);
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
        let max_abs = u0.abs().max(u1.abs());

        Self {
            u_curr_safe: u1,
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
pub(super) fn all_lanes_le_one_vec4<V: KernelVec4>(max_abs: V) -> bool {
    max_abs.all_le(V::splat(1.0))
}

#[inline(always)]
pub(super) fn all_lanes_le_one_stereo8<S: StereoVec8>(max_abs: S) -> bool {
    max_abs.all_le(S::splat(1.0))
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
    let odd = V::select_le(max_abs, zero, zero, raw_odd).add(zero);

    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let linear_part = m2.fma(one_third, d2);
    let quartic_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let raw_even = linear_part.sub(quartic_part);
    let even = V::select_le(max_abs, zero, zero, raw_even).add(zero);
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
    let odd = S::select_le(max_abs, zero, zero, raw_odd).add(zero);

    let m4 = m2.mul(m2);
    let d4 = d2.mul(d2);
    let linear_part = m2.fma(one_third, d2);
    let quartic_part = m4.fma(two.mul(m2), d2).fma(one_fifth, d4);
    let raw_even = linear_part.sub(quartic_part);
    let even = S::select_le(max_abs, zero, zero, raw_even).add(zero);
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
    let odd_out = V::select_le(geom.upper, neg_one, neg_one, odd_1).add(zero);

    let even_1 = V::select_le(one, geom.lower, zero, even_base);
    let even_out = V::select_le(geom.upper, neg_one, zero, even_1).add(zero);
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
    let odd_out = S::select_le(geom.upper, neg_one, neg_one, odd_1).add(zero);

    let even_1 = S::select_le(one, geom.lower, zero, even_base);
    let even_out = S::select_le(geom.upper, neg_one, zero, even_1).add(zero);
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
        return (odd_int, even_int, geom.u_curr_safe);
    }
    let inv = geom.distance.rcp_pos_nr2();
    let (odd, even) = finish_rail_crossing_odd_even_vec4(&geom, odd_int, even_int, inv);
    (odd, even, geom.u_curr_safe)
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

/// Fused evaluation of `PolyADAA1` odd/even and odd-only $C^2$ wavefolder blend on 4 SIMD lanes,
/// sharing segment geometry, the `all_lanes_le_one` fast path, and a single `rcp_pos_nr2` on
/// rail crossings while skipping `_wf_even` (`0-ULP` identical to separate evaluation + blend).
#[inline(always)]
#[must_use]
pub fn eval_poly_adaa1_with_wavefolder_vec4<V: KernelVec4>(
    u_prev: V,
    u_curr: V,
    wf_mix: V,
) -> (V, V, V) {
    let wf_clamped = wf_mix.clamp(V::zero(), V::splat(1.0));
    let geom = SegmentGeom4::from_endpoints(u_prev, u_curr);
    let (odd_int, even_int) = eval_interior_odd_even_vec4(geom.m, geom.m2, geom.d2, geom.max_abs);
    let interior = all_lanes_le_one_vec4(geom.max_abs);
    if wf_clamped.all_le(V::zero()) {
        if interior {
            return (odd_int, even_int, geom.u_curr_safe);
        }
        let inv = geom.distance.rcp_pos_nr2();
        let (odd, even) = finish_rail_crossing_odd_even_vec4(&geom, odd_int, even_int, inv);
        return (odd, even, geom.u_curr_safe);
    }
    let wf_odd_int =
        wavefolder::eval_c2_wf_odd_interior_vec4(geom.m, geom.m2, geom.d2, geom.max_abs);
    let (adaa_odd, adaa_even, wf_odd) = if interior {
        (odd_int, even_int, wf_odd_int)
    } else {
        let inv = geom.distance.rcp_pos_nr2();
        let (odd, even) = finish_rail_crossing_odd_even_vec4(&geom, odd_int, even_int, inv);
        let wf_odd = wavefolder::finish_c2_wf_odd_rail_crossing_vec4(&geom, wf_odd_int, inv);
        (odd, even, wf_odd)
    };
    let one_minus_wf = V::splat(1.0).sub(wf_clamped);
    let mixed_odd = adaa_odd
        .mul(one_minus_wf)
        .fma(wf_clamped, wf_odd)
        .add(V::zero());
    (mixed_odd, adaa_even, geom.u_curr_safe)
}

/// Fused evaluation of `PolyADAA1` odd/even and odd-only $C^2$ wavefolder blend across an
/// `8`-lane `StereoVec8` register, returning `(odd_mixed, even, u_curr_safe)` with `0-ULP`
/// parity against [`eval_poly_adaa1_stereo8`] + [`eval_c2_wavefolder_adaa_stereo8`].
#[inline(always)]
#[must_use]
pub fn eval_poly_adaa1_with_wavefolder_stereo8<S: StereoVec8>(
    u_prev: S,
    u_curr: S,
    wf_mix: S::Half,
) -> (S, S, S) {
    let wf_clamped = wf_mix.clamp(S::Half::zero(), S::Half::splat(1.0));
    let geom = SegmentGeom8::from_endpoints(u_prev, u_curr);
    let (odd_int, even_int) =
        eval_interior_odd_even_stereo8(geom.m, geom.m2, geom.d2, geom.max_abs);
    let interior = all_lanes_le_one_stereo8(geom.max_abs);
    if wf_clamped.all_le(S::Half::zero()) {
        if interior {
            return (odd_int, even_int, geom.u_curr_safe);
        }
        let inv = geom.distance.rcp_pos_nr2();
        let (odd, even) = finish_rail_crossing_odd_even_stereo8(&geom, odd_int, even_int, inv);
        return (odd, even, geom.u_curr_safe);
    }
    let wf_odd_int =
        wavefolder::eval_c2_wf_odd_interior_stereo8(geom.m, geom.m2, geom.d2, geom.max_abs);
    let (adaa_odd, adaa_even, wf_odd) = if interior {
        (odd_int, even_int, wf_odd_int)
    } else {
        let inv = geom.distance.rcp_pos_nr2();
        let (odd, even) = finish_rail_crossing_odd_even_stereo8(&geom, odd_int, even_int, inv);
        let wf_odd = wavefolder::finish_c2_wf_odd_rail_crossing_stereo8(&geom, wf_odd_int, inv);
        (odd, even, wf_odd)
    };
    let wf_mix_s = S::broadcast_half(wf_clamped);
    let one_minus_wf = S::splat(1.0).sub(wf_mix_s);
    let mixed_odd = adaa_odd
        .mul(one_minus_wf)
        .fma(wf_mix_s, wf_odd)
        .add(S::zero());
    (mixed_odd, adaa_even, geom.u_curr_safe)
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

    fn assert_zero_ulp_antisymmetric(pos: f32, neg: f32) {
        if pos == 0.0 {
            assert_eq!(pos.to_bits(), 0);
            assert_eq!(neg.to_bits(), 0);
        } else {
            assert_eq!(neg.to_bits(), (-pos).to_bits());
        }
    }

    /// verify[dsp.op-n.polyadaa.bspline-ac]
    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.linear-residual]
    /// verify[dsp.op-n.polyadaa.midpoint]
    #[test]
    fn test_poly_adaa1_exact_zero_signed_zero_nonfinite_and_known_values() {
        for z in [0.0_f32, -0.0_f32] {
            let (odd, even, curr) =
                eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(z), GpuVec4::splat(z));
            assert_eq!(odd.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(curr.to_array().map(f32::to_bits), [0; 4]);
        }

        // Mixed signed-zero and asymmetric non-finite/extreme endpoint safety.
        let (o_mz, e_mz, _) =
            eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(-0.0), GpuVec4::splat(0.5));
        let (o_pz, e_pz, _) =
            eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(0.0), GpuVec4::splat(-0.5));
        assert_zero_ulp_antisymmetric(o_mz.to_array()[0], o_pz.to_array()[0]);
        assert_eq!(e_mz.to_array()[0].to_bits(), e_pz.to_array()[0].to_bits());

        let bad_p = GpuVec4::from_array([f32::NAN, f32::INFINITY, -1.0e30, 1.0e30]);
        let bad_c = GpuVec4::from_array([f32::NEG_INFINITY, f32::NAN, 1.0e30, 2.0e30]);
        let (bo, be, bc) = eval_poly_adaa1_odd_even_vec4(bad_p, bad_c);
        assert_eq!(bo.to_array()[0].to_bits(), 0);
        assert_eq!(be.to_array()[0].to_bits(), 0);
        assert_eq!(bc.to_array()[0].to_bits(), 0);
        assert_eq!(bo.to_array()[1].to_bits(), 0);
        assert_eq!(bo.to_array()[2].to_bits(), 0);
        assert_eq!(bo.to_array()[3], 1.0);
        assert_eq!(be.to_array()[3].to_bits(), 0);
        assert!(bo.to_array().iter().all(|v| v.is_finite()));
        assert!(be.to_array().iter().all(|v| v.is_finite()));

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
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// verify[dsp.op-n.polyadaa.drive-bank]
        /// verify[dsp.op-n.polyadaa.midpoint]
        #[test]
        fn prop_poly_adaa1_interior_matches_oracle_and_zero_ulp_symmetry(
            a0 in -1.0_f32..1.0_f32,
            a1 in -1.0_f32..1.0_f32,
            a2 in prop_oneof![Just(-0.0_f32), Just(0.0_f32), -1.0_f32..1.0_f32],
            a3 in -1.0_f32..1.0_f32,
            b0 in -1.0_f32..1.0_f32,
            b1 in -1.0_f32..1.0_f32,
            b2 in prop_oneof![Just(-0.0_f32), Just(0.0_f32), -1.0_f32..1.0_f32],
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

        /// verify[dsp.op-n.polyadaa.clamp]
        /// verify[dsp.op-n.polyadaa.rail-crossing]
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
        fn prop_fused_poly_adaa1_with_wavefolder_zero_ulp_equivalence(
            prev_l in prop::array::uniform4(prop_oneof![Just(-0.0_f32), Just(0.0_f32), -8.0_f32..8.0_f32]),
            curr_l in prop::array::uniform4(prop_oneof![Just(-0.0_f32), Just(0.0_f32), -8.0_f32..8.0_f32]),
            prev_r in prop::array::uniform4(prop_oneof![Just(-1.0_f32), Just(1.0_f32), -0.95_f32..0.95_f32]),
            curr_r in prop::array::uniform4(prop_oneof![Just(-1.0_f32), Just(1.0_f32), -0.95_f32..0.95_f32]),
            wf_mix in prop::array::uniform4(prop_oneof![Just(0.0_f32), Just(1.0_f32), 0.01_f32..0.99_f32]),
        ) {
            let pl = GpuVec4::from_array(prev_l);
            let cl = GpuVec4::from_array(curr_l);
            let pr = GpuVec4::from_array(prev_r);
            let cr = GpuVec4::from_array(curr_r);
            let wf = GpuVec4::from_array(wf_mix);

            let (mut ref_ol, ref_el, ref_next_l) = eval_poly_adaa1_odd_even_vec4(pl, cl);
            let (mut ref_or, ref_er, ref_next_r) = eval_poly_adaa1_odd_even_vec4(pr, cr);
            if !wf.all_le(GpuVec4::zero()) {
                let (wfl, _) = eval_c2_wavefolder_adaa_vec4(pl, cl);
                let (wfr, _) = eval_c2_wavefolder_adaa_vec4(pr, cr);
                let omw = GpuVec4::splat(1.0).sub(wf);
                ref_ol = ref_ol.mul(omw).fma(wf, wfl).add(GpuVec4::zero());
                ref_or = ref_or.mul(omw).fma(wf, wfr).add(GpuVec4::zero());
            }

            let (f_ol, f_el, f_next_l) = eval_poly_adaa1_with_wavefolder_vec4(pl, cl, wf);
            let (f_or, f_er, f_next_r) = eval_poly_adaa1_with_wavefolder_vec4(pr, cr, wf);
            prop_assert_eq!(f_ol.to_array().map(f32::to_bits), ref_ol.to_array().map(f32::to_bits));
            prop_assert_eq!(f_el.to_array().map(f32::to_bits), ref_el.to_array().map(f32::to_bits));
            prop_assert_eq!(f_next_l.to_array().map(f32::to_bits), ref_next_l.to_array().map(f32::to_bits));
            prop_assert_eq!(f_or.to_array().map(f32::to_bits), ref_or.to_array().map(f32::to_bits));
            prop_assert_eq!(f_er.to_array().map(f32::to_bits), ref_er.to_array().map(f32::to_bits));
            prop_assert_eq!(f_next_r.to_array().map(f32::to_bits), ref_next_r.to_array().map(f32::to_bits));

            let st_p = InterleavedStereo8::from_lr(pl, pr);
            let st_c = InterleavedStereo8::from_lr(cl, cr);
            let (f_s_odd, f_s_even, f_s_next) =
                eval_poly_adaa1_with_wavefolder_stereo8(st_p, st_c, wf);
            let (fol, for_) = f_s_odd.into_lr();
            let (fel, fer) = f_s_even.into_lr();
            let (fnl, fnr) = f_s_next.into_lr();
            prop_assert_eq!(fol.to_array().map(f32::to_bits), ref_ol.to_array().map(f32::to_bits));
            prop_assert_eq!(for_.to_array().map(f32::to_bits), ref_or.to_array().map(f32::to_bits));
            prop_assert_eq!(fel.to_array().map(f32::to_bits), ref_el.to_array().map(f32::to_bits));
            prop_assert_eq!(fer.to_array().map(f32::to_bits), ref_er.to_array().map(f32::to_bits));
            prop_assert_eq!(fnl.to_array().map(f32::to_bits), ref_next_l.to_array().map(f32::to_bits));
            prop_assert_eq!(fnr.to_array().map(f32::to_bits), ref_next_r.to_array().map(f32::to_bits));
        }
    }

    #[test]
    fn test_fused_poly_adaa1_with_wavefolder_edge_cases_zero_ulp() {
        type WfEdgeCase = ([f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4]);
        let edge_cases: [WfEdgeCase; 5] = [
            // Interior with -0.0 and wf_mix == 0
            (
                [-0.0, 0.25, -0.75, 1.0],
                [0.0, -0.5, 0.75, -1.0],
                [-0.2, 0.0, 0.5, -0.5],
                [0.3, -0.0, -0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ),
            // Interior with mixed wf_mix including 0 and (0, 1)
            (
                [-0.0, 0.25, -0.75, 1.0],
                [0.0, -0.5, 0.75, -1.0],
                [-0.2, 0.0, 0.5, -0.5],
                [0.3, -0.0, -0.5, 0.5],
                [0.0, 0.25, 0.75, 1.0],
            ),
            // Mixed L interior / R rail-crossing
            (
                [-0.5, 0.2, -0.8, 0.1],
                [0.4, -0.3, 0.9, -0.1],
                [-3.0, 0.5, -2.5, 1.5],
                [2.5, -2.0, 3.0, 4.0],
                [0.0, 0.4, 0.8, 1.0],
            ),
            // Non-finite NaN / ±Inf scrubbing
            (
                [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0],
                [0.5, -0.5, f32::NAN, f32::INFINITY],
                [f32::NEG_INFINITY, 2.0, -2.0, f32::NAN],
                [1.0, f32::NAN, -1.0, -0.0],
                [0.0, 0.3, 0.7, 1.0],
            ),
            // Rail-crossing with wf_mix == 0
            (
                [-4.0, -0.5, 2.0, -1.5],
                [4.0, 3.0, 5.0, -3.5],
                [-2.0, 1.5, -0.25, 0.0],
                [2.0, -1.5, 2.5, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ),
        ];

        for (case_idx, &(pl_a, cl_a, pr_a, cr_a, wf_a)) in edge_cases.iter().enumerate() {
            let pl = GpuVec4::from_array(pl_a);
            let cl = GpuVec4::from_array(cl_a);
            let pr = GpuVec4::from_array(pr_a);
            let cr = GpuVec4::from_array(cr_a);
            let wf = GpuVec4::from_array(wf_a);
            let st_p = InterleavedStereo8::from_lr(pl, pr);
            let st_c = InterleavedStereo8::from_lr(cl, cr);

            let (mut ref_o, ref_e) = eval_poly_adaa1_stereo8(st_p, st_c);
            if !wf.all_le(GpuVec4::zero()) {
                let (wfo, _) = eval_c2_wavefolder_adaa_stereo8(st_p, st_c);
                let wf_s = InterleavedStereo8::broadcast_half(wf);
                let omw = InterleavedStereo8::splat(1.0).sub(wf_s);
                ref_o = ref_o
                    .mul(omw)
                    .fma(wf_s, wfo)
                    .add(InterleavedStereo8::zero());
            }
            let ref_next = scrub_endpoint_stereo8(st_c);

            let (f_o, f_e, f_next) = eval_poly_adaa1_with_wavefolder_stereo8(st_p, st_c, wf);
            let (fol, for_) = f_o.into_lr();
            let (rol, ror) = ref_o.into_lr();
            let (fel, fer) = f_e.into_lr();
            let (rel, rer) = ref_e.into_lr();
            let (fnl, fnr) = f_next.into_lr();
            let (rnl, rnr) = ref_next.into_lr();
            assert_eq!(
                fol.to_array().map(f32::to_bits),
                rol.to_array().map(f32::to_bits),
                "case {case_idx}"
            );
            assert_eq!(
                for_.to_array().map(f32::to_bits),
                ror.to_array().map(f32::to_bits),
                "case {case_idx}"
            );
            assert_eq!(
                fel.to_array().map(f32::to_bits),
                rel.to_array().map(f32::to_bits),
                "case {case_idx}"
            );
            assert_eq!(
                fer.to_array().map(f32::to_bits),
                rer.to_array().map(f32::to_bits),
                "case {case_idx}"
            );
            assert_eq!(
                fnl.to_array().map(f32::to_bits),
                rnl.to_array().map(f32::to_bits),
                "case {case_idx}"
            );
            assert_eq!(
                fnr.to_array().map(f32::to_bits),
                rnr.to_array().map(f32::to_bits),
                "case {case_idx}"
            );
        }
    }
}
