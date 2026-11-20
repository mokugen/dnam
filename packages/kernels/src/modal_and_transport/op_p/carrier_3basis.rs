//! Top-of-Frame 3-basis quadrature carrier projection (`[w0 * I * cos, w1 * Q * sin, w2 * sin]`),
//! relative stereo rotor offset rotation (`cos_R, sin_R`), and signed `OS2X` `Q0.32` increment
//! halving (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.rotor.q0-192-lfo]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`.

use crate::simd::KernelVec4;

/// Decoded 3-basis carrier projection weights `[w0, w1, w2]` for `[hi * cos, hq * sin, sin]`
/// across 4 crossbar phasor lanes (`ref[dsp.op-m.hilbert.lattice]`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarrierBasisWeightsVec4<V: KernelVec4> {
    pub w0: V,
    pub w1: V,
    pub w2: V,
}

/// True-stereo `H1` analytic inputs passed into the Top-of-Frame 3-basis carrier projection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StereoAnalyticQuadrature {
    pub i_l: f32,
    pub q_l: f32,
    pub i_r: f32,
    pub q_r: f32,
}

/// Stereo 4-lane quadrature rotor pair `(cos_L, sin_L, cos_R, sin_R)` formed via
/// [`rotate_carrier_stereo_phase_vec4`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StereoCarrierRotorVec4<V: KernelVec4> {
    pub cos_l: V,
    pub sin_l: V,
    pub cos_r: V,
    pub sin_r: V,
}

impl<V: KernelVec4> StereoCarrierRotorVec4<V> {
    /// Builds Left and Right rotor lanes by rotating Left `(cos_l, sin_l)` by relative
    /// stereo phase rotor `(cos_st, sin_st)`.
    #[inline(always)]
    #[must_use]
    pub fn from_left_and_offset(cos_l: V, sin_l: V, cos_st: V, sin_st: V) -> Self {
        let (cos_r, sin_r) = rotate_carrier_stereo_phase_vec4(cos_l, sin_l, cos_st, sin_st);
        Self {
            cos_l,
            sin_l,
            cos_r,
            sin_r,
        }
    }
}

/// Arithmetic right-shift halving (`inc >> 1`) of 4 signed `Q0.32` carrier phase increments when
/// running inside a 2× oversampled (`OS2X`) inner core loop (`ref[dsp.op-m.rotor.q0-192-lfo]`).
#[inline(always)]
#[must_use]
pub const fn halve_signed_q32_increments(inc: [i32; 4]) -> [i32; 4] {
    [inc[0] >> 1, inc[1] >> 1, inc[2] >> 1, inc[3] >> 1]
}

/// Rotates Left 4-lane carrier quadrature `(cos_l, sin_l)` by relative Right-minus-Left rotor
/// `(cos_st, sin_st)` produced via `algebraic_unit_rotor_f32`:
/// `cos_R = cos_L * c_st - sin_L * s_st`, `sin_R = sin_L * c_st + cos_L * s_st`.
#[inline(always)]
#[must_use]
pub fn rotate_carrier_stereo_phase_vec4<V: KernelVec4>(
    cos_l: V,
    sin_l: V,
    cos_st: V,
    sin_st: V,
) -> (V, V) {
    let cos_r = cos_l.mul(cos_st).fms(sin_l, sin_st);
    let sin_r = sin_l.mul(cos_st).fma(cos_l, sin_st);
    (cos_r, sin_r)
}

/// Shared single-channel 3-basis carrier combination:
/// `wet_scale * (w0 * (I * cos) + w1 * (Q * sin) + w2 * sin)`, flushed on subnormals.
#[inline(always)]
fn eval_carrier_3basis_channel_vec4<V: KernelVec4>(
    hi: f32,
    hq: f32,
    cos_v: V,
    sin_v: V,
    weights: &CarrierBasisWeightsVec4<V>,
    scale: V,
) -> V {
    let mod_i = V::splat(hi).mul(cos_v);
    let mod_q = V::splat(hq).mul(sin_v);
    let sum = weights
        .w0
        .mul(mod_i)
        .fma(weights.w1, mod_q)
        .fma(weights.w2, sin_v);
    scale.mul(sum).flush_subnormal()
}

/// Evaluates the stereo 4-lane 3-basis carrier projections `(car_L, car_R)` from true-stereo `H1`
/// analytic state, stereo quadrature rotor, decoded `carrier_basis_w` weights, and `wet_scale`
/// (`ref[dsp.op-m.hilbert.lattice]`).
#[inline(always)]
#[must_use]
pub fn eval_carrier_3basis_stereo_vec4<V: KernelVec4>(
    hilbert: &StereoAnalyticQuadrature,
    rotor: &StereoCarrierRotorVec4<V>,
    weights: &CarrierBasisWeightsVec4<V>,
    wet_scale: f32,
) -> (V, V) {
    let scale = V::splat(wet_scale);
    let car_l = eval_carrier_3basis_channel_vec4(
        hilbert.i_l,
        hilbert.q_l,
        rotor.cos_l,
        rotor.sin_l,
        weights,
        scale,
    );
    let car_r = eval_carrier_3basis_channel_vec4(
        hilbert.i_r,
        hilbert.q_r,
        rotor.cos_r,
        rotor.sin_r,
        weights,
        scale,
    );
    (car_l, car_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modal_and_transport::kautz::algebraic_unit_rotor_f32;
    use crate::simd::GpuVec4;
    use proptest::prelude::*;

    /// verify[dsp.op-m.hilbert.lattice]
    #[test]
    fn test_carrier_3basis_ssb_zero_weights_and_signed_halving() {
        assert_eq!(
            halve_signed_q32_increments([-4096, -1, 0, 8192]),
            [-2048, -1, 0, 4096]
        );

        let theta = core::f32::consts::FRAC_PI_4;
        let phi = 0.6_f32;
        let (c_st, s_st) = algebraic_unit_rotor_f32(theta);
        let rotor = StereoCarrierRotorVec4::from_left_and_offset(
            GpuVec4::splat(phi.cos()),
            GpuVec4::splat(phi.sin()),
            GpuVec4::splat(c_st),
            GpuVec4::splat(s_st),
        );
        assert!((rotor.cos_r.to_array()[0] - (phi + theta).cos()).abs() < 1.0e-6);
        assert!((rotor.sin_r.to_array()[0] - (phi + theta).sin()).abs() < 1.0e-6);

        let zero_w = CarrierBasisWeightsVec4::default();
        let hilb_zero = StereoAnalyticQuadrature {
            i_l: 0.5,
            q_l: -0.5,
            i_r: 0.25,
            q_r: -0.25,
        };
        let (zl, zr) = eval_carrier_3basis_stereo_vec4(&hilb_zero, &rotor, &zero_w, 1.5);
        assert_eq!(zl.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(zr.to_array().map(f32::to_bits), [0; 4]);

        let a = 0.35_f32;
        let hilb_ssb = StereoAnalyticQuadrature {
            i_l: a.cos(),
            q_l: a.sin(),
            i_r: a.cos(),
            q_r: a.sin(),
        };
        let w_ssb = CarrierBasisWeightsVec4 {
            w0: GpuVec4::splat(1.0),
            w1: GpuVec4::splat(-1.0),
            w2: GpuVec4::splat(0.0),
        };
        let (cl, sl) = eval_carrier_3basis_stereo_vec4(&hilb_ssb, &rotor, &w_ssb, 1.0);
        assert!((cl.to_array()[0] - (a + phi).cos()).abs() < 1.0e-6);
        assert!((sl.to_array()[0] - (a + phi + theta).cos()).abs() < 1.0e-6);
    }

    proptest! {
        #[test]
        fn prop_stereo_rotor_preserves_unit_circle(
            phi in -core::f32::consts::PI..core::f32::consts::PI,
            delta in -core::f32::consts::PI..core::f32::consts::PI,
        ) {
            let (c_st, s_st) = algebraic_unit_rotor_f32(delta);
            let (cr, sr) = rotate_carrier_stereo_phase_vec4(
                GpuVec4::splat(phi.cos()),
                GpuVec4::splat(phi.sin()),
                GpuVec4::splat(c_st),
                GpuVec4::splat(s_st),
            );
            let r2 = cr.to_array()[0] * cr.to_array()[0] + sr.to_array()[0] * sr.to_array()[0];
            prop_assert!((r2 - 1.0).abs() < 1.0e-5, "r2 = {r2}");
        }
    }
}
