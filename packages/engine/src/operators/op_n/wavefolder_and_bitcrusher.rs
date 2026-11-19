//! Op-N Cubed-Envelope $C^2$ Wavefolder, 2D Quadrature Hysteresis & Fixed-Hertz ZOH Bitcrusher.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.wavefolder.cubic-c2]`,
//! `ref[dsp.op-n.hysteresis.quadrature]`, `ref[dsp.op-n.bitcrusher.quantizer]`,
//! `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`).

pub mod bitcrusher;

pub use bitcrusher::{
    FixedHzBitcrusher, ZohBitcrusher, hertz_phase_parameters, hertz_phase_step,
    quantize_bitcrusher_nearest_ties_pos_inf, rescale_bitcrusher_phase_u128,
};

/// Instantaneous $C^2$ compactly supported wavefolder delegated to
/// [`dnam_kernels::eval_c2_wavefolder_adaa_vec4`].
/// Evaluates `w_odd(u) = u_bar * (1 - u_bar^2)^3` and `w_even(u) = u_bar^2 * (1 - u_bar^2)^3`
/// on `u_bar = clamp(u, -1, 1)` (`ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline(always)]
pub fn c2_wavefolder(u: f32) -> (f32, f32) {
    use dnam_kernels::{KernelVec4, simd::GpuVec4};
    let uv = GpuVec4::splat(u);
    let (wo, we) = dnam_kernels::eval_c2_wavefolder_adaa_vec4(uv, uv);
    (wo.to_array()[0], we.to_array()[0])
}

/// 2D Quadrature Phase-Space Hysteresis state (`ref[dsp.op-n.hysteresis.quadrature]`).
#[derive(Clone, Copy, Debug)]
pub struct QuadratureHysteresis2D {
    pub prev_s_re: f32,
    pub kappa_quad: f32,
}

impl Default for QuadratureHysteresis2D {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadratureHysteresis2D {
    pub const fn new() -> Self {
        Self {
            prev_s_re: 0.0,
            kappa_quad: 0.15,
        }
    }

    pub fn reset(&mut self) {
        self.prev_s_re = 0.0;
    }

    /// Steps the 2D hysteresis model given quadrature phase-space coordinates `(s_re, s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, s_re: f32, s_im: f32) -> f32 {
        self.step_with_rate(s_re, s_im, 1.0)
    }

    /// Steps the 2D hysteresis model with continuous-time ZOH differential rate scaling `diff_scale = 1 / rho_sr`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_with_rate(&mut self, s_re: f32, s_im: f32, diff_scale: f32) -> f32 {
        let delta_re = (s_re - self.prev_s_re) * diff_scale;
        let hyst = s_re * delta_re + self.kappa_quad * s_re * s_im;
        self.prev_s_re = s_re;
        hyst
    }
}

/// Evaluates the odd degree-7 interval-average `PolyADAA` basis polynomial $A_{\text{odd},3}$ in `f32`
/// for the cubed-envelope $C^2$ wavefolder on unclamped consecutive endpoints `(u_prev, u_curr)`
/// via [`dnam_kernels::eval_c2_wavefolder_adaa_vec4`] (`ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline(always)]
pub fn eval_adaa_cubic_c2_wavefolder_odd_f32(u_prev: f32, u_curr: f32) -> f32 {
    use dnam_kernels::{KernelVec4, simd::GpuVec4};
    let uv0 = GpuVec4::splat(u_prev);
    let uv1 = GpuVec4::splat(u_curr);
    dnam_kernels::eval_c2_wavefolder_adaa_vec4(uv0, uv1)
        .0
        .to_array()[0]
}

/// Evaluates the division-free degree-7/8 interval-average `PolyADAA` basis polynomials
/// $(A_{\text{odd},3}, A_{\text{even},3})$ for the cubed-envelope $C^2$ wavefolder across
/// interior and rail-crossing segments `[-1, 1]` using active-length normalization (`0 FDIV`,
/// `ref[dsp.op-n.wavefolder.cubic-c2]`).
#[inline]
pub fn eval_adaa_cubic_c2_wavefolder_f64(u_prev: f64, u_curr: f64) -> (f64, f64) {
    const ONE_THIRD: f64 = 1.0 / 3.0;
    const ONE_FIFTH: f64 = 1.0 / 5.0;
    const ONE_SEVENTH: f64 = 1.0 / 7.0;
    const ONE_NINTH: f64 = 1.0 / 9.0;
    const TWENTY_EIGHT_THIRDS: f64 = 28.0 / 3.0;
    const MAX_ENDPOINT: f64 = 1.0e18;
    const TINY_CROSSING_EPS: f64 = 1.0e-12;

    let p = if u_prev.is_finite() {
        u_prev.clamp(-MAX_ENDPOINT, MAX_ENDPOINT)
    } else {
        0.0
    };
    let c = if u_curr.is_finite() {
        u_curr.clamp(-MAX_ENDPOINT, MAX_ENDPOINT)
    } else {
        0.0
    };
    let lower = p.min(c);
    let upper = p.max(c);
    if upper <= -1.0 || lower >= 1.0 {
        return (0.0, 0.0);
    }

    let u0 = lower.max(-1.0);
    let u1 = upper.min(1.0);
    let m = 0.5 * (u1 + u0);
    let d = 0.5 * (u1 - u0);
    let m2 = m * m;
    let d2 = d * d;
    let m3 = m2 * m;
    let d3 = d2 * d;
    let m4 = m2 * m2;
    let d4 = d2 * d2;
    let odd_int = m
        * (1.0 - 3.0 * (m2 + d2) + 3.0 * m4 + 10.0 * m2 * d2 + 3.0 * d4
            - m3 * m3
            - 7.0 * m4 * d2
            - 7.0 * m2 * d4
            - d3 * d3);
    let even_int = m2 + d2 * ONE_THIRD - 3.0 * (m4 + 2.0 * m2 * d2 + ONE_FIFTH * d4)
        + 3.0 * (m3 * m3 + 5.0 * m4 * d2 + 3.0 * m2 * d4 + d3 * d3 * ONE_SEVENTH)
        - (m4 * m4
            + TWENTY_EIGHT_THIRDS * m3 * m3 * d2
            + 14.0 * m4 * d4
            + 4.0 * m2 * d3 * d3
            + d4 * d4 * ONE_NINTH);

    let distance = upper - lower;
    if (lower >= -1.0 && upper <= 1.0) || distance <= TINY_CROSSING_EPS {
        return (odd_int, even_int);
    }
    let inside = (u1 - u0).max(0.0);
    let scale = inside * crate::math::newton6_reciprocal_f64(distance);
    (scale * odd_int, scale * even_int)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::operators::eval_saturated_polyadaa_rail_crossing_f64;
    use proptest::prelude::*;

    /// verify[dsp.op-n.wavefolder.cubic-c2]
    /// verify[dsp.op-n.hysteresis.quadrature]
    #[test]
    fn test_wavefolder_hysteresis_and_rail_crossings() {
        let _guard = FtzDazGuard::new();

        let (w_odd_pos, w_even_pos) = c2_wavefolder(1.0);
        assert!(w_odd_pos.abs() < 1e-7 && w_even_pos.abs() < 1e-7);
        let (w_odd_neg, w_even_neg) = c2_wavefolder(-1.0);
        assert!(w_odd_neg.abs() < 1e-7 && w_even_neg.abs() < 1e-7);

        let eps = 1e-4_f32;
        let (w_near, _) = c2_wavefolder(1.0 - eps);
        let deriv = (w_near - w_odd_pos) / (-eps);
        assert!(deriv.abs() < 1e-3);

        let (wo, we) = c2_wavefolder(0.5);
        assert!((wo - 0.5 * 0.421_875).abs() < 1e-6);
        assert!((we - 0.25 * 0.421_875).abs() < 1e-6);

        let (ao_eq, ae_eq) = eval_adaa_cubic_c2_wavefolder_f64(0.5, 0.5);
        assert!((ao_eq - 0.5 * 0.421_875).abs() < 1e-12);
        assert!((ae_eq - 0.25 * 0.421_875).abs() < 1e-12);

        // Rail-crossing scaling: (0.0, 3.0) has inside/distance = 1/3, outside [-1,1] yields (0,0)
        assert_eq!(eval_adaa_cubic_c2_wavefolder_f64(2.0, 3.0), (0.0, 0.0));
        assert_eq!(eval_adaa_cubic_c2_wavefolder_odd_f32(2.0, 3.0), 0.0);
        let (ao_int, ae_int) = eval_adaa_cubic_c2_wavefolder_f64(0.0, 1.0);
        let (ao_cross, ae_cross) = eval_adaa_cubic_c2_wavefolder_f64(0.0, 3.0);
        assert!((ao_cross - ao_int / 3.0).abs() < 1e-12);
        assert!((ae_cross - ae_int / 3.0).abs() < 1e-12);
        for &(p, c) in &[(0.0_f32, 3.0_f32), (-2.5, 1.5), (-3.0, -1.5), (-1.2, 0.4)] {
            let o32 = eval_adaa_cubic_c2_wavefolder_odd_f32(p, c);
            let (o64, _) = eval_adaa_cubic_c2_wavefolder_f64(f64::from(p), f64::from(c));
            assert!((f64::from(o32) - o64).abs() < 2.0e-5);
        }

        let mut hyst = QuadratureHysteresis2D::default();
        assert_eq!(hyst.kappa_quad, 0.15);
        hyst.reset();
        let h1 = hyst.step(0.5, 0.2);
        assert!((h1 - (0.5 * 0.5 + 0.15 * 0.5 * 0.2)).abs() < 1e-6);
        let h2 = hyst.step(0.8, -0.1);
        assert!((h2 - (0.8 * 0.3 + 0.15 * 0.8 * (-0.1))).abs() < 1e-6);
        hyst.reset();
        let h3 = hyst.step_with_rate(0.5, 0.2, 2.0);
        assert!((h3 - (0.5 * 0.5 * 2.0 + 0.15 * 0.5 * 0.2)).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_c2_wavefolder_and_rail_crossing(
            u in -2.0_f32..2.0_f32,
            raw_prev in -10.0_f64..10.0_f64,
            raw_curr in -10.0_f64..10.0_f64,
        ) {
            let (w_odd, w_even) = c2_wavefolder(u);
            let (w_odd_neg, w_even_neg) = c2_wavefolder(-u);
            prop_assert!(w_odd.abs() <= 1.0001 && w_even.abs() <= 1.0001);
            prop_assert!((w_odd + w_odd_neg).abs() < 1e-6);
            prop_assert!((w_even - w_even_neg).abs() < 1e-6);
            if u.abs() >= 1.0 {
                prop_assert_eq!(w_odd, 0.0);
                prop_assert_eq!(w_even, 0.0);
            }

            let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(raw_prev, raw_curr);
            prop_assert!(odd.is_finite() && odd.abs() <= 1.0 + 1e-12);
            prop_assert!(even.is_finite() && (0.0..=1.0).contains(&even));

            let (wf_o, wf_e) = eval_adaa_cubic_c2_wavefolder_f64(raw_prev, raw_curr);
            prop_assert!(wf_o.is_finite() && wf_o.abs() <= 1.0);
            prop_assert!(wf_e.is_finite() && (0.0..=1.0).contains(&wf_e));
        }
    }
}
