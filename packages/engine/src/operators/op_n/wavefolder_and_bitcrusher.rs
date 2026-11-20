//! Op-N Cubed-Envelope $C^2$ Wavefolder, 2D Quadrature Hysteresis & Fixed-Hertz ZOH Bitcrusher.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_paralru_and_polyadaa.md` (`ref[dsp.op-n.wavefolder.cubic-c2]`,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use proptest::prelude::*;

    fn analytical_c2_wavefolder_odd_oracle_f64(u_prev: f64, u_curr: f64) -> f64 {
        let lower = u_prev.min(u_curr);
        let upper = u_prev.max(u_curr);
        if upper <= -1.0 || lower >= 1.0 {
            return 0.0;
        }
        let u0 = lower.max(-1.0);
        let u1 = upper.min(1.0);
        let dist = upper - lower;
        if dist <= 1.0e-12 {
            let m = 0.5 * (u0 + u1);
            let t = 1.0 - m * m;
            return m * t * t * t;
        }
        let antideriv = |u: f64| {
            let u2 = u * u;
            let u4 = u2 * u2;
            0.5 * u2 - 0.75 * u4 + 0.5 * u4 * u2 - 0.125 * u4 * u4
        };
        (antideriv(u1) - antideriv(u0)) / dist
    }

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

        assert!(
            (f64::from(eval_adaa_cubic_c2_wavefolder_odd_f32(0.5, 0.5)) - 0.5 * 0.421_875).abs()
                < 1e-6
        );

        // Rail-crossing scaling: (0.0, 3.0) has inside/distance = 1/3, outside [-1,1] yields 0.0
        assert_eq!(eval_adaa_cubic_c2_wavefolder_odd_f32(2.0, 3.0), 0.0);
        let ao_int = eval_adaa_cubic_c2_wavefolder_odd_f32(0.0, 1.0);
        let ao_cross = eval_adaa_cubic_c2_wavefolder_odd_f32(0.0, 3.0);
        assert!((ao_cross - ao_int / 3.0).abs() < 1e-6);
        for &(p, c) in &[(0.0_f32, 3.0_f32), (-2.5, 1.5), (-3.0, -1.5), (-1.2, 0.4)] {
            let o32 = eval_adaa_cubic_c2_wavefolder_odd_f32(p, c);
            let o64 = analytical_c2_wavefolder_odd_oracle_f64(f64::from(p), f64::from(c));
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

            let prev32 = raw_prev as f32;
            let curr32 = raw_curr as f32;
            let wf_o = eval_adaa_cubic_c2_wavefolder_odd_f32(prev32, curr32);
            prop_assert!(wf_o.is_finite() && wf_o.abs() <= 1.0 + 1e-6);
            let wf_oracle =
                analytical_c2_wavefolder_odd_oracle_f64(f64::from(prev32), f64::from(curr32));
            prop_assert!(wf_oracle.is_finite() && wf_oracle.abs() <= 1.0 + 1e-12);
            prop_assert!((f64::from(wf_o) - wf_oracle).abs() <= 2.5e-5);
        }
    }
}
