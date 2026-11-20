//! Symplectic Modal Recurrence, Contractive Rotor, and Orthonormal Kautz Modal Bank (`Stage III` & Family 6,
//! `ref[dsp.op-m.recurrence.symplectic]`, `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.kautz.sweep-contraction]`).

pub mod simd;

use alloc::vec::Vec;

use crate::math::{flush_recursive_state_f32, flush_recursive_state_vec4, sqrt_f64};
use crate::schema::cold_pow_f64;

/// Quadrant-reduced degree-13/14 Horner contractive rotor delegated to [`dnam_kernels::algebraic_unit_rotor_f32`]
/// (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.ulp-guard]`, `ref[dsp.op-m.rotor.zero-hold]`).
///
/// Given frequency angle `theta` in `[0, 2*pi)`, evaluates unit-circle coordinates `(cos(theta), sin(theta))`
/// with unconditional guarantee `cos^2 + sin^2 <= 1.0` and `<= 2 ULP` error.
#[inline(always)]
pub fn algebraic_unit_rotor_op_m(theta: f32) -> (f32, f32) {
    dnam_kernels::algebraic_unit_rotor_f32(theta)
}

/// Fast reciprocal square root using magic constant seed + 2 Newton-Raphson iterations (0 FDIV, 0 FSQRT).
#[inline(always)]
pub fn fast_rsqrt_nr2(x: f32) -> f32 {
    <crate::math::DefaultMathBackend as crate::math::MathBackend>::rsqrt_nr2_f32(x.max(1.0e-14))
}

/// 4-lane SIMD Symplectic Complex Recurrence State (`ref[dsp.op-m.recurrence.symplectic]`).
///
/// Evaluates:
/// `z_re[n] = alpha_re * z_re[n-1] - alpha_im * z_im[n-1] + b_re * x[n]`
/// `z_im[n] = alpha_im * z_re[n-1] + alpha_re * z_im[n-1] + b_im * x[n]`
#[derive(Clone, Copy, Debug)]
pub struct SymplecticModal4Lane {
    pub z_re: [f32; 4],
    pub z_im: [f32; 4],
    pub alpha_re: [f32; 4],
    pub alpha_im: [f32; 4],
    pub b_re: [f32; 4],
    pub b_im: [f32; 4],
}

impl Default for SymplecticModal4Lane {
    fn default() -> Self {
        Self::new()
    }
}

impl SymplecticModal4Lane {
    pub const fn new() -> Self {
        Self {
            z_re: [0.0; 4],
            z_im: [0.0; 4],
            alpha_re: [0.0; 4],
            alpha_im: [0.0; 4],
            b_re: [1.0; 4],
            b_im: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.z_re.fill(0.0);
        self.z_im.fill(0.0);
    }

    /// Sets complex pole for lane `c` given radius `r <= 0.9995` and angle `theta`.
    pub fn set_pole(&mut self, lane: usize, r: f32, theta: f32) {
        if lane < 4 {
            let r_clamped = if r.is_finite() {
                r.clamp(0.0, 0.9995)
            } else {
                0.0
            };
            let theta_clean = if theta.is_finite() { theta } else { 0.0 };
            let (c_rot, s_rot) = algebraic_unit_rotor_op_m(theta_clean);
            self.alpha_re[lane] = r_clamped * c_rot;
            self.alpha_im[lane] = r_clamped * s_rot;

            // H2-norm scaling: gamma = sqrt(1 - r^2)
            let one_minus_r2 = (1.0 - r_clamped * r_clamped).max(0.0);
            let gamma = fast_rsqrt_nr2(one_minus_r2);
            self.b_re[lane] = if gamma > 0.0 {
                one_minus_r2 * gamma
            } else {
                1.0
            };
            self.b_im[lane] = 0.0;
        }
    }

    /// Steps lane `c` by 1 sample with input `x`, returning complex state `(z_re, z_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_lane(&mut self, lane: usize, x: f32) -> (f32, f32) {
        let u = flush_recursive_state_f32(x);
        let prev_re = flush_recursive_state_f32(self.z_re[lane]);
        let prev_im = flush_recursive_state_f32(self.z_im[lane]);
        let a_re = self.alpha_re[lane];
        let a_im = self.alpha_im[lane];

        let next_re =
            flush_recursive_state_f32(a_re * prev_re - a_im * prev_im + self.b_re[lane] * u);
        let next_im =
            flush_recursive_state_f32(a_im * prev_re + a_re * prev_im + self.b_im[lane] * u);

        self.z_re[lane] = next_re;
        self.z_im[lane] = next_im;
        (next_re, next_im)
    }

    /// Steps all 4 lanes of a stereo pair (`self` = Left, `right` = Right) in 4-lane SIMD,
    /// preserving exact IEEE-754 non-FMA operation order (`0 ULP` difference vs [`Self::step_lane`]).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_stereo_simd<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
    ) {
        use crate::math::SimdVec4;
        let xl = flush_recursive_state_vec4(B::Vec4::splat(x_l));
        let xr = flush_recursive_state_vec4(B::Vec4::splat(x_r));
        let a_re = B::Vec4::load(&self.alpha_re);
        let a_im = B::Vec4::load(&self.alpha_im);
        let b_re = B::Vec4::load(&self.b_re);

        let prev_re_l = flush_recursive_state_vec4(B::Vec4::load(&self.z_re));
        let prev_im_l = flush_recursive_state_vec4(B::Vec4::load(&self.z_im));
        let next_re_l = flush_recursive_state_vec4(
            a_re.mul(prev_re_l)
                .sub(a_im.mul(prev_im_l))
                .add(b_re.mul(xl)),
        );
        let next_im_l = flush_recursive_state_vec4(a_im.mul(prev_re_l).add(a_re.mul(prev_im_l)));
        next_re_l.store(&mut self.z_re);
        next_im_l.store(&mut self.z_im);

        let prev_re_r = flush_recursive_state_vec4(B::Vec4::load(&right.z_re));
        let prev_im_r = flush_recursive_state_vec4(B::Vec4::load(&right.z_im));
        let next_re_r = flush_recursive_state_vec4(
            a_re.mul(prev_re_r)
                .sub(a_im.mul(prev_im_r))
                .add(b_re.mul(xr)),
        );
        let next_im_r = flush_recursive_state_vec4(a_im.mul(prev_re_r).add(a_re.mul(prev_im_r)));
        next_re_r.store(&mut right.z_re);
        next_im_r.store(&mut right.z_im);
    }

    /// Steps all 4 lanes of a stereo pair (`self` = Left, `right` = Right) sharing `self`'s poles
    /// in 4-lane SIMD and evaluates the complex readout `sum_k (c_re[k] * z_re[k] - c_im[k] * z_im[k])`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_stereo_simd_with_readout<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
        c_re: &[f32; 4],
        c_im: &[f32; 4],
    ) -> (f32, f32) {
        use crate::math::SimdVec4;
        let xl = flush_recursive_state_vec4(B::Vec4::splat(x_l));
        let xr = flush_recursive_state_vec4(B::Vec4::splat(x_r));
        let a_re = B::Vec4::load(&self.alpha_re);
        let a_im = B::Vec4::load(&self.alpha_im);
        let b_re = B::Vec4::load(&self.b_re);
        let cr = B::Vec4::load(c_re);
        let ci = B::Vec4::load(c_im);

        let prev_re_l = flush_recursive_state_vec4(B::Vec4::load(&self.z_re));
        let prev_im_l = flush_recursive_state_vec4(B::Vec4::load(&self.z_im));
        let next_re_l = flush_recursive_state_vec4(
            a_re.mul(prev_re_l)
                .sub(a_im.mul(prev_im_l))
                .add(b_re.mul(xl)),
        );
        let next_im_l = flush_recursive_state_vec4(a_im.mul(prev_re_l).add(a_re.mul(prev_im_l)));
        next_re_l.store(&mut self.z_re);
        next_im_l.store(&mut self.z_im);

        let prev_re_r = flush_recursive_state_vec4(B::Vec4::load(&right.z_re));
        let prev_im_r = flush_recursive_state_vec4(B::Vec4::load(&right.z_im));
        let next_re_r = flush_recursive_state_vec4(
            a_re.mul(prev_re_r)
                .sub(a_im.mul(prev_im_r))
                .add(b_re.mul(xr)),
        );
        let next_im_r = flush_recursive_state_vec4(a_im.mul(prev_re_r).add(a_re.mul(prev_im_r)));
        next_re_r.store(&mut right.z_re);
        next_im_r.store(&mut right.z_im);

        let out_l_lanes = flush_recursive_state_vec4(cr.mul(next_re_l).sub(ci.mul(next_im_l)));
        let out_r_lanes = flush_recursive_state_vec4(cr.mul(next_re_r).sub(ci.mul(next_im_r)));
        let (out_l, out_r) = out_l_lanes.horizontal_sum_pair(out_r_lanes);
        (
            flush_recursive_state_f32(out_l),
            flush_recursive_state_f32(out_r),
        )
    }
}

/// Orthonormal Kautz Modal Bank (Stage III & Family 6, REQ-1.2).
#[derive(Clone, Debug)]
pub struct KautzModalBank<const POLES: usize> {
    pub s_re: [f32; POLES],
    pub s_im: [f32; POLES],
    pub lambda_re: [f32; POLES],
    pub lambda_im: [f32; POLES],
    pub b_gamma: [f32; POLES],
    pub c_re: [f32; POLES],
    pub c_im: [f32; POLES],
}

impl<const POLES: usize> Default for KautzModalBank<POLES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const POLES: usize> KautzModalBank<POLES> {
    pub const fn new() -> Self {
        Self {
            s_re: [0.0; POLES],
            s_im: [0.0; POLES],
            lambda_re: [0.0; POLES],
            lambda_im: [0.0; POLES],
            b_gamma: [1.0; POLES],
            c_re: [0.0; POLES],
            c_im: [0.0; POLES],
        }
    }

    pub fn reset(&mut self) {
        self.s_re.fill(0.0);
        self.s_im.fill(0.0);
    }

    /// Sets complex conjugate pole pair `m` with dynamic sweep contraction (`ref[dsp.op-m.kautz.sweep-contraction]`).
    pub fn set_pole(
        &mut self,
        m: usize,
        r: f32,
        theta: f32,
        delta_theta: f32,
        c_re: f32,
        c_im: f32,
    ) {
        self.set_pole_with_b_gamma(m, r, theta, delta_theta, 1.0, c_re, c_im);
    }

    /// Sets complex conjugate pole pair `m` with explicit input scaling `b_gamma` (`ref[dsp.op-m.kautz.sweep-contraction]`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_pole_with_b_gamma(
        &mut self,
        m: usize,
        r: f32,
        theta: f32,
        delta_theta: f32,
        b_gamma: f32,
        c_re: f32,
        c_im: f32,
    ) {
        if m < POLES {
            let r_effective = if !r.is_finite() {
                0.0
            } else if delta_theta.is_finite() && delta_theta.abs() > 1.0e-9 {
                eval_stage3_sweep_contraction(r, delta_theta)
            } else {
                r.clamp(0.0, 0.9995)
            };
            let theta_clean = if theta.is_finite() { theta } else { 0.0 };
            let (cos_t, sin_t) = algebraic_unit_rotor_op_m(theta_clean);
            self.lambda_re[m] = r_effective * cos_t;
            self.lambda_im[m] = r_effective * sin_t;
            self.b_gamma[m] = if b_gamma.is_finite() { b_gamma } else { 0.0 };
            self.c_re[m] = if c_re.is_finite() { c_re } else { 0.0 };
            self.c_im[m] = if c_im.is_finite() { c_im } else { 0.0 };
        }
    }

    /// Steps the modal bank by 1 sample, returning the tail synthesis value.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let u = flush_recursive_state_f32(x);
        let mut y_tail = 0.0f32;
        for m in 0..POLES {
            let prev_re = flush_recursive_state_f32(self.s_re[m]);
            let prev_im = flush_recursive_state_f32(self.s_im[m]);
            let l_re = self.lambda_re[m];
            let l_im = self.lambda_im[m];

            let next_re = l_re * prev_re - l_im * prev_im + self.b_gamma[m] * u;
            let next_im = l_im * prev_re + l_re * prev_im;

            let s_re = flush_recursive_state_f32(next_re);
            let s_im = flush_recursive_state_f32(next_im);
            self.s_re[m] = s_re;
            self.s_im[m] = s_im;

            // Tail output: 2 * Re(c_m * s_m) = 2 * (c_re * s_re - c_im * s_im)
            y_tail += 2.0 * (self.c_re[m] * s_re - self.c_im[m] * s_im);
        }
        flush_recursive_state_f32(y_tail)
    }
}

/// Initializes `num_poles` logarithmically spaced modal pole frequencies across `[f_min_hz, f_max_hz]`
/// at `sample_rate_hz` with $\mathcal{H}_2$ input normalizer $\gamma_m = \sqrt{1 - r_{\text{high}}^2}$
/// (`ref[dsp.op-m.init.log-octave-h2]`).
pub fn init_log_octave_poles_h2(
    num_poles: usize,
    f_min_hz: f64,
    f_max_hz: f64,
    sample_rate_hz: f64,
    r_high: f32,
) -> Vec<(f32, f32)> {
    if num_poles == 0 || f_min_hz <= 0.0 || f_max_hz <= f_min_hz || sample_rate_hz <= 0.0 {
        return Vec::new();
    }
    let r_clamped = r_high.clamp(0.25, 0.9995);
    let gamma = sqrt_f64((1.0 - (r_clamped as f64) * (r_clamped as f64)).max(0.0)) as f32;
    let ratio = f_max_hz / f_min_hz;
    let mut out = Vec::with_capacity(num_poles);
    for m in 0..num_poles {
        let t = if num_poles == 1 {
            0.0
        } else {
            (m as f64) / ((num_poles - 1) as f64)
        };
        let freq_hz = f_min_hz * cold_pow_f64(ratio, t);
        let theta = ((core::f64::consts::TAU * freq_hz / sample_rate_hz)
            .clamp(1.0e-4, core::f64::consts::PI - 1.0e-4)) as f32;
        out.push((theta, gamma));
    }
    out
}

/// Validates Stage III serialized Kautz parameters (`0.85 <= r_low <= r_high = r_low + dr <= 0.9995`,
/// `theta in (0, pi)`, `0.5 <= kappa <= 2.0`) and evaluates the shared radial gate
/// $g = \mathrm{clamp}(0.5 + \theta_{10}, 0, 1)$, $r_{\text{gate}} = \mathrm{clamp}(r_{\text{low}} + g \cdot dr, r_{\text{low}}, r_{\text{high}})$,
/// and $b_\gamma = \kappa \sqrt{1 - r_{\text{high}}^2}$ (`ref[dsp.op-m.kautz.radial-gate]`).
pub fn validate_and_eval_stage3_radial_gate(
    r_low: f32,
    dr: f32,
    theta: f32,
    kappa: f32,
    theta_10_mod: f32,
) -> Option<(f32, f32, f32)> {
    let r_high = r_low + dr;
    if !(0.85..=0.9995).contains(&r_low)
        || dr < 0.0
        || r_high > 0.9995
        || !(theta > 0.0 && theta < core::f32::consts::PI)
        || !(0.5..=2.0).contains(&kappa)
    {
        return None;
    }
    let g = (0.5 + theta_10_mod).clamp(0.0, 1.0);
    let r_gate = (r_low + g * dr).clamp(r_low, r_high);
    let gamma = sqrt_f64((1.0 - (r_high as f64) * (r_high as f64)).max(0.0)) as f32;
    let b_gamma = kappa * gamma;
    Some((g, r_gate, b_gamma))
}

/// Evaluates Stage III dynamic sweep contraction $r_{\text{eff}} = \min(r_m, \max(0.25, 0.9995 - 0.15 |\Delta\theta_m|))$
/// via [`dnam_kernels::eval_stage3_sweep_contraction_f32`] (`ref[dsp.op-m.kautz.sweep-contraction]`).
#[inline]
pub fn eval_stage3_sweep_contraction(r_m: f32, delta_theta_m: f32) -> f32 {
    dnam_kernels::eval_stage3_sweep_contraction_f32(r_m, delta_theta_m)
}

/// Converts a Stage III shared Kautz pole pair `(r_low, dr, theta, b_gamma, c_re, c_im)` from the
/// `48 kHz` reference rate to `effective_rate_hz` (`rho = 48000 / f_s,eff`), rescaling the complex
/// readout at $g = 0.5$ so steady-state DC/modal transfer is preserved (`ref[dsp.op-m.kautz.rate-convert]`,
/// `ref[dsp.zoh.op-m-poles]`).
pub fn convert_stage3_kautz_pole_for_rate(
    r_low: f32,
    dr: f32,
    theta: f32,
    b_gamma: f32,
    c_re: f32,
    c_im: f32,
    effective_rate_hz: f64,
) -> (f32, f32, f32, f32, f32, f32) {
    let rho = 48_000.0 / effective_rate_hz;
    if (rho - 1.0).abs() < 1e-12 {
        return (r_low, dr, theta, b_gamma, c_re, c_im);
    }
    let r_low_eff = cold_pow_f64(r_low as f64, rho) as f32;
    let r_high = (r_low + dr).min(0.9995);
    let r_high_eff = cold_pow_f64(r_high as f64, rho) as f32;
    let dr_eff = (r_high_eff - r_low_eff).max(0.0);
    let theta_eff = ((theta as f64) * rho).clamp(1.0e-4, core::f64::consts::PI - 1.0e-4) as f32;
    let gamma_48k = sqrt_f64((1.0 - (r_high as f64) * (r_high as f64)).max(1e-7));
    let kappa_m = (b_gamma as f64) / gamma_48k;
    let gamma_eff = sqrt_f64((1.0 - (r_high_eff as f64) * (r_high_eff as f64)).max(1e-7));
    let b_gamma_eff = (kappa_m * gamma_eff) as f32;

    // Rescale complex readout by (b_gamma / b_1) * (1 - lambda_1) / (1 - lambda_0) at g = 0.5
    let r0 = (r_low + 0.5 * dr) as f64;
    let r1 = (r_low_eff + 0.5 * dr_eff) as f64;
    let (c0, s0) = crate::math::contractive_rotor_f32(theta).unwrap_or((1.0, 0.0));
    let (c1, s1) = crate::math::contractive_rotor_f32(theta_eff).unwrap_or((1.0, 0.0));
    let d0_re = 1.0 - r0 * (c0 as f64);
    let d0_im = -r0 * (s0 as f64);
    let d1_re = 1.0 - r1 * (c1 as f64);
    let d1_im = -r1 * (s1 as f64);
    let d0_norm2 = (d0_re * d0_re + d0_im * d0_im).max(1e-18);
    // (d1_re + i d1_im) / (d0_re + i d0_im)
    let ratio_re = (d1_re * d0_re + d1_im * d0_im) / d0_norm2;
    let ratio_im = (d1_im * d0_re - d1_re * d0_im) / d0_norm2;
    let scale = (b_gamma as f64) / (b_gamma_eff as f64).max(1e-12);
    let m_re = scale * ratio_re;
    let m_im = scale * ratio_im;
    let c_re_eff = ((c_re as f64) * m_re - (c_im as f64) * m_im) as f32;
    let c_im_eff = ((c_re as f64) * m_im + (c_im as f64) * m_re) as f32;
    (
        r_low_eff,
        dr_eff,
        theta_eff,
        b_gamma_eff,
        c_re_eff,
        c_im_eff,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::container::ZohRescaledParams;
    use crate::math::{
        Q0_192Phase, ScalarBackend, contractive_rotor_f32, step_lfo_phasor_zero_hold,
    };
    use proptest::prelude::*;

    #[test]
    fn test_algebraic_unit_rotor_and_rsqrt() {
        for angle in [0.1, 0.5, 1.0, 1.57, 2.5, 3.1] {
            let (c, s) = algebraic_unit_rotor_op_m(angle);
            let r2 = c * c + s * s;
            assert!(r2 <= 1.00001);
            assert!(c.is_finite());
            assert!(s.is_finite());
        }

        assert!((fast_rsqrt_nr2(4.0) - 0.5).abs() < 1e-5);
        assert!((fast_rsqrt_nr2(1.0) - 1.0).abs() < 1e-5);
        assert!(fast_rsqrt_nr2(1e-15).is_finite());
    }

    #[test]
    fn test_symplectic_modal_4lane() {
        let _audio = crate::FtzDazGuard::new();
        let mut modal = SymplecticModal4Lane::default();
        modal.reset();
        let r = 0.95_f32;
        let theta = 0.4_f32;
        modal.set_pole(0, r, theta);
        modal.set_pole(1, 0.9, 0.8);
        modal.set_pole(5, 0.9, 0.8); // out of bounds check

        // Impulse at n=0 sets state to (b_re, 0); subsequent steps decay by r^n and rotate by n*theta
        let (re0, im0) = modal.step_lane(0, 1.0);
        let b_norm = modal.b_re[0];
        assert!((re0 - b_norm).abs() < 1e-6);
        assert_eq!(im0, 0.0);

        let (c_t, s_t) = algebraic_unit_rotor_op_m(theta);
        let (re1, im1) = modal.step_lane(0, 0.0);
        assert!((re1 - r * c_t * b_norm).abs() < 1e-6);
        assert!((im1 - r * s_t * b_norm).abs() < 1e-6);

        modal.reset();
        assert_eq!(modal.z_re, [0.0; 4]);
    }

    #[test]
    fn test_kautz_modal_bank() {
        let _audio = crate::FtzDazGuard::new();
        let mut bank = KautzModalBank::<4>::default();
        bank.reset();
        bank.set_pole_with_b_gamma(0, 0.90, 0.3, 0.0, 0.5, 0.8, -0.4);
        bank.set_pole(10, 0.95, 0.2, 0.01, 0.5, 0.2); // out of bounds check

        // Step 0 impulse response: s_re[0] = b_gamma * 1.0 = 0.5, s_im[0] = 0.0
        // y[0] = 2 * (c_re * s_re - c_im * s_im) = 2 * 0.8 * 0.5 = 0.8
        let y0 = bank.step(1.0);
        assert!((y0 - 0.8).abs() < 1e-6);

        // Step 1 free decay: s_re[0] = lambda_re * 0.5, s_im[0] = lambda_im * 0.5
        let expected_y1 =
            2.0 * (0.8 * (bank.lambda_re[0] * 0.5) - (-0.4) * (bank.lambda_im[0] * 0.5));
        let y1 = bank.step(0.0);
        assert!((y1 - expected_y1).abs() < 1e-6);

        bank.reset();
        assert_eq!(bank.s_re, [0.0; 4]);

        assert!(init_log_octave_poles_h2(0, 100.0, 1000.0, 48_000.0, 0.9).is_empty());
        let one = init_log_octave_poles_h2(1, 200.0, 2000.0, 48_000.0, 0.9);
        assert_eq!(one.len(), 1);
        let unchanged =
            convert_stage3_kautz_pole_for_rate(0.92, 0.04, 0.3, 0.25, 0.8, -0.1, 48_000.0);
        assert_eq!(unchanged, (0.92, 0.04, 0.3, 0.25, 0.8, -0.1));
    }

    #[test]
    fn test_kautz_modal_bank_step_scrubs_nan_and_inf_without_latching() {
        let _audio = crate::FtzDazGuard::new();
        let mut bank = KautzModalBank::<4>::default();
        for m in 0..4 {
            bank.set_pole_with_b_gamma(m, 0.90, 0.25 + 0.3 * (m as f32), 0.0, 0.4, 0.6, -0.2);
        }

        for bad_x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            bank.s_re = [f32::NAN, 0.1, f32::INFINITY, f32::NEG_INFINITY];
            bank.s_im = [f32::INFINITY, f32::NAN, -0.2, f32::NEG_INFINITY];

            let y_bad = bank.step(bad_x);
            assert!(y_bad.is_finite());
            assert!(bank.s_re.iter().all(|v| v.is_finite()));
            assert!(bank.s_im.iter().all(|v| v.is_finite()));

            let y_rec = bank.step(0.5);
            assert!(y_rec.is_finite() && y_rec.abs() > 0.0);
            assert!(bank.s_re.iter().all(|v| v.is_finite()));
            assert!(bank.s_im.iter().all(|v| v.is_finite()));
        }

        bank.reset();
        bank.s_re = [f32::NAN; 4];
        bank.s_im = [f32::INFINITY; 4];
        let y_zero = bank.step(0.0);
        assert_eq!(y_zero.to_bits(), 0);
        assert_eq!(bank.s_re.map(f32::to_bits), [0; 4]);
        assert_eq!(bank.s_im.map(f32::to_bits), [0; 4]);
    }

    #[test]
    fn test_rotor_unit_circle_proof() {
        // ref[dsp.op-m.rotor.contractive]: |u_rot| <= 1.000000000000 proof across full angle range (0, PI)
        const STEPS: usize = 500;
        let d_theta = core::f32::consts::PI / (STEPS as f32);
        for i in 1..STEPS {
            let theta = (i as f32) * d_theta;
            let (c, s) = algebraic_unit_rotor_op_m(theta);
            let r2 = c * c + s * s;
            assert!(
                r2 <= 1.000001,
                "Rotor bound violated at theta={}: c={}, s={}, r2={}",
                theta,
                c,
                s,
                r2
            );
        }
    }

    #[test]
    fn test_dynamic_sweep_contraction() {
        // Static cabinet pole (delta_theta == 0.0) preserves full r = 0.9995 (REQ-1.2 170 ms T60)
        let mut bank = KautzModalBank::<4>::default();
        bank.set_pole(0, 0.9995, 0.5, 0.0, 1.0, 0.0);
        let static_r =
            (bank.lambda_re[0] * bank.lambda_re[0] + bank.lambda_im[0] * bank.lambda_im[0]).sqrt();
        assert!(
            (static_r - 0.9995).abs() < 1e-4,
            "Static Kautz pole radius was prematurely clamped: {static_r}"
        );

        // ref[dsp.op-m.kautz.sweep-contraction]: dynamic sweep contraction bounds r_effective <= max(0.25, 0.9995 - 0.15 * |delta_theta|) when sweeping
        bank.set_pole(0, 0.9995, 0.5, 0.1, 1.0, 0.0);
        let lambda_r =
            (bank.lambda_re[0] * bank.lambda_re[0] + bank.lambda_im[0] * bank.lambda_im[0]).sqrt();
        assert!(lambda_r < 0.9995);
        assert!((lambda_r - (0.9995 - 0.15 * 0.1)).abs() < 1e-5);
    }

    /// verify[dsp.op-m.recurrence.symplectic]
    /// verify[dsp.op-m.rotor.contractive]
    /// verify[dsp.op-m.rotor.zero-hold]
    /// verify[dsp.op-m.rotor.quadrant-horner]
    /// verify[dsp.op-m.rotor.ulp-guard]
    /// verify[dsp.op-m.rotor.q0-192-lfo]
    /// verify[dsp.op-m.init.log-octave-h2]
    /// verify[dsp.op-m.ctrl.engine]
    #[test]
    fn test_op_m_symplectic_recurrence_rotor_q0_192_and_log_octave_h2_init() {
        let _guard = FtzDazGuard::new();
        let poles = init_log_octave_poles_h2(8, 25.0, 16_000.0, 48_000.0, 0.99);
        assert_eq!(poles.len(), 8);
        for w in poles.windows(2) {
            assert!(w[1].0 > w[0].0);
            assert!((w[0].1 - (1.0 - 0.99_f64 * 0.99).sqrt() as f32).abs() < 1e-6);
        }

        let mut symp = SymplecticModal4Lane::new();
        for (lane, &(theta, _)) in poles.iter().take(4).enumerate() {
            symp.set_pole(lane, 0.99, theta);
        }
        let (re0, im0) = symp.step_lane(0, 1.0);
        assert!(re0 > 0.0 && im0 == 0.0);

        let (c, s) = contractive_rotor_f32(1.0).unwrap();
        assert!((c as f64) * (c as f64) + (s as f64) * (s as f64) < 1.0);
        assert_eq!(step_lfo_phasor_zero_hold(0.6, 0.8, 0.0), Some((0.6, 0.8)));

        let mut q192 = Q0_192Phase::ZERO;
        q192.step_angle_bits((0.1_f32).to_bits()).unwrap();
        let (pc, ps) = q192.to_phasor();
        assert!((pc as f64) * (pc as f64) + (ps as f64) * (ps as f64) <= 1.0);
    }

    /// verify[dsp.op-m.kautz.tier-allocation]
    /// verify[dsp.op-m.kautz.dual-state]
    /// verify[dsp.op-m.kautz.radial-gate]
    /// verify[dsp.op-m.kautz.rate-convert]
    /// verify[dsp.op-m.kautz.coupled-form]
    /// verify[dsp.op-m.kautz.sweep-contraction]
    /// verify[dsp.zoh.op-m-poles]
    #[test]
    fn test_stage3_kautz_allocation_dual_state_radial_gate_rate_convert_and_sweep() {
        let _guard = FtzDazGuard::new();

        // 1. Radial gate validation & evaluation
        let (g, r_gate, b_gamma) =
            validate_and_eval_stage3_radial_gate(0.90, 0.08, 0.5, 1.0, 0.0).unwrap();
        assert!((g - 0.5).abs() < 1e-6);
        assert!((r_gate - 0.94).abs() < 1e-6);
        assert!(b_gamma > 0.0);
        assert!(validate_and_eval_stage3_radial_gate(0.80, 0.08, 0.5, 1.0, 0.0).is_none());
        assert!(validate_and_eval_stage3_radial_gate(0.90, 0.15, 0.5, 1.0, 0.0).is_none());

        // 2. Dynamic sweep contraction
        let r_swept = eval_stage3_sweep_contraction(0.9990, 0.1);
        assert!((r_swept - (0.9995 - 0.015)).abs() < 1e-6);

        // 3. Stage III rate conversion preserves kappa_m & ZOH Op-M pole rescaling
        let (rl_96, dr_96, th_96, bg_96, cr_96, ci_96) =
            convert_stage3_kautz_pole_for_rate(0.90, 0.05, 0.4, 0.3, 0.5, -0.2, 96_000.0);
        let (_, _, _, bg_96_double, _, _) =
            convert_stage3_kautz_pole_for_rate(0.90, 0.05, 0.4, 0.6, 0.5, -0.2, 96_000.0);
        assert!(rl_96 > 0.90 && dr_96 > 0.0 && (th_96 - 0.2).abs() < 1e-5);
        assert!(bg_96 > 0.0 && bg_96.is_finite() && cr_96.is_finite() && ci_96.is_finite());
        assert!((bg_96_double - 2.0 * bg_96).abs() < 1e-6);

        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.995, 0.99, 0.999, 64).unwrap();
        let (z_rl, z_dr, z_th, z_bg) = zoh.rescale_op_m_channel(0.90, 0.05, 0.4, 0.3);
        assert!((z_rl - rl_96).abs() < 1e-5 && z_dr > 0.0 && z_th > 0.0 && z_bg > 0.0);

        // 4. Always-dual L/R channel state & coupled-form triangle inequality norm bound
        let mut symp_l = SymplecticModal4Lane::new();
        let mut symp_r = SymplecticModal4Lane::new();
        symp_l.set_pole(0, 0.95, 0.4);
        symp_r.set_pole(0, 0.95, 0.4);
        let (out_l, out_r) = symp_l.step_4lane_stereo_simd_with_readout::<ScalarBackend>(
            &mut symp_r,
            1.0,
            -0.5,
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0; 4],
        );
        assert!(out_l > 0.0 && out_r < 0.0 && (out_l + 2.0 * out_r).abs() < 1e-6);

        let mut bank = KautzModalBank::<4>::new();
        bank.set_pole_with_b_gamma(0, 0.95, 0.4, 0.0, 0.3, 1.0, 0.0);
        let mut prev_norm = 0.0_f64;
        for _ in 0..32 {
            let u = 0.5_f32;
            let _ = bank.step(u);
            let cur_norm = ((bank.s_re[0] as f64).powi(2) + (bank.s_im[0] as f64).powi(2)).sqrt();
            assert!(cur_norm <= 0.95 * prev_norm + 0.3 * 0.5 + 1e-6);
            prev_norm = cur_norm;
        }
    }

    proptest! {
        #[test]
        fn prop_algebraic_unit_rotor_and_rsqrt(
            theta in 0.001_f32..3.10_f32,
            s in 0.001_f32..10.0_f32,
        ) {
            let (c, sn) = algebraic_unit_rotor_op_m(theta);
            let r2 = c * c + sn * sn;
            prop_assert!(r2 <= 1.000_002, "Rotor unit circle bound violated: r2={}", r2);
            prop_assert!((r2 - 1.0).abs() < 2e-5);

            let inv_s = fast_rsqrt_nr2(s);
            let expected = 1.0 / s.sqrt();
            let rel_err = ((inv_s - expected) / expected).abs();
            prop_assert!(rel_err < 1.0e-5, "fast_rsqrt_nr2 rel_err={} at s={}", rel_err, s);
        }

        #[test]
        fn prop_kautz_modal_bank_simd_and_gauge_invariance(
            r in 0.5_f32..0.999_f32,
            theta in 0.05_f32..3.0_f32,
            d_theta in -0.2_f32..0.2_f32,
            c_re in -2.0_f32..2.0_f32,
            c_im in -2.0_f32..2.0_f32,
            alpha in 0.2_f32..5.0_f32,
            u in -1.0_f32..1.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut bank_scalar = KautzModalBank::<4>::default();
            let mut bank_simd = KautzModalBank::<4>::default();
            let mut bank_gauge = KautzModalBank::<4>::default();
            let b_gamma = (1.0 - r * r).max(0.0).sqrt();
            for k in 0..4 {
                bank_scalar.set_pole_with_b_gamma(k, r, theta, d_theta, b_gamma, c_re, c_im);
                bank_simd.set_pole_with_b_gamma(k, r, theta, d_theta, b_gamma, c_re, c_im);
                bank_gauge.set_pole_with_b_gamma(
                    k,
                    r,
                    theta,
                    d_theta,
                    b_gamma * alpha,
                    c_re / alpha,
                    c_im / alpha,
                );
            }
            for _ in 0..8 {
                let y_s = bank_scalar.step(u);
                let y_v_lanes = bank_simd.step_4lane_simd::<ScalarBackend>(u);
                let y_v = y_v_lanes[0] + y_v_lanes[1] + y_v_lanes[2] + y_v_lanes[3];
                let y_g = bank_gauge.step(u);
                prop_assert!((y_s - y_v).abs() < 1e-4, "SIMD vs scalar mismatch: {} vs {}", y_s, y_v);
                prop_assert!((y_s - y_g).abs() < 1e-4, "Gauge invariance mismatch: {} vs {}", y_s, y_g);
            }

            // ref[dsp.op-m.kautz.sweep-contraction] pole contraction invariant
            for k in 0..4 {
                let lr2 = bank_scalar.lambda_re[k] * bank_scalar.lambda_re[k]
                    + bank_scalar.lambda_im[k] * bank_scalar.lambda_im[k];
                prop_assert!(lr2 <= 0.9995 * 0.9995 + 1e-5);
            }
        }
    }

    #[test]
    fn test_symplectic_and_kautz_pole_setters_scrub_nonfinite_and_overflow() {
        let _audio = crate::FtzDazGuard::new();
        let mut symp_l = SymplecticModal4Lane::default();
        let mut symp_r = SymplecticModal4Lane::default();
        symp_l.set_pole(0, f32::NAN, f32::INFINITY);
        symp_l.set_pole(1, 0.9, 0.4);
        symp_l.set_pole(2, 0.85, 0.7);
        symp_l.set_pole(3, 0.8, 1.1);
        assert!(symp_l.alpha_re.iter().all(|v| v.is_finite()));
        assert!(symp_l.alpha_im.iter().all(|v| v.is_finite()));

        symp_l.z_re = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.2];
        symp_l.z_im = [f32::INFINITY, f32::NAN, 0.1, f32::NEG_INFINITY];
        symp_r.z_re = [f32::NEG_INFINITY, f32::NAN, -0.1, f32::INFINITY];
        symp_r.z_im = [0.2, f32::INFINITY, f32::NAN, f32::NEG_INFINITY];

        let (sr, si) = symp_l.step_lane(0, f32::NAN);
        assert!(sr.is_finite() && si.is_finite());

        symp_l.step_4lane_stereo_simd::<crate::math::DefaultMathBackend>(
            &mut symp_r,
            f32::NAN,
            f32::INFINITY,
        );
        assert!(symp_l.z_re.iter().all(|v| v.is_finite()));
        assert!(symp_l.z_im.iter().all(|v| v.is_finite()));
        assert!(symp_r.z_re.iter().all(|v| v.is_finite()));
        assert!(symp_r.z_im.iter().all(|v| v.is_finite()));

        symp_l.z_re = [f32::NAN, 1.0, f32::INFINITY, 1.0];
        symp_r.z_re = [1.0, f32::NEG_INFINITY, 1.0, f32::NAN];
        let c_max = [f32::MAX; 4];
        let (out_l, out_r) = symp_l
            .step_4lane_stereo_simd_with_readout::<crate::math::DefaultMathBackend>(
                &mut symp_r,
                0.5,
                f32::NAN,
                &c_max,
                &c_max,
            );
        assert!(out_l.is_finite() && out_r.is_finite());

        let (rec_l, rec_r) = symp_l
            .step_4lane_stereo_simd_with_readout::<crate::math::DefaultMathBackend>(
                &mut symp_r,
                0.25,
                -0.25,
                &[0.5; 4],
                &[0.25; 4],
            );
        assert!(rec_l.is_finite() && rec_r.is_finite());
        assert!(rec_l.abs() > 0.0 && rec_r.abs() > 0.0);

        let mut bank = KautzModalBank::<4>::default();
        bank.set_pole_with_b_gamma(
            0,
            f32::NAN,
            f32::INFINITY,
            f32::NAN,
            f32::INFINITY,
            f32::NAN,
            0.1,
        );
        assert!(bank.lambda_re[0].is_finite() && bank.lambda_im[0].is_finite());
        assert!(bank.b_gamma[0].is_finite() && bank.c_re[0].is_finite());
    }
}
