//! `Op-P` phase-quadrature Hilbert IIR lattice (`HilbertH1Kernel`) and Riemann sub-harmonic
//! tracker (`T1SubHarmonicKernel`) (`ref[dsp.op-m.hilbert.lattice]`,
//! `ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`,
//! `ref[dsp.op-m.tracker.toggle-sheet]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`, zero 64-bit scalars, and zero `libm` calls
//! in runtime execution (`ref[repo.code.shared-nostd-kernels]`).

use crate::simd::{
    DC_BLOCKER_10HZ_48K_BITS, GpuVec4, KernelVec4, MAX_PARALRU_R_F32, flush_subnormal_f32_bits,
};

/// Number of second-order all-pass stages per quadrature branch in [`HilbertH1Kernel`].
pub const HILBERT_H1_STAGES: usize = 6;

/// `48 kHz` anchor all-pass pole bit patterns for the `I` branch (`ref[dsp.op-m.hilbert.lattice]`).
pub const HILBERT_ALPHA_I_BITS: [u32; HILBERT_H1_STAGES] = [
    0x3D91_BEE6,
    0x3EEC_B8E1,
    0x3F48_DE0D,
    0x3F6D_91C9,
    0x3F7A_5939,
    0x3F7E_8E9D,
];

/// `48 kHz` anchor all-pass pole bit patterns for the `Q` branch (`ref[dsp.op-m.hilbert.lattice]`).
pub const HILBERT_ALPHA_Q_BITS: [u32; HILBERT_H1_STAGES] = [
    0x3E80_025F,
    0x3F25_F6B0,
    0x3F5F_ADA1,
    0x3F75_B332,
    0x3F7C_FFA1,
    0x3F7F_92D2,
];

/// Compile-time `48 kHz` `I`-branch Hilbert all-pass coefficients derived from
/// [`HILBERT_ALPHA_I_BITS`].
pub const HILBERT_ALPHA_I: [f32; HILBERT_H1_STAGES] = bits_to_f32_array6(HILBERT_ALPHA_I_BITS);

/// Compile-time `48 kHz` `Q`-branch Hilbert all-pass coefficients derived from
/// [`HILBERT_ALPHA_Q_BITS`].
pub const HILBERT_ALPHA_Q: [f32; HILBERT_H1_STAGES] = bits_to_f32_array6(HILBERT_ALPHA_Q_BITS);

/// Silence squelch squared-magnitude threshold (`r^2 < 1e-12`,
/// `ref[dsp.op-m.tracker.envelope-norm]`).
const T1_SILENCE_SQUELCH_R2: f32 = 1.0e-12;

/// Half-angle `(1 + u_re) * 0.5` singularity floor at `u_re = -1.0`.
const T1_HALF_ANGLE_EPS: f32 = 1.0e-12;

/// Exact binary32 `2^-18` tolerance for antipodal near-origin chord gating in
/// [`tracker_crosses_branch_cut_f32`] (`ref[dsp.op-m.tracker.toggle-sheet]`).
const BRANCH_CUT_CHORD_TOL: f32 = f32::from_bits((127 - 18) << 23);

const _: () = assert!(BRANCH_CUT_CHORD_TOL > 0.0 && BRANCH_CUT_CHORD_TOL < 1.0e-5);
const _: () = assert!(HILBERT_ALPHA_I[0] > 0.0 && HILBERT_ALPHA_I[5] < 1.0);
const _: () = assert!(HILBERT_ALPHA_Q[0] > 0.0 && HILBERT_ALPHA_Q[5] < 1.0);

#[inline(always)]
const fn bits_to_f32_array6(bits: [u32; HILBERT_H1_STAGES]) -> [f32; HILBERT_H1_STAGES] {
    [
        f32::from_bits(bits[0]),
        f32::from_bits(bits[1]),
        f32::from_bits(bits[2]),
        f32::from_bits(bits[3]),
        f32::from_bits(bits[4]),
        f32::from_bits(bits[5]),
    ]
}

#[inline(always)]
const fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

#[inline(always)]
fn clamp_allpass_coeffs(src: [f32; HILBERT_H1_STAGES]) -> [f32; HILBERT_H1_STAGES] {
    let mut out = [0.0_f32; HILBERT_H1_STAGES];
    let mut idx = 0;
    while idx < HILBERT_H1_STAGES {
        let a = src[idx];
        out[idx] = if a.is_finite() {
            a.clamp(-MAX_PARALRU_R_F32, MAX_PARALRU_R_F32)
        } else {
            0.0
        };
        idx += 1;
    }
    out
}

#[inline(always)]
fn step_allpass_chain(
    input: f32,
    alpha: &[f32; HILBERT_H1_STAGES],
    s0: &mut [f32; HILBERT_H1_STAGES],
    s1: &mut [f32; HILBERT_H1_STAGES],
) -> f32 {
    let mut v = input;
    for stage in 0..HILBERT_H1_STAGES {
        let a = alpha[stage];
        let y = flush_subnormal_f32_bits(a * v + s1[stage]);
        s1[stage] = s0[stage];
        s0[stage] = flush_subnormal_f32_bits(a * y - v);
        v = y;
    }
    v
}

/// 12-pole (`6` I + `6` Q) transposed direct-form II all-pass Hilbert quadrature filter bank
/// (`ref[dsp.op-m.hilbert.lattice]`).
///
/// Produces an analytic quadrature pair `(i_out, q_out)` with `~90 deg` phase separation across
/// the audio passband using strictly 32-bit `f32` arithmetic.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HilbertH1Kernel {
    /// All-pass coefficients for the 6 `I`-branch stages.
    pub alpha_i: [f32; HILBERT_H1_STAGES],
    /// All-pass coefficients for the 6 `Q`-branch stages.
    pub alpha_q: [f32; HILBERT_H1_STAGES],
    s0_i: [f32; HILBERT_H1_STAGES],
    s1_i: [f32; HILBERT_H1_STAGES],
    s0_q: [f32; HILBERT_H1_STAGES],
    s1_q: [f32; HILBERT_H1_STAGES],
    q_prev: f32,
}

impl Default for HilbertH1Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl HilbertH1Kernel {
    /// Creates a new [`HilbertH1Kernel`] initialized to the `48 kHz` anchor lattice with zeroed
    /// filter states.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha_i: HILBERT_ALPHA_I,
            alpha_q: HILBERT_ALPHA_Q,
            s0_i: [0.0; HILBERT_H1_STAGES],
            s1_i: [0.0; HILBERT_H1_STAGES],
            s0_q: [0.0; HILBERT_H1_STAGES],
            s1_q: [0.0; HILBERT_H1_STAGES],
            q_prev: 0.0,
        }
    }

    /// Creates a [`HilbertH1Kernel`] with custom rate-adapted all-pass coefficients clamped to
    /// `[-MAX_PARALRU_R_F32, MAX_PARALRU_R_F32]`.
    #[inline(always)]
    #[must_use]
    pub fn with_coeffs(
        alpha_i: [f32; HILBERT_H1_STAGES],
        alpha_q: [f32; HILBERT_H1_STAGES],
    ) -> Self {
        Self {
            alpha_i: clamp_allpass_coeffs(alpha_i),
            alpha_q: clamp_allpass_coeffs(alpha_q),
            s0_i: [0.0; HILBERT_H1_STAGES],
            s1_i: [0.0; HILBERT_H1_STAGES],
            s0_q: [0.0; HILBERT_H1_STAGES],
            s1_q: [0.0; HILBERT_H1_STAGES],
            q_prev: 0.0,
        }
    }

    /// Clears all internal delay states and the `Q`-branch unit delay to zero.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.s0_i = [0.0; HILBERT_H1_STAGES];
        self.s1_i = [0.0; HILBERT_H1_STAGES];
        self.s0_q = [0.0; HILBERT_H1_STAGES];
        self.s1_q = [0.0; HILBERT_H1_STAGES];
        self.q_prev = 0.0;
    }

    /// Pre-loads exact DC steady-state equilibrium across all 12 all-pass stages for a constant
    /// input `dc`, eliminating startup transients when jumping onto a steady bias.
    #[inline(always)]
    pub fn init_dc_equilibrium(&mut self, dc: f32) {
        let clean_dc = flush_subnormal_f32_bits(if dc.is_finite() { dc } else { 0.0 });
        let mut sign_i = -1.0_f32;
        let mut sign_q = -1.0_f32;
        for stage in 0..HILBERT_H1_STAGES {
            let eq_i = flush_subnormal_f32_bits(sign_i * (1.0 + self.alpha_i[stage]) * clean_dc);
            let eq_q = flush_subnormal_f32_bits(sign_q * (1.0 + self.alpha_q[stage]) * clean_dc);
            self.s0_i[stage] = eq_i;
            self.s1_i[stage] = eq_i;
            self.s0_q[stage] = eq_q;
            self.s1_q[stage] = eq_q;
            sign_i = -sign_i;
            sign_q = -sign_q;
        }
        self.q_prev = clean_dc;
    }

    /// Advances the branch-free 12-pole Hilbert lattice by one scalar sample and returns `(i, q)`
    /// (`ref[dsp.op-m.hilbert.lattice]`).
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let x_clean = flush_subnormal_f32_bits(if x.is_finite() { x } else { 0.0 });
        let i_out = step_allpass_chain(x_clean, &self.alpha_i, &mut self.s0_i, &mut self.s1_i);
        let q_next = step_allpass_chain(x_clean, &self.alpha_q, &mut self.s0_q, &mut self.s1_q);
        let q_out = self.q_prev;
        self.q_prev = q_next;
        (i_out, q_out)
    }

    /// Evaluates single-sideband frequency shifting using quadrature carrier `(cos_mod, sin_mod)`
    /// and `sideband_sign` (`+1.0` upper / `-1.0` lower sideband).
    #[inline(always)]
    pub fn step_ssb(&mut self, x: f32, cos_mod: f32, sin_mod: f32, sideband_sign: f32) -> f32 {
        let (i_out, q_out) = self.step(x);
        flush_subnormal_f32_bits(i_out * cos_mod + sideband_sign * q_out * sin_mod)
    }
}

/// Pure-`f32` negative-real-axis branch-cut detector with near-origin `2^-18` antipodal chord
/// gating (`ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline(always)]
#[must_use]
pub fn tracker_crosses_branch_cut_f32(
    prev_re: f32,
    prev_im: f32,
    cur_re: f32,
    cur_im: f32,
) -> bool {
    let dot = prev_re * cur_re + prev_im * cur_im;
    let det = prev_re * cur_im - prev_im * cur_re;
    if dot < 0.0 && abs_f32(det) <= BRANCH_CUT_CHORD_TOL {
        return false;
    }
    (prev_im >= 0.0 && cur_im < 0.0 && det > 0.0) || (prev_im < 0.0 && cur_im >= 0.0 && det < 0.0)
}

/// Pure 32-bit `T1` sub-harmonic tracker combining a `10 Hz` passband-normalized DC blocker,
/// `r^2 < 1e-12` silence squelch, `GpuVec4` NR2 `rsqrt` envelope normalization, rising-edge toggle
/// latch, and Riemann half-angle sheet flip (`ref[dsp.op-m.tracker.t1-dc-blocker]`,
/// `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1SubHarmonicKernel {
    dc_pole: f32,
    dc_gain: f32,
    dc_x_prev: f32,
    dc_y_prev: f32,
    prev_x: f32,
    prev_u_re: f32,
    prev_u_im: f32,
    prev_q: f32,
    sheet: f32,
    prev_valid: bool,
}

impl Default for T1SubHarmonicKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicKernel {
    /// Creates a new [`T1SubHarmonicKernel`] with the compile-time `10 Hz @ 48 kHz` DC blocker pole
    /// (`DC_BLOCKER_10HZ_48K_BITS`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        let dc_pole = f32::from_bits(DC_BLOCKER_10HZ_48K_BITS);
        Self {
            dc_pole,
            dc_gain: 0.5 * (1.0 + dc_pole),
            dc_x_prev: 0.0,
            dc_y_prev: 0.0,
            prev_x: 0.0,
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            prev_q: 1.0,
            sheet: 1.0,
            prev_valid: false,
        }
    }

    /// Creates a [`T1SubHarmonicKernel`] with a rate-adapted DC blocker pole clamped to
    /// `[0.0, MAX_PARALRU_R_F32]`.
    #[inline(always)]
    #[must_use]
    pub fn with_dc_pole(dc_pole: f32) -> Self {
        let pole = if dc_pole.is_finite() {
            dc_pole.clamp(0.0, MAX_PARALRU_R_F32)
        } else {
            f32::from_bits(DC_BLOCKER_10HZ_48K_BITS)
        };
        Self {
            dc_pole: pole,
            dc_gain: 0.5 * (1.0 + pole),
            ..Self::new()
        }
    }

    /// Resets DC blocker and Riemann tracker state to startup defaults.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.dc_x_prev = 0.0;
        self.dc_y_prev = 0.0;
        self.prev_x = 0.0;
        self.prev_u_re = 1.0;
        self.prev_u_im = 0.0;
        self.prev_q = 1.0;
        self.sheet = 1.0;
        self.prev_valid = false;
    }

    /// Returns the configured `(dc_pole, dc_gain)` pair.
    #[inline(always)]
    #[must_use]
    pub const fn dc_params(&self) -> (f32, f32) {
        (self.dc_pole, self.dc_gain)
    }

    /// Returns the current `(toggle_latch, sheet_sign, prev_valid)` state.
    #[inline(always)]
    #[must_use]
    pub const fn state(&self) -> (f32, f32, bool) {
        (self.prev_q, self.sheet, self.prev_valid)
    }

    /// Steps the internal pure-`f32` passband-normalized `10 Hz` DC blocker on `x`.
    #[inline(always)]
    pub fn step_dc_blocker(&mut self, x: f32) -> f32 {
        let x_clean = flush_subnormal_f32_bits(if x.is_finite() { x } else { 0.0 });
        let y = flush_subnormal_f32_bits(
            self.dc_gain * (x_clean - self.dc_x_prev) + self.dc_pole * self.dc_y_prev,
        );
        self.dc_x_prev = x_clean;
        self.dc_y_prev = y;
        y
    }

    /// Steps the full `T1` sub-harmonic tracker from raw `(i_raw, q_raw)` analytic components:
    /// runs the `10 Hz` DC blocker on `i_raw`, then executes envelope normalization, toggle latch,
    /// and Riemann sheet continuous half-angle output.
    #[inline(always)]
    pub fn step(&mut self, i_raw: f32, q_raw: f32) -> f32 {
        if !i_raw.is_finite() || !q_raw.is_finite() {
            self.prev_valid = false;
            return 0.0;
        }
        let i_dc = self.step_dc_blocker(i_raw);
        self.step_conditioned(i_dc, q_raw)
    }

    /// Advances the `T1` tracker on an already DC-blocked sample `i_dc` and quadrature `q_raw`.
    #[inline(always)]
    pub fn step_conditioned(&mut self, i_dc: f32, q_raw: f32) -> f32 {
        if !i_dc.is_finite() || !q_raw.is_finite() {
            self.prev_valid = false;
            return 0.0;
        }
        let i_clean = flush_subnormal_f32_bits(i_dc);
        let q_clean = flush_subnormal_f32_bits(q_raw);
        let r2 = i_clean * i_clean + q_clean * q_clean;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            self.prev_valid = false;
            self.prev_x = i_clean;
            return 0.0;
        }

        let inv_r = GpuVec4::splat(r2).rsqrt_pos_nr2().to_array()[0];
        let u_re = (i_clean * inv_r).clamp(-1.0, 1.0);
        let u_im = (q_clean * inv_r).clamp(-1.0, 1.0);

        if self.prev_x <= 0.0 && i_clean > 0.0 {
            self.prev_q = -self.prev_q;
        }
        if self.prev_valid
            && tracker_crosses_branch_cut_f32(self.prev_u_re, self.prev_u_im, u_re, u_im)
        {
            self.sheet = -self.sheet;
        }

        let half = ((1.0 + u_re) * 0.5).max(0.0);
        let sub_continuous = if half > T1_HALF_ANGLE_EPS {
            let rsqrt = GpuVec4::splat(half).rsqrt_pos_nr2().to_array()[0];
            (half * rsqrt).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.prev_x = i_clean;
        self.prev_u_re = u_re;
        self.prev_u_im = u_im;
        self.prev_valid = true;

        flush_subnormal_f32_bits(0.5 * (self.prev_q + self.sheet * sub_continuous))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_hilbert_h1_quadrature_energy_dc_equilibrium_and_ssb() {
        let mut h = HilbertH1Kernel::default();
        assert_eq!(h.alpha_i.map(f32::to_bits), HILBERT_ALPHA_I_BITS);
        assert_eq!(h.alpha_q.map(f32::to_bits), HILBERT_ALPHA_Q_BITS);

        h.init_dc_equilibrium(0.75);
        for _ in 0..8 {
            let (i_dc, q_dc) = h.step(0.75);
            assert!((i_dc - 0.75).abs() < 1.0e-5);
            assert!((q_dc - 0.75).abs() < 1.0e-5);
        }
        h.init_dc_equilibrium(f32::NAN);
        assert_eq!(h.step(0.0), (0.0, 0.0));

        h.reset();
        let mut e_i = 0.0_f64;
        let mut e_q = 0.0_f64;
        for n in 0..4096 {
            let x = if n == 0 { 1.0_f32 } else { 0.0_f32 };
            let (yi, yq) = h.step(x);
            e_i += f64::from(yi) * f64::from(yi);
            e_q += f64::from(yq) * f64::from(yq);
        }
        assert!((e_i - 1.0).abs() < 1.0e-3);
        assert!((e_q - 1.0).abs() < 1.0e-3);

        // 1 kHz quadrature accuracy check at 48 kHz after initial settle.
        h.reset();
        let omega = core::f64::consts::TAU * 1_000.0 / 48_000.0;
        let mut max_env_err = 0.0_f64;
        let mut dot_iq = 0.0_f64;
        for n in 0..4096 {
            let x = (omega * (n as f64)).cos() as f32;
            let (yi, yq) = h.step(x);
            if n >= 2048 {
                let env2 = f64::from(yi) * f64::from(yi) + f64::from(yq) * f64::from(yq);
                max_env_err = max_env_err.max((env2 - 1.0).abs());
                dot_iq += f64::from(yi) * f64::from(yq);
            }
        }
        assert!(max_env_err < 2.0e-3, "max_env_err = {max_env_err}");
        assert!(
            dot_iq.abs() / 2048.0 < 2.0e-3,
            "avg dot_iq = {}",
            dot_iq / 2048.0
        );

        let custom = HilbertH1Kernel::with_coeffs([f32::NAN; 6], [2.0; 6]);
        assert_eq!(custom.alpha_i, [0.0; 6]);
        assert_eq!(custom.alpha_q, [MAX_PARALRU_R_F32; 6]);
        let mut ssb = HilbertH1Kernel::new();
        ssb.init_dc_equilibrium(1.0);
        let shifted = ssb.step_ssb(1.0, 0.0, 1.0, -1.0);
        assert!((shifted + 1.0).abs() < 1.0e-5);
        assert!(ssb.step(f32::NAN).0.is_finite());
    }

    #[test]
    fn test_t1_subharmonic_kernel_all_branches_and_continuity() {
        let mut t1 = T1SubHarmonicKernel::default();
        assert_eq!(t1.dc_params().0.to_bits(), DC_BLOCKER_10HZ_48K_BITS);
        assert_eq!(t1.state(), (1.0, 1.0, false));

        // Silence squelch, overflow (`r^2 == +Inf`), & NaN guards.
        assert_eq!(t1.step(0.0, 0.0), 0.0);
        assert_eq!(t1.step(1.0e-8, -1.0e-8), 0.0);
        assert_eq!(t1.step(f32::NAN, f32::INFINITY), 0.0);
        assert_eq!(t1.step_conditioned(f32::NAN, 1.0), 0.0);
        assert_eq!(t1.step_conditioned(-1.0e20, 0.0), 0.0);
        assert!(!t1.state().2);

        // Exact half-angle boundary (`u_re = -1.0`) hits `half <= T1_HALF_ANGLE_EPS`.
        t1.reset();
        let at_pi_clamped = t1.step_conditioned(-0.527, 0.0);
        assert!((at_pi_clamped - 0.5).abs() < 5.0e-4);
        t1.reset();
        let at_pi = t1.step_conditioned(-1.0, 0.0);
        assert!((at_pi - 0.5).abs() < 2.0e-3);

        // Branch-cut geometry: CCW across `-Re`, CW across `-Re`, chord guard, and `+Re` crossing.
        assert!(tracker_crosses_branch_cut_f32(-0.98, 0.2, -0.98, -0.2));
        assert!(tracker_crosses_branch_cut_f32(-0.98, -0.2, -0.98, 0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.98, 0.2, 0.98, -0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.0, 1.0, 0.0, -1.0));

        // Sweep full 4*pi cycle via `step_conditioned` to verify toggle + Riemann sheet flip.
        t1.reset();
        let steps = 128;
        let mut saw_neg_sheet = false;
        let mut saw_neg_toggle = false;
        let mut prev_sub = 0.0_f32;
        for n in 0..(steps * 2) {
            let theta = core::f64::consts::TAU * (n as f64) / (steps as f64);
            let y = t1.step_conditioned(theta.cos() as f32, theta.sin() as f32);
            assert!(y.is_finite() && y.abs() <= 1.0001);
            let (q, sheet, valid) = t1.state();
            assert!(valid);
            if sheet < 0.0 {
                saw_neg_sheet = true;
            }
            if q < 0.0 {
                saw_neg_toggle = true;
            }
            if n > 0 {
                prev_sub = y;
            }
        }
        assert!(prev_sub.is_finite());
        assert!(saw_neg_sheet && saw_neg_toggle);

        // Test `step()` with DC-offset 200 Hz sine driving real DC blocker.
        let mut t1_custom = T1SubHarmonicKernel::with_dc_pole(0.995);
        assert_eq!(t1_custom.dc_params().0, 0.995);
        let t1_nan = T1SubHarmonicKernel::with_dc_pole(f32::NAN);
        assert_eq!(t1_nan.dc_params().0.to_bits(), DC_BLOCKER_10HZ_48K_BITS);
        for n in 0..512 {
            let ph = core::f64::consts::TAU * 200.0 * (n as f64) / 48_000.0;
            let y = t1_custom.step((ph.cos() + 0.5) as f32, ph.sin() as f32);
            assert!(y.is_finite() && y.abs() <= 1.05);
        }
    }

    proptest! {
        #[test]
        fn prop_t1_subharmonic_always_bounded_finite(
            samples in prop::collection::vec((-2.0_f32..2.0_f32, -2.0_f32..2.0_f32), 1..64),
        ) {
            let mut h = HilbertH1Kernel::new();
            let mut t1 = T1SubHarmonicKernel::new();
            for (x, q_ext) in samples {
                let (yi, yq) = h.step(x);
                prop_assert!(yi.is_finite() && yq.is_finite());
                let sub = t1.step(yi, yq + 0.01 * q_ext);
                prop_assert!(sub.is_finite() && sub.abs() <= 1.01);
            }
        }
    }
}
