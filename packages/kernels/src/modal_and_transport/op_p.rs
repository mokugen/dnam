//! `Op-P` three-stage input analytic basis: single upstream `5 Hz` passband-normalized DC blocker
//! (`T1DcBlockerKernel`), 12-pole Hilbert IIR lattice (`HilbertH1Kernel`), and `4`-feature
//! Riemann sub-harmonic tracker (`T1SubHarmonicKernel`) (`ref[dsp.op-m.hilbert.lattice]`,
//! `ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`,
//! `ref[dsp.op-m.tracker.toggle-sheet]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`, zero 64-bit scalars, and zero `libm` calls
//! in runtime execution (`ref[repo.code.shared-nostd-kernels]`).

use crate::simd::{
    DC_BLOCKER_5HZ_48K_F32, GpuVec4, KernelVec4, MAX_PARALRU_R_F32, flush_subnormal_f32_bits,
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

/// Compile-time `48 kHz` `5 Hz` passband-normalized DC blocker gain `0.5 * (1 + R)`
/// (`0x3F7F_EA90`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
const DC_BLOCKER_5HZ_48K_GAIN_F32: f32 = 0.5 * (1.0 + DC_BLOCKER_5HZ_48K_F32);

/// Silence squelch squared-magnitude threshold (`r^2 < 1e-12`,
/// `ref[dsp.op-m.tracker.envelope-norm]`).
const T1_SILENCE_SQUELCH_R2: f32 = 1.0e-12;

/// Half-angle `(1 + |u_re|) * 0.5` singularity floor (`ref[dsp.op-m.tracker.toggle-sheet]`).
const T1_HALF_ANGLE_EPS: f32 = 1.0e-12;

/// Exact binary32 `2^-18` tolerance for antipodal near-origin chord gating in
/// [`tracker_crosses_branch_cut_f32`] (`ref[dsp.op-m.tracker.toggle-sheet]`).
const BRANCH_CUT_CHORD_TOL: f32 = f32::from_bits((127 - 18) << 23);

const _: () = assert!(DC_BLOCKER_5HZ_48K_F32.to_bits() == 0x3F7F_D51F);
const _: () = assert!(DC_BLOCKER_5HZ_48K_GAIN_F32.to_bits() == 0x3F7F_EA90);
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

/// Single upstream `5 Hz` passband-normalized 1-pole DC blocker placed ahead of
/// [`HilbertH1Kernel`] in the canonical `Op-P` pipeline (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
///
/// Evaluates `d[n] = alpha_gain * (x[n] - x[n-1]) + pole * d[n-1]` with
/// `alpha_gain = 0.5 * (1.0 + pole)` (`DC_BLOCKER_5HZ_48K_F32 = 0x3F7F_D51F`, `0x3F7F_EA90` gain
/// at `48 kHz`), caller-overridable via continuous-time ZOH `pole`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1DcBlockerKernel {
    pole: f32,
    gain: f32,
    prev_x: f32,
    prev_y: f32,
}

impl Default for T1DcBlockerKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl T1DcBlockerKernel {
    /// Creates a new [`T1DcBlockerKernel`] configured for `48 kHz` (`DC_BLOCKER_5HZ_48K_F32`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pole: DC_BLOCKER_5HZ_48K_F32,
            gain: DC_BLOCKER_5HZ_48K_GAIN_F32,
            prev_x: 0.0,
            prev_y: 0.0,
        }
    }

    /// Creates a [`T1DcBlockerKernel`] with a rate-adapted ZOH pole clamped to
    /// `[0.0, MAX_PARALRU_R_F32]` and passband-normalized gain `0.5 * (1.0 + pole)`.
    #[inline(always)]
    #[must_use]
    pub fn with_pole(dc_pole: f32) -> Self {
        let pole = if dc_pole.is_finite() {
            dc_pole.clamp(0.0, MAX_PARALRU_R_F32)
        } else {
            DC_BLOCKER_5HZ_48K_F32
        };
        Self {
            pole,
            gain: 0.5 * (1.0 + pole),
            prev_x: 0.0,
            prev_y: 0.0,
        }
    }

    /// Clears internal delay state `(prev_x, prev_y)` to `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_y = 0.0;
    }

    /// Returns the configured `(pole, alpha_gain)` pair.
    #[inline(always)]
    #[must_use]
    pub const fn params(&self) -> (f32, f32) {
        (self.pole, self.gain)
    }

    /// Advances the `5 Hz` upstream DC blocker by one sample with subnormal flushing and
    /// non-finite scrubbing.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let x_clean = flush_subnormal_f32_bits(if x.is_finite() { x } else { 0.0 });
        let y =
            flush_subnormal_f32_bits(self.gain * (x_clean - self.prev_x) + self.pole * self.prev_y);
        if !y.is_finite() {
            self.reset();
            return 0.0;
        }
        self.prev_x = x_clean;
        self.prev_y = y;
        y
    }
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

/// Pure 32-bit `T1` sub-harmonic tracker operating on `(i_clean, q_clean)` conditioned by upstream
/// [`T1DcBlockerKernel`] -> [`HilbertH1Kernel`] (`ref[dsp.op-m.tracker.envelope-norm]`,
/// `ref[dsp.op-m.tracker.toggle-sheet]`).
///
/// Combines the 3-regime gate (`SILENCE`, `VALID`, `HYSTERESIS`), `GpuVec4` NR2 `rsqrt` analytic
/// envelope gate `env_gate`, rising-edge `-1/-2` octave divider toggles `(q_1, q_2)`, Riemann
/// half-angle sheet flip `s_sheet`, and cancellation-free half-angle recurrence to emit the exact
/// 4-channel vector:
/// `[sub1_cos * env_gate, sub1_sin * env_gate, sub1_sq * env_gate, sub2_sq * env_gate]`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1SubHarmonicKernel {
    prev_x: f32,
    prev_u_re: f32,
    prev_u_im: f32,
    prev_q: f32,
    prev_q2: f32,
    sheet: f32,
    prev_valid: bool,
}

impl Default for T1SubHarmonicKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicKernel {
    /// Creates a new [`T1SubHarmonicKernel`] in clean startup state (`prev_q = +1`, `prev_q2 = +1`,
    /// `sheet = +1`, `prev_valid = false`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev_x: 0.0,
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            prev_q: 1.0,
            prev_q2: 1.0,
            sheet: 1.0,
            prev_valid: false,
        }
    }

    /// Resets Riemann sheet, `-1/-2` octave divider latches, and phase continuity to startup
    /// defaults.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the current `(q1_toggle, q2_toggle, sheet_sign, prev_valid)` state.
    #[inline(always)]
    #[must_use]
    pub const fn state(&self) -> (f32, f32, f32, bool) {
        (self.prev_q, self.prev_q2, self.sheet, self.prev_valid)
    }

    /// Steps the `T1` sub-harmonic tracker on upstream `5 Hz` DC-blocked analytic `(i_in, q_in)`,
    /// returning
    /// `[sub1_cos * env_gate, sub1_sin * env_gate, sub1_sq * env_gate, sub2_sq * env_gate]`
    /// (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
    #[inline(always)]
    pub fn step(&mut self, i_in: f32, q_in: f32) -> [f32; 4] {
        if !i_in.is_finite() || !q_in.is_finite() {
            self.prev_valid = false;
            self.prev_x = 0.0;
            return [0.0; 4];
        }
        let i_clean = flush_subnormal_f32_bits(i_in);
        let q_clean = flush_subnormal_f32_bits(q_in);
        let r2 = i_clean * i_clean + q_clean * q_clean;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            self.prev_valid = false;
            self.prev_x = i_clean;
            return [0.0; 4];
        }

        let inv_r = GpuVec4::splat(r2).rsqrt_pos_nr2().to_array()[0];
        let u_re = (i_clean * inv_r).clamp(-1.0, 1.0);
        let u_im = (q_clean * inv_r).clamp(-1.0, 1.0);
        let env_gate = flush_subnormal_f32_bits(r2 * inv_r);

        if self.prev_valid && self.prev_x <= 0.0 && i_clean > 0.0 {
            self.prev_q = -self.prev_q;
            if self.prev_q > 0.0 {
                self.prev_q2 = -self.prev_q2;
            }
        }
        if self.prev_valid
            && tracker_crosses_branch_cut_f32(self.prev_u_re, self.prev_u_im, u_re, u_im)
        {
            self.sheet = -self.sheet;
        }

        let h = (0.5 * (1.0 + abs_f32(u_re))).max(T1_HALF_ANGLE_EPS);
        let rho_half = GpuVec4::splat(h).rsqrt_pos_nr2().to_array()[0];
        let v_maj = (h * rho_half).clamp(0.0, 1.0);
        let v_min = (abs_f32(u_im) * (0.5 * rho_half)).clamp(0.0, 1.0);
        let (c_1, abs_s_1) = if u_re >= 0.0 {
            (v_maj, v_min)
        } else {
            (v_min, v_maj)
        };
        let sigma_pos = if u_im >= 0.0 { 1.0_f32 } else { -1.0_f32 };
        let sub1_cos = self.sheet * c_1;
        let sub1_sin = self.sheet * sigma_pos * abs_s_1;

        self.prev_x = i_clean;
        self.prev_u_re = u_re;
        self.prev_u_im = u_im;
        self.prev_valid = true;

        [
            flush_subnormal_f32_bits(sub1_cos * env_gate),
            flush_subnormal_f32_bits(sub1_sin * env_gate),
            flush_subnormal_f32_bits(self.prev_q * env_gate),
            flush_subnormal_f32_bits(self.prev_q2 * env_gate),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// verify[dsp.op-m.tracker.t1-dc-blocker]
    #[test]
    fn test_t1_dc_blocker_5hz_rejects_dc_and_preserves_audio() {
        let mut dcb = T1DcBlockerKernel::default();
        assert_eq!(dcb.params().0.to_bits(), 0x3F7F_D51F);
        assert_eq!(dcb.params().1.to_bits(), 0x3F7F_EA90);

        let mut dc_out = 0.0_f32;
        for _ in 0..24_000 {
            dc_out = dcb.step(1.0);
        }
        assert!(dc_out.abs() < 1.0e-5, "residual DC = {dc_out}");

        dcb.reset();
        let mut peak_1k = 0.0_f32;
        let omega = core::f64::consts::TAU * 1_000.0 / 48_000.0;
        for n in 0..25_000 {
            let x = (0.75 + 0.5 * (omega * (n as f64)).sin()) as f32;
            let y = dcb.step(x);
            if n >= 24_000 {
                peak_1k = peak_1k.max(y.abs());
            }
        }
        assert!((peak_1k - 0.5).abs() < 1.0e-3, "peak_1k = {peak_1k}");

        assert_eq!(
            T1DcBlockerKernel::with_pole(f32::NAN).params().0.to_bits(),
            0x3F7F_D51F
        );
        assert_eq!(
            T1DcBlockerKernel::with_pole(2.0).params().0,
            MAX_PARALRU_R_F32
        );
        assert_eq!(T1DcBlockerKernel::with_pole(-0.5).params(), (0.0, 0.5));
        assert!(dcb.step(f32::NAN).is_finite());
        dcb.reset();
        assert_eq!(dcb.step(f32::NAN), 0.0);
        let _ = dcb.step(-3.0e38);
        assert_eq!(dcb.step(3.0e38), 0.0);
    }

    /// verify[dsp.op-m.hilbert.lattice]
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
        assert!(dot_iq.abs() / 2048.0 < 2.0e-3);

        let custom = HilbertH1Kernel::with_coeffs([f32::NAN; 6], [2.0; 6]);
        assert_eq!(custom.alpha_i, [0.0; 6]);
        assert_eq!(custom.alpha_q, [MAX_PARALRU_R_F32; 6]);
        let mut ssb = HilbertH1Kernel::new();
        ssb.init_dc_equilibrium(1.0);
        assert!((ssb.step_ssb(1.0, 0.0, 1.0, -1.0) + 1.0).abs() < 1.0e-5);
        assert!(ssb.step(f32::NAN).0.is_finite());
    }

    /// verify[dsp.op-m.tracker.envelope-norm]
    /// verify[dsp.op-m.tracker.toggle-sheet]
    #[test]
    fn test_t1_subharmonic_kernel_4_features_gates_and_continuity() {
        let mut t1 = T1SubHarmonicKernel::default();
        assert_eq!(t1.state(), (1.0, 1.0, 1.0, false));

        // Silence squelch, overflow (`r^2 == +Inf`), & NaN guards.
        assert_eq!(t1.step(0.0, 0.0), [0.0; 4]);
        assert_eq!(t1.step(1.0e-8, -1.0e-8), [0.0; 4]);
        assert_eq!(t1.step(f32::NAN, f32::INFINITY), [0.0; 4]);
        assert_eq!(t1.step(-1.0e20, 0.0), [0.0; 4]);
        assert!(!t1.state().3);

        // Resumption from silence does not invent a spurious zero-crossing.
        let resume = t1.step(0.5, 0.0);
        assert_eq!(t1.state(), (1.0, 1.0, 1.0, true));
        assert!((resume[0] - 0.5).abs() < 1.0e-3 && resume[1].abs() < 1.0e-6);
        assert!((resume[2] - 0.5).abs() < 1.0e-3 && (resume[3] - 0.5).abs() < 1.0e-3);

        // Cancellation-free half-angle boundary at `theta = pi` (`u_re = -1.0`).
        t1.reset();
        let at_pi = t1.step(-1.0, 0.0);
        assert!(at_pi[0].abs() < 1.0e-6 && (at_pi[1] - 1.0).abs() < 2.0e-3);

        // Branch-cut geometry: CCW across `-Re`, CW across `-Re`, `+Re`, and chord guard.
        assert!(tracker_crosses_branch_cut_f32(-0.98, 0.2, -0.98, -0.2));
        assert!(tracker_crosses_branch_cut_f32(-0.98, -0.2, -0.98, 0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.98, 0.2, 0.98, -0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.0, 1.0, 0.0, -1.0));

        // Sweep full 4*pi cycle verifying continuity of [w_cos, w_sin] and q1/q2/sheet toggles.
        t1.reset();
        let steps = 128;
        let amp = 0.75_f64;
        let (mut saw_neg_sheet, mut saw_neg_q1, mut saw_neg_q2) = (false, false, false);
        for n in 0..(steps * 2) {
            let theta = core::f64::consts::TAU * (n as f64) / (steps as f64);
            let f = t1.step((amp * theta.cos()) as f32, (amp * theta.sin()) as f32);
            let exp_cos = (amp * (0.5 * theta).cos()) as f32;
            let exp_sin = (amp * (0.5 * theta).sin()) as f32;
            assert!(
                (f[0] - exp_cos).abs() < 2.0e-3,
                "n={n}: {} vs {exp_cos}",
                f[0]
            );
            assert!(
                (f[1] - exp_sin).abs() < 2.0e-3,
                "n={n}: {} vs {exp_sin}",
                f[1]
            );
            assert!((f[2].abs() - (amp as f32)).abs() < 2.0e-3);
            assert!((f[3].abs() - (amp as f32)).abs() < 2.0e-3);
            let (q1, q2, sheet, valid) = t1.state();
            assert!(valid);
            saw_neg_sheet |= sheet < 0.0;
            saw_neg_q1 |= q1 < 0.0;
            saw_neg_q2 |= q2 < 0.0;
        }
        assert!(saw_neg_sheet && saw_neg_q1 && saw_neg_q2);
    }

    proptest! {
        #[test]
        fn prop_op_p_three_stage_pipeline_always_bounded_finite(
            samples in prop::collection::vec(-2.0_f32..2.0_f32, 1..64),
        ) {
            let mut dc = T1DcBlockerKernel::new();
            let mut h = HilbertH1Kernel::new();
            let mut t1 = T1SubHarmonicKernel::new();
            for x in samples {
                let x_dc = dc.step(x + 0.25);
                let (yi, yq) = h.step(x_dc);
                prop_assert!(x_dc.is_finite() && yi.is_finite() && yq.is_finite());
                let feat = t1.step(yi, yq);
                let env_ref = f64::from(yi).hypot(f64::from(yq)) as f32;
                for lane in feat {
                    prop_assert!(lane.is_finite());
                    prop_assert!(lane.abs() <= env_ref * 1.01 + 1.0e-5);
                }
                let half_norm = f64::from(feat[0]).hypot(f64::from(feat[1])) as f32;
                if feat[2] != 0.0 {
                    prop_assert!((half_norm - env_ref).abs() <= env_ref * 1.0e-2 + 1.0e-5);
                }
            }
        }
    }
}
