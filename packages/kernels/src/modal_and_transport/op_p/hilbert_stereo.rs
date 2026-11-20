//! Mono & true-stereo 12-pole (`6` I + `6` Q) elliptic all-pass Hilbert quadrature lattices
//! (`HilbertH1Kernel`, `HilbertH1StereoKernel`) and output-history rate-switch state remapping
//! (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.hilbert.rate-remap]`,
//! `ref[dsp.op-m.tracker.stereo-batching]`).
//!
//! Runtime audio paths are strictly 32-bit (`#![no_std]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`).

pub mod anchors;

pub use anchors::{
    HILBERT_ALPHA_I, HILBERT_ALPHA_I_BITS, HILBERT_ALPHA_Q, HILBERT_ALPHA_Q_BITS,
    HILBERT_H1_ANCHOR_BITS, HILBERT_H1_ANCHOR_RATES_HZ, HILBERT_H1_STAGES,
};
#[cfg(not(target_arch = "spirv"))]
pub use anchors::{interpolate_hilbert_h1_alphas, interpolate_hilbert_h1_alphas_f64};

use crate::simd::{MAX_PARALRU_R_F32, flush_subnormal_f32_bits, scrub_and_flush_f32_bits};

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

#[inline(always)]
fn step_allpass_chain_recording(
    input: f32,
    alpha: &[f32; HILBERT_H1_STAGES],
    s0: &mut [f32; HILBERT_H1_STAGES],
    s1: &mut [f32; HILBERT_H1_STAGES],
    y1: &mut [f32; HILBERT_H1_STAGES],
    y2: &mut [f32; HILBERT_H1_STAGES],
) -> f32 {
    let mut v = input;
    for stage in 0..HILBERT_H1_STAGES {
        let a = alpha[stage];
        let y = flush_subnormal_f32_bits(a * v + s1[stage]);
        s1[stage] = s0[stage];
        s0[stage] = flush_subnormal_f32_bits(a * y - v);
        y2[stage] = y1[stage];
        y1[stage] = y;
        v = y;
    }
    v
}

#[inline(always)]
fn write_dc_equilibrium_arrays(
    alpha_i: &[f32; HILBERT_H1_STAGES],
    alpha_q: &[f32; HILBERT_H1_STAGES],
    dc: f32,
    s0_i: &mut [f32; HILBERT_H1_STAGES],
    s1_i: &mut [f32; HILBERT_H1_STAGES],
    s0_q: &mut [f32; HILBERT_H1_STAGES],
    s1_q: &mut [f32; HILBERT_H1_STAGES],
) -> f32 {
    let clean_dc = scrub_and_flush_f32_bits(dc);
    let mut sign = -1.0_f32;
    for stage in 0..HILBERT_H1_STAGES {
        let eq_i = flush_subnormal_f32_bits(sign * (1.0 + alpha_i[stage]) * clean_dc);
        let eq_q = flush_subnormal_f32_bits(sign * (1.0 + alpha_q[stage]) * clean_dc);
        s0_i[stage] = eq_i;
        s1_i[stage] = eq_i;
        s0_q[stage] = eq_q;
        s1_q[stage] = eq_q;
        sign = -sign;
    }
    clean_dc
}

/// Remaps one 6-stage TDF-II all-pass chain's state cells `(s0, s1) = (s[n-1], s[n-2])` from
/// `old_alpha` to `new_alpha` using retained prior stage outputs `(y[n-1], y[n-2])`
/// (`ref[dsp.op-m.hilbert.rate-remap]`).
#[inline(always)]
pub fn remap_allpass_chain_states_from_outputs(
    old_alpha: &[f32; HILBERT_H1_STAGES],
    new_alpha: &[f32; HILBERT_H1_STAGES],
    s0: &mut [f32; HILBERT_H1_STAGES],
    s1: &mut [f32; HILBERT_H1_STAGES],
    y1: &[f32; HILBERT_H1_STAGES],
    y2: &[f32; HILBERT_H1_STAGES],
) {
    for stage in 0..HILBERT_H1_STAGES {
        let da = new_alpha[stage] - old_alpha[stage];
        s0[stage] = scrub_and_flush_f32_bits(s0[stage] + da * y1[stage]);
        s1[stage] = scrub_and_flush_f32_bits(s1[stage] + da * y2[stage]);
    }
}

/// Mono 12-pole (`6` I + `6` Q) transposed direct-form II all-pass Hilbert quadrature filter bank
/// (`ref[dsp.op-m.hilbert.lattice]`).
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
    /// Creates a new [`HilbertH1Kernel`] initialized to the `48 kHz` anchor lattice.
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

    /// Pre-loads exact DC steady-state equilibrium across all 12 all-pass stages for input `dc`.
    #[inline(always)]
    pub fn init_dc_equilibrium(&mut self, dc: f32) {
        self.q_prev = write_dc_equilibrium_arrays(
            &self.alpha_i,
            &self.alpha_q,
            dc,
            &mut self.s0_i,
            &mut self.s1_i,
            &mut self.s0_q,
            &mut self.s1_q,
        );
    }

    /// Advances the 12-pole Hilbert lattice by one scalar sample and returns `(i, q)`.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let x_clean = scrub_and_flush_f32_bits(x);
        let i_out = step_allpass_chain(x_clean, &self.alpha_i, &mut self.s0_i, &mut self.s1_i);
        let q_next = step_allpass_chain(x_clean, &self.alpha_q, &mut self.s0_q, &mut self.s1_q);
        let q_out = self.q_prev;
        self.q_prev = q_next;
        (i_out, q_out)
    }

    /// Evaluates single-sideband shifting with `(cos_mod, sin_mod)` and `sideband_sign`.
    #[inline(always)]
    pub fn step_ssb(&mut self, x: f32, cos_mod: f32, sin_mod: f32, sideband_sign: f32) -> f32 {
        let (i_out, q_out) = self.step(x);
        flush_subnormal_f32_bits(i_out * cos_mod + sideband_sign * q_out * sin_mod)
    }
}

/// Optional `192 B` stereo stage-output history `(y[n-1], y[n-2])` kept outside the `248 B` hot
/// state so rate transitions can remap TDF-II cells smoothly via
/// [`HilbertH1StereoKernel::remap_section_states_from_outputs`].
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HilbertH1RemapHistory {
    pub y1_i_l: [f32; HILBERT_H1_STAGES],
    pub y2_i_l: [f32; HILBERT_H1_STAGES],
    pub y1_q_l: [f32; HILBERT_H1_STAGES],
    pub y2_q_l: [f32; HILBERT_H1_STAGES],
    pub y1_i_r: [f32; HILBERT_H1_STAGES],
    pub y2_i_r: [f32; HILBERT_H1_STAGES],
    pub y1_q_r: [f32; HILBERT_H1_STAGES],
    pub y2_q_r: [f32; HILBERT_H1_STAGES],
}

impl HilbertH1RemapHistory {
    /// Clears recorded output history to zero.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Populates alternating-sign steady-state DC outputs (`stage 0 => -dc`, `stage 1 => +dc`, ...).
    #[inline(always)]
    pub fn init_dc_equilibrium(&mut self, dc_l: f32, dc_r: f32) {
        let cl = scrub_and_flush_f32_bits(dc_l);
        let cr = scrub_and_flush_f32_bits(dc_r);
        let mut sign = -1.0_f32;
        for stage in 0..HILBERT_H1_STAGES {
            self.y1_i_l[stage] = sign * cl;
            self.y2_i_l[stage] = sign * cl;
            self.y1_q_l[stage] = sign * cl;
            self.y2_q_l[stage] = sign * cl;
            self.y1_i_r[stage] = sign * cr;
            self.y2_i_r[stage] = sign * cr;
            self.y1_q_r[stage] = sign * cr;
            self.y2_q_r[stage] = sign * cr;
            sign = -sign;
        }
    }
}

/// Exact `248 B` (`align(4)`) true-stereo `H1` quadrature lattice storing `48 B` shared
/// `(alpha_i, alpha_q)` coefficients and `200 B` independent Left/Right TDF-II delay states
/// (`ref[dsp.op-m.tracker.stereo-batching]`, `ref[dsp.op-m.hilbert.rate-remap]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HilbertH1StereoKernel {
    /// Shared all-pass coefficients for the 6 `I`-branch stages.
    pub alpha_i: [f32; HILBERT_H1_STAGES],
    /// Shared all-pass coefficients for the 6 `Q`-branch stages.
    pub alpha_q: [f32; HILBERT_H1_STAGES],
    s0_i_l: [f32; HILBERT_H1_STAGES],
    s1_i_l: [f32; HILBERT_H1_STAGES],
    s0_q_l: [f32; HILBERT_H1_STAGES],
    s1_q_l: [f32; HILBERT_H1_STAGES],
    s0_i_r: [f32; HILBERT_H1_STAGES],
    s1_i_r: [f32; HILBERT_H1_STAGES],
    s0_q_r: [f32; HILBERT_H1_STAGES],
    s1_q_r: [f32; HILBERT_H1_STAGES],
    q_prev_l: f32,
    q_prev_r: f32,
}

const _: () = assert!(core::mem::size_of::<HilbertH1StereoKernel>() == 248);
const _: () = assert!(core::mem::align_of::<HilbertH1StereoKernel>() == 4);
const _: () = assert!(core::mem::size_of::<HilbertH1RemapHistory>() == 192);
const _: () = assert!(core::mem::align_of::<HilbertH1RemapHistory>() == 4);

impl Default for HilbertH1StereoKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl HilbertH1StereoKernel {
    /// Creates a true-stereo `H1` kernel initialized to the `48 kHz` anchor lattice.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            alpha_i: HILBERT_ALPHA_I,
            alpha_q: HILBERT_ALPHA_Q,
            s0_i_l: [0.0; HILBERT_H1_STAGES],
            s1_i_l: [0.0; HILBERT_H1_STAGES],
            s0_q_l: [0.0; HILBERT_H1_STAGES],
            s1_q_l: [0.0; HILBERT_H1_STAGES],
            s0_i_r: [0.0; HILBERT_H1_STAGES],
            s1_i_r: [0.0; HILBERT_H1_STAGES],
            s0_q_r: [0.0; HILBERT_H1_STAGES],
            s1_q_r: [0.0; HILBERT_H1_STAGES],
            q_prev_l: 0.0,
            q_prev_r: 0.0,
        }
    }

    /// Creates a true-stereo `H1` kernel with clamped shared all-pass coefficients.
    #[inline(always)]
    #[must_use]
    pub fn with_coeffs(
        alpha_i: [f32; HILBERT_H1_STAGES],
        alpha_q: [f32; HILBERT_H1_STAGES],
    ) -> Self {
        Self {
            alpha_i: clamp_allpass_coeffs(alpha_i),
            alpha_q: clamp_allpass_coeffs(alpha_q),
            s0_i_l: [0.0; HILBERT_H1_STAGES],
            s1_i_l: [0.0; HILBERT_H1_STAGES],
            s0_q_l: [0.0; HILBERT_H1_STAGES],
            s1_q_l: [0.0; HILBERT_H1_STAGES],
            s0_i_r: [0.0; HILBERT_H1_STAGES],
            s1_i_r: [0.0; HILBERT_H1_STAGES],
            s0_q_r: [0.0; HILBERT_H1_STAGES],
            s1_q_r: [0.0; HILBERT_H1_STAGES],
            q_prev_l: 0.0,
            q_prev_r: 0.0,
        }
    }

    /// Clears all Left and Right TDF-II delay cells and `q_prev` latches to zero.
    #[inline(always)]
    pub fn reset(&mut self) {
        let mut copy = Self::with_coeffs(self.alpha_i, self.alpha_q);
        copy.alpha_i = self.alpha_i;
        copy.alpha_q = self.alpha_q;
        *self = copy;
    }

    /// Loads independent Left and Right DC steady-state equilibrium across all 24 stages.
    #[inline(always)]
    pub fn init_dc_equilibrium(&mut self, dc_l: f32, dc_r: f32) {
        self.q_prev_l = write_dc_equilibrium_arrays(
            &self.alpha_i,
            &self.alpha_q,
            dc_l,
            &mut self.s0_i_l,
            &mut self.s1_i_l,
            &mut self.s0_q_l,
            &mut self.s1_q_l,
        );
        self.q_prev_r = write_dc_equilibrium_arrays(
            &self.alpha_i,
            &self.alpha_q,
            dc_r,
            &mut self.s0_i_r,
            &mut self.s1_i_r,
            &mut self.s0_q_r,
            &mut self.s1_q_r,
        );
    }

    /// Advances both Left and Right `H1` lattices and returns `((i_l, q_l), (i_r, q_r))`.
    #[inline(always)]
    pub fn step_stereo(&mut self, x_l: f32, x_r: f32) -> ((f32, f32), (f32, f32)) {
        let xl_clean = scrub_and_flush_f32_bits(x_l);
        let xr_clean = scrub_and_flush_f32_bits(x_r);
        let i_l = step_allpass_chain(xl_clean, &self.alpha_i, &mut self.s0_i_l, &mut self.s1_i_l);
        let q_l_next =
            step_allpass_chain(xl_clean, &self.alpha_q, &mut self.s0_q_l, &mut self.s1_q_l);
        let i_r = step_allpass_chain(xr_clean, &self.alpha_i, &mut self.s0_i_r, &mut self.s1_i_r);
        let q_r_next =
            step_allpass_chain(xr_clean, &self.alpha_q, &mut self.s0_q_r, &mut self.s1_q_r);
        let q_l = self.q_prev_l;
        let q_r = self.q_prev_r;
        self.q_prev_l = q_l_next;
        self.q_prev_r = q_r_next;
        ((i_l, q_l), (i_r, q_r))
    }

    /// Advances both channels while updating external `hist` with each stage's `(y[n-1], y[n-2])`.
    #[inline(always)]
    pub fn step_stereo_recording(
        &mut self,
        x_l: f32,
        x_r: f32,
        hist: &mut HilbertH1RemapHistory,
    ) -> ((f32, f32), (f32, f32)) {
        let xl_clean = scrub_and_flush_f32_bits(x_l);
        let xr_clean = scrub_and_flush_f32_bits(x_r);
        let i_l = step_allpass_chain_recording(
            xl_clean,
            &self.alpha_i,
            &mut self.s0_i_l,
            &mut self.s1_i_l,
            &mut hist.y1_i_l,
            &mut hist.y2_i_l,
        );
        let q_l_next = step_allpass_chain_recording(
            xl_clean,
            &self.alpha_q,
            &mut self.s0_q_l,
            &mut self.s1_q_l,
            &mut hist.y1_q_l,
            &mut hist.y2_q_l,
        );
        let i_r = step_allpass_chain_recording(
            xr_clean,
            &self.alpha_i,
            &mut self.s0_i_r,
            &mut self.s1_i_r,
            &mut hist.y1_i_r,
            &mut hist.y2_i_r,
        );
        let q_r_next = step_allpass_chain_recording(
            xr_clean,
            &self.alpha_q,
            &mut self.s0_q_r,
            &mut self.s1_q_r,
            &mut hist.y1_q_r,
            &mut hist.y2_q_r,
        );
        let q_l = self.q_prev_l;
        let q_r = self.q_prev_r;
        self.q_prev_l = q_l_next;
        self.q_prev_r = q_r_next;
        ((i_l, q_l), (i_r, q_r))
    }

    /// Transitions `(self.alpha_i, self.alpha_q)` to clamped `(new_alpha_i, new_alpha_q)` while
    /// remapping all L/R TDF-II state cells from retained stage outputs `hist`.
    #[inline(always)]
    pub fn remap_section_states_from_outputs(
        &mut self,
        new_alpha_i: [f32; HILBERT_H1_STAGES],
        new_alpha_q: [f32; HILBERT_H1_STAGES],
        hist: &HilbertH1RemapHistory,
    ) {
        let clamped_i = clamp_allpass_coeffs(new_alpha_i);
        let clamped_q = clamp_allpass_coeffs(new_alpha_q);
        remap_allpass_chain_states_from_outputs(
            &self.alpha_i,
            &clamped_i,
            &mut self.s0_i_l,
            &mut self.s1_i_l,
            &hist.y1_i_l,
            &hist.y2_i_l,
        );
        remap_allpass_chain_states_from_outputs(
            &self.alpha_q,
            &clamped_q,
            &mut self.s0_q_l,
            &mut self.s1_q_l,
            &hist.y1_q_l,
            &hist.y2_q_l,
        );
        remap_allpass_chain_states_from_outputs(
            &self.alpha_i,
            &clamped_i,
            &mut self.s0_i_r,
            &mut self.s1_i_r,
            &hist.y1_i_r,
            &hist.y2_i_r,
        );
        remap_allpass_chain_states_from_outputs(
            &self.alpha_q,
            &clamped_q,
            &mut self.s0_q_r,
            &mut self.s1_q_r,
            &hist.y1_q_r,
            &hist.y2_q_r,
        );
        self.alpha_i = clamped_i;
        self.alpha_q = clamped_q;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (mut e_i, mut e_q) = (0.0_f64, 0.0_f64);
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
        let (mut max_env_err, mut dot_iq) = (0.0_f64, 0.0_f64);
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

    /// verify[dsp.op-m.hilbert.rate-remap]
    /// verify[dsp.op-m.tracker.stereo-batching]
    #[test]
    fn test_hilbert_h1_stereo_independence_and_rate_remap() {
        let (ai_96, aq_96) = interpolate_hilbert_h1_alphas_f64(96_000.0);
        let st_def = HilbertH1StereoKernel::default();
        assert_eq!(st_def.alpha_i.map(f32::to_bits), HILBERT_ALPHA_I_BITS);
        let mut st = HilbertH1StereoKernel::with_coeffs(HILBERT_ALPHA_I, HILBERT_ALPHA_Q);
        let mut mono_r_96 = HilbertH1Kernel::with_coeffs(ai_96, aq_96);
        let mut hist = HilbertH1RemapHistory::default();

        st.init_dc_equilibrium(0.625, -0.375);
        hist.init_dc_equilibrium(0.625, -0.375);
        st.remap_section_states_from_outputs(ai_96, aq_96, &hist);
        let ((i_l, q_l), (i_r, q_r)) = st.step_stereo_recording(0.625, -0.375, &mut hist);
        assert!((i_l - 0.625).abs() < 1.0e-5 && (q_l - 0.625).abs() < 1.0e-5);
        assert!((i_r + 0.375).abs() < 1.0e-5 && (q_r + 0.375).abs() < 1.0e-5);
        hist.reset();
        assert_eq!(hist.y1_i_l, [0.0; HILBERT_H1_STAGES]);

        st.reset();
        mono_r_96.reset();
        for n in 0..32 {
            let xr = (0.3 * (n as f32)).sin();
            let ((_, _), (ir, qr)) = st.step_stereo(0.0, xr);
            let (mir, mqr) = mono_r_96.step(xr);
            assert_eq!(ir.to_bits(), mir.to_bits());
            assert_eq!(qr.to_bits(), mqr.to_bits());
        }
    }
}
