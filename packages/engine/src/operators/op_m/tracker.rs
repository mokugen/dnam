//! `T1` Sub-Harmonic Tracker, 160-Bit Midpoint `5 Hz` DC Blocker, Exponent-Split Phasor
//! Normalization, and Riemann Branch-Cut Toggle Sheet (`ref[dsp.op-m.tracker.t1-dc-blocker]`,
//! `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).

pub mod dc_blocker;

pub use dc_blocker::{
    OpMBackend, T1DcBlocker, compute_t1_dc_blocker_coeffs_bits, tracker_exp_scales_scalar_f32,
};

use super::kautz::fast_rsqrt_nr2;
use crate::math::flush_recursive_state_f32;
use dc_blocker::{
    T1_DC_BLOCKER_5HZ_48K_GAIN, T1_DC_BLOCKER_5HZ_48K_POLE, step_t1_dc_blocker_state,
};

/// Silence squelch squared-magnitude threshold (`r^2 < 1e-12`,
/// `ref[dsp.op-m.tracker.envelope-norm]`).
const T1_SILENCE_SQUELCH_R2: f32 = 1.0e-12;

/// Half-angle `(1 + |u_re|) * 0.5` factor (`ref[dsp.op-m.tracker.toggle-sheet]`).
const T1_HALF_ANGLE_SCALE: f32 = 0.5;

const F32_ABS_MASK: u32 = 0x7FFF_FFFF;
const F32_EXP_MASK: u32 = 0xFF;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_BIAS: i32 = dnam_kernels::IEEE754_F32_EXP_BIAS as i32;
const F32_MAX_NORM_EXP: i32 = F32_EXP_BIAS - 1;
const SUBNORMAL_PRESCALE_BITS: i32 = 24;
const TWO_POW_24_F32: f32 =
    f32::from_bits(((F32_EXP_BIAS + SUBNORMAL_PRESCALE_BITS) as u32) << F32_EXP_SHIFT);
const TWO_POW_NEG_24_F32: f32 =
    f32::from_bits(((F32_EXP_BIAS - SUBNORMAL_PRESCALE_BITS) as u32) << F32_EXP_SHIFT);

const _: () = assert!(T1_SILENCE_SQUELCH_R2 > 0.0);
const _: () = assert!((TWO_POW_24_F32 * TWO_POW_NEG_24_F32 - 1.0).to_bits() == 0);

/// Discrete 1-bit zero-crossing toggle latch and Riemann sheet tracker (`T1`, Family 1,
/// `ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Clone, Copy, Debug)]
pub struct SubHarmonicTracker {
    pub prev_x: f32,
    pub prev_q: f32,
    pub prev_u_re: f32,
    pub prev_u_im: f32,
    pub sheet: f32,
    pub dc_pole: f32,
    pub dc_gain: f32,
    pub dc_prev_x: f32,
    pub dc_prev_y: f32,
}

impl Default for SubHarmonicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SubHarmonicTracker {
    pub const fn new() -> Self {
        Self {
            prev_x: 0.0,
            prev_q: 1.0,
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            sheet: 1.0,
            dc_pole: T1_DC_BLOCKER_5HZ_48K_POLE,
            dc_gain: T1_DC_BLOCKER_5HZ_48K_GAIN,
            dc_prev_x: 0.0,
            dc_prev_y: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_q = 1.0;
        self.prev_u_re = 1.0;
        self.prev_u_im = 0.0;
        self.sheet = 1.0;
        self.dc_prev_x = 0.0;
        self.dc_prev_y = 0.0;
    }

    /// Recomputes the `5 Hz` `T1` DC blocker pole and passband gain for `sample_rate_hz` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: u64) {
        if let Some((r_bits, g_bits)) = compute_t1_dc_blocker_coeffs_bits(sample_rate_hz as f64) {
            self.dc_pole = f32::from_bits(r_bits);
            self.dc_gain = f32::from_bits(g_bits);
        }
    }

    /// Steps the `5 Hz` `T1` high-pass DC blocker on raw input `x` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
    #[inline(always)]
    pub fn step_dc_blocker(&mut self, x: f32) -> f32 {
        step_t1_dc_blocker_state(
            self.dc_pole,
            self.dc_gain,
            &mut self.dc_prev_x,
            &mut self.dc_prev_y,
            x,
        )
    }

    /// Steps the bistable sub-harmonic tracker using rising zero-crossings.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32, hilbert_x: f32) -> f32 {
        if !x.is_finite() || !hilbert_x.is_finite() {
            self.prev_x = 0.0;
            return 0.0;
        }
        // Analytic envelope normalization via fast NR2 rsqrt with silence squelch (R^2 < 1e-12)
        let x_clean = flush_recursive_state_f32(x);
        let h_clean = flush_recursive_state_f32(hilbert_x);
        let r2 = x_clean * x_clean + h_clean * h_clean;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            self.prev_x = 0.0;
            return 0.0;
        }

        // Toggle latch on rising edge (only when above silence squelch threshold)
        if self.prev_x <= 0.0 && x_clean > 0.0 {
            self.prev_q = -self.prev_q;
        }
        self.prev_x = x_clean;

        let rho = fast_rsqrt_nr2(r2);
        let u_re = (x_clean * rho).clamp(-1.0, 1.0);
        let u_im = (h_clean * rho).clamp(-1.0, 1.0);

        // Riemann branch cut tracking: branch flips when crossing negative real axis
        if ((self.prev_u_im >= 0.0 && u_im < 0.0) || (self.prev_u_im <= 0.0 && u_im > 0.0))
            && u_re < 0.0
        {
            self.sheet = -self.sheet;
        }
        self.prev_u_re = u_re;
        self.prev_u_im = u_im;

        // Continuous half-angle representation: sqrt((1 + u_re)/2)
        let half_angle = ((1.0 + u_re) * T1_HALF_ANGLE_SCALE).max(0.0);
        let half_rho = fast_rsqrt_nr2(half_angle);
        let sub_continuous = if half_rho > 0.0 {
            half_angle * half_rho
        } else {
            0.0
        };

        T1_HALF_ANGLE_SCALE * (self.prev_q + self.sheet * sub_continuous)
    }
}

#[inline(always)]
fn rsqrt_nr2_exact_f32(value: f32) -> f32 {
    <crate::math::DefaultMathBackend as crate::math::MathBackend>::rsqrt_nr2_f32(value)
}

/// Normalizes an analytic pair $(I, Q)$ using exponent-split pre-scaling by $2^{-\min(E, 126)}$
/// (with $2^{24}$ subnormal pre-scaling) and 2-step Newton `rsqrt` (`0 FSQRT`, `0 FDIV`,
/// `ref[dsp.op-m.tracker.envelope-norm]`). Returns `None` when $I = Q = 0$ or non-finite.
#[inline(always)]
pub fn normalize_tracker_phasor_exponent_split(i_val: f32, q_val: f32) -> Option<(f32, f32, f32)> {
    if !i_val.is_finite() || !q_val.is_finite() {
        return None;
    }
    let mut x = i_val;
    let mut y = q_val;
    let max_abs = x.abs().max(y.abs());
    if (max_abs.to_bits() & F32_ABS_MASK) == 0 {
        return None;
    }
    let mut exp_bits = (max_abs.to_bits() >> F32_EXP_SHIFT) & F32_EXP_MASK;
    let mut subnormal_restore = 1.0_f32;
    if exp_bits == 0 {
        x *= TWO_POW_24_F32;
        y *= TWO_POW_24_F32;
        exp_bits = (x.abs().max(y.abs()).to_bits() >> F32_EXP_SHIFT) & F32_EXP_MASK;
        subnormal_restore = TWO_POW_NEG_24_F32;
    }
    let power = ((exp_bits as i32) - F32_EXP_BIAS).min(F32_MAX_NORM_EXP);
    let scale_down = f32::from_bits(((F32_EXP_BIAS - power) as u32) << F32_EXP_SHIFT);
    let scale_up = f32::from_bits(((F32_EXP_BIAS + power) as u32) << F32_EXP_SHIFT);
    let sx = x * scale_down;
    let sy = y * scale_down;
    let sq = sx * sx + sy * sy;
    let inv = rsqrt_nr2_exact_f32(sq);
    let mag = (sq * inv) * scale_up * subnormal_restore;
    let u_re = (sx * inv).clamp(-1.0, 1.0);
    let u_im = (sy * inv).clamp(-1.0, 1.0);
    Some((u_re, u_im, mag))
}

/// Evaluates the smooth half-angle pair `(v_cos, v_sin)` from a normalized phasor `(u_re, u_im)`
/// (`ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline(always)]
pub fn eval_tracker_half_angle_pair(u_re: f32, u_im: f32) -> (f32, f32) {
    let re = u_re.clamp(-1.0, 1.0);
    let im = u_im.clamp(-1.0, 1.0);
    let h = T1_HALF_ANGLE_SCALE * (1.0 + re.abs());
    let inv = rsqrt_nr2_exact_f32(h);
    let major = (h * inv).clamp(0.0, 1.0);
    let minor = (im.abs() * (T1_HALF_ANGLE_SCALE * inv)).clamp(0.0, 1.0);
    if re >= 0.0 {
        (major, minor)
    } else {
        (minor, major)
    }
}

/// Returns `true` if consecutive normalized phasors `prev` and `curr` cross the negative real axis
/// branch cut, holding on near-origin chords (`p . u < 0` and `|D| <= 2^-18`,
/// `ref[dsp.op-m.tracker.toggle-sheet]`), delegated to [`dnam_kernels::tracker_crosses_branch_cut_f32`].
#[inline(always)]
pub fn tracker_crosses_branch_cut(prev: (f32, f32), curr: (f32, f32)) -> bool {
    let (px, py) = prev;
    let (x, y) = curr;
    if (py >= 0.0) == (y >= 0.0) || (px >= 0.0 && x >= 0.0) {
        return false;
    }
    dnam_kernels::tracker_crosses_branch_cut_f32(px, py, x, y)
}

/// Complete `T1` Sub-Harmonic Tracker wrapping the upstream `5 Hz` [`dnam_kernels::T1DcBlockerKernel`]
/// and [`dnam_kernels::T1SubHarmonicKernel`], emitting `[w_cos, w_sin, w_square, w_quarter]` for
/// `Lite` feature bus slots `4..11` (`ref[dsp.op-m.tracker.t1-dc-blocker]`,
/// `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Clone, Copy, Debug)]
pub struct T1SubHarmonicTracker {
    dc_blocker: dnam_kernels::T1DcBlockerKernel,
    kernel: dnam_kernels::T1SubHarmonicKernel,
}

impl Default for T1SubHarmonicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicTracker {
    /// Creates a `48 kHz` [`T1SubHarmonicTracker`] in clean startup state.
    pub const fn new() -> Self {
        Self {
            dc_blocker: dnam_kernels::T1DcBlockerKernel::new(),
            kernel: dnam_kernels::T1SubHarmonicKernel::new(),
        }
    }

    /// Resets upstream `5 Hz` DC blocker delay memory and sub-harmonic tracker phase/divider state.
    pub fn reset(&mut self) {
        self.dc_blocker.reset();
        self.kernel.reset();
    }

    /// Recomputes the `5 Hz` `T1` DC blocker pole via exact 160-bit midpoint search for
    /// `sample_rate_hz`, returning `(pole_bits, gain_bits)` if supported.
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: f64) -> Option<(u32, u32)> {
        let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(sample_rate_hz)?;
        self.dc_blocker = dnam_kernels::T1DcBlockerKernel::with_pole(f32::from_bits(p_bits));
        Some((p_bits, g_bits))
    }

    /// Steps the upstream `5 Hz` `T1` DC blocker on raw input `x`.
    #[inline(always)]
    pub fn step_dc_blocker(&mut self, x: f32) -> f32 {
        self.dc_blocker.step(x)
    }

    /// Returns `(q1_toggle, q2_toggle, sheet_sign, prev_valid)` from [`dnam_kernels::T1SubHarmonicKernel`].
    #[inline(always)]
    pub const fn state(&self) -> (f32, f32, f32, bool) {
        self.kernel.state()
    }

    /// Steps the tracker from DC-blocked analytic pair `(i_raw, q_raw)`, returning
    /// `[w_cos, w_sin, w_square, w_quarter]`.
    #[inline(always)]
    pub fn step_analytic(&mut self, i_raw: f32, q_raw: f32) -> [f32; 4] {
        self.kernel.step(i_raw, q_raw)
    }

    /// Steps a stereo pair (`left`, `right`) of `T1SubHarmonicTracker` instances simultaneously
    /// with `0 ULP` parity against [`Self::step_analytic`].
    #[inline(always)]
    pub fn step_analytic_stereo(
        left: &mut Self,
        right: &mut Self,
        i_l: f32,
        q_l: f32,
        i_r: f32,
        q_r: f32,
    ) -> ([f32; 4], [f32; 4]) {
        (left.step_analytic(i_l, q_l), right.step_analytic(i_r, q_r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;

    #[test]
    fn test_subharmonic_tracker() {
        let _audio = crate::FtzDazGuard::new();
        let mut tracker = SubHarmonicTracker::default();
        tracker.reset();
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, 1.0);

        // Silence squelch: R^2 < 1e-12 must return strictly 0.0 (zero DC offset on silence)
        assert_eq!(tracker.step(0.0, 0.0), 0.0);
        assert_eq!(tracker.step(1e-7, -1e-7), 0.0);

        // Rising zero-crossing flips prev_q (+1.0 -> -1.0 -> +1.0); branch cut at u_re < 0 flips sheet
        // Step 1: prev_u_im == 0.0 -> u_im > 0.0 with u_re < 0.0 flips sheet +1.0 -> -1.0
        let _ = tracker.step(-0.5, 0.2);
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, -1.0);
        // Step 2: Cross negative real axis again (u_re < 0, u_im crosses from + to -) -> sheet flips -1.0 -> +1.0
        let _ = tracker.step(-0.5, -0.2);
        assert_eq!(tracker.sheet, 1.0);
        // Rising zero-crossing on x (-0.5 -> +0.5) -> prev_q flips to -1.0
        let _ = tracker.step(0.5, 0.2);
        assert_eq!(tracker.prev_q, -1.0);

        tracker.reconfigure_sample_rate(96_000);
        assert!(tracker.step_dc_blocker(0.25).is_finite());

        tracker.reset();
        assert_eq!(tracker.prev_x, 0.0);
        assert_eq!(tracker.prev_q, 1.0);
        assert_eq!(tracker.sheet, 1.0);

        let sub = f32::from_bits(1);
        let (sub_re, sub_im, sub_mag) =
            normalize_tracker_phasor_exponent_split(sub, sub * 2.0).unwrap();
        assert!(sub_re.is_finite() && sub_im.is_finite() && sub_mag > 0.0);
        assert!(normalize_tracker_phasor_exponent_split(f32::NAN, 1.0).is_none());
        assert!(normalize_tracker_phasor_exponent_split(1.0, f32::INFINITY).is_none());
        assert_eq!(tracker.step(f32::NAN, 0.5).to_bits(), 0);
        assert_eq!(tracker.step(f32::INFINITY, 0.5).to_bits(), 0);
        assert!(tracker.step_dc_blocker(f32::NAN).is_finite());
        assert!(tracker.step_dc_blocker(0.25).is_finite());
        assert!(!tracker_crosses_branch_cut((-1.0, 1.0e-7), (1.0, -1.0e-7)));
        let _def_trk = T1SubHarmonicTracker::default();
    }

    /// verify[dsp.op-m.tracker.t1-dc-blocker]
    /// verify[dsp.op-m.tracker.envelope-norm]
    /// verify[dsp.op-m.tracker.toggle-sheet]
    #[test]
    fn test_t1_dc_blocker_vectors_envelope_norm_and_toggle_sheet() {
        // 1. All 8 canonical T1-COEFFICIENT-VECTORS
        for &(rate, exp_pole, exp_gain) in &[
            (44_100.0_f64, 0x3F7F_D155u32, 0x3F7F_E8AAu32),
            (48_000.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (48_001.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (88_200.0, 0x3F7F_E8A9, 0x3F7F_F454),
            (96_000.0, 0x3F7F_EA8F, 0x3F7F_F548),
            (176_400.0, 0x3F7F_F454, 0x3F7F_FA2A),
            (192_000.0, 0x3F7F_F547, 0x3F7F_FAA4),
            (384_000.0, 0x3F7F_FAA3, 0x3F7F_FD52),
        ] {
            let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate).unwrap();
            assert_eq!(
                (p_bits, g_bits),
                (exp_pole, exp_gain),
                "T1 mismatch at rate={rate}: got ({p_bits:#010x}, {g_bits:#010x})"
            );
        }
        assert!(compute_t1_dc_blocker_coeffs_bits(100_000.0).is_none());

        // 2. Exponent-split envelope normalization across quiet (2^-125) and loud (100.0) amplitudes
        for &amp in &[(-125.0_f64).exp2() as f32, 1.0e-30, 1.0e-12, 0.1, 100.0] {
            for idx in 0..=64 {
                let phase = -core::f64::consts::PI + core::f64::consts::TAU * (idx as f64) / 64.0;
                let i_in = ((amp as f64) * phase.cos()) as f32;
                let q_in = ((amp as f64) * phase.sin()) as f32;
                let (ux, uy, env) = normalize_tracker_phasor_exponent_split(i_in, q_in).unwrap();
                let (v_cos, v_sin) = eval_tracker_half_angle_pair(ux, uy);
                let exp_cos = (amp as f64) * (0.5 * phase).cos().abs();
                let exp_sin = (amp as f64) * (0.5 * phase).sin().abs();
                assert!(((env * v_cos) as f64 - exp_cos).abs() <= (amp as f64) * 2.5e-5);
                assert!(((env * v_sin) as f64 - exp_sin).abs() <= (amp as f64) * 2.5e-5);
            }
        }
        assert!(normalize_tracker_phasor_exponent_split(0.0, 0.0).is_none());
        assert!(normalize_tracker_phasor_exponent_split(-0.0, 0.0).is_none());

        // 3. Branch cut crossing and T1SubHarmonicTracker silence / axis-zero handling
        assert!(tracker_crosses_branch_cut(
            (150.0_f32.to_radians().cos(), 150.0_f32.to_radians().sin()),
            (300.0_f32.to_radians().cos(), 300.0_f32.to_radians().sin())
        ));
        assert!(!tracker_crosses_branch_cut((1.0, 0.0), (-1.0, 1.0e-7)));

        let _guard = FtzDazGuard::new();
        let mut dc = T1DcBlocker::new_at_rate(48_000.0).unwrap();
        let mut trk = T1SubHarmonicTracker::new();
        assert!(trk.reconfigure_sample_rate(96_000.0).is_some());
        assert!(trk.step_dc_blocker(0.25).is_finite());
        trk.reset();
        assert_eq!(trk.step_analytic(0.0, 0.0), [0.0; 4]);
        assert!(!trk.state().3);

        let f_axis = trk.step_analytic(0.0, 1.0);
        assert!(trk.state().3 && f_axis[2] > 0.0);
        let f_rise = trk.step_analytic(1.0, 1.0);
        assert!(f_rise[2] < 0.0); // q1 toggled +1 -> -1 on rising crossing

        let d_out = dc.step(0.5);
        assert!(d_out.is_finite());
        let mut legacy = SubHarmonicTracker::default();
        let mut ker = dnam_kernels::T1SubHarmonicKernel::new();
        let mut trk_l = T1SubHarmonicTracker::new();
        let mut trk_r = T1SubHarmonicTracker::new();
        for n in 0..32 {
            let angle = (n as f32) * 0.41;
            let (x, y) = (0.5 * angle.cos(), 0.5 * angle.sin());
            let _ = legacy.step(x, y);
            let k_out = ker.step(x, y);
            assert!(k_out.iter().all(|v| v.is_finite()));
            let (sl, sr) =
                T1SubHarmonicTracker::step_analytic_stereo(&mut trk_l, &mut trk_r, x, y, y, x);
            assert_eq!(sl.map(f32::to_bits), k_out.map(f32::to_bits));
            assert!(sr.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_subharmonic_tracker_huge_inputs_and_sine_recovery() {
        let _audio = FtzDazGuard::new();
        let mut tracker = SubHarmonicTracker::new();

        for &(huge_x, huge_h) in &[
            (1.0e20_f32, 1.0e20_f32),
            (-1.0e20, 1.0e20),
            (f32::MAX, f32::MAX),
            (f32::NAN, 0.5),
            (0.5, f32::INFINITY),
        ] {
            let y = tracker.step(huge_x, huge_h);
            assert_eq!(y.to_bits(), 0);
            assert!(tracker.prev_x.is_finite());
            assert!(tracker.prev_q.is_finite());
            assert!(tracker.prev_u_re.is_finite());
            assert!(tracker.prev_u_im.is_finite());
            assert!(tracker.sheet.is_finite());
        }

        let mut saw_nonzero = false;
        for n in 0..64 {
            let angle = (n as f32) * 0.35;
            let y = tracker.step(0.5 * angle.cos(), 0.5 * angle.sin());
            assert!(y.is_finite());
            assert!(tracker.prev_x.is_finite());
            assert!(tracker.prev_q.is_finite());
            assert!(tracker.prev_u_re.is_finite() && tracker.prev_u_re.abs() <= 1.0);
            assert!(tracker.prev_u_im.is_finite() && tracker.prev_u_im.abs() <= 1.0);
            assert!(tracker.sheet.is_finite());
            if y.abs() > 0.0 {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero);
    }
}

#[cfg(test)]
mod stage1_reference_tests {
    use crate::FtzDazGuard;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    use std::hint::black_box;

    const RATE: usize = 48_000;
    const TONE_PERIOD: usize = 48;
    const WARMUP: usize = RATE;
    const OBSERVATION: usize = 4_800;
    const AMPLITUDE: f64 = 1.0 / 4_194_304.0;
    const I_BRANCH_BITS: [u32; 6] = [
        0x3d91_bee6,
        0x3eec_b8e1,
        0x3f48_de0d,
        0x3f6d_91c9,
        0x3f7a_5939,
        0x3f7e_8e9d,
    ];

    #[derive(Default)]
    struct HilbertBranch {
        state: [[f32; 2]; 6],
    }

    impl HilbertBranch {
        fn step(&mut self, mut value: f32) -> f32 {
            for (state, bits) in self.state.iter_mut().zip(I_BRANCH_BITS) {
                let coefficient = f32::from_bits(bits);
                let output = coefficient * value + state[1];
                state[1] = state[0];
                state[0] = coefficient * output - value;
                value = output;
            }
            value
        }
    }

    struct Blocker {
        pole: f32,
        gain: f32,
        previous_input: f32,
        previous_output: f32,
    }

    impl Blocker {
        fn new() -> Self {
            let pole = 0.999_345_7_f32;
            let gain = ((1.0_f64 + f64::from(pole)) * 0.5) as f32;
            Self {
                pole,
                gain,
                previous_input: 0.0,
                previous_output: 0.0,
            }
        }

        fn step(&mut self, value: f32) -> f32 {
            let output =
                self.gain * (value - self.previous_input) + self.pole * self.previous_output;
            self.previous_input = value;
            self.previous_output = output;
            output
        }
    }

    #[derive(Default)]
    struct Projection {
        real: f64,
        imag: f64,
    }

    impl Projection {
        fn add(&mut self, sample: f32, index: usize) {
            let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
            self.real += f64::from(sample) * angle.cos();
            self.imag -= f64::from(sample) * angle.sin();
        }

        fn magnitude(&self) -> f64 {
            self.real.hypot(self.imag)
        }
    }

    #[test]
    fn preconditioned_tracker_preserves_a_quiet_tone_better_than_postconditioning_one_example() {
        let _guard = FtzDazGuard::new();
        let mut before_filter = Blocker::new();
        let mut before_branch = HilbertBranch::default();
        let mut after_branch = HilbertBranch::default();
        let mut after_filter = Blocker::new();
        assert_eq!(before_filter.pole.to_bits(), 0x3f7f_d51f);
        assert_eq!(before_filter.gain.to_bits(), 0x3f7f_ea90);
        let mut input_projection = Projection::default();
        let mut before_projection = Projection::default();
        let mut after_projection = Projection::default();

        for index in 0..WARMUP + OBSERVATION {
            let angle = core::f64::consts::TAU * (index % TONE_PERIOD) as f64 / TONE_PERIOD as f64;
            let input = (0.5 + AMPLITUDE * angle.cos()) as f32;
            let before = before_branch.step(before_filter.step(input));
            let after = after_filter.step(after_branch.step(input));
            if index >= WARMUP {
                input_projection.add(input, index);
                before_projection.add(before, index);
                after_projection.add(after, index);
            }
        }

        let input = input_projection.magnitude();
        let before = before_projection.magnitude();
        let difference = (after_projection.real - before_projection.real)
            .hypot(after_projection.imag - before_projection.imag);
        assert!((before / input - 1.0).abs() < 0.01);
        assert!(difference / before > 0.2);
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn flushed_normalized_i_does_not_preserve_the_raw_detector_sign() {
        let _guard = FtzDazGuard::new();
        let squared_norm = black_box(1.0_f32);
        let mut inverse = f32::from_bits(0x5f37_59df - (squared_norm.to_bits() >> 1));
        for _ in 0..2 {
            let half_norm = black_box(0.5 * squared_norm);
            let product = black_box(black_box(half_norm * inverse) * inverse);
            inverse = black_box(inverse * black_box(1.5 - product));
        }

        let raw_i = black_box(f32::MIN_POSITIVE);
        let normalized_i = black_box(raw_i * inverse);
        let later_positive = black_box(1.0_f32);
        let earlier_negative = black_box(-1.0_f32);
        assert!(raw_i > 0.0);
        assert!(inverse < 1.0);
        assert_eq!(normalized_i.to_bits() & 0x7fff_ffff, 0);
        assert!(!(raw_i <= 0.0 && later_positive > 0.0));
        assert!(normalized_i <= 0.0 && later_positive > 0.0);
        assert!(earlier_negative <= 0.0 && raw_i > 0.0);
        assert!(!(earlier_negative <= 0.0 && normalized_i > 0.0));
    }
}
