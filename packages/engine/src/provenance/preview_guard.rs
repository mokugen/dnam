//! Dynamic Preview Probe Guard & HMAC-Keyed Acoustic Micro-Echo Watermark (`preview_guard.rs`,
//! `ref[sec.preview.containment]`, `ref[repo.code.constant-eval-and-derivations]`).
//!
//! Implements Layers 5 and 6 of the `.dnam-prev` containment pipeline (`REQ-8.7`):
//! 1. **Layer 5 (Dynamic Synthetic-Sweep & Impulse Input Guard):** Monitors `3 kHz` (`16`-sample)
//!    crest factor $\gamma_{\text{crest}}[k]$ and zero-crossing regularity $\sigma_{\text{zcr}}^2[k]$
//!    strictly without `FDIV`, `FSQRT`, `IDIV`, `libm`, or heap allocations, smoothly engaging
//!    phase flutter on synthetic probe stimuli while keeping natural musical playing untouched.
//! 2. **Layer 6 (Sparse Micro-Echo Watermark):** Keyed via `HMAC-SHA256` from the session seed to
//!    inject $K_{\text{wm}} = 4$ distinct sparse micro-echoes $y_{\text{wm}}[n] = y[n] + \sum \beta_i y[n - d_i]$
//!    with $d_i \in [120, 800]$ samples and $|\beta_i| \le 10^{-42/20}$.

use core::f32::consts::LN_10;

use dnam_kernels::{cubic_lagrange_weights_f32, q32_smooth_sin_f32};
use dnam_vm::math::{recip_nr3_pos_f32, rsqrt_nr3_f32};
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{I16, U32};

use super::security::hmac_sha256;

const TICK_LEN: usize = 16;
const INV_TICK_LEN: f32 = 1.0 / (TICK_LEN as f32);
const ZCR_SUBWIN_TICKS: usize = 4;
const M_Z: usize = 8;
const INV_M_Z: f32 = 1.0 / (M_Z as f32);

/// Number of sparse acoustic micro-echo watermark taps ($K_{\text{wm}} = 4$).
pub const MICRO_ECHO_TAPS: usize = 4;
/// Minimum allowed micro-echo tap delay in samples ($120$).
pub const MICRO_ECHO_MIN_DELAY_SAMPLES: usize = 120;
/// Maximum allowed micro-echo tap delay in samples ($800$).
pub const MICRO_ECHO_MAX_DELAY_SAMPLES: usize = 800;

const MICRO_ECHO_RING_LEN: usize = 1024;
const MICRO_ECHO_RING_MASK: usize = MICRO_ECHO_RING_LEN - 1;
const FLUTTER_RING_LEN: usize = 128;
const FLUTTER_RING_MASK: usize = FLUTTER_RING_LEN - 1;
const FLUTTER_HALF_DEPTH_SAMPLES: f32 = 10.0;
const FLUTTER_MAX_DELAY_CLAMP: f32 = 24.0;
const FLUTTER_FREQ_HZ: f32 = 37.0;
const Q32_FULL_TURN_F32: f32 = 4_294_967_296.0;
const INV_REF_SAMPLE_RATE_HZ: f32 = 1.0 / 48_000.0;

const RMS_SLOW_DECAY: f32 = 0.968_75;
const RMS_VERY_SLOW_DECAY: f32 = 0.992_187_5;
const CREST_EPS: f32 = 1.0e-5;
const CREST_IMPULSE_THRESH: f32 = 3.35;
/// Minimum within-tick squared peak-to-mean-square crest ($\text{peak}^2 / \overline{x^2}$) separating isolated Dirac spikes from smooth multi-sine audio during slow envelope rises.
const TICK_LOCAL_IMPULSE_SQ_RATIO: f32 = 8.0;
const CREST_SINE_UPPER: f32 = 1.55;
const ENV_ACTIVE_RMS_MIN: f32 = 2.0e-3;
const ENV_FLAT_REL_THRESH: f32 = 0.065;
const ENV_FLAT_MIN_STREAK: u16 = 28;
const ZCR_VAR_SYNTH_THRESH: f32 = 0.85;
const ALPHA_ENGAGE_IMPULSE: f32 = 0.22;
const ALPHA_ENGAGE_SWEEP: f32 = 0.08;
const ALPHA_RELEASE: f32 = 0.004;

const fn const_exp_neg_f64(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut n = 1;
    while n <= 20 {
        term *= -x / (n as f64);
        sum += term;
        n += 1;
    }
    sum
}

const fn const_db_to_linear_neg42() -> f32 {
    // 10^(-42/20) = 0.01 * exp(-0.1 * ln(10))
    (0.01 * const_exp_neg_f64(0.1 * (LN_10 as f64))) as f32
}

/// Exact compile-time `-42 dB` linear gain ceiling ($10^{-42/20} \approx 0.007943282$).
pub const MICRO_ECHO_MAX_ABS_GAIN: f32 = const_db_to_linear_neg42();
const MICRO_ECHO_MIN_ABS_GAIN: f32 = MICRO_ECHO_MAX_ABS_GAIN * 0.55;

const ECHO_BAND_WIDTH: u64 = 155;
const ECHO_BAND_STARTS: [usize; MICRO_ECHO_TAPS] = [120, 290, 460, 630];

const _: () = assert!(MICRO_ECHO_RING_LEN.is_power_of_two());
const _: () = assert!(FLUTTER_RING_LEN.is_power_of_two());
const _: () = assert!(M_Z.is_power_of_two());
const _: () = assert!(MICRO_ECHO_MAX_DELAY_SAMPLES < MICRO_ECHO_RING_LEN);
const _: () = assert!(MICRO_ECHO_MAX_ABS_GAIN > 0.0079 && MICRO_ECHO_MAX_ABS_GAIN < 0.0080);
const _: () = assert!(
    ECHO_BAND_STARTS[MICRO_ECHO_TAPS - 1] + (ECHO_BAND_WIDTH as usize)
        <= MICRO_ECHO_MAX_DELAY_SAMPLES
);

#[inline]
const fn bounded_range_u32(val: u32, span: u64) -> usize {
    // Division-free Lemire range reduction over 32-bit uniform digest words.
    (((val as u64) * span) >> 32) as usize
}

fn derive_echo_taps_from_seed(
    seed: &[u8; 32],
) -> ([usize; MICRO_ECHO_TAPS], [f32; MICRO_ECHO_TAPS]) {
    let digest = hmac_sha256(seed, b"DNAM-PREVIEW-MICRO-ECHO-V1");
    let mut delays = [MICRO_ECHO_MIN_DELAY_SAMPLES; MICRO_ECHO_TAPS];
    let mut gains = [0.0_f32; MICRO_ECHO_TAPS];
    let span = MICRO_ECHO_MAX_ABS_GAIN - MICRO_ECHO_MIN_ABS_GAIN;

    for i in 0..MICRO_ECHO_TAPS {
        let word = U32::read_from_prefix(&digest[i * 4..])
            .map(|(w, _)| w.get())
            .unwrap_or(0);
        delays[i] = ECHO_BAND_STARTS[i] + bounded_range_u32(word, ECHO_BAND_WIDTH);

        let mag_raw = I16::read_from_prefix(&digest[16 + i * 2..])
            .map(|(v, _)| v.get())
            .unwrap_or(1);
        let sign = if (word & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
        let frac = ((mag_raw.unsigned_abs() as f32) * (1.0 / 32_768.0)).clamp(0.0, 1.0);
        gains[i] = sign * (MICRO_ECHO_MIN_ABS_GAIN + frac * span);
    }
    (delays, gains)
}

#[inline]
fn sanitize_sample_rate_hz(sample_rate_hz: f32) -> f32 {
    if sample_rate_hz.is_finite() && (8_000.0..=384_000.0).contains(&sample_rate_hz) {
        sample_rate_hz
    } else {
        48_000.0
    }
}

/// Real-time division-free `.dnam-prev` probe guard and micro-echo watermarker (`#[forbid(unsafe_code)]`).
#[derive(Clone, Debug)]
pub struct PreviewGuardState {
    base_delays_48k: [usize; MICRO_ECHO_TAPS],
    echo_delays: [usize; MICRO_ECHO_TAPS],
    echo_gains: [f32; MICRO_ECHO_TAPS],
    sample_rate_hz: f32,
    flutter_phase_q32: u32,
    flutter_inc_q32: u32,
    g_poison: f32,
    tick_sample_idx: usize,
    tick_peak: f32,
    tick_sum_sq: f32,
    rms_sq_slow: f32,
    rms_sq_very_slow: f32,
    env_flat_streak: u16,
    prev_det_sample: f32,
    subwin_ticks: usize,
    subwin_zcr: u32,
    zcr_ring: [f32; M_Z],
    zcr_idx: usize,
    zcr_filled: usize,
    flutter_ring_l: [f32; FLUTTER_RING_LEN],
    flutter_ring_r: [f32; FLUTTER_RING_LEN],
    flutter_write_idx: usize,
    echo_ring_l: [f32; MICRO_ECHO_RING_LEN],
    echo_ring_r: [f32; MICRO_ECHO_RING_LEN],
    echo_write_idx: usize,
}

impl PreviewGuardState {
    /// Constructs a [`PreviewGuardState`] keyed deterministically by `seed` at `48 kHz`.
    #[must_use]
    pub fn new_from_seed(seed: &[u8; 32]) -> Self {
        let (base_delays_48k, echo_gains) = derive_echo_taps_from_seed(seed);
        let mut state = Self {
            base_delays_48k,
            echo_delays: base_delays_48k,
            echo_gains,
            sample_rate_hz: 48_000.0,
            flutter_phase_q32: 0,
            flutter_inc_q32: 0,
            g_poison: 0.0,
            tick_sample_idx: 0,
            tick_peak: 0.0,
            tick_sum_sq: 0.0,
            rms_sq_slow: 0.0,
            rms_sq_very_slow: 0.0,
            env_flat_streak: 0,
            prev_det_sample: 0.0,
            subwin_ticks: 0,
            subwin_zcr: 0,
            zcr_ring: [0.0; M_Z],
            zcr_idx: 0,
            zcr_filled: 0,
            flutter_ring_l: [0.0; FLUTTER_RING_LEN],
            flutter_ring_r: [0.0; FLUTTER_RING_LEN],
            flutter_write_idx: 0,
            echo_ring_l: [0.0; MICRO_ECHO_RING_LEN],
            echo_ring_r: [0.0; MICRO_ECHO_RING_LEN],
            echo_write_idx: 0,
        };
        state.prepare(48_000.0);
        state
    }

    /// Re-times flutter oscillation increment and sparse micro-echo tap delays to `sample_rate_hz`
    /// without clearing static seed keying (`0 FDIV`).
    pub fn prepare(&mut self, sample_rate_hz: f32) {
        let sr = sanitize_sample_rate_hz(sample_rate_hz);
        self.sample_rate_hz = sr;
        let inv_sr = recip_nr3_pos_f32(sr);
        let inc_f =
            (FLUTTER_FREQ_HZ * Q32_FULL_TURN_F32 * inv_sr).clamp(1.0, Q32_FULL_TURN_F32 * 0.25);
        self.flutter_inc_q32 = inc_f as u32;
        let ratio = sr * INV_REF_SAMPLE_RATE_HZ;
        let max_base = self
            .base_delays_48k
            .iter()
            .copied()
            .max()
            .unwrap_or(MICRO_ECHO_MAX_DELAY_SAMPLES);
        let eff_ratio = if ((max_base as f32) * ratio + 0.5)
            <= (MICRO_ECHO_MAX_DELAY_SAMPLES as f32)
            && ((self.base_delays_48k[1] as f32) * ratio + 0.5)
                >= ((MICRO_ECHO_MIN_DELAY_SAMPLES + 1) as f32)
        {
            ratio
        } else {
            1.0
        };
        for (dst, &base_d) in self.echo_delays.iter_mut().zip(&self.base_delays_48k) {
            let scaled = ((base_d as f32) * eff_ratio + 0.5) as usize;
            *dst = scaled.clamp(MICRO_ECHO_MIN_DELAY_SAMPLES, MICRO_ECHO_MAX_DELAY_SAMPLES);
        }
    }

    /// Clears all real-time detector state and circular delay buffers to zero equilibrium.
    pub fn reset(&mut self) {
        self.flutter_phase_q32 = 0;
        self.g_poison = 0.0;
        self.tick_sample_idx = 0;
        self.tick_peak = 0.0;
        self.tick_sum_sq = 0.0;
        self.rms_sq_slow = 0.0;
        self.rms_sq_very_slow = 0.0;
        self.env_flat_streak = 0;
        self.prev_det_sample = 0.0;
        self.subwin_ticks = 0;
        self.subwin_zcr = 0;
        self.zcr_ring.fill(0.0);
        self.zcr_idx = 0;
        self.zcr_filled = 0;
        self.flutter_ring_l.fill(0.0);
        self.flutter_ring_r.fill(0.0);
        self.flutter_write_idx = 0;
        self.echo_ring_l.fill(0.0);
        self.echo_ring_r.fill(0.0);
        self.echo_write_idx = 0;
    }

    /// Returns the smoothed Layer 5 phase-flutter poison intensity $g_{\text{poison}} \in [0.0, 1.0]$.
    #[inline]
    #[must_use]
    pub fn poison_level(&self) -> f32 {
        self.g_poison
    }

    /// Returns the configured sample-rate-adapted micro-echo delays (`[120, 800]`) and linear gains.
    #[inline]
    #[must_use]
    pub fn micro_echo_taps(&self) -> (&[usize; MICRO_ECHO_TAPS], &[f32; MICRO_ECHO_TAPS]) {
        (&self.echo_delays, &self.echo_gains)
    }

    #[inline]
    fn step_3khz_detector(&mut self) {
        let mean_sq = self.tick_sum_sq * INV_TICK_LEN;
        self.rms_sq_slow = self.rms_sq_slow * RMS_SLOW_DECAY + mean_sq * (1.0 - RMS_SLOW_DECAY);
        self.rms_sq_very_slow =
            self.rms_sq_very_slow * RMS_VERY_SLOW_DECAY + mean_sq * (1.0 - RMS_VERY_SLOW_DECAY);
        let rms_slow = if self.rms_sq_slow > 1.0e-12 {
            self.rms_sq_slow * rsqrt_nr3_f32(self.rms_sq_slow)
        } else {
            0.0
        };
        let rms_very_slow = if self.rms_sq_very_slow > 1.0e-12 {
            self.rms_sq_very_slow * rsqrt_nr3_f32(self.rms_sq_very_slow)
        } else {
            0.0
        };
        let gamma_crest = self.tick_peak * recip_nr3_pos_f32(rms_slow + CREST_EPS);
        let env_rel = (rms_slow - rms_very_slow).abs() * recip_nr3_pos_f32(rms_very_slow + 1.0e-4);
        if rms_slow > ENV_ACTIVE_RMS_MIN && env_rel < ENV_FLAT_REL_THRESH {
            self.env_flat_streak = self.env_flat_streak.saturating_add(1);
        } else {
            self.env_flat_streak = 0;
        }

        self.subwin_ticks += 1;
        if self.subwin_ticks >= ZCR_SUBWIN_TICKS {
            self.subwin_ticks = 0;
            self.zcr_ring[self.zcr_idx] = self.subwin_zcr as f32;
            self.zcr_idx = (self.zcr_idx + 1) & (M_Z - 1);
            if self.zcr_filled < M_Z {
                self.zcr_filled += 1;
            }
            self.subwin_zcr = 0;
        }

        let mut sigma_zcr_sq = ZCR_VAR_SYNTH_THRESH + 1.0;
        if self.zcr_filled == M_Z {
            let mut sum_z = 0.0_f32;
            for &z in &self.zcr_ring {
                sum_z += z;
            }
            let z_bar = sum_z * INV_M_Z;
            let mut sum_sq_dev = 0.0_f32;
            for &z in &self.zcr_ring {
                let d = z - z_bar;
                sum_sq_dev += d * d;
            }
            sigma_zcr_sq = sum_sq_dev * INV_M_Z;
        }

        let impulse_probe = self.tick_peak > ENV_ACTIVE_RMS_MIN
            && gamma_crest >= CREST_IMPULSE_THRESH
            && (self.tick_peak * self.tick_peak) >= TICK_LOCAL_IMPULSE_SQ_RATIO * mean_sq;
        let sweep_probe = self.env_flat_streak >= ENV_FLAT_MIN_STREAK
            && sigma_zcr_sq <= ZCR_VAR_SYNTH_THRESH
            && gamma_crest <= CREST_SINE_UPPER;

        if impulse_probe {
            self.g_poison += ALPHA_ENGAGE_IMPULSE * (1.0 - self.g_poison);
        } else if sweep_probe {
            self.g_poison += ALPHA_ENGAGE_SWEEP * (1.0 - self.g_poison);
        } else {
            self.g_poison += ALPHA_RELEASE * (0.0 - self.g_poison);
        }
        self.g_poison = self.g_poison.clamp(0.0, 1.0);
        self.tick_peak = 0.0;
        self.tick_sum_sq = 0.0;
    }

    /// Processes one stereo host frame through Layer 5 probe guard + Layer 6 micro-echoes
    /// (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`, `0` heap allocs).
    #[inline]
    pub fn process_sample_lr(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let l = if in_l.is_finite() { in_l } else { 0.0 };
        let r = if in_r.is_finite() { in_r } else { 0.0 };

        let peak_lr = l.abs().max(r.abs());
        if peak_lr > self.tick_peak {
            self.tick_peak = peak_lr;
        }
        self.tick_sum_sq += 0.5 * (l * l + r * r);

        let det = 0.75 * l + 0.25 * r;
        if (self.prev_det_sample < 0.0 && det >= 0.0) || (self.prev_det_sample >= 0.0 && det < 0.0)
        {
            self.subwin_zcr += 1;
        }
        self.prev_det_sample = det;

        self.tick_sample_idx += 1;
        if self.tick_sample_idx >= TICK_LEN {
            self.tick_sample_idx = 0;
            self.step_3khz_detector();
        }

        let w_fl = self.flutter_write_idx;
        self.flutter_ring_l[w_fl] = l;
        self.flutter_ring_r[w_fl] = r;

        let sin_q32 = q32_smooth_sin_f32(self.flutter_phase_q32);
        self.flutter_phase_q32 = self.flutter_phase_q32.wrapping_add(self.flutter_inc_q32);

        let (fl_l, fl_r) = if self.g_poison > 1.0e-6 {
            let delay_f = (self.g_poison * FLUTTER_HALF_DEPTH_SAMPLES * (1.0 + sin_q32))
                .clamp(0.0, FLUTTER_MAX_DELAY_CLAMP);
            let d_int = delay_f as usize;
            let mu = delay_f - (d_int as f32);
            let tap = |ring: &[f32; FLUTTER_RING_LEN], off: usize| -> f32 {
                ring[w_fl.wrapping_sub(off) & FLUTTER_RING_MASK]
            };
            if d_int >= 1 {
                let w_lag = cubic_lagrange_weights_f32(mu);
                let fl_l = w_lag[0] * tap(&self.flutter_ring_l, d_int - 1)
                    + w_lag[1] * tap(&self.flutter_ring_l, d_int)
                    + w_lag[2] * tap(&self.flutter_ring_l, d_int + 1)
                    + w_lag[3] * tap(&self.flutter_ring_l, d_int + 2);
                let fl_r = w_lag[0] * tap(&self.flutter_ring_r, d_int - 1)
                    + w_lag[1] * tap(&self.flutter_ring_r, d_int)
                    + w_lag[2] * tap(&self.flutter_ring_r, d_int + 1)
                    + w_lag[3] * tap(&self.flutter_ring_r, d_int + 2);
                (fl_l, fl_r)
            } else {
                let s0_l = tap(&self.flutter_ring_l, 0);
                let s1_l = tap(&self.flutter_ring_l, 1);
                let s0_r = tap(&self.flutter_ring_r, 0);
                let s1_r = tap(&self.flutter_ring_r, 1);
                (s0_l + mu * (s1_l - s0_l), s0_r + mu * (s1_r - s0_r))
            }
        } else {
            (l, r)
        };
        self.flutter_write_idx = (w_fl + 1) & FLUTTER_RING_MASK;

        let w_echo = self.echo_write_idx;
        self.echo_ring_l[w_echo] = fl_l;
        self.echo_ring_r[w_echo] = fl_r;

        let mut wm_l = fl_l;
        let mut wm_r = fl_r;
        for i in 0..MICRO_ECHO_TAPS {
            let tap_idx = w_echo.wrapping_sub(self.echo_delays[i]) & MICRO_ECHO_RING_MASK;
            let beta = self.echo_gains[i];
            wm_l += beta * self.echo_ring_l[tap_idx];
            wm_r += beta * self.echo_ring_r[tap_idx];
        }
        self.echo_write_idx = (w_echo + 1) & MICRO_ECHO_RING_MASK;
        (wm_l, wm_r)
    }

    /// Processes stereo audio buffers in-place through Layer 5 probe guard + Layer 6 micro-echoes.
    pub fn process_block_in_place(&mut self, left: &mut [f32], right: &mut [f32]) {
        for (l_slot, r_slot) in left.iter_mut().zip(right.iter_mut()) {
            let (out_l, out_r) = self.process_sample_lr(*l_slot, *r_slot);
            *l_slot = out_l;
            *r_slot = out_r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_preview_guard_probe_engagement_vs_natural_music_and_micro_echo_bounds() {
        let seed_a = [0x4Au8; 32];
        let seed_b = [0x9Bu8; 32];
        let mut guard = PreviewGuardState::new_from_seed(&seed_a);
        let (delays_a, gains_a) = guard.micro_echo_taps();
        let delays_a = *delays_a;
        let gains_a = *gains_a;
        let guard_b = PreviewGuardState::new_from_seed(&seed_b);
        assert_ne!(delays_a, *guard_b.micro_echo_taps().0);

        // Verify -42 dB ceiling and [120, 800] sample delay bounds.
        for (&d, &g) in delays_a.iter().zip(&gains_a) {
            assert!((MICRO_ECHO_MIN_DELAY_SAMPLES..=MICRO_ECHO_MAX_DELAY_SAMPLES).contains(&d));
            assert!(g.abs() <= MICRO_ECHO_MAX_ABS_GAIN + 1e-9);
            assert!(20.0 * g.abs().log10() <= -42.0 + 1e-4);
        }

        // 1. Silence leaves g_poison at 0.0 and outputs strictly 0.0.
        for _ in 0..1_024 {
            assert_eq!(guard.process_sample_lr(0.0, 0.0), (0.0, 0.0));
        }
        assert_eq!(guard.poison_level(), 0.0);

        // 2. Micro-echo impulse response verification after flush.
        let (ol, or) = guard.process_sample_lr(1.0, -1.0);
        assert!((ol - 1.0).abs() < 1e-6);
        assert!((or + 1.0).abs() < 1e-6);
        for n in 1..=MICRO_ECHO_MAX_DELAY_SAMPLES {
            let (el, _) = guard.process_sample_lr(0.0, 0.0);
            if let Some(pos) = delays_a.iter().position(|&d| d == n) {
                assert!((el - gains_a[pos]).abs() < 1e-6);
            } else {
                assert_eq!(el, 0.0);
            }
        }
        guard.reset();
        assert_eq!(guard.poison_level(), 0.0);

        // 3. Natural dynamic multi-harmonic music must stay unengaged.
        let two_pi_48k = core::f32::consts::TAU / 48_000.0;
        for n in 0..8_192 {
            let t = n as f32;
            let env = 0.15
                + 0.55 * (1.0 + (two_pi_48k * 6.5 * t).sin()) * 0.5
                + 0.25 * (two_pi_48k * 17.0 * t).cos();
            let vibrato = 12.0 * (two_pi_48k * 5.2 * t).sin();
            let sig_l = env
                * (0.45 * (two_pi_48k * (146.8 + vibrato) * t).sin()
                    + 0.30 * (two_pi_48k * 220.0 * t + 0.4).sin()
                    + 0.25 * (two_pi_48k * 369.99 * t + 1.1).sin());
            let sig_r = env
                * (0.40 * (two_pi_48k * (146.8 + vibrato) * t + 0.3).sin()
                    + 0.35 * (two_pi_48k * 293.66 * t + 0.9).sin());
            let _ = guard.process_sample_lr(sig_l, sig_r);
        }
        assert!(
            guard.poison_level() < 0.02,
            "natural music unexpectedly engaged guard: {}",
            guard.poison_level()
        );

        // 4. Impulse train probe engages guard cleanly.
        guard.reset();
        for n in 0..4_096 {
            let s = if (n % 512) == 100 { 0.85 } else { 0.0 };
            let _ = guard.process_sample_lr(s, s);
        }
        assert!(
            guard.poison_level() > 0.20,
            "impulse probe failed to engage guard: {}",
            guard.poison_level()
        );

        // 5. Log sine sweep probe engages guard cleanly.
        guard.reset();
        let mut phase = 0.0_f64;
        for n in 0..8_192 {
            let frac = (n as f64) / 8_192.0;
            let f_hz = 200.0 * (16.0_f64).powf(frac);
            phase += core::f64::consts::TAU * f_hz / 48_000.0;
            let s = 0.5 * (phase.sin() as f32);
            let _ = guard.process_sample_lr(s, s);
        }
        assert!(
            guard.poison_level() > 0.20,
            "sine sweep probe failed to engage guard: {}",
            guard.poison_level()
        );
    }

    #[test]
    fn test_preview_guard_prepare_rates_and_block_parity() {
        let seed = [0x33u8; 32];
        let mut g_sample = PreviewGuardState::new_from_seed(&seed);
        let mut g_block = PreviewGuardState::new_from_seed(&seed);

        for sr in [
            8_000.0_f32,
            44_100.0,
            48_000.0,
            96_000.0,
            192_000.0,
            384_000.0,
        ] {
            g_sample.prepare(sr);
            g_block.prepare(sr);
            let delays = g_sample.micro_echo_taps().0;
            for &d in delays {
                assert!(
                    (MICRO_ECHO_MIN_DELAY_SAMPLES..=MICRO_ECHO_MAX_DELAY_SAMPLES).contains(&d),
                    "sr={sr} delay {d} out of bounds"
                );
            }
            for w in delays.windows(2) {
                assert!(w[0] < w[1], "sr={sr} collapsed micro-echo taps: {delays:?}");
            }
            for i in 0..MICRO_ECHO_TAPS {
                for j in (i + 1)..MICRO_ECHO_TAPS {
                    assert_ne!(
                        delays[i], delays[j],
                        "sr={sr} duplicate delay in {delays:?}"
                    );
                }
            }
        }

        // Invalid sample rate safely falls back to 48 kHz.
        g_sample.prepare(f32::NAN);
        g_block.prepare(0.0);
        assert_eq!(g_sample.micro_echo_taps().0, g_block.micro_echo_taps().0);

        let mut left = alloc::vec![0.0_f32; 256];
        let mut right = alloc::vec![0.0_f32; 256];
        for i in 0..256 {
            let x = 0.4 * ((i as f32) * 0.13).sin();
            left[i] = x;
            right[i] = -0.5 * x;
            let (sl, sr) = g_sample.process_sample_lr(left[i], right[i]);
            let mut b_l = [left[i]];
            let mut b_r = [right[i]];
            g_block.process_block_in_place(&mut b_l, &mut b_r);
            assert_eq!(sl.to_bits(), b_l[0].to_bits());
            assert_eq!(sr.to_bits(), b_r[0].to_bits());
        }
    }

    proptest! {
        #[test]
        fn prop_preview_guard_micro_echo_bounds_and_finite_output(
            seed in any::<[u8; 32]>(),
            amp in 0.05_f32..0.9_f32,
        ) {
            let mut guard = PreviewGuardState::new_from_seed(&seed);
            for sr in [44_100.0_f32, 48_000.0, 96_000.0, 192_000.0, 384_000.0] {
                guard.prepare(sr);
                let (delays, gains) = guard.micro_echo_taps();
                for w in delays.windows(2) {
                    prop_assert!(w[0] < w[1]);
                }
                for (&d, &g) in delays.iter().zip(gains) {
                    prop_assert!(
                        (MICRO_ECHO_MIN_DELAY_SAMPLES..=MICRO_ECHO_MAX_DELAY_SAMPLES).contains(&d)
                    );
                    prop_assert!(g.abs() <= MICRO_ECHO_MAX_ABS_GAIN + 1e-9);
                }
            }
            for n in 0..256 {
                let x = amp * (((n as f32) * 0.17).sin());
                let (ol, or) = guard.process_sample_lr(x, -x);
                prop_assert!(ol.is_finite() && or.is_finite());
            }
        }
    }
}
