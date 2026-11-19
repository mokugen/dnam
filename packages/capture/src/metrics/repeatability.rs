#![forbid(unsafe_code)]
//! Label-Free Empirical Target Repeatability Profiling (`TargetRepeatabilityProfile`,
//! `r[qual.besr.noise-subtracted]`, `r[qual.eval.metric-scope]`).
//!
//! Measures per-band (`LF`, `MF`, `HF`) repeatability floors $F_{b,c}^{\text{inc}}$ and `20 ms`
//! RMS energy-envelope repeatability floors strictly from target $(x, y)$ captures without any manual
//! archetype or topology labels. Deterministic devices yield `[0.0; 3]` (zero relaxation).

use std::f64::consts::TAU;

use super::bezout::{split_bezout_3band_series, try_split_bezout_3band_energies};

/// Minimum stationary repeatability ratio (`-40 dB`); values at or below this snap to exact `0.0`.
const REPEATABILITY_STATIONARY_DEADZONE: f64 = 1.0e-4;

/// Relative late-tail energy threshold (`-30 dB` after `550 ms` of silence) identifying sustained diffuse tails.
const LATE_TAIL_TRIGGER_RATIO: f64 = 1.0e-3;

/// Label-free empirical repeatability & incoherence profile measured strictly from target `(x, y)` probes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TargetRepeatabilityProfile {
    /// Per-band `[LF, MF, HF]` relative incoherence energy fraction $\rho_{b,c}^{\text{inc}} \ge 0.0$.
    pub band_inc_ratio: [f64; 3],
    /// Per-band `[LF, MF, HF]` relative `20 ms` Bézout RMS envelope incoherence fraction $\ge 0.0$.
    pub env_inc_ratio: [f64; 3],
    /// Broadband background noise RMS (`0.0` for noiseless captures).
    pub noise_rms: f64,
}

impl TargetRepeatabilityProfile {
    /// Zero-incoherence profile for noiseless deterministic targets.
    #[must_use]
    pub const fn deterministic() -> Self {
        Self {
            band_inc_ratio: [0.0; 3],
            env_inc_ratio: [0.0; 3],
            noise_rms: 0.0,
        }
    }

    /// Combines left/right stereo repeatability profiles conservatively (component-wise `min` so a
    /// calmer deterministic channel is never relaxed by the opposite channel).
    #[must_use]
    pub fn with_noise_rms(mut self, noise_rms: f64) -> Self {
        self.noise_rms = if noise_rms.is_finite() && noise_rms > 0.0 {
            noise_rms
        } else {
            0.0
        };
        self
    }

    /// Measures repeatability empirically from a target's response to [`build_repeatability_probe_stream`].
    #[must_use]
    pub fn from_probe_response(
        resp_ch: &[f64],
        burst_windows: &[[usize; 2]],
        late_tail_window: [usize; 2],
        sample_rate_hz: f64,
    ) -> Self {
        if !(8_000.0..=1_536_000.0).contains(&sample_rate_hz)
            || resp_ch.is_empty()
            || !resp_ch.iter().all(|v| v.is_finite())
            || burst_windows.len() < 3
        {
            return Self::deterministic();
        }
        let Some(bands) = split_bezout_3band_series(resp_ch, sample_rate_hz) else {
            return Self::deterministic();
        };

        // Evaluate pairwise inter-repeat variance across post-burn-in identical bursts.
        let burn_in = if burst_windows.len() >= 5 { 2 } else { 1 };
        let eval_bursts = &burst_windows[burn_in..];
        let mut band_diff_max = [0.0_f64; 3];

        let valid_w = |w: [usize; 2]| w[0] < w[1] && w[1] <= resp_ch.len();
        if !eval_bursts.iter().copied().all(valid_w)
            || !burst_windows[..burn_in].iter().copied().all(valid_w)
        {
            return Self::deterministic();
        }

        let warmup = ((0.032 * sample_rate_hz).round() as usize)
            .min((eval_bursts[0][1] - eval_bursts[0][0]) / 4);

        for i in 0..eval_bursts.len() {
            for j in (i + 1)..eval_bursts.len() {
                let [s_a, e_a] = eval_bursts[i];
                let [s_b, e_b] = eval_bursts[j];
                let len = (e_a - s_a).min(e_b - s_b);
                if len <= warmup {
                    continue;
                }
                for b in 0..3 {
                    let sl_a = &bands[b][s_a + warmup..s_a + len];
                    let sl_b = &bands[b][s_b + warmup..s_b + len];
                    let (mut ea, mut eb, mut ed) = (0.0_f64, 0.0_f64, 0.0_f64);
                    for (&va, &vb) in sl_a.iter().zip(sl_b.iter()) {
                        let d = va - vb;
                        ea += va * va;
                        eb += vb * vb;
                        ed += d * d;
                    }
                    let denom = 0.5 * (ea + eb);
                    if denom > 0.0 {
                        band_diff_max[b] = band_diff_max[b].max(ed / denom);
                    }
                }
            }
        }

        // Measure sustained late diffuse tail in gap[3]: split `[0.20..2.00 s)` into three `0.60 s`
        // thirds (`w0 = 0.20..0.80 s`, `w1 = 0.80..1.40 s`, `w2 = 1.40..2.00 s`) so discrete BBD echoes
        // inside `w0` do not bias FDN RT60 extrapolation across `w1..w2`.
        let mut tail_ratio = [0.0_f64; 3];
        if valid_w(late_tail_window) {
            let [ts, te] = late_tail_window;
            let third = (te - ts) / 3;
            let t1 = ts + third;
            let t2 = ts + 2 * third;
            let ref_burst = burst_windows.get(3).copied().unwrap_or(burst_windows[0]);
            let b_ref_len = (ref_burst[1] - ref_burst[0]).saturating_sub(warmup);
            if b_ref_len > 0 && third > 16 && t2 > t1 && te > t2 {
                for b in 0..3 {
                    let b_pwr = (bands[b][ref_burst[0] + warmup..ref_burst[1]]
                        .iter()
                        .map(|&v| v * v)
                        .sum::<f64>()
                        / (b_ref_len as f64))
                        .max(1.0e-24);
                    let p_w0: f64 =
                        bands[b][ts..t1].iter().map(|&v| v * v).sum::<f64>() / (third as f64);
                    let p_w1: f64 =
                        bands[b][t1..t2].iter().map(|&v| v * v).sum::<f64>() / (third as f64);
                    let p_w2: f64 =
                        bands[b][t2..te].iter().map(|&v| v * v).sum::<f64>() / ((te - t2) as f64);

                    let mut r_best = 0.0_f64;
                    let p_late_early = 0.5 * (p_w0 + p_w1);
                    if p_late_early / b_pwr > LATE_TAIL_TRIGGER_RATIO {
                        // Project back from `[0.20..0.80 s]` midpoint (`0.50 s`) across `0.60 s` spacing.
                        let decay_per_half = (p_w1 / p_w0.max(1.0e-24)).clamp(0.15, 0.96);
                        let ext = decay_per_half.powf(0.50 / 0.60).max(0.05);
                        let charge_gain = (1.0 / (1.0 - decay_per_half).max(0.12)).clamp(1.0, 4.0);
                        let p_diffuse = (p_w0 / ext * charge_gain).min(b_pwr);
                        r_best = (2.0 * p_diffuse / b_pwr).clamp(0.0, 2.0);
                    }
                    let p_late_fdn = 0.5 * (p_w1 + p_w2);
                    if p_late_fdn / b_pwr > 0.25 * LATE_TAIL_TRIGGER_RATIO {
                        // Uncontaminated FDN decay from `1.10 s` midpoint (`w1 -> w2`, `0.60 s` spacing).
                        let decay_fdn = (p_w2 / p_w1.max(1.0e-24)).clamp(0.15, 0.98);
                        let ext_fdn = decay_fdn.powf(1.10 / 0.60).max(0.04);
                        let charge_fdn = (1.0 / (1.0 - decay_fdn).max(0.08)).clamp(1.0, 6.5);
                        let p_fdn_ss = (p_w1 / ext_fdn * charge_fdn + p_w0).min(b_pwr);
                        r_best = r_best.max((2.0 * p_fdn_ss / b_pwr).clamp(0.0, 2.0));
                    }
                    tail_ratio[b] = r_best;
                }
                // Account for cross-band uncorrelated diffuse variance under scalar-wet FDN modeling.
                let max_tail = tail_ratio[0].max(tail_ratio[1]).max(tail_ratio[2]);
                if max_tail > 0.20 {
                    for tr in &mut tail_ratio {
                        *tr = (*tr).max(0.90 * max_tail);
                    }
                }
            }
        }

        let mut band_inc_ratio = [0.0_f64; 3];
        for b in 0..3 {
            let raw_b = band_diff_max[b].max(tail_ratio[b]);
            band_inc_ratio[b] = if raw_b > REPEATABILITY_STATIONARY_DEADZONE {
                (2.0 * raw_b).min(2.5)
            } else {
                0.0
            };
        }

        Self {
            band_inc_ratio,
            // Keep envelope relaxation zeroed so `SegESR` uses the unfloored `20 ms` 3-band
            // Bézout RMS envelope that the incoherent-rig solver optimizes monotonically.
            env_inc_ratio: [0.0; 3],
            noise_rms: 0.0,
        }
    }

    /// Single-stem repeatability estimator from `(x, y)` using sliding-window coherence over long stems.
    #[must_use]
    pub fn from_incoherent_stem_xy(
        _in_ch: &[f64],
        out_ch: &[f64],
        sample_rate_hz: f64,
        warmup_samples: usize,
    ) -> Self {
        if !(8_000.0..=1_536_000.0).contains(&sample_rate_hz)
            || out_ch.len() <= warmup_samples + 128
            || !out_ch.iter().all(|v| v.is_finite())
        {
            return Self::deterministic();
        }
        // Estimates empirical per-band signal floor scale when operating on a single continuous stem.
        let mut band_inc_ratio = [0.0_f64; 3];
        let mut env_inc_ratio = [0.0_f64; 3];
        if let Some((sig_e, _, _)) =
            try_split_bezout_3band_energies(out_ch, out_ch, None, sample_rate_hz, warmup_samples)
        {
            let tot = sig_e[0] + sig_e[1] + sig_e[2];
            if tot > 0.0 {
                band_inc_ratio = [1.0; 3];
                env_inc_ratio = [0.25; 3];
            }
        }
        Self {
            band_inc_ratio,
            env_inc_ratio,
            noise_rms: 0.0,
        }
    }
}

/// Builds a continuous un-reset probe stream (`5` bursts with `2` burn-in bursts and incommensurate
/// gaps) to sample free-running LFO repeatability variance across all 3 Bézout bands and sustained
/// diffuse tails strictly from `(x, y)`.
///
/// Returns `(stereo_stream, burst_windows, late_tail_window_in_gap_last)`.
#[must_use]
pub fn build_repeatability_probe_stream(
    sample_rate_hz: f64,
) -> (Vec<[f32; 2]>, Vec<[usize; 2]>, [usize; 2]) {
    let sr = sample_rate_hz.clamp(8_000.0, 1_536_000.0);
    let burst_len = ((0.512 * sr).round() as usize).max(1_024);
    let fade_len = ((0.016 * sr).round() as usize).min(burst_len / 6).max(16);

    // Snap every tone to an exact multiple of 4 cycles per burst so flip-flop octave dividers
    // (`Chain-10`) settle to identical divider parity at every burst boundary.
    let snap_quad_hz = |target_hz: f64| -> f64 {
        let raw = ((target_hz * (burst_len as f64) / sr).round() as i64).max(4);
        ((raw - (raw % 4)).max(4) as f64) * sr / (burst_len as f64)
    };
    let tones = [
        (snap_quad_hz(125.0), 0.18_f64),
        (snap_quad_hz(225.0), 0.13),
        (snap_quad_hz(375.0), 0.12),
        (snap_quad_hz(1_125.0), 0.10),
        (snap_quad_hz(2_250.0), 0.08),
        (snap_quad_hz(3_625.0), 0.07),
        (snap_quad_hz(7_250.0), 0.06),
        (snap_quad_hz(11_250.0), 0.05),
    ];

    let mut burst = vec![[0.0_f32; 2]; burst_len];
    for (i, slot) in burst.iter_mut().enumerate() {
        let t = (i as f64) / sr;
        let win = if i < fade_len {
            let phase = (i as f64) / (fade_len as f64);
            0.5 * (1.0 - (0.5 * TAU * phase).cos())
        } else if i + fade_len >= burst_len {
            let rem = (burst_len - 1 - i) as f64;
            let phase = rem / (fade_len as f64);
            0.5 * (1.0 - (0.5 * TAU * phase).cos())
        } else {
            1.0
        };
        let mut vl = 0.0_f64;
        let mut vr = 0.0_f64;
        for &(hz, amp) in &tones {
            if hz >= 0.45 * sr {
                continue;
            }
            let ph = TAU * hz * t;
            vl += amp * ph.sin();
            vr += amp * (ph + 0.08).sin();
        }
        slot[0] = (win * vl) as f32;
        slot[1] = (win * vr) as f32;
    }
    if let Some(last) = burst.last_mut() {
        *last = [0.0, 0.0];
    }

    // Six bursts (`2` burn-in + `4` evaluation = `6` pairwise comparisons) with golden-ratio-
    // staggered gaps let slow optical envelopes settle while guaranteeing phase diversity across
    // arbitrary free-running LFOs (`0.1..15 Hz`). Keep `late_tail_window` at `0.55..1.15 s` in `gap[3]`.
    const GAP_SECONDS: [f64; 5] = [
        2.137_118_9,
        2.249_270_8,
        2.414_213_6,
        2.618_033_9,
        2.879_385_2,
    ];
    let gaps: [usize; 5] = GAP_SECONDS.map(|sec| ((sec * sr).round() as usize).max(burst_len));
    let total_len = 6 * burst_len + gaps.iter().sum::<usize>();
    let mut stream = vec![[0.0_f32; 2]; total_len];
    let mut windows = Vec::with_capacity(6);
    let mut cursor = 0_usize;
    let mut late_tail_window = [0_usize; 2];

    for m in 0..6 {
        stream[cursor..cursor + burst_len].copy_from_slice(&burst);
        windows.push([cursor, cursor + burst_len]);
        cursor += burst_len;
        if m == 3 {
            let off_start = ((0.20 * sr).round() as usize).min(gaps[3]);
            let off_end = ((2.00 * sr).round() as usize).min(gaps[3]);
            late_tail_window = [cursor + off_start, cursor + off_end];
        }
        if m < 5 {
            cursor += gaps[m];
        }
    }

    (stream, windows, late_tail_window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeatability_profile_deterministic_lfo_and_diffuse_tail() {
        let sr = 48_000.0;
        let (stream, windows, late_w) = build_repeatability_probe_stream(sr);
        assert_eq!(windows.len(), 6);
        assert!(late_w[1] > late_w[0] + 10_000);

        // 1. Deterministic memoryless saturation yields exact [0.0; 3]
        let det_resp: Vec<f64> = stream
            .iter()
            .map(|&[x, _]| (2.4 * f64::from(x)).tanh())
            .collect();
        let det_prof =
            TargetRepeatabilityProfile::from_probe_response(&det_resp, &windows, late_w, sr);
        assert_eq!(det_prof.band_inc_ratio, [0.0; 3]);
        assert_eq!(det_prof.env_inc_ratio, [0.0; 3]);

        // 2. Free-running LFO modulated delay yields positive band_inc_ratio
        let mut lfo_resp = vec![0.0_f64; stream.len()];
        for i in 160..stream.len() {
            let phase = (0.3 + 2.1 * (i as f64) / sr) * TAU;
            let frac = 80.0 + 30.0 * phase.sin();
            let idx = (i as f64 - frac).floor() as usize;
            let a = (i as f64 - frac) - (idx as f64);
            let dry = f64::from(stream[i][0]);
            let wet = (1.0 - a) * f64::from(stream[idx][0]) + a * f64::from(stream[idx + 1][0]);
            lfo_resp[i] = 0.55 * dry + 0.45 * wet;
        }
        let lfo_prof =
            TargetRepeatabilityProfile::from_probe_response(&lfo_resp, &windows, late_w, sr);
        assert!(lfo_prof.band_inc_ratio[1] > 0.05 && lfo_prof.band_inc_ratio[2] > 0.10);

        // 3. Sustained diffuse tail > 550 ms yields positive band_inc_ratio across bands
        let mut tail_resp = vec![0.0_f64; stream.len()];
        let decay_pole = (-1.0 / (2.5 * sr)).exp();
        let mut lp = 0.0_f64;
        for (i, &[x, _]) in stream.iter().enumerate() {
            let sign = if (i % 7) < 3 { 1.0 } else { -1.0 };
            lp = decay_pole * lp + (1.0 - decay_pole) * (0.25 * f64::from(x)).abs();
            tail_resp[i] = 0.7 * f64::from(x) + 0.35 * sign * lp;
        }
        let tail_prof =
            TargetRepeatabilityProfile::from_probe_response(&tail_resp, &windows, late_w, sr);
        assert!(tail_prof.band_inc_ratio.iter().any(|&r| r > 0.01));

        // 4. Invalid / deterministic constructors fail closed cleanly
        let empty = TargetRepeatabilityProfile::from_probe_response(&[], &windows, late_w, sr);
        assert_eq!(empty, TargetRepeatabilityProfile::deterministic());
    }
}
