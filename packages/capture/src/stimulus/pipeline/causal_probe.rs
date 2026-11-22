#![forbid(unsafe_code)]
//! Causal black-box deadtime and first-arrival polarity estimation for `SONA v3.0` alignment.

/// Deterministic zero-preceded broadband black-box probe (`48 kHz`, `12,288` samples) for causal
/// deadtime and first-arrival polarity estimation without any fixture-specific metadata.
#[must_use]
pub fn build_v3_causal_probe_excitation() -> Vec<[f32; 2]> {
    use std::f64::consts::PI;
    const N: usize = 12_288;
    const SR: f64 = 48_000.0;
    let mut out = vec![[0.0_f32; 2]; N];
    for (i, slot) in out.iter_mut().enumerate().take(4_096).skip(512) {
        let k = i - 512;
        let t = k as f64 / SR;
        let burst = 0.58 * (-(k as f64) / 24.0).exp();
        let lf = 0.14 * (2.0 * PI * 110.0 * t).cos();
        let hf = 0.04 * (2.0 * PI * 733.0 * t).cos()
            + 0.02 * (2.0 * PI * 2_410.0 * t).cos()
            + 0.02 * (2.0 * PI * 6_910.0 * t).cos();
        let v = (burst + lf + hf).clamp(-0.85, 0.85) as f32;
        *slot = [v, v];
    }
    for (i, slot) in out.iter_mut().enumerate().take(8_192).skip(4_608) {
        let k = (i - 4_608) as f64 / 3_584.0;
        let f = 45.0 * (18_000.0_f64 / 45.0).powf(k);
        let ph = 2.0 * PI * f * ((i - 4_608) as f64 / SR);
        let env = if i == 4_608 { 1.0 } else { 0.85 };
        let v = (0.38 * env * (ph.sin() + 0.35 * (2.0 * PI * 90.0 * (i as f64 / SR)).sin()))
            .clamp(-0.48, 0.48) as f32;
        *slot = [v, v];
    }
    for (i, slot) in out.iter_mut().enumerate().skip(8_192) {
        let t = (i - 8_192) as f64 / SR;
        let step_gain = 0.15 + 0.20 * ((i - 8_192) / 1_024) as f64 / 3.0;
        let v = (step_gain
            * ((2.0 * PI * 85.0 * t).sin()
                + 0.6 * (2.0 * PI * 330.0 * t).sin()
                + 0.4 * (2.0 * PI * 2_500.0 * t).cos()))
        .clamp(-0.48, 0.48) as f32;
        *slot = [v, v];
    }
    out
}

fn corr_based_polarity(
    dry: &[f32],
    wet: &[f32],
    dry_onset: usize,
    lag_samples: usize,
    idle_dc: f64,
) -> bool {
    let n = dry.len().min(wet.len());
    let active_n = (n - dry_onset).saturating_sub(lag_samples);
    let (mut dlf1, mut dlf2, mut wlf1, mut wlf2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut dmid1, mut dmid2, mut wmid1, mut wmid2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut corr_lf, mut dlf_e, mut wlf_e) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut corr_mid, mut dmid_e, mut wmid_e) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut corr_raw, mut dr_e, mut wr_e) = (0.0_f64, 0.0_f64, 0.0_f64);
    for t in 0..active_n {
        let d_raw = dry[dry_onset + t];
        let w_raw = wet[dry_onset + lag_samples + t];
        let d = if d_raw.is_finite() {
            f64::from(d_raw)
        } else {
            0.0
        };
        let w = if w_raw.is_finite() {
            f64::from(w_raw) - idle_dc
        } else {
            0.0
        };
        dlf1 += 0.025 * (d - dlf1);
        dlf2 += 0.025 * (dlf1 - dlf2);
        wlf1 += 0.025 * (w - wlf1);
        wlf2 += 0.025 * (wlf1 - wlf2);
        corr_lf += dlf2 * wlf2;
        dlf_e += dlf2 * dlf2;
        wlf_e += wlf2 * wlf2;

        dmid1 += 0.075 * (d - dmid1);
        dmid2 += 0.075 * (dmid1 - dmid2);
        wmid1 += 0.075 * (w - wmid1);
        wmid2 += 0.075 * (wmid1 - wmid2);
        corr_mid += dmid2 * wmid2;
        dmid_e += dmid2 * dmid2;
        wmid_e += wmid2 * wmid2;

        corr_raw += d * w;
        dr_e += d * d;
        wr_e += w * w;
    }
    let n_lf = corr_lf / (dlf_e * wlf_e).sqrt().max(1.0e-18);
    let n_mid = corr_mid / (dmid_e * wmid_e).sqrt().max(1.0e-18);
    let n_raw = corr_raw / (dr_e * wr_e).sqrt().max(1.0e-18);
    let best_lp = if n_lf.abs() >= n_mid.abs() * 0.65 && n_lf.abs() > 1.0e-3 {
        n_lf
    } else {
        n_mid
    };
    if best_lp.abs() >= n_raw.abs() * 0.35 && best_lp.abs() > 1.0e-4 {
        best_lp < 0.0
    } else {
        n_raw < 0.0
    }
}

/// Estimates strictly causal non-negative integer deadtime (samples) and causal first-arrival
/// polarity inversion (`true` if wet is globally inverted relative to dry) from `(dry, wet)`.
#[must_use]
pub fn estimate_v3_causal_deadtime_and_polarity(dry: &[f32], wet: &[f32]) -> (f64, bool) {
    let n = dry.len().min(wet.len());
    if n < 64 {
        return (0.0, false);
    }
    let dry_peak = dry[..n]
        .iter()
        .map(|&v| {
            if v.is_finite() {
                f64::from(v.abs())
            } else {
                0.0
            }
        })
        .fold(0.0_f64, f64::max);
    if dry_peak <= 1.0e-9 {
        return (0.0, false);
    }
    let dry_onset = dry[..n]
        .iter()
        .position(|&v| v.is_finite() && f64::from(v.abs()) > dry_peak * 1.0e-4)
        .unwrap_or(n);
    if dry_onset >= n {
        return (0.0, false);
    }

    let (idle_dc, lag_samples) = if dry_onset >= 64 {
        let sum: f64 = wet[..dry_onset]
            .iter()
            .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
            .sum();
        let dc = sum / (dry_onset as f64);
        let idle_noise = wet[..dry_onset]
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    (f64::from(v) - dc).abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);
        let wet_ac_peak = wet[dry_onset..n]
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    (f64::from(v) - dc).abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);
        if wet_ac_peak <= idle_noise * 2.0 + 1.0e-8 {
            return (0.0, false);
        }
        let dc_ulp = (dc.abs() * f64::from(f32::EPSILON)).max(1.0e-12);
        let onset_thr = idle_noise * 6.0 + 4.0 * dc_ulp;
        let max_scan = n.min(dry_onset + 4_096);
        let lag = (dry_onset..max_scan)
            .find(|&t| wet[t].is_finite() && (f64::from(wet[t]) - dc).abs() > onset_thr)
            .map_or(0, |t| t - dry_onset);
        (dc, lag)
    } else {
        let sum: f64 = wet[..n]
            .iter()
            .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
            .sum();
        let dc = sum / (n as f64);
        let max_lag = (n / 4).min(512);
        let mut best_lag = 0usize;
        let mut best_abs = -1.0_f64;
        for lag in 0..=max_lag {
            let mut acc = 0.0_f64;
            for t in 0..(n - lag) {
                let d = if dry[t].is_finite() {
                    f64::from(dry[t])
                } else {
                    0.0
                };
                let w = if wet[t + lag].is_finite() {
                    f64::from(wet[t + lag]) - dc
                } else {
                    0.0
                };
                acc += d * w;
            }
            if acc.abs() > best_abs {
                best_abs = acc.abs();
                best_lag = lag;
            }
        }
        (dc, best_lag)
    };

    let lag_f64 = lag_samples as f64;
    if dry_onset >= 64 && dry_onset + lag_samples < n {
        let dry_end = n.min(dry_onset + 16);
        let dry_onset_peak = dry[dry_onset..dry_end]
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    f64::from(v.abs())
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);
        let wet_start = dry_onset + lag_samples;
        let wet_end = n.min(wet_start + 16);
        let wet_onset_peak = wet[wet_start..wet_end]
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    (f64::from(v) - idle_dc).abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);
        if dry_onset_peak > 1.0e-9 && wet_onset_peak > 1.0e-5 {
            let d_thr = 0.20 * dry_onset_peak;
            let dry_lead_sign = dry[dry_onset..dry_end]
                .iter()
                .find_map(|&v| {
                    let fv = if v.is_finite() { f64::from(v) } else { 0.0 };
                    (fv.abs() >= d_thr).then_some(if fv >= 0.0 { 1.0 } else { -1.0 })
                })
                .unwrap_or(1.0);
            let w_thr = 0.20 * wet_onset_peak;
            if let Some(w_lead) = wet[wet_start..wet_end].iter().find_map(|&v| {
                let fv = if v.is_finite() {
                    f64::from(v) - idle_dc
                } else {
                    0.0
                };
                (fv.abs() >= w_thr).then_some(fv)
            }) {
                return (lag_f64, (w_lead * dry_lead_sign) < 0.0);
            }
        }
    }

    (
        lag_f64,
        corr_based_polarity(dry, wet, dry_onset, lag_samples, idle_dc),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3_causal_deadtime_and_polarity_zero_and_delayed() {
        let probe = build_v3_causal_probe_excitation();
        let dry: Vec<f32> = probe.iter().map(|v| v[0]).collect();

        // Zero-lag minimum-phase lowpass + asymmetric tube-like waveshaper with DC offset.
        let mut lp = 0.0_f64;
        let mut zero_lag_wet = vec![0.0_f32; dry.len()];
        let mut zero_lag_inv = vec![0.0_f32; dry.len()];
        for (i, &x) in dry.iter().enumerate() {
            lp = 0.78 * lp + 0.22 * f64::from(x);
            let y = (2.2 * lp + 0.15 * f64::from(x) - 0.12).tanh() + 0.12_f64.tanh();
            zero_lag_wet[i] = y as f32;
            zero_lag_inv[i] = (-y) as f32;
        }
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &zero_lag_wet),
            (0.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &zero_lag_inv),
            (0.0, true)
        );

        // Synthetic 24-sample pure causal deadtime (+ filtered wave) in normal and inverted polarity.
        let mut del_pos = vec![0.0_f32; dry.len()];
        let mut del_neg = vec![0.0_f32; dry.len()];
        for i in 24..dry.len() {
            del_pos[i] = zero_lag_wet[i - 24];
            del_neg[i] = -zero_lag_wet[i - 24];
        }
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &del_pos),
            (24.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &del_neg),
            (24.0, true)
        );

        // 8-pole IIR + cubic noise gate with nonzero idle DC bias at lag = 0 and lag = 24.
        let mut s = [0.0_f64; 8];
        let mut gate_wet = vec![0.0_f32; dry.len()];
        let mut gate_inv = vec![0.0_f32; dry.len()];
        for (i, &x) in dry.iter().enumerate() {
            let u = f64::from(x);
            let mut v = u * u * u / (u * u + 0.03);
            for pole in &mut s {
                *pole += 0.45 * (v - *pole);
                v = *pole;
            }
            let y = (2.4 * v - 0.14).tanh();
            gate_wet[i] = y as f32;
            gate_inv[i] = (-y) as f32;
        }
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &gate_wet),
            (0.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &gate_inv),
            (0.0, true)
        );
        let mut gate_del = vec![gate_wet[0]; dry.len()];
        let mut gate_del_inv = vec![gate_inv[0]; dry.len()];
        gate_del[24..dry.len()].copy_from_slice(&gate_wet[..(dry.len() - 24)]);
        gate_del_inv[24..dry.len()].copy_from_slice(&gate_inv[..(dry.len() - 24)]);
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &gate_del),
            (24.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry, &gate_del_inv),
            (24.0, true)
        );

        // Resonant SVF BP (fc = 1100 Hz, Q = 7.0) and HP (fc = 900 Hz, Q = 4.0) with +2 OS delay.
        for (fc, q, is_hp) in [(1_100.0_f64, 7.0_f64, false), (900.0_f64, 4.0_f64, true)] {
            let g = (std::f64::consts::PI * fc / 48_000.0).tan();
            let k = 1.0 / q;
            let a1 = 1.0 / (1.0 + g * (g + k));
            let mut ic1 = 0.0_f64;
            let mut ic2 = 0.0_f64;
            let mut raw = vec![0.0_f32; dry.len()];
            for (i, &x) in dry.iter().enumerate() {
                let hp = (f64::from(x) - (k + g) * ic1 - ic2) * a1;
                let bp = g * hp + ic1;
                let lp = g * bp + ic2;
                ic1 = 2.0 * bp - ic1;
                ic2 = 2.0 * lp - ic2;
                raw[i] = if is_hp { hp } else { bp } as f32;
            }
            let mut pos = vec![0.0_f32; dry.len()];
            let mut neg = vec![0.0_f32; dry.len()];
            for i in 2..dry.len() {
                pos[i] = raw[i - 2];
                neg[i] = -raw[i - 2];
            }
            assert_eq!(
                estimate_v3_causal_deadtime_and_polarity(&dry, &pos),
                (2.0, false)
            );
            assert_eq!(
                estimate_v3_causal_deadtime_and_polarity(&dry, &neg),
                (2.0, true)
            );
        }
    }

    #[test]
    fn test_v3_alignment_early_onset_corr_polarity_and_guards() {
        use std::f64::consts::PI;

        // Short buffer and silent dry buffer guards
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&[0.1; 32], &[0.1; 32]),
            (0.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&vec![0.0; 256], &vec![0.2; 256]),
            (0.0, false)
        );

        // Delayed onset (>= 64) with silent/idle-only wet signal & NaN samples
        let mut dry_late = vec![0.0_f32; 256];
        let mut wet_idle = vec![0.0_f32; 256];
        dry_late[10] = f32::NAN;
        dry_late[80] = 0.5;
        wet_idle[10] = f32::NAN;
        wet_idle[80] = 1.0e-10;
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry_late, &wet_idle),
            (0.0, false)
        );

        // Early onset (< 64) broadband + LF cross-correlation lag & corr_based_polarity (+NaN)
        let n = 4_096_usize;
        let mut dry_early = vec![0.0_f32; n];
        for (t, dst) in dry_early.iter_mut().enumerate().skip(8) {
            let phase = (t - 8) as f64 / 48_000.0;
            *dst = (0.45 * (2.0 * PI * 95.0 * phase).sin()
                + 0.25 * (2.0 * PI * 640.0 * phase).cos()) as f32;
        }
        dry_early[5] = f32::NAN;
        dry_early[120] = f32::NAN;
        let mut wet_pos = vec![0.05_f32; n];
        let mut wet_neg = vec![0.05_f32; n];
        for t in 23..n {
            let src = dry_early[t - 23];
            let clean = if src.is_finite() { src } else { 0.0 };
            wet_pos[t] = 0.05 + 0.8 * clean;
            wet_neg[t] = 0.05 - 0.8 * clean;
        }
        wet_pos[30] = f32::NAN;
        wet_neg[30] = f32::NAN;
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry_early, &wet_pos),
            (23.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry_early, &wet_neg),
            (23.0, true)
        );

        // HF alternating burst (dry_onset < 64) where raw correlation decides polarity over LP
        let mut dry_hf = vec![0.0_f32; 128];
        let mut wet_hf_pos = vec![0.0_f32; 128];
        let mut wet_hf_neg = vec![0.0_f32; 128];
        for t in 4..100 {
            let v = if t % 2 == 0 { 0.6_f32 } else { -0.6_f32 };
            dry_hf[t] = v;
            wet_hf_pos[t + 5] = v;
            wet_hf_neg[t + 5] = -v;
        }
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry_hf, &wet_hf_pos),
            (5.0, false)
        );
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry_hf, &wet_hf_neg),
            (5.0, true)
        );

        // Mid-frequency tone (dry_onset < 64) where mid LP dominates low LP in corr_based_polarity
        let mut dry_mid = vec![0.0_f32; 512];
        let mut wet_mid_neg = vec![0.0_f32; 512];
        for t in 10..500 {
            let s = 0.5 * (2.0 * PI * 1_600.0 * (t as f64) / 48_000.0).sin() as f32;
            dry_mid[t] = s;
            wet_mid_neg[t] = -s;
        }
        assert_eq!(
            estimate_v3_causal_deadtime_and_polarity(&dry_mid, &wet_mid_neg),
            (0.0, true)
        );

        // Delayed onset (>= 64) with small wet peak (<= 1e-5) falling back to corr_based_polarity
        let mut dry_low_amp = vec![0.0_f32; 512];
        let mut wet_low_amp = vec![0.0_f32; 512];
        for t in 80..480 {
            let s = 5.0e-5_f32 * (2.0 * PI * 90.0 * (t as f64) / 48_000.0).sin() as f32;
            dry_low_amp[t] = s;
            wet_low_amp[t] = -0.1 * s;
        }
        let (lag_low, pol_low) =
            estimate_v3_causal_deadtime_and_polarity(&dry_low_amp, &wet_low_amp);
        assert_eq!(lag_low, 0.0);
        assert!(pol_low);
    }
}
