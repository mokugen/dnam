#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Causal Pre-Ringing Probe Guard & Finite AC Activity Helpers (`stimulus::pipeline::fullband::causal_guard`).

use crate::align::estimate_hybrid_multiregime_alignment;

/// Returns `true` when `wet` contains an active probe response above quiescent pre-onset noise.
#[must_use]
pub(super) fn has_causal_probe_activity(dry: &[f32], wet: &[f32]) -> bool {
    if dry.is_empty() || wet.is_empty() {
        return false;
    }
    let Some(onset_dry) = dry.iter().position(|v| v.is_finite() && v.abs() > 1.0e-4) else {
        return false;
    };
    let wet_ac_peak = finite_ac_peak(wet);
    if onset_dry < 32 {
        return wet_ac_peak > 1.0e-8;
    }
    let pre_end = (onset_dry / 2).max(16).min(wet.len());
    let pre = &wet[..pre_end];
    let (mut sum, mut cnt) = (0.0_f64, 0_usize);
    for &v in pre {
        if v.is_finite() {
            sum += f64::from(v);
            cnt += 1;
        }
    }
    if cnt == 0 {
        return false;
    }
    let dc = sum / (cnt as f64);
    let mut idle_noise = 0.0_f64;
    for &v in pre {
        if v.is_finite() {
            idle_noise = idle_noise.max((f64::from(v) - dc).abs());
        }
    }
    let dc_ulp = dc.abs() * f64::from(f32::EPSILON);
    let onset_thr = idle_noise * 6.0 + 4.0 * dc_ulp + 1.0e-8;
    wet_ac_peak > onset_thr
}

/// Computes the peak finite AC deviation `max |v - mean|` across `samples`.
#[must_use]
fn finite_ac_peak(samples: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    let mut cnt = 0_usize;
    for &v in samples {
        if v.is_finite() {
            sum += f64::from(v);
            cnt += 1;
        }
    }
    if cnt == 0 {
        return 0.0;
    }
    let mean = sum / (cnt as f64);
    let mut peak = 0.0_f64;
    for &v in samples {
        if v.is_finite() {
            peak = peak.max((f64::from(v) - mean).abs());
        }
    }
    peak
}

/// Guards `raw_causal_lag` from synthetic Kaiser-sinc pre-ringing false triggers while preserving
/// strict causal first-wavefront deadtime on IIR/all-pass/nonlinear targets (`|coh| < 0.98`).
#[must_use]
pub(super) fn refine_causal_probe_tau_coarse(dry: &[f32], wet: &[f32], raw_causal_lag: f64) -> f64 {
    let n = dry.len().min(wet.len());
    if n < 512 {
        return raw_causal_lag;
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
    let Some(dry_onset) = dry[..n]
        .iter()
        .position(|&v| v.is_finite() && f64::from(v.abs()) > dry_peak * 1.0e-4)
    else {
        return raw_causal_lag;
    };
    if dry_onset < 64 || dry_onset + 32 >= n {
        return raw_causal_lag;
    }
    let dc: f64 = wet[..dry_onset]
        .iter()
        .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
        .sum::<f64>()
        / (dry_onset as f64);
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
    if wet_ac_peak <= 1.0e-8 {
        return raw_causal_lag;
    }
    let ten_pct_thr = 0.10 * wet_ac_peak;
    let raw_lag_idx = raw_causal_lag.round().max(0.0) as usize;
    let max_scan = n.min(dry_onset + 4_096);
    let ten_pct_lag = (dry_onset..max_scan)
        .find(|&t| wet[t].is_finite() && (f64::from(wet[t]) - dc).abs() >= ten_pct_thr)
        .map_or(raw_lag_idx, |t| t - dry_onset);
    if ten_pct_lag > raw_lag_idx + 4 {
        let win = (n - dry_onset).min(4_096);
        let wet_ac: Vec<f32> = wet[..dry_onset + win]
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    (f64::from(v) - dc) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let max_lag = (ten_pct_lag + 64).min(2_048);
        if let Ok(est) = estimate_hybrid_multiregime_alignment(
            &dry[..wet_ac.len()],
            &wet_ac,
            48_000,
            max_lag,
            None,
            false,
        ) && est.linear_coherence.abs() >= 0.98
        {
            return est.total_lag_samples.round().max(0.0);
        }
    }
    raw_causal_lag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::realign_fractional_kaiser64;
    use crate::stimulus::pipeline::fullband::build_v3_causal_probe_excitation_48k;

    #[test]
    fn test_finite_ac_peak_and_causal_guard_edges() {
        assert_eq!(finite_ac_peak(&[]), 0.0);
        assert_eq!(finite_ac_peak(&[f32::NAN, f32::INFINITY]), 0.0);
        assert_eq!(finite_ac_peak(&[0.25, 0.25, 0.25]), 0.0);
        assert!((finite_ac_peak(&[1.0, -1.0, f32::NAN]) - 1.0).abs() < 1.0e-6);

        // Short, silent-dry, silent-wet, and early-onset fallback edges
        assert_eq!(
            refine_causal_probe_tau_coarse(&[0.0; 64], &[0.0; 64], 7.0),
            7.0
        );
        assert_eq!(
            refine_causal_probe_tau_coarse(&[0.0; 600], &[0.0; 600], 9.0),
            9.0
        );
        let mut early = vec![0.0_f32; 600];
        early[4] = 1.0;
        assert_eq!(refine_causal_probe_tau_coarse(&early, &early, 5.0), 5.0);

        let dry = build_v3_causal_probe_excitation_48k();
        assert_eq!(
            refine_causal_probe_tau_coarse(&dry, &vec![0.1_f32; dry.len()], 11.0),
            11.0
        );

        // Pre-ringing guard: 24-sample delayed Kaiser-64 probe passed raw_causal_lag = 4.0
        // (simulating an early 1.5% pre-ring crossing before the main impulse peak).
        let wet_delayed = realign_fractional_kaiser64(&dry, -24.0, 0.0, false);
        let guarded = refine_causal_probe_tau_coarse(&dry, &wet_delayed, 4.0);
        assert!((guarded - 24.0).abs() <= 1.0, "guarded={guarded}");

        // Activity gate rejects pure -80 dBFS idle noise, non-finite slices, constant DC, and silent dry,
        // while preserving genuine delayed probe energy above the quiescent pre-onset noise floor.
        assert!(!has_causal_probe_activity(&[], &dry));
        assert!(!has_causal_probe_activity(&dry, &[]));
        assert!(!has_causal_probe_activity(&[0.0; 600], &dry));
        assert!(!has_causal_probe_activity(&dry, &vec![f32::NAN; dry.len()]));
        assert!(!has_causal_probe_activity(&dry, &vec![0.25_f32; dry.len()]));
        assert!(has_causal_probe_activity(
            &[1.0_f32; 4],
            &[0.5_f32, -0.5_f32]
        ));
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
        let idle_noise: Vec<f32> = (0..dry.len())
            .map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng >> 32) as f64) / f64::from(u32::MAX) - 0.5;
                (2.0e-4 * u) as f32
            })
            .collect();
        assert!(!has_causal_probe_activity(&dry, &idle_noise));
        let probe_plus_noise: Vec<f32> = wet_delayed
            .iter()
            .zip(idle_noise.iter())
            .map(|(&s, &n)| 0.02 * s + n)
            .collect();
        assert!(has_causal_probe_activity(&dry, &probe_plus_noise));
    }
}
