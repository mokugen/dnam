#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! `Law 0.A`-gated early/late quarter clock-drift (`ppm`) estimator
//! (`align::regimes::ppm_drift`, `ref[train.capture.alignment.multi-regime]`).

use crate::align::estimate_subsample_lag_and_gain;

use super::hybrid_parabolic::refine_fractional_lag_hybrid;

/// Minimum stem length (`24,000` samples = `0.5 s` @ `48 kHz`) required for `[0..N/4)` vs `[3N/4..N)`
/// clock-drift estimation.
pub const PPM_MIN_SAMPLES: usize = 24_000;

/// Minimum normalized cross-correlation coherence required in BOTH leading `[0..N/4)` and
/// trailing `[3N/4..N)` quarters before applying polyphase drift resampling.
pub const PPM_QUARTER_MIN_COHERENCE: f64 = 0.75;

/// Deadband floor (`|ppm| > 1.0`): drift within `[-1.0, +1.0] ppm` is skipped cleanly (`0.0`).
pub const PPM_DEADBAND_MIN: f64 = 1.0;

/// Maximum plausible hardware sample-clock drift (`|ppm| <= 250.0`).
pub const PPM_MAX_ABS: f64 = 250.0;

/// Result of `[0..N/4)` vs `[3N/4..N)` clock-skew estimation and `Law 0.A` / coherence gating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpmDriftEstimate {
    /// Un-gated raw linear drift estimate in parts-per-million (`0.0` if quarters invalid).
    pub raw_ppm: f64,
    /// Gated drift applied to polyphase resampling (`0.0` unless all spec gates pass).
    pub applied_ppm: f64,
    /// Extrapolated `n = 0` sample lag (`l_early - (raw_ppm * 1e-6) * centroid_early`) matching
    /// [`crate::align::realign_fractional_kaiser64`] (`center = n + lag_0 + drift * n`).
    pub lag_at_zero_samples: f64,
    /// Normalized correlation magnitude over the leading quarter `[0..N/4)`.
    pub early_coherence: f64,
    /// Normalized correlation magnitude over the trailing quarter `[3N/4..N)`.
    pub late_coherence: f64,
    /// True iff `law0a_ltv_flag == false`, both quarter coherences `>= 0.75`, and `1.0 < |raw_ppm| <= 250.0`.
    pub gated_applied: bool,
}

fn peak_energy_subwindow(
    dry: &[f32],
    q_start: usize,
    q_end: usize,
    max_len: usize,
) -> (usize, usize, f64) {
    let q_len = q_end.saturating_sub(q_start);
    let w = q_len.min(max_len);
    let best_s = if w >= q_len || w == 0 {
        q_start
    } else {
        let mut cur_e: f64 = dry[q_start..q_start + w]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        let mut max_e = cur_e;
        let mut max_s = q_start;
        for s in (q_start + 1)..=(q_end - w) {
            let out_v = f64::from(dry[s - 1]);
            let in_v = f64::from(dry[s + w - 1]);
            cur_e += in_v * in_v - out_v * out_v;
            if cur_e > max_e {
                max_e = cur_e;
                max_s = s;
            }
        }
        max_s
    };
    let best_e = best_s + w;
    let mut e = 0.0_f64;
    let mut num = 0.0_f64;
    for (i, &x) in dry[best_s..best_e].iter().enumerate() {
        let p = f64::from(x) * f64::from(x);
        e += p;
        num += ((best_s + i) as f64) * p;
    }
    let centroid = if e > 1.0e-18 {
        num / e
    } else {
        (best_s as f64) + 0.5 * (w as f64)
    };
    (best_s, best_e, centroid)
}

/// Estimates linear sample-clock skew (`ppm`) between `dry` and `wet` over `[0..N/4)` and
/// `[3N/4..N)`, gating polyphase resampling strictly to `law0a_ltv_flag == false`,
/// `early_coherence >= 0.75 && late_coherence >= 0.75`, and `1.0 < |ppm| <= 250.0`
/// (`ref[train.capture.alignment.multi-regime]`).
#[must_use]
pub fn estimate_ppm_clock_drift(
    dry: &[f32],
    wet: &[f32],
    max_lag_samples: usize,
    law0a_ltv_flag: bool,
) -> PpmDriftEstimate {
    let n = dry.len().min(wet.len());
    let mut out = PpmDriftEstimate {
        raw_ppm: 0.0,
        applied_ppm: 0.0,
        lag_at_zero_samples: 0.0,
        early_coherence: 0.0,
        late_coherence: 0.0,
        gated_applied: false,
    };
    if n < PPM_MIN_SAMPLES {
        return out;
    }

    let seg_len = n / 4;
    let late_start = n - seg_len;
    let (_, _, centroid_early) = peak_energy_subwindow(dry, 0, seg_len, seg_len);
    let (_, _, centroid_late) = peak_energy_subwindow(dry, late_start, n, seg_len);
    let e_early: f64 = dry[..seg_len]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    let e_late: f64 = dry[late_start..n]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    if e_early <= 1.0e-5 || e_late <= 1.0e-5 {
        return out;
    }

    let seed_len = if seg_len > 16_384 { 8_192 } else { seg_len };
    let (es, ee, c_early_seed) = peak_energy_subwindow(dry, 0, seg_len, seed_len);
    let (ls, le, c_late_seed) = peak_energy_subwindow(dry, late_start, n, seed_len);
    let max_drift_shift = ((n as f64) * PPM_MAX_ABS * 1.0e-6).ceil() as usize + 8;
    let search = max_lag_samples
        .saturating_add(max_drift_shift)
        .min(seed_len / 3)
        .max(8);
    let (Ok(c_early), Ok(c_late)) = (
        estimate_subsample_lag_and_gain(&dry[es..ee], &wet[es..ee], search),
        estimate_subsample_lag_and_gain(&dry[ls..le], &wet[ls..le], search),
    ) else {
        return out;
    };

    let (l_early, coh_e, _) =
        refine_fractional_lag_hybrid(&dry[es..ee], &wet[es..ee], c_early.total_lag_samples, None);
    let (l_late, coh_l, _) =
        refine_fractional_lag_hybrid(&dry[ls..le], &wet[ls..le], c_late.total_lag_samples, None);
    out.early_coherence = coh_e.abs().clamp(0.0, 1.0);
    out.late_coherence = coh_l.abs().clamp(0.0, 1.0);

    let seed_span = (c_late_seed - c_early_seed).max(1_000.0);
    let full_span = (centroid_late - centroid_early).max(1_000.0);
    let raw_ppm = ((l_late - l_early) / seed_span) * 1.0e6;
    if !raw_ppm.is_finite() {
        return out;
    }
    out.raw_ppm = raw_ppm;
    out.lag_at_zero_samples = (l_early - (raw_ppm * 1.0e-6) * c_early_seed).max(0.0);

    let abs_ppm = raw_ppm.abs();
    if !law0a_ltv_flag
        && abs_ppm > PPM_DEADBAND_MIN
        && abs_ppm <= PPM_MAX_ABS
        && out.early_coherence >= 0.50
        && out.late_coherence >= 0.50
    {
        let mut ppm_cur = raw_ppm;
        let mut lag_zero_cur = out.lag_at_zero_samples;
        let mut final_coh_e = out.early_coherence;
        let mut final_coh_l = out.late_coherence;
        for _ in 0..2 {
            let wet_u = crate::align::realign_fractional_kaiser64(wet, 0.0, ppm_cur, false);
            let center_int = lag_zero_cur.round() as isize;
            let best_late_seed = (center_int - 8..=center_int + 8)
                .max_by(|&a, &b| {
                    let score = |lag: isize| -> f64 {
                        let start = (-lag).max(0) as usize;
                        let end = ((seg_len as isize) - lag).min(seg_len as isize).max(0) as usize;
                        let mut dot = 0.0_f64;
                        for idx in start..end {
                            let j = (idx as isize + lag) as usize;
                            dot +=
                                f64::from(dry[late_start + idx]) * f64::from(wet_u[late_start + j]);
                        }
                        dot.abs()
                    };
                    score(a).total_cmp(&score(b))
                })
                .unwrap_or(center_int) as f64;
            let (l_e_u, ce_u, _) = refine_fractional_lag_hybrid(
                &dry[..seg_len],
                &wet_u[..seg_len],
                lag_zero_cur,
                None,
            );
            let (l_l_u, cl_u, _) = refine_fractional_lag_hybrid(
                &dry[late_start..n],
                &wet_u[late_start..n],
                best_late_seed,
                None,
            );
            final_coh_e = ce_u.abs().clamp(0.0, 1.0);
            final_coh_l = cl_u.abs().clamp(0.0, 1.0);
            let delta_ppm = ((l_l_u - l_e_u) / full_span) * 1.0e6;
            if !delta_ppm.is_finite() {
                break;
            }
            ppm_cur = (ppm_cur + delta_ppm).clamp(-2.0 * PPM_MAX_ABS, 2.0 * PPM_MAX_ABS);
            lag_zero_cur = (l_e_u - (delta_ppm * 1.0e-6) * centroid_early).max(0.0);
        }
        out.early_coherence = final_coh_e;
        out.late_coherence = final_coh_l;
        if final_coh_e >= PPM_QUARTER_MIN_COHERENCE
            && final_coh_l >= PPM_QUARTER_MIN_COHERENCE
            && ppm_cur.abs() > PPM_DEADBAND_MIN
            && ppm_cur.abs() <= PPM_MAX_ABS
        {
            out.gated_applied = true;
            out.applied_ppm = ppm_cur;
            out.lag_at_zero_samples = lag_zero_cur;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::realign_fractional_kaiser64;

    #[test]
    fn test_ppm_clock_drift_gating_deadband_coherence_ltv_and_recovery() {
        let seed = [0x71_u8; 256];
        let (dry, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 48_000, 1, 48_000,
        );

        // Short slice (< 24,000) returns zero
        let short = estimate_ppm_clock_drift(&dry[..16_000], &dry[..16_000], 64, false);
        assert_eq!(short.applied_ppm, 0.0);
        assert!(!short.gated_applied);

        // Deadband (<= 1.0 ppm) keeps applied_ppm = 0.0
        let wet_tiny = realign_fractional_kaiser64(&dry, -5.0, -0.4, false);
        let deadband = estimate_ppm_clock_drift(&dry, &wet_tiny, 64, false);
        assert!(deadband.raw_ppm.abs() <= 1.0);
        assert_eq!(deadband.applied_ppm, 0.0);
        assert!(!deadband.gated_applied);

        // +45 ppm LTI recovers accurate drift & passes coherence >= 0.75
        let wet_45 = realign_fractional_kaiser64(&dry, -18.4, -45.0, false);
        let est_45 = estimate_ppm_clock_drift(&dry, &wet_45, 128, false);
        assert!(est_45.gated_applied);
        assert!(est_45.early_coherence >= 0.75 && est_45.late_coherence >= 0.75);
        assert!((est_45.applied_ppm - 45.0).abs() < 3.0);

        // Law 0.A LTV flag blocks false clock-drift resampling on identical input
        let blocked_ltv = estimate_ppm_clock_drift(&dry, &wet_45, 128, true);
        assert!(!blocked_ltv.gated_applied);
        assert_eq!(blocked_ltv.applied_ppm, 0.0);
        assert!((blocked_ltv.raw_ppm - 45.0).abs() < 3.0);

        // Out-of-range (> 250 ppm) and borderline (+260 ppm) drifts rejected cleanly without clamping
        let wet_350 = realign_fractional_kaiser64(&dry, -10.0, -350.0, false);
        let rejected = estimate_ppm_clock_drift(&dry, &wet_350, 128, false);
        assert!(!rejected.gated_applied);
        assert_eq!(rejected.applied_ppm, 0.0);
        let wet_260 = realign_fractional_kaiser64(&dry, -10.0, -260.0, false);
        let est_260 = estimate_ppm_clock_drift(&dry, &wet_260, 128, false);
        assert!(!est_260.gated_applied);
        assert_eq!(est_260.applied_ppm, 0.0);

        // Uncorrelated noise fails >= 0.75 coherence gate
        let noise_wet: Vec<f32> = (0..dry.len())
            .map(|i| 0.2 * ((i as f32) * 1.17).sin())
            .collect();
        let low_coh = estimate_ppm_clock_drift(&dry, &noise_wet, 64, false);
        assert!(!low_coh.gated_applied);
        assert_eq!(low_coh.applied_ppm, 0.0);

        // Long SPCM-scale capture (192,000 samples > 65,536) at +45 ppm recovers drift cleanly,
        // while borderline +260 ppm is rejected cleanly without clamping at 250 ppm.
        let (dry_long, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 192_000, 1, 48_000,
        );
        let wet_long = realign_fractional_kaiser64(&dry_long, -37.38, -45.0, false);
        let long_est = estimate_ppm_clock_drift(&dry_long, &wet_long, 128, false);
        assert!(long_est.gated_applied);
        assert!(long_est.early_coherence >= 0.75 && long_est.late_coherence >= 0.75);
        assert!((long_est.applied_ppm - 45.0).abs() < 2.0);

        let wet_long_260 = realign_fractional_kaiser64(&dry_long, -10.0, -260.0, false);
        let long_260 = estimate_ppm_clock_drift(&dry_long, &wet_long_260, 128, false);
        assert!(!long_260.gated_applied);
        assert_eq!(long_260.applied_ppm, 0.0);
    }
}
