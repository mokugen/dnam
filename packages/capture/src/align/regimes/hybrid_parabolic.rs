#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Hybrid coarse-guarded parabolic and bounded Kaiser-sinc sub-sample lag refinement
//! (`align::regimes::hybrid_parabolic`, `ref[train.capture.alignment.multi-regime]`).

use crate::align::realign_fractional_kaiser64;

/// Denominator threshold guarding flat cross-correlation plateaus (`|denom| <= 1e-12 => 0.0`).
pub const PARABOLIC_PLATEAU_EPS: f64 = 1.0e-12;

/// Maximum signed sub-sample excursion allowed around causal `tau_coarse` (`[-1.5, +1.5]` samples).
pub const HYBRID_SUBSAMPLE_MAX_DELTA: f64 = 1.5;

/// Computes the 3-point parabolic vertex offset `delta in [-0.5, +0.5]` around magnitude peak `k*`
/// (`y_m1 = |R[k*-1]|`, `y_0 = |R[k*]|`, `y_p1 = |R[k*+1]|`), returning `0.0` on flat plateaus
/// (`|2 * (y_m1 - 2 * y_0 + y_p1)| <= 1e-12`).
#[must_use]
pub fn parabolic_peak_offset(y_m1: f64, y_0: f64, y_p1: f64) -> f64 {
    if !y_m1.is_finite() || !y_0.is_finite() || !y_p1.is_finite() {
        return 0.0;
    }
    let denom = 2.0 * (y_m1 - 2.0 * y_0 + y_p1);
    if !denom.is_finite() || denom.abs() <= PARABOLIC_PLATEAU_EPS {
        return 0.0;
    }
    ((y_m1 - y_p1) / denom).clamp(-0.5, 0.5)
}

/// Constrains `k_star + delta` within `[-1.5, +1.5]` samples around causal `tau_coarse`,
/// clamped non-negative (`max(0.0, tau_coarse + clamp(k_star + delta - tau_coarse, -1.5, +1.5))`).
#[must_use]
pub fn constrain_subsample_around_coarse(k_star: f64, delta: f64, tau_coarse: f64) -> f64 {
    if !tau_coarse.is_finite() {
        let cand = if k_star.is_finite() && delta.is_finite() {
            k_star + delta
        } else if k_star.is_finite() {
            k_star
        } else {
            0.0
        };
        return cand.max(0.0);
    }
    let safe_coarse = tau_coarse.max(0.0);
    let safe_cand = if k_star.is_finite() && delta.is_finite() {
        k_star + delta
    } else {
        safe_coarse
    };
    let delta_tau =
        (safe_cand - safe_coarse).clamp(-HYBRID_SUBSAMPLE_MAX_DELTA, HYBRID_SUBSAMPLE_MAX_DELTA);
    (safe_coarse + delta_tau).max(0.0)
}

#[inline]
fn normalized_corr_at_int_lag(dry: &[f32], wet: &[f32], lag: isize) -> f64 {
    let n = dry.len().min(wet.len()) as isize;
    if n < 8 {
        return 0.0;
    }
    let start = (-lag).max(0);
    let end = (n - lag).min(n);
    if end <= start {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut e_d = 0.0_f64;
    let mut e_w = 0.0_f64;
    for i in start..end {
        let d = f64::from(dry[i as usize]);
        let w = f64::from(wet[(i + lag) as usize]);
        dot += d * w;
        e_d += d * d;
        e_w += w * w;
    }
    let denom = (e_d * e_w).sqrt();
    if denom > 1.0e-18 {
        (dot / denom).abs()
    } else {
        0.0
    }
}

pub(super) fn normalized_corr_at_fractional_lag(dry: &[f32], wet: &[f32], lag: f64) -> (f64, f64) {
    let shifted = realign_fractional_kaiser64(wet, lag, 0.0, false);
    let n = dry.len().min(shifted.len());
    let margin = ((lag.abs().ceil() as usize) + 34).min(n / 4);
    let start = margin;
    let end = n.saturating_sub(margin).max(start);
    let (slice_d, slice_w) = if end > start + 8 {
        (&dry[start..end], &shifted[start..end])
    } else {
        (&dry[..n], &shifted[..n])
    };
    let mut dot = 0.0_f64;
    let mut ex = 0.0_f64;
    let mut ey = 0.0_f64;
    for (&d, &w) in slice_d.iter().zip(slice_w) {
        let d64 = f64::from(d);
        let w64 = f64::from(w);
        dot += d64 * w64;
        ex += d64 * d64;
        ey += w64 * w64;
    }
    let denom = (ex * ey).sqrt().max(1.0e-30);
    let gain = if ex > 1.0e-30 { dot / ex } else { 0.0 };
    (dot / denom, gain)
}

/// Performs hybrid parabolic + bounded Kaiser-64 golden-section sub-sample refinement, optionally
/// guarded within `tau_coarse +/- 1.5` samples.
pub(super) fn refine_fractional_lag_hybrid(
    dry: &[f32],
    wet: &[f32],
    init_lag: f64,
    tau_coarse_opt: Option<f64>,
) -> (f64, f64, f64) {
    let tau_coarse_opt = tau_coarse_opt.filter(|v| v.is_finite());
    let anchor = if let Some(tc) = tau_coarse_opt {
        let safe_tc = tc.max(0.0);
        let bracket_lo = (safe_tc - HYBRID_SUBSAMPLE_MAX_DELTA).max(0.0);
        let bracket_hi = safe_tc + HYBRID_SUBSAMPLE_MAX_DELTA;
        if !init_lag.is_finite() || init_lag < bracket_lo || init_lag > bracket_hi {
            let mut best_lag = constrain_subsample_around_coarse(init_lag, 0.0, safe_tc);
            let mut best_coh = normalized_corr_at_fractional_lag(dry, wet, best_lag)
                .0
                .abs();
            for step in 0..=6 {
                let cand = (bracket_lo + 0.5 * (step as f64)).min(bracket_hi);
                let c_abs = normalized_corr_at_fractional_lag(dry, wet, cand).0.abs();
                if c_abs > best_coh {
                    best_coh = c_abs;
                    best_lag = cand;
                }
            }
            best_lag
        } else {
            constrain_subsample_around_coarse(init_lag, 0.0, tc)
        }
    } else {
        init_lag
    };
    let k_star = anchor.round() as isize;
    let y_m1 = normalized_corr_at_int_lag(dry, wet, k_star - 1);
    let y_0 = normalized_corr_at_int_lag(dry, wet, k_star);
    let y_p1 = normalized_corr_at_int_lag(dry, wet, k_star + 1);
    let denom = 2.0 * (y_m1 - 2.0 * y_0 + y_p1);
    let is_flat_plateau = !denom.is_finite() || denom.abs() <= PARABOLIC_PLATEAU_EPS;

    if is_flat_plateau {
        let lag = tau_coarse_opt.map_or(k_star as f64, |tc| {
            constrain_subsample_around_coarse(anchor, 0.0, tc)
        });
        let (signed_coh, signed_gain) = normalized_corr_at_fractional_lag(dry, wet, lag);
        return (lag, signed_coh, signed_gain);
    }

    let delta = parabolic_peak_offset(y_m1, y_0, y_p1);
    let parabolic_lag = tau_coarse_opt.map_or(k_star as f64 + delta, |tc| {
        constrain_subsample_around_coarse(k_star as f64, delta, tc)
    });

    let (mut a, mut b) = if let Some(tc) = tau_coarse_opt {
        let sc = tc.max(0.0);
        let lo = (parabolic_lag - 0.55)
            .max(sc - HYBRID_SUBSAMPLE_MAX_DELTA)
            .max(0.0);
        let hi = (parabolic_lag + 0.55)
            .min(sc + HYBRID_SUBSAMPLE_MAX_DELTA)
            .max(lo);
        (lo, hi)
    } else {
        (parabolic_lag - 0.55, parabolic_lag + 0.55)
    };

    const INV_PHI: f64 = 1.0 / core::f64::consts::GOLDEN_RATIO;
    const INV_PHI2: f64 = 1.0 - INV_PHI;
    let mut c = a + INV_PHI2 * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let mut fc = normalized_corr_at_fractional_lag(dry, wet, c).0.abs();
    let mut fd = normalized_corr_at_fractional_lag(dry, wet, d).0.abs();

    for _ in 0..20 {
        if fc >= fd {
            b = d;
            d = c;
            fd = fc;
            c = a + INV_PHI2 * (b - a);
            fc = normalized_corr_at_fractional_lag(dry, wet, c).0.abs();
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = normalized_corr_at_fractional_lag(dry, wet, d).0.abs();
        }
    }
    let best_lag = 0.5 * (a + b);
    let (signed_coh, signed_gain) = normalized_corr_at_fractional_lag(dry, wet, best_lag);
    (best_lag, signed_coh, signed_gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parabolic_plateau_guard_and_coarse_constraint_bounds() {
        assert_eq!(parabolic_peak_offset(0.9, 0.9, 0.9), 0.0);
        assert_eq!(parabolic_peak_offset(0.9, 0.9 + 1.0e-14, 0.9), 0.0);
        assert_eq!(parabolic_peak_offset(f64::NAN, 1.0, 0.5), 0.0);

        let d_pos = parabolic_peak_offset(0.6, 1.0, 0.8);
        assert!(d_pos > 0.0 && d_pos <= 0.5);
        let d_neg = parabolic_peak_offset(0.8, 1.0, 0.6);
        assert!((d_pos + d_neg).abs() < 1.0e-12);

        assert_eq!(constrain_subsample_around_coarse(50.0, 0.37, 25.0), 26.5);
        assert_eq!(constrain_subsample_around_coarse(10.0, 0.0, 25.0), 23.5);
        assert_eq!(constrain_subsample_around_coarse(-3.0, -0.4, 0.5), 0.0);
        assert!((constrain_subsample_around_coarse(143.0, 0.62, 143.0) - 143.62).abs() < 1.0e-12);
        assert!((constrain_subsample_around_coarse(12.0, 0.37, f64::NAN) - 12.37).abs() < 1.0e-12);
        assert_eq!(
            constrain_subsample_around_coarse(12.0, f64::NAN, f64::INFINITY),
            12.0
        );
        assert_eq!(
            constrain_subsample_around_coarse(f64::NAN, 0.37, f64::NAN),
            0.0
        );

        // Flat plateau input yields deterministic 0.0 fractional excursion
        let flat = vec![0.0_f32; 256];
        let (lag_flat, coh_flat, _) = refine_fractional_lag_hybrid(&flat, &flat, 12.0, Some(10.0));
        assert_eq!(lag_flat, 11.5);
        assert_eq!(coh_flat, 0.0);
    }

    #[test]
    fn test_refine_fractional_lag_hybrid_recovers_exact_delays_below_002_samp() {
        let seed = [0x42_u8; 256];
        let (dry, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 16_384, 1, 48_000,
        );
        for &true_lag in &[0.37_f64, 37.38_f64, 143.62_f64] {
            let delayed = realign_fractional_kaiser64(&dry, -true_lag, 0.0, false);
            let coarse_floor = true_lag.floor();
            let init = crate::align::estimate_subsample_lag_and_gain(&dry, &delayed, 256)
                .unwrap()
                .total_lag_samples;
            let (est_lag, coh, gain) =
                refine_fractional_lag_hybrid(&dry, &delayed, init, Some(coarse_floor));
            assert!(
                (est_lag - true_lag).abs() < 0.02,
                "true={true_lag} est={est_lag}"
            );
            assert!(coh > 0.98 && (gain - 1.0).abs() < 0.02);
            if (true_lag - 37.38).abs() < 1.0e-6 {
                for bad_tc in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                    let (bad_est, _, _) =
                        refine_fractional_lag_hybrid(&dry, &delayed, init, Some(bad_tc));
                    assert!(
                        bad_est.is_finite() && (bad_est - true_lag).abs() < 0.02,
                        "bad_tc={bad_tc} est={bad_est}"
                    );
                }
                let inf_align = crate::align::estimate_hybrid_multiregime_alignment(
                    &dry,
                    &delayed,
                    48_000,
                    128,
                    Some(f64::INFINITY),
                    false,
                )
                .unwrap();
                assert!(
                    inf_align.total_lag_samples.is_finite()
                        && (inf_align.total_lag_samples - true_lag).abs() < 0.02,
                    "inf_align lag={}",
                    inf_align.total_lag_samples
                );
            }
        }
    }
}
