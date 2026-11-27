#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Causal onset guard preventing minimum-phase or dispersive group delay from pulling aligned
//! lag past the causal leading edge (`tau_coarse`) into negative time
//! (`ref[train.capture.alignment.multi-regime]`, `ref[train.capture.invariants]`).

use crate::align::realign_fractional_kaiser64;

use super::HYBRID_SUBSAMPLE_MAX_DELTA;

const CAUSAL_FUTURE_TAPS: usize = 4;
const CAUSAL_PAST_TAPS: usize = 32;
const CAUSAL_FIR_TAPS: usize = CAUSAL_FUTURE_TAPS + CAUSAL_PAST_TAPS;
const KAISER64_HALF_SPAN: usize = 32;
const ACAUSAL_ENERGY_RATIO_LIMIT: f64 = 1.0e-3;
const ACAUSAL_RELATIVE_GROWTH_FACTOR: f64 = 2.5;
const ACAUSAL_MIN_R2: f64 = 0.25;
const CAUSAL_CHECK_WINDOW: usize = 8_192;
const CAUSAL_CHECK_RIDGE_REL: f64 = 1.0e-5;

const _: () = assert!(CAUSAL_FUTURE_TAPS >= 2);
const _: () = assert!(CAUSAL_PAST_TAPS >= 16);
const _: () = assert!(CAUSAL_FIR_TAPS == CAUSAL_FUTURE_TAPS + CAUSAL_PAST_TAPS);
const _: () = assert!(KAISER64_HALF_SPAN == 32);
const _: () = assert!(ACAUSAL_ENERGY_RATIO_LIMIT > 0.0 && ACAUSAL_ENERGY_RATIO_LIMIT < 1.0e-2);
const _: () = assert!(ACAUSAL_RELATIVE_GROWTH_FACTOR > 1.0);
const _: () = assert!(ACAUSAL_MIN_R2 > 0.0 && ACAUSAL_MIN_R2 < 1.0);
const _: () = assert!(CAUSAL_CHECK_WINDOW >= 1_024);
const _: () = assert!(CAUSAL_CHECK_RIDGE_REL > 0.0 && CAUSAL_CHECK_RIDGE_REL < 1.0e-2);

fn solve_cholesky_fir(
    mut a: [[f64; CAUSAL_FIR_TAPS]; CAUSAL_FIR_TAPS],
    b: [f64; CAUSAL_FIR_TAPS],
) -> Option<[f64; CAUSAL_FIR_TAPS]> {
    for i in 0..CAUSAL_FIR_TAPS {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= a[i][k] * a[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1.0e-18 {
                    return None;
                }
                a[i][j] = sum.sqrt();
            } else {
                a[i][j] = sum / a[j][j];
            }
        }
    }

    let mut y = [0.0_f64; CAUSAL_FIR_TAPS];
    for i in 0..CAUSAL_FIR_TAPS {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i][k] * y[k];
        }
        y[i] = sum / a[i][i];
    }

    let mut x = [0.0_f64; CAUSAL_FIR_TAPS];
    for i in (0..CAUSAL_FIR_TAPS).rev() {
        let mut sum = y[i];
        for k in (i + 1)..CAUSAL_FIR_TAPS {
            sum -= a[k][i] * x[k];
        }
        x[i] = sum / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

fn highest_energy_window_start(dry: &[f32], win_len: usize) -> usize {
    let n = dry.len();
    if n <= win_len {
        return 0;
    }
    let slack = n - win_len;
    let mut best_s = 0_usize;
    let mut best_e = -1.0_f64;
    for k in 0..8 {
        let s = (k * slack) / 7;
        let energy: f64 = dry[s..s + win_len]
            .iter()
            .map(|&v| {
                let d = f64::from(v);
                d * d
            })
            .sum();
        if energy.is_finite() && energy > best_e {
            best_e = energy;
            best_s = s;
        }
    }
    best_s
}

/// Fits a `36`-tap (`k in -4..=+31`) linear FIR predicting aligned `wet` from `dry`, returning the
/// fraction of FIR tap energy strictly in anti-causal taps (`k in -4..-1`) when the linear model
/// explains at least `ACAUSAL_MIN_R2` of variance.
fn acausal_fir_energy_ratio(dry: &[f32], wet: &[f32], lag: f64) -> Option<f64> {
    let n = dry.len().min(wet.len());
    if n < 256 || !lag.is_finite() {
        return None;
    }
    let win_len = n.min(CAUSAL_CHECK_WINDOW);
    let win_s = highest_energy_window_start(&dry[..n], win_len);
    let lag_ceil = lag.abs().ceil() as usize;
    let margin = (lag_ceil + KAISER64_HALF_SPAN + CAUSAL_FUTURE_TAPS + 4).min(n / 4);
    let ext_s = win_s.saturating_sub(margin);
    let ext_e = (win_s + win_len + margin).min(n);
    let wet_aligned = realign_fractional_kaiser64(&wet[ext_s..ext_e], lag, 0.0, false);
    let off = win_s - ext_s;

    let interior_cap = (ext_e - win_s).saturating_sub(lag_ceil + KAISER64_HALF_SPAN + 1);
    let m = if interior_cap >= 256 {
        win_len.min(interior_cap)
    } else {
        win_len.min(wet_aligned.len().saturating_sub(off))
    };
    if m < CAUSAL_PAST_TAPS + CAUSAL_FUTURE_TAPS + 64 {
        return None;
    }

    let t_lo = CAUSAL_PAST_TAPS - 1;
    let t_hi = m - CAUSAL_FUTURE_TAPS;
    let cnt = (t_hi - t_lo) as f64;
    let mut sum_w = 0.0_f64;
    let mut sum_d = 0.0_f64;
    for rel_t in t_lo..t_hi {
        sum_w += f64::from(wet_aligned[off + rel_t]);
        sum_d += f64::from(dry[win_s + rel_t]);
    }
    let mean_w = sum_w / cnt;
    let mean_d = sum_d / cnt;

    let mut gram = [[0.0_f64; CAUSAL_FIR_TAPS]; CAUSAL_FIR_TAPS];
    let mut rhs = [0.0_f64; CAUSAL_FIR_TAPS];
    for rel_t in t_lo..t_hi {
        let d_idx = win_s + rel_t;
        let w_val = f64::from(wet_aligned[off + rel_t]) - mean_w;
        let mut x = [0.0_f64; CAUSAL_FIR_TAPS];
        for (j, slot) in x.iter_mut().enumerate() {
            // j in 0..4 => k = -4..-1 (future); j in 4..36 => k = 0..31 (causal past)
            let k = (j as isize) - (CAUSAL_FUTURE_TAPS as isize);
            let idx = (d_idx as isize - k) as usize;
            *slot = f64::from(dry[idx]) - mean_d;
        }
        for i in 0..CAUSAL_FIR_TAPS {
            rhs[i] += x[i] * w_val;
            for j in 0..=i {
                gram[i][j] += x[i] * x[j];
            }
        }
    }
    for i in 0..CAUSAL_FIR_TAPS {
        for j in (i + 1)..CAUSAL_FIR_TAPS {
            gram[i][j] = gram[j][i];
        }
    }

    let diag_mean =
        (0..CAUSAL_FIR_TAPS).map(|i| gram[i][i]).sum::<f64>() / (CAUSAL_FIR_TAPS as f64);
    if !diag_mean.is_finite() || diag_mean <= 1.0e-18 {
        return None;
    }
    let ridge = (CAUSAL_CHECK_RIDGE_REL * diag_mean).max(1.0e-12);
    for (i, row) in gram.iter_mut().enumerate() {
        row[i] += ridge;
    }
    let h = solve_cholesky_fir(gram, rhs)?;

    let (mut w_sq, mut err_sq) = (0.0_f64, 0.0_f64);
    for rel_t in t_lo..t_hi {
        let d_idx = win_s + rel_t;
        let w_val = f64::from(wet_aligned[off + rel_t]) - mean_w;
        let mut pred = 0.0_f64;
        for (j, &hj) in h.iter().enumerate() {
            let k = (j as isize) - (CAUSAL_FUTURE_TAPS as isize);
            let idx = (d_idx as isize - k) as usize;
            pred += hj * (f64::from(dry[idx]) - mean_d);
        }
        let err = w_val - pred;
        w_sq += w_val * w_val;
        err_sq += err * err;
    }
    let r2 = if w_sq > 1.0e-18 {
        1.0 - err_sq / w_sq
    } else {
        0.0
    };
    let e_acausal: f64 = h[..CAUSAL_FUTURE_TAPS].iter().map(|v| v * v).sum();
    let e_total: f64 = h.iter().map(|v| v * v).sum();
    (e_total > 1.0e-18 && r2 > ACAUSAL_MIN_R2).then_some(e_acausal / e_total)
}

/// Guards `cand_lag` against advancing `wet` past its causal leading edge `tau_coarse_opt`
/// due to minimum-phase filter/oversampler group delay
/// (`ref[train.capture.alignment.multi-regime]`, `ref[train.capture.invariants]`).
pub(super) fn guard_lag_causal_onset(
    dry: &[f32],
    wet: &[f32],
    cand_lag: f64,
    unconstrained_lag: f64,
    tau_coarse_opt: Option<f64>,
) -> f64 {
    let Some(tc_raw) = tau_coarse_opt.filter(|v| v.is_finite()) else {
        return cand_lag;
    };
    let tc = tc_raw.max(0.0);
    if !cand_lag.is_finite() {
        return tc;
    }
    if cand_lag <= tc + 1.0e-6 {
        return cand_lag.max(0.0);
    }
    if unconstrained_lag.is_finite() && unconstrained_lag > tc + HYBRID_SUBSAMPLE_MAX_DELTA {
        return tc;
    }
    if cand_lag > tc + 0.05
        && let Some(r_cand) = acausal_fir_energy_ratio(dry, wet, cand_lag)
        && r_cand > ACAUSAL_ENERGY_RATIO_LIMIT
    {
        let r_tc = acausal_fir_energy_ratio(dry, wet, tc).unwrap_or(0.0);
        if r_cand > ACAUSAL_RELATIVE_GROWTH_FACTOR * r_tc {
            return tc;
        }
    }
    cand_lag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::regimes::hybrid_parabolic::refine_fractional_lag_hybrid;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    #[test]
    fn test_causal_guard_preserves_fractional_delays_and_blocks_minimum_phase_advance() {
        let seed = [0x51_u8; 256];
        let (dry, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 16_384, 1, 48_000,
        );

        // Edge guards: None / non-finite tau_coarse_opt passes cand_lag through; non-finite cand_lag clamps to tc.
        assert_eq!(guard_lag_causal_onset(&dry, &dry, 3.25, 9.0, None), 3.25);
        assert_eq!(
            guard_lag_causal_onset(&dry, &dry, 3.25, 9.0, Some(f64::NAN)),
            3.25
        );
        assert_eq!(
            guard_lag_causal_onset(&dry, &dry, f64::INFINITY, 10.0, Some(24.0)),
            24.0
        );
        assert_eq!(
            guard_lag_causal_onset(&dry, &dry, f64::NAN, 10.0, Some(-3.0)),
            0.0
        );
        assert_eq!(
            guard_lag_causal_onset(&dry, &dry, -2.5, 10.0, Some(1.0)),
            0.0
        );
        assert_eq!(
            guard_lag_causal_onset(&dry, &dry, 12.25, 30.0, Some(15.0)),
            12.25
        );

        // 1. Pure fractional delays (+ optional -50 dBFS hum+hiss) pass unclipped.
        for &true_lag in &[0.37_f64, 37.38_f64, 143.62_f64] {
            let delayed = realign_fractional_kaiser64(&dry, -true_lag, 0.0, false);
            let tc = true_lag.floor();
            let (cand_lag, _, _) = refine_fractional_lag_hybrid(&dry, &delayed, true_lag, Some(tc));
            let guarded = guard_lag_causal_onset(&dry, &delayed, cand_lag, true_lag, Some(tc));
            assert!(
                (guarded - true_lag).abs() < 0.02,
                "true={true_lag} cand={cand_lag} guarded={guarded}"
            );
        }

        let hum_amp = 10.0_f64.powf(-53.0 / 20.0) * 2.0_f64.sqrt();
        let delayed_hum: Vec<f32> = realign_fractional_kaiser64(&dry, -37.38, 0.0, false)
            .into_iter()
            .enumerate()
            .map(|(i, w)| {
                let hum = hum_amp * (2.0 * PI * 60.0 * (i as f64) / 48_000.0).sin();
                (f64::from(w) + hum) as f32
            })
            .collect();
        let (hum_cand, _, _) = refine_fractional_lag_hybrid(&dry, &delayed_hum, 37.38, Some(37.0));
        let hum_guarded = guard_lag_causal_onset(&dry, &delayed_hum, hum_cand, 37.38, Some(37.0));
        assert!(
            (hum_guarded - 37.38).abs() < 0.02,
            "hum_guarded={hum_guarded}"
        );

        // 2. Causal 1-pole LPF (fc = 2200 Hz, tc = 0.0) triggers Guard 2 and stays at 0.0.
        let a_lp = 1.0 - (-2.0 * PI * 2_200.0 / 48_000.0).exp();
        let mut lp_st = 0.0_f64;
        let lpf_wet: Vec<f32> = dry
            .iter()
            .map(|&x| {
                lp_st += a_lp * (f64::from(x) - lp_st);
                lp_st as f32
            })
            .collect();
        let (lpf_cand, _, _) = refine_fractional_lag_hybrid(&dry, &lpf_wet, 0.8, Some(0.0));
        let lpf_guarded = guard_lag_causal_onset(&dry, &lpf_wet, lpf_cand, lpf_cand, Some(0.0));
        assert_eq!(lpf_guarded, 0.0);

        // 3. Cascaded LP + HPF + tanh + LPF (Guard 2 sub-1.5 and Guard 1 >1.5, tc = 0.0 or 24.0).
        let a_hp = 1.0 - (-2.0 * PI * 720.0 / 48_000.0).exp();
        let (mut lp1, mut lp2, mut lp3, mut lp4, mut hp, mut post) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut sh_lp1, mut sh_lp2, mut sh_hp, mut sh_post) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut nl_shallow = Vec::with_capacity(dry.len());
        let mut nl_wet = Vec::with_capacity(dry.len());
        for &x in &dry {
            let xf = f64::from(x);
            sh_lp1 += 0.52 * (xf - sh_lp1);
            sh_lp2 += 0.52 * (sh_lp1 - sh_lp2);
            sh_hp += a_hp * (sh_lp2 - sh_hp);
            let sh_sat = (3.5 * (sh_lp2 - sh_hp)).tanh();
            sh_post += a_lp * (sh_sat - sh_post);
            nl_shallow.push(sh_post as f32);

            lp1 += 0.42 * (xf - lp1);
            lp2 += 0.42 * (lp1 - lp2);
            lp3 += 0.42 * (lp2 - lp3);
            lp4 += 0.42 * (lp3 - lp4);
            hp += a_hp * (lp4 - hp);
            let sat = (3.5 * (lp4 - hp)).tanh();
            post += a_lp * (sat - post);
            nl_wet.push(post as f32);
        }
        let sh_est = crate::align::estimate_subsample_lag_and_gain(&dry, &nl_shallow, 64).unwrap();
        assert_eq!(
            guard_lag_causal_onset(&dry, &nl_shallow, 1.5, sh_est.total_lag_samples, Some(0.0)),
            0.0
        );

        let nl_est = crate::align::estimate_subsample_lag_and_gain(&dry, &nl_wet, 64).unwrap();
        assert!(nl_est.total_lag_samples > 1.5);
        assert_eq!(
            guard_lag_causal_onset(&dry, &nl_wet, 1.5, nl_est.total_lag_samples, Some(0.0)),
            0.0
        );

        let mut nl_del24 = vec![0.0_f32; dry.len()];
        nl_del24[24..].copy_from_slice(&nl_wet[..(dry.len() - 24)]);
        let del_est = crate::align::estimate_subsample_lag_and_gain(&dry, &nl_del24, 64).unwrap();
        assert_eq!(
            guard_lag_causal_onset(&dry, &nl_del24, 25.5, del_est.total_lag_samples, Some(24.0)),
            24.0
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn prop_causal_guard_bounds_and_pure_delay_preservation(
            cand in 0.0_f64..40.0_f64,
            tc_raw in -5.0_f64..40.0_f64,
            unconstrained_delta in -2.0_f64..10.0_f64,
            pure_lag in 0.05_f64..20.0_f64,
        ) {
            let seed = [0x51_u8; 256];
            let (dry, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
                &seed, 2_048, 1, 48_000,
            );

            // 1. None / non-finite tau_coarse_opt returns cand_lag unchanged.
            prop_assert_eq!(guard_lag_causal_onset(&dry, &dry, cand, cand, None), cand);
            prop_assert_eq!(
                guard_lag_causal_onset(&dry, &dry, cand, cand, Some(f64::NAN)),
                cand
            );

            // 2. For finite tc_raw and cand >= 0.0, output lies in [0.0, cand].
            let tc = tc_raw.max(0.0);
            let out = guard_lag_causal_onset(
                &dry,
                &dry,
                cand,
                tc + unconstrained_delta,
                Some(tc_raw),
            );
            prop_assert!(out >= 0.0 && out <= cand);

            // 3. Guard 1: cand > tc + 1e-6 && unconstrained > tc + HYBRID_SUBSAMPLE_MAX_DELTA => tc.
            if cand > tc + 1.0e-6 && unconstrained_delta > HYBRID_SUBSAMPLE_MAX_DELTA + 1.0e-6 {
                prop_assert_eq!(out, tc);
            }

            // 4. Pure fractional delays with tc = floor(pure_lag) are preserved within < 0.02.
            let delayed = realign_fractional_kaiser64(&dry, -pure_lag, 0.0, false);
            let tc_pure = pure_lag.floor();
            let (cand_pure, _, _) =
                refine_fractional_lag_hybrid(&dry, &delayed, pure_lag, Some(tc_pure));
            let guarded_pure =
                guard_lag_causal_onset(&dry, &delayed, cand_pure, pure_lag, Some(tc_pure));
            prop_assert!(
                (guarded_pure - pure_lag).abs() < 0.02,
                "pure_lag={pure_lag} cand={cand_pure} guarded={guarded_pure}"
            );
        }
    }
}
