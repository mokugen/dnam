#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Schroeder Energy Decay Curve (`EDC`) $T_{60}$ Estimator over `Regime 2` Silent Decay Gaps
//! (`stimulus::pipeline::schoukens::edc`).
//!
//! Uses backward Schroeder integration with dynamic noise-floor knee truncation and Chu/Lundeby
//! tail-remainder correction so `250–300 ms` decay gaps accurately measure both short and multi-second
//! reverbs while strictly returning `0.0` on dry/non-tail targets.

/// +5 dB Lundeby knee margin over effective noise floor (`10^(5/10) = sqrt(10)`).
const KNEE_FLOOR_MULT_5DB: f64 = 3.162_277_660_168_379_5;
const _: () = assert!(
    -1.0e-12 < (KNEE_FLOOR_MULT_5DB * KNEE_FLOOR_MULT_5DB - 10.0)
        && (KNEE_FLOOR_MULT_5DB * KNEE_FLOOR_MULT_5DB - 10.0) < 1.0e-12
);
const MIN_GAP_DURATION_S: f64 = 0.050;
const RING_GUARD_SKIP_S: f64 = 0.006;
const SETTLE_GATE_START_S: f64 = 0.010;
const SETTLE_GATE_SPAN_S: f64 = 0.020;
const EARLY_SKIP_S: f64 = 0.044;
/// Minimum decay slope corresponding to physical maximum `T60 <= 30.0 s` (`-60 / 30 = -2.0 dB/s`).
const MIN_DECAY_SLOPE_DB_PER_S: f64 = -2.0;
const MIN_EDC_LINEARITY_R2: f64 = 0.45;
const MIN_UNTRUNCATED_SMOOTH_R2: f64 = 0.15;

/// Computes Schroeder Energy Decay Curve `T_60` (`edc_t60_s >= 0.0`) from aligned wet audio over
/// `Regime 2` `(gap_start, gap_end)` decay intervals (`ref[train.capture.excitation.codesigned]`).
#[must_use]
pub fn compute_schroeder_edc_t60_s(
    wet_l: &[f32],
    wet_r: Option<&[f32]>,
    gap_ranges: &[(usize, usize)],
    sample_rate_hz: u32,
    noise_floor_lin: f64,
) -> f32 {
    if sample_rate_hz == 0 || wet_l.is_empty() || gap_ranges.is_empty() {
        return 0.0;
    }
    let n_avail = wet_r.map_or(wet_l.len(), |r| wet_l.len().min(r.len()));
    let sr = f64::from(sample_rate_hz.clamp(8_000, 384_000));
    let safe_floor = if noise_floor_lin.is_finite() && noise_floor_lin > 0.0 {
        noise_floor_lin
    } else {
        1.0e-12
    };

    // Select the longest final gap inside n_avail that contains active post-10 ms decay tail.
    let min_gap = ((MIN_GAP_DURATION_S * sr).round() as usize).max(64);
    let mut best_gap: Option<(usize, usize)> = None;
    for &(gs, ge) in gap_ranges {
        let s = gs.min(n_avail);
        let e = ge.min(n_avail);
        if e.saturating_sub(s) >= min_gap {
            best_gap = Some((s, e));
        }
    }
    let Some((gs, ge)) = best_gap else {
        return 0.0;
    };

    let pre_start = gs.saturating_sub(min_gap);
    let mut pre_pow_sum = 0.0_f64;
    for i in pre_start..gs {
        let l = if wet_l[i].is_finite() { wet_l[i] } else { 0.0 };
        let r = wet_r
            .and_then(|wr| wr.get(i).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(l);
        let lf = f64::from(l);
        let rf = f64::from(r);
        pre_pow_sum += 0.5 * (lf * lf + rf * rf);
    }
    let pre_mean_p = (pre_pow_sum / ((gs - pre_start).max(1) as f64)).max(1.0e-18);

    // Compute instantaneous power across the gap, skipping the first 6 ms FIR/converter ring guard.
    let skip_ring = ((RING_GUARD_SKIP_S * sr).round() as usize).min((ge - gs) / 8);
    let s0 = gs + skip_ring;
    let m = ge.saturating_sub(s0);
    if m < 1_024 {
        return 0.0;
    }

    let mut pow = Vec::with_capacity(m);
    for i in s0..ge {
        let l = wet_l
            .get(i)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        let r = wet_r
            .and_then(|wr| wr.get(i).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(l);
        let pl = f64::from(l);
        let pr = f64::from(r);
        pow.push(0.5 * (pl * pl + pr * pr));
    }

    // Dry-target gate: inspect mean power in `[10 ms .. 30 ms]` after gap start.
    let settle_10ms = ((SETTLE_GATE_START_S * sr).round() as usize).min(m / 4);
    let gate_end = (settle_10ms + ((SETTLE_GATE_SPAN_S * sr).round() as usize)).min(m);
    let late_tail_mean =
        pow[settle_10ms..gate_end].iter().sum::<f64>() / ((gate_end - settle_10ms).max(1) as f64);
    if late_tail_mean <= (12.0 * safe_floor).max(2.0e-5 * pre_mean_p) {
        return 0.0;
    }

    // Only escalate `eff_floor` above `safe_floor` when the tail has fallen >= 20 dB below gap start
    // AND plateaued completely (`tail_mean >= 0.85 * pow_70_85`), so long-T60 reverbs never trigger it.
    let slice_70_85 = &pow[(m * 70) / 100..(m * 85) / 100];
    let pow_70_85 = slice_70_85.iter().sum::<f64>() / (slice_70_85.len().max(1) as f64);
    let tail_slice = &pow[(m * 85) / 100..];
    let tail_mean = tail_slice.iter().sum::<f64>() / (tail_slice.len().max(1) as f64);
    let eff_floor = if tail_mean >= 0.85 * pow_70_85 && tail_mean < 0.01 * late_tail_mean {
        safe_floor.max(tail_mean)
    } else {
        safe_floor
    };

    // Moving-average (`10 ms`) smoothed power envelope to find truncation knee (`<= 3.16 * eff_floor`)
    // and estimate decay slope for Chu/Lundeby truncation remainder `E_tail`.
    let win = ((SETTLE_GATE_START_S * sr).round() as usize).clamp(16, m / 6);
    let mut smooth = Vec::with_capacity(m - win + 1);
    let mut run = pow[..win].iter().sum::<f64>();
    smooth.push(run / (win as f64));
    for k in win..m {
        run += pow[k] - pow[k - win];
        smooth.push(run / (win as f64));
    }
    let knee_thresh = KNEE_FLOOR_MULT_5DB * eff_floor;
    let mut trunc = m;
    for (idx, &sm) in smooth.iter().enumerate() {
        if idx >= settle_10ms && sm <= knee_thresh {
            trunc = (idx + win / 2).max(settle_10ms + 64).min(m);
            break;
        }
    }

    // Coarse fit over post-50 ms log10(smooth) (ISO 3382 early-reflection skip) to estimate
    // decay rate and Lundeby truncation remainder E_tail accurately even under long pre-delays.
    let sm_end = trunc.min(smooth.len()).max(2);
    let coarse_start = ((EARLY_SKIP_S * sr).round() as usize).min(sm_end / 3);
    let fit_smooth = |lo: usize, hi: usize| -> (f64, f64, f64, f64, f64, f64) {
        let (mut sx, mut sy, mut syy, mut sxx, mut sxy, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (idx, &sm) in smooth[lo..hi].iter().enumerate() {
            if sm > knee_thresh && sm > 1.0e-18 {
                let t = ((lo + idx + win / 2) as f64) / sr;
                let db = 10.0 * sm.log10();
                sx += t;
                sy += db;
                syy += db * db;
                sxx += t * t;
                sxy += t * db;
                cnt += 1.0;
            }
        }
        (sx, sy, syy, sxx, sxy, cnt)
    };
    let (mut sx, mut sy, mut syy, mut sxx, mut sxy, mut cnt) = fit_smooth(coarse_start, sm_end);
    if cnt < 16.0 {
        (sx, sy, syy, sxx, sxy, cnt) = fit_smooth(0, sm_end);
    }
    let denom = cnt * sxx - sx * sx;
    if cnt < 16.0 || denom.abs() <= 1.0e-12 {
        return 0.0;
    }
    let num = cnt * sxy - sx * sy;
    let coarse_slope = num / denom;
    let ss_tot = cnt * syy - sy * sy;
    let coarse_r2 = if ss_tot > 1.0e-12 {
        (num * num) / (denom * ss_tot)
    } else {
        0.0
    };
    if !coarse_slope.is_finite()
        || coarse_slope >= MIN_DECAY_SLOPE_DB_PER_S
        || (trunc == m && coarse_r2 < MIN_UNTRUNCATED_SMOOTH_R2)
    {
        return 0.0;
    }

    // Subtract expected noise power on truncated prefix and add exponential tail remainder.
    let alpha_decay = (-coarse_slope * std::f64::consts::LN_10) / (10.0 * sr);
    let dt_rem = if trunc < m { 0.0 } else { 0.5 * (win as f64) };
    let p_end_smooth = (smooth[sm_end - 1] - eff_floor).max(0.0) * (-alpha_decay * dt_rem).exp();
    let e_tail = if alpha_decay > 1.0e-9 {
        p_end_smooth / alpha_decay
    } else {
        0.0
    };

    let mut edc = vec![0.0_f64; trunc];
    let mut acc = e_tail;
    for k in (0..trunc).rev() {
        acc += (pow[k] - eff_floor).max(0.0);
        edc[k] = acc;
    }
    let e0 = edc[0];
    if !e0.is_finite() || e0 <= 1.0e-18 {
        return 0.0;
    }

    // Final linear fit of Schroeder curve `10 * log10(edc[k] / e_ref)` over `[fit_start .. 85% trunc]`.
    let fit_end = ((trunc * 85) / 100).max(32).min(trunc);
    let fit_start = coarse_start.min(fit_end.saturating_sub(32));
    let e_ref = edc[fit_start].max(1.0e-18);
    let (mut fx, mut fy, mut fyy, mut fxx, mut fxy, mut fcnt) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (idx, &ek) in edc[fit_start..fit_end].iter().enumerate() {
        let rel = ek / e_ref;
        if rel > 1.0e-5 {
            let t = ((fit_start + idx) as f64) / sr;
            let db = 10.0 * rel.log10();
            fx += t;
            fy += db;
            fyy += db * db;
            fxx += t * t;
            fxy += t * db;
            fcnt += 1.0;
        }
    }
    let fdenom = fcnt * fxx - fx * fx;
    if fcnt < 16.0 || fdenom.abs() <= 1.0e-12 {
        return 0.0;
    }
    let fnum = fcnt * fxy - fx * fy;
    let final_slope = fnum / fdenom;
    let fss_tot = fcnt * fyy - fy * fy;
    let final_r2 = if fss_tot > 1.0e-12 {
        (fnum * fnum) / (fdenom * fss_tot)
    } else {
        0.0
    };
    if !final_slope.is_finite()
        || final_slope >= MIN_DECAY_SLOPE_DB_PER_S
        || !final_r2.is_finite()
        || final_r2 < MIN_EDC_LINEARITY_R2
    {
        return 0.0;
    }
    ((-60.0 / final_slope).clamp(0.0, 30.0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stimulus::pipeline::schoukens::regimes::synthesize_regime2_staircase_and_gaps;

    fn synth_reverb_convolved(dry: &[f32], t60_s: f64, wet_gain: f64) -> Vec<f32> {
        let len = dry.len();
        let mut out = dry.to_vec();
        // Multi-tap exponential reverb tail across 48 kHz comb delays to emulate FDN density:
        // st^2 decays by -60 dB at t = t60_s when per-sample pole magnitude is 10^(-3 / (48000 * t60_s)).
        let tap_delays = [611_usize, 907, 1_319, 1_789, 2_411];
        let decay = (-3.0_f64 * std::f64::consts::LN_10 / (48_000.0 * t60_s)).exp();
        let g_tap = wet_gain / (tap_delays.len() as f64);
        for (idx, &delay) in tap_delays.iter().enumerate() {
            let mut st = 0.0_f64;
            let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
            for i in 0..len {
                let x = if i >= delay {
                    f64::from(dry[i - delay])
                } else {
                    0.0
                };
                st = st * decay + x * g_tap;
                out[i] += (sign * st * 0.04) as f32;
            }
        }
        out
    }

    #[test]
    fn test_schroeder_edc_t60_dry_vs_short_medium_long_and_noisy_reverbs() {
        let r2 = synthesize_regime2_staircase_and_gaps(40_800);
        let gaps = vec![
            (3_600, 6_000),
            (9_600, 12_000),
            (16_800, 22_800),
            (28_800, 40_800),
        ];

        assert_eq!(
            compute_schroeder_edc_t60_s(&r2, None, &gaps, 48_000, 1e-12),
            0.0
        );

        for &target_t60 in &[0.25_f64, 0.80_f64, 2.00_f64] {
            let wet = synth_reverb_convolved(&r2, target_t60, 1.0);
            let est = compute_schroeder_edc_t60_s(&wet, None, &gaps, 48_000, 1.0e-10);
            let rel_err = ((f64::from(est) - target_t60) / target_t60).abs();
            assert!(
                rel_err < 0.25,
                "expected T60~{target_t60}s, got {est}s (rel_err={rel_err:.3})"
            );
        }

        // -50 dBFS additive noise still stops at noise knee cleanly
        let wet_08 = synth_reverb_convolved(&r2, 0.80, 1.0);
        let noise_amp = 10.0_f64.powf(-50.0 / 20.0);
        let noisy: Vec<f32> = wet_08
            .iter()
            .enumerate()
            .map(|(i, &w)| w + (noise_amp * 0.35 * ((i as f64) * 1.37).sin()) as f32)
            .collect();
        let est_noisy = compute_schroeder_edc_t60_s(&noisy, None, &gaps, 48_000, 1.0e-5);
        assert!(
            est_noisy > 0.35 && est_noisy < 1.35,
            "noisy T60={est_noisy}"
        );
    }

    #[test]
    fn test_schroeder_edc_t60_guards_noise_r2_nan_plateau_and_stereo() {
        assert_eq!(
            compute_schroeder_edc_t60_s(&[], None, &[], 48_000, 1e-8),
            0.0
        );
        let short_r2 = synthesize_regime2_staircase_and_gaps(20_000);
        // NaN inside pre-burst rejects cleanly without false decay tail.
        let mut nan_pre = short_r2.clone();
        nan_pre[1_000] = f32::NAN;
        assert_eq!(
            compute_schroeder_edc_t60_s(&nan_pre, None, &[(15_000, 20_000)], 48_000, 1e-8),
            0.0
        );

        // Stationary broadband pseudo-random noise across final gap must trigger R^2 guard -> 0.0.
        let mut noise_tail = short_r2.clone();
        let mut rng = 0xDEAD_BEEF_CAFE_BABE_u64;
        for s in &mut noise_tail[14_000..20_000] {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng >> 32) as f64) / f64::from(u32::MAX) - 0.5;
            *s = (0.04 * u) as f32;
        }
        assert_eq!(
            compute_schroeder_edc_t60_s(&noise_tail, None, &[(14_500, 20_000)], 48_000, 1e-8),
            0.0
        );

        // Positive rising envelope in gap fails slope guard -> 0.0.
        let mut rising = short_r2;
        for (idx, s) in rising[14_000..20_000].iter_mut().enumerate() {
            let amp = 0.01 + 0.05 * ((idx as f64) / 6_000.0);
            *s = (amp * ((idx as f64) * 0.13).sin()) as f32;
        }
        assert_eq!(
            compute_schroeder_edc_t60_s(&rising, None, &[(14_500, 20_000)], 48_000, 1e-8),
            0.0
        );

        // Stereo wet_r path preserves valid reverb T60 measurement.
        let r2 = synthesize_regime2_staircase_and_gaps(40_800);
        let gaps = vec![(28_800, 40_800)];
        let wet_st = synth_reverb_convolved(&r2, 0.80, 1.0);
        let t60_st = compute_schroeder_edc_t60_s(&r2, Some(&wet_st), &gaps, 48_000, 1e-10);
        assert!(t60_st > 0.35 && t60_st < 1.35, "stereo T60={t60_st}");

        // Short reverb (T60 = 0.25 s) decaying into an uncalibrated -40 dBFS broadband white-noise
        // plateau (calb_floor = 1e-10) must trigger plateau truncation via tail_mean >= 0.85 * pow_70_85.
        let mut wet_plateau = synth_reverb_convolved(&r2, 0.25, 4.0);
        let mut plateau_rng = 0x51AB_2345_6789_EF01_u64;
        for s in &mut wet_plateau {
            plateau_rng = plateau_rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((plateau_rng >> 32) as f64) / f64::from(u32::MAX) - 0.5;
            *s += (0.034_641 * u) as f32; // RMS = 0.01 (-40 dBFS)
        }
        let t60_plateau = compute_schroeder_edc_t60_s(&wet_plateau, None, &gaps, 48_000, 1.0e-10);
        assert!(
            (0.15..=0.40).contains(&t60_plateau),
            "plateau T60={t60_plateau}"
        );
    }
}
