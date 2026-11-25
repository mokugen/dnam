//! Companion Perceptual Metrics (`MR-IIR`, `LTAS` 33-Band 1/3-Octave Tilt, and `SegESR`).
//!
//! Canonical Reference: `docs/specs/architecture/psychoacoustic_quality_and_tiers.md` §3 (`r[qual.companion.*]`).

#![forbid(unsafe_code)]

mod ltas;
mod mr_iir;

pub(crate) use ltas::LtasTargetCache;
pub use ltas::{
    LtasEvaluationReport, LtasInterval, canonical_ltas_33_centers_hz,
    canonical_ltas_33_intervals_hz, compute_ltas_33band_max_deviation_db,
    evaluate_ltas_33band_report, evaluate_ltas_33band_report_with_max_freq,
};
pub(crate) use mr_iir::MrIirTargetCache;
pub use mr_iir::{compute_mr_iir_loss, compute_mr_iir_loss_with_gaps};

use super::bezout::{const_abs_f64, const_exp_f64, split_bezout_3band_series};

const fn const_pow10_f64(exp10: f64) -> f64 {
    const_exp_f64(exp10 * std::f64::consts::LN_10)
}

/// `+1.0 dB` linear power factor $10^{0.1} \approx 1.2589254117941673$ for noise-limited guards.
pub const ONE_DB_POWER_FACTOR: f64 = const_pow10_f64(0.1);
const _: () = assert!(
    const_abs_f64(
        ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            - 10.0
    ) < 1.0e-12
);

/// `+6.0 dB` linear power factor $10^{0.6} \approx 3.9810717055349722$ for active frame gating.
pub const SIX_DB_POWER_FACTOR: f64 = const_pow10_f64(0.6);
const _: () = assert!(const_abs_f64(SIX_DB_POWER_FACTOR - 3.981_071_705_534_972_2) < 1.0e-12);

pub(crate) const MAX_METRIC_SAMPLE_RATE_HZ: f64 = 1_536_000.0;

pub use dnam_train::losses::{
    DRY_SILENCE_EPS, DrySilentGapMask, MIN_DRY_SILENT_GAP_SECS, POST_T60_GAP_AMP_RATIO,
    SEG_ESR_UNEXCITED_BAND_AMP_RATIO, SEG_ESR_UNEXCITED_BAND_GUARD_RATIO,
};

// ============================================================================
// Segmental Frame-Wise ESR (`SegESR`, r[qual.companion.seg-esr])
// ============================================================================

/// Contiguous constant-sample-rate timeline segment for multi-rate `SegESR` aggregation (`ref[qual.companion.seg-esr]`).
#[derive(Clone, Copy, Debug)]
pub struct MultiRateSegEsrSlice<'a> {
    pub sample_rate_hz: f64,
    pub target: &'a [f64],
    pub model: &'a [f64],
    pub noise: Option<&'a [f64]>,
}

/// Variant of [`compute_multirate_seg_esr_with_profile_db`] that gates noiseless post-`T60`
/// dry-silent gaps via `gap_mask` while enforcing a strict `-60 dB` residual guard across each gap.
#[must_use]
pub fn compute_multirate_seg_esr_with_gaps_db(
    segments: &[MultiRateSegEsrSlice<'_>],
    env_inc_ratio: [f64; 3],
    gap_mask: Option<&DrySilentGapMask>,
    seg_abs_offset: usize,
) -> (f64, usize, bool) {
    if segments.is_empty()
        || !env_inc_ratio
            .iter()
            .all(|&r| r.is_finite() && (0.0..=4.0).contains(&r))
    {
        return (f64::INFINITY, 0, false);
    }
    let eff_gaps = gap_mask.filter(|g| !g.is_empty());
    let mut weighted_lin_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    let mut active_segs = 0usize;
    let mut cur_offset = seg_abs_offset;

    for seg in segments {
        let fs = seg.sample_rate_hz;
        if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&fs)
            || seg.target.is_empty()
            || seg.target.len() != seg.model.len()
            || !seg.target.iter().all(|v| v.is_finite())
            || !seg.model.iter().all(|v| v.is_finite())
        {
            return (f64::INFINITY, 0, false);
        }
        let n = seg.target.len();
        if let Some(ns) = seg.noise
            && (ns.len() < n || !ns[..n].iter().all(|v| v.is_finite()))
        {
            return (f64::INFINITY, 0, false);
        }

        let Some(t_bands) = split_bezout_3band_series(seg.target, fs) else {
            return (f64::INFINITY, active_segs, false);
        };
        let Some(m_bands) = split_bezout_3band_series(seg.model, fs) else {
            return (f64::INFINITY, active_segs, false);
        };
        let n_bands = if let Some(ns) = seg.noise {
            let Some(nb) = split_bezout_3band_series(&ns[..n], fs) else {
                return (f64::INFINITY, active_segs, false);
            };
            Some(nb)
        } else {
            None
        };

        let full_len = (fs / 50.0).round().max(1.0) as usize;
        let mut start = 0usize;
        while start < n {
            let end = (start + full_len).min(n);
            let n_f = end - start;
            let inv_nf = 1.0 / (n_f as f64);
            let mut e_sig_bb = 0.0_f64;
            let mut e_err_bb = 0.0_f64;
            let mut e_nse_bb = 0.0_f64;
            let mut nz_sig = false;
            let mut nz_err = false;
            let mut nz_nse = false;
            for i in start..end {
                let yt = seg.target[i];
                let ym = seg.model[i];
                let yn = seg.noise.map_or(0.0, |ns| ns[i]);
                let diff = ym - yt;
                e_sig_bb += yt * yt;
                e_err_bb += diff * diff;
                e_nse_bb += yn * yn;
                nz_sig |= yt != 0.0;
                nz_err |= diff != 0.0;
                nz_nse |= yn != 0.0;
            }
            if (nz_sig && e_sig_bb == 0.0)
                || (nz_err && e_err_bb == 0.0)
                || (nz_nse && e_nse_bb == 0.0)
                || !e_sig_bb.is_finite()
                || !e_err_bb.is_finite()
                || !e_nse_bb.is_finite()
            {
                return (f64::INFINITY, active_segs, false);
            }
            if seg.noise.is_none()
                && let Some(gm) = eff_gaps
                && gm
                    .overlap_span(cur_offset + start, cur_offset + end)
                    .is_some()
                && (e_sig_bb * inv_nf).sqrt() <= POST_T60_GAP_AMP_RATIO * gm.active_target_rms
            {
                if (e_err_bb * inv_nf).sqrt() > POST_T60_GAP_AMP_RATIO * gm.active_target_rms {
                    return (f64::INFINITY, active_segs, false);
                }
                start = end;
                continue;
            }
            let is_active = if e_nse_bb > 0.0 {
                e_sig_bb > SIX_DB_POWER_FACTOR * e_nse_bb
            } else {
                e_sig_bb > 0.0
            };

            if is_active {
                let bb_sig_rms = (e_sig_bb * inv_nf).sqrt();
                let mut r_sum = 0.0_f64;
                let mut excited_bands = 0_usize;
                for b in 0..3 {
                    let y_sq: f64 =
                        t_bands[b][start..end].iter().map(|&v| v * v).sum::<f64>() * inv_nf;
                    let m_sq: f64 =
                        m_bands[b][start..end].iter().map(|&v| v * v).sum::<f64>() * inv_nf;
                    let n_sq: f64 = n_bands.as_ref().map_or(0.0, |nb| {
                        nb[b][start..end].iter().map(|&v| v * v).sum::<f64>() * inv_nf
                    });
                    if !y_sq.is_finite() || !m_sq.is_finite() || !n_sq.is_finite() {
                        return (f64::INFINITY, active_segs, false);
                    }
                    let y_rms = y_sq.sqrt();
                    let m_rms = m_sq.sqrt();
                    let f_env = n_sq + env_inc_ratio[b] * y_sq;
                    if seg.noise.is_none()
                        && f_env == 0.0
                        && y_rms <= SEG_ESR_UNEXCITED_BAND_AMP_RATIO * bb_sig_rms
                    {
                        if (m_rms - y_rms).abs() > SEG_ESR_UNEXCITED_BAND_GUARD_RATIO * bb_sig_rms {
                            return (f64::INFINITY, active_segs, false);
                        }
                        continue;
                    }
                    let denom = y_sq + f_env;
                    let r_b = if denom > 0.0 {
                        let d_rms = m_rms - y_rms;
                        (d_rms * d_rms - f_env).max(0.0) / denom
                    } else if m_sq == 0.0 {
                        0.0
                    } else {
                        return (f64::INFINITY, active_segs, false);
                    };
                    r_sum += r_b;
                    excited_bands += 1;
                }
                if excited_bands == 0 {
                    return (f64::INFINITY, active_segs, false);
                }
                let r_f = r_sum / (excited_bands as f64);
                let w_f = (n_f as f64) / fs;
                weighted_lin_sum += w_f * r_f;
                weight_sum += w_f;
                active_segs += 1;
            } else if e_err_bb > ONE_DB_POWER_FACTOR * e_nse_bb {
                return (f64::INFINITY, active_segs, false);
            }
            start = end;
        }
        cur_offset = cur_offset.saturating_add(n);
    }

    if active_segs == 0
        || weight_sum <= 0.0
        || !weighted_lin_sum.is_finite()
        || !weight_sum.is_finite()
    {
        return (f64::INFINITY, 0, false);
    }
    let r_bar = weighted_lin_sum / weight_sum;
    if !r_bar.is_finite() {
        return (f64::INFINITY, active_segs, false);
    }
    let db = if r_bar == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * r_bar.log10()
    };
    (db, active_segs, true)
}

/// Evaluates duration-weighted `20 ms` (`L = round(fs / 50)`) 3-band Bézout RMS energy-envelope
/// `SegESR` across one or more constant-rate segments without spanning rate transitions, retaining
/// exact-duration-weighted partial tail frames and enforcing the `+1.0 dB` excluded-frame residual
/// guard (`ref[qual.companion.seg-esr]`).
#[must_use]
pub fn compute_multirate_seg_esr_with_profile_db(
    segments: &[MultiRateSegEsrSlice<'_>],
    env_inc_ratio: [f64; 3],
) -> (f64, usize, bool) {
    compute_multirate_seg_esr_with_gaps_db(segments, env_inc_ratio, None, 0)
}

/// Evaluates duration-weighted `20 ms` (`L = round(fs / 50)`) 3-band Bézout RMS energy-envelope
/// `SegESR` with zero incoherence relaxation (`env_inc_ratio = [0.0; 3]`).
#[must_use]
pub fn compute_multirate_seg_esr_db(segments: &[MultiRateSegEsrSlice<'_>]) -> (f64, usize, bool) {
    compute_multirate_seg_esr_with_profile_db(segments, [0.0; 3])
}

/// Computes `20 ms` (`L = round(fs / 50)`) duration-weighted Segmented ESR (`SegESR`) in dB
/// over active signal frames, including partial trailing frames (`ref[qual.companion.seg-esr]`).
#[must_use]
pub fn compute_seg_esr_20ms_db(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
) -> (f64, usize) {
    let (db, count, _valid) = compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
        sample_rate_hz,
        target,
        model,
        noise,
    }]);
    (db, count)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::mr_iir::compute_mr_iir_loss_general;
    use super::*;

    #[test]
    fn test_mr_iir_and_ltas_target_caches_match_reference() {
        let sr = 48_000.0;
        let centers = canonical_ltas_33_centers_hz();
        let n = 20_000usize;
        let mut tgt = vec![0.0_f64; n];
        let mut mdl = vec![0.0_f64; n];
        for (k, &hz) in centers.iter().enumerate() {
            let amp_err = 0.02 * ((k as f64) * 0.31).sin();
            for i in 0..n {
                let s = 0.03 * (2.0 * PI * hz * (i as f64) / sr + 0.2 * (k as f64)).sin();
                tgt[i] += s;
                mdl[i] += s * (1.0 + amp_err);
            }
        }
        let zeros = vec![0.0_f64; n];

        // MR-IIR cached vs reference general path (regular + unaligned tail + short window)
        let ref_mr = compute_mr_iir_loss_general(&tgt, &mdl, Some(&zeros), sr, 64);
        let fast_mr = compute_mr_iir_loss(&tgt, &mdl, None, sr, 64);
        assert!(
            (ref_mr - fast_mr).abs() < 1.0e-12,
            "ref={ref_mr} fast={fast_mr}"
        );

        let short_n = 2_048;
        let ref_short = compute_mr_iir_loss_general(
            &tgt[..short_n],
            &mdl[..short_n],
            Some(&zeros[..short_n]),
            sr,
            128,
        );
        let fast_short = compute_mr_iir_loss(&tgt[..short_n], &mdl[..short_n], None, sr, 128);
        assert!(
            (ref_short - fast_short).abs() < 1.0e-12,
            "ref_short={ref_short} fast_short={fast_short}"
        );

        let cache_mr = MrIirTargetCache::new(&tgt, sr, 64).unwrap();
        assert!(cache_mr.eval_model(&mdl[..short_n]).is_infinite());
        assert!(cache_mr.eval_model(&vec![1.0e-180; n]).is_infinite());

        // LTAS cached vs noise=Some(&zeros) reference
        let ref_ltas =
            evaluate_ltas_33band_report_with_max_freq(&tgt, &mdl, Some(&zeros), sr, 20_000.0);
        let mut cache_ltas = LtasTargetCache::new(&tgt, sr, 20_000.0).unwrap();
        let fast_ltas = cache_ltas.eval_model(&mdl);
        assert_eq!(ref_ltas.available, fast_ltas.available);
        assert_eq!(ref_ltas.active_band_count, fast_ltas.active_band_count);
        assert_eq!(ref_ltas.excluded_bands, fast_ltas.excluded_bands);
        assert!(
            (ref_ltas.max_adj_deviation_db - fast_ltas.max_adj_deviation_db).abs() < 1.0e-12,
            "ref_ltas={} fast_ltas={}",
            ref_ltas.max_adj_deviation_db,
            fast_ltas.max_adj_deviation_db
        );
        assert!(
            (ref_ltas.max_unadjusted_deviation_db - fast_ltas.max_unadjusted_deviation_db).abs()
                < 1.0e-12
        );

        // Pure DC / zero-residual self-null preserves exact 0 residual without paired-FFT leakage
        let dc = vec![0.25_f64; 16_384];
        let mut dc_cache = LtasTargetCache::new(&dc, sr, 20_000.0).unwrap();
        let dc_rep = dc_cache.eval_model(&dc);
        let ref_dc = evaluate_ltas_33band_report_with_max_freq(
            &dc,
            &dc,
            Some(&zeros[..16_384]),
            sr,
            20_000.0,
        );
        assert_eq!(dc_rep.available, ref_dc.available);
        assert_eq!(dc_rep.active_band_count, ref_dc.active_band_count);
        assert!(dc_rep.residual_band_power.iter().all(|&p| p == 0.0));
        let mut sil_cache = LtasTargetCache::new(&zeros[..16_384], sr, 20_000.0).unwrap();
        let sil_rep = sil_cache.eval_model(&zeros[..16_384]);
        assert!(!sil_rep.available && sil_rep.max_adj_deviation_db.is_infinite());
        assert!(sil_rep.residual_band_power.iter().all(|&p| p == 0.0));
        assert!(!cache_ltas.eval_model(&[0.0; 16]).available);

        // Noiseless SegESR unexcited-band guard: -55 dB LF transient on active 17 kHz chirp
        // passes cleanly without 1/y^2 singularity, while -30 dB spurious LF fails closed.
        let n_hf = 3_840_usize;
        let tau = std::f64::consts::TAU;
        let tgt_17k: Vec<f64> = (0..n_hf)
            .map(|i| 0.4 * (tau * 17_000.0 * (i as f64) / sr).sin())
            .collect();
        let mdl_lf_ok: Vec<f64> = tgt_17k
            .iter()
            .enumerate()
            .map(|(i, &y)| y + 0.4 * 1.8e-3 * (tau * 25.0 * (i as f64) / sr).sin())
            .collect();
        let (seg_ok_db, seg_ok_cnt, seg_ok) =
            compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
                sample_rate_hz: sr,
                target: &tgt_17k[960..],
                model: &mdl_lf_ok[960..],
                noise: None,
            }]);
        assert!(
            seg_ok && seg_ok_cnt == 3 && seg_ok_db < -25.0,
            "seg_ok_db={seg_ok_db}"
        );

        let mdl_lf_bad: Vec<f64> = tgt_17k
            .iter()
            .enumerate()
            .map(|(i, &y)| y + 0.4 * 3.0e-2 * (tau * 25.0 * (i as f64) / sr).sin())
            .collect();
        let (seg_bad_db, _, seg_bad) = compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
            sample_rate_hz: sr,
            target: &tgt_17k[960..],
            model: &mdl_lf_bad[960..],
            noise: None,
        }]);
        assert!(!seg_bad && seg_bad_db.is_infinite());

        // Cover compute_seg_esr_20ms_db (self-null -> NEG_INFINITY), gap_mask pass/fail,
        // noise floor gating (+6 dB active / +1 dB passive guard), and input validation guards.
        let (self_db, self_cnt) =
            compute_seg_esr_20ms_db(&tgt_17k[960..], &tgt_17k[960..], None, sr);
        assert!(self_db == f64::NEG_INFINITY && self_cnt > 0);

        let gap_mask = DrySilentGapMask {
            spans: vec![(0, 960)],
            active_target_rms: 0.4 * std::f64::consts::FRAC_1_SQRT_2,
        };
        let gap_tgt = vec![0.0_f64; 960];
        let mut sig_tail = gap_tgt.clone();
        sig_tail.extend_from_slice(&tgt_17k[960..]);
        let mut mdl_gap_ok = vec![1.0e-5_f64; 960];
        mdl_gap_ok.extend_from_slice(&tgt_17k[960..]);
        let (gap_ok_db, gap_ok_cnt, gap_ok) = compute_multirate_seg_esr_with_gaps_db(
            &[MultiRateSegEsrSlice {
                sample_rate_hz: sr,
                target: &sig_tail,
                model: &mdl_gap_ok,
                noise: None,
            }],
            [0.0; 3],
            Some(&gap_mask),
            0,
        );
        assert!(
            gap_ok && gap_ok_cnt > 0 && gap_ok_db < -60.0,
            "gap_ok_db={gap_ok_db}"
        );

        let mut mdl_gap_bad = vec![0.05_f64; 960];
        mdl_gap_bad.extend_from_slice(&tgt_17k[960..]);
        let (gap_bad_db, _, gap_bad) = compute_multirate_seg_esr_with_gaps_db(
            &[MultiRateSegEsrSlice {
                sample_rate_hz: sr,
                target: &sig_tail,
                model: &mdl_gap_bad,
                noise: None,
            }],
            [0.0; 3],
            Some(&gap_mask),
            0,
        );
        assert!(!gap_bad && gap_bad_db.is_infinite());

        let quiet_noise = vec![1.0e-3_f64; sig_tail.len()];
        let (nse_fail_db, _, nse_fail) = compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
            sample_rate_hz: sr,
            target: &sig_tail,
            model: &mdl_gap_bad,
            noise: Some(&quiet_noise),
        }]);
        assert!(!nse_fail && nse_fail_db.is_infinite());

        let (nse_ok_db, nse_ok_cnt, nse_ok) =
            compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
                sample_rate_hz: sr,
                target: &tgt_17k[960..],
                model: &mdl_lf_ok[960..],
                noise: Some(&quiet_noise[..tgt_17k[960..].len()]),
            }]);
        assert!(nse_ok && nse_ok_cnt > 0 && nse_ok_db.is_finite());

        assert!(!compute_multirate_seg_esr_db(&[]).2);
        assert!(
            !compute_multirate_seg_esr_with_profile_db(
                &[MultiRateSegEsrSlice {
                    sample_rate_hz: sr,
                    target: &tgt_17k[960..],
                    model: &tgt_17k[960..],
                    noise: None,
                }],
                [-1.0, 0.0, 0.0],
            )
            .2
        );
        assert!(
            !compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
                sample_rate_hz: 4_000.0,
                target: &tgt_17k[960..],
                model: &tgt_17k[960..],
                noise: None,
            }])
            .2
        );
        assert!(
            !compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
                sample_rate_hz: sr,
                target: &tgt_17k[960..],
                model: &tgt_17k[960..968],
                noise: None,
            }])
            .2
        );
        assert!(
            !compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
                sample_rate_hz: sr,
                target: &tgt_17k[960..],
                model: &tgt_17k[960..],
                noise: Some(&quiet_noise[..8]),
            }])
            .2
        );
    }
}
