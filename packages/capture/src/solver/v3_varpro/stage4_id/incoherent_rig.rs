#![forbid(unsafe_code)]
//! Incoherent modulation, BBD delay, and diffuse FDN spatial rig Stage IV solver (`incoherent_rig`).
//!
//! Fits an `Ultra` prime/comb FDN tank (`delay_lengths`, `g_loop`, `base_psi`, `wet_ratio`,
//! `tap_gains`) paired with `8192`-FFT minimum-phase cepstral initialization and Cholesky-whitened
//! Welch-33 minimax Stage III (`FIR` + `Kautz`) L-BFGS polish, then cascades top-down
//! `Ultra -> Pro -> Full -> Lite` maintaining strict 7-metric monotonicity and `< 0.020 dB` RMS lock.

mod cascade;
mod cepstral_eq;
mod comb_detect;
mod harmonic_cal;
mod lfo_detect;
mod mr_iir_adj;
pub(super) mod s2_modal;
mod screen;
mod wave_hinge;
mod welch_eq;
mod whitened_lbfgs;

use dnam_engine::Tier;
use dnam_engine::stages::V3MatryoshkaModel;

use crate::metrics::TargetRepeatabilityProfile;
use crate::metrics::certification::{PerceptualCertificationSummary, TargetChannelCertCache};
use crate::metrics::companion::LtasTargetCache;
use crate::solver::SolverQualityMode;
use crate::solver::backends::TrainerBackend;
use crate::solver::canonical_besr_warmup;
use crate::solver::progress::telemetry::{
    Stage4SummaryStep, VisualProbeAnchor, emit_stage4_summary_step,
};

use super::super::ltas_adj::{Welch33TargetSummary, build_welch33_target_summary};
use super::super::model_f64::{V3TrainDataset, V3TrainStem};
use super::{render_tier_stereo, summaries_monotone};
use cascade::{CascadeCtx, cascade_down_tiers, summary_passes_mode};
use cepstral_eq::build_eq_freq_grid;
use comb_detect::{CombSpec, build_comb_candidates, compute_broadband_rms, detect_comb_spec};
use harmonic_cal::{SweepHarmonicCheck, prepare_harmonic_tracks, slice_stem_prefix};
use mr_iir_adj::MrIirTargetCtx;
use screen::{
    ClosedLoopEqCtx, EqRunSpec, ScreenTargetCtx, UltraSearchBest, screen_track_finalists,
};
use wave_hinge::WaveTargetCtx;
use welch_eq::{WelchPolishSpec, polish_tier_welch33};

struct WelchSummaryPair<'a> {
    sum_l: &'a Welch33TargetSummary,
    sum_r: Option<&'a Welch33TargetSummary>,
}

#[derive(Clone)]
struct CertCachePair {
    cert_l: TargetChannelCertCache,
    cert_r: Option<TargetChannelCertCache>,
    rep_profiles: Option<[TargetRepeatabilityProfile; 2]>,
}

fn eval_tier_cert<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    certs: &mut CertCachePair,
) -> (
    PerceptualCertificationSummary,
    Option<PerceptualCertificationSummary>,
) {
    let n = stem.usable_len();
    let (ol, or) =
        render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
    let det_profile = TargetRepeatabilityProfile::deterministic();
    let ml: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
    let sl = certs
        .cert_l
        .certify_universal_with_prescreen(tier, &ml, &det_profile, None);
    let sr = certs.cert_r.as_mut().map(|cr| {
        let mr: Vec<f64> = or.iter().map(|&v| f64::from(v)).collect();
        cr.certify_universal_with_prescreen(tier, &mr, &det_profile, None)
    });
    (sl, sr)
}

fn eval_tier_passes_universal<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    certs: &mut CertCachePair,
    quality_mode: SolverQualityMode,
) -> bool {
    let Some([prof_l, prof_r]) = certs.rep_profiles else {
        return true;
    };
    let n = stem.usable_len();
    let (ol, or) =
        render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
    let ml: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
    let sl = certs
        .cert_l
        .certify_universal_with_prescreen(tier, &ml, &prof_l, None);
    let sr = certs.cert_r.as_mut().map(|cr| {
        let mr: Vec<f64> = or.iter().map(|&v| f64::from(v)).collect();
        cr.certify_universal_with_prescreen(tier, &mr, &prof_r, None)
    });
    let mode_ok = summary_passes_mode(&sl, quality_mode)
        && sr
            .as_ref()
            .is_none_or(|r| summary_passes_mode(r, quality_mode));
    if !mode_ok {
        return false;
    }
    if quality_mode != SolverQualityMode::BestUntilDiminishingReturns {
        return true;
    }
    sl.passes_target
        && sl.passes_hard_ceiling
        && sr.is_none_or(|r| r.passes_target && r.passes_hard_ceiling)
}

pub(super) fn db_beats_or_both_floor(lo: f64, hi: f64, margin: f64) -> bool {
    lo.is_finite() && hi.is_finite() && lo >= hi + margin
}

fn chan_beats_margin(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
    quality_mode: SolverQualityMode,
) -> bool {
    summary_passes_mode(lo, quality_mode)
        && db_beats_or_both_floor(lo.raw_besr_db, hi.raw_besr_db, 0.002)
        && (0..3).all(|b| db_beats_or_both_floor(lo.band_esr_db[b], hi.band_esr_db[b], 0.002))
        && lo.mr_iir >= hi.mr_iir + 0.0015
        && lo.ltas_max_db >= hi.ltas_max_db + 0.002
        && db_beats_or_both_floor(lo.seg_esr_db, hi.seg_esr_db, 0.002)
        && lo.ltas_excluded_bands == hi.ltas_excluded_bands
        && (0..33).all(|b| {
            !hi.ltas_excluded_bands[b]
                || lo.ltas_residual_band_power[b] >= hi.ltas_residual_band_power[b]
        })
}

fn model_perceptual_score(
    arr_l: &[PerceptualCertificationSummary; 4],
    arr_r: Option<&[PerceptualCertificationSummary; 4]>,
) -> f64 {
    let mut score = 0.0_f64;
    for t in 0..4 {
        let l = &arr_l[t];
        let (ltas, mr, besr) = match arr_r {
            Some(sr) => {
                let r = &sr[t];
                (
                    l.ltas_max_db.max(r.ltas_max_db),
                    l.mr_iir.max(r.mr_iir),
                    l.raw_besr_db.max(r.raw_besr_db),
                )
            }
            None => (l.ltas_max_db, l.mr_iir, l.raw_besr_db),
        };
        score += ltas + 0.25 * mr + 0.015 * besr;
    }
    score
}

/// Fits a `Stage IV` prime/comb FDN + `8192`-FFT cepstral init + Welch-33 minimax Stage III EQ
/// cascade for incoherent modulation/delay/FDN chains, maintaining strict 4-tier monotonicity.
#[inline(never)]
#[must_use]
pub(super) fn fit_incoherent_rig_cascade<B: TrainerBackend>(
    baseline: &V3MatryoshkaModel,
    dataset: &V3TrainDataset,
    mono_out: bool,
    stem0: &V3TrainStem,
) -> Option<V3MatryoshkaModel> {
    let sr_hz = dataset.sample_rate_hz;
    let n = stem0.usable_len();
    let warmup = stem0.warmup.max(canonical_besr_warmup(n));
    if n <= warmup + 128 {
        return None;
    }
    let tl_full: Vec<f64> = stem0.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
    let tr_full: Vec<f64> = stem0.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
    let tgt_rms = compute_broadband_rms(&stem0.tgt_l, &stem0.tgt_r, warmup, mono_out);
    if tgt_rms <= 1.0e-9 {
        return None;
    }
    let ltas_l = LtasTargetCache::new(&tl_full[warmup..], sr_hz, 20_000.0)?;
    let ltas_r = (!mono_out)
        .then(|| LtasTargetCache::new(&tr_full[warmup..], sr_hz, 20_000.0))
        .flatten();
    let base_sum_l = build_welch33_target_summary(&tl_full, warmup, sr_hz);
    let base_sum_r = (!mono_out).then(|| build_welch33_target_summary(&tr_full, warmup, sr_hz));
    let welch_pair = WelchSummaryPair {
        sum_l: &base_sum_l,
        sum_r: base_sum_r.as_ref(),
    };
    let wave_ctx = WaveTargetCtx::new(tl_full.clone(), tr_full.clone(), warmup, sr_hz);
    let mr_ctx = MrIirTargetCtx::new(&tl_full, &tr_full, warmup, sr_hz, mono_out);
    let mut certs = CertCachePair {
        cert_l: TargetChannelCertCache::new(&tl_full, sr_hz, warmup),
        cert_r: (!mono_out).then(|| TargetChannelCertCache::new(&tr_full, sr_hz, warmup)),
        rep_profiles: dataset.rep_profiles,
    };
    let grid = build_eq_freq_grid(sr_hz);
    let mut eq_ctx = ClosedLoopEqCtx {
        stem: stem0,
        warmup,
        sr_hz,
        grid: &grid,
        ltas_l,
        ltas_r,
    };

    let tracks = prepare_harmonic_tracks::<B>(baseline, dataset, stem0, warmup, tgt_rms, sr_hz);
    let comb_spec = detect_comb_spec(stem0, warmup, mono_out);
    let lfo_spec = lfo_detect::detect_lfo_modulation(
        stem0,
        warmup,
        sr_hz,
        mono_out,
        comb_spec.map(|c| (c.dl, c.dr)),
    );
    let effective_comb = comb_spec.or_else(|| {
        lfo_spec.map(|lfo| CombSpec {
            dl: (lfo.mean_delay_l.round() as usize).clamp(32, 32_000),
            dr: (lfo.mean_delay_r.round() as usize).clamp(32, 32_000),
            sigma_l: (lfo.depth_samples.round() as usize).clamp(4, 256),
            sigma_r: (lfo.depth_samples.round() as usize).clamp(4, 256),
            sl: lfo.sign_l,
            sr: lfo.sign_r,
            wr_l: 0.35,
            wr_r: 0.35,
        })
    });
    let comb_cands = effective_comb.map_or_else(Vec::new, |c| build_comb_candidates(c, lfo_spec));
    let mut sweep_check = SweepHarmonicCheck::from_dataset(dataset, mono_out);

    let mut finalists: Vec<UltraSearchBest> = Vec::new();
    let scr_limit = 131_072;
    if n > scr_limit {
        let stem_scr = slice_stem_prefix(stem0, scr_limit);
        let ns = stem_scr.usable_len();
        let ws = stem_scr.warmup;
        let tsl: Vec<f64> = stem_scr.tgt_l[..ns].iter().map(|&v| f64::from(v)).collect();
        let tsr: Vec<f64> = stem_scr.tgt_r[..ns].iter().map(|&v| f64::from(v)).collect();
        let sl_ltas = LtasTargetCache::new(&tsl[ws..], sr_hz, 20_000.0)?;
        let sr_ltas = (!mono_out)
            .then(|| LtasTargetCache::new(&tsr[ws..], sr_hz, 20_000.0))
            .flatten();
        let scr_sum_l = build_welch33_target_summary(&tsl, ws, sr_hz);
        let scr_sum_r = (!mono_out).then(|| build_welch33_target_summary(&tsr, ws, sr_hz));
        let scr_welch = WelchSummaryPair {
            sum_l: &scr_sum_l,
            sum_r: scr_sum_r.as_ref(),
        };
        let scr_wave = WaveTargetCtx::new(tsl.clone(), tsr.clone(), ws, sr_hz);
        let mut scr_certs = CertCachePair {
            cert_l: TargetChannelCertCache::new(&tsl, sr_hz, ws),
            cert_r: (!mono_out).then(|| TargetChannelCertCache::new(&tsr, sr_hz, ws)),
            rep_profiles: dataset.rep_profiles,
        };
        let mut scr_eq = ClosedLoopEqCtx {
            stem: &stem_scr,
            warmup: ws,
            sr_hz,
            grid: &grid,
            ltas_l: sl_ltas,
            ltas_r: sr_ltas,
        };
        let mut screen_ctx = ScreenTargetCtx {
            eq_ctx: &mut scr_eq,
            certs: &mut scr_certs,
            welch: &scr_welch,
            wave_ctx: &scr_wave,
            tgt_rms: compute_broadband_rms(&stem_scr.tgt_l, &stem_scr.tgt_r, ws, mono_out),
            sweep_check: sweep_check.as_mut(),
        };
        for track in &tracks {
            finalists.extend(screen_track_finalists::<B>(
                track,
                &comb_cands,
                &mut screen_ctx,
            ));
        }
    } else {
        let mut screen_ctx = ScreenTargetCtx {
            eq_ctx: &mut eq_ctx,
            certs: &mut certs,
            welch: &welch_pair,
            wave_ctx: &wave_ctx,
            tgt_rms,
            sweep_check: sweep_check.as_mut(),
        };
        for track in &tracks {
            finalists.extend(screen_track_finalists::<B>(
                track,
                &comb_cands,
                &mut screen_ctx,
            ));
        }
    }

    finalists.sort_by(|a, b| a.score.total_cmp(&b.score));
    let mut seen_keys: Vec<([usize; 4], i32, i32, i32, bool)> = Vec::new();
    let mut distinct_finalists: Vec<UltraSearchBest> = Vec::new();
    let mut best_lin_mod: Option<UltraSearchBest> = None;
    let mut best_lin_static: Option<UltraSearchBest> = None;
    let mut best_lin_diff: Option<UltraSearchBest> = None;
    let mut best_lin_any: Option<UltraSearchBest> = None;
    let mut best_lfo_any: Option<UltraSearchBest> = None;
    let mut best_diff_any: Option<UltraSearchBest> = None;
    let model_has_s2_drive = |m: &V3MatryoshkaModel| {
        m.readouts[3]
            .pre_cab_w
            .iter()
            .flatten()
            .any(|&w| w.abs() > 1.0e-6)
    };
    for f in finalists {
        let has_s2_drive = model_has_s2_drive(&f.track);
        let key = (
            f.tank.delays,
            (f.tank.wet_ratio * 200.0).round() as i32,
            (f.tank.g_loop * 100.0).round() as i32,
            (f.tank.lfo_depth_samples * 10.0).round() as i32,
            has_s2_drive,
        );
        if seen_keys.contains(&key) {
            continue;
        }
        seen_keys.push(key);
        let is_lfo = f.tank.lfo_depth_samples >= 1.5;
        let ok_ultra = f.score < 5.0;
        if !has_s2_drive {
            if best_lin_any.is_none() {
                best_lin_any = Some(f.clone());
            }
            if f.is_comb && is_lfo && best_lin_mod.is_none() {
                best_lin_mod = Some(f.clone());
            }
            if f.is_comb && !is_lfo && best_lin_static.is_none() {
                best_lin_static = Some(f.clone());
            }
            if ok_ultra && !f.is_comb && best_lin_diff.is_none() {
                best_lin_diff = Some(f.clone());
            }
        }
        if ok_ultra && is_lfo && best_lfo_any.is_none() {
            best_lfo_any = Some(f.clone());
        }
        if ok_ultra && !f.is_comb && best_diff_any.is_none() {
            best_diff_any = Some(f.clone());
        }
        if distinct_finalists.len() < 2 {
            distinct_finalists.push(f);
        }
    }
    let push_unique_finalist = |dst: &mut Vec<UltraSearchBest>,
                                cand_opt: Option<UltraSearchBest>| {
        if let Some(cand) = cand_opt
            && dst.len() < 5
        {
            let cand_s2 = model_has_s2_drive(&cand.track);
            let dup = dst.iter().any(|x| {
                x.tank.delays == cand.tank.delays && model_has_s2_drive(&x.track) == cand_s2
            });
            if !dup {
                dst.push(cand);
            }
        }
    };
    if !distinct_finalists.iter().any(|f| !f.is_comb) {
        push_unique_finalist(&mut distinct_finalists, best_lin_diff.or(best_diff_any));
    }
    if !distinct_finalists
        .iter()
        .any(|f| !model_has_s2_drive(&f.track) && f.tank.lfo_depth_samples >= 1.5)
    {
        push_unique_finalist(&mut distinct_finalists, best_lin_mod);
    }
    if !distinct_finalists
        .iter()
        .any(|f| !model_has_s2_drive(&f.track) && f.tank.lfo_depth_samples < 1.5)
    {
        push_unique_finalist(&mut distinct_finalists, best_lin_static);
    }
    if tracks.len() > 1
        && !distinct_finalists
            .iter()
            .any(|f| !model_has_s2_drive(&f.track))
    {
        push_unique_finalist(&mut distinct_finalists, best_lin_any);
    }
    if lfo_spec.is_some()
        && !distinct_finalists
            .iter()
            .any(|f| f.tank.lfo_depth_samples >= 1.5)
    {
        push_unique_finalist(&mut distinct_finalists, best_lfo_any);
    }

    let quality_mode = dataset.quality_mode;
    let lfo_active = lfo_spec.is_some();
    let (ultra_goal_db, ultra_iters, early_fast) = match quality_mode {
        SolverQualityMode::BestUntilDiminishingReturns => (0.0_f64, 68_usize, false),
        SolverQualityMode::ThresholdStop {
            max_polish_iters, ..
        } => (0.008_f64, max_polish_iters.clamp(8, 64), true),
    };
    let mut best_certified: Option<(f64, V3MatryoshkaModel)> = None;
    let mut passing_count = 0_usize;
    let anchor = VisualProbeAnchor::from_dataset(dataset);
    let mut emitted_ultra = false;

    let zero_tilt = [[0.0_f64; 33]; 2];
    for best in distinct_finalists {
        let base_tank = best.tank;
        let mut model = best.track;
        model.stage4[3] = base_tank.to_params();
        let mut db_fin = best.db;
        if n > scr_limit {
            eq_ctx.run_tier::<B>(
                &mut model,
                &EqRunSpec {
                    tier: Tier::Ultra,
                    target_rms: tgt_rms,
                    tilt_db: &zero_tilt,
                    passes: 2,
                    step_gain: 0.90,
                },
                &mut db_fin,
            );
        } else {
            model = best.model;
        }
        {
            let ultra_wave_pull = if lfo_active && base_tank.lfo_depth_samples >= 1.5 {
                0.015
            } else {
                0.010
            };
            let (wave_u_l, wave_u_r) = wave_ctx.make_specs(ultra_wave_pull, None, None);
            polish_tier_welch33::<B>(
                &mut model,
                &WelchPolishSpec {
                    tier: Tier::Ultra,
                    stem: stem0,
                    sum_l: &base_sum_l,
                    sum_r: base_sum_r.as_ref(),
                    wave_l: Some(&wave_u_l),
                    wave_r: (!mono_out).then_some(&wave_u_r),
                    mr_l: None,
                    mr_r: None,
                    lock_rms_lr: None,
                    goal_db: ultra_goal_db,
                    iters: ultra_iters,
                },
            );
            let (ul, ur) = eval_tier_cert::<B>(&model, Tier::Ultra, stem0, &mut certs);
            if ul.mr_iir > 0.76 || ur.as_ref().is_some_and(|r| r.mr_iir > 0.76) {
                let mr_u_pair = mr_ctx
                    .as_ref()
                    .map(|mc| mc.make_specs(0.08, Some((0.0, 0.76)), Some((0.0, 0.76))));
                polish_tier_welch33::<B>(
                    &mut model,
                    &WelchPolishSpec {
                        tier: Tier::Ultra,
                        stem: stem0,
                        sum_l: &base_sum_l,
                        sum_r: base_sum_r.as_ref(),
                        wave_l: Some(&wave_u_l),
                        wave_r: (!mono_out).then_some(&wave_u_r),
                        mr_l: mr_u_pair.as_ref().and_then(|p| p.0.as_ref()),
                        mr_r: if mono_out {
                            None
                        } else {
                            mr_u_pair.as_ref().and_then(|p| p.1.as_ref())
                        },
                        lock_rms_lr: None,
                        goal_db: ultra_goal_db,
                        iters: 24,
                    },
                );
            }
            if !emitted_ultra && dataset.progress.is_active() {
                emitted_ultra = true;
                let (u_l, u_r) = eval_tier_cert::<B>(&model, Tier::Ultra, stem0, &mut certs);
                emit_stage4_summary_step::<B>(
                    &dataset.progress,
                    anchor.as_ref(),
                    Stage4SummaryStep {
                        model: &model,
                        stem0,
                        tier: Tier::Ultra,
                        permille: 420,
                        step: 2,
                        total_steps: 5,
                        sl: &u_l,
                        sr: u_r.as_ref(),
                    },
                );
            }
        }
        let mut cascade_ctx = CascadeCtx {
            db_fin: &db_fin,
            eq_ctx: &mut eq_ctx,
            welch: &welch_pair,
            wave_ctx: &wave_ctx,
            mr_ctx: mr_ctx.as_ref(),
            certs: &mut certs,
            tgt_rms,
            lfo_active,
            quality_mode,
            progress: &dataset.progress,
            anchor: anchor.as_ref(),
        };
        if let Some((arr_l, arr_r)) =
            cascade_down_tiers::<B>(&mut model, base_tank, &mut cascade_ctx)
        {
            let all_pass = arr_l.iter().all(|s| summary_passes_mode(s, quality_mode))
                && arr_r
                    .as_ref()
                    .is_none_or(|sr| sr.iter().all(|s| summary_passes_mode(s, quality_mode)));
            if all_pass && summaries_monotone(&arr_l, arr_r.as_ref()) {
                let sc = model_perceptual_score(&arr_l, arr_r.as_ref());
                if early_fast {
                    return Some(model);
                }
                if best_certified.as_ref().is_none_or(|(b, _)| sc < *b) {
                    best_certified = Some((sc, model));
                }
                passing_count += 1;
                let ultra_excellent = arr_l[3].ltas_max_db <= 0.075
                    && arr_r.as_ref().is_none_or(|sr| sr[3].ltas_max_db <= 0.075)
                    && arr_l[2].ltas_max_db <= 0.140
                    && arr_r.as_ref().is_none_or(|sr| sr[2].ltas_max_db <= 0.140);
                if ultra_excellent || passing_count >= 2 {
                    break;
                }
            }
        }
    }
    best_certified.map(|(_, m)| m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::certification::verify_monotone_4tier_summaries;
    use crate::metrics::companion::canonical_ltas_33_centers_hz;
    use crate::solver::backends::ScalarTrainerBackend;
    use std::f64::consts::{FRAC_PI_4, TAU};

    #[test]
    fn test_incoherent_rig_cepstral_eq_and_monotone_cascade() {
        let n = 16_384;
        let sr_hz = 48_000.0;
        let centers = canonical_ltas_33_centers_hz();
        let mut dry = Vec::with_capacity(n);
        let mut seed: u64 = 0x517C_C1B7_2722_0A95;
        for i in 0..n {
            let t = (i as f64) / sr_hz;
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let white = ((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            let mut multi = 0.0_f64;
            for (k, &fc) in centers.iter().enumerate() {
                multi += (TAU * fc * t + 0.37 * (k as f64)).cos();
            }
            dry.push((0.18 * white + 0.028 * multi) as f32);
        }

        let read_frac = |buf: &[f32], pos: usize, delay: f64| -> f64 {
            let p = (pos as f64) - delay;
            if p <= 0.0 {
                return 0.0;
            }
            let idx = p.floor() as usize;
            let frac = p - (idx as f64);
            let x0 = f64::from(buf[idx]);
            let x1 = f64::from(buf.get(idx + 1).copied().unwrap_or(0.0));
            x0 + frac * (x1 - x0)
        };
        let mut tgt_l = Vec::with_capacity(n);
        let mut tgt_r = Vec::with_capacity(n);
        let (mut lp_l, mut lp_r) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let t = (i as f64) / sr_hz;
            let dl = 340.0 + 45.0 * (TAU * 1.4 * t).sin();
            let dr = 340.0 + 45.0 * (TAU * 1.4 * t + FRAC_PI_4).sin();
            let xl = 0.70 * f64::from(dry[i]) + 0.38 * read_frac(&dry, i, dl);
            let xr = 0.70 * f64::from(dry[i]) + 0.38 * read_frac(&dry, i, dr);
            lp_l = 0.32 * lp_l + 0.68 * xl;
            lp_r = 0.32 * lp_r + 0.68 * xr;
            tgt_l.push(lp_l as f32);
            tgt_r.push(lp_r as f32);
        }

        let stem0 = V3TrainStem {
            in_l: dry.clone(),
            in_r: dry,
            tgt_l: tgt_l.clone(),
            tgt_r: tgt_r.clone(),
            warmup: 1_024,
        };
        let ds = V3TrainDataset::new(vec![stem0.clone()], sr_hz);
        let base = V3MatryoshkaModel::dual_rate_default(false);
        let fitted = fit_incoherent_rig_cascade::<ScalarTrainerBackend>(&base, &ds, false, &stem0)
            .expect("Incoherent rig cascade fits spectral-matched monotone FDN");
        assert!(fitted.stage4[3].wet_mix > 0.0 || fitted.readouts[3].fir_l[0].abs() > 0.05);
        assert!(
            fitted.stage4[3].lfo_depth_samples > 5.0,
            "expected modulated Stage IV LFO depth > 5.0 samples, got {}",
            fitted.stage4[3].lfo_depth_samples
        );
        assert!(
            fitted.stage4[3].lfo_rate_hz > 0.5,
            "expected modulated Stage IV LFO rate > 0.5 Hz, got {}",
            fitted.stage4[3].lfo_rate_hz
        );

        let tl64: Vec<f64> = tgt_l.iter().map(|&v| f64::from(v)).collect();
        let tr64: Vec<f64> = tgt_r.iter().map(|&v| f64::from(v)).collect();
        let mut certs = CertCachePair {
            cert_l: TargetChannelCertCache::new(&tl64, sr_hz, 1_024),
            cert_r: Some(TargetChannelCertCache::new(&tr64, sr_hz, 1_024)),
            rep_profiles: None,
        };
        let mut sl_vec = Vec::with_capacity(4);
        let mut sr_vec = Vec::with_capacity(4);
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let (sl, sr_opt) =
                eval_tier_cert::<ScalarTrainerBackend>(&fitted, tier, &stem0, &mut certs);
            assert!(cascade::summary_passes_mode(
                &sl,
                crate::solver::SolverQualityMode::BestUntilDiminishingReturns
            ));
            let sr_val = sr_opt.unwrap();
            assert!(cascade::summary_passes_mode(
                &sr_val,
                crate::solver::SolverQualityMode::BestUntilDiminishingReturns
            ));
            sl_vec.push(sl);
            sr_vec.push(sr_val);
        }
        let sl_arr: [PerceptualCertificationSummary; 4] = sl_vec.try_into().unwrap();
        let sr_arr: [PerceptualCertificationSummary; 4] = sr_vec.try_into().unwrap();
        assert!(verify_monotone_4tier_summaries(&sl_arr));
        assert!(verify_monotone_4tier_summaries(&sr_arr));
    }
}
