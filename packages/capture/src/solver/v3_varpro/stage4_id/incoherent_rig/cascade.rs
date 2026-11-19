#![forbid(unsafe_code)]
//! Top-down 4-tier (`Ultra -> Pro -> Full -> Lite`) monotone FDN + Welch-33 cascade
//! (`incoherent_rig/cascade.rs`).

mod cands;
mod fallback;

use dnam_engine::Tier;
use dnam_engine::stages::V3MatryoshkaModel;

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::SolverQualityMode;
use crate::solver::backends::TrainerBackend;
use crate::solver::progress::TrainerProgressHandle;
use crate::solver::progress::telemetry::{
    Stage4SummaryStep, VisualProbeAnchor, emit_stage4_summary_step,
};
use crate::solver::v3_varpro::primitives::par_map_indexed;

use super::comb_detect::{FdnTankSpec, compute_broadband_rms, snap_stage4_broadband_rms};
use super::mr_iir_adj::MrIirTargetCtx;
use super::wave_hinge::WaveTargetCtx;
use super::welch_eq::{WelchPolishSpec, polish_tier_welch33, scale_summary_band_tilt};
use super::{
    CertCachePair, ClosedLoopEqCtx, EqRunSpec, WelchSummaryPair, eval_tier_cert,
    eval_tier_passes_universal, render_tier_stereo,
};
use cands::{CandFallbacks, build_tier_fdn_cands, eval_cand_rank, max_tier_delay_for_tau};
use fallback::{
    ChosenTierOut, DetuneFallbackArgs, PhaseSweepArgs, pair_beats_margin, rms_within_gate,
    sweep_unitary_phase_margin, try_phase_detune_fallback,
};

const RMS_OFFSET_DB: [f64; 4] = [0.006, 0.004, 0.002, 0.0];
const STEP_LTAS_DB: [f64; 4] = [0.095, 0.070, 0.028, 0.0];
const MAX_LTAS_CEIL: [f64; 4] = [0.75, 0.35, 0.15, 0.10];
const WAVE_PULL: [f64; 4] = [0.005, 0.008, 0.015, 0.020];

pub(super) fn summary_passes_mode(
    s: &PerceptualCertificationSummary,
    mode: SolverQualityMode,
) -> bool {
    let slack = match mode {
        SolverQualityMode::BestUntilDiminishingReturns => 0.0,
        SolverQualityMode::ThresholdStop {
            target_slack_milli_db,
            ..
        } => f64::from(target_slack_milli_db) * 1.0e-3,
    };
    let th = TierQualityThresholds::for_tier(s.tier);
    let tier_mr_ceil = match s.tier {
        Tier::Lite => 0.95,
        Tier::Full => 0.90,
        Tier::Pro => 0.85,
        Tier::Ultra => 0.80,
    };
    let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
    s.ltas_max_db.is_finite()
        && s.ltas_max_db <= th.max_ltas_db + slack
        && s.mr_iir.is_finite()
        && s.mr_iir >= 0.0
        && s.mr_iir <= tier_mr_ceil + slack
        && valid_db(s.seg_esr_db)
        && s.seg_esr_db <= 6.0 + slack
        && valid_db(s.raw_besr_db)
        && s.band_esr_db.iter().copied().all(valid_db)
        && !s.ltas_excluded_bands.iter().all(|&ex| ex)
}

fn channel_ms(sig: &[f32], warmup: usize, n: usize) -> f64 {
    let hi = n.min(sig.len());
    if hi <= warmup {
        return 0.0;
    }
    let cnt = (hi - warmup) as f64;
    sig[warmup..hi]
        .iter()
        .map(|&v| {
            let x = f64::from(v);
            x * x
        })
        .sum::<f64>()
        / cnt
}

fn tier_passes_all(
    sl: &PerceptualCertificationSummary,
    sr: Option<&PerceptualCertificationSummary>,
    mode: SolverQualityMode,
) -> bool {
    summary_passes_mode(sl, mode) && sr.is_none_or(|r| summary_passes_mode(r, mode))
}

pub(super) struct CascadeCtx<'a, 'b> {
    pub(super) db_fin: &'b [[f64; 33]; 2],
    pub(super) eq_ctx: &'b mut ClosedLoopEqCtx<'a>,
    pub(super) welch: &'b WelchSummaryPair<'a>,
    pub(super) wave_ctx: &'b WaveTargetCtx,
    pub(super) mr_ctx: Option<&'b MrIirTargetCtx>,
    pub(super) certs: &'b mut CertCachePair,
    pub(super) tgt_rms: f64,
    pub(super) lfo_active: bool,
    pub(super) quality_mode: SolverQualityMode,
    pub(super) progress: &'b TrainerProgressHandle,
    pub(super) anchor: Option<&'b VisualProbeAnchor>,
}

pub(super) fn cascade_down_tiers<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    base_tank: FdnTankSpec,
    ctx: &mut CascadeCtx<'_, '_>,
) -> Option<(
    [PerceptualCertificationSummary; 4],
    Option<[PerceptualCertificationSummary; 4]>,
)> {
    let stem0 = ctx.eq_ctx.stem;
    let warmup = ctx.eq_ctx.warmup;
    let mono_out = model.mono_out;
    let quality_mode = ctx.quality_mode;
    let (polish_goal_db, polish_iters, early_stop) = match quality_mode {
        SolverQualityMode::BestUntilDiminishingReturns => (0.0_f64, 56_usize, false),
        SolverQualityMode::ThresholdStop {
            max_polish_iters, ..
        } => (0.008_f64, max_polish_iters.clamp(8, 64), true),
    };
    let (mut ultra_l, mut ultra_r) = eval_tier_cert::<B>(model, Tier::Ultra, stem0, ctx.certs);
    let mut ultra_univ_ok =
        eval_tier_passes_universal::<B>(model, Tier::Ultra, stem0, ctx.certs, quality_mode);
    if (!tier_passes_all(&ultra_l, ultra_r.as_ref(), quality_mode) || !ultra_univ_ok)
        && ultra_l.mr_iir <= 0.80
        && ultra_l.seg_esr_db <= 6.0
        && ultra_r
            .as_ref()
            .is_none_or(|r| r.mr_iir <= 0.80 && r.seg_esr_db <= 6.0)
    {
        polish_tier_welch33::<B>(
            model,
            &WelchPolishSpec {
                tier: Tier::Ultra,
                stem: stem0,
                sum_l: ctx.welch.sum_l,
                sum_r: ctx.welch.sum_r,
                wave_l: None,
                wave_r: None,
                mr_l: None,
                mr_r: None,
                lock_rms_lr: None,
                goal_db: polish_goal_db,
                iters: polish_iters,
            },
        );
        (ultra_l, ultra_r) = eval_tier_cert::<B>(model, Tier::Ultra, stem0, ctx.certs);
        ultra_univ_ok =
            eval_tier_passes_universal::<B>(model, Tier::Ultra, stem0, ctx.certs, quality_mode);
    }
    if !tier_passes_all(&ultra_l, ultra_r.as_ref(), quality_mode) || !ultra_univ_ok {
        return None;
    }

    let active_bands: Vec<usize> = (0..33)
        .filter(|&b| ctx.welch.sum_l.active[b] && ctx.welch.sum_r.is_none_or(|sr| sr.active[b]))
        .collect();
    let tail_start = active_bands.len().saturating_sub(6);
    let tail_active = &active_bands[tail_start..];
    let n_u = stem0.usable_len();
    let (ol_u, or_u) = render_tier_stereo::<B>(
        &model.extract_tier(Tier::Ultra),
        &stem0.in_l[..n_u],
        &stem0.in_r[..n_u],
    );
    let ss_u_l = channel_ms(&ol_u, warmup, n_u);
    let ss_u_r = channel_ms(&or_u, warmup, n_u);
    let ultra_total_raw = compute_broadband_rms(&ol_u, &or_u, warmup, mono_out);
    let ultra_total_rms = if ultra_total_raw > 1.0e-9 {
        ultra_total_raw
    } else {
        ctx.tgt_rms
    };

    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut sum_l: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, Some(ultra_l)];
    let mut sum_r: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, ultra_r];
    let mut cur_tank = base_tank;
    let mut hi_rms = ultra_total_rms;

    for &tau in &[2_usize, 1, 0] {
        let tier = tiers[tau];
        let hi_l = sum_l[tau + 1].as_ref()?;
        let hi_r = sum_r[tau + 1].as_ref();
        let hi_ltas = hi_l.ltas_max_db.max(hi_r.map_or(0.0, |r| r.ltas_max_db));
        let max_ltas_frac = if tau == 2 { 0.76 } else { 0.82 };
        let amp_ltas = (hi_ltas + STEP_LTAS_DB[tau])
            .min(MAX_LTAS_CEIL[tau] * max_ltas_frac)
            .max((hi_ltas + 0.018).min(MAX_LTAS_CEIL[tau] * 0.90));
        let mut tilt = [0.0_f64; 33];
        let m_tail = tail_active.len().max(1) as f64;
        for (k, &b) in tail_active.iter().enumerate() {
            tilt[b] = -amp_ltas * ((k + 1) as f64) / m_tail;
        }
        let neg_tilt = tilt.map(|x| -x);
        let pos_tilt_lr = [neg_tilt, neg_tilt];
        let tilted_sum_l = scale_summary_band_tilt(ctx.welch.sum_l, &tilt);
        let tilted_sum_r = ctx.welch.sum_r.map(|sr| scale_summary_band_tilt(sr, &tilt));

        const WAVE_MARGIN_DB: [f64; 3] = [0.038, 0.045, 0.018];
        let w_off = WAVE_MARGIN_DB[tau];
        let lo_l_bounds = (
            hi_l.band_esr_db
                .map(|db| 10.0_f64.powf((db + w_off) / 10.0)),
            10.0_f64.powf((hi_l.raw_besr_db + w_off) / 10.0),
            10.0_f64.powf((hi_l.seg_esr_db + w_off) / 10.0),
        );
        let lo_r_bounds = hi_r.map(|hr| {
            (
                hr.band_esr_db.map(|db| 10.0_f64.powf((db + w_off) / 10.0)),
                10.0_f64.powf((hr.raw_besr_db + w_off) / 10.0),
                10.0_f64.powf((hr.seg_esr_db + w_off) / 10.0),
            )
        });
        let wave_pull = if ctx.lfo_active && cur_tank.lfo_depth_samples >= 1.5 {
            WAVE_PULL[tau]
        } else {
            0.5 * WAVE_PULL[tau]
        };
        let (wave_l, wave_r) = ctx
            .wave_ctx
            .make_specs(wave_pull, Some(lo_l_bounds), lo_r_bounds);

        let gain_sq = 10.0_f64.powf(RMS_OFFSET_DB[tau] / 10.0);
        let target_t_total_rms = ultra_total_rms * gain_sq.sqrt();
        let ch_l_t_sq = (if ss_u_l > 1.0e-18 {
            ss_u_l
        } else {
            ultra_total_rms * ultra_total_rms
        }) * gain_sq;
        let ch_r_t_sq = (if ss_u_r > 1.0e-18 {
            ss_u_r
        } else {
            ultra_total_rms * ultra_total_rms
        }) * gain_sq;
        let fdn_cands = build_tier_fdn_cands(cur_tank, tau);
        let tier_max_ltas = crate::metrics::TierQualityThresholds::for_tier(tier).max_ltas_db;
        let tier_mr_ceil: f64 = match tier {
            Tier::Lite => 0.95,
            Tier::Full => 0.90,
            Tier::Pro => 0.85,
            Tier::Ultra => 0.80,
        };
        let mr_pair = ctx.mr_ctx.map(|mc| {
            let lo_mr_l = hi_l.mr_iir + 0.005;
            let hi_mr_l = (tier_mr_ceil - 0.012).max(lo_mr_l + 0.006);
            let b_r = hi_r.map(|hr| {
                let lo_mr_r = hr.mr_iir + 0.005;
                let hi_mr_r = (tier_mr_ceil - 0.012).max(lo_mr_r + 0.006);
                (lo_mr_r, hi_mr_r)
            });
            mc.make_specs(0.0, Some((lo_mr_l, hi_mr_l)), b_r)
        });
        let mr_l = mr_pair.as_ref().and_then(|p| p.0.as_ref());
        let mr_r = if mono_out {
            None
        } else {
            mr_pair.as_ref().and_then(|p| p.1.as_ref())
        };

        let mut chosen = None;
        let mut fallbacks = CandFallbacks::default();
        let mut cand_jobs: Vec<(FdnTankSpec, usize, bool)> = fdn_cands
            .iter()
            .map(|&(tank_spec, eq_passes)| (tank_spec, eq_passes, false))
            .collect();
        for &(tank_spec, _) in &fdn_cands {
            if tank_spec.delays[0].abs_diff(cur_tank.delays[0]) <= 2
                && (tank_spec.wet_ratio - cur_tank.wet_ratio).abs() <= 0.14
                && (tank_spec.g_loop > 1.0e-5) == (cur_tank.g_loop > 1.0e-5)
            {
                cand_jobs.push((tank_spec, 0, true));
            }
        }
        let mut prescreened = Vec::with_capacity(cand_jobs.len());
        let pre_results = par_map_indexed(cand_jobs.len(), |idx| {
            let &(tank_spec, eq_passes, warm_from_hi) = &cand_jobs[idx];
            let mut trial = model.clone();
            trial.stage4[tau] = tank_spec.to_params();
            if warm_from_hi {
                let hi_ro = trial.readouts[tau + 1].clone();
                trial.readouts[tau].fir_l = hi_ro.fir_l;
                trial.readouts[tau].fir_r = hi_ro.fir_r;
                trial.readouts[tau].kautz_c_l = hi_ro.kautz_c_l;
                trial.readouts[tau].kautz_c_r = hi_ro.kautz_c_r;
                if trial.tier_os2x[tau] == trial.tier_os2x[tau + 1] {
                    trial.readouts[tau].direct_s2_w = hi_ro.direct_s2_w;
                }
                trial.readouts[tau].mask_to_tier(tier);
            } else {
                let cross_rate = trial.tier_os2x[tau] != trial.tier_os2x[3];
                let reset_db = cross_rate
                    || tank_spec.delays != cur_tank.delays
                    || (tank_spec.wet_ratio - cur_tank.wet_ratio).abs() > 0.04;
                let zero_tilt_lr = [[0.0_f64; 33]; 2];
                let mut db_try = if reset_db {
                    if cross_rate {
                        zero_tilt_lr
                    } else {
                        pos_tilt_lr
                    }
                } else {
                    let mut d = *ctx.db_fin;
                    for b in 0..33 {
                        d[0][b] = (d[0][b] + neg_tilt[b]).clamp(-22.0, 22.0);
                        d[1][b] = (d[1][b] + neg_tilt[b]).clamp(-22.0, 22.0);
                    }
                    d
                };
                let mut local_eq = ctx.eq_ctx.clone();
                local_eq.run_tier::<B>(
                    &mut trial,
                    &EqRunSpec {
                        tier,
                        target_rms: target_t_total_rms,
                        tilt_db: if cross_rate {
                            &zero_tilt_lr
                        } else {
                            &pos_tilt_lr
                        },
                        passes: if cross_rate {
                            eq_passes.max(4)
                        } else {
                            eq_passes
                        },
                        step_gain: if reset_db { 0.90 } else { 0.85 },
                    },
                    &mut db_try,
                );
            }
            let _ = snap_stage4_broadband_rms::<B>(
                &mut trial,
                tier,
                stem0,
                warmup,
                target_t_total_rms,
                mono_out,
            );
            let mut local_certs = ctx.certs.clone();
            let (sl, sr_opt) = eval_tier_cert::<B>(&trial, tier, stem0, &mut local_certs);
            let univ_ok =
                eval_tier_passes_universal::<B>(&trial, tier, stem0, &mut local_certs, quality_mode);
            let cand_rms = if pair_beats_margin(&sl, sr_opt.as_ref(), hi_l, hi_r, quality_mode)
                && univ_ok
            {
                rms_within_gate::<B>(
                    &trial,
                    tier,
                    stem0,
                    warmup,
                    target_t_total_rms,
                    hi_rms,
                    mono_out,
                )
            } else {
                None
            };
            let sc = eval_cand_rank(
                &sl,
                sr_opt.as_ref(),
                hi_l,
                hi_r,
                tier_mr_ceil,
                tier_max_ltas,
            );
            (trial, tank_spec, sl, sr_opt, cand_rms, univ_ok, sc)
        });
        let mut best_pre_score = f64::INFINITY;
        for (trial, tank_spec, sl, sr_opt, cand_rms_opt, univ_ok, sc) in pre_results {
            if let Some(cand_rms) = cand_rms_opt {
                let pick_score = sc.rank
                    + sl.ltas_max_db
                        .max(sr_opt.as_ref().map_or(0.0, |r| r.ltas_max_db));
                if early_stop || pick_score < best_pre_score {
                    best_pre_score = pick_score;
                    chosen = Some((
                        trial.readouts[tau].clone(),
                        trial.stage4[tau],
                        tank_spec,
                        sl,
                        sr_opt,
                        cand_rms,
                    ));
                    if early_stop {
                        break;
                    }
                }
                continue;
            }
            let rank = if univ_ok { sc.rank } else { sc.rank + 100.0 };
            if univ_ok {
                fallbacks.record(trial.clone(), tank_spec, sl, sr_opt, sc, quality_mode);
            }
            prescreened.push((rank, trial, tank_spec));
        }

        if chosen.is_none() {
            prescreened.sort_by(|a, b| a.0.total_cmp(&b.0));
            let compact_limit = max_tier_delay_for_tau(tau);
            let top_has_compact = prescreened.iter().take(2).any(|(_, _, spec)| {
                spec.delays.iter().copied().max().unwrap_or(0) <= compact_limit
            });
            let mut polish_list: Vec<_> = prescreened.iter().take(2).cloned().collect();
            if !top_has_compact
                && let Some(compact_entry) = prescreened
                    .iter()
                    .skip(2)
                    .find(|(_, _, spec)| {
                        spec.delays.iter().copied().max().unwrap_or(0) <= compact_limit
                    })
                    .cloned()
            {
                polish_list.push(compact_entry);
            } else if let Some(third) = prescreened.get(2).cloned() {
                polish_list.push(third);
            }
            let mut best_pol_score = f64::INFINITY;
            let mut record_chosen = |out: ChosenTierOut| -> bool {
                let pick_ltas = out
                    .3
                    .ltas_max_db
                    .max(out.4.as_ref().map_or(0.0, |r| r.ltas_max_db));
                if early_stop || pick_ltas < best_pol_score {
                    best_pol_score = pick_ltas;
                    chosen = Some(out);
                }
                early_stop || pick_ltas <= tier_max_ltas * 0.65
            };
            for (_, mut trial, tank_spec) in polish_list {
                polish_tier_welch33::<B>(
                    &mut trial,
                    &WelchPolishSpec {
                        tier,
                        stem: stem0,
                        sum_l: &tilted_sum_l,
                        sum_r: tilted_sum_r.as_ref(),
                        wave_l: Some(&wave_l),
                        wave_r: (!mono_out).then_some(&wave_r),
                        mr_l,
                        mr_r,
                        lock_rms_lr: Some((target_t_total_rms, ch_l_t_sq, ch_r_t_sq)),
                        goal_db: polish_goal_db,
                        iters: polish_iters,
                    },
                );
                let _ = snap_stage4_broadband_rms::<B>(
                    &mut trial,
                    tier,
                    stem0,
                    warmup,
                    target_t_total_rms,
                    mono_out,
                );
                let (mut sl, mut sr_opt) = eval_tier_cert::<B>(&trial, tier, stem0, ctx.certs);
                let mut univ_ok =
                    eval_tier_passes_universal::<B>(&trial, tier, stem0, ctx.certs, quality_mode);
                if pair_beats_margin(&sl, sr_opt.as_ref(), hi_l, hi_r, quality_mode)
                    && univ_ok
                    && let Some(cand_rms) = rms_within_gate::<B>(
                        &trial,
                        tier,
                        stem0,
                        warmup,
                        target_t_total_rms,
                        hi_rms,
                        mono_out,
                    )
                {
                    let done = record_chosen((
                        trial.readouts[tau].clone(),
                        trial.stage4[tau],
                        tank_spec,
                        sl,
                        sr_opt,
                        cand_rms,
                    ));
                    if done {
                        break;
                    }
                    continue;
                }
                let bad_mr_l = sl.mr_iir < hi_l.mr_iir + 0.002 || sl.mr_iir > tier_mr_ceil;
                let bad_mr_r = !mono_out
                    && sr_opt.as_ref().zip(hi_r).is_some_and(|(sr, hr)| {
                        sr.mr_iir < hr.mr_iir + 0.002 || sr.mr_iir > tier_mr_ceil
                    });
                if !bad_mr_l
                    && !bad_mr_r
                    && let Some(hit) = sweep_unitary_phase_margin::<B>(
                        &trial,
                        &PhaseSweepArgs {
                            tier,
                            tank_spec,
                            target_t_ac_rms: target_t_total_rms,
                            hi_ac_rms: hi_rms,
                            sl0: &sl,
                            sr0: sr_opt.as_ref(),
                            hi_l,
                            hi_r,
                        },
                        ctx,
                    )
                {
                    if record_chosen(hit) {
                        break;
                    }
                    continue;
                }
                let ltas_over = sl.ltas_max_db > tier_max_ltas
                    || sr_opt
                        .as_ref()
                        .is_some_and(|r| r.ltas_max_db > tier_max_ltas);
                if ltas_over {
                    polish_tier_welch33::<B>(
                        &mut trial,
                        &WelchPolishSpec {
                            tier,
                            stem: stem0,
                            sum_l: ctx.welch.sum_l,
                            sum_r: ctx.welch.sum_r,
                            wave_l: Some(&wave_l),
                            wave_r: (!mono_out).then_some(&wave_r),
                            mr_l,
                            mr_r,
                            lock_rms_lr: Some((target_t_total_rms, ch_l_t_sq, ch_r_t_sq)),
                            goal_db: polish_goal_db,
                            iters: polish_iters,
                        },
                    );
                    let _ = snap_stage4_broadband_rms::<B>(
                        &mut trial,
                        tier,
                        stem0,
                        warmup,
                        target_t_total_rms,
                        mono_out,
                    );
                    (sl, sr_opt) = eval_tier_cert::<B>(&trial, tier, stem0, ctx.certs);
                    univ_ok = eval_tier_passes_universal::<B>(
                        &trial,
                        tier,
                        stem0,
                        ctx.certs,
                        quality_mode,
                    );
                    if pair_beats_margin(&sl, sr_opt.as_ref(), hi_l, hi_r, quality_mode)
                        && univ_ok
                        && let Some(cand_rms) = rms_within_gate::<B>(
                            &trial,
                            tier,
                            stem0,
                            warmup,
                            target_t_total_rms,
                            hi_rms,
                            mono_out,
                        )
                    {
                        let done = record_chosen((
                            trial.readouts[tau].clone(),
                            trial.stage4[tau],
                            tank_spec,
                            sl,
                            sr_opt,
                            cand_rms,
                        ));
                        if done {
                            break;
                        }
                        continue;
                    }
                    if let Some(hit) = sweep_unitary_phase_margin::<B>(
                        &trial,
                        &PhaseSweepArgs {
                            tier,
                            tank_spec,
                            target_t_ac_rms: target_t_total_rms,
                            hi_ac_rms: hi_rms,
                            sl0: &sl,
                            sr0: sr_opt.as_ref(),
                            hi_l,
                            hi_r,
                        },
                        ctx,
                    ) {
                        if record_chosen(hit) {
                            break;
                        }
                        continue;
                    }
                    let still_ltas_over = sl.ltas_max_db > tier_max_ltas
                        || sr_opt
                            .as_ref()
                            .is_some_and(|r| r.ltas_max_db > tier_max_ltas);
                    if still_ltas_over {
                        polish_tier_welch33::<B>(
                            &mut trial,
                            &WelchPolishSpec {
                                tier,
                                stem: stem0,
                                sum_l: ctx.welch.sum_l,
                                sum_r: ctx.welch.sum_r,
                                wave_l: None,
                                wave_r: None,
                                mr_l: None,
                                mr_r: None,
                                lock_rms_lr: Some((target_t_total_rms, ch_l_t_sq, ch_r_t_sq)),
                                goal_db: polish_goal_db,
                                iters: 48,
                            },
                        );
                        let _ = snap_stage4_broadband_rms::<B>(
                            &mut trial,
                            tier,
                            stem0,
                            warmup,
                            target_t_total_rms,
                            mono_out,
                        );
                        (sl, sr_opt) = eval_tier_cert::<B>(&trial, tier, stem0, ctx.certs);
                        univ_ok = eval_tier_passes_universal::<B>(
                            &trial,
                            tier,
                            stem0,
                            ctx.certs,
                            quality_mode,
                        );
                        if pair_beats_margin(&sl, sr_opt.as_ref(), hi_l, hi_r, quality_mode)
                            && univ_ok
                            && let Some(cand_rms) = rms_within_gate::<B>(
                                &trial,
                                tier,
                                stem0,
                                warmup,
                                target_t_total_rms,
                                hi_rms,
                                mono_out,
                            )
                        {
                            let done = record_chosen((
                                trial.readouts[tau].clone(),
                                trial.stage4[tau],
                                tank_spec,
                                sl,
                                sr_opt,
                                cand_rms,
                            ));
                            if done {
                                break;
                            }
                            continue;
                        }
                        if let Some(hit) = sweep_unitary_phase_margin::<B>(
                            &trial,
                            &PhaseSweepArgs {
                                tier,
                                tank_spec,
                                target_t_ac_rms: target_t_total_rms,
                                hi_ac_rms: hi_rms,
                                sl0: &sl,
                                sr0: sr_opt.as_ref(),
                                hi_l,
                                hi_r,
                            },
                            ctx,
                        ) {
                            if record_chosen(hit) {
                                break;
                            }
                            continue;
                        }
                    }
                }
                if univ_ok {
                    let sc = eval_cand_rank(
                        &sl,
                        sr_opt.as_ref(),
                        hi_l,
                        hi_r,
                        tier_mr_ceil,
                        tier_max_ltas,
                    );
                    fallbacks.record(trial, tank_spec, sl, sr_opt, sc, quality_mode);
                }
            }
        }

        if chosen.is_none()
            && let Some((fb_trial, fb_spec, sl, sr_opt, _)) = fallbacks.full.or(fallbacks.near)
        {
            chosen = try_phase_detune_fallback::<B>(
                fb_trial,
                fb_spec,
                &DetuneFallbackArgs {
                    tier,
                    target_t_ac_rms: target_t_total_rms,
                    hi_ac_rms: hi_rms,
                    ch_sq: (ch_l_t_sq, ch_r_t_sq),
                    sl0: &sl,
                    sr0: sr_opt.as_ref(),
                    hi_l,
                    hi_r,
                    tilted_sum_l: &tilted_sum_l,
                    tilted_sum_r: tilted_sum_r.as_ref(),
                    mr_l,
                    mr_r,
                },
                ctx,
            );
        }
        let (ro_chosen, s4_chosen, spec_chosen, sl_chosen, sr_chosen, cand_rms) = chosen?;
        model.readouts[tau] = ro_chosen;
        model.stage4[tau] = s4_chosen;
        cur_tank = spec_chosen;
        hi_rms = cand_rms;
        sum_l[tau] = Some(sl_chosen);
        sum_r[tau] = sr_chosen;
        let (casc_perm, casc_step) = match tau {
            2 => (630_u16, 3_u32),
            1 => (815, 4),
            _ => (975, 5),
        };
        emit_stage4_summary_step::<B>(
            ctx.progress,
            ctx.anchor,
            Stage4SummaryStep {
                model,
                stem0,
                tier,
                permille: casc_perm,
                step: casc_step,
                total_steps: 5,
                sl: sum_l[tau].as_ref().unwrap(),
                sr: sum_r[tau].as_ref(),
            },
        );
    }

    let arr_l: [PerceptualCertificationSummary; 4] = [
        sum_l[0].take()?,
        sum_l[1].take()?,
        sum_l[2].take()?,
        sum_l[3].take()?,
    ];
    let arr_r: Option<[PerceptualCertificationSummary; 4]> = if mono_out {
        None
    } else {
        Some([
            sum_r[0].take()?,
            sum_r[1].take()?,
            sum_r[2].take()?,
            sum_r[3].take()?,
        ])
    };
    Some((arr_l, arr_r))
}
