#![forbid(unsafe_code)]
//! Phase-rotated Welch-33 EQ fallback and monotonicity/RMS gates (`cascade/fallback.rs`).

use dnam_engine::Tier;
use dnam_engine::stages::stage4_tank::v3_fdn::V3Stage4TankParams;
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::SolverQualityMode;
use crate::solver::backends::TrainerBackend;
use crate::solver::v3_varpro::ltas_adj::Welch33TargetSummary;
use crate::solver::v3_varpro::model_f64::V3TrainStem;

use super::super::comb_detect::{
    FdnTankSpec, apply_channel_phase_detune, compute_broadband_rms, snap_stage4_broadband_rms,
};
use super::super::mr_iir_adj::MrIirFloorSpec;
use super::super::wave_hinge::WaveFloorSpec;
use super::super::welch_eq::{WelchPolishSpec, polish_tier_welch33};
use super::super::{
    chan_beats_margin, db_beats_or_both_floor, eval_tier_cert, eval_tier_passes_universal,
    render_tier_stereo,
};
use super::{CascadeCtx, summary_passes_mode};

pub(super) type ChosenTierOut = (
    V3TierReadout,
    V3Stage4TankParams,
    FdnTankSpec,
    PerceptualCertificationSummary,
    Option<PerceptualCertificationSummary>,
    f64,
);

pub(super) fn rms_within_gate<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    warmup: usize,
    target_t_rms: f64,
    hi_rms: f64,
    mono_out: bool,
) -> Option<f64> {
    let n = stem.usable_len();
    let (ol, or) =
        render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
    let cand_rms = compute_broadband_rms(&ol, &or, warmup, mono_out);
    let ok = cand_rms > 1.0e-9
        && (20.0 * (cand_rms / target_t_rms).log10()).abs() <= 0.020
        && (20.0 * (cand_rms / hi_rms).log10()).abs() <= 0.045;
    ok.then_some(cand_rms)
}

pub(super) fn pair_beats_margin(
    sl: &PerceptualCertificationSummary,
    sr_opt: Option<&PerceptualCertificationSummary>,
    hi_l: &PerceptualCertificationSummary,
    hi_r: Option<&PerceptualCertificationSummary>,
    quality_mode: SolverQualityMode,
) -> bool {
    chan_beats_margin(sl, hi_l, quality_mode)
        && match (sr_opt, hi_r) {
            (Some(sr), Some(hr)) => chan_beats_margin(sr, hr, quality_mode),
            _ => true,
        }
}

fn chan_has_monotone_deficit(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> bool {
    !db_beats_or_both_floor(lo.raw_besr_db, hi.raw_besr_db, 0.002)
        || (0..3).any(|b| !db_beats_or_both_floor(lo.band_esr_db[b], hi.band_esr_db[b], 0.002))
        || lo.mr_iir < hi.mr_iir + 0.0015
        || lo.ltas_max_db < hi.ltas_max_db + 0.002
        || !db_beats_or_both_floor(lo.seg_esr_db, hi.seg_esr_db, 0.002)
        || lo.ltas_excluded_bands != hi.ltas_excluded_bands
        || (0..33).any(|b| {
            hi.ltas_excluded_bands[b]
                && lo.ltas_residual_band_power[b] < hi.ltas_residual_band_power[b]
        })
}

pub(super) struct DetuneFallbackArgs<'a> {
    pub(super) tier: Tier,
    pub(super) target_t_ac_rms: f64,
    pub(super) hi_ac_rms: f64,
    pub(super) ch_sq: (f64, f64),
    pub(super) sl0: &'a PerceptualCertificationSummary,
    pub(super) sr0: Option<&'a PerceptualCertificationSummary>,
    pub(super) hi_l: &'a PerceptualCertificationSummary,
    pub(super) hi_r: Option<&'a PerceptualCertificationSummary>,
    pub(super) tilted_sum_l: &'a Welch33TargetSummary,
    pub(super) tilted_sum_r: Option<&'a Welch33TargetSummary>,
    pub(super) mr_l: Option<&'a MrIirFloorSpec<'a>>,
    pub(super) mr_r: Option<&'a MrIirFloorSpec<'a>>,
}

const PHASE_DETUNE_SCHEDULE: [(f32, f32); 14] = [
    (0.006, 0.010),
    (0.010, 0.016),
    (-0.012, 0.020),
    (-0.025, 0.040),
    (0.015, 0.025),
    (0.035, 0.06),
    (0.06, 0.10),
    (-0.16, 0.22),
    (-0.36, 0.42),
    (-0.68, 0.68),
    (-1.02, 0.96),
    (-1.18, 1.25),
    (0.18, -0.28),
    (-0.85, -0.80),
];

pub(super) struct PhaseSweepArgs<'a> {
    pub(super) tier: Tier,
    pub(super) tank_spec: FdnTankSpec,
    pub(super) target_t_ac_rms: f64,
    pub(super) hi_ac_rms: f64,
    pub(super) sl0: &'a PerceptualCertificationSummary,
    pub(super) sr0: Option<&'a PerceptualCertificationSummary>,
    pub(super) hi_l: &'a PerceptualCertificationSummary,
    pub(super) hi_r: Option<&'a PerceptualCertificationSummary>,
}

fn chan_phase_only_ltas_short(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> bool {
    lo.ltas_max_db < hi.ltas_max_db + 0.002
        && db_beats_or_both_floor(lo.raw_besr_db, hi.raw_besr_db, 0.002)
        && (0..3).all(|b| db_beats_or_both_floor(lo.band_esr_db[b], hi.band_esr_db[b], 0.002))
        && lo.mr_iir >= hi.mr_iir + 0.0015
        && db_beats_or_both_floor(lo.seg_esr_db, hi.seg_esr_db, 0.002)
}

pub(super) fn sweep_unitary_phase_margin<B: TrainerBackend>(
    trial: &V3MatryoshkaModel,
    args: &PhaseSweepArgs<'_>,
    ctx: &mut CascadeCtx<'_, '_>,
) -> Option<ChosenTierOut> {
    let quality_mode = ctx.quality_mode;
    let passes0 = summary_passes_mode(args.sl0, quality_mode)
        && args
            .sr0
            .is_none_or(|r| summary_passes_mode(r, quality_mode));
    if !passes0 {
        return None;
    }
    let mono_out = trial.mono_out;
    let stem0 = ctx.eq_ctx.stem;
    let warmup = ctx.eq_ctx.warmup;
    let t = args.tier as usize;
    let def_l = chan_has_monotone_deficit(args.sl0, args.hi_l);
    let def_r = !mono_out
        && args
            .sr0
            .zip(args.hi_r)
            .is_some_and(|(sr, hr)| chan_has_monotone_deficit(sr, hr));
    if !def_l && !def_r {
        return None;
    }

    for &(alpha, phi) in &PHASE_DETUNE_SCHEDULE {
        let mut phased = trial.clone();
        if def_l {
            apply_channel_phase_detune(&mut phased, args.tier, 0, alpha, phi);
        }
        if def_r {
            apply_channel_phase_detune(&mut phased, args.tier, 1, alpha, phi);
        }
        let _ = snap_stage4_broadband_rms::<B>(
            &mut phased,
            args.tier,
            stem0,
            warmup,
            args.target_t_ac_rms,
            mono_out,
        );
        let (mut bl, mut br_opt) = eval_tier_cert::<B>(&phased, args.tier, stem0, ctx.certs);
        if pair_beats_margin(&bl, br_opt.as_ref(), args.hi_l, args.hi_r, quality_mode)
            && eval_tier_passes_universal::<B>(&phased, args.tier, stem0, ctx.certs, quality_mode)
            && let Some(cand_rms) = rms_within_gate::<B>(
                &phased,
                args.tier,
                stem0,
                warmup,
                args.target_t_ac_rms,
                args.hi_ac_rms,
                mono_out,
            )
        {
            return Some((
                phased.readouts[t].clone(),
                phased.stage4[t],
                args.tank_spec,
                bl,
                br_opt,
                cand_rms,
            ));
        }

        let need_ltas_l = chan_phase_only_ltas_short(&bl, args.hi_l);
        let need_ltas_r = !mono_out
            && br_opt
                .as_ref()
                .zip(args.hi_r)
                .is_some_and(|(br, hr)| chan_phase_only_ltas_short(br, hr));
        if need_ltas_l || need_ltas_r {
            if need_ltas_l {
                let f0 = phased.readouts[t].fir_l[0];
                let gap_l =
                    (args.hi_l.ltas_max_db + 0.006 - bl.ltas_max_db).clamp(0.002, 0.05) as f32;
                let d = 0.06 * gap_l * f0;
                phased.readouts[t].fir_l[0] += d;
                phased.readouts[t].fir_l[1] -= d;
                if mono_out {
                    phased.readouts[t].fir_r = phased.readouts[t].fir_l;
                }
            }
            if need_ltas_r && let (Some(br), Some(hr)) = (br_opt.as_ref(), args.hi_r) {
                let f0 = phased.readouts[t].fir_r[0];
                let gap_r = (hr.ltas_max_db + 0.006 - br.ltas_max_db).clamp(0.002, 0.05) as f32;
                let d = 0.06 * gap_r * f0;
                phased.readouts[t].fir_r[0] += d;
                phased.readouts[t].fir_r[1] -= d;
            }
            let _ = snap_stage4_broadband_rms::<B>(
                &mut phased,
                args.tier,
                stem0,
                warmup,
                args.target_t_ac_rms,
                mono_out,
            );
            (bl, br_opt) = eval_tier_cert::<B>(&phased, args.tier, stem0, ctx.certs);
            if pair_beats_margin(&bl, br_opt.as_ref(), args.hi_l, args.hi_r, quality_mode)
                && eval_tier_passes_universal::<B>(
                    &phased,
                    args.tier,
                    stem0,
                    ctx.certs,
                    quality_mode,
                )
                && let Some(cand_rms) = rms_within_gate::<B>(
                    &phased,
                    args.tier,
                    stem0,
                    warmup,
                    args.target_t_ac_rms,
                    args.hi_ac_rms,
                    mono_out,
                )
            {
                return Some((
                    phased.readouts[t].clone(),
                    phased.stage4[t],
                    args.tank_spec,
                    bl,
                    br_opt,
                    cand_rms,
                ));
            }
        }
    }
    None
}

pub(super) fn try_phase_detune_fallback<B: TrainerBackend>(
    trial: V3MatryoshkaModel,
    tank_spec: FdnTankSpec,
    args: &DetuneFallbackArgs<'_>,
    ctx: &mut CascadeCtx<'_, '_>,
) -> Option<ChosenTierOut> {
    let stem0 = ctx.eq_ctx.stem;
    let warmup = ctx.eq_ctx.warmup;
    let mono_out = trial.mono_out;
    let t = args.tier as usize;
    let quality_mode = ctx.quality_mode;
    let (polish_goal_db, step_b_iters) = match quality_mode {
        SolverQualityMode::BestUntilDiminishingReturns => (0.0_f64, 56_usize),
        SolverQualityMode::ThresholdStop {
            max_polish_iters, ..
        } => (0.008_f64, max_polish_iters.clamp(8, 64)),
    };
    let apply_detune = |mdl: &mut V3MatryoshkaModel, alpha: f32, phi: f32| {
        if chan_has_monotone_deficit(args.sl0, args.hi_l) {
            apply_channel_phase_detune(mdl, args.tier, 0, alpha, phi);
        }
        if !mono_out
            && let (Some(sr), Some(hr)) = (args.sr0, args.hi_r)
            && chan_has_monotone_deficit(sr, hr)
        {
            apply_channel_phase_detune(mdl, args.tier, 1, alpha, phi);
        }
    };

    let tier_max_ltas = crate::metrics::TierQualityThresholds::for_tier(args.tier).max_ltas_db;
    let slack_db = match quality_mode {
        SolverQualityMode::BestUntilDiminishingReturns => 0.0,
        SolverQualityMode::ThresholdStop {
            target_slack_milli_db,
            ..
        } => f64::from(target_slack_milli_db) * 1.0e-3,
    };
    let eff_max_ltas = tier_max_ltas + slack_db;
    let near_ltas = args.sl0.ltas_max_db <= eff_max_ltas + 1.50
        && args
            .sr0
            .is_none_or(|r| r.ltas_max_db <= eff_max_ltas + 1.50);
    if !near_ltas {
        return None;
    }

    // Step A: Unitary allpass + Kautz/s2_modal sweep when LTAS/MR thresholds already pass.
    if let Some(hit) = sweep_unitary_phase_margin::<B>(
        &trial,
        &PhaseSweepArgs {
            tier: args.tier,
            tank_spec,
            target_t_ac_rms: args.target_t_ac_rms,
            hi_ac_rms: args.hi_ac_rms,
            sl0: args.sl0,
            sr0: args.sr0,
            hi_l: args.hi_l,
            hi_r: args.hi_r,
        },
        ctx,
    ) {
        return Some(hit);
    }
    let any_mono_deficit = chan_has_monotone_deficit(args.sl0, args.hi_l)
        || (!mono_out
            && args
                .sr0
                .zip(args.hi_r)
                .is_some_and(|(sr, hr)| chan_has_monotone_deficit(sr, hr)));

    // Step B: Pure-convex Welch-33 LTAS polish first, then unitary sweep & gentle hinge fallback.
    const FB_WAVE_MARGIN_DB: [f64; 4] = [0.042, 0.055, 0.022, 0.0];
    let fb_off = FB_WAVE_MARGIN_DB[t];
    let boost_l = (
        args.hi_l
            .band_esr_db
            .map(|db| 10.0_f64.powf((db + fb_off) / 10.0)),
        10.0_f64.powf((args.hi_l.raw_besr_db + fb_off) / 10.0),
        10.0_f64.powf((args.hi_l.seg_esr_db + fb_off) / 10.0),
    );
    let boost_r = args.hi_r.map(|hr| {
        (
            hr.band_esr_db.map(|db| 10.0_f64.powf((db + fb_off) / 10.0)),
            10.0_f64.powf((hr.raw_besr_db + fb_off) / 10.0),
            10.0_f64.powf((hr.seg_esr_db + fb_off) / 10.0),
        )
    });
    let (bw_l, bw_r): (WaveFloorSpec<'_>, WaveFloorSpec<'_>) =
        ctx.wave_ctx.make_specs(0.0, Some(boost_l), boost_r);
    let tier_mr_ceil: f64 = match args.tier {
        Tier::Lite => 0.68,
        Tier::Full => 0.62,
        Tier::Pro => 0.56,
        Tier::Ultra => 0.52,
    };
    let ltas_over_ceil = args.sl0.ltas_max_db > eff_max_ltas
        || args.sr0.is_some_and(|r| r.ltas_max_db > eff_max_ltas);
    let use_untilted = ltas_over_ceil
        || args.sl0.ltas_max_db > eff_max_ltas * 0.60
        || args
            .sr0
            .is_some_and(|r| r.ltas_max_db > eff_max_ltas * 0.60);

    let detunes: &[(f32, f32)] = if any_mono_deficit {
        &[(0.0_f32, 0.0_f32), (-0.36, 0.42), (-0.85, -0.80)]
    } else {
        &[(0.0_f32, 0.0_f32)]
    };

    for &(alpha, phi) in detunes {
        let mut cand = trial.clone();
        if alpha != 0.0 || phi != 0.0 {
            apply_detune(&mut cand, alpha, phi);
        }
        let (sum_l_ref, sum_r_ref) = if use_untilted {
            (ctx.welch.sum_l, ctx.welch.sum_r)
        } else {
            (args.tilted_sum_l, args.tilted_sum_r)
        };
        polish_tier_welch33::<B>(
            &mut cand,
            &WelchPolishSpec {
                tier: args.tier,
                stem: stem0,
                sum_l: sum_l_ref,
                sum_r: sum_r_ref,
                wave_l: (!ltas_over_ceil).then_some(&bw_l),
                wave_r: (!ltas_over_ceil && !mono_out).then_some(&bw_r),
                mr_l: if ltas_over_ceil { None } else { args.mr_l },
                mr_r: if ltas_over_ceil { None } else { args.mr_r },
                lock_rms_lr: Some((args.target_t_ac_rms, args.ch_sq.0, args.ch_sq.1)),
                goal_db: polish_goal_db,
                iters: step_b_iters,
            },
        );
        let _ = snap_stage4_broadband_rms::<B>(
            &mut cand,
            args.tier,
            stem0,
            warmup,
            args.target_t_ac_rms,
            mono_out,
        );
        let (mut bl, mut br_opt) = eval_tier_cert::<B>(&cand, args.tier, stem0, ctx.certs);
        if pair_beats_margin(&bl, br_opt.as_ref(), args.hi_l, args.hi_r, quality_mode)
            && eval_tier_passes_universal::<B>(&cand, args.tier, stem0, ctx.certs, quality_mode)
            && let Some(cand_rms) = rms_within_gate::<B>(
                &cand,
                args.tier,
                stem0,
                warmup,
                args.target_t_ac_rms,
                args.hi_ac_rms,
                mono_out,
            )
        {
            return Some((
                cand.readouts[t].clone(),
                cand.stage4[t],
                tank_spec,
                bl,
                br_opt,
                cand_rms,
            ));
        }
        if let Some(hit) = sweep_unitary_phase_margin::<B>(
            &cand,
            &PhaseSweepArgs {
                tier: args.tier,
                tank_spec,
                target_t_ac_rms: args.target_t_ac_rms,
                hi_ac_rms: args.hi_ac_rms,
                sl0: &bl,
                sr0: br_opt.as_ref(),
                hi_l: args.hi_l,
                hi_r: args.hi_r,
            },
            ctx,
        ) {
            return Some(hit);
        }
        let bad_mr_l = bl.mr_iir < args.hi_l.mr_iir + 0.002 || bl.mr_iir > tier_mr_ceil;
        let bad_mr_r = !mono_out
            && br_opt
                .as_ref()
                .zip(args.hi_r)
                .is_some_and(|(sr, hr)| sr.mr_iir < hr.mr_iir + 0.002 || sr.mr_iir > tier_mr_ceil);
        let need_secondary = ltas_over_ceil
            && (bad_mr_l
                || bad_mr_r
                || (summary_passes_mode(&bl, quality_mode)
                    && br_opt
                        .as_ref()
                        .is_none_or(|r| summary_passes_mode(r, quality_mode))));
        if need_secondary {
            polish_tier_welch33::<B>(
                &mut cand,
                &WelchPolishSpec {
                    tier: args.tier,
                    stem: stem0,
                    sum_l: sum_l_ref,
                    sum_r: sum_r_ref,
                    wave_l: Some(&bw_l),
                    wave_r: (!mono_out).then_some(&bw_r),
                    mr_l: if bad_mr_l || bad_mr_r {
                        args.mr_l
                    } else {
                        None
                    },
                    mr_r: if bad_mr_l || bad_mr_r {
                        args.mr_r
                    } else {
                        None
                    },
                    lock_rms_lr: Some((args.target_t_ac_rms, args.ch_sq.0, args.ch_sq.1)),
                    goal_db: polish_goal_db,
                    iters: 28,
                },
            );
            let _ = snap_stage4_broadband_rms::<B>(
                &mut cand,
                args.tier,
                stem0,
                warmup,
                args.target_t_ac_rms,
                mono_out,
            );
            (bl, br_opt) = eval_tier_cert::<B>(&cand, args.tier, stem0, ctx.certs);
            if pair_beats_margin(&bl, br_opt.as_ref(), args.hi_l, args.hi_r, quality_mode)
                && eval_tier_passes_universal::<B>(&cand, args.tier, stem0, ctx.certs, quality_mode)
                && let Some(cand_rms) = rms_within_gate::<B>(
                    &cand,
                    args.tier,
                    stem0,
                    warmup,
                    args.target_t_ac_rms,
                    args.hi_ac_rms,
                    mono_out,
                )
            {
                return Some((
                    cand.readouts[t].clone(),
                    cand.stage4[t],
                    tank_spec,
                    bl,
                    br_opt,
                    cand_rms,
                ));
            }
            if let Some(hit) = sweep_unitary_phase_margin::<B>(
                &cand,
                &PhaseSweepArgs {
                    tier: args.tier,
                    tank_spec,
                    target_t_ac_rms: args.target_t_ac_rms,
                    hi_ac_rms: args.hi_ac_rms,
                    sl0: &bl,
                    sr0: br_opt.as_ref(),
                    hi_l: args.hi_l,
                    hi_r: args.hi_r,
                },
                ctx,
            ) {
                return Some(hit);
            }
        }
    }
    None
}
