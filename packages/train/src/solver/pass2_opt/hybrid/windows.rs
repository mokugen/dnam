//! Windowed chunking, single/joint-tier evaluation, VarPro refresh wrappers, progress reporting,
//! and progressive step scheduling helpers for [`super::hybrid`].

use core::ops::RangeInclusive;

use dnam_vm::Tier;

use crate::error::TrainError;
use crate::linalg::adamw::{AdamWConfig, AdamWState, cosine_lr_with_warmup};
use crate::linalg::coords::ScaledCoordSpace;
use crate::losses::{CompositeLossConfig, CompositeLossEvaluator, CompositeLossWeights};
use crate::runner::TrainRunner;

use super::super::super::model::{
    NUM_FIR_TAPS_ULTRA, SonaModelF64, StereoTrainingPairs, snap_non_readout_fp16,
};
use super::super::super::progress::{
    AtomicDnprProgress, ProgressMetrics, TrainingPhase, TrainingStatus,
};
use super::super::Pass2OptConfig;
use super::super::eval::{Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio};
use super::super::matryoshka::{MatryoshkaEvalCtx, eval_joint_matryoshka_and_grad};
use super::super::prescreen::pick_representative_window_start;
use super::super::qat_polish::run_pass2_ste_qat_polish;
use super::super::theta::{TIER_FIR_TAPS, TIER_KAUTZ, TIER_QUADS, ThetaLayout, has_active_op_t};
use super::super::varpro_sync::{
    VarProSyncCtx, materialize_tier_view, refresh_tier_readouts_varpro,
};

pub(super) struct HybridCtx<'a> {
    pub runner: &'a TrainRunner,
    pub pairs: StereoTrainingPairs<'a>,
    pub noise_pair: Option<(&'a [f64], &'a [f64])>,
    pub full_loss: &'a CompositeLossEvaluator,
    pub config: Pass2OptConfig,
    pub warmup: usize,
    pub max_tau: f64,
    pub max_tier: usize,
    pub fast_mode: bool,
    pub progress: Option<&'a AtomicDnprProgress>,
}

pub(super) struct WindowEval {
    pub start: usize,
    pub end: usize,
    pub warmup: usize,
    pub loss_eval: Option<CompositeLossEvaluator>,
}

impl WindowEval {
    pub(super) fn pairs<'a>(&self, full: StereoTrainingPairs<'a>) -> StereoTrainingPairs<'a> {
        StereoTrainingPairs {
            in_l: &full.in_l[self.start..self.end],
            in_r: &full.in_r[self.start..self.end],
            target_l: &full.target_l[self.start..self.end],
            target_r: &full.target_r[self.start..self.end],
        }
    }

    pub(super) fn loss<'a>(
        &'a self,
        full_loss: &'a CompositeLossEvaluator,
    ) -> &'a CompositeLossEvaluator {
        self.loss_eval.as_ref().unwrap_or(full_loss)
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.end - self.start
    }
}

fn build_window_at(ctx: &HybridCtx<'_>, start: usize, wl: usize) -> Result<WindowEval, TrainError> {
    let n = ctx.pairs.aligned_len();
    if wl >= n {
        return Ok(WindowEval {
            start: 0,
            end: n,
            warmup: ctx.warmup,
            loss_eval: None,
        });
    }
    let end = start + wl;
    let w_warmup = ctx.warmup.min(wl / 4);
    let nl = ctx.noise_pair.map(|(l, _)| &l[start..end]);
    let nr = ctx.noise_pair.map(|(_, r)| &r[start..end]);
    let eval = CompositeLossEvaluator::new_for_homotopy(
        &ctx.pairs.target_l[start..end],
        &ctx.pairs.target_r[start..end],
        nl,
        nr,
        CompositeLossConfig {
            sample_rate_hz: ctx.config.sample_rate_hz,
            warmup_frames: w_warmup,
            ..CompositeLossConfig::default()
        },
    )?;
    Ok(WindowEval {
        start,
        end,
        warmup: w_warmup,
        loss_eval: Some(eval),
    })
}

pub(super) fn build_windows(
    ctx: &HybridCtx<'_>,
    win_len: usize,
    max_windows: usize,
) -> Result<Vec<WindowEval>, TrainError> {
    let n = ctx.pairs.aligned_len();
    let wl = win_len.clamp(64.min(n), n);
    if wl >= n {
        return Ok(vec![build_window_at(ctx, 0, wl)?]);
    }
    let count = max_windows.max(1);
    let slack = n - wl;
    let mut windows = Vec::with_capacity(count);
    for w_i in 0..count {
        let start = if count == 1 {
            0
        } else {
            (w_i * slack) / (count - 1)
        };
        windows.push(build_window_at(ctx, start, wl)?);
    }
    Ok(windows)
}

pub(super) fn build_representative_window(
    ctx: &HybridCtx<'_>,
    win_len: usize,
) -> Result<WindowEval, TrainError> {
    let n = ctx.pairs.aligned_len();
    let wl = win_len.clamp(64.min(n), n);
    let start =
        pick_representative_window_start(&ctx.pairs.target_l[..n], &ctx.pairs.target_r[..n], wl);
    build_window_at(ctx, start, wl)
}

/// Maximum allowable `Ultra` `bESR` regression (in dB) when trading off composite perceptual
/// terms across Pass-2 phase boundaries (`0.75 dB` prevents silence/stopband collapse while
/// permitting genuine `LTAS`/`MR-IIR`/`Gabor` perceptual refinements).
pub(super) const MAX_ULTRA_BESR_REGRESSION_DB: f64 = 0.75;
const _: () = assert!(MAX_ULTRA_BESR_REGRESSION_DB > 0.0 && MAX_ULTRA_BESR_REGRESSION_DB <= 1.0);

/// Returns `true` when candidate full-capsule metrics `(accept_obj, ultra_loss, ultra_besr_db)`
/// satisfy objective non-regression without regressing `Ultra` loss or collapsing `Ultra` `bESR`.
#[inline]
pub(super) fn is_pass2_metrics_acceptable(
    baseline: (f64, f64, f64),
    cand: (f64, f64, f64),
    use_joint: bool,
    strict_obj: bool,
) -> bool {
    if !cand.0.is_finite() || !cand.1.is_finite() || !cand.2.is_finite() {
        return false;
    }
    let obj_ok = !baseline.0.is_finite()
        || if strict_obj {
            cand.0 < baseline.0
        } else {
            cand.0 <= baseline.0
        };
    let ultra_loss_ok =
        !use_joint || !baseline.1.is_finite() || cand.1 <= baseline.1 * 1.02 + 1.0e-12;
    let ultra_besr_ok =
        !baseline.2.is_finite() || cand.2 <= baseline.2 + MAX_ULTRA_BESR_REGRESSION_DB;
    obj_ok && ultra_loss_ok && ultra_besr_ok
}

/// Evaluates metrics across tier prefix `0..=eval_ti`, returning `(accept_obj, top_tier_loss, top_tier_besr_db)`.
/// When `use_joint` is active and `eval_ti > 0`, `accept_obj` is the nested log-Matryoshka
/// objective so lower-tier improvements are preserved while `top_tier_{loss,besr_db}` report `eval_ti`.
pub(super) fn eval_tier_prefix_metrics(
    ctx: &HybridCtx<'_>,
    model: &SonaModelF64,
    eval_ti: usize,
    use_joint: bool,
    weights: CompositeLossWeights,
) -> Result<(f64, f64, f64), TrainError> {
    let ti = eval_ti.min(3);
    if use_joint && ti > 0 {
        let layout = ThetaLayout::for_tier_from_model(model, ti, false, false, ctx.max_tau);
        let out = eval_joint_matryoshka_and_grad(
            &MatryoshkaEvalCtx {
                runner: ctx.runner,
                pairs: ctx.pairs,
                loss_eval: ctx.full_loss,
                weights,
                layout: &layout,
                space_jac: None,
                max_tier: ti,
                use_pcgrad: false,
            },
            model,
            false,
        )?;
        let accept = if out.joint_loss.is_finite() {
            out.joint_loss
        } else {
            f64::INFINITY
        };
        let top_loss = if out.tier_losses[ti].is_finite() {
            out.tier_losses[ti]
        } else {
            f64::INFINITY
        };
        return Ok((accept, top_loss, out.tier_besr_db[ti]));
    }

    let n = ctx.pairs.aligned_len();
    let q_t = TIER_QUADS[ti];
    let view = materialize_tier_view(model, ti);
    let buf = eval_model_pass2_audio(
        ctx.runner,
        &view,
        &ctx.pairs.in_l[..n],
        &ctx.pairs.in_r[..n],
        q_t,
    )?;
    let bd = ctx.full_loss.eval_and_adjoint_with_weights(
        &buf.pred_l,
        &buf.pred_r,
        weights,
        None,
        None,
    )?;
    let total = if bd.total.is_finite() {
        bd.total
    } else {
        f64::INFINITY
    };
    let besr_db = 10.0 * bd.besr.max(1.0e-12).log10();
    Ok((total, total, besr_db))
}

/// Evaluates full-capsule metrics across `0..=ctx.max_tier`.
pub(super) fn eval_full_capsule_metrics(
    ctx: &HybridCtx<'_>,
    model: &SonaModelF64,
    use_joint: bool,
    weights: CompositeLossWeights,
) -> Result<(f64, f64, f64), TrainError> {
    eval_tier_prefix_metrics(ctx, model, ctx.max_tier, use_joint, weights)
}

fn refresh_tier_range_varpro(
    ctx: &HybridCtx<'_>,
    win_opt: Option<&WindowEval>,
    weights: CompositeLossWeights,
    model: &mut SonaModelF64,
    tier_range: RangeInclusive<usize>,
    sync_max_tier: bool,
) -> Result<(), TrainError> {
    let (pairs, loss_eval, warmup) = match win_opt {
        Some(win) => (win.pairs(ctx.pairs), win.loss(ctx.full_loss), win.warmup),
        None => (ctx.pairs, ctx.full_loss, ctx.warmup),
    };
    let vctx = VarProSyncCtx {
        runner: ctx.runner,
        pairs,
        loss_eval,
        weights,
        warmup,
    };
    for ti in tier_range {
        let sync_top = sync_max_tier && ti == ctx.max_tier;
        let _ = refresh_tier_readouts_varpro(&vctx, model, ti, sync_top)?;
    }
    Ok(())
}

pub(super) fn refresh_active_tiers_varpro(
    ctx: &HybridCtx<'_>,
    win_opt: Option<&WindowEval>,
    weights: CompositeLossWeights,
    model: &mut SonaModelF64,
    highest_ti: usize,
    use_joint: bool,
) -> Result<(), TrainError> {
    let start_ti = if use_joint { 0 } else { highest_ti };
    refresh_tier_range_varpro(ctx, win_opt, weights, model, start_ti..=highest_ti, true)
}

pub(super) fn refresh_lower_tiers_varpro_full(
    ctx: &HybridCtx<'_>,
    weights: CompositeLossWeights,
    model: &mut SonaModelF64,
) -> Result<(), TrainError> {
    if ctx.max_tier == 0 {
        return Ok(());
    }
    refresh_tier_range_varpro(ctx, None, weights, model, 0..=(ctx.max_tier - 1), false)
}

pub(super) fn run_phase2d_guarded_qat(
    ctx: &HybridCtx<'_>,
    work_model: &mut SonaModelF64,
    pre_qat_loss: f64,
    pre_qat_besr_db: f64,
) -> Result<(usize, f64, f64, f64), TrainError> {
    let max_tier = ctx.max_tier;
    sync_top_level_from_tier_override(work_model, max_tier);
    if ctx.config.qat_polish_steps == 0 {
        snap_non_readout_fp16(work_model);
        return Ok((0_usize, pre_qat_loss, pre_qat_loss, pre_qat_besr_db));
    }

    let n = ctx.pairs.aligned_len();
    let use_joint = ctx.config.use_progressive_matryoshka && max_tier > 0;
    let weights_final = CompositeLossWeights::homotopy_at(1.0);
    let mut pre_qat_snapped = work_model.clone();
    snap_non_readout_fp16(&mut pre_qat_snapped);
    let pre_qat_full = eval_full_capsule_metrics(ctx, &pre_qat_snapped, use_joint, weights_final)?;

    let q_final = TIER_QUADS[max_tier];
    let include_extras = has_active_op_t(work_model, q_final) || work_model.active_kautz_pairs > 0;
    let qat_layout =
        ThetaLayout::for_tier_from_model_qat(work_model, max_tier, include_extras, ctx.max_tau);
    let qat_win = build_representative_window(ctx, n.min(4_096))?;
    let qat = run_pass2_ste_qat_polish(
        ctx.runner,
        work_model,
        &qat_win.pairs(ctx.pairs),
        qat_win.loss(ctx.full_loss),
        weights_final,
        &qat_layout,
        ctx.config,
    )?;
    sync_tier_override_from_top_level(work_model, max_tier);
    if use_joint {
        refresh_lower_tiers_varpro_full(ctx, weights_final, work_model)?;
    }

    if qat_win.len() == n && !use_joint {
        return Ok((
            qat.steps_run,
            qat.baseline_loss,
            qat.best_loss,
            qat.best_besr_db,
        ));
    }

    let post_qat_full = eval_full_capsule_metrics(ctx, work_model, use_joint, weights_final)?;
    let regressed = !post_qat_full.0.is_finite()
        || (pre_qat_full.0.is_finite() && post_qat_full.0 > pre_qat_full.0)
        || (pre_qat_full.1.is_finite()
            && (!post_qat_full.1.is_finite() || post_qat_full.1 > pre_qat_full.1))
        || (pre_qat_full.2.is_finite()
            && (!post_qat_full.2.is_finite() || post_qat_full.2 > pre_qat_full.2 + 0.5));
    if regressed {
        *work_model = pre_qat_snapped;
        Ok((
            qat.steps_run,
            pre_qat_full.1,
            pre_qat_full.1,
            pre_qat_full.2,
        ))
    } else {
        Ok((
            qat.steps_run,
            pre_qat_full.1.max(post_qat_full.1),
            post_qat_full.1,
            post_qat_full.2,
        ))
    }
}

pub(super) fn sync_top_level_from_tier_override(model: &mut SonaModelF64, tier_idx: usize) {
    let ti = tier_idx.min(3);
    let Some(ref ovr) = model.tier_overrides[ti] else {
        return;
    };
    let n_rows = TIER_QUADS[ti] * 4;
    model.direct_s2_u_gain = ovr.direct_s2_u_gain;
    model.direct_s2_hp_gain = ovr.direct_s2_hp_gain;
    for row in 0..n_rows {
        model.w_ro[row][6..12].copy_from_slice(&ovr.w_ro[row][6..12]);
    }
    let f_cap = model
        .active_fir_taps
        .min(TIER_FIR_TAPS[ti])
        .min(NUM_FIR_TAPS_ULTRA);
    model.fir_l[..f_cap].copy_from_slice(&ovr.fir_l[..f_cap]);
    model.fir_r[..f_cap].copy_from_slice(&ovr.fir_r[..f_cap]);
    let k_cap = model
        .active_kautz_pairs
        .min(TIER_KAUTZ[ti])
        .min(model.kautz.len());
    for k in 0..k_cap {
        if model.kautz[k].r_low > 0.0 && model.kautz[k].b_gamma > 0.0 {
            model.kautz[k].c_re_l = ovr.kautz_c_re_l[k];
            model.kautz[k].c_im_l = ovr.kautz_c_im_l[k];
            model.kautz[k].c_re_r = ovr.kautz_c_re_r[k];
            model.kautz[k].c_im_r = ovr.kautz_c_im_r[k];
        }
    }
}

pub(super) fn eval_single_tier_with_grad(
    ctx: &HybridCtx<'_>,
    win: &WindowEval,
    model: &SonaModelF64,
    weights: CompositeLossWeights,
    grad_layout: Option<&ThetaLayout>,
    tier_idx: usize,
) -> Result<(f64, Vec<f64>), TrainError> {
    let pairs = win.pairs(ctx.pairs);
    let loss_eval = win.loss(ctx.full_loss);
    let need_grad = grad_layout.is_some();
    let n = pairs.aligned_len();
    let ti = tier_idx.min(3);
    let q_t = TIER_QUADS[ti];
    let view = materialize_tier_view(model, ti);
    let buf = eval_model_pass2_audio(ctx.runner, &view, &pairs.in_l[..n], &pairs.in_r[..n], q_t)?;
    let mut gl = if need_grad {
        vec![0.0_f64; n]
    } else {
        Vec::new()
    };
    let mut gr = if need_grad {
        vec![0.0_f64; n]
    } else {
        Vec::new()
    };
    let bd = if need_grad {
        loss_eval.eval_and_adjoint_with_weights(
            &buf.pred_l,
            &buf.pred_r,
            weights,
            Some(&mut gl),
            Some(&mut gr),
        )?
    } else {
        loss_eval.eval_and_adjoint_with_weights(&buf.pred_l, &buf.pred_r, weights, None, None)?
    };
    let mut gx = Vec::new();
    if let Some(layout) = grad_layout {
        let grads = assemble_pass2_gradients(
            ctx.runner,
            &Pass2AdjInput {
                model: &view,
                in_l: &pairs.in_l[..n],
                in_r: &pairs.in_r[..n],
                buf: &buf,
                gl: &gl,
                gr: &gr,
                q_opt: layout.q_opt,
                k_opt: layout.k_opt,
            },
        )?;
        gx.resize(layout.dim(), 0.0);
        layout.write_grad(&grads, &mut gx);
    }
    Ok((bd.total, gx))
}

pub(super) fn sync_tier_override_from_top_level(model: &mut SonaModelF64, tier_idx: usize) {
    let ti = tier_idx.min(3);
    let Some(ref mut ovr) = model.tier_overrides[ti] else {
        return;
    };
    ovr.pre_conv_u_gain = model.pre_conv_u_gain;
    ovr.pre_conv_hp_gain = model.pre_conv_hp_gain;
    ovr.direct_s2_u_gain = model.direct_s2_u_gain;
    ovr.direct_s2_hp_gain = model.direct_s2_hp_gain;
    ovr.w_ro = model.w_ro;
    ovr.fir_l = model.fir_l;
    ovr.fir_r = model.fir_r;
    for (k, kp) in model.kautz.iter().enumerate() {
        ovr.kautz_b_gamma[k] = kp.b_gamma;
        ovr.kautz_c_re_l[k] = kp.c_re_l;
        ovr.kautz_c_im_l[k] = kp.c_im_l;
        ovr.kautz_c_re_r[k] = kp.c_re_r;
        ovr.kautz_c_im_r[k] = kp.c_im_r;
    }
}

pub(super) fn explore_candidate_short(
    ctx: &HybridCtx<'_>,
    cand: &mut SonaModelF64,
    win: &WindowEval,
    stage_ti: usize,
    steps: usize,
) -> Result<f64, TrainError> {
    let has_op_t = has_active_op_t(cand, TIER_QUADS[stage_ti]);
    let include_extras = has_op_t || cand.active_kautz_pairs > 0;
    let include_readouts = has_op_t;
    let layout = ThetaLayout::for_tier_from_model_pass2(
        cand,
        stage_ti,
        include_extras,
        include_readouts,
        ctx.max_tau,
    );
    let (init_phys, bounds) = layout.pack(cand);
    let scales = layout.coord_scales(win.len());
    let (space, mut u) = ScaledCoordSpace::new(
        "explore_candidate_short",
        &init_phys,
        &bounds,
        Some(&scales),
    )?;
    let mut x = vec![0.0_f64; layout.dim()];
    let mut jac = vec![0.0_f64; layout.dim()];
    let mut gu = vec![0.0_f64; layout.dim()];
    let mut adamw = AdamWState::new(layout.dim());
    let cfg = AdamWConfig {
        lr_max: 2.5e-3,
        lr_min: 5.0e-4,
        warmup_steps: 1,
        weight_decay: 0.0,
        grad_clip_norm: 10.0,
        ..AdamWConfig::default()
    };
    let w_expl = CompositeLossWeights::homotopy_at(0.25);

    let mut work = cand.clone();
    let mut best_model = cand.clone();
    let mut best_loss = f64::INFINITY;
    for step in 0..steps {
        space.decode(&u, &mut x, &mut jac);
        layout.unpack(&x, &mut work);
        let (l_val, gx) =
            eval_single_tier_with_grad(ctx, win, &work, w_expl, Some(&layout), stage_ti)?;
        if l_val.is_finite() && l_val < best_loss {
            best_loss = l_val;
            best_model = work.clone();
        }
        space.pullback_grad(&gx, &jac, &mut gu);
        let lr = cosine_lr_with_warmup(step, steps, &cfg);
        adamw.step_u(&mut u, &mut gu, lr, &cfg, Some(space.scales()));
    }
    let _ = refresh_active_tiers_varpro(ctx, Some(win), w_expl, &mut best_model, stage_ti, false);
    let (fin_l, _) = eval_single_tier_with_grad(ctx, win, &best_model, w_expl, None, stage_ti)?;
    *cand = best_model;
    Ok(if fin_l.is_finite() {
        fin_l.min(best_loss)
    } else {
        best_loss
    })
}

pub(super) fn publish_progress(
    progress: Option<&AtomicDnprProgress>,
    permille: u16,
    step: usize,
    total_steps: usize,
    best_loss: f64,
    active_tier: Tier,
) {
    if let Some(p) = progress {
        p.publish_metrics(&ProgressMetrics {
            status: TrainingStatus::Running,
            phase: TrainingPhase::Pass2Backbone,
            active_tier: Some(active_tier),
            progress_permille: permille.min(1000),
            step: step as u32,
            total_steps: total_steps as u32,
            stage_step: step as u32,
            stage_total_steps: total_steps as u32,
            best_loss,
            ..ProgressMetrics::default()
        });
    }
}

pub(super) fn allocate_stage_steps(stages: &[usize], total_steps: usize) -> Vec<(usize, usize)> {
    let raw_weights: Vec<usize> = stages
        .iter()
        .map(|&ti| match ti {
            0 => 1,
            1 | 2 => 2,
            _ => 3,
        })
        .collect();
    let w_sum: usize = raw_weights.iter().sum::<usize>().max(1);
    let mut assigned = vec![1_usize; stages.len()];
    let rem = total_steps.saturating_sub(stages.len());
    let mut used = stages.len();
    for (i, &rw) in raw_weights.iter().enumerate() {
        if i + 1 == stages.len() {
            assigned[i] += total_steps.saturating_sub(used);
        } else {
            let extra = (rem * rw) / w_sum;
            assigned[i] += extra;
            used += extra;
        }
    }
    stages.iter().copied().zip(assigned).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::canonical_ltas_33_centers_hz;
    use crate::solver::progress::TrainingStatus;

    #[test]
    fn test_windows_stage_plan_metrics_and_progress() {
        let plan = allocate_stage_steps(&[0, 1, 2, 3], 24);
        assert_eq!(plan.len(), 4);
        assert_eq!(plan.iter().map(|&(_, s)| s).sum::<usize>(), 24);
        assert!(plan.iter().all(|&(_, s)| s >= 1));

        let single = allocate_stage_steps(&[3], 5);
        assert_eq!(single, vec![(3, 5)]);

        let prog = AtomicDnprProgress::new();
        publish_progress(Some(&prog), 420, 7, 20, 0.0125, Tier::Pro);
        let snap = prog.snapshot();
        assert_eq!(snap.status(), TrainingStatus::Running);
        assert_eq!(snap.phase(), TrainingPhase::Pass2Backbone);
        assert_eq!(snap.progress_permille(), 420);

        let runner = TrainRunner::cpu();
        let n = 1_024_usize;
        let centers = canonical_ltas_33_centers_hz();
        let make_sig = |phase: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let sec = (t as f64) / 48_000.0;
                    centers
                        .iter()
                        .enumerate()
                        .map(|(b, &fc)| {
                            0.02 * (core::f64::consts::TAU * fc * sec + phase + 0.31 * (b as f64))
                                .sin()
                        })
                        .sum()
                })
                .collect()
        };
        let in_l = make_sig(0.0);
        let in_r = make_sig(0.5);
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.7 * v + 0.1 * v * v * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.65 * v - 0.1 * v * v * v).collect();
        let noise = vec![1.0e-4_f64; n];
        let cfg = Pass2OptConfig {
            sample_rate_hz: 48_000.0,
            warmup_frames: 32,
            active_quads: 6,
            ..Pass2OptConfig::default()
        };
        let full_loss = CompositeLossEvaluator::new_for_homotopy(
            &tgt_l,
            &tgt_r,
            Some(&noise),
            Some(&noise),
            CompositeLossConfig {
                sample_rate_hz: cfg.sample_rate_hz,
                warmup_frames: cfg.warmup_frames,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let ctx = HybridCtx {
            runner: &runner,
            pairs: StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            noise_pair: Some((&noise, &noise)),
            full_loss: &full_loss,
            config: cfg,
            warmup: 32,
            max_tau: 512.0,
            max_tier: 1,
            fast_mode: true,
            progress: Some(&prog),
        };

        let full_w = build_windows(&ctx, 2_048, 3).unwrap();
        assert_eq!(full_w.len(), 1);
        assert_eq!(full_w[0].len(), n);
        assert!(full_w[0].loss_eval.is_none());

        let sub_w = build_windows(&ctx, 512, 3).unwrap();
        assert_eq!(sub_w.len(), 3);
        assert_eq!((sub_w[0].start, sub_w[2].end), (0, n));
        assert!(
            sub_w
                .iter()
                .all(|w| w.len() == 512 && w.loss_eval.is_some())
        );

        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].tone_dir = [0.6; 4];
        model.w_ro[0][6] = 0.25;
        model.w_ro[0][7] = -0.20;
        let w1 = CompositeLossWeights::homotopy_at(1.0);
        let (acc_s, l_s, b_s) = eval_full_capsule_metrics(&ctx, &model, false, w1).unwrap();
        let (acc_j, l_j, b_j) = eval_full_capsule_metrics(&ctx, &model, true, w1).unwrap();
        assert!(acc_s.is_finite() && l_s.is_finite() && b_s.is_finite());
        assert!(acc_j.is_finite() && l_j.is_finite() && b_j.is_finite());

        let (sw_l, gx) = eval_single_tier_with_grad(&ctx, &sub_w[0], &model, w1, None, 0).unwrap();
        assert!(sw_l.is_finite() && gx.is_empty());
        refresh_active_tiers_varpro(&ctx, Some(&sub_w[0]), w1, &mut model, 1, false).unwrap();
        let (pm_a, pm_l, pm_b) = eval_tier_prefix_metrics(&ctx, &model, 0, true, w1).unwrap();
        assert!(pm_a.is_finite() && pm_l.is_finite() && pm_b.is_finite());
        sync_tier_override_from_top_level(&mut model, 0);
        let exp_l = explore_candidate_short(&ctx, &mut model, &sub_w[0], 0, 2).unwrap();
        assert!(exp_l.is_finite());

        // Silence-collapse rejection even when joint objective improves; genuine improvement accepted.
        assert!(!is_pass2_metrics_acceptable(
            (-1.0, 2000.0, -15.2),
            (-1.5, 1394.0, 0.0018),
            true,
            false
        ));
        assert!(is_pass2_metrics_acceptable(
            (-1.0, 10.0, -15.2),
            (-1.2, 9.5, -14.9),
            true,
            true
        ));
    }
}
