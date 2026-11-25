//! Four-phase Pass-2 hybrid trajectory optimizer (`Phase 2A..2D`):
//! 1. **Phase 2A:** Generic Sobol/log-spaced prescreening + Top-K diverse basin selection.
//! 2. **Phase 2B:** Progressive `4 -> 6 -> 8 -> 10` core-quad growth, chunked `AdamW` with
//!    8-term cosine homotopy, periodic `uplo='U'` VarPro readout refresh, and Arena-Scoped `PCGrad`.
//! 3. **Phase 2C:** Alternating block-coordinate `L-SSBroyden` (`phi = 0`, Strong Wolfe, no `PCGrad`)
//!    paired with monotone-gated VarPro refresh and `BestUntilDiminishingReturns` stopping.
//! 4. **Phase 2D:** Conditioned `u`-space STE `fp16` QAT polish under monotone non-regression.

mod phase2a;
mod windows;

use crate::error::TrainError;
use crate::linalg::adamw::{AdamWConfig, AdamWState, cosine_lr_with_warmup};
use crate::linalg::coords::ScaledCoordSpace;
use crate::linalg::ssbroyden::{SsBroydenConfig, SsBroydenState};
use crate::losses::{CompositeLossConfig, CompositeLossEvaluator, CompositeLossWeights};
use crate::runner::TrainRunner;

use dnam_vm::Tier;

use super::super::model::{SonaModelF64, StereoTrainingPairs};
use super::super::progress::AtomicDnprProgress;
use super::matryoshka::{MatryoshkaEvalCtx, eval_joint_matryoshka_and_grad, expand_model_to_tier};
use super::prescreen::tier_index_for_quads;
use super::theta::{TIER_QUADS, has_active_op_t};
use super::{DIMINISHING_RETURNS_DELTA_DB, Pass2OptConfig};
use phase2a::run_phase2a;
use windows::{
    HybridCtx, allocate_stage_steps, build_windows, eval_full_capsule_metrics,
    eval_single_tier_with_grad, eval_tier_prefix_metrics, is_pass2_metrics_acceptable,
    phase2_layout_for_tier, publish_progress, refresh_active_tiers_varpro,
    refresh_lower_tiers_varpro_full, run_phase2d_guarded_qat,
};

/// Raw outcome of the hybrid Phase 2A–2D pipeline prior to timing annotation.
pub(super) struct HybridRunOutcome {
    pub epochs_run: usize,
    pub qat_polish_steps_run: usize,
    pub qat_baseline_loss: f64,
    pub converged_diminishing_returns: bool,
    pub best_besr_db: f64,
    pub best_loss: f64,
    pub winning_seed_index: usize,
}

fn maybe_update_stage_best(
    single_chunk: bool,
    cand_loss: f64,
    work_model: &SonaModelF64,
    best_stage_loss: &mut f64,
    best_stage_model: &mut SonaModelF64,
) {
    if !cand_loss.is_finite() {
        return;
    }
    if single_chunk {
        if cand_loss < *best_stage_loss {
            *best_stage_loss = cand_loss;
            *best_stage_model = work_model.clone();
        }
    } else if !best_stage_loss.is_finite() || cand_loss <= 2.5 * (*best_stage_loss) + 0.5 {
        *best_stage_loss = (*best_stage_loss).min(cand_loss);
        *best_stage_model = work_model.clone();
    }
}

fn run_phase2b_progressive_adamw(
    ctx: &HybridCtx<'_>,
    model: &mut SonaModelF64,
    global_epochs: &mut usize,
    linear_only: bool,
) -> Result<(f64, f64, f64), TrainError> {
    let use_joint = ctx.config.use_progressive_matryoshka && ctx.max_tier > 0;
    let stages: Vec<usize> = if linear_only || !use_joint {
        vec![ctx.max_tier]
    } else {
        (0..=ctx.max_tier).collect()
    };
    let mut total_steps = if ctx.fast_mode {
        ctx.config
            .adamw_steps
            .min(ctx.config.max_epochs.max(1) * 4)
            .max(stages.len())
    } else {
        ctx.config.adamw_steps.max(stages.len() * 4)
    };
    if linear_only {
        total_steps = total_steps.min(16);
    }
    let stage_plan = allocate_stage_steps(&stages, total_steps);
    let chunks = build_windows(ctx, ctx.config.chunk_frames, 4)?;
    let refresh_every = ctx.config.varpro_refresh_interval.max(1);

    let init_m = eval_full_capsule_metrics(
        ctx,
        model,
        use_joint,
        CompositeLossWeights::homotopy_at(1.0),
    )?;
    let initial_snapshot = model.clone();

    let adamw_cfg = AdamWConfig {
        lr_max: 3.0e-3,
        lr_min: 1.0e-4,
        warmup_steps: (total_steps / 10).min(4),
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
        grad_clip_norm: 10.0,
    };

    let single_chunk = chunks.len() == 1;
    let mut global_step = 0_usize;
    let mut prev_stage: Option<usize> = None;
    for (stage_ti, stage_steps) in stage_plan {
        if let Some(pt) = prev_stage {
            expand_model_to_tier(model, pt, stage_ti);
            refresh_active_tiers_varpro(
                ctx,
                None,
                CompositeLossWeights::homotopy_at(0.5),
                model,
                stage_ti,
                false,
            )?;
        }
        prev_stage = Some(stage_ti);

        let stage_entry_model = model.clone();
        let stage_entry_m = eval_tier_prefix_metrics(
            ctx,
            model,
            stage_ti,
            use_joint,
            CompositeLossWeights::homotopy_at(1.0),
        )?;

        let q_t = TIER_QUADS[stage_ti];
        let has_op_t = has_active_op_t(model, q_t);
        let include_extras = has_op_t || model.active_kautz_pairs > 0;
        let include_readouts = has_op_t || linear_only;
        let layout = phase2_layout_for_tier(
            model,
            stage_ti,
            linear_only,
            include_extras,
            include_readouts,
            ctx.max_tau,
            ctx.kautz_r_max,
        );
        let (init_phys, bounds) = layout.pack(model);
        let scales = layout.coord_scales(chunks[0].len());
        let (space, mut u) = ScaledCoordSpace::new(
            "run_phase2b_progressive_adamw",
            &init_phys,
            &bounds,
            Some(&scales),
        )?;
        let ro_mask = layout.readout_coupled_mask();
        let dim = layout.dim();
        let mut x = vec![0.0_f64; dim];
        let mut jac = vec![0.0_f64; dim];
        let mut gu = vec![0.0_f64; dim];
        let mut adamw = AdamWState::new(dim);

        let mut work_model = model.clone();
        let mut best_stage_model = model.clone();
        let mut best_stage_loss = f64::INFINITY;

        for local_step in 0..stage_steps {
            let p = (global_step as f64) / ((total_steps.saturating_sub(1)).max(1) as f64);
            let weights = CompositeLossWeights::homotopy_at(p);
            let win = &chunks[global_step % chunks.len()];
            let pairs_w = win.pairs(ctx.pairs);
            let loss_w = win.loss(ctx.full_loss);

            space.decode(&u, &mut x, &mut jac);
            layout.unpack(&x, &mut work_model);
            let step_loss = if use_joint && stage_ti > 0 {
                let res = eval_joint_matryoshka_and_grad(
                    &MatryoshkaEvalCtx {
                        runner: ctx.runner,
                        pairs: pairs_w,
                        loss_eval: loss_w,
                        weights,
                        layout: &layout,
                        space_jac: Some((&space, &jac)),
                        max_tier: stage_ti,
                        use_pcgrad: ctx.config.use_pcgrad,
                    },
                    &work_model,
                    true,
                )?;
                if ctx.config.use_pcgrad {
                    gu.copy_from_slice(&res.grad_out);
                } else {
                    space.pullback_grad(&res.grad_out, &jac, &mut gu);
                }
                res.joint_loss
            } else {
                let (l_val, gx) = eval_single_tier_with_grad(
                    ctx,
                    win,
                    &work_model,
                    weights,
                    Some(&layout),
                    stage_ti,
                )?;
                space.pullback_grad(&gx, &jac, &mut gu);
                l_val
            };

            maybe_update_stage_best(
                single_chunk,
                step_loss,
                &work_model,
                &mut best_stage_loss,
                &mut best_stage_model,
            );

            let local_warm = (((local_step + 1) as f64) / 3.0).min(1.0);
            let lr = cosine_lr_with_warmup(global_step, total_steps, &adamw_cfg) * local_warm;
            adamw.step_u(&mut u, &mut gu, lr, &adamw_cfg, Some(space.scales()));
            *global_epochs += 1;
            global_step += 1;

            if (local_step + 1) % refresh_every == 0 || local_step + 1 == stage_steps {
                space.decode(&u, &mut x, &mut jac);
                layout.unpack(&x, &mut work_model);
                refresh_active_tiers_varpro(
                    ctx,
                    Some(win),
                    weights,
                    &mut work_model,
                    stage_ti,
                    use_joint,
                )?;
                adamw.reset_moments_mask(&ro_mask);
                if include_readouts {
                    let (re_x, _) = layout.pack(&work_model);
                    space.encode(&re_x, &mut u);
                }
                let post_loss = if use_joint && stage_ti > 0 {
                    eval_joint_matryoshka_and_grad(
                        &MatryoshkaEvalCtx {
                            runner: ctx.runner,
                            pairs: pairs_w,
                            loss_eval: loss_w,
                            weights,
                            layout: &layout,
                            space_jac: None,
                            max_tier: stage_ti,
                            use_pcgrad: false,
                        },
                        &work_model,
                        false,
                    )?
                    .joint_loss
                } else {
                    eval_single_tier_with_grad(ctx, win, &work_model, weights, None, stage_ti)?.0
                };
                maybe_update_stage_best(
                    single_chunk,
                    post_loss,
                    &work_model,
                    &mut best_stage_loss,
                    &mut best_stage_model,
                );
            }

            let perm = ((global_step * 500) / total_steps.max(1)).clamp(50, 500) as u16;
            publish_progress(
                ctx.progress,
                perm,
                global_step,
                total_steps,
                best_stage_loss,
                Tier::Ultra,
            );
        }

        let stage_end_m = eval_tier_prefix_metrics(
            ctx,
            &best_stage_model,
            stage_ti,
            use_joint,
            CompositeLossWeights::homotopy_at(1.0),
        )?;
        if stage_end_m.0.is_finite()
            && (!stage_entry_m.0.is_finite() || stage_end_m.0 <= stage_entry_m.0)
        {
            *model = best_stage_model;
        } else {
            *model = stage_entry_model;
        }
    }

    refresh_active_tiers_varpro(
        ctx,
        None,
        CompositeLossWeights::homotopy_at(1.0),
        model,
        ctx.max_tier,
        use_joint,
    )?;
    let final_m = eval_full_capsule_metrics(
        ctx,
        model,
        use_joint,
        CompositeLossWeights::homotopy_at(1.0),
    )?;
    if !is_pass2_metrics_acceptable(init_m, final_m, use_joint, false) {
        *model = initial_snapshot;
        if use_joint {
            refresh_lower_tiers_varpro_full(ctx, CompositeLossWeights::homotopy_at(1.0), model)?;
            return eval_full_capsule_metrics(
                ctx,
                model,
                use_joint,
                CompositeLossWeights::homotopy_at(1.0),
            );
        }
        Ok(init_m)
    } else {
        Ok(final_m)
    }
}

fn run_phase2c_ssbroyden(
    ctx: &HybridCtx<'_>,
    model: &mut SonaModelF64,
    global_epochs: &mut usize,
    init_metrics: (f64, f64, f64),
    linear_only: bool,
) -> Result<(f64, f64, bool), TrainError> {
    let final_ti = ctx.max_tier;
    let use_joint = ctx.config.use_progressive_matryoshka && final_ti > 0;
    let weights = CompositeLossWeights::homotopy_at(1.0);
    let n = ctx.pairs.aligned_len();
    let block_len = if ctx.fast_mode { n } else { n.min(4_096) };
    let max_blocks = if ctx.fast_mode {
        1
    } else if linear_only {
        ctx.config.max_epochs.div_ceil(4).clamp(1, 4)
    } else {
        ctx.config.max_epochs.div_ceil(4).clamp(1, 2)
    };
    let blocks = build_windows(ctx, block_len, max_blocks)?;
    let iters_per_block = if ctx.fast_mode {
        ctx.config
            .ssbroyden_block_iters
            .min(ctx.config.max_epochs.max(2))
    } else {
        ctx.config.ssbroyden_block_iters.clamp(4, 25)
    };

    let (mut best_obj, mut best_loss, mut best_besr_db) = init_metrics;
    let mut ultra_loss_floor = init_metrics.1;
    let mut ultra_besr_floor = init_metrics.2;
    let mut best_model = model.clone();
    let mut work_model = model.clone();
    let mut prev_besr_db = best_besr_db;
    let mut streak = 0_usize;
    let mut converged = false;

    for blk_idx in 0..max_blocks {
        let win = &blocks[blk_idx % blocks.len()];
        let pairs_w = win.pairs(ctx.pairs);
        let loss_w = win.loss(ctx.full_loss);
        let q_t = TIER_QUADS[final_ti];
        let has_op_t = has_active_op_t(&work_model, q_t);
        let include_extras = has_op_t || work_model.active_kautz_pairs > 0;
        let include_readouts = has_op_t || linear_only;
        let layout = phase2_layout_for_tier(
            &work_model,
            final_ti,
            linear_only,
            include_extras,
            include_readouts,
            ctx.max_tau,
            ctx.kautz_r_max,
        );
        let (init_phys, bounds) = layout.pack(&work_model);
        let scales = layout.coord_scales(win.len());

        let ssb_cfg = SsBroydenConfig {
            max_iters: iters_per_block,
            history_m: ctx.config.ssbroyden_history_m.clamp(16, 32),
            dense_dim_threshold: 512,
            broyden_phi: 0.0,
            wolfe_c1: 1.0e-4,
            wolfe_c2: 0.9,
            max_line_search: 8,
            tau_min: 1.0e-3,
            tau_max: 1.0e3,
            curvature_tol: 1.0e-10,
            patience: ctx.config.diminishing_patience.max(2),
            max_step_norm: 4.0,
            ..SsBroydenConfig::default()
        };

        let mut cand_model = work_model.clone();
        let res = SsBroydenState::minimize_scaled(
            &init_phys,
            &bounds,
            Some(&scales),
            &ssb_cfg,
            |cand_x, grad_opt| {
                layout.unpack(cand_x, &mut cand_model);
                if use_joint {
                    let out = eval_joint_matryoshka_and_grad(
                        &MatryoshkaEvalCtx {
                            runner: ctx.runner,
                            pairs: pairs_w,
                            loss_eval: loss_w,
                            weights,
                            layout: &layout,
                            space_jac: None,
                            max_tier: final_ti,
                            use_pcgrad: false,
                        },
                        &cand_model,
                        grad_opt.is_some(),
                    )?;
                    if let Some(g_dst) = grad_opt {
                        g_dst.copy_from_slice(&out.grad_out);
                    }
                    Ok(out.joint_loss)
                } else {
                    let g_layout = grad_opt.as_ref().map(|_| &layout);
                    let (loss_val, gx) = eval_single_tier_with_grad(
                        ctx,
                        win,
                        &cand_model,
                        weights,
                        g_layout,
                        final_ti,
                    )?;
                    if let Some(g_dst) = grad_opt {
                        g_dst.copy_from_slice(&gx);
                    }
                    Ok(loss_val)
                }
            },
        )?;
        *global_epochs += res.iters_run.max(1);

        layout.unpack(&res.best_params, &mut cand_model);
        refresh_active_tiers_varpro(ctx, None, weights, &mut cand_model, final_ti, use_joint)?;
        let (cand_obj, cand_loss, cand_besr_db) =
            eval_full_capsule_metrics(ctx, &cand_model, use_joint, weights)?;
        let accepted = is_pass2_metrics_acceptable(
            (best_obj, ultra_loss_floor, ultra_besr_floor),
            (cand_obj, cand_loss, cand_besr_db),
            use_joint,
            true,
        );
        if accepted {
            best_obj = cand_obj;
            best_loss = cand_loss;
            best_besr_db = cand_besr_db;
            ultra_loss_floor = ultra_loss_floor.min(cand_loss);
            ultra_besr_floor = ultra_besr_floor.min(cand_besr_db);
            best_model = cand_model.clone();
            work_model = cand_model;
        } else {
            work_model = best_model.clone();
        }

        if prev_besr_db.is_finite()
            && (prev_besr_db - cand_besr_db).abs() < DIMINISHING_RETURNS_DELTA_DB
        {
            streak += 1;
        } else {
            streak = 0;
        }
        prev_besr_db = cand_besr_db;
        let perm = 500 + (((blk_idx + 1) * 400) / max_blocks) as u16;
        publish_progress(
            ctx.progress,
            perm,
            blk_idx + 1,
            max_blocks,
            best_loss,
            Tier::Ultra,
        );
        if streak >= ctx.config.diminishing_patience {
            converged = true;
            break;
        }
    }

    *model = best_model;
    Ok((best_loss, best_besr_db, converged))
}

/// Executes the complete 4-phase hybrid Pass-2 solver (`Phase 2A..2D`).
pub(super) fn run_hybrid_pass2_optimize(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: StereoTrainingPairs<'_>,
    noise_pair: Option<(&[f64], &[f64])>,
    config: Pass2OptConfig,
    progress: Option<&AtomicDnprProgress>,
) -> Result<HybridRunOutcome, TrainError> {
    let n = pairs.aligned_len();
    if n == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "run_pass2_optimize",
            expected: 1,
            actual: 0,
        });
    }
    let q_target = config.active_quads.clamp(1, 10);
    let max_tier = tier_index_for_quads(q_target);
    let warmup = config.warmup_frames.min(n / 4);
    let max_tau = ((n / 2).min(4_096) as f64).max(8.0);
    let fast_mode = config.max_epochs <= 6;
    let kautz_r_max = super::super::pass1_probe::kautz_fit::estimate_physical_kautz_r_ceil(
        &pairs.in_l[..n],
        &pairs.in_r[..n],
        &pairs.target_l[..n],
        &pairs.target_r[..n],
        warmup,
        config.sample_rate_hz,
    );

    let noise_l = noise_pair.map(|(nl, _)| nl);
    let noise_r = noise_pair.map(|(_, nr)| nr);
    let full_loss = CompositeLossEvaluator::new_for_homotopy(
        &pairs.target_l[..n],
        &pairs.target_r[..n],
        noise_l,
        noise_r,
        CompositeLossConfig {
            sample_rate_hz: config.sample_rate_hz,
            warmup_frames: warmup,
            ..CompositeLossConfig::default()
        },
    )?;

    let ctx = HybridCtx {
        runner,
        pairs,
        noise_pair: noise_l.map(|nl| &nl[..n]).zip(noise_r.map(|nr| &nr[..n])),
        full_loss: &full_loss,
        config,
        warmup,
        max_tau,
        kautz_r_max,
        max_tier,
        fast_mode,
        progress,
    };

    let (mut work_model, winning_seed_index, linear_only) = run_phase2a(&ctx, model)?;
    publish_progress(progress, 50, 0, 1, f64::INFINITY, Tier::Lite);

    let mut epochs_run = 0_usize;
    let phase2b_metrics =
        run_phase2b_progressive_adamw(&ctx, &mut work_model, &mut epochs_run, linear_only)?;
    let (phase2c_loss, phase2c_besr_db, converged) = run_phase2c_ssbroyden(
        &ctx,
        &mut work_model,
        &mut epochs_run,
        phase2b_metrics,
        linear_only,
    )?;

    let (qat_polish_steps_run, qat_baseline_loss, best_loss, best_besr_db) =
        run_phase2d_guarded_qat(
            &ctx,
            &mut work_model,
            phase2c_loss,
            phase2c_besr_db,
            linear_only,
        )?;

    *model = work_model;
    publish_progress(
        progress,
        1_000,
        epochs_run,
        epochs_run.max(1),
        best_loss,
        Tier::Ultra,
    );

    Ok(HybridRunOutcome {
        epochs_run,
        qat_polish_steps_run,
        qat_baseline_loss,
        converged_diminishing_returns: converged,
        best_besr_db,
        best_loss,
        winning_seed_index,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::pass3_varpro::run_pass3_varpro_readout;
    use super::super::eval::eval_model_pass2_audio;
    use super::super::prescreen::generate_generic_sobol_seeds_with_offset;
    use super::super::run_pass2_optimize;
    use super::super::varpro_sync::materialize_tier_view;
    use super::*;
    use crate::losses::canonical_ltas_33_centers_hz;
    use crate::solver::pass1_probe::{Pass1ProbeConfig, run_pass1_probe_with_hint};

    fn render_nonlinear_multimode_target(in_l: &[f64], in_r: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Nonlinear waveshaper + delayed 16-mode resonant bank spanning `[100 Hz, 8 kHz]` so
        // `Lite (8) -> Full (12) -> Pro/Ultra (24/32)` progressively capture more modes natively.
        const MODES: [(f64, f64, f64); 16] = [
            (0.945, 0.04, 0.14),
            (0.940, 0.09, -0.13),
            (0.935, 0.15, 0.12),
            (0.930, 0.22, -0.11),
            (0.925, 0.30, 0.10),
            (0.920, 0.40, -0.09),
            (0.915, 0.52, 0.09),
            (0.910, 0.66, -0.08),
            (0.905, 0.82, 0.08),
            (0.900, 1.00, -0.07),
            (0.895, 1.20, 0.07),
            (0.890, 1.42, -0.06),
            (0.885, 1.66, 0.06),
            (0.880, 1.92, -0.05),
            (0.875, 2.20, 0.05),
            (0.870, 2.50, -0.04),
        ];
        let n = in_l.len().min(in_r.len());
        let mut tgt_l = vec![0.0_f64; n];
        let mut tgt_r = vec![0.0_f64; n];
        let mut st_l = [[0.0_f64; 2]; 16];
        let mut st_r = [[0.0_f64; 2]; 16];
        let mut ring_l = [0.0_f64; 64];
        let mut ring_r = [0.0_f64; 64];
        for t in 0..n {
            let nl_l = 0.55 * (1.8 * in_l[t]).tanh();
            let nl_r = 0.50 * (1.6 * in_r[t]).tanh();
            let del_l = ring_l[(t + 48) & 63];
            let del_r = ring_r[(t + 48) & 63];
            ring_l[t & 63] = nl_l;
            ring_r[t & 63] = nl_r;
            let (mut acc_l, mut acc_r) = (0.45 * nl_l, 0.45 * nl_r);
            for (m, &(r, th, c_re)) in MODES.iter().enumerate() {
                let bg = (1.0 - r * r).sqrt();
                let (st, ct) = th.sin_cos();
                let nl_nxt = r * (ct * st_l[m][0] - st * st_l[m][1]) + bg * del_l;
                st_l[m] = [nl_nxt, r * (st * st_l[m][0] + ct * st_l[m][1])];
                let nr_nxt = r * (ct * st_r[m][0] - st * st_r[m][1]) + bg * del_r;
                st_r[m] = [nr_nxt, r * (st * st_r[m][0] + ct * st_r[m][1])];
                acc_l += c_re * nl_nxt;
                acc_r += c_re * nr_nxt;
            }
            tgt_l[t] = acc_l;
            tgt_r[t] = acc_r;
        }
        (tgt_l, tgt_r)
    }

    #[test]
    fn test_hybrid_multiseed_besr_consistency_and_native_matryoshka_monotonicity() {
        let runner = TrainRunner::cpu();
        let n = 1_024_usize;
        let warmup = 64_usize;
        let centers = canonical_ltas_33_centers_hz();
        let make_signal = |ch_phase: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let sec = (t as f64) / 48_000.0;
                    centers
                        .iter()
                        .enumerate()
                        .map(|(b, &fc)| {
                            let ph = ch_phase + 0.43 * (b as f64) + 1.17 * (b as f64).sin();
                            let amp = if fc <= 10_000.0 { 0.025 } else { 0.008 };
                            amp * (core::f64::consts::TAU * fc * sec + ph).sin()
                        })
                        .sum()
                })
                .collect()
        };

        let in_l = make_signal(0.0);
        let in_r = make_signal(0.85);
        let (tgt_l, tgt_r) = render_nonlinear_multimode_target(&in_l, &in_r);
        let base_p1 = run_pass1_probe_with_hint(
            &in_l,
            &in_r,
            &tgt_l,
            &tgt_r,
            Pass1ProbeConfig {
                sample_rate_hz: 48_000.0,
                max_fir_taps: 128,
                max_kautz_pairs: 24,
                hint: None,
            },
        )
        .unwrap()
        .model;

        let loss_eval = CompositeLossEvaluator::new_for_homotopy(
            &tgt_l,
            &tgt_r,
            None,
            None,
            CompositeLossConfig {
                sample_rate_hz: 48_000.0,
                warmup_frames: warmup,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let w1 = CompositeLossWeights::homotopy_at(1.0);
        let seed_offsets = [0_usize, 32, 64];
        let s0 = generate_generic_sobol_seeds_with_offset(32, 48_000.0, seed_offsets[0]);
        let s1 = generate_generic_sobol_seeds_with_offset(32, 48_000.0, seed_offsets[1]);
        let s2 = generate_generic_sobol_seeds_with_offset(32, 48_000.0, seed_offsets[2]);
        assert!(s0 != s1 && s1 != s2 && s0 != s2);

        let pairs = StereoTrainingPairs {
            in_l: &in_l,
            in_r: &in_r,
            target_l: &tgt_l,
            target_r: &tgt_r,
        };

        let mut ultra_besrs = [0.0_f64; 3];
        let mut pre3_ultra_besrs = [0.0_f64; 3];
        let mut all_tier_besrs = [[0.0_f64; 4]; 3];
        for (s_idx, dst) in ultra_besrs.iter_mut().enumerate() {
            let mut model = base_p1.clone();
            let cfg = Pass2OptConfig {
                num_prescreen_seeds: 32,
                prescreen_seed_offset: seed_offsets[s_idx],
                max_epochs: 8,
                diminishing_patience: 3,
                sample_rate_hz: 48_000.0,
                warmup_frames: warmup,
                active_quads: 10,
                qat_polish_steps: 4,
                top_k_seeds: 4,
                adamw_steps: 20,
                chunk_frames: 1_024,
                varpro_refresh_interval: 10,
                use_progressive_matryoshka: true,
                use_pcgrad: true,
                ssbroyden_block_iters: 6,
                ssbroyden_history_m: 16,
            };
            let rep = run_pass2_optimize(&runner, &mut model, &pairs, cfg, None).unwrap();
            assert!(rep.best_loss.is_finite());

            let pre3_view = materialize_tier_view(&model, 3);
            let pre3_buf =
                eval_model_pass2_audio(&runner, &pre3_view, &in_l, &in_r, TIER_QUADS[3]).unwrap();
            let pre3_bd = loss_eval
                .eval_and_adjoint_with_weights(&pre3_buf.pred_l, &pre3_buf.pred_r, w1, None, None)
                .unwrap();
            pre3_ultra_besrs[s_idx] = 10.0 * pre3_bd.besr.max(1.0e-12).log10();

            run_pass3_varpro_readout(&runner, &mut model, &in_l, &in_r, &tgt_l, &tgt_r, warmup)
                .unwrap();

            let mut tier_besr = [0.0_f64; 4];
            for ti in 0..4 {
                let view = materialize_tier_view(&model, ti);
                let buf =
                    eval_model_pass2_audio(&runner, &view, &in_l, &in_r, TIER_QUADS[ti]).unwrap();
                let bd = loss_eval
                    .eval_and_adjoint_with_weights(&buf.pred_l, &buf.pred_r, w1, None, None)
                    .unwrap();
                tier_besr[ti] = 10.0 * bd.besr.max(1.0e-12).log10();
            }
            all_tier_besrs[s_idx] = tier_besr;
            *dst = tier_besr[3];
        }

        for s_idx in 0..3 {
            assert!(
                pre3_ultra_besrs[s_idx] < -20.0,
                "seed {s_idx} pre-Pass-3 Ultra bESR too weak: {:.3} dB",
                pre3_ultra_besrs[s_idx]
            );
            for ti in 1..4 {
                assert!(
                    all_tier_besrs[s_idx][ti] <= all_tier_besrs[s_idx][ti - 1] + 0.10,
                    "seed {s_idx} pre-Pass-5 tier {} regressed vs {}: besr={:?}",
                    ti,
                    ti - 1,
                    all_tier_besrs[s_idx]
                );
            }
            assert!(
                all_tier_besrs[s_idx][3] < -25.0,
                "seed {s_idx} Ultra bESR too weak: besr={:?}",
                all_tier_besrs[s_idx]
            );
        }

        let mean = ultra_besrs.iter().sum::<f64>() / 3.0;
        let var = ultra_besrs
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / 3.0;
        let std_db = var.sqrt();
        assert!(
            std_db < 0.3,
            "multi-seed Ultra bESR std {std_db:.4} dB >= 0.3 dB ({ultra_besrs:?})"
        );
    }
}
