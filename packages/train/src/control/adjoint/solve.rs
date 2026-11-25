//! Differentiable 31-slot stage-target optimization (`eval_stage_targets_loss_and_grad`,
//! `solve_point_stage_targets_adjoint`) using `Pass2` adjoints + `CompositeLossEvaluator` +
//! `SsBroydenState` + optional zero-alloc persistent `SonaVm` micro-polish, plus streaming
//! `3 kHz` VM trajectory evaluation (`eval_control_trajectory_3khz_vm`).

use dnam_vm::Tier;
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::DefaultMathBackend;

use crate::control::stability::{StageTargetBounds, eval_corner_barrier_loss_and_grad};
use crate::error::TrainError;
use crate::linalg::{OptResult, ParamBoundSpec, SsBroydenConfig, SsBroydenState};
use crate::losses::{
    CompositeLossBreakdown, CompositeLossConfig, CompositeLossEvaluator, CompositeLossWeights,
};
use crate::runner::TrainRunner;
use crate::solver::model::SonaModelF64;
use crate::solver::pass2_opt::eval::{
    Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio,
};

use super::hoist::{
    StageTargetAndLatentGrads, hoist_stage_targets_onto_model,
    pullback_stage_targets_and_latent_grads,
};

/// Evaluation result returned by [`eval_stage_targets_loss_and_grad`].
#[derive(Clone, Debug)]
pub struct StageTargetEvalResult {
    /// Perceptual composite loss breakdown across all active terms.
    pub breakdown: CompositeLossBreakdown,
    /// Analytical gradients wrt `stage_targets[0..32]` and `SonaStageLatentModsF64`, if requested.
    pub grads: Option<StageTargetAndLatentGrads>,
}

/// Configuration for [`solve_point_stage_targets_adjoint`].
#[derive(Clone, Debug, PartialEq)]
pub struct StageTargetSolveConfig {
    /// Active dispatch slot indices in `0..31` to optimize (`slot 31` always forced `0.0`).
    pub active_slots: Vec<usize>,
    /// Execution tier (`Lite`, `Full`, `Pro`, `Ultra`).
    pub tier: Tier,
    /// Pre-roll warmup frames excluded from loss scoring.
    pub warmup_frames: usize,
    /// Multi-objective weights passed to [`CompositeLossEvaluator`].
    pub loss_weights: CompositeLossWeights,
    /// Smooth logarithmic corner barrier weight on `StageTargetBounds`.
    pub barrier_weight: f64,
    /// Quadratic L2 regularization weight anchoring `stage_targets` toward `0.0`.
    pub l2_reg: f64,
    /// Bounded quasi-Newton solver configuration (`SsBroydenState`).
    pub opt: SsBroydenConfig,
    /// Number of zero-alloc persistent `SonaVm` coordinate micro-polish sweeps (`0` skips).
    pub vm_micro_polish_iters: usize,
    /// Fixed FD step for persistent-`SonaVm` micro-polish (`1e-2` canonical).
    pub vm_micro_polish_eps: f64,
}

impl Default for StageTargetSolveConfig {
    fn default() -> Self {
        Self {
            active_slots: (0..(STAGE_TARGET_SLOTS - 1)).collect(),
            tier: Tier::Ultra,
            warmup_frames: 64,
            loss_weights: CompositeLossWeights::from_config(&CompositeLossConfig::default()),
            barrier_weight: 1.0e-4,
            l2_reg: 1.0e-5,
            opt: SsBroydenConfig {
                max_iters: 24,
                ..SsBroydenConfig::default()
            },
            vm_micro_polish_iters: 0,
            vm_micro_polish_eps: 1.0e-2,
        }
    }
}

/// Result bundle produced by [`solve_point_stage_targets_adjoint`].
#[derive(Clone, Debug, PartialEq)]
pub struct StageTargetSolveReport {
    /// Solved `32`-slot `stage_targets` vector (`slot 31 == 0.0`).
    pub stage_targets: [f64; STAGE_TARGET_SLOTS],
    /// Initial objective loss (composite + barrier + L2) before optimization.
    pub initial_loss: f64,
    /// Final objective loss after `SsBroydenState` and optional `SonaVm` micro-polish.
    pub final_loss: f64,
    /// Number of quasi-Newton iterations completed.
    pub iters_run: usize,
    /// Number of objective/gradient evaluations performed.
    pub evals: usize,
    /// Whether the final persistent `SonaVm` micro-polish improved the objective.
    pub micro_polish_accepted: bool,
}

/// Borrowed input bundle for [`eval_stage_targets_loss_and_grad`].
#[derive(Clone, Copy)]
pub struct StageTargetEvalContext<'a> {
    /// Compiled `SONA` CPU/GPU forward and adjoint runner.
    pub runner: &'a TrainRunner,
    /// Unmodulated baseline continuous `f64` model.
    pub base: &'a SonaModelF64,
    /// Target runtime tier evaluated across Pass 2.
    pub tier: Tier,
    /// Left reference/stimulus channel in `f64`.
    pub in_l: &'a [f64],
    /// Right reference/stimulus channel in `f64`.
    pub in_r: &'a [f64],
    /// Configured composite perceptual loss evaluator.
    pub loss_eval: &'a CompositeLossEvaluator,
    /// Active composite loss scalar weights.
    pub weights: CompositeLossWeights,
}

impl core::fmt::Debug for StageTargetEvalContext<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StageTargetEvalContext")
            .field("tier", &self.tier)
            .field("in_len", &self.in_l.len().min(self.in_r.len()))
            .field("weights", &self.weights)
            .finish_non_exhaustive()
    }
}

/// Borrowed input bundle for [`solve_point_stage_targets_adjoint`].
#[derive(Clone, Copy)]
pub struct StageTargetSolveInput<'a> {
    /// Compiled `SONA` CPU/GPU forward and adjoint runner.
    pub runner: &'a TrainRunner,
    /// Unmodulated baseline continuous `f64` model.
    pub base: &'a SonaModelF64,
    /// Left reference/stimulus channel in `f64`.
    pub in_l: &'a [f64],
    /// Right reference/stimulus channel in `f64`.
    pub in_r: &'a [f64],
    /// Left wet target channel in `f64`.
    pub target_l: &'a [f64],
    /// Right wet target channel in `f64`.
    pub target_r: &'a [f64],
    /// Initial `32`-slot `stage_targets` seed (`slot 31` always forced `0.0`).
    pub init_targets: &'a [f64; STAGE_TARGET_SLOTS],
    /// Stage latent conditioning vector `z_stage[0..8]`.
    pub z_stage: &'a [f64; 8],
    /// Safety and dynamic stability bounds across all `32` slots.
    pub bounds: &'a StageTargetBounds,
}

impl core::fmt::Debug for StageTargetSolveInput<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StageTargetSolveInput")
            .field("in_len", &self.in_l.len().min(self.in_r.len()))
            .field("target_len", &self.target_l.len().min(self.target_r.len()))
            .field("z_stage", &self.z_stage)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct VmObjectiveCtx<'a> {
    loss_eval: &'a CompositeLossEvaluator,
    weights: CompositeLossWeights,
    bounds: &'a StageTargetBounds,
    barrier_weight: f64,
    l2_reg: f64,
    in_l32: &'a [f32],
    in_r32: &'a [f32],
}

/// Evaluates [`CompositeLossEvaluator`] and (optionally) exact reverse-mode gradients wrt
/// `stage_targets[0..32]` and `SonaStageLatentModsF64` via [`eval_model_pass2_audio`] +
/// [`assemble_pass2_gradients`].
pub fn eval_stage_targets_loss_and_grad(
    ctx: &StageTargetEvalContext<'_>,
    stage_targets: &[f64; STAGE_TARGET_SLOTS],
    z_stage: &[f64; 8],
    compute_grad: bool,
) -> Result<StageTargetEvalResult, TrainError> {
    let n = ctx.in_l.len().min(ctx.in_r.len());
    let q_count = ctx.tier.simd_groups();
    let hoisted = hoist_stage_targets_onto_model(ctx.base, stage_targets, z_stage, ctx.tier);
    let buf = eval_model_pass2_audio(
        ctx.runner,
        &hoisted.model,
        &ctx.in_l[..n],
        &ctx.in_r[..n],
        q_count,
    )?;

    let (breakdown, grads) = if compute_grad {
        let mut gl = vec![0.0_f64; n];
        let mut gr = vec![0.0_f64; n];
        let bd = ctx.loss_eval.eval_and_adjoint_with_weights(
            &buf.pred_l,
            &buf.pred_r,
            ctx.weights,
            Some(&mut gl),
            Some(&mut gr),
        )?;
        let active_k = hoisted
            .model
            .active_kautz_pairs
            .min(hoisted.model.kautz.len());
        let p2_grads = assemble_pass2_gradients(
            ctx.runner,
            &Pass2AdjInput {
                model: &hoisted.model,
                in_l: &ctx.in_l[..n],
                in_r: &ctx.in_r[..n],
                buf: &buf,
                gl: &gl,
                gr: &gr,
                q_opt: q_count,
                k_opt: active_k,
            },
        )?;
        let st_grads =
            pullback_stage_targets_and_latent_grads(ctx.base, &hoisted, z_stage, &p2_grads);
        (bd, Some(st_grads))
    } else {
        let bd = ctx.loss_eval.eval_and_adjoint_with_weights(
            &buf.pred_l,
            &buf.pred_r,
            ctx.weights,
            None,
            None,
        )?;
        (bd, None)
    };

    Ok(StageTargetEvalResult { breakdown, grads })
}

fn eval_vm_objective_with_reuse(
    vm: &mut dnam_vm::SonaVm<'_, DefaultMathBackend>,
    ctx: &VmObjectiveCtx<'_>,
    st: &[f64; STAGE_TARGET_SLOTS],
    z32: &[f32; 8],
) -> Result<f64, TrainError> {
    let n = ctx.in_l32.len().min(ctx.in_r32.len());
    let st32: [f32; STAGE_TARGET_SLOTS] = core::array::from_fn(|i| {
        if i + 1 < STAGE_TARGET_SLOTS && st[i].is_finite() {
            st[i] as f32
        } else {
            0.0
        }
    });
    vm.reset_with_latched_stage_targets(&st32, z32);
    let mut out_l32 = vec![0.0_f32; n];
    let mut out_r32 = vec![0.0_f32; n];
    vm.process_block(
        &ctx.in_l32[..n],
        &ctx.in_r32[..n],
        &mut out_l32,
        &mut out_r32,
    )?;
    let pred_l: Vec<f64> = out_l32.into_iter().map(f64::from).collect();
    let pred_r: Vec<f64> = out_r32.into_iter().map(f64::from).collect();
    let bd =
        ctx.loss_eval
            .eval_and_adjoint_with_weights(&pred_l, &pred_r, ctx.weights, None, None)?;
    let (bar, _) = eval_corner_barrier_loss_and_grad(st, ctx.bounds, ctx.barrier_weight, None);
    let l2: f64 = st[..STAGE_TARGET_SLOTS - 1]
        .iter()
        .map(|&v| v * v)
        .sum::<f64>();
    Ok(bd.total + bar + 0.5 * ctx.l2_reg * l2)
}

/// Optimizes bounded `stage_targets[0..31]` against `(target_l, target_r)` via reverse-mode
/// `Pass2` adjoints + `CompositeLossEvaluator` + `SsBroydenState` + optional zero-alloc persistent
/// `SonaVm` coordinate micro-polish (`ref[train.pipeline.control-projection-fit]`).
pub fn solve_point_stage_targets_adjoint(
    input: &StageTargetSolveInput<'_>,
    config: &StageTargetSolveConfig,
    vm_polish_opt: Option<&mut dnam_vm::SonaVm<'_, DefaultMathBackend>>,
) -> Result<StageTargetSolveReport, TrainError> {
    let StageTargetSolveInput {
        runner,
        base,
        in_l,
        in_r,
        target_l,
        target_r,
        init_targets,
        z_stage,
        bounds,
    } = *input;
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    if n <= config.warmup_frames {
        return Err(TrainError::InvalidDimensions {
            context: "solve_point_stage_targets_adjoint",
            expected: config.warmup_frames + 1,
            actual: n,
        });
    }

    let loss_cfg = CompositeLossConfig {
        sample_rate_hz: base.sample_rate_hz,
        warmup_frames: config.warmup_frames,
        w_besr: config.loss_weights.w_besr,
        w_mr_iir: config.loss_weights.w_mr_iir,
        w_spec_ot: config.loss_weights.w_spec_ot,
        w_gabor: config.loss_weights.w_gabor,
        w_if_gd: config.loss_weights.w_if_gd,
        w_ltas: config.loss_weights.w_ltas,
        w_seg_esr: config.loss_weights.w_seg_esr,
        w_nyquist: config.loss_weights.w_nyquist,
        w_mse: config.loss_weights.w_mse,
        ..CompositeLossConfig::default()
    };
    let loss_eval = CompositeLossEvaluator::new(&target_l[..n], &target_r[..n], loss_cfg)?;

    let mut opt_slots = Vec::new();
    let mut param_bounds = Vec::new();
    let mut init_vec = Vec::new();
    let mut coord_scales = Vec::new();
    let mut seen = [false; STAGE_TARGET_SLOTS];
    for &slot in &config.active_slots {
        if slot + 1 >= STAGE_TARGET_SLOTS || seen[slot] {
            continue;
        }
        seen[slot] = true;
        let lo = bounds.min[slot];
        let hi = bounds.max[slot];
        if !lo.is_finite() || !hi.is_finite() || hi - lo <= 1.0e-8 {
            continue;
        }
        let margin = 1.0e-6 * (hi - lo);
        let init_v = if init_targets[slot].is_finite() {
            init_targets[slot].clamp(lo + margin, hi - margin)
        } else {
            (0.5 * (lo + hi)).clamp(lo + margin, hi - margin)
        };
        opt_slots.push(slot);
        param_bounds.push(ParamBoundSpec::Box { lo, hi });
        init_vec.push(init_v);
        coord_scales.push((0.5 * (hi - lo)).clamp(0.1, 16.0));
    }

    let mut base_targets = *init_targets;
    base_targets[STAGE_TARGET_SLOTS - 1] = 0.0;
    for (i, dst) in base_targets[..STAGE_TARGET_SLOTS - 1]
        .iter_mut()
        .enumerate()
    {
        let lo = bounds.min[i];
        let hi = bounds.max[i];
        *dst = if dst.is_finite() && hi >= lo {
            dst.clamp(lo, hi)
        } else if hi >= lo && lo.is_finite() && hi.is_finite() {
            0.0_f64.clamp(lo, hi)
        } else {
            0.0
        };
    }

    let eval_ctx = StageTargetEvalContext {
        runner,
        base,
        tier: config.tier,
        in_l: &in_l[..n],
        in_r: &in_r[..n],
        loss_eval: &loss_eval,
        weights: config.loss_weights,
    };

    if opt_slots.is_empty() {
        let res = eval_stage_targets_loss_and_grad(&eval_ctx, &base_targets, z_stage, false)?;
        let (bar, _) =
            eval_corner_barrier_loss_and_grad(&base_targets, bounds, config.barrier_weight, None);
        let loss = res.breakdown.total + bar;
        return Ok(StageTargetSolveReport {
            stage_targets: base_targets,
            initial_loss: loss,
            final_loss: loss,
            iters_run: 0,
            evals: 1,
            micro_polish_accepted: false,
        });
    }

    let mut init_total_loss: Option<f64> = None;
    let l2_reg = config.l2_reg.max(0.0);
    let mut g_bar_buf = [0.0_f64; STAGE_TARGET_SLOTS];

    let OptResult {
        best_params,
        best_loss,
        iters_run,
        mut evals,
    } = SsBroydenState::minimize_scaled(
        &init_vec,
        &param_bounds,
        Some(&coord_scales),
        &config.opt,
        |x, gx_opt| {
            let mut cur = base_targets;
            for (&slot, &xv) in opt_slots.iter().zip(x) {
                cur[slot] = xv;
            }
            cur[STAGE_TARGET_SLOTS - 1] = 0.0;

            let need_grad = gx_opt.is_some();
            let ev = eval_stage_targets_loss_and_grad(&eval_ctx, &cur, z_stage, need_grad)?;
            let (bar_loss, _) = eval_corner_barrier_loss_and_grad(
                &cur,
                bounds,
                config.barrier_weight,
                need_grad.then_some(&mut g_bar_buf),
            );
            let l2_sum: f64 = cur[..STAGE_TARGET_SLOTS - 1].iter().map(|&v| v * v).sum();
            let tot = ev.breakdown.total + bar_loss + 0.5 * l2_reg * l2_sum;
            if init_total_loss.is_none() {
                init_total_loss = Some(tot);
            }
            if let (Some(gx), Some(g_st)) = (gx_opt, ev.grads.as_ref()) {
                for (dst, &slot) in gx.iter_mut().zip(&opt_slots) {
                    *dst = g_st.d_stage_targets[slot] + g_bar_buf[slot] + l2_reg * cur[slot];
                }
            }
            Ok(tot)
        },
    )?;

    let mut solved = base_targets;
    for (&slot, &xv) in opt_slots.iter().zip(&best_params) {
        solved[slot] = xv.clamp(bounds.min[slot], bounds.max[slot]);
    }
    solved[STAGE_TARGET_SLOTS - 1] = 0.0;

    let mut final_loss = best_loss;
    let mut micro_polish_accepted = false;
    if config.vm_micro_polish_iters > 0
        && let Some(vm) = vm_polish_opt
    {
        let in_l32: Vec<f32> = in_l[..n].iter().map(|&v| v as f32).collect();
        let in_r32: Vec<f32> = in_r[..n].iter().map(|&v| v as f32).collect();
        let z32: [f32; 8] = core::array::from_fn(|k| {
            if z_stage[k].is_finite() {
                z_stage[k] as f32
            } else {
                0.0
            }
        });
        let vm_ctx = VmObjectiveCtx {
            loss_eval: &loss_eval,
            weights: config.loss_weights,
            bounds,
            barrier_weight: config.barrier_weight,
            l2_reg,
            in_l32: &in_l32,
            in_r32: &in_r32,
        };
        let mut best_vm_loss = eval_vm_objective_with_reuse(vm, &vm_ctx, &solved, &z32)?;
        evals += 1;
        let eps = config.vm_micro_polish_eps.abs().clamp(1.0e-4, 0.25);
        for sweep in 0..config.vm_micro_polish_iters {
            let step = eps * 0.5_f64.powi(sweep as i32);
            let mut improved = false;
            for &slot in &opt_slots {
                let cur_v = solved[slot];
                let (lo, hi) = (bounds.min[slot], bounds.max[slot]);
                for cand_v in [(cur_v + step).clamp(lo, hi), (cur_v - step).clamp(lo, hi)] {
                    if (cand_v - cur_v).abs() <= 1.0e-12 {
                        continue;
                    }
                    let mut cand = solved;
                    cand[slot] = cand_v;
                    let cand_loss = eval_vm_objective_with_reuse(vm, &vm_ctx, &cand, &z32)?;
                    evals += 1;
                    if cand_loss.is_finite() && cand_loss < best_vm_loss {
                        best_vm_loss = cand_loss;
                        solved = cand;
                        micro_polish_accepted = true;
                        improved = true;
                        break;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        final_loss = best_vm_loss;
    }

    Ok(StageTargetSolveReport {
        stage_targets: solved,
        initial_loss: init_total_loss.unwrap_or(best_loss),
        final_loss,
        iters_run,
        evals,
        micro_polish_accepted,
    })
}

/// Renders stereo audio through a single persistent `SonaVm` across a continuous `3 kHz`
/// (`16`-sample) trajectory without `.dnam-bin` re-parsing (`ref[dsp.ctrl.dispatch.micro-ramp]`).
pub fn eval_control_trajectory_3khz_vm(
    vm: &mut dnam_vm::SonaVm<'_, DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    trajectory_3khz: &[[f32; STAGE_TARGET_SLOTS]],
    z_stage_3khz: &[[f32; 8]],
) -> Result<(Vec<f64>, Vec<f64>), TrainError> {
    let n = in_l.len().min(in_r.len());
    let mut out_l = vec![0.0_f64; n];
    let mut out_r = vec![0.0_f64; n];
    if n == 0 {
        return Ok((out_l, out_r));
    }
    let zero_t = [0.0_f32; STAGE_TARGET_SLOTS];
    let zero_z = [0.0_f32; 8];
    let t0 = trajectory_3khz.first().unwrap_or(&zero_t);
    let z0 = z_stage_3khz.first().unwrap_or(&zero_z);
    vm.reset_with_latched_stage_targets(t0, z0);

    let mut off = 0_usize;
    let mut tick = 0_usize;
    let mut blk_l = [0.0_f32; 16];
    let mut blk_r = [0.0_f32; 16];
    while off < n {
        let chunk = (n - off).min(16);
        if tick > 0 {
            let st = trajectory_3khz
                .get(tick)
                .or_else(|| trajectory_3khz.last())
                .unwrap_or(&zero_t);
            let zs = z_stage_3khz
                .get(tick)
                .or_else(|| z_stage_3khz.last())
                .unwrap_or(&zero_z);
            vm.set_latched_stage_targets_tick(st, zs);
        }
        vm.process_block(
            &in_l[off..off + chunk],
            &in_r[off..off + chunk],
            &mut blk_l[..chunk],
            &mut blk_r[..chunk],
        )?;
        for i in 0..chunk {
            out_l[off + i] = f64::from(blk_l[i]);
            out_r[off + i] = f64::from(blk_r[i]);
        }
        off += chunk;
        tick += 1;
    }
    Ok((out_l, out_r))
}
