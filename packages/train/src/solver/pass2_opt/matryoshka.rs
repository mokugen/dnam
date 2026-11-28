//! Progressive `4 -> 6 -> 8 -> 10` core-quad expansion and Joint 4-Tier Matryoshka gradient
//! assembly with log-scaled tier losses and Arena-Scoped Deterministic `PCGrad`
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use dnam_vm::SONA_INTERBLOCK_TILES;

use crate::error::TrainError;
use crate::linalg::adamw::apply_arena_scoped_pcgrad;
use crate::linalg::coords::ScaledCoordSpace;
use crate::losses::{CompositeLossEvaluator, CompositeLossWeights};
use crate::runner::TrainRunner;

use super::super::model::{SonaModelF64, StereoTrainingPairs};
use super::eval::core_adj::active_tiles_for_quads;
use super::eval::{Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio};
use super::theta::{TIER_FIR_TAPS, TIER_KAUTZ, TIER_QUADS, ThetaLayout};
use super::varpro_sync::materialize_tier_view;

/// Canonical Matryoshka log-loss weights across `[Lite, Full, Pro, Ultra]`.
pub(super) const MATRYOSHKA_TIER_WEIGHTS: [f64; 4] = [0.15, 0.20, 0.25, 0.40];
/// Canonical additive floor inside `ln(L_t + 1e-8)` preventing log-zero singularities.
pub(super) const MATRYOSHKA_LOG_EPS: f64 = 1.0e-8;

const _: () = assert!(
    MATRYOSHKA_TIER_WEIGHTS[0] > 0.0
        && MATRYOSHKA_TIER_WEIGHTS[1] > 0.0
        && MATRYOSHKA_TIER_WEIGHTS[2] > 0.0
        && MATRYOSHKA_TIER_WEIGHTS[3] > 0.0
        && (MATRYOSHKA_TIER_WEIGHTS[0]
            + MATRYOSHKA_TIER_WEIGHTS[1]
            + MATRYOSHKA_TIER_WEIGHTS[2]
            + MATRYOSHKA_TIER_WEIGHTS[3])
            > 0.999_999
        && (MATRYOSHKA_TIER_WEIGHTS[0]
            + MATRYOSHKA_TIER_WEIGHTS[1]
            + MATRYOSHKA_TIER_WEIGHTS[2]
            + MATRYOSHKA_TIER_WEIGHTS[3])
            < 1.000_001
);
const _: () = assert!(MATRYOSHKA_LOG_EPS > 0.0 && MATRYOSHKA_LOG_EPS < 1.0e-6);

/// Normalizes [`MATRYOSHKA_TIER_WEIGHTS`] over active tiers `0..=max_tier` (`0.0` above).
#[must_use]
pub(super) fn normalized_tier_weights(max_tier: usize) -> [f64; 4] {
    let mt = max_tier.min(3);
    let mut w = [0.0_f64; 4];
    let sum: f64 = MATRYOSHKA_TIER_WEIGHTS[..=mt].iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for t in 0..=mt {
        w[t] = MATRYOSHKA_TIER_WEIGHTS[t] * inv;
    }
    w
}

/// Progressively expands `model` from `prev_tier` to `next_tier` (`4 -> 6 -> 8 -> 10` quads):
/// - Zero-initializes every newly unlocked `InterBlockTile` targeting an existing lower-tier quad
///   (`SONA_INTERBLOCK_TILES[t].1 < TIER_QUADS[prev_tier]`, keeping lower-tier features invariant).
/// - Zeroes newly unlocked readout rows `w_ro[(prev_q * 4)..(next_q * 4)]` and seeds
///   `tier_overrides[next_tier]` from `tier_overrides[prev_tier]`.
pub(super) fn expand_model_to_tier(model: &mut SonaModelF64, prev_tier: usize, next_tier: usize) {
    let pt = prev_tier.min(3);
    let nt = next_tier.min(3);
    if nt <= pt {
        return;
    }
    let prev_q = TIER_QUADS[pt];
    let next_q = TIER_QUADS[nt];
    let prev_tiles = active_tiles_for_quads(prev_q);
    let next_tiles = active_tiles_for_quads(next_q);

    for (t_idx, tile) in model
        .tiles
        .iter_mut()
        .enumerate()
        .take(next_tiles)
        .skip(prev_tiles)
    {
        if SONA_INTERBLOCK_TILES[t_idx].1 < prev_q {
            *tile = Default::default();
        }
    }

    for row in (prev_q * 4)..(next_q * 4) {
        model.w_ro[row] = [0.0; 12];
    }

    let has_extra_modal =
        model.active_fir_taps > TIER_FIR_TAPS[pt] || model.active_kautz_pairs > TIER_KAUTZ[pt];
    if !has_extra_modal
        && model.tier_overrides[nt].is_none()
        && let Some(ref prev_ovr) = model.tier_overrides[pt]
    {
        let mut next_ovr = prev_ovr.clone();
        for row in (prev_q * 4)..(next_q * 4) {
            next_ovr.w_ro[row] = [0.0; 12];
        }
        let prev_f = TIER_FIR_TAPS[pt].min(next_ovr.fir_l.len());
        let next_f = TIER_FIR_TAPS[nt].min(next_ovr.fir_l.len());
        next_ovr.fir_l[prev_f..next_f].copy_from_slice(&model.fir_l[prev_f..next_f]);
        next_ovr.fir_r[prev_f..next_f].copy_from_slice(&model.fir_r[prev_f..next_f]);
        let prev_k = TIER_KAUTZ[pt].min(next_ovr.kautz_c_re_l.len());
        let next_k = TIER_KAUTZ[nt].min(next_ovr.kautz_c_re_l.len());
        for k in prev_k..next_k {
            next_ovr.kautz_b_gamma[k] = model.kautz[k].b_gamma;
            next_ovr.kautz_c_re_l[k] = model.kautz[k].c_re_l;
            next_ovr.kautz_c_im_l[k] = model.kautz[k].c_im_l;
            next_ovr.kautz_c_re_r[k] = model.kautz[k].c_re_r;
            next_ovr.kautz_c_im_r[k] = model.kautz[k].c_im_r;
        }
        model.tier_overrides[nt] = Some(next_ovr);
    }
}

/// Shared evaluation parameters for [`eval_joint_matryoshka_and_grad`].
pub(super) struct MatryoshkaEvalCtx<'a> {
    pub runner: &'a TrainRunner,
    pub pairs: StereoTrainingPairs<'a>,
    pub loss_eval: &'a CompositeLossEvaluator,
    pub weights: CompositeLossWeights,
    pub layout: &'a ThetaLayout,
    pub space_jac: Option<(&'a ScaledCoordSpace<'a>, &'a [f64])>,
    pub max_tier: usize,
    pub use_pcgrad: bool,
}

/// Output of [`eval_joint_matryoshka_and_grad`].
#[derive(Clone, Debug)]
pub(super) struct MatryoshkaEvalOutcome {
    pub joint_loss: f64,
    pub tier_losses: [f64; 4],
    pub tier_besr_db: [f64; 4],
    pub grad_out: Vec<f64>,
}

/// Evaluates nested Matryoshka objective $\sum_{t \le T_{\max}} w_t \ln(L_t + 10^{-8})$ over active
/// tiers `0..=max_tier` and optionally assembles exact (`use_pcgrad = false`) or Arena-Scoped
/// `PCGrad` (`use_pcgrad = true`) parameter gradients.
pub(super) fn eval_joint_matryoshka_and_grad(
    ctx: &MatryoshkaEvalCtx<'_>,
    model: &SonaModelF64,
    need_grad: bool,
) -> Result<MatryoshkaEvalOutcome, TrainError> {
    let mt = ctx.max_tier.min(3);
    let tier_weights = normalized_tier_weights(mt);
    let n = ctx.pairs.aligned_len();
    let dim = ctx.layout.dim();

    let mut joint_loss = 0.0_f64;
    let mut tier_losses = [0.0_f64; 4];
    let mut tier_besr_db = [0.0_f64; 4];
    let mut tier_grads: [Vec<f64>; 4] = Default::default();
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

    for (ti, &wt) in tier_weights.iter().enumerate().take(mt + 1) {
        if wt <= 0.0 {
            continue;
        }
        let view = materialize_tier_view(model, ti);
        let q_t = TIER_QUADS[ti];
        let k_t = view.active_kautz_pairs.min(TIER_KAUTZ[ti]);
        let buf = eval_model_pass2_audio(
            ctx.runner,
            &view,
            &ctx.pairs.in_l[..n],
            &ctx.pairs.in_r[..n],
            q_t,
        )?;
        let bd = if need_grad {
            gl.fill(0.0);
            gr.fill(0.0);
            ctx.loss_eval.eval_and_adjoint_with_weights(
                &buf.pred_l,
                &buf.pred_r,
                ctx.weights,
                Some(&mut gl),
                Some(&mut gr),
            )?
        } else {
            ctx.loss_eval.eval_and_adjoint_with_weights(
                &buf.pred_l,
                &buf.pred_r,
                ctx.weights,
                None,
                None,
            )?
        };
        let lt = if bd.total.is_finite() {
            bd.total.max(0.0)
        } else {
            1.0e12
        };
        tier_losses[ti] = lt;
        tier_besr_db[ti] = 10.0 * bd.besr.max(1.0e-12).log10();
        joint_loss += wt * (lt + MATRYOSHKA_LOG_EPS).ln();

        if need_grad {
            let log_scale = 1.0 / (lt + MATRYOSHKA_LOG_EPS);
            for v in &mut gl {
                *v *= log_scale;
            }
            for v in &mut gr {
                *v *= log_scale;
            }
            let grads = assemble_pass2_gradients(
                ctx.runner,
                &Pass2AdjInput {
                    model: &view,
                    in_l: &ctx.pairs.in_l[..n],
                    in_r: &ctx.pairs.in_r[..n],
                    buf: &buf,
                    gl: &gl,
                    gr: &gr,
                    q_opt: q_t,
                    k_opt: k_t,
                },
            )?;
            let mut gx = vec![0.0_f64; dim];
            ctx.layout.write_grad(&view, &grads, &mut gx);
            tier_grads[ti] = if ctx.use_pcgrad
                && let Some((space, jac)) = ctx.space_jac
            {
                let mut gu = vec![0.0_f64; dim];
                space.pullback_grad(&gx, jac, &mut gu);
                gu
            } else {
                gx
            };
        }
    }

    let mut grad_out = Vec::new();
    if need_grad {
        if ctx.use_pcgrad {
            grad_out =
                apply_arena_scoped_pcgrad(&tier_grads, tier_weights, &ctx.layout.arena_of_param);
        } else {
            grad_out = vec![0.0_f64; dim];
            for (ti, g_vec) in tier_grads.iter().enumerate().take(mt + 1) {
                let wt = tier_weights[ti];
                if g_vec.len() == dim && wt > 0.0 {
                    for (dst, &gv) in grad_out.iter_mut().zip(g_vec) {
                        if gv.is_finite() {
                            *dst += wt * gv;
                        }
                    }
                }
            }
        }
    }

    Ok(MatryoshkaEvalOutcome {
        joint_loss,
        tier_losses,
        tier_besr_db,
        grad_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::{CompositeLossConfig, canonical_ltas_33_centers_hz};

    #[test]
    fn test_expand_zero_init_preserves_lite_exactness_and_matryoshka_fd_parity() {
        let runner = TrainRunner::cpu();
        let n = 512_usize;
        let centers = canonical_ltas_33_centers_hz();
        let make_sig = |phase: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let sec = (t as f64) / 48_000.0;
                    centers
                        .iter()
                        .enumerate()
                        .map(|(b, &fc)| {
                            0.02 * (core::f64::consts::TAU * fc * sec + phase + 0.3 * (b as f64))
                                .sin()
                        })
                        .sum()
                })
                .collect()
        };
        let in_l = make_sig(0.0);
        let in_r = make_sig(0.6);
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.7 * v + 0.1 * v * v * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.6 * v - 0.1 * v * v * v).collect();

        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].tone_bp = [0.6, -0.4, 0.3, 0.2];
        model.core_quads[0].adaa_drive = [1.4, 1.0, 0.8, 0.6];
        model.w_ro[0][6] = 0.25;
        model.w_ro[0][7] = -0.20;
        // Pre-fill Full tiles with nonzero probes so expand_model_to_tier proves selective zeroing.
        for t in 3..13 {
            model.tiles[t].w_odd_4x4[0][0] = 0.77;
        }
        let lite_before = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, 4).unwrap();
        expand_model_to_tier(&mut model, 0, 1);
        let lite_after = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, 4).unwrap();
        assert_eq!(lite_before.pred_l, lite_after.pred_l);
        assert_eq!(lite_before.pred_r, lite_after.pred_r);
        let zeroed_dst_lt4 = (3..13)
            .filter(|&t| SONA_INTERBLOCK_TILES[t].1 < 4)
            .inspect(|&t| assert_eq!(model.tiles[t].w_odd_4x4[0][0], 0.0))
            .count();
        assert_eq!(zeroed_dst_lt4, 8);
        assert_eq!(model.tiles[3].w_odd_4x4[0][0], 0.77);
        assert_eq!(model.tiles[5].w_odd_4x4[0][0], 0.77);

        // Central FD parity on 2-tier Matryoshka joint log-loss (without PCGrad).
        model.core_quads[4].tone_dir = [0.4; 4];
        model.w_ro[16][6] = 0.15;
        model.w_ro[16][7] = -0.15;
        let loss_eval = CompositeLossEvaluator::new_for_homotopy(
            &tgt_l,
            &tgt_r,
            None,
            None,
            CompositeLossConfig {
                sample_rate_hz: 48_000.0,
                warmup_frames: 32,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let layout = ThetaLayout::for_tier_from_model(&model, 1, false, false, 128.0);
        let (theta0, bounds) = layout.pack(&model);
        let ctx = MatryoshkaEvalCtx {
            runner: &runner,
            pairs: StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            loss_eval: &loss_eval,
            weights: CompositeLossWeights::homotopy_at(0.5),
            layout: &layout,
            space_jac: None,
            max_tier: 1,
            use_pcgrad: false,
        };
        let res = eval_joint_matryoshka_and_grad(&ctx, &model, true).unwrap();

        // Pure-f64-path Core[4].tone_dir[0] and Core[0].in_mid_weight[0] FD checks.
        let mut work = model.clone();
        let mut eval_at = |idx: usize, delta: f64| -> f64 {
            let mut t = theta0.clone();
            t[idx] += delta;
            layout.unpack(&t, &mut work);
            eval_joint_matryoshka_and_grad(&ctx, &work, false)
                .unwrap()
                .joint_loss
        };
        let eps = 2.0e-3_f64;
        for idx in [15_usize, 18] {
            let fd = (eval_at(idx, eps) - eval_at(idx, -eps)) / (2.0 * eps);
            let ana = res.grad_out[idx];
            let scale = ana.abs().max(fd.abs()).max(1.0e-3);
            assert!(
                (ana - fd).abs() <= 5.0e-2 * scale,
                "idx={idx} Matryoshka FD mismatch: ana={ana} fd={fd}"
            );
        }

        // With PCGrad, gradient in u-space stays finite and non-empty.
        let scales = layout.coord_scales(n);
        let (space, u0) =
            ScaledCoordSpace::new("matryoshka_test", &theta0, &bounds, Some(&scales)).unwrap();
        let mut x0 = vec![0.0; layout.dim()];
        let mut jac0 = vec![0.0; layout.dim()];
        space.decode(&u0, &mut x0, &mut jac0);
        let ctx_pc = MatryoshkaEvalCtx {
            space_jac: Some((&space, &jac0)),
            use_pcgrad: true,
            ..ctx
        };
        let res_pc = eval_joint_matryoshka_and_grad(&ctx_pc, &model, true).unwrap();
        assert_eq!(res_pc.grad_out.len(), layout.dim());
        assert!(res_pc.grad_out.iter().all(|v| v.is_finite()));
    }
}
