//! Conditioned `u`-space Straight-Through Estimator (`STE`) quantization-aware `AdamW` polish
//! evaluating loss and adjoints at exact wire-quantized `binary16` / `binary32` parameters while
//! updating scaled `u`-space shadow weights under strict monotone non-regression
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use crate::error::TrainError;
use crate::linalg::adamw::{AdamWConfig, AdamWState, cosine_lr_with_warmup};
use crate::linalg::coords::ScaledCoordSpace;
use crate::losses::{CompositeLossEvaluator, CompositeLossWeights};
use crate::runner::TrainRunner;

use super::super::model::{
    SonaModelF64, StereoTrainingPairs, snap_model_fp16, snap_non_readout_fp16,
};
use super::Pass2OptConfig;
use super::eval::{Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio};
use super::theta::ThetaLayout;

pub(super) struct QatPolishOutcome {
    pub steps_run: usize,
    pub baseline_loss: f64,
    pub best_loss: f64,
    pub best_besr_db: f64,
}

/// Refines `model` from the Pass-2 continuous optimum using `config.qat_polish_steps`
/// line-search-free `STE` `AdamW` updates in conditioned `u`-space over the wire-quantized
/// forward/adjoint graph under `weights`.
///
/// Always returns a wire-snapped model with `best_loss <= baseline_loss` (or leaves `model`
/// untouched when `config.qat_polish_steps == 0`).
pub(super) fn run_pass2_ste_qat_polish(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    loss_eval: &CompositeLossEvaluator,
    weights: CompositeLossWeights,
    layout: &ThetaLayout,
    config: Pass2OptConfig,
) -> Result<QatPolishOutcome, TrainError> {
    let steps = config.qat_polish_steps;
    let n = pairs.aligned_len();
    let (in_l, in_r) = (&pairs.in_l[..n], &pairs.in_r[..n]);
    let q_count = layout.q_opt.clamp(1, 10);

    let mut baseline_model = model.clone();
    snap_model_fp16(&mut baseline_model);
    let base_buf = eval_model_pass2_audio(runner, &baseline_model, in_l, in_r, q_count)?;
    let mut gl = vec![0.0_f64; n];
    let mut gr = vec![0.0_f64; n];
    let base_bd = if steps > 0 {
        loss_eval.eval_and_adjoint_with_weights(
            &base_buf.pred_l,
            &base_buf.pred_r,
            weights,
            Some(&mut gl),
            Some(&mut gr),
        )?
    } else {
        loss_eval.eval_and_adjoint_with_weights(
            &base_buf.pred_l,
            &base_buf.pred_r,
            weights,
            None,
            None,
        )?
    };
    let baseline_loss = base_bd.total;
    let mut best_loss = baseline_loss;
    let mut best_besr_db = 10.0 * base_bd.besr.max(1.0e-12).log10();

    if steps == 0 || !layout.is_fp16_wire.iter().any(|&is_fp16| is_fp16) || layout.dim() == 0 {
        return Ok(QatPolishOutcome {
            steps_run: 0,
            baseline_loss,
            best_loss,
            best_besr_db,
        });
    }

    let (init_shadow, bounds) = layout.pack(model);
    let dim = layout.dim();
    let scales = layout.coord_scales(n);
    let (space, mut u_shadow) = ScaledCoordSpace::new(
        "run_pass2_ste_qat_polish",
        &init_shadow,
        &bounds,
        Some(&scales),
    )?;
    let mut x_shadow = vec![0.0_f64; dim];
    let mut jac_shadow = vec![0.0_f64; dim];
    space.decode(&u_shadow, &mut x_shadow, &mut jac_shadow);

    let mut best_model = baseline_model.clone();
    let mut work_model = baseline_model;
    let mut current_buf = base_buf;
    let mut current_model = work_model.clone();
    let mut gx = vec![0.0_f64; dim];
    let mut gu = vec![0.0_f64; dim];
    let mut adamw = AdamWState::new(dim);
    let adamw_cfg = AdamWConfig {
        lr_max: 1.0e-3,
        lr_min: 5.0e-5,
        warmup_steps: 0,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
        grad_clip_norm: 10.0,
    };

    for step in 0..steps {
        let grads = assemble_pass2_gradients(
            runner,
            &Pass2AdjInput {
                model: &current_model,
                in_l,
                in_r,
                buf: &current_buf,
                gl: &gl,
                gr: &gr,
                q_opt: layout.q_opt,
                k_opt: layout.k_opt,
            },
        )?;
        gx.fill(0.0);
        layout.write_grad(&grads, &mut gx);
        space.pullback_grad(&gx, &jac_shadow, &mut gu);

        let lr = cosine_lr_with_warmup(step, steps, &adamw_cfg);
        adamw.step_u(&mut u_shadow, &mut gu, lr, &adamw_cfg, Some(space.scales()));
        space.decode(&u_shadow, &mut x_shadow, &mut jac_shadow);

        work_model.clone_from(model);
        layout.unpack(&x_shadow, &mut work_model);
        snap_model_fp16(&mut work_model);
        let next_buf = eval_model_pass2_audio(runner, &work_model, in_l, in_r, q_count)?;

        let need_grad = step + 1 < steps;
        let bd = if need_grad {
            gl.fill(0.0);
            gr.fill(0.0);
            loss_eval.eval_and_adjoint_with_weights(
                &next_buf.pred_l,
                &next_buf.pred_r,
                weights,
                Some(&mut gl),
                Some(&mut gr),
            )?
        } else {
            loss_eval.eval_and_adjoint_with_weights(
                &next_buf.pred_l,
                &next_buf.pred_r,
                weights,
                None,
                None,
            )?
        };
        if bd.total.is_finite() && bd.total < best_loss {
            best_loss = bd.total;
            best_besr_db = 10.0 * bd.besr.max(1.0e-12).log10();
            best_model = work_model.clone();
        }
        current_buf = next_buf;
        current_model = work_model.clone();
    }

    *model = best_model;
    snap_non_readout_fp16(model);

    Ok(QatPolishOutcome {
        steps_run: steps,
        baseline_loss,
        best_loss,
        best_besr_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::CompositeLossConfig;

    #[test]
    fn test_ste_qat_polish_zero_steps_noop_monotone_non_regression_and_fp16_idempotent() {
        let runner = TrainRunner::cpu();
        let n = 256_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 0.4 * ((t as f64) * 0.09).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| 0.35 * ((t as f64) * 0.11).cos()).collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.8 * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.65 * v).collect();
        let loss_eval = CompositeLossEvaluator::new(
            &tgt_l,
            &tgt_r,
            CompositeLossConfig {
                sample_rate_hz: 48_000.0,
                warmup_frames: 16,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();

        let mut base = SonaModelF64::new(48_000.0);
        base.core_quads[0].svf_k_nl = [0.234_567, 0.0, 0.0, 0.0];
        base.core_quads[0].adaa_drive = [1.234_567, 0.5, 0.25, 0.1];
        base.core_quads[0].tone_dir = [0.5, 0.0, 0.0, 0.0];
        base.w_ro[0][6] = 0.612_345;
        base.direct_s2_u_gain = [0.212_345, 0.212_345];

        let layout = ThetaLayout::for_pass2_joint(&base, 2, 128.0);
        let pairs = StereoTrainingPairs {
            in_l: &in_l,
            in_r: &in_r,
            target_l: &tgt_l,
            target_r: &tgt_r,
        };
        let w = CompositeLossWeights::homotopy_at(1.0);

        let mut m_noop = base.clone();
        let cfg0 = Pass2OptConfig {
            active_quads: 2,
            qat_polish_steps: 0,
            ..Pass2OptConfig::default()
        };
        let out0 =
            run_pass2_ste_qat_polish(&runner, &mut m_noop, &pairs, &loss_eval, w, &layout, cfg0)
                .unwrap();
        assert_eq!(out0.steps_run, 0);
        assert_eq!(m_noop.w_ro[0], base.w_ro[0]);
        assert_eq!(m_noop.core_quads, base.core_quads);

        let mut m_qat = base;
        let cfg4 = Pass2OptConfig {
            active_quads: 2,
            qat_polish_steps: 4,
            ..Pass2OptConfig::default()
        };
        let out4 =
            run_pass2_ste_qat_polish(&runner, &mut m_qat, &pairs, &loss_eval, w, &layout, cfg4)
                .unwrap();
        assert_eq!(out4.steps_run, 4);
        assert!(
            out4.best_loss <= out4.baseline_loss,
            "QAT polish regressed: best={} baseline={}",
            out4.best_loss,
            out4.baseline_loss
        );
        let snapped_copy = {
            let mut c = m_qat.clone();
            snap_model_fp16(&mut c);
            c
        };
        assert_eq!(m_qat.core_quads, snapped_copy.core_quads);
        assert_eq!(m_qat.w_ro, snapped_copy.w_ro);
        assert_eq!(m_qat.direct_s2_u_gain, snapped_copy.direct_s2_u_gain);
        assert_eq!(m_qat.core_quads[1].svf_k_nl, [0.0; 4]);
    }
}
