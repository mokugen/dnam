//! Phase-2A prescreening, short candidate exploration, and pure-linear fast-path detection
//! (`ref[train.pipeline.backbone-trajectory-opt]`, `REQ-1.9`).

use crate::error::TrainError;
use crate::losses::CompositeLossWeights;

use super::super::super::model::SonaModelF64;
use super::super::prescreen::{prescreen_top_k_diverse, seed_pass1_inert_defaults};
use super::windows::{
    HybridCtx, build_representative_window, eval_full_capsule_metrics, explore_candidate_short,
    is_pass2_metrics_acceptable, is_pure_linear_pass1_model,
};

pub(super) fn run_phase2a(
    ctx: &HybridCtx<'_>,
    base_model: &SonaModelF64,
) -> Result<(SonaModelF64, usize, bool), TrainError> {
    let use_joint = ctx.config.use_progressive_matryoshka && ctx.max_tier > 0;
    let weights_final = CompositeLossWeights::homotopy_at(1.0);
    let pure_linear_pass1 = is_pure_linear_pass1_model(base_model);
    let pre_lin_check = if pure_linear_pass1 {
        let mut lin_base = base_model.clone();
        lin_base.oversample_upper_tiers_2x = false;
        let base_m = eval_full_capsule_metrics(ctx, &lin_base, use_joint, weights_final)?;
        Some((lin_base, base_m))
    } else {
        None
    };

    let top_k = if ctx.fast_mode {
        ctx.config.top_k_seeds.clamp(1, 2)
    } else {
        ctx.config.top_k_seeds.clamp(1, 8)
    };
    let mut cands = prescreen_top_k_diverse(
        ctx.runner,
        base_model,
        ctx.pairs,
        ctx.config,
        top_k,
        ctx.noise_pair,
    )?;
    if cands.is_empty() {
        return Ok((base_model.clone(), 0, false));
    }

    let (mut best_cand, best_seed) = if ctx.fast_mode || cands.len() == 1 {
        let win = cands.remove(0);
        (win.model, win.seed_index)
    } else {
        let explore_cap = if pre_lin_check
            .as_ref()
            .is_some_and(|(_, bm)| bm.2.is_finite() && bm.2 < -22.0)
        {
            2
        } else {
            4
        };
        let explore_n = cands.len().min(top_k.min(explore_cap));
        let rep_win = build_representative_window(ctx, ctx.config.chunk_frames.min(2_048))?;
        let init_ti = if use_joint { 0 } else { ctx.max_tier };
        let mut picked = cands[0].model.clone();
        let mut picked_seed = cands[0].seed_index;
        let mut best_score = f64::INFINITY;
        for cand in cands.iter_mut().take(explore_n) {
            let sc = explore_candidate_short(ctx, &mut cand.model, &rep_win, init_ti, 4)?;
            if sc.is_finite() && sc < best_score {
                best_score = sc;
                picked = cand.model.clone();
                picked_seed = cand.seed_index;
            }
        }
        (picked, picked_seed)
    };

    if let Some((lin_base, base_m)) = pre_lin_check {
        let cand_m = eval_full_capsule_metrics(ctx, &best_cand, use_joint, weights_final)?;
        let cand_beats_base_obj = cand_m.0.is_finite()
            && base_m.0.is_finite()
            && if use_joint {
                cand_m.0 < base_m.0 + (0.85_f64).ln()
            } else {
                cand_m.0 < base_m.0 * 0.85
            };
        if base_m.2.is_finite()
            && base_m.2 < -22.0
            && (!cand_m.2.is_finite() || cand_m.2 > base_m.2 - 1.5)
            && !cand_beats_base_obj
        {
            return Ok((lin_base, 0, true));
        }
    }

    let has_pass1_modal = base_model.active_fir_taps > 0 || base_model.active_kautz_pairs > 0;
    if has_pass1_modal {
        let pass1_seeded = seed_pass1_inert_defaults(
            base_model,
            None,
            best_seed,
            (
                ctx.config.num_prescreen_seeds,
                ctx.config.prescreen_seed_offset,
                ctx.config.sample_rate_hz,
            ),
        );
        let p1_m = eval_full_capsule_metrics(ctx, &pass1_seeded, use_joint, weights_final)?;
        let cand_m = eval_full_capsule_metrics(ctx, &best_cand, use_joint, weights_final)?;
        if !is_pass2_metrics_acceptable(p1_m, cand_m, use_joint, false) {
            best_cand = pass1_seeded;
        }
    }

    Ok((best_cand, best_seed, false))
}

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use crate::losses::{CompositeLossConfig, CompositeLossEvaluator};
    use crate::runner::TrainRunner;
    use crate::solver::model::{KautzPairF64, StereoTrainingPairs, TransportQuadF64};
    use crate::solver::pass1_probe::synth_kautz_unit_mode;
    use crate::solver::pass2_opt::{Pass2OptConfig, run_pass2_optimize};

    use super::*;

    #[test]
    fn test_phase2a_selects_linear_only_under_negative_joint_log_loss() {
        let runner = TrainRunner::cpu();
        let n = 1_536_usize;
        let sr = 48_000.0_f64;
        let warmup = 64_usize;
        let mut x_l = vec![0.0_f64; n];
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
        for (i, dst) in x_l.iter_mut().enumerate() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let white = ((rng >> 33) as f64) / ((1_u64 << 31) as f64) - 1.0;
            let t = (i as f64) / sr;
            *dst = 0.25 * white + 0.20 * (TAU * 180.0 * t).sin();
        }
        let x_r = x_l.clone();

        // Linear modal + feedforward Op-T target matching Pass-1 accuracy (< -30 dB, negative log-loss).
        let r = 0.965_f64;
        let th = TAU * 320.0 / sr;
        let bg = (1.0 - r * r).sqrt();
        let mut onset = vec![0.0_f64; n];
        onset[16..].copy_from_slice(&x_l[..(n - 16)]);
        let (mr, _) = synth_kautz_unit_mode(r, th, &onset);
        let mut base_sig = vec![0.0_f64; n];
        for t in 0..n {
            let f5 = if t >= 5 { 0.22 * x_l[t - 5] } else { 0.0 };
            base_sig[t] = 0.45 * x_l[t] + f5 + 2.0 * bg * 0.28 * mr[t];
        }
        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 32;
        model.fir_l[0] = 0.45;
        model.fir_r[0] = 0.45;
        model.fir_l[5] = 0.22;
        model.fir_r[5] = 0.22;
        model.pre_conv_u_gain = [1.0, 1.0];
        model.direct_s2_u_gain = [0.0, 0.0];
        model.active_kautz_pairs = 1;
        model.kautz[0] = KautzPairF64 {
            r_low: r,
            dr: 0.0,
            theta: th,
            b_gamma: bg,
            c_re_l: 0.28,
            c_im_l: 0.0,
            c_re_r: 0.28,
            c_im_r: 0.0,
        };
        model.transport_quads[0] = TransportQuadF64 {
            delay_samples: [64.0, 96.0, 64.0, 96.0],
            input_gains: [1.0, 1.0, 0.0, 0.0],
            wet_tap_l: [0.24, 0.0, 0.0, 0.0],
            wet_tap_r: [0.0, -0.18, 0.0, 0.0],
            allpass_delays: [0; 4],
            ext_f32: true,
            ..TransportQuadF64::default()
        };
        let (wet_l, wet_r, _) =
            crate::solver::pass2_opt::op_t_quad::eval_op_t_quad_stereo_and_adjoint(
                &base_sig,
                &base_sig,
                &model.transport_quads[0],
                sr,
                None,
            );
        let tgt_l: Vec<f64> = base_sig.iter().zip(&wet_l).map(|(&b, &w)| b + w).collect();
        let tgt_r: Vec<f64> = base_sig.iter().zip(&wet_r).map(|(&b, &w)| b + w).collect();

        let loss_eval = CompositeLossEvaluator::new_for_homotopy(
            &tgt_l,
            &tgt_r,
            None,
            None,
            CompositeLossConfig {
                sample_rate_hz: sr,
                warmup_frames: warmup,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let pairs = StereoTrainingPairs {
            in_l: &x_l,
            in_r: &x_r,
            target_l: &tgt_l,
            target_r: &tgt_r,
        };
        let cfg = Pass2OptConfig {
            num_prescreen_seeds: 16,
            top_k_seeds: 2,
            max_epochs: 8,
            sample_rate_hz: sr,
            warmup_frames: warmup,
            active_quads: 10,
            use_progressive_matryoshka: true,
            adamw_steps: 12,
            ssbroyden_block_iters: 4,
            qat_polish_steps: 2,
            ..Pass2OptConfig::default()
        };
        let hctx = HybridCtx {
            runner: &runner,
            pairs,
            noise_pair: None,
            full_loss: &loss_eval,
            config: cfg,
            warmup,
            max_tau: 512.0,
            kautz_r_max: 0.9992,
            max_tier: 3,
            fast_mode: false,
            progress: None,
        };

        let base_m =
            eval_full_capsule_metrics(&hctx, &model, true, CompositeLossWeights::homotopy_at(1.0))
                .unwrap();
        assert!(
            base_m.0 < 0.0,
            "expected negative Matryoshka log-loss, got {}",
            base_m.0
        );
        let (picked, _, linear_only) = run_phase2a(&hctx, &model).unwrap();
        assert!(
            linear_only,
            "run_phase2a must select linear_only under negative log-loss (base_m={base_m:?})"
        );
        assert!(is_pure_linear_pass1_model(&picked));

        let mut opt_model = model.clone();
        let outcome = run_pass2_optimize(&runner, &mut opt_model, &pairs, cfg, None).unwrap();
        assert!(
            outcome.best_besr_db <= base_m.2 + 0.1 || outcome.best_besr_db <= -60.0,
            "bESR regressed excessively: before={:.2} after={:.2}",
            base_m.2,
            outcome.best_besr_db
        );
        assert!(
            opt_model
                .w_ro
                .iter()
                .all(|row| row.iter().all(|&v| v == 0.0))
        );
        assert!(opt_model.core_quads.iter().all(|q| {
            q.adaa_drive.iter().all(|&v| v.abs() <= 1.0e-12)
                && q.wavefolder_mix.iter().all(|&v| v.abs() <= 1.0e-12)
        }));
    }
}
