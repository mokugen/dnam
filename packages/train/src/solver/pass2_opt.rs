//! Pass-2 generic Sobol prescreening + WebGPU/CPU 256-tile planned adjoint trajectory optimization
//! (`BestUntilDiminishingReturns`, exact FD-verified adjoints for `Op-N[Pre]` Fourier surrogate,
//! stereo continuous Kautz `(r_m, theta_m)` modal poles, and 4-lane `Op-T` recurrent transport,
//! `ref[train.pipeline.backbone-trajectory-opt]`).

pub mod adjoints;
mod eval;
pub mod op_t_quad;
mod qat_polish;
mod theta;

pub use adjoints::{
    eval_kautz_modal_and_adjoint, eval_kautz_modal_channel_and_adjoint, eval_op_n_pre_and_adjoint,
    eval_op_t_transport_and_adjoint,
};
pub use op_t_quad::{OpTQuadAdjointGrads, OpTQuadAdjointSlices, eval_op_t_quad_stereo_and_adjoint};

use crate::error::TrainError;
use crate::gpu::abi::{PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms, TILE_FRAMES};
use crate::linalg::lbfgs::{LbfgsConfig, TwoLoopLbfgs};
use crate::losses::{CompositeLossConfig, CompositeLossEvaluator};
use crate::runner::{PrescreenInput, TrainRunner};

use super::model::{SonaModelF64, StereoTrainingPairs, snap_non_readout_fp16};
use super::progress::{AtomicDnprProgress, ProgressMetrics, TrainingPhase, TrainingStatus};
use eval::{Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio};
use qat_polish::run_pass2_ste_qat_polish;
use theta::ThetaLayout;

use core::f64::consts::PI;

use dnam_kernels::MIN_TPT_G;
use dnam_vm::Tier;

/// Convergence delta in dB below which diminishing-returns patience increments (`0.01 dB/epoch`).
pub const DIMINISHING_RETURNS_DELTA_DB: f64 = 0.01;

/// Configuration for Pass-2 nonlinear + modal + transport optimization (`BestUntilDiminishingReturns`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass2OptConfig {
    /// Generic Sobol/log-spaced prescreen seed count (`64..=128`).
    pub num_prescreen_seeds: usize,
    /// Maximum outer optimization epochs (`BestUntilDiminishingReturns`).
    pub max_epochs: usize,
    /// Consecutive epochs with `|delta_bESR| < 0.01 dB` triggering clean convergence.
    pub diminishing_patience: usize,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Warmup frames excluded from loss.
    pub warmup_frames: usize,
    /// Active `CoreQuad` blocks updated during Pass 2 (`1..=10`).
    pub active_quads: usize,
    /// Straight-Through Estimator `fp16` QAT Adam polish steps after L-BFGS (`0` disables).
    pub qat_polish_steps: usize,
}

impl Default for Pass2OptConfig {
    fn default() -> Self {
        Self {
            num_prescreen_seeds: 64,
            max_epochs: 16,
            diminishing_patience: 4,
            sample_rate_hz: 48_000.0,
            warmup_frames: 64,
            active_quads: 4,
            qat_polish_steps: 8,
        }
    }
}

/// Summary returned after Pass-2 trajectory convergence (`BestUntilDiminishingReturns`).
#[derive(Clone, Debug, PartialEq)]
pub struct Pass2OptReport {
    /// Total outer optimization epochs executed.
    pub epochs_run: usize,
    /// Number of STE `fp16` QAT polish steps executed after L-BFGS.
    pub qat_polish_steps_run: usize,
    /// Quantized baseline composite loss immediately prior to STE QAT polish.
    pub qat_baseline_loss: f64,
    /// Whether convergence stopped on diminishing returns (`|delta_bESR| < 0.01 dB/epoch`).
    pub converged_diminishing_returns: bool,
    /// Best retained approximate composite `bESR` in dB.
    pub best_besr_db: f64,
    /// Best retained composite scalar loss.
    pub best_loss: f64,
    /// Prescreened winning generic seed index (`0..num_prescreen_seeds`).
    pub winning_seed_index: usize,
    /// Wall-clock duration in milliseconds (`0` on `wasm32`).
    pub elapsed_ms: u64,
}

#[inline]
fn van_der_corput(mut idx: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    while idx > 0 {
        f /= base as f64;
        r += f * ((idx % base) as f64);
        idx /= base;
    }
    r
}

/// Generates `64..=128` pure black-box Sobol + log-spaced generic initial parameter seeds across
/// audio bandwidth `[40 Hz, 12 kHz]` matching the 16-slot `prescreen_population` ABI
/// (`[fc_ratio, q_val, w_lp/bp/hp/dir, lru_*, rect_sign, sag_depth, adaa_drive, lru_bias_drive]`).
#[must_use]
pub fn generate_generic_sobol_seeds(num_seeds: usize, sr_hz: f64) -> Vec<f32> {
    let count = num_seeds.clamp(1, 128);
    let sr = sr_hz.max(8_000.0);
    let mut seeds = Vec::with_capacity(count * PRESCREEN_SEED_PARAM_STRIDE);
    for i in 0..count {
        let u0 = van_der_corput(i + 1, 2);
        let u1 = van_der_corput(i + 1, 3);
        let u2 = van_der_corput(i + 1, 5);
        let u3 = van_der_corput(i + 1, 7);
        let fc_hz = 40.0 * (12_000.0 / 40.0_f64).powf(u0);
        let fc_ratio = (fc_hz / sr).clamp(1.0e-4, 0.45) as f32;
        let q_val = (0.5 + 1.5 * u1) as f32;
        let drive = (0.25 * (16.0_f64).powf(u2)) as f32;
        let rel = (0.85 + 0.145 * u3) as f32;
        let row: [f32; PRESCREEN_SEED_PARAM_STRIDE] = [
            fc_ratio,
            q_val,
            0.25,
            0.25 + 0.5 * (u1 as f32),
            0.15,
            0.5 - (u2 as f32),
            rel,
            0.25,
            0.1 * ((u3 - 0.5).abs() as f32),
            0.0,
            0.0,
            0.1 * (u1 as f32),
            1.0,
            0.05 * (u0 as f32),
            drive,
            0.0,
        ];
        seeds.extend_from_slice(&row);
    }
    seeds
}

fn select_and_unpack_prescreen_winner(
    model: &mut SonaModelF64,
    seed_table: &[f32],
    scores: &[f32],
    n_seeds: usize,
    n_frames: usize,
) -> usize {
    let num_tiles = n_frames.div_ceil(TILE_FRAMES).max(1);
    let mut win_idx = 0_usize;
    let mut best_score = f64::INFINITY;
    for s in 0..n_seeds {
        let mut sum = 0.0_f64;
        let mut cnt = 0_usize;
        for tile in 0..num_tiles {
            if let Some(&val) = scores.get(s * num_tiles + tile)
                && val.is_finite()
            {
                sum += f64::from(val);
                cnt += 1;
            }
        }
        if cnt > 0 {
            let avg = sum / (cnt as f64);
            if avg < best_score {
                best_score = avg;
                win_idx = s;
            }
        }
    }

    let s = &seed_table
        [win_idx * PRESCREEN_SEED_PARAM_STRIDE..(win_idx + 1) * PRESCREEN_SEED_PARAM_STRIDE];
    let fc_ratio = f64::from(s[0]).clamp(1.0e-4, 0.45);
    let q_val = f64::from(s[1]).max(0.1);
    let g_val = (PI * fc_ratio).tan().clamp(f64::from(MIN_TPT_G), 64.0);
    let k_val = (1.0 / q_val).clamp(0.0, 8.0);
    let d0 = f64::from(s[14]).clamp(0.25, 16.0);
    let q0 = &mut model.core_quads[0];
    q0.svf_g = [g_val; 4];
    q0.svf_k = [k_val; 4];
    q0.tone_lp = [f64::from(s[2]); 4];
    q0.tone_bp = [f64::from(s[3]); 4];
    q0.tone_hp = [f64::from(s[4]); 4];
    q0.tone_dir = [f64::from(s[5]); 4];
    q0.lru_alpha_rel = [f64::from(s[6]); 4];
    q0.lru_alpha_atk = [f64::from(s[7]); 4];
    q0.lru_kappa = [f64::from(s[8]); 4];
    q0.lru_duhem_gamma = [f64::from(s[9]); 4];
    q0.lru_signed_mix = [f64::from(s[10]); 4];
    q0.lru_stereo_link = [f64::from(s[11]); 4];
    q0.rect_sign = [if s[12] < 0.0 { -1.0 } else { 1.0 }; 4];
    q0.sag_depth = [f64::from(s[13]); 4];
    q0.adaa_drive = [
        d0,
        (0.5 * d0).clamp(0.25, 16.0),
        (2.0 * d0).clamp(0.25, 16.0),
        (4.0 * d0).clamp(0.25, 16.0),
    ];
    q0.lru_bias_drive = [f64::from(s[15]); 4];
    if model.w_ro[0][6].abs() < 1.0e-9 && model.w_ro[0][7].abs() < 1.0e-9 {
        model.w_ro[0][6] = 0.25;
        model.w_ro[0][7] = 0.25;
    }
    win_idx
}

/// Runs Pass-2 Sobol prescreening + L-BFGS optimization across `Op-N[Pre]`, `CoreQuad` ADAA/readout,
/// stereo Kautz resonators, and 4-lane `Op-T` strictly under `BestUntilDiminishingReturns`
/// (`|delta_bESR| < 0.01 dB/epoch`).
pub fn run_pass2_optimize(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    config: Pass2OptConfig,
    progress: Option<&AtomicDnprProgress>,
) -> Result<Pass2OptReport, TrainError> {
    #[cfg(not(target_arch = "wasm32"))]
    let start_instant = std::time::Instant::now();

    let n = pairs.aligned_len();
    let (in_l, in_r, target_l, target_r) = (pairs.in_l, pairs.in_r, pairs.target_l, pairs.target_r);
    let loss_eval = CompositeLossEvaluator::new(
        &target_l[..n],
        &target_r[..n],
        CompositeLossConfig {
            sample_rate_hz: config.sample_rate_hz,
            warmup_frames: config.warmup_frames.min(n / 4),
            ..CompositeLossConfig::default()
        },
    )?;

    let n_seeds = config.num_prescreen_seeds.clamp(1, 128);
    let seed_table = generate_generic_sobol_seeds(n_seeds, config.sample_rate_hz);
    let in_pairs: Vec<[f32; 2]> = (0..n).map(|t| [in_l[t] as f32, in_r[t] as f32]).collect();
    let tgt_pairs: Vec<[f32; 2]> = (0..n)
        .map(|t| [target_l[t] as f32, target_r[t] as f32])
        .collect();
    let scores = runner.prescreen_population(&PrescreenInput {
        uniforms: PrescreenUniforms {
            num_seeds: n_seeds as u32,
            num_frames: n as u32,
            num_quads: 1,
            num_stems: 1,
            sample_rate_hz: config.sample_rate_hz as f32,
            eps_floor: 1.0e-5,
            _pad0: 0,
            _pad1: 0,
        },
        input_audio: &in_pairs,
        target_audio: &tgt_pairs,
        seed_params: &seed_table,
    })?;
    let win_idx = select_and_unpack_prescreen_winner(model, &seed_table, &scores, n_seeds, n);

    let q_count = config.active_quads.clamp(1, 10);
    let layout = ThetaLayout {
        q_opt: q_count,
        k_opt: model.active_kautz_pairs.min(8),
        op_t_enabled: model.transport_quads[0]
            .input_gains
            .iter()
            .chain(&model.transport_quads[0].wet_tap_l)
            .chain(&model.transport_quads[0].wet_tap_r)
            .any(|&v| v.abs() > 1.0e-9),
        max_tau: ((n / 2).min(4_096) as f64).max(8.0),
    };
    let (theta_init, bounds) = layout.pack(model);
    let lbfgs = TwoLoopLbfgs::new(LbfgsConfig {
        max_iters: config.max_epochs.max(1),
        history_m: 8,
        patience: config.max_epochs.max(1) + 2,
        ..LbfgsConfig::default()
    });
    let mut work_model = model.clone();
    let mut best_model = model.clone();
    let mut best_loss = f64::INFINITY;
    let mut best_besr_db = 0.0_f64;
    let mut prev_besr_db = f64::INFINITY;
    let mut streak = 0_usize;
    let mut eval_grad_calls = 0_usize;
    let mut converged = false;
    let mut gl = vec![0.0_f64; n];
    let mut gr = vec![0.0_f64; n];

    lbfgs.minimize(&theta_init, &bounds, |cand, grad_opt| {
        layout.unpack(cand, &mut work_model);
        let eval_buf =
            eval_model_pass2_audio(runner, &work_model, &in_l[..n], &in_r[..n], q_count)?;
        let Some(grad_theta) = grad_opt else {
            return Ok(loss_eval
                .eval_and_adjoint(&eval_buf.pred_l, &eval_buf.pred_r, None, None)?
                .total);
        };

        eval_grad_calls += 1;
        gl.fill(0.0);
        gr.fill(0.0);
        let bd = loss_eval.eval_and_adjoint(
            &eval_buf.pred_l,
            &eval_buf.pred_r,
            Some(&mut gl),
            Some(&mut gr),
        )?;
        let besr_db = 10.0 * bd.besr.max(1.0e-12).log10();
        if bd.total < best_loss {
            best_loss = bd.total;
            best_besr_db = besr_db;
            best_model = work_model.clone();
        }
        if eval_grad_calls >= 2 {
            let step_num = eval_grad_calls - 1;
            if prev_besr_db.is_finite()
                && (prev_besr_db - besr_db).abs() < DIMINISHING_RETURNS_DELTA_DB
            {
                streak += 1;
            } else {
                streak = 0;
            }
            if let Some(p) = progress {
                let perm = ((step_num * 500) / config.max_epochs.max(1)).min(500) as u16;
                p.publish_metrics(&ProgressMetrics {
                    status: TrainingStatus::Running,
                    phase: TrainingPhase::Pass2Backbone,
                    active_tier: Some(Tier::Ultra),
                    progress_permille: perm,
                    step: step_num as u32,
                    total_steps: config.max_epochs as u32,
                    stage_step: step_num as u32,
                    stage_total_steps: config.max_epochs as u32,
                    best_loss,
                    mr_iir: bd.mr_iir,
                    ..ProgressMetrics::default()
                });
            }
            if streak >= config.diminishing_patience {
                converged = true;
                grad_theta.fill(0.0);
                return Ok(bd.total);
            }
        }
        prev_besr_db = besr_db;

        let grads = assemble_pass2_gradients(
            runner,
            &Pass2AdjInput {
                model: &work_model,
                in_l: &in_l[..n],
                in_r: &in_r[..n],
                buf: &eval_buf,
                gl: &gl,
                gr: &gr,
                q_opt: layout.q_opt,
                k_opt: layout.k_opt,
            },
        )?;
        layout.write_grad(&grads, grad_theta);
        Ok(bd.total)
    })?;

    *model = best_model;
    let (qat_polish_steps_run, qat_baseline_loss) = if config.qat_polish_steps > 0 {
        let qat = run_pass2_ste_qat_polish(runner, model, pairs, &loss_eval, &layout, config)?;
        best_loss = qat.best_loss;
        best_besr_db = qat.best_besr_db;
        (qat.steps_run, qat.baseline_loss)
    } else {
        snap_non_readout_fp16(model);
        (0_usize, best_loss)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = start_instant.elapsed().as_millis() as u64;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0_u64;

    Ok(Pass2OptReport {
        epochs_run: eval_grad_calls.saturating_sub(1),
        qat_polish_steps_run,
        qat_baseline_loss,
        converged_diminishing_returns: converged,
        best_besr_db,
        best_loss,
        winning_seed_index: win_idx,
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass2_optimize_runs_sobol_prescreen_and_converges() {
        let runner = TrainRunner::cpu();
        let n = 512_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 0.35 * ((t as f64) * 0.05).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| 0.30 * ((t as f64) * 0.07).cos()).collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.7 * v + 0.2 * v * v * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.5 * v + 0.15 * v * v * v).collect();
        let mut model = SonaModelF64::new(48_000.0);
        let prog = AtomicDnprProgress::new();
        let rep = run_pass2_optimize(
            &runner,
            &mut model,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            Pass2OptConfig {
                num_prescreen_seeds: 64,
                max_epochs: 6,
                diminishing_patience: 2,
                sample_rate_hz: 48_000.0,
                warmup_frames: 32,
                active_quads: 2,
                qat_polish_steps: 4,
            },
            Some(&prog),
        )
        .unwrap();
        assert!(rep.winning_seed_index < 64);
        assert!(rep.best_loss.is_finite());
        assert_eq!(rep.qat_polish_steps_run, 4);
        assert!(rep.best_loss <= rep.qat_baseline_loss);
        assert_eq!(prog.snapshot().phase(), TrainingPhase::Pass2Backbone);

        // Exercise active_quads > 4 (covering Block 0 quads {0, 4}) with qat_polish_steps=0
        // (wire-snapped output) and qat_polish_steps=2 (monotone non-regression).
        let mut model6_nosnap = SonaModelF64::new(48_000.0);
        let rep6_0 = run_pass2_optimize(
            &runner,
            &mut model6_nosnap,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            Pass2OptConfig {
                num_prescreen_seeds: 16,
                max_epochs: 3,
                diminishing_patience: 2,
                sample_rate_hz: 48_000.0,
                warmup_frames: 32,
                active_quads: 6,
                qat_polish_steps: 0,
            },
            None,
        )
        .unwrap();
        assert_eq!(rep6_0.qat_polish_steps_run, 0);
        let mut ref_snapped = model6_nosnap.clone();
        snap_non_readout_fp16(&mut ref_snapped);
        assert_eq!(
            model6_nosnap.core_quads[4].adaa_drive,
            ref_snapped.core_quads[4].adaa_drive
        );
        assert_eq!(
            model6_nosnap.core_quads[4].svf_k_nl,
            ref_snapped.core_quads[4].svf_k_nl
        );
        assert_eq!(model6_nosnap.core_quads[5].svf_k_nl, [0.0; 4]);

        let mut model6_qat = SonaModelF64::new(48_000.0);
        let rep6_2 = run_pass2_optimize(
            &runner,
            &mut model6_qat,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            Pass2OptConfig {
                num_prescreen_seeds: 16,
                max_epochs: 3,
                diminishing_patience: 2,
                sample_rate_hz: 48_000.0,
                warmup_frames: 32,
                active_quads: 6,
                qat_polish_steps: 2,
            },
            None,
        )
        .unwrap();
        assert_eq!(rep6_2.qat_polish_steps_run, 2);
        assert!(rep6_2.best_loss <= rep6_2.qat_baseline_loss);
    }
}
