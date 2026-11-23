//! Pass-2 generic Sobol prescreening + hybrid `AdamW` / `VarPro` / `L-SSBroyden` / `STE QAT`
//! trajectory optimization (`BestUntilDiminishingReturns`, exact FD-verified adjoints for
//! `Op-N[Pre]` Fourier surrogate, stereo continuous Kautz `(r_m, theta_m)` modal poles, and
//! 4-lane `Op-T` recurrent transport, `ref[train.pipeline.backbone-trajectory-opt]`).

pub mod adjoints;
mod causal_loop;
pub(crate) mod crossbar;
mod eval;
mod hybrid;
mod matryoshka;
pub(crate) mod op_n_pre;
pub mod op_t_quad;
mod prescreen;
mod qat_polish;
pub(crate) mod sensory;
mod theta;
mod varpro_sync;
mod zoh_jacobian;

pub use adjoints::{
    eval_kautz_modal_and_adjoint, eval_kautz_modal_channel_and_adjoint,
    eval_op_t_transport_and_adjoint,
};
pub use op_n_pre::{OpNPreGrads, eval_op_n_pre_and_adjoint};
pub use op_t_quad::{OpTQuadAdjointGrads, OpTQuadAdjointSlices, eval_op_t_quad_stereo_and_adjoint};
pub use prescreen::generate_generic_sobol_seeds;

use crate::error::TrainError;
use crate::runner::TrainRunner;

use super::model::{SonaModelF64, StereoTrainingPairs};
use super::progress::AtomicDnprProgress;
use hybrid::run_hybrid_pass2_optimize;

/// Convergence delta in dB below which diminishing-returns patience increments (`0.01 dB/epoch`).
pub const DIMINISHING_RETURNS_DELTA_DB: f64 = 0.01;

/// Configuration for Pass-2 nonlinear + modal + transport optimization (`BestUntilDiminishingReturns`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass2OptConfig {
    /// Generic Sobol/log-spaced prescreen seed count (`64..=128`).
    pub num_prescreen_seeds: usize,
    /// Starting index offset into the quasi-random Sobol sequence for multi-seed exploration.
    pub prescreen_seed_offset: usize,
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
    /// Straight-Through Estimator `fp16` QAT AdamW polish steps after Phase 2C (`0` disables).
    pub qat_polish_steps: usize,
    /// Phase-2A diverse prescreen basins retained (`1..=8`).
    pub top_k_seeds: usize,
    /// Phase-2B chunked TBPTT `AdamW` step budget across progressive stages.
    pub adamw_steps: usize,
    /// Phase-2B chunk length in `48 kHz` frames (`2,048..=4,096` typical).
    pub chunk_frames: usize,
    /// Periodic host `uplo='U'` VarPro readout refresh interval in Phase-2B steps.
    pub varpro_refresh_interval: usize,
    /// Enables progressive `4 -> 6 -> 8 -> 10` expansion and joint 4-tier Matryoshka loss.
    pub use_progressive_matryoshka: bool,
    /// Enables Arena-Scoped Deterministic `PCGrad` during Phase-2B `AdamW` updates (never Phase 2C).
    pub use_pcgrad: bool,
    /// Phase-2C `L-SSBroyden` quasi-Newton steps per streamed block (`4..=25`).
    pub ssbroyden_block_iters: usize,
    /// Phase-2C `L-SSBroyden` secant history memory (`16..=32`).
    pub ssbroyden_history_m: usize,
}

impl Default for Pass2OptConfig {
    fn default() -> Self {
        Self {
            num_prescreen_seeds: 64,
            prescreen_seed_offset: 0,
            max_epochs: 16,
            diminishing_patience: 4,
            sample_rate_hz: 48_000.0,
            warmup_frames: 64,
            active_quads: 4,
            qat_polish_steps: 8,
            top_k_seeds: 4,
            adamw_steps: 40,
            chunk_frames: 4_096,
            varpro_refresh_interval: 15,
            use_progressive_matryoshka: true,
            use_pcgrad: true,
            ssbroyden_block_iters: 16,
            ssbroyden_history_m: 24,
        }
    }
}

impl Pass2OptConfig {
    /// Full desktop 4-tier Matryoshka + `PCGrad` + `L-SSBroyden` + STE `fp16` QAT configuration.
    #[must_use]
    pub const fn desktop_full(sample_rate_hz: f64, warmup_frames: usize) -> Self {
        Self {
            num_prescreen_seeds: 128,
            prescreen_seed_offset: 0,
            max_epochs: 12,
            diminishing_patience: 3,
            sample_rate_hz,
            warmup_frames,
            active_quads: 10,
            qat_polish_steps: 8,
            top_k_seeds: 8,
            adamw_steps: 36,
            chunk_frames: 2_048,
            varpro_refresh_interval: 18,
            use_progressive_matryoshka: true,
            use_pcgrad: true,
            ssbroyden_block_iters: 12,
            ssbroyden_history_m: 24,
        }
    }

    /// Compact browser/WebGPU or `ThresholdStop` 4-tier Matryoshka configuration.
    #[must_use]
    pub const fn browser_webgpu(sample_rate_hz: f64, warmup_frames: usize) -> Self {
        Self {
            num_prescreen_seeds: 64,
            prescreen_seed_offset: 0,
            max_epochs: 12,
            diminishing_patience: 2,
            sample_rate_hz,
            warmup_frames,
            active_quads: 10,
            qat_polish_steps: 8,
            top_k_seeds: 4,
            adamw_steps: 32,
            chunk_frames: 2_048,
            varpro_refresh_interval: 16,
            use_progressive_matryoshka: true,
            use_pcgrad: true,
            ssbroyden_block_iters: 12,
            ssbroyden_history_m: 16,
        }
    }
}

/// Summary returned after Pass-2 trajectory convergence (`BestUntilDiminishingReturns`).
#[derive(Clone, Debug, PartialEq)]
pub struct Pass2OptReport {
    /// Total outer optimization epochs executed.
    pub epochs_run: usize,
    /// Number of STE `fp16` QAT polish steps executed after Phase 2C.
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

/// Runs Pass-2 Sobol prescreening + 4-phase hybrid (`AdamW`/`VarPro` -> `L-SSBroyden` ->
/// `u`-space STE `fp16` QAT) optimization under `BestUntilDiminishingReturns`
/// (`|delta_bESR| < 0.01 dB/epoch`).
pub fn run_pass2_optimize(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    config: Pass2OptConfig,
    progress: Option<&AtomicDnprProgress>,
) -> Result<Pass2OptReport, TrainError> {
    run_pass2_optimize_with_noise(runner, model, pairs, None, config, progress)
}

/// Noise-floor-aware variant of [`run_pass2_optimize`] propagating target repeatability floors
/// `(noise_l, noise_r)` through Phase 2A–2D loss evaluators (`ref[qual.eval.thermal-hiss]`).
pub fn run_pass2_optimize_with_noise(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    noise_pair: Option<(&[f64], &[f64])>,
    config: Pass2OptConfig,
    progress: Option<&AtomicDnprProgress>,
) -> Result<Pass2OptReport, TrainError> {
    #[cfg(not(target_arch = "wasm32"))]
    let start_instant = std::time::Instant::now();

    let out = run_hybrid_pass2_optimize(runner, model, *pairs, noise_pair, config, progress)?;

    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = start_instant.elapsed().as_millis() as u64;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0_u64;

    Ok(Pass2OptReport {
        epochs_run: out.epochs_run,
        qat_polish_steps_run: out.qat_polish_steps_run,
        qat_baseline_loss: out.qat_baseline_loss,
        converged_diminishing_returns: out.converged_diminishing_returns,
        best_besr_db: out.best_besr_db,
        best_loss: out.best_loss,
        winning_seed_index: out.winning_seed_index,
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::canonical_ltas_33_centers_hz;
    use crate::solver::model::snap_non_readout_fp16;
    use crate::solver::progress::TrainingPhase;

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
                ..Pass2OptConfig::default()
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
                ..Pass2OptConfig::default()
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
                ..Pass2OptConfig::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(rep6_2.qat_polish_steps_run, 2);
        assert!(rep6_2.best_loss <= rep6_2.qat_baseline_loss);

        // Pass-1-seeded pure linear FIR target must preserve baseline accuracy without muting VCA.
        let mut fir_model = SonaModelF64::new(48_000.0);
        fir_model.direct_s2_u_gain = [0.0, 0.0];
        fir_model.active_fir_taps = 2;
        fir_model.fir_l[0] = 0.6;
        fir_model.fir_l[1] = -0.25;
        fir_model.fir_r[0] = 0.5;
        fir_model.fir_r[1] = 0.2;
        fir_model.pre_conv_u_gain = [1.0, 1.0];
        let mut fir_tgt_l = vec![0.0_f64; n];
        let mut fir_tgt_r = vec![0.0_f64; n];
        for t in 0..n {
            let xl_prev = if t > 0 { in_l[t - 1] } else { 0.0 };
            let xr_prev = if t > 0 { in_r[t - 1] } else { 0.0 };
            fir_tgt_l[t] = 0.6 * in_l[t] - 0.25 * xl_prev;
            fir_tgt_r[t] = 0.5 * in_r[t] + 0.2 * xr_prev;
        }
        let fir_rep = run_pass2_optimize(
            &runner,
            &mut fir_model,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &fir_tgt_l,
                target_r: &fir_tgt_r,
            },
            Pass2OptConfig {
                num_prescreen_seeds: 16,
                max_epochs: 6,
                diminishing_patience: 3,
                sample_rate_hz: 48_000.0,
                warmup_frames: 32,
                active_quads: 4,
                qat_polish_steps: 2,
                ..Pass2OptConfig::default()
            },
            None,
        )
        .unwrap();
        assert!(fir_rep.best_loss < 0.05, "best_loss={}", fir_rep.best_loss);
        let vca_dot: f64 = fir_model.sense_vca_w.iter().sum();
        assert!(
            vca_dot > -0.5,
            "unexpected negative sense_vca_w sum={vca_dot}"
        );
    }

    #[test]
    fn test_pass2_hybrid_multitier_matryoshka_and_noise_floor_converges() {
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
                            0.025
                                * (core::f64::consts::TAU * fc * sec + phase + 0.37 * (b as f64))
                                    .sin()
                        })
                        .sum()
                })
                .collect()
        };
        let in_l = make_sig(0.0);
        let in_r = make_sig(0.7);
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.72 * v + 0.14 * v * v * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.64 * v - 0.12 * v * v * v).collect();
        let noise_l = vec![1.0e-4_f64; n];
        let noise_r = vec![1.0e-4_f64; n];

        let mut model = SonaModelF64::new(48_000.0);
        let cfg = Pass2OptConfig {
            num_prescreen_seeds: 16,
            prescreen_seed_offset: 0,
            max_epochs: 6,
            diminishing_patience: 2,
            sample_rate_hz: 48_000.0,
            warmup_frames: 32,
            active_quads: 10,
            qat_polish_steps: 2,
            top_k_seeds: 2,
            adamw_steps: 8,
            chunk_frames: 512,
            varpro_refresh_interval: 4,
            use_progressive_matryoshka: true,
            use_pcgrad: true,
            ssbroyden_block_iters: 4,
            ssbroyden_history_m: 16,
        };
        let rep = run_pass2_optimize_with_noise(
            &runner,
            &mut model,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            Some((&noise_l, &noise_r)),
            cfg,
            None,
        )
        .unwrap();
        assert!(rep.best_loss.is_finite());
        assert!(rep.best_loss <= rep.qat_baseline_loss);
        assert!(
            rep.best_besr_db < -15.0,
            "multi-tier Matryoshka should fit cleanly: besr_db={}",
            rep.best_besr_db
        );
        let desktop_cfg = Pass2OptConfig::desktop_full(48_000.0, 64);
        assert_eq!(desktop_cfg.active_quads, 10);
        assert_eq!(desktop_cfg.top_k_seeds, 8);
        let browser_cfg = Pass2OptConfig::browser_webgpu(48_000.0, 64);
        assert_eq!(browser_cfg.chunk_frames, 2_048);
    }
}
