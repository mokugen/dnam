//! End-to-end 5-pass universal black-box `SONA v3.0` system identification orchestrator
//! (`Phase 1..3 Control Audit` + `Pass 1 Probe` + `Pass 2 WebGPU/CPU Adjoint Trajectory` +
//! `Pass 3 Nested VarPro/Schur` + `Pass 4 CP1/Wiener` + `Pass 5 Certified .dnam-bin Export`).

pub mod model;
pub mod pass1_probe;
pub mod pass2_opt;
pub mod pass3_varpro;
pub mod pass5_emit;
pub mod progress;

pub use model::{
    CoreQuadF64, InterBlockTileF64, KautzPairF64, NUM_CORE_QUADS_ULTRA, NUM_FIR_TAPS_ULTRA,
    NUM_INTER_BLOCK_TILES_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, NUM_TRANSPORT_QUADS_ULTRA, SonaModelF64,
    StereoTrainingPairs, TierReadoutPrefixF64, TransportQuadF64,
};
pub use pass1_probe::{
    OpPProbeSummary, Pass1ProbeReport, run_cpu_op_p_probe, run_pass1_probe,
    steiglitz_mcbride_poles, synth_kautz_unit_mode,
};
pub use pass2_opt::{
    DIMINISHING_RETURNS_DELTA_DB, Pass2OptConfig, Pass2OptReport, eval_kautz_modal_and_adjoint,
    eval_op_n_pre_and_adjoint, eval_op_t_transport_and_adjoint, generate_generic_sobol_seeds,
    run_pass2_optimize,
};
pub use pass3_varpro::{Pass3VarProReport, run_pass3_varpro_readout};
pub use pass5_emit::{
    MAX_INTER_TIER_LU_DRIFT, Pass5EmitConfig, Pass5EmitResult, compute_itu_bs1770_lufs,
    encode_model_to_dnam_bin, run_pass5_emit_and_certify,
};
pub use progress::{
    AtomicDnprProgress, DNPR_MAGIC, DnprProgressWire, PROGRESS_MAGIC_DNPR, ProgressMetrics,
    TrainingPhase, TrainingStatus, quantize_kautz_pole_q8, quantize_ltas_error_q20_i8,
};

use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use crate::control::clamp::{
    ControlSpec, Phase1ClampOutput, StageStabilityConstraint, audit_phase1_controls,
};
use crate::control::cp1::{
    Cp1ControlSample, Cp1FitConfig, Cp1WireBundle, fit_and_pack_cp1_audited,
};
use crate::control::wiener::{
    FourBandTptBank, solve_control_jacobian_32slot, solve_two_pass_four_band_wiener,
};
use crate::error::TrainError;
use crate::losses::Welch33TargetCache;
use crate::runner::TrainRunner;
use pass5_emit::run_pass5_emit_and_certify_with_digest;

/// Raw stereo audio capture at one physical control operating point used for Pass-4 Wiener
/// + 32-slot Jacobian extraction (`ref[train.pipeline.control-projection-fit]`).
#[derive(Clone, Debug)]
pub struct ControlAudioStem {
    /// Physical continuous control values matching [`SonaTrainDataset::controls`].
    pub phys_controls: Vec<f32>,
    /// Active discrete switch options matching [`SonaTrainDataset::controls`].
    pub switch_options: Vec<u8>,
    /// Dry Left excitation waveform.
    pub input_l: Vec<f64>,
    /// Dry Right excitation waveform.
    pub input_r: Vec<f64>,
    /// Aligned wet Left target captured at this control point.
    pub target_l: Vec<f64>,
    /// Aligned wet Right target captured at this control point.
    pub target_r: Vec<f64>,
}

/// Full training dataset supplied to [`solve_sona_model`] (zero fixture/brand assumptions).
#[derive(Clone, Debug)]
pub struct SonaTrainDataset {
    /// Sample rate in Hz (`44_100.0..=96_000.0`).
    pub sample_rate_hz: f64,
    /// Pre-roll warmup sample count.
    pub warmup_frames: usize,
    /// Dry stereo reference excitation `(input_l, input_r)`.
    pub input_l: Vec<f64>,
    /// Dry stereo Right channel (`== input_l` for mono-to-stereo / single-in).
    pub input_r: Vec<f64>,
    /// Aligned wet stereo target `(target_l, target_r)` at `StarCenter`.
    pub target_l: Vec<f64>,
    /// Aligned Right wet target at `StarCenter`.
    pub target_r: Vec<f64>,
    /// Parametric control schema specs (`len <= 255`).
    pub controls: Vec<ControlSpec>,
    /// Stability constraint linking control excursions to stage delay/loop limits.
    pub stability_constraint: StageStabilityConstraint,
    /// Multi-control operating points for `CP1` ANOVA/FM + `"CEXT"` dispatch fitting.
    pub control_samples: Vec<Cp1ControlSample>,
    /// Optional raw audio stems at non-center control points for Pass-4 Wiener + Jacobian extraction.
    pub control_audio_stems: Vec<ControlAudioStem>,
    /// Optional pure-binary `"TSYN"` tempo-sync rows.
    pub tsyn_entries: Vec<dnam_vm::bin::TempoSyncEntryWire>,
}

/// Top-level solver pipeline configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaSolverConfig {
    /// Early FIR tap capacity (`32 / 48 / 96 / 128`).
    pub max_fir_taps: usize,
    /// Kautz modal pair capacity (`8 / 12 / 24 / 32`).
    pub max_kautz_pairs: usize,
    /// Pass-2 optimization configuration (`BestUntilDiminishingReturns`).
    pub pass2: Pass2OptConfig,
    /// `CP1` regression configuration.
    pub cp1: Cp1FitConfig,
    /// Pass-5 `.dnam-bin` emission and calibration configuration.
    pub pass5: Pass5EmitConfig,
}

impl Default for SonaSolverConfig {
    fn default() -> Self {
        Self {
            max_fir_taps: 128,
            max_kautz_pairs: 32,
            pass2: Pass2OptConfig::default(),
            cp1: Cp1FitConfig::default(),
            pass5: Pass5EmitConfig::default(),
        }
    }
}

/// End-to-end solved SONA v3.0 artifact & diagnostic summary.
#[derive(Clone, Debug)]
pub struct SonaSolveResult {
    /// Final converged and stability-audited model parameters.
    pub model: SonaModelF64,
    /// Phase-1 control clamp audit report & serialized `64 B` descriptors.
    pub phase1_audit: Option<Phase1ClampOutput>,
    /// Fitted `CP1` + `"CEXT"` + `"TSYN"` control bus bundle.
    pub control_bus: Option<Cp1WireBundle>,
    /// Pass-1 universal probe report.
    pub pass1: Pass1ProbeReport,
    /// Pass-2 trajectory optimization report.
    pub pass2: Pass2OptReport,
    /// Pass-3 nested VarPro readout report.
    pub pass3: Pass3VarProReport,
    /// Pass-5 certified `.dnam-bin`, tier strips, LUFS & 4-metric reports.
    pub pass5: Pass5EmitResult,
}

/// Executes the complete 5-pass universal black-box `SONA v3.0` solver on `runner` (`CPU` or `wgpu`).
pub fn solve_sona_model(
    runner: &TrainRunner,
    dataset: &SonaTrainDataset,
    config: &SonaSolverConfig,
    progress: Option<&AtomicDnprProgress>,
) -> Result<SonaSolveResult, TrainError> {
    let n = dataset
        .input_l
        .len()
        .min(dataset.input_r.len())
        .min(dataset.target_l.len())
        .min(dataset.target_r.len());
    let warmup = dataset.warmup_frames.min(n / 4);

    if let Some(p) = progress {
        p.publish_metrics(&ProgressMetrics {
            status: progress::TrainingStatus::Running,
            phase: progress::TrainingPhase::Pass1Probe,
            ..ProgressMetrics::default()
        });
    }

    // Phase 1: `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]` audit.
    let phase1_audit = if dataset.controls.is_empty() {
        None
    } else {
        Some(audit_phase1_controls(
            &dataset.controls,
            &dataset.stability_constraint,
        )?)
    };

    // Pass 1: `< 0.5 ms` CPU `Op-P` probe + Steiglitz–McBride / FIR / Kautz / Op-T init.
    let (pass1, mut model) = run_pass1_probe(
        &dataset.input_l[..n],
        &dataset.input_r[..n],
        &dataset.target_l[..n],
        &dataset.target_r[..n],
        dataset.sample_rate_hz,
        config.max_fir_taps,
        config.max_kautz_pairs,
    )?;
    model.stability_constraint = dataset.stability_constraint.clone();

    // Pass 2: Generic Sobol prescreening + planned core scan + L-BFGS trajectory opt.
    let pairs = StereoTrainingPairs {
        in_l: &dataset.input_l[..n],
        in_r: &dataset.input_r[..n],
        target_l: &dataset.target_l[..n],
        target_r: &dataset.target_r[..n],
    };
    let pass2 = run_pass2_optimize(
        runner,
        &mut model,
        &pairs,
        Pass2OptConfig {
            sample_rate_hz: dataset.sample_rate_hz,
            warmup_frames: warmup,
            ..config.pass2
        },
        progress,
    )?;

    // Pass 3: Joint 4-tier nested VarPro + Schur prefix + `ro_pole_48k` sweep.
    let pass3 = run_pass3_varpro_readout(
        runner,
        &mut model,
        &dataset.input_l[..n],
        &dataset.input_r[..n],
        &dataset.target_l[..n],
        &dataset.target_r[..n],
        warmup,
    )?;

    // Pass 4: Wiener/32-slot stem extraction + `CP1` Lobatto ANOVA/FM + `"CEXT"` / `"TSYN"` fitting.
    model.control_audit = phase1_audit.clone();
    model.control_bus = None;

    let control_bus = if let Some(ref p1) = phase1_audit {
        let wiener_samps = synthesize_wiener_control_samples(
            &model,
            &dataset.control_audio_stems,
            warmup,
            config.pass5,
        )?;
        let mut samps = dataset.control_samples.clone();
        samps.extend(wiener_samps);
        if samps.is_empty() {
            samps.push(Cp1ControlSample {
                phys_controls: vec![0.0; dataset.controls.len()],
                switch_options: vec![0; dataset.controls.len()],
                stage_targets_base: [0.0; STAGE_TARGET_SLOTS],
                stage_targets_ultra_delta: None,
            });
        }
        Some(fit_and_pack_cp1_audited(
            &dataset.controls,
            p1,
            &samps,
            &dataset.tsyn_entries,
            config.cp1,
        )?)
    } else {
        None
    };

    model.control_bus = control_bus.clone();

    // Pass 5: `.dnam-bin` emit + LUFS calibration + Phase-2/3 final certification + 4-metric check.
    let pass5_pairs = model::StereoTrainingPairs {
        in_l: &dataset.input_l[..n],
        in_r: &dataset.input_r[..n],
        target_l: &dataset.target_l[..n],
        target_r: &dataset.target_r[..n],
    };
    let pass5 = run_pass5_emit_and_certify_with_digest(
        &mut model,
        &pass5_pairs,
        warmup,
        config.pass5,
        Some(pass3.stat_gram_digest),
    )?;

    if let Some(p) = progress {
        let u = &pass5.tier_reports[3];
        let clamped_ctrls = phase1_audit.as_ref().map_or(0_u16, |a| {
            a.locked_controls.iter().filter(|&&l| l).count() as u16
        });
        let win = n.saturating_sub(warmup).min(64);
        let peak_t = dataset.target_l[warmup..warmup + win]
            .iter()
            .fold(1.0e-6_f64, |acc, &v| acc.max(v.abs()));
        let inv_peak = 127.0 / peak_t;
        let mut target_wave_i8 = [0_i8; 64];
        let mut model_wave_i8 = [0_i8; 64];
        for k in 0..win {
            target_wave_i8[k] = (dataset.target_l[warmup + k] * inv_peak)
                .round()
                .clamp(-127.0, 127.0) as i8;
            model_wave_i8[k] = (pass5.vm_renders[3].0[warmup + k] * inv_peak)
                .round()
                .clamp(-127.0, 127.0) as i8;
        }
        let mut tier_besr_db = [f32::NAN; 4];
        let mut tier_ltas_db = [f32::NAN; 4];
        for (i, rep) in pass5.tier_reports.iter().enumerate() {
            tier_besr_db[i] = rep.besr_db as f32;
            tier_ltas_db[i] = rep.ltas_max_db as f32;
        }
        let mut poles = [[0_u8; 2]; 16];
        for (i, kp) in model
            .kautz
            .iter()
            .take(model.active_kautz_pairs.min(16))
            .enumerate()
        {
            poles[i] =
                quantize_kautz_pole_q8(kp.r_low + 0.5 * kp.dr, kp.theta, dataset.sample_rate_hz);
        }
        let q20 = match (
            Welch33TargetCache::new(&dataset.target_l[..n], warmup, dataset.sample_rate_hz),
            Welch33TargetCache::new(&pass5.vm_renders[3].0[..n], warmup, dataset.sample_rate_hz),
        ) {
            (Ok(tgt_cache), Ok(mdl_cache)) => quantize_ltas_error_q20_i8(
                &mdl_cache.tgt_band_p[..32],
                &tgt_cache.tgt_band_p[..32],
                tgt_cache.floor,
            ),
            _ => [0_i8; 32],
        };
        p.publish_metrics(&ProgressMetrics {
            status: TrainingStatus::Completed,
            phase: TrainingPhase::Completed,
            active_tier: Some(Tier::Ultra),
            progress_permille: 1_000,
            step: 5,
            total_steps: 5,
            stage_step: pass2.epochs_run as u32,
            stage_total_steps: config.pass2.max_epochs as u32,
            best_loss: pass2.best_loss,
            mr_iir: u.mr_iir,
            clamped_controls: clamped_ctrls,
            has_visual_frame: true,
            tier_besr_db,
            tier_ltas_db,
            target_wave_i8,
            model_wave_i8,
            ltas_error_q20_i8: q20,
            kautz_poles_q8: poles,
        });
    }

    Ok(SonaSolveResult {
        model,
        phase1_audit,
        control_bus,
        pass1,
        pass2,
        pass3,
        pass5,
    })
}

const PASS4_ACTIVE_SLOTS: [usize; 11] = [0, 1, 2, 6, 10, 11, 13, 15, 22, 26, 30];

struct Pass4StemProbeCtx<'a> {
    dnam_bin: &'a [u8],
    sample_rate_hz: f64,
}

fn render_star_with_stage_targets(
    ctx: &Pass4StemProbeCtx<'_>,
    tier: Tier,
    in_l: &[f64],
    in_r: &[f64],
    st32: &[f64; STAGE_TARGET_SLOTS],
) -> Result<Vec<f64>, TrainError> {
    let sliced = dnam_vm::bin::strip_dnam_bin_to_tier(ctx.dnam_bin, tier)?;
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(sliced, tier, ctx.sample_rate_hz as f32)?;
    let mut st32_f32 = [0.0_f32; STAGE_TARGET_SLOTS];
    for (dst, &src) in st32_f32.iter_mut().zip(st32) {
        *dst = if src.is_finite() { src as f32 } else { 0.0 };
    }
    {
        let cold = vm.slab_mut().cold_mut();
        cold.ctrl_bus = None;
        cold.latched_stage_params = st32_f32;
    }
    vm.step_control_tick_3khz();

    let n = in_l.len().min(in_r.len());
    let in_l32: Vec<f32> = in_l[..n].iter().map(|&v| v as f32).collect();
    let in_r32: Vec<f32> = in_r[..n].iter().map(|&v| v as f32).collect();
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];
    vm.process_block(&in_l32, &in_r32, &mut out_l, &mut out_r)?;
    Ok(out_l
        .into_iter()
        .zip(out_r)
        .map(|(l, r)| 0.5 * (f64::from(l) + f64::from(r)))
        .collect())
}

fn synthesize_wiener_control_samples(
    model: &SonaModelF64,
    stems: &[ControlAudioStem],
    warmup: usize,
    pass5_cfg: Pass5EmitConfig,
) -> Result<Vec<Cp1ControlSample>, TrainError> {
    if stems.is_empty() {
        return Ok(Vec::new());
    }
    let star_bin = encode_model_to_dnam_bin(model, pass5_cfg)?;
    let ctx = Pass4StemProbeCtx {
        dnam_bin: &star_bin,
        sample_rate_hz: model.sample_rate_hz,
    };
    let bank = FourBandTptBank::new(model.sample_rate_hz);
    let zero_st = [0.0_f64; STAGE_TARGET_SLOTS];
    let mut out = Vec::with_capacity(stems.len());

    for stem in stems {
        let m = stem
            .input_l
            .len()
            .min(stem.input_r.len())
            .min(stem.target_l.len())
            .min(stem.target_r.len());
        if m <= warmup {
            return Err(TrainError::InvalidDimensions {
                context: "synthesize_wiener_control_samples",
                expected: warmup + 1,
                actual: m,
            });
        }
        let tgt_m: Vec<f64> = (0..m)
            .map(|t| 0.5 * (stem.target_l[t] + stem.target_r[t]))
            .collect();
        let in_l = &stem.input_l[..m];
        let in_r = &stem.input_r[..m];
        let y_star_u = render_star_with_stage_targets(&ctx, Tier::Ultra, in_l, in_r, &zero_st)?;
        let y_star_l = render_star_with_stage_targets(&ctx, Tier::Lite, in_l, in_r, &zero_st)?;

        let st_pass1 = solve_control_jacobian_32slot(
            &tgt_m,
            &zero_st,
            &PASS4_ACTIVE_SLOTS,
            1.0e-2,
            1.0e-2,
            |st| render_star_with_stage_targets(&ctx, Tier::Ultra, in_l, in_r, st),
        )?;
        let y_mdl_u = render_star_with_stage_targets(&ctx, Tier::Ultra, in_l, in_r, &st_pass1)?;
        let y_mdl_l = render_star_with_stage_targets(&ctx, Tier::Lite, in_l, in_r, &st_pass1)?;

        let wr = solve_two_pass_four_band_wiener(
            &bank, &tgt_m, &y_star_u, &y_star_l, &y_mdl_u, &y_mdl_l, warmup,
        )?;
        let mut st_final = st_pass1;
        if wr.pass2_triggered {
            let bands = bank.split_4bands(&y_mdl_u);
            let mut tgt_w = vec![0.0_f64; m];
            for t in 0..m {
                let boosted = (0..4)
                    .map(|b| wr.wiener_band_gains[b] * bands[b][t])
                    .sum::<f64>();
                tgt_w[t] = 0.5 * tgt_m[t] + 0.5 * boosted;
            }
            let delta2 = solve_control_jacobian_32slot(
                &tgt_w,
                &st_pass1,
                &PASS4_ACTIVE_SLOTS,
                1.0e-2,
                2.0e-2,
                |st| render_star_with_stage_targets(&ctx, Tier::Ultra, in_l, in_r, st),
            )?;
            for s in PASS4_ACTIVE_SLOTS {
                st_final[s] = (st_pass1[s] + delta2[s]).clamp(-3.0, 3.0);
            }
        }

        out.push(Cp1ControlSample {
            phys_controls: stem.phys_controls.clone(),
            switch_options: stem.switch_options.clone(),
            stage_targets_base: st_final,
            stage_targets_ultra_delta: None,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_sona_model_end_to_end_cpu_smoke() {
        let runner = TrainRunner::cpu();
        let n = 512_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                0.35 * ((i as f64) * 0.06).sin()
                    + 0.2 * ((i as f64) * 0.31).cos()
                    + 0.1 * ((i as f64) * 0.95).sin()
            })
            .collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.8 * v).collect();
        let ds = SonaTrainDataset {
            sample_rate_hz: sr,
            warmup_frames: 64,
            input_l: in_l.clone(),
            input_r: in_l,
            target_l: tgt_l.clone(),
            target_r: tgt_l,
            controls: Vec::new(),
            stability_constraint: StageStabilityConstraint::default(),
            control_samples: Vec::new(),
            control_audio_stems: Vec::new(),
            tsyn_entries: Vec::new(),
        };
        let prog = AtomicDnprProgress::new();
        let mut cfg = SonaSolverConfig::default();
        cfg.pass2.max_epochs = 4;
        let res = solve_sona_model(&runner, &ds, &cfg, Some(&prog)).unwrap();
        assert!(dnam_vm::bin::DnamBinView::parse(&res.pass5.dnam_bin).is_ok());
        let snap = prog.snapshot();
        assert_eq!(snap.magic.get(), PROGRESS_MAGIC_DNPR);
        assert_eq!(snap.status(), TrainingStatus::Completed);
        assert_eq!(snap.phase(), TrainingPhase::Completed);
        assert_eq!(snap.progress_permille(), 1_000);
    }
}
