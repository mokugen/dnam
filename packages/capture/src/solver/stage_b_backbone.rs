#![forbid(unsafe_code)]
//! Stage B (`Joint 4-Tier SONA Backbone` & `256-D` `uplo='U'` `STAT` Contraction, `solver::stage_b_backbone`).
//!
//! Canonical Reference: `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.backbone-trajectory-opt]`).

use dnam_engine::schema::DnamError;
use dnam_train::solver::StereoTrainingPairs;
use dnam_train::solver::pass2_opt::Pass2OptConfig;
use dnam_train::{
    Pass1ProbeResult, SonaModelF64, TrainRunner, WeightedReadoutStem, decode_dnam_bin_to_model,
    extract_stat_256d_gram, patch_model_readouts_weighted_cholesky, run_pass1_probe,
    run_pass2_optimize, run_pass3_varpro_readout, uplo_u_packed_len,
};
use dnam_vm::bin::{ARENA_TIER_BYTES, DnamBinView};
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};
use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::F64;

use crate::proj::PROJ_STAT_BYTES_ULTRA;
use crate::proj::capsules::verify_proj_stat_uplo_u;
use crate::proj::schema::DnamProjManifest;
use crate::proj::verify_dnam_bin_zero_dc;
use crate::solver::progress::telemetry::{
    VisualMilestoneMeta, VisualProbeAnchor, emit_sona_visual_milestone,
};
use crate::solver::progress::{
    TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus,
};
use crate::solver::stage_a_probe::{encode_with_preserved_controls, has_populated_v3_backbone};
use crate::solver::{SolverQualityMode, canonical_besr_warmup};

/// Borrowed stereo stem reference and regression weight for fast weighted readout patching.
#[derive(Debug, Clone, Copy)]
pub struct WeightedStemRef<'a> {
    /// Dry stereo input waveform at `48 kHz`.
    pub dry_lr: &'a [[f32; 2]],
    /// Target wet stereo waveform at `48 kHz`.
    pub wet_lr: &'a [[f32; 2]],
    /// Non-negative stem weight in the `uplo='U'` normal equations.
    pub weight: f64,
}

const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
const _: () = assert!(ALL_TIERS.len() == 4);

/// Output summary and `uplo='U'` sufficient statistics from [`solve_stage_b_backbone`].
#[derive(Debug, Clone)]
pub struct StageBOutputs {
    /// Wall-clock Stage B duration in milliseconds.
    pub elapsed_ms: f32,
    /// Canonical `267,264 B` `uplo='U'` `STAT` chunk payload (`PROJ_STAT_BYTES_ULTRA`).
    pub stat_bytes: Vec<u8>,
    /// Packed upper-triangular `256 × 256` `StarCenter` Gram matrix (`32,896` elements).
    pub packed_gram_256: Vec<f64>,
    /// `256 × 2` stereo `StarCenter` feature-target cross-correlations.
    pub cross_256: [[f64; 2]; 256],
    /// Post-warmup `[Lite, Full, Pro, Ultra]` `StarCenter` ESR in dB.
    pub prefix_esr_db: [f32; 4],
}

/// Configuration for [`solve_stage_b_backbone_with_progress`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageBSolveConfig {
    /// Frame stride when contracting `256 × 256` `uplo='U'` Gram statistics (`default = 4`).
    pub gram_frame_stride: usize,
    /// Preserve pre-populated white-box `SONA` backbone weights while solving readouts.
    pub preserve_existing_backbone: bool,
    /// Optimization convergence vs. threshold-stop quality policy.
    pub quality_mode: SolverQualityMode,
    /// Optional empirical left/right repeatability profiles measured strictly from `(x, y)` probes.
    pub rep_profiles: Option<[crate::metrics::TargetRepeatabilityProfile; 2]>,
}

impl Default for StageBSolveConfig {
    fn default() -> Self {
        Self {
            gram_frame_stride: 4,
            preserve_existing_backbone: false,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            rep_profiles: None,
        }
    }
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

#[inline]
fn map_stage_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

fn validate_ultra_dnam_bin(dnam_bin: &[u8]) -> Result<(), DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(|_| DnamError::BufferTooSmall {
        expected: ARENA_TIER_BYTES[3],
        actual: dnam_bin.len(),
    })?;
    if view.retained_tier != Tier::Ultra {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_TIER_BYTES[3],
            actual: view.retained_tier.arena_bytes(),
        });
    }
    Ok(())
}

fn build_stereo_buffers(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    mono_out: bool,
    dual_in: bool,
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let keep_in_r = dual_in || !mono_out;
    let in_l: Vec<f64> = dry_lr[..n].iter().map(|lr| f64::from(lr[0])).collect();
    let in_r: Vec<f64> = dry_lr[..n]
        .iter()
        .map(|lr| f64::from(if keep_in_r { lr[1] } else { lr[0] }))
        .collect();
    let tgt_l: Vec<f64> = wet_lr[..n].iter().map(|lr| f64::from(lr[0])).collect();
    let tgt_r: Vec<f64> = wet_lr[..n]
        .iter()
        .map(|lr| f64::from(if mono_out { lr[0] } else { lr[1] }))
        .collect();
    (in_l, in_r, tgt_l, tgt_r)
}

/// Serializes `(packed_gram_256, cross_256)` into the canonical `267,264 B` `uplo='U'` `STAT` chunk
/// (`PROJ_STAT_BYTES_ULTRA`, `ref[container.proj.stat-uplo-u]`).
#[must_use]
pub fn serialize_v3_stat_chunk_ultra(
    packed_gram_256: &[f64],
    cross_256: &[[f64; 2]; 256],
) -> Vec<u8> {
    let n_pack = uplo_u_packed_len(256);
    let mut words = Vec::<F64>::with_capacity(n_pack + 512);
    for idx in 0..n_pack {
        let v = packed_gram_256.get(idx).copied().unwrap_or(0.0);
        words.push(F64::new(if v.is_finite() { v } else { 0.0 }));
    }
    for &[cl, cr] in cross_256 {
        words.push(F64::new(if cl.is_finite() { cl } else { 0.0 }));
        words.push(F64::new(if cr.is_finite() { cr } else { 0.0 }));
    }
    debug_assert_eq!(words.as_bytes().len(), PROJ_STAT_BYTES_ULTRA);
    words.as_bytes().to_vec()
}

fn eval_tier_esr_db(
    dnam_bin: &[u8],
    tier: Tier,
    mono_out: bool,
    dual_in: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> Result<f32, DnamError> {
    let n = dry_lr.len().min(wet_lr.len());
    if n == 0 {
        return Ok(0.0);
    }
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let mut vm =
        SonaVm::<DefaultMathBackend>::from_view_owned(&view, tier, 48_000.0, mono_out && !dual_in)
            .map_err(map_stage_err)?;
    let keep_in_r = dual_in || !mono_out;
    let in_l: Vec<f32> = dry_lr[..n].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr[..n]
        .iter()
        .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
        .collect();
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    vm.process_block(&in_l, &in_r, &mut out_l, &mut out_r)
        .map_err(map_stage_err)?;

    let warmup = canonical_besr_warmup(n).min(n.saturating_sub(1));
    let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
    for t in warmup..n {
        let tl = f64::from(wet_lr[t][0]);
        let dl = tl - f64::from(out_l[t]);
        sig_sq += tl * tl;
        err_sq += dl * dl;
        if !mono_out {
            let tr = f64::from(wet_lr[t][1]);
            let dr = tr - f64::from(out_r[t]);
            sig_sq += tr * tr;
            err_sq += dr * dr;
        }
    }
    if sig_sq <= 1.0e-18 {
        Ok(if err_sq <= 1.0e-18 { -120.0 } else { 0.0 })
    } else {
        Ok((10.0 * (err_sq / sig_sq).max(1.0e-16).log10()) as f32)
    }
}

fn eval_all_four_tier_esr_db(
    dnam_bin: &[u8],
    mono_out: bool,
    dual_in: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> Result<[f32; 4], DnamError> {
    let mut out = [0.0_f32; 4];
    for (idx, &tier) in ALL_TIERS.iter().enumerate() {
        out[idx] = eval_tier_esr_db(dnam_bin, tier, mono_out, dual_in, dry_lr, wet_lr)?;
    }
    Ok(out)
}

/// Serializes `dnam_bin`'s `Ultra` trajectory Gram statistics (`267,264 B` `uplo='U'` `STAT`) directly.
pub fn compute_v3_stat_chunk_bytes(
    dnam_bin: &[u8],
    mono_out: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_samples: usize,
    stride: usize,
) -> Result<Vec<u8>, DnamError> {
    validate_ultra_dnam_bin(dnam_bin)?;
    let n = active_samples.min(dry_lr.len()).min(wet_lr.len());
    let warmup = canonical_besr_warmup(n).min(n.saturating_sub(1));
    let (packed_gram_256, cross_256) = if n > warmup + 1 {
        let model = decode_dnam_bin_to_model(dnam_bin).map_err(map_stage_err)?;
        let (in_l, in_r, tgt_l, tgt_r) = build_stereo_buffers(dry_lr, wet_lr, mono_out, true, n);
        extract_stat_256d_gram(
            &TrainRunner::cpu(),
            &model,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            warmup,
            stride.max(1),
        )
        .map_err(map_stage_err)?
    } else {
        (vec![0.0; uplo_u_packed_len(256)], [[0.0; 2]; 256])
    };
    let stat_bytes = serialize_v3_stat_chunk_ultra(&packed_gram_256, &cross_256);
    let _ = verify_proj_stat_uplo_u(&stat_bytes)?;
    Ok(stat_bytes)
}

fn select_pass2_config(quality_mode: SolverQualityMode, warmup: usize) -> Pass2OptConfig {
    match quality_mode {
        SolverQualityMode::BestUntilDiminishingReturns => {
            Pass2OptConfig::desktop_full(48_000.0, warmup)
        }
        SolverQualityMode::ThresholdStop {
            max_polish_iters, ..
        } => Pass2OptConfig {
            max_epochs: max_polish_iters.clamp(2, 16),
            adamw_steps: (max_polish_iters * 2).clamp(4, 24),
            diminishing_patience: 2,
            qat_polish_steps: 2,
            ..Pass2OptConfig::browser_webgpu(48_000.0, warmup)
        },
    }
}

fn emit_stage_b_milestone(
    progress: &TrainerProgressHandle,
    model: &SonaModelF64,
    dry_lr: &[[f32; 2]],
    anchor: &Option<VisualProbeAnchor>,
    permille: u16,
    step: u32,
) {
    if let Some(a) = anchor {
        let m = a.render_len.min(dry_lr.len());
        let in_l: Vec<f32> = dry_lr[..m].iter().map(|lr| lr[0]).collect();
        let in_r: Vec<f32> = dry_lr[..m].iter().map(|lr| lr[1]).collect();
        emit_sona_visual_milestone(
            progress,
            model,
            &in_l,
            &in_r,
            a,
            VisualMilestoneMeta {
                phase: TrainingPhase::StageBBackbone,
                permille,
                step,
                total_steps: 2,
                stage_step: step,
                stage_total_steps: 2,
                tier: Tier::Ultra,
                best_loss: f32::NAN,
                mr_iir_loss: f32::NAN,
                tier_besr_db: [f32::NAN; 4],
                tier_ltas_db: [f32::NAN; 4],
            },
        );
    }
}

/// Executes Stage B (`Joint 4-Tier SONA Backbone` fit or white-box preservation) and
/// contracts the canonical `267,264 B` `uplo='U'` `STAT` chunk (`ref[train.pipeline.backbone-trajectory-opt]`).
pub fn solve_stage_b_backbone(
    manifest: &DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_frames_hint: usize,
    gram_frame_stride: usize,
    preserve_existing_backbone: bool,
) -> Result<StageBOutputs, DnamError> {
    solve_stage_b_backbone_with_progress(
        manifest,
        dnam_bin,
        dry_lr,
        wet_lr,
        active_frames_hint,
        StageBSolveConfig {
            gram_frame_stride,
            preserve_existing_backbone,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            rep_profiles: None,
        },
        &TrainerProgressHandle::default(),
    )
}

/// Arguments controlling [`solve_stage_b_backbone_with_pass1`].
#[derive(Debug, Clone)]
pub struct StageBBackboneSpec {
    /// Optional non-zero prefix frame limit (`0` = use all aligned frames).
    pub active_frames_hint: usize,
    /// Backbone solve configuration (`gram_frame_stride`, `quality_mode`, etc.).
    pub config: StageBSolveConfig,
    /// Optional precomputed Stage A [`Pass1ProbeResult`] reused without re-probing.
    pub precomputed_pass1: Option<Pass1ProbeResult>,
}

/// Variant of [`solve_stage_b_backbone`] accepting [`StageBSolveConfig`] and a hybrid
/// [`TrainerProgressHandle`] (`AtomicTrainerProgress` seqlock + coarse callback).
pub fn solve_stage_b_backbone_with_progress(
    manifest: &DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_frames_hint: usize,
    config: StageBSolveConfig,
    progress: &TrainerProgressHandle,
) -> Result<StageBOutputs, DnamError> {
    solve_stage_b_backbone_with_pass1(
        manifest,
        dnam_bin,
        dry_lr,
        wet_lr,
        StageBBackboneSpec {
            active_frames_hint,
            config,
            precomputed_pass1: None,
        },
        progress,
    )
}

/// Variant of [`solve_stage_b_backbone_with_progress`] accepting [`StageBBackboneSpec`] (with an
/// optional Stage A [`Pass1ProbeResult`]) so Stage B never repeats Pass 1 physics/pole probing.
pub fn solve_stage_b_backbone_with_pass1(
    manifest: &DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    spec: StageBBackboneSpec,
    progress: &TrainerProgressHandle,
) -> Result<StageBOutputs, DnamError> {
    let StageBBackboneSpec {
        active_frames_hint,
        config,
        precomputed_pass1,
    } = spec;
    validate_ultra_dnam_bin(dnam_bin)?;
    let timer = now_ms_token();
    let mono_out = manifest.base.io_config.output_mode == "mono";
    let dual_in = manifest.base.io_config.inputs >= 2;
    let avail = dry_lr.len().min(wet_lr.len());
    let active_samples = if active_frames_hint == 0 {
        avail
    } else {
        active_frames_hint.min(avail)
    };
    let eval_len =
        crate::solver::resolve_eval_active_len(manifest, dry_lr, Some(wet_lr), active_samples);

    let runner = TrainRunner::auto_detect();
    let warmup = canonical_besr_warmup(active_samples).min(active_samples.saturating_sub(1));

    if active_samples > warmup + 1 {
        let (in_l, in_r, tgt_l, tgt_r) =
            build_stereo_buffers(dry_lr, wet_lr, mono_out, dual_in, active_samples);
        let anchor = if progress.is_active() {
            let wet_l_f32: Vec<f32> = wet_lr[..active_samples].iter().map(|lr| lr[0]).collect();
            VisualProbeAnchor::from_stereo_probe(&wet_l_f32, warmup, 48_000.0)
        } else {
            None
        };

        if config.preserve_existing_backbone && has_populated_v3_backbone(dnam_bin) {
            let init_esr = eval_all_four_tier_esr_db(
                dnam_bin,
                mono_out,
                dual_in,
                &dry_lr[..eval_len],
                &wet_lr[..eval_len],
            )?;
            if !init_esr.iter().all(|e| e.is_finite()) || init_esr[3] > -70.0 {
                let mut existing = decode_dnam_bin_to_model(dnam_bin).map_err(map_stage_err)?;
                if existing.active_fir_taps == 0 {
                    existing.active_fir_taps = Tier::Ultra.early_fir_taps();
                }
                let stem: WeightedReadoutStem<'_> = (&in_l, &in_r, &tgt_l, &tgt_r, 1.0);
                patch_model_readouts_weighted_cholesky(&runner, &mut existing, &[stem], warmup)
                    .map_err(map_stage_err)?;
                *dnam_bin =
                    encode_with_preserved_controls(&mut existing, dnam_bin, mono_out, dual_in)?;
                verify_dnam_bin_zero_dc(dnam_bin)?;
            }
        } else {
            let mut model = if let Some(p1) = precomputed_pass1 {
                p1.model
            } else {
                run_pass1_probe(&in_l, &in_r, &tgt_l, &tgt_r, 48_000.0, 128, 32)
                    .map_err(map_stage_err)?
                    .1
            };
            emit_stage_b_milestone(progress, &model, dry_lr, &anchor, 250, 1);

            let pairs = StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            };
            let pass2_cfg = select_pass2_config(config.quality_mode, warmup);
            run_pass2_optimize(&runner, &mut model, &pairs, pass2_cfg, None)
                .map_err(map_stage_err)?;
            run_pass3_varpro_readout(&runner, &mut model, &in_l, &in_r, &tgt_l, &tgt_r, warmup)
                .map_err(map_stage_err)?;
            emit_stage_b_milestone(progress, &model, dry_lr, &anchor, 900, 2);

            *dnam_bin = encode_with_preserved_controls(&mut model, dnam_bin, mono_out, dual_in)?;
            verify_dnam_bin_zero_dc(dnam_bin)?;
        }
    }

    let prefix_esr_db = eval_all_four_tier_esr_db(
        dnam_bin,
        mono_out,
        dual_in,
        &dry_lr[..eval_len],
        &wet_lr[..eval_len],
    )?;
    if progress.is_active() {
        progress.emit(&TrainerProgressSnapshot {
            status: TrainingStatus::Running,
            phase: TrainingPhase::StageBBackbone,
            active_tier: Some(Tier::Ultra),
            progress_permille: 1_000,
            tier_besr_db: prefix_esr_db,
            ..Default::default()
        });
    }

    let (packed_gram_256, cross_256) = if active_samples > warmup + 1 {
        let model = decode_dnam_bin_to_model(dnam_bin).map_err(map_stage_err)?;
        let (in_l, in_r, tgt_l, tgt_r) =
            build_stereo_buffers(dry_lr, wet_lr, mono_out, dual_in, active_samples);
        extract_stat_256d_gram(
            &runner,
            &model,
            &StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            warmup,
            config.gram_frame_stride.max(1),
        )
        .map_err(map_stage_err)?
    } else {
        (vec![0.0; uplo_u_packed_len(256)], [[0.0; 2]; 256])
    };
    let stat_bytes = serialize_v3_stat_chunk_ultra(&packed_gram_256, &cross_256);
    let _ = verify_proj_stat_uplo_u(&stat_bytes)?;

    Ok(StageBOutputs {
        elapsed_ms: elapsed_ms_since(timer),
        stat_bytes,
        packed_gram_256,
        cross_256,
        prefix_esr_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;
    use dnam_train::{Pass5EmitConfig, encode_model_to_dnam_bin};
    use std::f64::consts::TAU;

    #[test]
    fn test_stage_b_backbone_fast_solve_preserve_and_stat_roundtrip() {
        let mut small_bin = vec![0u8; 256];
        let manifest = DnamProjManifest::default_reference();
        assert!(solve_stage_b_backbone(&manifest, &mut small_bin, &[], &[], 0, 1, false).is_err());

        let mut dnam_bin = initialize_backbone_dnam_bin_if_needed(None).expect("neutral dnam-bin");
        let centers = dnam_train::losses::canonical_ltas_33_centers_hz();
        let n = 4_096usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let mut sum = 0.0_f64;
                for (k, &hz) in centers.iter().enumerate() {
                    sum += (TAU * hz * t + 0.37 * (k as f64)).sin();
                }
                let s = (0.045 * sum) as f32;
                [s, s]
            })
            .collect();
        let wet: Vec<[f32; 2]> = dry
            .iter()
            .map(|&[l, r]| [0.75 * (2.2 * l).tanh(), 0.75 * (2.2 * r).tanh()])
            .collect();

        let out = solve_stage_b_backbone_with_progress(
            &manifest,
            &mut dnam_bin,
            &dry,
            &wet,
            n,
            StageBSolveConfig {
                gram_frame_stride: 2,
                preserve_existing_backbone: false,
                quality_mode: SolverQualityMode::ThresholdStop {
                    target_slack_milli_db: 0,
                    max_polish_iters: 4,
                },
                rep_profiles: None,
            },
            &TrainerProgressHandle::default(),
        )
        .expect("stage b fresh solve should succeed");
        assert_eq!(out.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
        assert_eq!(verify_proj_stat_uplo_u(&out.stat_bytes).unwrap(), 256);
        assert!(out.prefix_esr_db[3] < -12.0, "got {:?}", out.prefix_esr_db);

        // White-box preservation path: zeroed readouts recovered while preserving nonlinear quads.
        let mut preserved = decode_dnam_bin_to_model(&dnam_bin).expect("decode");
        preserved.core_quads[0].adaa_drive[0] = 0.65;
        let saved_q0 = preserved.core_quads[0];
        preserved.direct_s2_u_gain = [0.0; 2];
        preserved.w_ro = [[0.0; 12]; 40];
        preserved.fir_l = [0.0; 128];
        preserved.fir_r = [0.0; 128];
        preserved.tier_overrides = [None, None, None, None];
        dnam_bin =
            encode_model_to_dnam_bin(&preserved, Pass5EmitConfig::default()).expect("encode");

        let out_pres = solve_stage_b_backbone(&manifest, &mut dnam_bin, &dry, &wet, n, 2, true)
            .expect("stage b preserve solve should succeed");
        let after = decode_dnam_bin_to_model(&dnam_bin).expect("decode after");
        assert!((after.core_quads[0].adaa_drive[0] - saved_q0.adaa_drive[0]).abs() < 0.05);
        assert!(
            out_pres.prefix_esr_db[3] < -12.0,
            "got {:?}",
            out_pres.prefix_esr_db
        );
    }

    #[test]
    fn test_stage_b_stat_nan_progress_and_short_guards() {
        let mut bad_gram = vec![0.0; uplo_u_packed_len(256)];
        bad_gram[0] = f64::NAN;
        let stat = serialize_v3_stat_chunk_ultra(&bad_gram, &[[f64::INFINITY, -2.0]; 256]);
        assert_eq!(verify_proj_stat_uplo_u(&stat).unwrap(), 256);

        let dnam_bin = initialize_backbone_dnam_bin_if_needed(None).expect("neutral dnam-bin");
        assert_eq!(
            eval_tier_esr_db(&dnam_bin, Tier::Lite, true, false, &[], &[]).unwrap(),
            0.0
        );
        let z = [[0.0_f32; 2]; 128];
        let nz = [[0.25_f32; 2]; 2_048];
        assert_eq!(
            eval_tier_esr_db(&dnam_bin, Tier::Lite, true, false, &z, &z).unwrap(),
            -120.0
        );
        assert_eq!(
            eval_tier_esr_db(&dnam_bin, Tier::Lite, false, true, &z, &nz[..128]).unwrap(),
            0.0
        );

        let short_stat =
            compute_v3_stat_chunk_bytes(&dnam_bin, true, &z[..1], &z[..1], 1, 1).unwrap();
        assert_eq!(verify_proj_stat_uplo_u(&short_stat).unwrap(), 256);
        let ok_stat = compute_v3_stat_chunk_bytes(&dnam_bin, false, &nz, &nz, 2_048, 4).unwrap();
        assert_eq!(verify_proj_stat_uplo_u(&ok_stat).unwrap(), 256);

        let full_cfg = select_pass2_config(SolverQualityMode::BestUntilDiminishingReturns, 64);
        assert!(full_cfg.max_epochs >= 2);
        let manifest = DnamProjManifest::default_reference();
        let cb_calls = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let prog = TrainerProgressHandle::from_callback({
            let cb_calls = std::sync::Arc::clone(&cb_calls);
            move |_| *cb_calls.lock().unwrap() += 1
        });
        let mut bin_mut = dnam_bin;
        let out_short = solve_stage_b_backbone_with_progress(
            &manifest,
            &mut bin_mut,
            &nz[..1],
            &nz[..1],
            1,
            StageBSolveConfig::default(),
            &prog,
        )
        .unwrap();
        assert_eq!(out_short.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
        assert_eq!(*cb_calls.lock().unwrap(), 1);
        assert!(
            solve_stage_b_backbone_with_progress(
                &manifest,
                &mut bin_mut,
                &nz[..8],
                &nz[..8],
                8,
                StageBSolveConfig::default(),
                &prog,
            )
            .is_err()
        );
    }
}
