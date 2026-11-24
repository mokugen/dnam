#![forbid(unsafe_code)]
//! Non-Destructive `.dnam-proj` (`DNPJ`, `v3.0`) Capture Vault Editing (`proj::vault`):
//! `add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`,
//! `solve_captures_batch_into_proj`, and Two-Path (`Path 1` `0-ULP` Incremental vs. `Path 2`
//! Deterministic Full Re-Pool) `SONA v3.0` Solve.
//!
//! Canonical References:
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.lifecycle-states]`)
//! - `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.dispatch-solve]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.resume.kronecker-reassembly]`, `ref[train.cache.proj-vault]`)

pub mod edit;

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::Tier;
use dnam_engine::schema::DnamError;
use dnam_train::{Pass5EmitConfig, SonaModelF64, encode_model_to_dnam_bin};
use dnam_vm::bin::DnamBinView;
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{U16, U32};

use crate::CORESET_FP32_STEREO_BYTES;
pub use crate::audit::{
    recompute_vault_phase1_audit_and_clamps, recompute_vault_phase1_audit_and_clamps_with_windows,
};
use crate::solver::{
    SolverOptions, StageBBackboneSpec, StageBSolveConfig, StageDAuditOptions, StageDFinalizeInputs,
    TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus,
    decode_coreset_lr, finalize_stage_d_and_audit_with_panama_rows,
    run_stage_a_physics_probe_with_result, solve_stage_b_backbone_with_pass1, solve_stage_c_knobs,
};

use super::bundle::{VaultChunkInput, build_dnam_proj_container_from_vault_dbin};
use super::directory::{
    InpcHeader64BWire, ProjChunkKind, StmcHeader128BWire, StmcTrajectoryWindow,
    build_proj_stmc_capsule, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use super::schema::{DnamProjManifest, ExtractedPhysics, StemProvenance, StemRegistryEntry};
use super::{
    STMC_FLAG_ENABLED, STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STAR_CENTER, STMC_FLAG_STEREO,
    STMC_FLAG_WET_POLARITY_INVERTED,
};

pub use edit::{
    add_capture, add_capture_with_options, add_capture_with_progress, remove_capture,
    remove_capture_with_options, replace_capture, replace_capture_with_options,
    set_capture_enabled, set_capture_enabled_with_options, solve_captures_batch_into_proj,
    solve_captures_batch_with_options, solve_captures_batch_with_progress,
};

/// Specification of a single captured stem inserted or replaced in the `.dnam-proj` Capture Vault.
#[derive(Debug, Clone, PartialEq)]
pub struct VaultCaptureSpec {
    /// Unique stem identifier within the project.
    pub stem_id: String,
    /// True if this stem is the `StarCenter` (`s = 0`) anchor backing `STAT`.
    pub is_star_center: bool,
    /// Initial enable state (`true` = active in solve & Phase-1 audit).
    pub enabled: bool,
    /// Input pad linear gain applied during capture (`1.0` default).
    pub pad_gain_f32: f32,
    /// Sub-sample alignment lag in samples at `48 kHz`.
    pub fractional_lag_samples_f32: f32,
    /// Active physical control coordinates (`control_id -> f64 | String`).
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    /// Dry/wet PCM SHA-256 provenance and frame count.
    pub provenance: StemProvenance,
    /// Creator stem notes/tags (excluded from immutable JCS digest).
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Optional white-box / blind extracted physical descriptors.
    pub extracted_physics: Option<ExtractedPhysics>,
    /// `1..=36` trajectory windows (`R = 1` for stationary stops, `R > 1` for Mode A sweeps).
    pub windows: Vec<StmcTrajectoryWindow>,
    /// `88 B` `uplo='U'` band-energy PSD summary.
    pub band_psd_summary_88: [u8; 88],
    /// Disjoint-comb carrier count recorded in `STMC`.
    pub n_comb_bins: u32,
    /// `4,096 B` spectral, sideband, and ballistic summary block.
    pub spectral_summary_4096: [u8; 4_096],
    /// Full-bandwidth `fp32` (`<f4`) wet coreset bytes (`384,000 B` mono or `768,000 B` stereo at `2.0 s`).
    pub wet_coreset_fp32: Vec<u8>,
    /// Optional raw `SPCM` chunk (`b"SPCM"` header + aligned PCM samples).
    pub optional_spcm: Option<Vec<u8>>,
}

impl VaultCaptureSpec {
    pub(super) fn to_stmc_bytes(&self, stem_index: u16) -> Result<Vec<u8>, DnamError> {
        let mut flags = 0u16;
        if self.enabled {
            flags |= STMC_FLAG_ENABLED;
        }
        if self.is_star_center {
            flags |= STMC_FLAG_STAR_CENTER;
        }
        if self.windows.len() > 1 {
            flags |= STMC_FLAG_MODE_A_SWEEP;
        }
        let is_stereo = self.wet_coreset_fp32.len() >= CORESET_FP32_STEREO_BYTES
            || (self.provenance.num_frames > 0
                && self.wet_coreset_fp32.len() == (self.provenance.num_frames as usize) * 8);
        if is_stereo {
            flags |= STMC_FLAG_STEREO;
        }
        if self
            .user_metadata
            .get("wet_polarity_inverted")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            flags |= STMC_FLAG_WET_POLARITY_INVERTED;
        }
        build_proj_stmc_capsule(
            stem_index,
            flags,
            self.pad_gain_f32,
            self.fractional_lag_samples_f32,
            &self.band_psd_summary_88,
            self.n_comb_bins,
            &self.windows,
            &self.spectral_summary_4096,
            &self.wet_coreset_fp32,
        )
    }
}

/// Initializes a canonical 4-tier `SONA v3.0` `.dnam-bin` (`Ultra = 30,720 B` arena) from
/// `existing_dbin` or resting neutral identity.
pub(crate) fn initialize_backbone_dnam_bin_if_needed(
    existing_dbin: Option<&[u8]>,
) -> Result<Vec<u8>, DnamError> {
    if let Some(bytes) = existing_dbin
        && let Ok(view) = DnamBinView::parse(bytes)
        && view.retained_tier == Tier::Ultra
    {
        return Ok(bytes.to_vec());
    }
    let model = SonaModelF64::new(48_000.0);
    encode_model_to_dnam_bin(&model, Pass5EmitConfig::default())
        .map_err(|e| DnamError::SchemaValidationError(format!("{e}")))
}

pub(super) type ChunkDigest = (u32, [u8; 32]);

pub(super) struct VaultState {
    pub manifest: DnamProjManifest,
    pub calb: Option<(Vec<u8>, Option<ChunkDigest>)>,
    pub inpc: Option<(Vec<u8>, Option<ChunkDigest>)>,
    pub spcm_by_id: BTreeMap<u32, (Vec<u8>, Option<ChunkDigest>)>,
    pub stmc_by_id: BTreeMap<u32, Vec<u8>>,
    pub stmc_digests: BTreeMap<u32, ChunkDigest>,
    pub stat: Option<(Vec<u8>, Option<ChunkDigest>)>,
    pub wght_dnam_bin: Option<Vec<u8>>,
    pub compacted: bool,
}

pub(super) fn parse_validated_stmc_windows(
    stmc_buf: &[u8],
) -> Result<Vec<StmcTrajectoryWindow>, DnamError> {
    if stmc_buf.len() < 704 {
        return Ok(verify_proj_stmc_capsule(stmc_buf)?.windows);
    }
    let Ok((hdr, _)) = StmcHeader128BWire::ref_from_prefix(stmc_buf) else {
        return Ok(verify_proj_stmc_capsule(stmc_buf)?.windows);
    };
    let n_windows = usize::from(hdr.n_windows.get());
    let mut windows = Vec::with_capacity(n_windows);
    for idx in 0..n_windows {
        let off = 128 + idx * 16;
        windows.push(StmcTrajectoryWindow::from_bytes(&stmc_buf[off..off + 16])?);
    }
    Ok(windows)
}

pub(super) fn find_stem_pos(
    manifest: &DnamProjManifest,
    stem_id: &str,
) -> Result<usize, DnamError> {
    manifest
        .stems_registry
        .iter()
        .position(|s| s.stem_id == stem_id)
        .ok_or_else(|| {
            DnamError::SchemaValidationError(format!(
                "stem_id '{stem_id}' not found in stems_registry"
            ))
        })
}

pub(super) fn promote_first_enabled_star(stems: &mut [StemRegistryEntry]) {
    for entry in stems.iter_mut() {
        entry.is_star_center = false;
    }
    if let Some(first_enabled) = stems.iter_mut().find(|s| s.enabled) {
        first_enabled.is_star_center = true;
    }
}

fn audit_clamp_count(manifest: &DnamProjManifest) -> u16 {
    manifest.audit_report.as_ref().map_or(0, |a| {
        (a.phase1_clamps.len() + a.phase3_clamps.len()) as u16
    })
}

fn emit_vault_phase(
    progress: &TrainerProgressHandle,
    phase: TrainingPhase,
    permille: u16,
    clamped_controls: u16,
    tier_besr_db: [f32; 4],
) {
    if !progress.is_active() {
        return;
    }
    progress.emit(&TrainerProgressSnapshot {
        status: if phase == TrainingPhase::Completed {
            TrainingStatus::Completed
        } else {
            TrainingStatus::Running
        },
        phase,
        clamped_controls,
        progress_permille: permille,
        tier_besr_db,
        ..Default::default()
    });
}

pub(super) fn rebuild_vault_container(
    state: VaultState,
    inpc_entry: (Vec<u8>, Option<ChunkDigest>),
    force_full_repool: bool,
    touched_controls: Option<BTreeSet<u16>>,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    rebuild_vault_container_with_progress(
        state,
        inpc_entry,
        force_full_repool,
        touched_controls,
        options,
        &TrainerProgressHandle::default(),
    )
}

pub(super) fn rebuild_vault_container_with_progress(
    mut state: VaultState,
    inpc_entry: (Vec<u8>, Option<ChunkDigest>),
    force_full_repool: bool,
    touched_controls: Option<BTreeSet<u16>>,
    options: SolverOptions,
    progress: &TrainerProgressHandle,
) -> Result<Vec<u8>, DnamError> {
    let (inpc_bytes, inpc_digest) = inpc_entry;
    let inpc_channels = if inpc_digest.is_some()
        && let Ok((hdr, _)) = InpcHeader64BWire::ref_from_prefix(&inpc_bytes)
        && inpc_bytes.len() >= 640
    {
        usize::from(hdr.channels.get())
    } else {
        usize::from(verify_proj_inpc_capsule(&inpc_bytes)?.channels)
    };

    let mut enabled_windows = BTreeMap::new();
    for stem in &state.manifest.stems_registry {
        if let Some(stmc_buf) = state.stmc_by_id.get_mut(&stem.chunk_id) {
            if stmc_buf.len() < 4_800 {
                let _ = verify_proj_stmc_capsule(stmc_buf)?;
            }
            let (hdr, _) =
                StmcHeader128BWire::mut_from_prefix(stmc_buf.as_mut_slice()).map_err(|_| {
                    DnamError::SchemaValidationError("truncated STMC header".to_string())
                })?;
            let cur_flags = hdr.flags.get();
            let mut new_flags = cur_flags;
            if stem.enabled {
                new_flags |= STMC_FLAG_ENABLED;
            } else {
                new_flags &= !STMC_FLAG_ENABLED;
            }
            if stem.is_star_center {
                new_flags |= STMC_FLAG_STAR_CENTER;
            } else {
                new_flags &= !STMC_FLAG_STAR_CENTER;
            }
            if new_flags != cur_flags {
                hdr.flags = U16::new(new_flags);
                state.stmc_digests.remove(&stem.chunk_id);
            }
            if stem.enabled {
                enabled_windows.insert(stem.chunk_id, parse_validated_stmc_windows(stmc_buf)?);
            }
        }
    }
    recompute_vault_phase1_audit_and_clamps_with_windows(&mut state.manifest, &enabled_windows);

    let has_prior_wght = state.wght_dnam_bin.is_some();
    let mut dnam_bin = initialize_backbone_dnam_bin_if_needed(state.wght_dnam_bin.as_deref())?;
    let _ = state.manifest.base.compile_clamp_descriptors()?;

    let active_star = state
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .cloned();

    let need_path2 =
        force_full_repool || !has_prior_wght || (active_star.is_some() != state.stat.is_some());

    let dry_lr = decode_coreset_lr(&inpc_bytes[640..], inpc_channels);
    let hint_raw = inpc_bytes
        .get(328..332)
        .and_then(|s| U32::ref_from_bytes(s).ok())
        .map_or(0_usize, |v| v.get() as usize);
    let active_frames = if hint_raw > 0 && hint_raw <= dry_lr.len() {
        hint_raw
    } else {
        dry_lr.len()
    };

    let mut stat_out = state.stat;
    let mut star_wet_opt: Option<Vec<[f32; 2]>> = None;
    let mut fallback_esr: Option<[f32; 4]> = state
        .manifest
        .solver_telemetry
        .as_ref()
        .map(|s| s.tier_esr_db);
    let mut accum_solve_ms = 0.0_f32;
    let _ = touched_controls;

    if need_path2 && options.clear_dynamic_w_ctrl {
        crate::solver::stage_c_knobs::clear_dnam_bin_control_bus_state(&mut dnam_bin)?;
    }

    if need_path2 {
        if let Some(ref star_entry) = active_star
            && let Some(star_stmc_bytes) = state.stmc_by_id.get(&star_entry.chunk_id)
        {
            let star_flags = StmcHeader128BWire::ref_from_prefix(star_stmc_bytes)
                .map_or(0, |(h, _)| h.flags.get());
            let wet_lr = decode_coreset_lr(
                &star_stmc_bytes[4_800..],
                if (star_flags & STMC_FLAG_STEREO) != 0 {
                    2
                } else {
                    1
                },
            );
            let spec_arr: Option<&[u8; 4_096]> = star_stmc_bytes[704..4_800].try_into().ok();
            let eval_n = active_frames.min(wet_lr.len());
            let clamped = audit_clamp_count(&state.manifest);
            emit_vault_phase(
                progress,
                TrainingPhase::StageAPhysicsProbe,
                50,
                clamped,
                [f32::NAN; 4],
            );

            // Stage A: Physics Probe & Initial SONA Seeding (passed directly to Stage B)
            let (stage_a_rep, pass1_res) = run_stage_a_physics_probe_with_result(
                &state.manifest,
                &mut dnam_bin,
                &dry_lr[..eval_n],
                &wet_lr[..eval_n],
                spec_arr,
                options.stage_a,
            );
            accum_solve_ms += stage_a_rep.elapsed_ms;
            if let Some(warn) = stage_a_rep.phase2_rail_warning {
                let ar = state
                    .manifest
                    .audit_report
                    .get_or_insert_with(Default::default);
                if !ar.phase2_rail_warnings.contains(&warn) {
                    ar.phase2_rail_warnings.push(warn);
                }
            }

            // Stage B: Joint 4-Tier SONA Backbone & 256-D uplo='U' STAT Contraction
            let stage_b = solve_stage_b_backbone_with_pass1(
                &state.manifest,
                &mut dnam_bin,
                &dry_lr,
                &wet_lr,
                StageBBackboneSpec {
                    active_frames_hint: active_frames,
                    config: StageBSolveConfig {
                        gram_frame_stride: options.gram_frame_stride,
                        preserve_existing_backbone: stage_a_rep.preserved_backbone,
                        quality_mode: options.quality_mode,
                        rep_profiles: options.rep_profiles,
                    },
                    precomputed_pass1: pass1_res,
                },
                &progress.subrange(120, 760),
            )?;
            accum_solve_ms += stage_b.elapsed_ms;
            stat_out = Some((stage_b.stat_bytes, None));
            fallback_esr = Some(stage_b.prefix_esr_db);
            star_wet_opt = Some(wet_lr);
        } else {
            stat_out = None;
            if !options.stage_a.preserve_existing_backbone {
                dnam_bin = initialize_backbone_dnam_bin_if_needed(None)?;
            }
        }
    }

    // Stage C: Multi-Knob & Switch Conditioning (`CP1` + `CEXT` + `TSYN` + `W_ctrl` + `U_int`)
    emit_vault_phase(
        progress,
        TrainingPhase::StageCKnobs,
        if need_path2 { 780 } else { 250 },
        audit_clamp_count(&state.manifest),
        fallback_esr.unwrap_or([f32::NAN; 4]),
    );
    let stage_c = solve_stage_c_knobs(
        &state.manifest,
        &mut dnam_bin,
        &state.stmc_by_id,
        &dry_lr,
        active_frames,
        need_path2 && options.clear_dynamic_w_ctrl,
    )?;
    accum_solve_ms += stage_c.elapsed_ms;

    // Stage D: 3-Phase Audit, LUFS Loudness Trims (on Path 2), and `[STAB-ZERO-DC]` Certification
    emit_vault_phase(
        progress,
        TrainingPhase::StageDAudit,
        if need_path2 { 900 } else { 750 },
        audit_clamp_count(&state.manifest),
        fallback_esr.unwrap_or([f32::NAN; 4]),
    );
    let audit_opts = if need_path2 {
        StageDAuditOptions::PATH2_FULL
    } else {
        StageDAuditOptions::PATH1_HEAD_ONLY
    };
    let active_end = active_frames.min(dry_lr.len());
    let eval_end = star_wet_opt
        .as_deref()
        .map_or(active_end, |w| active_end.min(w.len()));
    let _stage_d = finalize_stage_d_and_audit_with_panama_rows(
        &mut state.manifest,
        &mut dnam_bin,
        StageDFinalizeInputs {
            dry_lr: &dry_lr[..eval_end],
            star_wet_lr: star_wet_opt.as_deref().map(|w| &w[..eval_end]),
            fallback_esr,
            prior_solve_ms: accum_solve_ms,
            audit_opts,
            panama_sample_rows: &stage_c.panama_sample_rows,
        },
    )?;

    let mut chunks = Vec::new();
    if let Some((calb, calb_dig)) = state.calb {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb,
            cached_digest: calb_dig,
        });
    }
    for (&cid, (spcm, spcm_dig)) in &state.spcm_by_id {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Spcm,
            chunk_id: cid,
            payload: spcm.clone(),
            cached_digest: *spcm_dig,
        });
    }
    if !state.stmc_by_id.is_empty() {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Inpc,
            chunk_id: 0,
            payload: inpc_bytes,
            cached_digest: inpc_digest,
        });
        for stem in &state.manifest.stems_registry {
            if let Some(stmc) = state.stmc_by_id.get(&stem.chunk_id) {
                chunks.push(VaultChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: stem.chunk_id,
                    payload: stmc.clone(),
                    cached_digest: state.stmc_digests.get(&stem.chunk_id).copied(),
                });
            }
        }
    }
    if let Some((stat, stat_dig)) = stat_out {
        chunks.push(VaultChunkInput {
            kind: ProjChunkKind::Stat,
            chunk_id: 0,
            payload: stat,
            cached_digest: stat_dig,
        });
    }

    let keep_compacted =
        state.compacted && state.spcm_by_id.is_empty() && !state.stmc_by_id.is_empty();
    let built = build_dnam_proj_container_from_vault_dbin(
        &mut state.manifest,
        chunks,
        &dnam_bin,
        keep_compacted,
    )?;
    let final_esr = state
        .manifest
        .solver_telemetry
        .as_ref()
        .map_or(fallback_esr.unwrap_or([f32::NAN; 4]), |s| s.tier_esr_db);
    emit_vault_phase(
        progress,
        TrainingPhase::Completed,
        1_000,
        audit_clamp_count(&state.manifest),
        final_esr,
    );
    Ok(built)
}
