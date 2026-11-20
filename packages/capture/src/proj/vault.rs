#![forbid(unsafe_code)]
//! Non-Destructive `.dnam-proj` (`DNPJ`) Capture Vault Editing (`proj::vault`):
//! `add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`,
//! `solve_captures_batch_into_proj`, and Two-Path (`Path 1` `0-ULP` Incremental vs. `Path 2`
//! Deterministic Full Re-Pool) `v3` 4-Stage Compositional Solve.
//!
//! Canonical References:
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.lifecycle-states]`)
//! - `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.dispatch-solve]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.resume.kronecker-reassembly]`, `ref[train.cache.proj-vault]`)

pub mod edit;

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::DnamError;
use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::weights::{ARENA_ULTRA_BYTES, pack_v3_matryoshka_to_arena};

use crate::CORESET_FP32_STEREO_BYTES;
pub use crate::audit::{
    recompute_vault_phase1_audit_and_clamps, recompute_vault_phase1_audit_and_clamps_with_windows,
};
use crate::solver::{
    SolverOptions, StageBSolveConfig, StageDAuditOptions, TrainerProgressHandle,
    TrainerProgressSnapshot, TrainingPhase, TrainingStatus, decode_coreset_lr,
    finalize_stage_d_and_audit, run_stage_a_physics_probe, solve_stage_b_backbone_with_progress,
    solve_stage_c_knobs,
};

use super::bundle::{
    VaultChunkInput, build_dnam_proj_container_from_vault, zero_arena_b_head_dc_bias,
};
use super::directory::{
    ProjChunkKind, StmcTrajectoryWindow, build_proj_stmc_capsule, verify_proj_inpc_capsule,
    verify_proj_stmc_capsule,
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

/// Initializes a canonical `28,672 B` `v3` Matryoshka arena from `existing_arena` or resting identity.
pub(crate) fn initialize_backbone_arena_if_needed(existing_arena: Option<&[u8]>) -> Vec<u8> {
    if let Some(bytes) = existing_arena
        && bytes.len() == ARENA_ULTRA_BYTES
        && bytes.iter().any(|&b| b != 0)
    {
        return bytes.to_vec();
    }
    let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
    if let Some(bytes) = existing_arena
        && bytes.iter().any(|&b| b != 0)
    {
        let copy_len = bytes.len().min(arena.len());
        arena[..copy_len].copy_from_slice(&bytes[..copy_len]);
    } else {
        let model = V3MatryoshkaModel::identity(true, false);
        pack_v3_matryoshka_to_arena(&model, &mut arena);
    }
    zero_arena_b_head_dc_bias(&mut arena);
    arena
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
    pub wght_arena: Option<Vec<u8>>,
    pub compacted: bool,
}

pub(super) fn parse_validated_stmc_windows(
    stmc_buf: &[u8],
) -> Result<Vec<StmcTrajectoryWindow>, DnamError> {
    if stmc_buf.len() < 704 {
        return Ok(verify_proj_stmc_capsule(stmc_buf)?.windows);
    }
    let n_windows = usize::from(u16::from_le_bytes([stmc_buf[10], stmc_buf[11]]));
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
    let inpc_channels = if inpc_digest.is_some() && inpc_bytes.len() >= 640 {
        usize::from(u16::from_le_bytes([inpc_bytes[6], inpc_bytes[7]]))
    } else {
        usize::from(verify_proj_inpc_capsule(&inpc_bytes)?.channels)
    };

    let mut enabled_windows = BTreeMap::new();
    for stem in &state.manifest.stems_registry {
        if let Some(stmc_buf) = state.stmc_by_id.get_mut(&stem.chunk_id) {
            if stmc_buf.len() < 4_800 {
                let _ = verify_proj_stmc_capsule(stmc_buf)?;
            }
            let cur_flags = u16::from_le_bytes([stmc_buf[8], stmc_buf[9]]);
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
                stmc_buf[8..10].copy_from_slice(&new_flags.to_le_bytes());
                state.stmc_digests.remove(&stem.chunk_id);
            }
            if stem.enabled {
                enabled_windows.insert(stem.chunk_id, parse_validated_stmc_windows(stmc_buf)?);
            }
        }
    }
    recompute_vault_phase1_audit_and_clamps_with_windows(&mut state.manifest, &enabled_windows);

    let has_prior_wght = state.wght_arena.is_some();
    let mut arena_28k = initialize_backbone_arena_if_needed(state.wght_arena.as_deref());
    let _ = crate::solver::stages::sync_clamp_descriptors_to_arena(
        &state.manifest.base,
        &mut arena_28k,
    )?;

    let active_star = state
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .cloned();

    let need_path2 =
        force_full_repool || !has_prior_wght || (active_star.is_some() != state.stat.is_some());

    let dry_lr = decode_coreset_lr(&inpc_bytes[640..], inpc_channels);
    let hint_raw = if inpc_bytes.len() >= 332 {
        u32::from_le_bytes(inpc_bytes[328..332].try_into().unwrap()) as usize
    } else {
        0
    };
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

    if need_path2 {
        if let Some(ref star_entry) = active_star
            && let Some(star_stmc_bytes) = state.stmc_by_id.get(&star_entry.chunk_id)
        {
            let star_flags = u16::from_le_bytes([star_stmc_bytes[8], star_stmc_bytes[9]]);
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

            // Stage A: Physics Probe & Physical Prior Lifting
            let stage_a_rep = run_stage_a_physics_probe(
                &state.manifest,
                &mut arena_28k,
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

            // Stage B: Joint 4-Tier Matryoshka Backbone & 256-D uplo='U' STAT Contraction
            let stage_b = solve_stage_b_backbone_with_progress(
                &state.manifest,
                &mut arena_28k,
                &dry_lr,
                &wet_lr,
                active_frames,
                StageBSolveConfig {
                    gram_frame_stride: options.gram_frame_stride,
                    preserve_existing_backbone: stage_a_rep.preserved_backbone,
                    quality_mode: options.quality_mode,
                    rep_profiles: options.rep_profiles,
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
                arena_28k = initialize_backbone_arena_if_needed(None);
            } else {
                zero_arena_b_head_dc_bias(&mut arena_28k);
            }
        }
    }

    // Stage C: Parallel Multi-Knob & Switch Conditioning (`CP1` + `U_int` + `W_ctrl`)
    emit_vault_phase(
        progress,
        TrainingPhase::StageCKnobs,
        if need_path2 { 780 } else { 250 },
        audit_clamp_count(&state.manifest),
        fallback_esr.unwrap_or([f32::NAN; 4]),
    );
    let stage_c = solve_stage_c_knobs(
        &state.manifest,
        &mut arena_28k,
        &state.stmc_by_id,
        &dry_lr,
        active_frames,
        need_path2 && options.clear_dynamic_w_ctrl,
    )?;
    accum_solve_ms += stage_c.elapsed_ms;

    // Stage D: 3-Phase Audit, LUFS Loudness Trims (on Path 2), and `[STAB-ZERO-DC]` Packing
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
    let _stage_d = finalize_stage_d_and_audit(
        &mut state.manifest,
        &mut arena_28k,
        &dry_lr[..eval_end],
        star_wet_opt.as_deref().map(|w| &w[..eval_end]),
        fallback_esr,
        accum_solve_ms,
        audit_opts,
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
    let built = build_dnam_proj_container_from_vault(
        &mut state.manifest,
        chunks,
        &arena_28k,
        &stage_c.cp1_slots,
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
