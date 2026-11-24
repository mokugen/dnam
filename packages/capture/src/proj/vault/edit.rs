#![forbid(unsafe_code)]
//! Capture Vault CRUD Operations (`add_capture`, `solve_captures_batch`, `remove_capture`,
//! `replace_capture`, `set_capture_enabled`) implementing Two-Path Incremental Vault Execution.

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::container::extract_dnam_bin;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;

use crate::audit::manifest::recompute_vault_phase1_audit_and_clamps;
use crate::proj::DnamProjView;
use crate::proj::PROJ_FLAG_COMPACTED_CAPSULES;
use crate::proj::bundle::{VaultChunkInput, build_dnam_proj_container_from_vault_dbin};
use crate::proj::capsules::verify_proj_stmc_capsule;
use crate::proj::directory::{ProjChunkKind, StmcTrajectoryWindow};
use crate::proj::schema::{DnamProjManifest, StemRegistryEntry};
use crate::solver::{SolverOptions, TrainerProgressHandle};

use super::{
    VaultCaptureSpec, VaultState, find_stem_pos, initialize_backbone_dnam_bin_if_needed,
    parse_validated_stmc_windows, promote_first_enabled_star, rebuild_vault_container,
    rebuild_vault_container_with_progress,
};

fn extract_vault_state(proj_bytes: &[u8]) -> Result<VaultState, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let (view, wght_pair_opt) = DnamProjView::from_bytes_with_wght_view(proj_bytes, &crypto, true)?;
    let compacted = (view.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES) != 0;
    let mut calb = None;
    let mut inpc = None;
    let mut spcm_by_id = BTreeMap::new();
    let mut stmc_by_id = BTreeMap::new();
    let mut stmc_digests = BTreeMap::new();
    let mut stat = None;
    let wght_dnam_bin = wght_pair_opt
        .and_then(|(wght_bytes, _)| extract_dnam_bin(wght_bytes).ok().map(|c| c.to_owned()));

    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let dig = Some((entry.crc32c, entry.sha256));
        match entry.kind()? {
            ProjChunkKind::Calb => calb = Some((slice.to_vec(), dig)),
            ProjChunkKind::Inpc => inpc = Some((slice.to_vec(), dig)),
            ProjChunkKind::Spcm => {
                spcm_by_id.insert(entry.chunk_id, (slice.to_vec(), dig));
            }
            ProjChunkKind::Stmc => {
                stmc_by_id.insert(entry.chunk_id, slice.to_vec());
                stmc_digests.insert(entry.chunk_id, (entry.crc32c, entry.sha256));
            }
            ProjChunkKind::Stat => stat = Some((slice.to_vec(), dig)),
            ProjChunkKind::Wght | ProjChunkKind::Auth => {}
        }
    }

    Ok(VaultState {
        manifest: view.manifest,
        calb,
        inpc,
        spcm_by_id,
        stmc_by_id,
        stmc_digests,
        stat,
        wght_dnam_bin,
        compacted,
    })
}

fn collect_touched_control_indices(
    manifest: &DnamProjManifest,
    controls_meta: &BTreeMap<String, serde_json::Value>,
    windows: &[StmcTrajectoryWindow],
) -> BTreeSet<u16> {
    let mut out = BTreeSet::new();
    for w in windows {
        out.insert(w.control_id);
    }
    for (idx, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        if controls_meta.contains_key(&ctrl.id) {
            out.insert(idx as u16);
        }
    }
    out
}

fn push_capture_into_vault_state(
    state: &mut VaultState,
    capture: VaultCaptureSpec,
) -> Result<(bool, BTreeSet<u16>), DnamError> {
    if state
        .manifest
        .stems_registry
        .iter()
        .any(|s| s.stem_id == capture.stem_id)
    {
        return Err(DnamError::SchemaValidationError(format!(
            "stem_id '{}' already exists in stems_registry",
            capture.stem_id
        )));
    }
    let next_chunk_id = state
        .manifest
        .stems_registry
        .iter()
        .map(|s| s.chunk_id)
        .max()
        .map_or(Ok(0_u32), |m| {
            m.checked_add(1).ok_or_else(|| {
                DnamError::SchemaValidationError("stems_registry chunk_id overflow".to_string())
            })
        })?;
    let stem_idx = u16::try_from(next_chunk_id).map_err(|_| {
        DnamError::SchemaValidationError("chunk_id exceeds u16 STMC stem_index".to_string())
    })?;
    let is_new_star = capture.is_star_center && capture.enabled;
    if is_new_star {
        for prev in &mut state.manifest.stems_registry {
            prev.is_star_center = false;
        }
    }
    let stmc_bytes = capture.to_stmc_bytes(stem_idx)?;
    let touched =
        collect_touched_control_indices(&state.manifest, &capture.controls_meta, &capture.windows);

    if let Some(spcm) = capture.optional_spcm {
        if spcm.len() < 64 || spcm[0..4] != *b"SPCM" {
            return Err(DnamError::SchemaValidationError(
                "invalid SPCM header".to_string(),
            ));
        }
        state.spcm_by_id.insert(next_chunk_id, (spcm, None));
    }
    state.stmc_digests.remove(&next_chunk_id);
    state.stmc_by_id.insert(next_chunk_id, stmc_bytes);
    state.manifest.stems_registry.push(StemRegistryEntry {
        stem_id: capture.stem_id,
        chunk_id: next_chunk_id,
        enabled: capture.enabled,
        is_star_center: capture.is_star_center,
        controls_meta: capture.controls_meta,
        provenance: capture.provenance,
        user_metadata: capture.user_metadata,
        extracted_physics: capture.extracted_physics,
    });
    Ok((is_new_star, touched))
}

/// Inserts a new capture into the `.dnam-proj` Capture Vault (`add_capture`), executing Path 1
/// (incremental fast-path, `0.000000 ULP` on `StarCenter`) when `StarCenter` is unchanged or Path 2
/// (deterministic full `STAT` + backbone re-pool) when adding a new `StarCenter` (`r[train.cache.proj-vault]`).
pub fn add_capture(
    proj_bytes: &[u8],
    capture: VaultCaptureSpec,
    inpc_if_missing: Option<&[u8]>,
) -> Result<Vec<u8>, DnamError> {
    add_capture_with_options(
        proj_bytes,
        capture,
        inpc_if_missing,
        SolverOptions::default(),
    )
}

/// Variant of [`add_capture`] accepting explicit [`SolverOptions`] (`SolverOptions::preserve_backbone()`).
pub fn add_capture_with_options(
    proj_bytes: &[u8],
    capture: VaultCaptureSpec,
    inpc_if_missing: Option<&[u8]>,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    add_capture_with_progress(
        proj_bytes,
        capture,
        inpc_if_missing,
        options,
        &TrainerProgressHandle::default(),
    )
}

/// Variant of [`add_capture_with_options`] emitting coarse progress into [`TrainerProgressHandle`].
pub fn add_capture_with_progress(
    proj_bytes: &[u8],
    capture: VaultCaptureSpec,
    inpc_if_missing: Option<&[u8]>,
    options: SolverOptions,
    progress: &TrainerProgressHandle,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let inpc_entry = match (state.inpc.take(), inpc_if_missing) {
        (Some(existing), _) => existing,
        (None, Some(provided)) => (provided.to_vec(), None),
        (None, None) => {
            return Err(DnamError::SchemaValidationError(
                "add_capture requires an INPC dry stimulus capsule".to_string(),
            ));
        }
    };
    let (is_new_star, touched) = push_capture_into_vault_state(&mut state, capture)?;
    rebuild_vault_container_with_progress(
        state,
        inpc_entry,
        is_new_star,
        Some(touched),
        options,
        progress,
    )
}

/// Inserts an entire batch of captured stems into `.dnam-proj` and executes the Deterministic 4-Stage
/// `v3` Matryoshka Solver in a single pass (`r[train.pipeline.dispatch-solve]`).
pub fn solve_captures_batch_into_proj(
    proj_bytes: &[u8],
    captures: Vec<VaultCaptureSpec>,
    inpc_if_missing: Option<&[u8]>,
) -> Result<Vec<u8>, DnamError> {
    solve_captures_batch_with_options(
        proj_bytes,
        captures,
        inpc_if_missing,
        SolverOptions::default(),
    )
}

/// Variant of [`solve_captures_batch_into_proj`] accepting explicit [`SolverOptions`].
pub fn solve_captures_batch_with_options(
    proj_bytes: &[u8],
    captures: Vec<VaultCaptureSpec>,
    inpc_if_missing: Option<&[u8]>,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    solve_captures_batch_with_progress(
        proj_bytes,
        captures,
        inpc_if_missing,
        options,
        &TrainerProgressHandle::default(),
    )
}

/// Variant of [`solve_captures_batch_with_options`] emitting coarse progress into [`TrainerProgressHandle`].
pub fn solve_captures_batch_with_progress(
    proj_bytes: &[u8],
    captures: Vec<VaultCaptureSpec>,
    inpc_if_missing: Option<&[u8]>,
    options: SolverOptions,
    progress: &TrainerProgressHandle,
) -> Result<Vec<u8>, DnamError> {
    if captures.is_empty() {
        return Ok(proj_bytes.to_vec());
    }
    let mut state = extract_vault_state(proj_bytes)?;
    let inpc_entry = match (state.inpc.take(), inpc_if_missing) {
        (Some(existing), _) => existing,
        (None, Some(provided)) => (provided.to_vec(), None),
        (None, None) => {
            return Err(DnamError::SchemaValidationError(
                "solve_captures_batch requires an INPC dry stimulus capsule".to_string(),
            ));
        }
    };
    let mut any_new_star = false;
    let mut all_touched = BTreeSet::new();
    for cap in captures {
        let (is_star, touched) = push_capture_into_vault_state(&mut state, cap)?;
        any_new_star |= is_star;
        all_touched.extend(touched);
    }
    if !state
        .manifest
        .stems_registry
        .iter()
        .any(|s| s.enabled && s.is_star_center)
    {
        promote_first_enabled_star(&mut state.manifest.stems_registry);
        any_new_star = true;
    }
    rebuild_vault_container_with_progress(
        state,
        inpc_entry,
        any_new_star,
        Some(all_touched),
        options,
        progress,
    )
}

/// Removes `stem_id` from the `.dnam-proj` Capture Vault (`remove_capture`), automatically
/// tightening Phase-1 `[CLAMP-RANGE]` / `[CLAMP-SWITCH]` bounds (`r[train.cache.proj-vault]`).
pub fn remove_capture(proj_bytes: &[u8], stem_id: &str) -> Result<Vec<u8>, DnamError> {
    remove_capture_with_options(proj_bytes, stem_id, SolverOptions::default())
}

/// Variant of [`remove_capture`] accepting explicit [`SolverOptions`].
pub fn remove_capture_with_options(
    proj_bytes: &[u8],
    stem_id: &str,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let pos = find_stem_pos(&state.manifest, stem_id)?;
    let removed = state.manifest.stems_registry.remove(pos);
    state.spcm_by_id.remove(&removed.chunk_id);
    state.stmc_digests.remove(&removed.chunk_id);
    let windows = state
        .stmc_by_id
        .remove(&removed.chunk_id)
        .as_deref()
        .and_then(|b| parse_validated_stmc_windows(b).ok())
        .unwrap_or_default();
    let touched =
        collect_touched_control_indices(&state.manifest, &removed.controls_meta, &windows);

    let was_active_star = removed.is_star_center && removed.enabled;
    if was_active_star {
        state.stat = None;
        promote_first_enabled_star(&mut state.manifest.stems_registry);
    }

    if state.manifest.stems_registry.is_empty() {
        recompute_vault_phase1_audit_and_clamps(&mut state.manifest);
        let mut chunks = Vec::new();
        if let Some((c, c_dig)) = state.calb {
            chunks.push(VaultChunkInput {
                kind: ProjChunkKind::Calb,
                chunk_id: 0,
                payload: c,
                cached_digest: c_dig,
            });
        }
        let dnam_bin = initialize_backbone_dnam_bin_if_needed(None)?;
        return build_dnam_proj_container_from_vault_dbin(
            &mut state.manifest,
            chunks,
            &dnam_bin,
            false,
        );
    }

    let inpc_entry = state.inpc.take().ok_or_else(|| {
        DnamError::SchemaValidationError("missing INPC chunk in non-empty vault".to_string())
    })?;
    rebuild_vault_container(state, inpc_entry, was_active_star, Some(touched), options)
}

fn clear_other_stars(stems: &mut [StemRegistryEntry], keep_pos: usize) {
    for (i, entry) in stems.iter_mut().enumerate() {
        entry.is_star_center = i == keep_pos;
    }
}

/// Replaces `stem_id` in-place inside the `.dnam-proj` Capture Vault (`replace_capture`), running
/// Path 1 when `StarCenter` is unchanged or Path 2 when `StarCenter` is replaced (`r[train.cache.proj-vault]`).
pub fn replace_capture(
    proj_bytes: &[u8],
    stem_id: &str,
    replacement: VaultCaptureSpec,
) -> Result<Vec<u8>, DnamError> {
    replace_capture_with_options(proj_bytes, stem_id, replacement, SolverOptions::default())
}

/// Variant of [`replace_capture`] accepting explicit [`SolverOptions`].
pub fn replace_capture_with_options(
    proj_bytes: &[u8],
    stem_id: &str,
    replacement: VaultCaptureSpec,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let pos = find_stem_pos(&state.manifest, stem_id)?;
    if state
        .manifest
        .stems_registry
        .iter()
        .enumerate()
        .any(|(i, s)| i != pos && s.stem_id == replacement.stem_id)
    {
        return Err(DnamError::SchemaValidationError(format!(
            "stem_id '{}' already exists in stems_registry",
            replacement.stem_id
        )));
    }
    let inpc_bytes = state.inpc.take().ok_or_else(|| {
        DnamError::SchemaValidationError("missing INPC chunk in replace_capture".to_string())
    })?;
    let chunk_id = state.manifest.stems_registry[pos].chunk_id;
    let stem_idx = u16::try_from(chunk_id).map_err(|_| {
        DnamError::SchemaValidationError("chunk_id exceeds u16 STMC stem_index".to_string())
    })?;
    let old_was_star = state.manifest.stems_registry[pos].is_star_center
        && state.manifest.stems_registry[pos].enabled;
    let new_is_star = replacement.is_star_center && replacement.enabled;
    let old_windows = state
        .stmc_by_id
        .get(&chunk_id)
        .and_then(|b| verify_proj_stmc_capsule(b).ok().map(|v| v.windows))
        .unwrap_or_default();

    let mut touched = collect_touched_control_indices(
        &state.manifest,
        &state.manifest.stems_registry[pos].controls_meta,
        &old_windows,
    );
    touched.extend(collect_touched_control_indices(
        &state.manifest,
        &replacement.controls_meta,
        &replacement.windows,
    ));

    state.stmc_digests.remove(&chunk_id);
    state
        .stmc_by_id
        .insert(chunk_id, replacement.to_stmc_bytes(stem_idx)?);
    match replacement.optional_spcm {
        Some(spcm) => {
            if spcm.len() < 64 || spcm[0..4] != *b"SPCM" {
                return Err(DnamError::SchemaValidationError(
                    "invalid SPCM header".to_string(),
                ));
            }
            state.spcm_by_id.insert(chunk_id, (spcm, None));
        }
        None => {
            state.spcm_by_id.remove(&chunk_id);
        }
    }
    state.manifest.stems_registry[pos] = StemRegistryEntry {
        stem_id: replacement.stem_id,
        chunk_id,
        enabled: replacement.enabled,
        is_star_center: replacement.is_star_center,
        controls_meta: replacement.controls_meta,
        provenance: replacement.provenance,
        user_metadata: replacement.user_metadata,
        extracted_physics: replacement.extracted_physics,
    };
    if new_is_star {
        clear_other_stars(&mut state.manifest.stems_registry, pos);
    } else if old_was_star {
        state.stat = None;
        promote_first_enabled_star(&mut state.manifest.stems_registry);
    }

    rebuild_vault_container(
        state,
        inpc_bytes,
        old_was_star || new_is_star,
        Some(touched),
        options,
    )
}

/// Non-destructively mutes (`enabled = false`) or unmutes (`enabled = true`) a capture in the
/// `.dnam-proj` Capture Vault (`set_capture_enabled`), keeping its `STMC` capsule in Region 3 while
/// updating `STMC_FLAG_ENABLED`, Phase-1 clamps, and `WGHT` (`r[train.cache.proj-vault]`).
pub fn set_capture_enabled(
    proj_bytes: &[u8],
    stem_id: &str,
    enabled: bool,
) -> Result<Vec<u8>, DnamError> {
    set_capture_enabled_with_options(proj_bytes, stem_id, enabled, SolverOptions::default())
}

/// Variant of [`set_capture_enabled`] accepting explicit [`SolverOptions`].
pub fn set_capture_enabled_with_options(
    proj_bytes: &[u8],
    stem_id: &str,
    enabled: bool,
    options: SolverOptions,
) -> Result<Vec<u8>, DnamError> {
    let mut state = extract_vault_state(proj_bytes)?;
    let pos = find_stem_pos(&state.manifest, stem_id)?;
    let inpc_bytes = state.inpc.take().ok_or_else(|| {
        DnamError::SchemaValidationError("missing INPC chunk in set_capture_enabled".to_string())
    })?;
    let chunk_id = state.manifest.stems_registry[pos].chunk_id;
    let stmc_buf = state.stmc_by_id.get(&chunk_id).ok_or_else(|| {
        DnamError::SchemaValidationError(format!(
            "missing STMC chunk {chunk_id} for stem '{stem_id}'"
        ))
    })?;
    let stmc_view = verify_proj_stmc_capsule(stmc_buf)?;
    let touched = collect_touched_control_indices(
        &state.manifest,
        &state.manifest.stems_registry[pos].controls_meta,
        &stmc_view.windows,
    );

    let was_active_star = state.manifest.stems_registry[pos].is_star_center
        && state.manifest.stems_registry[pos].enabled;
    state.manifest.stems_registry[pos].enabled = enabled;
    let mut force_full_repool = false;
    if was_active_star && !enabled {
        state.stat = None;
        force_full_repool = true;
        promote_first_enabled_star(&mut state.manifest.stems_registry);
    } else if enabled {
        let other_active_star = state
            .manifest
            .stems_registry
            .iter()
            .enumerate()
            .any(|(i, s)| i != pos && s.enabled && s.is_star_center);
        if state.manifest.stems_registry[pos].is_star_center || !other_active_star {
            clear_other_stars(&mut state.manifest.stems_registry, pos);
            state.stat = None;
            force_full_repool = true;
        }
    }

    rebuild_vault_container(state, inpc_bytes, force_full_repool, Some(touched), options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CORESET_FP32_FRAMES_PER_STEM;
    use crate::proj::bundle::{ProjChunkInput, build_dnam_proj_container_with_dnam_bin};
    use crate::proj::directory::{StmcTrajectoryWindow, build_proj_inpc_capsule};
    use crate::proj::schema::StemProvenance;
    use crate::proj::vault::VaultCaptureSpec;
    use zerocopy::IntoBytes;
    use zerocopy::byteorder::little_endian::F32;

    fn mk_spec(id: &str, star: bool, wet: Vec<u8>, spcm: Option<Vec<u8>>) -> VaultCaptureSpec {
        let sha = "0".repeat(64);
        VaultCaptureSpec {
            stem_id: id.to_string(),
            enabled: true,
            is_star_center: star,
            is_stereo: false,
            pad_gain_f32: 1.0,
            fractional_lag_samples_f32: 0.0,
            band_psd_summary_88: [0; 88],
            n_comb_bins: 0,
            windows: vec![StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: 0.5,
                window_weight: 1.0,
                emr_velocity: 0.0,
            }],
            spectral_summary_4096: [0; 4_096],
            wet_coreset_fp32: wet,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: sha.clone(),
                target_pcm_sha256: sha,
                num_frames: CORESET_FP32_FRAMES_PER_STEM as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
            optional_spcm: spcm,
        }
    }

    #[test]
    fn test_vault_edit_validation_guards_and_remove_last_stem() {
        let wet_fp32: Vec<u8> = vec![F32::new(0.0); CORESET_FP32_FRAMES_PER_STEM]
            .as_bytes()
            .to_vec();
        let mut sched = [0x44_u8; 320];
        sched[8..12].copy_from_slice(&1_024_u32.to_le_bytes());
        let inpc = build_proj_inpc_capsule(1, &[0x11; 256], &sched, &wet_fp32).unwrap();
        let dnam_bin = initialize_backbone_dnam_bin_if_needed(None).unwrap();

        let mut manifest_empty = DnamProjManifest::default_reference();
        manifest_empty.base.io_config.output_mode = "mono".into();
        let empty_proj =
            build_dnam_proj_container_with_dnam_bin(&mut manifest_empty, &[], &dnam_bin).unwrap();

        // Empty batch returns identical bytes; missing INPC / missing stem fail cleanly.
        assert_eq!(
            solve_captures_batch_into_proj(&empty_proj, Vec::new(), None).unwrap(),
            empty_proj
        );
        let spec0 = mk_spec("s0", true, wet_fp32.clone(), None);
        assert!(add_capture(&empty_proj, spec0.clone(), None).is_err());
        assert!(solve_captures_batch_into_proj(&empty_proj, vec![spec0.clone()], None).is_err());
        assert!(remove_capture(&empty_proj, "missing").is_err());
        assert!(replace_capture(&empty_proj, "missing", spec0.clone()).is_err());
        assert!(set_capture_enabled(&empty_proj, "missing", false).is_err());

        // Single-stem seeded project tests duplicate stem_id, missing INPC/STMC, invalid SPCM, and remove_capture emptying.
        let mut manifest_1 = manifest_empty.clone();
        manifest_1.stems_registry.push(StemRegistryEntry {
            stem_id: "s0".into(),
            chunk_id: 0,
            enabled: true,
            is_star_center: true,
            controls_meta: BTreeMap::new(),
            provenance: spec0.provenance.clone(),
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        });
        let mut spcm0 = vec![0_u8; 64];
        spcm0[..4].copy_from_slice(b"SPCM");
        let proj_no_inpc = build_dnam_proj_container_with_dnam_bin(
            &mut manifest_1.clone(),
            &[ProjChunkInput {
                kind: ProjChunkKind::Spcm,
                chunk_id: 0,
                payload: spcm0,
            }],
            &dnam_bin,
        )
        .unwrap();
        assert!(replace_capture(&proj_no_inpc, "s0", spec0.clone()).is_err());
        assert!(set_capture_enabled(&proj_no_inpc, "s0", false).is_err());

        let stmc0 = spec0.to_stmc_bytes(0).unwrap();
        let proj_1 = build_dnam_proj_container_with_dnam_bin(
            &mut manifest_1,
            &[
                ProjChunkInput {
                    kind: ProjChunkKind::Inpc,
                    chunk_id: 0,
                    payload: inpc.clone(),
                },
                ProjChunkInput {
                    kind: ProjChunkKind::Stmc,
                    chunk_id: 0,
                    payload: stmc0,
                },
            ],
            &dnam_bin,
        )
        .unwrap();
        assert!(add_capture(&proj_1, spec0.clone(), None).is_err());
        assert!(
            add_capture(
                &proj_1,
                mk_spec("s1", false, wet_fp32.clone(), Some(b"BAD!".to_vec())),
                None,
            )
            .is_err()
        );
        assert!(
            replace_capture(
                &proj_1,
                "s0",
                mk_spec("s0", true, wet_fp32, Some(b"BAD!".to_vec())),
            )
            .is_err()
        );

        let proj_emptied = remove_capture(&proj_1, "s0").unwrap();
        let v_empty =
            DnamProjView::from_bytes(&proj_emptied, &DefaultCryptoProvider::default()).unwrap();
        assert!(v_empty.manifest.stems_registry.is_empty());
    }
}
