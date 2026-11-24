#![forbid(unsafe_code)]
//! Multi-stem point & Mode A sweep collection plus `StarCenter` anchor preservation for Stage C
//! (`solver::stage_c_knobs::collect`).

use std::collections::BTreeMap;

use dnam_train::control::{ControlKindSpec, ControlSpec, Cp1ControlSample};
use dnam_vm::control::STAGE_TARGET_SLOTS;

use super::sweep_windows::{
    ModeASweepSchedule, SweepStemInputs, build_mode_a_sweep_schedule,
    extract_sweep_windows_for_stem,
};
use super::{
    A5_ACTIVE_BURST_END, A5_STATIONARY_FIT_LEN, A5_STATIONARY_OFFSET, decode_coreset_lr,
    parse_stem_control_values,
};
use crate::proj::capsules::{StmcTrajectoryWindow, verify_proj_stmc_capsule};
use crate::proj::schema::DnamProjManifest;
use crate::proj::{STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STEREO};

/// Pending Stage C control point sliced from an enabled capture stem (`Mode A` sweep or static).
pub(super) struct PendingPoint {
    pub(super) dry_off: usize,
    pub(super) len: usize,
    pub(super) wet_lr: Vec<[f32; 2]>,
    pub(super) phys_controls: Vec<f32>,
}

/// Full active trajectory metadata for an enabled `MODE_A_SWEEP` stem (`ref[train.pipeline.control-projection-fit]`).
pub(super) struct ModeASweepStem {
    pub(super) n_active: usize,
    pub(super) wet_lr: Vec<[f32; 2]>,
    pub(super) schedule: ModeASweepSchedule,
}

/// Aggregated observations extracted across all enabled stems in `manifest`.
pub(super) struct CollectedStageCPoints {
    pub(super) observed_points: Vec<Vec<f32>>,
    pub(super) pending: Vec<PendingPoint>,
    pub(super) parsed_windows: Vec<Vec<StmcTrajectoryWindow>>,
    pub(super) mode_a_sweeps: Vec<ModeASweepStem>,
}

/// Collects trajectory windows and pending fit points across all enabled stems (`star_center`
/// Mode A sweeps + off-center stems) so neither sweep metadata nor control points are dropped.
pub(super) fn collect_stage_c_points(
    manifest: &DnamProjManifest,
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    dry_lr: &[[f32; 2]],
    base_ctrls: &[f32],
    active_frames_hint: usize,
) -> CollectedStageCPoints {
    let avail = dry_lr.len();
    let n_active = if active_frames_hint == 0 {
        avail
    } else {
        active_frames_hint.min(avail)
    };
    let n_fit = if n_active >= 48_000 {
        4_096
    } else {
        2_048.min(n_active)
    };

    let mut observed_points = Vec::new();
    let mut pending = Vec::new();
    let mut parsed_windows: Vec<Vec<StmcTrajectoryWindow>> = Vec::new();
    let mut mode_a_sweeps = Vec::new();

    for stem in manifest.stems_registry.iter().filter(|s| s.enabled) {
        let Some(stmc_raw) = stmc_by_id.get(&stem.chunk_id) else {
            continue;
        };
        let Ok(stmc_view) = verify_proj_stmc_capsule(stmc_raw) else {
            continue;
        };
        let ch = if (stmc_view.flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let wet_lr = decode_coreset_lr(stmc_view.wet_coreset_fp32, ch);
        let base_meta_ctrls = parse_stem_control_values(manifest, &stem.controls_meta, base_ctrls);
        parsed_windows.push(stmc_view.windows.clone());

        if (stmc_view.flags & STMC_FLAG_MODE_A_SWEEP) != 0
            && !dry_lr.is_empty()
            && !wet_lr.is_empty()
        {
            let schedule = build_mode_a_sweep_schedule(&stmc_view.windows, &base_meta_ctrls);
            if schedule.m_max >= 2 {
                let sweep_n = n_active.min(wet_lr.len()).min(dry_lr.len());
                mode_a_sweeps.push(ModeASweepStem {
                    n_active: sweep_n,
                    wet_lr: wet_lr[..sweep_n].to_vec(),
                    schedule,
                });
            }
        }

        let mut pairs = Vec::new();
        let mut extracted = Vec::new();
        if !dry_lr.is_empty()
            && extract_sweep_windows_for_stem(
                &SweepStemInputs {
                    dry_lr,
                    wet_lr: &wet_lr,
                    n_fit,
                    n_active,
                    windows: &stmc_view.windows,
                    base_meta_ctrls: &base_meta_ctrls,
                },
                &mut pairs,
                &mut extracted,
            )
        {
            for win in extracted {
                observed_points.push(win.phys_controls.clone());
                let wet_slice = wet_lr[win.start_frame..win.start_frame + n_fit].to_vec();
                pending.push(PendingPoint {
                    dry_off: win.start_frame,
                    len: n_fit,
                    wet_lr: wet_slice,
                    phys_controls: win.phys_controls,
                });
            }
            continue;
        }

        if !stem.is_star_center && wet_lr.len() >= 64 && !dry_lr.is_empty() {
            observed_points.push(base_meta_ctrls.clone());
            if (A5_ACTIVE_BURST_END..48_000).contains(&n_active)
                && dry_lr.len() >= A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN
                && wet_lr.len() >= A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN
            {
                let wet_slice = wet_lr
                    [A5_STATIONARY_OFFSET..A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN]
                    .to_vec();
                pending.push(PendingPoint {
                    dry_off: A5_STATIONARY_OFFSET,
                    len: A5_STATIONARY_FIT_LEN,
                    wet_lr: wet_slice,
                    phys_controls: base_meta_ctrls,
                });
            } else {
                let len = wet_lr.len().min(n_fit).min(dry_lr.len());
                let wet_slice = wet_lr[..len].to_vec();
                pending.push(PendingPoint {
                    dry_off: 0,
                    len,
                    wet_lr: wet_slice,
                    phys_controls: base_meta_ctrls,
                });
            }
        }
    }

    CollectedStageCPoints {
        observed_points,
        pending,
        parsed_windows,
        mode_a_sweeps,
    }
}

/// Inserts an exact zero-residual `StarCenter` sample at index `0` when an enabled `StarCenter`
/// stem exists and no solved pending point coincides with `base_ctrls`, preventing single
/// off-center captures from subtracting themselves as the `CP1` origin.
pub(super) fn ensure_star_center_anchor(
    manifest: &DnamProjManifest,
    base_ctrls: &[f32],
    specs: &[ControlSpec],
    samples: &mut Vec<Cp1ControlSample>,
) {
    let has_star = manifest
        .stems_registry
        .iter()
        .any(|s| s.enabled && s.is_star_center);
    if !has_star {
        return;
    }
    let already_present = samples.iter().any(|samp| {
        samp.phys_controls.len() == base_ctrls.len()
            && samp
                .phys_controls
                .iter()
                .zip(base_ctrls)
                .all(|(&a, &b)| (a - b).abs() <= 1.0e-3)
    });
    if already_present {
        return;
    }
    let switch_options = specs
        .iter()
        .map(|sp| match sp.kind {
            ControlKindSpec::Switch { default_opt, .. } => default_opt,
            ControlKindSpec::Range { .. } => 0,
        })
        .collect();
    samples.insert(
        0,
        Cp1ControlSample {
            phys_controls: base_ctrls.to_vec(),
            switch_options,
            stage_targets_base: [0.0; STAGE_TARGET_SLOTS],
            stage_targets_ultra_delta: None,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_star_center_anchor_inserts_zero_origin_once() {
        let mut manifest = DnamProjManifest::default_reference();
        manifest
            .stems_registry
            .push(crate::proj::schema::StemRegistryEntry {
                stem_id: String::from("star"),
                chunk_id: 0,
                enabled: true,
                is_star_center: true,
                controls_meta: BTreeMap::new(),
                provenance: crate::proj::schema::StemProvenance {
                    input_pcm_sha256: "0".repeat(64),
                    target_pcm_sha256: "1".repeat(64),
                    num_frames: 64,
                },
                user_metadata: BTreeMap::new(),
                extracted_physics: None,
            });
        let specs = vec![
            ControlSpec {
                kind: ControlKindSpec::Range {
                    declared_min: 0.0,
                    declared_max: 10.0,
                    default_val: 5.0,
                    obs_min: 5.0,
                    obs_max: 8.0,
                    unique_values: 2,
                    lobatto_anchors: [5.0, 5.0, 8.0],
                },
            },
            ControlSpec {
                kind: ControlKindSpec::Switch {
                    num_options: 2,
                    default_opt: 1,
                    declared_mask: 0b11,
                    observed_mask: 0b11,
                },
            },
        ];
        let mut samples = vec![Cp1ControlSample {
            phys_controls: vec![8.0, 0.0],
            switch_options: vec![0, 0],
            stage_targets_base: [1.0; STAGE_TARGET_SLOTS],
            stage_targets_ultra_delta: None,
        }];
        ensure_star_center_anchor(&manifest, &[5.0, 1.0], &specs, &mut samples);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].phys_controls, vec![5.0, 1.0]);
        assert_eq!(samples[0].switch_options, vec![0, 1]);
        assert_eq!(samples[0].stage_targets_base, [0.0; STAGE_TARGET_SLOTS]);

        ensure_star_center_anchor(&manifest, &[5.0, 1.0], &specs, &mut samples);
        assert_eq!(samples.len(), 2);
    }
}
