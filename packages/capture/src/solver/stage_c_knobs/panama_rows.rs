#![forbid(unsafe_code)]
//! Pass-4 `PANAMA` sample-row construction (`M_max` synchronous `MODE_A_SWEEP` + stationary
//! stems, sorted ascending `(stem_id, window_idx)`, slot-`12` variance gating) and continuous
//! `3 kHz` `eval_control_trajectory_3khz_vm` verification (`ref[train.pipeline.dispatch-solve]`).

use std::collections::BTreeMap;

use dnam_engine::schema::DnamError;
use dnam_train::{
    CP1_DELAY_VARIANCE_GATE_SAMPLES2, PanamaSampleRow, eval_control_trajectory_3khz_vm,
    eval_static_control_bus_targets,
};
use dnam_vm::bin::DnamBinView;
use dnam_vm::control::{ControlBus, STAGE_TARGET_SLOTS};
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use super::collect::ModeASweepStem;
use super::sweep_windows::{
    build_3khz_control_trajectory, build_mode_a_sweep_schedule, sample_synchronous_m_max_controls,
};
use super::{map_stage_err, parse_stem_control_values, star_center_control_values};
use crate::proj::capsules::verify_proj_stmc_capsule;
use crate::proj::schema::DnamProjManifest;
use crate::proj::{STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STEREO};

/// Assembles sorted Pass-4 `(stem_id, window_idx)` sample rows across all enabled stems
/// (stationary + synchronous `M_max` `MODE_A_SWEEP`) from the fitted static `ControlBus`, zeroing
/// slot `12` (`tank_delay_mod`) when its sample population variance falls below `0.2` samples².
pub(super) fn build_panama_sample_rows_from_fitted_bus(
    manifest: &DnamProjManifest,
    dnam_bin: &[u8],
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    active_frames_hint: usize,
) -> Result<Vec<PanamaSampleRow>, DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    if view.header.num_controls.get() == 0 || manifest.base.controls_schema.is_empty() {
        return Ok(Vec::new());
    }
    let mut bus =
        ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view, view.retained_tier, 48_000.0)
            .map_err(map_stage_err)?;
    let base_ctrls = star_center_control_values(manifest);
    let mut rows = Vec::new();

    for stem in manifest.stems_registry.iter().filter(|s| s.enabled) {
        let base_meta = parse_stem_control_values(manifest, &stem.controls_meta, &base_ctrls);
        let stmc_opt = stmc_by_id
            .get(&stem.chunk_id)
            .and_then(|raw| verify_proj_stmc_capsule(raw).ok());

        let is_mode_a = stmc_opt
            .as_ref()
            .is_some_and(|sv| (sv.flags & STMC_FLAG_MODE_A_SWEEP) != 0);

        if let Some(ref sv) = stmc_opt
            && is_mode_a
        {
            let ch = if (sv.flags & STMC_FLAG_STEREO) != 0 {
                2
            } else {
                1
            };
            let wet_frames = (sv.wet_coreset_fp32.len() / (4 * ch)).max(1);
            let n_active = if active_frames_hint == 0 {
                wet_frames
            } else {
                active_frames_hint.min(wet_frames).max(1)
            };
            let sched = build_mode_a_sweep_schedule(&sv.windows, &base_meta);
            for (window_idx, ctrl_vec) in sample_synchronous_m_max_controls(&sched, n_active) {
                let updates: Vec<(usize, f32)> = ctrl_vec.iter().copied().enumerate().collect();
                let (targets, _) =
                    eval_static_control_bus_targets(&mut bus, &updates).map_err(map_stage_err)?;
                rows.push(PanamaSampleRow {
                    stem_id: stem.stem_id.clone(),
                    window_idx,
                    control_values: ctrl_vec.into_iter().map(f64::from).collect(),
                    stage_targets: targets,
                });
            }
        } else {
            let updates: Vec<(usize, f32)> = base_meta.iter().copied().enumerate().collect();
            let (targets, _) =
                eval_static_control_bus_targets(&mut bus, &updates).map_err(map_stage_err)?;
            rows.push(PanamaSampleRow {
                stem_id: stem.stem_id.clone(),
                window_idx: 0,
                control_values: base_meta.into_iter().map(f64::from).collect(),
                stage_targets: targets,
            });
        }
    }

    if !rows.is_empty() {
        let n_f = rows.len() as f64;
        let mean12 = rows.iter().map(|r| r.stage_targets[12]).sum::<f64>() / n_f;
        let var12 = rows
            .iter()
            .map(|r| (r.stage_targets[12] - mean12).powi(2))
            .sum::<f64>()
            / n_f;
        if var12 < CP1_DELAY_VARIANCE_GATE_SAMPLES2 {
            for r in &mut rows {
                r.stage_targets[12] = 0.0;
            }
        }
    }

    rows.sort_by(|a, b| {
        a.stem_id
            .cmp(&b.stem_id)
            .then(a.window_idx.cmp(&b.window_idx))
    });
    Ok(rows)
}

/// Evaluates post-warmup `SonaVm` waveform MSE over a continuous `3 kHz` `u(t)` trajectory for a
/// `MODE_A_SWEEP` stem via [`eval_control_trajectory_3khz_vm`].
pub(super) fn eval_mode_a_sweep_3khz_trajectory_mse(
    dnam_bin: &[u8],
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    tier: Tier,
    dry_lr: &[[f32; 2]],
    sweep: &ModeASweepStem,
    mono_out: bool,
    dual_in: bool,
) -> Result<f64, DnamError> {
    let n = sweep.n_active.min(dry_lr.len()).min(sweep.wet_lr.len());
    if n < 32 {
        return Ok(0.0);
    }
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let mut bus = ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view, tier, 48_000.0)
        .map_err(map_stage_err)?;
    let ctrl_traj = build_3khz_control_trajectory(&sweep.schedule, n);
    let mut traj_32 = Vec::with_capacity(ctrl_traj.len());
    let mut z_8 = Vec::with_capacity(ctrl_traj.len());
    for ctrls in &ctrl_traj {
        let updates: Vec<(usize, f32)> = ctrls.iter().copied().enumerate().collect();
        let (st64, z32) =
            eval_static_control_bus_targets(&mut bus, &updates).map_err(map_stage_err)?;
        let st32: [f32; STAGE_TARGET_SLOTS] = core::array::from_fn(|i| {
            if i + 1 < STAGE_TARGET_SLOTS && st64[i].is_finite() {
                st64[i] as f32
            } else {
                0.0
            }
        });
        traj_32.push(st32);
        z_8.push(z32);
    }

    let keep_in_r = dual_in || !mono_out;
    let in_l: Vec<f32> = dry_lr[..n].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr[..n]
        .iter()
        .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
        .collect();
    let (pred_l, pred_r) =
        eval_control_trajectory_3khz_vm(vm, &in_l, &in_r, &traj_32, &z_8).map_err(map_stage_err)?;

    let warmup = (n / 8).min(256).min(n.saturating_sub(8));
    let mut sq = 0.0_f64;
    let mut cnt = 0usize;
    for t in warmup..n {
        let dl = pred_l[t] - f64::from(sweep.wet_lr[t][0]);
        sq += dl * dl;
        cnt += 1;
        if !mono_out {
            let dr = pred_r[t] - f64::from(sweep.wet_lr[t][1]);
            sq += dr * dr;
            cnt += 1;
        }
    }
    Ok(if cnt == 0 { 0.0 } else { sq / (cnt as f64) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::capsules::build_proj_stmc_capsule;
    use crate::proj::schema::StemProvenance;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;
    use crate::proj::{STMC_FLAG_ENABLED, STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STAR_CENTER};
    use dnam_train::audit_phase1_controls;
    use dnam_train::control::{ControlKindSpec, ControlSpec, StageStabilityConstraint};
    use dnam_vm::bin::{DnamBinBuildInput, encode_dnam_bin};
    use dnam_vm::control::Cp1ContributorSlot;
    use zerocopy::byteorder::little_endian::F32;

    #[test]
    fn test_panama_rows_m_max_sorting_slot12_variance_gate_and_3khz_sweep() {
        let mut manifest = DnamProjManifest::default_reference();
        manifest.base.controls_schema = vec![
            serde_json::from_value(serde_json::json!({
                "id": "drive", "name": "Drive", "type": "range",
                "declared_range": [-1.0, 1.0], "safe_clamped_range": [-1.0, 1.0], "default": 0.0
            }))
            .unwrap(),
        ];

        let base_bin = initialize_backbone_dnam_bin_if_needed(None).unwrap();
        let bv = DnamBinView::parse(&base_bin).unwrap();
        let mut a0 = *bv.arena0;
        let p1 = audit_phase1_controls(
            &[ControlSpec {
                kind: ControlKindSpec::Range {
                    declared_min: -1.0,
                    declared_max: 1.0,
                    default_val: 0.0,
                    obs_min: -1.0,
                    obs_max: 1.0,
                    unique_values: 3,
                    lobatto_anchors: [-1.0, 0.0, 1.0],
                },
            }],
            &StageStabilityConstraint::default(),
        )
        .unwrap();
        a0.control_bus_lite.w_ctrl[0][0] = F32::new(0.5);
        a0.control_bus_lite.w_ctrl[12][0] = F32::new(0.1); // variance = 0.01 * E[u^2] < 0.2 -> gated!
        let mut slot = Cp1ContributorSlot::zero();
        slot.control_idx = 0;
        slot.lanes = 1;
        slot.aux = 0xFF;
        slot.main_base[0][0] = 1.0;
        let cp1_ov = dnam_engine::container::encode_cp1_shared_stream(&[slot]).unwrap();
        let dbin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: bv.header.reference_sample_rate_hz.get(),
            header_flags: bv.header.flags.get(),
            ext_ring_pow2_frames: bv.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: bv.runtime_meta.ext_ring_lines,
            ext_ring_format: bv.runtime_meta.ext_ring_format,
            declared_max_delay_48k: bv.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: bv.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: bv.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: bv.runtime_meta.ro_pole_48k.get(),
            switch_xfade_samples: bv.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: bv.runtime_meta.content_hash_prefix,
            controls: &p1.wire_descriptors,
            cp1_overflow: &cp1_ov,
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: bv.arena1,
            arena2: bv.arena2,
            arena3: bv.arena3,
        })
        .unwrap();

        let cap_n = crate::CORESET_FP32_FRAMES_PER_STEM;
        let stmc_bytes = vec![0u8; cap_n * 4];
        let win = |u: f32| crate::proj::capsules::StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: u,
            window_weight: 1.0,
            emr_velocity: 0.1,
        };
        let mut stmc_by_id = BTreeMap::new();
        stmc_by_id.insert(
            0,
            build_proj_stmc_capsule(
                0,
                STMC_FLAG_ENABLED | STMC_FLAG_STAR_CENTER,
                1.0,
                0.0,
                &[0u8; 88],
                0,
                &[win(0.0)],
                &[0u8; 4_096],
                &stmc_bytes,
            )
            .unwrap(),
        );
        stmc_by_id.insert(
            1,
            build_proj_stmc_capsule(
                1,
                STMC_FLAG_ENABLED | STMC_FLAG_MODE_A_SWEEP,
                1.0,
                0.0,
                &[0u8; 88],
                0,
                &[win(-1.0), win(0.0), win(1.0)],
                &[0u8; 4_096],
                &stmc_bytes,
            )
            .unwrap(),
        );
        let prov = StemProvenance {
            input_pcm_sha256: "0".repeat(64),
            target_pcm_sha256: "1".repeat(64),
            num_frames: cap_n as u64,
        };
        manifest.stems_registry = vec![
            crate::proj::schema::StemRegistryEntry {
                stem_id: "b_star".into(),
                chunk_id: 0,
                enabled: true,
                is_star_center: true,
                controls_meta: BTreeMap::from([("drive".into(), serde_json::json!(0.0))]),
                provenance: prov.clone(),
                user_metadata: BTreeMap::new(),
                extracted_physics: None,
            },
            crate::proj::schema::StemRegistryEntry {
                stem_id: "a_sweep".into(),
                chunk_id: 1,
                enabled: true,
                is_star_center: false,
                controls_meta: BTreeMap::from([("drive".into(), serde_json::json!(0.0))]),
                provenance: prov,
                user_metadata: BTreeMap::new(),
                extracted_physics: None,
            },
        ];

        let rows =
            build_panama_sample_rows_from_fitted_bus(&manifest, &dbin, &stmc_by_id, 97).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].stem_id, "a_sweep");
        assert_eq!(rows[0].window_idx, 0);
        assert_eq!(rows[1].window_idx, 1);
        assert_eq!(rows[2].window_idx, 2);
        assert_eq!(rows[3].stem_id, "b_star");
        assert!((rows[0].stage_targets[0] - (-0.5)).abs() < 1.0e-5);
        assert!(rows.iter().all(|r| r.stage_targets[12] == 0.0));

        let dry = vec![[0.1_f32, 0.1_f32]; 128];
        let sweep = ModeASweepStem {
            n_active: 128,
            wet_lr: vec![[0.0_f32; 2]; 128],
            schedule: build_mode_a_sweep_schedule(&[win(-1.0), win(1.0)], &[0.0]),
        };
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&dbin, Tier::Ultra, 48_000.0)
                .unwrap();
        let mse = eval_mode_a_sweep_3khz_trajectory_mse(
            &dbin,
            &mut vm,
            Tier::Ultra,
            &dry,
            &sweep,
            false,
            true,
        )
        .unwrap();
        assert!(mse.is_finite());
    }
}
