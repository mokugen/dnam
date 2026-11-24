#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Deterministic `< 10 ms` `PANAMA` audit-time corner prescription generator
//! (`dnam_capture::audit::dynamic_and_panama::panama`).

use dnam_engine::schema::{ControlKind, DnamError};
use dnam_train::{
    PanamaControlDesc, PanamaControlKind, PanamaInput, PanamaOutput, PanamaSampleRow,
    recommend_panama_corners,
};
use dnam_vm::bin::DnamBinView;
use dnam_vm::control::ControlClampDescriptor;

use super::corners::switch_active_options;
use super::map_err;
use crate::proj::schema::DnamProjManifest;

/// Evaluates deterministic `< 10 ms` `PANAMA` corner recommendations from Pass-4 solved sample rows,
/// replacing any stale `[PANAMA]` entries in `manifest.audit_report.prescriptions`
/// (`ref[train.audit.report.prescription]`).
pub fn append_panama_prescriptions(
    manifest: &mut DnamProjManifest,
    dnam_bin: &[u8],
    sample_rows: &[PanamaSampleRow],
) -> Result<Option<PanamaOutput>, DnamError> {
    if let Some(ar) = manifest.audit_report.as_mut() {
        ar.prescriptions.retain(|p| !p.starts_with("[PANAMA]"));
    }
    if sample_rows.is_empty() || manifest.base.controls_schema.is_empty() {
        return Ok(None);
    }

    let view = DnamBinView::parse(dnam_bin).map_err(map_err)?;
    let controls: Vec<PanamaControlDesc> = manifest
        .base
        .controls_schema
        .iter()
        .enumerate()
        .map(|(wire_ctrl_idx, ctrl)| {
            let kind = match ctrl.kind {
                ControlKind::Range => {
                    let [lo, hi] = ctrl
                        .safe_clamped_range
                        .or(ctrl.declared_range)
                        .unwrap_or([0.0, 1.0]);
                    let safe_min = f64::from(lo);
                    let safe_max = f64::from(hi).max(safe_min);
                    let star_center = ctrl
                        .default
                        .as_f64()
                        .unwrap_or(safe_min)
                        .clamp(safe_min, safe_max);
                    PanamaControlKind::Range {
                        safe_min,
                        safe_max,
                        star_center,
                    }
                }
                ControlKind::Switch => {
                    let declared_option_names = ctrl.declared_options.clone().unwrap_or_default();
                    let mut active_options: Vec<u8> = if let Some(ref act) = ctrl.active_options {
                        act.iter()
                            .filter_map(|name| {
                                declared_option_names
                                    .iter()
                                    .position(|d| d == name)
                                    .map(|idx| idx as u8)
                            })
                            .collect()
                    } else if let Some(w) = view.control_descriptors.get(wire_ctrl_idx) {
                        let d = ControlClampDescriptor::from_wire(w);
                        switch_active_options(&d).0
                    } else {
                        (0..declared_option_names.len().min(16) as u8).collect()
                    };
                    active_options.sort_unstable();
                    active_options.dedup();
                    let def_by_name = ctrl.default.as_str().and_then(|s| {
                        declared_option_names
                            .iter()
                            .position(|d| d == s)
                            .map(|idx| idx as u8)
                    });
                    let def_num = ctrl.default.as_u64().map(|v| v as u8);
                    let fallback = active_options.first().copied().unwrap_or(0);
                    let star_center_option = def_by_name.or(def_num).unwrap_or(fallback);
                    PanamaControlKind::Switch {
                        active_options,
                        declared_option_names,
                        star_center_option,
                    }
                }
            };
            PanamaControlDesc {
                control_id: ctrl.id.clone(),
                wire_ctrl_idx,
                kind,
            }
        })
        .collect();

    let enabled_shas: Vec<String> = manifest
        .stems_registry
        .iter()
        .filter(|s| s.enabled)
        .map(|s| s.provenance.target_pcm_sha256.clone())
        .collect();

    let out = recommend_panama_corners(&PanamaInput {
        view: &view,
        controls: &controls,
        sample_rows,
        enabled_stem_target_sha256s: &enabled_shas,
    })
    .map_err(map_err)?;

    if !out.prescriptions.is_empty() {
        let ar = manifest.audit_report.get_or_insert_with(Default::default);
        ar.prescriptions.extend(out.prescriptions.iter().cloned());
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::proj::schema::{AuditReport, StemProvenance, StemRegistryEntry};
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;
    use dnam_engine::schema::{ControlSchemaDef, ControlUiHints};
    use dnam_vm::bin::{ControlClampDescriptorWire, DnamBinBuildInput, encode_dnam_bin};
    use dnam_vm::control::STAGE_TARGET_SLOTS;

    fn mk_range_wire(min: f32, max: f32, def: f32) -> ControlClampDescriptorWire {
        let mut d = ControlClampDescriptor::empty();
        d.ctrl_kind = 0;
        d.lobatto_degree = 3;
        d.default_val = def;
        d.safe_min_base = min;
        d.safe_max_base = max;
        d.to_wire()
    }

    #[test]
    fn test_append_panama_prescriptions_replaces_stale_lines_without_duplicates() {
        let mut manifest = DnamProjManifest::default_reference();
        let mk_ctrl = |id: &str| ControlSchemaDef {
            id: id.into(),
            name: id.into(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(3),
            anchor_nodes_norm: None,
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(0.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        };
        manifest.base.controls_schema = vec![mk_ctrl("drive"), mk_ctrl("tone")];
        let mut controls_meta = BTreeMap::new();
        controls_meta.insert("drive".into(), serde_json::json!(0.0));
        controls_meta.insert("tone".into(), serde_json::json!(0.0));
        manifest.stems_registry = vec![StemRegistryEntry {
            stem_id: "s0".into(),
            chunk_id: 1,
            enabled: true,
            is_star_center: true,
            controls_meta,
            provenance: StemProvenance {
                input_pcm_sha256: "a".repeat(64),
                target_pcm_sha256: "b".repeat(64),
                num_frames: 48_000,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        }];
        manifest.audit_report = Some(AuditReport {
            prescriptions: vec![
                "[CLAMP-RANGE] 'x'".into(),
                "[PANAMA] Corner (stale=1.0000) (sigma2=0.5000)".into(),
            ],
            ..Default::default()
        });

        let base_bin = initialize_backbone_dnam_bin_if_needed(None).unwrap();
        let v0 = DnamBinView::parse(&base_bin).unwrap();
        let ctrls = [mk_range_wire(0.0, 10.0, 0.0), mk_range_wire(0.0, 10.0, 0.0)];
        let dbin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: v0.header.reference_sample_rate_hz.get(),
            header_flags: v0.header.flags.get(),
            ext_ring_pow2_frames: v0.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: v0.runtime_meta.ext_ring_lines,
            ext_ring_format: v0.runtime_meta.ext_ring_format,
            declared_max_delay_48k: v0.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: v0.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: v0.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: v0.runtime_meta.ro_pole_48k.get(),
            switch_xfade_samples: v0.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: v0.runtime_meta.content_hash_prefix,
            controls: &ctrls,
            cp1_overflow: &[],
            tsyn: &[],
            cext: v0.cext,
            arena0: v0.arena0,
            arena1: v0.arena1,
            arena2: v0.arena2,
            arena3: v0.arena3,
        })
        .unwrap();

        let mut t_d = [0.0_f64; STAGE_TARGET_SLOTS];
        t_d[0] = 0.8;
        let mut t_t = [0.0_f64; STAGE_TARGET_SLOTS];
        t_t[1] = 0.6;
        let mut t_m = [0.0_f64; STAGE_TARGET_SLOTS];
        t_m[0] = 0.9;
        t_m[1] = 0.7;
        let mk_row = |id: &str, ctrls: Vec<f64>, tgt: [f64; STAGE_TARGET_SLOTS]| PanamaSampleRow {
            stem_id: id.into(),
            window_idx: 0,
            control_values: ctrls,
            stage_targets: tgt,
        };
        let rows = vec![
            mk_row("s0", vec![0.0, 0.0], [0.0; STAGE_TARGET_SLOTS]),
            mk_row("s1", vec![10.0, 0.0], t_d),
            mk_row("s2", vec![0.0, 10.0], t_t),
            mk_row("s3", vec![5.0, 5.0], t_m),
        ];

        let out = append_panama_prescriptions(&mut manifest, &dbin, &rows)
            .unwrap()
            .unwrap();
        let _ = append_panama_prescriptions(&mut manifest, &dbin, &rows).unwrap();
        let presc = &manifest.audit_report.as_ref().unwrap().prescriptions;
        assert!(presc.iter().any(|l| l == "[CLAMP-RANGE] 'x'"));
        assert!(!presc.iter().any(|l| l.contains("stale=1.0000")));
        assert_eq!(
            presc.iter().filter(|l| l.starts_with("[PANAMA]")).count(),
            out.prescriptions.len()
        );
    }
}
