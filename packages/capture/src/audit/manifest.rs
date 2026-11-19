#![forbid(unsafe_code)]
//! Manifest-Level Phase-1 Control-Space Audit & Clamp Domains (`dnam_capture::audit::manifest`).

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::{ControlKind, ControlSchemaDef};

use crate::proj::capsules::StmcTrajectoryWindow;
use crate::proj::schema::{AuditReport, DnamProjManifest, StemRegistryEntry};

use super::{audit_phase1_clamp_range, audit_phase1_clamp_switch};

fn clamp_range_conditional_corners(
    ctrl: &mut ControlSchemaDef,
    ctrl_idx: usize,
    switch_ids: &[String],
    enabled_stems: &[&StemRegistryEntry],
    enabled_windows_by_stem: &BTreeMap<u32, Vec<StmcTrajectoryWindow>>,
    safe_bounds: (f32, f32),
    default_clamped: f32,
    phase1_tags: &mut BTreeSet<String>,
) {
    let (safe_lo, safe_hi) = safe_bounds;
    ctrl.conditional_min.clear();
    ctrl.conditional_max.clear();
    for sw_id in switch_ids {
        let mut by_opt: BTreeMap<String, (f32, f32)> = BTreeMap::new();
        for s in enabled_stems {
            let Some(opt_str) = s.controls_meta.get(sw_id).and_then(|v| v.as_str()) else {
                continue;
            };
            let mut stem_vals = Vec::new();
            if let Some(v_num) = s.controls_meta.get(&ctrl.id).and_then(|v| v.as_f64())
                && v_num.is_finite()
            {
                stem_vals.push((v_num as f32).clamp(safe_lo, safe_hi));
            }
            if let Some(wins) = enabled_windows_by_stem.get(&s.chunk_id) {
                for w in wins {
                    if usize::from(w.control_id) == ctrl_idx
                        && w.ctrl_kind == 0
                        && w.window_weight > 0.0
                        && w.u_phys.is_finite()
                    {
                        stem_vals.push(w.u_phys.clamp(safe_lo, safe_hi));
                    }
                }
            }
            if stem_vals.is_empty() {
                continue;
            }
            let key = format!("{sw_id}={opt_str}");
            for vf in stem_vals {
                by_opt
                    .entry(key.clone())
                    .and_modify(|(lo, hi)| {
                        *lo = lo.min(vf);
                        *hi = hi.max(vf);
                    })
                    .or_insert((vf, vf));
            }
        }
        for (cond_key, (c_lo, c_hi)) in by_opt {
            if (c_hi - c_lo).abs() <= 1.0e-4 && (c_lo - default_clamped).abs() <= 1.0e-4 {
                continue;
            }
            if c_lo > safe_lo + 1.0e-4 {
                ctrl.conditional_min.insert(cond_key.clone(), c_lo);
                phase1_tags.insert("CLAMP-CORNER".to_string());
            }
            if c_hi < safe_hi - 1.0e-4 {
                ctrl.conditional_max.insert(cond_key, c_hi);
                phase1_tags.insert("CLAMP-CORNER".to_string());
            }
        }
    }
}

fn audit_range_control(
    ctrl: &mut ControlSchemaDef,
    ctrl_idx: usize,
    switch_ids: &[String],
    enabled_stems: &[&StemRegistryEntry],
    enabled_windows_by_stem: &BTreeMap<u32, Vec<StmcTrajectoryWindow>>,
    phase1_tags: &mut BTreeSet<String>,
    prescriptions: &mut Vec<String>,
) {
    let decl = ctrl.declared_range.unwrap_or([0.0, 1.0]);
    let default_f32 = ctrl.default.as_f64().unwrap_or(f64::from(decl[0])) as f32;
    let mut vals: Vec<f32> = enabled_stems
        .iter()
        .filter_map(|s| s.controls_meta.get(&ctrl.id).and_then(|v| v.as_f64()))
        .map(|v| v as f32)
        .filter(|v| v.is_finite())
        .collect();
    for s in enabled_stems {
        if let Some(wins) = enabled_windows_by_stem.get(&s.chunk_id) {
            for w in wins {
                if usize::from(w.control_id) == ctrl_idx
                    && w.ctrl_kind == 0
                    && w.window_weight > 0.0
                    && w.u_phys.is_finite()
                {
                    vals.push(w.u_phys);
                }
            }
        }
    }

    let ((safe_lo, safe_hi), deg, clamped) = if vals.is_empty() {
        let c = default_f32.clamp(decl[0], decl[1]);
        ((c, c), 0u8, true)
    } else {
        vals.sort_by(|a, b| a.total_cmp(b));
        let mut distinct = 1usize;
        for pair in vals.windows(2) {
            if (pair[1] - pair[0]).abs() > 1.0e-4 {
                distinct += 1;
            }
        }
        let obs = (vals[0], *vals.last().unwrap());
        audit_phase1_clamp_range((decl[0], decl[1]), obs, distinct)
    };

    ctrl.safe_clamped_range = Some([safe_lo, safe_hi]);
    ctrl.lobatto_degree = Some(deg);
    let default_clamped = default_f32.clamp(safe_lo, safe_hi);
    clamp_range_conditional_corners(
        ctrl,
        ctrl_idx,
        switch_ids,
        enabled_stems,
        enabled_windows_by_stem,
        (safe_lo, safe_hi),
        default_clamped,
        phase1_tags,
    );

    if clamped {
        phase1_tags.insert("CLAMP-RANGE".to_string());
        ctrl.clamp_warning = Some(format!(
            "[CLAMP-RANGE] '{}': clamped [{:.3}, {:.3}] -> [{:.3}, {:.3}]",
            ctrl.id, decl[0], decl[1], safe_lo, safe_hi
        ));
        prescriptions.push(format!(
            "Capture stems at '{}={:.3}' and '{}={:.3}' to unlock full range",
            ctrl.id, decl[0], ctrl.id, decl[1]
        ));
    } else {
        ctrl.clamp_warning = None;
    }
}

fn audit_switch_control(
    ctrl: &mut ControlSchemaDef,
    ctrl_idx: usize,
    enabled_stems: &[&StemRegistryEntry],
    enabled_windows_by_stem: &BTreeMap<u32, Vec<StmcTrajectoryWindow>>,
    phase1_tags: &mut BTreeSet<String>,
    prescriptions: &mut Vec<String>,
) {
    let decl_opts = ctrl.declared_options.clone().unwrap_or_default();
    let decl_refs: Vec<&str> = decl_opts.iter().map(String::as_str).collect();
    let mut obs_strings: BTreeSet<String> = enabled_stems
        .iter()
        .filter_map(|s| s.controls_meta.get(&ctrl.id).and_then(|v| v.as_str()))
        .map(ToString::to_string)
        .collect();
    for s in enabled_stems {
        if let Some(wins) = enabled_windows_by_stem.get(&s.chunk_id) {
            for w in wins {
                if usize::from(w.control_id) == ctrl_idx
                    && w.ctrl_kind == 1
                    && w.window_weight > 0.0
                    && let Some(opt_name) = decl_opts.get(usize::from(w.active_opt))
                {
                    obs_strings.insert(opt_name.clone());
                }
            }
        }
    }
    let obs_refs: BTreeSet<&str> = obs_strings.iter().map(String::as_str).collect();
    let (_mask, disabled) = audit_phase1_clamp_switch(&decl_refs, &obs_refs);
    let mut active: Vec<String> = decl_opts
        .iter()
        .filter(|o| obs_strings.contains(*o))
        .cloned()
        .collect();
    if active.is_empty() && !decl_opts.is_empty() {
        let fallback = ctrl
            .default
            .as_str()
            .filter(|d| decl_opts.iter().any(|o| o == d))
            .unwrap_or(decl_opts[0].as_str())
            .to_string();
        active.push(fallback);
    }
    ctrl.active_options = Some(active);
    if !disabled.is_empty() {
        phase1_tags.insert("CLAMP-SWITCH".to_string());
        ctrl.clamp_warning = Some(format!(
            "[CLAMP-SWITCH] '{}': disabled unobserved options {:?}",
            ctrl.id, disabled
        ));
        for opt in disabled {
            prescriptions.push(format!(
                "Capture a stem with '{}={opt}' to unlock switch option",
                ctrl.id
            ));
        }
    } else {
        ctrl.clamp_warning = None;
    }
}

/// Recomputes Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-CORNER]` safe control domains
/// and updates `manifest.audit_report` across all stems with `enabled == true`.
pub fn recompute_vault_phase1_audit_and_clamps(manifest: &mut DnamProjManifest) {
    recompute_vault_phase1_audit_and_clamps_with_windows(manifest, &BTreeMap::new());
}

/// Recomputes Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-CORNER]` safe control domains
/// across all enabled stems in `manifest`, incorporating both `controls_meta` and positive-weight
/// [`StmcTrajectoryWindow`] observations from `enabled_windows_by_stem`.
pub fn recompute_vault_phase1_audit_and_clamps_with_windows(
    manifest: &mut DnamProjManifest,
    enabled_windows_by_stem: &BTreeMap<u32, Vec<StmcTrajectoryWindow>>,
) {
    let enabled_stems: Vec<&StemRegistryEntry> = manifest
        .stems_registry
        .iter()
        .filter(|s| s.enabled)
        .collect();

    let switch_ids: Vec<String> = manifest
        .base
        .controls_schema
        .iter()
        .filter(|c| c.kind == ControlKind::Switch)
        .map(|c| c.id.clone())
        .collect();

    let mut phase1_tags = BTreeSet::new();
    let mut prescriptions = Vec::new();

    for (ctrl_idx, ctrl) in manifest.base.controls_schema.iter_mut().enumerate() {
        match ctrl.kind {
            ControlKind::Range => audit_range_control(
                ctrl,
                ctrl_idx,
                &switch_ids,
                &enabled_stems,
                enabled_windows_by_stem,
                &mut phase1_tags,
                &mut prescriptions,
            ),
            ControlKind::Switch => audit_switch_control(
                ctrl,
                ctrl_idx,
                &enabled_stems,
                enabled_windows_by_stem,
                &mut phase1_tags,
                &mut prescriptions,
            ),
        }
    }

    let prev = manifest.audit_report.take().unwrap_or_default();
    manifest.audit_report = Some(AuditReport {
        phase1_clamps: phase1_tags.into_iter().collect(),
        phase2_rail_warnings: prev.phase2_rail_warnings,
        phase3_clamps: prev.phase3_clamps,
        prescriptions,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::StemProvenance;

    fn make_stem(
        chunk_id: u32,
        enabled: bool,
        meta: &[(&str, serde_json::Value)],
    ) -> StemRegistryEntry {
        StemRegistryEntry {
            stem_id: format!("stem-{chunk_id}"),
            chunk_id,
            enabled,
            is_star_center: chunk_id == 1,
            controls_meta: meta
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
            provenance: StemProvenance {
                input_pcm_sha256: "0".repeat(64),
                target_pcm_sha256: "1".repeat(64),
                num_frames: 64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        }
    }

    fn make_manifest_with_range_and_switch() -> DnamProjManifest {
        let mut m = DnamProjManifest::default_reference();
        m.base.controls_schema = vec![
            serde_json::from_value(serde_json::json!({
                "id": "gain",
                "name": "Gain",
                "type": "range",
                "declared_range": [0.0, 10.0],
                "default": 4.0
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "id": "mode",
                "name": "Mode",
                "type": "switch",
                "declared_options": ["clean", "crunch", "lead"],
                "default": "crunch"
            }))
            .unwrap(),
        ];
        m
    }

    #[test]
    fn test_phase1_range_windows_corners_and_switch_fallback() {
        // 1. Empty observations: range locks to clamped default, switch locks to default fallback
        let mut m_empty = make_manifest_with_range_and_switch();
        recompute_vault_phase1_audit_and_clamps(&mut m_empty);
        assert_eq!(
            m_empty.base.controls_schema[0].safe_clamped_range,
            Some([4.0, 4.0])
        );
        assert_eq!(m_empty.base.controls_schema[0].lobatto_degree, Some(0));
        assert_eq!(
            m_empty.base.controls_schema[1].active_options.as_deref(),
            Some(&["crunch".to_string()][..])
        );
        let rep0 = m_empty.audit_report.as_ref().unwrap();
        assert!(rep0.phase1_clamps.iter().any(|t| t == "CLAMP-RANGE"));
        assert!(rep0.phase1_clamps.iter().any(|t| t == "CLAMP-SWITCH"));
        assert!(!rep0.prescriptions.is_empty());

        // 2. Enabled + disabled stems + positive & zero-weight STMC windows + conditional corner narrowing
        let mut m = make_manifest_with_range_and_switch();
        m.stems_registry = vec![
            make_stem(
                1,
                true,
                &[
                    ("gain", serde_json::json!(2.0)),
                    ("mode", serde_json::json!("clean")),
                ],
            ),
            make_stem(
                2,
                true,
                &[
                    ("gain", serde_json::json!(5.0)),
                    ("mode", serde_json::json!("crunch")),
                ],
            ),
            make_stem(
                3,
                false, // Disabled stem must be ignored
                &[
                    ("gain", serde_json::json!(10.0)),
                    ("mode", serde_json::json!("lead")),
                ],
            ),
        ];

        let mut windows = BTreeMap::new();
        windows.insert(
            1u32,
            vec![
                StmcTrajectoryWindow {
                    control_id: 0,
                    ctrl_kind: 0,
                    active_opt: 0,
                    u_phys: 3.5,
                    window_weight: 0.75,
                    emr_velocity: 0.1,
                },
                StmcTrajectoryWindow {
                    control_id: 0,
                    ctrl_kind: 0,
                    active_opt: 0,
                    u_phys: 9.9,
                    window_weight: 0.0, // Zero weight ignored
                    emr_velocity: 0.1,
                },
            ],
        );
        windows.insert(
            2u32,
            vec![
                StmcTrajectoryWindow {
                    control_id: 0,
                    ctrl_kind: 0,
                    active_opt: 0,
                    u_phys: 8.0,
                    window_weight: 1.0,
                    emr_velocity: 0.1,
                },
                StmcTrajectoryWindow {
                    control_id: 1,
                    ctrl_kind: 1,
                    active_opt: 2, // Unlocks "lead" via window observation
                    window_weight: 0.5,
                    u_phys: 0.0,
                    emr_velocity: 0.1,
                },
            ],
        );

        recompute_vault_phase1_audit_and_clamps_with_windows(&mut m, &windows);
        let range_ctrl = &m.base.controls_schema[0];
        let switch_ctrl = &m.base.controls_schema[1];
        assert_eq!(range_ctrl.safe_clamped_range, Some([2.0, 8.0]));
        assert_eq!(range_ctrl.lobatto_degree, Some(3));
        assert_eq!(range_ctrl.conditional_max.get("mode=clean"), Some(&3.5));
        assert_eq!(range_ctrl.conditional_min.get("mode=crunch"), Some(&5.0));
        assert_eq!(
            switch_ctrl.active_options.as_deref(),
            Some(
                &[
                    "clean".to_string(),
                    "crunch".to_string(),
                    "lead".to_string()
                ][..]
            )
        );
        assert!(switch_ctrl.clamp_warning.is_none());

        let rep = m.audit_report.as_ref().unwrap();
        assert!(rep.phase1_clamps.iter().any(|t| t == "CLAMP-CORNER"));
        assert!(rep.phase1_clamps.iter().any(|t| t == "CLAMP-RANGE"));
        assert!(!rep.phase1_clamps.iter().any(|t| t == "CLAMP-SWITCH"));
    }

    #[test]
    fn test_phase1_full_observed_range_clears_warning() {
        let mut m = make_manifest_with_range_and_switch();
        m.stems_registry = vec![
            make_stem(
                1,
                true,
                &[
                    ("gain", serde_json::json!(0.0)),
                    ("mode", serde_json::json!("clean")),
                ],
            ),
            make_stem(
                2,
                true,
                &[
                    ("gain", serde_json::json!(10.0)),
                    ("mode", serde_json::json!("crunch")),
                ],
            ),
            make_stem(
                3,
                true,
                &[
                    ("gain", serde_json::json!(5.0)),
                    ("mode", serde_json::json!("lead")),
                ],
            ),
        ];
        recompute_vault_phase1_audit_and_clamps(&mut m);
        assert!(m.base.controls_schema[0].clamp_warning.is_none());
        assert!(m.base.controls_schema[1].clamp_warning.is_none());
        let rep = m.audit_report.as_ref().unwrap();
        assert!(!rep.phase1_clamps.iter().any(|t| t == "CLAMP-RANGE"));
        assert!(!rep.phase1_clamps.iter().any(|t| t == "CLAMP-SWITCH"));
    }

    #[test]
    fn test_phase1_stationary_default_switch_stem_does_not_clamp_range_corners() {
        let mut m = make_manifest_with_range_and_switch();
        m.stems_registry = vec![
            make_stem(
                1,
                true,
                &[
                    ("gain", serde_json::json!(4.0)),
                    ("mode", serde_json::json!("crunch")),
                ],
            ),
            make_stem(
                2,
                true,
                &[
                    ("gain", serde_json::json!(0.0)),
                    ("mode", serde_json::json!("crunch")),
                ],
            ),
            make_stem(
                3,
                true,
                &[
                    ("gain", serde_json::json!(10.0)),
                    ("mode", serde_json::json!("crunch")),
                ],
            ),
            make_stem(
                4,
                true,
                &[
                    ("gain", serde_json::json!(4.0)),
                    ("mode", serde_json::json!("lead")),
                ],
            ),
        ];
        recompute_vault_phase1_audit_and_clamps(&mut m);
        let range_ctrl = &m.base.controls_schema[0];
        let switch_ctrl = &m.base.controls_schema[1];
        assert_eq!(range_ctrl.safe_clamped_range, Some([0.0, 10.0]));
        assert!(range_ctrl.conditional_min.is_empty());
        assert!(range_ctrl.conditional_max.is_empty());
        assert!(
            switch_ctrl
                .active_options
                .as_ref()
                .is_some_and(|opts| opts.iter().any(|o| o == "lead"))
        );
        let rep = m.audit_report.as_ref().unwrap();
        assert!(!rep.phase1_clamps.iter().any(|t| t == "CLAMP-CORNER"));
    }
}
