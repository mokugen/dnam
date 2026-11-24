#![forbid(unsafe_code)]
//! Control Schema Bounds, Default Vectors, Metadata Mapping, and Linear Trajectory Interpolation
//! (`stimulus::sealing::controls`).

use std::collections::BTreeMap;

use dnam_engine::schema::{ControlKind, ControlSchemaDef};

use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::schema::DnamProjManifest;

/// Returns the `(lo, hi)` declared bounds of a schema control (defaulting to `(0.0, 1.0)`).
pub(crate) fn control_declared_bounds(ctrl: &ControlSchemaDef) -> (f32, f32) {
    let [lo, hi] = ctrl.declared_range.unwrap_or([0.0, 1.0]);
    (lo, hi)
}

/// Returns the clamped default scalar value for a continuous `range` control.
pub(crate) fn control_default_f32(ctrl: &ControlSchemaDef) -> f32 {
    let (lo, hi) = control_declared_bounds(ctrl);
    ctrl.default
        .as_f64()
        .map_or(0.5 * (lo + hi), |v| v as f32)
        .clamp(lo, hi)
}

/// Returns the zero-based default option index (as `f32`) for a `switch` control.
pub(crate) fn switch_default_index_f32(ctrl: &ControlSchemaDef) -> f32 {
    let Some(opts) = ctrl.declared_options.as_ref() else {
        return 0.0;
    };
    if opts.is_empty() {
        return 0.0;
    }
    let idx = ctrl
        .default
        .as_str()
        .and_then(|s| opts.iter().position(|o| o == s))
        .or_else(|| ctrl.default.as_u64().map(|u| u as usize))
        .unwrap_or(0)
        .min(opts.len() - 1);
    idx as f32
}

fn resolve_switch_option_name(
    ctrl: &ControlSchemaDef,
    idx: usize,
    values: &[f32],
    switch_overrides: &BTreeMap<usize, String>,
) -> String {
    if let Some(overridden) = switch_overrides.get(&idx) {
        return overridden.clone();
    }
    ctrl.declared_options
        .as_ref()
        .filter(|o| !o.is_empty())
        .and_then(|opts| {
            values.get(idx).and_then(|&v| {
                let k = (v.round().max(0.0) as usize).min(opts.len() - 1);
                opts.get(k).cloned()
            })
        })
        .or_else(|| ctrl.default.as_str().map(str::to_string))
        .or_else(|| {
            ctrl.declared_options
                .as_ref()
                .and_then(|o| o.first().cloned())
        })
        .unwrap_or_else(|| "default".to_string())
}

/// Serializes physical `range` values and active `switch` option names into `controls_meta`.
pub(crate) fn build_controls_meta_map(
    manifest: &DnamProjManifest,
    values: &[f32],
    switch_overrides: &BTreeMap<usize, String>,
) -> BTreeMap<String, serde_json::Value> {
    let mut map = BTreeMap::new();
    for (idx, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        match ctrl.kind {
            ControlKind::Range => {
                let v = f64::from(
                    values
                        .get(idx)
                        .copied()
                        .unwrap_or_else(|| control_default_f32(ctrl)),
                );
                if let Some(num) = serde_json::Number::from_f64(v) {
                    map.insert(ctrl.id.clone(), serde_json::Value::Number(num));
                }
            }
            ControlKind::Switch => {
                let opt = resolve_switch_option_name(ctrl, idx, values, switch_overrides);
                map.insert(ctrl.id.clone(), serde_json::Value::String(opt));
            }
        }
    }
    map
}

/// Builds the resting physical control vector ordered cleanly by `manifest.base.controls_schema`.
pub(crate) fn default_control_vector(manifest: &DnamProjManifest) -> Vec<f32> {
    manifest
        .base
        .controls_schema
        .iter()
        .map(|c| match c.kind {
            ControlKind::Range => control_default_f32(c),
            ControlKind::Switch => switch_default_index_f32(c),
        })
        .collect()
}

/// Inputs specifying linear per-frame control trajectory interpolation and `STMC` knot tracking.
pub(crate) struct TrajectoryInterpSpec<'a> {
    pub start_vals: &'a [f32],
    pub end_vals_opt: Option<&'a [f32]>,
    pub active_control_id: u16,
    pub window_weight: f32,
    pub t0: f32,
    pub t1: f32,
    pub reset_coreset_knots: bool,
    pub is_coreset: bool,
    pub frame_len: usize,
}

pub(crate) fn populate_coreset_knot_windows_with_kinds(
    start_vals: &[f32],
    end_vals: &[f32],
    is_switch: Option<&[bool]>,
    window_weight: f32,
    trajectory_windows: &mut Vec<StmcTrajectoryWindow>,
) {
    trajectory_windows.clear();
    let swept_indices: Vec<usize> = start_vals
        .iter()
        .zip(end_vals)
        .enumerate()
        .filter_map(|(p, (&v0, &v1))| ((v1 - v0).abs() > 1.0e-5).then_some(p))
        .collect();
    if swept_indices.is_empty() {
        return;
    }
    let knots = (36 / swept_indices.len()).clamp(2, 6);
    for k in 0..knots {
        let frac = (k as f32) / ((knots - 1) as f32);
        for &p in &swept_indices {
            let v0 = start_vals[p];
            let v1 = end_vals[p];
            let sw = is_switch.is_some_and(|flags| flags.get(p).copied().unwrap_or(false));
            if sw {
                let opt = if k < knots / 2 {
                    v0.round().max(0.0) as u8
                } else {
                    v1.round().max(0.0) as u8
                };
                trajectory_windows.push(StmcTrajectoryWindow {
                    control_id: p as u16,
                    ctrl_kind: 1,
                    active_opt: opt,
                    u_phys: 0.0,
                    window_weight,
                    emr_velocity: 0.0,
                });
            } else {
                trajectory_windows.push(StmcTrajectoryWindow {
                    control_id: p as u16,
                    ctrl_kind: 0,
                    active_opt: 0,
                    u_phys: v0 + (v1 - v0) * frac,
                    window_weight,
                    emr_velocity: 1.0 / (knots as f32),
                });
            }
        }
    }
}

fn populate_coreset_knot_windows(
    start_vals: &[f32],
    end_vals: &[f32],
    window_weight: f32,
    trajectory_windows: &mut Vec<StmcTrajectoryWindow>,
) {
    populate_coreset_knot_windows_with_kinds(
        start_vals,
        end_vals,
        None,
        window_weight,
        trajectory_windows,
    );
}

/// Linearly interpolates frame-level control trajectories and appends observed `STMC` windows.
pub(crate) fn interpolate_frame_trajectories(
    spec: &TrajectoryInterpSpec<'_>,
    trajectory_windows: &mut Vec<StmcTrajectoryWindow>,
) -> (Vec<f32>, Option<Vec<Vec<f32>>>) {
    let mut cur_ctrls = spec.start_vals.to_vec();
    let Some(end_vals) = spec
        .end_vals_opt
        .filter(|e| e.len() == spec.start_vals.len() && !e.is_empty())
    else {
        return (cur_ctrls, None);
    };

    if spec.reset_coreset_knots {
        populate_coreset_knot_windows(
            spec.start_vals,
            end_vals,
            spec.window_weight,
            trajectory_windows,
        );
    }

    let mut trajs = Vec::with_capacity(end_vals.len());
    let mut moved_any = false;
    let denom = spec.frame_len.saturating_sub(1).max(1);
    for (p, (&v0, &v1)) in spec.start_vals.iter().zip(end_vals).enumerate() {
        let s0 = v0 + (v1 - v0) * spec.t0;
        let s1 = v0 + (v1 - v0) * spec.t1;
        cur_ctrls[p] = s0;
        trajs.push(
            (0..spec.frame_len)
                .map(|i| s0 + (s1 - s0) * (i as f32 / denom as f32))
                .collect(),
        );
        if !spec.is_coreset && (v1 - v0).abs() > 1.0e-5 && trajectory_windows.len() < 36 {
            moved_any = true;
            trajectory_windows.push(StmcTrajectoryWindow {
                control_id: p as u16,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: s0,
                window_weight: spec.window_weight,
                emr_velocity: (spec.t1 - spec.t0).abs(),
            });
        }
    }
    if !spec.is_coreset
        && !moved_any
        && trajectory_windows.len() < 36
        && let Some(&u_val) = cur_ctrls.get(usize::from(spec.active_control_id))
    {
        trajectory_windows.push(StmcTrajectoryWindow {
            control_id: spec.active_control_id,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: u_val,
            window_weight: spec.window_weight,
            emr_velocity: (spec.t1 - spec.t0).abs(),
        });
    }
    (cur_ctrls, Some(trajs))
}

#[cfg(test)]
mod tests {
    use dnam_engine::schema::ControlUiHints;
    use serde_json::json;

    use super::*;

    fn mk_range(id: &str, range: Option<[f32; 2]>, default: serde_json::Value) -> ControlSchemaDef {
        ControlSchemaDef {
            id: id.to_string(),
            name: id.to_string(),
            kind: ControlKind::Range,
            unit: None,
            represented_switch_options: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: None,
            declared_range: range,
            safe_clamped_range: None,
            default,
            conditional_bounds: Vec::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }
    }

    fn mk_switch(
        id: &str,
        opts: Option<Vec<&str>>,
        default: serde_json::Value,
    ) -> ControlSchemaDef {
        ControlSchemaDef {
            id: id.to_string(),
            name: id.to_string(),
            kind: ControlKind::Switch,
            unit: None,
            represented_switch_options: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default,
            conditional_bounds: Vec::new(),
            declared_options: opts.map(|v| v.into_iter().map(str::to_string).collect()),
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn test_control_bounds_defaults_and_meta_map() {
        let r_mid = mk_range("drive", Some([1.0, 5.0]), json!(null));
        assert_eq!(control_declared_bounds(&r_mid), (1.0, 5.0));
        assert!((control_default_f32(&r_mid) - 3.0).abs() < 1.0e-6);

        let r_clamped = mk_range("trim", None, json!(9.0));
        assert_eq!(control_declared_bounds(&r_clamped), (0.0, 1.0));
        assert!((control_default_f32(&r_clamped) - 1.0).abs() < 1.0e-6);

        let sw_none = mk_switch("empty0", None, json!(0));
        assert_eq!(switch_default_index_f32(&sw_none), 0.0);
        let sw_empty = mk_switch("empty1", Some(vec![]), json!("a"));
        assert_eq!(switch_default_index_f32(&sw_empty), 0.0);
        let sw_str = mk_switch("voice", Some(vec!["clean", "lead", "hi"]), json!("lead"));
        assert_eq!(switch_default_index_f32(&sw_str), 1.0);
        let sw_u64 = mk_switch("mode", Some(vec!["a", "b"]), json!(99));
        assert_eq!(switch_default_index_f32(&sw_u64), 1.0);

        let mut manifest = DnamProjManifest::default_reference();
        manifest.base.controls_schema = vec![r_mid, sw_str, sw_u64, sw_none];
        let vec_def = default_control_vector(&manifest);
        assert_eq!(vec_def, vec![3.0, 1.0, 1.0, 0.0]);

        let mut overrides = BTreeMap::new();
        overrides.insert(2, "custom_b".to_string());
        let meta = build_controls_meta_map(&manifest, &[2.25, 2.0], &overrides);
        assert_eq!(meta["drive"], json!(2.25));
        assert_eq!(meta["voice"], json!("hi"));
        assert_eq!(meta["mode"], json!("custom_b"));
        assert_eq!(meta["empty0"], json!("default"));

        let fallback_meta = build_controls_meta_map(&manifest, &[], &BTreeMap::new());
        assert_eq!(fallback_meta["drive"], json!(3.0));
        assert_eq!(fallback_meta["voice"], json!("lead"));
        assert_eq!(fallback_meta["mode"], json!("a"));
    }

    #[test]
    fn test_interpolate_frame_trajectories_branches() {
        let mut wins = Vec::new();
        let (c0, t0) = interpolate_frame_trajectories(
            &TrajectoryInterpSpec {
                start_vals: &[0.2, 0.5],
                end_vals_opt: None,
                active_control_id: 0,
                window_weight: 1.0,
                t0: 0.0,
                t1: 0.5,
                reset_coreset_knots: false,
                is_coreset: false,
                frame_len: 4,
            },
            &mut wins,
        );
        assert_eq!(c0, vec![0.2, 0.5]);
        assert!(t0.is_none());

        let (c_mismatch, t_mismatch) = interpolate_frame_trajectories(
            &TrajectoryInterpSpec {
                start_vals: &[0.2, 0.5],
                end_vals_opt: Some(&[0.8]),
                active_control_id: 0,
                window_weight: 1.0,
                t0: 0.0,
                t1: 0.5,
                reset_coreset_knots: false,
                is_coreset: false,
                frame_len: 4,
            },
            &mut wins,
        );
        assert_eq!(c_mismatch, vec![0.2, 0.5]);
        assert!(t_mismatch.is_none());

        // Non-swept coreset reset produces no knot windows
        wins.push(StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 0.2,
            window_weight: 1.0,
            emr_velocity: 0.0,
        });
        let (_, t_static_coreset) = interpolate_frame_trajectories(
            &TrajectoryInterpSpec {
                start_vals: &[0.2, 0.5],
                end_vals_opt: Some(&[0.2, 0.5]),
                active_control_id: 1,
                window_weight: 0.5,
                t0: 0.0,
                t1: 1.0,
                reset_coreset_knots: true,
                is_coreset: true,
                frame_len: 4,
            },
            &mut wins,
        );
        assert!(t_static_coreset.is_some());
        assert!(wins.is_empty());

        // Swept coreset reset populates 6 knot windows, non-coreset static push adds 1 window
        let (c_swept, t_swept) = interpolate_frame_trajectories(
            &TrajectoryInterpSpec {
                start_vals: &[0.0, 0.5],
                end_vals_opt: Some(&[1.0, 0.5]),
                active_control_id: 0,
                window_weight: 0.75,
                t0: 0.2,
                t1: 0.6,
                reset_coreset_knots: true,
                is_coreset: false,
                frame_len: 4,
            },
            &mut wins,
        );
        assert!((c_swept[0] - 0.2).abs() < 1.0e-6);
        assert_eq!(t_swept.as_ref().unwrap()[0].len(), 4);
        assert_eq!(wins.len(), 7);

        let (_, _) = interpolate_frame_trajectories(
            &TrajectoryInterpSpec {
                start_vals: &[0.3, 0.5],
                end_vals_opt: Some(&[0.3, 0.5]),
                active_control_id: 1,
                window_weight: 0.9,
                t0: 0.0,
                t1: 0.25,
                reset_coreset_knots: false,
                is_coreset: false,
                frame_len: 0,
            },
            &mut wins,
        );
        assert_eq!(wins.len(), 8);
        assert_eq!(wins.last().unwrap().control_id, 1);

        // Switch kind emission via `populate_coreset_knot_windows_with_kinds` & exact endpoint `frame_len - 1`
        let mut sw_wins = Vec::new();
        populate_coreset_knot_windows_with_kinds(
            &[0.0, 0.0],
            &[1.0, 2.0],
            Some(&[false, true]),
            1.0,
            &mut sw_wins,
        );
        let switch_recs: Vec<_> = sw_wins.iter().filter(|w| w.control_id == 1).collect();
        assert!(switch_recs.len() >= 2);
        assert!(
            switch_recs
                .iter()
                .all(|w| w.ctrl_kind == 1 && w.u_phys == 0.0)
        );
        assert_eq!(switch_recs.first().unwrap().active_opt, 0);
        assert_eq!(switch_recs.last().unwrap().active_opt, 2);

        let mut end_wins = Vec::new();
        let (_, end_trajs) = interpolate_frame_trajectories(
            &TrajectoryInterpSpec {
                start_vals: &[0.0],
                end_vals_opt: Some(&[1.0]),
                active_control_id: 0,
                window_weight: 1.0,
                t0: 0.0,
                t1: 1.0,
                reset_coreset_knots: false,
                is_coreset: true,
                frame_len: 5,
            },
            &mut end_wins,
        );
        let tr = &end_trajs.unwrap()[0];
        assert!((tr[4] - 1.0).abs() < 1.0e-6);
    }
}
