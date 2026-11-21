#![forbid(unsafe_code)]
#![allow(dead_code)]
//! Calibration-role control schema builders shared by multi-port lifecycle tests.

use std::collections::BTreeMap;

use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    TempoSubdivision,
};

pub fn make_calibration_control(
    id: &str,
    name: &str,
    decl: [f32; 2],
    default_v: f32,
    role: ControlCalibrationRole,
    anchors: [f32; 3],
    subdiv: Option<TempoSubdivision>,
) -> ControlSchemaDef {
    ControlSchemaDef {
        id: id.to_string(),
        name: name.to_string(),
        kind: ControlKind::Range,
        unit: None,
        lobatto_degree: Some(2),
        anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
        declared_range: Some(decl),
        safe_clamped_range: Some(decl),
        default: serde_json::json!(default_v),
        conditional_bounds: Vec::new(),
        represented_switch_options: None,
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        calibration: Some(ControlCalibration {
            role,
            lobatto_physical_anchors: Some(anchors),
            conditional_physical_anchors: Vec::new(),
            tempo_sync_default_subdivision: subdiv,
        }),
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    }
}

pub fn build_calibration_controls_schema() -> Vec<ControlSchemaDef> {
    vec![
        make_calibration_control(
            "delay_ms",
            "Delay",
            [20.0, 800.0],
            250.0,
            ControlCalibrationRole::DelayTimeMs,
            [20.0, 250.0, 800.0],
            Some(TempoSubdivision::EighthDotted),
        ),
        make_calibration_control(
            "feedback",
            "Feedback",
            [0.0, 10.0],
            5.0,
            ControlCalibrationRole::Feedback,
            [0.1, 0.5, 1.15],
            None,
        ),
        make_calibration_control(
            "width",
            "Stereo Width",
            [0.0, 1.0],
            1.0,
            ControlCalibrationRole::StereoWidth,
            [0.0, 0.5, 1.0],
            None,
        ),
        make_calibration_control(
            "lfo_hz",
            "Mod Rate",
            [0.1, 8.0],
            1.0,
            ControlCalibrationRole::LfoRateHz,
            [0.1, 1.0, 8.0],
            Some(TempoSubdivision::Quarter),
        ),
        make_calibration_control(
            "mix",
            "Mix",
            [0.0, 1.0],
            0.5,
            ControlCalibrationRole::Mix,
            [0.0, 0.5, 1.0],
            None,
        ),
        make_calibration_control(
            "grain_sz",
            "Grain Size",
            [10.0, 500.0],
            100.0,
            ControlCalibrationRole::GrainSizeMs,
            [10.0, 100.0, 500.0],
            None,
        ),
        make_calibration_control(
            "grain_pitch",
            "Pitch",
            [-24.0, 24.0],
            0.0,
            ControlCalibrationRole::GrainPitchSemitones,
            [-24.0, 0.0, 24.0],
            None,
        ),
        make_calibration_control(
            "grain_spray",
            "Spray",
            [0.0, 1.0],
            0.2,
            ControlCalibrationRole::GrainSpray,
            [0.0, 0.2, 1.0],
            None,
        ),
        ControlSchemaDef {
            id: "stereo_mode".to_string(),
            name: "Stereo Mode".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("Stereo"),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: Some(vec!["Mono".to_string(), "Stereo".to_string()]),
            active_options: Some(vec!["Mono".to_string(), "Stereo".to_string()]),
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::StereoMode,
                lobatto_physical_anchors: None,
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
    ]
}
