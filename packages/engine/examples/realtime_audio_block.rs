//! Example 1: Real-Time Stereo & Mono Audio Block Processing (`examples/realtime_audio_block.rs`).
//!
//! Demonstrates loading a `.dnam` container into [`DynamicDnamEngine`] (which automatically
//! dispatches to `ARM SME`, `ARM64 NEON`, `x86 AVX2+FMA`, or `WASM SIMD128`), applying presets
//! (including disjoint conditional switch crossfading per `ref[dsp.ctrl.switch.conditional-batch]`),
//! tempo-syncing `DelayTimeMs` controls with out-of-range clamp badge reporting (`ref[dsp.tempo.physical-inversion]`),
//! ramping `SpilloverTrails` bypass (`ref[arch.topology.spillover-trails]`), switching quality tiers
//! (`Lite`, `Full`, `Pro`, `Ultra` `C = 40`), and inspecting `OUTSIDE_CERTIFIED_INPUT` (`bit 4`) on
//! [`DnamTelemetrySnapshot`].
//!
//! Run with:
//!   `cargo run -p dnam-engine --example realtime_audio_block`

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, StructuredConditionalBound, build_dnam_container,
};
use dnam_engine::graph::sona_core::SonaMatryoshkaModel;
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::math::spec_math::saturation_flags::OUTSIDE_CERTIFIED_INPUT;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, TempoSubdivision,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

fn build_demo_dnam_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = "SONA Crunch & Tape Echo".to_string();
    header.metadata.author = Some("DNAM Studio".to_string());
    header.metadata.description = Some("Demo stereo amp + modulated diffusive delay".to_string());
    header.metadata.license = Some("CC-BY-4.0".to_string());
    header.controls_schema = vec![
        ControlSchemaDef {
            id: "delay_time".to_string(),
            name: "Delay Time".to_string(),
            kind: ControlKind::Range,
            unit: Some("ms".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(0.2),
            conditional_bounds: vec![
                StructuredConditionalBound {
                    when: vec![("range_mode".to_string(), "short".to_string())],
                    min: Some(0.0),
                    max: Some(0.4),
                },
                StructuredConditionalBound {
                    when: vec![("range_mode".to_string(), "long".to_string())],
                    min: Some(0.6),
                    max: Some(1.0),
                },
            ],
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::DelayTimeMs,
                lobatto_physical_anchors: Some([50.0, 300.0, 800.0]),
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
            }),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "range_mode".to_string(),
            name: "Range Mode".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("short"),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: Some(vec!["short".to_string(), "long".to_string()]),
            active_options: Some(vec!["short".to_string(), "long".to_string()]),
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
    ];
    header.presets = vec![
        PresetDef {
            id: "slapback_short".to_string(),
            name: "Slapback Short".to_string(),
            description: Some("Short slapback range [0.0, 0.4]".to_string()),
            controls: BTreeMap::from([
                ("range_mode".to_string(), serde_json::json!("short")),
                ("delay_time".to_string(), serde_json::json!(0.25)),
            ]),
            user_metadata: BTreeMap::new(),
        },
        PresetDef {
            id: "ambient_long".to_string(),
            name: "Ambient Long".to_string(),
            description: Some("Disjoint long range [0.6, 1.0]".to_string()),
            controls: BTreeMap::from([
                ("range_mode".to_string(), serde_json::json!("long")),
                ("delay_time".to_string(), serde_json::json!(0.85)),
            ]),
            user_metadata: BTreeMap::new(),
        },
    ];
    header.io_config.capabilities.supports_freeze = true;

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let model = SonaMatryoshkaModel::dual_rate_default(false);
    pack_sona_matryoshka_to_arena(&model, &mut arena);
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn main() {
    let container = build_demo_dnam_container();

    // 1. Load model with automatic hardware SIMD backend selection (ARM SME / NEON / AVX2 / WASM SIMD)
    let dyn_engine = DynamicDnamEngine::from_bytes(container.as_bytes(), Tier::Full, 48_000)
        .expect("valid .dnam container");
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .expect("valid .dnam container");

    println!(
        "Loaded model '{}' on hardware backend: {} | supports_freeze = {}",
        engine.header().unwrap().metadata.name,
        dyn_engine.backend_kind().name(),
        engine.supports_freeze(),
    );

    // 2. Apply presets across disjoint conditional switch intervals ("slapback_short" -> "ambient_long")
    engine
        .apply_preset("slapback_short")
        .expect("preset 'slapback_short' exists");
    engine
        .apply_preset("ambient_long")
        .expect("preset 'ambient_long' triggers disjoint-interval crossfade");

    // 3. Process a 128-sample stereo buffer at 48 kHz
    let mut in_l = [0.0_f32; 128];
    let mut in_r = [0.0_f32; 128];
    for (n, (l, r)) in in_l.iter_mut().zip(in_r.iter_mut()).enumerate() {
        let phase = (n as f32) * 0.08;
        *l = 0.25 * phase.sin();
        *r = 0.25 * phase.cos();
    }
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 4. Sync delay time to host tempo (300 BPM 1/16T = 33.3 ms < 50 ms min -> clamped with UI badge bit)
    let updated = engine.sync_tempo(300.0, Some(TempoSubdivision::SixteenthTriplet), false);
    println!(
        "Tempo-synced {} control(s) | active_clamp_mask = 0x{:04x}",
        updated,
        engine.telemetry_snapshot().active_clamp_mask,
    );

    // 5. Engage SpilloverTrails bypass and switch to Ultra tier (`C = 40`) with 0 allocations
    engine.set_bypass(true, true);
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    engine.set_bypass(false, true);
    engine.set_tier(Tier::Ultra);
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 6. Feed a > +40 dBFS (+150.0) peak to verify OUTSIDE_CERTIFIED_INPUT telemetry flag (bit 4)
    in_l[0] = 150.0;
    engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);

    // 7. Inspect lock-free telemetry snapshot
    let snap = engine.telemetry_snapshot();
    println!(
        "Processed 4x128 stereo frames | Active Tier: {:?} | Max PolyADAA Drive: {:.4} | Max Pole Radius: {:.6} | Outside Certified Envelope Flag: {}",
        Tier::from_u8(snap.active_tier),
        snap.max_op_n_drive,
        snap.max_pole_radius,
        (snap.overflow_or_saturation_flags & OUTSIDE_CERTIFIED_INPUT) != 0,
    );
}
