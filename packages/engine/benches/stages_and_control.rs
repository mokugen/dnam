//! Stage Recurrence & Control Plane Criterion Benchmarks (`benches/stages_and_control.rs`).
//!
//! Benchmarks `TptLr4Crossover`, `V3Stage0Hot`, `V3Stage3Hot`, `V3Stage4TankHot`, Stage IV loop
//! saturator & blended knee, `eval_stage_modulation_laws`, `ControlBus::tick_3khz`,
//! `ExactIntFactorAccumulator`, `Q64`/`Q32` tempo sync & physical calibration inversion,
//! `apply_preset` disjoint switch crossfade, and `sync_tempo`.

use std::collections::BTreeMap;
use std::hint::black_box;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, StructuredConditionalBound, build_dnam_container,
};
use dnam_engine::control::{
    ExactIntFactorAccumulator, eval_q32_subdivision_phase, invert_tempo_physical_calibration,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, TempoSubdivision,
};
use dnam_engine::stages::router::{V3Stage0Hot, V3Stage0Params};
use dnam_engine::stages::sona_core::{SonaColdTierModel, SonaHotSlab, SonaMatryoshkaModel};
use dnam_engine::stages::stage3_cab::V3Stage3Hot;
use dnam_engine::stages::topology::{
    TptLr4Crossover, eval_stage_modulation_laws, eval_stage4_blended_feedback_knee,
    eval_stage4_loop_saturator,
};
use dnam_engine::stages::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::stages::{V3Stage4TankHot, V3Stage4TankParams};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};
use hotpath::criterion::{Criterion, criterion_group, criterion_main};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = "Stages and Control Benchmark Model".to_string();
    header.controls_schema = vec![
        ControlSchemaDef {
            id: "gain".to_string(),
            name: "Gain".to_string(),
            kind: ControlKind::Range,
            unit: Some("ms".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(0.2),
            conditional_bounds: vec![
                StructuredConditionalBound {
                    when: vec![("mode".to_string(), "a".to_string())],
                    min: None,
                    max: Some(0.4),
                },
                StructuredConditionalBound {
                    when: vec![("mode".to_string(), "b".to_string())],
                    min: Some(0.6),
                    max: None,
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
            id: "mode".to_string(),
            name: "Mode".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("a"),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: Some(vec!["a".to_string(), "b".to_string()]),
            active_options: Some(vec!["a".to_string(), "b".to_string()]),
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
    ];
    header.presets = vec![
        PresetDef {
            id: "preset_a".to_string(),
            name: "Preset A".to_string(),
            description: None,
            controls: BTreeMap::from([
                ("mode".to_string(), serde_json::json!("a")),
                ("gain".to_string(), serde_json::json!(0.2)),
            ]),
            user_metadata: BTreeMap::new(),
        },
        PresetDef {
            id: "preset_b_disjoint".to_string(),
            name: "Preset B Disjoint".to_string(),
            description: None,
            controls: BTreeMap::from([
                ("mode".to_string(), serde_json::json!("b")),
                ("gain".to_string(), serde_json::json!(0.8)),
            ]),
            user_metadata: BTreeMap::new(),
        },
    ];

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    for s4 in &mut model.stage4 {
        s4.g_loop = 0.80;
        s4.wet_mix = 0.15;
    }
    pack_sona_matryoshka_to_arena(&model, &mut arena);
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn bench_stages_and_tank(c: &mut Criterion) {
    let container = make_bench_container();
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();

    let mut group = c.benchmark_group("stages_and_tank");

    group.bench_function("sona_step_frame_full", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(engine.step_frame_sona(black_box(0.15), black_box(-0.15), false)));
    });

    group.bench_function("stage0_route_frame_no_return", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(
                engine
                    .stage0_router
                    .route_frame_no_return(black_box(0.15), black_box(-0.15)),
            )
        });
    });

    let mut s0_hot = V3Stage0Hot::from_cold(&V3Stage0Params::identity());
    group.bench_function("stage0_hot_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(s0_hot.step_stereo(black_box(0.15), black_box(-0.15))));
    });

    let mut s2_core = SonaHotSlab::<DefaultMathBackend>::from_cold_model(
        &SonaColdTierModel::identity_for_tier(Tier::Lite, false, false),
    );
    group.bench_function("stage2_sona_hot_slab_lite_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(s2_core.step_stereo(black_box(0.15), black_box(-0.15))));
    });

    let mut s3_cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
    s3_cold.readout.fir_l[..16].fill(0.0625);
    s3_cold.readout.fir_r[..16].fill(0.0625);
    let mut s3_hot = V3Stage3Hot::from_cold(&s3_cold);
    group.bench_function("stage3_fir_kautz_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(s3_hot.step_stereo::<DefaultMathBackend>(
                black_box(0.15),
                black_box(-0.15),
                2,
                2,
                false,
            ))
        });
    });

    let mut s4 = V3Stage4TankHot::from_cold(&V3Stage4TankParams::identity());
    group.bench_function("stage4_v3_tank_stereo", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(s4.step_stereo::<DefaultMathBackend>(black_box(0.15), black_box(-0.15)))
        });
    });

    let mut lr4 = TptLr4Crossover::new_48khz();
    group.bench_function("stage0_tpt_lr4_crossover", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| black_box(lr4.step(black_box(0.25))));
    });

    group.bench_function("stage4_loop_saturator_and_knee", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            let g = eval_stage4_blended_feedback_knee(
                black_box(0.35),
                black_box(0.85),
                black_box(0.25),
                black_box(0.50),
            )
            .unwrap_or(0.0);
            black_box(eval_stage4_loop_saturator(black_box(g), black_box(0.95)))
        });
    });

    group.finish();
}

fn bench_control_plane_and_ux(c: &mut Criterion) {
    let container = make_bench_container();
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container.as_bytes(), &crypto).unwrap();
    let arena = view.arena_storage.as_bytes();

    let mut group = c.benchmark_group("control_plane_and_ux");

    group.bench_function("control_bus_tick_3khz", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            engine.control_bus.set_control_value(0, black_box(0.75));
            let changed = engine.control_bus.tick_3khz(black_box(0.1), arena);
            black_box(changed)
        });
    });

    let theta = [0.25_f32; 16];
    group.bench_function("eval_stage_modulation_laws", |b| {
        let _audio = FtzDazGuard::new();
        b.iter(|| {
            black_box(eval_stage_modulation_laws(
                black_box(&theta),
                black_box([0.8, 0.7]),
                black_box(24.0),
                black_box(240.0),
                black_box(64.0),
                black_box(1024.0),
                black_box(0.82),
                black_box(0.5),
                black_box(1.0),
            ))
        });
    });

    let mut exact_acc = ExactIntFactorAccumulator::new(4);
    let contrib_bits = [
        0.25_f32.to_bits(),
        (-0.15_f32).to_bits(),
        0.4_f32.to_bits(),
        0.1_f32.to_bits(),
    ];
    group.bench_function("exact_int_factor_accumulator", |b| {
        b.iter(|| {
            exact_acc.reset();
            for (idx, &bits) in contrib_bits.iter().enumerate() {
                exact_acc.update_contributor(idx, black_box(bits)).unwrap();
            }
            black_box(exact_acc.interaction_f32_bits())
        });
    });

    group.bench_function("tempo_sync_q32_and_physical_inversion", |b| {
        b.iter(|| {
            let q32 = eval_q32_subdivision_phase(
                black_box(3),
                black_box(0x4000_0000),
                black_box((3, 4)),
                black_box(0),
            )
            .unwrap();
            let inv = invert_tempo_physical_calibration(
                (50.0, 800.0),
                (50.0, 800.0),
                [50.0, 300.0, 800.0],
                black_box(375.0),
                true,
            )
            .unwrap();
            black_box((q32, inv.normalized_f32))
        });
    });

    group.bench_function("ux_apply_preset_disjoint_switch_xfade", |b| {
        b.iter(|| {
            engine.apply_preset(black_box("preset_a")).unwrap();
            engine.apply_preset(black_box("preset_b_disjoint")).unwrap();
        });
    });

    group.bench_function("ux_sync_tempo_clamped", |b| {
        b.iter(|| {
            black_box(engine.sync_tempo(
                black_box(128.0),
                black_box(Some(TempoSubdivision::Quarter)),
                black_box(true),
            ))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_stages_and_tank, bench_control_plane_and_ux);
criterion_main!(benches);
