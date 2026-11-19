//! Non-Criterion Real-Time Worst-Case Execution Time (WCET), Tail-Latency Jitter (`P50`/`P99`/`P99.9`/`Max`),
//! and `x Real-Time @ 48 kHz` / `% Single-Core CPU` Benchmark (`benches/realtime_wcet.rs`).
//!
//! Run with:
//!   `cargo bench -p dnam-engine --bench realtime_wcet`

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    DnamContainerView, build_dnam_container,
};
use dnam_engine::engine::EngineBackend;
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlBusConfig, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
    ControlUiHints, DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, PresetDef,
    RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TempoSubdivision, TierProfile,
};
use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, NUM_GRU_BLOCKS, flat_gru_lane_index, tier_fir_segments, tier_kautz_banks,
};
use dnam_engine::stages::weights::pack_v3_matryoshka_to_arena;
use dnam_engine::stages::{V3ColdTierModel, V3CoreInstance, V3HotCore, V3MatryoshkaModel};
use dnam_engine::{DnamEngine, Tier};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut cond_min = BTreeMap::new();
    cond_min.insert("mode=a".to_string(), 0.0);
    cond_min.insert("mode=b".to_string(), 0.6);
    let mut cond_max = BTreeMap::new();
    cond_max.insert("mode=a".to_string(), 0.4);
    cond_max.insert("mode=b".to_string(), 1.0);

    let mut header = DnamJsonHeader {
        version: "2.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "WCET Benchmark Model".to_string(),
            author: None,
            description: None,
            license: None,
            created_at: "2026-10-23T00:00:00Z".to_string(),
            input_reference_dbfs: -20.5,
            provenance: DatasetProvenance::default(),
            distribution_policy: None,
            signatures: None,
        },
        user_metadata: BTreeMap::new(),
        io_config: IoConfig::default(),
        slimmable_tiers: SlimmableTiersConfig {
            lite: TierProfile {
                feature_bus_dim: 64,
                channels: 4,
                early_fir_taps: 16,
                kautz_modes: 4,
                tank_lines: 4,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -40.0,
                arena_end_byte_offset: ARENA_OFFSET_LITE_END,
                prefix_crc32c: 0,
            },
            full: TierProfile {
                feature_bus_dim: 128,
                channels: 8,
                early_fir_taps: 32,
                kautz_modes: 8,
                tank_lines: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -45.0,
                arena_end_byte_offset: ARENA_OFFSET_FULL_END,
                prefix_crc32c: 0,
            },
            pro: TierProfile {
                feature_bus_dim: 192,
                channels: 12,
                early_fir_taps: 48,
                kautz_modes: 16,
                tank_lines: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -48.0,
                arena_end_byte_offset: ARENA_OFFSET_PRO_END,
                prefix_crc32c: 0,
            },
            ultra: TierProfile {
                feature_bus_dim: 256,
                channels: 16,
                early_fir_taps: 64,
                kautz_modes: 32,
                tank_lines: 16,
                head_svd_rank: 8,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -51.0,
                arena_end_byte_offset: ARENA_OFFSET_ULTRA_END,
                prefix_crc32c: 0,
            },
        },
        controls_schema: vec![
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
                conditional_min: cond_min,
                conditional_max: cond_max,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::DelayTimeMs,
                    lobatto_physical_anchors: Some([50.0, 300.0, 800.0]),
                    conditional_physical_anchors: BTreeMap::new(),
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
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
                active_options: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
                clamp_warning: None,
                ui_hints: ControlUiHints::default(),
                calibration: None,
                user_metadata: BTreeMap::new(),
            },
        ],
        presets: vec![
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
        ],
        control_bus: ControlBusConfig::default(),
        stage0_router: RouterConfig::default(),
        stage1_pre: Stage1PreFxConfig::default(),
        stage2_cascade: Stage2CascadeConfig::default(),
        stage3_cabinet: Stage3CabinetKautzConfig::default(),
        stage4_tank: Stage4SpatialTankConfig::default(),
        delay_histories: Vec::new(),
        tensor_directory: BTreeMap::new(),
    };

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = V3MatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;
    model.stage0.clean_sub_mix = 0.05;
    model.stage0.op_p_sub_mix = 0.12;
    model.svf.env_oct_sens = 1.8;
    model.tone_svf.h0_proj_w = [0.2, -0.1, 0.15, 0.05];
    model.tone_svf.s1_proj_w = 0.3;
    model.ctrl.w_film_gain[0][0][0] = 0.35;
    for b in 0..NUM_GRU_BLOCKS {
        for dst in 0..MAX_BLOCK_LANES {
            model.blocks[b].w_in_n[0][dst] = 0.65;
            model.blocks[b].b_z[dst] = -0.60;
            model.blocks[b].b_r[dst] = 1.25;
            model.blocks[b].w_rec_n[dst][dst] = 0.15;
            if b > 0 {
                model.blocks[b].w_prev_n[dst][dst] = 0.45;
            }
        }
    }
    for s4 in &mut model.stage4 {
        s4.wet_mix = 0.15;
        s4.g_loop = 0.85;
        s4.lfo_rate_hz = 1.6;
        s4.lfo_depth_samples = 6.0;
        s4.lfo_stereo_phase = core::f32::consts::FRAC_PI_2;
    }
    pack_v3_matryoshka_to_arena(&model, &mut arena);
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn bench_tier<B: EngineBackend>(container: &[u8], tier: Tier, label: &str) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    let block_size = 128_usize;
    let num_blocks = 375_usize; // 48,000 samples = 1.0 second of 48 kHz stereo audio
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    for _ in 0..512 {
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
    }

    let t0 = Instant::now();
    for _ in 0..num_blocks {
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
    }
    let elapsed = t0.elapsed();
    let total_samples = (block_size * num_blocks) as f64;
    let ns_per_sample = (elapsed.as_nanos() as f64) / total_samples;
    let realtime_budget_ns = 1_000_000_000.0 / 48_000.0; // 20,833.33 ns/sample
    let x_realtime = realtime_budget_ns / ns_per_sample;
    let cpu_pct = (ns_per_sample / realtime_budget_ns) * 100.0;

    println!(
        "| {:<32} | {:<11} | {:>10.2} ns | {:>10.1}x | {:>7.2}% |",
        label,
        B::KIND.name(),
        ns_per_sample,
        x_realtime,
        cpu_pct
    );
}

fn bench_tier_bounce<B: EngineBackend>(container: &[u8], tier: Tier, label: &str) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    engine.set_offline_bounce(true);
    let block_size = 128_usize;
    let num_blocks = 375_usize; // 48,000 samples = 1.0 second of 48 kHz stereo audio
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    for _ in 0..128 {
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
    }

    let t0 = Instant::now();
    for _ in 0..num_blocks {
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
    }
    let elapsed = t0.elapsed();
    let total_samples = (block_size * num_blocks) as f64;
    let ns_per_sample = (elapsed.as_nanos() as f64) / total_samples;
    let realtime_budget_ns = 1_000_000_000.0 / 48_000.0; // 20,833.33 ns/sample
    let x_realtime = realtime_budget_ns / ns_per_sample;
    let cpu_pct = (ns_per_sample / realtime_budget_ns) * 100.0;

    println!(
        "| {:<32} | {:<11} | {:>10.2} ns | {:>10.1}x | {:>7.2}% |",
        label,
        B::KIND.name(),
        ns_per_sample,
        x_realtime,
        cpu_pct
    );
}

fn bench_wcet_tail_latency<B: EngineBackend>(container: &[u8], tier: Tier, label: &str) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    let in_l = [0.18_f32; 128];
    let in_r = [-0.18_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    let num_blocks = 2_000_usize;
    let mut latencies_us = Vec::with_capacity(num_blocks);

    for _ in 0..64 {
        engine.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    }

    for b_idx in 0..num_blocks {
        if b_idx % 250 == 0 {
            let preset_id = if (b_idx / 250) % 2 == 0 {
                "preset_a"
            } else {
                "preset_b_disjoint"
            };
            engine.apply_preset(black_box(preset_id)).unwrap();
        } else if b_idx % 125 == 0 {
            let _ = engine.sync_tempo(
                black_box(120.0 + ((b_idx as f32) * 0.05)),
                black_box(Some(TempoSubdivision::EighthDotted)),
                black_box(true),
            );
        } else {
            let knob_gain = ((b_idx as f32) * 0.013).sin().mul_add(0.15, 0.20);
            engine.set_control_value(0, black_box(knob_gain));
        }

        let t0 = Instant::now();
        engine.process_block(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[num_blocks / 2];
    let p99 = latencies_us[(num_blocks * 99) / 100];
    let p999 = latencies_us[(num_blocks * 999) / 1_000];
    let max_us = latencies_us[num_blocks - 1];
    let deadline_us = (128.0 / 48_000.0) * 1_000_000.0; // 2,666.67 us
    let max_budget_pct = (max_us / deadline_us) * 100.0;

    println!(
        "| {:<24} | {:>8.2} us | {:>8.2} us | {:>8.2} us | {:>8.2} us | {:>8.2}% |",
        label, p50, p99, p999, max_us, max_budget_pct
    );
}

#[cfg_attr(feature = "hotpath", hotpath::main(percentiles = [50, 95, 99], limit = 0))]
fn main() {
    let v3_only = std::env::args().any(|a| a == "v3_core");
    if !v3_only {
        let container_full = make_bench_container();

        let iters = 1_000;
        let crypto = DefaultCryptoProvider::default();
        let t_load = Instant::now();
        for _ in 0..iters {
            let v = DnamContainerView::from_bytes(black_box(container_full.as_bytes()), &crypto)
                .unwrap();
            black_box(v.preamble.header_crc32c);
        }
        let load_us = (t_load.elapsed().as_nanos() as f64) / (iters as f64 * 1000.0);

        println!(
            "=========================================================================================="
        );
        println!(
            " DNAM Real-Time SIMD Engine Telemetry Benchmark (48 kHz Stereo, 128-Sample Blocks)"
        );
        println!(" #![no_std] serde + alloc Container Load + CRC32C Verification: {load_us:.2} us");
        println!(
            "=========================================================================================="
        );
        println!(
            "| {:<32} | {:<11} | {:>13} | {:>11} | {:>8} |",
            "Tier & Subsystem Profile", "Backend", "Latency/Frame", "x Real-Time", "CPU Load"
        );
        println!(
            "|----------------------------------|-------------|---------------|-------------|----------|"
        );

        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Lite, "Lite (C=4)");
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Full, "Full (C=8)");
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Pro, "Pro (C=12)");
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Ultra, "Ultra (C=16)");
        bench_tier_bounce::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Ultra,
            "Ultra (C=16, 4x Offline Bounce)",
        );

        #[cfg(target_arch = "aarch64")]
        {
            use dnam_engine::math::ArmSmeBackend;
            bench_tier::<ArmSmeBackend>(
                container_full.as_bytes(),
                Tier::Full,
                "Full (C=8, ARM SME)",
            );
        }

        bench_tier::<ScalarBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=8, Scalar Baseline)",
        );
        println!(
            "==========================================================================================\n"
        );

        println!(
            "=========================================================================================="
        );
        println!(
            " Real-Time WCET & Tail-Latency Jitter Audit (2,000 x 128-Sample Blocks = 2,666.67 us Budget)"
        );
        println!(" Active Continuous Knob Modulation + Stage IV Tank Switch Crossfading");
        println!(
            "=========================================================================================="
        );
        println!(
            "| {:<24} | {:>11} | {:>11} | {:>11} | {:>11} | {:>9} |",
            "Tier Profile (SIMD)",
            "P50 Block",
            "P99 Block",
            "P99.9 Block",
            "Max Block",
            "Max/Budget"
        );
        println!(
            "|--------------------------|-------------|-------------|-------------|-------------|-----------|"
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Lite,
            "Lite (C=4, Modulated)",
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=8, Modulated)",
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Pro,
            "Pro (C=12, Modulated)",
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Ultra,
            "Ultra (C=16, Modulated)",
        );
        println!(
            "==========================================================================================\n"
        );
    }

    bench_v3_core_suite();
}

fn make_v3_bench_model(tier: Tier, os2x: bool, mono_out: bool) -> V3ColdTierModel {
    let mut m = V3ColdTierModel::identity_for_tier(tier, os2x, mono_out);
    m.stage0.xo_low_gain = 1.05;
    m.stage0.xo_high_gain = 0.95;
    m.stage0.clean_sub_mix = 0.05;
    m.stage0.op_p_sub_mix = 0.12;
    m.stage0.gate_thresh = 0.002;
    m.stage0.gate_hyst = 0.001;
    m.stage0.gate_inv_knee = 100.0;
    m.stage0.gate_floor = 0.1;
    m.os_params.up_pre_emph = 0.14;
    m.os_params.dec_aa_mix = 0.85;
    m.os_params.up_aa_mix = 0.85;
    m.ctrl.w_env_fc_oct[0] = 0.4;
    m.ctrl.w_tone_fc_oct[0] = 0.6;
    m.ctrl.w_tank_psi[0] = 0.25;
    m.ctrl.w_film_gain[0][0][0] = 0.35;
    m.tone_svf.h0_proj_w = [0.2, -0.1, 0.15, 0.05];
    m.tone_svf.s1_proj_w = 0.3;
    m.tone_svf.lp_mix = 0.35;
    m.tone_svf.bp_mix = 0.2;
    m.tone_svf.cab_inject_w = 0.2;
    m.tone_svf.b1_inject_w = 0.15;
    m.stage4.wet_mix = 0.15;
    m.stage4.g_loop = 0.85;
    m.stage4.lfo_rate_hz = 1.6;
    m.stage4.lfo_depth_samples = 6.0;
    m.stage4.lfo_stereo_phase = core::f32::consts::FRAC_PI_2;
    m.svf.env_oct_sens = 1.8;
    m.svf.env_comp = 2.5;
    m.svf.det_hp_mix = 0.35;
    m.svf.sec2_bp_feed = 0.25;
    m.svf.sec2_hp_feed = 0.15;
    m.svf.stage2_feed_taps[..7].copy_from_slice(&[0.7, 0.15, 0.25, -0.1, 0.15, 0.1, 0.05]);
    m.readout.svf_direct_taps_l[..7].copy_from_slice(&[0.05, 0.02, -0.01, 0.01, 0.03, 0.01, -0.02]);
    m.readout.svf_direct_taps_r[..7].copy_from_slice(&[0.04, 0.03, -0.02, 0.02, 0.02, -0.01, 0.01]);
    m.feat_params.pinch_k = 180.0;
    m.feat_params.slow_det_comp = 6.0;
    for b in 0..NUM_GRU_BLOCKS {
        for dst in 0..MAX_BLOCK_LANES {
            m.blocks[b].w_in_n[0][dst] = 0.65;
            m.blocks[b].w_in_z[0][dst] = 0.25;
            m.blocks[b].w_in_r[0][dst] = 0.15;
            m.blocks[b].b_n[dst] = -0.25;
            m.blocks[b].b_z[dst] = -0.60;
            m.blocks[b].b_r[dst] = 1.25;
            m.blocks[b].k_sag[dst] = 0.50;
            m.blocks[b].k_quad[dst] = 0.08;
            m.blocks[b].w_rec_n[dst][dst] = 0.15;
            if b > 0 {
                m.blocks[b].w_prev_n[dst][dst] = 0.45;
            }
        }
        let lane = flat_gru_lane_index(b, 0, 0);
        m.readout.pre_cab_w[0][lane] = 0.25;
        m.readout.pre_cab_w[1][lane] = 0.25;
        m.readout.direct_s2_w[0][lane] = 0.10;
        m.readout.direct_s2_w[1][lane] = 0.10;
    }
    for i in 0..tier_fir_segments(tier) * 16 {
        m.readout.fir_l[i] = 0.35 * 0.90_f32.powi(i as i32);
        m.readout.fir_r[i] = 0.32 * 0.88_f32.powi(i as i32);
    }
    for k in 0..tier_kautz_banks(tier) * 4 {
        m.readout.kautz_c_l[k] = (0.04, 0.02);
        m.readout.kautz_c_r[k] = (0.035, -0.015);
    }
    m
}

fn bench_v3_row(tier: Tier, os2x: bool, mono_out: bool, label: &str) {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let model = make_v3_bench_model(tier, os2x, mono_out);
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&model);
    let in_l = [0.22_f32; 128];
    let in_r = [-0.18_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    for _ in 0..100 {
        inst.process_block_stereo(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    let blocks = 2_000_usize;
    let total_samples = blocks * 128;
    let mut best_elapsed_ns = f64::INFINITY;
    for _pass in 0..3 {
        let t0 = Instant::now();
        for _ in 0..blocks {
            inst.process_block_stereo(
                black_box(&in_l),
                black_box(&in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        }
        best_elapsed_ns = best_elapsed_ns.min(t0.elapsed().as_nanos() as f64);
    }
    let elapsed_ns = best_elapsed_ns;
    black_box((out_l[0], out_r[0]));
    let ns_per_sample = elapsed_ns / (total_samples as f64);
    let budget_ns = 1_000_000_000.0 / 48_000.0;
    let x_rt = budget_ns / ns_per_sample;
    let cpu_pct = (ns_per_sample / budget_ns) * 100.0;
    let hot_b = V3HotCore::<DefaultMathBackend>::active_hot_bytes(tier, os2x);
    println!(
        "| {:<32} | {:>7} B | {:>10.2} ns | {:>10.1}x | {:>7.2}% |",
        label, hot_b, ns_per_sample, x_rt, cpu_pct
    );
}

fn print_wcet_stats(label: &str, latencies_us: &mut [f64]) {
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let num_blocks = latencies_us.len();
    let p50 = latencies_us[num_blocks / 2];
    let p99 = latencies_us[(num_blocks * 99) / 100];
    let p999 = latencies_us[(num_blocks * 999) / 1_000];
    let max_us = latencies_us[num_blocks - 1];
    let deadline_us = (128.0 / 48_000.0) * 1_000_000.0;
    let max_budget_pct = (max_us / deadline_us) * 100.0;
    println!(
        "| {:<28} | {:>8.2} us | {:>8.2} us | {:>8.2} us | {:>8.2} us | {:>8.2}% |",
        label, p50, p99, p999, max_us, max_budget_pct
    );
}

fn bench_v3_wcet_row(tier: Tier, os2x: bool, mono_out: bool, label: &str) {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let model = make_v3_bench_model(tier, os2x, mono_out);
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&model);
    let in_l = [0.22_f32; 128];
    let in_r = [-0.18_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    for _ in 0..64 {
        inst.process_block_stereo(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    let num_blocks = 2_000_usize;
    let mut latencies_us = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let t0 = Instant::now();
        inst.process_block_stereo(
            black_box(&in_l),
            black_box(&in_r),
            black_box(&mut out_l),
            black_box(&mut out_r),
        );
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
    }
    black_box((out_l[0], out_r[0]));
    print_wcet_stats(label, &mut latencies_us);
}

fn bench_v3_chain_wcet_row(tier: Tier, os2x: bool, instances: usize, label: &str) {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let model = make_v3_bench_model(tier, os2x, false);
    let mut chain: Vec<V3CoreInstance<DefaultMathBackend>> = (0..instances)
        .map(|_| V3CoreInstance::<DefaultMathBackend>::from_cold_model(&model))
        .collect();
    let in_l = [0.22_f32; 128];
    let in_r = [-0.18_f32; 128];
    let mut buf_a_l;
    let mut buf_a_r;
    let (mut buf_b_l, mut buf_b_r) = ([0.0_f32; 128], [0.0_f32; 128]);
    for _ in 0..64 {
        buf_a_l = in_l;
        buf_a_r = in_r;
        for inst in &mut chain {
            inst.process_block_stereo(&buf_a_l, &buf_a_r, &mut buf_b_l, &mut buf_b_r);
            buf_a_l = buf_b_l;
            buf_a_r = buf_b_r;
        }
    }
    let num_blocks = 2_000_usize;
    let mut latencies_us = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        buf_a_l = black_box(in_l);
        buf_a_r = black_box(in_r);
        let t0 = Instant::now();
        for inst in &mut chain {
            inst.process_block_stereo(&buf_a_l, &buf_a_r, &mut buf_b_l, &mut buf_b_r);
            buf_a_l = buf_b_l;
            buf_a_r = buf_b_r;
        }
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
        black_box((buf_a_l[0], buf_a_r[0]));
    }
    print_wcet_stats(label, &mut latencies_us);
}

fn bench_v3_core_suite() {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let core_bytes = core::mem::size_of::<V3HotCore<DefaultMathBackend>>();
    println!(
        "=========================================================================================="
    );
    println!(
        " DNAM v3 Cache-Coherent Hot Core Throughput (48 kHz, 128-Sample Blocks, sizeof(V3HotCore)={core_bytes} B)"
    );
    println!(
        "=========================================================================================="
    );
    println!(
        "| {:<32} | {:>9} | {:>13} | {:>11} | {:>8} |",
        "v3 Tier & OS Mode", "HotBytes", "Latency/Frame", "x Real-Time", "CPU Load"
    );
    println!(
        "|----------------------------------|-----------|---------------|-------------|----------|"
    );
    bench_v3_row(Tier::Lite, false, false, "v3 Lite ([1,1,1], 1x, Stereo)");
    bench_v3_row(Tier::Full, false, false, "v3 Full ([2,1,1], 1x, Stereo)");
    bench_v3_row(Tier::Pro, true, false, "v3 Pro ([2,2,1], 2x OS, Stereo)");
    bench_v3_row(
        Tier::Ultra,
        true,
        false,
        "v3 Ultra ([3,2,1], 2x OS, Stereo)",
    );
    println!(
        "==========================================================================================\n"
    );
    println!(
        "| {:<28} | {:>11} | {:>11} | {:>11} | {:>11} | {:>9} |",
        "v3 WCET (2000 x 128)", "P50 Block", "P99 Block", "P99.9 Block", "Max Block", "Max/Budget"
    );
    println!(
        "|------------------------------|-------------|-------------|-------------|-------------|-----------|"
    );
    bench_v3_wcet_row(Tier::Lite, false, false, "v3 Lite (1x, Stereo)");
    bench_v3_wcet_row(Tier::Full, false, false, "v3 Full (1x, Stereo)");
    bench_v3_wcet_row(Tier::Pro, true, false, "v3 Pro (2x OS, Stereo)");
    bench_v3_wcet_row(Tier::Ultra, true, false, "v3 Ultra (2x OS, Stereo)");
    println!(
        "|------------------------------|-------------|-------------|-------------|-------------|-----------|"
    );
    bench_v3_chain_wcet_row(Tier::Lite, false, 14, "14x Lite ([1,1,1], 1x)");
    bench_v3_chain_wcet_row(Tier::Full, false, 8, "8x Full ([2,1,1], 1x)");
    bench_v3_chain_wcet_row(Tier::Pro, true, 4, "4x Pro ([2,2,1], 2x)");
    bench_v3_chain_wcet_row(Tier::Ultra, true, 2, "2x Ultra ([3,2,1], 2x)");
    println!(
        "=========================================================================================="
    );
}
