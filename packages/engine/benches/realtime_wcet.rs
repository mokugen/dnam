//! Non-Criterion Real-Time Worst-Case Execution Time (WCET), Tail-Latency Jitter (`P50`/`P99`/`P99.9`/`Max`),
//! and `x Real-Time @ 48 kHz` / `% Single-Core CPU` Benchmark (`benches/realtime_wcet.rs`).
//!
//! Run with:
//!   `cargo bench -p dnam-engine --bench realtime_wcet`

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, StructuredConditionalBound, build_dnam_container,
};
use dnam_engine::engine::EngineBackend;
use dnam_engine::graph::sona_core::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, SonaColdTierModel, SonaControlBusParams,
    SonaHotSlab, SonaInstance, SonaMatryoshkaModel, tier_fir_segments, tier_kautz_banks,
};
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, TempoSubdivision,
};
use dnam_engine::{DnamEngine, Tier};
use zerocopy::{FromBytes, byteorder::little_endian::F32};

fn populate_active_ctrl(ctrl: &mut SonaControlBusParams) {
    ctrl.w_svf_g_oct[0] = 0.4;
    ctrl.w_svf_k_delta[0] = 0.15;
    ctrl.w_tank_psi[0] = 0.25;
    ctrl.w_adaa_drive_scale[0][0] = 0.35;
    ctrl.cp1_dir_count = 2;
    ctrl.cp1_dir_flags = 0;
    ctrl.cp1_dir_entries[0] = [0, 2, 0xFF];
    ctrl.cp1_dir_entries[1] = [1, (1 << 4) | 1, 0];
    if let Ok(base_words) = <[F32]>::mut_from_bytes(&mut ctrl.cp1_base) {
        let mut cursor = 0usize;
        for &lanes in &[2_usize, 1] {
            for l in 0..lanes {
                for d in 0..8 {
                    base_words[cursor + l * 8 + d] = F32::new(0.12 - 0.03 * (((l + d) % 4) as f32));
                }
            }
            for l in 0..lanes {
                base_words[cursor + 8 * lanes + l] = F32::new(0.25);
            }
            for d in 0..4 {
                base_words[cursor + 9 * lanes + d] = F32::new(0.20 + 0.02 * (d as f32));
            }
            cursor += 9 * lanes + 4;
        }
    }
    if let Ok(ultra_words) = <[F32]>::mut_from_bytes(&mut ctrl.cp1_ultra) {
        let mut cursor = 0usize;
        for &lanes in &[2_usize, 1] {
            for l in 0..lanes {
                for d in 0..8 {
                    ultra_words[cursor + l * 8 + d] = F32::new(0.06);
                }
            }
            for d in 0..4 {
                ultra_words[cursor + 8 * lanes + d] = F32::new(0.10);
            }
            cursor += 8 * lanes + 4;
        }
    }
    for r in 0..4 {
        for c in 0..8 {
            ctrl.u_int_base[r][c] = 0.05;
        }
    }
    for r in 0..16 {
        for c in 0..16 {
            ctrl.w_ctrl[r][c] = if c < 8 { 0.04 } else { 0.0 };
        }
    }
}

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = "WCET Benchmark Model".to_string();
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
            declared_options: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            active_options: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
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
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;
    model.stage0.clean_sub_mix = 0.05;
    populate_active_ctrl(&mut model.ctrl);
    for g in 0..MAX_SONA_GROUPS {
        for l in 0..4 {
            model.groups[g].in_x_weight[l] = if l == 0 { 0.65 } else { 0.10 };
            model.groups[g].adaa_drive[l] = 0.6;
            model.groups[g].lru_alpha_rel[l] = 0.90;
        }
    }
    for tile in &mut model.tiles[..MAX_SONA_TILES] {
        for col in 0..4 {
            tile.w_odd_4x4[col][col] = 0.15;
        }
    }
    for m in 0..MAX_KAUTZ_POLES {
        model.kautz_poles[m] = (0.90, 0.15 + 0.04 * ((m & 3) as f32));
    }
    for s4 in &mut model.stage4 {
        s4.wet_mix = 0.15;
        s4.g_loop = 0.85;
        s4.lfo_rate_hz = 1.6;
        s4.lfo_depth_samples = 6.0;
        s4.lfo_stereo_phase = core::f32::consts::FRAC_PI_2;
    }
    pack_sona_matryoshka_to_arena(&model, &mut arena);
    let raw = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn bench_tier<B: EngineBackend>(container: &[u8], tier: Tier, label: &str) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    engine.set_control_immediate(0, 0.35);
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
    engine.set_control_immediate(0, 0.35);
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

        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Lite, "Lite (C=16)");
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Full, "Full (C=24)");
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Pro, "Pro (C=32)");
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Ultra, "Ultra (C=40)");
        bench_tier_bounce::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Ultra,
            "Ultra (C=40, 4x Offline Bounce)",
        );

        #[cfg(target_arch = "aarch64")]
        {
            use dnam_engine::math::ArmSmeBackend;
            bench_tier::<ArmSmeBackend>(
                container_full.as_bytes(),
                Tier::Full,
                "Full (C=24, ARM SME)",
            );
        }

        bench_tier::<ScalarBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=24, Scalar Baseline)",
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
        println!(" Active Continuous Knob Modulation + Spatial Tank Switch Crossfading");
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
            "Lite (C=16, Modulated)",
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=24, Modulated)",
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Pro,
            "Pro (C=32, Modulated)",
        );
        bench_wcet_tail_latency::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Ultra,
            "Ultra (C=40, Modulated)",
        );
        println!(
            "==========================================================================================\n"
        );
    }

    bench_sona_core_suite();
}

fn make_sona_bench_model(tier: Tier, os2x: bool, mono_out: bool) -> SonaColdTierModel {
    let mut m = SonaColdTierModel::identity_for_tier(tier, os2x, mono_out);
    m.stage0.xo_low_gain = 1.05;
    m.stage0.xo_high_gain = 0.95;
    m.stage0.clean_sub_mix = 0.05;
    m.stage0.gate_thresh = 0.002;
    m.stage0.gate_hyst = 0.001;
    m.stage0.gate_inv_knee = 100.0;
    m.stage0.gate_floor = 0.1;
    m.os_params.up_pre_emph = 0.14;
    m.os_params.dec_aa_mix = 0.85;
    m.os_params.up_aa_mix = 0.85;
    populate_active_ctrl(&mut m.ctrl);
    m.stage4.wet_mix = 0.15;
    m.stage4.g_loop = 0.85;
    m.stage4.lfo_rate_hz = 1.6;
    m.stage4.lfo_depth_samples = 6.0;
    m.stage4.lfo_stereo_phase = core::f32::consts::FRAC_PI_2;
    for g in 0..MAX_SONA_GROUPS {
        for l in 0..4 {
            m.groups[g].in_x_weight[l] = if l == 0 { 0.65 } else { 0.12 };
            m.groups[g].adaa_drive[l] = 0.60;
            m.groups[g].wavefolder_mix[l] = 0.08;
            m.groups[g].sag_depth[l] = 0.20;
            m.groups[g].lru_alpha_rel[l] = 0.90;
            m.groups[g].lru_kappa[l] = 0.25;
        }
        for l in 0..4 {
            let lane = g * 4 + l;
            m.readout.pre_cab_w_odd[0][lane] = 0.25;
            m.readout.pre_cab_w_odd[1][lane] = 0.25;
            m.readout.direct_w_odd[0][lane] = 0.10;
            m.readout.direct_w_odd[1][lane] = 0.10;
        }
    }
    for tile in &mut m.tiles[..MAX_SONA_TILES] {
        for col in 0..4 {
            tile.w_odd_4x4[col][col] = 0.15;
        }
    }
    for i in 0..tier_fir_segments(tier) * 16 {
        m.readout.fir_l[i] = 0.35 * 0.90_f32.powi(i as i32);
        m.readout.fir_r[i] = 0.32 * 0.88_f32.powi(i as i32);
    }
    for k in 0..tier_kautz_banks(tier) * 4 {
        m.kautz_poles[k] = (0.90, 0.15 + 0.04 * ((k & 3) as f32));
        m.readout.kautz_c_l[k] = (0.04, 0.02);
        m.readout.kautz_c_r[k] = (0.035, -0.015);
    }
    m
}

fn bench_sona_row(tier: Tier, os2x: bool, mono_out: bool, label: &str) {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let model = make_sona_bench_model(tier, os2x, mono_out);
    let mut inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&model);
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
    let hot_b = SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(tier, os2x);
    let tier_ceiling = tier.expected_arena_bytes();
    assert!(
        hot_b <= tier_ceiling,
        "ActiveHot {} B exceeded {} L1D ceiling {} B",
        hot_b,
        tier.as_str(),
        tier_ceiling
    );
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

fn bench_sona_wcet_row(tier: Tier, os2x: bool, mono_out: bool, label: &str) {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let model = make_sona_bench_model(tier, os2x, mono_out);
    let mut inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&model);
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

fn bench_sona_chain_wcet_row(tier: Tier, os2x: bool, instances: usize, label: &str) {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let model = make_sona_bench_model(tier, os2x, false);
    let mut chain: Vec<SonaInstance<DefaultMathBackend>> = (0..instances)
        .map(|_| SonaInstance::<DefaultMathBackend>::from_cold_model(&model))
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

fn bench_sona_core_suite() {
    let _ftz = dnam_engine::FtzDazGuard::new();
    let core_bytes = core::mem::size_of::<SonaHotSlab<DefaultMathBackend>>();
    println!(
        "=========================================================================================="
    );
    println!(
        " DNAM SONA v3.0 Cache-Coherent Hot Slab Throughput (48 kHz, 128-Sample Blocks, sizeof(SonaHotSlab)={core_bytes} B)"
    );
    println!(
        "=========================================================================================="
    );
    println!(
        "| {:<32} | {:>9} | {:>13} | {:>11} | {:>8} |",
        "SONA Tier & OS Mode", "HotBytes", "Latency/Frame", "x Real-Time", "CPU Load"
    );
    println!(
        "|----------------------------------|-----------|---------------|-------------|----------|"
    );
    bench_sona_row(Tier::Lite, false, false, "SONA Lite ([1,1,1,1], 1x, C=16)");
    bench_sona_row(Tier::Full, true, false, "SONA Full ([2,2,1,1], 2x, C=24)");
    bench_sona_row(Tier::Pro, true, false, "SONA Pro ([3,2,2,1], 2x, C=32)");
    bench_sona_row(Tier::Ultra, true, false, "SONA Ultra ([4,3,2,1], 2x, C=40)");
    println!(
        "==========================================================================================\n"
    );
    println!(
        "| {:<28} | {:>11} | {:>11} | {:>11} | {:>11} | {:>9} |",
        "SONA WCET (2000 x 128)",
        "P50 Block",
        "P99 Block",
        "P99.9 Block",
        "Max Block",
        "Max/Budget"
    );
    println!(
        "|------------------------------|-------------|-------------|-------------|-------------|-----------|"
    );
    bench_sona_wcet_row(Tier::Lite, false, false, "SONA Lite (1x, C=16)");
    bench_sona_wcet_row(Tier::Full, true, false, "SONA Full (2x, C=24)");
    bench_sona_wcet_row(Tier::Pro, true, false, "SONA Pro (2x, C=32)");
    bench_sona_wcet_row(Tier::Ultra, true, false, "SONA Ultra (2x, C=40)");
    println!(
        "|------------------------------|-------------|-------------|-------------|-------------|-----------|"
    );
    bench_sona_chain_wcet_row(Tier::Lite, false, 14, "14x Lite ([1,1,1,1], 1x)");
    bench_sona_chain_wcet_row(Tier::Full, true, 8, "8x Full ([2,2,1,1], 2x)");
    bench_sona_chain_wcet_row(Tier::Pro, true, 4, "4x Pro ([3,2,2,1], 2x)");
    bench_sona_chain_wcet_row(Tier::Ultra, true, 2, "2x Ultra ([4,3,2,1], 2x)");
    println!(
        "=========================================================================================="
    );
}
