//! Non-Criterion Real-Time Worst-Case Execution Time (WCET), Tail-Latency Jitter (`P50`/`P99`/`P99.9`/`Max`),
//! and `x Real-Time @ 48 kHz` / `% Single-Core CPU` Benchmark (`benches/realtime_wcet.rs`).
//!
//! Run with:
//!   `cargo bench -p dnam-engine --bench realtime_wcet`

#[path = "../tests/sona_test_support/arena_builder.rs"]
mod arena_builder;

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use arena_builder::{build_sona_container_from_header_options, build_sona_header};
use dnam_engine::container::DnamContainerView;
use dnam_engine::engine::EngineBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{PresetDef, TempoSubdivision};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier, extract_dnam_bin};
use dnam_vm::graph::SonaHotSlab;
use dnam_vm::math::{DefaultMathBackend, ScalarBackend};
use dnam_vm::{SonaVm, sona_vm_process_block_default};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let mut header = build_sona_header();
    header.metadata.name = "WCET Benchmark Model".to_string();
    header.presets = vec![
        PresetDef {
            id: "preset_a".to_string(),
            name: "Preset A".to_string(),
            description: None,
            controls: BTreeMap::from([
                ("mode".to_string(), serde_json::json!("warm")),
                ("drive".to_string(), serde_json::json!(2.0)),
            ]),
            user_metadata: BTreeMap::new(),
        },
        PresetDef {
            id: "preset_b_disjoint".to_string(),
            name: "Preset B Disjoint".to_string(),
            description: None,
            controls: BTreeMap::from([
                ("mode".to_string(), serde_json::json!("bright")),
                ("drive".to_string(), serde_json::json!(8.0)),
            ]),
            user_metadata: BTreeMap::new(),
        },
    ];
    let (_arena, _dbin, raw) =
        build_sona_container_from_header_options(&mut header, Tier::Ultra, false, false, true, &[]);
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn print_throughput_row(label: &str, backend: &str, ns_per_sample: f64) {
    let budget_ns = 1_000_000_000.0 / 48_000.0;
    let x_rt = budget_ns / ns_per_sample;
    let cpu_pct = (ns_per_sample / budget_ns) * 100.0;
    println!(
        "| {:<32} | {:<11} | {:>10.2} ns | {:>10.1}x | {:>7.2}% |",
        label, backend, ns_per_sample, x_rt, cpu_pct
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

fn bench_tier<B: EngineBackend>(container: &[u8], tier: Tier, label: &str, offline: bool) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    engine.set_control_immediate(0, 6.5);
    if offline {
        engine.set_offline_bounce(true);
    }
    let block_size = 128_usize;
    let num_blocks = 375_usize;
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
    let ns_per_sample = (elapsed.as_nanos() as f64) / ((block_size * num_blocks) as f64);
    print_throughput_row(label, B::KIND.name(), ns_per_sample);
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
            let knob_gain = ((b_idx as f32) * 0.013).sin().mul_add(2.0, 5.0);
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
    print_wcet_stats(label, &mut latencies_us);
}

fn bench_sona_vm_row(dbin: &[u8], tier: Tier, os2x: bool, label: &str) {
    let _ftz = FtzDazGuard::new();
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(dbin, tier, 48_000.0).unwrap();
    let in_l = [0.22_f32; 128];
    let in_r = [-0.18_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    for _ in 0..100 {
        sona_vm_process_block_default(&mut vm, &in_l, &in_r, &mut out_l, &mut out_r).unwrap();
    }
    let blocks = 2_000_usize;
    let total_samples = blocks * 128;
    let mut best_elapsed_ns = f64::INFINITY;
    for _pass in 0..3 {
        let t0 = Instant::now();
        for _ in 0..blocks {
            sona_vm_process_block_default(
                &mut vm,
                black_box(&in_l),
                black_box(&in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            )
            .unwrap();
        }
        best_elapsed_ns = best_elapsed_ns.min(t0.elapsed().as_nanos() as f64);
    }
    black_box((out_l[0], out_r[0]));
    let ns_per_sample = best_elapsed_ns / (total_samples as f64);
    let budget_ns = 1_000_000_000.0 / 48_000.0;
    let x_rt = budget_ns / ns_per_sample;
    let cpu_pct = (ns_per_sample / budget_ns) * 100.0;
    let hot_b = SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(tier, os2x);
    let tier_ceiling = tier.expected_arena_bytes();
    assert!(
        hot_b <= tier_ceiling,
        "ActiveHot {hot_b} B exceeded {} L1D ceiling {tier_ceiling} B",
        tier.as_str(),
    );
    println!(
        "| {:<32} | {:>7} B | {:>10.2} ns | {:>10.1}x | {:>7.2}% |",
        label, hot_b, ns_per_sample, x_rt, cpu_pct
    );
}

fn bench_sona_vm_chain_wcet_row(dbin: &[u8], tier: Tier, instances: usize, label: &str) {
    let _ftz = FtzDazGuard::new();
    let mut chain: Vec<SonaVm<'static, DefaultMathBackend>> = (0..instances)
        .map(|_| SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(dbin, tier, 48_000.0).unwrap())
        .collect();
    let in_l = [0.22_f32; 128];
    let in_r = [-0.18_f32; 128];
    let (mut buf_b_l, mut buf_b_r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let num_blocks = 2_000_usize;
    let mut latencies_us = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let mut buf_a_l = black_box(in_l);
        let mut buf_a_r = black_box(in_r);
        let t0 = Instant::now();
        for vm in &mut chain {
            sona_vm_process_block_default(vm, &buf_a_l, &buf_a_r, &mut buf_b_l, &mut buf_b_r)
                .unwrap();
            buf_a_l = buf_b_l;
            buf_a_r = buf_b_r;
        }
        latencies_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
        black_box((buf_a_l[0], buf_a_r[0]));
    }
    print_wcet_stats(label, &mut latencies_us);
}

#[cfg_attr(feature = "hotpath", hotpath::main(percentiles = [50, 95, 99], limit = 0))]
fn main() {
    let container_full = make_bench_container();
    let dbin = extract_dnam_bin(container_full.as_bytes()).unwrap();
    let v3_only = std::env::args().any(|a| a == "v3_core");

    if !v3_only {
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
        println!(" #![no_std] Container Load + CRC32C Verification: {load_us:.2} us");
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
        bench_tier::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Lite,
            "Lite (C=16)",
            false,
        );
        bench_tier::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=24)",
            false,
        );
        bench_tier::<DefaultMathBackend>(container_full.as_bytes(), Tier::Pro, "Pro (C=32)", false);
        bench_tier::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Ultra,
            "Ultra (C=40)",
            false,
        );
        bench_tier::<DefaultMathBackend>(
            container_full.as_bytes(),
            Tier::Ultra,
            "Ultra (C=40, 4x Offline Bounce)",
            true,
        );
        #[cfg(target_arch = "aarch64")]
        {
            use dnam_vm::math::aarch64::ArmSmeBackend;
            bench_tier::<ArmSmeBackend>(
                container_full.as_bytes(),
                Tier::Full,
                "Full (C=24, ARM SME)",
                false,
            );
        }
        bench_tier::<ScalarBackend>(
            container_full.as_bytes(),
            Tier::Full,
            "Full (C=24, Scalar Baseline)",
            false,
        );
        println!(
            "==========================================================================================\n"
        );
        println!(
            "| {:<28} | {:>11} | {:>11} | {:>11} | {:>11} | {:>9} |",
            "Tier Profile (SIMD)",
            "P50 Block",
            "P99 Block",
            "P99.9 Block",
            "Max Block",
            "Max/Budget"
        );
        println!(
            "|------------------------------|-------------|-------------|-------------|-------------|-----------|"
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

    let core_bytes = core::mem::size_of::<SonaHotSlab<'_, DefaultMathBackend>>();
    println!(
        " DNAM SONA v3.0 Layer 0 VM Slab Throughput (48 kHz, 128-Sample Blocks, sizeof(SonaHotSlab)={core_bytes} B)"
    );
    println!(
        "| {:<32} | {:>9} | {:>13} | {:>11} | {:>8} |",
        "SONA Tier & OS Mode", "HotBytes", "Latency/Frame", "x Real-Time", "CPU Load"
    );
    println!(
        "|----------------------------------|-----------|---------------|-------------|----------|"
    );
    bench_sona_vm_row(dbin, Tier::Lite, false, "SONA Lite (1x, C=16)");
    bench_sona_vm_row(dbin, Tier::Full, true, "SONA Full (2x, C=24)");
    bench_sona_vm_row(dbin, Tier::Pro, true, "SONA Pro (2x, C=32)");
    bench_sona_vm_row(dbin, Tier::Ultra, true, "SONA Ultra (2x, C=40)");
    println!(
        "|------------------------------|-------------|-------------|-------------|-------------|-----------|"
    );
    bench_sona_vm_chain_wcet_row(dbin, Tier::Lite, 14, "14x Lite (1x)");
    bench_sona_vm_chain_wcet_row(dbin, Tier::Full, 8, "8x Full (2x)");
    bench_sona_vm_chain_wcet_row(dbin, Tier::Pro, 4, "4x Pro (2x)");
    bench_sona_vm_chain_wcet_row(dbin, Tier::Ultra, 2, "2x Ultra (2x)");
}
