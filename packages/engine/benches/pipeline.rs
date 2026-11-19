//! End-to-End Pipeline Criterion Benchmarks (`benches/pipeline.rs`).
//!
//! Benchmarks `DnamEngine` block processing across slimmable tiers (`Lite`, `Full`, `Pro`, `Ultra`),
//! math backends (`DefaultMathBackend`, `ScalarBackend`, `ArmSmeBackend`), I/O topologies,
//! DAW buffer sizes (`32..512`), subnormal/denormal stress, and multi-instance track scaling.

use std::collections::BTreeMap;
use std::hint::black_box;

use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
use dnam_engine::engine::EngineBackend;
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
use dnam_engine::stages::sona_core::{SonaControlBusParams, SonaMatryoshkaModel};
use dnam_engine::stages::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::{DnamEngine, Tier};
use hotpath::criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use zerocopy::{FromBytes, byteorder::little_endian::F32};

fn populate_active_ctrl(ctrl: &mut SonaControlBusParams) {
    ctrl.w_svf_g_oct[0] = 0.35;
    ctrl.w_svf_k_delta[0] = 0.12;
    ctrl.w_tank_psi[0] = 0.20;
    ctrl.w_adaa_drive_scale[0][0] = 0.25;
    ctrl.cp1_dir_count = 1;
    ctrl.cp1_dir_flags = 0;
    ctrl.cp1_dir_entries[0] = [0, 2, 0xFF];
    if let Ok(base_words) = <[F32]>::mut_from_bytes(&mut ctrl.cp1_base) {
        for l in 0..2_usize {
            for d in 0..8 {
                base_words[l * 8 + d] = F32::new(0.12 - 0.03 * (((l + d) % 4) as f32));
            }
            base_words[16 + l] = F32::new(0.25);
        }
        for d in 0..4 {
            base_words[18 + d] = F32::new(0.20 + 0.02 * (d as f32));
        }
    }
    if let Ok(ultra_words) = <[F32]>::mut_from_bytes(&mut ctrl.cp1_ultra) {
        for l in 0..2_usize {
            for d in 0..8 {
                ultra_words[l * 8 + d] = F32::new(0.06);
            }
        }
        for d in 0..4 {
            ultra_words[16 + d] = F32::new(0.10);
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
    header.metadata.name = "Pipeline Benchmark Model".to_string();
    header.controls_schema = vec![ControlSchemaDef {
        id: "gain".to_string(),
        name: "Gain".to_string(),
        kind: ControlKind::Range,
        unit: Some("%".to_string()),
        lobatto_degree: Some(2),
        anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
        declared_range: Some([0.0, 1.0]),
        safe_clamped_range: Some([0.0, 1.0]),
        default: serde_json::json!(0.85),
        conditional_bounds: Vec::new(),
        represented_switch_options: None,
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        ui_hints: ControlUiHints::default(),
        calibration: None,
        user_metadata: BTreeMap::new(),
    }];

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    populate_active_ctrl(&mut model.ctrl);
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

fn bench_tier_profile<B: EngineBackend>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    container: &[u8],
    tier: Tier,
    profile_name: &str,
) {
    let mut engine = DnamEngine::<B>::from_bytes(container, tier, 48_000).unwrap();
    engine.set_control_immediate(0, 0.85);
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    let id = BenchmarkId::new(B::KIND.name(), profile_name);
    group.bench_function(id, |b| {
        b.iter(|| {
            engine.process_block(
                black_box(&in_l),
                black_box(&in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
}

fn bench_process_block_tiers_and_backends(c: &mut Criterion) {
    let container_full = make_bench_container();

    let mut group = c.benchmark_group("process_block_128");
    group.throughput(Throughput::Elements(128));

    for (profile_name, bytes, tier) in [
        ("lite", container_full.as_bytes(), Tier::Lite),
        ("full", container_full.as_bytes(), Tier::Full),
        ("pro", container_full.as_bytes(), Tier::Pro),
        ("ultra", container_full.as_bytes(), Tier::Ultra),
    ] {
        bench_tier_profile::<DefaultMathBackend>(&mut group, bytes, tier, profile_name);
    }

    bench_tier_profile::<ScalarBackend>(&mut group, container_full.as_bytes(), Tier::Full, "full");

    #[cfg(target_arch = "aarch64")]
    {
        use dnam_engine::math::ArmSmeBackend;
        bench_tier_profile::<ArmSmeBackend>(
            &mut group,
            container_full.as_bytes(),
            Tier::Full,
            "full",
        );
    }

    group.finish();
}

fn bench_io_topologies_and_buffer_sizes(c: &mut Criterion) {
    let container = make_bench_container();

    let mut topo_group = c.benchmark_group("io_topologies_full_128");
    topo_group.throughput(Throughput::Elements(128));
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    topo_group.bench_function("mono_to_mono", |b| {
        b.iter(|| {
            engine.process_mono(black_box(&in_l), black_box(&mut out_l));
        });
    });
    topo_group.bench_function("mono_to_stereo", |b| {
        b.iter(|| {
            engine.process_mono_to_stereo(
                black_box(&in_l),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
    topo_group.bench_function("stereo_to_stereo", |b| {
        b.iter(|| {
            engine.process_block(
                black_box(&in_l),
                black_box(&in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
    topo_group.bench_function("stereo_in_place", |b| {
        b.iter(|| {
            engine.process_block_in_place(black_box(&mut out_l), black_box(&mut out_r));
        });
    });
    topo_group.finish();

    let mut buf_group = c.benchmark_group("daw_buffer_sizes_full");
    let in_l_max = [0.15_f32; 512];
    let in_r_max = [-0.15_f32; 512];
    let mut out_l_max = [0.0_f32; 512];
    let mut out_r_max = [0.0_f32; 512];
    for frames in [32_usize, 64, 128, 256, 512] {
        buf_group.throughput(Throughput::Elements(frames as u64));
        buf_group.bench_with_input(BenchmarkId::from_parameter(frames), &frames, |b, &n| {
            b.iter(|| {
                engine.set_control_value(0, black_box(0.65));
                engine.process_block(
                    black_box(&in_l_max[..n]),
                    black_box(&in_r_max[..n]),
                    black_box(&mut out_l_max[..n]),
                    black_box(&mut out_r_max[..n]),
                );
            });
        });
    }
    buf_group.finish();
}

fn bench_denormal_and_multi_instance(c: &mut Criterion) {
    let container = make_bench_container();

    let mut denorm_group = c.benchmark_group("denormal_subnormal_stress");
    denorm_group.throughput(Throughput::Elements(128));
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), Tier::Full, 48_000)
            .unwrap();
    let mut subnormal_in_l = [0.0_f32; 128];
    let mut subnormal_in_r = [0.0_f32; 128];
    for i in 0..128 {
        let val = f32::from_bits(1 + (i as u32));
        subnormal_in_l[i] = val;
        subnormal_in_r[i] = -val;
    }
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    denorm_group.bench_function("full_128_subnormal_decay", |b| {
        b.iter(|| {
            engine.process_block(
                black_box(&subnormal_in_l),
                black_box(&subnormal_in_r),
                black_box(&mut out_l),
                black_box(&mut out_r),
            );
        });
    });
    denorm_group.finish();

    let mut multi_group = c.benchmark_group("multi_instance_track_scaling_full_128");
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    for num_tracks in [1_usize, 8, 16] {
        multi_group.throughput(Throughput::Elements((num_tracks * 128) as u64));
        let mut tracks: Vec<DnamEngine<'_, DefaultMathBackend>> = (0..num_tracks)
            .map(|_| {
                DnamEngine::<DefaultMathBackend>::from_bytes(
                    container.as_bytes(),
                    Tier::Full,
                    48_000,
                )
                .unwrap()
            })
            .collect();
        multi_group.bench_with_input(
            BenchmarkId::from_parameter(num_tracks),
            &num_tracks,
            |b, _| {
                b.iter(|| {
                    for trk in tracks.iter_mut() {
                        trk.process_block(
                            black_box(&in_l),
                            black_box(&in_r),
                            black_box(&mut out_l),
                            black_box(&mut out_r),
                        );
                    }
                });
            },
        );
    }
    multi_group.finish();
}

fn bench_serial_chain_rack(c: &mut Criterion) {
    let container = make_bench_container();
    let mut group = c.benchmark_group("serial_chain_rack_128");
    let in_l = [0.15_f32; 128];
    let in_r = [-0.15_f32; 128];
    let mut ping_l = [0.0_f32; 128];
    let mut ping_r = [0.0_f32; 128];
    let mut pong_l = [0.0_f32; 128];
    let mut pong_r = [0.0_f32; 128];

    for &(label, tier, count) in &[
        ("lite_14x", Tier::Lite, 14_usize),
        ("full_8x", Tier::Full, 8_usize),
        ("pro_4x", Tier::Pro, 4_usize),
        ("ultra_2x", Tier::Ultra, 2_usize),
    ] {
        group.throughput(Throughput::Elements((count * 128) as u64));
        let mut chain: Vec<DnamEngine<'_, DefaultMathBackend>> = (0..count)
            .map(|_| {
                DnamEngine::<DefaultMathBackend>::from_bytes(container.as_bytes(), tier, 48_000)
                    .unwrap()
            })
            .collect();
        group.bench_function(label, |b| {
            b.iter(|| {
                chain[0].process_block(
                    black_box(&in_l),
                    black_box(&in_r),
                    black_box(&mut ping_l),
                    black_box(&mut ping_r),
                );
                for (idx, inst) in chain.iter_mut().enumerate().skip(1) {
                    if idx & 1 == 1 {
                        inst.process_block(
                            black_box(&ping_l),
                            black_box(&ping_r),
                            black_box(&mut pong_l),
                            black_box(&mut pong_r),
                        );
                    } else {
                        inst.process_block(
                            black_box(&pong_l),
                            black_box(&pong_r),
                            black_box(&mut ping_l),
                            black_box(&mut ping_r),
                        );
                    }
                }
                if count & 1 == 1 {
                    black_box((&ping_l, &ping_r));
                } else {
                    black_box((&pong_l, &pong_r));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_process_block_tiers_and_backends,
    bench_io_topologies_and_buffer_sizes,
    bench_denormal_and_multi_instance,
    bench_serial_chain_rack
);
criterion_main!(benches);
