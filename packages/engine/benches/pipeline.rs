//! End-to-End Layer 1 `DnamEngine` Pipeline Criterion Benchmarks (`benches/pipeline.rs`).

#[path = "../tests/sona_test_support/arena_builder.rs"]
mod arena_builder;

use std::hint::black_box;

use arena_builder::build_sona_extended_container;
use dnam_engine::engine::EngineBackend;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::{DnamEngine, Tier};
use dnam_vm::math::{DefaultMathBackend, ScalarBackend};
use hotpath::criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let raw = build_sona_extended_container(false, false);
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
    engine.set_control_immediate(0, 7.5);
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
        use dnam_vm::math::aarch64::ArmSmeBackend;
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
                engine.set_control_value(0, black_box(6.5));
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
