//! Container Framing, Embedded `.dnam-bin` Extraction, ZOH Rescaling & Provenance Criterion
//! Benchmarks (`benches/container_and_provenance.rs`).

#[path = "../tests/sona_test_support/arena_builder.rs"]
mod arena_builder;

use std::hint::black_box;

use arena_builder::build_sona_extended_container;
use dnam_engine::container::DnamContainerView;
use dnam_engine::memory::{AlignedBuffer, CacheLine64};
use dnam_engine::provenance::security::{
    WatermarkSeedScope, derive_watermark_seed_hmac_sha256,
    extract_cabinet_tier_fingerprint_component,
};
use dnam_engine::provenance::{DefaultCryptoProvider, extract_structural_hash_224d};
use dnam_engine::{DnamEngine, Tier, extract_dnam_bin};
use dnam_vm::ZohRateContext;
use dnam_vm::bin::DnamBinView;
use dnam_vm::math::DefaultMathBackend;
use hotpath::criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn make_bench_container() -> AlignedBuffer<CacheLine64> {
    let raw = build_sona_extended_container(false, false);
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn bench_container_and_zoh(c: &mut Criterion) {
    let container = make_bench_container();
    let crypto = DefaultCryptoProvider::default();
    let mut group = c.benchmark_group("container_and_zoh");
    group.throughput(Throughput::Bytes(container.as_bytes().len() as u64));

    group.bench_function("from_bytes_crc32c", |b| {
        b.iter(|| {
            let view =
                DnamContainerView::from_bytes(black_box(container.as_bytes()), &crypto).unwrap();
            black_box(view.preamble.header_crc32c)
        });
    });

    group.bench_function("extract_and_bind_dnam_bin", |b| {
        b.iter(|| {
            let dbin = extract_dnam_bin(black_box(container.as_bytes())).unwrap();
            let bin_view = DnamBinView::bind_prevalidated(dbin).unwrap();
            black_box(bin_view.retained_tier)
        });
    });

    group.bench_function("zoh_compute_96khz", |b| {
        b.iter(|| {
            let rescaled = ZohRateContext::compute(
                black_box(48_000.0),
                black_box(96_000.0),
                Tier::Full,
                true,
                false,
            )
            .unwrap();
            black_box(rescaled.k_ctrl)
        });
    });

    let mut engine = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    group.bench_function("hot_swap_reload_in_place", |b| {
        b.iter(|| {
            engine
                .reload_model_in_place(black_box(container.as_bytes()), Tier::Full, 48_000)
                .unwrap();
        });
    });

    group.finish();
}

fn bench_provenance_and_watermark(c: &mut Criterion) {
    let container = make_bench_container();
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(container.as_bytes(), &crypto).unwrap();
    let arena = view.arena_storage.as_bytes();

    let mut group = c.benchmark_group("provenance_and_watermark");

    group.bench_function("extract_structural_hash_224d", |b| {
        b.iter(|| black_box(extract_structural_hash_224d(black_box(arena)).unwrap()));
    });

    let fir_256b = [0u8; 256];
    let kautz_256b = [0u8; 256];
    group.bench_function("extract_cabinet_tier_fingerprint", |b| {
        b.iter(|| {
            black_box(
                extract_cabinet_tier_fingerprint_component(
                    black_box(&fir_256b),
                    black_box(&kautz_256b),
                    black_box(8),
                )
                .unwrap(),
            )
        });
    });

    let wm_key = [0x5Au8; 32];
    let wm_scope = WatermarkSeedScope {
        marketplace_pubkey: [0x11; 32],
        watermark_key_id: [0x22; 16],
        creator_signed_manifest_sha256: [0x33; 32],
        model_id: [0x44; 16],
        watermark_profile_sha256: [0x55; 32],
        owning_tier: 1,
    };
    let license_id = [0x66u8; 16];
    let buyer_id = [0x77u8; 16];
    group.bench_function("derive_watermark_seed_hmac_sha256", |b| {
        b.iter(|| {
            black_box(
                derive_watermark_seed_hmac_sha256(
                    black_box(&wm_key),
                    black_box(&wm_scope),
                    Some((black_box(&license_id), black_box(&buyer_id))),
                )
                .unwrap(),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_container_and_zoh,
    bench_provenance_and_watermark
);
criterion_main!(benches);
