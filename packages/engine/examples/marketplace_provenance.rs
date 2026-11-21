//! Example 3: Digital Marketplace Provenance, Isometric Gauge Watermarking & Tardos Tracing
//! (`examples/marketplace_provenance.rs`).
//!
//! Demonstrates:
//! 1. Extracting the 224-D Gauge-Invariant Structural Hash (`extract_structural_hash_224d`).
//! 2. Embedding a buyer-specific Exact Isometric Gauge Watermark (`apply_isometric_gauge_watermark`)
//!    and 40-lane `0-ULP` `SONA` sign-flip gauge (`apply_sona_sign_gauge_watermark`) that preserve
//!    the 224-D structural hash (`cosine_similarity == 1.000000`).
//! 3. Embedding and scoring the 7,680-coordinate `TardosCodebook7680` (`5,807` active Role B weights)
//!    with `Q_buyer` unrotation (`score_tardos_watermark_7680_with_buyer_gauge`).
//! 4. License-scoped HMAC-SHA256 seed derivation and Cabinet tier-local structural fingerprinting.
//! 5. Anti-clone revocation gate evaluation and `.dnam-prev` storefront preview containment.
//!
//! Run with:
//!   `cargo run -p dnam-engine --example marketplace_provenance`

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container_from_dnam_bin,
};
use dnam_engine::provenance::security::{
    WatermarkSeedScope, derive_watermark_seed_hmac_sha256, evaluate_anticlone_revocation_policy,
    extract_cabinet_tier_fingerprint_component,
};
use dnam_engine::provenance::{
    KAUTZ_TOTAL_CHANNELS, OpenArenaCryptoProvider, STAGE3_EARLY_FIR_BASES, STAGE3_KAUTZ_BASES,
    apply_isometric_gauge_watermark, apply_sona_sign_gauge_watermark, apply_tardos_watermark_7680,
    derive_sona_sign_gauge_40, extract_structural_hash_224d, generate_tardos_code_7680,
    score_tardos_watermark_7680_with_buyer_gauge, structural_cosine_similarity,
};
use dnam_engine::schema::{DnamError, DnamJsonHeader};
use dnam_kernels::f32_to_fp16;
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    DNBN_FLAG_AUTO_UNITY, DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, KautzSoaBlockWire,
    encode_dnam_bin,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromZeros, IntoBytes};

fn populate_kautz_block(blk: &mut KautzSoaBlockWire, base_phase: f32) {
    for lane in 0..4 {
        blk.r_low[lane] = F32::new(0.91);
        blk.theta[lane] = F32::new(base_phase + 0.03 * (lane as f32));
        blk.b_gamma[lane] = F32::new(0.35);
        blk.c_re_l[lane] = F32::new(0.20);
        blk.c_im_l[lane] = F32::new(-0.10);
        blk.c_re_r[lane] = F32::new(0.16);
        blk.c_im_r[lane] = F32::new(-0.08);
    }
}

fn populate_fir_slice(fir_l: &mut [F32], fir_r: &mut [F32]) {
    for tap in fir_l {
        *tap = F32::new(0.04);
    }
    for tap in fir_r {
        *tap = F32::new(0.03);
    }
}

fn build_populated_tiers() -> (
    Arena0LiteWire,
    Arena1FullDeltaWire,
    Arena2ProDeltaWire,
    Arena3UltraDeltaWire,
) {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
    for cq in &mut a0.core_quads {
        cq.in_mid_weight_fp16 = [U16::new(f32_to_fp16(0.6)); 4];
    }
    populate_fir_slice(&mut a0.conv_fir_lite.fir_l, &mut a0.conv_fir_lite.fir_r);
    for (idx, blk) in a0.conv_kautz_lite.iter_mut().enumerate() {
        populate_kautz_block(blk, 0.14 + 0.02 * (idx as f32));
    }

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    for cq in &mut a1.core_quads {
        cq.in_mid_weight_fp16 = [U16::new(f32_to_fp16(0.6)); 4];
    }
    populate_fir_slice(&mut a1.conv_fir_full.fir_l, &mut a1.conv_fir_full.fir_r);
    populate_kautz_block(&mut a1.conv_kautz_full, 0.18);

    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    for cq in &mut a2.core_quads {
        cq.in_mid_weight_fp16 = [U16::new(f32_to_fp16(0.6)); 4];
    }
    populate_fir_slice(&mut a2.conv_fir_pro.fir_l, &mut a2.conv_fir_pro.fir_r);
    for (idx, blk) in a2.conv_kautz_pro.iter_mut().enumerate() {
        populate_kautz_block(blk, 0.22 + 0.02 * (idx as f32));
    }

    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    for cq in &mut a3.core_quads {
        cq.in_mid_weight_fp16 = [U16::new(f32_to_fp16(0.6)); 4];
    }
    populate_fir_slice(&mut a3.conv_fir_ultra.fir_l, &mut a3.conv_fir_ultra.fir_r);
    for (idx, blk) in a3.conv_kautz_ultra.iter_mut().enumerate() {
        populate_kautz_block(blk, 0.28 + 0.02 * (idx as f32));
    }

    (a0, a1, a2, a3)
}

fn main() {
    // 1. Construct a 30,720-byte `SONA` `v3.0` `Arena 0..3` buffer from SONA v3.0 wire structs
    let (a0, a1, a2, a3) = build_populated_tiers();
    let mut master_arena = Vec::with_capacity(ARENA_OFFSET_ULTRA_END as usize);
    master_arena.extend_from_slice(a0.as_bytes());
    master_arena.extend_from_slice(a1.as_bytes());
    master_arena.extend_from_slice(a2.as_bytes());
    master_arena.extend_from_slice(a3.as_bytes());
    assert_eq!(master_arena.len(), ARENA_OFFSET_ULTRA_END as usize);

    // 2. Extract the 224-D gauge-invariant structural hash from the master model
    let hash_master = extract_structural_hash_224d(&master_arena).expect("valid master arena");

    // 3. Embed Mechanism A watermarks: 32-ch modal reciprocal scaling + SO(4) rotation + 40-lane 0-ULP SONA sign-flip
    let mut buyer1_arena = master_arena.clone();
    let mut kappa = [1.0_f32; KAUTZ_TOTAL_CHANNELS];
    for (c, k) in kappa.iter_mut().enumerate() {
        *k = if c % 2 == 0 { 1.03 } else { 0.97 };
    }
    let theta = 0.25_f32;
    let (ct, st) = (theta.cos(), theta.sin());
    let q_rot = [
        [ct, -st, 0.0, 0.0],
        [st, ct, 0.0, 0.0],
        [0.0, 0.0, ct, -st],
        [0.0, 0.0, st, ct],
    ];
    apply_isometric_gauge_watermark(&mut buyer1_arena, &kappa, &q_rot)
        .expect("isometric gauge watermark applied");
    let sona_signs = derive_sona_sign_gauge_40(&[0x42_u8; 32]);
    apply_sona_sign_gauge_watermark(&mut buyer1_arena, &sona_signs)
        .expect("SONA 0-ULP sign-flip gauge watermark applied");

    // 4. Verify the 224-D structural hash is invariant under both gauge transformations
    let hash_buyer1 = extract_structural_hash_224d(&buyer1_arena).expect("valid buyer1 arena");
    let cos_sim = structural_cosine_similarity(&hash_master, &hash_buyer1);
    println!(
        "224-D Structural Hash Cosine Similarity (Master vs. Gauge-Rotated Buyer #1): {:.8}",
        cos_sim
    );

    // 5. Embed & detect 7,680-coordinate Coupled Tardos Traitor-Tracing Code for Buyer #1
    let secret_seed = [0x42_u8; 32];
    let code_buyer1 = generate_tardos_code_7680(&secret_seed, "model_uuid_001", "buyer_1001");
    let code_buyer2 = generate_tardos_code_7680(&secret_seed, "model_uuid_001", "buyer_1002");
    let code_buyer3 = generate_tardos_code_7680(&secret_seed, "model_uuid_001", "buyer_1003");

    apply_tardos_watermark_7680(&mut buyer1_arena, &code_buyer1, 1.5e-3)
        .expect("Tardos watermark applied");

    for (buyer_id, code) in [
        ("buyer_1001 (actual)", &code_buyer1),
        ("buyer_1002 (innocent)", &code_buyer2),
        ("buyer_1003 (innocent)", &code_buyer3),
    ] {
        let score = score_tardos_watermark_7680_with_buyer_gauge(
            &master_arena,
            &buyer1_arena,
            code,
            Some(&q_rot),
        )
        .expect("Tardos correlation computed");
        println!(
            "Candidate {:<22} -> Tardos correlation score: {:+.4}",
            buyer_id, score
        );
    }

    // 6. License-scoped HMAC-SHA256 watermark seed derivation & Cabinet tier-local fingerprint (`Lite`: 8 pairs = 256 B)
    let wm_key = [0x5A_u8; 32];
    let scope = WatermarkSeedScope {
        marketplace_pubkey: [0x11; 32],
        watermark_key_id: [0x22; 16],
        creator_signed_manifest_sha256: [0x33; 32],
        model_id: [0x44; 16],
        watermark_profile_sha256: [0x55; 32],
        owning_tier: 1,
    };
    let buyer_seed =
        derive_watermark_seed_hmac_sha256(&wm_key, &scope, Some((&[0xAA; 16], &[0xBB; 16])))
            .expect("valid buyer seed");
    let fir0 = STAGE3_EARLY_FIR_BASES[0];
    let kautz0 = STAGE3_KAUTZ_BASES[0];
    let cabinet_fp = extract_cabinet_tier_fingerprint_component(
        &master_arena[fir0..fir0 + 256],
        &master_arena[kautz0..kautz0 + 256],
        8,
    )
    .expect("valid Cabinet tier-local fingerprint");
    println!(
        "Derived Buyer HMAC-SHA256 Seed[0..4]: {:02x?} | Cabinet Fingerprint Dims: {}",
        &buyer_seed[..4],
        cabinet_fp.len(),
    );

    // 7. Verify 3-stage anti-cloning gate policy & `v3.0` `"container_kind": "preview"` rejection (`PreviewContainerRejected`)
    let gate_action = evaluate_anticlone_revocation_policy(true, true, false);
    let mut prev_hdr = DnamJsonHeader::default_reference();
    prev_hdr.container_kind = String::from("preview");
    let dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: prev_hdr.sample_rate_train_hz,
        header_flags: DNBN_FLAG_STEREO_OUT | DNBN_FLAG_AUTO_UNITY,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: prev_hdr.io_config.tail_time_ms,
        ro_pole_48k: prev_hdr.linear_readout.ro_pole_48k,
        switch_xfade_samples: prev_hdr.transport.switch_xfade_samples,
        content_hash_prefix: [0; 8],
        controls: &[],
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
    .expect("encode preview .dnam-bin");
    let prev_bytes = build_dnam_container_from_dnam_bin(&mut prev_hdr, &dnam_bin)
        .expect("build preview hybrid container");
    let preview_err =
        DnamContainerView::from_bytes(&prev_bytes, &OpenArenaCryptoProvider).unwrap_err();
    assert!(matches!(preview_err, DnamError::PreviewContainerRejected));
    println!(
        "Anti-clone gate action (no device evidence): {gate_action:?} | Preview container guard: {preview_err}"
    );
}
