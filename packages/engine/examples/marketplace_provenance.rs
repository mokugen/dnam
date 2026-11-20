//! Example 3: Digital Marketplace Provenance, Isometric Gauge Watermarking & Tardos Tracing (`examples/marketplace_provenance.rs`).
//!
//! Demonstrates:
//! 1. Extracting the 224-D Gauge-Invariant Structural Hash (`extract_structural_hash_224d`).
//! 2. Embedding a buyer-specific Exact Isometric Gauge Watermark (`apply_isometric_gauge_watermark`)
//!    that preserves the 224-D structural hash (`cosine_similarity == 1.000000`).
//! 3. Embedding and scoring the 7,680-weight Spread-Spectrum Tardos Traitor-Tracing Code
//!    (`generate_tardos_code_7680`, `apply_tardos_watermark_7680`, `score_tardos_watermark_7680`).
//!
//! Run with:
//!   `cargo run -p dnam-engine --example marketplace_provenance`

use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container};
use dnam_engine::graph::sona_core::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, SONA_FIR_MAX_TAPS, SonaMatryoshkaModel, SonaTierReadout,
};
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::provenance::security::{
    WatermarkSeedScope, derive_watermark_seed_hmac_sha256, evaluate_anticlone_revocation_policy,
    extract_cabinet_tier_fingerprint_component,
};
use dnam_engine::provenance::{
    KAUTZ_TOTAL_CHANNELS, OpenArenaCryptoProvider, STAGE3_EARLY_FIR_BASES, STAGE3_KAUTZ_BASES,
    apply_isometric_gauge_watermark, apply_sona_sign_gauge_watermark, apply_tardos_watermark_7680,
    derive_sona_sign_gauge_40, extract_structural_hash_224d, generate_tardos_code_7680,
    score_tardos_watermark_7680, structural_cosine_similarity,
};
use dnam_engine::schema::{DnamError, DnamJsonHeader};

fn main() {
    // 1. Construct a 30,720-byte `SONA` `v3.0` `Arena 0..3` buffer with non-zero cores, FIR & Kautz modals
    let mut master_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    for g in 0..MAX_SONA_GROUPS {
        model.groups[g].in_x_weight = [0.6, 0.1, -0.1, 0.05];
    }
    for m in 0..MAX_KAUTZ_POLES {
        model.kautz_poles[m] = (0.91, 0.14 + 0.03 * ((m & 3) as f32));
    }
    for t in 0..4 {
        let mut ro = SonaTierReadout::default();
        for i in 0..SONA_FIR_MAX_TAPS {
            ro.fir_l[i] = 0.04;
            ro.fir_r[i] = 0.03;
        }
        for m in 0..MAX_KAUTZ_POLES {
            ro.kautz_b_gamma[m] = 0.35;
            ro.kautz_c_l[m] = (0.20, -0.10);
            ro.kautz_c_r[m] = (0.16, -0.08);
        }
        model.readouts[t] = ro;
    }
    pack_sona_matryoshka_to_arena(&model, &mut master_arena);

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

    // 5. Embed & detect 7,680-chip Coupled Tardos Traitor-Tracing Code for Buyer #1
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
        let score = score_tardos_watermark_7680(&master_arena, &buyer1_arena, code)
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
    let prev_bytes = build_dnam_container(&mut prev_hdr, &master_arena, &master_arena, false)
        .expect("build preview container");
    let preview_err =
        DnamContainerView::from_bytes(&prev_bytes, &OpenArenaCryptoProvider).unwrap_err();
    assert!(matches!(preview_err, DnamError::PreviewContainerRejected));
    println!(
        "Anti-clone gate action (no device evidence): {gate_action:?} | Preview container guard: {preview_err}"
    );
}
