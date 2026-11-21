//! Marketplace Provenance, Encrypted Arena, Isometric Gauge Audio Equivalence, Tardos Collusion
//! Tracing, Buyer Re-Framing, Preview Containment, and In-Place Preset Update Integration Suite
//! (`tests/marketplace_provenance_and_watermark.rs`).
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md` &
//! `docs/specs/security/watermarking_and_marketplace.md`.

#![forbid(unsafe_code)]

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use std::collections::BTreeMap;

use arena_builder::{
    assemble_arena_bytes, build_creator_dau1_container_from_dnam_bin,
    build_dnam_bin_from_header_wires, build_hybrid_container_for_arena, build_sona_wires, put_f32,
    read_f32,
};
use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, DnamContainerView,
    StructuredConditionalBound, extract_dnam_bin, strip_container_to_tier, update_header_in_place,
    validate_v3_dnam_layout_with_max_profile,
};
use dnam_engine::provenance::{
    Dau1Envelope, PHASE_CODING_MAX_RAD, RustCryptoProvider, STAGE3_KAUTZ_BLOCKS_PER_TIER,
    TARDOS_CASCADE_WEIGHTS, VerificationStatus, apply_isometric_gauge_watermark,
    apply_phase_coding_watermark, apply_sona_sign_gauge_watermark, apply_tardos_watermark_7680,
    build_creator_signing_message_for_tiers, build_marketplace_receipt_signing_message,
    chacha20_xor_arena_blocks, compute_immutable_header_sha256, derive_sona_sign_gauge_40,
    encode_hex, extract_structural_hash_224d, for_each_tardos_f32_offset_up_to,
    for_each_tardos_fp16_offset_up_to, generate_tardos_code_7680, is_role_a_cascade_index,
    reframe_buyer_container_with_dbin, reframe_buyer_dau1_container_with_dbin,
    score_phase_coding_arena, score_tardos_watermark_7680, sha256_digest,
    structural_cosine_similarity,
};
use dnam_engine::schema::{
    AuthorizedDevice, ControlKind, ControlSchemaDef, ControlUiHints, CreatorSignature,
    DatasetProvenance, DnamError, DnamJsonHeader, DnamProtectionProfile, MarketplaceReceipt,
    ModelSignatures, PresetDef,
};
use dnam_engine::{DnamEngine, Tier};
use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, DnamBinView, SONA_META_CONVOLVER_PREFIX_OVERRIDES,
    TRANSPORT_DELAY_BASES, TransportQuadWire,
};
use dnam_vm::math::DefaultMathBackend;
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, IntoBytes};

fn build_reference_header_and_arena_with_kautz_override(
    with_prefix_override: bool,
) -> (DnamJsonHeader, Vec<u8>, Vec<u8>) {
    let controls_schema = vec![
        ControlSchemaDef {
            id: "gain".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: Some("dB".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -0.6, 0.6]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([2.0, 8.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: Some("CLAMP-RANGE: [2.0, 8.0]".to_string()),
            ui_hints: ControlUiHints {
                midpoint: Some(4.0),
                ..Default::default()
            },
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "feedback".to_string(),
            name: "Repeats".to_string(),
            kind: ControlKind::Range,
            unit: Some("%".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: vec![StructuredConditionalBound {
                when: vec![("boost".to_string(), "on".to_string())],
                min: None,
                max: Some(9.368_421),
            }],
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: Some("CLAMP-STABILITY: <= 9.37 when boost=on".to_string()),
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "boost".to_string(),
            name: "Boost Mode".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("off"),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: Some(vec![
                "off".to_string(),
                "on".to_string(),
                "ultra".to_string(),
            ]),
            active_options: Some(vec!["off".to_string(), "on".to_string()]),
            clamp_warning: Some("CLAMP-SWITCH: disabled ['ultra']".to_string()),
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        },
    ];

    let mut preset_ctrls = BTreeMap::new();
    preset_ctrls.insert("gain".to_string(), serde_json::json!(6.5));
    preset_ctrls.insert("feedback".to_string(), serde_json::json!(9.0));
    preset_ctrls.insert("boost".to_string(), serde_json::json!("on"));

    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = "Test Vintage Tape Echo".to_string();
    header.metadata.author = Some("DNAM Test Suite".to_string());
    header.metadata.description = Some("Empirical verification test model".to_string());
    header.metadata.license = Some("MIT".to_string());
    header.metadata.provenance = DatasetProvenance {
        dataset_manifest_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_string(),
        config_recipe_sha256: None,
        epochs_trained: 120,
    };
    header.io_config.tail_time_ms = 1250.0;
    header.controls_schema = controls_schema;
    header.presets = vec![PresetDef {
        id: "dub_swell".to_string(),
        name: "Dub Swell".to_string(),
        description: None,
        controls: preset_ctrls,
        user_metadata: BTreeMap::new(),
    }];

    let (mut a0, a1, a2, mut a3) =
        build_sona_wires(&header.controls_schema, Tier::Ultra, false, false, true);
    if with_prefix_override {
        let mf = a0.crossbar.sona_meta_flags.get() | SONA_META_CONVOLVER_PREFIX_OVERRIDES;
        a0.crossbar.sona_meta_flags = U32::new(mf);
        a0.crossbar.cell_desc.cell_flags = U32::new(mf);
        if let Ok(c_words) =
            <[F32]>::mut_from_bytes(&mut a3.linear_readout_ultra.prefix_override[384..])
        {
            for w in c_words {
                *w = F32::new(0.08);
            }
        }
        for i in 0..24 {
            a3.linear_readout_ultra.w_ro_and_mods_fp16[488 + 96 + i] = U16::new(f32_to_fp16(0.25));
        }
        for i in 24..72 {
            a3.linear_readout_ultra.w_ro_and_mods_fp16[488 + 96 + i] = U16::new(f32_to_fp16(0.10));
        }
    }

    let template_dbin = build_dnam_bin_from_header_wires(
        &header,
        Tier::Ultra,
        (&a0, Some(&a1), Some(&a2), Some(&a3)),
        None,
        &[],
    );
    let arena = assemble_arena_bytes(&a0, &a1, &a2, &a3);
    (header, arena, template_dbin)
}

#[test]
fn test_marketplace_provenance_enclave_encryption_and_gauge_audio_equivalence() {
    let (mut header, master_arena, template_dbin) =
        build_reference_header_and_arena_with_kautz_override(false);
    let mut buyer_arena = master_arena.clone();
    let mut sign_only_arena = master_arena.clone();

    let total_kautz_channels: usize = STAGE3_KAUTZ_BLOCKS_PER_TIER.iter().map(|b| b * 4).sum();
    let mut kappa = vec![1.0_f32; total_kautz_channels];
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = 0.990 + 0.0005 * (i as f32);
    }
    let theta_q = 0.35_f32;
    let (cq, sq) = (theta_q.cos(), theta_q.sin());
    let q_buyer = [
        [cq, -sq, 0.0, 0.0],
        [sq, cq, 0.0, 0.0],
        [0.0, 0.0, cq, -sq],
        [0.0, 0.0, sq, cq],
    ];
    apply_isometric_gauge_watermark(&mut buyer_arena, &kappa, &q_buyer).unwrap();
    let mp_secret = *b"dnam-marketplace-master-key-2026";
    let sona_signs = derive_sona_sign_gauge_40(&mp_secret);
    apply_sona_sign_gauge_watermark(&mut buyer_arena, &sona_signs).unwrap();
    apply_sona_sign_gauge_watermark(&mut sign_only_arena, &sona_signs).unwrap();

    // 1. 224-D structural hash invariance + f32le TransportQuadWire::delay_samples sensitivity [48..52]
    let hash_master = extract_structural_hash_224d(&master_arena).unwrap();
    let hash_buyer = extract_structural_hash_224d(&buyer_arena).unwrap();
    let cos_sim = structural_cosine_similarity(&hash_master, &hash_buyer);
    assert!((cos_sim - 1.0).abs() < 1e-12);

    let mut delay_tweaked = master_arena.clone();
    let t0_off = TRANSPORT_DELAY_BASES[0];
    let (tq_m, _) = TransportQuadWire::mut_from_prefix(&mut delay_tweaked[t0_off..]).unwrap();
    tq_m.delay_samples[0] = F32::new(tq_m.delay_samples[0].get() + 19.5);
    let hash_delay = extract_structural_hash_224d(&delay_tweaked).unwrap();
    assert!((hash_delay[48] - (hash_master[48] + 19.5)).abs() < 1e-12);
    assert!((structural_cosine_similarity(&hash_master, &hash_delay) - 1.0).abs() > 1e-6);

    // Layer 2 [0.2, 0.8] phase coding preserves 224-D structural hash while remaining traceable
    let mut phase_arena = master_arena.clone();
    apply_phase_coding_watermark(&mut phase_arena, &mp_secret, PHASE_CODING_MAX_RAD).unwrap();
    let hash_phase = extract_structural_hash_224d(&phase_arena).unwrap();
    assert!((structural_cosine_similarity(&hash_master, &hash_phase) - 1.0).abs() < 1e-12);
    let z_phase = score_phase_coding_arena(&master_arena, &phase_arena, &mp_secret).unwrap();
    assert!(z_phase > 3.0, "Phase-coding Z-score too low: {z_phase:.2}");

    // Tier-sliced isometric + SONA sign gauge + Tardos across Lite, Full, and Pro prefixes
    let test_code = generate_tardos_code_7680(&mp_secret, "tape_echo_v1", "tier_probe");
    for &tier_len in &[
        ARENA_OFFSET_LITE_END as usize,
        ARENA_OFFSET_FULL_END as usize,
        ARENA_OFFSET_PRO_END as usize,
    ] {
        let mut sliced = master_arena[..tier_len].to_vec();
        apply_isometric_gauge_watermark(&mut sliced, &kappa, &q_buyer).unwrap();
        apply_sona_sign_gauge_watermark(&mut sliced, &sona_signs).unwrap();
        let h_m = extract_structural_hash_224d(&master_arena[..tier_len]).unwrap();
        let h_s = extract_structural_hash_224d(&sliced).unwrap();
        assert!((structural_cosine_similarity(&h_m, &h_s) - 1.0).abs() < 1e-12);

        let mut tardos_sliced = master_arena[..tier_len].to_vec();
        apply_tardos_watermark_7680(&mut tardos_sliced, &test_code, 2e-3).unwrap();
        let z_tier =
            score_tardos_watermark_7680(&master_arena[..tier_len], &tardos_sliced, &test_code)
                .unwrap();
        let min_z = if tier_len == ARENA_OFFSET_LITE_END as usize {
            8.0
        } else {
            15.0
        };
        assert!(z_tier > min_z);
    }

    // 2. Verify cyclic glu_mix sign-flip Mechanism A2 + SO(4) gauge 0-ULP audio parity through DnamEngine
    let master_bytes = build_hybrid_container_for_arena(
        &mut header.clone(),
        &template_dbin,
        &master_arena,
        &master_arena,
        false,
    );
    let sign_only_bytes = build_hybrid_container_for_arena(
        &mut header.clone(),
        &template_dbin,
        &sign_only_arena,
        &sign_only_arena,
        false,
    );
    let buyer_bytes = build_hybrid_container_for_arena(
        &mut header.clone(),
        &template_dbin,
        &buyer_arena,
        &buyer_arena,
        false,
    );
    assert!(extract_dnam_bin(&master_bytes).is_ok());

    let mut eng_m =
        DnamEngine::<DefaultMathBackend>::from_bytes(&master_bytes, Tier::Ultra, 48_000).unwrap();
    let mut eng_s =
        DnamEngine::<DefaultMathBackend>::from_bytes(&sign_only_bytes, Tier::Ultra, 48_000)
            .unwrap();
    let mut eng_b =
        DnamEngine::<DefaultMathBackend>::from_bytes(&buyer_bytes, Tier::Ultra, 48_000).unwrap();
    eng_m.apply_preset("dub_swell").unwrap();
    eng_s.apply_preset("dub_swell").unwrap();
    eng_b.apply_preset("dub_swell").unwrap();

    let in_l: Vec<f32> = (0..256).map(|n| 0.35 * ((n as f32) * 0.11).sin()).collect();
    let in_r: Vec<f32> = (0..256).map(|n| 0.35 * ((n as f32) * 0.07).cos()).collect();
    let (mut out_ml, mut out_mr) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    let (mut out_sl, mut out_sr) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    let (mut out_bl, mut out_br) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    eng_m.process_block(&in_l, &in_r, &mut out_ml, &mut out_mr);
    eng_s.process_block(&in_l, &in_r, &mut out_sl, &mut out_sr);
    eng_b.process_block(&in_l, &in_r, &mut out_bl, &mut out_br);

    let zh_m = eng_m.telemetry_snapshot().z_head;
    let zh_b = eng_b.telemetry_snapshot().z_head;
    assert!(zh_m.iter().any(|&z| z.abs() > 1e-6));
    assert!(zh_b.iter().any(|&z| z.abs() > 1e-6));
    assert_ne!(zh_m, zh_b, "SO(4) gauge must rotate z_head coordinates");
    for i in 0..256 {
        assert_eq!(out_ml[i].to_bits(), out_sl[i].to_bits());
        assert_eq!(out_mr[i].to_bits(), out_sr[i].to_bits());
    }

    let max_audio_diff = out_ml
        .iter()
        .zip(out_bl.iter())
        .chain(out_mr.iter().zip(out_br.iter()))
        .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        max_audio_diff < 2e-4,
        "SO(4) + fp16 gauge audio diff={max_audio_diff}"
    );

    // 2b. Verify SONA_META_CONVOLVER_PREFIX_OVERRIDES gauge parity with embedded dbin_prefix
    let (mut ov_header, ov_master_arena, ov_dbin) =
        build_reference_header_and_arena_with_kautz_override(true);
    let (a0_ov, _) = Arena0LiteWire::ref_from_prefix(&ov_master_arena).unwrap();
    assert_ne!(
        a0_ov.crossbar.sona_meta_flags.get() & SONA_META_CONVOLVER_PREFIX_OVERRIDES,
        0
    );
    let mut ov_buyer_arena = ov_master_arena.clone();
    apply_isometric_gauge_watermark(&mut ov_buyer_arena, &kappa, &q_buyer).unwrap();
    apply_sona_sign_gauge_watermark(&mut ov_buyer_arena, &sona_signs).unwrap();
    let hash_ov_m = extract_structural_hash_224d(&ov_master_arena).unwrap();
    let hash_ov_b = extract_structural_hash_224d(&ov_buyer_arena).unwrap();
    assert!((structural_cosine_similarity(&hash_ov_m, &hash_ov_b) - 1.0).abs() < 1e-12);
    let ov_master_bytes = build_hybrid_container_for_arena(
        &mut ov_header.clone(),
        &ov_dbin,
        &ov_master_arena,
        &ov_master_arena,
        false,
    );
    let ov_buyer_bytes = build_hybrid_container_for_arena(
        &mut ov_header,
        &ov_dbin,
        &ov_buyer_arena,
        &ov_buyer_arena,
        false,
    );
    let mut eng_ov_m =
        DnamEngine::<DefaultMathBackend>::from_bytes(&ov_master_bytes, Tier::Ultra, 48_000)
            .unwrap();
    let mut eng_ov_b =
        DnamEngine::<DefaultMathBackend>::from_bytes(&ov_buyer_bytes, Tier::Ultra, 48_000).unwrap();
    eng_ov_m.apply_preset("dub_swell").unwrap();
    eng_ov_b.apply_preset("dub_swell").unwrap();
    let (mut out_ov_ml, mut out_ov_mr) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    let (mut out_ov_bl, mut out_ov_br) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    eng_ov_m.process_block(&in_l, &in_r, &mut out_ov_ml, &mut out_ov_mr);
    eng_ov_b.process_block(&in_l, &in_r, &mut out_ov_bl, &mut out_ov_br);
    let ov_audio_diff = out_ov_ml
        .iter()
        .zip(out_ov_bl.iter())
        .chain(out_ov_mr.iter().zip(out_ov_br.iter()))
        .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        ov_audio_diff < 1e-3,
        "SONA_META_CONVOLVER_PREFIX_OVERRIDES audio diff={ov_audio_diff}"
    );

    // 3. Verify 5,807 Role B Tardos coordinates (with ou_to_z_stage_ultra_fp16[4] at 29_952..29_960
    //    untouched) + 4-buyer collusion survival + DAU1/DCRC buyer re-framing preserving L_master!
    let mut role_a_count = 0usize;
    let mut role_b_count = 0usize;
    let n_f32 = for_each_tardos_f32_offset_up_to(master_arena.len(), |idx, _| {
        if is_role_a_cascade_index(idx) {
            role_a_count += 1;
        } else {
            role_b_count += 1;
        }
    });
    let n_fp16 = for_each_tardos_fp16_offset_up_to(master_arena.len(), |idx, _| {
        if is_role_a_cascade_index(idx) {
            role_a_count += 1;
        } else {
            role_b_count += 1;
        }
    });
    assert_eq!(role_a_count, 0);
    assert_eq!(n_f32 + n_fp16, 5_807);
    assert_eq!(role_b_count, 5_807);

    let ou_z_slice_before = master_arena[29_952..29_960].to_vec();
    let colluder_ids = ["lic_017", "lic_042", "lic_108", "lic_163"];
    let mut colluder_arenas = Vec::new();
    let mut colluder_codes = Vec::new();
    for &cid in &colluder_ids {
        let code = generate_tardos_code_7680(&mp_secret, "tape_echo_v1", cid);
        let mut a = master_arena.clone();
        apply_tardos_watermark_7680(&mut a, &code, 2e-3).unwrap();
        assert_eq!(&a[29_952..29_960], ou_z_slice_before.as_slice());
        colluder_arenas.push(a);
        colluder_codes.push(code);
    }

    let mut pirated_arena = master_arena.clone();
    for_each_tardos_f32_offset_up_to(master_arena.len(), |_, off| {
        let sum: f32 = colluder_arenas.iter().map(|ca| read_f32(ca, off)).sum();
        let fp16_attacked = fp16_to_f32(f32_to_fp16(sum * 0.25));
        put_f32(&mut pirated_arena, off, fp16_attacked);
    });
    for_each_tardos_fp16_offset_up_to(master_arena.len(), |_, off| {
        let sum: f32 = colluder_arenas
            .iter()
            .map(|ca| {
                let (raw_u16, _) = U16::ref_from_prefix(&ca[off..]).unwrap();
                fp16_to_f32(raw_u16.get())
            })
            .sum();
        let bits = f32_to_fp16(sum * 0.25);
        let _ = U16::new(bits).write_to_prefix(&mut pirated_arena[off..off + 2]);
    });

    for code in &colluder_codes {
        let z = score_tardos_watermark_7680(&master_arena, &pirated_arena, code).unwrap();
        assert!(z > 8.5, "4-colluder Z-score too low: {z:.2}");
    }
    let innocent_code = generate_tardos_code_7680(&mp_secret, "tape_echo_v1", "innocent_buyer_999");
    let z_innocent =
        score_tardos_watermark_7680(&master_arena, &pirated_arena, &innocent_code).unwrap();
    assert!(z_innocent.abs() < 4.5 && TARDOS_CASCADE_WEIGHTS == 7_680);

    // DAU1 creator-signed + buyer Tardos reframe with embedded dbin_prefix preserving L_master & creator_signature
    let creator_seed_dau = [7u8; 32];
    let mp_seed_dau = [19u8; 32];
    let mp_pk_dau = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed_dau);
    let (dau_master, l_master, creator_sig_dau) = build_creator_dau1_container_from_dnam_bin(
        &header,
        &template_dbin,
        &creator_seed_dau,
        true,
    );
    let dau_buyer = reframe_buyer_dau1_container_with_dbin(
        &dau_master,
        &colluder_arenas[0],
        &mp_seed_dau,
        br#"{"license":"personal","tier":"ultra"}"#,
    )
    .unwrap();
    let dau_layout = validate_v3_dnam_layout_with_max_profile(&dau_buyer, 1).unwrap();
    let parsed_dau = Dau1Envelope::from_bytes(
        &dau_buyer[dau_layout.auth_extent.0..dau_layout.auth_extent.0 + dau_layout.auth_extent.1],
    )
    .unwrap();
    assert_eq!(parsed_dau.master_prefixes, l_master);
    assert_eq!(parsed_dau.creator_signature, creator_sig_dau);
    assert!(extract_dnam_bin(&dau_buyer).is_ok());
    let dau_crypto = RustCryptoProvider::new().with_marketplace_pubkey(mp_pk_dau);
    let dau_view = DnamContainerView::from_bytes(&dau_buyer, &dau_crypto).unwrap();
    assert!(matches!(
        dau_view.verification_status,
        VerificationStatus::MarketplaceVerified { .. }
    ));

    let dcrc_buyer = reframe_buyer_container_with_dbin(&master_bytes, &buyer_arena).unwrap();
    assert!(extract_dnam_bin(&dcrc_buyer).is_ok());

    // 4. Split-Payload Ed25519 + X25519 + ChaCha20 encrypted arena with embedded dbin_prefix + strip + DMU1 update
    use ed25519_dalek::{Signer, SigningKey};
    use x25519_dalek::{PublicKey as XPublicKey, StaticSecret as XSecret};

    let creator_sk = SigningKey::from_bytes(&[7u8; 32]);
    let creator_pk = creator_sk.verifying_key().to_bytes();
    let mp_sk = SigningKey::from_bytes(&[19u8; 32]);
    let mp_pk = mp_sk.verifying_key().to_bytes();

    let device_id_hash = "9c1185a5c5e9fc54612808977ee8f548";
    let dev_sk_bytes = [42u8; 32];
    let dev_pk_bytes = XPublicKey::from(&XSecret::from(dev_sk_bytes)).to_bytes();
    let eph_sk_bytes = [99u8; 32];
    let eph_pk_bytes = XPublicKey::from(&XSecret::from(eph_sk_bytes)).to_bytes();

    let cek = [0x5A_u8; 32];
    let nonce_96 = [0x11_u8; 12];
    let wrapped_cek =
        RustCryptoProvider::wrap_device_cek(&eph_sk_bytes, &dev_pk_bytes, device_id_hash, &cek)
            .unwrap();

    let imm_sha = compute_immutable_header_sha256(&header).unwrap();
    let master_sha = sha256_digest(&master_arena);
    let buyer_sha = sha256_digest(&buyer_arena);
    let master_tier_shas = [
        encode_hex(&sha256_digest(&master_arena[..ARENA_TIER_BYTES[0]])),
        encode_hex(&sha256_digest(&master_arena[..ARENA_TIER_BYTES[1]])),
        encode_hex(&sha256_digest(&master_arena[..ARENA_TIER_BYTES[2]])),
        encode_hex(&master_sha),
    ];
    let buyer_tier_shas = [
        encode_hex(&sha256_digest(&buyer_arena[..ARENA_TIER_BYTES[0]])),
        encode_hex(&sha256_digest(&buyer_arena[..ARENA_TIER_BYTES[1]])),
        encode_hex(&sha256_digest(&buyer_arena[..ARENA_TIER_BYTES[2]])),
        encode_hex(&buyer_sha),
    ];

    let creator_msg =
        build_creator_signing_message_for_tiers(&imm_sha, &master_sha, Some(&master_tier_shas))
            .unwrap();
    let creator_sig = creator_sk.sign(&creator_msg).to_bytes();

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f".to_string(),
        buyer_hash: encode_hex(&[0xAB; 32]),
        license_tier: "personal".to_string(),
        issued_at: "2026-10-23T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&buyer_sha),
        buyer_tier_arena_sha256: Some(buyer_tier_shas),
        encryption_mode: "x25519_chacha20_arena".to_string(),
        arena_nonce_96: Some(encode_hex(&nonce_96)),
        ephemeral_pubkey_x25519: Some(encode_hex(&eph_pk_bytes)),
        authorized_devices: vec![AuthorizedDevice {
            oem_vendor: "darkglass".to_string(),
            model_code: "anagram_rk3582".to_string(),
            device_id_hash: device_id_hash.to_string(),
            wrapped_cek: Some(encode_hex(&wrapped_cek)),
        }],
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: String::new(),
    };
    let receipt_msg =
        build_marketplace_receipt_signing_message(&imm_sha, &master_sha, &receipt).unwrap();
    receipt.receipt_sig_ed25519 = encode_hex(&mp_sk.sign(&receipt_msg).to_bytes());

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&imm_sha),
        master_arena_sha256: encode_hex(&master_sha),
        tier_arena_sha256: Some(master_tier_shas),
        creator_signature: CreatorSignature {
            signer_id: "did:web:studio.example".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    let mut encrypted_arena = buyer_arena.clone();
    chacha20_xor_arena_blocks(&mut encrypted_arena, &cek, &nonce_96, 0).unwrap();

    let mut container_bytes = build_hybrid_container_for_arena(
        &mut header,
        &template_dbin,
        &buyer_arena,
        &encrypted_arena,
        true,
    );

    header
        .user_metadata
        .insert("favorite".to_string(), serde_json::json!(true));
    update_header_in_place(&mut container_bytes, &header).unwrap();

    let crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, dev_sk_bytes)
        .with_marketplace_pubkey(mp_pk);

    let loaded_view = DnamContainerView::from_bytes(&container_bytes, &crypto).unwrap();
    assert_eq!(
        loaded_view.protection_profile,
        DnamProtectionProfile::DeviceAllowlisted
    );
    assert_eq!(loaded_view.arena_storage.as_bytes(), buyer_arena.as_slice());

    let mut eng_dec = DnamEngine::<DefaultMathBackend>::from_bytes_with_crypto(
        &container_bytes,
        Tier::Pro,
        48_000,
        &crypto,
    )
    .unwrap();
    let (mut dl, mut dr) = ([0.0_f32; 32], [0.0_f32; 32]);
    eng_dec.process_block(&in_l[..32], &in_r[..32], &mut dl, &mut dr);
    assert!(dl.iter().any(|&v| v.abs() > 1e-6));

    for max_tier in [Tier::Lite, Tier::Full, Tier::Pro] {
        let stripped = strip_container_to_tier(&container_bytes, max_tier).unwrap();
        let expected_arena_len = max_tier.expected_arena_bytes();
        let stripped_view = DnamContainerView::from_bytes(&stripped, &crypto).unwrap();
        assert_eq!(
            stripped_view.verification_status,
            VerificationStatus::MarketplaceVerified {
                signer_id: "did:web:studio.example".to_string(),
                marketplace_id: "dnam.store".to_string(),
                license_id: "0192b6c0-7e4a-7000-8000-1a2b3c4d5e6f".to_string(),
                license_tier: "personal".to_string(),
            }
        );
        assert_eq!(
            stripped_view.arena_storage.as_bytes(),
            &buyer_arena[..expected_arena_len]
        );
    }

    // 5. Verify .dnam-prev rejection on DAW loaders + reload_model_in_place and PreviewGuardState engagement
    let mut prev_header = DnamJsonHeader::default_reference();
    prev_header.container_kind = "preview".to_string();
    let prev_bytes = build_hybrid_container_for_arena(
        &mut prev_header,
        &template_dbin,
        &master_arena,
        &master_arena,
        false,
    );
    let _ = DnamBinView::parse(extract_dnam_bin(&prev_bytes).unwrap()).unwrap();
    assert!(matches!(
        DnamContainerView::from_bytes(&prev_bytes, &crypto),
        Err(DnamError::PreviewContainerRejected)
    ));
    assert_eq!(
        DnamEngine::<DefaultMathBackend>::from_bytes(&prev_bytes, Tier::Lite, 48_000).unwrap_err(),
        DnamError::PreviewContainerRejected
    );
    let mut pool = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    assert_eq!(
        pool.reload_model_in_place(&prev_bytes, Tier::Lite, 48_000)
            .unwrap_err(),
        DnamError::PreviewContainerRejected
    );

    let prev_view = DnamContainerView::from_bytes_allowing_preview(&prev_bytes, &crypto).unwrap();
    assert_eq!(prev_view.header.container_kind, "preview");

    let mut prev_eng =
        DnamEngine::<DefaultMathBackend>::from_preview_bytes(&prev_bytes, Tier::Ultra, 48_000)
            .unwrap();
    assert!(prev_eng.is_preview_mode());
    assert_eq!(prev_eng.effective_tier(), Tier::Lite);
    let mut imp = [0.0_f32; 64];
    imp[0] = 0.85;
    let (mut pl, mut pr) = ([0.0_f32; 64], [0.0_f32; 64]);
    prev_eng.process_block(&imp, &imp, &mut pl, &mut pr);
    assert!(prev_eng.preview_guard().poison_level() > 0.0);
}
