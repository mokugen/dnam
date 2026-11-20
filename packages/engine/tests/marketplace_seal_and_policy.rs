//! `.dnam-seal` (`DNSL`) Hardware Enclave Envelope, `OemDeviceCertificate`, and `DistributionPolicy`
//! Verification Suite (`tests/marketplace_seal_and_policy.rs`).
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md`.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_ULTRA_END, DnamContainerView, DnamPreambleWire,
    build_dnam_container,
};
use dnam_engine::graph::sona_core::SonaMatryoshkaModel;
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DNSL_MAGIC, Dme1HeaderWire, DnslDirectoryWire, DnslPreambleWire, RustCryptoProvider,
    build_creator_signing_message, build_marketplace_receipt_signing_message,
    compute_immutable_header_sha256, encode_hex, seal_dnam_container, sha256_digest,
    unseal_dnam_container, unseal_dnsl_up_to_tier, verify_oem_device_certificate,
};
use dnam_engine::schema::{
    AuthorizedDevice, CreatorSignature, DistributionPolicy, DnamError, DnamJsonHeader,
    DnamProtectionProfile, MarketplaceReceipt, ModelSignatures, PresetDef,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};
use zerocopy::FromBytes;

#[test]
fn test_dnam_seal_dnsl_envelope_oem_certificate_and_distribution_policy() {
    let mut header = DnamJsonHeader::default_reference();
    let mut master_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let model = SonaMatryoshkaModel::dual_rate_default(false);
    pack_sona_matryoshka_to_arena(&model, &mut master_arena);

    // 1. Configure Creator-Signed DistributionPolicy restricting model to EnclaveSealed on Darkglass Anagram
    header.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: vec!["darkglass:anagram_rk3582".to_string()],
    });

    let creator_seed = [21u8; 32];
    let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
    let mp_seed = [33u8; 32];
    let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);

    let oem_darkglass_seed = [55u8; 32];
    let oem_darkglass_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_darkglass_seed);
    let oem_other_seed = [66u8; 32];
    let oem_other_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_other_seed);

    let device_id_hash = "9c1185a5c5e9fc54612808977ee8f548";
    let dev_secret_x25519 = [44u8; 32];
    let dev_pubkey_x25519 = RustCryptoProvider::x25519_pubkey_from_secret(&dev_secret_x25519);

    let cert_anagram = RustCryptoProvider::sign_oem_device_certificate(
        &oem_darkglass_seed,
        "darkglass",
        "anagram_rk3582",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();
    let verified_x25519 =
        verify_oem_device_certificate(&cert_anagram, &oem_darkglass_pk, &RustCryptoProvider::new())
            .unwrap();
    assert_eq!(verified_x25519, dev_pubkey_x25519);

    assert!(
        verify_oem_device_certificate(&cert_anagram, &oem_other_pk, &RustCryptoProvider::new())
            .is_err()
    );

    // 2. Sign inner .dnam with creator signature + marketplace receipt (`encryption_mode = "none"`,
    //    wrapped exclusively inside canonical v3.0 `DNSL`/`DNPL` HPKE + AEAD records)
    let imm_sha = compute_immutable_header_sha256(&header).unwrap();
    let master_sha = sha256_digest(&master_arena);
    let creator_msg = build_creator_signing_message(&imm_sha, &master_sha);
    let creator_sig = RustCryptoProvider::ed25519_sign(&creator_seed, &creator_msg);

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-998877665544".to_string(),
        buyer_hash: encode_hex(&[0xCD; 32]),
        license_tier: "oem_bundle".to_string(),
        issued_at: "2026-10-27T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&master_sha),
        buyer_tier_arena_sha256: None,
        encryption_mode: "none".to_string(),
        arena_nonce_96: None,
        ephemeral_pubkey_x25519: None,
        authorized_devices: vec![AuthorizedDevice {
            oem_vendor: "darkglass".to_string(),
            model_code: "anagram_rk3582".to_string(),
            device_id_hash: device_id_hash.to_string(),
            wrapped_cek: None,
        }],
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: String::new(),
    };
    let receipt_msg =
        build_marketplace_receipt_signing_message(&imm_sha, &master_sha, &receipt).unwrap();
    receipt.receipt_sig_ed25519 =
        encode_hex(&RustCryptoProvider::ed25519_sign(&mp_seed, &receipt_msg));

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&imm_sha),
        master_arena_sha256: encode_hex(&master_sha),
        tier_arena_sha256: None,
        creator_signature: CreatorSignature {
            signer_id: "did:web:darkglass.com".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    let inner_dnam_bytes =
        build_dnam_container(&mut header, &master_arena, &master_arena, false).unwrap();

    // 3. Loading the unsealed inner .dnam directly must FAIL because DistributionPolicy requires EnclaveSealed!
    let crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, dev_secret_x25519)
        .with_marketplace_pubkey(mp_pk);
    assert!(DnamContainerView::from_bytes(&inner_dnam_bytes, &crypto).is_err());

    // 4. Attempting to seal for a different model ("darkglass:other_pedal") must FAIL DistributionPolicy!
    let cert_wrong_model = RustCryptoProvider::sign_oem_device_certificate(
        &oem_darkglass_seed,
        "darkglass",
        "other_pedal",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();
    let trusted_oems = [("darkglass", oem_darkglass_pk), ("other", oem_other_pk)];
    let eph_outer_sk = [88u8; 32];
    let cek_outer = [0xA5_u8; 32];
    let nonce_outer = [0x77_u8; 12];
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[cert_wrong_model],
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );

    // 5. Sealing for the authorized "darkglass:anagram_rk3582" certificate succeeds and produces 64B-aligned DNSL
    let sealed_bytes = seal_dnam_container(
        &inner_dnam_bytes,
        std::slice::from_ref(&cert_anagram),
        &trusted_oems,
        &mp_seed,
        &eph_outer_sk,
        &cek_outer,
        &nonce_outer,
    )
    .unwrap();
    assert_eq!(&sealed_bytes[0..4], &DNSL_MAGIC);
    assert_eq!(sealed_bytes[8], 2);
    assert_eq!(sealed_bytes.len() % 64, 0);
    assert!(sealed_bytes.len() > master_arena.len());
    assert!(sealed_bytes.len() + 16_384 > inner_dnam_bytes.len());
    assert_eq!(core::mem::size_of::<Dme1HeaderWire>(), 20);

    // 6. Unsealing verifies DAI1/DSA1 prefixes up to Tier::Full and recovers canonical v3.0 plaintext .dnam
    let unsealed_full = unseal_dnsl_up_to_tier(&sealed_bytes, &crypto, Tier::Full).unwrap();
    assert_eq!(unsealed_full.retained_tier, Tier::Full);
    assert_eq!(
        unsealed_full.arena_bytes.as_slice(),
        &master_arena[..ARENA_OFFSET_FULL_END as usize]
    );
    let unsealed_inner = unseal_dnam_container(&sealed_bytes, &crypto).unwrap();
    let (unsealed_preamble, _) = DnamPreambleWire::ref_from_prefix(&unsealed_inner).unwrap();
    assert_eq!(&unsealed_preamble.magic, b"DNAM");
    assert_eq!(unsealed_preamble.version_major.get(), 3);
    let unsealed_ultra = unseal_dnsl_up_to_tier(&sealed_bytes, &crypto, Tier::Ultra).unwrap();
    assert_eq!(unsealed_ultra.arena_bytes, master_arena);

    // 7. DnamContainerView::from_bytes & DynamicDnamEngine load .dnam-seal directly and report EnclaveSealed
    let sealed_view = DnamContainerView::from_bytes(&sealed_bytes, &crypto).unwrap();
    assert_eq!(
        sealed_view.protection_profile,
        DnamProtectionProfile::EnclaveSealed
    );
    assert_eq!(
        sealed_view.arena_storage.as_bytes(),
        master_arena.as_slice()
    );

    let mut dyn_eng = DynamicDnamEngine::from_bytes_owned_with_crypto(
        &sealed_bytes,
        Tier::Ultra,
        48_000,
        &crypto,
    )
    .unwrap();
    assert_eq!(
        dyn_eng.protection_profile(),
        DnamProtectionProfile::EnclaveSealed
    );
    let in_l = [0.25_f32; 64];
    let in_r = [0.25_f32; 64];
    let mut out_l = [0.0_f32; 64];
    let mut out_r = [0.0_f32; 64];
    dyn_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    assert!(out_l.iter().any(|&v| v.abs() > 1e-6));

    // 8. Verify seal error branches (DNSL/DME1 wire parsing, unaligned inner, empty recipients, unknown OEM, fail-closed MP key, tampered seal)
    assert!(DnslPreambleWire::from_bytes(&[0u8; 32]).is_err());
    assert!(DnslDirectoryWire::from_bytes(&[0u8; 32]).is_err());
    assert!(Dme1HeaderWire::from_bytes(&[0u8; 16]).is_err());
    let missing_mp_crypto =
        RustCryptoProvider::new().with_device_key(device_id_hash, dev_secret_x25519);
    assert!(unseal_dnam_container(&sealed_bytes, &missing_mp_crypto).is_err());
    assert!(DnamContainerView::from_bytes(&sealed_bytes, &missing_mp_crypto).is_err());
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes[..100],
            std::slice::from_ref(&cert_anagram),
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[],
            &trusted_oems,
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );
    assert!(
        seal_dnam_container(
            &inner_dnam_bytes,
            &[cert_anagram],
            &[],
            &mp_seed,
            &eph_outer_sk,
            &cek_outer,
            &nonce_outer,
        )
        .is_err()
    );

    let wrong_crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, [99u8; 32])
        .with_marketplace_pubkey(mp_pk);
    assert!(DnamContainerView::from_bytes(&sealed_bytes, &wrong_crypto).is_err());

    let missing_dev_crypto = RustCryptoProvider::new()
        .with_device_key("00000000000000000000000000000000", dev_secret_x25519)
        .with_marketplace_pubkey(mp_pk);
    assert!(unseal_dnam_container(&sealed_bytes, &missing_dev_crypto).is_err());

    let mut tampered_seal = sealed_bytes;
    tampered_seal[200] ^= 0x01;
    assert!(DnamContainerView::from_bytes(&tampered_seal, &crypto).is_err());
}

#[test]
fn test_tier_capped_dnam_seal_with_dme1_trailer_preserves_presets_and_fails_closed() {
    let mut header = DnamJsonHeader::default_reference();
    header.presets = vec![PresetDef {
        id: "enclave_crunch".to_string(),
        name: "Enclave Crunch".to_string(),
        description: Some("DME1 encrypted preset".to_string()),
        controls: BTreeMap::new(),
        user_metadata: BTreeMap::new(),
    }];
    header.metadata.distribution_policy = Some(DistributionPolicy {
        allowed_protection_profiles: vec![DnamProtectionProfile::EnclaveSealed],
        allowed_oem_vendors: vec!["darkglass:anagram_rk3582".to_string()],
    });

    let mut master_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    model.readouts[1].direct_w_odd[0][0] = 0.25;
    model.readouts[1].direct_w_odd[1][0] = 0.25;
    pack_sona_matryoshka_to_arena(&model, &mut master_arena);

    let creator_seed = [27u8; 32];
    let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
    let mp_seed = [39u8; 32];
    let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);
    let oem_seed = [59u8; 32];
    let oem_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_seed);

    let device_id_hash = "0123456789abcdef0123456789abcdef";
    let dev_secret_x25519 = [47u8; 32];
    let dev_pubkey_x25519 = RustCryptoProvider::x25519_pubkey_from_secret(&dev_secret_x25519);
    let cert = RustCryptoProvider::sign_oem_device_certificate(
        &oem_seed,
        "darkglass",
        "anagram_rk3582",
        device_id_hash,
        &dev_pubkey_x25519,
    )
    .unwrap();

    let imm_sha = compute_immutable_header_sha256(&header).unwrap();
    let master_sha = sha256_digest(&master_arena);
    let creator_msg = build_creator_signing_message(&imm_sha, &master_sha);
    let creator_sig = RustCryptoProvider::ed25519_sign(&creator_seed, &creator_msg);

    let mut receipt = MarketplaceReceipt {
        marketplace_id: "dnam.store".to_string(),
        license_id: "0192b6c0-7e4a-7000-8000-001122334455".to_string(),
        buyer_hash: encode_hex(&[0xAB; 32]),
        license_tier: "oem_bundle".to_string(),
        issued_at: "2026-10-27T00:00:00Z".to_string(),
        buyer_arena_sha256: encode_hex(&master_sha),
        buyer_tier_arena_sha256: None,
        encryption_mode: "none".to_string(),
        arena_nonce_96: None,
        ephemeral_pubkey_x25519: None,
        authorized_devices: vec![AuthorizedDevice {
            oem_vendor: "darkglass".to_string(),
            model_code: "anagram_rk3582".to_string(),
            device_id_hash: device_id_hash.to_string(),
            wrapped_cek: None,
        }],
        royalty_splits: Vec::new(),
        receipt_sig_ed25519: String::new(),
    };
    let receipt_msg =
        build_marketplace_receipt_signing_message(&imm_sha, &master_sha, &receipt).unwrap();
    receipt.receipt_sig_ed25519 =
        encode_hex(&RustCryptoProvider::ed25519_sign(&mp_seed, &receipt_msg));

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&imm_sha),
        master_arena_sha256: encode_hex(&master_sha),
        tier_arena_sha256: None,
        creator_signature: CreatorSignature {
            signer_id: "did:web:darkglass.com".to_string(),
            pubkey_ed25519: encode_hex(&creator_pk),
            sig_ed25519: encode_hex(&creator_sig),
        },
        marketplace_receipt: Some(receipt),
    });

    let inner_dnam_bytes =
        build_dnam_container(&mut header, &master_arena, &master_arena, false).unwrap();
    let sealed_bytes = seal_dnam_container(
        &inner_dnam_bytes,
        &[cert],
        &[("darkglass", oem_pk)],
        &mp_seed,
        &[89u8; 32],
        &[0xA6_u8; 32],
        &[0x78_u8; 12],
    )
    .unwrap();
    assert_eq!(
        &sealed_bytes[sealed_bytes.len() - 16..sealed_bytes.len() - 12],
        b"DMUE"
    );

    let crypto = RustCryptoProvider::new()
        .with_device_key(device_id_hash, dev_secret_x25519)
        .with_marketplace_pubkey(mp_pk);

    // 1. Tier-capped unsealing (Tier::Full on unstripped Ultra .dnam-seal with no tier_arena_sha256)
    //    must succeed and preserve decrypted DME1 presets.
    let unsealed_full = unseal_dnsl_up_to_tier(&sealed_bytes, &crypto, Tier::Full).unwrap();
    assert_eq!(unsealed_full.retained_tier, Tier::Full);
    assert!(unsealed_full.mutable_json.is_some());

    let view_full =
        DnamContainerView::from_bytes_with_max_tier(&sealed_bytes, &crypto, Tier::Full).unwrap();
    assert_eq!(view_full.preamble.highest_tier(), Tier::Full);
    assert_eq!(
        view_full.arena_storage.as_bytes().len(),
        ARENA_OFFSET_FULL_END as usize
    );
    assert_eq!(view_full.header.presets.len(), 1);
    assert_eq!(view_full.header.presets[0].id, "enclave_crunch");

    // 2. DnamEngine with enclave crypto decrypts DME1 presets at Tier::Full and processes audio,
    //    while DnamEngine::from_bytes_with_max_tier (DefaultCryptoProvider without keys) fails closed.
    let mut eng_crypto = DnamEngine::<DefaultMathBackend>::from_bytes_with_crypto(
        &sealed_bytes,
        Tier::Full,
        48_000,
        &crypto,
    )
    .unwrap();
    eng_crypto.apply_preset("enclave_crunch").unwrap();
    let in_l = [0.25_f32; 32];
    let in_r = [-0.25_f32; 32];
    let (mut out_l, mut out_r) = ([0.0_f32; 32], [0.0_f32; 32]);
    eng_crypto.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    assert!(out_l.iter().any(|&v| v.abs() > 1e-6));

    let err_default_crypto = DnamEngine::<DefaultMathBackend>::from_bytes_with_max_tier(
        &sealed_bytes,
        Tier::Full,
        Tier::Full,
        48_000,
    )
    .unwrap_err();
    assert!(matches!(
        err_default_crypto,
        DnamError::CryptoVerificationFailed(_)
    ));

    // 3. Also verify DnamEngine::from_bytes_with_max_tier succeeds at Tier::Full on an open
    //    unstripped Ultra .dnam with presets (no tier_arena_sha256) and preserves presets.
    let mut open_hdr = header;
    open_hdr.metadata.distribution_policy = None;
    open_hdr.metadata.signatures = None;
    let open_ultra_bytes =
        build_dnam_container(&mut open_hdr, &master_arena, &master_arena, false).unwrap();
    let mut eng_max_full = DnamEngine::<DefaultMathBackend>::from_bytes_with_max_tier(
        &open_ultra_bytes,
        Tier::Full,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_max_full.tier(), Tier::Full);
    eng_max_full.apply_preset("enclave_crunch").unwrap();

    // 4. Truncating or tampering DME1 / DMUE on the tier-capped .dnam-seal must fail closed.
    let trunc_dmue = &sealed_bytes[..sealed_bytes.len() - 8];
    assert!(unseal_dnsl_up_to_tier(trunc_dmue, &crypto, Tier::Full).is_err());
    assert!(DnamContainerView::from_bytes_with_max_tier(trunc_dmue, &crypto, Tier::Full).is_err());

    let trunc_dme1 = &sealed_bytes[..sealed_bytes.len() - 24];
    assert!(unseal_dnsl_up_to_tier(trunc_dme1, &crypto, Tier::Full).is_err());
    assert!(DnamContainerView::from_bytes_with_max_tier(trunc_dme1, &crypto, Tier::Full).is_err());

    let mut bad_dmue_magic = sealed_bytes.clone();
    let magic_off = bad_dmue_magic.len() - 16;
    bad_dmue_magic[magic_off] ^= 0x01;
    assert!(unseal_dnsl_up_to_tier(&bad_dmue_magic, &crypto, Tier::Full).is_err());
    assert!(
        DnamContainerView::from_bytes_with_max_tier(&bad_dmue_magic, &crypto, Tier::Full).is_err()
    );

    let mut bad_dmue_len = sealed_bytes.clone();
    let len_off = bad_dmue_len.len() - 12;
    bad_dmue_len[len_off] ^= 0x04;
    assert!(unseal_dnsl_up_to_tier(&bad_dmue_len, &crypto, Tier::Full).is_err());
    assert!(
        DnamContainerView::from_bytes_with_max_tier(&bad_dmue_len, &crypto, Tier::Full).is_err()
    );

    let mut bad_dme1_ct = sealed_bytes;
    let ct_off = bad_dme1_ct.len() - 24;
    bad_dme1_ct[ct_off] ^= 0x01;
    assert!(unseal_dnsl_up_to_tier(&bad_dme1_ct, &crypto, Tier::Full).is_err());
    assert!(
        DnamContainerView::from_bytes_with_max_tier(&bad_dme1_ct, &crypto, Tier::Full).is_err()
    );
}
