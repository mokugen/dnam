//! Empirical `SONA` (`v3.0`) Constructors, Calibration Roles, `W_ctrl` Modulation, Isomorphism,
//! and Property Verification Suite (`tests/sona_calibration_and_lifecycle.rs`).
//!
//! Canonical Reference: `docs/specs/architecture/sona_architecture.md` and `docs/specs/dsp/parametric_control_bus.md`.

#![forbid(unsafe_code)]

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use std::collections::BTreeMap;

use arena_builder::build_populated_sona_container;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, build_dnam_container, build_v3_open_dnam_container,
    strip_container_to_tier, validate_v3_open_dnam_layout,
};
use dnam_engine::graph::cabinet::CabinetHot;
use dnam_engine::graph::sona_core::{SonaColdTierModel, SonaHotSlab, SonaMatryoshkaModel};
use dnam_engine::graph::weights::{
    ARENA0_W_CTRL_OFFSET, pack_sona_matryoshka_to_arena, write_f32_le,
};
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::AlignedFp16Block;
use dnam_engine::operators::c2_wavefolder;
use dnam_engine::provenance::{
    VerificationStatus, apply_sona_sign_gauge_watermark, derive_sona_sign_gauge_40,
    extract_structural_hash_224d, structural_cosine_similarity,
};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamError, DnamJsonHeader, TempoSubdivision,
};
use dnam_engine::{DnamEngine, DynamicDnamEngine, FtzDazGuard, Tier};

/// Verifies `from_bytes_owned_with_max_tier`, `from_bytes_with_scratch`, `with_max_tier`,
/// `reload_model_in_place`, all `ControlCalibrationRole` variants (`DelayTimeMs`, `Feedback`,
/// `StereoWidth`, `FreezeLatch` as both `Switch` and `Range`), and `sync_tempo`.
#[test]
fn test_engine_constructors_calibration_roles_and_tempo_sync() {
    let (mut header, arena, _container) = build_populated_sona_container();
    header.io_router.sub_crossover_hz = 180.0;
    header.spatial_tank.tzf_reference_tap_samples = 24.0;
    header.spatial_tank.switch_xfade_samples = 96;
    header.control_bus.head_trust_region_radius = 12.0;
    header.io_config.capabilities.default_blend = 0.85;
    header.io_config.capabilities.wet_polarity_inverted = true;
    header.io_config.leveling.auto_unity_enabled = true;
    header.io_config.leveling.calibrated_trim_db = -1.5;
    header.controls_schema = vec![
        ControlSchemaDef {
            id: "delay_ms".to_string(),
            name: "Delay".to_string(),
            kind: ControlKind::Range,
            unit: Some("ms".to_string()),
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([20.0, 800.0]),
            safe_clamped_range: Some([20.0, 800.0]),
            default: serde_json::json!(250.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::DelayTimeMs,
                lobatto_physical_anchors: Some([20.0, 250.0, 800.0]),
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: Some(TempoSubdivision::EighthDotted),
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "feedback".to_string(),
            name: "Feedback".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::Feedback,
                lobatto_physical_anchors: Some([0.1, 0.5, 1.15]),
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "width".to_string(),
            name: "Stereo Width".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(1.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::StereoWidth,
                lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "freeze_sw".to_string(),
            name: "Freeze Switch".to_string(),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!("Normal"),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: Some(vec!["Normal".to_string(), "Freeze".to_string()]),
            active_options: Some(vec!["Normal".to_string(), "Freeze".to_string()]),
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::FreezeLatch,
                lobatto_physical_anchors: None,
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
        ControlSchemaDef {
            id: "freeze_rng".to_string(),
            name: "Freeze Pedal".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(0.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::FreezeLatch,
                lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
                conditional_physical_anchors: Vec::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        },
    ];

    let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

    // 1. Test from_bytes_owned_with_max_tier and with_max_tier
    let mut eng = DnamEngine::<ScalarBackend>::from_bytes_owned_with_max_tier(
        &container,
        Tier::Lite,
        Tier::Pro,
        48_000,
    )
    .unwrap();
    assert_eq!(eng.max_supported_tier(), Tier::Pro);
    assert!((eng.io_router.crossover_hz - 180.0).abs() < 1e-4);
    assert!((eng.zoh().tau_ref_samples_host - 24.0).abs() < 1e-6);
    assert_eq!(eng.zoh().switch_xfade_samples_host, 96);
    assert!((eng.control_bus.inv_trust_radius_mh2 - 1.0 / 144.0).abs() < 1e-6);
    eng.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng.set_tier(Tier::Ultra);
    assert_eq!(eng.tier(), Tier::Pro);

    // 2. Test calibration roles (DelayTimeMs, Feedback, StereoWidth, FreezeLatch Switch & Range)
    eng.set_control_value(0, 400.0);
    eng.set_control_value(1, 8.0);
    eng.set_control_value(2, 0.75);
    assert!((eng.stereo_width() - 0.75).abs() < 1e-4);

    eng.set_control_value(3, 1.0);
    assert!(eng.is_frozen());
    eng.set_control_value(3, 0.0);
    assert!(!eng.is_frozen());

    eng.set_control_value(4, 0.8);
    assert!(eng.is_frozen());
    eng.set_freeze(false);
    assert!(!eng.is_frozen());
    eng.set_stereo_width(0.5);
    assert!((eng.stereo_width() - 0.5).abs() < 1e-6);

    // 3. Test sync_tempo and reload_model_in_place
    let updated = eng.sync_tempo(120.0, Some(TempoSubdivision::Quarter), true);
    assert_eq!(updated, 1);

    let mut scratch = [AlignedFp16Block::default(); 4];
    let mut eng_scratch = DnamEngine::<ScalarBackend>::from_bytes_with_scratch(
        &container,
        &mut scratch,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_scratch.tier(), Tier::Full);
    let mut mono_out = [0.0f32; 8];
    eng_scratch.process_mono(&[0.1f32; 8], &mut mono_out);
    assert!(mono_out[0].is_finite());

    let mut eng_cap = DnamEngine::<ScalarBackend>::with_max_tier(Tier::Full);
    eng_cap
        .reload_model_in_place(&container, Tier::Lite, 48_000)
        .unwrap();
    assert_eq!(eng_cap.tier(), Tier::Lite);

    let dyn_eng = DynamicDnamEngine::from_bytes(&container, Tier::Full, 48_000).unwrap();
    assert_eq!(dyn_eng.tier(), Tier::Full);

    // 4. Verify hyper-wide stereo width (2.0) doubles L-R side difference vs reference (1.0)
    let mut eng_w1 =
        DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    let mut eng_w2 =
        DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    eng_w1.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng_w2.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng_w1.set_stereo_width(1.0);
    eng_w2.set_stereo_width(2.0);
    assert!((eng_w2.stereo_width() - 2.0).abs() < 1e-6);

    let in_l = [0.35_f32, -0.25, 0.45, -0.15];
    let in_r = [-0.35_f32, 0.25, -0.45, 0.15];
    let mut out1_l = [0.0_f32; 4];
    let mut out1_r = [0.0_f32; 4];
    let mut out2_l = [0.0_f32; 4];
    let mut out2_r = [0.0_f32; 4];
    eng_w1.process_block(&in_l, &in_r, &mut out1_l, &mut out1_r);
    eng_w2.process_block(&in_l, &in_r, &mut out2_l, &mut out2_r);
    for i in 0..4 {
        let side1 = out1_l[i] - out1_r[i];
        let side2 = out2_l[i] - out2_r[i];
        assert!(
            (side2 - 2.0 * side1).abs() < 1e-5,
            "Hyper-wide stereo width=2.0 failed at frame {i}: side1={side1}, side2={side2}"
        );
    }

    // 5. Verify W_ctrl dispatches dynamic modulations via ControlBus -> stage_targets
    let mut base_mod_arena = arena.clone();
    let w_ctrl_base = ARENA0_W_CTRL_OFFSET;
    base_mod_arena[w_ctrl_base..w_ctrl_base + 1_024].fill(0);
    let mut mod_arena = base_mod_arena.clone();
    // Row 0 (env_fc_scale), Row 2 (SONA ADAA drive scale), Row 9, Row 11 (tank_psi): col q1[0] (+16 B)
    write_f32_le(&mut mod_arena, w_ctrl_base + 16, 0.50);
    write_f32_le(&mut mod_arena, w_ctrl_base + 2 * 64 + 16, 0.40);
    write_f32_le(&mut mod_arena, w_ctrl_base + 9 * 64 + 16, 0.35);
    write_f32_le(&mut mod_arena, w_ctrl_base + 11 * 64 + 16, 0.25);
    let base_mod_container =
        build_dnam_container(&mut header, &base_mod_arena, &base_mod_arena, false).unwrap();
    let mod_container = build_dnam_container(&mut header, &mod_arena, &mod_arena, false).unwrap();
    let mut base_mod_eng =
        DnamEngine::<ScalarBackend>::from_bytes(&base_mod_container, Tier::Lite, 48_000).unwrap();
    let mut mod_eng =
        DnamEngine::<ScalarBackend>::from_bytes(&mod_container, Tier::Lite, 48_000).unwrap();
    base_mod_eng.set_control_immediate(0, 800.0);
    mod_eng.set_control_immediate(0, 800.0);
    assert!(mod_eng.control_bus.stage_targets[0].abs() > 1e-2);
    assert!(mod_eng.control_bus.stage_targets[2].abs() > 1e-2);
    assert!(mod_eng.control_bus.stage_targets[11].abs() > 1e-2);
    let mut bl = [0.0_f32; 8];
    let mut br = [0.0_f32; 8];
    let mut ml = [0.0_f32; 8];
    let mut mr = [0.0_f32; 8];
    base_mod_eng.process_block(&[0.3; 8], &[-0.3; 8], &mut bl, &mut br);
    mod_eng.process_block(&[0.3; 8], &[-0.3; 8], &mut ml, &mut mr);
    assert!(ml.iter().zip(bl.iter()).any(|(a, b)| (a - b).abs() > 1e-5));
}

/// verify[arch.isomorphism.guarantees]
/// verify[arch.isomorphism.sona-serialization]
/// verify[arch.isomorphism.stereo-preservation]
/// verify[arch.isomorphism.nested-bus]
/// verify[arch.refinements.summary]
/// verify[arch.verification.matrix]
#[test]
fn test_trainer_engine_isomorphism_stereo_preservation_and_nested_bus() {
    let _guard = FtzDazGuard::new();

    // Stage II SONA stereo serialization parity, 0-ULP sign-flip invariance & C2 wavefolder
    let cold_s2 = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
    let mut s2_core = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold_s2);
    let y2 = s2_core.step_stereo(0.4, -0.4);
    assert!(y2[0].is_finite() && y2[1].is_finite() && (y2[0] + y2[1]).abs() < 1e-6);
    let (wo, we) = c2_wavefolder(0.5);
    assert!(wo > 0.0 && we > 0.0);

    let mut header = DnamJsonHeader::default_reference();
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut mat = SonaMatryoshkaModel::dual_rate_default(false);
    mat.readouts.iter_mut().for_each(|ro| {
        ro.pre_cab_w_odd[0][0] = 0.25;
        ro.pre_cab_w_odd[1][0] = -0.25;
        ro.fir_l[0] = 0.5;
        ro.fir_r[0] = 0.5;
    });
    pack_sona_matryoshka_to_arena(&mat, &mut arena);

    // Verify 40-lane SONA 0-ULP sign-flip gauge watermark preserves hash & bit-exact audio
    let signs = derive_sona_sign_gauge_40(&[0x77; 32]);
    let mut gauged_arena = arena.clone();
    apply_sona_sign_gauge_watermark(&mut gauged_arena, &signs).unwrap();
    let h_a = extract_structural_hash_224d(&arena).unwrap();
    let h_b = extract_structural_hash_224d(&gauged_arena).unwrap();
    assert!((structural_cosine_similarity(&h_a, &h_b) - 1.0).abs() < 1e-12);

    let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let gauged_container =
        build_dnam_container(&mut header, &gauged_arena, &gauged_arena, false).unwrap();
    for &t in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let stripped = strip_container_to_tier(&container, t).unwrap();
        let stripped_g = strip_container_to_tier(&gauged_container, t).unwrap();
        let mut eng_s = DnamEngine::<ScalarBackend>::from_bytes(&stripped, t, 48_000).unwrap();
        let mut eng_d = DnamEngine::<DefaultMathBackend>::from_bytes(&stripped, t, 48_000).unwrap();
        let mut eng_g =
            DnamEngine::<DefaultMathBackend>::from_bytes(&stripped_g, t, 48_000).unwrap();
        let (mut ol_s, mut or_s) = ([0.0f32; 16], [0.0f32; 16]);
        let (mut ol_d, mut or_d) = ([0.0f32; 16], [0.0f32; 16]);
        let (mut ol_g, mut or_g) = ([0.0f32; 16], [0.0f32; 16]);
        eng_s.process_block(&[0.2; 16], &[-0.2; 16], &mut ol_s, &mut or_s);
        eng_d.process_block(&[0.2; 16], &[-0.2; 16], &mut ol_d, &mut or_d);
        eng_g.process_block(&[0.2; 16], &[-0.2; 16], &mut ol_g, &mut or_g);
        for i in 0..16 {
            assert!((ol_s[i] - ol_d[i]).abs() < 1e-6);
            assert!((or_s[i] - or_d[i]).abs() < 1e-6);
            assert_eq!(ol_d[i].to_bits(), ol_g[i].to_bits());
            assert_eq!(or_d[i].to_bits(), or_g[i].to_bits());
        }
    }

    // End-to-end v3.0 Open .dnam container loading & .dnam-prev rejection in DnamEngine
    let hdr_ref = DnamJsonHeader::default_reference();
    let hdr_bytes = serde_json::to_vec(&hdr_ref).unwrap();
    let t0 = vec![0u8; 4_608];
    let t1 = vec![0u8; 3_328];
    let t2 = vec![0u8; 7_936];
    let t3 = vec![0u8; 14_848];
    let v3_container = build_v3_open_dnam_container(
        &hdr_bytes,
        &[0u8; 64],
        &[0u8; 64],
        &[&t0, &t1, &t2, &t3],
        None,
        0,
    )
    .unwrap();
    let mut eng_v3 =
        DnamEngine::<ScalarBackend>::from_bytes(&v3_container, Tier::Ultra, 48_000).unwrap();
    assert_eq!(eng_v3.preamble().version_major, 3);
    let mut out_l = [0.0f32; 16];
    let mut out_r = [0.0f32; 16];
    eng_v3.process_block(&[0.1; 16], &[-0.1; 16], &mut out_l, &mut out_r);
    assert!(out_l.iter().all(|v| v.is_finite()));

    // Verify .dnam-prev (container_kind == "preview") is rejected by DnamEngine::from_bytes
    let mut hdr_val = serde_json::to_value(&hdr_ref).unwrap();
    hdr_val["container_kind"] = serde_json::json!("preview");
    let prev_hdr_bytes = serde_json::to_vec(&hdr_val).unwrap();
    let prev_container =
        build_v3_open_dnam_container(&prev_hdr_bytes, &[0u8; 64], &[0u8; 64], &[&t0], None, 0)
            .unwrap();
    assert_eq!(
        DnamEngine::<ScalarBackend>::from_bytes(&prev_container, Tier::Lite, 48_000).unwrap_err(),
        DnamError::PreviewContainerRejected
    );

    // Verify integrated runtime guards & saturation flags
    eng_v3.saturation_flags = 0;
    let bad_in = [f32::NAN, f32::INFINITY, 150.0, -0.25];
    let mut san_l = [0.0f32; 4];
    let mut san_r = [0.0f32; 4];
    eng_v3.process_block(&bad_in, &bad_in, &mut san_l, &mut san_r);
    assert!(san_l.iter().all(|v| v.is_finite()));
    assert_ne!(
        eng_v3.saturation_flags & dnam_engine::math::saturation_flags::INVALID_NONFINITE,
        0
    );
    assert_ne!(
        eng_v3.saturation_flags & dnam_engine::math::saturation_flags::OUTSIDE_CERTIFIED_INPUT,
        0
    );

    // ZOH-rescaled sample rate update & control decimation ratio
    eng_v3.set_sample_rate(96_000).unwrap();
    assert_eq!(eng_v3.sample_rate_hz(), 96_000);
    assert_eq!(eng_v3.zoh().control_decimation_host, 32);

    // Cabinet 16-sample delayed Kautz onset tap on CabinetHot
    let mut s3_cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
    s3_cold.readout.fir_l.fill(0.0);
    s3_cold.readout.fir_r.fill(0.0);
    s3_cold.readout.kautz_b_gamma[0] = 1.0;
    s3_cold.readout.kautz_c_l[0] = (1.0, 0.0);
    let mut cab = CabinetHot::from_cold(&s3_cold);
    for n in 0..16 {
        let [out_l, _] =
            cab.step_stereo::<DefaultMathBackend>(if n == 0 { 1.0 } else { 0.0 }, 0.0, 2, 1, false);
        assert_eq!(out_l, 0.0, "expected 16-sample onset silence at n={n}");
    }
    let [delayed_at_16, _] = cab.step_stereo::<DefaultMathBackend>(0.0, 0.0, 2, 1, false);
    assert!(
        delayed_at_16.abs() > 0.1,
        "expected non-zero Kautz onset response at sample 16, got {delayed_at_16}"
    );

    // Signed v3.0 DAU1 container verification & tampering rejection on DnamEngine::from_bytes
    #[cfg(feature = "crypto")]
    {
        use dnam_engine::provenance::{
            Dau1Envelope, RustCryptoProvider, build_creator_manifest_v1, build_plaintext_shared_v1,
            compute_plaintext_prefixes_v1, sha256_digest,
        };
        let creator_seed = [0x42u8; 32];
        let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
        let dummy_env = Dau1Envelope {
            auth_kind: 0,
            creator_pubkey: creator_pk,
            creator_max_tier: 1,
            master_prefixes: vec![[0u8; 32], [0u8; 32]],
            creator_signature: [0u8; 64],
            marketplace_pubkey: None,
            buyer_max_tier: None,
            policy_bytes: Vec::new(),
            buyer_prefixes: Vec::new(),
            marketplace_signature: None,
        };
        let dummy_auth = dummy_env.to_bytes().unwrap();
        let draft =
            build_v3_open_dnam_container(&hdr_bytes, &[0u8; 64], &dummy_auth, &[&t0, &t1], None, 0)
                .unwrap();
        let shared_framing = build_plaintext_shared_v1(&draft[0..64], &draft[64..192], &[0u8; 64]);
        let hdr_sha = sha256_digest(&hdr_bytes);
        let master_pfxs = compute_plaintext_prefixes_v1(
            b"DNAM-MASTER-PREFIX-V1",
            &hdr_sha,
            &shared_framing,
            &[&t0, &t1],
        )
        .unwrap();
        let manifest = build_creator_manifest_v1(&creator_pk, &hdr_sha, &master_pfxs).unwrap();
        let sig = RustCryptoProvider::ed25519_sign(&creator_seed, &manifest);
        let signed_env = Dau1Envelope {
            auth_kind: 0,
            creator_pubkey: creator_pk,
            creator_max_tier: 1,
            master_prefixes: master_pfxs,
            creator_signature: sig,
            marketplace_pubkey: None,
            buyer_max_tier: None,
            policy_bytes: Vec::new(),
            buyer_prefixes: Vec::new(),
            marketplace_signature: None,
        };
        let signed_auth = signed_env.to_bytes().unwrap();
        let signed_v3 = build_v3_open_dnam_container(
            &hdr_bytes,
            &[0u8; 64],
            &signed_auth,
            &[&t0, &t1],
            None,
            0,
        )
        .unwrap();
        let eng_signed =
            DnamEngine::<ScalarBackend>::from_bytes(&signed_v3, Tier::Full, 48_000).unwrap();
        assert!(matches!(
            eng_signed.verification_status(),
            VerificationStatus::CreatorVerified { .. }
        ));

        let stripped_lite = strip_container_to_tier(&signed_v3, Tier::Lite).unwrap();
        let eng_stripped =
            DnamEngine::<ScalarBackend>::from_bytes(&stripped_lite, Tier::Lite, 48_000).unwrap();
        assert!(matches!(
            eng_stripped.verification_status(),
            VerificationStatus::CreatorVerified { .. }
        ));

        let layout = validate_v3_open_dnam_layout(&signed_v3).unwrap();
        let mut tampered = signed_v3.clone();
        tampered[layout.tier_start] ^= 0xFF;
        assert!(DnamEngine::<ScalarBackend>::from_bytes(&tampered, Tier::Full, 48_000).is_err());
    }

    let snap = eng_v3.telemetry_snapshot();
    assert_ne!(
        snap.overflow_or_saturation_flags
            & dnam_engine::math::saturation_flags::OUTSIDE_CERTIFIED_INPUT,
        0
    );
    assert!(!eng_v3.supports_freeze());
    eng_v3.set_freeze(true);
    assert!(!eng_v3.is_frozen());
}
