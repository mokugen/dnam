//! End-to-End `SONA` (`v3.0`) Tier Equivalence, Multiport Patchbay (`4CM`/`WDW`/`Align`),
//! Calibration Roles, ZOH Multi-Rate, `4x` Bounce, Container Framing, and Architecture Spec
//! Verification Suite (`tests/sona_tier_and_multiport_lifecycle.rs`).

#![forbid(unsafe_code)]

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;
#[path = "sona_test_support/calibration_fixture.rs"]
mod calibration_fixture;

use core::mem::{align_of, size_of};

use arena_builder::{
    build_cext_dispatch, build_creator_dau1_container_from_dnam_bin,
    build_dnam_bin_from_header_wires, build_populated_sona_container,
    build_sona_container_from_header_options, build_sona_extended_container, build_sona_header,
    build_sona_wires,
};
use calibration_fixture::build_calibration_controls_schema;
use dnam_engine::container::{
    ARENA_FULL_END_CL, ARENA_LITE_END_CL, ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END,
    ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END, ARENA_PRO_END_CL, ARENA_ULTRA_END_CL,
    DnamContainerView, DnamDirectoryWire, DnamPreamble, DnamPreambleWire, V3ContainerBuildSpec,
    build_dnam_container_from_dnam_bin, build_v3_dnam_container_with_dbin,
    cp1_contributor_record_bytes, strip_container_to_tier,
};
use dnam_engine::engine::patchbay::eval_wdw_routing_outputs;
use dnam_engine::engine::{HostShellPatchbay, IoRoutingMode, ParallelAlignmentMode, WdwDryTapMode};
use dnam_engine::memory::AlignedFp16Block;
use dnam_engine::provenance::{
    Dme1HeaderWire, DnplPreambleWire, DnslPreambleWire, RustCryptoProvider, VerificationStatus,
    apply_sona_sign_gauge_watermark, derive_sona_sign_gauge_40, extract_structural_hash_224d,
    structural_cosine_similarity,
};
use dnam_engine::schema::{DnamError, DnamJsonHeader, TempoSubdivision};
use dnam_engine::{
    DnamAudioIo, DnamEngine, DynamicDnamEngine, FtzDazGuard, OFFLINE_BOUNCE_PDC_SAMPLES, Tier,
    extract_dnam_bin,
};
use dnam_kernels::{
    FULL_SPEC, GpuVec4, HilbertH1Kernel, KernelVec4, LITE_SPEC, OpNPreHotState, PRO_SPEC,
    QuadratureHysteresis2D, T1DcBlockerKernel, T1SubHarmonicKernel, ULTRA_SPEC,
    eval_c2_wavefolder_adaa_vec4, f32_to_fp16, fp16_to_f32, prewarp_tan_pi_f_over_fs_f32,
    read_pitch_grain_m2_f32, step_sona_block_stereo8,
};
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    DnamBinHeaderWire, DnamBinView,
};
use dnam_vm::control::{Cp1ContributorSlot, ExactIntFactorAccumulator, control_stride_for_rate};
use dnam_vm::graph::{
    Bounce4xOversampler, CONVOLVER_KAUTZ_ONSET_DELAY_48K, KautzStereoBank4, SONA_LITE_HOT_BYTES,
    SonaHotSlab,
};
use dnam_vm::math::{
    DefaultMathBackend, ScalarBackend, eval_monotone_log2_f32_bits, saturation_flags,
};
use proptest::prelude::*;
use zerocopy::FromBytes;

const CANONICAL_SAMPLE_RATES: [u32; 15] = [
    44_100, 48_000, 64_000, 88_200, 96_000, 128_000, 176_400, 192_000, 220_500, 240_000, 256_000,
    288_000, 320_000, 352_800, 384_000,
];

/// verify[container.dnam.tier-prefix]
/// verify[arch.tiers.arena-hierarchy]
/// verify[arch.tiers.hardware-budget]
/// verify[arch.tiers.maximize-quality-within-budget]
/// verify[arch.tiers.state-ram]
/// verify[arch.tiers.efficiency]
/// verify[arch.tiers.transition-contract]
#[test]
fn test_exact_tier_equivalence_unstripped_vs_stripped_with_dbin_and_bounce() {
    assert_eq!(ARENA_OFFSET_LITE_END, 4_608);
    assert_eq!(ARENA_OFFSET_FULL_END, 7_936);
    assert_eq!(ARENA_OFFSET_PRO_END, 15_872);
    assert_eq!(ARENA_OFFSET_ULTRA_END, 30_720);
    assert_eq!(
        (
            ARENA_LITE_END_CL,
            ARENA_FULL_END_CL,
            ARENA_PRO_END_CL,
            ARENA_ULTRA_END_CL
        ),
        (72, 124, 248, 480)
    );
    const { assert!(SONA_LITE_HOT_BYTES <= 4_608) };
    assert!(SonaHotSlab::<'_, DefaultMathBackend>::active_hot_bytes(Tier::Lite, false) <= 4_608);
    assert_eq!(
        (
            LITE_SPEC.total_lanes(),
            FULL_SPEC.total_lanes(),
            PRO_SPEC.total_lanes(),
            ULTRA_SPEC.total_lanes()
        ),
        (16, 24, 32, 40)
    );
    assert_eq!(
        (
            LITE_SPEC.fir_taps,
            FULL_SPEC.fir_taps,
            PRO_SPEC.fir_taps,
            ULTRA_SPEC.fir_taps
        ),
        (32, 48, 96, 128)
    );
    assert_eq!(
        (
            LITE_SPEC.kautz_pairs,
            FULL_SPEC.kautz_pairs,
            PRO_SPEC.kautz_pairs,
            ULTRA_SPEC.kautz_pairs
        ),
        (8, 12, 24, 32)
    );

    let (_hdr, _arena, ultra_container) = build_populated_sona_container();
    let preamble = DnamPreamble::from_bytes(&ultra_container).unwrap();
    assert_eq!(preamble.version_major, 3);
    assert_eq!(preamble.arena_lite_end_cl, 72);
    assert_eq!(preamble.arena_full_end_cl, 124);
    assert_eq!(preamble.arena_pro_end_cl, 248);
    assert_eq!(preamble.arena_ultra_end_cl, 480);
    assert_eq!(preamble.tensor_arena_offset % 64, 0);
    assert!(preamble.dbin_prefix_bytes > 0);
    let _ = DnamBinView::parse(extract_dnam_bin(&ultra_container).unwrap()).unwrap();

    // Verify Sample-0 DC resting equilibrium (zero input produces 0.000000 on sample 0)
    let mut eng_dc =
        DnamEngine::<DefaultMathBackend>::from_bytes(&ultra_container, Tier::Ultra, 48_000)
            .unwrap();
    eng_dc.reset_to_dc_equilibrium();
    let (mut zl, mut zr) = ([0.0_f32; 16], [0.0_f32; 16]);
    eng_dc.process_block(&[0.0; 16], &[0.0; 16], &mut zl, &mut zr);
    for i in 0..16 {
        assert!(zl[i].abs() < 1e-7 && zr[i].abs() < 1e-7);
    }

    let in_l: Vec<f32> = (0..128).map(|i| 0.3 * ((i as f32) * 0.21).sin()).collect();
    let in_r: Vec<f32> = (0..128)
        .map(|i| -0.25 * ((i as f32) * 0.17).cos())
        .collect();
    let run_pair = |eng: &mut DnamEngine<'_, ScalarBackend>| {
        let (mut ol, mut or) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
        eng.process_block(&in_l, &in_r, &mut ol, &mut or);
        (ol, or)
    };

    let mut eng_ultra_ref =
        DnamEngine::<ScalarBackend>::from_bytes(&ultra_container, Tier::Ultra, 48_000).unwrap();
    eng_ultra_ref.set_control_immediate(0, 8.8);
    eng_ultra_ref.set_control_immediate(2, 320.0);
    let (l_ultra_only, _) = run_pair(&mut eng_ultra_ref);

    let mut tier_outputs = Vec::new();
    for tier in [Tier::Lite, Tier::Full, Tier::Pro] {
        let stripped = strip_container_to_tier(&ultra_container, tier).unwrap();
        let stripped_dbin = extract_dnam_bin(&stripped).unwrap();
        assert_eq!(
            DnamBinView::parse(stripped_dbin).unwrap().retained_tier,
            tier
        );

        let mut eng_st = DnamEngine::<ScalarBackend>::from_bytes(&stripped, tier, 48_000).unwrap();
        let mut eng_ub =
            DnamEngine::<ScalarBackend>::from_bytes(&ultra_container, tier, 48_000).unwrap();
        let mut eng_us =
            DnamEngine::<ScalarBackend>::from_bytes(&ultra_container, Tier::Ultra, 48_000).unwrap();
        eng_us.set_tier(tier);
        eng_us.reset_to_dc_equilibrium();

        let (l_def, _) = run_pair(
            &mut DnamEngine::<ScalarBackend>::from_bytes(&stripped, tier, 48_000).unwrap(),
        );
        for eng in [&mut eng_st, &mut eng_ub, &mut eng_us] {
            eng.set_control_immediate(0, 8.8);
            eng.set_control_immediate(2, 320.0);
        }
        let (l_st, r_st) = run_pair(&mut eng_st);
        let (l_ub, r_ub) = run_pair(&mut eng_ub);
        let (l_us, r_us) = run_pair(&mut eng_us);

        assert!(l_st.iter().any(|&v| v.abs() > 1e-4));
        let ctrl_delta = l_st
            .iter()
            .zip(l_def.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            ctrl_delta > 1e-5,
            "control perturbation inactive on {tier:?}"
        );
        assert_eq!(
            l_ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            r_ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            r_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l_us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            r_us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            r_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_ne!(
            l_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_ultra_only.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        for eng in [&mut eng_st, &mut eng_ub, &mut eng_us] {
            eng.set_control_value(0, 3.5);
            eng.set_control_value(2, 180.0);
        }
        let (sl_st, sr_st) = run_pair(&mut eng_st);
        let (sl_ub, sr_ub) = run_pair(&mut eng_ub);
        let (sl_us, sr_us) = run_pair(&mut eng_us);
        assert_eq!(
            sl_ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sl_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            sr_ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sr_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            sl_us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sl_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            sr_us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sr_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        tier_outputs.push(l_st);
    }
    tier_outputs.push(l_ultra_only);
    for t in 0..3 {
        let diff: f32 = tier_outputs[t]
            .iter()
            .zip(tier_outputs[t + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "adjacent tiers {t} vs {} should differ", t + 1);
    }

    // Offline bounce maximizes tier when maximize_tier_on_bounce=true while preserving live tier
    eng_ultra_ref.set_tier(Tier::Lite);
    eng_ultra_ref.set_maximize_tier_on_bounce(true);
    let (mut bl, mut br) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    eng_ultra_ref.render_offline_bounce(&in_l, &in_r, &mut bl, &mut br);
    assert_eq!(eng_ultra_ref.tier(), Tier::Lite);
    assert!(bl.iter().any(|&v| v.abs() > 1e-4) && bl.iter().all(|v| v.is_finite()));
}

/// verify[arch.pillars.dual-space-control]
/// verify[arch.pillars.cp1-projections]
/// verify[arch.lifecycle.dual-rate]
/// verify[arch.isomorphism.guarantees]
/// verify[arch.isomorphism.sona-serialization]
/// verify[arch.isomorphism.stereo-preservation]
/// verify[arch.isomorphism.nested-bus]
/// verify[arch.refinements.summary]
/// verify[arch.verification.matrix]
#[test]
fn test_sona_constructors_calibration_roles_cext_isomorphism_and_dau1() {
    let mut header = build_sona_header();
    header.transport.tzf_reference_tap_samples = 24.0;
    header.transport.switch_xfade_samples = 96;
    header.control_bus.head_trust_region_radius = 12.0;
    header.io_config.capabilities.default_blend = 0.85;
    header.io_config.capabilities.wet_polarity_inverted = true;
    header.io_config.leveling.auto_unity_enabled = true;
    header.io_config.leveling.calibrated_trim_db = -1.5;
    header.controls_schema = build_calibration_controls_schema();

    let (_, dbin, container) =
        build_sona_container_from_header_options(&mut header, Tier::Ultra, false, false, true, &[]);
    let mut eng = DnamEngine::<ScalarBackend>::from_bytes_owned_with_max_tier(
        &container,
        Tier::Lite,
        Tier::Pro,
        48_000,
    )
    .unwrap();
    assert_eq!(eng.max_supported_tier(), Tier::Pro);
    eng.set_tier(Tier::Ultra);
    assert_eq!(eng.tier(), Tier::Pro);
    eng.configure_host_mix_and_leveling(1.0, 0.0, false);

    eng.set_control_value(0, 400.0);
    eng.set_control_value(1, 8.0);
    eng.set_control_value(2, 0.75);
    assert!((eng.stereo_width() - 0.75).abs() < 1e-4);
    eng.set_control_value(8, 0.0);
    assert!((eng.stereo_width() - 0.0).abs() < 1e-6);
    eng.set_stereo_width(0.5);
    assert!((eng.stereo_width() - 0.5).abs() < 1e-6);

    assert_eq!(
        eng.sync_tempo(120.0, Some(TempoSubdivision::Quarter), true),
        3
    );
    let _ = eng.register_tap_tempo(Some(TempoSubdivision::Eighth), false);
    eng.set_bpm(115.0);

    eng.set_live_analog_drift(true);
    assert!(eng.live_analog_drift_enabled());
    eng.set_live_analog_drift(false);

    // Host write-lock telemetry + Op-T write-enable
    assert_eq!(eng.telemetry_snapshot().host_write_lock, 0);
    eng.set_quad_write_enable(true);
    assert_eq!(eng.telemetry_snapshot().host_write_lock, 1);
    eng.set_quad_write_enable(false);
    eng.set_op_t_write_enable(true);
    eng.set_op_t_write_enable(false);

    // CP1 overflow + scratch loader + with_max_tier + reload + DynamicDnamEngine + Linear/Muted
    let mut scratch = [AlignedFp16Block::default(); 4];
    let mut eng_scratch = DnamEngine::<ScalarBackend>::from_bytes_with_scratch(
        &container,
        &mut scratch,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_scratch.tier(), Tier::Full);
    let mut mono_out = [0.0_f32; 8];
    eng_scratch.process_mono(&[0.1_f32; 8], &mut mono_out);
    assert!(mono_out[0].is_finite());

    let mut eng_cap = DnamEngine::<ScalarBackend>::with_max_tier(Tier::Full);
    eng_cap
        .reload_model_in_place(&container, Tier::Lite, 48_000)
        .unwrap();
    assert_eq!(eng_cap.tier(), Tier::Lite);

    let cp1_slot = Cp1ContributorSlot {
        lanes: 2,
        ctrl_kind: 0,
        control_idx: 0,
        aux: 0xFF,
        main_base: [[0.1; 8]; 3],
        emb: [0.5, 0.25, 0.0],
        v_base: [0.1, -0.05, 0.02, 0.01],
        main_ultra: [[0.04; 8]; 3],
        v_ultra: [0.02; 4],
        has_ultra: true,
    };
    let (_, _, cp1_container) = build_sona_container_from_header_options(
        &mut build_sona_header(),
        Tier::Ultra,
        false,
        false,
        true,
        &[cp1_slot],
    );
    let mut eng_cp1 =
        DnamEngine::<DefaultMathBackend>::from_bytes(&cp1_container, Tier::Ultra, 48_000).unwrap();
    eng_cp1.apply_preset("tight_lead").unwrap();
    eng_cp1.apply_preset_by_index(0).unwrap();
    let (mut pl, mut pr) = ([0.0_f32; 32], [0.0_f32; 32]);
    eng_cp1.process_block(&[0.25; 32], &[-0.25; 32], &mut pl, &mut pr);
    assert!(pl.iter().any(|&v| v.abs() > 1e-5));

    let lin_container = build_sona_extended_container(true, false);
    let muted_container = build_sona_extended_container(false, true);
    let mut dyn_eng = DynamicDnamEngine::from_bytes(&lin_container, Tier::Full, 48_000).unwrap();
    dyn_eng.process_block(&[0.2; 32], &[-0.2; 32], &mut pl, &mut pr);
    let mut muted_eng =
        DnamEngine::<DefaultMathBackend>::from_bytes(&muted_container, Tier::Lite, 48_000).unwrap();
    muted_eng.process_block(&[0.2; 32], &[-0.2; 32], &mut pl, &mut pr);

    // Hyper-wide stereo_width(2.0) doubles L-R side difference vs reference (1.0)
    let ext_container = build_sona_extended_container(false, false);
    let mut eng_w1 =
        DnamEngine::<ScalarBackend>::from_bytes(&ext_container, Tier::Lite, 48_000).unwrap();
    let mut eng_w2 =
        DnamEngine::<ScalarBackend>::from_bytes(&ext_container, Tier::Lite, 48_000).unwrap();
    eng_w1.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng_w2.configure_host_mix_and_leveling(1.0, 0.0, false);
    eng_w1.set_stereo_width(1.0);
    eng_w2.set_stereo_width(2.0);
    let in_l = [0.35_f32, -0.25, 0.45, -0.15];
    let in_r = [-0.35_f32, 0.25, -0.45, 0.15];
    let (mut out1_l, mut out1_r) = ([0.0_f32; 4], [0.0_f32; 4]);
    let (mut out2_l, mut out2_r) = ([0.0_f32; 4], [0.0_f32; 4]);
    eng_w1.process_block(&in_l, &in_r, &mut out1_l, &mut out1_r);
    eng_w2.process_block(&in_l, &in_r, &mut out2_l, &mut out2_r);
    for i in 0..4 {
        let side1 = out1_l[i] - out1_r[i];
        let side2 = out2_l[i] - out2_r[i];
        assert!((side2 - 2.0 * side1).abs() < 1e-5);
    }

    // W_ctrl + CEXT slot 16 (op_t_freeze_mix_delta) dynamic dispatch alters processed audio
    let hdr_cext = build_sona_header();
    let (a0, _a1, _a2, _a3) =
        build_sona_wires(&hdr_cext.controls_schema, Tier::Lite, false, false, true);
    let cext = build_cext_dispatch();
    let dbin_no_cext =
        build_dnam_bin_from_header_wires(&hdr_cext, Tier::Lite, (&a0, None, None, None), None, &[]);
    let dbin_cext = build_dnam_bin_from_header_wires(
        &hdr_cext,
        Tier::Lite,
        (&a0, None, None, None),
        Some(&cext),
        &[],
    );
    let c_no_cext =
        build_dnam_container_from_dnam_bin(&mut hdr_cext.clone(), &dbin_no_cext).unwrap();
    let c_cext = build_dnam_container_from_dnam_bin(&mut hdr_cext.clone(), &dbin_cext).unwrap();
    let mut eng_nc =
        DnamEngine::<ScalarBackend>::from_bytes(&c_no_cext, Tier::Lite, 48_000).unwrap();
    let mut eng_c = DnamEngine::<ScalarBackend>::from_bytes(&c_cext, Tier::Lite, 48_000).unwrap();
    eng_nc.set_control_immediate(0, 9.0);
    eng_c.set_control_immediate(0, 9.0);
    let (mut ncl, mut ncr, mut cl, mut cr) =
        ([0.0_f32; 32], [0.0_f32; 32], [0.0_f32; 32], [0.0_f32; 32]);
    eng_nc.process_block(&[0.3; 32], &[-0.3; 32], &mut ncl, &mut ncr);
    eng_c.process_block(&[0.3; 32], &[-0.3; 32], &mut cl, &mut cr);
    assert!(ncl.iter().zip(cl.iter()).any(|(a, b)| (a - b).abs() > 1e-6));

    // Stage II 40-lane SONA sign-flip 0-ULP invariance + DAU1 signing with dbin_prefix
    let view_master = DnamBinView::parse(&dbin).unwrap();
    let master_arena = &dbin[view_master.header.arena0_offset.get() as usize..];
    let mut gauged = master_arena.to_vec();
    let signs = derive_sona_sign_gauge_40(&[0x77; 32]);
    apply_sona_sign_gauge_watermark(&mut gauged, &signs).unwrap();
    let h_a = extract_structural_hash_224d(master_arena).unwrap();
    let h_b = extract_structural_hash_224d(&gauged).unwrap();
    assert!((structural_cosine_similarity(&h_a, &h_b) - 1.0).abs() < 1e-12);

    #[cfg(feature = "crypto")]
    {
        let creator_seed = [0x42u8; 32];
        let (signed_v3, _, _) =
            build_creator_dau1_container_from_dnam_bin(&header, &dbin, &creator_seed, true);
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
        let tampered_off = signed_v3.len() - 4_200;
        let mut tampered = signed_v3.clone();
        tampered[tampered_off] ^= 0xFF;
        assert!(DnamEngine::<ScalarBackend>::from_bytes(&tampered, Tier::Full, 48_000).is_err());
    }
}

/// verify[arch.topology.signal-flow]
/// verify[arch.topology.io-router]
/// verify[arch.topology.auto-unity-leveling]
/// verify[arch.topology.spillover-trails]
/// verify[arch.bounce.oversampling]
#[test]
fn test_host_shell_multiport_4cm_wdw_align_zoh_bounce_and_trails() {
    let mut header = build_sona_header();
    header.io_config.capabilities.wet_polarity_inverted = true;
    header.io_config.leveling.auto_unity_enabled = true;
    header.io_config.leveling.calibrated_trim_db = -3.0;
    let (_, _, container) =
        build_sona_container_from_header_options(&mut header, Tier::Ultra, false, false, true, &[]);

    // 1. HostShellPatchbay S_{ret->send} == 0_{2x2}, ReturnAligner, ParallelAlignLine, eval_wdw_routing_outputs
    let mut pb_a = HostShellPatchbay::new();
    let mut pb_b = HostShellPatchbay::new();
    pb_a.set_routing_mode(IoRoutingMode::FourCableMethod);
    pb_b.set_routing_mode(IoRoutingMode::FourCableMethod);
    pb_a.configure_return_alignment(0.35, 0.75);
    pb_a.configure_parallel_alignment(ParallelAlignmentMode::FullBand64Tap, true, 0.5, 0.25);
    for step in 0..16 {
        let inst = 0.3 * ((step as f32) * 0.25).sin();
        let (_, send_a, _) = pb_a.route_pre_vm([inst, -inst, 0.0, 0.0]);
        let (_, send_b, _) = pb_b.route_pre_vm([inst, -inst, 0.9, -0.7]);
        assert_eq!(send_a[0].to_bits(), send_b[0].to_bits());
        assert_eq!(send_a[1].to_bits(), send_b[1].to_bits());
    }
    let wdw_ports =
        eval_wdw_routing_outputs(WdwDryTapMode::DirectDry, 0.1, [0.4, 0.2], 1.5, [0.6, 0.2]);
    assert!((wdw_ports[0] - 0.4).abs() < 1e-6);
    assert!((wdw_ports[1] - 0.7).abs() < 1e-6);
    assert!((wdw_ports[2] - 0.1).abs() < 1e-6);

    // 2. WDW width invariance on dry_c + 4CM/Sidechain/IO/auto-unity/polarity/trails/bypass
    let (in_l, in_r) = ([0.65_f32; 32], [-0.25_f32; 32]);
    for tap_mode in [WdwDryTapMode::PreSpatialCore, WdwDryTapMode::DirectDry] {
        let mut eng_w1 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_w2 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng_w1.set_wdw_enabled(true);
        eng_w2.set_wdw_enabled(true);
        eng_w1.set_wdw_mode(tap_mode);
        eng_w2.set_wdw_mode(tap_mode);
        eng_w1.set_stereo_width(1.0);
        eng_w2.set_stereo_width(1.75);

        let (mut l1, mut r1, mut d1) = ([0.0_f32; 32], [0.0_f32; 32], [0.0_f32; 32]);
        let (mut l2, mut r2, mut d2) = ([0.0_f32; 32], [0.0_f32; 32], [0.0_f32; 32]);
        eng_w1.process_block_wdw(&in_l, &in_r, &mut l1, &mut r1, &mut d1);
        eng_w2.process_block_wdw(&in_l, &in_r, &mut l2, &mut r2, &mut d2);
        assert!(l1.iter().zip(r1.iter()).any(|(a, b)| (a - b).abs() > 1e-4));
        for i in 0..32 {
            assert!((0.5 * (l2[i] + r2[i]) - 0.5 * (l1[i] + r1[i])).abs() < 1e-5);
            assert!((0.5 * (l2[i] - r2[i]) - 1.75 * 0.5 * (l1[i] - r1[i])).abs() < 1e-5);
            assert_eq!(d1[i], d2[i]);
        }
    }

    let mut eng = DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    eng.configure_return_alignment(0.25, 0.5);
    eng.configure_parallel_alignment(ParallelAlignmentMode::MinLatency4Tap, false, 0.0, 0.0);
    let snap = eng.telemetry_snapshot();
    assert!(snap.mix_wet_gain < 0.0 && snap.mix_wet_gain.abs() < 0.85);

    eng.set_bypass(true, true);
    assert_eq!(eng.telemetry_snapshot().bypassed, 1);
    let (mut bl, mut br) = ([0.0_f32; 32], [0.0_f32; 32]);
    eng.process_block(&in_l, &in_r, &mut bl, &mut br);
    eng.set_bypass(true, false);
    eng.process_block(&in_l, &in_r, &mut bl, &mut br);
    assert_eq!(bl, in_l);
    assert_eq!(br, in_r);
    eng.set_bypass(false, false);

    // 3. ZOH sweep across all 15 canonical sample rates (44.1 kHz .. 384 kHz)
    for &sr in &CANONICAL_SAMPLE_RATES {
        eng.set_sample_rate(sr).unwrap();
        assert_eq!(eng.sample_rate_hz(), sr);
        assert_eq!(
            eng.zoh().k_ctrl,
            control_stride_for_rate(sr as f32) as usize
        );
        eng.process_block(&in_l, &in_r, &mut bl, &mut br);
        assert!(bl.iter().all(|v| v.is_finite()) && br.iter().all(|v| v.is_finite()));
    }
    eng.set_sample_rate(48_000).unwrap();

    // 4. 4x offline bounce PDC=12 & 4CM send downsampling alias rejection
    assert_eq!(OFFLINE_BOUNCE_PDC_SAMPLES, 12);
    assert_eq!(Bounce4xOversampler::pdc_host_samples(), 12);
    let run_4cm = |offline: bool, il: &[f32; 64], ir: &[f32; 64], ret: &[f32; 64]| {
        let mut e =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        e.set_offline_bounce(offline);
        let (mut ml, mut mr, mut sl, mut sr_out) =
            ([0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64]);
        e.process_block_4cm(il, ir, ret, ret, &mut ml, &mut mr, &mut sl, &mut sr_out);
        (ml, mr, sl, sr_out)
    };
    let (step_l, step_r, ret_zero, ret_sig) =
        ([0.4_f32; 64], [-0.4_f32; 64], [0.0_f32; 64], [0.3_f32; 64]);
    let (_, _, send_la, send_ra) = run_4cm(true, &step_l, &step_r, &ret_zero);
    let (_, _, send_lb, send_rb) = run_4cm(true, &step_l, &step_r, &ret_sig);
    assert_eq!(send_la, send_lb);
    assert_eq!(send_ra, send_rb);

    let mut bounce_pdc = Bounce4xOversampler::new();
    let mut ir = [0.0_f32; 32];
    for (i, slot) in ir.iter_mut().enumerate() {
        let up = bounce_pdc.upsample_4x(if i == 0 { 1.0 } else { 0.0 }, 0.0);
        *slot = bounce_pdc.downsample_4x(up)[0];
    }
    let peak_idx = ir
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap();
    assert_eq!(peak_idx, OFFLINE_BOUNCE_PDC_SAMPLES);
    bounce_pdc.reset();
    let mut dc_settled = 0.0_f32;
    for _ in 0..48 {
        let up = bounce_pdc.upsample_4x(0.4, -0.4);
        dc_settled = bounce_pdc.downsample_4x(up)[0];
    }
    assert!((dc_settled - 0.4).abs() < 1e-5);

    let nyq: [f32; 64] = core::array::from_fn(|i| if (i & 1) == 0 { 0.5 } else { -0.5 });
    bounce_pdc.reset();
    let mut nyq_4x = [0.0_f32; 64];
    for i in 0..64 {
        let up = bounce_pdc.upsample_4x(nyq[i], nyq[i]);
        nyq_4x[i] = bounce_pdc.downsample_4x(up)[0];
    }
    let peak_4x = nyq_4x[32..].iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        peak_4x < 0.5 * 0.55,
        "half-band 4x peak at fs/2 = {peak_4x}"
    );

    let (main_bounce, _, _, _) = run_4cm(true, &step_l, &step_r, &ret_zero);
    let (main_live, _, _, _) = run_4cm(false, &step_l, &step_r, &ret_zero);
    assert!(main_bounce.iter().all(|v| v.is_finite()) && main_live.iter().all(|v| v.is_finite()));
    assert_ne!(main_bounce, main_live);

    let mut os_stop = Bounce4xOversampler::new();
    let mut settled_stop = 0.0_f32;
    for i in 0..32 {
        let [dl, dr] = os_stop.downsample_4x([[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]]);
        if i >= 16 {
            settled_stop = settled_stop.max(dl.abs()).max(dr.abs());
        }
    }
    assert!(settled_stop < 1e-4);

    eng.process_block_sidechain(&in_l, &in_r, &in_l, &in_r, &mut bl, &mut br);
    eng.process_block_io(DnamAudioIo {
        in_1: &in_l,
        in_2: Some(&in_r),
        in_3: Some(&in_l),
        in_4: Some(&in_r),
        out_1: &mut bl,
        out_2: Some(&mut br),
        out_3: None,
        out_4: None,
    });
    assert!(bl.iter().all(|v| v.is_finite()) && br.iter().all(|v| v.is_finite()));
}

/// verify[arch.pillars.dual-rate-plane]
/// verify[arch.pillars.universal-black-box]
/// verify[arch.fp.callback-env]
/// verify[arch.fp.range-saturation]
/// verify[arch.operators.tri-kernel]
/// verify[arch.operators.self-similar]
/// verify[arch.simd.lane-packing]
/// verify[arch.operators.differentiability]
/// verify[sdk.rust.zerocopy-wire-structs]
/// verify[sdk.rust.shared-nostd-kernels]
/// verify[arch.families.coverage]
/// verify[arch.families.f1-bistables]
/// verify[arch.families.f2-hilbert-ssb]
/// verify[arch.families.f3-pitch]
/// verify[arch.families.f4-dynamics]
/// verify[arch.families.f5-bitcrusher]
/// verify[arch.families.f6-formants]
/// verify[arch.families.f7-multi-input]
#[test]
fn test_operators_zerocopy_wires_kernels_guards_and_seven_families() {
    let _ftz = FtzDazGuard::new();

    // 1. Zerocopy wire struct layout sizes & prefix checks
    assert_eq!(size_of::<DnamPreamble>(), 128);
    assert_eq!(align_of::<DnamPreamble>(), 64);
    assert_eq!(size_of::<DnamPreambleWire>(), 64);
    assert_eq!(align_of::<DnamPreambleWire>(), 64);
    assert_eq!(size_of::<DnamDirectoryWire>(), 128);
    assert_eq!(size_of::<Arena0LiteWire>(), 4_608);
    assert_eq!(size_of::<Arena1FullDeltaWire>(), 3_328);
    assert_eq!(size_of::<Arena2ProDeltaWire>(), 7_936);
    assert_eq!(size_of::<Arena3UltraDeltaWire>(), 14_848);
    assert_eq!(size_of::<DnslPreambleWire>(), 64);
    assert_eq!(size_of::<DnplPreambleWire>(), 64);
    assert_eq!(size_of::<Dme1HeaderWire>(), 20);
    assert_eq!(cp1_contributor_record_bytes(3, false), Some(124));
    assert_eq!(cp1_contributor_record_bytes(3, true), Some(112));
    assert_eq!(control_stride_for_rate(48_000.0), 16);

    // 2. Shared #![no_std] kernels + exact-int ANOVA FM accumulator
    assert_eq!(fp16_to_f32(f32_to_fp16(0.5)), 0.5);
    let g = prewarp_tan_pi_f_over_fs_f32(1_000.0 / 48_000.0);
    assert!(g.is_finite() && g > 0.06 && g < 0.07);
    let (wf_o, wf_e) = eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(0.1), GpuVec4::splat(0.5));
    assert!(wf_o.0[0].is_finite() && wf_e.0[0] >= 0.0);
    let _ = step_sona_block_stereo8::<<DefaultMathBackend as dnam_vm::math::MathBackend>::Stereo8>
        as fn(_, _, _) -> _;

    let mut acc = ExactIntFactorAccumulator::new(2);
    acc.update_contributor(0, (0.5_f32).to_bits()).unwrap();
    acc.update_contributor(1, (0.25_f32).to_bits()).unwrap();
    assert_eq!(acc.interaction_f32_bits(), Some((0.125_f32).to_bits()));

    // 3. Seven hardware families F1..F7 via dnam_kernels + dnam_vm + DnamEngine
    let mut dc = T1DcBlockerKernel::new();
    let mut h1 = HilbertH1Kernel::default();
    let mut trk = T1SubHarmonicKernel::new();
    let mut hyst = QuadratureHysteresis2D::new();
    let d = dc.step(0.5);
    let (i_v, q_v) = h1.step(d);
    assert!(hyst.step(i_v, q_v).is_finite());
    assert!(trk.step(i_v, q_v).iter().all(|v| v.is_finite()));
    let (ssb_u, ssb_l) = (i_v * 0.8 - q_v * 0.6, i_v * 0.8 + q_v * 0.6);
    assert!(ssb_u.is_finite() && ssb_l.is_finite());

    let dc_ring = [0.625_f32; 32];
    assert!((read_pitch_grain_m2_f32(&dc_ring, 31, 20, 12.0, 6.0, 0.37) - 0.625).abs() < 1e-6);

    assert!(
        (eval_monotone_log2_f32_bits((0.5_f32).to_bits())
            .unwrap()
            .unwrap()
            - (-1.0))
            .abs()
            < 1e-5
    );

    let mut pre = OpNPreHotState::default();
    let (pl_pre, pr_pre) = pre.step_stereo(0.33, -0.33, 0.5, 8.0);
    assert!(pl_pre.is_finite() && pr_pre.is_finite());

    let mut k_bank = KautzStereoBank4 {
        b_gamma: [1.0, 0.0, 0.0, 0.0],
        lambda_re: [0.95, 0.0, 0.0, 0.0],
        lambda_im: [0.25, 0.0, 0.0, 0.0],
        c_re_l: [0.5, 0.0, 0.0, 0.0],
        c_re_r: [-0.5, 0.0, 0.0, 0.0],
        ..Default::default()
    };
    let (kl, kr) = k_bank.step_stereo::<DefaultMathBackend>(1.0, -1.0);
    assert!(kl.is_finite() && kr.is_finite() && CONVOLVER_KAUTZ_ONSET_DELAY_48K == 16);

    // 4. End-to-end runtime NaN / Inf / range-saturation guards & telemetry flags in DnamEngine
    let (_, _, container) = build_populated_sona_container();
    let mut eng = DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
    let bad_in = [f32::NAN, f32::INFINITY, 150.0, -0.25];
    let (mut san_l, mut san_r) = ([0.0_f32; 4], [0.0_f32; 4]);
    eng.process_block(&bad_in, &bad_in, &mut san_l, &mut san_r);
    assert!(san_l.iter().all(|v| v.is_finite()) && san_r.iter().all(|v| v.is_finite()));
    let snap = eng.telemetry_snapshot();
    assert_ne!(
        snap.saturation_flags & saturation_flags::INVALID_NONFINITE,
        0
    );
    assert_ne!(
        snap.saturation_flags & saturation_flags::OUTSIDE_CERTIFIED_INPUT,
        0
    );
}

#[test]
fn test_reload_containment_and_legacy_json_rejection() {
    let (_hdr, _arena, valid) = build_populated_sona_container();
    let mut prev_hdr = build_sona_header();
    prev_hdr.container_kind = "preview".to_string();
    let (_, dbin, prev) = build_sona_container_from_header_options(
        &mut prev_hdr,
        Tier::Ultra,
        false,
        false,
        true,
        &[],
    );
    let (hdr_wire, _) = DnamBinHeaderWire::read_from_prefix(&dbin).unwrap();
    let a0_off = hdr_wire.arena0_offset.get() as usize;
    let empty_lite = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    let bad_json = build_v3_dnam_container_with_dbin(&V3ContainerBuildSpec {
        protection_profile: 0,
        header_bytes: b"{not valid json",
        shared_bytes: &[],
        auth_bytes: &[0u8; 64],
        dbin_prefix_bytes: &dbin[..a0_off],
        tier_slices: &[empty_lite.as_slice()],
        mutable_frame_bytes: None,
        auth_kind: 0,
    })
    .unwrap();

    // Negative JSON validation: fail-closed rejection of retired legacy keys
    let mut base_json: serde_json::Value = serde_json::to_value(build_sona_header()).unwrap();
    for legacy_key in [
        "io_router",
        "supports_freeze",
        "freeze_mutes_dry",
        "sub_crossover_hz",
    ] {
        let mut with_legacy = base_json.clone();
        with_legacy[legacy_key] = serde_json::json!({});
        let raw = serde_json::to_vec(&with_legacy).unwrap();
        assert!(
            DnamJsonHeader::from_json_slice(&raw).is_err(),
            "legacy key `{legacy_key}` must fail closed"
        );
    }
    base_json["io_config"]["capabilities"]["supports_freeze"] = serde_json::json!(true);
    let cap_raw = serde_json::to_vec(&base_json).unwrap();
    assert!(DnamJsonHeader::from_json_slice(&cap_raw).is_err());

    let mut owned =
        DnamEngine::<ScalarBackend>::from_bytes_owned(&valid, Tier::Lite, 48_000).unwrap();
    assert!(matches!(
        owned.reload_model_in_place_owned(&prev, Tier::Lite, 48_000),
        Err(DnamError::PreviewContainerRejected)
    ));
    assert!(matches!(
        owned.reload_model_in_place_owned(&bad_json, Tier::Lite, 48_000),
        Err(DnamError::InvalidJsonHeader(_))
    ));

    let mut dyn_owned = DynamicDnamEngine::from_bytes_owned(&valid, Tier::Lite, 48_000).unwrap();
    assert!(matches!(
        dyn_owned.reload_model_in_place_owned(&prev, Tier::Lite, 48_000),
        Err(DnamError::PreviewContainerRejected)
    ));
    assert!(matches!(
        dyn_owned.reload_model_in_place_owned(&bad_json, Tier::Lite, 48_000),
        Err(DnamError::InvalidJsonHeader(_))
    ));
    assert!(matches!(
        DnamContainerView::from_bytes(&prev, &RustCryptoProvider::default()),
        Err(DnamError::PreviewContainerRejected)
    ));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_end_to_end_sona_engine_tiers_rates_and_finite_outputs(
        tier_u8 in 0u8..=3u8,
        rate_idx in 0usize..15usize,
        in_l in prop::array::uniform16(-5.0_f32..5.0_f32),
        in_r in prop::array::uniform16(-5.0_f32..5.0_f32),
    ) {
        let sr = CANONICAL_SAMPLE_RATES[rate_idx];
        let tier = Tier::from_u8(tier_u8);
        let (_, _, v3_container) = build_populated_sona_container();
        let mut eng = DnamEngine::<ScalarBackend>::from_bytes(&v3_container, tier, sr).unwrap();
        let mut out_l = [0.0f32; 16];
        let mut out_r = [0.0f32; 16];
        eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        prop_assert!(out_l.iter().all(|v| v.is_finite() && v.abs() <= 1.0e9));
        prop_assert!(out_r.iter().all(|v| v.is_finite() && v.abs() <= 1.0e9));
    }
}
