#![forbid(unsafe_code)]
//! Integration coverage for `v2` extended TORA DSP pipelines, multi-port topologies,
//! `DynamicDnamEngine`, `FeatureBusBackend` paths, `Stage1` ext, `Stage2` WH/LTI/`conv6`,
//! and `Stage4` BBD/FDN/shimmer/multi-tier taps.

#[path = "v2_test_support/arena_builder.rs"]
mod arena_builder;

use arena_builder::{build_v2_extended_container, build_v2_header, populate_v2_arena};
use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::control::Cp1ContributorSlot;
#[cfg(target_arch = "aarch64")]
use dnam_engine::math::{ArmSmeBackend, ArmSve2Backend};
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::AlignedFp16Block;
use dnam_engine::schema::TempoSubdivision;
use dnam_engine::stages::stage2_gru::{V3InterBlockToneSvfParams, V3InterBlockToneSvfState};
use dnam_engine::stages::{
    Stage1PreDynamics, Stage4SpatialTank, V3ColdTierModel, V3HotCore, WdwDryTapMode,
};
use dnam_engine::{DnamAudioIo, DnamEngine, DynamicDnamEngine, FtzDazGuard, Tier};

fn exercise_backend_full_matrix<B: dnam_engine::engine::EngineBackend>(bytes: &[u8]) {
    let n = 64usize;
    let in_l: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.11).sin()).collect();
    let in_r: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.17).cos()).collect();
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];

    let mut eng = DnamEngine::<B>::from_bytes(bytes, Tier::Lite, 48_000).unwrap();
    for &sr in &[44_100_u32, 48_000, 96_000] {
        eng.set_sample_rate(sr).unwrap();
        for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            eng.set_tier(tier);
            eng.set_live_analog_drift(true);
            assert!(eng.live_analog_drift_enabled());
            eng.set_control_value(0, 7.5);
            eng.set_control_immediate(1, 0.0);
            eng.set_freeze(true);
            eng.set_freeze_amount(0.8);
            eng.configure_host_mix_and_leveling(0.8, -1.5, true);
            eng.sync_tempo(125.0, Some(TempoSubdivision::Eighth), true);
            let _ = eng.register_tap_tempo(None, false);
            eng.set_bpm(110.0);
            eng.apply_preset("tight_lead").unwrap();
            eng.apply_preset_by_index(0).unwrap();

            eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
            assert!(out_l.iter().all(|v| v.is_finite()));

            let mut ip_l = in_l.clone();
            let mut ip_r = in_r.clone();
            eng.process_block_in_place(&mut ip_l, &mut ip_r);
            eng.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
            eng.process_mono(&in_l, &mut out_l);
            eng.set_live_analog_drift(false);
            eng.set_freeze(false);
            eng.reset_controls_to_defaults();
        }
    }
    eng.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
    assert!(out_l.iter().all(|v| v.is_finite()));
    let snap = eng.telemetry_snapshot();
    assert!(snap.active_tier <= 3);
}

#[test]
fn test_v2_wh_and_lti_across_tiers_sample_rates_and_all_backends() {
    let wh_bytes = build_v2_extended_container(false, false);
    let lti_bytes = build_v2_extended_container(true, false);
    let muted_s4_bytes = build_v2_extended_container(false, true);

    exercise_backend_full_matrix::<ScalarBackend>(&wh_bytes);
    exercise_backend_full_matrix::<DefaultMathBackend>(&wh_bytes);
    exercise_backend_full_matrix::<ScalarBackend>(&lti_bytes);
    exercise_backend_full_matrix::<DefaultMathBackend>(&muted_s4_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        exercise_backend_full_matrix::<ArmSve2Backend>(&wh_bytes);
        exercise_backend_full_matrix::<ArmSmeBackend>(&lti_bytes);
    }

    let mut dyn_eng = DynamicDnamEngine::from_bytes(&wh_bytes, Tier::Ultra, 48_000).unwrap();
    let in_l = [0.25_f32; 48];
    let in_r = [-0.25_f32; 48];
    let mut out_l = [0.0_f32; 48];
    let mut out_r = [0.0_f32; 48];
    dyn_eng.configure_host_mix_and_leveling(0.65, 0.5, true);
    dyn_eng.sync_tempo(120.0, None, false);
    let _ = dyn_eng.register_tap_tempo(Some(TempoSubdivision::Quarter), true);
    dyn_eng.apply_preset("tight_lead").unwrap();
    dyn_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    assert!(out_l.iter().all(|v| v.is_finite()) && out_r.iter().all(|v| v.is_finite()));
}

#[test]
fn test_v2_multiport_4cm_wdw_mono_reload_and_constructors() {
    let wh_bytes = build_v2_extended_container(false, false);
    let lti_bytes = build_v2_extended_container(true, false);

    let mut scratch = vec![AlignedFp16Block::default(); 64];
    let eng_scratch = DnamEngine::<ScalarBackend>::from_bytes_with_scratch(
        &wh_bytes,
        &mut scratch,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_scratch.effective_tier(), Tier::Full);

    let mut eng_max_t = DnamEngine::<DefaultMathBackend>::with_max_tier(Tier::Pro);
    assert_eq!(eng_max_t.effective_tier(), Tier::Lite);
    eng_max_t
        .reload_model_in_place_owned(&wh_bytes, Tier::Ultra, 48_000)
        .unwrap();
    assert_eq!(eng_max_t.effective_tier(), Tier::Pro);
    eng_max_t.set_max_supported_tier(Tier::Full);
    eng_max_t.set_tier(Tier::Ultra);
    assert_eq!(eng_max_t.effective_tier(), Tier::Full);

    let eng_cap = DnamEngine::<ScalarBackend>::with_max_capacity();
    assert_eq!(eng_cap.effective_tier(), Tier::Lite);

    let mut eng_owned_lt =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&wh_bytes, Tier::Pro, 48_000).unwrap();
    let mut eng_max_ref = DnamEngine::<DefaultMathBackend>::from_bytes_with_max_tier(
        &wh_bytes,
        Tier::Ultra,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_max_ref.effective_tier(), Tier::Full);

    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned_with_max_tier(
        &wh_bytes,
        Tier::Pro,
        Tier::Ultra,
        48_000,
    )
    .unwrap();

    let n = 48usize;
    let inst_l = vec![0.3_f32; n];
    let inst_r = vec![-0.2_f32; n];
    let ret_l = vec![0.1_f32; n];
    let ret_r = vec![-0.1_f32; n];
    let mut main_l = vec![0.0_f32; n];
    let mut main_r = vec![0.0_f32; n];
    let mut send_l = vec![0.0_f32; n];
    let mut send_r = vec![0.0_f32; n];
    let mut dry_center = vec![0.0_f32; n];

    eng.set_stereo_width(0.0);
    eng.process_block_4cm(
        &inst_l,
        &inst_r,
        &ret_l,
        &ret_r,
        &mut main_l,
        &mut main_r,
        &mut send_l,
        &mut send_r,
    );
    assert!(main_l.iter().all(|v| v.is_finite()));

    eng.set_stereo_width(1.5);
    eng.stage0_router.wdw_tap_mode = WdwDryTapMode::PreSpatialCore;
    eng.process_block_wdw(&inst_l, &inst_r, &mut main_l, &mut main_r, &mut dry_center);
    assert!(dry_center.iter().all(|v| v.is_finite()));

    eng.stage0_router.wdw_tap_mode = WdwDryTapMode::DirectDry;
    eng.process_block_wdw(&inst_l, &inst_r, &mut main_l, &mut main_r, &mut dry_center);
    assert!(main_r.iter().all(|v| v.is_finite()));

    eng.process_block_sidechain(&inst_l, &inst_r, &ret_l, &ret_r, &mut main_l, &mut main_r);

    // Exercise process_block_io in both mono-only and 4x offline bounce modes
    eng.set_maximize_tier_on_bounce(true);
    eng.set_offline_bounce(true);
    eng.process_block_io(DnamAudioIo {
        in_1: &inst_l,
        in_2: None,
        in_3: None,
        in_4: None,
        out_1: &mut main_l,
        out_2: None,
        out_3: None,
        out_4: None,
    });
    assert!(main_l.iter().all(|v| v.is_finite()));
    eng.set_offline_bounce(false);
    eng.process_block_io(DnamAudioIo {
        in_1: &inst_l,
        in_2: None,
        in_3: Some(&ret_l),
        in_4: Some(&ret_r),
        out_1: &mut main_l,
        out_2: None,
        out_3: None,
        out_4: None,
    });

    eng_max_ref
        .reload_model_in_place(&lti_bytes, Tier::Ultra, 44_100)
        .unwrap();
    eng_max_ref.process_block(&inst_l, &inst_r, &mut main_l, &mut main_r);
    eng_owned_lt
        .reload_model_in_place_owned(&lti_bytes, Tier::Full, 96_000)
        .unwrap();
    eng_owned_lt.process_block(&inst_l, &inst_r, &mut main_l, &mut main_r);

    // Explicit CP1 container builder path
    let mut hdr = build_v2_header();
    let arena = populate_v2_arena(false, false);
    let slot = Cp1ContributorSlot {
        lanes: 2,
        ctrl_kind: 0,
        control_idx: 0,
        option_idx: 0xFF,
        partner_control_id: 0,
        main_base: [[0.1; 8]; 3],
        emb: [0.5, 0.25, 0.0],
        v_base: [0.1, -0.05, 0.02, 0.01],
        main_ultra: [[0.04; 8]; 3],
        v_ultra: [0.02; 4],
        has_ultra: true,
    };
    let cp1_bytes =
        build_dnam_container_with_cp1(&mut hdr, &arena, &arena, false, &[slot]).unwrap();
    let mut eng_cp1 =
        DnamEngine::<ScalarBackend>::from_bytes(&cp1_bytes, Tier::Ultra, 48_000).unwrap();
    eng_cp1.process_block(&inst_l, &inst_r, &mut main_l, &mut main_r);
    assert!(main_l.iter().all(|v| v.is_finite()));
}

#[test]
fn test_direct_stage1_stage2_conv6_and_stage4_subpaths() {
    let _ftz = FtzDazGuard::new();
    let arena = populate_v2_arena(false, false);

    // Stage 1 direct extended & mono topology branches
    let mut s1 = Stage1PreDynamics::new();
    s1.reserve_extended_capacity(512);
    s1.load_from_arena(&arena);
    let stage_params = [0.5_f32; 16];
    let (a_l, a_r) = s1.step_stereo_ext(0.3, -0.25, Some((0.2, -0.1)), &stage_params);
    assert!(a_l.is_finite() && a_r.is_finite());
    s1.ringmod_mix = 1.4;
    let (b_l, b_r) = s1.step_stereo_ext(0.4, 0.1, None, &[0.0_f32; 16]);
    assert!(b_l.is_finite() && b_r.is_finite());
    s1.ringmod_mix = 0.0;
    s1.chorus_wet_mix = 0.0;
    s1.rotary_am_depth = 0.0;
    let _ = s1.step_stereo_ext(0.25, -0.2, None, &[0.0_f32; 16]);
    for mode in 0..=3 {
        s1.topology_mode = mode;
        assert!(s1.step(0.2).is_finite());
    }
    s1.step_delta_branch(0.2, -0.2);
    s1.retune_op_m_pre_3khz(2.5, 0.7, true);
    s1.retune_op_m_pre_3khz(-2.5, 0.2, true);
    s1.reset();

    // Stage II v3 Tone SVF & Matryoshka V3HotCore execution paths
    let mut tone_svf = V3InterBlockToneSvfState::from_cold(&V3InterBlockToneSvfParams::identity());
    let [tl, tr] = tone_svf.step_stereo(0.2, -0.2);
    assert!(tl.is_finite() && tr.is_finite());

    for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let cold = V3ColdTierModel::identity_for_tier(tier, tier as u8 >= 2, false);
        let mut core_s = V3HotCore::<ScalarBackend>::from_cold_model(&cold);
        let mut core_d = V3HotCore::<DefaultMathBackend>::from_cold_model(&cold);
        let [yl, yr] = core_s.step_stereo(0.25, -0.25);
        let [zl, zr] = core_d.step_stereo(0.25, -0.25);
        assert!(
            yl.is_finite() && yr.is_finite() && (yl - zl).abs() < 1e-6 && (yr - zr).abs() < 1e-6
        );
    }

    // Stage 4 direct knee / coercive / multi-tier tap paths
    let mut s4 = Stage4SpatialTank::new();
    s4.reserve_extended_capacity(2_048, 8_192);
    s4.load_from_arena(&arena);
    s4.sigma_knee = 0.72;
    let knee_v = s4.eval_quiet_feedback_knee(0.4, 0.72, 0.35);
    let sat_v = s4.coercive_saturate(0.45);
    assert!(knee_v.is_finite() && sat_v.is_finite());
    let mut fb = [0.0_f32; 256];
    let (_wl, _wr, read_s, loop_s) = s4.step_stereo_with_readouts(0.25, -0.25);
    for max_tier in 1..=4 {
        s4.read_multi_tier_taps_into(max_tier, read_s, loop_s, &mut fb);
    }
    assert!(fb[246].is_finite());
}
