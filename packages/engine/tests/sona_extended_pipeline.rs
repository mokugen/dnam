#![forbid(unsafe_code)]
//! Integration coverage for `SONA` (`v3.0`) extended DSP pipelines, multi-port topologies,
//! `DynamicDnamEngine`, `CP1` projections, `CrossbarHot`, `SonaHotSlab`, and `SpatialTankHot` paths.

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use arena_builder::{build_sona_extended_container, build_sona_header, populate_sona_arena};
use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::graph::router::{CrossbarHot, CrossbarParams};
use dnam_engine::graph::topology::{eval_tank_blended_feedback_knee, eval_tank_loop_saturator};
use dnam_engine::graph::{
    SonaColdTierModel, SonaHotSlab, SonaInstance, SpatialTankHot, SpatialTankParams, WdwDryTapMode,
};
#[cfg(target_arch = "aarch64")]
use dnam_engine::math::{ArmSmeBackend, ArmSve2Backend};
use dnam_engine::math::{DefaultMathBackend, ScalarBackend};
use dnam_engine::memory::AlignedFp16Block;
use dnam_engine::schema::TempoSubdivision;
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
fn test_sona_nonlinear_and_linear_across_tiers_sample_rates_and_all_backends() {
    let nl_bytes = build_sona_extended_container(false, false);
    let lin_bytes = build_sona_extended_container(true, false);
    let muted_s4_bytes = build_sona_extended_container(false, true);

    exercise_backend_full_matrix::<ScalarBackend>(&nl_bytes);
    exercise_backend_full_matrix::<DefaultMathBackend>(&nl_bytes);
    exercise_backend_full_matrix::<ScalarBackend>(&lin_bytes);
    exercise_backend_full_matrix::<DefaultMathBackend>(&muted_s4_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        exercise_backend_full_matrix::<ArmSve2Backend>(&nl_bytes);
        exercise_backend_full_matrix::<ArmSmeBackend>(&lin_bytes);
    }

    let mut dyn_eng = DynamicDnamEngine::from_bytes(&nl_bytes, Tier::Ultra, 48_000).unwrap();
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
fn test_sona_multiport_4cm_wdw_mono_reload_and_constructors() {
    let nl_bytes = build_sona_extended_container(false, false);
    let lin_bytes = build_sona_extended_container(true, false);

    let mut scratch = vec![AlignedFp16Block::default(); 64];
    let eng_scratch = DnamEngine::<ScalarBackend>::from_bytes_with_scratch(
        &nl_bytes,
        &mut scratch,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_scratch.effective_tier(), Tier::Full);

    let mut eng_max_t = DnamEngine::<DefaultMathBackend>::with_max_tier(Tier::Pro);
    assert_eq!(eng_max_t.effective_tier(), Tier::Lite);
    eng_max_t
        .reload_model_in_place_owned(&nl_bytes, Tier::Ultra, 48_000)
        .unwrap();
    assert_eq!(eng_max_t.effective_tier(), Tier::Pro);
    eng_max_t.set_max_supported_tier(Tier::Full);
    eng_max_t.set_tier(Tier::Ultra);
    assert_eq!(eng_max_t.effective_tier(), Tier::Full);

    let eng_cap = DnamEngine::<ScalarBackend>::with_max_capacity();
    assert_eq!(eng_cap.effective_tier(), Tier::Lite);

    let mut eng_owned_lt =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&nl_bytes, Tier::Pro, 48_000).unwrap();
    let mut eng_max_ref = DnamEngine::<DefaultMathBackend>::from_bytes_with_max_tier(
        &nl_bytes,
        Tier::Ultra,
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_max_ref.effective_tier(), Tier::Full);

    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned_with_max_tier(
        &nl_bytes,
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
    eng.io_router.wdw_tap_mode = WdwDryTapMode::PreSpatialCore;
    eng.process_block_wdw(&inst_l, &inst_r, &mut main_l, &mut main_r, &mut dry_center);
    assert!(dry_center.iter().all(|v| v.is_finite()));

    eng.io_router.wdw_tap_mode = WdwDryTapMode::DirectDry;
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
        .reload_model_in_place(&lin_bytes, Tier::Ultra, 44_100)
        .unwrap();
    eng_max_ref.process_block(&inst_l, &inst_r, &mut main_l, &mut main_r);
    eng_owned_lt
        .reload_model_in_place_owned(&lin_bytes, Tier::Full, 96_000)
        .unwrap();
    eng_owned_lt.process_block(&inst_l, &inst_r, &mut main_l, &mut main_r);

    // Explicit CP1 container builder path
    let mut hdr = build_sona_header();
    let arena = populate_sona_arena(false, false);
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
fn test_direct_crossbar_sona_hot_slab_and_tank_subpaths() {
    let _ftz = FtzDazGuard::new();

    // Unconditional CrossbarHot path (gain + mix + hysteresis gate + Bezout crossover)
    let mut cb_params = CrossbarParams::identity();
    cb_params.xo_hz = 180.0;
    cb_params.xo_low_gain = 1.05;
    cb_params.xo_high_gain = 0.95;
    cb_params.clean_sub_mix = 0.25;
    cb_params.gate_thresh = 0.002;
    cb_params.gate_hyst = 0.001;
    cb_params.gate_inv_knee = 100.0;
    cb_params.gate_floor = 0.1;
    let mut cb = CrossbarHot::from_cold(&cb_params);
    assert_eq!(cb.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
    let y0 = cb.step_stereo(0.3, -0.25);
    assert!(y0.iter().all(|v| v.is_finite()));
    cb.reconfigure_sample_rate(&cb_params, 96_000.0);
    let y0_96k = cb.step_stereo(0.25, -0.2);
    assert!(y0_96k.iter().all(|v| v.is_finite()));
    cb.reset();

    // Unified SONA Matryoshka SonaHotSlab & SonaInstance execution parity across all 4 tiers
    for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let cold = SonaColdTierModel::identity_for_tier(tier, tier as u8 >= 1, false);
        let mut slab_s = SonaHotSlab::<ScalarBackend>::from_cold_model(&cold);
        let mut slab_d = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        let [yl, yr] = slab_s.step_stereo(0.25, -0.25);
        let [zl, zr] = slab_d.step_stereo(0.25, -0.25);
        assert!(
            yl.is_finite() && yr.is_finite() && (yl - zl).abs() < 1e-6 && (yr - zr).abs() < 1e-6
        );

        let mut inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&cold);
        let mut ol = [0.0_f32; 16];
        let mut or = [0.0_f32; 16];
        inst.process_block_stereo(&[0.25; 16], &[-0.25; 16], &mut ol, &mut or);
        assert!(ol.iter().all(|v| v.is_finite()) && or.iter().all(|v| v.is_finite()));
    }

    // SONA SpatialTankHot multi-quad & topology knee / loop-saturator paths
    let knee_v = eval_tank_blended_feedback_knee(0.35, 0.72, 0.25, 0.50).expect("valid knee input");
    let sat_v = eval_tank_loop_saturator(knee_v, 0.95).expect("valid saturator input");
    assert!(knee_v.is_finite() && sat_v.is_finite());

    let mut tp = SpatialTankParams::identity();
    tp.g_loop = 0.72;
    tp.wet_mix = 0.30;
    let mut tank = SpatialTankHot::from_cold(&tp);
    let [wl, wr] = tank.step_stereo::<DefaultMathBackend>(0.25, -0.25);
    assert!(wl.is_finite() && wr.is_finite());
    tank.freeze = true;
    let [fl, fr] = tank.step_stereo::<DefaultMathBackend>(0.0, 0.0);
    assert!(fl.is_finite() && fr.is_finite());
    tank.freeze = false;
    tank.reset();
}
