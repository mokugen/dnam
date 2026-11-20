//! End-to-End `SONA` (`v3.0`) Tier Equivalence (`r[dsp.head.tier-equivalence]`), Multiport WDW/4CM,
//! Reload Containment, and Property Verification Suite (`tests/sona_tier_and_multiport_lifecycle.rs`).
//!
//! Canonical Reference: `docs/specs/architecture/sona_architecture.md` and
//! `docs/specs/dsp/router_and_crossbar_topology.md`.

#![forbid(unsafe_code)]

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use arena_builder::build_populated_sona_container;
use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, build_dnam_container, build_v3_open_dnam_container,
    strip_container_to_tier,
};
use dnam_engine::graph::WdwDryTapMode;
use dnam_engine::graph::sona_core::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_TANK_LINES, MAX_TOTAL_SONA_LANES,
    SONA_FIR_MAX_TAPS, SonaControlBusParams, SonaCrossbarConfig, SonaMatryoshkaModel,
};
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::math::ScalarBackend;
use dnam_engine::schema::{DnamError, DnamJsonHeader};
use dnam_engine::{DnamEngine, DynamicDnamEngine, OFFLINE_BOUNCE_PDC_SAMPLES, Tier};
use proptest::prelude::*;
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::F32;

#[test]
fn test_exact_tier_equivalence_unstripped_vs_stripped_with_arena0_and_crossbars() {
    let mut header = arena_builder::build_sona_header();
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_hz = 180.0;
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;

    for grp in &mut model.groups[..MAX_SONA_GROUPS] {
        for l in 0..4 {
            grp.svf_g[l] = 0.16 + 0.05 * (l as f32);
            grp.svf_k[l] = core::f32::consts::FRAC_1_SQRT_2;
            grp.in_x_weight[l] = if l == 0 { 0.72 } else { 0.08 };
            grp.tone_lp[l] = 0.25;
            grp.tone_bp[l] = 0.12;
            grp.tone_dir[l] = 0.65;
            grp.rect_sign[l] = 1.0;
            grp.adaa_drive[l] = 0.70;
            grp.lru_kappa[l] = 0.35;
            grp.lru_alpha_atk[l] = 0.06;
            grp.lru_alpha_rel[l] = 0.88;
            grp.lru_bias_drive[l] = 0.05;
        }
    }
    for tile in &mut model.tiles[..MAX_SONA_TILES] {
        for r in 0..4 {
            tile.w_odd_4x4[r][r] = 0.14;
            tile.w_even_4x4[r][r] = 0.03;
            tile.w_lru_4x4[r][r] = 0.05;
        }
    }
    for (m, pole) in model.kautz_poles[..MAX_KAUTZ_POLES].iter_mut().enumerate() {
        *pole = (
            0.91 + 0.002 * ((m & 3) as f32),
            0.15 + 0.04 * ((m % 8) as f32),
        );
    }
    for (t, ro) in model.readouts.iter_mut().enumerate() {
        for ch in 0..2 {
            for l in 0..MAX_TOTAL_SONA_LANES {
                ro.pre_cab_w_odd[ch][l] = 0.045 * (((l + ch + t) % 5) as f32 - 1.5);
                ro.pre_cab_w_even[ch][l] = 0.015;
                ro.pre_cab_w_lru[ch][l] = 0.012;
                ro.direct_w_odd[ch][l] = 0.028;
            }
        }
        for tap in 0..SONA_FIR_MAX_TAPS {
            ro.fir_l[tap] = 0.025 * (((tap + t) % 5) as f32 - 2.0);
            ro.fir_r[tap] = 0.02 * (((tap + t + 1) % 5) as f32 - 2.0);
        }
        for m in 0..MAX_KAUTZ_POLES {
            ro.kautz_b_gamma[m] = 0.32 + 0.02 * (t as f32);
            ro.kautz_c_l[m] = (0.14, -0.07);
            ro.kautz_c_r[m] = (0.11, -0.05);
        }
        model.stage4[t].g_loop = 0.72;
        model.stage4[t].wet_mix = 0.25;
    }

    let cbs: [SonaCrossbarConfig; 4] = core::array::from_fn(|t| {
        let tf = (t + 1) as f32;
        let mut cb = SonaCrossbarConfig::identity();
        cb.phasor_freq_hz = [0.45 * tf, 1.1 * tf, 0.2 * tf, 0.0];
        cb.phasor_init_phase_q32 = [0x1000_0000 * tf as u32, 0x2000_0000, 0, 0];
        if t > 0 {
            cb.phasor_stereo_phase_rad = [0.25 * tf, 0.5 * tf, 0.1 * tf, 0.0];
        }
        cb.op_p_crossbar_gain = 0.12 + 0.04 * tf;
        cb.op_t_crossbar_gain = 0.08 + 0.03 * tf;
        for g in 0..MAX_SONA_GROUPS {
            cb.op_p_to_group[g][0] = 0.06 * tf;
            cb.phasor_to_group[g][0] = 0.04 * tf;
        }
        cb.opt_to_lr = [[0.05 * tf, -0.04 * tf]; MAX_TANK_LINES];
        cb
    });
    model.tier_crossbars = Some(cbs);

    // Populate ControlBus and give Arena 0 [4][4] DISTINCT weights from [8][10].
    let mut ctrl = SonaControlBusParams::identity();
    ctrl.cp1_dir_count = 3;
    ctrl.cp1_dir_entries[0] = [0, 2, 0];
    ctrl.cp1_dir_entries[1] = [1, (1 << 4) | 1, 1];
    ctrl.cp1_dir_entries[2] = [0, (2 << 4) | 1, 2];
    ctrl.u_int_base = [[0.05; 8]; 4];
    let (cp1_f32, _) = <[F32]>::mut_from_prefix(&mut ctrl.cp1_base).unwrap();
    cp1_f32[..80].fill(F32::new(0.08));
    ctrl.w_ctrl = [[0.04; 16]; 16];
    // Populate Arena 3 (`Tier::Ultra` only) weights & couplings to prove non-Ultra tiers produce
    // 0-ULP bitwise-identical output across unstripped vs stripped containers (`r[dsp.head.tier-equivalence]`).
    ctrl.w_ctrl_ultra = [[0.18; 4]; 16];
    ctrl.u_int_ultra[0] = [[0.32; 8]; 4];
    ctrl.u_int_ultra[1] = [[-0.24; 8]; 4];
    ctrl.u_int_ultra[2] = [[0.19; 8]; 4];
    let (cp1_u_f32, _) = <[F32]>::mut_from_prefix(&mut ctrl.cp1_ultra).unwrap();
    cp1_u_f32[..96].fill(F32::new(0.22));
    ctrl.core_mods_present = true;
    ctrl.stage_latent_core_mods_fp16[4 * 128] = dnam_kernels::f32_to_fp16(0.42);
    ctrl.w_svf_g_oct[0] = 0.35;
    ctrl.w_svf_k_delta[0] = 0.15;
    ctrl.w_kautz_oct[0] = 0.12;
    ctrl.w_kautz_damp[0] = 0.10;
    ctrl.w_tank_psi[0] = 0.20;
    ctrl.lite_block_scales_present = true;
    for k in 0..4 {
        for b in 0..4 {
            ctrl.w_adaa_drive_scale_lite[k][b] = 0.45;
            ctrl.w_adaa_drive_scale[k][b] = -0.35;
            ctrl.w_lru_bias_shift_lite[k][b] = 0.28;
            ctrl.w_lru_bias_shift[k][b] = -0.22;
        }
    }
    model.ctrl = ctrl;

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    pack_sona_matryoshka_to_arena(&model, &mut arena);
    let ultra_container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

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

    for tier in [Tier::Lite, Tier::Full, Tier::Pro] {
        let stripped = strip_container_to_tier(&ultra_container, tier).unwrap();
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
        let st_bits_l: Vec<u32> = l_st.iter().map(|v| v.to_bits()).collect();
        let st_bits_r: Vec<u32> = r_st.iter().map(|v| v.to_bits()).collect();
        let ub_bits_l: Vec<u32> = l_ub.iter().map(|v| v.to_bits()).collect();
        let ub_bits_r: Vec<u32> = r_ub.iter().map(|v| v.to_bits()).collect();
        let us_bits_l: Vec<u32> = l_us.iter().map(|v| v.to_bits()).collect();
        let us_bits_r: Vec<u32> = r_us.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            ub_bits_l, st_bits_l,
            "boot 0-ULP equivalence L failed on {tier:?}"
        );
        assert_eq!(
            ub_bits_r, st_bits_r,
            "boot 0-ULP equivalence R failed on {tier:?}"
        );
        assert_eq!(
            us_bits_l, st_bits_l,
            "set_tier 0-ULP equivalence L failed on {tier:?}"
        );
        assert_eq!(
            us_bits_r, st_bits_r,
            "set_tier 0-ULP equivalence R failed on {tier:?}"
        );
        assert_ne!(
            l_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l_ultra_only.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "Arena 3 Ultra delta should produce distinct Ultra output relative to {tier:?}"
        );

        // Slewed control updates (plus dynamic set_tier without reset) must also stay 0-ULP equivalent.
        for eng in [&mut eng_st, &mut eng_ub, &mut eng_us] {
            eng.set_control_value(0, 3.5);
            eng.set_control_value(2, 180.0);
        }
        let (sl_st, sr_st) = run_pair(&mut eng_st);
        let (sl_ub, sr_ub) = run_pair(&mut eng_ub);
        let (sl_us, sr_us) = run_pair(&mut eng_us);
        assert_eq!(
            sl_ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sl_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "slewed boot 0-ULP equivalence L failed on {tier:?}"
        );
        assert_eq!(
            sr_ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sr_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "slewed boot 0-ULP equivalence R failed on {tier:?}"
        );
        assert_eq!(
            sl_us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sl_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "slewed set_tier 0-ULP equivalence L failed on {tier:?}"
        );
        assert_eq!(
            sr_us.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            sr_st.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "slewed set_tier 0-ULP equivalence R failed on {tier:?}"
        );
    }
}

#[test]
fn test_reload_model_in_place_owned_rejects_preview_and_corrupt_json() {
    let (_hdr, _arena, valid) = build_populated_sona_container();
    let mut prev_hdr = arena_builder::build_sona_header();
    prev_hdr.container_kind = "preview".to_string();
    let prev_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let prev = build_dnam_container(&mut prev_hdr, &prev_arena, &prev_arena, false).unwrap();

    let t0 = vec![0u8; 4_608];
    let bad_json =
        build_v3_open_dnam_container(b"{not valid json", &[0u8; 64], &[0u8; 64], &[&t0], None, 0)
            .unwrap();

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
}

#[test]
fn test_wdw_single_width_application_and_4x_4cm_send_downsampling() {
    let (_hdr, _arena, container) = build_populated_sona_container();
    let (in_l, in_r) = ([0.65_f32; 32], [-0.25_f32; 32]);

    for tap_mode in [WdwDryTapMode::PreSpatialCore, WdwDryTapMode::DirectDry] {
        let mut eng_w1 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_w2 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng_w1.io_router.wdw_tap_mode = tap_mode;
        eng_w2.io_router.wdw_tap_mode = tap_mode;
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
            assert_eq!(d1[i], d2[i], "dry_c leaked stereo_width at sample {i}");
        }
    }

    let run_4cm = |offline: bool, il: &[f32; 64], ir: &[f32; 64], ret: &[f32; 64]| {
        let mut eng =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng.set_offline_bounce(offline);
        let (mut ml, mut mr, mut sl, mut sr) =
            ([0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64]);
        eng.process_block_4cm(il, ir, ret, ret, &mut ml, &mut mr, &mut sl, &mut sr);
        (sl, sr)
    };
    let (step_l, step_r, ret_zero, ret_sig) =
        ([0.4_f32; 64], [-0.4_f32; 64], [0.0_f32; 64], [0.3_f32; 64]);
    let (send_la, send_ra) = run_4cm(true, &step_l, &step_r, &ret_zero);
    let (send_lb, send_rb) = run_4cm(true, &step_l, &step_r, &ret_sig);
    let (send_1x_l, send_1x_r) = run_4cm(false, &step_l, &step_r, &ret_zero);

    assert_eq!(send_la, send_lb);
    assert_eq!(send_ra, send_rb);
    assert!(send_la.iter().all(|v| v.is_finite()));
    assert!(send_la[0].abs() < send_la[32].abs());
    let ref_idx = 32 - OFFLINE_BOUNCE_PDC_SAMPLES;
    assert!((send_la[32] - send_1x_l[ref_idx]).abs() < 0.02);
    assert!((send_ra[32] - send_1x_r[ref_idx]).abs() < 0.02);

    let nyq: [f32; 64] = core::array::from_fn(|i| if (i & 1) == 0 { 0.5 } else { -0.5 });
    let (sn_4x_l, _sn_4x_r) = run_4cm(true, &nyq, &nyq, &ret_zero);
    let (sn_1x_l, _sn_1x_r) = run_4cm(false, &nyq, &nyq, &ret_zero);
    let settled_4x_nyq = sn_4x_l[32..]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let settled_1x_nyq = sn_1x_l[32..]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    // Half-band upsampler + downsampler each contribute -6 dB at fs/2 (`~0.25x` of 1x Nyquist, vs `0.5x` without downsampling).
    assert!(
        settled_4x_nyq < settled_1x_nyq * 0.55,
        "4x Nyquist send peak {settled_4x_nyq} not attenuated relative to 1x peak {settled_1x_nyq}"
    );
    let mut os_stop = dnam_engine::engine::Oversampler4xState::new();
    let mut settled_stop = 0.0_f32;
    for i in 0..32 {
        let [dl, dr] = os_stop.downsample_4x([[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]]);
        if i >= 16 {
            settled_stop = settled_stop.max(dl.abs()).max(dr.abs());
        }
    }
    assert!(
        settled_stop < 1e-4,
        "4x Nyquist alias stopband peak {settled_stop} too high"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// verify[arch.pillars.dual-rate-plane]
    /// verify[arch.tiers.arena-hierarchy]
    /// verify[arch.fp.range-saturation]
    #[test]
    fn prop_end_to_end_sona_engine_tiers_rates_and_finite_outputs(
        tier_u8 in 0u8..=3u8,
        rate_idx in 0usize..4usize,
        in_l in prop::array::uniform16(-5.0_f32..5.0_f32),
        in_r in prop::array::uniform16(-5.0_f32..5.0_f32),
    ) {
        let rates = [44_100u32, 48_000, 88_200, 96_000];
        let sr = rates[rate_idx];
        let tier = Tier::from_u8(tier_u8);
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
        ).unwrap();
        let mut eng = DnamEngine::<ScalarBackend>::from_bytes(&v3_container, tier, sr).unwrap();
        let mut out_l = [0.0f32; 16];
        let mut out_r = [0.0f32; 16];
        eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        prop_assert!(out_l.iter().all(|v| v.is_finite() && v.abs() <= 1.0e9));
        prop_assert!(out_r.iter().all(|v| v.is_finite() && v.abs() <= 1.0e9));
    }
}
