//! Dedicated Milestone M10b integration & operator parity tests for `dnam-vm`:
//! pure `2x2` channel routing (`1->1`, `2->1`, `1->2`, `2->2`), `SensoryQuad` VCA (`slot 30`),
//! smooth `Op-N[Pre]` `zoh_hold_depth` & ZOH invariance, `ro_pole_48k` ZOH invariance, `CoreQuad`
//! M/S coupling + Selective-Duhem + stereo-link + `GLU`, `T1SubHarmonicKernel` 2-stage rotor,
//! `Op-T` `sigma_sat` / `freeze_mix` / stage `tank_dry_mix` / immediate write-lock, and tier monotonicity.

#![forbid(unsafe_code)]

use dnam_kernels::{
    MAX_PARALRU_R_F32, OpNPreHoisted, OpNPreHotState, QUAD_FLAG_HOST_WRITE_LOCK,
    T1SubHarmonicKernel, TANK_MAX_G_LIN_F32,
};
use dnam_vm::bin::{
    DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, DnamBinView, SONA_META_SENSORY_QUAD,
    SONA_META_TRANSPORT_TAP_GAINS,
};
use dnam_vm::control::bus::laws::eval_op_t_freeze_mix_eff;
use dnam_vm::math::{exp2_signed_oct_f32, powf_f64};
use dnam_vm::{
    CextDispatchWire, ControlClampDescriptor, CoreQuadWire, DefaultMathBackend, DnamBinBuildInput,
    SONA_FULL_HOT_DELTA_BYTES, SONA_LITE_HOT_BYTES, SONA_OS2X_HOT_BYTES, SONA_PRO_HOT_DELTA_BYTES,
    SONA_ULTRA_HOT_DELTA_BYTES, SonaHotSlab, SonaVm, Tier, encode_dnam_bin,
};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromZeros, IntoBytes};

use super::helpers::{
    assert_finite_non_silent, build_full_pipeline_ultra_fixture, encode_lite_fixture, fp16_arr,
    fp16_val, le_f32, make_core_quad, make_transport_quad, stereo_test_signal,
};

#[test]
fn test_m10b_routing_modes_1x1_2x1_1x2_2x2_and_step_vs_block_0ulp_parity() {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.sona_meta_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0.crossbar.cell_desc.cell_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0.core_quads[0] = make_core_quad(0);
    a0.core_quads[0].in_mid_weight_fp16 = fp16_arr([0.6, 0.3, -0.2, 0.1]);
    a0.core_quads[0].in_side_weight_fp16 = fp16_arr([0.4, -0.3, 0.25, -0.15]);
    a0.conv_fir_lite.fir_l[0] = F32::new(0.9);
    a0.conv_fir_lite.fir_r[0] = F32::new(0.7);
    a0.linear_readout_lite.direct_s2_u_gain = le_f32([0.55, 0.35]);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);
    a0.transport_lite = make_transport_quad(0);

    let bin_1to1 = encode_lite_fixture(&a0, DNBN_FLAG_SINGLE_INPUT, 0.99, &[], None);
    let bin_2to1 = encode_lite_fixture(&a0, 0, 0.99, &[], None);
    let bin_1to2 = encode_lite_fixture(
        &a0,
        DNBN_FLAG_SINGLE_INPUT | DNBN_FLAG_STEREO_OUT,
        0.99,
        &[],
        None,
    );
    let bin_2to2 = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);

    let (in_l, in_r) = stereo_test_signal(96, 21);
    let neg_in_l: Vec<f32> = in_l.iter().map(|&v| -v).collect();

    for (mode_name, bin, mono_in, mono_out) in [
        ("1->1", &bin_1to1, true, true),
        ("2->1", &bin_2to1, false, true),
        ("1->2", &bin_1to2, true, false),
        ("2->2", &bin_2to2, false, false),
    ] {
        let mut vm_blk =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Lite, 48_000.0).unwrap();
        let mut vm_stp =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Lite, 48_000.0).unwrap();
        assert_eq!(vm_blk.slab().single_input(), mono_in, "{mode_name}");
        assert_eq!(vm_blk.slab().mono_out(), mono_out, "{mode_name}");

        let (mut blk_l, mut blk_r) = ([0.0_f32; 96], [0.0_f32; 96]);
        vm_blk
            .process_block(&in_l, &in_r, &mut blk_l, &mut blk_r)
            .unwrap();
        assert_finite_non_silent(&blk_l, &blk_r, mode_name);

        let mut stereo_diff = 0.0_f32;
        for i in 0..96 {
            let (sl, sr) = vm_stp.step_frame(in_l[i], in_r[i]).unwrap();
            assert_eq!(sl.to_bits(), blk_l[i].to_bits(), "{mode_name} L@{i}");
            assert_eq!(sr.to_bits(), blk_r[i].to_bits(), "{mode_name} R@{i}");
            if mono_out {
                assert_eq!(
                    blk_l[i].to_bits(),
                    blk_r[i].to_bits(),
                    "{mode_name} mono@{i}"
                );
            } else {
                stereo_diff = stereo_diff.max((blk_l[i] - blk_r[i]).abs());
            }
        }
        if !mono_out {
            assert!(stereo_diff > 1.0e-4, "{mode_name} expected stereo image");
        }

        // Compare against (in_l, in_l): single-input ignores in_r bit-for-bit across step/block/4x;
        // dual-input must respond distinctly to in_r = -in_l.
        let mut vm_blk_dup =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Lite, 48_000.0).unwrap();
        let mut vm_stp_neg =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Lite, 48_000.0).unwrap();
        let (mut dup_l, mut dup_r) = ([0.0_f32; 96], [0.0_f32; 96]);
        vm_blk_dup
            .process_block(&in_l, &in_l, &mut dup_l, &mut dup_r)
            .unwrap();

        let mut in_r_sensitivity = 0.0_f32;
        for i in 0..96 {
            let (nl, nr) = vm_stp_neg.step_frame(in_l[i], neg_in_l[i]).unwrap();
            if mono_in {
                assert_eq!(
                    blk_l[i].to_bits(),
                    dup_l[i].to_bits(),
                    "{mode_name} blk L@{i}"
                );
                assert_eq!(
                    blk_r[i].to_bits(),
                    dup_r[i].to_bits(),
                    "{mode_name} blk R@{i}"
                );
                assert_eq!(nl.to_bits(), blk_l[i].to_bits(), "{mode_name} neg L@{i}");
                assert_eq!(nr.to_bits(), blk_r[i].to_bits(), "{mode_name} neg R@{i}");
            } else {
                in_r_sensitivity = in_r_sensitivity.max((blk_l[i] - nl).abs());
            }
        }
        if mono_in {
            let mut vm_4x_ref =
                SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Lite, 48_000.0)
                    .unwrap();
            let mut vm_4x_neg =
                SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Lite, 48_000.0)
                    .unwrap();
            vm_4x_ref.set_offline_bounce(true).unwrap();
            vm_4x_neg.set_offline_bounce(true).unwrap();
            for i in 0..32 {
                let (rl, rr) = vm_4x_ref.step_frame(in_l[i], in_l[i]).unwrap();
                let (nl, nr) = vm_4x_neg.step_frame(in_l[i], neg_in_l[i]).unwrap();
                assert_eq!(nl.to_bits(), rl.to_bits(), "{mode_name} 4x L@{i}");
                assert_eq!(nr.to_bits(), rr.to_bits(), "{mode_name} 4x R@{i}");
            }
        } else {
            assert!(
                in_r_sensitivity > 1.0e-4,
                "{mode_name} dual-input must depend on R input"
            );
        }
    }
}

#[test]
fn test_m10b_sensory_quad_vca_slot30_and_control_bus_envelope_loop() {
    let mut a0_pass = DnamBinBuildInput::default_arena0_lite();
    a0_pass.crossbar.sona_meta_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0_pass.crossbar.cell_desc.cell_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0_pass.core_quads[0] = make_core_quad(0);
    a0_pass.conv_fir_lite.fir_l[0] = F32::new(0.8);
    a0_pass.conv_fir_lite.fir_r[0] = F32::new(0.8);
    a0_pass.linear_readout_lite.direct_s2_u_gain = le_f32([0.5, 0.5]);
    a0_pass.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);

    let mut a0_vca = a0_pass;
    a0_vca.crossbar.sense_svf_g = le_f32([0.15, 0.30, 0.55, 0.85]);
    a0_vca.crossbar.sense_lru_alpha_rel = le_f32([0.90; 4]);
    a0_vca.crossbar.sense_lru_alpha_atk = le_f32([0.15; 4]);
    a0_vca.crossbar.sense_in_l_fp16 = fp16_arr([1.0; 4]);
    a0_vca.crossbar.sense_in_r_fp16 = fp16_arr([1.0; 4]);
    a0_vca.crossbar.sense_tone_lp_fp16 = fp16_arr([1.0; 4]);
    a0_vca.crossbar.sense_tone_bp_fp16 = fp16_arr([0.5; 4]);
    a0_vca.crossbar.sense_vca_w_fp16 = fp16_arr([-0.6, -0.5, -0.4, -0.3]);

    // Wire ctrl 0 -> main_base -> slot 30 (sense_vca_gain_delta) via CextDispatchWire row 14.
    let ctrl_desc = ControlClampDescriptor::range(-1.0, 1.0, 0.0, 1, [0.0; 3]).to_wire();
    a0_vca.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(1);
    a0_vca.control_bus_lite.cp1_dir.entries[0] = [0, 1, 0xFF];
    a0_vca.control_bus_lite.cp1_base[32..36].copy_from_slice(F32::new(1.0).as_bytes());
    a0_vca.control_bus_lite.cp1_base[36..40].copy_from_slice(F32::new(1.0).as_bytes());
    let mut cext = CextDispatchWire::new_zeroed();
    cext.w_ctrl_ext[14][0] = F32::new(-0.45); // ctrl 0 -> z_head[0] -> slot 30 bias
    cext.w_ctrl_ext[14][12] = F32::new(-0.35); // det_env[0] -> q3 -> slot 30 compression loop

    let bin_pass = encode_lite_fixture(&a0_pass, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let bin_vca = encode_lite_fixture(
        &a0_vca,
        DNBN_FLAG_STEREO_OUT,
        0.99,
        &[ctrl_desc],
        Some(&cext),
    );

    let mut vm_pass =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_pass, Tier::Lite, 48_000.0).unwrap();
    let mut vm_vca =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_vca, Tier::Lite, 48_000.0).unwrap();

    let (in_l, in_r) = stereo_test_signal(160, 22);
    let (mut pl, mut pr, mut vl, mut vr) = ([0.0_f32; 160], [0.0_f32; 160], [0.0; 160], [0.0; 160]);
    vm_pass
        .process_block(&in_l, &in_r, &mut pl, &mut pr)
        .unwrap();
    vm_vca
        .process_block(&in_l, &in_r, &mut vl, &mut vr)
        .unwrap();

    // SensoryQuad builds non-zero ballistic envelopes that loop back via ControlBus q3 -> slot 30.
    let bus_env = vm_vca.slab().cold().ctrl_bus.as_ref().unwrap().env_states;
    assert!(bus_env.iter().all(|&e| e > 1.0e-4 && e.is_finite()));
    assert!(vm_vca.slab().cold().latched_stage_params[30] < -1.0e-4);

    let pass_rms: f64 = pl.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let vca_rms: f64 = vl.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    assert!(
        vca_rms < pass_rms * 0.85,
        "VCA + slot 30 must attenuate RMS"
    );
}

#[test]
fn test_m10b_op_n_pre_smooth_zoh_depth_and_sample_rate_invariance() {
    // 1) Continuous smooth transition across zoh_hold_depth in OpNPreHotState::step_stereo.
    let mut prev_energy: Option<f64> = None;
    for depth in [0.0_f32, 0.25, 0.50, 0.75, 0.95] {
        let p = OpNPreHoisted::from_params(1.0, depth, 4.0, 1.0);
        let mut st = OpNPreHotState::zero();
        st.apply_hoisted(p);
        let mut energy = 0.0_f64;
        for n in 0..128 {
            let x = (2.0 * core::f32::consts::PI * 900.0 * (n as f32) / 48_000.0).sin() * 0.5;
            let (yl, yr) = st.step_stereo(x, -x, p.d_scale, p.delta_step);
            assert!(yl.is_finite() && yr.is_finite());
            energy += f64::from(yl) * f64::from(yl);
        }
        if let Some(prev) = prev_energy {
            assert!(
                (energy - prev).abs() < 12.0,
                "discontinuous Op-N[Pre] step jump at depth={depth}"
            );
        }
        prev_energy = Some(energy);
    }

    // 2) Cross-rate ZOH invariance (44.1 / 48 / 96 kHz) for a low-frequency 120 Hz sine in SonaVm.
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.sona_meta_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0.crossbar.cell_desc.cell_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0.crossbar.op_n_pre_base.gamma_bc = F32::new(1.0);
    a0.crossbar.op_n_pre_base.zoh_hold_depth = F32::new(0.88);
    a0.crossbar.op_n_pre_base.beta_q = F32::new(6.0);
    a0.core_quads[0] = make_core_quad(0);
    a0.conv_fir_lite.fir_l[0] = F32::new(1.0);
    a0.conv_fir_lite.fir_r[0] = F32::new(1.0);
    a0.linear_readout_lite.direct_s2_u_gain = le_f32([0.8, 0.8]);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);
    let bin = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);

    let mut rms_by_rate = Vec::new();
    for &fs in &[44_100.0_f32, 48_000.0, 96_000.0] {
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, fs).unwrap();
        let n_samples = (0.04 * f64::from(fs)).round() as usize;
        let in_vec: Vec<f32> = (0..n_samples)
            .map(|n| 0.4 * (2.0 * core::f32::consts::PI * 120.0 * (n as f32) / fs).sin())
            .collect();
        let (mut ol, mut or) = (vec![0.0_f32; n_samples], vec![0.0_f32; n_samples]);
        vm.process_block(&in_vec, &in_vec, &mut ol, &mut or)
            .unwrap();
        let tail_start = n_samples / 2;
        let mean_sq: f64 = ol[tail_start..]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / ((n_samples - tail_start) as f64);
        rms_by_rate.push(mean_sq.sqrt());
    }
    for &r in &rms_by_rate {
        assert!(
            (r - rms_by_rate[1]).abs() / rms_by_rate[1].max(1e-6) < 0.08,
            "Op-N[Pre] ZOH invariance drift: {rms_by_rate:?}"
        );
    }
}

#[test]
fn test_m10b_ro_pole_48k_zoh_sample_rate_invariance() {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.core_quads[0] = make_core_quad(0);
    a0.conv_fir_lite.fir_l[0] = F32::new(1.0);
    a0.conv_fir_lite.fir_r[0] = F32::new(1.0);
    a0.linear_readout_lite.direct_s2_hp_gain = le_f32([1.0, 1.0]);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);

    let ro_48k = 0.992_f32;
    let bin = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, ro_48k, &[], None);

    // Compare hp_lp_pole ZOH scaling across 44.1 / 48 / 96 kHz and Table 4 cold clamping.
    for &fs in &[44_100.0_f32, 48_000.0, 96_000.0] {
        let vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, fs).unwrap();
        let rho_core = if (fs - 48_000.0).abs() < 1.0 {
            1.0_f64
        } else {
            48_000.0_f64 / f64::from(fs)
        };
        let expected_pole = powf_f64(f64::from(ro_48k), rho_core) as f32;
        assert!(
            (vm.slab().lite.hp_lp_pole - expected_pole).abs() < 1.0e-5,
            "fs={fs} pole={} vs expected={expected_pole}",
            vm.slab().lite.hp_lp_pole
        );
        // Physical decay exponent over 10 ms (= fs * 0.01 samples) matches across sample rates.
        let n_10ms = 0.01_f64 * f64::from(fs);
        let gain_10ms = f64::from(vm.slab().lite.hp_lp_pole).powf(n_10ms);
        let ref_10ms = f64::from(ro_48k).powf(480.0);
        assert!((gain_10ms - ref_10ms).abs() < 1.0e-3);
    }

    // Table 4 cold clamping: < 0.0, NaN, and +inf clamp to 0.0; > MAX_PARALRU_R_F32 clamps to
    // MAX_PARALRU_R_F32 on cold state while ZOH hp_lp_pole stays <= MAX_PARALRU_R_F32.
    for (label, raw_ro, expect_zero) in [
        ("neg", -0.5_f32, true),
        ("nan", f32::NAN, true),
        ("inf", f32::INFINITY, true),
        ("gt_max_1.0", 1.0_f32, false),
        ("gt_max_1.5", 1.5_f32, false),
    ] {
        let bytes = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: dnam_vm::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: raw_ro,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap_or_else(|e| panic!("encode {label} must succeed: {e:?}"));
        assert!(DnamBinView::parse(&bytes).is_ok(), "parse {label}");
        for &fs in &[48_000.0_f32, 96_000.0_f32] {
            let vm =
                SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bytes, Tier::Lite, fs).unwrap();
            if expect_zero {
                assert_eq!(vm.slab().cold().ro_pole_48k, 0.0, "{label}@{fs}");
                assert_eq!(vm.slab().lite.hp_lp_pole, 0.0, "{label}@{fs}");
            } else {
                assert_eq!(
                    vm.slab().cold().ro_pole_48k,
                    MAX_PARALRU_R_F32,
                    "{label}@{fs}"
                );
                assert!(
                    vm.slab().lite.hp_lp_pole > 0.99
                        && vm.slab().lite.hp_lp_pole <= MAX_PARALRU_R_F32,
                    "{label}@{fs} hp_lp_pole={}",
                    vm.slab().lite.hp_lp_pole
                );
            }
        }
    }
    let bin_max = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, MAX_PARALRU_R_F32, &[], None);
    let vm_max =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_max, Tier::Lite, 96_000.0).unwrap();
    assert_eq!(vm_max.slab().cold().ro_pole_48k, MAX_PARALRU_R_F32);
    assert!(vm_max.slab().lite.hp_lp_pole <= MAX_PARALRU_R_F32);
}

#[test]
fn test_m10b_core_quad_ms_coupling_duhem_stereo_link_and_glu_mix() {
    let mut base_a0 = DnamBinBuildInput::default_arena0_lite();
    let mut q0 = CoreQuadWire::new_zeroed();
    q0.svf_g = le_f32([0.35; 4]);
    q0.svf_k_fp16 = fp16_arr([0.707; 4]);
    q0.lru_alpha_rel = le_f32([0.92; 4]);
    q0.lru_alpha_atk = le_f32([0.25; 4]);
    q0.lru_kappa_fp16 = fp16_arr([0.5; 4]);
    q0.adaa_drive_fp16 = fp16_arr([1.5; 4]);
    q0.tone_dir_fp16 = fp16_arr([1.0; 4]);
    q0.rect_sign_fp16 = fp16_arr([1.0; 4]);
    q0.in_mid_weight_fp16 = fp16_arr([1.0; 4]);
    q0.in_side_weight_fp16 = fp16_arr([0.0; 4]);
    base_a0.core_quads[0] = q0;
    base_a0.conv_fir_lite.fir_l[0] = F32::new(1.0);
    base_a0.conv_fir_lite.fir_r[0] = F32::new(1.0);
    base_a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);
    for l in 0..4 {
        base_a0.linear_readout_lite.w_ro_fp16[l * 12 + 4] = fp16_val(0.25);
        base_a0.linear_readout_lite.w_ro_fp16[l * 12 + 5] = fp16_val(0.25);
    }

    // 1) Pure Mid coupling rejects pure Side [x, -x]; pure Side rejects pure Mid [x, x].
    let bin_mid = encode_lite_fixture(&base_a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let mut side_a0 = base_a0;
    side_a0.core_quads[0].in_mid_weight_fp16 = fp16_arr([0.0; 4]);
    side_a0.core_quads[0].in_side_weight_fp16 = fp16_arr([1.0; 4]);
    let bin_side = encode_lite_fixture(&side_a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);

    let mut vm_mid =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_mid, Tier::Lite, 48_000.0).unwrap();
    let mut vm_side =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_side, Tier::Lite, 48_000.0).unwrap();
    for n in 0..32 {
        let s = 0.4 * (0.2 * (n as f32)).sin();
        let (ml, mr) = vm_mid.step_frame(s, -s).unwrap();
        let (sl, sr) = vm_side.step_frame(s, s).unwrap();
        assert_eq!((ml, mr), (0.0, 0.0), "Mid coupling must reject pure side");
        assert_eq!((sl, sr), (0.0, 0.0), "Side coupling must reject pure mid");
    }

    // 2) Enable Selective-Duhem hysteresis, stereo link, and GLU mix on full signal and verify divergence.
    let mut full_a0 = base_a0;
    full_a0.core_quads[0].in_side_weight_fp16 = fp16_arr([0.5; 4]);
    full_a0.core_quads[0].lru_duhem_gamma_fp16 = fp16_arr([16.0; 4]);
    full_a0.core_quads[0].lru_signed_mix_fp16 = fp16_arr([0.85; 4]);
    full_a0.core_quads[0].lru_stereo_link_fp16 = fp16_arr([0.9; 4]);
    full_a0.core_quads[0].glu_mix_fp16 = fp16_arr([2.0, -2.0, 1.5, -1.5]);

    let bin_plain = encode_lite_fixture(&base_a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let bin_ext = encode_lite_fixture(&full_a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let mut vm_p =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_plain, Tier::Lite, 48_000.0)
            .unwrap();
    let mut vm_e =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_ext, Tier::Lite, 48_000.0).unwrap();

    let (in_l, in_r) = stereo_test_signal(64, 24);
    let (mut pl, mut pr, mut el, mut er) = ([0.0_f32; 64], [0.0; 64], [0.0; 64], [0.0; 64]);
    vm_p.process_block(&in_l, &in_r, &mut pl, &mut pr).unwrap();
    vm_e.process_block(&in_l, &in_r, &mut el, &mut er).unwrap();
    assert_finite_non_silent(&el, &er, "Duhem + stereo-link + GLU");
    let diff = el
        .iter()
        .zip(pl.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        diff > 1.0e-3,
        "Duhem + stereo-link + GLU must alter readout"
    );
}

#[test]
fn test_m10b_t1_subharmonic_rotor_28b_octave_harmonics() {
    assert_eq!(core::mem::size_of::<T1SubHarmonicKernel>(), 28);
    assert_eq!(core::mem::align_of::<T1SubHarmonicKernel>(), 4);

    // Sweep analytic unit-circle octave pairs: f1 (theta) vs f2 (2*theta) for continuous sheet toggling.
    for &omega in &[0.08_f64, 0.16] {
        let mut rotor = T1SubHarmonicKernel::new();
        let mut prev_s1: Option<(f32, f32)> = None;
        let mut prev_s2: Option<(f32, f32)> = None;
        for n in 0..512 {
            let phase = omega * (n as f64);
            let out = rotor.step(phase.cos() as f32, phase.sin() as f32);
            let (c1, s1, c2, s2) = (out[0], out[1], out[2], out[3]);
            assert!(((c1 * c1 + s1 * s1) - 1.0).abs() < 1.0e-4);
            assert!(((c2 * c2 + s2 * s2) - 1.0).abs() < 1.0e-4);
            if let Some((pc1, ps1)) = prev_s1 {
                let chord1 = ((c1 - pc1).powi(2) + (s1 - ps1).powi(2)).sqrt();
                assert!(
                    chord1 < (omega as f32) * 0.75 + 0.01,
                    "sub1 branch-cut glitch"
                );
            }
            if let Some((pc2, ps2)) = prev_s2 {
                let chord2 = ((c2 - pc2).powi(2) + (s2 - ps2).powi(2)).sqrt();
                assert!(
                    chord2 < (omega as f32) * 0.45 + 0.01,
                    "sub2 branch-cut glitch"
                );
            }
            prev_s1 = Some((c1, s1));
            prev_s2 = Some((c2, s2));
        }
        let (sh1, sh2, _, valid) = rotor.state();
        assert!(valid && (sh1.abs() == 1.0) && (sh2.abs() == 1.0));
        assert_eq!(rotor.step(0.0, 0.0), [0.0; 4]);
        rotor.reset();
        assert!(!rotor.state().3);
    }

    // Exercise Op-P T1 rotor inside live SonaVm via crossbar_op_gains_fp16[0] across harmonic inputs.
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.sona_meta_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0.crossbar.cell_desc.cell_flags = U32::new(SONA_META_SENSORY_QUAD);
    a0.crossbar.crossbar_op_gains_fp16 = fp16_arr([1.0, 0.0]);
    a0.core_quads[0] = make_core_quad(0);
    a0.conv_fir_lite.fir_l[0] = F32::new(0.9);
    a0.conv_fir_lite.fir_r[0] = F32::new(0.9);
    a0.linear_readout_lite.stage_latent_mods_fp16[80..84].fill(fp16_val(0.6));
    a0.linear_readout_lite.direct_s2_u_gain = le_f32([0.4, 0.4]);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);
    let bin = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, 48_000.0).unwrap();
    let (in_l, in_r) = stereo_test_signal(256, 25);
    let (mut ol, mut or) = ([0.0_f32; 256], [0.0_f32; 256]);
    vm.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
    assert_finite_non_silent(&ol, &or, "Op-P T1 octave harmonic pipeline");
}

#[test]
fn test_m10b_op_t_sigma_sat_freeze_mix_dry_mix_and_immediate_write_lock() {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.sona_meta_flags = U32::new(SONA_META_TRANSPORT_TAP_GAINS);
    a0.crossbar.cell_desc.cell_flags = U32::new(SONA_META_TRANSPORT_TAP_GAINS);
    a0.core_quads[0] = make_core_quad(0);
    a0.conv_fir_lite.fir_l[0] = F32::new(0.9);
    a0.conv_fir_lite.fir_r[0] = F32::new(0.9);
    a0.linear_readout_lite.direct_s2_u_gain = le_f32([1.0, 1.0]);
    a0.transport_lite = make_transport_quad(0);
    a0.transport_lite.g_loop = F32::new(0.98);
    a0.transport_lite.sigma_knee_fp16 = fp16_val(0.50);
    a0.transport_lite.freeze_mix_fp16 = fp16_val(0.25);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(0.0); // pure wet transport

    let bin_pure_wet = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);
    let bin_with_dry = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);

    // 1) Stage tank_dry_mix law (0.0 vs 1.0): frame 0 pure wet is 0 before delay read arrival.
    let mut vm_wet =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_pure_wet, Tier::Lite, 48_000.0)
            .unwrap();
    let mut vm_dry =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_with_dry, Tier::Lite, 48_000.0)
            .unwrap();
    let (wl0, wr0) = vm_wet.step_frame(0.5, -0.3).unwrap();
    let (dl0, dr0) = vm_dry.step_frame(0.5, -0.3).unwrap();
    assert_eq!((wl0, wr0), (0.0, 0.0));
    assert!(dl0.abs() > 1.0e-3 && dr0.abs() > 1.0e-3);

    // 2) Extreme drive stays strictly bounded by unified sigma_sat cubic limiter.
    for _ in 0..512 {
        let (yl, yr) = vm_wet.step_frame(100.0, -100.0).unwrap();
        assert!(yl.is_finite() && yr.is_finite());
        assert!(yl.abs() < 25.0 && yr.abs() < 25.0);
    }

    // 3) Immediate set_op_t_write_enable(true/false) syncs HOST_WRITE_LOCK and mu_freeze without lag.
    assert_eq!(
        vm_dry.slab().cold().transport_quad_eff[0].mu_freeze,
        eval_op_t_freeze_mix_eff(0.25, 0.0, false)
    );
    vm_dry.set_op_t_write_enable(true);
    assert_ne!(
        vm_dry.slab().lite.transport_quad0.flags & QUAD_FLAG_HOST_WRITE_LOCK,
        0
    );
    assert_eq!(vm_dry.slab().cold().transport_quad_eff[0].mu_freeze, 1.0);

    vm_dry.set_op_t_write_enable(false);
    assert_eq!(
        vm_dry.slab().lite.transport_quad0.flags & QUAD_FLAG_HOST_WRITE_LOCK,
        0
    );
    assert_eq!(vm_dry.slab().cold().transport_quad_eff[0].mu_freeze, 0.25);

    // 4) Continuous freeze_mix across [0.0, 0.25, 0.5, 0.75, 1.0], freeze==1.0 0-ULP sustain, and slot-16 dynamic sweep.
    let mut a0_freeze = DnamBinBuildInput::default_arena0_lite();
    a0_freeze.crossbar.sona_meta_flags = U32::new(SONA_META_TRANSPORT_TAP_GAINS);
    a0_freeze.crossbar.cell_desc.cell_flags = U32::new(SONA_META_TRANSPORT_TAP_GAINS);
    a0_freeze.core_quads[0] = make_core_quad(0);
    a0_freeze.conv_fir_lite.fir_l[0] = F32::new(0.9);
    a0_freeze.conv_fir_lite.fir_r[0] = F32::new(0.9);
    a0_freeze.linear_readout_lite.direct_s2_u_gain = le_f32([1.0, 1.0]);
    a0_freeze.control_bus_lite.tank_dry_mix_fp16 = fp16_val(0.0);
    a0_freeze.transport_lite = make_transport_quad(0);
    a0_freeze.transport_lite.flags = 0;
    a0_freeze.transport_lite.allpass_delays = [0; 4];
    a0_freeze.transport_lite.allpass_g_fp16 = fp16_arr([0.0; 4]);
    a0_freeze.transport_lite.delay_samples = le_f32([16.0; 4]);
    a0_freeze.transport_lite.ramp_spans_samples = le_f32([0.0; 4]);
    a0_freeze.transport_lite.lfo_depths_fp16 = fp16_arr([0.0; 4]);
    a0_freeze.transport_lite.damp_pole = F32::new(0.0);
    a0_freeze.transport_lite.g_loop = F32::new(0.92);
    a0_freeze.transport_lite.sigma_knee_fp16 = fp16_val(2.0);
    a0_freeze.transport_lite.freeze_mix_fp16 = fp16_val(0.0);

    // Route ctrl 0 in [0, 1] -> z_head[0] -> slot 16 (op_t_freeze_mix_delta) cleanly at 1:1 scale.
    let freeze_ctrl = ControlClampDescriptor::range(0.0, 1.0, 0.0, 1, [-1.0, 0.0, 0.0]).to_wire();
    a0_freeze.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(1);
    a0_freeze.control_bus_lite.cp1_dir.entries[0] = [0, 1, 0xFF];
    a0_freeze.control_bus_lite.cp1_base[0..4].copy_from_slice(F32::new(0.5).as_bytes());
    let mut freeze_cext = CextDispatchWire::new_zeroed();
    freeze_cext.w_ctrl_ext[0][0] = F32::new(1.0);

    let bin_freeze_decay = encode_lite_fixture(
        &a0_freeze,
        DNBN_FLAG_STEREO_OUT,
        0.99,
        &[freeze_ctrl],
        Some(&freeze_cext),
    );
    let mut energies = [0.0_f64; 5];
    let fmix_levels = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
    for (idx, &fmix) in fmix_levels.iter().enumerate() {
        let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(
            &bin_freeze_decay,
            Tier::Lite,
            48_000.0,
        )
        .unwrap();
        // Prime ring cleanly with sustained non-zero drive across 3 full 3 kHz ticks (48 frames).
        for n in 0..48 {
            let s = 0.35 * (2.0 * core::f32::consts::PI * (n as f32) / 16.0).sin();
            let c = 0.25 * (2.0 * core::f32::consts::PI * (n as f32) / 16.0).cos();
            let _ = vm.step_frame(s, c).unwrap();
        }
        vm.set_control_physical(0, fmix).unwrap();
        // Complete the 16-sample uniform micro-ramp + 1 boundary frame so slot 16 latches fmix.
        for _ in 0..17 {
            let _ = vm.step_frame(0.0, 0.0).unwrap();
        }
        assert!(
            (vm.slab().cold().transport_quad_eff[0].mu_freeze - fmix).abs() < 1.0e-5,
            "fmix={fmix} vs mu_freeze={}",
            vm.slab().cold().transport_quad_eff[0].mu_freeze
        );

        let mut tail_energy = 0.0_f64;
        for _ in 0..128 {
            let (yl, yr) = vm.step_frame(0.0, 0.0).unwrap();
            assert!(yl.is_finite() && yr.is_finite());
            tail_energy += f64::from(yl) * f64::from(yl) + f64::from(yr) * f64::from(yr);
        }
        assert!(
            tail_energy > 1.0e-4,
            "expected non-zero tail at fmix={fmix}"
        );
        energies[idx] = tail_energy;
    }
    assert!(
        energies[0] < energies[1]
            && energies[1] < energies[2]
            && energies[2] < energies[3]
            && energies[3] < energies[4],
        "expected strictly monotonic freeze_mix retained tail energies, got {energies:?}"
    );

    // 4b) freeze_mix == 1.0 with unity loop gain / zero damping produces 0-ULP ring hold and periodic sustain.
    a0_freeze.transport_lite.g_loop = F32::new(TANK_MAX_G_LIN_F32);
    let bin_sustain = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: DNBN_FLAG_STEREO_OUT,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: 0.805,
        tail_time_ms: 0.0,
        ro_pole_48k: 0.99,
        switch_xfade_samples: 0,
        content_hash_prefix: [0; 8],
        controls: &[freeze_ctrl],
        cp1_overflow: &[],
        tsyn: &[],
        cext: Some(&freeze_cext),
        arena0: &a0_freeze,
        arena1: None,
        arena2: None,
        arena3: None,
    })
    .unwrap();
    let mut vm_sus =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_sustain, Tier::Lite, 48_000.0)
            .unwrap();
    for n in 0..48 {
        let s = 0.35 * (2.0 * core::f32::consts::PI * (n as f32) / 16.0).sin();
        let c = 0.25 * (2.0 * core::f32::consts::PI * (n as f32) / 16.0).cos();
        let _ = vm_sus.step_frame(s, c).unwrap();
    }
    vm_sus.set_control_physical(0, 1.0).unwrap();
    for _ in 0..17 {
        let _ = vm_sus.step_frame(0.0, 0.0).unwrap();
    }
    assert_eq!(vm_sus.slab().cold().transport_quad_eff[0].mu_freeze, 1.0);
    let frozen_ring = vm_sus.slab().lite.transport0_ring;
    for _ in 0..192 {
        let _ = vm_sus.step_frame(0.0, 0.0).unwrap();
    }
    assert_eq!(
        vm_sus.slab().lite.transport0_ring,
        frozen_ring,
        "freeze_mix == 1.0 must keep ring 0-ULP frozen"
    );
    let mut sustain_l = [0.0_f32; 64];
    let mut sustain_r = [0.0_f32; 64];
    for i in 0..64 {
        let (yl, yr) = vm_sus.step_frame(0.0, 0.0).unwrap();
        sustain_l[i] = yl;
        sustain_r[i] = yr;
    }
    assert_finite_non_silent(&sustain_l, &sustain_r, "freeze_mix == 1.0 sustain");
    for i in 0..48 {
        assert_eq!(
            sustain_l[i].to_bits(),
            sustain_l[i + 16].to_bits(),
            "0-ULP periodic sustain L@{i}"
        );
        assert_eq!(
            sustain_r[i].to_bits(),
            sustain_r[i + 16].to_bits(),
            "0-ULP periodic sustain R@{i}"
        );
    }

    // 4c) Dynamically sweeping slot 16 (op_t_freeze_mix_delta) from 0.0 -> 1.0 has no step glitch.
    let mut vm_sweep =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_freeze_decay, Tier::Lite, 48_000.0)
            .unwrap();
    let (mut prev_l, mut prev_r) = (0.0_f32, 0.0_f32);
    let mut prev_mu = -1.0_f32;
    let mut max_step_delta = 0.0_f32;
    for tick in 0..=64 {
        let fmix = (tick as f32) / 64.0;
        vm_sweep.set_control_physical(0, fmix).unwrap();
        for f in 0..16 {
            let n = tick * 16 + f;
            let s = 0.25 * (2.0 * core::f32::consts::PI * 220.0 * (n as f32) / 48_000.0).sin();
            let (yl, yr) = vm_sweep.step_frame(s, -0.8 * s).unwrap();
            assert!(yl.is_finite() && yr.is_finite());
            let cur_mu = vm_sweep.slab().cold().transport_quad_eff[0].mu_freeze;
            assert!(cur_mu >= prev_mu, "monotonic slot-16 sweep at sample={n}");
            prev_mu = cur_mu;
            if n > 0 {
                max_step_delta = max_step_delta
                    .max((yl - prev_l).abs())
                    .max((yr - prev_r).abs());
            }
            (prev_l, prev_r) = (yl, yr);
        }
    }
    let _ = vm_sweep.step_frame(0.0, 0.0).unwrap();
    let final_mu = vm_sweep.slab().cold().transport_quad_eff[0].mu_freeze;
    assert!((final_mu - 1.0).abs() < 1.0e-5, "final_mu={final_mu}");
    assert!(
        max_step_delta < 0.25,
        "unexpected freeze_mix sweep step glitch: {max_step_delta}"
    );
}

#[test]
fn test_m10b_tier_switching_monotonicity_and_bounded_rt_invariants() {
    assert_eq!(
        SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false),
        4_544
    );
    assert_eq!(
        SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Full, true),
        7_552
    );
    assert_eq!(
        SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Pro, true),
        11_136
    );
    assert_eq!(
        SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true),
        14_592
    );
    assert_eq!(
        SONA_LITE_HOT_BYTES
            + SONA_OS2X_HOT_BYTES
            + SONA_FULL_HOT_DELTA_BYTES
            + SONA_PRO_HOT_DELTA_BYTES
            + SONA_ULTRA_HOT_DELTA_BYTES,
        14_592
    );

    let ultra = build_full_pipeline_ultra_fixture();
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Lite, 48_000.0).unwrap();

    let expected_counts = [
        (Tier::Lite, 4_usize, 2_usize, 1_usize),
        (Tier::Full, 6, 3, 2),
        (Tier::Pro, 8, 6, 3),
        (Tier::Ultra, 10, 8, 4),
    ];
    let (in_l, in_r) = stereo_test_signal(64, 29);
    let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);

    let mut last_cores = 0;
    let mut last_kautz = 0;
    let mut last_trans = 0;
    for &(tier, exp_cores, exp_kautz, exp_trans) in &expected_counts {
        vm.switch_tier(tier).unwrap();
        let active_cores = vm.slab().lite.tier.simd_groups();
        let cold = vm.slab().cold();
        assert_eq!(active_cores, exp_cores);
        assert_eq!(cold.active_kautz_blocks, exp_kautz);
        assert_eq!(cold.active_transport_quads, exp_trans);
        assert!(exp_cores > last_cores && exp_kautz > last_kautz && exp_trans > last_trans);
        last_cores = exp_cores;
        last_kautz = exp_kautz;
        last_trans = exp_trans;

        vm.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
        assert_finite_non_silent(&ol, &or, &format!("tier switch {tier:?}"));
    }

    // Downgrade Ultra -> Lite and verify bounded output under high-amplitude pulses.
    vm.switch_tier(Tier::Lite).unwrap();
    for &t in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let mut vm_t =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, t, 48_000.0).unwrap();
        for n in 0..128 {
            let x = if (n & 7) == 0 { 16.0_f32 } else { -12.0_f32 };
            let (yl, yr) = vm_t.step_frame(x, -x).unwrap();
            assert!(yl.is_finite() && yr.is_finite());
            let _ = exp2_signed_oct_f32(0.0);
        }
    }
}
