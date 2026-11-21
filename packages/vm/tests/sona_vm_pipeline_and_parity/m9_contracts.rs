//! M9 clean-room contract verification tests: `384 kHz` live retiming, failure-atomic `192 kHz`
//! `4x` offline bounce rejection, `WDW` 4-port `m_c + m_w` / `ms_width` / `4CM` restoration,
//! unconditional `Op-P` + phasor stepping, `Op-T` 4-frame micro-ramp vs `3 kHz` Schmitt/`psi`
//! cadence, and non-advancing LFO/drift state on dirty control latches.

#![forbid(unsafe_code)]

use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16};

use dnam_kernels::{QUAD_FLAG_HOST_WRITE_LOCK, QUAD_FLAG_WE_HOLD_ACTIVE};
use dnam_vm::bin::{Arena1FullDeltaWire, DnamBinBuildInput, SonaGroupWire, encode_dnam_bin};
use dnam_vm::graph::CabinetHighSrMode;
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, f32_to_fp16};
use dnam_vm::{
    DefaultMathBackend, DynamicSonaVm, IoAlignmentMode, SonaVm, Tier, VmError, WdwDryTapMode,
};

use super::helpers::{build_full_pipeline_ultra_fixture, stereo_test_signal};
use crate::assert_finite_non_silent;

fn fp16_arr_4(vals: [f32; 4]) -> [U16; 4] {
    vals.map(|v| U16::new(f32_to_fp16(v)))
}

fn build_zero_crossbar_lite_full_fixture() -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.sona_meta_flags = zerocopy::byteorder::little_endian::U32::new(0);
    a0.crossbar.cell_desc.cell_flags = zerocopy::byteorder::little_endian::U32::new(0);
    a0.crossbar.cell_desc.tier_flags = U16::new(0x0003);
    a0.crossbar.cell_desc.oversample_factor = U16::new(2);
    a0.crossbar.phasor_freq_hz = [
        F32::new(110.0),
        F32::new(220.0),
        F32::new(330.0),
        F32::new(440.0),
    ];
    a0.crossbar.gate_and_crossbar_fp16 = [U16::new(0); 8];
    a0.crossbar.carrier_basis_w_fp16 = [[U16::new(0); 4]; 3];

    let mut grp = SonaGroupWire::new_zeroed();
    grp.svf_g = [F32::new(0.32); 4];
    grp.svf_k = [F32::new(0.707); 4];
    grp.lru_alpha_rel = [F32::new(0.25); 4];
    grp.lru_alpha_atk = [F32::new(0.99); 4];
    grp.in_x_weight = [
        F32::new(0.45),
        F32::new(0.35),
        F32::new(0.25),
        F32::new(0.15),
    ];
    grp.tone_lp_fp16 = fp16_arr_4([0.5; 4]);
    grp.tone_dir_fp16 = fp16_arr_4([1.0; 4]);
    grp.rect_sign_fp16 = fp16_arr_4([1.0; 4]);
    a0.groups = [grp; 4];

    a0.cabinet_fir_lite.fir_l[0] = F32::new(0.8);
    a0.cabinet_fir_lite.fir_r[0] = F32::new(0.8);
    a0.spatial_tank_lite.delay_samples = [F32::new(12.0); 4];
    a0.spatial_tank_lite.g_loop = F32::new(0.20);
    a0.spatial_tank_lite.dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
    a0.varpro_readout_lite.direct_s2_u_gain = [F32::new(0.4), F32::new(0.4)];
    for idx in 0..16 {
        a0.varpro_readout_lite.w_ro_fp16[idx * 12 + 4] = U16::new(f32_to_fp16(0.25));
        a0.varpro_readout_lite.w_ro_fp16[idx * 12 + 5] = U16::new(f32_to_fp16(0.25));
    }

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    a1.groups = [grp; 2];
    a1.cabinet_fir_full.fir_l[0] = F32::new(0.2);
    a1.cabinet_fir_full.fir_r[0] = F32::new(0.2);
    a1.spatial_tank_full.delay_samples = [F32::new(16.0); 4];
    a1.spatial_tank_full.g_loop = F32::new(0.15);
    a1.spatial_tank_full.r_dc = F32::new(TANK_DC_BLOCKER_48K_F32);

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 120,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        r_ac_default: 0.99,
        switch_xfade_samples: 0,
        content_hash_prefix: [0; 8],
        controls: &[],
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: None,
        arena3: None,
    })
    .expect("encode zero-crossbar Lite+Full fixture")
}

#[test]
fn test_retime_384k_and_failure_atomic_offline_bounce_192k() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(64, 11);
    let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);

    let mut dyn_vm = DynamicSonaVm::open_dnam_bin(&ultra, Tier::Ultra, 48_000.0).unwrap();
    dyn_vm.retime_sample_rate(384_000.0).unwrap();
    assert!(!dyn_vm.telemetry().os2x_active);
    dyn_vm
        .process_block(&in_l, &in_r, &mut ol, &mut or)
        .unwrap();
    assert_finite_non_silent(&ol, &or, "DynamicSonaVm retime 384k");

    let mut owned_vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 48_000.0).unwrap();
    owned_vm.retime_sample_rate(384_000.0).unwrap();
    assert_eq!(
        owned_vm.slab().cold().cabinet_fir_ext.mode(),
        CabinetHighSrMode::CaseA { stride_q: 8 }
    );
    assert!(!owned_vm.telemetry().os2x_active);
    owned_vm
        .process_block(&in_l, &in_r, &mut ol, &mut or)
        .unwrap();
    assert_finite_non_silent(&ol, &or, "SonaVm retime 384k");

    // 192 kHz * 4x offline bounce = 768 kHz > 384 kHz -> Err(InvalidSampleRate), zero state mutation.
    let mut vm_fail =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 192_000.0).unwrap();
    let mut vm_twin =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 192_000.0).unwrap();
    vm_fail
        .process_block(&in_l, &in_r, &mut ol, &mut or)
        .unwrap();
    vm_twin
        .process_block(&in_l, &in_r, &mut ol, &mut or)
        .unwrap();

    assert_eq!(
        vm_fail.set_offline_bounce(true),
        Err(VmError::InvalidSampleRate)
    );
    assert!(!vm_fail.slab().cold().offline_bounce);
    assert_eq!(vm_fail.get_latency_samples(), 0);
    assert_eq!(
        vm_fail.slab().cold().cabinet_fir_ext.mode(),
        CabinetHighSrMode::CaseA { stride_q: 4 }
    );

    let (mut l_fail, mut r_fail, mut l_twin, mut r_twin) =
        ([0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64]);
    vm_fail
        .process_block(&in_l, &in_r, &mut l_fail, &mut r_fail)
        .unwrap();
    vm_twin
        .process_block(&in_l, &in_r, &mut l_twin, &mut r_twin)
        .unwrap();
    assert_finite_non_silent(&l_fail, &r_fail, "192k post-failed-bounce audio");
    for i in 0..64 {
        assert_eq!(l_fail[i].to_bits(), l_twin[i].to_bits(), "atomic L@{i}");
        assert_eq!(r_fail[i].to_bits(), r_twin[i].to_bits(), "atomic R@{i}");
    }

    let mut dyn_fail = DynamicSonaVm::open_dnam_bin(&ultra, Tier::Ultra, 192_000.0).unwrap();
    assert_eq!(
        dyn_fail.set_offline_bounce(true),
        Err(VmError::InvalidSampleRate)
    );
    assert!(!dyn_fail.telemetry().offline_bounce);
    assert_eq!(dyn_fail.telemetry().latency_samples, 0);
}

#[test]
fn test_wdw_4port_partition_ms_width_and_4cm_restore() {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    let mf = dnam_vm::bin::SONA_META_ROUTER_EXT | dnam_vm::bin::SONA_META_TANK_TAP_GAINS;
    a0.crossbar.sona_meta_flags = zerocopy::byteorder::little_endian::U32::new(mf);
    a0.crossbar.cell_desc.cell_flags = zerocopy::byteorder::little_endian::U32::new(mf);
    a0.crossbar.s_router = [
        [F32::new(0.75), F32::new(0.25), F32::new(0.0), F32::new(0.0)],
        [F32::new(0.15), F32::new(0.85), F32::new(0.0), F32::new(0.0)],
        [F32::new(1.0), F32::new(0.0), F32::new(0.2), F32::new(0.0)],
        [F32::new(0.0), F32::new(1.0), F32::new(0.0), F32::new(0.2)],
    ];
    let mut grp = SonaGroupWire::new_zeroed();
    grp.svf_g = [F32::new(0.35); 4];
    grp.svf_k = [F32::new(0.707); 4];
    grp.in_x_weight = [F32::new(0.4); 4];
    grp.tone_dir_fp16 = fp16_arr_4([1.0; 4]);
    grp.rect_sign_fp16 = fp16_arr_4([1.0; 4]);
    a0.groups[0] = grp;
    a0.cabinet_fir_lite.fir_l[0] = F32::new(0.9);
    a0.cabinet_fir_lite.fir_r[0] = F32::new(0.8);
    a0.spatial_tank_lite.delay_samples = [
        F32::new(10.0),
        F32::new(14.0),
        F32::new(18.0),
        F32::new(22.0),
    ];
    a0.spatial_tank_lite.g_loop = F32::new(0.35);
    a0.spatial_tank_lite.input_gains_fp16 = fp16_arr_4([0.3; 4]);
    a0.spatial_tank_lite.wet_tap_l_fp16 = fp16_arr_4([0.35, 0.15, 0.25, 0.10]);
    a0.spatial_tank_lite.wet_tap_r_fp16 = fp16_arr_4([0.10, 0.25, 0.15, 0.35]);
    a0.spatial_tank_lite.dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
    a0.varpro_readout_lite.direct_s2_u_gain = [F32::new(0.6), F32::new(0.4)];

    let lite = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 120,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        r_ac_default: 0.99,
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
    .unwrap();

    let (in_l, in_r) = stereo_test_signal(48, 12);
    let mut vm_4cm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&lite, Tier::Lite, 48_000.0).unwrap();
    let mut vm_wdw =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&lite, Tier::Lite, 48_000.0).unwrap();
    let mut vm_width0 =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&lite, Tier::Lite, 48_000.0).unwrap();
    let mut vm_width2 =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&lite, Tier::Lite, 48_000.0).unwrap();

    vm_wdw.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
    vm_width0.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
    vm_width0.set_ms_width(0.0);
    vm_width2.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
    vm_width2.set_ms_width(2.0);

    let mut max_wet_diff = 0.0_f32;
    let mut max_width_side = 0.0_f32;
    for i in 0..32 {
        let (m_4cm, _s_4cm) = vm_4cm
            .step_frame_4port([in_l[i], in_r[i]], [0.0, 0.0], false)
            .unwrap();
        let (m_wdw, s_wdw) = vm_wdw
            .step_frame_4port([in_l[i], in_r[i]], [0.0, 0.0], false)
            .unwrap();
        let (m_w0, s_w0) = vm_width0
            .step_frame_4port([in_l[i], in_r[i]], [0.0, 0.0], false)
            .unwrap();
        let (m_w2, s_w2) = vm_width2
            .step_frame_4port([in_l[i], in_r[i]], [0.0, 0.0], false)
            .unwrap();

        // With g_dry_sum == 1.0: m_c + m_w == 0.5 * (y_4cm_L + y_4cm_R).
        let mid_partition = s_wdw[1] + 0.5 * (m_wdw[0] + m_wdw[1]);
        let mid_4cm = 0.5 * (m_4cm[0] + m_4cm[1]);
        assert!((mid_partition - mid_4cm).abs() < 1.0e-5, "partition @{i}");

        // ms_width scales Ports Out 2/3 without changing center dry Port Out 1 (`send_lr[1]`).
        assert_eq!(s_w0[1].to_bits(), s_w2[1].to_bits());
        assert_eq!(m_w0[0].to_bits(), m_w0[1].to_bits());
        let wet_side = 0.5 * (m_w2[0] - m_w2[1]);
        max_width_side = max_width_side.max(wet_side.abs());
        max_wet_diff = max_wet_diff.max((m_wdw[0] - m_4cm[0]).abs());
    }
    assert!(max_wet_diff > 1.0e-4 && max_width_side > 1.0e-5);

    // DirectDry taps 0.5*(inst_L + inst_R) when !has_return, or 0.5*(ret_aligned_L + R) when has_return.
    let mut vm_dir =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&lite, Tier::Lite, 48_000.0).unwrap();
    vm_dir.configure_return_alignment(IoAlignmentMode::MinLatency4Tap, 0.0, 0.0);
    vm_dir.set_wdw_mode(WdwDryTapMode::DirectDry);
    let (_m_inst, s_inst) = vm_dir
        .step_frame_4port([0.4, -0.2], [0.9, 0.9], false)
        .unwrap();
    assert!((s_inst[1] - 0.1).abs() < 1.0e-6);
    for _ in 0..8 {
        let (_m_ret, s_ret) = vm_dir
            .step_frame_4port([0.4, -0.2], [0.6, 0.2], true)
            .unwrap();
        let _ = s_ret;
    }
    let (_m_ret, s_ret) = vm_dir
        .step_frame_4port([0.4, -0.2], [0.6, 0.2], true)
        .unwrap();
    assert!((s_ret[1] - 0.4).abs() < 1.0e-5);

    // set_wdw_enabled(false) restores exact 4CM send_lr[1].
    vm_wdw.set_wdw_enabled(false);
    let (_m_restored, s_restored) = vm_wdw
        .step_frame_4port([in_l[32], in_r[32]], [0.0, 0.0], false)
        .unwrap();
    let (_m_base, s_base) = vm_4cm
        .step_frame_4port([in_l[32], in_r[32]], [0.0, 0.0], false)
        .unwrap();
    assert_eq!(s_restored[1].to_bits(), s_base[1].to_bits());
}

#[test]
fn test_unconditional_opp_and_phasor_stepping_across_tier_switch() {
    let bin = build_zero_crossbar_lite_full_fixture();
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, 48_000.0).unwrap();
    let (in_l, in_r) = stereo_test_signal(64, 13);
    let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);

    // Step in Lite where crossbar is zero/inactive; Op-P & phasors still advance unconditionally.
    vm.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
    assert_finite_non_silent(&ol, &or, "zero-crossbar Lite audio");
    let lite_phases = vm.slab().lite.phasors.phase;
    let lite_t1 = vm.slab().lite.t1;
    let last_iq = vm.slab().cold().last_op_p_iq_lr;
    let last_feat = vm.slab().cold().last_op_p_feat;
    assert_ne!(lite_phases, [0_u32; 4]);
    assert!(last_iq.iter().any(|&v| v.abs() > 1.0e-6));
    assert!(last_feat.iter().any(|&v| v.abs() > 1.0e-6));

    // Switch Lite -> Full: os2x_hot seeds directly from last_op_p_* without jumping back to 0.
    vm.switch_tier(Tier::Full).unwrap();
    assert_eq!(
        vm.slab().os2x_hot.op_p_prev_iq_lr.map(f32::to_bits),
        last_iq.map(f32::to_bits)
    );
    assert_eq!(
        vm.slab().os2x_hot.op_p_prev_feat.map(f32::to_bits),
        last_feat.map(f32::to_bits)
    );
    assert_eq!(vm.slab().lite.phasors.phase, lite_phases);
    assert_eq!(vm.slab().lite.t1, lite_t1);
    vm.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
    assert_ne!(vm.slab().lite.phasors.phase, lite_phases);
}

#[test]
fn test_opt_micro_ramp_4frame_latch_and_dirty_lfo_drift_invariance() {
    // 44.1 kHz has ctrl_stride = 15 so micro-ramp ticks (4, 8, 12) decouple from 3 kHz boundary (15).
    let ultra = build_full_pipeline_ultra_fixture();
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 44_100.0).unwrap();
    let (in_l, in_r) = stereo_test_signal(32, 14);

    // Trigger slew on tank_psi + change on slot 22 (ctrl 2) right before a clean boundary.
    vm.set_control_physical(0, 1_900.0).unwrap();
    vm.set_control_physical(2, 1.0).unwrap();
    let _ = vm.step_frame(in_l[0], in_r[0]).unwrap();
    let psi_tick0 = vm.slab().lite.tank_quad0.current_psi;
    let dry_tick0 = vm.slab().cold().tank_quad_eff[0].dry_mix_eff;
    let we_tick0 = vm.slab().lite.tank_quad0.flags & QUAD_FLAG_WE_HOLD_ACTIVE;

    // Frame 1..3 (sample_tick = 1, 2, 3): neither dry_mix_eff nor psi nor WE moves.
    for tick in 1..4 {
        let _ = vm.step_frame(in_l[tick], in_r[tick]).unwrap();
        assert_eq!(
            vm.slab().cold().tank_quad_eff[0].dry_mix_eff.to_bits(),
            dry_tick0.to_bits()
        );
        assert_eq!(
            vm.slab().lite.tank_quad0.current_psi.to_bits(),
            psi_tick0.to_bits()
        );
        assert_eq!(
            vm.slab().lite.tank_quad0.flags & QUAD_FLAG_WE_HOLD_ACTIVE,
            we_tick0
        );
    }

    // Frame 4..14: on (sample_tick & 3) == 0 (`4, 8, 12`) with ctrl_sub_counter != 0,
    // dry_mix_eff (non-slew Op-T slot) updates while current_psi and WE Schmitt remain held.
    let mut dry_latched_mid = false;
    for tick in 4..15 {
        let _ = vm.step_frame(in_l[tick], in_r[tick]).unwrap();
        let sub = vm.slab().cold().ctrl_bus.as_ref().unwrap().ctrl_sub_counter;
        if tick < 14 {
            assert_ne!(sub, 0);
        } else {
            assert_eq!(sub, 0);
        }
        if vm.slab().cold().tank_quad_eff[0].dry_mix_eff.to_bits() != dry_tick0.to_bits() {
            dry_latched_mid = true;
        }
        assert_eq!(
            vm.slab().lite.tank_quad0.current_psi.to_bits(),
            psi_tick0.to_bits()
        );
        assert_eq!(
            vm.slab().lite.tank_quad0.flags & QUAD_FLAG_WE_HOLD_ACTIVE,
            we_tick0
        );
    }
    assert!(
        dry_latched_mid,
        "Op-T non-slew slot must micro-ramp at 4-frame cadence"
    );

    // Frame 15 (ctrl_sub_counter == 0 before step): 3 kHz boundary advances current_psi slew.
    let _ = vm.step_frame(in_l[15], in_r[15]).unwrap();
    assert_ne!(
        vm.slab().lite.tank_quad0.current_psi.to_bits(),
        psi_tick0.to_bits(),
        "current_psi must update when ctrl_sub_counter == 0"
    );

    // Verify dirty-control switch_tier and retime_sample_rate do NOT advance lfo_phases_q192 or drift_prng_state.
    let (mut ol, mut or) = ([0.0_f32; 32], [0.0_f32; 32]);
    vm.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
    let lfo_before = vm.slab().cold().ctrl_bus.as_ref().unwrap().lfo_phases_q192;
    let prng_before = vm.slab().cold().ctrl_bus.as_ref().unwrap().drift_prng_state;

    vm.set_control_physical(0, 600.0).unwrap();
    assert!(vm.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
    vm.switch_tier(Tier::Pro).unwrap();
    assert!(!vm.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
    assert_eq!(
        vm.slab().cold().ctrl_bus.as_ref().unwrap().lfo_phases_q192,
        lfo_before
    );
    assert_eq!(
        vm.slab().cold().ctrl_bus.as_ref().unwrap().drift_prng_state,
        prng_before
    );

    vm.set_control_physical(0, 1_300.0).unwrap();
    assert!(vm.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
    vm.retime_sample_rate(96_000.0).unwrap();
    assert!(!vm.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
    assert_eq!(
        vm.slab().cold().ctrl_bus.as_ref().unwrap().lfo_phases_q192,
        lfo_before
    );
    assert_eq!(
        vm.slab().cold().ctrl_bus.as_ref().unwrap().drift_prng_state,
        prng_before
    );
}

#[test]
fn test_mid_window_ctrl_dirty_waits_for_3khz_boundary_with_zero_ulp_chunk_parity() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(32, 17);
    let mut vm_step =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 44_100.0).unwrap();
    let mut vm_blk =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 44_100.0).unwrap();

    let (mut step_l, mut step_r) = ([0.0_f32; 32], [0.0_f32; 32]);
    let (mut blk_l, mut blk_r) = ([0.0_f32; 32], [0.0_f32; 32]);

    for t in 0..5 {
        (step_l[t], step_r[t]) = vm_step.step_frame(in_l[t], in_r[t]).unwrap();
    }
    vm_blk
        .process_block(&in_l[..5], &in_r[..5], &mut blk_l[..5], &mut blk_r[..5])
        .unwrap();

    let stage_5 = vm_step
        .slab()
        .cold()
        .ctrl_bus
        .as_ref()
        .unwrap()
        .stage_targets;
    let psi_5 = vm_step.slab().lite.tank_quad0.current_psi;
    let we_5 = vm_step.slab().lite.tank_quad0.flags & QUAD_FLAG_WE_HOLD_ACTIVE;
    let oct_5 = vm_step.slab().cold().kautz_slew.prev_oct;
    assert_eq!(we_5, 0);

    // Mid-window writes at frame 5 (ctrl_sub_counter == 5) must NOT trigger 3 kHz steps early.
    vm_step.set_control_physical(1, 0.0).unwrap();
    vm_step.set_control_physical(0, 1_950.0).unwrap();
    vm_blk.set_control_physical(1, 0.0).unwrap();
    vm_blk.set_control_physical(0, 1_950.0).unwrap();

    for t in 5..15 {
        (step_l[t], step_r[t]) = vm_step.step_frame(in_l[t], in_r[t]).unwrap();
        assert!(vm_step.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
        assert_eq!(
            vm_step
                .slab()
                .cold()
                .ctrl_bus
                .as_ref()
                .unwrap()
                .stage_targets
                .map(f32::to_bits),
            stage_5.map(f32::to_bits),
            "mid-window control must defer until K_ctrl boundary @{t}"
        );
        assert_eq!(
            vm_step.slab().lite.tank_quad0.current_psi.to_bits(),
            psi_5.to_bits()
        );
        assert_eq!(
            vm_step.slab().lite.tank_quad0.flags & QUAD_FLAG_WE_HOLD_ACTIVE,
            we_5
        );
        assert_eq!(
            vm_step.slab().cold().kautz_slew.prev_oct.to_bits(),
            oct_5.to_bits()
        );
    }

    vm_blk
        .process_block(
            &in_l[5..15],
            &in_r[5..15],
            &mut blk_l[5..15],
            &mut blk_r[5..15],
        )
        .unwrap();
    for t in 0..15 {
        assert_eq!(
            step_l[t].to_bits(),
            blk_l[t].to_bits(),
            "pre-boundary L@{t}"
        );
        assert_eq!(
            step_r[t].to_bits(),
            blk_r[t].to_bits(),
            "pre-boundary R@{t}"
        );
    }

    // Frame 15 (ctrl_sub_counter == 0): boundary consumes dirty state and advances slews + WE.
    for t in 15..32 {
        (step_l[t], step_r[t]) = vm_step.step_frame(in_l[t], in_r[t]).unwrap();
        if t == 15 {
            assert!(!vm_step.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
            assert_ne!(
                vm_step.slab().lite.tank_quad0.flags & QUAD_FLAG_WE_HOLD_ACTIVE,
                0
            );
            assert_ne!(
                vm_step.slab().lite.tank_quad0.current_psi.to_bits(),
                psi_5.to_bits()
            );
        }
    }
    vm_blk
        .process_block(&in_l[15..], &in_r[15..], &mut blk_l[15..], &mut blk_r[15..])
        .unwrap();
    for t in 15..32 {
        assert_eq!(
            step_l[t].to_bits(),
            blk_l[t].to_bits(),
            "post-boundary L@{t}"
        );
        assert_eq!(
            step_r[t].to_bits(),
            blk_r[t].to_bits(),
            "post-boundary R@{t}"
        );
    }
}

#[test]
fn test_retained_quad_write_lock_sync_across_tier_downgrade_and_upgrade() {
    let ultra = build_full_pipeline_ultra_fixture();
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 48_000.0).unwrap();

    // Downgrade Ultra -> Lite, toggle host lock ON while active_tank_quads == 1, then restore Ultra.
    vm.switch_tier(Tier::Lite).unwrap();
    vm.set_quad_write_enable(true);
    vm.switch_tier(Tier::Ultra).unwrap();
    let flags_on = [
        vm.slab().lite.tank_quad0.flags,
        vm.slab().full_delta.tank_quad1.flags,
        vm.slab().pro_delta.tank_quad2.flags,
        vm.slab().ultra_delta.tank_quad3.flags,
    ];
    for (q, f) in flags_on.iter().enumerate() {
        assert_ne!(
            f & QUAD_FLAG_HOST_WRITE_LOCK,
            0,
            "Quad {q} lost HOST_WRITE_LOCK after Lite->Ultra upgrade"
        );
    }

    // Downgrade Ultra -> Lite, toggle host lock OFF, then restore Ultra and confirm all 4 cleared.
    vm.switch_tier(Tier::Lite).unwrap();
    vm.set_quad_write_enable(false);
    vm.switch_tier(Tier::Ultra).unwrap();
    let flags_off = [
        vm.slab().lite.tank_quad0.flags,
        vm.slab().full_delta.tank_quad1.flags,
        vm.slab().pro_delta.tank_quad2.flags,
        vm.slab().ultra_delta.tank_quad3.flags,
    ];
    for (q, f) in flags_off.iter().enumerate() {
        assert_eq!(
            f & QUAD_FLAG_HOST_WRITE_LOCK,
            0,
            "Quad {q} kept stale HOST_WRITE_LOCK after Lite->Ultra upgrade"
        );
    }
}

#[test]
fn test_zero_control_ou_drift_scrubbed_cleanly_on_drift_disable_and_offline_bounce() {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.groups[0].in_x_weight = [F32::new(0.4); 4];
    a0.groups[0].svf_g = [F32::new(0.3); 4];
    a0.groups[0].svf_k = [F32::new(0.707); 4];
    a0.groups[0].tone_dir_fp16 = fp16_arr_4([1.0; 4]);
    a0.groups[0].rect_sign_fp16 = fp16_arr_4([1.0; 4]);
    a0.cabinet_fir_lite.fir_l[0] = F32::new(0.8);
    a0.cabinet_fir_lite.fir_r[0] = F32::new(0.8);
    a0.spatial_tank_lite.delay_samples = [F32::new(12.0); 4];
    a0.spatial_tank_lite.g_loop = F32::new(0.2);
    a0.spatial_tank_lite.dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
    a0.varpro_readout_lite.direct_s2_u_gain = [F32::new(0.5), F32::new(0.5)];
    a0.control_bus_lite.lfo_dphi_train = [F32::new(0.0), F32::new(0.0)];
    a0.control_bus_lite.sigma_drift = F32::new(0.5);
    a0.control_bus_lite.rho_drift_train = F32::new(0.99);

    let bin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 120,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        r_ac_default: 0.99,
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
    .unwrap();

    let (in_l, in_r) = stereo_test_signal(128, 19);
    let (mut ol, mut or) = ([0.0_f32; 128], [0.0_f32; 128]);
    let mut vm_a =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, 48_000.0).unwrap();
    let mut vm_b =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, 48_000.0).unwrap();
    vm_a.set_live_analog_drift_enabled(true);
    vm_b.set_live_analog_drift_enabled(true);
    vm_a.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
    vm_b.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();

    let drift_before = vm_a.slab().cold().ctrl_bus.as_ref().unwrap().ou_drift_state;
    assert!(drift_before.iter().any(|&v| v.abs() > 1.0e-5));

    // Disabling live analog drift on 0-control model immediately zeroes OU state & hoisted targets.
    vm_a.set_live_analog_drift_enabled(false);
    let bus_a = vm_a.slab().cold().ctrl_bus.as_ref().unwrap();
    assert_eq!(bus_a.ou_drift_state.map(f32::to_bits), [0; 4]);
    for &slot in &[2_usize, 5, 10, 14] {
        assert_eq!(bus_a.current_stage_params[slot].to_bits(), 0);
        assert_eq!(vm_a.slab().cold().latched_stage_params[slot].to_bits(), 0);
    }
    assert_eq!(vm_a.slab().cold().kautz_slew.prev_oct.to_bits(), 0);

    // Entering offline bounce on 0-control model immediately scrubs stale OU drift before render.
    vm_b.set_offline_bounce(true).unwrap();
    let bus_b = vm_b.slab().cold().ctrl_bus.as_ref().unwrap();
    assert_eq!(bus_b.ou_drift_state.map(f32::to_bits), [0; 4]);
    for &slot in &[2_usize, 5, 10, 14] {
        assert_eq!(bus_b.current_stage_params[slot].to_bits(), 0);
        assert_eq!(vm_b.slab().cold().latched_stage_params[slot].to_bits(), 0);
    }
    assert_eq!(vm_b.slab().cold().kautz_slew.prev_oct.to_bits(), 0);
}
