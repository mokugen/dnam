//! M10b-aligned integration contract tests: `384 kHz` live retiming, failure-atomic
//! `192 kHz` `4x` offline bounce rejection, unconditional `Op-P` + phasor stepping across tier
//! transitions, `Op-T` 4-frame micro-ramp vs `3 kHz` `psi` slew cadence, dirty-control LFO/drift
//! isolation, `HOST_WRITE_LOCK` tier sync, and OU drift scrub on disable or offline bounce.

#![forbid(unsafe_code)]

use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16, U32};

use dnam_kernels::QUAD_FLAG_HOST_WRITE_LOCK;
use dnam_vm::bin::{Arena1FullDeltaWire, DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, encode_dnam_bin};
use dnam_vm::graph::ConvolverHighSrMode;
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, f32_to_fp16};
use dnam_vm::{DefaultMathBackend, DynamicSonaVm, SonaVm, Tier, VmError};

use super::helpers::{
    assert_finite_non_silent, build_full_pipeline_ultra_fixture, fp16_arr, make_core_quad,
    make_transport_quad, stereo_test_signal,
};

fn build_zero_crossbar_lite_full_fixture() -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.sona_meta_flags = U32::new(0);
    a0.crossbar.cell_desc.cell_flags = U32::new(0);
    a0.crossbar.cell_desc.tier_flags = U16::new(0x0003);
    a0.crossbar.cell_desc.oversample_factor = U16::new(2);
    a0.crossbar.phasor_freq_hz = [
        F32::new(110.0),
        F32::new(220.0),
        F32::new(330.0),
        F32::new(440.0),
    ];
    a0.crossbar.crossbar_op_gains_fp16 = [U16::new(0); 2];
    a0.crossbar.carrier_basis_w_fp16 = [[U16::new(0); 4]; 3];

    let mut quad = make_core_quad(0);
    quad.svf_g = [F32::new(0.32); 4];
    quad.svf_k_fp16 = fp16_arr([0.707; 4]);
    quad.lru_alpha_rel = [F32::new(0.25); 4];
    quad.lru_alpha_atk = [F32::new(0.99); 4];
    quad.in_mid_weight_fp16 = fp16_arr([0.45, 0.35, 0.25, 0.15]);
    quad.tone_lp_fp16 = fp16_arr([0.5; 4]);
    quad.tone_dir_fp16 = fp16_arr([1.0; 4]);
    quad.rect_sign_fp16 = fp16_arr([1.0; 4]);
    a0.core_quads = [quad; 4];

    a0.conv_fir_lite.fir_l[0] = F32::new(0.8);
    a0.conv_fir_lite.fir_r[0] = F32::new(0.8);
    a0.transport_lite = make_transport_quad(0);
    a0.transport_lite.delay_samples = [F32::new(12.0); 4];
    a0.transport_lite.g_loop = F32::new(0.20);
    a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.4), F32::new(0.4)];
    for idx in 0..16 {
        a0.linear_readout_lite.w_ro_fp16[idx * 12 + 4] = U16::new(f32_to_fp16(0.25));
        a0.linear_readout_lite.w_ro_fp16[idx * 12 + 5] = U16::new(f32_to_fp16(0.25));
    }

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    a1.core_quads = [quad; 2];
    a1.conv_fir_full.fir_l[0] = F32::new(0.2);
    a1.conv_fir_full.fir_r[0] = F32::new(0.2);
    a1.transport_full = make_transport_quad(1);
    a1.transport_full.delay_samples = [F32::new(16.0); 4];
    a1.transport_full.g_loop = F32::new(0.15);
    a1.transport_full.r_dc = F32::new(TANK_DC_BLOCKER_48K_F32);

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: DNBN_FLAG_STEREO_OUT,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 120,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        ro_pole_48k: 0.99,
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
        owned_vm.slab().cold().convolver_fir_ext.mode(),
        ConvolverHighSrMode::CaseA { stride_q: 8 }
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
        vm_fail.slab().cold().convolver_fir_ext.mode(),
        ConvolverHighSrMode::CaseA { stride_q: 4 }
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
fn test_unconditional_opp_and_phasor_stepping_across_tier_switch() {
    let bin = build_zero_crossbar_lite_full_fixture();
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, 48_000.0).unwrap();
    let (in_l, in_r) = stereo_test_signal(64, 13);
    let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);

    // Step in Lite where crossbar gains are zero/inactive; Op-P & phasors still advance unconditionally.
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

    // Trigger slew on tank_psi + change on slot 16 (ctrl 1) and slot 22 (ctrl 2) right before boundary.
    vm.set_controls_batch(&[(0, 1_900.0), (1, 0.0), (2, 1.0)])
        .unwrap();
    let _ = vm.step_frame(in_l[0], in_r[0]).unwrap();
    let psi_tick0 = vm.slab().lite.transport_quad0.current_psi;
    let freeze_tick0 = vm.slab().cold().transport_quad_eff[0].mu_freeze;
    let dry_bits0 = vm.slab().lite.tank_dry_mix_eff_fp16;

    // Frame 1..3 (sample_tick = 1, 2, 3): neither mu_freeze, dry_mix_eff, nor current_psi moves.
    for tick in 1..4 {
        let _ = vm.step_frame(in_l[tick], in_r[tick]).unwrap();
        assert_eq!(
            vm.slab().cold().transport_quad_eff[0].mu_freeze.to_bits(),
            freeze_tick0.to_bits()
        );
        assert_eq!(vm.slab().lite.tank_dry_mix_eff_fp16, dry_bits0);
        assert_eq!(
            vm.slab().lite.transport_quad0.current_psi.to_bits(),
            psi_tick0.to_bits()
        );
    }

    // Frame 4..14: on (sample_tick & 3) == 0 (`4, 8, 12`) with ctrl_sub_counter != 0,
    // mu_freeze & stage dry_mix update while current_psi waits for the 3 kHz boundary.
    let mut non_slew_latched_mid = false;
    for tick in 4..15 {
        let _ = vm.step_frame(in_l[tick], in_r[tick]).unwrap();
        let sub = vm.slab().cold().ctrl_bus.as_ref().unwrap().ctrl_sub_counter;
        if tick < 14 {
            assert_ne!(sub, 0);
        } else {
            assert_eq!(sub, 0);
        }
        if vm.slab().cold().transport_quad_eff[0].mu_freeze.to_bits() != freeze_tick0.to_bits()
            && vm.slab().lite.tank_dry_mix_eff_fp16 != dry_bits0
        {
            non_slew_latched_mid = true;
        }
        assert_eq!(
            vm.slab().lite.transport_quad0.current_psi.to_bits(),
            psi_tick0.to_bits()
        );
    }
    assert!(
        non_slew_latched_mid,
        "Op-T non-slew slots (freeze_mix & dry_mix) must micro-ramp at 4-frame cadence"
    );

    // Frame 15 (ctrl_sub_counter == 0 before step): 3 kHz boundary advances current_psi slew.
    let _ = vm.step_frame(in_l[15], in_r[15]).unwrap();
    assert_ne!(
        vm.slab().lite.transport_quad0.current_psi.to_bits(),
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
    let psi_5 = vm_step.slab().lite.transport_quad0.current_psi;
    let freeze_5 = vm_step.slab().cold().transport_quad_eff[0].mu_freeze;
    let oct_5 = vm_step.slab().cold().kautz_slew.prev_oct;

    // Mid-window writes at frame 5 (ctrl_sub_counter == 5) must defer until the 3 kHz boundary.
    vm_step
        .set_controls_batch(&[(1, 0.0), (0, 1_950.0)])
        .unwrap();
    vm_blk
        .set_controls_batch(&[(1, 0.0), (0, 1_950.0)])
        .unwrap();

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
            vm_step.slab().lite.transport_quad0.current_psi.to_bits(),
            psi_5.to_bits()
        );
        assert_eq!(
            vm_step.slab().cold().transport_quad_eff[0]
                .mu_freeze
                .to_bits(),
            freeze_5.to_bits()
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
        assert_eq!(step_l[t].to_bits(), blk_l[t].to_bits(), "pre L@{t}");
        assert_eq!(step_r[t].to_bits(), blk_r[t].to_bits(), "pre R@{t}");
    }

    // Frame 15 (ctrl_sub_counter == 0): boundary consumes dirty state and advances slews + freeze.
    for t in 15..32 {
        (step_l[t], step_r[t]) = vm_step.step_frame(in_l[t], in_r[t]).unwrap();
        if t == 15 {
            assert!(!vm_step.slab().cold().ctrl_bus.as_ref().unwrap().is_dirty);
            assert_ne!(
                vm_step.slab().lite.transport_quad0.current_psi.to_bits(),
                psi_5.to_bits()
            );
            assert_ne!(
                vm_step.slab().cold().transport_quad_eff[0]
                    .mu_freeze
                    .to_bits(),
                freeze_5.to_bits()
            );
        }
    }
    vm_blk
        .process_block(&in_l[15..], &in_r[15..], &mut blk_l[15..], &mut blk_r[15..])
        .unwrap();
    for t in 15..32 {
        assert_eq!(step_l[t].to_bits(), blk_l[t].to_bits(), "post L@{t}");
        assert_eq!(step_r[t].to_bits(), blk_r[t].to_bits(), "post R@{t}");
    }
}

#[test]
fn test_retained_quad_write_lock_sync_across_tier_downgrade_and_upgrade() {
    let ultra = build_full_pipeline_ultra_fixture();
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 48_000.0).unwrap();

    // Downgrade Ultra -> Lite, toggle host lock ON while active_transport_quads == 1, then restore Ultra.
    vm.switch_tier(Tier::Lite).unwrap();
    vm.set_quad_write_enable(true);
    assert_eq!(vm.slab().cold().transport_quad_eff[0].mu_freeze, 1.0);
    vm.switch_tier(Tier::Ultra).unwrap();
    let flags_on = [
        vm.slab().lite.transport_quad0.flags,
        vm.slab().full_delta.transport_quad1.flags,
        vm.slab().pro_delta.transport_quad2.flags,
        vm.slab().ultra_delta.transport_quad3.flags,
    ];
    for (q, f) in flags_on.iter().enumerate() {
        assert_ne!(
            f & QUAD_FLAG_HOST_WRITE_LOCK,
            0,
            "Quad {q} lost HOST_WRITE_LOCK after Lite->Ultra upgrade"
        );
        assert_eq!(vm.slab().cold().transport_quad_eff[q].mu_freeze, 1.0);
    }

    // Downgrade Ultra -> Lite, toggle host lock OFF immediately without 4-frame lag, then restore Ultra.
    vm.switch_tier(Tier::Lite).unwrap();
    vm.set_op_t_write_enable(false);
    assert!(vm.slab().cold().transport_quad_eff[0].mu_freeze < 1.0);
    vm.switch_tier(Tier::Ultra).unwrap();
    let flags_off = [
        vm.slab().lite.transport_quad0.flags,
        vm.slab().full_delta.transport_quad1.flags,
        vm.slab().pro_delta.transport_quad2.flags,
        vm.slab().ultra_delta.transport_quad3.flags,
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
    a0.core_quads[0] = make_core_quad(0);
    a0.conv_fir_lite.fir_l[0] = F32::new(0.8);
    a0.conv_fir_lite.fir_r[0] = F32::new(0.8);
    a0.transport_lite = make_transport_quad(0);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.5), F32::new(0.5)];
    a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
    a0.control_bus_lite.lfo_dphi_train = [F32::new(0.0), F32::new(0.0)];
    a0.control_bus_lite.sigma_drift = F32::new(0.5);
    a0.control_bus_lite.rho_drift_train = F32::new(0.99);
    a0.control_bus_lite.ou_to_z_stage_fp16 = fp16_arr([1.0, -0.5, 0.25, -0.75]);

    let bin = super::helpers::encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
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

    let bus_before = vm_a.slab().cold().ctrl_bus.as_ref().unwrap();
    assert!(bus_before.ou_drift_state.iter().any(|&v| v.abs() > 1.0e-5));
    assert_ne!(bus_before.eff_z_stage(), bus_before.z_stage());

    // Disabling live analog drift immediately zeroes OU state & eff_z_stage delta.
    vm_a.set_live_analog_drift_enabled(false);
    let bus_a = vm_a.slab().cold().ctrl_bus.as_ref().unwrap();
    assert_eq!(bus_a.ou_drift_state.map(f32::to_bits), [0; 4]);
    assert_eq!(
        bus_a.eff_z_stage().map(f32::to_bits),
        bus_a.z_stage().map(f32::to_bits)
    );
    for &slot in &[2_usize, 5, 10, 14] {
        assert_eq!(bus_a.current_stage_params[slot].to_bits(), 0);
        assert_eq!(vm_a.slab().cold().latched_stage_params[slot].to_bits(), 0);
    }

    // Entering offline bounce immediately scrubs stale OU drift before render.
    vm_b.set_offline_bounce(true).unwrap();
    let bus_b = vm_b.slab().cold().ctrl_bus.as_ref().unwrap();
    assert_eq!(bus_b.ou_drift_state.map(f32::to_bits), [0; 4]);
    assert_eq!(
        bus_b.eff_z_stage().map(f32::to_bits),
        bus_b.z_stage().map(f32::to_bits)
    );
}
