//! M16 integration contracts: Block 0 linearly-implicit `PHS` (`svf_k_nl_fp16`), fail-closed
//! Blocks 1..3 & retired bit-0 rejection, `0 B` `[0; 4]` Schroeder with weight-driven `SO(4)`
//! scattering, `process(1) == process(4) == process(16)` `0-ULP` parity across all delay regimes,
//! and `HOST_WRITE_LOCK` toggle preserving unlocked `(cos_scat, sin_scat)`.

#![forbid(unsafe_code)]

use dnam_kernels::algebraic_unit_rotor_f32;
use dnam_vm::bin::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_STEREO_OUT,
    DnamBinBuildInput, DnamBinView, encode_dnam_bin,
};
use dnam_vm::math::{ScalarBackend, TANK_DC_BLOCKER_48K_F32};
use dnam_vm::{DefaultMathBackend, GraphBackend, SonaVm, Tier, VmError};
use zerocopy::{
    FromZeros,
    byteorder::little_endian::{F32, U16},
};

use super::helpers::{
    build_full_pipeline_ultra_fixture, encode_lite_fixture, fp16_val, le_f32, make_core_quad,
    make_transport_quad, stereo_test_signal,
};

fn run_vm_chunks_backend<B: GraphBackend>(
    bin: &[u8],
    tier: Tier,
    rate_hz: f32,
    in_l: &[f32],
    in_r: &[f32],
    chunk_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut vm = SonaVm::<B>::open_dnam_bin_owned(bin, tier, rate_hz).unwrap();
    let n = in_l.len();
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    let mut off = 0_usize;
    while off < n {
        let step = chunk_len.min(n - off);
        vm.process_block(
            &in_l[off..off + step],
            &in_r[off..off + step],
            &mut out_l[off..off + step],
            &mut out_r[off..off + step],
        )
        .unwrap();
        off += step;
    }
    (out_l, out_r)
}

fn run_vm_chunks(
    bin: &[u8],
    tier: Tier,
    rate_hz: f32,
    in_l: &[f32],
    in_r: &[f32],
    chunk_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    run_vm_chunks_backend::<DefaultMathBackend>(bin, tier, rate_hz, in_l, in_r, chunk_len)
}

#[test]
fn test_m16_block0_phs_chunk_and_cross_isa_parity_and_fail_closed_blocks_1_to_3() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(192, 7);

    // Block 0 quads {0, 4, 6, 8} have svf_k_nl > 0 in make_core_quad; check 0-ULP process(1)==(4)==(16)
    // within each backend and <= 1e-5 cross-ISA parity (DefaultMathBackend FMA vs ScalarBackend non-FMA).
    for &rate in &[48_000.0_f32, 44_100.0_f32] {
        let (l1, r1) = run_vm_chunks(&ultra, Tier::Ultra, rate, &in_l, &in_r, 1);
        let (l4, r4) = run_vm_chunks(&ultra, Tier::Ultra, rate, &in_l, &in_r, 4);
        let (l16, r16) = run_vm_chunks(&ultra, Tier::Ultra, rate, &in_l, &in_r, 16);

        let (ls1, rs1) =
            run_vm_chunks_backend::<ScalarBackend>(&ultra, Tier::Ultra, rate, &in_l, &in_r, 1);
        let (ls16, rs16) =
            run_vm_chunks_backend::<ScalarBackend>(&ultra, Tier::Ultra, rate, &in_l, &in_r, 16);

        let mut max_isa_diff = 0.0_f32;
        for i in 0..192 {
            assert_eq!(l4[i].to_bits(), l1[i].to_bits(), "rate={rate} L4@{i}");
            assert_eq!(r4[i].to_bits(), r1[i].to_bits(), "rate={rate} R4@{i}");
            assert_eq!(l16[i].to_bits(), l1[i].to_bits(), "rate={rate} L16@{i}");
            assert_eq!(r16[i].to_bits(), r1[i].to_bits(), "rate={rate} R16@{i}");
            assert_eq!(ls16[i].to_bits(), ls1[i].to_bits(), "rate={rate} S16L@{i}");
            assert_eq!(rs16[i].to_bits(), rs1[i].to_bits(), "rate={rate} S16R@{i}");
            assert!(ls1[i].is_finite() && rs1[i].is_finite());
            max_isa_diff = max_isa_diff
                .max((ls1[i] - l1[i]).abs())
                .max((rs1[i] - r1[i]).abs());
        }
        assert!(
            max_isa_diff <= 1.0e-5,
            "rate={rate} Scalar vs Default cross-ISA diff={max_isa_diff}"
        );
    }

    // Confirm Block 0 PHS (svf_k_nl > 0) alters audio relative to linear (svf_k_nl == 0).
    let mut phs_a0 = DnamBinBuildInput::default_arena0_lite();
    phs_a0.core_quads[0] = make_core_quad(0);
    phs_a0.core_quads[0].in_mid_weight_fp16 = [fp16_val(2.0); 4];
    phs_a0.core_quads[0].adaa_drive_fp16 = [fp16_val(2.0); 4];
    phs_a0.core_quads[0].svf_k_nl_fp16 = [fp16_val(8.0); 4];
    phs_a0.conv_fir_lite.fir_l[0] = F32::new(0.8);
    phs_a0.conv_fir_lite.fir_r[0] = F32::new(0.8);
    for l in 0..16 {
        phs_a0.linear_readout_lite.w_ro_fp16[l * 12] = fp16_val(0.2);
        phs_a0.linear_readout_lite.w_ro_fp16[l * 12 + 1] = fp16_val(0.2);
        phs_a0.linear_readout_lite.w_ro_fp16[l * 12 + 6] = fp16_val(0.2);
        phs_a0.linear_readout_lite.w_ro_fp16[l * 12 + 7] = fp16_val(0.2);
    }
    phs_a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(1.0);
    let mut lin_a0 = phs_a0;
    lin_a0.core_quads[0].svf_k_nl_fp16 = [U16::new(0); 4];
    let bin_phs = encode_lite_fixture(&phs_a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let bin_lin = encode_lite_fixture(&lin_a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let (pl, pr) = run_vm_chunks(&bin_phs, Tier::Lite, 48_000.0, &in_l, &in_r, 16);
    let (ll, lr) = run_vm_chunks(&bin_lin, Tier::Lite, 48_000.0, &in_l, &in_r, 16);
    let phs_delta: f32 = pl
        .iter()
        .zip(ll.iter())
        .chain(pr.iter().zip(lr.iter()))
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        phs_delta > 1.0e-5,
        "Block 0 PHS delta too small: {phs_delta}"
    );

    // Mutating Block 1..3 quads with nonzero svf_k_nl_fp16 fails closed at open/validation.
    let mut bad_a0 = DnamBinBuildInput::default_arena0_lite();
    bad_a0.core_quads[0] = make_core_quad(0);
    bad_a0.core_quads[1].svf_k_nl_fp16[0] = fp16_val(0.25);
    let err_a0 = encode_dnam_bin(&DnamBinBuildInput {
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
        arena0: &bad_a0,
        arena1: None,
        arena2: None,
        arena3: None,
    })
    .unwrap_err();
    assert_eq!(err_a0, VmError::InvalidArenaHeader);

    // Also verify Arena1/2/3 Blocks 1..3 (local index 1) fail closed when non-zero.
    let mut good_a0 = DnamBinBuildInput::default_arena0_lite();
    good_a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    good_a0.core_quads[0] = make_core_quad(0);
    let mut bad_a1 = Arena1FullDeltaWire::new_zeroed();
    bad_a1.core_quads[1].svf_k_nl_fp16[3] = fp16_val(0.5);
    let a2_ok = Arena2ProDeltaWire::new_zeroed();
    let a3_ok = Arena3UltraDeltaWire::new_zeroed();
    let err_a1 = encode_dnam_bin(&DnamBinBuildInput {
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
        arena0: &good_a0,
        arena1: Some(&bad_a1),
        arena2: Some(&a2_ok),
        arena3: Some(&a3_ok),
    })
    .unwrap_err();
    assert_eq!(err_a1, VmError::InvalidArenaHeader);
}

#[test]
fn test_m16_op_t_so4_zero_byte_regimes_retired_bit0_and_host_write_lock() {
    // 1) Retired TRANSPORT_FLAG_SCHROEDER_ENABLE (bit 0) fails closed.
    let mut bad_bit0 = DnamBinBuildInput::default_arena0_lite();
    bad_bit0.transport_lite = make_transport_quad(0);
    bad_bit0.transport_lite.flags |= 0x01;
    let bin_ok_template = encode_lite_fixture(
        &{
            let mut a0 = DnamBinBuildInput::default_arena0_lite();
            a0.transport_lite = make_transport_quad(0);
            a0
        },
        DNBN_FLAG_STEREO_OUT,
        0.99,
        &[],
        None,
    );
    assert!(DnamBinView::parse(&bin_ok_template).is_ok());
    let err_bit0 = encode_dnam_bin(&DnamBinBuildInput {
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
        arena0: &bad_bit0,
        arena1: None,
        arena2: None,
        arena3: None,
    })
    .unwrap_err();
    assert_eq!(err_bit0, VmError::InvalidArenaHeader);

    // 2) allpass_delays == [0; 4] allocates 0 B cold RAM while still executing SO(4) scattering,
    // and preserves process(1) == process(4) == process(16) across [0; 4], 1..=3, and >= 4.
    let (in_l, in_r) = stereo_test_signal(160, 11);
    for (idx, ap_delays) in [[0_u8; 4], [1, 2, 3, 0], [11, 17, 23, 31]]
        .into_iter()
        .enumerate()
    {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.core_quads[0] = make_core_quad(0);
        a0.conv_fir_lite.fir_l[0] = F32::new(0.8);
        a0.conv_fir_lite.fir_r[0] = F32::new(0.8);
        a0.linear_readout_lite.direct_s2_u_gain = le_f32([0.6, 0.6]);
        a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(0.5);
        a0.transport_lite = make_transport_quad(0);
        a0.transport_lite.allpass_delays = ap_delays;
        a0.transport_lite.scat_psi_fp16 = fp16_val(0.65);

        let bin = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
        let vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Lite, 48_000.0).unwrap();
        if idx == 0 {
            let sch = &vm.slab().cold().transport_schroeder[0];
            assert_eq!((sch.cap_per_lane(), sch.backing_f32_capacity()), (0, 0));
        }

        let (l1, r1) = run_vm_chunks(&bin, Tier::Lite, 48_000.0, &in_l, &in_r, 1);
        let (l4, r4) = run_vm_chunks(&bin, Tier::Lite, 48_000.0, &in_l, &in_r, 4);
        let (l16, r16) = run_vm_chunks(&bin, Tier::Lite, 48_000.0, &in_l, &in_r, 16);
        for i in 0..160 {
            assert_eq!(l4[i].to_bits(), l1[i].to_bits(), "regime={idx} L4@{i}");
            assert_eq!(r4[i].to_bits(), r1[i].to_bits(), "regime={idx} R4@{i}");
            assert_eq!(l16[i].to_bits(), l1[i].to_bits(), "regime={idx} L16@{i}");
            assert_eq!(r16[i].to_bits(), r1[i].to_bits(), "regime={idx} R16@{i}");
        }

        // Verify SO(4) scattering weight alters output even when allpass_delays == [0; 4].
        if idx == 0 {
            a0.transport_lite.scat_psi_fp16 = fp16_val(0.0);
            let bin_id = encode_lite_fixture(&a0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
            let (lid, rid) = run_vm_chunks(&bin_id, Tier::Lite, 48_000.0, &in_l, &in_r, 16);
            let diff: f32 = l16
                .iter()
                .zip(lid.iter())
                .chain(r16.iter().zip(rid.iter()))
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                diff > 1.0e-5,
                "scat_psi had no effect on zero-delay tank: diff={diff}"
            );
        }
    }

    // 3) HOST_WRITE_LOCK engage (locked=true) -> release (locked=false) without an intervening
    // 3 kHz control tick preserves unlocked (cos_scat, sin_scat).
    let mut a0_lock = DnamBinBuildInput::default_arena0_lite();
    a0_lock.core_quads[0] = make_core_quad(0);
    a0_lock.transport_lite = make_transport_quad(0);
    a0_lock.transport_lite.freeze_mix_fp16 = fp16_val(0.25);
    a0_lock.transport_lite.scat_psi_fp16 = fp16_val(0.8);
    let bin_lock = encode_lite_fixture(&a0_lock, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let mut vm_locked =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_lock, Tier::Lite, 48_000.0).unwrap();
    vm_locked.step_control_tick_3khz();
    vm_locked.set_op_t_write_enable(true);
    let _ = vm_locked.step_frame(0.2, -0.2).unwrap();
    vm_locked.set_op_t_write_enable(false);

    let eff = &vm_locked.slab().cold().transport_quad_eff[0];
    let meta = &vm_locked.slab().cold().transport_quad_meta[0];
    let (ec, es) = algebraic_unit_rotor_f32((1.0 - meta.base_freeze_mix) * meta.base_scat_psi);
    assert_eq!(eff.cos_scat.to_bits(), ec.to_bits());
    assert_eq!(eff.sin_scat.to_bits(), es.to_bits());
    assert!((eff.cos_scat - 1.0).abs() > 1.0e-3 && eff.sin_scat.abs() > 0.1);
}
