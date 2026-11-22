//! Integration proof of backing pointer stability (`0` reallocation, `0` drops), fail-closed
//! capacity enforcement, `!alloc` borrowed slab hot-swap, and `0-ULP` audio parity for `SonaVm`
//! & `DynamicSonaVm` in-place `.dnam-bin` reload (`ref[container.dnam.vm-binary]`).

#![forbid(unsafe_code)]

use dnam_vm::bin::TRANSPORT_FLAG_EXT_F32;
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, f32_to_fp16};
use dnam_vm::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, ControlBusBacking,
    ControlClampDescriptor, Cp1ContributorSlot, DEFAULT_ANTI_ZIPPER_POLE, DnamBinBuildInput,
    DnamBinView, DynamicSonaVm, SonaColdTail, SonaVm, SonaVmBacking, TempoSyncEntryWire, Tier,
    VmError, VmReloadHeadroom, encode_dnam_bin,
};
use zerocopy::{
    FromZeros, IntoBytes,
    byteorder::little_endian::{F32, U16},
};

#[derive(Debug, PartialEq, Eq)]
struct BackingSignature {
    cold_ptr: usize,
    ctrl_addrs: (usize, usize, usize),
    fir_addr: usize,
    ext_addrs: [(usize, usize); 4],
    sch_addrs: [usize; 4],
}

fn capture_backing_signature<B: dnam_vm::GraphBackend>(vm: &SonaVm<'_, B>) -> BackingSignature {
    let cold = vm.slab().cold();
    BackingSignature {
        cold_ptr: core::ptr::from_ref(cold) as usize,
        ctrl_addrs: cold
            .ctrl_bus
            .as_ref()
            .expect("hydrated ctrl_bus")
            .backing
            .backing_storage_addrs(),
        fir_addr: cold.convolver_fir_ext.backing_storage_addr(),
        ext_addrs: core::array::from_fn(|q| cold.transport_ext[q].backing_storage_addrs()),
        sch_addrs: core::array::from_fn(|q| cold.transport_schroeder[q].backing_storage_addr()),
    }
}

fn make_tsyn_row(ctrl_idx: u16) -> TempoSyncEntryWire {
    let mut row = TempoSyncEntryWire::new_zeroed();
    row.ctrl_idx = U16::new(ctrl_idx);
    row.div_ratio = F32::new(1.0);
    row.bpm_factor = F32::new(60_000.0);
    row.lobatto_anchors = [10.0, 50.0, 100.0].map(F32::new);
    row.declared_min = F32::new(10.0);
    row.declared_max = F32::new(100.0);
    row
}

fn build_lite_inline_model() -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.crossbar_op_gains_fp16 = [U16::new(f32_to_fp16(0.5)), U16::new(f32_to_fp16(0.25))];
    a0.core_quads[0].svf_g = [0.22; 4].map(F32::new);
    a0.core_quads[0].tone_dir_fp16 = [U16::new(f32_to_fp16(0.8)); 4];
    a0.linear_readout_lite.w_ro_fp16[0] = U16::new(f32_to_fp16(0.4));
    a0.linear_readout_lite.w_ro_fp16[1] = U16::new(f32_to_fp16(0.3));
    a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.5));
    a0.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(1);
    a0.control_bus_lite.cp1_dir.entries[0] = [0, 0x01, 0];
    a0.control_bus_lite.cp1_base[..4].copy_from_slice(F32::new(0.25).as_bytes());
    a0.control_bus_lite.w_ctrl[0][0] = F32::new(0.15);
    a0.transport_lite.delay_samples = [24.0, 32.0, 40.0, 48.0].map(F32::new);
    a0.transport_lite.g_loop = F32::new(0.35);
    let controls = [ControlClampDescriptor::range(0.0, 1.0, 0.25, 1, [0.0; 3]).to_wire()];
    let input = DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0x01,
        ext_ring_pow2_frames: 7,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 120,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        ro_pole_48k: DEFAULT_ANTI_ZIPPER_POLE,
        switch_xfade_samples: 0,
        content_hash_prefix: [1; 8],
        controls: &controls,
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: None,
        arena2: None,
        arena3: None,
    };
    encode_dnam_bin(&input).unwrap()
}

fn build_ultra_ext_model(want_f32: bool, num_controls: usize, with_tsyn: bool) -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.crossbar.crossbar_op_gains_fp16 = [U16::new(f32_to_fp16(0.35)), U16::new(f32_to_fp16(0.45))];
    for q in 0..4 {
        a0.core_quads[q].svf_g = [0.31; 4].map(F32::new);
        a0.core_quads[q].tone_lp_fp16 = [U16::new(f32_to_fp16(0.6)); 4];
        a0.core_quads[q].tone_dir_fp16 = [U16::new(f32_to_fp16(0.4)); 4];
        a0.core_quads[q].adaa_drive_fp16 = [U16::new(f32_to_fp16(1.2)); 4];
    }
    a0.linear_readout_lite.w_ro_fp16[0] = U16::new(f32_to_fp16(-0.25));
    a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.25));
    a0.control_bus_lite.lfo_skew_fp16 = [U16::new(f32_to_fp16(0.5)), U16::new(0)];
    a0.transport_lite.delay_samples = [160.0, 176.0, 192.0, 200.0].map(F32::new);
    a0.transport_lite.g_loop = F32::new(0.5);
    a0.transport_lite.allpass_delays = [0, 17, 23, 31];
    a0.transport_lite.allpass_g_fp16 = [U16::new(f32_to_fp16(0.4)); 4];
    a0.transport_lite.flags = if want_f32 { TRANSPORT_FLAG_EXT_F32 } else { 0 };

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    a1.transport_full = a0.transport_lite;
    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    a2.transport_pro = [a0.transport_lite; 2];
    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    a3.transport_ultra = [a0.transport_lite; 2];

    let controls: Vec<_> = (0..num_controls)
        .map(|i| ControlClampDescriptor::range(0.0, 100.0, 20.0 + i as f32, 1, [0.0; 3]).to_wire())
        .collect();
    let tsyn_vec = if with_tsyn {
        vec![make_tsyn_row(0)]
    } else {
        Vec::new()
    };
    let input = DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0x01,
        ext_ring_pow2_frames: 9,
        ext_ring_lines: 4,
        ext_ring_format: u8::from(want_f32),
        declared_max_delay_48k: 220,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        ro_pole_48k: DEFAULT_ANTI_ZIPPER_POLE,
        switch_xfade_samples: 0,
        content_hash_prefix: [2; 8],
        controls: &controls,
        cp1_overflow: &[],
        tsyn: &tsyn_vec,
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    };
    encode_dnam_bin(&input).unwrap()
}

fn render_stereo_program<F: FnMut(&[f32], &[f32], &mut [f32], &mut [f32])>(
    mut f: F,
) -> (Vec<f32>, Vec<f32>) {
    let n = 128;
    let in_l: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.19).sin()).collect();
    let in_r: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.23).cos()).collect();
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];
    f(&in_l, &in_r, &mut out_l, &mut out_r);
    (out_l, out_r)
}

/// verify[container.dnam.vm-binary]
#[test]
fn test_owned_headroom_reload_zero_alloc_zero_drop_and_0ulp_parity() {
    let bin_lite = build_lite_inline_model();
    let bin_ultra_fp16 = build_ultra_ext_model(false, 4, true);
    let bin_ultra_f32 = build_ultra_ext_model(true, 3, false);

    let view_lite = DnamBinView::parse(&bin_lite).unwrap();
    let view_u_fp16 = DnamBinView::parse(&bin_ultra_fp16).unwrap();
    let view_u_f32 = DnamBinView::parse(&bin_ultra_f32).unwrap();

    // Fail-closed capacity test: exact-sized Lite VM rejects larger Ultra/multi-control model without moving backing pointers.
    let mut vm_small =
        SonaVm::<dnam_vm::DefaultMathBackend>::open_dnam_bin_owned(&bin_lite, Tier::Lite, 48_000.0)
            .unwrap();
    let small_sig = capture_backing_signature(&vm_small);
    let err = vm_small
        .reload_from_view_in_place(&view_u_fp16, Tier::Ultra, 48_000.0, false)
        .unwrap_err();
    assert_eq!(err, VmError::InsufficientBackingCapacity);
    assert_eq!(vm_small.telemetry().active_tier, Tier::Lite);
    assert_eq!(capture_backing_signature(&vm_small), small_sig);

    // Also verify undersized ext-ring headroom fails closed at 4x bounce (`1024 > 512`) without disturbing backing pointers.
    let small_ring_headroom =
        VmReloadHeadroom::for_live_engine().with_transport_headroom(512, 512, 512);
    let mut vm_small_ring =
        SonaVm::<dnam_vm::DefaultMathBackend>::open_dnam_bin_with_headroom_owned(
            &bin_lite,
            Tier::Lite,
            48_000.0,
            small_ring_headroom,
        )
        .unwrap();
    let small_ring_sig = capture_backing_signature(&vm_small_ring);
    let ring_err = vm_small_ring
        .reload_from_view_in_place(&view_u_f32, Tier::Ultra, 48_000.0, true)
        .unwrap_err();
    assert_eq!(ring_err, VmError::InsufficientBackingCapacity);
    assert_eq!(capture_backing_signature(&vm_small_ring), small_ring_sig);

    let headroom = VmReloadHeadroom::for_live_engine().with_transport_headroom(1_024, 1_024, 512);
    let mut vm = SonaVm::<dnam_vm::DefaultMathBackend>::open_dnam_bin_with_headroom_owned(
        &bin_lite,
        Tier::Lite,
        48_000.0,
        headroom,
    )
    .unwrap();
    let expected_sig = capture_backing_signature(&vm);
    let _ = render_stereo_program(|il, ir, ol, or| vm.process_block(il, ir, ol, or).unwrap());

    // Reload Lite -> Ultra (fp16 ext + Schroeder + TSYN) @ 96 kHz offline=false preserving every backing address.
    vm.reload_from_view_in_place(&view_u_fp16, Tier::Ultra, 96_000.0, false)
        .unwrap();
    assert_eq!(capture_backing_signature(&vm), expected_sig);
    let mut ref_u_fp16 = SonaVm::<dnam_vm::DefaultMathBackend>::from_view_owned(
        &view_u_fp16,
        Tier::Ultra,
        96_000.0,
        false,
    )
    .unwrap();
    let (ol_r, or_r) =
        render_stereo_program(|il, ir, ol, or| vm.process_block(il, ir, ol, or).unwrap());
    let (ol_f, or_f) =
        render_stereo_program(|il, ir, ol, or| ref_u_fp16.process_block(il, ir, ol, or).unwrap());
    assert!(
        ol_r.iter()
            .zip(ol_f.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        or_r.iter()
            .zip(or_f.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // Reload Ultra(fp16) -> Ultra(f32) @ 48 kHz offline_bounce=true preserving every backing address.
    vm.reload_from_view_in_place(&view_u_f32, Tier::Ultra, 48_000.0, true)
        .unwrap();
    assert_eq!(capture_backing_signature(&vm), expected_sig);
    let mut ref_u_f32 = SonaVm::<dnam_vm::DefaultMathBackend>::from_view_owned(
        &view_u_f32,
        Tier::Ultra,
        48_000.0,
        true,
    )
    .unwrap();
    let (ol_r2, or_r2) =
        render_stereo_program(|il, ir, ol, or| vm.process_block(il, ir, ol, or).unwrap());
    let (ol_f2, or_f2) =
        render_stereo_program(|il, ir, ol, or| ref_u_f32.process_block(il, ir, ol, or).unwrap());
    assert!(
        ol_r2
            .iter()
            .zip(ol_f2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        or_r2
            .iter()
            .zip(or_f2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // Reload Ultra(f32) -> Lite(inline) @ 48 kHz live preserving every backing address.
    vm.reload_from_view_in_place(&view_lite, Tier::Lite, 48_000.0, false)
        .unwrap();
    assert_eq!(capture_backing_signature(&vm), expected_sig);
    let mut ref_lite = SonaVm::<dnam_vm::DefaultMathBackend>::from_view_owned(
        &view_lite,
        Tier::Lite,
        48_000.0,
        false,
    )
    .unwrap();
    let (ol_r3, or_r3) =
        render_stereo_program(|il, ir, ol, or| vm.process_block(il, ir, ol, or).unwrap());
    let (ol_f3, or_f3) =
        render_stereo_program(|il, ir, ol, or| ref_lite.process_block(il, ir, ol, or).unwrap());
    assert!(
        ol_r3
            .iter()
            .zip(ol_f3.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        or_r3
            .iter()
            .zip(or_f3.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // DynamicSonaVm also reloads in place without failing closed.
    let mut dyn_vm =
        DynamicSonaVm::open_dnam_bin_with_headroom(&bin_lite, Tier::Lite, 48_000.0, headroom)
            .unwrap();
    dyn_vm
        .reload_from_view_in_place(&view_u_fp16, Tier::Ultra, 48_000.0, false)
        .unwrap();
    assert_eq!(dyn_vm.telemetry().active_tier, Tier::Ultra);
}

#[test]
fn test_no_alloc_borrowed_slab_reload_zero_alloc_and_0ulp_parity() {
    let bin_lite = build_lite_inline_model();
    let bin_ultra_f32 = build_ultra_ext_model(true, 3, true);
    let view_lite = DnamBinView::parse(&bin_lite).unwrap();
    let view_u_f32 = DnamBinView::parse(&bin_ultra_f32).unwrap();

    let mut d = [ControlClampDescriptor::empty(); 8];
    let mut r = [0.0; 8];
    let mut c = [0.0; 8];
    let mut n = [0.0; 8];
    let mut sw = [0u8; 8];
    let mut slots = [Cp1ContributorSlot::zero(); 8];
    let mut tsyn = [TempoSyncEntryWire::new_zeroed(); 4];
    let ctrl_backing = ControlBusBacking::from_mut_slices(
        &mut d, &mut r, &mut c, &mut n, &mut sw, &mut slots, &mut tsyn,
    )
    .unwrap();
    let mut fir_ext = vec![0.0_f32; dnam_vm::EXT_FIR_STORAGE_F32_LEN];
    let (mut fp0, mut fp1, mut fp2, mut fp3) = (
        vec![0u16; 2_048],
        vec![0u16; 2_048],
        vec![0u16; 2_048],
        vec![0u16; 2_048],
    );
    let (mut f0, mut f1, mut f2, mut f3) = (
        vec![0.0f32; 2_048],
        vec![0.0f32; 2_048],
        vec![0.0f32; 2_048],
        vec![0.0f32; 2_048],
    );
    let (mut s0, mut s1, mut s2, mut s3) = (
        vec![0.0f32; 256],
        vec![0.0f32; 256],
        vec![0.0f32; 256],
        vec![0.0f32; 256],
    );
    let mut cold = [SonaColdTail::<dnam_vm::DefaultMathBackend>::empty()];
    let backing = SonaVmBacking {
        control_bus: ctrl_backing,
        convolver_fir_ext: &mut fir_ext,
        transport_ext_fp16: [&mut fp0, &mut fp1, &mut fp2, &mut fp3],
        transport_ext_f32: [&mut f0, &mut f1, &mut f2, &mut f3],
        transport_schroeder: [&mut s0, &mut s1, &mut s2, &mut s3],
        cold_tail: &mut cold,
    };

    let mut vm = SonaVm::open_dnam_bin(&bin_lite, Tier::Lite, 48_000.0, backing).unwrap();
    let borrowed_sig = capture_backing_signature(&vm);
    vm.reload_from_view_in_place(&view_u_f32, Tier::Ultra, 48_000.0, false)
        .unwrap();
    assert_eq!(capture_backing_signature(&vm), borrowed_sig);
    let mut ref_u = SonaVm::<dnam_vm::DefaultMathBackend>::from_view_owned(
        &view_u_f32,
        Tier::Ultra,
        48_000.0,
        false,
    )
    .unwrap();
    let (ol_a, or_a) =
        render_stereo_program(|il, ir, ol, or| vm.process_block(il, ir, ol, or).unwrap());
    let (ol_b, or_b) =
        render_stereo_program(|il, ir, ol, or| ref_u.process_block(il, ir, ol, or).unwrap());
    assert!(
        ol_a.iter()
            .zip(ol_b.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        or_a.iter()
            .zip(or_b.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    vm.reload_from_view_in_place(&view_lite, Tier::Lite, 48_000.0, false)
        .unwrap();
    assert_eq!(capture_backing_signature(&vm), borrowed_sig);
}
