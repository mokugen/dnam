//! End-to-end integration and `proptest!` verification suite for `dnam-vm` `.dnam-bin` (`"DNBN"`),
//! zero-mutation `REQ-5.6` stripping, `!alloc` vs `alloc` `ControlBus`, `"CP1\0"`, `"CEXT"`, `"TSYN"`,
//! `SO(4)` `Q_buyer` gauge invariance, and `0-ULP` `StarCenter` equivalence.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Instant;

use dnam_kernels::{QUAD_FLAG_WE_HOLD_ACTIVE, update_we_schmitt_flag};
use dnam_vm::control::bus::laws::{eval_crossbar_wet_scale, eval_op_t_we_eff, eval_svf_g_oct};
use dnam_vm::math::exp2_signed_oct_f32;
use dnam_vm::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, CextDispatchWire, ControlBus,
    ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot, Cp1StreamDirEntryWire,
    Cp1StreamHeaderWire, DnamBinBuildInput, DnamBinView, ScalarBackend, TempoSyncEntryWire, Tier,
    VmError, computed_dnam_bin_byte_len, encode_dnam_bin, encode_dnam_bin_into,
    eval_3node_calibration_f64, eval_tsyn_target_value, invert_tsyn_lobatto_anchors,
    strip_dnam_bin_to_tier,
};
use proptest::prelude::*;
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromZeros, IntoBytes};

fn build_rich_ultra_fixture(
    with_cext: bool,
    with_cp1_overflow: bool,
    rot_q_buyer: Option<[[f32; 4]; 4]>,
) -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.crossbar.cell_desc.oversample_factor = U16::new(2);
    let a1 = Arena1FullDeltaWire::new_zeroed();
    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    // Mark Quad 2 ext_f32 to test OR propagation to DBRM::ext_ring_format.
    a2.spatial_tank_pro[0].flags = dnam_vm::bin::TANK_FLAG_EXT_F32;

    // 2 range controls + 1 switch control + optional conditional rule.
    let mut c0 = ControlClampDescriptor::range(100.0, 1_000.0, 550.0, 3, [0.0, -0.5, 0.5]);
    c0.cond_rules[0] = dnam_vm::ConditionalClampRule {
        switch_ctrl_idx: 2,
        switch_opt_idx: 1,
        rule_op: 1,
        bound_val: 750.0,
    };
    let c1 = ControlClampDescriptor::range(-12.0, 12.0, 0.0, 2, [0.0, -0.75, 0.75]);
    let c2 = ControlClampDescriptor::switch(2, 0b11, 0);
    let controls = [c0.to_wire(), c1.to_wire(), c2.to_wire()];

    // Define base main_base[0][0..8], v_base[4], w_ctrl[16][16], u_int_base[4][8].
    let mut main_b0 = [0.4_f32, -0.2, 0.1, 0.3, 0.25, -0.15, 0.05, 0.2];
    let mut main_b1 = [-0.3_f32, 0.2, 0.15, -0.1, 0.1, 0.2, -0.25, 0.05];
    let mut u_int_b = [[0.0_f32; 8]; 4];
    u_int_b[0] = [0.2, -0.1, 0.05, 0.15, 0.12, 0.08, -0.04, 0.02];

    let mut w_ctrl = [[0.0_f32; 16]; 16];
    w_ctrl[0][0] = 0.5;
    w_ctrl[0][1] = -0.25;
    w_ctrl[0][4] = 0.3;
    w_ctrl[13][0] = 0.2;
    w_ctrl[15][4] = 0.4;

    let mut cext = CextDispatchWire::new_zeroed();
    let mut w_ext0 = [0.0_f32; 16];
    w_ext0[0] = -1.2; // slot 16 op_t_we_drive
    w_ext0[4] = 0.25;

    let mut w_ext1 = [0.0_f32; 16];
    w_ext1[0] = 0.35;
    w_ext1[4] = 0.5;

    if let Some(q) = rot_q_buyer {
        let rot4 = |v: &mut [f32]| {
            let orig = [v[0], v[1], v[2], v[3]];
            for i in 0..4 {
                v[i] =
                    q[i][0] * orig[0] + q[i][1] * orig[1] + q[i][2] * orig[2] + q[i][3] * orig[3];
            }
        };
        rot4(&mut main_b0);
        rot4(&mut main_b1);
        for f in 0..4 {
            rot4(&mut u_int_b[f]);
        }
        for row in &mut w_ctrl {
            rot4(row);
        }
        rot4(&mut w_ext0);
        rot4(&mut w_ext1);
    }

    for col in 0..16 {
        cext.w_ctrl_ext[0][col] = F32::new(w_ext0[col]);
        cext.w_ctrl_ext[1][col] = F32::new(w_ext1[col]);
    }
    for f in 0..4 {
        for d in 0..8 {
            a0.control_bus_lite.u_int_base[f][d] = F32::new(u_int_b[f][d]);
        }
    }
    for r in 0..16 {
        for c in 0..16 {
            a0.control_bus_lite.w_ctrl[r][c] = F32::new(w_ctrl[r][c]);
        }
    }
    a3.control_bus_ultra_delta.w_ctrl_ultra[0][0] = F32::new(0.25);

    // Inline CP1: 2 contributors (ctrl 0 and ctrl 1).
    let mut s0_base = [0.0_f32; 13];
    s0_base[..8].copy_from_slice(&main_b0);
    s0_base[8] = 1.0;
    s0_base[9] = 0.5;
    let mut s1_base = [0.0_f32; 13];
    s1_base[..8].copy_from_slice(&main_b1);
    s1_base[8] = 1.0;
    s1_base[9] = -0.4;

    let mut cp1_stream = Vec::new();
    if with_cp1_overflow {
        let hdr = Cp1StreamHeaderWire {
            magic: *b"CP1\0",
            count: U16::new(2),
            flags: 1, // has_ultra
            reserved: 0,
        };
        let e0 = Cp1StreamDirEntryWire {
            control_idx: U16::new(0),
            raw_kl: 1,
            aux: 0xFF,
        };
        let e1 = Cp1StreamDirEntryWire {
            control_idx: U16::new(1),
            raw_kl: 1,
            aux: 0xFF,
        };
        cp1_stream.extend_from_slice(hdr.as_bytes());
        cp1_stream.extend_from_slice(e0.as_bytes());
        cp1_stream.extend_from_slice(e1.as_bytes());
        for &v in s0_base.iter().chain(s1_base.iter()) {
            cp1_stream.extend_from_slice(F32::new(v).as_bytes());
        }
        // 2 * (8*1 + 4) = 24 f32 Ultra deltas.
        let mut s0_ultra = [0.0_f32; 12];
        s0_ultra[4] = 0.3; // z_stage_ultra[0]
        s0_ultra[8] = 0.1;
        let s1_ultra = [0.0_f32; 12];
        for &v in s0_ultra.iter().chain(s1_ultra.iter()) {
            cp1_stream.extend_from_slice(F32::new(v).as_bytes());
        }
    } else {
        a0.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(2);
        a0.control_bus_lite.cp1_dir.entries[0] = [0, 1, 0xFF];
        a0.control_bus_lite.cp1_dir.entries[1] = [1, 1, 0xFF];
        for (i, &v) in s0_base.iter().enumerate() {
            a0.control_bus_lite.cp1_base[i * 4..(i + 1) * 4]
                .copy_from_slice(F32::new(v).as_bytes());
        }
        for (i, &v) in s1_base.iter().enumerate() {
            let off = 52 + i * 4;
            a0.control_bus_lite.cp1_base[off..off + 4].copy_from_slice(F32::new(v).as_bytes());
        }
        a3.control_bus_ultra_delta.cp1_ultra[16..20].copy_from_slice(F32::new(0.3).as_bytes());
        a3.control_bus_ultra_delta.cp1_ultra[32..36].copy_from_slice(F32::new(0.1).as_bytes());
    }

    let tsyn = [
        TempoSyncEntryWire {
            ctrl_idx: U16::new(0),
            switch_ctrl_idx: U16::new(2),
            switch_opt_idx: 0,
            has_switch_cond: 1,
            mode: 0,
            default_subdiv: 4,
            flags: U32::new(0),
            div_ratio: F32::new(1.0),
            bpm_factor: F32::new(1.0),
            lobatto_anchors: [F32::new(100.0), F32::new(400.0), F32::new(1_000.0)],
            declared_min: F32::new(100.0),
            declared_max: F32::new(1_000.0),
        },
        TempoSyncEntryWire {
            ctrl_idx: U16::new(0),
            switch_ctrl_idx: U16::new(2),
            switch_opt_idx: 1,
            has_switch_cond: 1,
            mode: 0,
            default_subdiv: 4,
            flags: U32::new(1), // CLAMP_ONLY
            div_ratio: F32::new(1.0),
            bpm_factor: F32::new(1.0),
            lobatto_anchors: [F32::new(100.0), F32::new(350.0), F32::new(750.0)],
            declared_min: F32::new(100.0),
            declared_max: F32::new(750.0),
        },
    ];

    let input = DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0,
        ext_ring_pow2_frames: 10,
        ext_ring_lines: 4,
        ext_ring_format: 0,
        declared_max_delay_48k: 8_192,
        dc_blocker_pole_5hz: dnam_vm::math::TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 1_500.0,
        r_ac_default: 0.995,
        switch_xfade_samples: 128,
        content_hash_prefix: [9, 8, 7, 6, 5, 4, 3, 2],
        controls: &controls,
        cp1_overflow: &cp1_stream,
        tsyn: &tsyn,
        cext: if with_cext { Some(&cext) } else { None },
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    };
    encode_dnam_bin(&input).unwrap()
}

#[test]
fn test_ultra_roundtrip_strip_to_lite_bind_latency_and_fail_closed_corruption() {
    let a0 = DnamBinBuildInput::default_arena0_lite();
    let lite_input = DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 4_096,
        dc_blocker_pole_5hz: dnam_vm::math::TANK_DC_BLOCKER_48K_F32,
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
    };
    let expected_lite_len = computed_dnam_bin_byte_len(&lite_input).unwrap();
    let mut lite_buf = vec![0u8; expected_lite_len];
    assert_eq!(
        encode_dnam_bin_into(&lite_input, &mut lite_buf).unwrap(),
        expected_lite_len
    );

    let ultra_bytes = build_rich_ultra_fixture(true, true, None);
    let view = DnamBinView::parse(&ultra_bytes).unwrap();
    assert_eq!(view.retained_tier, Tier::Ultra);
    assert_eq!(view.runtime_meta.ext_ring_format, 1);

    // Measure hot zero-copy binding latency over 10,000 iterations.
    let mut samples_ns = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let start = Instant::now();
        let v = DnamBinView::bind_prevalidated(black_box(&ultra_bytes)).unwrap();
        black_box(v.retained_tier);
        samples_ns.push(start.elapsed().as_nanos());
    }
    samples_ns.sort_unstable();
    let median_ns = samples_ns[5_000];
    if cfg!(not(debug_assertions)) {
        assert!(
            median_ns < 50,
            "bind_prevalidated median latency {median_ns} ns exceeded 50 ns budget"
        );
    } else {
        assert!(median_ns < 50_000);
    }

    // Zero-mutation REQ-5.6 tier stripping: Pro -> Full -> Lite.
    let lite_slice = strip_dnam_bin_to_tier(&ultra_bytes, Tier::Lite).unwrap();
    let pro_slice = strip_dnam_bin_to_tier(&ultra_bytes, Tier::Pro).unwrap();
    assert_eq!(&lite_slice[..128], &ultra_bytes[..128]);
    assert_eq!(
        DnamBinView::parse(pro_slice).unwrap().retained_tier,
        Tier::Pro
    );
    assert_eq!(
        DnamBinView::parse(lite_slice).unwrap().retained_tier,
        Tier::Lite
    );
    assert_eq!(
        strip_dnam_bin_to_tier(lite_slice, Tier::Full).unwrap_err(),
        VmError::InvalidTier
    );

    // Fail-closed corruption checks.
    let mut bad_pad = ultra_bytes.clone();
    let tsyn_pad =
        view.runtime_meta.tsyn_offset.get() as usize + view.runtime_meta.tsyn_len.get() as usize;
    bad_pad[tsyn_pad] = 0x5A;
    assert_eq!(
        DnamBinView::parse(&bad_pad).unwrap_err(),
        VmError::NonZeroAlignmentPadding
    );

    let mut bad_prefix_crc = ultra_bytes.clone();
    let a0_off = view.header.arena0_offset.get() as usize;
    bad_prefix_crc[a0_off - 1] ^= 1; // last byte of 1,280 B CEXT (aligned to 64 B)
    assert_eq!(
        DnamBinView::parse(&bad_prefix_crc).unwrap_err(),
        VmError::PrefixCrcMismatch
    );

    let mut bad_crc = ultra_bytes.clone();
    bad_crc[a0_off + 200] ^= 1;
    assert_eq!(
        DnamBinView::parse(&bad_crc).unwrap_err(),
        VmError::TierArenaCrcMismatch
    );

    let mut bad_freeze = ultra_bytes.clone();
    bad_freeze[10] |= 0x20; // evicted freeze bit 5
    assert_eq!(
        DnamBinView::parse(&bad_freeze).unwrap_err(),
        VmError::InvalidHeaderFlagsOrReserved
    );

    let mut bad_len = ultra_bytes;
    bad_len.pop();
    assert_eq!(
        DnamBinView::parse(&bad_len).unwrap_err(),
        VmError::LengthTierMismatch
    );
}

#[test]
fn test_starcenter_0ulp_sweeps_no_alloc_parity_and_so4_qbuyer_invariance() {
    let bytes_inline = build_rich_ultra_fixture(true, false, None);
    let bytes_overflow = build_rich_ultra_fixture(true, true, None);
    let view_inline = DnamBinView::parse(&bytes_inline).unwrap();
    let view_overflow = DnamBinView::parse(&bytes_overflow).unwrap();

    let mut bus_alloc = ControlBus::<dnam_vm::DefaultMathBackend>::from_dnam_bin_owned(
        &view_overflow,
        Tier::Ultra,
        48_000.0,
    )
    .unwrap();
    let mut bus_inline = ControlBus::<dnam_vm::DefaultMathBackend>::from_dnam_bin_owned(
        &view_inline,
        Tier::Ultra,
        48_000.0,
    )
    .unwrap();
    let mut bus_scalar =
        ControlBus::<ScalarBackend>::from_dnam_bin_owned(&view_inline, Tier::Ultra, 48_000.0)
            .unwrap();

    // Borrowed !alloc backing produces bit-identical results.
    let mut descs = [ControlClampDescriptor::empty(); 4];
    let mut raw = [0.0; 4];
    let mut clamped = [0.0; 4];
    let mut norm = [0.0; 4];
    let mut sw = [0u8; 4];
    let mut slots = [Cp1ContributorSlot::zero(); 8];
    let backing = ControlBusBacking::from_slices(
        &mut descs,
        &mut raw,
        &mut clamped,
        &mut norm,
        &mut sw,
        &mut slots,
        view_overflow.tsyn_entries,
    )
    .unwrap();
    let mut bus_no_alloc = ControlBus::<dnam_vm::DefaultMathBackend>::from_dnam_bin(
        &view_overflow,
        Tier::Ultra,
        48_000.0,
        backing,
    )
    .unwrap();

    assert_eq!(bus_alloc.stage_targets, [0.0; 32]);
    assert_eq!(bus_alloc.z_stage(), [0.0; 8]);

    // Sweep controls 200 times and return to StarCenter.
    for step in 0..200 {
        let v0 = 150.0 + ((step * 37) % 800) as f32;
        let v1 = -10.0 + ((step * 13) % 21) as f32;
        bus_alloc.set_controls_batch(&[(0, v0), (1, v1)]).unwrap();
        bus_alloc.tick_control_boundary(None, 0.1);
        bus_inline.set_controls_batch(&[(0, v0), (1, v1)]).unwrap();
        bus_inline.tick_control_boundary(None, 0.1);
        bus_scalar.set_controls_batch(&[(0, v0), (1, v1)]).unwrap();
        bus_scalar.tick_control_boundary(None, 0.1);
        bus_no_alloc
            .set_controls_batch(&[(0, v0), (1, v1)])
            .unwrap();
        bus_no_alloc.tick_control_boundary(None, 0.1);
        assert_eq!(bus_alloc.stage_targets, bus_inline.stage_targets);
        assert_eq!(bus_alloc.stage_targets, bus_no_alloc.stage_targets);
        for (&a, &s) in bus_alloc
            .stage_targets
            .iter()
            .zip(bus_scalar.stage_targets.iter())
        {
            assert!((a - s).abs() < 1.0e-6);
        }
    }

    // Slot 16 immediate snap + Schmitt trigger without ramp lag (and check linear ramp on slot 0).
    let prev_slot0 = bus_alloc.current_stage_params[0];
    bus_alloc
        .set_controls_batch(&[(0, 1_000.0), (1, -12.0)])
        .unwrap();
    bus_alloc.step_frame(None, 0.0);
    assert_eq!(bus_alloc.stage_ramp_deltas[16], 0.0);
    assert_eq!(
        bus_alloc.current_stage_params[16],
        bus_alloc.stage_targets[16]
    );
    assert!(bus_alloc.stage_ramp_deltas[0].abs() > 0.0);
    assert!(
        (bus_alloc.current_stage_params[0] - prev_slot0).abs()
            < (bus_alloc.stage_targets[0] - prev_slot0).abs()
    );
    let we_hold = (update_we_schmitt_flag(0, eval_op_t_we_eff(bus_alloc.current_stage_params[16]))
        & QUAD_FLAG_WE_HOLD_ACTIVE)
        != 0;
    assert!(we_hold);
    assert_eq!(
        update_we_schmitt_flag(QUAD_FLAG_WE_HOLD_ACTIVE, eval_op_t_we_eff(-0.50))
            & QUAD_FLAG_WE_HOLD_ACTIVE,
        QUAD_FLAG_WE_HOLD_ACTIVE
    );
    assert_eq!(
        update_we_schmitt_flag(QUAD_FLAG_WE_HOLD_ACTIVE, eval_op_t_we_eff(-0.40))
            & QUAD_FLAG_WE_HOLD_ACTIVE,
        0
    );
    assert_eq!(
        eval_crossbar_wet_scale(bus_alloc.current_stage_params[15]),
        1.0 + bus_alloc.current_stage_params[15]
    );

    // Return to StarCenter -> exact 0-ULP +0.0 across all targets!
    bus_alloc
        .set_controls_batch(&[(0, 550.0), (1, 0.0), (2, 0.0)])
        .unwrap();
    bus_alloc.tick_control_boundary(None, 0.0);
    assert_eq!(bus_alloc.z_head, [0.0; 4]);
    assert_eq!(bus_alloc.z_stage(), [0.0; 8]);
    for &t in &bus_alloc.stage_targets {
        assert_eq!(t.to_bits(), 0);
    }

    // SO(4) Q_buyer orthogonal rotation invariance on z_head vs 0-ULP invariant z_stage.
    let q_buyer = [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.6, -0.8],
        [0.0, 0.0, 0.8, 0.6],
    ];
    let bytes_rot = build_rich_ultra_fixture(true, true, Some(q_buyer));
    let view_rot = DnamBinView::parse(&bytes_rot).unwrap();
    let mut bus_orig =
        ControlBus::<ScalarBackend>::from_dnam_bin_owned(&view_overflow, Tier::Ultra, 48_000.0)
            .unwrap();
    let mut bus_rot =
        ControlBus::<ScalarBackend>::from_dnam_bin_owned(&view_rot, Tier::Ultra, 48_000.0).unwrap();

    bus_orig
        .set_controls_batch(&[(0, 820.0), (1, 6.0)])
        .unwrap();
    bus_orig.tick_control_boundary(None, 0.0);
    bus_rot.set_controls_batch(&[(0, 820.0), (1, 6.0)]).unwrap();
    bus_rot.tick_control_boundary(None, 0.0);

    // Unrotated z_stage and sensitivity weights are 0-ULP identical!
    for i in 0..8 {
        assert_eq!(
            bus_orig.z_stage()[i].to_bits(),
            bus_rot.z_stage()[i].to_bits()
        );
    }
    let w_svf = [dnam_vm::math::f32_to_fp16(0.25); 8];
    let oct_orig = eval_svf_g_oct(bus_orig.stage_targets[0], &w_svf, &bus_orig.z_stage(), 8);
    let oct_rot = eval_svf_g_oct(bus_rot.stage_targets[0], &w_svf, &bus_rot.z_stage(), 8);
    let g_orig = 0.5 * exp2_signed_oct_f32(oct_orig);
    let g_rot = 0.5 * exp2_signed_oct_f32(oct_rot);
    assert!((g_orig - g_rot).abs() < 1.0e-5);
}

#[test]
fn test_tsyn_conditional_multi_row_20_to_400_bpm() {
    let bytes = build_rich_ultra_fixture(true, false, None);
    let view = DnamBinView::parse(&bytes).unwrap();
    let mut bus =
        ControlBus::<ScalarBackend>::from_dnam_bin_owned(&view, Tier::Full, 44_100.0).unwrap();
    assert_eq!(bus.ctrl_stride, 15);

    // Default unthrottled switch setting -> row 0 [100, 400, 1000] ms.
    bus.set_control_physical(2, 0.0).unwrap();
    assert_eq!(bus.apply_tempo_bpm(120.0), 1);
    bus.tick_control_boundary(None, 0.0);
    let ms0 = eval_tsyn_target_value(&view.tsyn_entries[0], 120.0);
    assert!((ms0 - 500.0).abs() < 1.0e-2);
    let u0 = invert_tsyn_lobatto_anchors([100.0, 400.0, 1_000.0], ms0);
    let rec0 = eval_3node_calibration_f64([100.0, 400.0, 1_000.0], f64::from(u0));
    assert!((rec0 - f64::from(ms0)).abs() < 1.0e-2);

    // Clamped short-delay switch setting -> conditional clamp max = 750 ms + row 1 [100, 350, 750].
    bus.set_control_physical(2, 1.0).unwrap();
    assert_eq!(bus.apply_tempo_bpm(20.0), 1);
    bus.tick_control_boundary(None, 0.0);
    assert!(bus.backing.clamped_values[0] <= 750.0);
}

proptest! {
    #[test]
    fn prop_tsyn_period_and_frequency_roundtrip(bpm in 20.0_f32..400.0, mode in 0_u8..=1) {
        let entry = TempoSyncEntryWire {
            ctrl_idx: U16::new(0),
            switch_ctrl_idx: U16::new(0),
            switch_opt_idx: 0,
            has_switch_cond: 0,
            mode,
            default_subdiv: 4,
            flags: U32::new(0),
            div_ratio: F32::new(1.0),
            bpm_factor: F32::new(1.0),
            lobatto_anchors: [F32::new(0.25), F32::new(400.0), F32::new(1_000.0)],
            declared_min: F32::new(0.25),
            declared_max: F32::new(1_000.0),
        };
        let target = eval_tsyn_target_value(&entry, bpm);
        prop_assert!((0.25..=1_000.0).contains(&target));
        let u = invert_tsyn_lobatto_anchors([0.25, 400.0, 1_000.0], target);
        prop_assert!((-1.0..=1.0).contains(&u));
        let rec = eval_3node_calibration_f64([0.25, 400.0, 1_000.0], f64::from(u)) as f32;
        prop_assert!((rec - target).abs() < 5.0e-2);
    }
}

#[test]
fn test_dnam_bin_parse_bind_and_strip_error_paths() {
    let bytes = build_rich_ultra_fixture(true, true, None);
    let view = DnamBinView::parse(&bytes).unwrap();

    // Corrupt prefix CRC-32C field -> PrefixCrcMismatch in parse.
    let mut bad_prefix_crc = bytes.clone();
    bad_prefix_crc[20] ^= 0xFF;
    assert_eq!(
        DnamBinView::parse(&bad_prefix_crc).unwrap_err(),
        dnam_vm::types::VmError::PrefixCrcMismatch
    );

    // Corrupt arena byte -> TierArenaCrcMismatch.
    let mut bad_arena_crc = bytes.clone();
    let end = bad_arena_crc.len() - 1;
    bad_arena_crc[end] ^= 0xFF;
    assert_eq!(
        DnamBinView::parse(&bad_arena_crc).unwrap_err(),
        dnam_vm::types::VmError::TierArenaCrcMismatch
    );

    // Truncated length -> LengthTierMismatch for both parse and bind_prevalidated.
    assert_eq!(
        DnamBinView::parse(&bytes[..bytes.len() - 64]).unwrap_err(),
        dnam_vm::types::VmError::LengthTierMismatch
    );
    assert_eq!(
        DnamBinView::bind_prevalidated(&bytes[..bytes.len() - 64]).unwrap_err(),
        dnam_vm::types::VmError::LengthTierMismatch
    );
    assert_eq!(
        DnamBinView::bind_prevalidated(&bytes[..64]).unwrap_err(),
        dnam_vm::types::VmError::TruncatedPrefix
    );

    // Strip to Full, then verify stripping stripped Full up to Pro fails with InvalidTier.
    let stripped_full = dnam_vm::bin::strip_dnam_bin_to_tier(&bytes, Tier::Full).unwrap();
    assert_eq!(
        DnamBinView::bind_prevalidated(stripped_full)
            .unwrap()
            .retained_tier,
        Tier::Full
    );
    assert_eq!(
        dnam_vm::bin::strip_dnam_bin_to_tier(stripped_full, Tier::Pro).unwrap_err(),
        dnam_vm::types::VmError::InvalidTier
    );

    // bind_prevalidated fail-fast header rejections (magic, version, invalid tier, length mismatch).
    let mut bad_bind = bytes.clone();
    bad_bind[0] = 0;
    assert_eq!(
        DnamBinView::bind_prevalidated(&bad_bind).unwrap_err(),
        dnam_vm::types::VmError::InvalidMagic
    );
    bad_bind = bytes.clone();
    bad_bind[24] = 130; // unaligned arena0_offset
    assert_eq!(
        DnamBinView::bind_prevalidated(&bad_bind).unwrap_err(),
        dnam_vm::types::VmError::InvalidRegionLayout
    );
    bad_bind = bytes.clone();
    bad_bind[8] = 4; // max_present_tier > 3
    assert_eq!(
        DnamBinView::bind_prevalidated(&bad_bind).unwrap_err(),
        dnam_vm::types::VmError::InvalidTier
    );
    bad_bind = bytes.clone();
    bad_bind[8] = 0; // max_present_tier = 0 while bytes.len() == tier_end_offsets[3]
    assert_eq!(
        DnamBinView::bind_prevalidated(&bad_bind).unwrap_err(),
        dnam_vm::types::VmError::LengthTierMismatch
    );
    let _ = view;
}
