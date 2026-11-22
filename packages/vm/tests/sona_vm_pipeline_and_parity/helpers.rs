//! Deterministic SONA v3.0 fixture builders and `!alloc` stack runners for `dnam-vm` integration suites.

#![forbid(unsafe_code)]

use dnam_kernels::pack_wf_mix_and_rect_sign_fp16;
use dnam_vm::bin::{
    DNBN_FLAG_STEREO_OUT, PIPE_CANONICAL_V3_SONA, SONA_META_CORE_MODS_PRESENT,
    SONA_META_OS_EXT_PRESENT, SONA_META_SENSORY_QUAD, SONA_META_TRANSPORT_LFO,
    SONA_META_TRANSPORT_TAP_GAINS, TRANSPORT_FLAG_EXT_F32,
};
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, f32_to_fp16};
use dnam_vm::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    CextDispatchWire, ConditionalClampRule, ControlBusBacking, ControlClampDescriptor,
    CoreQuadWire, Cp1ContributorSlot, DefaultMathBackend, DnamBinBuildInput, KautzSoaBlockWire,
    OpNPreBaseWire, SonaColdTail, SonaCrossbarBlockWire, SonaInterBlockTilePairWire, SonaVm,
    SonaVmBacking, TempoSyncEntryWire, Tier, TransportQuadWire, encode_dnam_bin,
};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromZeros, IntoBytes};

pub(crate) fn le_f32<const N: usize>(vals: [f32; N]) -> [F32; N] {
    core::array::from_fn(|i| F32::new(vals[i]))
}

pub(crate) fn fp16_arr<const N: usize>(vals: [f32; N]) -> [U16; N] {
    core::array::from_fn(|i| U16::new(f32_to_fp16(vals[i])))
}

pub(crate) fn fp16_val(v: f32) -> U16 {
    U16::new(f32_to_fp16(v))
}

pub(crate) fn assert_finite_non_silent(l: &[f32], r: &[f32], label: &str) {
    assert!(
        l.iter().chain(r.iter()).all(|v| v.is_finite()),
        "non-finite output in {label}"
    );
    let energy: f64 = l
        .iter()
        .chain(r.iter())
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    assert!(
        energy > 1.0e-6,
        "unexpectedly silent output in {label}: energy={energy}"
    );
}

pub(crate) fn stereo_test_signal(len: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut in_l = Vec::with_capacity(len);
    let mut in_r = Vec::with_capacity(len);
    let phase_off = (seed as f32) * 0.13;
    for i in 0..len {
        let t = (i as f32) / 48_000.0;
        let l = 0.35 * (2.0 * core::f32::consts::PI * 220.0 * t + phase_off).sin()
            + 0.12 * (2.0 * core::f32::consts::PI * 660.0 * t).cos();
        let r = 0.32 * (2.0 * core::f32::consts::PI * 277.18 * t - phase_off).sin()
            + 0.14 * (2.0 * core::f32::consts::PI * 554.37 * t).cos();
        in_l.push(l);
        in_r.push(r);
    }
    (in_l, in_r)
}

pub(crate) fn encode_lite_fixture(
    arena0: &Arena0LiteWire,
    header_flags: u16,
    ro_pole_48k: f32,
    controls: &[dnam_vm::bin::ControlClampDescriptorWire],
    cext: Option<&CextDispatchWire>,
) -> Vec<u8> {
    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        ro_pole_48k,
        switch_xfade_samples: 0,
        content_hash_prefix: [0; 8],
        controls,
        cp1_overflow: &[],
        tsyn: &[],
        cext,
        arena0,
        arena1: None,
        arena2: None,
        arena3: None,
    })
    .expect("encode Lite fixture")
}

pub(crate) fn make_core_quad(q: usize) -> CoreQuadWire {
    let s = 1.0 + 0.08 * (q as f32);
    let rs = if (q & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
    CoreQuadWire {
        svf_g: le_f32([0.12 * s, 0.24 * s, 0.42 * s, 0.68 * s]),
        lru_alpha_rel: le_f32([0.82, 0.85, 0.88, 0.91]),
        lru_alpha_atk: le_f32([0.18, 0.22, 0.26, 0.30]),
        svf_k_fp16: fp16_arr([0.65, 0.72, 0.80, 0.90]),
        lru_kappa_fp16: fp16_arr([0.35, 0.40, 0.45, 0.50]),
        adaa_drive_fp16: fp16_arr([0.30, 0.35, 0.28, 0.32]),
        lru_bias_drive_fp16: fp16_arr([0.04 * rs, -0.03 * rs, 0.05 * rs, -0.02 * rs]),
        tone_lp_fp16: fp16_arr([0.35, 0.30, 0.25, 0.20]),
        tone_bp_fp16: fp16_arr([0.20, 0.25, 0.20, 0.15]),
        tone_hp_fp16: fp16_arr([0.15, 0.15, 0.20, 0.25]),
        tone_ap_fp16: fp16_arr([0.10, 0.12, 0.15, 0.18]),
        tone_dir_fp16: fp16_arr([0.20, 0.18, 0.20, 0.22]),
        sag_depth_fp16: fp16_arr([0.06, 0.05, 0.07, 0.04]),
        wavefolder_mix_fp16: [
            U16::new(pack_wf_mix_and_rect_sign_fp16(0.04, rs)),
            U16::new(pack_wf_mix_and_rect_sign_fp16(0.03, -rs)),
            U16::new(pack_wf_mix_and_rect_sign_fp16(0.05, rs)),
            U16::new(pack_wf_mix_and_rect_sign_fp16(0.04, -rs)),
        ],
        svf_k_nl_fp16: if matches!(q, 0 | 4 | 6 | 8) {
            fp16_arr([1.25, 0.85, 1.50, 0.60])
        } else {
            [U16::new(0); 4]
        },
        in_mid_weight_fp16: fp16_arr([0.22, -0.18, 0.25, -0.15]),
        in_side_weight_fp16: fp16_arr([0.08, 0.06, -0.07, 0.05]),
        lru_stereo_link_fp16: fp16_arr([0.15, 0.20, 0.10, 0.25]),
        lru_duhem_gamma_fp16: fp16_arr([0.50, 0.75, 0.60, 0.80]),
        lru_signed_mix_fp16: fp16_arr([0.60, 0.55, 0.65, 0.50]),
        glu_mix_fp16: fp16_arr([0.12, -0.08, 0.10, -0.06]),
    }
}

fn make_tile_wire(t: usize) -> SonaInterBlockTilePairWire {
    let mut tile = SonaInterBlockTilePairWire::new_zeroed();
    let sign = if (t & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
    let mag = 0.035 + 0.002 * ((t % 7) as f32);
    for s in 0..4 {
        for d in 0..4 {
            let v_odd = if s == d {
                sign * mag
            } else if ((s + 1) & 3) == d {
                -0.5 * sign * mag
            } else {
                0.0
            };
            tile.w_odd_4x4[s][d] = F32::new(v_odd);
            tile.w_even_4x4_fp16[s][d] = fp16_val(0.4 * v_odd);
            tile.w_lru_4x4_fp16[s][d] = fp16_val(0.3 * v_odd);
        }
    }
    tile
}

fn make_crossbar_block(tier_idx: usize) -> SonaCrossbarBlockWire {
    let mut blk = SonaCrossbarBlockWire::new_zeroed();
    let gain_scale = 0.06 + 0.01 * (tier_idx as f32);
    for q in 0..10 {
        for l in 0..4 {
            let s = if ((q + l) & 1) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            blk.op_p_to_core_fp16[q][l] = fp16_val(s * gain_scale);
            blk.phasor_to_core_fp16[q][l] = fp16_val(-s * gain_scale);
        }
    }
    for i in 0..16 {
        let s = if (i & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
        blk.opt_to_lr_fp16[i][0] = fp16_val(0.03 * s);
        blk.opt_to_lr_fp16[i][1] = fp16_val(-0.03 * s);
    }
    for l in 0..4 {
        blk.phasor_stereo_phase_fp16[l] = fp16_val(0.35 * ((l as f32) + 1.0));
    }
    blk
}

fn fill_fir_taps(dst_l: &mut [F32], dst_r: &mut [F32], base_offset: usize, scale: f32) {
    for i in 0..dst_l.len() {
        let lag = (base_offset + i) as f32;
        let decay = (-0.08 * lag).exp();
        let tap_l = scale
            * decay
            * (if i == 0 && base_offset == 0 {
                1.0
            } else {
                0.12
            });
        let tap_r = scale
            * decay
            * (if i == 0 && base_offset == 0 {
                0.95
            } else {
                -0.10
            });
        dst_l[i] = F32::new(tap_l);
        dst_r[i] = F32::new(tap_r);
    }
}

fn make_kautz_block(b: usize) -> KautzSoaBlockWire {
    let f = b as f32;
    KautzSoaBlockWire {
        r_low: le_f32([0.68, 0.72, 0.76, 0.80]),
        dr: le_f32([0.04, 0.05, 0.04, 0.06]),
        theta: le_f32([
            0.12 + 0.04 * f,
            0.28 + 0.05 * f,
            0.46 + 0.04 * f,
            0.68 + 0.03 * f,
        ]),
        b_gamma: le_f32([0.08, 0.07, 0.06, 0.05]),
        c_re_l: le_f32([0.025, -0.018, 0.015, -0.012]),
        c_im_l: le_f32([0.010, 0.014, -0.011, 0.009]),
        c_re_r: le_f32([0.022, 0.016, -0.014, 0.011]),
        c_im_r: le_f32([-0.012, 0.011, 0.013, -0.008]),
    }
}

pub(crate) fn make_transport_quad(q: usize) -> TransportQuadWire {
    const DELAYS: [[f32; 4]; 4] = [
        [17.0, 23.0, 31.0, 41.0],
        [47.0, 53.0, 59.0, 67.0],
        [71.0, 79.0, 83.0, 89.0],
        [97.0, 101.0, 107.0, 113.0],
    ];
    let flags = if q == 0 { 0 } else { TRANSPORT_FLAG_EXT_F32 };
    TransportQuadWire {
        delay_samples: le_f32(DELAYS[q]),
        ramp_spans_samples: le_f32([4.0, 5.0, 6.0, 7.0]),
        g_loop: F32::new(0.42),
        r_dc: F32::new(TANK_DC_BLOCKER_48K_F32),
        damp_pole: F32::new(0.25),
        lfo_rate_hz: F32::new(1.4 + 0.35 * (q as f32)),
        ramp_rate_hz: F32::new(0.75 + 0.15 * (q as f32)),
        reserved_52_56: F32::new(0.0),
        lfo_depths_fp16: fp16_arr([0.45, 0.40, 0.35, 0.30]),
        lfo_phase_off_q16: [0x0000_u16, 0x4000, 0x8000, 0xC000].map(U16::new),
        ramp_phase_off_q16: [0x1000_u16, 0x5000, 0x9000, 0xD000].map(U16::new),
        input_gains_fp16: fp16_arr([0.45, 0.35, 0.30, 0.25]),
        wet_tap_l_fp16: fp16_arr([0.14, -0.10, 0.12, -0.08]),
        wet_tap_r_fp16: fp16_arr([-0.09, 0.15, -0.07, 0.13]),
        allpass_g_fp16: if q == 0 {
            fp16_arr([0.35, -0.28, 0.24, -0.18])
        } else {
            fp16_arr([0.0; 4])
        },
        fb_psi_fp16: fp16_val(0.20 + 0.08 * (q as f32)),
        sigma_knee_fp16: fp16_val(0.85),
        freeze_mix_fp16: fp16_val(0.15 * (q as f32)),
        window_depth_fp16: fp16_val(0.45),
        scat_psi_fp16: fp16_val(0.35 - 0.08 * (q as f32)),
        allpass_delays: if q == 0 { [11, 17, 23, 31] } else { [0; 4] },
        flags,
        reserved: 0,
    }
}

fn populate_ro_words(words: &mut [U16], num_lanes: usize, write_tail_gains: bool) {
    let w_count = num_lanes * 12;
    let lane_scale = [
        1.0_f32, 0.8, 0.5, -0.5, 0.6, 0.7, 0.9, 0.75, -0.4, 0.4, 0.55, -0.55,
    ];
    for l in 0..num_lanes {
        let sign = if (l & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
        for (j, &sc) in lane_scale.iter().enumerate() {
            words[l * 12 + j] = fp16_val(0.045 * sign * sc);
        }
    }
    if write_tail_gains {
        let gains = [0.15_f32, 0.15, 0.05, 0.05, 0.60, 0.60, 0.05, 0.05];
        for (i, &g) in gains.iter().enumerate() {
            words[w_count + i] = fp16_val(g);
        }
    }
}

fn write_f32_slice(dst: &mut [u8], word_offset: usize, vals: &[f32]) {
    for (i, &v) in vals.iter().enumerate() {
        let byte_off = (word_offset + i) * 4;
        dst[byte_off..byte_off + 4].copy_from_slice(F32::new(v).as_bytes());
    }
}

fn fill_quads_and_tiles(
    quads: &mut [CoreQuadWire],
    q_off: usize,
    tiles: &mut [SonaInterBlockTilePairWire],
    t_off: usize,
) {
    for (i, dst) in quads.iter_mut().enumerate() {
        *dst = make_core_quad(q_off + i);
    }
    for (i, dst) in tiles.iter_mut().enumerate() {
        *dst = make_tile_wire(t_off + i);
    }
}

pub(crate) fn configure_default_sensory_and_crossbar(a0: &mut Arena0LiteWire, tier_flags: u16) {
    let meta_flags = SONA_META_SENSORY_QUAD
        | SONA_META_OS_EXT_PRESENT
        | SONA_META_TRANSPORT_TAP_GAINS
        | SONA_META_TRANSPORT_LFO
        | SONA_META_CORE_MODS_PRESENT;
    a0.crossbar.pipeline_flags = U32::new(PIPE_CANONICAL_V3_SONA);
    a0.crossbar.sona_meta_flags = U32::new(meta_flags);
    a0.crossbar.sense_svf_g = le_f32([0.08, 0.18, 0.35, 0.62]);
    a0.crossbar.sense_lru_alpha_rel = le_f32([0.92, 0.94, 0.96, 0.97]);
    a0.crossbar.sense_lru_alpha_atk = le_f32([0.20, 0.25, 0.30, 0.35]);
    a0.crossbar.sense_svf_k_fp16 = fp16_arr([0.707; 4]);
    a0.crossbar.sense_lru_kappa_fp16 = fp16_arr([0.25, 0.30, 0.35, 0.40]);
    a0.crossbar.sense_in_l_fp16 = fp16_arr([0.6, 0.4, 0.5, 0.3]);
    a0.crossbar.sense_in_r_fp16 = fp16_arr([0.4, 0.6, 0.3, 0.5]);
    a0.crossbar.sense_tone_lp_fp16 = fp16_arr([0.7, 0.5, 0.2, 0.1]);
    a0.crossbar.sense_tone_bp_fp16 = fp16_arr([0.2, 0.5, 0.6, 0.3]);
    a0.crossbar.sense_tone_hp_fp16 = fp16_arr([0.1, 0.2, 0.4, 0.7]);
    a0.crossbar.sense_vca_w_fp16 = fp16_arr([-0.15, -0.10, -0.08, -0.05]);
    a0.crossbar.crossbar_op_gains_fp16 = fp16_arr([0.14, 0.10]);
    a0.crossbar.cell_desc.tier_flags = U16::new(tier_flags);
    a0.crossbar.cell_desc.oversample_factor = U16::new(2);
    a0.crossbar.cell_desc.cell_flags = U32::new(meta_flags);
    a0.crossbar.os_up_quad_coeff = F32::new(-0.125);
    a0.crossbar.os_dec_c_prev1 = F32::new(0.15);
    a0.crossbar.os_dec_c0 = F32::new(0.50);
    a0.crossbar.os_up_pre_emph = F32::new(0.05);
    a0.crossbar.os_dec_aa_mix = F32::new(0.04);
    a0.crossbar.os_up_aa_mix = F32::new(0.04);
    a0.crossbar.os_up_zoh_mix = F32::new(0.02);
    a0.crossbar.carrier_basis_w_fp16 = [
        fp16_arr([0.14, -0.09, 0.11, -0.07]),
        fp16_arr([-0.08, 0.12, -0.06, 0.10]),
        fp16_arr([0.06, 0.05, -0.09, 0.04]),
    ];
    a0.crossbar.op_n_pre_base = OpNPreBaseWire {
        gamma_bc: F32::new(0.08),
        zoh_hold_depth: F32::new(0.05),
        beta_q: F32::new(0.10),
    };
    a0.crossbar.phasor_freq_hz = le_f32([1.5, 3.2, 5.1, 7.3]);
    a0.crossbar.phasor_init_phase_q32 = [0, 0x4000_0000, 0x8000_0000, 0xC000_0000].map(U32::new);
}

pub(crate) fn build_full_pipeline_ultra_fixture() -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    configure_default_sensory_and_crossbar(&mut a0, 0x000F);

    fill_quads_and_tiles(&mut a0.core_quads, 0, &mut a0.tiles, 0);
    fill_fir_taps(
        &mut a0.conv_fir_lite.fir_l,
        &mut a0.conv_fir_lite.fir_r,
        0,
        0.5,
    );
    a0.conv_kautz_lite = [make_kautz_block(0), make_kautz_block(1)];
    a0.transport_lite = make_transport_quad(0);
    a0.linear_readout_lite.pre_conv_u_gain = le_f32([0.15, 0.15]);
    a0.linear_readout_lite.pre_conv_hp_gain = le_f32([0.05, 0.05]);
    a0.linear_readout_lite.direct_s2_u_gain = le_f32([0.60, 0.60]);
    a0.linear_readout_lite.direct_s2_hp_gain = le_f32([0.05, 0.05]);
    for (i, w) in a0
        .linear_readout_lite
        .stage_latent_mods_fp16
        .iter_mut()
        .enumerate()
    {
        let sign = if (i & 1) == 0 { 1.0 } else { -1.0 };
        *w = fp16_val(if i < 8 {
            0.04
        } else if i < 80 {
            0.03 * sign
        } else {
            0.05
        });
    }
    populate_ro_words(&mut a0.linear_readout_lite.w_ro_fp16[..192], 16, false);
    a0.linear_readout_lite.w_ro_fp16[192..208].fill(fp16_val(0.06));

    a0.control_bus_lite.lfo_dphi_train = le_f32([0.004, 0.009]);
    a0.control_bus_lite.lfo_skew_fp16 = fp16_arr([0.25, -0.25]);
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(0.65);
    a0.control_bus_lite.sigma_drift = F32::new(0.02);
    a0.control_bus_lite.rho_drift_train = F32::new(0.99);
    a0.control_bus_lite.ou_to_z_stage_fp16 = fp16_arr([0.10, -0.05, 0.04, -0.02]);
    for &(r, c, v) in &[
        (0, 0, 0.18),
        (1, 1, 0.12),
        (10, 0, 0.10),
        (14, 2, 0.08),
        (15, 8, 0.05),
    ] {
        a0.control_bus_lite.w_ctrl[r][c] = F32::new(v);
    }
    a0.control_bus_lite.u_int_base[0][0] = F32::new(0.15);
    a0.control_bus_lite.u_int_base[0][1] = F32::new(-0.10);
    for q in 0..4 {
        a0.control_bus_lite.rho_head_and_ctrl_tail[24 + q * 2] = F32::new(0.04);
        a0.control_bus_lite.rho_head_and_ctrl_tail[24 + q * 2 + 1] = F32::new(-0.04);
    }
    a0.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(3);
    a0.control_bus_lite.cp1_dir.entries[0..3].copy_from_slice(&[
        [0, 1, 0xFF],
        [1, 1, 0xFF],
        [2, 1, 0xFF],
    ]);
    let cp1_slots: [(usize, [f32; 13]); 3] = [
        (
            0,
            [
                0.25, -0.15, 0.12, 0.18, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.1, -0.05, 0.05,
            ],
        ),
        (
            13,
            [
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0,
            ],
        ),
        (
            26,
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0,
            ],
        ),
    ];
    for (off, vals) in &cp1_slots {
        write_f32_slice(&mut a0.control_bus_lite.cp1_base, *off, vals);
    }

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    fill_quads_and_tiles(&mut a1.core_quads, 4, &mut a1.tiles, 3);
    a1.crossbar_full = make_crossbar_block(1);
    fill_fir_taps(
        &mut a1.conv_fir_full.fir_l,
        &mut a1.conv_fir_full.fir_r,
        32,
        0.45,
    );
    a1.conv_kautz_full = make_kautz_block(2);
    a1.transport_full = make_transport_quad(1);
    populate_ro_words(&mut a1.linear_readout_full.w_ro_and_mods_fp16, 24, true);

    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    fill_quads_and_tiles(&mut a2.core_quads, 6, &mut a2.tiles, 13);
    a2.crossbar_pro = make_crossbar_block(2);
    fill_fir_taps(
        &mut a2.conv_fir_pro.fir_l,
        &mut a2.conv_fir_pro.fir_r,
        48,
        0.40,
    );
    a2.conv_kautz_pro = [
        make_kautz_block(3),
        make_kautz_block(4),
        make_kautz_block(5),
    ];
    a2.transport_pro[0] = make_transport_quad(2);
    populate_ro_words(&mut a2.linear_readout_pro.w_ro_and_mods_fp16, 32, true);

    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    fill_quads_and_tiles(&mut a3.core_quads, 8, &mut a3.tiles, 23);
    a3.crossbar_ultra = make_crossbar_block(3);
    for k in 0..8 {
        let base = k * 128;
        for col in 0..120 {
            let s = if ((k + col) & 1) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            a3.stage_latent_core_mods_fp16[base + col] = fp16_val(0.015 * s);
        }
    }
    fill_fir_taps(
        &mut a3.conv_fir_ultra.fir_l,
        &mut a3.conv_fir_ultra.fir_r,
        96,
        0.35,
    );
    a3.conv_kautz_ultra = [make_kautz_block(6), make_kautz_block(7)];
    a3.transport_ultra[0] = make_transport_quad(3);
    populate_ro_words(&mut a3.linear_readout_ultra.w_ro_and_mods_fp16, 40, true);
    a3.control_bus_ultra_delta.w_ctrl_ultra[0][0] = F32::new(0.08);
    write_f32_slice(
        &mut a3.control_bus_ultra_delta.cp1_ultra,
        0,
        &[
            0.06, -0.04, 0.03, 0.05, 0.05, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0,
        ],
    );
    for i in 0..160 {
        let s = if (i & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[i] = fp16_val(0.02 * s);
    }
    for i in 0..4 {
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[160 + i] = fp16_val(0.03);
    }

    let mut c0 = ControlClampDescriptor::range(20.0, 2_000.0, 500.0, 3, [0.0, -0.5, 0.5]);
    c0.cond_rules[0] = ConditionalClampRule {
        switch_ctrl_idx: 3,
        switch_opt_idx: 1,
        rule_op: 1,
        bound_val: 1_200.0,
    };
    let c1 = ControlClampDescriptor::range(0.0, 1.0, 1.0, 1, [0.0, 0.0, 0.0]);
    let c2 = ControlClampDescriptor::range(0.0, 1.0, 0.0, 1, [0.0, 0.0, 0.0]);
    let c3 = ControlClampDescriptor::switch(3, 0b111, 0);
    let controls = [c0.to_wire(), c1.to_wire(), c2.to_wire(), c3.to_wire()];

    let tsyn = [TempoSyncEntryWire {
        ctrl_idx: U16::new(0),
        switch_ctrl_idx: U16::new(3),
        switch_opt_idx: 0,
        has_switch_cond: 0,
        mode: 1,
        default_subdiv: 4,
        flags: U32::new(0),
        div_ratio: F32::new(1.0),
        bpm_factor: F32::new(1.0),
        lobatto_anchors: le_f32([20.0, 500.0, 2_000.0]),
        declared_min: F32::new(20.0),
        declared_max: F32::new(2_000.0),
    }];

    let mut cext = CextDispatchWire::new_zeroed();
    // Row 0 -> slot 16 (op_t_freeze_mix_delta): ctrl 1 modulates freeze mix via q16_basis[0] (col 4).
    cext.w_ctrl_ext[0][4] = F32::new(0.60);
    // Row 6 -> slot 22 (op_t_dry_mix_mod): ctrl 2 modulates stage tank_dry_mix via q16_basis[1] (col 5).
    cext.w_ctrl_ext[6][5] = F32::new(0.35);
    cext.w_ctrl_ext[1][0] = F32::new(0.04);
    // Row 14 -> slot 30 (sense_vca_gain_delta).
    cext.w_ctrl_ext[14][0] = F32::new(0.05);

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: DNBN_FLAG_STEREO_OUT,
        ext_ring_pow2_frames: 10,
        ext_ring_lines: 4,
        ext_ring_format: 1,
        declared_max_delay_48k: 200,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 1_500.0,
        ro_pole_48k: 0.995,
        switch_xfade_samples: 64,
        content_hash_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
        controls: &controls,
        cp1_overflow: &[],
        tsyn: &tsyn,
        cext: Some(&cext),
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
    .expect("encode full pipeline ultra fixture")
}

pub(crate) fn run_no_alloc_lite(
    lite_bytes: &[u8],
    in_l: &[f32; 64],
    in_r: &[f32; 64],
) -> ([f32; 64], [f32; 64], [f32; 64], [f32; 64]) {
    let mut out_l_open = [0.0_f32; 64];
    let mut out_r_open = [0.0_f32; 64];
    let mut out_l_pre = [0.0_f32; 64];
    let mut out_r_pre = [0.0_f32; 64];

    for (pass, (dst_l, dst_r)) in [
        (&mut out_l_open, &mut out_r_open),
        (&mut out_l_pre, &mut out_r_pre),
    ]
    .into_iter()
    .enumerate()
    {
        let mut cold_tail = [SonaColdTail::<DefaultMathBackend>::empty()];
        let mut descs = [ControlClampDescriptor::empty(); 8];
        let (mut raw, mut clamped, mut norm, mut switches) =
            ([0.0_f32; 8], [0.0_f32; 8], [0.0_f32; 8], [0_u8; 8]);
        let mut contributors = [Cp1ContributorSlot::zero(); 8];
        let mut transport_fp16_q0 = [0_u16; 1_024 * 4];
        let mut transport_sch_q0 = [0.0_f32; 512];

        let ctrl = ControlBusBacking::from_slices(
            &mut descs,
            &mut raw,
            &mut clamped,
            &mut norm,
            &mut switches,
            &mut contributors,
            &[],
        )
        .expect("stack ctrl backing");
        let backing = SonaVmBacking {
            control_bus: ctrl,
            convolver_fir_ext: &mut [],
            transport_ext_fp16: [&mut transport_fp16_q0, &mut [], &mut [], &mut []],
            transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            transport_schroeder: [&mut transport_sch_q0, &mut [], &mut [], &mut []],
            cold_tail: &mut cold_tail,
        };
        let mut vm = if pass == 0 {
            SonaVm::<DefaultMathBackend>::open_dnam_bin(lite_bytes, Tier::Lite, 48_000.0, backing)
        } else {
            SonaVm::<DefaultMathBackend>::open_dnam_bin_prevalidated(
                lite_bytes,
                Tier::Lite,
                48_000.0,
                backing,
            )
        }
        .expect("no-alloc lite VM open");
        vm.process_block(in_l, in_r, dst_l, dst_r)
            .expect("no-alloc lite process_block");
    }

    (out_l_open, out_r_open, out_l_pre, out_r_pre)
}

fn build_q_matrix_so4(theta: f32) -> [[f32; 4]; 4] {
    let (s, c) = theta.sin_cos();
    [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, c, -s],
        [0.0, 0.0, s, c],
    ]
}

fn mat4_mul_vec(m: &[[f32; 4]; 4], v: &[f32; 4]) -> [f32; 4] {
    core::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + m[i][3] * v[3])
}

fn rotate_tile_q_m_qt(q: &[[f32; 4]; 4], m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for s in 0..4 {
        for d in 0..4 {
            let mut sum = 0.0_f32;
            for a in 0..4 {
                for b in 0..4 {
                    sum += q[s][a] * m[a][b] * q[d][b];
                }
            }
            out[s][d] = sum;
        }
    }
    out
}

struct GaugeWeights {
    in_mid: [[f32; 4]; 4],
    tiles_odd: [[[f32; 4]; 4]; 3],
    tiles_lru: [[[f32; 4]; 4]; 3],
    pre_lru: [[f32; 4]; 4],
    dir_lru: [[f32; 4]; 4],
    pre_odd: [[f32; 4]; 4],
    dir_odd: [[f32; 4]; 4],
}

impl GaugeWeights {
    fn rotate(&self, q: &[[f32; 4]; 4]) -> Self {
        Self {
            in_mid: core::array::from_fn(|g| mat4_mul_vec(q, &self.in_mid[g])),
            tiles_odd: core::array::from_fn(|t| rotate_tile_q_m_qt(q, &self.tiles_odd[t])),
            tiles_lru: core::array::from_fn(|t| rotate_tile_q_m_qt(q, &self.tiles_lru[t])),
            pre_lru: core::array::from_fn(|g| mat4_mul_vec(q, &self.pre_lru[g])),
            dir_lru: core::array::from_fn(|g| mat4_mul_vec(q, &self.dir_lru[g])),
            pre_odd: core::array::from_fn(|g| mat4_mul_vec(q, &self.pre_odd[g])),
            dir_odd: core::array::from_fn(|g| mat4_mul_vec(q, &self.dir_odd[g])),
        }
    }

    fn to_lite_arena(&self) -> Arena0LiteWire {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        let mf = SONA_META_SENSORY_QUAD | SONA_META_TRANSPORT_TAP_GAINS;
        a0.crossbar.sona_meta_flags = U32::new(mf);
        a0.crossbar.cell_desc.cell_flags = U32::new(mf);
        for q in 0..4 {
            let mut quad = CoreQuadWire::new_zeroed();
            quad.svf_g = le_f32([0.38; 4]);
            quad.svf_k_fp16 = fp16_arr([0.707; 4]);
            quad.lru_alpha_rel = le_f32([0.25; 4]);
            quad.lru_alpha_atk = le_f32([0.996; 4]);
            quad.in_mid_weight_fp16 = fp16_arr(self.in_mid[q]);
            quad.tone_lp_fp16 = fp16_arr([0.5; 4]);
            quad.tone_dir_fp16 = fp16_arr([1.0; 4]);
            quad.wavefolder_mix_fp16 = [U16::new(0); 4];
            a0.core_quads[q] = quad;
        }
        for t in 0..3 {
            let mut tile = SonaInterBlockTilePairWire::new_zeroed();
            for s in 0..4 {
                for d in 0..4 {
                    tile.w_odd_4x4[s][d] = F32::new(self.tiles_odd[t][s][d]);
                    tile.w_lru_4x4_fp16[s][d] = fp16_val(self.tiles_lru[t][s][d]);
                }
            }
            a0.tiles[t] = tile;
        }
        a0.conv_fir_lite.fir_l[0] = F32::new(0.8);
        a0.conv_fir_lite.fir_r[0] = F32::new(0.8);
        a0.transport_lite.delay_samples = le_f32([16.0; 4]);
        a0.transport_lite.g_loop = F32::new(0.25);
        a0.transport_lite.input_gains_fp16 = fp16_arr([0.25; 4]);
        a0.transport_lite.wet_tap_l_fp16 = fp16_arr([0.20; 4]);
        a0.transport_lite.wet_tap_r_fp16 = fp16_arr([0.20; 4]);
        a0.transport_lite.sigma_knee_fp16 = fp16_val(0.9);
        a0.control_bus_lite.tank_dry_mix_fp16 = fp16_val(0.5);
        a0.linear_readout_lite.direct_s2_u_gain = le_f32([0.02, 0.02]);

        for q in 0..4 {
            for k in 0..4 {
                let b = (q * 4 + k) * 12;
                let pairs = [
                    (0, self.pre_odd[q][k]),
                    (4, self.pre_lru[q][k]),
                    (6, self.dir_odd[q][k]),
                    (10, self.dir_lru[q][k]),
                ];
                for (off, val) in pairs {
                    let h = fp16_val(val);
                    a0.linear_readout_lite.w_ro_fp16[b + off] = h;
                    a0.linear_readout_lite.w_ro_fp16[b + off + 1] = h;
                }
            }
        }
        a0
    }
}

fn make_gauge_tile(t: usize, scale: f32) -> [[f32; 4]; 4] {
    core::array::from_fn(|s| {
        core::array::from_fn(|d| {
            let phase = ((t * 3 + s * 5 + d * 7) % 9) as f32 - 4.0;
            if s == d {
                scale * (1.0 + 0.1 * ((t as f32) - 1.0))
            } else {
                scale * 0.12 * phase
            }
        })
    })
}

fn make_gauge_ro(seed: f32) -> [[f32; 4]; 4] {
    core::array::from_fn(|q| {
        core::array::from_fn(|k| {
            let parity = if ((q + k) & 1) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            seed * parity * (1.0 - 0.12 * (q as f32)) * (1.0 - 0.1 * (k as f32))
        })
    })
}

pub(crate) fn build_so4_qbuyer_pair(theta: f32) -> (Vec<u8>, Vec<u8>) {
    let q = build_q_matrix_so4(theta);
    let base = GaugeWeights {
        in_mid: [
            [0.40, 0.16, 0.36, 0.12],
            [0.32, 0.14, 0.28, 0.10],
            [0.24, 0.10, 0.22, 0.08],
            [0.20, 0.08, 0.18, 0.06],
        ],
        tiles_odd: core::array::from_fn(|t| make_gauge_tile(t, 0.10)),
        tiles_lru: core::array::from_fn(|t| make_gauge_tile(t + 3, 0.18)),
        pre_lru: make_gauge_ro(0.45),
        dir_lru: make_gauge_ro(0.50),
        pre_odd: make_gauge_ro(0.25),
        dir_odd: make_gauge_ro(0.30),
    };
    let arenas = [base.to_lite_arena(), base.rotate(&q).to_lite_arena()];
    let [bytes_a, bytes_b] =
        arenas.map(|arena0| encode_lite_fixture(&arena0, DNBN_FLAG_STEREO_OUT, 0.99, &[], None));
    (bytes_a, bytes_b)
}
