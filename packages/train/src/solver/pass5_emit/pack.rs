//! `zerocopy 0.8` `Arena0..3`, `PrefixOverride*`, `SonaCrossbarBlockWire`, and
//! `.dnam-bin` serialization for Pass 5 (`ref[container.dnam.vm-binary]`).

mod blocks;
pub(super) mod readouts;

pub(super) use blocks::sanitize_clamp_f32;

use dnam_kernels::MAX_PARALRU_R_F32;
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    ConvolverFir16Wire, ConvolverFir32Wire, ConvolverFir48Wire, DNBN_FLAG_SINGLE_INPUT,
    DNBN_FLAG_STEREO_OUT, PIPE_CANONICAL_V3_SONA, SONA_CELL_MAGIC,
    SONA_META_CONVOLVER_PREFIX_OVERRIDES, SONA_META_OS_EXT_PRESENT, SONA_META_OS2X,
    SONA_META_SENSORY_QUAD, SONA_META_TIER_OS_EXPLICIT, SONA_META_TRANSPORT_LFO,
    SONA_META_TRANSPORT_TAP_GAINS, TRANSPORT_FLAG_EXT_F32, encode_dnam_bin,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use dnam_vm::{DnamBinBuildInput, SONA_CORE_QUAD_BLOCK_AND_LOCAL};
use sha2::{Digest, Sha256};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromZeros, IntoBytes};

use crate::control::stability::{MAX_RO_POLE_48K_F32, enforce_arena_block0_k_nl_and_clamp};
use crate::error::TrainError;

use super::super::model::{SonaModelF64, TransportQuadF64};
use super::Pass5EmitConfig;
use blocks::{
    fp16_clamped, pack_core_quad_wire, pack_crossbar_block_wire, pack_kautz_block, pack_tile_wire,
    pack_transport_quad_wire,
};
use readouts::{LITE_OPT_TO_LR_OFFSET, pack_tier_readouts};

/// Serializes `model` into a validated 4-tier `.dnam-bin` payload (`ref[container.dnam.vm-binary]`).
pub fn encode_model_to_dnam_bin(
    model: &SonaModelF64,
    config: Pass5EmitConfig,
) -> Result<Vec<u8>, TrainError> {
    let mut a0 = Arena0LiteWire::new_zeroed();
    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    let mut a3 = Arena3UltraDeltaWire::new_zeroed();

    let emit_prefix_overrides = model.prefix_override_needed[1..].iter().any(|&v| v);
    let mut meta_flags = SONA_META_OS_EXT_PRESENT
        | SONA_META_SENSORY_QUAD
        | SONA_META_TRANSPORT_TAP_GAINS
        | SONA_META_TRANSPORT_LFO
        | SONA_META_TIER_OS_EXPLICIT;
    if model.oversample_upper_tiers_2x {
        meta_flags |= SONA_META_OS2X | 0b1110;
    }
    if emit_prefix_overrides {
        meta_flags |= SONA_META_CONVOLVER_PREFIX_OVERRIDES;
    }

    a0.crossbar.pipeline_flags = U32::new(PIPE_CANONICAL_V3_SONA);
    a0.crossbar.sona_meta_flags = U32::new(meta_flags);
    a0.crossbar.anti_zipper_ramp_scale = F32::new(0.0625);
    a0.crossbar.os_up_quad_coeff = F32::new(sanitize_clamp_f32(
        model.os_up_quad_coeff,
        -0.20,
        0.05,
        -0.125,
    ));
    a0.crossbar.os_dec_c_prev1 =
        F32::new(sanitize_clamp_f32(model.os_dec_c_prev1, 0.0, 0.28, 0.15));
    a0.crossbar.os_dec_c0 = F32::new(sanitize_clamp_f32(model.os_dec_c0, 0.0, 0.55, 0.50));
    a0.crossbar.cell_desc.magic = SONA_CELL_MAGIC;
    a0.crossbar.cell_desc.version_major = U16::new(3);
    a0.crossbar.cell_desc.version_minor = U16::new(0);
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.crossbar.cell_desc.oversample_factor = U16::new(if model.oversample_upper_tiers_2x {
        2
    } else {
        1
    });
    a0.crossbar.cell_desc.cell_flags = U32::new(meta_flags);

    for i in 0..4 {
        a0.crossbar.sense_svf_g[i] =
            F32::new(sanitize_clamp_f32(model.sense_svf_g[i], 1.0e-5, 64.0, 0.25));
        a0.crossbar.sense_lru_alpha_rel[i] = F32::new(sanitize_clamp_f32(
            model.sense_lru_rel[i],
            0.0,
            MAX_PARALRU_R_F32,
            0.995,
        ));
        a0.crossbar.sense_lru_alpha_atk[i] =
            F32::new(sanitize_clamp_f32(model.sense_lru_atk[i], 0.0, 1.0, 0.25));
        a0.crossbar.sense_svf_k_fp16[i] = fp16_clamped(
            model.sense_svf_k[i],
            0.0,
            8.0,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        a0.crossbar.sense_lru_kappa_fp16[i] = fp16_clamped(model.sense_lru_kappa[i], 0.0, 1.0, 0.0);
        a0.crossbar.sense_vca_w_fp16[i] = fp16_clamped(model.sense_vca_w[i], -512.0, 512.0, 0.0);
        a0.crossbar.sense_in_l_fp16[i] = fp16_clamped(model.sense_in_l[i], -4.0, 4.0, 0.5);
        a0.crossbar.sense_in_r_fp16[i] = fp16_clamped(model.sense_in_r[i], -4.0, 4.0, 0.5);
        a0.crossbar.sense_tone_lp_fp16[i] = fp16_clamped(model.sense_tone_lp[i], -16.0, 16.0, 1.0);
        a0.crossbar.sense_tone_bp_fp16[i] = fp16_clamped(model.sense_tone_bp[i], -16.0, 16.0, 0.0);
        a0.crossbar.sense_tone_hp_fp16[i] = fp16_clamped(model.sense_tone_hp[i], -16.0, 16.0, 0.0);
        a0.crossbar.tier_gain_trim[i] = F32::new(sanitize_clamp_f32(
            model.tier_gain_trim[i],
            0.0625,
            16.0,
            1.0,
        ));
        a0.crossbar.phasor_freq_hz[i] = F32::new(sanitize_clamp_f32(
            model.phasor_freq_hz[i],
            0.0,
            20_000.0,
            0.0,
        ));
    }
    a0.crossbar.crossbar_op_gains_fp16 = [
        fp16_clamped(model.crossbar_op_gains[0], -4.0, 4.0, 0.0),
        fp16_clamped(model.crossbar_op_gains[1], -4.0, 4.0, 0.0),
    ];
    a0.crossbar.op_n_pre_base.gamma_bc =
        F32::new(sanitize_clamp_f32(model.op_n_pre_gamma_bc, 0.0, 1.0, 0.0));
    a0.crossbar.op_n_pre_base.zoh_hold_depth = F32::new(sanitize_clamp_f32(
        model.op_n_pre_zoh_hold_depth,
        0.0,
        0.9999,
        0.0,
    ));
    a0.crossbar.op_n_pre_base.beta_q =
        F32::new(sanitize_clamp_f32(model.op_n_pre_beta_q, 0.0, 16.0, 0.0));

    for i in 0..3 {
        for j in 0..4 {
            a0.crossbar.carrier_basis_w_fp16[i][j] =
                fp16_clamped(model.carrier_basis_w[i][j], -8.0, 8.0, 0.0);
        }
    }

    let xbar_blk = pack_crossbar_block_wire(model);
    a1.crossbar_full = xbar_blk;
    a2.crossbar_pro = xbar_blk;
    a3.crossbar_ultra = xbar_blk;

    for q in 0..4 {
        a0.core_quads[q] = pack_core_quad_wire(
            &model.core_quads[q],
            SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0,
        );
    }
    for q in 0..2 {
        a1.core_quads[q] = pack_core_quad_wire(
            &model.core_quads[4 + q],
            SONA_CORE_QUAD_BLOCK_AND_LOCAL[4 + q].0 == 0,
        );
        a2.core_quads[q] = pack_core_quad_wire(
            &model.core_quads[6 + q],
            SONA_CORE_QUAD_BLOCK_AND_LOCAL[6 + q].0 == 0,
        );
        a3.core_quads[q] = pack_core_quad_wire(
            &model.core_quads[8 + q],
            SONA_CORE_QUAD_BLOCK_AND_LOCAL[8 + q].0 == 0,
        );
    }
    for i in 0..3 {
        a0.tiles[i] = pack_tile_wire(&model.tiles[i]);
    }
    for i in 0..10 {
        a1.tiles[i] = pack_tile_wire(&model.tiles[3 + i]);
        a2.tiles[i] = pack_tile_wire(&model.tiles[13 + i]);
    }
    for i in 0..12 {
        a3.tiles[i] = pack_tile_wire(&model.tiles[23 + i]);
    }

    let lite_pfx = model.tier_overrides[0].as_deref();
    let lite_fir_l = lite_pfx.map_or(&model.fir_l, |p| &p.fir_l);
    let lite_fir_r = lite_pfx.map_or(&model.fir_r, |p| &p.fir_r);
    let mut fir_lite = ConvolverFir32Wire::new_zeroed();
    for i in 0..32 {
        fir_lite.fir_l[i] = F32::new(sanitize_clamp_f32(lite_fir_l[i], -16.0, 16.0, 0.0));
        fir_lite.fir_r[i] = F32::new(sanitize_clamp_f32(lite_fir_r[i], -16.0, 16.0, 0.0));
    }
    a0.conv_fir_lite = fir_lite;

    let mut fir_full = ConvolverFir16Wire::new_zeroed();
    let full_pfx = model.tier_overrides[1].as_deref();
    let full_fir_l = full_pfx.map_or(&model.fir_l, |p| &p.fir_l);
    let full_fir_r = full_pfx.map_or(&model.fir_r, |p| &p.fir_r);
    for i in 0..16 {
        fir_full.fir_l[i] = F32::new(sanitize_clamp_f32(full_fir_l[32 + i], -16.0, 16.0, 0.0));
        fir_full.fir_r[i] = F32::new(sanitize_clamp_f32(full_fir_r[32 + i], -16.0, 16.0, 0.0));
    }
    a1.conv_fir_full = fir_full;

    let mut fir_pro = ConvolverFir48Wire::new_zeroed();
    let pro_pfx = model.tier_overrides[2].as_deref();
    let pro_fir_l = pro_pfx.map_or(&model.fir_l, |p| &p.fir_l);
    let pro_fir_r = pro_pfx.map_or(&model.fir_r, |p| &p.fir_r);
    for i in 0..48 {
        fir_pro.fir_l[i] = F32::new(sanitize_clamp_f32(pro_fir_l[48 + i], -16.0, 16.0, 0.0));
        fir_pro.fir_r[i] = F32::new(sanitize_clamp_f32(pro_fir_r[48 + i], -16.0, 16.0, 0.0));
    }
    a2.conv_fir_pro = fir_pro;

    let mut fir_ultra = ConvolverFir32Wire::new_zeroed();
    let ult_pfx = model.tier_overrides[3].as_deref();
    let ult_fir_l = ult_pfx.map_or(&model.fir_l, |p| &p.fir_l);
    let ult_fir_r = ult_pfx.map_or(&model.fir_r, |p| &p.fir_r);
    for i in 0..32 {
        fir_ultra.fir_l[i] = F32::new(sanitize_clamp_f32(ult_fir_l[96 + i], -16.0, 16.0, 0.0));
        fir_ultra.fir_r[i] = F32::new(sanitize_clamp_f32(ult_fir_r[96 + i], -16.0, 16.0, 0.0));
    }
    a3.conv_fir_ultra = fir_ultra;

    a0.conv_kautz_lite[0] = pack_kautz_block(&model.kautz[0..4], lite_pfx, 0);
    a0.conv_kautz_lite[1] = pack_kautz_block(&model.kautz[4..8], lite_pfx, 4);
    a1.conv_kautz_full = pack_kautz_block(&model.kautz[8..12], full_pfx, 8);
    a2.conv_kautz_pro[0] = pack_kautz_block(&model.kautz[12..16], pro_pfx, 12);
    a2.conv_kautz_pro[1] = pack_kautz_block(&model.kautz[16..20], pro_pfx, 16);
    a2.conv_kautz_pro[2] = pack_kautz_block(&model.kautz[20..24], pro_pfx, 20);
    a3.conv_kautz_ultra[0] = pack_kautz_block(&model.kautz[24..28], ult_pfx, 24);
    a3.conv_kautz_ultra[1] = pack_kautz_block(&model.kautz[28..32], ult_pfx, 28);

    a0.transport_lite = pack_transport_quad_wire(&model.transport_quads[0]);
    a1.transport_full = pack_transport_quad_wire(&model.transport_quads[1]);
    a2.transport_pro[0] = pack_transport_quad_wire(&model.transport_quads[2]);
    a3.transport_ultra[0] = pack_transport_quad_wire(&model.transport_quads[3]);

    let any_ext_f32 = ((a0.transport_lite.flags
        | a1.transport_full.flags
        | a2.transport_pro[0].flags
        | a3.transport_ultra[0].flags)
        & TRANSPORT_FLAG_EXT_F32)
        != 0;
    let quad_max_total = |tq: &TransportQuadF64| -> (f64, f64) {
        let mut max_d = 5.0_f64;
        let mut max_tot = 5.0_f64;
        for k in 0..4 {
            let d = if tq.delay_samples[k].is_finite() {
                tq.delay_samples[k].clamp(5.0, 48_000.0)
            } else {
                16.0
            };
            let span = if tq.ramp_spans_samples[k].is_finite() {
                tq.ramp_spans_samples[k].clamp(0.0, 48_000.0)
            } else {
                0.0
            };
            let lfo = if tq.lfo_depths[k].is_finite() {
                tq.lfo_depths[k].clamp(0.0, 512.0)
            } else {
                0.0
            };
            max_d = max_d.max(d);
            max_tot = max_tot.max(d + 0.5 * span + lfo);
        }
        (max_d, max_tot)
    };
    let (max_delay_q0, max_total_q0) = quad_max_total(&model.transport_quads[0]);
    let mut max_delay_48k_f = max_delay_q0;
    let mut max_total_48k = max_total_q0;
    for tq in &model.transport_quads[1..] {
        let (md, mt) = quad_max_total(tq);
        max_delay_48k_f = max_delay_48k_f.max(md);
        max_total_48k = max_total_48k.max(mt);
    }
    let max_delay_48k = max_delay_48k_f.ceil().clamp(16.0, 48_000.0) as u32;
    let req_frames_48k = (max_total_48k.ceil().clamp(16.0, 524_288.0) as u32) + 6;
    let need_ext_q0 = ((a0.transport_lite.flags & TRANSPORT_FLAG_EXT_F32) != 0)
        || max_delay_q0 > 120.0
        || ((max_total_q0.ceil().clamp(16.0, 48_000.0) as u32) + 6) > 128;
    let ext_pow2 = if need_ext_q0 || any_ext_f32 || req_frames_48k > 128 {
        (req_frames_48k.max(256).next_power_of_two().trailing_zeros() as u8).clamp(8, 19)
    } else {
        7_u8
    };
    let (ext_lines, ext_fmt) = (16_u8, u8::from(any_ext_f32));

    pack_tier_readouts(
        model,
        emit_prefix_overrides,
        &mut a0,
        &mut a1,
        &mut a2,
        &mut a3,
    );

    if let Some(ref cb) = model.control_bus {
        a0.control_bus_lite = cb.arena0_cb;
        a3.control_bus_ultra_delta = cb.arena3_cb;
    } else {
        a0.control_bus_lite.rho_drift_train = F32::new(0.999);
        a0.control_bus_lite.trust_radius_mh = F32::new(2.5);
        a0.control_bus_lite.rho_head_and_ctrl_tail[0] = F32::new(2.5);
    }
    for idx in 0..4 {
        a0.control_bus_lite.rho_head_and_ctrl_tail[LITE_OPT_TO_LR_OFFSET + idx * 2] =
            F32::new(sanitize_clamp_f32(model.opt_to_lr[idx][0], -4.0, 4.0, 0.0));
        a0.control_bus_lite.rho_head_and_ctrl_tail[LITE_OPT_TO_LR_OFFSET + idx * 2 + 1] =
            F32::new(sanitize_clamp_f32(model.opt_to_lr[idx][1], -4.0, 4.0, 0.0));
    }
    a0.control_bus_lite.tank_dry_mix_fp16 = fp16_clamped(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let mut ro_pole = sanitize_clamp_f32(model.ro_pole_48k, 0.0, MAX_RO_POLE_48K_F32, 0.0);
    enforce_arena_block0_k_nl_and_clamp(
        &mut a0,
        Some(&mut a1),
        Some(&mut a2),
        Some(&mut a3),
        &mut ro_pole,
    );

    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-BIN-CONTENT-V3\0");
    hasher.update(a0.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let mut prefix8 = [0_u8; 8];
    prefix8.copy_from_slice(&digest[..8]);

    let mut h_flags = 0_u16;
    if config.stereo_out {
        h_flags |= DNBN_FLAG_STEREO_OUT;
    }
    if config.single_input {
        h_flags |= DNBN_FLAG_SINGLE_INPUT;
    }

    let ctrls = model
        .control_audit
        .as_ref()
        .map_or(&[][..], |a| a.wire_descriptors.as_slice());
    let cp1_over = model
        .control_bus
        .as_ref()
        .map_or(&[][..], |cb| cb.cp1_overflow.as_slice());
    let tsyn_slice = model
        .control_bus
        .as_ref()
        .map_or(&[][..], |cb| cb.tsyn.as_slice());
    let cext_opt = model.control_bus.as_ref().and_then(|cb| cb.cext.as_ref());

    Ok(encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: (sanitize_clamp_f32(
            model.sample_rate_hz,
            44_100.0,
            96_000.0,
            48_000.0,
        )
        .round() as u32)
            .clamp(44_100, 96_000),
        header_flags: h_flags,
        ext_ring_pow2_frames: ext_pow2,
        ext_ring_lines: ext_lines,
        ext_ring_format: ext_fmt,
        declared_max_delay_48k: max_delay_48k,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: sanitize_clamp_f32(f64::from(config.tail_time_ms), 0.0, 60_000.0, 80.0),
        ro_pole_48k: ro_pole,
        switch_xfade_samples: config.switch_xfade_samples,
        content_hash_prefix: prefix8,
        controls: ctrls,
        cp1_overflow: cp1_over,
        tsyn: tsyn_slice,
        cext: cext_opt,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::{f32_to_fp16, fp16_to_f32};
    use dnam_vm::bin::DnamBinView;
    use dnam_vm::math::DefaultMathBackend;
    use dnam_vm::{SonaVm, Tier};

    #[test]
    fn test_encode_model_roundtrips_block0_k_nl_and_scat_psi_through_vm() {
        let mut model = SonaModelF64::new(48_000.0);
        for q in 0..10 {
            model.core_quads[q].svf_k_nl = [0.25 + 0.1 * (q as f64), 0.5, 1.0, 2.0];
        }
        for tq in &mut model.transport_quads {
            tq.scat_psi = core::f64::consts::FRAC_PI_4;
            tq.allpass_delays = [3, 5, 7, 11];
        }
        let bin = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap();
        let view = DnamBinView::parse(&bin).unwrap();
        let a0 = view.arena0;
        let a1 = view.arena1.unwrap();
        let a2 = view.arena2.unwrap();
        let a3 = view.arena3.unwrap();

        // Block 0 quads {0, 4, 6, 8} keep clamped svf_k_nl_fp16; Blocks 1..=3 {1, 2, 3, 5, 7, 9} are zeroed.
        assert_ne!(a0.core_quads[0].svf_k_nl_fp16.map(U16::get), [0; 4]);
        for q_lin in 1..4 {
            assert_eq!(a0.core_quads[q_lin].svf_k_nl_fp16.map(U16::get), [0; 4]);
        }
        assert_ne!(a1.core_quads[0].svf_k_nl_fp16.map(U16::get), [0; 4]);
        assert_eq!(a1.core_quads[1].svf_k_nl_fp16.map(U16::get), [0; 4]);
        assert_ne!(a2.core_quads[0].svf_k_nl_fp16.map(U16::get), [0; 4]);
        assert_eq!(a2.core_quads[1].svf_k_nl_fp16.map(U16::get), [0; 4]);
        assert_ne!(a3.core_quads[0].svf_k_nl_fp16.map(U16::get), [0; 4]);
        assert_eq!(a3.core_quads[1].svf_k_nl_fp16.map(U16::get), [0; 4]);

        let expected_scat = f32_to_fp16(core::f32::consts::FRAC_PI_4);
        for tw in [
            &a0.transport_lite,
            &a1.transport_full,
            &a2.transport_pro[0],
            &a3.transport_ultra[0],
        ] {
            assert_eq!(tw.scat_psi_fp16.get(), expected_scat);
            assert_eq!(tw.allpass_delays, [3, 5, 7, 11]);
            assert_eq!(tw.flags & !TRANSPORT_FLAG_EXT_F32, 0);
            assert!(
                (fp16_to_f32(tw.scat_psi_fp16.get()) - core::f32::consts::FRAC_PI_4).abs() < 1.0e-3
            );
        }

        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Ultra, 48_000.0).unwrap();
        let mut out_l = [0.0_f32; 32];
        let mut out_r = [0.0_f32; 32];
        vm.process_block(&[0.0_f32; 32], &[0.0_f32; 32], &mut out_l, &mut out_r)
            .unwrap();
        assert!(out_l.iter().chain(&out_r).all(|v| v.to_bits() == 0));
    }
}
