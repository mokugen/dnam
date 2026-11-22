//! Reverse decoder converting validated Layer 0 `.dnam-bin` (`"DNBN"`) wire payloads back into
//! [`SonaModelF64`] (`ref[container.dnam.vm-binary]`, `ref[container.dnam.tier-prefix]`).

use dnam_kernels::{fp16_to_f32, unpack_wf_mix_and_rect_sign_fp16};
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, CoreQuadWire,
    DnamBinView, KautzSoaBlockWire, SONA_META_CONVOLVER_PREFIX_OVERRIDES, SONA_META_OS2X,
    SonaInterBlockTilePairWire, TRANSPORT_FLAG_EXT_F32, TransportQuadWire,
};
use zerocopy::FromZeros;

use crate::control::clamp::Phase1ClampOutput;
use crate::control::cp1::Cp1WireBundle;
use crate::error::TrainError;

use super::super::model::{
    CoreQuadF64, InterBlockTileF64, KautzPairF64, SonaModelF64, TierReadoutPrefixF64,
    TransportQuadF64,
};
use super::pack::readouts::PrefixOverrideWireDecode;

#[inline]
fn fp16_f64(bits: u16) -> f64 {
    f64::from(fp16_to_f32(bits))
}

fn unpack_core_quad_wire(w: &CoreQuadWire) -> CoreQuadF64 {
    let mut q = CoreQuadF64::default();
    for l in 0..4 {
        q.svf_g[l] = f64::from(w.svf_g[l].get());
        q.lru_alpha_rel[l] = f64::from(w.lru_alpha_rel[l].get());
        q.lru_alpha_atk[l] = f64::from(w.lru_alpha_atk[l].get());
        q.svf_k[l] = fp16_f64(w.svf_k_fp16[l].get());
        q.lru_kappa[l] = fp16_f64(w.lru_kappa_fp16[l].get());
        q.adaa_drive[l] = fp16_f64(w.adaa_drive_fp16[l].get());
        q.lru_bias_drive[l] = fp16_f64(w.lru_bias_drive_fp16[l].get());
        q.tone_lp[l] = fp16_f64(w.tone_lp_fp16[l].get());
        q.tone_bp[l] = fp16_f64(w.tone_bp_fp16[l].get());
        q.tone_hp[l] = fp16_f64(w.tone_hp_fp16[l].get());
        q.tone_ap[l] = fp16_f64(w.tone_ap_fp16[l].get());
        q.tone_dir[l] = fp16_f64(w.tone_dir_fp16[l].get());
        q.sag_depth[l] = fp16_f64(w.sag_depth_fp16[l].get());
        let (wf, rs) = unpack_wf_mix_and_rect_sign_fp16(w.wavefolder_mix_fp16[l].get());
        q.wavefolder_mix[l] = f64::from(wf);
        q.rect_sign[l] = f64::from(rs);
        q.in_mid_weight[l] = fp16_f64(w.in_mid_weight_fp16[l].get());
        q.in_side_weight[l] = fp16_f64(w.in_side_weight_fp16[l].get());
        q.lru_stereo_link[l] = fp16_f64(w.lru_stereo_link_fp16[l].get());
        q.lru_duhem_gamma[l] = fp16_f64(w.lru_duhem_gamma_fp16[l].get());
        q.lru_signed_mix[l] = fp16_f64(w.lru_signed_mix_fp16[l].get());
        q.glu_mix[l] = fp16_f64(w.glu_mix_fp16[l].get());
        q.svf_k_nl[l] = fp16_f64(w.svf_k_nl_fp16[l].get());
    }
    q
}

fn unpack_tile_wire(w: &SonaInterBlockTilePairWire) -> InterBlockTileF64 {
    let mut t = InterBlockTileF64::default();
    for s in 0..4 {
        for d in 0..4 {
            t.w_odd_4x4[s][d] = f64::from(w.w_odd_4x4[s][d].get());
            t.w_even_4x4[s][d] = fp16_f64(w.w_even_4x4_fp16[s][d].get());
            t.w_lru_4x4[s][d] = fp16_f64(w.w_lru_4x4_fp16[s][d].get());
        }
    }
    t
}

fn unpack_kautz_block(
    blk: &KautzSoaBlockWire,
    dst: &mut [KautzPairF64],
    pfx: &mut TierReadoutPrefixF64,
    base: usize,
) {
    for (i, kp) in dst.iter_mut().take(4).enumerate() {
        kp.r_low = f64::from(blk.r_low[i].get());
        kp.dr = f64::from(blk.dr[i].get());
        kp.theta = f64::from(blk.theta[i].get());
        kp.b_gamma = f64::from(blk.b_gamma[i].get());
        kp.c_re_l = f64::from(blk.c_re_l[i].get());
        kp.c_im_l = f64::from(blk.c_im_l[i].get());
        kp.c_re_r = f64::from(blk.c_re_r[i].get());
        kp.c_im_r = f64::from(blk.c_im_r[i].get());
        let idx = base + i;
        pfx.kautz_b_gamma[idx] = kp.b_gamma;
        pfx.kautz_c_re_l[idx] = kp.c_re_l;
        pfx.kautz_c_im_l[idx] = kp.c_im_l;
        pfx.kautz_c_re_r[idx] = kp.c_re_r;
        pfx.kautz_c_im_r[idx] = kp.c_im_r;
    }
}

fn unpack_transport_quad(w: &TransportQuadWire) -> TransportQuadF64 {
    let mut t = TransportQuadF64::default();
    for i in 0..4 {
        t.delay_samples[i] = f64::from(w.delay_samples[i].get());
        t.ramp_spans_samples[i] = f64::from(w.ramp_spans_samples[i].get());
        t.lfo_depths[i] = fp16_f64(w.lfo_depths_fp16[i].get());
        t.input_gains[i] = fp16_f64(w.input_gains_fp16[i].get());
        t.wet_tap_l[i] = fp16_f64(w.wet_tap_l_fp16[i].get());
        t.wet_tap_r[i] = fp16_f64(w.wet_tap_r_fp16[i].get());
        t.allpass_g[i] = fp16_f64(w.allpass_g_fp16[i].get());
    }
    t.allpass_delays = w.allpass_delays;
    t.g_loop = f64::from(w.g_loop.get());
    t.r_dc = f64::from(w.r_dc.get());
    t.damp_pole = f64::from(w.damp_pole.get());
    t.lfo_rate_hz = f64::from(w.lfo_rate_hz.get());
    t.ramp_rate_hz = f64::from(w.ramp_rate_hz.get());
    t.fb_psi = fp16_f64(w.fb_psi_fp16.get());
    t.scat_psi = fp16_f64(w.scat_psi_fp16.get());
    t.sigma_knee = fp16_f64(w.sigma_knee_fp16.get());
    t.freeze_mix = fp16_f64(w.freeze_mix_fp16.get());
    t.window_depth = fp16_f64(w.window_depth_fp16.get());
    t.ext_f32 = (w.flags & TRANSPORT_FLAG_EXT_F32) != 0;
    t
}

fn read_w_ro_fp16(
    src: &[zerocopy::byteorder::little_endian::U16],
    w_ro: &mut [[f64; 12]],
    lanes: usize,
) {
    for l in 0..lanes {
        for col in 0..12 {
            w_ro[l][col] = fp16_f64(src[l * 12 + col].get());
        }
    }
}

fn read_scalar_gains_fp16(
    src: &[zerocopy::byteorder::little_endian::U16],
    pfx: &mut TierReadoutPrefixF64,
) {
    pfx.pre_conv_u_gain = [fp16_f64(src[0].get()), fp16_f64(src[1].get())];
    pfx.pre_conv_hp_gain = [fp16_f64(src[2].get()), fp16_f64(src[3].get())];
    pfx.direct_s2_u_gain = [fp16_f64(src[4].get()), fp16_f64(src[5].get())];
    pfx.direct_s2_hp_gain = [fp16_f64(src[6].get()), fp16_f64(src[7].get())];
}

fn unpack_lite_prefix(a0: &Arena0LiteWire, kautz: &mut [KautzPairF64]) -> TierReadoutPrefixF64 {
    let mut pfx = TierReadoutPrefixF64::default();
    let ro = &a0.linear_readout_lite;
    pfx.pre_conv_u_gain = ro.pre_conv_u_gain.map(|v| f64::from(v.get()));
    pfx.pre_conv_hp_gain = ro.pre_conv_hp_gain.map(|v| f64::from(v.get()));
    pfx.direct_s2_u_gain = ro.direct_s2_u_gain.map(|v| f64::from(v.get()));
    pfx.direct_s2_hp_gain = ro.direct_s2_hp_gain.map(|v| f64::from(v.get()));
    read_w_ro_fp16(&ro.w_ro_fp16[..192], &mut pfx.w_ro, 16);
    for i in 0..32 {
        pfx.fir_l[i] = f64::from(a0.conv_fir_lite.fir_l[i].get());
        pfx.fir_r[i] = f64::from(a0.conv_fir_lite.fir_r[i].get());
    }
    unpack_kautz_block(&a0.conv_kautz_lite[0], &mut kautz[0..4], &mut pfx, 0);
    unpack_kautz_block(&a0.conv_kautz_lite[1], &mut kautz[4..8], &mut pfx, 4);
    pfx
}

fn unpack_full_prefix(
    a1: &Arena1FullDeltaWire,
    kautz: &mut [KautzPairF64],
    base_pfx: &TierReadoutPrefixF64,
    emit_pfx: bool,
) -> TierReadoutPrefixF64 {
    let mut pfx = base_pfx.clone();
    let ro = &a1.linear_readout_full;
    read_w_ro_fp16(&ro.w_ro_and_mods_fp16[..288], &mut pfx.w_ro, 24);
    read_scalar_gains_fp16(&ro.w_ro_and_mods_fp16[288..296], &mut pfx);
    for i in 0..16 {
        pfx.fir_l[32 + i] = f64::from(a1.conv_fir_full.fir_l[i].get());
        pfx.fir_r[32 + i] = f64::from(a1.conv_fir_full.fir_r[i].get());
    }
    unpack_kautz_block(&a1.conv_kautz_full, &mut kautz[8..12], &mut pfx, 8);
    if emit_pfx && let Some(ov) = PrefixOverrideWireDecode::full_from_bytes(&ro.prefix_override) {
        for i in 0..16 {
            pfx.fir_l[i] = f64::from(ov.fir_l[i].get());
            pfx.fir_r[i] = f64::from(ov.fir_r[i].get());
            pfx.fir_l[16 + i] = fp16_f64(ro.w_ro_and_mods_fp16[296 + i].get());
            pfx.fir_r[16 + i] = fp16_f64(ro.w_ro_and_mods_fp16[312 + i].get());
        }
        for m in 0..4 {
            pfx.kautz_c_re_l[m] = f64::from(ov.c_re_l[m].get());
            pfx.kautz_c_im_l[m] = f64::from(ov.c_im_l[m].get());
            pfx.kautz_c_re_r[m] = f64::from(ov.c_re_r[m].get());
            pfx.kautz_c_im_r[m] = f64::from(ov.c_im_r[m].get());
        }
        for m in 0..8 {
            pfx.kautz_b_gamma[m] = fp16_f64(ro.w_ro_and_mods_fp16[328 + m].get());
        }
        for idx in 0..4 {
            let off = 336 + idx * 4;
            pfx.kautz_c_re_l[4 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off].get());
            pfx.kautz_c_im_l[4 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 1].get());
            pfx.kautz_c_re_r[4 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 2].get());
            pfx.kautz_c_im_r[4 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 3].get());
        }
    }
    pfx
}

fn unpack_pro_prefix(
    a2: &Arena2ProDeltaWire,
    kautz: &mut [KautzPairF64],
    base_pfx: &TierReadoutPrefixF64,
    emit_pfx: bool,
) -> TierReadoutPrefixF64 {
    let mut pfx = base_pfx.clone();
    let ro = &a2.linear_readout_pro;
    read_w_ro_fp16(&ro.w_ro_and_mods_fp16[..384], &mut pfx.w_ro, 32);
    read_scalar_gains_fp16(&ro.w_ro_and_mods_fp16[384..392], &mut pfx);
    for i in 0..48 {
        pfx.fir_l[48 + i] = f64::from(a2.conv_fir_pro.fir_l[i].get());
        pfx.fir_r[48 + i] = f64::from(a2.conv_fir_pro.fir_r[i].get());
    }
    for b in 0..3 {
        let base = 12 + b * 4;
        unpack_kautz_block(
            &a2.conv_kautz_pro[b],
            &mut kautz[base..base + 4],
            &mut pfx,
            base,
        );
    }
    if emit_pfx && let Some(ov) = PrefixOverrideWireDecode::pro_from_bytes(&ro.prefix_override) {
        for i in 0..32 {
            pfx.fir_l[i] = f64::from(ov.fir_l[i].get());
            pfx.fir_r[i] = f64::from(ov.fir_r[i].get());
        }
        for i in 0..16 {
            pfx.fir_l[32 + i] = fp16_f64(ro.w_ro_and_mods_fp16[392 + i].get());
            pfx.fir_r[32 + i] = fp16_f64(ro.w_ro_and_mods_fp16[408 + i].get());
        }
        for m in 0..8 {
            pfx.kautz_c_re_l[m] = f64::from(ov.c_re_l[m].get());
            pfx.kautz_c_im_l[m] = f64::from(ov.c_im_l[m].get());
            pfx.kautz_c_re_r[m] = f64::from(ov.c_re_r[m].get());
            pfx.kautz_c_im_r[m] = f64::from(ov.c_im_r[m].get());
        }
        for m in 0..12 {
            pfx.kautz_b_gamma[m] = fp16_f64(ro.w_ro_and_mods_fp16[424 + m].get());
        }
        for idx in 0..4 {
            let off = 436 + idx * 4;
            pfx.kautz_c_re_l[8 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off].get());
            pfx.kautz_c_im_l[8 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 1].get());
            pfx.kautz_c_re_r[8 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 2].get());
            pfx.kautz_c_im_r[8 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 3].get());
        }
    }
    pfx
}

fn unpack_ultra_prefix(
    a3: &Arena3UltraDeltaWire,
    kautz: &mut [KautzPairF64],
    base_pfx: &TierReadoutPrefixF64,
    emit_pfx: bool,
) -> TierReadoutPrefixF64 {
    let mut pfx = base_pfx.clone();
    let ro = &a3.linear_readout_ultra;
    read_w_ro_fp16(&ro.w_ro_and_mods_fp16[..480], &mut pfx.w_ro, 40);
    read_scalar_gains_fp16(&ro.w_ro_and_mods_fp16[480..488], &mut pfx);
    for i in 0..32 {
        pfx.fir_l[96 + i] = f64::from(a3.conv_fir_ultra.fir_l[i].get());
        pfx.fir_r[96 + i] = f64::from(a3.conv_fir_ultra.fir_r[i].get());
    }
    for b in 0..2 {
        let base = 24 + b * 4;
        unpack_kautz_block(
            &a3.conv_kautz_ultra[b],
            &mut kautz[base..base + 4],
            &mut pfx,
            base,
        );
    }
    if emit_pfx && let Some(ov) = PrefixOverrideWireDecode::ultra_from_bytes(&ro.prefix_override) {
        for i in 0..48 {
            pfx.fir_l[i] = f64::from(ov.fir_l[i].get());
            pfx.fir_r[i] = f64::from(ov.fir_r[i].get());
            pfx.fir_l[48 + i] = fp16_f64(ro.w_ro_and_mods_fp16[488 + i].get());
            pfx.fir_r[48 + i] = fp16_f64(ro.w_ro_and_mods_fp16[536 + i].get());
        }
        for m in 0..12 {
            pfx.kautz_c_re_l[m] = f64::from(ov.c_re_l[m].get());
            pfx.kautz_c_im_l[m] = f64::from(ov.c_im_l[m].get());
            pfx.kautz_c_re_r[m] = f64::from(ov.c_re_r[m].get());
            pfx.kautz_c_im_r[m] = f64::from(ov.c_im_r[m].get());
        }
        for m in 0..24 {
            pfx.kautz_b_gamma[m] = fp16_f64(ro.w_ro_and_mods_fp16[584 + m].get());
        }
        for idx in 0..12 {
            let off = 608 + idx * 4;
            pfx.kautz_c_re_l[12 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off].get());
            pfx.kautz_c_im_l[12 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 1].get());
            pfx.kautz_c_re_r[12 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 2].get());
            pfx.kautz_c_im_r[12 + idx] = fp16_f64(ro.w_ro_and_mods_fp16[off + 3].get());
        }
    }
    pfx
}

fn apply_highest_prefix_to_model(model: &mut SonaModelF64, pfx: &TierReadoutPrefixF64) {
    model.pre_conv_u_gain = pfx.pre_conv_u_gain;
    model.pre_conv_hp_gain = pfx.pre_conv_hp_gain;
    model.direct_s2_u_gain = pfx.direct_s2_u_gain;
    model.direct_s2_hp_gain = pfx.direct_s2_hp_gain;
    model.w_ro = pfx.w_ro;
    model.fir_l = pfx.fir_l;
    model.fir_r = pfx.fir_r;
    for (k, kp) in model.kautz.iter_mut().enumerate() {
        kp.b_gamma = pfx.kautz_b_gamma[k];
        kp.c_re_l = pfx.kautz_c_re_l[k];
        kp.c_im_l = pfx.kautz_c_im_l[k];
        kp.c_re_r = pfx.kautz_c_re_r[k];
        kp.c_im_r = pfx.kautz_c_im_r[k];
    }
}

/// Decodes a validated Layer 0 `.dnam-bin` (`"DNBN"`) payload back into a [`SonaModelF64`]
/// (`ref[container.dnam.vm-binary]`, `ref[container.dnam.tier-prefix]`).
pub fn decode_dnam_bin_to_model(dnam_bin: &[u8]) -> Result<SonaModelF64, TrainError> {
    let view = DnamBinView::parse(dnam_bin)?;
    let a0 = view.arena0;
    let sr = f64::from(view.header.reference_sample_rate_hz.get());
    let mut model = SonaModelF64::new(sr);
    model.ro_pole_48k = f64::from(view.runtime_meta.ro_pole_48k.get());

    let meta = a0.crossbar.sona_meta_flags.get();
    model.oversample_upper_tiers_2x = (meta & SONA_META_OS2X) != 0;
    let emit_pfx = (meta & SONA_META_CONVOLVER_PREFIX_OVERRIDES) != 0;
    model.os_up_quad_coeff = f64::from(a0.crossbar.os_up_quad_coeff.get());
    model.os_dec_c_prev1 = f64::from(a0.crossbar.os_dec_c_prev1.get());
    model.os_dec_c0 = f64::from(a0.crossbar.os_dec_c0.get());

    for i in 0..4 {
        model.sense_svf_g[i] = f64::from(a0.crossbar.sense_svf_g[i].get());
        model.sense_lru_rel[i] = f64::from(a0.crossbar.sense_lru_alpha_rel[i].get());
        model.sense_lru_atk[i] = f64::from(a0.crossbar.sense_lru_alpha_atk[i].get());
        model.sense_svf_k[i] = fp16_f64(a0.crossbar.sense_svf_k_fp16[i].get());
        model.sense_lru_kappa[i] = fp16_f64(a0.crossbar.sense_lru_kappa_fp16[i].get());
        model.sense_vca_w[i] = fp16_f64(a0.crossbar.sense_vca_w_fp16[i].get());
        model.sense_in_l[i] = fp16_f64(a0.crossbar.sense_in_l_fp16[i].get());
        model.sense_in_r[i] = fp16_f64(a0.crossbar.sense_in_r_fp16[i].get());
        model.sense_tone_lp[i] = fp16_f64(a0.crossbar.sense_tone_lp_fp16[i].get());
        model.sense_tone_bp[i] = fp16_f64(a0.crossbar.sense_tone_bp_fp16[i].get());
        model.sense_tone_hp[i] = fp16_f64(a0.crossbar.sense_tone_hp_fp16[i].get());
        model.tier_gain_trim[i] = f64::from(a0.crossbar.tier_gain_trim[i].get());
        model.phasor_freq_hz[i] = f64::from(a0.crossbar.phasor_freq_hz[i].get());
    }
    model.crossbar_op_gains = [
        fp16_f64(a0.crossbar.crossbar_op_gains_fp16[0].get()),
        fp16_f64(a0.crossbar.crossbar_op_gains_fp16[1].get()),
    ];
    model.op_n_pre_gamma_bc = f64::from(a0.crossbar.op_n_pre_base.gamma_bc.get());
    model.op_n_pre_zoh_hold_depth = f64::from(a0.crossbar.op_n_pre_base.zoh_hold_depth.get());
    model.op_n_pre_beta_q = f64::from(a0.crossbar.op_n_pre_base.beta_q.get());
    for i in 0..3 {
        for j in 0..4 {
            model.carrier_basis_w[i][j] = fp16_f64(a0.crossbar.carrier_basis_w_fp16[i][j].get());
        }
    }

    for q in 0..4 {
        model.core_quads[q] = unpack_core_quad_wire(&a0.core_quads[q]);
        for l in 0..4 {
            model.op_p_to_core[q][l] =
                fp16_f64(a0.linear_readout_lite.stage_latent_mods_fp16[80 + q * 4 + l].get());
            model.phasor_to_core[q][l] =
                fp16_f64(a0.linear_readout_lite.w_ro_fp16[192 + q * 4 + l].get());
        }
        model.opt_to_lr[q] = [
            f64::from(a0.control_bus_lite.rho_head_and_ctrl_tail[24 + q * 2].get()),
            f64::from(a0.control_bus_lite.rho_head_and_ctrl_tail[24 + q * 2 + 1].get()),
        ];
    }
    for i in 0..3 {
        model.tiles[i] = unpack_tile_wire(&a0.tiles[i]);
    }
    model.transport_quads[0] = unpack_transport_quad(&a0.transport_lite);
    let raw_dry_mix = a0.control_bus_lite.tank_dry_mix_fp16.get();
    model.tank_dry_mix = if raw_dry_mix == 0 {
        0.0
    } else {
        fp16_f64(raw_dry_mix)
    };

    if let Some(xbar) = view
        .arena3
        .map(|a| &a.crossbar_ultra)
        .or_else(|| view.arena2.map(|a| &a.crossbar_pro))
        .or_else(|| view.arena1.map(|a| &a.crossbar_full))
    {
        for q in 0..10 {
            for l in 0..4 {
                model.op_p_to_core[q][l] = fp16_f64(xbar.op_p_to_core_fp16[q][l].get());
                model.phasor_to_core[q][l] = fp16_f64(xbar.phasor_to_core_fp16[q][l].get());
            }
        }
        for s in 0..16 {
            model.opt_to_lr[s][0] = fp16_f64(xbar.opt_to_lr_fp16[s][0].get());
            model.opt_to_lr[s][1] = fp16_f64(xbar.opt_to_lr_fp16[s][1].get());
        }
        for k in 0..4 {
            model.phasor_stereo_phase[k] = fp16_f64(xbar.phasor_stereo_phase_fp16[k].get());
        }
    }

    if let Some(a1) = view.arena1 {
        for q in 0..2 {
            model.core_quads[4 + q] = unpack_core_quad_wire(&a1.core_quads[q]);
        }
        for i in 0..10 {
            model.tiles[3 + i] = unpack_tile_wire(&a1.tiles[i]);
        }
        model.transport_quads[1] = unpack_transport_quad(&a1.transport_full);
    }
    if let Some(a2) = view.arena2 {
        for q in 0..2 {
            model.core_quads[6 + q] = unpack_core_quad_wire(&a2.core_quads[q]);
        }
        for i in 0..10 {
            model.tiles[13 + i] = unpack_tile_wire(&a2.tiles[i]);
        }
        model.transport_quads[2] = unpack_transport_quad(&a2.transport_pro[0]);
    }
    if let Some(a3) = view.arena3 {
        for q in 0..2 {
            model.core_quads[8 + q] = unpack_core_quad_wire(&a3.core_quads[q]);
        }
        for i in 0..12 {
            model.tiles[23 + i] = unpack_tile_wire(&a3.tiles[i]);
        }
        model.transport_quads[3] = unpack_transport_quad(&a3.transport_ultra[0]);
    }

    let pfx0 = unpack_lite_prefix(a0, &mut model.kautz);
    let pfx1 = view
        .arena1
        .map(|a1| unpack_full_prefix(a1, &mut model.kautz, &pfx0, emit_pfx));
    let pfx2 = view.arena2.map(|a2| {
        unpack_pro_prefix(
            a2,
            &mut model.kautz,
            pfx1.as_ref().unwrap_or(&pfx0),
            emit_pfx,
        )
    });
    let pfx3 = view.arena3.map(|a3| {
        unpack_ultra_prefix(
            a3,
            &mut model.kautz,
            pfx2.as_ref().or(pfx1.as_ref()).unwrap_or(&pfx0),
            emit_pfx,
        )
    });

    if let Some(ref p) = pfx3 {
        apply_highest_prefix_to_model(&mut model, p);
    } else if let Some(ref p) = pfx2 {
        apply_highest_prefix_to_model(&mut model, p);
    } else if let Some(ref p) = pfx1 {
        apply_highest_prefix_to_model(&mut model, p);
    } else {
        apply_highest_prefix_to_model(&mut model, &pfx0);
    }

    model.prefix_override_needed = [
        false,
        emit_pfx && pfx1.is_some(),
        emit_pfx && pfx2.is_some(),
        emit_pfx && pfx3.is_some(),
    ];
    if emit_pfx {
        model.tier_overrides[0] = Some(Box::new(pfx0));
        model.tier_overrides[1] = pfx1.map(Box::new);
        model.tier_overrides[2] = pfx2.map(Box::new);
        model.tier_overrides[3] = pfx3.map(Box::new);
    }

    model.active_fir_taps = model
        .fir_l
        .iter()
        .zip(&model.fir_r)
        .rposition(|(&l, &r)| l != 0.0 || r != 0.0)
        .map_or(0, |idx| idx + 1);
    model.active_kautz_pairs = model
        .kautz
        .iter()
        .rposition(|k| {
            k.r_low != 0.0
                || k.dr != 0.0
                || k.b_gamma != 0.0
                || k.c_re_l != 0.0
                || k.c_im_l != 0.0
                || k.c_re_r != 0.0
                || k.c_im_r != 0.0
        })
        .map_or(0, |idx| idx + 1);

    if !view.control_descriptors.is_empty() {
        model.control_audit = Some(Phase1ClampOutput {
            wire_descriptors: view.control_descriptors.to_vec(),
            locked_controls: view
                .control_descriptors
                .iter()
                .map(|d| d.ctrl_kind == 0 && d.lobatto_degree == 0)
                .collect(),
            safe_ranges: view
                .control_descriptors
                .iter()
                .map(|d| {
                    (
                        f64::from(d.safe_min_base.get()),
                        f64::from(d.safe_max_base.get()),
                    )
                })
                .collect(),
            audited_corners: 0,
            warnings: Vec::new(),
        });
    }
    let cb = &a0.control_bus_lite;
    let inline_cnt = usize::from(cb.cp1_dir.cp1_dir_count.get());
    let has_cb = inline_cnt > 0
        || !view.cp1_overflow.is_empty()
        || !view.tsyn_entries.is_empty()
        || view.cext.is_some()
        || cb.w_ctrl.iter().flatten().any(|v| v.get() != 0.0)
        || cb.u_int_base.iter().flatten().any(|v| v.get() != 0.0);
    if has_cb {
        model.control_bus = Some(Cp1WireBundle {
            arena0_cb: *cb,
            arena3_cb: view
                .arena3
                .map_or_else(ZeroedExt::zeroed_arena3_cb, |w| w.control_bus_ultra_delta),
            cext: view.cext.copied(),
            tsyn: view.tsyn_entries.to_vec(),
            cp1_overflow: view.cp1_overflow.to_vec(),
            contributor_count: inline_cnt.max(view.control_descriptors.len()),
            distinct_controls: view.control_descriptors.len(),
        });
    }

    Ok(model)
}

struct ZeroedExt;
impl ZeroedExt {
    #[inline]
    fn zeroed_arena3_cb() -> dnam_vm::bin::Arena3ControlBusWire {
        dnam_vm::bin::Arena3ControlBusWire::new_zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};
    use dnam_vm::math::DefaultMathBackend;
    use dnam_vm::{SonaVm, Tier};

    #[test]
    fn test_decode_dnam_bin_roundtrip_and_invalid_input() {
        assert!(decode_dnam_bin_to_model(&[0u8; 32]).is_err());

        let mut model = SonaModelF64::new(48_000.0);
        model.ro_pole_48k = 0.85;
        model.oversample_upper_tiers_2x = true;
        model.core_quads[0].svf_k_nl = [0.25, 0.5, 0.75, 1.0];
        model.core_quads[0].adaa_drive = [1.5, -1.5, 0.5, 0.0];
        model.core_quads[0].wavefolder_mix = [0.25, 0.0, 0.5, 0.125];
        model.core_quads[0].rect_sign = [-1.0, 1.0, -1.0, 1.0];
        model.core_quads[4].tone_dir = [0.2, 0.3, 0.4, 0.5];
        model.direct_s2_u_gain = [0.65, -0.45];
        model.fir_l[0] = 0.25;
        model.fir_r[3] = -0.125;
        model.kautz[0].r_low = 0.72;
        model.kautz[0].theta = 0.4;
        model.kautz[0].b_gamma = 0.5;
        model.kautz[0].c_re_l = 0.25;
        model.transport_quads[0].scat_psi = 0.5;
        model.transport_quads[0].allpass_delays = [3, 5, 7, 11];
        model.prefix_override_needed = [false, true, true, true];
        model.tier_overrides[0] = Some(Box::new(TierReadoutPrefixF64 {
            direct_s2_u_gain: [0.55, -0.35],
            ..TierReadoutPrefixF64::default()
        }));

        let bin1 = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap();
        let decoded = decode_dnam_bin_to_model(&bin1).unwrap();
        assert!((decoded.ro_pole_48k - 0.85).abs() < 1.0e-6);
        assert!(decoded.oversample_upper_tiers_2x);
        assert_eq!(decoded.core_quads[0].rect_sign, [-1.0, 1.0, -1.0, 1.0]);
        assert!((decoded.fir_l[0] - 0.25).abs() < 1.0e-6);
        assert!((decoded.kautz[0].r_low - 0.72).abs() < 1.0e-6);

        let bin2 = encode_model_to_dnam_bin(&decoded, Pass5EmitConfig::default()).unwrap();
        assert_eq!(bin1, bin2);

        let mut vm1 =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin1, Tier::Ultra, 48_000.0)
                .unwrap();
        let mut vm2 =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin2, Tier::Ultra, 48_000.0)
                .unwrap();
        let x: Vec<f32> = (0..64).map(|i| 0.2 * ((i as f32) * 0.17).sin()).collect();
        let (mut l1, mut r1, mut l2, mut r2) =
            (vec![0.0; 64], vec![0.0; 64], vec![0.0; 64], vec![0.0; 64]);
        vm1.process_block(&x, &x, &mut l1, &mut r1).unwrap();
        vm2.process_block(&x, &x, &mut l2, &mut r2).unwrap();
        assert_eq!(l1, l2);
        assert_eq!(r1, r2);

        let stripped_lite = dnam_vm::bin::strip_dnam_bin_to_tier(&bin1, Tier::Lite).unwrap();
        let dec_lite = decode_dnam_bin_to_model(stripped_lite).unwrap();
        assert!((dec_lite.direct_s2_u_gain[0] - 0.55).abs() < 1.0e-2);
    }
}
