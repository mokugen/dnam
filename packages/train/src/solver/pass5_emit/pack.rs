//! `zerocopy 0.8` `Arena0..3`, `PrefixOverride*`, `SonaCrossbarBlockWire`, and
//! `.dnam-bin` serialization for Pass 5 (`ref[container.dnam.vm-binary]`).

use core::mem::{align_of, size_of};

use dnam_kernels::{
    MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, MIN_TPT_G, TANK_MAX_G_LIN_F32, f32_to_fp16,
    fp16_to_f32, pack_wf_mix_and_rect_sign_fp16,
};
use dnam_vm::DnamBinBuildInput;
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    ConvolverFir16Wire, ConvolverFir32Wire, ConvolverFir48Wire, CoreQuadWire,
    DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, KautzSoaBlockWire, LinearReadoutFullWire,
    LinearReadoutLiteWire, LinearReadoutProWire, LinearReadoutUltraWire, PIPE_CANONICAL_V3_SONA,
    SONA_CELL_MAGIC, SONA_META_CONVOLVER_PREFIX_OVERRIDES, SONA_META_OS_EXT_PRESENT,
    SONA_META_OS2X, SONA_META_SENSORY_QUAD, SONA_META_TIER_OS_EXPLICIT, SONA_META_TRANSPORT_LFO,
    SONA_META_TRANSPORT_TAP_GAINS, SonaCrossbarBlockWire, SonaInterBlockTilePairWire,
    TRANSPORT_FLAG_EXT_F32, TransportQuadWire, encode_dnam_bin,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use sha2::{Digest, Sha256};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::control::stability::{MAX_RO_POLE_48K_F32, enforce_arena0_stability_invariants};
use crate::error::TrainError;

use super::super::model::{
    CoreQuadF64, InterBlockTileF64, KautzPairF64, SonaModelF64, TierReadoutPrefixF64,
    TransportQuadF64,
};
use super::Pass5EmitConfig;

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
struct PrefixOverrideFullEncodeWire {
    fir_l: [F32; 16],
    fir_r: [F32; 16],
    c_re_l: [F32; 4],
    c_im_l: [F32; 4],
    c_re_r: [F32; 4],
    c_im_r: [F32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
struct PrefixOverrideProEncodeWire {
    fir_l: [F32; 32],
    fir_r: [F32; 32],
    c_re_l: [F32; 8],
    c_im_l: [F32; 8],
    c_re_r: [F32; 8],
    c_im_r: [F32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
struct PrefixOverrideUltraEncodeWire {
    fir_l: [F32; 48],
    fir_r: [F32; 48],
    c_re_l: [F32; 12],
    c_im_l: [F32; 12],
    c_re_r: [F32; 12],
    c_im_r: [F32; 12],
}

const _: () = assert!(size_of::<PrefixOverrideFullEncodeWire>() == 192);
const _: () = assert!(align_of::<PrefixOverrideFullEncodeWire>() == 1);
const _: () = assert!(size_of::<PrefixOverrideProEncodeWire>() == 384);
const _: () = assert!(align_of::<PrefixOverrideProEncodeWire>() == 1);
const _: () = assert!(size_of::<PrefixOverrideUltraEncodeWire>() == 576);
const _: () = assert!(align_of::<PrefixOverrideUltraEncodeWire>() == 1);

const LITE_OPT_TO_LR_OFFSET: usize = 24;
const STAGE_LATENT_OP_P_TO_CORE_OFFSET: usize = 8 + 5 * 8 + 2 * (4 * 4);
const _: () = assert!(STAGE_LATENT_OP_P_TO_CORE_OFFSET == 80);

#[inline]
pub(super) fn sanitize_clamp_f32(v: f64, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        (v as f32).clamp(lo, hi)
    } else {
        fallback
    }
}

#[inline]
fn fp16_clamped(v: f64, lo: f32, hi: f32, fallback: f32) -> U16 {
    U16::new(f32_to_fp16(sanitize_clamp_f32(v, lo, hi, fallback)))
}

fn pack_core_quad_wire(q: &CoreQuadF64) -> CoreQuadWire {
    let mut w = CoreQuadWire::new_zeroed();
    for l in 0..4 {
        w.svf_g[l] = F32::new(sanitize_clamp_f32(q.svf_g[l], MIN_TPT_G, 64.0, 0.125));
        w.lru_alpha_rel[l] = F32::new(sanitize_clamp_f32(
            q.lru_alpha_rel[l],
            0.0,
            MAX_PARALRU_R_F32,
            0.995,
        ));
        w.lru_alpha_atk[l] = F32::new(sanitize_clamp_f32(q.lru_alpha_atk[l], 0.0, 1.0, 0.25));
        w.svf_k_fp16[l] = fp16_clamped(q.svf_k[l], 0.0, 8.0, core::f32::consts::FRAC_1_SQRT_2);
        w.lru_kappa_fp16[l] = fp16_clamped(q.lru_kappa[l], 0.0, 1.0, 0.0);
        w.adaa_drive_fp16[l] = fp16_clamped(q.adaa_drive[l], -32.0, 32.0, 0.0);
        w.lru_bias_drive_fp16[l] = fp16_clamped(q.lru_bias_drive[l], -16.0, 16.0, 0.0);
        w.tone_lp_fp16[l] = fp16_clamped(q.tone_lp[l], -16.0, 16.0, 0.0);
        w.tone_bp_fp16[l] = fp16_clamped(q.tone_bp[l], -16.0, 16.0, 0.0);
        w.tone_hp_fp16[l] = fp16_clamped(q.tone_hp[l], -16.0, 16.0, 0.0);
        w.tone_ap_fp16[l] = fp16_clamped(q.tone_ap[l], -16.0, 16.0, 0.0);
        w.tone_dir_fp16[l] = fp16_clamped(q.tone_dir[l], -16.0, 16.0, 0.0);
        w.sag_depth_fp16[l] = fp16_clamped(q.sag_depth[l], -8.0, 8.0, 0.0);
        let wf = sanitize_clamp_f32(q.wavefolder_mix[l], 0.0, 1.0, 0.0);
        let rs = if q.rect_sign[l].is_finite() && q.rect_sign[l] < 0.0 {
            -1.0
        } else {
            1.0
        };
        w.wavefolder_mix_fp16[l] = U16::new(pack_wf_mix_and_rect_sign_fp16(wf, rs));
        w.in_mid_weight_fp16[l] = fp16_clamped(q.in_mid_weight[l], -16.0, 16.0, 1.0);
        w.in_side_weight_fp16[l] = fp16_clamped(q.in_side_weight[l], -16.0, 16.0, 1.0);
        w.lru_stereo_link_fp16[l] = fp16_clamped(q.lru_stereo_link[l], 0.0, 1.0, 0.0);
        w.lru_duhem_gamma_fp16[l] =
            fp16_clamped(q.lru_duhem_gamma[l], 0.0, MAX_PARALRU_DUHEM_GAMMA_F32, 0.0);
        w.lru_signed_mix_fp16[l] = fp16_clamped(q.lru_signed_mix[l], 0.0, 1.0, 0.0);
        w.glu_mix_fp16[l] = fp16_clamped(q.glu_mix[l], -4.0, 4.0, 0.0);
    }
    w
}

fn pack_tile_wire(t: &InterBlockTileF64) -> SonaInterBlockTilePairWire {
    let mut w = SonaInterBlockTilePairWire::new_zeroed();
    for s in 0..4 {
        for d in 0..4 {
            w.w_odd_4x4[s][d] = F32::new(sanitize_clamp_f32(t.w_odd_4x4[s][d], -8.0, 8.0, 0.0));
            w.w_even_4x4_fp16[s][d] = fp16_clamped(t.w_even_4x4[s][d], -8.0, 8.0, 0.0);
            w.w_lru_4x4_fp16[s][d] = fp16_clamped(t.w_lru_4x4[s][d], -8.0, 8.0, 0.0);
        }
    }
    w
}

fn pack_kautz_block(
    pairs: &[KautzPairF64],
    override_src: Option<&TierReadoutPrefixF64>,
    base_pair_idx: usize,
) -> KautzSoaBlockWire {
    let mut b = KautzSoaBlockWire::new_zeroed();
    for (i, p) in pairs.iter().take(4).enumerate() {
        let r = sanitize_clamp_f32(p.r_low, 0.0, 0.9995, 0.0);
        let dr = sanitize_clamp_f32(p.dr, 0.0, (0.9995 - r).max(0.0), 0.0);
        let pair_idx = base_pair_idx + i;
        let (bg, cl, cil, cr, cir) = if let Some(ov) = override_src {
            (
                ov.kautz_b_gamma[pair_idx],
                ov.kautz_c_re_l[pair_idx],
                ov.kautz_c_im_l[pair_idx],
                ov.kautz_c_re_r[pair_idx],
                ov.kautz_c_im_r[pair_idx],
            )
        } else {
            (p.b_gamma, p.c_re_l, p.c_im_l, p.c_re_r, p.c_im_r)
        };
        b.r_low[i] = F32::new(r);
        b.dr[i] = F32::new(dr);
        b.theta[i] = F32::new(sanitize_clamp_f32(
            p.theta,
            0.0,
            core::f32::consts::PI - 1.0e-4,
            0.0,
        ));
        b.b_gamma[i] = F32::new(sanitize_clamp_f32(bg, 0.0, 8.0, 0.0));
        b.c_re_l[i] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
        b.c_im_l[i] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
        b.c_re_r[i] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
        b.c_im_r[i] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
    }
    b
}

fn pack_transport_quad_wire(t: &TransportQuadF64) -> TransportQuadWire {
    let mut w = TransportQuadWire::new_zeroed();
    let mut need_ext_f32 = t.ext_f32;
    for i in 0..4 {
        let d = sanitize_clamp_f32(t.delay_samples[i], 5.0, 48_000.0, 16.0);
        if d > 120.0 {
            need_ext_f32 = true;
        }
        w.delay_samples[i] = F32::new(d);
        w.ramp_spans_samples[i] = F32::new(sanitize_clamp_f32(
            t.ramp_spans_samples[i],
            0.0,
            48_000.0,
            0.0,
        ));
        w.lfo_depths_fp16[i] = fp16_clamped(t.lfo_depths[i], 0.0, 512.0, 0.0);
        w.input_gains_fp16[i] = fp16_clamped(t.input_gains[i], -2.0, 2.0, 0.0);
        w.wet_tap_l_fp16[i] = fp16_clamped(t.wet_tap_l[i], -2.0, 2.0, 0.0);
        w.wet_tap_r_fp16[i] = fp16_clamped(t.wet_tap_r[i], -2.0, 2.0, 0.0);
        w.allpass_g_fp16[i] = fp16_clamped(t.allpass_g[i], -0.75, 0.75, 0.0);
    }
    w.allpass_delays = if t.schroeder_enable {
        t.allpass_delays
    } else {
        [0; 4]
    };
    w.g_loop = F32::new(sanitize_clamp_f32(t.g_loop, 0.0, TANK_MAX_G_LIN_F32, 0.0));
    w.r_dc = F32::new(TANK_DC_BLOCKER_48K_F32);
    w.damp_pole = F32::new(sanitize_clamp_f32(t.damp_pole, 0.0, 0.9995, 0.0));
    w.lfo_rate_hz = F32::new(sanitize_clamp_f32(t.lfo_rate_hz, 0.0, 16.0, 0.0));
    w.ramp_rate_hz = F32::new(sanitize_clamp_f32(t.ramp_rate_hz, -16.0, 16.0, 0.0));
    w.fb_psi_fp16 = fp16_clamped(t.fb_psi, -core::f32::consts::PI, core::f32::consts::PI, 0.0);
    w.sigma_knee_fp16 = fp16_clamped(t.sigma_knee, 1.0e-3, 0.984_375, 0.984_375);
    w.freeze_mix_fp16 = fp16_clamped(t.freeze_mix, 0.0, 1.0, 0.0);
    w.window_depth_fp16 = fp16_clamped(t.window_depth, 0.0, 1.0, 0.0);
    w.flags = if need_ext_f32 {
        TRANSPORT_FLAG_EXT_F32
    } else {
        0
    };
    w
}

fn pack_crossbar_block_wire(model: &SonaModelF64) -> SonaCrossbarBlockWire {
    let mut blk = SonaCrossbarBlockWire::new_zeroed();
    for q in 0..10 {
        for l in 0..4 {
            blk.op_p_to_core_fp16[q][l] = fp16_clamped(model.op_p_to_core[q][l], -8.0, 8.0, 0.0);
            blk.phasor_to_core_fp16[q][l] =
                fp16_clamped(model.phasor_to_core[q][l], -8.0, 8.0, 0.0);
        }
    }
    for s in 0..16 {
        blk.opt_to_lr_fp16[s][0] = fp16_clamped(model.opt_to_lr[s][0], -4.0, 4.0, 0.0);
        blk.opt_to_lr_fp16[s][1] = fp16_clamped(model.opt_to_lr[s][1], -4.0, 4.0, 0.0);
    }
    for k in 0..4 {
        blk.phasor_stereo_phase_fp16[k] = fp16_clamped(
            model.phasor_stereo_phase[k],
            -core::f32::consts::PI,
            core::f32::consts::PI,
            0.0,
        );
    }
    blk
}

fn write_w_ro_fp16(w_ro: &[[f64; 12]], dst: &mut [U16], lanes: usize) {
    for l in 0..lanes {
        let base = l * 12;
        for col in 0..12 {
            dst[base + col] = fp16_clamped(w_ro[l][col], -64.0, 64.0, 0.0);
        }
    }
}

fn write_scalar_ro_gains_fp16(
    pre_u: [f64; 2],
    pre_hp: [f64; 2],
    dir_u: [f64; 2],
    dir_hp: [f64; 2],
    dst: &mut [U16],
) {
    let gains = [
        pre_u[0], pre_u[1], pre_hp[0], pre_hp[1], dir_u[0], dir_u[1], dir_hp[0], dir_hp[1],
    ];
    for (i, &g) in gains.iter().enumerate() {
        dst[i] = fp16_clamped(g, -64.0, 64.0, 0.0);
    }
}

fn pack_kautz_b_and_c(
    src: Option<&TierReadoutPrefixF64>,
    model: &SonaModelF64,
    pair_idx: usize,
) -> (f64, f64, f64, f64, f64) {
    if let Some(p) = src {
        (
            p.kautz_b_gamma[pair_idx],
            p.kautz_c_re_l[pair_idx],
            p.kautz_c_im_l[pair_idx],
            p.kautz_c_re_r[pair_idx],
            p.kautz_c_im_r[pair_idx],
        )
    } else {
        let kp = &model.kautz[pair_idx];
        (kp.b_gamma, kp.c_re_l, kp.c_im_l, kp.c_re_r, kp.c_im_r)
    }
}

fn pack_tier_readouts(
    model: &SonaModelF64,
    emit_prefix_overrides: bool,
    a0: &mut Arena0LiteWire,
    a1: &mut Arena1FullDeltaWire,
    a2: &mut Arena2ProDeltaWire,
    a3: &mut Arena3UltraDeltaWire,
) {
    let snap_fp16 = |v: f64| F32::new(fp16_to_f32(fp16_clamped(v, -64.0, 64.0, 0.0).get()));
    let lite_src = model.tier_overrides[0].as_deref();
    let mut ro_lite = LinearReadoutLiteWire::new_zeroed();
    ro_lite.pre_conv_u_gain = lite_src
        .map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain)
        .map(snap_fp16);
    ro_lite.pre_conv_hp_gain = lite_src
        .map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain)
        .map(snap_fp16);
    ro_lite.direct_s2_u_gain = lite_src
        .map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain)
        .map(snap_fp16);
    ro_lite.direct_s2_hp_gain = lite_src
        .map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain)
        .map(snap_fp16);
    let lite_w_ro = lite_src.map_or(&model.w_ro, |p| &p.w_ro);
    write_w_ro_fp16(lite_w_ro, &mut ro_lite.w_ro_fp16[..192], 16);
    for q in 0..4 {
        for l in 0..4 {
            ro_lite.stage_latent_mods_fp16[STAGE_LATENT_OP_P_TO_CORE_OFFSET + q * 4 + l] =
                fp16_clamped(model.op_p_to_core[q][l], -8.0, 8.0, 0.0);
            ro_lite.w_ro_fp16[192 + q * 4 + l] =
                fp16_clamped(model.phasor_to_core[q][l], -8.0, 8.0, 0.0);
        }
    }
    a0.linear_readout_lite = ro_lite;

    // Tier 1: Full (prefix 16/4 + delta 16..32 FIR, b_gamma[0..8], c[4..8]).
    let full_src = model.tier_overrides[1].as_deref();
    let mut ro_full = LinearReadoutFullWire::new_zeroed();
    write_w_ro_fp16(
        full_src.map_or(&model.w_ro, |p| &p.w_ro),
        &mut ro_full.w_ro_and_mods_fp16[..288],
        24,
    );
    write_scalar_ro_gains_fp16(
        full_src.map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain),
        full_src.map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain),
        full_src.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain),
        full_src.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain),
        &mut ro_full.w_ro_and_mods_fp16[288..296],
    );
    if emit_prefix_overrides {
        let fir_l = full_src.map_or(&model.fir_l, |p| &p.fir_l);
        let fir_r = full_src.map_or(&model.fir_r, |p| &p.fir_r);
        let mut pfx = PrefixOverrideFullEncodeWire::new_zeroed();
        for i in 0..16 {
            pfx.fir_l[i] = F32::new(sanitize_clamp_f32(fir_l[i], -16.0, 16.0, 0.0));
            pfx.fir_r[i] = F32::new(sanitize_clamp_f32(fir_r[i], -16.0, 16.0, 0.0));
            ro_full.w_ro_and_mods_fp16[296 + i] = fp16_clamped(fir_l[16 + i], -16.0, 16.0, 0.0);
            ro_full.w_ro_and_mods_fp16[312 + i] = fp16_clamped(fir_r[16 + i], -16.0, 16.0, 0.0);
        }
        for m in 0..4 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(full_src, model, m);
            pfx.c_re_l[m] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
            pfx.c_im_l[m] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
            pfx.c_re_r[m] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
            pfx.c_im_r[m] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
        }
        for m in 0..8 {
            let (bg, _, _, _, _) = pack_kautz_b_and_c(full_src, model, m);
            ro_full.w_ro_and_mods_fp16[328 + m] = fp16_clamped(bg, 0.0, 8.0, 0.0);
        }
        for idx in 0..4 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(full_src, model, 4 + idx);
            let off = 336 + idx * 4;
            ro_full.w_ro_and_mods_fp16[off] = fp16_clamped(cl, -8.0, 8.0, 0.0);
            ro_full.w_ro_and_mods_fp16[off + 1] = fp16_clamped(cil, -8.0, 8.0, 0.0);
            ro_full.w_ro_and_mods_fp16[off + 2] = fp16_clamped(cr, -8.0, 8.0, 0.0);
            ro_full.w_ro_and_mods_fp16[off + 3] = fp16_clamped(cir, -8.0, 8.0, 0.0);
        }
        let _ = pfx.write_to_prefix(&mut ro_full.prefix_override);
    }
    a1.linear_readout_full = ro_full;

    // Tier 2: Pro (prefix 32/8 + delta 32..48 FIR, b_gamma[0..12], c[8..12]).
    let pro_src = model.tier_overrides[2].as_deref();
    let mut ro_pro = LinearReadoutProWire::new_zeroed();
    write_w_ro_fp16(
        pro_src.map_or(&model.w_ro, |p| &p.w_ro),
        &mut ro_pro.w_ro_and_mods_fp16[..384],
        32,
    );
    write_scalar_ro_gains_fp16(
        pro_src.map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain),
        pro_src.map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain),
        pro_src.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain),
        pro_src.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain),
        &mut ro_pro.w_ro_and_mods_fp16[384..392],
    );
    if emit_prefix_overrides {
        let fir_l = pro_src.map_or(&model.fir_l, |p| &p.fir_l);
        let fir_r = pro_src.map_or(&model.fir_r, |p| &p.fir_r);
        let mut pfx = PrefixOverrideProEncodeWire::new_zeroed();
        for i in 0..32 {
            pfx.fir_l[i] = F32::new(sanitize_clamp_f32(fir_l[i], -16.0, 16.0, 0.0));
            pfx.fir_r[i] = F32::new(sanitize_clamp_f32(fir_r[i], -16.0, 16.0, 0.0));
        }
        for i in 0..16 {
            ro_pro.w_ro_and_mods_fp16[392 + i] = fp16_clamped(fir_l[32 + i], -16.0, 16.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[408 + i] = fp16_clamped(fir_r[32 + i], -16.0, 16.0, 0.0);
        }
        for m in 0..8 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(pro_src, model, m);
            pfx.c_re_l[m] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
            pfx.c_im_l[m] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
            pfx.c_re_r[m] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
            pfx.c_im_r[m] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
        }
        for m in 0..12 {
            let (bg, _, _, _, _) = pack_kautz_b_and_c(pro_src, model, m);
            ro_pro.w_ro_and_mods_fp16[424 + m] = fp16_clamped(bg, 0.0, 8.0, 0.0);
        }
        for idx in 0..4 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(pro_src, model, 8 + idx);
            let off = 436 + idx * 4;
            ro_pro.w_ro_and_mods_fp16[off] = fp16_clamped(cl, -8.0, 8.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[off + 1] = fp16_clamped(cil, -8.0, 8.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[off + 2] = fp16_clamped(cr, -8.0, 8.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[off + 3] = fp16_clamped(cir, -8.0, 8.0, 0.0);
        }
        let _ = pfx.write_to_prefix(&mut ro_pro.prefix_override);
    }
    a2.linear_readout_pro = ro_pro;

    // Tier 3: Ultra (prefix 48/12 + delta 48..96 FIR, b_gamma[0..24], c[12..24]).
    let ultra_src = model.tier_overrides[3].as_deref();
    let mut ro_ultra = LinearReadoutUltraWire::new_zeroed();
    write_w_ro_fp16(
        ultra_src.map_or(&model.w_ro, |p| &p.w_ro),
        &mut ro_ultra.w_ro_and_mods_fp16[..480],
        40,
    );
    write_scalar_ro_gains_fp16(
        ultra_src.map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain),
        ultra_src.map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain),
        ultra_src.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain),
        ultra_src.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain),
        &mut ro_ultra.w_ro_and_mods_fp16[480..488],
    );
    if emit_prefix_overrides {
        let fir_l = ultra_src.map_or(&model.fir_l, |p| &p.fir_l);
        let fir_r = ultra_src.map_or(&model.fir_r, |p| &p.fir_r);
        let mut pfx = PrefixOverrideUltraEncodeWire::new_zeroed();
        for i in 0..48 {
            pfx.fir_l[i] = F32::new(sanitize_clamp_f32(fir_l[i], -16.0, 16.0, 0.0));
            pfx.fir_r[i] = F32::new(sanitize_clamp_f32(fir_r[i], -16.0, 16.0, 0.0));
            ro_ultra.w_ro_and_mods_fp16[488 + i] = fp16_clamped(fir_l[48 + i], -16.0, 16.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[536 + i] = fp16_clamped(fir_r[48 + i], -16.0, 16.0, 0.0);
        }
        for m in 0..12 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(ultra_src, model, m);
            pfx.c_re_l[m] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
            pfx.c_im_l[m] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
            pfx.c_re_r[m] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
            pfx.c_im_r[m] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
        }
        for m in 0..24 {
            let (bg, _, _, _, _) = pack_kautz_b_and_c(ultra_src, model, m);
            ro_ultra.w_ro_and_mods_fp16[584 + m] = fp16_clamped(bg, 0.0, 8.0, 0.0);
        }
        for idx in 0..12 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(ultra_src, model, 12 + idx);
            let off = 608 + idx * 4;
            ro_ultra.w_ro_and_mods_fp16[off] = fp16_clamped(cl, -8.0, 8.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[off + 1] = fp16_clamped(cil, -8.0, 8.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[off + 2] = fp16_clamped(cr, -8.0, 8.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[off + 3] = fp16_clamped(cir, -8.0, 8.0, 0.0);
        }
        let _ = pfx.write_to_prefix(&mut ro_ultra.prefix_override);
    }
    a3.linear_readout_ultra = ro_ultra;
}

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
        a0.core_quads[q] = pack_core_quad_wire(&model.core_quads[q]);
    }
    for q in 0..2 {
        a1.core_quads[q] = pack_core_quad_wire(&model.core_quads[4 + q]);
        a2.core_quads[q] = pack_core_quad_wire(&model.core_quads[6 + q]);
        a3.core_quads[q] = pack_core_quad_wire(&model.core_quads[8 + q]);
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
    enforce_arena0_stability_invariants(&mut a0, &mut ro_pole);

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
