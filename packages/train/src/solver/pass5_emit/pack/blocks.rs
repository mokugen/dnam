//! Wire packing helpers for `CoreQuadWire`, `SonaInterBlockTilePairWire`, `KautzSoaBlockWire`,
//! `TransportQuadWire` (`scat_psi_fp16`), and `SonaCrossbarBlockWire` (`ref[container.dnam.vm-binary]`).

use dnam_kernels::{
    MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, MAX_SVF_K_NL, MIN_TPT_G, TANK_MAX_G_LIN_F32,
    f32_to_fp16, pack_wf_mix_and_rect_sign_fp16,
};
use dnam_vm::bin::{
    CoreQuadWire, KautzSoaBlockWire, SonaCrossbarBlockWire, SonaInterBlockTilePairWire,
    TRANSPORT_FLAG_EXT_F32, TransportQuadWire,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16};

use super::super::super::model::{
    CoreQuadF64, InterBlockTileF64, KautzPairF64, SonaModelF64, TierReadoutPrefixF64,
    TransportQuadF64,
};

#[inline]
pub(crate) fn sanitize_clamp_f32(v: f64, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        (v as f32).clamp(lo, hi)
    } else {
        fallback
    }
}

#[inline]
pub(super) fn fp16_clamped(v: f64, lo: f32, hi: f32, fallback: f32) -> U16 {
    let clamped = sanitize_clamp_f32(v, lo, hi, fallback);
    U16::new(f32_to_fp16(if clamped == 0.0 { 0.0 } else { clamped }))
}

/// Packs one [`CoreQuadF64`] into wire [`CoreQuadWire`], keeping `svf_k_nl_fp16` on Block 0 quads
/// (`is_block0 == true`, canonical `+0.0`) and forcing `0` on linear Blocks `1..=3`.
pub(super) fn pack_core_quad_wire(q: &CoreQuadF64, is_block0: bool) -> CoreQuadWire {
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
        w.svf_k_nl_fp16[l] = if is_block0 {
            fp16_clamped(q.svf_k_nl[l], 0.0, MAX_SVF_K_NL, 0.0)
        } else {
            U16::new(0)
        };
    }
    w
}

/// Packs one inter-block `4×4` tile pair into wire [`SonaInterBlockTilePairWire`].
pub(super) fn pack_tile_wire(t: &InterBlockTileF64) -> SonaInterBlockTilePairWire {
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

/// Packs up to `4` consecutive Kautz conjugate pairs into one [`KautzSoaBlockWire`].
pub(super) fn pack_kautz_block(
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

/// Packs one [`TransportQuadF64`] into [`TransportQuadWire`], writing continuous `scat_psi_fp16`,
/// unconditional `allpass_delays`, and zeroed bit-0 `flags`.
pub(super) fn pack_transport_quad_wire(t: &TransportQuadF64) -> TransportQuadWire {
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
    w.allpass_delays = t.allpass_delays;
    w.g_loop = F32::new(sanitize_clamp_f32(t.g_loop, 0.0, TANK_MAX_G_LIN_F32, 0.0));
    w.r_dc = F32::new(TANK_DC_BLOCKER_48K_F32);
    w.damp_pole = F32::new(sanitize_clamp_f32(t.damp_pole, 0.0, 0.9995, 0.0));
    w.lfo_rate_hz = F32::new(sanitize_clamp_f32(t.lfo_rate_hz, 0.0, 16.0, 0.0));
    w.ramp_rate_hz = F32::new(sanitize_clamp_f32(t.ramp_rate_hz, -16.0, 16.0, 0.0));
    w.fb_psi_fp16 = fp16_clamped(t.fb_psi, -core::f32::consts::PI, core::f32::consts::PI, 0.0);
    w.scat_psi_fp16 = fp16_clamped(
        t.scat_psi,
        -core::f32::consts::PI,
        core::f32::consts::PI,
        0.0,
    );
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

/// Packs full crossbar routing matrices into [`SonaCrossbarBlockWire`].
pub(super) fn pack_crossbar_block_wire(model: &SonaModelF64) -> SonaCrossbarBlockWire {
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
