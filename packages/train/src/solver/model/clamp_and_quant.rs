//! Stability bound clamping and wire-exact `binary16` / `binary32` quantization snapping
//! (`snap_model_fp16` & `snap_non_readout_fp16`) for STE QAT polish and `.dnam-bin` parity
//! (`ref[train.pipeline.backbone-trajectory-opt]`, `ref[container.dnam.vm-binary]`).

use core::f64::consts::PI;

use dnam_kernels::{
    MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, MAX_SVF_K_NL, MIN_TPT_G, TANK_MAX_G_LIN_F32,
    TANK_MAX_SIGMA_SAT_F32, TANK_MAX_SIGMA_SAT_FP16, f32_to_fp16, fp16_to_f32,
    pack_wf_mix_and_rect_sign_fp16, unpack_wf_mix_and_rect_sign_fp16,
};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;

use crate::control::stability::{
    MAX_RO_POLE_48K_F32, MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP,
};

use super::SonaModelF64;
use super::quads::{CoreQuadF64, NUM_FIR_TAPS_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, TierReadoutPrefixF64};

#[inline]
pub(super) fn sanitize_clamp(v: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

#[inline]
fn snap_f32(v: f64, lo: f32, hi: f32, fallback: f32) -> f64 {
    let x = if v.is_finite() {
        (v as f32).clamp(lo, hi)
    } else {
        fallback
    };
    f64::from(if x == 0.0 { 0.0 } else { x })
}

#[inline]
fn snap_fp16(v: f64, lo: f32, hi: f32, fallback: f32) -> f64 {
    let x = if v.is_finite() {
        (v as f32).clamp(lo, hi)
    } else {
        fallback
    };
    let snapped = fp16_to_f32(f32_to_fp16(if x == 0.0 { 0.0 } else { x }));
    f64::from(if snapped == 0.0 { 0.0 } else { snapped })
}

fn clamp_core_quad_f64(q: &mut CoreQuadF64, is_block0: bool) {
    let g_lo = f64::from(MIN_TPT_G);
    let r_hi = f64::from(MAX_PARALRU_R_F32);
    let duh_hi = f64::from(MAX_PARALRU_DUHEM_GAMMA_F32);
    let k_nl_hi = f64::from(MAX_SVF_K_NL);
    for l in 0..4 {
        q.svf_g[l] = sanitize_clamp(q.svf_g[l], g_lo, 64.0, 0.125);
        q.svf_k[l] = sanitize_clamp(q.svf_k[l], 0.0, 8.0, core::f64::consts::FRAC_1_SQRT_2);
        q.svf_k_nl[l] = if is_block0 {
            sanitize_clamp(q.svf_k_nl[l], 0.0, k_nl_hi, 0.0)
        } else {
            0.0
        };
        q.tone_lp[l] = sanitize_clamp(q.tone_lp[l], -16.0, 16.0, 0.0);
        q.tone_bp[l] = sanitize_clamp(q.tone_bp[l], -16.0, 16.0, 0.0);
        q.tone_hp[l] = sanitize_clamp(q.tone_hp[l], -16.0, 16.0, 0.0);
        q.tone_ap[l] = sanitize_clamp(q.tone_ap[l], -16.0, 16.0, 0.0);
        q.tone_dir[l] = sanitize_clamp(q.tone_dir[l], -16.0, 16.0, 0.0);
        q.lru_alpha_rel[l] = sanitize_clamp(q.lru_alpha_rel[l], 0.0, r_hi, 0.995);
        q.lru_alpha_atk[l] = sanitize_clamp(q.lru_alpha_atk[l], 0.0, 1.0, 0.25);
        q.lru_kappa[l] = sanitize_clamp(q.lru_kappa[l], 0.0, 1.0, 0.0);
        q.lru_duhem_gamma[l] = sanitize_clamp(q.lru_duhem_gamma[l], 0.0, duh_hi, 0.0);
        q.lru_signed_mix[l] = sanitize_clamp(q.lru_signed_mix[l], 0.0, 1.0, 0.0);
        q.lru_stereo_link[l] = sanitize_clamp(q.lru_stereo_link[l], 0.0, 1.0, 0.0);
        q.rect_sign[l] = if q.rect_sign[l].is_finite() && q.rect_sign[l] < 0.0 {
            -1.0
        } else {
            1.0
        };
        q.sag_depth[l] = sanitize_clamp(q.sag_depth[l], -8.0, 8.0, 0.0);
        q.adaa_drive[l] = sanitize_clamp(q.adaa_drive[l], -32.0, 32.0, 0.0);
        q.lru_bias_drive[l] = sanitize_clamp(q.lru_bias_drive[l], -16.0, 16.0, 0.0);
        q.wavefolder_mix[l] = sanitize_clamp(q.wavefolder_mix[l], 0.0, 1.0, 0.0);
        q.glu_mix[l] = sanitize_clamp(q.glu_mix[l], -4.0, 4.0, 0.0);
        q.g0[l] = sanitize_clamp(q.g0[l], 0.0, f64::from(dnam_kernels::MAX_G_PRE_GAIN), 1.0);
        q.g_inf[l] = sanitize_clamp(
            q.g_inf[l],
            0.0,
            f64::from(dnam_kernels::MAX_G_PRE_GAIN),
            1.0,
        );
        q.d_pre[l] = sanitize_clamp(q.d_pre[l], 0.0, 1.0e12, 0.0);
        q.se_w_sq[l] = sanitize_clamp(q.se_w_sq[l], 0.0, 1.0e6, 0.0);
        q.se_d_bus[l] = sanitize_clamp(q.se_d_bus[l], 0.0, 1.0e12, 0.0);
        q.lin_mix[l] = sanitize_clamp(q.lin_mix[l], 0.0, 1.0, 0.0);
        q.out_gain[l] = sanitize_clamp(
            q.out_gain[l],
            -f64::from(dnam_kernels::MAX_OUT_GAIN),
            f64::from(dnam_kernels::MAX_OUT_GAIN),
            1.0,
        );
        q.in_mid_weight[l] = sanitize_clamp(q.in_mid_weight[l], -16.0, 16.0, 0.0);
        q.in_side_weight[l] = sanitize_clamp(q.in_side_weight[l], -16.0, 16.0, 0.0);
    }
}

fn clamp_prefix_f64(pfx: &mut TierReadoutPrefixF64) {
    for c in 0..2 {
        pfx.pre_conv_u_gain[c] = sanitize_clamp(pfx.pre_conv_u_gain[c], -64.0, 64.0, 0.0);
        pfx.pre_conv_hp_gain[c] = sanitize_clamp(pfx.pre_conv_hp_gain[c], -64.0, 64.0, 0.0);
        pfx.direct_s2_u_gain[c] = sanitize_clamp(pfx.direct_s2_u_gain[c], -64.0, 64.0, 0.0);
        pfx.direct_s2_hp_gain[c] = sanitize_clamp(pfx.direct_s2_hp_gain[c], -64.0, 64.0, 0.0);
    }
    for row in &mut pfx.w_ro {
        for cell in row {
            *cell = sanitize_clamp(*cell, -64.0, 64.0, 0.0);
        }
    }
    for tap in &mut pfx.fir_l {
        *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
    }
    for tap in &mut pfx.fir_r {
        *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
    }
    for k in 0..NUM_KAUTZ_PAIRS_ULTRA {
        pfx.kautz_b_gamma[k] = sanitize_clamp(pfx.kautz_b_gamma[k], 0.0, 8.0, 0.0);
        pfx.kautz_c_re_l[k] = sanitize_clamp(pfx.kautz_c_re_l[k], -8.0, 8.0, 0.0);
        pfx.kautz_c_im_l[k] = sanitize_clamp(pfx.kautz_c_im_l[k], -8.0, 8.0, 0.0);
        pfx.kautz_c_re_r[k] = sanitize_clamp(pfx.kautz_c_re_r[k], -8.0, 8.0, 0.0);
        pfx.kautz_c_im_r[k] = sanitize_clamp(pfx.kautz_c_im_r[k], -8.0, 8.0, 0.0);
    }
}

#[inline]
fn fold_kautz_r_low_with_dr(r_low: f64, dr: f64) -> f64 {
    let dr_pos = if dr.is_finite() && dr > 0.0 { dr } else { 0.0 };
    if r_low.is_finite() {
        r_low + 0.5 * dr_pos
    } else {
        0.0
    }
}

/// Enforces all canonical Table-4 bounds across `SonaModelF64` (including Block 0 `svf_k_nl` and
/// zeroed `svf_k_nl` on Blocks 1..3, plus `Op-T` `scat_psi`).
pub(super) fn clamp_model_stability_bounds(model: &mut SonaModelF64, crossbar_coupled: bool) {
    model.ro_pole_48k = sanitize_clamp(model.ro_pole_48k, 0.0, MAX_RO_POLE_48K_F64, 0.0);
    model.op_n_pre_gamma_bc = sanitize_clamp(model.op_n_pre_gamma_bc, 0.0, 1.0, 0.0);
    model.op_n_pre_zoh_hold_depth = sanitize_clamp(model.op_n_pre_zoh_hold_depth, 0.0, 0.9999, 0.0);
    model.op_n_pre_beta_q = sanitize_clamp(model.op_n_pre_beta_q, 0.0, 16.0, 0.0);
    model.tank_dry_mix = sanitize_clamp(model.tank_dry_mix, -2.0, 2.0, 1.0);
    model.os_up_quad_coeff = sanitize_clamp(model.os_up_quad_coeff, -0.20, 0.05, -0.125);
    model.os_dec_c_prev1 = sanitize_clamp(model.os_dec_c_prev1, 0.0, 0.28, 0.15);
    model.os_dec_c0 = sanitize_clamp(model.os_dec_c0, 0.0, 0.55, 0.50);
    for t in &mut model.tier_gain_trim {
        *t = sanitize_clamp(*t, 0.0625, 16.0, 1.0);
    }
    for tile in &mut model.tiles {
        for s in 0..4 {
            for d in 0..4 {
                tile.w_odd_4x4[s][d] = sanitize_clamp(tile.w_odd_4x4[s][d], -8.0, 8.0, 0.0);
                tile.w_even_4x4[s][d] = sanitize_clamp(tile.w_even_4x4[s][d], -8.0, 8.0, 0.0);
                tile.w_lru_4x4[s][d] = sanitize_clamp(tile.w_lru_4x4[s][d], -8.0, 8.0, 0.0);
            }
        }
    }
    for (q_idx, q) in model.core_quads.iter_mut().enumerate() {
        let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q_idx].0 == 0;
        clamp_core_quad_f64(q, is_block0);
    }
    let g_lo = f64::from(MIN_TPT_G);
    let r_hi = f64::from(MAX_PARALRU_R_F32);
    let sr = sanitize_clamp(model.sample_rate_hz, 44_100.0, 96_000.0, 48_000.0);
    let phasor_lim = 0.49 * sr;
    for i in 0..4 {
        model.sense_svf_g[i] = sanitize_clamp(model.sense_svf_g[i], g_lo, 64.0, 0.25);
        model.sense_lru_rel[i] = sanitize_clamp(model.sense_lru_rel[i], 0.0, r_hi, 0.995);
        model.sense_lru_atk[i] = sanitize_clamp(model.sense_lru_atk[i], 0.0, 1.0, 0.25);
        model.sense_svf_k[i] = sanitize_clamp(
            model.sense_svf_k[i],
            0.0,
            8.0,
            core::f64::consts::FRAC_1_SQRT_2,
        );
        model.sense_lru_kappa[i] = sanitize_clamp(model.sense_lru_kappa[i], 0.0, 1.0, 0.0);
        model.sense_vca_w[i] = sanitize_clamp(model.sense_vca_w[i], -512.0, 512.0, 0.0);
        model.sense_in_l[i] = sanitize_clamp(model.sense_in_l[i], -4.0, 4.0, 0.5);
        model.sense_in_r[i] = sanitize_clamp(model.sense_in_r[i], -4.0, 4.0, 0.5);
        model.sense_tone_lp[i] = sanitize_clamp(model.sense_tone_lp[i], -16.0, 16.0, 1.0);
        model.sense_tone_bp[i] = sanitize_clamp(model.sense_tone_bp[i], -16.0, 16.0, 0.0);
        model.sense_tone_hp[i] = sanitize_clamp(model.sense_tone_hp[i], -16.0, 16.0, 0.0);
        model.phasor_freq_hz[i] =
            sanitize_clamp(model.phasor_freq_hz[i], -phasor_lim, phasor_lim, 0.0);
        model.phasor_stereo_phase[i] = sanitize_clamp(model.phasor_stereo_phase[i], -PI, PI, 0.0);
    }
    for g in &mut model.crossbar_op_gains {
        *g = sanitize_clamp(*g, -4.0, 4.0, 0.0);
    }
    for row in &mut model.carrier_basis_w {
        for cell in row {
            *cell = sanitize_clamp(*cell, -8.0, 8.0, 0.0);
        }
    }
    for q in 0..10 {
        for l in 0..4 {
            model.op_p_to_core[q][l] = sanitize_clamp(model.op_p_to_core[q][l], -8.0, 8.0, 0.0);
            model.phasor_to_core[q][l] = sanitize_clamp(model.phasor_to_core[q][l], -8.0, 8.0, 0.0);
        }
    }
    for row in &mut model.opt_to_lr {
        for cell in row {
            *cell = sanitize_clamp(*cell, -4.0, 4.0, 0.0);
        }
    }
    for c in 0..2 {
        model.pre_conv_u_gain[c] = sanitize_clamp(model.pre_conv_u_gain[c], -64.0, 64.0, 0.0);
        model.pre_conv_hp_gain[c] = sanitize_clamp(model.pre_conv_hp_gain[c], -64.0, 64.0, 0.0);
        model.direct_s2_u_gain[c] = sanitize_clamp(model.direct_s2_u_gain[c], -64.0, 64.0, 0.0);
        model.direct_s2_hp_gain[c] = sanitize_clamp(model.direct_s2_hp_gain[c], -64.0, 64.0, 0.0);
    }
    for row in &mut model.w_ro {
        for cell in row {
            *cell = sanitize_clamp(*cell, -64.0, 64.0, 0.0);
        }
    }
    for tap in &mut model.fir_l {
        *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
    }
    for tap in &mut model.fir_r {
        *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
    }
    let tau_floor = if crossbar_coupled {
        MIN_DELAY_CROSSBAR_SAMP
    } else {
        4.0
    };
    let g_loop_max = f64::from(TANK_MAX_G_LIN_F32);
    let r_dc_fallback = f64::from(TANK_DC_BLOCKER_48K_F32);
    for tq in &mut model.transport_quads {
        for k in 0..4 {
            tq.delay_samples[k] =
                sanitize_clamp(tq.delay_samples[k], tau_floor, 48_000.0, tau_floor);
            tq.ramp_spans_samples[k] = sanitize_clamp(tq.ramp_spans_samples[k], 0.0, 48_000.0, 0.0);
            tq.lfo_depths[k] = sanitize_clamp(tq.lfo_depths[k], 0.0, 512.0, 0.0);
            tq.input_gains[k] = sanitize_clamp(tq.input_gains[k], -2.0, 2.0, 0.0);
            tq.wet_tap_l[k] = sanitize_clamp(tq.wet_tap_l[k], -2.0, 2.0, 0.0);
            tq.wet_tap_r[k] = sanitize_clamp(tq.wet_tap_r[k], -2.0, 2.0, 0.0);
            tq.allpass_g[k] = sanitize_clamp(tq.allpass_g[k], -0.75, 0.75, 0.0);
        }
        tq.g_loop = sanitize_clamp(tq.g_loop, 0.0, g_loop_max, 0.0);
        tq.r_dc = sanitize_clamp(tq.r_dc, 0.95, f64::from(MAX_PARALRU_R_F32), r_dc_fallback);
        tq.damp_pole = sanitize_clamp(tq.damp_pole, 0.0, 0.9995, 0.0);
        tq.lfo_rate_hz = sanitize_clamp(tq.lfo_rate_hz, 0.0, 16.0, 0.0);
        tq.ramp_rate_hz = sanitize_clamp(tq.ramp_rate_hz, -16.0, 16.0, 0.0);
        tq.fb_psi = sanitize_clamp(tq.fb_psi, -PI, PI, 0.0);
        tq.scat_psi = sanitize_clamp(tq.scat_psi, -PI, PI, 0.0);
        tq.sigma_knee = sanitize_clamp(tq.sigma_knee, 1.0e-3, 0.984_375, 0.984_375);
        tq.freeze_mix = sanitize_clamp(tq.freeze_mix, 0.0, 0.999, 0.0);
        tq.window_depth = sanitize_clamp(tq.window_depth, 0.0, 1.0, 0.0);
    }
    let theta_max = f64::from(core::f32::consts::PI - 1.0e-4);
    for k in &mut model.kautz {
        let r_raw = fold_kautz_r_low_with_dr(k.r_low, k.dr);
        k.r_low = if !r_raw.is_finite() || r_raw <= 0.0 {
            0.0
        } else {
            r_raw.clamp(0.25, 0.9995)
        };
        k.dr = 0.0;
        k.theta = sanitize_clamp(k.theta, 0.0, theta_max, 0.0);
        k.b_gamma = sanitize_clamp(k.b_gamma, 0.0, 8.0, 0.0);
        k.c_re_l = sanitize_clamp(k.c_re_l, -8.0, 8.0, 0.0);
        k.c_im_l = sanitize_clamp(k.c_im_l, -8.0, 8.0, 0.0);
        k.c_re_r = sanitize_clamp(k.c_re_r, -8.0, 8.0, 0.0);
        k.c_im_r = sanitize_clamp(k.c_im_r, -8.0, 8.0, 0.0);
    }
    for ov in &mut model.tier_overrides {
        if let Some(pfx) = ov.as_deref_mut() {
            clamp_prefix_f64(pfx);
        }
    }
    model.stage_latent_mods.clamp_bounds();
    model.sense_vca_bias = sanitize_clamp(model.sense_vca_bias, -1.0, 1.0, 0.0);
    model.op_t_wet_scale = sanitize_clamp(model.op_t_wet_scale, 0.0, 2.0, 1.0);
    model.dc_blocker_pole_5hz = sanitize_clamp(
        model.dc_blocker_pole_5hz,
        0.8001,
        f64::from(MAX_PARALRU_R_F32),
        r_dc_fallback,
    );
}

/// Snaps all non-readout parameters across `CoreQuad`, `InterBlockTile`, `Op-N[Pre]`, `Op-T`,
/// `Crossbar`, and `SensoryQuad` to the exact `binary16` or `binary32` wire representation packed
/// into `.dnam-bin` (`ref[container.dnam.vm-binary]`, `ref[train.pipeline.backbone-trajectory-opt]`).
pub fn snap_non_readout_fp16(model: &mut SonaModelF64) {
    model.clamp_all_stability_bounds(true);
    let r_hi = MAX_PARALRU_R_F32;
    let duh_hi = MAX_PARALRU_DUHEM_GAMMA_F32;
    let k_nl_hi = MAX_SVF_K_NL;
    let sqrt_half = core::f32::consts::FRAC_1_SQRT_2;
    let pi_f32 = core::f32::consts::PI;

    for (q_idx, q) in model.core_quads.iter_mut().enumerate() {
        let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q_idx].0 == 0;
        for l in 0..4 {
            q.svf_g[l] = snap_f32(q.svf_g[l], MIN_TPT_G, 64.0, 0.125);
            q.lru_alpha_rel[l] = snap_f32(q.lru_alpha_rel[l], 0.0, r_hi, 0.995);
            q.lru_alpha_atk[l] = snap_f32(q.lru_alpha_atk[l], 0.0, 1.0, 0.25);
            q.svf_k[l] = snap_fp16(q.svf_k[l], 0.0, 8.0, sqrt_half);
            q.svf_k_nl[l] = if is_block0 {
                snap_fp16(q.svf_k_nl[l], 0.0, k_nl_hi, 0.0)
            } else {
                0.0
            };
            q.lru_kappa[l] = snap_fp16(q.lru_kappa[l], 0.0, 1.0, 0.0);
            q.adaa_drive[l] = snap_fp16(q.adaa_drive[l], -32.0, 32.0, 0.0);
            q.lru_bias_drive[l] = snap_fp16(q.lru_bias_drive[l], -16.0, 16.0, 0.0);
            q.tone_lp[l] = snap_fp16(q.tone_lp[l], -16.0, 16.0, 0.0);
            q.tone_bp[l] = snap_fp16(q.tone_bp[l], -16.0, 16.0, 0.0);
            q.tone_hp[l] = snap_fp16(q.tone_hp[l], -16.0, 16.0, 0.0);
            q.tone_ap[l] = snap_fp16(q.tone_ap[l], -16.0, 16.0, 0.0);
            q.tone_dir[l] = snap_fp16(q.tone_dir[l], -16.0, 16.0, 0.0);
            q.sag_depth[l] = snap_fp16(q.sag_depth[l], -8.0, 8.0, 0.0);
            q.in_mid_weight[l] = snap_fp16(q.in_mid_weight[l], -16.0, 16.0, 1.0);
            q.in_side_weight[l] = snap_fp16(q.in_side_weight[l], -16.0, 16.0, 1.0);
            q.lru_stereo_link[l] = snap_fp16(q.lru_stereo_link[l], 0.0, 1.0, 0.0);
            q.lru_duhem_gamma[l] = snap_fp16(q.lru_duhem_gamma[l], 0.0, duh_hi, 0.0);
            q.lru_signed_mix[l] = snap_fp16(q.lru_signed_mix[l], 0.0, 1.0, 0.0);
            q.glu_mix[l] = snap_fp16(q.glu_mix[l], -4.0, 4.0, 0.0);
            q.g0[l] = snap_fp16(q.g0[l], 0.0, dnam_kernels::MAX_G_PRE_GAIN, 1.0);
            q.g_inf[l] = snap_fp16(q.g_inf[l], 0.0, dnam_kernels::MAX_G_PRE_GAIN, 1.0);
            q.d_pre[l] = snap_fp16(q.d_pre[l], 0.0, 65_504.0, 0.0);
            q.se_w_sq[l] = snap_fp16(q.se_w_sq[l], 0.0, 65_504.0, 0.0);
            q.se_d_bus[l] = snap_fp16(q.se_d_bus[l], 0.0, 65_504.0, 0.0);
            q.lin_mix[l] = snap_fp16(q.lin_mix[l], 0.0, 1.0, 0.0);
            q.out_gain[l] = snap_fp16(
                q.out_gain[l],
                -dnam_kernels::MAX_OUT_GAIN,
                dnam_kernels::MAX_OUT_GAIN,
                1.0,
            );

            let wf = if q.wavefolder_mix[l].is_finite() {
                (q.wavefolder_mix[l] as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let rs = if q.rect_sign[l].is_finite() && q.rect_sign[l] < 0.0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            let (wf_q, rs_q) =
                unpack_wf_mix_and_rect_sign_fp16(pack_wf_mix_and_rect_sign_fp16(wf, rs));
            q.wavefolder_mix[l] = f64::from(wf_q);
            q.rect_sign[l] = f64::from(rs_q);
        }
    }

    for tile in &mut model.tiles {
        for s in 0..4 {
            for d in 0..4 {
                tile.w_odd_4x4[s][d] = snap_f32(tile.w_odd_4x4[s][d], -8.0, 8.0, 0.0);
                tile.w_even_4x4[s][d] = snap_fp16(tile.w_even_4x4[s][d], -8.0, 8.0, 0.0);
                tile.w_lru_4x4[s][d] = snap_fp16(tile.w_lru_4x4[s][d], -8.0, 8.0, 0.0);
            }
        }
    }

    model.op_n_pre_gamma_bc = snap_f32(model.op_n_pre_gamma_bc, 0.0, 1.0, 0.0);
    model.op_n_pre_zoh_hold_depth = snap_f32(model.op_n_pre_zoh_hold_depth, 0.0, 0.9999, 0.0);
    model.op_n_pre_beta_q = snap_f32(model.op_n_pre_beta_q, 0.0, 16.0, 0.0);

    for tq in &mut model.transport_quads {
        for k in 0..4 {
            tq.delay_samples[k] = snap_f32(tq.delay_samples[k], 5.0, 48_000.0, 16.0);
            tq.ramp_spans_samples[k] = snap_f32(tq.ramp_spans_samples[k], 0.0, 48_000.0, 0.0);
            tq.lfo_depths[k] = snap_fp16(tq.lfo_depths[k], 0.0, 512.0, 0.0);
            tq.input_gains[k] = snap_fp16(tq.input_gains[k], -2.0, 2.0, 0.0);
            tq.wet_tap_l[k] = snap_fp16(tq.wet_tap_l[k], -2.0, 2.0, 0.0);
            tq.wet_tap_r[k] = snap_fp16(tq.wet_tap_r[k], -2.0, 2.0, 0.0);
            tq.allpass_g[k] = snap_fp16(tq.allpass_g[k], -0.75, 0.75, 0.0);
        }
        tq.g_loop = snap_f32(tq.g_loop, 0.0, TANK_MAX_G_LIN_F32, 0.0);
        tq.r_dc = snap_f32(tq.r_dc, 0.95, MAX_PARALRU_R_F32, TANK_DC_BLOCKER_48K_F32);
        tq.damp_pole = snap_f32(tq.damp_pole, 0.0, 0.9995, 0.0);
        tq.lfo_rate_hz = snap_f32(tq.lfo_rate_hz, 0.0, 16.0, 0.0);
        tq.ramp_rate_hz = snap_f32(tq.ramp_rate_hz, -16.0, 16.0, 0.0);
        tq.fb_psi = snap_fp16(tq.fb_psi, -pi_f32, pi_f32, 0.0);
        tq.scat_psi = snap_fp16(tq.scat_psi, -pi_f32, pi_f32, 0.0);
        let sk_clamped = if tq.sigma_knee.is_finite() {
            (tq.sigma_knee as f32).clamp(1.0e-3, TANK_MAX_SIGMA_SAT_F32)
        } else {
            TANK_MAX_SIGMA_SAT_F32
        };
        let sk_bits = f32_to_fp16(sk_clamped).min(TANK_MAX_SIGMA_SAT_FP16);
        tq.sigma_knee = f64::from(fp16_to_f32(sk_bits));
        let fm_clamped = sanitize_clamp(tq.freeze_mix, 0.0, 0.999, 0.0) as f32;
        let mut fm_bits = f32_to_fp16(if fm_clamped == 0.0 { 0.0 } else { fm_clamped });
        if fp16_to_f32(fm_bits) > 0.999 {
            fm_bits = fm_bits.saturating_sub(1);
        }
        let fm_snapped = fp16_to_f32(fm_bits);
        tq.freeze_mix = f64::from(if fm_snapped == 0.0 { 0.0 } else { fm_snapped });
        tq.window_depth = snap_fp16(tq.window_depth, 0.0, 1.0, 0.0);
    }

    let phasor_lim_f32 =
        0.49 * (sanitize_clamp(model.sample_rate_hz, 44_100.0, 96_000.0, 48_000.0) as f32);
    for i in 0..4 {
        model.sense_svf_g[i] = snap_f32(model.sense_svf_g[i], MIN_TPT_G, 64.0, 0.25);
        model.sense_lru_rel[i] = snap_f32(model.sense_lru_rel[i], 0.0, r_hi, 0.995);
        model.sense_lru_atk[i] = snap_f32(model.sense_lru_atk[i], 0.0, 1.0, 0.25);
        model.sense_svf_k[i] = snap_fp16(model.sense_svf_k[i], 0.0, 8.0, sqrt_half);
        model.sense_lru_kappa[i] = snap_fp16(model.sense_lru_kappa[i], 0.0, 1.0, 0.0);
        model.sense_vca_w[i] = snap_fp16(model.sense_vca_w[i], -512.0, 512.0, 0.0);
        model.sense_in_l[i] = snap_fp16(model.sense_in_l[i], -4.0, 4.0, 0.5);
        model.sense_in_r[i] = snap_fp16(model.sense_in_r[i], -4.0, 4.0, 0.5);
        model.sense_tone_lp[i] = snap_fp16(model.sense_tone_lp[i], -16.0, 16.0, 1.0);
        model.sense_tone_bp[i] = snap_fp16(model.sense_tone_bp[i], -16.0, 16.0, 0.0);
        model.sense_tone_hp[i] = snap_fp16(model.sense_tone_hp[i], -16.0, 16.0, 0.0);
        model.tier_gain_trim[i] = snap_f32(model.tier_gain_trim[i], 0.0625, 16.0, 1.0);
        model.phasor_freq_hz[i] = snap_f32(
            model.phasor_freq_hz[i],
            -phasor_lim_f32,
            phasor_lim_f32,
            0.0,
        );
        model.phasor_stereo_phase[i] =
            snap_fp16(model.phasor_stereo_phase[i], -pi_f32, pi_f32, 0.0);
    }
    for i in 0..2 {
        model.crossbar_op_gains[i] = snap_fp16(model.crossbar_op_gains[i], -4.0, 4.0, 0.0);
    }
    for row in &mut model.carrier_basis_w {
        for cell in row {
            *cell = snap_fp16(*cell, -8.0, 8.0, 0.0);
        }
    }
    for q in 0..10 {
        for l in 0..4 {
            model.op_p_to_core[q][l] = snap_fp16(model.op_p_to_core[q][l], -8.0, 8.0, 0.0);
            model.phasor_to_core[q][l] = snap_fp16(model.phasor_to_core[q][l], -8.0, 8.0, 0.0);
        }
    }
    for row in &mut model.opt_to_lr {
        for cell in row {
            *cell = snap_fp16(*cell, -4.0, 4.0, 0.0);
        }
    }
    model.ro_pole_48k = snap_f32(model.ro_pole_48k, 0.0, MAX_RO_POLE_48K_F32, 0.0);
    model.dc_blocker_pole_5hz = snap_f32(
        model.dc_blocker_pole_5hz,
        0.8001,
        MAX_PARALRU_R_F32,
        TANK_DC_BLOCKER_48K_F32,
    );
    model.tank_dry_mix = snap_fp16(model.tank_dry_mix, -2.0, 2.0, 1.0);
    model.os_up_quad_coeff = snap_f32(model.os_up_quad_coeff, -0.20, 0.05, -0.125);
    model.os_dec_c_prev1 = snap_f32(model.os_dec_c_prev1, 0.0, 0.28, 0.15);
    model.os_dec_c0 = snap_f32(model.os_dec_c0, 0.0, 0.55, 0.50);
}

/// Snaps every non-readout and Pass-2 readout/convolver parameter in `SonaModelF64` to its exact
/// `.dnam-bin` `binary16` or `binary32` wire representation for STE QAT polish
/// (`ref[train.pipeline.backbone-trajectory-opt]`).
pub fn snap_model_fp16(model: &mut SonaModelF64) {
    snap_non_readout_fp16(model);
    for c in 0..2 {
        model.pre_conv_u_gain[c] = snap_fp16(model.pre_conv_u_gain[c], -64.0, 64.0, 0.0);
        model.pre_conv_hp_gain[c] = snap_fp16(model.pre_conv_hp_gain[c], -64.0, 64.0, 0.0);
        model.direct_s2_u_gain[c] = snap_fp16(model.direct_s2_u_gain[c], -64.0, 64.0, 0.0);
        model.direct_s2_hp_gain[c] = snap_fp16(model.direct_s2_hp_gain[c], -64.0, 64.0, 0.0);
    }
    for row in &mut model.w_ro {
        for cell in row {
            *cell = snap_fp16(*cell, -64.0, 64.0, 0.0);
        }
    }
    for i in 0..NUM_FIR_TAPS_ULTRA {
        model.fir_l[i] = snap_f32(model.fir_l[i], -16.0, 16.0, 0.0);
        model.fir_r[i] = snap_f32(model.fir_r[i], -16.0, 16.0, 0.0);
    }
    let theta_hi = core::f32::consts::PI - 1.0e-4;
    for kp in &mut model.kautz {
        let r_raw = fold_kautz_r_low_with_dr(kp.r_low, kp.dr);
        let r = if !r_raw.is_finite() || r_raw <= 0.0 {
            0.0
        } else {
            (r_raw as f32).clamp(0.25, 0.9995)
        };
        kp.r_low = f64::from(r);
        kp.dr = 0.0;
        kp.theta = snap_f32(kp.theta, 0.0, theta_hi, 0.0);
        kp.b_gamma = snap_f32(kp.b_gamma, 0.0, 8.0, 0.0);
        kp.c_re_l = snap_f32(kp.c_re_l, -8.0, 8.0, 0.0);
        kp.c_im_l = snap_f32(kp.c_im_l, -8.0, 8.0, 0.0);
        kp.c_re_r = snap_f32(kp.c_re_r, -8.0, 8.0, 0.0);
        kp.c_im_r = snap_f32(kp.c_im_r, -8.0, 8.0, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_model_fp16_preserves_rect_sign_masks_blocks13_and_is_idempotent() {
        let mut model = SonaModelF64::new(48_000.0);
        for q in &mut model.core_quads {
            q.svf_k_nl = [1.234_567, 2.5, 0.125, 19.0];
        }
        model.core_quads[0].wavefolder_mix = [0.0, 0.33333, 0.77777, 1.5];
        model.core_quads[0].rect_sign = [-1.0, -1.0, 1.0, -1.0];
        model.core_quads[0].adaa_drive = [1.234_567, -2.345_678, 0.0, 31.9];
        model.transport_quads[0].scat_psi = 0.654_321;
        model.transport_quads[0].sigma_knee = 1.5;
        model.w_ro[0][6] = 0.314_159_26;
        model.direct_s2_u_gain = [0.876_543, -0.123_456];

        snap_model_fp16(&mut model);
        assert_eq!(model.core_quads[0].rect_sign, [-1.0, -1.0, 1.0, -1.0]);
        assert!(model.core_quads[0].wavefolder_mix[0] == 0.0);
        assert!(model.core_quads[0].wavefolder_mix[3] == 1.0);
        assert!(model.core_quads[0].svf_k_nl[0] > 0.0);
        assert_eq!(model.core_quads[0].svf_k_nl[3], f64::from(MAX_SVF_K_NL));
        for q_idx in [1_usize, 2, 3, 5, 7, 9] {
            assert_eq!(model.core_quads[q_idx].svf_k_nl, [0.0; 4]);
        }
        // Blocks 0 quads 4, 6, 8 preserve clamped svf_k_nl as Block 0 members.
        assert!(model.core_quads[4].svf_k_nl[0] > 0.0);
        assert!(model.core_quads[6].svf_k_nl[0] > 0.0);
        assert!(model.core_quads[8].svf_k_nl[0] > 0.0);
        assert!(model.transport_quads[0].sigma_knee <= 0.984_375);

        let after_first = model.clone();
        snap_model_fp16(&mut model);
        assert_eq!(model.core_quads, after_first.core_quads);
        assert_eq!(model.tiles, after_first.tiles);
        assert_eq!(model.transport_quads, after_first.transport_quads);
        assert_eq!(model.w_ro, after_first.w_ro);
        assert_eq!(model.direct_s2_u_gain, after_first.direct_s2_u_gain);

        let mut mod_m = SonaModelF64::new(48_000.0);
        mod_m.phasor_freq_hz = [-1_250.0, -30_000.0, 420.0, 30_000.0];
        mod_m.transport_quads[0].r_dc = 0.985;
        mod_m.transport_quads[0].freeze_mix = 1.0;
        mod_m.kautz[0].r_low = 0.10;
        mod_m.kautz[0].dr = 0.12;
        mod_m.kautz[1].r_low = 0.0;
        mod_m.kautz[2].r_low = 0.86;
        mod_m.kautz[2].dr = 0.002;
        mod_m.sense_vca_w = [999.0, -999.0, f64::NAN, 12.0];
        mod_m.tiles[0].w_odd_4x4[0][0] = 16.0;
        snap_model_fp16(&mut mod_m);
        assert_eq!(mod_m.phasor_freq_hz[0], -1_250.0);
        assert!((mod_m.phasor_freq_hz[1] - (-0.49 * 48_000.0)).abs() < 1.0e-2);
        assert!((mod_m.transport_quads[0].r_dc - 0.985).abs() < 1.0e-6);
        assert!(mod_m.transport_quads[0].freeze_mix <= 0.999);
        assert_eq!(mod_m.kautz[0].r_low, 0.25);
        assert_eq!(mod_m.kautz[0].dr, 0.0);
        assert_eq!(mod_m.kautz[1].r_low, 0.0);
        assert_eq!(mod_m.kautz[2].r_low, f64::from(0.861_f32));
        assert_eq!(mod_m.kautz[2].dr, 0.0);
        assert_eq!(mod_m.sense_vca_w[0], 512.0);
        assert_eq!(mod_m.sense_vca_w[1], -512.0);
        assert_eq!(mod_m.tiles[0].w_odd_4x4[0][0], 8.0);
    }
}
