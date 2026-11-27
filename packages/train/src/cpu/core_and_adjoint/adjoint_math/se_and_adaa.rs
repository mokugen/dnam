//! `SONA v4.0` Stage-3 (`g_pre -> AlgADAA1(4096) + odd WF -> SE-Bus -> lin_mix -> GLU -> out_gain`)
//! forward terms, `SE-Bus` `b_all` staging fallback (`lin_mix != 0 || |out_gain| < 1e-5`), and
//! straight-through `[-4096.0, 4096.0]` endpoint clamp mask matching `tools/sona-shaders`'s
//! `adjoint_core/se_and_adaa.rs` (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.rail-crossing]`).

use dnam_kernels::{
    CoreQuadV4ParamsStereo4, G_PRE_E_MAX, GpuVec4, KernelVec4, MAX_GLU_MIX, MAX_OUT_GAIN,
    SE_BUS_ABS_MAX, canonicalize_zero_f32_bits, eval_alg_adaa1_backward_f32, eval_alg_adaa1_vec4,
};

use super::adaa_derivs::eval_c2_wf_odd_with_derivs;
use super::{
    LaneForwardFrame, QuadLaneParams, ReverseStepContext, clamp_nonneg_safe, clamp_range_safe,
    flush_zero, is_finite_f32,
};
use crate::gpu::abi::{CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE};

const G_PRE_V_ABS_MAX: f32 = 1.0e3;
const MAX_NONNEG_MOD_SCALE: f32 = 1.0e12;
const FAST_OUT_GAIN_MIN_ABS: f32 = 1.0e-5;
const SE_RCP_FLOOR: f32 = 1.0e-30;

const _: () = assert!(G_PRE_V_ABS_MAX * G_PRE_V_ABS_MAX == G_PRE_E_MAX);

#[derive(Copy, Clone, Default)]
pub(in crate::cpu::core_and_adjoint) struct GPreTerms {
    pub g_pre: f32,
    pub e: f32,
    pub de: f32,
    pub inv_den: f32,
    pub v_abs: f32,
    pub v_clean: f32,
}

#[inline(always)]
pub(super) fn eval_g_pre_terms_f32(v_eff: f32, lp: &QuadLaneParams) -> GPreTerms {
    let v_clean = flush_zero(v_eff);
    let v_abs = v_clean.abs().min(G_PRE_V_ABS_MAX);
    let e = (v_abs * v_abs).min(G_PRE_E_MAX);
    let de = (lp.d_pre * e).min(MAX_NONNEG_MOD_SCALE);
    let inv_den = if de <= 0.0 {
        1.0
    } else {
        GpuVec4::splat(1.0 + de).rcp_ge1_nr2().to_array()[0]
    };
    let g_pre = if de <= 0.0 {
        lp.g0
    } else {
        let lo = lp.g0.min(lp.g_inf);
        let hi = lp.g0.max(lp.g_inf);
        clamp_range_safe((lp.g0 + lp.g_inf * de) * inv_den, lo, hi)
    };
    GPreTerms {
        g_pre,
        e,
        de,
        inv_den,
        v_abs,
        v_clean,
    }
}

#[inline(always)]
fn eval_alg_adaa1_scalar_pair(r_prev_safe: f32, r_curr_safe: f32) -> (f32, f32) {
    let (odd, even, _) =
        eval_alg_adaa1_vec4(GpuVec4::splat(r_prev_safe), GpuVec4::splat(r_curr_safe));
    (odd.to_array()[0], even.to_array()[0])
}

#[inline(always)]
fn rcp_pos_f32(x: f32) -> f32 {
    GpuVec4::splat(x.max(SE_RCP_FLOOR)).rcp_pos_nr2().to_array()[0]
}

#[inline(always)]
fn eval_se_bus_gains_4(
    v_eff: [f32; 4],
    se_w_sq: [f32; 4],
    se_d_bus: [f32; 4],
) -> ([f32; 4], f32, u32, u32, u32) {
    let mut w_c = [0.0_f32; 4];
    let mut d_c = [0.0_f32; 4];
    let mut w_open_mask = 0_u32;
    let mut d_open_mask = 0_u32;
    let mut p_bus = 0.0_f32;
    for j in 0..4 {
        w_c[j] = clamp_nonneg_safe(se_w_sq[j], MAX_NONNEG_MOD_SCALE);
        d_c[j] = clamp_nonneg_safe(se_d_bus[j], MAX_NONNEG_MOD_SCALE);
        if is_finite_f32(se_w_sq[j]) && se_w_sq[j] >= 0.0 && se_w_sq[j] < MAX_NONNEG_MOD_SCALE {
            w_open_mask |= 1_u32 << j;
        }
        if is_finite_f32(se_d_bus[j]) && se_d_bus[j] >= 0.0 && se_d_bus[j] < MAX_NONNEG_MOD_SCALE {
            d_open_mask |= 1_u32 << j;
        }
        let va = clamp_nonneg_safe(flush_zero(v_eff[j]).abs(), SE_BUS_ABS_MAX);
        p_bus += w_c[j] * va;
    }

    let mut bt = [0.0_f32; 4];
    let mut bt_open_mask = 0_u32;
    for j in 0..4 {
        let raw = d_c[j] * p_bus;
        if raw < MAX_NONNEG_MOD_SCALE {
            bt_open_mask |= 1_u32 << j;
        }
        bt[j] = if raw <= 0.0 {
            0.0
        } else if raw < MAX_NONNEG_MOD_SCALE {
            raw
        } else {
            MAX_NONNEG_MOD_SCALE
        };
    }

    let inv = GpuVec4::splat(1.0)
        .add(GpuVec4::from_array(bt))
        .rcp_ge1_nr2()
        .to_array();
    let mut g_se = [1.0_f32; 4];
    for j in 0..4 {
        g_se[j] = if bt[j] <= 0.0 {
            1.0
        } else {
            clamp_range_safe(inv[j], 0.0, 1.0)
        };
    }
    (g_se, p_bus, bt_open_mask, w_open_mask, d_open_mask)
}

#[inline]
pub(in crate::cpu::core_and_adjoint) fn needs_b_staging(
    params: &CoreQuadV4ParamsStereo4<GpuVec4>,
) -> bool {
    let lm = params.lin_mix.to_array();
    let og = params.out_gain.to_array();
    for j in 0..4 {
        if clamp_range_safe(lm[j], 0.0, 1.0) != 0.0 {
            return true;
        }
        if clamp_range_safe(og[j], -MAX_OUT_GAIN, MAX_OUT_GAIN).abs() < FAST_OUT_GAIN_MIN_ABS {
            return true;
        }
    }
    false
}

#[inline(always)]
fn is_glu_zero(params: &CoreQuadV4ParamsStereo4<GpuVec4>) -> bool {
    let gm = params.glu_mix.to_array();
    gm.iter()
        .all(|&v| clamp_range_safe(v, -MAX_GLU_MIX, MAX_GLU_MIX) == 0.0)
}

#[derive(Copy, Clone, Default)]
struct ChannelStage3Out {
    adj_odd_wf: f32,
    adj_even: f32,
    adj_g_pre_tone_lin: f32,
    adj_veff_se: f32,
    grad_se_w_sq: f32,
    grad_se_d_bus: f32,
    grad_lin_mix: f32,
    grad_glu_mix: f32,
    grad_out_gain: f32,
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_channel_se_glu_out_pullback(
    ve_4: [f32; 4],
    o_arr: [f32; 4],
    e_arr: [f32; 4],
    dlo_arr: [f32; 4],
    dle_arr: [f32; 4],
    b_staged_arr: [f32; 4],
    params: &CoreQuadV4ParamsStereo4<GpuVec4>,
    lp: &QuadLaneParams,
    c: usize,
    a_odd_wf_c: f32,
    a_even_c: f32,
    g_pre_c: f32,
    tone_c: f32,
    glu_zero: bool,
    stage_b: bool,
) -> ChannelStage3Out {
    let w_arr = params.se_w_sq.to_array();
    let d_arr = params.se_d_bus.to_array();
    let (g_se, p_bus, bt_open_mask, w_open_mask, d_open_mask) =
        eval_se_bus_gains_4(ve_4, w_arr, d_arr);

    let lin_arr = params.lin_mix.to_array();
    let glu_arr = params.glu_mix.to_array();
    let og_arr = params.out_gain.to_array();

    let mut b = [0.0_f32; 4];
    if stage_b {
        b = b_staged_arr;
    } else {
        for j in 0..4 {
            let og = clamp_range_safe(og_arr[j], -MAX_OUT_GAIN, MAX_OUT_GAIN);
            let inv_og = rcp_pos_f32(og.abs());
            let signed_inv = if og < 0.0 { -inv_og } else { inv_og };
            b[j] = o_arr[j] * signed_inv;
        }
    }

    let mut adj_inner = [0.0_f32; 4];
    let mut adj_glu_term = [0.0_f32; 4];
    for j in 0..4 {
        let og_j = clamp_range_safe(og_arr[j], -MAX_OUT_GAIN, MAX_OUT_GAIN);
        let one_m_lin_j = 1.0 - clamp_range_safe(lin_arr[j], 0.0, 1.0);
        let gm_j = clamp_range_safe(glu_arr[j], -MAX_GLU_MIX, MAX_GLU_MIX);
        adj_inner[j] = dle_arr[j] * og_j * one_m_lin_j;
        adj_glu_term[j] = adj_inner[j] * gm_j;
    }

    let mut dl_gse_c = 0.0_f32;
    let mut sum_dg_dpbus = 0.0_f32;
    for j in 0..4 {
        let j_prev = (j + 3) & 3;
        let glu_cross = if glu_zero {
            0.0
        } else {
            adj_glu_term[j_prev] * b[j_prev] * b[j]
        };
        let dl_gse_j =
            (dlo_arr[j] * o_arr[j] + dle_arr[j] * e_arr[j] + glu_cross) * rcp_pos_f32(g_se[j]);
        if j == c {
            dl_gse_c = dl_gse_j;
        }
        if (bt_open_mask & (1_u32 << j)) != 0 {
            let d_j = clamp_nonneg_safe(d_arr[j], MAX_NONNEG_MOD_SCALE);
            sum_dg_dpbus += dl_gse_j * (-g_se[j] * g_se[j] * d_j);
        }
    }

    let ve_clean = flush_zero(ve_4[c]);
    let va_c = clamp_nonneg_safe(ve_clean.abs(), SE_BUS_ABS_MAX);
    let grad_se_w_sq = if (w_open_mask & (1_u32 << c)) != 0 {
        sum_dg_dpbus * va_c
    } else {
        0.0
    };
    let grad_se_d_bus = if (d_open_mask & (1_u32 << c)) != 0 && (bt_open_mask & (1_u32 << c)) != 0 {
        dl_gse_c * (-g_se[c] * g_se[c] * p_bus)
    } else {
        0.0
    };
    let v_abs_open = is_finite_f32(ve_clean) && ve_clean.abs() < SE_BUS_ABS_MAX;
    let sign_v = if ve_clean > 0.0 {
        1.0
    } else if ve_clean < 0.0 {
        -1.0
    } else {
        0.0
    };
    let w_sq_c = clamp_nonneg_safe(w_arr[c], MAX_NONNEG_MOD_SCALE);
    let adj_veff_se = if v_abs_open {
        sum_dg_dpbus * w_sq_c * sign_v
    } else {
        0.0
    };

    let c_next = (c + 1) & 3;
    let c_prev = (c + 3) & 3;
    let b_c = g_se[c] * a_odd_wf_c;
    let adj_b_c = if glu_zero {
        0.0
    } else {
        adj_glu_term[c] * b[c_next] + adj_glu_term[c_prev] * b[c_prev]
    };
    let adj_o_pre_c = dlo_arr[c] * lp.out_gain * g_se[c];
    let adj_odd_wf = adj_o_pre_c * (1.0 - lp.lin_mix) + adj_b_c * g_se[c];
    let adj_even = adj_inner[c] * g_se[c];
    let adj_g_pre_tone_lin = adj_o_pre_c * lp.lin_mix;

    let gp_tone = g_pre_c * tone_c;
    let o_pre_c = (1.0 - lp.lin_mix) * a_odd_wf_c + lp.lin_mix * gp_tone;
    let inner_c = g_se[c] * a_even_c
        + if glu_zero {
            0.0
        } else {
            lp.glu_mix * (b_c * b[c_next])
        };

    let grad_out_gain = if lp.out_open {
        dlo_arr[c] * g_se[c] * o_pre_c + dle_arr[c] * (1.0 - lp.lin_mix) * inner_c
    } else {
        0.0
    };
    let grad_lin_mix = if lp.lin_open {
        adj_o_pre_c * (gp_tone - a_odd_wf_c) - dle_arr[c] * lp.out_gain * inner_c
    } else {
        0.0
    };
    let grad_glu_mix = if lp.glu_open {
        adj_inner[c] * (b_c * b[c_next])
    } else {
        0.0
    };

    ChannelStage3Out {
        adj_odd_wf,
        adj_even,
        adj_g_pre_tone_lin,
        adj_veff_se,
        grad_se_w_sq,
        grad_se_d_bus,
        grad_lin_mix,
        grad_glu_mix,
        grad_out_gain,
    }
}

#[inline(always)]
fn eval_g_pre_grads_and_dv(
    gp: &GPreTerms,
    lp: &QuadLaneParams,
    adj_g_pre: f32,
) -> (f32, f32, f32, f32) {
    let de_open = (lp.d_pre * gp.e) < MAX_NONNEG_MOD_SCALE;
    let dg_dg0 = if lp.g0_open {
        if gp.de <= 0.0 { 1.0 } else { gp.inv_den }
    } else {
        0.0
    };
    let dg_dg_inf = if lp.g_inf_open && gp.de > 0.0 {
        gp.de * gp.inv_den
    } else {
        0.0
    };
    let dg_dde = (lp.g_inf - lp.g0) * gp.inv_den * gp.inv_den;
    let dg_dd_pre = if lp.d_pre_open && de_open {
        dg_dde * gp.e
    } else {
        0.0
    };
    let dg_dv_eff = if gp.de > 0.0 && de_open && gp.v_abs < G_PRE_V_ABS_MAX && gp.e < G_PRE_E_MAX {
        dg_dde * lp.d_pre * (2.0 * gp.v_clean)
    } else {
        0.0
    };
    (
        adj_g_pre * dg_dg0,
        adj_g_pre * dg_dg_inf,
        adj_g_pre * dg_dd_pre,
        adj_g_pre * dg_dv_eff,
    )
}

#[derive(Copy, Clone)]
pub(super) struct SeAndNonlinPullback {
    pub adj_r_prev_l: f32,
    pub adj_r_prev_r: f32,
    pub adj_veff_total_l: f32,
    pub adj_veff_total_r: f32,
    pub adj_tone_from_stage3_l: f32,
    pub adj_tone_from_stage3_r: f32,
    pub grad_g0: f32,
    pub grad_g_inf: f32,
    pub grad_d_pre: f32,
    pub grad_se_w_sq: f32,
    pub grad_se_d_bus: f32,
    pub grad_adaa_drive: f32,
    pub grad_lru_bias_drive: f32,
    pub grad_lin_mix: f32,
    pub grad_wf_mix: f32,
    pub grad_glu_mix: f32,
    pub grad_out_gain: f32,
}

/// Evaluates Stage-3 pullback across one frame, returning `adj_r_prev` (pre-carry) and parameter
/// gradients. Caller sums `adj_r_curr + carry_adj_r_next` and applies `r_curr_open` mask!
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_se_and_adaa_frame_pullback(
    ctx: &ReverseStepContext<'_>,
    params: &CoreQuadV4ParamsStereo4<GpuVec4>,
    lp: &QuadLaneParams,
    fwd: &LaneForwardFrame,
    b_staged: Option<([f32; 4], [f32; 4])>,
    frame: usize,
    adj_r_next: (f32, f32),
) -> SeAndNonlinPullback {
    let c = ctx.c;
    let glu_zero = is_glu_zero(params);
    let stage_b = b_staged.is_some();
    let (b_staged_l, b_staged_r) = b_staged.unwrap_or(([0.0; 4], [0.0; 4]));
    let one_minus_wf = 1.0 - lp.wf_mix;

    let (odd_l, even_l) = eval_alg_adaa1_scalar_pair(fwd.r_prev_l, fwd.r_curr_l);
    let (odd_r, even_r) = eval_alg_adaa1_scalar_pair(fwd.r_prev_r, fwd.r_curr_r);
    let wf_l = eval_c2_wf_odd_with_derivs(fwd.r_prev_l, fwd.r_curr_l, ctx.tiny_eps);
    let wf_r = eval_c2_wf_odd_with_derivs(fwd.r_prev_r, fwd.r_curr_r, ctx.tiny_eps);
    let a_odd_wf_l = one_minus_wf * odd_l + lp.wf_mix * wf_l.wf_odd;
    let a_odd_wf_r = one_minus_wf * odd_r + lp.wf_mix * wf_r.wf_odd;

    let feat_base = (ctx.q_global * ctx.num_frames + frame) * FEATURE_VEC4_STRIDE;
    let dl_ve_l = ctx.d_loss_d_features[feat_base];
    let dl_ve_r = ctx.d_loss_d_features[feat_base + 1];
    let dl_o_l = ctx.d_loss_d_features[feat_base + 2];
    let dl_o_r = ctx.d_loss_d_features[feat_base + 3];
    let dl_e_l = ctx.d_loss_d_features[feat_base + 4];
    let dl_e_r = ctx.d_loss_d_features[feat_base + 5];

    let ve_l = ctx.feature_outputs[feat_base];
    let ve_r = ctx.feature_outputs[feat_base + 1];
    let o_l = ctx.feature_outputs[feat_base + 2];
    let o_r = ctx.feature_outputs[feat_base + 3];
    let e_l = ctx.feature_outputs[feat_base + 4];
    let e_r = ctx.feature_outputs[feat_base + 5];

    let st3_l = eval_channel_se_glu_out_pullback(
        ve_l,
        o_l,
        e_l,
        dl_o_l,
        dl_e_l,
        b_staged_l,
        params,
        lp,
        c,
        a_odd_wf_l,
        even_l,
        fwd.g_pre_l.g_pre,
        fwd.tone_l,
        glu_zero,
        stage_b,
    );
    let st3_r = eval_channel_se_glu_out_pullback(
        ve_r,
        o_r,
        e_r,
        dl_o_r,
        dl_e_r,
        b_staged_r,
        params,
        lp,
        c,
        a_odd_wf_r,
        even_r,
        fwd.g_pre_r.g_pre,
        fwd.tone_r,
        glu_zero,
        stage_b,
    );

    let dl = eval_alg_adaa1_backward_f32(fwd.r_prev_l, fwd.r_curr_l);
    let dr = eval_alg_adaa1_backward_f32(fwd.r_prev_r, fwd.r_curr_r);

    let adj_r_curr_l = (one_minus_wf * dl.d_odd_dr_curr + lp.wf_mix * wf_l.d_wf_db)
        * st3_l.adj_odd_wf
        + dl.d_even_dr_curr * st3_l.adj_even;
    let adj_r_prev_l = (one_minus_wf * dl.d_odd_dr_prev + lp.wf_mix * wf_l.d_wf_da)
        * st3_l.adj_odd_wf
        + dl.d_even_dr_prev * st3_l.adj_even;

    let adj_r_curr_r = (one_minus_wf * dr.d_odd_dr_curr + lp.wf_mix * wf_r.d_wf_db)
        * st3_r.adj_odd_wf
        + dr.d_even_dr_curr * st3_r.adj_even;
    let adj_r_prev_r = (one_minus_wf * dr.d_odd_dr_prev + lp.wf_mix * wf_r.d_wf_da)
        * st3_r.adj_odd_wf
        + dr.d_even_dr_prev * st3_r.adj_even;

    let grad_wf_mix = if lp.wf_open {
        st3_l.adj_odd_wf * (wf_l.wf_odd - odd_l) + st3_r.adj_odd_wf * (wf_r.wf_odd - odd_r)
    } else {
        0.0
    };

    let adj_r_l = if fwd.r_curr_open_l {
        adj_r_curr_l + adj_r_next.0
    } else {
        0.0
    };
    let adj_r_r = if fwd.r_curr_open_r {
        adj_r_curr_r + adj_r_next.1
    } else {
        0.0
    };

    let adj_g_pre_l = adj_r_l * lp.adaa_drive * fwd.tone_l + st3_l.adj_g_pre_tone_lin * fwd.tone_l;
    let adj_g_pre_r = adj_r_r * lp.adaa_drive * fwd.tone_r + st3_r.adj_g_pre_tone_lin * fwd.tone_r;
    let (gg0_l, gginf_l, gdpre_l, gdve_l) = eval_g_pre_grads_and_dv(&fwd.g_pre_l, lp, adj_g_pre_l);
    let (gg0_r, gginf_r, gdpre_r, gdve_r) = eval_g_pre_grads_and_dv(&fwd.g_pre_r, lp, adj_g_pre_r);

    let adj_tone_from_stage3_l =
        adj_r_l * lp.adaa_drive * fwd.g_pre_l.g_pre + st3_l.adj_g_pre_tone_lin * fwd.g_pre_l.g_pre;
    let adj_tone_from_stage3_r =
        adj_r_r * lp.adaa_drive * fwd.g_pre_r.g_pre + st3_r.adj_g_pre_tone_lin * fwd.g_pre_r.g_pre;

    let adj_veff_total_l = dl_ve_l[c] + st3_l.adj_veff_se + adj_r_l * lp.lru_bias_drive + gdve_l;
    let adj_veff_total_r = dl_ve_r[c] + st3_r.adj_veff_se + adj_r_r * lp.lru_bias_drive + gdve_r;

    let grad_adaa_drive = if lp.adaa_drive_open {
        adj_r_l * fwd.g_pre_l.g_pre * fwd.tone_l + adj_r_r * fwd.g_pre_r.g_pre * fwd.tone_r
    } else {
        0.0
    };
    let grad_lru_bias_drive = if lp.lru_bias_drive_open {
        adj_r_l * fwd.v_eff_l + adj_r_r * fwd.v_eff_r
    } else {
        0.0
    };

    let _ = canonicalize_zero_f32_bits;
    SeAndNonlinPullback {
        adj_r_prev_l,
        adj_r_prev_r,
        adj_veff_total_l,
        adj_veff_total_r,
        adj_tone_from_stage3_l,
        adj_tone_from_stage3_r,
        grad_g0: gg0_l + gg0_r,
        grad_g_inf: gginf_l + gginf_r,
        grad_d_pre: gdpre_l + gdpre_r,
        grad_se_w_sq: st3_l.grad_se_w_sq + st3_r.grad_se_w_sq,
        grad_se_d_bus: st3_l.grad_se_d_bus + st3_r.grad_se_d_bus,
        grad_adaa_drive,
        grad_lru_bias_drive,
        grad_lin_mix: st3_l.grad_lin_mix + st3_r.grad_lin_mix,
        grad_wf_mix,
        grad_glu_mix: st3_l.grad_glu_mix + st3_r.grad_glu_mix,
        grad_out_gain: st3_l.grad_out_gain + st3_r.grad_out_gain,
    }
}

/// Computes `b_all_{l,r}[frame] = g_se * a_odd_wf` for one tile when `needs_b_staging` is true.
pub(in crate::cpu::core_and_adjoint) fn stage_b_all_for_tile(
    params: &CoreQuadV4ParamsStereo4<GpuVec4>,
    lane_frames: &[Vec<LaneForwardFrame>; 4],
    feature_outputs: &[Vec<[f32; 4]>],
    tiny_eps: f32,
) -> Vec<([f32; 4], [f32; 4])> {
    let len = lane_frames[0].len();
    let mut out = vec![([0.0_f32; 4], [0.0_f32; 4]); len];
    let w_arr = params.se_w_sq.to_array();
    let d_arr = params.se_d_bus.to_array();
    let wf_arr = params.wavefolder_mix.to_array();
    for local_t in 0..len {
        let ve_l = feature_outputs[local_t][0];
        let ve_r = feature_outputs[local_t][1];
        let (g_se_l, _, _, _, _) = eval_se_bus_gains_4(ve_l, w_arr, d_arr);
        let (g_se_r, _, _, _, _) = eval_se_bus_gains_4(ve_r, w_arr, d_arr);
        for c in 0..4 {
            let fwd = &lane_frames[c][local_t];
            let wf_c = clamp_range_safe(wf_arr[c], 0.0, 1.0);
            let (odd_l, _) = eval_alg_adaa1_scalar_pair(fwd.r_prev_l, fwd.r_curr_l);
            let (odd_r, _) = eval_alg_adaa1_scalar_pair(fwd.r_prev_r, fwd.r_curr_r);
            let wf_l = eval_c2_wf_odd_with_derivs(fwd.r_prev_l, fwd.r_curr_l, tiny_eps).wf_odd;
            let wf_r = eval_c2_wf_odd_with_derivs(fwd.r_prev_r, fwd.r_curr_r, tiny_eps).wf_odd;
            let a_l = (1.0 - wf_c) * odd_l + wf_c * wf_l;
            let a_r = (1.0 - wf_c) * odd_r + wf_c * wf_r;
            out[local_t].0[c] = g_se_l[c] * a_l;
            out[local_t].1[c] = g_se_r[c] * a_r;
        }
    }
    let _ = CORE_INPUT_VEC4_STRIDE;
    out
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        MAX_ALG_R, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4, SonaToneFolded3, SonaToneFoldedBlock0,
        SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
    };

    use super::*;
    use crate::cpu::core_and_adjoint::{run_adjoint_scan_cpu, run_fused_core_scan_cpu};
    use crate::gpu::abi::{AdjointScanUniforms, FusedScanUniforms, quad_param_slots};
    use crate::runner::types::{AdjointScanInput, FusedCoreScanInput, pack_core_quad_params};

    #[test]
    fn test_se_bus_b_staging_tiny_out_gain_and_clamp_mask_fd() {
        let g = GpuVec4::splat(prewarp_tan_pi_f_over_fs_f32(0.10));
        let k = GpuVec4::splat(0.75);
        let params = CoreQuadV4ParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneFoldedBlock0::from_folded3(&SonaToneFolded3::from_weights(
                &SonaToneWeights4 {
                    lp: GpuVec4::splat(0.35),
                    bp: GpuVec4::splat(0.25),
                    hp: GpuVec4::splat(0.05),
                    ap: GpuVec4::splat(-0.05),
                    dir: GpuVec4::splat(0.30),
                },
                k,
            )),
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.90),
                GpuVec4::splat(0.35),
                GpuVec4::splat(0.25),
                GpuVec4::splat(0.20),
                GpuVec4::splat(0.45),
                GpuVec4::splat(0.18),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::zero(),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            g0: GpuVec4::splat(0.85),
            g_inf: GpuVec4::splat(1.30),
            d_pre: GpuVec4::splat(0.60),
            se_w_sq: GpuVec4::from_array([0.5, 0.25, 0.4, 0.3]),
            se_d_bus: GpuVec4::from_array([0.4, 0.6, 0.2, 0.5]),
            adaa_drive: GpuVec4::splat(2.5),
            lru_bias_drive: GpuVec4::splat(0.3),
            lin_mix: GpuVec4::from_array([0.2, 0.35, 0.1, 0.4]),
            wavefolder_mix: GpuVec4::splat(0.35),
            glu_mix: GpuVec4::from_array([0.6, -0.5, 0.4, -0.3]),
            out_gain: GpuVec4::from_array([0.0, 1.0e-7, 0.8, -0.9]),
        };
        let packed = pack_core_quad_params(&params);
        let n = 32_usize;
        let mut core_in = vec![[0.0_f32; 4]; n * 2];
        for t in 0..n {
            let tf = t as f32;
            core_in[t * 2] = [0.5 * (0.2 * tf).sin(), 0.4 * (0.3 * tf).cos(), 0.35, -0.25];
            core_in[t * 2 + 1] = [
                -0.4 * (0.17 * tf).cos(),
                0.45 * (0.23 * tf).sin(),
                -0.3,
                0.2,
            ];
        }
        let fwd = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: n as u32,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 1,
            },
            core_inputs: &core_in,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        let dl_feat = vec![[0.15_f32, -0.1, 0.2, -0.12]; n * FEATURE_VEC4_STRIDE];
        let adj = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: n as u32,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 1,
            },
            core_inputs: &core_in,
            quad_params: &packed,
            tile_checkpoints: &fwd.tile_checkpoints,
            d_loss_d_features: &dl_feat,
            feature_outputs: &fwd.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();

        let eps = 2.0e-3_f32;
        for &slot in &[
            quad_param_slots::SE_W_SQ,
            quad_param_slots::SE_D_BUS,
            quad_param_slots::LIN_MIX,
            quad_param_slots::GLU_MIX,
            quad_param_slots::OUT_GAIN,
            quad_param_slots::G0,
            quad_param_slots::G_INF,
            quad_param_slots::D_PRE,
        ] {
            let mut p_p = packed;
            let mut p_m = packed;
            p_p[slot][2] += eps;
            p_m[slot][2] -= eps;
            let run = |p: &[[f32; 4]]| -> f64 {
                let f = run_fused_core_scan_cpu(&FusedCoreScanInput {
                    uniforms: FusedScanUniforms {
                        num_frames: n as u32,
                        num_tiles: 1,
                        num_quads: 1,
                        num_stems: 1,
                        tile_start: 0,
                        soft_rect_eps: SOFT_RECT_EPS,
                        tiny_crossing_eps: TINY_CROSSING_EPS,
                        block_idx: 1,
                    },
                    core_inputs: &core_in,
                    quad_params: p,
                    initial_checkpoints: None,
                })
                .unwrap();
                f.feature_outputs
                    .iter()
                    .zip(dl_feat.iter())
                    .map(|(a, b)| {
                        (0..4)
                            .map(|j| f64::from(a[j]) * f64::from(b[j]))
                            .sum::<f64>()
                    })
                    .sum()
            };
            let fd = (run(&p_p) - run(&p_m)) / f64::from(2.0 * eps);
            let got = f64::from(adj.param_grads[slot][2]);
            assert!(
                (got - fd).abs() <= 5.0e-3 * fd.abs().max(1.0),
                "Stage 3 slot {slot}: got={got} fd={fd}"
            );
        }

        // Verify |r_unclamped| > MAX_ALG_R (= 4096) zeros both adj_r_curr and adj_r_prev_next!
        let mut p_sat = packed;
        p_sat[quad_param_slots::TONE_C_U] = [1.0; 4];
        p_sat[quad_param_slots::TONE_C_LP] = [0.0; 4];
        p_sat[quad_param_slots::TONE_C_BP] = [0.0; 4];
        p_sat[quad_param_slots::G0] = [2.5; 4];
        p_sat[quad_param_slots::G_INF] = [2.5; 4];
        p_sat[quad_param_slots::ADAA_DRIVE] = [32.0; 4];
        p_sat[quad_param_slots::LIN_MIX] = [0.0; 4];
        let sat_in = vec![[100.0_f32; 4]; 4];
        let fwd_sat = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: 2,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 1,
            },
            core_inputs: &sat_in,
            quad_params: &p_sat,
            initial_checkpoints: None,
        })
        .unwrap();
        let dl_sat = vec![[0.0, 0.0, 1.0, 1.0]; 2 * FEATURE_VEC4_STRIDE];
        let adj_sat = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: 2,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 1,
            },
            core_inputs: &sat_in,
            quad_params: &p_sat,
            tile_checkpoints: &fwd_sat.tile_checkpoints,
            d_loss_d_features: &dl_sat,
            feature_outputs: &fwd_sat.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();
        assert_eq!(adj_sat.param_grads[quad_param_slots::ADAA_DRIVE], [0.0; 4]);
        let _ = MAX_ALG_R;
    }
}
