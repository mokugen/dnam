#![forbid(unsafe_code)]

//! SONA v4.0 Stage-3 pullback (`g_pre -> AlgADAA1(4096) + odd C^2 Wavefolder -> SE-Bus ->
//! lin_mix -> Bounded GLU -> out_gain`) shared across `pass_mode = 0` and `pass_mode = 2`
//! (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.core.interblock-tiles]`,
//! `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::{LaneForwardFrame, compute_lane_next_tile_first_r_and_aodd_wf};
use crate::abi::{ADJOINT_SCAN_SCRATCH_FLOATS, TILE_FRAMES};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, GPreTerms, QuadLaneParams, QuadParamsUnpacked,
    clamp_nonneg_safe, clamp_range_safe, eval_c2_wf_odd_with_derivs, flush_zero, get_lane,
    is_finite_bits, load_vec4, set_lane, store_vec4, wg_barrier,
};
use crate::reduce::rcp_pos_f32;
use dnam_kernels::{
    G_PRE_E_MAX, GpuVec4, KernelVec4, MAX_GLU_MIX, MAX_OUT_GAIN, SE_BUS_ABS_MAX,
    eval_alg_adaa1_backward_f32, eval_alg_adaa1_vec4,
};
use spirv_std::glam::Vec4;

const G_PRE_V_ABS_MAX: f32 = 1.0e3;
const MAX_NONNEG_MOD_SCALE: f32 = 1.0e12;
const FAST_OUT_GAIN_MIN_ABS: f32 = 1.0e-5;

const _: () = assert!(G_PRE_V_ABS_MAX * G_PRE_V_ABS_MAX == G_PRE_E_MAX);

/// Combined Stage-3 local + tile-seam sensitivities for one lane `c` and stereo pair `(L, R)`.
#[derive(Copy, Clone)]
pub struct SeAndNonlinPullback {
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

#[inline(always)]
fn eval_alg_adaa1_scalar_pair(r_prev_safe: f32, r_curr_safe: f32) -> (f32, f32) {
    let (odd, even, _) =
        eval_alg_adaa1_vec4(GpuVec4::splat(r_prev_safe), GpuVec4::splat(r_curr_safe));
    (odd.to_array()[0], even.to_array()[0])
}

#[inline(always)]
fn eval_se_bus_gains_4(
    v_eff: Vec4,
    se_w_sq: Vec4,
    se_d_bus: Vec4,
) -> ([f32; 4], f32, u32, u32, u32) {
    let v_arr = v_eff.to_array();
    let w_arr = se_w_sq.to_array();
    let d_arr = se_d_bus.to_array();
    let mut w_c = [0.0_f32; 4];
    let mut d_c = [0.0_f32; 4];
    let mut w_open_mask = 0_u32;
    let mut d_open_mask = 0_u32;
    let mut p_bus = 0.0_f32;
    for j in 0..4 {
        w_c[j] = clamp_nonneg_safe(w_arr[j], MAX_NONNEG_MOD_SCALE);
        d_c[j] = clamp_nonneg_safe(d_arr[j], MAX_NONNEG_MOD_SCALE);
        if is_finite_bits(w_arr[j]) && w_arr[j] >= 0.0 && w_arr[j] < MAX_NONNEG_MOD_SCALE {
            w_open_mask |= 1_u32 << j;
        }
        if is_finite_bits(d_arr[j]) && d_arr[j] >= 0.0 && d_arr[j] < MAX_NONNEG_MOD_SCALE {
            d_open_mask |= 1_u32 << j;
        }
        let va = clamp_nonneg_safe(flush_zero(v_arr[j]).abs(), SE_BUS_ABS_MAX);
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

/// Returns `true` iff `Slot 5` (`d_loss_d_inputs`) must stage `b_all = g_se * a_odd_wf` during
/// `pass0` because `o_tilde / out_gain` cannot recover `b_all` directly (`any_lin_nz || any_out_tiny`).
#[inline]
#[must_use]
pub fn needs_b_staging(params: &QuadParamsUnpacked) -> bool {
    let lm = params.lin_mix.to_array();
    let og = params.out_gain.to_array();
    let mut any_lin = false;
    let mut any_tiny_out = false;
    for j in 0..4 {
        if clamp_range_safe(lm[j], 0.0, 1.0) != 0.0 {
            any_lin = true;
        }
        if clamp_range_safe(og[j], -MAX_OUT_GAIN, MAX_OUT_GAIN).abs() < FAST_OUT_GAIN_MIN_ABS {
            any_tiny_out = true;
        }
    }
    any_lin || any_tiny_out
}

/// Returns `true` when `glu_mix` is identically `0.0` across all `4` lanes (Fast Path 1).
#[inline(always)]
fn is_glu_zero(params: &QuadParamsUnpacked) -> bool {
    let gm = params.glu_mix.to_array();
    let mut any_glu = false;
    for j in 0..4 {
        if clamp_range_safe(gm[j], -MAX_GLU_MIX, MAX_GLU_MIX) != 0.0 {
            any_glu = true;
        }
    }
    !any_glu
}

/// Stages `b_all_{l,r} = g_se * a_odd_wf` into `d_loss_d_inputs[in_base + 0/1]` for `pass0` when
/// `needs_b_staging(params)` is true (disjoint per workgroup `(q_tier, tile)`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn stage_b_all_into_slot5(
    core_inputs: &[Vec4],
    tile_checkpoints: &[Vec4],
    feature_outputs: &[Vec4],
    d_loss_d_inputs: &mut [Vec4],
    s_prev2: &mut [Vec4; TILE_FRAMES],
    u_tone2: &mut [spirv_std::glam::Vec2; crate::abi::ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [spirv_std::glam::Vec2; crate::abi::ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    params: &QuadParamsUnpacked,
    q_tier: usize,
    num_frames: usize,
    num_tiles: usize,
    tile: usize,
    valid_len: usize,
    lid: usize,
    is_phs: bool,
) {
    let i = lid & 255;
    let is_active = i < valid_len;
    let frame = tile * TILE_FRAMES + i;
    let feat_base = (q_tier * num_frames + frame) * FEATURE_VEC4_STRIDE;
    let ve_l = if is_active {
        load_vec4(feature_outputs, feat_base)
    } else {
        Vec4::ZERO
    };
    let ve_r = if is_active {
        load_vec4(feature_outputs, feat_base + 1)
    } else {
        Vec4::ZERO
    };
    let (g_se_l, _, _, _, _) = eval_se_bus_gains_4(ve_l, params.se_w_sq, params.se_d_bus);
    let (g_se_r, _, _, _, _) = eval_se_bus_gains_4(ve_r, params.se_w_sq, params.se_d_bus);

    let mut b_l = Vec4::ZERO;
    let mut b_r = Vec4::ZERO;
    for c in 0..4 {
        let lp = params.lane(c);
        let fwd = super::common::recompute_tile_forward_lane(
            core_inputs,
            tile_checkpoints,
            s_prev2,
            u_tone2,
            v_traj2,
            scan_scratch,
            q_tier,
            num_frames,
            num_tiles,
            tile,
            valid_len,
            i,
            c,
            &lp,
            is_phs,
        );
        let (rp_l, rp_r) = if i == 0 {
            (fwd.adaa_prev_init_l, fwd.adaa_prev_init_r)
        } else {
            (
                scan_scratch[(i - 1) & 255],
                scan_scratch[TILE_FRAMES + ((i - 1) & 255)],
            )
        };
        if is_active {
            let (odd_l, _) = eval_alg_adaa1_scalar_pair(rp_l, fwd.r_curr_l);
            let (odd_r, _) = eval_alg_adaa1_scalar_pair(rp_r, fwd.r_curr_r);
            let wf_l =
                eval_c2_wf_odd_with_derivs(rp_l, fwd.r_curr_l, dnam_kernels::TINY_CROSSING_EPS)
                    .wf_odd;
            let wf_r =
                eval_c2_wf_odd_with_derivs(rp_r, fwd.r_curr_r, dnam_kernels::TINY_CROSSING_EPS)
                    .wf_odd;
            let a_odd_wf_l = (1.0 - lp.wf_mix) * odd_l + lp.wf_mix * wf_l;
            let a_odd_wf_r = (1.0 - lp.wf_mix) * odd_r + lp.wf_mix * wf_r;
            set_lane(&mut b_l, c, g_se_l[c] * a_odd_wf_l);
            set_lane(&mut b_r, c, g_se_r[c] * a_odd_wf_r);
        }
        wg_barrier();
    }
    if is_active {
        let in_base = (q_tier * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
        store_vec4(d_loss_d_inputs, in_base, b_l);
        store_vec4(d_loss_d_inputs, in_base + 1, b_r);
    }
    wg_barrier();
}

/// Computes `r_next_seam` for lane `c` and (when `needs_b_staging`) `b_next_all_{l,r}` at the first
/// frame of `next_tile` using only read-only buffers (zero cross-workgroup `Slot 5` reads).
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
pub fn compute_next_tile_first_seam(
    core_inputs: &[Vec4],
    tile_checkpoints: &[Vec4],
    feature_outputs: &[Vec4],
    params: &QuadParamsUnpacked,
    q_tier: usize,
    num_frames: usize,
    num_tiles: usize,
    next_tile: usize,
    c: usize,
    is_phs: bool,
    stage_b: bool,
) -> ((f32, f32), Vec4, Vec4) {
    let next_frame = next_tile * TILE_FRAMES;
    if next_frame >= num_frames {
        return ((0.0, 0.0), Vec4::ZERO, Vec4::ZERO);
    }
    if !stage_b {
        let lp = params.lane(c);
        let (rl, rr, _, _) = compute_lane_next_tile_first_r_and_aodd_wf(
            core_inputs,
            tile_checkpoints,
            q_tier,
            num_frames,
            num_tiles,
            next_tile,
            c,
            &lp,
            is_phs,
        );
        return ((rl, rr), Vec4::ZERO, Vec4::ZERO);
    }

    let feat_base = (q_tier * num_frames + next_frame) * FEATURE_VEC4_STRIDE;
    let ve_l = load_vec4(feature_outputs, feat_base);
    let ve_r = load_vec4(feature_outputs, feat_base + 1);
    let (g_se_l, _, _, _, _) = eval_se_bus_gains_4(ve_l, params.se_w_sq, params.se_d_bus);
    let (g_se_r, _, _, _, _) = eval_se_bus_gains_4(ve_r, params.se_w_sq, params.se_d_bus);

    let mut r_seam = (0.0_f32, 0.0_f32);
    let mut b_l = Vec4::ZERO;
    let mut b_r = Vec4::ZERO;
    for j in 0..4 {
        let lp = params.lane(j);
        let (rl, rr, a_odd_wf_l, a_odd_wf_r) = compute_lane_next_tile_first_r_and_aodd_wf(
            core_inputs,
            tile_checkpoints,
            q_tier,
            num_frames,
            num_tiles,
            next_tile,
            j,
            &lp,
            is_phs,
        );
        if j == c {
            r_seam = (rl, rr);
        }
        set_lane(&mut b_l, j, g_se_l[j] * a_odd_wf_l);
        set_lane(&mut b_r, j, g_se_r[j] * a_odd_wf_r);
    }
    (r_seam, b_l, b_r)
}

#[derive(Copy, Clone)]
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
    ve_4: Vec4,
    o_tilde_4: Vec4,
    e_tilde_4: Vec4,
    dl_o_4: Vec4,
    dl_e_4: Vec4,
    b_staged_4: Vec4,
    params: &QuadParamsUnpacked,
    lp: &QuadLaneParams,
    c: usize,
    a_odd_wf_c: f32,
    a_even_c: f32,
    g_pre_c: f32,
    tone_c: f32,
    glu_zero: bool,
    stage_b: bool,
) -> ChannelStage3Out {
    let (g_se, p_bus, bt_open_mask, w_open_mask, d_open_mask) =
        eval_se_bus_gains_4(ve_4, params.se_w_sq, params.se_d_bus);

    let lin_arr = params.lin_mix.to_array();
    let glu_arr = params.glu_mix.to_array();
    let og_arr = params.out_gain.to_array();
    let o_arr = o_tilde_4.to_array();
    let e_arr = e_tilde_4.to_array();
    let dlo_arr = dl_o_4.to_array();
    let dle_arr = dl_e_4.to_array();
    let b_staged_arr = b_staged_4.to_array();

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

    let mut og_c = [0.0_f32; 4];
    let mut one_m_lin = [0.0_f32; 4];
    let mut adj_inner = [0.0_f32; 4];
    let mut adj_glu_term = [0.0_f32; 4];
    for j in 0..4 {
        og_c[j] = clamp_range_safe(og_arr[j], -MAX_OUT_GAIN, MAX_OUT_GAIN);
        let lin_j = clamp_range_safe(lin_arr[j], 0.0, 1.0);
        one_m_lin[j] = 1.0 - lin_j;
        let gm_j = clamp_range_safe(glu_arr[j], -MAX_GLU_MIX, MAX_GLU_MIX);
        adj_inner[j] = dle_arr[j] * og_c[j] * one_m_lin[j];
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
            let d_j = clamp_nonneg_safe(get_lane(params.se_d_bus, j), MAX_NONNEG_MOD_SCALE);
            sum_dg_dpbus += dl_gse_j * (-g_se[j] * g_se[j] * d_j);
        }
    }

    let ve_c = get_lane(ve_4, c);
    let ve_clean = flush_zero(ve_c);
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
    let v_abs_open = is_finite_bits(ve_clean) && ve_clean.abs() < SE_BUS_ABS_MAX;
    let sign_v = if ve_clean > 0.0 {
        1.0
    } else if ve_clean < 0.0 {
        -1.0
    } else {
        0.0
    };
    let w_sq_c = clamp_nonneg_safe(get_lane(params.se_w_sq, c), MAX_NONNEG_MOD_SCALE);
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

/// Evaluates full SONA v4.0 Stage-3 pullback `(g_pre -> AlgADAA1 + odd WF -> SE-Bus -> lin_mix ->
/// GLU -> out_gain)` across local frame and inter-tile `r_{256}` seam, applying the mandatory
/// straight-through pre-clamp mask (`|r_curr_unclamped| <= 4096.0`) over `(adj_r_curr + adj_r_prev_next)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn eval_se_and_adaa_local_and_seam_pullback(
    d_loss_d_features: &[Vec4],
    feature_outputs: &[Vec4],
    d_loss_d_inputs: &[Vec4],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    fwd: &LaneForwardFrame,
    params: &QuadParamsUnpacked,
    lp: &QuadLaneParams,
    q_global: usize,
    num_frames: usize,
    num_tiles: usize,
    tile: usize,
    valid_len: usize,
    lid: usize,
    c: usize,
    tiny_eps: f32,
    r_next_seam: (f32, f32),
    b_next_seam: (Vec4, Vec4),
    exit_adj_r_prev_seam: (f32, f32),
    stage_b: bool,
) -> SeAndNonlinPullback {
    let i = lid & 255;
    let is_active = i < valid_len;
    let frame = tile * TILE_FRAMES + i;
    let glu_zero = is_glu_zero(params);
    let one_minus_wf = 1.0 - lp.wf_mix;

    let (r_prev_l, r_prev_r) = if i == 0 {
        (fwd.adaa_prev_init_l, fwd.adaa_prev_init_r)
    } else {
        (
            scan_scratch[(i - 1) & 255],
            scan_scratch[TILE_FRAMES + ((i - 1) & 255)],
        )
    };

    let mut adj_r_curr_l = 0.0_f32;
    let mut adj_r_curr_r = 0.0_f32;
    let mut adj_r_prev_l = 0.0_f32;
    let mut adj_r_prev_r = 0.0_f32;
    let mut dl_veff_feat_l = 0.0_f32;
    let mut dl_veff_feat_r = 0.0_f32;
    let mut st3_l = ChannelStage3Out {
        adj_odd_wf: 0.0,
        adj_even: 0.0,
        adj_g_pre_tone_lin: 0.0,
        adj_veff_se: 0.0,
        grad_se_w_sq: 0.0,
        grad_se_d_bus: 0.0,
        grad_lin_mix: 0.0,
        grad_glu_mix: 0.0,
        grad_out_gain: 0.0,
    };
    let mut st3_r = st3_l;
    let mut grad_wf_mix = 0.0_f32;

    if is_active {
        let (odd_l, even_l) = eval_alg_adaa1_scalar_pair(r_prev_l, fwd.r_curr_l);
        let (odd_r, even_r) = eval_alg_adaa1_scalar_pair(r_prev_r, fwd.r_curr_r);
        let wf_l = eval_c2_wf_odd_with_derivs(r_prev_l, fwd.r_curr_l, tiny_eps);
        let wf_r = eval_c2_wf_odd_with_derivs(r_prev_r, fwd.r_curr_r, tiny_eps);
        let a_odd_wf_l = one_minus_wf * odd_l + lp.wf_mix * wf_l.wf_odd;
        let a_odd_wf_r = one_minus_wf * odd_r + lp.wf_mix * wf_r.wf_odd;

        let feat_base = (q_global * num_frames + frame) * FEATURE_VEC4_STRIDE;
        let dl_ve_l = load_vec4(d_loss_d_features, feat_base);
        let dl_ve_r = load_vec4(d_loss_d_features, feat_base + 1);
        dl_veff_feat_l = get_lane(dl_ve_l, c);
        dl_veff_feat_r = get_lane(dl_ve_r, c);
        let dl_o_l = load_vec4(d_loss_d_features, feat_base + 2);
        let dl_o_r = load_vec4(d_loss_d_features, feat_base + 3);
        let dl_e_l = load_vec4(d_loss_d_features, feat_base + 4);
        let dl_e_r = load_vec4(d_loss_d_features, feat_base + 5);

        let ve_l = load_vec4(feature_outputs, feat_base);
        let ve_r = load_vec4(feature_outputs, feat_base + 1);
        let o_l = load_vec4(feature_outputs, feat_base + 2);
        let o_r = load_vec4(feature_outputs, feat_base + 3);
        let e_l = load_vec4(feature_outputs, feat_base + 4);
        let e_r = load_vec4(feature_outputs, feat_base + 5);

        let (b_staged_l, b_staged_r) = if stage_b {
            let in_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
            (
                load_vec4(d_loss_d_inputs, in_base),
                load_vec4(d_loss_d_inputs, in_base + 1),
            )
        } else {
            (Vec4::ZERO, Vec4::ZERO)
        };

        st3_l = eval_channel_se_glu_out_pullback(
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
        st3_r = eval_channel_se_glu_out_pullback(
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

        let dl = eval_alg_adaa1_backward_f32(r_prev_l, fwd.r_curr_l);
        let dr = eval_alg_adaa1_backward_f32(r_prev_r, fwd.r_curr_r);

        adj_r_curr_l = (one_minus_wf * dl.d_odd_dr_curr + lp.wf_mix * wf_l.d_wf_db)
            * st3_l.adj_odd_wf
            + dl.d_even_dr_curr * st3_l.adj_even;
        adj_r_prev_l = (one_minus_wf * dl.d_odd_dr_prev + lp.wf_mix * wf_l.d_wf_da)
            * st3_l.adj_odd_wf
            + dl.d_even_dr_prev * st3_l.adj_even;

        adj_r_curr_r = (one_minus_wf * dr.d_odd_dr_curr + lp.wf_mix * wf_r.d_wf_db)
            * st3_r.adj_odd_wf
            + dr.d_even_dr_curr * st3_r.adj_even;
        adj_r_prev_r = (one_minus_wf * dr.d_odd_dr_prev + lp.wf_mix * wf_r.d_wf_da)
            * st3_r.adj_odd_wf
            + dr.d_even_dr_prev * st3_r.adj_even;

        if lp.wf_open {
            grad_wf_mix =
                st3_l.adj_odd_wf * (wf_l.wf_odd - odd_l) + st3_r.adj_odd_wf * (wf_r.wf_odd - odd_r);
        }
    }

    wg_barrier();
    scan_scratch[2 * TILE_FRAMES + i] = adj_r_prev_l;
    scan_scratch[3 * TILE_FRAMES + i] = adj_r_prev_r;
    wg_barrier();

    let mut adj_r_clamped_l = adj_r_curr_l;
    let mut adj_r_clamped_r = adj_r_curr_r;
    if i + 1 < valid_len {
        let next_idx = (i + 1) & 255;
        adj_r_clamped_l += scan_scratch[2 * TILE_FRAMES + next_idx];
        adj_r_clamped_r += scan_scratch[3 * TILE_FRAMES + next_idx];
    } else if i == TILE_FRAMES - 1 && tile + 1 < num_tiles {
        let next_frame = (tile + 1) * TILE_FRAMES;
        if next_frame < num_frames {
            let next_feat_base = (q_global * num_frames + next_frame) * FEATURE_VEC4_STRIDE;
            let n_ve_l = load_vec4(feature_outputs, next_feat_base);
            let n_ve_r = load_vec4(feature_outputs, next_feat_base + 1);
            let n_o_l = load_vec4(feature_outputs, next_feat_base + 2);
            let n_o_r = load_vec4(feature_outputs, next_feat_base + 3);
            let n_e_l = load_vec4(feature_outputs, next_feat_base + 4);
            let n_e_r = load_vec4(feature_outputs, next_feat_base + 5);
            let n_dlo_l = load_vec4(d_loss_d_features, next_feat_base + 2);
            let n_dlo_r = load_vec4(d_loss_d_features, next_feat_base + 3);
            let n_dle_l = load_vec4(d_loss_d_features, next_feat_base + 4);
            let n_dle_r = load_vec4(d_loss_d_features, next_feat_base + 5);
            let n_st3_l = eval_channel_se_glu_out_pullback(
                n_ve_l,
                n_o_l,
                n_e_l,
                n_dlo_l,
                n_dle_l,
                b_next_seam.0,
                params,
                lp,
                c,
                0.0,
                0.0,
                0.0,
                0.0,
                glu_zero,
                stage_b,
            );
            let n_st3_r = eval_channel_se_glu_out_pullback(
                n_ve_r,
                n_o_r,
                n_e_r,
                n_dlo_r,
                n_dle_r,
                b_next_seam.1,
                params,
                lp,
                c,
                0.0,
                0.0,
                0.0,
                0.0,
                glu_zero,
                stage_b,
            );
            let dl_seam = eval_alg_adaa1_backward_f32(fwd.r_curr_l, r_next_seam.0);
            let dr_seam = eval_alg_adaa1_backward_f32(fwd.r_curr_r, r_next_seam.1);
            let wf_seam_l = eval_c2_wf_odd_with_derivs(fwd.r_curr_l, r_next_seam.0, tiny_eps);
            let wf_seam_r = eval_c2_wf_odd_with_derivs(fwd.r_curr_r, r_next_seam.1, tiny_eps);
            adj_r_clamped_l += (one_minus_wf * dl_seam.d_odd_dr_prev
                + lp.wf_mix * wf_seam_l.d_wf_da)
                * n_st3_l.adj_odd_wf
                + dl_seam.d_even_dr_prev * n_st3_l.adj_even;
            adj_r_clamped_r += (one_minus_wf * dr_seam.d_odd_dr_prev
                + lp.wf_mix * wf_seam_r.d_wf_da)
                * n_st3_r.adj_odd_wf
                + dr_seam.d_even_dr_prev * n_st3_r.adj_even;
        }
    } else if tile + 1 == num_tiles && valid_len > 0 && i + 1 == valid_len {
        adj_r_clamped_l += exit_adj_r_prev_seam.0;
        adj_r_clamped_r += exit_adj_r_prev_seam.1;
    }
    wg_barrier();

    let adj_r_l = if is_active && fwd.r_curr_open_l {
        adj_r_clamped_l
    } else {
        0.0
    };
    let adj_r_r = if is_active && fwd.r_curr_open_r {
        adj_r_clamped_r
    } else {
        0.0
    };

    let adj_g_pre_l = adj_r_l * lp.adaa_drive * fwd.tone_l + st3_l.adj_g_pre_tone_lin * fwd.tone_l;
    let adj_g_pre_r = adj_r_r * lp.adaa_drive * fwd.tone_r + st3_r.adj_g_pre_tone_lin * fwd.tone_r;
    let (gg0_l, gginf_l, gdpre_l, gdve_l) = eval_g_pre_grads_and_dv(&fwd.g_pre_l, lp, adj_g_pre_l);
    let (gg0_r, gginf_r, gdpre_r, gdve_r) = eval_g_pre_grads_and_dv(&fwd.g_pre_r, lp, adj_g_pre_r);

    let adj_tone_from_stage3_l = if is_active {
        adj_r_l * lp.adaa_drive * fwd.g_pre_l.g_pre + st3_l.adj_g_pre_tone_lin * fwd.g_pre_l.g_pre
    } else {
        0.0
    };
    let adj_tone_from_stage3_r = if is_active {
        adj_r_r * lp.adaa_drive * fwd.g_pre_r.g_pre + st3_r.adj_g_pre_tone_lin * fwd.g_pre_r.g_pre
    } else {
        0.0
    };

    let adj_veff_total_l = if is_active {
        dl_veff_feat_l + st3_l.adj_veff_se + adj_r_l * lp.lru_bias_drive + gdve_l
    } else {
        0.0
    };
    let adj_veff_total_r = if is_active {
        dl_veff_feat_r + st3_r.adj_veff_se + adj_r_r * lp.lru_bias_drive + gdve_r
    } else {
        0.0
    };

    let grad_adaa_drive = if is_active && lp.adaa_drive_open {
        adj_r_l * fwd.g_pre_l.g_pre * fwd.tone_l + adj_r_r * fwd.g_pre_r.g_pre * fwd.tone_r
    } else {
        0.0
    };
    let grad_lru_bias_drive = if is_active && lp.lru_bias_drive_open {
        adj_r_l * fwd.v_eff_l + adj_r_r * fwd.v_eff_r
    } else {
        0.0
    };

    SeAndNonlinPullback {
        adj_r_prev_l,
        adj_r_prev_r,
        adj_veff_total_l,
        adj_veff_total_r,
        adj_tone_from_stage3_l,
        adj_tone_from_stage3_r,
        grad_g0: if is_active { gg0_l + gg0_r } else { 0.0 },
        grad_g_inf: if is_active { gginf_l + gginf_r } else { 0.0 },
        grad_d_pre: if is_active { gdpre_l + gdpre_r } else { 0.0 },
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
