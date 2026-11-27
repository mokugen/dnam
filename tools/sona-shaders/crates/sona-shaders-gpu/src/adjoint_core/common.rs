#![forbid(unsafe_code)]

//! Shared forward-tile recomputation and next-tile boundary helpers reused identically by
//! `pass_mode = 0` and `pass_mode = 2` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{
    ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S, ADJOINT_V_TRAJ2_VEC2S, TILE_FRAMES,
};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, GPreTerms, QuadLaneParams, TILE_CKPT_VEC4_STRIDE,
    eval_c2_wf_odd_with_derivs, eval_g_pre_terms_f32, eval_paralru_duhem_affine_step,
    eval_tone_block0_scalar, eval_tpt_svf_folded3_scalar, flush_zero, get_lane, is_duhem_decoupled,
    is_finite_bits, linear_forward_inclusive_scan_256_lr, load_vec4, scrub_endpoint,
    step_duhem_stereo_scalar, step_paralru_duhem_scalar, tpt_const_m_scan_256_lr,
    tpt_forward_exclusive_scan_256, wg_barrier,
};
use dnam_kernels::{
    GpuVec4, KernelVec4, LinearScanElem, MAX_ALG_R, MIN_TPT_G, TINY_CROSSING_EPS, TptScanElem,
    TptSvfCoeffs4, eval_alg_adaa1_vec4, eval_phs_pred_corr_pass_b_coeffs,
    phs_pred_corr_pass_b_scan_elem,
};
use spirv_std::glam::{Vec2, Vec4};

/// Per-frame, per-lane precomputed forward state stored across `pass_mode = 0` and `pass_mode = 2`.
#[derive(Copy, Clone)]
pub struct LaneForwardFrame {
    pub u_l: f32,
    pub u_r: f32,
    pub s_lin_prev_l: Vec2,
    pub s_lin_prev_r: Vec2,
    pub v1_lin_l: f32,
    pub v1_lin_r: f32,
    pub tone_l: f32,
    pub tone_r: f32,
    pub v_raw_l: f32,
    pub v_raw_r: f32,
    pub v_eff_l: f32,
    pub v_eff_r: f32,
    pub g_pre_l: GPreTerms,
    pub g_pre_r: GPreTerms,
    pub r_curr_l: f32,
    pub r_curr_r: f32,
    pub r_curr_open_l: bool,
    pub r_curr_open_r: bool,
    pub adaa_prev_init_l: f32,
    pub adaa_prev_init_r: f32,
    pub v_init_l: f32,
    pub v_init_r: f32,
}

#[inline(always)]
fn apply_stereo_link_scalar(vl: f32, vr: f32, lam: f32) -> (f32, f32) {
    let v_mid = 0.5 * (vl + vr);
    (vl + lam * (v_mid - vl), vr + lam * (v_mid - vr))
}

#[inline]
fn step_duhem_channel_into_scratch(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    valid_len: usize,
    is_r: bool,
    v_init: f32,
    lp: &QuadLaneParams,
) {
    let lru_coeffs = lp.to_lru_coeffs();
    let tone_base = if is_r { 2 * TILE_FRAMES } else { 0 };
    let bp_base = if is_r { 3 * TILE_FRAMES } else { TILE_FRAMES };
    let out_base = if is_r {
        5 * TILE_FRAMES
    } else {
        4 * TILE_FRAMES
    };
    let mut v = v_init;
    for t in 0..valid_len {
        let ti = t & 255;
        let tone = scan_scratch[tone_base + ti];
        let bp = scan_scratch[bp_base + ti];
        let raw = step_paralru_duhem_scalar(tone, bp, &mut v, lp, &lru_coeffs);
        scan_scratch[out_base + ti] = raw;
    }
}

/// Computes `(a1_lin, a2_lin, a3_lin)` for Block `0` Predictor-Corrector Pass A strictly from
/// `(SVF_G, SVF_K)` via `TptSvfCoeffs4::from_gk`, matching `dnam_kernels` (`0-ULP`).
#[inline(always)]
#[must_use]
pub fn phs_pass_a_lin_coeffs(lp: &QuadLaneParams) -> (f32, f32, f32) {
    let g_safe = if lp.svf_g > MIN_TPT_G {
        lp.svf_g
    } else {
        MIN_TPT_G
    };
    let k_safe = if lp.k > 0.0 { lp.k } else { 0.0 };
    let c = TptSvfCoeffs4::from_gk(GpuVec4::splat(g_safe), GpuVec4::splat(k_safe));
    (c.a1.to_array()[0], c.a2.to_array()[0], c.a3.to_array()[0])
}

/// Recomputes `s_prev2`, `u_tone2`, `v_traj2`, `s_lin_prev`, `g_pre`, and `r_curr`
/// (`scan_scratch[0..512]`) from `12`-Vec4 `tile_checkpoints[tile]` for one lane `c in 0..4`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn recompute_tile_forward_lane(
    core_inputs: &[Vec4],
    tile_checkpoints: &[Vec4],
    s_prev2: &mut [Vec4; TILE_FRAMES],
    u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    q_global: usize,
    num_frames: usize,
    num_tiles: usize,
    tile: usize,
    valid_len: usize,
    lid: usize,
    c: usize,
    lp: &QuadLaneParams,
    is_phs: bool,
) -> LaneForwardFrame {
    let i = lid & 255;
    let is_active = i < valid_len;
    let frame = tile * TILE_FRAMES + i;

    let ckpt_base = (q_global * num_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let s1_init_l = get_lane(load_vec4(tile_checkpoints, ckpt_base), c);
    let s1_init_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 1), c);
    let s2_init_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 2), c);
    let s2_init_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 3), c);
    let v_init_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 4), c);
    let v_init_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 5), c);
    let adaa_prev_init_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 6), c);
    let adaa_prev_init_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 7), c);
    let s1_lin_init_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 8), c);
    let s1_lin_init_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 9), c);
    let s2_lin_init_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 10), c);
    let s2_lin_init_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 11), c);

    let in_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
    let (u_l, u_r) = if is_active {
        (
            flush_zero(get_lane(load_vec4(core_inputs, in_base), c)),
            flush_zero(get_lane(load_vec4(core_inputs, in_base + 1), c)),
        )
    } else {
        (0.0, 0.0)
    };

    let (s_lin_prev_l, s_lin_prev_r, v1_lin_l, v1_lin_r) = if is_phs {
        let (lin_a1, lin_a2, lin_a3) = phs_pass_a_lin_coeffs(lp);
        let ([sl1, sl2], [sr1, sr2]) = tpt_const_m_scan_256_lr(
            scan_scratch,
            i,
            valid_len,
            lin_a1,
            lin_a2,
            lin_a3,
            u_l,
            u_r,
            is_active,
            [s1_lin_init_l, s2_lin_init_l],
            [s1_lin_init_r, s2_lin_init_r],
        );
        let v1l = if is_active {
            lin_a1 * sl1 + lin_a2 * (u_l - sl2)
        } else {
            0.0
        };
        let v1r = if is_active {
            lin_a1 * sr1 + lin_a2 * (u_r - sr2)
        } else {
            0.0
        };

        let pb_l = eval_phs_pred_corr_pass_b_coeffs(v1l, lp.svf_g, lp.k, lp.svf_k_nl);
        let elem_b_l = if is_active {
            phs_pred_corr_pass_b_scan_elem(&pb_l, u_l)
        } else {
            TptScanElem::identity()
        };
        let [s1_prev_l, s2_prev_l] =
            tpt_forward_exclusive_scan_256(scan_scratch, i, elem_b_l, [s1_init_l, s2_init_l]);

        let pb_r = eval_phs_pred_corr_pass_b_coeffs(v1r, lp.svf_g, lp.k, lp.svf_k_nl);
        let elem_b_r = if is_active {
            phs_pred_corr_pass_b_scan_elem(&pb_r, u_r)
        } else {
            TptScanElem::identity()
        };
        let [s1_prev_r, s2_prev_r] =
            tpt_forward_exclusive_scan_256(scan_scratch, i, elem_b_r, [s1_init_r, s2_init_r]);

        if is_active {
            let bp_l = pb_l.a1 * s1_prev_l + pb_l.a2 * (u_l - s2_prev_l);
            let lp_l = s2_prev_l + pb_l.a2 * s1_prev_l + pb_l.a3 * (u_l - s2_prev_l);
            let tone_l = eval_tone_block0_scalar(lp_l, bp_l, pb_l.k_eff, u_l, lp);

            let bp_r = pb_r.a1 * s1_prev_r + pb_r.a2 * (u_r - s2_prev_r);
            let lp_r = s2_prev_r + pb_r.a2 * s1_prev_r + pb_r.a3 * (u_r - s2_prev_r);
            let tone_r = eval_tone_block0_scalar(lp_r, bp_r, pb_r.k_eff, u_r, lp);

            s_prev2[i] = Vec4::new(s1_prev_l, s2_prev_l, s1_prev_r, s2_prev_r);
            u_tone2[i] = Vec2::new(tone_l, tone_r);
            scan_scratch[i] = tone_l;
            scan_scratch[TILE_FRAMES + i] = bp_l;
            scan_scratch[2 * TILE_FRAMES + i] = tone_r;
            scan_scratch[3 * TILE_FRAMES + i] = bp_r;
        } else {
            s_prev2[i] = Vec4::ZERO;
            u_tone2[i] = Vec2::ZERO;
            scan_scratch[i] = 0.0;
            scan_scratch[TILE_FRAMES + i] = 0.0;
            scan_scratch[2 * TILE_FRAMES + i] = 0.0;
            scan_scratch[3 * TILE_FRAMES + i] = 0.0;
        }
        wg_barrier();
        (Vec2::new(sl1, sl2), Vec2::new(sr1, sr2), v1l, v1r)
    } else {
        let svf_c = lp.to_svf_coeffs();
        let tone3 = lp.to_tone_folded3();
        let ([s1_prev_l, s2_prev_l], [s1_prev_r, s2_prev_r]) = tpt_const_m_scan_256_lr(
            scan_scratch,
            i,
            valid_len,
            lp.a1,
            lp.a2,
            lp.a3,
            u_l,
            u_r,
            is_active,
            [s1_init_l, s2_init_l],
            [s1_init_r, s2_init_r],
        );
        let (tone_l, bp_l) = eval_tpt_svf_folded3_scalar(u_l, s1_prev_l, s2_prev_l, &svf_c, &tone3);
        let (tone_r, bp_r) = eval_tpt_svf_folded3_scalar(u_r, s1_prev_r, s2_prev_r, &svf_c, &tone3);

        if is_active {
            s_prev2[i] = Vec4::new(s1_prev_l, s2_prev_l, s1_prev_r, s2_prev_r);
            u_tone2[i] = Vec2::new(tone_l, tone_r);
            scan_scratch[i] = tone_l;
            scan_scratch[TILE_FRAMES + i] = bp_l;
            scan_scratch[2 * TILE_FRAMES + i] = tone_r;
            scan_scratch[3 * TILE_FRAMES + i] = bp_r;
        } else {
            s_prev2[i] = Vec4::ZERO;
            u_tone2[i] = Vec2::ZERO;
            scan_scratch[i] = 0.0;
            scan_scratch[TILE_FRAMES + i] = 0.0;
            scan_scratch[2 * TILE_FRAMES + i] = 0.0;
            scan_scratch[3 * TILE_FRAMES + i] = 0.0;
        }
        wg_barrier();
        (Vec2::ZERO, Vec2::ZERO, 0.0, 0.0)
    };

    if is_duhem_decoupled(lp) {
        let (elem_l, elem_r) = if is_active {
            let tone_l = scan_scratch[i];
            let bp_l = scan_scratch[TILE_FRAMES + i];
            let tone_r = scan_scratch[2 * TILE_FRAMES + i];
            let bp_r = scan_scratch[3 * TILE_FRAMES + i];
            (
                eval_paralru_duhem_affine_step(lp.rect_sign * tone_l, bp_l, lp),
                eval_paralru_duhem_affine_step(lp.rect_sign * tone_r, bp_r, lp),
            )
        } else {
            (LinearScanElem::identity(), LinearScanElem::identity())
        };
        let (vl, vr) = linear_forward_inclusive_scan_256_lr(
            scan_scratch,
            i,
            elem_l,
            elem_r,
            v_init_l,
            v_init_r,
        );
        v_traj2[i] = if is_active {
            Vec2::new(flush_zero(vl), flush_zero(vr))
        } else {
            Vec2::ZERO
        };
    } else {
        if i < 2 {
            let is_r = i == 1;
            let v_init = if is_r { v_init_r } else { v_init_l };
            step_duhem_channel_into_scratch(scan_scratch, valid_len, is_r, v_init, lp);
        }
        wg_barrier();
        v_traj2[i] = if is_active {
            Vec2::new(
                scan_scratch[4 * TILE_FRAMES + i],
                scan_scratch[5 * TILE_FRAMES + i],
            )
        } else {
            Vec2::ZERO
        };
    }
    wg_barrier();

    let tone_lr = u_tone2[i];
    let v_raw = v_traj2[i];
    let (v_eff_l, v_eff_r) = apply_stereo_link_scalar(v_raw.x, v_raw.y, lp.stereo_link);
    let g_pre_l = eval_g_pre_terms_f32(v_eff_l, lp);
    let g_pre_r = eval_g_pre_terms_f32(v_eff_r, lp);

    let r_unclamped_l = if is_active {
        (g_pre_l.g_pre * lp.adaa_drive) * tone_lr.x + lp.lru_bias_drive * v_eff_l
    } else {
        0.0
    };
    let r_unclamped_r = if is_active {
        (g_pre_r.g_pre * lp.adaa_drive) * tone_lr.y + lp.lru_bias_drive * v_eff_r
    } else {
        0.0
    };
    let r_curr_open_l =
        is_active && is_finite_bits(r_unclamped_l) && r_unclamped_l.abs() <= MAX_ALG_R;
    let r_curr_open_r =
        is_active && is_finite_bits(r_unclamped_r) && r_unclamped_r.abs() <= MAX_ALG_R;
    let r_curr_l = if is_active {
        scrub_endpoint(r_unclamped_l)
    } else {
        0.0
    };
    let r_curr_r = if is_active {
        scrub_endpoint(r_unclamped_r)
    } else {
        0.0
    };
    scan_scratch[i] = r_curr_l;
    scan_scratch[TILE_FRAMES + i] = r_curr_r;
    wg_barrier();

    LaneForwardFrame {
        u_l,
        u_r,
        s_lin_prev_l,
        s_lin_prev_r,
        v1_lin_l,
        v1_lin_r,
        tone_l: tone_lr.x,
        tone_r: tone_lr.y,
        v_raw_l: v_raw.x,
        v_raw_r: v_raw.y,
        v_eff_l,
        v_eff_r,
        g_pre_l,
        g_pre_r,
        r_curr_l,
        r_curr_r,
        r_curr_open_l,
        r_curr_open_r,
        adaa_prev_init_l,
        adaa_prev_init_r,
        v_init_l,
        v_init_r,
    }
}

/// Evaluates frame `0` of `next_tile` for one lane `c`, returning `(r_next_l, r_next_r, a_odd_wf_l, a_odd_wf_r)`
/// strictly from read-only buffers (used across `255 <-> 256` tile seams).
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
pub fn compute_lane_next_tile_first_r_and_aodd_wf(
    core_inputs: &[Vec4],
    tile_checkpoints: &[Vec4],
    q_global: usize,
    num_frames: usize,
    num_tiles: usize,
    next_tile: usize,
    c: usize,
    lp: &QuadLaneParams,
    is_phs: bool,
) -> (f32, f32, f32, f32) {
    let next_frame = next_tile * TILE_FRAMES;
    if next_frame >= num_frames {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let ckpt_base = (q_global * num_tiles + next_tile) * TILE_CKPT_VEC4_STRIDE;
    let s1_l = get_lane(load_vec4(tile_checkpoints, ckpt_base), c);
    let s1_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 1), c);
    let s2_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 2), c);
    let s2_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 3), c);
    let mut vl = get_lane(load_vec4(tile_checkpoints, ckpt_base + 4), c);
    let mut vr = get_lane(load_vec4(tile_checkpoints, ckpt_base + 5), c);
    let rp_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 6), c);
    let rp_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 7), c);
    let s1_lin_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 8), c);
    let s1_lin_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 9), c);
    let s2_lin_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 10), c);
    let s2_lin_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 11), c);

    let in_base = (q_global * num_frames + next_frame) * CORE_INPUT_VEC4_STRIDE;
    let u_l = flush_zero(get_lane(load_vec4(core_inputs, in_base), c));
    let u_r = flush_zero(get_lane(load_vec4(core_inputs, in_base + 1), c));

    let ((tone_l, bp_l), (tone_r, bp_r)) = if is_phs {
        let (lin_a1, lin_a2, _) = phs_pass_a_lin_coeffs(lp);
        let v1_lin_l = lin_a1 * s1_lin_l + lin_a2 * (u_l - s2_lin_l);
        let v1_lin_r = lin_a1 * s1_lin_r + lin_a2 * (u_r - s2_lin_r);
        let pb_l = eval_phs_pred_corr_pass_b_coeffs(v1_lin_l, lp.svf_g, lp.k, lp.svf_k_nl);
        let pb_r = eval_phs_pred_corr_pass_b_coeffs(v1_lin_r, lp.svf_g, lp.k, lp.svf_k_nl);
        let bpl = pb_l.a1 * s1_l + pb_l.a2 * (u_l - s2_l);
        let lpl = s2_l + pb_l.a2 * s1_l + pb_l.a3 * (u_l - s2_l);
        let bpr = pb_r.a1 * s1_r + pb_r.a2 * (u_r - s2_r);
        let lpr = s2_r + pb_r.a2 * s1_r + pb_r.a3 * (u_r - s2_r);
        (
            (eval_tone_block0_scalar(lpl, bpl, pb_l.k_eff, u_l, lp), bpl),
            (eval_tone_block0_scalar(lpr, bpr, pb_r.k_eff, u_r, lp), bpr),
        )
    } else {
        let svf_c = lp.to_svf_coeffs();
        let tone3 = lp.to_tone_folded3();
        (
            eval_tpt_svf_folded3_scalar(u_l, s1_l, s2_l, &svf_c, &tone3),
            eval_tpt_svf_folded3_scalar(u_r, s1_r, s2_r, &svf_c, &tone3),
        )
    };

    let lru_coeffs = lp.to_lru_coeffs();
    let (_rl, _rr, ve_l, ve_r) = step_duhem_stereo_scalar(
        tone_l,
        tone_r,
        bp_l,
        bp_r,
        &mut vl,
        &mut vr,
        lp,
        &lru_coeffs,
    );
    let gp_l = eval_g_pre_terms_f32(ve_l, lp);
    let gp_r = eval_g_pre_terms_f32(ve_r, lp);
    let r_next_l = scrub_endpoint((gp_l.g_pre * lp.adaa_drive) * tone_l + lp.lru_bias_drive * ve_l);
    let r_next_r = scrub_endpoint((gp_r.g_pre * lp.adaa_drive) * tone_r + lp.lru_bias_drive * ve_r);

    let (odd_l, _, _) = eval_alg_adaa1_vec4(GpuVec4::splat(rp_l), GpuVec4::splat(r_next_l));
    let (odd_r, _, _) = eval_alg_adaa1_vec4(GpuVec4::splat(rp_r), GpuVec4::splat(r_next_r));
    let wf_l = eval_c2_wf_odd_with_derivs(rp_l, r_next_l, TINY_CROSSING_EPS).wf_odd;
    let wf_r = eval_c2_wf_odd_with_derivs(rp_r, r_next_r, TINY_CROSSING_EPS).wf_odd;
    let a_odd_wf_l = (1.0 - lp.wf_mix) * odd_l.to_array()[0] + lp.wf_mix * wf_l;
    let a_odd_wf_r = (1.0 - lp.wf_mix) * odd_r.to_array()[0] + lp.wf_mix * wf_r;
    (r_next_l, r_next_r, a_odd_wf_l, a_odd_wf_r)
}
