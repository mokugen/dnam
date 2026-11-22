#![forbid(unsafe_code)]

//! Shared forward-tile recomputation and local `(GLU -> PolyADAA1 + Wavefolder -> Sag)` pullback
//! helpers reused identically by `pass_mode = 0` and `pass_mode = 2` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{
    ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S, ADJOINT_V_TRAJ2_VEC2S, TILE_FRAMES,
};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, MIN_SAG_GAIN, QuadLaneParams,
    TILE_CKPT_VEC4_STRIDE, eval_adaa_and_wf_with_derivs, eval_tone_scalar, eval_tpt_svf_scalar,
    flush_zero, get_lane, load_vec4, step_duhem_stereo_scalar, step_tpt_svf_phs_scalar,
    tpt_exclusive_scan_256, wg_barrier,
};
use dnam_kernels::TptScanElem;
use spirv_std::glam::{Vec2, Vec4};

/// Per-frame, per-lane precomputed forward state stored across `pass_mode = 0` and `pass_mode = 2`.
#[derive(Copy, Clone)]
pub struct LaneForwardFrame {
    pub u_l: f32,
    pub u_r: f32,
    pub tone_l: f32,
    pub tone_r: f32,
    pub v_raw_l: f32,
    pub v_raw_r: f32,
    pub v_eff_l: f32,
    pub v_eff_r: f32,
    pub sag_raw_l: f32,
    pub sag_raw_r: f32,
    pub sag_l: f32,
    pub sag_r: f32,
    pub r_curr_l: f32,
    pub r_curr_r: f32,
    pub adaa_prev_init_l: f32,
    pub adaa_prev_init_r: f32,
    pub v_init_l: f32,
    pub v_init_r: f32,
}

/// Output of `eval_glu_pullback_at_frame` for one lane `c` and stereo pair `(L, R)`.
#[derive(Copy, Clone)]
pub struct GluPullback {
    pub adj_veff_feat_l: f32,
    pub adj_veff_feat_r: f32,
    pub adj_odd_wf_l: f32,
    pub adj_odd_wf_r: f32,
    pub adj_even_l: f32,
    pub adj_even_r: f32,
    pub grad_glu_mix: f32,
}

/// Combined local `r_curr` + `v_eff` sensitivities after `(GLU -> PolyADAA1 + Wavefolder -> Sag)`.
#[derive(Copy, Clone)]
pub struct LocalNonlinPullback {
    pub adj_r_l: f32,
    pub adj_r_r: f32,
    pub adj_veff_feat_l: f32,
    pub adj_veff_feat_r: f32,
    pub grad_glu_mix: f32,
    pub grad_wf_mix: f32,
}

#[inline(always)]
fn apply_stereo_link_scalar(vl: f32, vr: f32, lam: f32) -> (f32, f32) {
    let v_mid = 0.5 * (vl + vr);
    (vl + lam * (v_mid - vl), vr + lam * (v_mid - vr))
}

#[inline]
fn step_duhem_tile_into_sram(
    scan_scratch: &[f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    valid_len: usize,
    v_init_l: f32,
    v_init_r: f32,
    lp: &QuadLaneParams,
) {
    let lru_coeffs = lp.to_lru_coeffs();
    let mut vl = v_init_l;
    let mut vr = v_init_r;
    for t in 0..valid_len {
        let i = t & 255;
        let tone_l = scan_scratch[i];
        let bp_l = scan_scratch[TILE_FRAMES + i];
        let tone_r = scan_scratch[2 * TILE_FRAMES + i];
        let bp_r = scan_scratch[3 * TILE_FRAMES + i];
        let (rl, rr, _eff_l, _eff_r) = step_duhem_stereo_scalar(
            tone_l,
            tone_r,
            bp_l,
            bp_r,
            &mut vl,
            &mut vr,
            lp,
            &lru_coeffs,
        );
        v_traj2[i] = Vec2::new(rl, rr);
    }
}

/// Recomputes `s_prev2`, `u_tone2`, `v_traj2`, and `r_curr` (`scan_scratch[0..512]`) from
/// `tile_checkpoints[tile]` for one lane `c in 0..4`.
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

    let in_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
    let (u_l, u_r) = if is_active {
        (
            flush_zero(get_lane(load_vec4(core_inputs, in_base), c)),
            flush_zero(get_lane(load_vec4(core_inputs, in_base + 1), c)),
        )
    } else {
        (0.0, 0.0)
    };

    if is_phs {
        scan_scratch[4 * TILE_FRAMES + i] = u_l;
        scan_scratch[5 * TILE_FRAMES + i] = u_r;
        wg_barrier();
        if i == 0 {
            let mut s1_l = s1_init_l;
            let mut s2_l = s2_init_l;
            for t in 0..valid_len {
                let ti = t & 255;
                u_tone2[ti] = Vec2::new(s1_l, s2_l);
                let u = scan_scratch[4 * TILE_FRAMES + ti];
                let (lp_l, bp_l, hp_l, ap_l) = step_tpt_svf_phs_scalar(u, &mut s1_l, &mut s2_l, lp);
                scan_scratch[ti] = eval_tone_scalar(lp_l, bp_l, hp_l, ap_l, u, lp);
                scan_scratch[TILE_FRAMES + ti] = bp_l;
            }
        }
        if i == 1 {
            let mut s1_r = s1_init_r;
            let mut s2_r = s2_init_r;
            for t in 0..valid_len {
                let ti = t & 255;
                v_traj2[ti] = Vec2::new(s1_r, s2_r);
                let u = scan_scratch[5 * TILE_FRAMES + ti];
                let (lp_r, bp_r, hp_r, ap_r) = step_tpt_svf_phs_scalar(u, &mut s1_r, &mut s2_r, lp);
                scan_scratch[2 * TILE_FRAMES + ti] =
                    eval_tone_scalar(lp_r, bp_r, hp_r, ap_r, u, lp);
                scan_scratch[3 * TILE_FRAMES + ti] = bp_r;
            }
        }
        wg_barrier();
        if is_active {
            let sp_l = u_tone2[i];
            let sp_r = v_traj2[i];
            let tl = scan_scratch[i];
            let tr = scan_scratch[2 * TILE_FRAMES + i];
            s_prev2[i] = Vec4::new(sp_l.x, sp_l.y, sp_r.x, sp_r.y);
            u_tone2[i] = Vec2::new(tl, tr);
        } else {
            s_prev2[i] = Vec4::ZERO;
            u_tone2[i] = Vec2::ZERO;
            v_traj2[i] = Vec2::ZERO;
            scan_scratch[i] = 0.0;
            scan_scratch[TILE_FRAMES + i] = 0.0;
            scan_scratch[2 * TILE_FRAMES + i] = 0.0;
            scan_scratch[3 * TILE_FRAMES + i] = 0.0;
        }
        scan_scratch[4 * TILE_FRAMES + i] = 0.0;
        scan_scratch[5 * TILE_FRAMES + i] = 0.0;
        wg_barrier();
    } else {
        let svf_c = lp.to_svf_coeffs();
        let elem_l = if is_active {
            TptScanElem::from_step(lp.a1, lp.a2, lp.a3, u_l)
        } else {
            TptScanElem::identity()
        };
        let mut s1_prev_l = 0.0_f32;
        let mut s2_prev_l = 0.0_f32;
        let _ = tpt_exclusive_scan_256(
            scan_scratch,
            i,
            valid_len,
            elem_l,
            [s1_init_l, s2_init_l],
            &mut s1_prev_l,
            &mut s2_prev_l,
        );
        let (lp_l, bp_l, hp_l, ap_l) = eval_tpt_svf_scalar(u_l, s1_prev_l, s2_prev_l, &svf_c);
        let tone_l = eval_tone_scalar(lp_l, bp_l, hp_l, ap_l, u_l, lp);
        u_tone2[i] = Vec2::new(tone_l, bp_l);

        let elem_r = if is_active {
            TptScanElem::from_step(lp.a1, lp.a2, lp.a3, u_r)
        } else {
            TptScanElem::identity()
        };
        let mut s1_prev_r = 0.0_f32;
        let mut s2_prev_r = 0.0_f32;
        let _ = tpt_exclusive_scan_256(
            scan_scratch,
            i,
            valid_len,
            elem_r,
            [s1_init_r, s2_init_r],
            &mut s1_prev_r,
            &mut s2_prev_r,
        );
        let (lp_r, bp_r, hp_r, ap_r) = eval_tpt_svf_scalar(u_r, s1_prev_r, s2_prev_r, &svf_c);
        let tone_r = eval_tone_scalar(lp_r, bp_r, hp_r, ap_r, u_r, lp);

        let saved_l = u_tone2[i];
        s_prev2[i] = Vec4::new(s1_prev_l, s2_prev_l, s1_prev_r, s2_prev_r);
        u_tone2[i] = Vec2::new(saved_l.x, tone_r);
        scan_scratch[i] = saved_l.x;
        scan_scratch[TILE_FRAMES + i] = saved_l.y;
        scan_scratch[2 * TILE_FRAMES + i] = tone_r;
        scan_scratch[3 * TILE_FRAMES + i] = bp_r;
        if !is_active {
            v_traj2[i] = Vec2::ZERO;
        }
        wg_barrier();
    }

    if i == 0 {
        step_duhem_tile_into_sram(scan_scratch, v_traj2, valid_len, v_init_l, v_init_r, lp);
    }
    wg_barrier();

    let tone_lr = u_tone2[i];
    let v_raw = v_traj2[i];
    let (v_eff_l, v_eff_r) = apply_stereo_link_scalar(v_raw.x, v_raw.y, lp.stereo_link);
    let sag_raw_l = 1.0 - lp.sag_depth * v_eff_l;
    let sag_raw_r = 1.0 - lp.sag_depth * v_eff_r;
    let sag_l = sag_raw_l.max(MIN_SAG_GAIN);
    let sag_r = sag_raw_r.max(MIN_SAG_GAIN);
    let r_curr_l = if is_active {
        ((tone_lr.x * lp.adaa_drive) * sag_l + lp.lru_bias_drive * v_eff_l) + 0.0
    } else {
        0.0
    };
    let r_curr_r = if is_active {
        ((tone_lr.y * lp.adaa_drive) * sag_r + lp.lru_bias_drive * v_eff_r) + 0.0
    } else {
        0.0
    };
    scan_scratch[i] = r_curr_l;
    scan_scratch[TILE_FRAMES + i] = r_curr_r;
    wg_barrier();

    LaneForwardFrame {
        u_l,
        u_r,
        tone_l: tone_lr.x,
        tone_r: tone_lr.y,
        v_raw_l: v_raw.x,
        v_raw_r: v_raw.y,
        v_eff_l,
        v_eff_r,
        sag_raw_l,
        sag_raw_r,
        sag_l,
        sag_r,
        r_curr_l,
        r_curr_r,
        adaa_prev_init_l,
        adaa_prev_init_r,
        v_init_l,
        v_init_r,
    }
}

/// Evaluates cross-lane bounded `GLU` pullback at `(q_global, frame)` using `Slot 8` (`feature_outputs`).
#[inline]
#[must_use]
pub fn eval_glu_pullback_at_frame(
    d_loss_d_features: &[Vec4],
    feature_outputs: &[Vec4],
    q_global: usize,
    num_frames: usize,
    frame: usize,
    c: usize,
    glu_mix: Vec4,
) -> GluPullback {
    let feat_base = (q_global * num_frames + frame) * FEATURE_VEC4_STRIDE;
    let dl_veff_l = load_vec4(d_loss_d_features, feat_base);
    let dl_veff_r = load_vec4(d_loss_d_features, feat_base + 1);
    let dl_odd_l = load_vec4(d_loss_d_features, feat_base + 2);
    let dl_odd_r = load_vec4(d_loss_d_features, feat_base + 3);
    let dl_even_l = load_vec4(d_loss_d_features, feat_base + 4);
    let dl_even_r = load_vec4(d_loss_d_features, feat_base + 5);
    let fwd_odd_l = load_vec4(feature_outputs, feat_base + 2);
    let fwd_odd_r = load_vec4(feature_outputs, feat_base + 3);

    let c_next = (c + 1) & 3;
    let c_prev = (c + 3) & 3;
    let gm_c = get_lane(glu_mix, c);
    let gm_prev = get_lane(glu_mix, c_prev);

    let adj_odd_wf_l = get_lane(dl_odd_l, c)
        + get_lane(dl_even_l, c) * gm_c * get_lane(fwd_odd_l, c_next)
        + get_lane(dl_even_l, c_prev) * gm_prev * get_lane(fwd_odd_l, c_prev);
    let adj_odd_wf_r = get_lane(dl_odd_r, c)
        + get_lane(dl_even_r, c) * gm_c * get_lane(fwd_odd_r, c_next)
        + get_lane(dl_even_r, c_prev) * gm_prev * get_lane(fwd_odd_r, c_prev);

    let adj_even_l = get_lane(dl_even_l, c);
    let adj_even_r = get_lane(dl_even_r, c);
    let grad_glu_mix = adj_even_l * (get_lane(fwd_odd_l, c) * get_lane(fwd_odd_l, c_next))
        + adj_even_r * (get_lane(fwd_odd_r, c) * get_lane(fwd_odd_r, c_next));

    GluPullback {
        adj_veff_feat_l: get_lane(dl_veff_l, c),
        adj_veff_feat_r: get_lane(dl_veff_r, c),
        adj_odd_wf_l,
        adj_odd_wf_r,
        adj_even_l,
        adj_even_r,
        grad_glu_mix,
    }
}

/// Evaluates `(GLU -> PolyADAA1 + C^2 Wavefolder)` local derivatives, passes `adj_r_prev` back
/// one frame in `scan_scratch[512..1024]`, and adds tile-boundary seam sensitivities at `lid == 255`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn eval_adaa_local_and_seam_pullback(
    d_loss_d_features: &[Vec4],
    feature_outputs: &[Vec4],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    fwd: &LaneForwardFrame,
    q_global: usize,
    num_frames: usize,
    num_tiles: usize,
    tile: usize,
    valid_len: usize,
    lid: usize,
    c: usize,
    glu_mix: Vec4,
    lp: &QuadLaneParams,
    tiny_eps: f32,
    r_next_seam: (f32, f32),
) -> LocalNonlinPullback {
    let i = lid & 255;
    let is_active = i < valid_len;
    let frame = tile * TILE_FRAMES + i;

    let (r_prev_l, r_prev_r) = if i == 0 {
        (fwd.adaa_prev_init_l, fwd.adaa_prev_init_r)
    } else {
        (
            scan_scratch[(i - 1) & 255],
            scan_scratch[TILE_FRAMES + ((i - 1) & 255)],
        )
    };

    let glu = if is_active {
        eval_glu_pullback_at_frame(
            d_loss_d_features,
            feature_outputs,
            q_global,
            num_frames,
            frame,
            c,
            glu_mix,
        )
    } else {
        GluPullback {
            adj_veff_feat_l: 0.0,
            adj_veff_feat_r: 0.0,
            adj_odd_wf_l: 0.0,
            adj_odd_wf_r: 0.0,
            adj_even_l: 0.0,
            adj_even_r: 0.0,
            grad_glu_mix: 0.0,
        }
    };

    let one_minus_wf = 1.0 - lp.wf_mix;
    let mut adj_r_curr_l = 0.0_f32;
    let mut adj_r_curr_r = 0.0_f32;
    let mut adj_r_prev_l = 0.0_f32;
    let mut adj_r_prev_r = 0.0_f32;
    let mut grad_wf_mix = 0.0_f32;

    if is_active {
        let dl = eval_adaa_and_wf_with_derivs(r_prev_l, fwd.r_curr_l, tiny_eps);
        let dr = eval_adaa_and_wf_with_derivs(r_prev_r, fwd.r_curr_r, tiny_eps);

        adj_r_curr_l = (one_minus_wf * dl.d_odd_db + lp.wf_mix * dl.d_wf_db) * glu.adj_odd_wf_l
            + dl.d_even_db * glu.adj_even_l;
        adj_r_prev_l = (one_minus_wf * dl.d_odd_da + lp.wf_mix * dl.d_wf_da) * glu.adj_odd_wf_l
            + dl.d_even_da * glu.adj_even_l;

        adj_r_curr_r = (one_minus_wf * dr.d_odd_db + lp.wf_mix * dr.d_wf_db) * glu.adj_odd_wf_r
            + dr.d_even_db * glu.adj_even_r;
        adj_r_prev_r = (one_minus_wf * dr.d_odd_da + lp.wf_mix * dr.d_wf_da) * glu.adj_odd_wf_r
            + dr.d_even_da * glu.adj_even_r;

        if lp.wf_open {
            grad_wf_mix = glu.adj_odd_wf_l * (dl.wf_odd - dl.odd_poly)
                + glu.adj_odd_wf_r * (dr.wf_odd - dr.odd_poly);
        }
    }

    wg_barrier();
    scan_scratch[2 * TILE_FRAMES + i] = adj_r_prev_l;
    scan_scratch[3 * TILE_FRAMES + i] = adj_r_prev_r;
    wg_barrier();

    let mut adj_r_l = adj_r_curr_l;
    let mut adj_r_r = adj_r_curr_r;
    if i + 1 < valid_len {
        let next_idx = (i + 1) & 255;
        adj_r_l += scan_scratch[2 * TILE_FRAMES + next_idx];
        adj_r_r += scan_scratch[3 * TILE_FRAMES + next_idx];
    } else if i == TILE_FRAMES - 1 && tile + 1 < num_tiles {
        let next_frame = (tile + 1) * TILE_FRAMES;
        if next_frame < num_frames {
            let next_glu = eval_glu_pullback_at_frame(
                d_loss_d_features,
                feature_outputs,
                q_global,
                num_frames,
                next_frame,
                c,
                glu_mix,
            );
            let dl_seam = eval_adaa_and_wf_with_derivs(fwd.r_curr_l, r_next_seam.0, tiny_eps);
            let dr_seam = eval_adaa_and_wf_with_derivs(fwd.r_curr_r, r_next_seam.1, tiny_eps);
            adj_r_l += (one_minus_wf * dl_seam.d_odd_da + lp.wf_mix * dl_seam.d_wf_da)
                * next_glu.adj_odd_wf_l
                + dl_seam.d_even_da * next_glu.adj_even_l;
            adj_r_r += (one_minus_wf * dr_seam.d_odd_da + lp.wf_mix * dr_seam.d_wf_da)
                * next_glu.adj_odd_wf_r
                + dr_seam.d_even_da * next_glu.adj_even_r;
        }
    }
    wg_barrier();

    LocalNonlinPullback {
        adj_r_l,
        adj_r_r,
        adj_veff_feat_l: glu.adj_veff_feat_l,
        adj_veff_feat_r: glu.adj_veff_feat_r,
        grad_glu_mix: glu.grad_glu_mix,
        grad_wf_mix,
    }
}
