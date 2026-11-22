#![forbid(unsafe_code)]

//! `adjoint_scan_256` `pass_mode = 0`: parallel per-tile forward trajectory recomputation,
//! exact inter-tile seam `r_{256}` scalar evaluation, and level-1 reverse `ParaLRU` / `TPT-SVF`
//! boundary summary emission into `Slot 7` (`tile_adj_boundaries`)
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::{eval_adaa_local_and_seam_pullback, recompute_tile_forward_lane};
use super::pullback::{
    build_tpt_adjoint_elem, build_tpt_phs_adjoint_elem, eval_duhem_and_tone_pullback,
    eval_frame_duhem_terms_lr, eval_sag_and_stereo_pullback, phs_k_eff_loc_bar,
};
use crate::abi::{
    ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S, ADJOINT_V_TRAJ2_VEC2S, AdjointScanUniforms,
    TILE_FRAMES,
};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, MIN_SAG_GAIN, PhsFrameTerms, QUAD_PARAMS_VEC4_STRIDE, QuadLaneParams,
    QuadParamsUnpacked, TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_LANE_VEC4_STRIDE,
    TILE_CKPT_VEC4_STRIDE, eval_tone_scalar, eval_tpt_svf_phs_frame, eval_tpt_svf_scalar,
    flush_zero, get_lane, linear_adjoint_reverse_suffix_256_lr, load_vec4,
    step_duhem_stereo_scalar, step_tpt_svf_phs_scalar, store_vec4, sub_sat_usize,
    tpt_adjoint_reverse_suffix_256,
};
use dnam_kernels::LinearScanElem;
use spirv_std::glam::{Vec2, Vec4};

#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_next_tile_first_r(
    core_inputs: &[Vec4],
    tile_checkpoints: &[Vec4],
    q_global: usize,
    num_frames: usize,
    num_tiles: usize,
    next_tile: usize,
    c: usize,
    lp: &QuadLaneParams,
    is_phs: bool,
) -> (f32, f32) {
    let next_frame = next_tile * TILE_FRAMES;
    if next_frame >= num_frames {
        return (0.0, 0.0);
    }
    let ckpt_base = (q_global * num_tiles + next_tile) * TILE_CKPT_VEC4_STRIDE;
    let s1_l = get_lane(load_vec4(tile_checkpoints, ckpt_base), c);
    let s1_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 1), c);
    let s2_l = get_lane(load_vec4(tile_checkpoints, ckpt_base + 2), c);
    let s2_r = get_lane(load_vec4(tile_checkpoints, ckpt_base + 3), c);
    let mut vl = get_lane(load_vec4(tile_checkpoints, ckpt_base + 4), c);
    let mut vr = get_lane(load_vec4(tile_checkpoints, ckpt_base + 5), c);

    let in_base = (q_global * num_frames + next_frame) * CORE_INPUT_VEC4_STRIDE;
    let u_l = flush_zero(get_lane(load_vec4(core_inputs, in_base), c));
    let u_r = flush_zero(get_lane(load_vec4(core_inputs, in_base + 1), c));

    let ((lp_l, bp_l, hp_l, ap_l), (lp_r, bp_r, hp_r, ap_r)) = if is_phs {
        let mut s1_l_tmp = s1_l;
        let mut s2_l_tmp = s2_l;
        let mut s1_r_tmp = s1_r;
        let mut s2_r_tmp = s2_r;
        (
            step_tpt_svf_phs_scalar(u_l, &mut s1_l_tmp, &mut s2_l_tmp, lp),
            step_tpt_svf_phs_scalar(u_r, &mut s1_r_tmp, &mut s2_r_tmp, lp),
        )
    } else {
        let svf_c = lp.to_svf_coeffs();
        (
            eval_tpt_svf_scalar(u_l, s1_l, s2_l, &svf_c),
            eval_tpt_svf_scalar(u_r, s1_r, s2_r, &svf_c),
        )
    };
    let tone_l = eval_tone_scalar(lp_l, bp_l, hp_l, ap_l, u_l, lp);
    let tone_r = eval_tone_scalar(lp_r, bp_r, hp_r, ap_r, u_r, lp);

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
    let sag_l = (1.0 - lp.sag_depth * ve_l).max(MIN_SAG_GAIN);
    let sag_r = (1.0 - lp.sag_depth * ve_r).max(MIN_SAG_GAIN);
    (
        ((tone_l * lp.adaa_drive) * sag_l + lp.lru_bias_drive * ve_l) + 0.0,
        ((tone_r * lp.adaa_drive) * sag_r + lp.lru_bias_drive * ve_r) + 0.0,
    )
}

/// Executes `adjoint_scan_256` `pass_mode = 0` for one `(q_global, tile)` workgroup.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn run_pass0(
    q_global: usize,
    tile: usize,
    lid: usize,
    uniforms: &AdjointScanUniforms,
    core_inputs: &[Vec4],
    quad_params: &[Vec4],
    tile_checkpoints: &[Vec4],
    d_loss_d_features: &[Vec4],
    tile_adj_boundaries: &mut [Vec4],
    feature_outputs: &[Vec4],
    s_prev2: &mut [Vec4; TILE_FRAMES],
    u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
) {
    let i = lid & 255;
    let num_frames = uniforms.num_frames as usize;
    let num_tiles = (uniforms.num_tiles as usize).max(1);
    let valid_len = if tile < num_tiles {
        sub_sat_usize(num_frames, tile * TILE_FRAMES).min(TILE_FRAMES)
    } else {
        0
    };
    let is_active = i < valid_len;
    let tiny_eps = uniforms.tiny_crossing_eps;
    let is_phs = uniforms.block_idx == 0;

    let params = QuadParamsUnpacked::load(quad_params, q_global * QUAD_PARAMS_VEC4_STRIDE);

    for c in 0..4 {
        let lp = params.lane(c);
        let fwd = recompute_tile_forward_lane(
            core_inputs,
            tile_checkpoints,
            s_prev2,
            u_tone2,
            v_traj2,
            scan_scratch,
            q_global,
            num_frames,
            num_tiles,
            tile,
            valid_len,
            i,
            c,
            &lp,
            is_phs,
        );
        let r_first_l = fwd.r_curr_l;
        let r_first_r = fwd.r_curr_r;

        let r_next_seam = if i == TILE_FRAMES - 1 && tile + 1 < num_tiles {
            compute_next_tile_first_r(
                core_inputs,
                tile_checkpoints,
                q_global,
                num_frames,
                num_tiles,
                tile + 1,
                c,
                &lp,
                is_phs,
            )
        } else {
            (0.0, 0.0)
        };

        let nonlin = eval_adaa_local_and_seam_pullback(
            d_loss_d_features,
            feature_outputs,
            scan_scratch,
            &fwd,
            q_global,
            num_frames,
            num_tiles,
            tile,
            valid_len,
            i,
            c,
            params.glu_mix,
            &lp,
            tiny_eps,
            r_next_seam,
        );
        let sag = eval_sag_and_stereo_pullback(&fwd, &nonlin, &lp, is_active);

        let sp = s_prev2[i];
        let (phs_l, phs_r, taps_l, taps_r) = if is_phs {
            let fl = eval_tpt_svf_phs_frame(fwd.u_l, sp.x, sp.y, &lp);
            let fr = eval_tpt_svf_phs_frame(fwd.u_r, sp.z, sp.w, &lp);
            (fl, fr, fl.taps, fr.taps)
        } else {
            let svf_c = lp.to_svf_coeffs();
            (
                PhsFrameTerms::zero(),
                PhsFrameTerms::zero(),
                eval_tpt_svf_scalar(fwd.u_l, sp.x, sp.y, &svf_c),
                eval_tpt_svf_scalar(fwd.u_r, sp.z, sp.w, &svf_c),
            )
        };
        let (tl, tr) = eval_frame_duhem_terms_lr(v_traj2, &fwd, taps_l.1, taps_r.1, i, &lp);

        let lru_elem_l = if is_active {
            LinearScanElem {
                a: tl.jac,
                b: tl.jac * sag.adj_v_raw_l,
            }
        } else {
            LinearScanElem::identity()
        };
        let lru_elem_r = if is_active {
            LinearScanElem {
                a: tr.jac,
                b: tr.jac * sag.adj_v_raw_r,
            }
        } else {
            LinearScanElem::identity()
        };
        let (sum_lru_l, sum_lru_r, lam_after_l, lam_after_r) =
            linear_adjoint_reverse_suffix_256_lr(scan_scratch, i, lru_elem_l, lru_elem_r, 0.0, 0.0);

        let lam_l = if is_active {
            sag.adj_v_raw_l + lam_after_l
        } else {
            0.0
        };
        let lam_r = if is_active {
            sag.adj_v_raw_r + lam_after_r
        } else {
            0.0
        };

        let (ke_l, two_ke_l, ke_r, two_ke_r) = if is_phs {
            (
                phs_l.k_eff,
                phs_l.k_eff + phs_l.k_eff,
                phs_r.k_eff,
                phs_r.k_eff + phs_r.k_eff,
            )
        } else {
            (lp.k, lp.two_k, lp.k, lp.two_k)
        };
        let dt = eval_duhem_and_tone_pullback(
            &fwd, &sag, &tl, &tr, lam_l, lam_r, taps_l, taps_r, ke_l, two_ke_l, ke_r, two_ke_r,
            &lp, is_active,
        );

        let (tpt_elem_l, tpt_elem_r) = if is_phs {
            let k_loc_l = phs_k_eff_loc_bar(
                phs_l.a2,
                phs_l.a3,
                taps_l.1,
                dt.adj_hp_l,
                dt.adj_ap_l,
                dt.adj_v1_l,
                dt.adj_v2_l,
            );
            let k_loc_r = phs_k_eff_loc_bar(
                phs_r.a2,
                phs_r.a3,
                taps_r.1,
                dt.adj_hp_r,
                dt.adj_ap_r,
                dt.adj_v1_r,
                dt.adj_v2_r,
            );
            (
                build_tpt_phs_adjoint_elem(
                    phs_l.a1,
                    phs_l.a2,
                    phs_l.a3,
                    taps_l.1,
                    phs_l.gamma,
                    dt.adj_v1_l,
                    dt.adj_v2_l,
                    k_loc_l,
                    is_active,
                ),
                build_tpt_phs_adjoint_elem(
                    phs_r.a1,
                    phs_r.a2,
                    phs_r.a3,
                    taps_r.1,
                    phs_r.gamma,
                    dt.adj_v1_r,
                    dt.adj_v2_r,
                    k_loc_r,
                    is_active,
                ),
            )
        } else {
            (
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_l, dt.adj_v2_l, is_active),
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_r, dt.adj_v2_r, is_active),
            )
        };

        let mut psi0_l = 0.0_f32;
        let mut psi1_l = 0.0_f32;
        let sum_tpt_l = tpt_adjoint_reverse_suffix_256(
            scan_scratch,
            i,
            tpt_elem_l,
            [0.0, 0.0],
            &mut psi0_l,
            &mut psi1_l,
        );

        let mut psi0_r = 0.0_f32;
        let mut psi1_r = 0.0_f32;
        let sum_tpt_r = tpt_adjoint_reverse_suffix_256(
            scan_scratch,
            i,
            tpt_elem_r,
            [0.0, 0.0],
            &mut psi0_r,
            &mut psi1_r,
        );

        if i == 0 && tile < num_tiles {
            let slot = (q_global * num_tiles + tile) * TILE_ADJ_BOUNDARY_VEC4_STRIDE
                + c * TILE_ADJ_LANE_VEC4_STRIDE;
            store_vec4(
                tile_adj_boundaries,
                slot,
                Vec4::new(sum_tpt_l.t00, sum_tpt_l.t01, sum_tpt_l.t10, sum_tpt_l.t11),
            );
            store_vec4(
                tile_adj_boundaries,
                slot + 1,
                Vec4::new(sum_tpt_l.g0, sum_tpt_l.g1, sum_tpt_r.g0, sum_tpt_r.g1),
            );
            store_vec4(
                tile_adj_boundaries,
                slot + 2,
                Vec4::new(sum_lru_l.a, sum_lru_l.b, sum_lru_r.a, sum_lru_r.b),
            );
            store_vec4(
                tile_adj_boundaries,
                slot + 3,
                Vec4::new(r_first_l, r_first_r, 0.0, 0.0),
            );
            store_vec4(
                tile_adj_boundaries,
                slot + 4,
                Vec4::new(sum_tpt_r.t00, sum_tpt_r.t01, sum_tpt_r.t10, sum_tpt_r.t11),
            );
        }
    }
}
