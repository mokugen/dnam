#![forbid(unsafe_code)]

//! `adjoint_scan_256` `pass_mode = 2`: full tile BPTT using resolved cross-tile outgoing
//! boundaries (`Slot 7`), inter-tile seam `r_{256}` (`Slot 7 [slot + 20 + 3]`), and `Slot 8`
//! (`feature_outputs`) for `GLU` pullback, emitting `d_loss_d_inputs` (`Slot 5`) and
//! deterministic `param_grads` (`Slot 6`, zero `f32` atomics)
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::{eval_adaa_local_and_seam_pullback, recompute_tile_forward_lane};
use super::pullback::{
    build_tpt_adjoint_elem, build_tpt_phs_adjoint_elem, eval_duhem_and_tone_pullback,
    eval_frame_duhem_terms_lr, eval_sag_and_stereo_pullback, phs_k_eff_loc_bar,
};
use crate::abi::{
    ADJOINT_AUX_FLOATS, ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S, ADJOINT_V_TRAJ2_VEC2S,
    AdjointScanUniforms, EXIT_ADJ_BOUNDARY_VEC4_STRIDE, TILE_FRAMES,
};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, PhsFrameTerms, QUAD_PARAMS_VEC4_STRIDE, QuadParamsUnpacked,
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_LANE_VEC4_STRIDE, eval_tpt_svf_phs_frame,
    eval_tpt_svf_scalar, get_lane, linear_adjoint_reverse_suffix_256_lr, load_vec4, param_slots,
    reduce_batch6_in_scratch, reduce_sum_256_in_aux, set_lane, store_vec4, sub_sat_usize,
    tpt_adjoint_reverse_suffix_256,
};
use dnam_kernels::{LinearScanElem, MIN_TPT_G};
use spirv_std::glam::{Vec2, Vec4};

#[inline(always)]
fn write_param_grad_lane(param_grads: &mut [Vec4], base: usize, slot: usize, c: usize, val: f32) {
    let idx = base + slot;
    let mut vec = if c == 0 {
        Vec4::ZERO
    } else {
        load_vec4(param_grads, idx)
    };
    set_lane(&mut vec, c, val);
    store_vec4(param_grads, idx, vec);
}

/// Executes `adjoint_scan_256` `pass_mode = 2` for one `(q_tier, q_wave, tile)` workgroup.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn run_pass2(
    q_tier: usize,
    q_wave: usize,
    tile_base: usize,
    is_phs: bool,
    tile: usize,
    lid: usize,
    sg_lid: u32,
    uniforms: &AdjointScanUniforms,
    core_inputs: &[Vec4],
    quad_params: &[Vec4],
    tile_checkpoints: &[Vec4],
    d_loss_d_features: &[Vec4],
    d_loss_d_inputs: &mut [Vec4],
    param_grads: &mut [Vec4],
    tile_adj_boundaries: &[Vec4],
    feature_outputs: &[Vec4],
    s_prev2: &mut [Vec4; TILE_FRAMES],
    u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    aux: &mut [f32; ADJOINT_AUX_FLOATS],
) {
    let i = lid & 255;
    let num_frames = uniforms.num_frames as usize;
    let num_tiles = (uniforms.num_tiles as usize).max(1);
    let tile_active = tile < num_tiles;
    let valid_len = if tile_active {
        sub_sat_usize(num_frames, tile * TILE_FRAMES).min(TILE_FRAMES)
    } else {
        0
    };
    let frame = tile * TILE_FRAMES + i;
    let is_active = i < valid_len;
    let tiny_eps = uniforms.tiny_crossing_eps;
    let exit_base = q_wave * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;

    let params = QuadParamsUnpacked::load(quad_params, q_tier * QUAD_PARAMS_VEC4_STRIDE);
    let mut out_du_l = Vec4::ZERO;
    let mut out_du_r = Vec4::ZERO;

    for c in 0..4 {
        let lp = params.lane(c);
        let fwd = recompute_tile_forward_lane(
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

        let slot = tile_base
            + (q_wave * num_tiles + tile) * TILE_ADJ_BOUNDARY_VEC4_STRIDE
            + c * TILE_ADJ_LANE_VEC4_STRIDE;
        let (adj_s_out_l, adj_s_out_r, adj_v_out_l, adj_v_out_r) = if tile_active {
            let s0 = load_vec4(tile_adj_boundaries, slot);
            let s2 = load_vec4(tile_adj_boundaries, slot + 2);
            ([s0.x, s0.y], [s0.z, s0.w], s2.x, s2.y)
        } else {
            ([0.0, 0.0], [0.0, 0.0], 0.0, 0.0)
        };

        let r_next_seam = if i == TILE_FRAMES - 1 && tile + 1 < num_tiles {
            let seam = load_vec4(
                tile_adj_boundaries,
                slot + TILE_ADJ_BOUNDARY_VEC4_STRIDE + 3,
            );
            (seam.x, seam.y)
        } else {
            (0.0, 0.0)
        };
        let exit_adj_r_prev_seam = if tile + 1 == num_tiles && valid_len > 0 && i + 1 == valid_len {
            (
                get_lane(load_vec4(tile_adj_boundaries, exit_base + 6), c),
                get_lane(load_vec4(tile_adj_boundaries, exit_base + 7), c),
            )
        } else {
            (0.0, 0.0)
        };

        let nonlin = eval_adaa_local_and_seam_pullback(
            d_loss_d_features,
            feature_outputs,
            scan_scratch,
            &fwd,
            q_tier,
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
            exit_adj_r_prev_seam,
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
        let (_sum_lru_l, _sum_lru_r, lam_after_l, lam_after_r) =
            linear_adjoint_reverse_suffix_256_lr(
                scan_scratch,
                i,
                lru_elem_l,
                lru_elem_r,
                adj_v_out_l,
                adj_v_out_r,
            );

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

        let (k_loc_l, k_loc_r, tpt_elem_l, tpt_elem_r) = if is_phs {
            let kl = phs_k_eff_loc_bar(
                phs_l.a2,
                phs_l.a3,
                taps_l.1,
                dt.adj_hp_l,
                dt.adj_ap_l,
                dt.adj_v1_l,
                dt.adj_v2_l,
            );
            let kr = phs_k_eff_loc_bar(
                phs_r.a2,
                phs_r.a3,
                taps_r.1,
                dt.adj_hp_r,
                dt.adj_ap_r,
                dt.adj_v1_r,
                dt.adj_v2_r,
            );
            (
                kl,
                kr,
                build_tpt_phs_adjoint_elem(
                    phs_l.a1,
                    phs_l.a2,
                    phs_l.a3,
                    taps_l.1,
                    phs_l.gamma,
                    dt.adj_v1_l,
                    dt.adj_v2_l,
                    kl,
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
                    kr,
                    is_active,
                ),
            )
        } else {
            (
                0.0,
                0.0,
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_l, dt.adj_v2_l, is_active),
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_r, dt.adj_v2_r, is_active),
            )
        };

        let mut psi0_l = 0.0_f32;
        let mut psi1_l = 0.0_f32;
        let _ = tpt_adjoint_reverse_suffix_256(
            scan_scratch,
            i,
            tpt_elem_l,
            adj_s_out_l,
            &mut psi0_l,
            &mut psi1_l,
        );

        let mut psi0_r = 0.0_f32;
        let mut psi1_r = 0.0_f32;
        let _ = tpt_adjoint_reverse_suffix_256(
            scan_scratch,
            i,
            tpt_elem_r,
            adj_s_out_r,
            &mut psi0_r,
            &mut psi1_r,
        );

        let (
            g_a1,
            g_a2,
            g_a3,
            g_svf_k_frame,
            g_svf_two_k_frame,
            g_svf_k_nl_frame,
            g_svf_g_tpt_frame,
            du_l,
            du_r,
        ) = if is_active {
            let tot_v1_l = dt.adj_v1_l + (psi0_l + psi0_l);
            let tot_v2_l = dt.adj_v2_l + (psi1_l + psi1_l);
            let tot_v1_r = dt.adj_v1_r + (psi0_r + psi0_r);
            let tot_v2_r = dt.adj_v2_r + (psi1_r + psi1_r);

            let sp_prev = s_prev2[i];
            let v3_l = fwd.u_l - sp_prev.y;
            let v3_r = fwd.u_r - sp_prev.w;

            if is_phs {
                let v1_l = taps_l.1;
                let v1_r = taps_r.1;
                let k_eff_bar_l = k_loc_l - (v1_l + v1_l) * (phs_l.a2 * psi0_l + phs_l.a3 * psi1_l);
                let k_eff_bar_r = k_loc_r - (v1_r + v1_r) * (phs_r.a2 * psi0_r + phs_r.a3 * psi1_r);
                let g_k_frame = k_eff_bar_l + k_eff_bar_r;
                let g_k_nl_frame = k_eff_bar_l * phs_l.e + k_eff_bar_r * phs_r.e;
                let two_g_safe = (lp.two_svf_g * 0.5).max(MIN_TPT_G) * 2.0;
                let g_tpt_l = tot_v2_l * v1_l
                    + (phs_l.a1 * tot_v1_l + phs_l.a2 * tot_v2_l)
                        * (v3_l - (two_g_safe + phs_l.k_eff) * v1_l);
                let g_tpt_r = tot_v2_r * v1_r
                    + (phs_r.a1 * tot_v1_r + phs_r.a2 * tot_v2_r)
                        * (v3_r - (two_g_safe + phs_r.k_eff) * v1_r);
                let d_u_l = dt.adj_u_nonrec_l + phs_l.a2 * tot_v1_l + phs_l.a3 * tot_v2_l;
                let d_u_r = dt.adj_u_nonrec_r + phs_r.a2 * tot_v1_r + phs_r.a3 * tot_v2_r;
                (
                    0.0,
                    0.0,
                    0.0,
                    g_k_frame,
                    0.0,
                    g_k_nl_frame,
                    g_tpt_l + g_tpt_r,
                    d_u_l,
                    d_u_r,
                )
            } else {
                let a1_g = tot_v1_l * sp_prev.x + tot_v1_r * sp_prev.z;
                let a2_g = (tot_v1_l * v3_l + tot_v2_l * sp_prev.x)
                    + (tot_v1_r * v3_r + tot_v2_r * sp_prev.z);
                let a3_g = tot_v2_l * v3_l + tot_v2_r * v3_r;

                let d_u_l = dt.adj_u_nonrec_l + lp.a2 * tot_v1_l + lp.a3 * tot_v2_l;
                let d_u_r = dt.adj_u_nonrec_r + lp.a2 * tot_v1_r + lp.a3 * tot_v2_r;
                (
                    a1_g,
                    a2_g,
                    a3_g,
                    dt.g_svf_k,
                    dt.g_svf_two_k,
                    0.0,
                    0.0,
                    d_u_l,
                    d_u_r,
                )
            }
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        set_lane(&mut out_du_l, c, du_l);
        set_lane(&mut out_du_r, c, du_r);

        let r0 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                g_a1,
                g_a2,
                g_a3,
                g_svf_k_frame,
                g_svf_two_k_frame,
                dt.g_tone_lp,
            ],
        );
        let r1 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                dt.g_tone_bp,
                dt.g_tone_hp,
                dt.g_tone_ap,
                dt.g_tone_dir,
                dt.g_alpha_rel,
                dt.g_alpha_atk,
            ],
        );
        let r2 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                dt.g_kappa,
                dt.g_gamma,
                dt.g_signed_mix,
                sag.g_stereo_link,
                dt.g_two_svf_g,
                dt.g_rect_sign,
            ],
        );
        let r3 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                sag.g_sag_depth,
                sag.g_adaa_drive,
                sag.g_lru_bias_drive,
                nonlin.grad_wf_mix,
                g_svf_k_nl_frame,
                g_svf_g_tpt_frame,
            ],
        );
        let g_glu_mix = reduce_sum_256_in_aux(aux, i, sg_lid, nonlin.grad_glu_mix);

        if i == 0 && tile_active {
            let pg_base = (q_tier * num_tiles + tile) * QUAD_PARAMS_VEC4_STRIDE;
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A1, c, r0[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A2, c, r0[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A3, c, r0[2]);

            let (g_svf_k, g_svf_k_nl, g_svf_g) = if is_phs {
                (
                    if lp.k >= 0.0 { r0[3] } else { 0.0 },
                    if lp.svf_k_nl >= 0.0 { r3[4] } else { 0.0 },
                    if lp.svf_g >= MIN_TPT_G {
                        r3[5] + 2.0 * r2[4]
                    } else {
                        0.0
                    },
                )
            } else {
                // Total derivative dL/dk chaining through a1, a2, a3, and two_k
                // (da1/dk = -a2*a1, da2/dk = -a2*a2, da3/dk = -a3*a2, d(2k)/dk = 2).
                let g_k_lin =
                    r0[3] + 2.0 * r0[4] - lp.a2 * (r0[0] * lp.a1 + r0[1] * lp.a2 + r0[2] * lp.a3);
                let g_g_lin = if lp.svf_g >= MIN_TPT_G {
                    let denom_g_term = 2.0 * lp.a2 + lp.k * lp.a1;
                    let da1_dg = -lp.a1 * denom_g_term;
                    let da2_dg = lp.a1 - lp.a2 * denom_g_term;
                    let da3_dg = (lp.a2 + lp.a2) - lp.a3 * denom_g_term;
                    r0[0] * da1_dg + r0[1] * da2_dg + r0[2] * da3_dg + 2.0 * r2[4]
                } else {
                    0.0
                };
                (g_k_lin, 0.0, g_g_lin)
            };
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_K, c, g_svf_k);
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_TWO_K, c, r0[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_LP, c, r0[5]);

            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_BP, c, r1[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_HP, c, r1[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_AP, c, r1[2]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_DIR, c, r1[3]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_ALPHA_REL, c, r1[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_ALPHA_ATK, c, r1[5]);

            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_KAPPA, c, r2[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_GAMMA_DUHEM, c, r2[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_SIGNED_MIX, c, r2[2]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_STEREO_LINK, c, r2[3]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TWO_SVF_G, c, r2[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::RECT_SIGN, c, r2[5]);

            write_param_grad_lane(param_grads, pg_base, param_slots::SAG_DEPTH, c, r3[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::ADAA_DRIVE, c, r3[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_BIAS_DRIVE, c, r3[2]);
            write_param_grad_lane(param_grads, pg_base, param_slots::WAVEFOLDER_MIX, c, r3[3]);
            write_param_grad_lane(param_grads, pg_base, param_slots::GLU_MIX, c, g_glu_mix);
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_G, c, g_svf_g);
            write_param_grad_lane(param_grads, pg_base, param_slots::SVF_K_NL, c, g_svf_k_nl);
        }
    }

    let in_base = (q_tier * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
    if is_active {
        store_vec4(d_loss_d_inputs, in_base, out_du_l);
        store_vec4(d_loss_d_inputs, in_base + 1, out_du_r);
    }
}
