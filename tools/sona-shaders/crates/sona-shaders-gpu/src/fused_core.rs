#![forbid(unsafe_code)]

//! Persistent `256`-frame forward scan kernel (`fused_core_scan_256_impl`) executing
//! `3-tap Folded SVF / Pred-Corr PHS -> Smooth ParaLRU + Stereo Link -> g_pre -> AlgADAA1(4096) + Wavefolder -> SE-Bus -> GLU -> out_gain`
//! strictly inside `15,360 B` (`15.0 KiB`) workgroup SRAM across `2 × 128` sequential sub-tiles
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod fixed_point;

use crate::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE,
    FORWARD_AUX_SCRATCH_FLOATS, FORWARD_LANE_BUF_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS,
    FUSED_FLAG_FIXED_POINT_SCAN, FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE,
    TILE_FRAMES, resolve_quad_idx,
};
use crate::core_math::{
    QuadParamsUnpacked, clamp_nonneg_safe, clamp_range_safe, div_ceil_divfree,
    eval_paralru_duhem_affine_step, eval_tone_block0_scalar, eval_tpt_svf_folded3_scalar,
    flush_zero, get_lane, gpu_to_vec4, is_duhem_decoupled, linear_forward_exclusive_scan_128_pair,
    load_vec4, scrub_endpoint, step_paralru_duhem_scalar, store_vec4, sub_sat_usize,
    tpt_const_m_scan_128_pair, tpt_exclusive_scan_128_pair, vec4_to_gpu, wg_barrier,
};
use dnam_kernels::{
    GpuVec4, InterleavedStereo8, KernelVec4, LinearScanElem, MAX_ADAA_DRIVE, MAX_GLU_MIX,
    MAX_LRU_BIAS_DRIVE, MAX_OUT_GAIN, MIN_TPT_G, StereoVec8, TptScanElem, TptSvfCoeffs4,
    apply_stereo_link_vec4_pair, eval_alg_adaa1_with_wavefolder_vec4, eval_bounded_glu_stereo8,
    eval_g_pre_vec4, eval_phs_pred_corr_pass_b_coeffs, eval_se_bus_gain_vec4,
    phs_pred_corr_pass_b_scan_elem, scrub_clamped_alg_endpoint_vec4,
};
use fixed_point::{lane_slot, run_duhem_damped_deer_subtile_lane};
use spirv_std::glam::{UVec3, Vec4};

const SUBTILE_FRAMES: usize = TILE_FRAMES / 2;

#[inline(always)]
fn canonicalize_vec4_zero(v: Vec4) -> Vec4 {
    Vec4::new(
        flush_zero(v.x),
        flush_zero(v.y),
        flush_zero(v.z),
        flush_zero(v.w),
    )
}

#[inline(always)]
fn scrub_vec4_endpoints(v: Vec4) -> GpuVec4 {
    GpuVec4::from_array([
        scrub_endpoint(v.x),
        scrub_endpoint(v.y),
        scrub_endpoint(v.z),
        scrub_endpoint(v.w),
    ])
}

#[inline(always)]
fn clamp_vec4_range(v: Vec4, lo: f32, hi: f32) -> GpuVec4 {
    GpuVec4::from_array([
        clamp_range_safe(v.x, lo, hi),
        clamp_range_safe(v.y, lo, hi),
        clamp_range_safe(v.z, lo, hi),
        clamp_range_safe(v.w, lo, hi),
    ])
}

#[inline(always)]
fn load_ckpt_to_aux(
    tile_checkpoints: &[Vec4],
    q_tier: usize,
    ckpt_tiles: usize,
    tile: usize,
    aux_scratch: &mut [f32; FORWARD_AUX_SCRATCH_FLOATS],
) {
    let ckpt_base = (q_tier * ckpt_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let s1_l = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base)).to_array();
    let s1_r = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 1)).to_array();
    let s2_l = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 2)).to_array();
    let s2_r = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 3)).to_array();
    let v_l = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 4)).to_array();
    let v_r = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 5)).to_array();
    let ap_l = scrub_vec4_endpoints(load_vec4(tile_checkpoints, ckpt_base + 6)).to_array();
    let ap_r = scrub_vec4_endpoints(load_vec4(tile_checkpoints, ckpt_base + 7)).to_array();
    let s1_lin_l = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 8)).to_array();
    let s1_lin_r = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 9)).to_array();
    let s2_lin_l = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 10)).to_array();
    let s2_lin_r = canonicalize_vec4_zero(load_vec4(tile_checkpoints, ckpt_base + 11)).to_array();
    for c in 0..4 {
        aux_scratch[c] = s1_l[c];
        aux_scratch[4 + c] = s2_l[c];
        aux_scratch[8 + c] = s1_r[c];
        aux_scratch[12 + c] = s2_r[c];
        aux_scratch[16 + c] = v_l[c];
        aux_scratch[20 + c] = v_r[c];
        aux_scratch[24 + c] = ap_l[c];
        aux_scratch[28 + c] = ap_r[c];
        aux_scratch[32 + c] = s1_lin_l[c];
        aux_scratch[36 + c] = s2_lin_l[c];
        aux_scratch[40 + c] = s1_lin_r[c];
        aux_scratch[44 + c] = s2_lin_r[c];
    }
}

#[inline(always)]
fn store_aux_to_ckpt(
    tile_checkpoints: &mut [Vec4],
    q_tier: usize,
    ckpt_tiles: usize,
    tile: usize,
    aux_scratch: &[f32; FORWARD_AUX_SCRATCH_FLOATS],
) {
    let ckpt_base = (q_tier * ckpt_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let pack = |off: usize| {
        canonicalize_vec4_zero(Vec4::new(
            aux_scratch[off],
            aux_scratch[off + 1],
            aux_scratch[off + 2],
            aux_scratch[off + 3],
        ))
    };
    store_vec4(tile_checkpoints, ckpt_base, pack(0));
    store_vec4(tile_checkpoints, ckpt_base + 1, pack(8));
    store_vec4(tile_checkpoints, ckpt_base + 2, pack(4));
    store_vec4(tile_checkpoints, ckpt_base + 3, pack(12));
    store_vec4(tile_checkpoints, ckpt_base + 4, pack(16));
    store_vec4(tile_checkpoints, ckpt_base + 5, pack(20));
    store_vec4(tile_checkpoints, ckpt_base + 6, pack(24));
    store_vec4(tile_checkpoints, ckpt_base + 7, pack(28));
    store_vec4(tile_checkpoints, ckpt_base + 8, pack(32));
    store_vec4(tile_checkpoints, ckpt_base + 9, pack(40));
    store_vec4(tile_checkpoints, ckpt_base + 10, pack(36));
    store_vec4(tile_checkpoints, ckpt_base + 11, pack(44));
}

/// Executes the persistent wave-mapped `2 × 128` sub-tile forward `CoreQuad` scan (`15,360 B` SRAM).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fused_core_scan_256_impl(
    gid: UVec3,
    wg_id: UVec3,
    num_wg: UVec3,
    lid_u32: u32,
    uniforms: &FusedScanUniforms,
    core_inputs: &[Vec4],
    quad_params: &[Vec4],
    tile_checkpoints: &mut [Vec4],
    feature_outputs: &mut [Vec4],
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    lane_buf: &mut [f32; FORWARD_LANE_BUF_FLOATS],
    aux_scratch: &mut [f32; FORWARD_AUX_SCRATCH_FLOATS],
) {
    let _ = gid;
    let lid = (lid_u32 as usize) & 255;
    let t128 = lid & 127;
    let is_r128 = (lid >> 7) != 0;

    let q_local = wg_id.x as usize;
    let win_idx = wg_id.y as usize;
    let num_windows = (num_wg.y as usize).max(1);
    let stem = wg_id.z as usize;

    let num_quads = (uniforms.num_quads as usize).max(1);
    let quad_idx = resolve_quad_idx(uniforms.block_idx, q_local, num_quads);
    if quad_idx >= num_quads {
        return;
    }
    let q_tier = stem * num_quads + quad_idx;

    let is_direct = uniforms.block_idx == BLOCK_IDX_DIRECT_LINEAR;
    let use_fixed_point = !is_direct && (uniforms.block_idx & FUSED_FLAG_FIXED_POINT_SCAN) != 0;
    let is_phs = !is_direct && (uniforms.block_idx & 0x3) == 0;

    let num_frames = uniforms.num_frames as usize;
    let total_tiles = if num_frames > 0 {
        ((num_frames - 1) >> 8) + 1
    } else {
        0
    };
    let ckpt_tiles = (uniforms.num_tiles as usize).max(1);
    let max_tiles = (uniforms.num_tiles as usize).min(total_tiles);
    let tile_start = (uniforms.tile_start as usize).min(max_tiles);

    let (t_begin, t_end) = if num_windows <= 1 {
        (tile_start, max_tiles)
    } else {
        let span = sub_sat_usize(max_tiles, tile_start);
        let per_win = div_ceil_divfree(span, num_windows);
        let begin = (tile_start + win_idx * per_win).min(max_tiles);
        let end = (begin + per_win).min(max_tiles);
        (begin, end)
    };

    let q_base = q_tier * QUAD_PARAMS_VEC4_STRIDE;
    let params = QuadParamsUnpacked::load(quad_params, q_base);
    let stereo_link_g = GpuVec4::from_array([
        clamp_nonneg_safe(params.lru_stereo_link.x, 1.0),
        clamp_nonneg_safe(params.lru_stereo_link.y, 1.0),
        clamp_nonneg_safe(params.lru_stereo_link.z, 1.0),
        clamp_nonneg_safe(params.lru_stereo_link.w, 1.0),
    ]);
    let g0_g = vec4_to_gpu(params.g0);
    let g_inf_g = vec4_to_gpu(params.g_inf);
    let d_pre_g = vec4_to_gpu(params.d_pre);
    let se_w_sq_g = vec4_to_gpu(params.se_w_sq);
    let se_d_bus_g = vec4_to_gpu(params.se_d_bus);
    let adaa_drive_g = clamp_vec4_range(params.adaa_drive, -MAX_ADAA_DRIVE, MAX_ADAA_DRIVE);
    let lru_bias_g = clamp_vec4_range(
        params.lru_bias_drive,
        -MAX_LRU_BIAS_DRIVE,
        MAX_LRU_BIAS_DRIVE,
    );
    let lin_mix_g = clamp_vec4_range(params.lin_mix, 0.0, 1.0);
    let wf_mix_g = clamp_vec4_range(params.wavefolder_mix, 0.0, 1.0);
    let glu_mix_g = clamp_vec4_range(params.glu_mix, -MAX_GLU_MIX, MAX_GLU_MIX);
    let out_gain_g = clamp_vec4_range(params.out_gain, -MAX_OUT_GAIN, MAX_OUT_GAIN);
    let one_minus_lin_g = GpuVec4::splat(1.0).sub(lin_mix_g);

    let resuming_from_ckpt = num_windows <= 1 && tile_start > 0 && tile_start < ckpt_tiles;
    if lid == 0 {
        if resuming_from_ckpt {
            load_ckpt_to_aux(
                tile_checkpoints,
                q_tier,
                ckpt_tiles,
                tile_start,
                aux_scratch,
            );
        } else {
            for k in 0..48 {
                aux_scratch[k] = 0.0;
            }
        }
    }
    wg_barrier();

    for tile in t_begin..t_end {
        let tile_frame_base = tile * TILE_FRAMES;
        if lid == 0 && !(resuming_from_ckpt && tile == tile_start) {
            store_aux_to_ckpt(tile_checkpoints, q_tier, ckpt_tiles, tile, aux_scratch);
        }
        wg_barrier();

        for sub in 0..2 {
            let sub_frame_base = tile_frame_base + sub * SUBTILE_FRAMES;
            let sub_valid_len = sub_sat_usize(num_frames, sub_frame_base).min(SUBTILE_FRAMES);
            if sub_valid_len > 0 {
                let active128 = t128 < sub_valid_len;
                let frame = sub_frame_base + t128;
                let in_base = (q_tier * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
                let (in_l, in_r) = if active128 {
                    (
                        load_vec4(core_inputs, in_base),
                        load_vec4(core_inputs, in_base + 1),
                    )
                } else {
                    (Vec4::ZERO, Vec4::ZERO)
                };

                // Stage 1: 3-tap folded TPT-SVF or 2-pass O(log2 128) Predictor-Corrector PHS.
                for c in 0..4 {
                    let lp = params.lane(c);
                    let u_my = if active128 {
                        flush_zero(get_lane(if is_r128 { in_r } else { in_l }, c))
                    } else {
                        0.0
                    };
                    let s1_idx = if is_r128 { 8 + c } else { c };
                    let s2_idx = if is_r128 { 12 + c } else { 4 + c };

                    if is_phs {
                        let s1_lin_idx = if is_r128 { 40 + c } else { 32 + c };
                        let s2_lin_idx = if is_r128 { 44 + c } else { 36 + c };
                        let g_safe = if lp.svf_g > MIN_TPT_G {
                            lp.svf_g
                        } else {
                            MIN_TPT_G
                        };
                        let k_safe = if lp.k > 0.0 { lp.k } else { 0.0 };
                        let lin_c =
                            TptSvfCoeffs4::from_gk(GpuVec4::splat(g_safe), GpuVec4::splat(k_safe));
                        let lin_a1 = lin_c.a1.to_array()[0];
                        let lin_a2 = lin_c.a2.to_array()[0];
                        let lin_a3 = lin_c.a3.to_array()[0];
                        let mut s1_lin_prev = 0.0_f32;
                        let mut s2_lin_prev = 0.0_f32;
                        let s_lin_end = tpt_const_m_scan_128_pair(
                            scan_scratch,
                            lid,
                            sub_valid_len,
                            lin_a1,
                            lin_a2,
                            lin_a3,
                            u_my,
                            active128,
                            [aux_scratch[s1_lin_idx], aux_scratch[s2_lin_idx]],
                            &mut s1_lin_prev,
                            &mut s2_lin_prev,
                        );
                        let v1_lin = lin_a1 * s1_lin_prev + lin_a2 * (u_my - s2_lin_prev);
                        let pass_b =
                            eval_phs_pred_corr_pass_b_coeffs(v1_lin, lp.svf_g, lp.k, lp.svf_k_nl);
                        let elem_b = if active128 {
                            phs_pred_corr_pass_b_scan_elem(&pass_b, u_my)
                        } else {
                            TptScanElem::identity()
                        };
                        let mut s1_b_prev = 0.0_f32;
                        let mut s2_b_prev = 0.0_f32;
                        let s_b_end = tpt_exclusive_scan_128_pair(
                            scan_scratch,
                            lid,
                            sub_valid_len,
                            elem_b,
                            [aux_scratch[s1_idx], aux_scratch[s2_idx]],
                            &mut s1_b_prev,
                            &mut s2_b_prev,
                        );
                        if active128 {
                            let bp_s = pass_b.a1 * s1_b_prev + pass_b.a2 * (u_my - s2_b_prev);
                            let lp_s =
                                s2_b_prev + pass_b.a2 * s1_b_prev + pass_b.a3 * (u_my - s2_b_prev);
                            let tone_s =
                                eval_tone_block0_scalar(lp_s, bp_s, pass_b.k_eff, u_my, &lp);
                            let tone_f = if is_r128 { 2 } else { 0 };
                            lane_buf[lane_slot(c, tone_f, t128)] = tone_s;
                            lane_buf[lane_slot(c, tone_f + 1, t128)] = bp_s;
                        }
                        if lid == 0 {
                            aux_scratch[c] = flush_zero(s_b_end[0]);
                            aux_scratch[4 + c] = flush_zero(s_b_end[1]);
                            aux_scratch[32 + c] = flush_zero(s_lin_end[0]);
                            aux_scratch[36 + c] = flush_zero(s_lin_end[1]);
                        }
                        if lid == SUBTILE_FRAMES {
                            aux_scratch[8 + c] = flush_zero(s_b_end[0]);
                            aux_scratch[12 + c] = flush_zero(s_b_end[1]);
                            aux_scratch[40 + c] = flush_zero(s_lin_end[0]);
                            aux_scratch[44 + c] = flush_zero(s_lin_end[1]);
                        }
                        wg_barrier();
                    } else {
                        let svf_c = lp.to_svf_coeffs();
                        let tone3 = lp.to_tone_folded3();
                        let mut s1_prev = 0.0_f32;
                        let mut s2_prev = 0.0_f32;
                        let s_end = tpt_const_m_scan_128_pair(
                            scan_scratch,
                            lid,
                            sub_valid_len,
                            lp.a1,
                            lp.a2,
                            lp.a3,
                            u_my,
                            active128,
                            [aux_scratch[s1_idx], aux_scratch[s2_idx]],
                            &mut s1_prev,
                            &mut s2_prev,
                        );
                        if active128 {
                            let (tone_s, bp_s) =
                                eval_tpt_svf_folded3_scalar(u_my, s1_prev, s2_prev, &svf_c, &tone3);
                            let tone_f = if is_r128 { 2 } else { 0 };
                            lane_buf[lane_slot(c, tone_f, t128)] = tone_s;
                            lane_buf[lane_slot(c, tone_f + 1, t128)] = bp_s;
                        }
                        if lid == 0 {
                            aux_scratch[c] = flush_zero(s_end[0]);
                            aux_scratch[4 + c] = flush_zero(s_end[1]);
                        }
                        if lid == SUBTILE_FRAMES {
                            aux_scratch[8 + c] = flush_zero(s_end[0]);
                            aux_scratch[12 + c] = flush_zero(s_end[1]);
                        }
                        wg_barrier();
                    }
                }

                // Stage 2: Smooth SONA v4.0 Selective-Duhem ParaLRU storing `v_raw` into `bp` slots.
                for c in 0..4 {
                    let lp = params.lane(c);
                    if is_duhem_decoupled(&lp) {
                        let tone_f = if is_r128 { 2 } else { 0 };
                        let bp_f = tone_f + 1;
                        let elem = if active128 {
                            let tone_my = lane_buf[lane_slot(c, tone_f, t128)];
                            let bp_my = lane_buf[lane_slot(c, bp_f, t128)];
                            eval_paralru_duhem_affine_step(lp.rect_sign * tone_my, bp_my, &lp)
                        } else {
                            LinearScanElem::identity()
                        };
                        let v_init = aux_scratch[16 + (if is_r128 { 4 } else { 0 }) + c];
                        let (v_incl, v_end) = linear_forward_exclusive_scan_128_pair(
                            scan_scratch,
                            lid,
                            sub_valid_len,
                            elem,
                            v_init,
                        );
                        if active128 {
                            lane_buf[lane_slot(c, bp_f, t128)] = flush_zero(v_incl);
                        }
                        if lid == 0 {
                            aux_scratch[16 + c] = flush_zero(v_end);
                        }
                        if lid == SUBTILE_FRAMES {
                            aux_scratch[20 + c] = flush_zero(v_end);
                        }
                        wg_barrier();
                    }
                }

                if !use_fixed_point {
                    if lid < 8 {
                        let c = lid >> 1;
                        let is_r = (lid & 1) != 0;
                        let lp = params.lane(c);
                        if !is_duhem_decoupled(&lp) {
                            let lru_c = lp.to_lru_coeffs();
                            let tone_f = if is_r { 2 } else { 0 };
                            let bp_f = tone_f + 1;
                            let v_idx = if is_r { 20 + c } else { 16 + c };
                            let mut v = aux_scratch[v_idx];
                            for k in 0..sub_valid_len {
                                let ti = k & 127;
                                let tone = lane_buf[lane_slot(c, tone_f, ti)];
                                let bp = lane_buf[lane_slot(c, bp_f, ti)];
                                let v_raw =
                                    step_paralru_duhem_scalar(tone, bp, &mut v, &lp, &lru_c);
                                lane_buf[lane_slot(c, bp_f, ti)] = v_raw;
                            }
                            aux_scratch[v_idx] = flush_zero(v);
                        }
                    }
                    wg_barrier();
                } else {
                    for c in 0..4 {
                        let lp = params.lane(c);
                        if !is_duhem_decoupled(&lp) {
                            run_duhem_damped_deer_subtile_lane(
                                lid,
                                sub_valid_len,
                                c,
                                &lp,
                                scan_scratch,
                                lane_buf,
                                aux_scratch,
                            );
                        }
                    }
                }

                // Stage 3: Stereo Link -> g_pre -> AlgADAA1(4096) + Wavefolder -> SE-Bus -> GLU -> out_gain.
                let mut my_veff_l = GpuVec4::zero();
                let mut my_veff_r = GpuVec4::zero();
                let mut my_tl = GpuVec4::zero();
                let mut my_tr = GpuVec4::zero();
                let mut my_g_pre_l = GpuVec4::zero();
                let mut my_g_pre_r = GpuVec4::zero();
                let mut my_r_l = GpuVec4::zero();
                let mut my_r_r = GpuVec4::zero();
                if lid < SUBTILE_FRAMES && active128 {
                    let mut tl_arr = [0.0_f32; 4];
                    let mut tr_arr = [0.0_f32; 4];
                    let mut vr_l = [0.0_f32; 4];
                    let mut vr_r = [0.0_f32; 4];
                    for c in 0..4 {
                        tl_arr[c] = lane_buf[lane_slot(c, 0, t128)];
                        vr_l[c] = lane_buf[lane_slot(c, 1, t128)];
                        tr_arr[c] = lane_buf[lane_slot(c, 2, t128)];
                        vr_r[c] = lane_buf[lane_slot(c, 3, t128)];
                    }
                    my_tl = GpuVec4::from_array(tl_arr);
                    my_tr = GpuVec4::from_array(tr_arr);
                    let (eff_l_v, eff_r_v) = apply_stereo_link_vec4_pair(
                        GpuVec4::from_array(vr_l),
                        GpuVec4::from_array(vr_r),
                        stereo_link_g,
                    );
                    my_veff_l = eff_l_v;
                    my_veff_r = eff_r_v;
                    my_g_pre_l = eval_g_pre_vec4(eff_l_v, g0_g, g_inf_g, d_pre_g);
                    my_g_pre_r = eval_g_pre_vec4(eff_r_v, g0_g, g_inf_g, d_pre_g);
                    my_r_l = scrub_clamped_alg_endpoint_vec4(
                        my_g_pre_l
                            .mul(adaa_drive_g)
                            .mul(my_tl)
                            .fma(lru_bias_g, eff_l_v),
                    );
                    my_r_r = scrub_clamped_alg_endpoint_vec4(
                        my_g_pre_r
                            .mul(adaa_drive_g)
                            .mul(my_tr)
                            .fma(lru_bias_g, eff_r_v),
                    );
                    let rl_arr = my_r_l.to_array();
                    let rr_arr = my_r_r.to_array();
                    for c in 0..4 {
                        scan_scratch[c * SUBTILE_FRAMES + t128] = rl_arr[c];
                        scan_scratch[(4 + c) * SUBTILE_FRAMES + t128] = rr_arr[c];
                    }
                }
                wg_barrier();

                if lid < SUBTILE_FRAMES && active128 {
                    let mut rp_l = [0.0_f32; 4];
                    let mut rp_r = [0.0_f32; 4];
                    if t128 == 0 {
                        for c in 0..4 {
                            rp_l[c] = aux_scratch[24 + c];
                            rp_r[c] = aux_scratch[28 + c];
                        }
                    } else {
                        let prev = t128 - 1;
                        for c in 0..4 {
                            rp_l[c] = scan_scratch[c * SUBTILE_FRAMES + prev];
                            rp_r[c] = scan_scratch[(4 + c) * SUBTILE_FRAMES + prev];
                        }
                    }
                    let (a_odd_wf_l, a_even_l, r_safe_l) = eval_alg_adaa1_with_wavefolder_vec4(
                        GpuVec4::from_array(rp_l),
                        my_r_l,
                        wf_mix_g,
                    );
                    let (a_odd_wf_r, a_even_r, r_safe_r) = eval_alg_adaa1_with_wavefolder_vec4(
                        GpuVec4::from_array(rp_r),
                        my_r_r,
                        wf_mix_g,
                    );
                    let g_se_l = eval_se_bus_gain_vec4(my_veff_l, se_w_sq_g, se_d_bus_g);
                    let g_se_r = eval_se_bus_gain_vec4(my_veff_r, se_w_sq_g, se_d_bus_g);
                    let o_pre_l = one_minus_lin_g
                        .mul(a_odd_wf_l)
                        .fma(lin_mix_g, my_g_pre_l.mul(my_tl));
                    let o_pre_r = one_minus_lin_g
                        .mul(a_odd_wf_r)
                        .fma(lin_mix_g, my_g_pre_r.mul(my_tr));
                    let o_tilde_l = out_gain_g.mul(g_se_l).mul(o_pre_l).flush_recursive_state();
                    let o_tilde_r = out_gain_g.mul(g_se_r).mul(o_pre_r).flush_recursive_state();
                    let e_tilde = eval_bounded_glu_stereo8(
                        InterleavedStereo8::from_lr(a_odd_wf_l, a_odd_wf_r),
                        InterleavedStereo8::from_lr(a_even_l, a_even_r),
                        InterleavedStereo8::from_lr(g_se_l, g_se_r),
                        lin_mix_g,
                        glu_mix_g,
                        out_gain_g,
                    );
                    let (e_tilde_l, e_tilde_r) = e_tilde.into_lr();

                    let feat_base = (q_tier * num_frames + frame) * FEATURE_VEC4_STRIDE;
                    store_vec4(
                        feature_outputs,
                        feat_base,
                        canonicalize_vec4_zero(gpu_to_vec4(my_veff_l)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 1,
                        canonicalize_vec4_zero(gpu_to_vec4(my_veff_r)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 2,
                        canonicalize_vec4_zero(gpu_to_vec4(o_tilde_l)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 3,
                        canonicalize_vec4_zero(gpu_to_vec4(o_tilde_r)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 4,
                        canonicalize_vec4_zero(gpu_to_vec4(e_tilde_l)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 5,
                        canonicalize_vec4_zero(gpu_to_vec4(e_tilde_r)),
                    );
                    if t128 + 1 == sub_valid_len {
                        let safe_l = r_safe_l.to_array();
                        let safe_r = r_safe_r.to_array();
                        for c in 0..4 {
                            aux_scratch[24 + c] = safe_l[c];
                            aux_scratch[28 + c] = safe_r[c];
                        }
                    }
                }
                wg_barrier();
            }
        }
    }
}
