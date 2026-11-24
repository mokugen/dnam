#![forbid(unsafe_code)]

//! Persistent `256`-frame forward scan kernel (`fused_core_scan_256_impl`) executing
//! `TPT-SVF -> Selective-Duhem ParaLRU + Stereo Link -> Sag -> PolyADAA1 + C^2 Wavefolder -> GLU`
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
    MIN_SAG_GAIN, QuadParamsUnpacked, clamp_nonneg_safe, div_ceil_divfree, eval_tone_scalar,
    eval_tpt_svf_scalar, flush_zero, get_lane, gpu_to_vec4, load_vec4, scrub_endpoint,
    step_duhem_stereo_scalar, step_tpt_svf_phs_scalar, store_vec4, sub_sat_usize,
    tpt_exclusive_scan_128_pair, vec4_to_gpu, wg_barrier,
};
use dnam_kernels::{
    GpuVec4, InterleavedStereo8, KernelVec4, StereoVec8, TptScanElem, eval_c2_wavefolder_adaa_vec4,
    eval_poly_adaa1_odd_even_vec4,
};
use fixed_point::{lane_slot, run_duhem_damped_deer_subtile_lane, run_phs_passive_picard_subtile};
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
    for c in 0..4 {
        aux_scratch[c] = s1_l[c];
        aux_scratch[4 + c] = s2_l[c];
        aux_scratch[8 + c] = s1_r[c];
        aux_scratch[12 + c] = s2_r[c];
        aux_scratch[16 + c] = v_l[c];
        aux_scratch[20 + c] = v_r[c];
        aux_scratch[24 + c] = ap_l[c];
        aux_scratch[28 + c] = ap_r[c];
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
    let s1_l = Vec4::new(
        aux_scratch[0],
        aux_scratch[1],
        aux_scratch[2],
        aux_scratch[3],
    );
    let s2_l = Vec4::new(
        aux_scratch[4],
        aux_scratch[5],
        aux_scratch[6],
        aux_scratch[7],
    );
    let s1_r = Vec4::new(
        aux_scratch[8],
        aux_scratch[9],
        aux_scratch[10],
        aux_scratch[11],
    );
    let s2_r = Vec4::new(
        aux_scratch[12],
        aux_scratch[13],
        aux_scratch[14],
        aux_scratch[15],
    );
    let v_l = Vec4::new(
        aux_scratch[16],
        aux_scratch[17],
        aux_scratch[18],
        aux_scratch[19],
    );
    let v_r = Vec4::new(
        aux_scratch[20],
        aux_scratch[21],
        aux_scratch[22],
        aux_scratch[23],
    );
    let ap_l = Vec4::new(
        aux_scratch[24],
        aux_scratch[25],
        aux_scratch[26],
        aux_scratch[27],
    );
    let ap_r = Vec4::new(
        aux_scratch[28],
        aux_scratch[29],
        aux_scratch[30],
        aux_scratch[31],
    );
    store_vec4(tile_checkpoints, ckpt_base, canonicalize_vec4_zero(s1_l));
    store_vec4(
        tile_checkpoints,
        ckpt_base + 1,
        canonicalize_vec4_zero(s1_r),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 2,
        canonicalize_vec4_zero(s2_l),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 3,
        canonicalize_vec4_zero(s2_r),
    );
    store_vec4(tile_checkpoints, ckpt_base + 4, canonicalize_vec4_zero(v_l));
    store_vec4(tile_checkpoints, ckpt_base + 5, canonicalize_vec4_zero(v_r));
    store_vec4(
        tile_checkpoints,
        ckpt_base + 6,
        canonicalize_vec4_zero(ap_l),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 7,
        canonicalize_vec4_zero(ap_r),
    );
}

#[inline(always)]
fn eval_lane_block_outputs(
    r_prev_l: GpuVec4,
    r_curr_l: GpuVec4,
    r_prev_r: GpuVec4,
    r_curr_r: GpuVec4,
    wf_mix_g: GpuVec4,
    glu_mix_g: GpuVec4,
) -> (GpuVec4, GpuVec4, GpuVec4, GpuVec4) {
    let (mut odd_l, even_l, _) = eval_poly_adaa1_odd_even_vec4(r_prev_l, r_curr_l);
    let (mut odd_r, even_r, _) = eval_poly_adaa1_odd_even_vec4(r_prev_r, r_curr_r);

    if !wf_mix_g.all_le(GpuVec4::zero()) {
        let (wf_odd_l, _) = eval_c2_wavefolder_adaa_vec4(r_prev_l, r_curr_l);
        let (wf_odd_r, _) = eval_c2_wavefolder_adaa_vec4(r_prev_r, r_curr_r);
        let one_minus_wf = GpuVec4::splat(1.0).sub(wf_mix_g);
        odd_l = odd_l.mul(one_minus_wf).fma(wf_mix_g, wf_odd_l);
        odd_r = odd_r.mul(one_minus_wf).fma(wf_mix_g, wf_odd_r);
    }

    let odd_st = InterleavedStereo8::from_lr(odd_l, odd_r);
    let even_st = InterleavedStereo8::from_lr(even_l, even_r);
    let glu_term = odd_st.mul(odd_st.rotate_left_1());
    let even_glu = even_st.fma(InterleavedStereo8::broadcast_half(glu_mix_g), glu_term);
    let (even_l_out, even_r_out) = even_glu.into_lr();
    (odd_l, odd_r, even_l_out, even_r_out)
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
    let wf_mix_g = GpuVec4::from_array([
        clamp_nonneg_safe(params.wavefolder_mix.x, 1.0),
        clamp_nonneg_safe(params.wavefolder_mix.y, 1.0),
        clamp_nonneg_safe(params.wavefolder_mix.z, 1.0),
        clamp_nonneg_safe(params.wavefolder_mix.w, 1.0),
    ]);
    let glu_mix_g = vec4_to_gpu(params.glu_mix);

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
            for k in 0..32 {
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

                // Stage 1: TPT-SVF across 4 lanes (`tone`, `bp` written to `lane_buf[0..2048]`).
                if is_phs {
                    if !use_fixed_point {
                        if lid < SUBTILE_FRAMES {
                            let u_l_vec = if active128 { in_l } else { Vec4::ZERO };
                            let u_r_vec = if active128 { in_r } else { Vec4::ZERO };
                            for c in 0..4 {
                                scan_scratch[(c * 2) * SUBTILE_FRAMES + t128] =
                                    flush_zero(get_lane(u_l_vec, c));
                                scan_scratch[(c * 2 + 1) * SUBTILE_FRAMES + t128] =
                                    flush_zero(get_lane(u_r_vec, c));
                            }
                        }
                        wg_barrier();
                        if lid < 8 {
                            let c = lid >> 1;
                            let is_r = (lid & 1) != 0;
                            let lp = params.lane(c);
                            let s1_idx = if is_r { 8 + c } else { c };
                            let s2_idx = if is_r { 12 + c } else { 4 + c };
                            let mut s1 = aux_scratch[s1_idx];
                            let mut s2 = aux_scratch[s2_idx];
                            let u_base = (c * 2 + (lid & 1)) * SUBTILE_FRAMES;
                            let tone_f = if is_r { 2 } else { 0 };
                            for k in 0..sub_valid_len {
                                let ti = k & 127;
                                let u = scan_scratch[u_base + ti];
                                let (lp_s, bp_s, hp_s, ap_s) =
                                    step_tpt_svf_phs_scalar(u, &mut s1, &mut s2, &lp);
                                lane_buf[lane_slot(c, tone_f, ti)] =
                                    eval_tone_scalar(lp_s, bp_s, hp_s, ap_s, u, &lp);
                                lane_buf[lane_slot(c, tone_f + 1, ti)] = bp_s;
                            }
                            aux_scratch[s1_idx] = flush_zero(s1);
                            aux_scratch[s2_idx] = flush_zero(s2);
                        }
                        wg_barrier();
                    } else {
                        for c in 0..4 {
                            let lp = params.lane(c);
                            let u_my = if active128 {
                                flush_zero(get_lane(if is_r128 { in_r } else { in_l }, c))
                            } else {
                                0.0
                            };
                            run_phs_passive_picard_subtile(
                                lid,
                                sub_valid_len,
                                c,
                                u_my,
                                &lp,
                                scan_scratch,
                                lane_buf,
                                aux_scratch,
                            );
                        }
                    }
                } else {
                    for c in 0..4 {
                        let lp = params.lane(c);
                        let svf_c = lp.to_svf_coeffs();
                        let u_my = if active128 {
                            flush_zero(get_lane(if is_r128 { in_r } else { in_l }, c))
                        } else {
                            0.0
                        };
                        let elem = if active128 {
                            TptScanElem::from_step(lp.a1, lp.a2, lp.a3, u_my)
                        } else {
                            TptScanElem::identity()
                        };
                        let s1_idx = if is_r128 { 8 + c } else { c };
                        let s2_idx = if is_r128 { 12 + c } else { 4 + c };
                        let mut s1_prev = 0.0_f32;
                        let mut s2_prev = 0.0_f32;
                        let s_end = tpt_exclusive_scan_128_pair(
                            scan_scratch,
                            lid,
                            sub_valid_len,
                            elem,
                            [aux_scratch[s1_idx], aux_scratch[s2_idx]],
                            &mut s1_prev,
                            &mut s2_prev,
                        );
                        if active128 {
                            let (lp_s, bp_s, hp_s, ap_s) =
                                eval_tpt_svf_scalar(u_my, s1_prev, s2_prev, &svf_c);
                            let tone_s = eval_tone_scalar(lp_s, bp_s, hp_s, ap_s, u_my, &lp);
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

                // Stage 2: Selective-Duhem ParaLRU + stereo link (`4` concurrent lanes or DEER),
                // saving `(v_eff_L, v_eff_R)` in dead `bp` slots `lane_slot(c, 1/3, t)` (`tone` kept).
                if !use_fixed_point {
                    if lid < 4 {
                        let c = lid;
                        let lp = params.lane(c);
                        let lru_c = lp.to_lru_coeffs();
                        let mut vl = aux_scratch[16 + c];
                        let mut vr = aux_scratch[20 + c];
                        for k in 0..sub_valid_len {
                            let ti = k & 127;
                            let tl = lane_buf[lane_slot(c, 0, ti)];
                            let bl = lane_buf[lane_slot(c, 1, ti)];
                            let tr = lane_buf[lane_slot(c, 2, ti)];
                            let br = lane_buf[lane_slot(c, 3, ti)];
                            let (_rl, _rr, eff_l, eff_r) = step_duhem_stereo_scalar(
                                tl, tr, bl, br, &mut vl, &mut vr, &lp, &lru_c,
                            );
                            lane_buf[lane_slot(c, 1, ti)] = eff_l;
                            lane_buf[lane_slot(c, 3, ti)] = eff_r;
                        }
                        aux_scratch[16 + c] = flush_zero(vl);
                        aux_scratch[20 + c] = flush_zero(vr);
                    }
                    wg_barrier();
                } else {
                    for c in 0..4 {
                        let lp = params.lane(c);
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

                // Stage 3: Sag + r_curr (`scan_scratch[0..1024]`) -> PolyADAA1 + Wavefolder + GLU.
                let mut my_veff_l = [0.0_f32; 4];
                let mut my_veff_r = [0.0_f32; 4];
                let mut my_r_l = [0.0_f32; 4];
                let mut my_r_r = [0.0_f32; 4];
                if lid < SUBTILE_FRAMES && active128 {
                    for c in 0..4 {
                        let lp = params.lane(c);
                        let tl = lane_buf[lane_slot(c, 0, t128)];
                        let vel = lane_buf[lane_slot(c, 1, t128)];
                        let tr = lane_buf[lane_slot(c, 2, t128)];
                        let ver = lane_buf[lane_slot(c, 3, t128)];
                        my_veff_l[c] = vel;
                        my_veff_r[c] = ver;
                        let sag_l = (1.0 - lp.sag_depth * vel).max(MIN_SAG_GAIN);
                        let sag_r = (1.0 - lp.sag_depth * ver).max(MIN_SAG_GAIN);
                        let rl = flush_zero((tl * lp.adaa_drive) * sag_l + lp.lru_bias_drive * vel);
                        let rr = flush_zero((tr * lp.adaa_drive) * sag_r + lp.lru_bias_drive * ver);
                        my_r_l[c] = rl;
                        my_r_r[c] = rr;
                        scan_scratch[c * SUBTILE_FRAMES + t128] = rl;
                        scan_scratch[(4 + c) * SUBTILE_FRAMES + t128] = rr;
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
                    let (odd_l, odd_r, even_l_out, even_r_out) = eval_lane_block_outputs(
                        GpuVec4::from_array(rp_l),
                        GpuVec4::from_array(my_r_l),
                        GpuVec4::from_array(rp_r),
                        GpuVec4::from_array(my_r_r),
                        wf_mix_g,
                        glu_mix_g,
                    );
                    let feat_base = (q_tier * num_frames + frame) * FEATURE_VEC4_STRIDE;
                    store_vec4(
                        feature_outputs,
                        feat_base,
                        canonicalize_vec4_zero(Vec4::from(my_veff_l)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 1,
                        canonicalize_vec4_zero(Vec4::from(my_veff_r)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 2,
                        canonicalize_vec4_zero(gpu_to_vec4(odd_l)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 3,
                        canonicalize_vec4_zero(gpu_to_vec4(odd_r)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 4,
                        canonicalize_vec4_zero(gpu_to_vec4(even_l_out)),
                    );
                    store_vec4(
                        feature_outputs,
                        feat_base + 5,
                        canonicalize_vec4_zero(gpu_to_vec4(even_r_out)),
                    );
                    if t128 + 1 == sub_valid_len {
                        for c in 0..4 {
                            aux_scratch[24 + c] = scrub_endpoint(my_r_l[c]);
                            aux_scratch[28 + c] = scrub_endpoint(my_r_r[c]);
                        }
                    }
                }
                wg_barrier();
            }
        }
    }
}
