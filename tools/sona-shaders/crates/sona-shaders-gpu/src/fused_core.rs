#![forbid(unsafe_code)]

//! Persistent `256`-frame forward scan kernel (`fused_core_scan_256_impl`) executing
//! `TPT-SVF -> Selective-Duhem ParaLRU + Stereo Link -> Sag -> PolyADAA1 + C^2 Wavefolder -> GLU`
//! strictly inside `15,360 B` (`15.0 KiB`) workgroup SRAM (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{
    FORWARD_AUX_SCRATCH_FLOATS, FORWARD_LANE_BUF_VEC4S, FORWARD_SCAN_SCRATCH_FLOATS,
    FusedScanUniforms, TILE_FRAMES,
};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, MIN_SAG_GAIN, QUAD_PARAMS_VEC4_STRIDE,
    QuadLaneParams, QuadParamsUnpacked, TILE_CKPT_VEC4_STRIDE, clamp_nonneg_safe, div_ceil_divfree,
    eval_tone_scalar, eval_tpt_svf_scalar, flush_zero, get_lane, gpu_to_vec4, load_vec4,
    scrub_endpoint, set_lane, step_duhem_stereo_scalar, store_vec4, sub_sat_usize,
    tpt_exclusive_scan_256, vec4_to_gpu, wg_barrier,
};
use dnam_kernels::{
    GpuVec4, InterleavedStereo8, KernelVec4, SonaBlockStateStereo8, StereoVec8, TptScanElem,
    eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4,
};
use spirv_std::glam::{UVec3, Vec4};

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
fn read_tile_checkpoint(
    tile_checkpoints: &[Vec4],
    q_global: usize,
    ckpt_tiles: usize,
    tile: usize,
) -> SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>> {
    let ckpt_base = (q_global * ckpt_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let s1_l = vec4_to_gpu(canonicalize_vec4_zero(load_vec4(
        tile_checkpoints,
        ckpt_base,
    )));
    let s1_r = vec4_to_gpu(canonicalize_vec4_zero(load_vec4(
        tile_checkpoints,
        ckpt_base + 1,
    )));
    let s2_l = vec4_to_gpu(canonicalize_vec4_zero(load_vec4(
        tile_checkpoints,
        ckpt_base + 2,
    )));
    let s2_r = vec4_to_gpu(canonicalize_vec4_zero(load_vec4(
        tile_checkpoints,
        ckpt_base + 3,
    )));
    let v_l = vec4_to_gpu(canonicalize_vec4_zero(load_vec4(
        tile_checkpoints,
        ckpt_base + 4,
    )));
    let v_r = vec4_to_gpu(canonicalize_vec4_zero(load_vec4(
        tile_checkpoints,
        ckpt_base + 5,
    )));
    let ap_l = scrub_vec4_endpoints(load_vec4(tile_checkpoints, ckpt_base + 6));
    let ap_r = scrub_vec4_endpoints(load_vec4(tile_checkpoints, ckpt_base + 7));
    SonaBlockStateStereo8 {
        svf_s1: InterleavedStereo8::from_lr(s1_l, s1_r),
        svf_s2: InterleavedStereo8::from_lr(s2_l, s2_r),
        lru_v: InterleavedStereo8::from_lr(v_l, v_r),
        adaa_prev: InterleavedStereo8::from_lr(ap_l, ap_r),
    }
}

#[inline(always)]
fn write_tile_checkpoint(
    tile_checkpoints: &mut [Vec4],
    q_global: usize,
    ckpt_tiles: usize,
    tile: usize,
    state: &SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>>,
) {
    let ckpt_base = (q_global * ckpt_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let (s1_l, s1_r) = state.svf_s1.into_lr();
    let (s2_l, s2_r) = state.svf_s2.into_lr();
    let (v_l, v_r) = state.lru_v.into_lr();
    let (ap_l, ap_r) = state.adaa_prev.into_lr();

    store_vec4(
        tile_checkpoints,
        ckpt_base,
        canonicalize_vec4_zero(gpu_to_vec4(s1_l)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 1,
        canonicalize_vec4_zero(gpu_to_vec4(s1_r)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 2,
        canonicalize_vec4_zero(gpu_to_vec4(s2_l)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 3,
        canonicalize_vec4_zero(gpu_to_vec4(s2_r)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 4,
        canonicalize_vec4_zero(gpu_to_vec4(v_l)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 5,
        canonicalize_vec4_zero(gpu_to_vec4(v_r)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 6,
        canonicalize_vec4_zero(gpu_to_vec4(ap_l)),
    );
    store_vec4(
        tile_checkpoints,
        ckpt_base + 7,
        canonicalize_vec4_zero(gpu_to_vec4(ap_r)),
    );
}

#[inline]
fn step_tile_duhem_on_thread0(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    valid_len: usize,
    v_init_l: f32,
    v_init_r: f32,
    lp: &QuadLaneParams,
) -> (f32, f32) {
    let lru_coeffs = lp.to_lru_coeffs();
    let mut vl = v_init_l;
    let mut vr = v_init_r;
    for t in 0..valid_len {
        let i = t & 255;
        let tone_l = scan_scratch[i];
        let bp_l = scan_scratch[TILE_FRAMES + i];
        let tone_r = scan_scratch[2 * TILE_FRAMES + i];
        let bp_r = scan_scratch[3 * TILE_FRAMES + i];
        let (_rl, _rr, eff_l, eff_r) = step_duhem_stereo_scalar(
            tone_l,
            tone_r,
            bp_l,
            bp_r,
            &mut vl,
            &mut vr,
            lp,
            &lru_coeffs,
        );
        scan_scratch[4 * TILE_FRAMES + i] = eff_l;
        scan_scratch[5 * TILE_FRAMES + i] = eff_r;
    }
    (vl, vr)
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

/// Executes the persistent forward `CoreQuad` tile scan (`fused_core_scan_256`) over
/// `tile_start .. num_tiles` (or the per-window slice when `num_wg.y > 1`), preserving exact
/// `0-ULP` parity with `dnam_kernels::step_core_quad_stereo8` inside `15,360 B` SRAM.
///
/// Both `portable` and `subgroups` builds share this Blelloch `tpt_exclusive_scan_256` kernel
/// because the forward scan performs no `subgroup_f_add` workgroup scalar reductions.
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
    lane_buf: &mut [Vec4; FORWARD_LANE_BUF_VEC4S],
    aux_scratch: &mut [f32; FORWARD_AUX_SCRATCH_FLOATS],
) {
    let _ = gid;
    let lid = (lid_u32 as usize) & 255;

    let quads_in_block = (num_wg.x as usize).max(1);
    let q_local = wg_id.x as usize;
    let win_idx = wg_id.y as usize;
    let num_windows = (num_wg.y as usize).max(1);
    let stem = wg_id.z as usize;
    let q_global = stem * quads_in_block + q_local;

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

    let q_base = q_global * QUAD_PARAMS_VEC4_STRIDE;
    let params = QuadParamsUnpacked::load(quad_params, q_base);
    let wf_mix_g = GpuVec4::from_array([
        clamp_nonneg_safe(params.wavefolder_mix.x, 1.0),
        clamp_nonneg_safe(params.wavefolder_mix.y, 1.0),
        clamp_nonneg_safe(params.wavefolder_mix.z, 1.0),
        clamp_nonneg_safe(params.wavefolder_mix.w, 1.0),
    ]);
    let glu_mix_g = vec4_to_gpu(params.glu_mix);

    let resuming_from_ckpt = num_windows <= 1 && tile_start > 0 && tile_start < ckpt_tiles;
    let mut state = if resuming_from_ckpt {
        read_tile_checkpoint(tile_checkpoints, q_global, ckpt_tiles, tile_start)
    } else {
        SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero()
    };

    for tile in t_begin..t_end {
        let tile_frame_base = tile * TILE_FRAMES;
        let valid_len = sub_sat_usize(num_frames, tile_frame_base).min(TILE_FRAMES);
        let frame = tile_frame_base + lid;
        let is_active = lid < valid_len;

        if lid == 0 && !(resuming_from_ckpt && tile == tile_start) {
            write_tile_checkpoint(tile_checkpoints, q_global, ckpt_tiles, tile, &state);
        }

        lane_buf[lid] = Vec4::ZERO;
        lane_buf[TILE_FRAMES + lid] = Vec4::ZERO;

        let in_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
        let (in_l, in_r) = if is_active {
            (
                load_vec4(core_inputs, in_base),
                load_vec4(core_inputs, in_base + 1),
            )
        } else {
            (Vec4::ZERO, Vec4::ZERO)
        };

        let mut my_veff_l = [0.0_f32; 4];
        let mut my_veff_r = [0.0_f32; 4];
        let (s1_cur_l, s1_cur_r) = state.svf_s1.into_lr();
        let (s2_cur_l, s2_cur_r) = state.svf_s2.into_lr();
        let mut s1_end_l_arr = [0.0_f32; 4];
        let mut s1_end_r_arr = [0.0_f32; 4];
        let mut s2_end_l_arr = [0.0_f32; 4];
        let mut s2_end_r_arr = [0.0_f32; 4];

        for c in 0..4 {
            let lp = params.lane(c);
            let svf_c = lp.to_svf_coeffs();
            let u_l = if is_active {
                flush_zero(get_lane(in_l, c))
            } else {
                0.0
            };
            let u_r = if is_active {
                flush_zero(get_lane(in_r, c))
            } else {
                0.0
            };

            let elem_l = if is_active {
                TptScanElem::from_step(lp.a1, lp.a2, lp.a3, u_l)
            } else {
                TptScanElem::identity()
            };
            let mut s1_prev_l = 0.0_f32;
            let mut s2_prev_l = 0.0_f32;
            let s_end_l = tpt_exclusive_scan_256(
                scan_scratch,
                lid,
                valid_len,
                elem_l,
                [s1_cur_l.to_array()[c & 3], s2_cur_l.to_array()[c & 3]],
                &mut s1_prev_l,
                &mut s2_prev_l,
            );
            s1_end_l_arr[c & 3] = s_end_l[0];
            s2_end_l_arr[c & 3] = s_end_l[1];
            let (lp_l, bp_l, hp_l, ap_l) = eval_tpt_svf_scalar(u_l, s1_prev_l, s2_prev_l, &svf_c);
            let tone_l = eval_tone_scalar(lp_l, bp_l, hp_l, ap_l, u_l, &lp);
            aux_scratch[lid] = tone_l;
            set_lane(&mut lane_buf[lid], c, bp_l);

            let elem_r = if is_active {
                TptScanElem::from_step(lp.a1, lp.a2, lp.a3, u_r)
            } else {
                TptScanElem::identity()
            };
            let mut s1_prev_r = 0.0_f32;
            let mut s2_prev_r = 0.0_f32;
            let s_end_r = tpt_exclusive_scan_256(
                scan_scratch,
                lid,
                valid_len,
                elem_r,
                [s1_cur_r.to_array()[c & 3], s2_cur_r.to_array()[c & 3]],
                &mut s1_prev_r,
                &mut s2_prev_r,
            );
            s1_end_r_arr[c & 3] = s_end_r[0];
            s2_end_r_arr[c & 3] = s_end_r[1];
            let (lp_r, bp_r, hp_r, ap_r) = eval_tpt_svf_scalar(u_r, s1_prev_r, s2_prev_r, &svf_c);
            let tone_r = eval_tone_scalar(lp_r, bp_r, hp_r, ap_r, u_r, &lp);

            let tone_l_saved = aux_scratch[lid];
            let bp_l_saved = get_lane(lane_buf[lid], c);
            scan_scratch[lid] = tone_l_saved;
            scan_scratch[TILE_FRAMES + lid] = bp_l_saved;
            scan_scratch[2 * TILE_FRAMES + lid] = tone_r;
            scan_scratch[3 * TILE_FRAMES + lid] = bp_r;
            scan_scratch[4 * TILE_FRAMES + lid] = 0.0;
            scan_scratch[5 * TILE_FRAMES + lid] = 0.0;
            wg_barrier();

            if lid == 0 {
                let (vl_cur, vr_cur) = state.lru_v.into_lr();
                let mut vl_arr = vl_cur.to_array();
                let mut vr_arr = vr_cur.to_array();
                let (vl_end, vr_end) = step_tile_duhem_on_thread0(
                    scan_scratch,
                    valid_len,
                    vl_arr[c & 3],
                    vr_arr[c & 3],
                    &lp,
                );
                vl_arr[c & 3] = vl_end;
                vr_arr[c & 3] = vr_end;
                state.lru_v = InterleavedStereo8::from_lr(
                    GpuVec4::from_array(vl_arr),
                    GpuVec4::from_array(vr_arr),
                );
            }
            wg_barrier();

            let ve_l = if is_active {
                scan_scratch[4 * TILE_FRAMES + lid]
            } else {
                0.0
            };
            let ve_r = if is_active {
                scan_scratch[5 * TILE_FRAMES + lid]
            } else {
                0.0
            };
            my_veff_l[c & 3] = ve_l;
            my_veff_r[c & 3] = ve_r;

            let sag_l = (1.0 - lp.sag_depth * ve_l).max(MIN_SAG_GAIN);
            let sag_r = (1.0 - lp.sag_depth * ve_r).max(MIN_SAG_GAIN);
            let r_curr_l = if is_active {
                flush_zero((tone_l_saved * lp.adaa_drive) * sag_l + lp.lru_bias_drive * ve_l)
            } else {
                0.0
            };
            let r_curr_r = if is_active {
                flush_zero((tone_r * lp.adaa_drive) * sag_r + lp.lru_bias_drive * ve_r)
            } else {
                0.0
            };
            set_lane(&mut lane_buf[lid], c, r_curr_l);
            set_lane(&mut lane_buf[TILE_FRAMES + lid], c, r_curr_r);
            wg_barrier();
        }

        state.svf_s1 = InterleavedStereo8::from_lr(
            GpuVec4::from_array(s1_end_l_arr).flush_recursive_state(),
            GpuVec4::from_array(s1_end_r_arr).flush_recursive_state(),
        );
        state.svf_s2 = InterleavedStereo8::from_lr(
            GpuVec4::from_array(s2_end_l_arr).flush_recursive_state(),
            GpuVec4::from_array(s2_end_r_arr).flush_recursive_state(),
        );

        let r_curr_l_g = vec4_to_gpu(lane_buf[lid]);
        let r_curr_r_g = vec4_to_gpu(lane_buf[TILE_FRAMES + lid]);
        let (adaa_prev_l, adaa_prev_r) = state.adaa_prev.into_lr();
        let (r_prev_l_g, r_prev_r_g) = if lid == 0 {
            (adaa_prev_l, adaa_prev_r)
        } else {
            let prev_idx = (lid - 1) & 255;
            (
                vec4_to_gpu(lane_buf[prev_idx]),
                vec4_to_gpu(lane_buf[TILE_FRAMES + prev_idx]),
            )
        };

        let (odd_l, odd_r, even_l_out, even_r_out) = eval_lane_block_outputs(
            r_prev_l_g, r_curr_l_g, r_prev_r_g, r_curr_r_g, wf_mix_g, glu_mix_g,
        );

        if is_active {
            let feat_base = (q_global * num_frames + frame) * FEATURE_VEC4_STRIDE;
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
        }

        if valid_len > 0 {
            let last_idx = (valid_len - 1) & 255;
            state.adaa_prev = InterleavedStereo8::from_lr(
                scrub_vec4_endpoints(lane_buf[last_idx]),
                scrub_vec4_endpoints(lane_buf[TILE_FRAMES + last_idx]),
            );
        }
        wg_barrier();
    }
}
