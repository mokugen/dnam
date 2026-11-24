#![forbid(unsafe_code)]

//! Conflict-free wave-level inter-block DAG mixing forward and adjoint pullback kernel
//! (`tile_mix_fwd_adj`) executing all `35` lower-triangular `SONA_INTERBLOCK_TILES` strictly in
//! `2,048 B` SRAM with zero `f32` atomics (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{
    BLOCK_WAVE_ALL, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, MAX_WORKGROUP_SRAM_BYTES,
    TILE_DESC_VEC4_STRIDE, TILE_FRAMES, TILE_MIX_DIR_FORWARD, TILE_MIX_DIR_PARTIAL_WEIGHTS,
    TILE_MIX_DIR_PULLBACK, TILE_MIX_DIR_REDUCE_WEIGHTS, TILE_MIX_REDUCE_A_FLOATS,
    TILE_MIX_REDUCE_B_FLOATS, TILE_WEIGHT_GRAD_FLOATS, TileMixUniforms, resolve_quad_idx,
};
use crate::core_math::{load_vec4, store_vec4};
use crate::reduce::{WG_THREADS, clamp_idx, flush_zero, reduce_sum_f32_pair_256};
use core::mem::size_of;
use spirv_std::glam::{UVec3, Vec4};

/// Total `tile_mix_fwd_adj` workgroup SRAM footprint (`2,048 B <= 15,872 B`).
pub const TILE_MIX_SRAM_BYTES: usize =
    (TILE_MIX_REDUCE_A_FLOATS + TILE_MIX_REDUCE_B_FLOATS) * size_of::<f32>();

const _: () = assert!(TILE_MIX_SRAM_BYTES == 2_048);
const _: () = assert!(TILE_MIX_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);
const _: () = assert!(TILE_DESC_VEC4_STRIDE == 13);
const _: () = assert!(TILE_WEIGHT_GRAD_FLOATS == 48);

#[inline(always)]
fn accumulate_mat4x4_col_major(acc: Vec4, mat_cols: [Vec4; 4], src: Vec4) -> Vec4 {
    acc + mat_cols[0] * src.x + mat_cols[1] * src.y + mat_cols[2] * src.z + mat_cols[3] * src.w
}

#[inline(always)]
fn transpose_mat4x4_accum(cur: Vec4, mat_cols: [Vec4; 4], downstream: Vec4) -> Vec4 {
    Vec4::new(
        cur.x + mat_cols[0].dot(downstream),
        cur.y + mat_cols[1].dot(downstream),
        cur.z + mat_cols[2].dot(downstream),
        cur.w + mat_cols[3].dot(downstream),
    )
}

#[inline(always)]
fn canonicalize_vec4(v: Vec4) -> Vec4 {
    Vec4::new(
        flush_zero(v.x),
        flush_zero(v.y),
        flush_zero(v.z),
        flush_zero(v.w),
    )
}

#[inline]
fn run_forward_mode(
    wg_id: UVec3,
    lid: usize,
    uniforms: &TileMixUniforms,
    block_inputs: &mut [Vec4],
    tile_descriptors: &[Vec4],
    feature_outputs: &[Vec4],
) {
    let num_frames = uniforms.num_frames as usize;
    let num_quads = (uniforms.num_quads as usize).max(1);
    let num_active_tiles = uniforms.num_active_tiles as usize;
    let dst_quad = resolve_quad_idx(uniforms.block_wave, wg_id.x as usize, num_quads);
    if dst_quad >= num_quads {
        return;
    }

    let frame = (wg_id.y as usize) * TILE_FRAMES + lid;
    if frame >= num_frames {
        return;
    }
    let stem = wg_id.z as usize;
    let dst_in_base = ((stem * num_quads + dst_quad) * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
    let mut acc_l = load_vec4(block_inputs, dst_in_base);
    let mut acc_r = load_vec4(block_inputs, dst_in_base + 1);

    for tile in 0..num_active_tiles {
        let d_base = tile * TILE_DESC_VEC4_STRIDE;
        let header = load_vec4(tile_descriptors, d_base);
        let src_quad = header.x.to_bits() as usize;
        let desc_dst = header.y.to_bits() as usize;
        if desc_dst == dst_quad && src_quad < num_quads {
            let f_base = ((stem * num_quads + src_quad) * num_frames + frame) * FEATURE_VEC4_STRIDE;
            let v_eff_l = load_vec4(feature_outputs, f_base);
            let v_eff_r = load_vec4(feature_outputs, f_base + 1);
            let odd_l = load_vec4(feature_outputs, f_base + 2);
            let odd_r = load_vec4(feature_outputs, f_base + 3);
            let even_l = load_vec4(feature_outputs, f_base + 4);
            let even_r = load_vec4(feature_outputs, f_base + 5);

            let w_odd = [
                load_vec4(tile_descriptors, d_base + 1),
                load_vec4(tile_descriptors, d_base + 2),
                load_vec4(tile_descriptors, d_base + 3),
                load_vec4(tile_descriptors, d_base + 4),
            ];
            let w_even = [
                load_vec4(tile_descriptors, d_base + 5),
                load_vec4(tile_descriptors, d_base + 6),
                load_vec4(tile_descriptors, d_base + 7),
                load_vec4(tile_descriptors, d_base + 8),
            ];
            let w_lru = [
                load_vec4(tile_descriptors, d_base + 9),
                load_vec4(tile_descriptors, d_base + 10),
                load_vec4(tile_descriptors, d_base + 11),
                load_vec4(tile_descriptors, d_base + 12),
            ];

            acc_l = accumulate_mat4x4_col_major(acc_l, w_odd, odd_l);
            acc_l = accumulate_mat4x4_col_major(acc_l, w_even, even_l);
            acc_l = accumulate_mat4x4_col_major(acc_l, w_lru, v_eff_l);

            acc_r = accumulate_mat4x4_col_major(acc_r, w_odd, odd_r);
            acc_r = accumulate_mat4x4_col_major(acc_r, w_even, even_r);
            acc_r = accumulate_mat4x4_col_major(acc_r, w_lru, v_eff_r);
        }
    }

    store_vec4(block_inputs, dst_in_base, canonicalize_vec4(acc_l));
    store_vec4(block_inputs, dst_in_base + 1, canonicalize_vec4(acc_r));
}

#[inline]
fn run_pullback_mode(
    wg_id: UVec3,
    lid: usize,
    uniforms: &TileMixUniforms,
    d_loss_d_inputs: &[Vec4],
    tile_descriptors: &[Vec4],
    d_loss_d_features: &mut [Vec4],
) {
    let num_frames = uniforms.num_frames as usize;
    let num_quads = (uniforms.num_quads as usize).max(1);
    let num_active_tiles = uniforms.num_active_tiles as usize;
    let src_quad = if uniforms.block_wave == BLOCK_WAVE_ALL {
        let q = wg_id.x as usize;
        if q < num_quads { q } else { usize::MAX }
    } else {
        resolve_quad_idx(uniforms.block_wave, wg_id.x as usize, num_quads)
    };
    if src_quad >= num_quads {
        return;
    }

    let frame = (wg_id.y as usize) * TILE_FRAMES + lid;
    if frame >= num_frames {
        return;
    }
    let stem = wg_id.z as usize;
    let f_base = ((stem * num_quads + src_quad) * num_frames + frame) * FEATURE_VEC4_STRIDE;
    let mut g_veff_l = load_vec4(d_loss_d_features, f_base);
    let mut g_veff_r = load_vec4(d_loss_d_features, f_base + 1);
    let mut g_odd_l = load_vec4(d_loss_d_features, f_base + 2);
    let mut g_odd_r = load_vec4(d_loss_d_features, f_base + 3);
    let mut g_even_l = load_vec4(d_loss_d_features, f_base + 4);
    let mut g_even_r = load_vec4(d_loss_d_features, f_base + 5);

    for tile in 0..num_active_tiles {
        let d_base = tile * TILE_DESC_VEC4_STRIDE;
        let header = load_vec4(tile_descriptors, d_base);
        let desc_src = header.x.to_bits() as usize;
        let dst_quad = header.y.to_bits() as usize;
        if desc_src == src_quad && dst_quad < num_quads {
            let dl_base =
                ((stem * num_quads + dst_quad) * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
            let dl_l = load_vec4(d_loss_d_inputs, dl_base);
            let dl_r = load_vec4(d_loss_d_inputs, dl_base + 1);

            let w_odd = [
                load_vec4(tile_descriptors, d_base + 1),
                load_vec4(tile_descriptors, d_base + 2),
                load_vec4(tile_descriptors, d_base + 3),
                load_vec4(tile_descriptors, d_base + 4),
            ];
            let w_even = [
                load_vec4(tile_descriptors, d_base + 5),
                load_vec4(tile_descriptors, d_base + 6),
                load_vec4(tile_descriptors, d_base + 7),
                load_vec4(tile_descriptors, d_base + 8),
            ];
            let w_lru = [
                load_vec4(tile_descriptors, d_base + 9),
                load_vec4(tile_descriptors, d_base + 10),
                load_vec4(tile_descriptors, d_base + 11),
                load_vec4(tile_descriptors, d_base + 12),
            ];

            g_odd_l = transpose_mat4x4_accum(g_odd_l, w_odd, dl_l);
            g_odd_r = transpose_mat4x4_accum(g_odd_r, w_odd, dl_r);
            g_even_l = transpose_mat4x4_accum(g_even_l, w_even, dl_l);
            g_even_r = transpose_mat4x4_accum(g_even_r, w_even, dl_r);
            g_veff_l = transpose_mat4x4_accum(g_veff_l, w_lru, dl_l);
            g_veff_r = transpose_mat4x4_accum(g_veff_r, w_lru, dl_r);
        }
    }

    store_vec4(d_loss_d_features, f_base, canonicalize_vec4(g_veff_l));
    store_vec4(d_loss_d_features, f_base + 1, canonicalize_vec4(g_veff_r));
    store_vec4(d_loss_d_features, f_base + 2, canonicalize_vec4(g_odd_l));
    store_vec4(d_loss_d_features, f_base + 3, canonicalize_vec4(g_odd_r));
    store_vec4(d_loss_d_features, f_base + 4, canonicalize_vec4(g_even_l));
    store_vec4(d_loss_d_features, f_base + 5, canonicalize_vec4(g_even_r));
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn run_partial_weights_mode(
    wg_id: UVec3,
    lid: usize,
    sg_lid: u32,
    uniforms: &TileMixUniforms,
    d_loss_d_inputs: &[Vec4],
    tile_descriptors: &[Vec4],
    feature_outputs: &[Vec4],
    tile_weight_grads: &mut [f32],
    reduce_a: &mut [f32; TILE_MIX_REDUCE_A_FLOATS],
    reduce_b: &mut [f32; TILE_MIX_REDUCE_B_FLOATS],
) {
    let tile = wg_id.x as usize;
    let frame_tile = wg_id.y as usize;
    let stem = wg_id.z as usize;
    let num_frames = uniforms.num_frames as usize;
    let num_frame_tiles = (uniforms.num_tiles as usize).max(1);
    let num_active_tiles = uniforms.num_active_tiles as usize;
    let num_quads = (uniforms.num_quads as usize).max(1);
    if tile >= num_active_tiles || frame_tile >= num_frame_tiles {
        return;
    }

    let header = load_vec4(tile_descriptors, tile * TILE_DESC_VEC4_STRIDE);
    let src_quad = header.x.to_bits() as usize;
    let dst_quad = header.y.to_bits() as usize;
    let active_tile = src_quad < num_quads && dst_quad < num_quads;

    let frame = frame_tile * TILE_FRAMES + lid;
    let in_frame = active_tile && frame < num_frames;
    let (v_eff_l, v_eff_r, odd_l, odd_r, even_l, even_r, dl_l, dl_r) = if in_frame {
        let f_base = ((stem * num_quads + src_quad) * num_frames + frame) * FEATURE_VEC4_STRIDE;
        let dl_base = ((stem * num_quads + dst_quad) * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
        (
            load_vec4(feature_outputs, f_base),
            load_vec4(feature_outputs, f_base + 1),
            load_vec4(feature_outputs, f_base + 2),
            load_vec4(feature_outputs, f_base + 3),
            load_vec4(feature_outputs, f_base + 4),
            load_vec4(feature_outputs, f_base + 5),
            load_vec4(d_loss_d_inputs, dl_base),
            load_vec4(d_loss_d_inputs, dl_base + 1),
        )
    } else {
        (
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
        )
    };

    let out_base = (num_active_tiles
        + ((stem * num_active_tiles + tile) * num_frame_tiles + frame_tile))
        * TILE_WEIGHT_GRAD_FLOATS;

    for s in 0..4 {
        let ol_s = crate::core_math::get_lane(odd_l, s);
        let or_s = crate::core_math::get_lane(odd_r, s);
        let (s0, s1) = reduce_sum_f32_pair_256(
            reduce_a,
            reduce_b,
            lid,
            ol_s * dl_l.x + or_s * dl_r.x,
            ol_s * dl_l.y + or_s * dl_r.y,
            sg_lid,
        );
        let (s2, s3) = reduce_sum_f32_pair_256(
            reduce_a,
            reduce_b,
            lid,
            ol_s * dl_l.z + or_s * dl_r.z,
            ol_s * dl_l.w + or_s * dl_r.w,
            sg_lid,
        );

        let el_s = crate::core_math::get_lane(even_l, s);
        let er_s = crate::core_math::get_lane(even_r, s);
        let (e0, e1) = reduce_sum_f32_pair_256(
            reduce_a,
            reduce_b,
            lid,
            el_s * dl_l.x + er_s * dl_r.x,
            el_s * dl_l.y + er_s * dl_r.y,
            sg_lid,
        );
        let (e2, e3) = reduce_sum_f32_pair_256(
            reduce_a,
            reduce_b,
            lid,
            el_s * dl_l.z + er_s * dl_r.z,
            el_s * dl_l.w + er_s * dl_r.w,
            sg_lid,
        );

        let vl_s = crate::core_math::get_lane(v_eff_l, s);
        let vr_s = crate::core_math::get_lane(v_eff_r, s);
        let (l0, l1) = reduce_sum_f32_pair_256(
            reduce_a,
            reduce_b,
            lid,
            vl_s * dl_l.x + vr_s * dl_r.x,
            vl_s * dl_l.y + vr_s * dl_r.y,
            sg_lid,
        );
        let (l2, l3) = reduce_sum_f32_pair_256(
            reduce_a,
            reduce_b,
            lid,
            vl_s * dl_l.z + vr_s * dl_r.z,
            vl_s * dl_l.w + vr_s * dl_r.w,
            sg_lid,
        );

        if lid == 0 {
            let row = s * 4;
            tile_weight_grads[clamp_idx(out_base + row, tile_weight_grads.len())] = s0;
            tile_weight_grads[clamp_idx(out_base + row + 1, tile_weight_grads.len())] = s1;
            tile_weight_grads[clamp_idx(out_base + row + 2, tile_weight_grads.len())] = s2;
            tile_weight_grads[clamp_idx(out_base + row + 3, tile_weight_grads.len())] = s3;

            tile_weight_grads[clamp_idx(out_base + 16 + row, tile_weight_grads.len())] = e0;
            tile_weight_grads[clamp_idx(out_base + 16 + row + 1, tile_weight_grads.len())] = e1;
            tile_weight_grads[clamp_idx(out_base + 16 + row + 2, tile_weight_grads.len())] = e2;
            tile_weight_grads[clamp_idx(out_base + 16 + row + 3, tile_weight_grads.len())] = e3;

            tile_weight_grads[clamp_idx(out_base + 32 + row, tile_weight_grads.len())] = l0;
            tile_weight_grads[clamp_idx(out_base + 32 + row + 1, tile_weight_grads.len())] = l1;
            tile_weight_grads[clamp_idx(out_base + 32 + row + 2, tile_weight_grads.len())] = l2;
            tile_weight_grads[clamp_idx(out_base + 32 + row + 3, tile_weight_grads.len())] = l3;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn run_reduce_weights_mode(
    wg_id: UVec3,
    lid: usize,
    sg_lid: u32,
    uniforms: &TileMixUniforms,
    tile_weight_grads: &mut [f32],
    reduce_a: &mut [f32; TILE_MIX_REDUCE_A_FLOATS],
    reduce_b: &mut [f32; TILE_MIX_REDUCE_B_FLOATS],
) {
    let tile = wg_id.x as usize;
    let num_active_tiles = uniforms.num_active_tiles as usize;
    let num_frame_tiles = (uniforms.num_tiles as usize).max(1);
    let num_stems = (uniforms.num_stems as usize).max(1);
    if tile >= num_active_tiles {
        return;
    }

    let out_base = tile * TILE_WEIGHT_GRAD_FLOATS;
    let total_chunks = num_stems * num_frame_tiles;

    for pair_idx in 0..(TILE_WEIGHT_GRAD_FLOATS / 2) {
        let w0 = pair_idx * 2;
        let w1 = w0 + 1;
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;

        let mut stem = 0_usize;
        while stem < num_stems {
            let mut ft = lid;
            while ft < num_frame_tiles {
                let partial_idx =
                    num_active_tiles + ((stem * num_active_tiles + tile) * num_frame_tiles + ft);
                let src_base = partial_idx * TILE_WEIGHT_GRAD_FLOATS;
                acc0 += tile_weight_grads[clamp_idx(src_base + w0, tile_weight_grads.len())];
                acc1 += tile_weight_grads[clamp_idx(src_base + w1, tile_weight_grads.len())];
                ft += WG_THREADS;
            }
            stem += 1;
        }
        let _ = total_chunks;
        let (sum0, sum1) = reduce_sum_f32_pair_256(reduce_a, reduce_b, lid, acc0, acc1, sg_lid);
        if lid == 0 {
            tile_weight_grads[clamp_idx(out_base + w0, tile_weight_grads.len())] = flush_zero(sum0);
            tile_weight_grads[clamp_idx(out_base + w1, tile_weight_grads.len())] = flush_zero(sum1);
        }
    }
}

/// Executes `tile_mix_fwd_adj` (`dir_mode in 0..=3`) inside `2,048 B` workgroup SRAM.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn tile_mix_fwd_adj_impl(
    wg_id: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &TileMixUniforms,
    block_inputs_or_d_loss_d_inputs: &mut [Vec4],
    tile_descriptors: &[Vec4],
    feature_outputs: &[Vec4],
    tile_weight_grads: &mut [f32],
    d_loss_d_features: &mut [Vec4],
    reduce_a: &mut [f32; TILE_MIX_REDUCE_A_FLOATS],
    reduce_b: &mut [f32; TILE_MIX_REDUCE_B_FLOATS],
) {
    let lid = (lid_u32 as usize) & 255;
    match uniforms.dir_mode {
        TILE_MIX_DIR_FORWARD => run_forward_mode(
            wg_id,
            lid,
            uniforms,
            block_inputs_or_d_loss_d_inputs,
            tile_descriptors,
            feature_outputs,
        ),
        TILE_MIX_DIR_PULLBACK => run_pullback_mode(
            wg_id,
            lid,
            uniforms,
            block_inputs_or_d_loss_d_inputs,
            tile_descriptors,
            d_loss_d_features,
        ),
        TILE_MIX_DIR_PARTIAL_WEIGHTS => run_partial_weights_mode(
            wg_id,
            lid,
            sg_lid,
            uniforms,
            block_inputs_or_d_loss_d_inputs,
            tile_descriptors,
            feature_outputs,
            tile_weight_grads,
            reduce_a,
            reduce_b,
        ),
        TILE_MIX_DIR_REDUCE_WEIGHTS => run_reduce_weights_mode(
            wg_id,
            lid,
            sg_lid,
            uniforms,
            tile_weight_grads,
            reduce_a,
            reduce_b,
        ),
        _ => {}
    }
}
