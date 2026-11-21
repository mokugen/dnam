#![forbid(unsafe_code)]

//! Division-free upper-triangle Gram & RHS `i32` Ozaki mantissa-sliced accumulator
//! (`ozaki_gram_rhs_i32`) operating strictly in `5,120 B` workgroup SRAM
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{MAX_WORKGROUP_SRAM_BYTES, OZAKI_WORDS_PER_PAIR, OzakiGramUniforms};
use crate::reduce::{WG_THREADS, clamp_idx, is_finite_bits, reduce_sum_i32_5_256};
use core::mem::size_of;
use dnam_kernels::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_MANTISSA_BITS,
    IEEE754_F32_MANTISSA_MASK, IEEE754_F32_SIGN_MASK, OZAKI_MAX_CHUNK_LEN, accum_ozaki_pair_i32,
    pow2_ceil_fp32, slice_f32_mantissa_i8x3,
};
use spirv_std::glam::{UVec3, Vec2};

/// Workgroup `reduce_i32` capacity (`5 × 256 = 1,280 × i32 = 5,120 B`).
pub const OZAKI_REDUCE_I32_WORDS: usize = 5 * WG_THREADS;

/// Total `ozaki_gram_rhs_i32` workgroup SRAM footprint (`5,120 B <= 15,872 B`).
pub const OZAKI_GRAM_SRAM_BYTES: usize = OZAKI_REDUCE_I32_WORDS * size_of::<i32>();

const _: () = assert!(OZAKI_WORDS_PER_PAIR == 6);
const _: () = assert!(OZAKI_MAX_CHUNK_LEN == 1024);
const _: () = assert!(OZAKI_GRAM_SRAM_BYTES == 5_120);
const _: () = assert!(OZAKI_GRAM_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);

#[derive(Copy, Clone)]
struct OzakiPairTarget {
    valid: bool,
    src_y_is_feat: bool,
    col_x: usize,
    col_y: usize,
    scale_idx_x: usize,
    scale_idx_y: usize,
}

/// Decodes upper-triangle Gram `(i <= j)` or cross-correlation RHS `(i, ch)` strictly via
/// binary search with zero `IDIV`/`IREM` (`0..D` in at most `16` iterations).
#[inline]
fn decode_ozaki_pair_divfree(
    pair_idx: usize,
    feature_dim: usize,
    target_channels: usize,
) -> OzakiPairTarget {
    if feature_dim == 0 {
        return OzakiPairTarget {
            valid: false,
            src_y_is_feat: true,
            col_x: 0,
            col_y: 0,
            scale_idx_x: 0,
            scale_idx_y: 0,
        };
    }

    let num_gram_pairs = (feature_dim * (feature_dim + 1)) >> 1;
    if pair_idx < num_gram_pairs {
        let mut lo = 0_usize;
        let mut hi = feature_dim - 1;
        while lo < hi {
            let mid = lo + ((hi - lo + 1) >> 1);
            let rem = feature_dim - mid;
            let row_start = num_gram_pairs - ((rem * (rem + 1)) >> 1);
            if row_start <= pair_idx {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let rem = feature_dim - lo;
        let row_start = num_gram_pairs - ((rem * (rem + 1)) >> 1);
        let j = lo + (pair_idx - row_start);
        return OzakiPairTarget {
            valid: true,
            src_y_is_feat: true,
            col_x: lo,
            col_y: j,
            scale_idx_x: lo,
            scale_idx_y: j,
        };
    }

    let rhs_idx = pair_idx - num_gram_pairs;
    if target_channels == 0 || rhs_idx >= feature_dim * target_channels {
        return OzakiPairTarget {
            valid: false,
            src_y_is_feat: false,
            col_x: 0,
            col_y: 0,
            scale_idx_x: 0,
            scale_idx_y: 0,
        };
    }

    let mut lo = 0_usize;
    let mut hi = feature_dim - 1;
    while lo < hi {
        let mid = lo + ((hi - lo + 1) >> 1);
        if mid * target_channels <= rhs_idx {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let ch = rhs_idx - lo * target_channels;
    OzakiPairTarget {
        valid: true,
        src_y_is_feat: false,
        col_x: lo,
        col_y: ch,
        scale_idx_x: lo,
        scale_idx_y: feature_dim + ch,
    }
}

/// Resolves exact power-of-two `inv_scale` supporting both `[scale_or_max_abs, inv_scale]` and
/// `[max_abs, 0.0]` host packing conventions.
#[inline(always)]
fn resolve_inv_scale(entry: Vec2) -> f32 {
    let y_bits = entry.y.to_bits();
    let y_exp = (y_bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES;
    let is_pos_pow2 = (y_bits & (IEEE754_F32_SIGN_MASK | IEEE754_F32_MANTISSA_MASK)) == 0
        && y_exp > 0
        && y_exp < IEEE754_F32_EXP_ALL_ONES;
    let abs_x = f32::from_bits(entry.x.to_bits() & IEEE754_F32_ABS_MASK);
    if is_pos_pow2 && is_finite_bits(entry.x) && (abs_x * entry.y) <= 1.0 {
        entry.y
    } else {
        pow2_ceil_fp32(abs_x).1
    }
}

/// Executes one `(pair_idx = wg_id.x, chunk_idx = wg_id.y)` tile of `ozaki_gram_rhs_i32`
/// across `256` threads (`4` frames/thread up to `1024`, `5,120 B` SRAM, bit-exact `i32` sum).
#[inline]
pub fn ozaki_gram_rhs_i32_impl(
    wg_id: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &OzakiGramUniforms,
    features: &[f32],
    targets: &[f32],
    col_scales: &[Vec2],
    ozaki_buckets: &mut [i32],
    reduce_i32: &mut [i32; OZAKI_REDUCE_I32_WORDS],
) {
    let lid = lid_u32 as usize;
    let pair_idx = wg_id.x as usize;
    let chunk_idx = wg_id.y as usize;

    let num_frames = uniforms.num_frames as usize;
    let feature_dim = uniforms.feature_dim as usize;
    let target_channels = uniforms.target_channels as usize;
    let clen = (uniforms.chunk_len as usize).min(OZAKI_MAX_CHUNK_LEN);
    let num_chunks = (uniforms.num_chunks as usize).max(1);

    let pair = decode_ozaki_pair_divfree(pair_idx, feature_dim, target_channels);
    let sx_entry = col_scales[clamp_idx(pair.scale_idx_x, col_scales.len())];
    let sy_entry = col_scales[clamp_idx(pair.scale_idx_y, col_scales.len())];
    let inv_sx = resolve_inv_scale(sx_entry);
    let inv_sy = resolve_inv_scale(sy_entry);

    let chunk_base = chunk_idx * clen;
    let active_pair = pair.valid && chunk_idx < num_chunks;
    let mut acc = [0_i32; OZAKI_WORDS_PER_PAIR];

    for step in 0..4 {
        let local_idx = step * WG_THREADS + lid;
        let frame = chunk_base + local_idx;
        if active_pair && local_idx < clen && frame < num_frames {
            let x = features[clamp_idx(pair.col_x * num_frames + frame, features.len())];
            let y = if pair.src_y_is_feat {
                features[clamp_idx(pair.col_y * num_frames + frame, features.len())]
            } else {
                targets[clamp_idx(pair.col_y * num_frames + frame, targets.len())]
            };
            let sx = slice_f32_mantissa_i8x3(x, inv_sx);
            let sy = slice_f32_mantissa_i8x3(y, inv_sy);
            accum_ozaki_pair_i32(&mut acc, sx, sy);
        }
    }

    let sums = reduce_sum_i32_5_256(
        reduce_i32,
        lid,
        [acc[0], acc[1], acc[2], acc[3], acc[4]],
        sg_lid,
    );

    if lid == 0 && active_pair {
        let out_base = (pair_idx * num_chunks + chunk_idx) * OZAKI_WORDS_PER_PAIR;
        for k in 0..5 {
            ozaki_buckets[clamp_idx(out_base + k, ozaki_buckets.len())] = sums[k];
        }
        ozaki_buckets[clamp_idx(out_base + 5, ozaki_buckets.len())] = 0;
    }
}
