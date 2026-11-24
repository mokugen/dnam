//! Deterministic CPU reference implementation of 2D / split-64 `ozaki_gram_rhs_i32`
//! (`tail_base = active_pairs * 4`, `[hi64_lo, hi64_hi, lo64_lo, lo64_hi]`, and `0-ULP` host
//! `f64` reconstruction) (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{accum_ozaki_chunk_i32, pow2_ceil_fp32, reconstruct_ozaki_dot_f64};

use crate::error::TrainError;
use crate::gpu::abi::{
    GPU_OZAKI_MAX_CHUNK_LEN, OZAKI_FLAG_ACCUMULATE_SPLIT64, OZAKI_SPLIT64_WORDS_PER_PAIR,
    OZAKI_WORDS_PER_PAIR,
};
use crate::runner::types::{OzakiGramInput, OzakiGramOutput, OzakiSplit64Output};
use crate::wide_gram::{Wide192Accum, uplo_u_index, uplo_u_packed_len};

#[inline(always)]
pub(super) fn resolve_col_inv_scale(entry: [f32; 2]) -> f32 {
    if entry[1] > 0.0 && entry[1].is_finite() {
        entry[1]
    } else {
        pow2_ceil_fp32(entry[0]).1
    }
}

#[inline(always)]
pub(super) fn resolve_col_scale(entry: [f32; 2]) -> f32 {
    if entry[1] > 0.0 && entry[1].is_finite() {
        entry[0]
    } else {
        pow2_ceil_fp32(entry[0]).0
    }
}

fn validate_ozaki_slices(
    features: &[f32],
    targets: &[f32],
    col_scales: &[[f32; 2]],
    num_frames: usize,
    dim: usize,
    tc: usize,
    ctx: &'static str,
) -> Result<(), TrainError> {
    if num_frames == 0 || dim == 0 {
        return Err(TrainError::InvalidDimensions {
            context: ctx,
            expected: 1,
            actual: 0,
        });
    }
    if features.len() < dim * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: ctx,
            expected: dim * num_frames,
            actual: features.len(),
        });
    }
    if tc > 0 && targets.len() < tc * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: ctx,
            expected: tc * num_frames,
            actual: targets.len(),
        });
    }
    if col_scales.len() < dim + tc {
        return Err(TrainError::InvalidDimensions {
            context: ctx,
            expected: dim + tc,
            actual: col_scales.len(),
        });
    }
    Ok(())
}

/// Deterministic CPU reference implementation of 2-pass `ozaki_gram_rhs_i32` (`reduce_pass` `0` or `1`).
///
/// Enforces the `target_channels == 0` guard: trailing buffer slots beyond `gram_pairs` remain untouched.
pub fn run_ozaki_gram_rhs_i32_cpu(
    input: &OzakiGramInput<'_>,
) -> Result<OzakiGramOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let dim = u.feature_dim as usize;
    let tc = u.target_channels as usize;
    let chunk_len = (u.chunk_len as usize).clamp(1, GPU_OZAKI_MAX_CHUNK_LEN);
    let num_chunks = (u.num_chunks as usize).max(num_frames.div_ceil(chunk_len));

    validate_ozaki_slices(
        input.features,
        input.targets,
        input.col_scales,
        num_frames,
        dim,
        tc,
        "ozaki_gram_rhs_i32",
    )?;

    let gram_pairs = uplo_u_packed_len(dim);
    let active_pairs = gram_pairs + dim * tc;
    let tail_base = active_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let min_words = tail_base + active_pairs * num_chunks * OZAKI_WORDS_PER_PAIR;
    let mut ozaki_buckets = match input.seed_buckets {
        Some(seed) => {
            if seed.len() < min_words {
                return Err(TrainError::InvalidDimensions {
                    context: "ozaki_gram_rhs_i32(seed_buckets)",
                    expected: min_words,
                    actual: seed.len(),
                });
            }
            seed.to_vec()
        }
        None => vec![0_i32; min_words],
    };
    let num_pairs = (ozaki_buckets.len() - tail_base) / (num_chunks * OZAKI_WORDS_PER_PAIR);

    if u.reduce_pass == 0 {
        let mut pair_idx = 0_usize;
        for i in 0..dim {
            let isx = resolve_col_inv_scale(input.col_scales[i]);
            for j in i..dim {
                let isy = resolve_col_inv_scale(input.col_scales[j]);
                for ch in 0..num_chunks {
                    let s = ch * chunk_len;
                    let e = (s + chunk_len).min(num_frames);
                    let words = accum_ozaki_chunk_i32(
                        &input.features[i * num_frames + s..i * num_frames + e],
                        &input.features[j * num_frames + s..j * num_frames + e],
                        isx,
                        isy,
                    );
                    let dst = tail_base + (pair_idx * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                    ozaki_buckets[dst..dst + OZAKI_WORDS_PER_PAIR].copy_from_slice(&words);
                }
                pair_idx += 1;
            }
        }
        for i in 0..dim {
            let isx = resolve_col_inv_scale(input.col_scales[i]);
            for t_ch in 0..tc {
                let isy = resolve_col_inv_scale(input.col_scales[dim + t_ch]);
                for ch in 0..num_chunks {
                    let s = ch * chunk_len;
                    let e = (s + chunk_len).min(num_frames);
                    let words = accum_ozaki_chunk_i32(
                        &input.features[i * num_frames + s..i * num_frames + e],
                        &input.targets[t_ch * num_frames + s..t_ch * num_frames + e],
                        isx,
                        isy,
                    );
                    let dst = tail_base + (pair_idx * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                    ozaki_buckets[dst..dst + OZAKI_WORDS_PER_PAIR].copy_from_slice(&words);
                }
                pair_idx += 1;
            }
        }
    } else {
        let accumulate = (u._pad & OZAKI_FLAG_ACCUMULATE_SPLIT64) != 0;
        for pair_idx in 0..active_pairs {
            let (mut hi64, mut lo64) = if accumulate {
                let b = pair_idx * OZAKI_SPLIT64_WORDS_PER_PAIR;
                let h = (((ozaki_buckets[b + 1] as u32 as u64) << 32)
                    | (ozaki_buckets[b] as u32 as u64)) as i64;
                let l = (((ozaki_buckets[b + 3] as u32 as u64) << 32)
                    | (ozaki_buckets[b + 2] as u32 as u64)) as i64;
                (h, l)
            } else {
                (0_i64, 0_i64)
            };
            for ch in 0..num_chunks {
                let src = tail_base + (pair_idx * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                let w = &ozaki_buckets[src..src + OZAKI_WORDS_PER_PAIR];
                hi64 += (i64::from(w[0]) << 16) + (i64::from(w[1]) << 8) + i64::from(w[2]);
                lo64 += (i64::from(w[3]) << 8) + i64::from(w[4]);
            }
            let b = pair_idx * OZAKI_SPLIT64_WORDS_PER_PAIR;
            ozaki_buckets[b] = (hi64 as u64) as u32 as i32;
            ozaki_buckets[b + 1] = ((hi64 as u64) >> 32) as u32 as i32;
            ozaki_buckets[b + 2] = (lo64 as u64) as u32 as i32;
            ozaki_buckets[b + 3] = ((lo64 as u64) >> 32) as u32 as i32;
        }
    }

    Ok(OzakiGramOutput {
        ozaki_buckets,
        num_chunks,
        num_pairs,
    })
}

/// Runs full 2-pass split-64 Ozaki Gram + RHS reduction on CPU, returning `4 × u32` limbs per pair.
pub fn run_ozaki_gram_rhs_split64_cpu(
    features_col_major: &[f32],
    targets_col_major: &[f32],
    num_frames: usize,
    feature_dim: usize,
    target_channels: usize,
    col_scales: &[[f32; 2]],
) -> Result<OzakiSplit64Output, TrainError> {
    validate_ozaki_slices(
        features_col_major,
        targets_col_major,
        col_scales,
        num_frames,
        feature_dim,
        target_channels,
        "run_ozaki_gram_rhs_split64_cpu",
    )?;

    let gram_pairs = uplo_u_packed_len(feature_dim);
    let active_pairs = gram_pairs + feature_dim * target_channels;
    let num_chunks = num_frames.div_ceil(GPU_OZAKI_MAX_CHUNK_LEN);
    let mut split64_words = vec![0_u32; active_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR];

    let mut accumulate_pair =
        |pair_idx: usize, col_x: &[f32], isx: f32, col_y: &[f32], isy: f32| {
            let mut hi64 = 0_i64;
            let mut lo64 = 0_i64;
            for ch in 0..num_chunks {
                let s = ch * GPU_OZAKI_MAX_CHUNK_LEN;
                let e = (s + GPU_OZAKI_MAX_CHUNK_LEN).min(num_frames);
                let w = accum_ozaki_chunk_i32(&col_x[s..e], &col_y[s..e], isx, isy);
                hi64 += (i64::from(w[0]) << 16) + (i64::from(w[1]) << 8) + i64::from(w[2]);
                lo64 += (i64::from(w[3]) << 8) + i64::from(w[4]);
            }
            let b = pair_idx * OZAKI_SPLIT64_WORDS_PER_PAIR;
            let h_u = hi64 as u64;
            let l_u = lo64 as u64;
            split64_words[b] = h_u as u32;
            split64_words[b + 1] = (h_u >> 32) as u32;
            split64_words[b + 2] = l_u as u32;
            split64_words[b + 3] = (l_u >> 32) as u32;
        };

    let mut pair_idx = 0_usize;
    for i in 0..feature_dim {
        let isx = resolve_col_inv_scale(col_scales[i]);
        let col_x = &features_col_major[i * num_frames..(i + 1) * num_frames];
        for j in i..feature_dim {
            let isy = resolve_col_inv_scale(col_scales[j]);
            let col_y = &features_col_major[j * num_frames..(j + 1) * num_frames];
            accumulate_pair(pair_idx, col_x, isx, col_y, isy);
            pair_idx += 1;
        }
    }
    for i in 0..feature_dim {
        let isx = resolve_col_inv_scale(col_scales[i]);
        let col_x = &features_col_major[i * num_frames..(i + 1) * num_frames];
        for t_ch in 0..target_channels {
            let isy = resolve_col_inv_scale(col_scales[feature_dim + t_ch]);
            let col_y = &targets_col_major[t_ch * num_frames..(t_ch + 1) * num_frames];
            accumulate_pair(pair_idx, col_x, isx, col_y, isy);
            pair_idx += 1;
        }
    }

    Ok(OzakiSplit64Output {
        split64_words,
        num_pairs: active_pairs,
    })
}

/// Reconstructs column-major packed `uplo='U'` `f64` Gram matrix and RHS vectors from `i32`
/// Ozaki chunk buckets stored in the trailing region (`tail_base = active_pairs * 4`).
pub fn reconstruct_ozaki_gram_rhs_f64(
    output: &OzakiGramOutput,
    feature_dim: usize,
    target_channels: usize,
    col_scales: &[[f32; 2]],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
    if col_scales.len() < feature_dim + target_channels {
        return Err(TrainError::InvalidDimensions {
            context: "reconstruct_ozaki_gram_rhs_f64(col_scales)",
            expected: feature_dim + target_channels,
            actual: col_scales.len(),
        });
    }
    let gram_len = uplo_u_packed_len(feature_dim);
    let active_pairs = gram_len + feature_dim * target_channels;
    let tail_base = active_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let needed = tail_base + active_pairs * output.num_chunks * OZAKI_WORDS_PER_PAIR;
    if output.ozaki_buckets.len() < needed {
        return Err(TrainError::InvalidDimensions {
            context: "reconstruct_ozaki_gram_rhs_f64(ozaki_buckets)",
            expected: needed,
            actual: output.ozaki_buckets.len(),
        });
    }

    let mut gram_u = vec![0.0_f64; gram_len];
    let mut rhs = vec![vec![0.0_f64; target_channels]; feature_dim];
    let mut pair_idx = 0_usize;

    for i in 0..feature_dim {
        let sx = resolve_col_scale(col_scales[i]);
        for j in i..feature_dim {
            let sy = resolve_col_scale(col_scales[j]);
            let mut acc = Wide192Accum::zero();
            for ch in 0..output.num_chunks {
                let b = tail_base + (pair_idx * output.num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                let mut words = [0_i32; OZAKI_WORDS_PER_PAIR];
                words.copy_from_slice(&output.ozaki_buckets[b..b + OZAKI_WORDS_PER_PAIR]);
                acc.add_f64(reconstruct_ozaki_dot_f64(&words, sx, sy));
            }
            gram_u[uplo_u_index(i, j)] = acc.to_f64();
            pair_idx += 1;
        }
    }
    for i in 0..feature_dim {
        let sx = resolve_col_scale(col_scales[i]);
        for tc in 0..target_channels {
            let sy = resolve_col_scale(col_scales[feature_dim + tc]);
            let mut acc = Wide192Accum::zero();
            for ch in 0..output.num_chunks {
                let b = tail_base + (pair_idx * output.num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                let mut words = [0_i32; OZAKI_WORDS_PER_PAIR];
                words.copy_from_slice(&output.ozaki_buckets[b..b + OZAKI_WORDS_PER_PAIR]);
                acc.add_f64(reconstruct_ozaki_dot_f64(&words, sx, sy));
            }
            rhs[i][tc] = acc.to_f64();
            pair_idx += 1;
        }
    }
    Ok((gram_u, rhs))
}

/// Reconstructs `0-ULP` equivalent `f64` Gram matrix (`uplo='U'`) and RHS vectors directly from
/// compact split-64 limbs (`hi64 * 2^-32 + lo64 * 2^-48` combined in exact `i128`).
pub fn reconstruct_ozaki_split64_f64(
    output: &OzakiSplit64Output,
    feature_dim: usize,
    target_channels: usize,
    col_scales: &[[f32; 2]],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
    if col_scales.len() < feature_dim + target_channels {
        return Err(TrainError::InvalidDimensions {
            context: "reconstruct_ozaki_split64_f64(col_scales)",
            expected: feature_dim + target_channels,
            actual: col_scales.len(),
        });
    }
    let gram_len = uplo_u_packed_len(feature_dim);
    let active_pairs = gram_len + feature_dim * target_channels;
    let needed = active_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    if output.split64_words.len() < needed {
        return Err(TrainError::InvalidDimensions {
            context: "reconstruct_ozaki_split64_f64(split64_words)",
            expected: needed,
            actual: output.split64_words.len(),
        });
    }

    const SCALE_POW_M48: f64 = 1.0 / ((1_u64 << 48) as f64);
    let recon_pair = |pair_idx: usize, sx: f32, sy: f32| -> f64 {
        if !sx.is_finite() || !sy.is_finite() || sx == 0.0 || sy == 0.0 {
            return 0.0;
        }
        let b = pair_idx * OZAKI_SPLIT64_WORDS_PER_PAIR;
        let w = &output.split64_words[b..b + OZAKI_SPLIT64_WORDS_PER_PAIR];
        let hi = (((w[1] as u64) << 32) | (w[0] as u64)) as i64;
        let lo = (((w[3] as u64) << 32) | (w[2] as u64)) as i64;
        let total = ((hi as i128) << 16) + (lo as i128);
        (total as f64) * SCALE_POW_M48 * f64::from(sx) * f64::from(sy)
    };

    let mut gram_u = vec![0.0_f64; gram_len];
    let mut rhs = vec![vec![0.0_f64; target_channels]; feature_dim];
    let mut pair_idx = 0_usize;
    for i in 0..feature_dim {
        let sx = resolve_col_scale(col_scales[i]);
        for j in i..feature_dim {
            let sy = resolve_col_scale(col_scales[j]);
            gram_u[uplo_u_index(i, j)] = recon_pair(pair_idx, sx, sy);
            pair_idx += 1;
        }
    }
    for i in 0..feature_dim {
        let sx = resolve_col_scale(col_scales[i]);
        for tc in 0..target_channels {
            let sy = resolve_col_scale(col_scales[feature_dim + tc]);
            rhs[i][tc] = recon_pair(pair_idx, sx, sy);
            pair_idx += 1;
        }
    }
    Ok((gram_u, rhs))
}
