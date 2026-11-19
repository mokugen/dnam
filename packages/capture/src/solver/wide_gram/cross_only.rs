#![forbid(unsafe_code)]
//! `O(m · dim)` wide-exponent cross-correlation contraction (`solver::wide_gram::cross_only`).
//!
//! Shares frame validation, `[256]` micro-batch envelope analysis, and `192`-bit cancellation-safe
//! accumulation with `wide_gram` so shared-prefix tiers avoid redundant `O(m · dim²)` Gram matrices.

#[cfg(test)]
use super::mask256_or;
use super::{MICRO_BATCH_FRAMES, Wide192Accum, mask256_overlaps};

pub(super) struct MicroBatchBounds {
    pub(super) b_len: usize,
    pub(super) w_extreme: bool,
    pub(super) w_min: f64,
    pub(super) y_nz_mask: [[u64; 4]; 2],
    pub(super) max_y: [f64; 2],
    pub(super) min_nz_y: [f64; 2],
    pub(super) y_batch_mono: bool,
    pub(super) col_max_f: [f64; 256],
    pub(super) col_max_wf: [f64; 256],
    pub(super) col_wide_exp: [bool; 256],
    pub(super) col_extreme: [bool; 256],
    pub(super) batch_y: [[f64; 2]; MICRO_BATCH_FRAMES],
    pub(super) batch_w: [f64; MICRO_BATCH_FRAMES],
}

pub(super) fn scan_valid_frame_indices(
    features: &[[f32; 256]],
    targets_lr: &[[f64; 2]],
    weights: &[f64],
    frame_stride: usize,
    dim: usize,
) -> (Vec<usize>, usize) {
    let n_frames = features.len().min(targets_lr.len()).min(weights.len());
    let stride = frame_stride.max(1);
    let mut valid_indices = Vec::with_capacity(n_frames.div_ceil(stride));
    let mut scan_idx = 0usize;
    while scan_idx < n_frames {
        let w = weights[scan_idx];
        let y = targets_lr[scan_idx];
        let fb = &features[scan_idx][..dim];
        if w.is_finite()
            && w > 0.0
            && y[0].is_finite()
            && y[1].is_finite()
            && fb.iter().all(|v| v.is_finite())
        {
            valid_indices.push(scan_idx);
        }
        scan_idx += stride;
    }
    (valid_indices, n_frames)
}

pub(super) fn analyze_micro_batch_bounds(
    features: &[[f32; 256]],
    targets_lr: &[[f64; 2]],
    weights: &[f64],
    chunk_indices: &[usize],
    scale_w: f64,
    dim: usize,
) -> MicroBatchBounds {
    let b_len = chunk_indices.len();
    let mut w_max = 0.0_f64;
    let mut w_min = f64::INFINITY;
    let mut y_nz_mask = [[0_u64; 4]; 2];
    let mut max_y = [0.0_f64; 2];
    let mut min_nz_y = [f64::INFINITY; 2];
    let mut y_batch_mono = true;
    let mut col_max_f32 = [0.0_f32; 256];
    let mut col_min_nz_f32 = [f32::INFINITY; 256];
    let mut batch_y = [[0.0_f64; 2]; MICRO_BATCH_FRAMES];
    let mut batch_w = [0.0_f64; MICRO_BATCH_FRAMES];

    for (k, &idx) in chunk_indices.iter().enumerate() {
        let w = weights[idx] * scale_w;
        batch_w[k] = w;
        w_max = w_max.max(w);
        w_min = w_min.min(w);

        let y = targets_lr[idx];
        batch_y[k] = y;
        if y[0].to_bits() != y[1].to_bits() {
            y_batch_mono = false;
        }
        let w_word = k >> 6;
        let w_bit = 1_u64 << (k & 63);
        let ay0 = y[0].abs();
        if ay0 > 0.0 {
            y_nz_mask[0][w_word] |= w_bit;
            max_y[0] = max_y[0].max(ay0);
            min_nz_y[0] = min_nz_y[0].min(ay0);
        }
        let ay1 = y[1].abs();
        if ay1 > 0.0 {
            y_nz_mask[1][w_word] |= w_bit;
            max_y[1] = max_y[1].max(ay1);
            min_nz_y[1] = min_nz_y[1].min(ay1);
        }

        let fb = &features[idx];
        for d in 0..dim {
            let af = fb[d].abs();
            col_max_f32[d] = col_max_f32[d].max(af);
            let nz = if af > 0.0 { af } else { f32::INFINITY };
            col_min_nz_f32[d] = col_min_nz_f32[d].min(nz);
        }
    }

    let w_extreme = w_max > 1.0e90 || w_min < 1.0e-90;
    let mut col_max_f = [0.0_f64; 256];
    let mut col_max_wf = [0.0_f64; 256];
    let mut col_wide_exp = [false; 256];
    let mut col_extreme = [false; 256];

    for d in 0..dim {
        let mf = f64::from(col_max_f32[d]);
        if mf == 0.0 {
            continue;
        }
        let nf = f64::from(col_min_nz_f32[d]);
        col_max_f[d] = mf;
        if !w_extreme
            && mf * w_max <= nf * w_min * 65_536.0
            && mf * w_max <= 1.0e130
            && nf * w_min >= 1.0e-130
        {
            col_max_wf[d] = mf * w_max;
        } else {
            let mut mwf = 0.0_f64;
            let mut nwf = f64::INFINITY;
            for (k, &idx) in chunk_indices.iter().enumerate() {
                let fd = f64::from(features[idx][d]);
                if fd != 0.0 {
                    let awf = (batch_w[k] * fd).abs();
                    if awf > 0.0 && awf.is_finite() {
                        mwf = mwf.max(awf);
                        nwf = nwf.min(awf);
                    }
                }
            }
            col_max_wf[d] = mwf;
            let extreme = w_extreme
                || mf > 1.0e130
                || mwf > 1.0e130
                || nf < 1.0e-130
                || nwf < 1.0e-130
                || mwf == 0.0;
            col_extreme[d] = extreme;
            col_wide_exp[d] = extreme || mf > nf * 65_536.0 || mwf > nwf * 65_536.0;
        }
    }

    MicroBatchBounds {
        b_len,
        w_extreme,
        w_min,
        y_nz_mask,
        max_y,
        min_nz_y,
        y_batch_mono,
        col_max_f,
        col_max_wf,
        col_wide_exp,
        col_extreme,
        batch_y,
        batch_w,
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn accum_cross_corr_column(
    wide_c_d: &mut [Wide192Accum; 2],
    col_nz: [u64; 4],
    d: usize,
    bounds: &MicroBatchBounds,
    y_wide: [bool; 2],
    features: &[[f32; 256]],
    chunk_indices: &[usize],
    wf_at: impl Fn(usize) -> f64,
) {
    let ch_limit = if bounds.y_batch_mono { 1 } else { 2 };
    for ch in 0..ch_limit {
        if !mask256_overlaps(col_nz, bounds.y_nz_mask[ch]) {
            continue;
        }
        let fast_ok = !bounds.col_wide_exp[d] && !y_wide[ch];
        let mut used_fast = false;
        if fast_ok {
            let mut s = 0.0_f64;
            for k in 0..bounds.b_len {
                s += wf_at(k) * bounds.batch_y[k][ch];
            }
            if s.abs() > bounds.col_max_wf[d] * bounds.max_y[ch] * (1.0 / 4_294_967_296.0) {
                wide_c_d[ch].add_f64(s);
                if bounds.y_batch_mono {
                    wide_c_d[1].add_f64(s);
                }
                used_fast = true;
            }
        }
        if !used_fast {
            let mut acc = Wide192Accum::zero();
            for (k, &idx) in chunk_indices.iter().enumerate() {
                acc.add_three_way_f64(
                    bounds.batch_w[k],
                    f64::from(features[idx][d]),
                    bounds.batch_y[k][ch],
                );
            }
            wide_c_d[ch].merge(&acc);
            if bounds.y_batch_mono {
                wide_c_d[1].merge(&acc);
            }
        }
    }
}

/// Computes only the `dim × 2` wide-exponent cross-correlation matrix (`0-ULP` identical to
/// `contract_dim_gram_tiled_wide192_with_backend(...).1`) without constructing the Gram matrix.
#[cfg(test)]
pub(in crate::solver) fn contract_dim_cross_only_wide192(
    features: &[[f32; 256]],
    targets_lr: &[[f64; 2]],
    weights: &[f64],
    frame_stride: usize,
    dim: usize,
) -> Vec<[f64; 2]> {
    let dim = dim.clamp(4, 256) & !3;
    let (valid_indices, n_frames) =
        scan_valid_frame_indices(features, targets_lr, weights, frame_stride, dim);
    let sampled_frames = valid_indices.len();
    if sampled_frames == 0 {
        return vec![[0.0_f64; 2]; dim];
    }

    let mut wide_c = vec![[Wide192Accum::zero(); 2]; dim];
    let scale_w = (n_frames as f64) / (sampled_frames as f64);
    let mut wf_col = [0.0_f64; MICRO_BATCH_FRAMES];

    for chunk_indices in valid_indices.chunks(MICRO_BATCH_FRAMES) {
        let bounds =
            analyze_micro_batch_bounds(features, targets_lr, weights, chunk_indices, scale_w, dim);
        let y_either_nz = mask256_or(bounds.y_nz_mask[0], bounds.y_nz_mask[1]);
        if y_either_nz == [0; 4] {
            continue;
        }
        let y_wide = [
            bounds.max_y[0] > 1.0e130
                || bounds.min_nz_y[0] < 1.0e-130
                || bounds.max_y[0] > bounds.min_nz_y[0] * 65_536.0,
            bounds.max_y[1] > 1.0e130
                || bounds.min_nz_y[1] < 1.0e-130
                || bounds.max_y[1] > bounds.min_nz_y[1] * 65_536.0,
        ];
        for (d, wide_c_d) in wide_c.iter_mut().enumerate().take(dim) {
            if bounds.col_max_f[d] == 0.0 {
                continue;
            }
            let mut col_nz = [0_u64; 4];
            for (k, &idx) in chunk_indices.iter().enumerate() {
                let fd = f64::from(features[idx][d]);
                wf_col[k] = bounds.batch_w[k] * fd;
                if fd != 0.0 {
                    col_nz[k >> 6] |= 1_u64 << (k & 63);
                }
            }
            if !mask256_overlaps(col_nz, y_either_nz) {
                continue;
            }
            accum_cross_corr_column(
                wide_c_d,
                col_nz,
                d,
                &bounds,
                y_wide,
                features,
                chunk_indices,
                |k| wf_col[k],
            );
        }
    }

    wide_c
        .into_iter()
        .map(|[l, r]| [l.to_f64(), r.to_f64()])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::ScalarTrainerBackend;
    use crate::solver::wide_gram::contract_dim_gram_tiled_wide192_with_backend;

    #[test]
    fn test_cross_only_zero_ulp_parity_across_regimes() {
        for &(mono, wide_exp, stride, dim) in &[
            (false, false, 1usize, 64usize),
            (true, false, 2, 128),
            (false, true, 1, 192),
        ] {
            let n = 310usize;
            let mut feats = vec![[0.0_f32; 256]; n];
            let mut targets = vec![[0.0_f64; 2]; n];
            let mut weights = vec![1.0_f64; n];
            for t in 0..n {
                weights[t] = if t == 17 {
                    f64::NAN
                } else {
                    0.4 + 0.02 * ((t % 19) as f64)
                };
                for d in 0..256 {
                    let base = (((t * 17 + d * 11) as f32) * 0.041).sin() * 0.6;
                    feats[t][d] = if wide_exp && d % 7 == 0 && t % 5 == 0 {
                        base * 1.0e-7
                    } else if wide_exp && d % 7 == 0 && t == 40 {
                        1.0e4
                    } else {
                        base
                    };
                }
                let y0 = ((t as f64) * 0.09).cos() + if wide_exp && t == 3 { 1.0e6 } else { 0.0 }
                    - if wide_exp && t == 203 { 1.0e6 } else { 0.0 };
                let y1 = if mono { y0 } else { ((t as f64) * 0.05).sin() };
                targets[t] = [y0, y1];
            }
            let (_, ref_c) = contract_dim_gram_tiled_wide192_with_backend::<ScalarTrainerBackend>(
                &feats, &targets, &weights, stride, dim,
            );
            let fast_c = contract_dim_cross_only_wide192(&feats, &targets, &weights, stride, dim);
            assert_eq!(ref_c.len(), fast_c.len());
            for (r, f) in ref_c.iter().zip(&fast_c) {
                assert_eq!(r[0].to_bits(), f[0].to_bits());
                assert_eq!(r[1].to_bits(), f[1].to_bits());
            }
        }
        assert!(
            contract_dim_cross_only_wide192(&[], &[], &[], 1, 64)
                .iter()
                .all(|&v| v == [0.0, 0.0])
        );
    }
}
