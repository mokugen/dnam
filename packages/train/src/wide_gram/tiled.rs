//! L1D register-tiled (`4x4` / `8x4`) `uplo='U'` Gram and RHS contraction dispatched cleanly
//! through [`TrainerMathBackend`] with automatic per-tile [`Wide192Accum`] fallback on extreme
//! scales or intra-batch cancellation (`ref[train.gram.wide192]`).

use super::cross_only::{contract_dim_cross_only_wide192, integral_stride_and_step};
use super::{WIDE_GRAM_BATCH_FRAMES, Wide192Accum, uplo_u_index, uplo_u_packed_len};
use crate::math::{TrainerMathBackend, TrainerTask, dispatch_trainer};

const FAST_DYNAMIC_SPAN_RATIO: f64 = (1_u64 << 16) as f64;
const FAST_ABS_UPPER: f64 = 1.0e40;
const FAST_ABS_LOWER: f64 = 1.0e-40;
const FAST_CANCEL_RATIO: f64 = 4.0;

#[derive(Default)]
struct BlockBatchBuffer {
    f: Vec<[f64; 4]>,
    wf: Vec<[f64; 4]>,
    raw_w: Vec<f64>,
    sum_abs: [f64; 4],
    max_abs: [f64; 4],
    min_nz_abs: [f64; 4],
    has_extreme: bool,
}

impl BlockBatchBuffer {
    fn new(cap: usize) -> Self {
        Self {
            f: Vec::with_capacity(cap),
            wf: Vec::with_capacity(cap),
            raw_w: Vec::with_capacity(cap),
            sum_abs: [0.0; 4],
            max_abs: [0.0; 4],
            min_nz_abs: [f64::INFINITY; 4],
            has_extreme: false,
        }
    }

    fn clear(&mut self) {
        self.f.clear();
        self.wf.clear();
        self.raw_w.clear();
        self.sum_abs = [0.0; 4];
        self.max_abs = [0.0; 4];
        self.min_nz_abs = [f64::INFINITY; 4];
        self.has_extreme = false;
    }

    fn push(&mut self, row: &[f32], col_base: usize, active_cols: usize, w_eff: f64) {
        let mut f4 = [0.0_f64; 4];
        let mut wf4 = [0.0_f64; 4];
        let w_abs = w_eff.abs();
        if !w_eff.is_finite()
            || (w_abs > 0.0 && !(FAST_ABS_LOWER..=FAST_ABS_UPPER).contains(&w_abs))
        {
            self.has_extreme = true;
        }
        for c in 0..active_cols {
            let fv = f64::from(row[col_base + c]);
            let fa = fv.abs();
            f4[c] = fv;
            wf4[c] = w_eff * fv;
            if !fv.is_finite() {
                self.has_extreme = true;
            } else if fa > 0.0 {
                let wfa = wf4[c].abs();
                if !(FAST_ABS_LOWER..=FAST_ABS_UPPER).contains(&fa)
                    || !(FAST_ABS_LOWER..=FAST_ABS_UPPER).contains(&wfa)
                {
                    self.has_extreme = true;
                }
                self.sum_abs[c] += fa;
                if fa > self.max_abs[c] {
                    self.max_abs[c] = fa;
                }
                if fa < self.min_nz_abs[c] {
                    self.min_nz_abs[c] = fa;
                }
            }
        }
        self.f.push(f4);
        self.wf.push(wf4);
        self.raw_w.push(w_eff);
    }

    fn needs_wide_column(&self, c: usize) -> bool {
        self.has_extreme
            || (self.max_abs[c] > 0.0
                && self.max_abs[c] > self.min_nz_abs[c] * FAST_DYNAMIC_SPAN_RATIO)
    }
}

struct WideGramScratch {
    blocks: Vec<BlockBatchBuffer>,
    gram: Vec<Wide192Accum>,
}

impl WideGramScratch {
    fn new(dim: usize) -> Self {
        let num_blocks = dim.div_ceil(4);
        let blocks = (0..num_blocks)
            .map(|_| BlockBatchBuffer::new(WIDE_GRAM_BATCH_FRAMES))
            .collect();
        let gram = vec![Wide192Accum::zero(); uplo_u_packed_len(dim)];
        Self { blocks, gram }
    }

    fn clear_batch(&mut self) {
        for blk in &mut self.blocks {
            blk.clear();
        }
    }
}

#[inline]
fn accumulate_wide_block_pair(
    gram: &mut [Wide192Accum],
    b_i: &BlockBatchBuffer,
    b_j: &BlockBatchBuffer,
    base_i: usize,
    base_j: usize,
    dim: usize,
) {
    let m = b_i.f.len().min(b_j.f.len());
    let ci_max = (dim - base_i).min(4);
    let cj_max = (dim - base_j).min(4);
    for t in 0..m {
        let w = b_j.raw_w[t];
        if w == 0.0 {
            continue;
        }
        for lj in 0..cj_max {
            let fj = b_j.f[t][lj];
            if fj == 0.0 {
                continue;
            }
            let gj = base_j + lj;
            for li in 0..ci_max {
                let gi = base_i + li;
                if gi <= gj && b_i.f[t][li] != 0.0 {
                    gram[uplo_u_index(gi, gj)].add_triple_product(w, b_i.f[t][li], fj);
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
struct TileCellCoord {
    gi: usize,
    gj: usize,
    li: usize,
    lj: usize,
}

#[inline]
fn merge_tile_cell(
    gram: &mut [Wide192Accum],
    b_i: &BlockBatchBuffer,
    b_j: &BlockBatchBuffer,
    coord: TileCellCoord,
    tile_val: f64,
) {
    let m = b_i.f.len().min(b_j.f.len());
    let is_wide_inputs = b_i.needs_wide_column(coord.li) || b_j.needs_wide_column(coord.lj);
    let mut sum_abs_prod = 0.0_f64;
    let needs_check = is_wide_inputs || !tile_val.is_finite();
    if !needs_check {
        for t in 0..m {
            sum_abs_prod += (b_i.f[t][coord.li] * b_j.wf[t][coord.lj]).abs();
        }
    }
    if needs_check || sum_abs_prod > tile_val.abs() * FAST_CANCEL_RATIO {
        let idx = uplo_u_index(coord.gi, coord.gj);
        for t in 0..m {
            gram[idx].add_triple_product(b_j.raw_w[t], b_i.f[t][coord.li], b_j.f[t][coord.lj]);
        }
    } else {
        gram[uplo_u_index(coord.gi, coord.gj)].add_f64(tile_val);
    }
}

#[inline]
fn merge_4x4_tile_to_uplo(
    gram: &mut [Wide192Accum],
    b_i: &BlockBatchBuffer,
    b_j: &BlockBatchBuffer,
    base_i: usize,
    base_j: usize,
    dim: usize,
    tile: &[[f64; 4]; 4],
) {
    let ci_max = (dim - base_i).min(4);
    let cj_max = (dim - base_j).min(4);
    for lj in 0..cj_max {
        let gj = base_j + lj;
        for li in 0..ci_max {
            let gi = base_i + li;
            if gi <= gj {
                merge_tile_cell(
                    gram,
                    b_i,
                    b_j,
                    TileCellCoord { gi, gj, li, lj },
                    tile[lj][li],
                );
            }
        }
    }
}

fn flush_batch_tiled<B: TrainerMathBackend>(scratch: &mut WideGramScratch, dim: usize) {
    let num_blocks = scratch.blocks.len();
    for bj in 0..num_blocks {
        let base_j = bj * 4;
        let b_j = &scratch.blocks[bj];
        if b_j.has_extreme {
            for bi in 0..=bj {
                let b_i = &scratch.blocks[bi];
                accumulate_wide_block_pair(&mut scratch.gram, b_i, b_j, bi * 4, base_j, dim);
            }
            continue;
        }
        let mut bi = 0usize;
        while bi < bj {
            let b_i0 = &scratch.blocks[bi];
            let b_i1 = &scratch.blocks[bi + 1];
            if b_i0.has_extreme || b_i1.has_extreme {
                accumulate_wide_block_pair(&mut scratch.gram, b_i0, b_j, bi * 4, base_j, dim);
                accumulate_wide_block_pair(&mut scratch.gram, b_i1, b_j, (bi + 1) * 4, base_j, dim);
            } else {
                let (t0, t1) = B::contract_tile_8x4_ieee_f64(&b_i0.f, &b_i1.f, &b_j.wf);
                merge_4x4_tile_to_uplo(&mut scratch.gram, b_i0, b_j, bi * 4, base_j, dim, &t0);
                merge_4x4_tile_to_uplo(
                    &mut scratch.gram,
                    b_i1,
                    b_j,
                    (bi + 1) * 4,
                    base_j,
                    dim,
                    &t1,
                );
            }
            bi += 2;
        }
        while bi <= bj {
            let b_i = &scratch.blocks[bi];
            if b_i.has_extreme {
                accumulate_wide_block_pair(&mut scratch.gram, b_i, b_j, bi * 4, base_j, dim);
            } else {
                let tile = B::contract_tile_4x4_ieee_f64(&b_i.f, &b_j.wf);
                merge_4x4_tile_to_uplo(&mut scratch.gram, b_i, b_j, bi * 4, base_j, dim, &tile);
            }
            bi += 1;
        }
    }
    scratch.clear_batch();
}

/// Contracts the packed `uplo='U'` Gram matrix `G = F^T diag(w) F` and 2-channel cross RHS
/// `c = F^T diag(w) Y` using register-tiled SIMD contractions and [`Wide192Accum`] with an
/// explicit [`TrainerMathBackend`] (`ref[train.gram.wide192]`).
#[must_use]
pub fn contract_dim_gram_tiled_wide192_with_backend<B: TrainerMathBackend>(
    features: &[f32],
    targets: &[[f64; 2]],
    weights: &[f64],
    dim: usize,
    stride: usize,
) -> (Vec<f64>, Vec<[f64; 2]>) {
    let packed_len = uplo_u_packed_len(dim);
    let rhs = contract_dim_cross_only_wide192(features, targets, weights, dim, stride);
    if dim == 0 || targets.is_empty() || weights.is_empty() || features.len() < dim {
        return (vec![0.0_f64; packed_len], rhs);
    }
    let avail_frames = (features.len() / dim).min(targets.len()).min(weights.len());
    if avail_frames == 0 {
        return (vec![0.0_f64; packed_len], rhs);
    }
    let (step, stride_scale) = integral_stride_and_step(avail_frames, stride);
    let mut scratch = WideGramScratch::new(dim);
    let num_blocks = scratch.blocks.len();

    let mut frame_idx = 0usize;
    let mut in_batch = 0usize;
    while frame_idx < avail_frames {
        let w_eff = weights[frame_idx] * stride_scale;
        let row = &features[frame_idx * dim..(frame_idx + 1) * dim];
        for blk_idx in 0..num_blocks {
            let base = blk_idx * 4;
            let active_cols = (dim - base).min(4);
            scratch.blocks[blk_idx].push(row, base, active_cols, w_eff);
        }
        in_batch += 1;
        if in_batch == WIDE_GRAM_BATCH_FRAMES {
            flush_batch_tiled::<B>(&mut scratch, dim);
            in_batch = 0;
        }
        frame_idx += step;
    }
    if in_batch > 0 {
        flush_batch_tiled::<B>(&mut scratch, dim);
    }

    let gram_out = scratch.gram.iter().map(Wide192Accum::to_f64).collect();
    (gram_out, rhs)
}

/// Contracts the packed `uplo='U'` Gram matrix `G = F^T diag(w) F` and 2-channel cross RHS
/// `c = F^T diag(w) Y` using runtime-detected host SIMD (`DynamicTrainer::detect()`).
#[must_use]
pub fn contract_dim_gram_tiled_wide192(
    features: &[f32],
    targets: &[[f64; 2]],
    weights: &[f64],
    dim: usize,
    stride: usize,
) -> (Vec<f64>, Vec<[f64; 2]>) {
    struct TiledTask<'a> {
        features: &'a [f32],
        targets: &'a [[f64; 2]],
        weights: &'a [f64],
        dim: usize,
        stride: usize,
    }
    impl TrainerTask<(Vec<f64>, Vec<[f64; 2]>)> for TiledTask<'_> {
        fn run<B: TrainerMathBackend>(self) -> (Vec<f64>, Vec<[f64; 2]>) {
            contract_dim_gram_tiled_wide192_with_backend::<B>(
                self.features,
                self.targets,
                self.weights,
                self.dim,
                self.stride,
            )
        }
    }
    dispatch_trainer(TiledTask {
        features,
        targets,
        weights,
        dim,
        stride,
    })
}

/// Convenience wrapper for `[f32; 256]` stacked feature frames (`ref[train.gram.wide192]`).
#[must_use]
pub fn contract_256d_gram_tiled_wide192(
    feats: &[[f32; 256]],
    targets: &[[f64; 2]],
    weights: &[f64],
    stride: usize,
) -> (Vec<f64>, Vec<[f64; 2]>) {
    let flat: Vec<f32> = feats.iter().flat_map(|row| row.iter().copied()).collect();
    contract_dim_gram_tiled_wide192(&flat, targets, weights, 256, stride)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;

    #[test]
    fn test_tiled_wide192_edge_cancellations_extreme_scales_and_backend_parity() {
        // verify[train.gram.wide192]
        let tiny = 2.0_f32.powi(-30);
        let two_neg60 = 2.0_f64.powi(-60);
        let mut feats = vec![[0.0_f32; 256]; 8];
        let mut targets = vec![[0.0_f64; 2]; 8];
        let weights = vec![1.0_f64; 8];
        feats[0][0] = 1.0;
        feats[0][7] = 1.0;
        feats[0][1] = 1.0;
        feats[0][2] = 1.0;
        feats[0][3] = 2.0;
        feats[0][6] = 3.0;
        feats[0][12] = 1.0;
        targets[0] = [1.0, -1.0];

        feats[1][0] = tiny;
        feats[1][7] = tiny;
        feats[1][1] = tiny;
        feats[1][2] = tiny;
        feats[1][12] = 1.0;
        targets[1] = [two_neg60, two_neg60];

        feats[2][0] = -1.0;
        feats[2][7] = 1.0;
        feats[2][1] = -1.0;
        feats[2][2] = 1.0;
        feats[2][12] = 1.0;
        targets[2] = [-1.0, 1.0];

        let (g, c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
        assert_eq!(g[uplo_u_index(0, 7)].to_bits(), two_neg60.to_bits());
        assert_eq!(g[uplo_u_index(1, 2)].to_bits(), two_neg60.to_bits());
        assert_eq!(g[uplo_u_index(3, 3)], 4.0);
        assert_eq!(g[uplo_u_index(3, 6)], 6.0);
        assert_eq!(c[12][0].to_bits(), two_neg60.to_bits());
        assert_eq!(c[12][1].to_bits(), two_neg60.to_bits());

        // Cross-batch cancellation + partial intra-batch cancellation across 520 frames.
        let mut adv = vec![0.0_f32; 520 * 16];
        let adv_t = vec![[0.0_f64; 2]; 520];
        let adv_w = vec![1.0_f64; 520];
        adv[0] = 1.0;
        adv[7] = 1.0;
        adv[16] = 2.0_f32.powi(-28);
        adv[16 + 7] = 2.0_f32.powi(-28);
        adv[32] = -1.0;
        adv[32 + 7] = 1.0;
        adv[48] = 2.0_f32.powi(-9);
        adv[48 + 7] = 2.0_f32.powi(-9);
        adv[4] = 1.0;
        adv[11] = 1.0;
        adv[256 * 16 + 4] = tiny;
        adv[256 * 16 + 11] = tiny;
        adv[512 * 16 + 4] = -1.0;
        adv[512 * 16 + 11] = 1.0;

        let (g_adv, _) = contract_dim_gram_tiled_wide192(&adv, &adv_t, &adv_w, 16, 1);
        let (g_scl, _) = contract_dim_gram_tiled_wide192_with_backend::<ScalarBackend>(
            &adv, &adv_t, &adv_w, 16, 1,
        );
        let expected_partial = 2.0_f64.powi(-18) + 2.0_f64.powi(-56);
        assert_eq!(
            g_adv[uplo_u_index(0, 7)].to_bits(),
            expected_partial.to_bits()
        );
        assert_eq!(g_adv[uplo_u_index(4, 11)].to_bits(), two_neg60.to_bits());
        assert_eq!(
            g_adv[uplo_u_index(0, 7)].to_bits(),
            g_scl[uplo_u_index(0, 7)].to_bits()
        );

        // Extreme multi-scale weights without overflow/underflow
        let mut fb = [0.0_f32; 8];
        fb[0] = 1.0e35;
        fb[3] = 1.0e-35;
        fb[1] = 1.0e-35;
        fb[4] = 1.0e35;
        let feats_ext = [fb, fb].concat();
        let (g_ext, _) =
            contract_dim_gram_tiled_wide192(&feats_ext, &[[0.0; 2]; 2], &[1.0e-300, 1.0e300], 8, 1);
        let expected_hi = 1.0e300_f64 * (f64::from(1.0e-35_f32) * f64::from(1.0e35_f32));
        assert!(
            g_ext[uplo_u_index(1, 4)].is_finite()
                && (g_ext[uplo_u_index(1, 4)] - expected_hi).abs() <= 1.0e285
        );

        let const_feats = vec![1.0_f32; 100 * 6];
        let (g_s7, _) = contract_dim_gram_tiled_wide192(
            &const_feats,
            &vec![[0.0; 2]; 100],
            &vec![1.0; 100],
            6,
            7,
        );
        assert!((g_s7[uplo_u_index(0, 0)] - 100.0).abs() <= 1.0e-11);

        // NaN entry flags has_extreme via wide path and skips non-finite product safely.
        let mut nan_feats = vec![1.0_f32; 8];
        nan_feats[2] = f32::NAN;
        let (g_nan, _) =
            contract_dim_gram_tiled_wide192(&nan_feats, &[[0.0; 2]; 2], &[1.0, 1.0], 4, 1);
        assert!((g_nan[uplo_u_index(0, 0)] - 2.0).abs() <= 1.0e-12);
        assert_eq!(g_nan[uplo_u_index(2, 2)], 1.0);
    }
}
