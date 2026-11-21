//! Exact `Wide192Accum` cross-term contraction (`rhs = F^T diag(w) Y`) for arbitrary feature
//! dimensions and prime integral-scaled subsampling strides (`ref[train.gram.wide192]`).

use super::{WIDE_GRAM_BATCH_FRAMES, Wide192Accum};

#[inline]
pub(super) fn integral_stride_and_step(total_frames: usize, stride: usize) -> (usize, f64) {
    let step = stride.max(1);
    if total_frames == 0 || step == 1 {
        return (step, 1.0);
    }
    let sampled_count = total_frames.div_ceil(step);
    (step, (total_frames as f64) / (sampled_count as f64))
}

/// Contracts the weighted feature-target cross-product `rhs[d][ch] = sum_t w[t] * f[t, d] * y[t, ch]`
/// using [`Wide192Accum`] across batched frames so intra-batch and cross-batch cancellations retain
/// exact IEEE-754 accuracy (`ref[train.gram.wide192]`).
#[must_use]
pub fn contract_dim_cross_only_wide192(
    features: &[f32],
    targets: &[[f64; 2]],
    weights: &[f64],
    dim: usize,
    stride: usize,
) -> Vec<[f64; 2]> {
    let mut out = vec![[0.0_f64; 2]; dim];
    if dim == 0 || targets.is_empty() || weights.is_empty() || features.len() < dim {
        return out;
    }
    let avail_frames = (features.len() / dim).min(targets.len()).min(weights.len());
    if avail_frames == 0 {
        return out;
    }
    let (step, stride_scale) = integral_stride_and_step(avail_frames, stride);
    let mut global_acc = vec![[Wide192Accum::zero(); 2]; dim];
    let mut batch_acc = vec![[Wide192Accum::zero(); 2]; dim];

    let mut frame_idx = 0usize;
    let mut in_batch = 0usize;
    while frame_idx < avail_frames {
        let w_eff = weights[frame_idx] * stride_scale;
        if w_eff != 0.0 {
            let row = &features[frame_idx * dim..(frame_idx + 1) * dim];
            let tgt = targets[frame_idx];
            if tgt[0] != 0.0 || tgt[1] != 0.0 {
                for (d, &f_raw) in row.iter().enumerate() {
                    let fd = f64::from(f_raw);
                    if fd != 0.0 {
                        if tgt[0] != 0.0 {
                            batch_acc[d][0].add_triple_product(w_eff, fd, tgt[0]);
                        }
                        if tgt[1] != 0.0 {
                            batch_acc[d][1].add_triple_product(w_eff, fd, tgt[1]);
                        }
                    }
                }
            }
        }
        in_batch += 1;
        if in_batch == WIDE_GRAM_BATCH_FRAMES {
            for d in 0..dim {
                global_acc[d][0].add_wide(&batch_acc[d][0]);
                global_acc[d][1].add_wide(&batch_acc[d][1]);
                batch_acc[d] = [Wide192Accum::zero(); 2];
            }
            in_batch = 0;
        }
        frame_idx += step;
    }
    if in_batch > 0 {
        for d in 0..dim {
            global_acc[d][0].add_wide(&batch_acc[d][0]);
            global_acc[d][1].add_wide(&batch_acc[d][1]);
        }
    }
    for d in 0..dim {
        out[d][0] = global_acc[d][0].to_f64();
        out[d][1] = global_acc[d][1].to_f64();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_only_wide192_cancellation_and_prime_stride() {
        // verify[train.gram.wide192]
        let two_neg60 = 2.0_f64.powi(-60);
        let mut feats = vec![0.0_f32; 3 * 16];
        feats[12] = 1.0;
        feats[16 + 12] = 1.0;
        feats[32 + 12] = 1.0;
        let targets = [[1.0, -1.0], [two_neg60, two_neg60], [-1.0, 1.0]];
        let weights = [1.0, 1.0, 1.0];
        let c = contract_dim_cross_only_wide192(&feats, &targets, &weights, 16, 1);
        assert_eq!(c[12][0].to_bits(), two_neg60.to_bits());
        assert_eq!(c[12][1].to_bits(), two_neg60.to_bits());

        let const_feats = vec![1.0_f32; 100 * 4];
        let ones_tgt = vec![[1.0, -2.0]; 100];
        let unit_w = vec![1.0; 100];
        let cs7 = contract_dim_cross_only_wide192(&const_feats, &ones_tgt, &unit_w, 4, 7);
        assert!((cs7[0][0] - 100.0).abs() <= 1.0e-11);
        assert!((cs7[3][1] + 200.0).abs() <= 1.0e-11);

        assert_eq!(
            contract_dim_cross_only_wide192(&[], &ones_tgt, &unit_w, 0, 1).len(),
            0
        );
    }
}
