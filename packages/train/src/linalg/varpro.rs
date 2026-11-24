//! Ozaki / `Wide192Accum` normal-equation assembly and nested 4-tier Schur-complement prefix
//! VarPro re-balancing (`ref[train.pipeline.nested-varpro-schur]`, `ref[train.gram.wide192]`).

use dnam_kernels::pow2_ceil_fp32;

use crate::error::TrainError;
use crate::runner::TrainRunner;
use crate::wide_gram::{
    Wide192Accum, contract_dim_gram_tiled_wide192, uplo_u_index, uplo_u_packed_len,
};

use super::cholesky::{factorize_uplo_u_cholesky_exact_no_ridge, solve_factorized_uplo_u_2ch};

const OZAKI_MAX_FEATURE_DIM: usize = 1024;

fn assemble_via_wide192(
    features_col: &[f32],
    targets_col: &[f32],
    num_frames: usize,
    feature_dim: usize,
    target_channels: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut feat_row = vec![0.0_f32; num_frames * feature_dim];
    for c in 0..feature_dim {
        let col_slice = &features_col[c * num_frames..(c + 1) * num_frames];
        for t in 0..num_frames {
            feat_row[t * feature_dim + c] = col_slice[t];
        }
    }
    let weights = vec![1.0_f64; num_frames];
    let tgt_pair: Vec<[f64; 2]> = (0..num_frames)
        .map(|t| {
            let y0 = if target_channels > 0 {
                f64::from(targets_col[t])
            } else {
                0.0
            };
            let y1 = if target_channels > 1 {
                f64::from(targets_col[num_frames + t])
            } else {
                0.0
            };
            [y0, y1]
        })
        .collect();
    let (gram, rhs_pair) =
        contract_dim_gram_tiled_wide192(&feat_row, &tgt_pair, &weights, feature_dim, 1);

    let mut rhs = vec![vec![0.0_f64; target_channels]; feature_dim];
    for col in 0..feature_dim {
        for ch in 0..target_channels.min(2) {
            rhs[col][ch] = rhs_pair[col][ch];
        }
        for ch in 2..target_channels {
            let mut acc = Wide192Accum::zero();
            let f_col = &features_col[col * num_frames..(col + 1) * num_frames];
            let y_col = &targets_col[ch * num_frames..(ch + 1) * num_frames];
            for (&fv, &yv) in f_col.iter().zip(y_col) {
                acc.add_product(f64::from(fv), f64::from(yv));
            }
            rhs[col][ch] = acc.to_f64();
        }
    }
    (gram, rhs)
}

/// Assembles `uplo='U'` Gram `G = F^T F` and `RHS = F^T Y` in IEEE-754 `f64` from column-major
/// `f32` matrices via hardware `ozaki_gram_rhs_i32` (`N <= 1024`) with transparent chunked
/// `Wide192Accum` combination (`ref[train.gram.wide192]`, `ref[train.pipeline.backbone-trajectory-opt]`).
pub fn assemble_ozaki_normal_equations(
    runner: &TrainRunner,
    features_col_major: &[f32],
    targets_col_major: &[f32],
    num_frames: usize,
    feature_dim: usize,
    target_channels: usize,
    force_wide192: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
    if num_frames == 0 || feature_dim == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "assemble_ozaki_normal_equations",
            expected: 1,
            actual: 0,
        });
    }
    if features_col_major.len() < feature_dim * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: "assemble_ozaki_normal_equations(features)",
            expected: feature_dim * num_frames,
            actual: features_col_major.len(),
        });
    }
    if target_channels > 0 && targets_col_major.len() < target_channels * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: "assemble_ozaki_normal_equations(targets)",
            expected: target_channels * num_frames,
            actual: targets_col_major.len(),
        });
    }
    if !features_col_major[..feature_dim * num_frames]
        .iter()
        .all(|v| v.is_finite())
        || (target_channels > 0
            && !targets_col_major[..target_channels * num_frames]
                .iter()
                .all(|v| v.is_finite()))
    {
        return Err(TrainError::NonFiniteInput {
            context: "assemble_ozaki_normal_equations",
        });
    }

    let can_use_ozaki = !force_wide192 && feature_dim <= OZAKI_MAX_FEATURE_DIM;

    if !can_use_ozaki {
        return Ok(assemble_via_wide192(
            features_col_major,
            targets_col_major,
            num_frames,
            feature_dim,
            target_channels,
        ));
    }

    let mut col_scales = vec![[0.0_f32; 2]; feature_dim + target_channels];
    for c in 0..feature_dim {
        let col = &features_col_major[c * num_frames..(c + 1) * num_frames];
        let max_abs = col.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let (s, inv) = pow2_ceil_fp32(max_abs);
        col_scales[c] = [s, inv];
    }
    for ch in 0..target_channels {
        let col = &targets_col_major[ch * num_frames..(ch + 1) * num_frames];
        let max_abs = col.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let (s, inv) = pow2_ceil_fp32(max_abs);
        col_scales[feature_dim + ch] = [s, inv];
    }

    let split64 = runner.ozaki_gram_rhs_split64(
        features_col_major,
        targets_col_major,
        num_frames,
        feature_dim,
        target_channels,
        &col_scales,
    )?;
    runner.reconstruct_ozaki_split64_f64(&split64, feature_dim, target_channels, &col_scales)
}

/// Output of nested 4-tier Schur-complement prefix VarPro (`ref[train.pipeline.nested-varpro-schur]`).
#[derive(Clone, Debug, PartialEq)]
pub struct NestedTierPrefixResult {
    /// Joint optimal stereo weights for each nested prefix dimension `tier_dims[t]`.
    pub joint_prefix_weights: [Vec<[f64; 2]>; 4],
    /// Prefix-frozen incremental weights extending `tier_dims[t-1]` via exact Schur complement `S_t`.
    pub prefix_frozen_weights: [Vec<[f64; 2]>; 4],
    /// Quadratic VarPro objective reduction `0.5 * w^T G w - c^T w` across incremental tiers (`<= prev`).
    pub incremental_objectives: [[f64; 2]; 4],
    /// Quadratic VarPro objective reduction across joint re-balanced tiers (`<= incremental`).
    pub joint_objectives: [[f64; 2]; 4],
    /// `true` on `t >= 1` when `max |w_joint[..d_{t-1}] - w_prev| > prefix_override_eps`.
    pub prefix_override_needed: [bool; 4],
}

fn eval_quad_obj_2ch(w: &[[f64; 2]], rhs: &[[f64; 2]], gram: &[f64]) -> [f64; 2] {
    let dim = w.len();
    let mut lin = [0.0_f64; 2];
    let mut quad = [0.0_f64; 2];
    for j in 0..dim {
        lin[0] += rhs[j][0] * w[j][0];
        lin[1] += rhs[j][1] * w[j][1];
        let col_j = uplo_u_index(0, j);
        for i in 0..j {
            let g_ij = gram[col_j + i];
            quad[0] += 2.0 * g_ij * w[i][0] * w[j][0];
            quad[1] += 2.0 * g_ij * w[i][1] * w[j][1];
        }
        let g_jj = gram[col_j + j];
        quad[0] += g_jj * w[j][0] * w[j][0];
        quad[1] += g_jj * w[j][1] * w[j][1];
    }
    [0.5 * quad[0] - lin[0], 0.5 * quad[1] - lin[1]]
}

fn solve_prefix_gram_2ch_exact(
    gram_full: &[f64],
    rhs_full: &[[f64; 2]],
    dim: usize,
) -> Result<(Vec<[f64; 2]>, Vec<f64>), TrainError> {
    let sub_len = uplo_u_packed_len(dim);
    let r_upper = factorize_uplo_u_cholesky_exact_no_ridge(dim, &gram_full[..sub_len])?;
    let w = solve_factorized_uplo_u_2ch(dim, &r_upper, &rhs_full[..dim])?;
    Ok((w, r_upper))
}

fn solve_lower_transpose_r(dim: usize, r_upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; dim];
    for i in 0..dim {
        let col_i = uplo_u_index(0, i);
        let mut sum = rhs[i];
        for (&r_ki, &xk) in r_upper[col_i..col_i + i].iter().zip(&out[..i]) {
            sum -= r_ki * xk;
        }
        out[i] = sum / r_upper[col_i + i];
    }
    out
}

fn solve_upper_r(dim: usize, r_upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut out = rhs.to_vec();
    for k in (0..dim).rev() {
        let col_k = uplo_u_index(0, k);
        let xk = out[k] / r_upper[col_k + k];
        out[k] = xk;
        for (xi, &r_ik) in out[..k].iter_mut().zip(&r_upper[col_k..col_k + k]) {
            *xi -= r_ik * xk;
        }
    }
    out
}

/// Solves nested 4-tier prefix systems with exact Schur-complement re-balancing across
/// `Lite -> Full -> Pro -> Ultra` (`ref[train.pipeline.nested-varpro-schur]`).
pub fn solve_nested_tier_prefix_schur(
    gram_uplo: &[f64],
    rhs_l: &[f64],
    rhs_r: &[f64],
    tier_dims: [usize; 4],
    base_ridge: f64,
    prefix_override_eps: f64,
) -> Result<NestedTierPrefixResult, TrainError> {
    let total_dim = tier_dims[3];
    if tier_dims[0] == 0
        || tier_dims[0] > tier_dims[1]
        || tier_dims[1] > tier_dims[2]
        || tier_dims[2] > total_dim
    {
        return Err(TrainError::InvalidDimensions {
            context: "solve_nested_tier_prefix_schur(tier_dims)",
            expected: total_dim,
            actual: tier_dims[0],
        });
    }
    if rhs_l.len() != total_dim || rhs_r.len() != total_dim {
        return Err(TrainError::InvalidDimensions {
            context: "solve_nested_tier_prefix_schur(rhs)",
            expected: total_dim,
            actual: rhs_l.len().min(rhs_r.len()),
        });
    }
    if gram_uplo.len() != uplo_u_packed_len(total_dim) {
        return Err(TrainError::InvalidDimensions {
            context: "solve_nested_tier_prefix_schur(gram_uplo)",
            expected: uplo_u_packed_len(total_dim),
            actual: gram_uplo.len(),
        });
    }
    if !base_ridge.is_finite()
        || base_ridge < 0.0
        || !prefix_override_eps.is_finite()
        || prefix_override_eps < 0.0
        || !gram_uplo.iter().all(|v| v.is_finite())
        || !rhs_l.iter().all(|v| v.is_finite())
        || !rhs_r.iter().all(|v| v.is_finite())
    {
        return Err(TrainError::NonFiniteInput {
            context: "solve_nested_tier_prefix_schur",
        });
    }

    let ridge = base_ridge.max(1.0e-12);
    let mut reg_gram = gram_uplo.to_vec();
    for j in 0..total_dim {
        reg_gram[uplo_u_index(j, j)] += ridge;
    }
    let rhs_lr: Vec<[f64; 2]> = rhs_l.iter().zip(rhs_r).map(|(&l, &r)| [l, r]).collect();

    let mut joint_weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut frozen_weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut joint_obj = [[0.0_f64; 2]; 4];
    let mut frozen_obj = [[0.0_f64; 2]; 4];
    let mut prefix_override_needed = [false; 4];

    let (w0, mut prev_r_upper) = solve_prefix_gram_2ch_exact(&reg_gram, &rhs_lr, tier_dims[0])?;
    joint_obj[0] = eval_quad_obj_2ch(
        &w0,
        &rhs_lr[..tier_dims[0]],
        &reg_gram[..uplo_u_packed_len(tier_dims[0])],
    );
    frozen_obj[0] = joint_obj[0];
    joint_weights[0] = w0.clone();
    frozen_weights[0] = w0;

    for t in 1..4 {
        let d_prev = tier_dims[t - 1];
        let d_cur = tier_dims[t];
        let delta = d_cur - d_prev;
        let sub_gram_len = uplo_u_packed_len(d_cur);

        if delta == 0 {
            joint_weights[t] = joint_weights[t - 1].clone();
            frozen_weights[t] = frozen_weights[t - 1].clone();
            joint_obj[t] = joint_obj[t - 1];
            frozen_obj[t] = frozen_obj[t - 1];
            continue;
        }

        let mut g12_cols = vec![vec![0.0_f64; d_prev]; delta];
        let mut r12_cols = vec![vec![0.0_f64; d_prev]; delta];
        let mut m_cols = vec![vec![0.0_f64; d_prev]; delta];
        for b in 0..delta {
            let j = d_prev + b;
            let col_j = uplo_u_index(0, j);
            g12_cols[b].copy_from_slice(&reg_gram[col_j..col_j + d_prev]);
            r12_cols[b] = solve_lower_transpose_r(d_prev, &prev_r_upper, &g12_cols[b]);
            m_cols[b] = solve_upper_r(d_prev, &prev_r_upper, &r12_cols[b]);
        }

        // Joint solution via exact Schur complement elimination S_t = G22 - R12^T R12:
        // S_t * w2_joint = c2 - G12^T * w_{joint, prev}, w1_joint = w_{joint, prev} - M * w2_joint.
        let mut g22_uplo = vec![0.0_f64; uplo_u_packed_len(delta)];
        let mut schur_uplo = vec![0.0_f64; uplo_u_packed_len(delta)];
        for b in 0..delta {
            let jb = d_prev + b;
            let col_jb = uplo_u_index(0, jb);
            for a in 0..=b {
                let g22_ab = reg_gram[col_jb + d_prev + a];
                g22_uplo[uplo_u_index(a, b)] = g22_ab;
                let dot: f64 = r12_cols[a]
                    .iter()
                    .zip(&r12_cols[b])
                    .map(|(&ra, &rb)| ra * rb)
                    .sum();
                schur_uplo[uplo_u_index(a, b)] = g22_ab - dot;
            }
        }
        let g22_r = factorize_uplo_u_cholesky_exact_no_ridge(delta, &g22_uplo)?;
        let schur_r = factorize_uplo_u_cholesky_exact_no_ridge(delta, &schur_uplo)?;

        // Prefix-frozen solution: conditional optimum over block 2 with G22.
        let mut rem_frozen = vec![[0.0_f64; 2]; delta];
        for b in 0..delta {
            let mut sum0 = rhs_lr[d_prev + b][0];
            let mut sum1 = rhs_lr[d_prev + b][1];
            for (i, &g) in g12_cols[b].iter().enumerate() {
                sum0 -= g * frozen_weights[t - 1][i][0];
                sum1 -= g * frozen_weights[t - 1][i][1];
            }
            rem_frozen[b] = [sum0, sum1];
        }
        let w2_frozen = solve_factorized_uplo_u_2ch(delta, &g22_r, &rem_frozen)?;
        let mut w_fr = Vec::with_capacity(d_cur);
        w_fr.extend_from_slice(&frozen_weights[t - 1]);
        w_fr.extend_from_slice(&w2_frozen);
        frozen_obj[t] = eval_quad_obj_2ch(&w_fr, &rhs_lr[..d_cur], &reg_gram[..sub_gram_len]);
        frozen_weights[t] = w_fr;

        // Joint Schur-complement update using Schur RHS r_schur = c2 - G12^T * w_{joint, prev}.
        let mut rem_joint = vec![[0.0_f64; 2]; delta];
        for b in 0..delta {
            let mut sum0 = rhs_lr[d_prev + b][0];
            let mut sum1 = rhs_lr[d_prev + b][1];
            for (i, &g) in g12_cols[b].iter().enumerate() {
                sum0 -= g * joint_weights[t - 1][i][0];
                sum1 -= g * joint_weights[t - 1][i][1];
            }
            rem_joint[b] = [sum0, sum1];
        }
        let w2_joint = solve_factorized_uplo_u_2ch(delta, &schur_r, &rem_joint)?;
        let mut w_jt = Vec::with_capacity(d_cur);
        for i in 0..d_prev {
            let mut w1_0 = joint_weights[t - 1][i][0];
            let mut w1_1 = joint_weights[t - 1][i][1];
            for (b, m_col) in m_cols.iter().enumerate() {
                w1_0 -= m_col[i] * w2_joint[b][0];
                w1_1 -= m_col[i] * w2_joint[b][1];
            }
            w_jt.push([w1_0, w1_1]);
        }
        w_jt.extend_from_slice(&w2_joint);

        // Assemble R_cur = [[R_prev, R_12], [0, S_chol]] in uplo='U' without re-factorizing.
        let mut r_cur = vec![0.0_f64; sub_gram_len];
        let prev_len = prev_r_upper.len();
        r_cur[..prev_len].copy_from_slice(&prev_r_upper);
        for b in 0..delta {
            let j = d_prev + b;
            let dst_col = uplo_u_index(0, j);
            r_cur[dst_col..dst_col + d_prev].copy_from_slice(&r12_cols[b]);
            let src_col = uplo_u_index(0, b);
            r_cur[dst_col + d_prev..=dst_col + j].copy_from_slice(&schur_r[src_col..=src_col + b]);
        }
        joint_obj[t] = eval_quad_obj_2ch(&w_jt, &rhs_lr[..d_cur], &reg_gram[..sub_gram_len]);
        let max_diff = w_jt[..d_prev]
            .iter()
            .zip(&joint_weights[t - 1])
            .fold(0.0_f64, |m, (a, b)| {
                m.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs())
            });
        prefix_override_needed[t] = max_diff > prefix_override_eps;
        joint_weights[t] = w_jt;
        prev_r_upper = r_cur;
    }

    Ok(NestedTierPrefixResult {
        joint_prefix_weights: joint_weights,
        prefix_frozen_weights: frozen_weights,
        incremental_objectives: frozen_obj,
        joint_objectives: joint_obj,
        prefix_override_needed,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_ozaki_vs_wide192_normal_equation_assembly() {
        let runner = TrainRunner::cpu();
        let frames = 2_200_usize;
        let dim = 4_usize;
        let tc = 3_usize;
        let mut feats = vec![0.0_f32; dim * frames];
        let mut tgts = vec![0.0_f32; tc * frames];
        for c in 0..dim {
            for t in 0..frames {
                feats[c * frames + t] = ((t as f32) * 0.041 * (1 + c) as f32).sin() * 0.8;
            }
        }
        for ch in 0..tc {
            for t in 0..frames {
                tgts[ch * frames + t] =
                    0.5 * feats[t] - 0.3 * feats[frames + t] + 0.05 * (ch as f32);
            }
        }
        let (g_oz, r_oz) =
            assemble_ozaki_normal_equations(&runner, &feats, &tgts, frames, dim, tc, false)
                .unwrap();
        let (g_w, r_w) =
            assemble_ozaki_normal_equations(&runner, &feats, &tgts, frames, dim, tc, true).unwrap();
        for (a, b) in g_oz.iter().zip(&g_w) {
            assert!((a - b).abs() <= 1.0e-5 * a.abs().max(1.0));
        }
        for (ra, rb) in r_oz.iter().zip(&r_w) {
            for (a, b) in ra.iter().zip(rb) {
                assert!((a - b).abs() <= 1.0e-5 * a.abs().max(1.0));
            }
        }
        let mut nan_feats = feats;
        nan_feats[7] = f32::NAN;
        assert!(matches!(
            assemble_ozaki_normal_equations(&runner, &nan_feats, &tgts, frames, dim, tc, false),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            assemble_ozaki_normal_equations(&runner, &nan_feats, &tgts, frames, dim, tc, true),
            Err(TrainError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn test_schur_matches_direct_single_ridge_cholesky_without_double_perturbation() {
        let dim = 6_usize;
        let mut gram = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut rhs_l = vec![0.0_f64; dim];
        let mut rhs_r = vec![0.0_f64; dim];
        for t in 0..32 {
            let row = [
                ((t as f64) * 0.21).sin(),
                ((t as f64) * 0.37).cos(),
                ((t as f64) * 0.53).sin() * 0.5,
                ((t as f64) * 0.71).cos() * 0.8,
                ((t as f64) * 0.13).sin() + 0.2,
                ((t as f64) * 0.89).cos() * 0.6,
            ];
            let yl = 0.4 * row[0] - 0.3 * row[2] + 0.5 * row[4];
            let yr = -0.2 * row[1] + 0.7 * row[3] - 0.1 * row[5];
            for j in 0..dim {
                rhs_l[j] += row[j] * yl;
                rhs_r[j] += row[j] * yr;
                for i in 0..=j {
                    gram[uplo_u_index(i, j)] += row[i] * row[j];
                }
            }
        }
        let ridge = 1.0e-7_f64;
        let schur =
            solve_nested_tier_prefix_schur(&gram, &rhs_l, &rhs_r, [2, 3, 5, 6], ridge, 1.0e-9)
                .unwrap();
        let mut reg_gram = gram;
        for j in 0..dim {
            reg_gram[uplo_u_index(j, j)] += ridge;
        }
        let direct_r = factorize_uplo_u_cholesky_exact_no_ridge(dim, &reg_gram).unwrap();
        let rhs_lr: Vec<[f64; 2]> = rhs_l.iter().zip(&rhs_r).map(|(&l, &r)| [l, r]).collect();
        let direct_w = solve_factorized_uplo_u_2ch(dim, &direct_r, &rhs_lr).unwrap();
        for (a, b) in schur.joint_prefix_weights[3].iter().zip(&direct_w) {
            assert!((a[0] - b[0]).abs() <= 1.0e-11);
            assert!((a[1] - b[1]).abs() <= 1.0e-11);
        }
    }

    proptest! {
        #[test]
        fn prop_schur_prefix_objectives_decrease_monotonically(
            seed in 1_u64..1_000,
        ) {
            let dim = 8_usize;
            let n = 48_usize;
            let mut gram = vec![0.0_f64; uplo_u_packed_len(dim)];
            let mut rhs_l = vec![0.0_f64; dim];
            let mut rhs_r = vec![0.0_f64; dim];
            for t in 0..n {
                let mut row = [0.0_f64; 8];
                for (c, slot) in row.iter_mut().enumerate() {
                    let ang = (t as f64) * 0.17 + (c as f64) * 0.63 + (seed as f64) * 0.01;
                    *slot = ang.sin() + 0.35 * (2.3 * ang).cos();
                }
                let yl = 0.8 * row[0] - 0.5 * row[2] + 0.4 * row[5] - 0.25 * row[7];
                let yr = -0.3 * row[1] + 0.7 * row[3] + 0.3 * row[6];
                for j in 0..dim {
                    rhs_l[j] += row[j] * yl;
                    rhs_r[j] += row[j] * yr;
                    for i in 0..=j {
                        gram[uplo_u_index(i, j)] += row[i] * row[j];
                    }
                }
            }
            let res = solve_nested_tier_prefix_schur(
                &gram,
                &rhs_l,
                &rhs_r,
                [2, 4, 6, 8],
                1.0e-6,
                1.0e-7,
            ).unwrap();
            for t in 1..4 {
                for ch in 0..2 {
                    prop_assert!(
                        res.incremental_objectives[t][ch]
                            <= res.incremental_objectives[t - 1][ch] + 1.0e-10
                    );
                    prop_assert!(
                        res.joint_objectives[t][ch]
                            <= res.incremental_objectives[t][ch] + 1.0e-10
                    );
                }
            }
        }
    }
}
