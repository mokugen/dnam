//! Exact `f64` Preconditioned Conjugate Gradient (PCG) refinement for quantized (`Ozaki 3x8`)
//! upper-triangular Cholesky least-squares solutions (`ref[train.pipeline.nested-varpro-schur]`,
//! `ref[train.gram.accuracy-contract]`).

use super::solve_factorized_uplo_u;

const MAX_PCG_ITERS: usize = 16;
const _: () = assert!(MAX_PCG_ITERS > 0 && MAX_PCG_ITERS <= 64);

/// Problem specification for exact-`f64` [`refine_cholesky_lsq_pcg_f64`] using an already
/// factorized `uplo='U'` Cholesky matrix `M = R^T R` as an SPD preconditioner.
#[derive(Clone, Copy, Debug)]
pub struct PcgRefinementSpec<'a> {
    /// Column-major `f32` design matrix `F` of shape `[num_frames x dim]` (`col[j] = j * num_frames`).
    pub feats_col_major: &'a [f32],
    /// Target vector `y` of length `>= num_frames`.
    pub targets: &'a [f32],
    /// Number of valid rows `m` in each column and target slice.
    pub num_frames: usize,
    /// Number of active linear columns `d`.
    pub dim: usize,
    /// Positive stem weight multiplying `||F w - y||_2^2`.
    pub stem_weight: f64,
    /// Per-column anchor penalty weights `a[0..dim]` multiplying `0.5 * (w_j - w0_j)^2`.
    pub anchor_lam_diag: &'a [f64],
    /// Isotropic zero-centered ridge `rho_0 >= 0` multiplying `0.5 * ||w||_2^2` (`a_j + rho_0`
    /// matches the total diagonal shift inside `R^T R`).
    pub zero_ridge: f64,
    /// Optional anchor weights `w0[0..dim]` (`None` centers `anchor_lam_diag` at zero).
    pub anchor: Option<&'a [f64]>,
    /// Packed `uplo='U'` upper-triangular Cholesky factor `R` of dimension `dim`.
    pub r_upper: &'a [f64],
    /// Hard symmetric weight bound (`|w_j| <= max_abs_w`).
    pub max_abs_w: f64,
    /// Optional trust-region bound relative to `anchor` (`|w_j - w0_j| <= max_delta_w`).
    pub max_delta_w: Option<f64>,
    /// Maximum PCG iterations (clamped to `[0, 16]`).
    pub max_iters: usize,
}

#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (ca, ra) = a[..n].as_chunks::<4>();
    let (cb, rb) = b[..n].as_chunks::<4>();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (va, vb) in ca.iter().zip(cb) {
        s0 += va[0] * vb[0];
        s1 += va[1] * vb[1];
        s2 += va[2] * vb[2];
        s3 += va[3] * vb[3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

#[inline]
fn dot_f32_f64(col: &[f32], rhs: &[f64]) -> f64 {
    let n = col.len().min(rhs.len());
    let (ca, ra) = col[..n].as_chunks::<4>();
    let (cb, rb) = rhs[..n].as_chunks::<4>();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (va, vb) in ca.iter().zip(cb) {
        s0 += f64::from(va[0]) * vb[0];
        s1 += f64::from(va[1]) * vb[1];
        s2 += f64::from(va[2]) * vb[2];
        s3 += f64::from(va[3]) * vb[3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in ra.iter().zip(rb) {
        sum += f64::from(x) * y;
    }
    sum
}

#[inline]
fn apply_f_cols(feats_col_major: &[f32], m: usize, dim: usize, weights: &[f64], out: &mut [f64]) {
    out[..m].fill(0.0);
    for (j, &wj) in weights.iter().enumerate().take(dim) {
        if wj == 0.0 {
            continue;
        }
        let col = &feats_col_major[j * m..(j + 1) * m];
        let (co, ro) = out[..m].as_chunks_mut::<4>();
        let (cc, rc) = col.as_chunks::<4>();
        for (vo, vc) in co.iter_mut().zip(cc) {
            vo[0] += wj * f64::from(vc[0]);
            vo[1] += wj * f64::from(vc[1]);
            vo[2] += wj * f64::from(vc[2]);
            vo[3] += wj * f64::from(vc[3]);
        }
        for (vo, &vc) in ro.iter_mut().zip(rc) {
            *vo += wj * f64::from(vc);
        }
    }
}

#[inline]
fn anchor_at(anchor: Option<&[f64]>, j: usize) -> f64 {
    anchor.and_then(|a| a.get(j).copied()).unwrap_or(0.0)
}

#[inline]
fn eval_reg_lsq_objective(
    err_vec: &[f64],
    w: &[f64],
    anchor_lam: &[f64],
    zero_ridge: f64,
    anchor: Option<&[f64]>,
    stem_weight: f64,
) -> f64 {
    let data_term = stem_weight * dot_f64(err_vec, err_vec);
    let mut reg_term = 0.0_f64;
    for (j, (&wj, &aj)) in w.iter().zip(anchor_lam).enumerate() {
        let diff = wj - anchor_at(anchor, j);
        reg_term += aj * diff * diff + zero_ridge * wj * wj;
    }
    0.5 * (data_term + reg_term)
}

/// Refines an approximate Cholesky least-squares solution `w_init` toward the exact `f64`
/// minimizer of
/// `0.5 * (stem_weight * ||F w - y||_2^2 + sum_j a_j (w_j - w0_j)^2 + rho_0 * ||w||_2^2)`
/// via bounded, strictly monotone Preconditioned Conjugate Gradients using `r_upper` as preconditioner.
#[must_use]
pub fn refine_cholesky_lsq_pcg_f64(w_init: &[f64], spec: &PcgRefinementSpec<'_>) -> Vec<f64> {
    let dim = spec.dim;
    let m = spec.num_frames;
    let need_feats = dim.saturating_mul(m);
    if dim == 0
        || m == 0
        || spec.max_iters == 0
        || w_init.len() != dim
        || spec.anchor_lam_diag.len() != dim
        || spec.feats_col_major.len() < need_feats
        || spec.targets.len() < m
        || !spec.stem_weight.is_finite()
        || spec.stem_weight <= 0.0
        || !spec.max_abs_w.is_finite()
        || spec.max_abs_w <= 0.0
        || !w_init.iter().all(|v| v.is_finite())
        || !spec.feats_col_major[..need_feats]
            .iter()
            .all(|v| v.is_finite())
        || !spec.targets[..m].iter().all(|v| v.is_finite())
    {
        return w_init.to_vec();
    }

    let anchor_lam: Vec<f64> = spec
        .anchor_lam_diag
        .iter()
        .map(|&v| if v.is_finite() && v >= 0.0 { v } else { 0.0 })
        .collect();
    let rho_0 = if spec.zero_ridge.is_finite() && spec.zero_ridge >= 0.0 {
        spec.zero_ridge
    } else {
        0.0
    };
    let feats = &spec.feats_col_major[..need_feats];
    let y = &spec.targets[..m];
    let stem_w = spec.stem_weight;

    let mut w_cur = w_init.to_vec();
    let mut w_best = w_cur.clone();
    let mut e_cur = vec![0.0_f64; m];
    apply_f_cols(feats, m, dim, &w_cur, &mut e_cur);
    for (ec, &yt) in e_cur.iter_mut().zip(y) {
        *ec = f64::from(yt) - *ec;
    }

    let mut best_obj =
        eval_reg_lsq_objective(&e_cur, &w_cur, &anchor_lam, rho_0, spec.anchor, stem_w);
    if !best_obj.is_finite() {
        return w_best;
    }

    let mut r = vec![0.0_f64; dim];
    for j in 0..dim {
        let diff = w_cur[j] - anchor_at(spec.anchor, j);
        r[j] = stem_w * dot_f32_f64(&feats[j * m..(j + 1) * m], &e_cur)
            - anchor_lam[j] * diff
            - rho_0 * w_cur[j];
    }

    let Ok(mut p) = solve_factorized_uplo_u(dim, spec.r_upper, &r) else {
        return w_best;
    };
    let mut rz = dot_f64(&r, &p);
    let rz_init = rz;
    if !rz.is_finite() || rz <= 1.0e-28 * best_obj.max(1.0) {
        return w_best;
    }

    let mut fp = vec![0.0_f64; m];
    let mut ap = vec![0.0_f64; dim];

    for _iter in 0..spec.max_iters.min(MAX_PCG_ITERS) {
        apply_f_cols(feats, m, dim, &p, &mut fp);
        let mut reg_pap = 0.0_f64;
        for j in 0..dim {
            let total_lam_j = anchor_lam[j] + rho_0;
            ap[j] = stem_w * dot_f32_f64(&feats[j * m..(j + 1) * m], &fp) + total_lam_j * p[j];
            reg_pap += total_lam_j * p[j] * p[j];
        }
        let p_ap = stem_w * dot_f64(&fp, &fp) + reg_pap;
        if !p_ap.is_finite() || p_ap <= 1.0e-30 {
            break;
        }
        let alpha = rz / p_ap;
        if !alpha.is_finite() || alpha <= 0.0 {
            break;
        }

        for j in 0..dim {
            w_cur[j] += alpha * p[j];
            r[j] -= alpha * ap[j];
        }
        for (ec, &fpt) in e_cur.iter_mut().zip(&fp) {
            *ec -= alpha * fpt;
        }

        let bounds_ok = w_cur
            .iter()
            .all(|&v| v.is_finite() && v.abs() <= spec.max_abs_w);
        let delta_ok = spec.max_delta_w.is_none_or(|md| {
            w_cur
                .iter()
                .enumerate()
                .all(|(j, &v)| (v - anchor_at(spec.anchor, j)).abs() <= md)
        });
        if !bounds_ok || !delta_ok {
            break;
        }

        let cur_obj =
            eval_reg_lsq_objective(&e_cur, &w_cur, &anchor_lam, rho_0, spec.anchor, stem_w);
        if cur_obj.is_finite() && cur_obj < best_obj {
            best_obj = cur_obj;
            w_best.copy_from_slice(&w_cur);
        }

        let Ok(z_next) = solve_factorized_uplo_u(dim, spec.r_upper, &r) else {
            break;
        };
        let rz_next = dot_f64(&r, &z_next);
        if !rz_next.is_finite() || rz_next <= 1.0e-22 * rz_init.max(1.0e-30) {
            break;
        }
        let beta = rz_next / rz;
        if !beta.is_finite() || beta < 0.0 {
            break;
        }
        for j in 0..dim {
            p[j] = z_next[j] + beta * p[j];
        }
        rz = rz_next;
    }

    w_best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::cholesky::{factorize_uplo_u_cholesky, solve_uplo_u_cholesky};
    use crate::wide_gram::{contract_dim_gram_tiled_wide192, uplo_u_index};

    #[test]
    fn test_pcg_refine_recovers_exact_collinear_f64_from_noisy_gram_preconditioner() {
        let m = 256_usize;
        let dim = 6_usize;
        let mut feats_col = vec![0.0_f32; dim * m];
        for t in 0..m {
            let tf = t as f64;
            feats_col[t] = (0.07 * tf).sin() as f32;
            feats_col[m + t] = (0.19 * tf + 0.3).cos() as f32;
            feats_col[2 * m + t] = (0.43 * tf).sin() as f32;
            feats_col[3 * m + t] = 0.75 * feats_col[t] + 1.0e-3 * ((0.31 * tf).sin() as f32);
            feats_col[4 * m + t] = 0.50 * feats_col[m + t] + 0.25 * feats_col[t];
            feats_col[5 * m + t] = (0.83 * tf + 0.5).cos() as f32;
        }
        let w_true = [0.65_f64, -0.42, 0.31, -0.28, 0.19, -0.15];
        let targets: Vec<f32> = (0..m)
            .map(|t| {
                (0..dim)
                    .map(|j| w_true[j] * f64::from(feats_col[j * m + t]))
                    .sum::<f64>() as f32
            })
            .collect();

        let mut feats_row = vec![0.0_f32; m * dim];
        for j in 0..dim {
            for t in 0..m {
                feats_row[t * dim + j] = feats_col[j * m + t];
            }
        }
        let wts = vec![1.0_f64; m];
        let tgt_pairs: Vec<[f64; 2]> = targets.iter().map(|&y| [f64::from(y), 0.0]).collect();
        let (gram_exact, rhs_pairs) =
            contract_dim_gram_tiled_wide192(&feats_row, &tgt_pairs, &wts, dim, 1);
        let rhs_exact: Vec<f64> = rhs_pairs.iter().map(|p| p[0]).collect();

        // Simulate 3x8-bit Ozaki quantization floor (~2e-6 relative noise) on G and rhs.
        let mut gram_ozaki = gram_exact.clone();
        for i in 0..dim {
            for j in i..dim {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                gram_ozaki[uplo_u_index(i, j)] += sign * 2.0e-5 * (i as f64 + 1.0);
            }
        }
        let mut rhs_ozaki = rhs_exact.clone();
        for (j, rj) in rhs_ozaki.iter_mut().enumerate() {
            *rj += if j % 2 == 0 { 5.0e-6 } else { -5.0e-6 };
        }

        let ridge = 1.0e-10;
        let zero_anchor_lam = vec![0.0_f64; dim];
        let w_ref = solve_uplo_u_cholesky(&gram_exact, &rhs_exact, ridge).unwrap();
        let r_upper = factorize_uplo_u_cholesky(dim, &gram_ozaki, ridge).unwrap();
        let w_oz = solve_factorized_uplo_u(dim, &r_upper, &rhs_ozaki).unwrap();

        let err_oz: f64 = (0..m)
            .map(|t| {
                let pred: f64 = (0..dim)
                    .map(|j| w_oz[j] * f64::from(feats_col[j * m + t]))
                    .sum();
                let d = f64::from(targets[t]) - pred;
                d * d
            })
            .sum();

        let w_pcg = refine_cholesky_lsq_pcg_f64(
            &w_oz,
            &PcgRefinementSpec {
                feats_col_major: &feats_col,
                targets: &targets,
                num_frames: m,
                dim,
                stem_weight: 1.0,
                anchor_lam_diag: &zero_anchor_lam,
                zero_ridge: ridge,
                anchor: None,
                r_upper: &r_upper,
                max_abs_w: 7.5,
                max_delta_w: None,
                max_iters: 6,
            },
        );
        let err_pcg: f64 = (0..m)
            .map(|t| {
                let pred: f64 = (0..dim)
                    .map(|j| w_pcg[j] * f64::from(feats_col[j * m + t]))
                    .sum();
                let d = f64::from(targets[t]) - pred;
                d * d
            })
            .sum();
        let pred_diff_sq: f64 = (0..m)
            .map(|t| {
                let d: f64 = (0..dim)
                    .map(|j| (w_pcg[j] - w_ref[j]) * f64::from(feats_col[j * m + t]))
                    .sum();
                d * d
            })
            .sum();
        let max_w_diff = w_pcg
            .iter()
            .zip(&w_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err_pcg < err_oz * 1.0e-4 && pred_diff_sq < 1.0e-12 && max_w_diff < 1.0e-3,
            "PCG must recover exact f64 solution: err_oz={err_oz:.3e} err_pcg={err_pcg:.3e} pred_diff_sq={pred_diff_sq:.3e} max_w_diff={max_w_diff:.3e}"
        );

        // Exact initial vector triggers immediate early exit with unchanged weights.
        let w_unchanged = refine_cholesky_lsq_pcg_f64(
            &w_pcg,
            &PcgRefinementSpec {
                feats_col_major: &feats_col,
                targets: &targets,
                num_frames: m,
                dim,
                stem_weight: 1.0,
                anchor_lam_diag: &zero_anchor_lam,
                zero_ridge: ridge,
                anchor: None,
                r_upper: &r_upper,
                max_abs_w: 7.5,
                max_delta_w: None,
                max_iters: 6,
            },
        );
        assert_eq!(w_unchanged, w_pcg);
    }

    #[test]
    fn test_pcg_refine_anchored_bounds_and_guard_paths() {
        let m = 48_usize;
        let dim = 2_usize;
        let mut feats = vec![0.0_f32; dim * m];
        let mut targets = vec![0.0_f32; m];
        for t in 0..m {
            let x0 = ((t as f64) * 0.2).sin() as f32;
            let x1 = ((t as f64) * 0.35).cos() as f32;
            feats[t] = x0;
            feats[m + t] = x1;
            targets[t] = 3.0 * x0 - 2.0 * x1;
        }
        let gram = vec![24.0, 0.0, 24.0];
        let lam = vec![1.0, 1.0];
        let anchor = vec![0.1, -0.1];
        let r_upper = factorize_uplo_u_cholesky(dim, &gram, 1.0).unwrap();

        // Trust region max_delta_w = 0.01 prevents leaving anchor sphere even when w_init is within it.
        let w_bounded = refine_cholesky_lsq_pcg_f64(
            &anchor,
            &PcgRefinementSpec {
                feats_col_major: &feats,
                targets: &targets,
                num_frames: m,
                dim,
                stem_weight: 2.0,
                anchor_lam_diag: &lam,
                zero_ridge: 0.0,
                anchor: Some(&anchor),
                r_upper: &r_upper,
                max_abs_w: 7.5,
                max_delta_w: Some(0.01),
                max_iters: 4,
            },
        );
        assert_eq!(w_bounded, anchor);

        // Relaxed trust region moves monotonically toward anchored optimum.
        let w_anchored = refine_cholesky_lsq_pcg_f64(
            &anchor,
            &PcgRefinementSpec {
                feats_col_major: &feats,
                targets: &targets,
                num_frames: m,
                dim,
                stem_weight: 2.0,
                anchor_lam_diag: &lam,
                zero_ridge: 0.0,
                anchor: Some(&anchor),
                r_upper: &r_upper,
                max_abs_w: 7.5,
                max_delta_w: Some(5.0),
                max_iters: 6,
            },
        );
        assert!(w_anchored[0] > anchor[0] && w_anchored[1] < anchor[1]);

        // Guard paths return w_init unchanged cleanly.
        assert_eq!(
            refine_cholesky_lsq_pcg_f64(
                &[1.0, 2.0],
                &PcgRefinementSpec {
                    feats_col_major: &feats,
                    targets: &targets,
                    num_frames: m,
                    dim: 0,
                    stem_weight: 1.0,
                    anchor_lam_diag: &lam,
                    zero_ridge: 0.0,
                    anchor: None,
                    r_upper: &r_upper,
                    max_abs_w: 7.5,
                    max_delta_w: None,
                    max_iters: 4,
                },
            ),
            vec![1.0, 2.0]
        );
        assert_eq!(
            refine_cholesky_lsq_pcg_f64(
                &anchor,
                &PcgRefinementSpec {
                    feats_col_major: &feats,
                    targets: &targets,
                    num_frames: m,
                    dim,
                    stem_weight: 1.0,
                    anchor_lam_diag: &lam,
                    zero_ridge: 0.0,
                    anchor: None,
                    r_upper: &[1.0], // mismatched packed len
                    max_abs_w: 7.5,
                    max_delta_w: None,
                    max_iters: 4,
                },
            ),
            anchor
        );
    }

    #[test]
    fn test_pcg_refine_nonzero_anchor_zero_ridge_objective_parity() {
        let m = 96_usize;
        let dim = 3_usize;
        let mut feats = vec![0.0_f32; dim * m];
        let mut targets = vec![0.0_f32; m];
        for t in 0..m {
            let x0 = ((t as f64) * 0.17).sin() as f32;
            let x1 = ((t as f64) * 0.31).cos() as f32;
            let x2 = ((t as f64) * 0.47).sin() as f32;
            feats[t] = x0;
            feats[m + t] = x1;
            feats[2 * m + t] = x2;
            targets[t] = 0.8 * x0 - 0.6 * x1 + 0.4 * x2;
        }
        let mut feats_row = vec![0.0_f32; m * dim];
        for j in 0..dim {
            for t in 0..m {
                feats_row[t * dim + j] = feats[j * m + t];
            }
        }
        let wts = vec![1.0_f64; m];
        let tgt_pairs: Vec<[f64; 2]> = targets.iter().map(|&y| [f64::from(y), 0.0]).collect();
        let (mut g_sys, rhs_pairs) =
            contract_dim_gram_tiled_wide192(&feats_row, &tgt_pairs, &wts, dim, 1);

        let anchor = [2.5_f64, -1.5, 3.0];
        let anchor_lam = [1.0e-4_f64, 5.0e-2, 1.0e-4];
        let zero_ridge = 10.0_f64;
        let mut r_anchored = [0.0_f64; 3];
        for j in 0..dim {
            r_anchored[j] = rhs_pairs[j][0] + anchor_lam[j] * anchor[j];
            g_sys[uplo_u_index(j, j)] += anchor_lam[j] + zero_ridge;
        }
        let r_exact = factorize_uplo_u_cholesky(dim, &g_sys, 0.0).unwrap();
        let w_exact = solve_factorized_uplo_u(dim, &r_exact, &r_anchored).unwrap();

        // Exact Cholesky solution has zero spurious PCG residual even when zero_ridge >> anchor_lam!
        let w_from_exact = refine_cholesky_lsq_pcg_f64(
            &w_exact,
            &PcgRefinementSpec {
                feats_col_major: &feats,
                targets: &targets,
                num_frames: m,
                dim,
                stem_weight: 1.0,
                anchor_lam_diag: &anchor_lam,
                zero_ridge,
                anchor: Some(&anchor),
                r_upper: &r_exact,
                max_abs_w: 7.5,
                max_delta_w: None,
                max_iters: 8,
            },
        );
        for (a, b) in w_from_exact.iter().zip(&w_exact) {
            assert!((a - b).abs() < 1.0e-12, "spurious shift: {a} vs {b}");
        }

        // Ozaki-perturbed start converges cleanly onto w_exact rather than drifting toward anchor.
        let mut g_noisy = g_sys;
        g_noisy[uplo_u_index(0, 1)] += 3.0;
        let r_noisy = factorize_uplo_u_cholesky(dim, &g_noisy, 0.0).unwrap();
        let w_noisy = solve_factorized_uplo_u(dim, &r_noisy, &r_anchored).unwrap();
        let w_refined = refine_cholesky_lsq_pcg_f64(
            &w_noisy,
            &PcgRefinementSpec {
                feats_col_major: &feats,
                targets: &targets,
                num_frames: m,
                dim,
                stem_weight: 1.0,
                anchor_lam_diag: &anchor_lam,
                zero_ridge,
                anchor: Some(&anchor),
                r_upper: &r_noisy,
                max_abs_w: 7.5,
                max_delta_w: None,
                max_iters: 12,
            },
        );
        for ((&wr, &we), &wn) in w_refined.iter().zip(&w_exact).zip(&w_noisy) {
            assert!(
                (wr - we).abs() < (wn - we).abs() * 1.0e-4 + 1.0e-11,
                "failed to converge to w_exact: wr={wr}, we={we}, wn={wn}"
            );
        }
    }
}
