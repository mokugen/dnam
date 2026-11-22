//! Upper-triangular (`uplo='U'`) packed IEEE-754 `f64` Cholesky factorization and
//! regularized trust-region solvers (`ref[container.proj.stat-uplo-u]`, `ref[train.pipeline.nested-varpro-schur]`).

use crate::error::TrainError;
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

/// Floor applied to diagonal Tikhonov ridge during Cholesky factorization.
pub const MIN_CHOLESKY_RIDGE: f64 = 1.0e-12;

/// Returns `Some(dim * (dim + 1) / 2)` without intermediate overflow.
#[inline]
#[must_use]
pub const fn checked_uplo_u_packed_len(dim: usize) -> Option<usize> {
    if dim.is_multiple_of(2) {
        match dim.checked_add(1) {
            Some(d1) => (dim / 2).checked_mul(d1),
            None => None,
        }
    } else {
        dim.checked_mul(dim.div_ceil(2))
    }
}

fn factorize_uplo_u_cholesky_inner(
    dim: usize,
    gram_uplo: &[f64],
    eff_ridge: f64,
    context: &'static str,
) -> Result<Vec<f64>, TrainError> {
    let expected = checked_uplo_u_packed_len(dim).ok_or(TrainError::InvalidDimensions {
        context,
        expected: usize::MAX,
        actual: dim,
    })?;
    if gram_uplo.len() != expected {
        return Err(TrainError::InvalidDimensions {
            context,
            expected,
            actual: gram_uplo.len(),
        });
    }
    if !eff_ridge.is_finite() || eff_ridge < 0.0 || !gram_uplo.iter().all(|v| v.is_finite()) {
        return Err(TrainError::NonFiniteInput { context });
    }
    if dim == 0 {
        return Ok(Vec::new());
    }

    let mut r_upper = gram_uplo.to_vec();
    if eff_ridge > 0.0 {
        for j in 0..dim {
            r_upper[uplo_u_index(j, j)] += eff_ridge;
        }
    }

    for j in 0..dim {
        let col_j = uplo_u_index(0, j);
        for i in 0..j {
            let col_i = uplo_u_index(0, i);
            let (ci_chunks, ci_rem) = r_upper[col_i..col_i + i].as_chunks::<4>();
            let (cj_chunks, cj_rem) = r_upper[col_j..col_j + i].as_chunks::<4>();
            let (mut d0, mut d1, mut d2, mut d3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for (a, b) in ci_chunks.iter().zip(cj_chunks) {
                d0 += a[0] * b[0];
                d1 += a[1] * b[1];
                d2 += a[2] * b[2];
                d3 += a[3] * b[3];
            }
            let mut dot = (d0 + d1) + (d2 + d3);
            for (&r_ki, &r_kj) in ci_rem.iter().zip(cj_rem) {
                dot += r_ki * r_kj;
            }
            r_upper[col_j + i] = (r_upper[col_j + i] - dot) / r_upper[col_i + i];
        }
        let mut diag = r_upper[col_j + j];
        for &r_kj in &r_upper[col_j..col_j + j] {
            diag -= r_kj * r_kj;
        }
        if !diag.is_finite() || diag <= 0.0 {
            return Err(TrainError::SingularMatrix {
                context,
                pivot_index: j,
            });
        }
        r_upper[col_j + j] = diag.sqrt();
    }
    Ok(r_upper)
}

/// Factorizes `G + max(ridge, MIN_CHOLESKY_RIDGE) * I = R^T R` in `f64` from column-major `uplo='U'` storage.
pub fn factorize_uplo_u_cholesky(
    dim: usize,
    gram_uplo: &[f64],
    ridge: f64,
) -> Result<Vec<f64>, TrainError> {
    if !ridge.is_finite() || ridge < 0.0 {
        return Err(TrainError::NonFiniteInput {
            context: "factorize_uplo_u_cholesky",
        });
    }
    factorize_uplo_u_cholesky_inner(
        dim,
        gram_uplo,
        ridge.max(MIN_CHOLESKY_RIDGE),
        "factorize_uplo_u_cholesky",
    )
}

/// Factorizes an already-regularized `G = R^T R` in `f64` without injecting an extra diagonal ridge.
pub(crate) fn factorize_uplo_u_cholesky_exact_no_ridge(
    dim: usize,
    gram_uplo: &[f64],
) -> Result<Vec<f64>, TrainError> {
    factorize_uplo_u_cholesky_inner(
        dim,
        gram_uplo,
        0.0,
        "factorize_uplo_u_cholesky_exact_no_ridge",
    )
}

/// Solves `(R^T R) w = rhs` given an upper-triangular Cholesky factor `r_upper` in `uplo='U'` order.
pub fn solve_factorized_uplo_u(
    dim: usize,
    r_upper: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, TrainError> {
    let expected = checked_uplo_u_packed_len(dim).ok_or(TrainError::InvalidDimensions {
        context: "solve_factorized_uplo_u(dim)",
        expected: usize::MAX,
        actual: dim,
    })?;
    if r_upper.len() != expected {
        return Err(TrainError::InvalidDimensions {
            context: "solve_factorized_uplo_u(r_upper)",
            expected,
            actual: r_upper.len(),
        });
    }
    if rhs.len() != dim {
        return Err(TrainError::InvalidDimensions {
            context: "solve_factorized_uplo_u(rhs)",
            expected: dim,
            actual: rhs.len(),
        });
    }
    if !r_upper.iter().all(|v| v.is_finite()) || !rhs.iter().all(|v| v.is_finite()) {
        return Err(TrainError::NonFiniteInput {
            context: "solve_factorized_uplo_u",
        });
    }
    if dim == 0 {
        return Ok(Vec::new());
    }

    let mut y = vec![0.0_f64; dim];
    for i in 0..dim {
        let col_i = uplo_u_index(0, i);
        let mut sum = rhs[i];
        for (&r_ki, &yk) in r_upper[col_i..col_i + i].iter().zip(&y[..i]) {
            sum -= r_ki * yk;
        }
        let diag = r_upper[col_i + i];
        if !diag.is_finite() || diag <= 0.0 {
            return Err(TrainError::SingularMatrix {
                context: "solve_factorized_uplo_u",
                pivot_index: i,
            });
        }
        y[i] = sum / diag;
    }

    let mut w = y;
    for k in (0..dim).rev() {
        let col_k = uplo_u_index(0, k);
        let wk = w[k] / r_upper[col_k + k];
        w[k] = wk;
        for (wi, &r_ik) in w[..k].iter_mut().zip(&r_upper[col_k..col_k + k]) {
            *wi -= r_ik * wk;
        }
    }
    if !w.iter().all(|v| v.is_finite()) {
        return Err(TrainError::NonFiniteInput {
            context: "solve_factorized_uplo_u(output)",
        });
    }
    Ok(w)
}

/// Solves `(G + ridge * I) w = rhs` from `uplo='U'` packed storage.
pub fn solve_uplo_u_cholesky(
    gram_uplo: &[f64],
    rhs: &[f64],
    ridge: f64,
) -> Result<Vec<f64>, TrainError> {
    let dim = rhs.len();
    let r_upper = factorize_uplo_u_cholesky(dim, gram_uplo, ridge)?;
    solve_factorized_uplo_u(dim, &r_upper, rhs)
}

/// Solves stereo `(R^T R) [w_l, w_r] = [c_l, c_r]` simultaneously with a shared factor `r_upper`.
pub fn solve_factorized_uplo_u_2ch(
    dim: usize,
    r_upper: &[f64],
    rhs_lr: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, TrainError> {
    if rhs_lr.len() != dim {
        return Err(TrainError::InvalidDimensions {
            context: "solve_factorized_uplo_u_2ch(rhs_lr)",
            expected: dim,
            actual: rhs_lr.len(),
        });
    }
    let expected = checked_uplo_u_packed_len(dim).unwrap_or(0);
    if r_upper.len() != expected {
        return Err(TrainError::InvalidDimensions {
            context: "solve_factorized_uplo_u_2ch(r_upper)",
            expected,
            actual: r_upper.len(),
        });
    }
    if !rhs_lr.iter().all(|c| c[0].is_finite() && c[1].is_finite())
        || !r_upper.iter().all(|v| v.is_finite())
    {
        return Err(TrainError::NonFiniteInput {
            context: "solve_factorized_uplo_u_2ch",
        });
    }
    if dim == 0 {
        return Ok(Vec::new());
    }

    let mut y = vec![[0.0_f64; 2]; dim];
    for i in 0..dim {
        let col_i = uplo_u_index(0, i);
        let mut s0 = rhs_lr[i][0];
        let mut s1 = rhs_lr[i][1];
        for (&r_ki, yk) in r_upper[col_i..col_i + i].iter().zip(&y[..i]) {
            s0 -= r_ki * yk[0];
            s1 -= r_ki * yk[1];
        }
        let diag = r_upper[col_i + i];
        if !diag.is_finite() || diag <= 0.0 {
            return Err(TrainError::SingularMatrix {
                context: "solve_factorized_uplo_u_2ch",
                pivot_index: i,
            });
        }
        y[i] = [s0 / diag, s1 / diag];
    }

    let mut w = y;
    for k in (0..dim).rev() {
        let col_k = uplo_u_index(0, k);
        let diag = r_upper[col_k + k];
        let wk0 = w[k][0] / diag;
        let wk1 = w[k][1] / diag;
        w[k] = [wk0, wk1];
        for (wi, &r_ik) in w[..k].iter_mut().zip(&r_upper[col_k..col_k + k]) {
            wi[0] -= r_ik * wk0;
            wi[1] -= r_ik * wk1;
        }
    }
    if !w.iter().all(|c| c[0].is_finite() && c[1].is_finite()) {
        return Err(TrainError::NonFiniteInput {
            context: "solve_factorized_uplo_u_2ch(output)",
        });
    }
    Ok(w)
}

/// Solves stereo `(G + ridge * I) W = C` in `f64` from `uplo='U'` packed Gram storage.
pub fn solve_uplo_u_cholesky_2ch(
    gram_uplo: &[f64],
    rhs_lr: &[[f64; 2]],
    ridge: f64,
) -> Result<Vec<[f64; 2]>, TrainError> {
    let dim = rhs_lr.len();
    let r_upper = factorize_uplo_u_cholesky(dim, gram_uplo, ridge)?;
    solve_factorized_uplo_u_2ch(dim, &r_upper, rhs_lr)
}

/// Evaluates `G * w` cleanly where `G` is packed column-major `uplo='U'`.
#[must_use]
pub fn symv_uplo_u(dim: usize, gram_uplo: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; dim];
    if gram_uplo.len() != uplo_u_packed_len(dim) || w.len() != dim {
        return out;
    }
    for j in 0..dim {
        let col_j = uplo_u_index(0, j);
        let wj = w[j];
        let mut dot = 0.0_f64;
        for i in 0..j {
            let g_ij = gram_uplo[col_j + i];
            out[i] += g_ij * wj;
            dot += g_ij * w[i];
        }
        out[j] += dot + gram_uplo[col_j + j] * wj;
    }
    out
}

/// Solves a potentially ill-conditioned `uplo='U'` system via trace-scaled adaptive ridge ladder
/// (`1e-12..1.0`) with optional column bypass penalties and anchor regularization.
pub fn solve_uplo_adaptive_ridge(
    dim: usize,
    gram_uplo: &[f64],
    rhs: &[f64],
    bypass_cols: &[(usize, f64)],
    anchor: Option<(&[(usize, f64)], f64)>,
    min_ridge: f64,
) -> Result<(Vec<f64>, f64), TrainError> {
    if rhs.len() != dim {
        return Err(TrainError::InvalidDimensions {
            context: "solve_uplo_adaptive_ridge(rhs)",
            expected: dim,
            actual: rhs.len(),
        });
    }
    if gram_uplo.len() != uplo_u_packed_len(dim) {
        return Err(TrainError::InvalidDimensions {
            context: "solve_uplo_adaptive_ridge(gram_uplo)",
            expected: uplo_u_packed_len(dim),
            actual: gram_uplo.len(),
        });
    }
    if !min_ridge.is_finite()
        || min_ridge < 0.0
        || !rhs.iter().all(|v| v.is_finite())
        || !gram_uplo.iter().all(|v| v.is_finite())
    {
        return Err(TrainError::NonFiniteInput {
            context: "solve_uplo_adaptive_ridge",
        });
    }
    if dim == 0 {
        return Ok((Vec::new(), 0.0));
    }

    let mut trace = 0.0_f64;
    for j in 0..dim {
        trace += gram_uplo[uplo_u_index(j, j)].max(0.0);
    }
    let mean_diag = (trace / (dim as f64)).max(MIN_CHOLESKY_RIDGE);

    let mut g_work = gram_uplo.to_vec();
    let mut r_work = rhs.to_vec();
    for &(idx, scale) in bypass_cols {
        if idx < dim && scale.is_finite() && scale > 0.0 {
            g_work[uplo_u_index(idx, idx)] += scale * mean_diag;
        }
    }
    if let Some((anchor_entries, anchor_scale)) = anchor
        && anchor_scale.is_finite()
        && anchor_scale > 0.0
    {
        let lam_a = anchor_scale * mean_diag;
        for &(idx, val) in anchor_entries {
            if idx < dim && val.is_finite() {
                g_work[uplo_u_index(idx, idx)] += lam_a;
                r_work[idx] += lam_a * val;
            }
        }
    }

    const LADDERS: [f64; 9] = [
        1.0e-12, 1.0e-10, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-3, 1.0e-1, 1.0,
    ];
    let floor_ridge = min_ridge.max(MIN_CHOLESKY_RIDGE);
    let mut last_err = TrainError::SingularMatrix {
        context: "solve_uplo_adaptive_ridge",
        pivot_index: 0,
    };
    for &mult in &LADDERS {
        let ridge = (mult * mean_diag).max(floor_ridge);
        match solve_uplo_u_cholesky(&g_work, &r_work, ridge) {
            Ok(sol) => return Ok((sol, ridge)),
            Err(err) => last_err = err,
        }
    }
    Err(last_err)
}

/// Solves a regularized `uplo='U'` readout system while enforcing `||w[..pen_dim]||_2 <= max_l2_norm`
/// by bisecting a trace-scaled Tikhonov trust-region multiplier `mu`.
pub fn solve_uplo_norm_bounded(
    dim: usize,
    gram_uplo: &[f64],
    rhs: &[f64],
    pen_dim: usize,
    base_ridge: f64,
    max_l2_norm: f64,
) -> Result<(Vec<f64>, f64), TrainError> {
    if !max_l2_norm.is_finite() || max_l2_norm <= 0.0 {
        return Err(TrainError::NonFiniteInput {
            context: "solve_uplo_norm_bounded(max_l2_norm)",
        });
    }
    let pen = pen_dim.min(dim);
    let max_sq = max_l2_norm * max_l2_norm;
    let norm_sq = |w: &[f64]| -> f64 { w[..pen].iter().map(|v| v * v).sum() };

    let (w0, r0) = solve_uplo_adaptive_ridge(dim, gram_uplo, rhs, &[], None, base_ridge)?;
    if norm_sq(&w0) <= max_sq {
        return Ok((w0, 0.0));
    }

    let eval_mu = |mu: f64| -> Result<Vec<f64>, TrainError> {
        let bypass: Vec<(usize, f64)> = (0..pen).map(|i| (i, mu)).collect();
        solve_uplo_adaptive_ridge(dim, gram_uplo, rhs, &bypass, None, r0).map(|(w, _)| w)
    };

    const BRACKETS: [f64; 10] = [
        1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 1.0e2, 1.0e3, 1.0e5,
    ];
    let mut mu_lo = 0.0_f64;
    let mut mu_hi = *BRACKETS.last().unwrap_or(&1.0e5);
    let mut best_w = None;
    for &mu in &BRACKETS {
        let cand = eval_mu(mu)?;
        if norm_sq(&cand) <= max_sq {
            mu_hi = mu;
            best_w = Some(cand);
            break;
        }
        mu_lo = mu;
    }
    let Some(mut feasible_w) = best_w else {
        return Ok((eval_mu(mu_hi)?, mu_hi));
    };

    for _ in 0..12 {
        let mu_mid = if mu_lo > 0.0 {
            (mu_lo * mu_hi).sqrt()
        } else {
            0.5 * mu_hi
        };
        let cand = eval_mu(mu_mid)?;
        if norm_sq(&cand) <= max_sq {
            mu_hi = mu_mid;
            feasible_w = cand;
        } else {
            mu_lo = mu_mid;
        }
    }
    Ok((feasible_w, mu_hi))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_cholesky_exact_solve_errors_adaptive_ridge_and_norm_bound() {
        assert_eq!(checked_uplo_u_packed_len(4), Some(10));
        assert!(factorize_uplo_u_cholesky(0, &[], 0.0).unwrap().is_empty());
        assert!(solve_factorized_uplo_u(0, &[], &[]).unwrap().is_empty());
        assert!(solve_factorized_uplo_u_2ch(0, &[], &[]).unwrap().is_empty());
        assert!(
            solve_uplo_adaptive_ridge(0, &[], &[], &[], None, 0.0)
                .unwrap()
                .0
                .is_empty()
        );

        assert!(matches!(
            factorize_uplo_u_cholesky(3, &[1.0; 5], 0.0),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            factorize_uplo_u_cholesky(2, &[1.0, f64::NAN, 1.0], 0.0),
            Err(TrainError::NonFiniteInput { .. })
        ));
        // Negative diagonal yields SingularMatrix.
        assert!(matches!(
            factorize_uplo_u_cholesky(2, &[1.0, 0.0, -5.0], 1.0e-12),
            Err(TrainError::SingularMatrix { pivot_index: 1, .. })
        ));

        // Collinear system where col 0..2 match target ~ 0.5*x but col 0 & 1 are strongly anticorrelated.
        let dim = 4;
        let mut gram = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut rhs_lr = vec![[0.0_f64; 2]; dim];
        for t in 0..64 {
            let x = (0.17 * (t as f64)).sin();
            let dx = 1.0e-4 * (0.43 * (t as f64)).cos();
            let row = [x, x + dx, 0.5 * x, -0.3 * (0.29 * (t as f64)).sin()];
            let y_l = 0.5 * x + 5.0e-3 * (0.43 * (t as f64)).cos();
            let y_r = -0.25 * x;
            for j in 0..dim {
                rhs_lr[j][0] += row[j] * y_l;
                rhs_lr[j][1] += row[j] * y_r;
                for i in 0..=j {
                    gram[uplo_u_index(i, j)] += row[i] * row[j];
                }
            }
        }
        let w_2ch = solve_uplo_u_cholesky_2ch(&gram, &rhs_lr, 1.0e-10).unwrap();
        let rhs_l: Vec<f64> = rhs_lr.iter().map(|v| v[0]).collect();
        let w_l: Vec<f64> = w_2ch.iter().map(|v| v[0]).collect();
        let mut reg_gram = gram.clone();
        for j in 0..dim {
            reg_gram[uplo_u_index(j, j)] += 1.0e-10;
        }
        let gw = symv_uplo_u(dim, &reg_gram, &w_l);
        let max_res = gw
            .iter()
            .zip(&rhs_l)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_res <= 1.0e-11, "max_res={max_res}");

        let (w_bounded, mu) = solve_uplo_norm_bounded(dim, &gram, &rhs_l, 2, 1.0e-12, 2.0).unwrap();
        let l2 = (w_bounded[0] * w_bounded[0] + w_bounded[1] * w_bounded[1]).sqrt();
        assert!(mu > 0.0 && l2 <= 2.0 + 1.0e-9, "mu={mu} l2={l2}");

        let (w_anch, _) = solve_uplo_adaptive_ridge(
            dim,
            &gram,
            &rhs_l,
            &[(0, 0.1)],
            Some((&[(3, 1.5)], 10.0)),
            1.0e-9,
        )
        .unwrap();
        assert!((w_anch[3] - 1.5).abs() < 0.25);
    }

    #[test]
    fn test_cholesky_error_guards_and_edge_branches() {
        assert_eq!(checked_uplo_u_packed_len(3), Some(6));
        assert_eq!(checked_uplo_u_packed_len(usize::MAX), None);
        assert!(matches!(
            factorize_uplo_u_cholesky(2, &[1.0, 0.0, 1.0], -1.0),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u(2, &[1.0, 0.0], &[1.0, 1.0]),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u(2, &[1.0, 0.0, 1.0], &[1.0]),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u(2, &[1.0, 0.0, 1.0], &[f64::NAN, 1.0]),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u(2, &[1.0, 0.0, 0.0], &[1.0, 1.0]),
            Err(TrainError::SingularMatrix { pivot_index: 1, .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u(1, &[1.0e-308], &[1.0e20]),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u_2ch(2, &[1.0, 0.0, 1.0], &[[1.0, 1.0]]),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u_2ch(2, &[1.0, 0.0], &[[1.0, 1.0]; 2]),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u_2ch(2, &[1.0, 0.0, 1.0], &[[f64::NAN, 1.0], [1.0, 1.0]]),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u_2ch(2, &[1.0, 0.0, 0.0], &[[1.0, 1.0]; 2]),
            Err(TrainError::SingularMatrix { pivot_index: 1, .. })
        ));
        assert!(matches!(
            solve_factorized_uplo_u_2ch(1, &[1.0e-308], &[[1.0e20, 1.0]]),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert_eq!(symv_uplo_u(2, &[1.0], &[1.0, 1.0]), vec![0.0, 0.0]);
        assert!(matches!(
            solve_uplo_adaptive_ridge(2, &[1.0, 0.0, 1.0], &[1.0], &[], None, 0.0),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_uplo_adaptive_ridge(2, &[1.0, 0.0], &[1.0, 1.0], &[], None, 0.0),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_uplo_adaptive_ridge(2, &[1.0, 0.0, 1.0], &[1.0, 1.0], &[], None, -1.0),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            solve_uplo_adaptive_ridge(1, &[-10.0], &[1.0], &[], None, 0.0),
            Err(TrainError::SingularMatrix { .. })
        ));
        assert!(matches!(
            solve_uplo_norm_bounded(1, &[1.0], &[1.0], 1, 1.0e-12, 0.0),
            Err(TrainError::NonFiniteInput { .. })
        ));
        let (w_free, mu_free) =
            solve_uplo_norm_bounded(1, &[1.0], &[0.5], 1, 1.0e-12, 10.0).unwrap();
        assert_eq!(mu_free, 0.0);
        assert!((w_free[0] - 0.5).abs() <= 1.0e-9);
        let (w_tight, mu_tight) =
            solve_uplo_norm_bounded(1, &[1.0], &[1.0e12], 1, 1.0e-12, 1.0e-3).unwrap();
        assert!(mu_tight >= 1.0e5 && w_tight[0].is_finite());
    }

    proptest! {
        #[test]
        fn prop_cholesky_solves_random_spd_systems_to_machine_precision(
            a00 in 0.5_f64..3.0,
            a01 in -1.5_f64..1.5,
            a02 in -1.5_f64..1.5,
            a11 in 0.5_f64..3.0,
            a12 in -1.5_f64..1.5,
            a22 in 0.5_f64..3.0,
            x0 in -5.0_f64..5.0,
            x1 in -5.0_f64..5.0,
            x2 in -5.0_f64..5.0,
        ) {
            // Form SPD G = A^T A + 0.25 I from upper-triangular A.
            let rows = [[a00, a01, a02], [0.0, a11, a12], [0.0, 0.0, a22]];
            let mut gram = vec![0.0_f64; uplo_u_packed_len(3)];
            for row in &rows {
                for j in 0..3 {
                    for i in 0..=j {
                        gram[uplo_u_index(i, j)] += row[i] * row[j];
                    }
                }
            }
            let w_true = [x0, x1, x2];
            let rhs = symv_uplo_u(3, &gram, &w_true);
            let w_sol = solve_uplo_u_cholesky(&gram, &rhs, 1.0e-12).unwrap();
            let lhs = symv_uplo_u(3, &gram, &w_sol);
            for (a, b) in lhs.iter().zip(&rhs) {
                prop_assert!((a - b).abs() <= 1.0e-10);
            }
        }
    }
}
