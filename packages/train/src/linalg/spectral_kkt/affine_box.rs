//! Active-set augmented-Lagrangian Newton-KKT solver for quadratic MSE + ridge
//! subject to scaled affine `Welch-33` RMS box constraints.

use crate::error::TrainError;
use crate::linalg::cholesky::{solve_uplo_u_cholesky, symv_uplo_u};
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::{KKT_PRIMAL_RIDGE_MIN, LtasKktConfig, LtasKktResult};

/// Affine-box quadratic problem `0.5 w^T G w - rhs^T w + 0.5 ridge ||w||_2^2` subject to
/// `band_lo[b] <= band_const[b] + band_rows[b]^T w <= band_hi[b]` (`b = 0..num_bands`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AffineBoxKktProblem<'a> {
    pub num_bands: usize,
    pub feature_dim: usize,
    pub gram_uplo: &'a [f64],
    pub rhs: &'a [f64],
    pub band_rows: &'a [f64],
    pub band_const: &'a [f64],
    pub band_lo: &'a [f64],
    pub band_hi: &'a [f64],
}

fn validate_affine_box_problem(
    prob: &AffineBoxKktProblem<'_>,
    cfg: LtasKktConfig,
) -> Result<(), TrainError> {
    if prob.num_bands == 0 || prob.feature_dim == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "solve_ltas_affine_box_kkt",
            expected: 1,
            actual: 0,
        });
    }
    let packed = uplo_u_packed_len(prob.feature_dim);
    let expected_rows = prob.num_bands.saturating_mul(prob.feature_dim);
    if prob.gram_uplo.len() < packed
        || prob.rhs.len() < prob.feature_dim
        || prob.band_rows.len() < expected_rows
        || prob.band_const.len() < prob.num_bands
        || prob.band_lo.len() < prob.num_bands
        || prob.band_hi.len() < prob.num_bands
    {
        return Err(TrainError::InvalidDimensions {
            context: "solve_ltas_affine_box_kkt",
            expected: packed.max(expected_rows),
            actual: prob.gram_uplo.len().min(prob.band_rows.len()),
        });
    }
    let scalars_ok = cfg.ridge.is_finite()
        && cfg.ridge >= 0.0
        && cfg.penalty_rho.is_finite()
        && cfg.penalty_rho >= 0.0
        && cfg.max_tilt_db.is_finite()
        && prob.gram_uplo[..packed].iter().all(|v| v.is_finite())
        && prob.rhs[..prob.feature_dim].iter().all(|v| v.is_finite())
        && prob.band_rows[..expected_rows]
            .iter()
            .all(|v| v.is_finite())
        && prob.band_const[..prob.num_bands]
            .iter()
            .all(|v| v.is_finite())
        && prob.band_lo[..prob.num_bands].iter().all(|v| v.is_finite())
        && prob.band_hi[..prob.num_bands].iter().all(|v| v.is_finite());
    if !scalars_ok {
        return Err(TrainError::NonFiniteInput {
            context: "solve_ltas_affine_box_kkt",
        });
    }
    Ok(())
}

fn scale_affine_box_rows(prob: &AffineBoxKktProblem<'_>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut diag_sum = 0.0_f64;
    let mut diag_cnt = 0_usize;
    for j in 0..prob.feature_dim {
        let d = prob.gram_uplo[uplo_u_index(j, j)];
        if d > 0.0 {
            diag_sum += d;
            diag_cnt += 1;
        }
    }
    let diag_mean = if diag_cnt > 0 {
        (diag_sum / (diag_cnt as f64)).max(KKT_PRIMAL_RIDGE_MIN)
    } else {
        1.0
    };

    let mut max_norm_sq = 0.0_f64;
    for b in 0..prob.num_bands {
        let row = &prob.band_rows[b * prob.feature_dim..(b + 1) * prob.feature_dim];
        let norm_sq: f64 = row.iter().map(|&v| v * v).sum();
        if norm_sq.is_finite() {
            max_norm_sq = max_norm_sq.max(norm_sq);
        }
    }
    let global_s = (diag_mean / max_norm_sq.max(1.0e-30)).sqrt();
    let mut row_scales = vec![0.0_f64; prob.num_bands];
    let mut lo_hat = vec![0.0_f64; prob.num_bands];
    let mut hi_hat = vec![0.0_f64; prob.num_bands];
    for b in 0..prob.num_bands {
        let row = &prob.band_rows[b * prob.feature_dim..(b + 1) * prob.feature_dim];
        let norm_sq: f64 = row.iter().map(|&v| v * v).sum();
        if norm_sq > 1.0e-12 * max_norm_sq && norm_sq > 1.0e-30 {
            row_scales[b] = global_s;
            let lo_v = prob.band_lo[b].min(prob.band_hi[b]);
            let hi_v = prob.band_hi[b].max(prob.band_lo[b]);
            lo_hat[b] = global_s * (lo_v - prob.band_const[b]);
            hi_hat[b] = global_s * (hi_v - prob.band_const[b]);
        }
    }
    (row_scales, lo_hat, hi_hat)
}

#[allow(clippy::needless_range_loop)]
fn eval_box_al_value(st: &InnerBoxState<'_>, w: &[f64]) -> f64 {
    let prob = st.prob;
    let packed = uplo_u_packed_len(prob.feature_dim);
    let gw = symv_uplo_u(prob.feature_dim, &prob.gram_uplo[..packed], w);
    let q: f64 = w
        .iter()
        .zip(&gw)
        .zip(&prob.rhs[..prob.feature_dim])
        .map(|((&wi, &gi), &ri)| 0.5 * (wi * gi + st.ridge * wi * wi) - wi * ri)
        .sum();
    let mut pen = 0.0_f64;
    for b in 0..prob.num_bands {
        if st.row_scales[b] <= 0.0 {
            continue;
        }
        let row = &prob.band_rows[b * prob.feature_dim..(b + 1) * prob.feature_dim];
        let dot_w: f64 = row.iter().zip(w).map(|(&r, &wv)| r * wv).sum();
        let vh = st.row_scales[b] * dot_w;
        let pu = (st.mu_up[b] + st.rho * (vh - st.hi_hat[b])).max(0.0);
        let pl = (st.mu_lo[b] + st.rho * (st.lo_hat[b] - vh)).max(0.0);
        pen += 0.5 * (pu * pu + pl * pl) / st.rho;
    }
    q + pen
}

struct InnerBoxState<'a> {
    prob: &'a AffineBoxKktProblem<'a>,
    row_scales: &'a [f64],
    lo_hat: &'a [f64],
    hi_hat: &'a [f64],
    w: &'a mut Vec<f64>,
    mu_up: &'a [f64],
    mu_lo: &'a [f64],
    rho: f64,
    ridge: f64,
}

fn solve_inner_box_newton(st: &mut InnerBoxState<'_>) -> Result<(), TrainError> {
    let dim = st.prob.feature_dim;
    let packed = uplo_u_packed_len(dim);
    let mut gram_work = vec![0.0_f64; packed];
    let mut rhs_work = vec![0.0_f64; dim];
    let mut grad_w = vec![0.0_f64; dim];
    let mut act_up = vec![false; st.prob.num_bands];
    let mut act_lo = vec![false; st.prob.num_bands];

    let mut cur_al = eval_box_al_value(st, st.w);
    for inner in 0..12 {
        let mut changed = false;
        gram_work.copy_from_slice(&st.prob.gram_uplo[..packed]);
        rhs_work.copy_from_slice(&st.prob.rhs[..dim]);
        let gw = symv_uplo_u(dim, &st.prob.gram_uplo[..packed], st.w);
        for j in 0..dim {
            grad_w[j] = gw[j] + st.ridge * st.w[j] - st.prob.rhs[j];
        }

        for b in 0..st.prob.num_bands {
            if st.row_scales[b] <= 0.0 {
                act_up[b] = false;
                act_lo[b] = false;
                continue;
            }
            let s = st.row_scales[b];
            let row = &st.prob.band_rows[b * dim..(b + 1) * dim];
            let dot_w: f64 = row.iter().zip(st.w.iter()).map(|(&r, &wv)| r * wv).sum();
            let val_hat = s * dot_w;
            let eff_up = st.mu_up[b] + st.rho * (val_hat - st.hi_hat[b]);
            let eff_lo = st.mu_lo[b] + st.rho * (st.lo_hat[b] - val_hat);
            let need_up = eff_up > 0.0;
            let need_lo = !need_up && (eff_lo > 0.0);
            if need_up != act_up[b] || need_lo != act_lo[b] {
                changed = true;
            }
            act_up[b] = need_up;
            act_lo[b] = need_lo;
            if !need_up && !need_lo {
                continue;
            }
            let rhs_coeff = if need_up {
                st.rho * st.hi_hat[b] - st.mu_up[b]
            } else {
                st.rho * st.lo_hat[b] + st.mu_lo[b]
            };
            let grad_coeff = if need_up { eff_up } else { -eff_lo };
            for j in 0..dim {
                let aj = s * row[j];
                rhs_work[j] += rhs_coeff * aj;
                grad_w[j] += grad_coeff * aj;
                let rho_aj = st.rho * aj;
                for i in 0..=j {
                    gram_work[uplo_u_index(i, j)] += rho_aj * (s * row[i]);
                }
            }
        }
        if !changed && inner > 0 {
            break;
        }
        let w_full = solve_uplo_u_cholesky(&gram_work, &rhs_work, st.ridge)?;
        let grad_dot_d: f64 = grad_w
            .iter()
            .zip(&w_full)
            .zip(st.w.iter())
            .map(|((&gj, &wn), &wo)| gj * (wn - wo))
            .sum();
        let armijo_floor = 1.0e-12 * cur_al.abs().max(1.0);
        let mut accepted = false;
        for &ls in &[1.0_f64, 0.5, 0.25, 0.125, 0.0625] {
            let w_try: Vec<f64> =
                st.w.iter()
                    .zip(&w_full)
                    .map(|(&old_v, &new_v)| old_v + ls * (new_v - old_v))
                    .collect();
            let try_al = eval_box_al_value(st, &w_try);
            if try_al <= cur_al + 1.0e-4 * ls * grad_dot_d.min(0.0) + armijo_floor {
                let step_sq: f64 = w_try
                    .iter()
                    .zip(st.w.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                *st.w = w_try;
                cur_al = try_al;
                accepted = step_sq > 1.0e-20;
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    Ok(())
}

fn compute_box_kkt_metrics(
    prob: &AffineBoxKktProblem<'_>,
    row_scales: &[f64],
    lo_hat: &[f64],
    hi_hat: &[f64],
    w: &[f64],
    mu_up: &[f64],
    mu_lo: &[f64],
) -> (f64, f64) {
    let mut max_exceed_db = 0.0_f64;
    let mut kkt_comp = 0.0_f64;
    for b in 0..prob.num_bands {
        if row_scales[b] <= 0.0 {
            continue;
        }
        let row = &prob.band_rows[b * prob.feature_dim..(b + 1) * prob.feature_dim];
        let dot_w: f64 = row.iter().zip(w).map(|(&r, &wv)| r * wv).sum();
        let pred_v = (prob.band_const[b] + dot_w).abs().max(1.0e-18);
        let lo_v = prob.band_lo[b].min(prob.band_hi[b]).abs().max(1.0e-18);
        let hi_v = prob.band_hi[b].max(prob.band_lo[b]).abs().max(1.0e-18);
        if pred_v > hi_v {
            max_exceed_db = max_exceed_db.max(20.0 * (pred_v / hi_v).log10());
        } else if pred_v < lo_v {
            max_exceed_db = max_exceed_db.max(20.0 * (lo_v / pred_v).log10());
        }
        let val_hat = row_scales[b] * dot_w;
        kkt_comp = kkt_comp
            .max((mu_up[b] * (val_hat - hi_hat[b])).abs())
            .max((mu_lo[b] * (lo_hat[b] - val_hat)).abs());
    }
    (max_exceed_db, kkt_comp)
}

/// Active-set augmented-Lagrangian Newton-KKT solver for quadratic MSE + ridge subject to affine
/// `Welch-33` RMS box bounds (`ref[train.pipeline.nested-varpro-schur]`).
pub(crate) fn solve_ltas_affine_box_kkt(
    prob: &AffineBoxKktProblem<'_>,
    cfg: LtasKktConfig,
) -> Result<LtasKktResult, TrainError> {
    validate_affine_box_problem(prob, cfg)?;
    let (row_scales, lo_hat, hi_hat) = scale_affine_box_rows(prob);
    let packed = uplo_u_packed_len(prob.feature_dim);
    let ridge = cfg.ridge.max(KKT_PRIMAL_RIDGE_MIN);
    let mut mu_up = vec![0.0_f64; prob.num_bands];
    let mut mu_lo = vec![0.0_f64; prob.num_bands];
    let mut w = solve_uplo_u_cholesky(
        &prob.gram_uplo[..packed],
        &prob.rhs[..prob.feature_dim],
        ridge,
    )?;

    let rho_init = cfg.penalty_rho.max(0.25);
    let mut rho = rho_init;
    let rho_cap = (rho_init * 32.0).min(1.0e5);
    let mut prev_viol_hat = f64::INFINITY;
    let max_iters = cfg.max_outer_iters.max(1);

    for outer in 0..max_iters {
        let mut st = InnerBoxState {
            prob,
            row_scales: &row_scales,
            lo_hat: &lo_hat,
            hi_hat: &hi_hat,
            w: &mut w,
            mu_up: &mu_up,
            mu_lo: &mu_lo,
            rho,
            ridge,
        };
        solve_inner_box_newton(&mut st)?;

        let mut max_viol_hat = 0.0_f64;
        for b in 0..prob.num_bands {
            if row_scales[b] <= 0.0 {
                continue;
            }
            let row = &prob.band_rows[b * prob.feature_dim..(b + 1) * prob.feature_dim];
            let dot_w: f64 = row.iter().zip(&w).map(|(&r, &wv)| r * wv).sum();
            let val_hat = row_scales[b] * dot_w;
            max_viol_hat = max_viol_hat
                .max((val_hat - hi_hat[b]).max(0.0))
                .max((lo_hat[b] - val_hat).max(0.0));
        }
        if max_viol_hat <= 1.0e-9 || (outer >= 2 && max_viol_hat > 0.98 * prev_viol_hat) {
            break;
        }
        prev_viol_hat = max_viol_hat;
        for b in 0..prob.num_bands {
            if row_scales[b] <= 0.0 {
                continue;
            }
            let row = &prob.band_rows[b * prob.feature_dim..(b + 1) * prob.feature_dim];
            let dot_w: f64 = row.iter().zip(&w).map(|(&r, &wv)| r * wv).sum();
            let val_hat = row_scales[b] * dot_w;
            mu_up[b] = (mu_up[b] + rho * (val_hat - hi_hat[b])).max(0.0);
            mu_lo[b] = (mu_lo[b] + rho * (lo_hat[b] - val_hat)).max(0.0);
        }
        rho = (rho * 2.0).min(rho_cap);
    }

    let (max_exceed_db, kkt_comp) =
        compute_box_kkt_metrics(prob, &row_scales, &lo_hat, &hi_hat, &w, &mu_up, &mu_lo);
    Ok(LtasKktResult {
        weights: w,
        dual_upper: mu_up,
        dual_lower: mu_lo,
        max_tilt_db: max_exceed_db,
        kkt_complementarity_residual: kkt_comp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_ltas_affine_box_kkt_enforces_scaled_bounds_and_errors() {
        let empty = AffineBoxKktProblem {
            num_bands: 0,
            feature_dim: 1,
            gram_uplo: &[],
            rhs: &[],
            band_rows: &[],
            band_const: &[],
            band_lo: &[],
            band_hi: &[],
        };
        assert!(matches!(
            solve_ltas_affine_box_kkt(&empty, LtasKktConfig::default()),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let nan_prob = AffineBoxKktProblem {
            num_bands: 1,
            feature_dim: 1,
            gram_uplo: &[1.0],
            rhs: &[1.0],
            band_rows: &[1.0],
            band_const: &[0.0],
            band_lo: &[0.5],
            band_hi: &[f64::NAN],
        };
        assert!(matches!(
            solve_ltas_affine_box_kkt(&nan_prob, LtasKktConfig::default()),
            Err(TrainError::NonFiniteInput { .. })
        ));

        let scale = 1.0e4_f64;
        let gram = [scale * 2.0, 0.0, scale * 1.0];
        let rhs = [scale * 3.0, scale * 0.5];
        let band_rows = [1.0e-3, 0.0];
        let band_const = [0.25];
        let band_lo = [0.2508];
        let band_hi = [0.2512];
        let res = solve_ltas_affine_box_kkt(
            &AffineBoxKktProblem {
                num_bands: 1,
                feature_dim: 2,
                gram_uplo: &gram,
                rhs: &rhs,
                band_rows: &band_rows,
                band_const: &band_const,
                band_lo: &band_lo,
                band_hi: &band_hi,
            },
            LtasKktConfig {
                ridge: 1.0e-10,
                max_outer_iters: 12,
                max_tilt_db: 0.1,
                penalty_rho: 8.0,
            },
        )
        .unwrap();
        let val0 = band_const[0] + band_rows[0] * res.weights[0] + band_rows[1] * res.weights[1];
        assert!(
            val0 >= band_lo[0] - 1.0e-7 && val0 <= band_hi[0] + 1.0e-7,
            "val0={val0} must lie inside [{}, {}]",
            band_lo[0],
            band_hi[0]
        );
        assert!(res.max_tilt_db < 1.0e-4, "max_tilt_db={}", res.max_tilt_db);
        assert!(res.kkt_complementarity_residual < 1.0e-6);

        // Coupled 2-row box exercises active-set switching on both upper and lower bounds.
        let coupled_rows = [1.0, 1.0, 0.0, 1.0];
        let coupled_const = [0.0, 0.0];
        let coupled_lo = [-10.0, 0.8];
        let coupled_hi = [1.8, 2.0];
        let coupled = solve_ltas_affine_box_kkt(
            &AffineBoxKktProblem {
                num_bands: 2,
                feature_dim: 2,
                gram_uplo: &[2.0, 0.0, 2.0],
                rhs: &[4.0, 1.0],
                band_rows: &coupled_rows,
                band_const: &coupled_const,
                band_lo: &coupled_lo,
                band_hi: &coupled_hi,
            },
            LtasKktConfig {
                ridge: 1.0e-10,
                max_outer_iters: 12,
                max_tilt_db: 0.1,
                penalty_rho: 8.0,
            },
        )
        .unwrap();
        assert!(coupled.max_tilt_db < 1.0e-4, "tilt={}", coupled.max_tilt_db);
        assert!(coupled.weights[0] + coupled.weights[1] <= 1.8 + 1.0e-6);
        assert!(coupled.weights[1] >= 0.8 - 1.0e-6);
    }
}
