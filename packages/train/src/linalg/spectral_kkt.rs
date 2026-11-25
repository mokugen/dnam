//! 1/12-octave Excitation-Masking Ratio (`MM-Cholesky`) and active-set 33-band LTAS
//! Newton-KKT linear solvers (`ref[train.pipeline.nested-varpro-schur]`).

mod affine_box;

use crate::error::TrainError;
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::cholesky::{solve_uplo_adaptive_ridge, solve_uplo_u_cholesky, symv_uplo_u};
pub(crate) use affine_box::{AffineBoxKktProblem, solve_ltas_affine_box_kkt};

const DEFAULT_DB_NOISE_FLOOR: f64 = 1.0e-12;
const EMR_LOCK_HZ: f64 = 1_500.0;
const EMR_PHASE_LOCK_DB: f64 = 0.25;
const EMR_MIN_PHASE_ALPHA: f64 = 0.5;
pub(super) const KKT_PRIMAL_RIDGE_MIN: f64 = 1.0e-10;

/// Configuration for 1/12-octave Excitation/Masking weighted `MM-Cholesky` (`ref[train.pipeline.nested-varpro-schur]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmrMmConfig {
    /// Diagonal Tikhonov ridge floor.
    pub ridge: f64,
    /// Number of MM refinement passes (`>= 1`).
    pub max_iters: usize,
    /// ERB spreading width across adjacent 1/12-octave bands in bands (`default = 2.0`).
    pub masking_spread_bands: f64,
    /// Slope relaxing `alpha(omega) in [0.5, 1.0)` above `1.5 kHz` per dB over `0.25 dB`.
    pub phase_relax_slope: f64,
}

impl Default for EmrMmConfig {
    fn default() -> Self {
        Self {
            ridge: 1.0e-8,
            max_iters: 4,
            masking_spread_bands: 2.0,
            phase_relax_slope: 0.25,
        }
    }
}

/// Output from [`solve_emr_mm_cholesky`].
#[derive(Clone, Debug, PartialEq)]
pub struct EmrMmResult {
    /// Fitted weight vector `w`.
    pub weights: Vec<f64>,
    /// Pre-baked effective normal-equation RHS `c_eff = (G + Lambda) w`.
    pub effective_rhs: Vec<f64>,
    /// Inverse-masking weights `1 / sqrt(M_b)` used per 1/12-octave band.
    pub inv_masking_sqrt: Vec<f64>,
    /// Effective per-band phase lock `alpha_b in [0.5, 1.0]`.
    pub phase_lock_alpha: Vec<f64>,
}

/// Complex spectral projection matrices bundled for [`solve_emr_mm_cholesky`].
#[derive(Clone, Copy, Debug)]
pub struct EmrSpectralProblem<'a> {
    /// Number of 1/12-octave frequency bands.
    pub num_bands: usize,
    /// Feature basis dimension.
    pub feature_dim: usize,
    /// Band center frequencies in Hz (`num_bands`).
    pub band_freqs_hz: &'a [f64],
    /// Real part of row-major basis matrix `[num_bands][feature_dim]`.
    pub basis_re: &'a [f64],
    /// Imaginary part of row-major basis matrix `[num_bands][feature_dim]`.
    pub basis_im: &'a [f64],
    /// Real part of target complex spectrum (`num_bands`).
    pub target_re: &'a [f64],
    /// Imaginary part of target complex spectrum (`num_bands`).
    pub target_im: &'a [f64],
}

/// Solves a 1/12-octave Excitation-Masking Ratio (`M_b^{-1/2}`) `MM-Cholesky` spectral projection
/// problem with adaptive phase lock (`1.0` where `Delta L_b <= 0.25 dB`, relaxing in `[0.5, 1.0)`
/// only above `1.5 kHz`).
pub fn solve_emr_mm_cholesky(
    problem: &EmrSpectralProblem<'_>,
    config: EmrMmConfig,
) -> Result<EmrMmResult, TrainError> {
    let num_bands = problem.num_bands;
    let feature_dim = problem.feature_dim;
    if num_bands == 0 || feature_dim == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "solve_emr_mm_cholesky",
            expected: 1,
            actual: 0,
        });
    }
    let mat_len = num_bands * feature_dim;
    if problem.band_freqs_hz.len() != num_bands
        || problem.basis_re.len() != mat_len
        || problem.basis_im.len() != mat_len
        || problem.target_re.len() != num_bands
        || problem.target_im.len() != num_bands
    {
        return Err(TrainError::InvalidDimensions {
            context: "solve_emr_mm_cholesky(inputs)",
            expected: mat_len,
            actual: problem.basis_re.len().min(problem.basis_im.len()),
        });
    }
    if !config.masking_spread_bands.is_finite()
        || !config.ridge.is_finite()
        || config.ridge < 0.0
        || !config.phase_relax_slope.is_finite()
        || config.phase_relax_slope < 0.0
        || !problem.band_freqs_hz.iter().all(|v| v.is_finite())
        || !problem.basis_re.iter().all(|v| v.is_finite())
        || !problem.basis_im.iter().all(|v| v.is_finite())
        || !problem.target_re.iter().all(|v| v.is_finite())
        || !problem.target_im.iter().all(|v| v.is_finite())
    {
        return Err(TrainError::NonFiniteInput {
            context: "solve_emr_mm_cholesky",
        });
    }

    let spread = config.masking_spread_bands.max(0.25);
    let target_pow: Vec<f64> = problem
        .target_re
        .iter()
        .zip(problem.target_im)
        .map(|(&re, &im)| re * re + im * im)
        .collect();
    let mut inv_masking_sqrt = vec![1.0_f64; num_bands];
    let mut band_weights = vec![1.0_f64; num_bands];
    for b in 0..num_bands {
        let mut mask_pow = DEFAULT_DB_NOISE_FLOOR;
        for (b2, &tp) in target_pow.iter().enumerate() {
            let db = (b as f64) - (b2 as f64);
            mask_pow += (-0.5 * (db / spread) * (db / spread)).exp() * tp;
        }
        band_weights[b] = 1.0 / mask_pow;
        inv_masking_sqrt[b] = band_weights[b].sqrt();
    }

    let mut gram = vec![0.0_f64; uplo_u_packed_len(feature_dim)];
    for (b, &qb) in band_weights.iter().enumerate() {
        let row_re = &problem.basis_re[b * feature_dim..(b + 1) * feature_dim];
        let row_im = &problem.basis_im[b * feature_dim..(b + 1) * feature_dim];
        for j in 0..feature_dim {
            let col_j = uplo_u_index(0, j);
            for i in 0..=j {
                gram[col_j + i] += qb * (row_re[i] * row_re[j] + row_im[i] * row_im[j]);
            }
        }
    }

    let mut rhs = vec![0.0_f64; feature_dim];
    let mut alpha = vec![1.0_f64; num_bands];
    let mut weights = vec![0.0_f64; feature_dim];
    let mut applied_ridge = config.ridge.max(1.0e-12);
    let iters = config.max_iters.max(1);

    for iter in 0..iters {
        rhs.fill(0.0);
        for b in 0..num_bands {
            let row_re = &problem.basis_re[b * feature_dim..(b + 1) * feature_dim];
            let row_im = &problem.basis_im[b * feature_dim..(b + 1) * feature_dim];
            let mut tgt_re = problem.target_re[b];
            let mut tgt_im = problem.target_im[b];

            if iter > 0 {
                let mut pred_re = 0.0_f64;
                let mut pred_im = 0.0_f64;
                for d in 0..feature_dim {
                    pred_re += row_re[d] * weights[d];
                    pred_im += row_im[d] * weights[d];
                }
                let pred_mag = (pred_re * pred_re + pred_im * pred_im).sqrt();
                let tgt_mag = target_pow[b].sqrt();
                let dl_db = (10.0
                    * ((pred_mag * pred_mag + DEFAULT_DB_NOISE_FLOOR)
                        / (tgt_mag * tgt_mag + DEFAULT_DB_NOISE_FLOOR))
                        .log10())
                .abs();
                let a = if problem.band_freqs_hz[b] > EMR_LOCK_HZ && dl_db > EMR_PHASE_LOCK_DB {
                    (1.0 - config.phase_relax_slope * (dl_db - EMR_PHASE_LOCK_DB))
                        .clamp(EMR_MIN_PHASE_ALPHA, 1.0)
                } else {
                    1.0
                };
                alpha[b] = a;
                if a < 1.0 && pred_mag > 1.0e-12 && tgt_mag > 1.0e-12 {
                    let ur = a * (tgt_re / tgt_mag) + (1.0 - a) * (pred_re / pred_mag);
                    let ui = a * (tgt_im / tgt_mag) + (1.0 - a) * (pred_im / pred_mag);
                    let un = (ur * ur + ui * ui).sqrt().max(1.0e-12);
                    tgt_re = tgt_mag * ur / un;
                    tgt_im = tgt_mag * ui / un;
                }
            }

            let qb = band_weights[b];
            for d in 0..feature_dim {
                rhs[d] += qb * (row_re[d] * tgt_re + row_im[d] * tgt_im);
            }
        }
        let (sol, used_r) = solve_uplo_adaptive_ridge(
            feature_dim,
            &gram,
            &rhs,
            &[],
            None,
            config.ridge.max(1.0e-12),
        )?;
        weights = sol;
        applied_ridge = used_r;
    }

    let mut reg_gram = gram;
    for j in 0..feature_dim {
        reg_gram[uplo_u_index(j, j)] += applied_ridge;
    }
    let effective_rhs = symv_uplo_u(feature_dim, &reg_gram, &weights);

    Ok(EmrMmResult {
        weights,
        effective_rhs,
        inv_masking_sqrt,
        phase_lock_alpha: alpha,
    })
}

/// Configuration for active-set 33-band LTAS Newton-KKT linear solver (`ref[train.pipeline.nested-varpro-schur]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtasKktConfig {
    /// Primal diagonal regularization `ridge`.
    pub ridge: f64,
    /// Maximum outer augmented-Lagrangian / Newton-KKT dual iterations.
    pub max_outer_iters: usize,
    /// Symmetric tilt envelope limit `|Delta_b| <= max_tilt_db`.
    pub max_tilt_db: f64,
    /// Initial dual penalty parameter `rho > 0`.
    pub penalty_rho: f64,
}

impl Default for LtasKktConfig {
    fn default() -> Self {
        Self {
            ridge: 1.0e-8,
            max_outer_iters: 24,
            max_tilt_db: 0.75,
            penalty_rho: 4.0,
        }
    }
}

/// Output from [`solve_ltas_newton_kkt`].
#[derive(Clone, Debug, PartialEq)]
pub struct LtasKktResult {
    /// Constrained primal weight vector `w`.
    pub weights: Vec<f64>,
    /// Non-negative dual Lagrange multipliers (`>= 0`) for active upper band bounds.
    pub dual_upper: Vec<f64>,
    /// Non-negative dual Lagrange multipliers (`>= 0`) for active lower band bounds.
    pub dual_lower: Vec<f64>,
    /// Worst absolute band tilt `max_b |20 log10(|pred_b| / ref_b)|` across constrained bands (dB).
    pub max_tilt_db: f64,
    /// Maximum dual complementarity violation `max(|mu * slack|)`.
    pub kkt_complementarity_residual: f64,
}

/// Minimizes `0.5 w^T (G + ridge I) w - c^T w` subject to per-band LTAS linear amplitude bounds
/// `10^(-delta/20) * target_amp[b] <= row_b^T w <= 10^(+delta/20) * target_amp[b]` via active-set
/// Newton-KKT augmented-Lagrangian steps with guaranteed SPD Hessians.
pub fn solve_ltas_newton_kkt(
    feature_dim: usize,
    num_bands: usize,
    gram_uplo: &[f64],
    rhs: &[f64],
    band_rows: &[f64],
    band_target_amps: &[f64],
    config: LtasKktConfig,
) -> Result<LtasKktResult, TrainError> {
    if feature_dim == 0 || rhs.len() != feature_dim {
        return Err(TrainError::InvalidDimensions {
            context: "solve_ltas_newton_kkt(rhs)",
            expected: feature_dim.max(1),
            actual: rhs.len(),
        });
    }
    if gram_uplo.len() != uplo_u_packed_len(feature_dim) {
        return Err(TrainError::InvalidDimensions {
            context: "solve_ltas_newton_kkt(gram_uplo)",
            expected: uplo_u_packed_len(feature_dim),
            actual: gram_uplo.len(),
        });
    }
    if band_rows.len() != num_bands * feature_dim || band_target_amps.len() != num_bands {
        return Err(TrainError::InvalidDimensions {
            context: "solve_ltas_newton_kkt(bands)",
            expected: num_bands * feature_dim,
            actual: band_rows.len(),
        });
    }
    if !config.ridge.is_finite()
        || config.ridge < 0.0
        || !config.max_tilt_db.is_finite()
        || config.max_tilt_db < 0.0
        || !config.penalty_rho.is_finite()
        || config.penalty_rho <= 0.0
        || !gram_uplo.iter().all(|v| v.is_finite())
        || !rhs.iter().all(|v| v.is_finite())
        || !band_rows.iter().all(|v| v.is_finite())
        || !band_target_amps.iter().all(|v| v.is_finite() && *v >= 0.0)
    {
        return Err(TrainError::NonFiniteInput {
            context: "solve_ltas_newton_kkt",
        });
    }

    let gain_hi = 10.0_f64.powf(config.max_tilt_db.max(0.01) / 20.0);
    let gain_lo = 1.0 / gain_hi;
    let ridge = config.ridge.max(KKT_PRIMAL_RIDGE_MIN);

    let mut w = solve_uplo_u_cholesky(gram_uplo, rhs, ridge)?;
    let mut mu_up = vec![0.0_f64; num_bands];
    let mut mu_lo = vec![0.0_f64; num_bands];
    let mut rho = config.penalty_rho.max(0.5);
    let mut active_up = vec![false; num_bands];
    let mut active_lo = vec![false; num_bands];

    for _outer in 0..config.max_outer_iters.max(1) {
        // Inner active-set Newton convergence for the current dual estimate (mu_up, mu_lo, rho).
        for _inner in 0..8 {
            let mut h_kkt = gram_uplo.to_vec();
            let mut r_kkt = rhs.to_vec();
            let mut changed = false;

            for b in 0..num_bands {
                let row = &band_rows[b * feature_dim..(b + 1) * feature_dim];
                let pred_amp: f64 = row.iter().zip(&w).map(|(&a, &wi)| a * wi).sum();
                let lo = gain_lo * band_target_amps[b].max(1.0e-12);
                let hi = gain_hi * band_target_amps[b].max(1.0e-12);

                let v_up = pred_amp - hi;
                let a_up = mu_up[b] + rho * v_up > 0.0;
                if a_up != active_up[b] {
                    active_up[b] = a_up;
                    changed = true;
                }
                if a_up {
                    for j in 0..feature_dim {
                        let col_j = uplo_u_index(0, j);
                        for i in 0..=j {
                            h_kkt[col_j + i] += rho * row[i] * row[j];
                        }
                        r_kkt[j] += (rho * hi - mu_up[b]) * row[j];
                    }
                }
                let v_lo = lo - pred_amp;
                let a_lo = !a_up && (mu_lo[b] + rho * v_lo > 0.0);
                if a_lo != active_lo[b] {
                    active_lo[b] = a_lo;
                    changed = true;
                }
                if a_lo {
                    for j in 0..feature_dim {
                        let col_j = uplo_u_index(0, j);
                        for i in 0..=j {
                            h_kkt[col_j + i] += rho * row[i] * row[j];
                        }
                        r_kkt[j] += (rho * lo + mu_lo[b]) * row[j];
                    }
                }
            }

            let w_next = solve_uplo_u_cholesky(&h_kkt, &r_kkt, ridge)?;
            let step_sq: f64 = w_next.iter().zip(&w).map(|(a, b)| (a - b) * (a - b)).sum();
            w = w_next;
            if !changed || step_sq <= 1.0e-20 {
                break;
            }
        }

        let mut worst_viol = 0.0_f64;
        for b in 0..num_bands {
            let row = &band_rows[b * feature_dim..(b + 1) * feature_dim];
            let pred_amp: f64 = row.iter().zip(&w).map(|(&a, &wi)| a * wi).sum();
            let lo = gain_lo * band_target_amps[b].max(1.0e-12);
            let hi = gain_hi * band_target_amps[b].max(1.0e-12);
            let v_up = pred_amp - hi;
            let v_lo = lo - pred_amp;
            mu_up[b] = (mu_up[b] + rho * v_up).max(0.0);
            mu_lo[b] = (mu_lo[b] + rho * v_lo).max(0.0);
            worst_viol = worst_viol.max(v_up).max(v_lo);
        }
        if worst_viol <= 1.0e-7 {
            break;
        }
        rho = (rho * 2.0).min(1.0e6);
    }

    let mut max_tilt_db = 0.0_f64;
    let mut kkt_comp = 0.0_f64;
    for b in 0..num_bands {
        let row = &band_rows[b * feature_dim..(b + 1) * feature_dim];
        let pred_amp: f64 = row.iter().zip(&w).map(|(&a, &wi)| a * wi).sum();
        let ref_amp = band_target_amps[b].max(1.0e-12);
        let tilt = if pred_amp <= 0.0 {
            f64::INFINITY
        } else {
            (20.0 * (pred_amp / ref_amp).log10()).abs()
        };
        max_tilt_db = max_tilt_db.max(tilt);
        let lo = gain_lo * ref_amp;
        let hi = gain_hi * ref_amp;
        kkt_comp = kkt_comp
            .max((mu_up[b] * (pred_amp - hi)).abs())
            .max((mu_lo[b] * (lo - pred_amp)).abs());
    }

    Ok(LtasKktResult {
        weights: w,
        dual_upper: mu_up,
        dual_lower: mu_lo,
        max_tilt_db,
        kkt_complementarity_residual: kkt_comp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emr_mm_cholesky_and_ltas_newton_kkt() {
        let bands = [500.0_f64, 4_000.0_f64];
        let b_re = [1.0, 0.0, 0.25, 1.0];
        let b_im = [0.0, 0.0, 0.96, 0.0];
        let t_re = [1.0, -0.5];
        let t_im = [0.0, 0.85];
        let emr = solve_emr_mm_cholesky(
            &EmrSpectralProblem {
                num_bands: 2,
                feature_dim: 2,
                band_freqs_hz: &bands,
                basis_re: &b_re,
                basis_im: &b_im,
                target_re: &t_re,
                target_im: &t_im,
            },
            EmrMmConfig::default(),
        )
        .unwrap();
        assert_eq!(emr.phase_lock_alpha[0], 1.0);
        assert!(emr.phase_lock_alpha[1] >= 0.5 && emr.phase_lock_alpha[1] < 1.0);
        assert_eq!(emr.effective_rhs.len(), 2);

        let gram = [10.0, 0.0, 10.0];
        let rhs = [25.0, 2.0];
        let band_rows = [1.0, 0.0, 0.0, 1.0];
        let tgt_amps = [1.0, 1.0];
        let kkt = solve_ltas_newton_kkt(
            2,
            2,
            &gram,
            &rhs,
            &band_rows,
            &tgt_amps,
            LtasKktConfig {
                max_tilt_db: 0.75,
                max_outer_iters: 32,
                ..LtasKktConfig::default()
            },
        )
        .unwrap();
        assert!(kkt.max_tilt_db <= 0.75 + 1.0e-4, "tilt={}", kkt.max_tilt_db);
        assert!(
            kkt.kkt_complementarity_residual <= 1.0e-5,
            "kkt={}",
            kkt.kkt_complementarity_residual
        );
    }

    #[test]
    fn test_spectral_kkt_dimension_and_nonfinite_guards() {
        let empty = EmrSpectralProblem {
            num_bands: 0,
            feature_dim: 2,
            band_freqs_hz: &[],
            basis_re: &[],
            basis_im: &[],
            target_re: &[],
            target_im: &[],
        };
        assert!(matches!(
            solve_emr_mm_cholesky(&empty, EmrMmConfig::default()),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let bad_len = EmrSpectralProblem {
            num_bands: 1,
            feature_dim: 2,
            band_freqs_hz: &[100.0],
            basis_re: &[1.0],
            basis_im: &[0.0, 0.0],
            target_re: &[1.0],
            target_im: &[0.0],
        };
        assert!(matches!(
            solve_emr_mm_cholesky(&bad_len, EmrMmConfig::default()),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let valid = EmrSpectralProblem {
            num_bands: 1,
            feature_dim: 1,
            band_freqs_hz: &[100.0],
            basis_re: &[1.0],
            basis_im: &[0.0],
            target_re: &[1.0],
            target_im: &[0.0],
        };
        assert!(matches!(
            solve_emr_mm_cholesky(
                &valid,
                EmrMmConfig {
                    ridge: -1.0,
                    ..EmrMmConfig::default()
                },
            ),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            solve_ltas_newton_kkt(0, 1, &[], &[], &[], &[1.0], LtasKktConfig::default()),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_ltas_newton_kkt(1, 1, &[], &[1.0], &[1.0], &[1.0], LtasKktConfig::default()),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_ltas_newton_kkt(1, 1, &[1.0], &[1.0], &[], &[1.0], LtasKktConfig::default()),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            solve_ltas_newton_kkt(
                1,
                1,
                &[1.0],
                &[1.0],
                &[1.0],
                &[1.0],
                LtasKktConfig {
                    penalty_rho: -1.0,
                    ..LtasKktConfig::default()
                },
            ),
            Err(TrainError::NonFiniteInput { .. })
        ));
    }
}
