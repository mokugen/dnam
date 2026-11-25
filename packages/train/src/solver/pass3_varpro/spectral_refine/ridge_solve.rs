//! Scale-invariant Marquardt diagonal-ridge `uplo='U'` linear solvers, `+6 dB` BIBO checks,
//! and channel MSE / HF-MSE evaluators for Pass-3 `VarPro` (`ref[train.pipeline.nested-varpro-schur]`).

use crate::control::clamp::{MAX_GAIN_6DB_LIN, evaluate_readout_spectral_magnitudes};
use crate::error::TrainError;
use crate::linalg::cholesky::{factorize_uplo_u_cholesky_exact_no_ridge, solve_factorized_uplo_u};
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

/// Safe pre-snap magnitude ceiling leaving headroom inside the `[-8.0, +8.0]` readout wire range.
pub(super) const MAX_READOUT_ABS_BEFORE_SNAP: f64 = 7.5;
pub(super) const MAX_CANDIDATE_ABS_BEFORE_SNAP: f64 = 7.8;
/// Maximum BIBO l1 gain norm across linear readout + FIR columns (`ref[train.audit.phase2.clamp-rail]`).
pub(super) const MAX_READOUT_L1_NORM: f64 = 48.0;
const MIN_POSITIVE_DIAG_EPS: f64 = 1.0e-12;
pub(super) const MIN_RIDGE_ABS_FLOOR: f64 = 1.0e-5;
/// Gram diagonal energy ratio above median (`16×` amplitude = `256×` power) separating
/// extreme-gain outlier columns (e.g., high-pole `ParaLRU` envelopes) from normal basis columns.
const OUTLIER_DIAG_RATIO_CAP: f64 = 256.0;
const _: () = assert!(MAX_READOUT_L1_NORM > 0.0 && MAX_READOUT_L1_NORM < 100.0);
const _: () = assert!(
    MIN_POSITIVE_DIAG_EPS > 0.0
        && MIN_POSITIVE_DIAG_EPS < MIN_RIDGE_ABS_FLOOR
        && MIN_RIDGE_ABS_FLOOR < 1.0
        && OUTLIER_DIAG_RATIO_CAP >= 16.0
        && OUTLIER_DIAG_RATIO_CAP <= 1_024.0
);

/// Evaluates mean-squared error across `tgts_post` for one channel weight vector of length `dim`.
pub(in super::super) fn compute_channel_mse(
    weights: &[f64],
    feats_post: &[f32],
    tgts_post: &[f32],
    dim: usize,
) -> f64 {
    let n = tgts_post.len();
    let m = n.max(1);
    let mut sq = 0.0_f64;
    for (t, &tgt) in tgts_post.iter().enumerate() {
        let mut pred = 0.0_f64;
        for (c, &wc) in weights.iter().enumerate().take(dim) {
            pred += wc * f64::from(feats_post[c * n + t]);
        }
        let d = f64::from(tgt) - pred;
        sq += d * d;
    }
    sq / (m as f64)
}

/// Evaluates first-difference high-frequency error MSE `mean((e[t] - e[t-1])^2)` across
/// `tgts_post` for comparing `1x` Lite vs `2x` Full high-frequency fidelity.
pub(in super::super) fn compute_channel_hf_mse(
    weights: &[f64],
    feats_post: &[f32],
    tgts_post: &[f32],
    dim: usize,
) -> f64 {
    let n = tgts_post.len();
    if n < 2 {
        return 0.0;
    }
    let mut prev_err = 0.0_f64;
    let mut sq = 0.0_f64;
    for (t, &tgt) in tgts_post.iter().enumerate() {
        let mut pred = 0.0_f64;
        for (c, &wc) in weights.iter().enumerate().take(dim) {
            pred += wc * f64::from(feats_post[c * n + t]);
        }
        let err = f64::from(tgt) - pred;
        if t > 0 {
            let de = err - prev_err;
            sq += de * de;
        }
        prev_err = err;
    }
    sq / ((n - 1) as f64)
}

/// Computes the median positive Gram diagonal across `0..dim` so extreme-gain outlier columns
/// (e.g., `ParaLRU` envelopes) never inflate regularization on small-scale `FIR` / `odd` columns.
fn median_positive_diag(gram: &[f64], dim: usize) -> f64 {
    let mut pos = Vec::with_capacity(dim);
    for j in 0..dim {
        let idx = uplo_u_index(j, j);
        if let Some(&d) = gram.get(idx)
            && d.is_finite()
            && d > MIN_POSITIVE_DIAG_EPS
        {
            pos.push(d);
        }
    }
    if pos.is_empty() {
        return MIN_POSITIVE_DIAG_EPS;
    }
    pos.sort_by(f64::total_cmp);
    pos[pos.len() / 2]
}

/// Builds an outlier-trimmed Marquardt diagonal ridge vector `(diag_ridge, scalar_ridge)` over
/// `0..dim`: columns within `OUTLIER_DIAG_RATIO_CAP * median_positive_diag` share the exact
/// inlier mean diagonal scale (identical to classic `trace / dim` when no extreme outliers
/// exist), while extreme-gain outlier columns (`g_jj > cap`) are excluded from the shared mean
/// and damped individually at `rel * g_jj`.
pub(super) fn build_scale_invariant_diag_ridge(
    gram: &[f64],
    dim: usize,
    rel: f64,
) -> (Vec<f64>, f64) {
    let rel_safe = if rel.is_finite() && rel > 0.0 {
        rel
    } else {
        1.0e-5
    };
    let med = median_positive_diag(gram, dim);
    let outlier_cap = OUTLIER_DIAG_RATIO_CAP * med;
    let diag_entry = |j: usize| -> f64 {
        gram.get(uplo_u_index(j, j))
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.0)
    };
    let mut inlier_sum = 0.0_f64;
    let mut inlier_count = 0_usize;
    for j in 0..dim {
        let g_jj = diag_entry(j);
        if g_jj <= outlier_cap {
            inlier_sum += g_jj;
            inlier_count += 1;
        }
    }
    let base_diag = (inlier_sum / (inlier_count.max(1) as f64)).max(MIN_POSITIVE_DIAG_EPS);
    let scalar_ridge = (rel_safe * base_diag).max(MIN_RIDGE_ABS_FLOOR);
    let mut diag_ridge = Vec::with_capacity(dim);
    for j in 0..dim {
        let g_jj = diag_entry(j);
        let eff_scale = if g_jj > outlier_cap { g_jj } else { base_diag };
        diag_ridge.push((rel_safe * eff_scale).max(MIN_RIDGE_ABS_FLOOR));
    }
    (diag_ridge, scalar_ridge)
}

/// Solves `(G + diag(diag_ridge)) w = rhs` in `f64` from packed `uplo='U'` storage under
/// scale-invariant Marquardt diagonal regularization at relative scale `rel`.
pub(in super::super) fn solve_diag_ridge_uplo_u(
    gram: &[f64],
    rhs: &[f64],
    dim: usize,
    rel: f64,
) -> Result<Vec<f64>, TrainError> {
    if dim == 0 {
        return Ok(Vec::new());
    }
    let packed = uplo_u_packed_len(dim);
    if gram.len() < packed || rhs.len() < dim {
        return Err(TrainError::InvalidDimensions {
            context: "solve_diag_ridge_uplo_u",
            expected: packed.max(dim),
            actual: gram.len().min(rhs.len()),
        });
    }
    let (diag_ridge, _) = build_scale_invariant_diag_ridge(gram, dim, rel);
    let mut g_reg = gram[..packed].to_vec();
    for (j, &rj) in diag_ridge.iter().enumerate() {
        g_reg[uplo_u_index(j, j)] += rj;
    }
    let r_upper = factorize_uplo_u_cholesky_exact_no_ridge(dim, &g_reg)?;
    solve_factorized_uplo_u(dim, &r_upper, &rhs[..dim])
}

/// Checks wire amplitude & `+6 dB` readout spectral ceiling (`[MAX_GAIN_6DB_LIN]`).
#[inline]
pub(super) fn within_6db_and_bounds(
    weights: &[f64],
    ro_pole_48k: f64,
    sr_hz: f64,
    max_abs_bound: f64,
) -> bool {
    if weights.len() < 2
        || weights
            .iter()
            .any(|&v| !v.is_finite() || v.abs() > max_abs_bound)
    {
        return false;
    }
    let l1_norm: f64 = weights.iter().map(|v| v.abs()).sum();
    if !l1_norm.is_finite() || l1_norm > MAX_READOUT_L1_NORM {
        return false;
    }
    let mags =
        evaluate_readout_spectral_magnitudes(&[weights[0], weights[1]], ro_pole_48k, sr_hz, 16);
    mags.iter().all(|&v| v.is_finite() && v <= MAX_GAIN_6DB_LIN)
}

/// Solves a prefix sub-system `(gram[..packed(dim)], rhs[..dim])` with escalating
/// scale-invariant Marquardt diagonal regularization `rel in [1e-5 .. 1e4]` until weights remain
/// well-conditioned inside `[-7.5, +7.5]`, respect the BIBO `l1` norm ceiling, and respect `+6 dB`,
/// returning `(weights, used_rel)`.
pub(in super::super) fn solve_bounded_uplo_u(
    gram: &[f64],
    rhs: &[f64],
    dim: usize,
    ro_pole_48k: f64,
    sr_hz: f64,
) -> Result<(Vec<f64>, f64), TrainError> {
    let packed = uplo_u_packed_len(dim);
    let g_sub = &gram[..packed];
    let r_sub = &rhs[..dim];
    let mut last_sol = vec![0.0_f64; dim];
    let mut last_rel = 1.0e-5_f64;

    for &rel in &[
        1.0e-5_f64, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4,
    ] {
        if let Ok(sol) = solve_diag_ridge_uplo_u(g_sub, r_sub, dim, rel) {
            last_rel = rel;
            last_sol = sol;
            if within_6db_and_bounds(&last_sol, ro_pole_48k, sr_hz, MAX_READOUT_ABS_BEFORE_SNAP) {
                return Ok((last_sol, rel));
            }
        }
    }

    let mut max_abs = 0.0_f64;
    let mut l1_norm = 0.0_f64;
    for v in &mut last_sol {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            let av = v.abs();
            max_abs = max_abs.max(av);
            l1_norm += av;
        }
    }
    let scale = (MAX_READOUT_ABS_BEFORE_SNAP / max_abs.max(1.0e-12))
        .min(MAX_READOUT_L1_NORM / l1_norm.max(1.0e-12))
        .min(1.0);
    for v in &mut last_sol {
        *v *= scale;
    }
    Ok((last_sol, last_rel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_bounded_uplo_u_enforces_l1_norm_and_bibo_uniform_scaling() {
        let dim = 32_usize;
        let sr = 48_000.0_f64;
        let mut gram = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut rhs = vec![0.0_f64; dim];
        // Collinear features whose unregularized inverse produces huge +/- coefficients.
        for t in 0..64 {
            let base = ((t as f64) * 0.13).sin();
            let row: Vec<f64> = (0..dim)
                .map(|c| base + 1.0e-3 * (((t * 7 + c * 3) % 11) as f64 - 5.0))
                .collect();
            let tgt = 0.4 * base + 2.0 * (((t % 2) as f64) - 0.5);
            for c in 0..dim {
                rhs[c] += row[c] * tgt;
                for r in 0..=c {
                    gram[uplo_u_index(r, c)] += row[r] * row[c];
                }
            }
        }
        let (sol, _) = solve_bounded_uplo_u(&gram, &rhs, dim, 0.0, sr).unwrap();
        let l1: f64 = sol.iter().map(|v| v.abs()).sum();
        assert!(
            within_6db_and_bounds(&sol, 0.0, sr, MAX_READOUT_ABS_BEFORE_SNAP),
            "collinear solve must satisfy BIBO bounds: l1={l1}"
        );
        assert!(l1 <= MAX_READOUT_L1_NORM + 1.0e-9);

        // Extreme RHS forcing uniform scaling fallback preserves finite l1 <= MAX_READOUT_L1_NORM.
        let tiny_gram: Vec<f64> = (0..dim)
            .flat_map(|c| (0..=c).map(move |r| if r == c { 1.0e-6 } else { 0.0 }))
            .collect();
        let huge_rhs: Vec<f64> = (0..dim)
            .map(|c| if c % 2 == 0 { 10.0 } else { -10.0 })
            .collect();
        let (fallback_sol, _) = solve_bounded_uplo_u(&tiny_gram, &huge_rhs, dim, 0.0, sr).unwrap();
        let fb_l1: f64 = fallback_sol.iter().map(|v| v.abs()).sum();
        assert!(
            fallback_sol
                .iter()
                .all(|v| v.is_finite() && v.abs() <= MAX_READOUT_ABS_BEFORE_SNAP + 1.0e-9)
        );
        assert!(fb_l1 <= MAX_READOUT_L1_NORM + 1.0e-9, "fb_l1={fb_l1}");
    }

    #[test]
    fn test_solve_bounded_uplo_u_scale_invariant_across_paralru_and_fir_scales() {
        let m = 512_usize;
        let dim = 36_usize; // 4 ParaLRU-scale outliers + 32 small-scale correlated FIR taps
        let sr = 48_000.0_f64;
        let mut feats = vec![0.0_f64; dim * m];
        for c in 0..4 {
            for t in 0..m {
                feats[c * m + t] = 70.0 + 2.0 * (0.03 * (t as f64) + 0.4 * (c as f64)).sin();
            }
        }
        let mut base = vec![0.0_f64; m + 32];
        for (i, slot) in base.iter_mut().enumerate() {
            let ti = (i as f64) / sr;
            let raw: f64 = [250.0, 900.0, 2_800.0, 7_500.0, 14_000.0]
                .iter()
                .enumerate()
                .map(|(k, &f)| (core::f64::consts::TAU * f * ti + 0.3 * (k as f64)).sin())
                .sum();
            *slot = 0.28 * raw;
        }
        for c in 0..32 {
            for t in 0..m {
                let x0 = base[t + 32 - c];
                let x1 = base[t + 31 - c];
                feats[(4 + c) * m + t] = 0.5 * (x0 + x1);
            }
        }
        let tgts: Vec<f64> = (0..m).map(|t| 0.8 * base[t + 32]).collect();

        let mut gram = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut rhs = vec![0.0_f64; dim];
        for j in 0..dim {
            let fj = &feats[j * m..(j + 1) * m];
            rhs[j] = fj.iter().zip(&tgts).map(|(&a, &b)| a * b).sum();
            for i in 0..=j {
                let fi = &feats[i * m..(i + 1) * m];
                gram[uplo_u_index(i, j)] = fi.iter().zip(fj).map(|(&a, &b)| a * b).sum();
            }
        }

        let (sol, used_rel) = solve_bounded_uplo_u(&gram, &rhs, dim, 0.0, sr).unwrap();
        assert_eq!(used_rel, 1.0e-5);
        assert!(within_6db_and_bounds(
            &sol,
            0.0,
            sr,
            MAX_READOUT_ABS_BEFORE_SNAP
        ));
        let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
        for (t, &yt) in tgts.iter().enumerate() {
            let mut pred = 0.0_f64;
            for (c, &wc) in sol.iter().enumerate() {
                pred += wc * feats[c * m + t];
            }
            sig_sq += yt * yt;
            err_sq += (yt - pred).powi(2);
        }
        let esr_db = 10.0 * (err_sq / sig_sq.max(1.0e-18)).max(1.0e-16).log10();
        assert!(
            esr_db < -50.0,
            "outlier ParaLRU columns must not over-regularize FIR inversion: esr={esr_db:.2} dB"
        );
    }
}
