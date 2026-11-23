//! `O(N log N)` zero-padded radix-2 FFT Toeplitz autocorrelation/cross-correlation and `O(P^2)`
//! Levinson–Durbin solver for Pass-1 Early FIR lifting (`<= 128` taps)
//! (`ref[train.pipeline.physics-probe]`).

use crate::linalg::cholesky::solve_uplo_adaptive_ridge;
use crate::losses::Radix2FftPlan;
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::super::model::NUM_FIR_TAPS_ULTRA;

/// Solves symmetric positive-definite Toeplitz system `T * h = rhs` (`T[i,j] = r[|i - j|]`) via
/// `O(P^2)` Levinson–Durbin recursion, falling back to `solve_uplo_adaptive_ridge` only if any
/// reflection coefficient reaches `|k_m| >= 1.0 - 1e-12`.
fn solve_toeplitz_levinson_durbin(r: &[f64], rhs: &[f64], p: usize) -> Vec<f64> {
    if p == 0 || r.len() < p || rhs.len() < p || !r[0].is_finite() || r[0] <= 1.0e-18 {
        return vec![0.0; p];
    }
    let mut h = vec![0.0_f64; p];
    h[0] = rhs[0] / r[0];
    if p == 1 {
        return h;
    }

    let mut a = vec![0.0_f64; p];
    let mut a_next = vec![0.0_f64; p];
    a[0] = 1.0;
    let mut err_pred = r[0];

    for m in 1..p {
        let mut eps = 0.0_f64;
        for j in 0..m {
            eps += a[j] * r[m - j];
        }
        let k_m = -eps / err_pred.max(1.0e-30);
        if !k_m.is_finite() || k_m.abs() >= 1.0 - 1.0e-12 {
            return solve_toeplitz_cholesky_fallback(r, rhs, p);
        }
        a_next[0] = 1.0;
        for j in 1..m {
            a_next[j] = a[j] + k_m * a[m - j];
        }
        a_next[m] = k_m;
        a[..=m].copy_from_slice(&a_next[..=m]);
        err_pred *= 1.0 - k_m * k_m;
        if !err_pred.is_finite() || err_pred <= 1.0e-30 {
            return solve_toeplitz_cholesky_fallback(r, rhs, p);
        }

        let mut h_dot = 0.0_f64;
        for j in 0..m {
            h_dot += h[j] * r[m - j];
        }
        let c_m = (rhs[m] - h_dot) / err_pred;
        if !c_m.is_finite() {
            return solve_toeplitz_cholesky_fallback(r, rhs, p);
        }
        for j in 0..=m {
            h[j] += c_m * a[m - j];
        }
    }
    h
}

fn solve_toeplitz_cholesky_fallback(r: &[f64], rhs: &[f64], p: usize) -> Vec<f64> {
    let mut g_u = vec![0.0_f64; uplo_u_packed_len(p)];
    for c in 0..p {
        for row in 0..=c {
            g_u[uplo_u_index(row, c)] = r[c - row];
        }
    }
    let ridge = (1.0e-6 * r[0].abs()).max(1.0e-12);
    solve_uplo_adaptive_ridge(p, &g_u, &rhs[..p], &[], None, ridge)
        .map(|(sol, _)| sol)
        .unwrap_or_else(|_| vec![0.0; p])
}

/// Computes biased lag autocorrelation `r_x[0..p]` and cross-correlation `r_yx[0..p]` via
/// zero-padded `Radix2FftPlan` (`L_fft = (n + p).next_power_of_two()`).
fn fft_biased_toeplitz_moments(input: &[f64], target: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let n = input.len().min(target.len());
    let l_fft = (n + p).max(4).next_power_of_two();
    let Ok(plan) = Radix2FftPlan::new(l_fft) else {
        return (vec![0.0; p], vec![0.0; p]);
    };

    // Real-pair DFT trick: z = x + i * y lets one forward FFT yield both X(w) and Y(w).
    let mut re = vec![0.0_f64; l_fft];
    let mut im = vec![0.0_f64; l_fft];
    for t in 0..n {
        let xv = input[t];
        let yv = target[t];
        re[t] = if xv.is_finite() { xv } else { 0.0 };
        im[t] = if yv.is_finite() { yv } else { 0.0 };
    }
    plan.fft_in_place(&mut re, &mut im);

    let mut ac_re = vec![0.0_f64; l_fft];
    let mut ac_im = vec![0.0_f64; l_fft];
    let mut xc_re = vec![0.0_f64; l_fft];
    let mut xc_im = vec![0.0_f64; l_fft];
    for k in 0..l_fft {
        let neg_k = if k == 0 { 0 } else { l_fft - k };
        let (zp_r, zp_i) = (re[k], im[k]);
        let (zn_r, zn_i) = (re[neg_k], im[neg_k]);
        let x_r = 0.5 * (zp_r + zn_r);
        let x_i = 0.5 * (zp_i - zn_i);
        let y_r = 0.5 * (zp_i + zn_i);
        let y_i = 0.5 * (zn_r - zp_r);
        ac_re[k] = x_r * x_r + x_i * x_i;
        xc_re[k] = y_r * x_r + y_i * x_i;
        xc_im[k] = y_i * x_r - y_r * x_i;
    }
    plan.ifft_in_place(&mut ac_re, &mut ac_im);
    plan.ifft_in_place(&mut xc_re, &mut xc_im);

    let inv_n = 1.0 / (n as f64);
    let r_x = (0..p).map(|k| ac_re[k] * inv_n).collect();
    let r_yx = (0..p).map(|k| xc_re[k] * inv_n).collect();
    (r_x, r_yx)
}

/// Fits a regularized Early FIR (`<= 128` taps) in `O(N log N + P^2)` time via zero-padded
/// radix-2 FFT Toeplitz moments (`1e-6 * r_x[0]` single diagonal ridge) and Levinson–Durbin.
#[must_use]
pub(crate) fn fit_early_fir_taps(
    input: &[f64],
    target: &[f64],
    num_taps: usize,
) -> [f64; NUM_FIR_TAPS_ULTRA] {
    let mut taps = [0.0_f64; NUM_FIR_TAPS_ULTRA];
    let p = num_taps.min(NUM_FIR_TAPS_ULTRA);
    let n = input.len().min(target.len());
    if p == 0 || n <= p {
        return taps;
    }
    let (mut r_x, r_yx) = fft_biased_toeplitz_moments(&input[..n], &target[..n], p);
    if !r_x[0].is_finite() || r_x[0] <= 1.0e-18 {
        return taps;
    }
    // Single diagonal ridge r_x[0] <- r_x[0] * (1 + 1e-6).
    r_x[0] = r_x[0] * (1.0 + 1.0e-6) + 1.0e-15;
    let sol = solve_toeplitz_levinson_durbin(&r_x, &r_yx, p);
    for (i, &w) in sol.iter().enumerate() {
        taps[i] = if w.is_finite() {
            w.clamp(-16.0, 16.0)
        } else {
            0.0
        };
    }
    taps
}

/// Evaluates `fir` (`active_taps` length) over `input` and returns `target - fir(input)`.
#[must_use]
pub(crate) fn residual_after_fir(
    input: &[f64],
    target: &[f64],
    fir: &[f64; NUM_FIR_TAPS_ULTRA],
    active_taps: usize,
) -> Vec<f64> {
    let n = input.len().min(target.len());
    let taps = active_taps.min(NUM_FIR_TAPS_ULTRA);
    let mut res = vec![0.0_f64; n];
    for t in 0..n {
        let mut y_fir = 0.0_f64;
        for k in 0..taps.min(t + 1) {
            y_fir += fir[k] * input[t - k];
        }
        res[t] = target[t] - y_fir;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_toeplitz_levinson_exact_128tap_recovery_and_cholesky_parity() {
        let n = 4_096_usize;
        let p = 128_usize;
        let mut x = vec![0.0_f64; n];
        let mut seed = 0x1234_5678_9ABC_DEF1_u64;
        for v in &mut x {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = 2.0 * (((seed >> 33) as f64) / ((1_u64 << 31) as f64)) - 1.0;
        }

        let mut true_h = [0.0_f64; NUM_FIR_TAPS_ULTRA];
        for (k, slot) in true_h.iter_mut().enumerate() {
            *slot = 0.55 * (-0.045 * (k as f64)).exp() * ((k as f64) * 0.37).cos();
        }
        let synth_y = |sig: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0_f64; sig.len()];
            for t in 0..sig.len() {
                let mut acc = 0.0_f64;
                for k in 0..p.min(t + 1) {
                    acc += true_h[k] * sig[t - k];
                }
                out[t] = acc;
            }
            out
        };

        // Exact finite-support block (zero trailing ringout) checks exact algebra < 1e-10.
        let mut x_finite = x.clone();
        x_finite[n - p..].fill(0.0);
        let y_finite = synth_y(&x_finite);
        let fitted_exact = fit_early_fir_taps(&x_finite, &y_finite, p);
        let mse_exact: f64 = (0..p)
            .map(|k| {
                let d = fitted_exact[k] - true_h[k];
                d * d
            })
            .sum::<f64>()
            / (p as f64);
        assert!(
            mse_exact < 1.0e-10,
            "expected < 1e-10 exact finite-support FIR MSE, got {mse_exact}"
        );

        // Untapered broadband excitation recovers within < 1e-5 MSE and matches Cholesky < 1e-6.
        let y = synth_y(&x);
        let fitted = fit_early_fir_taps(&x, &y, p);
        let mse_raw: f64 = (0..p)
            .map(|k| {
                let d = fitted[k] - true_h[k];
                d * d
            })
            .sum::<f64>()
            / (p as f64);
        assert!(mse_raw < 1.0e-5, "expected < 1e-5 FIR MSE, got {mse_raw}");

        let (mut r_x, r_yx) = fft_biased_toeplitz_moments(&x, &y, p);
        r_x[0] = r_x[0] * (1.0 + 1.0e-6) + 1.0e-15;
        let chol = solve_toeplitz_cholesky_fallback(&r_x, &r_yx, p);
        for k in 0..p {
            assert!(
                (fitted[k] - chol[k]).abs() < 1.0e-6,
                "Levinson/Cholesky parity mismatch at tap {k}: {} vs {}",
                fitted[k],
                chol[k]
            );
        }

        // Silent excitation fails safe to zero taps.
        let zero_fit = fit_early_fir_taps(&vec![0.0; n], &y, p);
        assert!(zero_fit.iter().all(|&v| v == 0.0));
    }
}
