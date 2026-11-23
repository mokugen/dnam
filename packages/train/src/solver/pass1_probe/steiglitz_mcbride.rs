//! Iterative Steiglitz–McBride `(B(z), A(z))` prefiltering and Durand–Kerner conjugate pole
//! extraction with `0.9994 / 0.9992` radius ceiling (`ref[train.pipeline.physics-probe]`,
//! `ref[dsp.op-m.kautz.pole-extraction]`).

use core::f64::consts::{PI, TAU};

use crate::linalg::cholesky::solve_uplo_adaptive_ridge;

/// Maximum AR polynomial stabilization threshold (matched to `MAX_PASS1_KAUTZ_R = 0.9992`).
const MAX_AR_ROOT_THRESHOLD: f64 = 0.9994;
const MAX_STABILIZED_POLE_R: f64 = 0.9992;
const _: () = assert!(MAX_STABILIZED_POLE_R < MAX_AR_ROOT_THRESHOLD && MAX_AR_ROOT_THRESHOLD < 1.0);

#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cdiv(num: (f64, f64), den: (f64, f64)) -> (f64, f64) {
    let d2 = (den.0 * den.0 + den.1 * den.1).max(1.0e-30);
    (
        (num.0 * den.0 + num.1 * den.1) / d2,
        (num.1 * den.0 - num.0 * den.1) / d2,
    )
}

fn filter_all_pole_inverse(x: &[f64], a: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; x.len()];
    for t in 0..x.len() {
        let mut acc = x[t];
        for (k, &ak) in a.iter().enumerate() {
            if t > k {
                acc -= ak * y[t - k - 1];
            }
        }
        y[t] = acc;
    }
    y
}

fn solve_yule_walker_ar(sig: &[f64], order: usize) -> Vec<f64> {
    let n = sig.len();
    if order == 0 || n <= order + 4 {
        return Vec::new();
    }
    let packed = order * (order + 1) / 2;
    let mut g_u = vec![0.0_f64; packed];
    let mut rhs = vec![0.0_f64; order];
    for c in 0..order {
        let col_off = c * (c + 1) / 2;
        for r in 0..=c {
            let mut dot = 0.0_f64;
            for t in order..n {
                dot += sig[t - r - 1] * sig[t - c - 1];
            }
            g_u[col_off + r] = dot;
        }
        let mut rd = 0.0_f64;
        for t in order..n {
            rd += -sig[t] * sig[t - c - 1];
        }
        rhs[c] = rd;
    }
    // Pass raw normal matrix without pre-adding ridge so solve_uplo_adaptive_ridge applies it once.
    solve_uplo_adaptive_ridge(order, &g_u, &rhs, &[], None, 1.0e-7)
        .map(|(v, _)| v)
        .unwrap_or_else(|_| vec![0.0; order])
}

fn find_monic_roots(a: &[f64]) -> Vec<(f64, f64)> {
    let m = a.len();
    let mut roots: Vec<(f64, f64)> = (0..m)
        .map(|k| {
            let ang = TAU * ((k as f64) + 0.37) / (m as f64);
            (0.85 * ang.cos(), 0.85 * ang.sin())
        })
        .collect();
    for _ in 0..64 {
        let mut max_step = 0.0_f64;
        for i in 0..m {
            let zi = roots[i];
            let mut pval = (1.0_f64, 0.0_f64);
            for &ak in a {
                pval = cmul(pval, zi);
                pval.0 += ak;
            }
            let mut prod = (1.0_f64, 0.0_f64);
            for (j, &zj) in roots.iter().enumerate() {
                if i != j {
                    prod = cmul(prod, (zi.0 - zj.0, zi.1 - zj.1));
                }
            }
            let delta = cdiv(pval, prod);
            roots[i] = (zi.0 - delta.0, zi.1 - delta.1);
            max_step = max_step.max(delta.0 * delta.0 + delta.1 * delta.1);
        }
        if max_step < 1.0e-18 {
            break;
        }
    }
    roots
}

fn stabilize_ar_coeffs(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut roots = find_monic_roots(a);
    let r_thresh_sq = MAX_AR_ROOT_THRESHOLD * MAX_AR_ROOT_THRESHOLD;
    let needs_stabilize = roots
        .iter()
        .any(|&(re, im)| re * re + im * im >= r_thresh_sq);
    if !needs_stabilize {
        return a.to_vec();
    }
    let mut coeffs = vec![(0.0_f64, 0.0_f64); a.len() + 1];
    coeffs[0] = (1.0, 0.0);
    for r in &mut roots {
        let mag = (r.0 * r.0 + r.1 * r.1).sqrt();
        if mag >= MAX_STABILIZED_POLE_R {
            let s = MAX_STABILIZED_POLE_R / mag.max(1.0e-12);
            r.0 *= s;
            r.1 *= s;
        }
    }
    for (k, &root) in roots.iter().enumerate() {
        for j in (1..=k + 1).rev() {
            let prod = cmul(coeffs[j - 1], root);
            coeffs[j] = (coeffs[j].0 - prod.0, coeffs[j].1 - prod.1);
        }
    }
    coeffs[1..].iter().map(|c| c.0).collect()
}

fn solve_arx_prefiltered_step(x_f: &[f64], y_f: &[f64], m: usize) -> Option<Vec<f64>> {
    let n = x_f.len().min(y_f.len());
    if n <= m + 4 {
        return None;
    }
    let dim = 2 * m + 1;
    let packed = dim * (dim + 1) / 2;
    let mut g_u = vec![0.0_f64; packed];
    let mut rhs = vec![0.0_f64; dim];
    let mut phi = vec![0.0_f64; dim];
    let mut diag_sum = 0.0_f64;
    for t in m..n {
        for k in 0..m {
            phi[k] = -y_f[t - k - 1];
        }
        for j in 0..=m {
            phi[m + j] = x_f[t - j];
        }
        let yt = y_f[t];
        for c in 0..dim {
            let col_off = c * (c + 1) / 2;
            let pc = phi[c];
            rhs[c] += yt * pc;
            for r in 0..=c {
                g_u[col_off + r] += phi[r] * pc;
            }
            diag_sum += pc * pc;
        }
    }
    let ridge = (1.0e-9 * (diag_sum / (dim as f64)).max(1.0)).clamp(1.0e-11, 1.0e-6);
    // Pass raw normal matrix to avoid double-ridge bias on high-Q poles.
    let (sol, _) = solve_uplo_adaptive_ridge(dim, &g_u, &rhs, &[], None, ridge).ok()?;
    let a_next: Vec<f64> = sol[..m].to_vec();
    if a_next.iter().all(|v| v.is_finite()) {
        Some(stabilize_ar_coeffs(&a_next))
    } else {
        None
    }
}

/// Identifies `(B(z), A(z))` via iterative Steiglitz–McBride prefiltering over `(input, target)`
/// and extracts strictly stable, deduplicated upper-half-plane `(r, theta)` pole pairs up to
/// `r <= 0.9992` via Durand–Kerner.
#[must_use]
pub fn steiglitz_mcbride_poles(
    input: &[f64],
    target: &[f64],
    order: usize,
    iters: usize,
) -> Vec<(f64, f64)> {
    let n = input.len().min(target.len());
    if order < 2 || n < order + 8 {
        return Vec::new();
    }
    let in_e: f64 = input[..n].iter().map(|v| v * v).sum();
    let mut a = if in_e > 1.0e-12 {
        let m = order.min(24);
        let mut cur_a = vec![0.0_f64; m];
        for _ in 0..iters.clamp(1, 6) {
            let xf = filter_all_pole_inverse(&input[..n], &cur_a);
            let yf = filter_all_pole_inverse(&target[..n], &cur_a);
            if xf
                .iter()
                .chain(&yf)
                .any(|v| !v.is_finite() || v.abs() > 1.0e6)
            {
                break;
            }
            if let Some(next_a) = solve_arx_prefiltered_step(&xf, &yf, m) {
                let diff: f64 = cur_a
                    .iter()
                    .zip(&next_a)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                cur_a = next_a;
                if diff < 1.0e-5 {
                    break;
                }
            } else {
                break;
            }
        }
        cur_a
    } else {
        let m = order.min(24);
        let mut cur_a = solve_yule_walker_ar(&target[..n], m);
        for _ in 0..iters.min(6) {
            let w = filter_all_pole_inverse(&target[..n], &cur_a);
            if w.iter().any(|v| !v.is_finite() || v.abs() > 1.0e6) {
                break;
            }
            let next = solve_yule_walker_ar(&w, m);
            if next.iter().all(|v| v.is_finite()) {
                cur_a = stabilize_ar_coeffs(&next);
            }
        }
        cur_a
    };
    if a.is_empty() {
        return Vec::new();
    }
    a = stabilize_ar_coeffs(&a);
    let roots = find_monic_roots(&a);
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for (re, im) in roots {
        if !re.is_finite() || !im.is_finite() || im < -1.0e-6 {
            continue;
        }
        let r = (re * re + im * im).sqrt().clamp(0.0, MAX_STABILIZED_POLE_R);
        let theta = im.abs().atan2(re).clamp(1.0e-4, PI - 1.0e-4);
        if !pairs
            .iter()
            .any(|&(pr, pt)| (pr - r).abs() < 5.0e-4 && (pt - theta).abs() < 5.0e-4)
        {
            pairs.push((r, theta));
        }
    }
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steiglitz_mcbride_recovers_high_q_and_low_frequency_poles() {
        let n = 8_192_usize;
        let sr = 48_000.0_f64;
        let mut x = vec![0.0_f64; n];
        let (mut lp, mut seed) = (0.0_f64, 0xCAFE_BABE_1234_5679_u64);
        for v in &mut x {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let w = ((seed >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5;
            lp = 0.75 * lp + w;
            *v = lp;
        }

        // High-Q conjugate resonance at 1800 Hz, r = 0.9990.
        let r_true = 0.9990_f64;
        let th_true = TAU * 1_800.0 / sr;
        let a1 = -2.0 * r_true * th_true.cos();
        let a2 = r_true * r_true;
        let mut y = vec![0.0_f64; n];
        for t in 2..n {
            y[t] = x[t] - a1 * y[t - 1] - a2 * y[t - 2];
        }
        let poles = steiglitz_mcbride_poles(&x, &y, 2, 5);
        assert!(!poles.is_empty());
        assert!(
            (poles[0].0 - r_true).abs() < 1.0e-3 && poles[0].0 <= MAX_STABILIZED_POLE_R,
            "high-Q r mismatch: {:?} vs {r_true}",
            poles[0]
        );
        assert!(
            (poles[0].1 - th_true).abs() < 5.0e-3,
            "high-Q theta mismatch: {:?} vs {th_true}",
            poles[0]
        );

        // Low-frequency resonance at 50 Hz, r = 0.995.
        let th_50 = TAU * 50.0 / sr;
        let b1 = -2.0 * 0.995 * th_50.cos();
        let b2 = 0.995 * 0.995;
        let mut y_50 = vec![0.0_f64; n];
        for t in 2..n {
            y_50[t] = x[t] - b1 * y_50[t - 1] - b2 * y_50[t - 2];
        }
        let poles_50 = steiglitz_mcbride_poles(&x, &y_50, 2, 5);
        assert!(!poles_50.is_empty());
        assert!(
            (poles_50[0].1 - th_50).abs() < 3.0e-3,
            "50 Hz theta mismatch: {:?} vs {th_50}",
            poles_50[0]
        );
    }
}
