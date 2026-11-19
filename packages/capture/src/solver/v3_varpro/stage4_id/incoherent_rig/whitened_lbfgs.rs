#![forbid(unsafe_code)]
//! Cholesky-whitened bounded L-BFGS optimizer (`incoherent_rig/whitened_lbfgs.rs`).
//!
//! Preconditions badly scaled Stage III FIR/Kautz least-squares systems via `R^T R = G + ridge * I`,
//! transforming steps into whitened coordinates `z = R * theta` with a trust-capped initial step.

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn factorize_gram_upper(dim: usize, gram: &[f64]) -> Option<Vec<f64>> {
    let g_bar =
        ((0..dim).map(|i| gram[i * dim + i]).sum::<f64>() / (dim.max(1) as f64)).max(1.0e-12);
    let mut r = vec![0.0_f64; dim * dim];
    let ridge = 1.0e-5 * g_bar;
    for i in 0..dim {
        let mut s = gram[i * dim + i] + ridge;
        for k in 0..i {
            s -= r[k * dim + i] * r[k * dim + i];
        }
        if !s.is_finite() || s <= 1.0e-18 {
            return None;
        }
        let diag = s.sqrt();
        r[i * dim + i] = diag;
        let inv = 1.0 / diag;
        for j in (i + 1)..dim {
            let mut sum = gram[i * dim + j];
            for k in 0..i {
                sum -= r[k * dim + i] * r[k * dim + j];
            }
            r[i * dim + j] = sum * inv;
        }
    }
    Some(r)
}

fn mul_r_theta(dim: usize, r: &[f64], theta: &[f64], z: &mut [f64]) {
    for i in 0..dim {
        let mut s = 0.0_f64;
        for j in i..dim {
            s += r[i * dim + j] * theta[j];
        }
        z[i] = s;
    }
}

fn solve_r_z(dim: usize, r: &[f64], z: &[f64], theta: &mut [f64]) {
    for i in (0..dim).rev() {
        let mut s = z[i];
        for j in (i + 1)..dim {
            s -= r[i * dim + j] * theta[j];
        }
        theta[i] = s / r[i * dim + i];
    }
}

pub(super) fn solve_rt_g(dim: usize, r: &[f64], g_theta: &[f64], g_z: &mut [f64]) {
    for i in 0..dim {
        let mut s = g_theta[i];
        for k in 0..i {
            s -= r[k * dim + i] * g_z[k];
        }
        g_z[i] = s / r[i * dim + i];
    }
}

pub(super) struct WhitenedLbfgs {
    dim: usize,
    pub(super) r_mat: Vec<f64>,
    pub(super) z0: Vec<f64>,
    s_hist: Vec<Vec<f64>>,
    y_hist: Vec<Vec<f64>>,
    rho_hist: Vec<f64>,
}

impl WhitenedLbfgs {
    pub(super) fn new(dim: usize, gram: &[f64], theta0: &[f64]) -> Option<(Self, Vec<f64>)> {
        let r_mat = factorize_gram_upper(dim, gram)?;
        let mut z0 = vec![0.0_f64; dim];
        mul_r_theta(dim, &r_mat, theta0, &mut z0);
        let init_z = z0.clone();
        Some((
            Self {
                dim,
                r_mat,
                z0,
                s_hist: Vec::with_capacity(8),
                y_hist: Vec::with_capacity(8),
                rho_hist: Vec::with_capacity(8),
            },
            init_z,
        ))
    }

    pub(super) fn warm_reset(&mut self, theta_cur: &[f64]) -> Vec<f64> {
        self.s_hist.clear();
        self.y_hist.clear();
        self.rho_hist.clear();
        mul_r_theta(self.dim, &self.r_mat, theta_cur, &mut self.z0);
        self.z0.clone()
    }

    fn direction(&mut self, grad: &[f64], q: &mut [f64]) {
        for (qd, &g) in q.iter_mut().zip(grad) {
            *qd = -g;
        }
        let m = self.s_hist.len();
        if m == 0 {
            return;
        }
        let mut alpha = vec![0.0_f64; m];
        for i in (0..m).rev() {
            alpha[i] = self.rho_hist[i] * dot(&self.s_hist[i], q);
            for (qj, &yj) in q.iter_mut().zip(&self.y_hist[i]) {
                *qj -= alpha[i] * yj;
            }
        }
        if let (Some(sl), Some(yl)) = (self.s_hist.last(), self.y_hist.last()) {
            let yy = dot(yl, yl).max(1.0e-18);
            let scale = (dot(sl, yl) / yy).clamp(1.0e-8, 1.0e6);
            for qj in q.iter_mut() {
                *qj *= scale;
            }
        }
        for i in 0..m {
            let beta = self.rho_hist[i] * dot(&self.y_hist[i], q);
            for (qj, &sj) in q.iter_mut().zip(&self.s_hist[i]) {
                *qj += (alpha[i] - beta) * sj;
            }
        }
        if !dot(grad, q).is_finite() || dot(grad, q) >= 0.0 {
            self.s_hist.clear();
            self.y_hist.clear();
            self.rho_hist.clear();
            for (qd, &g) in q.iter_mut().zip(grad) {
                *qd = -g;
            }
        }
    }

    fn push(&mut self, s: Vec<f64>, y: Vec<f64>) {
        let sy = dot(&s, &y);
        if sy.is_finite() && sy > 1.0e-16 {
            if self.s_hist.len() == 8 {
                self.s_hist.remove(0);
                self.y_hist.remove(0);
                self.rho_hist.remove(0);
            }
            self.s_hist.push(s);
            self.y_hist.push(y);
            self.rho_hist.push(1.0 / sy);
        }
    }

    pub(super) fn minimize<F>(&mut self, mut z: Vec<f64>, iters: usize, mut eval: F) -> Vec<f64>
    where
        F: FnMut(&[f64], &[f64], Option<&mut [f64]>) -> f64,
    {
        let dim = self.dim;
        let (mut theta, mut grad) = (vec![0.0_f64; dim], vec![0.0_f64; dim]);
        let (mut cand_z, mut cand_th, mut cand_g) =
            (vec![0.0_f64; dim], vec![0.0_f64; dim], vec![0.0_f64; dim]);
        let mut q = vec![0.0_f64; dim];

        solve_r_z(dim, &self.r_mat, &z, &mut theta);
        let mut loss = eval(&theta, &z, Some(&mut grad));
        let (mut best_loss, mut best_theta) = (loss, theta.clone());
        let z_scale = dot(&self.z0, &self.z0).sqrt().max(1.0e-3);
        let mut stall_steps = 0_usize;

        for _ in 0..iters {
            self.direction(&grad, &mut q);
            let mut g_dot_q = dot(&grad, &q);
            if !g_dot_q.is_finite() || g_dot_q >= -1.0e-18 {
                break;
            }
            let q_norm = dot(&q, &q).sqrt().max(1.0e-12);
            let max_step = if self.s_hist.is_empty() {
                (0.5 * z_scale / q_norm).min(1.0)
            } else {
                (4.0 * z_scale / q_norm).min(1.0)
            };
            let mut step = max_step;
            let mut accepted = false;
            let mut converged = false;
            for _ in 0..20 {
                for j in 0..dim {
                    cand_z[j] = z[j] + step * q[j];
                }
                solve_r_z(dim, &self.r_mat, &cand_z, &mut cand_th);
                let l_try = eval(&cand_th, &cand_z, None);
                if l_try.is_finite() && l_try <= loss + 1.0e-4 * step * g_dot_q {
                    let rel_improv = loss - l_try;
                    let rel_tol = 1.0e-9 * loss.abs().max(1.0e-6);
                    let _ = eval(&cand_th, &cand_z, Some(&mut cand_g));
                    let s: Vec<f64> = cand_z.iter().zip(&z).map(|(a, b)| a - b).collect();
                    let y: Vec<f64> = cand_g.iter().zip(&grad).map(|(a, b)| a - b).collect();
                    self.push(s, y);
                    z.copy_from_slice(&cand_z);
                    theta.copy_from_slice(&cand_th);
                    grad.copy_from_slice(&cand_g);
                    loss = l_try;
                    if loss < best_loss {
                        best_loss = loss;
                        best_theta.copy_from_slice(&theta);
                    }
                    if rel_improv <= rel_tol {
                        stall_steps += 1;
                        if stall_steps >= 3 {
                            converged = true;
                        }
                    } else {
                        stall_steps = 0;
                    }
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if converged {
                break;
            }
            if !accepted {
                if !self.s_hist.is_empty() {
                    self.s_hist.clear();
                    self.y_hist.clear();
                    self.rho_hist.clear();
                    g_dot_q = -dot(&grad, &grad);
                    if g_dot_q < -1.0e-18 {
                        continue;
                    }
                }
                break;
            }
        }
        best_theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitened_lbfgs_converges_and_skips_grad_on_line_search_trials() {
        // Anisotropic quartic + coupled quadratic whose minimum is at [1.0, -0.5, 0.25].
        let target = [1.0_f64, -0.5, 0.25];
        let gram = [
            120.0, 24.0, -6.0, //
            24.0, 12.0, 1.5, //
            -6.0, 1.5, 4.0,
        ];
        let theta0 = [4.0_f64, 3.5, -2.5];
        let (mut opt, init_z) = WhitenedLbfgs::new(3, &gram, &theta0).unwrap();
        let r_mat = opt.r_mat.clone();

        let mut loss_only_calls = 0_usize;
        let mut grad_calls = 0_usize;
        let sol = opt.minimize(init_z, 24, |th, _z, grad_z| {
            let d0 = th[0] - target[0];
            let d1 = th[1] - target[1];
            let d2 = th[2] - target[2];
            // Steep quartic + quadratic so the first step overshoots and triggers backtracking.
            let val = 60.0 * d0 * d0
                + 80.0 * d0 * d0 * d0 * d0
                + 6.0 * d1 * d1
                + 2.0 * d2 * d2
                + 24.0 * d0 * d1;
            if let Some(g_th) = grad_z {
                grad_calls += 1;
                let g_orig = [
                    120.0 * d0 + 320.0 * d0 * d0 * d0 + 24.0 * d1,
                    12.0 * d1 + 24.0 * d0,
                    4.0 * d2,
                ];
                solve_rt_g(3, &r_mat, &g_orig, g_th);
            } else {
                loss_only_calls += 1;
            }
            val
        });

        assert!(loss_only_calls > 0);
        assert!(grad_calls <= 25);
        for (a, b) in sol.iter().zip(&target) {
            assert!((a - b).abs() < 1.0e-5, "sol={sol:?} target={target:?}");
        }
    }
}
