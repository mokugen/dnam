//! Two-loop bounded L-BFGS (`m <= 10`), Cholesky-whitened L-BFGS (`WhitenedLbfgs`),
//! and GPU/CPU `TrainRunner::inner_opt_step` bridge (`ref[train.pipeline.backbone-trajectory-opt]`).

pub use super::coords::ParamBoundSpec;

use crate::error::TrainError;
use crate::gpu::abi::{INNER_OPT_MODE_DIAG_PRECOND, InnerOptUniforms};
use crate::runner::{InnerOptInput, TrainRunner};

use super::cholesky::factorize_uplo_u_cholesky;
use super::coords::{OptResult, ScaledCoordSpace, dot_simd};

/// Alias to the shared [`OptResult`] returned by [`TwoLoopLbfgs::minimize`] / [`WhitenedLbfgs::minimize`].
pub type LbfgsResult = OptResult;

/// Configuration for [`TwoLoopLbfgs`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsConfig {
    /// Maximum optimizer outer steps.
    pub max_iters: usize,
    /// L-BFGS curvature memory (`1..=10`).
    pub history_m: usize,
    /// Armijo sufficient-decrease slope coefficient (`default = 1e-4`).
    pub armijo_c1: f64,
    /// Maximum backtracking trials per iteration.
    pub max_line_search: usize,
    /// Relative objective change threshold for stall termination.
    pub rel_tol: f64,
    /// Number of consecutive small improvements before early stop.
    pub patience: usize,
    /// Optional Euclidean trust cap on each step in unconstrained space.
    pub max_step_norm: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iters: 48,
            history_m: 10,
            armijo_c1: 1.0e-4,
            max_line_search: 16,
            rel_tol: 1.0e-9,
            patience: 4,
            max_step_norm: 8.0,
        }
    }
}

fn compute_two_loop_dir(
    grad: &[f64],
    h0_diag: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
    dir: &mut [f64],
) {
    let m = s_hist.len();
    if m == 0 {
        for ((d, &g), &h0) in dir.iter_mut().zip(grad).zip(h0_diag) {
            *d = -h0 * g;
        }
        return;
    }
    for (d, &g) in dir.iter_mut().zip(grad) {
        *d = -g;
    }
    let mut alpha = vec![0.0_f64; m];
    for i in (0..m).rev() {
        alpha[i] = rho_hist[i] * dot_simd(&s_hist[i], dir);
        for (dj, &yj) in dir.iter_mut().zip(&y_hist[i]) {
            *dj -= alpha[i] * yj;
        }
    }
    if let (Some(sl), Some(yl)) = (s_hist.last(), y_hist.last()) {
        let y_h0_y = yl
            .iter()
            .zip(h0_diag)
            .map(|(&y, &h0)| h0 * y * y)
            .sum::<f64>()
            .max(1.0e-18);
        let scale = (dot_simd(sl, yl) / y_h0_y).clamp(1.0e-8, 1.0e6);
        for (dj, &h0) in dir.iter_mut().zip(h0_diag) {
            *dj *= scale * h0;
        }
    }
    for i in 0..m {
        let beta = rho_hist[i] * dot_simd(&y_hist[i], dir);
        for (dj, &sj) in dir.iter_mut().zip(&s_hist[i]) {
            *dj += (alpha[i] - beta) * sj;
        }
    }
}

/// General two-loop L-BFGS optimizer with quadratic Armijo backtracking and steepest-descent
/// fallback (`m <= 10`).
#[derive(Clone, Debug, Default)]
pub struct TwoLoopLbfgs {
    config: LbfgsConfig,
}

impl TwoLoopLbfgs {
    /// Creates a two-loop L-BFGS optimizer with the given configuration.
    #[must_use]
    pub const fn new(config: LbfgsConfig) -> Self {
        Self { config }
    }

    /// Minimizes `eval(params, grad_opt)` over `bounds` starting at `init_physical`.
    /// When `grad_opt` is `None`, only the scalar loss needs to be computed for backtracking.
    pub fn minimize<F>(
        &self,
        init_physical: &[f64],
        bounds: &[ParamBoundSpec],
        eval: F,
    ) -> Result<LbfgsResult, TrainError>
    where
        F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
    {
        self.minimize_scaled(init_physical, bounds, None, eval)
    }

    /// Variant of [`Self::minimize`] applying symmetric coordinate scaling `u_i = s_i * u_raw,i`
    /// (`d(loss)/du_i = d(loss)/dx_i * (dx_i/du_raw,i / s_i)`) so steep parameters remain
    /// symmetrically preconditioned without breaking Armijo descent or BFGS secant symmetry.
    pub fn minimize_scaled<F>(
        &self,
        init_physical: &[f64],
        bounds: &[ParamBoundSpec],
        coord_scales: Option<&[f64]>,
        mut eval: F,
    ) -> Result<LbfgsResult, TrainError>
    where
        F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
    {
        let (space, mut u) = ScaledCoordSpace::new(
            "TwoLoopLbfgs::minimize",
            init_physical,
            bounds,
            coord_scales,
        )?;
        let dim = space.dim();
        let m_cap = self.config.history_m.clamp(1, 10);
        let mut x = vec![0.0_f64; dim];
        let mut dxd_u = vec![0.0_f64; dim];
        space.decode(&u, &mut x, &mut dxd_u);
        let mut h0_diag = vec![1.0_f64; dim];
        ScaledCoordSpace::diag_h0_from_jac(&dxd_u, &mut h0_diag);

        let mut grad_x = vec![0.0_f64; dim];
        let mut loss = eval(&x, Some(&mut grad_x))?;
        if !loss.is_finite() || !grad_x.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput {
                context: "TwoLoopLbfgs::minimize(init_eval)",
            });
        }
        let mut grad_u = vec![0.0_f64; dim];
        space.pullback_grad(&grad_x, &dxd_u, &mut grad_u);

        let mut best_params = x.clone();
        let mut best_loss = loss;
        let mut evals = 1_usize;
        let mut iters_run = 0_usize;

        let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m_cap);
        let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m_cap);
        let mut rho_hist: Vec<f64> = Vec::with_capacity(m_cap);
        let mut dir = vec![0.0_f64; dim];
        let mut cand_u = vec![0.0_f64; dim];
        let mut cand_x = vec![0.0_f64; dim];
        let mut cand_jac = vec![0.0_f64; dim];
        let mut cand_gx = vec![0.0_f64; dim];
        let mut small_streak = 0_usize;

        for _ in 0..self.config.max_iters {
            compute_two_loop_dir(&grad_u, &h0_diag, &s_hist, &y_hist, &rho_hist, &mut dir);
            let mut g_dot_d = dot_simd(&grad_u, &dir);
            if !g_dot_d.is_finite() || g_dot_d >= -1.0e-18 {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                for ((d, &g), &h0) in dir.iter_mut().zip(&grad_u).zip(&h0_diag) {
                    *d = -h0 * g;
                }
                g_dot_d = dot_simd(&grad_u, &dir);
                if !g_dot_d.is_finite() || g_dot_d >= -1.0e-18 {
                    break;
                }
            }

            let d_norm = dot_simd(&dir, &dir).sqrt().max(1.0e-12);
            let mut step = (self.config.max_step_norm / d_norm).min(1.0);
            let mut accepted = false;
            for _ in 0..self.config.max_line_search {
                for j in 0..dim {
                    cand_u[j] = u[j] + step * dir[j];
                }
                space.decode(&cand_u, &mut cand_x, &mut cand_jac);
                evals += 1;
                if let Ok(l_trial) = eval(&cand_x, None)
                    && l_trial.is_finite()
                    && l_trial <= loss + self.config.armijo_c1 * step * g_dot_d
                {
                    evals += 1;
                    if let Ok(l_full) = eval(&cand_x, Some(&mut cand_gx))
                        && l_full.is_finite()
                        && l_full <= loss + self.config.armijo_c1 * step * g_dot_d
                        && cand_gx.iter().all(|v| v.is_finite())
                    {
                        let prev_loss = loss;
                        let mut cand_gu = vec![0.0_f64; dim];
                        space.pullback_grad(&cand_gx, &cand_jac, &mut cand_gu);
                        let s: Vec<f64> = cand_u.iter().zip(&u).map(|(a, b)| a - b).collect();
                        let y: Vec<f64> = cand_gu.iter().zip(&grad_u).map(|(a, b)| a - b).collect();
                        let sy = dot_simd(&s, &y);
                        let sn = dot_simd(&s, &s).sqrt();
                        let yn = dot_simd(&y, &y).sqrt();
                        if sy.is_finite() && sy > 1.0e-14 * sn * yn && sy > 1.0e-18 {
                            if s_hist.len() == m_cap {
                                s_hist.remove(0);
                                y_hist.remove(0);
                                rho_hist.remove(0);
                            }
                            s_hist.push(s);
                            y_hist.push(y);
                            rho_hist.push(1.0 / sy);
                        }
                        u.copy_from_slice(&cand_u);
                        dxd_u.copy_from_slice(&cand_jac);
                        ScaledCoordSpace::diag_h0_from_jac(&dxd_u, &mut h0_diag);
                        grad_u = cand_gu;
                        loss = l_full;
                        iters_run += 1;
                        if loss < best_loss {
                            best_loss = loss;
                            best_params.copy_from_slice(&cand_x);
                        }
                        let impr = (prev_loss - loss).abs();
                        if impr <= self.config.rel_tol * prev_loss.abs().max(1.0e-12) {
                            small_streak += 1;
                        } else {
                            small_streak = 0;
                        }
                        accepted = true;
                        break;
                    }
                }
                step *= 0.35;
            }
            if !accepted || small_streak >= self.config.patience {
                break;
            }
        }

        Ok(LbfgsResult {
            best_params,
            best_loss,
            iters_run,
            evals,
        })
    }
}

/// Cholesky-whitened L-BFGS preconditioning steps into `z = R * theta` where `R^T R = G + ridge I`.
#[derive(Clone, Debug)]
pub struct WhitenedLbfgs {
    dim: usize,
    r_upper: Vec<f64>,
    z0: Vec<f64>,
}

impl WhitenedLbfgs {
    /// Builds a whitened L-BFGS preconditioner from a packed `uplo='U'` Gram matrix `gram_uplo`.
    pub fn from_uplo_u(
        dim: usize,
        gram_uplo: &[f64],
        theta0: &[f64],
        ridge_scale: f64,
    ) -> Result<Self, TrainError> {
        if theta0.len() != dim || dim == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "WhitenedLbfgs::from_uplo_u",
                expected: dim.max(1),
                actual: theta0.len(),
            });
        }
        let expected_gram = crate::wide_gram::uplo_u_packed_len(dim);
        if gram_uplo.len() != expected_gram {
            return Err(TrainError::InvalidDimensions {
                context: "WhitenedLbfgs::from_uplo_u(gram_uplo)",
                expected: expected_gram,
                actual: gram_uplo.len(),
            });
        }
        if !ridge_scale.is_finite()
            || ridge_scale < 0.0
            || !theta0.iter().all(|v| v.is_finite())
            || !gram_uplo.iter().all(|v| v.is_finite())
        {
            return Err(TrainError::NonFiniteInput {
                context: "WhitenedLbfgs::from_uplo_u",
            });
        }
        let mut trace = 0.0_f64;
        for j in 0..dim {
            trace += gram_uplo[crate::wide_gram::uplo_u_index(j, j)].max(0.0);
        }
        let mean_diag = (trace / (dim as f64)).max(1.0e-12);
        let ridge = (ridge_scale.max(1.0e-8) * mean_diag).max(1.0e-12);
        let r_upper = factorize_uplo_u_cholesky(dim, gram_uplo, ridge)?;
        let z0 = Self::mul_r_uplo(dim, &r_upper, theta0);
        Ok(Self { dim, r_upper, z0 })
    }

    fn mul_r_uplo(dim: usize, r_upper: &[f64], theta: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0_f64; dim];
        for (j, &tj) in theta.iter().enumerate() {
            let col_j = crate::wide_gram::uplo_u_index(0, j);
            for i in 0..=j {
                z[i] += r_upper[col_j + i] * tj;
            }
        }
        z
    }

    fn solve_r_uplo(&self, z: &[f64], theta: &mut [f64]) {
        theta.copy_from_slice(z);
        for k in (0..self.dim).rev() {
            let col_k = crate::wide_gram::uplo_u_index(0, k);
            let tk = theta[k] / self.r_upper[col_k + k];
            theta[k] = tk;
            for (ti, &r_ik) in theta[..k].iter_mut().zip(&self.r_upper[col_k..col_k + k]) {
                *ti -= r_ik * tk;
            }
        }
    }

    /// Transforms an ordinary physical gradient `g_theta` into `g_z = R^{-T} g_theta`.
    #[must_use]
    pub fn transform_grad_to_z(&self, g_theta: &[f64]) -> Vec<f64> {
        let mut g_z = vec![0.0_f64; self.dim];
        for i in 0..self.dim {
            let col_i = crate::wide_gram::uplo_u_index(0, i);
            let mut sum = g_theta[i];
            for (&r_ki, &gk) in self.r_upper[col_i..col_i + i].iter().zip(&g_z[..i]) {
                sum -= r_ki * gk;
            }
            g_z[i] = sum / self.r_upper[col_i + i];
        }
        g_z
    }

    /// Minimizes `eval(theta, g_theta_opt)` by optimizing in whitened coordinates `z`.
    pub fn minimize<F>(&self, config: LbfgsConfig, mut eval: F) -> Result<LbfgsResult, TrainError>
    where
        F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
    {
        let bounds = vec![ParamBoundSpec::Free; self.dim];
        let opt = TwoLoopLbfgs::new(config);
        let mut theta_buf = vec![0.0_f64; self.dim];
        let mut g_th = vec![0.0_f64; self.dim];
        let res = opt.minimize(&self.z0, &bounds, |z, gz_opt| {
            self.solve_r_uplo(z, &mut theta_buf);
            if let Some(gz) = gz_opt {
                let l = eval(&theta_buf, Some(&mut g_th))?;
                gz.copy_from_slice(&self.transform_grad_to_z(&g_th));
                Ok(l)
            } else {
                eval(&theta_buf, None)
            }
        })?;
        self.solve_r_uplo(&res.best_params, &mut theta_buf);
        Ok(LbfgsResult {
            best_params: theta_buf,
            best_loss: res.best_loss,
            iters_run: res.iters_run,
            evals: res.evals,
        })
    }
}

/// Computes `f32` diagonal `H_0` weights `1 / clamp(j^2, 1, 1e12)` from coordinate Jacobians
/// via [`ScaledCoordSpace::diag_h0_from_jac`] for GPU/CPU `inner_opt_step` preconditioning.
#[must_use]
pub fn diag_precond_f32_from_jac(jac: &[f64]) -> Vec<f32> {
    let mut h0 = vec![1.0_f64; jac.len()];
    ScaledCoordSpace::diag_h0_from_jac(jac, &mut h0);
    h0.into_iter().map(|v| v as f32).collect()
}

/// Bridges an iterative polish loop to `TrainRunner::inner_opt_step` (`m <= 32`) with optional
/// per-family diagonal preconditioning (`diag_precond`).
#[allow(clippy::too_many_arguments)]
pub fn step_runner_inner_opt_precond(
    runner: &TrainRunner,
    mut uniforms: InnerOptUniforms,
    theta: &mut Vec<f32>,
    grad: &[f32],
    param_bounds: &[[f32; 4]],
    lbfgs_s_y: &mut Vec<[f32; 4]>,
    search_dir: &mut Vec<f32>,
    diag_precond: Option<&[f32]>,
) -> Result<(), TrainError> {
    if diag_precond.is_some() {
        uniforms.mode_flags |= INNER_OPT_MODE_DIAG_PRECOND;
    }
    let out = runner.inner_opt_step(&InnerOptInput {
        uniforms,
        num_candidates: 1,
        theta,
        grad,
        param_bounds,
        lbfgs_s_y,
        search_dir,
        diag_precond,
    })?;
    *theta = out.theta;
    *lbfgs_s_y = out.lbfgs_s_y;
    *search_dir = out.search_dir;
    Ok(())
}

/// Bridges an iterative polish loop to `TrainRunner::inner_opt_step` (`m <= 32`).
pub fn step_runner_inner_opt(
    runner: &TrainRunner,
    uniforms: InnerOptUniforms,
    theta: &mut Vec<f32>,
    grad: &[f32],
    param_bounds: &[[f32; 4]],
    lbfgs_s_y: &mut Vec<[f32; 4]>,
    search_dir: &mut Vec<f32>,
) -> Result<(), TrainError> {
    step_runner_inner_opt_precond(
        runner,
        uniforms,
        theta,
        grad,
        param_bounds,
        lbfgs_s_y,
        search_dir,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::abi::{COORD_TRANSFORM_FREE, INNER_OPT_MODE_LSSBROYDEN};
    use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

    #[test]
    fn test_whitened_lbfgs_undersized_gram_and_backtracking_none_contract() {
        assert!(matches!(
            WhitenedLbfgs::from_uplo_u(3, &[1.0; 2], &[0.0; 3], 1.0e-6),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let opt = TwoLoopLbfgs::new(LbfgsConfig {
            max_iters: 2,
            max_step_norm: 4.0,
            ..LbfgsConfig::default()
        });
        let bounds = [ParamBoundSpec::Free];
        let mut none_evals = 0_usize;
        let mut some_evals = 0_usize;
        let res = opt
            .minimize(&[0.5], &bounds, |x, g_opt| {
                if let Some(g) = g_opt {
                    some_evals += 1;
                    g[0] = 4.0 * x[0];
                } else {
                    none_evals += 1;
                }
                Ok(2.0 * x[0] * x[0])
            })
            .unwrap();
        assert!(none_evals >= 2);
        assert_eq!(some_evals, 1 + res.iters_run);
        assert_eq!(none_evals + some_evals, res.evals);
    }

    #[test]
    fn test_two_loop_lbfgs_whitened_and_runner_bridge() {
        // Rosenbrock minimum at (1.0, 1.0) inside bounded domain.
        let opt = TwoLoopLbfgs::new(LbfgsConfig {
            max_iters: 64,
            ..LbfgsConfig::default()
        });
        let bounds = [
            ParamBoundSpec::Box { lo: -3.0, hi: 3.0 },
            ParamBoundSpec::Box { lo: -3.0, hi: 3.0 },
        ];
        let res = opt
            .minimize(&[-1.2, 1.5], &bounds, |x, g_opt| {
                let t1 = 1.0 - x[0];
                let t2 = x[1] - x[0] * x[0];
                let loss = t1 * t1 + 100.0 * t2 * t2;
                if let Some(g) = g_opt {
                    g[0] = -2.0 * t1 - 400.0 * x[0] * t2;
                    g[1] = 200.0 * t2;
                }
                Ok(loss)
            })
            .unwrap();
        assert!(res.best_loss < 1.0e-6, "loss={}", res.best_loss);
        assert!((res.best_params[0] - 1.0).abs() < 1.0e-2);
        assert!((res.best_params[1] - 1.0).abs() < 1.0e-2);

        // Whitened L-BFGS on ill-conditioned anisotropic quadratic + quartic.
        let target = [1.0_f64, -0.5, 0.25];
        let mut gram = vec![0.0_f64; uplo_u_packed_len(3)];
        gram[uplo_u_index(0, 0)] = 120.0;
        gram[uplo_u_index(0, 1)] = 24.0;
        gram[uplo_u_index(1, 1)] = 12.0;
        gram[uplo_u_index(0, 2)] = -6.0;
        gram[uplo_u_index(1, 2)] = 1.5;
        gram[uplo_u_index(2, 2)] = 4.0;
        let wl = WhitenedLbfgs::from_uplo_u(3, &gram, &[3.5, 2.5, -2.0], 1.0e-6).unwrap();
        let wres = wl
            .minimize(
                LbfgsConfig {
                    max_iters: 28,
                    ..LbfgsConfig::default()
                },
                |th, g_opt| {
                    let d0 = th[0] - target[0];
                    let d1 = th[1] - target[1];
                    let d2 = th[2] - target[2];
                    let loss = 60.0 * d0 * d0
                        + 40.0 * d0.powi(4)
                        + 6.0 * d1 * d1
                        + 2.0 * d2 * d2
                        + 24.0 * d0 * d1;
                    if let Some(g) = g_opt {
                        g[0] = 120.0 * d0 + 160.0 * d0.powi(3) + 24.0 * d1;
                        g[1] = 12.0 * d1 + 24.0 * d0;
                        g[2] = 4.0 * d2;
                    }
                    Ok(loss)
                },
            )
            .unwrap();
        for (a, b) in wres.best_params.iter().zip(&target) {
            assert!((a - b).abs() < 1.0e-4);
        }

        // Exercise TrainRunner::inner_opt_step bridge cleanly.
        let runner = TrainRunner::cpu();
        let mut th = vec![0.4_f32; 4];
        let grad = [0.1_f32; 4];
        let bnd = [[-1.0_f32, 1.0, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)]; 4];
        let mut sy = vec![[0.0_f32; 4]; 12];
        let mut sdir = vec![0.0_f32; 4];
        step_runner_inner_opt(
            &runner,
            InnerOptUniforms {
                num_params: 4,
                history_m: 2,
                step_idx: 0,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN,
                step_size: 0.2,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                grad_clip_norm: 0.0,
            },
            &mut th,
            &grad,
            &bnd,
            &mut sy,
            &mut sdir,
        )
        .unwrap();
        assert!(th.iter().all(|v| v.is_finite() && *v < 0.4));

        let jac = [1.0_f64, 10.0, 100.0, 1.0e4];
        let diag = diag_precond_f32_from_jac(&jac);
        assert!((diag[0] - 1.0).abs() < 1.0e-6);
        assert!((diag[1] - 1.0e-2).abs() < 1.0e-6);
        assert!((diag[2] - 1.0e-4).abs() < 1.0e-8);
        assert!((diag[3] - 1.0e-8).abs() < 1.0e-12);
        let mut th_p = vec![0.4_f32; 4];
        let mut sy_p = vec![[0.0_f32; 4]; 12];
        let mut dir_p = vec![0.0_f32; 4];
        step_runner_inner_opt_precond(
            &runner,
            InnerOptUniforms {
                num_params: 4,
                history_m: 2,
                step_idx: 0,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN,
                step_size: 0.2,
                damping_lambda: 0.0,
                min_bound_eps: 1.0e-4,
                grad_clip_norm: 0.0,
            },
            &mut th_p,
            &grad,
            &bnd,
            &mut sy_p,
            &mut dir_p,
            Some(&diag),
        )
        .unwrap();
        assert!((dir_p[0] / dir_p[1] - 100.0).abs() < 1.0e-2);
    }

    #[test]
    fn test_diagonal_h0_balances_multi_decade_bijection_steps() {
        let bounds = [
            ParamBoundSpec::AsinhBoxSym {
                scale: 4.0,
                limit: 512.0,
            },
            ParamBoundSpec::AsinhBoxSym {
                scale: 10.0,
                limit: 23_520.0,
            },
            ParamBoundSpec::Box { lo: 0.0, hi: 1.0 },
        ];
        let x0 = [0.0, 0.0, 0.5];
        let target = [0.5, -1.5, 0.3];

        // Verify iter-0 H0 diagonal scaling prevents >100x physical step amplification on wide bounds.
        let jac0: Vec<f64> = x0
            .iter()
            .zip(&bounds)
            .map(|(&p, &b)| b.to_physical_and_deriv(b.to_unconstrained(p)).1)
            .collect();
        let h0_diag: Vec<f64> = jac0
            .iter()
            .map(|&j| 1.0 / (j * j).clamp(1.0, 1.0e12))
            .collect();
        let gx0: Vec<f64> = x0.iter().zip(&target).map(|(x, t)| x - t).collect();
        let gu0: Vec<f64> = gx0.iter().zip(&jac0).map(|(g, j)| g * j).collect();
        let mut dir0 = [0.0_f64; 3];
        compute_two_loop_dir(&gu0, &h0_diag, &[], &[], &[], &mut dir0);
        // Physical step delta_x_j = jac0[j] * dir0[j] should equal -gx0[j] for wide-scale coords.
        assert!(((jac0[0] * dir0[0]) + gx0[0]).abs() < 1.0e-12);
        assert!(((jac0[1] * dir0[1]) + gx0[1]).abs() < 1.0e-12);

        let opt = TwoLoopLbfgs::new(LbfgsConfig {
            max_iters: 24,
            max_step_norm: 1.0,
            ..LbfgsConfig::default()
        });
        let res = opt
            .minimize(&x0, &bounds, |x, g_opt| {
                let d0 = x[0] - target[0];
                let d1 = x[1] - target[1];
                let d2 = x[2] - target[2];
                if let Some(g) = g_opt {
                    g[0] = d0;
                    g[1] = d1;
                    g[2] = d2;
                }
                Ok(0.5 * (d0 * d0 + d1 * d1 + d2 * d2))
            })
            .unwrap();
        assert!(res.best_loss < 1.0e-10, "best_loss={}", res.best_loss);
        for (a, b) in res.best_params.iter().zip(&target) {
            assert!((a - b).abs() < 1.0e-4);
        }
    }

    #[test]
    fn test_minimize_scaled_preserves_armijo_and_spd_secant_on_coupled_quadratic() {
        // Minimize 0.5 * [x0 - t0, x1 - t1]^T [[2, 0.5], [0.5, 2]] [x0 - t0, x1 - t1]
        // with extreme scale disparity [1.0, 100.0].
        let bounds = [ParamBoundSpec::Free, ParamBoundSpec::Free];
        let scales = [1.0_f64, 100.0_f64];
        let target = [0.75_f64, -0.025_f64];
        let x0 = [-0.5_f64, 0.5_f64];
        let opt = TwoLoopLbfgs::new(LbfgsConfig {
            max_iters: 48,
            max_step_norm: 2.0,
            ..LbfgsConfig::default()
        });
        assert!(
            opt.minimize_scaled(&x0, &bounds, Some(&[1.0]), |_, _| Ok(0.0))
                .is_err()
        );

        let mut losses = Vec::new();
        let res = opt
            .minimize_scaled(&x0, &bounds, Some(&scales), |x, g_opt| {
                let d0 = x[0] - target[0];
                let d1 = x[1] - target[1];
                let g0 = 2.0 * d0 + 0.5 * d1;
                let g1 = 0.5 * d0 + 2.0 * d1;
                let loss = 0.5 * (d0 * g0 + d1 * g1);
                if let Some(g) = g_opt {
                    g[0] = g0;
                    g[1] = g1;
                    losses.push(loss);
                }
                Ok(loss)
            })
            .unwrap();
        assert!(
            res.best_loss < 1.0e-12,
            "scaled best_loss={}",
            res.best_loss
        );
        assert!(losses.windows(2).all(|w| w[1] <= w[0] + 1.0e-14));
        assert!((res.best_params[0] - target[0]).abs() < 1.0e-6);
        assert!((res.best_params[1] - target[1]).abs() < 1.0e-6);
    }
}
