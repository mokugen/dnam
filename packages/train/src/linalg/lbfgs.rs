//! Smooth unconstrained parameter codecs, two-loop bounded L-BFGS (`m <= 10`),
//! Cholesky-whitened L-BFGS (`WhitenedLbfgs`), and GPU/CPU `TrainRunner::inner_opt_step` bridge
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use crate::error::TrainError;
use crate::gpu::abi::InnerOptUniforms;
use crate::math::{DynamicTrainer, TrainerMathBackend, TrainerTask};
use crate::runner::{InnerOptInput, TrainRunner};

use super::cholesky::factorize_uplo_u_cholesky;

/// Constraint mapping between a constrained physical scalar and unconstrained R.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamBoundSpec {
    /// Unconstrained scalar parameter.
    Free,
    /// Closed box `[lo, hi]` mapped smoothly via logit/sigmoid.
    Box {
        /// Strict lower bound.
        lo: f64,
        /// Strict upper bound.
        hi: f64,
    },
    /// Positive log-scaled box `[lo, hi]` mapped smoothly in log space.
    LogBox {
        /// Positive lower bound (`> 0.0`).
        lo: f64,
        /// Positive upper bound (`> lo`).
        hi: f64,
    },
    /// Complement `1.0 - c` with `c in [lo_c, hi_c]` mapped in log space (pole radius near `1.0`).
    ComplementLog {
        /// Positive lower bound on `1.0 - x`.
        lo_c: f64,
        /// Positive upper bound on `1.0 - x`.
        hi_c: f64,
    },
}

impl ParamBoundSpec {
    /// Encodes a physical parameter `v` into an unconstrained coordinate `u`.
    #[inline]
    #[must_use]
    pub fn to_unconstrained(self, v: f64) -> f64 {
        match self {
            Self::Free => v,
            Self::Box { lo, hi } => {
                let span = (hi - lo).max(1.0e-12);
                let vc = v.clamp(lo + 1.0e-7 * span, hi - 1.0e-7 * span);
                let p = ((vc - lo) / span).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::LogBox { lo, hi } => {
                let (la, lb) = (lo.max(1.0e-15).ln(), hi.max(lo * 1.000_001).ln());
                let vc = v.clamp(lo * (1.0 + 1.0e-7), hi * (1.0 - 1.0e-7));
                let p = ((vc.ln() - la) / (lb - la).max(1.0e-12)).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::ComplementLog { lo_c, hi_c } => {
                Self::LogBox { lo: lo_c, hi: hi_c }.to_unconstrained(1.0 - v)
            }
        }
    }

    /// Decodes unconstrained `u` to `(physical, d(physical)/du)`.
    #[inline]
    #[must_use]
    pub fn to_physical_and_deriv(self, u: f64) -> (f64, f64) {
        match self {
            Self::Free => (u, 1.0),
            Self::Box { lo, hi } => {
                let saturated = !u.is_finite() || u.abs() > 24.0;
                let uc = u.clamp(-24.0, 24.0);
                let s = 1.0 / (1.0 + (-uc).exp());
                let span = hi - lo;
                let deriv = if saturated { 0.0 } else { span * s * (1.0 - s) };
                (lo + span * s, deriv)
            }
            Self::LogBox { lo, hi } => {
                let saturated = !u.is_finite() || u.abs() > 24.0;
                let uc = u.clamp(-24.0, 24.0);
                let s = 1.0 / (1.0 + (-uc).exp());
                let (la, lb) = (lo.max(1.0e-15).ln(), hi.max(lo * 1.000_001).ln());
                let lspan = lb - la;
                let v = (la + lspan * s).exp();
                let deriv = if saturated {
                    0.0
                } else {
                    v * lspan * s * (1.0 - s)
                };
                (v, deriv)
            }
            Self::ComplementLog { lo_c, hi_c } => {
                let (c, dc) = Self::LogBox { lo: lo_c, hi: hi_c }.to_physical_and_deriv(u);
                (1.0 - c, if dc == 0.0 { 0.0 } else { -dc })
            }
        }
    }

    /// Hard-projects a scalar cleanly inside the spec domain.
    #[inline]
    #[must_use]
    pub fn project_physical(self, v: f64) -> f64 {
        match self {
            Self::Free => {
                if v.is_finite() {
                    v
                } else {
                    0.0
                }
            }
            Self::Box { lo, hi } => v.clamp(lo, hi),
            Self::LogBox { lo, hi } => v.clamp(lo, hi),
            Self::ComplementLog { lo_c, hi_c } => v.clamp(1.0 - hi_c, 1.0 - lo_c),
        }
    }
}

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

/// Result returned by [`TwoLoopLbfgs::minimize`] or [`WhitenedLbfgs::minimize`].
#[derive(Clone, Debug, PartialEq)]
pub struct LbfgsResult {
    /// Best parameter vector observed across initial evaluation and accepted steps.
    pub best_params: Vec<f64>,
    /// Minimum objective value achieved at `best_params`.
    pub best_loss: f64,
    /// Number of accepted outer iterations completed.
    pub iters_run: usize,
    /// Number of objective+gradient evaluations performed.
    pub evals: usize,
}

fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
    struct DotTask<'a>(&'a [f64], &'a [f64]);
    impl TrainerTask<f64> for DotTask<'_> {
        fn run<B: TrainerMathBackend>(self) -> f64 {
            B::dot_f64(self.0, self.1)
        }
    }
    DynamicTrainer::detect().dispatch(DotTask(a, b))
}

fn compute_two_loop_dir(
    grad: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
    dir: &mut [f64],
) {
    for (d, &g) in dir.iter_mut().zip(grad) {
        *d = -g;
    }
    let m = s_hist.len();
    if m == 0 {
        return;
    }
    let mut alpha = vec![0.0_f64; m];
    for i in (0..m).rev() {
        alpha[i] = rho_hist[i] * dot_simd(&s_hist[i], dir);
        for (dj, &yj) in dir.iter_mut().zip(&y_hist[i]) {
            *dj -= alpha[i] * yj;
        }
    }
    if let (Some(sl), Some(yl)) = (s_hist.last(), y_hist.last()) {
        let yy = dot_simd(yl, yl).max(1.0e-18);
        let scale = (dot_simd(sl, yl) / yy).clamp(1.0e-8, 1.0e6);
        for dj in dir.iter_mut() {
            *dj *= scale;
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
        mut eval: F,
    ) -> Result<LbfgsResult, TrainError>
    where
        F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
    {
        let dim = init_physical.len();
        if dim != bounds.len() {
            return Err(TrainError::InvalidDimensions {
                context: "TwoLoopLbfgs::minimize(bounds)",
                expected: dim,
                actual: bounds.len(),
            });
        }
        if dim == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "TwoLoopLbfgs::minimize(dim)",
                expected: 1,
                actual: 0,
            });
        }
        if !init_physical.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput {
                context: "TwoLoopLbfgs::minimize(init_physical)",
            });
        }

        let m_cap = self.config.history_m.clamp(1, 10);
        let mut u: Vec<f64> = init_physical
            .iter()
            .zip(bounds)
            .map(|(&p, &b)| b.to_unconstrained(p))
            .collect();
        let mut x = vec![0.0_f64; dim];
        let mut dxd_u = vec![0.0_f64; dim];
        let decode_all = |u_vec: &[f64], x_out: &mut [f64], jac_out: &mut [f64]| {
            for i in 0..dim {
                let (xi, ji) = bounds[i].to_physical_and_deriv(u_vec[i]);
                x_out[i] = xi;
                jac_out[i] = ji;
            }
        };

        decode_all(&u, &mut x, &mut dxd_u);
        let mut grad_x = vec![0.0_f64; dim];
        let mut loss = eval(&x, Some(&mut grad_x))?;
        if !loss.is_finite() || !grad_x.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput {
                context: "TwoLoopLbfgs::minimize(init_eval)",
            });
        }
        let mut grad_u: Vec<f64> = grad_x.iter().zip(&dxd_u).map(|(g, j)| g * j).collect();

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
            compute_two_loop_dir(&grad_u, &s_hist, &y_hist, &rho_hist, &mut dir);
            let mut g_dot_d = dot_simd(&grad_u, &dir);
            if !g_dot_d.is_finite() || g_dot_d >= -1.0e-18 {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                for (d, &g) in dir.iter_mut().zip(&grad_u) {
                    *d = -g;
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
                decode_all(&cand_u, &mut cand_x, &mut cand_jac);
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
                        let cand_gu: Vec<f64> =
                            cand_gx.iter().zip(&cand_jac).map(|(g, j)| g * j).collect();
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

/// Bridges an iterative polish loop to `TrainRunner::inner_opt_step` (`m <= 8`).
pub fn step_runner_inner_opt(
    runner: &TrainRunner,
    uniforms: InnerOptUniforms,
    theta: &mut Vec<f32>,
    grad: &[f32],
    param_bounds: &[[f32; 2]],
    lbfgs_s_y: &mut Vec<[f32; 4]>,
    search_dir: &mut Vec<f32>,
) -> Result<(), TrainError> {
    let out = runner.inner_opt_step(&InnerOptInput {
        uniforms,
        theta,
        grad,
        param_bounds,
        lbfgs_s_y,
        search_dir,
    })?;
    *theta = out.theta;
    *lbfgs_s_y = out.lbfgs_s_y;
    *search_dir = out.search_dir;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

    #[test]
    fn test_param_bound_codec_roundtrip_and_fd_jacobian() {
        let specs = [
            (ParamBoundSpec::Free, -1.75),
            (ParamBoundSpec::Box { lo: -2.0, hi: 3.5 }, 0.85),
            (
                ParamBoundSpec::LogBox {
                    lo: 1.0e-4,
                    hi: 10.0,
                },
                0.025,
            ),
            (
                ParamBoundSpec::ComplementLog {
                    lo_c: 5.0e-4,
                    hi_c: 0.75,
                },
                0.985,
            ),
        ];
        for (spec, v) in specs {
            let u = spec.to_unconstrained(v);
            let (v2, dv_du) = spec.to_physical_and_deriv(u);
            assert!((v2 - v).abs() <= 1.0e-9, "v={v} v2={v2}");
            let eps = 1.0e-6;
            let fd = (spec.to_physical_and_deriv(u + eps).0
                - spec.to_physical_and_deriv(u - eps).0)
                / (2.0 * eps);
            assert!(
                (dv_du - fd).abs() <= 1.0e-7 * dv_du.abs().max(1.0),
                "dv_du={dv_du} fd={fd}"
            );
            assert!((spec.project_physical(v2) - v2).abs() <= 1.0e-12);
        }
        assert_eq!(ParamBoundSpec::Free.project_physical(f64::NAN), 0.0);
        assert_eq!(
            ParamBoundSpec::Box { lo: -1.0, hi: 2.0 }
                .to_physical_and_deriv(30.0)
                .1,
            0.0
        );
        assert_eq!(
            ParamBoundSpec::LogBox { lo: 0.1, hi: 10.0 }
                .to_physical_and_deriv(-30.0)
                .1,
            0.0
        );
    }

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
        let bnd = [[-1.0_f32, 1.0]; 4];
        let mut sy = vec![[0.0_f32; 4]; 12];
        let mut sdir = vec![0.0_f32; 4];
        step_runner_inner_opt(
            &runner,
            InnerOptUniforms {
                num_params: 4,
                history_m: 2,
                step_idx: 0,
                line_search_iter: 0,
                step_size: 0.2,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                _pad: 0,
            },
            &mut th,
            &grad,
            &bnd,
            &mut sy,
            &mut sdir,
        )
        .unwrap();
        assert!(th.iter().all(|v| v.is_finite() && *v < 0.4));
    }
}
