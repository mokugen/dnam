//! Limited-Memory Self-Scaled Broyden (`L-SSBroyden`, `m = 16..=32`) and dense convex
//! Self-Scaled Broyden (`D_theta <= 512`, `phi in [0, 1]`) with Strong Wolfe line search
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

mod wolfe;

use std::collections::VecDeque;

use crate::error::TrainError;

use super::coords::{OptResult, ParamBoundSpec, ScaledCoordSpace, dot_simd};
use wolfe::{WolfeSpec, strong_wolfe_line_search};

const DEFAULT_SSBROYDEN_HISTORY_M: usize = 24;
const DEFAULT_DENSE_DIM_THRESHOLD: usize = 512;
const DEFAULT_WOLFE_C1: f64 = 1.0e-4;
const DEFAULT_WOLFE_C2: f64 = 0.9;
const DEFAULT_MAX_LINE_SEARCH: usize = 8;
const DEFAULT_TAU_MIN: f64 = 1.0e-3;
const DEFAULT_TAU_MAX: f64 = 1.0e3;

const _: () = assert!(DEFAULT_SSBROYDEN_HISTORY_M >= 16 && DEFAULT_SSBROYDEN_HISTORY_M <= 32);
const _: () = assert!(DEFAULT_DENSE_DIM_THRESHOLD == 512);
const _: () = assert!(DEFAULT_WOLFE_C1 > 0.0 && DEFAULT_WOLFE_C1 < DEFAULT_WOLFE_C2);
const _: () = assert!(DEFAULT_WOLFE_C2 < 1.0 && DEFAULT_MAX_LINE_SEARCH == 8);
const _: () = assert!(DEFAULT_TAU_MIN > 0.0 && DEFAULT_TAU_MIN < DEFAULT_TAU_MAX);

/// Hyperparameters for [`SsBroydenState`] (`L-SSBroyden` and dense self-scaled Broyden).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsBroydenConfig {
    /// Maximum outer quasi-Newton steps.
    pub max_iters: usize,
    /// Secant pair memory capacity (`16..=32` typical in L-mode).
    pub history_m: usize,
    /// Maximum dimension eligible for dense `D × D` inverse Hessian (`<= 512`).
    pub dense_dim_threshold: usize,
    /// Convex Broyden interpolation parameter in `[0.0, 1.0]` for dense mode (`0.0 = SS-BFGS`,
    /// `1.0 = SS-DFP`; limited-memory mode fixes `phi = 0.0`).
    pub broyden_phi: f64,
    /// Strong Wolfe Armijo slope parameter (`default = 1e-4`).
    pub wolfe_c1: f64,
    /// Strong Wolfe curvature slope parameter (`default = 0.9`).
    pub wolfe_c2: f64,
    /// Maximum line search trial evaluations per step (`default = 8`).
    pub max_line_search: usize,
    /// Minimum Oren–Spedicato self-scaling factor (`default = 1e-3`).
    pub tau_min: f64,
    /// Maximum Oren–Spedicato self-scaling factor (`default = 1e3`).
    pub tau_max: f64,
    /// Self-scaling denominator ridge (`default = 1e-12`).
    pub damping_lambda: f64,
    /// Curvature acceptance threshold `s^T y > curvature_tol * ||s|| * ||y||` (`default = 1e-10`).
    pub curvature_tol: f64,
    /// Relative objective change threshold for stall termination.
    pub rel_tol: f64,
    /// Consecutive small-improvement patience before stopping.
    pub patience: usize,
    /// Trust cap on initial unit-step Euclidean norm in `u`-space.
    pub max_step_norm: f64,
}

impl Default for SsBroydenConfig {
    fn default() -> Self {
        Self {
            max_iters: 24,
            history_m: DEFAULT_SSBROYDEN_HISTORY_M,
            dense_dim_threshold: DEFAULT_DENSE_DIM_THRESHOLD,
            broyden_phi: 0.25,
            wolfe_c1: DEFAULT_WOLFE_C1,
            wolfe_c2: DEFAULT_WOLFE_C2,
            max_line_search: DEFAULT_MAX_LINE_SEARCH,
            tau_min: DEFAULT_TAU_MIN,
            tau_max: DEFAULT_TAU_MAX,
            damping_lambda: 1.0e-12,
            curvature_tol: 1.0e-10,
            rel_tol: 1.0e-9,
            patience: 4,
            max_step_norm: 8.0,
        }
    }
}

#[derive(Clone, Debug)]
enum SsBroydenRep {
    Limited {
        m_cap: usize,
        pairs: VecDeque<(Vec<f64>, Vec<f64>, f64)>,
    },
    Dense {
        h_inv: Vec<f64>,
    },
}

/// Stateful Self-Scaled Broyden inverse-Hessian approximator (`dense` when `dim <= dense_dim_threshold`,
/// otherwise `L-SSBroyden` two-loop recursion).
#[derive(Clone, Debug)]
pub struct SsBroydenState {
    dim: usize,
    rep: SsBroydenRep,
}

impl SsBroydenState {
    /// Initializes an identity/diagonal quasi-Newton state according to `cfg`.
    #[must_use]
    pub fn new(dim: usize, cfg: &SsBroydenConfig, h0_diag: Option<&[f64]>) -> Self {
        let dense = dim <= cfg.dense_dim_threshold && cfg.dense_dim_threshold > 0 && dim > 0;
        let rep = if dense {
            let mut h_inv = vec![0.0_f64; dim * dim];
            for i in 0..dim {
                let d0 = h0_diag
                    .and_then(|d| d.get(i).copied())
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or(1.0);
                h_inv[i * dim + i] = d0;
            }
            SsBroydenRep::Dense { h_inv }
        } else {
            let m_cap = cfg.history_m.clamp(1, 32);
            SsBroydenRep::Limited {
                m_cap,
                pairs: VecDeque::with_capacity(m_cap),
            }
        };
        Self { dim, rep }
    }

    /// Resets curvature memory back to the diagonal/identity baseline (mandatory after periodic
    /// block-coordinate VarPro readout refreshes per M19 spec).
    pub fn reset_history(&mut self, h0_diag: Option<&[f64]>) {
        match &mut self.rep {
            SsBroydenRep::Limited { pairs, .. } => pairs.clear(),
            SsBroydenRep::Dense { h_inv } => {
                h_inv.fill(0.0);
                for i in 0..self.dim {
                    let d0 = h0_diag
                        .and_then(|d| d.get(i).copied())
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .unwrap_or(1.0);
                    h_inv[i * self.dim + i] = d0;
                }
            }
        }
    }

    /// Computes search direction `dir = -H * grad` using Oren–Spedicato self-scaling.
    pub fn compute_direction(
        &self,
        grad: &[f64],
        h0_diag: &[f64],
        cfg: &SsBroydenConfig,
        dir: &mut [f64],
    ) {
        let n = self.dim;
        match &self.rep {
            SsBroydenRep::Dense { h_inv } => {
                for i in 0..n {
                    dir[i] = -dot_simd(&h_inv[i * n..(i + 1) * n], grad);
                }
            }
            SsBroydenRep::Limited { pairs, .. } => {
                let m = pairs.len();
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
                for (i, (s_i, y_i, rho_i)) in pairs.iter().enumerate().rev() {
                    alpha[i] = *rho_i * dot_simd(s_i, dir);
                    for (dj, &yj) in dir.iter_mut().zip(y_i) {
                        *dj -= alpha[i] * yj;
                    }
                }
                if let Some((sl, yl, _)) = pairs.back() {
                    let y_h0_y = yl
                        .iter()
                        .zip(h0_diag)
                        .map(|(&y, &h0)| h0 * y * y)
                        .sum::<f64>();
                    let denom = (y_h0_y + cfg.damping_lambda.max(1.0e-18)).max(1.0e-18);
                    let tau_k = (dot_simd(sl, yl) / denom).clamp(cfg.tau_min, cfg.tau_max);
                    for (dj, &h0) in dir.iter_mut().zip(h0_diag) {
                        *dj *= tau_k * h0;
                    }
                }
                for (i, (s_i, y_i, rho_i)) in pairs.iter().enumerate() {
                    let beta = *rho_i * dot_simd(y_i, dir);
                    for (dj, &sj) in dir.iter_mut().zip(s_i) {
                        *dj += (alpha[i] - beta) * sj;
                    }
                }
            }
        }
    }

    /// Incorporates secant pair `(s, y)` into curvature memory if gated by `s^T y > tol * ||s|| * ||y||`.
    /// Returns `true` when accepted.
    pub fn update(&mut self, s: &[f64], y: &[f64], cfg: &SsBroydenConfig) -> bool {
        let sy = dot_simd(s, y);
        let sn = dot_simd(s, s).sqrt();
        let yn = dot_simd(y, y).sqrt();
        if !sy.is_finite() || sy <= cfg.curvature_tol * sn * yn || sy <= 1.0e-18 {
            return false;
        }
        let n = self.dim;
        match &mut self.rep {
            SsBroydenRep::Limited { m_cap, pairs } => {
                if pairs.len() == *m_cap {
                    pairs.pop_front();
                }
                pairs.push_back((s.to_vec(), y.to_vec(), 1.0 / sy));
                true
            }
            SsBroydenRep::Dense { h_inv } => {
                let mut hy = vec![0.0_f64; n];
                for (i, dst) in hy.iter_mut().enumerate() {
                    *dst = dot_simd(&h_inv[i * n..(i + 1) * n], y);
                }
                let yhy = dot_simd(y, &hy);
                if !yhy.is_finite() || yhy <= 1.0e-18 {
                    return false;
                }
                let denom = (yhy + cfg.damping_lambda.max(1.0e-18)).max(1.0e-18);
                let tau_k = (sy / denom).clamp(cfg.tau_min, cfg.tau_max);
                let yhsy = tau_k * yhy;
                let inv_sy = 1.0 / sy;
                let inv_yhsy = 1.0 / yhsy;
                let root_yhsy = yhsy.sqrt();
                let bfgs_w = 1.0 - cfg.broyden_phi.clamp(0.0, 1.0);
                let v: Vec<f64> = (0..n)
                    .map(|i| root_yhsy * (s[i] * inv_sy - (tau_k * hy[i]) * inv_yhsy))
                    .collect();

                for i in 0..n {
                    let hsy_i = tau_k * hy[i];
                    let si = s[i];
                    let vi = v[i];
                    for j in i..n {
                        let hsy_j = tau_k * hy[j];
                        let cur = tau_k * h_inv[i * n + j];
                        let dfp = cur + (si * s[j]) * inv_sy - (hsy_i * hsy_j) * inv_yhsy;
                        let val = dfp + bfgs_w * vi * v[j];
                        h_inv[i * n + j] = val;
                        h_inv[j * n + i] = val;
                    }
                }
                true
            }
        }
    }

    /// Runs `cfg.max_iters` Strong-Wolfe `SsBroyden` / `L-SSBroyden` steps over `ScaledCoordSpace`.
    pub fn minimize_scaled<F>(
        init_physical: &[f64],
        bounds: &[ParamBoundSpec],
        coord_scales: Option<&[f64]>,
        cfg: &SsBroydenConfig,
        mut eval: F,
    ) -> Result<OptResult, TrainError>
    where
        F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
    {
        let (space, mut u) = ScaledCoordSpace::new(
            "SsBroydenState::minimize_scaled",
            init_physical,
            bounds,
            coord_scales,
        )?;
        let dim = space.dim();
        let mut x = vec![0.0_f64; dim];
        let mut jac = vec![0.0_f64; dim];
        let mut h0_diag = vec![1.0_f64; dim];
        let mut gx = vec![0.0_f64; dim];
        let mut gu = vec![0.0_f64; dim];

        space.decode(&u, &mut x, &mut jac);
        ScaledCoordSpace::diag_h0_from_jac(&jac, &mut h0_diag);
        let mut loss = eval(&x, Some(&mut gx))?;
        if !loss.is_finite() || !gx.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput {
                context: "SsBroydenState::minimize_scaled(init_eval)",
            });
        }
        space.pullback_grad(&gx, &jac, &mut gu);

        let mut best_params = x.clone();
        let mut best_loss = loss;
        let mut evals = 1_usize;
        let mut iters_run = 0_usize;

        let mut state = Self::new(dim, cfg, Some(&h0_diag));
        let mut has_curvature = false;
        let mut dir = vec![0.0_f64; dim];
        let mut small_streak = 0_usize;

        for _ in 0..cfg.max_iters {
            state.compute_direction(&gu, &h0_diag, cfg, &mut dir);
            let mut g_dot_d = dot_simd(&gu, &dir);
            if !g_dot_d.is_finite() || g_dot_d >= -1.0e-18 {
                state.reset_history(Some(&h0_diag));
                has_curvature = false;
                for ((d, &g), &h0) in dir.iter_mut().zip(&gu).zip(&h0_diag) {
                    *d = -h0 * g;
                }
                g_dot_d = dot_simd(&gu, &dir);
                if !g_dot_d.is_finite() || g_dot_d >= -1.0e-18 {
                    break;
                }
            }

            let d_norm = dot_simd(&dir, &dir).sqrt().max(1.0e-12);
            let alpha_max = (cfg.max_step_norm / d_norm).max(1.0);
            let alpha_init = if !has_curvature {
                (0.1 / d_norm).min(1.0)
            } else {
                (cfg.max_step_norm / d_norm).min(1.0)
            };
            let ls = strong_wolfe_line_search(
                &space,
                &u,
                &dir,
                &WolfeSpec {
                    f0: loss,
                    g_dot_d0: g_dot_d,
                    c1: cfg.wolfe_c1,
                    c2: cfg.wolfe_c2,
                    alpha_init,
                    alpha_max,
                    max_evals: cfg.max_line_search,
                },
                &mut gx,
                &mut eval,
            )?;
            evals += ls.evals_used;
            let mut accepted = ls.accepted;
            if accepted.is_none() && has_curvature {
                state.reset_history(Some(&h0_diag));
                has_curvature = false;
                for ((d, &g), &h0) in dir.iter_mut().zip(&gu).zip(&h0_diag) {
                    *d = -h0 * g;
                }
                g_dot_d = dot_simd(&gu, &dir);
                if g_dot_d.is_finite() && g_dot_d < -1.0e-18 {
                    let r_norm = dot_simd(&dir, &dir).sqrt().max(1.0e-12);
                    let r_max = (cfg.max_step_norm / r_norm).max(1.0);
                    let r_init = (0.1 / r_norm).min(1.0);
                    let ls_retry = strong_wolfe_line_search(
                        &space,
                        &u,
                        &dir,
                        &WolfeSpec {
                            f0: loss,
                            g_dot_d0: g_dot_d,
                            c1: cfg.wolfe_c1,
                            c2: cfg.wolfe_c2,
                            alpha_init: r_init,
                            alpha_max: r_max,
                            max_evals: cfg.max_line_search,
                        },
                        &mut gx,
                        &mut eval,
                    )?;
                    evals += ls_retry.evals_used;
                    accepted = ls_retry.accepted;
                }
            }
            let Some((alpha, trial)) = accepted else {
                break;
            };

            let prev_loss = loss;
            let s: Vec<f64> = dir.iter().map(|&d| alpha * d).collect();
            let y: Vec<f64> = trial.grad_u.iter().zip(&gu).map(|(a, b)| a - b).collect();
            if state.update(&s, &y, cfg) {
                has_curvature = true;
            }

            for (dst, &sk) in u.iter_mut().zip(&s) {
                *dst += sk;
            }
            ScaledCoordSpace::diag_h0_from_jac(&trial.jac, &mut h0_diag);
            gu = trial.grad_u;
            loss = trial.loss;
            iters_run += 1;
            if loss < best_loss {
                best_loss = loss;
                best_params = trial.x;
            }
            let impr = (prev_loss - loss).abs();
            if impr <= cfg.rel_tol * prev_loss.abs().max(1.0e-12) {
                small_streak += 1;
            } else {
                small_streak = 0;
            }
            if small_streak >= cfg.patience {
                break;
            }
        }

        Ok(OptResult {
            best_params,
            best_loss,
            iters_run,
            evals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock_eval(x: &[f64], grad_opt: Option<&mut [f64]>) -> Result<f64, TrainError> {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let loss = a * a + 100.0 * b * b;
        if let Some(g) = grad_opt {
            g[0] = -2.0 * a - 400.0 * x[0] * b;
            g[1] = 200.0 * b;
        }
        Ok(loss)
    }

    #[test]
    fn test_dense_secant_identity_and_broyden_equivalence_to_limited_bfgs() {
        let cfg_dense_bfgs = SsBroydenConfig {
            broyden_phi: 0.0,
            dense_dim_threshold: 8,
            max_iters: 80,
            ..SsBroydenConfig::default()
        };
        let mut st = SsBroydenState::new(3, &cfg_dense_bfgs, None);
        let s = [0.25_f64, -0.15, 0.40];
        let y = [1.10_f64, -0.35, 0.90];
        assert!(st.update(&s, &y, &cfg_dense_bfgs));
        // Reject negative or zero curvature cleanly.
        assert!(!st.update(&s, &[-1.0, 0.0, 0.0], &cfg_dense_bfgs));

        if let SsBroydenRep::Dense { ref h_inv } = st.rep {
            for i in 0..3 {
                let mut hy_i = 0.0_f64;
                for j in 0..3 {
                    assert!((h_inv[i * 3 + j] - h_inv[j * 3 + i]).abs() < 1.0e-13);
                    hy_i += h_inv[i * 3 + j] * y[j];
                }
                assert!((hy_i - s[i]).abs() < 1.0e-12, "secant H*y != s at {i}");
            }
        } else {
            panic!("expected Dense mode");
        }

        // Compare phi = 0 dense Self-Scaled BFGS vs limited-memory mode on 2D Rosenbrock.
        let bounds = [ParamBoundSpec::Free, ParamBoundSpec::Free];
        let cfg_limited = SsBroydenConfig {
            dense_dim_threshold: 0,
            history_m: 32,
            max_iters: 80,
            ..SsBroydenConfig::default()
        };
        let r_dense = SsBroydenState::minimize_scaled(
            &[-1.2, 1.0],
            &bounds,
            None,
            &cfg_dense_bfgs,
            rosenbrock_eval,
        )
        .unwrap();
        let r_lim = SsBroydenState::minimize_scaled(
            &[-1.2, 1.0],
            &bounds,
            None,
            &cfg_limited,
            rosenbrock_eval,
        )
        .unwrap();
        assert!(
            r_dense.best_loss < 1.0e-10,
            "dense loss={}",
            r_dense.best_loss
        );
        assert!(r_lim.best_loss < 1.0e-10, "lim loss={}", r_lim.best_loss);
        assert!((r_dense.best_params[0] - 1.0).abs() < 1.0e-4);
        assert!((r_lim.best_params[0] - 1.0).abs() < 1.0e-4);

        // Convex Broyden phi = 0.25 + reset_history check + conditioned bounds.
        let cfg_broyden = SsBroydenConfig {
            max_iters: 80,
            ..SsBroydenConfig::default()
        };
        let box_bounds = [
            ParamBoundSpec::Box { lo: -2.0, hi: 3.0 },
            ParamBoundSpec::LogBox {
                lo: 1.0e-3,
                hi: 4.0,
            },
        ];
        let r_broy = SsBroydenState::minimize_scaled(
            &[-0.5, 0.2],
            &box_bounds,
            Some(&[1.0, 20.0]),
            &cfg_broyden,
            rosenbrock_eval,
        )
        .unwrap();
        assert!(
            r_broy.best_loss < 1.0e-8,
            "broyden loss={}",
            r_broy.best_loss
        );
        st.reset_history(None);
        if let SsBroydenRep::Dense { ref h_inv } = st.rep {
            assert_eq!(h_inv[0], 1.0);
            assert_eq!(h_inv[1], 0.0);
        }

        // Steeply scaled ill-conditioned quadratic (`0.5 * (1e4 * x0^2 + x1^2)`) must not abort at
        // `iters = 1` when `max_line_search = 8` and `max_step_norm = 4.0`.
        let steep_cfg = SsBroydenConfig {
            max_iters: 12,
            dense_dim_threshold: 0,
            broyden_phi: 0.0,
            max_line_search: 8,
            max_step_norm: 4.0,
            ..SsBroydenConfig::default()
        };
        let r_steep = SsBroydenState::minimize_scaled(
            &[1.0, 1.0],
            &bounds,
            None,
            &steep_cfg,
            |px, grad_opt| {
                let val = 0.5 * (1.0e4 * px[0] * px[0] + px[1] * px[1]);
                if let Some(g) = grad_opt {
                    g[0] = 1.0e4 * px[0];
                    g[1] = px[1];
                }
                Ok(val)
            },
        )
        .unwrap();
        assert!(
            r_steep.iters_run > 1,
            "steep quadratic aborted too early at iters={}",
            r_steep.iters_run
        );
        assert!(
            r_steep.best_loss < 1.0e-8,
            "steep quadratic failed to converge: best_loss={}",
            r_steep.best_loss
        );
    }
}
