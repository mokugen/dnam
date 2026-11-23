//! Conditioned `u`-space decoupled [`AdamWState`] optimizer with cosine LR warmup and
//! arena-scoped deterministic `PCGrad` (`ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::PI;

use crate::error::TrainError;

use super::coords::{OptResult, ParamBoundSpec, ScaledCoordSpace, U_SATURATION_LIMIT, dot_simd};

const ADAMW_DEFAULT_BETA1: f64 = 0.9;
const ADAMW_DEFAULT_BETA2: f64 = 0.999;
const ADAMW_DEFAULT_CLIP_NORM: f64 = 10.0;
const ADAMW_DEFAULT_LR_MAX: f64 = 3.0e-3;
const ADAMW_DEFAULT_LR_MIN: f64 = 1.0e-4;
const PCGRAD_EPS: f64 = 1.0e-18;

const _: () = assert!(ADAMW_DEFAULT_BETA1 > 0.0 && ADAMW_DEFAULT_BETA1 < 1.0);
const _: () = assert!(ADAMW_DEFAULT_BETA2 > ADAMW_DEFAULT_BETA1 && ADAMW_DEFAULT_BETA2 < 1.0);
const _: () = assert!(ADAMW_DEFAULT_CLIP_NORM == 10.0);
const _: () = assert!(ADAMW_DEFAULT_LR_MIN > 0.0 && ADAMW_DEFAULT_LR_MIN < ADAMW_DEFAULT_LR_MAX);
const _: () = assert!(PCGRAD_EPS > 0.0);

/// Hyperparameters for [`AdamWState`] in scaled unconstrained coordinates (`u`-space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWConfig {
    /// First-moment exponential decay (`default = 0.9`).
    pub beta1: f64,
    /// Second-moment exponential decay (`default = 0.999`).
    pub beta2: f64,
    /// Numerical denominator epsilon (`default = 1e-8`).
    pub eps: f64,
    /// Peak cosine-schedule learning rate (`default = 3e-3`).
    pub lr_max: f64,
    /// Terminal cosine-schedule learning rate (`default = 1e-4`).
    pub lr_min: f64,
    /// Linear warmup step count (`default = 10`).
    pub warmup_steps: usize,
    /// Decoupled `u`-space weight decay (`default = 0.0`, zero decay in `u`-space per M19 spec).
    pub weight_decay: f64,
    /// Global L2 gradient norm ceiling in `u`-space (`default = 10.0`).
    pub grad_clip_norm: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            beta1: ADAMW_DEFAULT_BETA1,
            beta2: ADAMW_DEFAULT_BETA2,
            eps: 1.0e-8,
            lr_max: ADAMW_DEFAULT_LR_MAX,
            lr_min: ADAMW_DEFAULT_LR_MIN,
            warmup_steps: 10,
            weight_decay: 0.0,
            grad_clip_norm: ADAMW_DEFAULT_CLIP_NORM,
        }
    }
}

/// Evaluates linear-warmup + cosine-decay learning rate at zero-based iteration `step` out of
/// `total_steps` (`lr_max -> lr_min`).
#[inline]
#[must_use]
pub fn cosine_lr_with_warmup(step: usize, total_steps: usize, cfg: &AdamWConfig) -> f64 {
    if total_steps <= 1 {
        return cfg.lr_max;
    }
    let warmup = cfg.warmup_steps.min(total_steps.saturating_sub(1));
    if warmup > 0 && step < warmup {
        return cfg.lr_max * ((step + 1) as f64) / (warmup as f64);
    }
    let denom = total_steps.saturating_sub(warmup).max(1) as f64;
    let progress = ((step.saturating_sub(warmup)) as f64 / denom).clamp(0.0, 1.0);
    cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1.0 + (PI * progress).cos())
}

/// Stateful `AdamW` moment tracker for chunked TBPTT, periodic VarPro refreshes, and QAT.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamWState {
    m: Vec<f64>,
    v: Vec<f64>,
    step_count: usize,
}

impl AdamWState {
    /// Allocates zero-initialized first and second moment buffers for `dim` parameters.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            m: vec![0.0_f64; dim],
            v: vec![0.0_f64; dim],
            step_count: 0,
        }
    }

    /// Returns parameter dimension covered by this state.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.m.len()
    }

    /// Returns total completed update steps since creation or full reset.
    #[inline]
    #[must_use]
    pub fn steps(&self) -> usize {
        self.step_count
    }

    /// Clears all first and second moments and resets step counter to `0`.
    pub fn reset_moments(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.step_count = 0;
    }

    /// Zeroes `m[i]` and `v[i]` selectively where `mask[i]` is `true` (used after periodic VarPro
    /// readout synchronization so readout-coupled directions restart smoothly).
    pub fn reset_moments_mask(&mut self, mask: &[bool]) {
        for ((mi, vi), &flag) in self.m.iter_mut().zip(&mut self.v).zip(mask) {
            if flag {
                *mi = 0.0;
                *vi = 0.0;
            }
        }
    }

    /// Performs one globally clipped `AdamW` step directly on scaled `u` coordinates using
    /// `grad_u`, clamping each element within `[-U_SATURATION_LIMIT * scale_i, +U_SATURATION_LIMIT * scale_i]`.
    pub fn step_u(
        &mut self,
        u: &mut [f64],
        grad_u: &mut [f64],
        lr: f64,
        cfg: &AdamWConfig,
        coord_scales: Option<&[f64]>,
    ) {
        let dim = self.m.len().min(u.len()).min(grad_u.len());
        if dim == 0 || !lr.is_finite() || lr <= 0.0 {
            return;
        }
        for g in &mut grad_u[..dim] {
            if !g.is_finite() {
                *g = 0.0;
            }
        }
        if cfg.grad_clip_norm > 0.0 && cfg.grad_clip_norm.is_finite() {
            let norm = dot_simd(&grad_u[..dim], &grad_u[..dim]).sqrt();
            if norm > cfg.grad_clip_norm {
                let scale = cfg.grad_clip_norm / norm;
                for g in &mut grad_u[..dim] {
                    *g *= scale;
                }
            }
        }

        self.step_count = self.step_count.saturating_add(1);
        let t = self.step_count as i32;
        let b1 = cfg.beta1.clamp(0.0, 0.99999);
        let b2 = cfg.beta2.clamp(0.0, 0.999999);
        let bc1 = (1.0 - b1.powi(t)).max(1.0e-12);
        let bc2 = (1.0 - b2.powi(t)).max(1.0e-12);
        let eps = cfg.eps.max(1.0e-15);
        let decay_factor = if cfg.weight_decay > 0.0 && cfg.weight_decay.is_finite() {
            (1.0 - lr * cfg.weight_decay).max(0.0)
        } else {
            1.0
        };

        for i in 0..dim {
            let g = grad_u[i];
            let mi = b1 * self.m[i] + (1.0 - b1) * g;
            let vi = b2 * self.v[i] + (1.0 - b2) * g * g;
            self.m[i] = mi;
            self.v[i] = vi;
            let m_hat = mi / bc1;
            let v_hat = (vi / bc2).max(0.0);
            let s = coord_scales
                .and_then(|cs| cs.get(i).copied())
                .filter(|&v| v.is_finite() && v > 0.0)
                .unwrap_or(1.0);
            let u_lim = U_SATURATION_LIMIT * s;
            let updated = u[i] * decay_factor - lr * m_hat / (v_hat.sqrt() + eps);
            if updated.is_finite() {
                u[i] = updated.clamp(-u_lim, u_lim);
            }
        }
    }

    /// Runs `total_steps` of conditioned `AdamW` with cosine warmup in scaled unconstrained space,
    /// returning the best observed physical parameter vector and objective value.
    pub fn minimize_scaled<F>(
        init_physical: &[f64],
        bounds: &[ParamBoundSpec],
        coord_scales: Option<&[f64]>,
        total_steps: usize,
        cfg: &AdamWConfig,
        mut eval: F,
    ) -> Result<OptResult, TrainError>
    where
        F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
    {
        let (space, mut u) = ScaledCoordSpace::new(
            "AdamWState::minimize_scaled",
            init_physical,
            bounds,
            coord_scales,
        )?;
        let dim = space.dim();
        let mut x = vec![0.0_f64; dim];
        let mut jac = vec![0.0_f64; dim];
        let mut gx = vec![0.0_f64; dim];
        let mut gu = vec![0.0_f64; dim];

        space.decode(&u, &mut x, &mut jac);
        let init_loss = eval(&x, Some(&mut gx))?;
        if !init_loss.is_finite() || !gx.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput {
                context: "AdamWState::minimize_scaled(init_eval)",
            });
        }
        let mut best_params = x.clone();
        let mut best_loss = init_loss;
        let mut evals = 1_usize;
        let mut iters_run = 0_usize;
        let mut state = Self::new(dim);

        if total_steps > 0 {
            space.pullback_grad(&gx, &jac, &mut gu);
            for step in 0..total_steps {
                let lr = cosine_lr_with_warmup(step, total_steps, cfg);
                state.step_u(&mut u, &mut gu, lr, cfg, Some(space.scales()));
                space.decode(&u, &mut x, &mut jac);
                iters_run += 1;

                let need_grad = step + 1 < total_steps;
                let g_opt = if need_grad {
                    Some(gx.as_mut_slice())
                } else {
                    None
                };
                let cand_loss = eval(&x, g_opt)?;
                evals += 1;
                if cand_loss.is_finite() && cand_loss < best_loss {
                    best_loss = cand_loss;
                    best_params.copy_from_slice(&x);
                }
                if need_grad {
                    space.pullback_grad(&gx, &jac, &mut gu);
                }
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

/// Applies Arena-Scoped Deterministic `PCGrad` across active tiers `[Lite, Full, Pro, Ultra]`,
/// projecting strictly within shared parameter subsets in exact deterministic order:
/// - Arena `0`: tiers `0 -> 1 -> 2 -> 3`
/// - Arena `1`: tiers `1 -> 2 -> 3`
/// - Arena `2`: tiers `2 -> 3`
/// - Arena `>= 3` (Arena 3 + readouts): unprojected weighted sum
///
/// Inactive tiers may pass empty slices (`vec![]`) or zero tier weights.
#[must_use]
pub fn apply_arena_scoped_pcgrad(
    tier_grads: &[Vec<f64>; 4],
    tier_weights: [f64; 4],
    arena_of_param: &[u8],
) -> Vec<f64> {
    let dim = arena_of_param.len();
    let mut out = vec![0.0_f64; dim];
    if dim == 0 {
        return out;
    }

    // Readouts & Arena 3 (`arena >= 3`) remain unprojected.
    for (ti, grad_t) in tier_grads.iter().enumerate() {
        let wt = tier_weights[ti];
        if grad_t.len() != dim || !wt.is_finite() || wt <= 0.0 {
            continue;
        }
        for (p, (&g, &arena)) in grad_t.iter().zip(arena_of_param).enumerate() {
            if ((arena == 3 && ti == 3) || arena >= 4) && g.is_finite() {
                out[p] += wt * g;
            }
        }
    }

    // Shared Matryoshka arenas `[0, 1, 2]` apply deterministic scoped PCGrad.
    for arena_id in 0_u8..=2_u8 {
        let min_tier = arena_id as usize;
        let active_tiers: Vec<usize> = (min_tier..4)
            .filter(|&t| {
                tier_grads[t].len() == dim && tier_weights[t].is_finite() && tier_weights[t] > 0.0
            })
            .collect();
        if active_tiers.is_empty() {
            continue;
        }
        let idxs: Vec<usize> = arena_of_param
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| (a == arena_id).then_some(i))
            .collect();
        if idxs.is_empty() {
            continue;
        }

        let m = idxs.len();
        let orig_sub: Vec<Vec<f64>> = active_tiers
            .iter()
            .map(|&t| {
                idxs.iter()
                    .map(|&p| {
                        let g = tier_grads[t][p];
                        if g.is_finite() { g } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let mut proj_sub = orig_sub.clone();

        for i in 0..active_tiers.len() {
            for j in 0..active_tiers.len() {
                if i == j {
                    continue;
                }
                let mut dot = 0.0_f64;
                let mut norm_sq = 0.0_f64;
                for k in 0..m {
                    let g_j = orig_sub[j][k];
                    dot += proj_sub[i][k] * g_j;
                    norm_sq += g_j * g_j;
                }
                if norm_sq > PCGRAD_EPS && dot < 0.0 {
                    let coeff = dot / (norm_sq + PCGRAD_EPS);
                    for k in 0..m {
                        proj_sub[i][k] -= coeff * orig_sub[j][k];
                    }
                }
            }
        }

        for (slot, &ti) in active_tiers.iter().enumerate() {
            let wt = tier_weights[ti];
            for (k, &p) in idxs.iter().enumerate() {
                out[p] += wt * proj_sub[slot][k];
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_cosine_warmup_clip_scaled_convergence_and_moment_mask() {
        let cfg = AdamWConfig::default();
        assert!((cosine_lr_with_warmup(0, 100, &cfg) - cfg.lr_max * 0.1).abs() < 1.0e-12);
        assert!((cosine_lr_with_warmup(9, 100, &cfg) - cfg.lr_max).abs() < 1.0e-12);
        assert!((cosine_lr_with_warmup(10, 100, &cfg) - cfg.lr_max).abs() < 1.0e-12);
        assert!((cosine_lr_with_warmup(100, 100, &cfg) - cfg.lr_min).abs() < 1.0e-12);
        assert!((cosine_lr_with_warmup(0, 1, &cfg) - cfg.lr_max).abs() < 1.0e-12);

        let mut st = AdamWState::new(3);
        let mut u = [0.0_f64, 0.0, 0.0];
        let mut g = [30.0_f64, 40.0, 0.0];
        st.step_u(&mut u, &mut g, 0.1, &cfg, None);
        assert_eq!(st.steps(), 1);
        assert!((g[0] - 6.0).abs() < 1.0e-12 && (g[1] - 8.0).abs() < 1.0e-12);
        st.reset_moments_mask(&[true, false, false]);
        assert_eq!(st.m[0], 0.0);
        assert!(st.m[1].abs() > 0.0);

        let bounds = [
            ParamBoundSpec::Box { lo: -2.0, hi: 4.0 },
            ParamBoundSpec::LogBox {
                lo: 1.0e-3,
                hi: 2.0,
            },
        ];
        let scales = [1.0, 4.0];
        let fast_cfg = AdamWConfig {
            lr_max: 0.25,
            lr_min: 1.0e-4,
            warmup_steps: 5,
            ..cfg
        };
        let res = AdamWState::minimize_scaled(
            &[-1.0, 0.01],
            &bounds,
            Some(&scales),
            250,
            &fast_cfg,
            |x, grad_opt| {
                let dx0 = x[0] - 1.25;
                let dx1 = x[1] - 0.35;
                let loss = dx0 * dx0 + 4.0 * dx1 * dx1;
                if let Some(gr) = grad_opt {
                    gr[0] = 2.0 * dx0;
                    gr[1] = 8.0 * dx1;
                }
                Ok(loss)
            },
        )
        .unwrap();
        assert!(res.best_loss < 1.0e-4, "best_loss={}", res.best_loss);
        assert!((res.best_params[0] - 1.25).abs() < 1.0e-2);
        assert!((res.best_params[1] - 0.35).abs() < 1.0e-2);
    }

    #[test]
    fn test_apply_arena_scoped_pcgrad_respects_arena_boundaries_and_order() {
        // Indices: 0,1 -> Arena 0; 2,3 -> Arena 1; 4 -> Arena 2; 5 -> Arena 3; 6 -> Readout (4).
        let arenas = [0_u8, 0, 1, 1, 2, 3, 4];
        let w = [0.25_f64; 4];

        // Pure Arena-0 conflict between Lite (g = [1, 0]) and Full (g = [-1, 1]):
        // Pro/Ultra = [0, 1] only on Arena 0; Arena 1..3 have non-zero entries that must stay decoupled.
        let mut grads = [
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![-1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 10.0],
            vec![0.0, 1.0, -2.0, 3.0, 4.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, -4.0, 5.0, 6.0],
        ];
        let out = apply_arena_scoped_pcgrad(&grads, w, &arenas);

        // Arena 0: Lite vs Full conflict removes negative projection while Arena 1..3 never pollute it.
        assert!(out[0].is_finite() && out[0] > -0.2);
        assert!(out[1] > 0.0);

        // Arena 1: g1=[2,0], g2=[-2,3], g3=[0,1]; conflict only inside {Full, Pro, Ultra} on indices 2..3.
        assert!((out[2] - 0.25 * (-2.0 + 2.0)).abs() > 1.0e-6);

        // Arena 2 (idx 4): Pro=+4, Ultra=-4 exactly cancel via symmetric projection -> 0.0.
        assert!(
            out[4].abs() < 1.0e-12,
            "expected Arena 2 cancel, got {}",
            out[4]
        );

        // Arena 3 (idx 5) & Readout (idx 6) must be exact weighted sums with zero projection.
        assert!((out[5] - 0.25 * 5.0).abs() < 1.0e-12);
        assert!((out[6] - 0.25 * (10.0 + 6.0)).abs() < 1.0e-12);

        // Non-conflicting gradients across all arenas match each arena's active-tier weight sum.
        for tier in &mut grads {
            *tier = vec![1.0; 7];
        }
        let aligned = apply_arena_scoped_pcgrad(&grads, w, &arenas);
        let expected = [1.0, 1.0, 0.75, 0.75, 0.50, 0.25, 1.0];
        for (&v, &exp) in aligned.iter().zip(&expected) {
            assert!((v - exp).abs() < 1.0e-12, "got {v}, expected {exp}");
        }
    }
}
