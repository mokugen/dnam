//! Smooth unconstrained coordinate bijections ([`ParamBoundSpec`]), symmetric coordinate scaling
//! ([`ScaledCoordSpace`]), and shared SIMD vector reductions (`ref[train.pipeline.backbone-trajectory-opt]`).

use crate::error::TrainError;
use crate::math::{DynamicTrainer, TrainerMathBackend, TrainerTask};

/// Shared optimization result returned by bounded `f64` optimizers (`L-BFGS`, `AdamW`, `L-SSBroyden`).
#[derive(Clone, Debug, PartialEq)]
pub struct OptResult {
    /// Best physical parameter vector observed across initial evaluation and accepted steps.
    pub best_params: Vec<f64>,
    /// Minimum objective value achieved at `best_params`.
    pub best_loss: f64,
    /// Number of accepted outer iterations completed.
    pub iters_run: usize,
    /// Number of objective/gradient evaluations performed.
    pub evals: usize,
}

/// Computes `a . b` via the detected `f64` SIMD [`TrainerMathBackend`].
#[inline]
#[must_use]
pub(crate) fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
    struct DotTask<'a>(&'a [f64], &'a [f64]);
    impl TrainerTask<f64> for DotTask<'_> {
        fn run<B: TrainerMathBackend>(self) -> f64 {
            B::dot_f64(self.0, self.1)
        }
    }
    DynamicTrainer::detect().dispatch(DotTask(a, b))
}

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
    /// Strictly increasing complement-log bijection `r(u) = 1.0 - c_max * (c_min / c_max)^sigmoid(u)`
    /// mapping `u in R` monotonically onto `[1.0 - hi_c, 1.0 - lo_c]`.
    ComplementLog {
        /// Positive lower bound on `1.0 - x`.
        lo_c: f64,
        /// Positive upper bound on `1.0 - x`.
        hi_c: f64,
    },
    /// Symmetric inverse-hyperbolic bijection `[-limit, +limit]` with linear slope near `0.0`
    /// (`v(u) = scale * sinh(asinh(limit / scale) * (2 * sigmoid(u) - 1))`).
    AsinhBoxSym {
        /// Linear transition scale (`> 0.0`).
        scale: f64,
        /// Symmetric magnitude ceiling (`> 0.0`).
        limit: f64,
    },
    /// Non-negative inverse-hyperbolic bijection `[0.0, limit]`
    /// (`v(u) = scale * sinh(asinh(limit / scale) * sigmoid(u))`).
    AsinhBoxNonNeg {
        /// Linear transition scale (`> 0.0`).
        scale: f64,
        /// Non-negative upper ceiling (`> 0.0`).
        limit: f64,
    },
}

/// Canonical saturation boundary in unscaled unconstrained `u`-space.
pub(crate) const U_SATURATION_LIMIT: f64 = 24.0;
const _: () = assert!(U_SATURATION_LIMIT > 10.0 && U_SATURATION_LIMIT <= 32.0);

#[inline]
fn sanitize_u(u: f64) -> (f64, bool) {
    if !u.is_finite() {
        let clamped = if u.is_nan() {
            0.0
        } else if u.is_sign_positive() {
            U_SATURATION_LIMIT
        } else {
            -U_SATURATION_LIMIT
        };
        return (clamped, true);
    }
    let saturated = u.abs() > U_SATURATION_LIMIT;
    (u.clamp(-U_SATURATION_LIMIT, U_SATURATION_LIMIT), saturated)
}

impl ParamBoundSpec {
    /// Returns a safe finite nominal interior point in physical coordinates.
    #[inline]
    #[must_use]
    pub fn default_midpoint(self) -> f64 {
        match self {
            Self::Free | Self::AsinhBoxSym { .. } => 0.0,
            Self::Box { lo, hi } | Self::LogBox { lo, hi } => 0.5 * (lo + hi),
            Self::ComplementLog { lo_c, hi_c } => 1.0 - 0.5 * (lo_c + hi_c),
            Self::AsinhBoxNonNeg { limit, .. } => 0.5 * limit.max(0.0),
        }
    }

    /// Encodes a physical parameter `v` into an unconstrained coordinate `u`.
    #[inline]
    #[must_use]
    pub fn to_unconstrained(self, v: f64) -> f64 {
        let vf = if v.is_finite() {
            v
        } else {
            self.default_midpoint()
        };
        match self {
            Self::Free => vf,
            Self::Box { lo, hi } => {
                let span = (hi - lo).max(1.0e-12);
                let vc = vf.clamp(lo + 1.0e-7 * span, hi - 1.0e-7 * span);
                let p = ((vc - lo) / span).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::LogBox { lo, hi } => {
                let (la, lb) = (lo.max(1.0e-15).ln(), hi.max(lo * 1.000_001).ln());
                let vc = vf.clamp(lo * (1.0 + 1.0e-7), hi * (1.0 - 1.0e-7));
                let p = ((vc.ln() - la) / (lb - la).max(1.0e-12)).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::ComplementLog { lo_c, hi_c } => {
                let c_min = lo_c.max(1.0e-15);
                let c_max = hi_c.max(c_min * 1.000_001);
                let (l_min, l_max) = (c_min.ln(), c_max.ln());
                let c_val = (1.0 - vf).clamp(c_min * (1.0 + 1.0e-7), c_max * (1.0 - 1.0e-7));
                let p = ((l_max - c_val.ln()) / (l_max - l_min).max(1.0e-12))
                    .clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::AsinhBoxSym { scale, limit } => {
                let s = scale.max(1.0e-12);
                let v_lim = limit.max(1.0e-12);
                let c = (v_lim / s).asinh().max(1.0e-12);
                let vc = vf.clamp(-v_lim, v_lim);
                let t = ((vc / s).asinh() / c).clamp(-(1.0 - 2.0e-7), 1.0 - 2.0e-7);
                let p = (0.5 * (t + 1.0)).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::AsinhBoxNonNeg { scale, limit } => {
                let s = scale.max(1.0e-12);
                let v_lim = limit.max(1.0e-12);
                let c = (v_lim / s).asinh().max(1.0e-12);
                let vc = vf.clamp(0.0, v_lim);
                let p = ((vc / s).asinh() / c).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
        }
    }

    /// Decodes unconstrained `u` to `(physical, d(physical)/du)`.
    #[inline]
    #[must_use]
    pub fn to_physical_and_deriv(self, u: f64) -> (f64, f64) {
        let (uc, saturated) = sanitize_u(u);
        match self {
            Self::Free => (if u.is_finite() { u } else { 0.0 }, 1.0),
            Self::Box { lo, hi } => {
                let s = 1.0 / (1.0 + (-uc).exp());
                let span = hi - lo;
                let deriv = if saturated { 0.0 } else { span * s * (1.0 - s) };
                (lo + span * s, deriv)
            }
            Self::LogBox { lo, hi } => {
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
                let s = 1.0 / (1.0 + (-uc).exp());
                let c_min = lo_c.max(1.0e-15);
                let c_max = hi_c.max(c_min * 1.000_001);
                let (l_min, l_max) = (c_min.ln(), c_max.ln());
                let c = (l_max + (l_min - l_max) * s).exp();
                let deriv = if saturated {
                    0.0
                } else {
                    c * (l_max - l_min) * s * (1.0 - s)
                };
                (1.0 - c, deriv)
            }
            Self::AsinhBoxSym { scale, limit } => {
                let s = scale.max(1.0e-12);
                let v_lim = limit.max(1.0e-12);
                let c = (v_lim / s).asinh();
                let t = (0.5 * uc).tanh();
                let v = s * (c * t).sinh();
                let deriv = if saturated {
                    0.0
                } else {
                    let sig = 1.0 / (1.0 + (-uc).exp());
                    let sig_comp = 1.0 / (1.0 + uc.exp());
                    2.0 * s * c * (c * t).cosh() * sig * sig_comp
                };
                (v, deriv)
            }
            Self::AsinhBoxNonNeg { scale, limit } => {
                let sig = 1.0 / (1.0 + (-uc).exp());
                let sig_comp = 1.0 / (1.0 + uc.exp());
                let s = scale.max(1.0e-12);
                let v_lim = limit.max(1.0e-12);
                let c = (v_lim / s).asinh();
                let v = s * (c * sig).sinh();
                let deriv = if saturated {
                    0.0
                } else {
                    s * c * (c * sig).cosh() * sig * sig_comp
                };
                (v, deriv)
            }
        }
    }

    /// Hard-projects a scalar cleanly inside the spec domain.
    #[inline]
    #[must_use]
    pub fn project_physical(self, v: f64) -> f64 {
        if !v.is_finite() {
            return self.default_midpoint();
        }
        match self {
            Self::Free => v,
            Self::Box { lo, hi } | Self::LogBox { lo, hi } => v.clamp(lo, hi),
            Self::ComplementLog { lo_c, hi_c } => v.clamp(1.0 - hi_c, 1.0 - lo_c),
            Self::AsinhBoxSym { limit, .. } => {
                let v_lim = limit.max(1.0e-12);
                v.clamp(-v_lim, v_lim)
            }
            Self::AsinhBoxNonNeg { limit, .. } => v.clamp(0.0, limit.max(1.0e-12)),
        }
    }

    /// Snaps physical `ComplementLog` values below `1e-7` to exact `0.0` at model pack time
    /// **only** when the lower bound `1.0 - hi_c <= 0.0` (`hi_c >= 1.0`), leaving strictly
    /// positive-floor domains (`kautz.r_low in [0.25, 0.9995]`, `r_dc in [0.95, 0.9999]`) intact.
    #[inline]
    #[must_use]
    pub fn snap_zero_edge(self, v: f64) -> f64 {
        match self {
            Self::ComplementLog { hi_c, .. } if hi_c >= 1.0 - 1.0e-15 && v < 1.0e-7 => 0.0,
            _ => v,
        }
    }
}

/// Symmetrically scaled unconstrained coordinate codec `u_i = s_i * u_raw,i` shared by `L-BFGS`,
/// `AdamW`, and `L-SSBroyden` (`d(loss)/du_i = d(loss)/dx_i * (dx_i/du_raw,i / s_i)`).
#[derive(Clone, Debug)]
pub struct ScaledCoordSpace<'a> {
    bounds: &'a [ParamBoundSpec],
    eff_scales: Vec<f64>,
}

impl<'a> ScaledCoordSpace<'a> {
    /// Validates coordinate dimensions and finite initialization, returning `(space, init_u)`.
    pub fn new(
        context: &'static str,
        init_physical: &[f64],
        bounds: &'a [ParamBoundSpec],
        coord_scales: Option<&[f64]>,
    ) -> Result<(Self, Vec<f64>), TrainError> {
        let dim = init_physical.len();
        if dim != bounds.len() {
            return Err(TrainError::InvalidDimensions {
                context,
                expected: dim,
                actual: bounds.len(),
            });
        }
        if let Some(scales) = coord_scales
            && scales.len() != dim
        {
            return Err(TrainError::InvalidDimensions {
                context,
                expected: dim,
                actual: scales.len(),
            });
        }
        if dim == 0 {
            return Err(TrainError::InvalidDimensions {
                context,
                expected: 1,
                actual: 0,
            });
        }
        if !init_physical.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput { context });
        }
        let eff_scales: Vec<f64> = match coord_scales {
            Some(scales) => scales
                .iter()
                .map(|&s| if s.is_finite() && s > 0.0 { s } else { 1.0 })
                .collect(),
            None => vec![1.0_f64; dim],
        };
        let init_u = init_physical
            .iter()
            .zip(bounds)
            .zip(&eff_scales)
            .map(|((&p, &b), &s)| b.to_unconstrained(p) * s)
            .collect();
        Ok((Self { bounds, eff_scales }, init_u))
    }

    /// Returns parameter vector dimensionality.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.bounds.len()
    }

    /// Returns the effective coordinate scaling factors `s_i > 0.0`.
    #[inline]
    #[must_use]
    pub fn scales(&self) -> &[f64] {
        &self.eff_scales
    }

    /// Encodes a physical vector `physical` into scaled unconstrained `u_out`.
    #[inline]
    pub fn encode(&self, physical: &[f64], u_out: &mut [f64]) {
        for (((dst, &p), &b), &s) in u_out
            .iter_mut()
            .zip(physical)
            .zip(self.bounds)
            .zip(&self.eff_scales)
        {
            *dst = b.to_unconstrained(p) * s;
        }
    }

    /// Decodes scaled unconstrained `u` to physical parameters `x_out` and scaled Jacobians `jac_out`.
    #[inline]
    pub fn decode(&self, u: &[f64], x_out: &mut [f64], jac_out: &mut [f64]) {
        for (i, ((&ui, &b), &s)) in u.iter().zip(self.bounds).zip(&self.eff_scales).enumerate() {
            let (xi, ji) = b.to_physical_and_deriv(ui / s);
            x_out[i] = xi;
            jac_out[i] = ji / s;
        }
    }

    /// Pulls physical gradient `grad_x` back into scaled unconstrained gradient `grad_u_out`.
    #[inline]
    pub fn pullback_grad(&self, grad_x: &[f64], jac: &[f64], grad_u_out: &mut [f64]) {
        for ((dst, &gx), &j) in grad_u_out.iter_mut().zip(grad_x).zip(jac) {
            *dst = gx * j;
        }
    }

    /// Populates diagonal Hessian preconditioning `h0_out = 1 / clamp(j^2, 1, 1e12)`.
    #[inline]
    pub fn diag_h0_from_jac(jac: &[f64], h0_out: &mut [f64]) {
        for (dst, &j) in h0_out.iter_mut().zip(jac) {
            *dst = 1.0 / (j * j).clamp(1.0, 1.0e12);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_param_bound_specs_roundtrip_monotonic_and_fd_jacobian() {
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
            (
                ParamBoundSpec::ComplementLog {
                    lo_c: 5.0e-6,
                    hi_c: 1.0,
                },
                0.9992,
            ),
            (
                ParamBoundSpec::AsinhBoxSym {
                    scale: 1.0,
                    limit: 32.0,
                },
                -14.25,
            ),
            (
                ParamBoundSpec::AsinhBoxSym {
                    scale: 10.0,
                    limit: 23_520.0,
                },
                0.0,
            ),
            (
                ParamBoundSpec::AsinhBoxNonNeg {
                    scale: 32.0,
                    limit: 48_000.0,
                },
                1_250.0,
            ),
        ];
        for (spec, v) in specs {
            let u = spec.to_unconstrained(v);
            let (v2, dv_du) = spec.to_physical_and_deriv(u);
            assert!((v2 - v).abs() <= 1.0e-9, "spec={spec:?} v={v} v2={v2}");
            assert!(
                dv_du > 0.0,
                "expected strictly positive dv/du for {spec:?}, got {dv_du}"
            );
            let eps = 1.0e-6;
            let fd = (spec.to_physical_and_deriv(u + eps).0
                - spec.to_physical_and_deriv(u - eps).0)
                / (2.0 * eps);
            assert!(
                (dv_du - fd).abs() <= 1.0e-8 * dv_du.abs().max(1.0),
                "spec={spec:?} dv_du={dv_du} fd={fd}"
            );
            assert!((spec.project_physical(v2) - v2).abs() <= 1.0e-12);
        }

        let cl = ParamBoundSpec::ComplementLog {
            lo_c: 5.0e-4,
            hi_c: 0.75,
        };
        let mut prev_r = -1.0_f64;
        for step in -12..=12 {
            let u = step as f64;
            let (r, dr) = cl.to_physical_and_deriv(u);
            assert!(
                r > prev_r,
                "ComplementLog not monotone at u={u}: {r} <= {prev_r}"
            );
            assert!(dr > 0.0, "ComplementLog deriv not positive at u={u}: {dr}");
            prev_r = r;
        }

        let sym = ParamBoundSpec::AsinhBoxSym {
            scale: 10.0,
            limit: 23_520.0,
        };
        for u in [-15.0_f64, 15.0] {
            let (_, dv_du) = sym.to_physical_and_deriv(u);
            let eps = 1.0e-4;
            let fd = (sym.to_physical_and_deriv(u + eps).0 - sym.to_physical_and_deriv(u - eps).0)
                / (2.0 * eps);
            assert!(
                (dv_du - fd).abs() <= 1.0e-6 * fd.abs().max(1.0),
                "AsinhBoxSym extreme u={u}: dv_du={dv_du} fd={fd}"
            );
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
        assert_eq!(sym.to_physical_and_deriv(-30.0).1, 0.0);
        assert_eq!(sym.to_physical_and_deriv(f64::NAN), (0.0, 0.0));
        assert_eq!(
            ParamBoundSpec::AsinhBoxNonNeg {
                scale: 1.0,
                limit: 16.0
            }
            .to_physical_and_deriv(30.0)
            .1,
            0.0
        );

        let zero_edge = ParamBoundSpec::ComplementLog {
            lo_c: 5.0e-6,
            hi_c: 1.0,
        };
        let nonzero_floor = ParamBoundSpec::ComplementLog {
            lo_c: 5.0e-4,
            hi_c: 0.75,
        };
        assert_eq!(zero_edge.snap_zero_edge(5.0e-8), 0.0);
        assert_eq!(zero_edge.snap_zero_edge(1.0e-3), 1.0e-3);
        assert_eq!(nonzero_floor.snap_zero_edge(5.0e-8), 5.0e-8);
        assert_eq!(nonzero_floor.snap_zero_edge(0.25), 0.25);

        let scales = [1.0, 50.0];
        let b = [
            ParamBoundSpec::Free,
            ParamBoundSpec::Box { lo: 0.0, hi: 1.0 },
        ];
        let (sp, mut u_vec) =
            ScaledCoordSpace::new("test", &[0.2, 0.75], &b, Some(&scales)).unwrap();
        assert_eq!(sp.dim(), 2);
        let (mut x_out, mut jac_out, mut gu) = ([0.0; 2], [0.0; 2], [0.0; 2]);
        sp.decode(&u_vec, &mut x_out, &mut jac_out);
        assert!((x_out[0] - 0.2).abs() < 1.0e-12);
        assert!((x_out[1] - 0.75).abs() < 1.0e-7);
        sp.pullback_grad(&[2.0, 3.0], &jac_out, &mut gu);
        assert!((gu[0] - 2.0).abs() < 1.0e-12);
        assert!((gu[1] - 3.0 * jac_out[1]).abs() < 1.0e-12);
        sp.encode(&[0.5, 0.25], &mut u_vec);
        sp.decode(&u_vec, &mut x_out, &mut jac_out);
        assert!((x_out[1] - 0.25).abs() < 1.0e-7);
        assert!(ScaledCoordSpace::new("bad", &[0.0], &b, None).is_err());
        assert!(ScaledCoordSpace::new("nan", &[f64::NAN, 0.5], &b, None).is_err());
        assert!(ScaledCoordSpace::new("empty", &[], &[], None).is_err());
    }
}
