//! Smooth unconstrained coordinate bijections ([`ParamBoundSpec`]) with exact chain-rule
//! derivatives and saturation protection (`ref[train.pipeline.backbone-trajectory-opt]`).

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

#[inline]
fn sanitize_u(u: f64) -> (f64, bool) {
    if !u.is_finite() {
        let clamped = if u.is_nan() {
            0.0
        } else if u.is_sign_positive() {
            24.0
        } else {
            -24.0
        };
        return (clamped, true);
    }
    let saturated = u.abs() > 24.0;
    (u.clamp(-24.0, 24.0), saturated)
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

        // ComplementLog must be strictly increasing across u in [-12, 12].
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

        // Verify extreme |u| = 15 for AsinhBoxSym has zero cancellation error in derivative.
        // Use eps = 1e-4 so f64 subtraction near |v| = 23,520 doesn't amplify ULP roundoff.
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

        // Saturation, NaN safety, and snap_zero_edge contract.
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
    }
}
