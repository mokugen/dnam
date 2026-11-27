//! Cancellation-free pure-`f32` backward sensitivities for clamped `AlgADAA1(0.5)` using
//! exact algebraic forms, 4-point linear Gauss-Legendre, and 5-point Rational-Möbius
//! Gauss-Legendre quadrature (`0 FDIV`, `0 FSQRT`, `0 f64`, `0` dynamic indexing,
//! `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[repo.code.constant-eval-and-derivations]`).

use crate::simd::{GpuVec4, KernelVec4, const_abs_f32};

use super::scrub_clamped_alg_endpoint_f32;

/// 4-point Gauss-Legendre quadrature nodes on `[0, 1]`, mapped via `x_i = 0.5 * (1 + xi_i)`
/// from roots `xi = +-sqrt((15 -+ 2*sqrt(30)) / 35)` of Legendre polynomial `P_4(xi)`
/// (`ref[repo.code.constant-eval-and-derivations]`).
const GL4_X0: f32 = 0.069_431_84;
const GL4_X1: f32 = 0.330_009_5;
const GL4_X2: f32 = 0.669_990_54;
const GL4_X3: f32 = 0.930_568_16;

/// 4-point Gauss-Legendre quadrature weights on `[0, 1]`: `(18 -+ sqrt(30)) / 72`
/// (`ref[repo.code.constant-eval-and-derivations]`).
const GL4_W0: f32 = 0.173_927_43;
const GL4_W1: f32 = 0.326_072_57;
const GL4_W2: f32 = 0.326_072_57;
const GL4_W3: f32 = 0.173_927_43;

/// 5-point Gauss-Legendre quadrature nodes on `[0, 1]`, mapped via `u_i = 0.5 * (1 + xi_i)`
/// from roots `xi = 0, +-1/3 * sqrt(5 -+ 2*sqrt(10/7))` of Legendre polynomial `P_5(xi)`
/// (`ref[repo.code.constant-eval-and-derivations]`).
const GL5_U0: f32 = 0.046_910_077;
const GL5_U1: f32 = 0.230_765_34;
const GL5_U2: f32 = 0.5;
const GL5_U3: f32 = 0.769_234_66;
const GL5_U4: f32 = 0.953_089_9;

/// 5-point Gauss-Legendre quadrature weights on `[0, 1]`: `(322 -+ 13*sqrt(70)) / 1800` and `64 / 225`
/// (`ref[repo.code.constant-eval-and-derivations]`).
const GL5_W0: f32 = 0.118_463_44;
const GL5_W1: f32 = 0.239_314_33;
const GL5_W2: f32 = 0.284_444_45;
const GL5_W3: f32 = 0.239_314_33;
const GL5_W4: f32 = 0.118_463_44;

/// Relative-span threshold `|r_n - r_p| <= 0.15 * max(|r_p|, |r_n|)` switching between
/// 4-point linear Gauss-Legendre and 5-point Rational-Möbius Gauss-Legendre quadrature.
const GL_LINEAR_SPAN_RATIO: f32 = 0.15;

/// Lower magnitude threshold `min(|r_p|, |r_n|) > 1.75` on same-sign intervals where direct
/// algebraic differencing suffers catastrophic cancellation and quadrature is strictly accurate.
const GL_MIN_ABS_THRESHOLD: f32 = 1.75;

const _: () = assert!(const_abs_f32((GL4_W0 + GL4_W1 + GL4_W2 + GL4_W3) - 1.0) < 1.0e-6);
const _: () = assert!(GL4_W0 == GL4_W3 && GL4_W1 == GL4_W2);
const _: () = assert!(const_abs_f32((GL4_X0 + GL4_X3) - 1.0) < 1.0e-6);
const _: () = assert!(const_abs_f32((GL4_X1 + GL4_X2) - 1.0) < 1.0e-6);
const _: () = assert!(const_abs_f32((GL5_W0 + GL5_W1 + GL5_W2 + GL5_W3 + GL5_W4) - 1.0) < 1.0e-6);
const _: () = assert!(GL5_W0 == GL5_W4 && GL5_W1 == GL5_W3);
const _: () = assert!(const_abs_f32((GL5_U0 + GL5_U4) - 1.0) < 1.0e-6);
const _: () = assert!(const_abs_f32((GL5_U1 + GL5_U3) - 1.0) < 1.0e-6);
const _: () = assert!(GL5_U2 == 0.5);

/// Backward partial derivatives of clamped `AlgADAA1(0.5)` with respect to clamped endpoints
/// `(r_prev, r_curr)` (`ref[dsp.op-n.polyadaa.rail-crossing]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AlgAdaa1BackwardOut {
    pub d_odd_dr_prev: f32,
    pub d_odd_dr_curr: f32,
    pub d_even_dr_prev: f32,
    pub d_even_dr_curr: f32,
}

#[inline(always)]
fn rsqrt_nr3_f32(x: f32) -> f32 {
    GpuVec4::splat(x).rsqrt_pos_nr3().to_array()[0]
}

#[inline(always)]
fn rcp_ge1_nr3_f32(x: f32) -> f32 {
    GpuVec4::splat(x).rcp_ge1_nr3().to_array()[0]
}

#[inline(always)]
fn eval_gl4_linear_term(rp: f32, diff: f32, x_i: f32, w_i: f32) -> f32 {
    let r = rp + x_i * diff;
    let inv_s = rsqrt_nr3_f32(1.0 + r * r);
    let inv_s2 = inv_s * inv_s;
    let inv_s5 = inv_s2 * inv_s2 * inv_s;
    w_i * (1.5 * x_i * inv_s5)
}

#[inline(always)]
fn eval_gl5_mobius_term(
    lo_p1: f32,
    hi_p1: f32,
    span: f32,
    scale_num: f32,
    an_ge_ap: bool,
    u_i: f32,
    w_i: f32,
) -> f32 {
    let d_u = hi_p1 - u_i * span;
    let inv_d = rcp_ge1_nr3_f32(d_u);
    let x_u = lo_p1 * hi_p1 * inv_d - 1.0;
    let inv_s = rsqrt_nr3_f32(1.0 + x_u * x_u);
    let inv_s2 = inv_s * inv_s;
    let inv_s5 = inv_s2 * inv_s2 * inv_s;
    let inv_d3 = inv_d * inv_d * inv_d;
    let u_fac = if an_ge_ap { u_i } else { 1.0 - u_i };
    w_i * (u_fac * scale_num * inv_d3) * (1.5 * inv_s5)
}

#[inline(always)]
fn eval_alg_adaa1_backward_curr_f32(rp: f32, rn: f32) -> (f32, f32) {
    let rn2 = rn * rn;
    let rp2 = rp * rp;
    let q_n = 1.0 + rn2;
    let q_p = 1.0 + rp2;

    let inv_s_n = rsqrt_nr3_f32(q_n);
    let inv_s_p = rsqrt_nr3_f32(q_p);
    let s_n = q_n * inv_s_n;
    let s_p = q_p * inv_s_p;
    let inv_sum = rcp_ge1_nr3_f32(s_n + s_p);

    let prod = rn * rp;
    let p_pos = if prod <= 0.0 {
        1.0 + s_n * s_p - prod
    } else {
        1.0 + (1.0 + rn2 + rp2) * rcp_ge1_nr3_f32(s_n * s_p + prod)
    };

    let inv_s_n3 = inv_s_n * inv_s_n * inv_s_n;
    let d_even = ((rn + rp) + rn * s_n * p_pos * inv_sum) * inv_sum * inv_s_n3 * inv_s_p;

    let ap = if rp < 0.0 { -rp } else { rp };
    let an = if rn < 0.0 { -rn } else { rn };
    let lo = if ap < an { ap } else { an };
    let hi = if ap > an { ap } else { an };
    let diff = rn - rp;
    let diff_abs = if diff < 0.0 { -diff } else { diff };

    let d_odd = if prod >= 0.0 && lo > GL_MIN_ABS_THRESHOLD {
        if diff_abs <= GL_LINEAR_SPAN_RATIO * hi {
            eval_gl4_linear_term(rp, diff, GL4_X0, GL4_W0)
                + eval_gl4_linear_term(rp, diff, GL4_X1, GL4_W1)
                + eval_gl4_linear_term(rp, diff, GL4_X2, GL4_W2)
                + eval_gl4_linear_term(rp, diff, GL4_X3, GL4_W3)
        } else {
            let lo_p1 = lo + 1.0;
            let hi_p1 = hi + 1.0;
            let span = hi - lo;
            let an_ge_ap = an >= ap;
            let scale_num = if an_ge_ap {
                lo_p1 * lo_p1 * hi_p1
            } else {
                lo_p1 * hi_p1 * hi_p1
            };
            eval_gl5_mobius_term(lo_p1, hi_p1, span, scale_num, an_ge_ap, GL5_U0, GL5_W0)
                + eval_gl5_mobius_term(lo_p1, hi_p1, span, scale_num, an_ge_ap, GL5_U1, GL5_W1)
                + eval_gl5_mobius_term(lo_p1, hi_p1, span, scale_num, an_ge_ap, GL5_U2, GL5_W2)
                + eval_gl5_mobius_term(lo_p1, hi_p1, span, scale_num, an_ge_ap, GL5_U3, GL5_W3)
                + eval_gl5_mobius_term(lo_p1, hi_p1, span, scale_num, an_ge_ap, GL5_U4, GL5_W4)
        }
    } else if prod >= 0.0 {
        let dn = rcp_ge1_nr3_f32(s_n + an);
        let dp = rcp_ge1_nr3_f32(s_p + ap);
        let dn2 = dn * dn;
        let dp2 = dp * dp;
        let num = an * dp * (an * an * dp + 2.0 * an)
            + an * (1.5 * dp2 + 2.0)
            + ap * ap * dn * (0.5 * dn2 + 1.5)
            + ap * (dn2 + 2.0)
            + dn * (0.25 * dn2 + 2.75)
            + 3.0 * dp;
        0.5 * num * inv_s_n3 * inv_s_p * inv_sum * inv_sum
    } else {
        let u_term = (rn + rp) * inv_sum;
        let v_term = 1.0 + 0.5 * inv_s_n * inv_s_p;
        p_pos * inv_s_n * inv_sum * inv_sum * v_term - 0.5 * u_term * rn * inv_s_n3 * inv_s_p
    };

    (d_odd + 0.0, d_even + 0.0)
}

/// Evaluates cancellation-free partial derivatives `(∂A_odd/∂r_prev, ∂A_odd/∂r_curr,
/// ∂A_even/∂r_prev, ∂A_even/∂r_curr)` of clamped `AlgADAA1(0.5)` purely in `f32`/`GpuVec4`
/// (`0 FDIV`, `0 FSQRT`, `0 f64`, `0` dynamic indexing, `ref[dsp.op-n.polyadaa.rail-crossing]`).
#[inline]
#[must_use]
pub fn eval_alg_adaa1_backward_f32(r_prev: f32, r_curr: f32) -> AlgAdaa1BackwardOut {
    let rp = scrub_clamped_alg_endpoint_f32(r_prev);
    let rn = scrub_clamped_alg_endpoint_f32(r_curr);
    let (d_odd_dr_curr, d_even_dr_curr) = eval_alg_adaa1_backward_curr_f32(rp, rn);
    let (d_odd_dr_prev, d_even_dr_prev) = eval_alg_adaa1_backward_curr_f32(rn, rp);
    AlgAdaa1BackwardOut {
        d_odd_dr_prev,
        d_odd_dr_curr,
        d_even_dr_prev,
        d_even_dr_curr,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MAX_ALG_R, eval_alg_adaa1_f32};
    use super::*;
    use proptest::prelude::*;

    /// Exact cancellation-free `f64` oracle using double-precision quadrature on the integral
    /// representation `A_odd(rp, rn) = int_0^1 f_odd(r(t)) dt` for `prod >= 0 && lo > 1.75`.
    fn alg_adaa1_backward_oracle_f64(rp_raw: f64, rn_raw: f64) -> (f64, f64) {
        let rp = rp_raw.clamp(-f64::from(MAX_ALG_R), f64::from(MAX_ALG_R));
        let rn = rn_raw.clamp(-f64::from(MAX_ALG_R), f64::from(MAX_ALG_R));
        let sp = (1.0 + rp * rp).sqrt();
        let sn = (1.0 + rn * rn).sqrt();
        let prod = rn * rp;
        let p_pos = if prod <= 0.0 {
            1.0 + sn * sp - prod
        } else {
            1.0 + (1.0 + rn * rn + rp * rp) / (sn * sp + prod)
        };
        let d_even = ((rn + rp) + rn * sn * p_pos / (sn + sp)) / ((sn + sp) * sn * sn * sn * sp);

        let ap = rp.abs();
        let an = rn.abs();
        let lo = ap.min(an);
        let hi = ap.max(an);
        let d_odd = if prod >= 0.0 && lo > f64::from(GL_MIN_ABS_THRESHOLD) {
            // 16-point composite Gauss-Legendre on Möbius-warped interval for machine-precision f64 truth.
            const N_SUB: usize = 64;
            let lo_p1 = lo + 1.0;
            let hi_p1 = hi + 1.0;
            let span = hi - lo;
            let an_ge_ap = an >= ap;
            let scale_num = if an_ge_ap {
                lo_p1 * lo_p1 * hi_p1
            } else {
                lo_p1 * hi_p1 * hi_p1
            };
            let gl5_u = [
                0.5 * (1.0 - (5.0 + 2.0 * (10.0_f64 / 7.0).sqrt()).sqrt() / 3.0),
                0.5 * (1.0 - (5.0 - 2.0 * (10.0_f64 / 7.0).sqrt()).sqrt() / 3.0),
                0.5,
                0.5 * (1.0 + (5.0 - 2.0 * (10.0_f64 / 7.0).sqrt()).sqrt() / 3.0),
                0.5 * (1.0 + (5.0 + 2.0 * (10.0_f64 / 7.0).sqrt()).sqrt() / 3.0),
            ];
            let gl5_w = [
                (322.0 - 13.0 * 70.0_f64.sqrt()) / 1800.0,
                (322.0 + 13.0 * 70.0_f64.sqrt()) / 1800.0,
                64.0 / 225.0,
                (322.0 + 13.0 * 70.0_f64.sqrt()) / 1800.0,
                (322.0 - 13.0 * 70.0_f64.sqrt()) / 1800.0,
            ];
            let h = 1.0 / (N_SUB as f64);
            let mut sum = 0.0_f64;
            for seg in 0..N_SUB {
                let a = (seg as f64) * h;
                for q in 0..5 {
                    let u = a + h * gl5_u[q];
                    let d_u = hi_p1 - u * span;
                    let x_u = (lo_p1 * hi_p1) / d_u - 1.0;
                    let inv_s = (1.0 + x_u * x_u).sqrt().recip();
                    let inv_s5 = inv_s * inv_s * inv_s * inv_s * inv_s;
                    let u_fac = if an_ge_ap { u } else { 1.0 - u };
                    sum +=
                        (h * gl5_w[q]) * (u_fac * scale_num / (d_u * d_u * d_u)) * (1.5 * inv_s5);
                }
            }
            sum
        } else {
            let sum_s = sn + sp;
            let u_term = (rn + rp) / sum_s;
            let v_term = 1.0 + 0.5 / (sn * sp);
            p_pos * v_term / (sn * sum_s * sum_s) - 0.5 * u_term * rn / (sn * sn * sn * sp)
        };
        (d_odd, d_even)
    }

    fn alg_adaa1_forward_f64(rp_raw: f64, rn_raw: f64) -> (f64, f64) {
        let rp = rp_raw.clamp(-f64::from(MAX_ALG_R), f64::from(MAX_ALG_R));
        let rn = rn_raw.clamp(-f64::from(MAX_ALG_R), f64::from(MAX_ALG_R));
        if rp == 0.0 && rn == 0.0 {
            return (0.0, 0.0);
        }
        let sp = (1.0 + rp * rp).sqrt();
        let sn = (1.0 + rn * rn).sqrt();
        let tp = (rp * rp) / (sp + 1.0);
        let tn = (rn * rn) / (sn + 1.0);
        let ts = tn + tp;
        let odd = ((rn + rp) / (sn + sp)) * (1.0 + 0.5 / (sn * sp));
        let even = (2.0 * ts + ts * ts + tn * tp * (1.0 + ts) + rn * rp) / ((sn + sp) * sn * sp);
        (odd, even.clamp(0.0, 1.0))
    }

    /// verify[dsp.op-n.polyadaa.rail-crossing]
    #[test]
    fn test_alg_adaa1_backward_exhaustive_578_pairs_and_sign_consistency() {
        let grid: [f32; 25] = [
            -4096.0, -3000.0, -1000.0, -250.0, -50.0, -10.0, -3.5, -2.1, -2.0, -1.5, -1.0, -0.25,
            0.0, 0.25, 1.0, 1.5, 2.0, 2.1, 3.5, 10.0, 50.0, 250.0, 1000.0, 3000.0, 4096.0,
        ];
        let mut pair_count = 0_usize;
        let mut max_dodd_rel_err = 0.0_f64;
        let mut max_deven_rel_err = 0.0_f64;

        for &rp in &grid {
            for &rn in &grid {
                pair_count += 1;
                let g = eval_alg_adaa1_backward_f32(rp, rn);
                assert!(g.d_odd_dr_prev.is_finite() && g.d_odd_dr_curr.is_finite());
                assert!(g.d_even_dr_prev.is_finite() && g.d_even_dr_curr.is_finite());
                assert!(
                    g.d_odd_dr_curr > 0.0,
                    "d_odd must be strictly positive at ({rp}, {rn})"
                );
                assert!(
                    g.d_odd_dr_prev > 0.0,
                    "d_odd_prev must be strictly positive at ({rp}, {rn})"
                );

                let (eo, ee) = alg_adaa1_backward_oracle_f64(f64::from(rp), f64::from(rn));
                if eo.abs() > 1.0e-18 {
                    let rel_o = (f64::from(g.d_odd_dr_curr) - eo).abs() / eo.abs();
                    max_dodd_rel_err = max_dodd_rel_err.max(rel_o);
                }
                if ee.abs() > 1.0e-12 {
                    let rel_e = (f64::from(g.d_even_dr_curr) - ee).abs() / ee.abs();
                    max_deven_rel_err = max_deven_rel_err.max(rel_e);
                } else {
                    assert!((f64::from(g.d_even_dr_curr) - ee).abs() < 1.0e-7);
                }
                if ee != 0.0 {
                    assert_eq!(
                        f64::from(g.d_even_dr_curr).signum(),
                        ee.signum(),
                        "sign flip on deven at ({rp}, {rn}): got={}, oracle={ee}",
                        g.d_even_dr_curr
                    );
                }
            }
        }
        assert!(pair_count >= 578, "tested {pair_count} pairs");
        assert!(
            max_dodd_rel_err <= 8.68e-6,
            "max_dodd_rel_err={max_dodd_rel_err}"
        );
        assert!(
            max_deven_rel_err <= 5.0e-6,
            "max_deven_rel_err={max_deven_rel_err}"
        );

        // Extreme ±1e9 / NaN / ±Inf safety + zero equilibrium.
        let z = eval_alg_adaa1_backward_f32(-0.0, -0.0);
        assert!(
            (z.d_odd_dr_prev - 0.75).abs() < 2.0e-7,
            "d_odd_prev={}",
            z.d_odd_dr_prev
        );
        assert!(
            (z.d_odd_dr_curr - 0.75).abs() < 2.0e-7,
            "d_odd_curr={}",
            z.d_odd_dr_curr
        );
        assert_eq!(z.d_even_dr_prev.to_bits(), 0);
        assert_eq!(z.d_even_dr_curr.to_bits(), 0);

        let ext = eval_alg_adaa1_backward_f32(-1.0e9, 1.0e9);
        let clamped_ref = eval_alg_adaa1_backward_f32(-MAX_ALG_R, MAX_ALG_R);
        assert_eq!(ext, clamped_ref);
        let nan_out = eval_alg_adaa1_backward_f32(f32::NAN, f32::INFINITY);
        assert_eq!(nan_out, z);

        // Verify forward/backward consistency on moderate points via f64 central FD.
        for &(rp, rn) in &[
            (-1.25_f32, 0.75_f32),
            (2.5, 2.7),
            (2.5, 50.0),
            (-12.0, -3.0),
        ] {
            let b = eval_alg_adaa1_backward_f32(rp, rn);
            let h = 1.0e-5_f64;
            let (op, ep) = alg_adaa1_forward_f64(f64::from(rp), f64::from(rn) + h);
            let (om, em) = alg_adaa1_forward_f64(f64::from(rp), f64::from(rn) - h);
            let fd_o = (op - om) / (2.0 * h);
            let fd_e = (ep - em) / (2.0 * h);
            assert!((f64::from(b.d_odd_dr_curr) - fd_o).abs() < 1.0e-6);
            assert!((f64::from(b.d_even_dr_curr) - fd_e).abs() < 1.0e-6);

            let (fwd_o, fwd_e, _) = eval_alg_adaa1_f32(rp, rn);
            assert!(fwd_o.is_finite() && fwd_e.is_finite());
        }

        // Regression pairs + dense cancellation sweep across lo in [0.5, 2.5] x hi in [50.0, 4096.0].
        let mut sweep_pairs: Vec<(f32, f32)> = vec![
            (1.912_571_4, 4046.244),
            (-1.912_571_4, -4046.244),
            (-1.999, -438.888_5),
            (1.999, 438.888_5),
            (1.90, 2000.0),
            (-1.90, -2000.0),
            (1.89, 2000.0),
            (-1.89, -2000.0),
        ];
        let hi_pts: [f32; 10] = [
            50.0, 100.0, 200.0, 438.888_5, 500.0, 1000.0, 2000.0, 3000.0, 4046.244, 4096.0,
        ];
        for step in 0..=40 {
            let lo = 0.5 + (step as f32) * 0.05;
            for &hi in &hi_pts {
                sweep_pairs.push((lo, hi));
                sweep_pairs.push((hi, lo));
                sweep_pairs.push((-lo, -hi));
                sweep_pairs.push((-hi, -lo));
            }
        }
        for step in 0..=35 {
            let lo = 1.70 + (step as f32) * 0.01;
            for &hi in &hi_pts {
                sweep_pairs.push((lo, hi));
                sweep_pairs.push((-lo, -hi));
            }
        }
        for &(rp, rn) in &sweep_pairs {
            let g = eval_alg_adaa1_backward_f32(rp, rn);
            assert!(g.d_odd_dr_curr > 0.0 && g.d_odd_dr_prev > 0.0);
            let (eo, ee) = alg_adaa1_backward_oracle_f64(f64::from(rp), f64::from(rn));
            let rel_o = (f64::from(g.d_odd_dr_curr) - eo).abs() / eo.abs();
            assert!(
                rel_o <= 8.68e-6,
                "d_odd rel_o={rel_o} > 8.68e-6 at ({rp}, {rn}): got={}, oracle={eo}",
                g.d_odd_dr_curr
            );
            if ee != 0.0 {
                assert_eq!(
                    f64::from(g.d_even_dr_curr).signum(),
                    ee.signum(),
                    "sign flip on deven at ({rp}, {rn})"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_alg_adaa1_backward_matches_oracle_across_full_domain(
            rp in prop_oneof![
                -4096.0_f32..4096.0_f32,
                -20.0_f32..20.0_f32,
                -2.05_f32..-1.75_f32,
                1.75_f32..2.05_f32,
                2.01_f32..4096.0_f32,
            ],
            rn in prop_oneof![
                -4096.0_f32..4096.0_f32,
                -20.0_f32..20.0_f32,
                -2.05_f32..-1.75_f32,
                1.75_f32..2.05_f32,
                2.01_f32..4096.0_f32,
            ],
        ) {
            let g = eval_alg_adaa1_backward_f32(rp, rn);
            let g_swap = eval_alg_adaa1_backward_f32(rn, rp);
            prop_assert_eq!(g.d_odd_dr_prev.to_bits(), g_swap.d_odd_dr_curr.to_bits());
            prop_assert_eq!(g.d_even_dr_prev.to_bits(), g_swap.d_even_dr_curr.to_bits());

            let (eo, ee) = alg_adaa1_backward_oracle_f64(f64::from(rp), f64::from(rn));
            if eo.abs() > 1.0e-18 {
                let rel_o = (f64::from(g.d_odd_dr_curr) - eo).abs() / eo.abs();
                prop_assert!(rel_o <= 8.68e-6, "rp={rp}, rn={rn}, rel_o={rel_o}");
            }
            if ee.abs() > 1.0e-12 {
                let abs_e = (f64::from(g.d_even_dr_curr) - ee).abs();
                let rel_e = abs_e / ee.abs();
                prop_assert!(
                    rel_e <= 1.0e-5 || abs_e <= 1.0e-6,
                    "rp={rp}, rn={rn}, rel_e={rel_e}, abs_e={abs_e}"
                );
            } else {
                prop_assert!((f64::from(g.d_even_dr_curr) - ee).abs() < 1.0e-7);
            }
        }
    }
}
