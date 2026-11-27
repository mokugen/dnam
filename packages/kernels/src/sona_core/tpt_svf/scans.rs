//! Forward and reverse-BPTT `2x2` associative scan operators for passive `ZDF` `TPT-SVF`
//! (`ref[dsp.op-m.recurrence.symplectic]`, `ref[arch.operators.differentiability]`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`,
//! and `spirv-unknown-naga-wgsl`-clean.

/// Forward `2x2` affine state-transition element `s_out = M * s_in + b` for associative Blelloch scans.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TptScanElem {
    pub m00: f32,
    pub m01: f32,
    pub m10: f32,
    pub m11: f32,
    pub b0: f32,
    pub b1: f32,
}

impl TptScanElem {
    /// Identity transition element (`M = I`, `b = [0, 0]`).
    #[inline(always)]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m00: 1.0,
            m01: 0.0,
            m10: 0.0,
            m11: 1.0,
            b0: 0.0,
            b1: 0.0,
        }
    }

    /// Constructs the single-step `2x2` passive `TPT-SVF` transition from `(a1, a2, a3)` and input `u`.
    #[inline(always)]
    #[must_use]
    pub fn from_step(a1: f32, a2: f32, a3: f32, u: f32) -> Self {
        let two_a2 = a2 + a2;
        let two_a3 = a3 + a3;
        Self {
            m00: (a1 + a1) - 1.0,
            m01: -two_a2,
            m10: two_a2,
            m11: 1.0 - two_a3,
            b0: two_a2 * u,
            b1: two_a3 * u,
        }
    }
}

/// Associatively composes `first` followed by `second`: `(M2 * M1, M2 * b1 + b2)`.
#[inline(always)]
#[must_use]
pub fn compose_tpt_scan(first: TptScanElem, second: TptScanElem) -> TptScanElem {
    TptScanElem {
        m00: second.m00 * first.m00 + second.m01 * first.m10,
        m01: second.m00 * first.m01 + second.m01 * first.m11,
        m10: second.m10 * first.m00 + second.m11 * first.m10,
        m11: second.m10 * first.m01 + second.m11 * first.m11,
        b0: second.m00 * first.b0 + second.m01 * first.b1 + second.b0,
        b1: second.m10 * first.b0 + second.m11 * first.b1 + second.b1,
    }
}

/// Evaluates `elem.M * s0 + elem.b` to recover the state from initial condition `s0`.
#[inline(always)]
#[must_use]
pub fn apply_tpt_init(elem: TptScanElem, s0: [f32; 2]) -> [f32; 2] {
    [
        elem.m00 * s0[0] + elem.m01 * s0[1] + elem.b0,
        elem.m10 * s0[0] + elem.m11 * s0[1] + elem.b1,
    ]
}

/// Reverse-BPTT `2x2` transpose adjoint scan element (`lambda_pre = T * lambda_post + g`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TptAdjointElem {
    pub t00: f32,
    pub t01: f32,
    pub t10: f32,
    pub t11: f32,
    pub g0: f32,
    pub g1: f32,
}

impl TptAdjointElem {
    /// Identity reverse-adjoint element (`T = I`, `g = [0, 0]`).
    #[inline(always)]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            t00: 1.0,
            t01: 0.0,
            t10: 0.0,
            t11: 1.0,
            g0: 0.0,
            g1: 0.0,
        }
    }

    /// Constructs the reverse-mode pullback element for `(a1, a2, a3)` and local loss gradient `(adj_s1, adj_s2)`.
    #[inline(always)]
    #[must_use]
    pub fn from_step_grad(a1: f32, a2: f32, a3: f32, adj_s1: f32, adj_s2: f32) -> Self {
        let two_a2 = a2 + a2;
        let two_a3 = a3 + a3;
        let t00 = (a1 + a1) - 1.0;
        let t01 = two_a2;
        let t10 = -two_a2;
        let t11 = 1.0 - two_a3;
        Self {
            t00,
            t01,
            t10,
            t11,
            g0: t00 * adj_s1 + t01 * adj_s2,
            g1: t10 * adj_s1 + t11 * adj_s2,
        }
    }
}

/// Associatively composes two reverse-BPTT `TPT-SVF` elements in reverse sweep order.
#[inline(always)]
#[must_use]
pub fn compose_tpt_adjoint_scan(
    later_in_reverse: TptAdjointElem,
    earlier_in_reverse: TptAdjointElem,
) -> TptAdjointElem {
    TptAdjointElem {
        t00: later_in_reverse.t00 * earlier_in_reverse.t00
            + later_in_reverse.t01 * earlier_in_reverse.t10,
        t01: later_in_reverse.t00 * earlier_in_reverse.t01
            + later_in_reverse.t01 * earlier_in_reverse.t11,
        t10: later_in_reverse.t10 * earlier_in_reverse.t00
            + later_in_reverse.t11 * earlier_in_reverse.t10,
        t11: later_in_reverse.t10 * earlier_in_reverse.t01
            + later_in_reverse.t11 * earlier_in_reverse.t11,
        g0: later_in_reverse.t00 * earlier_in_reverse.g0
            + later_in_reverse.t01 * earlier_in_reverse.g1
            + later_in_reverse.g0,
        g1: later_in_reverse.t10 * earlier_in_reverse.g0
            + later_in_reverse.t11 * earlier_in_reverse.g1
            + later_in_reverse.g1,
    }
}

/// Applies a reverse-adjoint element to an outgoing downstream adjoint `adj_out`.
#[inline(always)]
#[must_use]
pub fn apply_tpt_adjoint(elem: TptAdjointElem, adj_out: [f32; 2]) -> [f32; 2] {
    [
        elem.t00 * adj_out[0] + elem.t01 * adj_out[1] + elem.g0,
        elem.t10 * adj_out[0] + elem.t11 * adj_out[1] + elem.g1,
    ]
}

#[cfg(test)]
mod tests {
    use super::super::{
        TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32, step_tpt_svf_stereo8, step_tpt_svf_vec4,
    };
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8, KernelVec4, StereoVec8};
    use proptest::prelude::*;

    fn tpt_forward_loss_f64(
        a1: &[f64],
        a2: &[f64],
        a3: &[f64],
        u: &[f64],
        s_init: [f64; 2],
        w1: &[f64],
        w2: &[f64],
    ) -> f64 {
        let mut s1 = s_init[0];
        let mut s2 = s_init[1];
        let mut loss = 0.0;
        for t in 0..a1.len() {
            let m00 = 2.0 * a1[t] - 1.0;
            let m01 = -2.0 * a2[t];
            let m10 = 2.0 * a2[t];
            let m11 = 1.0 - 2.0 * a3[t];
            let b0 = 2.0 * a2[t] * u[t];
            let b1 = 2.0 * a3[t] * u[t];
            let s1_next = m00 * s1 + m01 * s2 + b0;
            let s2_next = m10 * s1 + m11 * s2 + b1;
            s1 = s1_next;
            s2 = s2_next;
            loss += w1[t] * s1 + w2[t] * s2;
        }
        loss
    }

    #[test]
    fn test_tpt_scan_identity_and_associativity() {
        let id = TptScanElem::identity();
        let e = TptScanElem::from_step(0.6, 0.3, 0.15, 0.8);
        assert_eq!(compose_tpt_scan(id, e), e);
        assert_eq!(compose_tpt_scan(e, id), e);
        assert_eq!(apply_tpt_init(id, [1.5, -2.5]), [1.5, -2.5]);

        let adj_id = TptAdjointElem::identity();
        let ae = TptAdjointElem::from_step_grad(0.6, 0.3, 0.15, 0.4, -0.7);
        assert_eq!(compose_tpt_adjoint_scan(adj_id, ae), ae);
        assert_eq!(compose_tpt_adjoint_scan(ae, adj_id), ae);
        assert_eq!(apply_tpt_adjoint(adj_id, [0.25, -0.5]), [0.25, -0.5]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_tpt_scan_matches_sequential_and_stereo8_parity(
            ratios in proptest::collection::vec(1.0e-3_f32..0.45_f32, 16),
            qs in proptest::collection::vec(0.25_f32..12.0_f32, 16),
            u_seq in proptest::collection::vec(-4.0_f32..4.0_f32, 16),
            s0_1 in -2.0_f32..2.0_f32,
            s0_2 in -2.0_f32..2.0_f32,
        ) {
            let mut elems = [TptScanElem::identity(); 16];
            let mut seq_s1 = GpuVec4::splat(s0_1);
            let mut seq_s2 = GpuVec4::splat(s0_2);
            let mut st_s1 = InterleavedStereo8::<GpuVec4>::splat(s0_1);
            let mut st_s2 = InterleavedStereo8::<GpuVec4>::splat(s0_2);
            let mut scan_acc = TptScanElem::identity();

            for t in 0..16 {
                let g = prewarp_tan_pi_f_over_fs_f32(ratios[t]);
                let k = 1.0_f32 / qs[t];
                let coeffs = TptSvfCoeffs4::from_gk(GpuVec4::splat(g), GpuVec4::splat(k));
                let a1 = coeffs.a1.to_array()[0];
                let a2 = coeffs.a2.to_array()[0];
                let a3 = coeffs.a3.to_array()[0];

                let u_v = GpuVec4::splat(u_seq[t]);
                let out_v = step_tpt_svf_vec4(u_v, &mut seq_s1, &mut seq_s2, &coeffs);
                let out_st = step_tpt_svf_stereo8(
                    InterleavedStereo8::splat(u_seq[t]),
                    &mut st_s1,
                    &mut st_s2,
                    &coeffs,
                );
                let (lp_l, lp_r) = out_st.lp.into_lr();
                prop_assert_eq!(lp_l.to_array(), out_v.lp.to_array());
                prop_assert_eq!(lp_r.to_array(), out_v.lp.to_array());

                elems[t] = TptScanElem::from_step(a1, a2, a3, u_seq[t]);
                let energy_0 = apply_tpt_init(
                    TptScanElem::from_step(a1, a2, a3, 0.0),
                    [1.0, 0.0],
                );
                let n0 = energy_0[0] * energy_0[0] + energy_0[1] * energy_0[1];
                prop_assert!(n0 <= 1.0 + 1.0e-4, "n0={n0}");

                scan_acc = compose_tpt_scan(scan_acc, elems[t]);
                let scanned_state = apply_tpt_init(scan_acc, [s0_1, s0_2]);
                prop_assert!((scanned_state[0] - seq_s1.to_array()[0]).abs() < 1.0e-4);
                prop_assert!((scanned_state[1] - seq_s2.to_array()[0]).abs() < 1.0e-4);
            }

            let l1: [TptScanElem; 8] =
                core::array::from_fn(|i| compose_tpt_scan(elems[2 * i], elems[2 * i + 1]));
            let l2: [TptScanElem; 4] =
                core::array::from_fn(|i| compose_tpt_scan(l1[2 * i], l1[2 * i + 1]));
            let tree_total = compose_tpt_scan(
                compose_tpt_scan(l2[0], l2[1]),
                compose_tpt_scan(l2[2], l2[3]),
            );
            let tree_final = apply_tpt_init(tree_total, [s0_1, s0_2]);
            prop_assert!((tree_final[0] - seq_s1.to_array()[0]).abs() < 1.0e-4);
            prop_assert!((tree_final[1] - seq_s2.to_array()[0]).abs() < 1.0e-4);
        }

        #[test]
        fn prop_tpt_adjoint_scan_matches_finite_differences(
            gs in proptest::collection::vec(0.05_f32..3.0_f32, 8),
            ks in proptest::collection::vec(0.05_f32..2.0_f32, 8),
            u_seq in proptest::collection::vec(-2.0_f32..2.0_f32, 8),
            w1_seq in proptest::collection::vec(-1.0_f32..1.0_f32, 8),
            w2_seq in proptest::collection::vec(-1.0_f32..1.0_f32, 8),
            s0_1 in -1.0_f32..1.0_f32,
            s0_2 in -1.0_f32..1.0_f32,
        ) {
            let mut a1 = [0.0_f32; 8];
            let mut a2 = [0.0_f32; 8];
            let mut a3 = [0.0_f32; 8];
            let mut s_prev_hist = [[0.0_f32; 2]; 8];
            let mut s_curr = [s0_1, s0_2];

            for t in 0..8 {
                let coeffs = TptSvfCoeffs4::from_gk(GpuVec4::splat(gs[t]), GpuVec4::splat(ks[t]));
                a1[t] = coeffs.a1.to_array()[0];
                a2[t] = coeffs.a2.to_array()[0];
                a3[t] = coeffs.a3.to_array()[0];
                s_prev_hist[t] = s_curr;
                s_curr = apply_tpt_init(TptScanElem::from_step(a1[t], a2[t], a3[t], u_seq[t]), s_curr);
            }

            let mut adj_elems = [TptAdjointElem::identity(); 8];
            let mut adj_u = [0.0_f32; 8];
            let mut adj_a1 = [0.0_f32; 8];
            let mut adj_a2 = [0.0_f32; 8];
            let mut adj_a3 = [0.0_f32; 8];
            let mut rev_acc = TptAdjointElem::identity();

            for rev in 0..8 {
                let t = 7 - rev;
                let lam = [w1_seq[t] + rev_acc.g0, w2_seq[t] + rev_acc.g1];
                let (two_a2, two_a3) = (a2[t] + a2[t], a3[t] + a3[t]);
                adj_u[t] = two_a2 * lam[0] + two_a3 * lam[1];
                adj_a1[t] = 2.0 * lam[0] * s_prev_hist[t][0];
                adj_a2[t] = 2.0 * lam[0] * (u_seq[t] - s_prev_hist[t][1])
                    + 2.0 * lam[1] * s_prev_hist[t][0];
                adj_a3[t] = 2.0 * lam[1] * (u_seq[t] - s_prev_hist[t][1]);

                adj_elems[t] =
                    TptAdjointElem::from_step_grad(a1[t], a2[t], a3[t], w1_seq[t], w2_seq[t]);
                rev_acc = compose_tpt_adjoint_scan(adj_elems[t], rev_acc);
            }
            let adj_s0 = apply_tpt_adjoint(rev_acc, [0.0, 0.0]);

            let q03 = compose_tpt_adjoint_scan(
                compose_tpt_adjoint_scan(adj_elems[0], adj_elems[1]),
                compose_tpt_adjoint_scan(adj_elems[2], adj_elems[3]),
            );
            let q47 = compose_tpt_adjoint_scan(
                compose_tpt_adjoint_scan(adj_elems[4], adj_elems[5]),
                compose_tpt_adjoint_scan(adj_elems[6], adj_elems[7]),
            );
            let tree_adj = compose_tpt_adjoint_scan(q03, q47);
            prop_assert!((tree_adj.g0 - adj_s0[0]).abs() < 1.0e-5);
            prop_assert!((tree_adj.g1 - adj_s0[1]).abs() < 1.0e-5);

            let (a1_64, a2_64, a3_64) = (a1.map(f64::from), a2.map(f64::from), a3.map(f64::from));
            let u_64: Vec<f64> = u_seq.iter().map(|&v| v as f64).collect();
            let w1_64: Vec<f64> = w1_seq.iter().map(|&v| v as f64).collect();
            let w2_64: Vec<f64> = w2_seq.iter().map(|&v| v as f64).collect();
            let s0_64 = [s0_1 as f64, s0_2 as f64];
            let eps = 1.0e-6_f64;
            let fd_s0 = |d0: f64, d1: f64| {
                let hi = tpt_forward_loss_f64(
                    &a1_64, &a2_64, &a3_64, &u_64, [s0_64[0] + d0, s0_64[1] + d1], &w1_64, &w2_64,
                );
                let lo = tpt_forward_loss_f64(
                    &a1_64, &a2_64, &a3_64, &u_64, [s0_64[0] - d0, s0_64[1] - d1], &w1_64, &w2_64,
                );
                (hi - lo) / (2.0 * eps)
            };
            prop_assert!((adj_s0[0] as f64 - fd_s0(eps, 0.0)).abs() < 1.0e-4);
            prop_assert!((adj_s0[1] as f64 - fd_s0(0.0, eps)).abs() < 1.0e-4);

            let fd_slice = |src: &[f64], t: usize, f: &dyn Fn(&[f64]) -> f64| {
                let mut v = src.to_vec();
                v[t] += eps;
                let hi = f(&v);
                v[t] -= 2.0 * eps;
                (hi - f(&v)) / (2.0 * eps)
            };
            for t in 0..8 {
                let fd_u = fd_slice(&u_64, t, &|v| {
                    tpt_forward_loss_f64(&a1_64, &a2_64, &a3_64, v, s0_64, &w1_64, &w2_64)
                });
                let fd_a1 = fd_slice(&a1_64, t, &|v| {
                    tpt_forward_loss_f64(v, &a2_64, &a3_64, &u_64, s0_64, &w1_64, &w2_64)
                });
                let fd_a2 = fd_slice(&a2_64, t, &|v| {
                    tpt_forward_loss_f64(&a1_64, v, &a3_64, &u_64, s0_64, &w1_64, &w2_64)
                });
                let fd_a3 = fd_slice(&a3_64, t, &|v| {
                    tpt_forward_loss_f64(&a1_64, &a2_64, v, &u_64, s0_64, &w1_64, &w2_64)
                });
                prop_assert!((adj_u[t] as f64 - fd_u).abs() < 1.0e-4);
                prop_assert!((adj_a1[t] as f64 - fd_a1).abs() < 1.0e-4);
                prop_assert!((adj_a2[t] as f64 - fd_a2).abs() < 1.0e-4);
                prop_assert!((adj_a3[t] as f64 - fd_a3).abs() < 1.0e-4);
            }
        }
    }
}
