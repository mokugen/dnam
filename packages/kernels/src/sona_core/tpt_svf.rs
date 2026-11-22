//! Unconditionally passive `2x2` ZDF `TPT-SVF` (`ref[dsp.op-m.recurrence.symplectic]`,
//! `ref[dsp.op-n.core.stereo-simd]`), division-free pure-`f32` bilinear prewarping,
//! and forward/adjoint `2x2` associative scan operators.

use crate::simd::{GpuVec4, KernelVec4, StereoVec8, const_taylor_inv_fact_f32};

pub mod phs;
pub use phs::{MAX_SVF_K_NL, SVF_PHS_E_MAX, step_tpt_svf_phs_stereo8, step_tpt_svf_phs_vec4};

/// Minimum normalized cutoff ratio `f_c / f_s` accepted by `prewarp_tan_pi_f_over_fs_f32`.
const MIN_PREWARP_RATIO: f32 = 1.0e-5;
/// Maximum normalized cutoff ratio `f_c / f_s` accepted by `prewarp_tan_pi_f_over_fs_f32`.
const MAX_PREWARP_RATIO: f32 = 0.495;
/// Octant boundary `f_c / f_s = 0.25` (`pi / 4`), splitting `sin / cos` and `cos / sin`.
const OCTANT_SPLIT_RATIO: f32 = 0.25;
/// Nyquist ratio `0.5` used for complementary co-function folding `0.5 - r`.
const NYQUIST_RATIO: f32 = 0.5;
/// Minimum positive tangent gain `g` for passive `TPT-SVF` coefficient derivation.
pub const MIN_TPT_G: f32 = 1.0e-5;

// Compile-time degree-11/10 Maclaurin coefficients for `sin(x)` and `cos(x)` on `[0, pi/4]`
// (`ref[repo.code.constant-eval-and-derivations]`; truncation error `< 2e-9` at `x = pi/4`).
const SIN_C1: f32 = const_taylor_inv_fact_f32(3, true);
const SIN_C2: f32 = const_taylor_inv_fact_f32(5, false);
const SIN_C3: f32 = const_taylor_inv_fact_f32(7, true);
const SIN_C4: f32 = const_taylor_inv_fact_f32(9, false);
const SIN_C5: f32 = const_taylor_inv_fact_f32(11, true);

const COS_C1: f32 = const_taylor_inv_fact_f32(2, true);
const COS_C2: f32 = const_taylor_inv_fact_f32(4, false);
const COS_C3: f32 = const_taylor_inv_fact_f32(6, true);
const COS_C4: f32 = const_taylor_inv_fact_f32(8, false);
const COS_C5: f32 = const_taylor_inv_fact_f32(10, true);

const _: () = assert!(MIN_PREWARP_RATIO > 0.0 && MIN_PREWARP_RATIO < OCTANT_SPLIT_RATIO);
const _: () = assert!(OCTANT_SPLIT_RATIO < MAX_PREWARP_RATIO && MAX_PREWARP_RATIO < NYQUIST_RATIO);
const _: () = assert!(MIN_TPT_G > 0.0);

/// Four-lane `TPT-SVF` precomputed coefficient bundle `(a1, a2, a3, k, 2k)`
/// (`ref[dsp.op-m.recurrence.symplectic]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TptSvfCoeffs4<V: KernelVec4> {
    pub a1: V,
    pub a2: V,
    pub a3: V,
    pub k: V,
    pub two_k: V,
}

impl<V: KernelVec4> TptSvfCoeffs4<V> {
    /// Derives `TPT-SVF` coefficients from prewarped tangent gain `g = tan(pi * f_c / f_s)`
    /// and damping `k = 1 / Q >= 0` using `rcp_ge1_nr2` with zero hardware `FDIV`.
    #[inline]
    #[must_use]
    pub fn from_gk(g: V, k: V) -> Self {
        let g = g.max(V::splat(MIN_TPT_G));
        let k = k.max(V::zero());
        let denom = V::splat(1.0).fma(g, g.add(k));
        let a1 = denom.rcp_ge1_nr2();
        let a2 = g.mul(a1);
        let a3 = g.mul(a2);
        let two_k = k.add(k);
        Self {
            a1,
            a2,
            a3,
            k,
            two_k,
        }
    }
}

/// Pure-`f32`, zero-`FDIV`, zero-`libm` bilinear prewarping approximation of `tan(pi * r)`
/// for normalized cutoff ratio `r = clamp(f_over_fs, 1e-5, 0.495)` (`< 5e-6` relative error).
///
/// Uses `tan(pi * r) = sin(pi * r) / cos(pi * r)` on `r <= 0.25` and complementary folding
/// `tan(pi * r) = cos(pi * (0.5 - r)) / sin(pi * (0.5 - r))` on `r > 0.25` via `rcp_pos_nr3`.
#[inline]
#[must_use]
pub fn prewarp_tan_pi_f_over_fs_f32(f_over_fs: f32) -> f32 {
    let r = if f_over_fs.is_nan() {
        MIN_PREWARP_RATIO
    } else {
        f_over_fs.clamp(MIN_PREWARP_RATIO, MAX_PREWARP_RATIO)
    };
    let lower_octant = r <= OCTANT_SPLIT_RATIO;
    let t = if lower_octant { r } else { NYQUIST_RATIO - r };
    let x = core::f32::consts::PI * t;
    let z = x * x;

    let sin_poly = 1.0 + z * (SIN_C1 + z * (SIN_C2 + z * (SIN_C3 + z * (SIN_C4 + z * SIN_C5))));
    let s = x * sin_poly;

    let c = 1.0 + z * (COS_C1 + z * (COS_C2 + z * (COS_C3 + z * (COS_C4 + z * COS_C5))));

    let (num, den) = if lower_octant { (s, c) } else { (c, s) };
    let inv_den = GpuVec4::splat(den).rcp_pos_nr3().to_array()[0];
    num * inv_den
}

/// Simultaneous four-mode filter taps emitted by a passive `2x2` ZDF `TPT-SVF` step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TptSvfOutputs<T> {
    pub lp: T,
    pub bp: T,
    pub hp: T,
    pub ap: T,
}

/// Advances 4 independent `TPT-SVF` modes in a `KernelVec4` register (`ref[dsp.op-m.recurrence.symplectic]`).
#[inline(always)]
pub fn step_tpt_svf_vec4<V: KernelVec4>(
    u: V,
    s1: &mut V,
    s2: &mut V,
    coeffs: &TptSvfCoeffs4<V>,
) -> TptSvfOutputs<V> {
    let v3 = u.sub(*s2);
    let v1 = coeffs.a1.mul(*s1).fma(coeffs.a2, v3);
    let v2 = s2.fma(coeffs.a2, *s1).fma(coeffs.a3, v3);
    *s1 = v1.add(v1).sub(*s1).flush_recursive_state();
    *s2 = v2.add(v2).sub(*s2).flush_recursive_state();
    let hp = u.fms(coeffs.k, v1).sub(v2);
    let ap = u.fms(coeffs.two_k, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

/// Advances stereo `L+R` `4`-mode `TPT-SVF` states in `StereoVec8` (`ref[dsp.op-n.core.stereo-simd]`),
/// broadcasting shared filter coefficients across both halves for `0-ULP` stereo parity.
#[inline(always)]
pub fn step_tpt_svf_stereo8<S: StereoVec8>(
    u: S,
    s1: &mut S,
    s2: &mut S,
    coeffs: &TptSvfCoeffs4<S::Half>,
) -> TptSvfOutputs<S> {
    let a1 = S::broadcast_half(coeffs.a1);
    let a2 = S::broadcast_half(coeffs.a2);
    let a3 = S::broadcast_half(coeffs.a3);
    let k = S::broadcast_half(coeffs.k);
    let two_k = S::broadcast_half(coeffs.two_k);
    let v3 = u.sub(*s2);
    let v1 = a1.mul(*s1).fma(a2, v3);
    let v2 = s2.fma(a2, *s1).fma(a3, v3);
    *s1 = v1.add(v1).sub(*s1).flush_recursive_state();
    *s2 = v2.add(v2).sub(*s2).flush_recursive_state();
    let hp = u.fms(k, v1).sub(v2);
    let ap = u.fms(two_k, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

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
///
/// For forward step `s_n = M_n s_{n-1} + b_n` with immediate step gradient `adj_s = dL_n / ds_n`,
/// the single-step element represents `lambda_{n-1} = M_n^T * (lambda_post + adj_s) = M_n^T * lambda_post + M_n^T * adj_s`,
/// so running an exclusive reverse suffix scan yields `lambda_n = adj_s_n + suffix_{n+1}.g` and `dL / ds_{-1} = full_scan.g`.
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

/// Associatively composes two reverse-BPTT `TPT-SVF` elements in reverse sweep order:
/// `earlier_in_reverse` (later time step `t_late`) is applied first, then `later_in_reverse`
/// (earlier time step `t_early < t_late`), forming:
/// `T = M_early^T * M_late^T = later_in_reverse.T * earlier_in_reverse.T`
/// and `g = later_in_reverse.T * earlier_in_reverse.g + later_in_reverse.g`.
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
    use super::*;
    use crate::simd::InterleavedStereo8;
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
    fn test_prewarp_tan_anchors_and_clamp_boundaries() {
        let cases = [
            (-1.0_f32, 1.0e-5_f64),
            (0.0, 1.0e-5),
            (1.0e-5, 1.0e-5),
            (0.001, 0.001),
            (0.125, 0.125),
            (0.25, 0.25),
            (0.250_01, 0.250_01),
            (0.4, 0.4),
            (0.495, 0.495),
            (0.9, 0.495),
            (f32::NAN, 1.0e-5),
        ];
        for (raw, clamped_f64) in cases {
            let got = prewarp_tan_pi_f_over_fs_f32(raw) as f64;
            let expected = (core::f64::consts::PI * clamped_f64).tan();
            let rel_err = (got - expected).abs() / expected.abs();
            assert!(
                rel_err < 5.0e-6,
                "raw={raw}: got={got}, expected={expected}, rel_err={rel_err}"
            );
        }
    }

    /// verify[dsp.op-n.core.tpt-tone-basis]
    #[test]
    fn test_tpt_svf_zero_equilibrium_and_filter_identities() {
        let g = GpuVec4::from_array([0.05, 0.2, 1.0, 4.0]);
        let k = GpuVec4::from_array([0.0, 0.5, core::f32::consts::SQRT_2, 2.0]);
        let coeffs = TptSvfCoeffs4::from_gk(g, k);

        let mut s1 = GpuVec4::zero();
        let mut s2 = GpuVec4::zero();
        let out = step_tpt_svf_vec4(GpuVec4::zero(), &mut s1, &mut s2, &coeffs);
        for v in [s1, s2, out.lp, out.bp, out.hp, out.ap] {
            for lane in v.to_array() {
                assert_eq!(lane.to_bits(), 0);
            }
        }

        let id_coeffs = TptSvfCoeffs4 {
            a1: GpuVec4::splat(1.0),
            a2: GpuVec4::zero(),
            a3: GpuVec4::zero(),
            k: GpuVec4::zero(),
            two_k: GpuVec4::zero(),
        };
        let mut sub_s1 = GpuVec4::splat(f32::from_bits(1));
        let mut sub_s2 = GpuVec4::splat(f32::from_bits(1));
        let _ = step_tpt_svf_vec4(GpuVec4::zero(), &mut sub_s1, &mut sub_s2, &id_coeffs);
        assert_eq!(sub_s1.to_array(), [0.0; 4]);
        assert_eq!(sub_s2.to_array(), [0.0; 4]);

        let mut st_s1 = InterleavedStereo8::<GpuVec4>::zero();
        let mut st_s2 = InterleavedStereo8::<GpuVec4>::zero();
        let u_l = GpuVec4::from_array([0.5, -0.75, 1.25, -0.25]);
        let u_r = GpuVec4::from_array([-0.3, 0.8, -1.1, 0.4]);
        let mut v_s1_l = GpuVec4::zero();
        let mut v_s2_l = GpuVec4::zero();
        let mut v_s1_r = GpuVec4::zero();
        let mut v_s2_r = GpuVec4::zero();

        for n in 0..16 {
            let scale = if n < 4 { 1.0 } else { 0.0 };
            let ul = u_l.mul(GpuVec4::splat(scale));
            let ur = u_r.mul(GpuVec4::splat(scale));
            let u_st = InterleavedStereo8::from_lr(ul, ur);
            let out_st = step_tpt_svf_stereo8(u_st, &mut st_s1, &mut st_s2, &coeffs);
            let out_l = step_tpt_svf_vec4(ul, &mut v_s1_l, &mut v_s2_l, &coeffs);
            let out_r = step_tpt_svf_vec4(ur, &mut v_s1_r, &mut v_s2_r, &coeffs);

            let (st_lp_l, st_lp_r) = out_st.lp.into_lr();
            let (st_bp_l, st_bp_r) = out_st.bp.into_lr();
            let (st_hp_l, st_hp_r) = out_st.hp.into_lr();
            let (st_ap_l, st_ap_r) = out_st.ap.into_lr();
            assert_eq!(st_lp_l.to_array(), out_l.lp.to_array());
            assert_eq!(st_lp_r.to_array(), out_r.lp.to_array());
            assert_eq!(st_bp_l.to_array(), out_l.bp.to_array());
            assert_eq!(st_bp_r.to_array(), out_r.bp.to_array());
            assert_eq!(st_hp_l.to_array(), out_l.hp.to_array());
            assert_eq!(st_hp_r.to_array(), out_r.hp.to_array());
            assert_eq!(st_ap_l.to_array(), out_l.ap.to_array());
            assert_eq!(st_ap_r.to_array(), out_r.ap.to_array());

            for lane in 0..4 {
                let sum = out_l.lp.to_array()[lane]
                    + k.to_array()[lane] * out_l.bp.to_array()[lane]
                    + out_l.hp.to_array()[lane];
                assert!((sum - ul.to_array()[lane]).abs() < 1.0e-5);
            }
        }
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
        fn prop_prewarp_tan_matches_f64(r in 1.0e-4_f32..0.49_f32) {
            let got = prewarp_tan_pi_f_over_fs_f32(r) as f64;
            let expected = (core::f64::consts::PI * (r as f64)).tan();
            let rel_err = (got - expected).abs() / expected.abs();
            prop_assert!(rel_err < 5.0e-6, "r={r}, rel_err={rel_err}");
        }

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
                // Verify passive zero-input energy norm <= 1 via Frobenius/spectral check on M:
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

            // Binary-tree associative reduction across all 16 elements.
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

            // Finite-difference checks against f64 oracle.
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
