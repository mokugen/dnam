//! Cabinet `Op-M` Kautz modal state-space steps, rapid-sweep radial contraction, associative
//! complex modal scan operators, and contractive rotor / `Q0.32` phasor facade
//! (`ref[dsp.op-m.kautz.sweep-contraction]`, `ref[dsp.op-m.kautz.coupled-form]`,
//! `ref[dsp.op-m.kautz.dual-state]`, `ref[dsp.op-m.recurrence.stereo-simd]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`/`UDIV`, zero 64-bit scalars, and zero `libm`
//! calls in runtime code (`ref[sdk.rust.shared-nostd-kernels]`).

pub mod rotor_and_phasor;

pub use rotor_and_phasor::{PhasorQ32Bank4, algebraic_unit_rotor_f32, eval_q32_quadrature_f32};

use crate::simd::{
    KernelVec4, MAX_PARALRU_R_F32, StereoVec8, flush_subnormal_f32_bits, scrub_and_flush_f32_bits,
};

/// Minimum contracted pole radius allowed during rapid cabinet formant sweeps
/// (`ref[dsp.op-m.kautz.sweep-contraction]`).
const KAUTZ_SWEEP_MIN_R: f32 = 0.25;

/// Zero-sweep baseline radius ceiling (`0.9995`, cabinet acoustic Kautz pole ceiling per
/// `docs/specs/dsp/op_m_modal_kautz.md`) for cabinet dynamic sweep contraction
/// (`ref[dsp.op-m.kautz.sweep-contraction]`).
const KAUTZ_SWEEP_BASE_CEIL: f32 = 0.9995;

/// Sweep-rate slope coefficient (`0.15 * |delta_theta|`) for cabinet contraction.
const KAUTZ_SWEEP_SLOPE: f32 = 0.15;

const _: () = assert!(KAUTZ_SWEEP_MIN_R > 0.0 && KAUTZ_SWEEP_MIN_R < KAUTZ_SWEEP_BASE_CEIL);
const _: () = assert!(KAUTZ_SWEEP_BASE_CEIL <= MAX_PARALRU_R_F32);

/// Evaluates cabinet rapid-sweep radial contraction (`ref[dsp.op-m.kautz.sweep-contraction]`)
/// purely in 32-bit `f32` (`min(radius, max(0.25, 0.9995 - 0.15 * |delta_theta|))`).
#[inline(always)]
#[must_use]
pub fn eval_kautz_sweep_contraction_f32(radius: f32, delta_theta_rad: f32) -> f32 {
    let r_clamped = if radius.is_finite() {
        radius.clamp(0.0, MAX_PARALRU_R_F32)
    } else {
        KAUTZ_SWEEP_MIN_R
    };
    let abs_dtheta = if delta_theta_rad.is_finite() {
        crate::simd::const_abs_f32(delta_theta_rad)
    } else {
        8.0
    };
    let ceiling = (KAUTZ_SWEEP_BASE_CEIL - KAUTZ_SWEEP_SLOPE * abs_dtheta)
        .clamp(KAUTZ_SWEEP_MIN_R, MAX_PARALRU_R_F32);
    r_clamped.min(ceiling)
}

/// 4-pole stereo cabinet Kautz modal state block (`4` shared complex conjugate pole pairs
/// across simultaneous `L` and `R` states, `ref[dsp.op-m.kautz.dual-state]`,
/// `ref[dsp.op-m.kautz.coupled-form]`, `ref[dsp.op-m.recurrence.stereo-simd]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KautzQuadState<V: KernelVec4> {
    /// Left-channel modal real state across 4 conjugate pole pairs.
    pub z_re_l: V,
    /// Left-channel modal imaginary state across 4 conjugate pole pairs.
    pub z_im_l: V,
    /// Right-channel modal real state across 4 conjugate pole pairs.
    pub z_re_r: V,
    /// Right-channel modal imaginary state across 4 conjugate pole pairs.
    pub z_im_r: V,
    /// Shared complex pole real part `r * cos(theta)`.
    pub alpha_re: V,
    /// Shared complex pole imaginary part `r * sin(theta)`.
    pub alpha_im: V,
    /// Shared normalized input drive weight `b_re = kappa * sqrt(1 - r_high^2)`.
    pub b_re: V,
    /// Left-channel modal readout real component.
    pub c_re_l: V,
    /// Left-channel modal readout imaginary component.
    pub c_im_l: V,
    /// Right-channel modal readout real component.
    pub c_re_r: V,
    /// Right-channel modal readout imaginary component.
    pub c_im_r: V,
}

impl<V: KernelVec4> KautzQuadState<V> {
    /// Constructs a zeroed stereo [`KautzQuadState`] from shared pole/drive parameters and
    /// independent `L`/`R` complex modal readouts.
    #[inline(always)]
    #[must_use]
    pub fn new(
        alpha_re: V,
        alpha_im: V,
        b_re: V,
        c_re_l: V,
        c_im_l: V,
        c_re_r: V,
        c_im_r: V,
    ) -> Self {
        Self {
            z_re_l: V::zero(),
            z_im_l: V::zero(),
            z_re_r: V::zero(),
            z_im_r: V::zero(),
            alpha_re,
            alpha_im,
            b_re,
            c_re_l,
            c_im_l,
            c_re_r,
            c_im_r,
        }
    }

    /// Zeros both Left and Right modal complex state vectors.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.z_re_l = V::zero();
        self.z_im_l = V::zero();
        self.z_re_r = V::zero();
        self.z_im_r = V::zero();
    }
}

/// Steps 4 conjugate Kautz modal pole pairs in both Left and Right channels simultaneously using
/// 4-lane [`KernelVec4`] vectors (`ref[dsp.op-m.kautz.coupled-form]`,
/// `ref[dsp.op-m.recurrence.symplectic]`).
///
/// Uses identical operation order and pairwise reductions as [`step_kautz_quad_stereo8`], yielding
/// exact `0-ULP` stereo parity against `InterleavedStereo8<V>`.
#[inline(always)]
pub fn step_kautz_quad_vec4<V: KernelVec4>(
    state: &mut KautzQuadState<V>,
    x_l: f32,
    x_r: f32,
) -> (f32, f32) {
    let xl = V::splat(scrub_and_flush_f32_bits(x_l));
    let xr = V::splat(scrub_and_flush_f32_bits(x_r));
    let z_re_l = state.z_re_l.flush_recursive_state();
    let z_im_l = state.z_im_l.flush_recursive_state();
    let z_re_r = state.z_re_r.flush_recursive_state();
    let z_im_r = state.z_im_r.flush_recursive_state();

    let next_re_l = state
        .b_re
        .mul(xl)
        .fma(state.alpha_re, z_re_l)
        .fms(state.alpha_im, z_im_l)
        .flush_recursive_state();
    let next_im_l = state
        .alpha_re
        .mul(z_im_l)
        .fma(state.alpha_im, z_re_l)
        .flush_recursive_state();

    let next_re_r = state
        .b_re
        .mul(xr)
        .fma(state.alpha_re, z_re_r)
        .fms(state.alpha_im, z_im_r)
        .flush_recursive_state();
    let next_im_r = state
        .alpha_re
        .mul(z_im_r)
        .fma(state.alpha_im, z_re_r)
        .flush_recursive_state();

    state.z_re_l = next_re_l;
    state.z_im_l = next_im_l;
    state.z_re_r = next_re_r;
    state.z_im_r = next_im_r;

    let modal_l = state.c_re_l.mul(next_re_l).fms(state.c_im_l, next_im_l);
    let modal_r = state.c_re_r.mul(next_re_r).fms(state.c_im_r, next_im_r);
    let (sum_l, sum_r) = modal_l.horizontal_sum_pair(modal_r);
    (
        scrub_and_flush_f32_bits(2.0 * sum_l),
        scrub_and_flush_f32_bits(2.0 * sum_r),
    )
}

/// Steps 4 shared conjugate Kautz modal pole pairs in both Left and Right channels simultaneously
/// in 8-lane [`StereoVec8`] with shared `broadcast_half` pole loads and exact `0-ULP` stereo
/// parity (`ref[dsp.op-m.recurrence.stereo-simd]`, `ref[arch.simd.two-level-traits]`).
#[inline(always)]
pub fn step_kautz_quad_stereo8<S: StereoVec8>(
    state: &mut KautzQuadState<S::Half>,
    x_l: f32,
    x_r: f32,
) -> (f32, f32) {
    let z_re = S::from_lr(state.z_re_l, state.z_re_r).flush_recursive_state();
    let z_im = S::from_lr(state.z_im_l, state.z_im_r).flush_recursive_state();
    let alpha_re = S::broadcast_half(state.alpha_re);
    let alpha_im = S::broadcast_half(state.alpha_im);
    let b_re = S::broadcast_half(state.b_re);
    let c_re = S::from_lr(state.c_re_l, state.c_re_r);
    let c_im = S::from_lr(state.c_im_l, state.c_im_r);
    let x = S::splat_lr(scrub_and_flush_f32_bits(x_l), scrub_and_flush_f32_bits(x_r));

    let next_re = b_re
        .mul(x)
        .fma(alpha_re, z_re)
        .fms(alpha_im, z_im)
        .flush_recursive_state();
    let next_im = alpha_re
        .mul(z_im)
        .fma(alpha_im, z_re)
        .flush_recursive_state();
    (state.z_re_l, state.z_re_r) = next_re.into_lr();
    (state.z_im_l, state.z_im_r) = next_im.into_lr();

    let modal = c_re.mul(next_re).fms(c_im, next_im);
    let (sum_l, sum_r) = modal.horizontal_sum_lr();
    (
        scrub_and_flush_f32_bits(2.0 * sum_l),
        scrub_and_flush_f32_bits(2.0 * sum_r),
    )
}

/// Associative affine complex recurrence element representing `z -> a * z + b` for Blelloch /
/// Kogge-Stone parallel modal block scans (`ref[sdk.rust.shared-nostd-kernels]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComplexModalScanElem {
    /// State transition real coefficient `a_re`.
    pub a_re: f32,
    /// State transition imaginary coefficient `a_im`.
    pub a_im: f32,
    /// Forced input injection real term `b_re`.
    pub b_re: f32,
    /// Forced input injection imaginary term `b_im`.
    pub b_im: f32,
}

impl Default for ComplexModalScanElem {
    fn default() -> Self {
        Self::identity()
    }
}

impl ComplexModalScanElem {
    /// Returns the scan monoid identity element `(a = 1 + 0i, b = 0 + 0i)`.
    #[inline(always)]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            a_re: 1.0,
            a_im: 0.0,
            b_re: 0.0,
            b_im: 0.0,
        }
    }

    /// Constructs the per-sample complex modal transition `z_next = alpha * z + b_re * x`.
    #[inline(always)]
    #[must_use]
    pub const fn from_step(alpha_re: f32, alpha_im: f32, b_re: f32, x: f32) -> Self {
        Self {
            a_re: flush_subnormal_f32_bits(alpha_re),
            a_im: flush_subnormal_f32_bits(alpha_im),
            b_re: flush_subnormal_f32_bits(b_re * x),
            b_im: 0.0,
        }
    }
}

/// Associative composition `(first ; second)` applying `first` followed by `second`:
/// `z -> second.a * (first.a * z + first.b) + second.b`.
#[inline(always)]
#[must_use]
pub const fn compose_complex_modal_scan(
    first: ComplexModalScanElem,
    second: ComplexModalScanElem,
) -> ComplexModalScanElem {
    let a_re = second.a_re * first.a_re - second.a_im * first.a_im;
    let a_im = second.a_re * first.a_im + second.a_im * first.a_re;
    let b_re = second.a_re * first.b_re - second.a_im * first.b_im + second.b_re;
    let b_im = second.a_re * first.b_im + second.a_im * first.b_re + second.b_im;
    ComplexModalScanElem {
        a_re: flush_subnormal_f32_bits(a_re),
        a_im: flush_subnormal_f32_bits(a_im),
        b_re: flush_subnormal_f32_bits(b_re),
        b_im: flush_subnormal_f32_bits(b_im),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    /// verify[dsp.op-m.kautz.sweep-contraction]
    /// verify[dsp.op-m.kautz.dual-state]
    /// verify[dsp.op-m.kautz.coupled-form]
    /// verify[dsp.op-m.recurrence.stereo-simd]
    #[test]
    fn test_kautz_sweep_contraction_and_zero_ulp_parity() {
        assert_eq!(eval_kautz_sweep_contraction_f32(0.9995, 0.0), 0.9995);
        assert_eq!(eval_kautz_sweep_contraction_f32(0.9999, 0.0), 0.9995);
        assert_eq!(eval_kautz_sweep_contraction_f32(0.85, 0.0), 0.85);
        let c_pos = eval_kautz_sweep_contraction_f32(0.9995, 0.1);
        assert!((c_pos - (0.9995 - 0.015)).abs() < 1.0e-6 && c_pos < 0.9995);
        let c_neg = eval_kautz_sweep_contraction_f32(0.9995, -0.2);
        assert!((c_neg - (0.9995 - 0.030)).abs() < 1.0e-6);
        assert_eq!(eval_kautz_sweep_contraction_f32(0.9995, 10.0), 0.25);
        assert_eq!(eval_kautz_sweep_contraction_f32(f32::NAN, f32::NAN), 0.25);

        let alpha_re = GpuVec4::from_array([0.92, 0.85, 0.70, 0.40]);
        let alpha_im = GpuVec4::from_array([0.12, 0.25, 0.45, 0.65]);
        let b_re = GpuVec4::from_array([0.1, 0.15, 0.2, 0.25]);
        let c_re_l = GpuVec4::from_array([0.5, -0.3, 0.2, 0.1]);
        let c_im_l = GpuVec4::from_array([0.1, 0.2, -0.15, 0.05]);
        let c_re_r = GpuVec4::from_array([0.4, 0.25, -0.1, 0.3]);
        let c_im_r = GpuVec4::from_array([-0.2, 0.1, 0.05, -0.15]);

        let mut kv = KautzQuadState::new(alpha_re, alpha_im, b_re, c_re_l, c_im_l, c_re_r, c_im_r);
        let mut ks = kv;
        for n in 0..64 {
            let xl = if n == 0 {
                1.0
            } else {
                ((n as f32) * 0.21).clamp(-1.0, 1.0) * 0.1
            };
            let xr = if n == 1 {
                -0.8
            } else {
                ((n as f32) * 0.13).clamp(-1.0, 1.0) * 0.1
            };
            let out_v = step_kautz_quad_vec4(&mut kv, xl, xr);
            let out_s = step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks, xl, xr);
            assert_eq!(out_v.0.to_bits(), out_s.0.to_bits());
            assert_eq!(out_v.1.to_bits(), out_s.1.to_bits());
            assert_eq!(kv, ks);
        }
        kv.reset();
        assert_eq!(kv.z_re_l, GpuVec4::zero());
    }

    #[test]
    fn test_complex_modal_scan_identity_associativity_and_sequential_equivalence() {
        let id = ComplexModalScanElem::default();
        let e = ComplexModalScanElem::from_step(0.9, 0.2, 0.15, 0.5);
        assert_eq!(compose_complex_modal_scan(id, e), e);
        assert_eq!(compose_complex_modal_scan(e, id), e);

        let a_re = 0.91_f32;
        let a_im = 0.23_f32;
        let b_re = 0.25_f32;
        let xs = [1.0_f32, -0.5, 0.25, 0.75, -0.125, 0.0, 0.4, -0.6];

        let mut state = KautzQuadState::new(
            GpuVec4::from_array([a_re, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([a_im, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([b_re, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([0.5, 0.0, 0.0, 0.0]),
            GpuVec4::zero(),
            GpuVec4::from_array([0.5, 0.0, 0.0, 0.0]),
            GpuVec4::zero(),
        );
        let mut scan = ComplexModalScanElem::identity();
        for &x in &xs {
            let _ = step_kautz_quad_vec4(&mut state, x, x);
            scan = compose_complex_modal_scan(
                scan,
                ComplexModalScanElem::from_step(a_re, a_im, b_re, x),
            );
        }
        assert!((state.z_re_l.to_array()[0] - scan.b_re).abs() < 1.0e-5);
        assert!((state.z_im_l.to_array()[0] - scan.b_im).abs() < 1.0e-5);

        let elems: [ComplexModalScanElem; 8] =
            core::array::from_fn(|i| ComplexModalScanElem::from_step(a_re, a_im, b_re, xs[i]));
        let p01 = compose_complex_modal_scan(elems[0], elems[1]);
        let p23 = compose_complex_modal_scan(elems[2], elems[3]);
        let p45 = compose_complex_modal_scan(elems[4], elems[5]);
        let p67 = compose_complex_modal_scan(elems[6], elems[7]);
        let tree = compose_complex_modal_scan(
            compose_complex_modal_scan(p01, p23),
            compose_complex_modal_scan(p45, p67),
        );
        assert!((tree.b_re - scan.b_re).abs() < 1.0e-5);
        assert!((tree.b_im - scan.b_im).abs() < 1.0e-5);
    }

    #[test]
    fn test_kautz_quad_scrubs_nonfinite_inputs_and_poisoned_state() {
        let base_state = || {
            KautzQuadState::new(
                GpuVec4::from_array([0.92, 0.85, 0.70, 0.40]),
                GpuVec4::from_array([0.12, 0.25, 0.45, 0.65]),
                GpuVec4::from_array([0.1, 0.15, 0.2, 0.25]),
                GpuVec4::from_array([0.5, -0.3, 0.2, 0.1]),
                GpuVec4::from_array([0.1, 0.2, -0.15, 0.05]),
                GpuVec4::from_array([0.4, 0.25, -0.1, 0.3]),
                GpuVec4::from_array([-0.2, 0.1, 0.05, -0.15]),
            )
        };
        let poison_all = |st: &mut KautzQuadState<GpuVec4>| {
            st.z_re_l = GpuVec4::from_array([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
            st.z_im_l =
                GpuVec4::from_array([f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::INFINITY]);
            st.z_re_r = GpuVec4::from_array([
                f32::NEG_INFINITY,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ]);
            st.z_im_r = GpuVec4::from_array([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        };
        let assert_state_finite = |st: &KautzQuadState<GpuVec4>| {
            for arr in [
                st.z_re_l.to_array(),
                st.z_im_l.to_array(),
                st.z_re_r.to_array(),
                st.z_im_r.to_array(),
            ] {
                assert!(arr.iter().all(|v| v.is_finite()));
            }
        };

        for (bad_l, bad_r) in [
            (f32::NAN, f32::INFINITY),
            (f32::INFINITY, f32::NEG_INFINITY),
            (f32::NEG_INFINITY, f32::NAN),
        ] {
            // Fully poisoned state + non-finite inputs scrub to exact +0.0 on poisoned & zero follow-up.
            let mut kv_zero = base_state();
            poison_all(&mut kv_zero);
            let mut ks_zero = kv_zero;
            let ov0 = step_kautz_quad_vec4(&mut kv_zero, bad_l, bad_r);
            let os0 =
                step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks_zero, bad_l, bad_r);
            assert_eq!(ov0.0.to_bits(), 0);
            assert_eq!(ov0.1.to_bits(), 0);
            assert_eq!(os0.0.to_bits(), 0);
            assert_eq!(os0.1.to_bits(), 0);
            assert_eq!(kv_zero, ks_zero);

            let ov1 = step_kautz_quad_vec4(&mut kv_zero, 0.0, 0.0);
            let os1 =
                step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks_zero, 0.0, 0.0);
            assert_eq!(ov1.0.to_bits(), 0);
            assert_eq!(ov1.1.to_bits(), 0);
            assert_eq!(os1.0.to_bits(), 0);
            assert_eq!(os1.1.to_bits(), 0);

            // Partially poisoned state + non-finite inputs scrub cleanly and recover on finite input.
            let mut kv = base_state();
            kv.z_re_l = GpuVec4::from_array([f32::NAN, 0.25, f32::INFINITY, f32::NEG_INFINITY]);
            kv.z_im_l = GpuVec4::from_array([f32::INFINITY, -0.1, f32::NAN, f32::NEG_INFINITY]);
            kv.z_re_r = GpuVec4::from_array([f32::NEG_INFINITY, f32::NAN, 0.2, f32::INFINITY]);
            kv.z_im_r = GpuVec4::from_array([0.05, f32::INFINITY, f32::NAN, f32::NEG_INFINITY]);
            let mut ks = kv;

            let out_v = step_kautz_quad_vec4(&mut kv, bad_l, bad_r);
            let out_s =
                step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks, bad_l, bad_r);
            assert!(out_v.0.is_finite() && out_v.1.is_finite());
            assert_eq!(out_v.0.to_bits(), out_s.0.to_bits());
            assert_eq!(out_v.1.to_bits(), out_s.1.to_bits());
            assert_state_finite(&kv);
            assert_eq!(kv, ks);

            let rec_v = step_kautz_quad_vec4(&mut kv, 0.5, -0.5);
            let rec_s = step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks, 0.5, -0.5);
            assert!(rec_v.0.is_finite() && rec_v.1.is_finite());
            assert!(rec_v.0.abs() > 0.0 && rec_v.1.abs() > 0.0);
            assert_eq!(rec_v.0.to_bits(), rec_s.0.to_bits());
            assert_eq!(rec_v.1.to_bits(), rec_s.1.to_bits());
            assert_state_finite(&kv);
            assert_eq!(kv, ks);
        }

        // Poisoned state with first finite sample (0.0, 0.0) scrubs directly to exact +0.0.
        let mut kv_fin = base_state();
        poison_all(&mut kv_fin);
        let mut ks_fin = kv_fin;
        let out_fv = step_kautz_quad_vec4(&mut kv_fin, 0.0, 0.0);
        let out_fs = step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks_fin, 0.0, 0.0);
        assert_eq!(out_fv.0.to_bits(), 0);
        assert_eq!(out_fv.1.to_bits(), 0);
        assert_eq!(out_fs.0.to_bits(), 0);
        assert_eq!(out_fs.1.to_bits(), 0);
        assert_eq!(kv_fin, ks_fin);
    }

    proptest! {
        #[test]
        fn prop_kautz_sweep_contraction_is_monotone_and_bounded(
            radius in 0.0_f32..1.2_f32,
            delta_theta in -6.0_f32..6.0_f32,
        ) {
            let r_eff = eval_kautz_sweep_contraction_f32(radius, delta_theta);
            prop_assert!(r_eff.is_finite());
            prop_assert!(r_eff <= 0.9995);
            prop_assert!(r_eff <= radius);
            if delta_theta.abs() > 0.05 {
                prop_assert!(r_eff <= eval_kautz_sweep_contraction_f32(radius, 0.0));
            }
        }

        #[test]
        fn prop_complex_modal_scan_associativity(
            xs in prop::array::uniform4(-1.0_f32..1.0_f32),
            r in 0.5_f32..0.99_f32,
            theta in -3.0_f32..3.0_f32,
        ) {
            let (ct, st) = algebraic_unit_rotor_f32(theta);
            let ar = r * ct;
            let ai = r * st;
            let b = 0.25_f32;
            let e0 = ComplexModalScanElem::from_step(ar, ai, b, xs[0]);
            let e1 = ComplexModalScanElem::from_step(ar, ai, b, xs[1]);
            let e2 = ComplexModalScanElem::from_step(ar, ai, b, xs[2]);
            let e3 = ComplexModalScanElem::from_step(ar, ai, b, xs[3]);
            let left = compose_complex_modal_scan(
                compose_complex_modal_scan(compose_complex_modal_scan(e0, e1), e2),
                e3,
            );
            let right = compose_complex_modal_scan(
                compose_complex_modal_scan(e0, e1),
                compose_complex_modal_scan(e2, e3),
            );
            prop_assert!((left.a_re - right.a_re).abs() < 1.0e-5);
            prop_assert!((left.a_im - right.a_im).abs() < 1.0e-5);
            prop_assert!((left.b_re - right.b_re).abs() < 1.0e-5);
            prop_assert!((left.b_im - right.b_im).abs() < 1.0e-5);
        }
    }
}
