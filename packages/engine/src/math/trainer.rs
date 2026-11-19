//! Double-precision (`f64`) 2-lane SIMD register abstraction (`SimdF64Vec2`) and
//! high-throughput trainer math primitives (`TrainerMathBackend`).
//!
//! Extends [`MathBackend`] so `dnam-capture` system identification (`VarPro`, 24-band Gabor
//! overlap-add, `Wide192Accum` Gram tiling, 3-band Bézout filtering, and `160`-pole Kautz
//! OMP/adjoint kernels) dispatches cleanly across `ARM SME`, `ARM SVE2`, `ARM NEON`,
//! `x86 AVX2+FMA`, `x86 SSE4.2`, `x86 SSE2`, `WASM SIMD128`, and `Scalar` backends.

pub mod f64_vec2;

#[cfg(test)]
use f64_vec2::verify_simd_f64_vec2;
pub use f64_vec2::{ScalarF64Vec2, SimdF64Vec2};
pub(crate) use f64_vec2::{contract_tile_4x4_vec2_ieee, contract_tile_8x4_vec2_ieee};
use f64_vec2::{load_vec2_quad, store_vec2_quad};

use super::{BackendKind, MathBackend, ScalarBackend};

/// Level-2 double-precision trainer hardware abstraction extending [`MathBackend`].
///
/// All default implementations operate in unrolled `Self::F64Vec2` SIMD registers, while
/// platform backends (`NeonBackend`, `ArmSmeBackend`, `X86Avx2FmaBackend`) override critical
/// wide kernels with target-specific intrinsics.
pub trait TrainerMathBackend: MathBackend {
    /// Underlying 128-bit 2-lane `f64` SIMD vector register type.
    type F64Vec2: SimdF64Vec2;

    /// Evaluates the 8-way SIMD-accumulated `f64` dot product of `a` and `b`.
    #[inline]
    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        let (ac, ar) = a[..n].as_chunks::<8>();
        let (bc, br) = b[..n].as_chunks::<8>();
        let mut acc = [Self::F64Vec2::zero(); 4];
        for (xc, yc) in ac.iter().zip(bc) {
            let x = load_vec2_quad::<Self::F64Vec2>(xc);
            let y = load_vec2_quad::<Self::F64Vec2>(yc);
            acc[0] = acc[0].fma(x[0], y[0]);
            acc[1] = acc[1].fma(x[1], y[1]);
            acc[2] = acc[2].fma(x[2], y[2]);
            acc[3] = acc[3].fma(x[3], y[3]);
        }
        let mut sum = acc[0].add(acc[1]).add(acc[2].add(acc[3])).horizontal_sum();
        for (&x, &y) in ar.iter().zip(br) {
            sum += x * y;
        }
        sum
    }

    /// Evaluates `(dot(x, b0), dot(x, b1))` in a single memory-read pass over `x`.
    #[inline]
    fn dot_pair_f64(x: &[f64], b0: &[f64], b1: &[f64]) -> (f64, f64) {
        let n = x.len().min(b0.len()).min(b1.len());
        let (xc, xr) = x[..n].as_chunks::<8>();
        let (c0, r0) = b0[..n].as_chunks::<8>();
        let (c1, r1) = b1[..n].as_chunks::<8>();
        let mut a0 = [Self::F64Vec2::zero(); 4];
        let mut a1 = [Self::F64Vec2::zero(); 4];
        for ((xv, y0), y1) in xc.iter().zip(c0).zip(c1) {
            let vx = load_vec2_quad::<Self::F64Vec2>(xv);
            let v0 = load_vec2_quad::<Self::F64Vec2>(y0);
            let v1 = load_vec2_quad::<Self::F64Vec2>(y1);
            a0[0] = a0[0].fma(vx[0], v0[0]);
            a1[0] = a1[0].fma(vx[0], v1[0]);
            a0[1] = a0[1].fma(vx[1], v0[1]);
            a1[1] = a1[1].fma(vx[1], v1[1]);
            a0[2] = a0[2].fma(vx[2], v0[2]);
            a1[2] = a1[2].fma(vx[2], v1[2]);
            a0[3] = a0[3].fma(vx[3], v0[3]);
            a1[3] = a1[3].fma(vx[3], v1[3]);
        }
        let red0 = a0[0].add(a0[1]).add(a0[2].add(a0[3]));
        let red1 = a1[0].add(a1[1]).add(a1[2].add(a1[3]));
        let (mut s0, mut s1) = red0.horizontal_sum_pair(red1);
        for ((&xv, &y0), &y1) in xr.iter().zip(r0).zip(r1) {
            s0 += xv * y0;
            s1 += xv * y1;
        }
        (s0, s1)
    }

    /// Evaluates the sum of squares `sum(x[i]^2)` across `x`.
    #[inline]
    fn sum_sq_f64(x: &[f64]) -> f64 {
        let (xc, xr) = x.as_chunks::<8>();
        let mut acc = [Self::F64Vec2::zero(); 4];
        for c in xc {
            let v = load_vec2_quad::<Self::F64Vec2>(c);
            acc[0] = acc[0].fma(v[0], v[0]);
            acc[1] = acc[1].fma(v[1], v[1]);
            acc[2] = acc[2].fma(v[2], v[2]);
            acc[3] = acc[3].fma(v[3], v[3]);
        }
        let mut sum = acc[0].add(acc[1]).add(acc[2].add(acc[3])).horizontal_sum();
        for &v in xr {
            sum += v * v;
        }
        sum
    }

    /// Updates `y[i] += alpha * x[i]` over `0..min(y.len(), x.len())`.
    #[inline]
    fn axpy_f64(y: &mut [f64], alpha: f64, x: &[f64]) {
        let n = y.len().min(x.len());
        let (yc, yr) = y[..n].as_chunks_mut::<8>();
        let (xc, xr) = x[..n].as_chunks::<8>();
        let av = Self::F64Vec2::splat(alpha);
        for (yd, xs) in yc.iter_mut().zip(xc) {
            let yv = load_vec2_quad::<Self::F64Vec2>(yd);
            let xv = load_vec2_quad::<Self::F64Vec2>(xs);
            store_vec2_quad(
                yd,
                [
                    yv[0].fma(av, xv[0]),
                    yv[1].fma(av, xv[1]),
                    yv[2].fma(av, xv[2]),
                    yv[3].fma(av, xv[3]),
                ],
            );
        }
        for (yd, &xs) in yr.iter_mut().zip(xr) {
            *yd += alpha * xs;
        }
    }

    /// Updates `y[i] += a0 * x0[i] + a1 * x1[i]` in a single pass over `y`.
    #[inline]
    fn axpy_pair_f64(y: &mut [f64], a0: f64, x0: &[f64], a1: f64, x1: &[f64]) {
        let n = y.len().min(x0.len()).min(x1.len());
        let (yc, yr) = y[..n].as_chunks_mut::<8>();
        let (c0, r0) = x0[..n].as_chunks::<8>();
        let (c1, r1) = x1[..n].as_chunks::<8>();
        let v_a0 = Self::F64Vec2::splat(a0);
        let v_a1 = Self::F64Vec2::splat(a1);
        for ((yd, s0), s1) in yc.iter_mut().zip(c0).zip(c1) {
            let yv = load_vec2_quad::<Self::F64Vec2>(yd);
            let x0v = load_vec2_quad::<Self::F64Vec2>(s0);
            let x1v = load_vec2_quad::<Self::F64Vec2>(s1);
            store_vec2_quad(
                yd,
                [
                    yv[0].fma(v_a0, x0v[0]).fma(v_a1, x1v[0]),
                    yv[1].fma(v_a0, x0v[1]).fma(v_a1, x1v[1]),
                    yv[2].fma(v_a0, x0v[2]).fma(v_a1, x1v[2]),
                    yv[3].fma(v_a0, x0v[3]).fma(v_a1, x1v[3]),
                ],
            );
        }
        for ((yd, &s0), &s1) in yr.iter_mut().zip(r0).zip(r1) {
            *yd += a0 * s0 + a1 * s1;
        }
    }

    /// Scales every element of `x` in place (`x[i] *= scale`).
    #[inline]
    fn scale_f64(x: &mut [f64], scale: f64) {
        let (xc, xr) = x.as_chunks_mut::<8>();
        let sv = Self::F64Vec2::splat(scale);
        for c in xc {
            let v = load_vec2_quad::<Self::F64Vec2>(c);
            store_vec2_quad(c, [v[0].mul(sv), v[1].mul(sv), v[2].mul(sv), v[3].mul(sv)]);
        }
        for v in xr {
            *v *= scale;
        }
    }

    /// Computes `dst[i] = f64::from(a[i]) - f64::from(b[i])` across `0..min(dst.len(), a.len(), b.len())`.
    #[inline(always)]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        let n = dst.len().min(a.len()).min(b.len());
        let (dc, dr) = dst[..n].as_chunks_mut::<4>();
        let (ac, ar) = a[..n].as_chunks::<4>();
        let (bc, br) = b[..n].as_chunks::<4>();
        for ((d4, a4), b4) in dc.iter_mut().zip(ac).zip(bc) {
            let d0 = Self::F64Vec2::from_f32_pair(a4[0], a4[1])
                .sub(Self::F64Vec2::from_f32_pair(b4[0], b4[1]))
                .to_array();
            let d1 = Self::F64Vec2::from_f32_pair(a4[2], a4[3])
                .sub(Self::F64Vec2::from_f32_pair(b4[2], b4[3]))
                .to_array();
            *d4 = [d0[0], d0[1], d1[0], d1[1]];
        }
        for ((d, &av), &bv) in dr.iter_mut().zip(ar).zip(br) {
            *d = f64::from(av) - f64::from(bv);
        }
    }

    /// Contracts an off-diagonal `4 x 4` block of inner products `out[r][c] = <cols_i[r], cols_j[c]>`
    /// in a single cache-friendly pass over the common slice prefix.
    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        let mut n = cols_i[0].len().min(cols_j[0].len());
        for k in 1..4 {
            n = n.min(cols_i[k].len()).min(cols_j[k].len());
        }
        let pairs = n / 2;
        let mut acc = [[Self::F64Vec2::zero(); 4]; 4];
        let mut t = 0usize;
        for _ in 0..pairs {
            let ci = [
                Self::F64Vec2::load(&[cols_i[0][t], cols_i[0][t + 1]]),
                Self::F64Vec2::load(&[cols_i[1][t], cols_i[1][t + 1]]),
                Self::F64Vec2::load(&[cols_i[2][t], cols_i[2][t + 1]]),
                Self::F64Vec2::load(&[cols_i[3][t], cols_i[3][t + 1]]),
            ];
            let cj = [
                Self::F64Vec2::load(&[cols_j[0][t], cols_j[0][t + 1]]),
                Self::F64Vec2::load(&[cols_j[1][t], cols_j[1][t + 1]]),
                Self::F64Vec2::load(&[cols_j[2][t], cols_j[2][t + 1]]),
                Self::F64Vec2::load(&[cols_j[3][t], cols_j[3][t + 1]]),
            ];
            for r in 0..4 {
                acc[r][0] = acc[r][0].fma(ci[r], cj[0]);
                acc[r][1] = acc[r][1].fma(ci[r], cj[1]);
                acc[r][2] = acc[r][2].fma(ci[r], cj[2]);
                acc[r][3] = acc[r][3].fma(ci[r], cj[3]);
            }
            t += 2;
        }
        let mut out = [[0.0_f64; 4]; 4];
        for r in 0..4 {
            let (s0, s1) = acc[r][0].horizontal_sum_pair(acc[r][1]);
            let (s2, s3) = acc[r][2].horizontal_sum_pair(acc[r][3]);
            out[r] = [s0, s1, s2, s3];
        }
        if t < n {
            for r in 0..4 {
                let xi = cols_i[r][t];
                out[r][0] += xi * cols_j[0][t];
                out[r][1] += xi * cols_j[1][t];
                out[r][2] += xi * cols_j[2][t];
                out[r][3] += xi * cols_j[3][t];
            }
        }
        out
    }

    /// Contracts the packed upper-triangular `4 x 4` diagonal Gram block
    /// `[g00, g01, g11, g02, g12, g22, g03, g13, g23, g33]` in a single pass.
    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        let n = cols[0]
            .len()
            .min(cols[1].len())
            .min(cols[2].len())
            .min(cols[3].len());
        let pairs = n / 2;
        let mut a = [Self::F64Vec2::zero(); 10];
        let mut t = 0usize;
        for _ in 0..pairs {
            let c0 = Self::F64Vec2::load(&[cols[0][t], cols[0][t + 1]]);
            let c1 = Self::F64Vec2::load(&[cols[1][t], cols[1][t + 1]]);
            let c2 = Self::F64Vec2::load(&[cols[2][t], cols[2][t + 1]]);
            let c3 = Self::F64Vec2::load(&[cols[3][t], cols[3][t + 1]]);
            a[0] = a[0].fma(c0, c0);
            a[1] = a[1].fma(c0, c1);
            a[2] = a[2].fma(c1, c1);
            a[3] = a[3].fma(c0, c2);
            a[4] = a[4].fma(c1, c2);
            a[5] = a[5].fma(c2, c2);
            a[6] = a[6].fma(c0, c3);
            a[7] = a[7].fma(c1, c3);
            a[8] = a[8].fma(c2, c3);
            a[9] = a[9].fma(c3, c3);
            t += 2;
        }
        let (s0, s1) = a[0].horizontal_sum_pair(a[1]);
        let (s2, s3) = a[2].horizontal_sum_pair(a[3]);
        let (s4, s5) = a[4].horizontal_sum_pair(a[5]);
        let (s6, s7) = a[6].horizontal_sum_pair(a[7]);
        let (s8, s9) = a[8].horizontal_sum_pair(a[9]);
        let mut out = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9];
        if t < n {
            let (x0, x1, x2, x3) = (cols[0][t], cols[1][t], cols[2][t], cols[3][t]);
            out[0] += x0 * x0;
            out[1] += x0 * x1;
            out[2] += x1 * x1;
            out[3] += x0 * x2;
            out[4] += x1 * x2;
            out[5] += x2 * x2;
            out[6] += x0 * x3;
            out[7] += x1 * x3;
            out[8] += x2 * x3;
            out[9] += x3 * x3;
        }
        out
    }

    /// Strict IEEE-754 non-fused `[4 x 4]` `Wide192Accum` tile contraction (`0-ULP` parity).
    #[inline]
    fn contract_tile_4x4_ieee_f64(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4] {
        contract_tile_4x4_vec2_ieee::<Self::F64Vec2>(f_i, wf_j)
    }

    /// Strict IEEE-754 non-fused `[8 x 4]` `Wide192Accum` tile contraction (`0-ULP` parity).
    #[inline]
    fn contract_tile_8x4_ieee_f64(
        f_i0: &[[f64; 4]],
        f_i1: &[[f64; 4]],
        wf_j: &[[f64; 4]],
    ) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
        contract_tile_8x4_vec2_ieee::<Self::F64Vec2>(f_i0, f_i1, wf_j)
    }

    /// Steps 7 cascaded 1-pole Bézout lowpass stages across `dim` (`<= 256`) features using
    /// strict IEEE-754 non-fused `(omr * prev) + (r * st)` updates (`0-ULP` bit-identical).
    #[inline]
    fn advance_bezout7_lp_dim_f64(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    ) {
        let bound = dim.min(256);
        let chunks = bound / 4;
        let rv = Self::F64Vec2::splat(r);
        let omrv = Self::F64Vec2::splat(omr);
        for c in 0..chunks {
            let d = c * 4;
            let mut prev0 = Self::F64Vec2::from_f32_pair(fb[d], fb[d + 1]);
            let mut prev1 = Self::F64Vec2::from_f32_pair(fb[d + 2], fb[d + 3]);
            for st in u.iter_mut() {
                let s0 = Self::F64Vec2::load(&[st[d], st[d + 1]]);
                let s1 = Self::F64Vec2::load(&[st[d + 2], st[d + 3]]);
                prev0 = omrv.mul(prev0).add(rv.mul(s0));
                prev1 = omrv.mul(prev1).add(rv.mul(s1));
                let out0 = prev0.to_array();
                let out1 = prev1.to_array();
                st[d] = out0[0];
                st[d + 1] = out0[1];
                st[d + 2] = out1[0];
                st[d + 3] = out1[1];
            }
        }
        for d in (chunks * 4)..bound {
            let mut prev = f64::from(fb[d]);
            for st in u.iter_mut() {
                prev = omr * prev + r * st[d];
                st[d] = prev;
            }
        }
    }

    /// Reads out degree-7 Bézout 3-band `[LF, MF, HF]` features (`f32`) from cascaded filter states.
    #[inline]
    fn readout_bezout7_bands_dim_f64(
        fb: &[f32; 256],
        dim: usize,
        u1: &[[f64; 256]; 7],
        u2: &[[f64; 256]; 7],
        out_lf: &mut [f32; 256],
        out_mf: &mut [f32; 256],
        out_hf: &mut [f32; 256],
    ) {
        let bound = dim.min(256);
        let c35 = Self::F64Vec2::splat(35.0);
        let c84 = Self::F64Vec2::splat(84.0);
        let c70 = Self::F64Vec2::splat(70.0);
        let c20 = Self::F64Vec2::splat(20.0);
        let chunks = bound / 2;
        for c in 0..chunks {
            let d = c * 2;
            let xv = Self::F64Vec2::from_f32_pair(fb[d], fb[d + 1]);
            let l1 = c35
                .mul(Self::F64Vec2::load(&[u1[3][d], u1[3][d + 1]]))
                .sub(c84.mul(Self::F64Vec2::load(&[u1[4][d], u1[4][d + 1]])))
                .add(c70.mul(Self::F64Vec2::load(&[u1[5][d], u1[5][d + 1]])))
                .sub(c20.mul(Self::F64Vec2::load(&[u1[6][d], u1[6][d + 1]])))
                .to_array();
            let l2 = c35
                .mul(Self::F64Vec2::load(&[u2[3][d], u2[3][d + 1]]))
                .sub(c84.mul(Self::F64Vec2::load(&[u2[4][d], u2[4][d + 1]])))
                .add(c70.mul(Self::F64Vec2::load(&[u2[5][d], u2[5][d + 1]])))
                .sub(c20.mul(Self::F64Vec2::load(&[u2[6][d], u2[6][d + 1]])))
                .to_array();
            let xa = xv.to_array();
            out_lf[d] = l1[0] as f32;
            out_lf[d + 1] = l1[1] as f32;
            out_mf[d] = (l2[0] - l1[0]) as f32;
            out_mf[d + 1] = (l2[1] - l1[1]) as f32;
            out_hf[d] = (xa[0] - l2[0]) as f32;
            out_hf[d + 1] = (xa[1] - l2[1]) as f32;
        }
        if (bound & 1) != 0 {
            let d = bound - 1;
            let x = f64::from(fb[d]);
            let l250 = 35.0 * u1[3][d] - 84.0 * u1[4][d] + 70.0 * u1[5][d] - 20.0 * u1[6][d];
            let l2500 = 35.0 * u2[3][d] - 84.0 * u2[4][d] + 70.0 * u2[5][d] - 20.0 * u2[6][d];
            out_lf[d] = l250 as f32;
            out_mf[d] = (l2500 - l250) as f32;
            out_hf[d] = (x - l2500) as f32;
        }
    }

    /// Returns the runtime [`BackendKind`] associated with this trainer math backend.
    #[inline(always)]
    fn trainer_backend_kind() -> BackendKind {
        Self::KIND
    }
}

impl TrainerMathBackend for ScalarBackend {
    type F64Vec2 = ScalarF64Vec2;
}

#[cfg(test)]
pub(crate) use self::tests::verify_trainer_backend_parity;

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use proptest::prelude::*;

    use super::*;

    pub(crate) fn verify_trainer_backend_parity<B: TrainerMathBackend>() {
        verify_simd_f64_vec2::<B::F64Vec2>();
        assert_eq!(B::trainer_backend_kind(), B::KIND);
        let approx = |u: &[f64], v: &[f64]| u.iter().zip(v).all(|(x, y)| (x - y).abs() <= 1.0e-11);
        let bits_eq =
            |u: &[f64], v: &[f64]| u.iter().zip(v).all(|(x, y)| x.to_bits() == y.to_bits());
        for len in [0usize, 1, 3, 8, 17] {
            let a: Vec<f64> = (0..len)
                .map(|i| 0.31 * ((i as f64) * 0.17).sin() - 0.11 * ((i % 5) as f64))
                .collect();
            let b: Vec<f64> = (0..len)
                .map(|i| -0.27 * ((i as f64) * 0.23).cos() + 0.09)
                .collect();
            let c: Vec<f64> = (0..len)
                .map(|i| 0.43 * ((i as f64) * 0.11).sin() + 0.05)
                .collect();
            assert!((ScalarBackend::dot_f64(&a, &b) - B::dot_f64(&a, &b)).abs() <= 1.0e-11);
            let ((s0, s1), (b0, b1)) = (
                ScalarBackend::dot_pair_f64(&a, &b, &c),
                B::dot_pair_f64(&a, &b, &c),
            );
            assert!((s0 - b0).abs() <= 1.0e-11 && (s1 - b1).abs() <= 1.0e-11);
            assert!((ScalarBackend::sum_sq_f64(&a) - B::sum_sq_f64(&a)).abs() <= 1.0e-11);

            let (mut ys, mut yb) = (a.clone(), a.clone());
            ScalarBackend::axpy_f64(&mut ys, 0.75, &b);
            B::axpy_f64(&mut yb, 0.75, &b);
            assert!(approx(&ys, &yb));

            let (mut yps, mut ypb) = (a.clone(), a.clone());
            ScalarBackend::axpy_pair_f64(&mut yps, 0.65, &b, -0.45, &c);
            B::axpy_pair_f64(&mut ypb, 0.65, &b, -0.45, &c);
            assert!(approx(&yps, &ypb));

            ScalarBackend::scale_f64(&mut ys, 1.25);
            B::scale_f64(&mut yb, 1.25);
            assert!(approx(&ys, &yb));

            let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
            let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
            let (mut ds, mut db) = (vec![0.0; len], vec![0.0; len]);
            ScalarBackend::sub_f32_to_f64(&mut ds, &a32, &b32);
            B::sub_f32_to_f64(&mut db, &a32, &b32);
            assert!(bits_eq(&ds, &db));
        }

        let cols_i: [Vec<f64>; 4] = core::array::from_fn(|r| {
            (0..17)
                .map(|t| (((t + r * 5) as f64) * 0.13).sin())
                .collect()
        });
        let cols_j: [Vec<f64>; 4] = core::array::from_fn(|c| {
            (0..17)
                .map(|t| (((t + c * 7) as f64) * 0.19).cos())
                .collect()
        });
        let ci = [&cols_i[0][..], &cols_i[1], &cols_i[2], &cols_i[3]];
        let cj = [&cols_j[0][..], &cols_j[1], &cols_j[2], &cols_j[3]];
        let (gs, gb) = (
            ScalarBackend::gram_tile_4x4_f64(ci, cj),
            B::gram_tile_4x4_f64(ci, cj),
        );
        for r in 0..4 {
            assert!(approx(&gs[r], &gb[r]));
        }
        assert!(approx(
            &ScalarBackend::gram_sym_block4_f64(ci),
            &B::gram_sym_block4_f64(ci)
        ));

        let fi0: Vec<[f64; 4]> = (0..15).map(|k| [k as f64 * 0.1, -0.3, 0.7, 0.2]).collect();
        let fi1: Vec<[f64; 4]> = (0..15)
            .map(|k| [-0.4, k as f64 * 0.05, 0.11, -0.9])
            .collect();
        let wfj: Vec<[f64; 4]> = (0..15)
            .map(|k| [0.5, -0.25, 0.125 + k as f64 * 0.03, -0.75])
            .collect();
        let (t4s, t4b) = (
            ScalarBackend::contract_tile_4x4_ieee_f64(&fi0, &wfj),
            B::contract_tile_4x4_ieee_f64(&fi0, &wfj),
        );
        let ((t8as, t8bs), (t8ab, t8bb)) = (
            ScalarBackend::contract_tile_8x4_ieee_f64(&fi0, &fi1, &wfj),
            B::contract_tile_8x4_ieee_f64(&fi0, &fi1, &wfj),
        );
        for lj in 0..4 {
            assert!(
                bits_eq(&t4s[lj], &t4b[lj])
                    && bits_eq(&t8as[lj], &t8ab[lj])
                    && bits_eq(&t8bs[lj], &t8bb[lj])
            );
        }

        let fb: [f32; 256] = core::array::from_fn(|idx| ((idx as f32) * 0.21).sin() * 0.4);
        let (mut u1s, mut u1b) = ([[0.0_f64; 256]; 7], [[0.0_f64; 256]; 7]);
        let (mut u2s, mut u2b) = ([[0.0_f64; 256]; 7], [[0.0_f64; 256]; 7]);
        for _ in 0..3 {
            ScalarBackend::advance_bezout7_lp_dim_f64(&fb, 65, 0.96, 0.04, &mut u1s);
            B::advance_bezout7_lp_dim_f64(&fb, 65, 0.96, 0.04, &mut u1b);
            ScalarBackend::advance_bezout7_lp_dim_f64(&fb, 65, 0.72, 0.28, &mut u2s);
            B::advance_bezout7_lp_dim_f64(&fb, 65, 0.72, 0.28, &mut u2b);
        }
        for st in 0..7 {
            assert!(
                bits_eq(&u1s[st][..65], &u1b[st][..65]) && bits_eq(&u2s[st][..65], &u2b[st][..65])
            );
        }
        let (mut lfs, mut mfs, mut hfs) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
        let (mut lfb, mut mfb, mut hfb) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
        ScalarBackend::readout_bezout7_bands_dim_f64(
            &fb, 65, &u1s, &u2s, &mut lfs, &mut mfs, &mut hfs,
        );
        B::readout_bezout7_bands_dim_f64(&fb, 65, &u1b, &u2b, &mut lfb, &mut mfb, &mut hfb);
        for d in 0..65 {
            assert!(
                lfs[d].to_bits() == lfb[d].to_bits()
                    && mfs[d].to_bits() == mfb[d].to_bits()
                    && hfs[d].to_bits() == hfb[d].to_bits()
            );
        }
    }

    #[test]
    fn test_trainer_math_scalar_parity() {
        verify_trainer_backend_parity::<ScalarBackend>();
    }

    proptest! {
        #[test]
        fn prop_trainer_dot_and_tile_ieee_parity(
            xs in prop::collection::vec(-10.0_f64..10.0_f64, 1..40),
            ys in prop::collection::vec(-10.0_f64..10.0_f64, 1..40),
        ) {
            let n = xs.len().min(ys.len());
            let s_dot = ScalarBackend::dot_f64(&xs[..n], &ys[..n]);
            let d_dot = crate::math::DefaultMathBackend::dot_f64(&xs[..n], &ys[..n]);
            prop_assert!((s_dot - d_dot).abs() <= 1.0e-11 * s_dot.abs().max(1.0));
            let frames = n / 4;
            if frames > 0 {
                let f_i: Vec<[f64; 4]> = (0..frames).map(|k| [xs[k * 4], xs[k * 4 + 1], xs[k * 4 + 2], xs[k * 4 + 3]]).collect();
                let wf_j: Vec<[f64; 4]> = (0..frames).map(|k| [ys[k * 4], ys[k * 4 + 1], ys[k * 4 + 2], ys[k * 4 + 3]]).collect();
                let t_s = ScalarBackend::contract_tile_4x4_ieee_f64(&f_i, &wf_j);
                let t_d = crate::math::DefaultMathBackend::contract_tile_4x4_ieee_f64(&f_i, &wf_j);
                for lj in 0..4 {
                    for li in 0..4 {
                        prop_assert_eq!(t_s[lj][li].to_bits(), t_d[lj][li].to_bits());
                    }
                }
            }
        }
    }
}
