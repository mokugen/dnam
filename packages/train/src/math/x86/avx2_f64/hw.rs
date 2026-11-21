//! 256-bit `AVX` / `AVX2+FMA` `f64` trainer primitives (`0-ULP` non-FMA tile contractions
//! and `FMADD` inner-product/Gram tiles) for [`super::super::avx2_f64`].

use core::arch::x86_64::{
    __m128, __m256d, _mm_add_pd, _mm_loadu_ps, _mm256_add_pd, _mm256_castpd256_pd128,
    _mm256_cvtps_pd, _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd,
    _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
};

use super::super::sse_f64::Sse2F64Vec2;
use crate::math::trainer::SimdF64Vec2;

#[target_feature(enable = "avx")]
#[inline]
fn hsum_pd4(v: __m256d) -> f64 {
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd::<1>(v);
    Sse2F64Vec2(_mm_add_pd(lo, hi)).horizontal_sum()
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_dot_f64_hw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (ac, ar) = a[..n].as_chunks::<16>();
    let (bc, br) = b[..n].as_chunks::<16>();
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    );
    for (xc, yc) in ac.iter().zip(bc) {
        // SAFETY: `xc` and `yc` are `[f64; 16]` (`4 x __m256d`).
        unsafe {
            let xp = xc.as_ptr();
            let yp = yc.as_ptr();
            acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(xp), _mm256_loadu_pd(yp), acc0);
            acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(4)), _mm256_loadu_pd(yp.add(4)), acc1);
            acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(xp.add(8)), _mm256_loadu_pd(yp.add(8)), acc2);
            acc3 = _mm256_fmadd_pd(
                _mm256_loadu_pd(xp.add(12)),
                _mm256_loadu_pd(yp.add(12)),
                acc3,
            );
        }
    }
    let mut sum = hsum_pd4(_mm256_add_pd(
        _mm256_add_pd(acc0, acc1),
        _mm256_add_pd(acc2, acc3),
    ));
    for (&x, &y) in ar.iter().zip(br) {
        sum += x * y;
    }
    sum
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_dot_pair_f64_hw(x: &[f64], b0: &[f64], b1: &[f64]) -> (f64, f64) {
    let n = x.len().min(b0.len()).min(b1.len());
    let (xc, xr) = x[..n].as_chunks::<8>();
    let (c0, r0) = b0[..n].as_chunks::<8>();
    let (c1, r1) = b1[..n].as_chunks::<8>();
    let (mut a00, mut a01, mut a10, mut a11) = (
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    );
    for ((xv, y0), y1) in xc.iter().zip(c0).zip(c1) {
        // SAFETY: `xv`, `y0`, `y1` are `[f64; 8]` (`2 x __m256d`).
        unsafe {
            let (vx0, vx1) = (
                _mm256_loadu_pd(xv.as_ptr()),
                _mm256_loadu_pd(xv.as_ptr().add(4)),
            );
            a00 = _mm256_fmadd_pd(vx0, _mm256_loadu_pd(y0.as_ptr()), a00);
            a01 = _mm256_fmadd_pd(vx1, _mm256_loadu_pd(y0.as_ptr().add(4)), a01);
            a10 = _mm256_fmadd_pd(vx0, _mm256_loadu_pd(y1.as_ptr()), a10);
            a11 = _mm256_fmadd_pd(vx1, _mm256_loadu_pd(y1.as_ptr().add(4)), a11);
        }
    }
    let (mut s0, mut s1) = (
        hsum_pd4(_mm256_add_pd(a00, a01)),
        hsum_pd4(_mm256_add_pd(a10, a11)),
    );
    for ((&xv, &y0), &y1) in xr.iter().zip(r0).zip(r1) {
        s0 += xv * y0;
        s1 += xv * y1;
    }
    (s0, s1)
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_sum_sq_f64_hw(x: &[f64]) -> f64 {
    let (xc, xr) = x.as_chunks::<16>();
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    );
    for c in xc {
        // SAFETY: `c` is `[f64; 16]` (`4 x __m256d`).
        unsafe {
            let p = c.as_ptr();
            let (v0, v1, v2, v3) = (
                _mm256_loadu_pd(p),
                _mm256_loadu_pd(p.add(4)),
                _mm256_loadu_pd(p.add(8)),
                _mm256_loadu_pd(p.add(12)),
            );
            acc0 = _mm256_fmadd_pd(v0, v0, acc0);
            acc1 = _mm256_fmadd_pd(v1, v1, acc1);
            acc2 = _mm256_fmadd_pd(v2, v2, acc2);
            acc3 = _mm256_fmadd_pd(v3, v3, acc3);
        }
    }
    let mut sum = hsum_pd4(_mm256_add_pd(
        _mm256_add_pd(acc0, acc1),
        _mm256_add_pd(acc2, acc3),
    ));
    for &v in xr {
        sum += v * v;
    }
    sum
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_axpy_f64_hw(y: &mut [f64], alpha: f64, x: &[f64]) {
    let n = y.len().min(x.len());
    let (yc, yr) = y[..n].as_chunks_mut::<8>();
    let (xc, xr) = x[..n].as_chunks::<8>();
    let av = _mm256_set1_pd(alpha);
    for (yd, xs) in yc.iter_mut().zip(xc) {
        // SAFETY: `yd` and `xs` are `[f64; 8]` (`2 x __m256d`).
        unsafe {
            let yp = yd.as_mut_ptr();
            let xp = xs.as_ptr();
            let r0 = _mm256_fmadd_pd(av, _mm256_loadu_pd(xp), _mm256_loadu_pd(yp));
            let r1 = _mm256_fmadd_pd(av, _mm256_loadu_pd(xp.add(4)), _mm256_loadu_pd(yp.add(4)));
            _mm256_storeu_pd(yp, r0);
            _mm256_storeu_pd(yp.add(4), r1);
        }
    }
    for (yd, &xs) in yr.iter_mut().zip(xr) {
        *yd += alpha * xs;
    }
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_axpy_pair_f64_hw(y: &mut [f64], a0: f64, x0: &[f64], a1: f64, x1: &[f64]) {
    let n = y.len().min(x0.len()).min(x1.len());
    let (yc, yr) = y[..n].as_chunks_mut::<8>();
    let (c0, r0) = x0[..n].as_chunks::<8>();
    let (c1, r1) = x1[..n].as_chunks::<8>();
    let (va0, va1) = (_mm256_set1_pd(a0), _mm256_set1_pd(a1));
    for ((yd, s0), s1) in yc.iter_mut().zip(c0).zip(c1) {
        // SAFETY: `yd`, `s0`, `s1` are `[f64; 8]` (`2 x __m256d`).
        unsafe {
            let yp = yd.as_mut_ptr();
            let (p0, p1) = (s0.as_ptr(), s1.as_ptr());
            let r_lo = _mm256_fmadd_pd(
                va1,
                _mm256_loadu_pd(p1),
                _mm256_fmadd_pd(va0, _mm256_loadu_pd(p0), _mm256_loadu_pd(yp)),
            );
            let r_hi = _mm256_fmadd_pd(
                va1,
                _mm256_loadu_pd(p1.add(4)),
                _mm256_fmadd_pd(va0, _mm256_loadu_pd(p0.add(4)), _mm256_loadu_pd(yp.add(4))),
            );
            _mm256_storeu_pd(yp, r_lo);
            _mm256_storeu_pd(yp.add(4), r_hi);
        }
    }
    for ((yd, &s0), &s1) in yr.iter_mut().zip(r0).zip(r1) {
        *yd += a0 * s0 + a1 * s1;
    }
}

#[target_feature(enable = "avx")]
#[inline]
pub(super) fn avx_scale_f64_hw(x: &mut [f64], scale: f64) {
    let (xc, xr) = x.as_chunks_mut::<8>();
    let sv = _mm256_set1_pd(scale);
    for c in xc {
        // SAFETY: `c` is `[f64; 8]` (`2 x __m256d`).
        unsafe {
            let p = c.as_mut_ptr();
            _mm256_storeu_pd(p, _mm256_mul_pd(_mm256_loadu_pd(p), sv));
            _mm256_storeu_pd(p.add(4), _mm256_mul_pd(_mm256_loadu_pd(p.add(4)), sv));
        }
    }
    for v in xr {
        *v *= scale;
    }
}

#[target_feature(enable = "avx")]
#[inline]
pub(super) fn avx_sub_f32_to_f64_hw(dst: &mut [f64], a: &[f32], b: &[f32]) {
    let n = dst.len().min(a.len()).min(b.len());
    let (dc, dr) = dst[..n].as_chunks_mut::<4>();
    let (ac, ar) = a[..n].as_chunks::<4>();
    let (bc, br) = b[..n].as_chunks::<4>();
    for ((d4, a4), b4) in dc.iter_mut().zip(ac).zip(bc) {
        // SAFETY: `a4`, `b4` are `[f32; 4]` and `d4` is `[f64; 4]`.
        unsafe {
            let va: __m128 = zerocopy::transmute!(*a4);
            let vb: __m128 = zerocopy::transmute!(*b4);
            _mm256_storeu_pd(
                d4.as_mut_ptr(),
                _mm256_sub_pd(_mm256_cvtps_pd(va), _mm256_cvtps_pd(vb)),
            );
        }
    }
    for ((d, &av), &bv) in dr.iter_mut().zip(ar).zip(br) {
        *d = f64::from(av) - f64::from(bv);
    }
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_gram_tile_4x4_hw(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
    let mut n = cols_i[0].len().min(cols_j[0].len());
    for k in 1..4 {
        n = n.min(cols_i[k].len()).min(cols_j[k].len());
    }
    let chunks = n / 4;
    let mut acc = [[_mm256_setzero_pd(); 4]; 4];
    for c in 0..chunks {
        let t = c * 4;
        // SAFETY: `t + 4 <= n` for all 8 column slices.
        unsafe {
            let ci = [
                _mm256_loadu_pd(cols_i[0].as_ptr().add(t)),
                _mm256_loadu_pd(cols_i[1].as_ptr().add(t)),
                _mm256_loadu_pd(cols_i[2].as_ptr().add(t)),
                _mm256_loadu_pd(cols_i[3].as_ptr().add(t)),
            ];
            let cj = [
                _mm256_loadu_pd(cols_j[0].as_ptr().add(t)),
                _mm256_loadu_pd(cols_j[1].as_ptr().add(t)),
                _mm256_loadu_pd(cols_j[2].as_ptr().add(t)),
                _mm256_loadu_pd(cols_j[3].as_ptr().add(t)),
            ];
            for r in 0..4 {
                acc[r][0] = _mm256_fmadd_pd(ci[r], cj[0], acc[r][0]);
                acc[r][1] = _mm256_fmadd_pd(ci[r], cj[1], acc[r][1]);
                acc[r][2] = _mm256_fmadd_pd(ci[r], cj[2], acc[r][2]);
                acc[r][3] = _mm256_fmadd_pd(ci[r], cj[3], acc[r][3]);
            }
        }
    }
    let mut out = [[0.0_f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = hsum_pd4(acc[r][c]);
        }
    }
    for t in (chunks * 4)..n {
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

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
pub(super) fn avx2_gram_sym_block4_hw(cols: [&[f64]; 4]) -> [f64; 10] {
    let n = cols[0]
        .len()
        .min(cols[1].len())
        .min(cols[2].len())
        .min(cols[3].len());
    let chunks = n / 4;
    let mut a = [_mm256_setzero_pd(); 10];
    for c in 0..chunks {
        let t = c * 4;
        // SAFETY: `t + 4 <= n` for all 4 column slices.
        unsafe {
            let c0 = _mm256_loadu_pd(cols[0].as_ptr().add(t));
            let c1 = _mm256_loadu_pd(cols[1].as_ptr().add(t));
            let c2 = _mm256_loadu_pd(cols[2].as_ptr().add(t));
            let c3 = _mm256_loadu_pd(cols[3].as_ptr().add(t));
            a[0] = _mm256_fmadd_pd(c0, c0, a[0]);
            a[1] = _mm256_fmadd_pd(c0, c1, a[1]);
            a[2] = _mm256_fmadd_pd(c1, c1, a[2]);
            a[3] = _mm256_fmadd_pd(c0, c2, a[3]);
            a[4] = _mm256_fmadd_pd(c1, c2, a[4]);
            a[5] = _mm256_fmadd_pd(c2, c2, a[5]);
            a[6] = _mm256_fmadd_pd(c0, c3, a[6]);
            a[7] = _mm256_fmadd_pd(c1, c3, a[7]);
            a[8] = _mm256_fmadd_pd(c2, c3, a[8]);
            a[9] = _mm256_fmadd_pd(c3, c3, a[9]);
        }
    }
    let mut out = [0.0_f64; 10];
    for k in 0..10 {
        out[k] = hsum_pd4(a[k]);
    }
    for t in (chunks * 4)..n {
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

/// Strictly non-FMA (`enable = "avx"`, without `fma`) 256-bit tile contraction (`0-ULP` parity).
#[target_feature(enable = "avx")]
#[inline]
pub(super) fn avx_contract_tile_4x4_ieee_hw(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4] {
    let m = f_i.len().min(wf_j.len());
    let (fi_pairs, fi_rem) = f_i[..m].as_chunks::<2>();
    let (wf_pairs, wf_rem) = wf_j[..m].as_chunks::<2>();
    let (mut a, mut b) = ([_mm256_setzero_pd(); 4], [_mm256_setzero_pd(); 4]);
    for ([fa, fb], [wa, wb]) in fi_pairs.iter().zip(wf_pairs) {
        // SAFETY: `fa`, `fb` are aligned/valid `[f64; 4]`.
        let (va, vb) = unsafe { (_mm256_loadu_pd(fa.as_ptr()), _mm256_loadu_pd(fb.as_ptr())) };
        for lj in 0..4 {
            a[lj] = _mm256_add_pd(a[lj], _mm256_mul_pd(va, _mm256_set1_pd(wa[lj])));
            b[lj] = _mm256_add_pd(b[lj], _mm256_mul_pd(vb, _mm256_set1_pd(wb[lj])));
        }
    }
    for (fa, wa) in fi_rem.iter().zip(wf_rem) {
        // SAFETY: `fa` is `[f64; 4]`.
        let va = unsafe { _mm256_loadu_pd(fa.as_ptr()) };
        for lj in 0..4 {
            a[lj] = _mm256_add_pd(a[lj], _mm256_mul_pd(va, _mm256_set1_pd(wa[lj])));
        }
    }
    let mut out = [[0.0_f64; 4]; 4];
    for lj in 0..4 {
        // SAFETY: `out[lj]` is `[f64; 4]`.
        unsafe { _mm256_storeu_pd(out[lj].as_mut_ptr(), _mm256_add_pd(a[lj], b[lj])) };
    }
    out
}

/// Strictly non-FMA (`enable = "avx"`, without `fma`) dual 256-bit tile contraction (`0-ULP` parity).
#[target_feature(enable = "avx")]
#[inline]
pub(super) fn avx_contract_tile_8x4_ieee_hw(
    f_i0: &[[f64; 4]],
    f_i1: &[[f64; 4]],
    wf_j: &[[f64; 4]],
) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let m = f_i0.len().min(f_i1.len()).min(wf_j.len());
    let (mut a, mut b) = ([_mm256_setzero_pd(); 4], [_mm256_setzero_pd(); 4]);
    for ((fa, fb), w) in f_i0[..m].iter().zip(&f_i1[..m]).zip(&wf_j[..m]) {
        // SAFETY: `fa`, `fb` are `[f64; 4]`.
        let (va, vb) = unsafe { (_mm256_loadu_pd(fa.as_ptr()), _mm256_loadu_pd(fb.as_ptr())) };
        for lj in 0..4 {
            let wv = _mm256_set1_pd(w[lj]);
            a[lj] = _mm256_add_pd(a[lj], _mm256_mul_pd(va, wv));
            b[lj] = _mm256_add_pd(b[lj], _mm256_mul_pd(vb, wv));
        }
    }
    let (mut out_a, mut out_b) = ([[0.0_f64; 4]; 4], [[0.0_f64; 4]; 4]);
    for lj in 0..4 {
        // SAFETY: `out_a[lj]` and `out_b[lj]` are `[f64; 4]`.
        unsafe {
            _mm256_storeu_pd(out_a[lj].as_mut_ptr(), a[lj]);
            _mm256_storeu_pd(out_b[lj].as_mut_ptr(), b[lj]);
        }
    }
    (out_a, out_b)
}

/// Strictly non-FMA (`enable = "avx"`, without `fma`) 4-wide Bézout cascade step (`0-ULP` parity).
#[target_feature(enable = "avx")]
#[inline]
pub(super) fn avx_advance_bezout7_lp_dim_hw(
    fb: &[f32; 256],
    dim: usize,
    r: f64,
    omr: f64,
    u: &mut [[f64; 256]; 7],
) {
    let bound = dim.min(256);
    let chunks = bound / 4;
    let (rv, omrv) = (_mm256_set1_pd(r), _mm256_set1_pd(omr));
    for c in 0..chunks {
        let d = c * 4;
        // SAFETY: `d + 4 <= bound <= 256`.
        let mut prev = unsafe { _mm256_cvtps_pd(_mm_loadu_ps(fb.as_ptr().add(d))) };
        for st in u.iter_mut() {
            // SAFETY: `d + 4 <= 256`.
            unsafe {
                let sv = _mm256_loadu_pd(st.as_ptr().add(d));
                prev = _mm256_add_pd(_mm256_mul_pd(omrv, prev), _mm256_mul_pd(rv, sv));
                _mm256_storeu_pd(st.as_mut_ptr().add(d), prev);
            }
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
