//! x86-64 double-precision trainer SIMD kernels (`SSE2`, `SSE4.2`, and `AVX2+FMA` with strict
//! non-FMA `AVX` paths for `0-ULP` IEEE-754 contraction parity).

mod avx2_hw;

use core::arch::x86_64::{
    __m128d, _mm_add_pd, _mm_andnot_pd, _mm_cvtps_pd, _mm_cvtsd_f64, _mm_fmadd_pd, _mm_fnmadd_pd,
    _mm_max_pd, _mm_min_pd, _mm_mul_pd, _mm_set1_pd, _mm_setr_pd, _mm_setr_ps, _mm_shuffle_pd,
    _mm_sub_pd,
};

use avx2_hw::{
    avx_advance_bezout7_lp_dim_hw, avx_contract_tile_4x4_ieee_hw, avx_contract_tile_8x4_ieee_hw,
    avx_scale_f64_hw, avx_sub_f32_to_f64_hw, avx2_axpy_f64_hw, avx2_axpy_pair_f64_hw,
    avx2_dot_f64_hw, avx2_dot_pair_f64_hw, avx2_gram_sym_block4_hw, avx2_gram_tile_4x4_hw,
    avx2_sum_sq_f64_hw,
};

use super::avx2::X86Avx2FmaBackend;
use super::sse2::X86Sse2Backend;
use super::sse42::X86Sse42Backend;
use crate::math::trainer::{SimdF64Vec2, TrainerMathBackend};

/// 128-bit x86-64 SSE2 2-lane `f64` register wrapper (`__m128d`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Sse2F64Vec2(pub __m128d);

/// 128-bit x86-64 AVX2/FMA 2-lane `f64` register wrapper (`__m128d`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaF64Vec2(pub __m128d);

#[inline(always)]
fn sse2_f64x2_from_f32_pair(a: f32, b: f32) -> __m128d {
    // SAFETY: SSE2 is part of the x86-64 architectural baseline.
    unsafe { _mm_cvtps_pd(_mm_setr_ps(a, b, 0.0, 0.0)) }
}

impl SimdF64Vec2 for Sse2F64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_set1_pd(val)) }
    }

    #[inline(always)]
    fn splat_pair(a: f64, b: f64) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_setr_pd(a, b)) }
    }

    #[inline(always)]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        Self(sse2_f64x2_from_f32_pair(a, b))
    }

    #[inline(always)]
    fn load(src: &[f64; 2]) -> Self {
        Self(zerocopy::transmute!(*src))
    }

    #[inline(always)]
    fn store(self, dst: &mut [f64; 2]) {
        *dst = zerocopy::transmute!(self.0);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_add_pd(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_sub_pd(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_mul_pd(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        self.add(a.mul(b))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        self.sub(a.mul(b))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_max_pd(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_min_pd(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe { Self(_mm_andnot_pd(_mm_set1_pd(-0.0_f64), self.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        // SAFETY: SSE2 is guaranteed on x86-64.
        unsafe {
            let hi = _mm_shuffle_pd::<1>(self.0, self.0);
            _mm_cvtsd_f64(_mm_add_pd(self.0, hi))
        }
    }
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_fmadd_f64(acc: __m128d, a: __m128d, b: __m128d) -> __m128d {
    _mm_fmadd_pd(a, b, acc)
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_fnmadd_f64(acc: __m128d, a: __m128d, b: __m128d) -> __m128d {
    _mm_fnmadd_pd(a, b, acc)
}

impl SimdF64Vec2 for Avx2FmaF64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        Self(Sse2F64Vec2::splat(val).0)
    }

    #[inline(always)]
    fn splat_pair(a: f64, b: f64) -> Self {
        Self(Sse2F64Vec2::splat_pair(a, b).0)
    }

    #[inline(always)]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        Self(sse2_f64x2_from_f32_pair(a, b))
    }

    #[inline(always)]
    fn load(src: &[f64; 2]) -> Self {
        Self(Sse2F64Vec2::load(src).0)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f64; 2]) {
        Sse2F64Vec2(self.0).store(dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(Sse2F64Vec2(self.0).add(Sse2F64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(Sse2F64Vec2(self.0).sub(Sse2F64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(Sse2F64Vec2(self.0).mul(Sse2F64Vec2(rhs.0)).0)
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { Self(avx2_fmadd_f64(self.0, a.0, b.0)) }
        } else {
            Self(
                Sse2F64Vec2(self.0)
                    .fma(Sse2F64Vec2(a.0), Sse2F64Vec2(b.0))
                    .0,
            )
        }
    }

    #[inline(always)]
    fn mul_add_ieee(self, a: Self, b: Self) -> Self {
        // SAFETY: SSE2 baseline `mul` + `add` guarantees strict non-FMA IEEE-754 rounding.
        unsafe { Self(_mm_add_pd(self.0, _mm_mul_pd(a.0, b.0))) }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { Self(avx2_fnmadd_f64(self.0, a.0, b.0)) }
        } else {
            Self(
                Sse2F64Vec2(self.0)
                    .fms(Sse2F64Vec2(a.0), Sse2F64Vec2(b.0))
                    .0,
            )
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(Sse2F64Vec2(self.0).max(Sse2F64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(Sse2F64Vec2(self.0).min(Sse2F64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(Sse2F64Vec2(self.0).abs().0)
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        Sse2F64Vec2(self.0).horizontal_sum()
    }
}

impl TrainerMathBackend for X86Sse2Backend {
    type F64Vec2 = Sse2F64Vec2;
}

impl TrainerMathBackend for X86Sse42Backend {
    type F64Vec2 = Sse2F64Vec2;
}

impl TrainerMathBackend for X86Avx2FmaBackend {
    type F64Vec2 = Avx2FmaF64Vec2;

    #[inline]
    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_dot_f64_hw(a, b) }
        } else {
            X86Sse2Backend::dot_f64(a, b)
        }
    }

    #[inline]
    fn dot_pair_f64(x: &[f64], b0: &[f64], b1: &[f64]) -> (f64, f64) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_dot_pair_f64_hw(x, b0, b1) }
        } else {
            X86Sse2Backend::dot_pair_f64(x, b0, b1)
        }
    }

    #[inline]
    fn sum_sq_f64(x: &[f64]) -> f64 {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_sum_sq_f64_hw(x) }
        } else {
            X86Sse2Backend::sum_sq_f64(x)
        }
    }

    #[inline]
    fn axpy_f64(y: &mut [f64], alpha: f64, x: &[f64]) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_axpy_f64_hw(y, alpha, x) }
        } else {
            X86Sse2Backend::axpy_f64(y, alpha, x);
        }
    }

    #[inline]
    fn axpy_pair_f64(y: &mut [f64], a0: f64, x0: &[f64], a1: f64, x1: &[f64]) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_axpy_pair_f64_hw(y, a0, x0, a1, x1) }
        } else {
            X86Sse2Backend::axpy_pair_f64(y, a0, x0, a1, x1);
        }
    }

    #[inline]
    fn scale_f64(x: &mut [f64], scale: f64) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx_scale_f64_hw(x, scale) }
        } else {
            X86Sse2Backend::scale_f64(x, scale);
        }
    }

    #[inline]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx_sub_f32_to_f64_hw(dst, a, b) }
        } else {
            X86Sse2Backend::sub_f32_to_f64(dst, a, b);
        }
    }

    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_gram_tile_4x4_hw(cols_i, cols_j) }
        } else {
            X86Sse2Backend::gram_tile_4x4_f64(cols_i, cols_j)
        }
    }

    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx2_gram_sym_block4_hw(cols) }
        } else {
            X86Sse2Backend::gram_sym_block4_f64(cols)
        }
    }

    #[inline]
    fn contract_tile_4x4_ieee_f64(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4] {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx_contract_tile_4x4_ieee_hw(f_i, wf_j) }
        } else {
            X86Sse2Backend::contract_tile_4x4_ieee_f64(f_i, wf_j)
        }
    }

    #[inline]
    fn contract_tile_8x4_ieee_f64(
        f_i0: &[[f64; 4]],
        f_i1: &[[f64; 4]],
        wf_j: &[[f64; 4]],
    ) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx_contract_tile_8x4_ieee_hw(f_i0, f_i1, wf_j) }
        } else {
            X86Sse2Backend::contract_tile_8x4_ieee_f64(f_i0, f_i1, wf_j)
        }
    }

    #[inline]
    fn advance_bezout7_lp_dim_f64(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    ) {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { avx_advance_bezout7_lp_dim_hw(fb, dim, r, omr, u) }
        } else {
            X86Sse2Backend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::trainer::verify_trainer_backend_parity;

    #[test]
    fn test_x86_trainer_math_backends_parity() {
        crate::math::warm_simd_feature_caches();
        verify_trainer_backend_parity::<X86Sse2Backend>();
        verify_trainer_backend_parity::<X86Sse42Backend>();
        verify_trainer_backend_parity::<X86Avx2FmaBackend>();
    }
}
