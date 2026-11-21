//! WebAssembly `v128` `f64x2` trainer vector (`WasmF64Vec2`) and [`TrainerMathBackend`]
//! implementation for [`crate::math::WasmSimd128Backend`].

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::{
    f32x4, f64x2_abs, f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_pmax, f64x2_pmin,
    f64x2_promote_low_f32x4, f64x2_replace_lane, f64x2_splat, f64x2_sub, v128,
};

use crate::math::WasmSimd128Backend;
#[cfg(not(target_feature = "simd128"))]
use crate::math::trainer::ScalarF64Vec2;
use crate::math::trainer::{SimdF64Vec2, TrainerMathBackend};

/// WebAssembly 2-lane `f64` vector (`v128` when `simd128` is enabled, or `[f64; 2]` fallback).
#[cfg(target_feature = "simd128")]
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct WasmF64Vec2(pub v128);

/// Fallback 2-lane `f64` vector on `wasm32` when `simd128` is not enabled at compile time.
#[cfg(not(target_feature = "simd128"))]
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct WasmF64Vec2(pub [f64; 2]);

#[cfg(target_feature = "simd128")]
impl SimdF64Vec2 for WasmF64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        Self(f64x2_splat(val))
    }

    #[inline(always)]
    fn splat_pair(a: f64, b: f64) -> Self {
        Self(f64x2_replace_lane::<1>(f64x2_splat(a), b))
    }

    #[inline(always)]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        Self(f64x2_promote_low_f32x4(f32x4(a, b, 0.0, 0.0)))
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
        Self(f64x2_add(self.0, rhs.0))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(f64x2_sub(self.0, rhs.0))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(f64x2_mul(self.0, rhs.0))
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(f64x2_add(self.0, f64x2_mul(a.0, b.0)))
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 2) };
        let s = f64x2_splat(f64x2_extract_lane::<LANE>(v.0));
        Self(f64x2_add(self.0, f64x2_mul(a.0, s)))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(f64x2_sub(self.0, f64x2_mul(a.0, b.0)))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // `f64x2.pmax(a, b)` returns `a` unless `a < b`, so `(rhs, self)` clamps `NaN` in `self` to `rhs`.
        Self(f64x2_pmax(rhs.0, self.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // `f64x2.pmin(a, b)` returns `a` unless `b < a`, so `(rhs, self)` clamps `NaN` in `self` to `rhs`.
        Self(f64x2_pmin(rhs.0, self.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(f64x2_abs(self.0))
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        f64x2_extract_lane::<0>(self.0) + f64x2_extract_lane::<1>(self.0)
    }
}

#[cfg(not(target_feature = "simd128"))]
impl SimdF64Vec2 for WasmF64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        Self(ScalarF64Vec2::splat(val).0)
    }

    #[inline(always)]
    fn load(src: &[f64; 2]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f64; 2]) {
        *dst = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(ScalarF64Vec2(self.0).add(ScalarF64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(ScalarF64Vec2(self.0).sub(ScalarF64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(ScalarF64Vec2(self.0).mul(ScalarF64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(
            ScalarF64Vec2(self.0)
                .fma(ScalarF64Vec2(a.0), ScalarF64Vec2(b.0))
                .0,
        )
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(
            ScalarF64Vec2(self.0)
                .fms(ScalarF64Vec2(a.0), ScalarF64Vec2(b.0))
                .0,
        )
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(ScalarF64Vec2(self.0).max(ScalarF64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(ScalarF64Vec2(self.0).min(ScalarF64Vec2(rhs.0)).0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(ScalarF64Vec2(self.0).abs().0)
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        ScalarF64Vec2(self.0).horizontal_sum()
    }
}

impl TrainerMathBackend for WasmSimd128Backend {
    type F64Vec2 = WasmF64Vec2;
}
