//! WebAssembly 128-bit `simd128` 4-lane `f32` SIMD vector ([`WasmVec4`]) and
//! [`WasmSimd128Backend`] (`#[forbid(unsafe_code)]`).

use crate::math::BackendKind;
use crate::math::MathBackend;
#[cfg(not(target_feature = "simd128"))]
use crate::math::SimdVec4;
use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};
#[cfg(not(target_feature = "simd128"))]
use crate::math::scalar::ScalarVec4;

/// Detects the active `wasm32` SIMD capability at compile time (`simd128` vs portable scalar).
#[inline]
pub const fn detect_wasm_backend() -> BackendKind {
    if cfg!(target_feature = "simd128") {
        BackendKind::WasmSimd128
    } else {
        BackendKind::Scalar
    }
}

#[cfg(target_feature = "simd128")]
mod simd128_impl {
    use core::arch::wasm32::{
        f32x4, f32x4_abs, f32x4_add, f32x4_extract_lane, f32x4_le, f32x4_max, f32x4_min, f32x4_mul,
        f32x4_neg, f32x4_splat, f32x4_sub, i32x4_shl, i32x4_splat, u32x4_add, u32x4_ge, u32x4_lt,
        u32x4_shr, u32x4_splat, u32x4_sub, v128, v128_and, v128_andnot, v128_bitselect, v128_xor,
    };
    use core::fmt;

    use crate::math::{
        IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_BIAS, IEEE754_F32_EXP_MASK, IEEE754_F32_IMPLICIT_BIT,
        IEEE754_F32_MANTISSA_BITS, IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED,
        IEEE754_F32_SIGN_MASK, SimdVec4,
    };

    /// WebAssembly `128-bit` `simd128` 4-lane `f32` vector register (`v128`).
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub struct WasmVec4(pub v128);

    impl fmt::Debug for WasmVec4 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("WasmVec4").field(&self.to_array()).finish()
        }
    }

    impl SimdVec4 for WasmVec4 {
        #[inline(always)]
        fn splat(val: f32) -> Self {
            Self(f32x4_splat(val))
        }

        #[inline(always)]
        fn load(src: &[f32; 4]) -> Self {
            Self(f32x4(src[0], src[1], src[2], src[3]))
        }

        #[inline(always)]
        fn store(self, dst: &mut [f32; 4]) {
            dst[0] = f32x4_extract_lane::<0>(self.0);
            dst[1] = f32x4_extract_lane::<1>(self.0);
            dst[2] = f32x4_extract_lane::<2>(self.0);
            dst[3] = f32x4_extract_lane::<3>(self.0);
        }

        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self(f32x4_add(self.0, rhs.0))
        }

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self(f32x4_sub(self.0, rhs.0))
        }

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self(f32x4_mul(self.0, rhs.0))
        }

        #[inline(always)]
        fn fma(self, a: Self, b: Self) -> Self {
            Self(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
        }

        #[inline(always)]
        fn fms(self, a: Self, b: Self) -> Self {
            Self(f32x4_sub(self.0, f32x4_mul(a.0, b.0)))
        }

        #[inline(always)]
        fn max(self, rhs: Self) -> Self {
            Self(f32x4_max(self.0, rhs.0))
        }

        #[inline(always)]
        fn min(self, rhs: Self) -> Self {
            Self(f32x4_min(self.0, rhs.0))
        }

        #[inline(always)]
        fn abs(self) -> Self {
            Self(f32x4_abs(self.0))
        }

        #[inline(always)]
        fn scrub_recursive_state(self) -> Self {
            let abs_bits = v128_and(self.0, u32x4_splat(IEEE754_F32_ABS_MASK));
            let is_sub = u32x4_lt(abs_bits, u32x4_splat(IEEE754_F32_IMPLICIT_BIT));
            let signed_zero = v128_and(self.0, u32x4_splat(IEEE754_F32_SIGN_MASK));
            let flushed = v128_bitselect(signed_zero, self.0, is_sub);
            let non_finite = u32x4_ge(abs_bits, u32x4_splat(IEEE754_F32_EXP_MASK));
            Self(v128_andnot(flushed, non_finite))
        }

        #[inline(always)]
        fn flush_subnormal(self) -> Self {
            let abs_bits = v128_and(self.0, u32x4_splat(IEEE754_F32_ABS_MASK));
            let is_sub = u32x4_lt(abs_bits, u32x4_splat(IEEE754_F32_IMPLICIT_BIT));
            let signed_zero = v128_and(self.0, u32x4_splat(IEEE754_F32_SIGN_MASK));
            Self(v128_bitselect(signed_zero, self.0, is_sub))
        }

        #[inline(always)]
        fn horizontal_sum(self) -> f32 {
            let a = self.to_array();
            (a[0] + a[1]) + (a[2] + a[3])
        }

        #[inline(always)]
        fn rcp_pos_seed_bits(self) -> Self {
            Self(u32x4_sub(u32x4_splat(IEEE754_F32_RCP_SEED), self.0))
        }

        #[inline(always)]
        fn rsqrt_pos_seed_bits(self) -> Self {
            Self(u32x4_sub(
                u32x4_splat(IEEE754_F32_RSQRT_SEED),
                u32x4_shr(self.0, 1),
            ))
        }

        #[inline(always)]
        fn pow2_from_biased_int_bits(self) -> Self {
            Self(i32x4_shl(
                u32x4_add(self.0, u32x4_splat(IEEE754_F32_EXP_BIAS)),
                IEEE754_F32_MANTISSA_BITS,
            ))
        }

        #[inline(always)]
        fn exp2_clamped5(self) -> Self {
            <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
        }
    }

    impl dnam_kernels::KernelVec4 for WasmVec4 {
        #[inline(always)]
        fn splat(val: f32) -> Self {
            <Self as SimdVec4>::splat(val)
        }

        #[inline(always)]
        fn load(src: &[f32; 4]) -> Self {
            <Self as SimdVec4>::load(src)
        }

        #[inline(always)]
        fn store(self, dst: &mut [f32; 4]) {
            <Self as SimdVec4>::store(self, dst);
        }

        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            <Self as SimdVec4>::add(self, rhs)
        }

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            <Self as SimdVec4>::sub(self, rhs)
        }

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            <Self as SimdVec4>::mul(self, rhs)
        }

        #[inline(always)]
        fn fma(self, a: Self, b: Self) -> Self {
            <Self as SimdVec4>::fma(self, a, b)
        }

        #[inline(always)]
        fn fms(self, a: Self, b: Self) -> Self {
            <Self as SimdVec4>::fms(self, a, b)
        }

        #[inline(always)]
        fn max(self, rhs: Self) -> Self {
            <Self as SimdVec4>::max(self, rhs)
        }

        #[inline(always)]
        fn min(self, rhs: Self) -> Self {
            <Self as SimdVec4>::min(self, rhs)
        }

        #[inline(always)]
        fn abs(self) -> Self {
            <Self as SimdVec4>::abs(self)
        }

        #[inline(always)]
        fn neg(self) -> Self {
            Self(f32x4_neg(self.0))
        }

        #[inline(always)]
        fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
            Self(v128_bitselect(
                if_true.0,
                if_false.0,
                f32x4_le(lhs.0, rhs.0),
            ))
        }

        #[inline(always)]
        fn horizontal_sum(self) -> f32 {
            <Self as SimdVec4>::horizontal_sum(self)
        }

        #[inline(always)]
        fn flush_subnormal(self) -> Self {
            <Self as SimdVec4>::flush_subnormal(self)
        }

        #[inline(always)]
        fn flush_recursive_state(self) -> Self {
            <Self as SimdVec4>::scrub_recursive_state(self)
        }

        #[inline(always)]
        fn rcp_pos_seed_bits(self) -> Self {
            <Self as SimdVec4>::rcp_pos_seed_bits(self)
        }

        #[inline(always)]
        fn rsqrt_pos_seed_bits(self) -> Self {
            <Self as SimdVec4>::rsqrt_pos_seed_bits(self)
        }

        #[inline(always)]
        fn pow2_from_biased_int_bits(self) -> Self {
            <Self as SimdVec4>::pow2_from_biased_int_bits(self)
        }
    }

    const _: fn(v128) -> v128 = |v| v128_xor(v, i32x4_splat(0));
}

#[cfg(target_feature = "simd128")]
pub use simd128_impl::WasmVec4;

/// Portable fallback `WasmVec4` when compiled without `+simd128` target feature.
#[cfg(not(target_feature = "simd128"))]
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct WasmVec4(pub ScalarVec4);

#[cfg(not(target_feature = "simd128"))]
impl SimdVec4 for WasmVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(ScalarVec4::splat(val))
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(ScalarVec4::load(src))
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        self.0.store(dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(self.0.fma(a.0, b.0))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(self.0.fms(a.0, b.0))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        Self(self.0.scrub_recursive_state())
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        self.0.horizontal_sum()
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self(self.0.rcp_pos_seed_bits())
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self(self.0.rsqrt_pos_seed_bits())
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self(self.0.pow2_from_biased_int_bits())
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
    }
}

#[cfg(not(target_feature = "simd128"))]
impl dnam_kernels::KernelVec4 for WasmVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        <Self as SimdVec4>::splat(val)
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        <Self as SimdVec4>::load(src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        <Self as SimdVec4>::store(self, dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        <Self as SimdVec4>::add(self, rhs)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        <Self as SimdVec4>::sub(self, rhs)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        <Self as SimdVec4>::mul(self, rhs)
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        <Self as SimdVec4>::fma(self, a, b)
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        <Self as SimdVec4>::fms(self, a, b)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        <Self as SimdVec4>::max(self, rhs)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        <Self as SimdVec4>::min(self, rhs)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        <Self as SimdVec4>::abs(self)
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        <Self as SimdVec4>::horizontal_sum(self)
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        <Self as SimdVec4>::rcp_pos_seed_bits(self)
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        <Self as SimdVec4>::rsqrt_pos_seed_bits(self)
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        <Self as SimdVec4>::pow2_from_biased_int_bits(self)
    }
}

/// WebAssembly 128-bit SIMD [`MathBackend`] (`WasmSimd128Backend`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WasmSimd128Backend;

impl Crc32cBackend for WasmSimd128Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        crc32c_software_slice(crc_init, data)
    }
}

impl MathBackend for WasmSimd128Backend {
    type Vec4 = WasmVec4;
    type Stereo8 = dnam_kernels::InterleavedStereo8<WasmVec4>;
    const NAME: &'static str = "WASM SIMD";
    const KIND: BackendKind = BackendKind::WasmSimd128;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_backend_identity_and_crc32c() {
        assert_eq!(WasmSimd128Backend::NAME, "WASM SIMD");
        assert_eq!(WasmSimd128Backend::crc32c(b"123456789"), 0xE306_9283);
        let _ = detect_wasm_backend();
    }
}
