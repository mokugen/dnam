//! x86-64 `AVX2+FMA` [`TrainerMathBackend`] implementation with strict non-FMA `AVX`
//! tile contractions and Bézout cascades (`ref[train.gram.wide192]`).

mod hw;

use hw::{
    avx_advance_bezout7_lp_dim_hw, avx_contract_tile_4x4_ieee_hw, avx_contract_tile_8x4_ieee_hw,
    avx_scale_f64_hw, avx_sub_f32_to_f64_hw, avx2_axpy_f64_hw, avx2_axpy_pair_f64_hw,
    avx2_dot_f64_hw, avx2_dot_pair_f64_hw, avx2_gram_sym_block4_hw, avx2_gram_tile_4x4_hw,
    avx2_sum_sq_f64_hw,
};

use super::sse_f64::Avx2FmaF64Vec2;
use crate::math::trainer::TrainerMathBackend;
use crate::math::{X86Avx2FmaBackend, X86Sse2Backend};

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
    use crate::math::trainer::verify_trainer_backend_parity;
    use crate::math::{X86Avx2FmaBackend, X86Sse2Backend, X86Sse42Backend};

    #[test]
    fn test_x86_trainer_math_backends_parity() {
        crate::math::warm_simd_feature_caches();
        verify_trainer_backend_parity::<X86Sse2Backend>();
        verify_trainer_backend_parity::<X86Sse42Backend>();
        verify_trainer_backend_parity::<X86Avx2FmaBackend>();
    }
}
