//! ARMv9.2-A Scalable Matrix Extension (`FEAT_SME` + `FEAT_SME_F64F64`) double-precision (`f64`)
//! trainer SIMD kernels (`TrainerMathBackend` for `ArmSmeBackend`).
//!
//! - Uses `FEAT_SME_F64F64` 2D `ZA` outer-product (`FMOPA` on `za0.d..za3.d`) tiles for `4x4` and
//!   symmetric Gram contractions when `ArmSmeBackend::is_f64f64_supported_cached()` is true.
//! - Uses Streaming SVE (`smstart sm` .. `smstop sm`) wide `f64` kernels without relying on
//!   `FEAT_SME_FA64`, keeping execution 100% legal on Apple M4 (`FEAT_SME` + `NEON` without
//!   `FEAT_SVE2` or `FEAT_SME_FA64`).
//! - Falls back at runtime to [`ArmSve2Backend`] when `FEAT_SVE2` is present, and otherwise to
//!   [`NeonBackend`].

mod kernels;

use kernels::{
    sme_ssve_advance_bezout7, sme_ssve_axpy_f64, sme_ssve_axpy_pair_f64, sme_ssve_dot_f64,
    sme_ssve_dot_pair_f64, sme_ssve_scale_f64, sme_ssve_sub_f32_to_f64, sme_ssve_sum_sq_f64,
    sme_za_gram_tile_4x4_f64,
};

use super::neon::NeonBackend;
use super::sme::ArmSmeBackend;
use super::sve2::ArmSve2Backend;
use super::trainer_f64::NeonF64Vec2;
use crate::math::trainer::{
    TrainerMathBackend, contract_tile_4x4_vec2_ieee, contract_tile_8x4_vec2_ieee,
};

/// Minimum vector length at which Streaming SVE (`smstart sm` .. `smstop sm`) amortizes transition.
const SME_SSVE_MIN_LEN: usize = 16;

impl TrainerMathBackend for ArmSmeBackend {
    type F64Vec2 = NeonF64Vec2;

    #[inline]
    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n <= min(a.len(), b.len())`.
            unsafe { sme_ssve_dot_f64(a.as_ptr(), b.as_ptr(), n) }
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::dot_f64(a, b)
        } else {
            NeonBackend::dot_f64(a, b)
        }
    }

    #[inline]
    fn dot_pair_f64(x: &[f64], b0: &[f64], b1: &[f64]) -> (f64, f64) {
        let n = x.len().min(b0.len()).min(b1.len());
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n` bounded across all three slices.
            unsafe { sme_ssve_dot_pair_f64(x.as_ptr(), b0.as_ptr(), b1.as_ptr(), n) }
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::dot_pair_f64(x, b0, b1)
        } else {
            NeonBackend::dot_pair_f64(x, b0, b1)
        }
    }

    #[inline]
    fn sum_sq_f64(x: &[f64]) -> f64 {
        let n = x.len();
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n == x.len()`.
            unsafe { sme_ssve_sum_sq_f64(x.as_ptr(), n) }
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::sum_sq_f64(x)
        } else {
            NeonBackend::sum_sq_f64(x)
        }
    }

    #[inline]
    fn axpy_f64(y: &mut [f64], alpha: f64, x: &[f64]) {
        let n = y.len().min(x.len());
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n` bounded across `y` and `x`.
            unsafe { sme_ssve_axpy_f64(y.as_mut_ptr(), alpha, x.as_ptr(), n) };
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::axpy_f64(y, alpha, x);
        } else {
            NeonBackend::axpy_f64(y, alpha, x);
        }
    }

    #[inline]
    fn axpy_pair_f64(y: &mut [f64], a0: f64, x0: &[f64], a1: f64, x1: &[f64]) {
        let n = y.len().min(x0.len()).min(x1.len());
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n` bounded across `y`, `x0`, and `x1`.
            unsafe { sme_ssve_axpy_pair_f64(y.as_mut_ptr(), a0, x0.as_ptr(), a1, x1.as_ptr(), n) };
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::axpy_pair_f64(y, a0, x0, a1, x1);
        } else {
            NeonBackend::axpy_pair_f64(y, a0, x0, a1, x1);
        }
    }

    #[inline]
    fn scale_f64(x: &mut [f64], scale: f64) {
        let n = x.len();
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n == x.len()`.
            unsafe { sme_ssve_scale_f64(x.as_mut_ptr(), scale, n) };
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::scale_f64(x, scale);
        } else {
            NeonBackend::scale_f64(x, scale);
        }
    }

    #[inline]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        let n = dst.len().min(a.len()).min(b.len());
        if n >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `n` bounded across `dst`, `a`, and `b`.
            unsafe { sme_ssve_sub_f32_to_f64(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n) };
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::sub_f32_to_f64(dst, a, b);
        } else {
            NeonBackend::sub_f32_to_f64(dst, a, b);
        }
    }

    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        let mut n = cols_i[0].len().min(cols_j[0].len());
        for k in 1..4 {
            n = n.min(cols_i[k].len()).min(cols_j[k].len());
        }
        let pairs = n / 2;
        if pairs > 0 && Self::is_f64f64_supported_cached() {
            // SAFETY: `FEAT_SME_F64F64` verified and every column slice has length >= `pairs * 2`.
            let mut out = unsafe {
                sme_za_gram_tile_4x4_f64(
                    cols_i[0].as_ptr(),
                    cols_i[1].as_ptr(),
                    cols_i[2].as_ptr(),
                    cols_i[3].as_ptr(),
                    cols_j[0].as_ptr(),
                    cols_j[1].as_ptr(),
                    cols_j[2].as_ptr(),
                    cols_j[3].as_ptr(),
                    pairs,
                )
            };
            let t = pairs * 2;
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
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::gram_tile_4x4_f64(cols_i, cols_j)
        } else {
            NeonBackend::gram_tile_4x4_f64(cols_i, cols_j)
        }
    }

    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        let n = cols[0]
            .len()
            .min(cols[1].len())
            .min(cols[2].len())
            .min(cols[3].len());
        let pairs = n / 2;
        if pairs > 0 && Self::is_f64f64_supported_cached() {
            // SAFETY: `FEAT_SME_F64F64` verified and every column slice has length >= `pairs * 2`.
            let mut m = unsafe {
                sme_za_gram_tile_4x4_f64(
                    cols[0].as_ptr(),
                    cols[1].as_ptr(),
                    cols[2].as_ptr(),
                    cols[3].as_ptr(),
                    cols[0].as_ptr(),
                    cols[1].as_ptr(),
                    cols[2].as_ptr(),
                    cols[3].as_ptr(),
                    pairs,
                )
            };
            let t = pairs * 2;
            if t < n {
                for r in 0..4 {
                    let xi = cols[r][t];
                    for c in r..4 {
                        m[r][c] += xi * cols[c][t];
                    }
                }
            }
            [
                m[0][0], m[0][1], m[1][1], m[0][2], m[1][2], m[2][2], m[0][3], m[1][3], m[2][3],
                m[3][3],
            ]
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::gram_sym_block4_f64(cols)
        } else {
            NeonBackend::gram_sym_block4_f64(cols)
        }
    }

    #[inline]
    fn contract_tile_4x4_ieee_f64(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4] {
        if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::contract_tile_4x4_ieee_f64(f_i, wf_j)
        } else {
            contract_tile_4x4_vec2_ieee::<NeonF64Vec2>(f_i, wf_j)
        }
    }

    #[inline]
    fn contract_tile_8x4_ieee_f64(
        f_i0: &[[f64; 4]],
        f_i1: &[[f64; 4]],
        wf_j: &[[f64; 4]],
    ) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
        if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::contract_tile_8x4_ieee_f64(f_i0, f_i1, wf_j)
        } else {
            contract_tile_8x4_vec2_ieee::<NeonF64Vec2>(f_i0, f_i1, wf_j)
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
        let bound = dim.min(256);
        if bound >= SME_SSVE_MIN_LEN && Self::is_supported_cached() {
            // SAFETY: `FEAT_SME` verified and `bound <= 256`.
            unsafe { sme_ssve_advance_bezout7(fb.as_ptr(), bound, r, omr, u.as_mut_ptr()) };
        } else if ArmSve2Backend::is_supported_cached() {
            ArmSve2Backend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
        } else {
            NeonBackend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
        }
    }
}
