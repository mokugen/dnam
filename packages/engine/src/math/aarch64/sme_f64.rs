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

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use proptest::prelude::*;

    use super::*;
    use crate::math::trainer::{ScalarF64Vec2, SimdF64Vec2};
    use crate::math::{BackendKind, ScalarBackend, warm_simd_feature_caches};

    #[test]
    fn test_sme_f64_trainer_all_lengths_and_fallback_parity() {
        warm_simd_feature_caches();
        assert_eq!(
            ArmSmeBackend::is_supported(),
            ArmSmeBackend::is_supported_cached()
        );
        assert_eq!(
            ArmSmeBackend::is_f64f64_supported(),
            ArmSmeBackend::is_f64f64_supported_cached()
        );
        assert_eq!(ArmSmeBackend::trainer_backend_kind(), BackendKind::ArmSme);

        let va = NeonF64Vec2::from_array([1.25, -3.5]);
        let vb = NeonF64Vec2::splat_pair(0.5, 2.0);
        let sa = ScalarF64Vec2::from_array([1.25, -3.5]);
        let sb = ScalarF64Vec2::splat_pair(0.5, 2.0);
        assert_eq!(
            va.mul_add_ieee(vb, NeonF64Vec2::splat(1.0)).to_array(),
            sa.mul_add_ieee(sb, ScalarF64Vec2::splat(1.0)).to_array()
        );
        assert_eq!(va.dot(vb), sa.dot(sb));

        for len in [0_usize, 1, 2, 3, 7, 15, 16, 17, 33] {
            let x: Vec<f64> = (0..len)
                .map(|i| 0.31 * ((i as f64) * 0.19).sin() - 0.09)
                .collect();
            let y: Vec<f64> = (0..len)
                .map(|i| -0.27 * ((i as f64) * 0.23).cos() + 0.07)
                .collect();
            let z: Vec<f64> = (0..len)
                .map(|i| 0.43 * ((i as f64) * 0.11).sin() + 0.04)
                .collect();

            let dot_diff = ArmSmeBackend::dot_f64(&x, &y) - ScalarBackend::dot_f64(&x, &y);
            assert!(dot_diff.abs() <= 1.0e-11);
            let (dp0, dp1) = ArmSmeBackend::dot_pair_f64(&x, &y, &z);
            let (sp0, sp1) = ScalarBackend::dot_pair_f64(&x, &y, &z);
            assert!((dp0 - sp0).abs() <= 1.0e-11 && (dp1 - sp1).abs() <= 1.0e-11);
            let sq_diff = ArmSmeBackend::sum_sq_f64(&x) - ScalarBackend::sum_sq_f64(&x);
            assert!(sq_diff.abs() <= 1.0e-11);

            let (mut ay, mut sy) = (x.clone(), x.clone());
            ArmSmeBackend::axpy_f64(&mut ay, 0.75, &y);
            ScalarBackend::axpy_f64(&mut sy, 0.75, &y);
            assert!(ay.iter().zip(&sy).all(|(u, v)| (u - v).abs() <= 1.0e-11));

            let (mut apy, mut spy) = (x.clone(), x.clone());
            ArmSmeBackend::axpy_pair_f64(&mut apy, 0.625, &y, -0.375, &z);
            ScalarBackend::axpy_pair_f64(&mut spy, 0.625, &y, -0.375, &z);
            assert!(apy.iter().zip(&spy).all(|(u, v)| (u - v).abs() <= 1.0e-11));

            ArmSmeBackend::scale_f64(&mut ay, 1.25);
            ScalarBackend::scale_f64(&mut sy, 1.25);
            assert!(ay.iter().zip(&sy).all(|(u, v)| (u - v).abs() <= 1.0e-11));

            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let (mut ds, mut db) = (vec![0.0_f64; len], vec![0.0_f64; len]);
            ArmSmeBackend::sub_f32_to_f64(&mut db, &x32, &y32);
            ScalarBackend::sub_f32_to_f64(&mut ds, &x32, &y32);
            assert!(db.iter().zip(&ds).all(|(u, v)| u.to_bits() == v.to_bits()));

            let ci = [x.as_slice(), y.as_slice(), z.as_slice(), x.as_slice()];
            let cj = [y.as_slice(), z.as_slice(), x.as_slice(), z.as_slice()];
            let gt_b = ArmSmeBackend::gram_tile_4x4_f64(ci, cj);
            let gt_s = ScalarBackend::gram_tile_4x4_f64(ci, cj);
            for r in 0..4 {
                assert!(
                    gt_b[r]
                        .iter()
                        .zip(&gt_s[r])
                        .all(|(u, v)| (u - v).abs() <= 1.0e-11)
                );
            }
            let gs_b = ArmSmeBackend::gram_sym_block4_f64(ci);
            let gs_s = ScalarBackend::gram_sym_block4_f64(ci);
            assert!(
                gs_b.iter()
                    .zip(&gs_s)
                    .all(|(u, v)| (u - v).abs() <= 1.0e-11)
            );

            let f0: Vec<[f64; 4]> = (0..len).map(|k| [k as f64 * 0.1, -0.3, 0.7, 0.2]).collect();
            let f1: Vec<[f64; 4]> = (0..len)
                .map(|k| [-0.4, k as f64 * 0.05, 0.11, -0.9])
                .collect();
            let wf: Vec<[f64; 4]> = (0..len)
                .map(|k| [0.5, -0.25, 0.125 + k as f64 * 0.02, -0.75])
                .collect();
            let t4_b = ArmSmeBackend::contract_tile_4x4_ieee_f64(&f0, &wf);
            let t4_s = ScalarBackend::contract_tile_4x4_ieee_f64(&f0, &wf);
            let (t8a_b, t8b_b) = ArmSmeBackend::contract_tile_8x4_ieee_f64(&f0, &f1, &wf);
            let (t8a_s, t8b_s) = ScalarBackend::contract_tile_8x4_ieee_f64(&f0, &f1, &wf);
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(t4_b[r][c].to_bits(), t4_s[r][c].to_bits());
                    assert_eq!(t8a_b[r][c].to_bits(), t8a_s[r][c].to_bits());
                    assert_eq!(t8b_b[r][c].to_bits(), t8b_s[r][c].to_bits());
                }
            }
        }
    }

    #[test]
    fn test_sme_f64_mismatched_slices_and_bezout7() {
        warm_simd_feature_caches();
        let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = [0.5_f64, -1.0, 1.5];
        let c = [2.0_f64, 3.0];
        assert!((ArmSmeBackend::dot_f64(&a, &b) - 3.0).abs() <= 1.0e-12);
        let (p0, p1) = ArmSmeBackend::dot_pair_f64(&a, &b, &c);
        assert!((p0 + 1.5).abs() <= 1.0e-12 && (p1 - 8.0).abs() <= 1.0e-12);

        let mut dst = [9.0_f64; 5];
        ArmSmeBackend::axpy_f64(&mut dst, 2.0, &b);
        assert_eq!(dst, [10.0, 7.0, 12.0, 9.0, 9.0]);
        ArmSmeBackend::axpy_pair_f64(&mut dst, 1.0, &b, -1.0, &c);
        assert_eq!(dst, [8.5, 3.0, 12.0, 9.0, 9.0]);

        let short_gt = ArmSmeBackend::gram_tile_4x4_f64([&a, &a, &b, &a], [&a, &c, &a, &a]);
        let scalar_gt = ScalarBackend::gram_tile_4x4_f64([&a, &a, &b, &a], [&a, &c, &a, &a]);
        for r in 0..4 {
            assert!(
                short_gt[r]
                    .iter()
                    .zip(&scalar_gt[r])
                    .all(|(u, v)| (u - v).abs() <= 1.0e-12)
            );
        }
        let short_gs = ArmSmeBackend::gram_sym_block4_f64([&a, &b, &c, &a]);
        let scalar_gs = ScalarBackend::gram_sym_block4_f64([&a, &b, &c, &a]);
        assert!(
            short_gs
                .iter()
                .zip(&scalar_gs)
                .all(|(u, v)| (u - v).abs() <= 1.0e-12)
        );

        let fb: [f32; 256] = core::array::from_fn(|idx| ((idx as f32) * 0.21).sin() * 0.4);
        for dim in [0_usize, 1, 3, 15, 16, 17, 65, 256, 300] {
            let (mut ub, mut us) = ([[0.0_f64; 256]; 7], [[0.0_f64; 256]; 7]);
            ArmSmeBackend::advance_bezout7_lp_dim_f64(&fb, dim, 0.95, 0.05, &mut ub);
            ScalarBackend::advance_bezout7_lp_dim_f64(&fb, dim, 0.95, 0.05, &mut us);
            for st in 0..7 {
                assert!(
                    ub[st]
                        .iter()
                        .zip(&us[st])
                        .all(|(u, v)| u.to_bits() == v.to_bits())
                );
            }
            let (mut lfb, mut mfb, mut hfb) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
            let (mut lfs, mut mfs, mut hfs) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
            ArmSmeBackend::readout_bezout7_bands_dim_f64(
                &fb, dim, &ub, &ub, &mut lfb, &mut mfb, &mut hfb,
            );
            ScalarBackend::readout_bezout7_bands_dim_f64(
                &fb, dim, &us, &us, &mut lfs, &mut mfs, &mut hfs,
            );
            for d in 0..256 {
                assert_eq!(lfb[d].to_bits(), lfs[d].to_bits());
                assert_eq!(mfb[d].to_bits(), mfs[d].to_bits());
                assert_eq!(hfb[d].to_bits(), hfs[d].to_bits());
            }
        }
    }

    proptest! {
        #[test]
        fn prop_sme_f64_trainer_matches_scalar_across_lengths(
            xs in prop::collection::vec(-5.0_f64..5.0_f64, 0..40),
            ys in prop::collection::vec(-5.0_f64..5.0_f64, 0..40),
        ) {
            warm_simd_feature_caches();
            let n = xs.len().min(ys.len());
            let d_dot = ArmSmeBackend::dot_f64(&xs[..n], &ys[..n]);
            let s_dot = ScalarBackend::dot_f64(&xs[..n], &ys[..n]);
            prop_assert!((d_dot - s_dot).abs() <= 1.0e-11 * s_dot.abs().max(1.0));
            let d_sq = ArmSmeBackend::sum_sq_f64(&xs[..n]);
            let s_sq = ScalarBackend::sum_sq_f64(&xs[..n]);
            prop_assert!((d_sq - s_sq).abs() <= 1.0e-11 * s_sq.abs().max(1.0));
            let x32: Vec<f32> = xs[..n].iter().map(|&v| v as f32).collect();
            let y32: Vec<f32> = ys[..n].iter().map(|&v| v as f32).collect();
            let (mut db, mut ds) = (vec![0.0; n], vec![0.0; n]);
            ArmSmeBackend::sub_f32_to_f64(&mut db, &x32, &y32);
            ScalarBackend::sub_f32_to_f64(&mut ds, &x32, &y32);
            for i in 0..n {
                prop_assert_eq!(db[i].to_bits(), ds[i].to_bits());
            }
        }
    }
}
