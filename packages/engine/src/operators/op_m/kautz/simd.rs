//! 4-lane SIMD and 8-lane stereo kernels for [`super::KautzModalBank`] (`Stage III`),
//! bridged onto [`dnam_kernels::step_kautz_quad_stereo8`] and [`crate::math::MathBackend::Stereo8`].

use dnam_kernels::{KautzQuadState, StereoVec8, step_kautz_quad_stereo8};

use super::KautzModalBank;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_vec4};

#[inline(always)]
fn step_kautz_pair_stereo8<B: MathBackend>(
    left: &mut KautzModalBank<4>,
    right: &mut KautzModalBank<4>,
    x_l_vec: B::Vec4,
    x_r_vec: B::Vec4,
) -> (B::Vec4, B::Vec4) {
    let l_re = B::Stereo8::broadcast_half(B::Vec4::load(&left.lambda_re));
    let l_im = B::Stereo8::broadcast_half(B::Vec4::load(&left.lambda_im));
    let b_g = B::Stereo8::broadcast_half(B::Vec4::load(&left.b_gamma));
    let c_r = B::Stereo8::from_lr(B::Vec4::load(&left.c_re), B::Vec4::load(&right.c_re));
    let c_i = B::Stereo8::from_lr(B::Vec4::load(&left.c_im), B::Vec4::load(&right.c_im));
    let prev_re = B::Stereo8::from_lr(B::Vec4::load(&left.s_re), B::Vec4::load(&right.s_re))
        .flush_recursive_state();
    let prev_im = B::Stereo8::from_lr(B::Vec4::load(&left.s_im), B::Vec4::load(&right.s_im))
        .flush_recursive_state();
    let x = B::Stereo8::from_lr(x_l_vec, x_r_vec).flush_recursive_state();

    let next_re = l_re
        .mul(prev_re)
        .sub(l_im.mul(prev_im))
        .fma(b_g, x)
        .flush_recursive_state();
    let next_im = l_im
        .mul(prev_re)
        .add(l_re.mul(prev_im))
        .flush_recursive_state();

    let (nre_l, nre_r) = next_re.into_lr();
    let (nim_l, nim_r) = next_im.into_lr();
    nre_l.store(&mut left.s_re);
    nim_l.store(&mut left.s_im);
    nre_r.store(&mut right.s_re);
    nim_r.store(&mut right.s_im);

    c_r.mul(next_re)
        .sub(c_i.mul(next_im))
        .flush_recursive_state()
        .into_lr()
}

impl KautzModalBank<4> {
    /// Builds a [`dnam_kernels::KautzQuadState`] mirror from a stereo `(left = self, right)` bank pair.
    #[inline(always)]
    #[must_use]
    pub fn to_kernel_state<V: dnam_kernels::KernelVec4>(&self, right: &Self) -> KautzQuadState<V> {
        let mut st = KautzQuadState::new(
            V::load(&self.lambda_re),
            V::load(&self.lambda_im),
            V::load(&self.b_gamma),
            V::load(&self.c_re),
            V::load(&self.c_im),
            V::load(&right.c_re),
            V::load(&right.c_im),
        );
        st.z_re_l = V::load(&self.s_re);
        st.z_im_l = V::load(&self.s_im);
        st.z_re_r = V::load(&right.s_re);
        st.z_im_r = V::load(&right.s_im);
        st
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) directly via
    /// [`dnam_kernels::step_kautz_quad_stereo8`] over `B::Stereo8`, returning summed `(y_l, y_r)`.
    #[inline(always)]
    pub fn step_4lane_stereo_kernel_sum<B: MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
    ) -> (f32, f32) {
        let mut st = self.to_kernel_state::<B::Vec4>(right);
        let (out_l, out_r) = step_kautz_quad_stereo8::<B::Stereo8>(&mut st, x_l, x_r);
        st.z_re_l.store(&mut self.s_re);
        st.z_im_l.store(&mut self.s_im);
        st.z_re_r.store(&mut right.s_re);
        st.z_im_r.store(&mut right.s_im);
        (
            crate::math::flush_recursive_state_f32(out_l),
            crate::math::flush_recursive_state_f32(out_r),
        )
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks sharing
    /// identical poles and residues (`lambda_re`, `lambda_im`, `b_gamma`, `c_re`, `c_im`) using
    /// `B::Stereo8` SIMD operations (`0 FDIV`, `0 FSQRT`), loading the 5 coefficient vectors once.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_stereo_simd_vec<B: MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
    ) -> (B::Vec4, B::Vec4) {
        self.step_4lane_stereo_simd_splatted::<B>(right, B::Vec4::splat(x_l), B::Vec4::splat(x_r))
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks with
    /// pre-splatted `x_l_vec` and `x_r_vec` (`0 FDIV`, `0 FSQRT`).
    #[inline(always)]
    pub fn step_4lane_stereo_simd_splatted<B: MathBackend>(
        &mut self,
        right: &mut Self,
        x_l_vec: B::Vec4,
        x_r_vec: B::Vec4,
    ) -> (B::Vec4, B::Vec4) {
        let (out_l, out_r) = step_kautz_pair_stereo8::<B>(self, right, x_l_vec, x_r_vec);
        let two = B::Vec4::splat(2.0);
        (
            flush_recursive_state_vec4(out_l.mul(two)),
            flush_recursive_state_vec4(out_r.mul(two)),
        )
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks with
    /// pre-splatted `x_l_vec` and `x_r_vec`, returning unscaled `Re(c * s)` (half of the 2*Re readout)
    /// so `(vl + vr)` equals `0.5 * (2*Re_L + 2*Re_R)` without 3 extra vector multiplies per block.
    #[inline(always)]
    pub fn step_4lane_stereo_simd_unscaled<B: MathBackend>(
        &mut self,
        right: &mut Self,
        x_l_vec: B::Vec4,
        x_r_vec: B::Vec4,
    ) -> (B::Vec4, B::Vec4) {
        step_kautz_pair_stereo8::<B>(self, right, x_l_vec, x_r_vec)
    }

    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the `B::Vec4` per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd_vec<B: MathBackend>(&mut self, x: f32) -> B::Vec4 {
        let prev_re = flush_recursive_state_vec4(B::Vec4::from_array(self.s_re));
        let prev_im = flush_recursive_state_vec4(B::Vec4::from_array(self.s_im));
        let l_re = B::Vec4::from_array(self.lambda_re);
        let l_im = B::Vec4::from_array(self.lambda_im);
        let b_g = B::Vec4::from_array(self.b_gamma);
        let c_r = B::Vec4::from_array(self.c_re);
        let c_i = B::Vec4::from_array(self.c_im);
        let x_vec = flush_recursive_state_vec4(B::Vec4::splat(x));

        let next_re = flush_recursive_state_vec4(
            l_re.mul(prev_re).sub(l_im.mul(prev_im)).add(b_g.mul(x_vec)),
        );
        let next_im = flush_recursive_state_vec4(l_im.mul(prev_re).add(l_re.mul(prev_im)));

        next_re.store(&mut self.s_re);
        next_im.store(&mut self.s_im);

        flush_recursive_state_vec4(
            c_r.mul(next_re)
                .sub(c_i.mul(next_im))
                .mul(B::Vec4::splat(2.0)),
        )
    }

    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the 4 per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd<B: MathBackend>(&mut self, x: f32) -> [f32; 4] {
        self.step_4lane_simd_vec::<B>(x).to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::math::{DefaultMathBackend, ScalarBackend, SimdVec4};

    #[test]
    fn test_kautz_4lane_stereo_simd_matches_scalar() {
        let _audio = FtzDazGuard::new();
        let mut s_l = KautzModalBank::<4>::default();
        let mut s_r = KautzModalBank::<4>::default();
        let mut v_l = KautzModalBank::<4>::default();
        let mut v_r = KautzModalBank::<4>::default();
        let mut u_l = KautzModalBank::<4>::default();
        let mut u_r = KautzModalBank::<4>::default();
        let r = 0.93_f32;
        let b_g = (1.0 - r * r).sqrt();
        for k in 0..4 {
            let theta = 0.2 + 0.4 * (k as f32);
            let cr = 0.6 - 0.1 * (k as f32);
            let ci = -0.25 + 0.08 * (k as f32);
            for bank in [&mut s_l, &mut s_r, &mut v_l, &mut v_r, &mut u_l, &mut u_r] {
                bank.set_pole_with_b_gamma(k, r, theta, 0.01, b_g, cr, ci);
            }
        }
        for &(xl, xr) in &[(0.8_f32, -0.4_f32), (-0.3, 0.7), (0.0, 0.15)] {
            let exp_l = s_l.step(xl);
            let exp_r = s_r.step(xr);
            let (vl, vr) = v_l.step_4lane_stereo_simd_vec::<ScalarBackend>(&mut v_r, xl, xr);
            assert!((vl.horizontal_sum() - exp_l).abs() < 1e-5);
            assert!((vr.horizontal_sum() - exp_r).abs() < 1e-5);

            type V = <DefaultMathBackend as crate::math::MathBackend>::Vec4;
            let (ul, ur) = u_l.step_4lane_stereo_simd_unscaled::<DefaultMathBackend>(
                &mut u_r,
                V::splat(xl),
                V::splat(xr),
            );
            assert!((2.0 * ul.horizontal_sum() - exp_l).abs() < 1e-5);
            assert!((2.0 * ur.horizontal_sum() - exp_r).abs() < 1e-5);
        }
    }

    #[test]
    fn test_kautz_4lane_stereo_kernel_sum_matches_simd_and_scalar() {
        let _audio = FtzDazGuard::new();
        let mut ks_l = KautzModalBank::<4>::default();
        let mut ks_r = KautzModalBank::<4>::default();
        let mut kd_l = KautzModalBank::<4>::default();
        let mut kd_r = KautzModalBank::<4>::default();
        let mut un_l = KautzModalBank::<4>::default();
        let mut un_r = KautzModalBank::<4>::default();
        let r = 0.91_f32;
        let b_g = (1.0 - r * r).sqrt();
        for k in 0..4 {
            let theta = 0.15 + 0.35 * (k as f32);
            let cr = 0.5 - 0.08 * (k as f32);
            let ci = -0.2 + 0.06 * (k as f32);
            for bank in [
                &mut ks_l, &mut ks_r, &mut kd_l, &mut kd_r, &mut un_l, &mut un_r,
            ] {
                bank.set_pole_with_b_gamma(k, r, theta, 0.0, b_g, cr, ci);
            }
        }
        type DefV = <DefaultMathBackend as MathBackend>::Vec4;
        for &(xl, xr) in &[(0.75_f32, -0.35_f32), (-0.4, 0.6), (0.1, -0.25)] {
            let (sl, sr) = ks_l.step_4lane_stereo_kernel_sum::<ScalarBackend>(&mut ks_r, xl, xr);
            let (dl, dr) =
                kd_l.step_4lane_stereo_kernel_sum::<DefaultMathBackend>(&mut kd_r, xl, xr);
            let (ul, ur) = un_l.step_4lane_stereo_simd_unscaled::<DefaultMathBackend>(
                &mut un_r,
                DefV::splat(xl),
                DefV::splat(xr),
            );
            assert!((sl - dl).abs() < 1.0e-5);
            assert!((sr - dr).abs() < 1.0e-5);
            assert!((2.0 * ul.horizontal_sum() - dl).abs() < 1.0e-5);
            assert!((2.0 * ur.horizontal_sum() - dr).abs() < 1.0e-5);
        }
    }

    #[test]
    fn test_kautz_simd_scrubs_nan_and_inf_without_latching() {
        let _audio = FtzDazGuard::new();
        let init_bank = || {
            let mut bank = KautzModalBank::<4>::default();
            let r = 0.90_f32;
            let b_g = (1.0 - r * r).sqrt();
            for k in 0..4 {
                bank.set_pole_with_b_gamma(k, r, 0.25 + 0.3 * (k as f32), 0.0, b_g, 0.5, -0.2);
            }
            bank
        };

        // 1. 4-lane mono SIMD recovers from NaN/Inf inputs and corrupted state without NaN output.
        for bad_x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut m_scal = init_bank();
            let mut m_def = init_bank();
            m_scal.s_re[0] = f32::NAN;
            m_scal.s_im[1] = f32::INFINITY;
            m_def.s_re[0] = f32::NAN;
            m_def.s_im[1] = f32::INFINITY;

            let o_s = m_scal.step_4lane_simd::<ScalarBackend>(bad_x);
            let o_d = m_def.step_4lane_simd::<DefaultMathBackend>(bad_x);
            assert!(o_s.iter().all(|v| v.is_finite()));
            assert!(o_d.iter().all(|v| v.is_finite()));
            assert!(m_scal.s_re.iter().all(|v| v.is_finite()));
            assert!(m_scal.s_im.iter().all(|v| v.is_finite()));
            assert!(m_def.s_re.iter().all(|v| v.is_finite()));
            assert!(m_def.s_im.iter().all(|v| v.is_finite()));

            let rec_s = m_scal.step_4lane_simd::<ScalarBackend>(0.5);
            let rec_d = m_def.step_4lane_simd::<DefaultMathBackend>(0.5);
            assert!(rec_s.iter().all(|v| v.is_finite()));
            assert!(rec_d.iter().all(|v| v.is_finite()));
            assert!(rec_s.iter().any(|v| v.abs() > 0.0));
        }

        // 2. 8-lane stereo SIMD scrubs NaN/Inf inputs and corrupted state without NaN latching.
        let mut st_l = init_bank();
        let mut st_r = init_bank();
        st_l.s_re[2] = f32::NAN;
        st_r.s_im[3] = f32::NEG_INFINITY;
        let (vl, vr) = st_l.step_4lane_stereo_simd_vec::<DefaultMathBackend>(
            &mut st_r,
            f32::NAN,
            f32::INFINITY,
        );
        assert!(vl.to_array().iter().all(|v| v.is_finite()));
        assert!(vr.to_array().iter().all(|v| v.is_finite()));
        assert!(st_l.s_re.iter().all(|v| v.is_finite()));
        assert!(st_l.s_im.iter().all(|v| v.is_finite()));
        assert!(st_r.s_re.iter().all(|v| v.is_finite()));
        assert!(st_r.s_im.iter().all(|v| v.is_finite()));

        let (rl, rr) = st_l.step_4lane_stereo_simd_vec::<DefaultMathBackend>(&mut st_r, 0.5, -0.5);
        assert!(rl.to_array().iter().all(|v| v.is_finite()));
        assert!(rr.to_array().iter().all(|v| v.is_finite()));
        assert!(rl.to_array().iter().any(|v| v.abs() > 0.0));

        // 3. step_4lane_stereo_kernel_sum scrubs NaN/Inf inputs and poisoned state registers
        //    (`<= 2 ULP` vs unfused ScalarBackend, `0 ULP` across DefaultMathBackend paths).
        let mut ks_l = init_bank();
        let mut ks_r = init_bank();
        let mut kd_l = init_bank();
        let mut kd_r = init_bank();
        let mut kv_l = init_bank();
        let mut kv_r = init_bank();
        for (bl, br) in [
            (&mut ks_l, &mut ks_r),
            (&mut kd_l, &mut kd_r),
            (&mut kv_l, &mut kv_r),
        ] {
            bl.s_re = [f32::NAN, 0.1, f32::INFINITY, f32::NEG_INFINITY];
            bl.s_im = [f32::INFINITY, f32::NAN, -0.2, f32::NEG_INFINITY];
            br.s_re = [f32::NEG_INFINITY, f32::NAN, 0.25, f32::INFINITY];
            br.s_im = [0.15, f32::INFINITY, f32::NAN, f32::NEG_INFINITY];
        }
        let (sl, sr) =
            ks_l.step_4lane_stereo_kernel_sum::<ScalarBackend>(&mut ks_r, f32::NAN, f32::INFINITY);
        let (dl, dr) = kd_l.step_4lane_stereo_kernel_sum::<DefaultMathBackend>(
            &mut kd_r,
            f32::NAN,
            f32::INFINITY,
        );
        let mut kv_st = kv_l.to_kernel_state::<<DefaultMathBackend as MathBackend>::Vec4>(&kv_r);
        let (vl2, vr2) = dnam_kernels::step_kautz_quad_vec4(&mut kv_st, f32::NAN, f32::INFINITY);
        assert!(sl.is_finite() && sr.is_finite());
        assert!(dl.is_finite() && dr.is_finite());
        assert_eq!(vl2.to_bits(), dl.to_bits());
        assert_eq!(vr2.to_bits(), dr.to_bits());
        assert!((sl.to_bits() as i64 - dl.to_bits() as i64).abs() <= 2);
        assert!((sr.to_bits() as i64 - dr.to_bits() as i64).abs() <= 2);
        for bank in [&ks_l, &ks_r, &kd_l, &kd_r] {
            assert!(bank.s_re.iter().all(|v| v.is_finite()));
            assert!(bank.s_im.iter().all(|v| v.is_finite()));
        }

        let (rec_sl, rec_sr) =
            ks_l.step_4lane_stereo_kernel_sum::<ScalarBackend>(&mut ks_r, 0.5, -0.5);
        let (rec_dl, rec_dr) =
            kd_l.step_4lane_stereo_kernel_sum::<DefaultMathBackend>(&mut kd_r, 0.5, -0.5);
        let (rec_vl2, rec_vr2) = dnam_kernels::step_kautz_quad_vec4(&mut kv_st, 0.5, -0.5);
        assert!(rec_sl.is_finite() && rec_sr.is_finite());
        assert!(rec_dl.is_finite() && rec_dr.is_finite());
        assert!(rec_dl.abs() > 0.0 && rec_dr.abs() > 0.0);
        assert_eq!(rec_vl2.to_bits(), rec_dl.to_bits());
        assert_eq!(rec_vr2.to_bits(), rec_dr.to_bits());
        assert!((rec_sl.to_bits() as i64 - rec_dl.to_bits() as i64).abs() <= 2);
        assert!((rec_sr.to_bits() as i64 - rec_dr.to_bits() as i64).abs() <= 2);

        // 4. Readout overflow (`c_re = [f32::MAX; 4]`) and vector/SIMD wrappers scrub NaN/Inf cleanly.
        let mut ovf_l = init_bank();
        let mut ovf_r = init_bank();
        ovf_l.c_re = [f32::MAX; 4];
        ovf_r.c_re = [f32::MAX; 4];
        ovf_l.s_re = [f32::NAN, 1.0, f32::INFINITY, 1.0];
        ovf_r.s_re = [1.0, f32::NEG_INFINITY, 1.0, f32::NAN];
        let (ov_vec_l, ov_vec_r) =
            ovf_l.step_4lane_stereo_simd_vec::<DefaultMathBackend>(&mut ovf_r, 0.5, 0.5);
        assert!(ov_vec_l.to_array().iter().all(|v| v.is_finite()));
        assert!(ov_vec_r.to_array().iter().all(|v| v.is_finite()));

        let mut mono = init_bank();
        mono.c_re = [f32::MAX; 4];
        mono.s_re = [f32::NAN, 1.0, f32::INFINITY, 1.0];
        let ov_mono = mono.step_4lane_simd_vec::<DefaultMathBackend>(1.0);
        assert!(ov_mono.to_array().iter().all(|v| v.is_finite()));
        let rec_mono = mono.step_4lane_simd::<DefaultMathBackend>(f32::NAN);
        assert!(rec_mono.iter().all(|v| v.is_finite()));
    }
}
