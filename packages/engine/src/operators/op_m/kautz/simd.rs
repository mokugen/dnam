//! 4-lane SIMD kernels for [`super::KautzModalBank`] (`Stage III`).

use super::KautzModalBank;
use crate::math::flush_recursive_state_vec4;

impl KautzModalBank<4> {
    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks sharing
    /// identical poles and residues (`lambda_re`, `lambda_im`, `b_gamma`, `c_re`, `c_im`) using
    /// `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`), loading the 5 coefficient vectors once.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_stereo_simd_vec<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
    ) -> (B::Vec4, B::Vec4) {
        use crate::math::SimdVec4;
        self.step_4lane_stereo_simd_splatted::<B>(right, B::Vec4::splat(x_l), B::Vec4::splat(x_r))
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks with
    /// pre-splatted `x_l_vec` and `x_r_vec` (`0 FDIV`, `0 FSQRT`).
    #[inline(always)]
    pub fn step_4lane_stereo_simd_splatted<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l_vec: B::Vec4,
        x_r_vec: B::Vec4,
    ) -> (B::Vec4, B::Vec4) {
        use crate::math::SimdVec4;
        let l_re = B::Vec4::load(&self.lambda_re);
        let l_im = B::Vec4::load(&self.lambda_im);
        let b_g = B::Vec4::load(&self.b_gamma);
        let c_r_l = B::Vec4::load(&self.c_re);
        let c_i_l = B::Vec4::load(&self.c_im);
        let c_r_r = B::Vec4::load(&right.c_re);
        let c_i_r = B::Vec4::load(&right.c_im);
        let two = B::Vec4::splat(2.0);

        let prev_re_l = B::Vec4::load(&self.s_re);
        let prev_im_l = B::Vec4::load(&self.s_im);
        let prev_re_r = B::Vec4::load(&right.s_re);
        let prev_im_r = B::Vec4::load(&right.s_im);

        let next_re_l = l_re
            .mul(prev_re_l)
            .sub(l_im.mul(prev_im_l))
            .fma(b_g, x_l_vec);
        let next_im_l = l_im.mul(prev_re_l).add(l_re.mul(prev_im_l));

        let next_re_r = l_re
            .mul(prev_re_r)
            .sub(l_im.mul(prev_im_r))
            .fma(b_g, x_r_vec);
        let next_im_r = l_im.mul(prev_re_r).add(l_re.mul(prev_im_r));

        flush_recursive_state_vec4(next_re_l).store(&mut self.s_re);
        flush_recursive_state_vec4(next_im_l).store(&mut self.s_im);
        flush_recursive_state_vec4(next_re_r).store(&mut right.s_re);
        flush_recursive_state_vec4(next_im_r).store(&mut right.s_im);

        let out_l = c_r_l.mul(next_re_l).sub(c_i_l.mul(next_im_l)).mul(two);
        let out_r = c_r_r.mul(next_re_r).sub(c_i_r.mul(next_im_r)).mul(two);
        (out_l, out_r)
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of 4-pole Kautz modal banks with
    /// pre-splatted `x_l_vec` and `x_r_vec`, returning unscaled `Re(c * s)` (half of the 2*Re readout)
    /// so `(vl + vr)` equals `0.5 * (2*Re_L + 2*Re_R)` without 3 extra vector multiplies per block.
    #[inline(always)]
    pub fn step_4lane_stereo_simd_unscaled<B: crate::math::MathBackend>(
        &mut self,
        right: &mut Self,
        x_l_vec: B::Vec4,
        x_r_vec: B::Vec4,
    ) -> (B::Vec4, B::Vec4) {
        use crate::math::SimdVec4;
        let l_re = B::Vec4::load(&self.lambda_re);
        let l_im = B::Vec4::load(&self.lambda_im);
        let b_g = B::Vec4::load(&self.b_gamma);
        let c_r_l = B::Vec4::load(&self.c_re);
        let c_i_l = B::Vec4::load(&self.c_im);
        let c_r_r = B::Vec4::load(&right.c_re);
        let c_i_r = B::Vec4::load(&right.c_im);

        let prev_re_l = B::Vec4::load(&self.s_re);
        let prev_im_l = B::Vec4::load(&self.s_im);
        let prev_re_r = B::Vec4::load(&right.s_re);
        let prev_im_r = B::Vec4::load(&right.s_im);

        let next_re_l = l_re
            .mul(prev_re_l)
            .sub(l_im.mul(prev_im_l))
            .fma(b_g, x_l_vec);
        let next_im_l = l_im.mul(prev_re_l).add(l_re.mul(prev_im_l));

        let next_re_r = l_re
            .mul(prev_re_r)
            .sub(l_im.mul(prev_im_r))
            .fma(b_g, x_r_vec);
        let next_im_r = l_im.mul(prev_re_r).add(l_re.mul(prev_im_r));

        flush_recursive_state_vec4(next_re_l).store(&mut self.s_re);
        flush_recursive_state_vec4(next_im_l).store(&mut self.s_im);
        flush_recursive_state_vec4(next_re_r).store(&mut right.s_re);
        flush_recursive_state_vec4(next_im_r).store(&mut right.s_im);

        let out_l = c_r_l.mul(next_re_l).sub(c_i_l.mul(next_im_l));
        let out_r = c_r_r.mul(next_re_r).sub(c_i_r.mul(next_im_r));
        (out_l, out_r)
    }

    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the `B::Vec4` per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd_vec<B: crate::math::MathBackend>(&mut self, x: f32) -> B::Vec4 {
        use crate::math::SimdVec4;
        let prev_re = B::Vec4::from_array(self.s_re);
        let prev_im = B::Vec4::from_array(self.s_im);
        let l_re = B::Vec4::from_array(self.lambda_re);
        let l_im = B::Vec4::from_array(self.lambda_im);
        let b_g = B::Vec4::from_array(self.b_gamma);
        let c_r = B::Vec4::from_array(self.c_re);
        let c_i = B::Vec4::from_array(self.c_im);
        let x_vec = B::Vec4::splat(x);

        let next_re = l_re.mul(prev_re).sub(l_im.mul(prev_im)).add(b_g.mul(x_vec));
        let next_im = l_im.mul(prev_re).add(l_re.mul(prev_im));

        self.s_re = flush_recursive_state_vec4(next_re).to_array();
        self.s_im = flush_recursive_state_vec4(next_im).to_array();

        c_r.mul(next_re)
            .sub(c_i.mul(next_im))
            .mul(B::Vec4::splat(2.0))
    }

    /// Steps the 4-pole Kautz modal bank using `B::Vec4` SIMD operations (`0 FDIV`, `0 FSQRT`),
    /// returning the 4 per-lane gauge-invariant modal outputs `2 * (c_re * s_re - c_im * s_im)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_4lane_simd<B: crate::math::MathBackend>(&mut self, x: f32) -> [f32; 4] {
        use crate::math::SimdVec4;
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
}
