//! Audio-rate stereo & sub-block Cytomic TPT/ZDF SVF loops (`V3Stage1SvfState`).

use super::coeffs::{
    SVF_G_MAX, SvfHoistedConsts, V3Stage1SvfParams, prewarp_svf_g1, tan_prewarp_vec4,
};
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32, flush_recursive_state_vec4};

/// Borrowed persistent scratch buffers for [`V3Stage1SvfState::step_subblock_fast`].
pub(crate) struct SvfSubblockOut<'a, const CAP: usize> {
    pub env: &'a mut [[f32; 2]; CAP],
    pub a: &'a mut [[[f32; 4]; 3]; CAP],
    pub u_s2: &'a mut [[f32; 2]; CAP],
    pub svf_dir: &'a mut [[f32; 2]; CAP],
    pub lp2: &'a mut [[f32; 2]; CAP],
}

/// Zero-lag stereo Stage I SVF state (`64 B` cache-line aligned, 1 line total).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3Stage1SvfState {
    pub det_lp_l: f32,
    pub det_lp_r: f32,
    pub env_l: f32,
    pub env_r: f32,
    pub ic1_s1_l: f32,
    pub ic2_s1_l: f32,
    pub ic1_s2_l: f32,
    pub ic2_s2_l: f32,
    pub ic1_s1_r: f32,
    pub ic2_s1_r: f32,
    pub ic1_s2_r: f32,
    pub ic2_s2_r: f32,
}

impl V3Stage1SvfState {
    /// Resets all filter and envelope state variables to exact resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline(always)]
    fn step_tpt_cached(
        u: f32,
        a1: f32,
        a2: f32,
        a3: f32,
        k: f32,
        ic1: &mut f32,
        ic2: &mut f32,
    ) -> (f32, f32, f32) {
        let c1 = *ic1;
        let c2 = *ic2;
        let v3 = u - c2;
        let v1 = a1 * c1 + a2 * v3;
        let v2 = c2 + a2 * c1 + a3 * v3;
        *ic1 = flush_recursive_state_f32(2.0 * v1 - c1);
        *ic2 = flush_recursive_state_f32(2.0 * v2 - c2);
        let lp = v2;
        let bp = v1;
        let hp = u - k * v1 - v2;
        (lp, bp, hp)
    }

    /// Sub-block stereo Stage I SVF kernel split into detector, 2-sample packed coefficient
    /// prewarping, and pure linear ZDF FMA recurrence.
    #[inline(always)]
    pub(crate) fn step_subblock_fast<B: MathBackend, const CAP: usize>(
        &mut self,
        c: &SvfHoistedConsts<B::Vec4>,
        stage1_in: &[[f32; 2]; CAP],
        out: &mut SvfSubblockOut<'_, CAP>,
        len: usize,
    ) {
        let n = len.min(CAP);
        if n == 0 {
            return;
        }
        let mut det_lp_l = self.det_lp_l;
        let mut det_lp_r = self.det_lp_r;
        let mut env_l = self.env_l;
        let mut env_r = self.env_r;
        for i in 0..n {
            let [x_l, x_r] = stage1_in[i];
            det_lp_l = flush_recursive_state_f32(c.lp_a * det_lp_l + c.lp_b * x_l);
            det_lp_r = flush_recursive_state_f32(c.lp_a * det_lp_r + c.lp_b * x_r);
            let d_raw_l = (x_l + c.det_hp_mix * (x_l - det_lp_l)).abs();
            let d_raw_r = (x_r + c.det_hp_mix * (x_r - det_lp_r)).abs();
            let diff = d_raw_r - d_raw_l;
            let det_l = d_raw_l + c.half_link * diff;
            let det_r = d_raw_r - c.half_link * diff;
            let alpha_l = if det_l > env_l {
                c.atk_alpha
            } else {
                c.rel_alpha
            };
            let alpha_r = if det_r > env_r {
                c.atk_alpha
            } else {
                c.rel_alpha
            };
            env_l = flush_recursive_state_f32(alpha_l * env_l + (1.0 - alpha_l) * det_l);
            env_r = flush_recursive_state_f32(alpha_r * env_r + (1.0 - alpha_r) * det_r);
            out.env[i] = [env_l, env_r];
        }
        self.det_lp_l = det_lp_l;
        self.det_lp_r = det_lp_r;
        self.env_l = env_l;
        self.env_r = env_r;

        let mut i = 0_usize;
        while i + 1 < n {
            let env_pair = B::Vec4::from_array([
                out.env[i][0],
                out.env[i][1],
                out.env[i + 1][0],
                out.env[i + 1][1],
            ]);
            let g1_pair = prewarp_svf_g1::<B>(c, env_pair);
            let g0 = g1_pair
                .dup_low_pair()
                .mul(c.ratio_v)
                .clamp(c.g_clip_lo, c.g_clip_hi);
            let g1 = g1_pair
                .dup_high_pair()
                .mul(c.ratio_v)
                .clamp(c.g_clip_lo, c.g_clip_hi);
            let (a1_0, a1_1) = c
                .one
                .fma(g0, g0.add(c.k_vec))
                .rcp_ge1_pair_nr2(c.one.fma(g1, g1.add(c.k_vec)));
            let a2_0 = g0.mul(a1_0);
            let a3_0 = g0.mul(a2_0);
            a1_0.store(&mut out.a[i][0]);
            a2_0.store(&mut out.a[i][1]);
            a3_0.store(&mut out.a[i][2]);
            let a2_1 = g1.mul(a1_1);
            let a3_1 = g1.mul(a2_1);
            a1_1.store(&mut out.a[i + 1][0]);
            a2_1.store(&mut out.a[i + 1][1]);
            a3_1.store(&mut out.a[i + 1][2]);
            i += 2;
        }
        if i < n {
            let g1_v = prewarp_svf_g1::<B>(c, B::Vec4::splat_pair(out.env[i][0], out.env[i][1]));
            let g_vec = g1_v.mul(c.ratio_v).clamp(c.g_clip_lo, c.g_clip_hi);
            let a1_v = c.one.fma(g_vec, g_vec.add(c.k_vec)).rcp_ge1_nr2();
            let a2_v = g_vec.mul(a1_v);
            let a3_v = g_vec.mul(a2_v);
            a1_v.store(&mut out.a[i][0]);
            a2_v.store(&mut out.a[i][1]);
            a3_v.store(&mut out.a[i][2]);
        }

        let two = B::Vec4::splat(2.0);
        let mut ic1_s1 = B::Vec4::splat_pair(self.ic1_s1_l, self.ic1_s1_r);
        let mut ic2_s1 = B::Vec4::splat_pair(self.ic2_s1_l, self.ic2_s1_r);
        let mut ic1_s2 = B::Vec4::splat_pair(self.ic1_s2_l, self.ic1_s2_r);
        let mut ic2_s2 = B::Vec4::splat_pair(self.ic2_s2_l, self.ic2_s2_r);
        for (j, &[x_l, x_r]) in stage1_in[..n].iter().enumerate() {
            let a1_v = B::Vec4::load(&out.a[j][0]);
            let a2_v = B::Vec4::load(&out.a[j][1]);
            let a3_v = B::Vec4::load(&out.a[j][2]);

            let x_1 = B::Vec4::splat_pair(x_l, x_r);
            let v3_1 = x_1.sub(ic2_s1);
            let v1_1 = a1_v.mul(ic1_s1).fma(a2_v, v3_1);
            let v2_1 = ic2_s1.fma(a2_v, ic1_s1).fma(a3_v, v3_1);
            ic1_s1 = flush_recursive_state_vec4(two.mul(v1_1).sub(ic1_s1));
            ic2_s1 = flush_recursive_state_vec4(two.mul(v2_1).sub(ic2_s1));
            let hp1 = x_1.sub(c.k_vec.mul(v1_1)).sub(v2_1);

            let lp1_lr = v2_1.dup_low_pair();
            let bp1_lr = v1_1.dup_low_pair();
            let hp1_lr = hp1.dup_low_pair();
            let u2_v = lp1_lr.fma(c.sec2_bp_v, bp1_lr).fma(c.sec2_hp_v, hp1_lr);

            let v3_2 = u2_v.sub(ic2_s2);
            let v1_2 = a1_v.mul(ic1_s2).fma(a2_v, v3_2);
            let v2_2 = ic2_s2.fma(a2_v, ic1_s2).fma(a3_v, v3_2);
            ic1_s2 = flush_recursive_state_vec4(two.mul(v1_2).sub(ic1_s2));
            ic2_s2 = flush_recursive_state_vec4(two.mul(v2_2).sub(ic2_s2));
            let hp2 = u2_v.sub(c.k_vec.mul(v1_2)).sub(v2_2);

            let tap1 = x_1
                .mul(c.tap_w[0])
                .fma(lp1_lr, c.tap_w[1])
                .fma(bp1_lr, c.tap_w[2])
                .fma(hp1_lr, c.tap_w[3]);
            let [us_l, us_r, d_l, d_r] = tap1
                .fma(v2_2.dup_high_pair(), c.tap_w[4])
                .fma(v1_2.dup_high_pair(), c.tap_w[5])
                .fma(hp2.dup_high_pair(), c.tap_w[6])
                .to_array();
            let [_, _, lp2_l, lp2_r] = v2_2.to_array();
            out.u_s2[j] = [us_l, us_r];
            out.svf_dir[j] = [d_l, d_r];
            out.lp2[j] = [lp2_l, lp2_r];
        }

        let [c1_1l, c1_1r, _, _] = ic1_s1.to_array();
        let [c2_1l, c2_1r, _, _] = ic2_s1.to_array();
        let [_, _, c1_2l, c1_2r] = ic1_s2.to_array();
        let [_, _, c2_2l, c2_2r] = ic2_s2.to_array();
        self.ic1_s1_l = c1_1l;
        self.ic1_s1_r = c1_1r;
        self.ic2_s1_l = c2_1l;
        self.ic2_s1_r = c2_1r;
        self.ic1_s2_l = c1_2l;
        self.ic1_s2_r = c1_2r;
        self.ic2_s2_l = c2_2l;
        self.ic2_s2_r = c2_2r;
    }

    #[inline(always)]
    fn step_stereo_core<B: MathBackend, const WITH_TAPS: bool>(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
        env_fc_scale: f32,
        env_k_delta: f32,
    ) -> ([f32; 2], [f32; 2], [[f32; 7]; 2]) {
        let lp_a = params.det_lp_pole;
        let lp_b = 1.0 - lp_a;
        self.det_lp_l = flush_recursive_state_f32(lp_a * self.det_lp_l + lp_b * x_l);
        self.det_lp_r = flush_recursive_state_f32(lp_a * self.det_lp_r + lp_b * x_r);
        let d_raw_l = (x_l + params.det_hp_mix * (x_l - self.det_lp_l)).abs();
        let d_raw_r = (x_r + params.det_hp_mix * (x_r - self.det_lp_r)).abs();

        let half_link = 0.5 * params.stereo_link;
        let diff = d_raw_r - d_raw_l;
        let det_l = d_raw_l + half_link * diff;
        let det_r = d_raw_r - half_link * diff;

        let alpha_l = if det_l > self.env_l {
            params.atk_alpha
        } else {
            params.rel_alpha
        };
        let alpha_r = if det_r > self.env_r {
            params.atk_alpha
        } else {
            params.rel_alpha
        };
        let env_l = flush_recursive_state_f32(alpha_l * self.env_l + (1.0 - alpha_l) * det_l);
        let env_r = flush_recursive_state_f32(alpha_r * self.env_r + (1.0 - alpha_r) * det_r);
        self.env_l = env_l;
        self.env_r = env_r;

        let one = B::Vec4::splat(1.0);
        let env_v = B::Vec4::splat_pair(env_l, env_r);
        let inv_e = one
            .fma(B::Vec4::splat(params.env_comp), env_v.mul(env_v))
            .rsqrt_pos_nr2();
        let oct_v = env_v.mul(inv_e).mul(B::Vec4::splat(params.env_oct_sens));
        let w1_v = B::Vec4::splat(params.base_g * env_fc_scale)
            .mul(oct_v.exp2_clamped5())
            .clamp(B::Vec4::splat(params.g_min), B::Vec4::splat(params.g_max));

        let g1_v = tan_prewarp_vec4::<B>(w1_v);
        let ratio_v = B::Vec4::from_array([1.0, 1.0, params.sec2_g_ratio, params.sec2_g_ratio]);
        let g_vec = g1_v
            .mul(ratio_v)
            .clamp(B::Vec4::splat(1.0e-4), B::Vec4::splat(SVF_G_MAX));

        let k1 = (params.damp_k1 + env_k_delta).clamp(0.05, 4.0);
        let k2 = (params.damp_k2 + env_k_delta).clamp(0.05, 4.0);
        let k_vec = B::Vec4::from_array([k1, k1, k2, k2]);
        let a1_v = one.fma(g_vec, g_vec.add(k_vec)).rcp_ge1_nr2();
        let a2_v = g_vec.mul(a1_v);
        let a3_v = g_vec.mul(a2_v);
        let [a1_1l, a1_1r, a1_2l, a1_2r] = a1_v.to_array();
        let [a2_1l, a2_1r, a2_2l, a2_2r] = a2_v.to_array();
        let [a3_1l, a3_1r, a3_2l, a3_2r] = a3_v.to_array();

        let (lp1_l, bp1_l, hp1_l) = Self::step_tpt_cached(
            x_l,
            a1_1l,
            a2_1l,
            a3_1l,
            k1,
            &mut self.ic1_s1_l,
            &mut self.ic2_s1_l,
        );
        let (lp1_r, bp1_r, hp1_r) = Self::step_tpt_cached(
            x_r,
            a1_1r,
            a2_1r,
            a3_1r,
            k1,
            &mut self.ic1_s1_r,
            &mut self.ic2_s1_r,
        );

        let u2_l = lp1_l + params.sec2_bp_feed * bp1_l + params.sec2_hp_feed * hp1_l;
        let u2_r = lp1_r + params.sec2_bp_feed * bp1_r + params.sec2_hp_feed * hp1_r;

        let (lp2_l, bp2_l, hp2_l) = Self::step_tpt_cached(
            u2_l,
            a1_2l,
            a2_2l,
            a3_2l,
            k2,
            &mut self.ic1_s2_l,
            &mut self.ic2_s2_l,
        );
        let (lp2_r, bp2_r, hp2_r) = Self::step_tpt_cached(
            u2_r,
            a1_2r,
            a2_2r,
            a3_2r,
            k2,
            &mut self.ic1_s2_r,
            &mut self.ic2_s2_r,
        );

        let tl0 = B::Vec4::from_array([x_l, lp1_l, bp1_l, hp1_l]);
        let tl1 = B::Vec4::from_array([lp2_l, bp2_l, hp2_l, 0.0]);
        let tr0 = B::Vec4::from_array([x_r, lp1_r, bp1_r, hp1_r]);
        let tr1 = B::Vec4::from_array([lp2_r, bp2_r, hp2_r, 0.0]);
        let s2_0 = B::Vec4::from_array([
            params.stage2_feed_taps[0],
            params.stage2_feed_taps[1],
            params.stage2_feed_taps[2],
            params.stage2_feed_taps[3],
        ]);
        let s2_1 = B::Vec4::from_array([
            params.stage2_feed_taps[4],
            params.stage2_feed_taps[5],
            params.stage2_feed_taps[6],
            0.0,
        ]);
        let dl_0 = B::Vec4::from_array([
            params.direct_taps_l[0],
            params.direct_taps_l[1],
            params.direct_taps_l[2],
            params.direct_taps_l[3],
        ]);
        let dl_1 = B::Vec4::from_array([
            params.direct_taps_l[4],
            params.direct_taps_l[5],
            params.direct_taps_l[6],
            0.0,
        ]);
        let dr_0 = B::Vec4::from_array([
            params.direct_taps_r[0],
            params.direct_taps_r[1],
            params.direct_taps_r[2],
            params.direct_taps_r[3],
        ]);
        let dr_1 = B::Vec4::from_array([
            params.direct_taps_r[4],
            params.direct_taps_r[5],
            params.direct_taps_r[6],
            0.0,
        ]);
        let (u_s2_l, u_s2_r) = tl0
            .mul(s2_0)
            .fma(tl1, s2_1)
            .horizontal_sum_pair(tr0.mul(s2_0).fma(tr1, s2_1));
        let (dir_l, dir_r) = tl0
            .mul(dl_0)
            .fma(tl1, dl_1)
            .horizontal_sum_pair(tr0.mul(dr_0).fma(tr1, dr_1));
        let taps = if WITH_TAPS {
            [
                [x_l, lp1_l, bp1_l, hp1_l, lp2_l, bp2_l, hp2_l],
                [x_r, lp1_r, bp1_r, hp1_r, lp2_r, bp2_r, hp2_r],
            ]
        } else {
            [[0.0; 7]; 2]
        };
        ([u_s2_l, u_s2_r], [dir_l, dir_r], taps)
    }

    /// Zero-lag full-rate stereo Stage I SVF returning `([u_s2_l, u_s2_r], [dir_l, dir_r], [taps_l, taps_r])`.
    #[inline(always)]
    pub fn step_stereo_fullrate_with_taps<B: MathBackend>(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
        env_fc_scale: f32,
        env_k_delta: f32,
    ) -> ([f32; 2], [f32; 2], [[f32; 7]; 2]) {
        self.step_stereo_core::<B, true>(params, x_l, x_r, env_fc_scale, env_k_delta)
    }

    /// Zero-lag full-rate stereo Stage I SVF fast path (`([u_s2_l, u_s2_r], [dir_l, dir_r])`).
    #[inline(always)]
    pub fn step_stereo_fullrate_fast<B: MathBackend>(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
        env_fc_scale: f32,
        env_k_delta: f32,
    ) -> ([f32; 2], [f32; 2]) {
        let (u_s2, direct, _) =
            self.step_stereo_core::<B, false>(params, x_l, x_r, env_fc_scale, env_k_delta);
        (u_s2, direct)
    }

    /// Zero-lag full-rate stereo Stage I SVF (`([u_s2_l, u_s2_r], [dir_l, dir_r])`).
    #[inline(always)]
    pub fn step_stereo_fullrate<B: MathBackend>(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
        env_fc_scale: f32,
        env_k_delta: f32,
    ) -> ([f32; 2], [f32; 2]) {
        self.step_stereo_fullrate_fast::<B>(params, x_l, x_r, env_fc_scale, env_k_delta)
    }

    /// Steps stereo Stage I SVF at full sample rate, returning `([u_s2_l, u_s2_r], [dir_l, dir_r], [taps_l, taps_r])`.
    #[inline(always)]
    pub fn step_stereo_with_taps(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
    ) -> ([f32; 2], [f32; 2], [[f32; 7]; 2]) {
        self.step_stereo_fullrate_with_taps::<crate::math::DefaultMathBackend>(
            params, x_l, x_r, 1.0, 0.0,
        )
    }

    /// Steps stereo Stage I SVF at full sample rate, returning `([u_s2_l, u_s2_r], [dir_l, dir_r])`.
    #[inline(always)]
    pub fn step_stereo(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
    ) -> ([f32; 2], [f32; 2]) {
        let (u_s2, direct, _) = self.step_stereo_with_taps(params, x_l, x_r);
        (u_s2, direct)
    }
}

#[cfg(test)]
mod tests {
    use super::super::coeffs::{
        algebraic_softclip_vec4, exp2_signed_oct_f32, recip_nr2_pos_f32, recip_nr2_pos_vec4,
        recip_nr3_pos_f32, recip_nr3_pos_vec4, rsqrt_nr2_f32, rsqrt_nr2_vec4, rsqrt_nr3_f32,
        rsqrt_nr3_vec4, tan_prewarp_f32, zero_dc_asym_sat_vec4,
    };
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_step_stereo_with_taps_matches_step_stereo_and_basis() {
        let mut params = V3Stage1SvfParams::identity_bypass();
        params.base_g = 0.04;
        params.sec2_g_ratio = 1.4;
        params.env_oct_sens = 3.2;
        params.stage2_feed_taps = [0.25, 0.4, 0.3, -0.1, 0.2, 0.15, -0.05, 0.0];
        params.direct_taps_l = [0.1, -0.2, 0.35, 0.05, 0.15, -0.1, 0.2, 0.0];
        params.direct_taps_r = [-0.05, 0.25, -0.15, 0.3, 0.1, 0.2, -0.1, 0.0];
        params.clamp_in_place();

        let (mut st_a, mut st_b) = (V3Stage1SvfState::default(), V3Stage1SvfState::default());
        for i in 0..32 {
            let xl = 0.35 * ((i as f32) * 0.19).sin();
            let xr = -0.28 * ((i as f32) * 0.23).cos();
            let (ua, da) = st_a.step_stereo(&params, xl, xr);
            let (ub, db, [tl, tr]) = st_b.step_stereo_with_taps(&params, xl, xr);
            assert_eq!(ua[0].to_bits(), ub[0].to_bits());
            assert_eq!(ua[1].to_bits(), ub[1].to_bits());
            assert_eq!(da[0].to_bits(), db[0].to_bits());
            assert_eq!(da[1].to_bits(), db[1].to_bits());
            assert_eq!(tl[0], xl);
            assert_eq!(tr[0], xr);
            let recon_ul: f32 = (0..7).map(|k| tl[k] * params.stage2_feed_taps[k]).sum();
            let recon_dr: f32 = (0..7).map(|k| tr[k] * params.direct_taps_r[k]).sum();
            assert!((recon_ul - ub[0]).abs() < 1.0e-7);
            assert!((recon_dr - db[1]).abs() < 1.0e-7);
        }
    }

    #[test]
    fn test_svf_primitives_vec4_backends_and_reset() {
        use crate::math::{DefaultMathBackend, ScalarBackend};

        assert!((recip_nr2_pos_f32(8.0) - 0.125).abs() < 1.0e-5);
        assert!((recip_nr3_pos_f32(8.0) - 0.125).abs() < 1.0e-6);
        assert!((rsqrt_nr2_f32(9.0) - (1.0 / 3.0)).abs() < 1.0e-3);
        assert!((rsqrt_nr3_f32(9.0) - (1.0 / 3.0)).abs() < 1.0e-5);
        assert!(exp2_signed_oct_f32(-4.0) > 0.0 && exp2_signed_oct_f32(4.0) > 1.0);
        assert!(tan_prewarp_f32(0.05) > 0.0 && tan_prewarp_f32(1.4) > 0.0);

        let vx_s = <ScalarBackend as MathBackend>::Vec4::load(&[-2.0, -0.5, 0.5, 2.0]);
        let bias_s = <ScalarBackend as MathBackend>::Vec4::splat(0.15);
        let bias0_s = algebraic_softclip_vec4::<ScalarBackend>(bias_s);
        let _ = (
            recip_nr2_pos_vec4::<ScalarBackend>(<ScalarBackend as MathBackend>::Vec4::splat(2.0)),
            recip_nr3_pos_vec4::<ScalarBackend>(<ScalarBackend as MathBackend>::Vec4::splat(2.0)),
            rsqrt_nr2_vec4::<ScalarBackend>(<ScalarBackend as MathBackend>::Vec4::splat(4.0)),
            rsqrt_nr3_vec4::<ScalarBackend>(<ScalarBackend as MathBackend>::Vec4::splat(4.0)),
            zero_dc_asym_sat_vec4::<ScalarBackend>(vx_s, bias_s, bias0_s),
        );
        let vx_d = <DefaultMathBackend as MathBackend>::Vec4::load(&[-2.0, -0.5, 0.5, 2.0]);
        let bias_d = <DefaultMathBackend as MathBackend>::Vec4::splat(0.15);
        let bias0_d = algebraic_softclip_vec4::<DefaultMathBackend>(bias_d);
        let _ = (
            recip_nr2_pos_vec4::<DefaultMathBackend>(
                <DefaultMathBackend as MathBackend>::Vec4::splat(2.0),
            ),
            recip_nr3_pos_vec4::<DefaultMathBackend>(
                <DefaultMathBackend as MathBackend>::Vec4::splat(2.0),
            ),
            rsqrt_nr2_vec4::<DefaultMathBackend>(<DefaultMathBackend as MathBackend>::Vec4::splat(
                4.0,
            )),
            rsqrt_nr3_vec4::<DefaultMathBackend>(<DefaultMathBackend as MathBackend>::Vec4::splat(
                4.0,
            )),
            zero_dc_asym_sat_vec4::<DefaultMathBackend>(vx_d, bias_d, bias0_d),
        );

        let mut svf_params = V3Stage1SvfParams {
            base_g: 0.12,
            sec2_g_ratio: 1.4,
            env_oct_sens: 2.0,
            det_hp_mix: 0.3,
            sec2_bp_feed: 0.25,
            sec2_hp_feed: 0.15,
            direct_taps_l: [0.1; 8],
            direct_taps_r: [0.1; 8],
            ..V3Stage1SvfParams::identity_bypass()
        };
        svf_params.clamp_in_place();
        let mut svf_state = V3Stage1SvfState::default();
        let (u_s2, direct, taps) = svf_state.step_stereo_with_taps(&svf_params, 0.4, -0.3);
        assert!(u_s2[0].is_finite() && direct[0].is_finite() && taps[0][0].is_finite());
        let (sl_pair, sr_pair) = svf_state.step_stereo(&svf_params, 0.3, -0.3);
        assert!(sl_pair[0].is_finite() && sr_pair[0].is_finite());
        svf_state.reset();
        assert_eq!(svf_state, V3Stage1SvfState::default());

        let hoisted = SvfHoistedConsts::<<DefaultMathBackend as MathBackend>::Vec4>::from_params(
            &svf_params,
            1.0,
            0.0,
        );
        let mut s_ref = V3Stage1SvfState::default();
        let mut s_fast = V3Stage1SvfState::default();
        let mut in_buf = [[0.0_f32; 2]; 16];
        for (i, pair) in in_buf.iter_mut().enumerate() {
            let t = i as f32;
            *pair = [0.45 * (t * 0.21).sin(), -0.35 * (t * 0.17).cos()];
        }
        let mut env_buf = [[0.0_f32; 2]; 16];
        let mut a_buf = [[[0.0_f32; 4]; 3]; 16];
        let mut u_fast = [[0.0_f32; 2]; 16];
        let mut d_fast = [[0.0_f32; 2]; 16];
        let mut lp2_fast = [[0.0_f32; 2]; 16];
        let mut sb_out = SvfSubblockOut {
            env: &mut env_buf,
            a: &mut a_buf,
            u_s2: &mut u_fast,
            svf_dir: &mut d_fast,
            lp2: &mut lp2_fast,
        };
        s_fast.step_subblock_fast::<DefaultMathBackend, 16>(&hoisted, &in_buf, &mut sb_out, 15);
        for i in 0..15 {
            let (u_ref, d_ref) = s_ref.step_stereo_fullrate_fast::<DefaultMathBackend>(
                &svf_params,
                in_buf[i][0],
                in_buf[i][1],
                1.0,
                0.0,
            );
            assert!((sb_out.u_s2[i][0] - u_ref[0]).abs() < 1.0e-5);
            assert!((sb_out.u_s2[i][1] - u_ref[1]).abs() < 1.0e-5);
            assert!((sb_out.svf_dir[i][0] - d_ref[0]).abs() < 1.0e-5);
            assert!((sb_out.svf_dir[i][1] - d_ref[1]).abs() < 1.0e-5);
        }
        assert!((s_fast.env_l - s_ref.env_l).abs() < 1.0e-6);
        assert!((s_fast.ic2_s2_r - s_ref.ic2_s2_r).abs() < 1.0e-5);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_stage1_svf_req7_unclipped_100_envelope(
            in_l in -100.0_f32..100.0_f32,
            in_r in -100.0_f32..100.0_f32,
            env_oct in -6.0_f32..6.0_f32,
        ) {
            use crate::math::DefaultMathBackend;

            let mut params = V3Stage1SvfParams {
                base_g: 0.15,
                env_oct_sens: env_oct,
                direct_taps_l: [1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                direct_taps_r: [1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ..V3Stage1SvfParams::identity_bypass()
            };
            params.clamp_in_place();
            let hoisted = SvfHoistedConsts::<<DefaultMathBackend as MathBackend>::Vec4>::from_params(
                &params,
                1.0,
                0.0,
            );
            let mut st = V3Stage1SvfState::default();
            let in_buf = [[in_l, in_r]; 4];
            let mut env_buf = [[0.0_f32; 2]; 4];
            let mut a_buf = [[[0.0_f32; 4]; 3]; 4];
            let mut u_buf = [[0.0_f32; 2]; 4];
            let mut d_buf = [[0.0_f32; 2]; 4];
            let mut lp2_buf = [[0.0_f32; 2]; 4];
            let mut sb_out = SvfSubblockOut {
                env: &mut env_buf,
                a: &mut a_buf,
                u_s2: &mut u_buf,
                svf_dir: &mut d_buf,
                lp2: &mut lp2_buf,
            };
            st.step_subblock_fast::<DefaultMathBackend, 4>(&hoisted, &in_buf, &mut sb_out, 4);
            prop_assert!((sb_out.u_s2[0][0] - in_l).abs() < 1.0e-5);
            prop_assert!((sb_out.u_s2[0][1] - in_r).abs() < 1.0e-5);
            prop_assert!(sb_out.svf_dir[3][0].is_finite() && sb_out.svf_dir[3][1].is_finite());
        }
    }
}
