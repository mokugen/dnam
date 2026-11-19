//! Multi-timescale input feature extractors & ZOH scaling for [`super::block::V3GruBlockHot`].

use super::block::NUM_IN_FEATS_B0;
use super::math_pade::recip_pos_ge1_f32;
use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};
use crate::stages::stage1_svf::rsqrt_nr3_f32;

/// Multi-timescale input feature extractor parameters stored in precision-preserving `beta = 1 - alpha`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3InputFeaturesParams {
    /// High-pass 1-pole tracker step `beta in [1e-4, 0.50]` (`s_lp += in_hp_beta * (u - s_lp)`).
    pub in_hp_beta: f32,
    /// Fast rectified envelope attack step `beta in [1e-4, 0.999]`.
    pub fast_atk_beta: f32,
    /// Fast rectified envelope release step `beta in [1e-5, 0.999]`.
    pub fast_rel_beta: f32,
    /// Quadratic pinch-off gate sharpness `>= 0.0` (`u * (1 - 1 / (1 + pinch_k * env_fast^2))`).
    pub pinch_k: f32,
    /// Slow power-supply sag envelope attack step `beta in [1e-5, 0.50]`.
    pub slow_atk_beta: f32,
    /// Slow power-supply sag envelope release step `beta in [1e-6, 0.10]`.
    pub slow_rel_beta: f32,
    /// Preamp-compressed power-supply sag detector saturation `c_det in [0.0, 64.0]`
    /// (`rect_slow = |u| / sqrt(1 + slow_det_comp * u^2)`).
    pub slow_det_comp: f32,
    /// Stereo key-link detector blend `key_link in [0.0, 1.0]` (`0.0` independent, `1.0` Aux-Key cross-keyed).
    pub key_link: f32,
    /// Detector amplitude deadzone threshold `det_thresh in [0.0, 2.0]` (`max(0, |u| - det_thresh)`).
    pub det_thresh: f32,
}

impl Default for V3InputFeaturesParams {
    fn default() -> Self {
        Self {
            in_hp_beta: 0.045,
            fast_atk_beta: 0.035,
            fast_rel_beta: 0.001,
            pinch_k: 0.0,
            slow_atk_beta: 0.0025,
            slow_rel_beta: 8.5e-6,
            slow_det_comp: 0.0,
            key_link: 0.0,
            det_thresh: 0.0,
        }
    }
}

impl V3InputFeaturesParams {
    /// Clamps filter and envelope step coefficients into unconditional BIBO bounds.
    pub fn clamp_in_place(&mut self) {
        self.in_hp_beta = self.in_hp_beta.clamp(1.0e-4, 0.50);
        self.fast_atk_beta = self.fast_atk_beta.clamp(1.0e-4, 0.999);
        self.fast_rel_beta = self.fast_rel_beta.clamp(1.0e-5, 0.999);
        self.pinch_k = self.pinch_k.clamp(0.0, 25_000.0);
        self.slow_atk_beta = self.slow_atk_beta.clamp(1.0e-5, 0.50);
        self.slow_rel_beta = self.slow_rel_beta.clamp(1.0e-6, 0.10);
        self.slow_det_comp = self.slow_det_comp.clamp(0.0, 64.0);
        self.key_link = self.key_link.clamp(0.0, 1.0);
        self.det_thresh = self.det_thresh.clamp(0.0, 2.0);
    }

    /// Rescales `2x` (`96 kHz`) ZOH step rates to `1x` (`48 kHz`): `beta_1x = beta * (2 - beta)`.
    pub fn rescale_2x_to_1x(&mut self) {
        self.in_hp_beta = (self.in_hp_beta * (2.0 - self.in_hp_beta)).clamp(1.0e-4, 0.50);
        self.fast_atk_beta = (self.fast_atk_beta * (2.0 - self.fast_atk_beta)).clamp(1.0e-4, 0.999);
        self.fast_rel_beta = (self.fast_rel_beta * (2.0 - self.fast_rel_beta)).clamp(1.0e-5, 0.999);
        self.slow_atk_beta = (self.slow_atk_beta * (2.0 - self.slow_atk_beta)).clamp(1.0e-5, 0.50);
        self.slow_rel_beta = (self.slow_rel_beta * (2.0 - self.slow_rel_beta)).clamp(1.0e-6, 0.10);
        self.clamp_in_place();
    }

    /// Rescales `beta = 1 - alpha` detector step rates under continuous-time ZOH (`alpha_eff = alpha^rho`).
    pub fn rescale_with_zoh(&mut self, zoh: &ZohRescaledParams) {
        self.clamp_in_place();
        if (zoh.rho_sr - 1.0).abs() <= 1.0e-7 {
            return;
        }
        let scale_beta = |beta: f32, lo: f32, hi: f32| -> f32 {
            let alpha = (1.0 - f64::from(beta)).clamp(1.0e-6, 0.999999);
            let alpha_eff = crate::schema::cold_pow_f64(alpha, zoh.rho_sr);
            (1.0 - alpha_eff as f32).clamp(lo, hi)
        };
        self.in_hp_beta = scale_beta(self.in_hp_beta, 1.0e-4, 0.50);
        self.fast_atk_beta = scale_beta(self.fast_atk_beta, 1.0e-4, 0.999);
        self.fast_rel_beta = scale_beta(self.fast_rel_beta, 1.0e-5, 0.999);
        self.slow_atk_beta = scale_beta(self.slow_atk_beta, 1.0e-5, 0.50);
        self.slow_rel_beta = scale_beta(self.slow_rel_beta, 1.0e-6, 0.10);
        self.clamp_in_place();
    }
}

/// Per-channel multi-timescale input feature recursive state (`16 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3InputFeaturesState {
    pub u_lp: f32,
    pub env_fast: f32,
    pub env_slow: f32,
    pub _pad: f32,
}

/// Packed `(f_vec0, f_vec1, slow_vec0, slow_vec1, u_hp)` stereo feature pair.
pub type StereoPairFeatures<V> = ([V; 2], [V; 2], [V; 2], [V; 2], [[f32; 2]; 2]);

impl V3InputFeaturesState {
    /// Resets recursive input feature states to exact resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline(always)]
    fn step_u_hp(&mut self, p: &V3InputFeaturesParams, u: f32) -> f32 {
        self.u_lp = flush_recursive_state_f32(self.u_lp + p.in_hp_beta * (u - self.u_lp));
        u - self.u_lp
    }

    #[inline(always)]
    fn step_u_hp_with_lp2(
        &mut self,
        p: &V3InputFeaturesParams,
        u: f32,
        lp2: f32,
        lp2_mix: f32,
    ) -> f32 {
        self.u_lp = flush_recursive_state_f32(self.u_lp + p.in_hp_beta * (u - self.u_lp));
        let eff_lp = if lp2_mix > 0.0 {
            self.u_lp + lp2_mix * (lp2 - self.u_lp)
        } else {
            self.u_lp
        };
        u - eff_lp
    }

    #[inline(always)]
    fn step_env_fast(&mut self, p: &V3InputFeaturesParams, rect: f32) -> f32 {
        let fb = if rect > self.env_fast {
            p.fast_atk_beta
        } else {
            p.fast_rel_beta
        };
        let ef = flush_recursive_state_f32(self.env_fast + fb * (rect - self.env_fast));
        self.env_fast = ef;
        ef
    }

    #[inline(always)]
    fn key_link_rects(p: &V3InputFeaturesParams, u_l: f32, u_r: f32) -> (f32, f32) {
        let al = u_l.abs();
        let ar = u_r.abs();
        let kl = p.key_link;
        let (rl, rr) = if kl <= 0.0 {
            (al, ar)
        } else {
            ((1.0 - kl) * al + kl * ar, (1.0 - kl) * ar + kl * al)
        };
        let th = p.det_thresh;
        if th <= 0.0 {
            (rl, rr)
        } else {
            ((rl - th).max(0.0), (rr - th).max(0.0))
        }
    }

    #[inline(always)]
    fn step_env_slow(&mut self, p: &V3InputFeaturesParams, rect_slow: f32) -> f32 {
        let sb = if rect_slow > self.env_slow {
            p.slow_atk_beta
        } else {
            p.slow_rel_beta
        };
        let es = flush_recursive_state_f32(self.env_slow + sb * (rect_slow - self.env_slow));
        self.env_slow = es;
        es
    }

    /// Advances input feature detectors by one sample and returns `[u, u_hp, u_pinch, env_fast, env_slow]`.
    #[inline(always)]
    pub fn step(&mut self, p: &V3InputFeaturesParams, u: f32) -> [f32; NUM_IN_FEATS_B0] {
        let u_hp = self.step_u_hp(p, u);
        let raw = u.abs();
        let rect = if p.det_thresh <= 0.0 {
            raw
        } else {
            (raw - p.det_thresh).max(0.0)
        };
        let env_fast = self.step_env_fast(p, rect);
        let rect_slow = if p.slow_det_comp > 0.0 {
            rect * rsqrt_nr3_f32(1.0 + p.slow_det_comp * rect * rect)
        } else {
            rect
        };
        let env_slow = self.step_env_slow(p, rect_slow);
        let u_pinch = if p.pinch_k > 0.0 {
            let g_pinch = 1.0 - recip_pos_ge1_f32(1.0 + p.pinch_k * env_fast * env_fast);
            u * g_pinch
        } else {
            0.0
        };
        [u, u_hp, u_pinch, env_fast, env_slow]
    }

    #[inline(always)]
    fn step_stereo_inner<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u_l: f32,
        u_r: f32,
        lp2: [f32; 2],
        lp2_mix: f32,
    ) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
        let u_hp_l = st_l.step_u_hp_with_lp2(p, u_l, lp2[0], lp2_mix);
        let u_hp_r = st_r.step_u_hp_with_lp2(p, u_r, lp2[1], lp2_mix);
        let (rect_l, rect_r) = Self::key_link_rects(p, u_l, u_r);
        let ef_l = st_l.step_env_fast(p, rect_l);
        let ef_r = st_r.step_env_fast(p, rect_r);

        let pinch_num_l = p.pinch_k * ef_l * ef_l;
        let pinch_num_r = p.pinch_k * ef_r * ef_r;
        let inv = B::Vec4::from_array([
            1.0 + p.slow_det_comp * rect_l * rect_l,
            1.0 + p.slow_det_comp * rect_r * rect_r,
            1.0 + pinch_num_l,
            1.0 + pinch_num_r,
        ])
        .rsqrt_pos_nr2()
        .to_array();

        let es_l = st_l.step_env_slow(p, rect_l * inv[0]);
        let es_r = st_r.step_env_slow(p, rect_r * inv[1]);
        let u_pinch_l = (u_l * pinch_num_l) * (inv[2] * inv[2]);
        let u_pinch_r = (u_r * pinch_num_r) * (inv[3] * inv[3]);
        (u_hp_l, u_hp_r, u_pinch_l, u_pinch_r, ef_l, ef_r, es_l, es_r)
    }

    /// Vectorized branchless stereo input-feature step (`1` in-register `rsqrt_pos_nr2` for both channels).
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u_l: f32,
        u_r: f32,
    ) -> ([f32; NUM_IN_FEATS_B0], [f32; NUM_IN_FEATS_B0]) {
        Self::step_stereo_with_lp2::<B>(st_l, st_r, p, u_l, u_r, [0.0; 2], 0.0)
    }

    /// Vectorized stereo input-feature step blending `Stage I lp2` into the low-pass subtracted `u_hp` path.
    #[inline(always)]
    pub fn step_stereo_with_lp2<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u_l: f32,
        u_r: f32,
        lp2: [f32; 2],
        lp2_mix: f32,
    ) -> ([f32; NUM_IN_FEATS_B0], [f32; NUM_IN_FEATS_B0]) {
        let (u_hp_l, u_hp_r, u_pinch_l, u_pinch_r, ef_l, ef_r, es_l, es_r) =
            Self::step_stereo_inner::<B>(st_l, st_r, p, u_l, u_r, lp2, lp2_mix);
        (
            [u_l, u_hp_l, u_pinch_l, ef_l, es_l],
            [u_r, u_hp_r, u_pinch_r, ef_r, es_r],
        )
    }

    /// Vectorized stereo input-feature step packing `[u, u_hp, u_pinch, env_fast]` into `Vec4` registers.
    #[inline(always)]
    pub fn step_stereo_vec4<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u_l: f32,
        u_r: f32,
    ) -> (B::Vec4, B::Vec4, B::Vec4, B::Vec4, f32, f32) {
        Self::step_stereo_with_lp2_vec4::<B>(st_l, st_r, p, u_l, u_r, [0.0; 2], 0.0)
    }

    /// Vectorized stereo input-feature step with `Stage I lp2` blend packing into `Vec4` registers.
    #[inline(always)]
    pub fn step_stereo_with_lp2_vec4<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u_l: f32,
        u_r: f32,
        lp2: [f32; 2],
        lp2_mix: f32,
    ) -> (B::Vec4, B::Vec4, B::Vec4, B::Vec4, f32, f32) {
        let (u_hp_l, u_hp_r, u_pinch_l, u_pinch_r, ef_l, ef_r, es_l, es_r) =
            Self::step_stereo_inner::<B>(st_l, st_r, p, u_l, u_r, lp2, lp2_mix);
        (
            B::Vec4::from_array([u_l, u_hp_l, u_pinch_l, ef_l]),
            B::Vec4::from_array([u_r, u_hp_r, u_pinch_r, ef_r]),
            B::Vec4::splat(es_l),
            B::Vec4::splat(es_r),
            u_hp_l,
            u_hp_r,
        )
    }

    /// Dual-sample (`t`, `t+1`) stereo input-feature step issuing two interleaved `rsqrt_pos_pair_nr2` ops.
    #[inline(always)]
    pub fn step_stereo_pair_vec4<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u0: [f32; 2],
        u1: [f32; 2],
    ) -> StereoPairFeatures<B::Vec4> {
        Self::step_stereo_pair_with_lp2_vec4::<B>(st_l, st_r, p, u0, u1, [0.0; 2], [0.0; 2], 0.0)
    }

    /// Dual-sample (`t`, `t+1`) stereo input-feature step with `Stage I lp2` blending.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "Dual-sample vectorized stereo feature evaluation keeps 4 rsqrt lanes batched"
    )]
    pub fn step_stereo_pair_with_lp2_vec4<B: MathBackend>(
        st_l: &mut Self,
        st_r: &mut Self,
        p: &V3InputFeaturesParams,
        u0: [f32; 2],
        u1: [f32; 2],
        lp2_0: [f32; 2],
        lp2_1: [f32; 2],
        lp2_mix: f32,
    ) -> StereoPairFeatures<B::Vec4> {
        let hp_l0 = st_l.step_u_hp_with_lp2(p, u0[0], lp2_0[0], lp2_mix);
        let hp_r0 = st_r.step_u_hp_with_lp2(p, u0[1], lp2_0[1], lp2_mix);
        let (rect_l0, rect_r0) = Self::key_link_rects(p, u0[0], u0[1]);
        let ef_l0 = st_l.step_env_fast(p, rect_l0);
        let ef_r0 = st_r.step_env_fast(p, rect_r0);

        let hp_l1 = st_l.step_u_hp_with_lp2(p, u1[0], lp2_1[0], lp2_mix);
        let hp_r1 = st_r.step_u_hp_with_lp2(p, u1[1], lp2_1[1], lp2_mix);
        let (rect_l1, rect_r1) = Self::key_link_rects(p, u1[0], u1[1]);
        let ef_l1 = st_l.step_env_fast(p, rect_l1);
        let ef_r1 = st_r.step_env_fast(p, rect_r1);

        let pn_l0 = p.pinch_k * ef_l0 * ef_l0;
        let pn_r0 = p.pinch_k * ef_r0 * ef_r0;
        let pn_l1 = p.pinch_k * ef_l1 * ef_l1;
        let pn_r1 = p.pinch_k * ef_r1 * ef_r1;

        let (inv_s, inv_p) = B::Vec4::rsqrt_pos_pair_nr2(
            B::Vec4::from_array([
                1.0 + p.slow_det_comp * rect_l0 * rect_l0,
                1.0 + p.slow_det_comp * rect_r0 * rect_r0,
                1.0 + p.slow_det_comp * rect_l1 * rect_l1,
                1.0 + p.slow_det_comp * rect_r1 * rect_r1,
            ]),
            B::Vec4::from_array([1.0 + pn_l0, 1.0 + pn_r0, 1.0 + pn_l1, 1.0 + pn_r1]),
        );
        let is = inv_s.to_array();
        let ip = inv_p.to_array();

        let es_l0 = st_l.step_env_slow(p, rect_l0 * is[0]);
        let es_r0 = st_r.step_env_slow(p, rect_r0 * is[1]);
        let es_l1 = st_l.step_env_slow(p, rect_l1 * is[2]);
        let es_r1 = st_r.step_env_slow(p, rect_r1 * is[3]);

        let pinch_l0 = (u0[0] * pn_l0) * (ip[0] * ip[0]);
        let pinch_r0 = (u0[1] * pn_r0) * (ip[1] * ip[1]);
        let pinch_l1 = (u1[0] * pn_l1) * (ip[2] * ip[2]);
        let pinch_r1 = (u1[1] * pn_r1) * (ip[3] * ip[3]);

        (
            [
                B::Vec4::from_array([u0[0], hp_l0, pinch_l0, ef_l0]),
                B::Vec4::from_array([u0[1], hp_r0, pinch_r0, ef_r0]),
            ],
            [
                B::Vec4::from_array([u1[0], hp_l1, pinch_l1, ef_l1]),
                B::Vec4::from_array([u1[1], hp_r1, pinch_r1, ef_r1]),
            ],
            [B::Vec4::splat(es_l0), B::Vec4::splat(es_r0)],
            [B::Vec4::splat(es_l1), B::Vec4::splat(es_r1)],
            [[hp_l0, hp_r0], [hp_l1, hp_r1]],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_v3_input_features_stereo_matches_scalar_zero_dc_and_zoh() {
        let mut p = V3InputFeaturesParams {
            pinch_k: 450.0,
            slow_det_comp: 12.0,
            ..V3InputFeaturesParams::default()
        };
        p.clamp_in_place();
        let (mut sl_z, mut sr_z) = (
            V3InputFeaturesState::default(),
            V3InputFeaturesState::default(),
        );
        let (z_l, z_r) = V3InputFeaturesState::step_stereo::<DefaultMathBackend>(
            &mut sl_z, &mut sr_z, &p, 0.0, 0.0,
        );
        assert_eq!(z_l.map(f32::to_bits), [0; NUM_IN_FEATS_B0]);
        assert_eq!(z_r.map(f32::to_bits), [0; NUM_IN_FEATS_B0]);

        let (mut stl, mut str_r) = (
            V3InputFeaturesState::default(),
            V3InputFeaturesState::default(),
        );
        let (mut scl, mut scr) = (
            V3InputFeaturesState::default(),
            V3InputFeaturesState::default(),
        );
        for n in 0..64 {
            let ul = 0.7 * (n as f32 * 0.17).sin();
            let ur = -0.35 * (n as f32 * 0.11).cos();
            let (fl, fr) = V3InputFeaturesState::step_stereo::<DefaultMathBackend>(
                &mut stl, &mut str_r, &p, ul, ur,
            );
            let rl = scl.step(&p, ul);
            let rr = scr.step(&p, ur);
            for k in 0..NUM_IN_FEATS_B0 {
                assert!((fl[k] - rl[k]).abs() < 1.0e-5, "L feat {k}");
                assert!((fr[k] - rr[k]).abs() < 1.0e-5, "R feat {k}");
            }
        }

        let zoh = ZohRescaledParams::compute(48_000, 96_000, 0.9993, 0.995, 64.0, 128).unwrap();
        let mut pz = p;
        pz.rescale_with_zoh(&zoh);
        assert!(pz.in_hp_beta < p.in_hp_beta);

        let mut p_key = V3InputFeaturesParams {
            key_link: 1.0,
            ..V3InputFeaturesParams::default()
        };
        p_key.clamp_in_place();
        let (mut kl, mut kr) = (
            V3InputFeaturesState::default(),
            V3InputFeaturesState::default(),
        );
        let (fl, fr) = V3InputFeaturesState::step_stereo::<DefaultMathBackend>(
            &mut kl, &mut kr, &p_key, 0.0, 0.8,
        );
        assert_eq!(fl[0].to_bits(), 0);
        assert_eq!(fl[1].to_bits(), 0);
        assert!(fl[3] > 0.02, "Aux-Key R drives L env_fast when key_link=1");
        assert_eq!(
            fr[3].to_bits(),
            0,
            "L silence gives 0 on R env_fast when key_link=1"
        );

        let p_th = V3InputFeaturesParams {
            det_thresh: 0.35,
            ..p_key
        };
        let (mut tl, mut tr) = (
            V3InputFeaturesState::default(),
            V3InputFeaturesState::default(),
        );
        let (below_l, _) = V3InputFeaturesState::step_stereo::<DefaultMathBackend>(
            &mut tl, &mut tr, &p_th, 0.2, 0.2,
        );
        assert_eq!(below_l[3].to_bits(), 0, "below det_thresh yields 0 env");
        let (lp2_out, _) = V3InputFeaturesState::step_stereo_with_lp2::<DefaultMathBackend>(
            &mut tl,
            &mut tr,
            &p_th,
            1.0,
            1.0,
            [0.25, 0.25],
            1.0,
        );
        assert!((lp2_out[1] - 0.75).abs() < 1.0e-6);
    }
}
