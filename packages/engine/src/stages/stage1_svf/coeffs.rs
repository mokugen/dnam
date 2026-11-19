//! Stage I Cytomic TPT/ZDF SVF Parameter Block, Sub-Block Hoisted Constants, ZOH Rescaling,
//! and Division-Free Real-Time Transcendental Helpers.

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, floor_f32};

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using IEEE-754 integer seed
/// `0x7EF3_11C3` followed by 2 Newton–Raphson iterations (`0 FDIV`, `< 6.6e-6` relative error).
#[inline(always)]
#[must_use]
pub fn recip_nr2_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(1.0e-6);
    let mut y = f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(d_safe.to_bits()));
    y = y * (2.0 - d_safe * y);
    y * (2.0 - d_safe * y)
}

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using IEEE-754 integer seed
/// `IEEE754_F32_RCP_SEED` (`0x7EF3_11C3`) followed by 3 Newton–Raphson iterations (`0 FDIV`, `< 1 ULP` over normal range).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(1.0e-6);
    let mut y = f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(d_safe.to_bits()));
    y = y * (2.0 - d_safe * y);
    y = y * (2.0 - d_safe * y);
    y * (2.0 - d_safe * y)
}

/// 4-lane SIMD positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using in-register
/// IEEE-754 integer seed + 2 SIMD FMA/FMS Newton steps (`0 FDIV`, `0` GPR spills).
#[inline(always)]
#[must_use]
pub fn recip_nr2_pos_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rcp_pos_nr2()
}

/// 4-lane SIMD positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using in-register
/// IEEE-754 integer seed + 3 SIMD FMA/FMS Newton steps (`0 FDIV`, `0` GPR spills).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rcp_pos_nr3()
}

/// Fast inverse square root `1.0 / sqrt(d)` (`d` clamped to `>= 1.0e-12`) using Quake III /
/// IEEE-754 seed `0x5F37_59DF` followed by 2 Newton–Raphson iterations (`0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr2_f32(d: f32) -> f32 {
    let d_safe = d.max(1.0e-12);
    let half_d = 0.5 * d_safe;
    let mut y =
        f32::from_bits(crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(d_safe.to_bits() >> 1));
    y = y * (1.5 - half_d * y * y);
    y * (1.5 - half_d * y * y)
}

/// Fast inverse square root `1.0 / sqrt(d)` (`d` clamped to `>= 1.0e-12`) using
/// `IEEE754_F32_RSQRT_SEED` (`0x5F37_59DF`) followed by 3 Newton–Raphson iterations (`0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_f32(d: f32) -> f32 {
    let d_safe = d.max(1.0e-12);
    let half_d = 0.5 * d_safe;
    let mut y =
        f32::from_bits(crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(d_safe.to_bits() >> 1));
    y = y * (1.5 - half_d * y * y);
    y = y * (1.5 - half_d * y * y);
    y * (1.5 - half_d * y * y)
}

/// 4-lane SIMD inverse square root `1.0 / sqrt(d)` using in-register integer seed + 2 Newton steps.
#[inline(always)]
#[must_use]
pub fn rsqrt_nr2_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rsqrt_pos_nr2()
}

/// 4-lane SIMD inverse square root `1.0 / sqrt(d)` using in-register integer seed + 3 Newton steps.
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rsqrt_pos_nr3()
}

/// Evaluates $z / \sqrt{1 + z^2}$ clamped safely to avoid overflow on 4 SIMD lanes.
#[inline(always)]
#[must_use]
pub fn algebraic_softclip_vec4<B: MathBackend>(z: B::Vec4) -> B::Vec4 {
    let zc = z.clamp(B::Vec4::splat(-64.0), B::Vec4::splat(64.0));
    let denom = B::Vec4::splat(1.0).fma(zc, zc);
    zc.mul(rsqrt_nr3_vec4::<B>(denom))
}

/// Zero-DC asymmetric saturator: $\phi(x + b) - \phi(b_0)$ where $\phi(z) = z / \sqrt{1 + z^2}$.
///
/// Precomputing `bias0_phi = algebraic_softclip_vec4::<B>(static_bias)` per backend guarantees bit-exact
/// `+0.0` at `(x = 0, b = static_bias)`.
#[inline(always)]
#[must_use]
pub fn zero_dc_asym_sat_vec4<B: MathBackend>(
    x: B::Vec4,
    bias: B::Vec4,
    bias0_phi: B::Vec4,
) -> B::Vec4 {
    algebraic_softclip_vec4::<B>(x.add(bias)).sub(bias0_phi)
}

/// $C^2$-continuous octave-warping curve approximating $2^{\text{oct}}$ for `oct in [-5.0, 5.0]`
/// using IEEE exponent construction + degree-5 Hermite-matched fractional polynomial (`0 FDIV`, `0 libm`).
#[expect(
    clippy::approx_constant,
    reason = "Hermite fractional exp2 polynomial constant matches ln(2) without libm"
)]
#[inline(always)]
#[must_use]
pub fn exp2_signed_oct_f32(oct: f32) -> f32 {
    const C1: f32 = 0.693_147_2;
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_712_965;
    const C4: f32 = 0.009_011_465;
    const C5: f32 = 0.001_901_882;

    let x = oct.clamp(-5.0, 5.0);
    let fl = floor_f32(x);
    let n_i32 = fl as i32;
    let f = (x - fl).clamp(0.0, 1.0);
    let poly = 1.0 + f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))));
    let exp_bits = ((127 + n_i32) as u32) << 23;
    f32::from_bits(exp_bits) * poly
}

/// Maximum Stage I normalized half-angle `w_half = pi * f0 / fs` supported across `1x`/`2x`.
pub const SVF_W_MAX: f32 = 1.38;
/// Maximum prewarped Stage I integrator coefficient `g = tan(w_half)`.
pub const SVF_G_MAX: f32 = 4.5;

/// 7th-order division-free bilinear `tan(w_half)` frequency pre-warping across 4 lanes.
#[inline(always)]
#[must_use]
pub(super) fn tan_prewarp_vec4<B: MathBackend>(w_half: B::Vec4) -> B::Vec4 {
    let x = w_half.clamp(B::Vec4::splat(1.0e-4), B::Vec4::splat(SVF_W_MAX));
    let x2 = x.mul(x);
    let poly = B::Vec4::splat(2.0 / 15.0).fma(x2, B::Vec4::splat(17.0 / 315.0));
    let mid = B::Vec4::splat(1.0 / 3.0).fma(x2, poly);
    let inner = B::Vec4::splat(1.0).fma(x2, mid);
    x.mul(inner)
        .clamp(B::Vec4::splat(1.0e-4), B::Vec4::splat(SVF_G_MAX))
}

/// 7th-order division-free bilinear `tan(w_half)` frequency pre-warping (`0 FDIV`, `0 libm`).
#[inline(always)]
#[must_use]
pub fn tan_prewarp_f32(w_half: f32) -> f32 {
    let x = w_half.clamp(1.0e-4, SVF_W_MAX);
    let x2 = x * x;
    (x * (1.0 + x2 * ((1.0 / 3.0) + x2 * ((2.0 / 15.0) + (17.0 / 315.0) * x2))))
        .clamp(1.0e-4, SVF_G_MAX)
}

/// Stage I Envelope-Warped 2-Section Cytomic TPT SVF parameters (`48 kHz` host-rate).
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Stage1SvfParams {
    /// Resting Section-1 integrator coefficient $g_0 = \tan(\pi f_0 / f_s) \in [10^{-4}, 1.45]$.
    pub base_g: f32,
    /// Section-2 frequency multiplier `g2_base / g1_base in [0.25, 4.0]`.
    pub sec2_g_ratio: f32,
    /// Envelope-to-octave sensitivity in `[-24.0, 24.0]` octaves.
    pub env_oct_sens: f32,
    /// Algebraic envelope compression factor `>= 0.0` (`env / sqrt(1 + env_comp * env^2)`).
    pub env_comp: f32,
    /// Envelope attack follower retention pole in `[0.001, 0.9999]`.
    pub atk_alpha: f32,
    /// Envelope release follower retention pole in `[0.001, 0.99995]`.
    pub rel_alpha: f32,
    /// Detector 1-pole LPF pole in `[0.05, 0.995]`.
    pub det_lp_pole: f32,
    /// Detector HP pre-emphasis mix (`det = x + det_hp_mix * (x - lp)`).
    pub det_hp_mix: f32,
    /// Section-1 Cytomic damping `k_1 = 1 / Q_1 in [0.05, 4.0]`.
    pub damp_k1: f32,
    /// Section-2 Cytomic damping `k_2 = 1 / Q_2 in [0.05, 4.0]`.
    pub damp_k2: f32,
    /// Section-1 BP feed into Section-2 input (`u2 = lp1 + sec2_bp_feed * bp1 + sec2_hp_feed * hp1`).
    pub sec2_bp_feed: f32,
    /// Section-1 HP feed into Section-2 input.
    pub sec2_hp_feed: f32,
    /// Stereo detector link in `[0.0, 1.0]` (`0` dual-mono, `1` symmetric linked).
    pub stereo_link: f32,
    /// Minimum Stage I `g1` tuning coefficient in `[1e-4, 1.45]` (envelope rail floor).
    pub g_min: f32,
    /// Maximum Stage I `g1` tuning coefficient in `[g_min, 1.45]` (envelope rail ceiling).
    pub g_max: f32,
    /// Optional Stage I `lp2` blend into Stage II `u_hp` (`eff_lp = u_lp + feat_lp2_mix * (lp2 - u_lp)`).
    pub feat_lp2_mix: f32,
    /// Output tap mix `[x, lp1, bp1, hp1, lp2, bp2, hp2]` feeding Stage II input (`0` padded to 8).
    pub stage2_feed_taps: [f32; 8],
    /// Direct Stage I parallel tap mix to Left output `[x, lp1, bp1, hp1, lp2, bp2, hp2, 0]`.
    pub direct_taps_l: [f32; 8],
    /// Direct Stage I parallel tap mix to Right output `[x, lp1, bp1, hp1, lp2, bp2, hp2, 0]`.
    pub direct_taps_r: [f32; 8],
}

impl Default for V3Stage1SvfParams {
    fn default() -> Self {
        Self::identity_bypass()
    }
}

impl V3Stage1SvfParams {
    /// Clean unity pass-through defaults (`u_s2 = x`, `direct_svf = 0`).
    #[must_use]
    pub const fn identity_bypass() -> Self {
        Self {
            base_g: 0.08,
            sec2_g_ratio: 1.0,
            env_oct_sens: 0.0,
            env_comp: 1.0,
            atk_alpha: 0.85,
            rel_alpha: 0.996,
            det_lp_pole: 0.90,
            det_hp_mix: 0.0,
            damp_k1: core::f32::consts::SQRT_2,
            damp_k2: core::f32::consts::SQRT_2,
            sec2_bp_feed: 0.0,
            sec2_hp_feed: 0.0,
            stereo_link: 0.5,
            g_min: 1.0e-4,
            g_max: 0.95,
            feat_lp2_mix: 0.0,
            stage2_feed_taps: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            direct_taps_l: [0.0; 8],
            direct_taps_r: [0.0; 8],
        }
    }

    /// Clamps nonlinear physical/stability parameters into unconditional passivity bounds.
    pub fn clamp_in_place(&mut self) {
        self.base_g = self.base_g.clamp(1.0e-4, SVF_W_MAX);
        self.sec2_g_ratio = self.sec2_g_ratio.clamp(0.25, 4.0);
        self.env_oct_sens = self.env_oct_sens.clamp(-24.0, 24.0);
        self.env_comp = self.env_comp.clamp(0.0, 16.0);
        self.atk_alpha = self.atk_alpha.clamp(0.001, 0.9999);
        self.rel_alpha = self.rel_alpha.clamp(0.001, 0.99995);
        self.det_lp_pole = self.det_lp_pole.clamp(0.05, 0.995);
        self.det_hp_mix = self.det_hp_mix.clamp(-1.0, 2.0);
        self.damp_k1 = self.damp_k1.clamp(0.05, 4.0);
        self.damp_k2 = self.damp_k2.clamp(0.05, 4.0);
        self.sec2_bp_feed = self.sec2_bp_feed.clamp(-2.0, 2.0);
        self.sec2_hp_feed = self.sec2_hp_feed.clamp(-2.0, 2.0);
        self.stereo_link = self.stereo_link.clamp(0.0, 1.0);
        self.g_min = self.g_min.clamp(1.0e-4, SVF_W_MAX);
        self.g_max = self.g_max.clamp(self.g_min, SVF_W_MAX);
        self.feat_lp2_mix = self.feat_lp2_mix.clamp(0.0, 1.0);
        self.stage2_feed_taps[7] = 0.0;
        self.direct_taps_l[7] = 0.0;
        self.direct_taps_r[7] = 0.0;
    }

    #[inline(always)]
    fn rescale_svf_w_2x_to_1x(w: f32) -> f32 {
        let wc = w.clamp(1.0e-4, SVF_W_MAX);
        (2.0 * wc * rsqrt_nr3_f32(1.0 + 1.5625 * wc * wc)).clamp(1.0e-4, SVF_W_MAX)
    }

    /// Rescales `2x` (`96 kHz`) normalized half-angle `w_half = pi*f/fs` and envelope poles to `1x` (`48 kHz`).
    pub fn rescale_2x_to_1x(&mut self) {
        self.base_g = Self::rescale_svf_w_2x_to_1x(self.base_g);
        self.g_min = Self::rescale_svf_w_2x_to_1x(self.g_min);
        self.g_max = Self::rescale_svf_w_2x_to_1x(self.g_max).clamp(self.g_min, SVF_W_MAX);
        self.det_lp_pole = (self.det_lp_pole * self.det_lp_pole).clamp(0.05, 0.995);
        self.atk_alpha = (self.atk_alpha * self.atk_alpha).clamp(0.001, 0.9999);
        self.rel_alpha = (self.rel_alpha * self.rel_alpha).clamp(0.001, 0.99995);
        self.clamp_in_place();
    }

    /// Rescales `48 kHz`-anchored Stage I normalized cutoff half-angles and detector poles for `zoh`.
    pub fn rescale_with_zoh(&mut self, zoh: &ZohRescaledParams) {
        self.clamp_in_place();
        if (zoh.rho_sr - 1.0).abs() <= 1.0e-7 {
            return;
        }
        if (zoh.rho_sr - 2.0).abs() <= 1.0e-7 {
            self.rescale_2x_to_1x();
            return;
        }
        let scale = zoh.rho_sr as f32;
        self.base_g = (self.base_g * scale).clamp(1.0e-4, SVF_W_MAX);
        self.g_min = (self.g_min * scale).clamp(1.0e-4, SVF_W_MAX);
        self.g_max = (self.g_max * scale).clamp(self.g_min, SVF_W_MAX);
        self.det_lp_pole = zoh.rescale_real_pole(self.det_lp_pole).clamp(0.05, 0.995);
        self.atk_alpha = zoh.rescale_real_pole(self.atk_alpha).clamp(0.001, 0.9999);
        self.rel_alpha = zoh.rescale_real_pole(self.rel_alpha).clamp(0.001, 0.99995);
        self.clamp_in_place();
    }
}

/// Sub-block-hoisted Stage I SVF constants so inner sample loops keep taps/bounds in SIMD registers.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SvfHoistedConsts<V: SimdVec4> {
    pub lp_a: f32,
    pub lp_b: f32,
    pub det_hp_mix: f32,
    pub half_link: f32,
    pub atk_alpha: f32,
    pub rel_alpha: f32,
    pub sec2_bp_v: V,
    pub sec2_hp_v: V,
    pub one: V,
    pub env_comp: V,
    pub env_oct_sens: V,
    pub base_w: V,
    pub g_min: V,
    pub g_max: V,
    pub ratio_v: V,
    pub g_clip_lo: V,
    pub g_clip_hi: V,
    pub k_vec: V,
    pub tap_w: [V; 7],
}

impl<V: SimdVec4> SvfHoistedConsts<V> {
    #[inline(always)]
    pub(crate) fn from_params(
        params: &V3Stage1SvfParams,
        env_fc_scale: f32,
        env_k_delta: f32,
    ) -> Self {
        let lp_a = params.det_lp_pole;
        let k1 = (params.damp_k1 + env_k_delta).clamp(0.05, 4.0);
        let k2 = (params.damp_k2 + env_k_delta).clamp(0.05, 4.0);
        let tap_w = core::array::from_fn(|k| {
            V::from_array([
                params.stage2_feed_taps[k],
                params.stage2_feed_taps[k],
                params.direct_taps_l[k],
                params.direct_taps_r[k],
            ])
        });
        Self {
            lp_a,
            lp_b: 1.0 - lp_a,
            det_hp_mix: params.det_hp_mix,
            half_link: 0.5 * params.stereo_link,
            atk_alpha: params.atk_alpha,
            rel_alpha: params.rel_alpha,
            sec2_bp_v: V::splat(params.sec2_bp_feed),
            sec2_hp_v: V::splat(params.sec2_hp_feed),
            one: V::splat(1.0),
            env_comp: V::splat(params.env_comp),
            env_oct_sens: V::splat(params.env_oct_sens),
            base_w: V::splat(params.base_g * env_fc_scale),
            g_min: V::splat(params.g_min),
            g_max: V::splat(params.g_max),
            ratio_v: V::from_array([1.0, 1.0, params.sec2_g_ratio, params.sec2_g_ratio]),
            g_clip_lo: V::splat(1.0e-4),
            g_clip_hi: V::splat(SVF_G_MAX),
            k_vec: V::from_array([k1, k1, k2, k2]),
            tap_w,
        }
    }
}

#[inline(always)]
pub(super) fn prewarp_svf_g1<B: MathBackend>(
    c: &SvfHoistedConsts<B::Vec4>,
    env_v: B::Vec4,
) -> B::Vec4 {
    let inv_e = c.one.fma(c.env_comp, env_v.mul(env_v)).rsqrt_pos_nr2();
    let oct_v = env_v.mul(inv_e).mul(c.env_oct_sens);
    let w1_v = c.base_w.mul(oct_v.exp2_clamped5()).clamp(c.g_min, c.g_max);
    tan_prewarp_vec4::<B>(w1_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tan_double_angle_rescale_clamp_and_zoh() {
        let mut p = V3Stage1SvfParams {
            base_g: 0.25,
            g_min: 0.05,
            g_max: 0.68,
            det_lp_pole: 0.8,
            ..V3Stage1SvfParams::identity_bypass()
        };
        p.rescale_2x_to_1x();
        assert!((p.g_min - 0.099_805).abs() < 1.0e-4);
        assert!((p.base_g - 0.477_24).abs() < 1.0e-4);
        assert!((p.g_max - 1.036_24).abs() < 1.0e-3);
        let g_1x = tan_prewarp_f32(p.g_max);
        let g_2x = tan_prewarp_f32(0.68);
        assert!((g_1x - 2.0 * g_2x).abs() / g_1x < 0.04);
        assert!((p.det_lp_pole - 0.64).abs() < 1.0e-6);
        assert!(p.g_max >= p.g_min && p.g_max <= SVF_W_MAX);

        let zoh_192 =
            ZohRescaledParams::compute(48_000, 192_000, 0.9993, 0.995, 64.0, 128).unwrap();
        let mut p_zoh = V3Stage1SvfParams {
            base_g: 0.40,
            det_lp_pole: 0.8,
            ..V3Stage1SvfParams::identity_bypass()
        };
        p_zoh.rescale_with_zoh(&zoh_192);
        assert!((p_zoh.base_g - 0.10).abs() < 1.0e-5);
        assert!(p_zoh.det_lp_pole > 0.9);
    }
}
