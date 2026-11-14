//! Stage I Envelope-Warped 2-Section Cytomic TPT/ZDF State-Variable Filter (`v3` Hot Core).
//!
//! Operates strictly at 1x host rate with zero `FDIV`, zero `FSQRT`, zero `libm`, zero integer
//! division, and zero allocations. Includes bit-hack + Newton–Raphson `recip_nr3_pos_f32`,
//! `rsqrt_nr3_f32`, `rsqrt_nr3_vec4`, a $C^2$-continuous quintic Hermite octave-warping curve
//! `exp2_signed_oct_f32`, and zero-DC algebraic asymmetric softclipping.

use crate::math::{MathBackend, SimdVec4, floor_f32, flush_recursive_state_f32};

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using IEEE-754 integer seed
/// `0x7EF1_27EA` followed by 3 Newton–Raphson iterations (`0 FDIV`, `< 1 ULP` over normal range).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(1.0e-6);
    let mut y = f32::from_bits(0x7EF1_27EA_u32.wrapping_sub(d_safe.to_bits()));
    y = y * (2.0 - d_safe * y);
    y = y * (2.0 - d_safe * y);
    y * (2.0 - d_safe * y)
}

/// Fast inverse square root `1.0 / sqrt(d)` (`d` clamped to `>= 1.0e-12`) using Quake III /
/// IEEE-754 seed `0x5F37_5A86` followed by 3 Newton–Raphson iterations (`0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_f32(d: f32) -> f32 {
    let d_safe = d.max(1.0e-12);
    let half_d = 0.5 * d_safe;
    let mut y = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(d_safe.to_bits() >> 1));
    y = y * (1.5 - half_d * y * y);
    y = y * (1.5 - half_d * y * y);
    y * (1.5 - half_d * y * y)
}

/// 4-lane SIMD inverse square root `1.0 / sqrt(d)` using integer seed + 3 SIMD FMA/FMS Newton steps.
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    let d_safe = d.max(B::Vec4::splat(1.0e-12));
    let arr = d_safe.to_array();
    let seeds = [
        f32::from_bits(0x5F37_5A86_u32.wrapping_sub(arr[0].to_bits() >> 1)),
        f32::from_bits(0x5F37_5A86_u32.wrapping_sub(arr[1].to_bits() >> 1)),
        f32::from_bits(0x5F37_5A86_u32.wrapping_sub(arr[2].to_bits() >> 1)),
        f32::from_bits(0x5F37_5A86_u32.wrapping_sub(arr[3].to_bits() >> 1)),
    ];
    let half_d = d_safe.mul(B::Vec4::splat(0.5));
    let three_halves = B::Vec4::splat(1.5);
    let mut y = B::Vec4::from_array(seeds);
    y = y.mul(three_halves.fms(half_d, y.mul(y)));
    y = y.mul(three_halves.fms(half_d, y.mul(y)));
    y.mul(three_halves.fms(half_d, y.mul(y)))
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
#[allow(clippy::approx_constant)]
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

/// Stage I Envelope-Warped 2-Section Cytomic TPT SVF parameters (`48 kHz` host-rate).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Stage1SvfParams {
    /// Resting Section-1 integrator coefficient $g_0 = \tan(\pi f_0 / f_s) \in [10^{-4}, 1.45]$.
    pub base_g: f32,
    /// Section-2 frequency multiplier `g2_base / g1_base in [0.25, 4.0]`.
    pub sec2_g_ratio: f32,
    /// Envelope-to-octave sensitivity in `[-16.0, 16.0]` octaves.
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
    /// Reserved alignment pad to place tap arrays cleanly on cache-line boundaries.
    pub _pad: f32,
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
            g_max: 1.45,
            _pad: 0.0,
            stage2_feed_taps: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            direct_taps_l: [0.0; 8],
            direct_taps_r: [0.0; 8],
        }
    }

    /// Clamps nonlinear physical/stability parameters into unconditional passivity bounds.
    pub fn clamp_in_place(&mut self) {
        self.base_g = self.base_g.clamp(1.0e-4, 1.45);
        self.sec2_g_ratio = self.sec2_g_ratio.clamp(0.25, 4.0);
        self.env_oct_sens = self.env_oct_sens.clamp(-16.0, 16.0);
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
        self.g_min = self.g_min.clamp(1.0e-4, 1.45);
        self.g_max = self.g_max.clamp(self.g_min, 1.45);
        self._pad = 0.0;
        self.stage2_feed_taps[7] = 0.0;
        self.direct_taps_l[7] = 0.0;
        self.direct_taps_r[7] = 0.0;
    }
}

/// Stereo Stage I SVF state (`12` `f32` values = `48` bytes, padded to 1 cache line).
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
    pub _pad: [f32; 4],
}

impl V3Stage1SvfState {
    /// Resets all filter and envelope state variables to exact resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline(always)]
    fn step_tpt_section(u: f32, g: f32, k: f32, ic1: &mut f32, ic2: &mut f32) -> (f32, f32, f32) {
        let a1 = recip_nr3_pos_f32(1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
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

    #[inline(always)]
    fn dot7(taps: &[f32; 7], w: &[f32; 8]) -> f32 {
        taps[0] * w[0]
            + taps[1] * w[1]
            + taps[2] * w[2]
            + taps[3] * w[3]
            + taps[4] * w[4]
            + taps[5] * w[5]
            + taps[6] * w[6]
    }

    /// Steps stereo Stage I SVF for one host sample pair, returning `([u_s2_l, u_s2_r], [dir_l, dir_r])`.
    #[inline(always)]
    pub fn step_stereo(
        &mut self,
        params: &V3Stage1SvfParams,
        x_l: f32,
        x_r: f32,
    ) -> ([f32; 2], [f32; 2]) {
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
        self.env_l = flush_recursive_state_f32(alpha_l * self.env_l + (1.0 - alpha_l) * det_l);
        self.env_r = flush_recursive_state_f32(alpha_r * self.env_r + (1.0 - alpha_r) * det_r);

        let e_l = self.env_l * rsqrt_nr3_f32(1.0 + params.env_comp * self.env_l * self.env_l);
        let e_r = self.env_r * rsqrt_nr3_f32(1.0 + params.env_comp * self.env_r * self.env_r);

        let g1_l = (params.base_g * exp2_signed_oct_f32(params.env_oct_sens * e_l))
            .clamp(params.g_min, params.g_max);
        let g1_r = (params.base_g * exp2_signed_oct_f32(params.env_oct_sens * e_r))
            .clamp(params.g_min, params.g_max);
        let g2_l = (g1_l * params.sec2_g_ratio).clamp(1.0e-4, 1.45);
        let g2_r = (g1_r * params.sec2_g_ratio).clamp(1.0e-4, 1.45);

        let (lp1_l, bp1_l, hp1_l) = Self::step_tpt_section(
            x_l,
            g1_l,
            params.damp_k1,
            &mut self.ic1_s1_l,
            &mut self.ic2_s1_l,
        );
        let (lp1_r, bp1_r, hp1_r) = Self::step_tpt_section(
            x_r,
            g1_r,
            params.damp_k1,
            &mut self.ic1_s1_r,
            &mut self.ic2_s1_r,
        );

        let u2_l = lp1_l + params.sec2_bp_feed * bp1_l + params.sec2_hp_feed * hp1_l;
        let u2_r = lp1_r + params.sec2_bp_feed * bp1_r + params.sec2_hp_feed * hp1_r;

        let (lp2_l, bp2_l, hp2_l) = Self::step_tpt_section(
            u2_l,
            g2_l,
            params.damp_k2,
            &mut self.ic1_s2_l,
            &mut self.ic2_s2_l,
        );
        let (lp2_r, bp2_r, hp2_r) = Self::step_tpt_section(
            u2_r,
            g2_r,
            params.damp_k2,
            &mut self.ic1_s2_r,
            &mut self.ic2_s2_r,
        );

        let taps_l = [x_l, lp1_l, bp1_l, hp1_l, lp2_l, bp2_l, hp2_l];
        let taps_r = [x_r, lp1_r, bp1_r, hp1_r, lp2_r, bp2_r, hp2_r];
        let u_s2 = [
            Self::dot7(&taps_l, &params.stage2_feed_taps),
            Self::dot7(&taps_r, &params.stage2_feed_taps),
        ];
        let direct = [
            Self::dot7(&taps_l, &params.direct_taps_l),
            Self::dot7(&taps_r, &params.direct_taps_r),
        ];
        (u_s2, direct)
    }
}
