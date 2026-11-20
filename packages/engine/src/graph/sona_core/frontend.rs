//! Zero-lag `2x` oversamplers (scalar & 4-lane stereo SIMD) for `SONA` (`sona_core/frontend.rs`).

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32, flush_recursive_state_vec4};

/// 9th-order Polyphase Elliptic Half-Band IIR branch 0 allpass coefficients (`HB_A0`, `> 68 dB` rejection).
pub const HB_A0: [f32; 2] = [0.079_866_426, 0.545_353_65];
/// 9th-order Polyphase Elliptic Half-Band IIR branch 1 allpass coefficients (`HB_A1`, `> 68 dB` rejection).
pub const HB_A1: [f32; 2] = [0.283_829_34, 0.834_411_9];

const UP_HB_C0: [f32; 4] = [HB_A0[0], HB_A0[0], HB_A1[0], HB_A1[0]];
const UP_HB_C1: [f32; 4] = [HB_A0[1], HB_A0[1], HB_A1[1], HB_A1[1]];
const INV_UP_HB_REST_S0: [f32; 4] = [
    1.0 / (1.0 + HB_A0[0]),
    1.0 / (1.0 + HB_A0[0]),
    1.0 / (1.0 + HB_A1[0]),
    1.0 / (1.0 + HB_A1[0]),
];
const INV_UP_HB_REST_S1: [f32; 4] = [
    1.0 / (1.0 + HB_A0[1]),
    1.0 / (1.0 + HB_A0[1]),
    1.0 / (1.0 + HB_A1[1]),
    1.0 / (1.0 + HB_A1[1]),
];
const INV_DEC_HB_REST: [f32; 4] = [
    1.0 / (1.0 + HB_A0[0]),
    1.0 / (1.0 + HB_A0[1]),
    1.0 / (1.0 + HB_A1[0]),
    1.0 / (1.0 + HB_A1[1]),
];

const _: () = assert!(HB_A0[0] > 0.0 && HB_A0[1] < 1.0 && HB_A1[0] > 0.0 && HB_A1[1] < 1.0);

#[inline(always)]
fn step_allpass_1_f32(x: f32, a: f32, state: &mut f32) -> f32 {
    let s_old = *state;
    let s_new = flush_recursive_state_f32(x - a * s_old);
    *state = s_new;
    flush_recursive_state_f32(a * s_new + s_old)
}

#[inline(always)]
fn step_allpass_1_vec4_reg<V: SimdVec4>(x: V, a: V, state: &mut V) -> V {
    let s_old = *state;
    let s_new = flush_recursive_state_vec4::<V>(x.fms(a, s_old));
    *state = s_new;
    flush_recursive_state_vec4::<V>(s_old.fma(a, s_new))
}

#[inline(always)]
fn step_allpass_1_vec4<V: SimdVec4>(x: V, a: V, state: &mut [f32; 4]) -> V {
    let mut s_reg = V::load(state);
    let y = step_allpass_1_vec4_reg(x, a, &mut s_reg);
    s_reg.store(state);
    y
}

/// Division-free 2nd-order allpass cascade (`0 FDIV`, exact `+0.0 -> +0.0` at rest).
#[inline(always)]
pub fn step_allpass_2_f32(x: f32, a: [f32; 2], s0: &mut f32, s1: &mut f32) -> f32 {
    let m = step_allpass_1_f32(x, a[0], s0);
    step_allpass_1_f32(m, a[1], s1)
}

/// Division-free 9th-order polyphase elliptic half-band `96 kHz -> 48 kHz` decimator (`0 FDIV`,
/// `> 68 dB` stopband attenuation, `< 0.0005 dB` passband ripple, exact `+0.0 -> +0.0` at rest).
#[inline(always)]
pub fn step_hb_decim_f32(y0: f32, y1: f32, st: &mut [f32; 4]) -> f32 {
    let [mut s0, mut s1, mut s2, mut s3] = *st;
    let b0 = step_allpass_2_f32(y1, HB_A0, &mut s0, &mut s1);
    let b1 = step_allpass_2_f32(y0, HB_A1, &mut s2, &mut s3);
    *st = [s0, s1, s2, s3];
    flush_recursive_state_f32(0.5 * (b0 + b1))
}

/// Hybrid zero-lag polynomial + 9th-order polyphase elliptic half-band `2x` oversampler parameters.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaOversamplerParams {
    /// Causal quadratic midpoint curvature coefficient in `[-0.20, 0.05]` (`default = -0.125`).
    pub up_quad_coeff: f32,
    /// Polyphase decimation tap on previous `sub=1` sample `y[n-1, 1]` in `[0.0, 0.28]`.
    pub dec_c_prev1: f32,
    /// Polyphase decimation tap on midpoint `sub=0` sample `y[n, 0]` in `[0.0, 0.55]`.
    pub dec_c0: f32,
    /// Complement decimation tap `1.0 - dec_c_prev1 - dec_c0 >= 0.17` on current `sub=1` sample.
    pub dec_c1: f32,
    /// Causal unity-DC pre-filter `xp = (1 - up_pre_emph) * x + up_pre_emph * prev_x` in `[-0.28, 1.0]`.
    pub up_pre_emph: f32,
    /// Linear blend between 3-tap polyphase FIR (`0.0`) and 9th-order half-band decimator (`1.0`).
    pub dec_aa_mix: f32,
    /// Linear blend between zero-lag quadratic upsampler (`0.0`) and 9th-order half-band interpolator (`1.0`).
    pub up_aa_mix: f32,
    /// Linear blend from quadratic midpoint (`0.0`) toward causal zero-order hold (`1.0`) for `x0`.
    pub up_zoh_mix: f32,
}

impl Default for SonaOversamplerParams {
    fn default() -> Self {
        Self {
            up_quad_coeff: -0.125,
            dec_c_prev1: 0.15,
            dec_c0: 0.50,
            dec_c1: 0.35,
            up_pre_emph: 0.0,
            dec_aa_mix: 0.0,
            up_aa_mix: 0.0,
            up_zoh_mix: 0.0,
        }
    }
}

impl SonaOversamplerParams {
    /// Clamps oversampler taps into stable passivity bounds (`dec_c_prev1 + dec_c0 + dec_c1 == 1`).
    pub fn clamp_in_place(&mut self) {
        self.up_quad_coeff = self.up_quad_coeff.clamp(-0.20, 0.05);
        self.dec_c_prev1 = self.dec_c_prev1.clamp(0.0, 0.28);
        self.dec_c0 = self.dec_c0.clamp(0.0, 0.55);
        self.dec_c1 = 1.0 - self.dec_c_prev1 - self.dec_c0;
        self.up_pre_emph = self.up_pre_emph.clamp(-0.28, 1.0);
        self.dec_aa_mix = self.dec_aa_mix.clamp(0.0, 1.0);
        self.up_aa_mix = self.up_aa_mix.clamp(0.0, 1.0);
        self.up_zoh_mix = self.up_zoh_mix.clamp(0.0, 1.0);
    }

    /// Re-clamps oversampler taps under ZOH sample-rate reconfiguration (`2x` half-band & polyphase
    /// filters operate at host-relative `[fs -> 2*fs]` rather than fixed analog Hz).
    pub fn rescale_with_zoh(&mut self, _zoh: &crate::container::ZohRescaledParams) {
        self.clamp_in_place();
    }
}

/// Per-channel branchless hybrid zero-lag + 9th-order polyphase half-band `2x` state (`96 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZeroLagOversampler2x {
    pub up_raw_prev: f32,
    pub up_prev1: f32,
    pub up_prev2: f32,
    pub dec_s2_prev1: f32,
    pub dec_dir_s2_prev1: f32,
    pub dec_dir_svf_prev1: f32,
    pub up_hb_state: [f32; 4],
    pub dec_s2_aa: [f32; 4],
    pub dec_dir_s2_aa: [f32; 4],
    pub dec_dir_svf_aa: [f32; 4],
    pub _pad: [f32; 2],
}

impl ZeroLagOversampler2x {
    /// Resets all oversampler delay and half-band allpass elements to exact `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Branchless causal `2x` upsampler blending quadratic midpoint and 9th-order half-band IIR.
    #[inline(always)]
    pub fn upsample(&mut self, x: f32, params: &SonaOversamplerParams) -> [f32; 2] {
        let xp = (1.0 - params.up_pre_emph) * x + params.up_pre_emph * self.up_raw_prev;
        self.up_raw_prev = x;
        let d2 = xp - 2.0 * self.up_prev1 + self.up_prev2;
        let c_p1 = 0.5 * (1.0 - params.up_zoh_mix);
        let x0_poly = (1.0 - c_p1) * xp + c_p1 * self.up_prev1 + params.up_quad_coeff * d2;
        let x1_poly = xp;
        self.up_prev2 = self.up_prev1;
        self.up_prev1 = xp;
        let [mut s0, mut s1, mut s2, mut s3] = self.up_hb_state;
        let x0_hb = step_allpass_2_f32(xp, HB_A0, &mut s0, &mut s1);
        let x1_hb = step_allpass_2_f32(xp, HB_A1, &mut s2, &mut s3);
        self.up_hb_state = [s0, s1, s2, s3];
        let m = params.up_aa_mix;
        [
            x0_poly + m * (x0_hb - x0_poly),
            x1_poly + m * (x1_hb - x1_poly),
        ]
    }

    /// Branchless hybrid polyphase FIR + 9th-order elliptic half-band `96 kHz -> 48 kHz` decimator.
    #[inline(always)]
    pub fn downsample(
        y0: f32,
        y1: f32,
        prev_y1: &mut f32,
        aa_state: &mut [f32; 4],
        params: &SonaOversamplerParams,
    ) -> f32 {
        let y_poly = params.dec_c_prev1 * (*prev_y1) + params.dec_c0 * y0 + params.dec_c1 * y1;
        *prev_y1 = y1;
        let y_aa = step_hb_decim_f32(y0, y1, aa_state);
        y_poly + params.dec_aa_mix * (y_aa - y_poly)
    }
}

/// Sub-block-hoisted upsampler half-band constants and scalar parameters for zero-spill loops.
#[derive(Clone, Copy)]
pub(crate) struct UpsampleConsts<V: SimdVec4> {
    pub(crate) c0: V,
    pub(crate) c1: V,
    pub(crate) up_mix_v: V,
    pub(crate) one_minus_pe: f32,
    pub(crate) up_pe: f32,
    pub(crate) up_quad: f32,
    pub(crate) c_prev1: f32,
    pub(crate) c_xp: f32,
}

impl<V: SimdVec4> UpsampleConsts<V> {
    #[inline(always)]
    pub(crate) fn from_params(params: &SonaOversamplerParams) -> Self {
        let c_prev1 = 0.5 * (1.0 - params.up_zoh_mix);
        Self {
            c0: V::load(&UP_HB_C0),
            c1: V::load(&UP_HB_C1),
            up_mix_v: V::splat(params.up_aa_mix),
            one_minus_pe: 1.0 - params.up_pre_emph,
            up_pe: params.up_pre_emph,
            up_quad: params.up_quad_coeff,
            c_prev1,
            c_xp: 1.0 - c_prev1,
        }
    }
}

/// Vectorized stereo `2x` upsampler + 4-channel `[s2_l, s2_r, dir_l, dir_r]` decimator (`144 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SonaStereoOversampler2x {
    pub dec_prev1: [f32; 4],
    pub up_hb_s0: [f32; 4],
    pub up_hb_s1: [f32; 4],
    pub dec_hb_state: [[f32; 4]; 4],
    pub up_raw_prev: [f32; 2],
    pub up_prev1: [f32; 2],
    pub up_prev2: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<SonaStereoOversampler2x>() == 144);

impl SonaStereoOversampler2x {
    /// Resets all oversampler delay and half-band states to exact `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Warm-seeds both upsampler and 4-channel decimator states to DC steady state for smooth tier switches.
    #[inline]
    pub fn warm_seed(&mut self, last_in: [f32; 2], last_s2_dir: [f32; 4]) {
        self.up_raw_prev = last_in;
        self.up_prev1 = last_in;
        self.up_prev2 = last_in;
        let in_pair = [last_in[0], last_in[1], last_in[0], last_in[1]];
        for c in 0..4 {
            self.up_hb_s0[c] = in_pair[c] * INV_UP_HB_REST_S0[c];
            self.up_hb_s1[c] = in_pair[c] * INV_UP_HB_REST_S1[c];
        }
        self.dec_prev1 = last_s2_dir;
        for (stage, &inv) in INV_DEC_HB_REST.iter().enumerate() {
            for c in 0..4 {
                self.dec_hb_state[stage][c] = last_s2_dir[c] * inv;
            }
        }
    }

    /// Branchless stereo `2x` upsampler returning `([xl0, xr0], [xl1, xr1])`.
    #[inline(always)]
    pub fn upsample_stereo<B: MathBackend>(
        &mut self,
        x_l: f32,
        x_r: f32,
        params: &SonaOversamplerParams,
    ) -> ([f32; 2], [f32; 2]) {
        let mut s0 = B::Vec4::load(&self.up_hb_s0);
        let mut s1 = B::Vec4::load(&self.up_hb_s1);
        let c = UpsampleConsts::<B::Vec4>::from_params(params);
        let out = self.upsample_stereo_reg::<B>(&mut s0, &mut s1, &c, x_l, x_r);
        s0.store(&mut self.up_hb_s0);
        s1.store(&mut self.up_hb_s1);
        out
    }

    /// Branchless stereo `2x` upsampler keeping `up_hb_s0/s1` in `SimdVec4` registers across a sub-block.
    #[inline(always)]
    pub(crate) fn upsample_stereo_reg<B: MathBackend>(
        &mut self,
        s0: &mut B::Vec4,
        s1: &mut B::Vec4,
        c: &UpsampleConsts<B::Vec4>,
        x_l: f32,
        x_r: f32,
    ) -> ([f32; 2], [f32; 2]) {
        let xp_l = c.one_minus_pe * x_l + c.up_pe * self.up_raw_prev[0];
        let xp_r = c.one_minus_pe * x_r + c.up_pe * self.up_raw_prev[1];
        self.up_raw_prev = [x_l, x_r];

        let d2_l = xp_l - 2.0 * self.up_prev1[0] + self.up_prev2[0];
        let d2_r = xp_r - 2.0 * self.up_prev1[1] + self.up_prev2[1];
        let x0_poly_l = c.c_xp * xp_l + c.c_prev1 * self.up_prev1[0] + c.up_quad * d2_l;
        let x0_poly_r = c.c_xp * xp_r + c.c_prev1 * self.up_prev1[1] + c.up_quad * d2_r;
        self.up_prev2 = self.up_prev1;
        self.up_prev1 = [xp_l, xp_r];

        let xp_v = B::Vec4::splat_pair(xp_l, xp_r);
        let m_v = step_allpass_1_vec4_reg::<B::Vec4>(xp_v, c.c0, s0);
        let hb_v = step_allpass_1_vec4_reg::<B::Vec4>(m_v, c.c1, s1);

        let poly_v = B::Vec4::from_array([x0_poly_l, x0_poly_r, xp_l, xp_r]);
        let up_v = poly_v.fma(c.up_mix_v, hb_v.sub(poly_v));
        let [xl0, xr0, xl1, xr1] = up_v.to_array();
        ([xl0, xr0], [xl1, xr1])
    }

    /// Branchless 4-channel SIMD `2x` sub-block decimator keeping allpass states in SIMD registers.
    #[inline(always)]
    pub(crate) fn downsample_subblock_2x<B: MathBackend>(
        &mut self,
        params: &SonaOversamplerParams,
        s2_sub: &[[f32; 4]],
        s2_dir: &mut [[f32; 4]; 16],
        len: usize,
    ) {
        let mut prev1 = B::Vec4::load(&self.dec_prev1);
        let mut s0 = B::Vec4::load(&self.dec_hb_state[0]);
        let mut s1 = B::Vec4::load(&self.dec_hb_state[1]);
        let mut s2 = B::Vec4::load(&self.dec_hb_state[2]);
        let mut s3 = B::Vec4::load(&self.dec_hb_state[3]);

        let dec_c_prev1 = B::Vec4::splat(params.dec_c_prev1);
        let dec_c0 = B::Vec4::splat(params.dec_c0);
        let dec_c1 = B::Vec4::splat(params.dec_c1);
        let dec_aa_mix = B::Vec4::splat(params.dec_aa_mix);
        let half = B::Vec4::splat(0.5);
        let a00 = B::Vec4::splat(HB_A0[0]);
        let a01 = B::Vec4::splat(HB_A0[1]);
        let a10 = B::Vec4::splat(HB_A1[0]);
        let a11 = B::Vec4::splat(HB_A1[1]);

        let (pairs, _) = s2_sub.as_chunks::<2>();
        for (pair, dst) in pairs.iter().zip(s2_dir.iter_mut()).take(len) {
            let y0 = B::Vec4::load(&pair[0]);
            let y1 = B::Vec4::load(&pair[1]);
            let y_poly = prev1.mul(dec_c_prev1).fma(dec_c0, y0).fma(dec_c1, y1);
            prev1 = y1;

            let m0 = step_allpass_1_vec4_reg::<B::Vec4>(y1, a00, &mut s0);
            let b0 = step_allpass_1_vec4_reg::<B::Vec4>(m0, a01, &mut s1);
            let m1 = step_allpass_1_vec4_reg::<B::Vec4>(y0, a10, &mut s2);
            let b1 = step_allpass_1_vec4_reg::<B::Vec4>(m1, a11, &mut s3);
            let y_aa = flush_recursive_state_vec4::<B::Vec4>(half.mul(b0.add(b1)));
            y_poly.fma(dec_aa_mix, y_aa.sub(y_poly)).store(dst);
        }

        prev1.store(&mut self.dec_prev1);
        s0.store(&mut self.dec_hb_state[0]);
        s1.store(&mut self.dec_hb_state[1]);
        s2.store(&mut self.dec_hb_state[2]);
        s3.store(&mut self.dec_hb_state[3]);
    }

    /// Branchless 4-channel SIMD `2x` decimator on `sub0, sub1 = [s2_l, s2_r, dir_l, dir_r]`.
    #[inline(always)]
    pub fn downsample_4ch<B: MathBackend>(
        &mut self,
        sub0: [f32; 4],
        sub1: [f32; 4],
        params: &SonaOversamplerParams,
    ) -> [f32; 4] {
        let y0 = B::Vec4::load(&sub0);
        let y1 = B::Vec4::load(&sub1);
        let prev1 = B::Vec4::load(&self.dec_prev1);
        y1.store(&mut self.dec_prev1);

        let y_poly = prev1
            .mul(B::Vec4::splat(params.dec_c_prev1))
            .fma(B::Vec4::splat(params.dec_c0), y0)
            .fma(B::Vec4::splat(params.dec_c1), y1);

        let m0 =
            step_allpass_1_vec4::<B::Vec4>(y1, B::Vec4::splat(HB_A0[0]), &mut self.dec_hb_state[0]);
        let b0 =
            step_allpass_1_vec4::<B::Vec4>(m0, B::Vec4::splat(HB_A0[1]), &mut self.dec_hb_state[1]);
        let m1 =
            step_allpass_1_vec4::<B::Vec4>(y0, B::Vec4::splat(HB_A1[0]), &mut self.dec_hb_state[2]);
        let b1 =
            step_allpass_1_vec4::<B::Vec4>(m1, B::Vec4::splat(HB_A1[1]), &mut self.dec_hb_state[3]);
        let y_aa = flush_recursive_state_vec4::<B::Vec4>(B::Vec4::splat(0.5).mul(b0.add(b1)));

        y_poly
            .fma(B::Vec4::splat(params.dec_aa_mix), y_aa.sub(y_poly))
            .to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_sona_oversampler_hb_dc_stopband_and_stereo_parity() {
        let mut p = SonaOversamplerParams {
            up_pre_emph: 0.14,
            dec_aa_mix: 1.0,
            up_aa_mix: 1.0,
            ..SonaOversamplerParams::default()
        };
        p.clamp_in_place();

        let mut os = ZeroLagOversampler2x::default();
        let mut prev_y1 = 0.0_f32;
        let mut aa = [0.0_f32; 4];
        let mut dc_out = 0.0_f32;
        for _ in 0..256 {
            let up = os.upsample(1.0, &p);
            dc_out = ZeroLagOversampler2x::downsample(up[0], up[1], &mut prev_y1, &mut aa, &p);
        }
        assert!((dc_out - 1.0).abs() < 1.0e-5, "dc_out={dc_out}");

        aa = [0.0; 4];
        prev_y1 = 0.0;
        let mut rms_sq = 0.0_f32;
        for k in 0..128 {
            let y = ZeroLagOversampler2x::downsample(1.0, -1.0, &mut prev_y1, &mut aa, &p);
            if k >= 32 {
                rms_sq += y * y;
            }
        }
        let rms = (rms_sq / 96.0).sqrt();
        assert!(
            rms < 1.0e-4,
            "96 kHz Nyquist attenuation insufficient: {rms}"
        );

        let mut st_os = SonaStereoOversampler2x::default();
        let mut sc_l = ZeroLagOversampler2x::default();
        let mut sc_r = ZeroLagOversampler2x::default();
        let (mut prev_d_l, mut prev_d_r) = (0.0_f32, 0.0_f32);
        let (mut aa_d_l, mut aa_d_r) = ([0.0_f32; 4], [0.0_f32; 4]);
        for n in 0..48 {
            let xl = 0.6 * (n as f32 * 0.13).sin();
            let xr = -0.45 * (n as f32 * 0.19).cos();
            let (u0, u1) = st_os.upsample_stereo::<DefaultMathBackend>(xl, xr, &p);
            let sl = sc_l.upsample(xl, &p);
            let sr = sc_r.upsample(xr, &p);
            assert!((u0[0] - sl[0]).abs() < 1.0e-6 && (u1[0] - sl[1]).abs() < 1.0e-6);
            assert!((u0[1] - sr[0]).abs() < 1.0e-6 && (u1[1] - sr[1]).abs() < 1.0e-6);
            let sub0 = [sl[0], sr[0], 0.3 * sl[0], -0.2 * sr[0]];
            let sub1 = [sl[1], sr[1], 0.3 * sl[1], -0.2 * sr[1]];
            let dec = st_os.downsample_4ch::<DefaultMathBackend>(sub0, sub1, &p);
            let dl = ZeroLagOversampler2x::downsample(
                sl[0],
                sl[1],
                &mut sc_l.dec_s2_prev1,
                &mut sc_l.dec_s2_aa,
                &p,
            );
            let dr = ZeroLagOversampler2x::downsample(
                sr[0],
                sr[1],
                &mut sc_r.dec_s2_prev1,
                &mut sc_r.dec_s2_aa,
                &p,
            );
            let ddl =
                ZeroLagOversampler2x::downsample(sub0[2], sub1[2], &mut prev_d_l, &mut aa_d_l, &p);
            let ddr =
                ZeroLagOversampler2x::downsample(sub0[3], sub1[3], &mut prev_d_r, &mut aa_d_r, &p);
            assert!((dec[0] - dl).abs() < 1.0e-5 && (dec[1] - dr).abs() < 1.0e-5);
            assert!((dec[2] - ddl).abs() < 1.0e-5 && (dec[3] - ddr).abs() < 1.0e-5);
        }

        let mut sb_os = SonaStereoOversampler2x::default();
        let mut scalar_os = SonaStereoOversampler2x::default();
        let mut sub_pairs = [[0.0_f32; 4]; 32];
        let mut ref_out = [[0.0_f32; 4]; 16];
        let mut sub_out = [[0.0_f32; 4]; 16];
        for i in 0..16 {
            let s0 = [
                0.4 * (i as f32 * 0.2).sin(),
                -0.3 * (i as f32 * 0.3).cos(),
                0.2 * (i as f32 * 0.1).sin(),
                -0.15 * (i as f32 * 0.25).cos(),
            ];
            let s1 = [
                0.35 * ((i as f32 + 0.5) * 0.2).sin(),
                -0.25 * ((i as f32 + 0.5) * 0.3).cos(),
                0.18 * ((i as f32 + 0.5) * 0.1).sin(),
                -0.12 * ((i as f32 + 0.5) * 0.25).cos(),
            ];
            sub_pairs[i * 2] = s0;
            sub_pairs[i * 2 + 1] = s1;
            ref_out[i] = scalar_os.downsample_4ch::<DefaultMathBackend>(s0, s1, &p);
        }
        sb_os.downsample_subblock_2x::<DefaultMathBackend>(&p, &sub_pairs, &mut sub_out, 16);
        for i in 0..16 {
            for c in 0..4 {
                assert!((sub_out[i][c] - ref_out[i][c]).abs() < 1.0e-6);
            }
        }
    }
}
