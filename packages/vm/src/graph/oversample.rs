//! Zero-lag `2x` stereo oversampler (`OS2X`), `SonaOversamplerParams` (`32 B`),
//! `SonaStereoOversampler2x` (`144 B`), `256 B` cache-aligned `SonaOs2xHot` working block, and
//! `4x` offline linear-phase bounce oversampler (`bounce4x`).
//! (`0 FDIV`, `!alloc`, `ref[dsp.router.signal-flow]`, `sona_architecture.md §2.3`).

#![forbid(unsafe_code)]

pub mod bounce4x;

use core::mem::{align_of, offset_of, size_of};

pub use bounce4x::{Bounce4xDecimator, Bounce4xOversampler, OFFLINE_BOUNCE_PDC_SAMPLES};

use crate::bin::{CrossbarWire, SONA_META_OS_EXT_PRESENT};
use crate::math::{
    MathBackend, SimdVec4, clamp_finite_f32, flush_recursive_state_f32, flush_recursive_state_vec4,
};

/// 9th-order Polyphase Elliptic Half-Band IIR branch 0 all-pass coefficients (`HB_A0`).
pub const HB_A0: [f32; 2] = [0.079_866_426, 0.545_353_65];
/// 9th-order Polyphase Elliptic Half-Band IIR branch 1 all-pass coefficients (`HB_A1`).
pub const HB_A1: [f32; 2] = [0.283_829_34, 0.834_411_9];

const UP_HB_C0: [f32; 4] = [HB_A0[0], HB_A0[0], HB_A1[0], HB_A1[0]];
const UP_HB_C1: [f32; 4] = [HB_A0[1], HB_A0[1], HB_A1[1], HB_A1[1]];

const fn inv_one_plus_f32(a: f32) -> f32 {
    (1.0_f64 / (1.0_f64 + (a as f64))) as f32
}

const INV_UP_HB_S0: [f32; 4] = [
    inv_one_plus_f32(HB_A0[0]),
    inv_one_plus_f32(HB_A0[0]),
    inv_one_plus_f32(HB_A1[0]),
    inv_one_plus_f32(HB_A1[0]),
];
const INV_UP_HB_S1: [f32; 4] = [
    inv_one_plus_f32(HB_A0[1]),
    inv_one_plus_f32(HB_A0[1]),
    inv_one_plus_f32(HB_A1[1]),
    inv_one_plus_f32(HB_A1[1]),
];
const INV_DEC_HB: [f32; 4] = [
    inv_one_plus_f32(HB_A0[0]),
    inv_one_plus_f32(HB_A0[1]),
    inv_one_plus_f32(HB_A1[0]),
    inv_one_plus_f32(HB_A1[1]),
];

const _: () = assert!(HB_A0[0] > 0.0 && HB_A0[1] < 1.0 && HB_A1[0] > 0.0 && HB_A1[1] < 1.0);
const _: () = assert!(INV_UP_HB_S0[0] > 0.0 && INV_DEC_HB[3] > 0.5);

#[inline(always)]
fn step_allpass_1_vec4<V: SimdVec4>(x: V, a: V, state: &mut V) -> V {
    let s_old = *state;
    let s_new = flush_recursive_state_vec4::<V>(x.fms(a, s_old));
    *state = s_new;
    flush_recursive_state_vec4::<V>(s_old.fma(a, s_new))
}

#[inline(always)]
fn step_allpass_1_arr4<V: SimdVec4>(x: V, a: V, state: &mut [f32; 4]) -> V {
    let mut s_reg = V::load(state);
    let y = step_allpass_1_vec4::<V>(x, a, &mut s_reg);
    s_reg.store(state);
    y
}

/// Hybrid zero-lag polynomial + 9th-order polyphase elliptic half-band `2x` oversampler parameters (`32 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaOversamplerParams {
    /// Causal quadratic midpoint curvature coefficient in `[-0.20, 0.05]` (`default = -0.125`).
    pub up_quad_coeff: f32,
    /// Polyphase decimation tap on `y[2n - 1]` in `[0.0, 0.28]` (`default = 0.15`).
    pub dec_c_prev1: f32,
    /// Polyphase decimation tap on `y[2n]` in `[0.0, 0.55]` (`default = 0.50`).
    pub dec_c0: f32,
    /// Derived unity-DC decimation tap `1.0 - dec_c_prev1 - dec_c0` on `y[2n + 1]` (`default = 0.35`).
    pub dec_c1: f32,
    /// Causal unity-DC pre-filter blend in `[-0.28, 1.0]` (`default = 0.0`).
    pub up_pre_emph: f32,
    /// Decimator blend between `poly3` FIR (`0.0`) and 9th-order half-band IIR (`1.0`).
    pub dec_aa_mix: f32,
    /// Upsampler blend between polynomial (`0.0`) and 9th-order half-band IIR (`1.0`).
    pub up_aa_mix: f32,
    /// Upsampler midpoint blend from quadratic (`0.0`) toward causal zero-order hold (`1.0`).
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
    /// Clamps parameters into stable bounds with non-finite fallbacks per `dnam_binary_format.md §2.3`.
    pub fn clamp_in_place(&mut self) {
        self.up_quad_coeff = clamp_finite_f32(self.up_quad_coeff, -0.20, 0.05, 0.0);
        self.dec_c_prev1 = clamp_finite_f32(self.dec_c_prev1, 0.0, 0.28, 0.15);
        self.dec_c0 = clamp_finite_f32(self.dec_c0, 0.0, 0.55, 0.50);
        self.dec_c1 = 1.0 - self.dec_c_prev1 - self.dec_c0;
        self.up_pre_emph = clamp_finite_f32(self.up_pre_emph, -0.28, 1.0, 0.0);
        self.dec_aa_mix = clamp_finite_f32(self.dec_aa_mix, 0.0, 1.0, 0.0);
        self.up_aa_mix = clamp_finite_f32(self.up_aa_mix, 0.0, 1.0, 0.0);
        self.up_zoh_mix = clamp_finite_f32(self.up_zoh_mix, 0.0, 1.0, 0.0);
    }

    /// Hydrates oversampler parameters from [`CrossbarWire`] honoring `SONA_META_OS_EXT_PRESENT`.
    pub fn from_crossbar_wire(cb: &CrossbarWire) -> Self {
        if (cb.sona_meta_flags.get() & SONA_META_OS_EXT_PRESENT) == 0 {
            return Self::default();
        }
        let mut p = Self {
            up_quad_coeff: cb.os_up_quad_coeff.get(),
            dec_c_prev1: cb.os_dec_c_prev1.get(),
            dec_c0: cb.os_dec_c0.get(),
            dec_c1: 0.0,
            up_pre_emph: cb.os_up_pre_emph.get(),
            dec_aa_mix: cb.os_dec_aa_mix.get(),
            up_aa_mix: cb.os_up_aa_mix.get(),
            up_zoh_mix: cb.os_up_zoh_mix.get(),
        };
        p.clamp_in_place();
        p
    }
}

/// Vectorized stereo `2x` upsampler + 4-channel `[pre_cab_L, pre_cab_R, direct_s2_L, direct_s2_R]`
/// decimator (`144 B`, `0 FDIV`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SonaStereoOversampler2x {
    /// Prior sub-frame-1 output `[pre_cab_L, pre_cab_R, direct_s2_L, direct_s2_R]` (`y[2n - 1]`).
    pub dec_prev1: [f32; 4],
    /// Upsampler stage 0 all-pass state (`[HB_A0[0], HB_A0[0], HB_A1[0], HB_A1[0]]`).
    pub up_hb_s0: [f32; 4],
    /// Upsampler stage 1 all-pass state (`[HB_A0[1], HB_A0[1], HB_A1[1], HB_A1[1]]`).
    pub up_hb_s1: [f32; 4],
    /// Decimator all-pass state `[HB_A0[0], HB_A0[1], HB_A1[0], HB_A1[1]]` across 4 channels.
    pub dec_hb_state: [[f32; 4]; 4],
    /// Previous raw input `[L, R]` for causal pre-emphasis.
    pub up_raw_prev: [f32; 2],
    /// Previous pre-emphasized input `x_p[n - 1]`.
    pub up_prev1: [f32; 2],
    /// Twice-delayed pre-emphasized input `x_p[n - 2]`.
    pub up_prev2: [f32; 2],
    _pad: [f32; 2],
}

impl SonaStereoOversampler2x {
    /// Resets all upsampler and decimator states to `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Warm-seeds both upsampler and 4-channel decimator to DC steady state (`0 FDIV`) for
    /// pop-free `1x -> 2x` tier switches (`router_and_crossbar_topology.md §5`).
    #[inline]
    pub fn warm_seed(&mut self, last_in: [f32; 2], last_s2_dir: [f32; 4]) {
        let in_safe = [
            flush_recursive_state_f32(last_in[0]),
            flush_recursive_state_f32(last_in[1]),
        ];
        let s2_safe = last_s2_dir.map(flush_recursive_state_f32);
        self.up_raw_prev = in_safe;
        self.up_prev1 = in_safe;
        self.up_prev2 = in_safe;

        let in_pair = [in_safe[0], in_safe[1], in_safe[0], in_safe[1]];
        for c in 0..4 {
            self.up_hb_s0[c] = flush_recursive_state_f32(in_pair[c] * INV_UP_HB_S0[c]);
            self.up_hb_s1[c] = flush_recursive_state_f32(in_pair[c] * INV_UP_HB_S1[c]);
        }

        self.dec_prev1 = s2_safe;
        for (stage, &inv) in INV_DEC_HB.iter().enumerate() {
            for c in 0..4 {
                self.dec_hb_state[stage][c] = flush_recursive_state_f32(s2_safe[c] * inv);
            }
        }
    }

    /// Upsamples one `1x` stereo frame `(x_l, x_r)` into two chronological `2x` stereo frames.
    #[inline(always)]
    pub fn upsample_stereo<B: MathBackend>(
        &mut self,
        x_l: f32,
        x_r: f32,
        params: &SonaOversamplerParams,
    ) -> ([f32; 2], [f32; 2]) {
        let one_minus_pe = 1.0 - params.up_pre_emph;
        let xp_l = one_minus_pe * x_l + params.up_pre_emph * self.up_raw_prev[0];
        let xp_r = one_minus_pe * x_r + params.up_pre_emph * self.up_raw_prev[1];
        self.up_raw_prev = [
            flush_recursive_state_f32(x_l),
            flush_recursive_state_f32(x_r),
        ];

        let d2_l = xp_l - 2.0 * self.up_prev1[0] + self.up_prev2[0];
        let d2_r = xp_r - 2.0 * self.up_prev1[1] + self.up_prev2[1];
        let c_p1 = 0.5 * (1.0 - params.up_zoh_mix);
        let one_minus_cp1 = 1.0 - c_p1;
        let x0_poly_l =
            one_minus_cp1 * xp_l + c_p1 * self.up_prev1[0] + params.up_quad_coeff * d2_l;
        let x0_poly_r =
            one_minus_cp1 * xp_r + c_p1 * self.up_prev1[1] + params.up_quad_coeff * d2_r;
        self.up_prev2 = self.up_prev1;
        self.up_prev1 = [
            flush_recursive_state_f32(xp_l),
            flush_recursive_state_f32(xp_r),
        ];

        let xp_v = B::Vec4::splat_pair(xp_l, xp_r);
        let mut s0 = B::Vec4::load(&self.up_hb_s0);
        let mut s1 = B::Vec4::load(&self.up_hb_s1);
        let m_v = step_allpass_1_vec4::<B::Vec4>(xp_v, B::Vec4::load(&UP_HB_C0), &mut s0);
        let hb_v = step_allpass_1_vec4::<B::Vec4>(m_v, B::Vec4::load(&UP_HB_C1), &mut s1);
        s0.store(&mut self.up_hb_s0);
        s1.store(&mut self.up_hb_s1);

        let poly_v = B::Vec4::from_array([x0_poly_l, x0_poly_r, xp_l, xp_r]);
        let up_v = poly_v.fma(B::Vec4::splat(params.up_aa_mix), hb_v.sub(poly_v));
        let [xl0, xr0, xl1, xr1] = up_v.to_array();
        ([xl0, xr0], [xl1, xr1])
    }

    /// Decimates two `2x` 4-channel sub-frames `sub0`, `sub1` (`[pre_cab_L, pre_cab_R, direct_s2_L, direct_s2_R]`)
    /// back into one `1x` 4-channel frame.
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
        flush_recursive_state_vec4::<B::Vec4>(y1).store(&mut self.dec_prev1);

        let y_poly = prev1
            .mul(B::Vec4::splat(params.dec_c_prev1))
            .fma(B::Vec4::splat(params.dec_c0), y0)
            .fma(B::Vec4::splat(params.dec_c1), y1);

        let m0 =
            step_allpass_1_arr4::<B::Vec4>(y1, B::Vec4::splat(HB_A0[0]), &mut self.dec_hb_state[0]);
        let b0 =
            step_allpass_1_arr4::<B::Vec4>(m0, B::Vec4::splat(HB_A0[1]), &mut self.dec_hb_state[1]);
        let m1 =
            step_allpass_1_arr4::<B::Vec4>(y0, B::Vec4::splat(HB_A1[0]), &mut self.dec_hb_state[2]);
        let b1 =
            step_allpass_1_arr4::<B::Vec4>(m1, B::Vec4::splat(HB_A1[1]), &mut self.dec_hb_state[3]);
        let y_hb = flush_recursive_state_vec4::<B::Vec4>(B::Vec4::splat(0.5).mul(b0.add(b1)));

        y_poly
            .fma(B::Vec4::splat(params.dec_aa_mix), y_hb.sub(y_poly))
            .to_array()
    }
}

/// Shared 4-cache-line (`256 B`) `OS2X` + `Op-P` midpoint-interpolation hot block (`Full..Ultra`,
/// `sona_architecture.md §2.3`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaOs2xHot {
    /// Clamped `2x` oversampler parameters (`32 B`).
    pub os_params: SonaOversamplerParams,
    /// Stereo `2x` upsampler + 4-channel decimator recursive state (`144 B`).
    pub os_stereo: SonaStereoOversampler2x,
    /// Previous `Op-P` stereo quadrature `[I_L, Q_L, I_R, Q_R]` (`16 B`).
    pub op_p_prev_iq_lr: [f32; 4],
    /// Previous `Op-P` `T1` 4-channel sub-harmonic features (`16 B`).
    pub op_p_prev_feat: [f32; 4],
    _pad: [u8; 48],
}

const _: () = assert!(size_of::<SonaOversamplerParams>() == 32);
const _: () = assert!(align_of::<SonaOversamplerParams>() == 16);
const _: () = assert!(size_of::<SonaStereoOversampler2x>() == 144);
const _: () = assert!(align_of::<SonaStereoOversampler2x>() == 16);
const _: () = assert!(size_of::<SonaOs2xHot>() == 256);
const _: () = assert!(align_of::<SonaOs2xHot>() == 64);
const _: () = assert!(offset_of!(SonaOs2xHot, os_params) == 0);
const _: () = assert!(offset_of!(SonaOs2xHot, os_stereo) == 32);
const _: () = assert!(offset_of!(SonaOs2xHot, op_p_prev_iq_lr) == 176);
const _: () = assert!(offset_of!(SonaOs2xHot, op_p_prev_feat) == 192);
const _: () = assert!(offset_of!(SonaOs2xHot, _pad) == 208);

impl Default for SonaOs2xHot {
    fn default() -> Self {
        Self::new(SonaOversamplerParams::default())
    }
}

impl SonaOs2xHot {
    /// Creates a zero-state [`SonaOs2xHot`] block configured with `os_params`.
    #[inline]
    pub const fn new(os_params: SonaOversamplerParams) -> Self {
        Self {
            os_params,
            os_stereo: SonaStereoOversampler2x {
                dec_prev1: [0.0; 4],
                up_hb_s0: [0.0; 4],
                up_hb_s1: [0.0; 4],
                dec_hb_state: [[0.0; 4]; 4],
                up_raw_prev: [0.0; 2],
                up_prev1: [0.0; 2],
                up_prev2: [0.0; 2],
                _pad: [0.0; 2],
            },
            op_p_prev_iq_lr: [0.0; 4],
            op_p_prev_feat: [0.0; 4],
            _pad: [0; 48],
        }
    }

    /// Hydrates a zero-state [`SonaOs2xHot`] block from [`CrossbarWire`].
    #[inline]
    pub fn from_crossbar_wire(cb: &CrossbarWire) -> Self {
        Self::new(SonaOversamplerParams::from_crossbar_wire(cb))
    }

    /// Resets oversampler state and previous `Op-P` interpolation buffers to `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        self.os_stereo.reset();
        self.op_p_prev_iq_lr = [0.0; 4];
        self.op_p_prev_feat = [0.0; 4];
    }

    /// Warm-seeds the internal `2x` oversampler to DC steady state for `1x -> 2x` tier switches.
    #[inline]
    pub fn warm_seed(&mut self, last_in: [f32; 2], last_s2_dir: [f32; 4]) {
        self.os_stereo.warm_seed(last_in, last_s2_dir);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U32};

    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};

    #[test]
    fn test_os2x_wire_clamp_zero_rest_and_simd_scalar_parity() {
        let mut cb = CrossbarWire::new_zeroed();
        cb.os_up_quad_coeff = F32::new(f32::NAN);
        cb.os_dec_c_prev1 = F32::new(f32::INFINITY);
        let p_clear = SonaOversamplerParams::from_crossbar_wire(&cb);
        assert_eq!(p_clear, SonaOversamplerParams::default());

        cb.sona_meta_flags = U32::new(SONA_META_OS_EXT_PRESENT);
        let p_nan = SonaOversamplerParams::from_crossbar_wire(&cb);
        assert_eq!(p_nan.up_quad_coeff.to_bits(), 0);
        assert_eq!(p_nan.dec_c_prev1.to_bits(), 0.15_f32.to_bits());
        assert_eq!(p_nan.dec_c0.to_bits(), 0);
        assert!((p_nan.dec_c1 - 0.85).abs() < 1.0e-6);

        let mut hot = SonaOs2xHot::from_crossbar_wire(&cb);
        let (u0, u1) =
            hot.os_stereo
                .upsample_stereo::<DefaultMathBackend>(0.0, 0.0, &hot.os_params);
        assert_eq!(u0.map(f32::to_bits), [0, 0]);
        assert_eq!(u1.map(f32::to_bits), [0, 0]);
        let d0 =
            hot.os_stereo
                .downsample_4ch::<DefaultMathBackend>([0.0; 4], [0.0; 4], &hot.os_params);
        assert_eq!(d0.map(f32::to_bits), [0; 4]);

        let p_aa = SonaOversamplerParams {
            up_pre_emph: 0.15,
            dec_aa_mix: 1.0,
            up_aa_mix: 1.0,
            ..SonaOversamplerParams::default()
        };
        let mut os_simd = SonaStereoOversampler2x::default();
        let mut os_scal = SonaStereoOversampler2x::default();
        for n in 0..32 {
            let xl = ((n as f32) * 0.1).clamp(-1.0, 1.0);
            let xr = -xl;
            let (s0, s1) = os_simd.upsample_stereo::<DefaultMathBackend>(xl, xr, &p_aa);
            let (c0, c1) = os_scal.upsample_stereo::<ScalarBackend>(xl, xr, &p_aa);
            assert!((s0[0] - c0[0]).abs() < 1.0e-6 && (s1[1] - c1[1]).abs() < 1.0e-6);
            let ds = os_simd.downsample_4ch::<DefaultMathBackend>(
                [s0[0], s0[1], xl, xr],
                [s1[0], s1[1], xl, xr],
                &p_aa,
            );
            let dc = os_scal.downsample_4ch::<ScalarBackend>(
                [c0[0], c0[1], xl, xr],
                [c1[0], c1[1], xl, xr],
                &p_aa,
            );
            for k in 0..4 {
                assert!((ds[k] - dc[k]).abs() < 1.0e-6);
            }
        }
        hot.reset();
        assert_eq!(hot.op_p_prev_iq_lr.map(f32::to_bits), [0; 4]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_warm_seed_dc_equilibrium_and_zero_rest(
            dc_in in prop::array::uniform2(-8.0_f32..=8.0_f32),
            dc_s2 in prop::array::uniform4(-8.0_f32..=8.0_f32),
            up_quad in -0.20_f32..=0.05_f32,
            dec_prev1 in 0.0_f32..=0.28_f32,
            dec_c0 in 0.0_f32..=0.55_f32,
            up_pe in -0.28_f32..=1.0_f32,
            dec_aa in 0.0_f32..=1.0_f32,
            up_aa in 0.0_f32..=1.0_f32,
            up_zoh in 0.0_f32..=1.0_f32,
        ) {
            let mut params = SonaOversamplerParams {
                up_quad_coeff: up_quad,
                dec_c_prev1: dec_prev1,
                dec_c0,
                dec_c1: 0.0,
                up_pre_emph: up_pe,
                dec_aa_mix: dec_aa,
                up_aa_mix: up_aa,
                up_zoh_mix: up_zoh,
            };
            params.clamp_in_place();

            let mut zero_os = SonaStereoOversampler2x::default();
            zero_os.warm_seed([0.0, 0.0], [0.0; 4]);
            let (z0, z1) = zero_os.upsample_stereo::<DefaultMathBackend>(0.0, 0.0, &params);
            prop_assert_eq!(z0.map(f32::to_bits), [0, 0]);
            prop_assert_eq!(z1.map(f32::to_bits), [0, 0]);
            let zd = zero_os.downsample_4ch::<DefaultMathBackend>([0.0; 4], [0.0; 4], &params);
            prop_assert_eq!(zd.map(f32::to_bits), [0; 4]);

            let mut hot = SonaOs2xHot::new(params);
            hot.warm_seed(dc_in, dc_s2);
            let (u0, u1) = hot
                .os_stereo
                .upsample_stereo::<DefaultMathBackend>(dc_in[0], dc_in[1], &hot.os_params);
            for ch in 0..2 {
                let tol = 1.0e-5 * dc_in[ch].abs().max(1.0);
                prop_assert!((u0[ch] - dc_in[ch]).abs() <= tol);
                prop_assert!((u1[ch] - dc_in[ch]).abs() <= tol);
            }
            let d = hot
                .os_stereo
                .downsample_4ch::<DefaultMathBackend>(dc_s2, dc_s2, &hot.os_params);
            for ch in 0..4 {
                let tol = 1.0e-5 * dc_s2[ch].abs().max(1.0);
                prop_assert!((d[ch] - dc_s2[ch]).abs() <= tol);
            }
        }
    }
}
