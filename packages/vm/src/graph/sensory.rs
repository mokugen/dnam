//! 4-lane `SensoryQuad` (`Op-M/N[Sense]`) continuous sensing bank and uniform pre-core stereo
//! input VCA (`#![forbid(unsafe_code)]`, `0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm` in RT,
//! `ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.router.lr4-crossover]`).
//!
//! Executes four learned `2x2` `ZDF`
//! `TPT-SVF` filters, full-wave contractive `ParaLRU` followers emitting non-negative envelopes
//! `e[0..4]` into `ControlBus` `q3`, and a stereo-linked input VCA `g_in in [0.0, 2.0]`.

#![forbid(unsafe_code)]

use core::mem::{align_of, size_of};

use dnam_kernels::{MAX_PARALRU_R_F32, TptSvfCoeffs4, step_paralru_vec4, step_tpt_svf_vec4};
use zerocopy::byteorder::little_endian::{F32, U16};

use crate::bin::arena_wire::CrossbarWire;
use crate::math::{
    MathBackend, ScalarVec4, SimdVec4, clamp_finite_f32, const_abs_f64, fp16_to_f32, powf_f64,
    rescale_tpt_svf_g_zoh,
};

const SENSE_SVF_G_MIN: f32 = 1.0e-5;
const SENSE_SVF_G_MAX: f32 = 64.0;
const SENSE_SVF_G_FALLBACK: f32 = 0.25;
const SENSE_SVF_K_MIN: f32 = 0.0;
const SENSE_SVF_K_MAX: f32 = 8.0;
const SENSE_SVF_K_FALLBACK: f32 = core::f32::consts::FRAC_1_SQRT_2;
const SENSE_IN_LIMIT: f32 = 4.0;
const SENSE_TONE_LIMIT: f32 = 16.0;
const SENSE_VCA_W_LIMIT: f32 = 512.0;
const SENSE_VCA_BIAS_LIMIT: f32 = 1.0;
const SENSE_VCA_GAIN_MIN: f32 = 0.0;
const SENSE_VCA_GAIN_MAX: f32 = 2.0;
const RHO_UNITY_EPS_F64: f64 = 1.0e-12;

const _: () = assert!(
    SENSE_SVF_G_MIN > 0.0
        && SENSE_SVF_G_MIN < SENSE_SVF_G_FALLBACK
        && SENSE_SVF_G_FALLBACK < SENSE_SVF_G_MAX
);
const _: () = assert!(
    SENSE_SVF_K_MIN >= 0.0
        && SENSE_SVF_K_FALLBACK > SENSE_SVF_K_MIN
        && SENSE_SVF_K_FALLBACK < SENSE_SVF_K_MAX
);
const _: () = assert!(SENSE_VCA_GAIN_MIN < 1.0 && 1.0 < SENSE_VCA_GAIN_MAX);

/// Hot `48 B` (`16 B`-aligned) recursive state for the 4-lane `SensoryQuad` (`Op-M/N[Sense]`)
/// sensing bank (`ref[dsp.router.ballistic-envelope-bank]`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SensoryQuadHot {
    /// Passive `TPT-SVF` first integrator state across 4 lanes.
    pub svf_s1: [f32; 4],
    /// Passive `TPT-SVF` second integrator state across 4 lanes.
    pub svf_s2: [f32; 4],
    /// Full-wave `ParaLRU` non-negative envelope state `e[0..4]` sampled into `ControlBus` `q3`.
    pub lru_v: [f32; 4],
}

const _: () = assert!(size_of::<SensoryQuadHot>() == 48);
const _: () = assert!(align_of::<SensoryQuadHot>() == 16);

impl SensoryQuadHot {
    /// Returns zero-initialized `SensoryQuadHot` resting equilibrium state.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            svf_s1: [0.0; 4],
            svf_s2: [0.0; 4],
            lru_v: [0.0; 4],
        }
    }

    /// Resets `TPT-SVF` and `ParaLRU` states back to analytical zero-DC resting equilibrium.
    #[inline]
    pub fn reset(&mut self) {
        self.svf_s1 = [0.0; 4];
        self.svf_s2 = [0.0; 4];
        self.lru_v = [0.0; 4];
    }

    /// Returns the 4-lane `ParaLRU` sensing envelopes `e[0..4]` for `ControlBus` `q3`.
    #[inline]
    #[must_use]
    pub const fn envelopes(&self) -> [f32; 4] {
        self.lru_v
    }
}

#[inline]
fn decode_fp16x4_clamped(src: &[U16; 4], lo: f32, hi: f32, fallback: f32) -> [f32; 4] {
    [
        clamp_finite_f32(fp16_to_f32(src[0].get()), lo, hi, fallback),
        clamp_finite_f32(fp16_to_f32(src[1].get()), lo, hi, fallback),
        clamp_finite_f32(fp16_to_f32(src[2].get()), lo, hi, fallback),
        clamp_finite_f32(fp16_to_f32(src[3].get()), lo, hi, fallback),
    ]
}

#[inline]
fn rescale_lru_pole_array(src: &[F32; 4], max_pole: f32, rho: f64, exact_48k: bool) -> [f32; 4] {
    let mut out = [0.0_f32; 4];
    for k in 0..4 {
        let raw = clamp_finite_f32(src[k].get(), 0.0, max_pole, 0.0);
        out[k] = if raw <= 0.0 {
            0.0
        } else if exact_48k {
            raw
        } else {
            (powf_f64(f64::from(raw), rho) as f32).clamp(0.0, max_pole)
        };
    }
    out
}

/// Cold-hydrated parameters for the 4-lane `SensoryQuad` (`Op-M/N[Sense]`) sensing bank and
/// uniform pre-core stereo input VCA (`ref[dsp.router.ballistic-envelope-bank]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensoryQuadCold {
    /// Rate-adapted `TPT-SVF` cutoff tangent gains `g` at `f_{s,vm}`.
    pub svf_g: [f32; 4],
    /// Clamped `TPT-SVF` damping factors `k`.
    pub svf_k: [f32; 4],
    /// Derived `TPT-SVF` `a1` coefficients (`1 / (1 + g(g + k))`).
    pub svf_a1: [f32; 4],
    /// Derived `TPT-SVF` `a2` coefficients (`g * a1`).
    pub svf_a2: [f32; 4],
    /// Derived `TPT-SVF` `a3` coefficients (`g * a2`).
    pub svf_a3: [f32; 4],
    /// Left channel stereo key mix weights (`[-4.0, 4.0]`).
    pub in_l: [f32; 4],
    /// Right channel stereo key mix weights (`[-4.0, 4.0]`).
    pub in_r: [f32; 4],
    /// `TPT-SVF` low-pass sensing tap weights (`[-16.0, 16.0]`).
    pub tone_lp: [f32; 4],
    /// `TPT-SVF` band-pass sensing tap weights (`[-16.0, 16.0]`).
    pub tone_bp: [f32; 4],
    /// `TPT-SVF` high-pass sensing tap weights (`[-16.0, 16.0]`).
    pub tone_hp: [f32; 4],
    /// ZOH-rescaled `ParaLRU` release poles (`[0.0, MAX_PARALRU_R_F32]`).
    pub lru_alpha_rel: [f32; 4],
    /// ZOH-rescaled `ParaLRU` attack poles (`[0.0, 1.0]`).
    pub lru_alpha_atk: [f32; 4],
    /// Clamped `ParaLRU` self-feedback gains `kappa` (`[0.0, 1.0]`).
    pub lru_kappa: [f32; 4],
    /// Input VCA envelope projection weights (`[-512.0, 512.0]`).
    pub vca_w: [f32; 4],
}

impl Default for SensoryQuadCold {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl SensoryQuadCold {
    /// Returns the safe identity cold configuration (`in_* = 0`, `tone_* = 0`, `lru_* = 0`,
    /// `vca_w = 0`, `svf_g = 0.25`, `svf_k = 1 / sqrt(2)`).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        let svf_g = [SENSE_SVF_G_FALLBACK; 4];
        let svf_k = [SENSE_SVF_K_FALLBACK; 4];
        let svf = TptSvfCoeffs4::from_gk(ScalarVec4(svf_g), ScalarVec4(svf_k));
        Self {
            svf_g,
            svf_k: svf.k.0,
            svf_a1: svf.a1.0,
            svf_a2: svf.a2.0,
            svf_a3: svf.a3.0,
            in_l: [0.0; 4],
            in_r: [0.0; 4],
            tone_lp: [0.0; 4],
            tone_bp: [0.0; 4],
            tone_hp: [0.0; 4],
            lru_alpha_rel: [0.0; 4],
            lru_alpha_atk: [0.0; 4],
            lru_kappa: [0.0; 4],
            vca_w: [0.0; 4],
        }
    }

    /// Hydrates `SensoryQuadCold` unconditionally from [`CrossbarWire`] with `rho_vm` ZOH
    /// conversion (`0 FDIV`, `0 FSQRT`, `0 libm`, `ref[container.dnam.load-validation]`).
    #[must_use]
    pub fn from_crossbar_wire(crossbar: &CrossbarWire, rho_vm: f64) -> Self {
        let rho = if rho_vm.is_finite() && rho_vm > 0.0 {
            rho_vm
        } else {
            1.0
        };
        let exact_48k = const_abs_f64(rho - 1.0) < RHO_UNITY_EPS_F64;

        let mut svf_g = [0.0_f32; 4];
        for k in 0..4 {
            let raw_g = clamp_finite_f32(
                crossbar.sense_svf_g[k].get(),
                SENSE_SVF_G_MIN,
                SENSE_SVF_G_MAX,
                SENSE_SVF_G_FALLBACK,
            );
            svf_g[k] = rescale_tpt_svf_g_zoh(raw_g, rho);
        }
        let svf_k = decode_fp16x4_clamped(
            &crossbar.sense_svf_k_fp16,
            SENSE_SVF_K_MIN,
            SENSE_SVF_K_MAX,
            SENSE_SVF_K_FALLBACK,
        );
        let svf = TptSvfCoeffs4::from_gk(ScalarVec4(svf_g), ScalarVec4(svf_k));

        let lru_alpha_rel = rescale_lru_pole_array(
            &crossbar.sense_lru_alpha_rel,
            MAX_PARALRU_R_F32,
            rho,
            exact_48k,
        );
        let lru_alpha_atk =
            rescale_lru_pole_array(&crossbar.sense_lru_alpha_atk, 1.0, rho, exact_48k);
        let lru_kappa = decode_fp16x4_clamped(&crossbar.sense_lru_kappa_fp16, 0.0, 1.0, 0.0);

        let in_l = decode_fp16x4_clamped(
            &crossbar.sense_in_l_fp16,
            -SENSE_IN_LIMIT,
            SENSE_IN_LIMIT,
            0.0,
        );
        let in_r = decode_fp16x4_clamped(
            &crossbar.sense_in_r_fp16,
            -SENSE_IN_LIMIT,
            SENSE_IN_LIMIT,
            0.0,
        );
        let tone_lp = decode_fp16x4_clamped(
            &crossbar.sense_tone_lp_fp16,
            -SENSE_TONE_LIMIT,
            SENSE_TONE_LIMIT,
            0.0,
        );
        let tone_bp = decode_fp16x4_clamped(
            &crossbar.sense_tone_bp_fp16,
            -SENSE_TONE_LIMIT,
            SENSE_TONE_LIMIT,
            0.0,
        );
        let tone_hp = decode_fp16x4_clamped(
            &crossbar.sense_tone_hp_fp16,
            -SENSE_TONE_LIMIT,
            SENSE_TONE_LIMIT,
            0.0,
        );
        let vca_w = decode_fp16x4_clamped(
            &crossbar.sense_vca_w_fp16,
            -SENSE_VCA_W_LIMIT,
            SENSE_VCA_W_LIMIT,
            0.0,
        );

        Self {
            svf_g,
            svf_k: svf.k.0,
            svf_a1: svf.a1.0,
            svf_a2: svf.a2.0,
            svf_a3: svf.a3.0,
            in_l,
            in_r,
            tone_lp,
            tone_bp,
            tone_hp,
            lru_alpha_rel,
            lru_alpha_atk,
            lru_kappa,
            vca_w,
        }
    }

    /// Steps 4 `SensoryQuad` (`Op-M/N[Sense]`) lanes and applies uniform stereo input VCA `g_in`
    /// (`0 FDIV`, `0 FSQRT`, `0 libm`, `0-ULP` identity when `g_in == 1.0`,
    /// `ref[dsp.router.ballistic-envelope-bank]`).
    #[inline(always)]
    pub fn step_frame<B: MathBackend>(
        &self,
        hot: &mut SensoryQuadHot,
        x_l: f32,
        x_r: f32,
        vca_bias_slot30: f32,
    ) -> (f32, f32) {
        let u_sense = B::Vec4::splat(x_l)
            .mul(B::Vec4::load(&self.in_l))
            .fma(B::Vec4::splat(x_r), B::Vec4::load(&self.in_r));

        let mut s1 = B::Vec4::load(&hot.svf_s1);
        let mut s2 = B::Vec4::load(&hot.svf_s2);
        let svf_k = B::Vec4::load(&self.svf_k);
        let svf = TptSvfCoeffs4 {
            a1: B::Vec4::load(&self.svf_a1),
            a2: B::Vec4::load(&self.svf_a2),
            a3: B::Vec4::load(&self.svf_a3),
            k: svf_k,
            two_k: svf_k.add(svf_k),
        };
        let out = step_tpt_svf_vec4(u_sense, &mut s1, &mut s2, &svf);
        let s_sense = out
            .lp
            .mul(B::Vec4::load(&self.tone_lp))
            .fma(out.bp, B::Vec4::load(&self.tone_bp))
            .fma(out.hp, B::Vec4::load(&self.tone_hp));

        let mut lru_v = B::Vec4::load(&hot.lru_v);
        let rect = s_sense.abs();
        let e = step_paralru_vec4(
            rect,
            &mut lru_v,
            B::Vec4::load(&self.lru_alpha_rel),
            B::Vec4::load(&self.lru_alpha_atk),
            B::Vec4::load(&self.lru_kappa),
        );

        s1.store(&mut hot.svf_s1);
        s2.store(&mut hot.svf_s2);
        lru_v.store(&mut hot.lru_v);

        let vca_prod = e.mul(B::Vec4::load(&self.vca_w)).to_array();
        let vca_dot = (vca_prod[0] + vca_prod[1]) + (vca_prod[2] + vca_prod[3]);
        let bias30 = clamp_finite_f32(
            vca_bias_slot30,
            -SENSE_VCA_BIAS_LIMIT,
            SENSE_VCA_BIAS_LIMIT,
            0.0,
        );
        let g_in = clamp_finite_f32(
            1.0 + bias30 + vca_dot,
            SENSE_VCA_GAIN_MIN,
            SENSE_VCA_GAIN_MAX,
            1.0,
        );
        if g_in == 1.0 {
            (x_l, x_r)
        } else {
            (g_in * x_l, g_in * x_r)
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    use super::*;
    use crate::math::{
        DefaultMathBackend, ScalarBackend, cold_atan_pos_f64, const_abs_f32, const_sin_cos_f64,
        const_tan_f64, f32_to_fp16, sqrt_f64,
    };

    /// verify[dsp.router.lr4-crossover]
    #[test]
    fn test_sensory_quad_layout_and_zero_init_passthrough_0ulp() {
        assert_eq!(size_of::<SensoryQuadHot>(), 48);
        assert_eq!(align_of::<SensoryQuadHot>(), 16);

        let mut hot_default = SensoryQuadHot::zero();
        let mut hot_wire = SensoryQuadHot::default();
        let cold_default = SensoryQuadCold::default();
        let cold_wire = SensoryQuadCold::from_crossbar_wire(&CrossbarWire::new_zeroed(), 1.0);

        for (x_l, x_r) in [(0.0_f32, -0.0_f32), (0.625, -0.375), (-3.5, 1.25)] {
            let (yl_d, yr_d) =
                cold_default.step_frame::<ScalarBackend>(&mut hot_default, x_l, x_r, 0.0);
            let (yl_w, yr_w) =
                cold_wire.step_frame::<DefaultMathBackend>(&mut hot_wire, x_l, x_r, 0.0);
            assert_eq!(yl_d.to_bits(), x_l.to_bits());
            assert_eq!(yr_d.to_bits(), x_r.to_bits());
            assert_eq!(yl_w.to_bits(), x_l.to_bits());
            assert_eq!(yr_w.to_bits(), x_r.to_bits());
            assert_eq!(hot_default.envelopes().map(f32::to_bits), [0; 4]);
            assert_eq!(hot_wire.envelopes().map(f32::to_bits), [0; 4]);
        }

        // Non-finite slot 30 bias falls back cleanly to 0.0 (g_in == 1.0 exact 0-ULP).
        let (yl_nan, yr_nan) =
            cold_wire.step_frame::<DefaultMathBackend>(&mut hot_wire, -0.0_f32, 0.25, f32::NAN);
        assert_eq!(yl_nan.to_bits(), (-0.0_f32).to_bits());
        assert_eq!(yr_nan.to_bits(), 0.25_f32.to_bits());
    }

    /// verify[dsp.op-t.ducking.vca]
    /// verify[dsp.router.ballistic-envelope-bank]
    /// verify[dsp.router.modulation-bus]
    /// verify[dsp.zoh.router-detector]
    #[test]
    fn test_sensory_quad_excitation_vca_clamp_zoh_and_reset() {
        let mut cb = CrossbarWire::new_zeroed();
        cb.sense_svf_g = [0.05, 0.25, 1.0, 4.0].map(F32::new);
        cb.sense_svf_k_fp16 = [0.5, 0.707, 1.0, 2.0].map(|v| U16::new(f32_to_fp16(v)));
        cb.sense_in_l_fp16 = [1.0, 0.5, 0.5, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        cb.sense_in_r_fp16 = [0.0, 0.5, -0.5, 1.0].map(|v| U16::new(f32_to_fp16(v)));
        cb.sense_tone_lp_fp16 = [1.0, 0.0, 0.0, 0.5].map(|v| U16::new(f32_to_fp16(v)));
        cb.sense_tone_bp_fp16 = [0.0, 1.0, 0.0, 0.5].map(|v| U16::new(f32_to_fp16(v)));
        cb.sense_tone_hp_fp16 = [0.0, 0.0, 1.0, -0.5].map(|v| U16::new(f32_to_fp16(v)));
        cb.sense_lru_alpha_rel = [0.99, 0.96, 0.90, 0.81].map(F32::new);
        cb.sense_lru_alpha_atk = [0.25, 0.50, 0.0, 0.75].map(F32::new);
        cb.sense_lru_kappa_fp16 = [0.25, 0.5, 0.0, 0.125].map(|v| U16::new(f32_to_fp16(v)));

        // ZOH 96 kHz (rho = 0.5) and 192 kHz (rho = 0.25) checks.
        let cold_96 = SensoryQuadCold::from_crossbar_wire(&cb, 0.5);
        let cold_192 = SensoryQuadCold::from_crossbar_wire(&cb, 0.25);
        for (idx, &g48) in [0.05_f32, 0.25, 1.0, 4.0].iter().enumerate() {
            let atan_g = cold_atan_pos_f64(f64::from(g48));
            let exp_96 = const_tan_f64(atan_g * 0.5) as f32;
            let exp_192 = const_tan_f64(atan_g * 0.25) as f32;
            assert!(const_abs_f32(cold_96.svf_g[idx] - exp_96) < 1.0e-5);
            assert!(const_abs_f32(cold_192.svf_g[idx] - exp_192) < 1.0e-5);
        }
        assert!(const_abs_f32(cold_96.lru_alpha_rel[0] - (sqrt_f64(0.99) as f32)) < 1.0e-6);
        assert!(const_abs_f32(cold_96.lru_alpha_atk[0] - (sqrt_f64(0.25) as f32)) < 1.0e-6);
        assert_eq!(cold_96.lru_alpha_atk[2].to_bits(), 0);
        assert!(
            const_abs_f32(cold_192.lru_alpha_rel[3] - (sqrt_f64(sqrt_f64(0.81)) as f32)) < 1.0e-6
        );

        // Positive envelope emission under tone excitation and resting zero after reset().
        let cold_48 = SensoryQuadCold::from_crossbar_wire(&cb, 1.0);
        let mut hot = SensoryQuadHot::zero();
        for n in 0..128 {
            let phase = (n as f64) * 0.21;
            let xl = const_sin_cos_f64(phase).0 as f32;
            let xr = const_sin_cos_f64(phase * 1.5).1 as f32;
            let (yl, yr) = cold_48.step_frame::<DefaultMathBackend>(&mut hot, xl, xr, 0.0);
            assert_eq!(yl.to_bits(), xl.to_bits());
            assert_eq!(yr.to_bits(), xr.to_bits());
        }
        let env = hot.envelopes();
        assert!(env.iter().all(|&e| e.is_finite() && e > 0.0));

        hot.reset();
        assert_eq!(hot.svf_s1.map(f32::to_bits), [0; 4]);
        assert_eq!(hot.svf_s2.map(f32::to_bits), [0; 4]);
        assert_eq!(hot.lru_v.map(f32::to_bits), [0; 4]);
        let (zr_l, zr_r) = cold_48.step_frame::<ScalarBackend>(&mut hot, 0.0, 0.0, 0.0);
        assert_eq!((zr_l.to_bits(), zr_r.to_bits()), (0, 0));
        assert_eq!(hot.envelopes().map(f32::to_bits), [0; 4]);

        // VCA ducking (clamped at 0.0) and expansion (clamped at 2.0) via vca_w + t[30].
        let mut cb_duck = cb;
        cb_duck.sense_vca_w_fp16 = [U16::new(f32_to_fp16(-512.0)); 4];
        let cold_duck = SensoryQuadCold::from_crossbar_wire(&cb_duck, 1.0);
        let mut hot_duck = SensoryQuadHot::zero();
        let mut min_ratio = f32::INFINITY;
        for n in 0..32 {
            let x = if n < 8 { 1.0 } else { 0.5 };
            let (yl, yr) = cold_duck.step_frame::<DefaultMathBackend>(&mut hot_duck, x, -x, -0.5);
            assert_eq!(yr, -yl);
            min_ratio = min_ratio.min(yl / x);
        }
        assert_eq!(min_ratio, 0.0);

        let mut cb_exp = cb;
        cb_exp.sense_vca_w_fp16 = [U16::new(f32_to_fp16(512.0)); 4];
        let cold_exp = SensoryQuadCold::from_crossbar_wire(&cb_exp, 1.0);
        let mut hot_exp = SensoryQuadHot::zero();
        let mut max_ratio = 0.0_f32;
        for _ in 0..32 {
            let (yl, yr) = cold_exp.step_frame::<DefaultMathBackend>(&mut hot_exp, 0.25, -0.5, 2.0);
            assert_eq!(yr, -2.0 * yl);
            max_ratio = max_ratio.max(yl / 0.25);
        }
        assert_eq!(max_ratio, 2.0);

        // Non-finite crossbar fallback verification.
        let mut cb_nan = CrossbarWire::new_zeroed();
        cb_nan.sense_svf_g = [F32::new(f32::NAN); 4];
        cb_nan.sense_svf_k_fp16 = [U16::new(0x7E00); 4];
        cb_nan.sense_vca_w_fp16 = [U16::new(0x7C00); 4];
        let cold_nan = SensoryQuadCold::from_crossbar_wire(&cb_nan, f64::NAN);
        assert_eq!(cold_nan.svf_g, [SENSE_SVF_G_FALLBACK; 4]);
        assert_eq!(cold_nan.svf_k, [SENSE_SVF_K_FALLBACK; 4]);
        assert_eq!(cold_nan.vca_w, [0.0; 4]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_sensory_quad_non_negative_envelopes_and_bounded_stereo_vca(
            in_l in -8.0_f32..=8.0_f32,
            in_r in -8.0_f32..=8.0_f32,
            vca_w in -600.0_f32..=600.0_f32,
            bias30 in -2.0_f32..=2.0_f32,
            x_l in -2.0_f32..=2.0_f32,
            x_r in -2.0_f32..=2.0_f32,
        ) {
            let mut cb = CrossbarWire::new_zeroed();
            cb.sense_svf_g = [0.02, 0.15, 0.8, 3.0].map(F32::new);
            cb.sense_svf_k_fp16 = [U16::new(f32_to_fp16(0.707)); 4];
            cb.sense_in_l_fp16 = [U16::new(f32_to_fp16(in_l)); 4];
            cb.sense_in_r_fp16 = [U16::new(f32_to_fp16(in_r)); 4];
            cb.sense_tone_lp_fp16 = [U16::new(f32_to_fp16(1.0)); 4];
            cb.sense_tone_bp_fp16 = [U16::new(f32_to_fp16(0.5)); 4];
            cb.sense_tone_hp_fp16 = [U16::new(f32_to_fp16(-0.5)); 4];
            cb.sense_lru_alpha_rel = [F32::new(0.98); 4];
            cb.sense_lru_alpha_atk = [F32::new(0.25); 4];
            cb.sense_lru_kappa_fp16 = [U16::new(f32_to_fp16(0.3)); 4];
            cb.sense_vca_w_fp16 = [U16::new(f32_to_fp16(vca_w)); 4];

            let cold = SensoryQuadCold::from_crossbar_wire(&cb, 1.0);
            let mut hot_s = SensoryQuadHot::zero();
            let mut hot_d = SensoryQuadHot::zero();
            for _ in 0..8 {
                let (sl, sr) = cold.step_frame::<ScalarBackend>(&mut hot_s, x_l, x_r, bias30);
                let (dl, dr) =
                    cold.step_frame::<DefaultMathBackend>(&mut hot_d, x_l, x_r, bias30);
                prop_assert!(sl.is_finite() && sr.is_finite());
                prop_assert!(dl.is_finite() && dr.is_finite());
                prop_assert!(const_abs_f32(sl) <= 2.0 * const_abs_f32(x_l) + 1.0e-5);
                prop_assert!(const_abs_f32(sr) <= 2.0 * const_abs_f32(x_r) + 1.0e-5);
                for (&es, &ed) in hot_s.envelopes().iter().zip(hot_d.envelopes().iter()) {
                    prop_assert!(es.is_finite() && es >= 0.0);
                    prop_assert!(ed.is_finite() && ed >= 0.0);
                    prop_assert!(const_abs_f32(es - ed) < 1.0e-4);
                }
            }
        }
    }
}
