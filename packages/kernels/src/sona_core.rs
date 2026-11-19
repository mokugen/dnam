//! Shared `#![no_std]` `SONA` state-space core kernels (`ref[dsp.op-n.stage2.sub-stages]`,
//! `ref[dsp.op-n.stage2.conv6-simd]`, `ref[dsp.op-n.commuted.dc-equilibrium]`,
//! `ref[repo.code.shared-nostd-kernels]`):
//! - [`tpt_svf`]: Unconditionally passive `2x2` ZDF `TPT-SVF` step, prewarping, and forward/adjoint scans.
//! - [`paralru`]: Contractive `1D` scalar `ParaLRU` step, zero-DC soft rectifier, Newton & adjoint scans.
//! - [`poly_adaa`]: Pure-`f32` divided-difference `PolyADAA1` and $C^2$ wavefolder kernels.

pub mod paralru;
pub mod poly_adaa;
pub mod tpt_svf;

use crate::simd::{KernelVec4, MAX_PARALRU_R_F32, StereoVec8};

pub use paralru::{
    LinearScanElem, ParaLruParams, ParaLruTileGrads, SOFT_RECT_EPS, apply_linear_scan,
    compose_linear_scan, compose_paralru_adjoint_scan, paralru_adjoint_step_elem,
    paralru_newton_step_elem, run_paralru_adjoint_scan_tile, run_paralru_newton_scan_tile,
    soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8, soft_rect_zero_dc_vec4, step_paralru_stereo8,
    step_paralru_vec4,
};
pub use poly_adaa::{
    eval_c2_wavefolder_adaa_stereo8, eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4,
    eval_poly_adaa1_quad_batched, eval_poly_adaa1_stereo8,
};
pub use tpt_svf::{
    TptAdjointElem, TptScanElem, TptSvfCoeffs4, TptSvfOutputs, apply_tpt_adjoint, apply_tpt_init,
    compose_tpt_adjoint_scan, compose_tpt_scan, prewarp_tan_pi_f_over_fs_f32, step_tpt_svf_stereo8,
    step_tpt_svf_vec4,
};

/// Simultaneous 4-mode `TPT-SVF` + direct-feed tone mixing weights (`ref[dsp.op-n.stage2.tone-basis]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaToneWeights4<V: KernelVec4> {
    pub lp: V,
    pub bp: V,
    pub hp: V,
    pub ap: V,
    pub dir: V,
}

impl<V: KernelVec4> SonaToneWeights4<V> {
    /// Selects pure low-pass output (`lp = 1.0`, remaining taps `0.0`).
    #[inline]
    #[must_use]
    pub fn lp_only() -> Self {
        Self {
            lp: V::splat(1.0),
            bp: V::zero(),
            hp: V::zero(),
            ap: V::zero(),
            dir: V::zero(),
        }
    }

    /// Selects unity direct dry bypass (`dir = 1.0`, SVF taps `0.0`).
    #[inline]
    #[must_use]
    pub fn direct_only() -> Self {
        Self {
            lp: V::zero(),
            bp: V::zero(),
            hp: V::zero(),
            ap: V::zero(),
            dir: V::splat(1.0),
        }
    }
}

/// Vectorized `1D` scalar `ParaLRU` ballistics for 4 SIMD lanes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaParaLruCoeffs4<V: KernelVec4> {
    pub alpha_rel: V,
    pub alpha_atk: V,
    pub kappa: V,
}

impl<V: KernelVec4> SonaParaLruCoeffs4<V> {
    /// Clamps 4-lane `ParaLRU` coefficients into the unconditionally contractive domain
    /// (`alpha_rel in [0, MAX_PARALRU_R_F32]`, `alpha_atk, kappa in [0, 1]`).
    #[inline]
    #[must_use]
    pub fn from_clamped(alpha_rel: V, alpha_atk: V, kappa: V) -> Self {
        let zero = V::zero();
        let one = V::splat(1.0);
        Self {
            alpha_rel: alpha_rel.clamp(zero, V::splat(MAX_PARALRU_R_F32)),
            alpha_atk: alpha_atk.clamp(zero, one),
            kappa: kappa.clamp(zero, one),
        }
    }
}

/// Complete parameter bundle for one 4-lane stereo `SONA` core group (`TPT-SVF -> ParaLRU -> PolyADAA1`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaBlockParamsStereo4<V: KernelVec4> {
    pub svf: TptSvfCoeffs4<V>,
    pub tone: SonaToneWeights4<V>,
    pub lru: SonaParaLruCoeffs4<V>,
    /// Pre-activation linear scale `(2/3) * d` folded at the `3 kHz` control boundary.
    pub adaa_drive: V,
}

/// Recursive stereo state for one 4-lane `SONA` core group (`S: StereoVec8`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaBlockStateStereo8<S: StereoVec8> {
    pub svf_s1: S,
    pub svf_s2: S,
    pub lru_v: S,
    pub adaa_prev: S,
}

impl<S: StereoVec8> SonaBlockStateStereo8<S> {
    /// Initializes the block state at analytical zero-DC resting equilibrium (`+0.0f32`).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            svf_s1: S::zero(),
            svf_s2: S::zero(),
            lru_v: S::zero(),
            adaa_prev: S::zero(),
        }
    }
}

/// Features emitted by [`step_sona_block_stereo8`] for direct-register `VarPro` readouts and
/// downstream Matryoshka blocks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaBlockOutputStereo8<S: StereoVec8> {
    pub lru_state: S,
    pub adaa_odd: S,
    pub adaa_even: S,
}

/// Executes one stereo `4`-lane `SONA` core block step (`TPT-SVF -> ParaLRU -> PolyADAA1`) in
/// `StereoVec8` registers (`0 FDIV`, `0 FSQRT`, `0-ULP` cross-channel symmetry).
#[inline(always)]
pub fn step_sona_block_stereo8<S: StereoVec8>(
    u: S,
    state: &mut SonaBlockStateStereo8<S>,
    params: &SonaBlockParamsStereo4<S::Half>,
) -> SonaBlockOutputStereo8<S> {
    let svf_out = step_tpt_svf_stereo8(u, &mut state.svf_s1, &mut state.svf_s2, &params.svf);
    let tone_in = svf_out
        .lp
        .mul(S::broadcast_half(params.tone.lp))
        .fma(svf_out.bp, S::broadcast_half(params.tone.bp))
        .fma(svf_out.hp, S::broadcast_half(params.tone.hp))
        .fma(svf_out.ap, S::broadcast_half(params.tone.ap))
        .fma(u, S::broadcast_half(params.tone.dir));

    let lru_state = step_paralru_stereo8(
        tone_in,
        &mut state.lru_v,
        params.lru.alpha_rel,
        params.lru.alpha_atk,
        params.lru.kappa,
    );

    let r_curr = lru_state.mul(S::broadcast_half(params.adaa_drive));
    let (adaa_odd, adaa_even) = eval_poly_adaa1_stereo8(state.adaa_prev, r_curr);
    state.adaa_prev = r_curr;

    SonaBlockOutputStereo8 {
        lru_state,
        adaa_odd,
        adaa_even,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};

    #[test]
    fn test_sona_block_zero_dc_equilibrium_and_dual_mono_parity() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.01),
            prewarp_tan_pi_f_over_fs_f32(0.05),
            prewarp_tan_pi_f_over_fs_f32(0.15),
            prewarp_tan_pi_f_over_fs_f32(0.35),
        ]);
        let k = GpuVec4::from_array([
            0.5,
            core::f32::consts::FRAC_1_SQRT_2,
            1.0,
            core::f32::consts::SQRT_2,
        ]);
        let params = SonaBlockParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4::lp_only(),
            lru: SonaParaLruCoeffs4::from_clamped(
                GpuVec4::from_array([0.99, 0.95, 0.9, 1.5]),
                GpuVec4::from_array([0.2, 0.5, 0.8, -0.1]),
                GpuVec4::from_array([0.0, 0.5, 1.0, 1.2]),
            ),
            adaa_drive: GpuVec4::from_array([0.5, 1.0, 2.5, 6.0]),
        };

        let mut state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for _ in 0..1_000 {
            let out = step_sona_block_stereo8(InterleavedStereo8::zero(), &mut state, &params);
            let (lru_l, lru_r) = out.lru_state.into_lr();
            let (odd_l, odd_r) = out.adaa_odd.into_lr();
            let (even_l, even_r) = out.adaa_even.into_lr();
            assert_eq!(lru_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(lru_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(odd_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(odd_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even_r.to_array().map(f32::to_bits), [0; 4]);
        }

        // Verify direct_only tone weights and dual-mono 0-ULP parity.
        let mut mono_params = params;
        mono_params.tone = SonaToneWeights4::direct_only();
        let mut st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for n in 0..64 {
            let sample = ((n as f32) * 0.23).sin() * 1.5;
            let u = InterleavedStereo8::splat_lr(sample, sample);
            let out = step_sona_block_stereo8(u, &mut st, &mono_params);
            let (ol, or) = out.adaa_odd.into_lr();
            let (el, er) = out.adaa_even.into_lr();
            assert_eq!(ol.to_array(), or.to_array());
            assert_eq!(el.to_array(), er.to_array());
        }
    }

    #[test]
    fn test_sona_block_stereo8_matches_vec4_and_bibo_recovery() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(80.0 / 48_000.0),
            prewarp_tan_pi_f_over_fs_f32(500.0 / 48_000.0),
            prewarp_tan_pi_f_over_fs_f32(2_400.0 / 48_000.0),
            prewarp_tan_pi_f_over_fs_f32(8_000.0 / 48_000.0),
        ]);
        let k = GpuVec4::from_array([1.0 / 12.0, 0.25, 0.707_106_77, 1.2]);
        let params = SonaBlockParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::splat(0.4),
                bp: GpuVec4::splat(0.3),
                hp: GpuVec4::splat(-0.2),
                ap: GpuVec4::splat(0.1),
                dir: GpuVec4::splat(0.25),
            },
            lru: SonaParaLruCoeffs4::from_clamped(
                GpuVec4::from_array([0.98, 0.96, 0.94, 0.9]),
                GpuVec4::from_array([0.1, 0.35, 0.65, 0.85]),
                GpuVec4::from_array([0.2, 0.5, 0.8, 1.0]),
            ),
            adaa_drive: GpuVec4::from_array([0.4, 1.0, 3.0, 8.0]),
        };

        let mut st_state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut s1_l = GpuVec4::zero();
        let mut s2_l = GpuVec4::zero();
        let mut v_l = GpuVec4::zero();
        let mut prev_l = GpuVec4::zero();
        let mut s1_r = GpuVec4::zero();
        let mut s2_r = GpuVec4::zero();
        let mut v_r = GpuVec4::zero();
        let mut prev_r = GpuVec4::zero();

        // Burst of +/-100.0 asymmetric stereo audio (256 frames) + 256 zero frames verifying 0-ULP parity.
        for n in 0..512 {
            let (xl, xr) = if n < 256 {
                let t = n as f32;
                (100.0 * (0.11 * t).sin(), -75.0 * (0.19 * t).cos())
            } else {
                (0.0, 0.0)
            };
            let ul = GpuVec4::splat(xl);
            let ur = GpuVec4::splat(xr);

            let out_st = step_sona_block_stereo8(
                InterleavedStereo8::from_lr(ul, ur),
                &mut st_state,
                &params,
            );

            let svf_l = step_tpt_svf_vec4(ul, &mut s1_l, &mut s2_l, &params.svf);
            let tone_l = svf_l
                .lp
                .mul(params.tone.lp)
                .fma(svf_l.bp, params.tone.bp)
                .fma(svf_l.hp, params.tone.hp)
                .fma(svf_l.ap, params.tone.ap)
                .fma(ul, params.tone.dir);
            let lru_l = step_paralru_vec4(
                tone_l,
                &mut v_l,
                params.lru.alpha_rel,
                params.lru.alpha_atk,
                params.lru.kappa,
            );
            let r_l = lru_l.mul(params.adaa_drive);
            let (odd_l, even_l, next_prev_l) = eval_poly_adaa1_odd_even_vec4(prev_l, r_l);
            prev_l = next_prev_l;

            let svf_r = step_tpt_svf_vec4(ur, &mut s1_r, &mut s2_r, &params.svf);
            let tone_r = svf_r
                .lp
                .mul(params.tone.lp)
                .fma(svf_r.bp, params.tone.bp)
                .fma(svf_r.hp, params.tone.hp)
                .fma(svf_r.ap, params.tone.ap)
                .fma(ur, params.tone.dir);
            let lru_r = step_paralru_vec4(
                tone_r,
                &mut v_r,
                params.lru.alpha_rel,
                params.lru.alpha_atk,
                params.lru.kappa,
            );
            let r_r = lru_r.mul(params.adaa_drive);
            let (odd_r, even_r, next_prev_r) = eval_poly_adaa1_odd_even_vec4(prev_r, r_r);
            prev_r = next_prev_r;

            let (st_lru_l, st_lru_r) = out_st.lru_state.into_lr();
            let (st_odd_l, st_odd_r) = out_st.adaa_odd.into_lr();
            let (st_even_l, st_even_r) = out_st.adaa_even.into_lr();
            assert_eq!(st_lru_l.to_array(), lru_l.to_array());
            assert_eq!(st_lru_r.to_array(), lru_r.to_array());
            assert_eq!(st_odd_l.to_array(), odd_l.to_array());
            assert_eq!(st_odd_r.to_array(), odd_r.to_array());
            assert_eq!(st_even_l.to_array(), even_l.to_array());
            assert_eq!(st_even_r.to_array(), even_r.to_array());

            for lane in 0..4 {
                assert!(odd_l.to_array()[lane].abs() <= 1.0 + 1.0e-5);
                assert!(odd_r.to_array()[lane].abs() <= 1.0 + 1.0e-5);
                assert!(even_l.to_array()[lane].abs() <= 0.3);
                assert!(even_r.to_array()[lane].abs() <= 0.3);
            }
        }

        // Step 32,768 additional zero frames (> 14 decay time constants of 80 Hz Q=12 at 48 kHz).
        for _ in 0..32_768 {
            let _ = step_sona_block_stereo8(InterleavedStereo8::zero(), &mut st_state, &params);
        }

        // All recursive states across all 4 lanes decay back below 1e-3.
        let (fin_svf1_l, fin_svf1_r) = st_state.svf_s1.into_lr();
        let (fin_lru_l, fin_lru_r) = st_state.lru_v.into_lr();
        for lane in 0..4 {
            assert!(fin_svf1_l.to_array()[lane].abs() < 1.0e-3);
            assert!(fin_svf1_r.to_array()[lane].abs() < 1.0e-3);
            assert!(fin_lru_l.to_array()[lane].abs() < 1.0e-3);
            assert!(fin_lru_r.to_array()[lane].abs() < 1.0e-3);
        }
    }
}
