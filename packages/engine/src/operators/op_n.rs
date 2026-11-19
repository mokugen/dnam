//! Op-N: Universal Nonlinear, Hysteretic & Commuted Cascade Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.polyadaa.clamp]`,
//! `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.ctrl.lobatto-basis]`, `ref[dsp.op-n.ctrl.trust-clip]`).

pub mod commuted_and_ctrl;
pub mod wavefolder_and_bitcrusher;

use crate::math::SimdVec4;

pub use commuted_and_ctrl::{
    eval_chebyshev_4harm, eval_commuted_vs_uncommuted_lti_cascade_f64, eval_lobatto_basis_3node,
    eval_stage2_stereo_coupling_16d, fold_stage2_latent_weights, trust_region_softclip_4d,
};
pub use dnam_kernels::{
    ParaLruParams, SOFT_RECT_EPS, SonaBlockOutputStereo8, SonaBlockParamsStereo4,
    SonaBlockStateStereo8, SonaGroupParamsStereo4, SonaParaLruCoeffs4, SonaToneWeights4,
    TptSvfCoeffs4, TptSvfOutputs, eval_c2_wavefolder_adaa_stereo8, eval_c2_wavefolder_adaa_vec4,
    eval_poly_adaa1_odd_even_vec4, eval_poly_adaa1_quad_batched, eval_poly_adaa1_stereo8,
    prewarp_tan_pi_f_over_fs_f32, soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8,
    soft_rect_zero_dc_vec4, step_paralru_stereo8, step_paralru_vec4, step_sona_block_stereo8,
    step_sona_group_stereo8, step_tpt_svf_stereo8, step_tpt_svf_vec4,
};
pub use wavefolder_and_bitcrusher::{
    FixedHzBitcrusher, QuadratureHysteresis2D, ZohBitcrusher, c2_wavefolder,
    eval_adaa_cubic_c2_wavefolder_f64, eval_adaa_cubic_c2_wavefolder_odd_f32,
    hertz_phase_parameters, hertz_phase_step, quantize_bitcrusher_nearest_ties_pos_inf,
    rescale_bitcrusher_phase_u128,
};

const TANH_R4_F64: [f64; 4] = crate::math::const_c1_clamped_tanh_coeffs(4.0);
const TANH_PADE_N1: f32 = TANH_R4_F64[0] as f32;
const TANH_PADE_N2: f32 = TANH_R4_F64[1] as f32;
const TANH_PADE_D1: f32 = TANH_R4_F64[2] as f32;
const TANH_PADE_D2: f32 = TANH_R4_F64[3] as f32;

/// Domain-specific SIMD backend trait for `Op-N` (division-free Padé `[5/4]` nonlinear activation, `0 FDIV`).
pub trait OpNBackend<V: SimdVec4> {
    /// Evaluates division-free 4-lane Padé `[5/4]` `tanh(x)` (`C^1` clamped at `±4`, `NR2` rcp, exact `+0.0`).
    fn tanh_pade5_4ch(x: V) -> V;
}

/// Division-free 4-lane SIMD Padé `[5/4]` `tanh(x)` helper implemented purely over [`SimdVec4`].
#[inline(always)]
#[must_use]
pub fn tanh_pade5_4ch_vec4<V: SimdVec4>(x: V) -> V {
    let coeffs = V::from_array([TANH_PADE_N1, TANH_PADE_N2, TANH_PADE_D1, TANH_PADE_D2]);
    let one = V::splat(1.0);
    let xc = x.clamp(V::splat(-4.0), V::splat(4.0));
    let s = xc.mul(xc);
    let s2 = s.mul(s);
    let num = one.fma_lane::<0>(s, coeffs).fma_lane::<1>(s2, coeffs);
    let den = one.fma_lane::<2>(s, coeffs).fma_lane::<3>(s2, coeffs);
    let inv_den = den.rcp_ge1_nr2();
    xc.mul(num).mul(inv_den).clamp(V::splat(-1.0), one)
}

impl OpNBackend<crate::math::ScalarVec4> for crate::math::ScalarBackend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::ScalarVec4) -> crate::math::ScalarVec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpNBackend<crate::math::NeonVec4> for crate::math::NeonBackend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::NeonVec4) -> crate::math::NeonVec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpNBackend<crate::math::Sve2Vec4> for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::Sve2Vec4) -> crate::math::Sve2Vec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpNBackend<crate::math::NeonVec4> for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::NeonVec4) -> crate::math::NeonVec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpNBackend<crate::math::Sse2Vec4> for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::Sse2Vec4) -> crate::math::Sse2Vec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpNBackend<crate::math::Sse42Vec4> for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::Sse42Vec4) -> crate::math::Sse42Vec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpNBackend<crate::math::Avx2FmaVec4> for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::Avx2FmaVec4) -> crate::math::Avx2FmaVec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(target_arch = "wasm32")]
impl OpNBackend<crate::math::WasmVec4> for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn tanh_pade5_4ch(x: crate::math::WasmVec4) -> crate::math::WasmVec4 {
        tanh_pade5_4ch_vec4(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, MathBackend, ScalarBackend};

    #[test]
    fn test_op_n_backend_tanh_pade5_zero_and_parity() {
        let vx_s = <ScalarBackend as MathBackend>::Vec4::from_array([-10.0, -1.0, 0.0, 1.5]);
        let vx_d = <DefaultMathBackend as MathBackend>::Vec4::from_array([-10.0, -1.0, 0.0, 1.5]);
        let y_s = ScalarBackend::tanh_pade5_4ch(vx_s).to_array();
        let y_d = DefaultMathBackend::tanh_pade5_4ch(vx_d).to_array();
        assert_eq!(y_s[2].to_bits(), 0);
        assert_eq!(y_d[2].to_bits(), 0);
        for i in 0..4 {
            assert!(y_s[i].abs() <= 1.0);
            assert!((y_s[i] - y_d[i]).abs() < 1.0e-5);
        }
    }

    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.op-n.polyadaa.rail-crossing]
    /// verify[dsp.op-n.polyadaa.linear-residual]
    /// verify[dsp.op-n.polyadaa.bspline-ac]
    /// verify[dsp.op-n.polyadaa.drive-bank]
    /// verify[dsp.op-n.stage2.conv6-simd]
    #[test]
    fn test_sona_polyadaa_and_group_kernel_contracts() {
        use crate::FtzDazGuard;
        use crate::operators::reference::eval_saturated_polyadaa_rail_crossing_f64;
        use dnam_kernels::{KernelVec4, StereoVec8};

        let _guard = FtzDazGuard::new();
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        type S8 = <DefaultMathBackend as MathBackend>::Stereo8;
        type VS = <ScalarBackend as MathBackend>::Vec4;
        type S8S = <ScalarBackend as MathBackend>::Stereo8;

        // 1. Clamp, midpoint, zero-DC bit identity, odd/even symmetry (bspline-ac), and linear-residual.
        let (odd0, even0, _) =
            eval_poly_adaa1_odd_even_vec4(<V as KernelVec4>::zero(), <V as KernelVec4>::zero());
        assert_eq!(KernelVec4::to_array(odd0).map(f32::to_bits), [0; 4]);
        assert_eq!(KernelVec4::to_array(even0).map(f32::to_bits), [0; 4]);

        let p_mid = <V as KernelVec4>::from_array([-3.0, -0.5, 0.25, 2.0]);
        let c_mid = <V as KernelVec4>::from_array([-2.0, 0.5, 0.25 + 1.0e-4, 3.0]);
        let (odd_mid, even_mid, _) = eval_poly_adaa1_odd_even_vec4(p_mid, c_mid);
        let (odd_neg, even_neg, _) =
            eval_poly_adaa1_odd_even_vec4(KernelVec4::neg(p_mid), KernelVec4::neg(c_mid));
        let odd_arr = KernelVec4::to_array(odd_mid);
        let even_arr = KernelVec4::to_array(even_mid);
        assert_eq!(odd_arr[0], -1.0);
        assert_eq!(odd_arr[3], 1.0);
        assert_eq!(even_arr[0].to_bits(), 0);
        assert_eq!(even_arr[3].to_bits(), 0);
        assert_eq!(
            KernelVec4::to_array(odd_neg).map(f32::to_bits),
            odd_arr.map(|v| if v == 0.0 { 0 } else { (-v).to_bits() })
        );
        assert_eq!(
            KernelVec4::to_array(even_neg).map(f32::to_bits),
            even_arr.map(f32::to_bits)
        );
        // Linear residual: small interior step converges to clamped cubic 1.5*u - 0.5*u^3 at segment midpoint.
        let u_lin = 0.5_f32 * (0.25 + (0.25 + 1.0e-4));
        let expected_lin = 1.5 * u_lin - 0.5 * u_lin * u_lin * u_lin;
        assert!((odd_arr[2] - expected_lin).abs() < 1.0e-5);

        // 2. Rail-crossing ADAA against exact f64 oracle (0 FDIV via Chebyshev rcp_pos_nr2).
        let rc_p = [-2.5_f32, -1.2, 0.3, -0.8];
        let rc_c = [1.7_f32, 0.4, 2.2, 1.5];
        let (rc_odd, rc_even, _) = eval_poly_adaa1_odd_even_vec4(
            <V as KernelVec4>::from_array(rc_p),
            <V as KernelVec4>::from_array(rc_c),
        );
        let rc_o_arr = KernelVec4::to_array(rc_odd);
        let rc_e_arr = KernelVec4::to_array(rc_even);
        for i in 0..4 {
            let (exp_o, exp_e, _) =
                eval_saturated_polyadaa_rail_crossing_f64(f64::from(rc_p[i]), f64::from(rc_c[i]));
            assert!((f64::from(rc_o_arr[i]) - exp_o).abs() < 2.0e-5);
            assert!((f64::from(rc_e_arr[i]) - exp_e).abs() < 2.0e-5);
        }

        // 3. Drive-bank & Conv6-SIMD StereoVec8 parity (`DefaultMathBackend` vs `ScalarBackend`).
        fn mk_group_params<KV: KernelVec4>(g_val: f32) -> SonaGroupParamsStereo4<KV> {
            SonaGroupParamsStereo4 {
                svf: TptSvfCoeffs4::from_gk(
                    KV::splat(prewarp_tan_pi_f_over_fs_f32(g_val)),
                    KV::splat(0.75),
                ),
                tone: SonaToneWeights4 {
                    lp: KV::from_array([0.4, 0.2, 0.1, 0.0]),
                    bp: KV::from_array([0.3, 0.4, 0.2, 0.4]),
                    hp: KV::from_array([0.0, 0.2, 0.4, 0.3]),
                    ap: KV::from_array([0.1, 0.0, 0.1, 0.1]),
                    dir: KV::from_array([0.2, 0.2, 0.2, 0.2]),
                },
                lru: SonaParaLruCoeffs4::from_clamped(
                    KV::from_array([0.96, 0.92, 0.85, 0.70]),
                    KV::from_array([0.25, 0.40, 0.60, 0.80]),
                    KV::from_array([0.10, 0.30, 0.55, 0.85]),
                ),
                rect_sign: KV::from_array([1.0, -1.0, 1.0, -1.0]),
                sag_depth: KV::from_array([0.1, 0.3, 0.5, 0.7]),
                adaa_drive: KV::from_array([0.8, 1.5, 2.5, 4.0]),
                lru_bias_drive: KV::from_array([0.0, 0.3, -0.4, 0.5]),
                wavefolder_mix: KV::from_array([0.0, 0.2, 0.5, 0.9]),
            }
        }

        let gp_d: SonaGroupParamsStereo4<V> = mk_group_params::<V>(0.08);
        let gp_s: SonaGroupParamsStereo4<VS> = mk_group_params::<VS>(0.08);
        let mut st_d = SonaBlockStateStereo8::<S8>::zero();
        let mut st_s = SonaBlockStateStereo8::<S8S>::zero();
        let mut st_b = SonaBlockStateStereo8::<S8>::zero();
        let bp_d = SonaBlockParamsStereo4 {
            svf: gp_d.svf,
            tone: gp_d.tone,
            lru: gp_d.lru,
            adaa_drive: gp_d.adaa_drive,
        };

        for n in 0..32 {
            let ul = ((n as f32) * 0.23).sin() * 1.4;
            let ur = ((n as f32) * 0.17).cos() * 1.2;
            let od = step_sona_group_stereo8(S8::splat_lr(ul, ur), &mut st_d, &gp_d);
            let os = step_sona_group_stereo8(S8S::splat_lr(ul, ur), &mut st_s, &gp_s);
            let ob = step_sona_block_stereo8(S8::splat_lr(ul, ur), &mut st_b, &bp_d);
            assert!(KernelVec4::to_array(ob.adaa_odd.into_lr().0)[0].is_finite());

            let (dl, dr) = od.adaa_odd.into_lr();
            let (sl, sr) = os.adaa_odd.into_lr();
            for lane in 0..4 {
                assert!(
                    (KernelVec4::to_array(dl)[lane] - KernelVec4::to_array(sl)[lane]).abs()
                        < 2.0e-5
                );
                assert!(
                    (KernelVec4::to_array(dr)[lane] - KernelVec4::to_array(sr)[lane]).abs()
                        < 2.0e-5
                );
            }
        }
    }
}
