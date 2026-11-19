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
}
