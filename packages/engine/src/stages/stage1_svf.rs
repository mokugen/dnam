//! Stage I Envelope-Warped 2-Section Cytomic TPT/ZDF State-Variable Filter.
//!
//! Executes unconditionally at 1x host rate with zero `FDIV`, zero `FSQRT`, zero `libm`, zero
//! integer division, and zero allocations (`ref[dsp.stage1.pre-bus]`). Submodule [`coeffs`] holds
//! the parameter block, hoisted SIMD constants, ZOH rescaling, and division-free helpers;
//! [`audio`] implements `V3Stage1SvfState` and its real-time stereo/sub-block loops.

pub mod audio;
pub mod coeffs;

pub(crate) use audio::SvfSubblockOut;
pub use audio::V3Stage1SvfState;
pub(crate) use coeffs::SvfHoistedConsts;
pub use coeffs::{
    SVF_G_MAX, SVF_W_MAX, V3Stage1SvfParams, algebraic_softclip_vec4, exp2_signed_oct_f32,
    recip_nr2_pos_f32, recip_nr2_pos_vec4, recip_nr3_pos_f32, recip_nr3_pos_vec4, rsqrt_nr2_f32,
    rsqrt_nr2_vec4, rsqrt_nr3_f32, rsqrt_nr3_vec4, tan_prewarp_f32, zero_dc_asym_sat_vec4,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ZohRescaledParams;

    #[test]
    fn test_stage1_svf_zoh_integration() {
        let mut p = V3Stage1SvfParams {
            base_g: 0.25,
            g_min: 0.05,
            g_max: 0.68,
            det_lp_pole: 0.8,
            ..V3Stage1SvfParams::identity_bypass()
        };
        let zoh_96k = ZohRescaledParams::compute(48_000, 96_000, 0.9993, 0.995, 64.0, 128).unwrap();
        p.rescale_with_zoh(&zoh_96k);
        assert!(p.det_lp_pole > 0.8);
        let mut st = V3Stage1SvfState::default();
        let (u_s2, dir) = st.step_stereo(&p, 0.5, -0.5);
        assert!((u_s2[0] - 0.5).abs() < 1e-6 && dir[1] == 0.0);
    }
}
