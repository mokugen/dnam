//! Compatibility [`V3CoreInstance`] wrapper & `DefaultMathBackend` entrypoints delegating to
//! [`crate::stages::sona_core`] (`stages/hot_core/instance.rs`).

#![forbid(unsafe_code)]

use crate::math::DefaultMathBackend;
pub use crate::stages::sona_core::{
    SonaInstance, V3CoreInstance, sona_process_block_default, sona_process_block_infallible_default,
};

/// Compatibility wrapper delegating to [`sona_process_block_default`].
#[inline(never)]
pub fn v3_process_block_default(
    core: &mut super::V3HotCore<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    sona_process_block_default(core, in_l, in_r, out_l, out_r);
}

/// Compatibility wrapper delegating to [`sona_process_block_infallible_default`].
#[inline(never)]
pub fn v3_process_block_infallible_default(
    inst: &mut V3CoreInstance<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), !> {
    sona_process_block_infallible_default(inst, in_l, in_r, out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    #[cfg(target_arch = "aarch64")]
    use crate::math::{ArmSmeBackend, ArmSve2Backend};
    use crate::math::{MathBackend, ScalarBackend};
    use crate::stages::StageBackend;
    use crate::stages::hot_core::V3HotCore;
    use crate::stages::sona_core::V3MatryoshkaModel;
    use crate::types::Tier;
    use alloc::vec;
    use alloc::vec::Vec;

    fn exercise_sona_backend<B: StageBackend>(os2x: bool, mono_out: bool) {
        let mat = V3MatryoshkaModel::identity(os2x, mono_out);
        let four = mat.to_four_tiers();
        assert_eq!(mat.extract_tier(Tier::Full).tier, Tier::Full);
        assert_eq!(four[3].tier, Tier::Ultra);

        let mut inst = V3CoreInstance::<B>::from_cold_model(&four[3]);
        assert_eq!(inst.hot_core().tier(), Tier::Ultra);
        let bytes_ultra = V3HotCore::<B>::active_hot_bytes(Tier::Ultra, os2x);
        inst.switch_tier(&four[0]);
        assert!(V3HotCore::<B>::active_hot_bytes(Tier::Lite, os2x) < bytes_ultra);
        inst.switch_tier(&four[1]);
        inst.switch_tier(&four[2]);
        inst.switch_tier(&four[3]);
        inst.reset_to_dc_equilibrium();

        let n = 48usize;
        let clean_l: Vec<f32> = (0..n).map(|i| 0.4 * ((i as f32) * 0.21).sin()).collect();
        let clean_r: Vec<f32> = (0..n).map(|i| 0.4 * ((i as f32) * 0.17).cos()).collect();
        let mut dirty_l = clean_l.clone();
        let mut dirty_r = clean_r.clone();
        dirty_l[4] = f32::NAN;
        dirty_r[4] = f32::INFINITY;
        dirty_l[5] = f32::NEG_INFINITY;
        dirty_r[5] = -100.0;
        let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        inst.process_block_stereo(&clean_l, &clean_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().chain(out_r.iter()).all(|v| v.is_finite()));

        let single = inst.step_stereo(0.25, -0.25);
        assert!(single[0].is_finite() && single[1].is_finite());

        let (mut out_il, mut out_ir) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        let Ok(()) = inst.process_block_infallible(&dirty_l, &dirty_r, &mut out_il, &mut out_ir);
        assert!(out_il.iter().chain(out_ir.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_sona_core_instance_backends_tiers_and_default_entrypoints() {
        let _ftz = FtzDazGuard::new();
        for &(os2x, mono_out) in &[(false, false), (true, false), (false, true), (true, true)] {
            exercise_sona_backend::<ScalarBackend>(os2x, mono_out);
            exercise_sona_backend::<DefaultMathBackend>(os2x, mono_out);
            #[cfg(target_arch = "aarch64")]
            {
                exercise_sona_backend::<ArmSve2Backend>(os2x, mono_out);
                exercise_sona_backend::<ArmSmeBackend>(os2x, mono_out);
            }
        }

        let ultra_model =
            crate::stages::sona_core::V3ColdTierModel::identity_for_tier(Tier::Ultra, false, false);
        let mut core_direct = V3HotCore::<DefaultMathBackend>::from_cold_model(&ultra_model);
        let mut inst_def = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&ultra_model);
        let in_l = [0.2_f32; 24];
        let in_r = [-0.2_f32; 24];
        let (mut out_l, mut out_r) = ([0.0_f32; 24], [0.0_f32; 24]);
        sona_process_block_default(&mut core_direct, &in_l, &in_r, &mut out_l, &mut out_r);
        let Ok(()) = sona_process_block_infallible_default(
            &mut inst_def,
            &in_l,
            &in_r,
            &mut out_l,
            &mut out_r,
        );
        assert!(out_l.iter().all(|v| v.is_finite()));
        v3_process_block_default(&mut core_direct, &in_l, &in_r, &mut out_l, &mut out_r);
        let Ok(()) = v3_process_block_infallible_default(
            &mut inst_def,
            &in_l,
            &in_r,
            &mut out_l,
            &mut out_r,
        );
        let _ = core::any::type_name::<crate::math::DefaultMathBackend>();
        let _ = <DefaultMathBackend as MathBackend>::NAME;
    }
}
