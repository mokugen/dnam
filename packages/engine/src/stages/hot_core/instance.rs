//! Owned heap/arena-allocated [`V3CoreInstance`] wrapper and concrete `DefaultMathBackend` entrypoints.

#![forbid(unsafe_code)]

use alloc::alloc::Global;
use alloc::boxed::Box;
use core::alloc::Allocator;

use crate::math::{DefaultMathBackend, MathBackend};

use super::V3HotCore;
use crate::stages::stage3_cab::V3ColdTierModel;

/// Owned heap/arena-allocated [`V3HotCore`] parameterized by [`core::alloc::Allocator`].
#[derive(Clone, Debug)]
pub struct V3CoreInstance<B: MathBackend = DefaultMathBackend, A: Allocator = Global> {
    hot: Box<V3HotCore<B>, A>,
}

impl<B: MathBackend> V3CoreInstance<B, Global> {
    /// Constructs a `V3CoreInstance` in the global allocator from `model`.
    #[must_use]
    pub fn from_cold_model(model: &V3ColdTierModel) -> Self {
        Self::from_cold_model_in(model, Global)
    }
}

impl<B: MathBackend, A: Allocator> V3CoreInstance<B, A> {
    /// Constructs a `V3CoreInstance` inside a custom [`core::alloc::Allocator`] (`Box::new_in`).
    #[must_use]
    pub fn from_cold_model_in(model: &V3ColdTierModel, alloc: A) -> Self {
        Self {
            hot: Box::new_in(V3HotCore::from_cold_model(model), alloc),
        }
    }

    /// Returns an immutable reference to the underlying cache-coherent [`V3HotCore`].
    #[inline(always)]
    #[must_use]
    pub fn hot_core(&self) -> &V3HotCore<B> {
        &self.hot
    }

    /// Returns a mutable reference to the underlying cache-coherent [`V3HotCore`].
    #[inline(always)]
    pub fn hot_core_mut(&mut self) -> &mut V3HotCore<B> {
        &mut self.hot
    }

    /// Switches active tier & readout weights in-place (`0` alloc, `0` reset).
    #[inline]
    pub fn switch_tier(&mut self, model: &V3ColdTierModel) {
        self.hot.switch_tier(model);
    }

    /// Resets all recursive DSP state to exact resting `+0.0`.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.hot.reset_to_dc_equilibrium();
    }

    /// Processes one sample pair (caller must hold [`crate::FtzDazGuard`] for batch loops).
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        self.hot.step_stereo(in_l, in_r)
    }

    /// Processes a contiguous stereo audio block with [`crate::FtzDazGuard`] enabled.
    #[inline(never)]
    pub fn process_block_stereo(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        self.hot.process_block_stereo(in_l, in_r, out_l, out_r);
    }

    /// Infallible stereo block execution (`Result<(), !>`).
    #[inline(never)]
    pub fn process_block_infallible(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), !> {
        self.hot.process_block_infallible(in_l, in_r, out_l, out_r)
    }
}

/// Concrete `DefaultMathBackend` entrypoints emitted directly into `dnam_engine` `.rlib` so
/// hardware/disassembly audits verify concrete NEON/SSE codegen (`0 FDIV`, `0 FSQRT`, `0 alloc`).
#[inline(never)]
pub fn v3_process_block_default(
    core: &mut V3HotCore<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    core.process_block_stereo(in_l, in_r, out_l, out_r);
}

/// Concrete infallible `DefaultMathBackend` entrypoint returning `Result<(), !>`.
#[inline(never)]
pub fn v3_process_block_infallible_default(
    inst: &mut V3CoreInstance<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), !> {
    inst.process_block_infallible(in_l, in_r, out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::math::ScalarBackend;
    #[cfg(target_arch = "aarch64")]
    use crate::math::{ArmSmeBackend, ArmSve2Backend};
    use crate::stages::stage3_cab::V3MatryoshkaModel;
    use crate::types::Tier;
    use alloc::vec;
    use alloc::vec::Vec;

    fn exercise_v3_backend<B: MathBackend>(os2x: bool, mono_out: bool) {
        let mat = V3MatryoshkaModel::identity(os2x, mono_out);
        let four = mat.to_four_tiers();
        let extracted_full = mat.extract_tier(Tier::Full);
        assert_eq!(extracted_full.tier, Tier::Full);
        assert_eq!(four[3].tier, Tier::Ultra);

        let mut inst = V3CoreInstance::<B>::from_cold_model(&four[3]);
        assert_eq!(inst.hot_core().tier, Tier::Ultra);
        let bytes_ultra = V3HotCore::<B>::active_hot_bytes(Tier::Ultra, os2x);
        inst.switch_tier(&four[0]);
        assert!(V3HotCore::<B>::active_hot_bytes(Tier::Lite, os2x) < bytes_ultra);
        inst.switch_tier(&four[1]);
        inst.switch_tier(&four[2]);
        inst.switch_tier(&four[3]);
        inst.hot_core_mut().reset_to_dc_equilibrium();
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
        let mut out_l = vec![0.0_f32; n];
        let mut out_r = vec![0.0_f32; n];

        inst.process_block_stereo(&clean_l, &clean_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|v| v.is_finite()) && out_r.iter().all(|v| v.is_finite()));

        let single = inst.step_stereo(0.25, -0.25);
        assert!(single[0].is_finite() && single[1].is_finite());

        let mut out_il = vec![0.0_f32; n];
        let mut out_ir = vec![0.0_f32; n];
        let Ok(()) = inst.process_block_infallible(&dirty_l, &dirty_r, &mut out_il, &mut out_ir);
        assert!(out_il.iter().all(|v| v.is_finite()) && out_ir.iter().all(|v| v.is_finite()));

        let _ = V3HotCore::<B>::active_hot_bytes(inst.hot_core().tier, os2x);
    }

    #[test]
    fn test_v3_core_instance_backends_tiers_and_default_entrypoints() {
        let _ftz = FtzDazGuard::new();

        for &(os2x, mono_out) in &[(false, false), (true, false), (false, true), (true, true)] {
            exercise_v3_backend::<ScalarBackend>(os2x, mono_out);
            exercise_v3_backend::<DefaultMathBackend>(os2x, mono_out);
            #[cfg(target_arch = "aarch64")]
            {
                exercise_v3_backend::<ArmSve2Backend>(os2x, mono_out);
                exercise_v3_backend::<ArmSmeBackend>(os2x, mono_out);
            }
        }

        let ultra_model = V3ColdTierModel::identity_for_tier(Tier::Ultra, false, false);
        let mut core_direct = V3HotCore::<DefaultMathBackend>::from_cold_model(&ultra_model);
        let mut inst_def = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&ultra_model);
        let frame = core_direct.step_stereo(0.25, -0.25);
        assert!(frame[0].is_finite() && frame[1].is_finite());

        let in_l = [0.2_f32; 24];
        let in_r = [-0.2_f32; 24];
        let mut out_l = [0.0_f32; 24];
        let mut out_r = [0.0_f32; 24];
        v3_process_block_default(&mut core_direct, &in_l, &in_r, &mut out_l, &mut out_r);
        let Ok(()) = v3_process_block_infallible_default(
            &mut inst_def,
            &in_l,
            &in_r,
            &mut out_l,
            &mut out_r,
        );
        assert!(out_l.iter().all(|v| v.is_finite()));
    }
}
