//! `SONA` `v3.0` Top-of-Frame Crossbar, 4-Block Matryoshka Core, Stage III & Stage IV Engine Core
//! (`stages/sona_core.rs`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub mod frontend;
pub mod model;
pub mod slab;
pub mod tick16;

use alloc::alloc::Global;
use alloc::boxed::Box;
use core::alloc::Allocator;

pub use frontend::{
    SonaOversamplerParams, SonaStereoOversampler2x, V3OversamplerParams, V3StereoOversampler2x,
    ZeroLagOversampler2x,
};
pub use model::{
    FIR_RING_CAP, FIR_RING_MASK, MAX_FIR_SEGMENTS, MAX_GROUPS_PER_BLOCK, MAX_KAUTZ_BANKS,
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_STAGE_LATENTS, MAX_TANK_LINES,
    MAX_TANK_QUADS, MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, NUM_SONA_MACRO_KNOBS,
    SONA_BLOCK_GROUP_SLOTS, SONA_FIR_MAX_TAPS, SONA_GROUP_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES,
    SonaColdTierModel, SonaControlBusParams, SonaCrossbarConfig, SonaGroupCold,
    SonaMatryoshkaModel, SonaTierReadout, SonaTileCold, V3ColdTierModel, V3ControlBusParams,
    V3MatryoshkaModel, V3TierReadout, flat_sona_lane_index, tier_block_groups_array,
    tier_fir_segments, tier_kautz_banks, tier_sona_groups, tier_spec, tier_tank_quads,
    tier_tile_count,
};
pub use slab::{
    SonaControlBusState, SonaGroupHot, SonaGroupReadoutHot, SonaHotSlab, SonaInterBlockTileHot,
};

use crate::math::DefaultMathBackend;
use crate::stages::StageBackend;

/// Owned heap/arena-allocated [`SonaHotSlab`] parameterized by [`core::alloc::Allocator`].
#[derive(Clone, Debug)]
pub struct SonaInstance<B: StageBackend = DefaultMathBackend, A: Allocator = Global> {
    hot: Box<SonaHotSlab<B>, A>,
}

/// Compatibility alias for [`SonaInstance`].
pub type V3CoreInstance<B = DefaultMathBackend, A = Global> = SonaInstance<B, A>;

impl<B: StageBackend> SonaInstance<B, Global> {
    /// Constructs a [`SonaInstance`] in the global allocator from `model`.
    #[must_use]
    pub fn from_cold_model(model: &SonaColdTierModel) -> Self {
        Self::from_cold_model_in(model, Global)
    }
}

impl<B: StageBackend, A: Allocator> SonaInstance<B, A> {
    /// Constructs a [`SonaInstance`] inside a custom [`core::alloc::Allocator`] (`Box::new_in`).
    #[must_use]
    pub fn from_cold_model_in(model: &SonaColdTierModel, alloc: A) -> Self {
        Self {
            hot: Box::new_in(SonaHotSlab::from_cold_model(model), alloc),
        }
    }

    /// Returns an immutable reference to the underlying cache-coherent [`SonaHotSlab`].
    #[inline(always)]
    #[must_use]
    pub fn hot_core(&self) -> &SonaHotSlab<B> {
        &self.hot
    }

    /// Returns a mutable reference to the underlying cache-coherent [`SonaHotSlab`].
    #[inline(always)]
    #[must_use]
    pub fn hot_core_mut(&mut self) -> &mut SonaHotSlab<B> {
        &mut self.hot
    }

    /// Switches active tier & readout weights in-place (`0` alloc, `0` reset).
    #[inline]
    pub fn switch_tier(&mut self, model: &SonaColdTierModel) {
        self.hot.switch_tier(model);
    }

    /// Reloads a new model in-place and resets runtime state (`0` heap alloc, `0` drop).
    #[inline]
    pub fn reload_in_place(&mut self, model: &SonaColdTierModel) {
        self.hot.reload_in_place(model);
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

/// Concrete `DefaultMathBackend` entrypoint emitted into `dnam_engine` `.rlib` for disassembly
/// audits (`0 FDIV`, `0 FSQRT`, `0 IDIV/UDIV`, `0 alloc`).
#[inline(never)]
pub fn sona_process_block_default(
    core: &mut SonaHotSlab<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    core.process_block_stereo(in_l, in_r, out_l, out_r);
}

/// Concrete infallible `DefaultMathBackend` entrypoint returning `Result<(), !>`.
#[inline(never)]
pub fn sona_process_block_infallible_default(
    inst: &mut SonaInstance<DefaultMathBackend>,
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
    use crate::container::ZohRescaledParams;
    use crate::math::ScalarBackend;
    use crate::types::{FtzDazGuard, Tier};

    #[test]
    fn test_sona_hot_slab_all_tiers_parity_subblock_and_zoh() {
        let _guard = FtzDazGuard::new();
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            for os2x in [false, true] {
                let mut cold = SonaColdTierModel::identity_for_tier(tier, os2x, false);
                cold.groups[0].adaa_drive = [1.2, 0.8, 0.4, 0.2];
                cold.groups[0].tone_lp = [0.6; 4];
                cold.groups[0].tone_dir = [0.4; 4];
                cold.readout.pre_cab_w_odd[0][0] = 0.35;
                cold.readout.pre_cab_w_odd[1][0] = -0.35;
                cold.tiles[0].w_odd_4x4[0][0] = 0.25;
                cold.crossbar.op_p_crossbar_gain = 0.3;
                cold.crossbar.op_p_to_group[0] = [0.2, 0.0, 0.0, 0.0];
                cold.crossbar.phasor_freq_hz = [2.5, 5.0, 7.5, 10.0];
                cold.crossbar.phasor_to_group[0] = [0.1, 0.0, 0.0, 0.0];

                let mut simd_inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&cold);
                let mut scal_inst = SonaInstance::<ScalarBackend>::from_cold_model(&cold);
                let mut blk_inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&cold);

                assert_eq!(simd_inst.step_stereo(0.0, 0.0).map(f32::to_bits), [0, 0]);
                simd_inst.reset_to_dc_equilibrium();

                let mut in_l = [0.0_f32; 32];
                let mut in_r = [0.0_f32; 32];
                for n in 0..32 {
                    in_l[n] = 0.5 * ((n as f32) * 0.19).sin();
                    in_r[n] = 0.4 * ((n as f32) * 0.13).cos();
                }
                let mut blk_l = [0.0_f32; 32];
                let mut blk_r = [0.0_f32; 32];
                blk_inst
                    .process_block_infallible(&in_l, &in_r, &mut blk_l, &mut blk_r)
                    .unwrap();

                for n in 0..32 {
                    let s_simd = simd_inst.step_stereo(in_l[n], in_r[n]);
                    let s_scal = scal_inst.step_stereo(in_l[n], in_r[n]);
                    assert!(s_simd[0].is_finite() && s_simd[1].is_finite());
                    assert!(
                        (s_simd[0] - s_scal[0]).abs() < 1.0e-5,
                        "Scalar/SIMD parity L diff tier={tier:?} os2x={os2x} n={n}"
                    );
                    assert!(
                        (s_simd[0] - blk_l[n]).abs() < 1.0e-6,
                        "step_stereo vs tick16 L diff tier={tier:?} os2x={os2x} n={n}"
                    );
                    assert!(
                        (s_simd[1] - blk_r[n]).abs() < 1.0e-6,
                        "step_stereo vs tick16 R diff tier={tier:?} os2x={os2x} n={n}"
                    );
                }

                // ZOH reconfigure across 44.1..192 kHz + clean equilibrium reset
                for sr in [44_100_u32, 96_000, 192_000] {
                    let zoh = ZohRescaledParams::new(48_000, sr);
                    simd_inst.hot_core_mut().reconfigure_with_zoh(&cold, &zoh);
                    let z_out = simd_inst.step_stereo(0.1, -0.1);
                    assert!(z_out[0].is_finite() && z_out[1].is_finite());
                }
                simd_inst.reset_to_dc_equilibrium();
                assert_eq!(simd_inst.step_stereo(0.0, 0.0).map(f32::to_bits), [0, 0]);
            }
        }
    }

    /// verify[dsp.head.mid-side-factorization]
    #[test]
    fn test_head_mid_side_factorization_mono_preserves_mid_slice() {
        let mut mat = SonaMatryoshkaModel::identity(false, false);
        mat.groups[0].in_x_weight[0] = 0.5;
        mat.groups[0].adaa_drive[0] = 1.0;
        for ro in &mut mat.readouts {
            ro.direct_w_odd[0][0] = 0.8;
            ro.direct_w_odd[1][0] = -0.2;
        }
        let stereo_model = mat.extract_tier(Tier::Lite);
        let mut mono_model = stereo_model.clone();
        mono_model.mono_out = true;

        let mut s_eng = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&stereo_model);
        let mut m_eng = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&mono_model);
        for i in 0..32 {
            let x = 0.35 * ((i as f32) * 0.19).sin();
            let [yl, yr] = s_eng.step_stereo(x, x);
            let [ml, mr] = m_eng.step_stereo(x, x);
            let expected_mid = 0.5 * (yl + yr);
            assert_eq!(ml.to_bits(), mr.to_bits());
            assert!(
                (ml - expected_mid).abs() < 1.0e-5,
                "i={i} ml={ml} expected_mid={expected_mid}"
            );
        }
    }
}
