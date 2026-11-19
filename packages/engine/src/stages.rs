//! TORA Stages Pipeline Core: Stage 0 8-Port Router, Physical `v3` Stages I–IV & Direct Readout.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` and
//! `docs/specs/architecture/sona_architecture.md`.
//!
//! Direct per-sample calls require a caller-held [`crate::FtzDazGuard`] for the
//! enclosing batch; engine block methods provide the guard automatically.

pub mod head;
pub mod hot_core;
pub mod router;
pub mod stage1_pre;
pub mod stage1_svf;
pub mod stage2_gru;
pub mod stage3_cab;
pub mod stage3_cabinet;
pub mod stage4_tank;
pub mod topology;
pub mod weights;

pub use topology::{
    BALLISTIC_ENVELOPE_BANK_TAUS_SEC, GAIN_BOUNDED_FIRST_TAP_64_BITS, ParallelAlignmentMode,
    STAGE4_MAX_G_LIN_BITS, STAGE4_MAX_V_CEIL_BITS, TptLr4Crossover, WdwDryTapMode,
    eval_bounce_6point_lagrange_tap, eval_live_first_tap_difference,
    eval_min_latency_4tap_alignment, eval_stage_modulation_laws, eval_stage4_blended_feedback_knee,
    eval_stage4_loop_saturator, eval_stereo_linked_aux_detector, eval_wdw_routing_outputs,
    pack_stage3_modal_block_features, prepare_ballistic_band_alphas,
    prepare_bounce_6point_lagrange_tap, prepare_kaiser_sinc_alignment_weights,
    prepare_live_first_tap_difference_coeffs, stage3_feature_bus_slot_ranges,
    validate_tensor_directory_slice,
};

pub use head::{HEAD_RAM_BYTES, MidSideKroneckerHead};
pub use hot_core::{
    V3CoreInstance, V3HotCore, v3_process_block_default, v3_process_block_infallible_default,
};
pub use router::{BezoutCleanCrossover, ROUTER_STATE_RAM_BYTES, ReturnAligner, Stage0Router};
pub use stage1_pre::{STAGE1_PRE_RAM_BYTES, Stage1PreDynamics};
pub use stage2_gru::Stage2Backend;
pub use stage3_cab::{V3ColdTierModel, V3MatryoshkaModel, V3TierReadout};
pub use stage3_cabinet::{
    STAGE3_CABINET_RAM_BYTES, Stage3Backend, Stage3CabinetModals, dot16_stereo_vec4,
};
pub use stage4_tank::{STAGE4_TANK_RAM_BYTES, Stage4SpatialTank, TankDcBlocker};
pub use weights::{
    ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, ArenaDirectoryView,
};

/// Composite SIMD backend trait for the TORA Stages 0–IV pipeline.
pub trait StageBackend:
    crate::math::MathBackend
    + crate::operators::op_m::OpMBackend
    + crate::operators::op_n::OpNBackend<Self::Vec4>
    + crate::operators::op_t::OpTBackend<Self::Vec4>
    + Stage2Backend<Self::Vec4>
    + Stage3Backend
{
}

impl<B> StageBackend for B where
    B: crate::math::MathBackend
        + crate::operators::op_m::OpMBackend
        + crate::operators::op_n::OpNBackend<B::Vec4>
        + crate::operators::op_t::OpTBackend<B::Vec4>
        + Stage2Backend<B::Vec4>
        + Stage3Backend
{
}

/// 3 kHz Control Plane state RAM footprint.
pub const CONTROL_PLANE_RAM_BYTES: usize = 256;

/// Active `Lite` state working-set footprint strictly within the 11.2 KB L1D hot budget.
pub const TOTAL_LITE_STATE_RAM_BYTES: usize =
    stage3_cab::active_hot_bytes_for_tier(crate::types::Tier::Lite, false);

const _: () = assert!(TOTAL_LITE_STATE_RAM_BYTES > 0 && TOTAL_LITE_STATE_RAM_BYTES <= 8_656);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::types::{FtzDazGuard, Tier};

    #[test]
    fn test_v3_hot_core_lite_budget_and_identity_silence() {
        let _guard = FtzDazGuard::new();
        let cold = V3ColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut core = V3HotCore::<DefaultMathBackend>::from_cold_model(&cold);
        let y0 = core.step_stereo(0.0, 0.0);
        assert_eq!(y0.map(f32::to_bits), [0, 0]);
        let y1 = core.step_stereo(0.25, -0.25);
        assert!(y1[0].is_finite() && y1[1].is_finite());
        core.reset_to_dc_equilibrium();
        let y2 = core.step_stereo(0.0, 0.0);
        assert_eq!(y2.map(f32::to_bits), [0, 0]);
    }
}
