//! `SONA` Graph Pipeline Core: 8-Port [`IoRouter`] & Causal Crossbar (`Op-P`, `4x Q0.32`),
//! `4`-Block Matryoshka State-Space Core (`SonaHotSlab`), Early `FIR` + `Kautz` Cabinet, & Spatial `Op-T` Tank.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_crossbar_topology.md` and
//! `docs/specs/architecture/sona_architecture.md`.
//!
//! Direct per-sample calls require a caller-held [`crate::FtzDazGuard`] for the
//! enclosing batch; engine block methods provide the guard automatically.

pub mod cabinet;
pub mod router;
pub mod sona_core;
pub mod tank;
pub mod topology;
pub mod weights;

pub use cabinet::{CabinetBackend, dot16_stereo_vec4};
pub use router::{BezoutCleanCrossover, IoRouter, ROUTER_STATE_RAM_BYTES, ReturnAligner};
pub use sona_core::{
    SonaColdTierModel, SonaHotSlab, SonaInstance, SonaMatryoshkaModel, SonaTierReadout,
    sona_process_block_default, sona_process_block_infallible_default,
};
pub use tank::{SpatialTankHot, SpatialTankParams, V3_TANK_CAP, V3_TANK_LINES};
pub use topology::{
    BALLISTIC_ENVELOPE_BANK_TAUS_SEC, GAIN_BOUNDED_FIRST_TAP_64_BITS, ParallelAlignmentMode,
    TANK_MAX_G_LIN_BITS, TANK_MAX_V_CEIL_BITS, TptLr4Crossover, WdwDryTapMode,
    cabinet_feature_bus_slot_ranges, eval_bounce_6point_lagrange_tap,
    eval_live_first_tap_difference, eval_min_latency_4tap_alignment, eval_stage_modulation_laws,
    eval_stereo_linked_aux_detector, eval_tank_blended_feedback_knee, eval_tank_loop_saturator,
    eval_wdw_routing_outputs, pack_cabinet_modal_block_features, prepare_ballistic_band_alphas,
    prepare_bounce_6point_lagrange_tap, prepare_kaiser_sinc_alignment_weights,
    prepare_live_first_tap_difference_coeffs, validate_tensor_directory_slice,
};
pub use weights::{
    ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, ArenaDirectoryView,
};

/// Composite SIMD backend trait for the `SONA` audio processing graph.
pub trait GraphBackend:
    crate::math::MathBackend
    + crate::operators::op_m::OpMBackend
    + crate::operators::op_n::OpNBackend<Self::Vec4>
    + crate::operators::op_t::OpTBackend<Self::Vec4>
    + CabinetBackend
{
}

impl<B> GraphBackend for B where
    B: crate::math::MathBackend
        + crate::operators::op_m::OpMBackend
        + crate::operators::op_n::OpNBackend<B::Vec4>
        + crate::operators::op_t::OpTBackend<B::Vec4>
        + CabinetBackend
{
}

/// 3 kHz Control Plane state RAM footprint.
pub const CONTROL_PLANE_RAM_BYTES: usize = 256;

/// Active `Lite` state working-set footprint strictly within the `4,608 B` (`4.5 KiB`) L1D hot budget.
pub const TOTAL_LITE_STATE_RAM_BYTES: usize =
    cabinet::active_hot_bytes_for_tier(crate::types::Tier::Lite, false);

const _: () = assert!(TOTAL_LITE_STATE_RAM_BYTES > 0 && TOTAL_LITE_STATE_RAM_BYTES <= 4_608);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::types::{FtzDazGuard, Tier};

    #[test]
    fn test_sona_hot_slab_lite_budget_and_identity_silence() {
        let _guard = FtzDazGuard::new();
        let cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut core = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        let y0 = core.step_stereo(0.0, 0.0);
        assert_eq!(y0.map(f32::to_bits), [0, 0]);
        let y1 = core.step_stereo(0.25, -0.25);
        assert!(y1[0].is_finite() && y1[1].is_finite());
        core.reset_to_dc_equilibrium();
        let y2 = core.step_stereo(0.0, 0.0);
        assert_eq!(y2.map(f32::to_bits), [0, 0]);
    }
}
