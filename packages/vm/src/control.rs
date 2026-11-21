//! Parametric `3 kHz` `ControlBus` (`#[forbid(unsafe_code)]`, `--no-default-features` `!alloc`-capable),
//! conditional clamping, `CP1` projections, `9/18`-word exact-integer accumulators, `TSYN` tempo sync,
//! and division-free micro-ramped `32`-slot modulation dispatch (`ref[dsp.ctrl.op-m.ballistic-lfo]`,
//! `ref[dsp.ctrl.dispatch.modulation-laws]`, `ref[dsp.tempo.subdivisions]`).

pub mod bus;
pub mod cp1;
pub mod descriptor;
pub mod exact_int;
pub mod tempo;

pub use bus::dispatch::{
    ANTI_ZIPPER_SNAP_EPS, DEFAULT_ANTI_ZIPPER_POLE, STAGE_TARGET_SLOTS, control_stride_for_rate,
    control_tick_ratio,
};
pub use bus::{
    ControlBus, ControlBusBacking, DefaultControlBus, step_control_frame_default,
    step_control_tick_3khz_default,
};
pub use cp1::{
    Cp1ContributorSlot, count_distinct_contributor_controls, decode_cp1_contributors,
    eval_uint_base_4x8, eval_uint_ultra_delta_3x4x8,
};
pub use descriptor::{ConditionalClampRule, ControlClampDescriptor};
pub use exact_int::{
    EXACT_SQUARE_WORDS, EXACT_SUM_WORDS, ExactIntFactorAccumulator, ExactIntMainAccumulator,
    MAX_EXACT_CONTRIBUTORS,
};
pub use tempo::{
    CANONICAL_SUBDIVISION_RATIOS, TempoSyncEvalResult, TempoSyncMode, apply_tsyn_table,
    eval_3node_calibration_f64, eval_tsyn_row_physical, eval_tsyn_target_value,
    host_frame_q64_increment, invert_tsyn_lobatto_anchors, subdivision_phase_q32,
};

pub use crate::math::trust_clip_outward_4d;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_stride_values_match_canonical_rates() {
        assert_eq!(control_stride_for_rate(44_100.0), 15);
        assert_eq!(control_stride_for_rate(48_000.0), 16);
        assert_eq!(control_stride_for_rate(96_000.0), 32);
        assert_eq!(control_stride_for_rate(384_000.0), 128);
        assert_eq!(STAGE_TARGET_SLOTS, 32);
    }
}
