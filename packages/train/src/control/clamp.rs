//! Phase-1 (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`) and Phase-3
//! (`[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`, `[NUMERIC-ENVELOPE]`) control-space auditing
//! (`ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase3.clamp-dynamic]`,
//! `ref[train.audit.phase3.clamp-head-gain]`, `docs/specs/training/control_space_audit.md`).

pub mod dynamic;
mod phase1;
mod phase3;

use dnam_vm::bin::ControlClampDescriptorWire;

pub use dynamic::{
    DYNAMIC_ALLPASS_G_MAX, DYNAMIC_DAMP_POLE_MAX, DYNAMIC_G_LOOP_MAX, DYNAMIC_KAUTZ_R_MAX,
    DynamicCornerAuditReport, DynamicCornerViolation, ExtremeCornerDcReport, ZERO_DC_CORNER_FRAMES,
    ZERO_DC_CORNER_MAX_ABS, check_corner_dynamic_stability, compute_dynamic_excursion_bounds_32,
    eval_static_control_bus_targets, verify_extreme_corners_zero_dc_256,
};
pub use phase1::audit_phase1_controls;
pub use phase3::{audit_phase3_dynamic_and_head_gain, evaluate_readout_spectral_magnitudes};

/// Linear amplitude ceiling corresponding to `+6.0 dB` (`ref[train.audit.phase3.clamp-head-gain]`).
pub const MAX_GAIN_6DB_LIN: f64 = 1.995_262_3;
/// Idempotent target linear gain (`~33 ULPs` below `+6.0 dB`) yielding 0-ULP re-audit invariance.
pub const MAX_GAIN_IDEMPOTENT: f64 = 1.995_258_3;

const _: () = assert!(MAX_GAIN_IDEMPOTENT < MAX_GAIN_6DB_LIN);

/// Kind and observed coverage statistics for one physical control parameter (`label-free`).
#[derive(Clone, Debug, PartialEq)]
pub enum ControlKindSpec {
    /// Continuous range knob (`ctrl_kind = 0`).
    Range {
        /// Declared minimum physical bound (used for `scale_norm` / `offset_norm`).
        declared_min: f64,
        /// Declared maximum physical bound (used for `scale_norm` / `offset_norm`).
        declared_max: f64,
        /// Neutral / default physical position.
        default_val: f64,
        /// Minimum observed physical value in dataset.
        obs_min: f64,
        /// Maximum observed physical value in dataset.
        obs_max: f64,
        /// Distinct observed values (`<= 1` locks `K_p = 0`).
        unique_values: usize,
        /// Three physical Lobatto calibration anchors `[v_min, v_mid, v_max]`.
        lobatto_anchors: [f32; 3],
    },
    /// Discrete switch (`ctrl_kind = 1`).
    Switch {
        /// Total declared switch options (`1..=16`).
        num_options: u8,
        /// Neutral default option index.
        default_opt: u8,
        /// Bitmask of declared switch options.
        declared_mask: u16,
        /// Bitmask of switch options observed in captured stems.
        observed_mask: u16,
    },
}

/// Physical control descriptor specification supplied to Phase-1 audit.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlSpec {
    /// Continuous range or discrete switch metadata.
    pub kind: ControlKindSpec,
}

/// Stage-level stability corner constraint mapping normalized control values + switch state
/// to `[tau_samples, tank_g_lin, loop_gain, paralru_r]`.
#[derive(Clone, Debug, PartialEq)]
pub struct StageStabilityConstraint {
    /// Target `Range` control index potentially restricted at hazardous switch/knob corners.
    pub delay_ctrl_idx: Option<usize>,
    /// Delay sensitivity (`d(tau) / d(norm_val)`) per control at neutral base `base_tau`.
    pub delay_slopes: Vec<f64>,
    /// Base `Op-T` causal delay in samples at normalized center.
    pub base_tau: f64,
    /// Tank `g_lin` base and normalized control sensitivities.
    pub base_g_lin: f64,
    /// Sensitivities of `tank_g_lin` per control.
    pub g_lin_slopes: Vec<f64>,
    /// Conditioning switch control index `(ctrl_idx, hazard_opt, extra_loop_gain)` tuples.
    pub switch_hazards: Vec<(usize, u8, f64)>,
    /// Feedback `loop_gain` base and normalized control sensitivities.
    pub base_loop_gain: f64,
    /// Sensitivities of `loop_gain` per control.
    pub loop_gain_slopes: Vec<f64>,
    /// Conditioning target `loop_gain` control index for conditional clamp rule emission.
    pub feedback_ctrl_idx: Option<usize>,
    /// Whether `Op-T` runs inside a top-of-frame crossbar (`>= 5.0`) or standalone (`>= 4.0`).
    pub crossbar_coupled: bool,
}

impl Default for StageStabilityConstraint {
    fn default() -> Self {
        Self {
            delay_ctrl_idx: None,
            delay_slopes: Vec::new(),
            base_tau: 16.0,
            base_g_lin: 0.5,
            g_lin_slopes: Vec::new(),
            switch_hazards: Vec::new(),
            base_loop_gain: 0.25,
            loop_gain_slopes: Vec::new(),
            feedback_ctrl_idx: None,
            crossbar_coupled: true,
        }
    }
}

/// Complete Phase-1 audit result containing wire descriptors, lock mask, and warnings.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase1ClampOutput {
    /// Emitted `64 B` wire control descriptors (`<= 255`).
    pub wire_descriptors: Vec<ControlClampDescriptorWire>,
    /// Controls locked static (`K_p = 0`) because `unique_values <= 1`.
    pub locked_controls: Vec<bool>,
    /// Audited `[min, max]` base ranges after `[CLAMP-RANGE]` & `[CLAMP-STABILITY]`.
    pub safe_ranges: Vec<(f64, f64)>,
    /// Total control corners audited in `[CLAMP-STABILITY]`.
    pub audited_corners: usize,
    /// Non-blocking warning banners logged for user/project inspection.
    pub warnings: Vec<String>,
}

/// Phase-3 dynamic excursion + direct-register `+6 dB` readout head-gain inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase3AuditInput {
    /// Minimum causal `Op-T` base delay in samples.
    pub min_delay_samples: f64,
    /// Peak LFO + envelope delay modulation depth in samples.
    pub max_delay_excursion: f64,
    /// Peak `ParaLRU` / Kautz pole radius + dynamic modulation sum.
    pub max_paralru_r_excursion: f64,
    /// Peak `Op-T` tank `g_lin` + modulation sum.
    pub max_tank_g_excursion: f64,
    /// Whether `Op-T` runs inside the crossbar (`>= 5.0` samples) vs standalone (`>= 4.0`).
    pub crossbar_coupled: bool,
    /// Active tier's `LinearReadout` `48 kHz` leak pole (`<= 0x3F7F_FFAC`).
    pub ro_pole_48k: f32,
    /// Left channel direct readout coefficients `[pre_u, pre_hp, dir_u, dir_hp]` + `W_ro` DC-free row gains.
    pub readout_gains_l: Vec<f64>,
    /// Right channel direct readout coefficients.
    pub readout_gains_r: Vec<f64>,
    /// Linear frequency-response gain samples across `[20 Hz, nyquist]` for Left & Right before clamping.
    pub freq_mag_l: Vec<f64>,
    /// Linear frequency-response gain samples for Right.
    pub freq_mag_r: Vec<f64>,
    /// Optional burst peak magnitude for `[NUMERIC-ENVELOPE]` verification.
    pub burst_peak_abs: f64,
}

/// Phase-3 `[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`, and `[NUMERIC-ENVELOPE]` audit report.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase3AuditReport {
    /// Clamped safe minimum `Op-T` delay after absorbing LFO/envelope excursion.
    pub tightened_min_delay: f64,
    /// Clamped LFO/envelope excursion depth.
    pub clamped_delay_excursion: f64,
    /// Clamped `ro_pole_48k` in `[0.0, 0x3F7F_FFAC]`.
    pub ro_pole_48k: f32,
    /// Applied linear head-gain scale (`1.0` when already `<= MAX_GAIN_6DB_LIN` for 0-ULP idempotence).
    pub head_gain_scale: f64,
    /// Peak observed linear spectral gain before scaling.
    pub peak_gain_before: f64,
    /// Peak linear spectral gain after scaling (`<= MAX_GAIN_6DB_LIN`).
    pub peak_gain_after: f64,
    /// Non-blocking audit banners.
    pub warnings: Vec<String>,
}
