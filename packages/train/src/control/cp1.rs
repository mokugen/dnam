//! Lobatto `CP1` ANOVA + Rank-4/8 Factorization Machine fitting, `32`-slot `W_ctrl` +
//! `"CEXT"` + `"TSYN"` parameter solver, and `zerocopy` inline vs `"CP1\0"` overflow packing
//! (`ref[dsp.ctrl.accum.cp1-scaling]`, `ref[dsp.ctrl.dispatch.modulation-laws]`).

mod fit;
mod subspace;
mod wire;

use dnam_vm::bin::{
    Arena0ControlBusWire, Arena3ControlBusWire, CextDispatchWire, TempoSyncEntryWire,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;

pub use fit::{fit_and_pack_cp1, fit_and_pack_cp1_audited};
pub use wire::{
    TSYN_SUBDIV_QUARTER, build_tsyn_entry_wire, build_tsyn_entry_wire_with_cond,
    pack_cp1_slots_into_wire,
};

/// Maximum delay variance threshold (`0.2` samples²) below which delay/LFO modulation gains are gated.
pub const CP1_DELAY_VARIANCE_GATE_SAMPLES2: f64 = 0.2;

/// Observed training operating point with fitted 32-slot stage targets (`[base, optional ultra_delta]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Cp1ControlSample {
    /// Physical control vector (`len == specs.len()`).
    pub phys_controls: Vec<f32>,
    /// Active switch option per switch control (`0` for range controls).
    pub switch_options: Vec<u8>,
    /// Fitted target `stage_targets[0..32]` at this operating point for base `Lite..Pro`.
    pub stage_targets_base: [f64; STAGE_TARGET_SLOTS],
    /// Optional fitted Ultra delta `stage_targets_ultra - stage_targets_base`.
    pub stage_targets_ultra_delta: Option<[f64; STAGE_TARGET_SLOTS]>,
}

/// Configuration for `CP1` ANOVA + Rank-4/8 FM + 32-slot dispatch regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cp1FitConfig {
    /// L2 ridge regularization weight.
    pub l2_reg: f64,
    /// Variance threshold below which canonical `Op-T` delay slot `12` (`tank_delay_mod`) is gated to zero.
    pub delay_variance_gate_samples2: f64,
    /// Emit Ultra deltas in `Cp1ContributorSlot` and `Arena3ControlBusWire`.
    pub fit_ultra: bool,
    /// 4-D trust-region radius `rho_max` stored into `trust_radius_mh`.
    pub trust_radius_mh: f32,
}

impl Default for Cp1FitConfig {
    fn default() -> Self {
        Self {
            l2_reg: 1.0e-3,
            delay_variance_gate_samples2: CP1_DELAY_VARIANCE_GATE_SAMPLES2,
            fit_ultra: true,
            trust_radius_mh: 2.5,
        }
    }
}

/// Wire output bundle produced by `fit_and_pack_cp1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cp1WireBundle {
    /// Populated inline `Arena0ControlBusWire`.
    pub arena0_cb: Arena0ControlBusWire,
    /// Populated `Arena3ControlBusWire` (Ultra `U_int`, `W_ctrl_ultra`, `cp1_ultra`).
    pub arena3_cb: Arena3ControlBusWire,
    /// Optional `"CEXT"` dispatch matrix for extended `stage_targets[16..31]`.
    pub cext: Option<CextDispatchWire>,
    /// Pure-binary `"TSYN"` rows matching rate/period range controls.
    pub tsyn: Vec<TempoSyncEntryWire>,
    /// Non-empty `"CP1\0"` overflow stream when inline directory/payload limits are exceeded.
    pub cp1_overflow: Vec<u8>,
    /// Decoded contributor count (`inline` or `"CP1\0"`).
    pub contributor_count: usize,
    /// Distinct active control count contributing to `CP1` (`>= 2` unlocks `U_int`).
    pub distinct_controls: usize,
}
