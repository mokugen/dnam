//! Pure Rust `#![no_std]` + `!alloc`-capable zero-JSON `SONA v3.0` Virtual Machine (`dnam-vm`).
//!
//! Depends strictly on [`dnam_kernels`] and [`zerocopy`], isolating Layer 0 `.dnam-bin` (`"DNBN"` /
//! `"DBRM"`) loading, zero-mutation `REQ-5.6` tier truncation, multi-ISA `32-bit` SIMD execution
//! ([`math::MathBackend`]), and parametric `3 kHz` [`control::ControlBus`] + `40 B` `"TSYN"`
//! tempo sync (`ref[sdk.rust.safe-rust-and-boundaries]`, `ref[container.dnam.vm-binary]`).

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![deny(unsafe_code)]
#![warn(unreachable_pub)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[forbid(unsafe_code)]
pub mod bin;
#[forbid(unsafe_code)]
pub mod control;
pub mod math;
pub mod types;

pub use bin::{
    Arena0ControlBusWire, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3ControlBusWire, Arena3UltraDeltaWire, CABINET_EARLY_FIR_BASES, CABINET_KAUTZ_BASES,
    CabinetFir16Wire, CabinetFir32Wire, CabinetFir48Wire, CextDispatchWire,
    ConditionalClampRuleWire, ControlClampDescriptorWire, Cp1InlineDirWire, Cp1StreamDirEntryWire,
    Cp1StreamHeaderWire, CrossbarWire, DnamBinBuildInput, DnamBinHeaderWire,
    DnamBinRuntimeMetaWire, DnamBinView, KautzSoaBlockWire, OpNPreBaseWire, SonaCellDescWire,
    SonaCrossbarBlockWire, SonaGroupWire, SonaInterBlockTilePairWire, TANK_DELAY_BASES,
    TankQuadWire, TempoSyncEntryWire, VARPRO_READOUT_BASES, VarProReadoutFullWire,
    VarProReadoutLiteWire, VarProReadoutProWire, VarProReadoutUltraWire,
    bind_dnam_bin_prevalidated, computed_dnam_bin_byte_len, encode_dnam_bin_into,
    strip_dnam_bin_to_tier,
};
#[cfg(feature = "alloc")]
pub use bin::{DnamBinBuildOwnedError, encode_dnam_bin};
pub use control::{
    ANTI_ZIPPER_SNAP_EPS, CANONICAL_SUBDIVISION_RATIOS, ConditionalClampRule, ControlBus,
    ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot, DEFAULT_ANTI_ZIPPER_POLE,
    DefaultControlBus, EXACT_SQUARE_WORDS, EXACT_SUM_WORDS, ExactIntFactorAccumulator,
    MAX_EXACT_CONTRIBUTORS, STAGE_TARGET_SLOTS, TempoSyncEvalResult, TempoSyncMode,
    apply_tsyn_table, control_stride_for_rate, control_tick_ratio,
    count_distinct_contributor_controls, decode_cp1_contributors, eval_3node_calibration_f64,
    eval_tsyn_row_physical, eval_tsyn_target_value, eval_uint_base_4x8,
    eval_uint_ultra_delta_3x4x8, invert_tsyn_lobatto_anchors, step_control_frame_default,
    step_control_tick_3khz_default, trust_clip_outward_4d,
};
pub use math::{
    BackendKind, CpuSimdFeatures, CpuSimdLevel, Crc32cBackend, DefaultMathBackend, MathBackend,
    ScalarBackend, ScalarVec4, SimdVec4, crc32c, crc32c_slice, crc32c_software_slice,
    detect_cpu_simd, detect_cpu_simd_features, warm_simd_feature_caches,
};
pub use types::{CacheLine64, ColdBuf, ColdSlice, FtzDazGuard, SonaTier, Tier, VmError};
