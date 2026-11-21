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
#[forbid(unsafe_code)]
pub mod graph;
pub mod math;
pub mod types;
#[forbid(unsafe_code)]
pub mod vm;

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
pub use graph::{
    Bounce4xOversampler, CABINET_FULL_DELTA_FIR_TAPS, CABINET_KAUTZ_ONSET_DELAY_48K,
    CABINET_LITE_FIR_TAPS, CABINET_MAX_CHUNK_FRAMES, CABINET_MAX_FIR_TAPS,
    CABINET_PRO_DELTA_FIR_TAPS, CABINET_SEG_TAPS, CABINET_ULTRA_DELTA_FIR_TAPS, CABINET_WORK_LEN,
    CabinetBackend, CabinetFirHistorySlices, CabinetFirWeightSlices, CabinetHighSrFirExt,
    CabinetHighSrMode, CrossbarFrame, DEFAULT_SIDECHAIN_LR4_HZ, EXT_FIR_STORAGE_F32_LEN,
    FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_OS2X, FLAG_SINGLE_INPUT,
    FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE, GraphBackend, HoistedCoreChunk, IoAlignmentMode,
    IoRouter, IoSidechainDetectorBank, KautzColdSlewState, KautzPrefixOverride, KautzStereoBank4,
    MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS, OFFLINE_BOUNCE_PDC_SAMPLES, ParallelAlignLine,
    PartitionedSchroederBank, QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig, QuadModSlots,
    QuadRingMut, QuadStepResult, RouterGateConfig, SONA_BLOCK_GROUP_SLOTS,
    SONA_GROUP_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES, SONA_ULTRA_GROUP_TRAVERSAL_ORDER,
    SonaColdTail, SonaFullHotDelta, SonaGroupHot, SonaGroupReadoutHot, SonaGroupReadoutRegs,
    SonaHotSlab, SonaInterBlockTileHot, SonaLiteHotPrefix, SonaOs2xHot, SonaOversamplerParams,
    SonaProHotDelta, SonaSlabBacking, SonaStereoOversampler2x, SonaUltraHotDelta,
    TANK_INLINE_QUAD0_CAP, TANK_MAX_EXT_CAP, TANK_MAX_QUADS, TANK_MIN_EXT_CAP,
    TIER_SWITCH_XFADE_SAMPLES, TankColdStorageSummary, TankExtRing, TankExtStorage,
    TankQuadHotCompact, TankTopOfFrameResult, TierCrossbarSnapshot, WdwDryTapMode, ZohRateContext,
    ballistic_alpha_f32, compute_schroeder_requirements, compute_tank_storage_requirements,
    dot16_stereo_simd, eval_stereo_linked_aux_detector, eval_wdw_routing_outputs,
    evaluate_matryoshka_groups, hoist_groups_and_readouts, hydrate_kautz_slice_and_zoh,
    hydrate_slab_from_dnam_bin, latch_immediate_at_init_or_reset, latch_quad_nonslew_targets,
    process_block_sona, process_cabinet_48k_chunk, process_tick16_const, required_ext_elements,
    rescale_paralru_lane, reset_tank, reset_to_dc_equilibrium, retime_sample_rate,
    retune_kautz_banks_3khz, round_sample_count_zoh, set_offline_bounce, set_op_t_write_enable,
    step_control_tick_3khz, step_frame_4port_sona, step_frame_4x_sona, step_frame_sona,
    step_tank_end_of_frame, step_tank_top_of_frame, switch_tier, tick_quad_3khz, tick_tank_3khz,
    voltage_ms_decode, voltage_ms_encode,
};
#[cfg(feature = "alloc")]
pub use graph::{SonaSlabOwnedBacking, hydrate_slab_owned};
pub use math::{
    BackendKind, CpuSimdFeatures, CpuSimdLevel, Crc32cBackend, DefaultMathBackend, MathBackend,
    ScalarBackend, ScalarVec4, SimdVec4, crc32c, crc32c_slice, crc32c_software_slice,
    detect_cpu_simd, detect_cpu_simd_features, warm_simd_feature_caches,
};
pub use types::{CacheLine64, ColdBuf, ColdSlice, FtzDazGuard, SonaTier, Tier, VmError};
#[cfg(feature = "alloc")]
pub use vm::{DynamicSonaVm, SonaInstance};
pub use vm::{
    SonaTelemetrySnapshot, SonaVm, SonaVmBacking, sona_vm_clear_saturation_flags_default,
    sona_vm_process_block_default, sona_vm_process_tick16_default,
    sona_vm_set_live_analog_drift_enabled_default, sona_vm_step_control_tick_default,
    sona_vm_step_frame_default,
};
