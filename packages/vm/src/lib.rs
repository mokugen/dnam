//! Pure Rust `#![no_std]` + `!alloc`-capable zero-JSON `SONA v4.0` Virtual Machine (`dnam-vm`).
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
    Arena3ControlBusWire, Arena3UltraDeltaWire, CONVOLVER_EARLY_FIR_BASES, CONVOLVER_KAUTZ_BASES,
    CONVOLVER_KAUTZ_BLOCKS_PER_TIER, CextDispatchWire, ConditionalClampRuleWire,
    ControlClampDescriptorWire, ConvolverFir16Wire, ConvolverFir32Wire, ConvolverFir48Wire,
    CoreQuadWire, Cp1InlineDirWire, Cp1StreamDirEntryWire, Cp1StreamHeaderWire, CrossbarWire,
    DnamBinBuildInput, DnamBinHeaderWire, DnamBinRuntimeMetaWire, DnamBinView, KautzSoaBlockWire,
    LINEAR_READOUT_BASES, LinearReadoutFullWire, LinearReadoutLiteWire, LinearReadoutProWire,
    LinearReadoutUltraWire, OpNPreBaseWire, SonaCellDescWire, SonaCrossbarBlockWire,
    SonaInterBlockTilePairWire, TRANSPORT_DELAY_BASES, TempoSyncEntryWire, TransportQuadWire,
    bind_dnam_bin_prevalidated, computed_dnam_bin_byte_len, encode_dnam_bin_into,
    strip_dnam_bin_to_tier,
};
#[cfg(feature = "alloc")]
pub use bin::{DnamBinBuildOwnedError, encode_dnam_bin};
pub use control::{
    ANTI_ZIPPER_SNAP_EPS, CANONICAL_SUBDIVISION_RATIOS, ClockTrackerState, ConditionalClampRule,
    ControlBus, ControlBusBacking, ControlClampDescriptor, ControlTsynBacking, Cp1ContributorSlot,
    DEFAULT_ANTI_ZIPPER_POLE, DefaultControlBus, EXACT_SQUARE_WORDS, EXACT_SUM_WORDS,
    ExactIntFactorAccumulator, MAX_EXACT_CONTRIBUTORS, OrderedClockPulseTracker,
    STAGE_TARGET_SLOTS, TempoSyncEvalResult, TempoSyncMode, TransportJumpPolicy, apply_tsyn_table,
    control_stride_for_rate, control_tick_ratio, count_distinct_contributor_controls,
    decode_cp1_contributors, eval_3node_calibration_f64, eval_event_rate_aware_step,
    eval_tsyn_row_physical, eval_tsyn_target_value, eval_uint_base_4x8,
    eval_uint_ultra_delta_3x4x8, invert_tsyn_lobatto_anchors, smooth_relock_q32_tick,
    step_control_frame_default, step_control_tick_3khz_default, subdivision_phase_q32,
    trust_clip_outward_4d,
};
pub use graph::{
    CONVOLVER_FULL_DELTA_FIR_TAPS, CONVOLVER_KAUTZ_ONSET_DELAY_48K, CONVOLVER_LITE_FIR_TAPS,
    CONVOLVER_MAX_CHUNK_FRAMES, CONVOLVER_MAX_FIR_TAPS, CONVOLVER_PRO_DELTA_FIR_TAPS,
    CONVOLVER_SEG_TAPS, CONVOLVER_ULTRA_DELTA_FIR_TAPS, CONVOLVER_WORK_LEN, ConvolverBackend,
    ConvolverFirHistorySlices, ConvolverFirWeightSlices, ConvolverHighSrFirExt,
    ConvolverHighSrMode, CoreOversampler, CoreOversamplerLiveHot, CoreOversamplerMasterCold,
    CoreQuadExtCold, CoreQuadHot, CoreQuadModulation, CoreQuadReadoutHot, CoreQuadReadoutRegs,
    CrossbarFrame, EXT_FIR_STORAGE_F32_LEN, FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT,
    FLAG_OS2X, FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE, GraphBackend,
    KautzColdSlewState, KautzPrefixOverride, KautzStereoBank4, MASTER_CTRL_DELAY_MICROCHUNKS,
    MAX_CORE_QUADS, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS, MAX_SONA_TILES, MAX_TOTAL_SONA_LANES,
    MasterCtrlDelayRingCold, MasterCtrlSnapshotCold, NUM_SONA_BLOCKS, PartitionedSchroederBank,
    QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig, QuadRingMut, QuadStepResult,
    QuadTopRead, RenderMode, SONA_BLOCK_QUAD_SLOTS, SONA_CORE_QUAD_BLOCK_AND_LOCAL,
    SONA_FULL_HOT_DELTA_BYTES, SONA_INTERBLOCK_TILES, SONA_LITE_HOT_BYTES,
    SONA_PRO_HOT_DELTA_BYTES, SONA_ULTRA_HOT_DELTA_BYTES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER,
    SensoryChunkRegs, SensoryParamsHot, SensoryQuadCold, SensoryQuadHot, SonaColdTail,
    SonaFullHotDelta, SonaHotSlab, SonaInterBlockTileHot, SonaLiteHotPrefix, SonaProHotDelta,
    SonaSlabBacking, SonaUltraHotDelta, TIER_SWITCH_XFADE_SAMPLES, TRANSPORT_ALLPASS_G_MAX,
    TRANSPORT_INLINE_QUAD0_CAP, TRANSPORT_MAX_EXT_CAP, TRANSPORT_MAX_QUADS, TRANSPORT_MIN_EXT_CAP,
    TRANSPORT_SCHROEDER_MAX_DELAY, TierCrossbarSnapshot, TransportColdStorageSummary,
    TransportExtRing, TransportExtStorage, TransportModSlots, TransportQuadHotCompact,
    TransportStageState, TransportTopOfFrameResult, VmReloadHeadroom, ZohRateContext,
    apply_transport_mod_slots, ballistic_alpha_f32, check_reload_slab_capacity,
    compute_quad_ext_requirements, compute_schroeder_requirements,
    compute_transport_storage_requirements, dot16_stereo_simd, evaluate_matryoshka_core_quads,
    f32_to_fp16_rtz_with_sat, hoist_core_quads_and_readouts, hydrate_kautz_slice_and_zoh,
    hydrate_slab_from_dnam_bin, init_log_octave_poles_h2_into, latch_immediate_at_init_or_reset,
    latch_quad_nonslew_targets, process_block_sona, process_convolver_48k_chunk,
    process_tick16_const, reload_slab_from_dnam_bin_in_place, required_ext_elements,
    rescale_paralru_lane, reset_to_dc_equilibrium, reset_transport, retime_sample_rate,
    retune_kautz_banks_3khz, round_sample_count_zoh, set_offline_bounce, set_op_t_write_enable,
    set_quad_write_enable, set_render_mode, sona_graph_convolver_chunk_default,
    sona_graph_dispatch_transport_3khz_default, sona_graph_eval_core_quads_default,
    sona_graph_op_n_pre_step_default, sona_graph_sensory_step_default,
    sona_graph_transport_end_default, sona_graph_transport_top_default, step_control_tick_3khz,
    step_frame_sona, step_transport_end_of_frame, step_transport_end_of_frame_with_output,
    step_transport_top_of_frame, step_transport_top_of_frame_reads, switch_tier, tick_quad_3khz,
    tick_transport_3khz,
};
#[cfg(feature = "alloc")]
pub use graph::{
    ExtRingAllocSpec, SonaSlabOwnedBacking, hydrate_slab_owned, hydrate_slab_owned_with_headroom,
};
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
