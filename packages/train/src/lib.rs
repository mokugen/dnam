#![deny(unsafe_code)]
//! `dnam-train` — High-precision double (`f64`), integer-exact Ozaki, 2-D memory planner,
//! and WebGPU/CPU compute primitives for SONA v3.0 system identification.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

#[forbid(unsafe_code)]
pub mod control;
#[forbid(unsafe_code)]
pub mod cpu;
#[forbid(unsafe_code)]
pub mod error;
#[forbid(unsafe_code)]
pub mod gpu;
#[forbid(unsafe_code)]
pub mod linalg;
#[forbid(unsafe_code)]
pub mod losses;
pub mod math;
#[forbid(unsafe_code)]
pub mod planner;
#[forbid(unsafe_code)]
pub mod runner;
#[forbid(unsafe_code)]
pub mod solver;
#[forbid(unsafe_code)]
pub mod wide_gram;

pub use control::{
    ACTIVE_BAND_MIN_ENERGY_FRAC, BAND_PHASE_DECORR_RHO_MAX, CORNER_BARRIER_FLOOR,
    CP1_DELAY_VARIANCE_GATE_SAMPLES2, ControlKindSpec, ControlSpec, Cp1ControlSample, Cp1FitConfig,
    Cp1WireBundle, DYNAMIC_ALLPASS_G_MAX, DYNAMIC_DAMP_POLE_MAX, DYNAMIC_G_LOOP_MAX,
    DYNAMIC_KAUTZ_R_MAX, DynamicCornerAuditReport, DynamicCornerViolation, ExtremeCornerDcReport,
    FOUR_BAND_HI_HZ, FOUR_BAND_LO_HZ, FOUR_BAND_RMS_MATCH_WEIGHT, FourBandTptBank,
    FourBandWienerResult, HoistedStageTargets, MAX_GAIN_6DB_LIN, MAX_GAIN_IDEMPOTENT,
    MAX_RO_POLE_48K_F32, MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP, MIN_DELAY_STANDALONE_SAMP,
    PANAMA_COND_BOUND_TOL, PANAMA_DELTA_TOL, PANAMA_ENSEMBLE_SIZE, PANAMA_EXCLUSION_DINF,
    PANAMA_FULL_GRID_MAX_PVAR, PANAMA_GAMMA_LEV, PANAMA_LAMBDA_CP1, PANAMA_LAMBDA_MULTS,
    PANAMA_MAX_RECOMMENDATIONS, PANAMA_MIN_SIGMA2, PANAMA_TOP_TRIPLES_COUNT, PanamaControlDesc,
    PanamaControlKind, PanamaCornerScore, PanamaInput, PanamaOutput, PanamaSampleRow,
    Phase1ClampOutput, Phase3AuditInput, Phase3AuditReport, RAIL_BURST_FRAMES, RailBurstReport,
    SAT_OCCUPANCY_LIMIT, SonaCoreStabilityParams, StageStabilityConstraint,
    StageTargetAndLatentGrads, StageTargetBounds, StageTargetEvalContext, StageTargetEvalResult,
    StageTargetSolveConfig, StageTargetSolveInput, StageTargetSolveReport, SweepWindowSegment,
    TSYN_SUBDIV_QUARTER, ZERO_DC_CORNER_FRAMES, ZERO_DC_CORNER_MAX_ABS, audit_phase1_controls,
    audit_phase3_dynamic_and_head_gain, build_tsyn_entry_wire, build_tsyn_entry_wire_with_cond,
    check_corner_dynamic_stability, check_preact_occupancy, compute_dynamic_excursion_bounds_32,
    compute_stems_key_sha256, enforce_arena_block0_k_nl_and_clamp,
    enforce_arena0_stability_invariants, eval_control_trajectory_3khz_vm,
    eval_corner_barrier_loss_and_grad, eval_stage_targets_loss_and_grad,
    eval_static_control_bus_targets, eval_stop_grad_corner_and_phys_barrier,
    evaluate_readout_spectral_magnitudes, extract_sweep_window_midpoints, fit_and_pack_cp1,
    fit_and_pack_cp1_audited, hoist_stage_targets_onto_model, pack_cp1_slots_into_wire,
    recommend_panama_corners, render_vm_mono_probe, render_vm_mono_probe_reuse,
    run_phase2_rail_burst, run_phase2_rail_burst_reuse, solve_control_jacobian_32slot,
    solve_point_stage_targets_adjoint, solve_two_pass_four_band_wiener,
    verify_extreme_corners_zero_dc_256,
};
pub use cpu::CpuRunner;
pub use error::TrainError;
pub use gpu::{
    KernelId, MANIFEST_JSON, ShaderVariant, abi, embedded_wgsl, prepare_for_browser_webgpu,
    verify_embedded_shader_digests, verify_manifest_digests,
};
#[cfg(feature = "gpu-wgpu")]
pub use gpu::{WgpuContext, WgpuRunner};
pub use linalg::{
    AdamWConfig, AdamWState, EmrMmConfig, EmrMmResult, EmrSpectralProblem, LbfgsConfig,
    LbfgsResult, LtasKktConfig, LtasKktResult, MIN_CHOLESKY_RIDGE, NestedTierPrefixResult,
    OptResult, ParamBoundSpec, ScaledCoordSpace, SsBroydenConfig, SsBroydenState, TwoLoopLbfgs,
    WhitenedLbfgs, apply_arena_scoped_pcgrad, assemble_ozaki_normal_equations,
    checked_uplo_u_packed_len, cosine_lr_with_warmup, factorize_uplo_u_cholesky,
    solve_emr_mm_cholesky, solve_factorized_uplo_u, solve_factorized_uplo_u_2ch,
    solve_ltas_newton_kkt, solve_nested_tier_prefix_schur, solve_uplo_adaptive_ridge,
    solve_uplo_norm_bounded, solve_uplo_u_cholesky, solve_uplo_u_cholesky_2ch,
    step_runner_inner_opt, symv_uplo_u,
};
pub use losses::{
    AnalyticGaborBank, BESR_WORST_BAND_OFFSET_DB, BesrChannelReport, BesrIntervalBounds,
    BesrIntervalDecision, BesrJointIntervalBounds, BesrLossSpec, Bezout7FilterBank,
    CompositeLossBreakdown, CompositeLossConfig, CompositeLossEvaluator, CompositeLossWeights,
    EXCESS_TAU_FRAC, EXCESS_TAU_MIN, GaborBandKernel, GaborTargetCache, IfGdTargetCache, K_GABOR,
    LtasChannelReport, LtasInterval, LtasLossMode, MrIirTargetCache, NoiseQualification,
    ONE_DB_POWER_FACTOR, Radix2FftPlan, SNR_EPS, SegEsrTargetCache, SpecOtTargetCache,
    TierCertificationReport, TierQualityFloors, Welch33TargetCache, build_hann_window,
    canonical_ltas_33_centers_hz, canonical_ltas_33_intervals_hz, certify_tier_audio,
    check_adjacent_tier_monotone, compute_besr_channel, compute_ltas_channel,
    compute_mr_iir_channel, compute_seg_esr_channel_db, compute_welch_seg_starts,
    eval_gabor_loss_and_adjoint, eval_if_gd_loss_and_adjoint, eval_ltas_and_nyquist_fused,
    eval_ltas_loss_and_adjoint, eval_mr_iir_loss_and_adjoint, eval_nyquist_hinge_and_adjoint,
    eval_runner_gabor_bezout_bridge, eval_seg_esr_cached_and_adjoint,
    eval_seg_esr_loss_and_adjoint, eval_spec_ot_loss_and_adjoint, evaluate_joint_interval_besr,
    evaluate_noise_qualified_besr, evaluate_uncertain_floor_besr, is_tier_pair_monotone,
    smooth_excess_sq, snr_weight,
};
pub use math::{
    BackendKind, CpuSimdFeatures, CpuSimdLevel, DefaultMathBackend, DynamicTrainer, MathBackend,
    ScalarBackend, ScalarF64Vec2, SimdF64Vec2, SimdVec4, TrainerMathBackend, TrainerTask,
    detect_best_backend, detect_cpu_simd, detect_cpu_simd_features, dispatch_trainer,
    warm_simd_feature_caches,
};
pub use planner::{
    BROWSER_WEBGPU_BUDGET_BYTES, CPU_WASM_BUDGET_BYTES, CoreScanBindingFootprints, CoreScanPlan,
    CoreScanWorkload, DESKTOP_CAPSULE_BUDGET_BYTES, ExecutionMode, MAX_WEBGPU_BINDING_BYTES,
    MemoryBudgetProfile, STREAMED_CKPT_EXTRA_TILES, STREAMING_HOST_DEFAULT_BUDGET_BYTES,
    STREAMING_HOST_MAX_TENSOR_BYTES, WAVE_STREAMING_MAX_FRAMES, plan_core_scan_execution,
};
pub use runner::{
    ActiveComputeBackend, AdjointScanInput, AdjointScanOutput, FusedCoreScanInput,
    FusedCoreScanOutput, GaborBezoutInput, GaborBezoutOutput, InnerOptInput, InnerOptOutput,
    OzakiGramInput, OzakiGramOutput, OzakiSplit64Output, PrescreenInput, TileMixInput,
    TileMixOutput, TrainRunner, WaveCoreAdjointInput, WaveCoreAdjointOutput, WaveCoreCascadeInput,
    WaveCoreCascadeOutput, pack_core_quad_params, pack_tile_descriptor, unpack_core_quad_params,
};
pub use solver::{
    AtomicDnprProgress, ControlAudioStem, CoreQuadF64, DIMINISHING_RETURNS_DELTA_DB, DNPR_MAGIC,
    DnprProgressWire, InterBlockTileF64, KautzPairF64, MAX_INTER_TIER_LU_DRIFT,
    NUM_CORE_QUADS_ULTRA, NUM_FIR_TAPS_ULTRA, NUM_INTER_BLOCK_TILES_ULTRA, NUM_KAUTZ_PAIRS_ULTRA,
    NUM_TRANSPORT_QUADS_ULTRA, OpNPreGrads, OpPProbeSummary, PROGRESS_MAGIC_DNPR, Pass1PhysicsHint,
    Pass1ProbeConfig, Pass1ProbeReport, Pass1ProbeResult, Pass2OptConfig, Pass2OptReport,
    Pass3VarProReport, Pass5EmitConfig, Pass5EmitResult, ProgressMetrics, SonaModelF64,
    SonaSolveResult, SonaSolverConfig, SonaStageLatentModsF64, SonaTrainDataset,
    StereoTrainingPairs, TierReadoutPrefixF64, TrainingPhase, TrainingStatus, TransportQuadF64,
    WeightedReadoutStem, compute_itu_bs1770_lufs, decode_dnam_bin_to_model,
    encode_model_to_dnam_bin, eval_kautz_modal_and_adjoint, eval_op_n_pre_and_adjoint,
    eval_op_t_transport_and_adjoint, extract_stat_256d_gram, generate_generic_sobol_seeds,
    patch_model_readouts_weighted_cholesky, quantize_kautz_pole_q8, quantize_ltas_error_q20_i8,
    run_cpu_op_p_probe, run_pass1_probe, run_pass1_probe_with_hint, run_pass2_optimize,
    run_pass2_optimize_with_noise, run_pass3_varpro_readout, run_pass5_emit_and_certify,
    solve_sona_model, solve_sona_model_with_pass1, steiglitz_mcbride_poles, synth_kautz_unit_mode,
};
pub use wide_gram::{
    WIDE_GRAM_BATCH_FRAMES, Wide192Accum, accumulate_uplo_u_wide192,
    contract_256d_gram_tiled_wide192, contract_dim_cross_only_wide192,
    contract_dim_gram_tiled_wide192, contract_dim_gram_tiled_wide192_with_backend,
    hash_gram_input_v3, hash_gram_input_v3_f32, uplo_u_index, uplo_u_packed_len,
};
