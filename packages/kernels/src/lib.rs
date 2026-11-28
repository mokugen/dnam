#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![warn(unreachable_pub)]

//! Shared `#![no_std]`, zero-dependency 32-bit `SONA` DSP, associative parallel-scan,
//! modal/transport, and exact `i32` Ozaki kernels (`ref[sdk.rust.shared-nostd-kernels]`,
//! `docs/specs/architecture/sona_architecture.md`).
//!
//! All non-host runtime modules execute strictly in 32-bit IEEE-754 `f32` and 32-bit
//! integer (`i32` / `u32`) arithmetic with zero external crate dependencies, zero heap
//! allocations, and zero `FDIV` / `FSQRT` / `libm` calls so both native CPU SIMD backends
//! and `tools/sona-shaders` (`cargo-gpu` `--target spirv-unknown-naga-wgsl`) compile one
//! shared mathematical source of truth.

pub mod modal_and_transport;
pub mod ozaki_i32;
pub mod simd;
pub mod sona_core;

pub use modal_and_transport::{
    CarrierBasisWeightsVec4, ComplexModalScanElem, FP16_FINITE_MAX_F32, HILBERT_ALPHA_I,
    HILBERT_ALPHA_I_BITS, HILBERT_ALPHA_Q, HILBERT_ALPHA_Q_BITS, HILBERT_H1_ANCHOR_BITS,
    HILBERT_H1_ANCHOR_RATES_HZ, HILBERT_H1_STAGES, HilbertH1Kernel, HilbertH1RemapHistory,
    HilbertH1StereoKernel, KautzQuadState, OpPAndPhasorOutput, OpTFeedbackParams, OpTFeedbackState,
    PhasorQ32Bank4, QUAD_FLAG_HAS_EXT, QUAD_FLAG_HOST_WRITE_LOCK, StereoAnalyticQuadrature,
    StereoCarrierRotorVec4, T1_PLUS_DEFAULT_SUB_LPF_96K_F32, T1_PLUS_DEFAULT_TRK_LPF_96K_F32,
    T1DcBlockerKernel, T1DcBlockerStereo, T1PlusKernel, T1PlusParams, T1SubHarmonicKernel,
    algebraic_unit_rotor_f32, apply_so4_even_chunk4, apply_so4_even_transpose_vec4,
    apply_so4_even_vec4, apply_so4_odd_chunk4, apply_so4_odd_transpose_vec4, apply_so4_odd_vec4,
    blend_catmull_rom_modulated_f32, bspline_barberpole_3phase_f32, catmull_rom_hermite_f32,
    clamp_transport_delay_f32, coercive_loop_limit_vec4, compose_complex_modal_scan,
    compute_half_alpha_cubic_f32, compute_op_p_and_phasor_pair, compute_op_t_mu_win_f32,
    compute_op_t_pitch_down_g_loop_eff, cubic_lagrange_weights_f32,
    eval_carrier_3basis_stereo_vec4, eval_kautz_sweep_contraction_f32,
    eval_op_t_diagonal_window_f32, eval_op_t_diagonal_window_vec4, eval_q32_quadrature_f32,
    eval_t1_plus_soft_square_f32, halve_signed_q32_increments, hermite_hann_partition_of_unity_f32,
    is_so4_identity, q32_smooth_sin_f32, read_barberpole_m3_f32, read_catmull_rom_f32,
    read_catmull_rom_fp16_u16, read_pitch_grain_m2_f32, remap_allpass_chain_states_from_outputs,
    rotate_carrier_stereo_phase_vec4, so4_theta_grad_vec4, step_kautz_quad_stereo8,
    step_kautz_quad_vec4, step_op_t_feedback_vec4, step_op_t_wet_dc_stereo_f32,
    step_schroeder_allpass, step_schroeder_quad_vec4, step_we_hold_lag, step_we_hold_lag_u16,
    tracker_crosses_branch_cut_f32, wgsl_triangle_fold_q32, write_ring_with_freeze_f32,
    write_ring_with_we_f32,
};
#[cfg(not(target_arch = "spirv"))]
pub use modal_and_transport::{
    compute_t1_dc_blocker_coeffs_bits, compute_t1_dc_blocker_coeffs_clamped,
    interpolate_hilbert_h1_alphas, interpolate_hilbert_h1_alphas_f64,
};
#[cfg(not(target_arch = "spirv"))]
pub use ozaki_i32::reconstruct_ozaki_dot_f64;
pub use ozaki_i32::{
    OZAKI_MAX_CHUNK_LEN, accum_ozaki_chunk_i32, accum_ozaki_pair_i32, pow2_ceil_fp32,
    round_ties_even_f32, slice_f32_mantissa_i8x3,
};
pub use simd::{
    BIASED_INT_MAGIC_F32, DC_BLOCKER_5HZ_48K_F32, DC_BLOCKER_10HZ_48K_BITS,
    F32_INT_EXACT_THRESHOLD, FAST_LOG2_CEIL, FAST_LOG2_FLOOR, FULL_SPEC, FULL_V4_SPEC, G_PRE_E_MAX,
    G_PRE_MAX, GpuVec4, HB15_DN_ODD_TAPS_OUTER, HB15_ODD_TAPS_OUTER, HB23_DN_ODD_TAPS_OUTER,
    HB23_HALF_ORDER, HB23_ODD_TAP_COUNT, HB23_ODD_TAPS_OUTER, HB63_DN_ODD_TAPS_OUTER,
    HB63_HALF_ORDER, HB63_KAISER_BETA, HB63_ODD_TAP_COUNT, HB63_ODD_TAPS_OUTER,
    HERMITE5_SIN_C1_F32, HERMITE5_SIN_C3_F32, HERMITE5_SIN_C5_F32, IEEE754_F32_ABS_MASK,
    IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS, IEEE754_F32_EXP_MASK, IEEE754_F32_IMPLICIT_BIT,
    IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK, IEEE754_F32_MIN_NORMAL_EXP,
    IEEE754_F32_MIN_SUBNORMAL_EXP, IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET, IEEE754_F32_ONE_BITS,
    IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, IEEE754_F32_SIGN_MASK,
    IEEE754_F32_SIGNIFICAND_BITS, IEEE754_F32_TO_FP16_MANTISSA_SHIFT, IEEE754_F64_EXP_ALL_ONES,
    IEEE754_F64_EXP_BIAS, IEEE754_F64_MANTISSA_BITS, IEEE754_F64_MAX_NORMAL_BIASED_EXP,
    IEEE754_FP16_ABS_MASK, IEEE754_FP16_EXP_ALL_ONES, IEEE754_FP16_EXP_BIAS, IEEE754_FP16_EXP_MASK,
    IEEE754_FP16_MANTISSA_BITS, IEEE754_FP16_MANTISSA_MASK, IEEE754_FP16_MAX_FINITE_BITS,
    IEEE754_FP16_SIGN_MASK, InterleavedStereo8, KernelVec4, LITE_SPEC, LITE_V4_SPEC,
    LOG_VCA_MAX_C_KNEE, LOG_VCA_MAX_ENV, LOG_VCA_MAX_OCT, LOG_VCA_MIN_C_KNEE, LOG_VCA_MIN_OCT,
    MAX_ALG_R, MAX_CORE_QUADS_V4, MAX_DUHEM_FLUX_ABS, MAX_LOOP_GAIN_F32, MAX_OUTGOING_PER_QUAD_V4,
    MAX_PARALRU_R_F32, MAX_SONA_TILES_V4, MAX_TOTAL_SONA_LANES_V4, NUM_SONA_BLOCKS_V4, PRO_SPEC,
    PRO_V4_SPEC, Q32_INV_2POW31_F32, Q32_INV_2POW32_F32, SE_BUS_ABS_MAX, SONA_V4_BLOCK_QUAD_SLOTS,
    SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL, SONA_V4_INTERBLOCK_TILES, SONA_V4_TILE_COUNTS,
    SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER, SonaTierSpec, StereoVec8, TANK_MAX_G_LIN_BITS,
    TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_F32, TANK_MAX_SIGMA_SAT_FP16, TANK_MAX_V_CEIL_BITS,
    TANK_MAX_V_CEIL_F32, ULTRA_SPEC, ULTRA_V4_SPEC, UnifiedLogKneeVcaParams,
    apply_stereo_log_knee_vca_stereo8, canonicalize_zero_f32_bits, const_abs_f32,
    const_dc_blocker_pole_bits, const_f32_pow2, const_hb15_zero_phase_response_f32,
    const_hb23_zero_phase_response_f32, const_hb63_odd_taps_outer,
    const_hb63_zero_phase_response_f32, const_taylor_inv_fact_f32, eval_unified_log_knee_vca_f32,
    eval_unified_log_knee_vca_vec4, exp2_clamped5_f32, exp2_clamped16_f32, f32_to_fp16,
    f32_to_fp16_u32, fast_log2_f32, fast_log2_vec4, flush_subnormal_f32_bits, fp16_to_f32,
    fp16_u32_to_f32, is_matryoshka_monotonic, os2x_midpoint_f32x4, os2x_midpoint_vec4,
    pack_fp16x2_u32, pack_wf_mix_and_rect_sign_fp16, scrub_and_flush_f32_bits, unpack_fp16x2_u32,
    unpack_wf_mix_and_rect_sign_fp16, verify_tier_spec,
};
pub use sona_core::{
    AlgAdaa1BackwardOut, CORE_OS_LIVE_LATENCY_1X, CORE_OS_MASTER_2X_LATENCY_1X,
    CORE_OS_MASTER_4X_LATENCY_1X, CoreQuadParamsStereo4, CoreQuadV4Block0StateStereo8,
    CoreQuadV4OutputStereo8, CoreQuadV4ParamsStereo4, HB15_DELAY_SAMPLES_1X, HB15_HIST_LEN,
    HB15_ODD_DELAY_LEN, HB23_DELAY_SAMPLES_4X, HB23_HIST_LEN, HB23_ODD_DELAY_LEN,
    HB63_DELAY_SAMPLES_2X, HB63_HIST_LEN, HB63_ODD_DELAY_LEN, HbFir15Downsampler4Ch,
    HbFir15UpsamplerStereo, HbFir23Downsampler4Ch, HbFir23UpsamplerStereo, HbFir63Downsampler4Ch,
    HbFir63UpsamplerStereo, LinearScanElem, MAX_ADAA_DRIVE, MAX_ADAA_ENDPOINT, MAX_E_TILDE_ABS,
    MAX_G_PRE_GAIN, MAX_GLU_MIX, MAX_LRU_BIAS_DRIVE, MAX_OUT_GAIN, MAX_PARALRU_DUHEM_GAMMA_F32,
    MAX_SVF_K_NL, MIN_SAG_GAIN, MIN_TPT_G, ONE_THIRD_F32, OP_N_PRE_BETA_BYPASS_EPS,
    OP_N_PRE_FULL_PASS_THRESH, OP_N_PRE_INV_TAU_F32, OP_N_PRE_MAX_BETA_Q,
    OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OP_N_PRE_MIN_DEADZONE_WIDTH, OP_N_PRE_MIN_RHO_EFF,
    OP_N_PRE_QUANT_CLAMP_F32, OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH, OpNPreHoisted, OpNPreHotState,
    OpNPreV4Hoisted, OpNPreV4HotState, ParaLruDuhemAdjointTrace, ParaLruDuhemForwardOut,
    ParaLruDuhemParams, ParaLruDuhemTileGrads, ParaLruParams, ParaLruTileGrads,
    PhsPredCorrAdjointTerms, PhsPredCorrPassBCoeffs, PhsPredCorrStepContext,
    QuadratureHysteresis2D, SOFT_RECT_EPS, SOFT_RECT_INV_TWO_EPS, SOFT_RECT_TWO_EPS, SVF_PHS_E_MAX,
    SonaBlockOutputStereo8, SonaBlockParamsStereo4, SonaBlockStateStereo8, SonaParaLruCoeffs4,
    SonaParaLruDuhemCoeffs4, SonaToneFolded3, SonaToneFoldedBlock0, SonaToneWeights4,
    TINY_CROSSING_EPS, TptAdjointElem, TptScanElem, TptSvfCoeffs4, TptSvfOutputs,
    TptSvfPhsPredCorrState, apply_linear_scan, apply_stereo_link_stereo8,
    apply_stereo_link_vec4_pair, apply_tpt_adjoint, apply_tpt_init, box2_quadrature_f32,
    box2_quadrature_stereo8, compose_linear_scan, compose_paralru_adjoint_scan,
    compose_tpt_adjoint_scan, compose_tpt_scan, compute_op_n_pre_inc_q32,
    compute_op_n_pre_v4_inc_q32, couple_ms_input_stereo8, eval_alg_adaa1_backward_f32,
    eval_alg_adaa1_f32, eval_alg_adaa1_stereo8, eval_alg_adaa1_vec4,
    eval_alg_adaa1_with_wavefolder_stereo8, eval_alg_adaa1_with_wavefolder_vec4,
    eval_bounded_glu_stereo8, eval_c2_wavefolder_adaa_stereo8, eval_c2_wavefolder_adaa_vec4,
    eval_g_pre_stereo8, eval_g_pre_vec4, eval_hb15_downsample_4ch_vec4,
    eval_hb15_downsample_stereo, eval_hb15_upsample_stereo, eval_hb23_downsample_4ch_vec4,
    eval_hb23_upsample_stereo, eval_hb63_downsample_4ch_vec4, eval_hb63_upsample_stereo,
    eval_phs_pred_corr_pass_b_coeffs, eval_phs_pred_corr_step_adjoint,
    eval_poly_adaa1_odd_even_vec4, eval_poly_adaa1_quad_batched, eval_poly_adaa1_stereo8,
    eval_poly_adaa1_with_wavefolder_stereo8, eval_poly_adaa1_with_wavefolder_vec4,
    eval_se_bus_gain_stereo8, eval_se_bus_gain_vec4, paralru_adjoint_step_elem,
    paralru_duhem_adjoint_step_elem, paralru_duhem_v4_adjoint_step_elem,
    phs_pred_corr_pass_a_adjoint_elem, phs_pred_corr_pass_a_scan_elem,
    phs_pred_corr_pass_b_adjoint_elem, phs_pred_corr_pass_b_scan_elem,
    prewarp_tan_pi_f_over_fs_f32, quantize_hermite_v4_f32, quantize_nearest_step_f32,
    quantize_smooth_fourier_f32, rect_sign_from_mask, run_paralru_adjoint_scan_partial_tile,
    run_paralru_adjoint_scan_tile, run_paralru_duhem_adjoint_scan_partial_tile,
    run_paralru_duhem_adjoint_scan_tile, run_paralru_duhem_forward_partial_tile,
    run_paralru_duhem_forward_tile, run_paralru_duhem_v4_adjoint_scan_partial_tile,
    run_paralru_duhem_v4_adjoint_scan_tile, run_paralru_duhem_v4_forward_partial_tile,
    run_paralru_duhem_v4_forward_tile, run_paralru_forward_partial_tile, run_paralru_forward_tile,
    scan_linear_prefix_tile, scrub_clamped_alg_endpoint_f32, scrub_clamped_alg_endpoint_stereo8,
    scrub_clamped_alg_endpoint_vec4, soft_rect_quartic_f32, soft_rect_quartic_stereo8,
    soft_rect_quartic_vec4, soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8,
    soft_rect_zero_dc_vec4, step_core_quad_phs_stereo8, step_core_quad_stereo8,
    step_core_quad_v4_phs_stereo8, step_core_quad_v4_stereo8, step_paralru_duhem_stereo8,
    step_paralru_duhem_v4_stereo8, step_paralru_duhem_v4_vec4, step_paralru_duhem_vec4,
    step_paralru_stereo8, step_paralru_v4_stereo8, step_paralru_v4_vec4, step_paralru_vec4,
    step_sona_block_stereo8, step_tpt_svf_folded3_stereo8, step_tpt_svf_folded3_vec4,
    step_tpt_svf_phs_stereo8, step_tpt_svf_phs_vec4, step_tpt_svf_pred_corr_phs_folded_stereo8,
    step_tpt_svf_pred_corr_phs_stereo8, step_tpt_svf_pred_corr_phs_vec4, step_tpt_svf_stereo8,
    step_tpt_svf_vec4,
};
