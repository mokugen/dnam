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
    PhasorQ32Bank4, QUAD_FLAG_HAS_EXT, QUAD_FLAG_HOST_WRITE_LOCK, QUAD_FLAG_SCHROEDER_ENABLE,
    StereoAnalyticQuadrature, StereoCarrierRotorVec4, T1DcBlockerKernel, T1DcBlockerStereo,
    T1SubHarmonicKernel, algebraic_unit_rotor_f32, blend_catmull_rom_modulated_f32,
    bspline_barberpole_3phase_f32, catmull_rom_hermite_f32, clamp_transport_delay_f32,
    coercive_loop_limit_vec4, compose_complex_modal_scan, compute_half_alpha_cubic_f32,
    compute_op_p_and_phasor_pair, compute_op_t_mu_win_f32, compute_op_t_pitch_down_g_loop_eff,
    cubic_lagrange_weights_f32, eval_carrier_3basis_stereo_vec4, eval_kautz_sweep_contraction_f32,
    eval_op_t_diagonal_window_f32, eval_op_t_diagonal_window_vec4, eval_q32_quadrature_f32,
    halve_signed_q32_increments, hermite_hann_partition_of_unity_f32, q32_smooth_sin_f32,
    read_barberpole_m3_f32, read_catmull_rom_f32, read_catmull_rom_fp16_u16,
    read_pitch_grain_m2_f32, remap_allpass_chain_states_from_outputs,
    rotate_carrier_stereo_phase_vec4, step_kautz_quad_stereo8, step_kautz_quad_vec4,
    step_op_t_feedback_vec4, step_op_t_wet_dc_stereo_f32, step_schroeder_allpass,
    step_schroeder_quad_vec4, step_we_hold_lag, step_we_hold_lag_u16,
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
    F32_INT_EXACT_THRESHOLD, FULL_SPEC, GpuVec4, HERMITE5_SIN_C1_F32, HERMITE5_SIN_C3_F32,
    HERMITE5_SIN_C5_F32, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_EXP_MASK, IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_BITS,
    IEEE754_F32_MANTISSA_MASK, IEEE754_F32_MIN_NORMAL_EXP, IEEE754_F32_MIN_SUBNORMAL_EXP,
    IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET, IEEE754_F32_ONE_BITS, IEEE754_F32_RCP_SEED,
    IEEE754_F32_RSQRT_SEED, IEEE754_F32_SIGN_MASK, IEEE754_F32_SIGNIFICAND_BITS,
    IEEE754_F32_TO_FP16_MANTISSA_SHIFT, IEEE754_F64_EXP_ALL_ONES, IEEE754_F64_EXP_BIAS,
    IEEE754_F64_MANTISSA_BITS, IEEE754_F64_MAX_NORMAL_BIASED_EXP, IEEE754_FP16_ABS_MASK,
    IEEE754_FP16_EXP_ALL_ONES, IEEE754_FP16_EXP_BIAS, IEEE754_FP16_EXP_MASK,
    IEEE754_FP16_MANTISSA_BITS, IEEE754_FP16_MANTISSA_MASK, IEEE754_FP16_MAX_FINITE_BITS,
    IEEE754_FP16_SIGN_MASK, InterleavedStereo8, KernelVec4, LITE_SPEC, MAX_LOOP_GAIN_F32,
    MAX_PARALRU_R_F32, PRO_SPEC, Q32_INV_2POW31_F32, Q32_INV_2POW32_F32, SonaTierSpec, StereoVec8,
    TANK_MAX_G_LIN_BITS, TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_F32, TANK_MAX_SIGMA_SAT_FP16,
    TANK_MAX_V_CEIL_BITS, TANK_MAX_V_CEIL_F32, ULTRA_SPEC, const_abs_f32, const_f32_pow2,
    const_taylor_inv_fact_f32, exp2_clamped16_f32, f32_to_fp16, f32_to_fp16_u32,
    flush_subnormal_f32_bits, fp16_to_f32, fp16_u32_to_f32, os2x_midpoint_f32x4,
    os2x_midpoint_vec4, pack_fp16x2_u32, scrub_and_flush_f32_bits, unpack_fp16x2_u32,
};
pub use sona_core::{
    CoreQuadParamsStereo4, LinearScanElem, MAX_PARALRU_DUHEM_GAMMA_F32, ONE_THIRD_F32,
    OP_N_PRE_BETA_BYPASS_EPS, OP_N_PRE_FULL_PASS_THRESH, OP_N_PRE_INV_TAU_F32, OP_N_PRE_MAX_BETA_Q,
    OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OP_N_PRE_MIN_RHO_EFF, OP_N_PRE_QUANT_CLAMP_F32, OpNPreHoisted,
    OpNPreHotState, ParaLruDuhemAdjointTrace, ParaLruDuhemForwardOut, ParaLruDuhemParams,
    ParaLruDuhemTileGrads, ParaLruParams, ParaLruTileGrads, QuadratureHysteresis2D, SOFT_RECT_EPS,
    SonaBlockOutputStereo8, SonaBlockParamsStereo4, SonaBlockStateStereo8, SonaParaLruCoeffs4,
    SonaParaLruDuhemCoeffs4, SonaToneWeights4, TptAdjointElem, TptScanElem, TptSvfCoeffs4,
    TptSvfOutputs, apply_linear_scan, apply_stereo_link_stereo8, apply_stereo_link_vec4_pair,
    apply_tpt_adjoint, apply_tpt_init, compose_linear_scan, compose_paralru_adjoint_scan,
    compose_tpt_adjoint_scan, compose_tpt_scan, compute_op_n_pre_inc_q32, couple_ms_input_stereo8,
    eval_c2_wavefolder_adaa_stereo8, eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4,
    eval_poly_adaa1_quad_batched, eval_poly_adaa1_stereo8, paralru_adjoint_step_elem,
    paralru_duhem_adjoint_step_elem, prewarp_tan_pi_f_over_fs_f32, quantize_nearest_step_f32,
    quantize_smooth_fourier_f32, run_paralru_adjoint_scan_partial_tile,
    run_paralru_adjoint_scan_tile, run_paralru_duhem_adjoint_scan_partial_tile,
    run_paralru_duhem_adjoint_scan_tile, run_paralru_duhem_forward_partial_tile,
    run_paralru_duhem_forward_tile, run_paralru_forward_partial_tile, run_paralru_forward_tile,
    scan_linear_prefix_tile, soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8,
    soft_rect_zero_dc_vec4, step_core_quad_stereo8, step_paralru_duhem_stereo8,
    step_paralru_duhem_vec4, step_paralru_stereo8, step_paralru_vec4, step_sona_block_stereo8,
    step_tpt_svf_stereo8, step_tpt_svf_vec4,
};
