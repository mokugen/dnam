//! Multi-platform `32-bit` SIMD & hardware math abstraction (`SimdVec4`, `MathBackend`,
//! `Crc32cBackend`, and startup `detect` probes) plus `#![no_std]` `!libm` ZOH cold math
//! (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.zoh.op-m-poles]`).

#[forbid(unsafe_code)]
pub mod backend;
#[forbid(unsafe_code)]
pub mod const_eval;
#[forbid(unsafe_code)]
pub mod crc32c;
#[forbid(unsafe_code)]
pub mod detect;
#[forbid(unsafe_code)]
pub mod fp16;
#[forbid(unsafe_code)]
pub mod init_math;
#[forbid(unsafe_code)]
pub mod kernels;
#[forbid(unsafe_code)]
pub mod scalar;
#[forbid(unsafe_code)]
pub mod spec_math;
#[forbid(unsafe_code)]
pub mod svf_math;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub mod x86;

#[cfg(target_arch = "wasm32")]
#[forbid(unsafe_code)]
pub mod wasm;

pub use backend::{
    MathBackend, SimdVec4, flush_recursive_state_f32, flush_recursive_state_vec4,
    flush_subnormal_f32_bits,
};
pub use const_eval::{
    DC_BLOCKER_5HZ_48K_F32, DC_BLOCKER_10HZ_48K_BITS, FULL_SPEC, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, IEEE754_F64_RSQRT_SEED, LITE_SPEC,
    MAX_LOOP_GAIN_F32, MAX_PARALRU_R_F32, PRO_SPEC, SonaTierSpec, TANK_DC_BLOCKER_48K_F32,
    TANK_MAX_G_LIN_BITS_DERIVED, TANK_MAX_V_CEIL_BITS_DERIVED, ULTRA_SPEC, const_abs_f64,
    const_c1_clamped_tanh_coeffs, const_cos_f64, const_dc_blocker_pole_bits,
    const_dc_blocker_pole_f32, const_f32_rcp_chebyshev_seed, const_hermite5_quarter_sine_coeffs,
    const_max_f32_bits_le, const_max_f32_bits_le_ratio, const_sin_cos_f64, const_sin_f64,
    const_sqrt_f64, const_tan_f64,
};
pub use crc32c::{Crc32cBackend, crc32c, crc32c_slice, crc32c_software_slice};
pub use detect::{
    BackendKind, CpuSimdFeatures, CpuSimdLevel, DefaultMathBackend, detect_best_backend,
    detect_cpu_simd, detect_cpu_simd_features, warm_simd_feature_caches,
};
pub use dnam_kernels::{InterleavedStereo8, KernelVec4, StereoVec8};
pub use fp16::{
    f32_to_fp16, f32_to_fp16 as f32_to_fp16_bits, f32_to_fp16_u32, fp16_to_f32,
    fp16_to_f32 as fp16_bits_to_f32, fp16_u32_to_f32, pack_fp16x2_u32, unpack_fp16x2_u32,
};
pub use init_math::{
    ceil_f32, ceil_f64, cold_atan_pos_f64, cos_pi_unit_f64, exp_f64, floor_f32, floor_f64, ln_f64,
    powf_f64, prewarp_tan_pi_f_over_fs_f32, rcp_nr6_f64, rcp_signed_f64, rescale_real_pole_zoh,
    rescale_tpt_svf_g_zoh, round_f32, round_f64, round_half_away_f64, sqrt_f64,
};
pub use kernels::{
    algebraic_unit_rotor, algebraic_unit_rotor_f64, clamp_pre_activation, cubic_lagrange_weights,
    cubic_lagrange_weights_f64, eval_poly_adaa_algebraic, eval_poly_adaa_algebraic_f64,
    eval_poly_adaa_antiderivative_f64, eval_poly_adaa_waveshaper, eval_poly_adaa_waveshaper_f64,
    stabilize_complex_rotor,
};
pub use scalar::{ScalarBackend, ScalarVec4};
pub use spec_math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, FP16_MAX_FINITE_F32, INV_TAU_256_LE, InvalidFloatInput,
    LOG2_MANTISSA_LUT_129_BITS, Log2EnvelopeDecomp, MAX_ROTOR_ANGLE_F32_BITS,
    PITCH_LN2_OVER_12_F64_BITS, Q0_192Phase, SAFETY_CEILING_F32, contractive_rotor_f32,
    contractive_rotor_f32_bits, decompose_positive_f32_bits, eval_monotone_log2_f32_bits,
    eval_pitch_profile_f32, f32_bits_at_rank, f32_coordinate_count, f32_nearest_admissible_bits,
    f32_rank_boundary_for_f64, f32_rank_intersection, f32_rank_of_bits, f32_to_fp16_with_telemetry,
    f32_unbiased_exponent, lookup_log2_mantissa_f32, newton6_reciprocal_f64, next_up_pos_f64,
    normalize_declared_range_bits, project_rotor_components_ulp_guard,
    q0_192_phase_increment_from_f32_bits, q32_phase_to_phasor, round_f64_to_f32_bits,
    sanitize_input_sample, sanitize_internal_sample, sanitize_output_sample, saturation_flags,
    step_lfo_phasor_zero_hold, trust_clip_fast_candidate, trust_clip_outward_4d,
    trust_clip_outward_8d, trust_clip_outward_checked_bits, trust_clip_reference_bisection_bits,
    upper_product_f64, upper_sum_squares_f64, widen_f32_bits,
};
pub use svf_math::{
    SVF_G_MAX, SVF_W_MAX, algebraic_softclip_vec4, exp2_signed_oct_f32, recip_nr2_pos_f32,
    recip_nr2_pos_vec4, recip_nr3_pos_f32, recip_nr3_pos_vec4, rsqrt_nr2_f32, rsqrt_nr2_vec4,
    rsqrt_nr3_f32, rsqrt_nr3_vec4, tan_prewarp_f32, tan_prewarp_vec4, zero_dc_asym_sat_vec4,
};

#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    ArmSmeBackend, ArmSve2Backend, NeonBackend, NeonStereo8, NeonVec4, SmeStereo8, Sve2Stereo8,
    Sve2Vec4,
};

#[cfg(target_arch = "x86_64")]
pub use x86::{
    Avx2FmaVec4, Avx2Stereo8, Sse2Vec4, Sse42Vec4, X86Avx2FmaBackend, X86SimdBackend,
    X86Sse2Backend, X86Sse42Backend, X86Stereo8, X86Vec4,
};

#[cfg(target_arch = "wasm32")]
pub use wasm::{WasmSimd128Backend, WasmVec4};
