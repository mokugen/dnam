#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

//! Shared `#![no_std]`, zero-dependency 32-bit `SONA` DSP, associative parallel-scan,
//! modal/transport, and exact `i32` Ozaki kernels (`ref[repo.code.shared-nostd-kernels]`,
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
    ComplexModalScanElem, HILBERT_ALPHA_I, HILBERT_ALPHA_I_BITS, HILBERT_ALPHA_Q,
    HILBERT_ALPHA_Q_BITS, HILBERT_H1_STAGES, HilbertH1Kernel, KautzQuadState, OpPAndPhasorOutput,
    PhasorQ32Bank4, T1SubHarmonicKernel, algebraic_unit_rotor_f32, bspline_barberpole_3phase_f32,
    catmull_rom_hermite_f32, coercive_loop_limit_vec4, compose_complex_modal_scan,
    compute_op_p_and_phasor_pair, cubic_lagrange_weights_f32, eval_stage3_sweep_contraction_f32,
    hermite_hann_partition_of_unity_f32, read_barberpole_m3_f32, read_catmull_rom_f32,
    read_catmull_rom_fp16_u16, read_pitch_grain_m2_f32, step_kautz_quad_stereo8,
    step_kautz_quad_vec4, step_schroeder_allpass, step_schroeder_quad_vec4,
    tracker_crosses_branch_cut_f32, write_ring_with_freeze_f32,
};
#[cfg(not(target_arch = "spirv"))]
pub use ozaki_i32::reconstruct_ozaki_dot_f64;
pub use ozaki_i32::{
    OZAKI_MAX_CHUNK_LEN, accum_ozaki_chunk_i32, accum_ozaki_pair_i32, pow2_ceil_fp32,
    slice_f32_mantissa_i8x3,
};
pub use simd::{
    DC_BLOCKER_5HZ_48K_F32, DC_BLOCKER_10HZ_48K_BITS, FULL_SPEC, GpuVec4, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, InterleavedStereo8, KernelVec4, LITE_SPEC,
    MAX_LOOP_GAIN_F32, MAX_PARALRU_R_F32, PRO_SPEC, SonaTierSpec, StereoVec8, ULTRA_SPEC,
    f32_to_fp16, f32_to_fp16_u32, flush_subnormal_f32_bits, fp16_to_f32, fp16_u32_to_f32,
    pack_fp16x2_u32, unpack_fp16x2_u32,
};
pub use sona_core::{
    LinearScanElem, ParaLruParams, ParaLruTileGrads, SOFT_RECT_EPS, SonaBlockOutputStereo8,
    SonaBlockParamsStereo4, SonaBlockStateStereo8, SonaParaLruCoeffs4, SonaToneWeights4,
    TptAdjointElem, TptScanElem, TptSvfCoeffs4, TptSvfOutputs, apply_linear_scan,
    apply_tpt_adjoint, apply_tpt_init, compose_linear_scan, compose_paralru_adjoint_scan,
    compose_tpt_adjoint_scan, compose_tpt_scan, eval_c2_wavefolder_adaa_stereo8,
    eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4, eval_poly_adaa1_quad_batched,
    eval_poly_adaa1_stereo8, paralru_adjoint_step_elem, paralru_newton_step_elem,
    prewarp_tan_pi_f_over_fs_f32, run_paralru_adjoint_scan_tile, run_paralru_newton_scan_tile,
    soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8, soft_rect_zero_dc_vec4, step_paralru_stereo8,
    step_paralru_vec4, step_sona_block_stereo8, step_tpt_svf_stereo8, step_tpt_svf_vec4,
};
