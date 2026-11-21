#![deny(unsafe_code)]
//! `dnam-train` — High-precision double (`f64`), integer-exact Ozaki, 2-D memory planner,
//! and WebGPU/CPU compute primitives for SONA v3.0 system identification (`M13`).

#![warn(missing_docs)]

#[forbid(unsafe_code)]
pub mod cpu;
#[forbid(unsafe_code)]
pub mod error;
#[forbid(unsafe_code)]
pub mod gpu;
pub mod math;
#[forbid(unsafe_code)]
pub mod planner;
#[forbid(unsafe_code)]
pub mod runner;
#[forbid(unsafe_code)]
pub mod wide_gram;

pub use cpu::CpuRunner;
pub use error::TrainError;
pub use gpu::{
    KernelId, MANIFEST_JSON, ShaderVariant, abi, embedded_wgsl, prepare_for_browser_webgpu,
    verify_embedded_shader_digests, verify_manifest_digests,
};
#[cfg(feature = "gpu-wgpu")]
pub use gpu::{WgpuContext, WgpuRunner};
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
    STREAMING_HOST_MAX_TENSOR_BYTES, plan_core_scan_execution,
};
pub use runner::{
    ActiveComputeBackend, AdjointScanInput, AdjointScanOutput, FusedCoreScanInput,
    FusedCoreScanOutput, GaborBezoutInput, GaborBezoutOutput, InnerOptInput, InnerOptOutput,
    OzakiGramInput, OzakiGramOutput, PrescreenInput, TrainRunner, pack_core_quad_params,
    unpack_core_quad_params,
};
pub use wide_gram::{
    WIDE_GRAM_BATCH_FRAMES, Wide192Accum, accumulate_uplo_u_wide192,
    contract_256d_gram_tiled_wide192, contract_dim_cross_only_wide192,
    contract_dim_gram_tiled_wide192, contract_dim_gram_tiled_wide192_with_backend,
    hash_gram_input_v3, hash_gram_input_v3_f32, uplo_u_index, uplo_u_packed_len,
};
