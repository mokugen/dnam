//! Double-precision (`f64`) trainer SIMD math backends (`TrainerMathBackend`, `SimdF64Vec2`,
//! `DynamicTrainer`) built atop `dnam-vm`'s runtime-dispatched [`MathBackend`] family.

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub mod aarch64;
#[forbid(unsafe_code)]
pub mod trainer;
#[cfg(target_arch = "wasm32")]
#[forbid(unsafe_code)]
pub mod wasm;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub use aarch64::NeonF64Vec2;
#[cfg(target_arch = "wasm32")]
pub use dnam_vm::math::WasmSimd128Backend;
#[cfg(target_arch = "aarch64")]
pub use dnam_vm::math::{ArmSmeBackend, ArmSve2Backend, NeonBackend};
pub use dnam_vm::math::{
    BackendKind, CpuSimdFeatures, CpuSimdLevel, DefaultMathBackend, MathBackend, ScalarBackend,
    SimdVec4, detect_best_backend, detect_cpu_simd, detect_cpu_simd_features,
    warm_simd_feature_caches,
};
#[cfg(target_arch = "x86_64")]
pub use dnam_vm::math::{X86Avx2FmaBackend, X86SimdBackend, X86Sse2Backend, X86Sse42Backend};
pub use trainer::{
    DynamicTrainer, ScalarF64Vec2, SimdF64Vec2, TrainerMathBackend, TrainerTask, dispatch_trainer,
};
#[cfg(target_arch = "wasm32")]
pub use wasm::WasmF64Vec2;
#[cfg(target_arch = "x86_64")]
pub use x86::{Avx2FmaF64Vec2, Sse2F64Vec2};
