//! GPU ABI (`abi`), embedded WGSL manifest verification (`shaders`), and native `wgpu 30.0.2`
//! persistent arena + compute execution (`arena`, `context`, `dispatch`, `ops` under feature `"gpu-wgpu"`).

pub mod abi;
#[cfg(feature = "gpu-wgpu")]
pub mod arena;
#[cfg(feature = "gpu-wgpu")]
pub mod context;
#[cfg(feature = "gpu-wgpu")]
pub mod dispatch;
#[cfg(feature = "gpu-wgpu")]
pub mod ops;
pub mod shaders;

#[cfg(feature = "gpu-wgpu")]
pub use arena::GpuTrainArena;
#[cfg(feature = "gpu-wgpu")]
pub use context::WgpuContext;
#[cfg(feature = "gpu-wgpu")]
pub use ops::WgpuRunner;

/// Shared reference-counted GPU handle (`Arc<T>` on multi-threaded native targets, `Rc<T>` on
/// single-threaded `wasm32` where `wgpu` WebGPU handles are `!Send + !Sync`).
#[cfg(all(feature = "gpu-wgpu", not(target_arch = "wasm32")))]
pub type SharedGpuRef<T> = std::sync::Arc<T>;

/// Shared reference-counted GPU handle (`Arc<T>` on multi-threaded native targets, `Rc<T>` on
/// single-threaded `wasm32` where `wgpu` WebGPU handles are `!Send + !Sync`).
#[cfg(all(feature = "gpu-wgpu", target_arch = "wasm32"))]
pub type SharedGpuRef<T> = std::rc::Rc<T>;
pub use shaders::{
    KernelId, MANIFEST_JSON, ShaderVariant, embedded_wgsl, prepare_for_browser_webgpu,
    verify_embedded_shader_digests, verify_manifest_digests,
};
