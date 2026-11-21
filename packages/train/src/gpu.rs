//! GPU ABI (`abi`), embedded WGSL manifest verification (`shaders`), and native `wgpu 30.0.2`
//! compute execution (`context`, `dispatch`, `ops` under feature `"gpu-wgpu"`).

pub mod abi;
#[cfg(feature = "gpu-wgpu")]
pub mod context;
#[cfg(feature = "gpu-wgpu")]
pub mod dispatch;
#[cfg(feature = "gpu-wgpu")]
pub mod ops;
pub mod shaders;

#[cfg(feature = "gpu-wgpu")]
pub use context::WgpuContext;
#[cfg(feature = "gpu-wgpu")]
pub use ops::WgpuRunner;
pub use shaders::{
    KernelId, MANIFEST_JSON, ShaderVariant, embedded_wgsl, prepare_for_browser_webgpu,
    verify_embedded_shader_digests, verify_manifest_digests,
};
