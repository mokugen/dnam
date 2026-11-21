//! AArch64 (`aarch64`) double-precision (`f64`) trainer SIMD backends
//! (`NeonBackend`, `ArmSve2Backend`, `ArmSmeBackend`).

pub mod neon_f64;
pub mod sve2_sme_f64;

pub use neon_f64::NeonF64Vec2;
