//! x86-64 (`x86_64`) double-precision (`f64`) trainer SIMD backends
//! (`X86Sse2Backend`, `X86Sse42Backend`, `X86Avx2FmaBackend`).

pub mod avx2_f64;
pub mod sse_f64;

pub use sse_f64::{Avx2FmaF64Vec2, Sse2F64Vec2};
