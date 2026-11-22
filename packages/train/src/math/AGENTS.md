# `dnam-train::math` — Double-Precision `f64` SIMD Backends Map

Scoped discovery guide for [`src/math.rs`](../math.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Strictly restricts `#[allow(unsafe_code)]` SIMD intrinsics to `aarch64/` and `x86/`, keeping `trainer.rs` and `wasm.rs` under `#[forbid(unsafe_code)]`, with non-FMA IEEE-754 parity against `ScalarBackend`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`trainer.rs`](trainer.rs), [`f64_vec2.rs`](trainer/f64_vec2.rs) | `SimdF64Vec2`, `ScalarF64Vec2`, `TrainerMathBackend`, `DynamicTrainer` runtime ISA dispatch | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`aarch64.rs`](aarch64.rs), [`neon_f64.rs`](aarch64/neon_f64.rs), [`sve2_sme_f64.rs`](aarch64/sve2_sme_f64.rs) | ARM NEON, SVE2, and SME `f64` contraction & dot backends | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`x86.rs`](x86.rs), [`sse_f64.rs`](x86/sse_f64.rs), [`avx2_f64.rs`](x86/avx2_f64.rs), [`hw.rs`](x86/avx2_f64/hw.rs) | x86 SSE2/SSE4.2 & AVX2 `f64` vector math backends | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`wasm.rs`](wasm.rs) | Safe WASM SIMD128 `f64x2` `TrainerMathBackend` implementation | [`rust_sdk.md`](../../../../docs/specs/sdks/rust_sdk.md) |
