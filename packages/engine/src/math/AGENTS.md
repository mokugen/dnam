# `src/math` — Hardware SIMD & Division-Free Algebraic Primitives

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`backend.rs`](backend.rs) | `SimdVec4` (4-lane `f32`) & `MathBackend` traits + cross-backend `proptest` | [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`const_eval.rs`](const_eval.rs) | Pure `#![no_std]` `const fn` trig, Padé `[5/4]`, Hermite sine, bit bounds & IEEE-754 seeds | [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`trainer.rs`](trainer.rs) | `SimdF64Vec2` (2-lane `f64`) & `TrainerMathBackend` traits (`ScalarBackend` + non-FMA IEEE tile/Bézout parity) | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`kernels.rs`](kernels.rs), [`spec_math.rs`](spec_math.rs), [`spec_math/AGENTS.md`](spec_math/AGENTS.md) | Division-free `PolyADAA`, quadrant-reduced Horner rotor, `Q0.192` phase, pitch/log2 LUT, outward trust clip, `f32` rank bounds | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md), [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`init_math.rs`](init_math.rs) | `#![no_std]` cold-init `f64` math (`exp_f64`, `ln_f64`, `powf_f64`, `sqrt_f64`) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`fp16.rs`](fp16.rs), [`crc32c.rs`](crc32c.rs) | IEEE-754 `f32` ↔ `fp16` (`u16`) & Castagnoli `CRC32C` (`0x82F63B78`) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`scalar.rs`](scalar.rs), [`wasm.rs`](wasm.rs) | `#[forbid(unsafe_code)]` `ScalarBackend` & WebAssembly `v128` SIMD128 `f32`/`f64` backend (`WasmSimd128Backend`) | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`aarch64.rs`](aarch64.rs), [`aarch64/AGENTS.md`](aarch64/AGENTS.md) | ARM64 `NEON`, `SVE2`, `SME` & `f64` trainer backends | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`x86.rs`](x86.rs), [`x86/AGENTS.md`](x86/AGENTS.md) | `x86_64` `SSE2`, `SSE4.2`, `AVX2+FMA` & `f64` trainer backends | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `zerocopy::transmute!` and SIMD safety rules.
