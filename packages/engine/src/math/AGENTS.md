# `src/math` — Hardware SIMD & Division-Free Algebraic Primitives

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`backend.rs`](backend.rs), [`simd_vec4.rs`](backend/simd_vec4.rs) | `SimdVec4` (`KernelVec4`) & `MathBackend` (`StereoVec8`) | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`const_eval.rs`](const_eval.rs) | `dnam_kernels` constants & engine `f64` `const fn` | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`svf_math.rs`](svf_math.rs) | `SVF` / NR2/NR3 reciprocal, `rsqrt`, `tan` prewarp, softclip | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`trainer.rs`](trainer.rs), [`f64_vec2.rs`](trainer/f64_vec2.rs) | `SimdF64Vec2` & `TrainerMathBackend` (`f64` parity) | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`kernels.rs`](kernels.rs), [`spec_math.rs`](spec_math.rs), [`spec_math/AGENTS.md`](spec_math/AGENTS.md) | `PolyADAA`, rotor, `Q0.192`, `log2` LUT, trust clip | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`init_math.rs`](init_math.rs) | Cold-init `f64` math (`exp_f64`, `ln_f64`, `powf_f64`, `sqrt_f64`) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`fp16.rs`](fp16.rs), [`crc32c.rs`](crc32c.rs) | `dnam_kernels` `fp16` codecs & `CRC32C` (`0x82F63B78`) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`scalar.rs`](scalar.rs), [`wasm.rs`](wasm.rs), [`wasm/f64.rs`](wasm/f64.rs) | `ScalarBackend` & `WasmSimd128Backend` (`v128` + `f64x2`) | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`aarch64.rs`](aarch64.rs), [`aarch64/AGENTS.md`](aarch64/AGENTS.md) | ARM64 `NEON`, `SVE2`, `SME` & `f64` trainer backends | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`x86.rs`](x86.rs), [`x86/AGENTS.md`](x86/AGENTS.md) | `x86_64` `SSE2`, `SSE4.2`, `AVX2+FMA` & `f64` backends | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `zerocopy::transmute!` and SIMD safety rules.