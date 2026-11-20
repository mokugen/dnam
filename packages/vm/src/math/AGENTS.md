# `dnam-vm::math` — Multi-ISA `32-bit` SIMD & Cold ZOH Math Map

Scoped discovery guide for [`src/math.rs`](../math.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Strictly excludes `f64` `TrainerMathBackend` and cryptographic hashing.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`backend.rs`](backend.rs), [`simd_vec4.rs`](backend/simd_vec4.rs), [`scalar.rs`](scalar.rs) | `SimdVec4`, `MathBackend`, `ScalarVec4`, `ScalarBackend` | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`const_eval.rs`](const_eval.rs), [`fp16.rs`](fp16.rs), [`crc32c.rs`](crc32c.rs) | `const fn` seeds, `fp16` codecs, Castagnoli `Crc32cBackend` | [`rust_sdk.md`](../../../../docs/specs/sdks/rust_sdk.md) |
| [`init_math.rs`](init_math.rs), [`svf_math.rs`](svf_math.rs), [`spec_math.rs`](spec_math.rs), [`detect.rs`](detect.rs) | `!libm` `ZOH`/`powf`/`tan`, `Q0.192` rotor, `trust_clip`, `CpuSimdLevel` | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`aarch64.rs`](aarch64.rs) (`neon`, `sve2`, `sme`), [`x86.rs`](x86.rs) (`sse2`, `sse42`, `avx2`), [`wasm.rs`](wasm.rs) | Intrinsic `f32` SIMD & hardware CRC-32C backends | [`ADR-0019`](../../../../docs/adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md) |
