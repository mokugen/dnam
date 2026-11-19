# `x86/` — x86_64 SSE2, SSE4.2 & AVX2/FMA Backend Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) for SIMD safety proofs and explicit backend trait implementation rules.

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../x86.rs`](../x86.rs) | Unified `x86_64` module entry & re-exports. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`sse2.rs`](sse2.rs) | Baseline `X86Sse2Backend` 4-lane `f32` SIMD implementation. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`sse42.rs`](sse42.rs) | `X86Sse42Backend` `f32` SIMD backend with SSE4.2 rounding/blends. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`avx2.rs`](avx2.rs), [`avx2/stereo8.rs`](avx2/stereo8.rs) | `X86Avx2FmaBackend` AVX2+FMA `Vec4` & wide 256-bit `Avx2Stereo8` (`StereoVec8`). | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`trainer_f64.rs`](trainer_f64.rs), [`trainer_f64/avx2_hw.rs`](trainer_f64/avx2_hw.rs) | `SimdF64Vec2` & `TrainerMathBackend` `f64` kernels for SSE2/AVX2+FMA. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |