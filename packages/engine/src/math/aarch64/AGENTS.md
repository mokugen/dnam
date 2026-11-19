# `aarch64/` — ARM64 NEON, SVE2 & SME Backend Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) for SIMD safety proofs, `zerocopy::transmute!`, and explicit backend trait implementation rules.

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../aarch64.rs`](../aarch64.rs) | Unified `aarch64` module entry & re-exports. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`neon.rs`](neon.rs) | `NeonBackend` 4-lane `f32` real-time SIMD implementation. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`sve2.rs`](sve2.rs) | `ArmSve2Backend` SVE2 `f32` vector implementation. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`sme.rs`](sme.rs) | `ArmSmeBackend` SME matrix/streaming `f32` implementation. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`trainer_f64.rs`](trainer_f64.rs) | `SimdF64Vec2` & `TrainerMathBackend` for NEON `f64`. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`sve2_f64.rs`](sve2_f64.rs) | `TrainerMathBackend` `f64` Gram/Bézout/adjoint kernels for SVE2. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`sme_f64.rs`](sme_f64.rs) | `TrainerMathBackend` `f64` tile/adjoint kernels for SME. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
