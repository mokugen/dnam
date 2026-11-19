# `aarch64/` — ARM64 NEON, SVE2 & SME Backend Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) for SIMD safety proofs, `zerocopy::transmute!`, and explicit backend trait implementation rules.

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../aarch64.rs`](../aarch64.rs) | Unified `aarch64` module entry & re-exports. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`neon.rs`](neon.rs), [`neon/sha2.rs`](neon/sha2.rs) | `NeonBackend` 4-lane `KernelVec4`, `InterleavedStereo8` & SHA-256. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`sve2.rs`](sve2.rs), [`sve2/backend.rs`](sve2/backend.rs) | `ArmSve2Backend` SVE2 `Sve2Vec4` + `Sve2Stereo8` (`StereoVec8`). | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`sme.rs`](sme.rs) | `ArmSmeBackend` SME `f32` + `SmeStereo8` (`StereoVec8`). | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`trainer_f64.rs`](trainer_f64.rs) | `SimdF64Vec2` & `TrainerMathBackend` for NEON `f64`. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`sve2_f64.rs`](sve2_f64.rs), [`sve2_f64/backend.rs`](sve2_f64/backend.rs) | `TrainerMathBackend` `f64` Gram/Bézout/adjoint kernels for SVE2. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`sme_f64.rs`](sme_f64.rs), [`sme_f64/kernels.rs`](sme_f64/kernels.rs) | `TrainerMathBackend` `f64` tile/adjoint kernels for SME. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |