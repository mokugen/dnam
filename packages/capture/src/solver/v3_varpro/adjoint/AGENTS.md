# `adjoint/` — Dual-Rate Forward, Reverse BPTT & ZOH Pullback Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/training/four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../adjoint.rs`](../adjoint.rs) | Matryoshka loss/gradient entrypoints & workspace driver. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`tier_pass.rs`](tier_pass.rs) | Per-tier forward evaluation & Stage II/III residual assembly. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`backward.rs`](backward.rs) | Reverse-mode BPTT across Stage I Env-SVF, feature bus & GRU blocks. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`rescale_1x.rs`](rescale_1x.rs) | Continuous-time `2x -> 1x` ZOH parameter calibration & Jacobian pullback. | [`tora_architecture.md`](../../../../../../docs/specs/architecture/tora_architecture.md) |
