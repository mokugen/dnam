# `adjoint/` — Dual-Rate Forward, Reverse BPTT & ZOH Pullback Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/training/webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../adjoint.rs`](../adjoint.rs) | Matryoshka loss/gradient entrypoints & workspace driver. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`tier_pass.rs`](tier_pass.rs) | Per-tier forward evaluation & Stage II/III residual assembly. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`backward.rs`](backward.rs) | Reverse-mode BPTT across Stage I Env-SVF, feature bus & GRU blocks. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`rescale_1x.rs`](rescale_1x.rs) | Continuous-time `2x -> 1x` ZOH parameter calibration & Jacobian pullback. | [`sona_architecture.md`](../../../../../../docs/specs/architecture/sona_architecture.md) |
