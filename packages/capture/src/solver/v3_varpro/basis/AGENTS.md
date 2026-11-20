# `basis/` — `v3` Bézout VarPro Basis Cache & Cholesky ALS Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/training/webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../basis.rs`](../basis.rs) | Per-tier Bézout 3-band Gram assembly & readout packing helpers. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`cache.rs`](cache.rs) | Cached rate-specific `Stage I/II/III` trajectory Gram accumulators. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`als.rs`](als.rs) | Joint 4-tier `uplo='U'` VarPro Cholesky & `pre_cab` alternating LS. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
