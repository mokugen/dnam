# `basis/` — `v3` Bézout VarPro Basis Cache & Cholesky ALS Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/training/four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../basis.rs`](../basis.rs) | Per-tier Bézout 3-band Gram assembly & readout packing helpers. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`cache.rs`](cache.rs) | Cached rate-specific `Stage I/II/III` trajectory Gram accumulators. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`als.rs`](als.rs) | Joint 4-tier `uplo='U'` VarPro Cholesky & `pre_cab` alternating LS. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
