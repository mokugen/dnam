# `v3_varpro/` — `v3` Matryoshka VarPro & Adjoint Solver Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) and [`../../../../docs/specs/architecture/tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../v3_varpro.rs`](../v3_varpro.rs) | `fit_v3_matryoshka`, `evaluate_cold_loss_f32_with_backend`. | [`tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md) |
| [`model_f64.rs`](model_f64.rs), [`dataset.rs`](dataset.rs) | `V3MatryoshkaOptF64`, `V3TrainDataset`, `ClampedCoreParams`. | [`tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md) |
| [`gru_core.rs`](gru_core.rs), [`svf_id.rs`](svf_id.rs), [`seed_init.rs`](seed_init.rs), [`seed_init/unlock.rs`](seed_init/unlock.rs) | GRU/SVF forward/reverse cells, seed bank & group unlocking. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`adjoint.rs`](adjoint.rs), [`adjoint/backward.rs`](adjoint/backward.rs) | Adjoint forward pass (`adjoint.rs`) & backward pass (`adjoint/backward.rs`). | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`ltas_adj.rs`](ltas_adj.rs), [`ltas_adj/fft.rs`](ltas_adj/fft.rs) | 33-band Welch LTAS objective, FFT plans & adjoints. | [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`bezout7.rs`](bezout7.rs), [`basis.rs`](basis.rs), [`basis/als.rs`](basis/als.rs), [`basis/cache.rs`](basis/cache.rs) | Bézout 3-band Grams, VarPro Cholesky & `pre_cab` ALS. | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`gabor.rs`](gabor.rs), [`omp.rs`](omp.rs), [`lbfgs.rs`](lbfgs.rs), [`primitives.rs`](primitives.rs) | Gabor init, OMP Kautz pole selection, L-BFGS & `uplo='U'` primitives. | [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`alignment.rs`](alignment.rs), [`polish.rs`](polish.rs), [`polish/`](polish/AGENTS.md) | Causal deadtime/polarity probe & `f32` readout polish submodules. | [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
