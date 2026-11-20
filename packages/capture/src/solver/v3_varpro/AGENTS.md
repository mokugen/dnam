# `v3_varpro/` — Phase-1 `VarPro` & Adjoint Map (Phase-5 `SONA` Cutover)

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../v3_varpro.rs`](../v3_varpro.rs), [`cold_eval.rs`](cold_eval.rs) | `fit_v3_matryoshka`, `evaluate_cold_loss_f32_with_backend`. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`model_f64.rs`](model_f64.rs), [`dataset.rs`](dataset.rs) | `V3MatryoshkaOptF64`, `V3TrainDataset`, `ClampedCoreParams`. | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`gru_core.rs`](gru_core.rs), [`svf_id.rs`](svf_id.rs), [`seed_init.rs`](seed_init.rs), [`seed_init/`](seed_init/AGENTS.md) | Forward/reverse cells, seed bank & unlocking. | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`adjoint.rs`](adjoint.rs), [`adjoint/`](adjoint/AGENTS.md) | Dual-rate (`1x`/`2x`) forward, BPTT & ZOH `1x` pullback. | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`ltas_adj.rs`](ltas_adj.rs), [`ltas_adj/`](ltas_adj/AGENTS.md) | 33-band Welch LTAS objective & adjoints. | [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`bezout7.rs`](bezout7.rs), [`basis.rs`](basis.rs), [`basis/`](basis/AGENTS.md) | Bézout 3-band Grams, VarPro Cholesky & `pre_cab`. | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`gabor.rs`](gabor.rs), [`omp.rs`](omp.rs), [`omp/`](omp/AGENTS.md), [`lbfgs.rs`](lbfgs.rs), [`primitives.rs`](primitives.rs) | Gabor init, OMP Kautz, L-BFGS & `uplo='U'`. | [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`alignment.rs`](alignment.rs), [`polish.rs`](polish.rs), [`polish/`](polish/AGENTS.md) | Causal deadtime probe & `f32` readout polish. | [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`stage0_id.rs`](stage0_id.rs), [`stage0_id/`](stage0_id/AGENTS.md), [`stage4_id.rs`](stage4_id.rs), [`stage4_id/`](stage4_id/AGENTS.md) | Black-box `Stage 0` (`Op-P`) & `Stage IV` solvers. | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
