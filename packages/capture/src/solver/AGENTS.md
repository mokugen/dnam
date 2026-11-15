# `packages/capture/src/solver/` — 5-Stage Solver Guide

See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs) | `SolverOptions`, `uplo='U'`, `bESR`. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`backends.rs`](backends.rs) | Supertraits (`backends/*.rs`), `DynamicTrainer`. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`wide_gram.rs`](wide_gram.rs) | `Wide192Accum`, `256x256` Gram. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`bezout_gram.rs`](bezout_gram.rs) | Bézout Gram, `bESR`. | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`bezout_schur.rs`](bezout_schur.rs) | SPD, `MM-Cholesky`. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`stage_a.rs`](stage_a.rs), [`stage_a/`](stage_a/AGENTS.md) | Stage A, `VP-GN`, `EXT`, system ID, reflections. | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage_b_c.rs`](stage_b_c.rs) | Schur, `STAT`. | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`stage_d.rs`](stage_d.rs) | `CP1`, SVD, WH FM. | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`stage_d_basis.rs`](stage_d_basis.rs) | `ctrl_basis_4`. | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`stage_e.rs`](stage_e.rs) | Zero-DC, watermarks. | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`stage_e_polish.rs`](stage_e_polish.rs) | 6-metric polish. | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`stages.rs`](stages.rs), [`stages/`](stages/AGENTS.md) | `AuditBounds`, `Mode 0` harvest. | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
| [`v3_varpro.rs`](v3_varpro.rs), [`v3_varpro/AGENTS.md`](v3_varpro/AGENTS.md) | `v3` Matryoshka VarPro/L-BFGS solver. | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
