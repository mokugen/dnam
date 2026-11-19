# `packages/capture/src/solver/` — `v3` Solver Pipeline Guide

See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md). Keep every `.rs` module `< 900` lines with zero `mod.rs`.

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs) | `SolverOptions`, `SolverQualityMode`, `uplo='U'`, `bESR` warmup, causal probe. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`progress.rs`](progress.rs), [`telemetry.rs`](progress/telemetry.rs) | `256 B` (`DNPR`) lock-free seqlock + coarse event trainer progress & live scope. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`backends.rs`](backends.rs) | Supertraits (`backends/*.rs`), `DynamicTrainer`. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`wide_gram.rs`](wide_gram.rs), [`cross_only.rs`](wide_gram/cross_only.rs) | `Wide192Accum`, `256x256` Gram & cross-only. | [`four_stage_pipeline.md`](../../../../docs/specs/training/four_stage_pipeline.md) |
| [`stage_a_probe.rs`](stage_a_probe.rs) | Stage A crossover, FIR-64, Kautz poles, Stage IV delays. | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage_b_backbone.rs`](stage_b_backbone.rs), [`whitebox.rs`](stage_b_backbone/whitebox.rs) | Joint 4-tier `StarCenter` Matryoshka solve, white-box `f32` recovery & `STAT`. | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`stage_c_knobs.rs`](stage_c_knobs.rs), [`jacobian.rs`](stage_c_knobs/jacobian.rs), [`sweep_windows.rs`](stage_c_knobs/sweep_windows.rs) | `CP1`, `U_int`, `W_ctrl` multi-knob projection, interior sweep extraction & Jacobians. | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`stage_d_audit.rs`](stage_d_audit.rs), [`watermark.rs`](stage_d_audit/watermark.rs) | `LUFS` trim, 3-Phase audit, buyer Tardos & modal watermarks. | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`stages.rs`](stages.rs) | Inline `CP1` & clamp descriptor arena synchronization. | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
| [`v3_varpro.rs`](v3_varpro.rs), [`v3_varpro/AGENTS.md`](v3_varpro/AGENTS.md) | `v3` Matryoshka VarPro/L-BFGS solver. | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
