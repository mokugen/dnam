# `packages/capture/src/solver/` — `SONA v3.0` Solver Adapter Map

See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) (`< 900` LOC, zero `mod.rs`, `100%` data-driven, delegates training & `VarPro` kernels to `dnam-train`).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs) | `SolverOptions`, `SolverQualityMode`, `dnam_train` re-exports, `uplo='U'` Gram, `bESR`. | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`progress.rs`](progress.rs), [`telemetry.rs`](progress/telemetry.rs) | `256 B` (`DNPR`) `zerocopy` seqlock telemetry & `SonaVm` visual scope. | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`stage_a_probe.rs`](stage_a_probe.rs) | Stage A `.dnam-bin` probe & `dnam_train::run_pass1_probe` adapter. | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage_b_backbone.rs`](stage_b_backbone.rs) | 4-tier `StarCenter` solve (`Pass 1..3`) & `256-D` `STAT` extraction. | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`stage_c_knobs.rs`](stage_c_knobs.rs), [`collect.rs`](stage_c_knobs/collect.rs), [`point_gn.rs`](stage_c_knobs/point_gn.rs), [`sweep_windows.rs`](stage_c_knobs/sweep_windows.rs) | `CP1`, `"CEXT"`, `U_int`, `W_ctrl` projection, stem collection & sweep midpoints. | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`stage_d_audit.rs`](stage_d_audit.rs), [`watermark.rs`](stage_d_audit/watermark.rs) | `LUFS` trim (`Pass 5`), 3-Phase audit, `SONA` sign gauge, Tardos & phase ID. | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`stages.rs`](stages.rs) | Inline `CP1` wire packing & clamp descriptor synchronization. | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
