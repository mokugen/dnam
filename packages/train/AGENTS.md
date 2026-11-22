# `packages/train` (`dnam-train`) — SONA v3.0 Training Compute & Universal `f64` Solver Map

Sibling training crate to `dnam-engine` depending strictly on [`dnam-kernels`](../kernels/AGENTS.md) and [`dnam-vm`](../vm/AGENTS.md) (plus optional `wgpu`, `zerocopy`, `sha2`), consuming AOT `.wgsl` artifacts from [`tools/sona-shaders`](../../tools/sona-shaders/AGENTS.md). Zero `dnam-fixtures` dependency. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`README.md`](README.md); run `moon check train` & `moon run train:coverage`.

| Domain | Map / Files | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Crate Modules (`src/`)** | [`src/AGENTS.md`](src/AGENTS.md) (`math`, `wide_gram`, `planner`, `runner`, `cpu`, `gpu`, `linalg`, `losses`, `control`, `solver`) | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md), [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Math & Gram Integration** | [`tests/math_and_wide_gram.rs`](tests/math_and_wide_gram.rs) | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Planner & CPU Kernels** | [`tests/planner_and_cpu_runners.rs`](tests/planner_and_cpu_runners.rs) | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Hardware CPU/`wgpu` Parity** | [`tests/cpu_wgpu_parity.rs`](tests/cpu_wgpu_parity.rs) | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **End-to-End Phased Solver** | [`tests/solver_end_to_end.rs`](tests/solver_end_to_end.rs) (`CP1`, `CEXT`, `TSYN`, `DNPR`, CPU/`wgpu` parity) | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
