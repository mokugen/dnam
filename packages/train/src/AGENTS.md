# `packages/train/src` — Rust Implementation Map (`dnam-train`)

Parent discovery map: [`../AGENTS.md`](../AGENTS.md). See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for `unsafe` isolation and dependency rules.

| Subsystem | Map / Files | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Errors** | [`lib.rs`](lib.rs), [`error.rs`](error.rs) | [`rust_sdk.md`](../../../docs/specs/sdks/rust_sdk.md) |
| **`f64` SIMD Math** | [`math.rs`](math.rs), [`math/AGENTS.md`](math/AGENTS.md) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **`Wide192Accum` Gram** | [`wide_gram.rs`](wide_gram.rs), [`tiled.rs`](wide_gram/tiled.rs), [`cross_only.rs`](wide_gram/cross_only.rs) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **2-D Planner & Runner** | [`planner.rs`](planner.rs), [`runner.rs`](runner.rs), [`types.rs`](runner/types.rs) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **CPU Reference Runners** | [`cpu.rs`](cpu.rs), [`cpu/AGENTS.md`](cpu/AGENTS.md) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **`wgpu` Compute & WGSL** | [`gpu.rs`](gpu.rs), [`gpu/AGENTS.md`](gpu/AGENTS.md) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Linear Algebra & VarPro** | [`linalg.rs`](linalg.rs), [`linalg/AGENTS.md`](linalg/AGENTS.md) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Perceptual Losses & Tiers** | [`losses.rs`](losses.rs), [`losses/AGENTS.md`](losses/AGENTS.md) | [`psychoacoustic_quality_and_tiers.md`](../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| **Control Audit & `CP1`** | [`control.rs`](control.rs), [`control/AGENTS.md`](control/AGENTS.md) | [`control_space_audit.md`](../../../docs/specs/training/control_space_audit.md) |
| **5-Pass SONA Solver** | [`solver.rs`](solver.rs), [`solver/AGENTS.md`](solver/AGENTS.md) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
