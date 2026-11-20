# `packages/capture/` — Pure-IO Capture, `.dnam-proj` & `VarPro` Guide

**`dnam-capture`** is the pure-Rust (`#![forbid(unsafe_code)]`, `wasm32`-safe) IO-buffer capture, `.dnam-proj` (`DNPJ`) container, and `SONA` (`32-bit WebGPU` + Host-`f64` `VarPro`) training crate, depending strictly on [`dnam-engine`](../engine/README.md) (`[dependencies]`) and [`dnam-fixtures`](../fixtures/README.md) (`[dev-dependencies]`). Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) before editing.

## Submodule & Documentation Routing Map

| Submodule | Scoped Guide & Files | Canonical Spec |
| :--- | :--- | :--- |
| **Root & Align (`src/`, `src/align/`)** | [`src/AGENTS.md`](src/AGENTS.md), [`src/align/AGENTS.md`](src/align/AGENTS.md), [`src/lib.rs`](src/lib.rs) | [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md) |
| **Stimulus (`src/stimulus/`)** | [`src/stimulus/AGENTS.md`](src/stimulus/AGENTS.md), [`src/stimulus.rs`](src/stimulus.rs) | [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md) |
| **Project Vault (`src/proj/`)** | [`src/proj/AGENTS.md`](src/proj/AGENTS.md), [`src/proj.rs`](src/proj.rs) | [`project_container_format.md`](../../docs/specs/containers/project_container_format.md) |
| **Metrics & White-Box (`src/metrics/`, `src/whitebox/`)** | [`src/metrics/AGENTS.md`](src/metrics/AGENTS.md), [`src/whitebox/AGENTS.md`](src/whitebox/AGENTS.md) | [`psychoacoustic_quality_and_tiers.md`](../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| **Solver (`src/solver/`)** | [`src/solver/AGENTS.md`](src/solver/AGENTS.md), [`src/solver.rs`](src/solver.rs) | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Tests (`tests/`)** | [`tests/AGENTS.md`](tests/AGENTS.md) | [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md) |
