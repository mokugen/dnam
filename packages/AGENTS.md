# `packages/` — Monorepo Package Discovery Map

Scoped discovery guide delegated from root [`AGENTS.md`](../AGENTS.md) covering all Rust, WebGPU, TypeScript, React, and Tauri packages. Each package directory also maintains a local `<pkg>/CONTRIBUTING.md` workflow guide.

| Package | Scoped `AGENTS.md` | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Kernels (`kernels/`)** | [`kernels/AGENTS.md`](kernels/AGENTS.md) | [`sona_architecture.md`](../docs/specs/architecture/sona_architecture.md), [`op_n_paralru_and_polyadaa.md`](../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **VM (`vm/`)** | [`vm/AGENTS.md`](vm/AGENTS.md) | [`dnam_binary_format.md`](../docs/specs/containers/dnam_binary_format.md), [`parametric_control_bus.md`](../docs/specs/dsp/parametric_control_bus.md), [`rust_sdk.md`](../docs/specs/sdks/rust_sdk.md) |
| **Engine (`engine/`)** | [`engine/AGENTS.md`](engine/AGENTS.md) | [`sona_architecture.md`](../docs/specs/architecture/sona_architecture.md), [`router_and_crossbar_topology.md`](../docs/specs/dsp/router_and_crossbar_topology.md) |
| **Capture (`capture/`)** | [`capture/AGENTS.md`](capture/AGENTS.md) | [`project_container_format.md`](../docs/specs/containers/project_container_format.md), [`webgpu_and_varpro_solver.md`](../docs/specs/training/webgpu_and_varpro_solver.md) |
| **JS (`js/`)** | [`js/AGENTS.md`](js/AGENTS.md) | [`typescript_sdk.md`](../docs/specs/sdks/typescript_sdk.md), [`dnam_binary_format.md`](../docs/specs/containers/dnam_binary_format.md) |
| **UI (`ui/`)** | [`ui/AGENTS.md`](ui/AGENTS.md) | [`control_space_audit.md`](../docs/specs/training/control_space_audit.md), [`watermarking_and_marketplace.md`](../docs/specs/security/watermarking_and_marketplace.md) |
| **Studio (`studio/`)** | [`studio/AGENTS.md`](studio/AGENTS.md) | [`capture_and_alignment.md`](../docs/specs/training/capture_and_alignment.md), [`project_container_format.md`](../docs/specs/containers/project_container_format.md) |
| **Fixtures (`fixtures/`)** | [`fixtures/AGENTS.md`](fixtures/AGENTS.md) | [`webgpu_and_varpro_solver.md`](../docs/specs/training/webgpu_and_varpro_solver.md), [`capture_and_alignment.md`](../docs/specs/training/capture_and_alignment.md) |
