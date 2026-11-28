# Dynamic Neural Audio Modeler (DNAM) — Agent Development & Discovery Guide

## 1. Global Tech Stack & Tooling Rules

- **Monorepo (`moon` + `mise`):** [`.moon/workspace.yml`](.moon/workspace.yml), [`mise.toml`](mise.toml): `moon check --all`, `moon check <pkg>`, `moon run <pkg>:<task>` (`studio:dev`, `studio:tauri-dev`, `ui:dev`).
- **Quality Gates:** `moon run <pkg>:coverage` (≥ 95%), `:format-check`, `:lint`, `:typecheck` (zero `@ts-ignore`/`any`), `:bench`.
- **Git:** Read `<pkg>/CONTRIBUTING.md`; sign off (`git commit -s`) per [`CLA.md`](CLA.md) & [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero Stalls (`REQ-6.2`, `REQ-6.3`):** Zero `FDIV`/`FSQRT`/`IDIV`/`libm`, unbounded loops, heap allocs, locks, or uncached syscalls in `48 kHz`/`3 kHz` loops ([`op_n_paralru_and_polyadaa`](docs/specs/dsp/op_n_paralru_and_polyadaa.md), [`op_m_modal_kautz`](docs/specs/dsp/op_m_modal_kautz.md), [`op_t_transport`](docs/specs/dsp/op_t_transport.md)).
2. **Unconditional Stability:** Enforce [`control_space_audit`](docs/specs/training/control_space_audit.md) BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC bounds across crossbar, core, cabinet, and tank.
3. **3-Phase Clamping:** Clamp unobserved ranges/switches/corners via `ControlBus` ([`parametric_control_bus`](docs/specs/dsp/parametric_control_bus.md)) across `WebGPU` + CPU `VarPro`.
4. **`64B` `zerocopy` `v4.0` Containers:** Validate `"sona"` `.dnam`/`.dnam-proj` via `zerocopy`, `Ed25519`, and `uplo='U'` ([`dnam_binary_format`](docs/specs/containers/dnam_binary_format.md), [`project_container_format`](docs/specs/containers/project_container_format.md)).

## 3. Codebase & Specification Discovery Map

| Package / Root | Guide | Canonical Specifications (SSOT) |
| :--- | :--- | :--- |
| **`packages/kernels/`** (`dnam-kernels`) | [`CONTRIBUTING`](packages/kernels/CONTRIBUTING.md) | [`sona_architecture`](docs/specs/architecture/sona_architecture.md), [`op_n_paralru_and_polyadaa`](docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **`packages/vm/`** (`dnam-vm`) | [`CONTRIBUTING`](packages/vm/CONTRIBUTING.md) | [`dnam_binary_format`](docs/specs/containers/dnam_binary_format.md), [`parametric_control_bus`](docs/specs/dsp/parametric_control_bus.md), [`rust_sdk`](docs/specs/sdks/rust_sdk.md) |
| **`packages/engine/`** (`dnam-engine`) | [`CONTRIBUTING`](packages/engine/CONTRIBUTING.md) | [`router_and_crossbar_topology`](docs/specs/dsp/router_and_crossbar_topology.md), [`provenance_and_enclave_drm`](docs/specs/security/provenance_and_enclave_drm.md) |
| **`packages/train/`** (`dnam-train`) | [`CONTRIBUTING`](packages/train/CONTRIBUTING.md) | [`webgpu_and_varpro_solver`](docs/specs/training/webgpu_and_varpro_solver.md), [`psychoacoustic_quality_and_tiers`](docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| **`packages/capture/`** (`dnam-capture`) | [`CONTRIBUTING`](packages/capture/CONTRIBUTING.md) | [`capture_and_alignment`](docs/specs/training/capture_and_alignment.md), [`project_container_format`](docs/specs/containers/project_container_format.md) |
| **`packages/js/`** (`@dnam-project/dnam`) | [`CONTRIBUTING`](packages/js/CONTRIBUTING.md) | [`typescript_sdk`](docs/specs/sdks/typescript_sdk.md), [`dnam_binary_format`](docs/specs/containers/dnam_binary_format.md) |
| **`packages/ui/`** (`@dnam-project/ui`) | [`CONTRIBUTING`](packages/ui/CONTRIBUTING.md) | [`control_space_audit`](docs/specs/training/control_space_audit.md), [`watermarking_and_marketplace`](docs/specs/security/watermarking_and_marketplace.md) |
| **`packages/studio/`** (`@dnam-project/studio`) | [`CONTRIBUTING`](packages/studio/CONTRIBUTING.md) | [`capture_and_alignment`](docs/specs/training/capture_and_alignment.md), [`project_container_format`](docs/specs/containers/project_container_format.md) |
| **`packages/fixtures/`** (`dnam-fixtures`) | [`CONTRIBUTING`](packages/fixtures/CONTRIBUTING.md) | [`webgpu_and_varpro_solver`](docs/specs/training/webgpu_and_varpro_solver.md), [`capture_and_alignment`](docs/specs/training/capture_and_alignment.md) |
| **`tools/sona-shaders/`** (`sona-shaders`) | [`moon.yml`](tools/sona-shaders/moon.yml) | [`webgpu_and_varpro_solver`](docs/specs/training/webgpu_and_varpro_solver.md), [`rust_sdk`](docs/specs/sdks/rust_sdk.md) |
| **`tests/`** (`dnam-tests`) | [`docs/CONTRIBUTING`](docs/CONTRIBUTING.md) | [`index`](docs/index.md), [`project_and_documentation_quality`](docs/specs/architecture/project_and_documentation_quality.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- See [`CONTRIBUTING.md` §5](CONTRIBUTING.md#5-maintaining-agentsmd--progressive-disclosure-documentation) and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the root-only `AGENTS.md` budget (`< 5,000` B, zero nested `AGENTS.md` files), SSOT rule routing, and [`README.md`](README.md) audience scope.
