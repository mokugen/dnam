# Dynamic Neural Audio Modeler (DNAM) — Agent Development & Discovery Guide

## 1. Global Tech Stack & Tooling Rules

- **Monorepo Orchestration (`moon` + `mise`):** Run builds, gates, tests, and dev servers via `moon` ([`.moon/workspace.yml`](.moon/workspace.yml) & [`mise.toml`](mise.toml)): `moon check --all`, `moon check <project>`, or `moon run <project>:<task>`.
- **Dev Servers:** `moon run studio:dev`, `moon run studio:tauri-dev`, and `moon run ui:dev`.
- **Quality Gates:** `moon run <project>:coverage` (≥ 95%), `moon run :format-check`, `moon run :lint`, `moon run :typecheck` (zero `# type: ignore` or `any`), and `moon run :bench`.
- **Git & Contributing:** Read `packages/<pkg>/CONTRIBUTING.md`; sign off commits (`git commit -s`) per [`CLA.md`](CLA.md) & [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero Stalls in Real-Time Loops (`REQ-6.2`, `REQ-6.3`):** Never use `FDIV`, `FSQRT`, integer division (`IDIV`/`DIV`/`SDIV`/`UDIV`), `libm` (`fmodf`), unbounded loops, heap allocations, locks, or uncached syscalls in $48\text{ kHz}$/$3\text{ kHz}$ loops; use [`op_n_commuted_cascade.md`](docs/specs/dsp/op_n_commuted_cascade.md) and [`op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md).
2. **Unconditional BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC Stability:** Enforce [`control_space_audit.md`](docs/specs/training/control_space_audit.md) bounds across `SONA` (`TPT-SVF`, `ParaLRU`, `PolyADAA1`, `Kautz`).
3. **Non-Stop 3-Phase Control Clamping:** Clamp unobserved ranges, switches, and corners per [`parametric_control_bus.md`](docs/specs/dsp/parametric_control_bus.md) during `WebGPU` + CPU `VarPro` solves.
4. **Fail-Fast `64B`-Aligned `zerocopy` `v3.0` Containers:** Validate `v3.0` `"sona"` `.dnam`/`.dnam-proj` via `zerocopy` wire structs, `Ed25519`, and `uplo='U'` per [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md) and [`project_container_format.md`](docs/specs/containers/project_container_format.md).

## 3. Progressive Disclosure & Codebase Discovery Map

| Package / Domain | Package `AGENTS.md` & Contributing | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Guardrails (`tests/`)** | [`README.md`](README.md), [`tests/AGENTS.md`](tests/AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | [`index.md`](docs/index.md), [`project_and_documentation_quality.md`](docs/specs/architecture/project_and_documentation_quality.md) |
| **Engine (`packages/engine/`, `kernels` Ph.2)** | [`engine/AGENTS.md`](packages/engine/AGENTS.md), [`engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) | [`sona_architecture.md`](docs/specs/architecture/sona_architecture.md), [`router_and_stages_topology.md`](docs/specs/dsp/router_and_stages_topology.md) |
| **Capture (`packages/capture/`)** | [`capture/AGENTS.md`](packages/capture/AGENTS.md), [`capture/CONTRIBUTING.md`](packages/capture/CONTRIBUTING.md) | [`project_container_format.md`](docs/specs/containers/project_container_format.md), [`four_stage_pipeline.md`](docs/specs/training/four_stage_pipeline.md) |
| **JS (`packages/js/`)** | [`js/AGENTS.md`](packages/js/AGENTS.md), [`js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) | [`typescript_sdk.md`](docs/specs/sdks/typescript_sdk.md), [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md) |
| **UI (`packages/ui/`)** | [`ui/AGENTS.md`](packages/ui/AGENTS.md), [`ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) | [`control_space_audit.md`](docs/specs/training/control_space_audit.md), [`watermarking_and_marketplace.md`](docs/specs/security/watermarking_and_marketplace.md) |
| **Studio (`packages/studio/`)** | [`studio/AGENTS.md`](packages/studio/AGENTS.md), [`studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) | [`capture_and_alignment.md`](docs/specs/training/capture_and_alignment.md), [`project_container_format.md`](docs/specs/containers/project_container_format.md) |
| **Fixtures (`packages/fixtures/`)** | [`fixtures/AGENTS.md`](packages/fixtures/AGENTS.md), [`fixtures/CONTRIBUTING.md`](packages/fixtures/CONTRIBUTING.md) | [`four_stage_pipeline.md`](docs/specs/training/four_stage_pipeline.md), [`capture_and_alignment.md`](docs/specs/training/capture_and_alignment.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- **Root `README.md` Audience:** Keep [`README.md`](README.md) focused on musicians and OEM/SDK overviews; keep setup and specs in [`CONTRIBUTING.md`](CONTRIBUTING.md), package `README.md`s, and [`docs/`](docs/index.md).
- **Zero Duplication:** Keep `AGENTS.md` files strictly as routing maps (`< 5,000` B root, `< 2,500` B scoped).
- **Single Source of Truth:** Document equations, rules (`r[...]`), and workflows in canonical `docs/` per [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
