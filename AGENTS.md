# Dynamic Neural Audio Modeler (DNAM) — Agent Development & Discovery Guide

## 1. Global Tech Stack & Tooling Rules

- **Monorepo Orchestration (`moon` + `mise`):** Run builds, gates, and tests via `moon` ([`.moon/workspace.yml`](.moon/workspace.yml) & [`mise.toml`](mise.toml)): `moon check --all`, `moon check <project>`, or `moon run <project>:<task>`.
- **Dev Servers:** `moon run studio:dev`, `moon run studio:tauri-dev`, and `moon run ui:dev`.
- **Quality Gates:** `moon run <project>:coverage` (≥ 95%), `moon run :format-check`, `moon run :lint`, `moon run :typecheck` (zero `@ts-ignore` or `any`), and `moon run :bench`.
- **Git & Contributing:** Read `packages/<pkg>/CONTRIBUTING.md`; sign off commits (`git commit -s`) per [`CLA.md`](CLA.md) & [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Core Cross-Cutting Architectural Invariants

1. **Zero Stalls in Real-Time Loops (`REQ-6.2`, `REQ-6.3`):** Avoid `FDIV`, `FSQRT`, `IDIV`/`DIV`, `libm` (`fmodf`), unbounded loops, heap allocs, locks, or uncached syscalls in `48 kHz`/`3 kHz` loops; see [`op_n_paralru_and_polyadaa.md`](docs/specs/dsp/op_n_paralru_and_polyadaa.md), [`op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md), and [`op_t_transport.md`](docs/specs/dsp/op_t_transport.md).
2. **Unconditional BIBO, Lyapunov, $\mathcal{H}_2$ & Zero-DC Stability:** Enforce [`control_space_audit.md`](docs/specs/training/control_space_audit.md) bounds across `SONA` crossbar, core, cabinet, and tank (`TPT-SVF`, `ParaLRU`, `PolyADAA1`, `Kautz`, `Op-T`).
3. **Non-Stop 3-Phase Control Clamping:** Clamp unobserved ranges, switches, and corners via `ControlBus` per [`parametric_control_bus.md`](docs/specs/dsp/parametric_control_bus.md) across `WebGPU` + CPU `VarPro`.
4. **Fail-Fast `64B`-Aligned `zerocopy` `v3.0` Containers:** Validate `v3.0` `"sona"` `.dnam`/`.dnam-proj` via `zerocopy` wire structs, `Ed25519`, and `uplo='U'` per [`dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md) and [`project_container_format.md`](docs/specs/containers/project_container_format.md).

## 3. Progressive Disclosure & Codebase Discovery Map

| Package / Domain | Scoped `AGENTS.md` & Contributing | Canonical Documentation (SSOT) |
| :--- | :--- | :--- |
| **Guardrails (`tests/`)** | [`README.md`](README.md), [`tests/AGENTS.md`](tests/AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | [`index.md`](docs/index.md), [`project_and_documentation_quality.md`](docs/specs/architecture/project_and_documentation_quality.md) |
| **Packages (`packages/`)** | [`packages/AGENTS.md`](packages/AGENTS.md) (`kernels`, `vm`, `engine`, `capture`, `js`, `ui`, `studio`, `fixtures`) | [`sona_architecture.md`](docs/specs/architecture/sona_architecture.md), [`rust_sdk.md`](docs/specs/sdks/rust_sdk.md), [`typescript_sdk.md`](docs/specs/sdks/typescript_sdk.md) |

## 4. Maintaining `AGENTS.md` & Documentation

- See [`CONTRIBUTING.md` §5](CONTRIBUTING.md#5-maintaining-agentsmd--progressive-disclosure-documentation) and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for `AGENTS.md` byte budgets (`< 5,000` B / `< 2,500` B), SSOT rule routing, and [`README.md`](README.md) audience scope.
