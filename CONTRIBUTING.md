# Contributing to Dynamic Neural Audio Modeler (DNAM)

Thank you for contributing to the **Dynamic Neural Audio Modeler (DNAM)** monorepo! All contributors—both human engineers and AI coding agents—must follow the licensing, architectural, numerical, testing, and documentation standards defined in this guide.

---

## 1. License, Contributor License Agreement (CLA) & Git Sign-Off

Dynamic Neural Audio Modeler (DNAM) is distributed under the [`DNAM Source-Available & Branding License 1.0.0`](LICENSE). To ensure contributions can be integrated and distributed across community, hardware DSP, and commercial deployments, **all contributors must agree to the [DNAM Contributor License Agreement (`CLA.md`)](CLA.md)**.

### How to Sign Off Commits

Include a standard Git `Signed-off-by` trailer on every commit using `-s`:

```bash
git commit -s -m "feat(engine): optimize SONA hot-slab block execution"
```

Or include the trailer explicitly in your commit message:

```text
Signed-off-by: Your Legal Name <your.email@example.com>
```

---

## 2. Monorepo Package & Specification Contributing Guides

DNAM is organized as a multi-language workspace across `docs/`, `packages/*`, `tools/`, and `tests/`. Start from the single root [`AGENTS.md`](AGENTS.md) discovery map, then read the relevant package `CONTRIBUTING.md` guide before making changes:

| Scope / Package | Directory | Contributing Guide | Package Reference |
| :--- | :--- | :--- | :--- |
| **Specification Manual (`docs/`)** | `docs/` | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | [`docs/index.md`](docs/index.md) |
| **Shared CPU/WebGPU Kernels (`dnam-kernels`)** | `packages/kernels/` | [`packages/kernels/CONTRIBUTING.md`](packages/kernels/CONTRIBUTING.md) | [`packages/kernels/README.md`](packages/kernels/README.md) |
| **Pure 2×2 SONA VM (`dnam-vm`)** | `packages/vm/` | [`packages/vm/CONTRIBUTING.md`](packages/vm/CONTRIBUTING.md) | [`packages/vm/README.md`](packages/vm/README.md) |
| **Rust SIMD Engine (`dnam-engine`)** | `packages/engine/` | [`packages/engine/CONTRIBUTING.md`](packages/engine/CONTRIBUTING.md) | [`packages/engine/README.md`](packages/engine/README.md) |
| **Training Solver & WebGPU (`dnam-train`)** | `packages/train/` | [`packages/train/CONTRIBUTING.md`](packages/train/CONTRIBUTING.md) | [`packages/train/README.md`](packages/train/README.md) |
| **AOT WGSL Shader Generator (`sona-shaders`)** | `tools/sona-shaders/` | [`CONTRIBUTING.md`](CONTRIBUTING.md) | [`tools/sona-shaders/moon.yml`](tools/sona-shaders/moon.yml) |
| **TypeScript / WebAudio SDK (`@dnam-project/dnam`)** | `packages/js/` | [`packages/js/CONTRIBUTING.md`](packages/js/CONTRIBUTING.md) | [`packages/js/README.md`](packages/js/README.md) |
| **Pure-Rust Capture, Solver & `.dnam-proj` (`dnam-capture`)** | `packages/capture/` | [`packages/capture/CONTRIBUTING.md`](packages/capture/CONTRIBUTING.md) | [`packages/capture/README.md`](packages/capture/README.md) |
| **Black-Box Ground-Truth Audio Fixtures (`dnam-fixtures`)** | `packages/fixtures/` | [`packages/fixtures/CONTRIBUTING.md`](packages/fixtures/CONTRIBUTING.md) | [`packages/fixtures/README.md`](packages/fixtures/README.md) |
| **Design System & Visualizers (`@dnam-project/ui`)** | `packages/ui/` | [`packages/ui/CONTRIBUTING.md`](packages/ui/CONTRIBUTING.md) | [`packages/ui/README.md`](packages/ui/README.md) |
| **Desktop Studio (`@dnam-project/studio`)** | `packages/studio/` | [`packages/studio/CONTRIBUTING.md`](packages/studio/CONTRIBUTING.md) | [`packages/studio/README.md`](packages/studio/README.md) |
| **Project Guardrail Suite (`dnam-tests`)** | `tests/` | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | [`tests/moon.yml`](tests/moon.yml) |

---

## 3. Workspace Setup, Local Development & Verification Commands

DNAM uses **`mise`** ([`mise.toml`](mise.toml)) to provision toolchains and **`moon`** ([`.moon/workspace.yml`](.moon/workspace.yml)) to orchestrate and cache tasks across **`cargo`** (Rust 2024 edition) and **`pnpm`** (Node.js `>= 26` LTS):

### 3.1 Setup & Local Development (Desktop Studio UI & React Cosmos)

```bash
# Install pinned toolchains (moon, node, pnpm, rust)
mise install

# Launch the Desktop Studio UI in browser dev mode (Vite HMR at http://localhost:1420)
moon run studio:dev

# Launch the native Tauri v2 Desktop Studio app (Rust host + WebView)
moon run studio:tauri-dev

# Launch the React Cosmos component & visualizer workbench (http://localhost:5000)
moon run ui:dev
```

### 3.2 Quality Gates, Tests & Benchmarks

```bash
# Run the unified moon DAG across all Rust, JS/TS packages, and project guardrails
moon check --all        # format-check, lint, typecheck, build, test
moon run :build
moon run :test
moon run :lint
moon run :format        # auto-format all Rust and TS/CSS files
moon run :format-check
moon run :typecheck
moon run :coverage      # >= 95% coverage across all packages
moon run :bench
moon run studio:test-e2e

# Check a single project via moon
moon check kernels
moon check vm
moon check engine
moon check capture
moon check fixtures
moon check js
moon check ui
moon check studio
moon check tests
```

---

## 4. Core Cross-Cutting Engineering & Numerical Invariants

1. **Zero `FSQRT` / `FDIV` / `libm` in Real-Time Loops:**
   - Never introduce floating-point division (`/`), hardware square-root (`FSQRT`), or `libm` transcendental calls (`sin`, `cos`, `exp`, `tanh`) into the $48\text{ kHz}$ per-sample audio loop or the $3\text{ kHz}$ control-bus update.
   - Always use the division-free pure-`f32` divided-difference `PolyADAA1` (`(m_n, d_n)` cubic odd + quartic even + `C²` wavefolder), passive `2×2` ZDF `TPT-SVF`, contractive `1D ParaLRU`, and `3 kHz` quadrant-reduced contractive rotor specified in [`docs/specs/architecture/sona_architecture.md`](docs/specs/architecture/sona_architecture.md), [`docs/specs/dsp/op_n_paralru_and_polyadaa.md`](docs/specs/dsp/op_n_paralru_and_polyadaa.md), and [`docs/specs/dsp/op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md).
2. **4-Lane SIMD, Compile-Time `const` `SONA` Tiers & 64-Byte Alignment:**
   - Organize `SONA` (`State-Space Operator Neural Architecture`) 4-block Matryoshka tiers (`Lite` ⊂ `Full` ⊂ `Pro` ⊂ `Ultra`: `[1,1,1,1]`, `[2,2,1,1]`, `[3,2,2,1]`, `[4,3,2,1]` as compile-time `const` profiles inside `64 KiB` L1D), `Cabinet` `FIR` + `Kautz` modal slices, and `Op-T` transport lines in exact 4-lane SIMD multiples (`KernelVec4` / `SimdVec4` / `StereoVec8`), and keep `SonaHotSlab` + `v3.0` (`"sona"`) `.dnam` / `.dnam-proj` binary chunks and arenas strictly 64-byte cache-line aligned.
3. **Unconditional Stability & Registered Clamps:**
   - Every pole radius (`r_c ≤ 0.9995`), `ParaLRU` contraction bound, `Op-T` loop gain (`g_lin,m ≤ 0.9990`), all-pass diffusion coefficient (`|a_diff,m| ≤ 0.75`), fractional delay (`τ_m ≥ 4.0`), and control range must obey the bounds in [`docs/specs/training/control_space_audit.md`](docs/specs/training/control_space_audit.md).
4. **Fail-Fast Validation:**
   - Never silently substitute default controls, sample rates, or tensor shapes when loading `v3.0` `.dnam` or `.dnam-proj` containers; return or raise explicit, typed diagnostic errors (`DnamError`).
5. **Safe-Rust-First, `zerocopy` Wire Structs & Strict `#[allow(unsafe_code)]` Discipline (`ref[repo.code.type-safety-and-imports]`):**
   - Never be lazy and reach for `#[allow(unsafe_code)]` or `unsafe` (`UnsafeCell`, `AtomicPtr`, raw pointer casts, or `get_unchecked`) when a 100% safe Rust design exists (such as `AtomicU64` word arrays for lock-free SPSC rings and seqlocks, ping-pong slot indices, disjoint struct field borrows, `zerocopy::transmute!`, or checked `napi` buffer views).
   - Use declarative `#[repr(C)]` `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` structs deriving `FromBytes, IntoBytes, KnownLayout, Immutable` (`Unaligned` where slice-cast) with compile-time `size_of` / alignment assertions instead of manual byte-slicing (`from_le_bytes` / `to_le_bytes`) for `.dnam`, `.dnam-proj`, `DNWB`, and WAV headers (see `.agents/rules/zerocopy-wire-structs.md` and [`docs/specs/architecture/project_and_documentation_quality.md`](docs/specs/architecture/project_and_documentation_quality.md)).
   - `#[allow(unsafe_code)]` may **only** be used when **(a)** it is strictly unavoidable at a hardware/OS intrinsic or C-ABI boundary (`packages/vm/src/{math/{aarch64,x86}.rs,types/fp_env.rs}` and `packages/engine/src/{ffi.rs,provenance/sha256_hw.rs}`), or **(b)** benchmarks prove that the safe Rust version causes a high performance impact on a real-time hot path **and** the `unsafe` version can be formally guaranteed sound with an explicit `// SAFETY:` proof.
   - All application and fixture Rust crates (`packages/studio/rust/`, `packages/capture/`, `packages/fixtures/`), safe VM modules in `packages/vm/src/` (`bin`, `control`, `graph`, `vm`, and safe `math`/`types` submodules), all safe modules in `packages/engine/src/` (`container`, `dynamic`, `engine`, `host`, `memory`, `provenance` safe submodules, `schema`, `types`), `#![no_std]` `packages/kernels` (`dnam-kernels`), and `packages/js/rust/` (which permits only safe `#[unsafe(no_mangle)]` WASM exports) must remain 100% safe Rust with zero `unsafe` blocks.
6. **Zero Duplicated Engine Logic in Language SDKs (`packages/js/`):**
   - Language SDKs (`@dnam-project/dnam`) and their FFI shims (`packages/js/rust/`) must **never** reimplement DSP math, ZOH rescaling, 3-Phase control clamping, Lobatto calibration inversion, schema validation, RFC 8785 JCS canonicalization, or provenance signature assembly in TypeScript (`ref[sdk.ts.runtime.dual-target-drivers]`).
   - Delegate 100% of domain logic to `dnam-engine` in Rust; keep SDK host-language layers strictly focused on idiomatic ergonomics, casing/type coercion, and zero-copy buffer/handle lifecycles.
7. **Property-Based Testing (`proptest!` / `fast-check`), TDD, Coverage (≥ 95%) & Benchmarks:**
   - **Mandatory `proptest!` / `fast-check` Usage:** Use `proptest!` in Rust unit (`#[cfg(test)]` inside `src/**`) and integration (`tests/*`) test suites and `fast-check` (`fc.assert(fc.property(...))`) in TypeScript test suites wherever possible to verify algebraic identities, ULP/rounding bounds, state stability, wire-format roundtrips, and cross-tier/cross-driver equivalence over arbitrary input domains rather than relying solely on fixed hand-picked examples.
   - Write a failing unit or property-based regression test before fixing a bug.
   - Unit, property, and integration tests must cite canonical `r[...]` spec rules via `verify[...]` and must **never** read or parse Markdown text blocks from `docs/` (only [`tests/docs_registry_guardrails.test.ts`](tests/docs_registry_guardrails.test.ts) reads `docs/` to verify references and registries).
   - Maintain ≥ 95% test coverage in Rust (`cargo llvm-cov`) and TypeScript (`vitest run --coverage`), and verify zero performance regressions with `cargo bench` and `vitest bench`.
8. **Modular Decomposition, Zero Code-Golfing & Co-Located Unit Tests (`ref[repo.code.modular-decomposition-and-tests]`):**
   - Treat file length ceilings (`< 900` lines across `packages/kernels`, `packages/engine`, `packages/capture`, `packages/fixtures`, `packages/ui`, `packages/js`, `packages/studio`, and `tests/`), wide argument lists, and the root `AGENTS.md` byte budget ([§5.1](#51-root-agentsmd-budget--package-contributing-guides) item 3) as modularity signals, not bin-packing constraints: decompose large monolithic functions, React components/hooks, and growing production modules into small single-purpose helpers and **hierarchical submodule trees** across both **Rust** (`src/<module>.rs` $\to$ `src/<module>/<sub_aspect>.rs`, `~250–650` lines total, zero `mod.rs`) and **TypeScript/React/JS** (`src/<domain>/<aspect>.ts(x)` $\to$ `src/<domain>/<aspect>/<leaf>.ts(x)` with direct concrete leaf imports, zero internal `index.ts` barrels or `export *` via Biome `noBarrelFile` + `noReExportAll` + `noExcessiveCognitiveComplexity`, zero-barrel granular subpath exports such as `@dnam-project/ui/base/*` in `@dnam-project/ui`, and re-export entrypoints restricted strictly to declared `package.json` `"exports"` in `@dnam-project/dnam`), and offload module invariants and workflows to `packages/<pkg>/CONTRIBUTING.md` rather than playing code golf (`\n\`), packing statements/JSX, or stripping documentation.
   - Follow the 4-tier layered testing hierarchy across both languages: co-locate leaf unit and property tests inside inline `#[cfg(test)] mod tests { ... }` (`proptest!`) in Rust submodules or adjacent `<leaf>.test.ts(x)` / `<leaf>.browser.test.ts(x)` (`fast-check`) beside TS/React modules; place smaller inter-unit integration suites inside inline `#[cfg(test)] mod tests { ... }` in parent Rust `name.rs` facades or `<domain>.test.ts(x)` suites inside `src/<domain>/`; reserve `packages/*/tests/<domain>.rs`, `packages/*/tests/integration/*.test.ts(x)`, `tests/profile/*.test.ts`, and `tests/e2e/*.spec.ts` for package-level integration/E2E suites (never `*_coverage.rs` or `*coverage*.test.ts(x)` grab-bags); and use top-level `tests/` for monorepo-wide integration and guardrail suites. Never create standalone `tests.rs` / `*_tests.rs` unit-test files, `src/**/__tests__/` folders, generic `src/**/test(s).ts(x)` files, or `src/**/*.spec.ts(x)` files (shared helpers belong in `test_support.rs`, `test_utils.rs`, `testing.rs`, `test-support.ts`, `test-utils.ts`, `testing.ts`, `tests/<suite>/*.rs`, or `tests/setup.ts`).
9. **Compile-Time Constant Evaluation & Documented Derivations (`ref[repo.code.constant-eval-and-derivations]`):**
   - Derive numerical constants, polynomial coefficients, filter poles, IEEE-754 boundary bit patterns, and cumulative wire/arena offsets via `const fn` / `const` expressions and `const _: () = assert!(...)` rather than bare magic literals; when offline/analytical derivation is too complex or expensive for compile-time evaluation, centralize the constant and document its complete derivation in `docs/specs/` (`.agents/rules/constant-eval-and-magic-numbers.md`).

---

## 5. Maintaining `AGENTS.md` & Progressive-Disclosure Documentation

DNAM enforces a **Progressive-Disclosure Documentation Architecture** and a **Single Source of Truth (SSOT)** policy so AI agents and developers never encounter stale or duplicated specifications. For detailed specification authoring, `r[...]` rule identifiers, MADR records, templates (`docs/templates/`), Mermaid diagrams, and GitHub MathJax LaTeX rules (`ref[repo.docs.spec-rules-and-adrs]`, `ref[repo.docs.markdown-table-math]`, `ref[repo.docs.github-latex-and-mermaid]`, `ref[repo.docs.relative-links-and-anchors]`), see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

### 5.1 Root `AGENTS.md` Budget & Package Contributing Guides

1. **Single Root [`AGENTS.md`](AGENTS.md) (`< 5,000` bytes):**
   - Loaded into the agent context window on every turn, with **zero nested `AGENTS.md` files** anywhere under `packages/`, `tools/`, or `tests/`.
   - Contains **only** the project summary, global `moon` + `mise` tooling rules, core cross-cutting invariants, and the top-level 11-row Codebase & Specification Discovery Map.
2. **Required Package `CONTRIBUTING.md` Guides:**
   - Every package directory under `packages/` maintains a `packages/<pkg>/CONTRIBUTING.md` guide (and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) covers `docs/` and `tests/`).
   - Documents package dependency boundaries, safety and numerical invariants, local `moon` commands, and pre-commit checklists.
3. **Offloading Rule (Never Wordsmith `AGENTS.md` to Fit Byte Budgets):**
   - When root [`AGENTS.md`](AGENTS.md) approaches its `< 5,000` byte budget as packages or modules evolve, **do not** compress or reword implementation details inside `AGENTS.md` or create nested `AGENTS.md` files. Instead, offload non-routing information to its canonical destination:
     - **System requirements (`REQ-*`), normative specifications (`r[...]`), psychoacoustic metrics (`bESR`), stability clamps, and MADR rationales (`ADR-NNNN`)** $\to$ Offload to [`docs/`](docs/index.md) using the templates in [`docs/templates/`](docs/templates/spec_template.md) (see [§5.2](#52-single-source-of-truth-ssot-documentation-map) and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)).
     - **Package architecture invariants, compiler/runtime flags (`#[forbid(unsafe_code)]`, `#![no_std]`, `zerocopy`), module layouts, and verification commands (`moon check <pkg>`)** $\to$ Offload to `packages/<pkg>/CONTRIBUTING.md`.
     - **Public API examples and CLI command/flag references** $\to$ Offload to `packages/<pkg>/README.md`.
     - **Root landing page ([`README.md`](README.md))** $\to$ Keep tailored for DNAM's target audience (musicians, tone creators, and producers first, with a concise overview for hardware OEMs and plugin/SDK integrators); developer setup, `moon` commands, and internal DSP/spec details belong in [`CONTRIBUTING.md`](CONTRIBUTING.md), `packages/<pkg>/README.md`, and [`docs/`](docs/index.md).
4. **Zero Inline LaTeX Math Inside Markdown Tables (`ref[repo.docs.markdown-table-math]`) & ASCII Layouts for Dense Maps:**
   - GitHub's MathJax renderer wraps inline math (`$...$`) inside Markdown table cells (`<th>` / `<td>`) in scrollable containers that display horizontal scrollbars. Never use `$...$` inside Markdown table rows; use plain text, Unicode symbols (`≤`, `≥`, `×`, `→`, `↔`, `Δ`, `σ`, `∈`), or inline code spans (`` `...` ``).
   - Prefer monospaced ASCII box-drawing diagrams (`text` code blocks) over Markdown tables or prose whenever fixed-width spatial alignment displays information more clearly (such as multi-arena parameter budgets, `64B` cache-line field layouts, hierarchical MAC ledgers, and SIMD lane grids); see [`docs/CONTRIBUTING.md` §4.5](docs/CONTRIBUTING.md#45-preferring-monospaced-ascii-diagrams-over-markdown-tables--prose).

### 5.2 Single-Source-of-Truth (SSOT) Documentation Map

Never copy mathematical equations, stability tables, or JSON schemas into `AGENTS.md` or `CONTRIBUTING.md`. Update the canonical specification in `docs/` (following [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`docs/templates/`](docs/templates/spec_template.md)) and link to it:

| When You Modify... | Update This Canonical Document (SSOT) |
| :--- | :--- |
| **System goals, complex full-rig requirements (`REQ-1.5`), hardware budgets, `bESR` tiers, or `SONA` topology** | [`docs/requirements/system_goals_and_requirements.md`](docs/requirements/system_goals_and_requirements.md), [`docs/specs/architecture/psychoacoustic_quality_and_tiers.md`](docs/specs/architecture/psychoacoustic_quality_and_tiers.md), [`docs/specs/architecture/sona_architecture.md`](docs/specs/architecture/sona_architecture.md) |
| **Any 3-phase control-space audit or automatic clamping rule (`r[train.audit.*]`)** | [`docs/specs/training/control_space_audit.md`](docs/specs/training/control_space_audit.md) |
| **`Op-N` `PolyADAA1`, `ParaLRU`, `C²` wavefolder, or DC resting equilibrium (`r[dsp.op-n.*]`)** | [`docs/specs/dsp/op_n_paralru_and_polyadaa.md`](docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **Op-M Modal Bank, symplectic recurrence, Kautz filter, or contractive rotor (`r[dsp.op-m.*]`)** | [`docs/specs/dsp/op_m_modal_kautz.md`](docs/specs/dsp/op_m_modal_kautz.md) |
| **Op-T Transport, Catmull-Rom fractional delay, diffusers, or dynamic ducking (`r[dsp.op-t.*]`)** | [`docs/specs/dsp/op_t_transport.md`](docs/specs/dsp/op_t_transport.md) |
| **Host-Shell 8-Port Patchbay, `SensoryQuad`, or `SONA` crossbar/core/convolver/transport topology (`r[dsp.router.*]`, `r[dsp.crossbar.*]`, `r[dsp.core.*]`, `r[dsp.cabinet.*]`, `r[dsp.tank.*]`, `r[dsp.zoh.*]`)** | [`docs/specs/dsp/router_and_crossbar_topology.md`](docs/specs/dsp/router_and_crossbar_topology.md) |
| **Parametric Control Bus, Lobatto HDMR, Rank-4/8 FM, Chebyshev LFO, or tempo sync (`r[dsp.ctrl.*]`, `r[dsp.tempo.*]`)** | [`docs/specs/dsp/parametric_control_bus.md`](docs/specs/dsp/parametric_control_bus.md) |
| **`v3.0` `.dnam` / `.dnam-proj` 64B preamble, `zerocopy` wire layouts, JCS manifest, or `INPC`/`STMC` capsules (`r[container.*]`)** | [`docs/specs/containers/dnam_binary_format.md`](docs/specs/containers/dnam_binary_format.md), [`docs/specs/containers/project_container_format.md`](docs/specs/containers/project_container_format.md) |
| **JCS SHA-256, `Ed25519` signatures, `.dnam-seal` (`DNSL`), or Buyer Watermarking (`r[sec.*]`)** | [`docs/specs/security/provenance_and_enclave_drm.md`](docs/specs/security/provenance_and_enclave_drm.md), [`docs/specs/security/watermarking_and_marketplace.md`](docs/specs/security/watermarking_and_marketplace.md) |
| **Capture alignment, `32-bit WebGPU` + Host-`f64` `VarPro` (`i32` Ozaki), `uplo='U'` Gram matrices, or stem expansion (`r[train.*]`)** | [`docs/specs/training/capture_and_alignment.md`](docs/specs/training/capture_and_alignment.md), [`docs/specs/training/webgpu_and_varpro_solver.md`](docs/specs/training/webgpu_and_varpro_solver.md), [`docs/specs/training/iterative_resume_and_caching.md`](docs/specs/training/iterative_resume_and_caching.md) |
| **Repository, code, `zerocopy` wire-struct, or documentation quality rules (`r[repo.*]`)** | [`docs/specs/architecture/project_and_documentation_quality.md`](docs/specs/architecture/project_and_documentation_quality.md) |
| **Architectural trade-offs, mathematical counterexamples, or rejected R&D candidates (`ADR-NNNN`)** | [`docs/adr/`](docs/adr/0022-sona-state-space-operator-and-webgpu-architecture.md) (use [`docs/templates/adr_template.md`](docs/templates/adr_template.md), `NNNN-superseded-*.md` for `status: superseded`, descriptive candidate names, and terse timestamped "why" `## Amendment Appendix` entries per [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)) |

---

## 6. Submitting a Pull Request

1. **Create a topic branch** from `main` (`git checkout -b feat/your-feature`).
2. **Run all package and workspace quality gates** (see [§3](#3-workspace-setup-local-development--verification-commands) and the relevant package `CONTRIBUTING.md`).
3. **Verify `AGENTS.md` byte budgets, registries, and Markdown link integrity** via `moon check tests`.
4. **Sign off your commit** (`git commit -s -m "feat(scope): summary"`).
5. **Open a Pull Request** against `main` describing the motivation, architectural invariants preserved, and benchmark/test results.
