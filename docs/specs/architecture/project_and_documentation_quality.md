---
id: spec.arch.project-quality
title: "Repository, Code & Documentation Quality Specification"
status: canonical
version: "3.0.0"
domain: architecture
rule_prefixes:
  - "r[repo."
requirements:
  - REQ-1.6
  - REQ-1.9
  - REQ-4.1
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
  - REQ-8.8
adrs:
  - ADR-0010
  - ADR-0019
  - ADR-0020
  - ADR-0022
---

# Repository, Code & Documentation Quality Specification

This specification defines the normative repository structure, static type-safety and import rules, shared `#![no_std]` kernel and declarative `zerocopy` wire-struct mandates, progressive-disclosure `AGENTS.md` byte budgets, 3-layer documentation separation (`requirements/`, `specs/`, `adr/`), Rust Reference `r[...]` / `ref[...]` / `verify[...]` specification rule identifiers, Markdown YAML frontmatter schemas, and GitHub CommonMark + MathJax / Mermaid rendering contracts across **Dynamic Neural Audio Modeler (DNAM)** ([`CONTRIBUTING.md`](../../../CONTRIBUTING.md), [`docs/CONTRIBUTING.md`](../../CONTRIBUTING.md)). For the architectural rationale behind these rules, see [`../../adr/0010-rust-reference-spec-rules-and-madr-decisions.md`](../../adr/0010-rust-reference-spec-rules-and-madr-decisions.md), [`../../adr/0020-modular-monorepo-and-dnam-proj-container.md`](../../adr/0020-modular-monorepo-and-dnam-proj-container.md), and [`../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md).

---

## 1. Static Type Safety & Package Import Boundaries

`r[repo.code.type-safety-and-imports]`

All packages across the monorepo must enforce strict static typing and lightweight runtime dependency boundaries:

1. **Zero Type Suppressions:** Zero `any` or `@ts-ignore` / `@ts-expect-error` directives in TypeScript (`packages/js/`, `packages/ui/`, `packages/studio/`, `tests/`), and strict `#![forbid(unsafe_code)]` / `#![deny(unsafe_code)]` across all Rust crates (`packages/kernels/`, `packages/engine/`, `packages/fixtures/`, `packages/capture/`, `packages/js/rust/`, `packages/studio/rust/`, and `tools/sona-shaders/`).
2. **Zero Forbidden Heavy Runtime Imports & Pure-IO Capture Boundaries:** `packages/kernels/` (`dnam-kernels`) must remain `#![no_std]` and `#![forbid(unsafe_code)]` with zero external crate dependencies, and `tools/sona-shaders/` stays isolated outside default workspace builds (`cargo-gpu` AOT compilation only). `packages/capture/` must remain 100% pure Rust (`dnam-capture`) over pure IO buffers without external Python/PyTorch/CUDA runtime dependencies, CLI parsers (`clap`), or OS audio drivers (`cpal`); all linear-algebra, WebGPU (`wgpu`) dispatch, alignment, and `.dnam-proj` container operations execute natively in safe Rust (`ADR-0020`, `ADR-0022`).
3. **Safe-Rust-First & Strict `#[allow(unsafe_code)]` Discipline:** All Rust crates (`packages/kernels/`, `packages/engine/`, `packages/fixtures/`, `packages/capture/`, `packages/js/rust/`, `packages/studio/rust/`) must enforce `#![forbid(unsafe_code)]` or `#![deny(unsafe_code)]`. Contributors and AI coding agents must **exhaustively search for safe Rust formulations first** (e.g., `core::sync::atomic::AtomicU64` word arrays for lock-free rings and seqlocks, atomic index ping-pong slots, disjoint struct field borrows, `zerocopy`, and checked `napi` slice borrows) and must never add `#[allow(unsafe_code)]` or `unsafe` blocks out of convenience. `#[allow(unsafe_code)]` is permitted **only** when:
   - **Strictly required by a hardware or C-ABI boundary** that has no safe Rust abstraction (CPU SIMD intrinsics in `packages/engine/src/math/`, hardware `FTZ`/`DAZ` CSR registers in `packages/engine/src/types.rs`, aligned heap allocation in `packages/engine/src/memory.rs`, or raw C-ABI entrypoints in `packages/engine/src/ffi.rs`); or
   - **Proven by benchmark to eliminate a high-impact real-time performance bottleneck** where the safe borrow-/bounds-checked formulation measurably violates real-time throughput/WCET budgets **and** every `unsafe` block is accompanied by a complete `// SAFETY:` proof covering aliasing, alignment, lifetimes, and thread-safety invariants.
4. `r[repo.code.realtime-disassembly-audit]` **Automated Release-Binary Disassembly Audit & Real-Time Source Guardrails (`REQ-6.2`, `REQ-6.3`, [`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
   - **Release-Binary Disassembly & Call-Graph Gate (`packages/engine/tests/realtime_disassembly_audit.rs`):** Wired into `moon run engine:test`, this integration test compiles `libdnam_engine.rlib` in `--release` mode, disassembles all object code via `objdump -dr` and `objdump -dr --demangle`, computes the transitive call-graph closure from all real-time entrypoints (`SonaInstance<*>::process_*`, `DnamEngine<*>::process_*`, `process_tick16_const*`, `step_frame*`, `step_control_tick_3khz`, `refresh_collapsed_head_weights`, `DynamicDnamEngine::process_*`, `ControlBus<*>::tick_3khz*`, `Stage3CabinetModals<*>::retune_kautz_formant_3khz`, `_dnam_engine_process_*`), and fails the build if any reachable real-time symbol contains floating-point division (`fdiv`, `vdiv*`, `divss`, `divps`, `divsd`, `divpd`), floating-point square root (`fsqrt`, `vsqrt*`, `sqrtss`, `sqrtps`, `sqrtsd`, `sqrtpd`), integer division/remainder (`udiv`, `sdiv`, `div`, `divl`, `divq`, `idiv`, `idivl`, `idivq`, `__udivti3`, `__divti3`, `__umodti3`, `__modti3`), heap allocation (`__rust_alloc`, `__rust_dealloc`, `__rust_realloc`, `__rust_alloc_zeroed`, including Rust v0 mangled shims), or uncached OS syscalls (`sysctlbyname`, `getauxval`), or if any symbol in the crate contains an external `libm` relocation (`fmodf`, `fmod`, `sin*`, `cos*`, `tan*`, `exp*`, `log*`, `pow*`, `sqrt*`, `cbrt*`, `hypot*`, `atan*`, `asin*`, `acos*`, `sinh*`, `cosh*`, `tanh*`).
   - **Static Source Guardrail (`tests/code_guardrails.test.ts`):** Verifies that non-test `packages/kernels/src/**/*.rs` and `packages/engine/src/**/*.rs` files contain zero `rem_euclid` calls (which LLVM lowers to `bl _fmodf`) and zero direct `f32`/`f64` transcendental method calls.
5. `r[repo.code.explicit-simd-and-coverage]` **Explicit SIMD Trait Implementations, Zero `coverage(off)` & Multi-Target Verification (`REQ-6.1`, [`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)):**
   - **Zero SIMD-Hiding Macros:** All SIMD backend modules (`packages/engine/src/math/aarch64/{neon,sve2,sme}`, `packages/engine/src/math/x86/{sse2,sse42,avx2}`, `packages/engine/src/math/wasm.rs`, `packages/engine/src/math/scalar.rs`, and `packages/kernels/` scalar/SIMD traits) must be written as explicit, directly inspectable Rust trait implementations without `macro_rules!` wrapper generation that obscures instruction lowering or line-coverage attribution.
   - **Zero `#[cfg_attr(coverage, coverage(off))]`:** Never suppress coverage instrumentation to pass the `>= 95%` coverage gate (`moon run <pkg>:coverage`). Cross-ISA backends must be verified via `moon run engine:test-sve2` (`linux/arm64` QEMU `-cpu max` for `ArmSve2Backend`) and `moon run engine:test-x86` (`x86_64-apple-darwin` under Rosetta 2 with `ROSETTA_ADVERTISE_AVX=1` for `X86Sse2Backend`, `X86Sse42Backend`, and `X86Avx2FmaBackend`).
6. `r[repo.code.modular-decomposition-and-tests]` **Modular Decomposition, Zero Code-Golfing & Co-Located Unit Tests (`REQ-6.1`, [`ADR-0020`](../../adr/0020-modular-monorepo-and-dnam-proj-container.md)):**
   - **Proactive Function & Hierarchical Submodule Decomposition (`Zero Code-Golfing`):** Proactively decompose large monolithic functions, React components/hooks, multi-stage pipelines, wide parameter lists, and growing modules into small, cohesive helpers, context types, and **hierarchical submodule trees** across both **Rust crates** (`packages/*/src/`, `packages/js/rust/`: `src/<module>.rs` $\to$ `src/<module>/<sub_aspect>.rs` $\to$ `src/<module>/<sub_aspect>/<leaf>.rs`, `~250–650` lines per file, zero `mod.rs`) and **TypeScript/React/JS packages** (`packages/js/src/`, `packages/ui/src/`, and downstream JS packages: concrete nested leaf modules `src/<domain>/<aspect>.ts(x)` $\to$ `src/<domain>/<aspect>/<leaf>.ts(x)` and `<driver>.ts` + `<driver>/<domain>.ts` with direct leaf imports, Biome `noBarrelFile` + `noReExportAll` + `noExcessiveCognitiveComplexity` enforcement, zero internal `index.ts` barrels or wildcard `export *`, zero-barrel granular subpath exports such as `@dnam-project/ui/base/*` in `@dnam-project/ui`, and explicit named re-exports permitted strictly on declared `package.json` `"exports"` entrypoints of `@dnam-project/dnam`), rather than stuffing flat siblings or multi-thousand-line drivers (`< 900` lines enforced across `packages/kernels/`, `packages/engine/`, `packages/capture/`, `packages/fixtures/`, `packages/ui/`, `packages/js/`, `packages/studio/`, and `tests/`). Never use `\`-newline continuations (`\n\`), statement/JSX packing, nested ternary chains, or comment/docstring stripping to save lines.
   - **4-Tier Layered Testing Hierarchy (`Unit` → `Inter-Unit` → `Package` → `Top-Level tests/`):** Co-locate leaf unit and property tests inside inline `#[cfg(test)] mod tests { ... }` (`proptest!`) at the bottom of each Rust submodule or adjacent `<leaf>.test.ts(x)` / `<leaf>.browser.test.ts(x)` (`fast-check` + ARIA/`axe-core`) next to each TS/React submodule; place smaller inter-unit integration tests inside inline `#[cfg(test)] mod tests { ... }` in parent Rust `name.rs` facades or `<domain>.test.ts(x)` domain suites inside `src/<domain>/` (without internal `index.ts` barrels); reserve `packages/*/tests/<domain>.rs`, `packages/*/tests/integration/*.test.ts(x)`, `tests/profile/*.test.ts`, and `tests/e2e/*.spec.ts` for package-level subsystem integration, N-API $\leftrightarrow$ WASM parity, E2E, and profiling suites (never `*_coverage.rs` or `*coverage*.test.ts(x)` grab-bag files); and treat top-level `tests/` as the highest level of monorepo-wide integration and guardrail testing. Never create isolated unit-test `.rs` files in `src/`/`rust/` (`tests.rs`, `*_tests.rs`, `test_*.rs`), external `mod tests;` / `mod *_tests;` file declarations, `src/**/__tests__/` directories, generic `src/**/test(s).ts(x)` files, or `src/**/*.spec.ts(x)` files (shared test-only helpers belong in `test_support.rs`, `test_utils.rs`, `testing.rs`, `test-support.ts`, `test-utils.ts`, `testing.ts`, `tests/<suite>/*.rs`, or `tests/setup.ts`).
   - **Hierarchical `AGENTS.md` Delegation:** Keep directory growth modular across scoped `AGENTS.md` files (`ref[repo.docs.agents-md-budgets]`) rather than wordsmithing routing tables.
7. `r[repo.code.constant-eval-and-derivations]` **Compile-Time Constant Evaluation, Derived Layouts & Documented Derivations (`REQ-6.1`, `REQ-6.2`, `.agents/rules/constant-eval-and-magic-numbers.md`):**
   - **Zero Unexplained Magic Numbers or Hand-Counted Offsets:** Derive trigonometric/filter poles, rational minimax/Padé/Hermite polynomial coefficients, bit-exact `f32`/`f64` boundary masks, binary packet offsets, and `64 B`-aligned container/arena offsets via pure `const fn` evaluators (`dnam_kernels::const_eval` / `packages/engine/src/math/const_eval.rs` or module-local `const fn`), cumulative `const` arithmetic, and `const _: () = assert!(...)` compile-time checks (and symbolic TypeScript derivations in `packages/js` and `packages/ui`).
   - **Canonical Spec Documentation When Compile-Time Derivation Is Infeasible or Costly:** Constants produced by offline minimax optimization, global two-binade floating-point search, or empirical calibration that cannot be evaluated cleanly or fast in `const fn` must be centralized in one canonical module, fully derived (system equations, rounding rules, and error bounds) in `docs/specs/`, and checked by `const` assertions or unit tests.
8. `r[repo.code.universal-black-box]` **Universal Black-Box Engine & Solver Purity (`REQ-1.6`, `REQ-1.9`, `.agents/rules/universal-black-box.md`):**
   - **Zero White-Box Device Sub-Circuits or Conditional Stage Flags in `packages/kernels/` and `packages/engine/`:** Real-time DSP stages must consist strictly of general mathematical basis operators configured via learned container weights and coefficients; never introduce device-, brand-, or fixture-specific circuit equations, hardcoded device crossover frequencies, or conditional stage-bypass booleans (`ref[arch.pillars.universal-black-box]`).
   - **Zero Target or Fixture Leakage in `packages/capture/src/`:** Model identification, seed initialization, and control-bus conditioning must remain 100% data-driven from dry/wet stems and generic control schemas, containing zero `dnam-fixtures` imports, fixture identifiers, or control-name heuristics.
9. `r[repo.code.zerocopy-wire-structs]` **Declarative `zerocopy` Wire & Arena Structs (`REQ-8.8`, `.agents/rules/zerocopy-wire-structs.md`, [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
   - Represent all binary preambles, directories, `SonaCellDescWire` cell descriptors, constant-sized `Arena0LiteWire..Arena3UltraDeltaWire` tier payloads, `.dnam-proj` `v3.0`, `DNWB`, and WAV headers via `#[repr(C)]` `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` structs deriving `FromBytes, IntoBytes, KnownLayout, Immutable` (`Unaligned` where slice-cast), paired with compile-time `size_of` / alignment `const _: () = assert!(...)` assertions under `#![forbid(unsafe_code)]` and zero manual `from_le_bytes` / `to_le_bytes` byte-offset slicing.
10. `r[repo.code.shared-nostd-kernels]` **Single-Source `#![no_std]` Kernels (`packages/kernels/`) & AOT `cargo-gpu` `.wgsl` Parity (`REQ-4.1`, `REQ-4.2`, [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
    - Centralize shared `f32`/`i32`/`u32` `SONA` operator math (`SimdVec4`/`GpuVec4`, `const_eval`, software `fp16` codec, `TPT-SVF` `mat2x2` step + compose, `ParaLRU` step + exact forward tile + `O(log2 N)` `LinearScanElem` reverse-adjoint scan, fast-path `PolyADAA1` + batched `rcp_pos_quad_nr2` + `C^2` wavefolder, `KautzModalBank`, `Op-T` readers, `HilbertH1`/`T1` tracker, and `pow2_ceil_fp32` + `3x8-bit` Ozaki slice helpers) inside `packages/kernels/` (`dnam-kernels`, `#![no_std]`, `#![forbid(unsafe_code)]`, `0` external crates) consumed identically by `packages/engine/` (`Rust 1.100.0`) and `tools/sona-shaders/` (`cargo-gpu` `--target spirv-unknown-naga-wgsl`).
    - Commit `naga`-validated `.wgsl` compute artifacts under `tools/sona-shaders/` so normal `moon check --all` builds execute `100%` on stable `Rust 1.100.0` (`0 KB` added to `dnam.wasm`), verify shader rebuild reproducibility in CI when `cargo-gpu` is active, and assert `0-ULP` CPU `ParaLRU` forward & `i32` Ozaki bucket parity, `<= 1.5 ULP`/step WGSL FMA operator parity, and scale-aware `<= 1.0e-4 * max(1.0, trajectory_peak)` parity on `256`-wide linear tree scans.

---

## 2. Two-Tier `AGENTS.md` Progressive-Disclosure Hierarchy

`r[repo.docs.agents-md-budgets]`

`AGENTS.md` files act strictly as concise routing maps pointing to canonical specifications in `docs/specs/` and `CONTRIBUTING.md` guides, never duplicating mathematical derivations or equations:

| `AGENTS.md` Tier | Target Files | Strict Byte Ceiling | Required Content |
| :--- | :--- | :---: | :--- |
| **Tier 1 (Monorepo Root)** | [`AGENTS.md`](../../../AGENTS.md) | `< 5,000 B` | Global `moon` + `mise` commands, core architectural invariants, and the package/domain discovery table. |
| **Tier 2 (Scoped Package / Suite)** | `packages/**/AGENTS.md`, [`tests/AGENTS.md`](../../../tests/AGENTS.md) | `< 2,500 B` | Module-to-file routing table, local `moon` task commands, and links to the canonical `docs/specs/` files. |

---

## 3. Three-Layer Documentation Architecture & YAML Frontmatter

### 3.1 Implementation-Agnostic Requirements Layer (`docs/requirements/`)

`r[repo.docs.requirements-layer]`

[`system_goals_and_requirements.md`](../../requirements/system_goals_and_requirements.md) defines **what** the system must achieve (`REQ-1.1` through `REQ-8.8`, 41 requirements across 8 top-level goals):

1. Every requirement ID `REQ-1.1` through `REQ-8.8` must appear in a canonical Markdown table row (`| **\`REQ-X.Y\`** | ... |`).
2. Outside Markdown link targets (`](...)`), the prose of `docs/requirements/` must remain strictly **implementation-agnostic**, containing zero container file extensions (`.dnam`, `.dnam-proj`, `.dnam-ckpt`, `.dnam-cache`, `.dnam-seal`, `.dnce`) and zero internal wire-schema field names (`max_supported_tier`, `user_metadata`, `royalty_splits`, `allowed_protection_profiles`, `allowed_oem_vendors`).

### 3.2 Specification Rules (`r[...]`, `ref[...]`, `verify[...]`), MADR Records & Frontmatter

`r[repo.docs.spec-rules-and-adrs]`

Every Markdown document under `docs/requirements/`, `docs/specs/`, `docs/adr/`, and `docs/templates/` must begin at line 1 with a YAML frontmatter block (`--- ... ---`) and obey the following traceability rules:

| Construct | Syntax & Location | Normative Invariant |
| :--- | :--- | :--- |
| **Rule Definition** | `` `r[<domain>.<subdomain>.<rule>]` `` on its own line or at the start of a numbered list item in `docs/specs/**/*.md` | Globally unique across the repository; matches one of the file's declared frontmatter `rule_prefixes`; every leaf normative section (`##` / `###` / `####`) in `docs/specs/**/*.md` must define at least one `r[...]` rule. Never used outside `docs/specs/` and `docs/templates/spec_template.md`. |
| **Rule Cross-Reference** | `ref[<domain>.<subdomain>.<rule>]` in `docs/adr/`, `docs/`, or source code comments | Must resolve to an existing `r[<domain>.<subdomain>.<rule>]` definition in `docs/specs/`. |
| **Test Verification Citation** | `verify[<domain>.<subdomain>.<rule>]` in test files (`tests/`, `packages/**/tests/`, and co-located unit tests in `packages/*/src/`) | Must resolve to an existing `r[<domain>.<subdomain>.<rule>]` definition in `docs/specs/`. Every `r[...]` rule in `docs/specs/` is tracked for test verification coverage. |
| **MADR Decision Files** | `docs/adr/NNNN-kebab-case-title.md` (`id: ADR-NNNN`, `status: proposed \| accepted \| superseded`) | Numbered sequentially starting at `0001`. Superseded ADRs must declare `superseded_by: ADR-NNNN` pointing to an active ADR, and normative documents in `docs/specs/` and `docs/requirements/` must never reference a superseded ADR. |
| **Authoring Templates** | [`requirement_template.md`](../../templates/requirement_template.md), [`spec_template.md`](../../templates/spec_template.md), [`adr_template.md`](../../templates/adr_template.md) | Canonical templates must exist in `docs/templates/` with valid YAML frontmatter. |
| **Zero Spec-Text Parsing in Tests** | All test files outside `tests/*_guardrails.test.ts` | Unit, integration, and reference tests must never read or parse Markdown files from `docs/`; they reference specifications via `verify[...]` citations. |

---

## 4. Markdown, Mermaid & GitHub MathJax Rendering Rules

### 4.1 Zero Inline LaTeX Math in Markdown Tables

`r[repo.docs.markdown-table-math]`

No Markdown table row (any line starting with `|`) in any `.md` file across the repository may contain a raw `$` character outside backtick code spans (`` `...` ``). Inside Markdown table cells, format symbolic expressions using backtick code spans with Unicode mathematical symbols or move multi-line derivations into `$$...$$` display math blocks above or below the table.

### 4.2 GitHub CommonMark + MathJax LaTeX & Mermaid Syntax

`r[repo.docs.github-latex-and-mermaid]`

All `$...$` and `$$...$$` math blocks and `mermaid` fenced blocks across `.md` files must satisfy GitHub's two-stage CommonMark + MathJax parser and Mermaid renderer:

1. **Balanced Braces & Delimiters:** Every `$...$` and `$$...$$` block must have balanced `{` and `}` braces and must not place spaces immediately inside inline `$ ... $` delimiters.
2. **Zero CommonMark Backslash-Escape Collisions in Math:** Never use `\_`, `\;`, `\,`, `\{`, `\}`, `\|`, `\\`, `\operatorname{...}`, `\text{...}` containing underscores, or `|<expr>|_1` / `|<expr>|_2` inside `$...$` or `$$...$$` (use `\lbrace`, `\rbrace`, `\Vert`, `\mathrm{...}`, and `\text{...}` with hyphens or spaces instead).
3. **Semicolon-Free Mermaid Sequence Diagrams:** Never use semicolons (`;`) inside a Mermaid `sequenceDiagram` block.

### 4.3 Relative Markdown Link & Heading Anchor Integrity

`r[repo.docs.relative-links-and-anchors]`

Every relative Markdown link (`[label](relative/path.md#anchor)` or `[label](#anchor)`) outside fenced code blocks across `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `CLA.md`, `docs/**/*.md`, `tests/**/*.md`, and `packages/**/*.md` must resolve to an existing file or directory on disk, and any `#section-anchor` fragment must match a GitHub-slugified heading in the target Markdown file.
