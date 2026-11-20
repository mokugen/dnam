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

This specification defines the language-agnostic repository engineering invariants for **Dynamic Neural Audio Modeler (DNAM)**: static type safety, isolated unsafe and FFI boundaries, acyclic dependency graphs, hierarchical modular decomposition, four-tier test organization, compile-time constant evaluation, universal black-box purity, progressive-disclosure routing maps, one-way clean-room documentation separation, and Markdown rendering rules.

For authoring workflows, see the [root contributing guide](../../../CONTRIBUTING.md) and [docs contributing guide](../../CONTRIBUTING.md). For architectural rationales behind these engineering guardrails, see [ADR-0010](../../adr/0010-rust-reference-spec-rules-and-madr-decisions.md), [ADR-0020](../../adr/0020-modular-monorepo-and-dnam-proj-container.md), and [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md).

---

## 1. Static Type Safety & Package Import Boundaries

All packages across the workspace enforce strict static typing, isolated unsafe and FFI boundaries, acyclic dependency ordering, hierarchical decomposition, compile-time constant derivation, and universal black-box purity.

### 1.1 Static Type Safety, Acyclic Dependencies & Unsafe Isolation

`r[repo.code.type-safety-and-imports]`

Every compiled and scripted module across the repository enforces strict static typing and compiler-level type checking. Untyped dynamic escape hatches, implicit dynamic casts, and type-checker suppression directives are forbidden across all production and test modules.

Workspace modules enforce a strict 5-layer diamond dependency graph alongside an isolated offline compute-shader generator. Zero-dependency portable 32-bit arithmetic and DSP primitives sit at the root, feeding a zero-allocation programmable DSP virtual machine for `.dnam-bin` inference without file parsing, cryptography, JSON metadata, or solver math.

Above the virtual machine, the system-identification solver library and the runtime DRM and host shell (`.dnam` wrapper, lock-free queues, Tardos watermarking, and preview sweep protection) remain strictly mutually independent siblings. Capture and project orchestration (`.dnam-proj`) sit at the diamond apex above both, followed downstream by host bindings, UI components, and application shells.

Capture and system-identification modules execute strictly over pure in-memory audio and container buffers. They must carry zero coupled CLI argument parsers, external ML frameworks, or OS audio device drivers ([ADR-0020](../../adr/0020-modular-monorepo-and-dnam-proj-container.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

Safe-by-default compilation prohibits memory-unsafe blocks across all pure-logic libraries, modules, and applications. Narrow unsafe opt-ins are permitted strictly at bounded low-level seams with exhaustive safety documentation:

| Permitted Exception Scope | Normative Boundary & Safety Requirement |
| :--- | :--- |
| **Hardware or FFI Boundary** | Allowed only where no safe abstraction exists: explicit vector intrinsics, hardware floating-point control-status register guards, cache-line-aligned heap allocation, and raw foreign-function ABI entrypoints. |
| **Benchmark-Proven RT Bottleneck** | Allowed only when profiling proves safe bounds checking violates real-time WCET budgets and every unsafe block carries a complete safety proof covering aliasing, alignment, lifetimes, and thread safety. |

### 1.2 Modular Decomposition, Zero Code-Golfing & 4-Tier Testing

`r[repo.code.modular-decomposition-and-tests]`

Large functions, UI components, state hooks, wide parameter lists, and growing modules must be decomposed proactively into cohesive single-purpose helpers and hierarchical submodule trees (`REQ-6.1`, [ADR-0020](../../adr/0020-modular-monorepo-and-dnam-proj-container.md)). Every source and test file across the repository must remain strictly below 900 lines, targeting 250–650 lines per leaf unit.

Production source trees organize growing domains into nested child submodules (`module -> sub-aspect -> leaf`) using parent files as clean facades rather than flat sibling dumps. Internal modules import concrete leaf paths directly with zero internal barrel re-export files or wildcard re-exports; explicit named re-exports are restricted strictly to declared public package entrypoints.

Contributors must never play code golf by collapsing multi-statement blocks onto single lines, inserting line-continuation escapes, nesting ternary chains, or stripping comments and docstrings to save lines. Growing directory maps must be delegated across scoped child `AGENTS.md` files rather than compressing routing prose.

| Test Tier | Normative Placement & Scope | Prohibited Anti-Patterns |
| :--- | :--- | :--- |
| **Tier 1 (Leaf Unit & Property)** | Co-located inline with or directly adjacent to the leaf production unit, covering unit invariants, property-based checks, and accessibility | No standalone isolated unit-test files or isolated test subdirectories inside production source trees |
| **Tier 2 (Inter-Unit Integration)** | Co-located at the parent facade or domain folder level to test collaboration across encapsulated child units | Shared test fixtures and builders belong strictly in explicitly named shared support modules, never internal barrels |
| **Tier 3 (Package Integration & E2E)** | Package-level integration, cross-runtime parity, performance profiling, and end-to-end suites named after the subsystem under test | Never create grab-bag coverage test files or duplicate unit-test hierarchies |
| **Tier 4 (Workspace Guardrails)** | Top-level workspace integration and architectural guardrail suites verifying cross-package invariants | Never bypass dependency or modularity boundaries for test convenience |

### 1.3 Compile-Time Constant Evaluation, Derived Layouts & Documented Derivations

`r[repo.code.constant-eval-and-derivations]`

Trigonometric filter poles, rational polynomial coefficients, exact IEEE-754 bit boundaries, packet strides, and 64-byte-aligned memory offsets must be derived via pure compile-time constant evaluators, cumulative symbolic arithmetic, and static compile-time assertions (`REQ-6.1`, `REQ-6.2`). Never hand-enter unexplained numeric literals or manually summed byte offsets.

When a constant originates from offline minimax optimization, global multi-binade floating-point analysis, spectral factorization, or empirical acoustic calibration that cannot execute cheaply at compile time, centralize it in one canonical module. Document the full mathematical derivation, rounding rule, and error bound inside the relevant specification and verify its invariants with static assertions or unit tests.

### 1.4 Universal Black-Box Engine & Solver Purity

`r[repo.code.universal-black-box]`

Real-time DSP paths must operate strictly over general mathematical basis operators parameterized by learned container weights (`REQ-1.6`, `REQ-1.9`, `ref[arch.pillars.universal-black-box]`). Never introduce device-, brand-, or circuit-specific white-box sub-circuits, hardcoded target crossover frequencies, or conditional stage-bypass booleans (`REQ-6.1`).

Model identification, seed initialization, and control-bus conditioning inside capture and training logic must remain 100% data-driven from dry/wet audio pairs and generic control schemas. Never import reference fixture packages into production identification modules, hardcode target fixture identifiers, or branch on control labels.

---

## 2. Two-Tier `AGENTS.md` Progressive-Disclosure Hierarchy

`r[repo.docs.agents-md-budgets]`

`AGENTS.md` files serve strictly as concise navigation maps pointing to canonical specifications and contributing guides, never duplicating mathematical derivations or schemas. Full normative rules belong in specifications written in calm, structured technical-manual prose (up to 650 characters per paragraph, crisp tables, and strict inline code pill discipline as enforced in §4).

| `AGENTS.md` Tier | Target Files | Strict Byte Ceiling | Required Content |
| :--- | :--- | :---: | :--- |
| **Tier 1 (Monorepo Root)** | [root agent router](../../../AGENTS.md) | < 5,000 B | Global orchestration commands, core cross-cutting invariants, and the top-level package discovery table. |
| **Tier 2 (Scoped Module / Suite)** | Package, module, and test-suite `AGENTS.md` files | < 2,500 B | Module-to-file routing table, local task commands, and links to canonical specification files. |

---

## 3. Three-Layer Documentation Architecture & Clean-Room Separation

### 3.1 Implementation-Agnostic Requirements Layer (`docs/requirements/`)

`r[repo.docs.requirements-layer]`

The [system goals and requirements](../../requirements/system_goals_and_requirements.md) define **what** the system must achieve (`REQ-1.1` through `REQ-8.8`, 41 requirements across 8 top-level goals). Every requirement ID must appear in a canonical Markdown table row beginning with ``| **`REQ-X.Y`** |``.

Outside Markdown link targets, requirement prose must remain strictly implementation-agnostic without internal file extensions, wire-schema identifiers, backend API names, or raw relative paths in link labels:

| Prohibited Requirement-Prose Category | Forbidden Tokens Outside Link Targets |
| :--- | :--- |
| **Container & Wire Extensions / Fields** | `.dnam-bin`, `.dnam`, `.dnam-proj`, `.dnam-ckpt`, `.dnam-cache`, `.dnam-seal`, `.dnce`, `max_supported_tier`, `user_metadata`, `royalty_splits`, `allowed_protection_profiles`, `allowed_oem_vendors`, `mono_only`, `JSON header`, `CL-0`, `CL-1`, `CL-2` |
| **Backend & Precision Implementation Tokens** | WebGPU, IEEE-754, float64, `f32`, `f64`, `fp16`, `i32`, `u32`, 0-ULP, 48 kHz MACs, SIMD lane, output head, output-head, Hann-windowed, `../specs/` |

### 3.2 Specification Rules (`r[...]`, `ref[...]`, `verify[...]`), Clean-Room Portability & ADRs

`r[repo.docs.spec-rules-and-adrs]`

Every Markdown document under requirements, specifications, ADRs, and authoring templates must begin at line 1 with a YAML frontmatter block (`--- ... ---`). Normative domain specifications—spanning architecture, quality, DSP, training, containers, and security—must preserve **One-Way Specification Implementation Independence**: those domain specifications stay 100% clean-room portable and never reference source-code directories, crate or package names, language-specific syntax, or language-specific SDK specifications, whereas SDK specifications, ADRs, and source code reference downward into them.

| Construct | Syntax & Location | Normative Invariant |
| :--- | :--- | :--- |
| **Rule Definition** | `r[<domain>.<subdomain>.<rule>]` standalone on its own line immediately under a `##`, `###`, or `####` heading in `docs/specs/` | Globally unique across the repository; matches frontmatter `rule_prefixes`; zero inline numbered-list definitions; zero skipped heading levels; every leaf normative section defines at least one rule. Never used outside canonical specification files and the spec template. |
| **Rule Cross-Reference** | `ref[<domain>.<subdomain>.<rule>]` in ADRs, docs, or source comments | Must resolve to an existing `r[<domain>.<subdomain>.<rule>]` definition in `docs/specs/`. |
| **Test Verification Citation** | `verify[<domain>.<subdomain>.<rule>]` in workspace guardrails, package integration suites, and co-located unit tests | Must resolve to an existing `r[<domain>.<subdomain>.<rule>]` definition in `docs/specs/`. Every spec rule is tracked for test verification coverage. |
| **MADR Decision Files** | `NNNN-kebab-case-title.md` (using `NNNN-superseded-kebab-case-title.md` when `status: superseded`; `id: ADR-NNNN`, `status: proposed \| accepted \| superseded`) in `docs/adr/` | Numbered sequentially starting at 0001 with descriptive design-alternative titles (never generic lettered or numbered shorthand labels). Superseded ADRs set `superseded_by: ADR-NNNN` pointing to an active ADR (never cited by specs/requirements); amended accepted ADRs record a terse timestamped `## Amendment Appendix` stating why, leaving detailed diffs to version history and specs. |
| **Authoring Templates** | [requirement_template.md](../../templates/requirement_template.md), [spec_template.md](../../templates/spec_template.md), [adr_template.md](../../templates/adr_template.md) | Canonical templates must exist in `docs/templates/` with valid YAML frontmatter. |
| **Zero Spec-Text Parsing in Tests** | All test suites outside automated documentation guardrail runners | Functional unit, integration, and reference tests must never read or parse Markdown files from `docs/`; they reference specifications solely via `verify[...]` citations. |

---

## 4. Markdown, Mermaid & GitHub MathJax Rendering Rules

### 4.1 Zero Inline LaTeX Math in Markdown Tables

`r[repo.docs.markdown-table-math]`

No Markdown table row (any line starting with `|`) in any `.md` file across the repository may contain a raw `$` character outside backtick code spans. Inside Markdown table cells, format symbolic expressions using Unicode mathematical symbols or concise code spans, and place full LaTeX equations in `$$...$$` display math blocks above or below the table.

### 4.2 GitHub CommonMark + MathJax LaTeX, Mermaid & Structured-Prose Rules

`r[repo.docs.github-latex-and-mermaid]`

All math blocks, Mermaid diagrams, and canonical specifications must satisfy GitHub's CommonMark + MathJax parser and structured technical-manual formatting checks:

| Rendering & Formatting Rule | Normative Syntax & Parser Requirement |
| :--- | :--- |
| **Delimiter & Brace Balance** | Every `$...$` and `$$...$$` block must have balanced `{` and `}` braces and whitespace-flanked inline `$...$` delimiters, never abutting parentheses, brackets, or hyphens such as `($...$)` or `$x$-norm`. |
| **CommonMark Escape Guard** | Zero escaped punctuation inside math blocks (`\_`, `\;`, `\,`, `\!`, `\{`, `\}`, `\|`, `\%`, `\\`). |
| **Macro & Mermaid Hygiene** | Zero disallowed macros (`\operatorname`, `\multicolumn`, `\texttt`, `\text{--`), zero underscores inside `\text{...}` or `\mathrm{...}`, quoted Mermaid labels containing special characters, and zero semicolons inside `sequenceDiagram` or state diagrams. |
| **Structured Prose Paragraphs** | Each prose paragraph in canonical specs must stay within 650 normalized characters and 8 inline code spans; every spec must contain at least 6 prose paragraphs, with `proseParagraphCount + tableDataRowCount > listItemCount`. |
| **Short Parallel Lists Only** | List items must stay within 240 normalized characters and 5 inline code spans; multi-sentence explanations belong in prose paragraphs or Markdown tables. |
| **Zero Nested Parentheticals** | Avoid `(outer (inner))` nesting in prose and list items outside inline code, math, or URLs. |
| **Strict Inline Code Pill Discipline** | Never wrap plain numbers, dimensions, byte sizes, frequencies, durations, multipliers, versions, bit widths, or domain/tier words (such as Lite, Full, Pro, Ultra, SONA, VarPro, WebGPU, Kautz, FIR, ZOH, LTAS, SHA-256, or Ed25519) in backticks; reserve backticks for concrete code, wire, schema, FourCC, and rule identifiers. |

### 4.3 Relative Markdown Link & Heading Anchor Integrity

`r[repo.docs.relative-links-and-anchors]`

Every relative Markdown link outside fenced code blocks across workspace documentation must resolve to an existing file or directory, and any `#section-anchor` fragment must match a GitHub-slugified heading in the target file. Inside `docs/`, link labels (`[label](target)`) must never start with `../` or `./`; always write human-readable titles or basenames.
