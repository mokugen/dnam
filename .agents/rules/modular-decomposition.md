---
trigger: always_on
---

# Modular Decomposition, `AGENTS.md` Delegation & Co-Located Unit Tests

These architectural rules are mandatory for every code and documentation edit in DNAM:

1. **Decompose Large Functions, Types & Modules into Hierarchical Submodule Trees (`ref[repo.code.modular-decomposition-and-tests]`):**
   - **Break Up Large Monolithic Functions & Wide Signatures:** Proactively decompose large monolithic functions, multi-stage procedural pipelines, and deeply nested loop/match bodies (e.g., functions growing past `~80–120` lines or needing `#[allow(clippy::too_many_arguments)]` / `#[allow(clippy::cognitive_complexity)]`) into small, named, single-purpose helper functions and cohesive domain/context structs with clear invariants that can be reasoned about and unit-tested directly.
   - **Build Hierarchical Submodule Trees, Not Just Flat Siblings:** When a module grows beyond a single cohesive responsibility or approaches `~600–700` lines (well below `< 900` LOC ceilings), organize its **production code** into **nested child submodules** (`src/<module>.rs` $\to$ `src/<module>/<sub_aspect>.rs` $\to$ `src/<module>/<sub_aspect>/<leaf>.rs`, zero `mod.rs`) as well as sibling modules where domain-appropriate. Avoid dumping flat `foo_bar.rs`, `foo_baz.rs` files into crowded parent directories; use parent modules as clean facades encapsulating private child submodules.
   - **Never Play Code Golf:** Never join multi-instruction inline assembly or statements with `\`-newline continuations (`\n\`), never collapse multi-line expressions onto one line, and never strip comments or docstrings to save lines.

2. **4-Tier Layered Testing Hierarchy (`ref[repo.code.modular-decomposition-and-tests]`):**
   - **Leaf Unit Tests (`src/<name>/<child>.rs`, `src/<name>/<child>/<leaf>.rs`):** Put unit tests and `proptest!` suites inside inline `#[cfg(test)] mod tests { ... }` at the bottom of the exact `.rs` production module they test (no blank line or attribute between `#[cfg(test)]` and `mod tests {`). If adding `#[cfg(test)] mod tests` pushes a module toward `~700–900` lines, **split the production code first** into clean submodules so each unit holds both its logic and its co-located unit tests comfortably (`~350–650` lines total).
   - **Parent Inter-Unit Integration Tests (`src/<name>.rs`):** Smaller integration tests are allowed inside inline `#[cfg(test)] mod tests { ... }` in parent/facade `"name.rs"` files (`src/stimulus.rs`, `src/align.rs`, `src/solver/v3_varpro.rs`, etc.) to test integration between the child units they encapsulate.
   - **Crate Integration Tests (`packages/<pkg>/tests/<domain>.rs`):** Keep package `tests/*.rs` strictly for cross-module / subsystem integration suites named after the domain or contract under test (never grab-bag `packages/*/tests/*_coverage.rs` files).
   - **Top-Level Workspace Tests (`tests/`):** Top-level `tests/` is the highest level of integration testing for monorepo-wide, cross-package, and guardrail integration suites (`moon check tests`).
   - **No Isolated Unit-Test Files in `src/`/`rust/`:** **Never** create isolated unit-test `.rs` files inside `src/` or `rust/` (`tests.rs`, `*_tests.rs`, `test_*.rs`) or external `mod tests;` / `mod *_tests;` file declarations. Isolated test-only `.rs` modules are allowed **strictly for shared test helpers and builders** (`test_support.rs`, `test_utils.rs`, `testing.rs`, or helper submodules under `tests/<suite>/*.rs`).

3. **Never Wordsmith `AGENTS.md` to Fit Byte Budgets (`CONTRIBUTING.md` §5.1.3):**
   - When an `AGENTS.md` approaches `2,500 B`, create a scoped subdirectory `AGENTS.md` (`< 2,500 B`) and link it from the parent `AGENTS.md`, or offload non-routing prose to `packages/<pkg>/CONTRIBUTING.md` / `docs/specs/`. Never compress wording in `AGENTS.md` to squeeze rows.
