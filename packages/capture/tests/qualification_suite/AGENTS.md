# `packages/capture/tests/qualification_suite/` — 3-Level Phased Harness

Read [`../AGENTS.md`](../AGENTS.md). Zero `unsafe_code`, `< 900` lines/file, and zero fixture-specific heuristics (`M25`).

| File | Responsibility |
| :--- | :--- |
| [`../qualification_suite.rs`](../qualification_suite.rs) | Suite entrypoint wiring shared harness & non-ignored self-tests. |
| [`adapter.rs`](adapter.rs) | Generic [`FixtureDistillationAdapter`](adapter.rs), manifests, and empirical repeatability probing. |
| [`excitations.rs`](excitations.rs) | Unseen `48 kHz` broadband/`A5` holdouts and off-knot interior control generators. |
| [`harness.rs`](harness.rs) | 3-level (`L1`/`L2`/`L3`) pipeline orchestrator, alignment extractor, and zero-ULP checks. |
| [`harness/eval.rs`](harness/eval.rs) | 4-tier VM renderer, 4-metric evaluator (`DrySilentGapMask` `L1`), and assertions. |
| [`self_tests.rs`](self_tests.rs) | Synthetic `BlackBoxModel` positive & negative falsification checks (CPU + `wgpu`). |
