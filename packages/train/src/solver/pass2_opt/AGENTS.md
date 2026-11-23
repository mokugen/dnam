# `dnam-train::solver::pass2_opt` — Phase 2A–2D Hybrid Optimizer Map

Scoped discovery guide for [`../pass2_opt.rs`](../pass2_opt.rs) (parent [`../AGENTS.md`](../AGENTS.md)). Strictly `#[forbid(unsafe_code)]`, `< 900` LOC per file, zero `mod.rs`, zero fixture heuristics.

| Module | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../pass2_opt.rs`](../pass2_opt.rs), [`hybrid.rs`](hybrid.rs), [`windows.rs`](hybrid/windows.rs) | `Pass2OptConfig`, Phase 2A–2D hybrid `AdamW`+`VarPro` $\to$ `L-SSBroyden` $\to$ QAT driver & windowing | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`prescreen.rs`](prescreen.rs), [`varpro_sync.rs`](varpro_sync.rs), [`basis.rs`](varpro_sync/basis.rs) | Sobol seed gen, Top-K diverse basin selection, onset-decoupled `uplo='U'` VarPro refresh | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`matryoshka.rs`](matryoshka.rs), [`qat_polish.rs`](qat_polish.rs) | `4 -> 6 -> 8 -> 10` growth, `ln(L_t+1e-8)` + `PCGrad`, `u`-space STE `fp16` QAT | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`theta.rs`](theta.rs), [`domains.rs`](theta/domains.rs), [`arena.rs`](theta/domains/arena.rs) | Full-subspace descriptors, `Arena0..3`/Readout masks, `1/N` ZOH scaling | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`eval.rs`](eval.rs), [`adjoints.rs`](adjoints.rs), [`causal_loop.rs`](causal_loop.rs), [`crossbar.rs`](crossbar.rs), [`op_t_quad.rs`](op_t_quad.rs), [`sensory.rs`](sensory.rs) | Exact `1×`/`2×` forward/BPTT adjoints across `SensoryQuad`, `Op-P`, `Op-N[Pre]`, `CoreQuad`, `Kautz`, `Op-T` | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
