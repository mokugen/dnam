# `dnam-train::solver` — 5-Pass Universal Black-Box SONA Solver Map

Scoped discovery guide for [`src/solver.rs`](../solver.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]`, `zerocopy 0.8` wire/telemetry serialization, zero fixture/control-name heuristics, and `BestUntilDiminishingReturns` stopping (`|ΔbESR| < 0.01 dB/epoch`).

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs), [`model.rs`](model.rs), [`quads.rs`](model/quads.rs), [`clamp_and_quant.rs`](model/clamp_and_quant.rs), [`cascade.rs`](model/cascade.rs) | `solve_sona_model`, `SonaModelF64`, `snap_model_fp16`, `Block 0` PHS `svf_k_nl` & `2×` cascade | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass1_probe.rs`](pass1_probe.rs), [`pass1_probe/AGENTS.md`](pass1_probe/AGENTS.md) | Dry `Op-P` probe, FIR, Kaufman-VarPro + SM + Kautz OMP, Wiener reflection deconvolution & `Op-T` seeding | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass2_opt.rs`](pass2_opt.rs), [`pass2_opt/AGENTS.md`](pass2_opt/AGENTS.md) | Phase 2A–2D hybrid `AdamW`+`VarPro` → `L-SSBroyden` → `u`-space QAT, Matryoshka `PCGrad`, `1×`/`2×` adjoints | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass3_varpro.rs`](pass3_varpro.rs), [`pass3_varpro/AGENTS.md`](pass3_varpro/AGENTS.md) | True-stereo `1×`/`2×` Schur VarPro, `ro_pole_48k` sweep, `Welch-33` SQP-KKT, `STAT` 256-D contraction & `fp16` snap | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass5_emit/AGENTS.md`](pass5_emit/AGENTS.md), [`progress.rs`](progress.rs) | `Arena0..3` packing, decode, `<= 0.05 LU` trim, monotone repair & `DNPR` seqlock | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md), [`iterative_resume_and_caching.md`](../../../../docs/specs/training/iterative_resume_and_caching.md) |
