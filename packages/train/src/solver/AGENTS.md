# `dnam-train::solver` — 5-Pass Universal Black-Box SONA Solver Map

Scoped discovery guide for [`src/solver.rs`](../solver.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]`, `zerocopy 0.8` wire/telemetry serialization, zero fixture/control-name heuristics, and `BestUntilDiminishingReturns` stopping (`|ΔbESR| < 0.01 dB/epoch`).

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../solver.rs`](../solver.rs), [`model.rs`](model.rs), [`cascade.rs`](model/cascade.rs) | `solve_sona_model`, `SonaTrainDataset`, `ControlAudioStem`, `SonaModelF64`, `TierReadoutPrefixF64`, ZOH & `2×` cascade | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass1_probe.rs`](pass1_probe.rs) | Dry CPU `Op-P` probe, regularized FIR, iterative Steiglitz–McBride `(B, A)` + stereo Kautz OMP & `Op-T` comb ID | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass2_opt.rs`](pass2_opt.rs), [`eval.rs`](pass2_opt/eval.rs), [`adjoints.rs`](pass2_opt/adjoints.rs), [`op_t_quad.rs`](pass2_opt/op_t_quad.rs) | 16-slot Sobol prescreening, L-BFGS trajectory opt & exact `Op-N[Pre]` / stereo Kautz / 4-lane `Op-T` adjoints | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass3_varpro.rs`](pass3_varpro.rs), [`features.rs`](pass3_varpro/features.rs) | True-stereo `1×`/`2×` Schur VarPro readout, `ro_pole_48k` sweep, EMR MM-Cholesky & 33-band `LTAS` KKT | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pass5_emit.rs`](pass5_emit.rs), [`pack.rs`](pass5_emit/pack.rs), [`certify.rs`](pass5_emit/certify.rs), [`progress.rs`](progress.rs) | `Arena0..3` & `PrefixOverride` packing, `<= 0.05 LU` trim, stereo 4-metric check, `STAT` digest & `DNPR` seqlock | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md), [`iterative_resume_and_caching.md`](../../../../docs/specs/training/iterative_resume_and_caching.md) |
