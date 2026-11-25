# `dnam-train::solver::pass3_varpro` — 4-Tier Nested VarPro & Spectral Refinement

Scoped discovery guide for [`../pass3_varpro.rs`](../pass3_varpro.rs) (parent [`../AGENTS.md`](../AGENTS.md)). Strictly deterministic CPU/GPU double-precision Schur readouts.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`features.rs`](features.rs), [`features/helpers.rs`](features/helpers.rs), [`tier_opt_sol.rs`](tier_opt_sol.rs), [`ro_sweep.rs`](ro_sweep.rs) | `1×`/`2×` design-matrix extraction, tier-specific readout optimization, and stereo `ro_pole_48k` 1-D sweep | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`spectral_refine.rs`](spectral_refine.rs), [`ridge_solve.rs`](spectral_refine/ridge_solve.rs), [`welch_kkt.rs`](spectral_refine/welch_kkt.rs) | Outlier-trimmed ridge, EMR `MM-Cholesky`, full-capture `Welch-33` SQP-KKT adjoint polish, and `l1`/BIBO rail guards | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md), [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`gram_snap.rs`](gram_snap.rs), [`pre_conv_proj.rs`](pre_conv_proj.rs), [`stat_and_patch.rs`](stat_and_patch.rs), [`tier_gate.rs`](tier_gate.rs), [`linear_gate.rs`](linear_gate.rs) | Gram `fp16` coordinate snapping, pre-convolver Schur projection, `STAT` 256-D contraction, tier inheritance & parsimony gates | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
