# `dnam-train::linalg` — Linear Algebra, VarPro Schur & L-BFGS Map

Scoped discovery guide for [`src/linalg.rs`](../linalg.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]`, `binary64` solver precision, zero external linear-algebra crates, and canonical `uplo='U'` storage.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`cholesky.rs`](cholesky.rs) | `uplo='U'` Cholesky factor/solve, adaptive ridge escalation & norm-bounded solver | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md), [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`varpro.rs`](varpro.rs) | Chunked Ozaki / `Wide192Accum` normal-equation assembly & nested 4-tier Schur-complement prefix solver | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`spectral_kkt.rs`](spectral_kkt.rs) | 1/12-octave `EMR` `MM-Cholesky` & active-set 33-band `LTAS` Newton-KKT envelope solver | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`coords.rs`](coords.rs) | `ParamBoundSpec` bijections, `ScaledCoordSpace` codecs & shared SIMD `dot_simd` | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`adamw.rs`](adamw.rs) | Decoupled `AdamWState`, cosine warmup LR & Arena-scoped deterministic `PCGrad` | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`ssbroyden.rs`](ssbroyden.rs), [`lbfgs.rs`](lbfgs.rs) | Strong-Wolfe `L-SSBroyden`/`SsBroyden`, `TwoLoopLbfgs`, `WhitenedLbfgs` & GPU bridge | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
