# `dnam-train::cpu` — Deterministic CPU Reference Runners Map

Scoped discovery guide for [`src/cpu.rs`](../cpu.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]` and `0-ULP` parity against `dnam-kernels` and AOT WGSL training shaders across all 6 compute entrypoints.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`core_and_adjoint.rs`](core_and_adjoint.rs), [`streamed.rs`](core_and_adjoint/streamed.rs) | `ResidentBatched` / `Streamed` `fused_core_scan_256` & 3-pass-equivalent `25/20` `adjoint_scan_256` | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`adjoint_math.rs`](core_and_adjoint/adjoint_math.rs), [`phs_svf.rs`](core_and_adjoint/adjoint_math/phs_svf.rs), [`adaa_derivs.rs`](core_and_adjoint/adjoint_math/adaa_derivs.rs) | Exact reverse-time Block 0 PHS `TPT-SVF`, `ParaLRU`, Selective-Duhem, sag, wavefolder & `PolyADAA1` adjoints | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`gabor_and_prescreen.rs`](gabor_and_prescreen.rs) | `prescreen_population` Sobol runner & `gabor_bezout_grad` Mode A/B objective + adjoints | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`ozaki_and_opt.rs`](ozaki_and_opt.rs) | Integer-exact 3×8-bit `ozaki_gram_rhs_i32`, `f64` reconstruction & `inner_opt_step` (`m <= 8`) | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
