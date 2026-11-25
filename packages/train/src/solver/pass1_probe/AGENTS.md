# `dnam-train::solver::pass1_probe` — Physics Probe & Modal Initialization

Scoped discovery guide for [`../pass1_probe.rs`](../pass1_probe.rs) (parent [`../AGENTS.md`](../AGENTS.md)). Zero fixture/device/control-name heuristics (`REQ-1.6`, `REQ-1.9`).

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`kautz_fit.rs`](kautz_fit.rs), [`kautz_varpro.rs`](kautz_varpro.rs), [`steiglitz_mcbride.rs`](steiglitz_mcbride.rs) | Audible/physical Kautz bounds, DC-invariant `r_max` estimation, 2-stage Kaufman VarPro + Steiglitz–McBride + Matryoshka OMP | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md), [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`series_reflections.rs`](series_reflections.rs), [`op_t_seed.rs`](op_t_seed.rs), [`peaks.rs`](op_t_seed/peaks.rs), [`reflections.rs`](op_t_seed/reflections.rs) | Alternating Wiener reflection deconvolution, `Op-T` comb/peak seeding, passive feed-forward `Quad 0` | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md), [`op_t_transport.md`](../../../../../docs/specs/dsp/op_t_transport.md) |
| [`fir_toeplitz.rs`](fir_toeplitz.rs), [`telemetry_init.rs`](telemetry_init.rs) | Early `FIR` onset Toeplitz init and front-end / `Op-N[Pre]` / `rect_sign` telemetry seeding | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
