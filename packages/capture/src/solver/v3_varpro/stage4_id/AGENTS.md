# `stage4_id/` — Black-Box Stage IV Reflection & FDN Solver Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/training/webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../stage0_id.rs`](../stage0_id.rs) | Black-box `Stage 0` (`Op-P`) sub-octave divider & Gauss-Newton solver. | [`router_and_crossbar_topology.md`](../../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`../stage4_id.rs`](../stage4_id.rs) | Stage IV bypass gate, cert caches & `room_lti`/`incoherent_rig` dispatch. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`room_lti.rs`](room_lti.rs), [`room_lti/wiener_echo.rs`](room_lti/wiener_echo.rs) | Welch Wiener IR (`R^2_lin`), cabinet-deconv 2-tap room OMP (`64..=960`), causal FDN deconv & continuous Kautz/modal refit. | [`webgpu_and_varpro_solver.md`](../../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`incoherent_rig.rs`](incoherent_rig.rs), [`incoherent_rig/`](incoherent_rig/AGENTS.md) | Modulated delay, BBD comb & diffuse FDN Stage IV solver with `P x P` Welch-33 EQ & 4-tier cascade. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
