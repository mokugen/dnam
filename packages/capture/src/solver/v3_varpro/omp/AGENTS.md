# `omp/` — Kautz Pole OMP & Coordinate Refinement Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/dsp/op_m_modal_kautz.md`](../../../../../../docs/specs/dsp/op_m_modal_kautz.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../omp.rs`](../omp.rs) | Stereo pooled Modified Gram-Schmidt Kautz OMP selector (`10` real + `64x6` complex poles). | [`op_m_modal_kautz.md`](../../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`refine.rs`](refine.rs) | Local `5 x 3` `(theta, r)` coordinate micro-refinement for OMP-selected poles. | [`op_m_modal_kautz.md`](../../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
