# `dnam-capture` Stage A Solver Submodules (`packages/capture/src/solver/stage_a/`)

Routing map for Stage A linear, deconvolved FIR, reflection, and physical extension identification (`REQ-1.6`). Parent facade: [`../stage_a.rs`](../stage_a.rs).

| Module | Responsibility | Canonical Specs |
| :--- | :--- | :--- |
| [`ext.rs`](ext.rs) | Blind Stage I–IV physical extension lifting and SVF surrogate estimation | [`four_stage_pipeline.md`](../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`id.rs`](id.rs) | Steiglitz-McBride IIR, early-FIR-64, and deconvolved Stage III system ID | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`refl.rs`](refl.rs) | Stereo Stage IV reflection and reverberation delay-anchor extraction | [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
