# `src/stages/sona_core` — `SONA` `v3.0` L1D Hot Slab & Runtime Execution

Implements the unified `SONA` `64 B`-aligned hot slab ([`SonaHotSlab`](slab.rs)) and allocation-pinned runtime ([`SonaInstance`](../sona_core.rs)):

| Module | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../sona_core.rs`](../sona_core.rs) | `SonaInstance<B, A>`, tier reconfiguration & infallible block entrypoints | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`frontend.rs`](frontend.rs) | `ZeroLagOversampler2x` / `SonaStereoOversampler2x` halfband `2x` path | [`router_and_crossbar_topology.md`](../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`model.rs`](model.rs) | `SonaMatryoshkaModel`, `SonaColdTierModel`, `SonaTierReadout` & group/tile cold params | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`slab.rs`](slab.rs) ([`layout.rs`](slab/layout.rs), [`lifecycle.rs`](slab/lifecycle.rs), [`ctrl_apply.rs`](slab/ctrl_apply.rs)) | Prefix-ordered `SonaHotSlab<B>` (`ActiveHot <= 4,608..28,672 B`), ZOH, tier switch & `3 kHz` folding | [`op_n_paralru_and_polyadaa.md`](../../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`tick16.rs`](tick16.rs) ([`crossbar_step.rs`](tick16/crossbar_step.rs)) | Const-generic `process_tick16_const`, `Op-P` (`5 Hz` DC + `H1` + `T1`) & `4x Q0.32` phasors | [`op_n_paralru_and_polyadaa.md`](../../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for L1D budget invariants (`TOTAL_LITE_STATE_RAM_BYTES <= 4,608`).
