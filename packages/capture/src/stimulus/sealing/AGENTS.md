# `packages/capture/src/stimulus/sealing/` — Sealing Submodule Map

Read [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Spec |
| :--- | :--- | :--- |
| [`../sealing.rs`](../sealing.rs) | Causal alignment, shared `StemSealRequest`, `INPC`/`STMC` vault sealing facade. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`controls.rs`](controls.rs) | Control bounds/defaults, metadata maps, `TrajectoryInterpSpec` trajectory interpolation. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`freeze_and_pad.rs`](freeze_and_pad.rs) | Sample-rate normalization, freeze-transition stem synthesis, `recapture_clipped_stem`. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
