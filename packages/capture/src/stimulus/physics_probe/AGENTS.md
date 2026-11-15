# `packages/capture/src/stimulus/physics_probe/` — Tail & Echo Submodules

Read [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Spec |
| :--- | :--- | :--- |
| [`../physics_probe.rs`](../physics_probe.rs) | Blind `StarCenter` physics extractor (`ExtractedPhysics`, `phys_*`). | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`echo_peaks.rs`](echo_peaks.rs) | Motif cross-correlation, whitened lag peak detection, stereo echo fit. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`tail.rs`](tail.rs) | `RT60` regression, BBD darkening, shimmer, ping-pong asymmetry, LFO rate. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
