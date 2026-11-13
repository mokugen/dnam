# `packages/capture/src/stimulus/` — Closed-Loop Excitation & Dual-Mode Pipeline Guide

Read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Spec |
| :--- | :--- | :--- |
| [`../stimulus.rs`](../stimulus.rs) | `AdaptiveStimulusController` (`next_frame`), telemetry, & `INPC`/`STMC` sealing. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`fft.rs`](fft.rs) | Radix-2 `f64` FFT/IFFT, disjoint-comb scheduler, Schroeder phases, cyclic frames. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`waveforms.rs`](waveforms.rs) | `A1..A6` primitives, `Theorem 4.7`, `Theorem 4.8` Bloom telemetry, encoders. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`safeguards.rs`](safeguards.rs) | `Laws 0.A–0.F` (`LTV`, echo hold, squeal, dropout, `-1.5 dBFS`, `OC-2`, clip). | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`laws.rs`](laws.rs) | `Laws 1–3`, `Theorems 4.5–4.6` (multi-port, endpoints, Hadamard), 4-Gate stop. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`physics_probe.rs`](physics_probe.rs) | Blind `StarCenter` physics extractor (`ExtractedPhysics`, `phys_*`). | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`physics_probe_tail.rs`](physics_probe_tail.rs) | Tail echo, RT60, BBD, & LFO helpers. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`pipeline.rs`](pipeline.rs) | Mode 1 streaming (`train_adaptive_target`), endpoint probing, `FreezeLatch`. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`null_test.rs`](null_test.rs) | `1.0s` Out-of-Sample Live Null-Test and sufficient-statistic `STAT` auto-patch. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`sealing.rs`](sealing.rs) | `INPC`/`STMC` coreset sealing, alignment, `FreezeLatch`, clipped stem recapture. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`batch.rs`](batch.rs) | Mode 2 sans-IO batch (`generate_stimulus_batch*`, `solve_from_responses`). | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
