# `dnam-train::losses` — Differentiable Perceptual Losses & Tier Certification Map

Scoped discovery guide for [`src/losses.rs`](../losses.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]` with internal pure-Rust radix-2 FFT, exact FD-verified reverse-time adjoints, and zero-slack adjacent-tier non-regression checking.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../losses.rs`](../losses.rs) | `CompositeLossEvaluator`, `certify_tier_audio`, `is_tier_pair_monotone` & `check_adjacent_tier_monotone` | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`fft.rs`](fft.rs) | Pure-Rust paired real radix-2 FFT, `build_hann_window` & `compute_welch_seg_starts` | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`bezout.rs`](bezout.rs), [`bezout/metrics.rs`](bezout/metrics.rs) | Degree-7 Bézout 3-band `bESR`, `SegESR` (20 ms), noise qualification & joint interval bounds | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`mr_iir.rs`](mr_iir.rs), [`ltas.rs`](ltas.rs) | 31 ms–250 ms multi-resolution pre-warped `MR-IIR`, 33-band Welch `LTAS` & 21.6–24 kHz stopband hinge | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`phase_ot.rs`](phase_ot.rs), [`spectral.rs`](spectral.rs) | Subband `SpecOT` CDF loss, energy-weighted `IF+GD` phase loss & `K=8` analytic Gabor filter bank | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
