# `dnam-train::losses` — Differentiable Perceptual Losses & Tier Certification Map

Scoped discovery guide for [`src/losses.rs`](../losses.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]` with internal pure-Rust radix-2 FFT, exact FD-verified reverse-time adjoints, and zero-slack adjacent-tier non-regression checking.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../losses.rs`](../losses.rs), [`certification.rs`](certification.rs) | `CompositeLossEvaluator`, `CompositeLossWeights::homotopy_at`, `certify_tier_audio` & monotonicity | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`fft.rs`](fft.rs) | Pure-Rust paired real radix-2 FFT, `build_hann_window` & `compute_welch_seg_starts` | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`noise_floor.rs`](noise_floor.rs) | Differentiable `snr_weight` and C∞ softplus excess-residual gating (`tau = max(0.05 F, 1e-12)`) | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`bezout.rs`](bezout.rs), [`bezout/metrics.rs`](bezout/metrics.rs) | Degree-7 Bézout 3-band `bESR`, cached `20 ms` `SegESR`, noise qualification & joint interval bounds | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`mr_iir.rs`](mr_iir.rs), [`ltas.rs`](ltas.rs), [`mode_eval.rs`](ltas/mode_eval.rs) | 5-scale (`256..4096`) 32-probe `MR-IIR`, fused 33-band Welch `LTAS` & stopband hinge | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`phase_ot.rs`](phase_ot.rs), [`spectral.rs`](spectral.rs) | 1-D `SpecOT` CDF loss, `IF+GD` phase coherence & 33-band constant-Q Gabor wavelet bank | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
