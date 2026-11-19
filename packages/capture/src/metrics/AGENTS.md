# `packages/capture/src/metrics/` — 4-Metric Perceptual Certification Guide

Implements the 4-metric psychoacoustic certification suite (`bESR`, 5-scale 32-probe `MR-IIR`, 33-band `1/3`-octave `LTAS`, `SegESR`) and 4-tier non-regression verification (`Lite`, `Full`, `Pro`, `Ultra`).

## Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../metrics.rs`](../metrics.rs), [`bezout.rs`](bezout.rs) | Degree-7 Bézout 3-band sum-denominator `bESR`, `const fn` Bézout constants, noise-limited qualification, and interval audit | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`repeatability.rs`](repeatability.rs) | Label-free `TargetRepeatabilityProfile` empirical probe variance & diffuse tail floor | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`companion.rs`](companion.rs) | Companion metric facade (`ONE_DB_POWER_FACTOR`, `SIX_DB_POWER_FACTOR`) and 3-band envelope `SegESR` | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`companion/mr_iir.rs`](companion/mr_iir.rs) | 5-scale 32-probe `MR-IIR` evaluator and target-invariant resonator cache (`MrIirTargetCache`) | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`companion/ltas.rs`](companion/ltas.rs) | 33-band `1/3`-octave Welch `LTAS` (`0.75 / 0.35 / 0.15 / 0.10 dB` tier limits) and paired-FFT cache | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`certification.rs`](certification.rs), [`certification/universal.rs`](certification/universal.rs) | Stem classification, `PerceptualCertificationSummary`, `certify_4metric_universal`, `certify_holdout_universal`, and monotonicity | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`certification/scope.rs`](certification/scope.rs) | Legacy archetype compatibility adapter (`certify_holdout_for_family`) | [`psychoacoustic_quality_and_tiers.md`](../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
