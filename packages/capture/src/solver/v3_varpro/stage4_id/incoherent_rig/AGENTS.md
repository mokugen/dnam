# `incoherent_rig/` — Stage IV FDN & Welch-33 EQ Solver Map

See [`../../../../../CONTRIBUTING.md`](../../../../../CONTRIBUTING.md) and [`../../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md`](../../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../incoherent_rig.rs`](../incoherent_rig.rs) | Top-level modulated delay/comb/diffuse FDN Stage IV coordinator. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`screen.rs`](screen.rs), [`cascade.rs`](cascade.rs), [`cascade/fallback.rs`](cascade/fallback.rs) | Finalist linear/comb FDN screening & top-down 4-tier cascade. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`lfo_detect.rs`](lfo_detect.rs), [`comb_detect.rs`](comb_detect.rs) | Closed-form LFO demodulation, Wiener comb ID & RMS calibration. | [`four_stage_pipeline.md`](../../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`harmonic_cal.rs`](harmonic_cal.rs), [`s2_modal.rs`](s2_modal.rs), [`cepstral_eq.rs`](cepstral_eq.rs) | Cross-rate harmonic track normalization, `Lite`/`Full` `1x` resonators & cepstral init. | [`four_stage_pipeline.md`](../../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`welch_eq.rs`](welch_eq.rs), [`welch_eq/basis.rs`](welch_eq/basis.rs), [`welch_eq/coupled.rs`](welch_eq/coupled.rs), [`whitened_lbfgs.rs`](whitened_lbfgs.rs) | Matrix-free batched basis & Cholesky-whitened L-BFGS Stage III EQ. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`wave_hinge.rs`](wave_hinge.rs), [`mr_iir_adj.rs`](mr_iir_adj.rs) | Causal Bézout/SegESR & MR-IIR monotonicity hinge adjoints. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
