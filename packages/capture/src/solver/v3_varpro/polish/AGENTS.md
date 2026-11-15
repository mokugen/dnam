# `polish/` — `v3` Exact `f32` Readout Polish & Subspace Repair Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../polish.rs`](../polish.rs) | Exact `f32` readout polish coordinator & monotonicity entry point. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`extract.rs`](extract.rs) | Exact `f32` decimated shared-core trajectory extraction. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
| [`prescreen.rs`](prescreen.rs) | Cold-model perceptual prescreen & `pre_cab` sharing check. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`subspace.rs`](subspace.rs) | Whitened BESR hinge + Welch-33 LTAS subspace L-BFGS polish. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`repair.rs`](repair.rs) | Candidate scoring, monotone blends & corridor repair. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
