# `ltas_adj/` — Welch LTAS Objective & Radix-2/4 FFT Adjoint Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../ltas_adj.rs`](../ltas_adj.rs) | 33-band Welch LTAS objective & spectral minimax adjoints. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`fft.rs`](fft.rs), [`fft/radix4.rs`](fft/radix4.rs), [`nyquist.rs`](nyquist.rs) | Cached real-to-complex FFT plans, radix-4 butterflies & Nyquist stopband hinge. | [`psychoacoustic_quality_and_tiers.md`](../../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
