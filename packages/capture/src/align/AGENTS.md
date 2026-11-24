# `packages/capture/src/align/` — 4-Regime Alignment & Loopback Calibration Guide

Read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) before editing.

| File | Core Responsibilities | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`../align.rs`](../align.rs), [`wav_codec.rs`](wav_codec.rs), [`wav_wire.rs`](wav_wire.rs) | 5-bit-depth `zerocopy` WAV PCM codec (`Int12`..`Float32`), `<f4` SHA-256, and coarse parabolic lag. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`regimes.rs`](regimes.rs), [`regimes/`](regimes/hybrid_parabolic.rs) | 4-regime hybrid alignment, Jacobi `H1` Hilbert envelopes ([`hilbert_env.rs`](regimes/hilbert_env.rs)), coarse-guarded parabolic (`hybrid_parabolic.rs`), `Law 0.A`-gated PPM (`ppm_drift.rs`), and `64`-tap Kaiser-sinc shift. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
| [`calibration.rs`](calibration.rs) | Phase-0 `0.5s` `CALB` loopback calibration, `400 ms` mic-pad ratio, `< 0.05 dB` null homing, `<= 0.15 dB` thermal gate, and live decay-tail monitor. | [`capture_and_alignment.md`](../../../../docs/specs/training/capture_and_alignment.md) |
