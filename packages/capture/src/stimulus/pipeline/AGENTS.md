# `pipeline/` — Mode 1 Streaming Fullband & Stem Capture Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) and [`../../../../../docs/specs/training/capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../pipeline.rs`](../pipeline.rs), [`sweeps.rs`](sweeps.rs) | Mode 1 closed-loop orchestrator (`train_adaptive_target`), `Mode A`/`B`/`Switch`/Hadamard dispatch. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`fullband.rs`](fullband.rs) + `fullband/`, [`causal_probe.rs`](causal_probe.rs) | Full-bandwidth render/seal bridge, `StarCenter` ([`star_schoukens.rs`](fullband/star_schoukens.rs)), causal pre-ring guard ([`causal_guard.rs`](fullband/causal_guard.rs)), & repeatability probe. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`schoukens.rs`](schoukens.rs) + `schoukens/` | Two-Tier `Schoukens–Pintelon` Regimes 1–4 ([`regimes.rs`](schoukens/regimes.rs)), `BLA`/even-odd/`Law 0.A` ([`analysis.rs`](schoukens/analysis.rs)), Schroeder `EDC` ([`edc.rs`](schoukens/edc.rs)), & `SPCM` condenser. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`stems.rs`](stems.rs) | Endpoint probing and closed-loop stem capture sealing. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
