# `pipeline/` — Mode 1 Streaming Fullband & Stem Capture Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) and [`../../../../../docs/specs/training/capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../pipeline.rs`](../pipeline.rs) | Mode 1 closed-loop orchestrator (`train_adaptive_target`), `Mode A`/`B`/`Switch`/Hadamard dispatch. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`fullband.rs`](fullband.rs), [`causal_probe.rs`](causal_probe.rs) | `96,000`-frame `20 Hz–20 kHz` excitation (`70,976` body + `1,024` settle + `24k` sweep) & stereo causal probe. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
| [`stems.rs`](stems.rs) | Endpoint probing and closed-loop stem capture sealing. | [`capture_and_alignment.md`](../../../../../docs/specs/training/capture_and_alignment.md) |
