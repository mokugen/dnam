# `stage2_gru/` — Zero-DC Matryoshka `GRU` & Inter-Block `TPT-SVF` Map

Follow [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) (`#![forbid(unsafe_code)]`, `0` `FDIV`/`FSQRT`/`IDIV`/`libm`).

| Module | Role | Canonical Reference |
| :--- | :--- | :--- |
| [`../stage2_gru.rs`](../stage2_gru.rs) | Stage II facade & `Stage2Backend` trait | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`block.rs`](block.rs) | Tapered `[1,1,1]..[3,2,1]` `V3GruBlockHot` & stereo tiles | [`tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md) |
| [`zoh.rs`](zoh.rs) | `V3GruBlockCold`, `2x -> 1x` & ZOH gate scaling | [`router_and_stages_topology.md`](../../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`math_pade.rs`](math_pade.rs) | `NR2` Padé `[5/4]` `tanh` & `sigmoid` primitives | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`tone_svf.rs`](tone_svf.rs) | Inter-block `3 kHz` `TPT-SVF` 3-band tone stack | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`film.rs`](film.rs) | `3 kHz` `FiLM` gain, Zero-DC `b_n`, control dispatch | [`parametric_control_bus.md`](../../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`frontend.rs`](frontend.rs) | `2x` zero-lag polyphase + half-band upsampler/decimator | [`tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md) |
| [`features.rs`](features.rs) | Multi-timescale RMS/pinch/sag input feature extractors | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`kernels.rs`](kernels.rs), [`kernels/stage2.rs`](kernels/stage2.rs), [`kernels/stage2/prepass.rs`](kernels/stage2/prepass.rs), [`kernels/stage2/readout.rs`](kernels/stage2/readout.rs) | Stack-scratch `1x`/`2x` Stage II & readout kernels | [`ADR-0021`](../../../../../docs/adr/0021-unified-matryoshka-gru-and-direct-register-stages.md) |
