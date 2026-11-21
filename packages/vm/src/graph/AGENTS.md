# `dnam-vm::graph` — Real-Time DSP Graph & L1D Slab Map

Scoped discovery guide for [`src/graph.rs`](../graph.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Strictly `#![forbid(unsafe_code)]`, `--no-default-features` `!alloc`, zero `FDIV`/`FSQRT`/`IDIV`/`libm` in audio loops.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`zoh.rs`](zoh.rs) | Dual-rate `ZohRateContext`, `ParaLRU` ZOH, `K_ctrl` stride | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`router.rs`](router.rs) ([`detector.rs`](router/detector.rs), [`align.rs`](router/align.rs)) | 8-port `IoRouter` (`S_{ret->send} == 0`), `LR4` envelope bank, aligner | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`oversample.rs`](oversample.rs) ([`bounce4x.rs`](oversample/bounce4x.rs)) | Zero-lag `2x` `SonaOs2xHot` (`256 B`) & `4x` `Bounce4xOversampler` | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`cabinet.rs`](cabinet.rs) ([`kautz.rs`](cabinet/kautz.rs), [`high_sr.rs`](cabinet/high_sr.rs), [`lagrange.rs`](cabinet/high_sr/lagrange.rs)) | Lag-ordered Early FIR, `KautzStereoBank4`, high-SR `6,144`-`f32` ext | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`tank.rs`](tank.rs) ([`quad.rs`](tank/quad.rs), [`step.rs`](tank/quad/step.rs), [`ring.rs`](tank/ring.rs)) | `TankQuadHotCompact` (`224 B`), `TankExtRing`, Schroeder bank, `WE` dual-latch | [`op_t_transport.md`](../../../../docs/specs/dsp/op_t_transport.md) |
| [`slab.rs`](slab.rs) ([`layout.rs`](slab/layout.rs), [`cold.rs`](slab/layout/cold.rs), [`hydrate.rs`](slab/hydrate.rs), [`apply_tier.rs`](slab/hydrate/apply_tier.rs), [`wire_decode.rs`](slab/hydrate/wire_decode.rs), [`lifecycle.rs`](slab/hydrate/lifecycle.rs), [`ctrl_latch.rs`](slab/ctrl_latch.rs), [`microchunk.rs`](slab/microchunk.rs), [`core_step.rs`](slab/microchunk/core_step.rs)) | Prefix-ordered L1D `SonaHotSlab` (`4,544..14,656 B`), cold ZOH hydration, `3 kHz` folding, `1/4`-frame causal micro-chunk stepping | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
