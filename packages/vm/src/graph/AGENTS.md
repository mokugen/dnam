# `dnam-vm::graph` — Real-Time DSP Graph & L1D Slab Map

Scoped discovery guide for [`src/graph.rs`](../graph.rs) (`ConvolverBackend` & `GraphBackend`, parent [`AGENTS.md`](../../AGENTS.md)). Strictly `#![forbid(unsafe_code)]`, `--no-default-features` `!alloc`, zero `FDIV`/`FSQRT`/`IDIV`/`libm` in audio loops.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`zoh.rs`](zoh.rs) | Dual-rate `ZohRateContext`, `ParaLRU` ZOH, `K_ctrl` stride | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`sensory.rs`](sensory.rs) | `4`-lane `SensoryQuadHot` (`48 B`) & `SensoryQuadCold` bank + stereo input VCA | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`oversample.rs`](oversample.rs) ([`bounce4x.rs`](oversample/bounce4x.rs)) | Zero-lag `2x` `SonaOs2xHot` (`256 B`) & `4x` `Bounce4xOversampler` | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`convolver.rs`](convolver.rs) ([`kautz.rs`](convolver/kautz.rs), [`high_sr.rs`](convolver/high_sr.rs), [`lagrange.rs`](convolver/high_sr/lagrange.rs)) | `ConvolverBackend` lag-ordered Early FIR, `KautzStereoBank4`, high-SR `ConvolverHighSrFirExt` (`6,144`-`f32`) | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`transport.rs`](transport.rs) ([`dispatch.rs`](transport/dispatch.rs), [`quad.rs`](transport/quad.rs), [`hydrate.rs`](transport/quad/hydrate.rs), [`step.rs`](transport/quad/step.rs), [`ring.rs`](transport/ring.rs), [`schroeder.rs`](transport/ring/schroeder.rs)) | `TransportQuadHotCompact` (`216 B`), `PartitionedSchroederBank`, `sigma_sat`, `freeze_mix`, `tank_dry_mix` | [`op_t_transport.md`](../../../../docs/specs/dsp/op_t_transport.md) |
| [`slab.rs`](slab.rs), [`slab/AGENTS.md`](slab/AGENTS.md) | L1D `SonaHotSlab` (`[4_544, 7_552, 11_136, 14_592] B`), `CoreQuadHot` (`144 B`), `CoreQuadExtCold` (`40 B`), `OpNPreHotState`, `ro_pole_48k` | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
