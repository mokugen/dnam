# `dnam-vm::graph::slab` — Prefix-Ordered L1D `SonaHotSlab` Map

Scoped discovery guide for [`slab.rs`](../slab.rs) (parent [`graph/AGENTS.md`](../AGENTS.md)). Strictly `#![forbid(unsafe_code)]`, `--no-default-features` `!alloc`-capable.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`layout.rs`](layout.rs), [`cold.rs`](layout/cold.rs) | `SonaHotSlab` (`[4_544, 7_552, 11_136, 14_592] B`), `CoreQuadHot` (`144 B`), `CoreQuadExtCold` (`40 B`), `OpNPreHotState` (`28 B`), `SonaColdTail` (`ro_pole_48k`, `base_tank_dry_mix`) | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`hydrate.rs`](hydrate.rs), [`apply_tier.rs`](hydrate/apply_tier.rs), [`wire_decode.rs`](hydrate/wire_decode.rs), [`lifecycle.rs`](hydrate/lifecycle.rs) | Cold `!alloc` hydration, ZOH `ro_pole_48k` / `CoreQuad` rescaling, prefix overrides, `switch_tier`, `retime_sample_rate`, `set_quad_write_enable` / `set_op_t_write_enable` | [`router_and_crossbar_topology.md`](../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`ctrl_latch.rs`](ctrl_latch.rs) | `3 kHz` target latching, `eff_z_stage` laws & register hoisting | [`parametric_control_bus.md`](../../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`microchunk.rs`](microchunk.rs), [`core_step.rs`](microchunk/core_step.rs) | `4`-frame micro-chunking, `Op-N[Pre]`, `Op-P` + `CoreQuad` M/S + `GLU`, `hp_lp` readout | [`router_and_crossbar_topology.md`](../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
