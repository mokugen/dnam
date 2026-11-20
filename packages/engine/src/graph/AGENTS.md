# `src/graph` — `SONA` Rig Topology & Matryoshka Core

Implements the `SONA` `v3.0` signal graph (`GraphBackend`, `IoRouter`, `SonaHotSlab`, `CabinetHot`, `SpatialTankHot`; see [`CONTRIBUTING.md`](../../CONTRIBUTING.md)).

## 1. Submodule Map

| Subsystem / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`router.rs`](router.rs) ([`crossbar.rs`](router/crossbar.rs)), [`topology.rs`](topology.rs) ([`router_and_head.rs`](topology/router_and_head.rs), [`tank_and_laws.rs`](topology/router_and_head/tank_and_laws.rs), [`zoh_fir_onset.rs`](topology/zoh_fir_onset.rs)) | `IoRouter`, `CrossbarHot`, ZOH/FIR onset & laws | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`sona_core.rs`](sona_core.rs), [`sona_core/AGENTS.md`](sona_core/AGENTS.md) | `SonaHotSlab`, `SonaInstance`, `Op-P` + `4x Q0.32` & `tick16` kernel | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`cabinet.rs`](cabinet.rs) ([`backend.rs`](cabinet/backend.rs), [`cold.rs`](cabinet/cold.rs), [`high_sr.rs`](cabinet/high_sr.rs), [`kautz_zoh.rs`](cabinet/kautz_zoh.rs)) | `CabinetHot`, `CabinetBackend`, FIR, bounce & Kautz banks | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`tank.rs`](tank.rs) ([`fdn.rs`](tank/fdn.rs), [`params.rs`](tank/fdn/params.rs), [`quad_state.rs`](tank/fdn/quad_state.rs)) | `SpatialTankHot`, `TankQuadHotCompact` & Schroeder bank | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`weights.rs`](weights.rs), [`weights/AGENTS.md`](weights/AGENTS.md) | 64B `Arena 0..3` directory & binary packing | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for graph chaining and memory layout.
