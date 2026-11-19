# `src/stages` — `SONA` Rig Topology & Matryoshka Core

Implements the Phase-1 Matryoshka stages (`Phase 2` `SONA` cutover; see [`CONTRIBUTING.md`](../../CONTRIBUTING.md)).

## 1. Submodule Map

| Stage / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`hot_core.rs`](hot_core.rs) ([`instance.rs`](hot_core/instance.rs)) | Compatibility facade delegating `V3HotCore` / `V3CoreInstance` to `sona_core` | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`router.rs`](router.rs) ([`v3_stage0.rs`](router/v3_stage0.rs)), [`topology.rs`](topology.rs) ([`router_and_head.rs`](topology/router_and_head.rs), [`zoh_fir_onset.rs`](topology/zoh_fir_onset.rs)) | Stage 0 router, ZOH/FIR onset & crossover | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`sona_core.rs`](sona_core.rs), [`sona_core/AGENTS.md`](sona_core/AGENTS.md) | `SonaHotSlab`, `SonaInstance`, `Op-P` + `4x Q0.32` & `tick16` kernel | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`stage3_cab.rs`](stage3_cab.rs) ([`backend.rs`](stage3_cab/backend.rs), [`cold.rs`](stage3_cab/cold.rs), [`high_sr.rs`](stage3_cab/high_sr.rs), [`kautz_zoh.rs`](stage3_cab/kautz_zoh.rs)) | Stage III FIR, high-SR bounce & Kautz banks | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage4_tank.rs`](stage4_tank.rs) ([`v3_fdn.rs`](stage4_tank/v3_fdn.rs), [`params.rs`](stage4_tank/v3_fdn/params.rs)) | Stage IV `Op-T` FDN quads & Schroeder bank | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`weights.rs`](weights.rs), [`weights/AGENTS.md`](weights/AGENTS.md) | 64B `Arena 0..3` directory & binary packing | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for stage chaining and memory layout.
