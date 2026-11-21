# `src/engine/patchbay` — Layer 1 Host-Shell 8-Port Patchbay & Aligners

Implements the Layer 1 outer 8-port host-shell router and sub-sample alignment lines (`#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../patchbay.rs`](../patchbay.rs) | `HostShellPatchbay`, `IoRoutingMode`, `WdwDryTapMode`, `eval_wdw_routing_outputs`, structural `S_{ret->send} = 0` isolation & M-S width | [`router_and_crossbar_topology.md`](../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`align.rs`](align.rs) | `ReturnAligner` (Catmull-Rom return latency line) & `ParallelAlignLine` (`FullBand64Tap` / `LowLatency16Tap` / `MinLatency4Tap`) + `proptest!` | [`router_and_crossbar_topology.md`](../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`align/kaiser.rs`](align/kaiser.rs) | Compile-time Kaiser-sinc alignment filter tap tables (`ParallelAlignmentMode`, `IoAlignmentMode`) via `dnam_vm::math` `const_eval` | [`router_and_crossbar_topology.md`](../../../../../docs/specs/dsp/router_and_crossbar_topology.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for real-time zero-division invariants.
