# `src/host` — Real-Time Control SPSC Ingress & Wait-Free Telemetry

Lock-free real-time host concurrency primitives (`#[forbid(unsafe_code)]`): 3-Lane SPSC control ingress (`SpscControlRing`, `AcceptedStateShadow`), wait-free Seqlock telemetry (`TelemetryCell`), ping-pong model swap (`RealtimeEngineSwap`), and unified `DnamHostHandle`.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`spsc.rs`](spsc.rs) | 3-Lane SPSC re-exports | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`spsc/ring.rs`](spsc/ring.rs) | 64B-aligned Lane 1/2/3 rings & packing | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`spsc/shadow.rs`](spsc/shadow.rs) | `AcceptedStateShadow` clamping & roles | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`spsc/producer.rs`](spsc/producer.rs) | Non-RT `ControlProducer` & presets | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`spsc/consumer.rs`](spsc/consumer.rs) | Real-time `ControlConsumer` drain | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`telemetry.rs`](telemetry.rs) | Wait-free Seqlock `TelemetryCell` | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`swap.rs`](swap.rs) | Lock-free `RETIRED_SLOTS=8` swap & zero RT drops | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`handle.rs`](handle.rs) | `DnamHostHandle` root & pre-warmed mutex | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`handle/lifecycle.rs`](handle/lifecycle.rs) | Non-blocking init, swap & telemetry | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`handle/controls.rs`](handle/controls.rs) | Wait-free SPSC control & shadow getters | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`handle/process.rs`](handle/process.rs) | Sub-block audio & offline bounce | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for real-time execution safety invariants.
