# `src/engine` — TORA Audio Engine Dispatcher & Runtime

Implements the top-level audio engine runtime dispatcher (`#[forbid(unsafe_code)]`):
- Dual-rate execution lifecycle (`48 kHz` audio / `3 kHz` control bus).
- 4-tier slimmable Matryoshka SIMD dispatch (`Lite` ⊂ `Full` ⊂ `Pro` ⊂ `Ultra`).
- Native Mono (`1 -> 1`, `2 -> 1`) and True Stereo (`1 -> 2`, `2 -> 2`) processing via [`V3HotCore`](../stages/hot_core.rs).
- Zero audio-loop allocations and L1D-resident state buffer management.

## 1. Submodule Map

| Component / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`process.rs`](process.rs) ([`multiport.rs`](process/multiport.rs)) | Dual-rate audio block callback, envelope guard, `4CM`/`WDW`/sidechain topology dispatch & `step_frame_tora` | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md), [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`runtime.rs`](runtime.rs) ([`loader.rs`](runtime/loader.rs)) | `V3MatryoshkaModel` unpacking, sample-rate reconfiguration, hot-swap reload & offline bounce | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`presets.rs`](presets.rs) | Stack-allocated preset batch resolution and calibrated role side effects | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md), [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`oversampler.rs`](oversampler.rs) | `4x` linear-phase polyphase FIR oversampler (`OFFLINE_BOUNCE_PDC_SAMPLES = 12`) | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`tempo.rs`](tempo.rs) | Host BPM tempo-sync subdivision resolver and octave folding | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for embedded memory budgets and real-time execution safety.
