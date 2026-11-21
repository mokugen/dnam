# `src/engine` — Layer 1 Audio Engine Wrapper, Patchbay & Runtime

Implements the Layer 1 engine shell (`#[forbid(unsafe_code)]`) wrapping live and bounce `dnam_vm::SonaVm` instances (`../engine.rs`):
- Dual-rate block execution, `HostShellPatchbay` 8-port multi-port dispatch, and `PreviewGuardState` hooks.
- Zero-allocation model hot-swapping, sample-rate retiming, preset resolution, tap tempo, and isolated 4× offline bounce.

## 1. Submodule Map

| Component / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../engine.rs`](../engine.rs), [`shell.rs`](shell.rs) | `DnamEngine<'a, B>`, `EngineBackend`, `OFFLINE_BOUNCE_PDC_SAMPLES`, patchbay/lock/telemetry accessors | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md), [`rust_sdk.md`](../../../../docs/specs/sdks/rust_sdk.md) |
| [`patchbay.rs`](patchbay.rs) ([`patchbay/AGENTS.md`](patchbay/AGENTS.md)) | 8-port `HostShellPatchbay` (`4CM`/`WDW`/DualAmp, `S_{ret->send} = 0`), return/parallel alignment & M-S width | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`process.rs`](process.rs) ([`multiport.rs`](process/multiport.rs)) | `DnamAudioIo`, stereo/mono & 4-port audio callbacks, post-mix & safety sanitization | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md), [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`runtime.rs`](runtime.rs) ([`bounce.rs`](runtime/bounce.rs), [`loader.rs`](runtime/loader.rs), [`owned.rs`](runtime/loader/owned.rs), [`json_scan.rs`](runtime/loader/json_scan.rs), [`controls_scan.rs`](runtime/loader/json_scan/controls_scan.rs)) | `.dnam`/`.dnam-seal`/`.dnam-prev` load, zero-alloc in-place hot-swap, owned reload & isolated `bounce_vm` render | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`presets.rs`](presets.rs) | Stack-allocated preset batch resolution and calibrated `StereoMode`/`StereoWidth` role side effects | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`tempo.rs`](tempo.rs) | Host BPM tap-tempo tracker and `TempoSubdivision` sync updates | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for memory budgets and real-time execution safety.
