# `packages/vm` (`dnam-vm`) — `#![no_std]` Real-Time VM Map

Zero-JSON, `#![no_std]` + `!alloc`-capable `SONA v3.0` virtual machine crate depending strictly on [`dnam-kernels`](../kernels/AGENTS.md) and `zerocopy`. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`README.md`](README.md); run `moon check vm`, `moon run vm:coverage` & `moon run vm:bench`.

| Module / Subsystem | Scoped `AGENTS.md` / Files | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Types** | [`src/lib.rs`](src/lib.rs), [`src/types.rs`](src/types.rs), [`cold_buf.rs`](src/types/cold_buf.rs), [`error.rs`](src/types/error.rs), [`fp_env.rs`](src/types/fp_env.rs) | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
| **32-bit `MathBackend`** | [`src/math.rs`](src/math.rs), [`src/math/AGENTS.md`](src/math/AGENTS.md) | [`op_n_paralru_and_polyadaa.md`](../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **`.dnam-bin` Loader & Stripper** | [`src/bin.rs`](src/bin.rs), [`header.rs`](src/bin/header.rs), [`arena_wire.rs`](src/bin/arena_wire.rs), [`view.rs`](src/bin/view.rs), [`validate.rs`](src/bin/view/validate.rs), [`arena.rs`](src/bin/view/validate/arena.rs), [`builder.rs`](src/bin/builder.rs) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |
| **`ControlBus` & `TempoSync`** | [`src/control.rs`](src/control.rs), [`src/control/AGENTS.md`](src/control/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **DSP Graph & `SonaHotSlab`** | [`src/graph.rs`](src/graph.rs), [`src/graph/AGENTS.md`](src/graph/AGENTS.md) | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
| **`SonaVm` & Dynamic Dispatch** | [`src/vm.rs`](src/vm.rs), [`src/vm/dynamic.rs`](src/vm/dynamic.rs), [`controls.rs`](src/vm/dynamic/controls.rs) | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
| **Integration Suites & WCET** | [`tests/dnam_bin_and_control_bus.rs`](tests/dnam_bin_and_control_bus.rs), [`tests/sona_vm_pipeline_and_parity.rs`](tests/sona_vm_pipeline_and_parity.rs) ([`m9_contracts.rs`](tests/sona_vm_pipeline_and_parity/m9_contracts.rs)), [`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs), [`benches/realtime_wcet.rs`](benches/realtime_wcet.rs) | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
