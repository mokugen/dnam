# `packages/vm` (`dnam-vm`) — `#![no_std]` Real-Time VM Map

Zero-JSON, `#![no_std]` + `!alloc`-capable `SONA v3.0` virtual machine crate depending strictly on [`dnam-kernels`](../kernels/AGENTS.md) and `zerocopy`. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`README.md`](README.md); run `moon check vm` & `moon run vm:coverage`.

| Module / Subsystem | Scoped `AGENTS.md` / Files | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Types** | [`src/lib.rs`](src/lib.rs), [`src/types.rs`](src/types.rs), [`cold_buf.rs`](src/types/cold_buf.rs), [`error.rs`](src/types/error.rs), [`fp_env.rs`](src/types/fp_env.rs) | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
| **32-bit `MathBackend`** | [`src/math.rs`](src/math.rs), [`src/math/AGENTS.md`](src/math/AGENTS.md) | [`op_n_paralru_and_polyadaa.md`](../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **`.dnam-bin` Loader & Stripper** | [`src/bin.rs`](src/bin.rs), [`header.rs`](src/bin/header.rs), [`arena_wire.rs`](src/bin/arena_wire.rs), [`view.rs`](src/bin/view.rs), [`builder.rs`](src/bin/builder.rs) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |
| **`ControlBus` & `TempoSync`** | [`src/control.rs`](src/control.rs), [`src/control/AGENTS.md`](src/control/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **Integration Suites** | [`tests/dnam_bin_and_control_bus.rs`](tests/dnam_bin_and_control_bus.rs), [`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs) | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
