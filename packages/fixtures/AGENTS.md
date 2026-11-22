# `packages/fixtures` (`dnam-fixtures`) — Agent Discovery & Module Map

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before modifying `packages/fixtures/`.

| Module / File | Responsibility | Canonical Specification (SSOT) |
| :--- | :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | `#![forbid(unsafe_code)]` simulation crate root (`BlackBoxModel`, `FixtureDeviceMetadata`, `FixtureBitDepth`, specs). | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md), [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/sim_kernels.rs`](src/sim_kernels.rs) | Offline `nalgebra` & `realfft` kernels (`UnitaryHouseholder4`, `DiscreteStateSpace`, `ImplicitNodalClipper2Node`, `FftConvolver`, `FftCabinetIrModel`). | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md), [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/chains.rs`](src/chains.rs) | `Oversampler4xF64` & `Chain-01..18` submodules (`drives`, `amps_and_cabs`, `mod_and_spatial`, `dynamics_and_pitch`, `rigs`). | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md), [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/adversaries.rs`](src/adversaries.rs) | 14 fault decorators across `interface_and_control`, `thermal_and_noise`, and `mechanical_and_capsule`. | [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md), [`control_space_audit.md`](../../docs/specs/training/control_space_audit.md) |
| [`src/test_utils.rs`](src/test_utils.rs) | `#[cfg(test)]` helpers (`sine_wave`, `run_mono_block`, `run_stereo_block`). | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| [`tests/prop_fixtures_invariants.rs`](tests/prop_fixtures_invariants.rs) | Integration & `proptest!` verification suite. | [`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md) |
