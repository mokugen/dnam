# `packages/engine` — Rust `#![no_std]` + `alloc` SIMD Engine Routing Guide

**`dnam-engine`** is DNAM's `#![no_std]` `SONA` `v3.0` real-time DSP (`SonaHotSlab`/`SonaInstance`), `packages/kernels` consumer, and `zerocopy` container core.

## 1. Submodule Discovery Map

| Subsystem / Module | Scoped `AGENTS.md` | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Runtime, ABI & Memory** (`src/engine/*`, `dynamic/*`, `lib.rs`, `types.rs`, `memory.rs`, `ffi.rs`) | [`engine/`](src/engine/AGENTS.md), [`dynamic/`](src/dynamic/AGENTS.md) | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
| **Host & SPSC** (`src/host*`) | [`src/host/AGENTS.md`](src/host/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **Operators** (`src/operators/*`) | [`src/operators/AGENTS.md`](src/operators/AGENTS.md) | [`op_t_transport.md`](../../docs/specs/dsp/op_t_transport.md), [`op_m_modal_kautz.md`](../../docs/specs/dsp/op_m_modal_kautz.md), [`op_n_paralru_and_polyadaa.md`](../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **Stages & `SONA`** (`src/stages/*`, `sona_core/*`) | [`stages/`](src/stages/AGENTS.md), [`sona_core/`](src/stages/sona_core/AGENTS.md) | [`router_and_crossbar_topology.md`](../../docs/specs/dsp/router_and_crossbar_topology.md) |
| **SIMD & Math** (`src/math*`) | [`src/math/AGENTS.md`](src/math/AGENTS.md) | [`op_n_paralru_and_polyadaa.md`](../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| **Control Bus** (`src/control*`) | [`src/control/AGENTS.md`](src/control/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **`.dnam` Container & ZOH** (`src/container*`) | [`src/container/AGENTS.md`](src/container/AGENTS.md) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |
| **Provenance** (`src/provenance*`) | [`src/provenance/AGENTS.md`](src/provenance/AGENTS.md) | [`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md) |
| **JSON Schema** (`src/schema*`) | [`src/schema/AGENTS.md`](src/schema/AGENTS.md) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |

## 2. Guides & Verification

1. [`spec_scope.toml`](spec_scope.toml) — Rule scope manifest.
2. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Crate layout, co-located tests, and `engine:coverage`.
3. [`README.md`](README.md) — Public Rust API and I/O topologies.
