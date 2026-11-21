# `packages/engine` — Rust `#![no_std]` + `alloc` Layer 1 Shell Routing Guide

**`dnam-engine`** is DNAM's `#![no_std]` + `alloc` Layer 1 `.dnam` container, multi-port `HostShellPatchbay`, SPSC host concurrency, DRM/watermark provenance, and C-ABI shell over [`dnam-vm`](../vm/AGENTS.md) and [`dnam-kernels`](../kernels/AGENTS.md).

## 1. Submodule Discovery Map

| Subsystem / Module | Scoped `AGENTS.md` | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Runtime Shell, Patchbay & ABI** (`src/engine.rs`, `src/engine/*`, `src/dynamic*`, `src/lib.rs`, `src/types.rs`, `src/memory.rs`, `src/ffi.rs`) | [`engine/`](src/engine/AGENTS.md), [`patchbay/`](src/engine/patchbay/AGENTS.md), [`dynamic/`](src/dynamic/AGENTS.md) | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md), [`router_and_crossbar_topology.md`](../../docs/specs/dsp/router_and_crossbar_topology.md), [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
| **Host & SPSC Concurrency** (`src/host.rs`, `src/host/*`) | [`src/host/AGENTS.md`](src/host/AGENTS.md) | [`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md) |
| **`.dnam` Container & `.dnam-bin` Bridge** (`src/container.rs`, `src/container/*`) | [`src/container/AGENTS.md`](src/container/AGENTS.md) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |
| **Provenance, DRM & Watermarking** (`src/provenance.rs`, `src/provenance/*`) | [`src/provenance/AGENTS.md`](src/provenance/AGENTS.md) | [`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../docs/specs/security/watermarking_and_marketplace.md) |
| **JSON Schema & Clamp Compiler** (`src/schema.rs`, `src/schema/*`) | [`src/schema/AGENTS.md`](src/schema/AGENTS.md) | [`dnam_binary_format.md`](../../docs/specs/containers/dnam_binary_format.md) |

## 2. Guides & Verification

1. [`spec_scope.toml`](spec_scope.toml) — Canonical `r[...]` rule scope manifest.
2. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Crate layout, `unsafe` isolation, co-located tests, and `engine:coverage`.
3. [`README.md`](README.md) — Public Rust API, C-ABI, and quickstart commands.
