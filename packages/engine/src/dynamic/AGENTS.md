# `src/dynamic` — Block-Level Dynamic Backend Engine Dispatcher

Implements runtime hardware backend selection (`DynamicDnamEngine<'a>`, `#[forbid(unsafe_code)]`):
- Block-level dynamic dispatch across `BackendKind` variants with **zero per-sample branching**.
- Supports Scalar, ARM64 NEON/SVE2/SME, x86-64 SSE2/SSE4.2/AVX2+FMA, and WASM SIMD128 via `dnam-vm`.
- Automatic CPU feature probing and model hot-swapping.

## 1. Submodule Map

| Component / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../dynamic.rs`](../dynamic.rs) | `DynamicDnamEngine<'a>` enum & runtime CPU backend loader | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`dispatch.rs`](dispatch.rs) | Block-level forwarding, patchbay/tier/bounce getters/setters & `proptest!` | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md), [`rust_sdk.md`](../../../../docs/specs/sdks/rust_sdk.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for memory budgets and real-time execution safety.
