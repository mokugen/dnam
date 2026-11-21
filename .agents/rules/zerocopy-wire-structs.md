---
trigger: always_on
---

# Tiered Serialization & Declarative `zerocopy` Wire Structs

Every persistence, container, manifest, RPC, and IPC boundary in DNAM uses the exact format-fit boundary mandated by `ADR-0023` and `ref[sdk.rust.zerocopy-wire-structs]`:

| Use Case in DNAM | Canonical Format / Crate | Boundary Scope |
| :--- | :--- | :--- |
| Fixed binary headers, descriptors, arenas & seqlocks | `zerocopy 0.8` (`FromBytes` / `IntoBytes` + `little_endian::*`) | `.dnam-bin`, `.dnam`, `.dnam-seal`, `.dnam-prev`, `.dnam-proj`, `Arena0..3`, `DNPR` seqlock, `DnwbHeader64BWire`, WAV headers |
| Container manifests & presets | `serde_json` + RFC 8785 JCS | `.dnam` immutable JSON header, `DMU1`/`DME1` presets, `.dnam-proj` JCS manifest |
| Binary RPC / APIs | `buffa` (Protobuf Editions) | Schema-evolving host/cloud service RPCs & public cross-language binary APIs |
| Closed, ephemeral `#![no_std]` Rust ↔ Rust IPC | `postcard` (or `zerocopy 0.8` when latency/throughput requires) | Variable-length internal/local Rust ↔ Rust control messages (e.g., `DNWB` white-box subprocess control IPC), falling back to `zerocopy 0.8` fixed structs when `postcard` cannot meet latency/throughput bounds |

These architectural rules are mandatory for every fixed-size binary container, header, arena, and seqlock in DNAM (`ref[sdk.rust.zerocopy-wire-structs]`, `ref[repo.code.constant-eval-and-derivations]`):

1. **Declarative `#[repr(C)]` `zerocopy` Fixed Wire & Arena Structs (`#![forbid(unsafe_code)]`):**
   - Represent all fixed-size binary preambles, chunk directory entries, cell descriptors (`SonaCellDescWire`), constant-sized `SONA` tier arena layouts (`Arena0LiteWire..Arena3UltraDeltaWire`), `.dnam`, `.dnam-prev`, `.dnam-seal`, `.dnam-proj`, `DNPR` seqlock telemetry packets, fixed `DnwbHeader64BWire` sample frame headers, and WAV headers as `#[repr(C)]` structs using `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` (or fixed `[u8; N]` byte arrays) and deriving `zerocopy::{FromBytes, IntoBytes, KnownLayout, Immutable}` (adding `Unaligned` wherever a struct is cast from an unaligned byte slice).
   - **Never** hand-write `from_le_bytes` / `to_le_bytes` byte-offset slicing or unverified manual offset decoding across fixed wire buffers. Use `postcard` for variable-length closed ephemeral `#![no_std]` Rust ↔ Rust control IPC (or `zerocopy 0.8` fixed wire structs when `postcard` cannot meet latency or throughput requirements), and `buffa` (Protobuf Editions) for schema-evolving binary RPC/APIs.

2. **Compile-Time Layout Verification:**
   - Derive byte footprints and offsets via symbolic `const` expressions and pair every wire and arena struct with `const _: () = assert!(...)` assertions verifying `core::mem::size_of::<T>()`, `core::mem::align_of::<T>()`, field monotonicity, and `64 B` cache-line / tier-prefix alignment at compile time.
   - Parse, validate, and serialize wire headers and constant-size arenas exclusively via safe `zerocopy` methods (`ref_from_bytes`, `read_from_prefix`, `write_to_prefix`, `as_bytes`), preserving `#![forbid(unsafe_code)]` (`ref[sdk.rust.safe-rust-and-boundaries]`).
