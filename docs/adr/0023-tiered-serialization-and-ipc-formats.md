---
id: ADR-0023
title: "Tiered Serialization & IPC Formats (zerocopy, RFC 8785 JCS, buffa, postcard)"
status: accepted
version: "3.0.0"
date: "2026-11-21"
supersedes: []
superseded_by: null
specs:
  - ../specs/sdks/rust_sdk.md
  - ../specs/containers/dnam_binary_format.md
  - ../specs/containers/project_container_format.md
requirements:
  - REQ-6.2
  - REQ-8.1
  - REQ-8.8
---

# ADR-0023: Tiered Serialization & IPC Formats (`zerocopy`, RFC 8785 JCS, `buffa`, `postcard`)

> [!NOTE]
> Records why DNAM standardizes on four distinct serialization layers across fixed L1D-resident wire structs, cryptographic manifests, external binary RPCs, and closed `#![no_std]` Rust IPC (`ref[sdk.rust.zerocopy-wire-structs]`, `REQ-6.2`, `REQ-8.1`, `REQ-8.8`).

---

## 1. Context & Problem Statement

DNAM spans incompatible latency, allocation, portability, and cryptographic guarantees:

1. **Audio-thread & `!alloc` VM loading (`< 50 ns`):** Layer 0 `.dnam-bin` (`4,608`–`30,720 B`) and fixed `256 B` `DNPR` / `64 B` `DnwbHeader64BWire` frames must bind directly from aligned read-only memory inside `#![no_std]` / `#![forbid(unsafe_code)]` with zero heap allocation or parsing loops.
2. **Cryptographic manifest canonicalization:** `.dnam`, `.dnam-seal`, and `.dnam-proj` Ed25519 signatures require byte-identical canonical JSON digests across TypeScript (`N-API` / browser) and Rust while allowing preset/catalog inspection in tools and browsers.
3. **Cross-language binary RPC & APIs:** Remote/IPC APIs crossing process, FFI, or service boundaries need explicit field tags, forward/backward schema evolution, and multi-language codegen.
4. **Closed, ephemeral `#![no_std]` Rust ↔ Rust IPC:** Local variable-length control envelopes (e.g., coupled white-box subprocess commands over `DNWB` pipes) require compact framing without codegen overhead or heap-allocating text parsers.

| Naive Single-Format Formulation | Failure Mechanism | Audible, Security, or Engineering Consequence |
| :--- | :--- | :--- |
| **`Monolithic JSON-Only Wire`** | Requires allocation, UTF-8 scanning, and string-to-float parsing during real-time preset/tier loading and control ticks. | Breaks `< 50 ns` Layer 0 `.dnam-bin` binding, violates `#![no_std]` `!alloc`, and allocates on audio paths (`REQ-6.2`). |
| **`Unversioned Rust-Only Wire Everywhere`** | Couples disk containers and public RPCs to unstable Rust in-memory layout or non-canonical field order. | Breaks TypeScript SDK interoperability, cross-version schema evolution, and Ed25519 manifest reproducibility (`REQ-8.1`). |
| **`Manual from_le_bytes Byte Slicing`** | Separates offset constants from struct definitions in imperative decoder routines. | Introduces off-by-one/endian bugs and obscures L1D alignment verification (`REQ-8.8`). |
| **`Protobuf Encoding for L1D Audio Arenas`** | Encodes varints/length prefixes requiring linear pass decoding into temporary heap structs before playback. | Destroys zero-copy `SonaVm::open_dnam_bin` binding and adds unnecessary decode latency. |

---

## 2. Decision & Format Fit Matrix

| Use Case in DNAM | Canonical Format / Crate | Rationale & Boundary Invariant |
| :--- | :--- | :--- |
| **Fixed binary headers, descriptors, arenas & seqlocks** (`ref[sdk.rust.zerocopy-wire-structs]`) | **`zerocopy 0.8`** (`FromBytes` / `IntoBytes` + `little_endian::*`) | Binds `.dnam-bin`, `.dnam`, `.dnam-seal`, `.dnam-prev`, `.dnam-proj`, `Arena0..3`, `DNPR`, `DnwbHeader64BWire`, and WAV headers in `< 50 ns` under `#![forbid(unsafe_code)]` and `--no-default-features` (`!alloc`), verified by compile-time `size_of`/`align_of`/`offset_of!` assertions. |
| **Container manifests & presets** (`ref[sec.jcs.immutable-header-digest]`) | **`serde_json` + RFC 8785 JCS** | Produces deterministic, human-inspectable UTF-8 metadata and byte-exact `immutable_header_sha256` across Rust and TypeScript SDKs without binary schema ambiguity. |
| **Binary RPC / APIs** | **`buffa` (Protobuf Editions)** | Provides clean pure-Rust Protobuf Editions evolution for cross-process / cross-language APIs and RPC envelopes where field-level backward/forward compatibility matters. |
| **Closed, ephemeral `#![no_std]` Rust ↔ Rust IPC** | **`postcard`** (or **`zerocopy 0.8`** when latency/throughput requires) | Minimizes framing bytes and CPU overhead for variable-length ephemeral control messages between coupled Rust binaries built from the same workspace revision (e.g., white-box runner subprocess control channels in `dnam-capture`), while permitting `zerocopy 0.8` fixed wire structs whenever `postcard` cannot meet latency or throughput requirements (such as `DNPR` seqlocks, streaming `DnwbHeader64BWire` audio frames, and hot-path IPC buffers). |

---

## 3. Consequences & Scope Boundaries

1. **Positive Consequences:** Each boundary pays zero unnecessary runtime, binary-size, or schema-drift cost: `dnam-vm` remains `#![no_std]` / `!alloc` with zero `serde` / `buffa` / `postcard` dependencies, cryptographic containers remain deterministic under RFC 8785 JCS, public RPCs gain Protobuf Editions schema safety, and internal Rust subprocess IPC avoids verbose text serialization or hand-rolled variable-length framing.
2. **Evaluated R&D Alternatives:** Considered reusing `zerocopy` fixed structs padded with large static arrays for variable-length subprocess control strings (`wasted bandwidth + artificial length caps`), using `bincode` (`lacks #![no_std] postcard ubiquity`), and using `prost` (`superseded by buffa` for pure-Rust Protobuf Editions ergonomics).
