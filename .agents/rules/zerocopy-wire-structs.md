---
trigger: always_on
---

# Declarative `zerocopy` Wire & Arena Structs

These architectural rules are mandatory for every container, IPC, and audio-file wire format in DNAM (`ref[repo.code.zerocopy-wire-structs]`, `ref[repo.code.constant-eval-and-derivations]`):

1. **Declarative `#[repr(C)]` `zerocopy` Wire & Arena Structs (`#![forbid(unsafe_code)]`):**
   - Represent all binary preambles, chunk directory entries, cell descriptors (`SonaCellDescWire`), constant-sized `SONA` tier arena layouts (`Arena0LiteWire..Arena3UltraDeltaWire`), `.dnam`, `.dnam-prev`, `.dnam-seal`, `.dnam-proj`, `DNWB`, and WAV headers as `#[repr(C)]` structs using `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` (or fixed `[u8; N]` byte arrays) and deriving `zerocopy::{FromBytes, IntoBytes, KnownLayout, Immutable}` (adding `Unaligned` wherever a struct is cast from an unaligned byte slice).
   - **Never** hand-write `from_le_bytes` / `to_le_bytes` byte-offset slicing or unverified manual offset decoding across wire buffers.

2. **Compile-Time Layout Verification:**
   - Derive byte footprints and offsets via symbolic `const` expressions and pair every wire and arena struct with `const _: () = assert!(...)` assertions verifying `core::mem::size_of::<T>()`, `core::mem::align_of::<T>()`, field monotonicity, and `64 B` cache-line / tier-prefix alignment at compile time.
   - Parse, validate, and serialize wire headers and constant-size arenas exclusively via safe `zerocopy` methods (`ref_from_bytes`, `read_from_prefix`, `write_to_prefix`, `as_bytes`), preserving `#![forbid(unsafe_code)]` (`ref[repo.code.type-safety-and-imports]`).
