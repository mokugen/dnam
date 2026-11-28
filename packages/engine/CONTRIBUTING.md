# Contributing to `dnam-engine` (`packages/engine/`)

This guide covers engineering, safety, testing, coverage, and benchmarking standards for **`dnam-engine`** ([`README.md`](README.md)), the `#![no_std]` + `alloc` Layer 1 `.dnam` shell, host patchbay, SPSC host, provenance, and C-ABI crate over [`dnam-vm`](../vm/README.md) and [`dnam-kernels`](../kernels/README.md).

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture, Rule Scope & Directory Layout

Consult the [root `AGENTS.md`](../../AGENTS.md) for the workspace discovery map and the module layout below:

- **`src/engine/` & `src/engine/patchbay/`:** `DnamEngine<'a, B>` wrapper over `dnam_vm::SonaVm`, Layer 1 `HostShellPatchbay` (Stereo, Mono, `4CM`, `WDW`, DualAmp, return/parallel alignment, M-S width), presets, tempo sync, and offline bounce.
- **`src/dynamic/`:** Runtime SIMD backend dispatch (`DynamicDnamEngine<'a>`) with zero per-sample branching.
- **`src/host/`:** Lock-free 3-lane SPSC control ingress, `AcceptedStateShadow`, 64 B Seqlock telemetry, ping-pong model swap, and `DnamHostHandle`.
- **`src/container/`:** 64 B `.dnam` preamble, `zerocopy` v3.0 `DnamContainerView`, `DMU1`/`DMUE` mutable suffix framing, prefix tier slimming, and zero-copy `.dnam-bin` extraction (`extract_dnam_bin`).
- **`src/provenance/`:** `DAU1` authentication envelopes, RFC 8785 JCS header digests, Ed25519 prefix commitments, `OemDeviceCertificate` attestation, RFC 9180 HPKE + ChaCha20-Poly1305 `.dnam-seal` (`DNSL`), buyer watermarking, and `.dnam-prev` containment (`PreviewGuardState`).
- **`src/schema/`:** `#![no_std]` `serde` JSON header types, UI skew curves, `ControlClampDescriptor` compilation, and `DnamError`.

### Specification Rule Scope (`spec_scope.toml`)

Every canonical `r[...]` rule across `docs/specs/` appears in [`spec_scope.toml`](spec_scope.toml) as either `scope = "engine"` (verified via `verify[...]` inside `packages/engine`) or `scope = "out-of-scope"` (with a non-empty `reason` naming `packages/vm`, `packages/kernels`, `packages/capture`, `packages/js`, or `tests/*.test.ts`). Layer 0 VM and kernel DSP rules remain `scope = "out-of-scope"` in `packages/engine` because `dnam-vm` and `dnam-kernels` own and verify them directly; `test_engine_spec_scope_manifest` enforces an exact 1:1 key set and verification parity.

### Test, Benchmark & Example Placement

- **Unit & Property Tests (`#[cfg(test)] mod tests`):** Co-locate all module unit tests, specification verification suites (`verify[...]`), and `proptest!` suites inside the exact submodule they test (`src/container/`, `src/dynamic/`, `src/engine/`, `src/host/`, `src/provenance/`, `src/schema/`, `src/memory.rs`, `src/types.rs`).
  - **Use `proptest!` Wherever Possible:** Sweep full input domains across wire parsers/serializers, alignment filters, SPSC/Seqlock concurrency primitives, schema clamp compilers, and cryptographic/watermarking helpers alongside deterministic boundary vector tests.
- **Integration Tests (`tests/`):** Keep `tests/` strictly for cross-cutting crate integration suites:
  - [`tests/backend_parity.rs`](tests/backend_parity.rs): Cross-ISA backend parity and multi-topology checks across `ScalarBackend`, AArch64 NEON/SVE2/SME, x86-64 SSE2/SSE4.2/AVX2+FMA, and WASM SIMD128.
  - [`tests/realtime_safety.rs`](tests/realtime_safety.rs): `CountingAllocator` verification of zero heap allocations/deallocations across real-time audio callbacks, `DnamHostHandle`, and C-ABI FFI.
  - [`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs): Automated release-binary `objdump -dr` disassembly and transitive call-graph audit verifying zero `FDIV`, `FSQRT`, `IDIV`/`UDIV`/`SDIV`, `libm`, allocator, or uncached OS syscall instructions (`ref[sdk.rust.realtime-disassembly-audit]`).
  - [`tests/sona_tier_and_multiport_lifecycle.rs`](tests/sona_tier_and_multiport_lifecycle.rs): Tier equivalence, multi-port `4CM`/`WDW`/parallel alignment, reload containment, ZOH, and 4× bounce lifecycle checks.
  - [`tests/marketplace_provenance_and_watermark.rs`](tests/marketplace_provenance_and_watermark.rs) & [`tests/marketplace_seal_and_policy.rs`](tests/marketplace_seal_and_policy.rs): End-to-end cryptographic provenance, `.dnam-seal` hardware enclave envelopes, and buyer watermarking.
  - [`tests/ffi_integration.rs`](tests/ffi_integration.rs): C-ABI lifecycle and null-pointer safety across all 15 `dnam_engine_*` exports.
- **Benchmarks (`benches/`):** Place throughput and WCET suites in [`benches/pipeline.rs`](benches/pipeline.rs), [`benches/container_and_provenance.rs`](benches/container_and_provenance.rs), and [`benches/realtime_wcet.rs`](benches/realtime_wcet.rs).
- **Examples (`examples/`):** Place runnable API demonstrations in [`examples/realtime_audio_block.rs`](examples/realtime_audio_block.rs), [`examples/embedded_zero_alloc.rs`](examples/embedded_zero_alloc.rs), [`examples/marketplace_provenance.rs`](examples/marketplace_provenance.rs), [`examples/container_slimming.rs`](examples/container_slimming.rs), [`examples/c_abi_host_integration.rs`](examples/c_abi_host_integration.rs), and [`examples/custom_operator_chain.rs`](examples/custom_operator_chain.rs).

---

## 2. Safety, `unsafe` Isolation & Real-Time Audio Invariants

1. **Strict `unsafe` Isolation & Safe-Rust-First Requirement (`ref[repo.code.type-safety-and-imports]`):**
   - `src/lib.rs` enforces `#![deny(unsafe_code)]`, permitting `#[allow(unsafe_code)]` **only** on [`src/ffi.rs`](src/ffi.rs) (`pub mod ffi`) and [`src/provenance/sha256_hw.rs`](src/provenance/sha256_hw.rs) (`#[cfg(target_arch = "aarch64")]` hardware SHA2 intrinsics).
   - All remaining submodules (`container`, `dynamic`, `engine`, `host`, `memory`, `schema`, `types`, and all safe `provenance` submodules) enforce `#[forbid(unsafe_code)]`.
   - Design lock-free rings, seqlocks, hot-swap handles, and wire codecs using safe Rust primitives (`core::sync::atomic::AtomicU64`, disjoint borrows, and declarative `#[repr(C)]` `zerocopy` wire structs deriving `FromBytes, IntoBytes, KnownLayout, Immutable` per [`.agents/rules/zerocopy-wire-structs.md`](../../.agents/rules/zerocopy-wire-structs.md)).
2. **Zero Audio-Thread Heap Allocations:**
   - `process_block`, `process_block_in_place`, `process_mono_to_stereo`, `process_mono`, `process_block_io`, `set_tier`, `set_sample_rate` (and C-ABI `dnam_engine_set_sample_rate`), `set_control_value`, `reset_to_dc_equilibrium`, `telemetry_snapshot`, and `reload_model_in_place` must never allocate or deallocate heap memory.
3. **Zero `FSQRT`, `FDIV`, `IDIV`/`UDIV`/`SDIV`, `libm`, or Nondeterministic Calls in 48 kHz / 3 kHz Paths (`ref[arch.simd.zero-stall-lowering]`):**
   - Never use floating-point division, square-root, integer division/modulo on runtime integers, `f32::rem_euclid`, `libm` calls, unbounded `while` loops, or uncached syscalls inside real-time patchbay, preview-guard, or audio-callback loops.
4. **Module-Specific Invariants:**
   - **`src/engine/`:** Keep `HostShellPatchbay` structurally isolated (`S_{ret->send} = 0`), clamp `ms_width` to `[0.0, 2.0]`, and derive `ParallelAlignLine` Kaiser-sinc taps at compile time via `dnam_vm::math::const_eval`.
   - **`src/container/` & `src/schema/`:** Validate checked bounds, 64 B alignment, tier-prefix integrity, and authenticated shared extents against the [container format](../../docs/specs/containers/dnam_binary_format.md); keep `serde` / `serde_json` configured with `default-features = false, features = ["alloc"]`.
   - **`src/provenance/`:** Apply the exact versioned [immutable-header projection](../../docs/specs/security/provenance_and_enclave_drm.md#21-preset-safe-canonical-header-digest-rfc-8785-jcs) without heuristic field exclusions; constrain mutable edits strictly to the authenticated extent and capacity.

---

## 3. Testing, Coverage (≥ 95%) & Benchmarking Commands

Run all commands from the monorepo root via `moon`:

```bash
# 1. Run all engine quality gates (format-check, lint, typecheck, build, test)
moon check engine

# 2. Run individual engine tasks
moon run engine:format-check
moon run engine:lint
moon run engine:test

# 3. Check the >= 95% instrumented-line coverage floor
moon run engine:coverage

# 4. Run real-time throughput benchmarks
moon run engine:bench

# 5. Profile Criterion benchmarks with hotpath-rs (--profile profiling)
HOTPATH_LIMIT=0 HOTPATH_CRITERION_ITERATIONS=500 \
  cargo bench -p dnam-engine --profile profiling --bench pipeline --features hotpath-alloc \
  -- --test --exact "process_block_128/ARM64 NEON/ultra"
```

Coverage is a floor, not a bug-free guarantee. Independently challenge the specification and implementation with boundary, differential, long-run, backend, integration, and target-hardware tests; inspect untested branches and platform-specific paths separately.

---

## 4. Pre-Commit Checklist for `packages/engine/`

- [ ] Every new or modified module includes `proptest!` property-based tests (alongside exact boundary/vector tests) co-located in its source file (`#[cfg(test)]`), with `verify[...]` citations matching [`spec_scope.toml`](spec_scope.toml).
- [ ] `moon check engine` passes all formatting, Clippy (`-D warnings`), build, and unit/`proptest`/integration suites (`tests/realtime_safety.rs`, `tests/backend_parity.rs`, `tests/realtime_disassembly_audit.rs`).
- [ ] `moon run engine:coverage` confirms ≥ 95% line coverage, including all newly added or modified functions and tier branches.
- [ ] If any public API or C-ABI signature changed, `packages/js/rust/` bindings and [`README.md`](README.md) / [`CONTRIBUTING.md`](CONTRIBUTING.md) are updated accordingly.
