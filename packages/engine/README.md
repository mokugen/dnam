# `dnam-engine` — `#![no_std]` + `alloc` Layer 1 `.dnam` Shell, Patchbay, Provenance & C-ABI

**`dnam-engine`** is the zero-audio-allocation Layer 1 container, multi-port routing, host concurrency, and marketplace provenance shell for **Dynamic Neural Audio Modeler (DNAM)**, wrapping [`dnam-vm`](../vm/README.md) (`SonaVm`) and [`dnam-kernels`](../kernels/README.md).

It targets embedded ARMv8.2-A/ARMv9.2-A pedalboards, desktop VST3/CLAP/AU plugins, analytical pure-Rust capture and training (`packages/capture`), and WebAssembly SIMD128 (`AudioWorklet`) / Node.js N-API runtimes (`packages/js`, `packages/studio`).

---

## Architecture Overview

- **`#![no_std]` + `alloc` & Zero Real-Time Allocations:**
  - Compiles without `std` using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`), `zeroize`, and `zerocopy`.
  - Enforces `#![deny(unsafe_code)]` at the crate root, `#[allow(unsafe_code)]` solely on [`src/ffi.rs`](src/ffi.rs) and [`src/provenance/sha256_hw.rs`](src/provenance/sha256_hw.rs), and `#[forbid(unsafe_code)]` across all safe modules.
  - Verified by the custom `#[global_allocator]` suite ([`tests/realtime_safety.rs`](tests/realtime_safety.rs)), release-binary disassembly audit ([`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs)), and [`spec_scope.toml`](spec_scope.toml) (Layer 1 engine-scoped `r[...]` rules verified via `verify[...]`).
- **Layer 1 `.dnam` Container & `.dnam-bin` Bridge (`src/container/`, `src/schema/`):**
  - Parses and validates v3.0 64 B-aligned `zerocopy` containers ([`DnamContainerView`](src/container/view.rs)), extracts Layer 0 `.dnam-bin` payloads ([`extract_dnam_bin`](src/container/view.rs)), performs zero-mutation prefix tier slimming ([`strip_container_to_tier`](src/container/strip.rs)), updates `DMU1`/`DMUE` mutable suffixes in place, and checks Castagnoli `CRC32C` digests.
  - Parses `#![no_std]` `.dnam` JSON headers ([`DnamJsonHeader`](src/schema/header.rs)), validates unknown keys and UI power-skew curves, and compiles control schemas into 64 B [`ControlClampDescriptor`](../vm/src/control/descriptor.rs) cache lines.
- **Host-Shell 8-Port Patchbay & Execution (`src/engine/`, `src/dynamic/`):**
  - Wraps isolated live and offline-bounce `dnam_vm::SonaVm` instances (`OFFLINE_BOUNCE_PDC_SAMPLES = 12`) inside [`DnamEngine`](src/engine.rs) and runtime-dispatched [`DynamicDnamEngine`](src/dynamic.rs) across Scalar, AArch64 NEON/SVE2/SME, x86-64 SSE2/SSE4.2/AVX2+FMA, and WASM SIMD128 backends.
  - [`HostShellPatchbay`](src/engine/patchbay.rs) implements Stereo, Mono, FourCableMethod (`4CM`), DualAmp, and WetDryWet (`WDW`) routing with structural return-to-send isolation, [`ReturnAligner`](src/engine/patchbay/align.rs) fractional return alignment, compile-time Kaiser-sinc [`ParallelAlignLine`](src/engine/patchbay/align.rs) sub-sample alignment, post-VM voltage-normalized Mid/Side stereo width, [`TempoSyncState`](src/engine/tempo.rs) tap-tempo synchronization, and stack-allocated preset batches.
- **Wait-Free Host Concurrency & C-ABI (`src/host/`, `src/ffi.rs`):**
  - Provides lock-free 3-lane SPSC control ingress (`SpscControlRing`, `AcceptedStateShadow`), 64 B-aligned Seqlock telemetry ([`TelemetryCell`](src/host/telemetry.rs), [`DnamTelemetrySnapshot`](src/types.rs)), non-blocking ping-pong model hot-swap ([`RealtimeEngineSwap`](src/host/swap.rs)), unified [`DnamHostHandle`](src/host/handle.rs), and 15 zero-allocation `dnam_engine_*` `extern "C"` exports ([`src/ffi.rs`](src/ffi.rs)).
- **Marketplace Provenance, Hardware DRM & Watermarking (`src/provenance/`):**
  - Implements preset-safe RFC 8785 JCS canonicalization, Ed25519 prefix commitments (`DAU1`/`DAI1`/`DSA1`) with fail-closed `DistributionPolicy` enforcement, `OemDeviceCertificate` verification, `.dnam-seal` (`DNSL`/`DNPL`/`DME1`) RFC 9180 HPKE + ChaCha20-Poly1305 AEAD framing, HMAC-SHA256 seed derivation, 224-D gauge-invariant structural hashing, multi-layer buyer watermarking (40-lane SONA sign-flip gauge, pole-angle phase coding, and disjoint 7,680-coord Tardos codes), and `.dnam-prev` containment via real-time [`PreviewGuardState`](src/provenance/preview_guard.rs).

---

## Floating-Point Scope & Layer 0 VM Budgets

Block-processing APIs establish and restore the audio floating-point environment automatically. Direct per-sample engine calls require one [`FtzDazGuard`](src/types.rs) around the entire batch; see the [audio-callback policy](../../docs/specs/architecture/sona_architecture.md#floating-point-environment-at-the-audio-callback).

Layer 0 DSP kernels, VM state RAM budgets, 2×/4× oversampling, and 3 kHz `ControlBus` evaluation live in [`dnam-vm`](../vm/README.md) and [`dnam-kernels`](../kernels/README.md); see the [hardware budget summary](../../docs/specs/architecture/sona_architecture.md#22-hardware--dsp-benchmark-summary).

---

## Package Structure

```text
packages/engine/
├── spec_scope.toml         # Manifest scoping every canonical r[...] spec rule
├── src/
│   ├── lib.rs              # Crate root (#![no_std] + alloc, #![deny(unsafe_code)]) & re-exports
│   ├── types.rs            # Tier, FtzDazGuard, TELEMETRY_CONTROL_SLOTS, DnamTelemetrySnapshot
│   ├── memory.rs           # 64B CacheLine64, AlignedFp16Block, AlignedBuffer page pre-faulter
│   ├── ffi.rs              # 15 zero-allocation extern "C" exports (dnam_engine_*)
│   ├── engine.rs & engine/ # DnamEngine<'a, B>, HostShellPatchbay, Aligners, Presets, TempoSync, Bounce
│   ├── dynamic.rs & dynamic/ # Runtime-dispatched DynamicDnamEngine<'a>
│   ├── host.rs & host/     # 3-lane SPSC rings, AcceptedStateShadow, Seqlock telemetry, DnamHostHandle
│   ├── container.rs & container/ # v3.0 .dnam parser, builder, stripper, DMU1, .dnam-bin extractor
│   ├── provenance.rs & provenance/ # JCS, Ed25519 DAU1, DNSL HPKE+AEAD, Watermarking, PreviewGuard
│   ├── schema.rs & schema/ # #![no_std] serde/serde_json .dnam header, UI hints, Clamp compiler
│   └── test_support.rs     # #[cfg(test)] exact rational, PRNG & test container helpers
├── tests/                  # Integration suites (backend_parity, realtime_safety,
│                           # realtime_disassembly_audit, ffi_integration, marketplace_*,
│                           # sona_tier_and_multiport_lifecycle)
├── benches/                # Throughput & WCET suites (pipeline, container_and_provenance,
│                           # realtime_wcet)
└── examples/               # Runnable API examples (realtime_audio_block, embedded_zero_alloc,
                            # marketplace_provenance, container_slimming, c_abi_host_integration,
                            # custom_operator_chain)
```

---

## Quickstart: Running Examples, Tests & Benchmarks

```bash
# Quality gates & unit/integration tests
moon check engine
moon run engine:test
moon run engine:coverage

# Benchmarks (all suites or individual targets)
moon run engine:bench
cargo bench -p dnam-engine --bench realtime_wcet
cargo bench -p dnam-engine --bench pipeline
cargo bench -p dnam-engine --bench container_and_provenance

# Profiling with hotpath-rs (--profile profiling)
HOTPATH_LIMIT=0 HOTPATH_CRITERION_ITERATIONS=500 \
  cargo bench -p dnam-engine --profile profiling --bench pipeline --features hotpath-alloc \
  -- --test --exact "process_block_128/ARM64 NEON/ultra"
cargo bench -p dnam-engine --profile profiling --bench realtime_wcet --features hotpath-alloc

# Runnable Examples
cargo run -p dnam-engine --example realtime_audio_block
cargo run -p dnam-engine --example embedded_zero_alloc
cargo run -p dnam-engine --example marketplace_provenance
cargo run -p dnam-engine --example container_slimming
cargo run -p dnam-engine --example c_abi_host_integration
cargo run -p dnam-engine --example custom_operator_chain
```

API demonstrations live in [`examples/`](examples/). The coverage gate is a minimum, not a bug-free guarantee; see the [engine test guidance](CONTRIBUTING.md#3-testing-coverage--95--benchmarking-commands).

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
