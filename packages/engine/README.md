# `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time SIMD Audio Engine

**`dnam-engine`** is the zero-audio-allocation, 16B/64B SIMD-aligned real-time inference, container, and provenance core for **Dynamic Neural Audio Modeler (DNAM)**.

It is engineered for embedded ARMv8.2-A/ARMv9.2-A pedalboards (**Rockchip RK3582 / Darkglass Anagram**), desktop **VST3/CLAP/AU** audio plugins, analytical **pure-Rust capture & training** (`packages/capture`), and **WebAssembly SIMD128 (`AudioWorklet`) / Node.js N-API** runtimes (`packages/js`, `packages/studio`).

---

## Architecture Overview

- **`#![no_std]` + `alloc` & Zero Real-Time Allocations:**
  - Compiles without `std` across all targets using `serde` and `serde_json` (`default-features = false, features = ["alloc"]`) and `zerocopy` (`0.8`).
  - Verified by the custom `#[global_allocator]` integration test ([`tests/realtime_safety.rs`](tests/realtime_safety.rs)), the release-binary disassembly audit ([`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs)), and [`spec_scope.toml`](spec_scope.toml) (`217` engine-scoped `r[...]` rules verified via `verify[...]`).
- **Zero `FSQRT`, `FDIV`, and `libm` Calls at $48\text{ kHz}$ & $3\text{ kHz}$:**
  - **Operators (`src/operators/`):**
    - **`Op-T` (Transport & Elastic Buffer):** Catmull-Rom cubic fractional delay with ZOH-rescaled dual-mode slew/crossfade, where $L_{\text{xfade}} = \max(16, \mathrm{round}(128/\rho_{\text{sr}}))$, branchless wrap, partitioned Schroeder diffusers, and dual-tap complementary Hermite-Hann grain reader satisfying $H(\phi_A) + H(\phi_B) \equiv 1$.
    - **`Op-M` (Modal Bank & Kautz):** 4-lane SIMD symplectic resonator, quadrant-reduced degree-13/14 Horner contractive rotor with adaptive 1–2 ULP guard, `Q0.192` LFO phase accumulator, 12-allpass (`6` I + `6` Q) `H1` Hilbert SSB lattice, `T1` sub-harmonic tracker with `5 Hz` upstream DC blocker, and Cabinet Bank (`cabinet`) 16-sample delayed Kautz modal bank.
    - **`Op-N` (Nonlinear Activation & Shaping):** Division-free Padé `[5/4]` `tanh` (`OpNBackend`), saturated `PolyADAA` reference verifier, analytical DC equilibrium, `C²` wavefolder, and `Q64.32` zero-order-hold bitcrusher with TPDF dither.
  - **Topology Subsystems, `SONA` State-Space Core & I/O Router (`src/graph/`, [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md)):**
    - **I/O Router (`io_router`) & Top-of-Frame Causal Crossbar:** 8-Port Matrix Router with return-to-send isolation, `200 Hz` `TPT` `LR4`/`Bèzout` crossover, Kaiser-sinc sub-sample alignment, WDW routing, hysteresis gate, plus (`SONA` `v3.0`) causal `Op-T` reads with $\tau \ge 4$, input-only `Op-P` (`HilbertH1Lattice6Sec` + `T1SubHarmonicTracker`), and `4`-lane `Q0.32` quadrature carriers (`0.05 Hz–20 kHz`).
    - **4-Block Matryoshka `SONA` Core (`ADR-0022`, [`SonaHotSlab`](src/graph/sona_core/slab.rs) / [`SonaInstance`](src/graph/sona_core.rs)):** Lower-triangular Matryoshka blocks (`Lite` `[1,1,1,1]` @ `1×` $\subset$ `Full` `[2,2,1,1]` @ `2×` $\subset$ `Pro` `[3,2,2,1]` @ `2×` $\subset$ `Ultra` `[4,3,2,1]` @ `2×`, `16 / 24 / 32 / 40` SVF modes = `4 / 6 / 8 / 10` quads, arenas `[4_608, 7_936, 15_872, 30_720]` B = `[72, 124, 248, 480]` CLs) combining passive `2×2` ZDF `TPT-SVF`, contractive `1D` scalar `ParaLRU` diode-RC ballistics, and pure-`f32` divided-difference `PolyADAA1` + `C²` wavefolder (`process_tick16_const` + `StereoVec8`, zero `FDIV`/`FSQRT`/`libm`).
    - **Cabinet Bank (`cabinet`) & Spatial Tank (`spatial_tank`):** Contiguous early reflection `FIR` (`32/48/96/128` target taps) and shared-pole orthonormal `Kautz` modal (`8/12/24/32` target pairs) with `ZOH` onset delay, plus coercive Spatial Tank `Op-T` quads (`1 / 2 / 3 / 4` quads = `4 / 8 / 12 / 16` feedback lines, `a ∈ {2, 4}` `fp16`/`f32` precision, integer-lag lossless freeze).
  - **Control Bus & Runtime UX (`src/control/`, [`ADR-0022`](../../docs/adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
    - $3\text{ kHz}$ node-anchored Lobatto basis, `CP1` tier-local projections (`8-D` Rank-4 in `Lite..Pro`, `16-D` Rank-8 in `Ultra`), 9-word/18-word `ExactIntFactorAccumulator`, outward-checked trust-region clipping, and `Q64`/`Q32` `20–400 BPM` tempo synchronization with 32-step binary-search physical calibration inversion ([`parametric_control_bus.md`](../../docs/specs/dsp/parametric_control_bus.md)).
    - Atomic preset batch evaluation with two-path output crossfading on disjoint conditional switch transitions (`apply_preset`), `SpilloverTrails` vs. `HardBypass` ramping (`set_bypass`), `supports_freeze` hardware freeze gating, out-of-range tempo-sync rank clamping (`active_clamp_mask`), and `OUTSIDE_CERTIFIED_INPUT` (`bit 4`) telemetry reporting above $\pm 100.0$ (`+40 dBFS`) without unmodeled soft-limiting.
- **Multi-Platform Hardware SIMD (`SimdVec4` & `MathBackend`):**
  - [`aarch64.rs`](src/math/aarch64.rs): Unified `aarch64` hardware library with runtime CPU feature detection (`ArmSmeBackend` with `FEAT_SME` $\to$ `ArmSve2Backend` with `FEAT_SVE2` $\to$ `NeonBackend` 128-bit NEON baseline).
  - [`x86.rs`](src/math/x86.rs): Unified `x86_64` hardware library with runtime `is_x86_feature_detected!` dispatch (`X86Avx2FmaBackend` `x86-64-v3` $\to$ `X86Sse42Backend` `x86-64-v2` $\to$ `X86Sse2Backend` `x86-64-v1`).
  - [`wasm.rs`](src/math/wasm.rs): Unified WebAssembly (`wasm32`) 128-bit SIMD (`v128`) backend (`WasmSimd128Backend`).
  - [`ScalarBackend`](src/math/scalar.rs): Portable 100% safe `#![no_std]` IEEE-754 reference backend.
- **Sample-Rate Conversion & Offline Bounce:**
  - [`src/container/zoh.rs`](src/container/zoh.rs): Continuous-Time ZOH parameter rescaling (`44.1–192 kHz`), Cabinet Bank (`cabinet`) first-tap/6-point Lagrange FIR resampling, and `4×` polyphase offline bounce ([`router_and_crossbar_topology.md` §9](../../docs/specs/dsp/router_and_crossbar_topology.md#9-continuous-time-zoh-rate-adaptation)).
- **Marketplace Provenance & Watermarking (`src/provenance/`):**
  - RFC 8785 JCS header canonicalization, `Ed25519` prefix commitments (`DAU1`/`DAI1`/`DSA1`) with fail-closed marketplace trust-anchor enforcement, `OemDeviceCertificate` V2, canonical `.dnam-seal` (`DNSL`/`DNPL`/`DME1`) RFC 9180 HPKE + `ChaCha20-Poly1305` AEAD (zero raw XOR), `HMAC-SHA256` watermark seed derivation, Cabinet Bank (`cabinet`) gauge-invariant modal fingerprinting, dual-mechanism buyer watermarking (`40`-lane `SONA` sign-flip, `Kautz` + `SO(4)` gauges, `[0.2, 0.8]` phase coding, and disjoint `7,680`-coord Tardos), and `.dnam-prev` (`container_kind: "preview"`) load/hot-swap rejection ([`provenance_and_enclave_drm.md`](../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../docs/specs/security/watermarking_and_marketplace.md)).

---

## Floating-Point Scope

Block-processing APIs establish and restore the audio floating-point environment automatically. Direct per-sample engine, operator, graph, or control-tick calls require one [`FtzDazGuard`](src/types.rs) around the entire batch. Low-level single-frame helpers step one stereo frame without block-level tempo advancement or `4×` offline bounce oversampling. See the [audio-callback policy](../../docs/specs/architecture/sona_architecture.md#floating-point-environment-at-the-audio-callback).

---

## `SONA` `v3.0` Target Budget Envelope (`48 kHz` Stereo, `128`-Sample Blocks)

In `SONA` `v3.0`, compile-time `const` tier dimensions (`LiteSpec`, `FullSpec`, `ProSpec`, `UltraSpec`) and constant-size `zerocopy` `Arena0LiteWire..Arena3UltraDeltaWire` layouts are sized for `realtime_wcet`, `realtime_disassembly_audit`, and `hotpath` efficiency (`process_tick16_const`, `StereoVec8`, zero-spill `SonaHotSlab`) to occupy `~85–90%` of each `REQ-6.1` `1× Cortex-A76` budget and `N × ActiveHot <= 64 KiB` L1D limit (benchmark via `cargo bench -p dnam-engine --bench realtime_wcet`; see [hardware budget requirements](../../docs/specs/architecture/sona_architecture.md#22-hardware--dsp-benchmark-summary)):

| Tier Profile | `REQ-6.1` `1× Cortex-A76` Ceiling (@ 80% RT Cap) | Target Core (`4`-Block `SONA` + `Cabinet Bank`) | Target `ActiveHot` Slab (`N × Slab ≤ 64 KiB` L1D) |
| :--- | :---: | :--- | :---: |
| **`Lite`** | **`≥ 14×`** (`≤ 1,190 ns`/frame) | `[1,1,1,1]` `16` SVF modes / `4` `TPT-SVF` quads (`1×`), `32` FIR + `8` Kautz, `1` `Op-T` tank quad (`4` lines) | **`≤ 4.5 KiB`** (`14× ≤ 63 KiB`) |
| **`Full`** | **`≥ 8×`** (`≤ 2,083 ns`/frame) | `[2,2,1,1]` `24` SVF modes / `6` `TPT-SVF` quads (`2×`), `48` FIR + `12` Kautz, `2` `Op-T` tank quads (`8` lines) | **`≤ 7,936 B` (`7.75 KiB`)** (`8× ≤ 62.0 KiB`, `63,488 B ≤ 64 KiB`) |
| **`Pro`** | **`≥ 4×`** (`≤ 4,166 ns`/frame) | `[3,2,2,1]` `32` SVF modes / `8` `TPT-SVF` quads (`2×`), `96` FIR + `24` Kautz, `3` `Op-T` tank quads (`12` lines) | **`≤ 15.5 KiB`** (`4× ≤ 62 KiB`) |
| **`Ultra`** | **`≥ 2×`** (`≤ 8,333 ns`/frame) + `4×` Bounce | `[4,3,2,1]` `40` SVF modes / `10` `TPT-SVF` quads (`2×` / `4×` bounce), `128` FIR + `32` Kautz, `4` `Op-T` tank quads (`16` lines) | **`≤ 28,672 B` (`28.0 KiB`)** (`2× ≤ 56.0 KiB`, `57,344 B ≤ 64 KiB`) |

---

## Package Structure

```text
packages/engine/
├── spec_scope.toml         # Manifest mapping all 322 r[...] spec rules (217 engine-scoped)
├── src/
│   ├── lib.rs              # Crate root (#![no_std] + alloc) & public API re-exports
│   ├── types.rs            # Tier (Lite/Full/Pro/Ultra), FtzDazGuard, DnamTelemetrySnapshot
│   ├── engine.rs           # Generic DnamEngine<'a, B> (4 I/O topologies, bounce, presets, hot-swap)
│   ├── engine/             # Block processing (process.rs) & runtime transitions (runtime.rs)
│   ├── dynamic.rs          # Runtime-dispatched DynamicDnamEngine<'a>
│   ├── dynamic/            # Backend dispatch & parity tests (dispatch.rs)
│   ├── memory.rs           # 64B CacheLine64, AlignedBuffer, ArenaStorage, DelayScratchStorage
│   ├── ffi.rs              # Zero-allocation extern "C" ABI (dnam_engine_*)
│   ├── test_support.rs     # #[cfg(test)] exact rational (Rat), SplitMix64 & ULP test harness
│   ├── math.rs & math/     # SimdVec4, MathBackend, spec_math.rs, NEON/SME/AVX2/WASM
│   ├── operators.rs & operators/ # Op-T (op_t.rs), Op-M (op_m.rs), Op-N (op_n.rs), reference.rs
│   ├── graph.rs & graph/   # I/O Router, SONA Core, Cabinet Bank & Spatial Tank
│   │                       # (topology.rs, router.rs, sona_core.rs, cabinet.rs,
│   │                       # tank.rs, weights.rs)
│   ├── control.rs & control/ # 3 kHz bus.rs, plane.rs (ExactIntFactorAccumulator, Q64/Q32 tempo)
│   ├── container.rs & container/ # wire.rs, preamble.rs, view.rs, zoh.rs, strip.rs
│   ├── provenance.rs & provenance/ # security.rs (DAU1/HPKE/DME1), seal.rs, watermark.rs
│   └── schema.rs & schema/ # #![no_std] serde/serde_json .dnam runtime header & UI skew
├── tests/                  # Cross-cutting integration tests (backend parity, 0-alloc, FFI, seal,
│                           # SONA pipeline/ZOH & calibration/lifecycle)
├── benches/                # Subsystem Criterion suites (pipeline, operators, graph_and_control,
│                           # container_and_provenance) & WCET jitter audit (realtime_wcet)
└── examples/               # 6 runnable API examples (realtime_audio_block, embedded_zero_alloc,
                            # marketplace_provenance, container_slimming,
                            # custom_operator_chain, c_abi_host_integration)
```

---

## Quickstart: Running Examples, Tests & Benchmarks

### Tests, Coverage & Benchmarks

```bash
# Quality gates & unit/integration tests
moon check engine
moon run engine:test
moon run engine:coverage

# Benchmarks (all suites or individual targets)
moon run engine:bench
cargo bench -p dnam-engine --bench realtime_wcet
cargo bench -p dnam-engine --bench pipeline
cargo bench -p dnam-engine --bench operators
cargo bench -p dnam-engine --bench graph_and_control
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
cargo run -p dnam-engine --example custom_operator_chain
cargo run -p dnam-engine --example c_abi_host_integration
```

API demonstrations live in [`examples/`](examples/). The coverage gate is a
minimum, not a bug-free guarantee; see the [engine test guidance](CONTRIBUTING.md#3-testing-coverage--95--benchmarking-commands).

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
