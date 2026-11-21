# `dnam-vm` — Zero-JSON `#![no_std]` SONA Real-Time Virtual Machine

**`dnam-vm`** (`packages/vm/`) is the zero-JSON, `#![no_std]` + `!alloc`-capable Layer 0 execution VM for **Dynamic Neural Audio Modeler (DNAM)** SONA models. It depends strictly on [`dnam-kernels`](../kernels/README.md) and `zerocopy`, cleanly separating deterministic real-time DSP execution from outer container wrappers (`dnam-engine`), JSON parsing, and offline training (`dnam-train`).

---

## Core Design Guarantees

- **Zero-Allocation `#![no_std]` Operation:** Compiles cleanly with `--no-default-features` on embedded targets, `wasm32-unknown-unknown`, `aarch64`, and `x86_64`, while optional `alloc` and `std` features enable convenient owned builders.
- **Sub-50 ns Zero-Copy `.dnam-bin` (`DNBN` / `DBRM`) Loading:** Validates declarative `#[repr(C)]` `zerocopy` headers, cache-line-aligned tier boundaries, Castagnoli `CRC-32C` digests, pipeline/meta flags, control descriptors, and `TSYN` tempo entries directly from immutable byte slices, and supports zero-mutation prefix truncation down to lower tiers (`REQ-5.6`).
- **Multi-ISA SIMD Execution (`MathBackend`):** Dispatches eight 32-bit float hardware backends (`ScalarBackend`, `NeonBackend`, `ArmSve2Backend`, `ArmSmeBackend`, `X86Sse2Backend`, `X86Sse42Backend`, `X86Avx2FmaBackend`, and `WasmSimd128Backend`) with lock-free cached probing and hardware `CRC-32C` acceleration.
- **Deterministic Parametric Control Bus (`ControlBus`):** Evaluates up to 255 physical controls, 3-slot switch-conditional range clamps, `CP1` + exact-integer 9-word/18-word sufficient-statistic accumulators, `Q0.192` Chebyshev LFOs, outward 4-D trust clipping, 32-channel target dispatch (`W_ctrl` plus optional `CEXT`), immediate write-enable snap on slot 16, and 12-subdivision `TSYN` tempo sync without `FDIV`, `FSQRT`, or `libm` calls.
- **Cache-Locked L1D `SonaHotSlab` & `SonaVm` / `DynamicSonaVm` Real-Time Execution:** Executes prefix-ordered `SonaHotSlab` arenas (4,544 to 14,656 bytes active hot footprint from Lite through Ultra) via 4-frame micro-chunks (zero-ULP chunk-split invariance across 1-frame and 16-frame schedules), `Op-N[Pre]` + `H1` + `T1` Matryoshka core and variable-projection readout, zero-lag 2x `SonaOs2xHot` and 4x `Bounce4xOversampler` (12 host-sample PDC), 8-port `IoRouter`, lag-ordered Cabinet Early FIR + `KautzStereoBank4`, and 224-byte `Op-T` Spatial Tank quads with dual independent `WE` latches and zero `FDIV`, `FSQRT`, `IDIV`, or `libm` calls.

---

## Module Architecture

| Module | Scope & Key Primitives |
| :--- | :--- |
| **[`src/lib.rs`](src/lib.rs)** | Crate root enforcing `#![cfg_attr(not(any(feature = "std", test)), no_std)]` and `#![deny(unsafe_code)]`. |
| **[`src/types.rs`](src/types.rs)** | Core execution primitives: `Tier` (`SonaTier`), `CacheLine64`, [`ColdBuf`](src/types/cold_buf.rs), [`VmError`](src/types/error.rs), and [`FtzDazGuard`](src/types/fp_env.rs) CSR guard. |
| **[`src/math.rs`](src/math.rs)** | Single-precision `SimdVec4`, `MathBackend`, `Crc32cBackend`, [`kernels.rs`](src/math/kernels.rs), [`const_eval.rs`](src/math/const_eval.rs), [`fp16.rs`](src/math/fp16.rs), [`init_math.rs`](src/math/init_math.rs) (`!libm` ZOH math), [`spec_math.rs`](src/math/spec_math.rs), [`svf_math.rs`](src/math/svf_math.rs), [`detect.rs`](src/math/detect.rs), and ISA modules [`aarch64.rs`](src/math/aarch64.rs), [`x86.rs`](src/math/x86.rs), [`wasm.rs`](src/math/wasm.rs). |
| **[`src/bin.rs`](src/bin.rs)** | Declarative `#[repr(C)]` `zerocopy` wire structures (`DnamBinHeaderWire`, `DnamBinRuntimeMetaWire`, `ControlClampDescriptorWire`, `TempoSyncEntryWire`, `CextDispatchWire`, and Arena zero-through-three payloads), [`view.rs`](src/bin/view.rs) parser & [`validate.rs`](src/bin/view/validate.rs) / [`arena.rs`](src/bin/view/validate/arena.rs) zero-mutation tier stripper, and [`builder.rs`](src/bin/builder.rs). |
| **[`src/control.rs`](src/control.rs)** | Parametric control plane under `#[forbid(unsafe_code)]`: [`descriptor.rs`](src/control/descriptor.rs), [`exact_int.rs`](src/control/exact_int.rs), [`cp1.rs`](src/control/cp1.rs), [`tempo.rs`](src/control/tempo.rs), and [`bus.rs`](src/control/bus.rs) (`ControlBusBacking`, [`load.rs`](src/control/bus/load.rs), [`tick.rs`](src/control/bus/tick.rs), [`dispatch.rs`](src/control/bus/dispatch.rs), [`laws.rs`](src/control/bus/laws.rs)). |
| **[`src/graph.rs`](src/graph.rs)** | Zero-alloc L1D DSP graph under `#[forbid(unsafe_code)]`: [`zoh.rs`](src/graph/zoh.rs), [`router.rs`](src/graph/router.rs), [`oversample.rs`](src/graph/oversample.rs), [`cabinet.rs`](src/graph/cabinet.rs), [`tank.rs`](src/graph/tank.rs), [`slab.rs`](src/graph/slab.rs), and [`GraphBackend` / `CabinetBackend`](src/graph.rs). |
| **[`src/vm.rs`](src/vm.rs)** | Public [`SonaVm`](src/vm.rs) & [`DynamicSonaVm`](src/vm/dynamic.rs) runtime dispatchers, [`SonaVmBacking`](src/vm.rs), [`SonaTelemetrySnapshot`](src/vm.rs), 2-port/4-port/offline-bounce stepping, 48-sample tier crossfade, and real-time disassembly roots. |
| **[`tests/dnam_bin_and_control_bus.rs`](tests/dnam_bin_and_control_bus.rs)** | End-to-end zero-copy `.dnam-bin` parsing/truncation latency, multi-tier invariance, `!alloc` `ControlBus`, `CEXT` dispatch, and `TSYN` calibration verification. |
| **[`tests/sona_vm_pipeline_and_parity.rs`](tests/sona_vm_pipeline_and_parity.rs)** ([`m9_contracts.rs`](tests/sona_vm_pipeline_and_parity/m9_contracts.rs)), **[`tests/realtime_disassembly_audit.rs`](tests/realtime_disassembly_audit.rs)** & **[`benches/realtime_wcet.rs`](benches/realtime_wcet.rs)** | Zero-ULP chunk-split invariance, Matryoshka tier equivalence, `SO(4)` `Q_buyer` gauge invariance, dual `WE` latches, cross-ISA parity, release `objdump` RT disassembly gate, and non-Criterion WCET + sub-50 ns bind latency harness. |

---

## Quickstart & Quality Gates

```bash
moon check vm
cargo check -p dnam-vm --no-default-features
cargo check -p dnam-vm --no-default-features --features alloc
cargo check -p dnam-vm --no-default-features --target wasm32-unknown-unknown
moon run vm:coverage
moon run vm:bench
```

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
