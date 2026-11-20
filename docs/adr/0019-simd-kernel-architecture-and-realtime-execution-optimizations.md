---
id: ADR-0019
title: "Two-Level SIMD Trait Architecture, Single-Library Multi-ISA Dispatch & Real-Time Loop Optimizations"
status: accepted
date: "2026-11-10"
deciders:
  - DNAM Architecture Team
supersedes: []
superseded_by: null
specs:
  - "../specs/architecture/sona_architecture.md"
  - "../specs/architecture/project_and_documentation_quality.md"
  - "../specs/dsp/op_m_modal_kautz.md"
  - "../specs/dsp/op_n_paralru_and_polyadaa.md"
  - "../specs/dsp/op_t_transport.md"
  - "../specs/dsp/router_and_crossbar_topology.md"
  - "../specs/dsp/parametric_control_bus.md"
  - "../specs/sdks/rust_sdk.md"
  - "../specs/sdks/typescript_sdk.md"
requirements:
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
---

# ADR-0019: Two-Level SIMD Trait Architecture, Single-Library Multi-ISA Dispatch & Real-Time Loop Optimizations

## Context and Problem Statement

To meet `REQ-6.1` (`≥ 14× Lite`, `≥ 8× Full`, `≥ 4× Pro`, and `≥ 2× Ultra` simultaneous `48 kHz` instances on a single `2.4 GHz` ARM Cortex-A76 core), `REQ-6.2` (zero pipeline-stalling `FDIV`, `FSQRT`, `IDIV`/`UDIV`/`SDIV`, `libm`/`fmodf` calls, or unbounded loops), and `REQ-6.3` (zero real-time allocations, zero blocking locks or uncached OS syscalls, and deterministic subnormal immunity) while maintaining `0.000000 ULP` bit parity across `ARM64` (`NEON`, `SVE2`, `SME`), `x86_64` (`SSE2`, `SSE4.2`, `AVX2+FMA`), `wasm32` (`SIMD128`), and `ScalarBackend`, how should `dnam-engine` structure its SIMD trait hierarchy, multi-ISA binary dispatch, stereo operator vectorization, direct-register `VarPro` readout collapse, and CI disassembly verification so real-time DSP code is implemented optimally the first time?

## Decision Drivers

- **Strict One-Way Dependency DAG (`engine -> stages -> operators -> math -> dnam-kernels`):** Low-level register abstractions in `math/` must never depend on higher-level operator or stage structs, yet wide-ISA backends (`AVX2`, `ARM SME`) must be able to override multi-vector stage/head kernels (`ref[arch.simd.two-level-traits]`).
- **Single Binary Per Architecture & Zero-Syscall Dispatch:** Shipping separate `.so`/`.dylib`/`.dll` builds per sub-ISA complicates distribution, while calling `sysctlbyname`/`getauxval`/`mrs` on the audio path violates `REQ-6.3` (`ref[arch.simd.single-library-dispatch]`).
- **Cross-Backend `0.000000 ULP` Bit Parity:** Hardware `FMA` vs. unfused multiply-add in sensitive Newton-Raphson iterations (`rsqrt_nr2_pair_f32`) or divergent control-plane math can cause ULP drift across architectures (`ref[arch.simd.single-library-dispatch]`).
- **True-Stereo & Multi-Lane Arithmetic Density:** Naive per-channel scalar loops waste `50%`–`75%` of 4-lane/8-lane SIMD registers, suffer per-tap modulo index overhead in FIR and `fp16` delay buffers, and recompute static control/head projections on every `48 kHz` sample.
- **Compiler Lowering Pitfalls (`bl _fmodf`, `udiv`, `fdiv`):** High-level Rust constructs such as `f32::rem_euclid`, `.is_multiple_of(self.block_size)`, and runtime `1.0 / (len - 1)` silently lower to external `libm` calls or variable-latency division instructions unless prohibited by design and caught by automated release-binary disassembly (`ref[arch.simd.zero-stall-lowering]`, `ref[sdk.rust.realtime-disassembly-audit]`).

## Considered Options

1. **Option 1 — Monolithic `MathBackend` Trait, Separate Per-ISA Binaries & Per-Channel Scalar Loops:** Put all operator/graph methods on a single trait in `math/`, compile separate binaries per CPU feature level, evaluate Left and Right channels sequentially, and recompute full-dimensional readout projections per audio sample.
2. **Option 2 — Macro-Generated SIMD Boilerplate & Source-Only Regex Linting:** Generate SIMD backends with `macro_rules!`, rely on compiler auto-vectorization for stereo/FIR/head loops, and check only source text for `/` or `sqrt`.
3. **Option 3 (Chosen) — Two-Level Domain-Scoped SIMD Trait Hierarchy, Single-Library `#[target_feature]` + `AtomicU8` Dispatch, Stereo/Pre-Collapsed Kernels & Automated Release Disassembly Audit.**

## Decision Outcome

Chosen option: **Option 3**, organized across five normative architectural pillars (`ref[arch.simd.two-level-traits]`, `ref[arch.simd.single-library-dispatch]`, `ref[arch.simd.zero-stall-lowering]`, `ref[sdk.rust.realtime-disassembly-audit]`, `ref[sdk.rust.explicit-simd-and-coverage]`):

### Pillar 1 — Two-Level SIMD Trait Hierarchy & Domain-Scoped Backend Traits (`ref[arch.simd.two-level-traits]`)

| Hierarchy Level | Trait & Module Location | Architectural Role & Key Primitives |
| :--- | :--- | :--- |
| **Level 1 (Register & Core Math)** | `SimdVec4` & `MathBackend` (`math/backend.rs`), `Crc32cBackend` (`math/crc32c.rs`) | Portable 4-lane `f32` register (`fma_lane::<const LANE: usize>`, `horizontal_sum_pair`, `splat_pair`, `pair_swap_neg`, `flush_subnormal`, paired with `flush_recursive_state_vec4`) plus `MathBackend` (`fma_mat4x4_vec4`, `f32x4_to_fp16x4`, `fp16x4_to_f32x4`, `rsqrt_nr2_f32`, `rsqrt_nr2_pair_f32`, `fast_unit_phasor`, and hardware `CRC32C`). |
| **Level 2 (Domain-Scoped Traits & Free Kernels)** | `dnam-kernels` `KernelVec4`/`StereoVec8` (`packages/kernels/`) plus domain traits (`OpMBackend`, `OpNBackend`, `OpTBackend`, `Stage3Backend`) and blanket supertraits (`StageBackend`, `EngineBackend`) | Defined in their domain modules (`engine -> stages -> operators -> math -> dnam-kernels`) with portable `SimdVec4` defaults; `X86Avx2FmaBackend` overrides `StereoVec8` (`Avx2Stereo8` for `step_sona_group_stereo8`) and `Stage3Backend::dot16_stereo` (`avx2_dot16_stereo`), while `ArmSmeBackend` uses `SmeStereo8` (`InterleavedStereo8<NeonVec4>`), delegates `Stage3Backend::dot16_stereo` to `ArmSve2Backend`/`NeonBackend` (`dot16_stereo_vec4`), and reserves SME/SSVE intrinsics for Level 1 `TrainerMathBackend`. |

- **Hardware `FtzDazGuard` + Unconditional Recursive State Scrubbing (`ref[arch.fp.callback-env]`):** `FtzDazGuard` sets `FPCR.FZ` on `aarch64` and `MXCSR.FTZ`/`DAZ` on `x86_64` once at block entry so hardware instructions flush intermediate arithmetic subnormals, while `flush_recursive_state_f32` (`scrub_and_flush_f32_bits`) and `flush_recursive_state_vec4` unconditionally execute branchless bitwise/SIMD scrubbing (`NaN`/`±Inf -> +0.0`, subnormals -> signed `±0.0`) on recursive writebacks across all targets (`aarch64`, `x86_64`, `wasm32`, and scalar).

### Pillar 2 — Single-Library Per-Architecture SIMD Dispatch & `0.000000 ULP` Parity (`ref[arch.simd.single-library-dispatch]`)

| Target Architecture | Compiled Backend Modules (`packages/engine/src/math/`) | Runtime Detection & Caching (`AtomicU8`) |
| :--- | :--- | :--- |
| **`aarch64`** | `aarch64/neon.rs` (`NeonBackend`), `aarch64/sve2.rs` (`ArmSve2Backend`), `aarch64/sme.rs` (`ArmSmeBackend`) | Function-level `#[target_feature(enable = "sve2")]` / `sme` + `AtomicU8`-cached `hw.optional.arm.FEAT_SME` / `getauxval(AT_HWCAP2)` + `ID_AA64PFR1_EL1` fallback. |
| **`x86_64`** | `x86/sse2.rs` (`X86Sse2Backend`), `x86/sse42.rs` (`X86Sse42Backend`), `x86/avx2.rs` (`X86Avx2FmaBackend`) | Function-level `#[target_feature(enable = "avx2,fma,f16c")]` / `sse4.2` + `AtomicU8`-cached `cpuid` (`AVX2 + FMA + F16C + OSXSAVE + XGETBV`). |
| **`wasm32` & Fallback** | `wasm.rs` (`WasmSimd128Backend`), `scalar.rs` (`ScalarBackend`) | Compile-time `target_feature = "simd128"` with zero-copy shared `WebAssembly.Memory` (`--import-memory`, `-zstack-size=786432`, `[profile.release]` / `[profile.profiling]` `opt-level = 3`, `lto = "thin"`, `codegen-units = 16`, `panic = "abort"`, `wasm-opt -O1`, `ref[sdk.ts.engine.zero-copy-and-async]`). |

- **Unfused `rsqrt_nr2_pair_f32` for Cross-ISA `0.000000 ULP` Parity:** All SIMD backends evaluate the 2-step Newton-Raphson `rsqrt` refinement $\rho_{j+1} = \rho_j(1.5 - (0.5 r^2 \rho_j)\rho_j)$ using separate IEEE-754 multiply and subtract instructions (`vmulq_f32`/`vsubq_f32`, `_mm_mul_ps`/`_mm_sub_ps`, `f32x4_mul`/`f32x4_sub`) rather than fused `vfmsq_f32`/`vfnmadd`, eliminating single-ULP `FMA` rounding drift against `ScalarBackend`.

### Pillar 3 — Stereo Vectorization & Operator Algorithmic Optimizations

1. **`Op-P`, `Op-M`, & Cabinet Kautz (`ref[dsp.op-m.recurrence.stereo-simd]`, `ref[dsp.op-m.rotor.angle-wrap-and-retune]`, `ref[dsp.op-m.tracker.stereo-batching]`):**
   - `HilbertH1Lattice6Sec::step_stereo` (`operators/op_m/hilbert.rs`) packs `[I_L, Q_L, I_R, Q_R]` into one `B::Vec4` register across all 6 `TDF-II` all-pass sections (`4×` reduction in lattice loops), while `SonaHotSlab` embeds mono `dnam-kernels` `HilbertH1Kernel` inside `compute_op_p_and_phasor_pair`.
   - `T1SubHarmonicKernel` (`dnam-kernels`) computes pure-`f32` `rsqrt_pos_nr2` normalization and branch-cut-guarded octave dividers (`0` `FSQRT`/`FDIV` calls) on mono input `0.5 * (s0_l + s0_r)` feeding crossbar `Op-P` lanes `0..4`, while `T1SubHarmonicTracker::step_analytic_stereo` evaluates standalone L/R pairs.
   - `KautzModalBank<4>::step_4lane_stereo_simd_unscaled` steps L/R shared-pole complex modal states together in 4-pole `B::Vec4` blocks, and `retune_kautz_slice_3khz` (`retune_slab_kautz`) rotates poles via `algebraic_unit_rotor` + `rsqrt_pos_nr3` orthonormal rescaling.
2. **`Op-N` `SONA` Core (`PolyADAA1` + `ParaLRU` + Passive `TPT-SVF`, `ref[dsp.op-n.polyadaa.stereo-and-rail-guard]`, `ref[dsp.op-n.core.stereo-simd]`):**
   - Pure-`f32` `PolyADAA1` evaluates interior samples (`|u| <= 1`) branch-free with zero reciprocal calls (`eval_interior_odd_even_*`) and computes rail-crossing reciprocals via `rcp_pos_nr2` in `StereoVec8` (or `KernelVec4::rcp_pos_quad_nr2` in `eval_poly_adaa1_quad_batched`).
   - `process_tick16_const` / `step_sona_group_stereo8` steps `StereoVec8` (or interleaved `2 × SimdVec4`) through passive `2×2` ZDF `TPT-SVF` + contractive `1D` `ParaLRU` + `PolyADAA1` across `16`-frame (`32` @ `2×` OS) control ticks inside `SonaHotSlab`.
3. **Cabinet Mirrored & Masked FIR History (`ref[dsp.cabinet.mirrored-fir-simd]`):**
   - `SonaHotSlab` maintains a 64-entry mirrored history on `Lite` (`fir_hist_l`), a 64-entry `& 63` ring on `Full`, and a 256-entry mirrored history on `Pro`/`Ultra` (`fir_hist_l_pro`), evaluating 16-tap slices via `dot16_ring_seg` + `SimdVec4::horizontal_sum_pair` (copying wrapped `Full` slices where start > 48 into a 16-element stack slice), while `Stage3Backend::dot16_stereo` backs standalone stereo cabinet FIR evaluation.
4. **`Op-T` & Spatial Tank (`ref[dsp.op-t.reader.contiguous-fp16x4]`, `ref[dsp.tank.simd-loop]`):**
   - Standalone `OpTDelayLine::read_delay_stereo` uses contiguous 4-lane `MathBackend::fp16x4_to_f32x4` hardware loads whenever `p_0 >= 2 && p_0 + 1 < HALF_CAPACITY` (`1,024` for `OP_T_LITE_RING_CAPACITY = 2,048`), while the FDN tank (`V3Stage4TankHot`) inline `Quad 0` uses frame-interleaved `[[u16; 4]; 128]` `InlineFrameRing::load_quad`.
   - `Stage4QuadHotCompact::step_subblock` and `step_stereo` (`V3Stage4TankHot`) vectorize the 4-line tank DC blockers, 4-lane rational feedback knee and cubic loop saturator (`step_loop_feedback`; standalone `Op-T` uses `coercive_loop_limit_vec4`), cached-angle Givens `pair_swap_neg` (`|target_psi - current_psi| <= 1e-7`), and 4-channel Householder reflection in `B::Vec4`.

### Pillar 4 — Contiguous `SonaHotSlab`, `3 kHz` Unified `W_ctrl` Dispatch & Cross-CGU Compile Hygiene (`ref[dsp.head.precollapse-and-zerocopy]`, `ref[dsp.ctrl.op-n.static-and-interior-fastpath]`, `ref[dsp.ctrl.dispatch.sparse-and-active-ramps]`)

1. **Contiguous Cache-Locked `SonaHotSlab` (`4,608 / 7,936 / 15,872 / 28,672 B`):** `SonaInstance` executes directly inside `SonaHotSlab` (`stage0_*` crossbar state + `Op-P` + `4× Q0.32` -> `SONA` Core groups/tiles -> early FIR/Kautz cabinet -> `V3Stage4TankHot` inline `Quad 0` `128`-frame `fp16` ring) with `0 B` transient heap allocations in real-time audio callbacks.
2. **Load-Time `W_readout` Unpacking & `3 kHz` Register Hoisting:** Static `fp16` `W_readout` (`VarProReadout*Wire`) unpacks at load/tier-switch into `SonaHotSlab` tier `ro_groups` (`lite.ro_groups`, `full_delta.ro_groups_full`, `pro_delta.ro_groups_pro`, `ultra_delta.ro_groups_ultra`), and operator modulation tables unpack into `SonaControlBusParams`, which `hoist_groups_and_readouts` decodes into Left/Right `B::Stereo8` `SonaGroupReadoutRegs` alongside modulated `SonaGroupParamsStereo4` at sub-block/frame entry (`ControlBus` evaluates `z_head`/`z_stage` and `stage_targets[0..16]`), avoiding per-sample projection MACs.
3. **`3 kHz` Unified Control Plane Fast Paths:** `ControlBus<B>` caches converted `Q0.192` LFO phase increments, skips static basis evaluation when `!is_dirty`, exits `trust_clip_outward_4d` early when `norm_sq + 1e-9 < r^2`, evaluates all 16 rows of `W_ctrl` (`+ Ultra 16×4 W_ctrl_ultra`) over unified `q16 = [z_head[0..4], ctrl_basis_4[0..4], lfo, env_states[0..4]]` in `ControlBus::evaluate_tank_modulation` (optional OU drift `[2, 5, 10, 14]`), and computes micro-ramp deltas via `4 × B::Vec4` in `compute_micro_ramp_deltas` (while runtime stepping applies scalar clamped `micro_ramp_advance_clamped` over `0..16`; `ControlBus::step_micro_ramp` provides `4 × B::Vec4` stepping).
4. **L1i-Resident Frame Kernel & Cross-CGU Monomorphization Deduplication:** Marking `process_tick16_const`, `step_control_tick_3khz`, and collapsed readout updates `#[inline(never)]` keeps hot instruction working sets L1i-resident and cuts LLVM compile time/RAM, while routing `DnamJsonHeader` `serde_json` parsing/serialization through non-generic `#[inline(never)]` `from_json_slice` / `to_json_vec` helpers eliminates duplicated cross-CGU `serde` monomorphizations.

### Pillar 5 — Zero Pipeline Stalls, Zero Nondeterministic Calls & Automated Disassembly Gate (`ref[arch.simd.zero-stall-lowering]`, `ref[sdk.rust.realtime-disassembly-audit]`, `ref[sdk.rust.explicit-simd-and-coverage]`)

1. **Division-Free & `libm`-Free Lowering:** `floor_f32((theta + PI) * INV_TAU)` replaces `rem_euclid` (`bl _fmodf`) and `while` loops; incremental `(block_idx, in_block_offset)` counters replace `pos / block_size` (`udiv`); cached `inv_xfade_denom` and `newton6_reciprocal_f64` / `sqrt_f64` eliminate all `fdiv` and `fsqrt` instructions across `packages/engine`.
2. **Automated CI Disassembly & Call-Graph Audit (`realtime_disassembly_audit.rs`):** Disassembles release `libdnam_engine.rlib` in `moon run engine:test` to verify `0` `fdiv`/`fsqrt`/`udiv`/`sdiv`/`div`/`idiv`/`libm`/allocator/syscall instructions across all functions in the real-time transitive call-graph closure and zero `libm` relocations crate-wide.
3. **Explicit SIMD Implementations, Zero `coverage(off)` & Cross-ISA Verification:** Prohibits `macro_rules!` SIMD wrappers and `#[cfg_attr(coverage, coverage(off))]`, and verifies `NeonBackend`/`ArmSve2Backend`/`ArmSmeBackend` (`engine:test` + `engine:test-sve2` `backend_parity` and `sona_extended_pipeline`) alongside `X86Sse2Backend`/`X86Sse42Backend`/`X86Avx2FmaBackend` under Rosetta 2 (`engine:test-x86`).

### Consequences

- **Positive:** Provides compile-time monomorphized `SimdVec4` / `StereoVec8` vectorization, register-hoisted `16`-frame `process_tick16_const` loops, direct-register `W_readout` collapse, `0.000000 ULP` cross-backend parity across 8 backends, `>= 95%` line coverage without `coverage(off)`, and `0` forbidden `FDIV`/`FSQRT`/`IDIV`/`libm` instructions verified automatically in CI (real-time WCET harnesses implemented in `benches/realtime_wcet.rs` and `benches/pipeline.rs`; historical pre-SONA `v2.0` `N-API` and `WASM SIMD128` baseline tables archived in [`ADR-0013`](0013-typescript-sdk-architecture-and-dual-target-runtime.md#3-decision-outcome)).
- **Negative:** Requires maintaining explicit trait implementations per ISA family (`neon`, `sve2`, `sme`, `sse2`, `sse42`, `avx2`, `wasm`, `scalar`) and using `newton6_reciprocal_f64` / `floor_f32` instead of built-in `/` or `rem_euclid`.
