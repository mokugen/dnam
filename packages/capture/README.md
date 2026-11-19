# `dnam-capture` — Pure-IO Capture, `.dnam-proj` (`DNPJ`) Container & `VarPro` Engine

**`dnam-capture`** (`packages/capture/`) is the 100% safe-Rust (`#![forbid(unsafe_code)]`, `wasm32`-safe) pure IO-buffer capture, single-file `.dnam-proj` (`DNPJ`, `v3.0` `"sona"` `zerocopy` cutover in Phase 5) project container, and `SONA` (`32-bit WebGPU` + Host-`f64` `VarPro`) system-identification crate for **Dynamic Neural Audio Modeler (DNAM)**.

By linking [`dnam-engine`](../engine/README.md) directly (`engine <- capture` `[dependencies]`, `fixtures <- capture` `[dev-dependencies]`), `dnam-capture` executes the exact `#![no_std]` + `alloc` DSP operators (`Op-T`, `Op-M`, `Op-N`), `SONA` `v3.0` (`TPT-SVF` + `1D ParaLRU` + `PolyADAA1` + `Stage III FIR+Kautz`) + `32-bit WebGPU` / `VarPro` kernels (`Phases 4–5`), `3 kHz` Parametric Control Bus, and `.dnam` binary framing across native hosts (`wgpu` / CPU `DynamicTrainer`), Node.js `N-API`, Browser `WebWorker`s, and Cloudflare Workers (`ref[arch.isomorphism.two-sweep]`, [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md), [`ADR-0020`](../../docs/adr/0020-modular-monorepo-and-dnam-proj-container.md), [`ADR-0022`](../../docs/adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

## Key Capabilities

- **Single-File Cloud-Ready `.dnam-proj` (`DNPJ`, `v3.0` `zerocopy` Target) Container & Two-Path Capture Vault ([`src/proj.rs`](src/proj.rs)):**
  - Front-loaded `64 B` Preamble + `64 B` Chunk Directory (`CALB`, `SPCM`, `INPC`, `STMC`, `STAT`, `WGHT`, `AUTH` with `crc32c` + `sha256`) + `64 B`-aligned RFC 8785 JCS `DnamProjManifest` inspectable via a single `HTTP Range: bytes=0-65535` read (`parse_proj_manifest_range`, [`project_container_format.md`](../../docs/specs/containers/project_container_format.md)).
  - Compositional Capture Vault (`add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`), `< 1 ms` raw `SPCM` stripping (`compact_to_capsules`), and `< 0.2 ms` runtime `.dnam` emission (`export_runtime_dnam`).
- **Closed-Loop `AdaptiveStimulusController` & Unified Dual-Mode API ([`src/stimulus.rs`](src/stimulus.rs)):**
  - `< 1.5 ms` per `96 ms` cyclic frame `A1..A6` synthesis, `Laws 0.A–0.F` hardware safeguards (including `Law 0.F` sub-`0 dBFS` microphone capsule / external preamp clip protection & `recapture_clipped_stem`), `Laws 1–3` + `Theorem 4.5–4.8` $D$-optimal active learning & Two-Episode Pump-and-Probe Bloom, Mode-1 streaming (`train_adaptive_target` + `1.0s` `run_live_null_test_and_autopatch`), and Mode-2 sans-IO batch (`generate_stimulus_batch`, `generate_refinement_stimulus_batch`, `solve_from_responses`).
- **5-Bit-Depth In-Memory WAV Codec & 4-Regime Sub-Sample Alignment ([`src/align.rs`](src/align.rs), [`src/align/regimes.rs`](src/align/regimes.rs)):**
  - Buffer-in/buffer-out `Int12`/`Int16`/`Int24`/`Int32`/`Float32` WAV PCM encoding/decoding, `<f4` SHA-256 provenance hashing, Phase-0 `0.5s` `CALB` deconvolution, and `<= 0.02`-sample 4-regime Kaiser-sinc alignment (`CleanLinear`, `PolarityInverted`, `DispersiveAllPass`, `NonlinearEnvelopeFallback`, [`capture_and_alignment.md`](../../docs/specs/training/capture_and_alignment.md)).
- **`32-Bit WebGPU` + Host-`f64` `VarPro` & `uplo='U'` `MM-Cholesky` Solver ([`src/solver.rs`](src/solver.rs)):**
  - `i32` Ozaki (`3×8`-bit slices) / `Wide192Accum` Gram accumulation (`uplo='U'`) and regularized `f64` `MM-Cholesky` + `33`-band `LTAS` KKT solver optimizing weights into the fixed compile-time `const` `SONA` tiers (`Lite ⊂ Full ⊂ Pro ⊂ Ultra` locked at `Gate 0b`, full `2.0–2.5 s` capsule trajectories, continuous `Kautz`/`Op-T` polish to `BestUntilDiminishingReturns`, [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md)).
- **4-Metric Psychoacoustic Certification, Resampler & `DNWB` White-Box Distillation ([`src/metrics.rs`](src/metrics.rs), [`src/resampler.rs`](src/resampler.rs), [`src/whitebox.rs`](src/whitebox.rs)):**
  - Evaluates noise-qualified 3-band `bESR`, 5-scale $\times$ 32-probe resonator bank `MR-IIR`, 33-band `1/3`-octave Welch `LTAS`, and `SegESR`, plus zero-phase Kaiser-sinc `8..192 kHz` $\leftrightarrow$ `48 kHz` resampling and `64 B` `DNWB` (`FaustRunner`, `SpiceRunner`) pipe execution ([`ADR-0018`](../../docs/adr/0018-signal-chain-fixtures-crate-and-whitebox-distillation.md)).

---

## Package Structure

```text
packages/capture/
├── Cargo.toml                  # dnam-capture pure library manifest (zero CLI / OS audio)
├── moon.yml                    # Rust quality, test, coverage (>= 95%) & bench tasks
├── src/
│   ├── lib.rs                  # Crate root (#![forbid(unsafe_code)]), Gram hash & fp16 helpers
│   ├── align.rs                # 5-bit-depth WAV PCM codec, <f4 SHA-256 & sub-sample alignment
│   ├── align/
│   │   └── regimes.rs          # CALB deconvolution, 4-regime Kaiser-sinc align & null/thermal/pad gates
│   ├── stimulus.rs             # AdaptiveStimulusController (<1.5ms/96ms frame) & capsule sealing
│   ├── stimulus/
│   │   ├── fft.rs              # Radix-2 f64 FFT/IFFT, disjoint-comb scheduler & Schroeder phases
│   │   ├── waveforms.rs        # A1..A6 primitives, Theorem 4.7-4.8 Bloom & fp16 coreset/PSD encoders
│   │   ├── safeguards.rs       # Laws 0.A-0.F hardware safeguards & CaptureWarning
│   │   ├── laws.rs             # Laws 1-3, Theorem 4.5-4.6 sweeps/topologies & 4-Gate Cramér-Rao stop
│   │   ├── pipeline.rs         # Mode 1 train_adaptive_target & 1.0s Live Null-Test / <5ms Auto-Patch
│   │   └── batch.rs            # Mode 2 sans-IO batch API & 400ms mic-pad recapture_clipped_stem
│   ├── targets.rs              # DistillationTarget, CaptureBitDepth & stem batch types
│   ├── resampler.rs            # Zero-phase Kaiser-sinc 8..192 kHz <-> 48 kHz resampler
│   ├── audit.rs                # 3-Phase Control-Space Audit range/switch/corner/stability clamps
│   ├── proj.rs                 # .dnam-proj (DNPJ) constants, uplo='U' & fp16 helpers (v3.0 in Phase 5)
│   ├── proj/
│   │   ├── schema.rs           # DnamProjManifest, StemRegistryEntry (enabled) & JCS digest
│   │   ├── directory.rs        # 64B Preamble, 64B Chunk Directory, Range parser & AUTH
│   │   ├── capsules.rs         # CALB (64B + IR), INPC (24.6/48.6 KB), STMC (28.8/52.8 KB) & STAT
│   │   ├── bundle.rs           # DnamProjView, builder, compact_to_capsules & export_runtime_dnam
│   │   └── vault.rs            # Two-Path Capture Vault (add/remove/replace/set_capture_enabled)
│   ├── metrics.rs              # Degree-7 Bezout 3-band bESR, noise qualification & interval audit
│   ├── metrics/
│   │   ├── companion.rs        # 5-scale 32-probe MR-IIR, 33-band 1/3-octave LTAS & SegESR
│   │   └── certification.rs    # PerceptualCertificationSummary & 4-tier non-regression
│   ├── whitebox.rs             # 64B DnwbHeader64B binary request/response wire framing
│   ├── whitebox/
│   │   ├── runners.rs          # SpiceRunner (adaptive ODE + PCHIP 4x + 309-tap Kaiser-sinc)
│   │   ├── faust.rs            # FaustRunner recursive faust -json UI parser & DSP runner
│   │   ├── ipc.rs              # Concurrent non-deadlocking DNWB stdin/stdout subprocess transport
│   │   └── session.rs          # Persistent stateful DnwbWorkerSession & stream server
│   ├── solver.rs               # uplo='U' Gram accumulation, Cholesky solver & bESR helpers
│   └── solver/
│       └── stages.rs           # AuditBounds, Step-0 variance gating & Family-5 surrogate
├── tests/
│   ├── proj_suite.rs                # .dnam-proj Range, DAU1/AUTH & vault suites (proj_suite/*)
│   ├── stimulus_and_align_suite.rs  # Stimulus, alignment, bit-depth & spec suites (stimulus_and_align_suite/*)
│   ├── solver_suite.rs              # VarPro oracles, chains, adversaries & audit suites (solver_suite/*)
│   └── qualification_matrix.rs      # Full 13x4 & v3 qualification (--ignored; run via capture:qualify)
└── benches/
    └── bench_capture.rs        # Criterion Cholesky, compact_to_capsules & export_runtime_dnam
```

---

## Quickstart

```bash
# Run all capture quality gates (format-check, clippy, typecheck, build, test)
moon check capture

# Enforce >= 95% line coverage
moon run capture:coverage

# Run full multi-chain --release acoustic qualification matrices
moon run capture:qualify
```
