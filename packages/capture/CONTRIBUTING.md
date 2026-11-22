# Contributing to `dnam-capture` (`packages/capture/`)

This guide covers engineering, numerical accuracy, safety, testing, and coverage standards for **`dnam-capture`** ([`README.md`](README.md)), the pure-IO capture, `.dnam-proj` (`DNPJ` v3.0) container, and `SONA v3.0` (`dnam-train` WebGPU/CPU + `VarPro`) orchestration crate for **DNAM**.

All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Directory Layout

Consult [`AGENTS.md`](AGENTS.md), [`src/AGENTS.md`](src/AGENTS.md), [`src/align/AGENTS.md`](src/align/AGENTS.md), [`src/audit/AGENTS.md`](src/audit/AGENTS.md), [`src/stimulus/AGENTS.md`](src/stimulus/AGENTS.md), [`src/proj/AGENTS.md`](src/proj/AGENTS.md), [`src/metrics/AGENTS.md`](src/metrics/AGENTS.md), [`src/whitebox/AGENTS.md`](src/whitebox/AGENTS.md), [`src/solver/AGENTS.md`](src/solver/AGENTS.md), and [`tests/AGENTS.md`](tests/AGENTS.md):

- **`src/lib.rs`:** Crate root (`#![forbid(unsafe_code)]`, pure IO buffers, zero CLI/OS audio), `DNAM-GRAM-INPUT-V3` SHA-256 fingerprint (`compute_gram_input_fingerprint`), and IEEE-754 RNE `fp16` coreset quantization (`f32_to_fp16_bits`, `fp16_bits_to_f32`).
- **`src/align.rs`, `src/align/{wav_wire,regimes,calibration}.rs`, `src/targets.rs`, `src/resampler.rs` & `src/audit{,.rs,/manifest.rs}`:** 5-bit-depth (`Int12`/`Int16`/`Int24`/`Int32`/`Float32`) `zerocopy` in-memory WAV PCM codec, `<f4` SHA-256 provenance digest, Phase-0 `CALB` loopback deconvolution, 4-regime Kaiser-sinc sub-sample alignment (`<= 0.02` samples), `< 0.05 dB` acoustic null homing, `<= 0.15 dB` thermal recovery gate, `400 ms` mic-pad probe ratio, `DistillationTarget`, zero-phase Kaiser-sinc `8..192 kHz <-> 48 kHz` resampling, and Non-Stop 3-Phase Control-Space Audit helpers.
- **`src/stimulus.rs` & `src/stimulus/{fft,waveforms,safeguards,laws,physics_probe,pipeline,null_test,sealing,batch}.rs`:** `< 1.5 ms` closed-loop `AdaptiveStimulusController`, `A1..A6` excitation primitives, `Laws 0.A–0.F` hardware safeguards (`Law 0.F` sub-`0 dBFS` mic-capsule/preamp clip & `recapture_clipped_stem`), `Laws 1–3` & `Theorem 4.5–4.8` active learning, Mode-1 streaming (`train_adaptive_target` + `1.0s` `run_live_null_test_and_autopatch`), and Mode-2 sans-IO batch (`generate_stimulus_batch`, `generate_refinement_stimulus_batch`, `solve_from_responses`).
- **`src/proj.rs` & `src/proj/{schema,directory,capsules,bundle,vault}.rs`:** Single-file cloud-ready `.dnam-proj` (`DNPJ`, `v3.0` `"sona"` `zerocopy`) container (`parse_proj_manifest_range`, `DnamProjView`, `build_dnam_proj_container`, `compact_to_capsules`, `< 0.2 ms` `export_runtime_dnam`), `CALB`/`SPCM`/`INPC`/`STMC`/`STAT`/`WGHT`/`AUTH` chunk framing, `uplo='U'` packing, and Two-Path Capture Vault (`add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`).
- **`src/metrics.rs` & `src/metrics/{bezout,companion,certification,repeatability}.rs`:** 4-metric psychoacoustic certification suite (`bESR`, 5-scale 32-probe `MR-IIR`, 33-band Welch `LTAS`, `SegESR`) and 4-tier non-regression verification (`§4` minimum floors to surpass).
- **`src/whitebox.rs` & `src/whitebox/{wire,runners,faust,ipc,session}.rs`:** `64 B` `DnwbHeader64BWire` `zerocopy` & `postcard` pipe IPC framing, `FaustRunner`, `SpiceRunner` (`PCHIP` $4f_s$ + `309`-tap Kaiser-sinc $4\times \to 1\times$ decimation), and persistent `DnwbWorkerSession`.
- **`src/solver.rs` & `src/solver/{progress,stage_a_probe,stage_b_backbone,stage_c_knobs,stage_d_audit,stages}.rs`:** Column-major `uplo='U'` Gram accumulation (`i32` Ozaki on WebGPU / `Wide192Accum` on CPU via `dnam-train`), regularized `f64` `MM-Cholesky` + `33`-band `LTAS` KKT solver, `CP1` / `"CEXT"` control projection, sum-denominator Noise-Subtracted `bESR`, continuous `Kautz`/`Op-T` polish (`BestUntilDiminishingReturns`), and Non-Stop 3-Phase Control-Space Audit bounds.
- **`tests/*.rs`:** Integration, adversarial boundary, and `proptest!` verification suites (`verify[...]`, see [`tests/AGENTS.md`](tests/AGENTS.md)).
- **`benches/bench_capture.rs`:** Criterion throughput benchmarks for `uplo='U'` Cholesky solves, `compact_to_capsules`, and `export_runtime_dnam`.

---

## 2. Core Invariants

1. **100% Safe Rust (`#![forbid(unsafe_code)]`), `zerocopy` Wire Structs, Pure IO Buffers & File Hygiene:**
   - `dnam-capture` enforces `#![forbid(unsafe_code)]` across all library, test, and benchmark targets (`ref[repo.code.type-safety-and-imports]`, `ref[sdk.rust.safe-rust-and-boundaries]`), zero `mod.rs` files (`ref[sdk.rust.modular-layout-and-proptest]`), zero `clap` or `cpal` dependencies, `< 900` lines per `.rs` file, and declarative `zerocopy` wire structs (`ref[sdk.rust.zerocopy-wire-structs]`, `.agents/rules/zerocopy-wire-structs.md`) with `postcard` IPC (`ADR-0023`).
2. **Exact Engine-Capture Isomorphism & `100%` Black-Box Solver Purity (`ref[arch.isomorphism.two-sweep]`, `ref[arch.isomorphism.audit-wiring]`):**
   - Never duplicate DSP operator math (`Op-T`, `Op-M`, `Op-N`), `3 kHz` control-bus recurrences, or `.dnam` binary framing; depend directly on [`dnam-engine`](../engine/README.md) and [`dnam-train`](../train/README.md), and keep `src/` `100%` data-driven with zero fixture or circuit-template heuristics (`.agents/rules/universal-black-box.md`).
3. **`f64` Sufficient Statistics & `uplo='U'` Packing (`ref[container.proj.stat-uplo-u]`, `ref[train.gram.accuracy-contract]`):**
   - Accumulate and factor all Gram matrices $\mathbf{G}$ and cross-correlations $\mathbf{C}$ in IEEE-754 `f64` column-major upper-triangular order (`uplo='U'`).
4. **Non-Stop 3-Phase Control-Space Audit (`ref[train.audit.phase1.clamp-stability]`):**
   - Never block unattended capture or vault editing on interactive prompts; clamp unobserved control ranges, switches, and stability bounds automatically.

---

## 3. Quality Gates & Verification Commands

Run all checks from the workspace root via `moon`:

```bash
moon check capture
moon run capture:coverage
moon run capture:qualify
moon run capture:bench
```
