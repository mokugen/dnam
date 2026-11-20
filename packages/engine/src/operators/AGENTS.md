# `src/operators` — `SONA` DSP Kernel Operators (`Op-T`, `Op-M`, `Op-N`)

Implements the three fundamental real-time DSP kernel operators of `SONA` (`#[forbid(unsafe_code)]`):
1. **Op-T (Transport & Elastic Buffer):** Fractional delay, Dual-Mode slew/crossfade, binary max-tree guard, and diffusers.
2. **Op-M (Modal Bank & Kautz):** Symplectic recurrence, contractive unit rotor, `H1` Hilbert, `T1` tracker, and Kautz filter.
3. **Op-N (Nonlinear Activation & Shaping):** `dnam_kernels` `SONA` core (`PolyADAA1`, `ParaLRU`, `TPT-SVF`), `C^2` wavefolder, `Q64.32` bitcrusher, & `OpNBackend`.

## 1. Submodule Map

| Operator / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`op_t.rs`](op_t.rs) ([`delay_and_diffuser.rs`](op_t/delay_and_diffuser.rs), [`window_and_vca.rs`](op_t/delay_and_diffuser/window_and_vca.rs), [`ring_and_guard.rs`](op_t/ring_and_guard.rs)) | `dnam_kernels` Catmull-Rom delay, pending coalescing, barberpole, `O(1)` max-tree, diffusers | [`op_t_transport.md`](../../../../docs/specs/dsp/op_t_transport.md) |
| [`op_m.rs`](op_m.rs) ([`kautz.rs`](op_m/kautz.rs), [`kautz/simd.rs`](op_m/kautz/simd.rs), [`hilbert.rs`](op_m/hilbert.rs), [`anchors.rs`](op_m/hilbert/anchors.rs), [`tracker.rs`](op_m/tracker.rs), [`dc_blocker.rs`](op_m/tracker/dc_blocker.rs)) | `dnam_kernels` Kautz `StereoVec8`, contractive rotor, `H1` Hilbert & `T1` tracker | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`op_n.rs`](op_n.rs) ([`wavefolder_and_bitcrusher.rs`](op_n/wavefolder_and_bitcrusher.rs), [`bitcrusher.rs`](op_n/wavefolder_and_bitcrusher/bitcrusher.rs), [`commuted_and_ctrl.rs`](op_n/commuted_and_ctrl.rs)) | `dnam_kernels` `SONA` core, `C^2` wavefolder, `Q64.32` bitcrusher & `OpNBackend` | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`reference.rs`](reference.rs) ([`poly_and_op_m.rs`](reference/poly_and_op_m.rs), [`op_t_and_ctrl.rs`](reference/op_t_and_ctrl.rs), [`rail_crossing.rs`](reference/rail_crossing.rs)) | Standalone `f64` reference DSP & saturated `PolyADAA` rail-crossing verifier | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for SIMD vector register and zero-allocation invariants.
