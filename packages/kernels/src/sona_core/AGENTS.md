# `dnam-kernels::sona_core` — Matryoshka `CoreQuad`, `Op-N[Pre]` & Scans Map

Submodule map for [`sona_core.rs`](../sona_core.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, `32-bit` `0-ULP` stereo & `SPIR-V` clean.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`core_quad.rs`](core_quad.rs) | `CoreQuadParamsStereo4`, `step_core_quad_stereo8` (`Selective-Duhem` + `GLU`), `couple_ms_input_stereo8` | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`op_n_pre.rs`](op_n_pre.rs) | `28 B` `OpNPreHotState`, `24 B` `OpNPreHoisted`, ZOH `Q0.32` + Fourier staircase bitcrusher | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`tpt_svf.rs`](tpt_svf.rs) | Passive `2×2` ZDF `TPT-SVF` step, prewarping & forward/adjoint scans | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`paralru.rs`](paralru.rs), [`scans.rs`](paralru/scans.rs), [`adjoint.rs`](paralru/adjoint.rs) | Contractive `1D` `ParaLRU`, zero-DC soft rectifier, exact forward tile & reverse adjoint scan | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`duhem_scans.rs`](paralru/duhem_scans.rs), [`duhem_adjoint.rs`](paralru/duhem_adjoint.rs) | Joint `(L, R)` `Selective-Duhem` stereo-linked `ParaLRU` forward & `O(log2 N)` adjoint scans | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`poly_adaa.rs`](poly_adaa.rs), [`wavefolder.rs`](poly_adaa/wavefolder.rs) | Pure-`f32` divided-difference `PolyADAA1` & `C²` wavefolder | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
