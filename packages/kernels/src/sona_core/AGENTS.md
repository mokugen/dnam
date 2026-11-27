# `dnam-kernels::sona_core` — Matryoshka `CoreQuad`, `Op-N[Pre]` & Scans Map

Submodule map for [`sona_core.rs`](../sona_core.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, `32-bit` `0-ULP` stereo & `SPIR-V` clean.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`core_quad.rs`](core_quad.rs), [`v4.rs`](core_quad/v4.rs), [`dynamics.rs`](core_quad/dynamics.rs) | `CoreQuad` v3/v4 (`g_pre`, `SE-Bus`, `GLU`, folded `TPT-SVF` & Pred-Corr `PHS`) | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`op_n_pre.rs`](op_n_pre.rs), [`v4.rs`](op_n_pre/v4.rs) | `Op-N[Pre]` v3 (`28 B`) & v4 (`36 B` Hermite `Q` + hard-capable `Q0.32` ZOH) | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`os2x_hbfir.rs`](os2x_hbfir.rs) | All-tier `2x` `15`-tap symmetric linear-phase `HB-FIR` up/downsampler (`4` FMAs) | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`tpt_svf.rs`](tpt_svf.rs), [`tone_folded.rs`](tpt_svf/tone_folded.rs), [`scans.rs`](tpt_svf/scans.rs), [`phs.rs`](tpt_svf/phs.rs), [`phs/scans.rs`](tpt_svf/phs/scans.rs) | `2×2` `ZDF` `TPT-SVF`, `3`-tap folded tone, `O(log2 N)` scans & Pred-Corr `PHS` | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`paralru.rs`](paralru.rs), [`scans.rs`](paralru/scans.rs), [`adjoint.rs`](paralru/adjoint.rs) | Contractive `1D` `ParaLRU`, zero-DC soft rectifier, exact forward tile & reverse adjoint scan | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`duhem_scans.rs`](paralru/duhem_scans.rs), [`duhem_adjoint.rs`](paralru/duhem_adjoint.rs) | Joint `(L, R)` `Selective-Duhem` stereo-linked `ParaLRU` forward & `O(log2 N)` adjoint scans | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`poly_adaa.rs`](poly_adaa.rs), [`wavefolder.rs`](poly_adaa/wavefolder.rs) | Pure-`f32` divided-difference `PolyADAA1` & `C²` wavefolder | [`op_n_paralru_and_polyadaa.md`](../../../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
