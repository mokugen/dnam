# `dnam-kernels::modal_and_transport` — Modal, Analytic `Op-P` & Transport `Op-T` Kernels

Submodule map for [`modal_and_transport.rs`](../modal_and_transport.rs) (parent [`AGENTS.md`](../../AGENTS.md)).

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`kautz.rs`](kautz.rs), [`rotor_and_phasor.rs`](kautz/rotor_and_phasor.rs) | Contractive unit rotor, `Q0.32` quadrature, stereo `Kautz` scan | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`op_p.rs`](op_p.rs), [`hilbert_stereo.rs`](op_p/hilbert_stereo.rs), [`anchors.rs`](op_p/hilbert_stereo/anchors.rs), [`dc_and_tracker.rs`](op_p/dc_and_tracker.rs), [`carrier_3basis.rs`](op_p/carrier_3basis.rs) | Stereo `H1` lattice, 15-rate rational anchors, `5 Hz` `T1` DC & sub-harmonic tracker, 3-basis carrier | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`op_t.rs`](op_t.rs), [`interp_and_window.rs`](op_t/interp_and_window.rs), [`quad_ortho.rs`](op_t/quad_ortho.rs) | Catmull-Rom readers, `Q0.32` window/sine/pitch-down, Schroeder diffuser, `WE` dual-latch & feedback | [`op_t_transport.md`](../../../../docs/specs/dsp/op_t_transport.md) |
