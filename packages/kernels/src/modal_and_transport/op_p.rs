//! `Op-P` three-stage input analytic basis, `HILBERT-H1-ANCHORS` rational interpolation,
//! `T1` upstream `5 Hz` stereo DC blocker & 160-bit coefficient synthesis, and 3-basis carrier
//! readout (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.tracker.t1-dc-blocker]`,
//! `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).

pub mod carrier_3basis;
pub mod dc_and_tracker;
pub mod hilbert_stereo;

pub use carrier_3basis::{
    CarrierBasisWeightsVec4, StereoAnalyticQuadrature, StereoCarrierRotorVec4,
    eval_carrier_3basis_stereo_vec4, halve_signed_q32_increments, rotate_carrier_stereo_phase_vec4,
};
pub use dc_and_tracker::{
    T1DcBlockerKernel, T1DcBlockerStereo, T1SubHarmonicKernel, tracker_crosses_branch_cut_f32,
};
#[cfg(not(target_arch = "spirv"))]
pub use dc_and_tracker::{compute_t1_dc_blocker_coeffs_bits, compute_t1_dc_blocker_coeffs_clamped};
pub use hilbert_stereo::{
    HILBERT_ALPHA_I, HILBERT_ALPHA_I_BITS, HILBERT_ALPHA_Q, HILBERT_ALPHA_Q_BITS,
    HILBERT_H1_ANCHOR_BITS, HILBERT_H1_ANCHOR_RATES_HZ, HILBERT_H1_STAGES, HilbertH1Kernel,
    HilbertH1RemapHistory, HilbertH1StereoKernel, remap_allpass_chain_states_from_outputs,
};
#[cfg(not(target_arch = "spirv"))]
pub use hilbert_stereo::{interpolate_hilbert_h1_alphas, interpolate_hilbert_h1_alphas_f64};
