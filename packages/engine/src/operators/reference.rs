//! Standalone `f64` scalar reference DSP kernels for `SONA` Operators (`Op-T`, `Op-M`, `Op-N`, Control).
//!
//! Provides canonical mathematical references for bit-accurate parity testing across all SIMD backends.

pub mod op_t_and_ctrl;
pub mod poly_and_op_m;

pub use op_t_and_ctrl::{
    catmull_rom_hermite_f64, cubic_lagrange_weights_f64, eval_chebyshev_lfo_4harm_f64,
    eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64, eval_trust_region_softclip_f64,
    givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_t_coercive_step_f64,
    op_t_tank_slice_f64,
};
pub use poly_and_op_m::{
    op_m_modal_step_f64, poly_adaa_algebraic_f64, poly_adaa_slice_f64, poly_adaa_step_f64,
};
