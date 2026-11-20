//! `SONA` Zero-FDIV SIMD Operator Kernels (`Op-T`, `Op-M`, `Op-N`, and `reference`).
//!
//! Subsystems:
//! - [`op_t`]: Universal Transport, Granular Windowing & All-Pass Diffusion Operator.
//! - [`op_m`]: Universal Symplectic Modal, Hilbert, Sub-Harmonic & Kautz Operator.
//! - [`op_n`]: Universal Nonlinear, Hysteretic & Commuted Cascade Operator.
//! - [`reference`]: Bit-accurate scalar `f64` reference implementations.
//!
//! Direct per-sample calls to real-time kernels require a caller-held
//! [`crate::FtzDazGuard`] for the enclosing batch.

pub mod op_m;
pub mod op_n;
pub mod op_t;
pub mod reference;

pub use op_m::{
    HILBERT_H1_ANCHORS_15, HilbertH1Kernel, HilbertH1Lattice6Sec, HilbertSsbLattice10th,
    KautzModalBank, KautzQuadState, OpMBackend, PhasorQ32Bank4, SubHarmonicTracker,
    SymplecticModal4Lane, T1DcBlocker, T1SubHarmonicKernel, T1SubHarmonicTracker,
    algebraic_unit_rotor_op_m, compute_t1_dc_blocker_coeffs_bits, convert_kautz_pole_for_rate,
    encode_hilbert_h1_anchor_table_780b, eval_kautz_sweep_contraction,
    eval_tracker_half_angle_pair, fast_rsqrt_nr2, init_log_octave_poles_h2,
    interpolate_hilbert_h1_between_anchors_bits, interpolate_hilbert_h1_profile_bits,
    normalize_tracker_phasor_exponent_split, step_kautz_quad_stereo8, step_kautz_quad_vec4,
    tracker_crosses_branch_cut, tracker_exp_scales_scalar_f32, validate_and_eval_kautz_radial_gate,
};
pub use op_n::{
    FixedHzBitcrusher, OpNBackend, ParaLruParams, QuadratureHysteresis2D, SOFT_RECT_EPS,
    SonaBlockOutputStereo8, SonaBlockParamsStereo4, SonaBlockStateStereo8, SonaGroupParamsStereo4,
    SonaParaLruCoeffs4, SonaToneWeights4, TptSvfCoeffs4, TptSvfOutputs, ZohBitcrusher,
    c2_wavefolder, eval_adaa_cubic_c2_wavefolder_odd_f32, eval_c2_wavefolder_adaa_stereo8,
    eval_c2_wavefolder_adaa_vec4, eval_chebyshev_4harm,
    eval_commuted_vs_uncommuted_lti_cascade_f64, eval_core_stereo_coupling_16d,
    eval_lobatto_basis_3node, eval_poly_adaa1_odd_even_vec4, eval_poly_adaa1_quad_batched,
    eval_poly_adaa1_stereo8, fold_core_latent_weights, hertz_phase_parameters, hertz_phase_step,
    prewarp_tan_pi_f_over_fs_f32, quantize_bitcrusher_nearest_ties_pos_inf,
    rescale_bitcrusher_phase_u128, soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8,
    soft_rect_zero_dc_vec4, step_paralru_stereo8, step_paralru_vec4, step_sona_block_stereo8,
    step_sona_group_stereo8, step_tpt_svf_stereo8, step_tpt_svf_vec4, tanh_pade5_4ch_vec4,
    trust_region_softclip_4d,
};
pub use op_t::{
    BallisticDuckingVca, FP16_DECAY_GAIN_BITS, OP_T_FULL_RING_CAPACITY, OP_T_LITE_RING_CAPACITY,
    OP_T_ULTRA_RING_CAPACITY, OpTBackend, OpTBlockMaxTreeGuard, OpTDelayLine,
    PartitionedSchroederDiffuserBank, SCHROEDER_COPRIME_DELAYS, admit_op_t_memory_budget,
    bounded_q32_reader_position, bspline_barberpole_3phase, catmull_rom_hermite_cubic,
    coercive_loop_limit_vec4, convert_schroeder_delay_for_rate, cubic_lagrange_weights_4ch_vec4,
    directed_fp16_decay_write_bits, eval_m2_pitch_grain_delays, hermite_hann_partition_of_unity,
    op_t_branchless_next_write, op_t_branchless_wrap_index, op_t_fractional_history_capacity,
    q32_decode_reader_position, raised_cosine_barberpole_3phase, read_catmull_rom_f32,
    read_catmull_rom_fp16_u16, step_schroeder_allpass, step_schroeder_quad_vec4,
    write_ring_with_freeze_f32,
};
pub use reference::{
    catmull_rom_hermite_f64, cubic_lagrange_weights_f64, eval_chebyshev_lfo_4harm_f64,
    eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64, eval_trust_region_softclip_f64,
    givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_m_modal_step_f64,
    op_t_coercive_step_f64, op_t_tank_slice_f64, poly_adaa_algebraic_f64, poly_adaa_slice_f64,
    poly_adaa_step_f64,
};
