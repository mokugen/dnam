//! Causal top-of-frame fractional delay transport (`Op-T`), orthogonal `WE` / Hermite-Hann /
//! pitch-down windowing, partitioned Schroeder all-pass diffusion, feedback conditioning, and
//! Family 4 ducking VCA (`ref[dsp.op-t.reader.catmull-rom]`,
//! `ref[dsp.op-t.window.partition-of-unity]`, `ref[dsp.op-t.diffuser.recurrence]`,
//! `ref[dsp.tank.loop-saturator]`, `ref[dsp.op-t.ducking.vca]`).

pub mod interp_and_window;
pub mod quad_ortho;

pub use interp_and_window::{
    blend_catmull_rom_modulated_f32, bspline_barberpole_3phase_f32, catmull_rom_hermite_f32,
    clamp_transport_delay_f32, compute_half_alpha_cubic_f32, compute_op_t_mu_win_f32,
    compute_op_t_pitch_down_g_loop_eff, cubic_lagrange_weights_f32, eval_op_t_diagonal_window_f32,
    eval_op_t_diagonal_window_vec4, hermite_hann_partition_of_unity_f32, q32_smooth_sin_f32,
    read_barberpole_m3_f32, read_catmull_rom_f32, read_catmull_rom_fp16_u16,
    read_pitch_grain_m2_f32, wgsl_triangle_fold_q32,
};
pub use quad_ortho::{
    FP16_FINITE_MAX_F32, OpTFeedbackParams, OpTFeedbackState, QUAD_FLAG_HAS_EXT,
    QUAD_FLAG_HOST_WRITE_LOCK, QUAD_FLAG_SCHROEDER_ENABLE, QUAD_FLAG_WE_HOLD_ACTIVE,
    QUAD_WE_LOCK_MASK, WE_SCHMITT_HOLD_LOW_F32, WE_SCHMITT_RELEASE_HIGH_F32,
    coercive_loop_limit_vec4, step_op_t_ducking_vca_f32, step_op_t_feedback_vec4,
    step_op_t_wet_dc_stereo_f32, step_schroeder_allpass, step_schroeder_quad_vec4,
    step_we_hold_lag, step_we_hold_lag_u16, update_we_schmitt_flag, write_ring_with_freeze_f32,
    write_ring_with_we_f32,
};
