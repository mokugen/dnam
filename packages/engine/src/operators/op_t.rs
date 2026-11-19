//! Op-T: Universal Transport, Granular Windowing & All-Pass Diffusion Operator.
//!
//! Canonical Reference: `docs/specs/dsp/op_t_transport.md` (`ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.op-t.transition.dual-mode]`, `ref[dsp.op-t.diffuser.recurrence]`).

pub mod delay_and_diffuser;
pub mod ring_and_guard;

pub use delay_and_diffuser::{
    BallisticDuckingVca, OP_T_FULL_RING_CAPACITY, OP_T_LITE_RING_CAPACITY,
    OP_T_ULTRA_RING_CAPACITY, OpTBackend, OpTDelayLine, SCHROEDER_COPRIME_DELAYS,
    catmull_rom_hermite_cubic, cubic_lagrange_weights_4ch_vec4, hermite_hann_partition_of_unity,
    raised_cosine_barberpole_3phase,
};
pub use dnam_kernels::{
    coercive_loop_limit_vec4, read_catmull_rom_f32, read_catmull_rom_fp16_u16,
    step_schroeder_allpass, step_schroeder_quad_vec4, write_ring_with_freeze_f32,
};
pub use ring_and_guard::{
    FP16_DECAY_GAIN_BITS, OpTBlockMaxTreeGuard, PartitionedSchroederDiffuserBank,
    admit_op_t_memory_budget, bounded_q32_reader_position, bspline_barberpole_3phase,
    convert_schroeder_delay_for_rate, directed_fp16_decay_write_bits, eval_m2_pitch_grain_delays,
    op_t_branchless_next_write, op_t_branchless_wrap_index, op_t_fractional_history_capacity,
    q32_decode_reader_position,
};
