//! Normative bit-exact and finite-precision mathematical kernels (`src/math/spec_math.rs`).

pub mod float_rank;
pub mod ieee_bits;
pub mod pitch_log2;
pub mod rotor_q192;
pub mod saturation;
pub mod trust_clip;

pub use float_rank::*;
pub use ieee_bits::*;
pub use pitch_log2::*;
pub use rotor_q192::*;
pub use saturation::*;
pub use trust_clip::*;
