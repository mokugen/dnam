//! Overdrive, Fuzz, Parallel Clean-Blend, Tri-Band Distortion, & ZOH Quantizer Chains
//! (`Chain-01..03`, `Chain-14`, `Chain-18`).

#![forbid(unsafe_code)]

pub mod clean_blend;
pub mod gated_fuzz;
pub mod symmetric_od;
pub mod triband_entropia_dist;
pub mod zoh_quantizer;

pub use clean_blend::Chain03ParallelCleanBlend;
pub use gated_fuzz::Chain02GatedAsymmetricFuzz;
pub use symmetric_od::Chain01SymmetricOd;
pub use triband_entropia_dist::Chain14TriBandCompDistortion;
pub use zoh_quantizer::Chain18ZohQuantizerBitcrusher;
