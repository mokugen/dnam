//! Overdrive, Fuzz, Parallel Clean-Blend & Tri-Band Distortion Chains (`Chain-01..03`, `Chain-14`).

#![forbid(unsafe_code)]

pub mod clean_blend;
pub mod gated_fuzz;
pub mod symmetric_od;
pub mod triband_entropia_dist;

pub use clean_blend::Chain03ParallelCleanBlend;
pub use gated_fuzz::Chain02GatedAsymmetricFuzz;
pub use symmetric_od::Chain01SymmetricOd;
pub use triband_entropia_dist::Chain14TriBandCompDistortion;
