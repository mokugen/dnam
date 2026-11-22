//! Optical Compressor (`Chain-09`), Octave Divider & Ring Mod (`Chain-10`), Tri-Band VCA Compressor (`Chain-13`),
//! and Granular Pitch Shifter & Shimmer Tank (`Chain-16`).

#![forbid(unsafe_code)]

pub mod octave_ringmod;
pub mod optical_comp;
pub mod pitch_shimmer;
pub mod triband_vca_comp;

pub use octave_ringmod::Chain10Oc2OctaveDividerAndRingMod;
pub use optical_comp::Chain09La2aOpticalCompressor;
pub use pitch_shimmer::Chain16GranularPitchShimmer;
pub use triband_vca_comp::Chain13TriBandVcaCompressor;
