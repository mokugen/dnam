//! Optical Compressor (`Chain-09`), Octave Divider & Ring Mod (`Chain-10`), and Tri-Band VCA Compressor (`Chain-13`).

#![forbid(unsafe_code)]

pub mod octave_ringmod;
pub mod optical_comp;
pub mod triband_vca_comp;

pub use octave_ringmod::Chain10Oc2OctaveDividerAndRingMod;
pub use optical_comp::Chain09La2aOpticalCompressor;
pub use triband_vca_comp::Chain13TriBandVcaCompressor;
