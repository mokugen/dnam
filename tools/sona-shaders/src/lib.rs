#![forbid(unsafe_code)]

//! Offline AOT `cargo-gpu` (`spirv-unknown-naga-wgsl`) compiler, Naga validator, manifest
//! registry, and `zerocopy` host/GPU ABI definitions for `SONA` v3.0 WebGPU compute kernels
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod abi;
pub mod compiler;
pub mod manifest;
pub mod validate;
