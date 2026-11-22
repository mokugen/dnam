#![forbid(unsafe_code)]

//! `#![no_std]` SPIR-V uniform ABI structs and layout strides matching `sona_shaders::abi`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[repo.code.constant-eval-and-derivations]`).

use core::mem::{align_of, offset_of, size_of};

/// Number of threads per workgroup and frames per tile across all core scan kernels.
pub const WORKGROUP_SIZE: usize = 256;
pub const TILE_FRAMES: usize = WORKGROUP_SIZE;

/// Strict portable workgroup SRAM ceiling (`15,872 B = 15.5 KiB`, reserving `512 B` for driver/subgroup scratch).
pub const MAX_WORKGROUP_SRAM_BYTES: usize = 16 * 1024 - 512;

/// Number of `Vec4` vectors per `(q_global, frame)` in `core_inputs` and `d_loss_d_inputs` (`L`, `R`).
pub const CORE_INPUT_VEC4_STRIDE: usize = 2;
/// Number of `Vec4` vectors per `(stem, quad)` in `quad_params` and per `(stem, q_local, tile)` in `param_grads`.
pub const QUAD_PARAMS_VEC4_STRIDE: usize = param_slots::TOTAL_SLOTS;
/// Number of `Vec4` vectors per `(stem, q_local, tile)` checkpoint in `tile_checkpoints`.
pub const TILE_CKPT_VEC4_STRIDE: usize = 8;
/// Number of `Vec4` vectors per `(stem, q_local, frame)` in `feature_outputs` and `d_loss_d_features`.
pub const FEATURE_VEC4_STRIDE: usize = 6;
/// Number of `Vec4` summary vectors per lane channel `c in 0..4` in `tile_adj_boundaries`.
pub const TILE_ADJ_LANE_VEC4_STRIDE: usize = 5;
/// Number of `Vec4` vectors per `(stem, q_local, tile)` in `tile_adj_boundaries` (`4` lanes × `5` slots).
pub const TILE_ADJ_BOUNDARY_VEC4_STRIDE: usize = 4 * TILE_ADJ_LANE_VEC4_STRIDE;
/// Number of `f32` scalars per seed/quad entry in `seed_params`.
pub const PRESCREEN_SEED_PARAM_STRIDE: usize = 16;
/// Number of `i32` words stored per `(pair, chunk)` entry (`[d0, d1, d2, d3, d4, 0]`).
pub const OZAKI_WORDS_PER_PAIR: usize = 6;
/// Maximum GPU-resident L-BFGS history window `m`.
pub const LBFGS_MAX_HISTORY_M: usize = 8;

/// Canonical `25`-slot indices matching `sona_shaders::abi::quad_param_slots`.
pub mod param_slots {
    pub const SVF_A1: usize = 0;
    pub const SVF_A2: usize = 1;
    pub const SVF_A3: usize = 2;
    pub const SVF_K: usize = 3;
    pub const SVF_TWO_K: usize = 4;
    pub const TONE_LP: usize = 5;
    pub const TONE_BP: usize = 6;
    pub const TONE_HP: usize = 7;
    pub const TONE_AP: usize = 8;
    pub const TONE_DIR: usize = 9;
    pub const LRU_ALPHA_REL: usize = 10;
    pub const LRU_ALPHA_ATK: usize = 11;
    pub const LRU_KAPPA: usize = 12;
    pub const LRU_GAMMA_DUHEM: usize = 13;
    pub const LRU_SIGNED_MIX: usize = 14;
    pub const LRU_STEREO_LINK: usize = 15;
    pub const TWO_SVF_G: usize = 16;
    pub const RECT_SIGN: usize = 17;
    pub const SAG_DEPTH: usize = 18;
    pub const ADAA_DRIVE: usize = 19;
    pub const LRU_BIAS_DRIVE: usize = 20;
    pub const WAVEFOLDER_MIX: usize = 21;
    pub const GLU_MIX: usize = 22;
    pub const SVF_G: usize = 23;
    pub const SVF_K_NL: usize = 24;
    pub const TOTAL_SLOTS: usize = SVF_K_NL + 1;
}

/// Forward workgroup SRAM partitioning (`15,360 B <= 15,872 B`).
pub const FORWARD_SCAN_SCRATCH_FLOATS: usize = 6 * TILE_FRAMES;
pub const FORWARD_LANE_BUF_VEC4S: usize = 2 * TILE_FRAMES;
pub const FORWARD_AUX_SCRATCH_FLOATS: usize = TILE_FRAMES;

/// Adjoint workgroup SRAM partitioning (`14,848 B <= 15,872 B`).
pub const ADJOINT_S_PREV2_VEC4S: usize = TILE_FRAMES;
pub const ADJOINT_U_TONE2_VEC2S: usize = TILE_FRAMES;
pub const ADJOINT_V_TRAJ2_VEC2S: usize = TILE_FRAMES;
pub const ADJOINT_SCAN_SCRATCH_FLOATS: usize = 6 * TILE_FRAMES;
pub const ADJOINT_AUX_FLOATS: usize = TILE_FRAMES / 2;

/// Packed `32 B` uniform block for `prescreen_population`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct PrescreenUniforms {
    pub num_seeds: u32,
    pub num_frames: u32,
    pub num_quads: u32,
    pub num_stems: u32,
    pub sample_rate_hz: f32,
    pub eps_floor: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Packed `32 B` uniform block for `fused_core_scan_256`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct FusedScanUniforms {
    pub num_frames: u32,
    pub num_tiles: u32,
    pub num_quads: u32,
    pub num_stems: u32,
    pub tile_start: u32,
    pub soft_rect_eps: f32,
    pub tiny_crossing_eps: f32,
    pub block_idx: u32,
}

/// Packed `32 B` uniform block for `adjoint_scan_256`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct AdjointScanUniforms {
    pub num_frames: u32,
    pub num_tiles: u32,
    pub num_quads: u32,
    pub num_stems: u32,
    pub pass_mode: u32,
    pub soft_rect_eps: f32,
    pub tiny_crossing_eps: f32,
    pub block_idx: u32,
}

/// Packed `32 B` uniform block for `gabor_bezout_grad`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GaborBezoutUniforms {
    pub num_frames: u32,
    pub fft_size: u32,
    pub hop_size: u32,
    pub num_stems: u32,
    pub w_lf: f32,
    pub w_mf: f32,
    pub w_hf: f32,
    pub eps_noise: f32,
}

/// Packed `32 B` uniform block for `ozaki_gram_rhs_i32`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct OzakiGramUniforms {
    pub num_frames: u32,
    pub feature_dim: u32,
    pub target_channels: u32,
    pub chunk_len: u32,
    pub num_chunks: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Packed `32 B` uniform block for `inner_opt_step`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct InnerOptUniforms {
    pub num_params: u32,
    pub history_m: u32,
    pub step_idx: u32,
    pub line_search_iter: u32,
    pub step_size: f32,
    pub damping_lambda: f32,
    pub min_bound_eps: f32,
    pub _pad: u32,
}

const _: () = assert!(WORKGROUP_SIZE == 256 && TILE_FRAMES == 256);
const _: () = assert!(MAX_WORKGROUP_SRAM_BYTES == 15_872);
const _: () = assert!(CORE_INPUT_VEC4_STRIDE == 2);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == 25);
const _: () = assert!(TILE_CKPT_VEC4_STRIDE == 8);
const _: () = assert!(FEATURE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_LANE_VEC4_STRIDE == 5);
const _: () = assert!(TILE_ADJ_BOUNDARY_VEC4_STRIDE == 20);
const _: () = assert!(PRESCREEN_SEED_PARAM_STRIDE == 16);
const _: () = assert!(OZAKI_WORDS_PER_PAIR == 6);
const _: () = assert!(LBFGS_MAX_HISTORY_M == 8);

const _: () = assert!(size_of::<PrescreenUniforms>() == 32 && align_of::<PrescreenUniforms>() == 4);
const _: () = assert!(offset_of!(PrescreenUniforms, _pad1) == 28);
const _: () = assert!(size_of::<FusedScanUniforms>() == 32 && align_of::<FusedScanUniforms>() == 4);
const _: () = assert!(offset_of!(FusedScanUniforms, block_idx) == 28);
const _: () =
    assert!(size_of::<AdjointScanUniforms>() == 32 && align_of::<AdjointScanUniforms>() == 4);
const _: () = assert!(offset_of!(AdjointScanUniforms, block_idx) == 28);
const _: () =
    assert!(size_of::<GaborBezoutUniforms>() == 32 && align_of::<GaborBezoutUniforms>() == 4);
const _: () = assert!(offset_of!(GaborBezoutUniforms, eps_noise) == 28);
const _: () = assert!(size_of::<OzakiGramUniforms>() == 32 && align_of::<OzakiGramUniforms>() == 4);
const _: () = assert!(offset_of!(OzakiGramUniforms, _pad2) == 28);
const _: () = assert!(size_of::<InnerOptUniforms>() == 32 && align_of::<InnerOptUniforms>() == 4);
const _: () = assert!(offset_of!(InnerOptUniforms, _pad) == 28);
