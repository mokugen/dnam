#![forbid(unsafe_code)]

//! `#![no_std]` SPIR-V uniform ABI structs and layout strides matching `sona_shaders::abi`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[repo.code.constant-eval-and-derivations]`).

use core::mem::{align_of, offset_of, size_of};

/// Number of threads per workgroup and frames per tile across all core scan kernels.
pub const WORKGROUP_SIZE: usize = 256;
pub const TILE_FRAMES: usize = WORKGROUP_SIZE;

/// Strict portable workgroup SRAM ceiling (`15,872 B = 15.5 KiB`, reserving `512 B` for driver/subgroup scratch).
pub const MAX_WORKGROUP_SRAM_BYTES: usize = 16 * 1024 - 512;
/// W3C WebGPU baseline limit for storage buffers per compute shader stage (`8`).
pub const MAX_WEBGPU_STORAGE_BUFFERS_PER_STAGE: usize = 8;

/// Number of `Vec4` vectors per `(q_global, frame)` in `core_inputs` and `d_loss_d_inputs` (`L`, `R`).
pub const CORE_INPUT_VEC4_STRIDE: usize = 2;
/// Number of `Vec4` vectors per `(stem, quad)` in `quad_params` and per `(stem, q_local, tile)` in `param_grads`.
pub const QUAD_PARAMS_VEC4_STRIDE: usize = param_slots::TOTAL_SLOTS;
/// Number of `Vec4` vectors per `(stem, q_local, tile)` checkpoint in `tile_checkpoints`.
pub const TILE_CKPT_VEC4_STRIDE: usize = 8;
/// Number of `Vec4` vectors per `(stem, q_local, frame)` in `feature_outputs` and `d_loss_d_features`.
pub const FEATURE_VEC4_STRIDE: usize = 6;
/// Number of `Vec4` summary vectors per lane channel `c in 0..4` in `tile_adj_boundaries`
/// (`T_L`, `(g0_L, g1_L, g0_R, g1_R)`, `ParaLRU`, `ADAA seam + frame-0 adj_r_prev`, `T_R`, `k_vs`).
pub const TILE_ADJ_LANE_VEC4_STRIDE: usize = 6;
/// Number of `Vec4` vectors per `(stem, q_local, tile)` in `tile_adj_boundaries` (`4` lanes × `6` slots = `384 B`).
pub const TILE_ADJ_BOUNDARY_VEC4_STRIDE: usize = 4 * TILE_ADJ_LANE_VEC4_STRIDE;
/// Number of `Vec4` vectors per `(stem, q_local)` in `Slot 7` (`tile_adj_boundaries`) leading
/// wave-local cross-block header (`in_exit[0..8]` + `out_entry[8..16]` = `256 B`).
pub const EXIT_ADJ_BOUNDARY_VEC4_STRIDE: usize = 16;
/// Number of `f32` scalars per seed/quad entry in `seed_params`.
pub const PRESCREEN_SEED_PARAM_STRIDE: usize = 16;
/// Number of `i32` words stored per `(pair, chunk)` entry in Ozaki pass 0 (`[d0, d1, d2, d3, d4, 0]`).
pub const OZAKI_WORDS_PER_PAIR: usize = 6;
/// Number of `u32` words stored per pair in Ozaki split-64 pass 1 (`[hi64_lo, hi64_hi, lo64_lo, lo64_hi]` = `16 B`).
pub const OZAKI_SPLIT64_WORDS_PER_PAIR: usize = 4;
/// Default 2D pair tile row width for `ozaki_gram_rhs_i32`.
pub const OZAKI_PAIR_TILE_ROWS_DEFAULT: u32 = 256;
/// Optional Ozaki pass 1 `_pad` bit enabling in-place split-64 accumulation across chunk batches.
pub const OZAKI_FLAG_ACCUMULATE_SPLIT64: u32 = 1;
/// Maximum GPU-resident L-SSBroyden / L-BFGS history window `m`.
pub const LSSBROYDEN_MAX_HISTORY_M: usize = 32;
pub const LBFGS_MAX_HISTORY_M: usize = LSSBROYDEN_MAX_HISTORY_M;

/// Sentinel invalid `CoreQuad` slot inside `BLOCK_QUAD_SLOTS`.
pub const INVALID_QUAD_SLOT: usize = 16;
/// Sentinel `block_idx` selecting direct flat `quad_idx = q_local` linear non-PHS dispatches.
pub const BLOCK_IDX_DIRECT_LINEAR: u32 = 0xFFFF_FFFF;
/// Sentinel `block_wave` selecting all `0..num_quads` in `tile_mix_fwd_adj` modes 0 and 1.
pub const BLOCK_WAVE_ALL: u32 = 0xFFFF_FFFF;
/// Optional `FusedScanUniforms.block_idx` flag bit enabling damped-DEER / passive-Picard fixed-point mode.
pub const FUSED_FLAG_FIXED_POINT_SCAN: u32 = 1 << 30;

/// Canonical 4-block Matryoshka wave -> `CoreQuad` index map
/// (Block 0 `{0,4,6,8}`, Block 1 `{1,5,9}`, Block 2 `{2,7}`, Block 3 `{3}`).
pub const BLOCK_QUAD_SLOTS: [[usize; 4]; 4] = [
    [0, 4, 6, 8],
    [1, 5, 9, INVALID_QUAD_SLOT],
    [2, 7, INVALID_QUAD_SLOT, INVALID_QUAD_SLOT],
    [3, INVALID_QUAD_SLOT, INVALID_QUAD_SLOT, INVALID_QUAD_SLOT],
];

/// Returns the number of active quads belonging to `block_idx` under `num_quads`.
#[inline(always)]
#[must_use]
pub const fn block_quad_count(block_idx: usize, num_quads: usize) -> usize {
    let row = BLOCK_QUAD_SLOTS[block_idx & 3];
    let mut count = 0_usize;
    let mut k = 0_usize;
    while k < 4 {
        if row[k] < num_quads {
            count += 1;
        }
        k += 1;
    }
    count
}

/// Resolves wave-local `q_local` into canonical tier-global `quad_idx` (`< num_quads` or `INVALID_QUAD_SLOT`).
#[inline(always)]
#[must_use]
pub const fn resolve_quad_idx(block_idx: u32, q_local: usize, num_quads: usize) -> usize {
    if block_idx == BLOCK_IDX_DIRECT_LINEAR {
        if q_local < num_quads {
            q_local
        } else {
            INVALID_QUAD_SLOT
        }
    } else if q_local < 4 {
        let slot = BLOCK_QUAD_SLOTS[(block_idx as usize) & 3][q_local];
        if slot < num_quads {
            slot
        } else {
            INVALID_QUAD_SLOT
        }
    } else {
        INVALID_QUAD_SLOT
    }
}

/// Inter-block tile descriptor stride (`13 × f32[4] = 208 B` per active tile in `tile_mix_fwd_adj` `Slot 2`).
pub const TILE_DESC_VEC4_STRIDE: usize = 13;
/// Inter-block tile weight gradient scalar stride (`48` floats = `3 × 4×4` matrices per active tile).
pub const TILE_WEIGHT_GRAD_FLOATS: usize = 48;
/// `tile_mix_fwd_adj` execution mode selectors.
pub const TILE_MIX_DIR_FORWARD: u32 = 0;
pub const TILE_MIX_DIR_PULLBACK: u32 = 1;
pub const TILE_MIX_DIR_PARTIAL_WEIGHTS: u32 = 2;
pub const TILE_MIX_DIR_REDUCE_WEIGHTS: u32 = 3;

/// `inner_opt_step` mode bitflags (`InnerOptUniforms.mode_flags`).
pub const INNER_OPT_MODE_LSSBROYDEN: u32 = 1 << 0;
pub const INNER_OPT_MODE_STE_FP16: u32 = 1 << 1;
pub const INNER_OPT_MODE_LINE_SEARCH_TRIAL: u32 = 1 << 2;
pub const INNER_OPT_MODE_RESET_STATE: u32 = 1 << 3;
pub const INNER_OPT_MODE_DIAG_PRECOND: u32 = 1 << 4;

/// `inner_opt_step` `Slot 3` `packed_meta` coordinate transform IDs (bits `0..4`) and `ste_fp16` bit `8`.
pub const COORD_TRANSFORM_FREE: u32 = 0;
pub const COORD_TRANSFORM_BOX: u32 = 1;
pub const COORD_TRANSFORM_LOG_BOX: u32 = 2;
pub const COORD_TRANSFORM_COMPLEMENT_LOG: u32 = 3;
pub const COORD_TRANSFORM_ASINH_BOX_SYM: u32 = 4;
pub const COORD_TRANSFORM_ASINH_BOX_NONNEG: u32 = 5;
pub const STE_FP16_META_BIT: u32 = 1 << 8;

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
pub const FORWARD_LANE_BUF_FLOATS: usize = 8 * TILE_FRAMES;
pub const FORWARD_AUX_SCRATCH_FLOATS: usize = TILE_FRAMES;

/// `tile_mix_fwd_adj` workgroup SRAM partitioning (`2,048 B <= 15,872 B`).
pub const TILE_MIX_REDUCE_A_FLOATS: usize = TILE_FRAMES;
pub const TILE_MIX_REDUCE_B_FLOATS: usize = TILE_FRAMES;

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

/// Packed `32 B` uniform block for `tile_mix_fwd_adj`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TileMixUniforms {
    pub num_frames: u32,
    pub num_tiles: u32,
    pub num_active_tiles: u32,
    pub dir_mode: u32,
    pub num_stems: u32,
    pub block_wave: u32,
    pub num_quads: u32,
    pub _pad1: u32,
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
    pub pair_tile_rows: u32,
    pub reduce_pass: u32,
    pub _pad: u32,
}

/// Packed `32 B` uniform block for `inner_opt_step`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct InnerOptUniforms {
    pub num_params: u32,
    pub history_m: u32,
    pub step_idx: u32,
    pub mode_flags: u32,
    pub step_size: f32,
    pub damping_lambda: f32,
    pub min_bound_eps: f32,
    pub grad_clip_norm: f32,
}

const _: () = assert!(WORKGROUP_SIZE == 256 && TILE_FRAMES == 256);
const _: () = assert!(MAX_WORKGROUP_SRAM_BYTES == 15_872);
const _: () = assert!(MAX_WEBGPU_STORAGE_BUFFERS_PER_STAGE == 8);
const _: () = assert!(FORWARD_LANE_BUF_FLOATS == 2_048);
const _: () = assert!(TILE_MIX_REDUCE_A_FLOATS + TILE_MIX_REDUCE_B_FLOATS == 512);
const _: () = assert!(CORE_INPUT_VEC4_STRIDE == 2);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == 25);
const _: () = assert!(TILE_CKPT_VEC4_STRIDE == 8);
const _: () = assert!(FEATURE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_LANE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_BOUNDARY_VEC4_STRIDE == 24);
const _: () = assert!(EXIT_ADJ_BOUNDARY_VEC4_STRIDE == 16);
const _: () = assert!(TILE_DESC_VEC4_STRIDE == 13);
const _: () = assert!(TILE_WEIGHT_GRAD_FLOATS == 48);
const _: () = assert!(PRESCREEN_SEED_PARAM_STRIDE == 16);
const _: () = assert!(OZAKI_WORDS_PER_PAIR == 6 && OZAKI_SPLIT64_WORDS_PER_PAIR == 4);
const _: () = assert!(LSSBROYDEN_MAX_HISTORY_M == 32 && LBFGS_MAX_HISTORY_M == 32);
const _: () = assert!(block_quad_count(0, 4) == 1 && block_quad_count(0, 10) == 4);
const _: () = assert!(block_quad_count(1, 10) == 3 && block_quad_count(2, 10) == 2);
const _: () = assert!(block_quad_count(3, 10) == 1);

const _: () = assert!(size_of::<PrescreenUniforms>() == 32 && align_of::<PrescreenUniforms>() == 4);
const _: () = assert!(offset_of!(PrescreenUniforms, _pad1) == 28);
const _: () = assert!(size_of::<FusedScanUniforms>() == 32 && align_of::<FusedScanUniforms>() == 4);
const _: () = assert!(offset_of!(FusedScanUniforms, block_idx) == 28);
const _: () =
    assert!(size_of::<AdjointScanUniforms>() == 32 && align_of::<AdjointScanUniforms>() == 4);
const _: () = assert!(offset_of!(AdjointScanUniforms, block_idx) == 28);
const _: () = assert!(size_of::<TileMixUniforms>() == 32 && align_of::<TileMixUniforms>() == 4);
const _: () = assert!(offset_of!(TileMixUniforms, _pad1) == 28);
const _: () =
    assert!(size_of::<GaborBezoutUniforms>() == 32 && align_of::<GaborBezoutUniforms>() == 4);
const _: () = assert!(offset_of!(GaborBezoutUniforms, eps_noise) == 28);
const _: () = assert!(size_of::<OzakiGramUniforms>() == 32 && align_of::<OzakiGramUniforms>() == 4);
const _: () = assert!(offset_of!(OzakiGramUniforms, _pad) == 28);
const _: () = assert!(size_of::<InnerOptUniforms>() == 32 && align_of::<InnerOptUniforms>() == 4);
const _: () = assert!(offset_of!(InnerOptUniforms, grad_clip_norm) == 28);
