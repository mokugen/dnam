#![forbid(unsafe_code)]

//! Shared `32 B` `zerocopy` uniform structs, `15,872 B` workgroup SRAM ceilings, and
//! `CoreQuadV4ParamsStereo4` `28 × Vec4` buffer layout strides (`ref[train.pipeline.webgpu-scans-and-ozaki]`,
//! `ref[sdk.rust.zerocopy-wire-structs]`, `ref[repo.code.constant-eval-and-derivations]`).

use core::mem::{align_of, offset_of, size_of};
use dnam_kernels::{
    CoreQuadV4ParamsStereo4, GpuVec4, MAX_OUTGOING_PER_QUAD_V4, MAX_SONA_TILES_V4,
    OZAKI_MAX_CHUNK_LEN, SOFT_RECT_EPS, SONA_V4_BLOCK_QUAD_SLOTS, SONA_V4_INTERBLOCK_TILES,
    SONA_V4_TILE_COUNTS,
};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Maximum outgoing inter-block tiles per `CoreQuad` across `SONA v4.0` (`7`).
pub const MAX_OUTGOING_PER_QUAD: usize = MAX_OUTGOING_PER_QUAD_V4;

/// Maximum lower-triangular `4x4` inter-block tiles across `SONA v4.0` (`37`).
pub const MAX_SONA_TILES: usize = MAX_SONA_TILES_V4;

/// Cumulative `SONA v4.0` inter-block tile counts per tier `[Lite, Full, Pro, Ultra]`.
pub const SONA_TILE_COUNTS: [usize; 4] = SONA_V4_TILE_COUNTS;

/// Canonical `37` `SONA v4.0` lower-triangular inter-block edges `(src_quad, dst_quad)`.
pub const SONA_INTERBLOCK_TILES: [(usize, usize); MAX_SONA_TILES_V4] = SONA_V4_INTERBLOCK_TILES;

/// Maximum Ozaki chunk length imported directly from `dnam-kernels` (`1024`).
pub const GPU_OZAKI_MAX_CHUNK_LEN: usize = OZAKI_MAX_CHUNK_LEN;

/// Canonical smooth soft-rectifier smoothing radius imported from `dnam-kernels`.
pub const GPU_SOFT_RECT_EPS: f32 = SOFT_RECT_EPS;

/// Size in bytes of every uniform buffer block at `@group(0) @binding(0)` (`8 × 32-bit` words).
pub const UNIFORM_STRUCT_BYTES: usize = size_of::<[u32; 8]>();

/// Number of compute invocations per workgroup (`(256, 1, 1)`).
pub const WORKGROUP_SIZE: u32 = 256;

/// Number of audio frames processed per persistent workgroup tile (`256`).
pub const TILE_FRAMES: usize = WORKGROUP_SIZE as usize;

/// Baseline hardware WebGPU `maxComputeWorkgroupStorageSize` (`16 KiB`).
pub const MAX_HARDWARE_WORKGROUP_SRAM_BYTES: usize = 16 * 1024;

/// Headroom reserved for GPU driver and optional hardware subgroup scratch (`512 B`).
pub const RESERVED_DRIVER_SCRATCH_BYTES: usize = 512;

/// Strict portable workgroup SRAM ceiling (`15,872 B = 15.5 KiB`).
pub const MAX_WORKGROUP_SRAM_BYTES: usize =
    MAX_HARDWARE_WORKGROUP_SRAM_BYTES - RESERVED_DRIVER_SCRATCH_BYTES;

/// Forward `fused_core_scan_256` `scan_scratch` capacity (`1,536 × f32 = 6,144 B`).
pub const FORWARD_SCAN_SCRATCH_FLOATS: usize = 6 * TILE_FRAMES;
/// Forward `fused_core_scan_256` lane-major scalar `lane_buf` capacity (`2,048 × f32 = 8,192 B`).
pub const FORWARD_LANE_BUF_FLOATS: usize = 8 * TILE_FRAMES;
/// Forward `fused_core_scan_256` `aux_scratch` capacity (`256 × f32 = 1,024 B`).
pub const FORWARD_AUX_SCRATCH_FLOATS: usize = TILE_FRAMES;

/// Total forward kernel workgroup SRAM footprint (`15,360 B = 15.0 KiB <= 15,872 B`).
pub const FUSED_CORE_SRAM_BYTES: usize = FORWARD_SCAN_SCRATCH_FLOATS * size_of::<f32>()
    + FORWARD_LANE_BUF_FLOATS * size_of::<f32>()
    + FORWARD_AUX_SCRATCH_FLOATS * size_of::<f32>();

/// `tile_mix_fwd_adj` workgroup SRAM reduction capacity (`2,048 B <= 15,872 B`).
pub const TILE_MIX_REDUCE_A_FLOATS: usize = TILE_FRAMES;
pub const TILE_MIX_REDUCE_B_FLOATS: usize = TILE_FRAMES;
pub const TILE_MIX_SRAM_BYTES: u32 =
    ((TILE_MIX_REDUCE_A_FLOATS + TILE_MIX_REDUCE_B_FLOATS) * size_of::<f32>()) as u32;

/// Adjoint `adjoint_scan_256` `s_prev2` capacity (`256 × Vec4 = 4,096 B`).
pub const ADJOINT_S_PREV2_VEC4S: usize = TILE_FRAMES;
/// Adjoint `adjoint_scan_256` `u_tone2` capacity (`256 × Vec2 = 2,048 B`).
pub const ADJOINT_U_TONE2_VEC2S: usize = TILE_FRAMES;
/// Adjoint `adjoint_scan_256` `v_traj2` capacity (`256 × Vec2 = 2,048 B`).
pub const ADJOINT_V_TRAJ2_VEC2S: usize = TILE_FRAMES;
/// Adjoint `adjoint_scan_256` `scan_scratch` capacity (`1,536 × f32 = 6,144 B`).
pub const ADJOINT_SCAN_SCRATCH_FLOATS: usize = 6 * TILE_FRAMES;
/// Adjoint `adjoint_scan_256` `aux` capacity (`128 × f32 = 512 B`).
pub const ADJOINT_AUX_FLOATS: usize = TILE_FRAMES / 2;

/// Total adjoint kernel workgroup SRAM footprint (`14,848 B = 14.5 KiB <= 15,872 B`).
pub const ADJOINT_CORE_SRAM_BYTES: usize = ADJOINT_S_PREV2_VEC4S * size_of::<[f32; 4]>()
    + ADJOINT_U_TONE2_VEC2S * size_of::<[f32; 2]>()
    + ADJOINT_V_TRAJ2_VEC2S * size_of::<[f32; 2]>()
    + ADJOINT_SCAN_SCRATCH_FLOATS * size_of::<f32>()
    + ADJOINT_AUX_FLOATS * size_of::<f32>();

/// Slot indices (`0..28`) for the canonical `28 × [f32; 4]` SONA v4.0 packing of
/// [`dnam_kernels::CoreQuadV4ParamsStereo4`] per `(stem, quad)` in `quad_params` & `param_grads`
/// (`two_k = k + k` and `two_svf_g = svf_g + svf_g` are derived in `QuadLaneParams`):
/// - `0..=3`: `SVF_A1..A3, SVF_K` (`4 × Vec4`)
/// - `4..=7`: `TONE_C_{LP, BP, KBP, U}` (`4 × Vec4`, folded 3-tap + Block 0 PHS `c_kbp`)
/// - `8..=13`: `LRU_{ALPHA_REL, ALPHA_ATK, KAPPA, GAMMA_DUHEM, SIGNED_MIX, STEREO_LINK}` (`6 × Vec4`)
/// - `14..=15`: `SVF_G, SVF_K_NL` (`2 × Vec4`)
/// - `16..=19`: `RECT_SIGN, G0, G_INF, D_PRE` (`4 × Vec4`)
/// - `20..=21`: `SE_W_SQ, SE_D_BUS` (`2 × Vec4`)
/// - `22..=27`: `ADAA_DRIVE, LRU_BIAS_DRIVE, LIN_MIX, WAVEFOLDER_MIX, GLU_MIX, OUT_GAIN` (`6 × Vec4`)
pub mod quad_param_slots {
    pub const SVF_A1: usize = 0;
    pub const SVF_A2: usize = 1;
    pub const SVF_A3: usize = 2;
    pub const SVF_K: usize = 3;
    pub const TONE_C_LP: usize = 4;
    pub const TONE_C_BP: usize = 5;
    pub const TONE_C_KBP: usize = 6;
    pub const TONE_C_U: usize = 7;
    pub const LRU_ALPHA_REL: usize = 8;
    pub const LRU_ALPHA_ATK: usize = 9;
    pub const LRU_KAPPA: usize = 10;
    pub const LRU_GAMMA_DUHEM: usize = 11;
    pub const LRU_SIGNED_MIX: usize = 12;
    pub const LRU_STEREO_LINK: usize = 13;
    pub const SVF_G: usize = 14;
    pub const SVF_K_NL: usize = 15;
    pub const RECT_SIGN: usize = 16;
    pub const G0: usize = 17;
    pub const G_INF: usize = 18;
    pub const D_PRE: usize = 19;
    pub const SE_W_SQ: usize = 20;
    pub const SE_D_BUS: usize = 21;
    pub const ADAA_DRIVE: usize = 22;
    pub const LRU_BIAS_DRIVE: usize = 23;
    pub const LIN_MIX: usize = 24;
    pub const WAVEFOLDER_MIX: usize = 25;
    pub const GLU_MIX: usize = 26;
    pub const OUT_GAIN: usize = 27;
    pub const TOTAL_SLOTS: usize = OUT_GAIN + 1;
}

/// W3C WebGPU baseline limit for storage buffers per compute shader stage (`8`).
pub const MAX_WEBGPU_STORAGE_BUFFERS_PER_STAGE: usize = 8;

/// Number of `Vec4` (`f32[4]`) vectors per `(stem, quad, frame)` in `core_inputs` and `d_loss_d_inputs` (`L`, `R`).
pub const CORE_INPUT_VEC4_STRIDE: usize = 2;

/// Number of `Vec4` (`f32[4]`) vectors per `(stem, quad)` in `quad_params` and `param_grads`.
pub const QUAD_PARAMS_VEC4_STRIDE: usize = quad_param_slots::TOTAL_SLOTS;

/// Number of `Vec4` vectors per `(stem, q_local, tile)` checkpoint:
/// `(s1_L, s1_R, s2_L, s2_R, v_L, v_R, adaa_prev_L, adaa_prev_R, s1_lin_L, s1_lin_R, s2_lin_L, s2_lin_R)`.
pub const TILE_CKPT_VEC4_STRIDE: usize = 12;

/// Number of `Vec4` vectors per `(stem, q_local, frame)` emitted into `feature_outputs`
/// `(v_eff_L, v_eff_R, o_tilde_L, o_tilde_R, e_tilde_L, e_tilde_R)`.
pub const FEATURE_VEC4_STRIDE: usize = 6;

/// Number of `Vec4` summary vectors per lane channel `c in 0..4` in `tile_adj_boundaries`
/// (`T_b_L / s_b_out`, `g_b`, `ParaLRU`, `ADAA seam`, `T_b_R`, `k_vs_b`, `T_lin / s_lin_out`,
/// `g_lin_0`, `g_lin_k`, `g_lin_b0`, `g_lin_b1`).
pub const TILE_ADJ_LANE_VEC4_STRIDE: usize = 11;

/// Number of `Vec4` vectors per `(stem, q_local, tile)` in `tile_adj_boundaries`
/// (`4` lane channels × `11` summary `Vec4`s = `44` `Vec4`s = `704 B`).
pub const TILE_ADJ_BOUNDARY_VEC4_STRIDE: usize = 4 * TILE_ADJ_LANE_VEC4_STRIDE;

/// Number of `Vec4` vectors per `(stem, q_local)` in `Slot 7` (`tile_adj_boundaries`) leading
/// wave-local cross-block header (`in_exit[0..12]` + `out_entry[12..24]` = `384 B`).
pub const EXIT_ADJ_BOUNDARY_VEC4_STRIDE: usize = 24;

/// Number of `f32` scalars per seed in `prescreen_population`'s `seed_params`.
pub const PRESCREEN_SEED_PARAM_STRIDE: usize = 16;

/// Number of `i32` words per `(pair, chunk)` in Ozaki pass 0 (`[d0, d1, d2, d3, d4, pad]`).
pub const OZAKI_WORDS_PER_PAIR: usize = 6;

/// Number of `u32` words per pair in Ozaki split-64 pass 1 (`[hi64_lo, hi64_hi, lo64_lo, lo64_hi]`).
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

const fn option_slot_or_invalid(slot: Option<usize>) -> usize {
    match slot {
        Some(idx) => idx,
        None => INVALID_QUAD_SLOT,
    }
}

/// Canonical `SONA v4.0` 4-block Matryoshka `[3,3,2,2]` wave -> `CoreQuad` index map
/// (Block 0 `{0,4,8}`, Block 1 `{1,5,9}`, Block 2 `{2,6}`, Block 3 `{3,7}`).
pub const BLOCK_QUAD_SLOTS: [[usize; 4]; 4] = [
    [
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[0][0]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[0][1]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[0][2]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[0][3]),
    ],
    [
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[1][0]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[1][1]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[1][2]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[1][3]),
    ],
    [
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[2][0]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[2][1]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[2][2]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[2][3]),
    ],
    [
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[3][0]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[3][1]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[3][2]),
        option_slot_or_invalid(SONA_V4_BLOCK_QUAD_SLOTS[3][3]),
    ],
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

/// Prescreen workgroup SRAM footprint (`6*256 + 256 + 256` floats = `8,192 B <= 15,872 B`).
pub const PRESCREEN_SRAM_BYTES: u32 =
    ((6 * TILE_FRAMES + TILE_FRAMES + TILE_FRAMES) * size_of::<f32>()) as u32;

/// Gabor/Bézout workgroup SRAM footprint (`256*Vec4 + 256*Vec4 + 256*Vec2 + 256*f32` = `11,264 B`).
pub const GABOR_BEZOUT_SRAM_BYTES: u32 = (TILE_FRAMES * size_of::<[f32; 4]>()
    + TILE_FRAMES * size_of::<[f32; 4]>()
    + TILE_FRAMES * size_of::<[f32; 2]>()
    + TILE_FRAMES * size_of::<f32>()) as u32;

/// Ozaki Gram/RHS workgroup SRAM footprint (`5 * 256` `i32`s = `5,120 B`).
pub const OZAKI_GRAM_SRAM_BYTES: u32 = (5 * TILE_FRAMES * size_of::<i32>()) as u32;

/// Inner optimizer workgroup SRAM footprint (`256 + 256 + 128` floats = `2,560 B`).
pub const INNER_OPT_SRAM_BYTES: u32 = ((TILE_FRAMES + TILE_FRAMES + 128) * size_of::<f32>()) as u32;

/// Packed `32 B` uniform block for `prescreen_population`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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
///
/// Mode B (`fft_size > 0`) dispatches disjoint windows per stem (`hop_size >= fft_size`) so
/// `d_loss_d_pred` writebacks stay race-free across workgroups.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
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

const _: () = assert!(UNIFORM_STRUCT_BYTES == 32);
const _: () = assert!(WORKGROUP_SIZE == 256);
const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(MAX_WORKGROUP_SRAM_BYTES == 15_872);
const _: () = assert!(PRESCREEN_SRAM_BYTES == 8_192);
const _: () = assert!(FUSED_CORE_SRAM_BYTES == 15_360);
const _: () = assert!(TILE_MIX_SRAM_BYTES == 2_048);
const _: () = assert!(ADJOINT_CORE_SRAM_BYTES == 14_848);
const _: () = assert!(GABOR_BEZOUT_SRAM_BYTES == 11_264);
const _: () = assert!(OZAKI_GRAM_SRAM_BYTES == 5_120);
const _: () = assert!(INNER_OPT_SRAM_BYTES == 2_560);
const _: () = assert!(PRESCREEN_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(FUSED_CORE_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);
const _: () = assert!(TILE_MIX_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(ADJOINT_CORE_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);
const _: () = assert!(MAX_WEBGPU_STORAGE_BUFFERS_PER_STAGE == 8);
const _: () = assert!(GABOR_BEZOUT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(OZAKI_GRAM_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(INNER_OPT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(CORE_INPUT_VEC4_STRIDE == 2);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == 28);
const _: () = assert!(
    size_of::<CoreQuadV4ParamsStereo4<GpuVec4>>()
        == (QUAD_PARAMS_VEC4_STRIDE + 1) * size_of::<[f32; 4]>()
);
const _: () = assert!(GPU_OZAKI_MAX_CHUNK_LEN == 1024);
const _: () = assert!(GPU_SOFT_RECT_EPS > 0.0);
const _: () = assert!(TILE_CKPT_VEC4_STRIDE == 12);
const _: () = assert!(FEATURE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_LANE_VEC4_STRIDE == 11);
const _: () = assert!(TILE_ADJ_BOUNDARY_VEC4_STRIDE == 44);
const _: () = assert!(EXIT_ADJ_BOUNDARY_VEC4_STRIDE == 24);
const _: () = assert!(MAX_OUTGOING_PER_QUAD == 7 && MAX_SONA_TILES == 37);
const _: () = assert!(TILE_DESC_VEC4_STRIDE == 13);
const _: () = assert!(TILE_WEIGHT_GRAD_FLOATS == 48);
const _: () = assert!(PRESCREEN_SEED_PARAM_STRIDE == 16);
const _: () = assert!(OZAKI_WORDS_PER_PAIR == 6 && OZAKI_SPLIT64_WORDS_PER_PAIR == 4);
const _: () = assert!(LSSBROYDEN_MAX_HISTORY_M == 32 && LBFGS_MAX_HISTORY_M == 32);
const _: () = assert!(block_quad_count(0, 4) == 1 && block_quad_count(0, 10) == 3);
const _: () = assert!(block_quad_count(1, 10) == 3 && block_quad_count(2, 10) == 2);
const _: () = assert!(block_quad_count(3, 10) == 2);

const _: () = assert!(size_of::<PrescreenUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<PrescreenUniforms>() == 4);
const _: () = assert!(offset_of!(PrescreenUniforms, _pad1) == 28);

const _: () = assert!(size_of::<FusedScanUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<FusedScanUniforms>() == 4);
const _: () = assert!(offset_of!(FusedScanUniforms, block_idx) == 28);

const _: () = assert!(size_of::<AdjointScanUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<AdjointScanUniforms>() == 4);
const _: () = assert!(offset_of!(AdjointScanUniforms, block_idx) == 28);

const _: () = assert!(size_of::<TileMixUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<TileMixUniforms>() == 4);
const _: () = assert!(offset_of!(TileMixUniforms, _pad1) == 28);

const _: () = assert!(size_of::<GaborBezoutUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<GaborBezoutUniforms>() == 4);
const _: () = assert!(offset_of!(GaborBezoutUniforms, eps_noise) == 28);

const _: () = assert!(size_of::<OzakiGramUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<OzakiGramUniforms>() == 4);
const _: () = assert!(offset_of!(OzakiGramUniforms, _pad) == 28);

const _: () = assert!(size_of::<InnerOptUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<InnerOptUniforms>() == 4);
const _: () = assert!(offset_of!(InnerOptUniforms, grad_clip_norm) == 28);
