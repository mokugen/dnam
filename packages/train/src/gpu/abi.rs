//! Shared `32 B` `zerocopy` uniform structs, `15,872 B` workgroup SRAM ceilings, and
//! `CoreQuadParamsStereo4` `24 × Vec4` buffer layout strides (`ref[train.pipeline.webgpu-scans-and-ozaki]`,
//! `ref[sdk.rust.zerocopy-wire-structs]`, `ref[repo.code.constant-eval-and-derivations]`).

use core::mem::{align_of, offset_of, size_of};

use dnam_kernels::{CoreQuadParamsStereo4, GpuVec4, OZAKI_MAX_CHUNK_LEN, SOFT_RECT_EPS};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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
/// Forward `fused_core_scan_256` `lane_buf` capacity (`512 × Vec4 = 8,192 B`).
pub const FORWARD_LANE_BUF_VEC4S: usize = 2 * TILE_FRAMES;
/// Forward `fused_core_scan_256` `aux_scratch` capacity (`256 × f32 = 1,024 B`).
pub const FORWARD_AUX_SCRATCH_FLOATS: usize = TILE_FRAMES;

/// Total forward kernel workgroup SRAM footprint (`15,360 B = 15.0 KiB <= 15,872 B`).
pub const FUSED_CORE_SRAM_BYTES: usize = FORWARD_SCAN_SCRATCH_FLOATS * size_of::<f32>()
    + FORWARD_LANE_BUF_VEC4S * size_of::<[f32; 4]>()
    + FORWARD_AUX_SCRATCH_FLOATS * size_of::<f32>();

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

/// Slot indices (`0..24`) for the exact uncompressed `24 × [f32; 4]` packing of
/// [`dnam_kernels::CoreQuadParamsStereo4`] per `(stem, quad)` in `quad_params` & `param_grads`.
pub mod quad_param_slots {
    /// `TPT-SVF` `a1` coefficient (`1 / (1 + g * (g + k))`).
    pub const SVF_A1: usize = 0;
    /// `TPT-SVF` `a2` coefficient (`g * a1`).
    pub const SVF_A2: usize = 1;
    /// `TPT-SVF` `a3` coefficient (`g * a2`).
    pub const SVF_A3: usize = 2;
    /// `TPT-SVF` damping `k`.
    pub const SVF_K: usize = 3;
    /// `TPT-SVF` `2 * k`.
    pub const SVF_TWO_K: usize = 4;
    /// Tone lowpass tap weight.
    pub const TONE_LP: usize = 5;
    /// Tone bandpass tap weight.
    pub const TONE_BP: usize = 6;
    /// Tone highpass tap weight.
    pub const TONE_HP: usize = 7;
    /// Tone allpass tap weight.
    pub const TONE_AP: usize = 8;
    /// Tone direct bypass tap weight.
    pub const TONE_DIR: usize = 9;
    /// Selective-Duhem `ParaLRU` release pole `alpha_rel`.
    pub const LRU_ALPHA_REL: usize = 10;
    /// Selective-Duhem `ParaLRU` attack slew `alpha_atk`.
    pub const LRU_ALPHA_ATK: usize = 11;
    /// Selective-Duhem hysteresis loop weight `kappa`.
    pub const LRU_KAPPA: usize = 12;
    /// Selective-Duhem reactive flux modulation `gamma_duhem`.
    pub const LRU_GAMMA_DUHEM: usize = 13;
    /// Selective-Duhem odd/even rectifier blend `signed_mix`.
    pub const LRU_SIGNED_MIX: usize = 14;
    /// Selective-Duhem cross-channel envelope link `stereo_link`.
    pub const LRU_STEREO_LINK: usize = 15;
    /// `2 * svf_g` used for bandpass reactive flux calculation.
    pub const TWO_SVF_G: usize = 16;
    /// Rectification polarity sign (`+1.0` or `-1.0`).
    pub const RECT_SIGN: usize = 17;
    /// Envelope sag modulation depth.
    pub const SAG_DEPTH: usize = 18;
    /// `PolyADAA1` saturator drive.
    pub const ADAA_DRIVE: usize = 19;
    /// Envelope DC bias feed into saturator.
    pub const LRU_BIAS_DRIVE: usize = 20;
    /// $C^2$ wavefolder blend weight in `[0, 1]`.
    pub const WAVEFOLDER_MIX: usize = 21;
    /// Bounded cross-lane GLU modulation weight.
    pub const GLU_MIX: usize = 22;
    /// `TPT-SVF` frequency-warped `g` (`two_svf_g * 0.5`).
    pub const SVF_G: usize = 23;
    /// Total `[f32; 4]` slots per `(stem, quad)` (`24`).
    pub const TOTAL_SLOTS: usize = SVF_G + 1;
}

/// Number of `Vec4` (`f32[4]`) vectors per `(stem, quad, frame)` in `core_inputs` and `d_loss_d_inputs` (`L`, `R`).
pub const CORE_INPUT_VEC4_STRIDE: usize = 2;

/// Number of `Vec4` (`f32[4]`) vectors per `(stem, quad)` in `quad_params` and `param_grads`.
pub const QUAD_PARAMS_VEC4_STRIDE: usize = quad_param_slots::TOTAL_SLOTS;

/// Number of `Vec4` vectors per `(stem, q_local, tile)` checkpoint
/// `(s1_L, s1_R, s2_L, s2_R, v_L, v_R, adaa_prev_L, adaa_prev_R)`.
pub const TILE_CKPT_VEC4_STRIDE: usize = 8;

/// Number of `Vec4` vectors per `(stem, q_local, frame)` emitted into `feature_outputs`
/// `(v_eff_L, v_eff_R, adaa_odd_L, adaa_odd_R, adaa_even_L, adaa_even_R)`.
pub const FEATURE_VEC4_STRIDE: usize = 6;

/// Number of `Vec4` vectors per `(stem, q_local, tile)` in `tile_adj_boundaries`
/// (`4` lane channels × `4` summary `Vec4`s = `256 B`).
pub const TILE_ADJ_BOUNDARY_VEC4_STRIDE: usize = 4 * 4;

/// Number of `f32` scalars per seed in `prescreen_population`'s `seed_params`.
pub const PRESCREEN_SEED_PARAM_STRIDE: usize = 16;

/// Number of `i32` words per upper-triangle Ozaki pair `(d0, d1, d2, d3, d4, pad)`.
pub const OZAKI_WORDS_PER_PAIR: usize = 6;

/// Maximum GPU-resident L-BFGS history window `m`.
pub const LBFGS_MAX_HISTORY_M: usize = 8;

/// Prescreen workgroup SRAM footprint (`8,192 B <= 15,872 B`).
pub const PRESCREEN_SRAM_BYTES: u32 =
    ((6 * TILE_FRAMES + TILE_FRAMES + TILE_FRAMES) * size_of::<f32>()) as u32;

/// Gabor/Bézout workgroup SRAM footprint (`11,264 B <= 15,872 B`).
pub const GABOR_BEZOUT_SRAM_BYTES: u32 = (TILE_FRAMES * size_of::<[f32; 4]>()
    + TILE_FRAMES * size_of::<[f32; 4]>()
    + TILE_FRAMES * size_of::<[f32; 2]>()
    + TILE_FRAMES * size_of::<f32>()) as u32;

/// Ozaki Gram/RHS workgroup SRAM footprint (`5,120 B <= 15,872 B`).
pub const OZAKI_GRAM_SRAM_BYTES: u32 = (5 * TILE_FRAMES * size_of::<i32>()) as u32;

/// Inner L-BFGS optimizer workgroup SRAM footprint (`2,176 B <= 15,872 B`).
pub const INNER_OPT_SRAM_BYTES: u32 = ((TILE_FRAMES + TILE_FRAMES + 32) * size_of::<f32>()) as u32;

/// Packed `32 B` uniform block for `prescreen_population`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct PrescreenUniforms {
    /// Number of candidate parameter seeds evaluated in parallel.
    pub num_seeds: u32,
    /// Number of audio frames per stem.
    pub num_frames: u32,
    /// Number of `CoreQuad` stages summed per seed.
    pub num_quads: u32,
    /// Number of training stems.
    pub num_stems: u32,
    /// Audio sample rate in Hz (for diagnostic metadata).
    pub sample_rate_hz: f32,
    /// Energy denominator stabilization floor `eps_floor`.
    pub eps_floor: f32,
    /// Reserved alignment padding (`0`).
    pub _pad0: u32,
    /// Reserved alignment padding (`0`).
    pub _pad1: u32,
}

/// Packed `32 B` uniform block for `fused_core_scan_256`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FusedScanUniforms {
    /// Number of frames in this dispatch (`<= time_block_frames`).
    pub num_frames: u32,
    /// Number of tiles in checkpoint stride (`>= ceil(num_frames / 256)`).
    pub num_tiles: u32,
    /// Number of `CoreQuad` blocks in this dispatch.
    pub num_quads: u32,
    /// Number of stems in this dispatch (`<= stem_group`).
    pub num_stems: u32,
    /// Starting tile index (`> 0` resumes state from `tile_checkpoints[tile_start]`).
    pub tile_start: u32,
    /// Soft rectifier smoothing epsilon (`SOFT_RECT_EPS`).
    pub soft_rect_eps: f32,
    /// `PolyADAA1` tiny crossing distance threshold (`TINY_CROSSING_EPS`).
    pub tiny_crossing_eps: f32,
    /// Cascade block index in `0..4`.
    pub block_idx: u32,
}

/// Packed `32 B` uniform block for `adjoint_scan_256`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct AdjointScanUniforms {
    /// Total number of frames.
    pub num_frames: u32,
    /// Number of `256`-frame tiles.
    pub num_tiles: u32,
    /// Number of `CoreQuad` blocks.
    pub num_quads: u32,
    /// Number of training stems.
    pub num_stems: u32,
    /// Pass selector in `{0, 1, 2}` for 3-pass multi-tile reverse BPTT.
    pub pass_mode: u32,
    /// Soft rectifier smoothing epsilon (`SOFT_RECT_EPS`).
    pub soft_rect_eps: f32,
    /// `PolyADAA1` tiny crossing distance threshold (`TINY_CROSSING_EPS`).
    pub tiny_crossing_eps: f32,
    /// Cascade block index in `0..4`.
    pub block_idx: u32,
}

/// Packed `32 B` uniform block for `gabor_bezout_grad`.
///
/// Mode B (`fft_size > 0`) enforces `hop_size >= fft_size` per dispatch for disjoint writebacks.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct GaborBezoutUniforms {
    /// Total frames per stem.
    pub num_frames: u32,
    /// Radix-2 FFT length in `{16, 32, 64, 128, 256}`, or `0` for Mode A causal Bézout FIR.
    pub fft_size: u32,
    /// Window hop size (`>= fft_size` in Mode B) or Bézout tap count (`<= 64` in Mode A).
    pub hop_size: u32,
    /// Number of stems.
    pub num_stems: u32,
    /// Low-band perceptual ESR weight.
    pub w_lf: f32,
    /// Mid-band perceptual ESR weight.
    pub w_mf: f32,
    /// High-band perceptual ESR weight.
    pub w_hf: f32,
    /// Energy denominator stabilization floor.
    pub eps_noise: f32,
}

/// Packed `32 B` uniform block for `ozaki_gram_rhs_i32`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct OzakiGramUniforms {
    /// Total sample frames `N`.
    pub num_frames: u32,
    /// Feature dimension `D`.
    pub feature_dim: u32,
    /// Target channels (`0..=2`).
    pub target_channels: u32,
    /// Frame chunk length (`<= 1024`).
    pub chunk_len: u32,
    /// Number of chunks `ceil(num_frames / chunk_len)`.
    pub num_chunks: u32,
    /// Reserved alignment padding (`0`).
    pub _pad0: u32,
    /// Reserved alignment padding (`0`).
    pub _pad1: u32,
    /// Reserved alignment padding (`0`).
    pub _pad2: u32,
}

/// Packed `32 B` uniform block for `inner_opt_step`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct InnerOptUniforms {
    /// Number of optimization parameters.
    pub num_params: u32,
    /// Maximum L-BFGS history window `m <= 8`.
    pub history_m: u32,
    /// Outer optimization step index (`0..`).
    pub step_idx: u32,
    /// Armijo backtracking line-search trial (`0` computes direction; `> 0` reuses `search_dir`).
    pub line_search_iter: u32,
    /// Step size `alpha`.
    pub step_size: f32,
    /// Initial Hessian damping `lambda`.
    pub damping_lambda: f32,
    /// Interior box-constraint margin `min_bound_eps`.
    pub min_bound_eps: f32,
    /// Reserved alignment padding (`0`).
    pub _pad: u32,
}

const _: () = assert!(size_of::<PrescreenUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(size_of::<FusedScanUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(size_of::<AdjointScanUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(size_of::<GaborBezoutUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(size_of::<OzakiGramUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(size_of::<InnerOptUniforms>() == UNIFORM_STRUCT_BYTES);
const _: () = assert!(align_of::<PrescreenUniforms>() == 4);
const _: () = assert!(align_of::<FusedScanUniforms>() == 4);
const _: () = assert!(align_of::<AdjointScanUniforms>() == 4);
const _: () = assert!(align_of::<GaborBezoutUniforms>() == 4);
const _: () = assert!(align_of::<OzakiGramUniforms>() == 4);
const _: () = assert!(align_of::<InnerOptUniforms>() == 4);
const _: () = assert!(offset_of!(PrescreenUniforms, eps_floor) == 20);
const _: () = assert!(offset_of!(FusedScanUniforms, block_idx) == 28);
const _: () = assert!(offset_of!(AdjointScanUniforms, block_idx) == 28);
const _: () = assert!(offset_of!(GaborBezoutUniforms, eps_noise) == 28);
const _: () = assert!(offset_of!(OzakiGramUniforms, num_chunks) == 16);
const _: () = assert!(offset_of!(InnerOptUniforms, min_bound_eps) == 24);
const _: () = assert!(UNIFORM_STRUCT_BYTES == 32);
const _: () = assert!(WORKGROUP_SIZE == 256);
const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(MAX_WORKGROUP_SRAM_BYTES == 15_872);
const _: () = assert!(PRESCREEN_SRAM_BYTES == 8_192);
const _: () = assert!(FUSED_CORE_SRAM_BYTES == 15_360);
const _: () = assert!(ADJOINT_CORE_SRAM_BYTES == 14_848);
const _: () = assert!(GABOR_BEZOUT_SRAM_BYTES == 11_264);
const _: () = assert!(OZAKI_GRAM_SRAM_BYTES == 5_120);
const _: () = assert!(INNER_OPT_SRAM_BYTES == 2_176);
const _: () = assert!(PRESCREEN_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(FUSED_CORE_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);
const _: () = assert!(ADJOINT_CORE_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);
const _: () = assert!(GABOR_BEZOUT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(OZAKI_GRAM_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(INNER_OPT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES as u32);
const _: () = assert!(CORE_INPUT_VEC4_STRIDE == 2);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == 24);
const _: () = assert!(
    size_of::<CoreQuadParamsStereo4<GpuVec4>>()
        == (QUAD_PARAMS_VEC4_STRIDE - 1) * size_of::<[f32; 4]>()
);
const _: () = assert!(GPU_OZAKI_MAX_CHUNK_LEN == 1024);
const _: () = assert!(GPU_SOFT_RECT_EPS > 0.0);
const _: () = assert!(TILE_CKPT_VEC4_STRIDE == 8);
const _: () = assert!(FEATURE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_BOUNDARY_VEC4_STRIDE == 16);
const _: () = assert!(PRESCREEN_SEED_PARAM_STRIDE == 16);
const _: () = assert!(OZAKI_WORDS_PER_PAIR == 6);
const _: () = assert!(LBFGS_MAX_HISTORY_M == 8);
