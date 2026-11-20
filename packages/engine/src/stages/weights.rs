//! Declarative `#[repr(C)]` `zerocopy` `SONA` `v3.0` Arena Wire Layout (`4,608 / 7,936 / 15,872 / 30,720 B`).
//!
//! Canonical Reference: `docs/specs/containers/dnam_binary_format.md` (`ref[container.dnam.tier-prefix]`,
//! `ref[sdk.rust.zerocopy-wire-structs]`, `ref[repo.code.constant-eval-and-derivations]`).
//!
//! Exact `64 B`-aligned Matryoshka `SONA` Arena Layout:
//! - `Arena 0` (`Lite`, `0..4,608 B`, `72 CL`): [`Arena0LiteWire`] (`Stage0CrossbarWire`,
//!   `SonaCoreLiteWire`, `Stage3Fir32Wire`, `[KautzSoaBlockWire; 2]`, `Stage4QuadWire`,
//!   `VarProReadoutLiteWire`, `Arena0ControlBusWire`).
//! - `Arena 1` (`Full Delta`, `4,608..7,936 B`, `52 CL`, cum `124 CL`): [`Arena1FullDeltaWire`].
//! - `Arena 2` (`Pro Delta`, `7,936..15,872 B`, `124 CL`, cum `248 CL`): [`Arena2ProDeltaWire`].
//! - `Arena 3` (`Ultra Delta`, `15,872..30,720 B`, `232 CL`, cum `480 CL`): [`Arena3UltraDeltaWire`].

pub mod pack_sona;

pub use pack_sona::{
    pack_sona_matryoshka_to_arena, pack_v3_matryoshka_to_arena, unpack_sona_matryoshka_from_arena,
    unpack_sona_tier_from_arena, unpack_v3_matryoshka_from_arena, unpack_v3_tier_from_arena,
    write_f32_le, write_u32_le,
};

use core::mem::{offset_of, size_of};

use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::types::Tier;

/// Size of one cache line in bytes (`64 B`).
pub const ARENA_CACHE_LINE_BYTES: usize = 64;

/// Cache-line deltas per arena (`72 / 52 / 124 / 232`).
pub const ARENA0_CL: usize = 72;
pub const ARENA1_CL: usize = 52;
pub const ARENA2_CL: usize = 124;
pub const ARENA3_CL: usize = 232;

/// Cumulative cache-line and byte footprints across `Arena 0..3`.
pub const ARENA_LITE_CL: usize = ARENA0_CL;
pub const ARENA_LITE_BYTES: usize = ARENA_LITE_CL * ARENA_CACHE_LINE_BYTES;
pub const ARENA1_BYTES: usize = ARENA1_CL * ARENA_CACHE_LINE_BYTES;
pub const ARENA_FULL_CL: usize = ARENA_LITE_CL + ARENA1_CL;
pub const ARENA_FULL_BYTES: usize = ARENA_LITE_BYTES + ARENA1_BYTES;
pub const ARENA2_BYTES: usize = ARENA2_CL * ARENA_CACHE_LINE_BYTES;
pub const ARENA_PRO_CL: usize = ARENA_FULL_CL + ARENA2_CL;
pub const ARENA_PRO_BYTES: usize = ARENA_FULL_BYTES + ARENA2_BYTES;
pub const ARENA3_BYTES: usize = ARENA3_CL * ARENA_CACHE_LINE_BYTES;
pub const ARENA_ULTRA_CL: usize = ARENA_PRO_CL + ARENA3_CL;
pub const ARENA_ULTRA_BYTES: usize = ARENA_PRO_BYTES + ARENA3_BYTES;

/// Magic tag embedded inside [`SonaCellDescWire`].
pub const SONA_CELL_MAGIC: [u8; 4] = *b"SONA";

/// Pipeline flags word (`64..68`) carrying high `0x444E` (`"DN"`) magic and low `PIPE_FLAG_*`.
pub const PIPE_MAGIC_MASK: u32 = 0xFFFF_0000;
pub const PIPE_MAGIC_HI: u32 = ((b'D' as u32) << 24) | ((b'N' as u32) << 16);
pub const PIPE_FLAG_ZERO_OP_T_PRE: u32 = 1 << 0;
pub const PIPE_FLAG_STAGE1_EXT: u32 = 1 << 1;
pub const PIPE_FLAG_STAGE2_EXT: u32 = 1 << 2;
pub const PIPE_FLAG_STAGE3_EXT: u32 = 1 << 3;
pub const PIPE_FLAG_STAGE4_EXT: u32 = 1 << 4;
pub const PIPE_FLAG_UNIFIED_W_CTRL: u32 = 1 << 5;
pub const PIPE_FLAG_V3_MATRYOSHKA: u32 = 1 << 6;
pub const PIPE_FLAG_SONA: u32 = 1 << 7;
pub const PIPE_FLAG_ALL_SONA: u32 = PIPE_FLAG_STAGE3_EXT
    | PIPE_FLAG_STAGE4_EXT
    | PIPE_FLAG_UNIFIED_W_CTRL
    | PIPE_FLAG_V3_MATRYOSHKA
    | PIPE_FLAG_SONA;

/// `sona_meta_flags` / `v3_meta_flags` bit flags (`offset 84`, `dnam_binary_format.md` §2.3).
pub const V3_META_OS2X: u32 = 1 << 4;
pub const SONA_META_TIER_OS_EXPLICIT: u32 = 1 << 6;
pub const V3_META_STAGE3_PREFIX_OVERRIDES: u32 = 1 << 7;
pub const SONA_META_OS_EXT_PRESENT: u32 = 1 << 8;
pub const V3_META_CORE_MODS_PRESENT: u32 = 1 << 9;
pub const V3_META_STAGE4_TAP_GAINS: u32 = 1 << 16;
pub const V3_META_STAGE0_EXT: u32 = 1 << 17;
pub const V3_META_STAGE0_ROUTE_CLEAN: u32 = 1 << 18;
pub const V3_META_STAGE0_SUB2_PHASE: u32 = 1 << 19;
pub const V3_META_STAGE4_LFO: u32 = 1 << 20;

/// Default `3 kHz` -> host control micro-ramp inverse stride (`1 / 16`).
pub const ANTI_ZIPPER_MICRO_RAMP_SCALE_DEFAULT: f32 = 1.0 / 16.0;

// ============================================================================
// Declarative #[repr(C)] zerocopy Wire Structs (Arena 0..3)
// ============================================================================

/// Self-describing `SONA` `v3.0` cell descriptor inside `Stage0CrossbarWire` (`16 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaCellDescWire {
    pub magic: [u8; 4],
    pub version_major: U16,
    pub version_minor: U16,
    pub tier_flags: U16,
    pub oversample_factor: U16,
    pub cell_flags: U32,
}

/// Top-level flags, tier gain trims, Stage 0 matrix/crossover, and anti-zipper scale (`64 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaMetaWire {
    pub pipeline_flags: U32,
    pub tier_gain_trim: [F32; 4],
    pub v3_meta_flags: U32,
    pub dry_wet_mix: F32,
    pub r_ac: F32,
    pub stage0_gain: [F32; 2],
    pub stage0_mix_matrix: [[F32; 2]; 2],
    pub stage0_xo_hz: F32,
    pub anti_zipper_ramp_scale: F32,
}

/// Oversampler parameters, crossover band trims, and `Op-P` / `T1` Stage 0 config (`64 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaOsAndExtWire {
    pub os_up_quad_coeff: F32,
    pub os_dec_c_prev1: F32,
    pub os_dec_c0: F32,
    pub os_up_pre_emph: F32,
    pub os_dec_aa_mix: F32,
    pub os_up_aa_mix: F32,
    pub os_up_zoh_mix: F32,
    pub stage0_xo_low_gain: F32,
    pub stage0_xo_high_gain: F32,
    pub stage0_clean_sub_mix: F32,
    pub stage0_op_p_sub_mix: F32,
    pub stage0_op_p_lpf_pole: F32,
    pub stage0_op_p_env_pole: F32,
    pub stage0_op_p_sub2_mix: F32,
    pub stage0_op_p_oct_mix: F32,
    pub stage0_op_p_env_mix: F32,
}

/// `SonaCellDescWire`, `4x Q0.32` crossbar phasor frequencies/phases, and noise gate (`64 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaPhasorsAndDescWire {
    pub cell_desc: SonaCellDescWire,
    pub phasor_freq_hz: [F32; 4],
    pub phasor_init_phase_q32: [U32; 4],
    pub gate_and_crossbar_fp16: [U16; 8],
}

/// Unified Stage 0 router + Top-of-Frame crossbar header in `Arena 0` (`256 B`, `0..256`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage0CrossbarWire {
    pub s_router: [[F32; 4]; 4],
    pub meta: SonaMetaWire,
    pub os_ext: SonaOsAndExtWire,
    pub cell_phasors: SonaPhasorsAndDescWire,
}

/// Wire parameters for one `4`-lane stereo `SONA` core group (`192 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaGroupWire {
    pub svf_g: [F32; 4],
    pub svf_k: [F32; 4],
    pub lru_alpha_rel: [F32; 4],
    pub lru_alpha_atk: [F32; 4],
    pub lru_kappa: [F32; 4],
    pub adaa_drive: [F32; 4],
    pub lru_bias_drive: [F32; 4],
    pub in_x_weight: [F32; 4],
    pub tone_lp_fp16: [U16; 4],
    pub tone_bp_fp16: [U16; 4],
    pub tone_hp_fp16: [U16; 4],
    pub tone_ap_fp16: [U16; 4],
    pub tone_dir_fp16: [U16; 4],
    pub sag_depth_fp16: [U16; 4],
    pub wavefolder_mix_fp16: [U16; 4],
    pub rect_sign_fp16: [U16; 4],
}

/// Lower-triangular `4x4` inter-block spatial mixing tile triple (`odd`, `even`, `lru`, `128 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaInterBlockTilePairWire {
    pub w_odd_4x4: [[F32; 4]; 4],
    pub w_even_4x4_fp16: [[U16; 4]; 4],
    pub w_lru_4x4_fp16: [[U16; 4]; 4],
}

/// Crossbar coupling matrices (`Op-P` + `4x Q0.32` phasors + causal `Op-T` reads -> core, `256 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaCrossbarBlockWire {
    pub op_p_to_group_fp16: [[U16; 4]; 10],
    pub phasor_to_group_fp16: [[U16; 4]; 10],
    pub opt_to_lr_fp16: [[U16; 2]; 16],
    pub phasor_stereo_phase_fp16: [U16; 8],
    pub crossbar_reserved_fp16: [U16; 8],
}

/// `Lite` (`[1, 1, 1, 1]`) `SONA` core slice in `Arena 0` (`1,152 B` at `256..1,408`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaCoreLiteWire {
    pub groups: [SonaGroupWire; 4],
    pub tiles: [SonaInterBlockTilePairWire; 3],
}

/// `Full` (`[2, 2, 1, 1]`) `SONA` core delta slice in `Arena 1` (`1,920 B` at `4,608..6,528`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaCoreFullDeltaWire {
    pub groups: [SonaGroupWire; 2],
    pub tiles: [SonaInterBlockTilePairWire; 10],
    pub crossbar: SonaCrossbarBlockWire,
}

/// `Pro` (`[3, 2, 2, 1]`) `SONA` core delta slice in `Arena 2` (`4,608 B` at `7,936..12,544`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaCoreProDeltaWire {
    pub groups: [SonaGroupWire; 2],
    pub tiles: [SonaInterBlockTilePairWire; 10],
    pub crossbar_pro: SonaCrossbarBlockWire,
    pub reserve: [u8; 2_688],
}

/// `Ultra` (`[4, 3, 2, 1]`) `SONA` core delta slice in `Arena 3` (`6,656 B` at `15,872..22,528`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct SonaCoreUltraDeltaWire {
    pub groups: [SonaGroupWire; 2],
    pub tiles: [SonaInterBlockTilePairWire; 12],
    pub crossbar_ultra: SonaCrossbarBlockWire,
    pub stage_latent_core_mods_fp16: [U16; 1_024],
    pub reserve: [u8; 2_432],
}

/// Stage III `16`-tap stereo Early FIR slice (`128 B` for `Full`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage3Fir16Wire {
    pub fir_l: [F32; 16],
    pub fir_r: [F32; 16],
}

/// Stage III `32`-tap stereo Early FIR slice (`256 B` for `Lite` & `Ultra`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage3Fir32Wire {
    pub fir_l: [F32; 32],
    pub fir_r: [F32; 32],
}

/// Stage III `48`-tap stereo Early FIR slice (`384 B` for `Pro`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage3Fir48Wire {
    pub fir_l: [F32; 48],
    pub fir_r: [F32; 48],
}

/// Stage III `4`-pair shared-pole Kautz SoA modal block (`128 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct KautzSoaBlockWire {
    pub r_low: [F32; 4],
    pub dr: [F32; 4],
    pub theta: [F32; 4],
    pub b_gamma: [F32; 4],
    pub c_re_l: [F32; 4],
    pub c_im_l: [F32; 4],
    pub c_re_r: [F32; 4],
    pub c_im_r: [F32; 4],
}

/// Stage IV `4`-line `Op-T` tank quad, Schroeder diffuser delays/gains, LFO, and VCA (`128 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage4QuadWire {
    pub delay_lengths: [U32; 4],
    pub tap_gains: [F32; 4],
    pub allpass_delays: [U32; 4],
    pub allpass_g: [F32; 4],
    pub g_loop: F32,
    pub sigma_knee: F32,
    pub base_psi: F32,
    pub inject_gain: F32,
    pub dry_mix: F32,
    pub wet_mix: F32,
    pub r_dc: F32,
    pub vca_thresh: F32,
    pub vca_kappa: F32,
    pub vca_min_gain: F32,
    pub vca_att: F32,
    pub vca_rel: F32,
    pub lfo_rate_hz: F32,
    pub lfo_depth_samples: F32,
    pub lfo_stereo_phase: F32,
    pub ext_format: U32,
}

/// `Lite` direct `VarPro` readout subspace + stage-latent modulations (`640 B` at `2,048..2,688`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct VarProReadoutLiteWire {
    pub pre_cab_u_gain: [F32; 2],
    pub pre_cab_hp_gain: [F32; 2],
    pub direct_s2_u_gain: [F32; 2],
    pub direct_s2_hp_gain: [F32; 2],
    pub stage_latent_mods_fp16: [U16; 96],
    pub w_ro_fp16: [U16; 208],
}

/// Stage III `Full` prefix FIR (`16` taps/ch) & Kautz (`4` pairs) residue override slice (`192 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage3PrefixOverrideFullWire {
    pub fir_l: [F32; 16],
    pub fir_r: [F32; 16],
    pub c_re_l: [F32; 4],
    pub c_im_l: [F32; 4],
    pub c_re_r: [F32; 4],
    pub c_im_r: [F32; 4],
}

/// Stage III `Pro` prefix FIR (`32` taps/ch) & Kautz (`8` pairs) residue override slice (`384 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage3PrefixOverrideProWire {
    pub fir_l: [F32; 32],
    pub fir_r: [F32; 32],
    pub c_re_l: [F32; 8],
    pub c_im_l: [F32; 8],
    pub c_re_r: [F32; 8],
    pub c_im_r: [F32; 8],
}

/// Stage III `Ultra` prefix FIR (`48` taps/ch) & Kautz (`12` pairs) residue override slice (`576 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Stage3PrefixOverrideUltraWire {
    pub fir_l: [F32; 48],
    pub fir_r: [F32; 48],
    pub c_re_l: [F32; 12],
    pub c_im_l: [F32; 12],
    pub c_re_r: [F32; 12],
    pub c_im_r: [F32; 12],
}

/// `Full` direct `VarPro` readout delta (`1,024 B` at `6,912..7,936`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct VarProReadoutFullWire {
    pub prefix_override: Stage3PrefixOverrideFullWire,
    pub w_ro_and_mods_fp16: [U16; 416],
}

/// `Pro` direct `VarPro` readout delta (`2,304 B` at `13,568..15,872`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct VarProReadoutProWire {
    pub prefix_override: Stage3PrefixOverrideProWire,
    pub w_ro_and_mods_fp16: [U16; 960],
}

/// `Ultra` direct `VarPro` readout delta (`3,584 B` at `23,296..26,880`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct VarProReadoutUltraWire {
    pub prefix_override: Stage3PrefixOverrideUltraWire,
    pub w_ro_and_mods_fp16: [U16; 1_504],
}

/// Canonical inline `CP1` contributor directory inside `Arena0ControlBusWire` (`28 B` at `2,724..2,752`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Cp1InlineDirWire {
    pub cp1_dir_count: U16,
    pub flags: U16,
    pub entries: [[u8; 3]; 8],
}

/// Self-contained `Arena 0` `3 kHz` `ControlBus` descriptors, `U_int`, `CP1`, and `W_ctrl` (`1,920 B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Arena0ControlBusWire {
    pub lfo_dphi_train: [F32; 2],
    pub sigma_drift: F32,
    pub rho_drift_train: F32,
    pub env_poles: [F32; 3],
    pub dither_snr_scale: F32,
    pub trust_radius_mh: F32,
    pub cp1_dir: Cp1InlineDirWire,
    pub u_int_base: [[F32; 8]; 4],
    pub cp1_base: [u8; 576],
    pub w_ctrl: [[F32; 16]; 16],
    pub rho_head_and_ctrl_tail: [F32; 32],
}

/// `Arena 3` `Ultra` `8-D` `ControlBus` extensions (`3,840 B` at `26,880..30,720`).
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Arena3ControlBusWire {
    pub w_ctrl_ultra: [[F32; 16]; 4],
    pub u_int_ultra: [[[F32; 8]; 4]; 3],
    pub cp1_ultra: [u8; 2_112],
    pub stage_latent_ctrl_ultra_fp16: [U16; 256],
    pub reserve: [u8; 576],
}

/// Complete `Arena 0` (`Lite`, `0..4,608 B`, `72 CL`) wire payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Arena0LiteWire {
    pub stage0: Stage0CrossbarWire,
    pub sona_core: SonaCoreLiteWire,
    pub fir_lite: Stage3Fir32Wire,
    pub kautz_lite: [KautzSoaBlockWire; 2],
    pub stage4_lite: Stage4QuadWire,
    pub readout_lite: VarProReadoutLiteWire,
    pub ctrl_base: Arena0ControlBusWire,
}

/// Incremental `Arena 1` (`Full Delta`, `4,608..7,936 B`, `52 CL`, `3,328 B`) wire payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Arena1FullDeltaWire {
    pub sona_core_full: SonaCoreFullDeltaWire,
    pub fir_full: Stage3Fir16Wire,
    pub kautz_full: [KautzSoaBlockWire; 1],
    pub stage4_full: Stage4QuadWire,
    pub varpro_readout_full: VarProReadoutFullWire,
}

/// Incremental `Arena 2` (`Pro Delta`, `7,936..15,872 B`, `124 CL`, `7,936 B`) wire payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Arena2ProDeltaWire {
    pub sona_core_pro: SonaCoreProDeltaWire,
    pub fir_pro: Stage3Fir48Wire,
    pub kautz_pro: [KautzSoaBlockWire; 3],
    pub stage4_pro: [Stage4QuadWire; 2],
    pub varpro_readout_pro: VarProReadoutProWire,
}

/// Incremental `Arena 3` (`Ultra Delta`, `15,872..30,720 B`, `232 CL`, `14,848 B`) wire payload.
#[repr(C)]
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct Arena3UltraDeltaWire {
    pub sona_core_ultra: SonaCoreUltraDeltaWire,
    pub fir_ultra: Stage3Fir32Wire,
    pub kautz_ultra: [KautzSoaBlockWire; 2],
    pub stage4_ultra: [Stage4QuadWire; 2],
    pub varpro_readout_ultra: VarProReadoutUltraWire,
    pub ctrl_ultra: Arena3ControlBusWire,
}

// ============================================================================
// Canonical Byte Offset Constants & Compile-Time Layout Assertions
// ============================================================================

pub const ARENA0_ROUTER_OFFSET: usize = offset_of!(Arena0LiteWire, stage0);
pub const ARENA0_ROUTER_BYTES: usize = 64;
pub const ARENA0_META_OFFSET: usize = ARENA0_ROUTER_OFFSET + offset_of!(Stage0CrossbarWire, meta);
pub const ARENA0_PIPELINE_FLAGS_OFFSET: usize =
    ARENA0_META_OFFSET + offset_of!(SonaMetaWire, pipeline_flags);
pub const ARENA0_TIER_GAIN_TRIM_OFFSET: usize =
    ARENA0_META_OFFSET + offset_of!(SonaMetaWire, tier_gain_trim);
pub const ARENA0_V3_META_OFFSET: usize =
    ARENA0_META_OFFSET + offset_of!(SonaMetaWire, v3_meta_flags);
pub const ARENA0_DRY_WET_OFFSET: usize = ARENA0_META_OFFSET + offset_of!(SonaMetaWire, dry_wet_mix);
pub const ARENA0_R_AC_OFFSET: usize = ARENA0_META_OFFSET + offset_of!(SonaMetaWire, r_ac);
pub const ARENA0_STAGE0_PARAMS_OFFSET: usize =
    ARENA0_META_OFFSET + offset_of!(SonaMetaWire, stage0_gain);
pub const ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET: usize =
    ARENA0_META_OFFSET + offset_of!(SonaMetaWire, anti_zipper_ramp_scale);
pub const ARENA0_OS_PARAMS_OFFSET: usize =
    ARENA0_ROUTER_OFFSET + offset_of!(Stage0CrossbarWire, os_ext);

pub const ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET: usize = offset_of!(Arena0LiteWire, ctrl_base);
pub const ARENA0_CP1_DIR_OFFSET: usize =
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + offset_of!(Arena0ControlBusWire, cp1_dir);
pub const ARENA0_UINT_BASE_OFFSET: usize =
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + offset_of!(Arena0ControlBusWire, u_int_base);
pub const ARENA0_CP1_BASE_OFFSET: usize =
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + offset_of!(Arena0ControlBusWire, cp1_base);
pub const ARENA0_CP1_BASE_END: usize = ARENA0_CP1_BASE_OFFSET + 576;
pub const ARENA0_W_CTRL_OFFSET: usize =
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + offset_of!(Arena0ControlBusWire, w_ctrl);
pub const ARENA0_RHO_HEAD_OFFSET: usize =
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + offset_of!(Arena0ControlBusWire, rho_head_and_ctrl_tail);

pub const ARENA3_CTRL_OFFSET: usize =
    ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, ctrl_ultra);
pub const ARENA3_W_CTRL_ULTRA_OFFSET: usize =
    ARENA3_CTRL_OFFSET + offset_of!(Arena3ControlBusWire, w_ctrl_ultra);
pub const ARENA3_UINT_ULTRA_OFFSET: usize =
    ARENA3_CTRL_OFFSET + offset_of!(Arena3ControlBusWire, u_int_ultra);
pub const ARENA3_CP1_ULTRA_OFFSET: usize =
    ARENA3_CTRL_OFFSET + offset_of!(Arena3ControlBusWire, cp1_ultra);
pub const ARENA3_CP1_ULTRA_END: usize = ARENA3_CP1_ULTRA_OFFSET + 2_112;

pub const STAGE3_EARLY_FIR_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, fir_lite),
    ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, fir_full),
    ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, fir_pro),
    ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, fir_ultra),
];
pub const STAGE3_KAUTZ_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, kautz_lite),
    ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, kautz_full),
    ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, kautz_pro),
    ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, kautz_ultra),
];
pub const STAGE3_KAUTZ_BLOCKS_PER_TIER: [usize; 4] = [2, 1, 3, 2];
pub const STAGE3_KAUTZ_TIER_CHANNEL_BASE: [usize; 4] = [0, 8, 12, 24];

pub const STAGE4_DELAY_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, stage4_lite),
    ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, stage4_full),
    ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, stage4_pro),
    ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, stage4_ultra),
];

pub const VARPRO_READOUT_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, readout_lite),
    ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, varpro_readout_full),
    ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, varpro_readout_pro),
    ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, varpro_readout_ultra),
];

pub const ARENA1_STAGE3_PREFIX_OVERRIDE_OFFSET: usize = VARPRO_READOUT_BASES[1];
pub const ARENA2_STAGE3_PREFIX_OVERRIDE_OFFSET: usize = VARPRO_READOUT_BASES[2];
pub const ARENA3_STAGE3_PREFIX_OVERRIDE_OFFSET: usize = VARPRO_READOUT_BASES[3];

const _: () = assert!(size_of::<SonaCellDescWire>() == 16 && align_of::<SonaCellDescWire>() == 1);
const _: () = assert!(size_of::<SonaMetaWire>() == 64 && align_of::<SonaMetaWire>() == 1);
const _: () = assert!(size_of::<SonaOsAndExtWire>() == 64 && align_of::<SonaOsAndExtWire>() == 1);
const _: () =
    assert!(size_of::<SonaPhasorsAndDescWire>() == 64 && align_of::<SonaPhasorsAndDescWire>() == 1);
const _: () =
    assert!(size_of::<Stage0CrossbarWire>() == 256 && align_of::<Stage0CrossbarWire>() == 1);
const _: () = assert!(size_of::<SonaGroupWire>() == 192 && align_of::<SonaGroupWire>() == 1);
const _: () = assert!(
    size_of::<SonaInterBlockTilePairWire>() == 128 && align_of::<SonaInterBlockTilePairWire>() == 1
);
const _: () =
    assert!(size_of::<SonaCrossbarBlockWire>() == 256 && align_of::<SonaCrossbarBlockWire>() == 1);
const _: () =
    assert!(size_of::<SonaCoreLiteWire>() == 1_152 && align_of::<SonaCoreLiteWire>() == 1);
const _: () = assert!(
    size_of::<SonaCoreFullDeltaWire>() == 1_920 && align_of::<SonaCoreFullDeltaWire>() == 1
);
const _: () =
    assert!(size_of::<SonaCoreProDeltaWire>() == 4_608 && align_of::<SonaCoreProDeltaWire>() == 1);
const _: () = assert!(
    size_of::<SonaCoreUltraDeltaWire>() == 6_656 && align_of::<SonaCoreUltraDeltaWire>() == 1
);
const _: () = assert!(size_of::<Stage3Fir16Wire>() == 128 && align_of::<Stage3Fir16Wire>() == 1);
const _: () = assert!(size_of::<Stage3Fir32Wire>() == 256 && align_of::<Stage3Fir32Wire>() == 1);
const _: () = assert!(size_of::<Stage3Fir48Wire>() == 384 && align_of::<Stage3Fir48Wire>() == 1);
const _: () =
    assert!(size_of::<KautzSoaBlockWire>() == 128 && align_of::<KautzSoaBlockWire>() == 1);
const _: () = assert!(size_of::<Stage4QuadWire>() == 128 && align_of::<Stage4QuadWire>() == 1);
const _: () =
    assert!(size_of::<VarProReadoutLiteWire>() == 640 && align_of::<VarProReadoutLiteWire>() == 1);
const _: () = assert!(
    size_of::<Stage3PrefixOverrideFullWire>() == 192
        && align_of::<Stage3PrefixOverrideFullWire>() == 1
);
const _: () = assert!(
    size_of::<Stage3PrefixOverrideProWire>() == 384
        && align_of::<Stage3PrefixOverrideProWire>() == 1
);
const _: () = assert!(
    size_of::<Stage3PrefixOverrideUltraWire>() == 576
        && align_of::<Stage3PrefixOverrideUltraWire>() == 1
);
const _: () = assert!(
    size_of::<VarProReadoutFullWire>() == 1_024 && align_of::<VarProReadoutFullWire>() == 1
);
const _: () =
    assert!(size_of::<VarProReadoutProWire>() == 2_304 && align_of::<VarProReadoutProWire>() == 1);
const _: () = assert!(
    size_of::<VarProReadoutUltraWire>() == 3_584 && align_of::<VarProReadoutUltraWire>() == 1
);
const _: () = assert!(size_of::<Cp1InlineDirWire>() == 28 && align_of::<Cp1InlineDirWire>() == 1);
const _: () =
    assert!(size_of::<Arena0ControlBusWire>() == 1_920 && align_of::<Arena0ControlBusWire>() == 1);
const _: () =
    assert!(size_of::<Arena3ControlBusWire>() == 3_840 && align_of::<Arena3ControlBusWire>() == 1);
const _: () = assert!(
    size_of::<Arena0LiteWire>() == 4_608
        && align_of::<Arena0LiteWire>() == 1
        && ARENA_LITE_BYTES == 4_608
);
const _: () = assert!(
    size_of::<Arena1FullDeltaWire>() == 3_328
        && align_of::<Arena1FullDeltaWire>() == 1
        && ARENA_FULL_BYTES == 7_936
);
const _: () = assert!(
    size_of::<Arena2ProDeltaWire>() == 7_936
        && align_of::<Arena2ProDeltaWire>() == 1
        && ARENA_PRO_BYTES == 15_872
);
const _: () = assert!(
    size_of::<Arena3UltraDeltaWire>() == 14_848
        && align_of::<Arena3UltraDeltaWire>() == 1
        && ARENA_ULTRA_BYTES == 30_720
);
const _: () = assert!(
    ARENA_PRO_BYTES
        + offset_of!(Arena3UltraDeltaWire, sona_core_ultra)
        + offset_of!(SonaCoreUltraDeltaWire, stage_latent_core_mods_fp16)
        == 18_048
);
const _: () = assert!(
    ARENA3_CTRL_OFFSET + offset_of!(Arena3ControlBusWire, stage_latent_ctrl_ultra_fp16) == 29_632
);
const _: () = assert!(ARENA_LITE_BYTES + size_of::<Arena1FullDeltaWire>() == ARENA_FULL_BYTES);
const _: () = assert!(ARENA_FULL_BYTES + size_of::<Arena2ProDeltaWire>() == ARENA_PRO_BYTES);
const _: () = assert!(ARENA_PRO_BYTES + size_of::<Arena3UltraDeltaWire>() == ARENA_ULTRA_BYTES);
const _: () = assert!(ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET == 2_688);
const _: () = assert!(ARENA0_CP1_DIR_OFFSET == 2_724);
const _: () = assert!(ARENA0_UINT_BASE_OFFSET == 2_752);
const _: () = assert!(ARENA0_CP1_BASE_OFFSET == 2_880 && ARENA0_CP1_BASE_END == 3_456);
const _: () = assert!(ARENA0_W_CTRL_OFFSET == 3_456 && ARENA0_RHO_HEAD_OFFSET == 4_480);
const _: () = assert!(ARENA3_W_CTRL_ULTRA_OFFSET == 26_880 && ARENA3_UINT_ULTRA_OFFSET == 27_136);
const _: () = assert!(ARENA3_CP1_ULTRA_OFFSET == 27_520 && ARENA3_CP1_ULTRA_END == 29_632);
const _: () = assert!(STAGE3_EARLY_FIR_BASES[0] == 1_408 && STAGE3_EARLY_FIR_BASES[1] == 6_528);
const _: () = assert!(STAGE3_EARLY_FIR_BASES[2] == 12_544 && STAGE3_EARLY_FIR_BASES[3] == 22_528);
const _: () = assert!(STAGE3_KAUTZ_BASES[0] == 1_664 && STAGE3_KAUTZ_BASES[1] == 6_656);
const _: () = assert!(STAGE3_KAUTZ_BASES[2] == 12_928 && STAGE3_KAUTZ_BASES[3] == 22_784);
const _: () = assert!(STAGE4_DELAY_BASES[0] == 1_920 && STAGE4_DELAY_BASES[1] == 6_784);
const _: () = assert!(STAGE4_DELAY_BASES[2] == 13_312 && STAGE4_DELAY_BASES[3] == 23_040);
const _: () = assert!(VARPRO_READOUT_BASES[0] == 2_048 && VARPRO_READOUT_BASES[1] == 6_912);
const _: () = assert!(VARPRO_READOUT_BASES[2] == 13_568 && VARPRO_READOUT_BASES[3] == 23_296);

/// Reads a little-endian `f32` via `zerocopy` (returns `0.0` when out of bounds).
#[inline(always)]
pub fn read_f32_le(slice: &[u8], offset: usize) -> f32 {
    slice
        .get(offset..)
        .and_then(|tail| F32::read_from_prefix(tail).ok())
        .map_or(0.0, |(v, _)| v.get())
}

/// Reads a little-endian `u32` via `zerocopy` (returns `0` when out of bounds).
#[inline(always)]
pub fn read_u32_le(slice: &[u8], offset: usize) -> u32 {
    slice
        .get(offset..)
        .and_then(|tail| U32::read_from_prefix(tail).ok())
        .map_or(0, |(v, _)| v.get())
}

/// Returns `true` when the canonical pipeline flags magic (`0x444E`) and `flag` are set at byte `64..68`.
#[inline(always)]
pub fn has_pipeline_flag(arena: &[u8], flag: u32) -> bool {
    let word = read_u32_le(arena, ARENA0_PIPELINE_FLAGS_OFFSET);
    (word & PIPE_MAGIC_MASK) == PIPE_MAGIC_HI && (word & flag) != 0
}

/// Zero-copy `zerocopy` directory view into the `SONA` `v3.0` (`4,608..30,720 B`) tensor arena.
#[derive(Clone, Copy, Debug)]
pub struct ArenaDirectoryView<'a> {
    pub raw: &'a [u8],
    pub tier: Tier,
}

impl<'a> ArenaDirectoryView<'a> {
    /// Validates tier prefix capacity (`4,608 / 7,936 / 15,872 / 30,720 B`).
    pub fn new(bytes: &'a [u8], tier: Tier) -> Result<Self, &'static str> {
        if bytes.len() < tier.expected_arena_bytes() {
            return Err("[DNAM-WEIGHTS] Arena buffer smaller than tier requirement");
        }
        Ok(Self { raw: bytes, tier })
    }

    /// Zero-copy view of `Arena 0` (`Lite`, `4,608 B`).
    #[inline]
    pub fn arena0_wire(&self) -> Option<&'a Arena0LiteWire> {
        Arena0LiteWire::ref_from_prefix(self.raw)
            .ok()
            .map(|(w, _)| w)
    }

    /// Zero-copy view of `Arena 1` (`Full Delta`, `3,328 B`).
    #[inline]
    pub fn arena1_wire(&self) -> Option<&'a Arena1FullDeltaWire> {
        self.raw
            .get(ARENA_LITE_BYTES..)
            .and_then(|tail| Arena1FullDeltaWire::ref_from_prefix(tail).ok())
            .map(|(w, _)| w)
    }

    /// Zero-copy view of `Arena 2` (`Pro Delta`, `7,936 B`).
    #[inline]
    pub fn arena2_wire(&self) -> Option<&'a Arena2ProDeltaWire> {
        self.raw
            .get(ARENA_FULL_BYTES..)
            .and_then(|tail| Arena2ProDeltaWire::ref_from_prefix(tail).ok())
            .map(|(w, _)| w)
    }

    /// Zero-copy view of `Arena 3` (`Ultra Delta`, `14,848 B`).
    #[inline]
    pub fn arena3_wire(&self) -> Option<&'a Arena3UltraDeltaWire> {
        self.raw
            .get(ARENA_PRO_BYTES..)
            .and_then(|tail| Arena3UltraDeltaWire::ref_from_prefix(tail).ok())
            .map(|(w, _)| w)
    }

    /// Reads an `f32` parameter at `byte_offset`.
    #[inline(always)]
    pub fn read_f32(&self, byte_offset: usize) -> f32 {
        read_f32_le(self.raw, byte_offset)
    }

    /// Copies `count` floats starting at `byte_offset` into `dst`.
    pub fn read_f32_into(&self, byte_offset: usize, dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = self.read_f32(byte_offset + i * 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_sona_arena_directory_view_and_zerocopy_wires() {
        let mut buf = vec![0_u8; ARENA_ULTRA_BYTES];
        let a0 = Arena0LiteWire::mut_from_prefix(&mut buf).unwrap().0;
        a0.stage0.meta.pipeline_flags = U32::new(PIPE_MAGIC_HI | PIPE_FLAG_SONA);
        a0.stage0.meta.tier_gain_trim =
            [F32::new(1.5), F32::new(2.25), F32::new(1.0), F32::new(1.0)];
        a0.stage0.cell_phasors.cell_desc.magic = SONA_CELL_MAGIC;
        a0.stage0.cell_phasors.cell_desc.version_major = U16::new(3);
        a0.stage0.cell_phasors.cell_desc.version_minor = U16::new(0);

        assert!(has_pipeline_flag(&buf, PIPE_FLAG_SONA));
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let view = ArenaDirectoryView::new(&buf, tier).unwrap();
            assert_eq!(view.tier, tier);
            assert_eq!(view.read_f32(68), 1.5);
            let w0 = view.arena0_wire().unwrap();
            assert_eq!(w0.stage0.cell_phasors.cell_desc.magic, *b"SONA");
            assert_eq!(w0.stage0.cell_phasors.cell_desc.version_major.get(), 3);
            let mut out = [0.0_f32; 2];
            view.read_f32_into(68, &mut out);
            assert_eq!(out, [1.5, 2.25]);
        }

        assert!(ArenaDirectoryView::new(&[0_u8; 100], Tier::Lite).is_err());
        assert_eq!(read_f32_le(&[1, 2, 3], 0), 0.0);
        assert_eq!(read_u32_le(&[1, 2, 3], 0), 0);
        assert_eq!(ARENA_LITE_BYTES, 4_608);
        assert_eq!(ARENA_FULL_BYTES, 7_936);
        assert_eq!(ARENA_PRO_BYTES, 15_872);
        assert_eq!(ARENA_ULTRA_BYTES, 30_720);
    }
}
