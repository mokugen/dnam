//! Weights Directory & Cache-Line Arena Memory Mapping (`v3` `4,608 / 10,240 / 18,432 / 28,672 B`).
//!
//! Canonical Reference: `docs/specs/containers/dnam_binary_format.md` (`ref[container.dnam.tier-prefix]`).
//!
//! Exact `64 B`-aligned Matryoshka `v3` Arena Layout:
//! - `Arena 0` (`Lite`, `0..4,608 B`, `72 CL`): `s_router` (`0..64`), `flags/meta/stage0/os/tone` (`64..256`),
//!   `stage1_pre_lite` (`256..384`), `stage2_gru_lite` (`384..2,368`), `stage3_early_fir_lite` (`2,368..2,496`),
//!   `stage3_kautz_modals_lite` (`2,496..2,624`), `stage4_tank_lite` (`2,624..2,688`),
//!   `direct_readout_lite` (`2,688..3,008`), `ControlBus` (`3,008..4,608`).
//! - `Arena 1` (`Full`, `4,608..10,240 B`, `88 CL`, cum `160 CL`): `stage1_pre_full` .. `direct_readout_full`,
//!   plus `V3ControlBusParams` (`6,528..9,088`) & reserve (`9,088..10,240`).
//! - `Arena 2` (`Pro`, `10,240..18,432 B`, `128 CL`, cum `288 CL`): `2x` `stage1_pre_pro` .. `direct_readout_pro`
//!   & reserve (`15,616..18,432`).
//! - `Arena 3` (`Ultra`, `18,432..28,672 B`, `160 CL`, cum `448 CL`): `2x` `stage1_pre_ultra` ..
//!   `direct_readout_ultra`, `W_ctrl_ultra` (`21,120..21,376`), `U_int_ultra` (`21,376..21,760`),
//!   `CP1_ultra` (`21,760..23,872`), & reserve (`23,872..28,672`).

pub mod pack_v3;

pub use pack_v3::{
    GRU_FEED_BYTES, GRU_TILE_TRIPLE_BYTES, pack_v3_matryoshka_to_arena,
    unpack_v3_matryoshka_from_arena, unpack_v3_tier_from_arena, write_f32_le, write_u32_le,
};

use crate::types::Tier;

/// Size of one cache line in bytes (`64 B`).
pub const ARENA_CACHE_LINE_BYTES: usize = 64;

/// Cache-line deltas per arena (`72 / 88 / 128 / 160`).
pub const ARENA0_CL: usize = 72;
pub const ARENA1_CL: usize = 88;
pub const ARENA2_CL: usize = 128;
pub const ARENA3_CL: usize = 160;

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

// ============================================================================
// Arena 0 (Lite): Cumulative Byte Layout (`0..ARENA_LITE_BYTES`)
// ============================================================================
pub const ARENA0_ROUTER_OFFSET: usize = 0;
pub const ARENA0_ROUTER_BYTES: usize = ARENA_CACHE_LINE_BYTES;

pub const ARENA0_META_OFFSET: usize = ARENA0_ROUTER_OFFSET + ARENA0_ROUTER_BYTES;
pub const ARENA0_META_BYTES: usize = 3 * ARENA_CACHE_LINE_BYTES;

/// Pipeline flags `u32` offset inside `Arena 0` (`64..68`).
/// High 16 bits carry magic `0x444E` (`"DN"`); low 16 bits carry `PIPE_FLAG_*`.
pub const ARENA0_PIPELINE_FLAGS_OFFSET: usize = ARENA0_META_OFFSET;
pub const PIPE_MAGIC_MASK: u32 = 0xFFFF_0000;
pub const PIPE_MAGIC_HI: u32 = ((b'D' as u32) << 24) | ((b'N' as u32) << 16);
pub const PIPE_FLAG_ZERO_OP_T_PRE: u32 = 1 << 0;
pub const PIPE_FLAG_STAGE1_EXT: u32 = 1 << 1;
pub const PIPE_FLAG_STAGE2_EXT: u32 = 1 << 2;
pub const PIPE_FLAG_STAGE3_EXT: u32 = 1 << 3;
pub const PIPE_FLAG_STAGE4_EXT: u32 = 1 << 4;
pub const PIPE_FLAG_UNIFIED_W_CTRL: u32 = 1 << 5;
pub const PIPE_FLAG_V3_MATRYOSHKA: u32 = 1 << 6;
pub const PIPE_FLAG_ALL_V2: u32 = PIPE_FLAG_ZERO_OP_T_PRE
    | PIPE_FLAG_STAGE1_EXT
    | PIPE_FLAG_STAGE2_EXT
    | PIPE_FLAG_STAGE3_EXT
    | PIPE_FLAG_STAGE4_EXT
    | PIPE_FLAG_UNIFIED_W_CTRL;

pub const ARENA0_TIER_GAIN_TRIM_OFFSET: usize = ARENA0_PIPELINE_FLAGS_OFFSET + 4;
pub const ARENA0_V3_META_OFFSET: usize = ARENA0_TIER_GAIN_TRIM_OFFSET + 4 * 4;
pub const V3_META_STAGE3_PREFIX_OVERRIDES: u32 = 1 << 7;
pub const V3_META_STAGE4_TAP_GAINS: u32 = 1 << 16;
pub const V3_META_STAGE0_EXT: u32 = 1 << 17;
pub const V3_META_STAGE0_ROUTE_CLEAN: u32 = 1 << 18;
pub const V3_META_STAGE0_SUB2_PHASE: u32 = 1 << 19;
pub const V3_META_STAGE4_LFO: u32 = 1 << 20;

pub const ARENA0_DRY_WET_OFFSET: usize = ARENA0_V3_META_OFFSET + 4;
pub const ARENA0_R_AC_OFFSET: usize = ARENA0_DRY_WET_OFFSET + 4;
pub const ARENA0_STAGE0_PARAMS_OFFSET: usize = ARENA0_R_AC_OFFSET + 4;
pub const ARENA0_OS_PARAMS_OFFSET: usize = ARENA0_STAGE0_PARAMS_OFFSET + 7 * 4;
pub const ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET: usize = ARENA0_OS_PARAMS_OFFSET + 3 * 4;
pub const ARENA0_TONE_SVF_PARAMS_OFFSET: usize = ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET + 4;
pub const ARENA0_OS_EXT_PARAMS_OFFSET: usize = ARENA0_TONE_SVF_PARAMS_OFFSET + 14 * 4;
pub const ANTI_ZIPPER_MICRO_RAMP_SCALE_DEFAULT: f32 = 1.0 / 16.0;

pub const ARENA0_STAGE1_PRE_OFFSET: usize = ARENA0_META_OFFSET + ARENA0_META_BYTES;
pub const ARENA0_STAGE1_PRE_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

/// Legacy standalone extended Stage I offset (`360..488 B`, gated by `PIPE_FLAG_STAGE1_EXT`).
pub const ARENA0_STAGE1_EXT_OFFSET: usize = ARENA0_STAGE1_PRE_OFFSET + 104;
pub const ARENA0_STAGE1_EXT_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;
/// Legacy standalone extended Stage IV offset (gated by `PIPE_FLAG_STAGE4_EXT`).
pub const ARENA0_STAGE4_EXT_OFFSET: usize = ARENA_LITE_BYTES + 96;
pub const ARENA0_STAGE4_EXT_BYTES: usize = ARENA_CACHE_LINE_BYTES + 4;

pub const ARENA0_STAGE2_GRU_OFFSET: usize = ARENA0_STAGE1_PRE_OFFSET + ARENA0_STAGE1_PRE_BYTES;
pub const ARENA0_STAGE2_GRU_BYTES: usize = 31 * ARENA_CACHE_LINE_BYTES;

pub const ARENA0_STAGE0_EXT_OFFSET: usize = ARENA0_STAGE2_GRU_OFFSET + 30 * ARENA_CACHE_LINE_BYTES;
pub const ARENA0_STAGE0_ENV_MIX_OFFSET: usize = ARENA0_STAGE0_EXT_OFFSET + 32;
pub const ARENA0_FEAT_DET_THRESH_OFFSET: usize = ARENA0_STAGE0_ENV_MIX_OFFSET + 4;

pub const ARENA0_STAGE3_EARLY_FIR_OFFSET: usize =
    ARENA0_STAGE2_GRU_OFFSET + ARENA0_STAGE2_GRU_BYTES;
pub const ARENA0_STAGE3_EARLY_FIR_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA0_STAGE3_KAUTZ_OFFSET: usize =
    ARENA0_STAGE3_EARLY_FIR_OFFSET + ARENA0_STAGE3_EARLY_FIR_BYTES;
pub const ARENA0_STAGE3_KAUTZ_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA0_STAGE4_TANK_OFFSET: usize = ARENA0_STAGE3_KAUTZ_OFFSET + ARENA0_STAGE3_KAUTZ_BYTES;
pub const ARENA0_STAGE4_TANK_BYTES: usize = ARENA_CACHE_LINE_BYTES;

pub const ARENA0_DIRECT_READOUT_OFFSET: usize =
    ARENA0_STAGE4_TANK_OFFSET + ARENA0_STAGE4_TANK_BYTES;
pub const ARENA0_DIRECT_READOUT_BYTES: usize = 5 * ARENA_CACHE_LINE_BYTES;

pub const ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET: usize =
    ARENA0_DIRECT_READOUT_OFFSET + ARENA0_DIRECT_READOUT_BYTES;
pub const ARENA0_CONTROL_BUS_AND_CLAMPS_BYTES: usize = 25 * ARENA_CACHE_LINE_BYTES;
pub const ARENA0_CP1_DIR_OFFSET: usize = ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + 36;
pub const ARENA0_UINT_BASE_OFFSET: usize =
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + ARENA_CACHE_LINE_BYTES;
pub const ARENA0_CP1_BASE_OFFSET: usize = ARENA0_UINT_BASE_OFFSET + 2 * ARENA_CACHE_LINE_BYTES;
pub const ARENA0_CP1_BASE_END: usize = ARENA0_CP1_BASE_OFFSET + 5 * ARENA_CACHE_LINE_BYTES;
pub const ARENA0_W_CTRL_OFFSET: usize = ARENA0_CP1_BASE_END;
pub const ARENA0_W_CTRL_BYTES: usize = 16 * ARENA_CACHE_LINE_BYTES;
pub const ARENA0_RHO_HEAD_OFFSET: usize = ARENA0_W_CTRL_OFFSET + ARENA0_W_CTRL_BYTES;

const _: () = assert!(ARENA0_OS_EXT_PARAMS_OFFSET + 5 * 4 <= ARENA0_STAGE1_PRE_OFFSET);
const _: () = assert!(
    ARENA0_ROUTER_BYTES
        + ARENA0_META_BYTES
        + ARENA0_STAGE1_PRE_BYTES
        + ARENA0_STAGE2_GRU_BYTES
        + ARENA0_STAGE3_EARLY_FIR_BYTES
        + ARENA0_STAGE3_KAUTZ_BYTES
        + ARENA0_STAGE4_TANK_BYTES
        + ARENA0_DIRECT_READOUT_BYTES
        + ARENA0_CONTROL_BUS_AND_CLAMPS_BYTES
        == ARENA_LITE_BYTES
);
const _: () = assert!(ARENA0_RHO_HEAD_OFFSET + ARENA_CACHE_LINE_BYTES == ARENA_LITE_BYTES);

// ============================================================================
// Arena 1 (Full): Delta Tensors (`ARENA_LITE_BYTES..ARENA_FULL_BYTES`)
// ============================================================================
pub const ARENA1_STAGE1_PRE_OFFSET: usize = ARENA_LITE_BYTES;
pub const ARENA1_STAGE1_PRE_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA1_STAGE2_GRU_OFFSET: usize = ARENA1_STAGE1_PRE_OFFSET + ARENA1_STAGE1_PRE_BYTES;
pub const ARENA1_STAGE2_GRU_BYTES: usize = 17 * ARENA_CACHE_LINE_BYTES;

pub const ARENA1_STAGE3_EARLY_FIR_OFFSET: usize =
    ARENA1_STAGE2_GRU_OFFSET + ARENA1_STAGE2_GRU_BYTES;
pub const ARENA1_STAGE3_EARLY_FIR_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA1_STAGE3_KAUTZ_OFFSET: usize =
    ARENA1_STAGE3_EARLY_FIR_OFFSET + ARENA1_STAGE3_EARLY_FIR_BYTES;
pub const ARENA1_STAGE3_KAUTZ_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA1_STAGE4_TANK_OFFSET: usize = ARENA1_STAGE3_KAUTZ_OFFSET + ARENA1_STAGE3_KAUTZ_BYTES;
pub const ARENA1_STAGE4_TANK_BYTES: usize = ARENA_CACHE_LINE_BYTES;

pub const ARENA1_DIRECT_READOUT_OFFSET: usize =
    ARENA1_STAGE4_TANK_OFFSET + ARENA1_STAGE4_TANK_BYTES;
pub const ARENA1_DIRECT_READOUT_BYTES: usize = 6 * ARENA_CACHE_LINE_BYTES;

pub const ARENA1_CTRL_PARAMS_OFFSET: usize =
    ARENA1_DIRECT_READOUT_OFFSET + ARENA1_DIRECT_READOUT_BYTES;
pub const ARENA1_CTRL_PARAMS_BYTES: usize = 40 * ARENA_CACHE_LINE_BYTES;

pub const ARENA1_RESERVE_OFFSET: usize = ARENA1_CTRL_PARAMS_OFFSET + ARENA1_CTRL_PARAMS_BYTES;
pub const ARENA1_RESERVE_BYTES: usize = 18 * ARENA_CACHE_LINE_BYTES;

const _: () = assert!(
    ARENA1_STAGE1_PRE_BYTES
        + ARENA1_STAGE2_GRU_BYTES
        + ARENA1_STAGE3_EARLY_FIR_BYTES
        + ARENA1_STAGE3_KAUTZ_BYTES
        + ARENA1_STAGE4_TANK_BYTES
        + ARENA1_DIRECT_READOUT_BYTES
        + ARENA1_CTRL_PARAMS_BYTES
        + ARENA1_RESERVE_BYTES
        == ARENA1_BYTES
);
const _: () = assert!(ARENA_LITE_BYTES + ARENA1_BYTES == ARENA_FULL_BYTES);

// ============================================================================
// Arena 2 (Pro): Complete 2x Core `[2,2,1]` + Readout (`ARENA_FULL..ARENA_PRO`)
// ============================================================================
pub const ARENA2_STAGE1_PRE_OFFSET: usize = ARENA_FULL_BYTES;
pub const ARENA2_STAGE1_PRE_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA2_STAGE2_GRU_OFFSET: usize = ARENA2_STAGE1_PRE_OFFSET + ARENA2_STAGE1_PRE_BYTES;
pub const ARENA2_STAGE2_GRU_BYTES: usize = 70 * ARENA_CACHE_LINE_BYTES;

pub const ARENA2_STAGE3_EARLY_FIR_OFFSET: usize =
    ARENA2_STAGE2_GRU_OFFSET + ARENA2_STAGE2_GRU_BYTES;
pub const ARENA2_STAGE3_EARLY_FIR_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA2_STAGE3_KAUTZ_OFFSET: usize =
    ARENA2_STAGE3_EARLY_FIR_OFFSET + ARENA2_STAGE3_EARLY_FIR_BYTES;
pub const ARENA2_STAGE3_KAUTZ_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA2_STAGE4_TANK_OFFSET: usize = ARENA2_STAGE3_KAUTZ_OFFSET + ARENA2_STAGE3_KAUTZ_BYTES;
pub const ARENA2_STAGE4_TANK_BYTES: usize = ARENA_CACHE_LINE_BYTES;

pub const ARENA2_DIRECT_READOUT_OFFSET: usize =
    ARENA2_STAGE4_TANK_OFFSET + ARENA2_STAGE4_TANK_BYTES;
pub const ARENA2_DIRECT_READOUT_BYTES: usize = 7 * ARENA_CACHE_LINE_BYTES;

pub const ARENA2_RESERVE_OFFSET: usize = ARENA2_DIRECT_READOUT_OFFSET + ARENA2_DIRECT_READOUT_BYTES;
pub const ARENA2_RESERVE_BYTES: usize = 44 * ARENA_CACHE_LINE_BYTES;
pub const ARENA2_FEAT_DET_THRESH_OFFSET: usize = ARENA2_RESERVE_OFFSET + 16;

const _: () = assert!(
    ARENA2_STAGE1_PRE_BYTES
        + ARENA2_STAGE2_GRU_BYTES
        + ARENA2_STAGE3_EARLY_FIR_BYTES
        + ARENA2_STAGE3_KAUTZ_BYTES
        + ARENA2_STAGE4_TANK_BYTES
        + ARENA2_DIRECT_READOUT_BYTES
        + ARENA2_RESERVE_BYTES
        == ARENA2_BYTES
);
const _: () = assert!(ARENA_FULL_BYTES + ARENA2_BYTES == ARENA_PRO_BYTES);

// ============================================================================
// Arena 3 (Ultra): 2x `[3,2,1]` Delta + Ultra Ctrl/CP1 (`ARENA_PRO..ARENA_ULTRA`)
// ============================================================================
pub const ARENA3_STAGE1_PRE_OFFSET: usize = ARENA_PRO_BYTES;
pub const ARENA3_STAGE1_PRE_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_STAGE2_GRU_OFFSET: usize = ARENA3_STAGE1_PRE_OFFSET + ARENA3_STAGE1_PRE_BYTES;
pub const ARENA3_STAGE2_GRU_BYTES: usize = 26 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_STAGE3_EARLY_FIR_OFFSET: usize =
    ARENA3_STAGE2_GRU_OFFSET + ARENA3_STAGE2_GRU_BYTES;
pub const ARENA3_STAGE3_EARLY_FIR_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_STAGE3_KAUTZ_OFFSET: usize =
    ARENA3_STAGE3_EARLY_FIR_OFFSET + ARENA3_STAGE3_EARLY_FIR_BYTES;
pub const ARENA3_STAGE3_KAUTZ_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_STAGE4_TANK_OFFSET: usize = ARENA3_STAGE3_KAUTZ_OFFSET + ARENA3_STAGE3_KAUTZ_BYTES;
pub const ARENA3_STAGE4_TANK_BYTES: usize = 2 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_DIRECT_READOUT_OFFSET: usize =
    ARENA3_STAGE4_TANK_OFFSET + ARENA3_STAGE4_TANK_BYTES;
pub const ARENA3_DIRECT_READOUT_BYTES: usize = 8 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_W_CTRL_ULTRA_OFFSET: usize =
    ARENA3_DIRECT_READOUT_OFFSET + ARENA3_DIRECT_READOUT_BYTES;
pub const ARENA3_W_CTRL_ULTRA_BYTES: usize = 4 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_UINT_ULTRA_OFFSET: usize = ARENA3_W_CTRL_ULTRA_OFFSET + ARENA3_W_CTRL_ULTRA_BYTES;
pub const ARENA3_UINT_ULTRA_BYTES: usize = 6 * ARENA_CACHE_LINE_BYTES;

pub const ARENA3_CP1_ULTRA_OFFSET: usize = ARENA3_UINT_ULTRA_OFFSET + ARENA3_UINT_ULTRA_BYTES;
pub const ARENA3_CP1_ULTRA_BYTES: usize = 33 * ARENA_CACHE_LINE_BYTES;
pub const ARENA3_CP1_ULTRA_END: usize = ARENA3_CP1_ULTRA_OFFSET + ARENA3_CP1_ULTRA_BYTES;

pub const ARENA3_RESERVE_OFFSET: usize = ARENA3_CP1_ULTRA_END;
pub const ARENA3_RESERVE_BYTES: usize = 75 * ARENA_CACHE_LINE_BYTES;

pub const STAGE4_TAP_GAINS_OFFSETS: [usize; 4] = [
    ARENA0_STAGE0_EXT_OFFSET + 16,
    ARENA1_RESERVE_OFFSET,
    ARENA2_RESERVE_OFFSET,
    ARENA3_STAGE4_TANK_OFFSET + ARENA_CACHE_LINE_BYTES,
];
pub const STAGE4_LFO_OFFSETS: [usize; 4] = [
    ARENA0_FEAT_DET_THRESH_OFFSET + 4,
    STAGE4_TAP_GAINS_OFFSETS[1] + 16,
    ARENA2_FEAT_DET_THRESH_OFFSET + 4,
    STAGE4_TAP_GAINS_OFFSETS[3] + 16,
];
const _: () = assert!(STAGE4_LFO_OFFSETS[0] >= ARENA0_FEAT_DET_THRESH_OFFSET + 4);
const _: () = assert!(STAGE4_LFO_OFFSETS[0] + 12 <= ARENA0_STAGE3_EARLY_FIR_OFFSET);
const _: () = assert!(STAGE4_LFO_OFFSETS[1] >= STAGE4_TAP_GAINS_OFFSETS[1] + 16);
const _: () = assert!(STAGE4_LFO_OFFSETS[1] + 12 <= ARENA_FULL_BYTES);
const _: () = assert!(STAGE4_LFO_OFFSETS[2] >= ARENA2_FEAT_DET_THRESH_OFFSET + 4);
const _: () = assert!(STAGE4_LFO_OFFSETS[2] + 12 <= ARENA_PRO_BYTES);
const _: () = assert!(STAGE4_LFO_OFFSETS[3] >= STAGE4_TAP_GAINS_OFFSETS[3] + 16);
const _: () = assert!(STAGE4_LFO_OFFSETS[3] + 12 <= ARENA3_DIRECT_READOUT_OFFSET);

const _: () = assert!(
    ARENA3_STAGE1_PRE_BYTES
        + ARENA3_STAGE2_GRU_BYTES
        + ARENA3_STAGE3_EARLY_FIR_BYTES
        + ARENA3_STAGE3_KAUTZ_BYTES
        + ARENA3_STAGE4_TANK_BYTES
        + ARENA3_DIRECT_READOUT_BYTES
        + ARENA3_W_CTRL_ULTRA_BYTES
        + ARENA3_UINT_ULTRA_BYTES
        + ARENA3_CP1_ULTRA_BYTES
        + ARENA3_RESERVE_BYTES
        == ARENA3_BYTES
);
const _: () = assert!(ARENA_PRO_BYTES + ARENA3_BYTES == ARENA_ULTRA_BYTES);
const _: () = assert!(ARENA_LITE_CL + ARENA1_CL + ARENA2_CL + ARENA3_CL == ARENA_ULTRA_CL);

/// Per-tier Stage III prefix FIR/Kautz override slots (`Full: 192 B`, `Pro: 384 B`, `Ultra: 576 B`).
pub const ARENA1_STAGE3_PREFIX_OVERRIDE_OFFSET: usize = ARENA1_STAGE2_GRU_OFFSET + 704;
pub const ARENA2_STAGE3_PREFIX_OVERRIDE_OFFSET: usize = ARENA2_STAGE2_GRU_OFFSET + 3_840;
pub const ARENA3_STAGE3_PREFIX_OVERRIDE_OFFSET: usize = ARENA3_STAGE2_GRU_OFFSET + 1_024;
const _: () = assert!(ARENA1_STAGE3_PREFIX_OVERRIDE_OFFSET + 192 <= ARENA1_STAGE3_EARLY_FIR_OFFSET);
const _: () = assert!(ARENA2_STAGE3_PREFIX_OVERRIDE_OFFSET + 384 <= ARENA2_STAGE3_EARLY_FIR_OFFSET);
const _: () = assert!(ARENA3_STAGE3_PREFIX_OVERRIDE_OFFSET + 576 <= ARENA3_STAGE3_EARLY_FIR_OFFSET);

/// Read an explicit little-endian `f32` safely from an arena byte slice (returns `0.0` if out of bounds).
#[inline(always)]
pub fn read_f32_le(slice: &[u8], offset: usize) -> f32 {
    if offset + 4 <= slice.len() {
        f32::from_le_bytes([
            slice[offset],
            slice[offset + 1],
            slice[offset + 2],
            slice[offset + 3],
        ])
    } else {
        0.0
    }
}

/// Read an explicit little-endian `u32` safely from an arena byte slice.
#[inline(always)]
pub fn read_u32_le(slice: &[u8], offset: usize) -> u32 {
    if offset + 4 <= slice.len() {
        u32::from_le_bytes([
            slice[offset],
            slice[offset + 1],
            slice[offset + 2],
            slice[offset + 3],
        ])
    } else {
        0
    }
}

/// Returns `true` when the canonical pipeline flags magic (`0x444E`) and `flag` are set at byte `64..68`.
#[inline(always)]
pub fn has_pipeline_flag(arena: &[u8], flag: u32) -> bool {
    let word = read_u32_le(arena, ARENA0_PIPELINE_FLAGS_OFFSET);
    (word & PIPE_MAGIC_MASK) == PIPE_MAGIC_HI && (word & flag) != 0
}

/// Zero-copy directory view into the `28,672 B` (`v3` Matryoshka) tensor arena.
#[derive(Clone, Copy, Debug)]
pub struct ArenaDirectoryView<'a> {
    pub raw: &'a [u8],
    pub tier: Tier,
}

impl<'a> ArenaDirectoryView<'a> {
    /// Validates tier prefix capacity (`4,608 / 10,240 / 18,432 / 28,672 B`).
    pub fn new(bytes: &'a [u8], tier: Tier) -> Result<Self, &'static str> {
        let min_len = tier.expected_arena_bytes();
        if bytes.len() < min_len {
            return Err("[DNAM-WEIGHTS] Arena buffer smaller than tier requirement");
        }
        Ok(Self { raw: bytes, tier })
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
    fn test_v3_arena_directory_view_and_offsets() {
        let mut buf = vec![0u8; ARENA_ULTRA_BYTES];
        buf[ARENA0_PIPELINE_FLAGS_OFFSET..ARENA0_PIPELINE_FLAGS_OFFSET + 4]
            .copy_from_slice(&(PIPE_MAGIC_HI | PIPE_FLAG_V3_MATRYOSHKA).to_le_bytes());
        assert!(has_pipeline_flag(&buf, PIPE_FLAG_V3_MATRYOSHKA));

        write_f32_le(&mut buf, 68, 1.5);
        write_f32_le(&mut buf, 72, 2.25);

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let view = ArenaDirectoryView::new(&buf, tier).unwrap();
            assert_eq!(view.tier, tier);
            assert_eq!(view.read_f32(68), 1.5);
            let mut out = [0.0_f32; 2];
            view.read_f32_into(68, &mut out);
            assert_eq!(out, [1.5, 2.25]);
        }

        assert!(ArenaDirectoryView::new(&[0u8; 100], Tier::Lite).is_err());
        assert_eq!(read_f32_le(&[1, 2, 3], 0), 0.0);
        assert_eq!(read_u32_le(&[1, 2, 3], 0), 0);

        assert_eq!(ARENA_LITE_BYTES, 4_608);
        assert_eq!(ARENA_FULL_BYTES, 10_240);
        assert_eq!(ARENA_PRO_BYTES, 18_432);
        assert_eq!(ARENA_ULTRA_BYTES, 28_672);
    }
}
