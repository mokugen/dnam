//! Canonical `0..7,680` Disjoint Coordinate Partition (`64` Role A + `5,811` Role B + `1,805` Unassigned)
//! derived strictly from `stages/weights.rs` `#[repr(C)]` wire offsets (`ref[sec.watermark.tardos-code]`,
//! `ref[repo.code.constant-eval-and-derivations]`).

use core::mem::offset_of;

use crate::provenance::TARDOS_CASCADE_WEIGHTS;
use crate::stages::weights::{
    ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, Arena0ControlBusWire,
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3ControlBusWire,
    Arena3UltraDeltaWire, KautzSoaBlockWire, SonaCoreProDeltaWire, SonaCoreUltraDeltaWire,
    Stage4QuadWire, VarProReadoutFullWire, VarProReadoutLiteWire, VarProReadoutProWire,
    VarProReadoutUltraWire,
};

/// Descriptor for one contiguous carrier span in the `0..7,680` Tardos partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TardosSpan {
    pub tier: u8,
    pub start_idx: usize,
    pub end_idx: usize,
    pub byte_base: usize,
    pub is_w_ctrl_role_b: bool,
}

impl TardosSpan {
    #[inline]
    pub const fn len(&self) -> usize {
        self.end_idx - self.start_idx
    }
}

/// Cumulative `[start, end, byte_base)` tier headers for backward API compatibility (`w_ctrl[0]` Role A starts each tier).
pub const TARDOS_CASCADE_RANGES: [(usize, usize, usize); 4] = [
    (
        0,
        383,
        offset_of!(Arena0LiteWire, ctrl_base) + offset_of!(Arena0ControlBusWire, w_ctrl),
    ),
    (
        655,
        751,
        ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, varpro_readout_full),
    ),
    (
        1_159,
        2_103,
        ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, varpro_readout_pro),
    ),
    (
        3_051,
        3_995,
        ARENA_PRO_BYTES
            + offset_of!(Arena3UltraDeltaWire, ctrl_ultra)
            + offset_of!(Arena3ControlBusWire, w_ctrl_ultra),
    ),
];

/// `Arena 3` (`Ultra`) `ControlBus` reserve `f32` span (`3,995..4,139` at `30,144..30,720`, `144` Role B carriers).
pub const TARDOS_ULTRA_RESERVE_RANGE: (usize, usize, usize) = (
    3_995,
    4_139,
    ARENA_PRO_BYTES
        + offset_of!(Arena3UltraDeltaWire, ctrl_ultra)
        + offset_of!(Arena3ControlBusWire, reserve),
);

/// Contiguous linear `fp16` (`U16`) Role B span bounds per tier `[start, end, byte_base)`.
pub const TARDOS_W_READOUT_FP16_RANGES: [(usize, usize, usize); 4] = [
    (
        383,
        655,
        offset_of!(Arena0LiteWire, readout_lite)
            + offset_of!(VarProReadoutLiteWire, stage_latent_mods_fp16),
    ),
    (
        751,
        1_159,
        ARENA_LITE_BYTES
            + offset_of!(Arena1FullDeltaWire, varpro_readout_full)
            + offset_of!(VarProReadoutFullWire, w_ro_and_mods_fp16),
    ),
    (
        2_103,
        3_051,
        ARENA_FULL_BYTES
            + offset_of!(Arena2ProDeltaWire, varpro_readout_pro)
            + offset_of!(VarProReadoutProWire, w_ro_and_mods_fp16),
    ),
    (
        4_139,
        5_875,
        ARENA_PRO_BYTES
            + offset_of!(Arena3UltraDeltaWire, varpro_readout_ultra)
            + offset_of!(VarProReadoutUltraWire, w_ro_and_mods_fp16),
    ),
];

/// Exact per-tier coordinate bounds `(tier_start_idx, tier_end_idx)` across `0..5,875` (`[655, 504, 1892, 2824]`).
pub const TARDOS_TIER_INDEX_BOUNDS: [(usize, usize); 4] =
    [(0, 655), (655, 1_159), (1_159, 3_051), (3_051, 5_875)];

const A0_WCTRL: usize =
    offset_of!(Arena0LiteWire, ctrl_base) + offset_of!(Arena0ControlBusWire, w_ctrl);
const A0_KAUTZ: usize = offset_of!(Arena0LiteWire, kautz_lite);
const A0_RO: usize = offset_of!(Arena0LiteWire, readout_lite);
const A0_TAIL: usize = offset_of!(Arena0LiteWire, ctrl_base)
    + offset_of!(Arena0ControlBusWire, rho_head_and_ctrl_tail);

const A1_KAUTZ: usize = ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, kautz_full);
const A1_RO: usize = ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, varpro_readout_full);

const A2_PRO_RES: usize = ARENA_FULL_BYTES
    + offset_of!(Arena2ProDeltaWire, sona_core_pro)
    + offset_of!(SonaCoreProDeltaWire, reserve);
const A2_S4_1: usize = ARENA_FULL_BYTES
    + offset_of!(Arena2ProDeltaWire, stage4_pro)
    + core::mem::size_of::<Stage4QuadWire>();
const A2_KAUTZ: usize = ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, kautz_pro);
const A2_RO: usize = ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, varpro_readout_pro);

const A3_ULTRA_RES: usize = ARENA_PRO_BYTES
    + offset_of!(Arena3UltraDeltaWire, sona_core_ultra)
    + offset_of!(SonaCoreUltraDeltaWire, reserve);
const A3_S4_1: usize = ARENA_PRO_BYTES
    + offset_of!(Arena3UltraDeltaWire, stage4_ultra)
    + core::mem::size_of::<Stage4QuadWire>();
const A3_KAUTZ: usize = ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, kautz_ultra);
const A3_RO: usize = ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, varpro_readout_ultra);
const A3_CTRL: usize = ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, ctrl_ultra);

/// Complete disjoint Role B `F32` carrier spans across `Arena 0..3` (`2,447` floats total).
pub const TARDOS_F32_SPANS: &[TardosSpan] = &[
    // Tier 0 (Arena 0): 16..383 (367 Role B F32)
    TardosSpan {
        tier: 0,
        start_idx: 16,
        end_idx: 80,
        byte_base: offset_of!(Arena0LiteWire, fir_lite),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 0,
        start_idx: 80,
        end_idx: 96,
        byte_base: A0_KAUTZ + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 0,
        start_idx: 96,
        end_idx: 112,
        byte_base: A0_KAUTZ + 128 + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 0,
        start_idx: 112,
        end_idx: 120,
        byte_base: A0_RO + offset_of!(VarProReadoutLiteWire, pre_cab_u_gain),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 0,
        start_idx: 120,
        end_idx: 360,
        byte_base: A0_WCTRL + 64,
        is_w_ctrl_role_b: true,
    },
    TardosSpan {
        tier: 0,
        start_idx: 360,
        end_idx: 383,
        byte_base: A0_TAIL + 4,
        is_w_ctrl_role_b: false,
    },
    // Tier 1 (Arena 1): 671..751 (80 Role B F32)
    TardosSpan {
        tier: 1,
        start_idx: 671,
        end_idx: 703,
        byte_base: ARENA_LITE_BYTES + offset_of!(Arena1FullDeltaWire, fir_full),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 1,
        start_idx: 703,
        end_idx: 719,
        byte_base: A1_KAUTZ + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 1,
        start_idx: 719,
        end_idx: 751,
        byte_base: A1_RO + 64,
        is_w_ctrl_role_b: false,
    },
    // Tier 2 (Arena 2): 1,175..2,103 (928 Role B F32)
    TardosSpan {
        tier: 2,
        start_idx: 1_175,
        end_idx: 1_847,
        byte_base: A2_PRO_RES,
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 1_847,
        end_idx: 1_879,
        byte_base: A2_S4_1,
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 1_879,
        end_idx: 1_975,
        byte_base: ARENA_FULL_BYTES + offset_of!(Arena2ProDeltaWire, fir_pro),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 1_975,
        end_idx: 1_991,
        byte_base: A2_KAUTZ + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 1_991,
        end_idx: 2_007,
        byte_base: A2_KAUTZ + 128 + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 2_007,
        end_idx: 2_023,
        byte_base: A2_KAUTZ + 256 + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 2_023,
        end_idx: 2_103,
        byte_base: A2_RO + 64,
        is_w_ctrl_role_b: false,
    },
    // Tier 3 (Arena 3): 3,067..4,139 (1,072 Role B F32)
    TardosSpan {
        tier: 3,
        start_idx: 3_067,
        end_idx: 3_675,
        byte_base: A3_ULTRA_RES,
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_675,
        end_idx: 3_707,
        byte_base: A3_S4_1,
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_707,
        end_idx: 3_771,
        byte_base: ARENA_PRO_BYTES + offset_of!(Arena3UltraDeltaWire, fir_ultra),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_771,
        end_idx: 3_787,
        byte_base: A3_KAUTZ + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_787,
        end_idx: 3_803,
        byte_base: A3_KAUTZ + 128 + offset_of!(KautzSoaBlockWire, c_re_l),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_803,
        end_idx: 3_947,
        byte_base: A3_RO + offset_of!(VarProReadoutUltraWire, prefix_override),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_947,
        end_idx: 3_995,
        byte_base: A3_CTRL + offset_of!(Arena3ControlBusWire, w_ctrl_ultra) + 64,
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 3_995,
        end_idx: 4_139,
        byte_base: A3_CTRL + offset_of!(Arena3ControlBusWire, reserve),
        is_w_ctrl_role_b: false,
    },
];

/// Complete disjoint Role B `U16` (`fp16`) carrier spans across `Arena 0..3` (`3,364` half-floats total).
pub const TARDOS_FP16_SPANS: &[TardosSpan] = &[
    // Tier 0 (Arena 0): 383..655 (272 Role B U16; excludes crossbar [80..96] & [192..208])
    TardosSpan {
        tier: 0,
        start_idx: 383,
        end_idx: 463,
        byte_base: A0_RO + offset_of!(VarProReadoutLiteWire, stage_latent_mods_fp16),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 0,
        start_idx: 463,
        end_idx: 655,
        byte_base: A0_RO + offset_of!(VarProReadoutLiteWire, w_ro_fp16),
        is_w_ctrl_role_b: false,
    },
    // Tier 1 (Arena 1): 751..1,159 (408 Role B U16; excludes b_gamma[328..336])
    TardosSpan {
        tier: 1,
        start_idx: 751,
        end_idx: 1_079,
        byte_base: A1_RO + offset_of!(VarProReadoutFullWire, w_ro_and_mods_fp16),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 1,
        start_idx: 1_079,
        end_idx: 1_159,
        byte_base: A1_RO + offset_of!(VarProReadoutFullWire, w_ro_and_mods_fp16) + 336 * 2,
        is_w_ctrl_role_b: false,
    },
    // Tier 2 (Arena 2): 2,103..3,051 (948 Role B U16; excludes b_gamma[424..436])
    TardosSpan {
        tier: 2,
        start_idx: 2_103,
        end_idx: 2_527,
        byte_base: A2_RO + offset_of!(VarProReadoutProWire, w_ro_and_mods_fp16),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 2,
        start_idx: 2_527,
        end_idx: 3_051,
        byte_base: A2_RO + offset_of!(VarProReadoutProWire, w_ro_and_mods_fp16) + 436 * 2,
        is_w_ctrl_role_b: false,
    },
    // Tier 3 (Arena 3): 4,139..5,875 (1,736 Role B U16; excludes b_gamma[584..608] & core_mods)
    TardosSpan {
        tier: 3,
        start_idx: 4_139,
        end_idx: 4_723,
        byte_base: A3_RO + offset_of!(VarProReadoutUltraWire, w_ro_and_mods_fp16),
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 4_723,
        end_idx: 5_619,
        byte_base: A3_RO + offset_of!(VarProReadoutUltraWire, w_ro_and_mods_fp16) + 608 * 2,
        is_w_ctrl_role_b: false,
    },
    TardosSpan {
        tier: 3,
        start_idx: 5_619,
        end_idx: 5_875,
        byte_base: A3_CTRL + offset_of!(Arena3ControlBusWire, stage_latent_ctrl_ultra_fp16),
        is_w_ctrl_role_b: false,
    },
];

const fn sum_span_counts(spans: &[TardosSpan]) -> usize {
    let mut sum = 0;
    let mut i = 0;
    while i < spans.len() {
        assert!(spans[i].end_idx > spans[i].start_idx);
        sum += spans[i].end_idx - spans[i].start_idx;
        i += 1;
    }
    sum
}

const fn check_spans_disjoint(spans: &[TardosSpan], elem_bytes: usize, arena_limit: usize) -> bool {
    let mut i = 0;
    while i < spans.len() {
        let end_byte = spans[i].byte_base + (spans[i].end_idx - spans[i].start_idx) * elem_bytes;
        if end_byte > arena_limit {
            return false;
        }
        let mut j = i + 1;
        while j < spans.len() {
            let a0 = spans[i].start_idx;
            let a1 = spans[i].end_idx;
            let b0 = spans[j].start_idx;
            let b1 = spans[j].end_idx;
            if a0 < b1 && b0 < a1 {
                return false;
            }
            let ba0 = spans[i].byte_base;
            let ba1 = ba0 + (a1 - a0) * elem_bytes;
            let bb0 = spans[j].byte_base;
            let bb1 = bb0 + (b1 - b0) * elem_bytes;
            if ba0 < bb1 && bb0 < ba1 {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

const F32_ROLE_B_COUNT: usize = sum_span_counts(TARDOS_F32_SPANS);
const FP16_ROLE_B_COUNT: usize = sum_span_counts(TARDOS_FP16_SPANS);

const _: () = assert!(F32_ROLE_B_COUNT == 2_447);
const _: () = assert!(FP16_ROLE_B_COUNT == 3_364);
const _: () = assert!(64 + F32_ROLE_B_COUNT + FP16_ROLE_B_COUNT + 1_805 == TARDOS_CASCADE_WEIGHTS);
const _: () = assert!(check_spans_disjoint(TARDOS_F32_SPANS, 4, ARENA_ULTRA_BYTES));
const _: () = assert!(check_spans_disjoint(
    TARDOS_FP16_SPANS,
    2,
    ARENA_ULTRA_BYTES
));
const _: () = assert!(A0_WCTRL == 3_456);
const _: () = assert!(A0_TAIL + 4 == 4_484);
const _: () = assert!(A0_TAIL + 4 + 23 * 4 == 4_576);
const _: () = assert!(A1_RO == 6_912);
const _: () = assert!(A2_PRO_RES == 9_856);
const _: () = assert!(A2_RO == 13_568);
const _: () = assert!(A3_ULTRA_RES == 20_096);
const _: () = assert!(A3_RO == 23_296);
const _: () = assert!(A3_CTRL == 26_880);
