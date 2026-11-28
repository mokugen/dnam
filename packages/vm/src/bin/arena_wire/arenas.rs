//! Declarative `64 B`-aligned `v4.0` cumulative tier arena wire structs (`Arena0LiteWire..Arena3UltraDeltaWire`)
//! and `ControlBus` v4 `GRA` tables (`Arena0ControlBusWire`, `Arena3ControlBusWire`), with all slice bases
//! and `ARENA_TIER_BYTES` derived strictly at compile time (`ref[container.dnam.tier-prefix]`,
//! `ref[container.dnam.tensor-directory]`, `ref[repo.code.constant-eval-and-derivations]`).

use core::mem::{align_of, offset_of, size_of};
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::blocks::{
    ConvolverFir16Wire, ConvolverFir32Wire, ConvolverFir48Wire, CoreQuadWire, CrossbarWire,
    ErSparseBank8Wire, KautzSoaBlockWire, LinearReadoutFullWire, LinearReadoutLiteWire,
    LinearReadoutProWire, LinearReadoutUltraWire, SonaCrossbarBlockWire,
    SonaInterBlockTilePairWire, TransportQuadWire,
};
use crate::bin::header::Cp1InlineDirWire;

/// `2,048 B` (`32 CL`) self-contained Arena 0 `ControlBus` v4 `GRA` wire payload.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena0ControlBusWire {
    /// 48 kHz LFO training phase increments (`[-pi, pi]`).
    pub lfo_dphi_train: [F32; 2],
    /// LFO trapezoid skew parameters (`[-1.0, 1.0]`, `fp16`).
    pub lfo_skew_fp16: [U16; 2],
    /// `ControlBus`-driven tank dry mix (`[-2.0, 2.0]`, `fp16`).
    pub tank_dry_mix_fp16: U16,
    /// Reserved scalar (`0x0000`).
    pub reserved_ctrl_fp16: U16,
    /// Reserved scalar pad (`16 B`, aligns `u_int_base` to `64 B` after retired drift/OU scalars).
    pub reserved_gra_scalars: [u8; 16],
    /// Trust-region Mahalanobis radius (`[0.01, 16.0]`, fallback `2.5` / `1.0`).
    pub trust_radius_mh: F32,
    /// Inline `CP1` contributor directory (`28 B`).
    pub cp1_dir: Cp1InlineDirWire,
    /// Exact-integer `4x4` base interaction accumulator (`z_head[0..4]`, `64 B` @ offset `64`).
    pub u_int_base: [[F32; 4]; 4],
    /// Self-contained Lite..Pro `GRA` weights (`SwiGLU` + `Padé-[1/1]` + `1.5-Entmax`, `768 B` @ `128`).
    pub gra_lite: [u8; 768],
    /// Unified 16x16 control dispatch matrix (`REQ-8.5` Tardos Role B, `1,024 B` @ `896`).
    pub w_ctrl: [[F32; 16]; 16],
    /// `rho_head_and_ctrl_tail[0..32]` (`[1..24]` Tardos `REQ-8.5` intact, `128 B` @ `1,920`).
    pub rho_head_and_ctrl_tail: [F32; 32],
}

impl Default for Arena0ControlBusWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<Cp1InlineDirWire>() == 28);
const _: () = assert!(size_of::<Arena0ControlBusWire>() == 2_048);
const _: () = assert!(align_of::<Arena0ControlBusWire>() == 1);
const _: () = assert!(offset_of!(Arena0ControlBusWire, u_int_base) == 64);
const _: () = assert!(offset_of!(Arena0ControlBusWire, gra_lite) == 128);
const _: () = assert!(offset_of!(Arena0ControlBusWire, w_ctrl) == 896);
const _: () = assert!(offset_of!(Arena0ControlBusWire, rho_head_and_ctrl_tail) == 1_920);

/// `3,840 B` (`60 CL`) Arena 3 `ControlBus` v4 `GRA` Ultra delta (`<= 9,280 B` total control budget).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena3ControlBusWire {
    /// Ultra control dispatch delta (`REQ-8.5` Tardos Role B, `256 B`).
    pub w_ctrl_ultra: [[F32; 4]; 16],
    /// Ultra exact-integer `4x4` head interaction accumulator (`z_head[0..4]` `SO(4)`, `64 B`).
    pub u_int_ultra: [[F32; 4]; 4],
    /// Ultra `GRA` delta (`SwiGLU` H=16, `1.5-Entmax` M=16, `g_edge[0..6]`, `op_t_q[0..16]`, `2,432 B`).
    pub gra_ultra_delta: [u8; 2_432],
    /// `[0..160]` Ultra stage-latent mods, `[160..164]` retired zero, `[164..256]` Tardos `5,779..5,871`.
    pub stage_latent_ctrl_ultra_fp16: [U16; 256],
    /// Reserved Tardos Role B / zero reserve (`576 B`).
    pub reserve: [u8; 576],
}

impl Default for Arena3ControlBusWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<Arena3ControlBusWire>() == 3_840);
const _: () = assert!(align_of::<Arena3ControlBusWire>() == 1);
const _: () =
    assert!(size_of::<Arena0ControlBusWire>() + size_of::<Arena3ControlBusWire>() == 5_888);
const _: () =
    assert!(size_of::<Arena0ControlBusWire>() + size_of::<Arena3ControlBusWire>() <= 9_280);

/// Tier 0 (`Lite` `[1, 1, 1, 1]`, `5` tiles, `32` FIR, `8` ErSparse, `8` Kautz) `64 B`-multiple wire arena.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena0LiteWire {
    pub crossbar: CrossbarWire,
    pub core_quads: [CoreQuadWire; 4],
    pub tiles: [SonaInterBlockTilePairWire; 5],
    pub conv_fir_lite: ConvolverFir32Wire,
    pub conv_er_lite: ErSparseBank8Wire,
    pub conv_fir_morph_lite_fp16: [U16; 24],
    pub conv_kautz_lite: [KautzSoaBlockWire; 2],
    pub transport_lite: TransportQuadWire,
    pub linear_readout_lite: LinearReadoutLiteWire,
    pub control_bus_lite: Arena0ControlBusWire,
}

/// Tier 1 (`Full` `[2, 2, 1, 1]`, `+8 = 13` tiles, `+16 = 48` FIR, `+8 = 16` ErSparse, `+4` Kautz) delta.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena1FullDeltaWire {
    pub core_quads: [CoreQuadWire; 2],
    pub tiles: [SonaInterBlockTilePairWire; 8],
    pub crossbar_full: SonaCrossbarBlockWire,
    pub reserve_core: [u8; 32],
    pub conv_fir_full: ConvolverFir16Wire,
    pub conv_er_full: ErSparseBank8Wire,
    pub conv_fir_morph_full_fp16: [U16; 24],
    pub conv_kautz_full: KautzSoaBlockWire,
    pub transport_full: TransportQuadWire,
    pub linear_readout_full: LinearReadoutFullWire,
}

/// Tier 2 (`Pro` `[2, 2, 2, 2]`, `+11 = 24` tiles, `+48 = 96` FIR, `+12` Kautz) delta.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena2ProDeltaWire {
    pub core_quads: [CoreQuadWire; 2],
    pub tiles: [SonaInterBlockTilePairWire; 11],
    pub crossbar_pro: SonaCrossbarBlockWire,
    pub reserve_core: [u8; 2_720],
    pub conv_fir_pro: ConvolverFir48Wire,
    pub conv_kautz_pro: [KautzSoaBlockWire; 3],
    pub transport_pro: [TransportQuadWire; 2],
    pub linear_readout_pro: LinearReadoutProWire,
}

impl Default for Arena2ProDeltaWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

/// Tier 3 (`Ultra` `[3, 3, 2, 2]`, `+13 = 37` tiles, `+32 = 128` FIR, `+8` Kautz) delta.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena3UltraDeltaWire {
    pub core_quads: [CoreQuadWire; 2],
    pub tiles: [SonaInterBlockTilePairWire; 13],
    pub crossbar_ultra: SonaCrossbarBlockWire,
    pub reserve_core: [u8; 2_464],
    pub conv_fir_ultra: ConvolverFir32Wire,
    pub conv_kautz_ultra: [KautzSoaBlockWire; 2],
    pub transport_ultra: [TransportQuadWire; 2],
    pub linear_readout_ultra: LinearReadoutUltraWire,
    pub control_bus_ultra_delta: Arena3ControlBusWire,
}

impl Default for Arena3UltraDeltaWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

/// Exact `64 B`-aligned byte footprint of [`Arena0LiteWire`].
pub const ARENA0_LITE_BYTES: usize = size_of::<Arena0LiteWire>();
/// Cumulative byte size of Tier 0 + Tier 1 (`Full`).
pub const ARENA1_FULL_CUMULATIVE_BYTES: usize =
    ARENA0_LITE_BYTES + size_of::<Arena1FullDeltaWire>();
/// Cumulative byte size of Tier 0..=2 (`Pro`).
pub const ARENA2_PRO_CUMULATIVE_BYTES: usize =
    ARENA1_FULL_CUMULATIVE_BYTES + size_of::<Arena2ProDeltaWire>();
/// Cumulative byte size of Tier 0..=3 (`Ultra`).
pub const ARENA3_ULTRA_CUMULATIVE_BYTES: usize =
    ARENA2_PRO_CUMULATIVE_BYTES + size_of::<Arena3UltraDeltaWire>();

/// Canonical `64 B`-aligned early-FIR slice bases relative to `arena0_offset` per tier.
pub const CONVOLVER_EARLY_FIR_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, conv_fir_lite),
    ARENA0_LITE_BYTES + offset_of!(Arena1FullDeltaWire, conv_fir_full),
    ARENA1_FULL_CUMULATIVE_BYTES + offset_of!(Arena2ProDeltaWire, conv_fir_pro),
    ARENA2_PRO_CUMULATIVE_BYTES + offset_of!(Arena3UltraDeltaWire, conv_fir_ultra),
];

/// Canonical `64 B`-aligned Kautz modal block bases relative to `arena0_offset` per tier.
pub const CONVOLVER_KAUTZ_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, conv_kautz_lite),
    ARENA0_LITE_BYTES + offset_of!(Arena1FullDeltaWire, conv_kautz_full),
    ARENA1_FULL_CUMULATIVE_BYTES + offset_of!(Arena2ProDeltaWire, conv_kautz_pro),
    ARENA2_PRO_CUMULATIVE_BYTES + offset_of!(Arena3UltraDeltaWire, conv_kautz_ultra),
];

/// Number of `KautzSoaBlockWire` (4-pair) blocks per tier `[2, 1, 3, 2]` (`8 / 12 / 24 / 32` pairs).
pub const CONVOLVER_KAUTZ_BLOCKS_PER_TIER: [usize; 4] = [2, 1, 3, 2];

/// Canonical `64 B`-aligned `TransportQuadWire` bases relative to `arena0_offset` per tier.
pub const TRANSPORT_DELAY_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, transport_lite),
    ARENA0_LITE_BYTES + offset_of!(Arena1FullDeltaWire, transport_full),
    ARENA1_FULL_CUMULATIVE_BYTES + offset_of!(Arena2ProDeltaWire, transport_pro),
    ARENA2_PRO_CUMULATIVE_BYTES + offset_of!(Arena3UltraDeltaWire, transport_ultra),
];

/// Canonical `64 B`-aligned `LinearReadout*Wire` bases relative to `arena0_offset` per tier.
pub const LINEAR_READOUT_BASES: [usize; 4] = [
    offset_of!(Arena0LiteWire, linear_readout_lite),
    ARENA0_LITE_BYTES + offset_of!(Arena1FullDeltaWire, linear_readout_full),
    ARENA1_FULL_CUMULATIVE_BYTES + offset_of!(Arena2ProDeltaWire, linear_readout_pro),
    ARENA2_PRO_CUMULATIVE_BYTES + offset_of!(Arena3UltraDeltaWire, linear_readout_ultra),
];

const _: () = assert!(align_of::<Arena0LiteWire>() == 1);
const _: () = assert!(align_of::<Arena1FullDeltaWire>() == 1);
const _: () = assert!(align_of::<Arena2ProDeltaWire>() == 1);
const _: () = assert!(align_of::<Arena3UltraDeltaWire>() == 1);
const _: () = assert!(ARENA0_LITE_BYTES.is_multiple_of(64));
const _: () = assert!(ARENA1_FULL_CUMULATIVE_BYTES.is_multiple_of(64));
const _: () = assert!(ARENA2_PRO_CUMULATIVE_BYTES.is_multiple_of(64));
const _: () = assert!(ARENA3_ULTRA_CUMULATIVE_BYTES.is_multiple_of(64));
const _: () = assert!(ARENA3_ULTRA_CUMULATIVE_BYTES <= 35_840);
const _: () = assert!(CONVOLVER_EARLY_FIR_BASES[0].is_multiple_of(64));
const _: () = assert!(CONVOLVER_EARLY_FIR_BASES[1].is_multiple_of(64));
const _: () = assert!(CONVOLVER_EARLY_FIR_BASES[2].is_multiple_of(64));
const _: () = assert!(CONVOLVER_EARLY_FIR_BASES[3].is_multiple_of(64));
const _: () = assert!(CONVOLVER_KAUTZ_BASES[0].is_multiple_of(64));
const _: () = assert!(CONVOLVER_KAUTZ_BASES[1].is_multiple_of(64));
const _: () = assert!(CONVOLVER_KAUTZ_BASES[2].is_multiple_of(64));
const _: () = assert!(CONVOLVER_KAUTZ_BASES[3].is_multiple_of(64));
const _: () = assert!(TRANSPORT_DELAY_BASES[0].is_multiple_of(64));
const _: () = assert!(TRANSPORT_DELAY_BASES[1].is_multiple_of(64));
const _: () = assert!(TRANSPORT_DELAY_BASES[2].is_multiple_of(64));
const _: () = assert!(TRANSPORT_DELAY_BASES[3].is_multiple_of(64));
const _: () = assert!(LINEAR_READOUT_BASES[0].is_multiple_of(64));
const _: () = assert!(LINEAR_READOUT_BASES[1].is_multiple_of(64));
const _: () = assert!(LINEAR_READOUT_BASES[2].is_multiple_of(64));
const _: () = assert!(LINEAR_READOUT_BASES[3].is_multiple_of(64));
const _: () = assert!(offset_of!(Arena0LiteWire, tiles) == 1_216);
const _: () = assert!(offset_of!(Arena0LiteWire, control_bus_lite) == 3_328);
const _: () =
    assert!(offset_of!(Arena0LiteWire, conv_er_lite) == CONVOLVER_EARLY_FIR_BASES[0] + 256);
const _: () = assert!(
    offset_of!(Arena1FullDeltaWire, conv_er_full)
        == offset_of!(Arena1FullDeltaWire, conv_fir_full) + 128
);
const _: () = assert!(
    offset_of!(Arena2ProDeltaWire, transport_pro) + size_of::<TransportQuadWire>()
        < size_of::<Arena2ProDeltaWire>()
);
const _: () = assert!(
    offset_of!(Arena3UltraDeltaWire, transport_ultra) + size_of::<TransportQuadWire>()
        < size_of::<Arena3UltraDeltaWire>()
);
