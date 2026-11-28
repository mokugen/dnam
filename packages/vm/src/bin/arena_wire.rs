//! Declarative `#[repr(C)]` `zerocopy` v4.0 wire constants, tier-arena layouts, and block types
//! (`ref[container.dnam.vm-binary]`, `ref[container.dnam.tier-prefix]`).

pub mod arenas;
pub mod blocks;

pub use arenas::{
    ARENA0_LITE_BYTES, ARENA1_FULL_CUMULATIVE_BYTES, ARENA2_PRO_CUMULATIVE_BYTES,
    ARENA3_ULTRA_CUMULATIVE_BYTES, Arena0ControlBusWire, Arena0LiteWire, Arena1FullDeltaWire,
    Arena2ProDeltaWire, Arena3ControlBusWire, Arena3UltraDeltaWire, CONVOLVER_EARLY_FIR_BASES,
    CONVOLVER_KAUTZ_BASES, CONVOLVER_KAUTZ_BLOCKS_PER_TIER, LINEAR_READOUT_BASES,
    TRANSPORT_DELAY_BASES,
};
pub use blocks::{
    ConvolverFir16Wire, ConvolverFir32Wire, ConvolverFir48Wire, CoreQuadWire, CrossbarWire,
    ErSparseBank8Wire, KautzSoaBlockWire, LinearReadoutFullWire, LinearReadoutLiteWire,
    LinearReadoutProWire, LinearReadoutUltraWire, OpNPreBaseWire, SonaCellDescWire,
    SonaCrossbarBlockWire, SonaInterBlockTilePairWire, TransportQuadWire,
};

/// Upper magic nibble `0x444E_0000` (`"DN"`) in `CrossbarWire::pipeline_flags`.
pub const PIPE_MAGIC_HI: u32 = 0x444E_0000;
/// Bitmask for `PIPE_MAGIC_HI`.
pub const PIPE_MAGIC_MASK: u32 = 0xFFFF_0000;
/// Informational flag: extended `ModalConvolver` payload present.
pub const PIPE_FLAG_CONVOLVER_EXT: u32 = 1 << 3;
/// Informational flag: extended `TransportQuad` payload present.
pub const PIPE_FLAG_TRANSPORT_EXT: u32 = 1 << 4;
/// Required flag: unified `W_ctrl` matrix layout active.
pub const PIPE_FLAG_UNIFIED_W_CTRL: u32 = 1 << 5;
/// Required flag: `SONA v4.0` causal Matryoshka tier-prefix layout active.
pub const PIPE_FLAG_V4_MATRYOSHKA: u32 = 1 << 6;
/// Required flag: `SONA` DSP pipeline active.
pub const PIPE_FLAG_SONA: u32 = 1 << 7;
/// Required `SONA v4.0` pipeline bitmask (`0x00E0`).
pub const PIPE_REQUIRED_SONA_MASK: u32 =
    PIPE_FLAG_UNIFIED_W_CTRL | PIPE_FLAG_V4_MATRYOSHKA | PIPE_FLAG_SONA;
/// Canonical `SONA v4.0` `pipeline_flags` (`0x444E_00F8`).
pub const PIPE_CANONICAL_V4_SONA: u32 =
    PIPE_MAGIC_HI | PIPE_REQUIRED_SONA_MASK | PIPE_FLAG_CONVOLVER_EXT | PIPE_FLAG_TRANSPORT_EXT;

/// `sona_meta_flags` bit 4: base `2x` `HB-FIR` `SONA` oversampling active (`canonical 1 in v4.0`).
pub const SONA_META_OS2X: u32 = 1 << 4;
/// `sona_meta_flags` bit 7: higher tiers carry nonzero `ModalConvolver` prefix overrides.
pub const SONA_META_CONVOLVER_PREFIX_OVERRIDES: u32 = 1 << 7;
/// `sona_meta_flags` bit 8: `SensoryQuad` Log-Knee VCA + `Op-P` `T1+` + `Op-N[Pre]` ext active.
pub const SONA_META_OS_EXT_PRESENT: u32 = 1 << 8;
/// `sona_meta_flags` bit 16: `Op-T` v4 transport tap gains active.
pub const SONA_META_TRANSPORT_TAP_GAINS: u32 = 1 << 16;
/// `sona_meta_flags` bit 17: `SensoryQuad` v4 active.
pub const SONA_META_SENSORY_QUAD: u32 = 1 << 17;
/// `sona_meta_flags` bit 18: `ControlBus` v4 `GRA` tables active.
pub const SONA_META_GRA_CONTROL_BUS: u32 = 1 << 18;
/// `sona_meta_flags` bit 20: `Op-T` LFO & modulation active.
pub const SONA_META_TRANSPORT_LFO: u32 = 1 << 20;
/// Reserved `sona_meta_flags` bits (includes retired bit 6 `TIER_OS_EXPLICIT` & bit 9 `CORE_MODS_PRESENT`).
pub const SONA_META_RESERVED_MASK: u32 =
    (1 << 5) | (1 << 6) | (1 << 9) | 0xFC00 | (1 << 19) | 0xFFE0_0000;

/// `TransportQuadWire::flags` bit 1: external `f32` delay lines enabled (`Pro` / `Ultra`).
pub const TRANSPORT_FLAG_EXT_F32: u8 = 1 << 1;

const _: () = assert!(PIPE_CANONICAL_V4_SONA == 0x444E_00F8);
const _: () = assert!((SONA_META_RESERVED_MASK & SONA_META_GRA_CONTROL_BUS) == 0);
const _: () = assert!((SONA_META_RESERVED_MASK & (1 << 6)) != 0);
const _: () = assert!((SONA_META_RESERVED_MASK & (1 << 9)) != 0);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn arena_wire_v4_sizes_and_64b_alignment_hold() {
        assert_eq!(size_of::<OpNPreBaseWire>(), 12);
        assert_eq!(size_of::<CrossbarWire>(), 256);
        assert_eq!(size_of::<CoreQuadWire>(), 240);
        assert_eq!(size_of::<ErSparseBank8Wire>(), 80);
        assert_eq!(size_of::<KautzSoaBlockWire>(), 128);
        assert_eq!(size_of::<TransportQuadWire>(), 192);
        assert_eq!(size_of::<Arena0ControlBusWire>(), 2_048);
        assert_eq!(size_of::<Arena3ControlBusWire>(), 3_840);
        assert_eq!(ARENA0_LITE_BYTES % 64, 0);
        assert_eq!(ARENA1_FULL_CUMULATIVE_BYTES % 64, 0);
        assert_eq!(ARENA2_PRO_CUMULATIVE_BYTES % 64, 0);
        assert_eq!(ARENA3_ULTRA_CUMULATIVE_BYTES % 64, 0);
    }
}
