//! Pure `zerocopy` `.dnam-bin` (`"DNBN"`) container wire structs, `< 50 ns` validator
//! ([`DnamBinView`]), zero-mutation tier stripper ([`strip_dnam_bin_to_tier`]), and deterministic
//! builder (`#[forbid(unsafe_code)]`, `ref[container.dnam.vm-binary]`, `ref[container.dnam.tier-prefix]`).

pub mod arena_wire;
pub mod builder;
pub mod header;
pub mod view;

pub use arena_wire::{
    ARENA0_LITE_BYTES, ARENA1_FULL_CUMULATIVE_BYTES, ARENA2_PRO_CUMULATIVE_BYTES,
    ARENA3_ULTRA_CUMULATIVE_BYTES, Arena0ControlBusWire, Arena0LiteWire, Arena1FullDeltaWire,
    Arena2ProDeltaWire, Arena3ControlBusWire, Arena3UltraDeltaWire, CONVOLVER_EARLY_FIR_BASES,
    CONVOLVER_KAUTZ_BASES, CONVOLVER_KAUTZ_BLOCKS_PER_TIER, ConvolverFir16Wire, ConvolverFir32Wire,
    ConvolverFir48Wire, CoreQuadWire, CrossbarWire, ErSparseBank8Wire, KautzSoaBlockWire,
    LINEAR_READOUT_BASES, LinearReadoutFullWire, LinearReadoutLiteWire, LinearReadoutProWire,
    LinearReadoutUltraWire, OpNPreBaseWire, PIPE_CANONICAL_V4_SONA, PIPE_FLAG_CONVOLVER_EXT,
    PIPE_FLAG_SONA, PIPE_FLAG_TRANSPORT_EXT, PIPE_FLAG_UNIFIED_W_CTRL, PIPE_FLAG_V4_MATRYOSHKA,
    PIPE_MAGIC_HI, PIPE_MAGIC_MASK, PIPE_REQUIRED_SONA_MASK, SONA_META_CONVOLVER_PREFIX_OVERRIDES,
    SONA_META_GRA_CONTROL_BUS, SONA_META_OS_EXT_PRESENT, SONA_META_OS2X, SONA_META_RESERVED_MASK,
    SONA_META_SENSORY_QUAD, SONA_META_TRANSPORT_LFO, SONA_META_TRANSPORT_TAP_GAINS,
    SonaCellDescWire, SonaCrossbarBlockWire, SonaInterBlockTilePairWire, TRANSPORT_DELAY_BASES,
    TRANSPORT_FLAG_EXT_F32, TransportQuadWire,
};
pub use builder::{DnamBinBuildInput, computed_dnam_bin_byte_len, encode_dnam_bin_into};
#[cfg(feature = "alloc")]
pub use builder::{DnamBinBuildOwnedError, encode_dnam_bin};
pub use header::{
    ARENA_TIER_BYTES, CEXT_DISPATCH_BYTES, CP1_STREAM_FLAG_HAS_ULTRA, CP1_STREAM_MAGIC,
    CextDispatchWire, ConditionalClampRuleWire, ControlClampDescriptorWire, Cp1InlineDirWire,
    Cp1StreamDirEntryWire, Cp1StreamHeaderWire, DBRM_MAGIC, DNBN_FLAG_AUTO_UNITY,
    DNBN_FLAG_FREEZE_EVICTED_MASK, DNBN_FLAG_RESERVED_HI_MASK, DNBN_FLAG_SINGLE_INPUT,
    DNBN_FLAG_STEREO_OUT, DNBN_FLAG_WET_POLARITY_INV, DNBN_MAGIC, DNBN_VERSION_MAJOR,
    DNBN_VERSION_MINOR, DnamBinHeaderWire, DnamBinRuntimeMetaWire, MAX_NUM_CONTROLS,
    SONA_CELL_MAGIC, TSYN_ENTRY_BYTES, TSYN_FLAG_CLAMP_ONLY, TempoSyncEntryWire, align64,
};
pub use view::{DnamBinView, bind_dnam_bin_prevalidated, strip_dnam_bin_to_tier};

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[container.dnam.tensor-directory]
    #[test]
    fn test_bin_module_constants_align_to_cache_lines() {
        for &size in &ARENA_TIER_BYTES {
            assert_eq!(size % 64, 0);
        }
        assert_eq!(align64(128), 128);
        assert_eq!(align64(129), 192);
        assert_eq!(
            ARENA_TIER_BYTES,
            [
                ARENA0_LITE_BYTES,
                ARENA1_FULL_CUMULATIVE_BYTES,
                ARENA2_PRO_CUMULATIVE_BYTES,
                ARENA3_ULTRA_CUMULATIVE_BYTES,
            ]
        );
    }
}
