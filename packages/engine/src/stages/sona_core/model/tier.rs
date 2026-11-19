//! Matryoshka `SONA` Tier Topology Constants, Group/Tile Indexing & Compile-Time Spec Lookup
//! (`stages/sona_core/model/tier.rs`).

use dnam_kernels::{FULL_SPEC, LITE_SPEC, PRO_SPEC, SonaTierSpec, ULTRA_SPEC};

use crate::types::Tier;

/// Number of sequential Matryoshka `SONA` core blocks across every tier (`4`).
pub const NUM_SONA_BLOCKS: usize = 4;
/// Maximum number of `4`-lane stereo `SONA` groups across all `4` blocks (`[4, 3, 2, 1] -> 10`).
pub const MAX_SONA_GROUPS: usize = ULTRA_SPEC.block_sum();
/// Tapered per-block `4`-lane group capacities at `Ultra` (`[4, 3, 2, 1]`).
pub const MAX_GROUPS_PER_BLOCK: [usize; NUM_SONA_BLOCKS] = ULTRA_SPEC.blocks;
/// Maximum total `SONA` core channels per stereo side (`40`).
pub const MAX_TOTAL_SONA_LANES: usize = ULTRA_SPEC.total_lanes();
/// Total number of inter-block `(src_block < dst_block)` `4x4` spatial mixing tile triples (`35`).
pub const MAX_SONA_TILES: usize = count_interblock_tiles(ULTRA_SPEC.blocks);
/// Maximum Stage III Early FIR taps per stereo channel (`128` at `Ultra`).
pub const SONA_FIR_MAX_TAPS: usize = ULTRA_SPEC.fir_taps;
/// Maximum `16`-tap Stage III Early FIR SIMD segments (`8` segments = `128` taps).
pub const MAX_FIR_SEGMENTS: usize = SONA_FIR_MAX_TAPS / 16;
/// Maximum Stage III shared conjugate Kautz modal pole pairs (`32` at `Ultra`).
pub const MAX_KAUTZ_POLES: usize = ULTRA_SPEC.kautz_pairs;
/// Maximum `4`-pair Stage III Kautz modal banks (`8` banks = `32` pole pairs).
pub const MAX_KAUTZ_BANKS: usize = MAX_KAUTZ_POLES / 4;
/// Power-of-two Stage III FIR ring capacity (`128` samples, mirrored to `256`).
pub const FIR_RING_CAP: usize = 128;
/// Power-of-two mask (`FIR_RING_CAP - 1`).
pub const FIR_RING_MASK: usize = FIR_RING_CAP - 1;
/// Maximum Stage IV `Op-T` tank quad groups (`4` quads = `16` delay lines).
pub const MAX_TANK_QUADS: usize = 4;
/// Maximum Stage IV `Op-T` delay lines (`16`).
pub const MAX_TANK_LINES: usize = MAX_TANK_QUADS * 4;
/// Number of `3 kHz` macro control knobs in `SonaControlBusParams`.
pub const NUM_SONA_MACRO_KNOBS: usize = 8;
/// Stage-latent vector rank (`4-D` for `Lite..Pro`, `8-D` for `Ultra`).
pub const MAX_STAGE_LATENTS: usize = 8;

/// Counts lower-triangular inter-block `4x4` tile edges across a `[B0, B1, B2, B3]` block taper.
pub const fn count_interblock_tiles(blocks: [usize; 4]) -> usize {
    blocks[0] * (blocks[1] + blocks[2] + blocks[3])
        + blocks[1] * (blocks[2] + blocks[3])
        + blocks[2] * blocks[3]
}

const _: () = assert!(LITE_SPEC.blocks.len() == NUM_SONA_BLOCKS);
const _: () = assert!(MAX_SONA_GROUPS == 10);
const _: () = assert!(MAX_TOTAL_SONA_LANES == 40);
const _: () = assert!(MAX_SONA_TILES == 35);
const _: () = assert!(SONA_FIR_MAX_TAPS == 128 && MAX_FIR_SEGMENTS == 8);
const _: () = assert!(MAX_KAUTZ_POLES == 32 && MAX_KAUTZ_BANKS == 8);

/// Canonical global `(block_idx, local_group_idx)` assignment preserving strict Matryoshka prefix
/// nesting (`0..4` = `Lite [1,1,1,1]`, `4..6` = `Full`, `6..8` = `Pro`, `8..10` = `Ultra`).
pub const SONA_GROUP_BLOCK_AND_LOCAL: [(usize, usize); MAX_SONA_GROUPS] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (0, 2),
    (2, 1),
    (0, 3),
    (1, 2),
];

/// Global group indices active inside each of the `4` Matryoshka blocks `[B0, B1, B2, B3]`.
pub const SONA_BLOCK_GROUP_SLOTS: [[usize; 4]; NUM_SONA_BLOCKS] = [
    [0, 4, 6, 8],
    [1, 5, 9, usize::MAX],
    [2, 7, usize::MAX, usize::MAX],
    [3, usize::MAX, usize::MAX, usize::MAX],
];

/// All `35` lower-triangular inter-block edges `(src_group_idx, dst_group_idx)` ordered strictly by
/// tier introduction (`0..3` `Lite`, `3..13` `Full`, `13..23` `Pro`, `23..35` `Ultra`).
pub const SONA_INTERBLOCK_TILES: [(usize, usize); MAX_SONA_TILES] = [
    // Lite (3 adjacent pairs across [1, 1, 1, 1])
    (0, 1),
    (1, 2),
    (2, 3),
    // Full (+10 inter-block pairs across [2, 2, 1, 1])
    (0, 5),
    (4, 1),
    (4, 5),
    (0, 2),
    (4, 2),
    (0, 3),
    (4, 3),
    (5, 2),
    (1, 3),
    (5, 3),
    // Pro (+10 inter-block pairs across [3, 2, 2, 1])
    (6, 1),
    (6, 5),
    (0, 7),
    (4, 7),
    (6, 2),
    (6, 7),
    (6, 3),
    (1, 7),
    (5, 7),
    (7, 3),
    // Ultra (+12 inter-block pairs across [4, 3, 2, 1])
    (0, 9),
    (4, 9),
    (6, 9),
    (8, 1),
    (8, 5),
    (8, 9),
    (8, 2),
    (8, 7),
    (8, 3),
    (9, 2),
    (9, 7),
    (9, 3),
];

const fn verify_interblock_tiles() -> bool {
    let mut i = 0;
    while i < MAX_SONA_TILES {
        let (sg, dg) = SONA_INTERBLOCK_TILES[i];
        if sg >= MAX_SONA_GROUPS || dg >= MAX_SONA_GROUPS {
            return false;
        }
        let sb = SONA_GROUP_BLOCK_AND_LOCAL[sg].0;
        let db = SONA_GROUP_BLOCK_AND_LOCAL[dg].0;
        if sb >= db {
            return false;
        }
        let mut j = i + 1;
        while j < MAX_SONA_TILES {
            let (sg2, dg2) = SONA_INTERBLOCK_TILES[j];
            if sg2 == sg && dg2 == dg {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

const _: () = assert!(verify_interblock_tiles());

/// Returns the compile-time locked [`SonaTierSpec`] for `tier` (`LiteSpec..UltraSpec`).
#[inline(always)]
#[must_use]
pub const fn tier_spec(tier: Tier) -> &'static SonaTierSpec {
    match tier {
        Tier::Lite => &LITE_SPEC,
        Tier::Full => &FULL_SPEC,
        Tier::Pro => &PRO_SPEC,
        Tier::Ultra => &ULTRA_SPEC,
    }
}

/// Returns `[B0, B1, B2, B3]` group counts for `tier` from the locked [`SonaTierSpec`].
#[inline(always)]
#[must_use]
pub const fn tier_block_groups_array(tier: Tier) -> [usize; NUM_SONA_BLOCKS] {
    tier_spec(tier).blocks
}

/// Returns active group count across all `4` blocks for `tier` (`4 / 6 / 8 / 10`).
#[inline(always)]
#[must_use]
pub const fn tier_sona_groups(tier: Tier) -> usize {
    tier_spec(tier).block_sum()
}

/// Returns active inter-block `4x4` tile pair count for `tier` (`3 / 13 / 23 / 35`).
#[inline(always)]
#[must_use]
pub const fn tier_tile_count(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 3,
        _ => count_interblock_tiles(tier_spec(tier).blocks),
    }
}

/// Returns active Stage III `16`-tap Early FIR segment count for `tier` (`2 / 3 / 6 / 8`).
#[inline(always)]
#[must_use]
pub const fn tier_fir_segments(tier: Tier) -> usize {
    tier_spec(tier).fir_taps / 16
}

/// Returns active Stage III `4`-pair Kautz modal bank count for `tier` (`2 / 3 / 6 / 8`).
#[inline(always)]
#[must_use]
pub const fn tier_kautz_banks(tier: Tier) -> usize {
    tier_spec(tier).kautz_pairs / 4
}

/// Returns active Stage IV `Op-T` tank quad count for `tier` (`1 / 2 / 3 / 4`).
#[inline(always)]
#[must_use]
pub const fn tier_tank_quads(tier: Tier) -> usize {
    tier_spec(tier).opt_lines / 4
}

/// Flattens `(group_idx, lane)` into `[0..MAX_TOTAL_SONA_LANES)`.
#[inline(always)]
#[must_use]
pub const fn flat_sona_lane_index(group_idx: usize, lane: usize) -> usize {
    (group_idx << 2) + (lane & 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sona_tier_counts_and_dag_tile_order() {
        assert_eq!(tier_sona_groups(Tier::Lite), 4);
        assert_eq!(tier_sona_groups(Tier::Full), 6);
        assert_eq!(tier_sona_groups(Tier::Pro), 8);
        assert_eq!(tier_sona_groups(Tier::Ultra), 10);
        assert_eq!(tier_tile_count(Tier::Lite), 3);
        assert_eq!(tier_tile_count(Tier::Full), 13);
        assert_eq!(tier_tile_count(Tier::Pro), 23);
        assert_eq!(tier_tile_count(Tier::Ultra), 35);
        assert_eq!(tier_fir_segments(Tier::Lite), 2);
        assert_eq!(tier_fir_segments(Tier::Full), 3);
        assert_eq!(tier_fir_segments(Tier::Pro), 6);
        assert_eq!(tier_fir_segments(Tier::Ultra), 8);
    }
}
