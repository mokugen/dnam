//! Compile-time fixed-shape `SONA` tier capacity profiles (`v3.0` compatibility + `v4.0`
//! all-tier `2x` Matryoshka specs & `37`-tile causal topology, `MAX_OUTGOING_PER_QUAD_V4 = 7`,
//! `ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.op-n.core.interblock-tiles]`,
//! `ref[arch.simd.lane-packing]`).
//!
//! Strictly `#![no_std]`, zero `f64`/`u64`/`i64`, `spirv-unknown-naga-wgsl`-clean.

/// Maximum sequential Matryoshka blocks across every `SONA` tier (`4`).
pub const NUM_SONA_BLOCKS_V4: usize = 4;

/// Maximum `4`-lane stereo `CoreQuad` groups across `SONA v4.0` `Ultra` (`3 + 3 + 2 + 2 = 10`).
pub const MAX_CORE_QUADS_V4: usize = 10;

/// Maximum total `SONA v4.0` core lanes (`10 * 4 = 40`).
pub const MAX_TOTAL_SONA_LANES_V4: usize = MAX_CORE_QUADS_V4 * 4;

/// Maximum outgoing inter-block tiles per `CoreQuad` across `SONA v4.0` `Ultra` (`7`).
pub const MAX_OUTGOING_PER_QUAD_V4: usize = 7;

/// Maximum lower-triangular `4x4` inter-block tile pairs across `SONA v4.0` (`5 + 8 + 11 + 13 = 37`).
pub const MAX_SONA_TILES_V4: usize = 37;

/// Cumulative `SONA v4.0` inter-block tile counts per tier `[Lite, Full, Pro, Ultra]`.
pub const SONA_V4_TILE_COUNTS: [usize; 4] = [5, 13, 24, 37];

/// Compile-time fixed-shape tier capacity profile locked at `Gate 0b` (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SonaTierSpec {
    /// Number of 4-lane `Vec4` quads in each of the 4 Matryoshka `SONA` core blocks.
    pub blocks: [usize; 4],
    /// Number of 4-mode `TPT-SVF` `Vec4` quads in the `SONA` core.
    pub svf_quads: usize,
    /// Number of cabinet early-reflection FIR taps per stereo channel.
    pub fir_taps: usize,
    /// Number of cabinet shared conjugate Kautz modal pole pairs.
    pub kautz_pairs: usize,
    /// Number of `Op-T` transport delay lines (`M` lines = `4 * quads`).
    pub opt_lines: usize,
    /// Core oversampling factor (`1` or `2` for real-time execution).
    pub oversample_factor: usize,
}

impl SonaTierSpec {
    /// Constructs a compile-time `SONA` tier profile with `svf_quads` derived from `blocks`
    /// and `opt_lines = 4 * tank_quads`.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        blocks: [usize; 4],
        fir_taps: usize,
        kautz_pairs: usize,
        tank_quads: usize,
        oversample_factor: usize,
    ) -> Self {
        let svf_quads = blocks[0] + blocks[1] + blocks[2] + blocks[3];
        Self {
            blocks,
            svf_quads,
            fir_taps,
            kautz_pairs,
            opt_lines: 4 * tank_quads,
            oversample_factor,
        }
    }

    /// Returns the total number of 4-lane `Vec4` groups across all 4 Matryoshka `SONA` blocks.
    #[inline(always)]
    #[must_use]
    pub const fn total_groups(&self) -> usize {
        self.blocks[0] + self.blocks[1] + self.blocks[2] + self.blocks[3]
    }

    /// Returns the total number of 4-lane `Vec4` quads across all 4 Matryoshka `SONA` core blocks.
    #[inline(always)]
    #[must_use]
    pub const fn block_sum(&self) -> usize {
        self.total_groups()
    }

    /// Returns the total number of scalar `SONA` core lanes (`4 * self.total_groups()`).
    #[inline(always)]
    #[must_use]
    pub const fn lanes(&self) -> usize {
        4 * self.total_groups()
    }

    /// Returns the total number of scalar `SONA` core lanes (`4 * self.block_sum()`).
    #[inline(always)]
    #[must_use]
    pub const fn total_lanes(&self) -> usize {
        self.lanes()
    }

    /// Returns the number of tank `Op-T` transport delay quads (`opt_lines >> 2`).
    #[inline(always)]
    #[must_use]
    pub const fn tank_quads(&self) -> usize {
        self.opt_lines >> 2
    }
}

/// `v3.0` `Lite` compile-time tier capacity profile (`[1, 1, 1, 1]`, `16` lanes @ `1x`).
pub const LITE_SPEC: SonaTierSpec = SonaTierSpec::new([1, 1, 1, 1], 32, 8, 1, 1);

/// `v3.0` `Full` compile-time tier capacity profile (`[2, 2, 1, 1]`, `24` lanes @ `2x`).
pub const FULL_SPEC: SonaTierSpec = SonaTierSpec::new([2, 2, 1, 1], 48, 12, 2, 2);

/// `v3.0` `Pro` compile-time tier capacity profile (`[3, 2, 2, 1]`, `32` lanes @ `2x`).
pub const PRO_SPEC: SonaTierSpec = SonaTierSpec::new([3, 2, 2, 1], 96, 24, 3, 2);

/// `v3.0` `Ultra` compile-time tier capacity profile (`[4, 3, 2, 1]`, `40` lanes @ `2x`).
pub const ULTRA_SPEC: SonaTierSpec = SonaTierSpec::new([4, 3, 2, 1], 128, 32, 4, 2);

/// `SONA v4.0` `Lite` compile-time tier profile (`[1, 1, 1, 1]`, `16` lanes @ `2x HB-FIR`, `5` tiles).
pub const LITE_V4_SPEC: SonaTierSpec = SonaTierSpec::new([1, 1, 1, 1], 32, 8, 1, 2);

/// `SONA v4.0` `Full` compile-time tier profile (`[2, 2, 1, 1]`, `24` lanes @ `2x HB-FIR`, `13` tiles).
pub const FULL_V4_SPEC: SonaTierSpec = SonaTierSpec::new([2, 2, 1, 1], 48, 12, 2, 2);

/// `SONA v4.0` `Pro` compile-time tier profile (`[2, 2, 2, 2]`, `32` lanes @ `2x HB-FIR`, `24` tiles).
pub const PRO_V4_SPEC: SonaTierSpec = SonaTierSpec::new([2, 2, 2, 2], 96, 24, 3, 2);

/// `SONA v4.0` `Ultra` compile-time tier profile (`[3, 3, 2, 2]`, `40` lanes @ `2x HB-FIR`, `37` tiles).
pub const ULTRA_V4_SPEC: SonaTierSpec = SonaTierSpec::new([3, 3, 2, 2], 128, 32, 4, 2);

/// Canonical `SONA v4.0` global `(block_idx, local_quad_idx)` assignment for `CoreQuad` slots `0..10`
/// (`ref[arch.simd.lane-packing]`).
pub const SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL: [(usize, usize); MAX_CORE_QUADS_V4] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
];

/// `SONA v4.0` `CoreQuad` slots per Matryoshka block `[Block 0..Block 3]` in prefix-nested order
/// (`ref[dsp.op-n.core.interblock-tiles]`).
pub const SONA_V4_BLOCK_QUAD_SLOTS: [[Option<usize>; 4]; NUM_SONA_BLOCKS_V4] = [
    [Some(0), Some(4), Some(8), None],
    [Some(1), Some(5), Some(9), None],
    [Some(2), Some(6), None, None],
    [Some(3), Some(7), None, None],
];

/// Causal `SONA v4.0` block-major `CoreQuad` evaluation sequence across all `10` `Ultra` quads.
pub const SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER: [usize; MAX_CORE_QUADS_V4] =
    [0, 4, 8, 1, 5, 9, 2, 6, 3, 7];

/// All `37` `SONA v4.0` lower-triangular inter-block edges `(src_quad, dst_quad)` ordered strictly
/// by tier arrival (`0..5` `Lite`, `5..13` `Full`, `13..24` `Pro`, `24..37` `Ultra`,
/// `MAX_OUTGOING_PER_QUAD_V4 == 7`, `ref[dsp.op-n.core.interblock-tiles]`).
pub const SONA_V4_INTERBLOCK_TILES: [(usize, usize); MAX_SONA_TILES_V4] = [
    // 0..5 (Lite, +5)
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 2),
    (1, 3),
    // 5..13 (Full, +8)
    (0, 3),
    (0, 5),
    (4, 1),
    (4, 5),
    (4, 2),
    (4, 3),
    (5, 2),
    (5, 3),
    // 13..24 (Pro, +11)
    (0, 6),
    (4, 6),
    (0, 7),
    (4, 7),
    (1, 6),
    (5, 6),
    (1, 7),
    (5, 7),
    (2, 7),
    (6, 3),
    (6, 7),
    // 24..37 (Ultra, +13)
    (0, 9),
    (4, 9),
    (8, 1),
    (8, 5),
    (8, 9),
    (8, 2),
    (8, 6),
    (8, 3),
    (8, 7),
    (9, 2),
    (9, 6),
    (9, 3),
    (9, 7),
];

/// Verifies every capacity field on a [`SonaTierSpec`] against expected tier dimensions.
#[must_use]
pub const fn verify_tier_spec(
    spec: &SonaTierSpec,
    expected_groups: usize,
    expected_tank_quads: usize,
    expected_fir: usize,
    expected_kautz: usize,
    expected_os: usize,
) -> bool {
    spec.total_groups() == expected_groups
        && spec.block_sum() == expected_groups
        && spec.svf_quads == expected_groups
        && spec.svf_quads * 4 == spec.lanes()
        && spec.lanes() == spec.total_lanes()
        && (spec.opt_lines & 3) == 0
        && spec.tank_quads() == expected_tank_quads
        && spec.opt_lines == expected_tank_quads * 4
        && spec.fir_taps == expected_fir
        && spec.kautz_pairs == expected_kautz
        && spec.oversample_factor == expected_os
}

/// Checks strict Matryoshka prefix monotonicity `lo ⊂ hi` (`total_lanes(lo) < total_lanes(hi)`).
#[must_use]
pub const fn is_matryoshka_monotonic(lo: &SonaTierSpec, hi: &SonaTierSpec) -> bool {
    lo.blocks[0] <= hi.blocks[0]
        && lo.blocks[1] <= hi.blocks[1]
        && lo.blocks[2] <= hi.blocks[2]
        && lo.blocks[3] <= hi.blocks[3]
        && lo.svf_quads <= hi.svf_quads
        && lo.fir_taps <= hi.fir_taps
        && lo.kautz_pairs <= hi.kautz_pairs
        && lo.opt_lines <= hi.opt_lines
        && lo.oversample_factor <= hi.oversample_factor
        && lo.total_lanes() < hi.total_lanes()
}

const fn verify_v4_topology_with(
    tiles: &[(usize, usize); MAX_SONA_TILES_V4],
    quad_locs: &[(usize, usize); MAX_CORE_QUADS_V4],
    block_slots: &[[Option<usize>; 4]; NUM_SONA_BLOCKS_V4],
    traversal_order: &[usize; MAX_CORE_QUADS_V4],
) -> bool {
    let mut out_counts = [0_usize; MAX_CORE_QUADS_V4];
    let mut i = 0;
    while i < MAX_SONA_TILES_V4 {
        let (sq, dq) = tiles[i];
        let max_quad_for_tier = if i < SONA_V4_TILE_COUNTS[0] {
            4
        } else if i < SONA_V4_TILE_COUNTS[1] {
            6
        } else if i < SONA_V4_TILE_COUNTS[2] {
            8
        } else {
            10
        };
        if sq >= max_quad_for_tier || dq >= max_quad_for_tier {
            return false;
        }
        if quad_locs[sq].0 >= quad_locs[dq].0 {
            return false;
        }
        out_counts[sq] += 1;
        let mut j = i + 1;
        while j < MAX_SONA_TILES_V4 {
            if tiles[j].0 == sq && tiles[j].1 == dq {
                return false;
            }
            j += 1;
        }
        i += 1;
    }

    let mut max_out = 0;
    let mut q = 0;
    while q < MAX_CORE_QUADS_V4 {
        if out_counts[q] > max_out {
            max_out = out_counts[q];
        }
        let (blk, loc) = quad_locs[q];
        if blk >= NUM_SONA_BLOCKS_V4 || loc >= 4 {
            return false;
        }
        match block_slots[blk][loc] {
            Some(slot_q) if slot_q == q => {}
            _ => return false,
        }
        q += 1;
    }
    if max_out != MAX_OUTGOING_PER_QUAD_V4 {
        return false;
    }

    let mut idx = 0;
    let mut b = 0;
    while b < NUM_SONA_BLOCKS_V4 {
        let mut s = 0;
        while s < 4 {
            if let Some(q_id) = block_slots[b][s] {
                if traversal_order[idx] != q_id {
                    return false;
                }
                idx += 1;
            }
            s += 1;
        }
        b += 1;
    }
    idx == MAX_CORE_QUADS_V4
}

const fn verify_v4_interblock_tiles_and_slots() -> bool {
    verify_v4_topology_with(
        &SONA_V4_INTERBLOCK_TILES,
        &SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL,
        &SONA_V4_BLOCK_QUAD_SLOTS,
        &SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER,
    )
}

const _: () = assert!(LITE_SPEC.svf_quads * 4 == LITE_SPEC.lanes());
const _: () = assert!(FULL_SPEC.svf_quads * 4 == FULL_SPEC.lanes());
const _: () = assert!(PRO_SPEC.svf_quads * 4 == PRO_SPEC.lanes());
const _: () = assert!(ULTRA_SPEC.svf_quads * 4 == ULTRA_SPEC.lanes());
const _: () = assert!(verify_tier_spec(&LITE_SPEC, 4, 1, 32, 8, 1));
const _: () = assert!(verify_tier_spec(&FULL_SPEC, 6, 2, 48, 12, 2));
const _: () = assert!(verify_tier_spec(&PRO_SPEC, 8, 3, 96, 24, 2));
const _: () = assert!(verify_tier_spec(&ULTRA_SPEC, 10, 4, 128, 32, 2));
const _: () = assert!(is_matryoshka_monotonic(&LITE_SPEC, &FULL_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&FULL_SPEC, &PRO_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&PRO_SPEC, &ULTRA_SPEC));

const _: () = assert!(verify_tier_spec(&LITE_V4_SPEC, 4, 1, 32, 8, 2));
const _: () = assert!(verify_tier_spec(&FULL_V4_SPEC, 6, 2, 48, 12, 2));
const _: () = assert!(verify_tier_spec(&PRO_V4_SPEC, 8, 3, 96, 24, 2));
const _: () = assert!(verify_tier_spec(&ULTRA_V4_SPEC, 10, 4, 128, 32, 2));
const _: () = assert!(is_matryoshka_monotonic(&LITE_V4_SPEC, &FULL_V4_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&FULL_V4_SPEC, &PRO_V4_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&PRO_V4_SPEC, &ULTRA_V4_SPEC));
const _: () = assert!(MAX_OUTGOING_PER_QUAD_V4 == 7);
const _: () = assert!(MAX_SONA_TILES_V4 == 37);
const _: () = assert!(MAX_TOTAL_SONA_LANES_V4 == 40);
const _: () = assert!(verify_v4_interblock_tiles_and_slots());

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[repo.code.constant-eval-and-derivations]
    /// verify[dsp.op-n.core.interblock-tiles]
    #[test]
    fn test_v3_and_v4_tier_specs_matryoshka_and_37_tile_topology() {
        assert!(verify_tier_spec(&LITE_SPEC, 4, 1, 32, 8, 1));
        assert!(verify_tier_spec(&FULL_SPEC, 6, 2, 48, 12, 2));
        assert!(verify_tier_spec(&PRO_SPEC, 8, 3, 96, 24, 2));
        assert!(verify_tier_spec(&ULTRA_SPEC, 10, 4, 128, 32, 2));
        assert!(!is_matryoshka_monotonic(&FULL_SPEC, &LITE_SPEC));

        assert!(verify_tier_spec(&LITE_V4_SPEC, 4, 1, 32, 8, 2));
        assert!(verify_tier_spec(&FULL_V4_SPEC, 6, 2, 48, 12, 2));
        assert!(verify_tier_spec(&PRO_V4_SPEC, 8, 3, 96, 24, 2));
        assert!(verify_tier_spec(&ULTRA_V4_SPEC, 10, 4, 128, 32, 2));
        assert!(!is_matryoshka_monotonic(&PRO_V4_SPEC, &FULL_V4_SPEC));
        assert_eq!(LITE_V4_SPEC.lanes(), 16);
        assert_eq!(FULL_V4_SPEC.lanes(), 24);
        assert_eq!(PRO_V4_SPEC.lanes(), 32);
        assert_eq!(ULTRA_V4_SPEC.lanes(), 40);
        assert!(verify_v4_interblock_tiles_and_slots());

        let mut out_counts = [0_usize; MAX_CORE_QUADS_V4];
        for &(sq, _) in &SONA_V4_INTERBLOCK_TILES {
            out_counts[sq] += 1;
        }
        assert_eq!(out_counts, [7, 4, 2, 0, 7, 4, 2, 0, 7, 4]);

        let custom = SonaTierSpec::new([1, 1, 1, 1], 32, 8, 1, 2);
        assert_eq!(custom, LITE_V4_SPEC);
        assert_eq!(custom.block_sum(), 4);
        assert_eq!(custom.total_lanes(), 16);
        assert_eq!(custom.tank_quads(), 1);
        assert!(!verify_tier_spec(&custom, 5, 1, 32, 8, 2));
        assert!(!verify_tier_spec(&custom, 4, 2, 32, 8, 2));
        assert!(!verify_tier_spec(&custom, 4, 1, 64, 8, 2));
        assert!(!verify_tier_spec(&custom, 4, 1, 32, 12, 2));
        assert!(!verify_tier_spec(&custom, 4, 1, 32, 8, 1));
        assert!(is_matryoshka_monotonic(&LITE_V4_SPEC, &FULL_V4_SPEC));
        assert!(!is_matryoshka_monotonic(&ULTRA_V4_SPEC, &ULTRA_V4_SPEC));

        let mut bad_tier_tiles = SONA_V4_INTERBLOCK_TILES;
        bad_tier_tiles[0] = (4, 1);
        assert!(!verify_v4_topology_with(
            &bad_tier_tiles,
            &SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL,
            &SONA_V4_BLOCK_QUAD_SLOTS,
            &SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER,
        ));
        let mut bad_causal_tiles = SONA_V4_INTERBLOCK_TILES;
        bad_causal_tiles[0] = (1, 0);
        assert!(!verify_v4_topology_with(
            &bad_causal_tiles,
            &SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL,
            &SONA_V4_BLOCK_QUAD_SLOTS,
            &SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER,
        ));
        let mut dup_tiles = SONA_V4_INTERBLOCK_TILES;
        dup_tiles[1] = dup_tiles[0];
        assert!(!verify_v4_topology_with(
            &dup_tiles,
            &SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL,
            &SONA_V4_BLOCK_QUAD_SLOTS,
            &SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER,
        ));
        let mut bad_locs = SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL;
        bad_locs[9] = (9, 0);
        assert!(!verify_v4_topology_with(
            &SONA_V4_INTERBLOCK_TILES,
            &bad_locs,
            &SONA_V4_BLOCK_QUAD_SLOTS,
            &SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER,
        ));
        let mut bad_slots = SONA_V4_BLOCK_QUAD_SLOTS;
        bad_slots[0][0] = None;
        assert!(!verify_v4_topology_with(
            &SONA_V4_INTERBLOCK_TILES,
            &SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL,
            &bad_slots,
            &SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER,
        ));
        let mut bad_order = SONA_V4_ULTRA_QUAD_TRAVERSAL_ORDER;
        bad_order.swap(0, 1);
        assert!(!verify_v4_topology_with(
            &SONA_V4_INTERBLOCK_TILES,
            &SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL,
            &SONA_V4_BLOCK_QUAD_SLOTS,
            &bad_order,
        ));
    }
}
