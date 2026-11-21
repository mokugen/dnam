//! 224-D gauge-invariant structural fingerprinting and cosine similarity.
//!
//! Canonical Reference: `docs/specs/security/watermarking_and_marketplace.md`.

use core::mem::offset_of;

use dnam_kernels::fp16_to_f32;
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::U16;

use crate::schema::DnamError;
use dnam_vm::bin::{
    ARENA_TIER_BYTES, CONVOLVER_KAUTZ_BLOCKS_PER_TIER, KautzSoaBlockWire, LINEAR_READOUT_BASES,
    LinearReadoutFullWire, LinearReadoutLiteWire, LinearReadoutProWire, LinearReadoutUltraWire,
    TRANSPORT_DELAY_BASES, TransportQuadWire,
};
use dnam_vm::math::floor_f64;

use super::watermark::{kautz_soa_byte_offset, max_tiers_for_arena_len};
use super::{STRUCTURAL_HASH_DIM, STRUCTURAL_HASH_TIER_DIM};

#[inline]
pub(crate) fn quantize_pole_angle_1e2(theta: f64) -> f64 {
    // Aligns 1:1 with the `10^-2 rad` `[0.2, 0.8]` floor-cell guard band in `phase.rs` (`ref[sec.watermark.seed-derivation]`).
    floor_f64(theta * 100.0) * 0.01
}

/// Extracts the 224-dimensional prefix-composable gauge-invariant structural hash vector $\mathbf{f}_{\text{struct}} \in \mathbb{R}^{224}$.
///
/// Ordered strictly by tier $t \in \{0, 1, 2, 3\}$ (`[56*t .. 56*(t+1)]`):
/// - `[0..16]`: modal/Kautz invariant composite pole signatures $r_{\text{low},b,c} + 0.5\,\Delta r_{b,c} + 0.1\,\lfloor\theta_{0,b,c}\rceil_{10^{-2}}$,
/// - `[16..32]`: gauge-invariant real products $I_{\text{real},b,c} = b_{\gamma,b,c} \cdot (c_{\text{real},L,b,c} + c_{\text{real},R,b,c})$,
/// - `[32..48]`: gauge-invariant imaginary products $I_{\text{imag},b,c} = b_{\gamma,b,c} \cdot (c_{\text{imag},L,b,c} + c_{\text{imag},R,b,c})$,
/// - `[48..52]`: 4 `TransportQuadWire::delay_samples` invariants for tier $t$ (IEEE-754 `f32le` widened to `f64`, `0.0` when non-finite),
/// - `[52..56]`: 4 `LinearReadout` `fp16` `W_readout` energy invariants for tier $t$ (invariant under `sona_gauge` & `isometric_gauge`).
///
/// Any stripped higher tier remains zero-filled, enabling exact prefix cosine similarity across tier-stripped containers.
pub fn extract_structural_hash_224d(arena: &[u8]) -> Result<[f64; STRUCTURAL_HASH_DIM], DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_TIER_BYTES[0],
            actual: arena.len(),
        });
    }
    let mut vec224 = [0.0_f64; STRUCTURAL_HASH_DIM];

    const W_RO_OFFSETS: [usize; 4] = [
        LINEAR_READOUT_BASES[0] + offset_of!(LinearReadoutLiteWire, w_ro_fp16),
        LINEAR_READOUT_BASES[1] + offset_of!(LinearReadoutFullWire, w_ro_and_mods_fp16),
        LINEAR_READOUT_BASES[2] + offset_of!(LinearReadoutProWire, w_ro_and_mods_fp16),
        LINEAR_READOUT_BASES[3] + offset_of!(LinearReadoutUltraWire, w_ro_and_mods_fp16),
    ];
    const W_RO_SEG_U16S: [usize; 4] = [48, 72, 96, 120];

    for tier in 0..max_tiers {
        let t_base = tier * STRUCTURAL_HASH_TIER_DIM;
        let blocks = CONVOLVER_KAUTZ_BLOCKS_PER_TIER[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            let Ok((k_blk, _)) = KautzSoaBlockWire::ref_from_prefix(&arena[base..]) else {
                continue;
            };
            for lane in 0..4 {
                let ch_in_tier = blk * 4 + lane;
                let r_low = f64::from(k_blk.r_low[lane].get());
                let dr = f64::from(k_blk.dr[lane].get());
                let theta = quantize_pole_angle_1e2(f64::from(k_blk.theta[lane].get()));
                let b_gamma = f64::from(k_blk.b_gamma[lane].get());
                let c_re_l = f64::from(k_blk.c_re_l[lane].get());
                let c_im_l = f64::from(k_blk.c_im_l[lane].get());
                let c_re_r = f64::from(k_blk.c_re_r[lane].get());
                let c_im_r = f64::from(k_blk.c_im_r[lane].get());

                vec224[t_base + ch_in_tier] = r_low + 0.5 * dr + 0.1 * theta;
                vec224[t_base + 16 + ch_in_tier] = b_gamma * (c_re_l + c_re_r);
                vec224[t_base + 32 + ch_in_tier] = b_gamma * (c_im_l + c_im_r);
            }
        }

        // 4 transport delay invariants for this tier (`f32le` widened to `f64`, `0.0` if non-finite)
        if let Some(delay_tail) = arena.get(TRANSPORT_DELAY_BASES[tier]..)
            && let Ok((tq, _)) = TransportQuadWire::ref_from_prefix(delay_tail)
        {
            for lane in 0..4 {
                let f = tq.delay_samples[lane].get();
                vec224[t_base + 48 + lane] = if f.is_finite() {
                    let v = f64::from(f);
                    if v == 0.0 { 0.0 } else { v }
                } else {
                    0.0
                };
            }
        }

        // 4 Direct `LinearReadout` `fp16` energy norms (`4 * [48, 72, 96, 120]` U16s = `[192, 288, 384, 480]`)
        let ro_base = W_RO_OFFSETS[tier];
        let seg_len = W_RO_SEG_U16S[tier];
        if let Some(ro_tail) = arena.get(ro_base..)
            && let Ok((ro_words, _)) = <[U16]>::ref_from_prefix_with_elems(ro_tail, 4 * seg_len)
        {
            for seg in 0..4 {
                let mut energy = 0.0_f64;
                for i in 0..seg_len {
                    let v = f64::from(fp16_to_f32(ro_words[seg * seg_len + i].get()));
                    if v.is_finite() {
                        energy += v * v;
                    }
                }
                vec224[t_base + 52 + seg] = energy;
            }
        }
    }

    Ok(vec224)
}

#[inline]
fn populated_tier_count(v: &[f64; STRUCTURAL_HASH_DIM]) -> usize {
    let mut tiers = 1_usize;
    for t in 1..4 {
        let slice = &v[t * STRUCTURAL_HASH_TIER_DIM..(t + 1) * STRUCTURAL_HASH_TIER_DIM];
        if slice.iter().any(|&x| x.abs() > 1e-30) {
            tiers = t + 1;
        }
    }
    tiers
}

/// Computes the prefix-composable cosine similarity $\cos(\mathbf{f}_1, \mathbf{f}_2) \in [-1, +1]$
/// over the shared populated tier prefix (`56 * min(tiers_a, tiers_b)` dimensions).
pub fn structural_cosine_similarity(
    a: &[f64; STRUCTURAL_HASH_DIM],
    b: &[f64; STRUCTURAL_HASH_DIM],
) -> f64 {
    let shared_tiers = populated_tier_count(a).min(populated_tier_count(b));
    let active_dim = shared_tiers * STRUCTURAL_HASH_TIER_DIM;
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut i = 0;
    while i < active_dim {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        i += 1;
    }
    let denom = crate::schema::cold_sqrt_f64(na * nb);
    if denom <= 1e-30 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_structural_cosine_similarity_scale_and_prefix_invariance(
            base_val in 0.05_f64..10.0_f64,
            scale in 0.1_f64..20.0_f64,
            extra_val in 0.05_f64..10.0_f64,
        ) {
            let mut f_lite = [0.0_f64; STRUCTURAL_HASH_DIM];
            let mut f_full = [0.0_f64; STRUCTURAL_HASH_DIM];
            for i in 0..STRUCTURAL_HASH_TIER_DIM {
                f_lite[i] = base_val * ((i + 1) as f64);
                f_full[i] = f_lite[i] * scale;
                f_full[STRUCTURAL_HASH_TIER_DIM + i] = extra_val * ((i + 1) as f64);
            }
            // Prefix-composable cosine similarity between Lite (1 tier) and scaled Full (2 tiers) must be 1.0
            let sim = structural_cosine_similarity(&f_lite, &f_full);
            prop_assert!((sim - 1.0).abs() < 1e-12, "Expected 1.0, got {}", sim);
        }
    }

    #[test]
    fn test_extract_structural_hash_224d_transport_f32le_and_linear_readout() {
        use dnam_kernels::f32_to_fp16;
        use zerocopy::byteorder::little_endian::F32;

        let mut arena = alloc::vec![0u8; ARENA_TIER_BYTES[3]];
        let (tq0, _) =
            TransportQuadWire::mut_from_prefix(&mut arena[TRANSPORT_DELAY_BASES[0]..]).unwrap();
        tq0.delay_samples = [
            F32::new(42.5_f32),
            F32::new(f32::NAN),
            F32::new(f32::INFINITY),
            F32::new(-0.0_f32),
        ];

        let ro0_off = LINEAR_READOUT_BASES[0] + offset_of!(LinearReadoutLiteWire, w_ro_fp16);
        let (ro0, _) = <[U16]>::mut_from_prefix_with_elems(&mut arena[ro0_off..], 192).unwrap();
        ro0[0] = U16::new(f32_to_fp16(1.5));
        ro0[48] = U16::new(f32_to_fp16(-2.0));

        let hash = extract_structural_hash_224d(&arena).unwrap();
        assert!((hash[48] - 42.5).abs() < 1e-12);
        assert_eq!(hash[49], 0.0);
        assert_eq!(hash[50], 0.0);
        assert_eq!(hash[51], 0.0);
        assert!((hash[52] - 2.25).abs() < 1e-6);
        assert!((hash[53] - 4.0).abs() < 1e-6);
    }
}
