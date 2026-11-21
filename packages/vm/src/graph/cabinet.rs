//! Tier-partitioned `48 kHz` lag-ordered Cabinet early FIR + modal onset stage and submodules
//! ([`kautz`], [`high_sr`]) (`ref[dsp.cabinet.mirrored-fir-simd]`, `ref[dsp.op-m.kautz.tier-allocation]`).

pub mod high_sr;
pub mod kautz;

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};

/// Number of `f32` taps evaluated per SIMD 4x4 tile segment (`16`).
pub const CABINET_SEG_TAPS: usize = 16;

/// `Lite` tier cumulative `48 kHz` FIR length per channel (`32`).
pub const CABINET_LITE_FIR_TAPS: usize = 32;

/// `Full` delta tier additional `48 kHz` FIR taps per channel (`+16`, cumulative `48`).
pub const CABINET_FULL_DELTA_FIR_TAPS: usize = 16;

/// `Pro` delta tier additional `48 kHz` FIR taps per channel (`+48`, cumulative `96`).
pub const CABINET_PRO_DELTA_FIR_TAPS: usize = 48;

/// `Ultra` delta tier additional `48 kHz` FIR taps per channel (`+32`, cumulative `128`).
pub const CABINET_ULTRA_DELTA_FIR_TAPS: usize = 32;

/// Maximum cumulative `48 kHz` early FIR taps per channel across all tiers (`128`).
pub const CABINET_MAX_FIR_TAPS: usize = CABINET_LITE_FIR_TAPS
    + CABINET_FULL_DELTA_FIR_TAPS
    + CABINET_PRO_DELTA_FIR_TAPS
    + CABINET_ULTRA_DELTA_FIR_TAPS;

/// Maximum causal micro-chunk length supported by the stack FIR work buffer (`4` samples).
pub const CABINET_MAX_CHUNK_FRAMES: usize = 4;

/// Total stack scratch work-buffer elements per channel (`N_max + 4 = 132` floats = `528 B`).
pub const CABINET_WORK_LEN: usize = CABINET_MAX_FIR_TAPS + CABINET_MAX_CHUNK_FRAMES;

/// Nominal `48 kHz` Kautz modal bank onset delay (`16` samples, `0.333 ms`).
pub const CABINET_KAUTZ_ONSET_DELAY_48K: usize = 16;

const MAX_SEGMENTS: usize = CABINET_MAX_FIR_TAPS / CABINET_SEG_TAPS;

const _: () = assert!(CABINET_MAX_FIR_TAPS == 128);
const _: () = assert!(CABINET_WORK_LEN == 132);
const _: () = assert!(MAX_SEGMENTS == 8);
const _: () = assert!(CABINET_SEG_TAPS.is_power_of_two());
const _: () = assert!((CABINET_LITE_FIR_TAPS & (CABINET_SEG_TAPS - 1)) == 0);
const _: () = assert!((CABINET_FULL_DELTA_FIR_TAPS & (CABINET_SEG_TAPS - 1)) == 0);
const _: () = assert!((CABINET_PRO_DELTA_FIR_TAPS & (CABINET_SEG_TAPS - 1)) == 0);
const _: () = assert!((CABINET_ULTRA_DELTA_FIR_TAPS & (CABINET_SEG_TAPS - 1)) == 0);
const _: () = assert!(CABINET_KAUTZ_ONSET_DELAY_48K < CABINET_LITE_FIR_TAPS);

/// Mutable partitioned `48 kHz` stereo lag-ordered FIR delay history slices (`Lite` 32,
/// `Full` +16, `Pro` +48, `Ultra` +32 taps/ch). Inactive upper tiers pass `&mut []`.
#[derive(Debug)]
pub struct CabinetFirHistorySlices<'a> {
    /// Required `Lite` tier left history (`>= 32` floats).
    pub lite_l: &'a mut [f32],
    /// Required `Lite` tier right history (`>= 32` floats).
    pub lite_r: &'a mut [f32],
    /// Optional `Full` delta left history (`>= 16` floats when active).
    pub full_l: &'a mut [f32],
    /// Optional `Full` delta right history (`>= 16` floats when active).
    pub full_r: &'a mut [f32],
    /// Optional `Pro` delta left history (`>= 48` floats when active).
    pub pro_l: &'a mut [f32],
    /// Optional `Pro` delta right history (`>= 48` floats when active).
    pub pro_r: &'a mut [f32],
    /// Optional `Ultra` delta left history (`>= 32` floats when active).
    pub ultra_l: &'a mut [f32],
    /// Optional `Ultra` delta right history (`>= 32` floats when active).
    pub ultra_r: &'a mut [f32],
}

impl<'a> CabinetFirHistorySlices<'a> {
    /// Constructs a `Lite`-only history view (`32` taps/ch).
    #[inline]
    pub fn lite(lite_l: &'a mut [f32], lite_r: &'a mut [f32]) -> Self {
        Self {
            lite_l,
            lite_r,
            full_l: &mut [],
            full_r: &mut [],
            pro_l: &mut [],
            pro_r: &mut [],
            ultra_l: &mut [],
            ultra_r: &mut [],
        }
    }

    /// Zeroes all populated tier history slices (`reset()`).
    #[inline]
    pub fn reset(&mut self) {
        self.lite_l.fill(0.0);
        self.lite_r.fill(0.0);
        self.full_l.fill(0.0);
        self.full_r.fill(0.0);
        self.pro_l.fill(0.0);
        self.pro_r.fill(0.0);
        self.ultra_l.fill(0.0);
        self.ultra_r.fill(0.0);
    }

    #[inline]
    fn has_lite(&self) -> bool {
        self.lite_l.len() >= CABINET_LITE_FIR_TAPS && self.lite_r.len() >= CABINET_LITE_FIR_TAPS
    }

    #[inline]
    fn has_full(&self) -> bool {
        self.full_l.len() >= CABINET_FULL_DELTA_FIR_TAPS
            && self.full_r.len() >= CABINET_FULL_DELTA_FIR_TAPS
    }

    #[inline]
    fn has_pro(&self) -> bool {
        self.pro_l.len() >= CABINET_PRO_DELTA_FIR_TAPS
            && self.pro_r.len() >= CABINET_PRO_DELTA_FIR_TAPS
    }

    #[inline]
    fn has_ultra(&self) -> bool {
        self.ultra_l.len() >= CABINET_ULTRA_DELTA_FIR_TAPS
            && self.ultra_r.len() >= CABINET_ULTRA_DELTA_FIR_TAPS
    }
}

/// Immutable partitioned `48 kHz` stereo FIR tap weight slices (`Lite` 32, `Full` +16,
/// `Pro` +48, `Ultra` +32 taps/ch). Inactive upper tiers pass `&[]`.
#[derive(Clone, Copy, Debug)]
pub struct CabinetFirWeightSlices<'a> {
    /// Required `Lite` tier left FIR coefficients (`>= 32` floats).
    pub lite_l: &'a [f32],
    /// Required `Lite` tier right FIR coefficients (`>= 32` floats).
    pub lite_r: &'a [f32],
    /// Optional `Full` delta left FIR coefficients (`>= 16` floats when active).
    pub full_l: &'a [f32],
    /// Optional `Full` delta right FIR coefficients (`>= 16` floats when active).
    pub full_r: &'a [f32],
    /// Optional `Pro` delta left FIR coefficients (`>= 48` floats when active).
    pub pro_l: &'a [f32],
    /// Optional `Pro` delta right FIR coefficients (`>= 48` floats when active).
    pub pro_r: &'a [f32],
    /// Optional `Ultra` delta left FIR coefficients (`>= 32` floats when active).
    pub ultra_l: &'a [f32],
    /// Optional `Ultra` delta right FIR coefficients (`>= 32` floats when active).
    pub ultra_r: &'a [f32],
}

impl<'a> CabinetFirWeightSlices<'a> {
    /// Constructs a `Lite`-only weight view (`32` taps/ch).
    #[inline]
    pub const fn lite(lite_l: &'a [f32], lite_r: &'a [f32]) -> Self {
        Self {
            lite_l,
            lite_r,
            full_l: &[],
            full_r: &[],
            pro_l: &[],
            pro_r: &[],
            ultra_l: &[],
            ultra_r: &[],
        }
    }

    /// Returns cumulative active `48 kHz` tap count supported by `self` (`0`, `32`, `48`, `96`, `128`).
    #[inline]
    pub fn active_taps(&self) -> usize {
        if !self.has_lite() {
            return 0;
        }
        let mut n = CABINET_LITE_FIR_TAPS;
        if self.has_full() {
            n += CABINET_FULL_DELTA_FIR_TAPS;
            if self.has_pro() {
                n += CABINET_PRO_DELTA_FIR_TAPS;
                if self.has_ultra() {
                    n += CABINET_ULTRA_DELTA_FIR_TAPS;
                }
            }
        }
        n
    }

    /// Copies cumulative weights into flat `[f32; 128]` buffers (useful for `CabinetHighSrFirExt::prepare`),
    /// returning the active tap count copied.
    pub fn copy_cumulative_weights(
        &self,
        out_l: &mut [f32; CABINET_MAX_FIR_TAPS],
        out_r: &mut [f32; CABINET_MAX_FIR_TAPS],
    ) -> usize {
        out_l.fill(0.0);
        out_r.fill(0.0);
        let active = self.active_taps();
        if active >= 32 {
            out_l[..32].copy_from_slice(&self.lite_l[..32]);
            out_r[..32].copy_from_slice(&self.lite_r[..32]);
        }
        if active >= 48 {
            out_l[32..48].copy_from_slice(&self.full_l[..16]);
            out_r[32..48].copy_from_slice(&self.full_r[..16]);
        }
        if active >= 96 {
            out_l[48..96].copy_from_slice(&self.pro_l[..48]);
            out_r[48..96].copy_from_slice(&self.pro_r[..48]);
        }
        if active == 128 {
            out_l[96..128].copy_from_slice(&self.ultra_l[..32]);
            out_r[96..128].copy_from_slice(&self.ultra_r[..32]);
        }
        active
    }

    #[inline]
    fn has_lite(&self) -> bool {
        self.lite_l.len() >= CABINET_LITE_FIR_TAPS && self.lite_r.len() >= CABINET_LITE_FIR_TAPS
    }

    #[inline]
    fn has_full(&self) -> bool {
        self.full_l.len() >= CABINET_FULL_DELTA_FIR_TAPS
            && self.full_r.len() >= CABINET_FULL_DELTA_FIR_TAPS
    }

    #[inline]
    fn has_pro(&self) -> bool {
        self.pro_l.len() >= CABINET_PRO_DELTA_FIR_TAPS
            && self.pro_r.len() >= CABINET_PRO_DELTA_FIR_TAPS
    }

    #[inline]
    fn has_ultra(&self) -> bool {
        self.ultra_l.len() >= CABINET_ULTRA_DELTA_FIR_TAPS
            && self.ultra_r.len() >= CABINET_ULTRA_DELTA_FIR_TAPS
    }
}

/// Default `16`-tap stereo dot-product kernel over any [`MathBackend`] (`dot16_stereo` signature).
#[inline(always)]
pub fn dot16_stereo_simd<B: MathBackend>(
    w_l: &[[f32; 4]; 4],
    w_r: &[[f32; 4]; 4],
    x_l: &[[f32; 4]; 4],
    x_r: &[[f32; 4]; 4],
) -> (f32, f32) {
    let acc0_l = B::Vec4::load(&w_l[0]).mul(B::Vec4::load(&x_l[0]));
    let acc0_r = B::Vec4::load(&w_r[0]).mul(B::Vec4::load(&x_r[0]));
    let acc1_l = acc0_l.fma(B::Vec4::load(&w_l[1]), B::Vec4::load(&x_l[1]));
    let acc1_r = acc0_r.fma(B::Vec4::load(&w_r[1]), B::Vec4::load(&x_r[1]));
    let acc2_l = acc1_l.fma(B::Vec4::load(&w_l[2]), B::Vec4::load(&x_l[2]));
    let acc2_r = acc1_r.fma(B::Vec4::load(&w_r[2]), B::Vec4::load(&x_r[2]));
    let acc3_l = acc2_l.fma(B::Vec4::load(&w_l[3]), B::Vec4::load(&x_l[3]));
    let acc3_r = acc2_r.fma(B::Vec4::load(&w_r[3]), B::Vec4::load(&x_r[3]));
    acc3_l.horizontal_sum_pair(acc3_r)
}

#[inline(always)]
fn slice16_to_4x4(src: &[f32]) -> [[f32; 4]; 4] {
    [
        [src[0], src[1], src[2], src[3]],
        [src[4], src[5], src[6], src[7]],
        [src[8], src[9], src[10], src[11]],
        [src[12], src[13], src[14], src[15]],
    ]
}

#[inline]
fn resolve_active_segments(
    hist: &CabinetFirHistorySlices<'_>,
    weights: &CabinetFirWeightSlices<'_>,
) -> usize {
    if !hist.has_lite() || !weights.has_lite() {
        return 0;
    }
    let mut segs = 2;
    if hist.has_full() && weights.has_full() {
        segs += 1;
        if hist.has_pro() && weights.has_pro() {
            segs += 3;
            if hist.has_ultra() && weights.has_ultra() {
                segs += 2;
            }
        }
    }
    segs
}

#[inline]
fn weight_segment_pair<'a>(
    weights: &'a CabinetFirWeightSlices<'_>,
    seg: usize,
) -> (&'a [f32], &'a [f32]) {
    match seg {
        0 => (&weights.lite_l[..16], &weights.lite_r[..16]),
        1 => (&weights.lite_l[16..32], &weights.lite_r[16..32]),
        2 => (&weights.full_l[..16], &weights.full_r[..16]),
        3 => (&weights.pro_l[..16], &weights.pro_r[..16]),
        4 => (&weights.pro_l[16..32], &weights.pro_r[16..32]),
        5 => (&weights.pro_l[32..48], &weights.pro_r[32..48]),
        6 => (&weights.ultra_l[..16], &weights.ultra_r[..16]),
        _ => (&weights.ultra_l[16..32], &weights.ultra_r[16..32]),
    }
}

/// Evaluates `C` (`1..=4`) frames of lag-ordered `48 kHz` Cabinet early FIR + `16`-sample modal
/// onset into `early_out` and `onset_out`, updating partitioned lag-ordered histories with
/// `0-ULP` chunk parity (`C = 1 x 4 == C = 4`, `ref[dsp.cabinet.mirrored-fir-simd]`).
#[inline]
pub fn process_cabinet_48k_chunk<const C: usize, F>(
    hist: &mut CabinetFirHistorySlices<'_>,
    weights: &CabinetFirWeightSlices<'_>,
    inputs_lr: &[[f32; 2]; C],
    early_out: &mut [[f32; 2]; C],
    onset_out: &mut [[f32; 2]; C],
    dot16: F,
) where
    F: Fn(&[[f32; 4]; 4], &[[f32; 4]; 4], &[[f32; 4]; 4], &[[f32; 4]; 4]) -> (f32, f32),
{
    let num_segs = resolve_active_segments(hist, weights);
    if C == 0 || C > CABINET_MAX_CHUNK_FRAMES || num_segs == 0 {
        early_out.fill([0.0, 0.0]);
        onset_out.fill([0.0, 0.0]);
        return;
    }

    let mut work_l = [0.0_f32; CABINET_WORK_LEN];
    let mut work_r = [0.0_f32; CABINET_WORK_LEN];

    for s in 0..C {
        work_l[C - 1 - s] = flush_recursive_state_f32(inputs_lr[s][0]);
        work_r[C - 1 - s] = flush_recursive_state_f32(inputs_lr[s][1]);
    }
    work_l[C..C + 32].copy_from_slice(&hist.lite_l[..32]);
    work_r[C..C + 32].copy_from_slice(&hist.lite_r[..32]);
    if num_segs >= 3 {
        work_l[C + 32..C + 48].copy_from_slice(&hist.full_l[..16]);
        work_r[C + 32..C + 48].copy_from_slice(&hist.full_r[..16]);
    }
    if num_segs >= 6 {
        work_l[C + 48..C + 96].copy_from_slice(&hist.pro_l[..48]);
        work_r[C + 48..C + 96].copy_from_slice(&hist.pro_r[..48]);
    }
    if num_segs == 8 {
        work_l[C + 96..C + 128].copy_from_slice(&hist.ultra_l[..32]);
        work_r[C + 96..C + 128].copy_from_slice(&hist.ultra_r[..32]);
    }

    let mut w_tiles_l = [[[0.0_f32; 4]; 4]; MAX_SEGMENTS];
    let mut w_tiles_r = [[[0.0_f32; 4]; 4]; MAX_SEGMENTS];
    for seg in 0..num_segs {
        let (wl, wr) = weight_segment_pair(weights, seg);
        w_tiles_l[seg] = slice16_to_4x4(wl);
        w_tiles_r[seg] = slice16_to_4x4(wr);
    }

    for s in 0..C {
        let start = C - 1 - s;
        onset_out[s] = [
            work_l[start + CABINET_KAUTZ_ONSET_DELAY_48K],
            work_r[start + CABINET_KAUTZ_ONSET_DELAY_48K],
        ];
        let mut acc_l = 0.0_f32;
        let mut acc_r = 0.0_f32;
        for seg in 0..num_segs {
            let off = start + seg * CABINET_SEG_TAPS;
            let xl = slice16_to_4x4(&work_l[off..off + CABINET_SEG_TAPS]);
            let xr = slice16_to_4x4(&work_r[off..off + CABINET_SEG_TAPS]);
            let (seg_l, seg_r) = dot16(&w_tiles_l[seg], &w_tiles_r[seg], &xl, &xr);
            acc_l += seg_l;
            acc_r += seg_r;
        }
        early_out[s] = [acc_l, acc_r];
    }

    hist.lite_l[..32].copy_from_slice(&work_l[..32]);
    hist.lite_r[..32].copy_from_slice(&work_r[..32]);
    if num_segs >= 3 {
        hist.full_l[..16].copy_from_slice(&work_l[32..48]);
        hist.full_r[..16].copy_from_slice(&work_r[32..48]);
    }
    if num_segs >= 6 {
        hist.pro_l[..48].copy_from_slice(&work_l[48..96]);
        hist.pro_r[..48].copy_from_slice(&work_r[48..96]);
    }
    if num_segs == 8 {
        hist.ultra_l[..32].copy_from_slice(&work_l[96..128]);
        hist.ultra_r[..32].copy_from_slice(&work_r[96..128]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};

    #[test]
    fn test_cabinet_48k_zero_ulp_chunk_parity_across_all_tiers_and_onset_16() {
        let mut wl = [0.0_f32; CABINET_MAX_FIR_TAPS];
        let mut wr = [0.0_f32; CABINET_MAX_FIR_TAPS];
        for i in 0..CABINET_MAX_FIR_TAPS {
            wl[i] = ((i as f32) + 1.0) * 0.003_75 * if i & 1 == 0 { 1.0 } else { -0.5 };
            wr[i] = ((i as f32) + 2.0) * 0.002_5 * if i % 3 == 0 { -1.0 } else { 0.75 };
        }

        let inputs: [[f32; 2]; 24] = core::array::from_fn(|i| {
            [
                if i == 0 {
                    1.0
                } else {
                    ((i as f32) - 5.0) * 0.09
                },
                if i == 0 {
                    -0.5
                } else {
                    (11.0 - (i as f32)) * 0.07
                },
            ]
        });

        for tier_taps in [32_usize, 48, 96, 128] {
            let weights = CabinetFirWeightSlices {
                lite_l: &wl[..32],
                lite_r: &wr[..32],
                full_l: if tier_taps >= 48 { &wl[32..48] } else { &[] },
                full_r: if tier_taps >= 48 { &wr[32..48] } else { &[] },
                pro_l: if tier_taps >= 96 { &wl[48..96] } else { &[] },
                pro_r: if tier_taps >= 96 { &wr[48..96] } else { &[] },
                ultra_l: if tier_taps == 128 { &wl[96..128] } else { &[] },
                ultra_r: if tier_taps == 128 { &wr[96..128] } else { &[] },
            };
            assert_eq!(weights.active_taps(), tier_taps);
            if tier_taps == 128 {
                let (mut flat_l, mut flat_r) = ([0.0_f32; 128], [0.0_f32; 128]);
                assert_eq!(
                    weights.copy_cumulative_weights(&mut flat_l, &mut flat_r),
                    128
                );
                assert_eq!(flat_l, wl);
                assert_eq!(flat_r, wr);
            }

            let (mut hl1, mut hr1) = ([0.0_f32; 128], [0.0_f32; 128]);
            let (mut hl4, mut hr4) = ([0.0_f32; 128], [0.0_f32; 128]);

            fn make_hist<'a>(
                l: &'a mut [f32; 128],
                r: &'a mut [f32; 128],
                tier_taps: usize,
            ) -> CabinetFirHistorySlices<'a> {
                let (l_lite, l_rest) = l.split_at_mut(32);
                let (l_full, l_rest2) = l_rest.split_at_mut(16);
                let (l_pro, l_ultra) = l_rest2.split_at_mut(48);
                let (r_lite, r_rest) = r.split_at_mut(32);
                let (r_full, r_rest2) = r_rest.split_at_mut(16);
                let (r_pro, r_ultra) = r_rest2.split_at_mut(48);
                CabinetFirHistorySlices {
                    lite_l: l_lite,
                    lite_r: r_lite,
                    full_l: if tier_taps >= 48 { l_full } else { &mut [] },
                    full_r: if tier_taps >= 48 { r_full } else { &mut [] },
                    pro_l: if tier_taps >= 96 { l_pro } else { &mut [] },
                    pro_r: if tier_taps >= 96 { r_pro } else { &mut [] },
                    ultra_l: if tier_taps == 128 { l_ultra } else { &mut [] },
                    ultra_r: if tier_taps == 128 { r_ultra } else { &mut [] },
                }
            }

            let mut out1_e = [[0.0_f32; 2]; 24];
            let mut out1_o = [[0.0_f32; 2]; 24];
            {
                let mut h1 = make_hist(&mut hl1, &mut hr1, tier_taps);
                for i in 0..24 {
                    let mut e = [[0.0_f32; 2]; 1];
                    let mut o = [[0.0_f32; 2]; 1];
                    process_cabinet_48k_chunk::<1, _>(
                        &mut h1,
                        &weights,
                        &[inputs[i]],
                        &mut e,
                        &mut o,
                        dot16_stereo_simd::<ScalarBackend>,
                    );
                    out1_e[i] = e[0];
                    out1_o[i] = o[0];
                }
            }

            let mut out4_e = [[0.0_f32; 2]; 24];
            let mut out4_o = [[0.0_f32; 2]; 24];
            {
                let mut h4 = make_hist(&mut hl4, &mut hr4, tier_taps);
                for chk in 0..6 {
                    let base = chk * 4;
                    let chunk_in: [[f32; 2]; 4] = [
                        inputs[base],
                        inputs[base + 1],
                        inputs[base + 2],
                        inputs[base + 3],
                    ];
                    let mut e = [[0.0_f32; 2]; 4];
                    let mut o = [[0.0_f32; 2]; 4];
                    process_cabinet_48k_chunk::<4, _>(
                        &mut h4,
                        &weights,
                        &chunk_in,
                        &mut e,
                        &mut o,
                        dot16_stereo_simd::<ScalarBackend>,
                    );
                    out4_e[base..base + 4].copy_from_slice(&e);
                    out4_o[base..base + 4].copy_from_slice(&o);
                }
            }

            for i in 0..24 {
                assert_eq!(out1_e[i][0].to_bits(), out4_e[i][0].to_bits());
                assert_eq!(out1_e[i][1].to_bits(), out4_e[i][1].to_bits());
                assert_eq!(out1_o[i][0].to_bits(), out4_o[i][0].to_bits());
                assert_eq!(out1_o[i][1].to_bits(), out4_o[i][1].to_bits());
            }
            for t in 0..tier_taps {
                assert_eq!(hl1[t].to_bits(), hl4[t].to_bits());
                assert_eq!(hr1[t].to_bits(), hr4[t].to_bits());
            }
        }

        // Verify exact 16-sample onset delay with DefaultMathBackend on Lite impulse.
        let (mut hl, mut hr) = ([0.0_f32; 32], [0.0_f32; 32]);
        let mut h = CabinetFirHistorySlices::lite(&mut hl, &mut hr);
        let w = CabinetFirWeightSlices::lite(&wl[..32], &wr[..32]);
        let mut impulse = [[0.0_f32; 2]; 20];
        impulse[0] = [1.0, -0.25];
        for n in 0..5 {
            let base = n * 4;
            let cin = [
                impulse[base],
                impulse[base + 1],
                impulse[base + 2],
                impulse[base + 3],
            ];
            let (mut eo, mut oo) = ([[0.0_f32; 2]; 4], [[0.0_f32; 2]; 4]);
            process_cabinet_48k_chunk::<4, _>(
                &mut h,
                &w,
                &cin,
                &mut eo,
                &mut oo,
                dot16_stereo_simd::<DefaultMathBackend>,
            );
            for k in 0..4 {
                let sample_idx = base + k;
                let expected = if sample_idx == 16 {
                    [1.0, -0.25]
                } else {
                    [0.0, 0.0]
                };
                assert_eq!(oo[k], expected, "sample_idx={sample_idx}");
            }
        }
        h.reset();
        assert_eq!(hl, [0.0; 32]);
    }
}
