//! Master-mode `4x` (`63`-tap + `23`-tap) linear-phase half-band FIR oversampling stages and
//! `Box2` composite quadrature helpers (`ref[arch.bounce.oversampling]`, `ADR-0026`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`, and
//! `spirv-unknown-naga-wgsl`-clean.

use crate::simd::{
    HB23_DN_ODD_TAPS_OUTER, HB23_ODD_TAP_COUNT, HB23_ODD_TAPS_OUTER, HB63_DN_ODD_TAPS_OUTER,
    HB63_ODD_TAP_COUNT, HB63_ODD_TAPS_OUTER, KernelVec4, StereoVec8, canonicalize_zero_f32_bits,
};

/// Stage-1 (`1x <-> 2x`) half-order group delay in `2x` samples for `HB63` (`31`).
pub const HB63_DELAY_SAMPLES_2X: usize = 31;

/// Stage-2 (`2x <-> 4x`) half-order group delay in `4x` samples for `HB23` (`11`).
pub const HB23_DELAY_SAMPLES_4X: usize = 11;

/// Live-mode (`15`-tap `2x`) exact integer round-trip PDC latency in `1x` host samples (`7`).
pub const CORE_OS_LIVE_LATENCY_1X: usize = 7;

/// Master-mode `4x` (`44.1..96 kHz`: `HB63` + `z^-1@2x` + `HB23`) exact integer PDC latency (`37`).
pub const CORE_OS_MASTER_4X_LATENCY_1X: usize = 37;

/// Master-mode `2x` (`176.4/192 kHz`: `HB63` only) exact integer PDC latency (`31`).
pub const CORE_OS_MASTER_2X_LATENCY_1X: usize = 31;

/// Past `1x`-frame history depth for symmetric `63`-tap polyphase even pairs (`31`).
pub const HB63_HIST_LEN: usize = 31;

/// Past `1x`-frame delay depth for the odd center-tap branch in `2x -> 1x` `HB63` decimation (`16`).
pub const HB63_ODD_DELAY_LEN: usize = 16;

/// Past `2x`-frame history depth for symmetric `23`-tap polyphase even pairs (`11`).
pub const HB23_HIST_LEN: usize = 11;

/// Past `2x`-frame delay depth for the odd center-tap branch in `4x -> 2x` `HB23` decimation (`6`).
pub const HB23_ODD_DELAY_LEN: usize = 6;

const _: () = assert!(HB63_DELAY_SAMPLES_2X == 31 && HB63_HIST_LEN == 31);
const _: () = assert!(HB63_ODD_DELAY_LEN == HB63_ODD_TAP_COUNT);
const _: () = assert!(HB23_DELAY_SAMPLES_4X == 11 && HB23_HIST_LEN == 11);
const _: () = assert!(HB23_ODD_DELAY_LEN == HB23_ODD_TAP_COUNT);
const _: () = assert!(CORE_OS_LIVE_LATENCY_1X == 7);
const _: () = assert!(CORE_OS_MASTER_2X_LATENCY_1X == 31);
const _: () = assert!(CORE_OS_MASTER_4X_LATENCY_1X == 37);

/// Evaluates scalar `Box2` 2-subpanel composite antiderivative quadrature over consecutive
/// same-phase `4x` core outputs `(prev_sample, curr_sample)` (`0.5 * (prev_sample + curr_sample)`),
/// bypassed cleanly to `curr_sample` when `lin_mix >= 1.0` (paired with
/// [`crate::simd::os2x_midpoint_vec4`], `ref[arch.bounce.oversampling]`).
#[inline(always)]
#[must_use]
pub fn box2_quadrature_f32(prev_sample: f32, curr_sample: f32, lin_mix: f32) -> f32 {
    if lin_mix >= 1.0 {
        canonicalize_zero_f32_bits(curr_sample)
    } else {
        canonicalize_zero_f32_bits(0.5 * (prev_sample + curr_sample))
    }
}

/// Evaluates branchless `StereoVec8` `Box2` 2-subpanel composite antiderivative quadrature over
/// consecutive same-phase `4x` outputs `(prev_sample, curr_sample)`, bypassing `Box2` per lane
/// when `lin_mix >= 1.0` (`lin_mix == 1.0` already aligns via
/// [`crate::simd::os2x_midpoint_vec4`], `ref[arch.bounce.oversampling]`).
#[inline(always)]
#[must_use]
pub fn box2_quadrature_stereo8<S: StereoVec8>(prev_sample: S, curr_sample: S, lin_mix: S) -> S {
    let half = S::Half::splat(0.5);
    let one = S::Half::splat(1.0);
    let (prev_l, prev_r) = prev_sample.into_lr();
    let (curr_l, curr_r) = curr_sample.into_lr();
    let (mix_l, mix_r) = lin_mix.into_lr();
    let avg_l = prev_l.add(curr_l).mul(half);
    let avg_r = prev_r.add(curr_r).mul(half);
    let out_l = S::Half::select_le(one, mix_l, curr_l, avg_l);
    let out_r = S::Half::select_le(one, mix_r, curr_r, avg_r);
    S::from_lr(out_l, out_r)
}

/// Evaluates stateless `1x -> 2x` `63`-tap Kaiser `HB-FIR` upsampling for stereo `[L, R]`,
/// returning `[sub0, sub1]` (`16` FMAs/ch, `0` FMAs on `sub1 = hist[14]`).
#[inline(always)]
#[must_use]
pub fn eval_hb63_upsample_stereo(
    x_1x: [f32; 2],
    hist: &[[f32; 2]; HB63_HIST_LEN],
) -> [[f32; 2]; 2] {
    let t = HB63_ODD_TAPS_OUTER;
    let x_l = canonicalize_zero_f32_bits(x_1x[0]);
    let x_r = canonicalize_zero_f32_bits(x_1x[1]);
    let mut s0_l = t[0] * (x_l + hist[HB63_HIST_LEN - 1][0]);
    let mut s0_r = t[0] * (x_r + hist[HB63_HIST_LEN - 1][1]);
    let mut k = 1_usize;
    while k < HB63_ODD_TAP_COUNT {
        let a = &hist[k - 1];
        let b = &hist[HB63_HIST_LEN - 1 - k];
        s0_l += t[k] * (a[0] + b[0]);
        s0_r += t[k] * (a[1] + b[1]);
        k += 1;
    }
    [[s0_l, s0_r], hist[HB63_ODD_TAP_COUNT - 2]]
}

/// Evaluates stateless `2x -> 1x` `63`-tap Kaiser `HB-FIR` decimation across `4` channels
/// (`[pre_L, pre_R, dir_L, dir_R]`).
#[inline(always)]
#[must_use]
pub fn eval_hb63_downsample_4ch_vec4<V: KernelVec4>(
    sub0: V,
    even_hist: &[V; HB63_HIST_LEN],
    odd_delayed: V,
) -> V {
    let d = HB63_DN_ODD_TAPS_OUTER;
    let mut acc = odd_delayed.mul(V::splat(0.5));
    acc = acc.fma(V::splat(d[0]), sub0.add(even_hist[HB63_HIST_LEN - 1]));
    let mut k = 1_usize;
    while k < HB63_ODD_TAP_COUNT {
        let pair = even_hist[k - 1].add(even_hist[HB63_HIST_LEN - 1 - k]);
        acc = acc.fma(V::splat(d[k]), pair);
        k += 1;
    }
    acc
}

/// Evaluates stateless `2x -> 4x` `23`-tap Remez `HB-FIR` upsampling for stereo `[L, R]`,
/// returning `[sub0, sub1]`.
#[inline(always)]
#[must_use]
pub fn eval_hb23_upsample_stereo(
    x_2x: [f32; 2],
    hist: &[[f32; 2]; HB23_HIST_LEN],
) -> [[f32; 2]; 2] {
    let t = HB23_ODD_TAPS_OUTER;
    let x_l = canonicalize_zero_f32_bits(x_2x[0]);
    let x_r = canonicalize_zero_f32_bits(x_2x[1]);
    let s0_l = t[0] * (x_l + hist[10][0])
        + t[1] * (hist[0][0] + hist[9][0])
        + t[2] * (hist[1][0] + hist[8][0])
        + t[3] * (hist[2][0] + hist[7][0])
        + t[4] * (hist[3][0] + hist[6][0])
        + t[5] * (hist[4][0] + hist[5][0]);
    let s0_r = t[0] * (x_r + hist[10][1])
        + t[1] * (hist[0][1] + hist[9][1])
        + t[2] * (hist[1][1] + hist[8][1])
        + t[3] * (hist[2][1] + hist[7][1])
        + t[4] * (hist[3][1] + hist[6][1])
        + t[5] * (hist[4][1] + hist[5][1]);
    [[s0_l, s0_r], hist[HB23_ODD_TAP_COUNT - 2]]
}

/// Evaluates stateless `4x -> 2x` `23`-tap Remez `HB-FIR` decimation across `4` channels.
#[inline(always)]
#[must_use]
pub fn eval_hb23_downsample_4ch_vec4<V: KernelVec4>(
    sub0: V,
    even_hist: &[V; HB23_HIST_LEN],
    odd_delayed: V,
) -> V {
    let d = HB23_DN_ODD_TAPS_OUTER;
    let acc = odd_delayed.mul(V::splat(0.5));
    let acc = acc.fma(V::splat(d[0]), sub0.add(even_hist[10]));
    let acc = acc.fma(V::splat(d[1]), even_hist[0].add(even_hist[9]));
    let acc = acc.fma(V::splat(d[2]), even_hist[1].add(even_hist[8]));
    let acc = acc.fma(V::splat(d[3]), even_hist[2].add(even_hist[7]));
    let acc = acc.fma(V::splat(d[4]), even_hist[3].add(even_hist[6]));
    acc.fma(V::splat(d[5]), even_hist[4].add(even_hist[5]))
}

/// Zero-allocation `248 B` (`align(4)`) `1x -> 2x` stereo `63`-tap Kaiser `HB-FIR` upsampler.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HbFir63UpsamplerStereo {
    /// Past `31` `1x`-rate stereo frames `[x[n-1], ..., x[n-31]]`.
    pub hist: [[f32; 2]; HB63_HIST_LEN],
}

const _: () = assert!(core::mem::size_of::<HbFir63UpsamplerStereo>() == 248);
const _: () = assert!(core::mem::align_of::<HbFir63UpsamplerStereo>() == 4);

impl Default for HbFir63UpsamplerStereo {
    fn default() -> Self {
        Self::new()
    }
}

impl HbFir63UpsamplerStereo {
    /// Creates a zero-initialized [`HbFir63UpsamplerStereo`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hist: [[0.0; 2]; HB63_HIST_LEN],
        }
    }

    /// Resets all delay slots to canonical `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Steps one `1x` stereo frame `[L, R]` into two chronological `2x` stereo frames `[sub0, sub1]`.
    #[inline(always)]
    pub fn step(&mut self, x_1x: [f32; 2]) -> [[f32; 2]; 2] {
        let out = eval_hb63_upsample_stereo(x_1x, &self.hist);
        let mut idx = HB63_HIST_LEN - 1;
        while idx > 0 {
            self.hist[idx] = self.hist[idx - 1];
            idx -= 1;
        }
        self.hist[0] = [
            canonicalize_zero_f32_bits(x_1x[0]),
            canonicalize_zero_f32_bits(x_1x[1]),
        ];
        [
            [
                canonicalize_zero_f32_bits(out[0][0]),
                canonicalize_zero_f32_bits(out[0][1]),
            ],
            [
                canonicalize_zero_f32_bits(out[1][0]),
                canonicalize_zero_f32_bits(out[1][1]),
            ],
        ]
    }
}

/// Zero-allocation `752 B` (`align(4)`) `2x -> 1x` `4`-channel `63`-tap Kaiser `HB-FIR` downsampler.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HbFir63Downsampler4Ch {
    /// Past `31` even-phase `2x` frames `[sub0[n-1], ..., sub0[n-31]]`.
    pub even_hist: [[f32; 4]; HB63_HIST_LEN],
    /// Past `16` odd-phase `2x` frames `[sub1[n-1], ..., sub1[n-16]]`.
    pub odd_delay: [[f32; 4]; HB63_ODD_DELAY_LEN],
}

const _: () = assert!(core::mem::size_of::<HbFir63Downsampler4Ch>() == 752);
const _: () = assert!(core::mem::align_of::<HbFir63Downsampler4Ch>() == 4);

impl Default for HbFir63Downsampler4Ch {
    fn default() -> Self {
        Self::new()
    }
}

impl HbFir63Downsampler4Ch {
    /// Creates a zero-initialized [`HbFir63Downsampler4Ch`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            even_hist: [[0.0; 4]; HB63_HIST_LEN],
            odd_delay: [[0.0; 4]; HB63_ODD_DELAY_LEN],
        }
    }

    /// Resets all delay histories to canonical `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Decimates two chronological `2x` `KernelVec4` frames `(sub0, sub1)` into one `1x` `KernelVec4`.
    #[inline(always)]
    pub fn step_vec4<V: KernelVec4>(&mut self, sub0: V, sub1: V) -> V {
        let s0 = sub0.flush_recursive_state();
        let s1 = sub1.flush_recursive_state();
        let hist_vecs = core::array::from_fn(|i| V::load(&self.even_hist[i]));
        let odd_delayed = V::load(&self.odd_delay[HB63_ODD_DELAY_LEN - 1]);
        let out =
            eval_hb63_downsample_4ch_vec4(s0, &hist_vecs, odd_delayed).flush_recursive_state();

        let mut i = HB63_HIST_LEN - 1;
        while i > 0 {
            self.even_hist[i] = self.even_hist[i - 1];
            i -= 1;
        }
        self.even_hist[0] = s0.to_array();

        let mut j = HB63_ODD_DELAY_LEN - 1;
        while j > 0 {
            self.odd_delay[j] = self.odd_delay[j - 1];
            j -= 1;
        }
        self.odd_delay[0] = s1.to_array();
        out
    }

    /// Decimates two chronological `2x` `[f32; 4]` frames `(sub0, sub1)` into one `1x` `[f32; 4]`.
    #[inline(always)]
    pub fn step_arrays(&mut self, sub0: [f32; 4], sub1: [f32; 4]) -> [f32; 4] {
        use crate::simd::GpuVec4;
        self.step_vec4(GpuVec4::from_array(sub0), GpuVec4::from_array(sub1))
            .to_array()
    }
}

/// Zero-allocation `88 B` (`align(4)`) `2x -> 4x` stereo `23`-tap Remez `HB-FIR` upsampler.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HbFir23UpsamplerStereo {
    /// Past `11` `2x`-rate stereo frames `[x[n-1], ..., x[n-11]]`.
    pub hist: [[f32; 2]; HB23_HIST_LEN],
}

const _: () = assert!(core::mem::size_of::<HbFir23UpsamplerStereo>() == 88);
const _: () = assert!(core::mem::align_of::<HbFir23UpsamplerStereo>() == 4);

impl Default for HbFir23UpsamplerStereo {
    fn default() -> Self {
        Self::new()
    }
}

impl HbFir23UpsamplerStereo {
    /// Creates a zero-initialized [`HbFir23UpsamplerStereo`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hist: [[0.0; 2]; HB23_HIST_LEN],
        }
    }

    /// Resets all delay slots to canonical `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Steps one `2x` stereo frame `[L, R]` into two chronological `4x` stereo frames `[sub0, sub1]`.
    #[inline(always)]
    pub fn step(&mut self, x_2x: [f32; 2]) -> [[f32; 2]; 2] {
        let out = eval_hb23_upsample_stereo(x_2x, &self.hist);
        let mut idx = HB23_HIST_LEN - 1;
        while idx > 0 {
            self.hist[idx] = self.hist[idx - 1];
            idx -= 1;
        }
        self.hist[0] = [
            canonicalize_zero_f32_bits(x_2x[0]),
            canonicalize_zero_f32_bits(x_2x[1]),
        ];
        [
            [
                canonicalize_zero_f32_bits(out[0][0]),
                canonicalize_zero_f32_bits(out[0][1]),
            ],
            [
                canonicalize_zero_f32_bits(out[1][0]),
                canonicalize_zero_f32_bits(out[1][1]),
            ],
        ]
    }
}

/// Zero-allocation `272 B` (`align(4)`) `4x -> 2x` `4`-channel `23`-tap Remez `HB-FIR` downsampler.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HbFir23Downsampler4Ch {
    /// Past `11` even-phase `4x` frames `[sub0[n-1], ..., sub0[n-11]]`.
    pub even_hist: [[f32; 4]; HB23_HIST_LEN],
    /// Past `6` odd-phase `4x` frames `[sub1[n-1], ..., sub1[n-6]]`.
    pub odd_delay: [[f32; 4]; HB23_ODD_DELAY_LEN],
}

const _: () = assert!(core::mem::size_of::<HbFir23Downsampler4Ch>() == 272);
const _: () = assert!(core::mem::align_of::<HbFir23Downsampler4Ch>() == 4);

impl Default for HbFir23Downsampler4Ch {
    fn default() -> Self {
        Self::new()
    }
}

impl HbFir23Downsampler4Ch {
    /// Creates a zero-initialized [`HbFir23Downsampler4Ch`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            even_hist: [[0.0; 4]; HB23_HIST_LEN],
            odd_delay: [[0.0; 4]; HB23_ODD_DELAY_LEN],
        }
    }

    /// Resets all delay histories to canonical `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Decimates two chronological `4x` `KernelVec4` frames `(sub0, sub1)` into one `2x` `KernelVec4`.
    #[inline(always)]
    pub fn step_vec4<V: KernelVec4>(&mut self, sub0: V, sub1: V) -> V {
        let s0 = sub0.flush_recursive_state();
        let s1 = sub1.flush_recursive_state();
        let hist_vecs = core::array::from_fn(|i| V::load(&self.even_hist[i]));
        let odd_delayed = V::load(&self.odd_delay[HB23_ODD_DELAY_LEN - 1]);
        let out =
            eval_hb23_downsample_4ch_vec4(s0, &hist_vecs, odd_delayed).flush_recursive_state();

        let mut i = HB23_HIST_LEN - 1;
        while i > 0 {
            self.even_hist[i] = self.even_hist[i - 1];
            i -= 1;
        }
        self.even_hist[0] = s0.to_array();

        let mut j = HB23_ODD_DELAY_LEN - 1;
        while j > 0 {
            self.odd_delay[j] = self.odd_delay[j - 1];
            j -= 1;
        }
        self.odd_delay[0] = s1.to_array();
        out
    }

    /// Decimates two chronological `4x` `[f32; 4]` frames `(sub0, sub1)` into one `2x` `[f32; 4]`.
    #[inline(always)]
    pub fn step_arrays(&mut self, sub0: [f32; 4], sub1: [f32; 4]) -> [f32; 4] {
        use crate::simd::GpuVec4;
        self.step_vec4(GpuVec4::from_array(sub0), GpuVec4::from_array(sub1))
            .to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    #[test]
    fn hb63_hb23_and_master_4x_chain_exact_pdc_zero_dc_and_symmetry() {
        let mut up63 = HbFir63UpsamplerStereo::new();
        let mut dn63 = HbFir63Downsampler4Ch::new();
        let mut up23 = HbFir23UpsamplerStereo::new();
        let mut dn23 = HbFir23Downsampler4Ch::new();

        for &dirty in &[-0.0_f32, f32::from_bits(1), f32::NAN, f32::INFINITY] {
            let s2 = up63.step([dirty, dirty]);
            let s4a = up23.step(s2[0]);
            let s4b = up23.step(s2[1]);
            let y2a = dn23.step_arrays(
                [s4a[0][0], s4a[0][1], s4a[0][0], s4a[0][1]],
                [s4a[1][0], s4a[1][1], s4a[1][0], s4a[1][1]],
            );
            let y2b = dn23.step_arrays(
                [s4b[0][0], s4b[0][1], s4b[0][0], s4b[0][1]],
                [s4b[1][0], s4b[1][1], s4b[1][0], s4b[1][1]],
            );
            let y1 = dn63.step_arrays(y2a, y2b);
            assert_eq!(y1.map(f32::to_bits), [0_u32; 4]);
        }

        // HB63 standalone round-trip: peak at 31 (1x), symmetric, 0-ULP stereo.
        up63.reset();
        dn63.reset();
        let mut ir63 = [0.0_f32; 72];
        for (n, dst) in ir63.iter_mut().enumerate() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let s2 = up63.step([x, x]);
            let out = dn63.step_arrays(
                [s2[0][0], s2[0][1], s2[0][0], s2[0][1]],
                [s2[1][0], s2[1][1], s2[1][0], s2[1][1]],
            );
            assert_eq!(out[0].to_bits(), out[1].to_bits());
            assert_eq!(out[0].to_bits(), out[2].to_bits());
            assert_eq!(out[2].to_bits(), out[3].to_bits());
            *dst = out[0];
        }
        let peak63 = ir63
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak63, CORE_OS_MASTER_2X_LATENCY_1X);
        for d in 1..=31 {
            assert!(
                (ir63[31 - d] - ir63[31 + d]).abs() < 1.0e-6,
                "hb63 asym d={d}"
            );
        }

        // HB23 standalone round-trip: peak at 11 (2x), symmetric.
        up23.reset();
        dn23.reset();
        let mut ir23 = [0.0_f32; 32];
        for (n, dst) in ir23.iter_mut().enumerate() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let s4 = up23.step([x, x]);
            let out = dn23.step_arrays(
                [s4[0][0], s4[0][1], s4[0][0], s4[0][1]],
                [s4[1][0], s4[1][1], s4[1][0], s4[1][1]],
            );
            *dst = out[0];
        }
        let peak23 = ir23
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak23, HB23_DELAY_SAMPLES_4X);
        for d in 1..=11 {
            assert!(
                (ir23[11 - d] - ir23[11 + d]).abs() < 1.0e-6,
                "hb23 asym d={d}"
            );
        }

        // Full 4x Master chain (HB63 -> z^-1@2x -> HB23 -> HB23 -> HB63): exact peak at 37 (1x).
        up63.reset();
        dn63.reset();
        up23.reset();
        dn23.reset();
        let mut z_m1_2x = [0.0_f32; 2];
        let mut ir4x = [0.0_f32; 96];
        for (n, dst) in ir4x.iter_mut().enumerate() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let s2 = up63.step([x, x]);
            let d2_a = z_m1_2x;
            let d2_b = s2[0];
            z_m1_2x = s2[1];
            let s4a = up23.step(d2_a);
            let s4b = up23.step(d2_b);
            let y2a = dn23.step_arrays(
                [s4a[0][0], s4a[0][1], s4a[0][0], s4a[0][1]],
                [s4a[1][0], s4a[1][1], s4a[1][0], s4a[1][1]],
            );
            let y2b = dn23.step_arrays(
                [s4b[0][0], s4b[0][1], s4b[0][0], s4b[0][1]],
                [s4b[1][0], s4b[1][1], s4b[1][0], s4b[1][1]],
            );
            let out = dn63.step_arrays(y2a, y2b);
            *dst = out[0];
        }
        let peak4x = ir4x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak4x, CORE_OS_MASTER_4X_LATENCY_1X);
        for d in 1..=37 {
            assert!(
                (ir4x[37 - d] - ir4x[37 + d]).abs() < 2.0e-6,
                "master 4x asym d={d}"
            );
        }
        let sum4x: f64 = ir4x.iter().map(|&v| f64::from(v)).sum();
        assert!((sum4x - 1.0).abs() < 1.0e-5, "4x DC gain = {sum4x}");
    }

    #[test]
    fn box2_quadrature_averages_nonlin_and_bypasses_on_lin_mix_one() {
        assert_eq!(box2_quadrature_f32(1.0, 3.0, 0.25), 2.0);
        assert_eq!(box2_quadrature_f32(1.0, 3.0, 1.0), 3.0);
        assert_eq!(box2_quadrature_f32(-0.0, -0.0, 0.0).to_bits(), 0);

        let odd = InterleavedStereo8 {
            left: GpuVec4::from_array([1.0, 2.0, -4.0, 0.5]),
            right: GpuVec4::from_array([-1.0, 4.0, 2.0, 1.5]),
        };
        let even = InterleavedStereo8 {
            left: GpuVec4::from_array([3.0, 6.0, 2.0, 2.5]),
            right: GpuVec4::from_array([5.0, 8.0, -2.0, 3.5]),
        };
        let mix = InterleavedStereo8 {
            left: GpuVec4::from_array([0.0, 1.0, 0.5, 1.0]),
            right: GpuVec4::from_array([1.0, 0.25, 0.0, 1.0]),
        };
        let out = box2_quadrature_stereo8(odd, even, mix);
        assert_eq!(out.left.to_array(), [2.0, 6.0, -1.0, 2.5]);
        assert_eq!(out.right.to_array(), [5.0, 6.0, 0.0, 3.5]);
    }

    proptest! {
        #[test]
        fn prop_box2_stereo8_matches_scalar_per_lane(
            odd_l in prop::array::uniform4(-4.0_f32..4.0_f32),
            odd_r in prop::array::uniform4(-4.0_f32..4.0_f32),
            even_l in prop::array::uniform4(-4.0_f32..4.0_f32),
            even_r in prop::array::uniform4(-4.0_f32..4.0_f32),
            lin_mix in prop::array::uniform4(0.0_f32..=1.0_f32),
        ) {
            let odd = InterleavedStereo8 {
                left: GpuVec4::from_array(odd_l),
                right: GpuVec4::from_array(odd_r),
            };
            let even = InterleavedStereo8 {
                left: GpuVec4::from_array(even_l),
                right: GpuVec4::from_array(even_r),
            };
            let mix = InterleavedStereo8 {
                left: GpuVec4::from_array(lin_mix),
                right: GpuVec4::from_array(lin_mix),
            };
            let out = box2_quadrature_stereo8(odd, even, mix);
            for lane in 0..4 {
                let s_l = box2_quadrature_f32(odd_l[lane], even_l[lane], lin_mix[lane]);
                let s_r = box2_quadrature_f32(odd_r[lane], even_r[lane], lin_mix[lane]);
                prop_assert_eq!(out.left.to_array()[lane].to_bits(), s_l.to_bits());
                prop_assert_eq!(out.right.to_array()[lane].to_bits(), s_r.to_bits());
            }
        }
    }
}
