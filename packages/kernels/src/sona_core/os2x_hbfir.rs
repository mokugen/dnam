//! All-tier `2x` `15`-tap symmetric linear-phase half-band FIR (`HB-FIR`) `OS2X` upsampler and
//! `4`-channel downsampler (`4` FMAs/samp, exact `7` `1x`-sample pure integer delay, `0.000°` phase
//! dispersion, `ref[dsp.router.signal-flow]`, `ref[dsp.op-n.commuted.causality]`,
//! `ref[arch.simd.zero-stall-lowering]`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`,
//! and `spirv-unknown-naga-wgsl`-clean.

use crate::simd::{
    HB15_DN_ODD_TAPS_OUTER, HB15_ODD_TAPS_OUTER, KernelVec4, canonicalize_zero_f32_bits,
};

/// Exact `1x`-sample round-trip integer group delay of the `15`-tap `HB-FIR` `OS2X` pair (`7`).
pub const HB15_DELAY_SAMPLES_1X: usize = 7;

/// Past `1x`-frame history depth for symmetric `15`-tap polyphase even pairs (`7`).
pub const HB15_HIST_LEN: usize = 7;

/// Past `1x`-frame delay depth for the odd center-tap branch in `2x -> 1x` decimation (`4`).
pub const HB15_ODD_DELAY_LEN: usize = 4;

const _: () = assert!(HB15_DELAY_SAMPLES_1X == 7);
const _: () = assert!(HB15_HIST_LEN == 7 && HB15_ODD_DELAY_LEN == 4);

/// Evaluates stateless `1x -> 2x` `15`-tap `HB-FIR` polyphase upsampling for stereo `[L, R]`,
/// returning `[sub0, sub1]` (`4` FMAs/ch, `0` FMAs on `sub1 = hist[2]`).
#[inline(always)]
#[must_use]
pub fn eval_hb15_upsample_stereo(
    x_1x: [f32; 2],
    hist: &[[f32; 2]; HB15_HIST_LEN],
) -> [[f32; 2]; 2] {
    let t = HB15_ODD_TAPS_OUTER;
    let x_l = canonicalize_zero_f32_bits(x_1x[0]);
    let x_r = canonicalize_zero_f32_bits(x_1x[1]);
    let s0_l = t[0] * (x_l + hist[6][0])
        + t[1] * (hist[0][0] + hist[5][0])
        + t[2] * (hist[1][0] + hist[4][0])
        + t[3] * (hist[2][0] + hist[3][0]);
    let s0_r = t[0] * (x_r + hist[6][1])
        + t[1] * (hist[0][1] + hist[5][1])
        + t[2] * (hist[1][1] + hist[4][1])
        + t[3] * (hist[2][1] + hist[3][1]);
    [[s0_l, s0_r], hist[2]]
}

/// Evaluates stateless `2x -> 1x` `15`-tap `HB-FIR` decimation across `4` channels in a
/// [`KernelVec4`] register (`[pre_L, pre_R, dir_L, dir_R]`, `4` FMAs + `1` mul).
#[inline(always)]
#[must_use]
pub fn eval_hb15_downsample_4ch_vec4<V: KernelVec4>(
    sub0: V,
    even_hist: &[V; HB15_HIST_LEN],
    odd_delayed: V,
) -> V {
    let d = HB15_DN_ODD_TAPS_OUTER;
    let p0 = sub0.add(even_hist[6]);
    let p1 = even_hist[0].add(even_hist[5]);
    let p2 = even_hist[1].add(even_hist[4]);
    let p3 = even_hist[2].add(even_hist[3]);
    let acc = odd_delayed.mul(V::splat(0.5));
    let acc = acc.fma(V::splat(d[0]), p0);
    let acc = acc.fma(V::splat(d[1]), p1);
    let acc = acc.fma(V::splat(d[2]), p2);
    acc.fma(V::splat(d[3]), p3)
}

/// Evaluates stateless `2x -> 1x` `15`-tap `HB-FIR` decimation for a stereo `[L, R]` pair.
#[inline(always)]
#[must_use]
pub fn eval_hb15_downsample_stereo(
    sub0: [f32; 2],
    even_hist: &[[f32; 2]; HB15_HIST_LEN],
    odd_delayed: [f32; 2],
) -> [f32; 2] {
    let d = HB15_DN_ODD_TAPS_OUTER;
    let out_l = 0.5 * odd_delayed[0]
        + d[0] * (sub0[0] + even_hist[6][0])
        + d[1] * (even_hist[0][0] + even_hist[5][0])
        + d[2] * (even_hist[1][0] + even_hist[4][0])
        + d[3] * (even_hist[2][0] + even_hist[3][0]);
    let out_r = 0.5 * odd_delayed[1]
        + d[0] * (sub0[1] + even_hist[6][1])
        + d[1] * (even_hist[0][1] + even_hist[5][1])
        + d[2] * (even_hist[1][1] + even_hist[4][1])
        + d[3] * (even_hist[2][1] + even_hist[3][1]);
    [out_l, out_r]
}

/// Zero-allocation `56 B` (`align(4)`) `1x -> 2x` stereo `15`-tap linear-phase `HB-FIR` upsampler
/// (`ref[dsp.router.signal-flow]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HbFir15UpsamplerStereo {
    /// Past `7` `1x`-rate stereo frames `[x[n-1], ..., x[n-7]]`.
    pub hist: [[f32; 2]; HB15_HIST_LEN],
}

const _: () = assert!(core::mem::size_of::<HbFir15UpsamplerStereo>() == 56);
const _: () = assert!(core::mem::align_of::<HbFir15UpsamplerStereo>() == 4);

impl HbFir15UpsamplerStereo {
    /// Creates a zero-initialized [`HbFir15UpsamplerStereo`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hist: [[0.0; 2]; HB15_HIST_LEN],
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
        let out = eval_hb15_upsample_stereo(x_1x, &self.hist);
        let mut idx = HB15_HIST_LEN - 1;
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

/// Zero-allocation `176 B` (`align(4)`) `2x -> 1x` `4`-channel (`[pre_L, pre_R, dir_L, dir_R]`)
/// `15`-tap linear-phase `HB-FIR` downsampler (`ref[dsp.router.signal-flow]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HbFir15Downsampler4Ch {
    /// Past `7` even-phase `2x` frames `[sub0[n-1], ..., sub0[n-7]]`.
    pub even_hist: [[f32; 4]; HB15_HIST_LEN],
    /// Past `4` odd-phase `2x` frames `[sub1[n-1], ..., sub1[n-4]]`.
    pub odd_delay: [[f32; 4]; HB15_ODD_DELAY_LEN],
}

const _: () = assert!(core::mem::size_of::<HbFir15Downsampler4Ch>() == 176);
const _: () = assert!(core::mem::align_of::<HbFir15Downsampler4Ch>() == 4);

impl HbFir15Downsampler4Ch {
    /// Creates a zero-initialized [`HbFir15Downsampler4Ch`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            even_hist: [[0.0; 4]; HB15_HIST_LEN],
            odd_delay: [[0.0; 4]; HB15_ODD_DELAY_LEN],
        }
    }

    /// Resets all even and odd delay histories to canonical `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Decimates two chronological `2x` `KernelVec4` frames `(sub0, sub1)` into one `1x` `KernelVec4`.
    #[inline(always)]
    pub fn step_vec4<V: KernelVec4>(&mut self, sub0: V, sub1: V) -> V {
        let s0 = sub0.flush_recursive_state();
        let s1 = sub1.flush_recursive_state();
        let hist_vecs = [
            V::load(&self.even_hist[0]),
            V::load(&self.even_hist[1]),
            V::load(&self.even_hist[2]),
            V::load(&self.even_hist[3]),
            V::load(&self.even_hist[4]),
            V::load(&self.even_hist[5]),
            V::load(&self.even_hist[6]),
        ];
        let odd_delayed = V::load(&self.odd_delay[HB15_ODD_DELAY_LEN - 1]);
        let out =
            eval_hb15_downsample_4ch_vec4(s0, &hist_vecs, odd_delayed).flush_recursive_state();

        let mut i = HB15_HIST_LEN - 1;
        while i > 0 {
            self.even_hist[i] = self.even_hist[i - 1];
            i -= 1;
        }
        self.even_hist[0] = s0.to_array();

        let mut j = HB15_ODD_DELAY_LEN - 1;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, HB15_ODD_TAPS_OUTER};

    fn dft_hb15_f64(omega_rad: f64) -> f64 {
        let h7 = f64::from(HB15_ODD_TAPS_OUTER[0]);
        let h5 = f64::from(HB15_ODD_TAPS_OUTER[1]);
        let h3 = f64::from(HB15_ODD_TAPS_OUTER[2]);
        let h1 = f64::from(HB15_ODD_TAPS_OUTER[3]);
        0.5 + h1 * omega_rad.cos()
            + h3 * (3.0 * omega_rad).cos()
            + h5 * (5.0 * omega_rad).cos()
            + h7 * (7.0 * omega_rad).cos()
    }

    /// verify[dsp.router.signal-flow]
    /// verify[dsp.op-n.commuted.causality]
    /// verify[arch.simd.zero-stall-lowering]
    #[test]
    fn test_hb15_delay_symmetry_zero_dc_stereo_parity_and_dft_oracle() {
        let mut up = HbFir15UpsamplerStereo::new();
        let mut dn = HbFir15Downsampler4Ch::new();

        // 1. Exact Zero-DC +0.0 across -0.0, subnormals, NaN, and +-Inf.
        for &dirty in &[-0.0_f32, f32::from_bits(1), f32::NAN, f32::INFINITY] {
            let up_out = up.step([dirty, dirty]);
            let dn_out = dn.step_arrays(
                [up_out[0][0], up_out[0][1], up_out[0][0], up_out[0][1]],
                [up_out[1][0], up_out[1][1], up_out[1][0], up_out[1][1]],
            );
            assert_eq!(dn_out.map(f32::to_bits), [0_u32; 4]);
        }

        // 2. Impulse response: exact peak at index 7 (1x), symmetric IR (0.000 deg dispersion), 0-ULP stereo.
        up.reset();
        dn.reset();
        let mut ir = [0.0_f32; 20];
        for n in 0..ir.len() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let up_out = up.step([x, x]);
            let s0 = GpuVec4::from_array([up_out[0][0], up_out[0][1], up_out[0][0], up_out[0][1]]);
            let s1 = GpuVec4::from_array([up_out[1][0], up_out[1][1], up_out[1][0], up_out[1][1]]);
            let dn_out = dn.step_vec4(s0, s1).to_array();
            assert_eq!(dn_out[0].to_bits(), dn_out[1].to_bits());
            assert_eq!(dn_out[0].to_bits(), dn_out[2].to_bits());
            assert_eq!(dn_out[2].to_bits(), dn_out[3].to_bits());
            ir[n] = dn_out[0];
        }
        let peak_idx = ir
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak_idx, HB15_DELAY_SAMPLES_1X);
        for d in 1..=7 {
            assert!((ir[7 - d] - ir[7 + d]).abs() < 1.0e-7, "asym at d={d}");
        }
        let ir_sum: f64 = ir.iter().map(|&v| f64::from(v)).sum();
        assert!((ir_sum - 1.0).abs() < 1.0e-6);

        // Stateless stereo downsampler parity vs 4-ch Vec4 downsampler.
        let mut even_st = [[0.0_f32; 2]; HB15_HIST_LEN];
        even_st[0] = [0.25, -0.5];
        even_st[4] = [0.125, 0.75];
        let st_out = eval_hb15_downsample_stereo([0.3, -0.4], &even_st, [0.6, -0.2]);
        let v_hist = even_st.map(|p| GpuVec4::from_array([p[0], p[1], 0.0, 0.0]));
        let v_out = eval_hb15_downsample_4ch_vec4(
            GpuVec4::from_array([0.3, -0.4, 0.0, 0.0]),
            &v_hist,
            GpuVec4::from_array([0.6, -0.2, 0.0, 0.0]),
        )
        .to_array();
        assert!((st_out[0] - v_out[0]).abs() < 1.0e-7);
        assert!((st_out[1] - v_out[1]).abs() < 1.0e-7);

        // 3. f64 DFT passband & stopband oracle.
        for k in 0..=173 {
            let fk = (k as f64) * 0.1;
            let w = core::f64::consts::TAU * fk / 96.0;
            let mag_db = 20.0 * dft_hb15_f64(w).abs().log10();
            assert!(mag_db.abs() <= 0.08, "passband fk={fk}: {mag_db} dB");
        }
        for k in 300..=480 {
            let fk = (k as f64) * 0.1;
            let w = core::f64::consts::TAU * fk / 96.0;
            let mag_db = 20.0 * dft_hb15_f64(w).abs().max(1.0e-12).log10();
            assert!(mag_db <= -31.9, "stopband 30k+ fk={fk}: {mag_db} dB");
            if fk >= 32.0 {
                assert!(mag_db <= -41.2, "stopband 32k+ fk={fk}: {mag_db} dB");
            }
        }
    }

    /// verify[dsp.router.signal-flow]
    #[test]
    fn test_hb15_vs_4x_bounce_lti_bandlimited_parity() {
        // Compare 2x HB15 roundtrip vs 4x dyadic HB_2X_TAPS roundtrip on bandlimited guitar tones.
        // Delay = 7 1x-samples for HB15 vs 9 1x-samples for HB_2X (19-tap).
        const HB2X_TAPS: [f64; 5] = [
            35.0 / 65_536.0,
            -405.0 / 65_536.0,
            2_268.0 / 65_536.0,
            -8_820.0 / 65_536.0,
            39_690.0 / 65_536.0,
        ];
        let mut up = HbFir15UpsamplerStereo::new();
        let mut dn = HbFir15Downsampler4Ch::new();
        let mut x_hist = [0.0_f64; 10];
        let mut e2_hist = [0.0_f64; 10];
        let mut o2_delay = [0.0_f64; 6];

        let mut err2 = 0.0_f64;
        let mut ref2 = 0.0_f64;
        let mut out_hb15 = [0.0_f64; 512];
        let mut out_hb2x = [0.0_f64; 512];

        for n in 0..512 {
            let t = (n as f64) / 48_000.0;
            // Guitar / audio multitone up to 5.274 kHz
            let x = 0.35 * (core::f64::consts::TAU * 82.41 * t).sin()
                + 0.25 * (core::f64::consts::TAU * 440.0 * t).sin()
                + 0.2 * (core::f64::consts::TAU * 2_637.0 * t).sin()
                + 0.15 * (core::f64::consts::TAU * 5_274.0 * t).sin();
            let xf = x as f32;
            let u = up.step([xf, xf]);
            let d = dn.step_arrays([u[0][0], u[0][1], 0.0, 0.0], [u[1][0], u[1][1], 0.0, 0.0]);
            out_hb15[n] = f64::from(d[0]);

            // 19-tap HB2X roundtrip reference
            for k in (1..10).rev() {
                x_hist[k] = x_hist[k - 1];
            }
            x_hist[0] = x;
            let mut sub0 = 0.0_f64;
            for m in 0..5 {
                sub0 += HB2X_TAPS[m] * (x_hist[m] + x_hist[9 - m]);
            }
            let sub1 = x_hist[4];
            for k in (1..10).rev() {
                e2_hist[k] = e2_hist[k - 1];
            }
            e2_hist[0] = sub0;
            for k in (1..6).rev() {
                o2_delay[k] = o2_delay[k - 1];
            }
            o2_delay[0] = sub1;
            let mut filt = 0.0_f64;
            for m in 0..5 {
                filt += HB2X_TAPS[m] * (e2_hist[m] + e2_hist[9 - m]);
            }
            out_hb2x[n] = 0.5 * (filt + o2_delay[5]);
        }

        // Align 7-samp HB15 delay against 9-samp HB2X delay in steady state.
        for n in 64..500 {
            let a = out_hb15[n];
            let b = out_hb2x[n + 2];
            err2 += (a - b).powi(2);
            ref2 += b.powi(2);
        }
        let parity_db = 10.0 * (err2 / ref2).log10();
        assert!(
            parity_db <= -42.0,
            "LTI parity vs 4x bounce = {parity_db} dB"
        );
    }
}
