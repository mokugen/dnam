//! Offline `4x` linear-phase dyadic-rational polyphase stereo oversampler (`Bounce4xOversampler`)
//! with exact `OFFLINE_BOUNCE_PDC_SAMPLES = 12` host-sample group delay (`0 FDIV`, `!alloc`,
//! `ref[dsp.router.signal-flow]`, `router_and_crossbar_topology.md §5`).

#![forbid(unsafe_code)]

use crate::math::flush_recursive_state_f32;

/// Exact linear-phase Plugin Delay Compensation (`PDC`) in host samples during `4x` offline bounce.
pub const OFFLINE_BOUNCE_PDC_SAMPLES: usize = 12;

/// Stage 1 (`1x <-> 2x`) 19-tap symmetric half-band odd-branch dyadic taps (`K_1 = 5`, sum = `0.5`).
pub const HB_2X_TAPS: [f32; 5] = [
    35.0 / 65_536.0,
    -405.0 / 65_536.0,
    2_268.0 / 65_536.0,
    -8_820.0 / 65_536.0,
    39_690.0 / 65_536.0,
];

/// Stage 2 (`2x <-> 4x`) 11-tap symmetric half-band odd-branch dyadic taps (`K_2 = 3`, sum = `0.5`).
pub const HB_4X_TAPS: [f32; 3] = [3.0 / 256.0, -25.0 / 256.0, 150.0 / 256.0];

const fn sum_taps_f64(taps: &[f32]) -> f64 {
    let mut acc = 0.0_f64;
    let mut i = 0;
    while i < taps.len() {
        acc += taps[i] as f64;
        i += 1;
    }
    acc
}

const _: () = assert!(OFFLINE_BOUNCE_PDC_SAMPLES == 12);
const _: () = assert!(sum_taps_f64(&HB_2X_TAPS) == 0.5);
const _: () = assert!(sum_taps_f64(&HB_4X_TAPS) == 0.5);

#[inline(always)]
fn push_shift_stereo(hist: &mut [[f32; 2]], sample: [f32; 2]) {
    for k in (1..hist.len()).rev() {
        hist[k] = hist[k - 1];
    }
    hist[0] = [
        flush_recursive_state_f32(sample[0]),
        flush_recursive_state_f32(sample[1]),
    ];
}

#[inline(always)]
fn eval_symmetric_taps(hist: &[[f32; 2]], taps: &[f32]) -> [f32; 2] {
    let last = hist.len() - 1;
    let mut acc = [0.0_f32; 2];
    for (k, &h) in taps.iter().enumerate() {
        let a = hist[k];
        let b = hist[last - k];
        acc[0] += h * (a[0] + b[0]);
        acc[1] += h * (a[1] + b[1]);
    }
    acc
}

/// Zero-allocation 2-stage `4x` linear-phase polyphase stereo oversampler (`1x -> 4x` & `4x -> 1x`).
///
/// Cascades the 19-tap `1x <-> 2x` (`K_1 = 5`) and 11-tap `2x <-> 4x` (`K_2 = 3`) symmetric
/// dyadic half-band FIR stages with a 1-sample `2x` phase-alignment delay (`stage2_align_z1`)
/// to achieve an exact integer `12`-host-sample group delay (`OFFLINE_BOUNCE_PDC_SAMPLES`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounce4xOversampler {
    up1_hist: [[f32; 2]; 10],
    up2_hist: [[f32; 2]; 6],
    dn2_even_hist: [[f32; 2]; 6],
    dn2_odd_delay: [[f32; 2]; 4],
    stage2_align_z1: [f32; 2],
    dn1_even_hist: [[f32; 2]; 10],
    dn1_odd_delay: [[f32; 2]; 6],
}

impl Bounce4xOversampler {
    /// Creates a zeroed [`Bounce4xOversampler`] at resting `+0.0` equilibrium.
    #[inline]
    pub const fn new() -> Self {
        Self {
            up1_hist: [[0.0; 2]; 10],
            up2_hist: [[0.0; 2]; 6],
            dn2_even_hist: [[0.0; 2]; 6],
            dn2_odd_delay: [[0.0; 2]; 4],
            stage2_align_z1: [0.0; 2],
            dn1_even_hist: [[0.0; 2]; 10],
            dn1_odd_delay: [[0.0; 2]; 6],
        }
    }

    /// Resets all FIR delay lines to `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the exact host-sample linear-phase PDC (`12`).
    #[inline]
    pub const fn pdc_host_samples() -> usize {
        OFFLINE_BOUNCE_PDC_SAMPLES
    }

    /// Upsamples one `1x` stereo frame `(in_l, in_r)` into four chronological `4x` stereo frames.
    #[inline(always)]
    pub fn upsample_4x(&mut self, in_l: f32, in_r: f32) -> [[f32; 2]; 4] {
        push_shift_stereo(&mut self.up1_hist, [in_l, in_r]);
        let s2x_0 = eval_symmetric_taps(&self.up1_hist, &HB_2X_TAPS);
        let s2x_1 = self.up1_hist[4];
        let s2x = [s2x_0, s2x_1];

        let mut out_4x = [[0.0_f32; 2]; 4];
        for (m, &u2) in s2x.iter().enumerate() {
            push_shift_stereo(&mut self.up2_hist, u2);
            let s4x_0 = eval_symmetric_taps(&self.up2_hist, &HB_4X_TAPS);
            let s4x_1 = self.up2_hist[2];
            out_4x[m * 2] = s4x_0;
            out_4x[m * 2 + 1] = s4x_1;
        }
        out_4x
    }

    /// Decimates four chronological `4x` stereo frames `w4x` back into one `1x` stereo frame.
    #[inline(always)]
    pub fn downsample_4x(&mut self, w4x: [[f32; 2]; 4]) -> [f32; 2] {
        let mut v2x = [[0.0_f32; 2]; 2];
        for (m, slot) in v2x.iter_mut().enumerate() {
            push_shift_stereo(&mut self.dn2_even_hist, w4x[m * 2]);
            push_shift_stereo(&mut self.dn2_odd_delay, w4x[m * 2 + 1]);
            let filt2 = eval_symmetric_taps(&self.dn2_even_hist, &HB_4X_TAPS);
            let odd2 = self.dn2_odd_delay[3];
            let raw2 = [
                flush_recursive_state_f32(0.5 * (filt2[0] + odd2[0])),
                flush_recursive_state_f32(0.5 * (filt2[1] + odd2[1])),
            ];
            *slot = self.stage2_align_z1;
            self.stage2_align_z1 = raw2;
        }

        push_shift_stereo(&mut self.dn1_even_hist, v2x[0]);
        push_shift_stereo(&mut self.dn1_odd_delay, v2x[1]);
        let filt1 = eval_symmetric_taps(&self.dn1_even_hist, &HB_2X_TAPS);
        let odd1 = self.dn1_odd_delay[5];
        [
            flush_recursive_state_f32(0.5 * (filt1[0] + odd1[0])),
            flush_recursive_state_f32(0.5 * (filt1[1] + odd1[1])),
        ]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::const_sin_cos_f64;

    #[test]
    fn test_bounce4x_pdc_symmetry_dc_unity_and_zero_rest() {
        let mut os = Bounce4xOversampler::new();
        assert_eq!(Bounce4xOversampler::pdc_host_samples(), 12);

        let up_zero = os.upsample_4x(0.0, 0.0);
        for frame in up_zero {
            assert_eq!(frame.map(f32::to_bits), [0, 0]);
        }
        let dn_zero = os.downsample_4x(up_zero);
        assert_eq!(dn_zero.map(f32::to_bits), [0, 0]);

        let mut ir = [0.0_f32; 32];
        for (i, slot) in ir.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let up = os.upsample_4x(x, -x);
            let dn = os.downsample_4x(up);
            assert!((dn[0] + dn[1]).abs() < 1.0e-7);
            *slot = dn[0];
        }
        let peak_idx = ir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(peak_idx, OFFLINE_BOUNCE_PDC_SAMPLES);
        for k in 1..=12 {
            assert!(
                (ir[12 - k] - ir[12 + k]).abs() < 1.0e-6,
                "symmetry break at offset {k}: {} vs {}",
                ir[12 - k],
                ir[12 + k]
            );
        }

        os.reset();
        let mut dc_last = [0.0_f32; 2];
        for _ in 0..64 {
            let up = os.upsample_4x(0.75, -0.25);
            dc_last = os.downsample_4x(up);
        }
        assert!((dc_last[0] - 0.75).abs() < 1.0e-6);
        assert!((dc_last[1] + 0.25).abs() < 1.0e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_bounce4x_low_freq_sine_passband_and_dc_gain(
            dc in -2.0_f32..=2.0_f32,
            amp in 0.0_f32..=1.0_f32,
        ) {
            const OMEGA: f64 = core::f64::consts::TAU * 440.0 / 48_000.0;
            let mut os = Bounce4xOversampler::new();
            let mut out = [0.0_f32; 96];
            for (n, slot) in out.iter_mut().enumerate() {
                let s = dc + amp * (const_sin_cos_f64((n as f64) * OMEGA).0 as f32);
                let up = os.upsample_4x(s, s);
                let dn = os.downsample_4x(up);
                prop_assert!(dn[0].is_finite());
                *slot = dn[0];
            }
            for n in 36..80 {
                let phase = ((n - OFFLINE_BOUNCE_PDC_SAMPLES) as f64) * OMEGA;
                let expected = dc + amp * (const_sin_cos_f64(phase).0 as f32);
                prop_assert!((out[n] - expected).abs() < 2.0e-3);
            }
        }
    }
}
