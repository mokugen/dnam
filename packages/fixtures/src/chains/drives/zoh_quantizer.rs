//! `Chain-18`: ZOH Sample-Rate Reducer & Bitcrusher (`2 -> 2`, Post-DAC TPT-SVF Reconstruction).

#![forbid(unsafe_code)]

use crate::chains::{const_phys_to_norm, flush_subnormal_f64};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
    sanitize_sample_f64,
};
use core::f64::consts::{PI, SQRT_2, TAU};

const SR_ZOH_DEFAULT_NORM: f32 = const_phys_to_norm(6_000.0, 500.0, 24_000.0);
const BITS_DEFAULT_NORM: f32 = const_phys_to_norm(6.0, 2.0, 16.0);
const CUTOFF_DEFAULT_NORM: f32 = const_phys_to_norm(8_000.0, 1_200.0, 18_000.0);
const MIX_DEFAULT_NORM: f32 = const_phys_to_norm(80.0, 0.0, 100.0);

const _: () = assert!(BITS_DEFAULT_NORM < 0.0 && MIX_DEFAULT_NORM > 0.59);

const CHAIN18_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-18-ZOH-Quantizer-Bitcrusher",
    name: "ZOH Sample-Rate Reducer & Bitcrusher",
    category: "bitcrusher",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 10.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "sample_rate_zoh",
            name: "ZOH Clock Rate",
            unit: "Hz",
            default_norm: SR_ZOH_DEFAULT_NORM,
            min_physical: 500.0,
            max_physical: 24_000.0,
        },
        FixtureControlSpec {
            id: "bit_depth_bits",
            name: "Quantizer Resolution",
            unit: "bits",
            default_norm: BITS_DEFAULT_NORM,
            min_physical: 2.0,
            max_physical: 16.0,
        },
        FixtureControlSpec {
            id: "filter_cutoff_hz",
            name: "DAC Reconstruction Cutoff",
            unit: "Hz",
            default_norm: CUTOFF_DEFAULT_NORM,
            min_physical: 1_200.0,
            max_physical: 18_000.0,
        },
        FixtureControlSpec {
            id: "mix",
            name: "Mix",
            unit: "%",
            default_norm: MIX_DEFAULT_NORM,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "quant_mode",
        name: "Quantizer Curve",
        options: &["Mid-Tread ZOH", "Soft-Step Crush"],
        default_index: 0,
    }],
};

#[inline]
fn quantize_level(v: f64, levels_half: f64, soft_crush: bool) -> f64 {
    let clamped = sanitize_sample_f64(v).clamp(-8.0, 8.0);
    if soft_crush {
        let q = (clamped * levels_half).round() / levels_half;
        let blend = clamped - 0.85 * (TAU * levels_half * clamped).sin() / (TAU * levels_half);
        flush_subnormal_f64((0.5 * q + 0.5 * blend).tanh())
    } else {
        let q = (clamped * levels_half).round() / levels_half;
        flush_subnormal_f64(q)
    }
}

/// `Chain-18-ZOH-Quantizer-Bitcrusher`: `2 -> 2` independent stereo fixed-Hz ZOH sample-rate
/// reducer + variable-depth staircase quantizer + post-DAC 2-pole TPT-SVF analog reconstruction filter.
#[derive(Clone, Debug)]
pub struct Chain18ZohQuantizerBitcrusher {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    phase: [f64; 2],
    held: [f64; 2],
    svf_s1: [f64; 2],
    svf_s2: [f64; 2],
}

impl Default for Chain18ZohQuantizerBitcrusher {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain18ZohQuantizerBitcrusher {
    /// Creates a `Chain-18` ZOH quantizer/bitcrusher fixture at `48 kHz` / `Float32`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    /// Creates a `Chain-18` fixture configured for `(sample_rate_hz, bit_depth)`.
    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        Self {
            sample_rate_hz: clamp_sample_rate_hz(sample_rate_hz),
            bit_depth,
            phase: [1.0, 1.0],
            held: [0.0, 0.0],
            svf_s1: [0.0, 0.0],
            svf_s2: [0.0, 0.0],
        }
    }

    /// Processes a single stereo sample pair at the configured device sample rate and bit depth.
    #[inline]
    pub fn step_stereo(
        &mut self,
        in_l: f32,
        in_r: f32,
        controls: &[f32],
        switches: &[u8],
    ) -> [f32; 2] {
        let xl = self.bit_depth.quantize_sample(in_l);
        let xr = self.bit_depth.quantize_sample(in_r);

        let sr_u = f64::from(sanitize_norm(
            controls.first().copied().unwrap_or(SR_ZOH_DEFAULT_NORM),
            SR_ZOH_DEFAULT_NORM,
        ));
        let bits_u = f64::from(sanitize_norm(
            controls.get(1).copied().unwrap_or(BITS_DEFAULT_NORM),
            BITS_DEFAULT_NORM,
        ));
        let fc_u = f64::from(sanitize_norm(
            controls.get(2).copied().unwrap_or(CUTOFF_DEFAULT_NORM),
            CUTOFF_DEFAULT_NORM,
        ));
        let mix_u = f64::from(sanitize_norm(
            controls.get(3).copied().unwrap_or(MIX_DEFAULT_NORM),
            MIX_DEFAULT_NORM,
        ));
        let soft_crush = switches.first().copied().unwrap_or(0) != 0;

        let fs = f64::from(self.sample_rate_hz);
        let zoh_hz = (12_250.0 + 11_750.0 * sr_u).clamp(500.0, fs);
        let inc = (zoh_hz / fs).clamp(1.0e-5, 1.0);
        let bits = 9.0 + 7.0 * bits_u;
        let levels_half = (2.0_f64).powf(bits - 1.0);
        let fc_hz = (9_600.0 + 8_400.0 * fc_u).clamp(400.0, 0.45 * fs);
        let g = (PI * fc_hz / fs).tan();
        let a1 = 1.0 / (1.0 + g * (g + SQRT_2));
        let a2 = g * a1;
        let a3 = g * a2;
        let wet_mix = 0.5 * (mix_u + 1.0);

        let inputs = [f64::from(xl), f64::from(xr)];
        let mut out = [0.0_f64; 2];
        for ch in 0..2 {
            let x = inputs[ch];
            let p_next = self.phase[ch] + inc;
            if p_next >= 1.0 {
                self.phase[ch] = p_next.fract();
                self.held[ch] = quantize_level(x, levels_half, soft_crush);
            } else {
                self.phase[ch] = p_next;
            }
            let h = self.held[ch];
            let v3 = h - self.svf_s2[ch];
            let v1 = a1 * self.svf_s1[ch] + a2 * v3;
            let v2 = self.svf_s2[ch] + a2 * self.svf_s1[ch] + a3 * v3;
            self.svf_s1[ch] = flush_subnormal_f64(2.0 * v1 - self.svf_s1[ch]);
            self.svf_s2[ch] = flush_subnormal_f64(2.0 * v2 - self.svf_s2[ch]);
            let wet = flush_subnormal_f64(v2);
            out[ch] = flush_subnormal_f64((1.0 - wet_mix) * x + wet_mix * wet);
        }

        [
            self.bit_depth.quantize_f64_to_f32(out[0]),
            self.bit_depth.quantize_f64_to_f32(out[1]),
        ]
    }
}

impl BlackBoxModel for Chain18ZohQuantizerBitcrusher {
    fn reset(&mut self) {
        self.phase = [1.0, 1.0];
        self.held = [0.0, 0.0];
        self.svf_s1 = [0.0, 0.0];
        self.svf_s2 = [0.0, 0.0];
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN18_METADATA
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = clamp_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.bit_depth = bit_depth;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let xl = inputs.first().copied().unwrap_or(0.0);
        let xr = inputs.get(1).copied().unwrap_or(xl);
        let [yl, yr] = self.step_stereo(xl, xr, controls, switches);
        crate::chains::write_stereo_output(outputs, yl, yr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{exercise_blackbox_metadata_and_format, run_stereo_block, sine_wave};

    #[test]
    fn test_chain18_zoh_quantizer_distortion_isolation_and_bypass() {
        let mut c18 = Chain18ZohQuantizerBitcrusher::default();
        assert_eq!(c18.metadata().id, "Chain-18-ZOH-Quantizer-Bitcrusher");
        exercise_blackbox_metadata_and_format(&mut c18);

        let dry_l = sine_wave(1_024, 330.0, 0.55);
        let dry_r = sine_wave(1_024, 510.0, 0.45);
        let (bypass_l, bypass_r) =
            run_stereo_block(&mut c18, &dry_l, &dry_r, &[0.0, 0.0, 0.0, -1.0], &[0]);
        assert_eq!(bypass_l, dry_l);
        assert_eq!(bypass_r, dry_r);

        // Coarser ZOH clock + lower bit depth raises quantization/aliasing error over 16-bit full-rate.
        let mut mse = |sr_u: f32, bits_u: f32, mode: u8| -> f64 {
            c18.reset();
            let (yl, _) =
                run_stereo_block(&mut c18, &dry_l, &dry_r, &[sr_u, bits_u, 1.0, 1.0], &[mode]);
            yl.iter()
                .zip(&dry_l)
                .map(|(&y, &d)| f64::from(y - d).powi(2))
                .sum::<f64>()
        };
        let err_fine = mse(1.0, 1.0, 0);
        let err_crushed = mse(-0.8, -0.8, 0);
        let err_soft = mse(-0.8, -0.8, 1);
        assert!(err_crushed > err_fine * 4.0);
        assert!((err_soft - err_crushed).abs() > 1.0e-4);

        // Independent L/R zero crosstalk check.
        c18.reset();
        let (iso_l, iso_r) = run_stereo_block(
            &mut c18,
            &dry_l,
            &vec![0.0_f32; 1_024],
            &[0.0, 0.0, 0.0, 1.0],
            &[0],
        );
        assert!(iso_l.iter().any(|&v| v.abs() > 0.1));
        assert!(iso_r.iter().all(|&v| v == 0.0));
    }
}
