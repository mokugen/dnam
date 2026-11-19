//! `Chain-14`: Darkglass Entropia-Style Tri-Band Sub-Compressed & Mid/High Asymmetric Distortion.

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, TptLr4Crossover2, const_phys_to_norm, flush_subnormal_f64,
    scale_alpha_from_192k, substep_rate_4x,
};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz,
};

const CHAIN14_LOW_LPF_MIN_HZ: f32 = 60.0;
const CHAIN14_LOW_LPF_MAX_HZ: f32 = 600.0;
const CHAIN14_LOW_LPF_DEF_HZ: f32 = 250.0;
const CHAIN14_MID_LPF_MIN_HZ: f32 = 400.0;
const CHAIN14_MID_LPF_MAX_HZ: f32 = 3_600.0;
const CHAIN14_MID_LPF_DEF_HZ: f32 = 1_200.0;
const CHAIN14_HIGH_LPF_MIN_HZ: f32 = 2_000.0;
const CHAIN14_HIGH_LPF_MAX_HZ: f32 = 16_000.0;
const CHAIN14_HIGH_LPF_DEF_HZ: f32 = 8_000.0;
const CHAIN14_MIN_BAND_RATIO: f64 = 1.25;

/// `(drive_multiplier, asym_bias, voicing_tilt_mix)` per mid voicing option.
const CHAIN14_MID_VOICINGS: [(f64, f64, f64); 4] = [
    (1.00, 0.10, 0.12),  // VMT (Warm Tube)
    (1.35, 0.06, -0.18), // B3K (CMOS Grunt)
    (1.20, 0.18, 0.28),  // Alpha (Tight Bite)
    (1.65, 0.14, 0.48),  // Omega (Heavy Scoop)
];

/// `(drive_multiplier, asym_bias, post_emphasis_mix)` per high voicing option.
const CHAIN14_HIGH_VOICINGS: [(f64, f64, f64); 4] = [
    (1.30, 0.06, 0.00),  // B3K Turbo (Crisp CMOS)
    (0.90, 0.12, 0.22),  // VMT (Smooth Edge)
    (1.50, 0.16, -0.25), // Alpha Turbo (2.8k Bite)
    (1.85, 0.02, 0.35),  // Chinchilla (Hard Clip)
];

const CHAIN14_CONTROLS: [FixtureControlSpec; 11] = [
    FixtureControlSpec {
        id: "low_lpf",
        name: "Low Crossover (LPF)",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN14_LOW_LPF_DEF_HZ,
            CHAIN14_LOW_LPF_MIN_HZ,
            CHAIN14_LOW_LPF_MAX_HZ,
        ),
        min_physical: CHAIN14_LOW_LPF_MIN_HZ,
        max_physical: CHAIN14_LOW_LPF_MAX_HZ,
    },
    FixtureControlSpec {
        id: "mid_hpf",
        name: "Mid Low-Cut (HPF)",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN14_LOW_LPF_DEF_HZ,
            CHAIN14_LOW_LPF_MIN_HZ,
            CHAIN14_LOW_LPF_MAX_HZ,
        ),
        min_physical: CHAIN14_LOW_LPF_MIN_HZ,
        max_physical: CHAIN14_LOW_LPF_MAX_HZ,
    },
    FixtureControlSpec {
        id: "mid_lpf",
        name: "Mid High-Cut (LPF)",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN14_MID_LPF_DEF_HZ,
            CHAIN14_MID_LPF_MIN_HZ,
            CHAIN14_MID_LPF_MAX_HZ,
        ),
        min_physical: CHAIN14_MID_LPF_MIN_HZ,
        max_physical: CHAIN14_MID_LPF_MAX_HZ,
    },
    FixtureControlSpec {
        id: "high_hpf",
        name: "High Crossover (HPF)",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN14_MID_LPF_DEF_HZ,
            CHAIN14_MID_LPF_MIN_HZ,
            CHAIN14_MID_LPF_MAX_HZ,
        ),
        min_physical: CHAIN14_MID_LPF_MIN_HZ,
        max_physical: CHAIN14_MID_LPF_MAX_HZ,
    },
    FixtureControlSpec {
        id: "high_lpf",
        name: "High Cut (LPF)",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN14_HIGH_LPF_DEF_HZ,
            CHAIN14_HIGH_LPF_MIN_HZ,
            CHAIN14_HIGH_LPF_MAX_HZ,
        ),
        min_physical: CHAIN14_HIGH_LPF_MIN_HZ,
        max_physical: CHAIN14_HIGH_LPF_MAX_HZ,
    },
    FixtureControlSpec {
        id: "low_comp",
        name: "Low Comp",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "low_level",
        name: "Low Level",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
    FixtureControlSpec {
        id: "mid_drive",
        name: "Mid Drive",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "mid_level",
        name: "Mid Level",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
    FixtureControlSpec {
        id: "high_drive",
        name: "High Drive",
        unit: "%",
        default_norm: 0.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "high_level",
        name: "High Level",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
];

const CHAIN14_MID_VOICING_OPTS: [&str; 4] = [
    "VMT (Warm Tube)",
    "B3K (CMOS Grunt)",
    "Alpha (Tight Bite)",
    "Omega (Heavy Scoop)",
];

const CHAIN14_HIGH_VOICING_OPTS: [&str; 4] = [
    "B3K Turbo (Crisp CMOS)",
    "VMT (Smooth Edge)",
    "Alpha Turbo (2.8k Bite)",
    "Chinchilla (Hard Clip)",
];

const CHAIN14_SWITCHES: [FixtureSwitchSpec; 2] = [
    FixtureSwitchSpec {
        id: "mid_voicing",
        name: "Mid Drive Voicing",
        options: &CHAIN14_MID_VOICING_OPTS,
        default_index: 0,
    },
    FixtureSwitchSpec {
        id: "high_voicing",
        name: "High Drive Voicing",
        options: &CHAIN14_HIGH_VOICING_OPTS,
        default_index: 0,
    },
];

const CHAIN14_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-14-TriBand-Comp-Distortion",
    name: "Tri-Band Multi-Drive & Sub-Comp Distortion",
    category: "distortion",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 120.0,
    supports_tempo_sync: false,
    controls: &CHAIN14_CONTROLS,
    switches: &CHAIN14_SWITCHES,
};

const _: () = assert!(CHAIN14_CONTROLS.len() == 11);
const _: () = assert!(CHAIN14_SWITCHES.len() == 2);
const _: () = assert!(CHAIN14_MID_VOICINGS.len() == CHAIN14_MID_VOICING_OPTS.len());
const _: () = assert!(CHAIN14_HIGH_VOICINGS.len() == CHAIN14_HIGH_VOICING_OPTS.len());
const _: () =
    assert!(CHAIN14_CONTROLS[0].default_norm >= -1.0 && CHAIN14_CONTROLS[0].default_norm <= 1.0);
const _: () =
    assert!(CHAIN14_CONTROLS[2].default_norm >= -1.0 && CHAIN14_CONTROLS[2].default_norm <= 1.0);
const _: () =
    assert!(CHAIN14_CONTROLS[4].default_norm >= -1.0 && CHAIN14_CONTROLS[4].default_norm <= 1.0);

/// `Chain-14-TriBand-Comp-Distortion`: `1 -> 1` `4x`-oversampled Entropia-inspired tri-band
/// processor with per-band crossover/cut filters, compressed sub bass, and independent mid/high
/// drives and circuit voicings.
#[derive(Clone, Debug)]
pub struct Chain14TriBandCompDistortion {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    low_lpf_flt: TptLr4Crossover2,
    mid_hpf_flt: TptLr4Crossover2,
    mid_lpf_flt: TptLr4Crossover2,
    high_hpf_flt: TptLr4Crossover2,
    high_lpf_flt: TptLr4Crossover2,
    sub_env: f64,
    mid_dc_x1: f64,
    mid_dc_y1: f64,
    hi_dc_x1: f64,
    hi_dc_y1: f64,
    mid_tilt_st: f64,
    hi_tilt_st: f64,
}

impl Default for Chain14TriBandCompDistortion {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain14TriBandCompDistortion {
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        Self {
            sample_rate_hz: clamp_sample_rate_hz(sample_rate_hz),
            bit_depth,
            os: Oversampler4xF64::default(),
            low_lpf_flt: TptLr4Crossover2::default(),
            mid_hpf_flt: TptLr4Crossover2::default(),
            mid_lpf_flt: TptLr4Crossover2::default(),
            high_hpf_flt: TptLr4Crossover2::default(),
            high_lpf_flt: TptLr4Crossover2::default(),
            sub_env: 0.0,
            mid_dc_x1: 0.0,
            mid_dc_y1: 0.0,
            hi_dc_x1: 0.0,
            hi_dc_y1: 0.0,
            mid_tilt_st: 0.0,
            hi_tilt_st: 0.0,
        }
    }

    #[inline]
    fn control_phys(idx: usize, controls: &[f32]) -> f64 {
        let spec = &CHAIN14_CONTROLS[idx];
        let norm = controls.get(idx).copied().unwrap_or(spec.default_norm);
        f64::from(spec.norm_to_physical(norm))
    }

    #[inline]
    fn control_norm(idx: usize, controls: &[f32]) -> f64 {
        let spec = &CHAIN14_CONTROLS[idx];
        let norm = controls.get(idx).copied().unwrap_or(spec.default_norm);
        f64::from(crate::sanitize_norm(norm, spec.default_norm))
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let fc_low_lpf = Self::control_phys(0, controls);
        let fc_mid_hpf = Self::control_phys(1, controls);
        let fc_mid_lpf = Self::control_phys(2, controls).max(fc_mid_hpf * CHAIN14_MIN_BAND_RATIO);
        let fc_high_hpf = Self::control_phys(3, controls);
        let fc_high_lpf = Self::control_phys(4, controls).max(fc_high_hpf * CHAIN14_MIN_BAND_RATIO);

        let low_comp_pct = Self::control_phys(5, controls);
        let low_trim = 10.0_f64.powf(Self::control_phys(6, controls) / 20.0);
        let mid_drive_u = Self::control_norm(7, controls);
        let mid_trim = 10.0_f64.powf(Self::control_phys(8, controls) / 20.0);
        let hi_drive_u = Self::control_norm(9, controls);
        let hi_trim = 10.0_f64.powf(Self::control_phys(10, controls) / 20.0);

        let mid_voice = usize::from(
            switches
                .first()
                .copied()
                .unwrap_or(CHAIN14_SWITCHES[0].default_index)
                .min(3),
        );
        let hi_voice = usize::from(
            switches
                .get(1)
                .copied()
                .unwrap_or(CHAIN14_SWITCHES[1].default_index)
                .min(3),
        );

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_sub_atk = scale_alpha_from_192k(0.002_5, fs_4x);
        let a_sub_rel = scale_alpha_from_192k(0.000_12, fs_4x);
        let dc_pole = (0.9994_f64).powf(192_000.0 / fs_4x);
        let a_mid_tilt = scale_alpha_from_192k(0.020, fs_4x);
        let a_hi_tilt = scale_alpha_from_192k(0.090, fs_4x);

        let comp_depth = 0.6 + 6.4 * (low_comp_pct / 100.0);
        let (mid_mul, mid_bias, mid_tilt_mix) = CHAIN14_MID_VOICINGS[mid_voice];
        let mid_gain = 1.6 * 2.0_f64.powf(1.8 * mid_drive_u) * mid_mul;
        let mid_quiescent = mid_bias.tanh();

        let (hi_mul, hi_bias, hi_tilt_mix) = CHAIN14_HIGH_VOICINGS[hi_voice];
        let hi_gain = 1.8 * 2.0_f64.powf(1.8 * hi_drive_u) * hi_mul;
        let hi_quiescent = hi_bias.tanh();

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let (lo, _) = self.low_lpf_flt.step(u, fc_low_lpf, fs_4x);
                let (_, mid_bp_in) = self.mid_hpf_flt.step(u, fc_mid_hpf, fs_4x);
                let (mid, _) = self.mid_lpf_flt.step(mid_bp_in, fc_mid_lpf, fs_4x);
                let (_, hi_in) = self.high_hpf_flt.step(u, fc_high_hpf, fs_4x);

                // Low band: tight FET compressor + band level
                let lo_abs = lo.abs();
                let a_sub = if lo_abs > self.sub_env {
                    a_sub_atk
                } else {
                    a_sub_rel
                };
                self.sub_env = flush_subnormal_f64(self.sub_env + a_sub * (lo_abs - self.sub_env));
                let sub_comp_g = 1.0 / (1.0 + comp_depth * self.sub_env);
                let lo_out = low_trim * sub_comp_g * lo;

                // Mid band: independent drive, circuit voicing & band level
                let mid_raw = (mid_gain * mid + mid_bias).tanh() - mid_quiescent;
                let mid_ac =
                    flush_subnormal_f64(mid_raw - self.mid_dc_x1 + dc_pole * self.mid_dc_y1);
                self.mid_dc_x1 = mid_raw;
                self.mid_dc_y1 = mid_ac;
                self.mid_tilt_st = flush_subnormal_f64(
                    self.mid_tilt_st + a_mid_tilt * (mid_ac - self.mid_tilt_st),
                );
                let mid_out = mid_trim * (mid_ac - mid_tilt_mix * self.mid_tilt_st);

                // High band: independent drive, voicing emphasis, LPF & band level
                let hi_raw = (hi_gain * hi_in + hi_bias).tanh() - hi_quiescent;
                let hi_ac = flush_subnormal_f64(hi_raw - self.hi_dc_x1 + dc_pole * self.hi_dc_y1);
                self.hi_dc_x1 = hi_raw;
                self.hi_dc_y1 = hi_ac;
                self.hi_tilt_st =
                    flush_subnormal_f64(self.hi_tilt_st + a_hi_tilt * (hi_ac - self.hi_tilt_st));
                let hi_voiced = hi_ac - hi_tilt_mix * self.hi_tilt_st;
                let (hi_filtered, _) = self.high_lpf_flt.step(hi_voiced, fc_high_lpf, fs_4x);
                let hi_out = hi_trim * hi_filtered;

                let combined = lo_out + 0.72 * mid_out + 0.62 * hi_out;
                [flush_subnormal_f64(0.78 * combined), 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain14TriBandCompDistortion {
    fn reset(&mut self) {
        self.os.reset();
        self.low_lpf_flt.reset();
        self.mid_hpf_flt.reset();
        self.mid_lpf_flt.reset();
        self.high_hpf_flt.reset();
        self.high_lpf_flt.reset();
        self.sub_env = 0.0;
        self.mid_dc_x1 = 0.0;
        self.mid_dc_y1 = 0.0;
        self.hi_dc_x1 = 0.0;
        self.hi_dc_y1 = 0.0;
        self.mid_tilt_st = 0.0;
        self.hi_tilt_st = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN14_METADATA
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
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let y = self.step_sample(inputs.first().copied().unwrap_or(0.0), controls, switches);
        if let Some(o) = outputs.first_mut() {
            *o = y;
        }
        if outputs.len() > 1 {
            outputs[1..].fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_mono_block, sine_wave};

    #[test]
    fn test_chain14_triband_xovers_drives_voicings_and_zero_dc() {
        let meta = Chain14TriBandCompDistortion::new().metadata();
        let base_c = meta.default_controls();
        let base_s = meta.default_switches();
        let diff_sum = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| f64::from((x - y).abs()))
                .sum()
        };

        // Mid voicing and high voicing must independently alter their respective bands.
        let x_mid = sine_wave(1_536, 650.0, 0.5);
        let y_mv0 = run_mono_block(
            &mut Chain14TriBandCompDistortion::new(),
            &x_mid,
            &base_c,
            &[0, 0],
        );
        let y_mv3 = run_mono_block(
            &mut Chain14TriBandCompDistortion::new(),
            &x_mid,
            &base_c,
            &[3, 0],
        );
        assert!(
            diff_sum(&y_mv0, &y_mv3) > 1.0,
            "mid voicing should alter mid tone"
        );

        let x_hi = sine_wave(1_536, 2_400.0, 0.5);
        let y_hv0 = run_mono_block(
            &mut Chain14TriBandCompDistortion::new(),
            &x_hi,
            &base_c,
            &[0, 0],
        );
        let y_hv3 = run_mono_block(
            &mut Chain14TriBandCompDistortion::new(),
            &x_hi,
            &base_c,
            &[0, 3],
        );
        assert!(
            diff_sum(&y_hv0, &y_hv3) > 1.0,
            "high voicing should alter high tone"
        );

        // Mid LPF / High HPF / High LPF should measurably sculpt band content.
        let mut c_mlpf_lo = base_c.clone();
        let mut c_mlpf_hi = base_c.clone();
        c_mlpf_lo[2] = -0.85;
        c_mlpf_hi[2] = 0.85;
        let y_ml = run_mono_block(
            &mut Chain14TriBandCompDistortion::new(),
            &x_hi,
            &c_mlpf_lo,
            &base_s,
        );
        let y_mh = run_mono_block(
            &mut Chain14TriBandCompDistortion::new(),
            &x_hi,
            &c_mlpf_hi,
            &base_s,
        );
        assert!(diff_sum(&y_ml, &y_mh) > 1.0, "mid_lpf should gate 2.4 kHz");

        let mut fx_z = Chain14TriBandCompDistortion::new();
        let y_zero = run_mono_block(&mut fx_z, &[0.0_f32; 128], &base_c, &[1, 2]);
        assert!(y_zero.iter().all(|&v| v == 0.0));
        let _ = run_mono_block(&mut fx_z, &x_hi, &base_c, &[1, 2]);
        fx_z.reset();
        let y_reset = run_mono_block(&mut fx_z, &[0.0_f32; 128], &base_c, &[1, 2]);
        assert!(y_reset.iter().all(|&v| v == 0.0));
    }
}
