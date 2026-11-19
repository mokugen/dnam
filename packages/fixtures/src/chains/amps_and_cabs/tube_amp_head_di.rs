//! `Chain-15`: Standalone Multi-Stage Tube Amp Head with Reactive Load DI (`1 -> 1`, no Cab IR).

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, flush_subnormal_f64, scale_alpha_from_192k, scale_modal_pole_from_192k,
    substep_rate_4x,
};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};

const CHAIN15_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-15-Tube-Amp-Head-DI",
    name: "Standalone Tube Amp Head (Reactive Load DI)",
    category: "amplifier",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 800.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "gain",
            name: "Preamp Gain",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "bass",
            name: "Bass",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "mid",
            name: "Mid",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "treble",
            name: "Treble",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
        FixtureControlSpec {
            id: "presence",
            name: "Presence (NFB)",
            unit: "%",
            default_norm: 0.0,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "channel",
        name: "Channel",
        options: &["Crunch", "Lead"],
        default_index: 0,
    }],
};

/// `Chain-15-Tube-Amp-Head-DI`: `1 -> 1` `4x`-oversampled standalone tube amplifier head with
/// grid conduction, `FMV` tone stack, `EL34` push-pull `B+` supply sag, presence NFB shelf, and
/// reactive speaker-load electrical impedance resonance (`~95 Hz` + `~2.8 kHz`, zero cab IR).
#[derive(Clone, Debug)]
pub struct Chain15TubeAmpHeadDi {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    v1a_bias_rc: f64,
    interstage_hp: f64,
    interstage_lp: f64,
    fmv_low_state: f64,
    fmv_mid_state: f64,
    fmv_high_state: f64,
    power_sag_env: f64,
    pres_shelf_st: f64,
    load_re: [f64; 2],
    load_im: [f64; 2],
    dc_x1: f64,
    dc_y1: f64,
}

impl Default for Chain15TubeAmpHeadDi {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain15TubeAmpHeadDi {
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
            v1a_bias_rc: 0.0,
            interstage_hp: 0.0,
            interstage_lp: 0.0,
            fmv_low_state: 0.0,
            fmv_mid_state: 0.0,
            fmv_high_state: 0.0,
            power_sag_env: 0.0,
            pres_shelf_st: 0.0,
            load_re: [0.0; 2],
            load_im: [0.0; 2],
            dc_x1: 0.0,
            dc_y1: 0.0,
        }
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let gain_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let bass_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.0), 0.0));
        let mid_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(0.0), 0.0));
        let treb_u = f64::from(sanitize_norm(controls.get(3).copied().unwrap_or(0.0), 0.0));
        let pres_u = f64::from(sanitize_norm(controls.get(4).copied().unwrap_or(0.0), 0.0));
        let lead = switches.first().copied().unwrap_or(0) != 0;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_grid_charge = scale_alpha_from_192k(0.008, fs_4x);
        let a_grid_bleed = scale_alpha_from_192k(0.000_4, fs_4x);
        let a_hp = scale_alpha_from_192k(0.003_5, fs_4x);
        let a_lp = scale_alpha_from_192k(0.18, fs_4x);
        let a_fmv_lo = scale_alpha_from_192k(0.006, fs_4x);
        let a_fmv_md = scale_alpha_from_192k(0.032, fs_4x);
        let a_fmv_hi = scale_alpha_from_192k(0.115, fs_4x);
        let a_sag_atk = scale_alpha_from_192k(0.001_5, fs_4x);
        let a_sag_rel = scale_alpha_from_192k(4.0e-6, fs_4x);
        let a_pres = scale_alpha_from_192k(0.055, fs_4x);
        let dc_pole = (0.9994_f64).powf(192_000.0 / fs_4x);

        // Electrical reactive load modes (~95 Hz cabinet voice-coil resonance & ~2.8 kHz inductance rise).
        const LOAD_RADII_192: [f64; 2] = [0.9968, 0.9680];
        const LOAD_FREQS_192: [f64; 2] = [0.003_1, 0.092_0];
        const LOAD_GAINS: [f64; 2] = [0.32, 0.24];
        let mut poles = [(0.0_f64, 0.0_f64, 0.0_f64); 2];
        for m in 0..2 {
            let (r, w) = scale_modal_pole_from_192k(LOAD_RADII_192[m], LOAD_FREQS_192[m], fs_4x);
            let (s, c) = w.sin_cos();
            poles[m] = (r, s, c);
        }

        let g1 = (2.2 + 5.5 * (0.5 * (gain_u + 1.0))) * if lead { 1.85 } else { 1.0 };
        let b_norm = 0.5 * (bass_u + 1.0);
        let m_norm = 0.5 * (mid_u + 1.0);
        let t_norm = 0.5 * (treb_u + 1.0);
        let z_load = 1.0 + 0.45 * b_norm + 0.55 * m_norm + 0.35 * t_norm;
        let w_b = (0.3 + 0.9 * b_norm) / z_load;
        let w_m = (0.2 + 0.8 * m_norm) / z_load;
        let w_t = (0.3 + 1.1 * t_norm) / z_load;
        let pres_gain = 0.65 * (pres_u + 1.0);
        let v1b_quiescent = (-0.10_f64).tanh();

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let v1a_in = g1 * u - 0.32 * self.v1a_bias_rc;
                let v1a = (v1a_in + 0.16 * v1a_in * v1a_in).tanh();
                let grid_cond = (v1a_in - 0.4).max(0.0);
                self.v1a_bias_rc = flush_subnormal_f64(
                    self.v1a_bias_rc + a_grid_charge * grid_cond - a_grid_bleed * self.v1a_bias_rc,
                );

                self.interstage_hp =
                    flush_subnormal_f64(self.interstage_hp + a_hp * (v1a - self.interstage_hp));
                let coupled = v1a - self.interstage_hp;
                self.interstage_lp =
                    flush_subnormal_f64(self.interstage_lp + a_lp * (coupled - self.interstage_lp));

                let v1b_scale = if lead { 3.6 } else { 2.4 };
                let v1b_out = (v1b_scale * self.interstage_lp - 0.10).tanh() - v1b_quiescent;

                self.fmv_low_state = flush_subnormal_f64(
                    self.fmv_low_state + a_fmv_lo * (v1b_out - self.fmv_low_state),
                );
                self.fmv_mid_state = flush_subnormal_f64(
                    self.fmv_mid_state + a_fmv_md * (v1b_out - self.fmv_mid_state),
                );
                self.fmv_high_state = flush_subnormal_f64(
                    self.fmv_high_state + a_fmv_hi * (v1b_out - self.fmv_high_state),
                );
                let fmv_lo = self.fmv_low_state;
                let fmv_md = 0.65 * (self.fmv_mid_state - self.fmv_low_state)
                    + 0.35 * (self.fmv_high_state - self.fmv_mid_state);
                let fmv_hi = v1b_out - self.fmv_high_state;
                let fmv_sig = w_b * fmv_lo + w_m * fmv_md + w_t * fmv_hi;

                let pwr_drive = 2.1 * fmv_sig;
                let pwr_env = pwr_drive.abs();
                let sag_alpha = if pwr_env > self.power_sag_env {
                    a_sag_atk
                } else {
                    a_sag_rel
                };
                self.power_sag_env = flush_subnormal_f64(
                    self.power_sag_env + sag_alpha * (pwr_env - self.power_sag_env),
                );
                let headroom = (1.0 / (1.0 + 0.42 * self.power_sag_env)).clamp(0.48, 1.0);
                let pwr_out = headroom * (pwr_drive / headroom).tanh();

                self.pres_shelf_st = flush_subnormal_f64(
                    self.pres_shelf_st + a_pres * (pwr_out - self.pres_shelf_st),
                );
                let pwr_pres = pwr_out + pres_gain * (pwr_out - self.pres_shelf_st);

                let mut di_sum = pwr_pres;
                for m in 0..2 {
                    let (r, s, c) = poles[m];
                    let nr = flush_subnormal_f64(
                        r * (c * self.load_re[m] - s * self.load_im[m]) + pwr_pres,
                    );
                    let ni = flush_subnormal_f64(r * (s * self.load_re[m] + c * self.load_im[m]));
                    self.load_re[m] = nr;
                    self.load_im[m] = ni;
                    di_sum += (1.0 - r) * LOAD_GAINS[m] * nr;
                }
                let di_ac = flush_subnormal_f64(di_sum - self.dc_x1 + dc_pole * self.dc_y1);
                self.dc_x1 = di_sum;
                self.dc_y1 = di_ac;
                [flush_subnormal_f64(0.45 * di_ac), 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain15TubeAmpHeadDi {
    fn reset(&mut self) {
        self.os.reset();
        self.v1a_bias_rc = 0.0;
        self.interstage_hp = 0.0;
        self.interstage_lp = 0.0;
        self.fmv_low_state = 0.0;
        self.fmv_mid_state = 0.0;
        self.fmv_high_state = 0.0;
        self.power_sag_env = 0.0;
        self.pres_shelf_st = 0.0;
        self.load_re = [0.0; 2];
        self.load_im = [0.0; 2];
        self.dc_x1 = 0.0;
        self.dc_y1 = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN15_METADATA
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
    fn test_chain15_tube_head_di_channels_and_zero_dc() {
        let mut head_crunch = Chain15TubeAmpHeadDi::new();
        let mut head_lead = Chain15TubeAmpHeadDi::new();
        let x = sine_wave(1_536, 110.0, 0.45);
        let yc = run_mono_block(&mut head_crunch, &x, &[0.0; 5], &[0]);
        let yl = run_mono_block(&mut head_lead, &x, &[0.0; 5], &[1]);
        let diff: f64 = yc
            .iter()
            .zip(yl.iter())
            .map(|(&a, &b)| f64::from((a - b).abs()))
            .sum();
        assert!(diff > 1.0, "Lead switch should add gain");

        head_crunch.reset();
        let y_zero = run_mono_block(&mut head_crunch, &[0.0_f32; 128], &[0.5; 5], &[1]);
        assert!(y_zero.iter().all(|&v| v == 0.0));
    }
}
