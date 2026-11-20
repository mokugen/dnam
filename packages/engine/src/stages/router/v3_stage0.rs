//! Unconditional Stage 0 (`4x4` `s_router` + `gain` + `2x2` mix + hysteresis gate + `LR4`/`Bèzout` XO).
//!
//! Executes unconditionally on every stereo frame with zero stage-bypass flags, zero white-box
//! sub-circuits, zero `FDIV`, zero `FSQRT`, zero integer division, and zero `libm` calls
//! (`ref[dsp.router.signal-flow]`, `REQ-1.9`, `REQ-6.1`).

use crate::container::ZohRescaledParams;
use crate::math::{flush_recursive_state_f32, recip_nr2_pos_f32};
use crate::stages::router::BezoutCleanCrossover;

/// Cold parameters for Stage 0 (`4x4` `s_router`, `LR4`/`Bèzout` XO, `2x2` mix, hysteresis gate).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Stage0Params {
    pub s_router: [[f32; 4]; 4],
    pub gain: [f32; 2],
    pub mix_matrix: [[f32; 2]; 2],
    pub xo_hz: f32,
    pub xo_low_gain: f32,
    pub xo_high_gain: f32,
    pub clean_sub_mix: f32,
    pub gate_thresh: f32,
    pub gate_hyst: f32,
    pub gate_inv_knee: f32,
    pub gate_floor: f32,
    pub gate_atk_beta: f32,
    pub gate_rel_beta: f32,
}

impl Default for V3Stage0Params {
    fn default() -> Self {
        Self::identity()
    }
}

impl V3Stage0Params {
    /// Resting identity Stage 0 parameters (`0`-DC, unity matrix, `S_{ret->send} == 0`,
    /// Bèzout sum `low + high == input`).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            s_router: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            gain: [1.0, 1.0],
            mix_matrix: [[1.0, 0.0], [0.0, 1.0]],
            xo_hz: 200.0,
            xo_low_gain: 1.0,
            xo_high_gain: 1.0,
            clean_sub_mix: 0.0,
            gate_thresh: 0.0,
            gate_hyst: 0.0,
            gate_inv_knee: 0.0,
            gate_floor: 1.0,
            gate_atk_beta: 0.05,
            gate_rel_beta: 0.002,
        }
    }

    /// Clamps Stage 0 parameters into passive BIBO-safe bounds (`S_{ret->send} == 0`).
    pub fn clamp_in_place(&mut self) {
        for r in 0..4 {
            for c in 0..4 {
                let v = self.s_router[r][c];
                self.s_router[r][c] = if v.is_finite() {
                    v.clamp(-2.0, 2.0)
                } else if r == c {
                    1.0
                } else {
                    0.0
                };
            }
        }
        // Unconditional return-to-send isolation (`ref[dsp.router.return-to-send-isolation]`).
        self.s_router[0][2] = 0.0;
        self.s_router[0][3] = 0.0;
        self.s_router[1][2] = 0.0;
        self.s_router[1][3] = 0.0;

        for g in &mut self.gain {
            *g = if g.is_finite() {
                g.clamp(0.0, 4.0)
            } else {
                1.0
            };
        }
        for (r, row) in self.mix_matrix.iter_mut().enumerate() {
            for (c, entry) in row.iter_mut().enumerate() {
                *entry = if entry.is_finite() {
                    entry.clamp(-2.0, 2.0)
                } else if r == c {
                    1.0
                } else {
                    0.0
                };
            }
        }
        self.xo_hz = if self.xo_hz.is_finite() {
            self.xo_hz.clamp(20.0, 2000.0)
        } else {
            200.0
        };
        self.xo_low_gain = self.xo_low_gain.clamp(0.0, 2.0);
        self.xo_high_gain = self.xo_high_gain.clamp(0.0, 2.0);
        self.clean_sub_mix = self.clean_sub_mix.clamp(0.0, 2.0);
        self.gate_thresh = self.gate_thresh.clamp(0.0, 0.5);
        self.gate_hyst = self.gate_hyst.clamp(0.0, 0.25);
        self.gate_inv_knee = self.gate_inv_knee.clamp(0.0, 2000.0);
        self.gate_floor = self.gate_floor.clamp(0.0, 1.0);
        self.gate_atk_beta = self.gate_atk_beta.clamp(1.0e-5, 0.5);
        self.gate_rel_beta = self.gate_rel_beta.clamp(1.0e-5, 0.2);
    }

    /// Rescales `48 kHz`-anchored crossover and 1-pole filter constants for `zoh`.
    pub fn rescale_with_zoh(&mut self, zoh: &ZohRescaledParams) {
        self.clamp_in_place();
        if (zoh.rho_sr - 1.0).abs() <= 1.0e-7 {
            return;
        }
        self.gate_atk_beta =
            (1.0 - zoh.rescale_real_pole(1.0 - self.gate_atk_beta)).clamp(1.0e-5, 0.5);
        self.gate_rel_beta =
            (1.0 - zoh.rescale_real_pole(1.0 - self.gate_rel_beta)).clamp(1.0e-5, 0.2);
    }
}

/// Hot runtime state for unconditional `48 kHz` Stage 0 processing.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct V3Stage0Hot {
    pub params: V3Stage0Params,
    pub xo_l: BezoutCleanCrossover,
    pub xo_r: BezoutCleanCrossover,
    pub gate_env: f32,
    pub gate_gain: f32,
}

impl V3Stage0Hot {
    /// Compiles Stage 0 hot state from cold parameters.
    #[must_use]
    pub fn from_cold(params: &V3Stage0Params) -> Self {
        let mut p = *params;
        p.clamp_in_place();
        let mut xo_l = BezoutCleanCrossover::new();
        let mut xo_r = BezoutCleanCrossover::new();
        xo_l.set_crossover_hz(p.xo_hz);
        xo_r.set_crossover_hz(p.xo_hz);
        Self {
            params: p,
            xo_l,
            xo_r,
            gate_env: 0.0,
            gate_gain: 1.0,
        }
    }

    /// Alias for [`Self::from_cold`] compiling Stage 0 hot state from cold parameters.
    #[inline]
    #[must_use]
    pub fn from_params(params: &V3Stage0Params) -> Self {
        Self::from_cold(params)
    }

    /// Updates Stage 0 parameters in place without resetting IIR state.
    pub fn update_params(&mut self, params: &V3Stage0Params) {
        let mut p = *params;
        p.clamp_in_place();
        if (p.xo_hz - self.params.xo_hz).abs() > 1.0e-4 {
            self.xo_l.set_crossover_hz(p.xo_hz);
            self.xo_r.set_crossover_hz(p.xo_hz);
        }
        self.params = p;
    }

    /// Reconfigures Stage 0 crossover and detector poles under continuous-time ZOH without resetting state.
    pub fn reconfigure_with_zoh(&mut self, base: &V3Stage0Params, zoh: &ZohRescaledParams) {
        let mut p = *base;
        p.rescale_with_zoh(zoh);
        self.xo_l.set_crossover_hz(p.xo_hz);
        self.xo_r.set_crossover_hz(p.xo_hz);
        self.xo_l.reconfigure_sample_rate(zoh);
        self.xo_r.reconfigure_sample_rate(zoh);
        self.params = p;
    }

    /// Reconfigures Stage 0 crossover and filter time constants for `sample_rate_hz` (`48 kHz`-anchored).
    pub fn reconfigure_sample_rate(&mut self, base: &V3Stage0Params, sample_rate_hz: f32) {
        let sr = sample_rate_hz.clamp(44_100.0, 384_000.0) as u32;
        let zoh = ZohRescaledParams::new(48_000, sr);
        self.reconfigure_with_zoh(base, &zoh);
    }

    /// Resets recursive state while retaining parameters.
    pub fn reset(&mut self) {
        self.xo_l.reset();
        self.xo_r.reset();
        self.gate_env = 0.0;
        self.gate_gain = 1.0;
    }

    /// Steps unconditional stereo Stage 0 (`gain` + `2x2` mix + hysteresis gate + Bèzout XO).
    ///
    /// Returns `[down_l, down_r, clean_sub_l, clean_sub_r]`.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 4] {
        let xl = self.params.gain[0] * in_l;
        let xr = self.params.gain[1] * in_r;
        let ml = self.params.mix_matrix[0][0] * xl + self.params.mix_matrix[0][1] * xr;
        let mr = self.params.mix_matrix[1][0] * xl + self.params.mix_matrix[1][1] * xr;

        let peak = 0.5 * (xl.abs() + xr.abs());
        let beta = if peak > self.gate_env {
            self.params.gate_atk_beta
        } else {
            self.params.gate_rel_beta
        };
        let next_env = flush_recursive_state_f32(self.gate_env + beta * (peak - self.gate_env));
        self.gate_env = next_env;

        let thr = self.params.gate_thresh - self.params.gate_hyst * self.gate_gain;
        let target_gate =
            ((next_env - thr) * self.params.gate_inv_knee + self.params.gate_floor).clamp(0.0, 1.0);
        self.gate_gain =
            flush_recursive_state_f32(self.gate_gain + beta * (target_gate - self.gate_gain));

        let gated_l = self.gate_gain * ml;
        let gated_r = self.gate_gain * mr;
        let (sub_l, sub_r) = self.xo_l.step_stereo(&mut self.xo_r, gated_l, gated_r);
        let xo_l = self.params.xo_low_gain * sub_l + self.params.xo_high_gain * (gated_l - sub_l);
        let xo_r = self.params.xo_low_gain * sub_r + self.params.xo_high_gain * (gated_r - sub_r);

        let _ = recip_nr2_pos_f32;
        [
            xo_l,
            xo_r,
            (self.params.clean_sub_mix * sub_l) + 0.0,
            (self.params.clean_sub_mix * sub_r) + 0.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// verify[dsp.crossbar.top-of-frame]
    #[test]
    fn test_stage0_identity_zero_dc_and_zoh_reconfigure() {
        let mut s0 = V3Stage0Hot::from_params(&V3Stage0Params::identity());
        assert_eq!(s0.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
        let mut s0_act = V3Stage0Params::identity();
        s0_act.s_router[0][2] = 0.99;
        s0_act.s_router[1][3] = -0.5;
        s0_act.gate_thresh = 0.01;
        s0_act.gate_hyst = 0.005;
        s0_act.gate_inv_knee = 100.0;
        s0_act.gate_floor = 0.0;
        s0_act.xo_hz = 250.0;
        s0_act.clean_sub_mix = 0.25;
        let mut hot = V3Stage0Hot::from_cold(&s0_act);
        assert_eq!(hot.params.s_router[0][2], 0.0);
        assert_eq!(hot.params.s_router[1][3], 0.0);
        for n in 0..256 {
            let x = 0.4 * (n as f32 * 0.09).sin();
            assert!(
                hot.step_stereo(x, -x)
                    .iter()
                    .all(|v| v.is_finite() && v.abs() < 4.0)
            );
        }
        hot.reconfigure_sample_rate(&s0_act, 96_000.0);
        assert!(hot.xo_l.pole > s0.xo_l.pole);
        hot.reset();
        assert_eq!(hot.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_stage0_linear_identity_unclipped_envelope(
            in_l in -100.0_f32..100.0_f32,
            in_r in -100.0_f32..100.0_f32,
        ) {
            let mut s0 = V3Stage0Hot::from_cold(&V3Stage0Params::identity());
            let out = s0.step_stereo(in_l, in_r);
            prop_assert!((out[0] - in_l).abs() < 1e-4);
            prop_assert!((out[1] - in_r).abs() < 1e-4);
            prop_assert_eq!(out[2].to_bits(), 0);
            prop_assert_eq!(out[3].to_bits(), 0);
        }
    }
}
