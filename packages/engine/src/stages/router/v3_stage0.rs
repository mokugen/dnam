//! Unconditional Stage 0 (`LR4`/`Bèzout` Crossover + `Op-P` Sub-Harmonic Divider + Hysteresis Gate).
//!
//! Executes unconditionally on every stereo frame with zero stage-bypass flags, zero `FDIV`,
//! zero `FSQRT`, zero integer division, and zero `libm` calls (`ref[dsp.router.signal-flow]`).

use crate::container::ZohRescaledParams;
use crate::math::flush_recursive_state_f32;
use crate::stages::router::BezoutCleanCrossover;
use crate::stages::stage1_svf::recip_nr2_pos_f32;

/// Cold parameters for Stage 0 (`LR4`/`Bèzout` crossover, `2x2` mix, hysteresis gate, `Op-P` sub divider).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Stage0Params {
    pub gain: [f32; 2],
    pub mix_matrix: [[f32; 2]; 2],
    pub xo_hz: f32,
    pub xo_low_gain: f32,
    pub xo_high_gain: f32,
    pub clean_sub_mix: f32,
    pub op_p_sub_mix: f32,
    pub op_p_lpf_pole: f32,
    /// Optional independent envelope follower pole for `Op-P` (`0.0` follows `op_p_lpf_pole`).
    pub op_p_env_pole: f32,
    /// Mix gain for the `-2 Oct` flip-flop sub-octave divider.
    pub op_p_sub2_mix: f32,
    /// Mix gain for the full-wave octave doubler term (`|u|`).
    pub op_p_oct_mix: f32,
    /// Mix gain for the `Op-P` rectification baseline compensation (`env`).
    pub op_p_env_mix: f32,
    pub gate_thresh: f32,
    pub gate_hyst: f32,
    pub gate_inv_knee: f32,
    pub gate_floor: f32,
    pub gate_atk_beta: f32,
    pub gate_rel_beta: f32,
    /// Clocks the `-2 Oct` divider on the alternate `-1 Oct` transition (`false` / `true`).
    pub op_p_sub2_phase: bool,
    /// Routes `Op-P` output to `clean_sub` (`out[2..4]`) bypassing Stage I–III when `true`.
    pub op_p_route_clean: bool,
}

impl Default for V3Stage0Params {
    fn default() -> Self {
        Self::identity()
    }
}

impl V3Stage0Params {
    /// Resting identity Stage 0 parameters (`0`-DC, unity matrix, Bèzout sum `low + high == input`).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            gain: [1.0, 1.0],
            mix_matrix: [[1.0, 0.0], [0.0, 1.0]],
            xo_hz: 200.0,
            xo_low_gain: 1.0,
            xo_high_gain: 1.0,
            clean_sub_mix: 0.0,
            op_p_sub_mix: 0.0,
            op_p_lpf_pole: 0.85,
            op_p_env_pole: 0.0,
            op_p_sub2_mix: 0.0,
            op_p_oct_mix: 0.0,
            op_p_env_mix: 0.0,
            gate_thresh: 0.0,
            gate_hyst: 0.0,
            gate_inv_knee: 0.0,
            gate_floor: 1.0,
            gate_atk_beta: 0.05,
            gate_rel_beta: 0.002,
            op_p_sub2_phase: false,
            op_p_route_clean: false,
        }
    }

    /// Clamps Stage 0 parameters into passive BIBO-safe bounds.
    pub fn clamp_in_place(&mut self) {
        for g in &mut self.gain {
            *g = g.clamp(0.0, 4.0);
        }
        for row in &mut self.mix_matrix {
            for entry in row {
                *entry = entry.clamp(-2.0, 2.0);
            }
        }
        self.xo_hz = self.xo_hz.clamp(20.0, 2000.0);
        self.xo_low_gain = self.xo_low_gain.clamp(0.0, 2.0);
        self.xo_high_gain = self.xo_high_gain.clamp(0.0, 2.0);
        self.clean_sub_mix = self.clean_sub_mix.clamp(0.0, 2.0);
        self.op_p_sub_mix = self.op_p_sub_mix.clamp(-2.0, 2.0);
        self.op_p_lpf_pole = self.op_p_lpf_pole.clamp(0.0, 0.9995);
        self.op_p_env_pole = self.op_p_env_pole.clamp(0.0, 0.9999);
        self.op_p_sub2_mix = self.op_p_sub2_mix.clamp(-2.0, 2.0);
        self.op_p_oct_mix = self.op_p_oct_mix.clamp(-4.0, 4.0);
        self.op_p_env_mix = self.op_p_env_mix.clamp(-4.0, 4.0);
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
        self.op_p_lpf_pole = zoh.rescale_real_pole(self.op_p_lpf_pole).clamp(0.0, 0.9995);
        if self.op_p_env_pole > 0.0 {
            self.op_p_env_pole = zoh.rescale_real_pole(self.op_p_env_pole).clamp(0.0, 0.9999);
        }
        self.gate_atk_beta =
            (1.0 - zoh.rescale_real_pole(1.0 - self.gate_atk_beta)).clamp(1.0e-5, 0.5);
        self.gate_rel_beta =
            (1.0 - zoh.rescale_real_pole(1.0 - self.gate_rel_beta)).clamp(1.0e-5, 0.2);
    }
}

#[inline(always)]
fn pole_48k_to_alpha_4x(pole: f32) -> f32 {
    let d = (1.0 - pole).clamp(0.0, 0.5);
    d * (0.25
        + d * (0.09375
            + d * (0.054_687_5 + d * (0.037_597_656 + d * (0.028_198_242 + d * 0.022_323_608)))))
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
    pub sub_env: [f32; 2],
    pub sub_lp: [f32; 2],
    pub sub_prev_x: [f32; 2],
    pub sub_up: [[[f32; 2]; 2]; 2],
    pub sub_prev_sign: [u8; 2],
    pub sub_cnt: [u8; 2],
    pub sub_div: [i8; 2],
    pub sub_div2: [i8; 2],
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
            sub_env: [0.0; 2],
            sub_lp: [0.0; 2],
            sub_prev_x: [0.0; 2],
            sub_up: [[[0.0; 2]; 2]; 2],
            sub_prev_sign: [1; 2],
            sub_cnt: [0; 2],
            sub_div: [1; 2],
            sub_div2: [1; 2],
        }
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
        if let Ok(zoh) = ZohRescaledParams::compute(48_000, sr, 0.9993, 0.995, 64.0, 128) {
            self.reconfigure_with_zoh(base, &zoh);
        }
    }

    /// Resets recursive state while retaining parameters.
    pub fn reset(&mut self) {
        self.xo_l.reset();
        self.xo_r.reset();
        self.gate_env = 0.0;
        self.gate_gain = 1.0;
        self.sub_env = [0.0; 2];
        self.sub_lp = [0.0; 2];
        self.sub_prev_x = [0.0; 2];
        self.sub_up = [[[0.0; 2]; 2]; 2];
        self.sub_prev_sign = [1; 2];
        self.sub_cnt = [0; 2];
        self.sub_div = [1; 2];
        self.sub_div2 = [1; 2];
    }

    /// Steps unconditional stereo Stage 0 (`2x2` mix + hysteresis gate + Bèzout XO + `Op-P`).
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

        let enhanced_sub = self.params.op_p_route_clean
            || self.params.op_p_sub2_mix != 0.0
            || self.params.op_p_oct_mix != 0.0
            || self.params.op_p_env_mix != 0.0;
        if !enhanced_sub {
            let op_sub_l = self.step_sub_divider_legacy_ch(0, gated_l);
            let op_sub_r = self.step_sub_divider_legacy_ch(1, gated_r);
            return [
                xo_l + self.params.op_p_sub_mix * op_sub_l,
                xo_r + self.params.op_p_sub_mix * op_sub_r,
                0.0 + self.params.clean_sub_mix * sub_l,
                0.0 + self.params.clean_sub_mix * sub_r,
            ];
        }
        if self.params.op_p_route_clean {
            let op_sub_l = self.step_sub_divider_ch(0, gated_l);
            let op_sub_r = self.step_sub_divider_ch(1, gated_r);
            [
                xo_l,
                xo_r,
                self.params.clean_sub_mix * sub_l + op_sub_l,
                self.params.clean_sub_mix * sub_r + op_sub_r,
            ]
        } else {
            let op_sub_l = self.step_sub_divider_4x_ch(0, gated_l);
            let op_sub_r = self.step_sub_divider_4x_ch(1, gated_r);
            [
                xo_l + op_sub_l,
                xo_r + op_sub_r,
                self.params.clean_sub_mix * sub_l,
                self.params.clean_sub_mix * sub_r,
            ]
        }
    }

    #[inline(always)]
    fn effective_env_pole(&self) -> f32 {
        if self.params.op_p_env_pole > 0.0 {
            self.params.op_p_env_pole
        } else {
            self.params.op_p_lpf_pole
        }
    }

    #[inline(always)]
    fn step_sub_divider_legacy_ch(&mut self, ch: usize, x: f32) -> f32 {
        let p = self.params.op_p_lpf_pole;
        let p_env = self.effective_env_pole();
        let lp = flush_recursive_state_f32(p * self.sub_lp[ch] + (1.0 - p) * x);
        self.sub_lp[ch] = lp;
        let env = flush_recursive_state_f32(p_env * self.sub_env[ch] + (1.0 - p_env) * x.abs());
        self.sub_env[ch] = env;
        let pos = u8::from(lp >= 0.0);
        if self.sub_prev_sign[ch] == 0 && pos == 1 {
            self.sub_div[ch] = -self.sub_div[ch];
        }
        self.sub_prev_sign[ch] = pos;
        f32::from(self.sub_div[ch]) * env
    }

    #[inline(always)]
    fn step_sub_divider_ch(&mut self, ch: usize, x: f32) -> f32 {
        let p = self.params.op_p_lpf_pole;
        let p_env = self.effective_env_pole();
        let prev_lp = self.sub_lp[ch];
        let lp = flush_recursive_state_f32(p * prev_lp + (1.0 - p) * x);
        self.sub_lp[ch] = lp;
        let abs_x = x.abs();
        let env = flush_recursive_state_f32(p_env * self.sub_env[ch] + (1.0 - p_env) * abs_x);
        self.sub_env[ch] = env;
        let pos = u8::from(lp >= 0.0);
        let (d1, d2) = if self.sub_prev_sign[ch] == 0 && pos == 1 {
            let denom = (lp - prev_lp).max(1.0e-12);
            let alpha = (lp * recip_nr2_pos_f32(denom)).clamp(0.0, 1.0);
            let old1 = self.sub_div[ch];
            let new1 = -old1;
            self.sub_div[ch] = new1;
            let d1_f = f32::from(old1);
            let d1_eff = d1_f + alpha * (f32::from(new1) - d1_f);
            let cnt = self.sub_cnt[ch];
            self.sub_cnt[ch] = cnt.wrapping_add(1);
            let d2_eff = if (cnt & 1) == u8::from(self.params.op_p_sub2_phase) {
                let old2 = self.sub_div2[ch];
                let new2 = -old2;
                self.sub_div2[ch] = new2;
                let d2_f = f32::from(old2);
                d2_f + alpha * (f32::from(new2) - d2_f)
            } else {
                f32::from(self.sub_div2[ch])
            };
            (d1_eff, d2_eff)
        } else {
            (f32::from(self.sub_div[ch]), f32::from(self.sub_div2[ch]))
        };
        self.sub_prev_sign[ch] = pos;
        env * (self.params.op_p_sub_mix * d1
            + self.params.op_p_sub2_mix * d2
            + self.params.op_p_env_mix)
            + self.params.op_p_oct_mix * abs_x
    }

    /// Steps the `4x` Butter4 anti-imaging `Op-P` sub-octave divider on channel `ch`,
    /// returning `[u_avg, s1_avg, s2_avg, oct_avg, env_avg]` (`0` `FDIV`, `0` `FSQRT`, `0` `libm`).
    #[inline(always)]
    pub fn step_op_p_4x_components_ch(&mut self, ch: usize, x: f32) -> [f32; 5] {
        const fn butter4_stage_coeffs(two_r_idx: f64) -> (f32, f32, f32) {
            let g = crate::math::const_tan_f64(core::f64::consts::PI * 0.475 / 4.0);
            let two_r = 2.0 * crate::math::const_sin_f64(two_r_idx * core::f64::consts::PI / 8.0);
            let a1 = 1.0 / (1.0 + g * (g + two_r));
            let a2 = g * a1;
            let a3 = g * a2;
            (a1 as f32, a2 as f32, a3 as f32)
        }
        const B4_COEFFS: [(f32, f32, f32); 2] =
            [butter4_stage_coeffs(1.0), butter4_stage_coeffs(3.0)];
        let xl = 1.185 * x - 0.185 * self.sub_prev_x[ch];
        self.sub_prev_x[ch] = x;
        let a_trk = pole_48k_to_alpha_4x(self.params.op_p_lpf_pole);
        let a_env = pole_48k_to_alpha_4x(self.effective_env_pole());
        let phase_bit = u8::from(self.params.op_p_sub2_phase);
        let mut acc_u = 0.0_f32;
        let mut acc_s1 = 0.0_f32;
        let mut acc_s2 = 0.0_f32;
        let mut acc_oct = 0.0_f32;
        let mut acc_env = 0.0_f32;
        for _ in 0..4 {
            let mut u = xl;
            for (st, &(a1, a2, a3)) in self.sub_up[ch].iter_mut().zip(&B4_COEFFS) {
                let v3 = u - st[1];
                let v1 = a1 * st[0] + a2 * v3;
                let v2 = st[1] + a2 * st[0] + a3 * v3;
                st[0] = flush_recursive_state_f32(2.0 * v1 - st[0]);
                st[1] = flush_recursive_state_f32(2.0 * v2 - st[1]);
                u = flush_recursive_state_f32(v2);
            }
            let lp = flush_recursive_state_f32(self.sub_lp[ch] + a_trk * (u - self.sub_lp[ch]));
            self.sub_lp[ch] = lp;
            let abs_u = u.abs();
            let env =
                flush_recursive_state_f32(self.sub_env[ch] + a_env * (abs_u - self.sub_env[ch]));
            self.sub_env[ch] = env;
            let pos = u8::from(lp >= 0.0);
            if self.sub_prev_sign[ch] == 0 && pos == 1 {
                self.sub_div[ch] = -self.sub_div[ch];
                let next_cnt = self.sub_cnt[ch].wrapping_add(1);
                self.sub_cnt[ch] = next_cnt;
                if (next_cnt & 1) == phase_bit {
                    self.sub_div2[ch] = -self.sub_div2[ch];
                }
            }
            self.sub_prev_sign[ch] = pos;
            acc_u += u;
            acc_s1 += f32::from(self.sub_div[ch]) * env;
            acc_s2 += f32::from(self.sub_div2[ch]) * env;
            acc_oct += abs_u;
            acc_env += env;
        }
        [
            0.25 * acc_u,
            0.25 * acc_s1,
            0.25 * acc_s2,
            0.25 * acc_oct,
            0.25 * acc_env,
        ]
    }

    #[inline(always)]
    fn step_sub_divider_4x_ch(&mut self, ch: usize, x: f32) -> f32 {
        let [u_avg, s1, s2, oct, env] = self.step_op_p_4x_components_ch(ch, x);
        (u_avg - x)
            + self.params.op_p_sub_mix * s1
            + self.params.op_p_sub2_mix * s2
            + self.params.op_p_oct_mix * oct
            + self.params.op_p_env_mix * env
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stage0_identity_zero_dc_and_zoh_reconfigure() {
        let mut s0 = V3Stage0Hot::from_cold(&V3Stage0Params::identity());
        assert_eq!(s0.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
        let mut s0_act = V3Stage0Params::identity();
        s0_act.op_p_sub_mix = 0.25;
        s0_act.gate_thresh = 0.01;
        s0_act.gate_hyst = 0.005;
        s0_act.gate_inv_knee = 100.0;
        s0_act.gate_floor = 0.0;
        s0_act.xo_hz = 250.0;
        let mut hot = V3Stage0Hot::from_cold(&s0_act);
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
        assert!(hot.params.op_p_lpf_pole > s0_act.op_p_lpf_pole);
        hot.reset();
        assert_eq!(hot.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
    }

    #[test]
    fn test_stage0_op_p_sub2_clean_routing_and_zoh() {
        let p_sub = V3Stage0Params {
            op_p_sub_mix: 0.6,
            op_p_lpf_pole: 0.9,
            op_p_env_pole: 0.99,
            op_p_sub2_mix: 0.4,
            op_p_sub2_phase: false,
            op_p_route_clean: true,
            ..V3Stage0Params::identity()
        };
        let p_phase = V3Stage0Params {
            op_p_sub2_phase: true,
            ..p_sub
        };
        let mut h_a = V3Stage0Hot::from_cold(&p_sub);
        let mut h_b = V3Stage0Hot::from_cold(&p_phase);
        assert_eq!(h_a.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
        let mut clean_energy = 0.0_f32;
        let mut phase_diff = 0.0_f32;
        let mut down_diff = 0.0_f32;
        for n in 0..512 {
            let x = 0.75 * (n as f32 * 0.04 * core::f32::consts::TAU).sin();
            let oa = h_a.step_stereo(x, x);
            let ob = h_b.step_stereo(x, x);
            clean_energy += oa[2] * oa[2];
            phase_diff += (oa[2] - ob[2]).abs();
            down_diff += (oa[0] - x).abs();
        }
        assert!(clean_energy > 1.0);
        assert!(phase_diff > 0.5);
        assert!(down_diff < 1e-3, "route_clean keeps down path Bèzout unity");
        let mut p_oct = p_sub;
        p_oct.op_p_oct_mix = 0.77;
        p_oct.op_p_env_mix = -0.35;
        let mut h_oct = V3Stage0Hot::from_cold(&p_oct);
        assert_eq!(h_oct.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 4]);
        let y_oct = h_oct.step_stereo(-0.5, -0.5);
        assert!(y_oct[2].abs() > 0.1 && y_oct[0] < 0.0);
        let mut h_pos = V3Stage0Hot::from_cold(&p_sub);
        let _ = h_pos.step_stereo(0.25, 0.25);
        assert_eq!(h_pos.sub_div, [1, 1]);
        let _ = h_pos.step_stereo(-1.0, -1.0);
        let _ = h_pos.step_stereo(1.0, 1.0);
        assert_eq!(h_pos.sub_div, [-1, -1]);
        h_a.reconfigure_sample_rate(&p_sub, 96_000.0);
        assert!(h_a.params.op_p_env_pole > p_sub.op_p_env_pole);
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
