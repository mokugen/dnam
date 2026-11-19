//! Zero-Overhead Post-Stage-3 Multi-Band & Broadband Output Conditioner
//! (`stages::hot_core::post_cond`).
//!
//! Activated exclusively when `Stage C` drives log-domain broadband gain (`stage_targets[9]`)
//! or 4-band partition-of-unity TPT EQ (`stage_targets[10, 12, 13, 14]`) and
//! `V3InterBlockToneSvfState` is idle. Resting `StarCenter` keeps `post_cond_active == false` for
//! bitwise `0-ULP` backbone invariance.

#![forbid(unsafe_code)]

use core::f64::consts::PI;

use crate::math::const_eval::const_sin_cos_f64;
use crate::math::{flush_recursive_state_f32, newton6_reciprocal_f64};
use crate::stages::stage1_svf::{exp2_signed_oct_f32, recip_nr3_pos_f32, tan_prewarp_f32};

const POST_SUB_FC_HZ: f64 = 250.0;
const POST_MID_FC_HZ: f64 = 1_800.0;
const POST_MID_K_DAMP: f32 = 1.25;
const POST_REF_SR_HZ: f64 = 48_000.0;

#[inline]
#[must_use]
const fn derive_post_sub_a_48k() -> f32 {
    let (s, c) = const_sin_cos_f64(PI * POST_SUB_FC_HZ / POST_REF_SR_HZ);
    let g = s / c;
    (g / (1.0 + g)) as f32
}

#[inline]
#[must_use]
const fn derive_post_mid_a123_48k() -> [f32; 3] {
    let (s, c) = const_sin_cos_f64(PI * POST_MID_FC_HZ / POST_REF_SR_HZ);
    let g = s / c;
    let k = POST_MID_K_DAMP as f64;
    let a1 = 1.0 / (1.0 + g * (g + k));
    let a2 = g * a1;
    let a3 = g * a2;
    [a1 as f32, a2 as f32, a3 as f32]
}

const POST_A_SUB_48K: f32 = derive_post_sub_a_48k();
const POST_A123_48K: [f32; 3] = derive_post_mid_a123_48k();
const POST_A1_48K: f32 = POST_A123_48K[0];
const POST_A2_48K: f32 = POST_A123_48K[1];
const POST_A3_48K: f32 = POST_A123_48K[2];

const _: () = assert!(
    POST_A_SUB_48K > 0.0
        && POST_A_SUB_48K < 1.0
        && POST_A1_48K > 0.0
        && POST_A1_48K < 1.0
        && POST_A2_48K > 0.0
        && POST_A3_48K > 0.0
);

/// Passive post-Stage-3 strictly positive log-octave 4-band partition-of-unity conditioner
/// (`250 Hz` TPT lowpass + `1.8 kHz` Cytomic SVF `LM/Pres/Hi`) with `3 kHz` division-free ZOH updates.
#[derive(Clone, Copy, Debug)]
pub struct V3PostStageConditioner {
    pub a_sub: f32,
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub g_sub: f32,
    pub g_lm: f32,
    pub g_pres: f32,
    pub g_hi: f32,
    pub sub_z_l: f32,
    pub sub_z_r: f32,
    pub ic1_l: f32,
    pub ic2_l: f32,
    pub ic1_r: f32,
    pub ic2_r: f32,
}

impl Default for V3PostStageConditioner {
    fn default() -> Self {
        Self::new_48k()
    }
}

impl V3PostStageConditioner {
    /// Initializes unity bypass state with compile-time derived `48 kHz` passive TPT coefficients.
    #[must_use]
    pub const fn new_48k() -> Self {
        Self {
            a_sub: POST_A_SUB_48K,
            a1: POST_A1_48K,
            a2: POST_A2_48K,
            a3: POST_A3_48K,
            g_sub: 1.0,
            g_lm: 1.0,
            g_pres: 1.0,
            g_hi: 1.0,
            sub_z_l: 0.0,
            sub_z_r: 0.0,
            ic1_l: 0.0,
            ic2_l: 0.0,
            ic1_r: 0.0,
            ic2_r: 0.0,
        }
    }

    /// Reloads default `48 kHz` unity state and clears all recursive integrators.
    pub fn reload_48k(&mut self) {
        *self = Self::new_48k();
    }

    /// Resets recursive integrators and band gains to exact `+0.0` equilibrium while keeping
    /// the current host sample-rate filter coefficients.
    pub fn reset(&mut self) {
        self.g_sub = 1.0;
        self.g_lm = 1.0;
        self.g_pres = 1.0;
        self.g_hi = 1.0;
        self.sub_z_l = 0.0;
        self.sub_z_r = 0.0;
        self.ic1_l = 0.0;
        self.ic2_l = 0.0;
        self.ic1_r = 0.0;
        self.ic2_r = 0.0;
    }

    /// Recomputes passive TPT filter coefficients for `host_sr_hz` using division-free `NR`.
    pub fn reconfigure_sample_rate(&mut self, host_sr_hz: f32) {
        let sr = f64::from(host_sr_hz.clamp(8_000.0, 384_000.0));
        let inv_sr = newton6_reciprocal_f64(sr);
        let w_sub = ((PI * POST_SUB_FC_HZ * inv_sr) as f32).clamp(1.0e-4, 1.48);
        let g_sub = tan_prewarp_f32(w_sub);
        self.a_sub = g_sub * recip_nr3_pos_f32(1.0 + g_sub);

        let w_mid = ((PI * POST_MID_FC_HZ * inv_sr) as f32).clamp(1.0e-4, 1.48);
        let g_mid = tan_prewarp_f32(w_mid);
        self.a1 = recip_nr3_pos_f32(1.0 + g_mid * (g_mid + POST_MID_K_DAMP));
        self.a2 = g_mid * self.a1;
        self.a3 = g_mid * self.a2;
    }

    /// Updates `3 kHz` log-octave 4-band gains (`exp2`) and reports whether audio-rate processing is needed.
    #[inline(always)]
    pub fn refresh_3khz(
        &mut self,
        gain_oct: f32,
        sub_oct: f32,
        lm_oct: f32,
        pres_oct: f32,
        hi_oct: f32,
        enabled: bool,
    ) -> bool {
        if !enabled
            || (gain_oct.abs() <= 1.0e-7
                && sub_oct.abs() <= 1.0e-7
                && lm_oct.abs() <= 1.0e-7
                && pres_oct.abs() <= 1.0e-7
                && hi_oct.abs() <= 1.0e-7)
        {
            self.g_sub = 1.0;
            self.g_lm = 1.0;
            self.g_pres = 1.0;
            self.g_hi = 1.0;
            return false;
        }
        let gm = gain_oct.clamp(-4.0, 2.5);
        self.g_sub = exp2_signed_oct_f32((gm + sub_oct.clamp(-2.5, 2.5)).clamp(-4.5, 3.0));
        self.g_lm = exp2_signed_oct_f32((gm + lm_oct.clamp(-2.5, 2.5)).clamp(-4.5, 3.0));
        self.g_pres = exp2_signed_oct_f32((gm + pres_oct.clamp(-2.5, 2.5)).clamp(-4.5, 3.0));
        self.g_hi = exp2_signed_oct_f32((gm + hi_oct.clamp(-2.5, 2.5)).clamp(-4.5, 3.0));
        true
    }

    #[inline(always)]
    fn step_channel(&self, sub_z: f32, ic1: f32, ic2: f32, x: f32) -> (f32, f32, f32, f32) {
        let v_sub = self.a_sub * (x - sub_z);
        let lp_sub = v_sub + sub_z;
        let next_sub_z = flush_recursive_state_f32(lp_sub + v_sub);

        let v3 = x - ic2;
        let v1 = self.a1 * ic1 + self.a2 * v3;
        let v2 = ic2 + self.a2 * ic1 + self.a3 * v3;
        let next_ic1 = flush_recursive_state_f32(2.0 * v1 - ic1);
        let next_ic2 = flush_recursive_state_f32(2.0 * v2 - ic2);

        let lm = v2 - lp_sub;
        let pres = POST_MID_K_DAMP * v1;
        let hp = x - pres - v2;

        let y = self.g_sub * lp_sub + self.g_lm * lm + self.g_pres * pres + self.g_hi * hp;
        let out = if y == 0.0 {
            0.0
        } else {
            flush_recursive_state_f32(y)
        };
        (out, next_sub_z, next_ic1, next_ic2)
    }

    /// Processes a single stereo frame in real time (`0` allocations, `0 FDIV`).
    #[inline(always)]
    pub fn step_stereo(&mut self, l: f32, r: f32, mono_out: bool) -> [f32; 2] {
        let (yl, szl, c1l, c2l) = self.step_channel(self.sub_z_l, self.ic1_l, self.ic2_l, l);
        self.sub_z_l = szl;
        self.ic1_l = c1l;
        self.ic2_l = c2l;
        if mono_out {
            [yl, yl]
        } else {
            let (yr, szr, c1r, c2r) = self.step_channel(self.sub_z_r, self.ic1_r, self.ic2_r, r);
            self.sub_z_r = szr;
            self.ic1_r = c1r;
            self.ic2_r = c2r;
            [yl, yr]
        }
    }

    /// Processes a stereo sub-block slice in place (`0` allocations, `0 FDIV`).
    #[inline(always)]
    pub fn process_slice<const MONO: bool>(
        &mut self,
        out_l: &mut [f32],
        out_r: &mut [f32],
        len: usize,
    ) {
        let n = len.min(out_l.len()).min(out_r.len());
        let mut szl = self.sub_z_l;
        let mut c1l = self.ic1_l;
        let mut c2l = self.ic2_l;
        let mut szr = self.sub_z_r;
        let mut c1r = self.ic1_r;
        let mut c2r = self.ic2_r;
        for i in 0..n {
            let (yl, n_szl, n_c1l, n_c2l) = self.step_channel(szl, c1l, c2l, out_l[i]);
            szl = n_szl;
            c1l = n_c1l;
            c2l = n_c2l;
            out_l[i] = yl;
            if MONO {
                out_r[i] = yl;
            } else {
                let (yr, n_szr, n_c1r, n_c2r) = self.step_channel(szr, c1r, c2r, out_r[i]);
                szr = n_szr;
                c1r = n_c1r;
                c2r = n_c2r;
                out_r[i] = yr;
            }
        }
        self.sub_z_l = szl;
        self.ic1_l = c1l;
        self.ic2_l = c2l;
        self.sub_z_r = szr;
        self.ic1_r = c1r;
        self.ic2_r = c2r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    #[test]
    fn test_post_cond_bypass_zero_dc_attenuation_boost_and_zoh() {
        let mut cond = V3PostStageConditioner::new_48k();
        assert!(!cond.refresh_3khz(0.0, 0.0, 0.0, 0.0, 0.0, true));
        assert!(!cond.refresh_3khz(1.0, 0.5, 0.5, 0.5, 0.5, false));

        // Zero-DC equilibrium: +0.0 input yields exact +0.0 output bits.
        assert!(cond.refresh_3khz(1.0, 0.5, -0.3, 0.4, 0.6, true));
        assert_eq!(cond.step_stereo(0.0, 0.0, false).map(f32::to_bits), [0; 2]);

        // -24 dB broadband attenuation (gain_oct = -4.0 -> 2^-4 = 0.0625)
        cond.reset();
        assert!(cond.refresh_3khz(-4.0, 0.0, 0.0, 0.0, 0.0, true));
        let [yl_att, yr_att] = cond.step_stereo(0.45, -0.35, false);
        assert!(yl_att.abs() < 0.035 && yr_att.abs() < 0.035);
        assert!(yl_att > 0.0 && yr_att < 0.0);

        // +6 dB broadband boost (gain_oct = +1.0 -> exact 2.0 * x via partition of unity)
        cond.reset();
        assert!(cond.refresh_3khz(1.0, 0.0, 0.0, 0.0, 0.0, true));
        let in_l = [0.25_f32, -0.30, 0.125, 0.0];
        let in_r = [-0.15_f32, 0.40, -0.25, 0.0];
        let (mut l, mut r) = (in_l, in_r);
        cond.process_slice::<false>(&mut l, &mut r, 4);
        for i in 0..4 {
            assert!((l[i] - 2.0 * in_l[i]).abs() < 1.0e-6);
            assert!((r[i] - 2.0 * in_r[i]).abs() < 1.0e-6);
        }

        // 4-band spectral selectivity & 96 kHz ZOH
        cond.reset();
        cond.reconfigure_sample_rate(96_000.0);
        assert!(cond.a_sub > 0.0 && cond.a_sub < POST_A_SUB_48K);
        assert!(cond.a1 > POST_A1_48K);
        cond.reconfigure_sample_rate(48_000.0);
        assert!((cond.a_sub - POST_A_SUB_48K).abs() < 1.0e-5);

        let measure_rms = |c: &mut V3PostStageConditioner, freq_hz: f64| -> f64 {
            c.reset();
            c.refresh_3khz(0.0, 1.5, 0.0, 0.0, -1.5, true);
            let mut sum = 0.0_f64;
            for i in 0..512 {
                let x = (TAU * freq_hz * (i as f64) / 48_000.0).sin() as f32;
                let [y, _] = c.step_stereo(x, x, true);
                if i >= 128 {
                    sum += f64::from(y) * f64::from(y);
                }
            }
            sum / 384.0
        };
        let lo_e = measure_rms(&mut cond, 90.0);
        let hi_e = measure_rms(&mut cond, 10_000.0);
        assert!(
            lo_e > hi_e * 8.0,
            "expected low boost vs high cut: lo={lo_e}, hi={hi_e}"
        );
    }
}
