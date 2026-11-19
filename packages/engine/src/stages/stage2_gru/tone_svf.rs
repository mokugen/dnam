//! Inter-Block 3-Band Tone Stack `TPT-SVF` (`Block 0 -> Tone -> Block 1`, `stage2_gru/tone_svf.rs`).
//!
//! Refreshes normalized cutoff & band EQ at `3 kHz` (`tick & 15 == 0`) with zero audio-rate
//! divisions (`ref[dsp.op-n.stage2.tone-basis]`).

use crate::math::{flush_recursive_state_f32, newton6_reciprocal_f64};
use crate::stages::stage1_svf::{exp2_signed_oct_f32, recip_nr3_pos_f32, tan_prewarp_f32};

const TONE_BASE_MIN_HZ: f32 = 25.0;
const TONE_BASE_MAX_HZ: f32 = 16_000.0;

/// Cold parameters for the Inter-Block Tone `TPT-SVF` (`Block 0 -> Tone -> Block 1`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3InterBlockToneSvfParams {
    pub base_cutoff_hz: f32,
    pub sample_rate_hz: f32,
    pub q_res: f32,
    pub dry_mix: f32,
    pub lp_mix: f32,
    pub bp_mix: f32,
    pub hp_mix: f32,
    /// Projection weights from Block 0 shared `Lite` lanes `[0..4]` into Tone SVF input.
    pub h0_proj_w: [f32; 4],
    pub s1_proj_w: f32,
    /// Injection weights adding Tone SVF output directly into Stage III `cab_acc` (and Block 1 feat 0).
    pub cab_inject_w: f32,
    pub b1_inject_w: f32,
}

impl Default for V3InterBlockToneSvfParams {
    fn default() -> Self {
        Self::identity()
    }
}

impl V3InterBlockToneSvfParams {
    /// Resting identity tone stack (`dry_mix = 1.0`, zero inter-block injection).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            base_cutoff_hz: 1_200.0,
            sample_rate_hz: 48_000.0,
            q_res: 0.707_106_77,
            dry_mix: 1.0,
            lp_mix: 0.0,
            bp_mix: 0.0,
            hp_mix: 0.0,
            h0_proj_w: [0.0; 4],
            s1_proj_w: 0.0,
            cab_inject_w: 0.0,
            b1_inject_w: 0.0,
        }
    }

    /// Clamps tone SVF parameters into unconditionally passive TPT bounds.
    pub fn clamp_in_place(&mut self) {
        self.base_cutoff_hz = self
            .base_cutoff_hz
            .clamp(TONE_BASE_MIN_HZ, TONE_BASE_MAX_HZ);
        self.sample_rate_hz = self.sample_rate_hz.clamp(44_100.0, 384_000.0);
        self.q_res = self.q_res.clamp(0.3, 6.0);
        self.dry_mix = self.dry_mix.clamp(-2.0, 2.0);
        self.lp_mix = self.lp_mix.clamp(-2.0, 2.0);
        self.bp_mix = self.bp_mix.clamp(-2.0, 2.0);
        self.hp_mix = self.hp_mix.clamp(-2.0, 2.0);
        self.s1_proj_w = self.s1_proj_w.clamp(-4.0, 4.0);
        self.cab_inject_w = self.cab_inject_w.clamp(-4.0, 4.0);
        self.b1_inject_w = self.b1_inject_w.clamp(-4.0, 4.0);
        for w in &mut self.h0_proj_w {
            *w = w.clamp(-4.0, 4.0);
        }
    }
}

/// Hot stereo Cytomic `TPT-SVF` with `3 kHz` (`tick & 15 == 0`) coefficient updates and zero `FDIV`.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct V3InterBlockToneSvfState {
    pub params: V3InterBlockToneSvfParams,
    pub base_norm_w: f32,
    pub base_k_damp: f32,
    pub k_damp: f32,
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub dry_eff: f32,
    pub lp_eff: f32,
    pub bp_eff: f32,
    pub hp_eff: f32,
    pub ic1_l: f32,
    pub ic2_l: f32,
    pub ic1_r: f32,
    pub ic2_r: f32,
}

impl V3InterBlockToneSvfState {
    /// Compiles Tone SVF state and initial `f64`-exact Cytomic coefficients.
    #[must_use]
    pub fn from_cold(params: &V3InterBlockToneSvfParams) -> Self {
        let mut p = *params;
        p.clamp_in_place();
        let inv_sr = newton6_reciprocal_f64(f64::from(p.sample_rate_hz));
        let base_norm_w = (core::f64::consts::PI * f64::from(p.base_cutoff_hz) * inv_sr) as f32;
        let base_k_damp = (newton6_reciprocal_f64(f64::from(p.q_res)) as f32).clamp(0.16, 3.33);
        let mut st = Self {
            params: p,
            base_norm_w,
            base_k_damp,
            k_damp: base_k_damp,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            dry_eff: p.dry_mix,
            lp_eff: p.lp_mix,
            bp_eff: p.bp_mix,
            hp_eff: p.hp_mix,
            ic1_l: 0.0,
            ic2_l: 0.0,
            ic1_r: 0.0,
            ic2_r: 0.0,
        };
        st.refresh_3khz(0.0, 0.0, [0.0; 3]);
        st
    }

    /// Updates Tone SVF parameters in place while keeping integrator states continuous.
    pub fn update_params(&mut self, params: &V3InterBlockToneSvfParams) {
        let mut p = *params;
        p.clamp_in_place();
        let inv_sr = newton6_reciprocal_f64(f64::from(p.sample_rate_hz));
        self.base_norm_w = (core::f64::consts::PI * f64::from(p.base_cutoff_hz) * inv_sr) as f32;
        self.base_k_damp = (newton6_reciprocal_f64(f64::from(p.q_res)) as f32).clamp(0.16, 3.33);
        self.params = p;
    }

    /// Updates effective host rate under ZOH (`sample_rate_hz` scaled by `os2x` factor).
    pub fn reconfigure_sample_rate(&mut self, effective_sr_hz: f32) {
        let mut p = self.params;
        p.sample_rate_hz = effective_sr_hz.clamp(44_100.0, 384_000.0);
        self.update_params(&p);
        self.refresh_3khz(0.0, 0.0, [0.0; 3]);
    }

    /// Recomputes `(a1, a2, a3)` and gains at `3 kHz` (`0 FDIV`, division-free `NR3`).
    #[inline(always)]
    pub fn refresh_3khz(&mut self, fc_oct_delta: f32, k_delta: f32, eq_delta: [f32; 3]) {
        let scale = exp2_signed_oct_f32(fc_oct_delta.clamp(-3.0, 3.0));
        let w = (self.base_norm_w * scale).clamp(1.0e-4, 1.48);
        let g = tan_prewarp_f32(w);
        let k = (self.base_k_damp + k_delta).clamp(0.16, 3.33);
        let a1 = recip_nr3_pos_f32(1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        self.k_damp = k;
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
        self.dry_eff = self.params.dry_mix;
        self.lp_eff = (self.params.lp_mix + eq_delta[0]).clamp(-2.5, 2.5);
        self.bp_eff = (self.params.bp_mix + eq_delta[1]).clamp(-2.5, 2.5);
        self.hp_eff = (self.params.hp_mix + eq_delta[2]).clamp(-2.5, 2.5);
    }

    /// Resets integrator state while keeping coefficients.
    pub fn reset(&mut self) {
        self.ic1_l = 0.0;
        self.ic2_l = 0.0;
        self.ic1_r = 0.0;
        self.ic2_r = 0.0;
    }

    /// Steps stereo Cytomic `TPT-SVF` unconditionally on Block 0 + Stage I input projections.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let (v1_l, v2_l) = Self::step_channel(
            in_l,
            self.a1,
            self.a2,
            self.a3,
            &mut self.ic1_l,
            &mut self.ic2_l,
        );
        let (v1_r, v2_r) = Self::step_channel(
            in_r,
            self.a1,
            self.a2,
            self.a3,
            &mut self.ic1_r,
            &mut self.ic2_r,
        );
        let hp_l = in_l - self.k_damp * v1_l - v2_l;
        let hp_r = in_r - self.k_damp * v1_r - v2_r;
        let y_l = self.dry_eff * in_l + self.lp_eff * v2_l + self.bp_eff * v1_l + self.hp_eff * hp_l;
        let y_r = self.dry_eff * in_r + self.lp_eff * v2_r + self.bp_eff * v1_r + self.hp_eff * hp_r;
        [
            if y_l == 0.0 {
                0.0
            } else {
                flush_recursive_state_f32(y_l)
            },
            if y_r == 0.0 {
                0.0
            } else {
                flush_recursive_state_f32(y_r)
            },
        ]
    }

    #[inline(always)]
    fn step_channel(
        x: f32,
        a1: f32,
        a2: f32,
        a3: f32,
        ic1eq: &mut f32,
        ic2eq: &mut f32,
    ) -> (f32, f32) {
        let c1 = *ic1eq;
        let c2 = *ic2eq;
        let v3 = x - c2;
        let v1 = a1 * c1 + a2 * v3;
        let v2 = c2 + a2 * c1 + a3 * v3;
        *ic1eq = flush_recursive_state_f32(2.0 * v1 - c1);
        *ic2eq = flush_recursive_state_f32(2.0 * v2 - c2);
        (v1, v2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tone_svf_identity_and_filter_bands() {
        let mut p_tone = V3InterBlockToneSvfParams::identity();
        p_tone.lp_mix = 0.35;
        p_tone.bp_mix = 0.25;
        p_tone.hp_mix = 0.15;
        let mut tone = V3InterBlockToneSvfState::from_cold(&p_tone);
        tone.refresh_3khz(0.6, -0.1, [0.2, -0.1, 0.15]);
        assert_eq!(tone.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 2]);
        for n in 0..256 {
            let x = 0.6 * (n as f32 * 0.12).sin();
            let [l, r] = tone.step_stereo(x, -0.5 * x);
            assert!(l.is_finite() && r.is_finite() && l.abs() < 10.0 && r.abs() < 10.0);
        }
        tone.reconfigure_sample_rate(96_000.0);
        assert!(tone.a1 > 0.0);
        tone.reset();
        assert_eq!(tone.step_stereo(0.0, 0.0).map(f32::to_bits), [0; 2]);
    }
}
