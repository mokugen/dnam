//! Extended Stage I v2 operators (`Stage1PreDynamics::step_stereo_ext`).
//!
//! Decomposes the extended Stage I stereo pipeline into focused DSP helpers
//! with strict `0 FDIV / 0 FSQRT / 0 libm / 0 alloc`:
//! - Envelope gate
//! - Dual-release optical / sidechain compressor
//! - Sub-octave divider, Octavia rectifier & audio-rate ring modulator
//! - Envelope-followed resonant TPT state-variable filter (SVF)
//! - Stereo chorus & rotary AM/Doppler horn inertia

use super::{
    Stage1PreDynamics, exp2_pos_approx_f32, flush_recursive_state_f32, read_sample_linear,
    recip_nr2_pos_f32,
};
use crate::math::spec_math::q32_phase_to_phasor;
use crate::operators::op_m::algebraic_unit_rotor_op_m;

#[inline(always)]
fn step_svf_channel(
    x: f32,
    ic1: &mut f32,
    ic2: &mut f32,
    coeffs: [f32; 4],
    weights: [f32; 3],
) -> (f32, f32, f32, f32) {
    let [k, a1, a2, a3] = coeffs;
    let [lp_w, bp_w, hp_w] = weights;
    let v3 = x - *ic2;
    let v1 = a1 * *ic1 + a2 * v3;
    let v2 = *ic2 + a2 * *ic1 + a3 * v3;
    *ic1 = flush_recursive_state_f32(2.0 * v1 - *ic1);
    *ic2 = flush_recursive_state_f32(2.0 * v2 - *ic2);
    let hp = x - k * v1 - v2;
    let svf = lp_w * v2 + bp_w * v1 + hp_w * hp;
    (v2, v1, hp, svf)
}

impl Stage1PreDynamics {
    #[inline(always)]
    fn apply_ext_gate(&mut self, x_l: &mut f32, x_r: &mut f32, gate_param: f32) {
        if self.gate_ext_thresh <= 1.0e-6 {
            return;
        }
        let det = 0.5 * (x_l.abs() + x_r.abs());
        let a_g = if det > self.gate_ext_env {
            self.gate_atk_alpha
        } else {
            self.gate_rel_alpha
        };
        self.gate_ext_env =
            flush_recursive_state_f32(self.gate_ext_env + a_g * (det - self.gate_ext_env));
        let inv_t = self.gate_ext_inv_thresh * (1.0 - 0.5 * gate_param).clamp(0.25, 4.0);
        let u = (self.gate_ext_env * inv_t).clamp(0.0, 1.0);
        let g = u * u * (3.0 - 2.0 * u);
        *x_l *= g;
        *x_r *= g;
    }

    #[inline(always)]
    fn apply_ext_compressor(
        &mut self,
        x_l: &mut f32,
        x_r: &mut f32,
        aux_key: Option<(f32, f32)>,
        thresh_param: f32,
        slope_param: f32,
    ) {
        if self.comp_mix <= 1.0e-6 {
            return;
        }
        let sc = if self.comp_sidechain_aux_mix > 1.0e-6 {
            let (main_det, aux_det) = if let Some((al, ar)) = aux_key {
                (0.5 * (x_l.abs() + x_r.abs()), 0.5 * (al.abs() + ar.abs()))
            } else {
                let md = x_l.abs();
                let ad = x_r.abs();
                if self.comp_sidechain_aux_mix > 0.5 {
                    *x_r = *x_l;
                }
                (md, ad)
            };
            (1.0 - self.comp_sidechain_aux_mix) * main_det + self.comp_sidechain_aux_mix * aux_det
        } else {
            0.5 * (x_l.abs() + x_r.abs())
        };
        let af = if sc > self.comp_fast_env {
            self.comp_atk_alpha
        } else {
            self.comp_rel_fast_alpha
        };
        let asl = if sc > self.comp_slow_env {
            0.25 * self.comp_atk_alpha
        } else {
            self.comp_rel_slow_alpha
        };
        self.comp_fast_env =
            flush_recursive_state_f32(self.comp_fast_env + af * (sc - self.comp_fast_env));
        self.comp_slow_env =
            flush_recursive_state_f32(self.comp_slow_env + asl * (sc - self.comp_slow_env));
        let env = 0.65 * self.comp_fast_env + 0.35 * self.comp_slow_env;
        let thr = (self.comp_thresh * (1.0 - 0.4 * thresh_param).clamp(0.15, 3.0)).max(1.0e-4);
        let over = (env - thr).max(0.0);
        let slope = (self.comp_slope * (1.0 + 0.5 * slope_param).clamp(0.1, 4.0)).max(0.0);
        let gr =
            recip_nr2_pos_f32(1.0 + slope * over * recip_nr2_pos_f32(thr)) * self.comp_makeup_gain;
        let wet_m = self.comp_mix;
        let dry_m = 1.0 - wet_m;
        *x_l = dry_m * *x_l + wet_m * (gr * *x_l);
        *x_r = dry_m * *x_r + wet_m * (gr * *x_r);
    }

    #[inline(always)]
    fn apply_ext_octave_and_ringmod(
        &mut self,
        x_l: &mut f32,
        x_r: &mut f32,
        ring_param: f32,
        ext_fb_l: &mut [f32; 4],
        ext_fb_r: &mut [f32; 4],
        has_ext_fb: &mut bool,
    ) {
        let any_active = self.octave_rectify_mix > 1.0e-6
            || self.octave_sub1_mix > 1.0e-6
            || self.octave_sub2_mix > 1.0e-6
            || self.ringmod_mix > 1.0e-6;
        if !any_active && self.base_ringmod_freq_hz <= 0.0 {
            return;
        }
        let mono = 0.5 * (*x_l + *x_r);
        self.oct_tracker_lp = flush_recursive_state_f32(
            self.oct_tracker_lp + self.oct_trk_alpha * (mono - self.oct_tracker_lp),
        );
        self.oct_env_follow = flush_recursive_state_f32(
            self.oct_env_follow + self.oct_env_alpha * (mono.abs() - self.oct_env_follow),
        );
        if self.oct_tracker_lp >= 0.0 && self.oct_prev_sign < 0.0 {
            self.oct_ff1 = -self.oct_ff1;
            self.oct_sub1_count = self.oct_sub1_count.wrapping_add(1);
            if (self.oct_sub1_count & 1) == 0 {
                self.oct_ff2 = -self.oct_ff2;
            }
        }
        self.oct_prev_sign = if self.oct_tracker_lp >= 0.0 {
            1.0
        } else {
            -1.0
        };

        let rect_l = x_l.abs() - 0.5 * self.oct_env_follow;
        let rect_r = x_r.abs() - 0.5 * self.oct_env_follow;
        let sub1 = self.oct_ff1 * self.oct_env_follow;
        let sub2 = self.oct_ff2 * self.oct_env_follow;

        let mut rm_prod_l = 0.0_f32;
        let mut rm_prod_r = 0.0_f32;
        let (rm_l, rm_r) = if self.ringmod_mix > 1.0e-6 || self.base_ringmod_freq_hz > 0.0 {
            let inc = if ring_param.abs() <= 1.0e-6 {
                self.ringmod_inc_q32
            } else {
                ((self.ringmod_inc_q32 as f32)
                    * exp2_pos_approx_f32((1.0 + ring_param).clamp(0.0, 3.5))
                    * 0.5) as u32
            };
            let (rm_cos, _) = q32_phase_to_phasor(self.ringmod_phase_q32);
            self.ringmod_phase_q32 = self.ringmod_phase_q32.wrapping_add(inc);
            rm_prod_l = *x_l * rm_cos;
            rm_prod_r = *x_r * rm_cos;
            let rm_gain = self.ringmod_mix;
            if rm_gain <= 1.0e-6 {
                (0.0, 0.0)
            } else if rm_gain <= 1.0 {
                (
                    (1.0 - rm_gain) * *x_l + rm_gain * rm_prod_l - *x_l,
                    (1.0 - rm_gain) * *x_r + rm_gain * rm_prod_r - *x_r,
                )
            } else {
                (rm_gain * rm_prod_l, rm_gain * rm_prod_r)
            }
        } else {
            (0.0, 0.0)
        };

        if any_active {
            *ext_fb_l = [rect_l, sub1, sub2, rm_prod_l];
            *ext_fb_r = [rect_r, sub1, sub2, rm_prod_r];
            *has_ext_fb = true;
        }

        *x_l += self.octave_rectify_mix * rect_l
            + self.octave_sub1_mix * sub1
            + self.octave_sub2_mix * sub2
            + rm_l;
        *x_r += self.octave_rectify_mix * rect_r
            + self.octave_sub1_mix * sub1
            + self.octave_sub2_mix * sub2
            + rm_r;
    }

    #[inline(always)]
    fn apply_ext_svf(
        &mut self,
        x_l: &mut f32,
        x_r: &mut f32,
        stage_params: &[f32; 16],
        ext_fb_l: &mut [f32; 4],
        ext_fb_r: &mut [f32; 4],
        has_ext_fb: &mut bool,
    ) {
        if self.svf_mix <= 1.0e-6 {
            return;
        }
        let pre_svf_l = *x_l;
        let pre_svf_r = *x_r;
        let det = 0.5 * (x_l.abs() + x_r.abs());
        let a_svf = if det > self.svf_env_state {
            self.svf_atk_alpha
        } else {
            self.svf_rel_alpha
        };
        self.svf_env_state =
            flush_recursive_state_f32(self.svf_env_state + a_svf * (det - self.svf_env_state));
        let oct_shift = (self.svf_env_sens
            * (1.0 + 0.5 * stage_params[14]).clamp(0.0, 3.0)
            * self.svf_env_state
            + stage_params[9])
            .clamp(0.0, 3.8);
        let fc = (self.svf_base_fc_hz * exp2_pos_approx_f32(oct_shift)).clamp(20.0, 18_000.0);
        let w = (core::f32::consts::PI * fc * self.svf_inv_fs_eff).clamp(1.0e-4, 1.35);
        let (c_w, s_w) = algebraic_unit_rotor_op_m(w);
        let g = s_w * recip_nr2_pos_f32(c_w.max(0.05));
        let k = (self.svf_k_damp * (1.0 - 0.4 * stage_params[10]).clamp(0.2, 2.5)).clamp(0.05, 4.0);
        let a1 = recip_nr2_pos_f32(1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let coeffs = [k, a1, a2, a3];
        let weights = [self.svf_lp_w, self.svf_bp_w, self.svf_hp_w];
        let (v2_l, v1_l, hp_l, svf_l) = step_svf_channel(
            *x_l,
            &mut self.svf_ic1_l,
            &mut self.svf_ic2_l,
            coeffs,
            weights,
        );
        let (v2_r, v1_r, hp_r, svf_r) = step_svf_channel(
            *x_r,
            &mut self.svf_ic1_r,
            &mut self.svf_ic2_r,
            coeffs,
            weights,
        );

        if !*has_ext_fb {
            *ext_fb_l = [v2_l, v1_l, hp_l, pre_svf_l];
            *ext_fb_r = [v2_r, v1_r, hp_r, pre_svf_r];
            *has_ext_fb = true;
        }

        let dry_m = 1.0 - self.svf_mix;
        *x_l = dry_m * *x_l + self.svf_mix * svf_l;
        *x_r = dry_m * *x_r + self.svf_mix * svf_r;
    }

    #[inline(always)]
    fn apply_ext_chorus_and_rotary(
        &mut self,
        y_l: &mut f32,
        y_r: &mut f32,
        stage_params: &[f32; 16],
        ext_fb_l: &mut [f32; 4],
        ext_fb_r: &mut [f32; 4],
        has_ext_fb: &mut bool,
    ) {
        if (self.chorus_wet_mix <= 1.0e-6 && self.rotary_am_depth <= 1.0e-6)
            || self.chorus_buf_l.len() < 4_096
            || self.chorus_buf_r.len() < 4_096
        {
            return;
        }
        let pre_chorus_l = *y_l;
        let pre_chorus_r = *y_r;
        let wr = self.chorus_wr_idx;
        let mask = self.chorus_buf_l.len() - 1;
        let max_d = (mask - 4) as f32;
        self.chorus_buf_l[wr] = flush_recursive_state_f32(*y_l);
        self.chorus_buf_r[wr] = flush_recursive_state_f32(*y_r);

        let rate_scale = (1.0 + 0.5 * stage_params[15]).clamp(0.2, 4.0);
        let inc_l = ((self.chorus_inc_l_q32 as f32) * rate_scale) as u32;
        let inc_r = ((self.chorus_inc_r_q32 as f32) * rate_scale) as u32;
        let (_, ch_sin_l) = q32_phase_to_phasor(self.chorus_phase_l_q32);
        let (ch_cos_r, _) = q32_phase_to_phasor(self.chorus_phase_r_q32);
        self.chorus_phase_l_q32 = self.chorus_phase_l_q32.wrapping_add(inc_l);
        self.chorus_phase_r_q32 = self.chorus_phase_r_q32.wrapping_add(inc_r);

        let horn_target =
            (self.base_horn_freq_hz * (1.0 + stage_params[15]).clamp(0.1, 8.0)).clamp(0.0, 40.0);
        self.horn_target_hz = horn_target;
        self.horn_current_hz = flush_recursive_state_f32(
            self.horn_current_hz + self.horn_inertia_alpha * (horn_target - self.horn_current_hz),
        );
        let horn_inc = (self.horn_current_hz * self.freq_to_q32_scale) as u32;
        let (horn_cos, horn_sin) = q32_phase_to_phasor(self.horn_phase_q32);
        self.horn_phase_q32 = self.horn_phase_q32.wrapping_add(horn_inc);

        let depth_scale = if self.svf_mix <= 1.0e-6 {
            (1.0 + 0.5 * stage_params[14]).clamp(0.0, 3.0)
        } else {
            1.0
        };
        let eff_depth = self.chorus_depth_host * depth_scale;
        let d_l =
            (self.chorus_delay_host + eff_depth * ch_sin_l + self.rotary_doppler_host * horn_sin)
                .clamp(1.0, max_d);
        let d_r = (self.chorus_delay_host + eff_depth * ch_cos_r
            - self.rotary_doppler_host * horn_sin)
            .clamp(1.0, max_d);

        let mut wet_l = read_sample_linear(&self.chorus_buf_l, mask, wr, d_l);
        let mut wet_r = read_sample_linear(&self.chorus_buf_r, mask, wr, d_r);
        if self.rotary_am_depth > 1.0e-6 {
            wet_l *= 1.0 + self.rotary_am_depth * horn_cos;
            wet_r *= 1.0 - self.rotary_am_depth * horn_cos;
        }
        if !*has_ext_fb {
            *ext_fb_l = [wet_l, wet_r, pre_chorus_l, pre_chorus_r];
            *ext_fb_r = [wet_r, wet_l, pre_chorus_r, pre_chorus_l];
            *has_ext_fb = true;
        }
        let dry_c = 1.0 - self.chorus_wet_mix;
        *y_l = dry_c * *y_l + self.chorus_wet_mix * wet_l;
        *y_r = dry_c * *y_r + self.chorus_wet_mix * wet_r;
        self.chorus_wr_idx = (wr + 1) & mask;
    }

    /// Evaluates extended Stage I operators (envelope gate, dual-release optical/sidechain compressor,
    /// octave divider & audio-rate ring modulator, envelope-followed TPT SVF, core `Op-N`/`Op-M`/`Op-T`,
    /// and stereo chorus / rotary AM+Doppler) with strict `0 FDIV / 0 FSQRT / 0 libm / 0 alloc`.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_stereo_ext(
        &mut self,
        in_l: f32,
        in_r: f32,
        aux_key: Option<(f32, f32)>,
        stage_params: &[f32; 16],
    ) -> (f32, f32) {
        if !self.ext_active {
            return self.step_stereo(in_l, in_r);
        }

        let mut x_l = in_l;
        let mut x_r = in_r;

        self.apply_ext_gate(&mut x_l, &mut x_r, stage_params[0]);
        self.apply_ext_compressor(
            &mut x_l,
            &mut x_r,
            aux_key,
            stage_params[1],
            stage_params[2],
        );

        let mut ext_fb_l = [0.0_f32; 4];
        let mut ext_fb_r = [0.0_f32; 4];
        let mut has_ext_fb = false;
        self.apply_ext_octave_and_ringmod(
            &mut x_l,
            &mut x_r,
            stage_params[13],
            &mut ext_fb_l,
            &mut ext_fb_r,
            &mut has_ext_fb,
        );
        self.apply_ext_svf(
            &mut x_l,
            &mut x_r,
            stage_params,
            &mut ext_fb_l,
            &mut ext_fb_r,
            &mut has_ext_fb,
        );

        let (mut y_l, mut y_r) = self.step_stereo(x_l, x_r);
        self.apply_ext_chorus_and_rotary(
            &mut y_l,
            &mut y_r,
            stage_params,
            &mut ext_fb_l,
            &mut ext_fb_r,
            &mut has_ext_fb,
        );

        if has_ext_fb && self.pitch_blend.abs() <= 1.0e-6 {
            self.last_tracker_l = ext_fb_l;
            self.last_tracker_r = ext_fb_r;
        }

        (y_l, y_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_ext_pre() -> Stage1PreDynamics {
        let mut pre = Stage1PreDynamics {
            ext_active: true,
            op_t_bypass: true,
            ..Default::default()
        };
        pre.op_m_pre.set_pole(0, 0.85, 0.35);
        pre.op_m_pre_r.set_pole(0, 0.85, 0.35);
        pre.op_t_pre.set_target_delay(8.0, false);
        pre
    }

    #[test]
    fn test_step_stereo_ext_bypass_gate_and_compressor_modes() {
        let _audio = crate::FtzDazGuard::new();
        let params = [0.0_f32; 16];

        // Inactive ext_active delegates directly to step_stereo
        let mut pre_bypass = make_ext_pre();
        pre_bypass.ext_active = false;
        let (bl, br) = pre_bypass.step_stereo_ext(0.4, -0.2, None, &params);
        assert!(bl.is_finite() && br.is_finite());

        // Gate attack + release and non-sidechain compressor attack + release
        let mut pre = make_ext_pre();
        pre.gate_ext_thresh = 0.05;
        pre.gate_ext_inv_thresh = 20.0;
        pre.gate_atk_alpha = 0.5;
        pre.gate_rel_alpha = 0.1;
        pre.comp_mix = 0.7;
        pre.comp_sidechain_aux_mix = 0.0;
        pre.comp_thresh = 0.1;
        pre.comp_slope = 2.0;
        pre.comp_atk_alpha = 0.4;
        pre.comp_rel_fast_alpha = 0.2;
        pre.comp_rel_slow_alpha = 0.05;
        pre.comp_makeup_gain = 1.25;

        let (hl, hr) = pre.step_stereo_ext(0.8, -0.6, None, &params);
        assert!(hl.is_finite() && hr.is_finite());
        assert!(pre.gate_ext_env > 0.0 && pre.comp_fast_env > 0.0 && pre.comp_slow_env > 0.0);

        let prev_gate_env = pre.gate_ext_env;
        let prev_fast_env = pre.comp_fast_env;
        let (rl, rr) = pre.step_stereo_ext(0.01, -0.01, None, &params);
        assert!(rl.is_finite() && rr.is_finite());
        assert!(pre.gate_ext_env < prev_gate_env);
        assert!(pre.comp_fast_env < prev_fast_env);

        // Sidechain with aux_key = None and comp_sidechain_aux_mix > 0.5 mirrors x_r = x_l,
        // whereas aux_mix <= 0.5 preserves distinct channels, and Some(aux) uses external key.
        let mut pre_mirror = make_ext_pre();
        pre_mirror.comp_mix = 1.0;
        pre_mirror.comp_sidechain_aux_mix = 0.75;
        pre_mirror.comp_thresh = 0.2;
        pre_mirror.comp_slope = 1.0;
        pre_mirror.comp_atk_alpha = 0.5;
        pre_mirror.comp_makeup_gain = 1.0;
        let (ml, mr) = pre_mirror.step_stereo_ext(0.6, -0.1, None, &params);
        assert!(ml.abs() > 1.0e-3 && (ml - mr).abs() < 1.0e-5);

        pre_mirror.reset();
        pre_mirror.comp_sidechain_aux_mix = 0.25;
        let (nl, nr) = pre_mirror.step_stereo_ext(0.6, -0.6, None, &params);
        assert!((nl - nr).abs() > 1.0e-3);
        let (al, ar) = pre_mirror.step_stereo_ext(0.5, -0.25, Some((0.9, -0.9)), &params);
        assert!(al.is_finite() && ar.is_finite());
    }

    #[test]
    fn test_step_stereo_ext_octave_ringmod_svf_and_chorus_branches() {
        let _audio = crate::FtzDazGuard::new();

        // 1. Ringmod carrier running with ringmod_mix == 0.0 and non-zero stage_params[13],
        //    combined with SVF alone (!has_ext_fb before SVF, both env attack & release).
        let mut pre = make_ext_pre();
        pre.base_ringmod_freq_hz = 120.0;
        pre.ringmod_inc_q32 = 10_737_418;
        pre.ringmod_mix = 0.0;
        pre.svf_mix = 0.6;
        pre.svf_base_fc_hz = 400.0;
        pre.svf_env_sens = 2.0;
        pre.svf_k_damp = 0.8;
        pre.svf_lp_w = 0.5;
        pre.svf_bp_w = 0.3;
        pre.svf_hp_w = 0.2;
        pre.svf_atk_alpha = 0.5;
        pre.svf_rel_alpha = 0.1;
        pre.svf_inv_fs_eff = 1.0 / 48_000.0;

        let mut mod_params = [0.0_f32; 16];
        mod_params[13] = 0.5;
        let (sl, sr) = pre.step_stereo_ext(0.7, -0.4, None, &mod_params);
        assert!(sl.is_finite() && sr.is_finite());
        assert_ne!(pre.ringmod_phase_q32, 0);
        assert!(pre.last_tracker_l[3].abs() > 0.0);
        let prev_svf_env = pre.svf_env_state;
        let _ = pre.step_stereo_ext(0.05, 0.02, None, &mod_params);
        assert!(pre.svf_env_state < prev_svf_env);

        // 2. Octave dividers (crossing zero to flip both ff1 and ff2) + ringmod mix <= 1 and > 1
        let mut pre_oct = make_ext_pre();
        pre_oct.octave_rectify_mix = 0.4;
        pre_oct.octave_sub1_mix = 0.3;
        pre_oct.octave_sub2_mix = 0.2;
        pre_oct.ringmod_mix = 0.5;
        pre_oct.ringmod_inc_q32 = 20_000_000;
        pre_oct.oct_trk_alpha = 0.8;
        pre_oct.oct_env_alpha = 0.5;
        pre_oct.oct_ff1 = 1.0;
        pre_oct.oct_ff2 = 1.0;
        pre_oct.oct_prev_sign = 1.0;
        for sample in [0.5, -0.5, 0.5, -0.5, 0.5] {
            let (ol, or) = pre_oct.step_stereo_ext(sample, -sample * 0.5, None, &[0.0_f32; 16]);
            assert!(ol.is_finite() && or.is_finite());
        }
        pre_oct.ringmod_mix = 1.5;
        let (ol2, or2) = pre_oct.step_stereo_ext(0.4, -0.4, None, &[0.0_f32; 16]);
        assert!(ol2.is_finite() && or2.is_finite());

        // 3. Chorus & Rotary alone (svf_mix = 0.0 => !has_ext_fb & depth_scale modulated),
        //    plus a pass with svf_mix > 0.0 so depth_scale == 1.0 branch executes.
        let mut pre_chorus = make_ext_pre();
        pre_chorus.chorus_wet_mix = 0.5;
        pre_chorus.chorus_delay_host = 24.0;
        pre_chorus.chorus_depth_host = 6.0;
        pre_chorus.chorus_inc_l_q32 = 100_000;
        pre_chorus.chorus_inc_r_q32 = 120_000;
        pre_chorus.base_horn_freq_hz = 4.0;
        pre_chorus.horn_inertia_alpha = 0.2;
        pre_chorus.freq_to_q32_scale = ((1u64 << 32) as f64 / 48_000.0) as f32;
        pre_chorus.rotary_doppler_host = 2.0;
        pre_chorus.rotary_am_depth = 0.35;
        pre_chorus.chorus_buf_l = vec![0.0; 4_096];
        pre_chorus.chorus_buf_r = vec![0.0; 4_096];

        let mut ch_params = [0.0_f32; 16];
        ch_params[14] = 0.4;
        ch_params[15] = 0.5;
        let (cl, cr) = pre_chorus.step_stereo_ext(0.6, -0.6, None, &ch_params);
        assert!(cl.is_finite() && cr.is_finite());
        assert_eq!(pre_chorus.chorus_wr_idx, 1);

        pre_chorus.svf_mix = 0.4;
        pre_chorus.svf_base_fc_hz = 600.0;
        pre_chorus.svf_k_damp = 1.0;
        pre_chorus.svf_lp_w = 1.0;
        pre_chorus.svf_inv_fs_eff = 1.0 / 48_000.0;
        let (cl2, cr2) = pre_chorus.step_stereo_ext(0.3, -0.3, None, &ch_params);
        assert!(cl2.is_finite() && cr2.is_finite());
        assert_eq!(pre_chorus.chorus_wr_idx, 2);
    }
}
