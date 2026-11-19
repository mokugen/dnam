//! $3\text{ kHz}$ Control-Bus tick execution, static basis evaluation, tank modulation, and micro-ramping.

pub mod simd_and_clip;

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};

use simd_and_clip::read_vec4_le;
pub use simd_and_clip::trust_clip_outward_4d;

use super::ControlBus;
use super::cp1_and_uint::{eval_uint_base_4x8, eval_uint_ultra_delta_3x4x8};
use crate::control::ANTI_ZIPPER_SNAP_EPS;
use crate::control::record::LatentAccumulator64B;

impl<B: MathBackend> ControlBus<B> {
    /// Executes one $3\text{ kHz}$ Control-Bus tick (`0 FDIV`, `0 FSQRT`, `0` heap allocations).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline]
    pub fn tick_3khz(&mut self, audio_abs_energy: f32, arena_bytes: &[u8]) -> bool {
        self.tick_3khz_with_detectors(audio_abs_energy, None, arena_bytes)
    }

    /// Executes one $3\text{ kHz}$ Control-Bus tick with optional 4-band stereo-linked ballistic
    /// detector states `[e_low, e_mid, e_high, e_aux]` from `Stage 0` (`r[dsp.ctrl.modulators.dynamic]`).
    #[inline]
    pub fn tick_3khz_with_detectors(
        &mut self,
        audio_abs_energy: f32,
        stage0_detectors: Option<[f32; 4]>,
        arena_bytes: &[u8],
    ) -> bool {
        if let Some(det) = stage0_detectors {
            for b in 0..4 {
                self.env_states[b] = flush_recursive_state_f32(det[b]);
            }
        } else {
            for b in 0..3 {
                let pole = self.env_poles[b];
                self.env_states[b] = flush_recursive_state_f32(
                    pole * self.env_states[b] + (1.0 - pole) * audio_abs_energy,
                );
            }
            // Causal attack onset detector (fast high-band envelope minus slow low-band envelope)
            self.env_states[3] = (self.env_states[2] - self.env_states[0]).max(0.0);
        }
        for j in 0..2 {
            let angle_bits = self.lfo_angle_bits[j];
            if (angle_bits & 0x7fff_ffff) == 0 {
                let c_hold = self.lfo_phasors[j][0];
                let two_x = 2.0 * c_hold;
                let t2 = two_x * c_hold - 1.0;
                let t3 = two_x * t2 - c_hold;
                let t4 = two_x * t3 - t2;
                let base = j * 4;
                self.lfo_harmonics[base] = c_hold;
                self.lfo_harmonics[base + 1] = t2;
                self.lfo_harmonics[base + 2] = t3;
                self.lfo_harmonics[base + 3] = t4;
                continue;
            }
            if self.lfo_cached_angle_bits[j] != angle_bits {
                self.lfo_cached_angle_bits[j] = angle_bits;
                self.lfo_cached_q192_inc[j] =
                    crate::math::q0_192_phase_increment_from_f32_bits(angle_bits);
            }
            let (c_next, s_next) = if let Some((neg, mag)) = self.lfo_cached_q192_inc[j] {
                self.lfo_phases_q192[j].advance_signed(neg, mag);
                self.lfo_phases_q192[j].to_phasor()
            } else {
                let c_prev = self.lfo_phasors[j][0];
                let s_prev = self.lfo_phasors[j][1];
                let c_inc = self.lfo_rotors[j][0];
                let s_inc = self.lfo_rotors[j][1];

                let c_hat = c_prev * c_inc - s_prev * s_inc;
                let s_hat = s_prev * c_inc + c_prev * s_inc;
                let r2 = c_hat * c_hat + s_hat * s_hat;
                let y1 = 1.5 - 0.5 * r2;
                (c_hat * y1, s_hat * y1)
            };
            self.lfo_phasors[j] = [
                flush_recursive_state_f32(c_next),
                flush_recursive_state_f32(s_next),
            ];

            let t1 = c_next;
            let two_x = 2.0 * t1;
            let t2 = two_x * t1 - 1.0;
            let t3 = two_x * t2 - t1;
            let t4 = two_x * t3 - t2;
            let base = j * 4;
            self.lfo_harmonics[base] = t1;
            self.lfo_harmonics[base + 1] = t2;
            self.lfo_harmonics[base + 2] = t3;
            self.lfo_harmonics[base + 3] = t4;
        }

        if !self.is_dirty {
            self.evaluate_tank_modulation(arena_bytes);
            return false;
        }

        let mut still_slewing = false;
        let pole = self.anti_zipper_pole;
        let one_minus_pole = 1.0 - pole;
        for p in 0..self.num_controls {
            if self.descriptors[p].ctrl_kind == 0 {
                let target = self.target_u_norm[p];
                let curr = self.current_u_norm[p];
                let diff = target - curr;
                if diff.abs() > ANTI_ZIPPER_SNAP_EPS {
                    let next = pole * curr + one_minus_pole * target;
                    if (target - next).abs() <= ANTI_ZIPPER_SNAP_EPS {
                        self.current_u_norm[p] = flush_recursive_state_f32(target);
                    } else {
                        self.current_u_norm[p] = flush_recursive_state_f32(next);
                        still_slewing = true;
                    }
                } else {
                    self.current_u_norm[p] = flush_recursive_state_f32(target);
                }
            }
        }

        self.evaluate_static_control_basis(arena_bytes);

        self.is_dirty = still_slewing;
        true
    }

    /// Snaps `current_u_norm` to `target_u_norm`, evaluates `z_head`, `z_head_ultra`, and
    /// `evaluate_tank_modulation` at the current control positions and LFO/envelope states
    /// **without** advancing LFO phasors or envelope followers, and clears `is_dirty`.
    #[inline]
    pub fn refresh_static_state(&mut self, arena_bytes: &[u8]) {
        self.unified_w_ctrl = crate::stages::weights::has_pipeline_flag(
            arena_bytes,
            crate::stages::weights::PIPE_FLAG_UNIFIED_W_CTRL,
        );
        for (current, &target) in self.current_u_norm.iter_mut().zip(&self.target_u_norm) {
            *current = flush_recursive_state_f32(target);
        }
        if self.num_contributors == 0 {
            self.latent_acc.reset();
            self.s_vx_ultra = [0.0; 4];
            self.s_v2x2_ultra = [0.0; 4];
            for acc in &mut self.exact_fm_acc {
                acc.reset();
            }
            let n = self
                .num_controls
                .min(self.raw_values.len())
                .min(crate::control::plane::MAX_EXACT_CONTRIBUTORS);
            let mut def_sw = [0_u8; crate::control::plane::MAX_EXACT_CONTRIBUTORS];
            for (p, slot) in def_sw[..n].iter_mut().enumerate() {
                if self.descriptors[p].ctrl_kind == 1 {
                    *slot = (self.descriptors[p].default_val.max(0.0) + 0.5) as u8;
                }
            }
            for p in 0..n {
                if self.descriptors[p].ctrl_kind == 0 {
                    let def_val = self.descriptors[p].default_val;
                    let (_, u_def, _) =
                        self.descriptors[p].clamp_and_normalize(def_val, &def_sw[..n]);
                    self.prev_eval_u_norm[p] = flush_recursive_state_f32(u_def);
                    let lob = self.descriptors[p].eval_lobatto_basis(self.prev_eval_u_norm[p]);
                    self.prev_eval_psi[p] = lob[0] + lob[1] + lob[2];
                } else {
                    self.prev_eval_u_norm[p] = 0.0;
                    self.prev_eval_psi[p] = 0.0;
                }
            }
        }
        self.evaluate_static_control_basis(arena_bytes);
        self.is_dirty = false;
    }

    #[inline]
    pub(super) fn evaluate_static_control_basis(&mut self, arena_bytes: &[u8]) {
        if self.num_contributors > 0 {
            let mut z_main_base = [0.0_f32; 8];
            let mut z_main_ultra = [0.0_f32; 8];
            let mut s_emb = [0.0_f32; 3];
            for c in 0..self.num_contributors.min(self.contributors.len()) {
                let slot = &self.contributors[c];
                let p = slot.control_idx as usize;
                let effective_kind = if slot.ctrl_kind != 0 {
                    slot.ctrl_kind
                } else if p < self.num_controls {
                    self.descriptors[p].ctrl_kind
                } else {
                    0
                };
                let (lob, scalar) = match effective_kind {
                    0 if p < self.num_controls => {
                        let u = self.current_u_norm[p];
                        (self.descriptors[p].eval_lobatto_basis(u), u)
                    }
                    1 if p < self.num_controls => {
                        let active_opt = self.active_switch_options[p];
                        let def_opt = (self.descriptors[p].default_val.max(0.0) + 0.5) as u8;
                        let s = if slot.option_idx == 0xFF {
                            if active_opt != def_opt { 1.0 } else { 0.0 }
                        } else if active_opt != def_opt && active_opt == slot.option_idx {
                            1.0
                        } else {
                            0.0
                        };
                        ([s, 0.0, 0.0], s)
                    }
                    2 => {
                        let q = slot.partner_control_id as usize;
                        if p < self.num_controls && q < self.num_controls {
                            let u_p = self.current_u_norm[p];
                            let u_q = self.current_u_norm[q];
                            let lob_p = self.descriptors[p].eval_lobatto_basis(u_p);
                            let lob_q = self.descriptors[q].eval_lobatto_basis(u_q);
                            let prod = lob_p[0] * lob_q[0];
                            ([prod, lob_p[1] * lob_q[0], lob_p[0] * lob_q[1]], prod)
                        } else {
                            ([0.0; 3], 0.0)
                        }
                    }
                    _ => ([0.0; 3], 0.0),
                };
                for k in 0..3 {
                    let lk = lob[k];
                    if lk.abs() > 1e-12 {
                        for d in 0..8 {
                            z_main_base[d] += slot.main_base[k][d] * lk;
                            if slot.has_ultra {
                                z_main_ultra[d] += slot.main_ultra[k][d] * lk;
                            }
                        }
                    }
                }
                let has_emb = slot.emb.iter().any(|&w| w.abs() > 1e-12);
                let z_c = if has_emb {
                    slot.emb[0] * lob[0] + slot.emb[1] * lob[1] + slot.emb[2] * lob[2]
                } else {
                    scalar
                };
                for e in 0..3 {
                    s_emb[e] += slot.emb[e] * scalar;
                }
                if c < self.exact_fm_acc[0].count {
                    for f in 0..4 {
                        let _ = self.exact_fm_acc[f]
                            .update_contributor_rt(c, (slot.v_base[f] * z_c).to_bits());
                        let v_u = if slot.has_ultra {
                            slot.v_ultra[f] * z_c
                        } else {
                            0.0
                        };
                        let _ = self.exact_fm_acc[4 + f].update_contributor_rt(c, v_u.to_bits());
                    }
                }
            }
            let mut z_fm_base = [0.0_f32; 4];
            let mut z_fm_ultra = [0.0_f32; 4];
            for f in 0..4 {
                if let Some(bits) = self.exact_fm_acc[f].twice_pairs.to_interaction_f32_bits() {
                    z_fm_base[f] = f32::from_bits(bits);
                }
                if let Some(bits) = self.exact_fm_acc[4 + f]
                    .twice_pairs
                    .to_interaction_f32_bits()
                {
                    z_fm_ultra[f] = f32::from_bits(bits);
                }
            }
            let (uint_base, has_uint_base) = eval_uint_base_4x8(arena_bytes, z_fm_base, s_emb);
            let mut z_base_8 = [0.0_f32; 8];
            for d in 0..8 {
                z_base_8[d] = z_main_base[d] + uint_base[d];
            }
            if !has_uint_base {
                for f in 0..4 {
                    z_base_8[f] += z_fm_base[f];
                    z_base_8[4 + f] += z_fm_base[f];
                }
            }

            if arena_bytes.len() >= crate::stages::weights::ARENA_ULTRA_BYTES {
                let (delta_base_8, uint_ultra_8, has_uint_ultra) =
                    eval_uint_ultra_delta_3x4x8(arena_bytes, z_fm_base, z_fm_ultra, s_emb);
                for d in 0..8 {
                    z_base_8[d] += delta_base_8[d];
                }
                let mut z_ultra_8 = [0.0_f32; 8];
                for d in 0..8 {
                    z_ultra_8[d] = z_main_ultra[d] + uint_ultra_8[d];
                }
                if !has_uint_ultra {
                    for f in 0..4 {
                        z_ultra_8[f] += z_fm_ultra[f];
                        z_ultra_8[4 + f] += z_fm_ultra[f];
                    }
                }
                let mut z_head_u4 = [z_ultra_8[0], z_ultra_8[1], z_ultra_8[2], z_ultra_8[3]];
                let mut z_stage_u4 = [z_ultra_8[4], z_ultra_8[5], z_ultra_8[6], z_ultra_8[7]];
                trust_clip_outward_4d(&mut z_head_u4, self.trust_radius_mh);
                trust_clip_outward_4d(&mut z_stage_u4, self.trust_radius_mh);
                self.z_head_ultra = z_head_u4;
                self.z_stage_ultra = z_stage_u4;
            } else {
                self.z_head_ultra = [0.0; 4];
                self.z_stage_ultra = [0.0; 4];
            }
            let mut z_head_4 = [z_base_8[0], z_base_8[1], z_base_8[2], z_base_8[3]];
            let mut z_stage_4 = [z_base_8[4], z_base_8[5], z_base_8[6], z_base_8[7]];
            trust_clip_outward_4d(&mut z_head_4, self.trust_radius_mh);
            trust_clip_outward_4d(&mut z_stage_4, self.trust_radius_mh);
            self.z_head = z_head_4;
            self.ctrl_basis_4 = z_stage_4;
            self.evaluate_tank_modulation(arena_bytes);
            return;
        }

        // Apply Theorem 3 O(1) Incremental Single-Knob Delta Update Law over any moved controls p in 0..P
        for p in 0..self.num_controls {
            let u_new = self.current_u_norm[p];
            let u_old = self.prev_eval_u_norm[p];
            let psi_new = if self.descriptors[p].ctrl_kind == 0 {
                let lob = self.descriptors[p].eval_lobatto_basis(u_new);
                lob[0] + lob[1] + lob[2]
            } else {
                let active_opt = self.active_switch_options[p];
                let def_opt = (self.descriptors[p].default_val.max(0.0) + 0.5) as u8;
                if active_opt != def_opt { 1.0 } else { 0.0 }
            };
            let psi_old = self.prev_eval_psi[p];
            let delta_psi = psi_new - psi_old;
            if (u_new - u_old).abs() > 1e-12 || delta_psi.abs() > 1e-12 {
                let proj = &self.projections[p];
                self.latent_acc.update_delta(proj, u_old, u_new, delta_psi);
                LatentAccumulator64B::update_delta_ultra(
                    proj,
                    u_old,
                    u_new,
                    &mut self.s_vx_ultra,
                    &mut self.s_v2x2_ultra,
                );
                if p < self.exact_fm_acc[0].count {
                    for f in 0..4 {
                        let _ = self.exact_fm_acc[f]
                            .update_contributor_rt(p, (proj.v_factors[f] * u_new).to_bits());
                        let _ = self.exact_fm_acc[4 + f]
                            .update_contributor_rt(p, (proj.v_factors[4 + f] * u_new).to_bits());
                    }
                }
                self.prev_eval_u_norm[p] = u_new;
                self.prev_eval_psi[p] = psi_new;
            }
        }
        let mut z_acc_8 = self.latent_acc.evaluate_z_ctrl();
        for f in 0..4 {
            if let Some(bits) = self.exact_fm_acc[f].twice_pairs.to_interaction_f32_bits() {
                z_acc_8[4 + f] = f32::from_bits(bits);
            }
        }
        let mut z_acc_ultra_4 =
            LatentAccumulator64B::evaluate_z_ultra_fm(&self.s_vx_ultra, &self.s_v2x2_ultra);
        for f in 0..4 {
            if let Some(bits) = self.exact_fm_acc[4 + f]
                .twice_pairs
                .to_interaction_f32_bits()
            {
                z_acc_ultra_4[f] = f32::from_bits(bits);
            }
        }

        let mut z_head_4 = [z_acc_8[0], z_acc_8[1], z_acc_8[2], z_acc_8[3]];
        let mut z_stage_4 = [z_acc_8[4], z_acc_8[5], z_acc_8[6], z_acc_8[7]];
        if z_stage_4.iter().all(|&v| v.abs() <= 1e-12) && self.num_controls > 0 {
            let lob = self.descriptors[0].eval_lobatto_basis(self.current_u_norm[0]);
            z_stage_4 = [lob[0], lob[1], lob[2], 0.0];
        }
        trust_clip_outward_4d(&mut z_head_4, self.trust_radius_mh);
        trust_clip_outward_4d(&mut z_stage_4, self.trust_radius_mh);
        self.z_head = z_head_4;
        self.ctrl_basis_4 = z_stage_4;

        if arena_bytes.len() >= crate::stages::weights::ARENA_ULTRA_BYTES {
            trust_clip_outward_4d(&mut z_acc_ultra_4, self.trust_radius_mh);
            self.z_head_ultra = z_acc_ultra_4;
            self.z_stage_ultra = [0.0; 4];
        } else {
            self.z_head_ultra = [0.0; 4];
            self.z_stage_ultra = [0.0; 4];
        }

        self.evaluate_tank_modulation(arena_bytes);
    }

    /// Evaluates the $3\text{ kHz}$ `8 × 32` / `16 × 16` Control/LFO/Envelope $\to$ Stage & Spatial Tank
    /// state-space projection matrix `W_ctrl` (`3,520..4,544` in `Arena 0`, plus
    /// `16 × 4` Ultra delta at `21,120..21,376` in `Arena 3`, `0 FDIV`, `0 FSQRT`).
    #[inline]
    pub(super) fn evaluate_tank_modulation(&mut self, arena_bytes: &[u8]) {
        const W_CTRL_TANK_BASE: usize = crate::stages::weights::ARENA0_W_CTRL_OFFSET;
        const W_CTRL_ULTRA_BASE: usize = crate::stages::weights::ARENA3_W_CTRL_ULTRA_OFFSET;
        if arena_bytes.len() < W_CTRL_TANK_BASE + 1024 {
            self.tank_mod_dirty = false;
            return;
        }

        if self.unified_w_ctrl {
            let has_z_ctrl = self.z_head.iter().any(|&v| v != 0.0)
                || self.ctrl_basis_4.iter().any(|&v| v != 0.0);
            let has_ultra_ctrl = arena_bytes.len() >= W_CTRL_ULTRA_BASE + 256
                && self.z_stage_ultra.iter().any(|&v| v != 0.0);
            let q0 = B::Vec4::load(&self.z_head);
            let q1 = B::Vec4::load(&self.ctrl_basis_4);
            let q2_arr = [
                self.lfo_harmonics[0] + 0.35 * self.lfo_harmonics[1],
                self.lfo_harmonics[4] + 0.35 * self.lfo_harmonics[5],
                self.lfo_phasors[0][0],
                self.lfo_phasors[0][1],
            ];
            let q2 = B::Vec4::load(&q2_arr);
            let q3 = B::Vec4::load(&self.env_states);
            let qu = B::Vec4::load(&self.z_stage_ultra);
            for r in 0..16 {
                let row_off = W_CTRL_TANK_BASE + r * 64;
                let mut v_acc = read_vec4_le::<B>(arena_bytes, row_off + 32)
                    .mul(q2)
                    .fma(read_vec4_le::<B>(arena_bytes, row_off + 48), q3);
                if has_z_ctrl {
                    v_acc = v_acc
                        .fma(read_vec4_le::<B>(arena_bytes, row_off), q0)
                        .fma(read_vec4_le::<B>(arena_bytes, row_off + 16), q1);
                }
                if has_ultra_ctrl {
                    v_acc = v_acc.fma(
                        read_vec4_le::<B>(arena_bytes, W_CTRL_ULTRA_BASE + r * 16),
                        qu,
                    );
                }
                self.stage_targets[r] = v_acc.horizontal_sum();
            }
            self.tank_delay_mod_samples =
                [self.stage_targets[12], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            self.tank_gain_targets = [
                self.stage_targets[13],
                self.stage_targets[13],
                self.stage_targets[13],
                self.stage_targets[13],
                0.0,
                0.0,
                0.0,
                0.0,
            ];
            self.tank_state_mod = [0.0, self.stage_targets[11], 0.0, 0.0];
            self.tank_mod_dirty = true;
            return;
        }

        let lfo0 =
            self.lfo_harmonics[0] + 0.35 * self.lfo_harmonics[1] + 0.15 * self.lfo_harmonics[2];
        let lfo1 =
            self.lfo_harmonics[4] + 0.35 * self.lfo_harmonics[5] + 0.15 * self.lfo_harmonics[6];
        let env_sustain = 0.5 * (self.env_states[0] + self.env_states[1]);
        let env_onset = self.env_states[3];
        let q8 = [
            self.ctrl_basis_4[0],
            self.ctrl_basis_4[1],
            self.ctrl_basis_4[2],
            self.ctrl_basis_4[3],
            lfo0,
            lfo1,
            env_sustain,
            env_onset,
        ];

        let mut acc0 = B::Vec4::zero();
        let mut acc2 = B::Vec4::zero();
        let mut acc7 = B::Vec4::zero();
        for (row, &q_val) in q8.iter().enumerate() {
            if q_val.abs() > 1e-12 {
                let q_vec = B::Vec4::splat(q_val);
                let row_off = W_CTRL_TANK_BASE + row * 128;
                acc0 = acc0.fma(q_vec, read_vec4_le::<B>(arena_bytes, row_off));
                acc2 = acc2.fma(q_vec, read_vec4_le::<B>(arena_bytes, row_off + 32));
                acc7 = acc7.fma(q_vec, read_vec4_le::<B>(arena_bytes, row_off + 112));
            }
        }

        let mut d0 = [0.0f32; 4];
        acc0.store(&mut d0);
        self.tank_delay_mod_samples[0] = d0[0];
        self.tank_delay_mod_samples[1] = d0[1];
        self.tank_delay_mod_samples[2] = d0[2];
        self.tank_delay_mod_samples[3] = d0[3];

        let mut g0 = [0.0f32; 4];
        acc2.store(&mut g0);
        self.tank_gain_targets[0] = g0[0];
        self.tank_gain_targets[1] = g0[1];
        self.tank_gain_targets[2] = g0[2];
        self.tank_gain_targets[3] = g0[3];

        acc7.store(&mut self.tank_state_mod);

        // Also evaluate 16x16 + 16x4 stage dispatch into stage_targets (`r[dsp.ctrl.modulators.dynamic]`)
        let has_z_ctrl =
            self.z_head.iter().any(|&v| v != 0.0) || self.ctrl_basis_4.iter().any(|&v| v != 0.0);
        let has_ultra_ctrl = arena_bytes.len() >= W_CTRL_ULTRA_BASE + 256
            && self.z_stage_ultra.iter().any(|&v| v != 0.0);
        let q0 = B::Vec4::load(&self.z_head);
        let q1 = B::Vec4::load(&self.ctrl_basis_4);
        let q2_arr = [
            self.lfo_phasors[0][0],
            self.lfo_phasors[0][1],
            self.lfo_harmonics[1],
            self.lfo_harmonics[5],
        ];
        let q2 = B::Vec4::load(&q2_arr);
        let q3 = B::Vec4::load(&self.env_states);
        let qu = B::Vec4::load(&self.z_stage_ultra);
        let r_range =
            if self.live_analog_drift_enabled && !self.offline_bounce && self.sigma_drift > 0.0 {
                2..13
            } else {
                3..12
            };
        for r in r_range {
            let row_off = W_CTRL_TANK_BASE + r * 64;
            let mut v_acc = read_vec4_le::<B>(arena_bytes, row_off + 32)
                .mul(q2)
                .fma(read_vec4_le::<B>(arena_bytes, row_off + 48), q3);
            if has_z_ctrl {
                v_acc = v_acc
                    .fma(read_vec4_le::<B>(arena_bytes, row_off), q0)
                    .fma(read_vec4_le::<B>(arena_bytes, row_off + 16), q1);
            }
            if has_ultra_ctrl {
                v_acc = v_acc.fma(
                    read_vec4_le::<B>(arena_bytes, W_CTRL_ULTRA_BASE + r * 16),
                    qu,
                );
            }
            self.stage_targets[r] = v_acc.horizontal_sum();
        }

        if self.live_analog_drift_enabled && !self.offline_bounce && self.sigma_drift > 0.0 {
            let rho = self.rho_drift_host.clamp(0.0, 0.9999);
            let scale = self.sigma_drift * (1.0 - rho);
            for d in 0..4 {
                let mut x = self.drift_prng_state;
                if x == 0 {
                    x = Self::DEFAULT_DRIFT_PRNG_SEED;
                }
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.drift_prng_state = x;
                let rand_u32 = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;
                let xi = ((rand_u32 >> 8) as f32) * (2.0 / 16_777_216.0) - 1.0;
                self.ou_drift_state[d] =
                    flush_recursive_state_f32(rho * self.ou_drift_state[d] + scale * xi);
            }
            self.stage_targets[2] += self.ou_drift_state[0];
            self.stage_targets[5] += self.ou_drift_state[1];
            self.stage_targets[8] += 0.25 * self.ou_drift_state[2];
            self.stage_targets[12] += 0.25 * self.ou_drift_state[3];
        } else {
            self.ou_drift_state = [0.0; 4];
        }

        self.tank_mod_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ARENA_OFFSET_LITE_END;
    use crate::control::{ConditionalClampRule, ControlClampDescriptor};
    use crate::math::ScalarBackend;
    use alloc::boxed::Box;
    use alloc::vec;
    use proptest::prelude::*;

    #[test]
    fn test_static_basis_edge_kinds_and_zero_prng_seed_drift() {
        let _audio = crate::FtzDazGuard::new();
        let mut arena = vec![0u8; crate::stages::weights::ARENA_ULTRA_BYTES];
        let flags = crate::stages::weights::PIPE_MAGIC_HI
            | crate::stages::weights::PIPE_FLAG_UNIFIED_W_CTRL;
        arena[crate::stages::weights::ARENA0_PIPELINE_FLAGS_OFFSET
            ..crate::stages::weights::ARENA0_PIPELINE_FLAGS_OFFSET + 4]
            .copy_from_slice(&flags.to_le_bytes());

        let mut bus = ControlBus::<ScalarBackend> {
            unified_w_ctrl: true,
            num_controls: 2,
            ..Default::default()
        };
        bus.descriptors[0] = ControlClampDescriptor {
            ctrl_kind: 0,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            ..ControlClampDescriptor::empty()
        };
        bus.descriptors[1] = ControlClampDescriptor {
            ctrl_kind: 1,
            lobatto_degree: 1,
            active_options_mask: 0b11,
            default_val: 1.0,
            ..ControlClampDescriptor::empty()
        };
        bus.active_switch_options[1] = 0; // != default (1) -> active delta 1.0
        bus.target_u_norm[0] = 0.25;
        bus.current_u_norm[0] = 0.25;
        bus.prev_eval_u_norm[0] = 0.25;

        // Exercise out-of-range range slot (kind 0, p >= num_controls), whole switch delta (kind 1, 0xFF),
        // out-of-range bilinear slot (kind 2, q >= num_controls), valid bilinear slot, and unknown kind.
        bus.num_contributors = 5;
        bus.contributors[0] = crate::control::Cp1ContributorSlot {
            lanes: 1,
            ctrl_kind: 0,
            control_idx: 200,
            ..Default::default()
        };
        bus.contributors[1] = crate::control::Cp1ContributorSlot {
            lanes: 1,
            ctrl_kind: 1,
            control_idx: 1,
            option_idx: 0xFF,
            ..Default::default()
        };
        bus.contributors[2] = crate::control::Cp1ContributorSlot {
            lanes: 1,
            ctrl_kind: 2,
            control_idx: 0,
            partner_control_id: 200,
            ..Default::default()
        };
        bus.contributors[3] = crate::control::Cp1ContributorSlot {
            lanes: 1,
            ctrl_kind: 2,
            control_idx: 0,
            partner_control_id: 1,
            v_base: [0.1, 0.2, 0.3, 0.0],
            v_ultra: [0.05; 4],
            has_ultra: true,
            ..Default::default()
        };
        bus.contributors[4] = crate::control::Cp1ContributorSlot {
            lanes: 1,
            ctrl_kind: 9,
            ..Default::default()
        };

        bus.evaluate_static_control_basis(&arena);
        bus.unified_w_ctrl = false;
        bus.active_switch_options[1] = 1; // == default -> 0.0 branch
        bus.live_analog_drift_enabled = true;
        bus.offline_bounce = false;
        bus.sigma_drift = 0.02;
        bus.rho_drift_host = 0.9;
        bus.drift_prng_state = 0;
        bus.evaluate_static_control_basis(&arena);
        assert_ne!(bus.drift_prng_state, 0);
        for v in bus.ou_drift_state {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_control_bus_slewing_and_trust_region_softclip() {
        let _audio = crate::FtzDazGuard::new();
        let mut arena0 = vec![0u8; crate::stages::weights::ARENA_ULTRA_BYTES];
        arena0[3_008 + 16..3_008 + 20].copy_from_slice(&0.95_f32.to_le_bytes());
        arena0[4_544..4_548].copy_from_slice(&0.5_f32.to_le_bytes());
        // Populate cp1_lanes_dir at 3,044 with 2 inline CP1 contributors (ctrl 0: 3-lane Range, ctrl 1: 1-lane Switch)
        arena0[3_044..3_046].copy_from_slice(&2_u16.to_le_bytes());
        arena0[3_046] = 3;
        arena0[3_047] = (1 << 4) | 1;
        for off in (3_200..3_520).step_by(4) {
            arena0[off..off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
        }
        // Populate inline Ultra CP1 payload (21,760..23,872) with large weights to exercise outward_checked_trust_clip
        for off in (21_760..23_872).step_by(4) {
            arena0[off..off + 4].copy_from_slice(&2.0_f32.to_le_bytes());
        }
        let mut cb = Box::new(ControlBus::<ScalarBackend>::default());
        cb.load_from_arena0(&arena0, 2, 1.0);
        cb.descriptors[0] = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 2,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0],
            _reserved: 0,
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        cb.descriptors[1] = ControlClampDescriptor {
            ctrl_kind: 1,
            lobatto_degree: 1,
            active_options_mask: 0b0010,
            default_val: 0.0,
            safe_min_base: 0.0,
            safe_max_base: 1.0,
            scale_norm: 1.0,
            offset_norm: 0.0,
            anchor_nodes_norm: [0.0; 3],
            _reserved: 0,
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        cb.num_controls = 2;
        cb.set_control_value(0, 8.0);
        cb.set_control_value(99, 10.0);
        cb.set_control_immediate(99, 10.0);
        assert!(cb.tick_3khz(0.5, &arena0));

        cb.anti_zipper_pole = 0.0;
        cb.set_control_value(0, 10.0);
        cb.set_control_immediate(1, 0.0);
        assert_eq!(cb.clamped_values[1], 1.0);
        assert!(cb.tick_3khz(0.5, &arena0));
        assert!(!cb.tick_3khz(0.5, &arena0));

        // Also test canonical R17-E2 3-byte cp1_lanes_dir encoding: (control_id, packed_kind_and_lanes, option_or_partner)
        let mut arena_3b = vec![0u8; crate::stages::weights::ARENA_ULTRA_BYTES];
        arena_3b[3_044] = 3;
        // Slot 0: control 0, Range (kind 0), 2 lanes
        arena_3b[3_048] = 0;
        arena_3b[3_049] = 2;
        arena_3b[3_050] = 0;
        // Slot 1: control 1, Switch (kind 1), 1 lane, option 1
        arena_3b[3_051] = 1;
        arena_3b[3_052] = (1 << 4) | 1;
        arena_3b[3_053] = 1;
        // Slot 2: control 0 x control 2 Cross-term (kind 2), 1 lane, partner 2
        arena_3b[3_054] = 0;
        arena_3b[3_055] = (2 << 4) | 1;
        arena_3b[3_056] = 2;
        for off in (3_072..3_200).step_by(4) {
            arena_3b[off..off + 4].copy_from_slice(&0.15_f32.to_le_bytes());
        }
        for off in (3_200..3_520).step_by(4) {
            arena_3b[off..off + 4].copy_from_slice(&0.25_f32.to_le_bytes());
        }
        for off in (21_376..22_000).step_by(4) {
            arena_3b[off..off + 4].copy_from_slice(&0.10_f32.to_le_bytes());
        }
        cb.ensure_control_capacity(3);
        cb.descriptors[2] = cb.descriptors[0];
        cb.num_contributors = 0;
        cb.load_from_arena0(&arena_3b, 3, 1.0);
        assert_eq!(cb.num_contributors, 3);
        cb.set_control_immediate(0, 8.0);
        cb.set_control_immediate(1, 1.0);
        cb.set_control_immediate(2, 8.0);
        cb.refresh_static_state(&arena_3b);
        assert!(cb.z_head[0].abs() > 0.0);
        assert!(cb.z_head_ultra[0].abs() > 0.0);

        // Short arena slice exercises early return in evaluate_tank_modulation
        let short_arena = [0u8; 1024];
        cb.refresh_static_state(&short_arena);
        assert!(!cb.tank_mod_dirty);

        let q16 = ControlBus::<ScalarBackend>::assemble_q16(&[1.0; 8], &[2.0; 8]);
        assert_eq!(&q16[0..8], &[1.0; 8]);
        assert_eq!(&q16[8..16], &[2.0; 8]);

        // Verify live analog drift is active when live_analog_drift_enabled && !offline_bounce,
        // and strictly suppressed (0-ULP deterministic) when offline_bounce == true.
        arena_3b[3_016..3_020].copy_from_slice(&0.02_f32.to_le_bytes()); // sigma_drift
        arena_3b[3_020..3_024].copy_from_slice(&0.95_f32.to_le_bytes()); // rho_drift_train
        cb.load_from_arena0(&arena_3b, 3, 1.0);
        cb.live_analog_drift_enabled = true;
        cb.offline_bounce = false;
        cb.refresh_static_state(&arena_3b);
        assert!(cb.ou_drift_state.iter().any(|&v| v.abs() > 0.0));
        cb.offline_bounce = true;
        cb.refresh_static_state(&arena_3b);
        assert_eq!(cb.ou_drift_state, [0.0; 4]);
    }

    #[test]
    fn test_micro_ramping_deltas_and_stepping() {
        let _audio = crate::FtzDazGuard::new();
        let curr = [1.0_f32; 16];
        let next = [2.0_f32; 16];
        let deltas_16 = ControlBus::<ScalarBackend>::compute_micro_ramp_deltas_16(&curr, &next);
        assert!((deltas_16[0] - 0.0625).abs() < 1e-6);

        let mut audio = curr;
        for _ in 0..16 {
            ControlBus::<ScalarBackend>::step_micro_ramp(&mut audio, &deltas_16);
        }
        for i in 0..16 {
            assert!((audio[i] - 2.0).abs() < 1e-5);
        }

        // Test with custom alpha_ramp for non-native sample rates (e.g. 96 kHz, K_ctrl = 32)
        let alpha_32 = 1.0 / 32.0;
        let deltas_32 =
            ControlBus::<ScalarBackend>::compute_micro_ramp_deltas(&curr, &next, alpha_32);
        assert!((deltas_32[0] - 0.03125).abs() < 1e-6);
        let mut audio32 = curr;
        for _ in 0..32 {
            ControlBus::<ScalarBackend>::step_micro_ramp(&mut audio32, &deltas_32);
        }
        for i in 0..16 {
            assert!((audio32[i] - 2.0).abs() < 1e-5);
        }
    }

    proptest! {
        #[test]
        fn prop_control_bus_lfo_unit_circle_and_micro_ramping(
            c_step in -0.99_f32..0.99_f32,
            t_curr in -10.0_f32..10.0_f32,
            t_next in -10.0_f32..10.0_f32,
            steps in 1_usize..64_usize,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let s_step = (1.0 - c_step * c_step).max(0.0).sqrt();
            let mut arena0 = vec![0u8; ARENA_OFFSET_LITE_END as usize];
            arena0[3_008..3_012].copy_from_slice(&(c_step.abs() * 0.5).to_le_bytes());
            arena0[3_012..3_016].copy_from_slice(&(s_step * 0.5).to_le_bytes());

            let mut cb = Box::new(ControlBus::<ScalarBackend>::default());
            cb.load_from_arena0(&arena0, 0, 1.0);
            for _ in 0..32 {
                cb.tick_3khz(0.25, &arena0);
                let r0_sq = cb.lfo_phasors[0][0] * cb.lfo_phasors[0][0]
                    + cb.lfo_phasors[0][1] * cb.lfo_phasors[0][1];
                let r1_sq = cb.lfo_phasors[1][0] * cb.lfo_phasors[1][0]
                    + cb.lfo_phasors[1][1] * cb.lfo_phasors[1][1];
                prop_assert!((r0_sq - 1.0).abs() < 1e-4, "LFO 0 drifted off unit circle: {}", r0_sq);
                prop_assert!((r1_sq - 1.0).abs() < 1e-4, "LFO 1 drifted off unit circle: {}", r1_sq);
            }

            // Micro-ramping exact reconstruction over K steps
            let curr = [t_curr; 16];
            let next = [t_next; 16];
            let alpha = 1.0 / (steps as f32);
            let deltas = ControlBus::<ScalarBackend>::compute_micro_ramp_deltas(&curr, &next, alpha);
            let mut audio = curr;
            for _ in 0..steps {
                ControlBus::<ScalarBackend>::step_micro_ramp(&mut audio, &deltas);
            }
            prop_assert!((audio[0] - t_next).abs() < 1e-3);
        }
    }
}
