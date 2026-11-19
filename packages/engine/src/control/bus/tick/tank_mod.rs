//! $3\text{ kHz}$ Control-Bus tank & stage modulation evaluation and optional OU analog drift.

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};

use super::super::ControlBus;
use super::simd_and_clip::read_vec4_le;

impl<B: MathBackend> ControlBus<B> {
    /// Evaluates the $3\text{ kHz}$ `8 × 32` / `16 × 16` Control/LFO/Envelope $\to$ Stage & Spatial Tank
    /// state-space projection matrix `W_ctrl` (`ARENA0_W_CTRL_OFFSET = 3,456..4,480` in `Arena 0`,
    /// plus `ARENA3_W_CTRL_ULTRA_OFFSET = 26,880..27,136` (`16 × 4`) in `Arena 3`, `0 FDIV`, `0 FSQRT`).
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
            let has_ultra_ctrl = self.ultra_tier_active
                && arena_bytes.len() >= W_CTRL_ULTRA_BASE + 256
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
            self.apply_ou_drift();
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
        let has_ultra_ctrl = self.ultra_tier_active
            && arena_bytes.len() >= W_CTRL_ULTRA_BASE + 256
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
                2..15
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

        self.apply_ou_drift();
        self.tank_mod_dirty = true;
    }

    /// Applies optional Ornstein-Uhlenbeck analog drift to `stage_targets` slots `[2, 5, 10, 14]`
    /// strictly when `live_analog_drift_enabled && !offline_bounce && sigma_drift > 0.0`
    /// (`r[dsp.ctrl.modulators.dynamic]`, `r[dsp.ctrl.dispatch.sparse-and-active-ramps]`).
    #[inline]
    pub(super) fn apply_ou_drift(&mut self) {
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
            self.stage_targets[10] += 0.25 * self.ou_drift_state[2];
            self.stage_targets[14] += 0.25 * self.ou_drift_state[3];
        } else {
            self.ou_drift_state = [0.0; 4];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;
    use crate::stages::weights::ARENA_LITE_BYTES;
    use alloc::vec;

    #[test]
    fn test_unified_w_ctrl_ou_drift_slots_2_5_10_14_and_offline_suppression() {
        let _audio = crate::FtzDazGuard::new();
        let arena = vec![0u8; ARENA_LITE_BYTES];
        let mut bus = ControlBus::<ScalarBackend> {
            unified_w_ctrl: true,
            live_analog_drift_enabled: true,
            offline_bounce: false,
            sigma_drift: 0.04,
            rho_drift_host: 0.9,
            drift_prng_state: 0,
            ..Default::default()
        };
        bus.evaluate_tank_modulation(&arena);
        assert_ne!(bus.drift_prng_state, 0);
        assert_eq!(bus.stage_targets[2], bus.ou_drift_state[0]);
        assert_eq!(bus.stage_targets[5], bus.ou_drift_state[1]);
        assert_eq!(bus.stage_targets[10], 0.25 * bus.ou_drift_state[2]);
        assert_eq!(bus.stage_targets[14], 0.25 * bus.ou_drift_state[3]);
        for &slot in &[0usize, 1, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15] {
            assert_eq!(
                bus.stage_targets[slot], 0.0,
                "slot {slot} must be untouched by OU drift"
            );
        }

        bus.offline_bounce = true;
        bus.evaluate_tank_modulation(&arena);
        assert_eq!(bus.ou_drift_state, [0.0; 4]);
        assert_eq!(bus.stage_targets, [0.0; 16]);
    }
}
