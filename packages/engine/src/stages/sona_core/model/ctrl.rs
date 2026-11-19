//! Cold Top-of-Frame Crossbar & `3 kHz` Control Bus Configuration (`stages/sona_core/model/ctrl.rs`).
//!
//! Canonical Reference: `docs/specs/dsp/parametric_control_bus.md`,
//! `docs/specs/containers/dnam_binary_format.md`.

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use super::params::quantize_fp16_roundtrip_f32;
use super::tier::{MAX_SONA_GROUPS, MAX_TANK_LINES, NUM_SONA_BLOCKS, NUM_SONA_MACRO_KNOBS};

/// Number of `fp16` core-modulation columns per latent row in `Arena 3` (`8 x 128`).
pub const CORE_MODS_STRIDE: usize = 128;

/// Cold configuration for Top-of-Frame `Op-P` + `4x Q0.32` phasors + causal `Op-T` crossbar coupling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaCrossbarConfig {
    pub phasor_freq_hz: [f32; 4],
    pub phasor_init_phase_q32: [u32; 4],
    pub phasor_stereo_phase_rad: [f32; 4],
    pub op_p_crossbar_gain: f32,
    pub op_t_crossbar_gain: f32,
    pub op_p_to_group: [[f32; 4]; MAX_SONA_GROUPS],
    pub phasor_to_group: [[f32; 4]; MAX_SONA_GROUPS],
    pub opt_to_lr: [[f32; 2]; MAX_TANK_LINES],
}

impl Default for SonaCrossbarConfig {
    fn default() -> Self {
        Self::identity()
    }
}

impl SonaCrossbarConfig {
    /// Resting silent/zeroed crossbar configuration.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            phasor_freq_hz: [0.0; 4],
            phasor_init_phase_q32: [0; 4],
            phasor_stereo_phase_rad: [0.0; 4],
            op_p_crossbar_gain: 0.0,
            op_t_crossbar_gain: 0.0,
            op_p_to_group: [[0.0; 4]; MAX_SONA_GROUPS],
            phasor_to_group: [[0.0; 4]; MAX_SONA_GROUPS],
            opt_to_lr: [[0.0; 2]; MAX_TANK_LINES],
        }
    }

    /// Clamps frequencies/phases and quantizes non-recursive crossbar gains & matrices in `fp16`.
    pub fn clamp_in_place(&mut self) {
        for f in &mut self.phasor_freq_hz {
            *f = if f.is_finite() {
                f.clamp(0.0, 20_000.0)
            } else {
                0.0
            };
        }
        for p in &mut self.phasor_stereo_phase_rad {
            *p = if p.is_finite() {
                p.clamp(-core::f32::consts::PI, core::f32::consts::PI)
            } else {
                0.0
            };
        }
        self.op_p_crossbar_gain =
            quantize_fp16_roundtrip_f32(self.op_p_crossbar_gain.clamp(-4.0, 4.0));
        self.op_t_crossbar_gain =
            quantize_fp16_roundtrip_f32(self.op_t_crossbar_gain.clamp(-4.0, 4.0));
        for g in 0..MAX_SONA_GROUPS {
            for k in 0..4 {
                self.op_p_to_group[g][k] =
                    quantize_fp16_roundtrip_f32(if self.op_p_to_group[g][k].is_finite() {
                        self.op_p_to_group[g][k].clamp(-8.0, 8.0)
                    } else {
                        0.0
                    });
                self.phasor_to_group[g][k] =
                    quantize_fp16_roundtrip_f32(if self.phasor_to_group[g][k].is_finite() {
                        self.phasor_to_group[g][k].clamp(-8.0, 8.0)
                    } else {
                        0.0
                    });
            }
        }
        for line in 0..MAX_TANK_LINES {
            for ch in 0..2 {
                self.opt_to_lr[line][ch] =
                    quantize_fp16_roundtrip_f32(if self.opt_to_lr[line][ch].is_finite() {
                        self.opt_to_lr[line][ch].clamp(-4.0, 4.0)
                    } else {
                        0.0
                    });
            }
        }
    }
}

/// Complete `Arena 0` (`1,920 B`) + `Arena 3` (`3,840 B`) `3 kHz` `ControlBus` wire payload &
/// `Ultra` latent core-modulation tables (`V3_META_CORE_MODS_PRESENT`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaControlBusParams {
    pub default_knobs: [f32; NUM_SONA_MACRO_KNOBS],
    pub w_svf_g_oct: [f32; NUM_SONA_MACRO_KNOBS],
    pub w_svf_k_delta: [f32; NUM_SONA_MACRO_KNOBS],
    pub w_kautz_oct: [f32; NUM_SONA_MACRO_KNOBS],
    pub w_kautz_damp: [f32; NUM_SONA_MACRO_KNOBS],
    pub w_tank_psi: [f32; NUM_SONA_MACRO_KNOBS],
    pub w_adaa_drive_scale: [[f32; MAX_SONA_GROUPS]; NUM_SONA_MACRO_KNOBS],
    pub w_lru_bias_shift: [[f32; MAX_SONA_GROUPS]; NUM_SONA_MACRO_KNOBS],
    pub w_adaa_drive_scale_lite: [[f32; NUM_SONA_BLOCKS]; 4],
    pub w_lru_bias_shift_lite: [[f32; NUM_SONA_BLOCKS]; 4],
    pub lite_block_scales_present: bool,
    pub lfo_dphi_train: [f32; 2],
    pub sigma_drift: f32,
    pub rho_drift_train: f32,
    pub env_poles: [f32; 3],
    pub dither_snr_scale: f32,
    pub trust_radius_mh: f32,
    pub cp1_dir_count: u16,
    pub cp1_dir_flags: u16,
    pub cp1_dir_entries: [[u8; 3]; 8],
    pub u_int_base: [[f32; 8]; 4],
    pub cp1_base: [u8; 576],
    pub w_ctrl: [[f32; 16]; 16],
    pub rho_head_and_ctrl_tail: [f32; 32],
    pub w_ctrl_ultra: [[f32; 16]; 4],
    pub u_int_ultra: [[[f32; 8]; 4]; 3],
    pub cp1_ultra: [u8; 2_112],
    pub stage_latent_core_mods_fp16: [u16; 1_024],
    pub core_mods_present: bool,
}

/// Compatibility alias for [`SonaControlBusParams`].
pub type V3ControlBusParams = SonaControlBusParams;

impl Default for SonaControlBusParams {
    fn default() -> Self {
        Self::identity()
    }
}

impl SonaControlBusParams {
    /// Resting identity `ControlBus` state (`default_knobs` `0.5` metadata-only, zero modulations).
    #[must_use]
    pub const fn identity() -> Self {
        let mut rho_tail = [0.0_f32; 32];
        rho_tail[0] = 1.0;
        Self {
            default_knobs: [0.5; NUM_SONA_MACRO_KNOBS],
            w_svf_g_oct: [0.0; NUM_SONA_MACRO_KNOBS],
            w_svf_k_delta: [0.0; NUM_SONA_MACRO_KNOBS],
            w_kautz_oct: [0.0; NUM_SONA_MACRO_KNOBS],
            w_kautz_damp: [0.0; NUM_SONA_MACRO_KNOBS],
            w_tank_psi: [0.0; NUM_SONA_MACRO_KNOBS],
            w_adaa_drive_scale: [[0.0; MAX_SONA_GROUPS]; NUM_SONA_MACRO_KNOBS],
            w_lru_bias_shift: [[0.0; MAX_SONA_GROUPS]; NUM_SONA_MACRO_KNOBS],
            w_adaa_drive_scale_lite: [[0.0; NUM_SONA_BLOCKS]; 4],
            w_lru_bias_shift_lite: [[0.0; NUM_SONA_BLOCKS]; 4],
            lite_block_scales_present: false,
            lfo_dphi_train: [0.0; 2],
            sigma_drift: 0.0,
            rho_drift_train: 0.99,
            env_poles: [0.95, 0.99, 0.998],
            dither_snr_scale: 0.0,
            trust_radius_mh: 1.0,
            cp1_dir_count: 0,
            cp1_dir_flags: 0,
            cp1_dir_entries: [[0; 3]; 8],
            u_int_base: [[0.0; 8]; 4],
            cp1_base: [0; 576],
            w_ctrl: [[0.0; 16]; 16],
            rho_head_and_ctrl_tail: rho_tail,
            w_ctrl_ultra: [[0.0; 16]; 4],
            u_int_ultra: [[[0.0; 8]; 4]; 3],
            cp1_ultra: [0; 2_112],
            stage_latent_core_mods_fp16: [0; 1_024],
            core_mods_present: false,
        }
    }

    /// Reads one `fp16` cell `(latent in 0..8, col in 0..128)` from `stage_latent_core_mods_fp16`.
    #[inline(always)]
    #[must_use]
    pub fn core_mod(&self, latent: usize, col: usize) -> f32 {
        if latent >= NUM_SONA_MACRO_KNOBS || col >= CORE_MODS_STRIDE {
            return 0.0;
        }
        fp16_to_f32(self.stage_latent_core_mods_fp16[latent * CORE_MODS_STRIDE + col])
    }

    /// Sets one `fp16` cell `(latent in 0..8, col in 0..128)` in `stage_latent_core_mods_fp16`.
    #[inline(always)]
    pub fn set_core_mod(&mut self, latent: usize, col: usize, value: f32) {
        if latent >= NUM_SONA_MACRO_KNOBS || col >= CORE_MODS_STRIDE {
            return;
        }
        let clamped = if value.is_finite() {
            value.clamp(-4.0, 4.0)
        } else {
            0.0
        };
        self.stage_latent_core_mods_fp16[latent * CORE_MODS_STRIDE + col] = f32_to_fp16(clamped);
    }

    /// Sanitizes all control bus weights, OU drift poles, trust radius, and latent modulation tables.
    pub fn clamp_in_place(&mut self) {
        for k in 0..NUM_SONA_MACRO_KNOBS {
            self.default_knobs[k] = self.default_knobs[k].clamp(0.0, 1.0);
            self.w_svf_g_oct[k] = self.w_svf_g_oct[k].clamp(-3.0, 3.0);
            self.w_svf_k_delta[k] = self.w_svf_k_delta[k].clamp(-1.5, 1.5);
            self.w_kautz_oct[k] = self.w_kautz_oct[k].clamp(-1.5, 1.5);
            self.w_kautz_damp[k] = self.w_kautz_damp[k].clamp(-1.5, 1.5);
            self.w_tank_psi[k] =
                self.w_tank_psi[k].clamp(-core::f32::consts::PI, core::f32::consts::PI);
            for g in 0..MAX_SONA_GROUPS {
                self.w_adaa_drive_scale[k][g] = self.w_adaa_drive_scale[k][g].clamp(-1.5, 1.5);
                self.w_lru_bias_shift[k][g] = self.w_lru_bias_shift[k][g].clamp(-2.0, 2.0);
            }
        }
        for k in 0..4 {
            for b in 0..NUM_SONA_BLOCKS {
                self.w_adaa_drive_scale_lite[k][b] =
                    if self.w_adaa_drive_scale_lite[k][b].is_finite() {
                        self.w_adaa_drive_scale_lite[k][b].clamp(-1.5, 1.5)
                    } else {
                        0.0
                    };
                self.w_lru_bias_shift_lite[k][b] = if self.w_lru_bias_shift_lite[k][b].is_finite() {
                    self.w_lru_bias_shift_lite[k][b].clamp(-2.0, 2.0)
                } else {
                    0.0
                };
            }
        }
        for p in &mut self.lfo_dphi_train {
            *p = if p.is_finite() {
                p.clamp(-core::f32::consts::PI, core::f32::consts::PI)
            } else {
                0.0
            };
        }
        self.sigma_drift = if self.sigma_drift.is_finite() {
            self.sigma_drift.clamp(0.0, 2.0)
        } else {
            0.0
        };
        self.rho_drift_train = if self.rho_drift_train.is_finite() {
            self.rho_drift_train.clamp(0.0, 0.999_9)
        } else {
            0.99
        };
        for p in &mut self.env_poles {
            *p = if p.is_finite() {
                p.clamp(0.0, 0.999_95)
            } else {
                0.95
            };
        }
        self.dither_snr_scale = if self.dither_snr_scale.is_finite() {
            self.dither_snr_scale.clamp(0.0, 4.0)
        } else {
            0.0
        };
        self.trust_radius_mh = if self.trust_radius_mh.is_finite() {
            self.trust_radius_mh.clamp(0.01, 16.0)
        } else {
            1.0
        };
        self.rho_head_and_ctrl_tail[0] = self.trust_radius_mh;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossbar_fp16_quantization_and_control_bus_core_mods() {
        let mut cb = SonaCrossbarConfig::identity();
        cb.op_p_crossbar_gain = 0.123_456_7;
        cb.opt_to_lr[0] = [0.333_333_3, -0.777_777_8];
        cb.op_p_to_group[1] = [0.111_111, -0.222_222, 100.0, f32::NAN];
        cb.clamp_in_place();
        assert_eq!(
            cb.op_p_crossbar_gain.to_bits(),
            fp16_to_f32(f32_to_fp16(0.123_456_7)).to_bits()
        );
        assert_eq!(
            cb.opt_to_lr[0][0].to_bits(),
            fp16_to_f32(f32_to_fp16(0.333_333_3)).to_bits()
        );
        assert_eq!(cb.op_p_to_group[1][2], 8.0);
        assert_eq!(cb.op_p_to_group[1][3], 0.0);

        let mut bus = SonaControlBusParams::identity();
        assert!(!bus.lite_block_scales_present);
        bus.w_adaa_drive_scale_lite[2][3] = 5.0;
        bus.w_adaa_drive_scale_lite[0][1] = f32::NAN;
        bus.w_lru_bias_shift_lite[1][2] = -4.0;
        bus.w_lru_bias_shift_lite[3][0] = f32::INFINITY;
        bus.clamp_in_place();
        assert_eq!(bus.w_adaa_drive_scale_lite[2][3], 1.5);
        assert_eq!(bus.w_adaa_drive_scale_lite[0][1], 0.0);
        assert_eq!(bus.w_lru_bias_shift_lite[1][2], -2.0);
        assert_eq!(bus.w_lru_bias_shift_lite[3][0], 0.0);
        bus.set_core_mod(3, 41, 0.5);
        assert!((bus.core_mod(3, 41) - 0.5).abs() < 1e-6);
        assert_eq!(bus.core_mod(9, 0), 0.0);
    }
}
