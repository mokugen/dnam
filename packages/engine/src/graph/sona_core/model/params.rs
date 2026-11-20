//! Cold `SONA` Group/Tile Parameters, ZOH Pole Rescaling & `VarPro` Tier Readouts
//! (`graph/sona_core/model/params.rs`).

use dnam_kernels::{MAX_PARALRU_R_F32, f32_to_fp16, fp16_to_f32, prewarp_tan_pi_f_over_fs_f32};

use super::tier::{
    MAX_KAUTZ_POLES, MAX_TOTAL_SONA_LANES, SONA_FIR_MAX_TAPS, tier_sona_groups, tier_spec,
};
use crate::container::ZohRescaledParams;
use crate::math::const_eval::const_sin_cos_f64;
use crate::schema::cold_pow_f64;
use crate::types::Tier;

#[inline]
pub(crate) fn quantize_fp16_roundtrip_f32(x: f32) -> f32 {
    if x.is_finite() {
        fp16_to_f32(f32_to_fp16(x))
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn cold_atan_pos_f64(g: f64) -> f64 {
    if g <= 0.0 || !g.is_finite() {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0_f64, core::f64::consts::FRAC_PI_2 - 1.0e-7);
    for _ in 0..52 {
        let mid = 0.5 * (lo + hi);
        let (s, c) = const_sin_cos_f64(mid);
        if s < g * c {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[inline]
pub(crate) fn rescale_tpt_svf_g_zoh(g_ref: f32, rho_sr: f64) -> f32 {
    let g_pos = if g_ref.is_finite() {
        g_ref.clamp(1.0e-5, 64.0)
    } else {
        0.1
    };
    if (rho_sr - 1.0).abs() <= 1.0e-9 {
        return g_pos;
    }
    let ratio_ref = cold_atan_pos_f64(f64::from(g_pos)) * core::f64::consts::FRAC_1_PI;
    let ratio_eff = (ratio_ref * rho_sr).clamp(1.0e-5, 0.495) as f32;
    prewarp_tan_pi_f_over_fs_f32(ratio_eff)
}

/// Cold representation of one `4`-lane stereo `SONA` core group (`TPT-SVF -> ParaLRU -> PolyADAA1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaGroupCold {
    pub svf_g: [f32; 4],
    pub svf_k: [f32; 4],
    pub lru_alpha_rel: [f32; 4],
    pub lru_alpha_atk: [f32; 4],
    pub lru_kappa: [f32; 4],
    pub adaa_drive: [f32; 4],
    pub lru_bias_drive: [f32; 4],
    pub in_x_weight: [f32; 4],
    pub tone_lp: [f32; 4],
    pub tone_bp: [f32; 4],
    pub tone_hp: [f32; 4],
    pub tone_ap: [f32; 4],
    pub tone_dir: [f32; 4],
    pub sag_depth: [f32; 4],
    pub wavefolder_mix: [f32; 4],
    pub rect_sign: [f32; 4],
}

impl Default for SonaGroupCold {
    fn default() -> Self {
        Self::identity_group(0)
    }
}

impl SonaGroupCold {
    /// Builds a resting identity `SONA` group passing clean input linearly with zero DC offset.
    #[must_use]
    pub const fn identity_group(group_idx: usize) -> Self {
        let in_w = if group_idx == 0 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        let adaa_d = if group_idx == 0 {
            [0.5, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        Self {
            svf_g: [0.25; 4],
            svf_k: [core::f32::consts::FRAC_1_SQRT_2; 4],
            lru_alpha_rel: [0.9; 4],
            lru_alpha_atk: [0.0; 4],
            lru_kappa: [0.0; 4],
            adaa_drive: adaa_d,
            lru_bias_drive: [0.0; 4],
            in_x_weight: in_w,
            tone_lp: [0.0; 4],
            tone_bp: [0.0; 4],
            tone_hp: [0.0; 4],
            tone_ap: [0.0; 4],
            tone_dir: [1.0; 4],
            sag_depth: [0.0; 4],
            wavefolder_mix: [0.0; 4],
            rect_sign: [1.0; 4],
        }
    }

    /// Clamps all group parameters into unconditionally passive (`||A_tpt||_2 <= 1`) and
    /// contractive (`0 <= alpha_rel <= MAX_PARALRU_R_F32`) bounds.
    pub fn clamp_in_place(&mut self) {
        for l in 0..4 {
            self.svf_g[l] = if self.svf_g[l].is_finite() {
                self.svf_g[l].clamp(1.0e-5, 64.0)
            } else {
                0.25
            };
            self.svf_k[l] = if self.svf_k[l].is_finite() {
                self.svf_k[l].clamp(0.0, 8.0)
            } else {
                core::f32::consts::FRAC_1_SQRT_2
            };
            self.lru_alpha_rel[l] = if self.lru_alpha_rel[l].is_finite() {
                self.lru_alpha_rel[l].clamp(0.0, MAX_PARALRU_R_F32)
            } else {
                0.0
            };
            self.lru_alpha_atk[l] = if self.lru_alpha_atk[l].is_finite() {
                self.lru_alpha_atk[l].clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.lru_kappa[l] = if self.lru_kappa[l].is_finite() {
                self.lru_kappa[l].clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.adaa_drive[l] = if self.adaa_drive[l].is_finite() {
                self.adaa_drive[l].clamp(-32.0, 32.0)
            } else {
                0.0
            };
            self.lru_bias_drive[l] = if self.lru_bias_drive[l].is_finite() {
                self.lru_bias_drive[l].clamp(-16.0, 16.0)
            } else {
                0.0
            };
            self.in_x_weight[l] = if self.in_x_weight[l].is_finite() {
                self.in_x_weight[l].clamp(-16.0, 16.0)
            } else {
                0.0
            };
            for arr in [
                &mut self.tone_lp,
                &mut self.tone_bp,
                &mut self.tone_hp,
                &mut self.tone_ap,
                &mut self.tone_dir,
            ] {
                arr[l] = quantize_fp16_roundtrip_f32(if arr[l].is_finite() {
                    arr[l].clamp(-16.0, 16.0)
                } else {
                    0.0
                });
            }
            self.sag_depth[l] = quantize_fp16_roundtrip_f32(if self.sag_depth[l].is_finite() {
                self.sag_depth[l].clamp(-8.0, 8.0)
            } else {
                0.0
            });
            self.wavefolder_mix[l] =
                quantize_fp16_roundtrip_f32(if self.wavefolder_mix[l].is_finite() {
                    self.wavefolder_mix[l].clamp(0.0, 1.0)
                } else {
                    0.0
                });
            self.rect_sign[l] = if self.rect_sign[l] < 0.0 { -1.0 } else { 1.0 };
        }
    }

    /// Rescales `96 kHz` (`2x` OS) coefficients to `48 kHz` (`1x`) via continuous-time ZOH.
    pub fn rescale_2x_to_1x(&mut self) {
        self.rescale_with_zoh_ratio(2.0);
    }

    /// Rescales `TPT-SVF` prewarped `g` (`tan(pi * fc / fs_eff)`) and `ParaLRU` poles (`alpha^rho_sr`)
    /// under continuous-time ZOH (`ref[dsp.op-n.core.matryoshka-blocks]`).
    pub fn rescale_with_zoh(&mut self, zoh: &ZohRescaledParams) {
        self.rescale_with_zoh_ratio(zoh.rho_sr);
    }

    pub(crate) fn rescale_with_zoh_ratio(&mut self, rho_sr: f64) {
        self.clamp_in_place();
        if (rho_sr - 1.0).abs() <= 1.0e-7 {
            return;
        }
        for l in 0..4 {
            self.svf_g[l] = rescale_tpt_svf_g_zoh(self.svf_g[l], rho_sr);
            let ar = f64::from(self.lru_alpha_rel[l]);
            if ar > 0.0 {
                self.lru_alpha_rel[l] =
                    (cold_pow_f64(ar, rho_sr) as f32).clamp(0.0, MAX_PARALRU_R_F32);
            }
            let aa = f64::from(self.lru_alpha_atk[l]);
            if aa > 0.0 && aa < 1.0 {
                self.lru_alpha_atk[l] = (cold_pow_f64(aa, rho_sr) as f32).clamp(0.0, 1.0);
            }
        }
    }
}

/// Cold `4x4` inter-block spatial mixing tile triple (`odd`, `even`, `lru`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaTileCold {
    pub w_odd_4x4: [[f32; 4]; 4],
    pub w_even_4x4: [[f32; 4]; 4],
    pub w_lru_4x4: [[f32; 4]; 4],
}

impl Default for SonaTileCold {
    fn default() -> Self {
        Self::zeros()
    }
}

impl SonaTileCold {
    /// Returns an all-zero inter-block mixing tile triple.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            w_odd_4x4: [[0.0; 4]; 4],
            w_even_4x4: [[0.0; 4]; 4],
            w_lru_4x4: [[0.0; 4]; 4],
        }
    }

    /// Clamps and quantizes non-recursive `w_even`/`w_lru` entries to `fp16`.
    pub fn clamp_in_place(&mut self) {
        for r in 0..4 {
            for c in 0..4 {
                self.w_odd_4x4[r][c] = if self.w_odd_4x4[r][c].is_finite() {
                    self.w_odd_4x4[r][c].clamp(-8.0, 8.0)
                } else {
                    0.0
                };
                self.w_even_4x4[r][c] =
                    quantize_fp16_roundtrip_f32(if self.w_even_4x4[r][c].is_finite() {
                        self.w_even_4x4[r][c].clamp(-8.0, 8.0)
                    } else {
                        0.0
                    });
                self.w_lru_4x4[r][c] =
                    quantize_fp16_roundtrip_f32(if self.w_lru_4x4[r][c].is_finite() {
                        self.w_lru_4x4[r][c].clamp(-8.0, 8.0)
                    } else {
                        0.0
                    });
            }
        }
    }
}

/// Per-tier linear `VarPro` readout head over `SONA` core lanes, Stage III FIR, and Kautz modals.
#[derive(Clone, Debug, PartialEq)]
pub struct SonaTierReadout {
    pub pre_cab_u_gain: [f32; 2],
    pub pre_cab_hp_gain: [f32; 2],
    pub pre_cab_w_odd: [[f32; MAX_TOTAL_SONA_LANES]; 2],
    pub pre_cab_w_even: [[f32; MAX_TOTAL_SONA_LANES]; 2],
    pub pre_cab_w_lru: [[f32; MAX_TOTAL_SONA_LANES]; 2],
    pub direct_s2_u_gain: [f32; 2],
    pub direct_s2_hp_gain: [f32; 2],
    pub direct_w_odd: [[f32; MAX_TOTAL_SONA_LANES]; 2],
    pub direct_w_even: [[f32; MAX_TOTAL_SONA_LANES]; 2],
    pub direct_w_lru: [[f32; MAX_TOTAL_SONA_LANES]; 2],
    pub fir_l: [f32; SONA_FIR_MAX_TAPS],
    pub fir_r: [f32; SONA_FIR_MAX_TAPS],
    pub kautz_b_gamma: [f32; MAX_KAUTZ_POLES],
    pub kautz_c_l: [(f32, f32); MAX_KAUTZ_POLES],
    pub kautz_c_r: [(f32, f32); MAX_KAUTZ_POLES],
}

impl Default for SonaTierReadout {
    fn default() -> Self {
        let mut r = Self::zeros();
        r.pre_cab_u_gain = [1.0, 1.0];
        r.fir_l[0] = 1.0;
        r.fir_r[0] = 1.0;
        r
    }
}

impl SonaTierReadout {
    /// Returns an all-zero `VarPro` linear readout head.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            pre_cab_u_gain: [0.0; 2],
            pre_cab_hp_gain: [0.0; 2],
            pre_cab_w_odd: [[0.0; MAX_TOTAL_SONA_LANES]; 2],
            pre_cab_w_even: [[0.0; MAX_TOTAL_SONA_LANES]; 2],
            pre_cab_w_lru: [[0.0; MAX_TOTAL_SONA_LANES]; 2],
            direct_s2_u_gain: [0.0; 2],
            direct_s2_hp_gain: [0.0; 2],
            direct_w_odd: [[0.0; MAX_TOTAL_SONA_LANES]; 2],
            direct_w_even: [[0.0; MAX_TOTAL_SONA_LANES]; 2],
            direct_w_lru: [[0.0; MAX_TOTAL_SONA_LANES]; 2],
            fir_l: [0.0; SONA_FIR_MAX_TAPS],
            fir_r: [0.0; SONA_FIR_MAX_TAPS],
            kautz_b_gamma: [0.0; MAX_KAUTZ_POLES],
            kautz_c_l: [(0.0, 0.0); MAX_KAUTZ_POLES],
            kautz_c_r: [(0.0, 0.0); MAX_KAUTZ_POLES],
        }
    }

    /// Zeros any coefficients beyond `tier`'s locked compile-time [`super::SonaTierSpec`].
    pub fn mask_to_tier(&mut self, tier: Tier) {
        let g_cnt = tier_sona_groups(tier);
        let lane_cap = (g_cnt << 2).min(MAX_TOTAL_SONA_LANES);
        let f_cnt = tier_spec(tier).fir_taps;
        let k_cnt = tier_spec(tier).kautz_pairs;
        for ch in 0..2 {
            self.pre_cab_w_odd[ch][lane_cap..].fill(0.0);
            self.pre_cab_w_even[ch][lane_cap..].fill(0.0);
            self.pre_cab_w_lru[ch][lane_cap..].fill(0.0);
            self.direct_w_odd[ch][lane_cap..].fill(0.0);
            self.direct_w_even[ch][lane_cap..].fill(0.0);
            self.direct_w_lru[ch][lane_cap..].fill(0.0);
        }
        self.fir_l[f_cnt..].fill(0.0);
        self.fir_r[f_cnt..].fill(0.0);
        for m in k_cnt..MAX_KAUTZ_POLES {
            self.kautz_b_gamma[m] = 0.0;
            self.kautz_c_l[m] = (0.0, 0.0);
            self.kautz_c_r[m] = (0.0, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sona_group_zoh_scaling_and_readout_mask() {
        let mut g = SonaGroupCold::identity_group(0);
        g.svf_g = [prewarp_tan_pi_f_over_fs_f32(1_000.0 / 48_000.0); 4];
        g.lru_alpha_rel = [0.9; 4];
        let mut g_96 = g;
        g_96.rescale_with_zoh(&ZohRescaledParams::new(48_000, 96_000));
        assert!((g_96.svf_g[0] - prewarp_tan_pi_f_over_fs_f32(1_000.0 / 96_000.0)).abs() < 1.0e-5);
        assert!((g_96.lru_alpha_rel[0] * g_96.lru_alpha_rel[0] - 0.9).abs() < 1.0e-5);

        let mut ro = SonaTierReadout::default();
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Lite);
        assert_eq!(ro.fir_l[31], 0.5);
        assert_eq!(ro.fir_l[32], 0.0);
    }

    #[test]
    fn test_sona_model_params_nonfinite_clamping_defaults_and_zoh_branches() {
        assert_eq!(quantize_fp16_roundtrip_f32(f32::NAN), 0.0);
        assert_eq!(quantize_fp16_roundtrip_f32(f32::INFINITY), 0.0);
        assert_eq!(cold_atan_pos_f64(0.0), 0.0);
        assert_eq!(cold_atan_pos_f64(-1.0), 0.0);
        assert_eq!(cold_atan_pos_f64(f64::NAN), 0.0);
        assert_eq!(rescale_tpt_svf_g_zoh(0.25, 1.0), 0.25);
        assert!(
            (rescale_tpt_svf_g_zoh(f32::NAN, 0.5) - rescale_tpt_svf_g_zoh(0.1, 0.5)).abs() < 1e-7
        );

        assert_eq!(SonaGroupCold::default(), SonaGroupCold::identity_group(0));
        assert_eq!(SonaTileCold::default(), SonaTileCold::zeros());

        let mut bad_g = SonaGroupCold {
            svf_g: [f32::NAN; 4],
            svf_k: [f32::INFINITY; 4],
            lru_alpha_rel: [f32::NAN; 4],
            lru_alpha_atk: [f32::NAN; 4],
            lru_kappa: [f32::NAN; 4],
            adaa_drive: [f32::NAN; 4],
            lru_bias_drive: [f32::NAN; 4],
            in_x_weight: [f32::NAN; 4],
            tone_lp: [f32::NAN; 4],
            tone_bp: [f32::NAN; 4],
            tone_hp: [f32::NAN; 4],
            tone_ap: [f32::NAN; 4],
            tone_dir: [f32::NAN; 4],
            sag_depth: [f32::NAN; 4],
            wavefolder_mix: [f32::NAN; 4],
            rect_sign: [-2.0; 4],
        };
        bad_g.clamp_in_place();
        assert_eq!(bad_g.svf_g, [0.25; 4]);
        assert_eq!(
            bad_g.svf_k.map(f32::to_bits),
            [core::f32::consts::FRAC_1_SQRT_2.to_bits(); 4]
        );
        assert_eq!(bad_g.lru_alpha_rel, [0.0; 4]);
        assert_eq!(bad_g.lru_alpha_atk, [0.0; 4]);
        assert_eq!(bad_g.lru_kappa, [0.0; 4]);
        assert_eq!(bad_g.adaa_drive, [0.0; 4]);
        assert_eq!(bad_g.lru_bias_drive, [0.0; 4]);
        assert_eq!(bad_g.in_x_weight, [0.0; 4]);
        assert_eq!(bad_g.tone_dir, [0.0; 4]);
        assert_eq!(bad_g.sag_depth, [0.0; 4]);
        assert_eq!(bad_g.wavefolder_mix, [0.0; 4]);
        assert_eq!(bad_g.rect_sign, [-1.0; 4]);

        let mut g_zoh = SonaGroupCold::identity_group(0);
        g_zoh.lru_alpha_rel = [0.0, 0.9, 0.5, 0.0];
        g_zoh.lru_alpha_atk = [0.0, 0.5, 1.0, 0.25];
        let mut g_noop = g_zoh;
        g_noop.rescale_with_zoh_ratio(1.0);
        assert_eq!(g_noop, g_zoh);

        let mut g_2x = g_zoh;
        g_2x.rescale_2x_to_1x();
        assert!(g_2x.svf_g[0] > g_zoh.svf_g[0]);
        assert_eq!(g_2x.lru_alpha_rel[0], 0.0);
        assert!(g_2x.lru_alpha_atk[1] < 0.5);
        assert_eq!(g_2x.lru_alpha_atk[2], 1.0);

        let mut tile = SonaTileCold {
            w_odd_4x4: [[f32::NAN; 4], [100.0; 4], [-100.0; 4], [0.5; 4]],
            w_even_4x4: [[f32::INFINITY; 4], [100.0; 4], [-100.0; 4], [0.25; 4]],
            w_lru_4x4: [[f32::NEG_INFINITY; 4], [100.0; 4], [-100.0; 4], [-0.25; 4]],
        };
        tile.clamp_in_place();
        assert_eq!(tile.w_odd_4x4[0], [0.0; 4]);
        assert_eq!(tile.w_odd_4x4[1], [8.0; 4]);
        assert_eq!(tile.w_even_4x4[0], [0.0; 4]);
        assert_eq!(tile.w_even_4x4[1], [8.0; 4]);
        assert_eq!(tile.w_lru_4x4[0], [0.0; 4]);
        assert_eq!(tile.w_lru_4x4[2], [-8.0; 4]);
    }
}
