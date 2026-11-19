//! Cold Readout, Rate-Specific Calibration, and Nested Matryoshka Model Definitions.

use crate::stages::router::V3Stage0Params;
use crate::stages::stage1_svf::V3Stage1SvfParams;
use crate::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS,
    V3ControlBusParams, V3GruBlockCold, V3InputFeaturesParams, V3InterBlockToneSvfParams,
    V3OversamplerParams, flat_gru_lane_index, tier_block_groups, tier_fir_segments,
    tier_gru_blocks, tier_kautz_banks,
};
use crate::stages::stage4_tank::v3_fdn::V3Stage4TankParams;
use crate::types::Tier;

/// Optional dedicated `1x` (`48 kHz`) front-end and Stage II calibration preserving `Lite ⊂ Full` prefix nesting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Rate1xCalibration {
    pub svf: V3Stage1SvfParams,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockCold; NUM_GRU_BLOCKS],
}

/// Per-tier compact linear readout head over shared SVF taps, Stage II GRU lanes, FIR, and Kautz.
#[derive(Clone, Debug, PartialEq)]
pub struct V3TierReadout {
    pub svf_direct_taps_l: [f32; 8],
    pub svf_direct_taps_r: [f32; 8],
    pub pre_cab_u_gain: [f32; 2],
    pub pre_cab_hp_gain: [f32; 2],
    pub pre_cab_w: [[f32; MAX_TOTAL_GRU_LANES]; 2],
    pub direct_s2_u_gain: [f32; 2],
    pub direct_s2_hp_gain: [f32; 2],
    pub direct_s2_w: [[f32; MAX_TOTAL_GRU_LANES]; 2],
    pub fir_l: [f32; 64],
    pub fir_r: [f32; 64],
    pub kautz_b_gamma: [f32; MAX_KAUTZ_POLES],
    pub kautz_c_l: [(f32, f32); MAX_KAUTZ_POLES],
    pub kautz_c_r: [(f32, f32); MAX_KAUTZ_POLES],
}

impl Default for V3TierReadout {
    fn default() -> Self {
        let mut r = Self::zeros();
        r.pre_cab_u_gain = [1.0, 1.0];
        r.fir_l[0] = 1.0;
        r.fir_r[0] = 1.0;
        r
    }
}

impl V3TierReadout {
    /// Returns an all-zero linear readout head.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            svf_direct_taps_l: [0.0; 8],
            svf_direct_taps_r: [0.0; 8],
            pre_cab_u_gain: [0.0; 2],
            pre_cab_hp_gain: [0.0; 2],
            pre_cab_w: [[0.0; MAX_TOTAL_GRU_LANES]; 2],
            direct_s2_u_gain: [0.0; 2],
            direct_s2_hp_gain: [0.0; 2],
            direct_s2_w: [[0.0; MAX_TOTAL_GRU_LANES]; 2],
            fir_l: [0.0; 64],
            fir_r: [0.0; 64],
            kautz_b_gamma: [0.0; MAX_KAUTZ_POLES],
            kautz_c_l: [(0.0, 0.0); MAX_KAUTZ_POLES],
            kautz_c_r: [(0.0, 0.0); MAX_KAUTZ_POLES],
        }
    }

    /// Zeros any coefficients beyond `tier`'s tapered `(block_groups, fir_segments, kautz_banks)`.
    pub fn mask_to_tier(&mut self, tier: Tier) {
        let f_cnt = tier_fir_segments(tier) * 16;
        let k_cnt = tier_kautz_banks(tier) * 4;
        self.svf_direct_taps_l[7] = 0.0;
        self.svf_direct_taps_r[7] = 0.0;
        for ch in 0..2 {
            for b in 0..NUM_GRU_BLOCKS {
                for g in tier_block_groups(tier, b)..MAX_GROUPS {
                    for l in 0..4 {
                        let idx = flat_gru_lane_index(b, g, l);
                        self.pre_cab_w[ch][idx] = 0.0;
                        self.direct_s2_w[ch][idx] = 0.0;
                    }
                }
            }
        }
        for i in f_cnt..64 {
            self.fir_l[i] = 0.0;
            self.fir_r[i] = 0.0;
        }
        for m in k_cnt..MAX_KAUTZ_POLES {
            self.kautz_b_gamma[m] = 0.0;
            self.kautz_c_l[m] = (0.0, 0.0);
            self.kautz_c_r[m] = (0.0, 0.0);
        }
    }
}

/// Complete single-tier cold parameter set for a `v3` core instance.
#[derive(Clone, Debug, PartialEq)]
pub struct V3ColdTierModel {
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub num_blocks: usize,
    pub stage0: V3Stage0Params,
    pub ctrl: V3ControlBusParams,
    pub tone_svf: V3InterBlockToneSvfParams,
    pub os_params: V3OversamplerParams,
    pub svf: V3Stage1SvfParams,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockCold; NUM_GRU_BLOCKS],
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub readout: V3TierReadout,
    pub stage4: V3Stage4TankParams,
}

impl V3ColdTierModel {
    /// Builds a resting identity/unity cold model for `tier`.
    #[must_use]
    pub fn identity_for_tier(tier: Tier, os2x: bool, mono_out: bool) -> Self {
        let mut stage4 = V3Stage4TankParams::identity();
        stage4.delay_lengths = V3Stage4TankParams::default_delays_for_tier(tier);
        Self {
            tier,
            os2x,
            mono_out,
            num_blocks: tier_gru_blocks(tier),
            stage0: V3Stage0Params::identity(),
            ctrl: V3ControlBusParams::identity(),
            tone_svf: V3InterBlockToneSvfParams::identity(),
            os_params: V3OversamplerParams::default(),
            svf: V3Stage1SvfParams::identity_bypass(),
            feat_params: V3InputFeaturesParams::default(),
            blocks: [V3GruBlockCold::default(); NUM_GRU_BLOCKS],
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            readout: V3TierReadout::default(),
            stage4,
        }
    }
}

/// Unified single-run 4-tier Matryoshka model sharing one causal core across `[Lite..Ultra]`.
#[derive(Clone, Debug, PartialEq)]
pub struct V3MatryoshkaModel {
    pub os2x: bool,
    pub tier_os2x: [bool; 4],
    pub tier_blocks: [usize; 4],
    pub mono_out: bool,
    pub stage0: V3Stage0Params,
    pub ctrl: V3ControlBusParams,
    pub tone_svf: V3InterBlockToneSvfParams,
    pub os_params: V3OversamplerParams,
    pub svf: V3Stage1SvfParams,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockCold; NUM_GRU_BLOCKS],
    pub rate_1x_cal: Option<V3Rate1xCalibration>,
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub readouts: [V3TierReadout; 4],
    pub stage4: [V3Stage4TankParams; 4],
}

fn default_stage4_per_tier() -> [V3Stage4TankParams; 4] {
    [
        V3Stage4TankParams {
            delay_lengths: V3Stage4TankParams::default_delays_for_tier(Tier::Lite),
            ..V3Stage4TankParams::identity()
        },
        V3Stage4TankParams {
            delay_lengths: V3Stage4TankParams::default_delays_for_tier(Tier::Full),
            ..V3Stage4TankParams::identity()
        },
        V3Stage4TankParams {
            delay_lengths: V3Stage4TankParams::default_delays_for_tier(Tier::Pro),
            ..V3Stage4TankParams::identity()
        },
        V3Stage4TankParams {
            delay_lengths: V3Stage4TankParams::default_delays_for_tier(Tier::Ultra),
            ..V3Stage4TankParams::identity()
        },
    ]
}

impl V3MatryoshkaModel {
    /// Builds a resting identity unified 4-tier Matryoshka model with uniform oversampling mode.
    #[must_use]
    pub fn identity(os2x: bool, mono_out: bool) -> Self {
        Self {
            os2x,
            tier_os2x: [os2x; 4],
            tier_blocks: [
                tier_gru_blocks(Tier::Lite),
                tier_gru_blocks(Tier::Full),
                tier_gru_blocks(Tier::Pro),
                tier_gru_blocks(Tier::Ultra),
            ],
            mono_out,
            stage0: V3Stage0Params::identity(),
            ctrl: V3ControlBusParams::identity(),
            tone_svf: V3InterBlockToneSvfParams::identity(),
            os_params: V3OversamplerParams::default(),
            svf: V3Stage1SvfParams::identity_bypass(),
            feat_params: V3InputFeaturesParams::default(),
            blocks: [V3GruBlockCold::default(); NUM_GRU_BLOCKS],
            rate_1x_cal: None,
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            readouts: [
                V3TierReadout::default(),
                V3TierReadout::default(),
                V3TierReadout::default(),
                V3TierReadout::default(),
            ],
            stage4: default_stage4_per_tier(),
        }
    }

    /// Builds a resting dual-rate 4-tier Matryoshka model (`1x` for Lite/Full, `2x` for Pro/Ultra;
    /// `3` GRU blocks across all tiers).
    #[must_use]
    pub fn dual_rate_default(mono_out: bool) -> Self {
        let mut model = Self::identity(true, mono_out);
        model.tier_os2x = [false, false, true, true];
        model
    }

    /// Extracts an executable [`V3ColdTierModel`] view for `tier` sharing the common core.
    #[must_use]
    pub fn extract_tier(&self, tier: Tier) -> V3ColdTierModel {
        let t = tier as usize;
        let tier_os2x = self.tier_os2x[t];
        let num_blocks = self.tier_blocks[t].clamp(1, NUM_GRU_BLOCKS);
        let mut readout = self.readouts[t].clone();
        readout.mask_to_tier(tier);
        for b in num_blocks..NUM_GRU_BLOCKS {
            let base = b * MAX_BLOCK_LANES;
            for lane in base..base + MAX_BLOCK_LANES {
                readout.pre_cab_w[0][lane] = 0.0;
                readout.pre_cab_w[1][lane] = 0.0;
                readout.direct_s2_w[0][lane] = 0.0;
                readout.direct_s2_w[1][lane] = 0.0;
            }
        }
        let (svf, feat_params, blocks) = if !tier_os2x {
            if let Some(ref cal) = self.rate_1x_cal {
                let mut s = cal.svf;
                let mut f = cal.feat_params;
                let mut b = cal.blocks;
                s.clamp_in_place();
                f.clamp_in_place();
                for blk in &mut b[..num_blocks] {
                    blk.clamp_in_place();
                }
                (s, f, b)
            } else if self.os2x {
                let mut s = self.svf;
                let mut f = self.feat_params;
                let mut b = self.blocks;
                s.rescale_2x_to_1x();
                f.rescale_2x_to_1x();
                for blk in &mut b[..num_blocks] {
                    blk.rescale_2x_to_1x();
                }
                (s, f, b)
            } else {
                (self.svf, self.feat_params, self.blocks)
            }
        } else {
            (self.svf, self.feat_params, self.blocks)
        };
        V3ColdTierModel {
            tier,
            os2x: tier_os2x,
            mono_out: self.mono_out,
            num_blocks,
            stage0: self.stage0,
            ctrl: self.ctrl,
            tone_svf: self.tone_svf,
            os_params: self.os_params,
            svf,
            feat_params,
            blocks,
            kautz_poles: self.kautz_poles,
            readout,
            stage4: self.stage4[t],
        }
    }

    /// Extracts all four `[Lite, Full, Pro, Ultra]` executable models from this single-run core.
    #[must_use]
    pub fn to_four_tiers(&self) -> [V3ColdTierModel; 4] {
        [
            self.extract_tier(Tier::Lite),
            self.extract_tier(Tier::Full),
            self.extract_tier(Tier::Pro),
            self.extract_tier(Tier::Ultra),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readout_mask_and_rate_1x_cal_extraction() {
        let mut ro = V3TierReadout::default();
        ro.fir_l.fill(0.5);
        ro.fir_r.fill(0.5);
        ro.mask_to_tier(Tier::Lite);
        assert_eq!(ro.fir_l[20], 0.0);

        let mut mat = V3MatryoshkaModel::dual_rate_default(false);
        mat.svf.base_g = 0.30;
        mat.blocks[0].w_in_n[0][0] = 2.5;
        let lite_fallback = mat.extract_tier(Tier::Lite);
        assert_ne!(lite_fallback.svf.base_g, mat.svf.base_g);

        let mut cal_svf = V3Stage1SvfParams::identity_bypass();
        cal_svf.base_g = 0.42;
        let cal_feat = V3InputFeaturesParams {
            in_hp_beta: 0.08,
            ..V3InputFeaturesParams::default()
        };
        let mut cal_blocks = [V3GruBlockCold::default(); NUM_GRU_BLOCKS];
        cal_blocks[0].w_in_n[0][0] = 3.75;
        mat.rate_1x_cal = Some(V3Rate1xCalibration {
            svf: cal_svf,
            feat_params: cal_feat,
            blocks: cal_blocks,
        });

        let lite_cal = mat.extract_tier(Tier::Lite);
        let full_cal = mat.extract_tier(Tier::Full);
        let pro_2x = mat.extract_tier(Tier::Pro);
        let ultra_2x = mat.extract_tier(Tier::Ultra);
        assert_eq!(lite_cal.svf.base_g, 0.42);
        assert_eq!(full_cal.svf.base_g, 0.42);
        assert_eq!(lite_cal.blocks[0].w_in_n[0][0], 3.75);
        assert_eq!(full_cal.blocks[0].w_in_n[0][0], 3.75);
        assert_eq!(pro_2x.svf.base_g, 0.30);
        assert_eq!(ultra_2x.blocks[0].w_in_n[0][0], 2.5);
    }
}
