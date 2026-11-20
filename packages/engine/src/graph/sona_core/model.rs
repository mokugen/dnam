//! Cold `SONA` Matryoshka Model Facade, Continuous-Time ZOH Scaling & Tier Extraction
//! (`graph/sona_core/model.rs`).

pub mod ctrl;
pub mod params;
pub mod tier;

use dnam_kernels::{FULL_SPEC, LITE_SPEC, PRO_SPEC, ULTRA_SPEC};

pub use ctrl::{CORE_MODS_STRIDE, SonaControlBusParams, SonaCrossbarConfig};
pub use params::{SonaGroupCold, SonaTierReadout, SonaTileCold};
pub use tier::{
    FIR_RING_CAP, FIR_RING_MASK, MAX_FIR_SEGMENTS, MAX_GROUPS_PER_BLOCK, MAX_KAUTZ_BANKS,
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_STAGE_LATENTS, MAX_TANK_LINES,
    MAX_TANK_QUADS, MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, NUM_SONA_MACRO_KNOBS,
    SONA_BLOCK_GROUP_SLOTS, SONA_FIR_MAX_TAPS, SONA_GROUP_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES,
    count_interblock_tiles, flat_sona_lane_index, tier_block_groups_array, tier_fir_segments,
    tier_kautz_banks, tier_sona_groups, tier_spec, tier_tank_quads, tier_tile_count,
};

use super::frontend::SonaOversamplerParams;
use crate::graph::router::CrossbarParams;
use crate::graph::tank::fdn::SpatialTankParams;
use crate::types::Tier;

/// Complete single-tier cold parameter bundle for constructing a [`crate::graph::SonaInstance`].
#[derive(Clone, Debug, PartialEq)]
pub struct SonaColdTierModel {
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub stage0: CrossbarParams,
    pub crossbar: SonaCrossbarConfig,
    pub ctrl: SonaControlBusParams,
    pub os_params: SonaOversamplerParams,
    pub groups: [SonaGroupCold; MAX_SONA_GROUPS],
    pub tiles: [SonaTileCold; MAX_SONA_TILES],
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub readout: SonaTierReadout,
    pub stage4: SpatialTankParams,
}

impl SonaColdTierModel {
    /// Builds a resting zero-DC identity cold model for `tier`.
    #[must_use]
    pub fn identity_for_tier(tier: Tier, os2x: bool, mono_out: bool) -> Self {
        let stage4 = SpatialTankParams::identity_for_tier(tier);
        Self {
            tier,
            os2x,
            mono_out,
            stage0: CrossbarParams::identity(),
            crossbar: SonaCrossbarConfig::identity(),
            ctrl: SonaControlBusParams::identity(),
            os_params: SonaOversamplerParams::default(),
            groups: core::array::from_fn(SonaGroupCold::identity_group),
            tiles: [SonaTileCold::zeros(); MAX_SONA_TILES],
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            readout: SonaTierReadout::default(),
            stage4,
        }
    }
}

/// Unified single-run `4`-tier Matryoshka `SONA` model sharing one prefix-nested core across `[Lite..Ultra]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SonaMatryoshkaModel {
    pub os2x: bool,
    pub tier_os2x: [bool; 4],
    pub mono_out: bool,
    pub stage0: CrossbarParams,
    pub crossbar: SonaCrossbarConfig,
    pub tier_crossbars: Option<[SonaCrossbarConfig; 4]>,
    pub ctrl: SonaControlBusParams,
    pub os_params: SonaOversamplerParams,
    pub groups: [SonaGroupCold; MAX_SONA_GROUPS],
    pub tiles: [SonaTileCold; MAX_SONA_TILES],
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub readouts: [SonaTierReadout; 4],
    pub stage4: [SpatialTankParams; 4],
}

fn default_stage4_per_tier() -> [SpatialTankParams; 4] {
    [
        SpatialTankParams::identity_for_tier(Tier::Lite),
        SpatialTankParams::identity_for_tier(Tier::Full),
        SpatialTankParams::identity_for_tier(Tier::Pro),
        SpatialTankParams::identity_for_tier(Tier::Ultra),
    ]
}

impl SonaMatryoshkaModel {
    /// Builds a resting identity `SONA` Matryoshka model with uniform oversampling mode.
    #[must_use]
    pub fn identity(os2x: bool, mono_out: bool) -> Self {
        Self {
            os2x,
            tier_os2x: [os2x; 4],
            mono_out,
            stage0: CrossbarParams::identity(),
            crossbar: SonaCrossbarConfig::identity(),
            tier_crossbars: None,
            ctrl: SonaControlBusParams::identity(),
            os_params: SonaOversamplerParams::default(),
            groups: core::array::from_fn(SonaGroupCold::identity_group),
            tiles: [SonaTileCold::zeros(); MAX_SONA_TILES],
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            readouts: [
                SonaTierReadout::default(),
                SonaTierReadout::default(),
                SonaTierReadout::default(),
                SonaTierReadout::default(),
            ],
            stage4: default_stage4_per_tier(),
        }
    }

    /// Builds a resting dual-rate `SONA` `4`-tier model matching [`tier_spec`] oversample factors
    /// (`1x` for `Lite`, `2x` for `Full..Ultra`).
    #[must_use]
    pub fn dual_rate_default(mono_out: bool) -> Self {
        let mut model = Self::identity(true, mono_out);
        model.tier_os2x = [
            LITE_SPEC.oversample_factor > 1,
            FULL_SPEC.oversample_factor > 1,
            PRO_SPEC.oversample_factor > 1,
            ULTRA_SPEC.oversample_factor > 1,
        ];
        model
    }

    /// Extracts an executable [`SonaColdTierModel`] view for `tier` sharing the common core.
    #[must_use]
    pub fn extract_tier(&self, tier: Tier) -> SonaColdTierModel {
        let t = tier as usize;
        let tier_os2x = self.tier_os2x[t];
        let mut readout = self.readouts[t].clone();
        readout.mask_to_tier(tier);
        let mut groups = self.groups;
        let mut tiles = self.tiles;
        for g in &mut groups {
            g.clamp_in_place();
        }
        for tile in &mut tiles {
            tile.clamp_in_place();
        }
        let mut crossbar = self
            .tier_crossbars
            .as_ref()
            .map_or(self.crossbar, |cbs| cbs[t]);
        if tier == Tier::Lite {
            crossbar.phasor_stereo_phase_rad = [0.0; 4];
        }
        let mut ctrl = self.ctrl;
        if tier != Tier::Ultra && ctrl.lite_block_scales_present {
            for k in 0..4 {
                for b in 0..4 {
                    ctrl.w_adaa_drive_scale[k][b] = ctrl.w_adaa_drive_scale_lite[k][b];
                    ctrl.w_lru_bias_shift[k][b] = ctrl.w_lru_bias_shift_lite[k][b];
                }
            }
        }
        SonaColdTierModel {
            tier,
            os2x: tier_os2x,
            mono_out: self.mono_out,
            stage0: self.stage0,
            crossbar,
            ctrl,
            os_params: self.os_params,
            groups,
            tiles,
            kautz_poles: self.kautz_poles,
            readout,
            stage4: self.stage4[t],
        }
    }

    /// Extracts all four `[Lite, Full, Pro, Ultra]` executable models from this single-run core.
    #[must_use]
    pub fn to_four_tiers(&self) -> [SonaColdTierModel; 4] {
        [
            self.extract_tier(Tier::Lite),
            self.extract_tier(Tier::Full),
            self.extract_tier(Tier::Pro),
            self.extract_tier(Tier::Ultra),
        ]
    }

    /// Returns maximum Stage III Kautz modal pole pairs supported across `Arena 0..3`.
    #[must_use]
    pub const fn max_kautz_poles() -> usize {
        MAX_KAUTZ_POLES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sona_matryoshka_extract_tiers() {
        let mut mat = SonaMatryoshkaModel::dual_rate_default(false);
        mat.groups[0].svf_g[0] = 0.25;
        mat.groups[0].clamp_in_place();
        assert!(!mat.tier_os2x[0] && mat.tier_os2x[1]);
        let four = mat.to_four_tiers();
        assert_eq!(four[0].tier, Tier::Lite);
        assert_eq!(four[3].tier, Tier::Ultra);
        // Canonical 48,000 Hz anchor: 1x tiers preserve unscaled base parameters instead of rescaling 2x -> 1x.
        assert_eq!(four[0].groups[0].svf_g[0], four[1].groups[0].svf_g[0]);
        assert_eq!(SonaMatryoshkaModel::max_kautz_poles(), 32);

        let mut per_tier = SonaMatryoshkaModel::identity(false, false);
        let mut cbs = [SonaCrossbarConfig::identity(); 4];
        cbs[0].op_p_to_group[0][0] = 0.25;
        cbs[0].phasor_stereo_phase_rad = [0.5; 4];
        cbs[1].op_p_to_group[0][0] = 0.5;
        cbs[2].op_p_to_group[0][0] = 0.75;
        cbs[3].op_p_to_group[0][0] = 1.0;
        per_tier.tier_crossbars = Some(cbs);
        per_tier.ctrl.lite_block_scales_present = true;
        per_tier.ctrl.w_adaa_drive_scale_lite[1][2] = 0.42;
        per_tier.ctrl.w_lru_bias_shift_lite[2][1] = -0.31;
        per_tier.ctrl.w_adaa_drive_scale[1][2] = 0.99;
        per_tier.ctrl.w_lru_bias_shift[2][1] = -0.88;

        let lite = per_tier.extract_tier(Tier::Lite);
        let full = per_tier.extract_tier(Tier::Full);
        let ultra = per_tier.extract_tier(Tier::Ultra);
        assert_eq!(lite.crossbar.op_p_to_group[0][0], 0.25);
        assert_eq!(lite.crossbar.phasor_stereo_phase_rad, [0.0; 4]);
        assert_eq!(full.crossbar.op_p_to_group[0][0], 0.5);
        assert_eq!(ultra.crossbar.op_p_to_group[0][0], 1.0);
        assert_eq!(lite.ctrl.w_adaa_drive_scale[1][2], 0.42);
        assert_eq!(full.ctrl.w_lru_bias_shift[2][1], -0.31);
        assert_eq!(ultra.ctrl.w_adaa_drive_scale[1][2], 0.99);
        assert_eq!(ultra.ctrl.w_lru_bias_shift[2][1], -0.88);
    }
}
