//! Cold tail storage, `!alloc` [`SonaSlabBacking`], and cumulative [`SonaHotSlab`] container
//! (`#![no_std]`, `#![forbid(unsafe_code)]`, `ref[arch.tiers.state-ram]`).

#![forbid(unsafe_code)]

use core::fmt;
use core::mem::offset_of;

use dnam_kernels::{HilbertH1RemapHistory, OpNPreHoisted, SonaBlockStateStereo8};
use zerocopy::FromZeros;

use crate::bin::{CrossbarWire, KautzSoaBlockWire, OpNPreBaseWire, TankQuadWire};
use crate::control::{ControlBus, ControlBusBacking};
use crate::graph::GraphBackend;
use crate::graph::cabinet::high_sr::CabinetHighSrFirExt;
use crate::graph::cabinet::kautz::{
    KautzColdSlewState, KautzStereoBank4, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS,
};
use crate::graph::oversample::{Bounce4xOversampler, SonaOs2xHot};
use crate::graph::router::IoRouter;
use crate::graph::slab::microchunk::core_step::HoistedCoreChunk;
use crate::graph::tank::ring::{PartitionedSchroederBank, TankExtRing};
use crate::graph::tank::{QuadColdMeta, QuadEffectiveTargets, TANK_MAX_QUADS, TankQuadHotCompact};
use crate::graph::zoh::ZohRateContext;
use crate::math::{DefaultMathBackend, ScalarBackend, f32_to_fp16};
use crate::types::{ColdBuf, Tier};

use super::{
    FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_OS2X, FLAG_SINGLE_INPUT,
    FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE, MAX_SONA_GROUPS, MAX_TOTAL_SONA_LANES,
    SonaFullHotDelta, SonaGroupHot, SonaGroupReadoutHot, SonaInterBlockTileHot, SonaLiteHotPrefix,
    SonaProHotDelta, SonaUltraHotDelta,
};

/// Per-tier crossbar replacement weights cached in cold storage for seamless `switch_tier`.
#[derive(Clone, Copy, Debug)]
pub struct TierCrossbarSnapshot {
    /// Group `Op-P` drive matrix (`fp16`) for all `10` SONA groups.
    pub op_p_to_group: [[u16; 4]; MAX_SONA_GROUPS],
    /// Group `Q0.32` phasor drive matrix (`fp16`) for all `10` SONA groups.
    pub phasor_to_group: [[u16; 4]; MAX_SONA_GROUPS],
    /// Tank `[L, R]` tap readout matrix (`fp16`) for `4 * 4 = 16` tank lines.
    pub opt_to_lr: [[u16; 2]; TANK_MAX_QUADS * 4],
    /// Stereo phasor offset cosines (`fp16`).
    pub phasor_st_cos: [u16; 4],
    /// Stereo phasor offset sines (`fp16`).
    pub phasor_st_sin: [u16; 4],
    /// Output channel direct/residual gains (`fp16`).
    pub ro_gains_fp16: [u16; 8],
    /// Complete-replacement group readout weights `[0..10]` active at this tier.
    pub readouts: [SonaGroupReadoutHot; MAX_SONA_GROUPS],
}

impl Default for TierCrossbarSnapshot {
    fn default() -> Self {
        Self {
            op_p_to_group: [[0; 4]; MAX_SONA_GROUPS],
            phasor_to_group: [[0; 4]; MAX_SONA_GROUPS],
            opt_to_lr: [[0; 2]; TANK_MAX_QUADS * 4],
            phasor_st_cos: [f32_to_fp16(1.0); 4],
            phasor_st_sin: [0; 4],
            ro_gains_fp16: [0; 8],
            readouts: [SonaGroupReadoutHot::default(); MAX_SONA_GROUPS],
        }
    }
}

/// Out-of-L1D cold tail holding control plane, rate/tier metadata, oversampler & tank extensions.
#[derive(Debug)]
pub struct SonaColdTail<'a, B: GraphBackend> {
    pub ctrl_bus: Option<ControlBus<'a, B>>,
    pub router: IoRouter,
    pub kautz_slew: KautzColdSlewState,
    pub cabinet_fir_ext: CabinetHighSrFirExt<'a>,
    pub tank_quad_meta: [QuadColdMeta; TANK_MAX_QUADS],
    pub tank_quad_eff: [QuadEffectiveTargets; TANK_MAX_QUADS],
    pub tank_ext: [TankExtRing<'a>; TANK_MAX_QUADS],
    pub tank_schroeder: [PartitionedSchroederBank<'a>; TANK_MAX_QUADS],
    pub bounce4x: Bounce4xOversampler,
    pub wire_crossbar: CrossbarWire,
    pub wire_kautz: [KautzSoaBlockWire; MAX_KAUTZ_BANKS],
    pub wire_tank_quads: [TankQuadWire; TANK_MAX_QUADS],
    pub base_groups: [SonaGroupHot; MAX_SONA_GROUPS],
    pub tier_fir_l: [[f32; 128]; 4],
    pub tier_fir_r: [[f32; 128]; 4],
    pub tier_kautz_b_gamma: [[f32; MAX_KAUTZ_PAIRS]; 4],
    pub tier_kautz_c_l: [[(f32, f32); MAX_KAUTZ_PAIRS]; 4],
    pub tier_kautz_c_r: [[(f32, f32); MAX_KAUTZ_PAIRS]; 4],
    pub tier_crossbars: [TierCrossbarSnapshot; 4],
    pub base_phasor_freq_hz: [f32; 4],
    pub base_phasor_init_phase_q32: [u32; 4],
    pub base_op_p_gain: f32,
    pub base_op_t_gain: f32,
    pub base_op_n_pre: OpNPreBaseWire,
    pub stage_latent_mods_fp16: [u16; 96],
    pub stage_latent_ctrl_ultra_fp16: [u16; 256],
    pub stage_latent_core_mods_fp16: [u16; 1_024],
    pub tier_os2x: [bool; 4],
    pub tier_gain_trim: [f32; 4],
    pub core_mods_present: bool,
    pub core_mods_active: bool,
    pub meta_flags: u32,
    pub max_present_tier: Tier,
    pub retained_tier: Tier,
    pub zoh: ZohRateContext,
    pub latched_stage_params: [f32; 32],
    pub latched_z_stage: [f32; 8],
    pub lane_g_scale: [f32; MAX_TOTAL_SONA_LANES],
    pub lane_k_delta: [f32; MAX_TOTAL_SONA_LANES],
    pub lane_wf_delta: [f32; MAX_TOTAL_SONA_LANES],
    pub group_adaa_scale: [f32; MAX_SONA_GROUPS],
    pub group_lru_bias: [f32; MAX_SONA_GROUPS],
    pub op_n_pre_hoisted: OpNPreHoisted,
    pub rem_hist: HilbertH1RemapHistory,
    pub xfade_rem: u16,
    pub xfade_prev_out: [f32; 2],
    pub last_in: [f32; 2],
    pub last_s2_dir: [f32; 4],
    pub last_op_p_iq_lr: [f32; 4],
    pub last_op_p_feat: [f32; 4],
    pub saturation_flags: u32,
    pub offline_bounce: bool,
    pub declared_max_delay_48k: u32,
    pub ext_ring_pow2_frames: u8,
    pub dc_blocker_pole_5hz: f32,
    pub active_kautz_blocks: usize,
    pub active_tank_quads: usize,
    pub hoisted_core: HoistedCoreChunk<B>,
    pub hoisted_core_valid: bool,
}

impl<B: GraphBackend> SonaColdTail<'_, B> {
    /// Creates an unhydrated cold tail with zeroed buffers for `!alloc` initialization.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            ctrl_bus: None,
            router: IoRouter::default(),
            kautz_slew: KautzColdSlewState::default(),
            cabinet_fir_ext: CabinetHighSrFirExt::empty(),
            tank_quad_meta: [QuadColdMeta::default(); TANK_MAX_QUADS],
            tank_quad_eff: [QuadEffectiveTargets::default(); TANK_MAX_QUADS],
            tank_ext: [
                TankExtRing::none(),
                TankExtRing::none(),
                TankExtRing::none(),
                TankExtRing::none(),
            ],
            tank_schroeder: [
                PartitionedSchroederBank::disabled(),
                PartitionedSchroederBank::disabled(),
                PartitionedSchroederBank::disabled(),
                PartitionedSchroederBank::disabled(),
            ],
            bounce4x: Bounce4xOversampler::new(),
            wire_crossbar: CrossbarWire::new_zeroed(),
            wire_kautz: [KautzSoaBlockWire::new_zeroed(); MAX_KAUTZ_BANKS],
            wire_tank_quads: [TankQuadWire::new_zeroed(); TANK_MAX_QUADS],
            base_groups: [SonaGroupHot::default(); MAX_SONA_GROUPS],
            tier_fir_l: [[0.0; 128]; 4],
            tier_fir_r: [[0.0; 128]; 4],
            tier_kautz_b_gamma: [[0.0; MAX_KAUTZ_PAIRS]; 4],
            tier_kautz_c_l: [[(0.0, 0.0); MAX_KAUTZ_PAIRS]; 4],
            tier_kautz_c_r: [[(0.0, 0.0); MAX_KAUTZ_PAIRS]; 4],
            tier_crossbars: [TierCrossbarSnapshot::default(); 4],
            base_phasor_freq_hz: [0.0; 4],
            base_phasor_init_phase_q32: [0; 4],
            base_op_p_gain: 0.0,
            base_op_t_gain: 0.0,
            base_op_n_pre: OpNPreBaseWire::new_zeroed(),
            stage_latent_mods_fp16: [0; 96],
            stage_latent_ctrl_ultra_fp16: [0; 256],
            stage_latent_core_mods_fp16: [0; 1_024],
            tier_os2x: [false; 4],
            tier_gain_trim: [1.0; 4],
            core_mods_present: false,
            core_mods_active: false,
            meta_flags: 0,
            max_present_tier: Tier::Lite,
            retained_tier: Tier::Lite,
            zoh: ZohRateContext::default(),
            latched_stage_params: [0.0; 32],
            latched_z_stage: [0.0; 8],
            lane_g_scale: [1.0; MAX_TOTAL_SONA_LANES],
            lane_k_delta: [0.0; MAX_TOTAL_SONA_LANES],
            lane_wf_delta: [0.0; MAX_TOTAL_SONA_LANES],
            group_adaa_scale: [1.0; MAX_SONA_GROUPS],
            group_lru_bias: [0.0; MAX_SONA_GROUPS],
            op_n_pre_hoisted: OpNPreHoisted::from_params(0.0, 0.0, 0.0, 1.0),
            rem_hist: HilbertH1RemapHistory::default(),
            xfade_rem: 0,
            xfade_prev_out: [0.0; 2],
            last_in: [0.0; 2],
            last_s2_dir: [0.0; 4],
            last_op_p_iq_lr: [0.0; 4],
            last_op_p_feat: [0.0; 4],
            saturation_flags: 0,
            offline_bounce: false,
            declared_max_delay_48k: 4_096,
            ext_ring_pow2_frames: 0,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            active_kautz_blocks: 2,
            active_tank_quads: 1,
            hoisted_core: HoistedCoreChunk::empty(),
            hoisted_core_valid: false,
        }
    }
}

impl<B: GraphBackend> Default for SonaColdTail<'_, B> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Borrowed `!alloc` cold storage slices required to hydrate a [`SonaHotSlab`].
pub struct SonaSlabBacking<'a, B: GraphBackend> {
    pub control_bus: ControlBusBacking<'a>,
    pub cabinet_fir_ext: &'a mut [f32],
    pub tank_ext_fp16: [&'a mut [u16]; TANK_MAX_QUADS],
    pub tank_ext_f32: [&'a mut [f32]; TANK_MAX_QUADS],
    pub tank_schroeder: [&'a mut [f32]; TANK_MAX_QUADS],
    pub cold_tail: &'a mut [SonaColdTail<'a, B>],
}

impl<B: GraphBackend> fmt::Debug for SonaSlabBacking<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SonaSlabBacking")
            .field("cabinet_fir_ext_len", &self.cabinet_fir_ext.len())
            .field("cold_tail_len", &self.cold_tail.len())
            .finish_non_exhaustive()
    }
}

/// Prefix-ordered `64 B`-aligned `SONA v3.0` L1D hot execution slab (`4,544..14,656 B` active hot).
#[repr(C, align(64))]
#[derive(Debug)]
pub struct SonaHotSlab<'a, B: GraphBackend = DefaultMathBackend> {
    pub lite: SonaLiteHotPrefix<B>,
    pub os2x_hot: SonaOs2xHot,
    pub full_delta: SonaFullHotDelta<B>,
    pub pro_delta: SonaProHotDelta<B>,
    pub ultra_delta: SonaUltraHotDelta<B>,
    pub cold: ColdBuf<'a, SonaColdTail<'a, B>>,
}

impl<'a, B: GraphBackend> SonaHotSlab<'a, B> {
    /// Constructs a zeroed slab bound to a `ColdBuf<SonaColdTail>` (`!alloc`).
    #[must_use]
    pub fn empty_with_cold(cold: ColdBuf<'a, SonaColdTail<'a, B>>) -> Self {
        Self {
            lite: SonaLiteHotPrefix::empty(),
            os2x_hot: SonaOs2xHot::default(),
            full_delta: SonaFullHotDelta::empty(),
            pro_delta: SonaProHotDelta::empty(),
            ultra_delta: SonaUltraHotDelta::empty(),
            cold,
        }
    }

    /// Active prefix hot working-set byte footprint for `(tier, os2x)`.
    #[inline(always)]
    #[must_use]
    pub const fn active_hot_bytes(tier: Tier, os2x: bool) -> usize {
        let lite = size_of::<SonaLiteHotPrefix<B>>();
        let os = size_of::<SonaOs2xHot>();
        let full = size_of::<SonaFullHotDelta<B>>();
        let pro = size_of::<SonaProHotDelta<B>>();
        let ultra = size_of::<SonaUltraHotDelta<B>>();
        match tier {
            Tier::Lite => {
                if os2x {
                    lite + os
                } else {
                    lite
                }
            }
            Tier::Full => lite + os + full,
            Tier::Pro => lite + os + full + pro,
            Tier::Ultra => lite + os + full + pro + ultra,
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn os2x(&self) -> bool {
        (self.lite.flags & FLAG_OS2X) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn mono_out(&self) -> bool {
        (self.lite.flags & FLAG_MONO_OUT) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn crossbar_active(&self) -> bool {
        (self.lite.flags & FLAG_CROSSBAR_ACTIVE) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn xfade_active(&self) -> bool {
        (self.lite.flags & FLAG_XFADE_ACTIVE) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn cold_fir_ext(&self) -> bool {
        (self.lite.flags & FLAG_COLD_FIR_EXT) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn single_input(&self) -> bool {
        (self.lite.flags & FLAG_SINGLE_INPUT) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn wet_polarity_inv(&self) -> bool {
        (self.lite.flags & FLAG_WET_POLARITY_INV) != 0
    }

    #[inline(always)]
    pub(crate) fn set_flag(&mut self, mask: u8, on: bool) {
        if on {
            self.lite.flags |= mask;
        } else {
            self.lite.flags &= !mask;
        }
    }

    #[inline(always)]
    pub(crate) fn state_mut(&mut self, g: usize) -> &mut SonaBlockStateStereo8<B::Stereo8> {
        match g {
            0..=3 => &mut self.lite.core_states[g],
            4..=5 => &mut self.full_delta.core_states_full[g - 4],
            6..=7 => &mut self.pro_delta.core_states_pro[g - 6],
            _ => &mut self.ultra_delta.core_states_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn group_hot(&self, g: usize) -> &SonaGroupHot {
        match g {
            0..=3 => &self.lite.groups[g],
            4..=5 => &self.full_delta.groups_full[g - 4],
            6..=7 => &self.pro_delta.groups_pro[g - 6],
            _ => &self.ultra_delta.groups_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn group_hot_mut(&mut self, g: usize) -> &mut SonaGroupHot {
        match g {
            0..=3 => &mut self.lite.groups[g],
            4..=5 => &mut self.full_delta.groups_full[g - 4],
            6..=7 => &mut self.pro_delta.groups_pro[g - 6],
            _ => &mut self.ultra_delta.groups_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn tile_hot_mut(&mut self, t: usize) -> &mut SonaInterBlockTileHot {
        match t {
            0..=2 => &mut self.lite.tiles[t],
            3..=12 => &mut self.full_delta.tiles_full[t - 3],
            13..=22 => &mut self.pro_delta.tiles_pro[t - 13],
            _ => &mut self.ultra_delta.tiles_ultra[t - 23],
        }
    }

    #[inline(always)]
    pub(crate) fn readout_hot(&self, g: usize) -> &SonaGroupReadoutHot {
        match g {
            0..=3 => &self.lite.readouts[g],
            4..=5 => &self.full_delta.readouts_full[g - 4],
            6..=7 => &self.pro_delta.readouts_pro[g - 6],
            _ => &self.ultra_delta.readouts_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn readout_hot_mut(&mut self, g: usize) -> &mut SonaGroupReadoutHot {
        match g {
            0..=3 => &mut self.lite.readouts[g],
            4..=5 => &mut self.full_delta.readouts_full[g - 4],
            6..=7 => &mut self.pro_delta.readouts_pro[g - 6],
            _ => &mut self.ultra_delta.readouts_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn op_p_to_group_fp16(&self, g: usize) -> &[u16; 4] {
        match g {
            0..=3 => &self.lite.op_p_to_group_lite_fp16[g],
            4..=5 => &self.full_delta.op_p_to_group_full_fp16[g - 4],
            6..=7 => &self.pro_delta.op_p_to_group_pro_fp16[g - 6],
            _ => &self.ultra_delta.op_p_to_group_ultra_fp16[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn phasor_to_group_fp16(&self, g: usize) -> &[u16; 4] {
        match g {
            0..=3 => &self.lite.phasor_to_group_lite_fp16[g],
            4..=5 => &self.full_delta.phasor_to_group_full_fp16[g - 4],
            6..=7 => &self.pro_delta.phasor_to_group_pro_fp16[g - 6],
            _ => &self.ultra_delta.phasor_to_group_ultra_fp16[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn opt_to_lr_fp16(&self, idx: usize) -> [u16; 2] {
        match idx {
            0..=3 => self.lite.opt_to_lr_lite_fp16[idx],
            4..=7 => self.full_delta.opt_to_lr_full_fp16[idx - 4],
            8..=11 => self.pro_delta.opt_to_lr_pro_fp16[idx - 8],
            _ => self.ultra_delta.opt_to_lr_ultra_fp16[idx - 12],
        }
    }

    #[inline(always)]
    pub(crate) fn rect_mask4(&self, g: usize) -> u8 {
        match g {
            0..=3 => ((self.lite.rect_sign_mask_lite >> (g * 4)) & 0x0F) as u8,
            4..=5 => (self.full_delta.rect_sign_mask_full >> ((g - 4) * 4)) & 0x0F,
            6..=7 => (self.pro_delta.rect_sign_mask_pro >> ((g - 6) * 4)) & 0x0F,
            _ => (self.ultra_delta.rect_sign_mask_ultra >> ((g - 8) * 4)) & 0x0F,
        }
    }

    #[inline(always)]
    pub(crate) fn quad_hot(&self, q: usize) -> &TankQuadHotCompact {
        match q {
            0 => &self.lite.tank_quad0,
            1 => &self.full_delta.tank_quad1,
            2 => &self.pro_delta.tank_quad2,
            _ => &self.ultra_delta.tank_quad3,
        }
    }

    #[inline(always)]
    pub(crate) fn quad_hot_mut(&mut self, q: usize) -> &mut TankQuadHotCompact {
        match q {
            0 => &mut self.lite.tank_quad0,
            1 => &mut self.full_delta.tank_quad1,
            2 => &mut self.pro_delta.tank_quad2,
            _ => &mut self.ultra_delta.tank_quad3,
        }
    }

    #[inline(always)]
    pub(crate) fn kautz_bank_mut(&mut self, b: usize) -> &mut KautzStereoBank4 {
        match b {
            0..=1 => &mut self.lite.kautz[b],
            2 => &mut self.full_delta.kautz_full[0],
            3..=5 => &mut self.pro_delta.kautz_pro[b - 3],
            _ => &mut self.ultra_delta.kautz_ultra[b - 6],
        }
    }

    /// Immutable reference to [`SonaColdTail`].
    #[inline(always)]
    pub fn cold(&self) -> &SonaColdTail<'a, B> {
        &self.cold.as_slice()[0]
    }

    /// Mutable reference to [`SonaColdTail`].
    #[inline(always)]
    pub fn cold_mut(&mut self) -> &mut SonaColdTail<'a, B> {
        &mut self.cold.as_mut_slice()[0]
    }
}

const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, lite) == 0);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, os2x_hot) == 4_544);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, full_delta) == 4_800);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, pro_delta) == 7_552);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, ultra_delta) == 11_200);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, cold) == 14_656);

const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, lite) == 0);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, os2x_hot) == 4_544);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, full_delta) == 4_800);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, pro_delta) == 7_552);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, ultra_delta) == 11_200);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, cold) == 14_656);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_cold_empty_and_hot_prefix_footprints() {
        let mut cold_slot = [SonaColdTail::<ScalarBackend>::empty()];
        let slab = SonaHotSlab::<ScalarBackend>::empty_with_cold(ColdBuf::borrowed(&mut cold_slot));
        assert_eq!(slab.lite.tier, Tier::Lite);
        assert_eq!(
            SonaHotSlab::<ScalarBackend>::active_hot_bytes(Tier::Lite, false),
            4_544
        );
        assert_eq!(
            SonaHotSlab::<ScalarBackend>::active_hot_bytes(Tier::Lite, true),
            4_800
        );
        assert_eq!(
            SonaHotSlab::<ScalarBackend>::active_hot_bytes(Tier::Full, false),
            7_552
        );
        assert_eq!(
            SonaHotSlab::<ScalarBackend>::active_hot_bytes(Tier::Pro, true),
            11_200
        );
        assert_eq!(
            SonaHotSlab::<ScalarBackend>::active_hot_bytes(Tier::Ultra, true),
            14_656
        );
    }
}
