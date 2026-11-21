//! Cold tail storage, `!alloc` [`SonaSlabBacking`], and cumulative [`SonaHotSlab`] container
//! (`#![no_std]`, `#![forbid(unsafe_code)]`, `ref[arch.tiers.state-ram]`).

#![forbid(unsafe_code)]

use core::fmt;
use core::mem::offset_of;

use dnam_kernels::{HilbertH1RemapHistory, OpNPreHoisted, SonaBlockStateStereo8};
use zerocopy::FromZeros;

use crate::bin::{CrossbarWire, KautzSoaBlockWire, OpNPreBaseWire, TransportQuadWire};
use crate::control::{ControlBus, ControlBusBacking};
use crate::graph::GraphBackend;
use crate::graph::convolver::high_sr::ConvolverHighSrFirExt;
use crate::graph::convolver::kautz::{
    KautzColdSlewState, KautzStereoBank4, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS,
};
use crate::graph::oversample::{Bounce4xDecimator, Bounce4xOversampler, SonaOs2xHot};
use crate::graph::sensory::SensoryQuadCold;
use crate::graph::slab::microchunk::core_step::HoistedCoreChunk;
use crate::graph::transport::ring::{PartitionedSchroederBank, TransportExtRing};
use crate::graph::transport::{
    QuadColdMeta, QuadEffectiveTargets, TRANSPORT_MAX_QUADS, TransportQuadHotCompact,
};
use crate::graph::zoh::ZohRateContext;
use crate::math::{DefaultMathBackend, ScalarBackend, f32_to_fp16};
use crate::types::{ColdBuf, Tier};

use super::{
    CoreQuadExtCold, CoreQuadHot, CoreQuadReadoutHot, FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE,
    FLAG_MONO_OUT, FLAG_OS2X, FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE,
    MAX_CORE_QUADS, MAX_TOTAL_SONA_LANES, SONA_FULL_HOT_DELTA_BYTES, SONA_LITE_HOT_BYTES,
    SONA_OS2X_HOT_BYTES, SONA_PRO_HOT_DELTA_BYTES, SONA_ULTRA_HOT_DELTA_BYTES, SonaFullHotDelta,
    SonaInterBlockTileHot, SonaLiteHotPrefix, SonaProHotDelta, SonaUltraHotDelta,
};

const CUM_OS2X_OFFSET: usize = SONA_LITE_HOT_BYTES;
const CUM_FULL_OFFSET: usize = CUM_OS2X_OFFSET + SONA_OS2X_HOT_BYTES;
const CUM_PRO_OFFSET: usize = CUM_FULL_OFFSET + SONA_FULL_HOT_DELTA_BYTES;
const CUM_ULTRA_OFFSET: usize = CUM_PRO_OFFSET + SONA_PRO_HOT_DELTA_BYTES;
const CUM_COLD_OFFSET: usize = CUM_ULTRA_OFFSET + SONA_ULTRA_HOT_DELTA_BYTES;

const _: () = assert!(CUM_OS2X_OFFSET == 4_544);
const _: () = assert!(CUM_FULL_OFFSET == 4_800);
const _: () = assert!(CUM_PRO_OFFSET == 7_552);
const _: () = assert!(CUM_ULTRA_OFFSET == 11_136);
const _: () = assert!(CUM_COLD_OFFSET == 14_592);

/// Per-tier crossbar replacement weights cached in cold storage for seamless `switch_tier`.
#[derive(Clone, Copy, Debug)]
pub struct TierCrossbarSnapshot {
    /// `CoreQuad` `Op-P` drive matrix (`fp16`) for all `10` SONA `CoreQuad`s.
    pub op_p_to_core: [[u16; 4]; MAX_CORE_QUADS],
    /// `CoreQuad` `Q0.32` phasor drive matrix (`fp16`) for all `10` SONA `CoreQuad`s.
    pub phasor_to_core: [[u16; 4]; MAX_CORE_QUADS],
    /// Transport `[L, R]` tap readout matrix (`fp16`) for `4 * 4 = 16` transport lines.
    pub opt_to_lr: [[u16; 2]; TRANSPORT_MAX_QUADS * 4],
    /// Stereo phasor offset cosines (`fp16`).
    pub phasor_st_cos: [u16; 4],
    /// Stereo phasor offset sines (`fp16`).
    pub phasor_st_sin: [u16; 4],
    /// Output channel direct/residual gains (`fp16`).
    pub ro_gains_fp16: [u16; 8],
    /// Complete-replacement `CoreQuad` readout weights `[0..10]` active at this tier.
    pub readouts: [CoreQuadReadoutHot; MAX_CORE_QUADS],
}

impl Default for TierCrossbarSnapshot {
    fn default() -> Self {
        Self {
            op_p_to_core: [[0; 4]; MAX_CORE_QUADS],
            phasor_to_core: [[0; 4]; MAX_CORE_QUADS],
            opt_to_lr: [[0; 2]; TRANSPORT_MAX_QUADS * 4],
            phasor_st_cos: [f32_to_fp16(1.0); 4],
            phasor_st_sin: [0; 4],
            ro_gains_fp16: [0; 8],
            readouts: [CoreQuadReadoutHot::default(); MAX_CORE_QUADS],
        }
    }
}

/// Out-of-L1D cold tail holding control plane, sensory bank, rate/tier metadata, oversampler, and
/// `ModalConvolver` / `Transport` cold extensions.
#[derive(Debug)]
pub struct SonaColdTail<'a, B: GraphBackend> {
    pub ctrl_bus: Option<ControlBus<'a, B>>,
    pub sensory_cold: SensoryQuadCold,
    pub kautz_slew: KautzColdSlewState,
    pub convolver_fir_ext: ConvolverHighSrFirExt<'a>,
    pub transport_quad_meta: [QuadColdMeta; TRANSPORT_MAX_QUADS],
    pub transport_quad_eff: [QuadEffectiveTargets; TRANSPORT_MAX_QUADS],
    pub transport_ext: [TransportExtRing<'a>; TRANSPORT_MAX_QUADS],
    pub transport_schroeder: [PartitionedSchroederBank<'a>; TRANSPORT_MAX_QUADS],
    pub bounce4x: Bounce4xOversampler,
    pub bounce4x_core_decim: Bounce4xDecimator,
    pub bounce4x_wet_decim: Bounce4xDecimator,
    pub wire_crossbar: CrossbarWire,
    pub wire_kautz: [KautzSoaBlockWire; MAX_KAUTZ_BANKS],
    pub wire_transport_quads: [TransportQuadWire; TRANSPORT_MAX_QUADS],
    pub base_core_quads: [CoreQuadHot; MAX_CORE_QUADS],
    pub base_core_ext: [CoreQuadExtCold; MAX_CORE_QUADS],
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
    pub base_tank_dry_mix: f32,
    pub ro_pole_48k: f32,
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
    pub quad_adaa_scale: [f32; MAX_CORE_QUADS],
    pub quad_lru_bias: [f32; MAX_CORE_QUADS],
    pub op_n_pre_hoisted: OpNPreHoisted,
    pub rem_hist: HilbertH1RemapHistory,
    pub xfade_rem: u16,
    pub xfade_prev_out: [f32; 2],
    pub xfade_prev_core: [f32; 2],
    pub xfade_prev_wet: [f32; 2],
    pub last_in: [f32; 2],
    pub last_s2_dir: [f32; 4],
    pub last_op_p_iq_lr: [f32; 4],
    pub last_op_p_feat: [f32; 4],
    pub last_core_lr: [f32; 2],
    pub last_transport_wet_lr: [f32; 2],
    pub saturation_flags: u32,
    pub offline_bounce: bool,
    pub declared_max_delay_48k: u32,
    pub ext_ring_pow2_frames: u8,
    pub dc_blocker_pole_5hz: f32,
    pub active_kautz_blocks: usize,
    pub active_transport_quads: usize,
    pub hoisted_core: HoistedCoreChunk<B>,
    pub hoisted_core_valid: bool,
}

impl<B: GraphBackend> SonaColdTail<'_, B> {
    /// Creates an unhydrated cold tail with zeroed buffers for `!alloc` initialization.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            ctrl_bus: None,
            sensory_cold: SensoryQuadCold::zero(),
            kautz_slew: KautzColdSlewState::default(),
            convolver_fir_ext: ConvolverHighSrFirExt::empty(),
            transport_quad_meta: [QuadColdMeta::default(); TRANSPORT_MAX_QUADS],
            transport_quad_eff: [QuadEffectiveTargets::default(); TRANSPORT_MAX_QUADS],
            transport_ext: [
                TransportExtRing::none(),
                TransportExtRing::none(),
                TransportExtRing::none(),
                TransportExtRing::none(),
            ],
            transport_schroeder: [
                PartitionedSchroederBank::disabled(),
                PartitionedSchroederBank::disabled(),
                PartitionedSchroederBank::disabled(),
                PartitionedSchroederBank::disabled(),
            ],
            bounce4x: Bounce4xOversampler::new(),
            bounce4x_core_decim: Bounce4xDecimator::new(),
            bounce4x_wet_decim: Bounce4xDecimator::new(),
            wire_crossbar: CrossbarWire::new_zeroed(),
            wire_kautz: [KautzSoaBlockWire::new_zeroed(); MAX_KAUTZ_BANKS],
            wire_transport_quads: [TransportQuadWire::new_zeroed(); TRANSPORT_MAX_QUADS],
            base_core_quads: [CoreQuadHot::default(); MAX_CORE_QUADS],
            base_core_ext: [CoreQuadExtCold::default(); MAX_CORE_QUADS],
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
            base_tank_dry_mix: 0.0,
            ro_pole_48k: 0.0,
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
            quad_adaa_scale: [1.0; MAX_CORE_QUADS],
            quad_lru_bias: [0.0; MAX_CORE_QUADS],
            op_n_pre_hoisted: OpNPreHoisted::bypass(),
            rem_hist: HilbertH1RemapHistory::default(),
            xfade_rem: 0,
            xfade_prev_out: [0.0; 2],
            xfade_prev_core: [0.0; 2],
            xfade_prev_wet: [0.0; 2],
            last_in: [0.0; 2],
            last_s2_dir: [0.0; 4],
            last_op_p_iq_lr: [0.0; 4],
            last_op_p_feat: [0.0; 4],
            last_core_lr: [0.0; 2],
            last_transport_wet_lr: [0.0; 2],
            saturation_flags: 0,
            offline_bounce: false,
            declared_max_delay_48k: 4_096,
            ext_ring_pow2_frames: 0,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            active_kautz_blocks: 2,
            active_transport_quads: 1,
            hoisted_core: HoistedCoreChunk::empty(),
            hoisted_core_valid: false,
        }
    }

    /// Resets all wire/runtime metadata fields while preserving allocated cold buffers (`0` drops).
    pub(crate) fn reset_metadata_preserving_backing(&mut self) {
        self.sensory_cold = SensoryQuadCold::zero();
        self.kautz_slew = KautzColdSlewState::default();
        self.transport_quad_meta = [QuadColdMeta::default(); TRANSPORT_MAX_QUADS];
        self.transport_quad_eff = [QuadEffectiveTargets::default(); TRANSPORT_MAX_QUADS];
        self.bounce4x = Bounce4xOversampler::new();
        self.bounce4x_core_decim = Bounce4xDecimator::new();
        self.bounce4x_wet_decim = Bounce4xDecimator::new();
        self.wire_crossbar = CrossbarWire::new_zeroed();
        self.wire_kautz = [KautzSoaBlockWire::new_zeroed(); MAX_KAUTZ_BANKS];
        self.wire_transport_quads = [TransportQuadWire::new_zeroed(); TRANSPORT_MAX_QUADS];
        self.base_core_quads = [CoreQuadHot::default(); MAX_CORE_QUADS];
        self.base_core_ext = [CoreQuadExtCold::default(); MAX_CORE_QUADS];
        self.tier_fir_l = [[0.0; 128]; 4];
        self.tier_fir_r = [[0.0; 128]; 4];
        self.tier_kautz_b_gamma = [[0.0; MAX_KAUTZ_PAIRS]; 4];
        self.tier_kautz_c_l = [[(0.0, 0.0); MAX_KAUTZ_PAIRS]; 4];
        self.tier_kautz_c_r = [[(0.0, 0.0); MAX_KAUTZ_PAIRS]; 4];
        self.tier_crossbars = [TierCrossbarSnapshot::default(); 4];
        self.base_phasor_freq_hz = [0.0; 4];
        self.base_phasor_init_phase_q32 = [0; 4];
        self.base_op_p_gain = 0.0;
        self.base_op_t_gain = 0.0;
        self.base_tank_dry_mix = 0.0;
        self.ro_pole_48k = 0.0;
        self.base_op_n_pre = OpNPreBaseWire::new_zeroed();
        self.stage_latent_mods_fp16 = [0; 96];
        self.stage_latent_ctrl_ultra_fp16 = [0; 256];
        self.stage_latent_core_mods_fp16 = [0; 1_024];
        self.tier_os2x = [false; 4];
        self.tier_gain_trim = [1.0; 4];
        self.core_mods_present = false;
        self.core_mods_active = false;
        self.meta_flags = 0;
        self.max_present_tier = Tier::Lite;
        self.retained_tier = Tier::Lite;
        self.zoh = ZohRateContext::default();
        self.latched_stage_params = [0.0; 32];
        self.latched_z_stage = [0.0; 8];
        self.lane_g_scale = [1.0; MAX_TOTAL_SONA_LANES];
        self.lane_k_delta = [0.0; MAX_TOTAL_SONA_LANES];
        self.lane_wf_delta = [0.0; MAX_TOTAL_SONA_LANES];
        self.quad_adaa_scale = [1.0; MAX_CORE_QUADS];
        self.quad_lru_bias = [0.0; MAX_CORE_QUADS];
        self.op_n_pre_hoisted = OpNPreHoisted::bypass();
        self.rem_hist = HilbertH1RemapHistory::default();
        self.xfade_rem = 0;
        self.xfade_prev_out = [0.0; 2];
        self.xfade_prev_core = [0.0; 2];
        self.xfade_prev_wet = [0.0; 2];
        self.last_in = [0.0; 2];
        self.last_s2_dir = [0.0; 4];
        self.last_op_p_iq_lr = [0.0; 4];
        self.last_op_p_feat = [0.0; 4];
        self.last_core_lr = [0.0; 2];
        self.last_transport_wet_lr = [0.0; 2];
        self.saturation_flags = 0;
        self.offline_bounce = false;
        self.declared_max_delay_48k = 4_096;
        self.ext_ring_pow2_frames = 0;
        self.dc_blocker_pole_5hz = crate::math::TANK_DC_BLOCKER_48K_F32;
        self.active_kautz_blocks = 2;
        self.active_transport_quads = 1;
        self.hoisted_core = HoistedCoreChunk::empty();
        self.hoisted_core_valid = false;
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
    pub convolver_fir_ext: &'a mut [f32],
    pub transport_ext_fp16: [&'a mut [u16]; TRANSPORT_MAX_QUADS],
    pub transport_ext_f32: [&'a mut [f32]; TRANSPORT_MAX_QUADS],
    pub transport_schroeder: [&'a mut [f32]; TRANSPORT_MAX_QUADS],
    pub cold_tail: &'a mut [SonaColdTail<'a, B>],
}

impl<B: GraphBackend> fmt::Debug for SonaSlabBacking<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SonaSlabBacking")
            .field("convolver_fir_ext_len", &self.convolver_fir_ext.len())
            .field("cold_tail_len", &self.cold_tail.len())
            .finish_non_exhaustive()
    }
}

/// Prefix-ordered `64 B`-aligned `SONA v3.0` L1D hot execution slab (`4,544..14,592 B` active hot).
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
        let lite = core::mem::size_of::<SonaLiteHotPrefix<B>>();
        let os = core::mem::size_of::<SonaOs2xHot>();
        let full = core::mem::size_of::<SonaFullHotDelta<B>>();
        let pro = core::mem::size_of::<SonaProHotDelta<B>>();
        let ultra = core::mem::size_of::<SonaUltraHotDelta<B>>();
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
    pub(crate) fn state_mut(&mut self, q: usize) -> &mut SonaBlockStateStereo8<B::Stereo8> {
        match q {
            0..=3 => &mut self.lite.core_states[q],
            4..=5 => &mut self.full_delta.core_states_full[q - 4],
            6..=7 => &mut self.pro_delta.core_states_pro[q - 6],
            _ => &mut self.ultra_delta.core_states_ultra[q - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn core_quad_hot(&self, q: usize) -> &CoreQuadHot {
        match q {
            0..=3 => &self.lite.core_quads[q],
            4..=5 => &self.full_delta.core_quads_full[q - 4],
            6..=7 => &self.pro_delta.core_quads_pro[q - 6],
            _ => &self.ultra_delta.core_quads_ultra[q - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn core_quad_hot_mut(&mut self, q: usize) -> &mut CoreQuadHot {
        match q {
            0..=3 => &mut self.lite.core_quads[q],
            4..=5 => &mut self.full_delta.core_quads_full[q - 4],
            6..=7 => &mut self.pro_delta.core_quads_pro[q - 6],
            _ => &mut self.ultra_delta.core_quads_ultra[q - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn core_quad_ext(&self, q: usize) -> &CoreQuadExtCold {
        &self.cold().base_core_ext[q]
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
    pub(crate) fn readout_hot(&self, q: usize) -> &CoreQuadReadoutHot {
        match q {
            0..=3 => &self.lite.readouts[q],
            4..=5 => &self.full_delta.readouts_full[q - 4],
            6..=7 => &self.pro_delta.readouts_pro[q - 6],
            _ => &self.ultra_delta.readouts_ultra[q - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn readout_hot_mut(&mut self, q: usize) -> &mut CoreQuadReadoutHot {
        match q {
            0..=3 => &mut self.lite.readouts[q],
            4..=5 => &mut self.full_delta.readouts_full[q - 4],
            6..=7 => &mut self.pro_delta.readouts_pro[q - 6],
            _ => &mut self.ultra_delta.readouts_ultra[q - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn op_p_to_core_fp16(&self, q: usize) -> &[u16; 4] {
        match q {
            0..=3 => &self.lite.op_p_to_core_lite_fp16[q],
            4..=5 => &self.full_delta.op_p_to_core_full_fp16[q - 4],
            6..=7 => &self.pro_delta.op_p_to_core_pro_fp16[q - 6],
            _ => &self.ultra_delta.op_p_to_core_ultra_fp16[q - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn phasor_to_core_fp16(&self, q: usize) -> &[u16; 4] {
        match q {
            0..=3 => &self.lite.phasor_to_core_lite_fp16[q],
            4..=5 => &self.full_delta.phasor_to_core_full_fp16[q - 4],
            6..=7 => &self.pro_delta.phasor_to_core_pro_fp16[q - 6],
            _ => &self.ultra_delta.phasor_to_core_ultra_fp16[q - 8],
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
    pub(crate) fn rect_mask4(&self, q: usize) -> u8 {
        match q {
            0..=3 => ((self.lite.rect_sign_mask_lite >> (q * 4)) & 0x0F) as u8,
            4..=5 => (self.full_delta.rect_sign_mask_full >> ((q - 4) * 4)) & 0x0F,
            6..=7 => (self.pro_delta.rect_sign_mask_pro >> ((q - 6) * 4)) & 0x0F,
            _ => (self.ultra_delta.rect_sign_mask_ultra >> ((q - 8) * 4)) & 0x0F,
        }
    }

    #[inline(always)]
    pub(crate) fn transport_quad_hot(&self, q: usize) -> &TransportQuadHotCompact {
        match q {
            0 => &self.lite.transport_quad0,
            1 => &self.full_delta.transport_quad1,
            2 => &self.pro_delta.transport_quad2,
            _ => &self.ultra_delta.transport_quad3,
        }
    }

    #[inline(always)]
    pub(crate) fn transport_quad_hot_mut(&mut self, q: usize) -> &mut TransportQuadHotCompact {
        match q {
            0 => &mut self.lite.transport_quad0,
            1 => &mut self.full_delta.transport_quad1,
            2 => &mut self.pro_delta.transport_quad2,
            _ => &mut self.ultra_delta.transport_quad3,
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
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, os2x_hot) == CUM_OS2X_OFFSET);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, full_delta) == CUM_FULL_OFFSET);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, pro_delta) == CUM_PRO_OFFSET);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, ultra_delta) == CUM_ULTRA_OFFSET);
const _: () = assert!(offset_of!(SonaHotSlab<'_, ScalarBackend>, cold) == CUM_COLD_OFFSET);

const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, lite) == 0);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, os2x_hot) == CUM_OS2X_OFFSET);
const _: () =
    assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, full_delta) == CUM_FULL_OFFSET);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, pro_delta) == CUM_PRO_OFFSET);
const _: () =
    assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, ultra_delta) == CUM_ULTRA_OFFSET);
const _: () = assert!(offset_of!(SonaHotSlab<'_, DefaultMathBackend>, cold) == CUM_COLD_OFFSET);

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
            11_136
        );
        assert_eq!(
            SonaHotSlab::<ScalarBackend>::active_hot_bytes(Tier::Ultra, true),
            14_592
        );
    }
}
