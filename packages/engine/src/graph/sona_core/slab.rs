//! Cache-Coherent `64 B`-Aligned [`SonaHotSlab`] & Lifecycle (`graph/sona_core/slab.rs`,
//! `#![forbid(unsafe_code)]`).
//!
//! Packs [`SonaLiteHotPrefix`] (`4,608 B` = `72 CL`) at byte `0`, followed strictly by
//! `os2x_hot`, `full_delta`, `pro_delta`, `ultra_delta`, and boxed [`SonaColdTail`], guaranteeing
//! `ActiveHot <= 4,608 / 7,936 / 15,872 / 28,672 B` (`<= 30,720 B` L1D ceiling).

#![forbid(unsafe_code)]

pub mod ctrl_apply;
pub mod layout;
pub mod lifecycle;

use alloc::boxed::Box;
use core::marker::PhantomData;

use dnam_kernels::{
    DC_BLOCKER_5HZ_48K_F32, HILBERT_ALPHA_I, HILBERT_ALPHA_Q, HilbertH1Kernel, PhasorQ32Bank4,
    SonaBlockStateStereo8, StereoVec8, T1DcBlockerKernel, T1SubHarmonicKernel, f32_to_fp16,
};

use self::ctrl_apply::{sync_stage3_with_optional_zoh, sync_stage4_params};
pub use self::layout::{
    CrossbarFrame, FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_FREEZE, FLAG_MONO_OUT, FLAG_OS2X,
    FLAG_XFADE_ACTIVE, SonaColdTail, SonaControlBusState, SonaFullHotDelta, SonaGroupHot,
    SonaGroupReadoutHot, SonaGroupReadoutRegs, SonaInterBlockTileHot, SonaLiteHotPrefix,
    SonaOs2xHot, SonaProHotDelta, SonaUltraHotDelta, TIER_SWITCH_XFADE_SAMPLES,
    sona_slab_active_hot_bytes,
};
use crate::container::ZohRescaledParams;
use crate::graph::GraphBackend;
use crate::graph::cabinet::{CabinetHighSrFirExt, KautzColdSlewState, KautzStereoBank4};
use crate::graph::router::{BezoutCleanCrossover, CrossbarParams};
use crate::graph::sona_core::frontend::SonaStereoOversampler2x;
use crate::graph::sona_core::model::{
    MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_TANK_LINES, MAX_TANK_QUADS, SonaColdTierModel,
    SonaCrossbarConfig, SonaTierReadout,
};
use crate::graph::tank::fdn::{TankQuadHotCompact, V3_TANK_CAP};
use crate::math::{DefaultMathBackend, algebraic_unit_rotor, newton6_reciprocal_f64};
use crate::types::Tier;

fn compute_crossbar_active(cb: &SonaCrossbarConfig) -> bool {
    cb.op_p_crossbar_gain.abs() > 1.0e-7
        || cb.op_t_crossbar_gain.abs() > 1.0e-7
        || cb.phasor_freq_hz.iter().any(|&f| f.abs() > 1.0e-7)
        || cb.op_p_to_group.iter().flatten().any(|&v| v.abs() > 1.0e-7)
        || cb
            .phasor_to_group
            .iter()
            .flatten()
            .any(|&v| v.abs() > 1.0e-7)
        || cb.opt_to_lr.iter().flatten().any(|&v| v.abs() > 1.0e-7)
}

fn fill_ro_hot(ro: &SonaTierReadout, g: usize) -> SonaGroupReadoutHot {
    let base = g << 2;
    let copy4 = |src: &[f32]| -> [u16; 4] {
        [
            f32_to_fp16(src[base]),
            f32_to_fp16(src[base + 1]),
            f32_to_fp16(src[base + 2]),
            f32_to_fp16(src[base + 3]),
        ]
    };
    SonaGroupReadoutHot {
        pre_cab_odd_l_fp16: copy4(&ro.pre_cab_w_odd[0]),
        pre_cab_odd_r_fp16: copy4(&ro.pre_cab_w_odd[1]),
        pre_cab_even_l_fp16: copy4(&ro.pre_cab_w_even[0]),
        pre_cab_even_r_fp16: copy4(&ro.pre_cab_w_even[1]),
        pre_cab_lru_l_fp16: copy4(&ro.pre_cab_w_lru[0]),
        pre_cab_lru_r_fp16: copy4(&ro.pre_cab_w_lru[1]),
        direct_odd_l_fp16: copy4(&ro.direct_w_odd[0]),
        direct_odd_r_fp16: copy4(&ro.direct_w_odd[1]),
        direct_even_l_fp16: copy4(&ro.direct_w_even[0]),
        direct_even_r_fp16: copy4(&ro.direct_w_even[1]),
        direct_lru_l_fp16: copy4(&ro.direct_w_lru[0]),
        direct_lru_r_fp16: copy4(&ro.direct_w_lru[1]),
    }
}

fn compute_xo_pole(xo_hz: f32, zoh: Option<&ZohRescaledParams>) -> f32 {
    let mut xo = BezoutCleanCrossover::new();
    xo.set_crossover_hz(xo_hz);
    if let Some(z) = zoh {
        xo.reconfigure_sample_rate(z);
    }
    xo.pole
}

fn rescale_op_p_front(
    t1_dc: &mut T1DcBlockerKernel,
    hilbert: &mut HilbertH1Kernel,
    t1: &mut T1SubHarmonicKernel,
    phasors: &mut PhasorQ32Bank4,
    freqs_hz: &[f32; 4],
    zoh: Option<&ZohRescaledParams>,
) {
    match zoh {
        Some(z) if (z.rho_sr - 1.0).abs() >= 1.0e-12 => {
            *t1_dc = T1DcBlockerKernel::with_pole(z.rescale_real_pole(DC_BLOCKER_5HZ_48K_F32));
            let ai = HILBERT_ALPHA_I.map(|a| z.rescale_real_pole(a));
            let aq = HILBERT_ALPHA_Q.map(|a| z.rescale_real_pole(a));
            *hilbert = HilbertH1Kernel::with_coeffs(ai, aq);
            let inv_sr = (newton6_reciprocal_f64(z.effective_sample_rate_hz().max(8_000.0))) as f32;
            let prev_phase = phasors.phase;
            *phasors = PhasorQ32Bank4::from_freqs_hz(*freqs_hz, inv_sr);
            phasors.phase = prev_phase;
        }
        _ => {
            *t1_dc = T1DcBlockerKernel::new();
            *hilbert = HilbertH1Kernel::new();
            let prev_phase = phasors.phase;
            *phasors = PhasorQ32Bank4::from_freqs_hz(*freqs_hz, 1.0 / 48_000.0);
            phasors.phase = prev_phase;
        }
    }
    let _ = t1;
}

/// Unified, cache-coherent, `64 B`-aligned `SONA` `v3.0` real-time execution slab (`Lite..Ultra`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaHotSlab<B: GraphBackend = DefaultMathBackend> {
    pub lite: SonaLiteHotPrefix<B>,
    pub os2x_hot: SonaOs2xHot,
    pub full_delta: SonaFullHotDelta<B>,
    pub pro_delta: SonaProHotDelta<B>,
    pub ultra_delta: SonaUltraHotDelta<B>,
    pub cold: Box<SonaColdTail>,
    _backend: PhantomData<B>,
}

impl<B: GraphBackend> SonaHotSlab<B> {
    /// Returns active L1D working-set bytes `[0..ActiveHotBytes)` per `(tier, os2x)`.
    #[inline]
    #[must_use]
    pub const fn active_hot_bytes(tier: Tier, os2x: bool) -> usize {
        sona_slab_active_hot_bytes(tier, os2x)
    }

    #[inline(always)]
    #[must_use]
    pub const fn tier(&self) -> Tier {
        self.lite.tier
    }

    #[inline(always)]
    #[must_use]
    pub const fn os2x(&self) -> bool {
        (self.lite.flags & FLAG_OS2X) != 0
    }

    /// Enables or disables internal `2x` oversampling cleanly (e.g. disabled under external `4x` bounce).
    #[inline]
    pub fn set_os2x(&mut self, enabled: bool) {
        if enabled {
            self.lite.flags |= FLAG_OS2X;
        } else {
            self.lite.flags &= !FLAG_OS2X;
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn mono_out(&self) -> bool {
        (self.lite.flags & FLAG_MONO_OUT) != 0
    }

    #[inline]
    pub fn set_mono_out(&mut self, enabled: bool) {
        if enabled {
            self.lite.flags |= FLAG_MONO_OUT;
        } else {
            self.lite.flags &= !FLAG_MONO_OUT;
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn crossbar_active(&self) -> bool {
        (self.lite.flags & FLAG_CROSSBAR_ACTIVE) != 0
    }

    #[inline(always)]
    #[must_use]
    pub const fn sample_tick(&self) -> u32 {
        self.lite.sample_tick
    }

    #[inline(always)]
    #[must_use]
    pub const fn tank_freeze(&self) -> bool {
        (self.lite.flags & FLAG_FREEZE) != 0
    }

    #[inline]
    pub fn set_tank_freeze(&mut self, freeze: bool) {
        if freeze {
            self.lite.flags |= FLAG_FREEZE;
        } else {
            self.lite.flags &= !FLAG_FREEZE;
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn tank_g_loop(&self) -> f32 {
        self.lite.quad0.g_loop
    }

    /// Returns peak active feedback loop gain `g_loop` across all active tank quads of the current tier.
    #[inline]
    #[must_use]
    pub fn max_active_loop_gain(&self) -> f32 {
        let active_quads = crate::graph::sona_core::model::tier_tank_quads(self.lite.tier);
        let mut peak = 0.0_f32;
        for q in 0..active_quads {
            let g = self.quad_hot(q).g_loop;
            if g.is_finite() {
                peak = peak.max(g);
            }
        }
        peak
    }

    #[inline(always)]
    #[must_use]
    pub fn tank_dry_mix(&self) -> f32 {
        crate::math::fp16::fp16_to_f32(self.lite.quad0.nonrec_fp16[1])
    }

    #[inline(always)]
    #[must_use]
    pub fn tank_effective_cap(&self) -> usize {
        self.cold.stage4_ext[0]
            .as_ref()
            .map_or(V3_TANK_CAP, |e| e.cap)
    }

    #[inline(always)]
    #[must_use]
    pub fn last_s2_dir(&self) -> [f32; 4] {
        self.cold.last_s2_dir
    }

    /// Updates `Stage 0` `s_router` (`S_{ret->send} == 0` enforced unconditionally).
    pub fn set_s0_s_router(&mut self, mut s_router: [[f32; 4]; 4]) {
        s_router[0][2] = 0.0;
        s_router[0][3] = 0.0;
        s_router[1][2] = 0.0;
        s_router[1][3] = 0.0;
        self.cold.base_stage0.s_router = s_router;
        self.cold.base_stage0.clamp_in_place();
    }

    #[inline(always)]
    pub(crate) fn state_mut(&mut self, g: usize) -> &mut SonaBlockStateStereo8<B::Stereo8> {
        match g {
            0..=3 => &mut self.lite.sona_states[g],
            4..=5 => &mut self.full_delta.sona_states_full[g - 4],
            6..=7 => &mut self.pro_delta.sona_states_pro[g - 6],
            _ => &mut self.ultra_delta.sona_states_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn group_hot(&self, g: usize) -> &SonaGroupHot {
        match g {
            0..=3 => &self.lite.sona_groups[g],
            4..=5 => &self.full_delta.sona_groups_full[g - 4],
            6..=7 => &self.pro_delta.sona_groups_pro[g - 6],
            _ => &self.ultra_delta.sona_groups_ultra[g - 8],
        }
    }

    #[inline(always)]
    pub(crate) fn set_group_hot(&mut self, g: usize, hot: SonaGroupHot, mask4: u8) {
        let m4 = mask4 & 0x0F;
        match g {
            0..=3 => {
                self.lite.sona_groups[g] = hot;
                let shift = g << 2;
                self.lite.rect_sign_mask_lite =
                    (self.lite.rect_sign_mask_lite & !(0x0F << shift)) | (u16::from(m4) << shift);
            }
            4..=5 => {
                self.full_delta.sona_groups_full[g - 4] = hot;
                let shift = (g - 4) << 2;
                self.full_delta.rect_sign_mask_full =
                    (self.full_delta.rect_sign_mask_full & !(0x0F << shift)) | (m4 << shift);
            }
            6..=7 => {
                self.pro_delta.sona_groups_pro[g - 6] = hot;
                let shift = (g - 6) << 2;
                self.pro_delta.rect_sign_mask_pro =
                    (self.pro_delta.rect_sign_mask_pro & !(0x0F << shift)) | (m4 << shift);
            }
            _ => {
                self.ultra_delta.sona_groups_ultra[g - 8] = hot;
                let shift = (g - 8) << 2;
                self.ultra_delta.rect_sign_mask_ultra =
                    (self.ultra_delta.rect_sign_mask_ultra & !(0x0F << shift)) | (m4 << shift);
            }
        }
    }

    #[inline(always)]
    pub(crate) fn rect_mask4(&self, g: usize) -> u8 {
        match g {
            0..=3 => ((self.lite.rect_sign_mask_lite >> (g << 2)) & 0x0F) as u8,
            4..=5 => (self.full_delta.rect_sign_mask_full >> ((g - 4) << 2)) & 0x0F,
            6..=7 => (self.pro_delta.rect_sign_mask_pro >> ((g - 6) << 2)) & 0x0F,
            _ => (self.ultra_delta.rect_sign_mask_ultra >> ((g - 8) << 2)) & 0x0F,
        }
    }

    #[inline(always)]
    pub(crate) fn tile_hot(&self, ti: usize) -> &SonaInterBlockTileHot {
        match ti {
            0..=2 => &self.lite.sona_tiles[ti],
            3..=12 => &self.full_delta.sona_tiles_full[ti - 3],
            13..=22 => &self.pro_delta.sona_tiles_pro[ti - 13],
            _ => &self.ultra_delta.sona_tiles_ultra[ti - 23],
        }
    }

    #[inline(always)]
    pub(crate) fn ro_group_hot(&self, g: usize) -> &SonaGroupReadoutHot {
        match g {
            0..=3 => &self.lite.ro_groups[g],
            4..=5 => &self.full_delta.ro_groups_full[g - 4],
            6..=7 => &self.pro_delta.ro_groups_pro[g - 6],
            _ => &self.ultra_delta.ro_groups_ultra[g - 8],
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
    pub(crate) fn opt_to_lr_fp16(&self, line: usize) -> [u16; 2] {
        match line {
            0..=3 => self.lite.opt_to_lr_lite_fp16[line],
            4..=7 => self.full_delta.opt_to_lr_full_fp16[line - 4],
            8..=11 => self.pro_delta.opt_to_lr_pro_fp16[line - 8],
            _ => self.ultra_delta.opt_to_lr_ultra_fp16[line - 12],
        }
    }

    #[inline(always)]
    pub(crate) fn quad_hot(&self, q: usize) -> &TankQuadHotCompact {
        match q {
            0 => &self.lite.quad0,
            1 => &self.full_delta.quad1,
            2 => &self.pro_delta.quad2,
            _ => &self.ultra_delta.quad3,
        }
    }

    #[inline(always)]
    pub(crate) fn quad_hot_mut(&mut self, q: usize) -> &mut TankQuadHotCompact {
        match q {
            0 => &mut self.lite.quad0,
            1 => &mut self.full_delta.quad1,
            2 => &mut self.pro_delta.quad2,
            _ => &mut self.ultra_delta.quad3,
        }
    }

    /// Initializes a cache-coherent [`SonaHotSlab`] from a cold tier model (`0` RT allocs).
    #[must_use]
    pub fn from_cold_model(model: &SonaColdTierModel) -> Self {
        Self::from_cold_model_with_stage4_cap(model, 0)
    }

    /// Initializes [`SonaHotSlab`] with explicit cold Stage IV headroom (`stage4_min_cap`).
    #[must_use]
    pub fn from_cold_model_with_stage4_cap(
        model: &SonaColdTierModel,
        stage4_min_cap: usize,
    ) -> Self {
        crate::math::warm_simd_feature_caches();
        let zero_st = SonaBlockStateStereo8 {
            svf_s1: B::Stereo8::zero(),
            svf_s2: B::Stereo8::zero(),
            lru_v: B::Stereo8::zero(),
            adaa_prev: B::Stereo8::zero(),
        };
        let mut cb = model.crossbar;
        cb.clamp_in_place();
        let cb_active = compute_crossbar_active(&cb);
        let mut s0 = model.stage0;
        s0.clamp_in_place();
        let mut os_p = model.os_params;
        os_p.clamp_in_place();
        let mut s4 = model.stage4;
        s4.clamp_in_place();
        let mut ctrl_box = Box::new(model.ctrl);
        ctrl_box.clamp_in_place();

        let mut flags = 0_u8;
        if model.os2x {
            flags |= FLAG_OS2X;
        }
        if model.mono_out {
            flags |= FLAG_MONO_OUT;
        }
        if cb_active {
            flags |= FLAG_CROSSBAR_ACTIVE;
        }

        let q_init =
            |q: usize| TankQuadHotCompact::from_params(&s4.effective_quad_params(q), 48_000.0);
        let phasors = PhasorQ32Bank4::new(
            cb.phasor_init_phase_q32,
            PhasorQ32Bank4::from_freqs_hz(cb.phasor_freq_hz, 1.0 / 48_000.0).inc,
        );

        let mut slab = Self {
            lite: SonaLiteHotPrefix {
                sona_states: [zero_st; 4],
                sona_groups: [SonaGroupHot::default(); 4],
                sona_tiles: [SonaInterBlockTileHot::default(); 3],
                ro_groups: [SonaGroupReadoutHot::default(); 4],
                fir_taps_l: [[[0.0; 4]; 4]; 2],
                fir_taps_r: [[[0.0; 4]; 4]; 2],
                fir_hist_l: [0.0; 64],
                fir_hist_r: [0.0; 64],
                kautz: [KautzStereoBank4::new(); 2],
                quad0_frames: [[0; MAX_TANK_LINES / 4]; V3_TANK_CAP],
                quad0: q_init(0),
                hilbert: HilbertH1Kernel::new(),
                t1: T1SubHarmonicKernel::new(),
                phasors,
                t1_dc: T1DcBlockerKernel::new(),
                stage0_xo_x0: [0.0; 2],
                stage0_xo_y_l: [0.0; 4],
                stage0_xo_y_r: [0.0; 4],
                stage0_xo_pole: compute_xo_pole(s0.xo_hz, None),
                stage0_gate_env: 0.0,
                stage0_gate_gain: 1.0,
                stage0_hp_lp_pole: 0.95,
                stage0_hp_lp_state: [0.0; 2],
                ro_gains: [0.0; 8],
                sample_tick: 0,
                op_p_to_group_lite_fp16: [[0; 4]; 4],
                phasor_to_group_lite_fp16: [[0; 4]; 4],
                opt_to_lr_lite_fp16: [[0; 2]; 4],
                phasor_st_cos_fp16: [f32_to_fp16(1.0); 4],
                phasor_st_sin_fp16: [0; 4],
                stage0_gain_mix_fp16: [0; 6],
                stage0_xo_fp16: [0; 3],
                stage0_gate_fp16: [0; 6],
                op_gains_fp16: [0; 2],
                fir_write_pos: 0,
                fir_onset_samples: 16,
                rect_sign_mask_lite: 0,
                tier: model.tier,
                flags,
            },
            os2x_hot: SonaOs2xHot {
                os_params: os_p,
                os_stereo: SonaStereoOversampler2x::default(),
            },
            full_delta: SonaFullHotDelta {
                sona_states_full: [zero_st; 2],
                sona_groups_full: [SonaGroupHot::default(); 2],
                sona_tiles_full: [SonaInterBlockTileHot::default(); 10],
                ro_groups_full: [SonaGroupReadoutHot::default(); 2],
                fir_taps_l_full: [[[0.0; 4]; 4]; 1],
                fir_taps_r_full: [[[0.0; 4]; 4]; 1],
                kautz_full: [KautzStereoBank4::new(); 1],
                quad1: q_init(1),
                op_p_to_group_full_fp16: [[0; 4]; 2],
                phasor_to_group_full_fp16: [[0; 4]; 2],
                opt_to_lr_full_fp16: [[0; 2]; 4],
                rect_sign_mask_full: 0,
            },
            pro_delta: SonaProHotDelta {
                sona_states_pro: [zero_st; 2],
                sona_groups_pro: [SonaGroupHot::default(); 2],
                sona_tiles_pro: [SonaInterBlockTileHot::default(); 10],
                ro_groups_pro: [SonaGroupReadoutHot::default(); 2],
                fir_taps_l_pro: [[[0.0; 4]; 4]; 3],
                fir_taps_r_pro: [[[0.0; 4]; 4]; 3],
                fir_hist_l_pro: [0.0; 256],
                fir_hist_r_pro: [0.0; 256],
                kautz_pro: [KautzStereoBank4::new(); 3],
                quad2: q_init(2),
                op_p_to_group_pro_fp16: [[0; 4]; 2],
                phasor_to_group_pro_fp16: [[0; 4]; 2],
                opt_to_lr_pro_fp16: [[0; 2]; 4],
                rect_sign_mask_pro: 0,
            },
            ultra_delta: SonaUltraHotDelta {
                sona_states_ultra: [zero_st; 2],
                sona_groups_ultra: [SonaGroupHot::default(); 2],
                sona_tiles_ultra: [SonaInterBlockTileHot::default(); 12],
                ro_groups_ultra: [SonaGroupReadoutHot::default(); 2],
                fir_taps_l_ultra: [[[0.0; 4]; 4]; 2],
                fir_taps_r_ultra: [[[0.0; 4]; 4]; 2],
                kautz_ultra: [KautzStereoBank4::new(); 2],
                quad3: q_init(3),
                op_p_to_group_ultra_fp16: [[0; 4]; 2],
                phasor_to_group_ultra_fp16: [[0; 4]; 2],
                opt_to_lr_ultra_fp16: [[0; 2]; 4],
                rect_sign_mask_ultra: 0,
            },
            cold: Box::new(SonaColdTail {
                base_stage0: s0,
                base_groups: model.groups,
                base_os_params: os_p,
                base_fir_l: model.readout.fir_l,
                base_fir_r: model.readout.fir_r,
                base_kautz_poles: model.kautz_poles,
                base_kautz_b_gamma: model.readout.kautz_b_gamma,
                base_tank_g_loop: core::array::from_fn(|q| s4.effective_quad_params(q).g_loop),
                base_tank_wet_mix: core::array::from_fn(|q| s4.effective_quad_params(q).wet_mix),
                base_op_p_gain: cb.op_p_crossbar_gain,
                base_op_t_gain: cb.op_t_crossbar_gain,
                base_active_quads: s4.active_quads.clamp(1, MAX_TANK_QUADS),
                kautz_slew: KautzColdSlewState::new(),
                stage3_fir_ext: CabinetHighSrFirExt::new_preallocated(),
                stage4_ext: [None, None, None, None],
                stage4_ext_active: [false; MAX_TANK_QUADS],
                stage4_schroeder: [None, None, None, None],
                ctrl_params: ctrl_box,
                ctrl_state: SonaControlBusState::identity(),
                crossbar_freqs_hz: cb.phasor_freq_hz,
                crossbar_init_phase_q32: cb.phasor_init_phase_q32,
                crossbar_stereo_phase_rad: cb.phasor_stereo_phase_rad,
                last_in: [0.0; 2],
                last_s2_dir: [0.0; 4],
                last_stage_targets: [0.0; 16],
                last_z_stage: [0.0; 8],
                xfade_rem: 0,
                xfade_total: TIER_SWITCH_XFADE_SAMPLES,
                xfade_prev_out: [0.0; 2],
                zoh: None,
                os_factor: if model.os2x { 2 } else { 1 },
                is_bounce_4x: false,
                base_crossbar_active: cb_active,
                max_op_n_drive: 0.0,
                max_pole_radius: 0.0,
            }),
            _backend: PhantomData,
        };
        slab.apply_weights_common(model, None);
        slab.ensure_cold_tank_cap(&s4, stage4_min_cap.max(2_048));
        sync_stage3_with_optional_zoh(
            &mut slab,
            &model.readout,
            &model.kautz_poles,
            1,
            false,
            None,
        );
        sync_stage4_params(&mut slab, &s4, None, true);
        slab
    }

    fn write_stage0_scalars(&mut self, s0: &CrossbarParams) {
        self.lite.stage0_gain_mix_fp16 = [
            f32_to_fp16(s0.gain[0]),
            f32_to_fp16(s0.gain[1]),
            f32_to_fp16(s0.mix_matrix[0][0]),
            f32_to_fp16(s0.mix_matrix[0][1]),
            f32_to_fp16(s0.mix_matrix[1][0]),
            f32_to_fp16(s0.mix_matrix[1][1]),
        ];
        self.lite.stage0_xo_fp16 = [
            f32_to_fp16(s0.xo_low_gain),
            f32_to_fp16(s0.xo_high_gain),
            f32_to_fp16(s0.clean_sub_mix),
        ];
        self.lite.stage0_gate_fp16 = [
            f32_to_fp16(s0.gate_thresh),
            f32_to_fp16(s0.gate_hyst),
            f32_to_fp16(s0.gate_inv_knee),
            f32_to_fp16(s0.gate_floor),
            f32_to_fp16(s0.gate_atk_beta),
            f32_to_fp16(s0.gate_rel_beta),
        ];
    }

    pub(super) fn apply_weights_common(
        &mut self,
        model: &SonaColdTierModel,
        zoh: Option<&ZohRescaledParams>,
    ) {
        self.lite.tier = model.tier;
        self.set_os2x(model.os2x);
        self.set_mono_out(model.mono_out);

        let mut s0 = model.stage0;
        if let Some(z) = zoh {
            s0.rescale_with_zoh(z);
        } else {
            s0.clamp_in_place();
        }
        self.cold.base_stage0 = model.stage0;
        self.cold.base_stage0.clamp_in_place();
        self.lite.stage0_xo_pole = compute_xo_pole(s0.xo_hz, zoh);
        self.write_stage0_scalars(&s0);

        let mut osp = model.os_params;
        if let Some(z) = zoh {
            osp.rescale_with_zoh(z);
        } else {
            osp.clamp_in_place();
        }
        self.cold.base_os_params = model.os_params;
        self.cold.base_os_params.clamp_in_place();
        self.os2x_hot.os_params = osp;

        let rho_eff = match zoh {
            Some(z) if model.os2x && !self.cold.is_bounce_4x => z.rho_sr * 0.5,
            Some(z) => z.rho_sr,
            None => {
                if model.os2x && !self.cold.is_bounce_4x {
                    0.5
                } else {
                    1.0
                }
            }
        };
        self.lite.stage0_hp_lp_pole =
            crate::schema::cold_pow_f64(0.95, rho_eff).clamp(1.0e-4, 0.999_995) as f32;
        for g in 0..MAX_SONA_GROUPS {
            self.cold.base_groups[g] = model.groups[g];
            let mut gc = model.groups[g];
            if (rho_eff - 1.0).abs() > 1.0e-7 {
                gc.rescale_with_zoh_ratio(rho_eff);
            }
            let (hot, mask4) = SonaGroupHot::from_cold_with_mask(&gc);
            self.set_group_hot(g, hot, mask4);
        }
        for t in 0..MAX_SONA_TILES {
            let th = SonaInterBlockTileHot::from_cold(&model.tiles[t]);
            match t {
                0..=2 => self.lite.sona_tiles[t] = th,
                3..=12 => self.full_delta.sona_tiles_full[t - 3] = th,
                13..=22 => self.pro_delta.sona_tiles_pro[t - 13] = th,
                _ => self.ultra_delta.sona_tiles_ultra[t - 23] = th,
            }
        }
        for g in 0..MAX_SONA_GROUPS {
            let rh = fill_ro_hot(&model.readout, g);
            match g {
                0..=3 => self.lite.ro_groups[g] = rh,
                4..=5 => self.full_delta.ro_groups_full[g - 4] = rh,
                6..=7 => self.pro_delta.ro_groups_pro[g - 6] = rh,
                _ => self.ultra_delta.ro_groups_ultra[g - 8] = rh,
            }
        }
        self.lite.ro_gains = [
            model.readout.pre_cab_u_gain[0],
            model.readout.pre_cab_u_gain[1],
            model.readout.pre_cab_hp_gain[0],
            model.readout.pre_cab_hp_gain[1],
            model.readout.direct_s2_u_gain[0],
            model.readout.direct_s2_u_gain[1],
            model.readout.direct_s2_hp_gain[0],
            model.readout.direct_s2_hp_gain[1],
        ];

        let mut cb = model.crossbar;
        cb.clamp_in_place();
        let cb_active = compute_crossbar_active(&cb);
        self.cold.base_crossbar_active = cb_active;
        if cb_active {
            self.lite.flags |= FLAG_CROSSBAR_ACTIVE;
        } else {
            self.lite.flags &= !FLAG_CROSSBAR_ACTIVE;
        }
        self.cold.base_op_p_gain = cb.op_p_crossbar_gain;
        self.cold.base_op_t_gain = cb.op_t_crossbar_gain;
        self.cold.crossbar_freqs_hz = cb.phasor_freq_hz;
        self.cold.crossbar_init_phase_q32 = cb.phasor_init_phase_q32;
        self.cold.crossbar_stereo_phase_rad = cb.phasor_stereo_phase_rad;
        self.lite.op_gains_fp16 = [
            f32_to_fp16(cb.op_p_crossbar_gain),
            f32_to_fp16(cb.op_t_crossbar_gain),
        ];
        for lane in 0..4 {
            let (c, s) = algebraic_unit_rotor(cb.phasor_stereo_phase_rad[lane]);
            self.lite.phasor_st_cos_fp16[lane] = f32_to_fp16(c);
            self.lite.phasor_st_sin_fp16[lane] = f32_to_fp16(s);
        }
        for g in 0..MAX_SONA_GROUPS {
            let op_fp16 = cb.op_p_to_group[g].map(f32_to_fp16);
            let ph_fp16 = cb.phasor_to_group[g].map(f32_to_fp16);
            match g {
                0..=3 => {
                    self.lite.op_p_to_group_lite_fp16[g] = op_fp16;
                    self.lite.phasor_to_group_lite_fp16[g] = ph_fp16;
                }
                4..=5 => {
                    self.full_delta.op_p_to_group_full_fp16[g - 4] = op_fp16;
                    self.full_delta.phasor_to_group_full_fp16[g - 4] = ph_fp16;
                }
                6..=7 => {
                    self.pro_delta.op_p_to_group_pro_fp16[g - 6] = op_fp16;
                    self.pro_delta.phasor_to_group_pro_fp16[g - 6] = ph_fp16;
                }
                _ => {
                    self.ultra_delta.op_p_to_group_ultra_fp16[g - 8] = op_fp16;
                    self.ultra_delta.phasor_to_group_ultra_fp16[g - 8] = ph_fp16;
                }
            }
        }
        for line in 0..MAX_TANK_LINES {
            let lr = [
                f32_to_fp16(cb.opt_to_lr[line][0]),
                f32_to_fp16(cb.opt_to_lr[line][1]),
            ];
            match line {
                0..=3 => self.lite.opt_to_lr_lite_fp16[line] = lr,
                4..=7 => self.full_delta.opt_to_lr_full_fp16[line - 4] = lr,
                8..=11 => self.pro_delta.opt_to_lr_pro_fp16[line - 8] = lr,
                _ => self.ultra_delta.opt_to_lr_ultra_fp16[line - 12] = lr,
            }
        }
        *self.cold.ctrl_params = model.ctrl;
        self.cold.ctrl_params.clamp_in_place();
        rescale_op_p_front(
            &mut self.lite.t1_dc,
            &mut self.lite.hilbert,
            &mut self.lite.t1,
            &mut self.lite.phasors,
            &cb.phasor_freq_hz,
            zoh,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage0_hp_lp_pole_uses_rho_eff_zoh() {
        let zoh48 = ZohRescaledParams::new(48_000, 48_000);
        let zoh96 = ZohRescaledParams::new(48_000, 96_000);

        let cold_1x = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut slab_1x = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold_1x);
        slab_1x.reconfigure_with_zoh(&cold_1x, &zoh48);
        assert!((slab_1x.lite.stage0_hp_lp_pole - 0.95).abs() < 1.0e-6);

        let cold_2x = SonaColdTierModel::identity_for_tier(Tier::Full, true, false);
        let mut slab_2x = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold_2x);
        let expected_2x_48k = crate::schema::cold_pow_f64(0.95, 0.5) as f32;
        assert!((slab_2x.lite.stage0_hp_lp_pole - expected_2x_48k).abs() < 1.0e-6);
        slab_2x.reconfigure_with_zoh(&cold_2x, &zoh48);
        assert!((slab_2x.lite.stage0_hp_lp_pole - expected_2x_48k).abs() < 1.0e-6);

        slab_2x.reconfigure_with_zoh(&cold_2x, &zoh96);
        let expected_2x_96k = crate::schema::cold_pow_f64(0.95, 0.25) as f32;
        assert!((slab_2x.lite.stage0_hp_lp_pole - expected_2x_96k).abs() < 1.0e-6);
    }

    #[test]
    fn test_live_slab_pole_radius_and_multi_quad_loop_gain() {
        let mut cold = SonaColdTierModel::identity_for_tier(Tier::Pro, false, false);
        cold.groups[0].lru_alpha_rel[1] = 0.91;
        cold.kautz_poles[16] = (0.97, 0.45);
        cold.stage4.quads[0].g_loop = 0.35;
        cold.stage4.g_loop = 0.35;
        cold.stage4.quads[1].g_loop = 0.62;
        cold.stage4.quads[2].g_loop = 0.89;
        cold.stage4.quads[3].g_loop = 0.95;

        let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        assert!((slab.cold.max_pole_radius - 0.97).abs() < 1.0e-5);
        assert!((slab.max_active_loop_gain() - 0.89).abs() < 1.0e-6);

        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        slab.reconfigure_with_zoh(&cold, &zoh96);
        let expected_zoh_r = zoh96.rescale_real_pole(0.97);
        assert!((slab.cold.max_pole_radius - expected_zoh_r).abs() < 1.0e-4);
    }
}
