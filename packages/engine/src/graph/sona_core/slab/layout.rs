//! Prefix-Nested L1D [`SonaLiteHotPrefix`]..[`SonaUltraHotDelta`] Layouts & Cold Tail
//! (`graph/sona_core/slab/layout.rs`, `#![forbid(unsafe_code)]`).
//!
//! Enforces `ActiveHot <= 4,608 / 7,936 / 15,872 / 30,720 B` (`docs/specs/architecture/sona_architecture.md`
//! §2.3) by packing all `Lite` audio state and `fp16` weights strictly inside `[0..4,608 B)` ahead
//! of `os2x_hot`, higher-tier expansion deltas, and boxed [`SonaColdTail`].

#![forbid(unsafe_code)]

use alloc::boxed::Box;
use core::mem::{offset_of, size_of};

use dnam_kernels::{
    HilbertH1Kernel, PhasorQ32Bank4, SonaBlockStateStereo8, SonaGroupParamsStereo4,
    SonaParaLruCoeffs4, SonaToneWeights4, StereoVec8, T1DcBlockerKernel, T1SubHarmonicKernel,
    TptSvfCoeffs4, f32_to_fp16,
};

use crate::container::ZohRescaledParams;
use crate::graph::GraphBackend;
use crate::graph::cabinet::{CabinetHighSrFirExt, KautzColdSlewState, KautzStereoBank4};
use crate::graph::router::CrossbarParams;
use crate::graph::sona_core::frontend::{SonaOversamplerParams, SonaStereoOversampler2x};
use crate::graph::sona_core::model::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_TANK_QUADS, MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS,
    SONA_FIR_MAX_TAPS, SonaControlBusParams, SonaGroupCold, SonaTileCold,
};
use crate::graph::tank::fdn::{
    PartitionedSchroederBank, TankExtRing, TankQuadHotCompact, V3_TANK_CAP,
};
use crate::math::{DefaultMathBackend, MathBackend, ScalarBackend, SimdVec4};
use crate::types::Tier;

pub const FLAG_OS2X: u8 = 1 << 0;
pub const FLAG_MONO_OUT: u8 = 1 << 1;
pub const FLAG_CROSSBAR_ACTIVE: u8 = 1 << 2;
pub const FLAG_FREEZE: u8 = 1 << 3;
pub const FLAG_XFADE_ACTIVE: u8 = 1 << 4;
pub const FLAG_COLD_FIR_EXT: u8 = 1 << 5;
pub const TIER_SWITCH_XFADE_SAMPLES: u16 = 48;

/// Compact `144 B` (`9`-float-equivalent) hot `4`-lane stereo `SONA` group.
///
/// Recursive poles (`svf_g`, `lru_alpha_rel`, `lru_alpha_atk`) remain `f32`; non-recursive static
/// weights use IEEE-754 binary16 while `rect_sign` polarity bits pack into the slab bitmask.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct SonaGroupHot {
    pub svf_g: [f32; 4],
    pub lru_alpha_rel: [f32; 4],
    pub lru_alpha_atk: [f32; 4],
    pub svf_k_fp16: [u16; 4],
    pub lru_kappa_fp16: [u16; 4],
    pub adaa_drive_fp16: [u16; 4],
    pub lru_bias_drive_fp16: [u16; 4],
    pub in_x_weight_fp16: [u16; 4],
    pub tone_lp_fp16: [u16; 4],
    pub tone_bp_fp16: [u16; 4],
    pub tone_hp_fp16: [u16; 4],
    pub tone_ap_fp16: [u16; 4],
    pub tone_dir_fp16: [u16; 4],
    pub sag_depth_fp16: [u16; 4],
    pub wavefolder_mix_fp16: [u16; 4],
}

const _: () = assert!(size_of::<SonaGroupHot>() == 144);

impl Default for SonaGroupHot {
    fn default() -> Self {
        Self::from_cold(&SonaGroupCold::identity_group(0))
    }
}

impl SonaGroupHot {
    /// Compiles `(group, rect_sign_mask_4bit)` from cold parameters (`0` RT allocs).
    #[must_use]
    pub fn from_cold_with_mask(c: &SonaGroupCold) -> (Self, u8) {
        let mut p = *c;
        p.clamp_in_place();
        let mut mask = 0_u8;
        for l in 0..4 {
            if p.rect_sign[l] < 0.0 {
                mask |= 1 << l;
            }
        }
        let hot = Self {
            svf_g: p.svf_g,
            lru_alpha_rel: p.lru_alpha_rel,
            lru_alpha_atk: p.lru_alpha_atk,
            svf_k_fp16: p.svf_k.map(f32_to_fp16),
            lru_kappa_fp16: p.lru_kappa.map(f32_to_fp16),
            adaa_drive_fp16: p.adaa_drive.map(f32_to_fp16),
            lru_bias_drive_fp16: p.lru_bias_drive.map(f32_to_fp16),
            in_x_weight_fp16: p.in_x_weight.map(f32_to_fp16),
            tone_lp_fp16: p.tone_lp.map(f32_to_fp16),
            tone_bp_fp16: p.tone_bp.map(f32_to_fp16),
            tone_hp_fp16: p.tone_hp.map(f32_to_fp16),
            tone_ap_fp16: p.tone_ap.map(f32_to_fp16),
            tone_dir_fp16: p.tone_dir.map(f32_to_fp16),
            sag_depth_fp16: p.sag_depth.map(f32_to_fp16),
            wavefolder_mix_fp16: p.wavefolder_mix.map(f32_to_fp16),
        };
        (hot, mask)
    }

    /// Compiles hot `144 B` group representation from cold parameters.
    #[must_use]
    pub fn from_cold(c: &SonaGroupCold) -> Self {
        Self::from_cold_with_mask(c).0
    }

    /// Loads `in_x_weight` `[f32; 4]` directly into a SIMD register.
    #[inline(always)]
    #[must_use]
    pub fn in_x_weight_vec4<B: MathBackend>(&self) -> B::Vec4 {
        B::fp16x4_to_vec4(&self.in_x_weight_fp16)
    }

    /// Materializes `SonaGroupParamsStereo4<B::Vec4>` once per `16`-frame tick with per-lane mods.
    #[inline(always)]
    #[must_use]
    pub fn to_kernel_params_modulated<B: MathBackend>(
        &self,
        g_scale: B::Vec4,
        k_delta: B::Vec4,
        adaa_scale: f32,
        bias_delta: f32,
        wf_delta: B::Vec4,
        rect_mask4: u8,
    ) -> SonaGroupParamsStereo4<B::Vec4> {
        let g = B::Vec4::load(&self.svf_g)
            .mul(g_scale)
            .clamp(B::Vec4::splat(1.0e-5), B::Vec4::splat(64.0));
        let k = B::fp16x4_to_vec4(&self.svf_k_fp16)
            .add(k_delta)
            .clamp(B::Vec4::zero(), B::Vec4::splat(8.0));
        let sign_arr = [
            if (rect_mask4 & 1) != 0 { -1.0_f32 } else { 1.0 },
            if (rect_mask4 & 2) != 0 { -1.0_f32 } else { 1.0 },
            if (rect_mask4 & 4) != 0 { -1.0_f32 } else { 1.0 },
            if (rect_mask4 & 8) != 0 { -1.0_f32 } else { 1.0 },
        ];
        let rect_sign = B::Vec4::load(&sign_arr);
        SonaGroupParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: B::fp16x4_to_vec4(&self.tone_lp_fp16),
                bp: B::fp16x4_to_vec4(&self.tone_bp_fp16),
                hp: B::fp16x4_to_vec4(&self.tone_hp_fp16),
                ap: B::fp16x4_to_vec4(&self.tone_ap_fp16),
                dir: B::fp16x4_to_vec4(&self.tone_dir_fp16),
            },
            lru: SonaParaLruCoeffs4::from_clamped(
                B::Vec4::load(&self.lru_alpha_rel),
                B::Vec4::load(&self.lru_alpha_atk),
                B::fp16x4_to_vec4(&self.lru_kappa_fp16),
            ),
            rect_sign,
            sag_depth: B::fp16x4_to_vec4(&self.sag_depth_fp16),
            adaa_drive: B::fp16x4_to_vec4(&self.adaa_drive_fp16).mul(B::Vec4::splat(adaa_scale)),
            lru_bias_drive: B::fp16x4_to_vec4(&self.lru_bias_drive_fp16)
                .fma(rect_sign, B::Vec4::splat(bias_delta)),
            wavefolder_mix: B::fp16x4_to_vec4(&self.wavefolder_mix_fp16)
                .add(wf_delta)
                .clamp(B::Vec4::zero(), B::Vec4::splat(1.0)),
        }
    }

    /// Materializes `SonaGroupParamsStereo4<B::Vec4>` with uniform group-level scalars.
    #[inline(always)]
    #[must_use]
    pub fn to_kernel_params<B: MathBackend>(
        &self,
        g_scale: f32,
        k_delta: f32,
        adaa_scale: f32,
        bias_delta: f32,
    ) -> SonaGroupParamsStereo4<B::Vec4> {
        self.to_kernel_params_modulated::<B>(
            B::Vec4::splat(g_scale),
            B::Vec4::splat(k_delta),
            adaa_scale,
            bias_delta,
            B::Vec4::zero(),
            0,
        )
    }
}

/// Compact `128 B` hot `4x4` inter-block spatial mixing tile matching
/// [`crate::graph::weights::SonaInterBlockTilePairWire`].
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct SonaInterBlockTileHot {
    pub w_odd_4x4: [[f32; 4]; 4],
    pub w_even_4x4_fp16: [[u16; 4]; 4],
    pub w_lru_4x4_fp16: [[u16; 4]; 4],
}

const _: () = assert!(size_of::<SonaInterBlockTileHot>() == 128);

impl Default for SonaInterBlockTileHot {
    fn default() -> Self {
        Self::from_cold(&SonaTileCold::zeros())
    }
}

impl SonaInterBlockTileHot {
    /// Compiles `128 B` hot tile from cold `4x4` weights.
    #[must_use]
    pub fn from_cold(t: &SonaTileCold) -> Self {
        let mut c = *t;
        c.clamp_in_place();
        Self {
            w_odd_4x4: c.w_odd_4x4,
            w_even_4x4_fp16: c.w_even_4x4.map(|row| row.map(f32_to_fp16)),
            w_lru_4x4_fp16: c.w_lru_4x4.map(|row| row.map(f32_to_fp16)),
        }
    }

    /// Decodes `w_even` & `w_lru` `4x4` `fp16` matrices into `f32` column arrays.
    #[inline(always)]
    #[must_use]
    pub fn decode_even_lru<B: MathBackend>(&self) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
        (
            [
                B::fp16x4_to_f32x4(&self.w_even_4x4_fp16[0]),
                B::fp16x4_to_f32x4(&self.w_even_4x4_fp16[1]),
                B::fp16x4_to_f32x4(&self.w_even_4x4_fp16[2]),
                B::fp16x4_to_f32x4(&self.w_even_4x4_fp16[3]),
            ],
            [
                B::fp16x4_to_f32x4(&self.w_lru_4x4_fp16[0]),
                B::fp16x4_to_f32x4(&self.w_lru_4x4_fp16[1]),
                B::fp16x4_to_f32x4(&self.w_lru_4x4_fp16[2]),
                B::fp16x4_to_f32x4(&self.w_lru_4x4_fp16[3]),
            ],
        )
    }
}

/// Group-major `fp16` `VarPro` linear readout slice (`96 B` per group: `6` stereo pairs × `4` lanes).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SonaGroupReadoutHot {
    pub pre_cab_odd_l_fp16: [u16; 4],
    pub pre_cab_odd_r_fp16: [u16; 4],
    pub pre_cab_even_l_fp16: [u16; 4],
    pub pre_cab_even_r_fp16: [u16; 4],
    pub pre_cab_lru_l_fp16: [u16; 4],
    pub pre_cab_lru_r_fp16: [u16; 4],
    pub direct_odd_l_fp16: [u16; 4],
    pub direct_odd_r_fp16: [u16; 4],
    pub direct_even_l_fp16: [u16; 4],
    pub direct_even_r_fp16: [u16; 4],
    pub direct_lru_l_fp16: [u16; 4],
    pub direct_lru_r_fp16: [u16; 4],
}

const _: () = assert!(size_of::<SonaGroupReadoutHot>() == 96);

/// Register-hoisted stereo readout weights decoded once per `16`-frame tick (+ `in_x_weight`).
#[derive(Clone, Copy, Debug)]
pub struct SonaGroupReadoutRegs<V: SimdVec4, S: StereoVec8<Half = V>> {
    pub in_x_weight: V,
    pub op_p_w: V,
    pub ph_w: V,
    pub pre_cab_odd: S,
    pub pre_cab_even: S,
    pub pre_cab_lru: S,
    pub direct_odd: S,
    pub direct_even: S,
    pub direct_lru: S,
}

impl SonaGroupReadoutHot {
    /// Decodes one group's `fp16` readout weights and crossbar weights into SIMD registers once per tick.
    #[inline(always)]
    #[must_use]
    pub fn decode_regs<B: GraphBackend>(
        &self,
        in_x_weight: B::Vec4,
        op_p_w: B::Vec4,
        ph_w: B::Vec4,
    ) -> SonaGroupReadoutRegs<B::Vec4, B::Stereo8> {
        let pair = |l: &[u16; 4], r: &[u16; 4]| {
            B::Stereo8::from_lr(B::fp16x4_to_vec4(l), B::fp16x4_to_vec4(r))
        };
        SonaGroupReadoutRegs {
            in_x_weight,
            op_p_w,
            ph_w,
            pre_cab_odd: pair(&self.pre_cab_odd_l_fp16, &self.pre_cab_odd_r_fp16),
            pre_cab_even: pair(&self.pre_cab_even_l_fp16, &self.pre_cab_even_r_fp16),
            pre_cab_lru: pair(&self.pre_cab_lru_l_fp16, &self.pre_cab_lru_r_fp16),
            direct_odd: pair(&self.direct_odd_l_fp16, &self.direct_odd_r_fp16),
            direct_even: pair(&self.direct_even_l_fp16, &self.direct_even_r_fp16),
            direct_lru: pair(&self.direct_lru_l_fp16, &self.direct_lru_r_fp16),
        }
    }
}

/// Top-of-frame crossbar values computed once per host frame outside group and `2x` OS loops.
#[derive(Clone, Copy, Debug)]
pub struct CrossbarFrame<V: SimdVec4> {
    pub active: bool,
    pub op_p_vec: V,
    pub ssb_l: V,
    pub ssb_r: V,
}

/// Contiguous `4,608 B` (`72 CL`) L1D hot prefix executing every `Tier::Lite` audio frame.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaLiteHotPrefix<B: GraphBackend> {
    pub sona_states: [SonaBlockStateStereo8<B::Stereo8>; 4],
    pub sona_groups: [SonaGroupHot; 4],
    pub sona_tiles: [SonaInterBlockTileHot; 3],
    pub ro_groups: [SonaGroupReadoutHot; 4],
    pub fir_taps_l: [[[f32; 4]; 4]; 2],
    pub fir_taps_r: [[[f32; 4]; 4]; 2],
    pub fir_hist_l: [f32; 64],
    pub fir_hist_r: [f32; 64],
    pub kautz: [KautzStereoBank4; 2],
    pub quad0_frames: [[u16; 4]; V3_TANK_CAP],
    pub quad0: TankQuadHotCompact,
    pub hilbert: HilbertH1Kernel,
    pub t1: T1SubHarmonicKernel,
    pub phasors: PhasorQ32Bank4,
    pub t1_dc: T1DcBlockerKernel,
    pub stage0_xo_x0: [f32; 2],
    pub stage0_xo_y_l: [f32; 4],
    pub stage0_xo_y_r: [f32; 4],
    pub stage0_xo_pole: f32,
    pub stage0_gate_env: f32,
    pub stage0_gate_gain: f32,
    pub stage0_hp_lp_pole: f32,
    pub stage0_hp_lp_state: [f32; 2],
    pub ro_gains: [f32; 8],
    pub sample_tick: u32,
    pub op_p_to_group_lite_fp16: [[u16; 4]; 4],
    pub phasor_to_group_lite_fp16: [[u16; 4]; 4],
    pub opt_to_lr_lite_fp16: [[u16; 2]; 4],
    pub phasor_st_cos_fp16: [u16; 4],
    pub phasor_st_sin_fp16: [u16; 4],
    pub stage0_gain_mix_fp16: [u16; 6],
    pub stage0_xo_fp16: [u16; 3],
    pub stage0_gate_fp16: [u16; 6],
    pub op_gains_fp16: [u16; 2],
    pub fir_write_pos: u16,
    pub fir_onset_samples: u16,
    pub rect_sign_mask_lite: u16,
    pub tier: Tier,
    pub flags: u8,
}

/// Optional `2x` zero-lag oversampler hot block (`192 B`) immediately following `lite`.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SonaOs2xHot {
    pub os_params: SonaOversamplerParams,
    pub os_stereo: SonaStereoOversampler2x,
}

/// `Tier::Full` hot delta slice (`2,560 B`: groups `4..6`, tiles `3..13`, FIR `32..48`, Kautz `2..3`, Quad `1`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaFullHotDelta<B: GraphBackend> {
    pub sona_states_full: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub sona_groups_full: [SonaGroupHot; 2],
    pub sona_tiles_full: [SonaInterBlockTileHot; 10],
    pub ro_groups_full: [SonaGroupReadoutHot; 2],
    pub fir_taps_l_full: [[[f32; 4]; 4]; 1],
    pub fir_taps_r_full: [[[f32; 4]; 4]; 1],
    pub kautz_full: [KautzStereoBank4; 1],
    pub quad1: TankQuadHotCompact,
    pub op_p_to_group_full_fp16: [[u16; 4]; 2],
    pub phasor_to_group_full_fp16: [[u16; 4]; 2],
    pub opt_to_lr_full_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_full: u8,
}

/// `Tier::Pro` hot delta slice (`5,184 B`: groups `6..8`, tiles `13..23`, FIR `48..96` + `128`-cap ring, Kautz `3..6`, Quad `2`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaProHotDelta<B: GraphBackend> {
    pub sona_states_pro: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub sona_groups_pro: [SonaGroupHot; 2],
    pub sona_tiles_pro: [SonaInterBlockTileHot; 10],
    pub ro_groups_pro: [SonaGroupReadoutHot; 2],
    pub fir_taps_l_pro: [[[f32; 4]; 4]; 3],
    pub fir_taps_r_pro: [[[f32; 4]; 4]; 3],
    pub fir_hist_l_pro: [f32; 256],
    pub fir_hist_r_pro: [f32; 256],
    pub kautz_pro: [KautzStereoBank4; 3],
    pub quad2: TankQuadHotCompact,
    pub op_p_to_group_pro_fp16: [[u16; 4]; 2],
    pub phasor_to_group_pro_fp16: [[u16; 4]; 2],
    pub opt_to_lr_pro_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_pro: u8,
}

/// `Tier::Ultra` hot delta slice (`3,136 B`: groups `8..10`, tiles `23..35`, FIR `96..128`, Kautz `6..8`, Quad `3`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaUltraHotDelta<B: GraphBackend> {
    pub sona_states_ultra: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub sona_groups_ultra: [SonaGroupHot; 2],
    pub sona_tiles_ultra: [SonaInterBlockTileHot; 12],
    pub ro_groups_ultra: [SonaGroupReadoutHot; 2],
    pub fir_taps_l_ultra: [[[f32; 4]; 4]; 2],
    pub fir_taps_r_ultra: [[[f32; 4]; 4]; 2],
    pub kautz_ultra: [KautzStereoBank4; 2],
    pub quad3: TankQuadHotCompact,
    pub op_p_to_group_ultra_fp16: [[u16; 4]; 2],
    pub phasor_to_group_ultra_fp16: [[u16; 4]; 2],
    pub opt_to_lr_ultra_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_ultra: u8,
}

/// Trailing `3 kHz` ControlBus runtime modulation targets for [`super::SonaHotSlab`].
#[derive(Clone, Copy, Debug)]
pub struct SonaControlBusState {
    pub env_fc_scale: f32,
    pub env_k_delta: f32,
    pub block_adaa_scale: [f32; NUM_SONA_BLOCKS],
    pub block_lru_bias: [f32; NUM_SONA_BLOCKS],
    pub group_adaa_scale: [f32; MAX_SONA_GROUPS],
    pub group_lru_bias: [f32; MAX_SONA_GROUPS],
    pub lane_g_scale: [f32; MAX_TOTAL_SONA_LANES],
    pub lane_k_delta: [f32; MAX_TOTAL_SONA_LANES],
    pub lane_wf_delta: [f32; MAX_TOTAL_SONA_LANES],
    pub kautz_oct_delta: f32,
    pub kautz_damp_delta: f32,
    pub tank_delay_mod: f32,
    pub tank_gain_mod: f32,
    pub tank_psi_delta: f32,
    pub crossbar_wet_mod: f32,
    pub core_mods_active: bool,
    pub dirty: bool,
}

impl Default for SonaControlBusState {
    fn default() -> Self {
        Self::identity()
    }
}

impl SonaControlBusState {
    /// Resting identity control state (`1.0` multiplicative scales, `0.0` additive offsets).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            env_fc_scale: 1.0,
            env_k_delta: 0.0,
            block_adaa_scale: [1.0; NUM_SONA_BLOCKS],
            block_lru_bias: [0.0; NUM_SONA_BLOCKS],
            group_adaa_scale: [1.0; MAX_SONA_GROUPS],
            group_lru_bias: [0.0; MAX_SONA_GROUPS],
            lane_g_scale: [1.0; MAX_TOTAL_SONA_LANES],
            lane_k_delta: [0.0; MAX_TOTAL_SONA_LANES],
            lane_wf_delta: [0.0; MAX_TOTAL_SONA_LANES],
            kautz_oct_delta: 0.0,
            kautz_damp_delta: 0.0,
            tank_delay_mod: 0.0,
            tank_gain_mod: 0.0,
            tank_psi_delta: 0.0,
            crossbar_wet_mod: 0.0,
            core_mods_active: false,
            dirty: false,
        }
    }
}

/// Out-of-L1D cold storage for base ZOH parameters, `stage3_fir_ext`, `stage4_ext`, Schroeder,
/// `stage_latent_*` tables, and tier-switch crossfade history.
#[derive(Clone, Debug)]
pub struct SonaColdTail {
    pub base_stage0: CrossbarParams,
    pub base_groups: [SonaGroupCold; MAX_SONA_GROUPS],
    pub base_os_params: SonaOversamplerParams,
    pub base_fir_l: [f32; SONA_FIR_MAX_TAPS],
    pub base_fir_r: [f32; SONA_FIR_MAX_TAPS],
    pub base_kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub base_kautz_b_gamma: [f32; MAX_KAUTZ_POLES],
    pub base_tank_g_loop: [f32; MAX_TANK_QUADS],
    pub base_tank_wet_mix: [f32; MAX_TANK_QUADS],
    pub base_op_p_gain: f32,
    pub base_op_t_gain: f32,
    pub base_active_quads: usize,
    pub kautz_slew: KautzColdSlewState,
    pub stage3_fir_ext: CabinetHighSrFirExt,
    pub(crate) stage4_ext: [Option<TankExtRing>; MAX_TANK_QUADS],
    pub stage4_ext_active: [bool; MAX_TANK_QUADS],
    pub(crate) stage4_schroeder: [Option<PartitionedSchroederBank>; MAX_TANK_QUADS],
    pub ctrl_params: Box<SonaControlBusParams>,
    pub ctrl_state: SonaControlBusState,
    pub crossbar_freqs_hz: [f32; 4],
    pub crossbar_init_phase_q32: [u32; 4],
    pub crossbar_stereo_phase_rad: [f32; 4],
    pub last_in: [f32; 2],
    pub last_s2_dir: [f32; 4],
    pub last_stage_targets: [f32; 16],
    pub last_z_stage: [f32; 8],
    pub xfade_rem: u16,
    pub xfade_total: u16,
    pub xfade_prev_out: [f32; 2],
    pub zoh: Option<ZohRescaledParams>,
    pub os_factor: usize,
    pub is_bounce_4x: bool,
    pub base_crossbar_active: bool,
    pub max_op_n_drive: f32,
    pub max_pole_radius: f32,
}

/// Exact `ActiveHot` working-set bytes `[0..ActiveHotBytes)` per `(tier, os2x)` matching
/// `SonaLiteHotPrefix..SonaUltraHotDelta` cache-line boundaries (`<= 4,608 / 7,936 / 15,872 / 28,672 B`).
#[inline]
#[must_use]
pub const fn sona_slab_active_hot_bytes(tier: Tier, os2x: bool) -> usize {
    let lite = size_of::<SonaLiteHotPrefix<DefaultMathBackend>>();
    let os = if os2x { size_of::<SonaOs2xHot>() } else { 0 };
    let full = size_of::<SonaFullHotDelta<DefaultMathBackend>>();
    let pro = size_of::<SonaProHotDelta<DefaultMathBackend>>();
    let ultra = size_of::<SonaUltraHotDelta<DefaultMathBackend>>();
    match tier {
        Tier::Lite => lite + os,
        Tier::Full => lite + os + full,
        Tier::Pro => lite + os + full + pro,
        Tier::Ultra => lite + os + full + pro + ultra,
    }
}

const _: () = assert!(size_of::<SonaLiteHotPrefix<DefaultMathBackend>>() <= 4_608);
const _: () = assert!(size_of::<SonaLiteHotPrefix<ScalarBackend>>() <= 4_608);
const _: () = assert!(sona_slab_active_hot_bytes(Tier::Lite, false) <= 4_608);
const _: () = assert!(sona_slab_active_hot_bytes(Tier::Full, true) <= 7_936);
const _: () = assert!(sona_slab_active_hot_bytes(Tier::Pro, true) <= 15_872);
const _: () = assert!(sona_slab_active_hot_bytes(Tier::Ultra, true) <= 28_672);
const _: () = assert!(
    offset_of!(super::SonaHotSlab<DefaultMathBackend>, os2x_hot)
        >= size_of::<SonaLiteHotPrefix<DefaultMathBackend>>()
);
const _: () = assert!(
    offset_of!(super::SonaHotSlab<DefaultMathBackend>, full_delta)
        + size_of::<SonaFullHotDelta<DefaultMathBackend>>()
        <= 7_936
);
const _: () = assert!(
    offset_of!(super::SonaHotSlab<DefaultMathBackend>, pro_delta)
        + size_of::<SonaProHotDelta<DefaultMathBackend>>()
        <= 15_872
);
const _: () = assert!(
    offset_of!(super::SonaHotSlab<DefaultMathBackend>, ultra_delta)
        + size_of::<SonaUltraHotDelta<DefaultMathBackend>>()
        <= 28_672
);
const _: () = assert!(
    offset_of!(super::SonaHotSlab<DefaultMathBackend>, cold)
        >= offset_of!(super::SonaHotSlab<DefaultMathBackend>, ultra_delta)
            + size_of::<SonaUltraHotDelta<DefaultMathBackend>>()
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sona_hot_prefix_and_active_hot_budgets() {
        assert_eq!(size_of::<SonaLiteHotPrefix<DefaultMathBackend>>(), 4_608);
        assert_eq!(size_of::<SonaOs2xHot>(), 192);
        assert!(sona_slab_active_hot_bytes(Tier::Lite, false) == 4_608);
        assert!(sona_slab_active_hot_bytes(Tier::Full, true) <= 7_936);
        assert!(sona_slab_active_hot_bytes(Tier::Pro, true) <= 15_872);
        assert!(sona_slab_active_hot_bytes(Tier::Ultra, true) <= 28_672);
        let _ = MAX_TANK_QUADS;
    }

    #[test]
    fn test_sona_hot_group_kernel_params_tile_decode_and_ctrl_default() {
        use crate::math::SimdVec4;
        let mut cold_g = SonaGroupCold::identity_group(0);
        cold_g.rect_sign = [-1.0; 4];
        cold_g.wavefolder_mix = [0.25; 4];
        let (hot_g, mask4) = SonaGroupHot::from_cold_with_mask(&cold_g);
        assert_eq!(mask4, 0x0F);

        let p_uni = hot_g.to_kernel_params::<DefaultMathBackend>(1.2, 0.1, 0.8, -0.05);
        let p_mod = hot_g.to_kernel_params_modulated::<DefaultMathBackend>(
            <DefaultMathBackend as MathBackend>::Vec4::splat(1.2),
            <DefaultMathBackend as MathBackend>::Vec4::splat(0.1),
            0.8,
            -0.05,
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            0,
        );
        assert_eq!(p_uni.adaa_drive.to_array(), p_mod.adaa_drive.to_array());
        assert_eq!(
            p_uni.lru_bias_drive.to_array(),
            p_mod.lru_bias_drive.to_array()
        );
        assert_eq!(
            p_uni.wavefolder_mix.to_array(),
            p_mod.wavefolder_mix.to_array()
        );
        assert_eq!(p_uni.rect_sign.to_array(), p_mod.rect_sign.to_array());
        let p_neg = hot_g.to_kernel_params_modulated::<ScalarBackend>(
            <ScalarBackend as MathBackend>::Vec4::splat(1.0),
            <ScalarBackend as MathBackend>::Vec4::zero(),
            1.0,
            0.0,
            <ScalarBackend as MathBackend>::Vec4::zero(),
            mask4,
        );
        assert_eq!(p_neg.rect_sign.to_array(), [-1.0; 4]);

        let mut cold_t = SonaTileCold::zeros();
        cold_t.w_even_4x4[0] = [0.25, -0.5, 0.75, 1.0];
        cold_t.w_lru_4x4[3] = [-0.25, 0.5, -0.75, 1.5];
        let hot_t = SonaInterBlockTileHot::from_cold(&cold_t);
        let (ev_simd, lru_simd) = hot_t.decode_even_lru::<DefaultMathBackend>();
        let (ev_scal, lru_scal) = hot_t.decode_even_lru::<ScalarBackend>();
        assert_eq!(
            ev_simd.map(|row| row.map(f32::to_bits)),
            ev_scal.map(|row| row.map(f32::to_bits))
        );
        assert_eq!(
            lru_simd.map(|row| row.map(f32::to_bits)),
            lru_scal.map(|row| row.map(f32::to_bits))
        );
        assert_eq!(ev_simd[0], [0.25, -0.5, 0.75, 1.0]);
        assert_eq!(lru_simd[3], [-0.25, 0.5, -0.75, 1.5]);

        let ctrl = SonaControlBusState::default();
        assert_eq!(ctrl.env_fc_scale, 1.0);
        assert_eq!(ctrl.env_k_delta, 0.0);
        assert!(!ctrl.core_mods_active);
        assert!(!ctrl.dirty);
    }
}
