//! Prefix-ordered L1D `SonaHotSlab` (`4,544 / 4,800 / 7,552 / 11,200 / 14,656 B`) layouts and
//! canonical Matryoshka topology constants (`#![no_std]`, `#![forbid(unsafe_code)]`, `ref[arch.tiers.state-ram]`).

#![forbid(unsafe_code)]

pub mod cold;

use core::mem::{align_of, offset_of, size_of};

use dnam_kernels::{
    HilbertH1StereoKernel, OpNPreHotState, PhasorQ32Bank4, SonaBlockStateStereo8,
    SonaGroupParamsStereo4, SonaParaLruCoeffs4, SonaToneWeights4, StereoVec8, T1DcBlockerStereo,
    T1SubHarmonicKernel, TptSvfCoeffs4,
};

use crate::graph::GraphBackend;
use crate::graph::cabinet::kautz::KautzStereoBank4;
use crate::graph::oversample::SonaOs2xHot;
use crate::graph::tank::TankQuadHotCompact;
use crate::graph::tank::ring::TANK_INLINE_QUAD0_CAP;
use crate::math::{DefaultMathBackend, MathBackend, ScalarBackend, SimdVec4, f32_to_fp16};
use crate::types::Tier;

pub use cold::{SonaColdTail, SonaHotSlab, SonaSlabBacking, TierCrossbarSnapshot};

/// Hot slab flag: core `2x` zero-lag oversampling (`OS2X`) is active (`Full..Ultra`).
pub const FLAG_OS2X: u8 = 1 << 0;
/// Hot slab flag: fold final stereo output into identical mono channels `(0.5 * (L + R))`.
pub const FLAG_MONO_OUT: u8 = 1 << 1;
/// Hot slab flag: Top-of-Frame crossbar (`Op-P`, `Q0.32` phasors, `Op-T` taps) is active.
pub const FLAG_CROSSBAR_ACTIVE: u8 = 1 << 2;
/// Hot slab flag: deterministic `48`-sample tier-switch output crossfade is active.
pub const FLAG_XFADE_ACTIVE: u8 = 1 << 3;
/// Hot slab flag: Cabinet Early FIR executes via off-`48 kHz` cold extension (`6,144` floats).
pub const FLAG_COLD_FIR_EXT: u8 = 1 << 4;
/// Hot slab flag: model expects single-channel (`mono_in`) routing on input ports.
pub const FLAG_SINGLE_INPUT: u8 = 1 << 5;
/// Hot slab flag: invert wet output polarity (`-1.0`) when configured by container metadata.
pub const FLAG_WET_POLARITY_INV: u8 = 1 << 6;
/// Fixed linear crossfade duration across runtime tier switches (`48` host frames).
pub const TIER_SWITCH_XFADE_SAMPLES: u16 = 48;

/// Maximum sequential Matryoshka blocks across every SONA tier (`4`).
pub const NUM_SONA_BLOCKS: usize = 4;
/// Maximum `4`-lane stereo SONA groups across `Ultra` (`4 + 3 + 2 + 1 = 10`).
pub const MAX_SONA_GROUPS: usize = 10;
/// Maximum total SONA core lanes (`10 * 4 = 40`).
pub const MAX_TOTAL_SONA_LANES: usize = 40;
/// Maximum lower-triangular `4x4` inter-block tile pairs (`3 + 10 + 10 + 12 = 35`).
pub const MAX_SONA_TILES: usize = 35;

/// Canonical global `(block_idx, local_group_idx)` assignment for groups `0..10`.
pub const SONA_GROUP_BLOCK_AND_LOCAL: [(usize, usize); MAX_SONA_GROUPS] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (0, 2),
    (2, 1),
    (0, 3),
    (1, 2),
];

/// Group slots per Matryoshka block `[Block 0..Block 3]` in prefix-nested order.
pub const SONA_BLOCK_GROUP_SLOTS: [[Option<usize>; 4]; NUM_SONA_BLOCKS] = [
    [Some(0), Some(4), Some(6), Some(8)],
    [Some(1), Some(5), Some(9), None],
    [Some(2), Some(7), None, None],
    [Some(3), None, None, None],
];

/// Causal Matryoshka block-major group evaluation sequence across all `10` Ultra groups.
pub const SONA_ULTRA_GROUP_TRAVERSAL_ORDER: [usize; MAX_SONA_GROUPS] =
    [0, 4, 6, 8, 1, 5, 9, 2, 7, 3];

/// All `35` lower-triangular inter-block edges `(src_group, dst_group)` ordered by tier arrival
/// (`0..3` `Lite`, `3..13` `Full`, `13..23` `Pro`, `23..35` `Ultra`).
pub const SONA_INTERBLOCK_TILES: [(usize, usize); MAX_SONA_TILES] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 5),
    (4, 1),
    (4, 5),
    (0, 2),
    (4, 2),
    (0, 3),
    (4, 3),
    (5, 2),
    (1, 3),
    (5, 3),
    (6, 1),
    (6, 5),
    (0, 7),
    (4, 7),
    (6, 2),
    (6, 7),
    (6, 3),
    (1, 7),
    (5, 7),
    (7, 3),
    (0, 9),
    (4, 9),
    (6, 9),
    (8, 1),
    (8, 5),
    (8, 9),
    (8, 2),
    (8, 7),
    (8, 3),
    (9, 2),
    (9, 7),
    (9, 3),
];

const fn verify_interblock_tiles_and_slots() -> bool {
    let mut i = 0;
    while i < MAX_SONA_TILES {
        let (sg, dg) = SONA_INTERBLOCK_TILES[i];
        if sg >= MAX_SONA_GROUPS || dg >= MAX_SONA_GROUPS {
            return false;
        }
        if SONA_GROUP_BLOCK_AND_LOCAL[sg].0 >= SONA_GROUP_BLOCK_AND_LOCAL[dg].0 {
            return false;
        }
        let mut j = i + 1;
        while j < MAX_SONA_TILES {
            if SONA_INTERBLOCK_TILES[j].0 == sg && SONA_INTERBLOCK_TILES[j].1 == dg {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

const _: () = assert!(verify_interblock_tiles_and_slots());

/// Compact `144 B` (`16 B`-aligned) hot representation of one `4`-lane stereo SONA group.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaGroupHot {
    /// Effective `TPT-SVF` tangent gain `g` at `f_{s,core}` across 4 lanes.
    pub svf_g: [f32; 4],
    /// ZOH-rescaled `ParaLRU` release poles `alpha_rel`.
    pub lru_alpha_rel: [f32; 4],
    /// ZOH-rescaled `ParaLRU` attack poles `alpha_atk`.
    pub lru_alpha_atk: [f32; 4],
    /// Base `TPT-SVF` damping `k` (`fp16`).
    pub svf_k_fp16: [u16; 4],
    /// Clamped `ParaLRU` detector gain `kappa` (`fp16`).
    pub lru_kappa_fp16: [u16; 4],
    /// Pre-activation `PolyADAA1` linear drive `(2/3) * d` (`fp16`).
    pub adaa_drive_fp16: [u16; 4],
    /// `ParaLRU` asymmetric bias drive (`fp16`).
    pub lru_bias_drive_fp16: [u16; 4],
    /// Direct frame input weight per lane (`fp16`).
    pub in_x_weight_fp16: [u16; 4],
    /// `TPT-SVF` low-pass tone tap weight (`fp16`).
    pub tone_lp_fp16: [u16; 4],
    /// `TPT-SVF` band-pass tone tap weight (`fp16`).
    pub tone_bp_fp16: [u16; 4],
    /// `TPT-SVF` high-pass tone tap weight (`fp16`).
    pub tone_hp_fp16: [u16; 4],
    /// `TPT-SVF` all-pass tone tap weight (`fp16`).
    pub tone_ap_fp16: [u16; 4],
    /// Direct bypass tone weight (`fp16`).
    pub tone_dir_fp16: [u16; 4],
    /// Power-supply sag depth (`fp16`).
    pub sag_depth_fp16: [u16; 4],
    /// `C^2` wavefolder blend ratio (`fp16`).
    pub wavefolder_mix_fp16: [u16; 4],
}

impl Default for SonaGroupHot {
    fn default() -> Self {
        Self {
            svf_g: [0.05; 4],
            lru_alpha_rel: [0.0; 4],
            lru_alpha_atk: [0.0; 4],
            svf_k_fp16: [f32_to_fp16(core::f32::consts::FRAC_1_SQRT_2); 4],
            lru_kappa_fp16: [0; 4],
            adaa_drive_fp16: [0; 4],
            lru_bias_drive_fp16: [0; 4],
            in_x_weight_fp16: [0; 4],
            tone_lp_fp16: [0; 4],
            tone_bp_fp16: [0; 4],
            tone_hp_fp16: [0; 4],
            tone_ap_fp16: [0; 4],
            tone_dir_fp16: [f32_to_fp16(1.0); 4],
            sag_depth_fp16: [0; 4],
            wavefolder_mix_fp16: [0; 4],
        }
    }
}

impl SonaGroupHot {
    /// Materializes `in_x_weight` `[f32; 4]` directly into a SIMD register.
    #[inline(always)]
    #[must_use]
    pub fn in_x_weight_vec4<B: MathBackend>(&self) -> B::Vec4 {
        B::fp16x4_to_vec4(&self.in_x_weight_fp16)
    }

    /// Materializes [`SonaGroupParamsStereo4`] in SIMD registers with per-lane `3 kHz` targets.
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
                .add(rect_sign.mul(B::Vec4::splat(bias_delta)))
                .clamp(B::Vec4::splat(-16.0), B::Vec4::splat(16.0)),
            wavefolder_mix: B::fp16x4_to_vec4(&self.wavefolder_mix_fp16)
                .add(wf_delta)
                .clamp(B::Vec4::zero(), B::Vec4::splat(1.0)),
        }
    }
}

/// Compact `128 B` (`16 B`-aligned) `4x4` column-major inter-block mixing tile.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SonaInterBlockTileHot {
    /// Odd-channel coupling `w_odd_4x4[src_lane][dst_lane]` in full `binary32`.
    pub w_odd_4x4: [[f32; 4]; 4],
    /// Even-channel coupling `w_even_4x4[src_lane][dst_lane]` in `fp16`.
    pub w_even_4x4_fp16: [[u16; 4]; 4],
    /// `ParaLRU` fast-path coupling `w_lru_4x4[src_lane][dst_lane]` in `fp16`.
    pub w_lru_4x4_fp16: [[u16; 4]; 4],
}

/// Group-major `96 B` (`8 B`-aligned) `VarPro` readout slice (`6` stereo pairs × `4` lanes).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// Register-resident unpacked stereo readout & crossbar weights for one group.
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
    /// Decodes one group's `96 B` `fp16` weights and crossbar vectors into SIMD registers.
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

/// Top-of-Frame crossbar outputs computed once per core sub-sample ahead of group evaluation.
#[derive(Clone, Copy, Debug)]
pub struct CrossbarFrame<V: SimdVec4> {
    pub active: bool,
    pub op_p_vec: V,
    pub car_l: V,
    pub car_r: V,
}

/// Contiguous `4,544 B` (`71 CL`) L1D hot prefix holding every `Tier::Lite` audio state and weight.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaLiteHotPrefix<B: GraphBackend> {
    pub core_states: [SonaBlockStateStereo8<B::Stereo8>; 4],
    pub groups: [SonaGroupHot; 4],
    pub tiles: [SonaInterBlockTileHot; 3],
    pub readouts: [SonaGroupReadoutHot; 4],
    pub kautz: [KautzStereoBank4; 2],
    pub tank_quad0: TankQuadHotCompact,
    pub tank0_ring: [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    pub fir_weights_l: [[[f32; 4]; 4]; 2],
    pub fir_weights_r: [[[f32; 4]; 4]; 2],
    pub fir_hist_l: [f32; 32],
    pub fir_hist_r: [f32; 32],
    pub hilbert: HilbertH1StereoKernel,
    pub t1: T1SubHarmonicKernel,
    pub phasors: PhasorQ32Bank4,
    pub t1_dc: T1DcBlockerStereo,
    pub op_n_pre: OpNPreHotState,
    pub gate_env: f32,
    pub gate_gain: f32,
    pub hp_lp_pole: f32,
    pub hp_lp_state: [f32; 2],
    pub sample_tick: u32,
    pub ro_gains_fp16: [u16; 8],
    pub carrier_basis_w_fp16: [[u16; 4]; 3],
    pub op_p_to_group_lite_fp16: [[u16; 4]; 4],
    pub phasor_to_group_lite_fp16: [[u16; 4]; 4],
    pub opt_to_lr_lite_fp16: [[u16; 2]; 4],
    pub phasor_st_cos_fp16: [u16; 4],
    pub phasor_st_sin_fp16: [u16; 4],
    pub _reserved_hot_header_u16: [u16; 15],
    pub rect_sign_mask_lite: u16,
    pub tier: Tier,
    pub flags: u8,
}

impl<B: GraphBackend> SonaLiteHotPrefix<B> {
    /// Zero-initialized `Tier::Lite` hot prefix (`!alloc`).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states: [SonaBlockStateStereo8::zero(); 4],
            groups: [SonaGroupHot::default(); 4],
            tiles: [SonaInterBlockTileHot::default(); 3],
            readouts: [SonaGroupReadoutHot::default(); 4],
            kautz: [KautzStereoBank4::default(); 2],
            tank_quad0: TankQuadHotCompact::default(),
            tank0_ring: [[0; 4]; TANK_INLINE_QUAD0_CAP],
            fir_weights_l: [[[0.0; 4]; 4]; 2],
            fir_weights_r: [[[0.0; 4]; 4]; 2],
            fir_hist_l: [0.0; 32],
            fir_hist_r: [0.0; 32],
            hilbert: HilbertH1StereoKernel::with_coeffs([0.0; 6], [0.0; 6]),
            t1: T1SubHarmonicKernel::new(),
            phasors: PhasorQ32Bank4::new([0; 4], [0; 4]),
            t1_dc: T1DcBlockerStereo::default(),
            op_n_pre: OpNPreHotState::default(),
            gate_env: 0.0,
            gate_gain: 1.0,
            hp_lp_pole: 0.0,
            hp_lp_state: [0.0; 2],
            sample_tick: 0,
            ro_gains_fp16: [0; 8],
            carrier_basis_w_fp16: [[0; 4]; 3],
            op_p_to_group_lite_fp16: [[0; 4]; 4],
            phasor_to_group_lite_fp16: [[0; 4]; 4],
            opt_to_lr_lite_fp16: [[0; 2]; 4],
            phasor_st_cos_fp16: [f32_to_fp16(1.0); 4],
            phasor_st_sin_fp16: [0; 4],
            _reserved_hot_header_u16: [0; 15],
            rect_sign_mask_lite: 0,
            tier: Tier::Lite,
            flags: 0,
        }
    }
}

/// `Tier::Full` hot delta (`2,752 B` = `43 CL`; cumulative `7,552 B` after `SonaOs2xHot`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaFullHotDelta<B: GraphBackend> {
    pub core_states_full: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub groups_full: [SonaGroupHot; 2],
    pub tiles_full: [SonaInterBlockTileHot; 10],
    pub readouts_full: [SonaGroupReadoutHot; 2],
    pub fir_weights_l_full: [[[f32; 4]; 4]; 1],
    pub fir_weights_r_full: [[[f32; 4]; 4]; 1],
    pub fir_hist_l_full: [f32; 16],
    pub fir_hist_r_full: [f32; 16],
    pub kautz_full: [KautzStereoBank4; 1],
    pub tank_quad1: TankQuadHotCompact,
    pub op_p_to_group_full_fp16: [[u16; 4]; 2],
    pub phasor_to_group_full_fp16: [[u16; 4]; 2],
    pub opt_to_lr_full_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_full: u8,
}

impl<B: GraphBackend> SonaFullHotDelta<B> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states_full: [SonaBlockStateStereo8::zero(); 2],
            groups_full: [SonaGroupHot::default(); 2],
            tiles_full: [SonaInterBlockTileHot::default(); 10],
            readouts_full: [SonaGroupReadoutHot::default(); 2],
            fir_weights_l_full: [[[0.0; 4]; 4]; 1],
            fir_weights_r_full: [[[0.0; 4]; 4]; 1],
            fir_hist_l_full: [0.0; 16],
            fir_hist_r_full: [0.0; 16],
            kautz_full: [KautzStereoBank4::default(); 1],
            tank_quad1: TankQuadHotCompact::default(),
            op_p_to_group_full_fp16: [[0; 4]; 2],
            phasor_to_group_full_fp16: [[0; 4]; 2],
            opt_to_lr_full_fp16: [[0; 2]; 4],
            rect_sign_mask_full: 0,
        }
    }
}

/// `Tier::Pro` hot delta (`3,648 B` = `57 CL`; cumulative `11,200 B`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaProHotDelta<B: GraphBackend> {
    pub core_states_pro: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub groups_pro: [SonaGroupHot; 2],
    pub tiles_pro: [SonaInterBlockTileHot; 10],
    pub readouts_pro: [SonaGroupReadoutHot; 2],
    pub fir_weights_l_pro: [[[f32; 4]; 4]; 3],
    pub fir_weights_r_pro: [[[f32; 4]; 4]; 3],
    pub fir_hist_l_pro: [f32; 48],
    pub fir_hist_r_pro: [f32; 48],
    pub kautz_pro: [KautzStereoBank4; 3],
    pub tank_quad2: TankQuadHotCompact,
    pub op_p_to_group_pro_fp16: [[u16; 4]; 2],
    pub phasor_to_group_pro_fp16: [[u16; 4]; 2],
    pub opt_to_lr_pro_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_pro: u8,
}

impl<B: GraphBackend> SonaProHotDelta<B> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states_pro: [SonaBlockStateStereo8::zero(); 2],
            groups_pro: [SonaGroupHot::default(); 2],
            tiles_pro: [SonaInterBlockTileHot::default(); 10],
            readouts_pro: [SonaGroupReadoutHot::default(); 2],
            fir_weights_l_pro: [[[0.0; 4]; 4]; 3],
            fir_weights_r_pro: [[[0.0; 4]; 4]; 3],
            fir_hist_l_pro: [0.0; 48],
            fir_hist_r_pro: [0.0; 48],
            kautz_pro: [KautzStereoBank4::default(); 3],
            tank_quad2: TankQuadHotCompact::default(),
            op_p_to_group_pro_fp16: [[0; 4]; 2],
            phasor_to_group_pro_fp16: [[0; 4]; 2],
            opt_to_lr_pro_fp16: [[0; 2]; 4],
            rect_sign_mask_pro: 0,
        }
    }
}

/// `Tier::Ultra` hot delta (`3,456 B` = `54 CL`; cumulative `14,656 B`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaUltraHotDelta<B: GraphBackend> {
    pub core_states_ultra: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub groups_ultra: [SonaGroupHot; 2],
    pub tiles_ultra: [SonaInterBlockTileHot; 12],
    pub readouts_ultra: [SonaGroupReadoutHot; 2],
    pub fir_weights_l_ultra: [[[f32; 4]; 4]; 2],
    pub fir_weights_r_ultra: [[[f32; 4]; 4]; 2],
    pub fir_hist_l_ultra: [f32; 32],
    pub fir_hist_r_ultra: [f32; 32],
    pub kautz_ultra: [KautzStereoBank4; 2],
    pub tank_quad3: TankQuadHotCompact,
    pub op_p_to_group_ultra_fp16: [[u16; 4]; 2],
    pub phasor_to_group_ultra_fp16: [[u16; 4]; 2],
    pub opt_to_lr_ultra_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_ultra: u8,
}

impl<B: GraphBackend> SonaUltraHotDelta<B> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states_ultra: [SonaBlockStateStereo8::zero(); 2],
            groups_ultra: [SonaGroupHot::default(); 2],
            tiles_ultra: [SonaInterBlockTileHot::default(); 12],
            readouts_ultra: [SonaGroupReadoutHot::default(); 2],
            fir_weights_l_ultra: [[[0.0; 4]; 4]; 2],
            fir_weights_r_ultra: [[[0.0; 4]; 4]; 2],
            fir_hist_l_ultra: [0.0; 32],
            fir_hist_r_ultra: [0.0; 32],
            kautz_ultra: [KautzStereoBank4::default(); 2],
            tank_quad3: TankQuadHotCompact::default(),
            op_p_to_group_ultra_fp16: [[0; 4]; 2],
            phasor_to_group_ultra_fp16: [[0; 4]; 2],
            opt_to_lr_ultra_fp16: [[0; 2]; 4],
            rect_sign_mask_ultra: 0,
        }
    }
}

const _: () = assert!(size_of::<SonaGroupHot>() == 144 && align_of::<SonaGroupHot>() == 16);
const _: () =
    assert!(size_of::<SonaInterBlockTileHot>() == 128 && align_of::<SonaInterBlockTileHot>() == 16);
const _: () =
    assert!(size_of::<SonaGroupReadoutHot>() == 96 && align_of::<SonaGroupReadoutHot>() == 8);
const _: () = assert!(size_of::<SonaOs2xHot>() == 256 && align_of::<SonaOs2xHot>() == 64);

const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, flags) + 1 == 4_518);
const _: () = assert!(
    size_of::<SonaLiteHotPrefix<ScalarBackend>>() == 4_544
        && align_of::<SonaLiteHotPrefix<ScalarBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaFullHotDelta<ScalarBackend>>() == 2_752
        && align_of::<SonaFullHotDelta<ScalarBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaProHotDelta<ScalarBackend>>() == 3_648
        && align_of::<SonaProHotDelta<ScalarBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaUltraHotDelta<ScalarBackend>>() == 3_456
        && align_of::<SonaUltraHotDelta<ScalarBackend>>() == 64
);

const _: () = assert!(
    size_of::<SonaLiteHotPrefix<DefaultMathBackend>>() == 4_544
        && align_of::<SonaLiteHotPrefix<DefaultMathBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaFullHotDelta<DefaultMathBackend>>() == 2_752
        && align_of::<SonaFullHotDelta<DefaultMathBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaProHotDelta<DefaultMathBackend>>() == 3_648
        && align_of::<SonaProHotDelta<DefaultMathBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaUltraHotDelta<DefaultMathBackend>>() == 3_456
        && align_of::<SonaUltraHotDelta<DefaultMathBackend>>() == 64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_prefix_topology_and_tile_count() {
        assert_eq!(
            SONA_ULTRA_GROUP_TRAVERSAL_ORDER,
            [0, 4, 6, 8, 1, 5, 9, 2, 7, 3]
        );
        assert_eq!(SONA_INTERBLOCK_TILES.len(), 35);
        assert_eq!(
            SONA_BLOCK_GROUP_SLOTS[0],
            [Some(0), Some(4), Some(6), Some(8)]
        );
        let lit = SonaLiteHotPrefix::<ScalarBackend>::empty();
        assert_eq!(lit.gate_gain, 1.0);

        let mut g = SonaGroupHot::default();
        for l in 0..4 {
            g.svf_g[l] = 0.25;
            g.lru_bias_drive_fp16[l] = crate::math::f32_to_fp16(if l < 2 { -15.0 } else { 15.0 });
        }
        let kp = g.to_kernel_params_modulated::<ScalarBackend>(
            crate::math::scalar::ScalarVec4([1.0; 4]),
            crate::math::scalar::ScalarVec4([0.0; 4]),
            1.0,
            2.0,
            crate::math::scalar::ScalarVec4([0.0; 4]),
            0b1010,
        );
        assert_eq!(kp.lru_bias_drive.to_array(), [-13.0, -16.0, 16.0, 13.0]);
    }
}
