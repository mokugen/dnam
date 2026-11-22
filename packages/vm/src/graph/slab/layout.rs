//! Prefix-ordered L1D `SonaHotSlab` (`4,544 / 4,800 / 7,552 / 11,136 / 14,592 B`) layouts and
//! canonical Matryoshka topology constants (`#![no_std]`, `#![forbid(unsafe_code)]`, `ref[arch.tiers.state-ram]`).

#![forbid(unsafe_code)]

pub mod cold;

use core::mem::{align_of, offset_of, size_of};

use dnam_kernels::{
    CoreQuadParamsStereo4, HilbertH1StereoKernel, MAX_SVF_K_NL, OpNPreHotState, PhasorQ32Bank4,
    SonaBlockStateStereo8, SonaParaLruDuhemCoeffs4, SonaToneWeights4, StereoVec8,
    T1DcBlockerStereo, T1SubHarmonicKernel, TptSvfCoeffs4,
};

use crate::graph::GraphBackend;
use crate::graph::convolver::kautz::KautzStereoBank4;
use crate::graph::oversample::SonaOs2xHot;
use crate::graph::sensory::SensoryQuadHot;
use crate::graph::transport::TransportQuadHotCompact;
use crate::graph::transport::ring::TRANSPORT_INLINE_QUAD0_CAP;
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
/// Hot slab flag: `ModalConvolver` Early FIR executes via off-`48 kHz` cold extension (`6,144` floats).
pub const FLAG_COLD_FIR_EXT: u8 = 1 << 4;
/// Hot slab flag: model expects single-channel (`mono_in`) routing on input ports.
pub const FLAG_SINGLE_INPUT: u8 = 1 << 5;
/// Hot slab flag: invert wet output polarity (`-1.0`) when configured by container metadata.
pub const FLAG_WET_POLARITY_INV: u8 = 1 << 6;
/// Fixed linear crossfade duration across runtime tier switches (`48` host frames).
pub const TIER_SWITCH_XFADE_SAMPLES: u16 = 48;

/// Exact byte footprint of [`SonaLiteHotPrefix`].
pub const SONA_LITE_HOT_BYTES: usize = 4_544;
/// Exact byte footprint of [`SonaOs2xHot`].
pub const SONA_OS2X_HOT_BYTES: usize = 256;
/// Exact byte delta of [`SonaFullHotDelta`].
pub const SONA_FULL_HOT_DELTA_BYTES: usize = 2_752;
/// Exact byte delta of [`SonaProHotDelta`].
pub const SONA_PRO_HOT_DELTA_BYTES: usize = 3_584;
/// Exact byte delta of [`SonaUltraHotDelta`].
pub const SONA_ULTRA_HOT_DELTA_BYTES: usize = 3_456;

/// Maximum sequential Matryoshka blocks across every SONA tier (`4`).
pub const NUM_SONA_BLOCKS: usize = 4;
/// Maximum `4`-lane stereo `CoreQuad` blocks across `Ultra` (`4 + 3 + 2 + 1 = 10`).
pub const MAX_CORE_QUADS: usize = 10;
/// Maximum total SONA core lanes (`10 * 4 = 40`).
pub const MAX_TOTAL_SONA_LANES: usize = MAX_CORE_QUADS * 4;
/// Maximum lower-triangular `4x4` inter-block tile pairs (`3 + 10 + 10 + 12 = 35`).
pub const MAX_SONA_TILES: usize = 35;

/// Canonical global `(block_idx, local_quad_idx)` assignment for `CoreQuad` slots `0..10`.
pub const SONA_CORE_QUAD_BLOCK_AND_LOCAL: [(usize, usize); MAX_CORE_QUADS] = [
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

/// `CoreQuad` slots per Matryoshka block `[Block 0..Block 3]` in prefix-nested order.
pub const SONA_BLOCK_QUAD_SLOTS: [[Option<usize>; 4]; NUM_SONA_BLOCKS] = [
    [Some(0), Some(4), Some(6), Some(8)],
    [Some(1), Some(5), Some(9), None],
    [Some(2), Some(7), None, None],
    [Some(3), None, None, None],
];

/// Causal Matryoshka block-major `CoreQuad` evaluation sequence across all `10` Ultra quads.
pub const SONA_ULTRA_QUAD_TRAVERSAL_ORDER: [usize; MAX_CORE_QUADS] = [0, 4, 6, 8, 1, 5, 9, 2, 7, 3];

/// All `35` lower-triangular inter-block edges `(src_quad, dst_quad)` ordered by tier arrival
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
        let (sq, dq) = SONA_INTERBLOCK_TILES[i];
        if sq >= MAX_CORE_QUADS || dq >= MAX_CORE_QUADS {
            return false;
        }
        if SONA_CORE_QUAD_BLOCK_AND_LOCAL[sq].0 >= SONA_CORE_QUAD_BLOCK_AND_LOCAL[dq].0 {
            return false;
        }
        let mut j = i + 1;
        while j < MAX_SONA_TILES {
            if SONA_INTERBLOCK_TILES[j].0 == sq && SONA_INTERBLOCK_TILES[j].1 == dq {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

const _: () = assert!(verify_interblock_tiles_and_slots());

/// Compact `144 B` (`16 B`-aligned) hot representation of one `4`-lane stereo `CoreQuad`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreQuadHot {
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
    /// Direct Mid input weight `(0.5 * (L + R))` per lane (`fp16`).
    pub in_mid_weight_fp16: [u16; 4],
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

impl Default for CoreQuadHot {
    fn default() -> Self {
        Self {
            svf_g: [0.05; 4],
            lru_alpha_rel: [0.0; 4],
            lru_alpha_atk: [0.0; 4],
            svf_k_fp16: [f32_to_fp16(core::f32::consts::FRAC_1_SQRT_2); 4],
            lru_kappa_fp16: [0; 4],
            adaa_drive_fp16: [0; 4],
            lru_bias_drive_fp16: [0; 4],
            in_mid_weight_fp16: [0; 4],
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

/// Compact `48 B` (`8 B`-aligned) cold extension per `CoreQuad` holding Selective-Duhem,
/// stereo-link, Side M/S input coupling, Block 0 PHS `svf_k_nl_fp16`, and `GLU` mix weights
/// hoisted into [`CoreQuadParamsStereo4`] and [`CoreQuadReadoutRegs`].
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoreQuadExtCold {
    /// Direct Side input coupling weights `(0.5 * (L - R))` per lane (`[-16.0, 16.0]`, `fp16`).
    pub in_side_weight_fp16: [u16; 4],
    /// `ParaLRU` inter-channel stereo link (`[0.0, 1.0]`, `fp16`).
    pub lru_stereo_link_fp16: [u16; 4],
    /// Selective-Duhem magnetic hysteresis derivative gate gain (`[0.0, 64.0]`, `fp16`).
    pub lru_duhem_gamma_fp16: [u16; 4],
    /// Selective-Duhem signed vs rectified flux blend (`[0.0, 1.0]`, `fp16`).
    pub lru_signed_mix_fp16: [u16; 4],
    /// Bounded cross-lane `GLU` bilinear coupling (`[-4.0, 4.0]`, `fp16`).
    pub glu_mix_fp16: [u16; 4],
    /// Block 0 linearly-implicit PHS nonlinear SVF damping (`[0.0, 16.0]`, `fp16`; `0x0000` on Blocks 1..3).
    pub svf_k_nl_fp16: [u16; 4],
}

impl CoreQuadExtCold {
    /// Materializes `in_side_weight` `[f32; 4]` directly into a SIMD register.
    #[inline(always)]
    #[must_use]
    pub fn in_side_weight_vec4<B: MathBackend>(&self) -> B::Vec4 {
        B::fp16x4_to_vec4(&self.in_side_weight_fp16)
    }
}

/// Per-quad `3 kHz` `ControlBus` modulation vectors and rectifier sign mask consumed by
/// [`CoreQuadHot::to_kernel_params_modulated`].
#[derive(Clone, Copy, Debug)]
pub struct CoreQuadModulation<V> {
    /// Per-lane `SVF` cutoff multiplier.
    pub g_scale: V,
    /// Per-lane `SVF` damping offset.
    pub k_delta: V,
    /// Quad-level `PolyADAA1` drive multiplier.
    pub adaa_scale: f32,
    /// Quad-level signed `ParaLRU` saturation bias offset.
    pub bias_delta: f32,
    /// Per-lane wavefolder mix offset.
    pub wf_delta: V,
    /// Low 4 bits encoding per-lane rectifier sign (`bit l = 1` selects `-1.0`, else `+1.0`).
    pub rect_mask4: u8,
}

impl CoreQuadHot {
    /// Materializes `in_mid_weight` `[f32; 4]` directly into a SIMD register.
    #[inline(always)]
    #[must_use]
    pub fn in_mid_weight_vec4<B: MathBackend>(&self) -> B::Vec4 {
        B::fp16x4_to_vec4(&self.in_mid_weight_fp16)
    }

    /// Materializes [`CoreQuadParamsStereo4`] in SIMD registers with per-lane `3 kHz` targets
    /// combined with cold Selective-Duhem, stereo-link, and `GLU` extension parameters.
    #[inline(always)]
    #[must_use]
    pub fn to_kernel_params_modulated<B: MathBackend>(
        &self,
        ext: &CoreQuadExtCold,
        modulation: CoreQuadModulation<B::Vec4>,
    ) -> CoreQuadParamsStereo4<B::Vec4> {
        let g = B::Vec4::load(&self.svf_g)
            .mul(modulation.g_scale)
            .clamp(B::Vec4::splat(1.0e-5), B::Vec4::splat(64.0));
        let k = B::fp16x4_to_vec4(&self.svf_k_fp16)
            .add(modulation.k_delta)
            .clamp(B::Vec4::zero(), B::Vec4::splat(8.0));
        let sign_arr = [
            if (modulation.rect_mask4 & 1) != 0 {
                -1.0_f32
            } else {
                1.0
            },
            if (modulation.rect_mask4 & 2) != 0 {
                -1.0_f32
            } else {
                1.0
            },
            if (modulation.rect_mask4 & 4) != 0 {
                -1.0_f32
            } else {
                1.0
            },
            if (modulation.rect_mask4 & 8) != 0 {
                -1.0_f32
            } else {
                1.0
            },
        ];
        let rect_sign = B::Vec4::load(&sign_arr);
        CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: B::fp16x4_to_vec4(&self.tone_lp_fp16),
                bp: B::fp16x4_to_vec4(&self.tone_bp_fp16),
                hp: B::fp16x4_to_vec4(&self.tone_hp_fp16),
                ap: B::fp16x4_to_vec4(&self.tone_ap_fp16),
                dir: B::fp16x4_to_vec4(&self.tone_dir_fp16),
            },
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                B::Vec4::load(&self.lru_alpha_rel),
                B::Vec4::load(&self.lru_alpha_atk),
                B::fp16x4_to_vec4(&self.lru_kappa_fp16),
                B::fp16x4_to_vec4(&ext.lru_duhem_gamma_fp16),
                B::fp16x4_to_vec4(&ext.lru_signed_mix_fp16),
                B::fp16x4_to_vec4(&ext.lru_stereo_link_fp16),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: B::fp16x4_to_vec4(&ext.svf_k_nl_fp16)
                .clamp(B::Vec4::zero(), B::Vec4::splat(MAX_SVF_K_NL)),
            rect_sign,
            sag_depth: B::fp16x4_to_vec4(&self.sag_depth_fp16),
            adaa_drive: B::fp16x4_to_vec4(&self.adaa_drive_fp16)
                .mul(B::Vec4::splat(modulation.adaa_scale)),
            lru_bias_drive: B::fp16x4_to_vec4(&self.lru_bias_drive_fp16)
                .add(rect_sign.mul(B::Vec4::splat(modulation.bias_delta)))
                .clamp(B::Vec4::splat(-16.0), B::Vec4::splat(16.0)),
            wavefolder_mix: B::fp16x4_to_vec4(&self.wavefolder_mix_fp16)
                .add(modulation.wf_delta)
                .clamp(B::Vec4::zero(), B::Vec4::splat(1.0)),
            glu_mix: B::fp16x4_to_vec4(&ext.glu_mix_fp16)
                .clamp(B::Vec4::splat(-4.0), B::Vec4::splat(4.0)),
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

/// Quad-major `96 B` (`8 B`-aligned) `LinearReadout` slice (`6` stereo pairs × `4` lanes).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoreQuadReadoutHot {
    pub pre_conv_odd_l_fp16: [u16; 4],
    pub pre_conv_odd_r_fp16: [u16; 4],
    pub pre_conv_even_l_fp16: [u16; 4],
    pub pre_conv_even_r_fp16: [u16; 4],
    pub pre_conv_lru_l_fp16: [u16; 4],
    pub pre_conv_lru_r_fp16: [u16; 4],
    pub direct_odd_l_fp16: [u16; 4],
    pub direct_odd_r_fp16: [u16; 4],
    pub direct_even_l_fp16: [u16; 4],
    pub direct_even_r_fp16: [u16; 4],
    pub direct_lru_l_fp16: [u16; 4],
    pub direct_lru_r_fp16: [u16; 4],
}

/// Register-resident unpacked stereo readout & crossbar weights for one `CoreQuad`.
#[derive(Clone, Copy, Debug)]
pub struct CoreQuadReadoutRegs<V: SimdVec4, S: StereoVec8<Half = V>> {
    pub in_mid_weight: V,
    pub in_side_weight: V,
    pub op_p_w: V,
    pub ph_w: V,
    pub pre_conv_odd: S,
    pub pre_conv_even: S,
    pub pre_conv_lru: S,
    pub direct_odd: S,
    pub direct_even: S,
    pub direct_lru: S,
}

impl CoreQuadReadoutHot {
    /// Decodes one `CoreQuad`'s `96 B` `fp16` weights and crossbar vectors into SIMD registers.
    #[inline(always)]
    #[must_use]
    pub fn decode_regs<B: GraphBackend>(
        &self,
        in_mid_weight: B::Vec4,
        in_side_weight: B::Vec4,
        op_p_w: B::Vec4,
        ph_w: B::Vec4,
    ) -> CoreQuadReadoutRegs<B::Vec4, B::Stereo8> {
        let pair = |l: &[u16; 4], r: &[u16; 4]| {
            B::Stereo8::from_lr(B::fp16x4_to_vec4(l), B::fp16x4_to_vec4(r))
        };
        CoreQuadReadoutRegs {
            in_mid_weight,
            in_side_weight,
            op_p_w,
            ph_w,
            pre_conv_odd: pair(&self.pre_conv_odd_l_fp16, &self.pre_conv_odd_r_fp16),
            pre_conv_even: pair(&self.pre_conv_even_l_fp16, &self.pre_conv_even_r_fp16),
            pre_conv_lru: pair(&self.pre_conv_lru_l_fp16, &self.pre_conv_lru_r_fp16),
            direct_odd: pair(&self.direct_odd_l_fp16, &self.direct_odd_r_fp16),
            direct_even: pair(&self.direct_even_l_fp16, &self.direct_even_r_fp16),
            direct_lru: pair(&self.direct_lru_l_fp16, &self.direct_lru_r_fp16),
        }
    }
}

/// Top-of-Frame crossbar outputs computed once per core sub-sample ahead of quad evaluation.
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
    pub core_quads: [CoreQuadHot; 4],
    pub tiles: [SonaInterBlockTileHot; 3],
    pub readouts: [CoreQuadReadoutHot; 4],
    pub kautz: [KautzStereoBank4; 2],
    pub sensory_quad: SensoryQuadHot,
    pub transport_quad0: TransportQuadHotCompact,
    pub transport0_ring: [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP],
    pub fir_weights_l: [[[f32; 4]; 4]; 2],
    pub fir_weights_r: [[[f32; 4]; 4]; 2],
    pub fir_hist_l: [f32; 32],
    pub fir_hist_r: [f32; 32],
    pub hilbert: HilbertH1StereoKernel,
    pub t1: T1SubHarmonicKernel,
    pub phasors: PhasorQ32Bank4,
    pub t1_dc: T1DcBlockerStereo,
    pub op_n_pre: OpNPreHotState,
    pub hp_lp_pole: f32,
    pub hp_lp_state: [f32; 2],
    pub sample_tick: u32,
    pub ro_gains_fp16: [u16; 8],
    pub carrier_basis_w_fp16: [[u16; 4]; 3],
    pub op_p_to_core_lite_fp16: [[u16; 4]; 4],
    pub phasor_to_core_lite_fp16: [[u16; 4]; 4],
    pub opt_to_lr_lite_fp16: [[u16; 2]; 4],
    pub phasor_st_cos_fp16: [u16; 4],
    pub phasor_st_sin_fp16: [u16; 4],
    pub tank_dry_mix_eff_fp16: u16,
    pub _reserved_hot_header_u16: [u16; 9],
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
            core_quads: [CoreQuadHot::default(); 4],
            tiles: [SonaInterBlockTileHot::default(); 3],
            readouts: [CoreQuadReadoutHot::default(); 4],
            kautz: [KautzStereoBank4::default(); 2],
            sensory_quad: SensoryQuadHot::zero(),
            transport_quad0: TransportQuadHotCompact::default(),
            transport0_ring: [[0; 4]; TRANSPORT_INLINE_QUAD0_CAP],
            fir_weights_l: [[[0.0; 4]; 4]; 2],
            fir_weights_r: [[[0.0; 4]; 4]; 2],
            fir_hist_l: [0.0; 32],
            fir_hist_r: [0.0; 32],
            hilbert: HilbertH1StereoKernel::with_coeffs([0.0; 6], [0.0; 6]),
            t1: T1SubHarmonicKernel::new(),
            phasors: PhasorQ32Bank4::new([0; 4], [0; 4]),
            t1_dc: T1DcBlockerStereo::default(),
            op_n_pre: OpNPreHotState::default(),
            hp_lp_pole: 0.0,
            hp_lp_state: [0.0; 2],
            sample_tick: 0,
            ro_gains_fp16: [0; 8],
            carrier_basis_w_fp16: [[0; 4]; 3],
            op_p_to_core_lite_fp16: [[0; 4]; 4],
            phasor_to_core_lite_fp16: [[0; 4]; 4],
            opt_to_lr_lite_fp16: [[0; 2]; 4],
            phasor_st_cos_fp16: [f32_to_fp16(1.0); 4],
            phasor_st_sin_fp16: [0; 4],
            tank_dry_mix_eff_fp16: 0,
            _reserved_hot_header_u16: [0; 9],
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
    pub core_quads_full: [CoreQuadHot; 2],
    pub tiles_full: [SonaInterBlockTileHot; 10],
    pub readouts_full: [CoreQuadReadoutHot; 2],
    pub fir_weights_l_full: [[[f32; 4]; 4]; 1],
    pub fir_weights_r_full: [[[f32; 4]; 4]; 1],
    pub fir_hist_l_full: [f32; 16],
    pub fir_hist_r_full: [f32; 16],
    pub kautz_full: [KautzStereoBank4; 1],
    pub transport_quad1: TransportQuadHotCompact,
    pub op_p_to_core_full_fp16: [[u16; 4]; 2],
    pub phasor_to_core_full_fp16: [[u16; 4]; 2],
    pub opt_to_lr_full_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_full: u8,
}

impl<B: GraphBackend> SonaFullHotDelta<B> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states_full: [SonaBlockStateStereo8::zero(); 2],
            core_quads_full: [CoreQuadHot::default(); 2],
            tiles_full: [SonaInterBlockTileHot::default(); 10],
            readouts_full: [CoreQuadReadoutHot::default(); 2],
            fir_weights_l_full: [[[0.0; 4]; 4]; 1],
            fir_weights_r_full: [[[0.0; 4]; 4]; 1],
            fir_hist_l_full: [0.0; 16],
            fir_hist_r_full: [0.0; 16],
            kautz_full: [KautzStereoBank4::default(); 1],
            transport_quad1: TransportQuadHotCompact::default(),
            op_p_to_core_full_fp16: [[0; 4]; 2],
            phasor_to_core_full_fp16: [[0; 4]; 2],
            opt_to_lr_full_fp16: [[0; 2]; 4],
            rect_sign_mask_full: 0,
        }
    }
}

/// `Tier::Pro` hot delta (`3,584 B` = `56 CL`; cumulative `11,136 B`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaProHotDelta<B: GraphBackend> {
    pub core_states_pro: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub core_quads_pro: [CoreQuadHot; 2],
    pub tiles_pro: [SonaInterBlockTileHot; 10],
    pub readouts_pro: [CoreQuadReadoutHot; 2],
    pub fir_weights_l_pro: [[[f32; 4]; 4]; 3],
    pub fir_weights_r_pro: [[[f32; 4]; 4]; 3],
    pub fir_hist_l_pro: [f32; 48],
    pub fir_hist_r_pro: [f32; 48],
    pub kautz_pro: [KautzStereoBank4; 3],
    pub transport_quad2: TransportQuadHotCompact,
    pub op_p_to_core_pro_fp16: [[u16; 4]; 2],
    pub phasor_to_core_pro_fp16: [[u16; 4]; 2],
    pub opt_to_lr_pro_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_pro: u8,
}

impl<B: GraphBackend> SonaProHotDelta<B> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states_pro: [SonaBlockStateStereo8::zero(); 2],
            core_quads_pro: [CoreQuadHot::default(); 2],
            tiles_pro: [SonaInterBlockTileHot::default(); 10],
            readouts_pro: [CoreQuadReadoutHot::default(); 2],
            fir_weights_l_pro: [[[0.0; 4]; 4]; 3],
            fir_weights_r_pro: [[[0.0; 4]; 4]; 3],
            fir_hist_l_pro: [0.0; 48],
            fir_hist_r_pro: [0.0; 48],
            kautz_pro: [KautzStereoBank4::default(); 3],
            transport_quad2: TransportQuadHotCompact::default(),
            op_p_to_core_pro_fp16: [[0; 4]; 2],
            phasor_to_core_pro_fp16: [[0; 4]; 2],
            opt_to_lr_pro_fp16: [[0; 2]; 4],
            rect_sign_mask_pro: 0,
        }
    }
}

/// `Tier::Ultra` hot delta (`3,456 B` = `54 CL`; cumulative `14,592 B`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct SonaUltraHotDelta<B: GraphBackend> {
    pub core_states_ultra: [SonaBlockStateStereo8<B::Stereo8>; 2],
    pub core_quads_ultra: [CoreQuadHot; 2],
    pub tiles_ultra: [SonaInterBlockTileHot; 12],
    pub readouts_ultra: [CoreQuadReadoutHot; 2],
    pub fir_weights_l_ultra: [[[f32; 4]; 4]; 2],
    pub fir_weights_r_ultra: [[[f32; 4]; 4]; 2],
    pub fir_hist_l_ultra: [f32; 32],
    pub fir_hist_r_ultra: [f32; 32],
    pub kautz_ultra: [KautzStereoBank4; 2],
    pub transport_quad3: TransportQuadHotCompact,
    pub op_p_to_core_ultra_fp16: [[u16; 4]; 2],
    pub phasor_to_core_ultra_fp16: [[u16; 4]; 2],
    pub opt_to_lr_ultra_fp16: [[u16; 2]; 4],
    pub rect_sign_mask_ultra: u8,
}

impl<B: GraphBackend> SonaUltraHotDelta<B> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_states_ultra: [SonaBlockStateStereo8::zero(); 2],
            core_quads_ultra: [CoreQuadHot::default(); 2],
            tiles_ultra: [SonaInterBlockTileHot::default(); 12],
            readouts_ultra: [CoreQuadReadoutHot::default(); 2],
            fir_weights_l_ultra: [[[0.0; 4]; 4]; 2],
            fir_weights_r_ultra: [[[0.0; 4]; 4]; 2],
            fir_hist_l_ultra: [0.0; 32],
            fir_hist_r_ultra: [0.0; 32],
            kautz_ultra: [KautzStereoBank4::default(); 2],
            transport_quad3: TransportQuadHotCompact::default(),
            op_p_to_core_ultra_fp16: [[0; 4]; 2],
            phasor_to_core_ultra_fp16: [[0; 4]; 2],
            opt_to_lr_ultra_fp16: [[0; 2]; 4],
            rect_sign_mask_ultra: 0,
        }
    }
}

const _: () = assert!(size_of::<CoreQuadHot>() == 144 && align_of::<CoreQuadHot>() == 16);
const _: () = assert!(size_of::<CoreQuadExtCold>() == 48 && align_of::<CoreQuadExtCold>() == 8);
const _: () =
    assert!(size_of::<SonaInterBlockTileHot>() == 128 && align_of::<SonaInterBlockTileHot>() == 16);
const _: () =
    assert!(size_of::<CoreQuadReadoutHot>() == 96 && align_of::<CoreQuadReadoutHot>() == 8);
const _: () =
    assert!(size_of::<SonaOs2xHot>() == SONA_OS2X_HOT_BYTES && align_of::<SonaOs2xHot>() == 64);

const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, sensory_quad) == 2_208);
const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, transport_quad0) == 2_256);
const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, tank_dry_mix_eff_fp16) == 4_520);
const _: () =
    assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, _reserved_hot_header_u16) == 4_522);
const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, rect_sign_mask_lite) == 4_540);
const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, tier) == 4_542);
const _: () = assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, flags) == 4_543);
const _: () =
    assert!(offset_of!(SonaLiteHotPrefix<ScalarBackend>, flags) + 1 == SONA_LITE_HOT_BYTES);

const _: () = assert!(
    size_of::<SonaLiteHotPrefix<ScalarBackend>>() == SONA_LITE_HOT_BYTES
        && align_of::<SonaLiteHotPrefix<ScalarBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaFullHotDelta<ScalarBackend>>() == SONA_FULL_HOT_DELTA_BYTES
        && align_of::<SonaFullHotDelta<ScalarBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaProHotDelta<ScalarBackend>>() == SONA_PRO_HOT_DELTA_BYTES
        && align_of::<SonaProHotDelta<ScalarBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaUltraHotDelta<ScalarBackend>>() == SONA_ULTRA_HOT_DELTA_BYTES
        && align_of::<SonaUltraHotDelta<ScalarBackend>>() == 64
);

const _: () =
    assert!(offset_of!(SonaLiteHotPrefix<DefaultMathBackend>, flags) + 1 == SONA_LITE_HOT_BYTES);
const _: () = assert!(
    size_of::<SonaLiteHotPrefix<DefaultMathBackend>>() == SONA_LITE_HOT_BYTES
        && align_of::<SonaLiteHotPrefix<DefaultMathBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaFullHotDelta<DefaultMathBackend>>() == SONA_FULL_HOT_DELTA_BYTES
        && align_of::<SonaFullHotDelta<DefaultMathBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaProHotDelta<DefaultMathBackend>>() == SONA_PRO_HOT_DELTA_BYTES
        && align_of::<SonaProHotDelta<DefaultMathBackend>>() == 64
);
const _: () = assert!(
    size_of::<SonaUltraHotDelta<DefaultMathBackend>>() == SONA_ULTRA_HOT_DELTA_BYTES
        && align_of::<SonaUltraHotDelta<DefaultMathBackend>>() == 64
);

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[dsp.head.direct-register-readout]
    /// verify[dsp.head.lower-triangular]
    /// verify[dsp.op-n.core.interblock-tiles]
    #[test]
    fn test_layout_prefix_topology_tile_count_and_duhem_params() {
        assert_eq!(core::mem::size_of::<CoreQuadReadoutHot>(), 96);
        assert_eq!(
            SONA_ULTRA_QUAD_TRAVERSAL_ORDER,
            [0, 4, 6, 8, 1, 5, 9, 2, 7, 3]
        );
        assert_eq!(SONA_INTERBLOCK_TILES.len(), 35);
        assert_eq!(
            SONA_BLOCK_QUAD_SLOTS[0],
            [Some(0), Some(4), Some(6), Some(8)]
        );
        let lit = SonaLiteHotPrefix::<ScalarBackend>::empty();
        assert_eq!(lit.tank_dry_mix_eff_fp16, 0);
        assert_eq!(lit.sensory_quad, SensoryQuadHot::zero());

        let mut q = CoreQuadHot::default();
        for l in 0..4 {
            q.svf_g[l] = 0.25;
            q.lru_bias_drive_fp16[l] = f32_to_fp16(if l < 2 { -15.0 } else { 15.0 });
        }
        let ext = CoreQuadExtCold {
            svf_k_nl_fp16: [0; 4],
            in_side_weight_fp16: [f32_to_fp16(0.5); 4],
            lru_stereo_link_fp16: [f32_to_fp16(0.25); 4],
            lru_duhem_gamma_fp16: [f32_to_fp16(12.0); 4],
            lru_signed_mix_fp16: [f32_to_fp16(0.75); 4],
            glu_mix_fp16: [f32_to_fp16(-1.5); 4],
        };
        let kp = q.to_kernel_params_modulated::<ScalarBackend>(
            &ext,
            CoreQuadModulation {
                g_scale: crate::math::scalar::ScalarVec4([1.0; 4]),
                k_delta: crate::math::scalar::ScalarVec4([0.0; 4]),
                adaa_scale: 1.0,
                bias_delta: 2.0,
                wf_delta: crate::math::scalar::ScalarVec4([0.0; 4]),
                rect_mask4: 0b1010,
            },
        );
        assert_eq!(kp.lru_bias_drive.to_array(), [-13.0, -16.0, 16.0, 13.0]);
        assert_eq!(kp.two_svf_g.to_array(), [0.5; 4]);
        assert_eq!(kp.glu_mix.to_array(), [-1.5; 4]);
        assert_eq!(kp.lru.gamma_duhem.to_array(), [12.0; 4]);
        assert_eq!(kp.lru.signed_mix.to_array(), [0.75; 4]);
        assert_eq!(kp.lru.stereo_link.to_array(), [0.25; 4]);
        assert_eq!(
            ext.in_side_weight_vec4::<ScalarBackend>().to_array(),
            [0.5; 4]
        );
    }
}
