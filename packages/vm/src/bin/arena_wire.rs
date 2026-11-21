//! Declarative `#[repr(C)]` `zerocopy` `SONA v3.0` Arena 0..3 tensor and descriptor blocks
//! (`4,608 / 7,936 / 15,872 / 30,720 B` cumulative tier-prefix payloads, `ref[container.dnam.tier-prefix]`,
//! `ref[container.dnam.tensor-directory]`).

use core::mem::{align_of, offset_of, size_of};

use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::header::Cp1InlineDirWire;

/// Relative `tier_start` offsets for `ModalConvolver` Early FIR slices across `Lite..Ultra`.
pub const CONVOLVER_EARLY_FIR_BASES: [usize; 4] = [1_408, 6_528, 12_544, 22_528];
/// Relative `tier_start` offsets for `ModalConvolver` Kautz SoA blocks across `Lite..Ultra`.
pub const CONVOLVER_KAUTZ_BASES: [usize; 4] = [1_664, 6_656, 12_928, 22_784];
/// Number of `128 B` Kautz SoA blocks introduced in each tier (`Lite..Ultra`).
pub const CONVOLVER_KAUTZ_BLOCKS_PER_TIER: [usize; 4] = [2, 1, 3, 2];
/// Relative `tier_start` offsets for `TransportQuad` (`Op-T`) blocks across `Lite..Ultra`.
pub const TRANSPORT_DELAY_BASES: [usize; 4] = [1_920, 6_784, 13_312, 23_040];
/// Relative `tier_start` offsets for `LinearReadout` blocks across `Lite..Ultra`.
pub const LINEAR_READOUT_BASES: [usize; 4] = [2_048, 6_912, 13_568, 23_296];

/// Required upper halfword magic in `CrossbarWire::pipeline_flags`.
pub const PIPE_MAGIC_HI: u32 = 0x444E_0000;
/// Bitmask for `PIPE_MAGIC_HI`.
pub const PIPE_MAGIC_MASK: u32 = 0xFFFF_0000;
/// `pipeline_flags` bit 3: informational convolver extension flag.
pub const PIPE_FLAG_CONVOLVER_EXT: u32 = 1 << 3;
/// `pipeline_flags` bit 4: informational transport extension flag.
pub const PIPE_FLAG_TRANSPORT_EXT: u32 = 1 << 4;
/// `pipeline_flags` bit 5: unified `W_ctrl` + `"CEXT"` dispatch semantics.
pub const PIPE_FLAG_UNIFIED_W_CTRL: u32 = 1 << 5;
/// `pipeline_flags` bit 6: v3 Matryoshka nested core layout.
pub const PIPE_FLAG_V3_MATRYOSHKA: u32 = 1 << 6;
/// `pipeline_flags` bit 7: SONA engine active.
pub const PIPE_FLAG_SONA: u32 = 1 << 7;
/// Required non-magic flags mask `(0x00E0)` in `CrossbarWire::pipeline_flags`.
pub const PIPE_REQUIRED_SONA_MASK: u32 =
    PIPE_FLAG_UNIFIED_W_CTRL | PIPE_FLAG_V3_MATRYOSHKA | PIPE_FLAG_SONA;
/// Canonical v3.0 `SONA` `pipeline_flags` constant (`0x444E_00F8`).
pub const PIPE_CANONICAL_V3_SONA: u32 =
    PIPE_MAGIC_HI | PIPE_REQUIRED_SONA_MASK | PIPE_FLAG_CONVOLVER_EXT | PIPE_FLAG_TRANSPORT_EXT;

/// `sona_meta_flags` bit 4: summary core 2× oversampling flag.
pub const SONA_META_OS2X: u32 = 1 << 4;
/// `sona_meta_flags` bit 6: bits `0..=3` authoritatively specify per-tier `tier_os2x`.
pub const SONA_META_TIER_OS_EXPLICIT: u32 = 1 << 6;
/// `sona_meta_flags` bit 7: higher tier applies `ModalConvolver` Prefix Override.
pub const SONA_META_CONVOLVER_PREFIX_OVERRIDES: u32 = 1 << 7;
/// `sona_meta_flags` bit 8: decode `os_and_front_ext` (`128..192`) verbatim.
pub const SONA_META_OS_EXT_PRESENT: u32 = 1 << 8;
/// `sona_meta_flags` bit 9: Ultra `stage_latent_core_mods_fp16` active.
pub const SONA_META_CORE_MODS_PRESENT: u32 = 1 << 9;
/// `sona_meta_flags` bit 16: transport tap gains enabled (`1` canonical in v3.0).
pub const SONA_META_TRANSPORT_TAP_GAINS: u32 = 1 << 16;
/// `sona_meta_flags` bit 17: `SensoryQuad` (`Op-M/N[Sense]`) enabled (`1` canonical in v3.0).
pub const SONA_META_SENSORY_QUAD: u32 = 1 << 17;
/// `sona_meta_flags` bit 20: transport LFO enabled (`1` canonical in v3.0).
pub const SONA_META_TRANSPORT_LFO: u32 = 1 << 20;
/// Reserved `sona_meta_flags` / `cell_flags` bits (`5, 10..=15, 18..=19, 21..=31`).
pub const SONA_META_RESERVED_MASK: u32 = (1 << 5) | 0xFC00 | (0b11 << 18) | 0xFFE0_0000;

/// `TransportQuadWire::flags` bit 0: Schroeder diffuser stage enabled.
pub const TRANSPORT_FLAG_SCHROEDER_ENABLE: u8 = 1 << 0;
/// `TransportQuadWire::flags` bit 1: external ring uses `f32` (`1`) instead of `fp16` (`0`).
pub const TRANSPORT_FLAG_EXT_F32: u8 = 1 << 1;

/// `Op-N[Pre]` base front-end record (`12 B` at Arena 0 offset `180..192`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct OpNPreBaseWire {
    /// Wet bitcrush / blend depth (`[0.0, 1.0]`).
    pub gamma_bc: F32,
    /// `1.0 - rho_zoh` (`[0.0, 0.9999]`; `0.0` = ZOH bypass).
    pub zoh_hold_depth: F32,
    /// Log2 quantizer step depth (`[0.0, 16.0]`; `0.0` = quantizer bypass).
    pub beta_q: F32,
}

const _: () = assert!(size_of::<OpNPreBaseWire>() == 12);
const _: () = assert!(align_of::<OpNPreBaseWire>() == 1);

/// Embedded `SONA` Cell Descriptor (`16 B` at Arena 0 offset `192..208`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaCellDescWire {
    /// Magic tag: `"SONA"`.
    pub magic: [u8; 4],
    /// Major cell descriptor version (`3`).
    pub version_major: U16,
    /// Minor cell descriptor version (`0`).
    pub version_minor: U16,
    /// Original tier bitmask (`0x01=Lite`, `0x03=Full`, `0x07=Pro`, `0x0F=Ultra`).
    pub tier_flags: U16,
    /// Oversampling factor (`1` or `2`).
    pub oversample_factor: U16,
    /// Mirror of `CrossbarWire::sona_meta_flags`.
    pub cell_flags: U32,
}

const _: () = assert!(size_of::<SonaCellDescWire>() == 16);
const _: () = assert!(align_of::<SonaCellDescWire>() == 1);

/// Crossbar & `SensoryQuad` Block (`256 B` at Arena 0 offset `0..256`, `ref[container.dnam.tensor-directory]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CrossbarWire {
    /// `SensoryQuad` `48 kHz` `TPT-SVF` cutoff tangent gains `[1e-5, 64.0]`.
    pub sense_svf_g: [F32; 4],
    /// `SensoryQuad` `48 kHz` `ParaLRU` release poles `[0.0, MAX_PARALRU_R_F32]`.
    pub sense_lru_alpha_rel: [F32; 4],
    /// `SensoryQuad` `48 kHz` `ParaLRU` attack poles `[0.0, 1.0]`.
    pub sense_lru_alpha_atk: [F32; 4],
    /// `SensoryQuad` `TPT-SVF` damping factors `[0.0, 8.0]` (`fp16`).
    pub sense_svf_k_fp16: [U16; 4],
    /// `SensoryQuad` `ParaLRU` feedback gains `[0.0, 1.0]` (`fp16`).
    pub sense_lru_kappa_fp16: [U16; 4],
    /// Pipeline version & feature flags (`PIPE_CANONICAL_V3_SONA = 0x444E_00F8`).
    pub pipeline_flags: U32,
    /// Per-tier output loudness alignment trim `[Lite, Full, Pro, Ultra]`.
    pub tier_gain_trim: [F32; 4],
    /// SONA feature flag word mirrored in `cell_desc.cell_flags`.
    pub sona_meta_flags: U32,
    /// `SensoryQuad` Left stereo key weights `[-4.0, 4.0]` (`fp16`).
    pub sense_in_l_fp16: [U16; 4],
    /// `SensoryQuad` Right stereo key weights `[-4.0, 4.0]` (`fp16`).
    pub sense_in_r_fp16: [U16; 4],
    /// `SensoryQuad` `TPT-SVF` low-pass sensing weights `[-16.0, 16.0]` (`fp16`).
    pub sense_tone_lp_fp16: [U16; 4],
    /// `SensoryQuad` `TPT-SVF` band-pass sensing weights `[-16.0, 16.0]` (`fp16`).
    pub sense_tone_bp_fp16: [U16; 4],
    /// Crossbar gains `[op_p_crossbar_gain, op_t_crossbar_gain]` in `[-4.0, 4.0]` (`fp16`).
    pub crossbar_op_gains_fp16: [U16; 2],
    /// 48 kHz anti-zipper micro-ramp step fraction (`0.0625` default when `<= 1e-7`).
    pub anti_zipper_ramp_scale: F32,
    /// Oversampler upsample quadratic coefficient (`-0.125` default).
    pub os_up_quad_coeff: F32,
    /// Oversampler decimator `x[n-1]` weight (`0.15` default).
    pub os_dec_c_prev1: F32,
    /// Oversampler decimator `x[n]` weight (`0.50` default).
    pub os_dec_c0: F32,
    /// Oversampler pre-emphasis scalar (`0.0` default).
    pub os_up_pre_emph: F32,
    /// Oversampler decimator AA mix (`0.0` default).
    pub os_dec_aa_mix: F32,
    /// Oversampler upsampler AA mix (`0.0` default).
    pub os_up_aa_mix: F32,
    /// Oversampler ZOH mix (`0.0` default).
    pub os_up_zoh_mix: F32,
    /// 3-basis carrier projection `[hi*cos, hq*sin, sin]` in IEEE-754 `fp16` (`24 B`).
    pub carrier_basis_w_fp16: [[U16; 4]; 3],
    /// `Op-N[Pre]` base front-end parameters (`12 B`).
    pub op_n_pre_base: OpNPreBaseWire,
    /// Embedded `"SONA"` cell descriptor (`16 B`).
    pub cell_desc: SonaCellDescWire,
    /// Top-of-frame `Q0.32` phasor carrier frequencies in Hz (`[4]`).
    pub phasor_freq_hz: [F32; 4],
    /// Initial `Q0.32` carrier phases (`[4]`).
    pub phasor_init_phase_q32: [U32; 4],
    /// `SensoryQuad` `TPT-SVF` high-pass sensing weights `[-16.0, 16.0]` (`fp16`).
    pub sense_tone_hp_fp16: [U16; 4],
    /// `SensoryQuad` input VCA weights `[-512.0, 512.0]` (`fp16`).
    pub sense_vca_w_fp16: [U16; 4],
}

const _: () = assert!(size_of::<CrossbarWire>() == 256);
const _: () = assert!(align_of::<CrossbarWire>() == 1);
const _: () = assert!(offset_of!(CrossbarWire, sense_svf_k_fp16) == 48);
const _: () = assert!(offset_of!(CrossbarWire, pipeline_flags) == 64);
const _: () = assert!(offset_of!(CrossbarWire, sona_meta_flags) == 84);
const _: () = assert!(offset_of!(CrossbarWire, sense_in_l_fp16) == 88);
const _: () = assert!(offset_of!(CrossbarWire, crossbar_op_gains_fp16) == 120);
const _: () = assert!(offset_of!(CrossbarWire, anti_zipper_ramp_scale) == 124);
const _: () = assert!(offset_of!(CrossbarWire, os_up_quad_coeff) == 128);
const _: () = assert!(offset_of!(CrossbarWire, carrier_basis_w_fp16) == 156);
const _: () = assert!(offset_of!(CrossbarWire, op_n_pre_base) == 180);
const _: () = assert!(offset_of!(CrossbarWire, cell_desc) == 192);
const _: () = assert!(offset_of!(CrossbarWire, sense_tone_hp_fp16) == 240);
const _: () = assert!(offset_of!(CrossbarWire, sense_vca_w_fp16) == 248);

/// 4-lane `CoreQuad` Block (`192 B`, `ref[container.dnam.tensor-directory]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CoreQuadWire {
    /// TPT-SVF `g` coefficients (`48 kHz`, `[1e-5, 64.0]`).
    pub svf_g: [F32; 4],
    /// ParaLRU release decay poles (`[0.0, MAX_PARALRU_R_F32]`).
    pub lru_alpha_rel: [F32; 4],
    /// ParaLRU attack poles (`[0.0, 1.0]`).
    pub lru_alpha_atk: [F32; 4],
    /// TPT-SVF `k` damping factors (`[0.0, 8.0]`, `fp16`).
    pub svf_k_fp16: [U16; 4],
    /// ParaLRU envelope detector gain `kappa` (`[0.0, 1.0]`, `fp16`).
    pub lru_kappa_fp16: [U16; 4],
    /// PolyADAA1 pre-activation drive (`[-32.0, 32.0]`, `fp16`).
    pub adaa_drive_fp16: [U16; 4],
    /// ParaLRU bias drive (`[-16.0, 16.0]`, `fp16`).
    pub lru_bias_drive_fp16: [U16; 4],
    /// SVF low-pass mix weights (`[-16.0, 16.0]`, `fp16`).
    pub tone_lp_fp16: [U16; 4],
    /// SVF band-pass mix weights (`[-16.0, 16.0]`, `fp16`).
    pub tone_bp_fp16: [U16; 4],
    /// SVF high-pass mix weights (`[-16.0, 16.0]`, `fp16`).
    pub tone_hp_fp16: [U16; 4],
    /// SVF all-pass mix weights (`[-16.0, 16.0]`, `fp16`).
    pub tone_ap_fp16: [U16; 4],
    /// Direct dry bypass weights (`[-16.0, 16.0]`, `fp16`).
    pub tone_dir_fp16: [U16; 4],
    /// Sag supply coupling depths (`[-8.0, 8.0]`, `fp16`).
    pub sag_depth_fp16: [U16; 4],
    /// Wavefolder wet mix (`[0.0, 1.0]`, `fp16`).
    pub wavefolder_mix_fp16: [U16; 4],
    /// Mechanism A2 sign parity (`fp16`, decoded to `{-1.0, +1.0}`).
    pub rect_sign_fp16: [U16; 4],
    /// Direct Mid input coupling weights (`[-16.0, 16.0]`, `fp16`).
    pub in_mid_weight_fp16: [U16; 4],
    /// Direct Side input coupling weights (`[-16.0, 16.0]`, `fp16`).
    pub in_side_weight_fp16: [U16; 4],
    /// ParaLRU inter-channel stereo link (`[0.0, 1.0]`, `fp16`).
    pub lru_stereo_link_fp16: [U16; 4],
    /// Selective-Duhem magnetic hysteresis gain (`[0.0, 64.0]`, `fp16`).
    pub lru_duhem_gamma_fp16: [U16; 4],
    /// Selective-Duhem signed flux blend (`[0.0, 1.0]`, `fp16`).
    pub lru_signed_mix_fp16: [U16; 4],
    /// Bounded cross-lane GLU mix (`[-4.0, 4.0]`, `fp16`).
    pub glu_mix_fp16: [U16; 4],
}

const _: () = assert!(size_of::<CoreQuadWire>() == 192);
const _: () = assert!(align_of::<CoreQuadWire>() == 1);
const _: () = assert!(offset_of!(CoreQuadWire, svf_k_fp16) == 48);
const _: () = assert!(offset_of!(CoreQuadWire, in_mid_weight_fp16) == 144);
const _: () = assert!(offset_of!(CoreQuadWire, in_side_weight_fp16) == 152);
const _: () = assert!(offset_of!(CoreQuadWire, lru_stereo_link_fp16) == 160);
const _: () = assert!(offset_of!(CoreQuadWire, lru_duhem_gamma_fp16) == 168);
const _: () = assert!(offset_of!(CoreQuadWire, lru_signed_mix_fp16) == 176);
const _: () = assert!(offset_of!(CoreQuadWire, glu_mix_fp16) == 184);

/// 4×4 Inter-Block Tile Pair (`128 B`, column-major `[src_lane][dst_lane]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaInterBlockTilePairWire {
    /// Linear odd-channel coupling `w_odd_4x4` (`64 B`).
    pub w_odd_4x4: [[F32; 4]; 4],
    /// Nonlinear even-channel coupling `w_even_4x4` in `fp16` (`32 B`).
    pub w_even_4x4_fp16: [[U16; 4]; 4],
    /// ParaLRU fast-path coupling `w_lru_4x4` in `fp16` (`32 B`).
    pub w_lru_4x4_fp16: [[U16; 4]; 4],
}

const _: () = assert!(size_of::<SonaInterBlockTilePairWire>() == 128);
const _: () = assert!(align_of::<SonaInterBlockTilePairWire>() == 1);

/// Full/Pro/Ultra `256 B` SONA Crossbar Coupling Block (`ref[container.dnam.tensor-directory]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaCrossbarBlockWire {
    /// Packed `fp16` weights from `Op-P` / `Op-M` buses to each of `10` `CoreQuad`s (`80 B`).
    pub op_p_to_core_fp16: [[U16; 4]; 10],
    /// Packed `fp16` `Q0.32` phasor drives to each of `10` `CoreQuad`s (`80 B`).
    pub phasor_to_core_fp16: [[U16; 4]; 10],
    /// Packed `fp16` `Op-T` crossbar `16 -> [L, R]` readout matrix (`64 B`, complete replacement).
    pub opt_to_lr_fp16: [[U16; 2]; 16],
    /// Stereo phasor offsets (`[0..4]` in `[-pi, pi]`, `[4..8]` reserved `0`).
    pub phasor_stereo_phase_fp16: [U16; 8],
    /// Reserved coupling slots (must equal `0`).
    pub crossbar_reserved_fp16: [U16; 8],
}

const _: () = assert!(size_of::<SonaCrossbarBlockWire>() == 256);
const _: () = assert!(align_of::<SonaCrossbarBlockWire>() == 1);

/// Full `ModalConvolver` Early FIR slice (`16` taps/ch = `128 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConvolverFir16Wire {
    /// Left channel early FIR taps.
    pub fir_l: [F32; 16],
    /// Right channel early FIR taps.
    pub fir_r: [F32; 16],
}

/// Lite & Ultra `ModalConvolver` Early FIR slice (`32` taps/ch = `256 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConvolverFir32Wire {
    /// Left channel early FIR taps.
    pub fir_l: [F32; 32],
    /// Right channel early FIR taps.
    pub fir_r: [F32; 32],
}

/// Pro `ModalConvolver` Early FIR slice (`48` taps/ch = `384 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConvolverFir48Wire {
    /// Left channel early FIR taps.
    pub fir_l: [F32; 48],
    /// Right channel early FIR taps.
    pub fir_r: [F32; 48],
}

/// 4-pair shared-pole Kautz SoA Block (`128 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct KautzSoaBlockWire {
    /// Lower radial pole anchor.
    pub r_low: [F32; 4],
    /// Radial spread delta (`r_base = clamp(r_low + 0.5 * dr, 0.0, 0.9995)`).
    pub dr: [F32; 4],
    /// Modal carrier frequencies in radians/sample (`48 kHz`).
    pub theta: [F32; 4],
    /// Modal excitation gains `b_gamma`.
    pub b_gamma: [F32; 4],
    /// Left real residue `c_re_l`.
    pub c_re_l: [F32; 4],
    /// Left imag residue `c_im_l`.
    pub c_im_l: [F32; 4],
    /// Right real residue `c_re_r`.
    pub c_re_r: [F32; 4],
    /// Right imag residue `c_im_r`.
    pub c_im_r: [F32; 4],
}

const _: () = assert!(size_of::<ConvolverFir16Wire>() == 128);
const _: () = assert!(size_of::<ConvolverFir32Wire>() == 256);
const _: () = assert!(size_of::<ConvolverFir48Wire>() == 384);
const _: () = assert!(size_of::<KautzSoaBlockWire>() == 128);

/// Unified `128 B` `TransportQuad` (`Op-T` 4-lane block, `ref[dsp.op-t.ring.declaration]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TransportQuadWire {
    /// Nominal delay times in `48 kHz` samples (`[5.0, 48_000.0]`).
    pub delay_samples: [F32; 4],
    /// Time-stretcher ramp window spans in `48 kHz` samples.
    pub ramp_spans_samples: [F32; 4],
    /// Feedback loop attenuation scalar (`[0.0, 0x3F7F_BE76]`).
    pub g_loop: F32,
    /// Loop DC/AC blocker pole (`[0.95, 0.9999]`).
    pub r_dc: F32,
    /// Low-pass loop damping pole (`[0.0, 0.9995]`).
    pub damp_pole: F32,
    /// Delay LFO rate in Hz (`[0.0, 16.0]`).
    pub lfo_rate_hz: F32,
    /// Grain/delay ramp rate in Hz (`[-16.0, 16.0]`).
    pub ramp_rate_hz: F32,
    /// Reserved alignment scalar (`0.0`, ignored on decode).
    pub reserved_52_56: F32,
    /// Per-lane LFO modulation depths in `48 kHz` samples (`fp16`).
    pub lfo_depths_fp16: [U16; 4],
    /// Initial `Q0.16` LFO phases.
    pub lfo_phase_off_q16: [U16; 4],
    /// Initial `Q0.16` ramp phases.
    pub ramp_phase_off_q16: [U16; 4],
    /// Per-lane quad input gains (`fp16`, `[-2.0, 2.0]`).
    pub input_gains_fp16: [U16; 4],
    /// Per-lane Left wet tap weights (`fp16`, `[-2.0, 2.0]`).
    pub wet_tap_l_fp16: [U16; 4],
    /// Per-lane Right wet tap weights (`fp16`, `[-2.0, 2.0]`).
    pub wet_tap_r_fp16: [U16; 4],
    /// Per-lane Schroeder all-pass diffuser coefficients (`fp16`, `[-0.75, 0.75]`).
    pub allpass_g_fp16: [U16; 4],
    /// Givens feedback rotation angle in radians (`fp16`, `[-pi, pi]`).
    pub fb_psi_fp16: U16,
    /// Unified cubic feedback limiter ceiling `sigma_sat` (`fp16`, `[1e-3, 0.984375]`).
    pub sigma_knee_fp16: U16,
    /// Base continuous passive recirculation mix `freeze_mix` (`fp16`, `[0.0, 1.0]`).
    pub freeze_mix_fp16: U16,
    /// Time-stretcher Hann window depth (`fp16`, `[0.0, 1.0]`).
    pub window_depth_fp16: U16,
    /// Reserved slot (`0`, ignored on decode).
    pub reserved_120_122: U16,
    /// Per-lane Schroeder all-pass integer delays (`<= 255` at `48 kHz`).
    pub allpass_delays: [u8; 4],
    /// Bit 0 = `schroeder_enable`, bit 1 = `ext_f32`, bits `2..=7` strictly `0`.
    pub flags: u8,
    /// Reserved byte (`0`).
    pub reserved: u8,
}

const _: () = assert!(size_of::<TransportQuadWire>() == 128);
const _: () = assert!(align_of::<TransportQuadWire>() == 1);
const _: () = assert!(offset_of!(TransportQuadWire, reserved_52_56) == 52);
const _: () = assert!(offset_of!(TransportQuadWire, lfo_depths_fp16) == 56);
const _: () = assert!(offset_of!(TransportQuadWire, fb_psi_fp16) == 112);
const _: () = assert!(offset_of!(TransportQuadWire, freeze_mix_fp16) == 116);
const _: () = assert!(offset_of!(TransportQuadWire, reserved_120_122) == 120);
const _: () = assert!(offset_of!(TransportQuadWire, allpass_delays) == 122);
const _: () = assert!(offset_of!(TransportQuadWire, flags) == 126);

/// Lite `LinearReadout` Block (`640 B` at Arena 0 offset `2,048..2,688`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutLiteWire {
    /// Pre-convolver linear residual gain `[L, R]`.
    pub pre_conv_u_gain: [F32; 2],
    /// Pre-convolver high-pass residual gain `[L, R]`.
    pub pre_conv_hp_gain: [F32; 2],
    /// Direct post-convolver second-order residual gain `[L, R]`.
    pub direct_s2_u_gain: [F32; 2],
    /// Direct high-pass residual gain `[L, R]`.
    pub direct_s2_hp_gain: [F32; 2],
    /// Unrotated `z_stage` sensitivities (`[0..8]` defaults, `[8..48]` global slopes,
    /// `[48..80]` Lite..Pro `[4][4]` `adaa`/`lru` scales, `[80..96]` `op_p_to_core`).
    pub stage_latent_mods_fp16: [U16; 96],
    /// Static `W_ro,0` (`[0..192]`) and `phasor_to_core[0..4][0..4]` (`[192..208]`).
    pub w_ro_fp16: [U16; 208],
}

/// Full `LinearReadout` Block (`1,024 B` at `tier_start + 6,912`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutFullWire {
    /// Full `ModalConvolver` Prefix Override (`192 B`).
    pub prefix_override: [u8; 192],
    /// `W_ro,0[24×12]`, `8` `[L, R]` channel gains, extra FIR/Kautz slots, zero-filled tail.
    pub w_ro_and_mods_fp16: [U16; 416],
}

/// Pro `LinearReadout` Block (`2,304 B` at `tier_start + 13,568`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutProWire {
    /// Pro `ModalConvolver` Prefix Override (`384 B`).
    pub prefix_override: [u8; 384],
    /// `W_ro,0[32×12]`, `8` `[L, R]` channel gains, extra FIR/Kautz slots, zero-filled tail.
    pub w_ro_and_mods_fp16: [U16; 960],
}

/// Ultra `LinearReadout` Block (`3,584 B` at `tier_start + 23,296`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutUltraWire {
    /// Ultra `ModalConvolver` Prefix Override (`576 B`).
    pub prefix_override: [u8; 576],
    /// `W_ro,0[40×12]`, `8` `[L, R]` channel gains, extra FIR/Kautz slots, zero-filled tail.
    pub w_ro_and_mods_fp16: [U16; 1_504],
}

const _: () = assert!(size_of::<LinearReadoutLiteWire>() == 640);
const _: () = assert!(size_of::<LinearReadoutFullWire>() == 1_024);
const _: () = assert!(size_of::<LinearReadoutProWire>() == 2_304);
const _: () = assert!(size_of::<LinearReadoutUltraWire>() == 3_584);

/// Arena 0 Control Plane (`1,920 B` at Arena 0 offset `2,688..4,608`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena0ControlBusWire {
    /// Reference `3 kHz` `Q0.192` LFO phase increments `[LFO0, LFO1]` in radians/tick.
    pub lfo_dphi_train: [F32; 2],
    /// OU analog drift noise amplitude (`[0.0, 2.0]`).
    pub sigma_drift: F32,
    /// Reference `3 kHz` OU autocorrelation pole `(0.0, 0.9999]`.
    pub rho_drift_train: F32,
    /// 2nd-harmonic quadrature LFO waveshaper skew `[beta_0, beta_1]` in `[-1.0, 1.0]` (`fp16`).
    pub lfo_skew_fp16: [U16; 2],
    /// Stage-level `Transport` dry pass-through mix `[-2.0, 2.0]` (`fp16`, `0x0000 -> 0.0`, NaN -> `1.0`).
    pub tank_dry_mix_fp16: U16,
    /// Reserved control scalar (`0`).
    pub reserved_ctrl_fp16: U16,
    /// Learned projection weights from OU drift `ou_drift_state[0..4]` onto `z_stage[0..4]` (`fp16`).
    pub ou_to_z_stage_fp16: [U16; 4],
    /// 4-D trust-region radius `rho_max` (`[0.01, 16.0]`, default `2.5`).
    pub trust_radius_mh: F32,
    /// Inline `CP1` contributor directory (`<= 8` entries).
    pub cp1_dir: Cp1InlineDirWire,
    /// `U_int^base[4][8]` rank-4 FM interaction readout (`128 B`).
    pub u_int_base: [[F32; 8]; 4],
    /// Inline `CP1` concatenated `4(9k + 4) B` contributor slices (`576 B`).
    pub cp1_base: [u8; 576],
    /// Unified `3 kHz` dispatch matrix `W_ctrl[16][16]` (`1,024 B`).
    pub w_ctrl: [[F32; 16]; 16],
    /// `[0]` mirrors `trust_radius_mh`, `[1..24]` Role B watermark carrier (`0.0` unwatermarked),
    /// `[24..32]` Lite `Op-T` crossbar `opt_to_lr[0..4][0..2]`.
    pub rho_head_and_ctrl_tail: [F32; 32],
}

/// Arena 3 Control Plane Delta (`3,840 B` at `tier_start + 26,880..30,720`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena3ControlBusWire {
    /// Ultra `3 kHz` delta dispatch matrix `W_ctrl_ultra[16][4]` (`256 B`).
    pub w_ctrl_ultra: [[F32; 4]; 16],
    /// Ultra rank-8 FM interaction deltas `3 × U_int,Delta[4][8]` (`384 B`).
    pub u_int_ultra: [[[F32; 8]; 4]; 3],
    /// Inline Ultra `CP1` concatenated `4(8k + 4) B` slices (`2,112 B`).
    pub cp1_ultra: [u8; 2_112],
    /// Ultra `[0..80]` `[8][10]` `w_adaa_drive_scale`, `[80..160]` `[8][10]` `w_lru_bias_shift`,
    /// `[160..164]` `ou_to_z_stage_ultra_fp16[4]`, `[164..256]` reserved zero tail.
    pub stage_latent_ctrl_ultra_fp16: [U16; 256],
    /// Reserved alignment tail (`0x00` unwatermarked).
    pub reserve: [u8; 576],
}

const _: () = assert!(size_of::<Arena0ControlBusWire>() == 1_920);
const _: () = assert!(size_of::<Arena3ControlBusWire>() == 3_840);
const _: () = assert!(offset_of!(Arena0ControlBusWire, lfo_skew_fp16) == 16);
const _: () = assert!(offset_of!(Arena0ControlBusWire, tank_dry_mix_fp16) == 20);
const _: () = assert!(offset_of!(Arena0ControlBusWire, reserved_ctrl_fp16) == 22);
const _: () = assert!(offset_of!(Arena0ControlBusWire, ou_to_z_stage_fp16) == 24);
const _: () = assert!(offset_of!(Arena0ControlBusWire, trust_radius_mh) == 32);
const _: () = assert!(offset_of!(Arena0ControlBusWire, cp1_dir) == 36);
const _: () = assert!(offset_of!(Arena0ControlBusWire, u_int_base) == 64);
const _: () = assert!(offset_of!(Arena0ControlBusWire, cp1_base) == 192);
const _: () = assert!(offset_of!(Arena0ControlBusWire, w_ctrl) == 768);
const _: () = assert!(offset_of!(Arena0ControlBusWire, rho_head_and_ctrl_tail) == 1_792);

/// Self-contained Arena 0 Lite payload (`4,608 B` = `72` cache lines, `REQ-5.6`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena0LiteWire {
    /// Crossbar & `SensoryQuad` Block (`0..256`).
    pub crossbar: CrossbarWire,
    /// Lite `CoreQuad` blocks `0..4` (`256..1,024`).
    pub core_quads: [CoreQuadWire; 4],
    /// Lite inter-block tile pairs `0..3` (`1,024..1,408`).
    pub tiles: [SonaInterBlockTilePairWire; 3],
    /// Lite `ModalConvolver` Early FIR (`32` taps/ch, `1,408..1,664`).
    pub conv_fir_lite: ConvolverFir32Wire,
    /// Lite `ModalConvolver` Kautz blocks `0..2` (`1,664..1,920`).
    pub conv_kautz_lite: [KautzSoaBlockWire; 2],
    /// `TransportQuad` 0 (`1,920..2,048`).
    pub transport_lite: TransportQuadWire,
    /// Lite `LinearReadout` (`2,048..2,688`).
    pub linear_readout_lite: LinearReadoutLiteWire,
    /// Arena 0 Control Plane (`2,688..4,608`).
    pub control_bus_lite: Arena0ControlBusWire,
}

/// Arena 1 Full Delta payload (`3,328 B`, cumulative `7,936 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena1FullDeltaWire {
    /// Full `CoreQuad` blocks `4..6` (`4,608..4,992`).
    pub core_quads: [CoreQuadWire; 2],
    /// Full inter-block tile pairs `3..13` (`4,992..6,272`).
    pub tiles: [SonaInterBlockTilePairWire; 10],
    /// Full Crossbar Coupling Block (`6,272..6,528`).
    pub crossbar_full: SonaCrossbarBlockWire,
    /// Full `ModalConvolver` Early FIR delta (`+16` taps/ch, `6,528..6,656`).
    pub conv_fir_full: ConvolverFir16Wire,
    /// Full `ModalConvolver` Kautz block (`+4` pairs, `6,656..6,784`).
    pub conv_kautz_full: KautzSoaBlockWire,
    /// `TransportQuad` 1 (`6,784..6,912`).
    pub transport_full: TransportQuadWire,
    /// Full `LinearReadout` (`6,912..7,936`).
    pub linear_readout_full: LinearReadoutFullWire,
}

/// Arena 2 Pro Delta payload (`7,936 B`, cumulative `15,872 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena2ProDeltaWire {
    /// Pro `CoreQuad` blocks `6..8` (`7,936..8,320`).
    pub core_quads: [CoreQuadWire; 2],
    /// Pro inter-block tile pairs `13..23` (`8,320..9,600`).
    pub tiles: [SonaInterBlockTilePairWire; 10],
    /// Pro Crossbar Coupling Block (`9,600..9,856`).
    pub crossbar_pro: SonaCrossbarBlockWire,
    /// Reserved alignment pad (`9,856..12,544`, `0x00` unwatermarked).
    pub reserve_core: [u8; 2_688],
    /// Pro `ModalConvolver` Early FIR delta (`+48` taps/ch, `12,544..12,928`).
    pub conv_fir_pro: ConvolverFir48Wire,
    /// Pro `ModalConvolver` Kautz blocks (`+12` pairs, `12,928..13,312`).
    pub conv_kautz_pro: [KautzSoaBlockWire; 3],
    /// `TransportQuad`s (`[0]` = Quad 2, `[1]` = `128 B` reserve, `13,312..13,568`).
    pub transport_pro: [TransportQuadWire; 2],
    /// Pro `LinearReadout` (`13,568..15,872`).
    pub linear_readout_pro: LinearReadoutProWire,
}

/// Arena 3 Ultra Delta payload (`14,848 B`, cumulative `30,720 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena3UltraDeltaWire {
    /// Ultra `CoreQuad` blocks `8..10` (`15,872..16,256`).
    pub core_quads: [CoreQuadWire; 2],
    /// Ultra inter-block tile pairs `23..35` (`16,256..17,792`).
    pub tiles: [SonaInterBlockTilePairWire; 12],
    /// Ultra Crossbar Coupling Block (`17,792..18,048`).
    pub crossbar_ultra: SonaCrossbarBlockWire,
    /// Row-major `[8][128]` unrotated `z_stage` per-lane sensitivity (`18,048..20,096`).
    pub stage_latent_core_mods_fp16: [U16; 1_024],
    /// Reserved alignment pad (`20,096..22,528`, `0x00` unwatermarked).
    pub reserve_core: [u8; 2_432],
    /// Ultra `ModalConvolver` Early FIR delta (`+32` taps/ch, `22,528..22,784`).
    pub conv_fir_ultra: ConvolverFir32Wire,
    /// Ultra `ModalConvolver` Kautz blocks (`+8` pairs, `22,784..23,040`).
    pub conv_kautz_ultra: [KautzSoaBlockWire; 2],
    /// `TransportQuad`s (`[0]` = Quad 3, `[1]` = `128 B` reserve, `23,040..23,296`).
    pub transport_ultra: [TransportQuadWire; 2],
    /// Ultra `LinearReadout` (`23,296..26,880`).
    pub linear_readout_ultra: LinearReadoutUltraWire,
    /// Arena 3 Control Plane (`26,880..30,720`).
    pub control_bus_ultra_delta: Arena3ControlBusWire,
}

const _: () = assert!(size_of::<Arena0LiteWire>() == 4_608);
const _: () = assert!(size_of::<Arena1FullDeltaWire>() == 3_328);
const _: () = assert!(size_of::<Arena2ProDeltaWire>() == 7_936);
const _: () = assert!(size_of::<Arena3UltraDeltaWire>() == 14_848);
const _: () = assert!(
    size_of::<Arena0LiteWire>()
        + size_of::<Arena1FullDeltaWire>()
        + size_of::<Arena2ProDeltaWire>()
        + size_of::<Arena3UltraDeltaWire>()
        == 30_720
);
const _: () = assert!(offset_of!(Arena0LiteWire, conv_fir_lite) == CONVOLVER_EARLY_FIR_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, conv_kautz_lite) == CONVOLVER_KAUTZ_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, transport_lite) == TRANSPORT_DELAY_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, linear_readout_lite) == LINEAR_READOUT_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, control_bus_lite) == 2_688);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, conv_fir_full)
        == CONVOLVER_EARLY_FIR_BASES[1]
);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, conv_kautz_full)
        == CONVOLVER_KAUTZ_BASES[1]
);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, transport_full)
        == TRANSPORT_DELAY_BASES[1]
);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, linear_readout_full)
        == LINEAR_READOUT_BASES[1]
);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, conv_fir_pro) == CONVOLVER_EARLY_FIR_BASES[2]);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, conv_kautz_pro) == CONVOLVER_KAUTZ_BASES[2]);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, transport_pro) == TRANSPORT_DELAY_BASES[2]);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, linear_readout_pro) == LINEAR_READOUT_BASES[2]);
const _: () = assert!(
    15_872 + offset_of!(Arena3UltraDeltaWire, conv_fir_ultra) == CONVOLVER_EARLY_FIR_BASES[3]
);
const _: () = assert!(
    15_872 + offset_of!(Arena3UltraDeltaWire, conv_kautz_ultra) == CONVOLVER_KAUTZ_BASES[3]
);
const _: () =
    assert!(15_872 + offset_of!(Arena3UltraDeltaWire, transport_ultra) == TRANSPORT_DELAY_BASES[3]);
const _: () = assert!(
    15_872 + offset_of!(Arena3UltraDeltaWire, linear_readout_ultra) == LINEAR_READOUT_BASES[3]
);
const _: () = assert!(15_872 + offset_of!(Arena3UltraDeltaWire, control_bus_ultra_delta) == 26_880);

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromZeros;

    #[test]
    fn test_arena_wire_total_sizes_and_bases() {
        let a0 = Arena0LiteWire::new_zeroed();
        let a1 = Arena1FullDeltaWire::new_zeroed();
        let a2 = Arena2ProDeltaWire::new_zeroed();
        let a3 = Arena3UltraDeltaWire::new_zeroed();
        assert_eq!(a0.as_bytes().len(), 4_608);
        assert_eq!(a1.as_bytes().len(), 3_328);
        assert_eq!(a2.as_bytes().len(), 7_936);
        assert_eq!(a3.as_bytes().len(), 14_848);
        assert_eq!(CONVOLVER_KAUTZ_BLOCKS_PER_TIER, [2, 1, 3, 2]);
    }
}
