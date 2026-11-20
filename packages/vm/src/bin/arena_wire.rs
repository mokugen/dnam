//! Declarative `#[repr(C)]` `zerocopy` `SONA v3.0` Arena 0..3 tensor and descriptor blocks
//! (`4,608 / 7,936 / 15,872 / 30,720 B` cumulative tier-prefix payloads, `ref[container.dnam.tier-prefix]`,
//! `ref[container.dnam.tensor-directory]`).

use core::mem::{align_of, offset_of, size_of};

use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::header::Cp1InlineDirWire;

/// Relative `tier_start` offsets for Cabinet Early FIR slices across `Lite..Ultra`.
pub const CABINET_EARLY_FIR_BASES: [usize; 4] = [1_408, 6_528, 12_544, 22_528];
/// Relative `tier_start` offsets for Cabinet Kautz SoA blocks across `Lite..Ultra`.
pub const CABINET_KAUTZ_BASES: [usize; 4] = [1_664, 6_656, 12_928, 22_784];
/// Number of `128 B` Kautz SoA blocks introduced in each tier (`Lite..Ultra`).
pub const CABINET_KAUTZ_BLOCKS_PER_TIER: [usize; 4] = [2, 1, 3, 2];
/// Relative `tier_start` offsets for Spatial Tank Quads across `Lite..Ultra`.
pub const TANK_DELAY_BASES: [usize; 4] = [1_920, 6_784, 13_312, 23_040];
/// Relative `tier_start` offsets for VarPro Readout blocks across `Lite..Ultra`.
pub const VARPRO_READOUT_BASES: [usize; 4] = [2_048, 6_912, 13_568, 23_296];

/// Required upper halfword magic in `CrossbarWire::pipeline_flags`.
pub const PIPE_MAGIC_HI: u32 = 0x444E_0000;
/// Bitmask for `PIPE_MAGIC_HI`.
pub const PIPE_MAGIC_MASK: u32 = 0xFFFF_0000;
/// `pipeline_flags` bit 3: informational cabinet extension flag.
pub const PIPE_FLAG_CABINET_EXT: u32 = 1 << 3;
/// `pipeline_flags` bit 4: informational spatial tank extension flag.
pub const PIPE_FLAG_TANK_EXT: u32 = 1 << 4;
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
    PIPE_MAGIC_HI | PIPE_REQUIRED_SONA_MASK | PIPE_FLAG_CABINET_EXT | PIPE_FLAG_TANK_EXT;

/// `sona_meta_flags` bit 4: summary core 2× oversampling flag.
pub const SONA_META_OS2X: u32 = 1 << 4;
/// `sona_meta_flags` bit 6: bits `0..=3` authoritatively specify per-tier `tier_os2x`.
pub const SONA_META_TIER_OS_EXPLICIT: u32 = 1 << 6;
/// `sona_meta_flags` bit 7: higher tier applies Cabinet Prefix Override.
pub const SONA_META_CABINET_PREFIX_OVERRIDES: u32 = 1 << 7;
/// `sona_meta_flags` bit 8: decode `os_and_front_ext` (`128..192`) verbatim.
pub const SONA_META_OS_EXT_PRESENT: u32 = 1 << 8;
/// `sona_meta_flags` bit 9: Ultra `stage_latent_core_mods_fp16` active.
pub const SONA_META_CORE_MODS_PRESENT: u32 = 1 << 9;
/// `sona_meta_flags` bit 16: tank tap gains enabled (`1` canonical in v3.0).
pub const SONA_META_TANK_TAP_GAINS: u32 = 1 << 16;
/// `sona_meta_flags` bit 17: router gain, mix, and hysteretic gate decoded verbatim.
pub const SONA_META_ROUTER_EXT: u32 = 1 << 17;
/// `sona_meta_flags` bit 20: tank LFO enabled (`1` canonical in v3.0).
pub const SONA_META_TANK_LFO: u32 = 1 << 20;
/// Reserved `sona_meta_flags` / `cell_flags` bits (`5, 10..=15, 18..=19, 21..=31`).
pub const SONA_META_RESERVED_MASK: u32 = (1 << 5) | 0xFC00 | (0b11 << 18) | 0xFFE0_0000;

/// `TankQuadWire::flags` bit 0: Schroeder diffuser stage enabled.
pub const TANK_FLAG_SCHROEDER_ENABLE: u8 = 1 << 0;
/// `TankQuadWire::flags` bit 1: external ring uses `f32` (`1`) instead of `fp16` (`0`).
pub const TANK_FLAG_EXT_F32: u8 = 1 << 1;

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

/// Crossbar Block (`256 B` at Arena 0 offset `0..256`, `ref[container.dnam.tensor-directory]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CrossbarWire {
    /// Causal 4-port crossbar `S_router` (`S_{ret->send} == 0`).
    pub s_router: [[F32; 4]; 4],
    /// Pipeline version & feature flags (`PIPE_CANONICAL_V3_SONA = 0x444E_00F8`).
    pub pipeline_flags: U32,
    /// Per-tier output loudness alignment trim `[Lite, Full, Pro, Ultra]`.
    pub tier_gain_trim: [F32; 4],
    /// SONA feature flag word mirrored in `cell_desc.cell_flags`.
    pub sona_meta_flags: U32,
    /// Cold summary dry/wet blend mirror of Quad 0 `dry_mix`.
    pub dry_wet_mix: F32,
    /// Cold summary anti-creep DC/AC pole mirror of Quad 0 `r_dc`.
    pub r_ac: F32,
    /// I/O router input gains `[L, R]`.
    pub router_gain: [F32; 2],
    /// I/O router 2×2 mid/side/channel mix matrix.
    pub router_mix_matrix: [[F32; 2]; 2],
    /// Reserved scalar (`0.0`).
    pub reserved_120_124: F32,
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
    /// Packed `fp16` router gate `[0..6]` and `[op_p_crossbar_gain, op_t_crossbar_gain]`.
    pub gate_and_crossbar_fp16: [U16; 8],
}

const _: () = assert!(size_of::<CrossbarWire>() == 256);
const _: () = assert!(align_of::<CrossbarWire>() == 1);
const _: () = assert!(offset_of!(CrossbarWire, pipeline_flags) == 64);
const _: () = assert!(offset_of!(CrossbarWire, sona_meta_flags) == 84);
const _: () = assert!(offset_of!(CrossbarWire, anti_zipper_ramp_scale) == 124);
const _: () = assert!(offset_of!(CrossbarWire, os_up_quad_coeff) == 128);
const _: () = assert!(offset_of!(CrossbarWire, carrier_basis_w_fp16) == 156);
const _: () = assert!(offset_of!(CrossbarWire, op_n_pre_base) == 180);
const _: () = assert!(offset_of!(CrossbarWire, cell_desc) == 192);
const _: () = assert!(offset_of!(CrossbarWire, gate_and_crossbar_fp16) == 240);

/// 4-lane SONA Group Block (`192 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaGroupWire {
    /// TPT-SVF `g` coefficients (`48 kHz`).
    pub svf_g: [F32; 4],
    /// TPT-SVF `k` damping factors.
    pub svf_k: [F32; 4],
    /// ParaLRU release decay poles (`<= MAX_PARALRU_R_F32`).
    pub lru_alpha_rel: [F32; 4],
    /// ParaLRU attack poles.
    pub lru_alpha_atk: [F32; 4],
    /// ParaLRU envelope detector gain `kappa`.
    pub lru_kappa: [F32; 4],
    /// PolyADAA1 pre-activation drive.
    pub adaa_drive: [F32; 4],
    /// ParaLRU bias drive.
    pub lru_bias_drive: [F32; 4],
    /// Direct input lane weights.
    pub in_x_weight: [F32; 4],
    /// SVF low-pass mix weights (`fp16`).
    pub tone_lp_fp16: [U16; 4],
    /// SVF band-pass mix weights (`fp16`).
    pub tone_bp_fp16: [U16; 4],
    /// SVF high-pass mix weights (`fp16`).
    pub tone_hp_fp16: [U16; 4],
    /// SVF all-pass mix weights (`fp16`).
    pub tone_ap_fp16: [U16; 4],
    /// Direct dry bypass weights (`fp16`).
    pub tone_dir_fp16: [U16; 4],
    /// Sag supply coupling depths (`fp16`).
    pub sag_depth_fp16: [U16; 4],
    /// Wavefolder wet mix (`fp16`).
    pub wavefolder_mix_fp16: [U16; 4],
    /// Mechanism A2 sign parity (`fp16`, decoded to `{-1.0, +1.0}`).
    pub rect_sign_fp16: [U16; 4],
}

const _: () = assert!(size_of::<SonaGroupWire>() == 192);
const _: () = assert!(align_of::<SonaGroupWire>() == 1);

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
    /// Packed `fp16` weights from `Op-P` / `Op-M` buses to each of `10` SONA groups (`80 B`).
    pub op_p_to_group_fp16: [[U16; 4]; 10],
    /// Packed `fp16` `Q0.32` phasor drives to each of `10` SONA groups (`80 B`).
    pub phasor_to_group_fp16: [[U16; 4]; 10],
    /// Packed `fp16` spatial tank `16 -> [L, R]` readout matrix (`64 B`, complete replacement).
    pub opt_to_lr_fp16: [[U16; 2]; 16],
    /// Stereo phasor offsets (`[0..4]` in `[-pi, pi]`, `[4..8]` reserved `0`).
    pub phasor_stereo_phase_fp16: [U16; 8],
    /// Reserved coupling slots (must equal `0`).
    pub crossbar_reserved_fp16: [U16; 8],
}

const _: () = assert!(size_of::<SonaCrossbarBlockWire>() == 256);
const _: () = assert!(align_of::<SonaCrossbarBlockWire>() == 1);

/// Full Cabinet Early FIR slice (`16` taps/ch = `128 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CabinetFir16Wire {
    /// Left channel early FIR taps.
    pub fir_l: [F32; 16],
    /// Right channel early FIR taps.
    pub fir_r: [F32; 16],
}

/// Lite & Ultra Cabinet Early FIR slice (`32` taps/ch = `256 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CabinetFir32Wire {
    /// Left channel early FIR taps.
    pub fir_l: [F32; 32],
    /// Right channel early FIR taps.
    pub fir_r: [F32; 32],
}

/// Pro Cabinet Early FIR slice (`48` taps/ch = `384 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CabinetFir48Wire {
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

const _: () = assert!(size_of::<CabinetFir16Wire>() == 128);
const _: () = assert!(size_of::<CabinetFir32Wire>() == 256);
const _: () = assert!(size_of::<CabinetFir48Wire>() == 384);
const _: () = assert!(size_of::<KautzSoaBlockWire>() == 128);

/// Unified `128 B` Spatial Tank Quad (`Op-T` 4-lane block, `ref[dsp.op-t.ring.declaration]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TankQuadWire {
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
    /// Tank delay LFO rate in Hz (`[0.0, 16.0]`).
    pub lfo_rate_hz: F32,
    /// Grain/delay ramp rate in Hz (`[-16.0, 16.0]`).
    pub ramp_rate_hz: F32,
    /// Envelope ducker follower pole (`[0.0, 0.99999]`).
    pub vca_pole: F32,
    /// Per-lane LFO modulation depths in `48 kHz` samples (`fp16`).
    pub lfo_depths_fp16: [U16; 4],
    /// Initial `Q0.16` LFO phases.
    pub lfo_phase_off_q16: [U16; 4],
    /// Initial `Q0.16` ramp phases.
    pub ramp_phase_off_q16: [U16; 4],
    /// Per-lane quad input gains (`fp16`).
    pub input_gains_fp16: [U16; 4],
    /// Per-lane Left wet tap weights (`fp16`).
    pub wet_tap_l_fp16: [U16; 4],
    /// Per-lane Right wet tap weights (`fp16`).
    pub wet_tap_r_fp16: [U16; 4],
    /// Per-lane Schroeder all-pass diffuser coefficients (`fp16`, `[-0.75, 0.75]`).
    pub allpass_g_fp16: [U16; 4],
    /// Hadamard/Givens feedback rotation angle in radians (`fp16`).
    pub fb_psi_fp16: U16,
    /// Soft knee feedback limiter threshold (`fp16`).
    pub sigma_knee_fp16: U16,
    /// Direct quad dry mix weight (`fp16`).
    pub dry_mix_fp16: U16,
    /// Time-stretcher Hann window depth (`fp16`, `[0.0, 1.0]`).
    pub window_depth_fp16: U16,
    /// Envelope VCA ducking sensitivity (`fp16`).
    pub vca_sens_fp16: U16,
    /// Per-lane Schroeder all-pass integer delays (`<= 255` at `48 kHz`).
    pub allpass_delays: [u8; 4],
    /// Bit 0 = `schroeder_enable`, bit 1 = `ext_f32`, bits `2..=7` strictly `0`.
    pub flags: u8,
    /// Reserved byte (`0`).
    pub reserved: u8,
}

const _: () = assert!(size_of::<TankQuadWire>() == 128);
const _: () = assert!(align_of::<TankQuadWire>() == 1);
const _: () = assert!(offset_of!(TankQuadWire, lfo_depths_fp16) == 56);
const _: () = assert!(offset_of!(TankQuadWire, fb_psi_fp16) == 112);
const _: () = assert!(offset_of!(TankQuadWire, allpass_delays) == 122);
const _: () = assert!(offset_of!(TankQuadWire, flags) == 126);

/// Lite VarPro Readout Block (`640 B` at Arena 0 offset `2,048..2,688`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct VarProReadoutLiteWire {
    /// Pre-cabinet linear residual gain `[L, R]`.
    pub pre_cab_u_gain: [F32; 2],
    /// Pre-cabinet high-pass residual gain `[L, R]`.
    pub pre_cab_hp_gain: [F32; 2],
    /// Direct post-cabinet second-order residual gain `[L, R]`.
    pub direct_s2_u_gain: [F32; 2],
    /// Direct high-pass residual gain `[L, R]`.
    pub direct_s2_hp_gain: [F32; 2],
    /// Unrotated `z_stage` sensitivities (`[0..8]` defaults, `[8..48]` global slopes,
    /// `[48..80]` Lite..Pro `[4][4]` `adaa`/`lru` scales, `[80..96]` `op_p_to_group`).
    pub stage_latent_mods_fp16: [U16; 96],
    /// Static `W_ro,0` (`[0..192]`) and `phasor_to_group[0..4][0..4]` (`[192..208]`).
    pub w_ro_fp16: [U16; 208],
}

/// Full VarPro Readout Block (`1,024 B` at `tier_start + 6,912`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct VarProReadoutFullWire {
    /// Full Cabinet Prefix Override (`192 B`).
    pub prefix_override: [u8; 192],
    /// `W_ro,0[24×12]`, `8` `[L, R]` channel gains, extra FIR/Kautz slots, zero-filled tail.
    pub w_ro_and_mods_fp16: [U16; 416],
}

/// Pro VarPro Readout Block (`2,304 B` at `tier_start + 13,568`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct VarProReadoutProWire {
    /// Pro Cabinet Prefix Override (`384 B`).
    pub prefix_override: [u8; 384],
    /// `W_ro,0[32×12]`, `8` `[L, R]` channel gains, extra FIR/Kautz slots, zero-filled tail.
    pub w_ro_and_mods_fp16: [U16; 960],
}

/// Ultra VarPro Readout Block (`3,584 B` at `tier_start + 23,296`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct VarProReadoutUltraWire {
    /// Ultra Cabinet Prefix Override (`576 B`).
    pub prefix_override: [u8; 576],
    /// `W_ro,0[40×12]`, `8` `[L, R]` channel gains, extra FIR/Kautz slots, zero-filled tail.
    pub w_ro_and_mods_fp16: [U16; 1_504],
}

const _: () = assert!(size_of::<VarProReadoutLiteWire>() == 640);
const _: () = assert!(size_of::<VarProReadoutFullWire>() == 1_024);
const _: () = assert!(size_of::<VarProReadoutProWire>() == 2_304);
const _: () = assert!(size_of::<VarProReadoutUltraWire>() == 3_584);

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
    /// Standalone audio envelope poles `[0.95, 0.99, 0.998]`.
    pub env_poles: [F32; 3],
    /// Host-shell dither metadata (`0.0` in deterministic VM).
    pub dither_snr_scale: F32,
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
    /// `[24..32]` Lite spatial tank `opt_to_lr[0..4][0..2]`.
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
    /// Ultra `[8][10]` `w_adaa_drive_scale` + `[8][10]` `w_lru_bias_shift` + reserved zero tail.
    pub stage_latent_ctrl_ultra_fp16: [U16; 256],
    /// Reserved alignment tail (`0x00` unwatermarked).
    pub reserve: [u8; 576],
}

const _: () = assert!(size_of::<Arena0ControlBusWire>() == 1_920);
const _: () = assert!(size_of::<Arena3ControlBusWire>() == 3_840);
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
    /// Crossbar Block (`0..256`).
    pub crossbar: CrossbarWire,
    /// Lite core groups `0..4` (`256..1,024`).
    pub groups: [SonaGroupWire; 4],
    /// Lite inter-block tile pairs `0..3` (`1,024..1,408`).
    pub tiles: [SonaInterBlockTilePairWire; 3],
    /// Lite Cabinet Early FIR (`32` taps/ch, `1,408..1,664`).
    pub cabinet_fir_lite: CabinetFir32Wire,
    /// Lite Cabinet Kautz blocks `0..2` (`1,664..1,920`).
    pub cabinet_kautz_lite: [KautzSoaBlockWire; 2],
    /// Spatial Tank Quad 0 (`1,920..2,048`).
    pub spatial_tank_lite: TankQuadWire,
    /// Lite VarPro Readout (`2,048..2,688`).
    pub varpro_readout_lite: VarProReadoutLiteWire,
    /// Arena 0 Control Plane (`2,688..4,608`).
    pub control_bus_lite: Arena0ControlBusWire,
}

/// Arena 1 Full Delta payload (`3,328 B`, cumulative `7,936 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena1FullDeltaWire {
    /// Full core groups `4..6` (`4,608..4,992`).
    pub groups: [SonaGroupWire; 2],
    /// Full inter-block tile pairs `3..13` (`4,992..6,272`).
    pub tiles: [SonaInterBlockTilePairWire; 10],
    /// Full Crossbar Coupling Block (`6,272..6,528`).
    pub crossbar_full: SonaCrossbarBlockWire,
    /// Full Cabinet Early FIR delta (`+16` taps/ch, `6,528..6,656`).
    pub cabinet_fir_full: CabinetFir16Wire,
    /// Full Cabinet Kautz block (`+4` pairs, `6,656..6,784`).
    pub cabinet_kautz_full: KautzSoaBlockWire,
    /// Spatial Tank Quad 1 (`6,784..6,912`).
    pub spatial_tank_full: TankQuadWire,
    /// Full VarPro Readout (`6,912..7,936`).
    pub varpro_readout_full: VarProReadoutFullWire,
}

/// Arena 2 Pro Delta payload (`7,936 B`, cumulative `15,872 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena2ProDeltaWire {
    /// Pro core groups `6..8` (`7,936..8,320`).
    pub groups: [SonaGroupWire; 2],
    /// Pro inter-block tile pairs `13..23` (`8,320..9,600`).
    pub tiles: [SonaInterBlockTilePairWire; 10],
    /// Pro Crossbar Coupling Block (`9,600..9,856`).
    pub crossbar_pro: SonaCrossbarBlockWire,
    /// Reserved alignment pad (`9,856..12,544`, `0x00` unwatermarked).
    pub reserve_core: [u8; 2_688],
    /// Pro Cabinet Early FIR delta (`+48` taps/ch, `12,544..12,928`).
    pub cabinet_fir_pro: CabinetFir48Wire,
    /// Pro Cabinet Kautz blocks (`+12` pairs, `12,928..13,312`).
    pub cabinet_kautz_pro: [KautzSoaBlockWire; 3],
    /// Spatial Tank Quads (`[0]` = Quad 2, `[1]` = `128 B` reserve, `13,312..13,568`).
    pub spatial_tank_pro: [TankQuadWire; 2],
    /// Pro VarPro Readout (`13,568..15,872`).
    pub varpro_readout_pro: VarProReadoutProWire,
}

/// Arena 3 Ultra Delta payload (`14,848 B`, cumulative `30,720 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Arena3UltraDeltaWire {
    /// Ultra core groups `8..10` (`15,872..16,256`).
    pub groups: [SonaGroupWire; 2],
    /// Ultra inter-block tile pairs `23..35` (`16,256..17,792`).
    pub tiles: [SonaInterBlockTilePairWire; 12],
    /// Ultra Crossbar Coupling Block (`17,792..18,048`).
    pub crossbar_ultra: SonaCrossbarBlockWire,
    /// Row-major `[8][128]` unrotated `z_stage` per-lane sensitivity (`18,048..20,096`).
    pub stage_latent_core_mods_fp16: [U16; 1_024],
    /// Reserved alignment pad (`20,096..22,528`, `0x00` unwatermarked).
    pub reserve_core: [u8; 2_432],
    /// Ultra Cabinet Early FIR delta (`+32` taps/ch, `22,528..22,784`).
    pub cabinet_fir_ultra: CabinetFir32Wire,
    /// Ultra Cabinet Kautz blocks (`+8` pairs, `22,784..23,040`).
    pub cabinet_kautz_ultra: [KautzSoaBlockWire; 2],
    /// Spatial Tank Quads (`[0]` = Quad 3, `[1]` = `128 B` reserve, `23,040..23,296`).
    pub spatial_tank_ultra: [TankQuadWire; 2],
    /// Ultra VarPro Readout (`23,296..26,880`).
    pub varpro_readout_ultra: VarProReadoutUltraWire,
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
const _: () = assert!(offset_of!(Arena0LiteWire, cabinet_fir_lite) == CABINET_EARLY_FIR_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, cabinet_kautz_lite) == CABINET_KAUTZ_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, spatial_tank_lite) == TANK_DELAY_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, varpro_readout_lite) == VARPRO_READOUT_BASES[0]);
const _: () = assert!(offset_of!(Arena0LiteWire, control_bus_lite) == 2_688);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, cabinet_fir_full)
        == CABINET_EARLY_FIR_BASES[1]
);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, cabinet_kautz_full)
        == CABINET_KAUTZ_BASES[1]
);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, spatial_tank_full)
        == TANK_DELAY_BASES[1]
);
const _: () = assert!(
    size_of::<Arena0LiteWire>() + offset_of!(Arena1FullDeltaWire, varpro_readout_full)
        == VARPRO_READOUT_BASES[1]
);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, cabinet_fir_pro) == CABINET_EARLY_FIR_BASES[2]);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, cabinet_kautz_pro) == CABINET_KAUTZ_BASES[2]);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, spatial_tank_pro) == TANK_DELAY_BASES[2]);
const _: () =
    assert!(7_936 + offset_of!(Arena2ProDeltaWire, varpro_readout_pro) == VARPRO_READOUT_BASES[2]);
const _: () = assert!(
    15_872 + offset_of!(Arena3UltraDeltaWire, cabinet_fir_ultra) == CABINET_EARLY_FIR_BASES[3]
);
const _: () = assert!(
    15_872 + offset_of!(Arena3UltraDeltaWire, cabinet_kautz_ultra) == CABINET_KAUTZ_BASES[3]
);
const _: () =
    assert!(15_872 + offset_of!(Arena3UltraDeltaWire, spatial_tank_ultra) == TANK_DELAY_BASES[3]);
const _: () = assert!(
    15_872 + offset_of!(Arena3UltraDeltaWire, varpro_readout_ultra) == VARPRO_READOUT_BASES[3]
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
        assert_eq!(CABINET_KAUTZ_BLOCKS_PER_TIER, [2, 1, 3, 2]);
    }
}
