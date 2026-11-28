//! Declarative `#[repr(C)]` `zerocopy` v4.0 wire block structs (`OpNPreBaseWire`,
//! `SonaCellDescWire`, `CrossbarWire`, `CoreQuadWire`, `SonaInterBlockTilePairWire`,
//! `SonaCrossbarBlockWire`, `ConvolverFir*Wire`, `ErSparseBank8Wire`, `KautzSoaBlockWire`,
//! `TransportQuadWire`, `LinearReadout*Wire`).

use core::mem::{align_of, offset_of, size_of};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned};

/// `12 B` `Op-N[Pre]` v4 record embedded at `CrossbarWire` bytes `180..192`.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct OpNPreBaseWire {
    /// Commutation bias `gamma_bc` in `[0.0, 1.0]`.
    pub gamma_bc: F32,
    /// Fixed ZOH sample-and-hold depth in `[0.0, 0.9999]`.
    pub zoh_hold_depth: F32,
    /// Quantization depth `beta_q` (`fp16` in `[0.0, 16.0]`, bit-exact `0.0` bypass).
    pub beta_q_fp16: U16,
    /// Quantizer edge sharpness `kappa_h` (`fp16` in `[0.0, 1.0]`).
    pub kappa_h_fp16: U16,
}

const _: () = assert!(size_of::<OpNPreBaseWire>() == 12);
const _: () = assert!(align_of::<OpNPreBaseWire>() == 1);

/// `16 B` `SONA` Cell Descriptor header inside `CrossbarWire` bytes `192..208`.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaCellDescWire {
    /// Magic `"SONA"` (`*b"SONA"`).
    pub magic: [u8; 4],
    /// Format major (`4`).
    pub version_major: U16,
    /// Format minor (`0`).
    pub version_minor: U16,
    /// Cumulative present-tier bitmask (`0x01=Lite`, `0x03=Full`, `0x07=Pro`, `0x0F=Ultra`).
    pub tier_flags: U16,
    /// Canonical `v4.0` core oversampling base factor (`2` live `HB-FIR`; Master upgrades to `4`).
    pub oversample_factor: U16,
    /// Mirror of `CrossbarWire::sona_meta_flags`.
    pub cell_flags: U32,
}

const _: () = assert!(size_of::<SonaCellDescWire>() == 16);
const _: () = assert!(align_of::<SonaCellDescWire>() == 1);

/// `256 B` (`4 CL`) `SensoryQuad` v4, pipeline flags, `Op-P` v4 `T1+`, `Op-N[Pre]` v4, and phasor
/// crossbar block at Arena 0 bytes `0..256`.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CrossbarWire {
    /// `SensoryQuad` 48 kHz TPT-SVF cutoff tangent gains `[1e-5, 64.0]` (fallback `0.25`).
    pub sense_svf_g: [F32; 4],
    /// `SensoryQuad` 48 kHz ParaLRU release poles `[0.0, 0x3F7F_FFAC]`.
    pub sense_lru_alpha_rel: [F32; 4],
    /// `SensoryQuad` 48 kHz ParaLRU attack poles `[0.0, 1.0]`.
    pub sense_lru_alpha_atk: [F32; 4],
    /// `SensoryQuad` TPT-SVF damping `fp16` `[0.0, 8.0]`.
    pub sense_svf_k_fp16: [U16; 4],
    /// `SensoryQuad` ParaLRU feedback `fp16` `[0.0, 1.0]`.
    pub sense_lru_kappa_fp16: [U16; 4],
    /// Pipeline flags (`PIPE_CANONICAL_V4_SONA = 0x444E_00F8`, byte offset `64`).
    pub pipeline_flags: U32,
    /// Cumulative `[Lite, Full, Pro, Ultra]` linear output trim (`[0.0625, 16.0]`, byte offset `68`).
    pub tier_gain_trim: [F32; 4],
    /// `SONA` feature flags (`SONA_META_*`, mirrored in `cell_desc.cell_flags`, byte offset `84`).
    pub sona_meta_flags: U32,
    /// Left input projection weights into `SensoryQuad` (`[-4.0, 4.0]`).
    pub sense_in_l_fp16: [U16; 4],
    /// Right input projection weights into `SensoryQuad` (`[-4.0, 4.0]`).
    pub sense_in_r_fp16: [U16; 4],
    /// `SensoryQuad` lowpass envelope feature mix (`[-16.0, 16.0]`).
    pub sense_tone_lp_fp16: [U16; 4],
    /// `SensoryQuad` bandpass envelope feature mix (`[-16.0, 16.0]`).
    pub sense_tone_bp_fp16: [U16; 4],
    /// Crossbar operator gains `[op_p_crossbar_gain, op_t_crossbar_gain]` (`[-4.0, 4.0]`).
    pub crossbar_op_gains_fp16: [U16; 2],
    /// Microchunk anti-zipper scale `4 / K_ctrl` in `(0.0, 1.0]` (default `0.25`, offset `124`).
    pub anti_zipper_ramp_scale: F32,
    /// Repurposed `128..140`: `SensoryQuad` Log-Knee VCA threshold `theta_lin` (`>= 0.0`).
    pub sense_vca_theta_lin: F32,
    /// Log-Knee VCA knee softness `c_knee` (`>= 1e-6`).
    pub sense_vca_c_knee: F32,
    /// `Op-P` v4 `T1+` tracker lowpass pole `g_trk` (`[1e-5, 64.0]`).
    pub op_p_pole_trk: F32,
    /// Repurposed `140..156`: Log-Knee VCA makeup `b_makeup` (`[-5.0, +4.0]` octaves, `fp16`).
    pub sense_vca_b_makeup_fp16: U16,
    /// Log-Knee VCA slope `s_slope` (`[0.0, 4.0]`, `fp16`).
    pub sense_vca_s_slope_fp16: U16,
    /// Log-Knee VCA level scale `l_scale` (`[0.0, 16.0]`, `fp16`).
    pub sense_vca_l_scale_fp16: U16,
    /// `SensoryQuad` peak-vs-RMS envelope detector mix (`[0.0, 1.0]`, `fp16`).
    pub sense_peak_rms_mix_fp16: U16,
    /// `Op-P` v4 `T1+` subharmonic octave mix (`[-4.0, 4.0]`, `fp16`).
    pub op_p_alpha_oct_fp16: U16,
    /// `Op-P` v4 `T1+` second-octave mix (`[-4.0, 4.0]`, `fp16`).
    pub op_p_beta_oc2_fp16: U16,
    /// `Op-P` v4 `T1+` stereo two-band crossover mix (`[-4.0, 4.0]`, `fp16`).
    pub op_p_stereo_mix_fp16: U16,
    /// `Op-P` v4 `T1+` sub-band rect/soft-square mix (`[-4.0, 4.0]`, `fp16`).
    pub op_p_sub_band_mix_fp16: U16,
    /// `Op-P` 3-basis carrier weights `[hi*cos, hq*sin, sin]` (`fp16[3][4]`, bytes `156..180`).
    pub carrier_basis_w_fp16: [[U16; 4]; 3],
    /// `Op-N[Pre]` v4 parameters (`bytes 180..192`).
    pub op_n_pre_base: OpNPreBaseWire,
    /// `SONA` v4 cell descriptor (`bytes 192..208`).
    pub cell_desc: SonaCellDescWire,
    /// 48 kHz Q0.32 phasor bank carrier frequencies in Hz (`bytes 208..224`).
    pub phasor_freq_hz: [F32; 4],
    /// Q0.32 initial phases (`bytes 224..240`).
    pub phasor_init_phase_q32: [U32; 4],
    /// `SensoryQuad` highpass envelope feature mix (`[-16.0, 16.0]`, `bytes 240..248`).
    pub sense_tone_hp_fp16: [U16; 4],
    /// Unified Log-Knee VCA weights (`[-512.0, 512.0]`, `bytes 248..256`).
    pub sense_vca_w_fp16: [U16; 4],
}

const _: () = assert!(size_of::<CrossbarWire>() == 256);
const _: () = assert!(align_of::<CrossbarWire>() == 1);
const _: () = assert!(offset_of!(CrossbarWire, pipeline_flags) == 64);
const _: () = assert!(offset_of!(CrossbarWire, tier_gain_trim) == 68);
const _: () = assert!(offset_of!(CrossbarWire, sona_meta_flags) == 84);
const _: () = assert!(offset_of!(CrossbarWire, sense_vca_theta_lin) == 128);
const _: () = assert!(offset_of!(CrossbarWire, carrier_basis_w_fp16) == 156);
const _: () = assert!(offset_of!(CrossbarWire, op_n_pre_base) == 180);
const _: () = assert!(offset_of!(CrossbarWire, cell_desc) == 192);
const _: () = assert!(offset_of!(CrossbarWire, phasor_freq_hz) == 208);
const _: () = assert!(offset_of!(CrossbarWire, sense_tone_hp_fp16) == 240);

/// `240 B` (`176 B` hot + `64 B` ext cold) 4-lane `SONA v4.0` `CoreQuad` parameter block.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CoreQuadWire {
    /// TPT-SVF cutoff tangent gains `[1e-5, 64.0]`.
    pub svf_g: [F32; 4],
    /// ParaLRU release poles `[0.0, 0x3F7F_FFAC]`.
    pub lru_alpha_rel: [F32; 4],
    /// ParaLRU attack poles `[0.0, 1.0]`.
    pub lru_alpha_atk: [F32; 4],
    /// TPT-SVF damping `2R` (`[0.0, 8.0]`, `fp16`).
    pub svf_k_fp16: [U16; 4],
    /// ParaLRU self-feedback `kappa` (`[0.0, 1.0]`, `fp16`).
    pub lru_kappa_fp16: [U16; 4],
    /// `AlgADAA1` saturation drive (`[-32.0, 32.0]`, `fp16`).
    pub adaa_drive_fp16: [U16; 4],
    /// ParaLRU bias drive (`[-16.0, 16.0]`, `fp16`).
    pub lru_bias_drive_fp16: [U16; 4],
    /// 3-tap folded tone lowpass weight `c_lp` (`[-16.0, 16.0]`, `fp16`).
    pub c_lp_fp16: [U16; 4],
    /// 3-tap folded tone bandpass weight `c_bp` (`[-16.0, 16.0]`, `fp16`).
    pub c_bp_fp16: [U16; 4],
    /// Block 0 damped-bandpass weight `c_kbp` (`[-16.0, 16.0]` on Block 0; `0x0000` on Blocks 1..3).
    pub c_kbp_fp16: [U16; 4],
    /// 3-tap folded tone direct input weight `c_u` (`[-16.0, 16.0]`, `fp16`).
    pub c_u_fp16: [U16; 4],
    /// `g_pre(v^2)` zero-amplitude gain `g0` (`[0.0, 2.5]`, `fp16`).
    pub g0_fp16: [U16; 4],
    /// `g_pre(v^2)` high-amplitude asymptote `g_inf` (`[0.0, 2.5]`, `fp16`).
    pub g_inf_fp16: [U16; 4],
    /// Strictly unsigned $C^2$ wavefolder crossfade `[0.0, 1.0]` (`(bits & 0x8000) == 0`, offset `128`).
    pub wavefolder_mix_fp16: [U16; 4],
    /// Block 0 PHS nonlinear damping (`[0.0, 16.0]` on Block 0; `0x0000` on Blocks 1..3, offset `136`).
    pub svf_k_nl_fp16: [U16; 4],
    /// Pre-emphasis saturation scale `d_pre` (`[0.0, 65_504.0]`, `fp16`).
    pub d_pre_fp16: [U16; 4],
    /// Polarized linear bypass mix `lin_mix` (`[0.0, 1.0]`, `fp16`).
    pub lin_mix_fp16: [U16; 4],
    /// Stage energy bus square weight `se_w_sq` (`[0.0, 65_504.0]`, `fp16`).
    pub se_w_sq_fp16: [U16; 4],
    /// Stage energy bus coupling gain `se_d_bus` (`[0.0, 65_504.0]`, `fp16`).
    pub se_d_bus_fp16: [U16; 4],
    /// Post-core output gain `out_gain` (`[-4.0, 4.0]`, `fp16`; `176 B` ext start).
    pub out_gain_fp16: [U16; 4],
    /// M/S input mid projection weight (`[-16.0, 16.0]`, `fp16`).
    pub in_mid_weight_fp16: [U16; 4],
    /// M/S input side projection weight (`[-16.0, 16.0]`, `fp16`).
    pub in_side_weight_fp16: [U16; 4],
    /// ParaLRU stereo envelope link `[0.0, 1.0]` (`fp16`).
    pub lru_stereo_link_fp16: [U16; 4],
    /// Selective-Duhem flux modulation `gamma` (`[0.0, 64.0]`, `fp16`).
    pub lru_duhem_gamma_fp16: [U16; 4],
    /// Signed vs soft-rectified ParaLRU mix `[0.0, 1.0]` (`fp16`).
    pub lru_signed_mix_fp16: [U16; 4],
    /// Bounded GLU modulation depth (`[-4.0, 4.0]`, `fp16`).
    pub glu_mix_fp16: [U16; 4],
    /// Explicit per-lane rectifier sign bitmask (`0 = +1.0, 1 = -1.0`, bits `0..4`, `4..16` zero).
    pub rect_sign_mask: U16,
    /// Reserved zero alignment padding (`[0; 3]`).
    pub reserved_rect_pad: [U16; 3],
}

const _: () = assert!(size_of::<CoreQuadWire>() == 240);
const _: () = assert!(align_of::<CoreQuadWire>() == 1);
const _: () = assert!(offset_of!(CoreQuadWire, wavefolder_mix_fp16) == 128);
const _: () = assert!(offset_of!(CoreQuadWire, svf_k_nl_fp16) == 136);
const _: () = assert!(offset_of!(CoreQuadWire, out_gain_fp16) == 176);
const _: () = assert!(offset_of!(CoreQuadWire, rect_sign_mask) == 232);

/// `128 B` (`2 CL`) directed inter-block coupling tile pair (`5 / 13 / 24 / 37` cumulative).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaInterBlockTilePairWire {
    /// Odd nonlinear cross-coupling `4x4` matrix `[src][dst]` (`f32le`).
    pub w_odd_4x4: [[F32; 4]; 4],
    /// Even linear/ADAA cross-coupling `4x4` matrix `[src][dst]` (`fp16`).
    pub w_even_4x4_fp16: [[U16; 4]; 4],
    /// ParaLRU rectifier output cross-coupling `4x4` matrix `[src][dst]` (`fp16`).
    pub w_lru_4x4_fp16: [[U16; 4]; 4],
}

const _: () = assert!(size_of::<SonaInterBlockTilePairWire>() == 128);
const _: () = assert!(align_of::<SonaInterBlockTilePairWire>() == 1);

/// `256 B` (`4 CL`) full-replacement SONA crossbar coupling block (Full / Pro / Ultra).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SonaCrossbarBlockWire {
    /// `Op-P` -> 10 `CoreQuad` slots coupling (`[-8.0, 8.0]`, `fp16`).
    pub op_p_to_core_fp16: [[U16; 4]; 10],
    /// Phasor carrier -> 10 `CoreQuad` slots coupling (`[-8.0, 8.0]`, `fp16`).
    pub phasor_to_core_fp16: [[U16; 4]; 10],
    /// `Op-T` tank -> `LinearReadout` feedback couplings (`[-4.0, 4.0]`, `fp16`).
    pub opt_to_lr_fp16: [[U16; 2]; 16],
    /// Stereo phasor relative phase `[0..4]` in `[-pi, pi]`; `[4..8]` reserved `0`.
    pub phasor_stereo_phase_fp16: [U16; 8],
    /// Reserved `0x0000`.
    pub crossbar_reserved_fp16: [U16; 8],
}

const _: () = assert!(size_of::<SonaCrossbarBlockWire>() == 256);
const _: () = assert!(align_of::<SonaCrossbarBlockWire>() == 1);

/// `128 B` (`2 CL`) 16-tap stereo early FIR block.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConvolverFir16Wire {
    pub fir_l: [F32; 16],
    pub fir_r: [F32; 16],
}

const _: () = assert!(size_of::<ConvolverFir16Wire>() == 128);
const _: () = assert!(align_of::<ConvolverFir16Wire>() == 1);

/// `256 B` (`4 CL`) 32-tap stereo early FIR block.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConvolverFir32Wire {
    pub fir_l: [F32; 32],
    pub fir_r: [F32; 32],
}

const _: () = assert!(size_of::<ConvolverFir32Wire>() == 256);
const _: () = assert!(align_of::<ConvolverFir32Wire>() == 1);

/// `384 B` (`6 CL`) 48-tap stereo early FIR block.
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConvolverFir48Wire {
    pub fir_l: [F32; 48],
    pub fir_r: [F32; 48],
}

impl Default for ConvolverFir48Wire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<ConvolverFir48Wire>() == 384);
const _: () = assert!(align_of::<ConvolverFir48Wire>() == 1);

/// `80 B` `ErSparseBank` `8`-tap `2x2` MIMO early-reflection bank (`8` Lite, `+8 = 16` Full–Ultra).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ErSparseBank8Wire {
    /// 48 kHz base integer tap delays in `[0, 1020]`.
    pub tap_delay_48k: [U16; 8],
    /// Left-to-left MIMO gains (`fp16`).
    pub tap_gain_ll_fp16: [U16; 8],
    /// Right-to-left MIMO gains (`fp16`).
    pub tap_gain_lr_fp16: [U16; 8],
    /// Left-to-right MIMO gains (`fp16`).
    pub tap_gain_rl_fp16: [U16; 8],
    /// Right-to-right MIMO gains (`fp16`).
    pub tap_gain_rr_fp16: [U16; 8],
}

const _: () = assert!(size_of::<ErSparseBank8Wire>() == 80);
const _: () = assert!(align_of::<ErSparseBank8Wire>() == 1);

/// `128 B` (`2 CL`) SoA 4-pair shared-pole Kautz modal block (`v4.0` reclaims `dr` bytes `16..32`).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct KautzSoaBlockWire {
    /// Base pole radii `r_base_48k` in `[0.0, 0.9995]`.
    pub r_low: [F32; 4],
    /// Reclaimed bytes `16..24`: per-pair integer onset delay in `[0, 1023]` @ 48 kHz.
    pub onset_delay_samp: [U16; 4],
    /// Reclaimed bytes `24..32`: stereo cross-coupling modal gains (`fp16`).
    pub c_cross_fp16: [U16; 4],
    /// Pole angles `theta_48k` in `[0.0, pi - 1e-4]`.
    pub theta: [F32; 4],
    /// Excitation input scaling (`>= 0.0`).
    pub b_gamma: [F32; 4],
    /// Left real modal weights (`REQ-8.5` Tardos span).
    pub c_re_l: [F32; 4],
    /// Left imag modal weights (`REQ-8.5` Tardos span).
    pub c_im_l: [F32; 4],
    /// Right real modal weights (`REQ-8.5` Tardos span).
    pub c_re_r: [F32; 4],
    /// Right imag modal weights (`REQ-8.5` Tardos span).
    pub c_im_r: [F32; 4],
}

const _: () = assert!(size_of::<KautzSoaBlockWire>() == 128);
const _: () = assert!(align_of::<KautzSoaBlockWire>() == 1);
const _: () = assert!(offset_of!(KautzSoaBlockWire, onset_delay_samp) == 16);
const _: () = assert!(offset_of!(KautzSoaBlockWire, c_cross_fp16) == 24);
const _: () = assert!(offset_of!(KautzSoaBlockWire, c_re_l) == 64);

/// `192 B` (`3 CL`) `Op-T` v4 `TransportQuad` descriptor (`128 B` base + `64 B` v4 extension block).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TransportQuadWire {
    /// Base 48 kHz fractional loop delays (`[5.0, max_tau]`; `>= 42.0` if crossbar active).
    pub delay_samples: [F32; 4],
    /// Fractional grain ramp span lengths in samples.
    pub ramp_spans_samples: [F32; 4],
    /// Bipolar loop feedback gain (`[-0.9990, +0.9990]`).
    pub g_loop: F32,
    /// Tank DC blocker pole (`[0.95, 0.999995]`).
    pub r_dc: F32,
    /// Tank HF damping pole (`[0.0, 0.9995]`).
    pub damp_pole: F32,
    /// LFO rate in Hz (`[0.0, 16.0]`).
    pub lfo_rate_hz: F32,
    /// Pitch/ramp frequency in Hz (`[-64.0, +64.0]`).
    pub ramp_rate_hz: F32,
    /// Reserved scalar slot (`0.0`, bytes `52..56`).
    pub reserved_52_56: F32,
    /// LFO delay-modulation depths (`fp16`).
    pub lfo_depths_fp16: [U16; 4],
    /// Q0.16 initial LFO phase offsets.
    pub lfo_phase_off_q16: [U16; 4],
    /// Q0.16 initial grain ramp phase offsets.
    pub ramp_phase_off_q16: [U16; 4],
    /// Per-lane tank input injection gains (`fp16`).
    pub input_gains_fp16: [U16; 4],
    /// Left wet tap gains (`fp16`).
    pub wet_tap_l_fp16: [U16; 4],
    /// Right wet tap gains (`fp16`).
    pub wet_tap_r_fp16: [U16; 4],
    /// Schroeder allpass coefficients (`[-0.75, 0.75]`, `fp16`).
    pub allpass_g_fp16: [U16; 4],
    /// Feedback scattering Givens angle `psi` (`fp16`).
    pub fb_psi_fp16: U16,
    /// Unified loop saturation ceiling `sigma_knee` (`[1e-3, 4.0]`, max `0x4400`, `fp16`).
    pub sigma_knee_fp16: U16,
    /// Continuous unitary freeze mix (`[0.0, 1.0]`, `fp16`).
    pub freeze_mix_fp16: U16,
    /// Grain crossfade window depth (`fp16`).
    pub window_depth_fp16: U16,
    /// Input/output scattering angle `psi` (`fp16`).
    pub scat_psi_fp16: U16,
    /// Integer Schroeder allpass delays at 48 kHz (`u8[4]`).
    pub allpass_delays: [u8; 4],
    /// Flags (`bit 1 = ext_f32`).
    pub flags: u8,
    /// Reserved byte (`0`).
    pub reserved: u8,
    /// v4 extension (`128..136`): per-lane LFO rate multipliers (`fp16[4]`).
    pub lfo_rate_ratio_fp16: [U16; 4],
    /// v4 extension (`136..148`): causal lower-triangular series matrix `M_series` (`6 * fp16`).
    pub m_series_fp16: [U16; 6],
    /// v4 extension (`148..156`): per-lane tap feedback/crossfeed modulation (`fp16[4]`).
    pub m_tap_fp16: [U16; 4],
    /// v4 extension (`156..158`): passive feedback rotation parameter `alpha` (`fp16`).
    pub m_fb_alpha_fp16: U16,
    /// v4 extension (`158..160`): passive feedback parameter `beta` (`fp16`).
    pub m_fb_beta_fp16: U16,
    /// v4 extension (`160..162`): loop amplitude modulation depth (`fp16`).
    pub am_depth_fp16: U16,
    /// v4 extension (`162..164`): asymmetric even-harmonic saturation blend (`fp16`).
    pub eta_even_fp16: U16,
    /// v4 extension (`164..166`): sum-bus feedback coupling `alpha_sum_fb` (`fp16`).
    pub alpha_sum_fb_fp16: U16,
    /// v4 extension (`166..168`): orthonormal freeze crossmix `freeze_ortho_mix` (`fp16`).
    pub freeze_ortho_mix_fp16: U16,
    /// v4 extension (`168..192`): reserved zero tail.
    pub reserved_v4_ext: [U16; 12],
}

const _: () = assert!(size_of::<TransportQuadWire>() == 192);
const _: () = assert!(align_of::<TransportQuadWire>() == 1);
const _: () = assert!(offset_of!(TransportQuadWire, lfo_rate_ratio_fp16) == 128);
const _: () = assert!(offset_of!(TransportQuadWire, m_series_fp16) == 136);

/// `640 B` (`10 CL`) Lite half-precision readout and stage-latent block (`REQ-8.5` Tardos span).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutLiteWire {
    pub pre_conv_u_gain: [F32; 2],
    pub pre_conv_hp_gain: [F32; 2],
    pub direct_s2_u_gain: [F32; 2],
    pub direct_s2_hp_gain: [F32; 2],
    pub stage_latent_mods_fp16: [U16; 96],
    pub w_ro_fp16: [U16; 208],
}

impl Default for LinearReadoutLiteWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<LinearReadoutLiteWire>() == 640);

/// `1,024 B` (`16 CL`) Full delta readout + 192 B convolver prefix override (`REQ-8.5`).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutFullWire {
    pub prefix_override: [u8; 192],
    pub w_ro_and_mods_fp16: [U16; 416],
}

impl Default for LinearReadoutFullWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<LinearReadoutFullWire>() == 1_024);

/// `2,304 B` (`36 CL`) Pro delta readout + 384 B convolver prefix override (`REQ-8.5`).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutProWire {
    pub prefix_override: [u8; 384],
    pub w_ro_and_mods_fp16: [U16; 960],
}

impl Default for LinearReadoutProWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<LinearReadoutProWire>() == 2_304);

/// `3,584 B` (`56 CL`) Ultra delta readout + 576 B convolver prefix override (`REQ-8.5`).
#[repr(C)]
#[derive(
    Copy, Clone, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct LinearReadoutUltraWire {
    pub prefix_override: [u8; 576],
    pub w_ro_and_mods_fp16: [U16; 1_504],
}

impl Default for LinearReadoutUltraWire {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

const _: () = assert!(size_of::<LinearReadoutUltraWire>() == 3_584);
