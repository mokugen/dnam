//! Double-precision (`f64`) universal `SONA v3.0` state container ([`SonaModelF64`]),
//! rate-ratio ZOH rescaling, causal quad cascade, and GPU/CPU [`CoreQuadParamsStereo4`] packing.

pub(crate) mod cascade;

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, KernelVec4, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32,
    MIN_TPT_G, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4, SonaToneWeights4, TANK_MAX_G_LIN_F32,
    TINY_CROSSING_EPS, TptSvfCoeffs4,
};
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, rescale_real_pole_zoh, rescale_tpt_svf_g_zoh};
use dnam_vm::rescale_paralru_lane;

use crate::control::clamp::{Phase1ClampOutput, StageStabilityConstraint};
use crate::control::cp1::Cp1WireBundle;
use crate::control::stability::{MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP};
use crate::error::TrainError;
use crate::gpu::abi::{FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE, TILE_FRAMES};
use crate::planner::{CoreScanWorkload, MemoryBudgetProfile, plan_core_scan_execution};
use crate::runner::{
    ActiveComputeBackend, FusedCoreScanInput, FusedCoreScanOutput, TrainRunner,
    pack_core_quad_params,
};

/// Number of `CoreQuad` blocks in Ultra (`10` quads = `40` stereo lanes).
pub const NUM_CORE_QUADS_ULTRA: usize = 10;
/// Total inter-block tile pairs in Ultra (`3 + 10 + 10 + 12 = 35`).
pub const NUM_INTER_BLOCK_TILES_ULTRA: usize = 35;
/// Early FIR tap capacity in Ultra (`32 + 16 + 48 + 32 = 128`).
pub const NUM_FIR_TAPS_ULTRA: usize = 128;
/// Kautz modal pole-pair capacity in Ultra (`8 + 4 + 12 + 8 = 32`).
pub const NUM_KAUTZ_PAIRS_ULTRA: usize = 32;
/// `Op-T` transport quad count in Ultra (`4` quads = `16` lines).
pub const NUM_TRANSPORT_QUADS_ULTRA: usize = 4;

/// `f64` representation of one 4-lane stereo `CoreQuad` (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreQuadF64 {
    /// TPT-SVF cutoff tangent gain `g` at `2x` (`96 kHz` reference) per lane.
    pub svf_g: [f64; 4],
    /// TPT-SVF damping factor `k = 1 / Q` per lane.
    pub svf_k: [f64; 4],
    /// Low-pass tone weight per lane.
    pub tone_lp: [f64; 4],
    /// Band-pass tone weight per lane.
    pub tone_bp: [f64; 4],
    /// High-pass tone weight per lane.
    pub tone_hp: [f64; 4],
    /// All-pass tone weight per lane.
    pub tone_ap: [f64; 4],
    /// Direct bypass weight per lane.
    pub tone_dir: [f64; 4],
    /// `ParaLRU` release pole (`[0.0, MAX_PARALRU_R]`) at `2x`.
    pub lru_alpha_rel: [f64; 4],
    /// `ParaLRU` attack pole (`[0.0, 1.0]`) at `2x`.
    pub lru_alpha_atk: [f64; 4],
    /// `ParaLRU` envelope feedback gain `kappa`.
    pub lru_kappa: [f64; 4],
    /// Selective-Duhem stiffness `gamma_duhem`.
    pub lru_duhem_gamma: [f64; 4],
    /// Selective-Duhem signed flux blend `[0.0, 1.0]`.
    pub lru_signed_mix: [f64; 4],
    /// Stereo detector coupling `[0.0, 1.0]`.
    pub lru_stereo_link: [f64; 4],
    /// Sign parity (`+1.0` or `-1.0`).
    pub rect_sign: [f64; 4],
    /// Power-supply sag depth `[-8.0, 8.0]`.
    pub sag_depth: [f64; 4],
    /// `PolyADAA1` linear drive (`[-32.0, 32.0]`).
    pub adaa_drive: [f64; 4],
    /// `ParaLRU` pre-activation bias injection (`[-16.0, 16.0]`).
    pub lru_bias_drive: [f64; 4],
    /// `C^2` wavefolder blend `[0.0, 1.0]`.
    pub wavefolder_mix: [f64; 4],
    /// Cross-lane bounded `GLU` mix `[-4.0, 4.0]`.
    pub glu_mix: [f64; 4],
    /// Mid input coupling weight.
    pub in_mid_weight: [f64; 4],
    /// Side input coupling weight.
    pub in_side_weight: [f64; 4],
}

impl Default for CoreQuadF64 {
    fn default() -> Self {
        Self {
            svf_g: [0.125; 4],
            svf_k: [core::f64::consts::FRAC_1_SQRT_2; 4],
            tone_lp: [0.0; 4],
            tone_bp: [0.0; 4],
            tone_hp: [0.0; 4],
            tone_ap: [0.0; 4],
            tone_dir: [1.0, 0.0, 0.0, 0.0],
            lru_alpha_rel: [0.995; 4],
            lru_alpha_atk: [0.25; 4],
            lru_kappa: [0.0; 4],
            lru_duhem_gamma: [0.0; 4],
            lru_signed_mix: [0.0; 4],
            lru_stereo_link: [0.0; 4],
            rect_sign: [1.0; 4],
            sag_depth: [0.0; 4],
            adaa_drive: [0.0; 4],
            lru_bias_drive: [0.0; 4],
            wavefolder_mix: [0.0; 4],
            glu_mix: [0.0; 4],
            in_mid_weight: [1.0; 4],
            in_side_weight: [1.0; 4],
        }
    }
}

impl CoreQuadF64 {
    /// Converts `self` into the single-source `32-bit` GPU/CPU uniform bundle `CoreQuadParamsStereo4<GpuVec4>`.
    #[must_use]
    pub fn to_kernel_params(&self) -> CoreQuadParamsStereo4<GpuVec4> {
        let g = GpuVec4::from_array(self.svf_g.map(|v| (v as f32).max(MIN_TPT_G)));
        let k = GpuVec4::from_array(self.svf_k.map(|v| (v as f32).clamp(0.0, 8.0)));
        CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::from_array(self.tone_lp.map(|v| v as f32)),
                bp: GpuVec4::from_array(self.tone_bp.map(|v| v as f32)),
                hp: GpuVec4::from_array(self.tone_hp.map(|v| v as f32)),
                ap: GpuVec4::from_array(self.tone_ap.map(|v| v as f32)),
                dir: GpuVec4::from_array(self.tone_dir.map(|v| v as f32)),
            },
            lru: SonaParaLruDuhemCoeffs4 {
                alpha_rel: GpuVec4::from_array(
                    self.lru_alpha_rel
                        .map(|v| (v as f32).clamp(0.0, MAX_PARALRU_R_F32)),
                ),
                alpha_atk: GpuVec4::from_array(
                    self.lru_alpha_atk.map(|v| (v as f32).clamp(0.0, 1.0)),
                ),
                kappa: GpuVec4::from_array(self.lru_kappa.map(|v| (v as f32).clamp(0.0, 1.0))),
                gamma_duhem: GpuVec4::from_array(
                    self.lru_duhem_gamma
                        .map(|v| (v as f32).clamp(0.0, MAX_PARALRU_DUHEM_GAMMA_F32)),
                ),
                signed_mix: GpuVec4::from_array(
                    self.lru_signed_mix.map(|v| (v as f32).clamp(0.0, 1.0)),
                ),
                stereo_link: GpuVec4::from_array(
                    self.lru_stereo_link.map(|v| (v as f32).clamp(0.0, 1.0)),
                ),
            },
            two_svf_g: g.mul(GpuVec4::splat(2.0)),
            rect_sign: GpuVec4::from_array(
                self.rect_sign.map(|v| if v < 0.0 { -1.0 } else { 1.0 }),
            ),
            sag_depth: GpuVec4::from_array(self.sag_depth.map(|v| (v as f32).clamp(-8.0, 8.0))),
            adaa_drive: GpuVec4::from_array(self.adaa_drive.map(|v| (v as f32).clamp(-32.0, 32.0))),
            lru_bias_drive: GpuVec4::from_array(
                self.lru_bias_drive.map(|v| (v as f32).clamp(-16.0, 16.0)),
            ),
            wavefolder_mix: GpuVec4::from_array(
                self.wavefolder_mix.map(|v| (v as f32).clamp(0.0, 1.0)),
            ),
            glu_mix: GpuVec4::from_array(self.glu_mix.map(|v| (v as f32).clamp(-4.0, 4.0))),
        }
    }

    /// Rescales `CoreQuadF64` from reference `1x` parameters to execution rate ratio `rho_core`
    /// (`f_ref / f_core`, `ref[dsp.zoh.op-m-poles]`, `ref[dsp.zoh.core-paralru]`).
    #[must_use]
    pub fn rescaled_for_rho(&self, rho_core: f64) -> Self {
        let mut out = *self;
        let rho = if rho_core.is_finite() && rho_core > 0.0 {
            rho_core.clamp(0.125, 8.0)
        } else {
            1.0
        };
        if (rho - 1.0).abs() <= 1.0e-12 {
            return out;
        }
        for lane in 0..4 {
            out.svf_g[lane] = f64::from(rescale_tpt_svf_g_zoh(self.svf_g[lane] as f32, rho));
            let (rel, atk, kap) = rescale_paralru_lane(
                self.lru_alpha_rel[lane] as f32,
                self.lru_alpha_atk[lane] as f32,
                self.lru_kappa[lane] as f32,
                rho,
            );
            out.lru_alpha_rel[lane] = f64::from(rel).clamp(0.0, f64::from(MAX_PARALRU_R_F32));
            out.lru_alpha_atk[lane] = f64::from(atk).clamp(0.0, 1.0);
            out.lru_kappa[lane] = f64::from(kap).clamp(0.0, 1.0);
        }
        out
    }

    /// Derives exact `1x` `48 kHz` ZOH-rescaled parameters using [`Self::rescaled_for_rho`]
    /// (`ref[dsp.zoh.op-m-poles]`).
    #[must_use]
    pub fn rescaled_1x_zoh(&self, rho_sr: f64) -> Self {
        let mut out = *self;
        let rho = if rho_sr.is_finite() && rho_sr > 0.0 {
            rho_sr.clamp(0.25, 8.0)
        } else {
            2.0
        };
        for lane in 0..4 {
            out.svf_g[lane] = f64::from(rescale_tpt_svf_g_zoh(self.svf_g[lane] as f32, rho));
            out.lru_alpha_rel[lane] =
                f64::from(rescale_real_pole_zoh(self.lru_alpha_rel[lane] as f32, rho))
                    .clamp(0.0, f64::from(MAX_PARALRU_R_F32));
            out.lru_alpha_atk[lane] =
                f64::from(rescale_real_pole_zoh(self.lru_alpha_atk[lane] as f32, rho))
                    .clamp(0.0, 1.0);
        }
        out
    }
}

/// Per-tier independent `LinearReadout` + `ModalConvolver` override solution emitted by Pass 3
/// for prefix-override tiers (`1×` Lite vs `2×` Full/Pro/Ultra) (`ref[container.dnam.tier-prefix]`).
#[derive(Clone, Debug, PartialEq)]
pub struct TierReadoutPrefixF64 {
    /// Pre-convolver linear input bypass gain `[L, R]`.
    pub pre_conv_u_gain: [f64; 2],
    /// Pre-convolver high-pass bypass gain `[L, R]`.
    pub pre_conv_hp_gain: [f64; 2],
    /// Post-convolver S2 direct linear gain `[L, R]`.
    pub direct_s2_u_gain: [f64; 2],
    /// Post-convolver S2 direct high-pass gain `[L, R]`.
    pub direct_s2_hp_gain: [f64; 2],
    /// Linear readout weights `[lane 0..40][col 0..12]`.
    pub w_ro: [[f64; 12]; 40],
    /// Left channel Early FIR taps (`128` max).
    pub fir_l: [f64; NUM_FIR_TAPS_ULTRA],
    /// Right channel Early FIR taps (`128` max).
    pub fir_r: [f64; NUM_FIR_TAPS_ULTRA],
    /// Shared Kautz modal excitation gains `b_gamma`.
    pub kautz_b_gamma: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    /// Left channel Kautz real residues `c_re_l`.
    pub kautz_c_re_l: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    /// Left channel Kautz imag residues `c_im_l`.
    pub kautz_c_im_l: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    /// Right channel Kautz real residues `c_re_r`.
    pub kautz_c_re_r: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    /// Right channel Kautz imag residues `c_im_r`.
    pub kautz_c_im_r: [f64; NUM_KAUTZ_PAIRS_ULTRA],
}

impl Default for TierReadoutPrefixF64 {
    fn default() -> Self {
        Self {
            pre_conv_u_gain: [0.0; 2],
            pre_conv_hp_gain: [0.0; 2],
            direct_s2_u_gain: [0.0; 2],
            direct_s2_hp_gain: [0.0; 2],
            w_ro: [[0.0; 12]; 40],
            fir_l: [0.0; NUM_FIR_TAPS_ULTRA],
            fir_r: [0.0; NUM_FIR_TAPS_ULTRA],
            kautz_b_gamma: [0.0; NUM_KAUTZ_PAIRS_ULTRA],
            kautz_c_re_l: [0.0; NUM_KAUTZ_PAIRS_ULTRA],
            kautz_c_im_l: [0.0; NUM_KAUTZ_PAIRS_ULTRA],
            kautz_c_re_r: [0.0; NUM_KAUTZ_PAIRS_ULTRA],
            kautz_c_im_r: [0.0; NUM_KAUTZ_PAIRS_ULTRA],
        }
    }
}

/// 4×4 inter-block coupling matrices per tile pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterBlockTileF64 {
    /// Odd nonlinear register matrix `[src_lane][dst_lane]`.
    pub w_odd_4x4: [[f64; 4]; 4],
    /// Even rectified/GLU matrix `[src_lane][dst_lane]`.
    pub w_even_4x4: [[f64; 4]; 4],
    /// Fast `ParaLRU` coupling matrix `[src_lane][dst_lane]`.
    pub w_lru_4x4: [[f64; 4]; 4],
}

impl Default for InterBlockTileF64 {
    fn default() -> Self {
        Self {
            w_odd_4x4: [[0.0; 4]; 4],
            w_even_4x4: [[0.0; 4]; 4],
            w_lru_4x4: [[0.0; 4]; 4],
        }
    }
}

/// Continuous complex Kautz conjugate modal pole pair + stereo residues (`ref[dsp.op-m.kautz.coupled-form]`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KautzPairF64 {
    /// Real radial anchor `r_low` in `[0.0, 0.9995]`.
    pub r_low: f64,
    /// Radial spread delta (`r_base = r_low + 0.5 * dr`).
    pub dr: f64,
    /// Carrier frequency `theta` in radians/sample (`48 kHz`).
    pub theta: f64,
    /// Input coupling `b_gamma`.
    pub b_gamma: f64,
    /// Left real residue `c_re_l`.
    pub c_re_l: f64,
    /// Left imag residue `c_im_l`.
    pub c_im_l: f64,
    /// Right real residue `c_re_r`.
    pub c_re_r: f64,
    /// Right imag residue `c_im_r`.
    pub c_im_r: f64,
}

/// 4-lane `Op-T` recurrent transport block (`ref[dsp.op-t.ring.declaration]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportQuadF64 {
    /// Nominal per-lane fractional delays (`>= 5.0` crossbar-coupled samples).
    pub delay_samples: [f64; 4],
    /// Grain/pitch ramp spans in samples.
    pub ramp_spans_samples: [f64; 4],
    /// Loop attenuation scalar (`[0.0, 0.999]`).
    pub g_loop: f64,
    /// DC blocker pole (`TANK_DC_BLOCKER_48K`).
    pub r_dc: f64,
    /// Feedback low-pass damping pole (`[0.0, 0.9995]`).
    pub damp_pole: f64,
    /// LFO rate in Hz (`[0.0, 16.0]`).
    pub lfo_rate_hz: f64,
    /// Pitch/grain ramp rate in Hz (`[-16.0, 16.0]`).
    pub ramp_rate_hz: f64,
    /// Per-lane LFO modulation depths in samples.
    pub lfo_depths: [f64; 4],
    /// Per-lane quad input weights.
    pub input_gains: [f64; 4],
    /// Per-lane Left wet tap gains.
    pub wet_tap_l: [f64; 4],
    /// Per-lane Right wet tap gains.
    pub wet_tap_r: [f64; 4],
    /// Schroeder diffuser coefficients (`[-0.75, 0.75]`).
    pub allpass_g: [f64; 4],
    /// Schroeder allpass delays in samples (`<= 255`).
    pub allpass_delays: [u8; 4],
    /// Givens rotation feedback mixing angle in radians.
    pub fb_psi: f64,
    /// Cubic feedback knee `sigma_sat` (`[1e-3, 0.984375]`).
    pub sigma_knee: f64,
    /// Passive continuous recirculation mix `[0.0, 1.0]`.
    pub freeze_mix: f64,
    /// Hann crossfade window depth `[0.0, 1.0]`.
    pub window_depth: f64,
    /// Enable Schroeder allpass chain.
    pub schroeder_enable: bool,
    /// Use external large f32 ring buffer (`ext_ring_pow2_frames >= 7`).
    pub ext_f32: bool,
}

impl Default for TransportQuadF64 {
    fn default() -> Self {
        Self {
            delay_samples: [MIN_DELAY_CROSSBAR_SAMP.max(16.0); 4],
            ramp_spans_samples: [0.0; 4],
            g_loop: 0.0,
            r_dc: f64::from(TANK_DC_BLOCKER_48K_F32),
            damp_pole: 0.0,
            lfo_rate_hz: 0.0,
            ramp_rate_hz: 0.0,
            lfo_depths: [0.0; 4],
            input_gains: [0.0; 4],
            wet_tap_l: [0.0; 4],
            wet_tap_r: [0.0; 4],
            allpass_g: [0.0; 4],
            allpass_delays: [0; 4],
            fb_psi: 0.0,
            sigma_knee: 0.984_375,
            freeze_mix: 0.0,
            window_depth: 0.0,
            schroeder_enable: false,
            ext_f32: false,
        }
    }
}

/// Complete universal 4-tier causally nested `SONA v3.0` model in host `binary64`.
#[derive(Clone, Debug)]
pub struct SonaModelF64 {
    /// Reference training sample rate in Hz (`48_000.0` canonical).
    pub sample_rate_hz: f64,
    /// 10 Ultra causally nested `CoreQuad` blocks (`0..4` Lite, `4..6` Full, `6..8` Pro, `8..10` Ultra).
    pub core_quads: [CoreQuadF64; NUM_CORE_QUADS_ULTRA],
    /// 35 inter-block coupling tile pairs (`0..3` Lite, `3..13` Full, `13..23` Pro, `23..35` Ultra).
    pub tiles: [InterBlockTileF64; NUM_INTER_BLOCK_TILES_ULTRA],
    /// `Op-N[Pre]` wet bitcrush depth `[0.0, 1.0]`.
    pub op_n_pre_gamma_bc: f64,
    /// `Op-N[Pre]` ZOH hold depth `[0.0, 0.9999]`.
    pub op_n_pre_zoh_hold_depth: f64,
    /// `Op-N[Pre]` quantizer sharpness `beta_q` (`[0.0, 16.0]`).
    pub op_n_pre_beta_q: f64,
    /// Crossbar `Op-P` and `Op-T` feed gains `[op_p, op_t]`.
    pub crossbar_op_gains: [f64; 2],
    /// Top-of-frame `Q0.32` phasor frequencies in Hz `[4]`.
    pub phasor_freq_hz: [f64; 4],
    /// Carrier 3-basis projection weights `[3][4]`.
    pub carrier_basis_w: [[f64; 4]; 3],
    /// Direct pre-convolver linear input residual bypass `[L, R]`.
    pub pre_conv_u_gain: [f64; 2],
    /// Direct pre-convolver high-pass residual gain `[L, R]`.
    pub pre_conv_hp_gain: [f64; 2],
    /// Direct post-convolver S2 linear gain `[L, R]`.
    pub direct_s2_u_gain: [f64; 2],
    /// Direct post-convolver S2 high-pass gain `[L, R]`.
    pub direct_s2_hp_gain: [f64; 2],
    /// Learned `48 kHz` `LinearReadout` leak pole `ro_pole_48k` (`[0.0, 0x3F7F_FFAC]`).
    pub ro_pole_48k: f64,
    /// Ultra-nested Left readout row matrix `[40][12]` (Lite uses `0..16`, Full `0..24`, Pro `0..32`).
    pub w_ro: [[f64; 12]; 40],
    /// `ModalConvolver` Left early FIR (`128` taps).
    pub fir_l: [f64; NUM_FIR_TAPS_ULTRA],
    /// `ModalConvolver` Right early FIR (`128` taps).
    pub fir_r: [f64; NUM_FIR_TAPS_ULTRA],
    /// `ModalConvolver` Kautz shared-pole pairs (`32` pairs).
    pub kautz: [KautzPairF64; NUM_KAUTZ_PAIRS_ULTRA],
    /// Active Kautz pair count actually fitted (`<= 32`).
    pub active_kautz_pairs: usize,
    /// Active FIR tap count (`<= 128`).
    pub active_fir_taps: usize,
    /// `Op-T` transport quads `0..4`.
    pub transport_quads: [TransportQuadF64; NUM_TRANSPORT_QUADS_ULTRA],
    /// Per-tier output loudness calibration scalars `[Lite, Full, Pro, Ultra]`.
    pub tier_gain_trim: [f64; 4],
    /// Optional Phase-1 audited control space metadata.
    pub control_audit: Option<Phase1ClampOutput>,
    /// Optional fitted `CP1` + `"CEXT"` + `"TSYN"` control bus bundle.
    pub control_bus: Option<Cp1WireBundle>,
    /// Control-space stability specification captured during Phase-1 audit.
    pub stability_constraint: StageStabilityConstraint,
    /// Optional `SensoryQuad` parameters `[svf_g, rel, atk, k, kappa, in_l, in_r, vca_w]`.
    pub sense_svf_g: [f64; 4],
    /// `SensoryQuad` release poles.
    pub sense_lru_rel: [f64; 4],
    /// `SensoryQuad` attack poles.
    pub sense_lru_atk: [f64; 4],
    /// `SensoryQuad` SVF damping factors.
    pub sense_svf_k: [f64; 4],
    /// `SensoryQuad` ParaLRU kappa.
    pub sense_lru_kappa: [f64; 4],
    /// `SensoryQuad` input VCA weights.
    pub sense_vca_w: [f64; 4],
    /// `SensoryQuad` Left stereo key weights `[-4.0, 4.0]`.
    pub sense_in_l: [f64; 4],
    /// `SensoryQuad` Right stereo key weights `[-4.0, 4.0]`.
    pub sense_in_r: [f64; 4],
    /// `SensoryQuad` TPT-SVF low-pass weights `[-16.0, 16.0]`.
    pub sense_tone_lp: [f64; 4],
    /// `SensoryQuad` TPT-SVF band-pass weights `[-16.0, 16.0]`.
    pub sense_tone_bp: [f64; 4],
    /// `SensoryQuad` TPT-SVF high-pass weights `[-16.0, 16.0]`.
    pub sense_tone_hp: [f64; 4],
    /// Crossbar `Op-P` -> core weights `[10][4]`.
    pub op_p_to_core: [[f64; 4]; NUM_CORE_QUADS_ULTRA],
    /// Crossbar `Q0.32` phasor -> core drive weights `[10][4]`.
    pub phasor_to_core: [[f64; 4]; NUM_CORE_QUADS_ULTRA],
    /// Crossbar `Opt` -> `[L, R]` direct matrix `[16][2]`.
    pub opt_to_lr: [[f64; 2]; 16],
    /// Stereo phasor carrier offsets in `[-pi, pi]`.
    pub phasor_stereo_phase: [f64; 4],
    /// Stage-level `Op-T` dry pass-through mix `[-2.0, 2.0]` (`1.0` canonical).
    pub tank_dry_mix: f64,
    /// Run `Full/Pro/Ultra` at `2×` core oversampling while keeping `Lite` at `1×`.
    pub oversample_upper_tiers_2x: bool,
    /// `2×` upsampler quadratic interpolation coefficient (`-0.125` canonical).
    pub os_up_quad_coeff: f64,
    /// `2×` decimator `x[n-1]` weight (`0.15` canonical).
    pub os_dec_c_prev1: f64,
    /// `2×` decimator `x[n]` weight (`0.50` canonical).
    pub os_dec_c0: f64,
    /// Independent Pass-3 per-tier readout + convolver prefix overrides `[Lite..Ultra]`.
    pub tier_overrides: [Option<Box<TierReadoutPrefixF64>>; 4],
    /// Per-tier flags indicating whether `SONA_META_CONVOLVER_PREFIX_OVERRIDES` is needed.
    pub prefix_override_needed: [bool; 4],
}

impl Default for SonaModelF64 {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

#[inline]
fn sanitize_clamp(v: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

fn clamp_core_quad_f64(q: &mut CoreQuadF64) {
    let g_lo = f64::from(MIN_TPT_G);
    let r_hi = f64::from(MAX_PARALRU_R_F32);
    let duh_hi = f64::from(MAX_PARALRU_DUHEM_GAMMA_F32);
    for l in 0..4 {
        q.svf_g[l] = sanitize_clamp(q.svf_g[l], g_lo, 64.0, 0.125);
        q.svf_k[l] = sanitize_clamp(q.svf_k[l], 0.0, 8.0, core::f64::consts::FRAC_1_SQRT_2);
        q.tone_lp[l] = sanitize_clamp(q.tone_lp[l], -16.0, 16.0, 0.0);
        q.tone_bp[l] = sanitize_clamp(q.tone_bp[l], -16.0, 16.0, 0.0);
        q.tone_hp[l] = sanitize_clamp(q.tone_hp[l], -16.0, 16.0, 0.0);
        q.tone_ap[l] = sanitize_clamp(q.tone_ap[l], -16.0, 16.0, 0.0);
        q.tone_dir[l] = sanitize_clamp(q.tone_dir[l], -16.0, 16.0, 0.0);
        q.lru_alpha_rel[l] = sanitize_clamp(q.lru_alpha_rel[l], 0.0, r_hi, 0.995);
        q.lru_alpha_atk[l] = sanitize_clamp(q.lru_alpha_atk[l], 0.0, 1.0, 0.25);
        q.lru_kappa[l] = sanitize_clamp(q.lru_kappa[l], 0.0, 1.0, 0.0);
        q.lru_duhem_gamma[l] = sanitize_clamp(q.lru_duhem_gamma[l], 0.0, duh_hi, 0.0);
        q.lru_signed_mix[l] = sanitize_clamp(q.lru_signed_mix[l], 0.0, 1.0, 0.0);
        q.lru_stereo_link[l] = sanitize_clamp(q.lru_stereo_link[l], 0.0, 1.0, 0.0);
        q.rect_sign[l] = if q.rect_sign[l].is_finite() && q.rect_sign[l] < 0.0 {
            -1.0
        } else {
            1.0
        };
        q.sag_depth[l] = sanitize_clamp(q.sag_depth[l], -8.0, 8.0, 0.0);
        q.adaa_drive[l] = sanitize_clamp(q.adaa_drive[l], -32.0, 32.0, 0.0);
        q.lru_bias_drive[l] = sanitize_clamp(q.lru_bias_drive[l], -16.0, 16.0, 0.0);
        q.wavefolder_mix[l] = sanitize_clamp(q.wavefolder_mix[l], 0.0, 1.0, 0.0);
        q.glu_mix[l] = sanitize_clamp(q.glu_mix[l], -4.0, 4.0, 0.0);
        q.in_mid_weight[l] = sanitize_clamp(q.in_mid_weight[l], -16.0, 16.0, 0.0);
        q.in_side_weight[l] = sanitize_clamp(q.in_side_weight[l], -16.0, 16.0, 0.0);
    }
}

fn clamp_prefix_f64(pfx: &mut TierReadoutPrefixF64) {
    for c in 0..2 {
        pfx.pre_conv_u_gain[c] = sanitize_clamp(pfx.pre_conv_u_gain[c], -64.0, 64.0, 0.0);
        pfx.pre_conv_hp_gain[c] = sanitize_clamp(pfx.pre_conv_hp_gain[c], -64.0, 64.0, 0.0);
        pfx.direct_s2_u_gain[c] = sanitize_clamp(pfx.direct_s2_u_gain[c], -64.0, 64.0, 0.0);
        pfx.direct_s2_hp_gain[c] = sanitize_clamp(pfx.direct_s2_hp_gain[c], -64.0, 64.0, 0.0);
    }
    for row in &mut pfx.w_ro {
        for cell in row {
            *cell = sanitize_clamp(*cell, -64.0, 64.0, 0.0);
        }
    }
    for tap in &mut pfx.fir_l {
        *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
    }
    for tap in &mut pfx.fir_r {
        *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
    }
    for k in 0..NUM_KAUTZ_PAIRS_ULTRA {
        pfx.kautz_b_gamma[k] = sanitize_clamp(pfx.kautz_b_gamma[k], 0.0, 8.0, 0.0);
        pfx.kautz_c_re_l[k] = sanitize_clamp(pfx.kautz_c_re_l[k], -8.0, 8.0, 0.0);
        pfx.kautz_c_im_l[k] = sanitize_clamp(pfx.kautz_c_im_l[k], -8.0, 8.0, 0.0);
        pfx.kautz_c_re_r[k] = sanitize_clamp(pfx.kautz_c_re_r[k], -8.0, 8.0, 0.0);
        pfx.kautz_c_im_r[k] = sanitize_clamp(pfx.kautz_c_im_r[k], -8.0, 8.0, 0.0);
    }
}

impl SonaModelF64 {
    /// Initializes a zero-DC neutral black-box `SonaModelF64` at `sample_rate_hz`.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let sr = sanitize_clamp(sample_rate_hz, 44_100.0, 96_000.0, 48_000.0);
        Self {
            sample_rate_hz: sr,
            core_quads: [CoreQuadF64::default(); NUM_CORE_QUADS_ULTRA],
            tiles: [InterBlockTileF64::default(); NUM_INTER_BLOCK_TILES_ULTRA],
            op_n_pre_gamma_bc: 0.0,
            op_n_pre_zoh_hold_depth: 0.0,
            op_n_pre_beta_q: 0.0,
            crossbar_op_gains: [0.0; 2],
            phasor_freq_hz: [0.0; 4],
            carrier_basis_w: [[0.0; 4]; 3],
            pre_conv_u_gain: [0.0; 2],
            pre_conv_hp_gain: [0.0; 2],
            direct_s2_u_gain: [1.0, 1.0],
            direct_s2_hp_gain: [0.0; 2],
            ro_pole_48k: 0.0,
            w_ro: [[0.0; 12]; 40],
            fir_l: [0.0; NUM_FIR_TAPS_ULTRA],
            fir_r: [0.0; NUM_FIR_TAPS_ULTRA],
            kautz: [KautzPairF64::default(); NUM_KAUTZ_PAIRS_ULTRA],
            active_kautz_pairs: 0,
            active_fir_taps: 0,
            transport_quads: [TransportQuadF64::default(); NUM_TRANSPORT_QUADS_ULTRA],
            tier_gain_trim: [1.0; 4],
            control_audit: None,
            control_bus: None,
            stability_constraint: StageStabilityConstraint::default(),
            sense_svf_g: [0.25; 4],
            sense_lru_rel: [0.995; 4],
            sense_lru_atk: [0.25; 4],
            sense_svf_k: [core::f64::consts::FRAC_1_SQRT_2; 4],
            sense_lru_kappa: [0.0; 4],
            sense_vca_w: [0.0; 4],
            sense_in_l: [0.5; 4],
            sense_in_r: [0.5; 4],
            sense_tone_lp: [1.0; 4],
            sense_tone_bp: [0.0; 4],
            sense_tone_hp: [0.0; 4],
            op_p_to_core: [[0.0; 4]; NUM_CORE_QUADS_ULTRA],
            phasor_to_core: [[0.0; 4]; NUM_CORE_QUADS_ULTRA],
            opt_to_lr: [[0.0; 2]; 16],
            phasor_stereo_phase: [0.0; 4],
            tank_dry_mix: 1.0,
            oversample_upper_tiers_2x: true,
            os_up_quad_coeff: -0.125,
            os_dec_c_prev1: 0.15,
            os_dec_c0: 0.50,
            tier_overrides: [None, None, None, None],
            prefix_override_needed: [false; 4],
        }
    }

    /// Packs the first `num_quads` `CoreQuad` blocks into flat `[[f32; 4]]` runner parameter buffers.
    pub fn pack_core_quads_for_runner(
        &self,
        num_quads: usize,
    ) -> Result<Vec<[f32; 4]>, TrainError> {
        if num_quads > NUM_CORE_QUADS_ULTRA {
            return Err(TrainError::InvalidDimensions {
                context: "SonaModelF64::pack_core_quads_for_runner",
                expected: NUM_CORE_QUADS_ULTRA,
                actual: num_quads,
            });
        }
        let mut flat = Vec::with_capacity(num_quads * QUAD_PARAMS_VEC4_STRIDE);
        for q in &self.core_quads[..num_quads] {
            flat.extend_from_slice(&pack_core_quad_params(&q.to_kernel_params()));
        }
        Ok(flat)
    }

    /// Clamps learned `ro_pole_48k`, `CoreQuad` poles/drives, `Op-N[Pre]`, `Op-T` delays,
    /// `Kautz` poles, `w_ro`/`fir`, and `tier_gain_trim` cleanly onto `dnam-vm` invariants.
    pub fn clamp_all_stability_bounds(&mut self, crossbar_coupled: bool) {
        self.ro_pole_48k = sanitize_clamp(self.ro_pole_48k, 0.0, MAX_RO_POLE_48K_F64, 0.0);
        self.op_n_pre_gamma_bc = sanitize_clamp(self.op_n_pre_gamma_bc, 0.0, 1.0, 0.0);
        self.op_n_pre_zoh_hold_depth =
            sanitize_clamp(self.op_n_pre_zoh_hold_depth, 0.0, 0.9999, 0.0);
        self.op_n_pre_beta_q = sanitize_clamp(self.op_n_pre_beta_q, 0.0, 16.0, 0.0);
        self.tank_dry_mix = sanitize_clamp(self.tank_dry_mix, -2.0, 2.0, 1.0);
        self.os_up_quad_coeff = sanitize_clamp(self.os_up_quad_coeff, -0.20, 0.05, -0.125);
        self.os_dec_c_prev1 = sanitize_clamp(self.os_dec_c_prev1, 0.0, 0.28, 0.15);
        self.os_dec_c0 = sanitize_clamp(self.os_dec_c0, 0.0, 0.55, 0.50);
        for t in &mut self.tier_gain_trim {
            *t = sanitize_clamp(*t, 0.0625, 16.0, 1.0);
        }
        for q in &mut self.core_quads {
            clamp_core_quad_f64(q);
        }
        for c in 0..2 {
            self.pre_conv_u_gain[c] = sanitize_clamp(self.pre_conv_u_gain[c], -64.0, 64.0, 0.0);
            self.pre_conv_hp_gain[c] = sanitize_clamp(self.pre_conv_hp_gain[c], -64.0, 64.0, 0.0);
            self.direct_s2_u_gain[c] = sanitize_clamp(self.direct_s2_u_gain[c], -64.0, 64.0, 0.0);
            self.direct_s2_hp_gain[c] = sanitize_clamp(self.direct_s2_hp_gain[c], -64.0, 64.0, 0.0);
        }
        for row in &mut self.w_ro {
            for cell in row {
                *cell = sanitize_clamp(*cell, -64.0, 64.0, 0.0);
            }
        }
        for tap in &mut self.fir_l {
            *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
        }
        for tap in &mut self.fir_r {
            *tap = sanitize_clamp(*tap, -16.0, 16.0, 0.0);
        }
        let tau_floor = if crossbar_coupled {
            MIN_DELAY_CROSSBAR_SAMP
        } else {
            4.0
        };
        let g_loop_max = f64::from(TANK_MAX_G_LIN_F32);
        for tq in &mut self.transport_quads {
            for d in &mut tq.delay_samples {
                *d = sanitize_clamp(*d, tau_floor, 48_000.0, tau_floor);
            }
            for lfo in &mut tq.lfo_depths {
                *lfo = sanitize_clamp(*lfo, 0.0, 512.0, 0.0);
            }
            tq.g_loop = sanitize_clamp(tq.g_loop, 0.0, g_loop_max, 0.0);
            tq.damp_pole = sanitize_clamp(tq.damp_pole, 0.0, 0.9995, 0.0);
            tq.sigma_knee = sanitize_clamp(tq.sigma_knee, 1.0e-3, 0.984_375, 0.984_375);
            tq.freeze_mix = sanitize_clamp(tq.freeze_mix, 0.0, 1.0, 0.0);
            tq.window_depth = sanitize_clamp(tq.window_depth, 0.0, 1.0, 0.0);
        }
        let theta_max = f64::from(core::f32::consts::PI - 1.0e-4);
        for k in &mut self.kautz {
            k.r_low = sanitize_clamp(k.r_low, 0.0, 0.9995, 0.0);
            k.dr = sanitize_clamp(k.dr, 0.0, 0.9995 - k.r_low, 0.0);
            k.theta = sanitize_clamp(k.theta, 0.0, theta_max, 0.0);
            k.b_gamma = sanitize_clamp(k.b_gamma, 0.0, 8.0, 0.0);
            k.c_re_l = sanitize_clamp(k.c_re_l, -8.0, 8.0, 0.0);
            k.c_im_l = sanitize_clamp(k.c_im_l, -8.0, 8.0, 0.0);
            k.c_re_r = sanitize_clamp(k.c_re_r, -8.0, 8.0, 0.0);
            k.c_im_r = sanitize_clamp(k.c_im_r, -8.0, 8.0, 0.0);
        }
        for ov in &mut self.tier_overrides {
            if let Some(pfx) = ov.as_deref_mut() {
                clamp_prefix_f64(pfx);
            }
        }
    }
}

/// Borrowed stereo training input/target slice bundle (`in_l, in_r, target_l, target_r`).
#[derive(Clone, Copy, Debug)]
pub struct StereoTrainingPairs<'a> {
    /// Dry Left channel input excitation.
    pub in_l: &'a [f64],
    /// Dry Right channel input excitation.
    pub in_r: &'a [f64],
    /// Aligned Left wet target waveform.
    pub target_l: &'a [f64],
    /// Aligned Right wet target waveform.
    pub target_r: &'a [f64],
}

impl<'a> StereoTrainingPairs<'a> {
    /// Returns the common aligned frame count across all 4 channels.
    #[inline]
    #[must_use]
    pub fn aligned_len(&self) -> usize {
        self.in_l
            .len()
            .min(self.in_r.len())
            .min(self.target_l.len())
            .min(self.target_r.len())
    }
}

/// Plans via [`plan_core_scan_execution`] and executes `fused_core_scan_256` on `runner`.
pub(crate) fn run_planned_core_scan(
    runner: &TrainRunner,
    num_frames: usize,
    num_quads: usize,
    num_stems: usize,
    include_adjoint: bool,
    core_inputs: &[[f32; 4]],
    quad_params: &[[f32; 4]],
) -> Result<FusedCoreScanOutput, TrainError> {
    let adapter_cap = match runner.active_backend() {
        ActiveComputeBackend::CpuSimd { .. } => None,
        #[cfg(feature = "gpu-wgpu")]
        ActiveComputeBackend::Wgpu {
            max_storage_binding_bytes,
            ..
        } => Some(max_storage_binding_bytes),
    };
    let plan = plan_core_scan_execution(
        CoreScanWorkload {
            num_frames,
            num_quads,
            num_stems,
            include_adjoint,
        },
        MemoryBudgetProfile::DesktopCapsule,
        adapter_cap,
    )?;
    runner.run_fused_core_scan_planned(
        &FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_frames.div_ceil(TILE_FRAMES) as u32,
                num_quads: num_quads as u32,
                num_stems: num_stems as u32,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs,
            quad_params,
            initial_checkpoints: None,
        },
        &plan,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sona_model_zoh_rescaling_packing_and_clamps() {
        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].svf_g = [0.25; 4];
        model.core_quads[0].lru_alpha_rel = [0.995; 4];
        let zoh_1x = model.core_quads[0].rescaled_1x_zoh(2.0);
        assert!(zoh_1x.lru_alpha_rel[0] < model.core_quads[0].lru_alpha_rel[0]);
        assert!(zoh_1x.svf_g[0] > 0.0 && zoh_1x.svf_g[0].is_finite());

        let rho_2x = model.core_quads[0].rescaled_for_rho(0.5);
        assert!(rho_2x.lru_alpha_rel[0] > model.core_quads[0].lru_alpha_rel[0]);
        assert!(rho_2x.svf_g[0] < model.core_quads[0].svf_g[0]);

        let packed = model.pack_core_quads_for_runner(4).unwrap();
        assert_eq!(packed.len(), 4 * QUAD_PARAMS_VEC4_STRIDE);

        model.ro_pole_48k = 1.0;
        model.transport_quads[0].delay_samples[0] = 1.0;
        model.transport_quads[0].g_loop = 0.9999;
        model.core_quads[0].svf_g[0] = f64::NAN;
        model.core_quads[0].svf_g[1] = 1.0e-9;
        model.tank_dry_mix = f64::NAN;
        model.clamp_all_stability_bounds(true);
        assert_eq!(model.ro_pole_48k, MAX_RO_POLE_48K_F64);
        assert_eq!(model.transport_quads[0].delay_samples[0], 5.0);
        assert_eq!(
            model.transport_quads[0].g_loop,
            f64::from(TANK_MAX_G_LIN_F32)
        );
        assert_eq!(model.core_quads[0].svf_g[0], 0.125);
        assert_eq!(model.core_quads[0].svf_g[1], f64::from(MIN_TPT_G));
        assert_eq!(model.tank_dry_mix, 1.0);
    }
}
