//! `f64` structural parameter bundles for `CoreQuad`, `InterBlockTile`, `TierReadoutPrefix`,
//! `KautzPair`, and `Op-T` `TransportQuad` (`ref[dsp.op-n.core.matryoshka-blocks]`).

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, KernelVec4, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32,
    MAX_SVF_K_NL, MIN_TPT_G, SonaParaLruDuhemCoeffs4, SonaToneWeights4, TptSvfCoeffs4,
};
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, rescale_real_pole_zoh, rescale_tpt_svf_g_zoh};
use dnam_vm::rescale_paralru_lane;

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;

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
    /// Block 0 linearly-implicit PHS nonlinear SVF damping `k_nl` (`[0.0, 16.0]`; `0.0` on Blocks 1..3).
    pub svf_k_nl: [f64; 4],
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
            svf_k_nl: [0.0; 4],
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
    /// Converts `self` into `32-bit` GPU/CPU uniform params for `block_idx`, strictly masking
    /// `svf_k_nl` to `0.0` on linear Blocks `1..=3` (`ref[dsp.op-n.core.tpt-tone-basis]`).
    #[must_use]
    pub fn to_kernel_params_for_block(&self, block_idx: u32) -> CoreQuadParamsStereo4<GpuVec4> {
        #[inline]
        fn clamp_f32_finite(v: f64, lo: f32, hi: f32, fallback: f32) -> f32 {
            if v.is_finite() {
                (v as f32).clamp(lo, hi)
            } else {
                fallback
            }
        }
        let g = GpuVec4::from_array(
            self.svf_g
                .map(|v| clamp_f32_finite(v, MIN_TPT_G, 64.0, 0.125)),
        );
        let k = GpuVec4::from_array(
            self.svf_k
                .map(|v| clamp_f32_finite(v, 0.0, 8.0, core::f32::consts::FRAC_1_SQRT_2)),
        );
        let svf_k_nl = if block_idx == 0 {
            GpuVec4::from_array(
                self.svf_k_nl
                    .map(|v| clamp_f32_finite(v, 0.0, MAX_SVF_K_NL, 0.0)),
            )
        } else {
            GpuVec4::from_array([0.0; 4])
        };
        CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::from_array(
                    self.tone_lp.map(|v| clamp_f32_finite(v, -16.0, 16.0, 0.0)),
                ),
                bp: GpuVec4::from_array(
                    self.tone_bp.map(|v| clamp_f32_finite(v, -16.0, 16.0, 0.0)),
                ),
                hp: GpuVec4::from_array(
                    self.tone_hp.map(|v| clamp_f32_finite(v, -16.0, 16.0, 0.0)),
                ),
                ap: GpuVec4::from_array(
                    self.tone_ap.map(|v| clamp_f32_finite(v, -16.0, 16.0, 0.0)),
                ),
                dir: GpuVec4::from_array(
                    self.tone_dir.map(|v| clamp_f32_finite(v, -16.0, 16.0, 0.0)),
                ),
            },
            lru: SonaParaLruDuhemCoeffs4 {
                alpha_rel: GpuVec4::from_array(
                    self.lru_alpha_rel
                        .map(|v| clamp_f32_finite(v, 0.0, MAX_PARALRU_R_F32, 0.995)),
                ),
                alpha_atk: GpuVec4::from_array(
                    self.lru_alpha_atk
                        .map(|v| clamp_f32_finite(v, 0.0, 1.0, 0.25)),
                ),
                kappa: GpuVec4::from_array(
                    self.lru_kappa.map(|v| clamp_f32_finite(v, 0.0, 1.0, 0.0)),
                ),
                gamma_duhem: GpuVec4::from_array(
                    self.lru_duhem_gamma
                        .map(|v| clamp_f32_finite(v, 0.0, MAX_PARALRU_DUHEM_GAMMA_F32, 0.0)),
                ),
                signed_mix: GpuVec4::from_array(
                    self.lru_signed_mix
                        .map(|v| clamp_f32_finite(v, 0.0, 1.0, 0.0)),
                ),
                stereo_link: GpuVec4::from_array(
                    self.lru_stereo_link
                        .map(|v| clamp_f32_finite(v, 0.0, 1.0, 0.0)),
                ),
            },
            two_svf_g: g.mul(GpuVec4::from_array([2.0; 4])),
            svf_k_nl,
            rect_sign: GpuVec4::from_array(
                self.rect_sign
                    .map(|v| if v.is_finite() && v < 0.0 { -1.0 } else { 1.0 }),
            ),
            sag_depth: GpuVec4::from_array(
                self.sag_depth.map(|v| clamp_f32_finite(v, -8.0, 8.0, 0.0)),
            ),
            adaa_drive: GpuVec4::from_array(
                self.adaa_drive
                    .map(|v| clamp_f32_finite(v, -32.0, 32.0, 0.0)),
            ),
            lru_bias_drive: GpuVec4::from_array(
                self.lru_bias_drive
                    .map(|v| clamp_f32_finite(v, -16.0, 16.0, 0.0)),
            ),
            wavefolder_mix: GpuVec4::from_array(
                self.wavefolder_mix
                    .map(|v| clamp_f32_finite(v, 0.0, 1.0, 0.0)),
            ),
            glu_mix: GpuVec4::from_array(self.glu_mix.map(|v| clamp_f32_finite(v, -4.0, 4.0, 0.0))),
        }
    }

    /// Converts `self` into the single-source `32-bit` GPU/CPU Block 0 `CoreQuadParamsStereo4<GpuVec4>`.
    #[must_use]
    pub fn to_kernel_params(&self) -> CoreQuadParamsStereo4<GpuVec4> {
        self.to_kernel_params_for_block(0)
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
    /// Schroeder allpass delays in samples (`<= 255`; `0` isolates pure `SO(4)` identity).
    pub allpass_delays: [u8; 4],
    /// Givens rotation feedback mixing angle in radians (`[-pi, pi]`).
    pub fb_psi: f64,
    /// Cubic feedback knee `sigma_sat` (`[1e-3, 0.984375]`).
    pub sigma_knee: f64,
    /// Passive continuous recirculation mix `[0.0, 1.0]`.
    pub freeze_mix: f64,
    /// Hann crossfade window depth `[0.0, 1.0]`.
    pub window_depth: f64,
    /// Base `SO(4)` paraunitary diffuser scattering rotor angle in radians (`[-pi, pi]`, `0.0` = identity).
    pub scat_psi: f64,
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
            scat_psi: 0.0,
            ext_f32: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::KernelVec4;

    #[test]
    fn test_core_quad_k_nl_block_masking_and_rate_invariance() {
        let q = CoreQuadF64 {
            svf_k_nl: [1.5, 22.0, -0.5, f64::NAN],
            ..CoreQuadF64::default()
        };
        let p0 = q.to_kernel_params_for_block(0);
        assert_eq!(p0.svf_k_nl.to_array(), [1.5, MAX_SVF_K_NL, 0.0, 0.0]);
        let p1 = q.to_kernel_params_for_block(1);
        assert_eq!(p1.svf_k_nl.to_array(), [0.0; 4]);
        assert_eq!(q.to_kernel_params().svf_k_nl, p0.svf_k_nl);

        let rescaled = q.rescaled_for_rho(0.5);
        assert_eq!(&rescaled.svf_k_nl[..3], &q.svf_k_nl[..3]);
        assert!(rescaled.svf_k_nl[3].is_nan());
        let zoh = q.rescaled_1x_zoh(2.0);
        assert_eq!(&zoh.svf_k_nl[..3], &q.svf_k_nl[..3]);
        assert!(zoh.svf_k_nl[3].is_nan());

        let q_poison = CoreQuadF64 {
            svf_g: [f64::NAN, -1.0, 128.0, 0.2],
            svf_k: [f64::NAN, -2.0, 20.0, 0.5],
            tone_lp: [f64::NAN, -99.0, 99.0, 1.0],
            tone_bp: [f64::NAN; 4],
            tone_hp: [f64::INFINITY; 4],
            tone_ap: [f64::NEG_INFINITY; 4],
            tone_dir: [f64::NAN; 4],
            lru_alpha_rel: [f64::NAN; 4],
            lru_alpha_atk: [f64::NAN; 4],
            lru_kappa: [f64::NAN; 4],
            lru_duhem_gamma: [f64::NAN; 4],
            lru_signed_mix: [f64::NAN; 4],
            lru_stereo_link: [f64::NAN; 4],
            rect_sign: [f64::NAN, -2.0, 3.0, -0.0],
            sag_depth: [f64::NAN; 4],
            adaa_drive: [f64::NAN; 4],
            lru_bias_drive: [f64::NAN; 4],
            wavefolder_mix: [f64::NAN; 4],
            glu_mix: [f64::NAN; 4],
            ..CoreQuadF64::default()
        };
        let p_clean = q_poison.to_kernel_params_for_block(0);
        assert!(p_clean.tone.lp.to_array().iter().all(|v| v.is_finite()));
        assert_eq!(p_clean.tone.lp.to_array()[1..3], [-16.0, 16.0]);
        assert!(p_clean.svf.k.to_array().iter().all(|v| v.is_finite()));
        assert!(
            p_clean
                .lru
                .alpha_rel
                .to_array()
                .iter()
                .all(|v| v.is_finite())
        );
        assert_eq!(p_clean.rect_sign.to_array(), [1.0, -1.0, 1.0, 1.0]);
        assert!(p_clean.adaa_drive.to_array().iter().all(|v| v.is_finite()));
        assert!(p_clean.glu_mix.to_array().iter().all(|v| v.is_finite()));
    }
}
