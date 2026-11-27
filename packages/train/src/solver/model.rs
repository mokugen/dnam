//! Double-precision (`f64`) universal `SONA v3.0` state container ([`SonaModelF64`]),
//! rate-ratio ZOH rescaling, causal quad cascade, stability clamping, wire quantization
//! snapping, and GPU/CPU [`dnam_kernels::CoreQuadParamsStereo4`] packing.

pub(crate) mod cascade;
mod clamp_and_quant;
mod quads;
pub mod stage_latent;
pub(crate) mod v3_core_bridge;

pub use clamp_and_quant::{snap_model_fp16, snap_non_readout_fp16};
pub use quads::{
    CoreQuadF64, InterBlockTileF64, KautzPairF64, NUM_CORE_QUADS_ULTRA, NUM_FIR_TAPS_ULTRA,
    NUM_INTER_BLOCK_TILES_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, NUM_TRANSPORT_QUADS_ULTRA,
    TierReadoutPrefixF64, TransportQuadF64,
};
pub use stage_latent::SonaStageLatentModsF64;

use dnam_kernels::SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL;

use crate::control::clamp::{Phase1ClampOutput, StageStabilityConstraint};
use crate::control::cp1::Cp1WireBundle;
use crate::error::TrainError;
use crate::gpu::abi::QUAD_PARAMS_VEC4_STRIDE;
use crate::runner::pack_core_quad_params;

use clamp_and_quant::{clamp_model_stability_bounds, sanitize_clamp};

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
    /// Initial `Q0.32` phasor carrier phases `[4]`.
    pub phasor_init_phase_q32: [u32; 4],
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
    /// `Op-T` `48 kHz` stereo wet DF-I DC blocker pole (`(0.8, 0x3F7F_FFAC]`, `5 Hz` default).
    pub dc_blocker_pole_5hz: f64,
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
    /// Trainable `stage_latent_mods_fp16`, `stage_latent_ctrl_ultra_fp16`, and
    /// `SONA_META_CORE_MODS_PRESENT` `stage_latent_core_mods_fp16` sensitivity tables.
    pub stage_latent_mods: SonaStageLatentModsF64,
    /// Runtime-only `SensoryQuad` slot-30 threshold bias (`[-1.0, 1.0]`, not wire-serialized).
    pub sense_vca_bias: f64,
    /// Runtime-only `Op-T` slot-15 wet scaling (`[0.0, 2.0]`, `1.0` default, not wire-serialized).
    pub op_t_wet_scale: f64,
}

impl Default for SonaModelF64 {
    fn default() -> Self {
        Self::new(48_000.0)
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
            phasor_init_phase_q32: [0; 4],
            carrier_basis_w: [[0.0; 4]; 3],
            pre_conv_u_gain: [0.0; 2],
            pre_conv_hp_gain: [0.0; 2],
            direct_s2_u_gain: [1.0, 1.0],
            direct_s2_hp_gain: [0.0; 2],
            ro_pole_48k: 0.0,
            dc_blocker_pole_5hz: f64::from(dnam_vm::math::TANK_DC_BLOCKER_48K_F32),
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
            stage_latent_mods: SonaStageLatentModsF64::default(),
            sense_vca_bias: 0.0,
            op_t_wet_scale: 1.0,
        }
    }

    /// Packs the first `num_quads` `CoreQuad` blocks into flat `[[f32; 4]]` runner parameter buffers,
    /// masking `SVF_K_NL` to zero on linear Blocks `1..=3`.
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
        for (q_idx, q) in self.core_quads[..num_quads].iter().enumerate() {
            let block_idx = SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL[q_idx].0 as u32;
            flat.extend_from_slice(&pack_core_quad_params(
                &q.to_v4_kernel_params_for_block(block_idx),
            ));
        }
        Ok(flat)
    }

    /// Clamps learned `ro_pole_48k`, `CoreQuad` poles/drives/`svf_k_nl`, `Op-N[Pre]`, `Op-T` delays &
    /// `scat_psi`, `Kautz` poles, `w_ro`/`fir`, and `tier_gain_trim` cleanly onto `dnam-vm` invariants.
    pub fn clamp_all_stability_bounds(&mut self, crossbar_coupled: bool) {
        clamp_model_stability_bounds(self, crossbar_coupled);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::stability::MAX_RO_POLE_48K_F64;
    use crate::gpu::abi::quad_param_slots;
    use dnam_kernels::{MAX_SVF_K_NL, MIN_TPT_G, TANK_MAX_G_LIN_F32};

    #[test]
    fn test_sona_model_zoh_rescaling_packing_and_clamps() {
        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].svf_g = [0.25; 4];
        model.core_quads[0].lru_alpha_rel = [0.995; 4];
        model.core_quads[0].svf_k_nl = [0.85, 0.25, 0.0, 2.5];
        model.core_quads[1].svf_k_nl = [3.5; 4];
        let zoh_1x = model.core_quads[0].rescaled_1x_zoh(2.0);
        assert!(zoh_1x.lru_alpha_rel[0] < model.core_quads[0].lru_alpha_rel[0]);
        assert!(zoh_1x.svf_g[0] > 0.0 && zoh_1x.svf_g[0].is_finite());

        let rho_2x = model.core_quads[0].rescaled_for_rho(0.5);
        assert!(rho_2x.lru_alpha_rel[0] > model.core_quads[0].lru_alpha_rel[0]);
        assert!(rho_2x.svf_g[0] < model.core_quads[0].svf_g[0]);

        let packed = model.pack_core_quads_for_runner(4).unwrap();
        assert_eq!(packed.len(), 4 * QUAD_PARAMS_VEC4_STRIDE);
        assert_eq!(packed[quad_param_slots::SVF_K_NL], [0.85, 0.25, 0.0, 2.5]);
        assert_eq!(
            packed[QUAD_PARAMS_VEC4_STRIDE + quad_param_slots::SVF_K_NL],
            [0.0; 4]
        );

        model.ro_pole_48k = 1.0;
        model.transport_quads[0].delay_samples[0] = 1.0;
        model.transport_quads[0].g_loop = 0.9999;
        model.transport_quads[0].scat_psi = 10.0;
        model.core_quads[0].svf_g[0] = f64::NAN;
        model.core_quads[0].svf_g[1] = 1.0e-9;
        model.core_quads[0].svf_k_nl[0] = 99.0;
        model.tank_dry_mix = f64::NAN;
        model.clamp_all_stability_bounds(true);
        assert_eq!(model.ro_pole_48k, MAX_RO_POLE_48K_F64);
        assert_eq!(model.transport_quads[0].delay_samples[0], 5.0);
        assert_eq!(
            model.transport_quads[0].g_loop,
            f64::from(TANK_MAX_G_LIN_F32)
        );
        assert_eq!(model.transport_quads[0].scat_psi, core::f64::consts::PI);
        assert_eq!(model.core_quads[0].svf_g[0], 0.125);
        assert_eq!(model.core_quads[0].svf_g[1], f64::from(MIN_TPT_G));
        assert_eq!(model.core_quads[0].svf_k_nl[0], f64::from(MAX_SVF_K_NL));
        assert_eq!(model.core_quads[1].svf_k_nl, [0.0; 4]);
        assert_eq!(model.tank_dry_mix, 1.0);
    }
}
