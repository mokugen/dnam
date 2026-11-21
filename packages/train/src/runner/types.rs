//! Shared parameter-packing helpers and kernel I/O bundle types consumed identically by both
//! [`crate::cpu::CpuRunner`] and `WgpuRunner` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, KernelVec4, SonaParaLruDuhemCoeffs4, SonaToneWeights4,
    TptSvfCoeffs4,
};

use crate::error::TrainError;
use crate::gpu::abi::{
    AdjointScanUniforms, GaborBezoutUniforms, InnerOptUniforms, OzakiGramUniforms,
    PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms, QUAD_PARAMS_VEC4_STRIDE, quad_param_slots,
};

/// Packs a [`CoreQuadParamsStereo4`] into the canonical `24 × [f32; 4]` GPU/CPU uniform-stride block
/// (including `SVF_G = two_svf_g * 0.5`).
#[must_use]
pub fn pack_core_quad_params(
    params: &CoreQuadParamsStereo4<GpuVec4>,
) -> [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE] {
    let g = params.two_svf_g.mul(GpuVec4::splat(0.5));
    let mut out = [[0.0_f32; 4]; QUAD_PARAMS_VEC4_STRIDE];
    out[quad_param_slots::SVF_A1] = params.svf.a1.to_array();
    out[quad_param_slots::SVF_A2] = params.svf.a2.to_array();
    out[quad_param_slots::SVF_A3] = params.svf.a3.to_array();
    out[quad_param_slots::SVF_K] = params.svf.k.to_array();
    out[quad_param_slots::SVF_TWO_K] = params.svf.two_k.to_array();
    out[quad_param_slots::TONE_LP] = params.tone.lp.to_array();
    out[quad_param_slots::TONE_BP] = params.tone.bp.to_array();
    out[quad_param_slots::TONE_HP] = params.tone.hp.to_array();
    out[quad_param_slots::TONE_AP] = params.tone.ap.to_array();
    out[quad_param_slots::TONE_DIR] = params.tone.dir.to_array();
    out[quad_param_slots::LRU_ALPHA_REL] = params.lru.alpha_rel.to_array();
    out[quad_param_slots::LRU_ALPHA_ATK] = params.lru.alpha_atk.to_array();
    out[quad_param_slots::LRU_KAPPA] = params.lru.kappa.to_array();
    out[quad_param_slots::LRU_GAMMA_DUHEM] = params.lru.gamma_duhem.to_array();
    out[quad_param_slots::LRU_SIGNED_MIX] = params.lru.signed_mix.to_array();
    out[quad_param_slots::LRU_STEREO_LINK] = params.lru.stereo_link.to_array();
    out[quad_param_slots::TWO_SVF_G] = params.two_svf_g.to_array();
    out[quad_param_slots::RECT_SIGN] = params.rect_sign.to_array();
    out[quad_param_slots::SAG_DEPTH] = params.sag_depth.to_array();
    out[quad_param_slots::ADAA_DRIVE] = params.adaa_drive.to_array();
    out[quad_param_slots::LRU_BIAS_DRIVE] = params.lru_bias_drive.to_array();
    out[quad_param_slots::WAVEFOLDER_MIX] = params.wavefolder_mix.to_array();
    out[quad_param_slots::GLU_MIX] = params.glu_mix.to_array();
    out[quad_param_slots::SVF_G] = g.to_array();
    out
}

/// Unpacks one `24 × [f32; 4]` quad parameter slice back into [`CoreQuadParamsStereo4<GpuVec4>`].
pub fn unpack_core_quad_params(
    packed: &[[f32; 4]],
) -> Result<CoreQuadParamsStereo4<GpuVec4>, TrainError> {
    if packed.len() < QUAD_PARAMS_VEC4_STRIDE {
        return Err(TrainError::InvalidDimensions {
            context: "unpack_core_quad_params",
            expected: QUAD_PARAMS_VEC4_STRIDE,
            actual: packed.len(),
        });
    }
    let svf = TptSvfCoeffs4 {
        a1: GpuVec4::from_array(packed[quad_param_slots::SVF_A1]),
        a2: GpuVec4::from_array(packed[quad_param_slots::SVF_A2]),
        a3: GpuVec4::from_array(packed[quad_param_slots::SVF_A3]),
        k: GpuVec4::from_array(packed[quad_param_slots::SVF_K]),
        two_k: GpuVec4::from_array(packed[quad_param_slots::SVF_TWO_K]),
    };
    let tone = SonaToneWeights4 {
        lp: GpuVec4::from_array(packed[quad_param_slots::TONE_LP]),
        bp: GpuVec4::from_array(packed[quad_param_slots::TONE_BP]),
        hp: GpuVec4::from_array(packed[quad_param_slots::TONE_HP]),
        ap: GpuVec4::from_array(packed[quad_param_slots::TONE_AP]),
        dir: GpuVec4::from_array(packed[quad_param_slots::TONE_DIR]),
    };
    let lru = SonaParaLruDuhemCoeffs4::from_clamped(
        GpuVec4::from_array(packed[quad_param_slots::LRU_ALPHA_REL]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_ALPHA_ATK]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_KAPPA]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_GAMMA_DUHEM]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_SIGNED_MIX]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_STEREO_LINK]),
    );
    Ok(CoreQuadParamsStereo4 {
        svf,
        tone,
        lru,
        two_svf_g: GpuVec4::from_array(packed[quad_param_slots::TWO_SVF_G]),
        rect_sign: GpuVec4::from_array(packed[quad_param_slots::RECT_SIGN]),
        sag_depth: GpuVec4::from_array(packed[quad_param_slots::SAG_DEPTH]),
        adaa_drive: GpuVec4::from_array(packed[quad_param_slots::ADAA_DRIVE]),
        lru_bias_drive: GpuVec4::from_array(packed[quad_param_slots::LRU_BIAS_DRIVE]),
        wavefolder_mix: GpuVec4::from_array(packed[quad_param_slots::WAVEFOLDER_MIX]),
        glu_mix: GpuVec4::from_array(packed[quad_param_slots::GLU_MIX]),
    })
}

/// Shared input bundle for `prescreen_population`.
#[derive(Clone, Debug)]
pub struct PrescreenInput<'a> {
    /// Dispatch dimensions and scaling floor.
    pub uniforms: PrescreenUniforms,
    /// Stereo dry input stems (`num_stems * num_frames`).
    pub input_audio: &'a [[f32; 2]],
    /// Stereo reference target stems (`num_stems * num_frames`).
    pub target_audio: &'a [[f32; 2]],
    /// Flattened seed parameter table (`num_seeds * num_quads * 16` or per-seed `16`).
    pub seed_params: &'a [f32],
}

impl PrescreenInput<'_> {
    pub(crate) fn validate(&self) -> Result<(usize, usize, usize, usize), TrainError> {
        let num_seeds = self.uniforms.num_seeds as usize;
        let num_frames = self.uniforms.num_frames as usize;
        let num_quads = self.uniforms.num_quads as usize;
        let num_stems = self.uniforms.num_stems as usize;
        if num_seeds == 0 || num_frames == 0 || num_quads == 0 || num_stems == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "prescreen_population",
                expected: 1,
                actual: 0,
            });
        }
        let expected_audio = num_stems * num_frames;
        if self.input_audio.len() < expected_audio {
            return Err(TrainError::InvalidDimensions {
                context: "prescreen_population(input_audio)",
                expected: expected_audio,
                actual: self.input_audio.len(),
            });
        }
        if self.target_audio.len() < expected_audio {
            return Err(TrainError::InvalidDimensions {
                context: "prescreen_population(target_audio)",
                expected: expected_audio,
                actual: self.target_audio.len(),
            });
        }
        let min_seeds = num_seeds * PRESCREEN_SEED_PARAM_STRIDE;
        if self.seed_params.len() < min_seeds {
            return Err(TrainError::InvalidDimensions {
                context: "prescreen_population(seed_params)",
                expected: min_seeds,
                actual: self.seed_params.len(),
            });
        }
        Ok((num_seeds, num_frames, num_quads, num_stems))
    }
}

/// Shared input bundle for `fused_core_scan_256`.
#[derive(Clone, Debug)]
pub struct FusedCoreScanInput<'a> {
    /// Scan uniform block (`num_frames, num_tiles, num_quads, num_stems, tile_start, ...`).
    pub uniforms: crate::gpu::abi::FusedScanUniforms,
    /// Interleaved `(L, R)` `Vec4` input frames (`total_q * num_frames * 2`).
    pub core_inputs: &'a [[f32; 4]],
    /// Packed `24 × Vec4` parameters per `(stem, quad)` (`total_q * 24`).
    pub quad_params: &'a [[f32; 4]],
    /// Optional pre-seeded checkpoint table (`total_q * num_tiles * 8`) for `tile_start > 0` resume.
    pub initial_checkpoints: Option<&'a [[f32; 4]]>,
}

/// Output buffers produced by `fused_core_scan_256`.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedCoreScanOutput {
    /// Per-tile boundary checkpoints (`total_q * num_tiles * 8`).
    pub tile_checkpoints: Vec<[f32; 4]>,
    /// Per-frame extracted features `(v_eff_L, v_eff_R, adaa_odd_L, adaa_odd_R, adaa_even_L, adaa_even_R)` (`total_q * num_frames * 6`).
    pub feature_outputs: Vec<[f32; 4]>,
}

/// Shared input bundle for `adjoint_scan_256`.
#[derive(Clone, Debug)]
pub struct AdjointScanInput<'a> {
    /// Adjoint uniform block (`num_frames, num_tiles, num_quads, num_stems, ...`).
    pub uniforms: AdjointScanUniforms,
    /// Forward `core_inputs` (`total_q * num_frames * 2`).
    pub core_inputs: &'a [[f32; 4]],
    /// Packed `24 × Vec4` parameters per `(stem, quad)` (`total_q * 24`).
    pub quad_params: &'a [[f32; 4]],
    /// Forward `tile_checkpoints` (`total_q * num_tiles * 8`).
    pub tile_checkpoints: &'a [[f32; 4]],
    /// Upstream loss gradient w.r.t. `feature_outputs` (`total_q * num_frames * 6`).
    pub d_loss_d_features: &'a [[f32; 4]],
    /// Forward `feature_outputs` (`total_q * num_frames * 6`) for cross-lane GLU pullback.
    pub feature_outputs: &'a [[f32; 4]],
}

/// Output buffers produced by `adjoint_scan_256`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjointScanOutput {
    /// Input gradients (`total_q * num_frames * 2`).
    pub d_loss_d_inputs: Vec<[f32; 4]>,
    /// Per-tile parameter gradients (`total_q * num_tiles * 24`).
    pub param_grads: Vec<[f32; 4]>,
}

/// Shared input bundle for `gabor_bezout_grad`.
#[derive(Clone, Debug)]
pub struct GaborBezoutInput<'a> {
    /// Mode A (`fft_size == 0`) or Mode B (`fft_size > 0`) uniform block.
    pub uniforms: GaborBezoutUniforms,
    /// Predicted stereo stems (`num_stems * num_frames`).
    pub pred_audio: &'a [[f32; 2]],
    /// Target stereo stems (`num_stems * num_frames`).
    pub target_audio: &'a [[f32; 2]],
    /// Mode A filter taps `[lf, mf, hf, _]` (`>= hop_size`) or Mode B window/band weights `[win, lf, mf, hf]` (`>= fft_size`).
    pub bezout_taps: &'a [[f32; 4]],
}

/// Output buffers produced by `gabor_bezout_grad`.
#[derive(Clone, Debug, PartialEq)]
pub struct GaborBezoutOutput {
    /// Time-domain stereo gradient w.r.t. `pred_audio` (`num_stems * num_frames`).
    pub d_loss_d_pred: Vec<[f32; 2]>,
    /// Per-(stem, window) `[loss_lf, loss_mf, loss_hf, total_loss]`.
    pub band_losses: Vec<[f32; 4]>,
}

/// Shared input bundle for `ozaki_gram_rhs_i32`.
#[derive(Clone, Debug)]
pub struct OzakiGramInput<'a> {
    /// Matrix dimensions and chunking metadata.
    pub uniforms: OzakiGramUniforms,
    /// Column-major feature matrix `features[col * num_frames + t]` (`feature_dim * num_frames`).
    pub features: &'a [f32],
    /// Column-major target matrix `targets[ch * num_frames + t]` (`target_channels * num_frames`; pass `&[]` if `target_channels == 0`).
    pub targets: &'a [f32],
    /// Per-column scale pairs `[max_abs_or_scale, inv_scale]` (`feature_dim + target_channels`).
    pub col_scales: &'a [[f32; 2]],
    /// Optional pre-filled `ozaki_buckets` buffer (used to verify sentinel preservation when `target_channels == 0`).
    pub seed_buckets: Option<&'a [i32]>,
}

/// Output `i32` Ozaki accumulators produced by `ozaki_gram_rhs_i32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OzakiGramOutput {
    /// Flat `i32` accumulators `[pair * num_chunks * 6 + ch * 6 + word]`.
    pub ozaki_buckets: Vec<i32>,
    /// Number of chunk tiles along `num_frames`.
    pub num_chunks: usize,
    /// Effective dispatched pair count.
    pub num_pairs: usize,
}

/// Shared input bundle for `inner_opt_step`.
#[derive(Clone, Debug)]
pub struct InnerOptInput<'a> {
    /// Step index, L-BFGS history window `history_m <= 8`, line search iter, and damping parameters.
    pub uniforms: InnerOptUniforms,
    /// Current parameter vector (`num_params`).
    pub theta: &'a [f32],
    /// Current objective gradient (`num_params`).
    pub grad: &'a [f32],
    /// Inclusive parameter box bounds `[lo, hi]` (`num_params`).
    pub param_bounds: &'a [[f32; 2]],
    /// L-BFGS circular `(s, y)` history plus anchor row `(history_m + 1) * num_params`.
    pub lbfgs_s_y: &'a [[f32; 4]],
    /// Search direction vector (`num_params`).
    pub search_dir: &'a [f32],
}

/// Updated optimization state emitted by `inner_opt_step`.
#[derive(Clone, Debug, PartialEq)]
pub struct InnerOptOutput {
    /// Updated clamped parameter vector (`num_params`).
    pub theta: Vec<f32>,
    /// Updated L-BFGS history buffer `(history_m + 1) * num_params`.
    pub lbfgs_s_y: Vec<[f32; 4]>,
    /// Updated or preserved search direction (`num_params`).
    pub search_dir: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        SonaParaLruDuhemCoeffs4, SonaToneWeights4, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
    };

    use super::*;

    #[test]
    fn test_pack_and_unpack_core_quad_params_roundtrip() {
        let g = GpuVec4::splat(prewarp_tan_pi_f_over_fs_f32(0.08));
        let k = GpuVec4::splat(0.75);
        let orig = CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::splat(0.4),
                bp: GpuVec4::splat(0.2),
                hp: GpuVec4::splat(-0.1),
                ap: GpuVec4::splat(0.05),
                dir: GpuVec4::splat(0.3),
            },
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.85),
                GpuVec4::splat(0.35),
                GpuVec4::splat(0.25),
                GpuVec4::splat(0.15),
                GpuVec4::splat(0.45),
                GpuVec4::splat(0.20),
            ),
            two_svf_g: g.add(g),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: GpuVec4::splat(0.25),
            adaa_drive: GpuVec4::splat(1.5),
            lru_bias_drive: GpuVec4::splat(0.1),
            wavefolder_mix: GpuVec4::splat(0.4),
            glu_mix: GpuVec4::splat(0.5),
        };
        let packed = pack_core_quad_params(&orig);
        assert_eq!(packed[quad_param_slots::SVF_G], g.to_array());
        let unpacked = unpack_core_quad_params(&packed).unwrap();
        assert_eq!(unpacked.svf.a1.to_array(), orig.svf.a1.to_array());
        assert_eq!(unpacked.glu_mix.to_array(), orig.glu_mix.to_array());
        assert!(matches!(
            unpack_core_quad_params(&packed[..4]),
            Err(TrainError::InvalidDimensions { .. })
        ));

        let valid_prescreen = PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 2,
                num_frames: 4,
                num_quads: 1,
                num_stems: 1,
                sample_rate_hz: 48_000.0,
                eps_floor: 1.0e-5,
                _pad0: 0,
                _pad1: 0,
            },
            input_audio: &[[0.1, 0.2]; 4],
            target_audio: &[[0.0, 0.0]; 4],
            seed_params: &[0.0; 32],
        };
        assert_eq!(valid_prescreen.validate().unwrap(), (2, 4, 1, 1));
        let bad_zero = PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 0,
                ..valid_prescreen.uniforms
            },
            ..valid_prescreen
        };
        assert!(matches!(
            bad_zero.validate(),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let bad_in = PrescreenInput {
            input_audio: &[],
            ..valid_prescreen
        };
        assert!(matches!(
            bad_in.validate(),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let bad_tgt = PrescreenInput {
            target_audio: &[],
            ..valid_prescreen
        };
        assert!(matches!(
            bad_tgt.validate(),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let bad_seed = PrescreenInput {
            seed_params: &[],
            ..valid_prescreen
        };
        assert!(matches!(
            bad_seed.validate(),
            Err(TrainError::InvalidDimensions { .. })
        ));
    }
}
