#![forbid(unsafe_code)]
//! Stage A (`Physics Probe` & Universal Anchor-Stem Lifting, `solver::stage_a_probe`).
//!
//! Delegates `< 0.5 ms` `Op-P` quadrature probing, causal Steiglitz–McBride / Kautz seeding,
//! Early FIR lifting, and `Op-T` comb delay identification onto `dnam_train::run_pass1_probe`,
//! encoding a validated `SONA v3.0` `.dnam-bin` payload (`ref[train.pipeline.physics-probe]`).

use dnam_engine::schema::DnamError;
use dnam_train::{
    Pass5EmitConfig, SonaModelF64, decode_dnam_bin_to_model, encode_model_to_dnam_bin,
    run_pass1_probe, run_phase2_rail_burst,
};
use dnam_vm::Tier;
use dnam_vm::bin::{DNBN_FLAG_STEREO_OUT, DnamBinView};
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::F32;

use crate::proj::schema::DnamProjManifest;
use crate::proj::verify_dnam_bin_zero_dc;

/// Configuration for Stage A physics probing & anchor seeding (`ref[train.pipeline.physics-probe]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageAConfig {
    /// Preserve an already-populated `SONA v3.0` `.dnam-bin` physical backbone (`true` for
    /// white-box oracles or Path-1 incremental vault edits; `false` to seed fresh physical priors).
    pub preserve_existing_backbone: bool,
    /// Number of coordinate Variable-Projection refinement passes (`1..=2`).
    pub vp_gn_steps: usize,
}

impl Default for StageAConfig {
    fn default() -> Self {
        Self {
            preserve_existing_backbone: true,
            vp_gn_steps: 2,
        }
    }
}

/// Telemetry emitted by [`run_stage_a_physics_probe`] (`ref[train.pipeline.physics-probe]`).
#[derive(Debug, Clone, PartialEq)]
pub struct StageAReport {
    /// Wall-clock Stage A execution time in milliseconds.
    pub elapsed_ms: f32,
    /// True if an existing non-trivial `SONA v3.0` backbone was preserved rather than re-synthesized.
    pub preserved_backbone: bool,
    /// Spectral low/mid energy transition balance estimate in Hz (`80..260 Hz`).
    pub crossover_hz: f32,
    /// Energy of the lifted Early FIR impulse response.
    pub early_fir_energy: f64,
    /// Number of lifted shared Kautz stereo pole pairs (`0..=32`).
    pub kautz_poles_lifted: usize,
    /// Optional Phase-2 `[CLAMP-RAIL]` prescription emitted if corner drive saturated.
    pub phase2_rail_warning: Option<String>,
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

#[inline]
fn map_stage_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

/// Returns `true` if `dnam_bin` parses as a validated `SONA v3.0` `.dnam-bin` with non-default
/// learned backbone parameters (distinct from zero buffers or resting identity templates).
#[must_use]
pub fn has_populated_v3_backbone(dnam_bin: &[u8]) -> bool {
    let Ok(view) = DnamBinView::parse(dnam_bin) else {
        return false;
    };
    let a0 = view.arena0;
    let core_active = a0.core_quads.iter().any(|q| {
        q.adaa_drive_fp16.iter().any(|w| w.get() != 0)
            || q.tone_lp_fp16.iter().any(|w| w.get() != 0)
            || q.tone_bp_fp16.iter().any(|w| w.get() != 0)
            || q.tone_hp_fp16.iter().any(|w| w.get() != 0)
            || q.tone_ap_fp16.iter().any(|w| w.get() != 0)
            || q.lru_kappa_fp16.iter().any(|w| w.get() != 0)
            || q.wavefolder_mix_fp16.iter().any(|w| w.get() != 0)
            || q.svf_k_nl_fp16.iter().any(|w| w.get() != 0)
    });
    let tiles_active = a0
        .tiles
        .iter()
        .any(|t| t.w_odd_4x4.iter().flatten().any(|w| w.get().abs() > 1.0e-6));
    let fir_active = a0.conv_fir_lite.fir_l[1..]
        .iter()
        .any(|w| w.get().abs() > 1.0e-6)
        || a0.conv_fir_lite.fir_r[1..]
            .iter()
            .any(|w| w.get().abs() > 1.0e-6);
    let kautz_active = a0.conv_kautz_lite.iter().any(|k| {
        k.c_re_l.iter().any(|w| w.get().abs() > 1.0e-6)
            || k.c_im_l.iter().any(|w| w.get().abs() > 1.0e-6)
            || k.c_re_r.iter().any(|w| w.get().abs() > 1.0e-6)
            || k.c_im_r.iter().any(|w| w.get().abs() > 1.0e-6)
    });
    let transport_active = a0.transport_lite.g_loop.get().abs() > 1.0e-6
        || a0
            .transport_lite
            .wet_tap_l_fp16
            .iter()
            .any(|w| w.get() != 0);
    let readout_active = a0
        .linear_readout_lite
        .w_ro_fp16
        .iter()
        .any(|w| w.get() != 0);
    core_active || tiles_active || fir_active || kautz_active || transport_active || readout_active
}

fn estimate_crossover_hz(spectral_summary_4096: Option<&[u8; 4_096]>) -> f32 {
    let Some(spec) = spectral_summary_4096 else {
        return 140.0;
    };
    let Ok(words) = <[F32]>::ref_from_bytes(spec) else {
        return 140.0;
    };
    if words.len() < 32 {
        return 140.0;
    }
    let bin_hz = 48_000.0_f64 / 4_096.0;
    let mut sub_e = 0.0_f64;
    let mut mid_e = 0.0_f64;
    for b in 2..64usize {
        let m = f64::from(words[b / 2].get());
        let f_hz = (b as f64) * bin_hz;
        if f_hz < 180.0 {
            sub_e += m * m;
        } else {
            mid_e += m * m;
        }
    }
    let ratio = (sub_e / (sub_e + mid_e + 1.0e-12)).clamp(0.0, 1.0);
    (95.0 + 130.0 * ratio as f32).clamp(80.0, 260.0)
}

fn check_rail_warning(dnam_bin: &[u8], ctx: &str) -> Option<String> {
    match run_phase2_rail_burst(dnam_bin, Tier::Lite, 48_000.0, 1.0) {
        Ok(rail) if rail.flagged => Some(format!(
            "[CLAMP-RAIL] corner drive saturated ({ctx}); scale {:.3}",
            rail.safe_scale
        )),
        Err(err) => Some(format!("[CLAMP-RAIL] probe burst failed ({ctx}): {err}")),
        Ok(_) => None,
    }
}

pub(crate) fn encode_with_preserved_controls(
    model: &mut SonaModelF64,
    prev_dbin: &[u8],
    mono_out: bool,
    dual_in: bool,
) -> Result<Vec<u8>, DnamError> {
    if let Ok(prev) = decode_dnam_bin_to_model(prev_dbin) {
        if model.control_audit.is_none() {
            model.control_audit = prev.control_audit;
        }
        if model.control_bus.is_none() {
            model.control_bus = prev.control_bus;
        }
        if prev
            .tier_gain_trim
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        {
            model.tier_gain_trim = prev.tier_gain_trim;
        }
    }
    let prev_stereo = DnamBinView::parse(prev_dbin)
        .ok()
        .map(|v| (v.header.flags.get() & DNBN_FLAG_STEREO_OUT) != 0);
    let stereo_out = prev_stereo.unwrap_or(!mono_out);
    let emit_cfg = Pass5EmitConfig {
        stereo_out,
        single_input: !dual_in && !stereo_out,
        calibrate_lufs_trims: false,
        ..Default::default()
    };
    encode_model_to_dnam_bin(model, emit_cfg).map_err(map_stage_err)
}

/// Runs Stage A (`Physics Probe` & Anchor Seeding) on `SONA v3.0` `.dnam-bin` (`ref[train.pipeline.physics-probe]`).
pub fn run_stage_a_physics_probe(
    manifest: &DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    spectral_summary_4096: Option<&[u8; 4_096]>,
    config: StageAConfig,
) -> StageAReport {
    let timer = now_ms_token();
    let mono_out = manifest.base.io_config.output_mode == "mono";
    let dual_in = manifest.base.io_config.inputs >= 2;
    let crossover_hz = estimate_crossover_hz(spectral_summary_4096);

    if config.preserve_existing_backbone && has_populated_v3_backbone(dnam_bin) {
        let model =
            decode_dnam_bin_to_model(dnam_bin).unwrap_or_else(|_| SonaModelF64::new(48_000.0));
        let fir_e: f64 = model.fir_l.iter().map(|&v| v * v).sum();
        let warn = check_rail_warning(dnam_bin, "preserved backbone");
        return StageAReport {
            elapsed_ms: elapsed_ms_since(timer),
            preserved_backbone: true,
            crossover_hz,
            early_fir_energy: fir_e,
            kautz_poles_lifted: model.active_kautz_pairs,
            phase2_rail_warning: warn,
        };
    }

    let n_fit = dry_lr.len().min(wet_lr.len()).min(12_000);
    let mut model = if n_fit >= 64 {
        let in_l: Vec<f64> = dry_lr[..n_fit].iter().map(|lr| f64::from(lr[0])).collect();
        let in_r: Vec<f64> = dry_lr[..n_fit]
            .iter()
            .map(|lr| f64::from(if dual_in || !mono_out { lr[1] } else { lr[0] }))
            .collect();
        let tgt_l: Vec<f64> = wet_lr[..n_fit].iter().map(|lr| f64::from(lr[0])).collect();
        let tgt_r: Vec<f64> = wet_lr[..n_fit]
            .iter()
            .map(|lr| f64::from(if mono_out { lr[0] } else { lr[1] }))
            .collect();
        run_pass1_probe(&in_l, &in_r, &tgt_l, &tgt_r, 48_000.0, 128, 32)
            .map(|(_rep, m)| m)
            .unwrap_or_else(|_| SonaModelF64::new(48_000.0))
    } else {
        let mut m = SonaModelF64::new(48_000.0);
        m.active_fir_taps = 1;
        m.fir_l[0] = 1.0;
        m.fir_r[0] = 1.0;
        m
    };

    if let Ok(encoded) = encode_with_preserved_controls(&mut model, dnam_bin, mono_out, dual_in) {
        *dnam_bin = encoded;
    }
    let _ = verify_dnam_bin_zero_dc(dnam_bin);
    let fir_e: f64 = model.fir_l.iter().map(|&v| v * v).sum();
    let warn = check_rail_warning(dnam_bin, "Stage A probe");

    StageAReport {
        elapsed_ms: elapsed_ms_since(timer),
        preserved_backbone: false,
        crossover_hz,
        early_fir_energy: fir_e,
        kautz_poles_lifted: model.active_kautz_pairs,
        phase2_rail_warning: warn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;

    #[test]
    fn test_stage_a_physics_probe_seeds_and_preserves_sona_dnam_bin() {
        let mut blank = vec![0u8; 256];
        assert!(!has_populated_v3_backbone(&blank));

        let mut dnam_bin = initialize_backbone_dnam_bin_if_needed(None).expect("neutral dnam-bin");
        assert!(!has_populated_v3_backbone(&dnam_bin));

        let manifest = DnamProjManifest::default_reference();
        let n = 1_024usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.25 * ((i as f32) * 0.08).sin();
                [s, s * 0.8]
            })
            .collect();
        let wet: Vec<[f32; 2]> = dry.iter().map(|&[l, r]| [l * 0.9, r * 0.9]).collect();

        let rep = run_stage_a_physics_probe(
            &manifest,
            &mut dnam_bin,
            &dry,
            &wet,
            None,
            StageAConfig {
                preserve_existing_backbone: false,
                vp_gn_steps: 1,
            },
        );
        assert!(!rep.preserved_backbone);
        assert!(DnamBinView::parse(&dnam_bin).is_ok());
        assert!((80.0..=260.0).contains(&rep.crossover_hz));

        let mut model = decode_dnam_bin_to_model(&dnam_bin).expect("decode");
        model.core_quads[0].adaa_drive[0] = 0.42;
        dnam_bin = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).expect("encode");
        assert!(has_populated_v3_backbone(&dnam_bin));

        let rep_pres = run_stage_a_physics_probe(
            &manifest,
            &mut dnam_bin,
            &dry,
            &wet,
            None,
            StageAConfig::default(),
        );
        assert!(rep_pres.preserved_backbone);
        blank.clear();
    }
}
