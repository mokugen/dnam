#![forbid(unsafe_code)]
//! Stage A (`Physics Probe` & Universal Anchor-Stem Lifting, `solver::stage_a_probe`).
//!
//! Delegates `< 0.5 ms` `Op-P` quadrature probing, causal Steiglitz–McBride / Kautz seeding,
//! Early FIR lifting, and `Op-T` comb delay identification onto `dnam_train::run_pass1_probe`,
//! encoding a validated `SONA v3.0` `.dnam-bin` payload (`ref[train.pipeline.physics-probe]`).

use dnam_engine::schema::DnamError;
use dnam_train::{
    Pass1PhysicsHint, Pass1ProbeConfig, Pass1ProbeResult, Pass5EmitConfig, SonaModelF64,
    decode_dnam_bin_to_model, encode_model_to_dnam_bin, run_pass1_probe_with_hint,
    run_phase2_rail_burst,
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

/// Builds a 100% data-driven [`Pass1PhysicsHint`] strictly from numeric scalar descriptors on
/// `manifest.stems_registry` (`extracted_physics` and `user_metadata["phys_*"]`) without any
/// device, brand, or control-name heuristics (`REQ-1.6`, `REQ-1.9`).
#[must_use]
pub fn pass1_hint_from_manifest(manifest: &DnamProjManifest) -> Pass1PhysicsHint {
    let mut hint = Pass1PhysicsHint {
        has_stereo_width: Some(manifest.base.io_config.output_mode != "mono"),
        ..Pass1PhysicsHint::default()
    };
    let stem_opt = manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .or_else(|| manifest.stems_registry.iter().find(|s| s.enabled));
    let Some(stem) = stem_opt else {
        return hint;
    };
    if let Some(ref ep) = stem.extracted_physics {
        if let Some(&d) = ep.delays_ms.iter().find(|&&v| v.is_finite() && v > 0.0) {
            hint.delay_ms = Some(f64::from(d));
            if let Some(&g) = ep
                .decay_gains_lin
                .iter()
                .find(|&&v| v.is_finite() && v > 1.0e-4 && v < 0.9999)
            {
                let rt60 = -3.0 * (f64::from(d) * 1.0e-3) / f64::from(g).log10();
                if rt60.is_finite() && rt60 > 0.0 {
                    hint.rt60_s = Some(rt60);
                }
            }
        }
        if let Some(&r) = ep.lfo_freqs_hz.iter().find(|&&v| v.is_finite() && v > 0.0) {
            hint.modulation_rate_hz = Some(f64::from(r));
        }
    }
    let read_f64 = |key: &str| stem.user_metadata.get(key).and_then(|v| v.as_f64());
    if let Some(v) = read_f64("phys_rt60_s") {
        hint.rt60_s = Some(v);
    } else if let Some(v_ms) = read_f64("phys_rt60_ms").filter(|v| v.is_finite() && *v > 0.0) {
        hint.rt60_s = Some(v_ms * 1.0e-3);
    }
    if let Some(v) = read_f64("phys_delay_ms") {
        hint.delay_ms = Some(v);
    } else if let Some(v) = read_f64("phys_short_delay_ms").filter(|v| v.is_finite() && *v > 0.0) {
        hint.delay_ms = Some(v);
    }
    let odd_harm = read_f64("phys_odd_harm").filter(|v| v.is_finite());
    let even_harm = read_f64("phys_even_harm").filter(|v| v.is_finite());
    if let Some(v) = read_f64("phys_thd_ratio") {
        hint.thd_ratio = Some(v);
    } else if odd_harm.is_some() || even_harm.is_some() {
        let thd = odd_harm.unwrap_or(0.0).max(0.0) + even_harm.unwrap_or(0.0).max(0.0);
        hint.thd_ratio = Some(thd);
    }
    if let Some(v) = read_f64("phys_even_odd_asymmetry") {
        hint.even_odd_asymmetry = Some(v);
    } else if odd_harm.is_some() || even_harm.is_some() {
        let odd = odd_harm.unwrap_or(0.0).max(0.0);
        let even = even_harm.unwrap_or(0.0).max(0.0);
        let denom = odd.abs() + even.abs();
        hint.even_odd_asymmetry = Some(if denom > 1.0e-9 {
            ((even - odd) / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        });
    }
    if let Some(v) = read_f64("phys_modulation_rate_hz") {
        hint.modulation_rate_hz = Some(v);
    } else if let Some(v) = read_f64("phys_lfo_hz").filter(|v| v.is_finite() && *v > 0.0) {
        hint.modulation_rate_hz = Some(v);
    }
    if let Some(v) = read_f64("phys_noise_floor_db") {
        hint.noise_floor_db = Some(v);
    }
    if let Some(v) = stem
        .user_metadata
        .get("phys_has_stereo_width")
        .and_then(|v| v.as_bool())
    {
        hint.has_stereo_width = Some(v);
    } else if let Some(stereo_diff) = read_f64("phys_stereo_diff").filter(|v| v.is_finite()) {
        hint.has_stereo_width = Some(stereo_diff > 0.05);
    }
    hint
}

/// Runs Stage A (`Physics Probe` & Anchor Seeding) on `SONA v3.0` `.dnam-bin`, returning both
/// [`StageAReport`] and the live [`Pass1ProbeResult`] for direct Stage B reuse
/// (`ref[train.pipeline.physics-probe]`).
pub fn run_stage_a_physics_probe_with_result(
    manifest: &DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    spectral_summary_4096: Option<&[u8; 4_096]>,
    config: StageAConfig,
) -> (StageAReport, Option<Pass1ProbeResult>) {
    let timer = now_ms_token();
    let mono_out = manifest.base.io_config.output_mode == "mono";
    let dual_in = manifest.base.io_config.inputs >= 2;
    let crossover_hz = estimate_crossover_hz(spectral_summary_4096);

    if config.preserve_existing_backbone && has_populated_v3_backbone(dnam_bin) {
        let model =
            decode_dnam_bin_to_model(dnam_bin).unwrap_or_else(|_| SonaModelF64::new(48_000.0));
        let fir_e: f64 = model.fir_l.iter().map(|&v| v * v).sum();
        let warn = check_rail_warning(dnam_bin, "preserved backbone");
        return (
            StageAReport {
                elapsed_ms: elapsed_ms_since(timer),
                preserved_backbone: true,
                crossover_hz,
                early_fir_energy: fir_e,
                kautz_poles_lifted: model.active_kautz_pairs,
                phase2_rail_warning: warn,
            },
            None,
        );
    }

    let hint = pass1_hint_from_manifest(manifest);
    let n_fit = dry_lr.len().min(wet_lr.len());
    let mut pass1_res = if n_fit >= 64 {
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
        run_pass1_probe_with_hint(
            &in_l,
            &in_r,
            &tgt_l,
            &tgt_r,
            Pass1ProbeConfig {
                sample_rate_hz: 48_000.0,
                max_fir_taps: 128,
                max_kautz_pairs: 32,
                hint: Some(&hint),
            },
        )
        .ok()
    } else {
        None
    };

    let mut fallback_model = if let Some(ref mut res) = pass1_res {
        res.model.clone()
    } else {
        let mut m = SonaModelF64::new(48_000.0);
        m.active_fir_taps = 1;
        m.fir_l[0] = 1.0;
        m.fir_r[0] = 1.0;
        m
    };

    if let Ok(encoded) =
        encode_with_preserved_controls(&mut fallback_model, dnam_bin, mono_out, dual_in)
    {
        *dnam_bin = encoded;
    }
    if let Some(ref mut res) = pass1_res {
        res.model.control_audit = fallback_model.control_audit.clone();
        res.model.control_bus = fallback_model.control_bus.clone();
    }
    let _ = verify_dnam_bin_zero_dc(dnam_bin);
    let fir_e: f64 = fallback_model.fir_l.iter().map(|&v| v * v).sum();
    let warn = check_rail_warning(dnam_bin, "Stage A probe");

    (
        StageAReport {
            elapsed_ms: elapsed_ms_since(timer),
            preserved_backbone: false,
            crossover_hz,
            early_fir_energy: fir_e,
            kautz_poles_lifted: fallback_model.active_kautz_pairs,
            phase2_rail_warning: warn,
        },
        pass1_res,
    )
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
    run_stage_a_physics_probe_with_result(
        manifest,
        dnam_bin,
        dry_lr,
        wet_lr,
        spectral_summary_4096,
        config,
    )
    .0
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

        // Full-length probing (> 12,000 frames) + Pass1ProbeResult handoff + manifest hints.
        let n_long = 16_000_usize;
        let dry_long: Vec<[f32; 2]> = (0..n_long)
            .map(|i| {
                let v = if i == 13_000 {
                    0.9_f32
                } else {
                    0.05 * ((i as f32) * 0.13).sin()
                };
                [v, v]
            })
            .collect();
        let mut wet_long = vec![[0.0_f32; 2]; n_long];
        for t in 0..n_long {
            wet_long[t][0] = 0.5 * dry_long[t][0]
                + if t >= 200 {
                    0.4 * dry_long[t - 200][0]
                } else {
                    0.0
                };
            wet_long[t][1] = 0.5 * dry_long[t][1]
                + if t >= 200 {
                    0.4 * dry_long[t - 200][1]
                } else {
                    0.0
                };
        }
        let mut bin_long = initialize_backbone_dnam_bin_if_needed(None).expect("neutral");
        let (rep_long, res_opt) = run_stage_a_physics_probe_with_result(
            &manifest,
            &mut bin_long,
            &dry_long,
            &wet_long,
            None,
            StageAConfig {
                preserve_existing_backbone: false,
                vp_gn_steps: 1,
            },
        );
        assert!(!rep_long.preserved_backbone);
        assert!(res_opt.is_some());
        blank.clear();
    }

    #[test]
    fn test_pass1_hint_from_manifest_consumes_extract_star_physics_metadata_keys() {
        use crate::stimulus::physics_probe::extract_star_physics_from_audio;

        let n = 8_192_usize;
        let sr = 48_000.0_f32;
        let mut dry_l = vec![0.0_f32; n];
        let mut wet_l = vec![0.0_f32; n];
        let mut wet_r = vec![0.0_f32; n];
        for t in 0..n {
            let ph = 2.0 * core::f32::consts::PI * 220.0 * (t as f32) / sr;
            dry_l[t] = 0.4 * ph.sin();
            let d = if t >= 240 { dry_l[t - 240] } else { 0.0 };
            wet_l[t] = 0.7 * dry_l[t] + 0.15 * dry_l[t] * dry_l[t] + 0.35 * d;
            wet_r[t] = 0.7 * dry_l[t] + 0.15 * dry_l[t] * dry_l[t] - 0.30 * d;
        }
        let analysis = extract_star_physics_from_audio(
            &dry_l,
            Some(&dry_l),
            &wet_l,
            Some(&wet_r),
            &dry_l,
            &wet_l,
            Some(&wet_r),
            &wet_l,
            Some(&wet_r),
            &[8, 16, 32],
        );

        let mut manifest = DnamProjManifest::default_reference();
        manifest
            .stems_registry
            .push(crate::proj::schema::StemRegistryEntry {
                stem_id: "star_center".to_string(),
                chunk_id: 1,
                enabled: true,
                is_star_center: true,
                controls_meta: Default::default(),
                provenance: crate::proj::schema::StemProvenance {
                    input_pcm_sha256: "0".repeat(64),
                    target_pcm_sha256: "0".repeat(64),
                    num_frames: n as u64,
                },
                user_metadata: analysis.metadata.clone(),
                extracted_physics: Some(analysis.extracted.clone()),
            });
        let raw_hint = pass1_hint_from_manifest(&manifest);
        assert!(raw_hint.thd_ratio.is_some());
        assert!(raw_hint.even_odd_asymmetry.is_some());
        assert!(raw_hint.has_stereo_width.is_some());

        {
            let stem = manifest.stems_registry.first_mut().expect("star stem");
            stem.user_metadata
                .insert("phys_rt60_ms".to_string(), serde_json::json!(420.0));
            stem.user_metadata
                .insert("phys_short_delay_ms".to_string(), serde_json::json!(6.5));
            stem.user_metadata
                .insert("phys_lfo_hz".to_string(), serde_json::json!(1.25));
            stem.user_metadata
                .insert("phys_odd_harm".to_string(), serde_json::json!(0.05));
            stem.user_metadata
                .insert("phys_even_harm".to_string(), serde_json::json!(0.15));
            stem.user_metadata
                .insert("phys_stereo_diff".to_string(), serde_json::json!(0.02));
        }

        let hint = pass1_hint_from_manifest(&manifest);
        assert!((hint.rt60_s.expect("rt60_s") - 0.42).abs() < 1.0e-6);
        assert!((hint.delay_ms.expect("delay_ms") - 6.5).abs() < 1.0e-6);
        assert!((hint.thd_ratio.expect("thd_ratio") - 0.20).abs() < 1.0e-6);
        assert!((hint.even_odd_asymmetry.expect("asym") - 0.5).abs() < 1.0e-6);
        assert!((hint.modulation_rate_hz.expect("mod_hz") - 1.25).abs() < 1.0e-6);
        assert_eq!(hint.has_stereo_width, Some(false));

        // Direct override keys take precedence over raw physics extractor keys.
        let stem = manifest.stems_registry.first_mut().expect("star stem");
        stem.user_metadata
            .insert("phys_rt60_s".to_string(), serde_json::json!(1.3));
        stem.user_metadata
            .insert("phys_has_stereo_width".to_string(), serde_json::json!(true));
        let overridden = pass1_hint_from_manifest(&manifest);
        assert!((overridden.rt60_s.expect("override rt60") - 1.3).abs() < 1.0e-6);
        assert_eq!(overridden.has_stereo_width, Some(true));
    }
}
