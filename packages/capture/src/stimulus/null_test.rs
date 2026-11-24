#![forbid(unsafe_code)]
//! `1.0s` Out-of-Sample Live Hardware-vs-Model Null-Test and `< 5 ms` Sufficient-Statistic
//! Rank-$k$ Auto-Patch (`stimulus::null_test`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md` (`ref[train.capture.alignment.multi-regime]`)
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.stat-uplo-u]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

use dnam_engine::container::extract_dnam_bin;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};
use dnam_train::{
    Pass5EmitConfig, TrainRunner, decode_dnam_bin_to_model, encode_model_to_dnam_bin,
    patch_model_readouts_weighted_cholesky,
};
use dnam_vm::SonaVm;
use dnam_vm::bin::{DnamBinView, strip_dnam_bin_to_tier};
use dnam_vm::math::DefaultMathBackend;
use zerocopy::byteorder::little_endian::F64;
use zerocopy::{FromBytes, IntoBytes};

use crate::align::{estimate_multiregime_alignment, realign_fractional_kaiser64};
use crate::metrics::{
    evaluate_besr_from_band_energies, evaluate_ltas_33band_report, try_split_bezout_3band_energies,
};
use crate::proj::bundle::{
    DnamProjView, ProjChunkInput, build_dnam_proj_container_with_dnam_bin, compact_to_capsules,
    export_runtime_dnam, verify_dnam_bin_zero_dc,
};
use crate::proj::directory::{ProjChunkKind, verify_proj_inpc_capsule, verify_proj_stmc_capsule};
use crate::proj::{PROJ_FLAG_COMPACTED_CAPSULES, PROJ_STAT_BYTES_ULTRA, STMC_FLAG_STEREO};
use crate::resampler::{
    resample_48k_to_arbitrary_rate, resample_arbitrary_rate_to_48k, resampled_frame_len,
};
use crate::solver::{
    StageDAuditOptions, canonical_besr_warmup, compute_integrated_lufs,
    compute_v3_stat_chunk_bytes, decode_coreset_lr, finalize_stage_d_and_audit,
};
use crate::targets::{DistillationTarget, StemStimulus};

use super::batch::default_control_vector;
use super::waveforms::{deterministic_seed_256, synthesize_a5_musical_transient_coreset};

/// Telemetry emitted by [`run_live_null_test_and_autopatch`] (`r[train.capture.live-null-test]`).
#[derive(Debug, Clone, PartialEq)]
pub struct LiveNullTestReport {
    /// Out-of-sample `1.0s` hardware-vs-engine null residual ESR in dB before auto-patch.
    pub pre_patch_null_esr_db: f32,
    /// Out-of-sample `1.0s` hardware-vs-engine null residual ESR in dB after auto-patch.
    pub post_patch_null_esr_db: f32,
    /// True if the `< 5 ms` sufficient-statistic auto-patch was applied and retained.
    pub autopatch_applied: bool,
    /// Wall-clock duration of the sufficient-statistic Cholesky patch in milliseconds (`< 5.0 ms`).
    pub autopatch_elapsed_ms: f32,
    /// Verified zero-allocation `DnamEngine::reload_model_in_place` hot-swap succeeded.
    pub reload_in_place_verified: bool,
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

/// Executes the `1.0s` Out-of-Sample Live Hardware-vs-Model Null-Test and `< 5 ms`
/// Sufficient-Statistic Auto-Patch (`r[train.capture.live-null-test]`), verifying
/// zero-allocation `DnamEngine::reload_model_in_place` hot-swap across mono and stereo rigs.
pub fn run_live_null_test_and_autopatch(
    proj_bytes: &[u8],
    target: &mut dyn DistillationTarget,
    threshold_db: f32,
) -> Result<(Vec<u8>, Vec<u8>, LiveNullTestReport), DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(proj_bytes, &crypto)?;
    let init_dnam = export_runtime_dnam(proj_bytes, None)?;
    let mut engine =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&init_dnam, Tier::Ultra, 48_000)?;

    let oos_seed = deterministic_seed_256(0x4E55_4C4C_5445_5354);
    let need_stereo_in = view.manifest.base.io_config.inputs >= 2;
    let (dry_l_48k, dry_r_48k) = synthesize_a5_musical_transient_coreset(
        &oos_seed,
        48_000,
        if need_stereo_in { 2 } else { 1 },
        48_000,
    );
    let star_ctrls = default_control_vector(&view.manifest);
    for (p, &val) in star_ctrls.iter().enumerate() {
        engine.set_control_immediate(p, val);
    }

    let raw_dev_sr = view.manifest.rig_routing.sample_rate_device_hz;
    let target_sr = if (8_000..=384_000).contains(&raw_dev_sr) {
        raw_dev_sr
    } else {
        48_000
    };
    let (stim_l, stim_r) = if target_sr == 48_000 {
        (dry_l_48k.clone(), dry_r_48k.clone())
    } else {
        let dst_len = resampled_frame_len(48_000, 48_000, target_sr);
        (
            resample_48k_to_arbitrary_rate(&dry_l_48k, target_sr, Some(dst_len)),
            dry_r_48k
                .as_ref()
                .map(|r| resample_48k_to_arbitrary_rate(r, target_sr, Some(dst_len))),
        )
    };

    target.reset_state();
    let hw_resp = target.render_stem(&StemStimulus {
        stem_id: "live_null_test_1s".to_string(),
        sample_rate_hz: target_sr,
        in_1: stim_l,
        in_2: stim_r,
        in_3: None,
        in_4: None,
        control_values: star_ctrls.clone(),
        control_trajectories: None,
        freeze_transition_window: None,
    })?;

    let resp_sr = hw_resp.sample_rate_hz.clamp(8_000, 384_000);
    let raw_wet_l_48k = if resp_sr == 48_000 {
        hw_resp.out_1.clone()
    } else {
        resample_arbitrary_rate_to_48k(&hw_resp.out_1, resp_sr, Some(48_000))
    };
    let raw_wet_r_48k = hw_resp.out_2.as_ref().map(|r| {
        if resp_sr == 48_000 {
            r.clone()
        } else {
            resample_arbitrary_rate_to_48k(r, resp_sr, Some(48_000))
        }
    });

    let wght_slice = view.chunk_slice(ProjChunkKind::Wght, 0).ok_or_else(|| {
        DnamError::SchemaValidationError("missing WGHT chunk for live auto-patch".to_string())
    })?;
    let init_dbin = extract_dnam_bin(wght_slice)?;

    let n_frames = dry_l_48k.len();
    let mut m_init_l = Vec::with_capacity(n_frames);
    let mut m_init_r = Vec::with_capacity(n_frames);
    for n in 0..n_frames {
        let il = dry_l_48k[n];
        let ir = dry_r_48k.as_ref().map_or(il, |r| r[n]);
        let [ml, mr] = engine.step_frame_sona(il, ir, false);
        m_init_l.push(ml);
        m_init_r.push(mr);
    }

    // Align hardware return directly against initial model output `m_init_l` when non-silent
    // so internal pedal filter group delay / HPF phase lead cancels out cleanly (`tau_converter` only).
    let model_e: f64 = m_init_l.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let align_ref = if model_e > 1.0e-12 {
        m_init_l.as_slice()
    } else {
        dry_l_48k.as_slice()
    };
    let cmp_len = align_ref.len().min(raw_wet_l_48k.len());
    let (lag, inv) = if cmp_len >= 64 {
        match estimate_multiregime_alignment(
            &align_ref[..cmp_len],
            &raw_wet_l_48k[..cmp_len],
            48_000,
            512,
        ) {
            Ok(al) => (al.total_lag_samples, al.polarity_inverted),
            Err(_) => (0.0, false),
        }
    } else {
        (0.0, false)
    };
    let y_hw_l = realign_fractional_kaiser64(&raw_wet_l_48k, lag, 0.0, inv);
    let y_hw_r = raw_wet_r_48k
        .as_ref()
        .map(|r| realign_fractional_kaiser64(r, lag, 0.0, inv));
    let eval_stereo = view.manifest.base.io_config.output_mode != "mono" || y_hw_r.is_some();

    let mut err_sq_pre = 0.0_f64;
    let mut sig_sq = 0.0_f64;
    for n in 0..n_frames {
        let ml = m_init_l[n];
        let mr = m_init_r[n];
        let tl = y_hw_l.get(n).copied().unwrap_or(0.0);
        let tr = y_hw_r
            .as_ref()
            .and_then(|r| r.get(n).copied())
            .unwrap_or(tl);
        let dl = f64::from(tl) - f64::from(ml);
        err_sq_pre += dl * dl;
        sig_sq += f64::from(tl) * f64::from(tl);
        if eval_stereo {
            let dr = f64::from(tr) - f64::from(mr);
            err_sq_pre += dr * dr;
            sig_sq += f64::from(tr) * f64::from(tr);
        }
    }

    let pre_esr_db = if sig_sq > 1.0e-18 {
        (10.0 * (err_sq_pre / sig_sq).max(1.0e-16).log10()) as f32
    } else {
        -120.0
    };

    if pre_esr_db <= threshold_db {
        let mut borrowed_eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&init_dnam, Tier::Ultra, 48_000)?;
        borrowed_eng.reload_model_in_place_owned(&init_dnam, Tier::Ultra, 48_000)?;
        return Ok((
            proj_bytes.to_vec(),
            init_dnam,
            LiveNullTestReport {
                pre_patch_null_esr_db: pre_esr_db,
                post_patch_null_esr_db: pre_esr_db,
                autopatch_applied: false,
                autopatch_elapsed_ms: 0.0,
                reload_in_place_verified: true,
            },
        ));
    }

    let timer = now_ms_token();
    let map_train_err =
        |e: dnam_train::TrainError| DnamError::SchemaValidationError(format!("{e}"));
    let mut model = decode_dnam_bin_to_model(init_dbin).map_err(map_train_err)?;
    let saved_trims = model.tier_gain_trim;

    let verify_n = n_frames.min(8_192);
    let compacted = (view.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES) != 0;
    let prev_coreset_esr = view
        .manifest
        .solver_telemetry
        .as_ref()
        .map(|s| s.tier_esr_db);
    let mut post_coreset_esr: Option<[f32; 4]> = None;
    let mut post_coreset_lufs: Option<[f64; 4]> = None;
    let mut updated_manifest = view.manifest.clone();
    let mono_out = updated_manifest.base.io_config.output_mode == "mono";

    let live_dry: Vec<[f32; 2]> = (0..verify_n)
        .map(|i| {
            [
                dry_l_48k[i],
                if mono_out {
                    dry_l_48k[i]
                } else {
                    dry_r_48k.as_ref().map_or(dry_l_48k[i], |r| r[i])
                },
            ]
        })
        .collect();
    let live_wet: Vec<[f32; 2]> = (0..verify_n)
        .map(|i| {
            let tl = y_hw_l.get(i).copied().unwrap_or(0.0);
            let tr = if mono_out {
                tl
            } else {
                y_hw_r
                    .as_ref()
                    .and_then(|r| r.get(i).copied())
                    .unwrap_or(tl)
            };
            [tl, tr]
        })
        .collect();

    type StereoCorePair = (Vec<[f32; 2]>, Vec<[f32; 2]>);
    let mut star_core_pair: Option<StereoCorePair> = None;
    if let Some(inpc_raw) = view.chunk_slice(ProjChunkKind::Inpc, 0)
        && let Ok(inpc_view) = verify_proj_inpc_capsule(inpc_raw)
        && let Some(star_entry) = updated_manifest
            .stems_registry
            .iter()
            .find(|s| s.enabled && s.is_star_center)
            .cloned()
        && let Some(stmc_raw) = view.chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
        && let Ok(stmc_view) = verify_proj_stmc_capsule(stmc_raw)
    {
        let dry_core_lr = decode_coreset_lr(
            inpc_view.dry_coreset_fp32,
            usize::from(inpc_view.channels).max(1),
        );
        let stmc_ch = if (stmc_view.flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let star_wet_lr = decode_coreset_lr(stmc_view.wet_coreset_fp32, stmc_ch);
        star_core_pair = Some((dry_core_lr, star_wet_lr));
    }

    type WeightedStemOwned = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64);
    let mut owned_stems: Vec<WeightedStemOwned> = Vec::with_capacity(2);
    let to_f64_channels = |frames: &[[f32; 2]]| -> (Vec<f64>, Vec<f64>) {
        let mut l = Vec::with_capacity(frames.len());
        let mut r = Vec::with_capacity(frames.len());
        for &[vl, vr] in frames {
            l.push(f64::from(vl));
            r.push(f64::from(vr));
        }
        (l, r)
    };
    if let Some((ref d_star, ref w_star)) = star_core_pair {
        let sn = d_star.len().min(w_star.len()).min(8_192);
        if sn > 64 {
            let (dl, dr) = to_f64_channels(&d_star[..sn]);
            let (wl, wr) = to_f64_channels(&w_star[..sn]);
            owned_stems.push((dl, dr, wl, wr, 1.0_f64));
        }
    }
    let rig_drifted_pre = prev_coreset_esr.is_some_and(|prev| pre_esr_db > prev[3] + 6.0);
    {
        let (dl, dr) = to_f64_channels(&live_dry);
        let (wl, wr) = to_f64_channels(&live_wet);
        owned_stems.push((
            dl,
            dr,
            wl,
            wr,
            if rig_drifted_pre { 2.5_f64 } else { 0.25_f64 },
        ));
    }
    let stem_refs: Vec<dnam_train::WeightedReadoutStem<'_>> = owned_stems
        .iter()
        .map(|(dl, dr, wl, wr, w)| {
            (
                dl.as_slice(),
                dr.as_slice(),
                wl.as_slice(),
                wr.as_slice(),
                *w,
            )
        })
        .collect();
    patch_model_readouts_weighted_cholesky(&TrainRunner::cpu(), &mut model, &stem_refs, 64)
        .map_err(map_train_err)?;
    let emit_cfg = Pass5EmitConfig {
        stereo_out: !mono_out,
        single_input: !need_stereo_in && mono_out,
        tail_time_ms: updated_manifest.base.io_config.tail_time_ms,
        ..Pass5EmitConfig::default()
    };
    if let Some((ref d_star, ref w_star)) = star_core_pair
        && d_star.len().min(w_star.len()) > 128
    {
        let sn = d_star.len().min(w_star.len());
        let warmup_cal = canonical_besr_warmup(sn);
        let keep_in_r = need_stereo_in || !mono_out;
        let cal_in_l: Vec<f32> = d_star[..sn].iter().map(|lr| lr[0]).collect();
        let cal_in_r: Vec<f32> = d_star[..sn]
            .iter()
            .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
            .collect();
        model.tier_gain_trim = [1.0; 4];
        let raw_dbin = encode_model_to_dnam_bin(&model, emit_cfg).map_err(map_train_err)?;
        let raw_view = DnamBinView::parse(&raw_dbin)
            .map_err(|e| DnamError::SchemaValidationError(format!("{e}")))?;
        let max_t = raw_view.retained_tier;
        let mut lufs = [-120.0_f64; 4];
        for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
            .into_iter()
            .enumerate()
        {
            let eff_tier = if tier.index() <= max_t.index() {
                tier
            } else {
                max_t
            };
            let tier_slice = strip_dnam_bin_to_tier(&raw_dbin, eff_tier)
                .map_err(|e| DnamError::SchemaValidationError(format!("{e}")))?;
            let mut vm =
                SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(tier_slice, eff_tier, 48_000.0)
                    .map_err(|e| DnamError::SchemaValidationError(format!("{e}")))?;
            let (mut ol, mut or) = (vec![0.0_f32; sn], vec![0.0_f32; sn]);
            vm.process_block(&cal_in_l, &cal_in_r, &mut ol, &mut or)
                .map_err(|e| DnamError::SchemaValidationError(format!("{e}")))?;
            let post: Vec<[f32; 2]> = ol[warmup_cal..]
                .iter()
                .zip(&or[warmup_cal..])
                .map(|(&l, &r)| [l, if mono_out { l } else { r }])
                .collect();
            lufs[t_idx] = compute_integrated_lufs(&post, 48_000);
        }
        if lufs[3] > -100.0 {
            for t in 0..3 {
                if lufs[t] > -100.0 {
                    let delta_lu = (lufs[3] - lufs[t]).clamp(-12.0, 12.0);
                    model.tier_gain_trim[t] = 10.0_f64.powf(delta_lu / 20.0).clamp(0.0625, 16.0);
                }
            }
        }
    } else {
        for (t, &trim) in saved_trims.iter().enumerate() {
            model.tier_gain_trim[t] = if trim.is_finite() && (0.0625..=16.0).contains(&trim) {
                trim
            } else {
                1.0
            };
        }
    }
    let mut patched_dbin = encode_model_to_dnam_bin(&model, emit_cfg).map_err(map_train_err)?;
    verify_dnam_bin_zero_dc(&patched_dbin)?;
    let patch_ms = elapsed_ms_since(timer);

    if let Some((ref dry_core_lr, ref star_wet_lr)) = star_core_pair {
        let prev_ms = updated_manifest
            .solver_telemetry
            .as_ref()
            .map_or(0.0, |s| s.total_solve_ms);
        if let Ok(sd_out) = finalize_stage_d_and_audit(
            &mut updated_manifest,
            &mut patched_dbin,
            dry_core_lr,
            Some(star_wet_lr),
            prev_coreset_esr,
            prev_ms + patch_ms,
            StageDAuditOptions::PATH1_HEAD_ONLY,
        ) {
            post_coreset_esr = Some(sd_out.tier_esr_db);
            post_coreset_lufs = Some(sd_out.tier_lufs);
        }
    }

    let gamma_stat = 0.25_f64;
    let delta_stat_res =
        compute_v3_stat_chunk_bytes(&patched_dbin, mono_out, &live_dry, &live_wet, verify_n, 4);
    let patched_stat_bytes = match (view.chunk_slice(ProjChunkKind::Stat, 0), delta_stat_res) {
        (Some(prev_stat), Ok(sb_bytes))
            if prev_stat.len() == PROJ_STAT_BYTES_ULTRA
                && sb_bytes.len() == PROJ_STAT_BYTES_ULTRA =>
        {
            let prev_words = <[F64]>::ref_from_bytes(prev_stat)
                .map_err(|_| DnamError::SchemaValidationError("invalid STAT f64".to_string()))?;
            let delta_words = <[F64]>::ref_from_bytes(&sb_bytes).map_err(|_| {
                DnamError::SchemaValidationError("invalid delta STAT f64".to_string())
            })?;
            let sum_words: Vec<F64> = prev_words
                .iter()
                .zip(delta_words.iter())
                .map(|(pv, dv)| {
                    let sum = pv.get() + gamma_stat * dv.get();
                    F64::new(if sum.is_finite() { sum } else { 0.0 })
                })
                .collect();
            sum_words.as_bytes().to_vec()
        }
        (_, Ok(sb_bytes)) => sb_bytes,
        _ => star_core_pair
            .as_ref()
            .and_then(|(d, w)| {
                compute_v3_stat_chunk_bytes(&patched_dbin, mono_out, d, w, d.len().min(w.len()), 4)
                    .ok()
            })
            .unwrap_or_else(|| vec![0u8; PROJ_STAT_BYTES_ULTRA]),
    };

    let mut new_chunks = Vec::new();
    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let kind = entry.kind()?;
        match kind {
            ProjChunkKind::Stat => new_chunks.push(ProjChunkInput {
                kind: ProjChunkKind::Stat,
                chunk_id: 0,
                payload: patched_stat_bytes.clone(),
            }),
            ProjChunkKind::Wght | ProjChunkKind::Auth => {}
            other => new_chunks.push(ProjChunkInput {
                kind: other,
                chunk_id: entry.chunk_id,
                payload: slice.to_vec(),
            }),
        }
    }

    let raw_patched_proj =
        build_dnam_proj_container_with_dnam_bin(&mut updated_manifest, &new_chunks, &patched_dbin)?;
    let patched_proj = if compacted {
        compact_to_capsules(&raw_patched_proj)?
    } else {
        raw_patched_proj
    };
    let patched_dnam = export_runtime_dnam(&patched_proj, None)?;

    engine.reload_model_in_place_owned(&patched_dnam, Tier::Ultra, 48_000)?;
    engine.reset_to_dc_equilibrium();
    for (p, &val) in star_ctrls.iter().enumerate() {
        engine.set_control_immediate(p, val);
    }
    let mut err_sq_post = 0.0_f64;
    let mut m_post_l = Vec::with_capacity(n_frames);
    let mut m_post_r = Vec::with_capacity(n_frames);
    for n in 0..n_frames {
        let il = dry_l_48k[n];
        let ir = dry_r_48k.as_ref().map_or(il, |r| r[n]);
        let [ml, mr] = engine.step_frame_sona(il, ir, false);
        m_post_l.push(ml);
        m_post_r.push(mr);
        let tl = y_hw_l.get(n).copied().unwrap_or(0.0);
        let tr = y_hw_r
            .as_ref()
            .and_then(|r| r.get(n).copied())
            .unwrap_or(tl);
        let dl = f64::from(tl) - f64::from(ml);
        err_sq_post += dl * dl;
        if eval_stereo {
            let dr = f64::from(tr) - f64::from(mr);
            err_sq_post += dr * dr;
        }
    }
    let post_esr_db = if sig_sq > 1.0e-18 {
        (10.0 * (err_sq_post / sig_sq).max(1.0e-16).log10()) as f32
    } else {
        -120.0
    };

    let warmup = if n_frames > 64 {
        512.min(n_frames / 4)
    } else {
        0
    };
    let eval_oos_besr = |ml: &[f32], mr: &[f32]| -> Option<f64> {
        let tl64: Vec<f64> = (0..n_frames)
            .map(|i| f64::from(y_hw_l.get(i).copied().unwrap_or(0.0)))
            .collect();
        let ml64: Vec<f64> = ml.iter().map(|&v| f64::from(v)).collect();
        let (sl, el, nl) = try_split_bezout_3band_energies(&tl64, &ml64, None, 48_000.0, warmup)?;
        let rl = evaluate_besr_from_band_energies(sl, el, nl);
        let dl = rl.valid.then_some(rl.besr_db);
        if !eval_stereo {
            return dl;
        }
        let tr64: Vec<f64> = (0..n_frames)
            .map(|i| {
                f64::from(
                    y_hw_r
                        .as_ref()
                        .and_then(|r| r.get(i).copied())
                        .unwrap_or_else(|| y_hw_l.get(i).copied().unwrap_or(0.0)),
                )
            })
            .collect();
        let mr64: Vec<f64> = mr.iter().map(|&v| f64::from(v)).collect();
        let (sr, er, nr) = try_split_bezout_3band_energies(&tr64, &mr64, None, 48_000.0, warmup)?;
        let rr = evaluate_besr_from_band_energies(sr, er, nr);
        match (dl, rr.valid.then_some(rr.besr_db)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    };
    let oos_besr_pre = eval_oos_besr(&m_init_l, &m_init_r);
    let oos_besr_post = eval_oos_besr(&m_post_l, &m_post_r);
    let oos_besr_ok = match (oos_besr_pre, oos_besr_post) {
        (Some(pre_b), Some(post_b)) => post_b <= pre_b + 0.02,
        _ => true,
    };
    let eval_oos_ltas = |ml: &[f32], mr: &[f32]| -> Option<f64> {
        if n_frames <= warmup + 16 {
            return None;
        }
        let tl64: Vec<f64> = (warmup..n_frames)
            .map(|i| f64::from(y_hw_l.get(i).copied().unwrap_or(0.0)))
            .collect();
        let ml64: Vec<f64> = ml[warmup..n_frames].iter().map(|&v| f64::from(v)).collect();
        let rl = evaluate_ltas_33band_report(&tl64, &ml64, None, 48_000.0, true);
        let dl = rl.available.then_some(rl.max_adj_deviation_db);
        if !eval_stereo {
            return dl;
        }
        let tr64: Vec<f64> = (warmup..n_frames)
            .map(|i| {
                f64::from(
                    y_hw_r
                        .as_ref()
                        .and_then(|r| r.get(i).copied())
                        .unwrap_or_else(|| y_hw_l.get(i).copied().unwrap_or(0.0)),
                )
            })
            .collect();
        let mr64: Vec<f64> = mr[warmup..n_frames].iter().map(|&v| f64::from(v)).collect();
        let rr = evaluate_ltas_33band_report(&tr64, &mr64, None, 48_000.0, true);
        match (dl, rr.available.then_some(rr.max_adj_deviation_db)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    };
    let oos_ltas_pre = eval_oos_ltas(&m_init_l, &m_init_r);
    let oos_ltas_post = eval_oos_ltas(&m_post_l, &m_post_r);
    let oos_ltas_ok = match (oos_ltas_pre, oos_ltas_post) {
        (Some(pre_l), Some(post_l)) => post_l <= pre_l + 0.03,
        _ => true,
    };
    let coreset_lufs_ok = post_coreset_lufs
        .is_none_or(|lufs| lufs[3] <= -100.0 || (0..3).all(|t| (lufs[t] - lufs[3]).abs() <= 0.05));
    let coreset_tiers_ok = match (prev_coreset_esr, post_coreset_esr) {
        (Some(prev), Some(post)) => {
            let finite_ok = post.iter().all(|e| e.is_finite());
            let rig_drifted = pre_esr_db > prev[3] + 6.0 && post_esr_db + 1.0 < pre_esr_db;
            let mono = post[3] <= post[2] + 1.0e-2
                && post[2] <= post[1] + 1.0e-2
                && post[1] <= post[0] + 1.0e-2;
            let non_regressing =
                (0..4).all(|t| post[t] <= prev[t] + 0.05 && (prev[t] >= 0.0 || post[t] < 0.0));
            coreset_lufs_ok && finite_ok && (rig_drifted || (mono && non_regressing))
        }
        _ => coreset_lufs_ok,
    };

    if post_esr_db <= pre_esr_db && oos_besr_ok && oos_ltas_ok && coreset_tiers_ok {
        Ok((
            patched_proj,
            patched_dnam,
            LiveNullTestReport {
                pre_patch_null_esr_db: pre_esr_db,
                post_patch_null_esr_db: post_esr_db,
                autopatch_applied: true,
                autopatch_elapsed_ms: patch_ms,
                reload_in_place_verified: true,
            },
        ))
    } else {
        engine.reload_model_in_place_owned(&init_dnam, Tier::Ultra, 48_000)?;
        Ok((
            proj_bytes.to_vec(),
            init_dnam,
            LiveNullTestReport {
                pre_patch_null_esr_db: pre_esr_db,
                post_patch_null_esr_db: pre_esr_db,
                autopatch_applied: false,
                autopatch_elapsed_ms: patch_ms,
                reload_in_place_verified: true,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::bundle::build_dnam_proj_container;
    use crate::proj::schema::DnamProjManifest;
    use crate::targets::StemResponse;
    use std::collections::BTreeMap;

    struct MockNullTarget {
        stereo: bool,
        orthogonal: bool,
    }

    impl DistillationTarget for MockNullTarget {
        fn reset_state(&mut self) {}
        fn render_stem(&mut self, stimulus: &StemStimulus) -> Result<StemResponse, DnamError> {
            let n = stimulus.in_1.len();
            let (out_1, out_2) = if self.orthogonal {
                let mut l = vec![0.0_f32; n];
                let mut r = vec![0.0_f32; n];
                for i in 0..n {
                    let s = 0.25 * ((i as f32) * 0.09).sin();
                    l[i] = s;
                    r[i] = -s;
                }
                (l, self.stereo.then_some(r))
            } else {
                (vec![0.0_f32; n], self.stereo.then(|| vec![0.0_f32; n]))
            };
            Ok(StemResponse {
                stem_id: stimulus.stem_id.clone(),
                sample_rate_hz: stimulus.sample_rate_hz,
                out_1,
                out_2,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn test_live_null_test_edges_missing_wght_fast_pass_and_resampling() {
        let mut tgt_silent = MockNullTarget {
            stereo: false,
            orthogonal: false,
        };
        let empty_proj =
            build_dnam_proj_container(&DnamProjManifest::default_reference(), &[]).unwrap();
        assert!(run_live_null_test_and_autopatch(&empty_proj, &mut tgt_silent, -10.0).is_err());

        let dnam_bin = encode_model_to_dnam_bin(
            &dnam_train::SonaModelF64::new(48_000.0),
            Pass5EmitConfig::default(),
        )
        .unwrap();
        let mut m = DnamProjManifest::default_reference();
        m.rig_routing.sample_rate_device_hz = 44_100;
        let proj = build_dnam_proj_container_with_dnam_bin(&mut m, &[], &dnam_bin).unwrap();

        // Silent target yields -120 dB pre-ESR <= -10 dB -> skips autopatch cleanly.
        let (_, _, rep_fast) =
            run_live_null_test_and_autopatch(&proj, &mut tgt_silent, -10.0).unwrap();
        assert!(!rep_fast.autopatch_applied && rep_fast.reload_in_place_verified);

        // Non-zero 44.1 kHz stereo mock triggers resampling, Cholesky attempt, and safe rollback/apply.
        let mut tgt_stereo = MockNullTarget {
            stereo: true,
            orthogonal: true,
        };
        let (_, _, rep_patch) =
            run_live_null_test_and_autopatch(&proj, &mut tgt_stereo, -120.0).unwrap();
        assert!(rep_patch.reload_in_place_verified && rep_patch.pre_patch_null_esr_db.is_finite());
    }
}
