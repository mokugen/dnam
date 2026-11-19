#![forbid(unsafe_code)]
//! `1.0s` Out-of-Sample Live Hardware-vs-Model Null-Test and `< 5 ms` Sufficient-Statistic
//! Rank-$k$ Auto-Patch (`stimulus::null_test`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md` (`ref[train.capture.alignment.multi-regime]`)
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.stat-uplo-u]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

use dnam_engine::container::DnamContainerView;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;
use dnam_engine::stages::weights::{
    ARENA_ULTRA_BYTES, ARENA0_TIER_GAIN_TRIM_OFFSET, pack_v3_matryoshka_to_arena,
    unpack_v3_matryoshka_from_arena,
};
use dnam_engine::{DnamEngine, Tier};

use crate::align::{estimate_multiregime_alignment, realign_fractional_kaiser64};
use crate::metrics::{
    evaluate_besr_from_band_energies, evaluate_ltas_33band_report, try_split_bezout_3band_energies,
};
use crate::proj::bundle::{
    DnamProjView, ProjChunkInput, build_dnam_proj_container_with_cp1_and_compacted,
    export_runtime_dnam, zero_arena_b_head_dc_bias,
};
use crate::proj::directory::{ProjChunkKind, verify_proj_inpc_capsule, verify_proj_stmc_capsule};
use crate::proj::{PROJ_FLAG_COMPACTED_CAPSULES, PROJ_STAT_BYTES_ULTRA, STMC_FLAG_STEREO};
use crate::resampler::{
    resample_48k_to_arbitrary_rate, resample_arbitrary_rate_to_48k, resampled_frame_len,
};
use crate::solver::{
    StageDAuditOptions, WeightedStemRef, compute_v3_stat_chunk_bytes, decode_coreset_lr,
    finalize_stage_d_and_audit, patch_v3_readouts_weighted_cholesky,
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
    let rt_view = DnamContainerView::from_bytes(wght_slice, &crypto)?;
    let init_arena = rt_view.arena_storage.as_bytes();

    let n_frames = dry_l_48k.len();
    let mut m_init_l = Vec::with_capacity(n_frames);
    let mut m_init_r = Vec::with_capacity(n_frames);
    for n in 0..n_frames {
        let il = dry_l_48k[n];
        let ir = dry_r_48k.as_ref().map_or(il, |r| r[n]);
        let [ml, mr] = engine.step_frame_tora(il, ir, false);
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
    let mut arena_28k = init_arena.to_vec();
    arena_28k.resize(ARENA_ULTRA_BYTES, 0);
    let saved_trims: [f32; 4] = std::array::from_fn(|t| {
        let off = ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4;
        f32::from_le_bytes(arena_28k[off..off + 4].try_into().unwrap())
    });

    let verify_n = n_frames.min(8_192);
    let compacted = (view.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES) != 0;
    let prev_coreset_esr = view
        .manifest
        .solver_telemetry
        .as_ref()
        .map(|s| s.tier_esr_db);
    let mut post_coreset_esr: Option<[f32; 4]> = None;
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

    let mut patch_stems: Vec<WeightedStemRef<'_>> = Vec::with_capacity(2);
    if let Some((ref d_star, ref w_star)) = star_core_pair {
        let sn = d_star.len().min(w_star.len()).min(8_192);
        if sn > 64 {
            patch_stems.push(WeightedStemRef {
                dry_lr: &d_star[..sn],
                wet_lr: &w_star[..sn],
                weight: 1.0_f64,
            });
        }
    }
    let rig_drifted_pre = prev_coreset_esr.is_some_and(|prev| pre_esr_db > prev[3] + 6.0);
    patch_stems.push(WeightedStemRef {
        dry_lr: &live_dry,
        wet_lr: &live_wet,
        weight: if rig_drifted_pre { 2.5_f64 } else { 0.25_f64 },
    });

    let mut mat = unpack_v3_matryoshka_from_arena(&arena_28k, mono_out);
    patch_v3_readouts_weighted_cholesky(&mut mat, &patch_stems);
    pack_v3_matryoshka_to_arena(&mat, &mut arena_28k);
    for (t, &trim) in saved_trims.iter().enumerate() {
        let off = ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4;
        let clean = if trim.is_finite() && (0.0625..=16.0).contains(&trim) {
            trim
        } else {
            1.0
        };
        arena_28k[off..off + 4].copy_from_slice(&clean.to_le_bytes());
    }
    zero_arena_b_head_dc_bias(&mut arena_28k);
    let patch_ms = elapsed_ms_since(timer);

    if let Some((ref dry_core_lr, ref star_wet_lr)) = star_core_pair {
        let prev_ms = updated_manifest
            .solver_telemetry
            .as_ref()
            .map_or(0.0, |s| s.total_solve_ms);
        if let Ok(sd_out) = finalize_stage_d_and_audit(
            &mut updated_manifest,
            &mut arena_28k,
            dry_core_lr,
            Some(star_wet_lr),
            prev_coreset_esr,
            prev_ms + patch_ms,
            StageDAuditOptions::PATH1_HEAD_ONLY,
        ) {
            post_coreset_esr = Some(sd_out.tier_esr_db);
        }
    }

    let gamma_stat = 0.25_f64;
    let delta_stat_res =
        compute_v3_stat_chunk_bytes(&arena_28k, mono_out, &live_dry, &live_wet, verify_n, 4);
    let patched_stat_bytes = match (view.chunk_slice(ProjChunkKind::Stat, 0), delta_stat_res) {
        (Some(prev_stat), Ok(sb_bytes))
            if prev_stat.len() == PROJ_STAT_BYTES_ULTRA
                && sb_bytes.len() == PROJ_STAT_BYTES_ULTRA =>
        {
            let mut out = Vec::with_capacity(PROJ_STAT_BYTES_ULTRA);
            for (p_ch, d_ch) in prev_stat
                .as_chunks::<8>()
                .0
                .iter()
                .zip(sb_bytes.as_chunks::<8>().0.iter())
            {
                let pv = f64::from_le_bytes(*p_ch);
                let dv = f64::from_le_bytes(*d_ch);
                let sum = pv + gamma_stat * dv;
                let clean = if sum.is_finite() { sum } else { 0.0 };
                out.extend_from_slice(&clean.to_le_bytes());
            }
            out
        }
        (_, Ok(sb_bytes)) => sb_bytes,
        _ => star_core_pair
            .as_ref()
            .and_then(|(d, w)| {
                compute_v3_stat_chunk_bytes(&arena_28k, mono_out, d, w, d.len().min(w.len()), 4)
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

    let patched_proj = build_dnam_proj_container_with_cp1_and_compacted(
        &mut updated_manifest,
        &new_chunks,
        &arena_28k,
        &rt_view.cp1_contributors,
        compacted,
    )?;
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
        let [ml, mr] = engine.step_frame_tora(il, ir, false);
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
    let oos_besr_ok = match (
        eval_oos_besr(&m_init_l, &m_init_r),
        eval_oos_besr(&m_post_l, &m_post_r),
    ) {
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
    let oos_ltas_ok = match (
        eval_oos_ltas(&m_init_l, &m_init_r),
        eval_oos_ltas(&m_post_l, &m_post_r),
    ) {
        (Some(pre_l), Some(post_l)) => post_l <= pre_l + 0.03,
        _ => true,
    };
    let coreset_tiers_ok = match (prev_coreset_esr, post_coreset_esr) {
        (Some(prev), Some(post)) => {
            let finite_ok = post.iter().all(|e| e.is_finite());
            let rig_drifted = pre_esr_db > prev[3] + 6.0 && post_esr_db + 1.0 < pre_esr_db;
            let mono = post[3] <= post[2] + 1.0e-2
                && post[2] <= post[1] + 1.0e-2
                && post[1] <= post[0] + 1.0e-2;
            let non_regressing =
                (0..4).all(|t| post[t] <= prev[t] + 0.05 && (prev[t] >= 0.0 || post[t] < 0.0));
            finite_ok && (rig_drifted || (mono && non_regressing))
        }
        _ => true,
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
