#![forbid(unsafe_code)]
//! Stage A V2 Blind Physical Extension Lifting (`solver::stage_a_ext`, `REQ-1.6`).
//!
//! Populates `Arena 0` V2 extension blocks (`ARENA0_STAGE1_EXT`, `ARENA0_STAGE2_EXT`,
//! `ARENA0_STAGE4_EXT`, `PIPE_FLAG_ALL_V2`, Stage 0 sidechain isolation, and multi-tier Stage II
//! nonlinear branches `4_224..4_272`) purely from blind audio physics & `phys_*` metadata, followed
//! by strictly-improving coordinate refinement on `StarCenter`.

use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::stages::weights::{
    ARENA0_STAGE1_EXT_OFFSET as ARENA0_STAGE1_EXT, ARENA0_STAGE2_EXT_OFFSET as ARENA0_STAGE2_EXT,
    ARENA0_STAGE4_EXT_OFFSET as ARENA0_STAGE4_EXT, PIPE_FLAG_ALL_V2, PIPE_MAGIC_HI,
};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};

use crate::proj::schema::DnamProjManifest;

use super::solve_uplo_u_cholesky_2ch;
use super::stage_a::STAGE3_EARLY_FIR_BASES;
use super::stage_a_refl::{fit_svf_tpt_surrogate, solve_svf_weights_3x3};

#[inline]
pub(super) fn write_f32_le(buf: &mut [u8], off: usize, val: f32) {
    if off + 4 <= buf.len() {
        let clean = if val.is_finite() {
            if val == 0.0 { 0.0 } else { val }
        } else {
            0.0
        };
        buf[off..off + 4].copy_from_slice(&clean.to_le_bytes());
    }
}

#[inline]
fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    if off + 4 <= buf.len() {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    } else {
        0.0
    }
}

pub(crate) fn star_has_meta(manifest: &DnamProjManifest, key: &str) -> bool {
    manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .or_else(|| manifest.stems_registry.iter().find(|s| s.enabled))
        .is_some_and(|s| s.user_metadata.contains_key(key))
}

pub(crate) fn star_meta_f64(manifest: &DnamProjManifest, key: &str, def: f64) -> f64 {
    manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id.starts_with("star"))
        .or_else(|| manifest.stems_registry.first())
        .and_then(|s| s.user_metadata.get(key))
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite())
        .unwrap_or(def)
}

fn solve_svf_lp_bp_hp_weights(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    n_eval: usize,
) -> Option<[f32; 3]> {
    let mut probe = arena_64k.to_vec();
    let _ = super::stages::sync_clamp_descriptors_to_arena(&manifest.base, &mut probe);
    let mut hdr = manifest.base.clone();
    let bytes = build_dnam_container_with_cp1(&mut hdr, &probe, &probe, false, &[]).ok()?;
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes(&bytes, Tier::Full, 48_000).ok()?;
    let star_ctrls = super::star_center_control_values(manifest);
    for (p, &v) in star_ctrls.iter().enumerate() {
        eng.set_control_immediate(p, v);
    }
    let _ftz = FtzDazGuard::new();
    let eval_stereo = manifest.base.io_config.output_mode != "mono";
    let mut svf_basis = Vec::with_capacity(n_eval);
    let mut tgt = Vec::with_capacity(n_eval);
    let mut fb = [0.0_f32; 256];
    for i in 0..n_eval {
        let _ = eng.step_frame_with_feature_bus(dry_lr[i][0], dry_lr[i][1], &mut fb);
        svf_basis.push([f64::from(fb[4]), f64::from(fb[5]), f64::from(fb[6])]);
        tgt.push(if eval_stereo {
            0.5 * f64::from(wet_lr[i][0] + wet_lr[i][1])
        } else {
            f64::from(wet_lr[i][0])
        });
    }
    let start = n_eval.min(512);
    if n_eval <= start + 8 {
        return None;
    }
    let (w, _, _) = solve_svf_weights_3x3(&svf_basis, &tgt, start, n_eval)?;
    let max_abs = w.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let scale = if max_abs > 1.5 { max_abs / 1.5 } else { 1.0 };
    Some([
        (w[0] / scale).clamp(-2.0, 2.0) as f32,
        (w[1] / scale).clamp(-2.0, 2.0) as f32,
        (w[2] / scale).clamp(-2.0, 2.0) as f32,
    ])
}

/// Evaluates Stage I + II nonlinear distortion error after factoring out a causal 8-tap LTI
/// Wiener projection + `clean_sub` regressor (`Stage III` surrogate) on `dry_lr[..n_eval]`.
fn eval_stage12_nl_loss(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    n_eval: usize,
) -> f64 {
    let mut probe = arena_64k.to_vec();
    let _ = super::stages::sync_clamp_descriptors_to_arena(&manifest.base, &mut probe);
    for &fb in &STAGE3_EARLY_FIR_BASES {
        if fb + 128 <= probe.len() {
            probe[fb..fb + 128].fill(0);
        }
    }
    write_f32_le(&mut probe, STAGE3_EARLY_FIR_BASES[0], 1.0);
    write_f32_le(&mut probe, STAGE3_EARLY_FIR_BASES[0] + 64, 1.0);
    // Bypass Kautz and Stage IV during pre-S3 nonlinear envelope/harmonic scoring
    for off in [
        ARENA0_STAGE4_EXT,
        ARENA0_STAGE4_EXT + 4,
        ARENA0_STAGE4_EXT + 28,
        4_688,
    ] {
        write_f32_le(&mut probe, off, 0.0);
    }

    let mut hdr = manifest.base.clone();
    let Ok(bytes) = build_dnam_container_with_cp1(&mut hdr, &probe, &probe, false, &[]) else {
        return f64::INFINITY;
    };
    let Ok(mut eng) = DnamEngine::<DefaultMathBackend>::from_bytes(&bytes, Tier::Full, 48_000)
    else {
        return f64::INFINITY;
    };
    let star_ctrls = super::star_center_control_values(manifest);
    for (p, &v) in star_ctrls.iter().enumerate() {
        eng.set_control_immediate(p, v);
    }
    let _ftz = FtzDazGuard::new();
    let svf_active = read_f32_le(arena_64k, ARENA0_STAGE1_EXT + 100) > 0.5;
    let clean_bypass = read_f32_le(arena_64k, ARENA0_STAGE2_EXT + 24);
    let mut mdl = vec![0.0_f64; n_eval];
    let mut sub = vec![0.0_f64; n_eval];
    let mut svf_basis = Vec::with_capacity(if svf_active { n_eval } else { 0 });
    let mut tgt = vec![0.0_f64; n_eval];
    let mut fb = [0.0_f32; 256];
    let eval_stereo = manifest.base.io_config.output_mode != "mono";
    for i in 0..n_eval {
        let _ = eng.step_frame_with_feature_bus(dry_lr[i][0], dry_lr[i][1], &mut fb);
        mdl[i] = f64::from(fb[58] - fb[76]);
        sub[i] = f64::from(fb[60]);
        if svf_active {
            svf_basis.push([f64::from(fb[4]), f64::from(fb[5]), f64::from(fb[6])]);
        }
        tgt[i] = if eval_stereo {
            0.5 * f64::from(wet_lr[i][0] + wet_lr[i][1])
        } else {
            f64::from(wet_lr[i][0])
        };
    }

    const K: usize = 8;
    const K_LP: usize = 3;
    const K_TOT: usize = K + K_LP + 1;
    let start = n_eval.min(512);
    if n_eval <= start + K {
        return f64::INFINITY;
    }
    if svf_active
        && (clean_bypass >= 0.80 || star_meta_f64(manifest, "phys_odd_harm", 0.0) < 0.08)
        && let Some((w_svf, rel_err, _)) = solve_svf_weights_3x3(&svf_basis, &tgt, start, n_eval)
    {
        for i in 0..n_eval {
            mdl[i] = w_svf[0] * svf_basis[i][0]
                + w_svf[1] * svf_basis[i][1]
                + w_svf[2] * svf_basis[i][2];
        }
        // Score direct 3x3 fit + envelope tracking
        let _ = rel_err;
    }
    const LP_ALPHAS: [f64; K_LP] = [0.9590, 0.8947, 0.7304];
    let mut lp_mdl = vec![[0.0_f64; K_LP]; n_eval];
    let mut lp_st = [0.0_f64; K_LP];
    for i in 0..n_eval {
        let x = mdl[i];
        for p in 0..K_LP {
            lp_st[p] = LP_ALPHAS[p] * lp_st[p] + (1.0 - LP_ALPHAS[p]) * x;
            lp_mdl[i][p] = lp_st[p];
        }
    }
    let mut gram = vec![0.0_f64; K_TOT * (K_TOT + 1) / 2];
    let mut rhs = [[0.0_f64; 2]; K_TOT];
    let mut y_e = 0.0_f64;
    for i in start..n_eval {
        let y = tgt[i];
        y_e += y * y;
        let mut reg = [0.0_f64; K_TOT];
        for c in 0..K {
            reg[c] = mdl[i - c];
        }
        reg[K..(K_LP + K)].copy_from_slice(&lp_mdl[i][..K_LP]);
        reg[K + K_LP] = sub[i];
        for c in 0..K_TOT {
            let xc = reg[c];
            let col = super::uplo_u_index(0, c);
            for r in 0..=c {
                gram[col + r] += reg[r] * xc;
            }
            rhs[c][0] += xc * y;
        }
    }
    if y_e <= 1.0e-12 {
        return 0.0;
    }
    let tr: f64 = (0..K_TOT).map(|k| gram[super::uplo_u_index(k, k)]).sum();
    let Ok(h) = solve_uplo_u_cholesky_2ch(&gram, &rhs, (1.0e-5 * tr).max(1.0e-12)) else {
        return f64::INFINITY;
    };
    let mut err_e = 0.0_f64;
    let mut pred_vec = vec![0.0_f64; n_eval];
    for i in start..n_eval {
        let mut pred = h[K + K_LP][0] * sub[i];
        for c in 0..K {
            pred += h[c][0] * mdl[i - c];
        }
        for p in 0..K_LP {
            pred += h[K + p][0] * lp_mdl[i][p];
        }
        pred_vec[i] = pred;
        let d = tgt[i] - pred;
        err_e += d * d;
    }
    // Also score 64-sample RMS envelope tracking so comp/gate dynamics refine accurately
    let blk = 64_usize;
    let mut env_err = 0.0_f64;
    let (mut sum_m, mut sum_t) = (0.0_f64, 0.0_f64);
    for off in (start..n_eval - blk).step_by(blk) {
        let em: f64 = (0..blk)
            .map(|k| pred_vec[off + k] * pred_vec[off + k])
            .sum::<f64>()
            .sqrt();
        let et: f64 = (0..blk)
            .map(|k| tgt[off + k] * tgt[off + k])
            .sum::<f64>()
            .sqrt();
        sum_m += em;
        sum_t += et;
    }
    let sc = if sum_m > 1.0e-9 { sum_t / sum_m } else { 1.0 };
    for off in (start..n_eval - blk).step_by(blk) {
        let em: f64 = (0..blk)
            .map(|k| pred_vec[off + k] * pred_vec[off + k])
            .sum::<f64>()
            .sqrt()
            * sc;
        let et: f64 = (0..blk)
            .map(|k| tgt[off + k] * tgt[off + k])
            .sum::<f64>()
            .sqrt();
        env_err += (et - em) * (et - em);
    }
    (err_e + 0.35 * env_err * (blk as f64)) / y_e
}

pub(super) fn coord_refine_arena_scalars(
    offsets: &[(usize, f32, f32)],
    best_loss: &mut f64,
    arena_64k: &mut [u8],
    eval_fn: &mut impl FnMut(&[u8]) -> f64,
) {
    if offsets.is_empty() {
        return;
    }
    for _pass in 0..2 {
        for &(off, lo, hi) in offsets {
            let cur = read_f32_le(arena_64k, off).clamp(lo, hi);
            let span = (hi - lo).max(1.0e-6);
            let base_step = (cur.abs() * 0.25).max(0.05 * span);
            let mut local_best_v = cur;
            for &mult in &[0.0_f32, 0.65, 1.35, -1.0] {
                let cand = if mult == 0.0 {
                    if lo <= 0.0 && hi >= 0.0 {
                        0.0
                    } else {
                        continue;
                    }
                } else if mult < 0.0 {
                    (cur - base_step).clamp(lo, hi)
                } else if cur.abs() > 1.0e-5 {
                    (cur * mult).clamp(lo, hi)
                } else {
                    (cur + base_step * (mult - 0.5)).clamp(lo, hi)
                };
                if (cand - local_best_v).abs() <= 1.0e-6 {
                    continue;
                }
                write_f32_le(arena_64k, off, cand);
                let l = eval_fn(arena_64k);
                if l < *best_loss * 0.996 {
                    *best_loss = l;
                    local_best_v = cand;
                } else {
                    write_f32_le(arena_64k, off, local_best_v);
                }
            }
        }
    }
}

/// Blind Stage 0 + V2 Stage I & Stage II lifting before Stage III FIR/Kautz (`REQ-1.6`).
pub(crate) fn lift_v2_pre_s3(
    manifest: &DnamProjManifest,
    arena_64k: &mut [u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    n_fit: usize,
    nl_drive: f32,
    amp_scale: f32,
) {
    if arena_64k.len() < 4_772 || n_fit < 512 || amp_scale <= 0.0 {
        return;
    }
    let flags = PIPE_FLAG_ALL_V2 | PIPE_MAGIC_HI;
    arena_64k[64..68].copy_from_slice(&flags.to_le_bytes());

    let default_coherence = (1.0 - f64::from(nl_drive)).clamp(0.0, 1.0);
    let coherence = star_meta_f64(manifest, "phys_coherence", default_coherence) as f32;
    let odd_harm = star_meta_f64(manifest, "phys_odd_harm", 0.0) as f32;
    let even_harm = star_meta_f64(manifest, "phys_even_harm", 0.0) as f32;
    let subharm = star_meta_f64(manifest, "phys_subharm_ratio", 0.0) as f32;
    let sideband = star_meta_f64(manifest, "phys_sideband_ratio", 0.0) as f32;
    let ringmod_hz = star_meta_f64(manifest, "phys_ringmod_hz", 0.0) as f32;
    let pre_hpf_hz = star_meta_f64(manifest, "phys_pre_hpf_hz", 0.0) as f32;
    let pre_lpf_hz = star_meta_f64(manifest, "phys_pre_lpf_hz", 20_000.0) as f32;
    let sub_bass_db = star_meta_f64(manifest, "phys_sub_bass_db", 0.0) as f32;
    let spectral_q = star_meta_f64(manifest, "phys_spectral_q", 1.0) as f32;
    let mid_gain = star_meta_f64(manifest, "phys_mid_gain", 1.0) as f32;
    let gate_ratio = star_meta_f64(manifest, "phys_gate_ratio", 1.0) as f32;
    let comp_ratio = star_meta_f64(manifest, "phys_comp_ratio", 1.0) as f32;
    let env_bright = star_meta_f64(manifest, "phys_env_bright_shift", 1.0) as f32;
    let lfo_hz = star_meta_f64(manifest, "phys_lfo_hz", 0.0) as f32;
    let lfo_depth = star_meta_f64(manifest, "phys_lfo_depth", 0.0) as f32;
    let stereo_diff = star_meta_f64(manifest, "phys_stereo_diff", 0.0) as f32;
    let short_dly_ms = star_meta_f64(manifest, "phys_short_delay_ms", 0.0) as f32;
    let tail_ratio = star_meta_f64(manifest, "phys_tail_energy_ratio", 0.0) as f32;

    // 1. Stage 0 Sidechain isolation check for 2-in, 1-out rigs
    let mut aux_sc = 0.0_f32;
    if manifest.base.io_config.inputs >= 2 && manifest.base.io_config.output_mode == "mono" {
        let mut dot_rw = 0.0_f64;
        let mut e_r = 0.0_f64;
        let mut e_d = 0.0_f64;
        let mut gain_lo_r = Vec::new();
        let mut gain_hi_r = Vec::new();
        for blk in (0..n_fit.saturating_sub(256)).step_by(256) {
            let (mut sum_dl, mut sum_dr, mut sum_wl) = (0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..256 {
                let (dl, dr, wl) = (
                    f64::from(dry_lr[blk + k][0]),
                    f64::from(dry_lr[blk + k][1]),
                    f64::from(wet_lr[blk + k][0]),
                );
                dot_rw += (dr - dl) * wl;
                e_r += (dr - dl) * (dr - dl);
                e_d += dl * dl;
                sum_dl += dl * dl;
                sum_dr += dr * dr;
                sum_wl += wl * wl;
            }
            if sum_dl > 1.0e-5 {
                let g = (sum_wl / sum_dl).sqrt();
                if sum_dr > 2.2 * sum_dl {
                    gain_hi_r.push(g);
                } else if sum_dr < 0.8 * sum_dl {
                    gain_lo_r.push(g);
                }
            }
        }
        let corr_diff = dot_rw.abs() / ((e_r * e_d).sqrt() + 1.0e-12);
        let m_lo = gain_lo_r.iter().sum::<f64>() / (gain_lo_r.len().max(1) as f64);
        let m_hi = gain_hi_r.iter().sum::<f64>() / (gain_hi_r.len().max(1) as f64);
        if e_r > 0.05 * e_d
            && (corr_diff < 0.25
                || (!gain_hi_r.is_empty() && !gain_lo_r.is_empty() && m_hi < 0.82 * m_lo))
        {
            aux_sc = 1.0;
            // Route in_1 -> main_r and in_1 -> down_r so aux_key (in_2) drives only sidechain detector
            write_f32_le(arena_64k, 16, 1.0);
            write_f32_le(arena_64k, 20, 0.0);
            write_f32_le(arena_64k, 48, 1.0);
            write_f32_le(arena_64k, 52, 0.0);
        }
    }

    // 2. ARENA0_STAGE1_EXT (360..488)
    arena_64k[ARENA0_STAGE1_EXT..ARENA0_STAGE1_EXT + 128].fill(0);
    let mut refine_s12: Vec<(usize, f32, f32)> = Vec::new();
    let amp_cab_nl = (odd_harm + even_harm > 0.18) && (pre_lpf_hz < 10_000.0 || comp_ratio < 0.72);
    let clean_comp = comp_ratio < 0.88
        && gate_ratio >= 0.65
        && odd_harm < 0.12
        && even_harm < 0.08
        && (odd_harm + even_harm) < 0.14
        && nl_drive < 0.18;
    if (clean_comp && !amp_cab_nl) || aux_sc > 0.5 {
        let slope = ((1.0 - comp_ratio) * 1.4).clamp(0.25, 0.90);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 4, 0.14); // comp_thresh
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 8, slope); // comp_slope
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 12, 0.008); // comp_atk_48k
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 16, 0.0015); // comp_rel_fast_48k
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 20, 0.00015); // comp_rel_slow_48k
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 24, mid_gain.clamp(0.5, 4.0));
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 28, 1.0); // comp_mix
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 32, aux_sc);
        refine_s12.extend([
            (ARENA0_STAGE1_EXT + 4, 0.02, 0.45),
            (ARENA0_STAGE1_EXT + 8, 0.10, 0.95),
            (ARENA0_STAGE1_EXT + 24, 0.5, 5.0),
        ]);
    }
    if gate_ratio < 0.65 && env_bright < 1.28 && subharm < 0.10 {
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 36, 0.036); // gate_ext_thresh
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 40, 0.12); // gate_atk_48k
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 44, 0.002); // gate_rel_48k
        refine_s12.push((ARENA0_STAGE1_EXT + 36, 0.006, 0.10));
    }
    if subharm > 0.12 && gate_ratio >= 0.65 {
        write_f32_le(
            arena_64k,
            ARENA0_STAGE1_EXT + 52,
            (subharm * 2.4).clamp(0.15, 1.0),
        ); // sub1
        write_f32_le(
            arena_64k,
            ARENA0_STAGE1_EXT + 56,
            (subharm * 1.4).clamp(0.05, 0.8),
        ); // sub2
        if even_harm > 0.10 {
            write_f32_le(
                arena_64k,
                ARENA0_STAGE1_EXT + 48,
                (even_harm * 1.5).clamp(0.05, 0.9),
            );
        }
        refine_s12.push((ARENA0_STAGE1_EXT + 52, 0.0, 1.2));
        refine_s12.push((ARENA0_STAGE1_EXT + 56, 0.0, 1.0));
    }
    if ringmod_hz > 50.0 && sideband > 0.14 {
        write_f32_le(
            arena_64k,
            ARENA0_STAGE1_EXT + 60,
            (sideband * 1.2).clamp(0.10, 1.0),
        );
        write_f32_le(
            arena_64k,
            ARENA0_STAGE1_EXT + 64,
            ringmod_hz.clamp(25.0, 2_000.0),
        );
        refine_s12.push((ARENA0_STAGE1_EXT + 60, 0.0, 1.0));
    }
    if lfo_hz > 0.15 && (stereo_diff > 0.05 || short_dly_ms > 1.5) && tail_ratio < 0.12 {
        let dly_48k = if short_dly_ms > 1.5 {
            (short_dly_ms * 48.0).clamp(48.0, 1_400.0)
        } else {
            408.0
        };
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 68, 0.55); // chorus_wet_mix
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 72, dly_48k);
        write_f32_le(
            arena_64k,
            ARENA0_STAGE1_EXT + 76,
            (lfo_depth * 64.0).clamp(8.0, 140.0),
        );
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 80, lfo_hz.clamp(0.15, 12.0));
        write_f32_le(
            arena_64k,
            ARENA0_STAGE1_EXT + 84,
            (lfo_hz * 1.13).clamp(0.15, 14.0),
        );
        if stereo_diff > 0.18 {
            write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 88, lfo_hz.clamp(0.2, 10.0));
            write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 92, 16.0);
            write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 96, 0.22);
            refine_s12.push((ARENA0_STAGE1_EXT + 96, 0.0, 0.65));
        }
        refine_s12.push((ARENA0_STAGE1_EXT + 68, 0.0, 1.0));
    }
    let eval_stereo = manifest.base.io_config.output_mode != "mono";
    let svf_active = (env_bright > 1.32
        || (spectral_q > 2.2 && odd_harm < 0.14 && pre_lpf_hz >= 8_000.0))
        && gate_ratio >= 0.65
        && subharm < 0.10
        && sideband < 0.12
        && tail_ratio < 0.08
        && !amp_cab_nl;
    if svf_active {
        let (fc0, sens0, k0, [lp0, bp0, hp0]) = fit_svf_tpt_surrogate(
            dry_lr,
            wet_lr,
            eval_stereo,
            odd_harm >= 0.06 || nl_drive >= 0.05,
            n_fit.min(10_650),
        )
        .unwrap_or((
            340.0,
            ((env_bright - 1.0) * 1.8).clamp(0.2, 4.5),
            (1.0 / spectral_q.clamp(0.7, 10.0)).clamp(0.08, 0.95),
            [0.0, 1.0, 0.0],
        ));
        // Reset core Stage I op_m_pre to identity (b_re[0]=1) so pre-SVF resonant poles do not color input.
        arena_64k[256..336].fill(0);
        write_f32_le(arena_64k, 304, 1.0);
        write_f32_le(arena_64k, 352, 1.0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 100, 1.0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 104, fc0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 108, sens0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 112, k0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 116, lp0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 120, bp0);
        write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 124, hp0);
        refine_s12.extend([
            (ARENA0_STAGE1_EXT + 104, 120.0, 2_200.0),
            (ARENA0_STAGE1_EXT + 108, 0.25, 15.0),
            (ARENA0_STAGE1_EXT + 112, 0.06, 1.20),
        ]);
    }

    // 3. ARENA0_STAGE2_EXT (512..552) + higher-tier Stage II cascade branches (4_224..4_272)
    let tau = std::f32::consts::TAU;
    let is_pure_lti = nl_drive < 0.015
        && odd_harm < 0.03
        && even_harm < 0.03
        && subharm < 0.03
        && sideband < 0.03
        && spectral_q <= 2.0
        && env_bright <= 1.32
        && comp_ratio >= 0.88
        && gate_ratio >= 0.65
        && lfo_hz <= 0.15;
    let nl_amt = ((1.0 - coherence).max(odd_harm + even_harm)).clamp(0.0, 1.0);
    let pre_hpf_p = if pre_hpf_hz > 25.0 {
        (-tau * pre_hpf_hz / 48_000.0).exp().clamp(0.5, 0.999)
    } else {
        0.0
    };
    let pre_lpf_p = if pre_lpf_hz < 15_000.0 {
        (-tau * pre_lpf_hz / 48_000.0).exp().clamp(0.0, 0.98)
    } else {
        0.0
    };
    let clean_mod_chain = (svf_active || comp_ratio < 0.88 || lfo_hz > 0.15)
        && odd_harm < 0.12
        && even_harm < 0.10
        && gate_ratio >= 0.65;
    let clean_bypass = if clean_mod_chain
        || (coherence > 0.94 && odd_harm < 0.05 && even_harm < 0.05)
    {
        0.92
    } else if svf_active {
        0.55
    } else if gate_ratio >= 0.65 && odd_harm < 0.25 && sub_bass_db > -6.0 && pre_lpf_hz >= 8_000.0 {
        0.25
    } else {
        0.0
    };
    let zero_amp_eq = clean_bypass >= 0.90 || svf_active;
    write_f32_le(arena_64k, ARENA0_STAGE2_EXT, pre_hpf_p);
    write_f32_le(arena_64k, ARENA0_STAGE2_EXT + 4, pre_lpf_p);
    write_f32_le(
        arena_64k,
        ARENA0_STAGE2_EXT + 8,
        if zero_amp_eq { 0.0 } else { 0.15 * nl_amt },
    ); // l1_hpf_mix
    write_f32_le(
        arena_64k,
        ARENA0_STAGE2_EXT + 12,
        if zero_amp_eq { 0.0 } else { 0.15 * nl_amt },
    ); // l2_hpf_mix
    write_f32_le(arena_64k, ARENA0_STAGE2_EXT + 16, 0.0); // v1b_wf_mix
    write_f32_le(
        arena_64k,
        ARENA0_STAGE2_EXT + 20,
        if zero_amp_eq { 0.0 } else { 0.25 * nl_amt },
    ); // pwr_hyst_mix
    write_f32_le(arena_64k, ARENA0_STAGE2_EXT + 24, clean_bypass);
    write_f32_le(
        arena_64k,
        ARENA0_STAGE2_EXT + 28,
        if !clean_mod_chain && sub_bass_db > -3.0 && nl_amt > 0.12 {
            0.35
        } else {
            0.0
        },
    );
    write_f32_le(
        arena_64k,
        ARENA0_STAGE2_EXT + 32,
        if is_pure_lti { 1.0 } else { 0.0 },
    ); // linear_v1a_v1b
    write_f32_le(arena_64k, ARENA0_STAGE2_EXT + 36, 1.0); // stage2_out_gain
    if is_pure_lti || nl_amt < 0.25 {
        let base_s = amp_scale.clamp(0.35, 1.5);
        for (t, &cw_base) in super::stage_a::CASCADE_W_BASES.iter().enumerate() {
            let tier_s = base_s * (1.0 - 0.08 * (t as f32));
            for k in 0..16 {
                let kl = if !eval_stereo {
                    if t == 0 { (k & 3) | (k & 8) } else { k & 7 }
                } else {
                    k
                };
                let sign = if (kl + t) % 2 == 0 {
                    1.0_f32
                } else {
                    -0.85_f32
                };
                let w = tier_s * sign * (0.65 + 0.05 * ((kl % 5) as f32));
                write_f32_le(arena_64k, cw_base + k * 4, w);
            }
        }
    }

    // Multi-tier cascade branches (Full V2A, Pro V2B, Ultra V3) for strictly monotone nonlinear richness
    let b_drive = (0.35 + 1.8 * nl_amt).clamp(0.15, 2.5);
    let branch_cfgs: [[f32; 4]; 3] = [
        [
            0.42,
            0.985,
            b_drive * 0.85,
            if even_harm > 0.06 { 0.14 } else { 0.0 },
        ],
        [
            0.65,
            0.960,
            b_drive * 1.15,
            if even_harm > 0.06 { -0.18 } else { 0.05 },
        ],
        [
            0.82,
            0.920,
            b_drive * 1.45,
            if even_harm > 0.06 { 0.22 } else { -0.06 },
        ],
    ];
    for (bi, vals) in branch_cfgs.iter().enumerate() {
        for (k, &v) in vals.iter().enumerate() {
            write_f32_le(arena_64k, 4_224 + bi * 16 + k * 4, v);
        }
    }

    let is_nl = odd_harm >= 0.035 || nl_drive >= 0.05;
    let min_drv = if is_nl { 0.28 } else { 0.05 };
    let cb_lo: f32 = if clean_mod_chain && !svf_active && odd_harm < 0.06 && even_harm < 0.05 {
        0.78
    } else {
        0.0
    };
    let max_cb: f32 = if gate_ratio < 0.65 {
        0.02
    } else if (odd_harm >= 0.10 || nl_drive >= 0.08) && clean_bypass < 0.05 && !svf_active {
        0.06
    } else {
        0.95
    };
    let hi_pg = if svf_active {
        2.80
    } else if comp_ratio < 0.68 && is_nl {
        3.20
    } else {
        2.20
    };
    if !is_pure_lti && (nl_amt > 0.08 || !refine_s12.is_empty()) {
        let (hpf_lo, hpf_hi) = if pre_hpf_hz > 80.0 {
            ((pre_hpf_p * 0.82).clamp(0.50, 0.90), 0.995)
        } else {
            (0.0, pre_hpf_p.max(0.35))
        };
        refine_s12.extend([
            (4_112, min_drv, 1.65),
            (4_136, min_drv, 1.50),
            (352, 0.50, hi_pg),
            (ARENA0_STAGE2_EXT + 24, cb_lo, max_cb.max(cb_lo)),
            (ARENA0_STAGE2_EXT, hpf_lo, hpf_hi),
        ]);
        if even_harm > 0.02 {
            refine_s12.push((4_120, -0.45, 0.45));
        }
        if is_nl && !svf_active && clean_bypass < 0.10 {
            refine_s12.push((ARENA0_STAGE2_EXT + 36, 0.08, 2.50));
        }
        let n_eval = if svf_active {
            n_fit.min(10_650)
        } else if is_nl {
            n_fit.min(9_000)
        } else {
            n_fit.min(4_096)
        };
        let mut best_loss = eval_stage12_nl_loss(manifest, arena_64k, dry_lr, wet_lr, n_eval);
        if best_loss.is_finite() {
            if is_nl && !svf_active {
                let (init_pg, init_v1a) =
                    (read_f32_le(arena_64k, 352), read_f32_le(arena_64k, 4_112));
                let mut best_probe = (pre_hpf_p, init_pg, init_v1a);
                let cands = if pre_hpf_hz > 80.0 {
                    [
                        (0.910, 1.85, 1.35),
                        (0.965, 1.35, 1.05),
                        (0.982, 1.65, 1.25),
                        (0.940, 1.95, 1.40),
                        (0.990, 1.15, 0.85),
                        (pre_hpf_p, 1.75, 1.25),
                    ]
                } else {
                    [
                        (pre_hpf_p, 1.35, 1.05),
                        (pre_hpf_p, 1.65, 1.25),
                        (pre_hpf_p, 1.85, 1.35),
                        (0.20, 1.55, 1.15),
                        (0.0, 1.25, 0.95),
                        (0.0, 1.75, 1.25),
                    ]
                };
                for &(hpf_c, pg_c, v1a_c) in &cands {
                    write_f32_le(arena_64k, ARENA0_STAGE2_EXT, hpf_c);
                    write_f32_le(arena_64k, 352, pg_c);
                    write_f32_le(arena_64k, 4_112, v1a_c);
                    write_f32_le(arena_64k, 4_116, v1a_c);
                    let l = eval_stage12_nl_loss(manifest, arena_64k, dry_lr, wet_lr, n_eval);
                    if l < best_loss * 0.996 {
                        (best_loss, best_probe) = (l, (hpf_c, pg_c, v1a_c));
                    }
                }
                write_f32_le(arena_64k, ARENA0_STAGE2_EXT, best_probe.0);
                write_f32_le(arena_64k, 352, best_probe.1);
                write_f32_le(arena_64k, 4_112, best_probe.2);
                write_f32_le(arena_64k, 4_116, best_probe.2);
            }
            coord_refine_arena_scalars(&refine_s12, &mut best_loss, arena_64k, &mut |buf| {
                eval_stage12_nl_loss(manifest, buf, dry_lr, wet_lr, n_eval)
            });
            // Keep L/R drives & even bias symmetric on mono targets (and when even harmonics exist)
            write_f32_le(arena_64k, 4_116, read_f32_le(arena_64k, 4_112));
            write_f32_le(arena_64k, 4_140, read_f32_le(arena_64k, 4_136));
            if !eval_stereo || even_harm > 0.02 {
                write_f32_le(arena_64k, 4_124, read_f32_le(arena_64k, 4_120));
                write_f32_le(arena_64k, 4_148, read_f32_le(arena_64k, 4_144));
            }
            if !eval_stereo {
                write_f32_le(arena_64k, 4_132, read_f32_le(arena_64k, 4_128));
                write_f32_le(arena_64k, 4_156, read_f32_le(arena_64k, 4_152));
            }
        }
        let refined_cb = read_f32_le(arena_64k, ARENA0_STAGE2_EXT + 24);
        if svf_active
            && (refined_cb >= 0.35 || odd_harm < 0.28)
            && let Some([lp_w, bp_w, hp_w]) =
                solve_svf_lp_bp_hp_weights(manifest, arena_64k, dry_lr, wet_lr, n_eval)
        {
            write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 116, lp_w);
            write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 120, bp_w);
            write_f32_le(arena_64k, ARENA0_STAGE1_EXT + 124, hp_w);
        }
    }
}

pub(crate) use super::stage_a_refl::lift_v2_post_s3_stage4;
