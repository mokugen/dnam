#![forbid(unsafe_code)]
//! Stage D Mode-0 (`StarCenter`) Feature Harvest & Cached-Gram Resolution (`solver::stages::mode0_harvest`).

use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::{ControlCalibrationRole, DnamError};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};

use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::schema::DnamProjManifest;

use super::super::stage_b_c::{HEAD_BASES, ULTRA_MODES_5_TO_8_BASE, compute_role_b_multipliers};
use super::super::wide_gram::{Wide192Accum, contract_256d_gram_tiled_wide192};
use super::super::{uplo_u_index, uplo_u_packed_len};
use super::{read_f32_le, sync_clamp_descriptors_to_arena};

pub(crate) struct StageDMode0Resolved {
    pub features: Vec<[f32; 256]>,
    pub pred0_lr: Vec<[f32; 2]>,
    pub compact_strided: bool,
    pub n_total_frames: usize,
    pub gram_256: Vec<f64>,
    pub sample_stride: usize,
}

pub(crate) type PrecomputedBackboneRefs<'a> = (&'a [[f32; 256]], &'a [[f32; 2]]);

/// Optional cached packed `256x256` Gram matrix, EMR weights, and target frame stride.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Mode0GramHint<'a> {
    pub cached_packed_gram_256: Option<&'a [f64]>,
    pub star_emr_w: Option<&'a [f64]>,
    pub gram_frame_stride: usize,
}

fn can_use_mode0_backbone_fastpath(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    features: &[[f32; 256]],
    backbone_out_lr: &[[f32; 2]],
) -> bool {
    if features.len() != backbone_out_lr.len() || arena_64k.len() < 60_160 {
        return false;
    }
    if manifest.base.io_config.output_mode == "wdw" || (arena_64k[112] & 1) != 0 {
        return false;
    }
    if read_f32_le(arena_64k, 9_024) != 0.0 || read_f32_le(arena_64k, 9_028) != 0.0 {
        return false;
    }
    if manifest.base.controls_schema.iter().any(|c| {
        c.calibration.as_ref().is_some_and(|cal| {
            matches!(
                cal.role,
                ControlCalibrationRole::Mix
                    | ControlCalibrationRole::StereoMode
                    | ControlCalibrationRole::StereoWidth
                    | ControlCalibrationRole::FreezeLatch
            )
        })
    }) {
        return false;
    }
    let dry_g = read_f32_le(arena_64k, 116);
    let wet_g = read_f32_le(arena_64k, 120);
    if dry_g.abs() > 1.0e-7 || (wet_g - 1.0).abs() > 1.0e-6 {
        return false;
    }
    let ultra_trim = read_f32_le(arena_64k, 10_624 + 3 * 4);
    if (0.0625..=16.0).contains(&ultra_trim) && (ultra_trim - 1.0).abs() > 1.0e-6 {
        return false;
    }
    !backbone_out_lr.iter().any(|&[bl, br]| {
        !bl.is_finite() || !br.is_finite() || bl.abs() >= 5.0e8 || br.abs() >= 5.0e8
    })
}

#[inline]
fn dot_mode0_head_simd_order(fb: &[f32; 256], w0_eff: &[[f32; 2]; 256]) -> (f32, f32) {
    // Match `eval_tier_feature_bus_head_dot` 4xVec4 SIMD reduction tree order
    let mut acc_l = [[0.0_f32; 4]; 4];
    let mut acc_r = [[0.0_f32; 4]; 4];
    for g4 in 0..16 {
        let base = g4 * 16;
        for b in 0..4 {
            let d0 = base + b * 4;
            for lane in 0..4 {
                let fd = fb[d0 + lane];
                acc_l[b][lane] += fd * w0_eff[d0 + lane][0];
                acc_r[b][lane] += fd * w0_eff[d0 + lane][1];
            }
        }
    }
    let sl: [f32; 4] =
        std::array::from_fn(|c| (acc_l[0][c] + acc_l[1][c]) + (acc_l[2][c] + acc_l[3][c]));
    let sr: [f32; 4] =
        std::array::from_fn(|c| (acc_r[0][c] + acc_r[1][c]) + (acc_r[2][c] + acc_r[3][c]));
    (
        (sl[0] + sl[1]) + (sl[2] + sl[3]),
        (sr[0] + sr[1]) + (sr[2] + sr[3]),
    )
}

/// Evaluates `Mode 0` (`StarCenter`) predictions directly from `StageBHarvest`'s `features` and
/// `backbone_out_lr` when standard unit-wet / non-WDW routing applies (`0` redundant engine passes).
pub(crate) fn compute_mode0_pred_from_backbone_harvest(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    features: &[[f32; 256]],
    backbone_out_lr: &[[f32; 2]],
) -> Option<Vec<[f32; 2]>> {
    if !can_use_mode0_backbone_fastpath(manifest, arena_64k, features, backbone_out_lr) {
        return None;
    }

    let role_b = compute_role_b_multipliers(arena_64k);
    let mut w0_eff = [[0.0_f32; 2]; 256];
    for (t, &h_base) in HEAD_BASES.iter().enumerate() {
        for i in 0..64 {
            let idx = t * 64 + i;
            let wl = read_f32_le(arena_64k, h_base + i * 4);
            let wr = read_f32_le(arena_64k, h_base + 256 + i * 4);
            w0_eff[idx][0] = wl * (role_b[idx][0] as f32);
            w0_eff[idx][1] = wr * (role_b[idx][1] as f32);
        }
    }
    let mono_only = manifest.base.io_config.output_mode == "mono";
    let mut out = Vec::with_capacity(features.len());
    for (fb, &[bl, br]) in features.iter().zip(backbone_out_lr.iter()) {
        let (hl, hr) = dot_mode0_head_simd_order(fb, &w0_eff);
        let raw_l = bl + hl;
        let raw_r = br + hr;
        let yl = if raw_l.is_finite() {
            raw_l.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        let yr_s = if raw_r.is_finite() {
            raw_r.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        out.push([yl, if mono_only { yl } else { yr_s }]);
    }
    Some(out)
}

fn diag_matches_cached_gram(
    features: &[[f32; 256]],
    feat_step: usize,
    emr_w: &[f64],
    emr_step: usize,
    n_total: usize,
    probe: (usize, f64),
) -> bool {
    let (d_probe, target_diag) = probe;
    let mut sampled = 0usize;
    let mut fi = 0usize;
    let mut ei = 0usize;
    while fi < features.len() && ei < n_total {
        if emr_w[ei].is_finite() && emr_w[ei] > 0.0 && features[fi][d_probe].is_finite() {
            sampled += 1;
        }
        fi += feat_step;
        ei += emr_step;
    }
    if sampled == 0 {
        return false;
    }
    let scale = (n_total as f64) / (sampled as f64);
    let mut acc = Wide192Accum::zero();
    let mut fi2 = 0usize;
    let mut ei2 = 0usize;
    while fi2 < features.len() && ei2 < n_total {
        let w = emr_w[ei2] * scale;
        let f = f64::from(features[fi2][d_probe]);
        if w.is_finite() && w > 0.0 && f.is_finite() {
            acc.add_product(f, w * f);
        }
        fi2 += feat_step;
        ei2 += emr_step;
    }
    let est = acc.to_f64();
    (est - target_diag).abs() <= 1.0e-6 * (1.0 + target_diag.abs())
}

/// Validates and resolves `cached_packed_gram_256` against `features` (auto-detecting the exact
/// `gram_frame_stride` used to build `STAT` in `< 0.01 ms`), or re-contracts via `Wide192Accum`.
pub(crate) fn resolve_cached_gram_or_recontract(
    cached_packed_gram_256: Option<&[f64]>,
    features: &[[f32; 256]],
    star_emr_w: Option<&[f64]>,
    requested_stride: usize,
) -> (Vec<f64>, usize) {
    let n_frames = features.len();
    let req_s = requested_stride.max(1);
    if n_frames == 0 {
        return (vec![0.0_f64; uplo_u_packed_len(256)], req_s);
    }

    if let Some(g) = cached_packed_gram_256
        && g.len() == uplo_u_packed_len(256)
        && let Some(emr_w) = star_emr_w
        && emr_w.len() >= n_frames
    {
        if let Some(d_probe) = (0..256).find(|&d| g[uplo_u_index(d, d)] > 1.0e-12) {
            let probe = (d_probe, g[uplo_u_index(d_probe, d_probe)]);
            let mut candidates = [req_s, 4, 2, 1, 8];
            let mut seen = [0usize; 5];
            let mut seen_len = 0usize;
            for cand in &mut candidates {
                if !seen[..seen_len].contains(cand) {
                    seen[seen_len] = *cand;
                    seen_len += 1;
                }
            }
            for &s in &seen[..seen_len] {
                if diag_matches_cached_gram(features, s, emr_w, s, n_frames, probe) {
                    return (g.to_vec(), s);
                }
            }
        } else {
            return (g.to_vec(), req_s);
        }
    }

    let weights: Vec<f64> = match star_emr_w {
        Some(w) if w.len() >= n_frames => w[..n_frames].to_vec(),
        _ => vec![1.0_f64; n_frames],
    };
    let zero_targets = vec![[0.0_f64; 2]; n_frames];
    let (gram, _) = contract_256d_gram_tiled_wide192(features, &zero_targets, &weights, req_s);
    (gram, req_s)
}

#[inline]
pub(crate) fn decode_coreset_frame_lr_at(fp16: &[u8], channels: usize, idx: usize) -> [f32; 2] {
    if channels == 2 {
        let o = idx * 4;
        let l = crate::proj::fp16_bits_to_f32(u16::from_le_bytes([fp16[o], fp16[o + 1]]));
        let r = crate::proj::fp16_bits_to_f32(u16::from_le_bytes([fp16[o + 2], fp16[o + 3]]));
        [l, r]
    } else {
        let o = idx * 2;
        let v = crate::proj::fp16_bits_to_f32(u16::from_le_bytes([fp16[o], fp16[o + 1]]));
        [v, v]
    }
}

fn try_resolve_precomputed_mode0(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    precomputed: PrecomputedBackboneRefs<'_>,
    gram_hint: Mode0GramHint<'_>,
) -> Option<StageDMode0Resolved> {
    let (pre_f, pre_back) = precomputed;
    let req_s = gram_hint.gram_frame_stride.max(1);
    if req_s > 1
        && pre_f.len() == pre_back.len().div_ceil(req_s)
        && let Some(g) = gram_hint.cached_packed_gram_256
        && g.len() == uplo_u_packed_len(256)
    {
        let back_sub: Vec<[f32; 2]> = (0..pre_f.len()).map(|k| pre_back[k * req_s]).collect();
        if let Some(pred0) =
            compute_mode0_pred_from_backbone_harvest(manifest, arena_64k, pre_f, &back_sub)
        {
            return Some(StageDMode0Resolved {
                features: pre_f.to_vec(),
                pred0_lr: pred0,
                compact_strided: true,
                n_total_frames: pre_back.len(),
                gram_256: g.to_vec(),
                sample_stride: req_s,
            });
        }
    }
    let pred0 = compute_mode0_pred_from_backbone_harvest(manifest, arena_64k, pre_f, pre_back)?;
    let n_total = pre_f.len();
    let (gram_256, sample_stride) = resolve_cached_gram_or_recontract(
        gram_hint.cached_packed_gram_256,
        pre_f,
        gram_hint.star_emr_w,
        gram_hint.gram_frame_stride,
    );
    if sample_stride > 1 {
        let m = n_total.div_ceil(sample_stride);
        let mut f_sub = Vec::with_capacity(m);
        let mut p_sub = Vec::with_capacity(m);
        let mut idx = 0usize;
        while idx < n_total {
            f_sub.push(pre_f[idx]);
            p_sub.push(pred0[idx]);
            idx += sample_stride;
        }
        return Some(StageDMode0Resolved {
            features: f_sub,
            pred0_lr: p_sub,
            compact_strided: true,
            n_total_frames: n_total,
            gram_256,
            sample_stride,
        });
    }
    Some(StageDMode0Resolved {
        features: pre_f.to_vec(),
        pred0_lr: pred0,
        compact_strided: false,
        n_total_frames: n_total,
        gram_256,
        sample_stride,
    })
}

fn build_mode0_container_bytes(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let mut mode0_arena = arena_64k.to_vec();
    for &h_base in &HEAD_BASES {
        if mode0_arena.len() >= h_base + 5 * 512 {
            mode0_arena[h_base + 512..h_base + 5 * 512].fill(0);
        }
    }
    if mode0_arena.len() >= ULTRA_MODES_5_TO_8_BASE + 8_192 {
        mode0_arena[ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + 8_192].fill(0);
    }
    if mode0_arena.len() >= 10_560 {
        mode0_arena[7_524..8_512].fill(0);
    }
    if mode0_arena.len() >= 63_424 {
        mode0_arena[60_928..63_424].fill(0);
    }
    zero_arena_b_head_dc_bias(&mut mode0_arena);
    let _ = sync_clamp_descriptors_to_arena(&manifest.base, &mut mode0_arena)?;

    let mut temp_hdr = manifest.base.clone();
    build_dnam_container_with_cp1(&mut temp_hdr, &mode0_arena, &mode0_arena, false, &[])
}

fn init_mode0_engine<'a>(
    dnam_bytes: &'a [u8],
    star_ctrls: &[f32],
) -> Result<DnamEngine<'a, DefaultMathBackend>, DnamError> {
    let mut engine = DnamEngine::<DefaultMathBackend>::from_bytes(dnam_bytes, Tier::Ultra, 48_000)?;
    for (p, &val) in star_ctrls.iter().enumerate() {
        engine.set_control_immediate(p, val);
    }
    Ok(engine)
}

fn try_harvest_strided_with_cached_gram(
    engine: &mut DnamEngine<'_, DefaultMathBackend>,
    dry_lr: &[[f32; 2]],
    cached_g: &[f64],
    star_emr_w: Option<&[f64]>,
    req_s: usize,
) -> Option<StageDMode0Resolved> {
    let n_total = dry_lr.len();
    let m_sub = n_total.div_ceil(req_s);
    let mut f_sub = Vec::with_capacity(m_sub);
    let mut p_sub = Vec::with_capacity(m_sub);
    let mut fb = [0.0_f32; 256];
    {
        let _ftz = FtzDazGuard::new();
        for (n, &[il, ir]) in dry_lr.iter().enumerate() {
            if n % req_s == 0 {
                let y = engine.step_frame_with_feature_bus(il, ir, &mut fb);
                f_sub.push(fb);
                p_sub.push(y);
            } else {
                let _ = engine.step_frame_tora(il, ir, false);
            }
        }
    }

    let stride_matches = if let Some(emr_w) = star_emr_w
        && emr_w.len() >= n_total
        && let Some(d_probe) = (0..256).find(|&d| cached_g[uplo_u_index(d, d)] > 1.0e-12)
    {
        let probe = (d_probe, cached_g[uplo_u_index(d_probe, d_probe)]);
        diag_matches_cached_gram(&f_sub, 1, emr_w, req_s, n_total, probe)
    } else {
        true
    };

    stride_matches.then(|| StageDMode0Resolved {
        features: f_sub,
        pred0_lr: p_sub,
        compact_strided: true,
        n_total_frames: n_total,
        gram_256: cached_g.to_vec(),
        sample_stride: req_s,
    })
}

/// Harvests or reuses `Mode 0` predictions and `256-D` features for Stage D, avoiding `12.28 MB`
/// full-frame feature allocations on Path 1 when `cached_packed_gram_256` matches `gram_frame_stride`.
pub(crate) fn harvest_and_resolve_mode0_for_stage_d(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    dry_lr: &[[f32; 2]],
    star_ctrls: &[f32],
    precomputed_backbone_harvest: Option<PrecomputedBackboneRefs<'_>>,
    gram_hint: Mode0GramHint<'_>,
) -> Result<StageDMode0Resolved, DnamError> {
    if let Some(precomputed) = precomputed_backbone_harvest
        && let Some(resolved) =
            try_resolve_precomputed_mode0(manifest, arena_64k, precomputed, gram_hint)
    {
        return Ok(resolved);
    }

    let dnam_bytes = build_mode0_container_bytes(manifest, arena_64k)?;
    let mut engine = init_mode0_engine(&dnam_bytes, star_ctrls)?;
    let n_total = dry_lr.len();
    let req_s = gram_hint.gram_frame_stride.max(1);
    if req_s > 1
        && n_total > 0
        && let Some(g) = gram_hint.cached_packed_gram_256
        && g.len() == uplo_u_packed_len(256)
    {
        if let Some(strided) = try_harvest_strided_with_cached_gram(
            &mut engine,
            dry_lr,
            g,
            gram_hint.star_emr_w,
            req_s,
        ) {
            return Ok(strided);
        }

        // Fallback when STAT was built with a different stride than req_s (e.g. stride = 2 vs 4)
        engine = init_mode0_engine(&dnam_bytes, star_ctrls)?;
    }

    let mut in_l = Vec::with_capacity(n_total);
    let mut in_r = Vec::with_capacity(n_total);
    for &[il, ir] in dry_lr {
        in_l.push(il);
        in_r.push(ir);
    }
    let mut features = vec![[0.0_f32; 256]; n_total];
    let mut out_l = vec![0.0_f32; n_total];
    let mut out_r = vec![0.0_f32; n_total];
    engine.process_block_with_feature_bus(&in_l, &in_r, &mut out_l, &mut out_r, &mut features);
    let pred0_lr = out_l.into_iter().zip(out_r).map(|(l, r)| [l, r]).collect();
    let (gram_256, sample_stride) = resolve_cached_gram_or_recontract(
        gram_hint.cached_packed_gram_256,
        &features,
        gram_hint.star_emr_w,
        gram_hint.gram_frame_stride,
    );
    Ok(StageDMode0Resolved {
        features,
        pred0_lr,
        compact_strided: false,
        n_total_frames: n_total,
        gram_256,
        sample_stride,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diag_matches_cached_gram_branches() {
        let mut feats = vec![[0.0_f32; 256]; 8];
        for (idx, f) in feats.iter_mut().enumerate() {
            f[0] = (idx as f32) * 0.25 + 0.5;
        }
        let zero_w = [0.0_f64; 8];
        assert!(!diag_matches_cached_gram(
            &feats,
            2,
            &zero_w,
            2,
            8,
            (0, 1.0)
        ));

        let emr = [1.0_f64; 8];
        // Stride = 2 averages 4 samples [0.5, 1.0, 1.5, 2.0]^2 * 2 = 0.25 + 1.0 + 2.25 + 4.0 = 7.5 * 2 = 15.0
        assert!(diag_matches_cached_gram(&feats, 2, &emr, 2, 8, (0, 15.0)));
        assert!(!diag_matches_cached_gram(&feats, 2, &emr, 2, 8, (0, 42.0)));
    }

    #[test]
    fn test_harvest_and_resolve_mode0_non_precomputed_strided_paths() {
        let mut m = DnamProjManifest::default_reference();
        m.base.controls_schema.clear();
        let mut arena_64k = crate::proj::vault::initialize_backbone_arena_if_needed(None);
        // Set nonzero cascade taps at ARENA0 Stage II Cascade W (byte 1_024) so feature bus has non-zero energy
        for k in 0..8 {
            let off = 1_024 + k * 4;
            arena_64k[off..off + 4].copy_from_slice(&0.5_f32.to_le_bytes());
        }

        let dry_lr: Vec<[f32; 2]> = (0..32)
            .map(|i| {
                let x = ((i as f32) * 0.37).sin() * 0.4;
                [x, -x * 0.7]
            })
            .collect();
        let emr_w = vec![1.0_f64; dry_lr.len()];

        // Build full baseline features via stride = 1 first so we can contract an exact stride-2 cached Gram
        let base_full = harvest_and_resolve_mode0_for_stage_d(
            &m,
            &arena_64k,
            &dry_lr,
            &[],
            None,
            Mode0GramHint {
                cached_packed_gram_256: None,
                star_emr_w: Some(&emr_w),
                gram_frame_stride: 1,
            },
        )
        .expect("stride=1 non-precomputed resolve should succeed");
        assert!(!base_full.compact_strided);
        assert_eq!(base_full.features.len(), 32);

        let zero_targets = vec![[0.0_f64; 2]; dry_lr.len()];
        let (matching_g, _) =
            contract_256d_gram_tiled_wide192(&base_full.features, &zero_targets, &emr_w, 2);
        let probe_diag_present = (0..256).any(|d| matching_g[uplo_u_index(d, d)] > 1.0e-12);
        assert!(
            probe_diag_present,
            "feature bus must yield nonzero diagonal energy"
        );

        // Fast path: non-precomputed strided harvest with matching cached Gram diagonal
        let fast = harvest_and_resolve_mode0_for_stage_d(
            &m,
            &arena_64k,
            &dry_lr,
            &[],
            None,
            Mode0GramHint {
                cached_packed_gram_256: Some(&matching_g),
                star_emr_w: Some(&emr_w),
                gram_frame_stride: 2,
            },
        )
        .expect("matching cached-gram strided harvest should succeed");
        assert!(fast.compact_strided);
        assert_eq!(fast.sample_stride, 2);
        assert_eq!(fast.features.len(), 16);
        assert_eq!(fast.pred0_lr.len(), 16);

        // Mismatch fallback path: perturb active diagonal so strided check rejects cache and re-runs full block
        let d_probe = (0..256)
            .find(|&d| matching_g[uplo_u_index(d, d)] > 1.0e-12)
            .unwrap();
        let mut mismatch_g = matching_g;
        mismatch_g[uplo_u_index(d_probe, d_probe)] *= 3.5;

        let fallback = harvest_and_resolve_mode0_for_stage_d(
            &m,
            &arena_64k,
            &dry_lr,
            &[],
            None,
            Mode0GramHint {
                cached_packed_gram_256: Some(&mismatch_g),
                star_emr_w: Some(&emr_w),
                gram_frame_stride: 2,
            },
        )
        .expect("mismatched cached-gram fallback should recover cleanly");
        assert!(!fallback.compact_strided);
        assert_eq!(fallback.features.len(), 32);
        assert_eq!(fallback.pred0_lr.len(), 32);
    }
}
