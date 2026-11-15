#![forbid(unsafe_code)]
//! Generic Multi-Timescale `96 kHz` Black-Box Physical Seed Bank & Gabor Homotopy Prescreen (`v3`).

use crate::solver::backends::TrainerBackend;

use super::adjoint::{AdjointWorkspace, evaluate_matryoshka_loss_f64_with_backend};
use super::basis::solve_matryoshka_varpro_cholesky_with_backend;
use super::lbfgs::{run_lbfgs_core_matryoshka_with_backend, run_lbfgs_matryoshka_with_backend};
use super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};
use super::omp::select_omp_kautz_poles_matryoshka_with_backend;
use super::primitives::par_map_vec;
pub use super::seed_init::*;

fn varpro_if_improves<B: TrainerBackend>(
    m: &mut V3MatryoshkaOptF64,
    ds: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    als_iters: usize,
) {
    let cur = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    let backup = m.clone();
    solve_matryoshka_varpro_cholesky_with_backend::<B>(m, ds, als_iters);
    let next = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    if !(next.is_finite() && (!cur.is_finite() || next <= cur)) {
        *m = backup;
    }
}

fn omp_if_improves<B: TrainerBackend>(
    m: &mut V3MatryoshkaOptF64,
    ds: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) {
    let cur = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    let backup = m.clone();
    select_omp_kautz_poles_matryoshka_with_backend::<B>(m, ds);
    let next = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    if !(next.is_finite() && (!cur.is_finite() || next <= cur)) {
        *m = backup;
    }
}

fn rms_scale_lite_readout<B: TrainerBackend>(
    m: &mut V3MatryoshkaOptF64,
    ds_stem0: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) {
    let Some(st0) = ds_stem0.stems.first() else {
        return;
    };
    let n = st0.usable_len();
    let w = st0.warmup.min(n / 2);
    if w >= n {
        return;
    }
    let _ = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds_stem0, ws);
    let active_ch = if m.mono_out { 1 } else { 2 };
    for ch in 0..active_ch {
        let tgt = if ch == 0 { &st0.tgt_l } else { &st0.tgt_r };
        let mut e_y = 0.0_f64;
        let mut e_t = 0.0_f64;
        for t in w..n {
            let yv = ws.y_tier[ch][t];
            let tv = f64::from(tgt[t]);
            e_y += yv * yv;
            e_t += tv * tv;
        }
        if e_y > 1.0e-18 && e_t > 1.0e-18 {
            let scale = (e_t / e_y).sqrt().clamp(0.15, 6.0);
            m.readouts[0].pre_cab_u_gain[ch] *= scale;
            m.readouts[0].pre_cab_hp_gain[ch] *= scale;
            for v in &mut m.readouts[0].pre_cab_w[ch] {
                *v *= scale;
            }
        }
    }
    m.mirror_mono();
}

fn with_scaled_band_weights(ds: &V3TrainDataset, factor: f64) -> V3TrainDataset {
    let mut out = ds.clone();
    for bw in &mut out.stem_band_weights {
        for ch_bw in bw {
            for v in ch_bw {
                *v *= factor;
            }
        }
    }
    out
}

/// Two-phase archetype-preserving Gabor homotopy prescreen in `Lite (G=1)` (`4_096 -> 8_192`).
pub fn prescreen_matryoshka_seeds_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
    top_k: usize,
) -> Vec<V3MatryoshkaOptF64> {
    let ds_0a_base = dataset.primary_subwindow(4_096);
    let ds_0a = ds_0a_base.with_gabor_weights(0.08, 0.04);
    let ds_svf_0 = with_scaled_band_weights(&ds_0a_base.with_gabor_weights(1.0, 0.30), 0.02);
    let ds_svf_hyb = with_scaled_band_weights(&ds_0a_base.with_gabor_weights(0.45, 0.16), 0.30);

    let ds_0b_base = dataset.curriculum_subwindow(8_192, 4_096);
    let ds_0b = ds_0b_base.with_gabor_weights(0.08, 0.05);
    let ds_0b_hyb = with_scaled_band_weights(&ds_0b_base.with_gabor_weights(0.40, 0.15), 0.35);
    let ds_gab_only = with_scaled_band_weights(&ds_0b_base.with_gabor_weights(1.0, 0.25), 0.0);

    let tagged = build_matryoshka_seed_bank_tagged(mono_out);
    let phase0a_res = par_map_vec(tagged, |_, (arch, mut m)| {
        let mut ws = AdjointWorkspace::default();
        let l = if arch == SeedArchetype::EnvResonantSvf {
            let mut m_direct = m.clone();
            varpro_if_improves::<B>(&mut m_direct, &ds_svf_hyb, &mut ws, 1);
            let l_direct =
                evaluate_matryoshka_loss_f64_with_backend::<B>(&m_direct, &ds_svf_hyb, &mut ws);
            rms_scale_lite_readout::<B>(&mut m, &ds_svf_0, &mut ws);
            let _ = run_lbfgs_core_matryoshka_with_backend::<B>(&mut m, &ds_svf_0, 6);
            varpro_if_improves::<B>(&mut m, &ds_svf_hyb, &mut ws, 1);
            let l_core = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_svf_hyb, &mut ws);
            if l_direct.is_finite() && (!l_core.is_finite() || l_direct < l_core) {
                m = m_direct;
            }
            let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut m, &ds_svf_hyb, 6);
            varpro_if_improves::<B>(&mut m, &ds_svf_hyb, &mut ws, 1);
            evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_svf_hyb, &mut ws)
        } else {
            varpro_if_improves::<B>(&mut m, &ds_0a, &mut ws, 1);
            let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut m, &ds_0a, 5);
            varpro_if_improves::<B>(&mut m, &ds_0a, &mut ws, 1);
            evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_0a, &mut ws)
        };
        let score = if l.is_finite() { l } else { f64::INFINITY };
        (arch, score, m)
    });
    let mut arch_best: [Option<(f64, V3MatryoshkaOptF64)>; 4] = [None, None, None, None];
    for (arch, score, m) in phase0a_res {
        let slot = &mut arch_best[arch.index()];
        if slot.as_ref().is_none_or(|(best_l, _)| score < *best_l) {
            *slot = Some((score, m));
        }
    }

    let svf_idx = SeedArchetype::EnvResonantSvf.index();
    let arch_entries: Vec<(usize, V3MatryoshkaOptF64)> = arch_best
        .into_iter()
        .enumerate()
        .filter_map(|(arch_idx, slot)| slot.map(|(_, m)| (arch_idx, m)))
        .collect();
    let mut ranked: Vec<(f64, f64, bool, V3MatryoshkaOptF64)> =
        par_map_vec(arch_entries, |_, (arch_idx, mut m)| {
            let mut ws = AdjointWorkspace::default();
            let is_svf = arch_idx == svf_idx;
            let (m_hyb_opt, l_0b_before, l_gab_hyb) = if is_svf {
                let pre_hyb =
                    evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_0b_hyb, &mut ws);
                let backup_hyb = m.clone();
                let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut m, &ds_0b_hyb, 6);
                omp_if_improves::<B>(&mut m, &ds_0b_hyb, &mut ws);
                varpro_if_improves::<B>(&mut m, &ds_0b_hyb, &mut ws, 1);
                let post_hyb =
                    evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_0b_hyb, &mut ws);
                if !(post_hyb.is_finite() && (!pre_hyb.is_finite() || post_hyb <= pre_hyb)) {
                    m = backup_hyb;
                }
                let l0_pre = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_0b, &mut ws);
                let lg = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_gab_only, &mut ws);
                (Some(m.clone()), l0_pre, lg)
            } else {
                (None, f64::INFINITY, f64::INFINITY)
            };
            omp_if_improves::<B>(&mut m, &ds_0b, &mut ws);
            varpro_if_improves::<B>(&mut m, &ds_0b, &mut ws, 1);
            let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut m, &ds_0b, 8);
            varpro_if_improves::<B>(&mut m, &ds_0b, &mut ws, 1);
            let mut l = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_0b, &mut ws);
            if let Some(m_hyb) = m_hyb_opt
                && (!l.is_finite() || (l_0b_before.is_finite() && l > l_0b_before))
            {
                m = m_hyb;
                l = l_0b_before;
            }
            let l_gab = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_gab_only, &mut ws)
                .min(l_gab_hyb);
            (
                if l.is_finite() { l } else { f64::INFINITY },
                if l_gab.is_finite() {
                    l_gab
                } else {
                    f64::INFINITY
                },
                is_svf,
                m,
            )
        });
    let svf_strict_gab_winner = ranked
        .iter()
        .find(|x| x.2)
        .is_some_and(|svf| ranked.iter().all(|other| other.2 || svf.1 < other.1));
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    if top_k >= 2
        && svf_strict_gab_winner
        && !ranked.iter().take(2).any(|x| x.2)
        && let Some(pos) = ranked.iter().position(|x| x.2)
    {
        ranked.swap(1, pos);
    }
    ranked
        .into_iter()
        .take(top_k.max(1))
        .map(|(_, _, _, m)| m)
        .collect()
}

#[cfg(test)]
mod tests {
    use dnam_engine::stages::v3_core::cascade::MAX_GROUPS;

    use super::*;

    #[test]
    fn test_seed_bank_has_16_96k_seeds_4_archetypes_and_unlock_preserves_lite() {
        let tagged = build_matryoshka_seed_bank_tagged(true);
        assert_eq!(tagged.len(), 16);
        for arch in [
            SeedArchetype::GeneralDrive,
            SeedArchetype::SeriesTubeSag,
            SeedArchetype::PinchBiasClipper,
            SeedArchetype::EnvResonantSvf,
        ] {
            assert_eq!(tagged.iter().filter(|(a, _)| *a == arch).count(), 4);
        }
        let mut seeds = build_matryoshka_seed_bank(true);
        assert_eq!(seeds.len(), 16);
        assert!(seeds[10].feat_pinch_k > 200.0 && seeds[11].feat_pinch_k > 400.0);
        assert!(seeds[12].svf_env_oct_sens > 4.0 && seeds[13].svf_env_oct_sens > 5.0);
        assert!((seeds[12].svf_env_comp - 0.001).abs() < 1e-9);
        assert!((seeds[12].readouts[0].pre_cab_u_gain[0] - 0.45).abs() < 1e-9);
        assert!(seeds[0].blocks[0].b_z[0] > 2.0 && seeds[0].blocks[0].b_z[3] <= -4.5);
        unlock_matryoshka_groups_after_prescreen(&mut seeds[0]);
        assert_eq!(seeds[0].max_active_groups, MAX_GROUPS);
        assert!(seeds[0].blocks[0].b_z[7] <= -4.5);
        assert_eq!(seeds[0].readouts[1], seeds[0].readouts[0]);
        assert!(seeds[4].blocks[0].b_z[0] < -6.0 && seeds[4].blocks[0].b_z[1] < -6.0);
        unlock_matryoshka_groups_after_prescreen(&mut seeds[4]);
        assert_eq!(seeds[4].max_active_groups, MAX_GROUPS);
        assert!((seeds[4].blocks[0].b_z[7] + 4.772).abs() < 1e-9);
        assert_eq!(seeds[4].readouts[1], seeds[4].readouts[0]);
    }
}
