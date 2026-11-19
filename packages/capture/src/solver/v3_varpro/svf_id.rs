#![forbid(unsafe_code)]
//! Generic Multi-Timescale `96 kHz` Black-Box Physical Seed Bank & Gabor Homotopy Prescreen (`v3`).

use crate::solver::backends::TrainerBackend;

use super::adjoint::{AdjointWorkspace, evaluate_matryoshka_loss_f64_with_backend};
use super::basis::solve_matryoshka_varpro_cholesky_with_backend;
use super::lbfgs::run_lbfgs_with_ws;
use super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};
use super::omp::select_omp_kautz_poles_matryoshka_with_backend;
use super::primitives::par_map_vec;
pub use super::seed_init::*;

#[inline(never)]
fn varpro_if_improves<B: TrainerBackend>(
    m: &mut V3MatryoshkaOptF64,
    ds: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    als_iters: usize,
    cur_loss: &mut f64,
) {
    if !cur_loss.is_finite() {
        *cur_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    }
    let cur = *cur_loss;
    let backup = Box::new(m.clone());
    solve_matryoshka_varpro_cholesky_with_backend::<B>(m, ds, als_iters);
    let next = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    if next.is_finite() && (!cur.is_finite() || next <= cur) {
        *cur_loss = next;
    } else {
        m.clone_from(&backup);
    }
}

#[inline(never)]
fn omp_if_improves<B: TrainerBackend>(
    m: &mut V3MatryoshkaOptF64,
    ds: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    cur_loss: &mut f64,
) {
    if !cur_loss.is_finite() {
        *cur_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    }
    let cur = *cur_loss;
    let backup = Box::new(m.clone());
    select_omp_kautz_poles_matryoshka_with_backend::<B>(m, ds);
    let next = evaluate_matryoshka_loss_f64_with_backend::<B>(m, ds, ws);
    if next.is_finite() && (!cur.is_finite() || next <= cur) {
        *cur_loss = next;
    } else {
        m.clone_from(&backup);
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
    let tws0 = ws.tier_ws[0]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for ch in 0..active_ch {
        let tgt = if ch == 0 { &st0.tgt_l } else { &st0.tgt_r };
        let mut e_y = 0.0_f64;
        let mut e_t = 0.0_f64;
        for t in w..n {
            let yv = tws0.y_tier[ch][t];
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

/// Two-phase archetype-preserving Gabor homotopy prescreen in `Lite (G=1)` (`4_096 -> 5_120`).
#[inline(never)]
pub fn prescreen_matryoshka_seeds_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
    top_k: usize,
) -> Vec<V3MatryoshkaOptF64> {
    let ds_0a_base = dataset.primary_subwindow_opts(4_096, true, false);
    let ds_0a = ds_0a_base.with_gabor_weights(0.08, 0.04);
    let ds_svf_0 = with_scaled_band_weights(&ds_0a_base.with_gabor_weights(1.0, 0.30), 0.02);
    let ds_svf_hyb = with_scaled_band_weights(&ds_0a_base.with_gabor_weights(0.45, 0.16), 0.30);

    let ds_0b_base = dataset.curriculum_subwindow_opts(5_120, 2_560, true, false);
    let ds_0b = ds_0b_base.with_gabor_weights(0.08, 0.05);
    let ds_0b_hyb = with_scaled_band_weights(&ds_0b_base.with_gabor_weights(0.40, 0.15), 0.35);
    let ds_gab_only = with_scaled_band_weights(&ds_0b_base.with_gabor_weights(1.0, 0.25), 0.0);

    let tagged: Vec<(SeedArchetype, Box<V3MatryoshkaOptF64>)> =
        build_matryoshka_seed_bank_tagged(mono_out)
            .into_iter()
            .map(|(a, m)| (a, Box::new(m)))
            .collect();
    let eval_phase0a = |arch: SeedArchetype, mut m: Box<V3MatryoshkaOptF64>| {
        let mut ws = AdjointWorkspace::default();
        let l = if arch == SeedArchetype::EnvResonantSvf {
            let mut m_direct = m.clone();
            let mut l_direct = f64::INFINITY;
            varpro_if_improves::<B>(&mut m_direct, &ds_svf_hyb, &mut ws, 1, &mut l_direct);
            rms_scale_lite_readout::<B>(&mut m, &ds_svf_0, &mut ws);
            let _ = run_lbfgs_with_ws::<B>(&mut m, &ds_svf_0, &mut ws, 4, true, false, true);
            let mut l_core = f64::INFINITY;
            varpro_if_improves::<B>(&mut m, &ds_svf_hyb, &mut ws, 1, &mut l_core);
            if l_direct.is_finite() && (!l_core.is_finite() || l_direct < l_core) {
                m = m_direct;
            }
            let mut l_svf =
                run_lbfgs_with_ws::<B>(&mut m, &ds_svf_hyb, &mut ws, 4, true, true, true);
            varpro_if_improves::<B>(&mut m, &ds_svf_hyb, &mut ws, 1, &mut l_svf);
            l_svf
        } else {
            let mut l_cur = f64::INFINITY;
            varpro_if_improves::<B>(&mut m, &ds_0a, &mut ws, 1, &mut l_cur);
            l_cur = run_lbfgs_with_ws::<B>(&mut m, &ds_0a, &mut ws, 3, true, true, true);
            varpro_if_improves::<B>(&mut m, &ds_0a, &mut ws, 1, &mut l_cur);
            l_cur
        };
        let score = if l.is_finite() { l } else { f64::INFINITY };
        (arch, score, m)
    };
    let phase0a_res = par_map_vec(tagged, |_, (arch, m)| eval_phase0a(arch, m));
    let mut arch_best: [Option<(f64, Box<V3MatryoshkaOptF64>)>; 4] = [None, None, None, None];
    for (arch, score, m) in phase0a_res {
        let slot = &mut arch_best[arch.index()];
        if slot.as_ref().is_none_or(|(best_l, _)| score < *best_l) {
            *slot = Some((score, m));
        }
    }

    let spec_raw: Vec<(SeedArchetype, Box<V3MatryoshkaOptF64>)> =
        build_specialty_seeds_for_dataset(dataset, mono_out)
            .into_iter()
            .map(|(a, m)| (a, Box::new(m)))
            .collect();
    let mut spec_res = par_map_vec(spec_raw, |_, (arch, m)| eval_phase0a(arch, m));
    spec_res.sort_by(|a, b| a.1.total_cmp(&b.1));

    let svf_idx = SeedArchetype::EnvResonantSvf.index();
    let mut arch_entries: Vec<(usize, Box<V3MatryoshkaOptF64>)> = arch_best
        .into_iter()
        .enumerate()
        .filter_map(|(arch_idx, slot)| slot.map(|(_, m)| (arch_idx, m)))
        .collect();
    let mut non_key_taken = 0_usize;
    let mut key_taken = 0_usize;
    for (_, _, m) in spec_res {
        if m.feat_key_link > 0.5 {
            if key_taken < 2 {
                arch_entries.push((usize::MAX, m));
                key_taken += 1;
            }
        } else if non_key_taken < 6 {
            arch_entries.push((usize::MAX, m));
            non_key_taken += 1;
        }
    }
    let mut ranked: Vec<(f64, f64, bool, Box<V3MatryoshkaOptF64>)> =
        par_map_vec(arch_entries, |_, (arch_idx, mut m)| {
            let mut ws = AdjointWorkspace::default();
            let is_svf = arch_idx == svf_idx;
            let (m_hyb_opt, mut l, l_0b_before, l_gab_hyb): (
                Option<Box<V3MatryoshkaOptF64>>,
                f64,
                f64,
                f64,
            ) = if is_svf {
                let backup_hyb = m.clone();
                let mut hyb_loss =
                    run_lbfgs_with_ws::<B>(&mut m, &ds_0b_hyb, &mut ws, 4, true, true, true);
                if hyb_loss.is_finite() {
                    omp_if_improves::<B>(&mut m, &ds_0b_hyb, &mut ws, &mut hyb_loss);
                    varpro_if_improves::<B>(&mut m, &ds_0b_hyb, &mut ws, 1, &mut hyb_loss);
                } else {
                    m = backup_hyb;
                }
                let l0_pre = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_0b, &mut ws);
                let lg = evaluate_matryoshka_loss_f64_with_backend::<B>(&m, &ds_gab_only, &mut ws);
                (Some(m.clone()), l0_pre, l0_pre, lg)
            } else {
                (None, f64::INFINITY, f64::INFINITY, f64::INFINITY)
            };
            varpro_if_improves::<B>(&mut m, &ds_0b, &mut ws, 1, &mut l);
            if !is_svf {
                omp_if_improves::<B>(&mut m, &ds_0b, &mut ws, &mut l);
                varpro_if_improves::<B>(&mut m, &ds_0b, &mut ws, 1, &mut l);
            }
            l = run_lbfgs_with_ws::<B>(&mut m, &ds_0b, &mut ws, 4, true, true, true);
            varpro_if_improves::<B>(&mut m, &ds_0b, &mut ws, 1, &mut l);
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
    let is_split_core =
        |m: &V3MatryoshkaOptF64| m.svf_feat_lp2_mix > 0.5 && m.svf_env_oct_sens.abs() < 0.25;
    if top_k >= 2 {
        if !ranked.iter().take(2).any(|x| x.3.feat_key_link > 0.5)
            && let Some(pos) = ranked.iter().position(|x| x.3.feat_key_link > 0.5)
        {
            ranked.swap(1, pos);
        } else if svf_strict_gab_winner
            && !ranked.iter().take(2).any(|x| x.2)
            && let Some(pos) = ranked.iter().position(|x| x.2)
        {
            ranked.swap(1, pos);
        } else if !ranked.iter().take(2).any(|x| is_split_core(&x.3))
            && let Some(pos) = ranked.iter().position(|x| is_split_core(&x.3))
            && ranked[0].0.is_finite()
            && ranked[pos].0 <= ranked[0].0 * 1.25
        {
            ranked.swap(1, pos);
        }
    }
    ranked
        .into_iter()
        .take(top_k.max(1))
        .map(|(_, _, _, m)| *m)
        .collect()
}

/// Restores zero-input GRU contraction and overdrive silence recovery prior to cold polish.
pub(super) fn apply_pre_polish_overdrive_recovery(model: &mut V3MatryoshkaOptF64) {
    let mut cold = Box::new(model.to_matryoshka_cold());
    let mut dirty = false;
    for (b_idx, blk) in cold.blocks.iter_mut().enumerate() {
        dirty |= crate::audit::ensure_gru_block_zero_input_contraction(b_idx, blk);
    }
    if let Some(ref mut cal) = cold.rate_1x_cal {
        for (b_idx, blk) in cal.blocks.iter_mut().enumerate() {
            dirty |= crate::audit::ensure_gru_block_zero_input_contraction(b_idx, blk);
        }
    }
    dirty |= crate::audit::ensure_model_overdrive_silence_recovery(&mut cold);
    if dirty {
        *model = V3MatryoshkaOptF64::from_matryoshka_cold(&cold);
    }
}

#[cfg(test)]
mod tests {
    use dnam_engine::stages::stage2_gru::MAX_GROUPS;

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
