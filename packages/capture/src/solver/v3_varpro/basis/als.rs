#![forbid(unsafe_code)]
//! `pre_cab` alternating least-squares updates and nested 4-tier Matryoshka VarPro Cholesky solves.

use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, NUM_GRU_BLOCKS, tier_block_lanes, tier_fir_segments, tier_groups,
    tier_kautz_banks,
};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};
use crate::solver::uplo_u_packed_len;

use super::super::gru_core::{AdjointWorkspace, forward_shared_core};
use super::super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset};
use super::super::primitives::{ClampedCoreParams, par_map_indexed};
use super::cache::{
    PreSplitStemChan, SubspaceGramData, accum_band_gram_and_rhs_slices, assemble_tier_band_grams,
    eval_bezout_loss_from_gram,
};
use super::{
    ALL_TIERS, StemChanTraj, apply_fixed_cabinet_f64, eval_s2_from_readout, map_prev_tier_anchor,
    normalize_pre_cab_chan, normalize_pre_cab_gauge, solve_uplo_with_bypass_and_anchor,
    unpack_theta_k_to_readout,
};

struct AlsGramProblem {
    gram: Vec<f64>,
    rhs: Vec<[f64; 2]>,
    c_tot: f64,
}

fn collect_s2_per_stem(
    r: &V3TierReadoutF64,
    ch: usize,
    c_t: usize,
    stems_pre: &[PreSplitStemChan],
    trajs: &[StemChanTraj],
) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new(); stems_pre.len()];
    for (si, st) in stems_pre.iter().enumerate() {
        if st.seg > 0 {
            let tr = &trajs[si];
            eval_s2_from_readout(
                r,
                ch,
                c_t,
                &tr.u[..st.n],
                &tr.hp[..st.n],
                &tr.h[..st.n],
                &mut out[si],
            );
        }
    }
    out
}

fn prepare_cabinet_fir_work(
    r: &V3TierReadoutF64,
    ch: usize,
    f_t: usize,
    k_t: usize,
) -> (Vec<f64>, bool) {
    let fir = if ch == 0 { &r.fir_l } else { &r.fir_r };
    let kc = if ch == 0 { &r.kautz_c_l } else { &r.kautz_c_r };
    let cab_sq: f64 = fir[..f_t].iter().map(|v| v * v).sum::<f64>()
        + kc[..k_t].iter().map(|c| c.0 * c.0 + c.1 * c.1).sum::<f64>();
    let mut fir_work = fir[..f_t].to_vec();
    let empty_cab = cab_sq < 1.0e-6;
    if empty_cab && !fir_work.is_empty() {
        fir_work[0] = 1.0;
    }
    (fir_work, empty_cab)
}

#[allow(clippy::too_many_arguments)]
fn build_cab_stem_columns<B: TrainerBackend>(
    q: usize,
    c_t: usize,
    n: usize,
    w: usize,
    tr: &StemChanTraj,
    fir_slice: &[f64],
    k_slice: &[(f64, f64, f64, f64, f64, f64)],
    kc_slice: &[(f64, f64)],
    dataset: &V3TrainDataset,
) -> [Vec<f64>; 3] {
    let mut tmp_z = vec![0.0_f64; n];
    if q == 0 {
        tmp_z.copy_from_slice(&tr.u[..n]);
    } else if q == 1 {
        tmp_z.copy_from_slice(&tr.hp[..n]);
    } else {
        let blk = (q - 2) / c_t;
        let l = (q - 2) % c_t;
        let col_idx = blk * MAX_BLOCK_LANES + l;
        for t in 0..n {
            tmp_z[t] = tr.h[t][col_idx];
        }
    }
    let mut cab_z = vec![0.0_f64; n];
    apply_fixed_cabinet_f64::<B>(&tmp_z, fir_slice, k_slice, kc_slice, &mut cab_z);
    let mut split_buf = [Vec::new(), Vec::new(), Vec::new()];
    dataset.bank.split_3band(&cab_z, &mut split_buf);
    std::array::from_fn(|b| split_buf[b][w..n].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_als_gram_and_rhs<B: TrainerBackend>(
    c_t: usize,
    q_pre: usize,
    p_tot: usize,
    dataset: &V3TrainDataset,
    trajs: &[StemChanTraj],
    stems_pre: &[PreSplitStemChan],
    fir_slice: &[f64],
    k_slice: &[(f64, f64, f64, f64, f64, f64)],
    kc_slice: &[(f64, f64)],
) -> AlsGramProblem {
    let pk_tot = uplo_u_packed_len(p_tot);
    let mut gram = vec![0.0_f64; pk_tot];
    let mut rhs = vec![[0.0_f64; 2]; p_tot];
    let mut c_tot = 0.0_f64;
    let mut g_b = vec![0.0_f64; pk_tot];
    let mut r_b = vec![[0.0_f64; 2]; p_tot];

    for (si, st) in stems_pre.iter().enumerate() {
        if st.seg == 0 {
            continue;
        }
        let n = st.n;
        let w = st.w;
        let tr = &trajs[si];
        let build_cab = |q: usize| {
            build_cab_stem_columns::<B>(q, c_t, n, w, tr, fir_slice, k_slice, kc_slice, dataset)
        };
        let cab_b: Vec<[Vec<f64>; 3]> = if n >= 8_192 {
            par_map_indexed(q_pre, build_cab)
        } else {
            (0..q_pre).map(build_cab).collect()
        };
        for b in 0..3 {
            let mut col_refs: Vec<&[f64]> = Vec::with_capacity(p_tot);
            for cab_q in cab_b.iter().take(q_pre) {
                col_refs.push(&cab_q[b]);
            }
            for i in 0..7 {
                col_refs.push(&st.tap_b[b][i]);
            }
            col_refs.push(&st.u_b[b]);
            col_refs.push(&st.hp_b[b]);
            for blk in 0..NUM_GRU_BLOCKS {
                let cache_base = blk * st.c_max;
                for l in 0..c_t {
                    col_refs.push(&st.h_b[b][cache_base + l]);
                }
            }
            accum_band_gram_and_rhs_slices::<B>(
                &col_refs,
                &st.tgt_b[b],
                st.bw[b],
                &mut g_b,
                &mut r_b,
                &mut gram,
                &mut rhs,
            );
            c_tot += st.bw[b] * st.e_sig[b];
        }
    }

    AlsGramProblem { gram, rhs, c_tot }
}

fn pack_als_baseline(
    r: &V3TierReadoutF64,
    ch: usize,
    c_t: usize,
    q_pre: usize,
    p_tot: usize,
    empty_cab: bool,
) -> Vec<f64> {
    let mut sol_before = vec![0.0_f64; p_tot];
    if !empty_cab {
        sol_before[0] = r.pre_cab_u_gain[ch];
        sol_before[1] = r.pre_cab_hp_gain[ch];
        for b in 0..NUM_GRU_BLOCKS {
            for l in 0..c_t {
                sol_before[2 + b * c_t + l] = r.pre_cab_w[ch][b * MAX_BLOCK_LANES + l];
            }
        }
    }
    let dir_taps = if ch == 0 {
        &r.svf_direct_taps_l
    } else {
        &r.svf_direct_taps_r
    };
    sol_before[q_pre..q_pre + 7].copy_from_slice(&dir_taps[..7]);
    sol_before[q_pre + 7] = r.direct_s2_u_gain[ch];
    sol_before[q_pre + 8] = r.direct_s2_hp_gain[ch];
    for b in 0..NUM_GRU_BLOCKS {
        for l in 0..c_t {
            sol_before[q_pre + 9 + b * c_t + l] = r.direct_s2_w[ch][b * MAX_BLOCK_LANES + l];
        }
    }
    sol_before
}

fn build_als_anchor(
    src_r: &V3TierReadoutF64,
    has_prev: bool,
    ch: usize,
    c_t: usize,
    c_keep: usize,
    q_pre: usize,
    p_tot: usize,
) -> Vec<(usize, f64)> {
    let mut anchor = Vec::with_capacity(p_tot);
    anchor.push((0, src_r.pre_cab_u_gain[ch]));
    anchor.push((1, src_r.pre_cab_hp_gain[ch]));
    for b in 0..NUM_GRU_BLOCKS {
        for l in 0..c_t {
            let v = if l < c_keep {
                src_r.pre_cab_w[ch][b * MAX_BLOCK_LANES + l]
            } else {
                0.0
            };
            anchor.push((2 + b * c_t + l, v));
        }
    }
    if has_prev {
        let taps = if ch == 0 {
            &src_r.svf_direct_taps_l
        } else {
            &src_r.svf_direct_taps_r
        };
        for (i, &tv) in taps.iter().take(7).enumerate() {
            anchor.push((q_pre + i, tv));
        }
        anchor.push((q_pre + 7, src_r.direct_s2_u_gain[ch]));
        anchor.push((q_pre + 8, src_r.direct_s2_hp_gain[ch]));
        for b in 0..NUM_GRU_BLOCKS {
            for l in 0..c_t {
                let v = if l < c_keep {
                    src_r.direct_s2_w[ch][b * MAX_BLOCK_LANES + l]
                } else {
                    0.0
                };
                anchor.push((q_pre + 9 + b * c_t + l, v));
            }
        }
    }
    anchor
}

#[allow(clippy::too_many_arguments)]
fn commit_als_solution(
    r: &V3TierReadoutF64,
    prev_r: Option<&V3TierReadoutF64>,
    tau: usize,
    ch: usize,
    c_t: usize,
    q_pre: usize,
    empty_cab: bool,
    sol: &[f64],
) -> V3TierReadoutF64 {
    let mut cand = r.clone();
    if empty_cab {
        let cf = if ch == 0 {
            &mut cand.fir_l
        } else {
            &mut cand.fir_r
        };
        cf[0] = 1.0;
    }
    cand.pre_cab_u_gain[ch] = sol[0];
    cand.pre_cab_hp_gain[ch] = sol[1];
    cand.pre_cab_w[ch].fill(0.0);
    for b in 0..NUM_GRU_BLOCKS {
        for l in 0..c_t {
            cand.pre_cab_w[ch][b * MAX_BLOCK_LANES + l] = sol[2 + b * c_t + l];
        }
    }
    if ch == 0 {
        cand.svf_direct_taps_l[..7].copy_from_slice(&sol[q_pre..q_pre + 7]);
    } else {
        cand.svf_direct_taps_r[..7].copy_from_slice(&sol[q_pre..q_pre + 7]);
    }
    cand.direct_s2_u_gain[ch] = sol[q_pre + 7];
    cand.direct_s2_hp_gain[ch] = sol[q_pre + 8];
    cand.direct_s2_w[ch].fill(0.0);
    for b in 0..NUM_GRU_BLOCKS {
        for l in 0..c_t {
            cand.direct_s2_w[ch][b * MAX_BLOCK_LANES + l] = sol[q_pre + 9 + b * c_t + l];
        }
    }
    normalize_pre_cab_chan(&mut cand, prev_r, tau, ch);
    cand
}

#[allow(clippy::too_many_arguments)]
pub(in crate::solver::v3_varpro) fn als_update_pre_cab_chan<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    tau: usize,
    ch: usize,
    dataset: &V3TrainDataset,
    trajs: &[StemChanTraj],
    stems_pre: &[PreSplitStemChan],
    kautz_pre6: &[(f64, f64, f64, f64, f64, f64)],
) -> bool {
    let tier = ALL_TIERS[tau];
    let c_t = tier_block_lanes(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = tier_kautz_banks(tier) * 4;
    let r = &model.readouts[tau];
    let kc = if ch == 0 { &r.kautz_c_l } else { &r.kautz_c_r };
    let (fir_work, empty_cab) = prepare_cabinet_fir_work(r, ch, f_t, k_t);

    let q_pre = 2 + 3 * c_t;
    let p_tot = q_pre + 9 + 3 * c_t;
    let prob = accumulate_als_gram_and_rhs::<B>(
        c_t,
        q_pre,
        p_tot,
        dataset,
        trajs,
        stems_pre,
        &fir_work[..f_t],
        &kautz_pre6[..k_t],
        &kc[..k_t],
    );

    let sol_before = pack_als_baseline(r, ch, c_t, q_pre, p_tot, empty_cab);
    let loss_before =
        eval_bezout_loss_from_gram(p_tot, &prob.gram, &prob.rhs, prob.c_tot, &sol_before);

    let prev_r = (tau > 0 && model.active_tier_weights[tau - 1] > 0.0)
        .then(|| model.readouts[tau - 1].clone());
    let (src_r, c_keep, anchor_scale) = match prev_r.as_ref() {
        Some(pr) => (pr, tier_block_lanes(ALL_TIERS[tau - 1]), 2.0e-6),
        None => (r, c_t, 1.0e-6),
    };
    let anchor = build_als_anchor(src_r, prev_r.is_some(), ch, c_t, c_keep, q_pre, p_tot);
    let bypass = bypass_ridge_cols(q_pre, c_t, model.svf_env_oct_sens.abs() > 0.5);
    let sol = solve_uplo_with_bypass_and_anchor(
        p_tot,
        &prob.gram,
        &prob.rhs,
        &bypass,
        &anchor,
        anchor_scale,
    );
    let loss_after = eval_bezout_loss_from_gram(p_tot, &prob.gram, &prob.rhs, prob.c_tot, &sol);
    if !(loss_after.is_finite() && loss_after < loss_before) {
        return false;
    }
    model.readouts[tau] =
        commit_als_solution(r, prev_r.as_ref(), tau, ch, c_t, q_pre, empty_cab, &sol);
    true
}

fn bypass_ridge_cols(offset: usize, c_t: usize, sweeping_svf: bool) -> Vec<(usize, f64)> {
    let mut cols = Vec::with_capacity(9 + 3 * c_t);
    let (lin_ridge, dir_s2_ridge, w_ridge) = if sweeping_svf {
        (5.0e-4, 2.5e-2, 2.0e-4)
    } else {
        (5.0e-4, 2.5e-4, 1.5e-4)
    };
    for i in 0..7 {
        cols.push((offset + i, lin_ridge));
    }
    for i in 7..9 {
        cols.push((offset + i, dir_s2_ridge));
    }
    for i in 0..3 * c_t {
        cols.push((offset + 9 + i, w_ridge));
    }
    cols
}

pub(in crate::solver::v3_varpro) fn solve_tier_output_k_chan<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    tau: usize,
    ch: usize,
    k_keep: usize,
    dataset: &V3TrainDataset,
    trajs: &[StemChanTraj],
    stems_pre: &[PreSplitStemChan],
) -> SubspaceGramData {
    let tier = ALL_TIERS[tau];
    let c_t = tier_block_lanes(tier);
    let k_t = (tier_kautz_banks(tier) * 4).min(k_keep);
    let s2_per_stem = collect_s2_per_stem(&model.readouts[tau], ch, c_t, stems_pre, trajs);
    let grams = assemble_tier_band_grams::<B>(
        tier,
        &model.kautz_poles,
        k_t,
        &dataset.bank,
        stems_pre,
        &s2_per_stem,
    );
    let anchor: Vec<(usize, f64)> = if tau > 0 && model.active_tier_weights[tau - 1] > 0.0 {
        map_prev_tier_anchor(tau, &model.readouts[tau - 1], ch)
            .into_iter()
            .filter(|&(idx, _)| idx < grams.p_dim)
            .collect()
    } else {
        Vec::new()
    };
    let bypass = bypass_ridge_cols(0, c_t, model.svf_env_oct_sens.abs() > 0.5);
    let theta = solve_uplo_with_bypass_and_anchor(
        grams.p_dim,
        &grams.gram_tot,
        &grams.rhs_tot,
        &bypass,
        &anchor,
        2.0e-6,
    );
    unpack_theta_k_to_readout(tier, ch, k_t, &theta, &mut model.readouts[tau]);
    grams
}

pub(super) fn solve_tier_output_chan<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    tau: usize,
    ch: usize,
    dataset: &V3TrainDataset,
    trajs: &[StemChanTraj],
    stems_pre: &[PreSplitStemChan],
) {
    let _ = solve_tier_output_k_chan::<B>(
        model,
        tau,
        ch,
        tier_kautz_banks(ALL_TIERS[tau]) * 4,
        dataset,
        trajs,
        stems_pre,
    );
}

fn collect_shared_core_trajs(
    model: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    dataset: &V3TrainDataset,
    c_blk: usize,
    active_ch: usize,
    sub_steps: usize,
) -> [Vec<StemChanTraj>; 2] {
    let run_stem = |si: usize| -> [StemChanTraj; 2] {
        let st = &dataset.stems[si];
        let n = st.usable_len();
        let mut ws = AdjointWorkspace::default();
        ws.ensure_capacity(n, sub_steps, &[0; 4], false, true);
        if n > 0 {
            forward_shared_core(
                model,
                cp,
                &st.in_l[..n],
                &st.in_r[..n],
                n,
                c_blk,
                active_ch,
                false,
                true,
                &mut ws,
            );
        }
        std::array::from_fn(|ch| {
            if ch < active_ch {
                StemChanTraj {
                    tap: ws.tap_dec[ch][..n].to_vec(),
                    u: ws.u_dec[ch][..n].to_vec(),
                    hp: ws.hp_dec[ch][..n].to_vec(),
                    h: ws.h_dec[ch][..n].to_vec(),
                }
            } else {
                StemChanTraj {
                    tap: Vec::new(),
                    u: Vec::new(),
                    hp: Vec::new(),
                    h: Vec::new(),
                }
            }
        })
    };
    let stem_res: Vec<[StemChanTraj; 2]> =
        if dataset.stems.len() >= 2 && dataset.stems[0].usable_len() >= 8_192 {
            par_map_indexed(dataset.stems.len(), run_stem)
        } else {
            (0..dataset.stems.len()).map(run_stem).collect()
        };
    let mut trajs = [
        Vec::with_capacity(dataset.stems.len()),
        Vec::with_capacity(dataset.stems.len()),
    ];
    for [t0, t1] in stem_res {
        trajs[0].push(t0);
        if active_ch > 1 {
            trajs[1].push(t1);
        }
    }
    trajs
}

/// Solves nested Bézout-weighted Matryoshka linear readouts (+ `pre_cab` ALS) via `B`.
pub fn solve_matryoshka_varpro_cholesky_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    als_iters: usize,
) {
    let active_ch = if model.mono_out { 1 } else { 2 };
    normalize_pre_cab_gauge(&mut model.readouts, active_ch);
    let cp = ClampedCoreParams::from_model(model);
    let g_cnt = model.max_active_groups.clamp(1, MAX_GROUPS);
    let c_blk = g_cnt * 4;
    let sub_steps = if model.os2x { 2 } else { 1 };
    let trajs = collect_shared_core_trajs(model, &cp, dataset, c_blk, active_ch, sub_steps);

    for (ch, ch_trajs) in trajs.iter().enumerate().take(active_ch) {
        let stems_pre = PreSplitStemChan::build_vec::<B>(dataset, ch_trajs, ch, c_blk);
        for (tau, &tier) in ALL_TIERS.iter().enumerate() {
            if tier_groups(tier) > g_cnt || model.active_tier_weights[tau] <= 0.0 {
                continue;
            }
            solve_tier_output_chan::<B>(model, tau, ch, dataset, ch_trajs, &stems_pre);
            for _ in 0..als_iters {
                if als_update_pre_cab_chan::<B>(
                    model,
                    tau,
                    ch,
                    dataset,
                    ch_trajs,
                    &stems_pre,
                    &cp.kautz_pre,
                ) {
                    solve_tier_output_chan::<B>(model, tau, ch, dataset, ch_trajs, &stems_pre);
                } else {
                    break;
                }
            }
        }
    }
    normalize_pre_cab_gauge(&mut model.readouts, active_ch);
    model.mirror_mono();
}

/// Dispatches [`solve_matryoshka_varpro_cholesky_with_backend`] via fastest CPU backend.
pub fn solve_matryoshka_varpro_cholesky(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    als_iters: usize,
) {
    struct Task<'a>(&'a mut V3MatryoshkaOptF64, &'a V3TrainDataset, usize);
    impl TrainerTask<()> for Task<'_> {
        fn run<B: TrainerBackend>(self) {
            solve_matryoshka_varpro_cholesky_with_backend::<B>(self.0, self.1, self.2);
        }
    }
    DynamicTrainer::detect().dispatch(Task(model, dataset, als_iters));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_als_helpers_pack_anchor_and_commit_roundtrip() {
        let mut r = V3TierReadoutF64::zeros();
        r.pre_cab_u_gain[0] = 0.25;
        r.pre_cab_hp_gain[0] = -0.15;
        r.pre_cab_w[0][0] = 0.5;
        r.svf_direct_taps_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        r.direct_s2_u_gain[0] = -0.3;
        r.direct_s2_hp_gain[0] = 0.4;
        r.direct_s2_w[0][MAX_BLOCK_LANES] = -0.2;

        let c_t = 4;
        let q_pre = 2 + 3 * c_t;
        let p_tot = q_pre + 9 + 3 * c_t;
        let sol = pack_als_baseline(&r, 0, c_t, q_pre, p_tot, false);
        assert_eq!(sol[0], 0.25);
        assert_eq!(sol[q_pre + 6], 0.7);
        assert_eq!(sol[q_pre + 9 + c_t], -0.2);

        let anchor = build_als_anchor(&r, true, 0, c_t, 4, q_pre, p_tot);
        assert_eq!(anchor.len(), p_tot);
        assert_eq!(anchor[0], (0, 0.25));

        let bypass = bypass_ridge_cols(q_pre, c_t, true);
        assert_eq!(bypass.len(), 9 + 3 * c_t);
        assert!((bypass[7].1 - 2.5e-2).abs() < 1.0e-12);

        let committed = commit_als_solution(&r, None, 0, 0, c_t, q_pre, true, &sol);
        assert!(committed.fir_l[0] > 0.0);
        assert!((committed.fir_l[0] * committed.pre_cab_u_gain[0] - 0.25).abs() < 1.0e-12);
        assert_eq!(committed.svf_direct_taps_l, r.svf_direct_taps_l);
    }
}
