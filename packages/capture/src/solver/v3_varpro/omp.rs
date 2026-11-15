#![forbid(unsafe_code)]
//! Hyperbolic Cauchy/Szegő Kautz Pole Overlap & Progressive-Slot Orthogonal Matching Pursuit (`OMP`).

use std::f64::consts::PI;

use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_KAUTZ_POLES, NUM_GRU_BLOCKS, tier_block_lanes,
    tier_fir_segments, tier_groups,
};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::basis::PreSplitStemChan;
use super::basis::{
    ALL_TIERS, KAUTZ_DELAY_SAMPLES, StemChanTraj, apply_fixed_cabinet_f64, eval_s2_from_readout,
    kautz_pre_tuples, solve_matryoshka_varpro_cholesky_with_backend, solve_tier_output_k_chan,
    split_append_3band,
};
use super::gru_core::{AdjointWorkspace, forward_shared_core};
use super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};
use super::primitives::{ClampedCoreParams, par_map_indexed};

/// Normalized Cauchy/Szegő inner-product overlap between two stable discrete-time poles in `[0, 1]`.
pub(crate) fn kautz_pole_overlap(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (ra, tha) = (a.0.clamp(0.25, 0.9985), a.1.clamp(0.0, PI - 1.0e-3));
    let (rb, thb) = (b.0.clamp(0.25, 0.9985), b.1.clamp(0.0, PI - 1.0e-3));
    let num = ((1.0 - ra * ra).max(1.0e-8) * (1.0 - rb * rb).max(1.0e-8)).sqrt();
    let prod = ra * rb;
    let d1 = (1.0 + prod * prod - 2.0 * prod * (tha - thb).cos())
        .max(1.0e-12)
        .sqrt();
    let d2 = (1.0 + prod * prod - 2.0 * prod * (tha + thb).cos())
        .max(1.0e-12)
        .sqrt();
    (num / d1.min(d2)).clamp(0.0, 1.0)
}

fn build_pole_dictionary(sr_hz: f64) -> Vec<(f64, f64)> {
    const REAL_RADII: [f64; 10] = [
        0.35, 0.62, 0.82, 0.92, 0.972, 0.986, 0.994, 0.9955, 0.9968, 0.9982,
    ];
    const RADII: [f64; 7] = [0.60, 0.78, 0.90, 0.955, 0.982, 0.9945, 0.9978];
    let f_min = 14.0_f64;
    let f_max = (0.43 * sr_hz).max(200.0);
    let mut poles = Vec::with_capacity(REAL_RADII.len() + 36 * RADII.len());
    for &r in &REAL_RADII {
        poles.push((r, 0.0));
    }
    for fi in 0..36 {
        let u = (fi as f64) / 35.0;
        let freq = f_min * (f_max / f_min).powf(u);
        let th = (2.0 * PI * freq / sr_hz).clamp(1.5e-3, PI - 1.0e-3);
        for &r in &RADII {
            poles.push((r, th));
        }
    }
    poles
}

/// Greedily selects progressive Matryoshka Kautz pole slots `0..4 / 4..8 / 8..16 / 16..32` via `B`.
pub fn select_omp_kautz_poles_matryoshka_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
) {
    const SLOTS: [(usize, usize, usize); 4] = [(0, 0, 4), (1, 4, 8), (2, 8, 16), (3, 16, 32)];
    let Some(st0_orig) = dataset.stems.first() else {
        return;
    };
    let n = st0_orig.usable_len().min(20_480);
    if n <= 64 {
        return;
    }
    let ds_omp = dataset.subwindow_bezout_only(n);
    let st0 = &ds_omp.stems[0];
    let w = st0.warmup.min(n / 2);
    let seg = n.saturating_sub(w);
    let m_w = 3 * seg;
    if m_w == 0 {
        return;
    }
    let bw = ds_omp.stem_band_weights[0][0];
    let sq_bw = [
        bw[0].max(0.0).sqrt(),
        bw[1].max(0.0).sqrt(),
        bw[2].max(0.0).sqrt(),
    ];

    let g_cnt = model.max_active_groups.clamp(1, MAX_GROUPS);
    let c_blk = g_cnt * 4;
    let active_ch = if model.mono_out { 1 } else { 2 };
    let sub_steps = if model.os2x { 2 } else { 1 };
    let cp = ClampedCoreParams::from_model(model);
    let build_traj0 = |si: usize| -> StemChanTraj {
        let st = &ds_omp.stems[si];
        let ni = st.usable_len();
        let mut ws = AdjointWorkspace::default();
        ws.ensure_capacity(ni, sub_steps, &[0; 4], false, true);
        if ni > 0 {
            forward_shared_core(
                model,
                &cp,
                &st.in_l[..ni],
                &st.in_r[..ni],
                ni,
                c_blk,
                active_ch,
                false,
                true,
                &mut ws,
            );
        }
        StemChanTraj {
            tap: ws.tap_dec[0][..ni].to_vec(),
            u: ws.u_dec[0][..ni].to_vec(),
            hp: ws.hp_dec[0][..ni].to_vec(),
            h: ws.h_dec[0][..ni].to_vec(),
        }
    };
    let trajs_0: Vec<StemChanTraj> = if ds_omp.stems.len() >= 2 {
        par_map_indexed(ds_omp.stems.len(), build_traj0)
    } else {
        (0..ds_omp.stems.len()).map(build_traj0).collect()
    };
    let stems_pre = PreSplitStemChan::build_vec::<B>(&ds_omp, &trajs_0, 0, c_blk);

    let dict = build_pole_dictionary(ds_omp.sample_rate_hz);
    let dict_pre = kautz_pre_tuples(&dict);
    let mut dict_re = vec![vec![0.0_f64; m_w]; dict.len()];
    let mut dict_im = vec![vec![0.0_f64; m_w]; dict.len()];
    let mut s2_bands = [Vec::new(), Vec::new(), Vec::new()];
    let mut s2_ref = Vec::new();
    let mut prev_s2 = Vec::new();
    let mut res_raw = vec![0.0_f64; n];
    let mut res_w = Vec::with_capacity(m_w);
    let mut scratch_k = vec![vec![0.0_f64; n]; 2 * MAX_KAUTZ_POLES];
    let mut prev_re = Vec::new();
    let mut prev_im = Vec::new();
    let tr0 = &trajs_0[0];

    let mut cab_buf = vec![0.0_f64; n];
    for &(tau, s_start, s_end) in &SLOTS {
        if tier_groups(ALL_TIERS[tau]) > g_cnt || model.active_tier_weights[tau] <= 0.0 {
            continue;
        }
        let _ =
            solve_tier_output_k_chan::<B>(model, tau, 0, s_start, &ds_omp, &trajs_0, &stems_pre);
        let tier = ALL_TIERS[tau];
        let c_t = tier_block_lanes(tier);
        let f_t = tier_fir_segments(tier) * 16;
        let cp_cur = ClampedCoreParams::from_model(model);
        let r = &model.readouts[tau];
        eval_s2_from_readout(r, 0, c_t, &tr0.u, &tr0.hp, &tr0.h, &mut s2_ref);
        if prev_s2 != s2_ref {
            ds_omp.bank.split_3band(&s2_ref, &mut s2_bands);
            let n_dict = dict.len();
            let n_chunks = 16.min(n_dict.max(1));
            let chunk_sz = n_dict.div_ceil(n_chunks);
            let chunks = par_map_indexed(n_chunks, |ci| {
                let start = (ci * chunk_sz).min(n_dict);
                let end = (start + chunk_sz).min(n_dict);
                let cnt = end.saturating_sub(start);
                let mut local_cols = vec![vec![0.0_f64; n]; 2 * cnt];
                let mut local_re = vec![vec![0.0_f64; m_w]; cnt];
                let mut local_im = vec![vec![0.0_f64; m_w]; cnt];
                if cnt > 0 {
                    for (b, s2_b) in s2_bands.iter().enumerate() {
                        B::filter_pole_bank_columns(
                            s2_b,
                            KAUTZ_DELAY_SAMPLES,
                            &dict_pre[start..end],
                            &mut local_cols,
                            0,
                        );
                        let scale = sq_bw[b];
                        let dst_off = b * seg;
                        for k in 0..cnt {
                            let src_re = &local_cols[2 * k][w..n];
                            let src_im = &local_cols[2 * k + 1][w..n];
                            for off in 0..seg {
                                local_re[k][dst_off + off] = scale * src_re[off];
                                local_im[k][dst_off + off] = scale * src_im[off];
                            }
                        }
                    }
                    for k in 0..cnt {
                        B::orthonormalize_pole_pair(&mut local_re[k], &mut local_im[k]);
                    }
                }
                (start, local_re, local_im)
            });
            for (start, local_re, local_im) in chunks {
                for (k, (re, im)) in local_re.into_iter().zip(local_im).enumerate() {
                    dict_re[start + k] = re;
                    dict_im[start + k] = im;
                }
            }
            prev_s2.clear();
            prev_s2.extend_from_slice(&s2_ref);
        }

        apply_fixed_cabinet_f64::<B>(
            &s2_ref,
            &r.fir_l[..f_t],
            &cp_cur.kautz_pre[..s_start],
            &r.kautz_c_l[..s_start],
            &mut cab_buf,
        );
        for t in 0..n {
            let mut y = r.direct_s2_u_gain[0] * tr0.u[t] + r.direct_s2_hp_gain[0] * tr0.hp[t];
            for i in 0..7 {
                y += r.svf_direct_taps_l[i] * tr0.tap[t][i];
            }
            for b in 0..NUM_GRU_BLOCKS {
                for l in 0..c_t {
                    y += r.direct_s2_w[0][b * MAX_BLOCK_LANES + l]
                        * tr0.h[t][b * MAX_BLOCK_LANES + l];
                }
            }
            res_raw[t] = f64::from(st0.tgt_l[t]) - (y + cab_buf[t]);
        }
        res_w.clear();
        split_append_3band(&ds_omp.bank, &res_raw, w, sq_bw, &mut res_w);

        if s_start > 0 {
            let prev_pre = kautz_pre_tuples(&model.kautz_poles[..s_start]);
            prev_re.resize(s_start, vec![0.0_f64; m_w]);
            prev_im.resize(s_start, vec![0.0_f64; m_w]);
            for p_vec in prev_re.iter_mut().chain(prev_im.iter_mut()) {
                p_vec.resize(m_w, 0.0);
            }
            for (b, s2_b) in s2_bands.iter().enumerate() {
                B::filter_pole_bank_columns(
                    s2_b,
                    KAUTZ_DELAY_SAMPLES,
                    &prev_pre,
                    &mut scratch_k[..2 * s_start],
                    0,
                );
                let scale = sq_bw[b];
                let dst_off = b * seg;
                for prev in 0..s_start {
                    let src_re = &scratch_k[2 * prev][w..n];
                    let src_im = &scratch_k[2 * prev + 1][w..n];
                    for off in 0..seg {
                        prev_re[prev][dst_off + off] = scale * src_re[off];
                        prev_im[prev][dst_off + off] = scale * src_im[off];
                    }
                }
            }
            for prev in 0..s_start {
                B::orthonormalize_pole_pair(&mut prev_re[prev], &mut prev_im[prev]);
                let c1 = B::dot_f64(&res_w, &prev_re[prev]);
                B::axpy_f64(&mut res_w, -c1, &prev_re[prev]);
                let c2 = B::dot_f64(&res_w, &prev_im[prev]);
                B::axpy_f64(&mut res_w, -c2, &prev_im[prev]);
            }
        }
        for slot in s_start..s_end {
            let mut best_di = 0;
            let mut best_score = -1.0_f64;
            for (di, cand) in dict.iter().enumerate() {
                let dup = (0..slot).any(|p| kautz_pole_overlap(*cand, model.kautz_poles[p]) > 0.85);
                if dup {
                    continue;
                }
                let sc = B::score_pole_pair(&res_w, &dict_re[di], &dict_im[di]);
                if sc.is_finite() && sc > best_score {
                    best_score = sc;
                    best_di = di;
                }
            }
            model.kautz_poles[slot] = dict[best_di];
            let c1 = B::dot_f64(&res_w, &dict_re[best_di]);
            B::axpy_f64(&mut res_w, -c1, &dict_re[best_di]);
            let c2 = B::dot_f64(&res_w, &dict_im[best_di]);
            B::axpy_f64(&mut res_w, -c2, &dict_im[best_di]);
        }
    }
    solve_matryoshka_varpro_cholesky_with_backend::<B>(model, &ds_omp, 0);
}

/// Replaces any Kautz pole slots that drifted within hyperbolic collision threshold `> 0.88`.
pub fn repair_collided_kautz_poles(model: &mut V3MatryoshkaOptF64, sr_hz: f64) {
    let dict = build_pole_dictionary(sr_hz);
    for slot in 1..MAX_KAUTZ_POLES {
        let cur = model.kautz_poles[slot];
        if !(0..slot).any(|p| kautz_pole_overlap(cur, model.kautz_poles[p]) > 0.88) {
            continue;
        }
        let mut best = cur;
        let mut best_d = f64::INFINITY;
        for &cand in &dict {
            if (0..slot).all(|p| kautz_pole_overlap(cand, model.kautz_poles[p]) <= 0.84) {
                let d = (cand.0 - cur.0).powi(2) + (cand.1 - cur.1).powi(2);
                if d < best_d {
                    best_d = d;
                    best = cand;
                }
            }
        }
        model.kautz_poles[slot] = best;
    }
}

/// Dispatches [`select_omp_kautz_poles_matryoshka_with_backend`] via fastest CPU backend.
pub fn select_omp_kautz_poles_matryoshka(model: &mut V3MatryoshkaOptF64, dataset: &V3TrainDataset) {
    struct Task<'a>(&'a mut V3MatryoshkaOptF64, &'a V3TrainDataset);
    impl TrainerTask<()> for Task<'_> {
        fn run<B: TrainerBackend>(self) {
            select_omp_kautz_poles_matryoshka_with_backend::<B>(self.0, self.1);
        }
    }
    DynamicTrainer::detect().dispatch(Task(model, dataset));
}

#[cfg(test)]
mod tests {
    use dnam_engine::stages::v3_core::V3MatryoshkaModel;

    use super::*;
    use crate::solver::v3_varpro::model_f64::V3TrainStem;

    fn make_resonant_stem(freq_hz: f64, len: usize) -> V3TrainStem {
        let mut in_sig = vec![0.0_f32; len];
        let mut tgt_sig = vec![0.0_f32; len];
        let (mut lp1, mut lp2) = (0.0_f64, 0.0_f64);
        for t in 0..len {
            let phase = (t as f64) / 48_000.0;
            let x = 0.35 * (2.0 * PI * freq_hz * phase).sin()
                + 0.15 * (2.0 * PI * (freq_hz * 1.75) * phase).cos();
            lp1 = 0.85 * lp1 + 0.15 * x;
            lp2 = 0.90 * lp2 + 0.10 * lp1;
            in_sig[t] = x as f32;
            tgt_sig[t] = (0.55 * lp2 + 0.25 * (2.0 * x).tanh()) as f32;
        }
        V3TrainStem {
            in_l: in_sig.clone(),
            in_r: in_sig,
            tgt_l: tgt_sig.clone(),
            tgt_r: tgt_sig,
            warmup: 32,
        }
    }

    #[test]
    fn test_omp_pole_selection_guards_multi_stem_and_collision_repair() {
        let mut model =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, true));
        let empty_ds = V3TrainDataset::new(vec![], 48_000.0);
        select_omp_kautz_poles_matryoshka(&mut model, &empty_ds);

        let short_stem = make_resonant_stem(220.0, 32);
        let short_ds = V3TrainDataset::new(vec![short_stem], 48_000.0);
        select_omp_kautz_poles_matryoshka(&mut model, &short_ds);

        // First test Lite-only active group gating, then full 4-tier progressive selection
        let ds_two = V3TrainDataset::new(
            vec![
                make_resonant_stem(180.0, 1_536),
                make_resonant_stem(740.0, 1_536),
            ],
            48_000.0,
        );
        model.max_active_groups = 1;
        model.active_tier_weights = [1.0, 0.0, 0.0, 0.0];
        select_omp_kautz_poles_matryoshka(&mut model, &ds_two);

        model.max_active_groups = MAX_GROUPS;
        model.active_tier_weights = [0.2, 0.2, 0.3, 0.3];
        select_omp_kautz_poles_matryoshka(&mut model, &ds_two);
        for &(r, th) in &model.kautz_poles {
            assert!(r > 0.0 && r < 1.0, "invalid Kautz pole radius {r}");
            assert!(th.is_finite() && (0.0..PI).contains(&th));
        }

        // Exact collision triggers replacement with a separated dictionary candidate
        model.kautz_poles[1] = model.kautz_poles[0];
        repair_collided_kautz_poles(&mut model, 48_000.0);
        let overlap = kautz_pole_overlap(model.kautz_poles[0], model.kautz_poles[1]);
        assert!(overlap <= 0.85, "repaired pole still overlapped: {overlap}");
    }
}
