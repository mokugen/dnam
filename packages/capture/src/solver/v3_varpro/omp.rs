#![forbid(unsafe_code)]
//! Hyperbolic Cauchy/Szegő Kautz Pole Overlap & Progressive-Slot Orthogonal Matching Pursuit (`OMP`).

mod refine;

use std::f64::consts::PI;

use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_KAUTZ_POLES, NUM_GRU_BLOCKS, tier_block_lanes_array,
    tier_fir_segments, tier_groups,
};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::basis::{
    ALL_TIERS, KAUTZ_DELAY_SAMPLES, PreSplitStemChan, StemChanTraj, active_block_lanes,
    apply_fixed_cabinet_f64, eval_s2_from_readout, kautz_pre_tuples, normalize_pre_cab_gauge,
    solve_matryoshka_varpro_cholesky_with_backend, solve_tier_output_chan,
    solve_tier_output_k_chan, split_append_3band,
};
use super::gru_core::{AdjointWorkspace, forward_shared_core};
use super::model_f64::{Bezout7FilterBank, V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset};
use super::primitives::{ClampedCoreParams, par_map_indexed};
use refine::{RefineCtx, StemMeta, refine_selected_pole_coord};

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

fn push_log_freq_band(
    poles: &mut Vec<(f64, f64)>,
    sr_hz: f64,
    f_lo: f64,
    f_hi: f64,
    count: usize,
    include_start: bool,
    include_end: bool,
) {
    const RADII: [f64; 6] = [0.62, 0.80, 0.91, 0.960, 0.985, 0.9965];
    let denom = if include_start && include_end {
        count.saturating_sub(1).max(1) as f64
    } else {
        count as f64
    };
    let ratio = (f_hi / f_lo).max(1.0);
    for idx in 0..count {
        let step = if include_start { idx } else { idx + 1 };
        let u = (step as f64) / denom;
        let freq = f_lo * ratio.powf(u);
        let th = (2.0 * PI * freq / sr_hz).clamp(1.5e-3, PI - 1.0e-3);
        for &r in &RADII {
            poles.push((r, th));
        }
    }
}

fn build_pole_dictionary(sr_hz: f64) -> Vec<(f64, f64)> {
    const REAL_RADII: [f64; 10] = [
        0.35, 0.62, 0.82, 0.92, 0.972, 0.986, 0.994, 0.9955, 0.9968, 0.9982,
    ];
    let mut poles = Vec::with_capacity(REAL_RADII.len() + 64 * 6);
    for &r in &REAL_RADII {
        poles.push((r, 0.0));
    }
    let f_max = (0.43 * sr_hz).max(5_600.0);
    push_log_freq_band(&mut poles, sr_hz, 14.0, 25.0, 8, true, false);
    push_log_freq_band(&mut poles, sr_hz, 25.0, 5_500.0, 48, true, true);
    push_log_freq_band(&mut poles, sr_hz, 5_500.0, f_max, 8, false, true);
    poles
}

fn collect_pool_trajs(
    model: &V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    c_blk: [usize; NUM_GRU_BLOCKS],
    active_ch: usize,
) -> [Vec<StemChanTraj>; 2] {
    let sub_steps = if model.os2x { 2 } else { 1 };
    let cp = ClampedCoreParams::from_model(model);
    let build_stem = |si: usize| -> [StemChanTraj; 2] {
        let st = &dataset.stems[si];
        let ni = st.usable_len();
        let mut ws = AdjointWorkspace::default();
        ws.ensure_capacity(ni, sub_steps, active_ch, &[0; 4], false, true);
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
        std::array::from_fn(|ch| {
            if ch < active_ch {
                StemChanTraj {
                    tap: ws.tap_dec[ch][..ni].to_vec(),
                    u: ws.u_dec[ch][..ni].to_vec(),
                    hp: ws.hp_dec[ch][..ni].to_vec(),
                    h: ws.h_dec[ch][..ni].to_vec(),
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
    let stem_res: Vec<[StemChanTraj; 2]> = if dataset.stems.len() >= 2 {
        par_map_indexed(dataset.stems.len(), build_stem)
    } else {
        (0..dataset.stems.len()).map(build_stem).collect()
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

fn fill_pole_bank_pooled<B: TrainerBackend>(
    s2_bands_pool: &[[Vec<f64>; 3]],
    pool_meta: &[StemMeta],
    sq_bw_pool: &[[f64; 3]],
    tuples: &[(f64, f64, f64)],
    m_w: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let cnt = tuples.len();
    let mut out_re = vec![vec![0.0_f64; m_w]; cnt];
    let mut out_im = vec![vec![0.0_f64; m_w]; cnt];
    if cnt == 0 {
        return (out_re, out_im);
    }
    for (si, meta) in pool_meta.iter().enumerate() {
        if meta.seg == 0 {
            continue;
        }
        let mut local_cols = vec![vec![0.0_f64; meta.n]; 2 * cnt];
        for (b, s2_b) in s2_bands_pool[si].iter().enumerate() {
            B::filter_pole_bank_columns(s2_b, KAUTZ_DELAY_SAMPLES, tuples, &mut local_cols, 0);
            let scale = sq_bw_pool[si][b];
            let dst_off = meta.m_off + b * meta.seg;
            for k in 0..cnt {
                let src_re = &local_cols[2 * k][meta.w..meta.n];
                let src_im = &local_cols[2 * k + 1][meta.w..meta.n];
                let dst_re = &mut out_re[k][dst_off..dst_off + meta.seg];
                let dst_im = &mut out_im[k][dst_off..dst_off + meta.seg];
                for off in 0..meta.seg {
                    dst_re[off] = scale * src_re[off];
                    dst_im[off] = scale * src_im[off];
                }
            }
        }
    }
    for k in 0..cnt {
        B::orthonormalize_pole_pair(&mut out_re[k], &mut out_im[k]);
    }
    (out_re, out_im)
}

fn project_and_orthonormalize_pair<B: TrainerBackend>(
    basis: &[(Vec<f64>, Vec<f64>)],
    re: &mut [f64],
    im: &mut [f64],
) {
    for (prev_re, prev_im) in basis {
        let c_rr = B::dot_f64(re, prev_re);
        B::axpy_f64(re, -c_rr, prev_re);
        let c_ri = B::dot_f64(re, prev_im);
        B::axpy_f64(re, -c_ri, prev_im);
        let c_ir = B::dot_f64(im, prev_re);
        B::axpy_f64(im, -c_ir, prev_re);
        let c_ii = B::dot_f64(im, prev_im);
        B::axpy_f64(im, -c_ii, prev_im);
    }
    B::orthonormalize_pole_pair(re, im);
}

fn deflate_residual_pair<B: TrainerBackend>(res_w: &mut [f64], re: &[f64], im: &[f64]) {
    let c1 = B::dot_f64(res_w, re);
    B::axpy_f64(res_w, -c1, re);
    let c2 = B::dot_f64(res_w, im);
    B::axpy_f64(res_w, -c2, im);
}

struct ChanPoolUpdateCtx<'a> {
    ch: usize,
    c_t: [usize; NUM_GRU_BLOCKS],
    f_t: usize,
    s_start: usize,
    m_w: usize,
    pool_meta: &'a [StemMeta],
    sq_bw: &'a [[f64; 3]],
    bank: &'a Bezout7FilterBank,
    dict_pre: &'a [(f64, f64, f64)],
    prev_poles: &'a [(f64, f64)],
}

pub(super) struct ChannelOmpState {
    pub(super) dict_re: Vec<Vec<f64>>,
    pub(super) dict_im: Vec<Vec<f64>>,
    pub(super) s2_bands: Vec<[Vec<f64>; 3]>,
    pub(super) prev_s2: Vec<Vec<f64>>,
    pub(super) res_w: Vec<f64>,
    pub(super) ortho_basis: Vec<(Vec<f64>, Vec<f64>)>,
}

fn update_channel_for_tier_slot<B: TrainerBackend>(
    ctx: &ChanPoolUpdateCtx<'_>,
    r: &V3TierReadoutF64,
    kautz_pre: &[(f64, f64, f64, f64, f64, f64)],
    ds_omp: &V3TrainDataset,
    trajs_ch: &[StemChanTraj],
    state: &mut ChannelOmpState,
) {
    let n_pool = ctx.pool_meta.len();
    let mut s2_refs = vec![Vec::new(); n_pool];
    for si in 0..n_pool {
        let tr = &trajs_ch[si];
        eval_s2_from_readout(r, ctx.ch, ctx.c_t, &tr.u, &tr.hp, &tr.h, &mut s2_refs[si]);
    }

    if state.prev_s2 != s2_refs {
        state
            .s2_bands
            .resize_with(n_pool, || [Vec::new(), Vec::new(), Vec::new()]);
        for si in 0..n_pool {
            ctx.bank.split_3band(&s2_refs[si], &mut state.s2_bands[si]);
        }
        let n_dict = ctx.dict_pre.len();
        let n_chunks = 16.min(n_dict.max(1));
        let chunk_sz = n_dict.div_ceil(n_chunks);
        let chunks = par_map_indexed(n_chunks, |ci| {
            let start = (ci * chunk_sz).min(n_dict);
            let end = (start + chunk_sz).min(n_dict);
            let (l_re, l_im) = fill_pole_bank_pooled::<B>(
                &state.s2_bands,
                ctx.pool_meta,
                ctx.sq_bw,
                &ctx.dict_pre[start..end],
                ctx.m_w,
            );
            (start, l_re, l_im)
        });
        for (start, local_re, local_im) in chunks {
            for (k, (re, im)) in local_re.into_iter().zip(local_im).enumerate() {
                state.dict_re[start + k] = re;
                state.dict_im[start + k] = im;
            }
        }
        state.prev_s2 = s2_refs.clone();
    }

    let direct_taps = if ctx.ch == 0 {
        &r.svf_direct_taps_l
    } else {
        &r.svf_direct_taps_r
    };
    let fir_slice = if ctx.ch == 0 {
        &r.fir_l[..ctx.f_t]
    } else {
        &r.fir_r[..ctx.f_t]
    };
    let kc_slice = if ctx.ch == 0 {
        &r.kautz_c_l[..ctx.s_start]
    } else {
        &r.kautz_c_r[..ctx.s_start]
    };
    state.res_w.clear();
    let mut cab_buf = Vec::new();
    let mut res_raw = Vec::new();
    for (si, meta) in ctx.pool_meta.iter().enumerate() {
        let tr = &trajs_ch[si];
        let st = &ds_omp.stems[si];
        let tgt = if ctx.ch == 0 { &st.tgt_l } else { &st.tgt_r };
        cab_buf.resize(meta.n, 0.0);
        res_raw.resize(meta.n, 0.0);
        apply_fixed_cabinet_f64::<B>(
            &s2_refs[si],
            fir_slice,
            &kautz_pre[..ctx.s_start],
            kc_slice,
            &mut cab_buf,
        );
        for t in 0..meta.n {
            let mut y =
                r.direct_s2_u_gain[ctx.ch] * tr.u[t] + r.direct_s2_hp_gain[ctx.ch] * tr.hp[t];
            for i in 0..7 {
                y += direct_taps[i] * tr.tap[t][i];
            }
            for b in 0..NUM_GRU_BLOCKS {
                for l in 0..ctx.c_t[b] {
                    y += r.direct_s2_w[ctx.ch][b * MAX_BLOCK_LANES + l]
                        * tr.h[t][b * MAX_BLOCK_LANES + l];
                }
            }
            res_raw[t] = f64::from(tgt[t]) - (y + cab_buf[t]);
        }
        split_append_3band(ctx.bank, &res_raw, meta.w, ctx.sq_bw[si], &mut state.res_w);
    }

    state.ortho_basis.clear();
    if ctx.s_start > 0 {
        let prev_pre = kautz_pre_tuples(ctx.prev_poles);
        let (mut prev_re, mut prev_im) = fill_pole_bank_pooled::<B>(
            &state.s2_bands,
            ctx.pool_meta,
            ctx.sq_bw,
            &prev_pre,
            ctx.m_w,
        );
        for prev in 0..ctx.s_start {
            project_and_orthonormalize_pair::<B>(
                &state.ortho_basis,
                &mut prev_re[prev],
                &mut prev_im[prev],
            );
            deflate_residual_pair::<B>(&mut state.res_w, &prev_re[prev], &prev_im[prev]);
            state.ortho_basis.push((
                std::mem::take(&mut prev_re[prev]),
                std::mem::take(&mut prev_im[prev]),
            ));
        }
    }
}

fn stem_has_active_energy(st: &super::model_f64::V3TrainStem) -> bool {
    let n = st.usable_len();
    let w = st.warmup.min(n / 2);
    if n <= w + 16 {
        return false;
    }
    let dry_e: f64 = st.in_l[w..n]
        .iter()
        .zip(&st.in_r[w..n])
        .map(|(&l, &r)| f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r))
        .sum();
    let tgt_e: f64 = st.tgt_l[w..n]
        .iter()
        .zip(&st.tgt_r[w..n])
        .map(|(&l, &r)| f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r))
        .sum();
    dry_e > 1.0e-6 && tgt_e > 1.0e-6
}

/// Greedily selects progressive Matryoshka Kautz pole slots `0..4 / 4..8 / 8..12 / 12..16` via `B`.
pub fn select_omp_kautz_poles_matryoshka_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
) {
    const SLOTS: [(usize, usize, usize); 4] = [(0, 0, 4), (1, 4, 8), (2, 8, 12), (3, 12, 16)];
    let Some(st0_orig) = dataset.stems.first() else {
        return;
    };
    let g_cnt = model.max_active_groups.clamp(1, MAX_GROUPS);
    let n_cap = if g_cnt == 1 { 6_144 } else { 10_240 };
    let n = st0_orig.usable_len().min(n_cap);
    if n <= 64 {
        return;
    }
    let keep_sec = dataset.stems.get(1).is_some_and(stem_has_active_energy);
    let ds_omp = if keep_sec {
        dataset.subwindow_bezout_only(n)
    } else {
        dataset.primary_subwindow_opts(n, false, false)
    };
    let n_pool = ds_omp.stems.len().min(2);
    let mut pool_meta = Vec::with_capacity(n_pool);
    let mut m_w = 0_usize;
    for st in ds_omp.stems.iter().take(n_pool) {
        let ni = st.usable_len();
        let wi = st.warmup.min(ni / 2);
        let seg = ni.saturating_sub(wi);
        pool_meta.push(StemMeta {
            n: ni,
            w: wi,
            seg,
            m_off: m_w,
        });
        m_w += 3 * seg;
    }
    if m_w == 0 {
        return;
    }

    let c_blk = active_block_lanes(g_cnt);
    let active_ch = if model.mono_out { 1 } else { 2 };
    normalize_pre_cab_gauge(&mut model.readouts, active_ch);
    model.mirror_mono();

    let trajs = collect_pool_trajs(model, &ds_omp, c_blk, active_ch);
    let stems_pre: Vec<Vec<PreSplitStemChan>> = (0..active_ch)
        .map(|ch| PreSplitStemChan::build_vec::<B>(&ds_omp, &trajs[ch], ch, c_blk))
        .collect();
    let sq_bw_all: Vec<Vec<[f64; 3]>> = (0..active_ch)
        .map(|ch| {
            (0..n_pool)
                .map(|si| ds_omp.stem_band_weights[si][ch].map(|v| v.max(0.0).sqrt()))
                .collect()
        })
        .collect();

    let dict = build_pole_dictionary(ds_omp.sample_rate_hz);
    let dict_pre = kautz_pre_tuples(&dict);
    let mut chan_states: Vec<ChannelOmpState> = (0..active_ch)
        .map(|_| ChannelOmpState {
            dict_re: vec![vec![0.0_f64; m_w]; dict.len()],
            dict_im: vec![vec![0.0_f64; m_w]; dict.len()],
            s2_bands: Vec::new(),
            prev_s2: Vec::new(),
            res_w: Vec::with_capacity(m_w),
            ortho_basis: Vec::with_capacity(MAX_KAUTZ_POLES),
        })
        .collect();

    for &(tau, s_start, s_end) in &SLOTS {
        if tier_groups(ALL_TIERS[tau]) > g_cnt || model.active_tier_weights[tau] <= 0.0 {
            continue;
        }
        for ch in 0..active_ch {
            let _ = solve_tier_output_k_chan::<B>(
                model,
                tau,
                ch,
                s_start,
                &ds_omp,
                &trajs[ch],
                &stems_pre[ch],
            );
        }
        let tier = ALL_TIERS[tau];
        let c_t = tier_block_lanes_array(tier);
        let f_t = tier_fir_segments(tier) * 16;
        let cp_cur = ClampedCoreParams::from_model(model);
        let r = model.readouts[tau].clone();
        for ch in 0..active_ch {
            let ctx = ChanPoolUpdateCtx {
                ch,
                c_t,
                f_t,
                s_start,
                m_w,
                pool_meta: &pool_meta,
                sq_bw: &sq_bw_all[ch],
                bank: &ds_omp.bank,
                dict_pre: &dict_pre,
                prev_poles: &model.kautz_poles[..s_start],
            };
            update_channel_for_tier_slot::<B>(
                &ctx,
                &r,
                &cp_cur.kautz_pre,
                &ds_omp,
                &trajs[ch],
                &mut chan_states[ch],
            );
        }

        for slot in s_start..s_end {
            let mut best_di = 0;
            let mut best_score = -1.0_f64;
            for (di, cand) in dict.iter().enumerate() {
                if (0..slot).any(|p| kautz_pole_overlap(*cand, model.kautz_poles[p]) > 0.85) {
                    continue;
                }
                let mut sc = 0.0_f64;
                for state in &chan_states {
                    sc += B::score_pole_pair(&state.res_w, &state.dict_re[di], &state.dict_im[di]);
                }
                if sc.is_finite() && sc > best_score {
                    best_score = sc;
                    best_di = di;
                }
            }
            let r_ctx = RefineCtx {
                active_ch,
                m_w,
                pool_meta: &pool_meta,
                sq_bw: &sq_bw_all,
                channels: &chan_states,
            };
            let refined =
                refine_selected_pole_coord::<B>(dict[best_di], &model.kautz_poles[..slot], &r_ctx);
            model.kautz_poles[slot] = refined.pole;
            for (state, (bre, bim)) in chan_states
                .iter_mut()
                .zip(refined.basis_re_ch.into_iter().zip(refined.basis_im_ch))
            {
                deflate_residual_pair::<B>(&mut state.res_w, &bre, &bim);
                state.ortho_basis.push((bre, bim));
            }
        }
    }

    let uniform_rate = !model.os2x
        || ALL_TIERS.iter().enumerate().all(|(tau, &tier)| {
            tier_groups(tier) > g_cnt
                || model.active_tier_weights[tau] <= 0.0
                || model.tier_os2x[tau]
        });
    if uniform_rate {
        for ch in 0..active_ch {
            for (tau, &tier) in ALL_TIERS.iter().enumerate() {
                if tier_groups(tier) <= g_cnt && model.active_tier_weights[tau] > 0.0 {
                    solve_tier_output_chan::<B>(
                        model,
                        tau,
                        ch,
                        &ds_omp,
                        &trajs[ch],
                        &stems_pre[ch],
                    );
                }
            }
        }
        normalize_pre_cab_gauge(&mut model.readouts, active_ch);
        model.mirror_mono();
    } else {
        solve_matryoshka_varpro_cholesky_with_backend::<B>(model, &ds_omp, 0);
    }
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
    use dnam_engine::stages::V3MatryoshkaModel;

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

    fn run_biquad_resonator(inp: &[f32], freq_hz: f64, r: f64, sr_hz: f64) -> Vec<f32> {
        let th = 2.0 * PI * freq_hz / sr_hz;
        let a1 = -2.0 * r * th.cos();
        let a2 = r * r;
        let bg = (1.0 - a2).sqrt();
        let (mut y1, mut y2) = (0.0_f64, 0.0_f64);
        inp.iter()
            .map(|&x| {
                let y = bg * f64::from(x) - a1 * y1 - a2 * y2;
                y2 = y1;
                y1 = y;
                y as f32
            })
            .collect()
    }

    #[test]
    fn test_omp_pole_selection_guards_multi_stem_and_collision_repair() {
        let dict = build_pole_dictionary(48_000.0);
        assert_eq!(dict.len(), 10 + 64 * 6);

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

    #[test]
    fn test_omp_stereo_asymmetric_resonances_and_offgrid_refinement() {
        let n = 4_096_usize;
        let sr_hz = 48_000.0_f64;
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut in_l = Vec::with_capacity(n);
        let mut in_r = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            in_l.push((((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0) as f32 * 0.35);
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            in_r.push((((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0) as f32 * 0.35);
        }
        let f_l = 482.0_f64;
        let f_r = 1_310.0_f64;
        let tgt_l = run_biquad_resonator(&in_l, f_l, 0.974, sr_hz);
        let tgt_r = run_biquad_resonator(&in_r, f_r, 0.968, sr_hz);
        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l,
                in_r,
                tgt_l,
                tgt_r,
                warmup: 64,
            }],
            sr_hz,
        );

        let cold = V3MatryoshkaModel::dual_rate_default(false);
        let mut model = V3MatryoshkaOptF64::from_matryoshka_cold(&cold);
        select_omp_kautz_poles_matryoshka(&mut model, &ds);

        let th_l = 2.0 * PI * f_l / sr_hz;
        let th_r = 2.0 * PI * f_r / sr_hz;
        let best_l = model.kautz_poles[..8]
            .iter()
            .map(|&p| kautz_pole_overlap(p, (0.974, th_l)))
            .fold(0.0_f64, f64::max);
        let best_r = model.kautz_poles[..8]
            .iter()
            .map(|&p| kautz_pole_overlap(p, (0.968, th_r)))
            .fold(0.0_f64, f64::max);
        assert!(best_l > 0.75, "OMP missed left resonance: overlap={best_l}");
        assert!(
            best_r > 0.75,
            "OMP missed right resonance: overlap={best_r}"
        );
        assert_ne!(
            model.readouts[3].kautz_c_l[0], model.readouts[3].kautz_c_r[0],
            "stereo L/R Kautz readouts should be solved independently"
        );
    }
}
