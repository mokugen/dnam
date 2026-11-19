#![forbid(unsafe_code)]
//! Exact 5-scale 32-probe `MR-IIR` loss & reverse-time IIR state adjoint
//! (`incoherent_rig/mr_iir_adj.rs`), matching [`crate::metrics::compute_mr_iir_loss`].

use std::f64::consts::PI;

use crate::solver::v3_varpro::primitives::par_map_indexed;

const SCALES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const NUM_PROBES: usize = 32;
const PROBE_GROUPS: usize = NUM_PROBES / 4;
const SUB_CHUNK: usize = 64;

#[derive(Clone)]
struct ScaleAdjTarget {
    norm: f64,
    rm2: f64,
    inv_win_len: f64,
    trc: [[f64; 4]; PROBE_GROUPS],
    checkpoints: Vec<usize>,
    win_cp_idx: Vec<(usize, usize)>,
    win_rms_t: [Vec<[f64; 4]>; PROBE_GROUPS],
    sc_den_sqrt: f64,
    cell_count: f64,
}

#[derive(Clone)]
pub(super) struct MrIirTargetCacheAdj {
    warmup: usize,
    t_sel: usize,
    scales: Vec<ScaleAdjTarget>,
}

fn build_win_bounds(m_scale: usize, t_sel: usize) -> (usize, Vec<(usize, usize)>) {
    if t_sel < m_scale {
        return (t_sel, vec![(0, t_sel)]);
    }
    let hop = m_scale / 4;
    let chunks_per_win = m_scale / SUB_CHUNK;
    let chunks_per_hop = hop / SUB_CHUNK;
    let mut n_full = 0_usize;
    while (n_full * chunks_per_hop + chunks_per_win) * SUB_CHUNK <= t_sel {
        n_full += 1;
    }
    let end_start = n_full * hop;
    let has_tail = (t_sel - m_scale) + hop != end_start;
    let mut bounds = Vec::with_capacity(n_full + usize::from(has_tail));
    for w in 0..n_full {
        let s = w * hop;
        bounds.push((s, s + m_scale));
    }
    if has_tail {
        bounds.push((t_sel - m_scale, t_sel));
    }
    (m_scale, bounds)
}

fn build_trc(m_scale: usize, sr_hz: f64, f_max: f64) -> (f64, f64, [[f64; 4]; PROBE_GROUPS]) {
    let r_m = (-PI / (m_scale as f64)).exp();
    let rm2 = r_m * r_m;
    let norm = 0.5 * (1.0 - rm2);
    let mut trc = [[0.0_f64; 4]; PROBE_GROUPS];
    for (g, grp) in trc.iter_mut().enumerate() {
        for (l, dst) in grp.iter_mut().enumerate() {
            let p = g * 4 + l;
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            *dst = 2.0 * r_m * (2.0 * PI * f_k / sr_hz).cos();
        }
    }
    (norm, rm2, trc)
}

fn build_checkpoints_and_indices(
    t_sel: usize,
    win_bounds: &[(usize, usize)],
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut checkpoints = Vec::with_capacity(win_bounds.len() * 2 + 2);
    checkpoints.push(0);
    checkpoints.push(t_sel);
    for &(ws, we) in win_bounds {
        checkpoints.push(ws);
        checkpoints.push(we);
    }
    checkpoints.sort_unstable();
    checkpoints.dedup();
    let win_cp_idx = win_bounds
        .iter()
        .map(|&(ws, we)| {
            let cp_ws = checkpoints.binary_search(&ws).expect("checkpoint present");
            let cp_we = checkpoints.binary_search(&we).expect("checkpoint present");
            (cp_ws, cp_we)
        })
        .collect();
    (checkpoints, win_cp_idx)
}

fn step_group_checkpoints_sq(
    sig: &[f64],
    warmup: usize,
    checkpoints: &[usize],
    norm: f64,
    rm2: f64,
    trc: [f64; 4],
    cp_ps: &mut [[f64; 4]],
) {
    let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
    let (mut y1, mut y2) = ([0.0_f64; 4], [0.0_f64; 4]);
    for &x in &sig[..warmup] {
        let nd = norm * (x - x2);
        for l in 0..4 {
            let b = nd + trc[l] * y1[l] - rm2 * y2[l];
            y2[l] = y1[l];
            y1[l] = b;
        }
        x2 = x1;
        x1 = x;
    }
    let mut acc = [0.0_f64; 4];
    cp_ps[0] = acc;
    let mut cur = warmup;
    for (c_idx, &cp_rel) in checkpoints[1..].iter().enumerate() {
        let end = warmup + cp_rel;
        for &x in &sig[cur..end] {
            let nd = norm * (x - x2);
            for l in 0..4 {
                let b = nd + trc[l] * y1[l] - rm2 * y2[l];
                y2[l] = y1[l];
                y1[l] = b;
                acc[l] += b * b;
            }
            x2 = x1;
            x1 = x;
        }
        cp_ps[c_idx + 1] = acc;
        cur = end;
    }
}

impl MrIirTargetCacheAdj {
    pub(super) fn new(target: &[f64], warmup: usize, sr_hz: f64) -> Option<Self> {
        let f_max = (0.45 * sr_hz).min(20_000.0);
        if target.len() <= warmup + 16 || f_max <= 20.0 || sr_hz < 8_000.0 {
            return None;
        }
        let t_sel = target.len() - warmup;
        let mut cp_ps = Vec::new();
        let mut scales = Vec::with_capacity(SCALES.len());
        for &m_scale in &SCALES {
            let (norm, rm2, trc) = build_trc(m_scale, sr_hz, f_max);
            let (win_len, win_bounds) = build_win_bounds(m_scale, t_sel);
            let (checkpoints, win_cp_idx) = build_checkpoints_and_indices(t_sel, &win_bounds);
            cp_ps.resize(checkpoints.len(), [0.0; 4]);
            let inv_win_len = 1.0 / (win_len as f64);
            let mut sc_den_sq = 0.0_f64;
            let win_rms_t = std::array::from_fn(|g| {
                step_group_checkpoints_sq(
                    target,
                    warmup,
                    &checkpoints,
                    norm,
                    rm2,
                    trc[g],
                    &mut cp_ps,
                );
                let mut rows = Vec::with_capacity(win_cp_idx.len());
                for &(cp_ws, cp_we) in &win_cp_idx {
                    let mut r4 = [0.0_f64; 4];
                    for l in 0..4 {
                        let e = (cp_ps[cp_we][l] - cp_ps[cp_ws][l]).max(0.0);
                        let rms = (e * inv_win_len).sqrt();
                        r4[l] = rms;
                        sc_den_sq += rms * rms;
                    }
                    rows.push(r4);
                }
                rows
            });
            let cell_count = (PROBE_GROUPS * win_cp_idx.len() * 4) as f64;
            scales.push(ScaleAdjTarget {
                norm,
                rm2,
                inv_win_len,
                trc,
                checkpoints,
                win_cp_idx,
                win_rms_t,
                sc_den_sqrt: sc_den_sq.sqrt(),
                cell_count,
            });
        }
        Some(Self {
            warmup,
            t_sel,
            scales,
        })
    }
}

pub(super) struct MrIirFloorSpec<'a> {
    cache: &'a MrIirTargetCacheAdj,
    pull_weight: f64,
    bounds: Option<(f64, f64)>,
}

pub(super) struct MrIirTargetCtx {
    l: MrIirTargetCacheAdj,
    r: Option<MrIirTargetCacheAdj>,
}

impl MrIirTargetCtx {
    pub(super) fn new(
        tl: &[f64],
        tr: &[f64],
        warmup: usize,
        sr_hz: f64,
        mono_out: bool,
    ) -> Option<Self> {
        let l = MrIirTargetCacheAdj::new(tl, warmup, sr_hz)?;
        let r = if mono_out {
            None
        } else {
            MrIirTargetCacheAdj::new(tr, warmup, sr_hz)
        };
        Some(Self { l, r })
    }

    pub(super) fn make_specs(
        &self,
        pull_weight: f64,
        bounds_l: Option<(f64, f64)>,
        bounds_r: Option<(f64, f64)>,
    ) -> (Option<MrIirFloorSpec<'_>>, Option<MrIirFloorSpec<'_>>) {
        let sl = Some(MrIirFloorSpec {
            cache: &self.l,
            pull_weight,
            bounds: bounds_l,
        });
        let sr = self.r.as_ref().map(|rc| MrIirFloorSpec {
            cache: rc,
            pull_weight,
            bounds: bounds_r,
        });
        (sl, sr)
    }
}

#[derive(Default)]
struct MrIirBackwardBuf {
    cp_w_diff: Vec<[f64; 4]>,
    b_bar: Vec<[f64; 4]>,
}

#[derive(Default)]
pub(super) struct MrIirScratch {
    cp_ps: Vec<[f64; 4]>,
    d_sc: Vec<[f64; 4]>,
    d_mag: Vec<[f64; 4]>,
    d_mr_d_em: [Vec<[f64; 4]>; SCALES.len()],
    bw: MrIirBackwardBuf,
    group_dys: [Vec<f64>; PROBE_GROUPS],
    group_bws: [MrIirBackwardBuf; PROBE_GROUPS],
}

fn backprop_group(
    y_hat: &[f64],
    warmup: usize,
    scale: &ScaleAdjTarget,
    trc: [f64; 4],
    d_win: &[[f64; 4]],
    bw: &mut MrIirBackwardBuf,
    dy: &mut [f64],
) {
    let n = y_hat.len();
    let t_sel = n - warmup;
    let n_cp = scale.checkpoints.len();
    bw.cp_w_diff.resize(n_cp, [0.0; 4]);
    bw.cp_w_diff.fill([0.0; 4]);
    for (&(cp_ws, cp_we), &dm) in scale.win_cp_idx.iter().zip(d_win) {
        for l in 0..4 {
            bw.cp_w_diff[cp_ws][l] += dm[l];
            bw.cp_w_diff[cp_we][l] -= dm[l];
        }
    }

    bw.b_bar.resize(t_sel, [0.0; 4]);
    let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
    let (mut y1, mut y2) = ([0.0_f64; 4], [0.0_f64; 4]);
    for &x in &y_hat[..warmup] {
        let nd = scale.norm * (x - x2);
        for l in 0..4 {
            let b = nd + trc[l] * y1[l] - scale.rm2 * y2[l];
            y2[l] = y1[l];
            y1[l] = b;
        }
        x2 = x1;
        x1 = x;
    }

    let mut w_cur = [0.0_f64; 4];
    for c in 0..n_cp.saturating_sub(1) {
        let wd = bw.cp_w_diff[c];
        for l in 0..4 {
            w_cur[l] += wd[l];
        }
        let rel_start = scale.checkpoints[c];
        let rel_end = scale.checkpoints[c + 1];
        for (k, &x) in y_hat[warmup + rel_start..warmup + rel_end]
            .iter()
            .enumerate()
        {
            let nd = scale.norm * (x - x2);
            let bar = &mut bw.b_bar[rel_start + k];
            for l in 0..4 {
                let b = nd + trc[l] * y1[l] - scale.rm2 * y2[l];
                y2[l] = y1[l];
                y1[l] = b;
                bar[l] = w_cur[l] * b;
            }
            x2 = x1;
            x1 = x;
        }
    }

    let (mut lam1, mut lam2) = ([0.0_f64; 4], [0.0_f64; 4]);
    for rel in (0..t_sel).rev() {
        let bb = bw.b_bar[rel];
        let mut s_diff = 0.0_f64;
        for l in 0..4 {
            let lam0 = bb[l] + trc[l] * lam1[l] - scale.rm2 * lam2[l];
            s_diff += lam0 - lam2[l];
            lam2[l] = lam1[l];
            lam1[l] = lam0;
        }
        dy[warmup + rel] += scale.norm * s_diff;
    }
    for dy_i in dy[..warmup].iter_mut().rev() {
        let mut s_diff = 0.0_f64;
        for l in 0..4 {
            let lam0 = trc[l] * lam1[l] - scale.rm2 * lam2[l];
            s_diff += lam0 - lam2[l];
            lam2[l] = lam1[l];
            lam1[l] = lam0;
        }
        *dy_i += scale.norm * s_diff;
    }
}

fn accumulate_group_cells(
    scale: &ScaleAdjTarget,
    g: usize,
    cp_ps: &[[f64; 4]],
    mut d_sc_out: Option<&mut [[f64; 4]]>,
    mut d_mag_out: Option<&mut [[f64; 4]]>,
) -> (f64, f64) {
    let mut sc_num_sq = 0.0_f64;
    let mut mag_sum = 0.0_f64;
    for (w, &(cp_ws, cp_we)) in scale.win_cp_idx.iter().enumerate() {
        let rt4 = scale.win_rms_t[g][w];
        for l in 0..4 {
            let a_t = rt4[l];
            let em = (cp_ps[cp_we][l] - cp_ps[cp_ws][l]).max(0.0);
            let rms_m = ((em + 1.0e-30) * scale.inv_win_len).sqrt();
            let diff = rms_m - a_t;
            sc_num_sq += diff * diff;
            let (dm_val, dm_grad) = if a_t > 1.0e-18 {
                let u = (rms_m / a_t).ln();
                let r_u = (u * u + 1.0e-6).sqrt();
                (
                    r_u - 1.0e-3,
                    (u / r_u) * scale.inv_win_len / (2.0 * rms_m * rms_m * scale.cell_count),
                )
            } else {
                (0.0, 0.0)
            };
            mag_sum += dm_val;
            if let (Some(ds), Some(dm)) = (d_sc_out.as_deref_mut(), d_mag_out.as_deref_mut()) {
                ds[w][l] = diff * scale.inv_win_len / (2.0 * rms_m);
                dm[w][l] = dm_grad;
            }
        }
    }
    (sc_num_sq, mag_sum)
}

fn finish_scale_forward(
    scale: &ScaleAdjTarget,
    inv_scales: f64,
    sc_num_sq: f64,
    mag_sum: f64,
    need_grad: bool,
    d_sc: &[[f64; 4]],
    d_mag: &[[f64; 4]],
    d_mr_d_em: &mut Vec<[f64; 4]>,
) -> f64 {
    let sc_root = (sc_num_sq + 1.0e-30).sqrt();
    let l_sc = if scale.sc_den_sqrt > 1.0e-30 {
        sc_root / scale.sc_den_sqrt
    } else {
        0.0
    };
    let l_mag = mag_sum / scale.cell_count;
    if need_grad {
        let total_rows = PROBE_GROUPS * scale.win_cp_idx.len();
        d_mr_d_em.resize(total_rows, [0.0; 4]);
        let inv_sc_den = if scale.sc_den_sqrt > 1.0e-30 {
            1.0 / (scale.sc_den_sqrt * sc_root)
        } else {
            0.0
        };
        for idx in 0..total_rows {
            for l in 0..4 {
                d_mr_d_em[idx][l] = inv_scales * (d_sc[idx][l] * inv_sc_den + d_mag[idx][l]);
            }
        }
    }
    inv_scales * (l_sc + l_mag)
}

pub(super) fn eval_mr_iir_and_adjoint(
    spec: &MrIirFloorSpec<'_>,
    y_hat: &[f64],
    dy: Option<&mut [f64]>,
    sc: &mut MrIirScratch,
) -> f64 {
    let cache = spec.cache;
    let n = cache.warmup + cache.t_sel;
    if y_hat.len() < n {
        return 0.0;
    }
    let y_slice = &y_hat[..n];
    let inv_scales = 1.0 / (cache.scales.len() as f64);
    let need_grad = dy.is_some();
    let mut mr_val = 0.0_f64;

    if n >= 8_192 {
        let outs = par_map_indexed(PROBE_GROUPS, |g| {
            let mut per_scale = Vec::with_capacity(cache.scales.len());
            for scale in &cache.scales {
                let m_win = scale.win_cp_idx.len();
                let mut cp_ps = vec![[0.0_f64; 4]; scale.checkpoints.len()];
                step_group_checkpoints_sq(
                    y_slice,
                    cache.warmup,
                    &scale.checkpoints,
                    scale.norm,
                    scale.rm2,
                    scale.trc[g],
                    &mut cp_ps,
                );
                let mut grad_pair =
                    need_grad.then(|| (vec![[0.0_f64; 4]; m_win], vec![[0.0_f64; 4]; m_win]));
                let (ds_opt, dm_opt) = match grad_pair.as_mut() {
                    Some((ds, dm)) => (Some(ds.as_mut_slice()), Some(dm.as_mut_slice())),
                    None => (None, None),
                };
                let (sc_sq_g, mag_g) = accumulate_group_cells(scale, g, &cp_ps, ds_opt, dm_opt);
                per_scale.push((sc_sq_g, mag_g, grad_pair));
            }
            per_scale
        });
        for (si, scale) in cache.scales.iter().enumerate() {
            let m_win = scale.win_cp_idx.len();
            let total_rows = PROBE_GROUPS * m_win;
            if need_grad {
                sc.d_sc.resize(total_rows, [0.0; 4]);
                sc.d_mag.resize(total_rows, [0.0; 4]);
            }
            let mut sc_num_sq = 0.0_f64;
            let mut mag_sum = 0.0_f64;
            for g in 0..PROBE_GROUPS {
                let (sc_sq_g, mag_g, ref gp) = outs[g][si];
                sc_num_sq += sc_sq_g;
                mag_sum += mag_g;
                if let Some((ds, dm)) = gp {
                    let dst_range = g * m_win..(g + 1) * m_win;
                    sc.d_sc[dst_range.clone()].copy_from_slice(ds);
                    sc.d_mag[dst_range].copy_from_slice(dm);
                }
            }
            mr_val += finish_scale_forward(
                scale,
                inv_scales,
                sc_num_sq,
                mag_sum,
                need_grad,
                &sc.d_sc,
                &sc.d_mag,
                &mut sc.d_mr_d_em[si],
            );
        }
    } else {
        for (si, scale) in cache.scales.iter().enumerate() {
            let m_win = scale.win_cp_idx.len();
            let total_rows = PROBE_GROUPS * m_win;
            sc.cp_ps.resize(scale.checkpoints.len(), [0.0; 4]);
            if need_grad {
                sc.d_sc.resize(total_rows, [0.0; 4]);
                sc.d_mag.resize(total_rows, [0.0; 4]);
            }
            let mut sc_num_sq = 0.0_f64;
            let mut mag_sum = 0.0_f64;
            for g in 0..PROBE_GROUPS {
                step_group_checkpoints_sq(
                    y_slice,
                    cache.warmup,
                    &scale.checkpoints,
                    scale.norm,
                    scale.rm2,
                    scale.trc[g],
                    &mut sc.cp_ps,
                );
                let row_range = g * m_win..(g + 1) * m_win;
                let (ds_opt, dm_opt) = if need_grad {
                    (
                        Some(&mut sc.d_sc[row_range.clone()]),
                        Some(&mut sc.d_mag[row_range]),
                    )
                } else {
                    (None, None)
                };
                let (sc_sq_g, mag_g) = accumulate_group_cells(scale, g, &sc.cp_ps, ds_opt, dm_opt);
                sc_num_sq += sc_sq_g;
                mag_sum += mag_g;
            }
            mr_val += finish_scale_forward(
                scale,
                inv_scales,
                sc_num_sq,
                mag_sum,
                need_grad,
                &sc.d_sc,
                &sc.d_mag,
                &mut sc.d_mr_d_em[si],
            );
        }
    }

    let mut loss = spec.pull_weight * mr_val;
    let mut d_mr = spec.pull_weight;
    if let Some((min_mr, max_mr)) = spec.bounds {
        if mr_val > max_mr {
            let ex = mr_val - max_mr;
            loss += 40.0 * ex * ex;
            d_mr += 80.0 * ex;
        } else if mr_val < min_mr {
            let def = min_mr - mr_val;
            loss += 2_000.0 * def * def;
            d_mr -= 4_000.0 * def;
        }
    }

    if let Some(dy_out) = dy
        && d_mr != 0.0
    {
        for si in 0..cache.scales.len() {
            for row in &mut sc.d_mr_d_em[si] {
                for v in row {
                    *v *= 2.0 * d_mr;
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let run_parallel_bw = n >= 8_192 && std::env::var_os("LLVM_PROFILE_FILE").is_none();
        #[cfg(target_arch = "wasm32")]
        let run_parallel_bw = false;
        if run_parallel_bw {
            #[cfg(not(target_arch = "wasm32"))]
            {
                for dy_loc in &mut sc.group_dys {
                    dy_loc.resize(n, 0.0);
                    dy_loc[..n].fill(0.0);
                }
                let scales = &cache.scales;
                let d_mr_d_em = &sc.d_mr_d_em;
                std::thread::scope(|s| {
                    for (g, (dy_loc, bw_loc)) in sc
                        .group_dys
                        .iter_mut()
                        .zip(sc.group_bws.iter_mut())
                        .enumerate()
                    {
                        s.spawn(move || {
                            for (si, scale) in scales.iter().enumerate() {
                                let m_win = scale.win_cp_idx.len();
                                let d_win = &d_mr_d_em[si][g * m_win..(g + 1) * m_win];
                                backprop_group(
                                    y_slice,
                                    cache.warmup,
                                    scale,
                                    scale.trc[g],
                                    d_win,
                                    bw_loc,
                                    dy_loc,
                                );
                            }
                        });
                    }
                });
                for dy_loc in &sc.group_dys {
                    for (dst, &v) in dy_out[..n].iter_mut().zip(&dy_loc[..n]) {
                        *dst += v;
                    }
                }
            }
        } else {
            for (si, scale) in cache.scales.iter().enumerate() {
                let m_win = scale.win_cp_idx.len();
                for g in 0..PROBE_GROUPS {
                    let d_win = &sc.d_mr_d_em[si][g * m_win..(g + 1) * m_win];
                    backprop_group(
                        y_slice,
                        cache.warmup,
                        scale,
                        scale.trc[g],
                        d_win,
                        &mut sc.bw,
                        &mut dy_out[..n],
                    );
                }
            }
        }
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::companion::MrIirTargetCache;
    use crate::metrics::compute_mr_iir_loss;
    use std::f64::consts::TAU;

    #[test]
    fn test_mr_iir_loss_parity_and_reverse_adjoint_matches_fd() {
        for n in [6_400_usize, 9_600] {
            let warmup = 384;
            let sr = 48_000.0;
            let tgt: Vec<f64> = (0..n)
                .map(|i| {
                    let t = (i as f64) / sr;
                    0.28 * (TAU * 190.0 * t).sin() + 0.14 * (TAU * 1_450.0 * t + 0.4).cos()
                })
                .collect();
            let mdl: Vec<f64> = (0..n)
                .map(|i| {
                    let t = (i as f64) / sr;
                    0.24 * (TAU * 190.0 * t + 0.25).sin() + 0.18 * (TAU * 1_450.0 * t - 0.1).cos()
                })
                .collect();

            let ctx0 = MrIirTargetCtx::new(&tgt, &tgt, 0, sr, true).unwrap();
            let mut sc = MrIirScratch::default();
            let (sp0, _) = ctx0.make_specs(1.0, None, None);
            let mr0 = eval_mr_iir_and_adjoint(sp0.as_ref().unwrap(), &mdl, None, &mut sc);
            let pub_ref = compute_mr_iir_loss(&tgt, &mdl, None, sr, 0);
            assert!(
                (mr0 - pub_ref).abs() < 2.0e-3,
                "n={n} mr0={mr0} pub_ref={pub_ref}"
            );

            let ref_mr = MrIirTargetCache::new(&tgt, sr, warmup)
                .unwrap()
                .eval_model(&mdl);
            let ctx = MrIirTargetCtx::new(&tgt, &tgt, warmup, sr, true).unwrap();
            let (sp_pull, _) = ctx.make_specs(1.0, None, None);
            let mr_val = eval_mr_iir_and_adjoint(sp_pull.as_ref().unwrap(), &mdl, None, &mut sc);
            assert!(
                (mr_val - ref_mr).abs() < 2.0e-3,
                "n={n} mr_val={mr_val} ref={ref_mr}"
            );

            for (tag, bounds) in [
                ("pull", None),
                ("upper", Some((0.0, mr_val * 0.5))),
                ("lower", Some((mr_val * 1.5, mr_val * 2.0))),
            ] {
                let (sp, _) = ctx.make_specs(1.2, bounds, None);
                let spec = sp.as_ref().unwrap();
                let mut dy = vec![0.0_f64; n];
                let l0 = eval_mr_iir_and_adjoint(spec, &mdl, Some(&mut dy), &mut sc);
                assert!(l0.is_finite());
                let h = 1.0e-7_f64;
                let pk = dy.iter().map(|v| v.abs()).fold(1.0e-8, f64::max);
                let (mut yp, mut ym, mut d_ana) = (mdl.clone(), mdl.clone(), 0.0_f64);
                for i in 0..n {
                    let dir = dy[i] / pk + 0.02 * ((i as f64) * 0.13).sin();
                    d_ana += dy[i] * dir;
                    yp[i] += h * dir;
                    ym[i] -= h * dir;
                }
                let lp = eval_mr_iir_and_adjoint(spec, &yp, None, &mut sc);
                let lm = eval_mr_iir_and_adjoint(spec, &ym, None, &mut sc);
                let d_fd = (lp - lm) / (2.0 * h);
                let rel = (d_ana - d_fd).abs() / d_ana.abs().max(d_fd.abs()).max(1.0e-10);
                assert!(rel < 1.0e-5, "n={n} {tag} rel={rel} ana={d_ana} fd={d_fd}");
            }
        }
    }
}
