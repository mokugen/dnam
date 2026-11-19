#![forbid(unsafe_code)]
//! 5-Scale Floor-Clamped Multi-Resolution IIR Resonator Bank (`MR-IIR`, `r[qual.companion.mr-iir]`).

use std::f64::consts::PI;

use super::MAX_METRIC_SAMPLE_RATE_HZ;

const SCALES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const NUM_PROBES: usize = 32;
const PROBE_GROUPS: usize = NUM_PROBES / 4;
const SUB_CHUNK: usize = 64;

#[inline]
fn has_tiny_nonzero(sig: &[f64]) -> bool {
    sig.iter().any(|&v| v != 0.0 && v.abs() < 1.0e-100)
}

#[inline]
fn probe_norm_and_trc(
    m_scale: usize,
    sample_rate_hz: f64,
    f_max: f64,
) -> (f64, f64, [[f64; 4]; PROBE_GROUPS]) {
    let r_m = (-PI / (m_scale as f64)).exp();
    let r_m_sq = r_m * r_m;
    let norm = 0.5 * (1.0 - r_m_sq);
    let mut trc = [[0.0_f64; 4]; PROBE_GROUPS];
    for (g, grp) in trc.iter_mut().enumerate() {
        for (l, slot) in grp.iter_mut().enumerate() {
            let p = g * 4 + l;
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            let omega = 2.0 * PI * f_k / sample_rate_hz;
            *slot = 2.0 * r_m * omega.cos();
        }
    }
    (norm, r_m_sq, trc)
}

#[derive(Clone, Debug)]
struct ScaleGeometry {
    m_scale: usize,
    norm: f64,
    r_m_sq: f64,
    trc: [[f64; 4]; PROBE_GROUPS],
    chunks_per_win: usize,
    chunks_per_hop: usize,
    num_full_windows: usize,
    has_unaligned_tail: bool,
    tail_len: usize,
    tail_abs_start: usize,
}

impl ScaleGeometry {
    fn new(
        m_scale: usize,
        sample_rate_hz: f64,
        f_max: f64,
        warmup_samples: usize,
        t_sel: usize,
    ) -> Self {
        let (norm, r_m_sq, trc) = probe_norm_and_trc(m_scale, sample_rate_hz, f_max);
        let hop = m_scale / 4;
        let chunks_per_win = m_scale / SUB_CHUNK;
        let chunks_per_hop = hop / SUB_CHUNK;
        let mut num_full_windows = 0usize;
        if t_sel >= m_scale {
            while (num_full_windows * chunks_per_hop + chunks_per_win) * SUB_CHUNK <= t_sel {
                num_full_windows += 1;
            }
        }
        let end_start = num_full_windows * chunks_per_hop * SUB_CHUNK;
        let has_unaligned_tail = if t_sel < m_scale {
            true
        } else {
            let last_start = t_sel - m_scale;
            last_start + hop != end_start
        };
        let tail_len = t_sel.min(m_scale);
        let tail_abs_start = warmup_samples + t_sel.saturating_sub(m_scale);
        Self {
            m_scale,
            norm,
            r_m_sq,
            trc,
            chunks_per_win,
            chunks_per_hop,
            num_full_windows,
            has_unaligned_tail,
            tail_len,
            tail_abs_start,
        }
    }
}

#[derive(Clone, Debug)]
struct CachedScaleTarget {
    geom: ScaleGeometry,
    // Packed window target stats for each probe group [g][win]: `rms_t` and `ln_rms_t`.
    win_rms_t: Vec<[f64; 4]>,
    win_ln_rms_t: Vec<[f64; 4]>,
    sc_den_sq: f64,
    scale_nz_t: bool,
    target_invalid: bool,
}

#[derive(Default)]
struct SignalGroupWorkspace {
    chunk_e: Vec<[f64; 4]>,
    chunk_nz: Vec<[bool; 4]>,
    tail_buf: Vec<[f64; 4]>,
}

impl SignalGroupWorkspace {
    fn new(num_full_chunks: usize) -> Self {
        Self {
            chunk_e: vec![[0.0_f64; 4]; num_full_chunks],
            chunk_nz: vec![[false; 4]; num_full_chunks],
            tail_buf: vec![[0.0_f64; 4]; 4096],
        }
    }
}

/// Steps 4 bandpass probes in parallel for a signal with no sub-`1e-100` samples.
fn step_signal_group(
    sig: &[f64],
    warmup_samples: usize,
    geom: &ScaleGeometry,
    trc: [f64; 4],
    ws: &mut SignalGroupWorkspace,
) {
    let norm = geom.norm;
    let r_m_sq = geom.r_m_sq;
    let t_sel = sig.len() - warmup_samples;
    let num_full_chunks = t_sel / SUB_CHUNK;
    let full_end = warmup_samples + num_full_chunks * SUB_CHUNK;

    let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
    let mut y1 = [0.0_f64; 4];
    let mut y2 = [0.0_f64; 4];

    for &x in &sig[..warmup_samples] {
        let drive = x - x2;
        let nd = norm * drive;
        for l in 0..4 {
            let mut b = nd + trc[l] * y1[l] - r_m_sq * y2[l];
            if drive == 0.0 && b.abs() < 1.0e-140 && y1[l].abs() < 1.0e-140 {
                b = 0.0;
            }
            y2[l] = y1[l];
            y1[l] = b;
        }
        x2 = x1;
        x1 = x;
    }

    let tail_abs_start = geom.tail_abs_start;
    let pre_tail_chunks = if geom.has_unaligned_tail {
        (tail_abs_start.saturating_sub(warmup_samples)) / SUB_CHUNK
    } else {
        num_full_chunks
    };
    let tail_split_start = warmup_samples + pre_tail_chunks * SUB_CHUNK;

    for c in 0..num_full_chunks {
        let c_start = warmup_samples + c * SUB_CHUNK;
        let mut ce = [0.0_f64; 4];
        let mut cnz = [false; 4];
        if c_start + SUB_CHUNK <= tail_split_start || !geom.has_unaligned_tail {
            for &x in &sig[c_start..c_start + SUB_CHUNK] {
                let drive = x - x2;
                let nd = norm * drive;
                for l in 0..4 {
                    let mut b = nd + trc[l] * y1[l] - r_m_sq * y2[l];
                    if drive == 0.0 && b.abs() < 1.0e-140 && y1[l].abs() < 1.0e-140 {
                        b = 0.0;
                    }
                    y2[l] = y1[l];
                    y1[l] = b;
                    ce[l] += b * b;
                    cnz[l] |= b != 0.0;
                }
                x2 = x1;
                x1 = x;
            }
        } else {
            for idx in c_start..c_start + SUB_CHUNK {
                let x = sig[idx];
                let drive = x - x2;
                let nd = norm * drive;
                let mut bout = [0.0_f64; 4];
                for l in 0..4 {
                    let mut b = nd + trc[l] * y1[l] - r_m_sq * y2[l];
                    if drive == 0.0 && b.abs() < 1.0e-140 && y1[l].abs() < 1.0e-140 {
                        b = 0.0;
                    }
                    y2[l] = y1[l];
                    y1[l] = b;
                    ce[l] += b * b;
                    cnz[l] |= b != 0.0;
                    bout[l] = b;
                }
                if idx >= tail_abs_start {
                    ws.tail_buf[idx - tail_abs_start] = bout;
                }
                x2 = x1;
                x1 = x;
            }
        }
        ws.chunk_e[c] = ce;
        ws.chunk_nz[c] = cnz;
    }

    for (idx, &x) in sig.iter().enumerate().skip(full_end) {
        let drive = x - x2;
        let nd = norm * drive;
        let mut bout = [0.0_f64; 4];
        for l in 0..4 {
            let mut b = nd + trc[l] * y1[l] - r_m_sq * y2[l];
            if drive == 0.0 && b.abs() < 1.0e-140 && y1[l].abs() < 1.0e-140 {
                b = 0.0;
            }
            y2[l] = y1[l];
            y1[l] = b;
            bout[l] = b;
        }
        if geom.has_unaligned_tail && idx >= tail_abs_start {
            ws.tail_buf[idx - tail_abs_start] = bout;
        }
        x2 = x1;
        x1 = x;
    }
}

/// Reusable pre-evaluated `MR-IIR` target resonator cache (`noise = None`).
#[derive(Clone, Debug)]
pub(crate) struct MrIirTargetCache {
    target: Vec<f64>,
    sample_rate_hz: f64,
    warmup_samples: usize,
    num_full_chunks: usize,
    scales: [CachedScaleTarget; 5],
}

impl MrIirTargetCache {
    /// Constructs a cached target evaluator, returning `None` if preconditions fail or if
    /// `target` contains ultra-tiny nonzero samples (`< 1e-100`).
    #[must_use]
    pub(crate) fn new(target: &[f64], sample_rate_hz: f64, warmup_samples: usize) -> Option<Self> {
        if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
            || target.is_empty()
            || warmup_samples >= target.len()
            || !target.iter().all(|v| v.is_finite())
            || has_tiny_nonzero(target)
        {
            return None;
        }
        let f_max = (0.45 * sample_rate_hz).min(20_000.0);
        if f_max <= 20.0 {
            return None;
        }
        let t_sel = target.len() - warmup_samples;
        let num_full_chunks = t_sel / SUB_CHUNK;
        let mut ws = SignalGroupWorkspace::new(num_full_chunks);

        let scales = std::array::from_fn(|si| {
            let m_scale = SCALES[si];
            let geom = ScaleGeometry::new(m_scale, sample_rate_hz, f_max, warmup_samples, t_sel);
            let wins_per_group = if t_sel < m_scale {
                1
            } else {
                geom.num_full_windows + usize::from(geom.has_unaligned_tail)
            };
            let inv_sqrt_w = 1.0 / (geom.tail_len as f64).sqrt();
            let mut win_rms_t = Vec::with_capacity(PROBE_GROUPS * wins_per_group);
            let mut win_ln_rms_t = Vec::with_capacity(PROBE_GROUPS * wins_per_group);
            let mut sc_den_sq = 0.0_f64;
            let mut scale_nz_t = false;
            let mut target_invalid = false;

            for g in 0..PROBE_GROUPS {
                step_signal_group(target, warmup_samples, &geom, geom.trc[g], &mut ws);
                let mut record_win = |e4: [f64; 4], h4: [bool; 4]| {
                    let mut rms4 = [0.0_f64; 4];
                    let mut ln4 = [0.0_f64; 4];
                    for l in 0..4 {
                        if !e4[l].is_finite() {
                            target_invalid = true;
                            continue;
                        }
                        scale_nz_t |= h4[l];
                        let rms = e4[l].sqrt() * inv_sqrt_w;
                        if (h4[l] && (e4[l] == 0.0 || rms == 0.0)) || !rms.is_finite() {
                            target_invalid = true;
                        }
                        rms4[l] = rms;
                        if rms > 0.0 {
                            ln4[l] = rms.ln();
                            sc_den_sq += rms * rms;
                        }
                    }
                    win_rms_t.push(rms4);
                    win_ln_rms_t.push(ln4);
                };

                if t_sel < m_scale {
                    let mut et = [0.0_f64; 4];
                    let mut ht = [false; 4];
                    for &v4 in &ws.tail_buf[..geom.tail_len] {
                        for l in 0..4 {
                            et[l] += v4[l] * v4[l];
                            ht[l] |= v4[l] != 0.0;
                        }
                    }
                    record_win(et, ht);
                } else {
                    for w_idx in 0..geom.num_full_windows {
                        let c0 = w_idx * geom.chunks_per_hop;
                        let c_end = c0 + geom.chunks_per_win;
                        let mut et = [0.0_f64; 4];
                        let mut ht = [false; 4];
                        for c in c0..c_end {
                            for l in 0..4 {
                                et[l] += ws.chunk_e[c][l];
                                ht[l] |= ws.chunk_nz[c][l];
                            }
                        }
                        record_win(et, ht);
                    }
                    if geom.has_unaligned_tail {
                        let mut et = [0.0_f64; 4];
                        let mut ht = [false; 4];
                        for &v4 in &ws.tail_buf[..geom.tail_len] {
                            for l in 0..4 {
                                et[l] += v4[l] * v4[l];
                                ht[l] |= v4[l] != 0.0;
                            }
                        }
                        record_win(et, ht);
                    }
                }
            }

            CachedScaleTarget {
                geom,
                win_rms_t,
                win_ln_rms_t,
                sc_den_sq,
                scale_nz_t,
                target_invalid,
            }
        });

        Some(Self {
            target: target.to_vec(),
            sample_rate_hz,
            warmup_samples,
            num_full_chunks,
            scales,
        })
    }

    /// Evaluates `MR-IIR` loss for `model` against the cached `target`.
    #[must_use]
    pub(crate) fn eval_model(&self, model: &[f64]) -> f64 {
        if model.len() != self.target.len() || !model.iter().all(|v| v.is_finite()) {
            return f64::INFINITY;
        }
        if has_tiny_nonzero(model) {
            return compute_mr_iir_loss_general(
                &self.target,
                model,
                None,
                self.sample_rate_hz,
                self.warmup_samples,
            );
        }

        let t_sel = self.target.len() - self.warmup_samples;
        let mut ws = SignalGroupWorkspace::new(self.num_full_chunks);
        let mut total_loss = 0.0_f64;

        for cached in &self.scales {
            if cached.target_invalid {
                return f64::INFINITY;
            }
            let geom = &cached.geom;
            let inv_sqrt_w = 1.0 / (geom.tail_len as f64).sqrt();
            let mut sc_num_sq = 0.0_f64;
            let mut mag_diff_sum = 0.0_f64;
            let mut cell_count = 0usize;
            let mut scale_nz = cached.scale_nz_t;
            let mut win_cursor = 0usize;

            for g in 0..PROBE_GROUPS {
                step_signal_group(model, self.warmup_samples, geom, geom.trc[g], &mut ws);
                let mut eval_win = |em4: [f64; 4], hm4: [bool; 4]| -> Result<(), ()> {
                    let rt4 = cached.win_rms_t[win_cursor];
                    let lnt4 = cached.win_ln_rms_t[win_cursor];
                    win_cursor += 1;
                    for l in 0..4 {
                        if !em4[l].is_finite() {
                            return Err(());
                        }
                        scale_nz |= hm4[l];
                        let rms_m = em4[l].sqrt() * inv_sqrt_w;
                        if (hm4[l] && (em4[l] == 0.0 || rms_m == 0.0)) || !rms_m.is_finite() {
                            return Err(());
                        }
                        let a_t = rt4[l];
                        if a_t == 0.0 && rms_m == 0.0 {
                            cell_count += 1;
                            continue;
                        }
                        if a_t == 0.0 || rms_m == 0.0 {
                            return Err(());
                        }
                        let diff = a_t - rms_m;
                        sc_num_sq += diff * diff;
                        mag_diff_sum += (lnt4[l] - rms_m.ln()).abs();
                        cell_count += 1;
                    }
                    Ok(())
                };

                if t_sel < geom.m_scale {
                    let mut em = [0.0_f64; 4];
                    let mut hm = [false; 4];
                    for &v4 in &ws.tail_buf[..geom.tail_len] {
                        for l in 0..4 {
                            em[l] += v4[l] * v4[l];
                            hm[l] |= v4[l] != 0.0;
                        }
                    }
                    if eval_win(em, hm).is_err() {
                        return f64::INFINITY;
                    }
                } else {
                    for w_idx in 0..geom.num_full_windows {
                        let c0 = w_idx * geom.chunks_per_hop;
                        let c_end = c0 + geom.chunks_per_win;
                        let mut em = [0.0_f64; 4];
                        let mut hm = [false; 4];
                        for c in c0..c_end {
                            for l in 0..4 {
                                em[l] += ws.chunk_e[c][l];
                                hm[l] |= ws.chunk_nz[c][l];
                            }
                        }
                        if eval_win(em, hm).is_err() {
                            return f64::INFINITY;
                        }
                    }
                    if geom.has_unaligned_tail {
                        let mut em = [0.0_f64; 4];
                        let mut hm = [false; 4];
                        for &v4 in &ws.tail_buf[..geom.tail_len] {
                            for l in 0..4 {
                                em[l] += v4[l] * v4[l];
                                hm[l] |= v4[l] != 0.0;
                            }
                        }
                        if eval_win(em, hm).is_err() {
                            return f64::INFINITY;
                        }
                    }
                }
            }

            if !sc_num_sq.is_finite() || !cached.sc_den_sq.is_finite() || !mag_diff_sum.is_finite()
            {
                return f64::INFINITY;
            }
            let l_sc = if cached.sc_den_sq == 0.0 {
                if sc_num_sq == 0.0 && !scale_nz {
                    0.0
                } else {
                    return f64::INFINITY;
                }
            } else {
                sc_num_sq.sqrt() / cached.sc_den_sq.sqrt()
            };
            let l_mag = if cell_count == 0 {
                0.0
            } else {
                mag_diff_sum / (cell_count as f64)
            };
            total_loss += l_sc + l_mag;
        }

        let out = total_loss / (SCALES.len() as f64);
        if out.is_finite() { out } else { f64::INFINITY }
    }
}

/// Computes the 5-Scale Floor-Clamped Multi-Resolution IIR Resonator Bank Loss (`MR-IIR`,
/// $M \in \{256, 512, 1024, 2048, 4096\}$, pole radius $r_M = \exp(-\pi / M)$, hop $H = M / 4$,
/// $K = 32$ log-spaced probes, `ref[qual.companion.mr-iir]`, `ref[qual.besr.finite-timeline]`).
///
/// Returns `f64::INFINITY` on empty, length-mismatched, non-finite, or invalid-rate inputs so
/// invalid evaluations never pass tier gates.
#[must_use]
pub fn compute_mr_iir_loss(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> f64 {
    if noise.is_none()
        && target.len() == model.len()
        && !has_tiny_nonzero(model)
        && let Some(cache) = MrIirTargetCache::new(target, sample_rate_hz, warmup_samples)
    {
        return cache.eval_model(model);
    }
    compute_mr_iir_loss_general(target, model, noise, sample_rate_hz, warmup_samples)
}

pub(crate) fn compute_mr_iir_loss_general(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> f64 {
    if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
        || target.is_empty()
        || target.len() != model.len()
        || warmup_samples >= target.len()
    {
        return f64::INFINITY;
    }
    let n_total = target.len();
    if !target.iter().all(|v| v.is_finite()) || !model.iter().all(|v| v.is_finite()) {
        return f64::INFINITY;
    }
    if let Some(ns) = noise
        && (ns.len() < n_total || !ns[..n_total].iter().all(|v| v.is_finite()))
    {
        return f64::INFINITY;
    }

    let f_max = (0.45 * sample_rate_hz).min(20_000.0);
    if f_max <= 20.0 {
        return f64::INFINITY;
    }
    let t_sel = n_total - warmup_samples;
    let num_full_chunks = t_sel / SUB_CHUNK;
    let mut chunk_t = vec![0.0_f64; num_full_chunks];
    let mut chunk_m = vec![0.0_f64; num_full_chunks];
    let mut chunk_n = vec![0.0_f64; num_full_chunks];
    let mut nz_t = vec![false; num_full_chunks];
    let mut nz_m = vec![false; num_full_chunks];
    let mut nz_n = vec![false; num_full_chunks];
    let mut tail_t = [0.0_f64; 4096];
    let mut tail_m = [0.0_f64; 4096];
    let mut tail_n = [0.0_f64; 4096];

    let mut total_loss = 0.0_f64;

    for &m_scale in &SCALES {
        let hop = m_scale / 4;
        let r_m = (-PI / (m_scale as f64)).exp();
        let r_m_sq = r_m * r_m;
        let norm = 0.5 * (1.0 - r_m_sq);
        let tail_start = t_sel.saturating_sub(m_scale);

        let mut sc_num_sq = 0.0_f64;
        let mut sc_den_sq = 0.0_f64;
        let mut mag_diff_sum = 0.0_f64;
        let mut cell_count = 0usize;
        let mut scale_nz = false;
        let mut scale_nz_n = false;
        let mut scale_nse_sq = 0.0_f64;

        for p in 0..NUM_PROBES {
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            let omega = 2.0 * PI * f_k / sample_rate_hz;
            let two_r_cos = 2.0 * r_m * omega.cos();

            let (mut x1_t, mut x2_t, mut y1_t, mut y2_t) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut x1_m, mut x2_m, mut y1_m, mut y2_m) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut x1_n, mut x2_n, mut y1_n, mut y2_n) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

            chunk_t.fill(0.0);
            chunk_m.fill(0.0);
            chunk_n.fill(0.0);
            nz_t.fill(false);
            nz_m.fill(false);
            nz_n.fill(false);

            for i in 0..n_total {
                let xt = target[i];
                let drive_t = xt - x2_t;
                let mut bt = norm * drive_t + two_r_cos * y1_t - r_m_sq * y2_t;
                if drive_t == 0.0
                    && (xt == 0.0 || xt.abs() >= 1.0e-140)
                    && bt.abs() < 1.0e-140
                    && y1_t.abs() < 1.0e-140
                {
                    bt = 0.0;
                }
                x2_t = x1_t;
                x1_t = xt;
                y2_t = y1_t;
                y1_t = bt;

                let xm = model[i];
                let drive_m = xm - x2_m;
                let mut bm = norm * drive_m + two_r_cos * y1_m - r_m_sq * y2_m;
                if drive_m == 0.0
                    && (xm == 0.0 || xm.abs() >= 1.0e-140)
                    && bm.abs() < 1.0e-140
                    && y1_m.abs() < 1.0e-140
                {
                    bm = 0.0;
                }
                x2_m = x1_m;
                x1_m = xm;
                y2_m = y1_m;
                y1_m = bm;

                let xn = noise.map_or(0.0, |ns| ns[i]);
                let drive_n = xn - x2_n;
                let mut bn = norm * drive_n + two_r_cos * y1_n - r_m_sq * y2_n;
                if drive_n == 0.0
                    && (xn == 0.0 || xn.abs() >= 1.0e-140)
                    && bn.abs() < 1.0e-140
                    && y1_n.abs() < 1.0e-140
                {
                    bn = 0.0;
                }
                x2_n = x1_n;
                x1_n = xn;
                y2_n = y1_n;
                y1_n = bn;

                if i >= warmup_samples {
                    let sub_t = xt != 0.0 && xt.abs() < 1.0e-140;
                    let sub_m = xm != 0.0 && xm.abs() < 1.0e-140;
                    let sub_n = xn != 0.0 && xn.abs() < 1.0e-140;
                    let u = i - warmup_samples;
                    let c = u / SUB_CHUNK;
                    if c < num_full_chunks {
                        chunk_t[c] += bt * bt;
                        chunk_m[c] += bm * bm;
                        chunk_n[c] += bn * bn;
                        nz_t[c] |= bt != 0.0 || sub_t;
                        nz_m[c] |= bm != 0.0 || sub_m;
                        nz_n[c] |= bn != 0.0 || sub_n;
                    }
                    if u >= tail_start {
                        let ti = u - tail_start;
                        tail_t[ti] = if bt == 0.0 && sub_t { xt } else { bt };
                        tail_m[ti] = if bm == 0.0 && sub_m { xm } else { bm };
                        tail_n[ti] = if bn == 0.0 && sub_n { xn } else { bn };
                    }
                }
            }

            let mut eval_energies = |e_t: f64,
                                     e_m: f64,
                                     e_n: f64,
                                     ht: bool,
                                     hm: bool,
                                     hn: bool,
                                     w: usize|
             -> Result<(), ()> {
                if !e_t.is_finite() || !e_m.is_finite() || !e_n.is_finite() {
                    return Err(());
                }
                scale_nz |= ht || hm || hn;
                scale_nz_n |= hn;
                scale_nse_sq += e_n;
                let inv_sqrt_len = 1.0 / (w as f64).sqrt();
                let rms_t = e_t.sqrt() * inv_sqrt_len;
                let rms_m = e_m.sqrt() * inv_sqrt_len;
                let sigma = e_n.sqrt() * inv_sqrt_len;
                if (hn && (e_n == 0.0 || sigma == 0.0))
                    || (((ht && (e_t == 0.0 || rms_t == 0.0))
                        || (hm && (e_m == 0.0 || rms_m == 0.0)))
                        && sigma < 1.6e-162)
                {
                    return Err(());
                }
                let (a_t, a_m) = (rms_t.max(sigma), rms_m.max(sigma));
                if a_t == 0.0 && a_m == 0.0 {
                    cell_count += 1;
                    return Ok(());
                }
                if a_t == 0.0 || a_m == 0.0 || !a_t.is_finite() || !a_m.is_finite() {
                    return Err(());
                }
                let diff = a_t - a_m;
                sc_num_sq += diff * diff;
                sc_den_sq += a_t * a_t;
                mag_diff_sum += (a_t.ln() - a_m.ln()).abs();
                cell_count += 1;
                Ok(())
            };

            let sum_tail = |len: usize,
                            tt: &[f64],
                            tm: &[f64],
                            tn: &[f64]|
             -> (f64, f64, f64, bool, bool, bool) {
                let (mut et, mut em, mut en) = (0.0_f64, 0.0_f64, 0.0_f64);
                let (mut ht, mut hm, mut hn) = (false, false, false);
                for k in 0..len {
                    let (vt, vm, vn) = (tt[k], tm[k], tn[k]);
                    et += vt * vt;
                    em += vm * vm;
                    en += vn * vn;
                    ht |= vt != 0.0;
                    hm |= vm != 0.0;
                    hn |= vn != 0.0;
                }
                (et, em, en, ht, hm, hn)
            };

            if t_sel < m_scale {
                let (et, em, en, ht, hm, hn) = sum_tail(t_sel, &tail_t, &tail_m, &tail_n);
                if eval_energies(et, em, en, ht, hm, hn, t_sel).is_err() {
                    return f64::INFINITY;
                }
            } else {
                let chunks_per_win = m_scale / SUB_CHUNK;
                let chunks_per_hop = hop / SUB_CHUNK;
                let mut c0 = 0usize;
                while (c0 + chunks_per_win) * SUB_CHUNK <= t_sel {
                    let c_end = c0 + chunks_per_win;
                    let et: f64 = chunk_t[c0..c_end].iter().sum();
                    let em: f64 = chunk_m[c0..c_end].iter().sum();
                    let en: f64 = chunk_n[c0..c_end].iter().sum();
                    let ht = nz_t[c0..c_end].iter().any(|&b| b);
                    let hm = nz_m[c0..c_end].iter().any(|&b| b);
                    let hn = nz_n[c0..c_end].iter().any(|&b| b);
                    if eval_energies(et, em, en, ht, hm, hn, m_scale).is_err() {
                        return f64::INFINITY;
                    }
                    c0 += chunks_per_hop;
                }
                let start = c0 * SUB_CHUNK;
                let last_start = t_sel - m_scale;
                if last_start + hop != start {
                    let (et, em, en, ht, hm, hn) = sum_tail(m_scale, &tail_t, &tail_m, &tail_n);
                    if eval_energies(et, em, en, ht, hm, hn, m_scale).is_err() {
                        return f64::INFINITY;
                    }
                }
            }
        }

        if !sc_num_sq.is_finite()
            || !sc_den_sq.is_finite()
            || !mag_diff_sum.is_finite()
            || (scale_nz_n && scale_nse_sq == 0.0)
        {
            return f64::INFINITY;
        }
        let l_sc = if sc_den_sq == 0.0 {
            if sc_num_sq == 0.0 && !scale_nz {
                0.0
            } else {
                return f64::INFINITY;
            }
        } else {
            sc_num_sq.sqrt() / sc_den_sq.sqrt()
        };
        let l_mag = if cell_count == 0 {
            0.0
        } else {
            mag_diff_sum / (cell_count as f64)
        };
        total_loss += l_sc + l_mag;
    }

    let out = total_loss / (SCALES.len() as f64);
    if out.is_finite() { out } else { f64::INFINITY }
}
