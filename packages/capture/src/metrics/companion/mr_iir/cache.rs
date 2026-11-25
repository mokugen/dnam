#![forbid(unsafe_code)]
//! Cached 5-scale `MR-IIR` target resonator evaluator (`noise = None`, `r[qual.companion.mr-iir]`).

use super::super::MAX_METRIC_SAMPLE_RATE_HZ;
use super::{
    PROBE_GROUPS, SCALES, SUB_CHUNK, compute_mr_iir_loss_general, has_tiny_nonzero,
    probe_norm_and_trc,
};

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

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use super::*;

    #[test]
    fn test_mr_iir_cache_matches_general_reference_and_zero_self_null() {
        let sr = 48_000.0;
        let n = 6_000_usize;
        let mut tgt = vec![0.0_f64; n];
        let mut mdl = vec![0.0_f64; n];
        for i in 0..n {
            let t = (i as f64) / sr;
            let s = 0.25 * (TAU * 310.0 * t).sin() + 0.15 * (TAU * 2_400.0 * t).cos();
            tgt[i] = s;
            mdl[i] = 1.02 * s;
        }
        let fast = MrIirTargetCache::new(&tgt, sr, 64)
            .unwrap()
            .eval_model(&mdl);
        let ref_v = compute_mr_iir_loss_general(&tgt, &mdl, None, sr, 64);
        assert!((fast - ref_v).abs() < 1.0e-12, "fast={fast} ref={ref_v}");

        let zeros = vec![0.0_f64; 1_024];
        let z_loss = MrIirTargetCache::new(&zeros, sr, 32)
            .unwrap()
            .eval_model(&zeros);
        assert_eq!(z_loss, 0.0);
    }
}
