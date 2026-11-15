#![forbid(unsafe_code)]
//! Double-precision `v3` training stem and dataset containers with precomputed spectral summaries.

use super::bezout7::Bezout7FilterBank;
use super::gabor::{AnalyticGaborBank, GaborTargetSummary, build_gabor_target_summary};
use super::ltas_adj::{Welch33TargetSummary, build_welch33_target_summary};

/// Single dry+wet stereo identification stem.
#[derive(Clone, Debug)]
pub struct V3TrainStem {
    pub in_l: Vec<f32>,
    pub in_r: Vec<f32>,
    pub tgt_l: Vec<f32>,
    pub tgt_r: Vec<f32>,
    pub warmup: usize,
}

impl V3TrainStem {
    /// Returns usable aligned sample length across all 4 buffers.
    #[inline]
    #[must_use]
    pub fn usable_len(&self) -> usize {
        self.in_l
            .len()
            .min(self.in_r.len())
            .min(self.tgt_l.len())
            .min(self.tgt_r.len())
    }
}

/// Complete multi-stem black-box training dataset with precomputed perceptual target summaries.
#[derive(Clone, Debug)]
pub struct V3TrainDataset {
    pub stems: Vec<V3TrainStem>,
    pub stem_weights: Vec<f64>,
    pub sample_rate_hz: f64,
    pub bank: Bezout7FilterBank,
    pub gabor_bank: AnalyticGaborBank,
    pub gabor_summaries: Vec<[GaborTargetSummary; 2]>,
    pub welch33_summaries: Vec<[Welch33TargetSummary; 2]>,
    pub stem_band_weights: Vec<[[f64; 3]; 2]>,
    pub gabor_env_weight: f64,
    pub gabor_ltas_weight: f64,
    pub welch_ltas_weight: f64,
}

fn compute_stem_band_weights(
    bank: &Bezout7FilterBank,
    tl: &[f64],
    tr: &[f64],
    w: usize,
    sw: f64,
) -> [[f64; 3]; 2] {
    let mut ch_bw = [[0.0_f64; 3]; 2];
    let e_l = bank.band_energies(tl, w);
    let e_r = if tl == tr {
        e_l
    } else {
        bank.band_energies(tr, w)
    };
    for (ch, e) in [e_l, e_r].into_iter().enumerate() {
        let mean_e = (e[0] + e[1] + e[2]) / 3.0;
        let floor = (mean_e * 1.0e-5).max(1.0e-12);
        for b in 0..3 {
            ch_bw[ch][b] = sw / (3.0 * e[b].max(floor));
        }
    }
    ch_bw
}

impl V3TrainDataset {
    /// Builds a uniformly weighted dataset with default spectral regularization.
    #[must_use]
    pub fn new(stems: Vec<V3TrainStem>, sample_rate_hz: f64) -> Self {
        let w = if stems.is_empty() {
            1.0
        } else {
            1.0 / (stems.len() as f64)
        };
        let weighted = stems.into_iter().map(|s| (s, w)).collect();
        Self::new_weighted(weighted, sample_rate_hz)
    }

    /// Builds a weighted dataset precomputing Bézout-3, Gabor-33, and Welch-33 summaries.
    #[must_use]
    pub fn new_weighted(weighted_stems: Vec<(V3TrainStem, f64)>, sample_rate_hz: f64) -> Self {
        let sr = sample_rate_hz.max(8_000.0);
        let bank = Bezout7FilterBank::new(sr);
        let gabor_bank = AnalyticGaborBank::new(sr);
        let w_sum: f64 = weighted_stems
            .iter()
            .map(|(_, w)| w.max(0.0))
            .sum::<f64>()
            .max(1.0e-12);

        let mut stems = Vec::with_capacity(weighted_stems.len());
        let mut stem_weights = Vec::with_capacity(weighted_stems.len());
        let mut stem_band_weights = Vec::with_capacity(weighted_stems.len());
        let mut gabor_summaries = Vec::with_capacity(weighted_stems.len());
        let mut welch33_summaries = Vec::with_capacity(weighted_stems.len());

        for (st, raw_w) in weighted_stems {
            let sw = raw_w.max(0.0) / w_sum;
            let n = st.usable_len();
            let w = st.warmup.min(n);

            let tl: Vec<f64> = st.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
            let tr: Vec<f64> = st.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
            let ch_bw = compute_stem_band_weights(&bank, &tl, &tr, w, sw);
            let same_ch = tl == tr;
            stem_band_weights.push(ch_bw);
            let gs_l = build_gabor_target_summary(&tl, w, &gabor_bank);
            let gs_r = if same_ch {
                gs_l.clone()
            } else {
                build_gabor_target_summary(&tr, w, &gabor_bank)
            };
            gabor_summaries.push([gs_l, gs_r]);
            let ws_l = build_welch33_target_summary(&tl, w, sr);
            let ws_r = if same_ch {
                ws_l.clone()
            } else {
                build_welch33_target_summary(&tr, w, sr)
            };
            welch33_summaries.push([ws_l, ws_r]);
            stems.push(st);
            stem_weights.push(sw);
        }

        Self {
            stems,
            stem_weights,
            sample_rate_hz: sr,
            bank,
            gabor_bank,
            gabor_summaries,
            welch33_summaries,
            stem_band_weights,
            gabor_env_weight: 0.0,
            gabor_ltas_weight: 0.0,
            welch_ltas_weight: 0.0,
        }
    }

    /// Returns a shallow dataset clone configured with Gabor weights (`welch_ltas_weight = 0`).
    #[must_use]
    pub fn with_gabor_weights(&self, env_weight: f64, ltas_weight: f64) -> Self {
        let mut out = self.clone();
        out.gabor_env_weight = env_weight.max(0.0);
        out.gabor_ltas_weight = ltas_weight.max(0.0);
        out.welch_ltas_weight = 0.0;
        out
    }

    /// Returns a dataset clone configured with Gabor-33 and Welch-33 spectral regularization weights.
    #[must_use]
    pub fn with_spectral_weights(
        &self,
        gabor_env_weight: f64,
        gabor_ltas_weight: f64,
        welch_ltas_weight: f64,
    ) -> Self {
        let mut out = self.clone();
        out.gabor_env_weight = gabor_env_weight.max(0.0);
        out.gabor_ltas_weight = gabor_ltas_weight.max(0.0);
        out.welch_ltas_weight = welch_ltas_weight.max(0.0);
        out
    }

    /// Slices a single stem from `start` for `len` samples with explicit `warmup`.
    fn slice_stem(st: &V3TrainStem, start: usize, len: usize, warmup: usize) -> V3TrainStem {
        let end = start + len;
        V3TrainStem {
            in_l: st.in_l[start..end].to_vec(),
            in_r: st.in_r[start..end].to_vec(),
            tgt_l: st.tgt_l[start..end].to_vec(),
            tgt_r: st.tgt_r[start..end].to_vec(),
            warmup,
        }
    }

    /// Truncates a single stem to at most `max_samples`.
    fn truncate_stem(st: &V3TrainStem, max_samples: usize) -> V3TrainStem {
        let n = st.usable_len().min(max_samples.max(64));
        Self::slice_stem(st, 0, n, st.warmup.min(n / 5))
    }

    fn clone_with_pairs(&self, pairs: Vec<(V3TrainStem, f64)>) -> Self {
        let mut out = Self::new_weighted(pairs, self.sample_rate_hz);
        out.gabor_env_weight = self.gabor_env_weight;
        out.gabor_ltas_weight = self.gabor_ltas_weight;
        out.welch_ltas_weight = self.welch_ltas_weight;
        out
    }

    /// Returns a truncated curriculum dataset capped at `max_samples` per stem.
    #[must_use]
    pub fn subwindow(&self, max_samples: usize) -> Self {
        let pairs = self
            .stems
            .iter()
            .zip(&self.stem_weights)
            .map(|(st, &w)| (Self::truncate_stem(st, max_samples), w))
            .collect();
        self.clone_with_pairs(pairs)
    }

    /// Returns a curriculum dataset capping the primary stem at `primary_max` and secondary stems at `secondary_max`.
    #[must_use]
    pub fn curriculum_subwindow(&self, primary_max: usize, secondary_max: usize) -> Self {
        let pairs = self
            .stems
            .iter()
            .zip(&self.stem_weights)
            .enumerate()
            .map(|(si, (st, &w))| {
                if si == 0 {
                    (Self::truncate_stem(st, primary_max), w)
                } else {
                    let total = st.usable_len();
                    let n = total.min(secondary_max.max(64));
                    if total > n.saturating_mul(2) {
                        let start = (total - n) / 2;
                        let warm = (n / 5).clamp(64, 3_072).min(n / 2);
                        (Self::slice_stem(st, start, n, warm), w)
                    } else {
                        (Self::slice_stem(st, 0, n, st.warmup.min(n / 5)), w)
                    }
                }
            })
            .collect();
        self.clone_with_pairs(pairs)
    }

    /// Returns a fast curriculum dataset using only the primary highest-weight stem capped at `max_samples`.
    #[must_use]
    pub fn primary_subwindow(&self, max_samples: usize) -> Self {
        let mut best: Option<(&V3TrainStem, f64)> = None;
        for (st, &w) in self.stems.iter().zip(&self.stem_weights) {
            if best.is_none_or(|(_, bw)| w > bw) {
                best = Some((st, w));
            }
        }
        let pairs = best
            .map(|(st, w)| vec![(Self::truncate_stem(st, max_samples), w)])
            .unwrap_or_default();
        self.clone_with_pairs(pairs)
    }

    /// Returns a fast Bézout-only truncated dataset skipping Gabor-33 and Welch-33 summary builds.
    #[must_use]
    pub fn subwindow_bezout_only(&self, max_samples: usize) -> Self {
        let sr = self.sample_rate_hz;
        let bank = Bezout7FilterBank::new(sr);
        let gabor_bank = self.gabor_bank.clone();
        let empty_gab = build_gabor_target_summary(&[], 0, &gabor_bank);
        let empty_wel = build_welch33_target_summary(&[], 0, sr);
        let w_sum: f64 = self
            .stem_weights
            .iter()
            .map(|w| w.max(0.0))
            .sum::<f64>()
            .max(1.0e-12);
        let mut stems = Vec::with_capacity(self.stems.len());
        let mut stem_weights = Vec::with_capacity(self.stems.len());
        let mut stem_band_weights = Vec::with_capacity(self.stems.len());
        let mut gabor_summaries = Vec::with_capacity(self.stems.len());
        let mut welch33_summaries = Vec::with_capacity(self.stems.len());
        for (st, &raw_w) in self.stems.iter().zip(&self.stem_weights) {
            let sw = raw_w.max(0.0) / w_sum;
            let n = st.usable_len().min(max_samples.max(64));
            let warmup = st.warmup.min(n / 5);
            let tl: Vec<f64> = st.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
            let tr: Vec<f64> = st.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
            stem_band_weights.push(compute_stem_band_weights(&bank, &tl, &tr, warmup, sw));
            gabor_summaries.push([empty_gab.clone(), empty_gab.clone()]);
            welch33_summaries.push([empty_wel.clone(), empty_wel.clone()]);
            stems.push(V3TrainStem {
                in_l: st.in_l[..n].to_vec(),
                in_r: st.in_r[..n].to_vec(),
                tgt_l: st.tgt_l[..n].to_vec(),
                tgt_r: st.tgt_r[..n].to_vec(),
                warmup,
            });
            stem_weights.push(sw);
        }
        Self {
            stems,
            stem_weights,
            sample_rate_hz: sr,
            bank,
            gabor_bank,
            gabor_summaries,
            welch33_summaries,
            stem_band_weights,
            gabor_env_weight: 0.0,
            gabor_ltas_weight: 0.0,
            welch_ltas_weight: 0.0,
        }
    }
}
