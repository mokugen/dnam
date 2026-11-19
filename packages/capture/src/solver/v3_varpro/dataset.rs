#![forbid(unsafe_code)]
//! Double-precision `v3` training stem and dataset containers with precomputed spectral summaries.

use crate::metrics::TargetRepeatabilityProfile;
use crate::solver::SolverQualityMode;
use crate::solver::progress::TrainerProgressHandle;

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
    pub full_cert_stem: Option<V3TrainStem>,
    pub rep_profiles: Option<[TargetRepeatabilityProfile; 2]>,
    pub sample_rate_hz: f64,
    pub bank: Bezout7FilterBank,
    pub gabor_bank: AnalyticGaborBank,
    pub gabor_summaries: Vec<[GaborTargetSummary; 2]>,
    pub welch33_summaries: Vec<[Welch33TargetSummary; 2]>,
    pub stem_band_weights: Vec<[[f64; 3]; 2]>,
    pub gabor_env_weight: f64,
    pub gabor_ltas_weight: f64,
    pub welch_ltas_weight: f64,
    pub quality_mode: SolverQualityMode,
    pub progress: TrainerProgressHandle,
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
        Self::new_weighted_opts(weighted_stems, sample_rate_hz, true, true)
    }

    /// Builds a weighted dataset, computing Gabor-33 and/or Welch-33 spectral summaries on demand.
    #[must_use]
    pub fn new_weighted_opts(
        weighted_stems: Vec<(V3TrainStem, f64)>,
        sample_rate_hz: f64,
        need_gabor: bool,
        need_welch: bool,
    ) -> Self {
        let sr = sample_rate_hz.max(8_000.0);
        let bank = Bezout7FilterBank::new(sr);
        let gabor_bank = AnalyticGaborBank::new(sr);
        let empty_gab = (!need_gabor).then(|| build_gabor_target_summary(&[], 0, &gabor_bank));
        let empty_wel = (!need_welch).then(|| build_welch33_target_summary(&[], 0, sr));
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
            let gs_l = match &empty_gab {
                Some(e) => e.clone(),
                None => build_gabor_target_summary(&tl, w, &gabor_bank),
            };
            let gs_r = if same_ch {
                gs_l.clone()
            } else if let Some(e) = &empty_gab {
                e.clone()
            } else {
                build_gabor_target_summary(&tr, w, &gabor_bank)
            };
            gabor_summaries.push([gs_l, gs_r]);
            let ws_l = match &empty_wel {
                Some(e) => e.clone(),
                None => build_welch33_target_summary(&tl, w, sr),
            };
            let ws_r = if same_ch {
                ws_l.clone()
            } else if let Some(e) = &empty_wel {
                e.clone()
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
            full_cert_stem: None,
            rep_profiles: None,
            sample_rate_hz: sr,
            bank,
            gabor_bank,
            gabor_summaries,
            welch33_summaries,
            stem_band_weights,
            gabor_env_weight: 0.0,
            gabor_ltas_weight: 0.0,
            welch_ltas_weight: 0.0,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            progress: TrainerProgressHandle::default(),
        }
    }

    /// Returns a shallow dataset clone configured with explicit quality vs. wall-clock mode.
    #[must_use]
    pub fn with_quality_mode(&self, quality_mode: SolverQualityMode) -> Self {
        let mut out = self.clone();
        out.quality_mode = quality_mode;
        out
    }

    /// Returns a shallow dataset clone attached to `progress` (no-op overhead when inactive).
    #[must_use]
    pub fn with_progress(&self, progress: TrainerProgressHandle) -> Self {
        let mut out = self.clone();
        out.progress = progress;
        out
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

    fn secondary_window_stem(st: &V3TrainStem, secondary_max: usize) -> V3TrainStem {
        let total = st.usable_len();
        let n = total.min(secondary_max.max(64));
        if total > n.saturating_mul(2) {
            let start = (total - n) / 2;
            let warm = (n / 5).clamp(64, 3_072).min(n / 2);
            Self::slice_stem(st, start, n, warm)
        } else {
            Self::slice_stem(st, 0, n, st.warmup.min(n / 5))
        }
    }

    fn clone_with_pairs_opts(
        &self,
        pairs: Vec<(V3TrainStem, f64)>,
        need_gabor: bool,
        need_welch: bool,
    ) -> Self {
        let mut out = Self::new_weighted_opts(pairs, self.sample_rate_hz, need_gabor, need_welch);
        out.full_cert_stem = self.full_cert_stem.clone();
        out.rep_profiles = self.rep_profiles;
        out.gabor_env_weight = self.gabor_env_weight;
        out.gabor_ltas_weight = self.gabor_ltas_weight;
        out.welch_ltas_weight = self.welch_ltas_weight;
        out.quality_mode = self.quality_mode;
        out.progress = self.progress.clone();
        out
    }

    fn clone_with_pairs(&self, pairs: Vec<(V3TrainStem, f64)>) -> Self {
        self.clone_with_pairs_opts(pairs, true, true)
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
        self.curriculum_subwindow_opts(primary_max, secondary_max, true, true)
    }

    /// Returns a curriculum dataset with selective Gabor-33 / Welch-33 summary computation.
    #[must_use]
    pub fn curriculum_subwindow_opts(
        &self,
        primary_max: usize,
        secondary_max: usize,
        need_gabor: bool,
        need_welch: bool,
    ) -> Self {
        let pairs = self
            .stems
            .iter()
            .zip(&self.stem_weights)
            .enumerate()
            .map(|(si, (st, &w))| {
                if si == 0 {
                    (Self::truncate_stem(st, primary_max), w)
                } else {
                    (Self::secondary_window_stem(st, secondary_max), w)
                }
            })
            .collect();
        self.clone_with_pairs_opts(pairs, need_gabor, need_welch)
    }

    /// Returns a fast curriculum dataset using only the primary highest-weight stem capped at `max_samples`.
    #[must_use]
    pub fn primary_subwindow(&self, max_samples: usize) -> Self {
        self.primary_subwindow_opts(max_samples, true, true)
    }

    /// Returns a primary-stem curriculum dataset with selective Gabor-33 / Welch-33 computation.
    #[must_use]
    pub fn primary_subwindow_opts(
        &self,
        max_samples: usize,
        need_gabor: bool,
        need_welch: bool,
    ) -> Self {
        let mut best: Option<(&V3TrainStem, f64)> = None;
        for (st, &w) in self.stems.iter().zip(&self.stem_weights) {
            if best.is_none_or(|(_, bw)| w > bw) {
                best = Some((st, w));
            }
        }
        let pairs = best
            .map(|(st, w)| vec![(Self::truncate_stem(st, max_samples), w)])
            .unwrap_or_default();
        self.clone_with_pairs_opts(pairs, need_gabor, need_welch)
    }

    /// Splits the primary stem into up to `max_windows` evenly spaced windows for parallel BPTT.
    #[must_use]
    pub fn windowed_bptt_subwindow(
        &self,
        window_len: usize,
        max_windows: usize,
        burn_in: usize,
        secondary_max: usize,
        need_gabor: bool,
        need_welch: bool,
    ) -> Self {
        if self.stems.is_empty() {
            return self.clone_with_pairs_opts(Vec::new(), need_gabor, need_welch);
        }
        let mut pairs = Vec::with_capacity(max_windows.max(1) + self.stems.len());
        let st0 = &self.stems[0];
        let total = st0.usable_len();
        let w0 = self.stem_weights[0];
        if total == 0 {
            pairs.push((Self::truncate_stem(st0, window_len), w0));
        } else {
            let win_n = total.min(window_len.max(64));
            if total <= win_n || max_windows <= 1 {
                let warm = st0.warmup.min(win_n / 4);
                pairs.push((Self::slice_stem(st0, 0, win_n, warm), w0));
            } else {
                let k_win = max_windows.min((total / win_n).max(2));
                let span = total - win_n;
                let w_each = w0 / (k_win as f64);
                for idx in 0..k_win {
                    let start = (span * idx) / (k_win - 1);
                    let warm = if idx == 0 {
                        st0.warmup.min(win_n / 4)
                    } else {
                        burn_in.clamp(16.min(win_n / 2), win_n / 2)
                    };
                    pairs.push((Self::slice_stem(st0, start, win_n, warm), w_each));
                }
            }
        }
        for (st, &w) in self.stems.iter().zip(&self.stem_weights).skip(1) {
            pairs.push((Self::secondary_window_stem(st, secondary_max), w));
        }
        self.clone_with_pairs_opts(pairs, need_gabor, need_welch)
    }

    /// Returns a Stage 3 polish dataset preserving primary hardware warmup and Welch summaries.
    #[must_use]
    pub fn polish_subwindow(&self, primary_max: usize, secondary_max: usize) -> Self {
        let pairs = self
            .stems
            .iter()
            .zip(&self.stem_weights)
            .enumerate()
            .map(|(si, (st, &w))| {
                if si == 0 {
                    let n = st.usable_len().min(primary_max.max(64));
                    (Self::slice_stem(st, 0, n, st.warmup.min(n / 2)), w)
                } else {
                    (Self::secondary_window_stem(st, secondary_max), w)
                }
            })
            .collect();
        self.clone_with_pairs_opts(pairs, false, true)
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
            full_cert_stem: self.full_cert_stem.clone(),
            rep_profiles: self.rep_profiles,
            sample_rate_hz: sr,
            bank,
            gabor_bank,
            gabor_summaries,
            welch33_summaries,
            stem_band_weights,
            gabor_env_weight: 0.0,
            gabor_ltas_weight: 0.0,
            welch_ltas_weight: 0.0,
            quality_mode: self.quality_mode,
            progress: self.progress.clone(),
        }
    }

    /// Returns a Stage 3 polish dataset reusing stem `0` (and its cached Welch-33/Bézout stats)
    /// while capping secondary stems at `secondary_max` without recomputing `FFT` summaries.
    #[must_use]
    pub fn for_stage3_polish(&self, secondary_max: usize) -> Self {
        if self.stems.is_empty() {
            return self.clone();
        }
        let sr = self.sample_rate_hz;
        let bank = self.bank;
        let gabor_bank = self.gabor_bank.clone();
        let empty_gab = build_gabor_target_summary(&[], 0, &gabor_bank);
        let empty_wel = build_welch33_target_summary(&[], 0, sr);

        let n_stems = self.stems.len();
        let mut stems = Vec::with_capacity(n_stems);
        let mut gabor_summaries = Vec::with_capacity(n_stems);
        let mut welch33_summaries = Vec::with_capacity(n_stems);
        let mut stem_band_weights = Vec::with_capacity(n_stems);

        let st0 = &self.stems[0];
        let n0 = st0.usable_len();
        let wel0 = if self.welch33_summaries[0][0].len == n0 {
            self.welch33_summaries[0].clone()
        } else {
            let w0 = st0.warmup.min(n0);
            let tl0: Vec<f64> = st0.tgt_l[..n0].iter().map(|&v| f64::from(v)).collect();
            let tr0: Vec<f64> = st0.tgt_r[..n0].iter().map(|&v| f64::from(v)).collect();
            let wl = build_welch33_target_summary(&tl0, w0, sr);
            let wr = if tl0 == tr0 {
                wl.clone()
            } else {
                build_welch33_target_summary(&tr0, w0, sr)
            };
            [wl, wr]
        };

        stems.push(st0.clone());
        gabor_summaries.push([empty_gab.clone(), empty_gab.clone()]);
        welch33_summaries.push(wel0);
        stem_band_weights.push(self.stem_band_weights[0]);

        for (st, &sw) in self.stems.iter().zip(&self.stem_weights).skip(1) {
            let sub = Self::secondary_window_stem(st, secondary_max);
            let n = sub.usable_len();
            let w = sub.warmup.min(n);
            let tl: Vec<f64> = sub.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
            let tr: Vec<f64> = sub.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
            stem_band_weights.push(compute_stem_band_weights(&bank, &tl, &tr, w, sw));
            gabor_summaries.push([empty_gab.clone(), empty_gab.clone()]);
            welch33_summaries.push([empty_wel.clone(), empty_wel.clone()]);
            stems.push(sub);
        }

        Self {
            stems,
            stem_weights: self.stem_weights.clone(),
            full_cert_stem: self.full_cert_stem.clone(),
            rep_profiles: self.rep_profiles,
            sample_rate_hz: sr,
            bank,
            gabor_bank,
            gabor_summaries,
            welch33_summaries,
            stem_band_weights,
            gabor_env_weight: self.gabor_env_weight,
            gabor_ltas_weight: self.gabor_ltas_weight,
            welch_ltas_weight: self.welch_ltas_weight,
            quality_mode: self.quality_mode,
            progress: self.progress.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_empty_weighted_and_subwindows() {
        let empty = V3TrainDataset::new(Vec::new(), 48_000.0);
        assert!(empty.stems.is_empty());
        assert!(empty.primary_subwindow(64).stems.is_empty());
        assert!(empty.for_stage3_polish(64).stems.is_empty());

        let stem_a = V3TrainStem {
            in_l: vec![0.1_f32; 256],
            in_r: vec![-0.1_f32; 256],
            tgt_l: vec![0.2_f32; 256],
            tgt_r: vec![0.05_f32; 256],
            warmup: 16,
        };
        let stem_b = V3TrainStem {
            in_l: vec![0.15_f32; 512],
            in_r: vec![0.15_f32; 512],
            tgt_l: vec![0.25_f32; 512],
            tgt_r: vec![0.25_f32; 512],
            warmup: 32,
        };
        let ds = V3TrainDataset::new_weighted(vec![(stem_a, 0.25), (stem_b, 0.75)], 48_000.0);
        assert_eq!(ds.stems.len(), 2);

        let cur = ds.curriculum_subwindow(128, 128);
        assert_eq!(cur.stems[0].usable_len(), 128);
        assert_eq!(cur.stems[1].usable_len(), 128);

        let prim = ds.primary_subwindow(96);
        assert_eq!(prim.stems.len(), 1);
        assert_eq!(prim.stems[0].usable_len(), 96);

        let bez = ds.subwindow_bezout_only(128);
        assert_eq!(bez.gabor_env_weight, 0.0);
        assert_eq!(bez.welch_ltas_weight, 0.0);

        let no_spec = V3TrainDataset::new_weighted_opts(
            ds.stems
                .iter()
                .cloned()
                .zip(ds.stem_weights.iter().copied())
                .collect(),
            48_000.0,
            false,
            false,
        );
        assert!(no_spec.gabor_summaries[0][0].tgt_env.is_empty());
        assert_eq!(no_spec.stems.len(), 2);

        let wbptt = ds.windowed_bptt_subwindow(96, 3, 64, 128, false, true);
        assert_eq!(wbptt.stems.len(), 3);
        assert_eq!(wbptt.stems[0].usable_len(), 96);

        let pol = ds.polish_subwindow(200, 128);
        assert_eq!(pol.stems[0].usable_len(), 200);
        assert_eq!(pol.stems[0].warmup, 16);
        assert_eq!(pol.stems[1].usable_len(), 128);

        let st3 = ds.for_stage3_polish(128);
        assert_eq!(st3.stems[0].usable_len(), 256);
        assert_eq!(st3.stems[1].usable_len(), 128);
        assert_eq!(st3.stem_band_weights[0], ds.stem_band_weights[0]);

        assert_eq!(
            ds.quality_mode,
            SolverQualityMode::BestUntilDiminishingReturns
        );
        let fast_mode = SolverQualityMode::ThresholdStop {
            target_slack_milli_db: 250,
            max_polish_iters: 24,
        };
        let ds_fast = ds.with_quality_mode(fast_mode);
        assert_eq!(ds_fast.quality_mode, fast_mode);
        assert_eq!(ds_fast.primary_subwindow(64).quality_mode, fast_mode);
        assert_eq!(ds_fast.subwindow_bezout_only(64).quality_mode, fast_mode);
        assert_eq!(ds_fast.for_stage3_polish(64).quality_mode, fast_mode);

        let prog = TrainerProgressHandle::from_callback(|_| {});
        let ds_prog = ds.with_progress(prog);
        assert!(ds_prog.progress.is_active());
        assert!(ds_prog.primary_subwindow(64).progress.is_active());
        assert!(ds_prog.subwindow_bezout_only(64).progress.is_active());
        assert!(ds_prog.for_stage3_polish(64).progress.is_active());
    }
}
