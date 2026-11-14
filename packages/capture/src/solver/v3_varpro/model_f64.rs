#![forbid(unsafe_code)]
//! `f64` optimization parameter mirror (`V3OptModelF64`), `V3TrainDataset`, and 3-band LR4 filter bank.

use std::f64::consts::PI;

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::cascade::{
    MAX_CASCADE_LANES, MAX_KAUTZ_POLES, NUM_CASCADE_BLOCKS, tier_cascade_lanes,
};
use dnam_engine::stages::v3_core::svf::V3Stage1SvfParams;
use dnam_engine::stages::v3_core::{V3CascadeBlockCold, V3ColdTierModel};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::gabor::{
    AnalyticGaborBank, GaborAdjointScratch, GaborTargetSummary, build_gabor_target_summary,
    eval_gabor_loss_and_adjoint_with_backend as gabor_loss_and_adjoint_with_backend,
};

pub(crate) const HB_A0: [f64; 2] = [0.079_866_426, 0.545_353_65];
pub(crate) const HB_A1: [f64; 2] = [0.283_829_34, 0.834_411_89];

/// Single labeled stereo capture stem for `v3` `VarPro` + adjoint system identification.
#[derive(Clone, Debug)]
pub struct V3TrainStem {
    pub in_l: Vec<f32>,
    pub in_r: Vec<f32>,
    pub tgt_l: Vec<f32>,
    pub tgt_r: Vec<f32>,
    pub warmup: usize,
}

#[inline(always)]
fn step_bezout7_lp(x: f64, r: f64, omr: f64, u: &mut [f64; 7]) -> f64 {
    let mut prev = x;
    for stage in u.iter_mut() {
        *stage = omr * prev + r * *stage;
        prev = *stage;
    }
    35.0 * u[3] - 84.0 * u[4] + 70.0 * u[5] - 20.0 * u[6]
}

#[inline]
fn accumulate_bezout7_lp_adjoint(dy: &[f64], r: f64, omr: f64, dx: &mut [f64]) {
    let mut du_future = [0.0_f64; 7];
    for t in (0..dy.len()).rev() {
        let mut du = du_future;
        let dy_t = dy[t];
        du[3] += 35.0 * dy_t;
        du[4] -= 84.0 * dy_t;
        du[5] += 70.0 * dy_t;
        du[6] -= 20.0 * dy_t;
        for k in (1..7).rev() {
            du_future[k] = r * du[k];
            du[k - 1] += omr * du[k];
        }
        du_future[0] = r * du[0];
        dx[t] += omr * du[0];
    }
}

/// Exact degree-7 complementary Binomial/Bézout 3-band crossover (`250 Hz` & `2500 Hz`) matching
/// `dnam_capture::metrics::try_split_bezout_3band_energies`.
#[derive(Clone, Copy, Debug)]
pub struct Bezout7FilterBank {
    r1: f64,
    omr1: f64,
    r2: f64,
    omr2: f64,
}

impl Bezout7FilterBank {
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let r1 = (-2.0 * PI * 250.0 / sample_rate_hz).exp();
        let r2 = (-2.0 * PI * 2_500.0 / sample_rate_hz).exp();
        Self {
            r1,
            omr1: 1.0 - r1,
            r2,
            omr2: 1.0 - r2,
        }
    }

    pub fn split_3band(&self, x: &[f64], bands: &mut [Vec<f64>; 3]) {
        let n = x.len();
        for b in bands.iter_mut() {
            b.resize(n, 0.0);
        }
        let (mut u1, mut u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        for t in 0..n {
            let xt = x[t];
            let l250 = step_bezout7_lp(xt, self.r1, self.omr1, &mut u1);
            let l2500 = step_bezout7_lp(xt, self.r2, self.omr2, &mut u2);
            bands[0][t] = l250;
            bands[1][t] = l2500 - l250;
            bands[2][t] = xt - l2500;
        }
    }

    pub fn accumulate_adjoint(
        &self,
        d_bands: &[Vec<f64>; 3],
        dx: &mut [f64],
        dy_scratch: &mut [Vec<f64>; 2],
    ) {
        let n = dx.len();
        for s in dy_scratch.iter_mut() {
            s.resize(n, 0.0);
        }
        for t in 0..n {
            let (dlf, dmf, dhf) = (d_bands[0][t], d_bands[1][t], d_bands[2][t]);
            dy_scratch[0][t] = dlf - dmf;
            dy_scratch[1][t] = dmf - dhf;
            dx[t] += dhf;
        }
        accumulate_bezout7_lp_adjoint(&dy_scratch[0], self.r1, self.omr1, dx);
        accumulate_bezout7_lp_adjoint(&dy_scratch[1], self.r2, self.omr2, dx);
    }
}

/// Multi-stem training dataset with precomputed Bézout `bESR` and 24-band Gabor summaries.
#[derive(Clone, Debug)]
pub struct V3TrainDataset {
    pub stems: Vec<V3TrainStem>,
    pub sample_rate_hz: f64,
    pub bank: Bezout7FilterBank,
    pub stem_band_weights: Vec<[[f64; 3]; 2]>,
    pub stem_weights: Vec<f64>,
    pub gabor_bank: AnalyticGaborBank,
    pub gabor_targets: Vec<[GaborTargetSummary; 2]>,
    pub gabor_env_weight: f64,
    pub gabor_ltas_weight: f64,
}

impl V3TrainDataset {
    #[must_use]
    pub fn new(stems: Vec<V3TrainStem>, sample_rate_hz: f64) -> Self {
        let weighted = stems.into_iter().map(|s| (s, 1.0)).collect();
        Self::new_weighted(weighted, sample_rate_hz)
    }

    #[must_use]
    pub fn new_weighted(weighted_stems: Vec<(V3TrainStem, f64)>, sample_rate_hz: f64) -> Self {
        let bank = Bezout7FilterBank::new(sample_rate_hz);
        let gabor_bank = AnalyticGaborBank::new(sample_rate_hz);
        let raw_sum: f64 = weighted_stems.iter().map(|(_, w)| w.max(0.0)).sum();
        let fallback_w = 1.0 / (weighted_stems.len().max(1) as f64);

        let mut stems = Vec::with_capacity(weighted_stems.len());
        let mut stem_weights = Vec::with_capacity(weighted_stems.len());
        let mut stem_band_weights = Vec::with_capacity(weighted_stems.len());
        let mut gabor_targets = Vec::with_capacity(weighted_stems.len());
        let mut bands = [Vec::new(), Vec::new(), Vec::new()];

        for (st, w_raw) in weighted_stems {
            let norm_w = if raw_sum > 0.0 {
                w_raw.max(0.0) / raw_sum
            } else {
                fallback_w
            };
            let n = st
                .tgt_l
                .len()
                .min(st.tgt_r.len())
                .min(st.in_l.len())
                .min(st.in_r.len());
            let w = st.warmup.min(n);
            let mut bw = [[0.0_f64; 3]; 2];
            let mut gab_ch = [GaborTargetSummary::default(), GaborTargetSummary::default()];
            for ch in 0..2 {
                let src = if ch == 0 {
                    &st.tgt_l[..n]
                } else {
                    &st.tgt_r[..n]
                };
                let x_f64: Vec<f64> = src.iter().map(|&v| v as f64).collect();
                bank.split_3band(&x_f64, &mut bands);
                let e_tot: f64 = x_f64[w..].iter().map(|&v| v * v).sum();
                for b in 0..3 {
                    let e_b: f64 = bands[b][w..].iter().map(|&v| v * v).sum();
                    bw[ch][b] = norm_w / (3.0 * (e_b.max(1.0e-9 * e_tot) + 1.0e-15));
                }
                gab_ch[ch] = build_gabor_target_summary(&x_f64, w, &gabor_bank);
            }
            stems.push(st);
            stem_weights.push(norm_w);
            stem_band_weights.push(bw);
            gabor_targets.push(gab_ch);
        }

        Self {
            stems,
            sample_rate_hz,
            bank,
            stem_band_weights,
            stem_weights,
            gabor_bank,
            gabor_targets,
            gabor_env_weight: 0.0,
            gabor_ltas_weight: 0.0,
        }
    }

    #[must_use]
    pub fn with_gabor_weights(&self, gabor_env_weight: f64, gabor_ltas_weight: f64) -> Self {
        let mut out = self.clone();
        out.gabor_env_weight = gabor_env_weight.max(0.0);
        out.gabor_ltas_weight = gabor_ltas_weight.max(0.0);
        out
    }

    /// Evaluates the closed-form 24-band Gabor envelope + log-LTAS loss and accumulates its exact
    /// overlap-add reverse adjoint into `grad_dy` when provided using backend `B`.
    pub fn eval_gabor_loss_and_adjoint_with_backend<B: TrainerBackend>(
        &self,
        stem_idx: usize,
        ch: usize,
        ch_scale: f64,
        y_mdl: &[f64],
        grad_dy: Option<&mut [f64]>,
        scratch: &mut GaborAdjointScratch,
    ) -> f64 {
        gabor_loss_and_adjoint_with_backend::<B>(
            &self.gabor_bank,
            &self.gabor_targets[stem_idx][ch],
            y_mdl,
            grad_dy,
            self.stem_weights[stem_idx],
            ch_scale,
            self.gabor_env_weight,
            self.gabor_ltas_weight,
            scratch,
        )
    }

    /// Evaluates the closed-form 24-band Gabor envelope + log-LTAS loss and accumulates its exact
    /// overlap-add reverse adjoint into `grad_dy` when provided.
    pub fn eval_gabor_loss_and_adjoint(
        &self,
        stem_idx: usize,
        ch: usize,
        ch_scale: f64,
        y_mdl: &[f64],
        grad_dy: Option<&mut [f64]>,
        scratch: &mut GaborAdjointScratch,
    ) -> f64 {
        struct GaborEval<'a> {
            ds: &'a V3TrainDataset,
            stem_idx: usize,
            ch: usize,
            ch_scale: f64,
            y_mdl: &'a [f64],
            grad_dy: Option<&'a mut [f64]>,
            scratch: &'a mut GaborAdjointScratch,
        }
        impl TrainerTask<f64> for GaborEval<'_> {
            fn run<B: TrainerBackend>(self) -> f64 {
                self.ds.eval_gabor_loss_and_adjoint_with_backend::<B>(
                    self.stem_idx,
                    self.ch,
                    self.ch_scale,
                    self.y_mdl,
                    self.grad_dy,
                    self.scratch,
                )
            }
        }
        DynamicTrainer::detect().dispatch(GaborEval {
            ds: self,
            stem_idx,
            ch,
            ch_scale,
            y_mdl,
            grad_dy,
            scratch,
        })
    }
}

/// Complete `f64` mirror of [`V3CascadeBlockCold`] (used both for exact `f64` L-BFGS iterates and gradients).
#[derive(Clone, Debug)]
pub struct V3OptBlockF64 {
    pub atk_alpha: f64,
    pub rel_alpha: f64,
    pub gate_mix: f64,
    pub gate_sharpness: f64,
    pub in_u_gain: [f64; MAX_CASCADE_LANES],
    pub inter_mat: [[f64; MAX_CASCADE_LANES]; MAX_CASCADE_LANES],
    pub lp_pole: [f64; MAX_CASCADE_LANES],
    pub lp_mix: [f64; MAX_CASCADE_LANES],
    pub hp_mix: [f64; MAX_CASCADE_LANES],
    pub static_bias: [f64; MAX_CASCADE_LANES],
    pub env_bias_sens: [f64; MAX_CASCADE_LANES],
    pub skip_u_gain: [f64; MAX_CASCADE_LANES],
    pub prev_skip_gain: [f64; MAX_CASCADE_LANES],
}

impl Default for V3OptBlockF64 {
    fn default() -> Self {
        Self {
            atk_alpha: 0.0,
            rel_alpha: 0.0,
            gate_mix: 0.0,
            gate_sharpness: 0.0,
            in_u_gain: [0.0; MAX_CASCADE_LANES],
            inter_mat: [[0.0; MAX_CASCADE_LANES]; MAX_CASCADE_LANES],
            lp_pole: [0.0; MAX_CASCADE_LANES],
            lp_mix: [0.0; MAX_CASCADE_LANES],
            hp_mix: [0.0; MAX_CASCADE_LANES],
            static_bias: [0.0; MAX_CASCADE_LANES],
            env_bias_sens: [0.0; MAX_CASCADE_LANES],
            skip_u_gain: [0.0; MAX_CASCADE_LANES],
            prev_skip_gain: [0.0; MAX_CASCADE_LANES],
        }
    }
}

impl V3OptBlockF64 {
    #[must_use]
    pub fn from_cold(b: &V3CascadeBlockCold) -> Self {
        let mut out = Self {
            atk_alpha: b.atk_alpha as f64,
            rel_alpha: b.rel_alpha as f64,
            gate_mix: b.gate_mix as f64,
            gate_sharpness: b.gate_sharpness as f64,
            ..Self::default()
        };
        for l in 0..MAX_CASCADE_LANES {
            out.in_u_gain[l] = b.in_u_gain[l] as f64;
            out.lp_pole[l] = b.lp_pole[l] as f64;
            out.lp_mix[l] = b.lp_mix[l] as f64;
            out.hp_mix[l] = b.hp_mix[l] as f64;
            out.static_bias[l] = b.static_bias[l] as f64;
            out.env_bias_sens[l] = b.env_bias_sens[l] as f64;
            out.skip_u_gain[l] = b.skip_u_gain[l] as f64;
            out.prev_skip_gain[l] = b.prev_skip_gain[l] as f64;
            for s in 0..MAX_CASCADE_LANES {
                out.inter_mat[l][s] = b.inter_mat[l][s] as f64;
            }
        }
        out
    }

    #[must_use]
    pub fn to_cold(&self) -> V3CascadeBlockCold {
        let mut out = V3CascadeBlockCold {
            atk_alpha: self.atk_alpha as f32,
            rel_alpha: self.rel_alpha as f32,
            gate_mix: self.gate_mix as f32,
            gate_sharpness: self.gate_sharpness as f32,
            ..V3CascadeBlockCold::default()
        };
        for l in 0..MAX_CASCADE_LANES {
            out.in_u_gain[l] = self.in_u_gain[l] as f32;
            out.lp_pole[l] = self.lp_pole[l] as f32;
            out.lp_mix[l] = self.lp_mix[l] as f32;
            out.hp_mix[l] = self.hp_mix[l] as f32;
            out.static_bias[l] = self.static_bias[l] as f32;
            out.env_bias_sens[l] = self.env_bias_sens[l] as f32;
            out.skip_u_gain[l] = self.skip_u_gain[l] as f32;
            out.prev_skip_gain[l] = self.prev_skip_gain[l] as f32;
            for s in 0..MAX_CASCADE_LANES {
                out.inter_mat[l][s] = self.inter_mat[l][s] as f32;
            }
        }
        out.clamp_in_place();
        out
    }
}

/// Complete `f64` mirror of [`V3ColdTierModel`] (used for smooth L-BFGS optimization AND exact gradients).
#[derive(Clone, Debug)]
pub struct V3OptModelF64 {
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub svf_base_g: f64,
    pub svf_sec2_g_ratio: f64,
    pub svf_env_oct_sens: f64,
    pub svf_env_comp: f64,
    pub svf_atk_alpha: f64,
    pub svf_rel_alpha: f64,
    pub svf_det_lp_pole: f64,
    pub svf_det_hp_mix: f64,
    pub svf_damp_k1: f64,
    pub svf_damp_k2: f64,
    pub svf_sec2_bp_feed: f64,
    pub svf_sec2_hp_feed: f64,
    pub svf_stereo_link: f64,
    pub svf_g_min: f64,
    pub svf_g_max: f64,
    pub svf_stage2_feed_taps: [f64; 7],
    pub svf_direct_taps_l: [f64; 7],
    pub svf_direct_taps_r: [f64; 7],
    pub blocks: [V3OptBlockF64; NUM_CASCADE_BLOCKS],
    pub pre_cab_u_gain: [f64; 2],
    pub pre_cab_w: [[f64; MAX_CASCADE_LANES]; 2],
    pub direct_s2_w: [[f64; MAX_CASCADE_LANES]; 2],
    pub fir_l: [f64; 64],
    pub fir_r: [f64; 64],
    pub kautz_poles: [(f64, f64); MAX_KAUTZ_POLES],
    pub kautz_c_l: [(f64, f64); MAX_KAUTZ_POLES],
    pub kautz_c_r: [(f64, f64); MAX_KAUTZ_POLES],
}

impl V3OptModelF64 {
    #[must_use]
    pub fn zeros(tier: Tier, os2x: bool, mono_out: bool) -> Self {
        Self {
            tier,
            os2x,
            mono_out,
            svf_base_g: 0.0,
            svf_sec2_g_ratio: 0.0,
            svf_env_oct_sens: 0.0,
            svf_env_comp: 0.0,
            svf_atk_alpha: 0.0,
            svf_rel_alpha: 0.0,
            svf_det_lp_pole: 0.0,
            svf_det_hp_mix: 0.0,
            svf_damp_k1: 0.0,
            svf_damp_k2: 0.0,
            svf_sec2_bp_feed: 0.0,
            svf_sec2_hp_feed: 0.0,
            svf_stereo_link: 0.0,
            svf_g_min: 0.0,
            svf_g_max: 0.0,
            svf_stage2_feed_taps: [0.0; 7],
            svf_direct_taps_l: [0.0; 7],
            svf_direct_taps_r: [0.0; 7],
            blocks: [
                V3OptBlockF64::default(),
                V3OptBlockF64::default(),
                V3OptBlockF64::default(),
            ],
            pre_cab_u_gain: [0.0; 2],
            pre_cab_w: [[0.0; MAX_CASCADE_LANES]; 2],
            direct_s2_w: [[0.0; MAX_CASCADE_LANES]; 2],
            fir_l: [0.0; 64],
            fir_r: [0.0; 64],
            kautz_poles: [(0.0, 0.0); MAX_KAUTZ_POLES],
            kautz_c_l: [(0.0, 0.0); MAX_KAUTZ_POLES],
            kautz_c_r: [(0.0, 0.0); MAX_KAUTZ_POLES],
        }
    }

    #[must_use]
    pub fn from_cold_model(m: &V3ColdTierModel) -> Self {
        let mut out = Self::zeros(m.tier, m.os2x, m.mono_out);
        let s = &m.svf;
        out.svf_base_g = s.base_g as f64;
        out.svf_sec2_g_ratio = s.sec2_g_ratio as f64;
        out.svf_env_oct_sens = s.env_oct_sens as f64;
        out.svf_env_comp = s.env_comp as f64;
        out.svf_atk_alpha = s.atk_alpha as f64;
        out.svf_rel_alpha = s.rel_alpha as f64;
        out.svf_det_lp_pole = s.det_lp_pole as f64;
        out.svf_det_hp_mix = s.det_hp_mix as f64;
        out.svf_damp_k1 = s.damp_k1 as f64;
        out.svf_damp_k2 = s.damp_k2 as f64;
        out.svf_sec2_bp_feed = s.sec2_bp_feed as f64;
        out.svf_sec2_hp_feed = s.sec2_hp_feed as f64;
        out.svf_stereo_link = s.stereo_link as f64;
        out.svf_g_min = s.g_min as f64;
        out.svf_g_max = s.g_max as f64;
        for i in 0..7 {
            out.svf_stage2_feed_taps[i] = s.stage2_feed_taps[i] as f64;
            out.svf_direct_taps_l[i] = s.direct_taps_l[i] as f64;
            out.svf_direct_taps_r[i] = s.direct_taps_r[i] as f64;
        }
        for b in 0..NUM_CASCADE_BLOCKS {
            out.blocks[b] = V3OptBlockF64::from_cold(&m.blocks[b]);
        }
        out.pre_cab_u_gain = [m.pre_cab_u_gain[0] as f64, m.pre_cab_u_gain[1] as f64];
        for l in 0..MAX_CASCADE_LANES {
            for ch in 0..2 {
                out.pre_cab_w[ch][l] = m.pre_cab_w[ch][l] as f64;
                out.direct_s2_w[ch][l] = m.direct_s2_w[ch][l] as f64;
            }
        }
        for k in 0..64 {
            out.fir_l[k] = m.fir_l[k] as f64;
            out.fir_r[k] = m.fir_r[k] as f64;
        }
        for k in 0..MAX_KAUTZ_POLES {
            out.kautz_poles[k] = (m.kautz_poles[k].0 as f64, m.kautz_poles[k].1 as f64);
            out.kautz_c_l[k] = (m.kautz_c_l[k].0 as f64, m.kautz_c_l[k].1 as f64);
            out.kautz_c_r[k] = (m.kautz_c_r[k].0 as f64, m.kautz_c_r[k].1 as f64);
        }
        out
    }

    #[must_use]
    pub fn to_cold_model(&self) -> V3ColdTierModel {
        let mut svf = V3Stage1SvfParams {
            base_g: self.svf_base_g as f32,
            sec2_g_ratio: self.svf_sec2_g_ratio as f32,
            env_oct_sens: self.svf_env_oct_sens as f32,
            env_comp: self.svf_env_comp as f32,
            atk_alpha: self.svf_atk_alpha as f32,
            rel_alpha: self.svf_rel_alpha as f32,
            det_lp_pole: self.svf_det_lp_pole as f32,
            det_hp_mix: self.svf_det_hp_mix as f32,
            damp_k1: self.svf_damp_k1 as f32,
            damp_k2: self.svf_damp_k2 as f32,
            sec2_bp_feed: self.svf_sec2_bp_feed as f32,
            sec2_hp_feed: self.svf_sec2_hp_feed as f32,
            stereo_link: self.svf_stereo_link as f32,
            g_min: self.svf_g_min as f32,
            g_max: self.svf_g_max as f32,
            _pad: 0.0,
            stage2_feed_taps: [0.0; 8],
            direct_taps_l: [0.0; 8],
            direct_taps_r: [0.0; 8],
        };
        for i in 0..7 {
            svf.stage2_feed_taps[i] = self.svf_stage2_feed_taps[i] as f32;
            svf.direct_taps_l[i] = self.svf_direct_taps_l[i] as f32;
            svf.direct_taps_r[i] = self.svf_direct_taps_r[i] as f32;
        }
        svf.clamp_in_place();

        let mut m = V3ColdTierModel::identity_for_tier(self.tier, self.os2x, self.mono_out);
        m.svf = svf;
        m.blocks = [
            self.blocks[0].to_cold(),
            self.blocks[1].to_cold(),
            self.blocks[2].to_cold(),
        ];
        m.pre_cab_u_gain = [self.pre_cab_u_gain[0] as f32, self.pre_cab_u_gain[1] as f32];
        let c_cnt = tier_cascade_lanes(self.tier);
        for ch in 0..2 {
            let eff_ch = if self.mono_out { 0 } else { ch };
            for l in 0..c_cnt {
                m.pre_cab_w[ch][l] = self.pre_cab_w[eff_ch][l] as f32;
                m.direct_s2_w[ch][l] = self.direct_s2_w[eff_ch][l] as f32;
            }
        }
        for k in 0..64 {
            m.fir_l[k] = self.fir_l[k] as f32;
            m.fir_r[k] = if self.mono_out {
                self.fir_l[k] as f32
            } else {
                self.fir_r[k] as f32
            };
        }
        for k in 0..MAX_KAUTZ_POLES {
            m.kautz_poles[k] = (
                (self.kautz_poles[k].0 as f32).clamp(0.50, 0.997),
                (self.kautz_poles[k].1 as f32).clamp(1.0e-3, std::f32::consts::PI - 1.0e-3),
            );
            m.kautz_c_l[k] = (self.kautz_c_l[k].0 as f32, self.kautz_c_l[k].1 as f32);
            m.kautz_c_r[k] = if self.mono_out {
                m.kautz_c_l[k]
            } else {
                (self.kautz_c_r[k].0 as f32, self.kautz_c_r[k].1 as f32)
            };
        }
        m
    }

    /// Convex interpolation `(1 - a) * self + a * target` across all continuous parameters
    /// for monotone blending at identical oversampling rates (`os2x`).
    #[must_use]
    pub fn blend_same_os(&self, target: &Self, a: f64) -> Self {
        let mut out = self.clone();
        let l = |x: f64, y: f64| x + a * (y - x);
        let lerp_s = |dst: &mut [f64], src: &[f64]| {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += a * (s - *d);
            }
        };
        out.svf_base_g = l(out.svf_base_g, target.svf_base_g);
        out.svf_sec2_g_ratio = l(out.svf_sec2_g_ratio, target.svf_sec2_g_ratio);
        out.svf_env_oct_sens = l(out.svf_env_oct_sens, target.svf_env_oct_sens);
        out.svf_env_comp = l(out.svf_env_comp, target.svf_env_comp);
        out.svf_atk_alpha = l(out.svf_atk_alpha, target.svf_atk_alpha);
        out.svf_rel_alpha = l(out.svf_rel_alpha, target.svf_rel_alpha);
        out.svf_det_lp_pole = l(out.svf_det_lp_pole, target.svf_det_lp_pole);
        out.svf_det_hp_mix = l(out.svf_det_hp_mix, target.svf_det_hp_mix);
        out.svf_damp_k1 = l(out.svf_damp_k1, target.svf_damp_k1);
        out.svf_damp_k2 = l(out.svf_damp_k2, target.svf_damp_k2);
        out.svf_sec2_bp_feed = l(out.svf_sec2_bp_feed, target.svf_sec2_bp_feed);
        out.svf_sec2_hp_feed = l(out.svf_sec2_hp_feed, target.svf_sec2_hp_feed);
        out.svf_stereo_link = l(out.svf_stereo_link, target.svf_stereo_link);
        out.svf_g_min = l(out.svf_g_min, target.svf_g_min);
        out.svf_g_max = l(out.svf_g_max, target.svf_g_max);
        lerp_s(&mut out.svf_stage2_feed_taps, &target.svf_stage2_feed_taps);
        lerp_s(&mut out.svf_direct_taps_l, &target.svf_direct_taps_l);
        lerp_s(&mut out.svf_direct_taps_r, &target.svf_direct_taps_r);
        for (db, sb) in out.blocks.iter_mut().zip(&target.blocks) {
            db.atk_alpha = l(db.atk_alpha, sb.atk_alpha);
            db.rel_alpha = l(db.rel_alpha, sb.rel_alpha);
            db.gate_mix = l(db.gate_mix, sb.gate_mix);
            db.gate_sharpness = l(db.gate_sharpness, sb.gate_sharpness);
            lerp_s(&mut db.in_u_gain, &sb.in_u_gain);
            lerp_s(&mut db.lp_pole, &sb.lp_pole);
            lerp_s(&mut db.lp_mix, &sb.lp_mix);
            lerp_s(&mut db.hp_mix, &sb.hp_mix);
            lerp_s(&mut db.static_bias, &sb.static_bias);
            lerp_s(&mut db.env_bias_sens, &sb.env_bias_sens);
            lerp_s(&mut db.skip_u_gain, &sb.skip_u_gain);
            lerp_s(&mut db.prev_skip_gain, &sb.prev_skip_gain);
            for (dr, sr) in db.inter_mat.iter_mut().zip(&sb.inter_mat) {
                lerp_s(dr, sr);
            }
        }
        lerp_s(&mut out.pre_cab_u_gain, &target.pre_cab_u_gain);
        for ch in 0..2 {
            lerp_s(&mut out.pre_cab_w[ch], &target.pre_cab_w[ch]);
            lerp_s(&mut out.direct_s2_w[ch], &target.direct_s2_w[ch]);
        }
        lerp_s(&mut out.fir_l, &target.fir_l);
        lerp_s(&mut out.fir_r, &target.fir_r);
        for m in 0..MAX_KAUTZ_POLES {
            out.kautz_poles[m] = (
                l(out.kautz_poles[m].0, target.kautz_poles[m].0),
                l(out.kautz_poles[m].1, target.kautz_poles[m].1),
            );
            out.kautz_c_l[m] = (
                l(out.kautz_c_l[m].0, target.kautz_c_l[m].0),
                l(out.kautz_c_l[m].1, target.kautz_c_l[m].1),
            );
            out.kautz_c_r[m] = (
                l(out.kautz_c_r[m].0, target.kautz_c_r[m].0),
                l(out.kautz_c_r[m].1, target.kautz_c_r[m].1),
            );
        }
        out
    }
}
