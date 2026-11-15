#![forbid(unsafe_code)]
//! Double-precision Matryoshka `v3` Parameterization, Bezout-7 Filter Bank, and Dataset Containers.

use std::f64::consts::PI;

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS,
    NUM_IN_FEATS_B0, V3InputFeaturesParams, V3OversamplerParams,
};
use dnam_engine::stages::v3_core::{V3GruBlockCold, V3MatryoshkaModel, V3SvfParams, V3TierReadout};

pub use super::bezout7::Bezout7FilterBank;
pub use super::dataset::{V3TrainDataset, V3TrainStem};

/// Double-precision mirror of [`V3GruBlockCold`].
#[derive(Clone, Debug, PartialEq)]
pub struct V3GruBlockF64 {
    pub w_in_z: [[f64; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    pub w_in_r: [[f64; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    pub w_in_n: [[f64; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    pub w_prev_z: [[f64; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    pub w_prev_r: [[f64; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    pub w_prev_n: [[f64; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    pub w_rec_z: [[f64; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    pub w_rec_r: [[f64; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    pub w_rec_n: [[f64; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    pub b_z: [f64; MAX_BLOCK_LANES],
    pub b_r: [f64; MAX_BLOCK_LANES],
    pub b_n: [f64; MAX_BLOCK_LANES],
    pub k_sag: [f64; MAX_BLOCK_LANES],
    pub k_quad: [f64; MAX_BLOCK_LANES],
}

impl Default for V3GruBlockF64 {
    fn default() -> Self {
        Self {
            w_in_z: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_in_r: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_in_n: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_prev_z: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_prev_r: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_prev_n: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_z: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_r: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_n: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            b_z: [0.0; MAX_BLOCK_LANES],
            b_r: [1.5; MAX_BLOCK_LANES],
            b_n: [0.0; MAX_BLOCK_LANES],
            k_sag: [0.0; MAX_BLOCK_LANES],
            k_quad: [0.0; MAX_BLOCK_LANES],
        }
    }
}

impl V3GruBlockF64 {
    /// Returns an all-zero block (suitable as reverse-mode gradient accumulator).
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            b_r: [0.0; MAX_BLOCK_LANES],
            ..Self::default()
        }
    }

    /// Zeros strictly upper-triangular group entries so lower tiers remain causal prefixes.
    pub fn enforce_lower_triangular(&mut self) {
        for dst in 0..MAX_BLOCK_LANES {
            let dst_g = dst >> 2;
            for src in 0..MAX_BLOCK_LANES {
                if (src >> 2) > dst_g {
                    self.w_prev_z[dst][src] = 0.0;
                    self.w_prev_r[dst][src] = 0.0;
                    self.w_prev_n[dst][src] = 0.0;
                    self.w_rec_z[dst][src] = 0.0;
                    self.w_rec_r[dst][src] = 0.0;
                    self.w_rec_n[dst][src] = 0.0;
                }
            }
        }
    }

    /// Converts from cold `f32` engine representation.
    #[must_use]
    pub fn from_cold(c: &V3GruBlockCold) -> Self {
        let mut out = Self::zeros();
        for f in 0..NUM_IN_FEATS_B0 {
            for l in 0..MAX_BLOCK_LANES {
                out.w_in_z[f][l] = f64::from(c.w_in_z[f][l]);
                out.w_in_r[f][l] = f64::from(c.w_in_r[f][l]);
                out.w_in_n[f][l] = f64::from(c.w_in_n[f][l]);
            }
        }
        for d in 0..MAX_BLOCK_LANES {
            out.b_z[d] = f64::from(c.b_z[d]);
            out.b_r[d] = f64::from(c.b_r[d]);
            out.b_n[d] = f64::from(c.b_n[d]);
            out.k_sag[d] = f64::from(c.k_sag[d]);
            out.k_quad[d] = f64::from(c.k_quad[d]);
            for s in 0..MAX_BLOCK_LANES {
                out.w_prev_z[d][s] = f64::from(c.w_prev_z[d][s]);
                out.w_prev_r[d][s] = f64::from(c.w_prev_r[d][s]);
                out.w_prev_n[d][s] = f64::from(c.w_prev_n[d][s]);
                out.w_rec_z[d][s] = f64::from(c.w_rec_z[d][s]);
                out.w_rec_r[d][s] = f64::from(c.w_rec_r[d][s]);
                out.w_rec_n[d][s] = f64::from(c.w_rec_n[d][s]);
            }
        }
        out.enforce_lower_triangular();
        out
    }

    /// Accumulates another block (e.g., per-stem gradient) element-wise into `self`.
    pub fn add_assign(&mut self, rhs: &Self) {
        for f in 0..NUM_IN_FEATS_B0 {
            for l in 0..MAX_BLOCK_LANES {
                self.w_in_z[f][l] += rhs.w_in_z[f][l];
                self.w_in_r[f][l] += rhs.w_in_r[f][l];
                self.w_in_n[f][l] += rhs.w_in_n[f][l];
            }
        }
        for d in 0..MAX_BLOCK_LANES {
            self.b_z[d] += rhs.b_z[d];
            self.b_r[d] += rhs.b_r[d];
            self.b_n[d] += rhs.b_n[d];
            self.k_sag[d] += rhs.k_sag[d];
            self.k_quad[d] += rhs.k_quad[d];
            for s in 0..MAX_BLOCK_LANES {
                self.w_prev_z[d][s] += rhs.w_prev_z[d][s];
                self.w_prev_r[d][s] += rhs.w_prev_r[d][s];
                self.w_prev_n[d][s] += rhs.w_prev_n[d][s];
                self.w_rec_z[d][s] += rhs.w_rec_z[d][s];
                self.w_rec_r[d][s] += rhs.w_rec_r[d][s];
                self.w_rec_n[d][s] += rhs.w_rec_n[d][s];
            }
        }
    }

    /// Converts to clamped `f32` engine representation.
    #[must_use]
    pub fn to_cold(&self) -> V3GruBlockCold {
        let mut out = V3GruBlockCold::default();
        for f in 0..NUM_IN_FEATS_B0 {
            for l in 0..MAX_BLOCK_LANES {
                out.w_in_z[f][l] = self.w_in_z[f][l] as f32;
                out.w_in_r[f][l] = self.w_in_r[f][l] as f32;
                out.w_in_n[f][l] = self.w_in_n[f][l] as f32;
            }
        }
        for d in 0..MAX_BLOCK_LANES {
            out.b_z[d] = self.b_z[d].clamp(-8.0, 8.2) as f32;
            out.b_r[d] = self.b_r[d].clamp(-8.0, 8.2) as f32;
            out.b_n[d] = self.b_n[d].clamp(-3.8, 3.8) as f32;
            out.k_sag[d] = self.k_sag[d].clamp(0.0, 8.0) as f32;
            out.k_quad[d] = self.k_quad[d].clamp(-0.45, 0.45) as f32;
            for s in 0..MAX_BLOCK_LANES {
                out.w_prev_z[d][s] = self.w_prev_z[d][s] as f32;
                out.w_prev_r[d][s] = self.w_prev_r[d][s] as f32;
                out.w_prev_n[d][s] = self.w_prev_n[d][s] as f32;
                out.w_rec_z[d][s] = self.w_rec_z[d][s] as f32;
                out.w_rec_r[d][s] = self.w_rec_r[d][s] as f32;
                out.w_rec_n[d][s] = self.w_rec_n[d][s] as f32;
            }
        }
        out.clamp_in_place();
        out
    }
}

/// Double-precision mirror of [`V3TierReadout`].
#[derive(Clone, Debug, PartialEq)]
pub struct V3TierReadoutF64 {
    pub svf_direct_taps_l: [f64; 7],
    pub svf_direct_taps_r: [f64; 7],
    pub pre_cab_u_gain: [f64; 2],
    pub pre_cab_hp_gain: [f64; 2],
    pub pre_cab_w: [[f64; MAX_TOTAL_GRU_LANES]; 2],
    pub direct_s2_u_gain: [f64; 2],
    pub direct_s2_hp_gain: [f64; 2],
    pub direct_s2_w: [[f64; MAX_TOTAL_GRU_LANES]; 2],
    pub fir_l: [f64; 64],
    pub fir_r: [f64; 64],
    pub kautz_c_l: [(f64, f64); MAX_KAUTZ_POLES],
    pub kautz_c_r: [(f64, f64); MAX_KAUTZ_POLES],
}

impl Default for V3TierReadoutF64 {
    fn default() -> Self {
        let mut r = Self::zeros();
        r.pre_cab_u_gain = [1.0, 1.0];
        r.fir_l[0] = 1.0;
        r.fir_r[0] = 1.0;
        r
    }
}

impl V3TierReadoutF64 {
    /// Returns an all-zero readout head (for gradient accumulation).
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            svf_direct_taps_l: [0.0; 7],
            svf_direct_taps_r: [0.0; 7],
            pre_cab_u_gain: [0.0; 2],
            pre_cab_hp_gain: [0.0; 2],
            pre_cab_w: [[0.0; MAX_TOTAL_GRU_LANES]; 2],
            direct_s2_u_gain: [0.0; 2],
            direct_s2_hp_gain: [0.0; 2],
            direct_s2_w: [[0.0; MAX_TOTAL_GRU_LANES]; 2],
            fir_l: [0.0; 64],
            fir_r: [0.0; 64],
            kautz_c_l: [(0.0, 0.0); MAX_KAUTZ_POLES],
            kautz_c_r: [(0.0, 0.0); MAX_KAUTZ_POLES],
        }
    }

    /// Converts from `f32` engine readout head.
    #[must_use]
    pub fn from_cold(r: &V3TierReadout) -> Self {
        let mut out = Self::zeros();
        for i in 0..7 {
            out.svf_direct_taps_l[i] = f64::from(r.svf_direct_taps_l[i]);
            out.svf_direct_taps_r[i] = f64::from(r.svf_direct_taps_r[i]);
        }
        for ch in 0..2 {
            out.pre_cab_u_gain[ch] = f64::from(r.pre_cab_u_gain[ch]);
            out.pre_cab_hp_gain[ch] = f64::from(r.pre_cab_hp_gain[ch]);
            out.direct_s2_u_gain[ch] = f64::from(r.direct_s2_u_gain[ch]);
            out.direct_s2_hp_gain[ch] = f64::from(r.direct_s2_hp_gain[ch]);
            for i in 0..MAX_TOTAL_GRU_LANES {
                out.pre_cab_w[ch][i] = f64::from(r.pre_cab_w[ch][i]);
                out.direct_s2_w[ch][i] = f64::from(r.direct_s2_w[ch][i]);
            }
        }
        for i in 0..64 {
            out.fir_l[i] = f64::from(r.fir_l[i]);
            out.fir_r[i] = f64::from(r.fir_r[i]);
        }
        for m in 0..MAX_KAUTZ_POLES {
            out.kautz_c_l[m] = (f64::from(r.kautz_c_l[m].0), f64::from(r.kautz_c_l[m].1));
            out.kautz_c_r[m] = (f64::from(r.kautz_c_r[m].0), f64::from(r.kautz_c_r[m].1));
        }
        out
    }

    /// Converts to masked `f32` engine readout head for `tier`.
    #[must_use]
    pub fn to_cold(&self, tier: Tier) -> V3TierReadout {
        let mut out = V3TierReadout::zeros();
        for i in 0..7 {
            out.svf_direct_taps_l[i] = self.svf_direct_taps_l[i] as f32;
            out.svf_direct_taps_r[i] = self.svf_direct_taps_r[i] as f32;
        }
        for ch in 0..2 {
            out.pre_cab_u_gain[ch] = self.pre_cab_u_gain[ch] as f32;
            out.pre_cab_hp_gain[ch] = self.pre_cab_hp_gain[ch] as f32;
            out.direct_s2_u_gain[ch] = self.direct_s2_u_gain[ch] as f32;
            out.direct_s2_hp_gain[ch] = self.direct_s2_hp_gain[ch] as f32;
            for i in 0..MAX_TOTAL_GRU_LANES {
                out.pre_cab_w[ch][i] = self.pre_cab_w[ch][i] as f32;
                out.direct_s2_w[ch][i] = self.direct_s2_w[ch][i] as f32;
            }
        }
        for i in 0..64 {
            out.fir_l[i] = self.fir_l[i] as f32;
            out.fir_r[i] = self.fir_r[i] as f32;
        }
        for m in 0..MAX_KAUTZ_POLES {
            out.kautz_c_l[m] = (self.kautz_c_l[m].0 as f32, self.kautz_c_l[m].1 as f32);
            out.kautz_c_r[m] = (self.kautz_c_r[m].0 as f32, self.kautz_c_r[m].1 as f32);
        }
        out.mask_to_tier(tier);
        out
    }

    /// Accumulates another readout (e.g., per-stem gradient) element-wise into `self`.
    pub fn add_assign(&mut self, rhs: &Self) {
        for i in 0..7 {
            self.svf_direct_taps_l[i] += rhs.svf_direct_taps_l[i];
            self.svf_direct_taps_r[i] += rhs.svf_direct_taps_r[i];
        }
        for ch in 0..2 {
            self.pre_cab_u_gain[ch] += rhs.pre_cab_u_gain[ch];
            self.pre_cab_hp_gain[ch] += rhs.pre_cab_hp_gain[ch];
            self.direct_s2_u_gain[ch] += rhs.direct_s2_u_gain[ch];
            self.direct_s2_hp_gain[ch] += rhs.direct_s2_hp_gain[ch];
            for i in 0..MAX_TOTAL_GRU_LANES {
                self.pre_cab_w[ch][i] += rhs.pre_cab_w[ch][i];
                self.direct_s2_w[ch][i] += rhs.direct_s2_w[ch][i];
            }
        }
        for i in 0..64 {
            self.fir_l[i] += rhs.fir_l[i];
            self.fir_r[i] += rhs.fir_r[i];
        }
        for m in 0..MAX_KAUTZ_POLES {
            self.kautz_c_l[m].0 += rhs.kautz_c_l[m].0;
            self.kautz_c_l[m].1 += rhs.kautz_c_l[m].1;
            self.kautz_c_r[m].0 += rhs.kautz_c_r[m].0;
            self.kautz_c_r[m].1 += rhs.kautz_c_r[m].1;
        }
    }
}

/// Complete double-precision single-run 4-tier Matryoshka `v3` optimization state.
#[derive(Clone, Debug, PartialEq)]
pub struct V3MatryoshkaOptF64 {
    pub os2x: bool,
    pub mono_out: bool,
    pub max_active_groups: usize,
    pub active_tier_weights: [f64; 4],
    pub os_up_quad: f64,
    pub os_dec_c_prev1: f64,
    pub os_dec_c0: f64,
    pub os_up_pre_emph: f64,
    pub os_dec_aa_mix: f64,
    pub os_up_aa_mix: f64,
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
    pub feat_in_hp_beta: f64,
    pub feat_fast_atk_beta: f64,
    pub feat_fast_rel_beta: f64,
    pub feat_pinch_k: f64,
    pub feat_slow_atk_beta: f64,
    pub feat_slow_rel_beta: f64,
    pub feat_slow_det_comp: f64,
    pub blocks: [V3GruBlockF64; NUM_GRU_BLOCKS],
    pub kautz_poles: [(f64, f64); MAX_KAUTZ_POLES],
    pub readouts: [V3TierReadoutF64; 4],
}

impl V3MatryoshkaOptF64 {
    /// Returns an all-zero gradient accumulator with matching structural flags and weights.
    #[must_use]
    pub fn zero_like(&self) -> Self {
        Self {
            os2x: self.os2x,
            mono_out: self.mono_out,
            max_active_groups: self.max_active_groups,
            active_tier_weights: self.active_tier_weights,
            os_up_quad: 0.0,
            os_dec_c_prev1: 0.0,
            os_dec_c0: 0.0,
            os_up_pre_emph: 0.0,
            os_dec_aa_mix: 0.0,
            os_up_aa_mix: 0.0,
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
            feat_in_hp_beta: 0.0,
            feat_fast_atk_beta: 0.0,
            feat_fast_rel_beta: 0.0,
            feat_pinch_k: 0.0,
            feat_slow_atk_beta: 0.0,
            feat_slow_rel_beta: 0.0,
            feat_slow_det_comp: 0.0,
            blocks: std::array::from_fn(|_| V3GruBlockF64::zeros()),
            kautz_poles: [(0.0, 0.0); MAX_KAUTZ_POLES],
            readouts: std::array::from_fn(|_| V3TierReadoutF64::zeros()),
        }
    }

    /// Accumulates another gradient struct deterministically into `self`.
    pub fn add_assign_grad(&mut self, rhs: &Self) {
        self.os_up_quad += rhs.os_up_quad;
        self.os_dec_c_prev1 += rhs.os_dec_c_prev1;
        self.os_dec_c0 += rhs.os_dec_c0;
        self.os_up_pre_emph += rhs.os_up_pre_emph;
        self.os_dec_aa_mix += rhs.os_dec_aa_mix;
        self.os_up_aa_mix += rhs.os_up_aa_mix;
        self.svf_base_g += rhs.svf_base_g;
        self.svf_sec2_g_ratio += rhs.svf_sec2_g_ratio;
        self.svf_env_oct_sens += rhs.svf_env_oct_sens;
        self.svf_env_comp += rhs.svf_env_comp;
        self.svf_atk_alpha += rhs.svf_atk_alpha;
        self.svf_rel_alpha += rhs.svf_rel_alpha;
        self.svf_det_lp_pole += rhs.svf_det_lp_pole;
        self.svf_det_hp_mix += rhs.svf_det_hp_mix;
        self.svf_damp_k1 += rhs.svf_damp_k1;
        self.svf_damp_k2 += rhs.svf_damp_k2;
        self.svf_sec2_bp_feed += rhs.svf_sec2_bp_feed;
        self.svf_sec2_hp_feed += rhs.svf_sec2_hp_feed;
        self.svf_stereo_link += rhs.svf_stereo_link;
        self.svf_g_min += rhs.svf_g_min;
        self.svf_g_max += rhs.svf_g_max;
        for i in 0..7 {
            self.svf_stage2_feed_taps[i] += rhs.svf_stage2_feed_taps[i];
        }
        self.feat_in_hp_beta += rhs.feat_in_hp_beta;
        self.feat_fast_atk_beta += rhs.feat_fast_atk_beta;
        self.feat_fast_rel_beta += rhs.feat_fast_rel_beta;
        self.feat_pinch_k += rhs.feat_pinch_k;
        self.feat_slow_atk_beta += rhs.feat_slow_atk_beta;
        self.feat_slow_rel_beta += rhs.feat_slow_rel_beta;
        self.feat_slow_det_comp += rhs.feat_slow_det_comp;
        for b in 0..NUM_GRU_BLOCKS {
            self.blocks[b].add_assign(&rhs.blocks[b]);
        }
        for m in 0..MAX_KAUTZ_POLES {
            self.kautz_poles[m].0 += rhs.kautz_poles[m].0;
            self.kautz_poles[m].1 += rhs.kautz_poles[m].1;
        }
        for tau in 0..4 {
            self.readouts[tau].add_assign(&rhs.readouts[tau]);
        }
    }

    /// Copies left-channel readout weights to right channel when `mono_out` is enabled.
    pub fn mirror_mono(&mut self) {
        if !self.mono_out {
            return;
        }
        for r in &mut self.readouts {
            r.svf_direct_taps_r = r.svf_direct_taps_l;
            r.pre_cab_u_gain[1] = r.pre_cab_u_gain[0];
            r.pre_cab_hp_gain[1] = r.pre_cab_hp_gain[0];
            r.pre_cab_w[1] = r.pre_cab_w[0];
            r.direct_s2_u_gain[1] = r.direct_s2_u_gain[0];
            r.direct_s2_hp_gain[1] = r.direct_s2_hp_gain[0];
            r.direct_s2_w[1] = r.direct_s2_w[0];
            r.fir_r = r.fir_l;
            r.kautz_c_r = r.kautz_c_l;
        }
    }

    /// Converts from cold `f32` [`V3MatryoshkaModel`].
    #[must_use]
    pub fn from_matryoshka_cold(m: &V3MatryoshkaModel) -> Self {
        Self {
            os2x: m.os2x,
            mono_out: m.mono_out,
            max_active_groups: MAX_GROUPS,
            active_tier_weights: [0.15, 0.20, 0.28, 0.37],
            os_up_quad: f64::from(m.os_params.up_quad_coeff),
            os_dec_c_prev1: f64::from(m.os_params.dec_c_prev1),
            os_dec_c0: f64::from(m.os_params.dec_c0),
            os_up_pre_emph: f64::from(m.os_params.up_pre_emph),
            os_dec_aa_mix: f64::from(m.os_params.dec_aa_mix),
            os_up_aa_mix: f64::from(m.os_params.up_aa_mix),
            svf_base_g: f64::from(m.svf.base_g),
            svf_sec2_g_ratio: f64::from(m.svf.sec2_g_ratio),
            svf_env_oct_sens: f64::from(m.svf.env_oct_sens),
            svf_env_comp: f64::from(m.svf.env_comp),
            svf_atk_alpha: f64::from(m.svf.atk_alpha),
            svf_rel_alpha: f64::from(m.svf.rel_alpha),
            svf_det_lp_pole: f64::from(m.svf.det_lp_pole),
            svf_det_hp_mix: f64::from(m.svf.det_hp_mix),
            svf_damp_k1: f64::from(m.svf.damp_k1),
            svf_damp_k2: f64::from(m.svf.damp_k2),
            svf_sec2_bp_feed: f64::from(m.svf.sec2_bp_feed),
            svf_sec2_hp_feed: f64::from(m.svf.sec2_hp_feed),
            svf_stereo_link: f64::from(m.svf.stereo_link),
            svf_g_min: f64::from(m.svf.g_min),
            svf_g_max: f64::from(m.svf.g_max),
            svf_stage2_feed_taps: std::array::from_fn(|i| f64::from(m.svf.stage2_feed_taps[i])),
            feat_in_hp_beta: f64::from(m.feat_params.in_hp_beta),
            feat_fast_atk_beta: f64::from(m.feat_params.fast_atk_beta),
            feat_fast_rel_beta: f64::from(m.feat_params.fast_rel_beta),
            feat_pinch_k: f64::from(m.feat_params.pinch_k),
            feat_slow_atk_beta: f64::from(m.feat_params.slow_atk_beta),
            feat_slow_rel_beta: f64::from(m.feat_params.slow_rel_beta),
            feat_slow_det_comp: f64::from(m.feat_params.slow_det_comp),
            blocks: [
                V3GruBlockF64::from_cold(&m.blocks[0]),
                V3GruBlockF64::from_cold(&m.blocks[1]),
                V3GruBlockF64::from_cold(&m.blocks[2]),
            ],
            kautz_poles: m.kautz_poles.map(|(r, th)| (f64::from(r), f64::from(th))),
            readouts: [
                V3TierReadoutF64::from_cold(&m.readouts[0]),
                V3TierReadoutF64::from_cold(&m.readouts[1]),
                V3TierReadoutF64::from_cold(&m.readouts[2]),
                V3TierReadoutF64::from_cold(&m.readouts[3]),
            ],
        }
    }

    /// Converts into clamped `f32` engine [`V3MatryoshkaModel`].
    #[must_use]
    pub fn to_matryoshka_cold(&self) -> V3MatryoshkaModel {
        let mut os_params = V3OversamplerParams {
            up_quad_coeff: self.os_up_quad as f32,
            dec_c_prev1: self.os_dec_c_prev1 as f32,
            dec_c0: self.os_dec_c0 as f32,
            dec_c1: (1.0 - self.os_dec_c_prev1 - self.os_dec_c0) as f32,
            up_pre_emph: self.os_up_pre_emph as f32,
            dec_aa_mix: self.os_dec_aa_mix as f32,
            up_aa_mix: self.os_up_aa_mix as f32,
            _pad: [0.0; 1],
        };
        os_params.clamp_in_place();

        let mut svf = V3SvfParams::identity_bypass();
        svf.base_g = self.svf_base_g as f32;
        svf.sec2_g_ratio = self.svf_sec2_g_ratio as f32;
        svf.env_oct_sens = self.svf_env_oct_sens as f32;
        svf.env_comp = self.svf_env_comp as f32;
        svf.atk_alpha = self.svf_atk_alpha as f32;
        svf.rel_alpha = self.svf_rel_alpha as f32;
        svf.det_lp_pole = self.svf_det_lp_pole as f32;
        svf.det_hp_mix = self.svf_det_hp_mix as f32;
        svf.damp_k1 = self.svf_damp_k1 as f32;
        svf.damp_k2 = self.svf_damp_k2 as f32;
        svf.sec2_bp_feed = self.svf_sec2_bp_feed as f32;
        svf.sec2_hp_feed = self.svf_sec2_hp_feed as f32;
        svf.stereo_link = self.svf_stereo_link as f32;
        svf.g_min = self.svf_g_min as f32;
        svf.g_max = self.svf_g_max as f32;
        for i in 0..7 {
            svf.stage2_feed_taps[i] = self.svf_stage2_feed_taps[i] as f32;
        }
        svf.clamp_in_place();

        let mut feat_params = V3InputFeaturesParams {
            in_hp_beta: self.feat_in_hp_beta as f32,
            fast_atk_beta: self.feat_fast_atk_beta as f32,
            fast_rel_beta: self.feat_fast_rel_beta as f32,
            pinch_k: self.feat_pinch_k as f32,
            slow_atk_beta: self.feat_slow_atk_beta as f32,
            slow_rel_beta: self.feat_slow_rel_beta as f32,
            slow_det_comp: self.feat_slow_det_comp as f32,
            _pad: 0.0,
        };
        feat_params.clamp_in_place();

        V3MatryoshkaModel {
            os2x: self.os2x,
            mono_out: self.mono_out,
            os_params,
            svf,
            feat_params,
            blocks: self.blocks.each_ref().map(V3GruBlockF64::to_cold),
            kautz_poles: self.kautz_poles.map(|(r, th)| {
                (
                    r.clamp(0.25, 0.9985) as f32,
                    th.clamp(0.0, PI - 1.0e-3) as f32,
                )
            }),
            readouts: [
                self.readouts[0].to_cold(Tier::Lite),
                self.readouts[1].to_cold(Tier::Full),
                self.readouts[2].to_cold(Tier::Pro),
                self.readouts[3].to_cold(Tier::Ultra),
            ],
        }
    }
}
