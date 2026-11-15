//! Stage III Early FIR + 32-Pole Kautz Modal Bank and Nested Matryoshka Cold Models (`v3` Core).

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_vec4, sqrt_f64};
use crate::operators::op_m::KautzModalBank;
use crate::stages::v3_core::cascade::{
    FIR_RING_CAP, FIR_RING_MASK, LOWER_TRI_TILES, MAX_BLOCK_LANES, MAX_FIR_SEGMENTS, MAX_GROUPS,
    MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, V3GruBlockCold,
    V3GruGroupHot, V3InputFeaturesParams, V3InputFeaturesState, V3OversamplerParams,
    ZeroLagOversampler2x, flat_gru_lane_index, tier_fir_segments, tier_groups, tier_kautz_banks,
};
use crate::stages::v3_core::svf::{V3Stage1SvfParams, V3Stage1SvfState};
use crate::types::Tier;

/// Per-tier compact linear readout head over shared SVF taps, Stage II GRU lanes, FIR, and Kautz.
#[derive(Clone, Debug, PartialEq)]
pub struct V3TierReadout {
    pub svf_direct_taps_l: [f32; 8],
    pub svf_direct_taps_r: [f32; 8],
    pub pre_cab_u_gain: [f32; 2],
    pub pre_cab_hp_gain: [f32; 2],
    pub pre_cab_w: [[f32; MAX_TOTAL_GRU_LANES]; 2],
    pub direct_s2_u_gain: [f32; 2],
    pub direct_s2_hp_gain: [f32; 2],
    pub direct_s2_w: [[f32; MAX_TOTAL_GRU_LANES]; 2],
    pub fir_l: [f32; 64],
    pub fir_r: [f32; 64],
    pub kautz_c_l: [(f32, f32); MAX_KAUTZ_POLES],
    pub kautz_c_r: [(f32, f32); MAX_KAUTZ_POLES],
}

impl Default for V3TierReadout {
    fn default() -> Self {
        let mut r = Self::zeros();
        r.pre_cab_u_gain = [1.0, 1.0];
        r.fir_l[0] = 1.0;
        r.fir_r[0] = 1.0;
        r
    }
}

impl V3TierReadout {
    /// Returns an all-zero linear readout head.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            svf_direct_taps_l: [0.0; 8],
            svf_direct_taps_r: [0.0; 8],
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

    /// Zeros any coefficients beyond `tier`'s active `(groups, fir_segments, kautz_banks)`.
    pub fn mask_to_tier(&mut self, tier: Tier) {
        let g_cnt = tier_groups(tier);
        let f_cnt = tier_fir_segments(tier) * 16;
        let k_cnt = tier_kautz_banks(tier) * 4;
        self.svf_direct_taps_l[7] = 0.0;
        self.svf_direct_taps_r[7] = 0.0;
        for ch in 0..2 {
            for b in 0..NUM_GRU_BLOCKS {
                for g in g_cnt..MAX_GROUPS {
                    for l in 0..4 {
                        let idx = flat_gru_lane_index(b, g, l);
                        self.pre_cab_w[ch][idx] = 0.0;
                        self.direct_s2_w[ch][idx] = 0.0;
                    }
                }
            }
        }
        for i in f_cnt..64 {
            self.fir_l[i] = 0.0;
            self.fir_r[i] = 0.0;
        }
        for m in k_cnt..MAX_KAUTZ_POLES {
            self.kautz_c_l[m] = (0.0, 0.0);
            self.kautz_c_r[m] = (0.0, 0.0);
        }
    }
}

/// Complete single-tier cold parameter set for a `v3` core instance.
#[derive(Clone, Debug, PartialEq)]
pub struct V3ColdTierModel {
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub os_params: V3OversamplerParams,
    pub svf: V3Stage1SvfParams,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockCold; NUM_GRU_BLOCKS],
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub readout: V3TierReadout,
}

impl V3ColdTierModel {
    /// Builds a resting identity/unity cold model for `tier`.
    #[must_use]
    pub fn identity_for_tier(tier: Tier, os2x: bool, mono_out: bool) -> Self {
        Self {
            tier,
            os2x,
            mono_out,
            os_params: V3OversamplerParams::default(),
            svf: V3Stage1SvfParams::identity_bypass(),
            feat_params: V3InputFeaturesParams::default(),
            blocks: [
                V3GruBlockCold::default(),
                V3GruBlockCold::default(),
                V3GruBlockCold::default(),
            ],
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            readout: V3TierReadout::default(),
        }
    }
}

/// Unified single-run 4-tier Matryoshka model sharing one causal core across `[Lite..Ultra]`.
#[derive(Clone, Debug, PartialEq)]
pub struct V3MatryoshkaModel {
    pub os2x: bool,
    pub mono_out: bool,
    pub os_params: V3OversamplerParams,
    pub svf: V3Stage1SvfParams,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockCold; NUM_GRU_BLOCKS],
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub readouts: [V3TierReadout; 4],
}

impl V3MatryoshkaModel {
    /// Builds a resting identity unified 4-tier Matryoshka model.
    #[must_use]
    pub fn identity(os2x: bool, mono_out: bool) -> Self {
        Self {
            os2x,
            mono_out,
            os_params: V3OversamplerParams::default(),
            svf: V3Stage1SvfParams::identity_bypass(),
            feat_params: V3InputFeaturesParams::default(),
            blocks: [
                V3GruBlockCold::default(),
                V3GruBlockCold::default(),
                V3GruBlockCold::default(),
            ],
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            readouts: [
                V3TierReadout::default(),
                V3TierReadout::default(),
                V3TierReadout::default(),
                V3TierReadout::default(),
            ],
        }
    }

    /// Extracts an executable [`V3ColdTierModel`] view for `tier` sharing the common core.
    #[must_use]
    pub fn extract_tier(&self, tier: Tier) -> V3ColdTierModel {
        let mut readout = self.readouts[tier as usize].clone();
        readout.mask_to_tier(tier);
        V3ColdTierModel {
            tier,
            os2x: self.os2x,
            mono_out: self.mono_out,
            os_params: self.os_params,
            svf: self.svf,
            feat_params: self.feat_params,
            blocks: self.blocks,
            kautz_poles: self.kautz_poles,
            readout,
        }
    }

    /// Extracts all four `[Lite, Full, Pro, Ultra]` executable models from this single-run core.
    #[must_use]
    pub fn to_four_tiers(&self) -> [V3ColdTierModel; 4] {
        [
            self.extract_tier(Tier::Lite),
            self.extract_tier(Tier::Full),
            self.extract_tier(Tier::Pro),
            self.extract_tier(Tier::Ultra),
        ]
    }
}

/// Contiguous Stage III Early FIR + 32-pole Kautz stereo cabinet parameters and state.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct V3Stage3Hot {
    pub fir_taps_l: [[[f32; 4]; 4]; MAX_FIR_SEGMENTS],
    pub fir_taps_r: [[[f32; 4]; 4]; MAX_FIR_SEGMENTS],
    pub kautz_l: [KautzModalBank<4>; MAX_KAUTZ_BANKS],
    pub kautz_r: [KautzModalBank<4>; MAX_KAUTZ_BANKS],
    pub fir_hist_l: [f32; FIR_RING_CAP * 2],
    pub fir_hist_r: [f32; FIR_RING_CAP * 2],
    pub write_pos: usize,
}

impl Default for V3Stage3Hot {
    fn default() -> Self {
        Self {
            fir_taps_l: [[[0.0; 4]; 4]; MAX_FIR_SEGMENTS],
            fir_taps_r: [[[0.0; 4]; 4]; MAX_FIR_SEGMENTS],
            kautz_l: core::array::from_fn(|_| KautzModalBank::new()),
            kautz_r: core::array::from_fn(|_| KautzModalBank::new()),
            fir_hist_l: [0.0; FIR_RING_CAP * 2],
            fir_hist_r: [0.0; FIR_RING_CAP * 2],
            write_pos: 0,
        }
    }
}

impl V3Stage3Hot {
    /// Populates Stage III FIR taps and Kautz poles/residues from cold readout and poles.
    #[must_use]
    pub fn from_cold(model: &V3ColdTierModel) -> Self {
        let mut out = Self::default();
        out.update_readout(&model.readout, &model.kautz_poles);
        out
    }

    /// Updates Stage III FIR taps and Kautz poles/residues in-place without clearing ring/bank state.
    pub fn update_readout(
        &mut self,
        readout: &V3TierReadout,
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    ) {
        for seg in 0..MAX_FIR_SEGMENTS {
            for k4 in 0..4 {
                for i in 0..4 {
                    let idx = (seg << 4) + (k4 << 2) + i;
                    self.fir_taps_l[seg][k4][i] = readout.fir_l[idx];
                    self.fir_taps_r[seg][k4][i] = readout.fir_r[idx];
                }
            }
        }
        for b in 0..MAX_KAUTZ_BANKS {
            for lane in 0..4 {
                let m = (b << 2) + lane;
                let (r_raw, theta_raw) = kautz_poles[m];
                let r = r_raw.clamp(0.25, 0.9985);
                let theta = theta_raw.clamp(0.0, core::f32::consts::PI - 1.0e-3);
                let b_gamma = sqrt_f64((1.0 - (r as f64) * (r as f64)).max(1.0e-7)) as f32;
                let (cl_re, cl_im) = readout.kautz_c_l[m];
                let (cr_re, cr_im) = readout.kautz_c_r[m];
                self.kautz_l[b].set_pole_with_b_gamma(lane, r, theta, 0.0, b_gamma, cl_re, cl_im);
                self.kautz_r[b].set_pole_with_b_gamma(lane, r, theta, 0.0, b_gamma, cr_re, cr_im);
            }
        }
    }

    /// Resets the dual-mirrored FIR ring buffers and all Kautz banks to resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        self.fir_hist_l.fill(0.0);
        self.fir_hist_r.fill(0.0);
        self.write_pos = 0;
        for b in 0..MAX_KAUTZ_BANKS {
            self.kautz_l[b].reset();
            self.kautz_r[b].reset();
        }
    }

    #[inline(always)]
    fn push_fir(&mut self, s2_l: f32, s2_r: f32) -> usize {
        let pos = self.write_pos.wrapping_sub(1) & FIR_RING_MASK;
        self.write_pos = pos;
        self.fir_hist_l[pos] = s2_l;
        self.fir_hist_l[pos + FIR_RING_CAP] = s2_l;
        self.fir_hist_r[pos] = s2_r;
        self.fir_hist_r[pos + FIR_RING_CAP] = s2_r;
        pos
    }

    #[inline(always)]
    fn eval_fir_l<B: MathBackend>(&self, pos: usize, num_fir_segments: usize) -> f32 {
        let mut acc = B::Vec4::zero();
        for seg in 0..num_fir_segments.min(MAX_FIR_SEGMENTS) {
            let s = (pos & FIR_RING_MASK) + ((seg & 3) << 4);
            let t = &self.fir_taps_l[seg];
            for k4 in 0..4 {
                let off = s + (k4 << 2);
                let h = B::Vec4::from_array([
                    self.fir_hist_l[off],
                    self.fir_hist_l[off + 1],
                    self.fir_hist_l[off + 2],
                    self.fir_hist_l[off + 3],
                ]);
                acc = acc.fma(h, B::Vec4::load(&t[k4]));
            }
        }
        acc.horizontal_sum()
    }

    #[inline(always)]
    fn eval_fir_stereo<B: MathBackend>(&self, pos: usize, num_fir_segments: usize) -> (f32, f32) {
        let mut acc_l = B::Vec4::zero();
        let mut acc_r = B::Vec4::zero();
        for seg in 0..num_fir_segments.min(MAX_FIR_SEGMENTS) {
            let s = (pos & FIR_RING_MASK) + ((seg & 3) << 4);
            let tl = &self.fir_taps_l[seg];
            let tr = &self.fir_taps_r[seg];
            for k4 in 0..4 {
                let off = s + (k4 << 2);
                let hl = B::Vec4::from_array([
                    self.fir_hist_l[off],
                    self.fir_hist_l[off + 1],
                    self.fir_hist_l[off + 2],
                    self.fir_hist_l[off + 3],
                ]);
                let hr = B::Vec4::from_array([
                    self.fir_hist_r[off],
                    self.fir_hist_r[off + 1],
                    self.fir_hist_r[off + 2],
                    self.fir_hist_r[off + 3],
                ]);
                acc_l = acc_l.fma(hl, B::Vec4::load(&tl[k4]));
                acc_r = acc_r.fma(hr, B::Vec4::load(&tr[k4]));
            }
        }
        acc_l.horizontal_sum_pair(acc_r)
    }

    #[inline(always)]
    fn step_kautz_mono_unscaled<B: MathBackend>(
        bank: &mut KautzModalBank<4>,
        x_vec: B::Vec4,
    ) -> B::Vec4 {
        let l_re = B::Vec4::load(&bank.lambda_re);
        let l_im = B::Vec4::load(&bank.lambda_im);
        let b_g = B::Vec4::load(&bank.b_gamma);
        let c_r = B::Vec4::load(&bank.c_re);
        let c_i = B::Vec4::load(&bank.c_im);
        let prev_re = B::Vec4::load(&bank.s_re);
        let prev_im = B::Vec4::load(&bank.s_im);
        let next_re = l_re.mul(prev_re).sub(l_im.mul(prev_im)).fma(b_g, x_vec);
        let next_im = l_im.mul(prev_re).add(l_re.mul(prev_im));
        flush_recursive_state_vec4::<B::Vec4>(next_re).store(&mut bank.s_re);
        flush_recursive_state_vec4::<B::Vec4>(next_im).store(&mut bank.s_im);
        c_r.mul(next_re).sub(c_i.mul(next_im))
    }

    /// Steps Stage III FIR + Kautz modal banks for stereo inputs `(s2_l, s2_r)`.
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(
        &mut self,
        s2_l: f32,
        s2_r: f32,
        num_fir_segments: usize,
        num_kautz_banks: usize,
        mono_out: bool,
    ) -> [f32; 2] {
        let pos = self.push_fir(s2_l, s2_r);
        let delayed_l = self.fir_hist_l[(pos & FIR_RING_MASK) + 16];
        let delayed_r = self.fir_hist_r[(pos & FIR_RING_MASK) + 16];
        let x_l_vec = B::Vec4::splat(delayed_l);
        let x_r_vec = B::Vec4::splat(delayed_r);

        if mono_out {
            let fir_l = self.eval_fir_l::<B>(pos, num_fir_segments);
            let mut k_acc = B::Vec4::zero();
            for b in 0..num_kautz_banks.min(MAX_KAUTZ_BANKS) {
                k_acc = k_acc.add(Self::step_kautz_mono_unscaled::<B>(
                    &mut self.kautz_l[b],
                    x_l_vec,
                ));
            }
            let out_l = fir_l + k_acc.horizontal_sum();
            [out_l, out_l]
        } else {
            let (fir_l, fir_r) = self.eval_fir_stereo::<B>(pos, num_fir_segments);
            let mut k_acc_l = B::Vec4::zero();
            let mut k_acc_r = B::Vec4::zero();
            for b in 0..num_kautz_banks.min(MAX_KAUTZ_BANKS) {
                let (vl, vr) = self.kautz_l[b].step_4lane_stereo_simd_unscaled::<B>(
                    &mut self.kautz_r[b],
                    x_l_vec,
                    x_r_vec,
                );
                k_acc_l = k_acc_l.add(vl);
                k_acc_r = k_acc_r.add(vr);
            }
            let (kl, kr) = k_acc_l.horizontal_sum_pair(k_acc_r);
            [fir_l + kl, fir_r + kr]
        }
    }
}

/// Computes active L1D working-set bytes read/written per host stereo sample for `tier` and `os2x`.
#[must_use]
pub const fn active_hot_bytes_for_tier(tier: Tier, os2x: bool) -> usize {
    let g = tier_groups(tier);
    let f = tier_fir_segments(tier);
    let k = tier_kautz_banks(tier);
    let tri_tiles = (g * (g + 1)) >> 1;
    let svf_bytes =
        core::mem::size_of::<V3Stage1SvfParams>() + core::mem::size_of::<V3Stage1SvfState>();
    let os_bytes = if os2x {
        core::mem::size_of::<V3OversamplerParams>()
            + 2 * core::mem::size_of::<ZeroLagOversampler2x>()
    } else {
        0
    };
    let feat_bytes = core::mem::size_of::<V3InputFeaturesParams>()
        + 2 * core::mem::size_of::<V3InputFeaturesState>();
    let gru_group_bytes = NUM_GRU_BLOCKS * g * core::mem::size_of::<V3GruGroupHot>();
    let gru_k_quad_bytes = NUM_GRU_BLOCKS * g * 16;
    // Block 0: 3 w_rec tiles; Blocks 1 & 2: 3 w_prev + 3 w_rec tiles each => 15 tiles per (dst_g, src_g).
    let gru_tile_bytes = 15 * tri_tiles * 64;
    let _ = LOWER_TRI_TILES;
    let _ = MAX_BLOCK_LANES;
    let gru_state_bytes = 2 * NUM_GRU_BLOCKS * g * 16;
    let bridge_bytes = 2 * (16 + NUM_GRU_BLOCKS * g * 16 * 2);
    let fir_bytes = 2 * f * 64 + 2 * f * 64;
    let kautz_bytes = 2 * k * core::mem::size_of::<KautzModalBank<4>>();
    svf_bytes
        + os_bytes
        + feat_bytes
        + gru_group_bytes
        + gru_k_quad_bytes
        + gru_tile_bytes
        + gru_state_bytes
        + bridge_bytes
        + fir_bytes
        + kautz_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3_readout_mask_and_cold_tier_hot_bytes() {
        let mut ro = V3TierReadout::default();
        ro.fir_l.fill(0.5);
        ro.fir_r.fill(0.5);
        ro.mask_to_tier(Tier::Lite);
        assert_eq!(ro.fir_l[20], 0.0);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Full);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Pro);
        ro.mask_to_tier(Tier::Ultra);

        let mat = V3MatryoshkaModel::identity(true, false);
        let four = mat.to_four_tiers();
        assert_eq!(four[0].tier, Tier::Lite);
        assert_eq!(four[3].tier, Tier::Ultra);
        assert_eq!(mat.extract_tier(Tier::Full).tier, Tier::Full);

        let cold_ultra = V3ColdTierModel::identity_for_tier(Tier::Ultra, false, false);
        assert_eq!(cold_ultra.tier, Tier::Ultra);
        assert!(
            active_hot_bytes_for_tier(Tier::Lite, true)
                < active_hot_bytes_for_tier(Tier::Ultra, true)
        );
    }
}
