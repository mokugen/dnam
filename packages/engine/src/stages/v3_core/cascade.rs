//! Stage II Block-Lower-Triangular Zero-DC Matryoshka `GRU`, Zero-Lag `2x` Oversampler, and
//! Multi-Timescale Input Feature Extractor (`v3` Core).
//!
//! All parameter and state tiles are packed by 4-lane SIMD group (`Lite = 1`, `Full = 2`,
//! `Pro = 3`, `Ultra = 4` groups per block across 3 sequential GRU blocks -> `12/24/36/48` total
//! recurrent lanes) in lower-triangular prefix order so causal state trajectories for lower tiers
//! are bit-for-bit identical prefixes of higher tiers.

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32, flush_recursive_state_vec4};
use crate::stages::v3_core::math_pade::{recip_pos_ge1_f32, sigmoid_pade5_vec4, tanh_pade5_vec4};
use crate::stages::v3_core::svf::{recip_nr3_pos_vec4, rsqrt_nr3_f32};
use crate::types::Tier;

/// Maximum number of 4-lane SIMD channel groups per GRU block (`16` lanes/block at `Ultra`).
pub const MAX_GROUPS: usize = 4;
/// Number of sequential Matryoshka GRU blocks in Stage II.
pub const NUM_GRU_BLOCKS: usize = 3;
/// Alias matching historical constant name (`3` Stage II blocks).
pub const NUM_CASCADE_BLOCKS: usize = NUM_GRU_BLOCKS;
/// Maximum lanes per Stage II GRU block (`4 * MAX_GROUPS = 16`).
pub const MAX_BLOCK_LANES: usize = MAX_GROUPS * 4;
/// Maximum total Stage II recurrent lanes across all 3 blocks (`48`).
pub const MAX_TOTAL_GRU_LANES: usize = NUM_GRU_BLOCKS * MAX_BLOCK_LANES;
/// Legacy constant alias (`48` total active Stage II lanes at `Ultra`).
pub const MAX_CASCADE_LANES: usize = MAX_TOTAL_GRU_LANES;
/// Number of lower-triangular `4x4` matrix tiles (`1 + 2 + 3 + 4 = 10`).
pub const LOWER_TRI_TILES: usize = (MAX_GROUPS * (MAX_GROUPS + 1)) >> 1;
/// Number of multi-timescale input features consumed by Block 0 (`[u, u_hp, u_pinch, env_fast, env_slow]`).
pub const NUM_IN_FEATS_B0: usize = 5;
/// Number of multi-timescale input features consumed by Blocks 1 and 2 (`[u, u_hp, u_pinch, env_fast, env_slow]`).
pub const NUM_IN_FEATS_BN: usize = 5;
/// Maximum number of 16-tap Stage III Early FIR segments (`64` taps at `Ultra`).
pub const MAX_FIR_SEGMENTS: usize = 4;
/// Maximum number of 4-pole Kautz modal banks (`32` complex conjugate pairs at `Ultra`).
pub const MAX_KAUTZ_BANKS: usize = 8;
/// Maximum Stage III conjugate pole pairs (`32`).
pub const MAX_KAUTZ_POLES: usize = MAX_KAUTZ_BANKS * 4;
/// Power-of-two Stage III FIR ring capacity (`64` samples, mirrored to `128`).
pub const FIR_RING_CAP: usize = 64;
/// Power-of-two mask (`FIR_RING_CAP - 1`).
pub const FIR_RING_MASK: usize = FIR_RING_CAP - 1;

/// Active 4-lane group count (`G`) per GRU block for `tier` (`1, 2, 3, 4`).
#[inline(always)]
#[must_use]
pub const fn tier_groups(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 1,
        Tier::Full => 2,
        Tier::Pro => 3,
        Tier::Ultra => 4,
    }
}

/// Active lane count per GRU block for `tier` (`4, 8, 12, 16`).
#[inline(always)]
#[must_use]
pub const fn tier_block_lanes(tier: Tier) -> usize {
    tier_groups(tier) << 2
}

/// Active total GRU lanes across all 3 blocks for `tier` (`12, 24, 36, 48`).
#[inline(always)]
#[must_use]
pub const fn tier_total_gru_lanes(tier: Tier) -> usize {
    NUM_GRU_BLOCKS * tier_block_lanes(tier)
}

/// Legacy helper returning active lanes per Stage II block (`4, 8, 12, 16`).
#[inline(always)]
#[must_use]
pub const fn tier_cascade_lanes(tier: Tier) -> usize {
    tier_block_lanes(tier)
}

/// Active Stage III 16-tap FIR segment count (`1, 2, 3, 4` -> `16, 32, 48, 64` taps).
#[inline(always)]
#[must_use]
pub const fn tier_fir_segments(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 1,
        Tier::Full => 2,
        Tier::Pro => 3,
        Tier::Ultra => 4,
    }
}

/// Active Stage III 4-pole Kautz bank count (`1, 2, 4, 8` -> `4, 8, 16, 32` pole pairs).
#[inline(always)]
#[must_use]
pub const fn tier_kautz_banks(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 1,
        Tier::Full => 2,
        Tier::Pro => 4,
        Tier::Ultra => 8,
    }
}

/// Packed lower-triangular `4x4` tile index for `(dst_g, src_g)` (`src_g <= dst_g`, `0..10`).
#[inline(always)]
#[must_use]
pub const fn tri_tile_index(dst_g: usize, src_g: usize) -> usize {
    ((dst_g * (dst_g + 1)) >> 1) + src_g
}

/// Flattened readout lane index for `(block_idx, grp_idx, lane_idx)` in `0..MAX_TOTAL_GRU_LANES`.
#[inline(always)]
#[must_use]
pub const fn flat_gru_lane_index(block_idx: usize, grp_idx: usize, lane_idx: usize) -> usize {
    (block_idx << 4) + (grp_idx << 2) + lane_idx
}

/// 9th-order Polyphase Elliptic Half-Band IIR branch 0 allpass coefficients (`HB_A0`, `> 68 dB` rejection).
pub const HB_A0: [f32; 2] = [0.079_866_426, 0.545_353_65];
/// 9th-order Polyphase Elliptic Half-Band IIR branch 1 allpass coefficients (`HB_A1`, `> 68 dB` rejection).
pub const HB_A1: [f32; 2] = [0.283_829_34, 0.834_411_9];

#[inline(always)]
fn step_allpass_1_f32(x: f32, a: f32, state: &mut f32) -> f32 {
    let s_old = *state;
    let s_new = flush_recursive_state_f32(x - a * s_old);
    *state = s_new;
    flush_recursive_state_f32(a * s_new + s_old)
}

/// Division-free 2nd-order allpass cascade (`0 FDIV`, exact `+0.0 -> +0.0` at rest).
#[inline(always)]
pub fn step_allpass_2_f32(x: f32, a: [f32; 2], s0: &mut f32, s1: &mut f32) -> f32 {
    let m = step_allpass_1_f32(x, a[0], s0);
    step_allpass_1_f32(m, a[1], s1)
}

/// Division-free 9th-order polyphase elliptic half-band `96 kHz -> 48 kHz` decimator (`0 FDIV`,
/// `> 68 dB` stopband attenuation, `< 0.0005 dB` passband ripple, exact `+0.0 -> +0.0` at rest).
#[inline(always)]
pub fn step_hb_decim_f32(y0: f32, y1: f32, st: &mut [f32; 4]) -> f32 {
    let [mut s0, mut s1, mut s2, mut s3] = *st;
    let b0 = step_allpass_2_f32(y1, HB_A0, &mut s0, &mut s1);
    let b1 = step_allpass_2_f32(y0, HB_A1, &mut s2, &mut s3);
    *st = [s0, s1, s2, s3];
    flush_recursive_state_f32(0.5 * (b0 + b1))
}

/// Hybrid zero-lag polynomial + 9th-order polyphase elliptic half-band `2x` oversampler parameters.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3OversamplerParams {
    /// Causal quadratic midpoint curvature coefficient in `[-0.20, 0.05]` (`default = -0.125`).
    pub up_quad_coeff: f32,
    /// Polyphase decimation tap on previous `sub=1` sample `y[n-1, 1]` in `[0.0, 0.28]`.
    pub dec_c_prev1: f32,
    /// Polyphase decimation tap on midpoint `sub=0` sample `y[n, 0]` in `[0.0, 0.55]`.
    pub dec_c0: f32,
    /// Complement decimation tap `1.0 - dec_c_prev1 - dec_c0 >= 0.17` on current `sub=1` sample.
    pub dec_c1: f32,
    /// Causal unity-DC pre-filter `xp = (1 - up_pre_emph) * x + up_pre_emph * prev_x` in `[-0.28, 1.0]`.
    pub up_pre_emph: f32,
    /// Linear blend between 3-tap polyphase FIR (`0.0`) and 9th-order half-band decimator (`1.0`).
    pub dec_aa_mix: f32,
    /// Linear blend between zero-lag quadratic upsampler (`0.0`) and 9th-order half-band interpolator (`1.0`).
    pub up_aa_mix: f32,
    pub _pad: [f32; 1],
}

impl Default for V3OversamplerParams {
    fn default() -> Self {
        Self {
            up_quad_coeff: -0.125,
            dec_c_prev1: 0.15,
            dec_c0: 0.50,
            dec_c1: 0.35,
            up_pre_emph: 0.0,
            dec_aa_mix: 0.0,
            up_aa_mix: 0.0,
            _pad: [0.0; 1],
        }
    }
}

impl V3OversamplerParams {
    /// Clamps oversampler taps into stable passivity bounds (`dec_c_prev1 + dec_c0 + dec_c1 == 1`).
    pub fn clamp_in_place(&mut self) {
        self.up_quad_coeff = self.up_quad_coeff.clamp(-0.20, 0.05);
        self.dec_c_prev1 = self.dec_c_prev1.clamp(0.0, 0.28);
        self.dec_c0 = self.dec_c0.clamp(0.0, 0.55);
        self.dec_c1 = 1.0 - self.dec_c_prev1 - self.dec_c0;
        self.up_pre_emph = self.up_pre_emph.clamp(-0.28, 1.0);
        self.dec_aa_mix = self.dec_aa_mix.clamp(0.0, 1.0);
        self.up_aa_mix = self.up_aa_mix.clamp(0.0, 1.0);
        self._pad = [0.0; 1];
    }
}

/// Per-channel hybrid zero-lag + 9th-order polyphase half-band `2x` upsampler & decimator state (`96 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZeroLagOversampler2x {
    pub up_raw_prev: f32,
    pub up_prev1: f32,
    pub up_prev2: f32,
    pub dec_s2_prev1: f32,
    pub dec_dir_s2_prev1: f32,
    pub dec_dir_svf_prev1: f32,
    pub up_hb_state: [f32; 4],
    pub dec_s2_aa: [f32; 4],
    pub dec_dir_s2_aa: [f32; 4],
    pub dec_dir_svf_aa: [f32; 4],
    pub _pad: [f32; 2],
}

impl ZeroLagOversampler2x {
    /// Resets all oversampler delay and half-band allpass elements to exact `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Causal `2x` upsampler blending quadratic zero-lag midpoint and 9th-order polyphase half-band IIR.
    #[inline(always)]
    pub fn upsample(&mut self, x: f32, params: &V3OversamplerParams) -> [f32; 2] {
        let xp = (1.0 - params.up_pre_emph) * x + params.up_pre_emph * self.up_raw_prev;
        self.up_raw_prev = x;
        let d2 = xp - 2.0 * self.up_prev1 + self.up_prev2;
        let x0_poly = 0.5 * (self.up_prev1 + xp) + params.up_quad_coeff * d2;
        let x1_poly = xp;
        self.up_prev2 = self.up_prev1;
        self.up_prev1 = xp;
        if params.up_aa_mix > 0.0 {
            let [mut s0, mut s1, mut s2, mut s3] = self.up_hb_state;
            let x0_hb = step_allpass_2_f32(xp, HB_A0, &mut s0, &mut s1);
            let x1_hb = step_allpass_2_f32(xp, HB_A1, &mut s2, &mut s3);
            self.up_hb_state = [s0, s1, s2, s3];
            let m = params.up_aa_mix;
            [
                x0_poly + m * (x0_hb - x0_poly),
                x1_poly + m * (x1_hb - x1_poly),
            ]
        } else {
            [x0_poly, x1_poly]
        }
    }

    /// Hybrid polyphase FIR + 9th-order elliptic half-band `96 kHz -> 48 kHz` decimator.
    #[inline(always)]
    pub fn downsample(
        y0: f32,
        y1: f32,
        prev_y1: &mut f32,
        aa_state: &mut [f32; 4],
        params: &V3OversamplerParams,
    ) -> f32 {
        let y_poly = params.dec_c_prev1 * (*prev_y1) + params.dec_c0 * y0 + params.dec_c1 * y1;
        *prev_y1 = y1;
        if params.dec_aa_mix > 0.0 {
            let y_aa = step_hb_decim_f32(y0, y1, aa_state);
            y_poly + params.dec_aa_mix * (y_aa - y_poly)
        } else {
            y_poly
        }
    }
}

/// Multi-timescale input feature extractor parameters stored in precision-preserving `beta = 1 - alpha`.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3InputFeaturesParams {
    /// High-pass 1-pole tracker step `beta in [1e-4, 0.50]` (`s_lp += in_hp_beta * (u - s_lp)`).
    pub in_hp_beta: f32,
    /// Fast rectified envelope attack step `beta in [1e-4, 0.999]`.
    pub fast_atk_beta: f32,
    /// Fast rectified envelope release step `beta in [1e-5, 0.999]`.
    pub fast_rel_beta: f32,
    /// Quadratic pinch-off gate sharpness `>= 0.0` (`u * (1 - 1 / (1 + pinch_k * env_fast^2))`).
    pub pinch_k: f32,
    /// Slow power-supply sag envelope attack step `beta in [1e-5, 0.50]`.
    pub slow_atk_beta: f32,
    /// Slow power-supply sag envelope release step `beta in [1e-6, 0.10]`.
    pub slow_rel_beta: f32,
    /// Preamp-compressed power-supply sag detector saturation `c_det in [0.0, 64.0]`
    /// (`rect_slow = |u| / sqrt(1 + slow_det_comp * u^2)`).
    pub slow_det_comp: f32,
    pub _pad: f32,
}

impl Default for V3InputFeaturesParams {
    fn default() -> Self {
        Self {
            in_hp_beta: 0.045,
            fast_atk_beta: 0.035,
            fast_rel_beta: 0.001,
            pinch_k: 0.0,
            slow_atk_beta: 0.0025,
            slow_rel_beta: 8.5e-6,
            slow_det_comp: 0.0,
            _pad: 0.0,
        }
    }
}

impl V3InputFeaturesParams {
    /// Clamps filter and envelope step coefficients into unconditional BIBO bounds.
    pub fn clamp_in_place(&mut self) {
        self.in_hp_beta = self.in_hp_beta.clamp(1.0e-4, 0.50);
        self.fast_atk_beta = self.fast_atk_beta.clamp(1.0e-4, 0.999);
        self.fast_rel_beta = self.fast_rel_beta.clamp(1.0e-5, 0.999);
        self.pinch_k = self.pinch_k.clamp(0.0, 25_000.0);
        self.slow_atk_beta = self.slow_atk_beta.clamp(1.0e-5, 0.50);
        self.slow_rel_beta = self.slow_rel_beta.clamp(1.0e-6, 0.10);
        self.slow_det_comp = self.slow_det_comp.clamp(0.0, 64.0);
        self._pad = 0.0;
    }
}

/// Per-channel multi-timescale input feature recursive state (`16 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3InputFeaturesState {
    pub u_lp: f32,
    pub env_fast: f32,
    pub env_slow: f32,
    pub _pad: f32,
}

impl V3InputFeaturesState {
    /// Resets recursive input feature states to exact `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Advances input feature detectors by one sample and returns `[u, u_hp, u_pinch, env_fast, env_slow]`.
    #[inline(always)]
    pub fn step(&mut self, p: &V3InputFeaturesParams, u: f32) -> [f32; NUM_IN_FEATS_B0] {
        self.u_lp = flush_recursive_state_f32(self.u_lp + p.in_hp_beta * (u - self.u_lp));
        let u_hp = u - self.u_lp;
        let rect = u.abs();
        let fb = if rect > self.env_fast {
            p.fast_atk_beta
        } else {
            p.fast_rel_beta
        };
        self.env_fast = flush_recursive_state_f32(self.env_fast + fb * (rect - self.env_fast));
        let rect_slow = if p.slow_det_comp > 0.0 {
            rect * rsqrt_nr3_f32(1.0 + p.slow_det_comp * rect * rect)
        } else {
            rect
        };
        let sb = if rect_slow > self.env_slow {
            p.slow_atk_beta
        } else {
            p.slow_rel_beta
        };
        self.env_slow = flush_recursive_state_f32(self.env_slow + sb * (rect_slow - self.env_slow));
        let u_pinch = if p.pinch_k > 0.0 {
            let g_pinch = 1.0 - recip_pos_ge1_f32(1.0 + p.pinch_k * self.env_fast * self.env_fast);
            u * g_pinch
        } else {
            0.0
        };
        [u, u_hp, u_pinch, self.env_fast, self.env_slow]
    }
}

/// Column-major `4x4` matrix tile (`64 B`, aligned for 4-lane SIMD column broadcasts).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat4Tile {
    pub cols: [[f32; 4]; 4],
}

/// Cold representation of one Matryoshka Stage II GRU block (`16` lanes/block max, lower-triangular).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3GruBlockCold {
    /// Update gate input weights `[feat_idx][dst_lane]`.
    pub w_in_z: [[f32; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    /// Reset gate input weights `[feat_idx][dst_lane]`.
    pub w_in_r: [[f32; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    /// Candidate input weights `[feat_idx][dst_lane]`.
    pub w_in_n: [[f32; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    /// Inter-block update weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_prev_z: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Inter-block reset weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_prev_r: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Inter-block candidate weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_prev_n: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Recurrent update weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_rec_z: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Recurrent reset weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_rec_r: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Recurrent candidate weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_rec_n: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Update gate biases (`b_z in [-8.0, 8.2]`).
    pub b_z: [f32; MAX_BLOCK_LANES],
    /// Reset gate biases (`b_r in [-8.0, 8.2]`).
    pub b_r: [f32; MAX_BLOCK_LANES],
    /// Candidate equilibrium-cancelled biases (`b_n in [-3.8, 3.8]`).
    pub b_n: [f32; MAX_BLOCK_LANES],
    /// Multiplicative zero-DC power-supply headroom sag (`k_sag in [0.0, 8.0]`).
    pub k_sag: [f32; MAX_BLOCK_LANES],
    /// Zero-DC even-order triode square-law curvature (`k_quad in [-0.45, 0.45]`).
    pub k_quad: [f32; MAX_BLOCK_LANES],
}

impl Default for V3GruBlockCold {
    fn default() -> Self {
        Self {
            w_in_z: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_in_r: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_in_n: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_prev_z: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_prev_r: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_prev_n: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_z: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_r: [[f32::default(); MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_n: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            b_z: [0.0; MAX_BLOCK_LANES],
            b_r: [1.5; MAX_BLOCK_LANES],
            b_n: [0.0; MAX_BLOCK_LANES],
            k_sag: [0.0; MAX_BLOCK_LANES],
            k_quad: [0.0; MAX_BLOCK_LANES],
        }
    }
}

/// Historical cold block type alias for module ergonomics.
pub type V3CascadeBlockCold = V3GruBlockCold;

impl V3GruBlockCold {
    /// Clamps biases, sag depth, and enforces strict block-lower-triangular sparsity (`src_g <= dst_g`).
    pub fn clamp_in_place(&mut self) {
        for dst in 0..MAX_BLOCK_LANES {
            self.b_z[dst] = self.b_z[dst].clamp(-8.0, 8.2);
            self.b_r[dst] = self.b_r[dst].clamp(-8.0, 8.2);
            self.b_n[dst] = self.b_n[dst].clamp(-3.8, 3.8);
            self.k_sag[dst] = self.k_sag[dst].clamp(0.0, 8.0);
            self.k_quad[dst] = self.k_quad[dst].clamp(-0.45, 0.45);
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
}

/// Per-group (`4` lanes) packed GRU input & bias parameters (`320 B`, 5 cache lines).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3GruGroupHot {
    pub w_in_z: [[f32; 4]; NUM_IN_FEATS_B0],
    pub w_in_r: [[f32; 4]; NUM_IN_FEATS_B0],
    pub w_in_n: [[f32; 4]; NUM_IN_FEATS_B0],
    pub b_z: [f32; 4],
    pub b_r: [f32; 4],
    pub b_n: [f32; 4],
    /// Backend-precomputed `tanh_pade5_vec4::<B>(b_n)` guaranteeing bit-exact `+0.0` at rest.
    pub tanh_b_n: [f32; 4],
    /// Multiplicative zero-DC power-supply headroom sag (`[0.0, 8.0]`).
    pub k_sag: [f32; 4],
}

/// Hot L1D-resident Matryoshka GRU block (`64 B` aligned; lower-triangular prefix packed).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3GruBlockHot {
    pub groups: [V3GruGroupHot; MAX_GROUPS],
    pub k_quad: [[f32; 4]; MAX_GROUPS],
    pub w_prev_z: [Mat4Tile; LOWER_TRI_TILES],
    pub w_prev_r: [Mat4Tile; LOWER_TRI_TILES],
    pub w_prev_n: [Mat4Tile; LOWER_TRI_TILES],
    pub w_rec_z: [Mat4Tile; LOWER_TRI_TILES],
    pub w_rec_r: [Mat4Tile; LOWER_TRI_TILES],
    pub w_rec_n: [Mat4Tile; LOWER_TRI_TILES],
}

fn pack_lower_tri_tiles(
    src_mat: &[[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    g_cnt: usize,
) -> [Mat4Tile; LOWER_TRI_TILES] {
    let mut out = [Mat4Tile::default(); LOWER_TRI_TILES];
    for dst_g in 0..g_cnt {
        for src_g in 0..=dst_g {
            let mut cols = [[0.0_f32; 4]; 4];
            for col in 0..4 {
                for row in 0..4 {
                    cols[col][row] = src_mat[(dst_g << 2) + row][(src_g << 2) + col];
                }
            }
            out[tri_tile_index(dst_g, src_g)].cols = cols;
        }
    }
    out
}

impl V3GruBlockHot {
    /// Packs `V3GruBlockCold` into contiguous lower-triangular SIMD tiles with precomputed `tanh(b_n)`.
    pub fn from_cold<B: MathBackend>(cold: &V3GruBlockCold, num_groups: usize) -> Self {
        let mut c = *cold;
        c.clamp_in_place();
        let g_cnt = num_groups.clamp(1, MAX_GROUPS);
        let mut out = Self::default();
        for g in 0..g_cnt {
            let base = g << 2;
            let mut grp = V3GruGroupHot::default();
            for f in 0..NUM_IN_FEATS_B0 {
                for l in 0..4 {
                    grp.w_in_z[f][l] = c.w_in_z[f][base + l];
                    grp.w_in_r[f][l] = c.w_in_r[f][base + l];
                    grp.w_in_n[f][l] = c.w_in_n[f][base + l];
                }
            }
            for l in 0..4 {
                grp.b_z[l] = c.b_z[base + l];
                grp.b_r[l] = c.b_r[base + l];
                grp.b_n[l] = c.b_n[base + l];
                grp.k_sag[l] = c.k_sag[base + l];
                out.k_quad[g][l] = c.k_quad[base + l];
            }
            grp.tanh_b_n = tanh_pade5_vec4::<B>(B::Vec4::load(&grp.b_n)).to_array();
            out.groups[g] = grp;
        }
        out.w_prev_z = pack_lower_tri_tiles(&c.w_prev_z, g_cnt);
        out.w_prev_r = pack_lower_tri_tiles(&c.w_prev_r, g_cnt);
        out.w_prev_n = pack_lower_tri_tiles(&c.w_prev_n, g_cnt);
        out.w_rec_z = pack_lower_tri_tiles(&c.w_rec_z, g_cnt);
        out.w_rec_r = pack_lower_tri_tiles(&c.w_rec_r, g_cnt);
        out.w_rec_n = pack_lower_tri_tiles(&c.w_rec_n, g_cnt);
        out
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn step_block<B: MathBackend>(
        &self,
        block_idx: usize,
        num_groups: usize,
        feats: &[f32; NUM_IN_FEATS_B0],
        prev_block_h: &[[f32; 4]; MAX_GROUPS],
        block_h: &mut [[f32; 4]; MAX_GROUPS],
        cab_acc: &mut B::Vec4,
        dir_acc: &mut B::Vec4,
        pre_cab_w_blk: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w_blk: &[[f32; 4]; MAX_GROUPS],
    ) {
        let h_old = *block_h;
        for dst_g in 0..num_groups {
            let grp = &self.groups[dst_g];
            let mut az = B::Vec4::load(&grp.b_z);
            let mut ar = B::Vec4::load(&grp.b_r);
            let mut an_in = B::Vec4::zero();
            for (f, &fv) in feats.iter().enumerate() {
                let xf = B::Vec4::splat(fv);
                az = az.fma(xf, B::Vec4::load(&grp.w_in_z[f]));
                ar = ar.fma(xf, B::Vec4::load(&grp.w_in_r[f]));
                an_in = an_in.fma(xf, B::Vec4::load(&grp.w_in_n[f]));
            }
            if block_idx > 0 {
                for src_g in 0..=dst_g {
                    let tile_idx = tri_tile_index(dst_g, src_g);
                    let x_prev = &prev_block_h[src_g];
                    az = B::fma_mat4x4_vec4(az, &self.w_prev_z[tile_idx].cols, x_prev);
                    ar = B::fma_mat4x4_vec4(ar, &self.w_prev_r[tile_idx].cols, x_prev);
                    an_in = B::fma_mat4x4_vec4(an_in, &self.w_prev_n[tile_idx].cols, x_prev);
                }
            }

            let mut rec_n = B::Vec4::zero();
            for src_g in 0..=dst_g {
                let tile_idx = tri_tile_index(dst_g, src_g);
                let x_old = &h_old[src_g];
                az = B::fma_mat4x4_vec4(az, &self.w_rec_z[tile_idx].cols, x_old);
                ar = B::fma_mat4x4_vec4(ar, &self.w_rec_r[tile_idx].cols, x_old);
                rec_n = B::fma_mat4x4_vec4(rec_n, &self.w_rec_n[tile_idx].cols, x_old);
            }

            let z = sigmoid_pade5_vec4::<B>(az);
            let r = sigmoid_pade5_vec4::<B>(ar);
            let a_sig = an_in
                .add(rec_n)
                .clamp(B::Vec4::splat(-12.0), B::Vec4::splat(12.0));
            let a_eff = a_sig.fma(a_sig.mul(a_sig), B::Vec4::load(&self.k_quad[dst_g]));
            let one = B::Vec4::splat(1.0);
            let s_sag = B::Vec4::load(&grp.k_sag).mul(B::Vec4::splat(feats[4]));
            let g_sag = one.add(s_sag);
            let h_sag = one.fms(s_sag, recip_nr3_pos_vec4::<B>(g_sag));
            let an = B::Vec4::load(&grp.b_n).fma(g_sag, a_eff);
            let n = h_sag.mul(tanh_pade5_vec4::<B>(an).sub(B::Vec4::load(&grp.tanh_b_n)));
            let h_old_v = B::Vec4::load(&h_old[dst_g]);
            let h_ret = one.sub(z).mul(h_old_v);
            let h_new = flush_recursive_state_vec4::<B::Vec4>(h_ret.fma(r, n))
                .clamp(B::Vec4::splat(-2.5), B::Vec4::splat(2.5));

            let mut h_arr = [0.0_f32; 4];
            h_new.store(&mut h_arr);
            block_h[dst_g] = h_arr;
            *cab_acc = cab_acc.fma(h_new, B::Vec4::load(&pre_cab_w_blk[dst_g]));
            *dir_acc = dir_acc.fma(h_new, B::Vec4::load(&direct_s2_w_blk[dst_g]));
        }
    }
}

/// Per-channel Stage II Matryoshka GRU recursive state (`MAX_GROUPS = 4` × `3` blocks).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3GruChannelState {
    pub feat: V3InputFeaturesState,
    pub h: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
}

impl V3GruChannelState {
    /// Resets all input feature and GRU hidden states to exact resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3_oversampler_hb_dc_and_stopband() {
        let mut p = V3OversamplerParams {
            up_pre_emph: 0.18,
            dec_aa_mix: 1.0,
            up_aa_mix: 1.0,
            ..V3OversamplerParams::default()
        };
        p.clamp_in_place();
        let mut os = ZeroLagOversampler2x::default();
        let mut prev_y1 = 0.0_f32;
        let mut aa = [0.0_f32; 4];
        let mut dc_out = 0.0_f32;
        for _ in 0..256 {
            let up = os.upsample(1.0, &p);
            dc_out = ZeroLagOversampler2x::downsample(up[0], up[1], &mut prev_y1, &mut aa, &p);
        }
        assert!((dc_out - 1.0).abs() < 1.0e-5, "dc_out={dc_out}");

        // Alternating +/-1 at 96 kHz -> 48 kHz stopband alias must be suppressed to ~0.
        aa = [0.0; 4];
        prev_y1 = 0.0;
        let mut rms_sq = 0.0_f32;
        for k in 0..128 {
            let y = ZeroLagOversampler2x::downsample(1.0, -1.0, &mut prev_y1, &mut aa, &p);
            if k >= 32 {
                rms_sq += y * y;
            }
        }
        let rms = (rms_sq / 96.0).sqrt();
        assert!(
            rms < 1.0e-4,
            "96 kHz Nyquist attenuation insufficient: rms={rms}"
        );
    }
}
