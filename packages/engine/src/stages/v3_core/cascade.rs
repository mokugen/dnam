//! Stage II Learned 4-Lane SIMD Nonlinear Cascade, 2x Polyphase Half-Band IIR Oversampler,
//! and Contiguous Stage III Early FIR + Kautz Modal Bank (`v3` Core).
//!
//! All data layouts are packed monotonically by tier (`Lite = 1`, `Full = 2`, `Pro = 3`, `Ultra = 4`
//! 4-lane groups) so hot loops touch strictly contiguous 64-byte cache lines with zero holes,
//! zero `FDIV`, zero `FSQRT`, zero `libm`, and zero heap allocations.

use crate::math::{
    MathBackend, SimdVec4, flush_recursive_state_f32, flush_recursive_state_vec4, sqrt_f64,
};
use crate::operators::op_m::KautzModalBank;
use crate::stages::v3_core::svf::{
    V3Stage1SvfParams, V3Stage1SvfState, algebraic_softclip_vec4, recip_nr3_pos_f32,
    zero_dc_asym_sat_vec4,
};
use crate::types::Tier;

/// Maximum number of 4-lane SIMD channel groups (`C_max = 16` channels at `Ultra`).
pub const MAX_GROUPS: usize = 4;
/// Maximum number of Stage II nonlinear cascade channels (`4 * MAX_GROUPS = 16`).
pub const MAX_CASCADE_LANES: usize = MAX_GROUPS * 4;
/// Number of sequential nonlinear cascade blocks in Stage II.
pub const NUM_CASCADE_BLOCKS: usize = 3;
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

const HB_A0: [f32; 2] = [0.079_866_426, 0.545_353_65];
const HB_A1: [f32; 2] = [0.283_829_34, 0.834_411_9];

/// Active Stage II 4-lane group count (`C / 4`) for a given tier (`1, 2, 3, 4`).
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

/// Active Stage II channel width `C` for a given tier (`4, 8, 12, 16`).
#[inline(always)]
#[must_use]
pub const fn tier_cascade_lanes(tier: Tier) -> usize {
    tier_groups(tier) * 4
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

#[inline(always)]
fn step_allpass_1(x: f32, a: f32, state: &mut f32) -> f32 {
    let s_old = *state;
    let s_new = flush_recursive_state_f32(x - a * s_old);
    *state = s_new;
    a * s_new + s_old
}

#[inline(always)]
fn step_allpass_2(x: f32, a: [f32; 2], s: &mut [f32; 2]) -> f32 {
    let y0 = step_allpass_1(x, a[0], &mut s[0]);
    step_allpass_1(y0, a[1], &mut s[1])
}

/// Per-channel 2x polyphase half-band elliptic IIR oversampler state (upsample `u_s2` + downsample
/// `[s2_out, direct_s2]`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChannelOversampler2x {
    pub up_s0: [f32; 2],
    pub up_s1: [f32; 2],
    pub dn_s2_0: [f32; 2],
    pub dn_s2_1: [f32; 2],
    pub dn_dir_0: [f32; 2],
    pub dn_dir_1: [f32; 2],
    pub _pad: [f32; 4],
}

impl ChannelOversampler2x {
    /// Resets all allpass internal states to `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Upsamples one `1x` input sample into `[y_even, y_odd]` at `2x` rate.
    #[inline(always)]
    pub fn upsample(&mut self, x: f32) -> [f32; 2] {
        [
            step_allpass_2(x, HB_A0, &mut self.up_s0),
            step_allpass_2(x, HB_A1, &mut self.up_s1),
        ]
    }

    /// Downsamples a `2x` pair `[y_even, y_odd]` from the Stage II->III bridge (`dn_s2`).
    #[inline(always)]
    pub fn downsample_s2(&mut self, y_even: f32, y_odd: f32) -> f32 {
        0.5 * (step_allpass_2(y_odd, HB_A0, &mut self.dn_s2_0)
            + step_allpass_2(y_even, HB_A1, &mut self.dn_s2_1))
    }

    /// Downsamples a `2x` pair `[y_even, y_odd]` from the direct Stage II parallel path (`dn_dir`).
    #[inline(always)]
    pub fn downsample_dir(&mut self, y_even: f32, y_odd: f32) -> f32 {
        0.5 * (step_allpass_2(y_odd, HB_A0, &mut self.dn_dir_0)
            + step_allpass_2(y_even, HB_A1, &mut self.dn_dir_1))
    }
}

/// Cold representation of one Stage II nonlinear cascade block (`C_max = 16` lanes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3CascadeBlockCold {
    /// Envelope attack retention pole (already at Stage II sample rate: `1x` or `2x`).
    pub atk_alpha: f32,
    /// Envelope release retention pole (already at Stage II sample rate: `1x` or `2x`).
    pub rel_alpha: f32,
    /// Expander/gate blend in `[0.0, 1.0]` (`0` = always open, `1` = full gated fuzz).
    pub gate_mix: f32,
    /// Expander/gate sharpness (`>= 0.0`): `g = (1 - m) + m * (1 - 1 / (1 + sharp * q^2))`.
    pub gate_sharpness: f32,
    /// Direct Stage I signal injection gain into each lane.
    pub in_u_gain: [f32; MAX_CASCADE_LANES],
    /// Row-major dense inter-block coupling `[dst_lane][src_lane]` (unused for block 0).
    pub inter_mat: [[f32; MAX_CASCADE_LANES]; MAX_CASCADE_LANES],
    /// 1-pole coupling filter pole per lane in `[0.02, 0.9985]`.
    pub lp_pole: [f32; MAX_CASCADE_LANES],
    /// Low-pass coupling mix per lane.
    pub lp_mix: [f32; MAX_CASCADE_LANES],
    /// High-pass coupling mix per lane (`u_pre = lp_mix * s_lp + hp_mix * (v - s_lp)`).
    pub hp_mix: [f32; MAX_CASCADE_LANES],
    /// Resting grid bias per lane (zero-DC cancelled via `phi(x_drv + b) - phi(b0)`).
    pub static_bias: [f32; MAX_CASCADE_LANES],
    /// Dynamic sag/bias shift sensitivity per unit block envelope `q`.
    pub env_bias_sens: [f32; MAX_CASCADE_LANES],
    /// Pre-drive linear skip coefficient per lane (`+ skip_u_gain * u_pre`).
    pub skip_u_gain: [f32; MAX_CASCADE_LANES],
    /// Previous-block residual skip gain per lane (unused for block 0).
    pub prev_skip_gain: [f32; MAX_CASCADE_LANES],
}

impl Default for V3CascadeBlockCold {
    fn default() -> Self {
        Self {
            atk_alpha: 0.70,
            rel_alpha: 0.994,
            gate_mix: 0.0,
            gate_sharpness: 8.0,
            in_u_gain: [0.0; MAX_CASCADE_LANES],
            inter_mat: [[0.0; MAX_CASCADE_LANES]; MAX_CASCADE_LANES],
            lp_pole: [0.50; MAX_CASCADE_LANES],
            lp_mix: [1.0; MAX_CASCADE_LANES],
            hp_mix: [1.0; MAX_CASCADE_LANES],
            static_bias: [0.0; MAX_CASCADE_LANES],
            env_bias_sens: [0.0; MAX_CASCADE_LANES],
            skip_u_gain: [0.0; MAX_CASCADE_LANES],
            prev_skip_gain: [0.0; MAX_CASCADE_LANES],
        }
    }
}

impl V3CascadeBlockCold {
    /// Clamps physical poles and mixes to unconditional BIBO stability bounds.
    pub fn clamp_in_place(&mut self) {
        self.atk_alpha = self.atk_alpha.clamp(0.001, 0.9999);
        self.rel_alpha = self.rel_alpha.clamp(0.001, 0.99995);
        self.gate_mix = self.gate_mix.clamp(0.0, 1.0);
        self.gate_sharpness = self.gate_sharpness.clamp(0.0, 1.0e4);
        for c in 0..MAX_CASCADE_LANES {
            self.lp_pole[c] = self.lp_pole[c].clamp(0.02, 0.9985);
            self.static_bias[c] = self.static_bias[c].clamp(-4.0, 4.0);
            self.env_bias_sens[c] = self.env_bias_sens[c].clamp(-4.0, 4.0);
        }
    }
}

/// Complete cold parameter set for a `v3` Tier model (produced by `packages/capture`).
#[derive(Clone, Debug, PartialEq)]
pub struct V3ColdTierModel {
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub svf: V3Stage1SvfParams,
    pub blocks: [V3CascadeBlockCold; NUM_CASCADE_BLOCKS],
    /// Pre-cabinet Stage I feed gain into Stage II->III bridge (`[L, R]`).
    pub pre_cab_u_gain: [f32; 2],
    /// Pre-cabinet Stage II last-block channel projection (`[L/R][lane]`).
    pub pre_cab_w: [[f32; MAX_CASCADE_LANES]; 2],
    /// Parallel Stage II direct-to-output weights (`[L/R][lane]`, Stage III bypass).
    pub direct_s2_w: [[f32; MAX_CASCADE_LANES]; 2],
    /// Left Early FIR taps (`16/32/48/64` active).
    pub fir_l: [f32; 64],
    /// Right Early FIR taps (`16/32/48/64` active).
    pub fir_r: [f32; 64],
    /// Kautz conjugate pole `(r_m, theta_m)` pairs (`4/8/16/32` active).
    pub kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    /// Left complex Kautz readout weights `(c_re, c_im)` (`y_kautz = sum(c_re * s_re - c_im * s_im)`).
    pub kautz_c_l: [(f32, f32); MAX_KAUTZ_POLES],
    /// Right complex Kautz readout weights `(c_re, c_im)`.
    pub kautz_c_r: [(f32, f32); MAX_KAUTZ_POLES],
}

impl V3ColdTierModel {
    /// Builds a resting identity/unity cold model for `tier`.
    #[must_use]
    pub fn identity_for_tier(tier: Tier, os2x: bool, mono_out: bool) -> Self {
        let mut model = Self {
            tier,
            os2x,
            mono_out,
            svf: V3Stage1SvfParams::identity_bypass(),
            blocks: [
                V3CascadeBlockCold::default(),
                V3CascadeBlockCold::default(),
                V3CascadeBlockCold::default(),
            ],
            pre_cab_u_gain: [1.0, 1.0],
            pre_cab_w: [[0.0; MAX_CASCADE_LANES]; 2],
            direct_s2_w: [[0.0; MAX_CASCADE_LANES]; 2],
            fir_l: [0.0; 64],
            fir_r: [0.0; 64],
            kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            kautz_c_l: [(0.0, 0.0); MAX_KAUTZ_POLES],
            kautz_c_r: [(0.0, 0.0); MAX_KAUTZ_POLES],
        };
        model.fir_l[0] = 1.0;
        model.fir_r[0] = 1.0;
        model
    }
}

/// 4-lane SIMD lane-group parameter block (`160 B`, packed sequentially `[0..num_groups]`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3LaneGroupHot {
    pub in_u_gain: [f32; 4],
    pub lp_pole: [f32; 4],
    pub lp_one_minus_pole: [f32; 4],
    pub lp_mix: [f32; 4],
    pub hp_mix: [f32; 4],
    pub static_bias: [f32; 4],
    pub env_bias_sens: [f32; 4],
    pub static_bias_phi: [f32; 4],
    pub skip_u_gain: [f32; 4],
    pub prev_skip_gain: [f32; 4],
}

/// Hot Stage II cascade block parameters (`64 B` aligned; active groups + active `G*G` column tiles
/// live strictly at `[0..G]` and `[0..G*G]` with zero unused holes).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3CascadeBlockHot {
    pub atk_alpha: f32,
    pub rel_alpha: f32,
    pub gate_mix: f32,
    pub gate_sharpness: f32,
    pub groups: [V3LaneGroupHot; MAX_GROUPS],
    pub inter_tiles: [[[f32; 4]; 4]; MAX_GROUPS * MAX_GROUPS],
}

impl V3CascadeBlockHot {
    /// Packs `V3CascadeBlockCold` into contiguous `num_groups` slices and column-major `4x4` tiles.
    pub fn from_cold<B: MathBackend>(cold: &V3CascadeBlockCold, num_groups: usize) -> Self {
        let mut out = Self {
            atk_alpha: cold.atk_alpha.clamp(0.001, 0.9999),
            rel_alpha: cold.rel_alpha.clamp(0.001, 0.99995),
            gate_mix: cold.gate_mix.clamp(0.0, 1.0),
            gate_sharpness: cold.gate_sharpness.clamp(0.0, 1.0e4),
            ..Self::default()
        };
        let g_cnt = num_groups.clamp(1, MAX_GROUPS);
        for g in 0..g_cnt {
            let base = g * 4;
            let mut grp = V3LaneGroupHot::default();
            for l in 0..4 {
                let idx = base + l;
                let a = cold.lp_pole[idx].clamp(0.02, 0.9985);
                grp.in_u_gain[l] = cold.in_u_gain[idx];
                grp.lp_pole[l] = a;
                grp.lp_one_minus_pole[l] = 1.0 - a;
                grp.lp_mix[l] = cold.lp_mix[idx];
                grp.hp_mix[l] = cold.hp_mix[idx];
                grp.static_bias[l] = cold.static_bias[idx].clamp(-4.0, 4.0);
                grp.env_bias_sens[l] = cold.env_bias_sens[idx].clamp(-4.0, 4.0);
                grp.skip_u_gain[l] = cold.skip_u_gain[idx];
                grp.prev_skip_gain[l] = cold.prev_skip_gain[idx];
            }
            // Precompute exact backend-matched phi(static_bias) for bit-exact +0.0 DC equilibrium.
            grp.static_bias_phi =
                algebraic_softclip_vec4::<B>(B::Vec4::load(&grp.static_bias)).to_array();
            out.groups[g] = grp;
        }
        for dst_g in 0..g_cnt {
            for src_g in 0..g_cnt {
                let tile_idx = dst_g * g_cnt + src_g;
                let mut col_major = [[0.0_f32; 4]; 4];
                for col in 0..4 {
                    for row in 0..4 {
                        col_major[col][row] = cold.inter_mat[dst_g * 4 + row][src_g * 4 + col];
                    }
                }
                out.inter_tiles[tile_idx] = col_major;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn step_channel<B: MathBackend>(
        &self,
        block_idx: usize,
        num_groups: usize,
        u_s2: f32,
        h_buf: &mut [[f32; 4]; MAX_GROUPS],
        lp_state: &mut [[f32; 4]; MAX_GROUPS],
        env_q: &mut f32,
    ) {
        let u_vec = B::Vec4::splat(u_s2);
        let mut u_pre = [B::Vec4::zero(); MAX_GROUPS];
        let mut h_prev_vecs = [B::Vec4::zero(); MAX_GROUPS];
        let mut h_prev_arr = [[0.0_f32; 4]; MAX_GROUPS];
        if block_idx > 0 {
            for g in 0..num_groups {
                h_prev_arr[g] = h_buf[g];
                h_prev_vecs[g] = B::Vec4::load(&h_prev_arr[g]);
            }
        }

        let mut abs_sum_vec = B::Vec4::zero();
        for dst_g in 0..num_groups {
            let grp = &self.groups[dst_g];
            let mut v = u_vec.mul(B::Vec4::load(&grp.in_u_gain));
            if block_idx > 0 {
                let row_base = dst_g * num_groups;
                for src_g in 0..num_groups {
                    v = B::fma_mat4x4_vec4(
                        v,
                        &self.inter_tiles[row_base + src_g],
                        &h_prev_arr[src_g],
                    );
                }
            }
            let a = B::Vec4::load(&grp.lp_pole);
            let oma = B::Vec4::load(&grp.lp_one_minus_pole);
            let s_prev = B::Vec4::load(&lp_state[dst_g]);
            let s_next = flush_recursive_state_vec4::<B::Vec4>(a.mul(s_prev).fma(oma, v));
            s_next.store(&mut lp_state[dst_g]);

            let lp_m = B::Vec4::load(&grp.lp_mix);
            let hp_m = B::Vec4::load(&grp.hp_mix);
            let up = lp_m.mul(s_next).fma(hp_m, v.sub(s_next));
            abs_sum_vec = abs_sum_vec.add(up.abs());
            u_pre[dst_g] = up;
        }

        // Fixed 4-lane normalization keeps Lite -> Full -> Pro -> Ultra zero-padded lifting bit-exact.
        let det = 0.25_f32 * abs_sum_vec.horizontal_sum();
        let alpha = if det > *env_q {
            self.atk_alpha
        } else {
            self.rel_alpha
        };
        let q = flush_recursive_state_f32(alpha * *env_q + (1.0 - alpha) * det);
        *env_q = q;

        let g_gate = if self.gate_mix > 0.0 {
            let q2 = q * q;
            (1.0 - self.gate_mix)
                + self.gate_mix * (1.0 - recip_nr3_pos_f32(1.0 + self.gate_sharpness * q2))
        } else {
            1.0
        };
        let g_vec = B::Vec4::splat(g_gate);
        let q_vec = B::Vec4::splat(q);

        for dst_g in 0..num_groups {
            let grp = &self.groups[dst_g];
            let up = u_pre[dst_g];
            let x_drv = up.mul(g_vec);
            let bias =
                B::Vec4::load(&grp.static_bias).fma(B::Vec4::load(&grp.env_bias_sens), q_vec);
            let bias0_phi = B::Vec4::load(&grp.static_bias_phi);
            let nl = zero_dc_asym_sat_vec4::<B>(x_drv, bias, bias0_phi);
            let mut h_next = nl.fma(B::Vec4::load(&grp.skip_u_gain), up);
            if block_idx > 0 {
                h_next = h_next.fma(B::Vec4::load(&grp.prev_skip_gain), h_prev_vecs[dst_g]);
            }
            h_next.store(&mut h_buf[dst_g]);
        }
    }
}

/// Per-channel Stage II recursive state (`MAX_GROUPS = 4`).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3CascadeChannelState {
    pub lp_state: [[[f32; 4]; MAX_GROUPS]; NUM_CASCADE_BLOCKS],
    pub env_q: [f32; NUM_CASCADE_BLOCKS],
    pub _pad: f32,
}

impl V3CascadeChannelState {
    /// Resets all Stage II recursive filter and envelope states to `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
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
    /// Populates Stage III FIR taps and Kautz poles/residues from `V3ColdTierModel`.
    pub fn from_cold(model: &V3ColdTierModel) -> Self {
        let mut out = Self::default();
        for seg in 0..MAX_FIR_SEGMENTS {
            for k4 in 0..4 {
                for i in 0..4 {
                    let idx = seg * 16 + k4 * 4 + i;
                    out.fir_taps_l[seg][k4][i] = model.fir_l[idx];
                    out.fir_taps_r[seg][k4][i] = model.fir_r[idx];
                }
            }
        }
        for b in 0..MAX_KAUTZ_BANKS {
            for lane in 0..4 {
                let m = b * 4 + lane;
                let (r_raw, theta_raw) = model.kautz_poles[m];
                let r = r_raw.clamp(0.50, 0.997);
                let theta = theta_raw.clamp(1.0e-3, core::f32::consts::PI - 1.0e-3);
                let b_gamma = sqrt_f64((1.0 - (r as f64) * (r as f64)).max(1.0e-7)) as f32;
                let (cl_re, cl_im) = model.kautz_c_l[m];
                let (cr_re, cr_im) = model.kautz_c_r[m];
                out.kautz_l[b].set_pole_with_b_gamma(lane, r, theta, 0.0, b_gamma, cl_re, cl_im);
                out.kautz_r[b].set_pole_with_b_gamma(lane, r, theta, 0.0, b_gamma, cr_re, cr_im);
            }
        }
        out
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
                let off = s + k4 * 4;
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
                let off = s + k4 * 4;
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

/// Computes the exact active hot working-set bytes read/written per host sample for `tier`,
/// normalized to stereo active operation.
#[must_use]
pub const fn active_hot_bytes_for_tier(tier: Tier, os2x: bool) -> usize {
    let g = tier_groups(tier);
    let f = tier_fir_segments(tier);
    let k = tier_kautz_banks(tier);
    let svf_bytes =
        core::mem::size_of::<V3Stage1SvfParams>() + core::mem::size_of::<V3Stage1SvfState>();
    let os_bytes = if os2x {
        2 * core::mem::size_of::<ChannelOversampler2x>()
    } else {
        0
    };
    let group_params = 3 * g * core::mem::size_of::<V3LaneGroupHot>();
    let inter_bytes = 2 * g * g * core::mem::size_of::<[[f32; 4]; 4]>();
    let cascade_state = 2 * (3 * g * 16 + 16);
    let bridge_bytes = 64 + 4 * g * 16;
    let fir_bytes = 2 * f * 64 + 2 * f * 64;
    let kautz_bytes = 2 * k * core::mem::size_of::<KautzModalBank<4>>();
    svf_bytes
        + os_bytes
        + group_params
        + inter_bytes
        + cascade_state
        + bridge_bytes
        + fir_bytes
        + kautz_bytes
}
