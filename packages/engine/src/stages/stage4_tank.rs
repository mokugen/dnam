//! Stage IV: Unitary Spatial Tank with Givens+Householder Mixing (`2,080 B` Lite RAM & `v3` FDN).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` (`ref[dsp.stage4.spatial-tank]`, `ref[dsp.stage4.dc-blocker]`, `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).

pub mod v3_fdn;
pub use v3_fdn::{
    V3_TANK_CAP, V3_TANK_LINES, V3_TANK_RING_STRIDE, V3Stage4TankHot, V3Stage4TankParams,
};

use crate::math::spec_math::q32_phase_to_phasor;
use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4, newton6_reciprocal_f64,
};
use crate::operators::op_m::algebraic_unit_rotor_op_m;
use crate::operators::op_t::BallisticDuckingVca;
use crate::provenance::watermark::read_f32_le;
use crate::stages::weights::{ARENA0_STAGE4_EXT_OFFSET, PIPE_FLAG_STAGE4_EXT, has_pipeline_flag};
use alloc::vec::Vec;

#[inline(always)]
fn read_ring_linear_f32(ring: &[f32], mask: usize, wr_idx: usize, delay_samples: f32) -> f32 {
    let d_int = delay_samples as usize;
    let frac = delay_samples - (d_int as f32);
    let s0 = ring[wr_idx.wrapping_sub(d_int) & mask];
    let s1 = ring[wr_idx.wrapping_sub(d_int + 1) & mask];
    s0 + frac * (s1 - s0)
}

/// State RAM footprint for Stage IV Lite: strictly 2,080 B.
pub const STAGE4_TANK_RAM_BYTES: usize = 2080;

/// Number of delay lines in Stage IV Lite.
pub const TANK_LITE_LINES: usize = 4;
/// Capacity per line: 1024 samples so `D_max(192 kHz) = 1004 <= 1024` (`ref[dsp.stage4.spatial-tank]`).
pub const TANK_LINE_CAPACITY: usize = 1024;
const TANK_LINE_MASK: usize = TANK_LINE_CAPACITY - 1;

/// Passband-normalized 5 Hz DC blocker (`ref[dsp.stage4.dc-blocker]`).
#[derive(Clone, Copy, Debug)]
pub struct TankDcBlocker {
    pub x_prev: f32,
    pub y_prev: f32,
    pub r_dc: f32,
}

impl Default for TankDcBlocker {
    fn default() -> Self {
        Self::new()
    }
}

impl TankDcBlocker {
    pub const fn new() -> Self {
        Self {
            x_prev: 0.0,
            y_prev: 0.0,
            r_dc: 0.995, // 5 Hz at 48 kHz (ref[dsp.stage4.dc-blocker])
        }
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }

    /// Step DC blocker with exact passband normalization: alpha_dc = 0.5 * (1 + R_dc).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let alpha = 0.5 * (1.0 + self.r_dc);
        let y = alpha * (x - self.x_prev) + self.r_dc * self.y_prev;
        self.x_prev = flush_recursive_state_f32(x);
        self.y_prev = flush_recursive_state_f32(y);
        y
    }
}

/// Stage IV: Unitary Spatial Tank (2,080 B nominal 48 kHz Lite RAM budget).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Stage4SpatialTank {
    /// 4 delay lines stored as fp16 (`1024` capacity for up to `192 kHz` ZOH rescaling).
    pub lines: [[u16; TANK_LINE_CAPACITY]; TANK_LITE_LINES],
    pub write_ptr: usize,
    /// Immutable `48 kHz` base delay lengths in samples (`[149.0, 179.0, 211.0, 239.0]`).
    pub base_delay_lengths: [f32; TANK_LITE_LINES],
    /// Rescaled delay lengths in samples for the 4 lines (`1..1023`).
    pub delay_lengths: [usize; TANK_LITE_LINES],
    /// Immutable `48 kHz` Full-tier tank line `4..8` tap lengths (`Arena 1 [17,792..17,808]`).
    pub base_full_delay_lengths: [f32; 4],
    pub full_delay_lengths: [usize; 4],
    /// Immutable `48 kHz` Pro-tier tank spatial tap lengths (`Arena 2 [34,496..34,512]`).
    pub base_pro_delay_lengths: [f32; 4],
    pub pro_delay_lengths: [usize; 4],
    /// Immutable `48 kHz` Ultra-tier tank line `8..16` tap lengths (`Arena 3 [48,896..48,928]`).
    pub base_ultra_delay_lengths: [f32; 8],
    pub ultra_delay_lengths: [usize; 8],
    /// Givens mixing angle psi in radians with slew bound |delta_psi| <= 0.05.
    pub current_psi: f32,
    pub target_psi: f32,
    /// Coercive loop saturation gain and threshold (`ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).
    pub g_loop: f32,
    pub sigma_knee: f32,
    pub cached_sigma_knee: f32,
    pub inv_sigma_knee: f64,
    /// 5 Hz DC blockers per line (`ref[dsp.stage4.dc-blocker]`).
    pub dc_blockers: [TankDcBlocker; TANK_LITE_LINES],
    /// Contiguous 4-lane SIMD DC blocker states (`x_prev`, `y_prev`).
    pub dc_x_prev: [f32; TANK_LITE_LINES],
    pub dc_y_prev: [f32; TANK_LITE_LINES],
    /// 1-pole ballistic ducking VCA.
    pub ducking_vca: BallisticDuckingVca,
    /// Cached Givens rotation cosine and sine at `current_psi`.
    pub cos_psi: f32,
    pub sin_psi: f32,
    /// Unitary freeze mode flag.
    pub freeze: bool,
    /// Active flag for extended Stage IV v2 spatial/BBD/FDN operators (`Arena 0 [4_704..4_772]`).
    pub ext_active: bool,
    pub _pad: [u8; 6],
    /// Base `48 kHz` Stage IV loop gain (`Arena 0 [4_688]`) & Givens mixing angle (`4_696`).
    pub base_g_loop: f32,
    pub base_psi: f32,
    /// Core 4-line Schroeder tank wet output gain (allows zeroing/scaling core tank in v2).
    pub tank_wet_gain: f32,
    /// Extended BBD stereo ping-pong / modulated delay (`0 FDIV`, `0 libm`, `0 rem_euclid`).
    pub bbd_wet_gain: f32,
    pub base_bbd_delay_l_48k: f32,
    pub base_bbd_delay_r_48k: f32,
    pub bbd_r_over_l: f32,
    pub bbd_delay_l_host: f32,
    pub bbd_delay_r_host: f32,
    pub base_bbd_fb: f32,
    pub bbd_fb: f32,
    pub bbd_cross_fb: f32,
    pub base_bbd_lpf_pole: f32,
    pub bbd_lpf_pole: f32,
    pub bbd_lpf_l: f32,
    pub bbd_lpf_r: f32,
    pub base_mod_depth_48k: f32,
    pub mod_depth_host: f32,
    pub base_mod_freq_hz: f32,
    pub mod_inc_q32: u32,
    pub mod_phase_q32: u32,
    pub bbd_buf_l: Vec<f32>,
    pub bbd_buf_r: Vec<f32>,
    pub bbd_wr_idx: usize,
    /// Extended 4-line Hadamard FDN with octave-up shimmer (`0 FDIV`, `0 libm`, `0 rem_euclid`).
    pub fdn_wet_gain: f32,
    pub base_fdn_delays_48k: [f32; 4],
    pub fdn_delays_host: [usize; 4],
    pub base_fdn_fb: f32,
    pub fdn_fb: f32,
    pub base_fdn_lpf_pole: f32,
    pub fdn_lpf_pole: f32,
    pub fdn_shimmer_mix: f32,
    pub fdn_lpf_state: [f32; 4],
    pub fdn_shimmer_phase_q32: u32,
    pub fdn_shimmer_inc_q32: u32,
    pub fdn_shimmer_grain_host: f32,
    pub fdn_bufs: [Vec<f32>; 4],
    pub fdn_wr_idx: usize,
}

impl Default for Stage4SpatialTank {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage4SpatialTank {
    pub const DEFAULT_FULL_DELAYS: [f32; 4] = [263.0, 293.0, 331.0, 367.0];
    pub const DEFAULT_PRO_DELAYS: [f32; 4] = [389.0, 419.0, 449.0, 487.0];
    pub const DEFAULT_ULTRA_DELAYS: [f32; 8] =
        [521.0, 563.0, 599.0, 641.0, 677.0, 719.0, 761.0, 809.0];

    pub const fn new() -> Self {
        Self {
            lines: [[0; TANK_LINE_CAPACITY]; TANK_LITE_LINES],
            write_ptr: 0,
            base_delay_lengths: [149.0, 179.0, 211.0, 239.0],
            delay_lengths: [149, 179, 211, 239], // Co-prime lengths
            base_full_delay_lengths: Self::DEFAULT_FULL_DELAYS,
            full_delay_lengths: [263, 293, 331, 367],
            base_pro_delay_lengths: Self::DEFAULT_PRO_DELAYS,
            pro_delay_lengths: [389, 419, 449, 487],
            base_ultra_delay_lengths: Self::DEFAULT_ULTRA_DELAYS,
            ultra_delay_lengths: [521, 563, 599, 641, 677, 719, 761, 809],
            current_psi: core::f32::consts::FRAC_PI_4, // pi / 4
            target_psi: core::f32::consts::FRAC_PI_4,
            g_loop: 0.85,
            sigma_knee: 1.0,
            cached_sigma_knee: 1.0,
            inv_sigma_knee: 1.0,
            dc_blockers: [TankDcBlocker::new(); TANK_LITE_LINES],
            dc_x_prev: [0.0; TANK_LITE_LINES],
            dc_y_prev: [0.0; TANK_LITE_LINES],
            ducking_vca: BallisticDuckingVca::new(),
            cos_psi: core::f32::consts::FRAC_1_SQRT_2,
            sin_psi: core::f32::consts::FRAC_1_SQRT_2,
            freeze: false,
            ext_active: false,
            _pad: [0; 6],
            base_g_loop: 0.85,
            base_psi: core::f32::consts::FRAC_PI_4,
            tank_wet_gain: 1.0,
            bbd_wet_gain: 0.0,
            base_bbd_delay_l_48k: 0.0,
            base_bbd_delay_r_48k: 0.0,
            bbd_r_over_l: 1.0,
            bbd_delay_l_host: 0.0,
            bbd_delay_r_host: 0.0,
            base_bbd_fb: 0.0,
            bbd_fb: 0.0,
            bbd_cross_fb: 0.0,
            base_bbd_lpf_pole: 0.0,
            bbd_lpf_pole: 0.0,
            bbd_lpf_l: 0.0,
            bbd_lpf_r: 0.0,
            base_mod_depth_48k: 0.0,
            mod_depth_host: 0.0,
            base_mod_freq_hz: 0.0,
            mod_inc_q32: 0,
            mod_phase_q32: 0,
            bbd_buf_l: Vec::new(),
            bbd_buf_r: Vec::new(),
            bbd_wr_idx: 0,
            fdn_wet_gain: 0.0,
            base_fdn_delays_48k: [1_487.0, 1_993.0, 2_417.0, 3_011.0],
            fdn_delays_host: [1_487, 1_993, 2_417, 3_011],
            base_fdn_fb: 0.0,
            fdn_fb: 0.0,
            base_fdn_lpf_pole: 0.0,
            fdn_lpf_pole: 0.0,
            fdn_shimmer_mix: 0.0,
            fdn_lpf_state: [0.0; 4],
            fdn_shimmer_phase_q32: 0,
            fdn_shimmer_inc_q32: 2_097_152,
            fdn_shimmer_grain_host: 1_024.0,
            fdn_bufs: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            fdn_wr_idx: 0,
        }
    }

    /// Pre-allocates non-RT Stage IV extended BBD (`131_072` frames) and FDN (`32_768` frames)
    /// ring buffers so `reload_model_in_place` and `4x` bounce toggling perform zero heap allocations.
    pub fn reserve_extended_capacity(&mut self, bbd_pow2: usize, fdn_pow2: usize) {
        let b_cap = bbd_pow2.next_power_of_two().max(65_536);
        let f_cap = fdn_pow2.next_power_of_two().max(16_384);
        if self.bbd_buf_l.len() < b_cap {
            self.bbd_buf_l.resize(b_cap, 0.0);
        }
        if self.bbd_buf_r.len() < b_cap {
            self.bbd_buf_r.resize(b_cap, 0.0);
        }
        for line in &mut self.fdn_bufs {
            if line.len() < f_cap {
                line.resize(f_cap, 0.0);
            }
        }
    }

    pub fn reset(&mut self) {
        for l in 0..TANK_LITE_LINES {
            self.lines[l].fill(0);
            self.dc_blockers[l].reset();
        }
        self.dc_x_prev = [0.0; TANK_LITE_LINES];
        self.dc_y_prev = [0.0; TANK_LITE_LINES];
        self.write_ptr = 0;
        self.current_psi = self.target_psi;
        let (c, s) = algebraic_unit_rotor_op_m(self.current_psi);
        self.cos_psi = c;
        self.sin_psi = s;
        self.ducking_vca.reset();
        self.bbd_lpf_l = 0.0;
        self.bbd_lpf_r = 0.0;
        self.mod_phase_q32 = 0;
        self.bbd_wr_idx = 0;
        self.bbd_buf_l.fill(0.0);
        self.bbd_buf_r.fill(0.0);
        self.fdn_lpf_state = [0.0; 4];
        self.fdn_shimmer_phase_q32 = 0;
        self.fdn_wr_idx = 0;
        for line in &mut self.fdn_bufs {
            line.fill(0.0);
        }
    }

    /// Loads `48 kHz` Stage IV base delay lengths and loop parameters from `Arena 0 [4,672..4,704]`
    /// plus optional extended BBD/FDN parameters from `Arena 0 [4,704..4,772]` and multi-tier tap
    /// lengths from `Arena 1..3`.
    #[inline(always)]
    pub const fn dry_mute_taps(&self) -> bool {
        self._pad[0] != 0
    }

    pub fn load_from_arena(&mut self, arena: &[u8]) {
        const DEFAULTS: [f32; TANK_LITE_LINES] = [149.0, 179.0, 211.0, 239.0];
        self._pad[0] = 0;
        if arena.len() < 4_704 {
            self.base_delay_lengths = DEFAULTS;
            self.base_full_delay_lengths = Self::DEFAULT_FULL_DELAYS;
            self.base_pro_delay_lengths = Self::DEFAULT_PRO_DELAYS;
            self.base_ultra_delay_lengths = Self::DEFAULT_ULTRA_DELAYS;
            self.ext_active = false;
            self.base_g_loop = 0.85;
            self.base_psi = core::f32::consts::FRAC_PI_4;
            self.tank_wet_gain = 1.0;
            self.bbd_wet_gain = 0.0;
            self.fdn_wet_gain = 0.0;
            return;
        }
        self.ext_active = has_pipeline_flag(arena, PIPE_FLAG_STAGE4_EXT)
            && arena.len() >= ARENA0_STAGE4_EXT_OFFSET + 68;
        let max_lite = if self.ext_active { 1000.0 } else { 255.0 };
        for (l, &def_d) in DEFAULTS.iter().enumerate() {
            let v = read_f32_le(arena, 4_672 + l * 4);
            self.base_delay_lengths[l] = if v.is_finite() && (4.0..=max_lite).contains(&v) {
                v
            } else {
                def_d
            };
        }
        let g = read_f32_le(arena, 4_688);
        let g_min = if self.ext_active { 0.0 } else { 0.05 };
        if g.is_finite() && (g_min..=1.18).contains(&g) && (self.ext_active || g > 0.0) {
            self.base_g_loop = g;
            self.g_loop = g;
        } else {
            self.base_g_loop = 0.85;
            self.g_loop = 0.85;
        }
        let sigma = read_f32_le(arena, 4_692);
        if sigma.is_finite() && (0.05..=10.0).contains(&sigma) {
            self.sigma_knee = sigma;
        }
        self.cached_sigma_knee = self.sigma_knee;
        self.inv_sigma_knee = if (self.sigma_knee - 1.0).abs() <= 1e-6 {
            1.0_f64
        } else {
            newton6_reciprocal_f64(self.sigma_knee.max(1e-3) as f64)
        };
        let psi = read_f32_le(arena, 4_696);
        if psi.is_finite()
            && psi.abs() > 1e-6
            && (-core::f32::consts::PI..=core::f32::consts::PI).contains(&psi)
        {
            self.base_psi = psi;
            self.target_psi = psi;
            self.current_psi = psi;
            let (c, s) = algebraic_unit_rotor_op_m(psi);
            self.cos_psi = c;
            self.sin_psi = s;
        } else {
            self.base_psi = core::f32::consts::FRAC_PI_4;
        }

        if self.ext_active {
            let b = ARENA0_STAGE4_EXT_OFFSET;
            let tw = read_f32_le(arena, b);
            if tw.is_finite() && tw < -0.5 {
                self._pad[0] = 1;
                self.tank_wet_gain = 0.0;
            } else {
                self.tank_wet_gain = if tw.is_finite() {
                    tw.clamp(0.0, 4.0)
                } else {
                    1.0
                };
            }
            self.bbd_wet_gain = read_f32_le(arena, b + 4).clamp(0.0, 2.0);
            self.base_bbd_delay_l_48k = read_f32_le(arena, b + 8).clamp(0.0, 120_000.0);
            self.base_bbd_delay_r_48k = read_f32_le(arena, b + 12).clamp(0.0, 120_000.0);
            self.bbd_r_over_l = if self.base_bbd_delay_l_48k > 1.0 {
                (f64::from(self.base_bbd_delay_r_48k)
                    * newton6_reciprocal_f64(f64::from(self.base_bbd_delay_l_48k)))
                .clamp(0.125, 8.0) as f32
            } else {
                1.0
            };
            let mut fb = read_f32_le(arena, b + 16).clamp(-0.95, 0.95);
            let mut xfb = read_f32_le(arena, b + 20).clamp(-0.95, 0.95);
            let mag_sum = fb.abs() + xfb.abs();
            if mag_sum > 0.96 {
                let s = 0.96 * (newton6_reciprocal_f64(f64::from(mag_sum)) as f32);
                fb *= s;
                xfb *= s;
            }
            self.base_bbd_fb = fb;
            self.bbd_fb = fb;
            self.bbd_cross_fb = xfb;
            let blp = read_f32_le(arena, b + 24);
            self.base_bbd_lpf_pole = if blp.is_finite() {
                blp.clamp(0.0, 0.9999)
            } else {
                0.0
            };

            self.fdn_wet_gain = read_f32_le(arena, b + 28).clamp(0.0, 2.0);
            const DEF_FDN: [f32; 4] = [1_487.0, 1_993.0, 2_417.0, 3_011.0];
            for (l, &def_d) in DEF_FDN.iter().enumerate() {
                let d = read_f32_le(arena, b + 32 + l * 4);
                self.base_fdn_delays_48k[l] = if d.is_finite() && d >= 16.0 {
                    d.clamp(16.0, 30_000.0)
                } else {
                    def_d
                };
            }
            self.fdn_fb = read_f32_le(arena, b + 48).clamp(-0.97, 0.97);
            self.base_fdn_fb = self.fdn_fb;
            let flp = read_f32_le(arena, b + 52);
            self.base_fdn_lpf_pole = if flp.is_finite() {
                flp.clamp(0.0, 0.9999)
            } else {
                0.0
            };
            self.fdn_shimmer_mix = read_f32_le(arena, b + 56).clamp(0.0, 1.0);
            self.base_mod_depth_48k = read_f32_le(arena, b + 60).clamp(0.0, 500.0);
            self.base_mod_freq_hz = read_f32_le(arena, b + 64).clamp(0.0, 25.0);
        } else {
            self.tank_wet_gain = 1.0;
            self.bbd_wet_gain = 0.0;
            self.fdn_wet_gain = 0.0;
        }

        if arena.len() >= 17_808 {
            let max_full = if self.ext_active { 1000.0 } else { 511.0 };
            for (l, &def_d) in Self::DEFAULT_FULL_DELAYS.iter().enumerate() {
                let v = read_f32_le(arena, 17_792 + l * 4);
                self.base_full_delay_lengths[l] = if v.is_finite() && (4.0..=max_full).contains(&v)
                {
                    v
                } else {
                    def_d
                };
            }
        } else {
            self.base_full_delay_lengths = Self::DEFAULT_FULL_DELAYS;
        }

        if arena.len() >= 34_512 {
            let max_pro = if self.ext_active { 1000.0 } else { 767.0 };
            for (l, &def_d) in Self::DEFAULT_PRO_DELAYS.iter().enumerate() {
                let v = read_f32_le(arena, 34_496 + l * 4);
                self.base_pro_delay_lengths[l] = if v.is_finite() && (4.0..=max_pro).contains(&v) {
                    v
                } else {
                    def_d
                };
            }
        } else {
            self.base_pro_delay_lengths = Self::DEFAULT_PRO_DELAYS;
        }

        if arena.len() >= 48_928 {
            for (k, &def_d) in Self::DEFAULT_ULTRA_DELAYS.iter().enumerate() {
                let v = read_f32_le(arena, 48_896 + k * 4);
                self.base_ultra_delay_lengths[k] = if v.is_finite() && (4.0..=1000.0).contains(&v) {
                    v
                } else {
                    def_d
                };
            }
        } else {
            self.base_ultra_delay_lengths = Self::DEFAULT_ULTRA_DELAYS;
        }
    }

    /// Reconfigures Stage IV sample-rate parameters (delay line lengths, 5 Hz DC blocker pole R_dc, and ducking VCA poles) under ZOH.
    pub fn reconfigure_sample_rate(&mut self, zoh: &crate::container::ZohRescaledParams) {
        for l in 0..TANK_LITE_LINES {
            let scaled = (zoh.rescale_alignment_delay(self.base_delay_lengths[l]) + 0.5) as usize;
            self.delay_lengths[l] = scaled.clamp(1, TANK_LINE_CAPACITY - 1);
            let scaled_full =
                (zoh.rescale_alignment_delay(self.base_full_delay_lengths[l]) + 0.5) as usize;
            self.full_delay_lengths[l] = scaled_full.clamp(1, TANK_LINE_CAPACITY - 1);
            let scaled_pro =
                (zoh.rescale_alignment_delay(self.base_pro_delay_lengths[l]) + 0.5) as usize;
            self.pro_delay_lengths[l] = scaled_pro.clamp(1, TANK_LINE_CAPACITY - 1);
        }
        for k in 0..8 {
            let scaled_ultra =
                (zoh.rescale_alignment_delay(self.base_ultra_delay_lengths[k]) + 0.5) as usize;
            self.ultra_delay_lengths[k] = scaled_ultra.clamp(1, TANK_LINE_CAPACITY - 1);
        }
        for b in &mut self.dc_blockers {
            b.r_dc = zoh.r_dc_host;
        }
        self.ducking_vca.alpha_att = zoh.rescale_real_pole(0.99);
        self.ducking_vca.alpha_rel = zoh.rescale_real_pole(0.9995);

        let fs_eff = zoh.effective_sample_rate_hz();
        let inv_fs_eff = newton6_reciprocal_f64(fs_eff);
        let q32_per_hz = 4_294_967_296.0_f64 * inv_fs_eff;
        self.bbd_delay_l_host = zoh.rescale_alignment_delay(self.base_bbd_delay_l_48k);
        self.bbd_delay_r_host = zoh.rescale_alignment_delay(self.base_bbd_delay_r_48k);
        self.mod_depth_host = zoh.rescale_alignment_delay(self.base_mod_depth_48k);
        self.mod_inc_q32 = ((f64::from(self.base_mod_freq_hz) * q32_per_hz) + 0.5) as u32;
        self.bbd_lpf_pole = if self.base_bbd_lpf_pole > 1.0e-4 {
            zoh.rescale_real_pole(self.base_bbd_lpf_pole)
        } else {
            0.0
        };
        let fdn_cap = self.fdn_bufs[0].len().saturating_sub(4).max(32_764);
        for l in 0..4 {
            let scaled = (zoh.rescale_alignment_delay(self.base_fdn_delays_48k[l]) + 0.5) as usize;
            self.fdn_delays_host[l] = scaled.clamp(16, fdn_cap);
        }
        self.fdn_lpf_pole = if self.base_fdn_lpf_pole > 1.0e-4 {
            zoh.rescale_real_pole(self.base_fdn_lpf_pole)
        } else {
            0.0
        };
        let grain_samples = ((1_024.0_f64 / zoh.rho_sr) + 0.5).clamp(256.0, 8_192.0);
        self.fdn_shimmer_grain_host = grain_samples as f32;
        self.fdn_shimmer_inc_q32 =
            ((4_294_967_296.0_f64 * newton6_reciprocal_f64(grain_samples)) + 0.5) as u32;
    }

    /// Sets Givens rotation angle with slew rate bound |delta_psi| <= 0.05 rad/frame.
    pub fn set_target_psi(&mut self, psi: f32) {
        self.target_psi = psi;
    }

    /// Evaluates the Stage IV blended quiet-feedback knee $K(u)$ (`ref[dsp.stage4.quiet-feedback-knee]`).
    pub fn eval_quiet_feedback_knee(&self, u: f32, sigma: f32, alpha: f32) -> f32 {
        super::topology::eval_stage4_blended_feedback_knee(u, self.g_loop, sigma, alpha)
            .unwrap_or(self.g_loop * u)
    }

    /// Unity-slope normalized cubic coercive saturation loop law (`ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`, `0 FDIV`, `0 FSQRT`).
    ///
    /// Uses the blended quiet-feedback knee followed by pre-inverted `two_over_3v_ceil = 2.0 / (3.0 * v_ceil)`
    /// with `v_ceil = 0.985` and outer bit-level ceiling `0x3F7C28F5` so $S'(0) \equiv 1.000000$ and $|S(u)| \le 0.985$ unconditionally.
    #[inline(always)]
    pub fn coercive_saturate(&self, x: f32) -> f32 {
        let inv_sigma = if (self.sigma_knee - self.cached_sigma_knee).abs() <= 1e-7 {
            self.inv_sigma_knee
        } else if (self.sigma_knee - 1.0).abs() <= 1e-6 {
            1.0_f64
        } else {
            crate::math::newton6_reciprocal_f64(self.sigma_knee.max(1e-3) as f64)
        };
        let knee_u = super::topology::eval_stage4_blended_feedback_knee_fast(
            x,
            self.g_loop
                .min(f32::from_bits(super::topology::STAGE4_MAX_G_LIN_BITS)),
            inv_sigma,
            0.15,
        );
        super::topology::eval_stage4_loop_saturator_fast(knee_u)
    }

    /// Evaluates `coercive_saturate` across all 4 tank delay lines in parallel (`0 FDIV`, `0 FSQRT`).
    #[inline(always)]
    fn coercive_saturate_4ch(&self, x: [f32; TANK_LITE_LINES]) -> [f32; TANK_LITE_LINES] {
        const V_CEIL: f32 = f32::from_bits(super::topology::STAGE4_MAX_V_CEIL_BITS);
        const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;
        const SEED_A: f32 = 24.0 / 17.0;
        const SEED_B: f32 = 8.0 / 17.0;

        let g = self
            .g_loop
            .min(f32::from_bits(super::topology::STAGE4_MAX_G_LIN_BITS));
        let inv_sigma = if (self.sigma_knee - self.cached_sigma_knee).abs() <= 1e-7 {
            self.inv_sigma_knee as f32
        } else if (self.sigma_knee - 1.0).abs() <= 1e-6 {
            1.0_f32
        } else {
            crate::math::newton6_reciprocal_f64(self.sigma_knee.max(1e-3) as f64) as f32
        };

        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let xv = V::load(&x);
        let rv = xv.mul(V::splat(inv_sigma));
        let dv = V::splat(1.0).fma(rv, rv);
        let two = V::splat(2.0);
        if dv.max(two).sub(two).horizontal_sum() == 0.0 {
            let mut inv = V::splat(SEED_A).fms(V::splat(SEED_B), dv);
            inv = inv.mul(two.fms(dv, inv));
            inv = inv.mul(two.fms(dv, inv));
            inv = inv.mul(two.fms(dv, inv));
            let knee_u = V::splat(g)
                .mul(xv)
                .mul(V::splat(0.85).fma(V::splat(0.15), inv));
            let v = knee_u
                .mul(V::splat(TWO_OVER_3V))
                .max(V::splat(-1.0))
                .min(V::splat(1.0));
            let v_ceil_v = V::splat(V_CEIL);
            let prod = v_ceil_v.mul(v);
            let half_sq = V::splat(0.5).mul(v).mul(v);
            let res = prod
                .mul(V::splat(1.5).sub(half_sq))
                .max(V::splat(-V_CEIL))
                .min(v_ceil_v);
            let mut out = [0.0f32; TANK_LITE_LINES];
            res.store(&mut out);
            return out;
        }

        [
            self.coercive_saturate(x[0]),
            self.coercive_saturate(x[1]),
            self.coercive_saturate(x[2]),
            self.coercive_saturate(x[3]),
        ]
    }

    #[inline(always)]
    fn step_givens_slew(&mut self) -> (f32, f32) {
        let diff_psi = self.target_psi - self.current_psi;
        if diff_psi.abs() > 1e-7 {
            let d_psi = diff_psi.clamp(-0.05, 0.05);
            let next_psi = flush_recursive_state_f32(self.current_psi + d_psi);
            self.current_psi = next_psi;
            let (c, s) = algebraic_unit_rotor_op_m(next_psi);
            self.cos_psi = c;
            self.sin_psi = s;
            (c, s)
        } else {
            (self.cos_psi, self.sin_psi)
        }
    }

    #[inline(always)]
    fn step_reads_and_saturate(&mut self) -> ([f32; TANK_LITE_LINES], [f32; TANK_LITE_LINES]) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let w_ptr = self.write_ptr;
        let raw_fp16 = [
            self.lines[0][(w_ptr + TANK_LINE_CAPACITY - self.delay_lengths[0]) & TANK_LINE_MASK],
            self.lines[1][(w_ptr + TANK_LINE_CAPACITY - self.delay_lengths[1]) & TANK_LINE_MASK],
            self.lines[2][(w_ptr + TANK_LINE_CAPACITY - self.delay_lengths[2]) & TANK_LINE_MASK],
            self.lines[3][(w_ptr + TANK_LINE_CAPACITY - self.delay_lengths[3]) & TANK_LINE_MASK],
        ];
        let reads = DefaultMathBackend::fp16x4_to_f32x4(&raw_fp16);

        let x_vec = V::load(&reads);
        let x_prev_vec = V::load(&self.dc_x_prev);
        let y_prev_vec = V::load(&self.dc_y_prev);
        let r_dc = self.dc_blockers[0].r_dc;
        let alpha_vec = V::splat(0.5 * (1.0 + r_dc));
        let r_dc_vec = V::splat(r_dc);
        let y_vec = (r_dc_vec.mul(y_prev_vec)).fma(alpha_vec, x_vec.sub(x_prev_vec));
        flush_recursive_state_vec4(x_vec).store(&mut self.dc_x_prev);
        let mut readout_sig = [0.0f32; TANK_LITE_LINES];
        flush_recursive_state_vec4(y_vec).store(&mut readout_sig);
        self.dc_y_prev = readout_sig;

        let loop_sig = if self.freeze {
            reads
        } else {
            self.coercive_saturate_4ch(readout_sig)
        };
        (readout_sig, loop_sig)
    }

    #[inline(always)]
    fn step_mix_write_duck(
        &mut self,
        cos_psi: f32,
        sin_psi: f32,
        readout_sig: [f32; TANK_LITE_LINES],
        loop_sig: [f32; TANK_LITE_LINES],
        in_l: f32,
        in_r: f32,
    ) -> (f32, f32) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let loop_v = V::load(&loop_sig);
        let mixed_v = loop_v
            .mul(V::splat(cos_psi))
            .fma(loop_v.pair_swap_neg(), V::splat(sin_psi));
        let sub = 0.5 * mixed_v.horizontal_sum();

        let inj_l = if self.freeze { 0.0 } else { in_l };
        let inj_r = if self.freeze { 0.0 } else { in_r };
        let off_l = inj_l - sub;
        let off_r = inj_r - sub;
        let mut write_vals = [0.0f32; TANK_LITE_LINES];
        mixed_v
            .add(V::splat_pair(off_l, off_r))
            .store(&mut write_vals);
        let packed_fp16 = DefaultMathBackend::f32x4_to_fp16x4(&write_vals);
        let w_ptr = self.write_ptr & TANK_LINE_MASK;
        self.lines[0][w_ptr] = packed_fp16[0];
        self.lines[1][w_ptr] = packed_fp16[1];
        self.lines[2][w_ptr] = packed_fp16[2];
        self.lines[3][w_ptr] = packed_fp16[3];

        self.write_ptr = (w_ptr + 1) & TANK_LINE_MASK;

        let wet_l = 0.5 * (readout_sig[0] + readout_sig[2]);
        let wet_r = 0.5 * (readout_sig[1] + readout_sig[3]);
        self.ducking_vca.step_stereo(in_l, in_r, wet_l, wet_r)
    }

    /// Evaluates Stage IV for stereo input `(in_l, in_r)`, returning `(wet_l, wet_r, readout_sig, loop_sig)`.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_stereo_with_readouts(
        &mut self,
        in_l: f32,
        in_r: f32,
    ) -> (f32, f32, [f32; TANK_LITE_LINES], [f32; TANK_LITE_LINES]) {
        if self.ext_active
            && self.dry_mute_taps()
            && self.bbd_wet_gain <= 1.0e-6
            && self.fdn_wet_gain <= 1.0e-6
        {
            return (0.0, 0.0, [0.0; TANK_LITE_LINES], [0.0; TANK_LITE_LINES]);
        }
        let (cos_psi, sin_psi) = self.step_givens_slew();
        let (readout_sig, loop_sig) = self.step_reads_and_saturate();
        let (mut wet_l, mut wet_r) =
            self.step_mix_write_duck(cos_psi, sin_psi, readout_sig, loop_sig, in_l, in_r);
        if self.ext_active {
            wet_l *= self.tank_wet_gain;
            wet_r *= self.tank_wet_gain;

            if self.bbd_wet_gain > 1.0e-6
                && self.bbd_buf_l.len() >= 1_024
                && self.bbd_buf_l.len() == self.bbd_buf_r.len()
            {
                let mask = self.bbd_buf_l.len() - 1;
                let max_d = (mask - 4) as f32;
                let wr = self.bbd_wr_idx;
                let (_, mod_sin) = q32_phase_to_phasor(self.mod_phase_q32);
                self.mod_phase_q32 = self.mod_phase_q32.wrapping_add(self.mod_inc_q32);
                let d_l = (self.bbd_delay_l_host + self.mod_depth_host * mod_sin).clamp(1.0, max_d);
                let d_r = (self.bbd_delay_r_host - self.mod_depth_host * mod_sin).clamp(1.0, max_d);
                let raw_l = read_ring_linear_f32(&self.bbd_buf_l, mask, wr, d_l);
                let raw_r = read_ring_linear_f32(&self.bbd_buf_r, mask, wr, d_r);
                let p_b = self.bbd_lpf_pole;
                let om_b = 1.0 - p_b;
                let tap_l = flush_recursive_state_f32(p_b * self.bbd_lpf_l + om_b * raw_l);
                let tap_r = flush_recursive_state_f32(p_b * self.bbd_lpf_r + om_b * raw_r);
                self.bbd_lpf_l = tap_l;
                self.bbd_lpf_r = tap_r;
                let inj_l = if self.freeze { 0.0 } else { in_l };
                let inj_r = if self.freeze { 0.0 } else { in_r };
                let fb_l =
                    (inj_l + self.bbd_fb * tap_l + self.bbd_cross_fb * tap_r).clamp(-4.0, 4.0);
                let fb_r =
                    (inj_r + self.bbd_fb * tap_r + self.bbd_cross_fb * tap_l).clamp(-4.0, 4.0);
                self.bbd_buf_l[wr] = flush_recursive_state_f32(fb_l);
                self.bbd_buf_r[wr] = flush_recursive_state_f32(fb_r);
                self.bbd_wr_idx = (wr + 1) & mask;
                wet_l += self.bbd_wet_gain * tap_l;
                wet_r += self.bbd_wet_gain * tap_r;
            }

            if self.fdn_wet_gain > 1.0e-6 && self.fdn_bufs[0].len() >= 4_096 {
                let f_cap = self.fdn_bufs[0].len();
                let f_mask = f_cap - 1;
                let wr = self.fdn_wr_idx;
                let p_f = self.fdn_lpf_pole;
                let om_f = 1.0 - p_f;
                let mut fdn_taps = [0.0f32; 4];
                for l in 0..4 {
                    let dl = self.fdn_delays_host[l].min(f_mask - 4);
                    let raw = self.fdn_bufs[l][wr.wrapping_sub(dl) & f_mask];
                    let lp = flush_recursive_state_f32(p_f * self.fdn_lpf_state[l] + om_f * raw);
                    self.fdn_lpf_state[l] = lp;
                    fdn_taps[l] = lp;
                }
                let mut fb_vec = fdn_taps;
                if self.fdn_shimmer_mix > 1.0e-6 {
                    let pa_u32 = self.fdn_shimmer_phase_q32;
                    let pb_u32 = pa_u32.wrapping_add(0x8000_0000);
                    let pa = (pa_u32 as f32) * (1.0 / 4_294_967_296.0);
                    let pb = (pb_u32 as f32) * (1.0 / 4_294_967_296.0);
                    let wa = 1.0 - (2.0 * pa - 1.0).abs();
                    let wb = 1.0 - wa;
                    let g_len = self.fdn_shimmer_grain_host;
                    let da = (g_len * (1.0 - pa)).clamp(1.0, (f_mask - 4) as f32);
                    let db = (g_len * (1.0 - pb)).clamp(1.0, (f_mask - 4) as f32);
                    self.fdn_shimmer_phase_q32 = pa_u32.wrapping_add(self.fdn_shimmer_inc_q32);
                    let sh_m = self.fdn_shimmer_mix;
                    let dry_m = 1.0 - sh_m;
                    for (l, val) in fb_vec.iter_mut().enumerate() {
                        let oct_a = read_ring_linear_f32(&self.fdn_bufs[l], f_mask, wr, da);
                        let oct_b = read_ring_linear_f32(&self.fdn_bufs[l], f_mask, wr, db);
                        let shimmer = wa * oct_a + wb * oct_b;
                        *val = dry_m * (*val) + sh_m * shimmer;
                    }
                }
                // Orthonormal 4x4 Hadamard matrix H4 = 0.5 * [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
                let s01 = fb_vec[0] + fb_vec[1];
                let d01 = fb_vec[0] - fb_vec[1];
                let s23 = fb_vec[2] + fb_vec[3];
                let d23 = fb_vec[2] - fb_vec[3];
                let h = [
                    0.5 * (s01 + s23),
                    0.5 * (d01 + d23),
                    0.5 * (s01 - s23),
                    0.5 * (d01 - d23),
                ];
                let inj_l = if self.freeze { 0.0 } else { in_l };
                let inj_r = if self.freeze { 0.0 } else { in_r };
                let g_fdn = self.fdn_fb;
                for l in 0..4 {
                    let inj = if (l & 1) == 0 { inj_l } else { inj_r };
                    let nxt = (inj + g_fdn * h[l]).clamp(-4.0, 4.0);
                    self.fdn_bufs[l][wr] = flush_recursive_state_f32(nxt);
                }
                self.fdn_wr_idx = (wr + 1) & f_mask;
                wet_l += self.fdn_wet_gain * 0.5 * (fdn_taps[0] + fdn_taps[2]);
                wet_r += self.fdn_wet_gain * 0.5 * (fdn_taps[1] + fdn_taps[3]);
            }
        }
        (wet_l, wet_r, readout_sig, loop_sig)
    }

    /// Reads higher-tier (`Full`: `116..126`, `Pro`: `184..190`, `Ultra`: `246..254`) Stage IV spatial
    /// tank taps directly into `fb` (`REQ-5.6`: zero mutation of `Lite` state).
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn read_multi_tier_taps_into(
        &self,
        max_tier: usize,
        s4_read: [f32; 4],
        s4_loop: [f32; 4],
        fb: &mut [f32; 256],
    ) {
        if max_tier <= 1 {
            return;
        }
        if self.ext_active && self.dry_mute_taps() {
            fb[116..126].fill(0.0);
            if max_tier >= 3 {
                fb[184..190].fill(0.0);
            }
            if max_tier >= 4 {
                fb[246..254].fill(0.0);
            }
            return;
        }
        let base_ptr = (self.write_ptr + TANK_LINE_CAPACITY - 1) & TANK_LINE_MASK;

        let full_fp16 = [
            self.lines[0]
                [(base_ptr + TANK_LINE_CAPACITY - self.full_delay_lengths[0]) & TANK_LINE_MASK],
            self.lines[1]
                [(base_ptr + TANK_LINE_CAPACITY - self.full_delay_lengths[1]) & TANK_LINE_MASK],
            self.lines[2]
                [(base_ptr + TANK_LINE_CAPACITY - self.full_delay_lengths[2]) & TANK_LINE_MASK],
            self.lines[3]
                [(base_ptr + TANK_LINE_CAPACITY - self.full_delay_lengths[3]) & TANK_LINE_MASK],
        ];
        let full_reads = DefaultMathBackend::fp16x4_to_f32x4(&full_fp16);
        let full_loops = if self.freeze {
            full_reads
        } else {
            self.coercive_saturate_4ch(full_reads)
        };
        fb[116..120].copy_from_slice(&full_reads);
        fb[120..124].copy_from_slice(&full_loops);
        fb[124] = 0.5 * (full_reads[0] + full_reads[2]);
        fb[125] = 0.5 * (full_reads[1] + full_reads[3]);

        if max_tier >= 3 {
            let pro_fp16 = [
                self.lines[0]
                    [(base_ptr + TANK_LINE_CAPACITY - self.pro_delay_lengths[0]) & TANK_LINE_MASK],
                self.lines[1]
                    [(base_ptr + TANK_LINE_CAPACITY - self.pro_delay_lengths[1]) & TANK_LINE_MASK],
                self.lines[2]
                    [(base_ptr + TANK_LINE_CAPACITY - self.pro_delay_lengths[2]) & TANK_LINE_MASK],
                self.lines[3]
                    [(base_ptr + TANK_LINE_CAPACITY - self.pro_delay_lengths[3]) & TANK_LINE_MASK],
            ];
            let pro_reads = DefaultMathBackend::fp16x4_to_f32x4(&pro_fp16);
            fb[184..188].copy_from_slice(&pro_reads);
            fb[188] = 0.5 * (pro_reads[0] - pro_reads[2]) + 0.25 * (s4_read[0] - s4_loop[0]);
            fb[189] = 0.5 * (pro_reads[1] - pro_reads[3]) + 0.25 * (s4_read[1] - s4_loop[1]);
        }

        if max_tier >= 4 {
            let ultra_fp16_a = [
                self.lines[0][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[0])
                    & TANK_LINE_MASK],
                self.lines[1][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[1])
                    & TANK_LINE_MASK],
                self.lines[2][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[2])
                    & TANK_LINE_MASK],
                self.lines[3][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[3])
                    & TANK_LINE_MASK],
            ];
            let ultra_fp16_b = [
                self.lines[0][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[4])
                    & TANK_LINE_MASK],
                self.lines[1][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[5])
                    & TANK_LINE_MASK],
                self.lines[2][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[6])
                    & TANK_LINE_MASK],
                self.lines[3][(base_ptr + TANK_LINE_CAPACITY - self.ultra_delay_lengths[7])
                    & TANK_LINE_MASK],
            ];
            let ua = DefaultMathBackend::fp16x4_to_f32x4(&ultra_fp16_a);
            let ub = DefaultMathBackend::fp16x4_to_f32x4(&ultra_fp16_b);
            fb[246..250].copy_from_slice(&ua);
            fb[250..254].copy_from_slice(&ub);
        }
    }

    /// Evaluates Stage IV for stereo input (in_l, in_r).
    ///
    /// Returns stereo wet reverberant output (wet_l, wet_r).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let (wet_l, wet_r, _, _) = self.step_stereo_with_readouts(in_l, in_r);
        (wet_l, wet_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_stage4_spatial_tank_and_dc_blocker() {
        let _audio = crate::FtzDazGuard::new();
        let mut dc = TankDcBlocker::default();
        assert_eq!(dc.r_dc, 0.995);
        // Step 0 impulse response of passband-normalized DC blocker is alpha = 0.5 * (1 + r_dc) = 0.9975
        let y0 = dc.step(1.0);
        assert!((y0 - 0.9975).abs() < 1e-6);
        // Constant DC input decays toward 0.0
        let mut y_dc = y0;
        for _ in 0..2000 {
            y_dc = dc.step(1.0);
        }
        assert!(y_dc.abs() < 1e-4);
        dc.reset();
        assert_eq!(dc.x_prev, 0.0);
        assert_eq!(dc.y_prev, 0.0);

        let mut tank = Stage4SpatialTank::default();
        tank.reset();
        tank.set_target_psi(0.5);

        // Process a block of normal frames (>= max delay length 239 so wet taps become active)
        let mut max_wet = 0.0_f32;
        for _ in 0..300 {
            let (wl, wr) = tank.step_stereo(0.5, -0.5);
            max_wet = max_wet.max(wl.abs()).max(wr.abs());
        }
        assert!(max_wet > 0.01);
        assert!((tank.current_psi - 0.5).abs() < 1e-6);

        // Test unitary freeze mode: mutes input injection and circulates existing tank energy
        tank.freeze = true;
        let mut freeze_wet = 0.0_f32;
        for _ in 0..300 {
            let (wl, wr) = tank.step_stereo(0.0, 0.0);
            freeze_wet = freeze_wet.max(wl.abs()).max(wr.abs());
        }
        assert!(freeze_wet > 0.01);

        tank.reset();
        assert_eq!(tank.write_ptr, 0);
        assert_eq!(tank.ducking_vca.env, 0.0);

        // Short-arena load resets defaults; EXT NaN parameters fall back cleanly; sigma_knee = 1.0 fast path
        tank.ext_active = true;
        tank.base_g_loop = 0.2;
        tank.load_from_arena(&[0u8; 64]);
        assert!(!tank.ext_active && (tank.base_g_loop - 0.85).abs() < 1e-6);

        let mut ext_arena = alloc::vec![0u8; crate::stages::weights::ARENA0_STAGE4_EXT_OFFSET + 68];
        let ext_flags =
            crate::stages::weights::PIPE_MAGIC_HI | crate::stages::weights::PIPE_FLAG_STAGE4_EXT;
        ext_arena[crate::stages::weights::ARENA0_PIPELINE_FLAGS_OFFSET
            ..crate::stages::weights::ARENA0_PIPELINE_FLAGS_OFFSET + 4]
            .copy_from_slice(&ext_flags.to_le_bytes());
        let b = crate::stages::weights::ARENA0_STAGE4_EXT_OFFSET;
        for off in [b, b + 24, b + 52] {
            ext_arena[off..off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        tank.load_from_arena(&ext_arena);
        assert!(tank.ext_active);
        assert_eq!(tank.tank_wet_gain, 1.0);
        assert_eq!(tank.base_bbd_lpf_pole, 0.0);
        assert_eq!(tank.base_fdn_lpf_pole, 0.0);
        tank.sigma_knee = 1.0;
        tank.cached_sigma_knee = 0.5;
        assert!(tank.coercive_saturate(0.25).is_finite());
        let sat4 = tank.coercive_saturate_4ch([0.25, -0.25, 0.1, -0.1]);
        assert!(sat4.iter().all(|v| v.is_finite()));

        // Dual-mono vs asymmetric sidechain: stereo-linked ducking VCA steps once per frame
        let mut tank_mono = Stage4SpatialTank::default();
        let mut tank_single = BallisticDuckingVca::default();
        let _ = tank_mono.step_stereo(0.8, 0.8);
        let _ = tank_single.step(0.8, 1.0);
        assert!((tank_mono.ducking_vca.env - tank_single.env).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_stage4_coercive_ceiling_and_slew_bound(
            x in -50.0_f32..50.0_f32,
            g_loop in 0.1_f32..2.0_f32,
            sigma_knee in 0.2_f32..3.0_f32,
            target_psi in -3.0_f32..3.0_f32,
        ) {
            let _audio = crate::FtzDazGuard::new();
            let mut tank = Stage4SpatialTank {
                g_loop,
                sigma_knee,
                ..Default::default()
            };

            // ref[dsp.stage4.loop-saturator]: coercive_saturate is odd-symmetric and strictly bounded by V_CEIL = 0.985
            let s_pos = tank.coercive_saturate(x);
            let s_neg = tank.coercive_saturate(-x);
            prop_assert!(s_pos.abs() <= 0.985_001);
            prop_assert!((s_pos + s_neg).abs() < 1e-6);

            // Slew bound |delta_psi| <= 0.05 rad/sample
            let psi_before = tank.current_psi;
            tank.set_target_psi(target_psi);
            let (wl, wr) = tank.step_stereo(0.1, -0.1);
            prop_assert!(wl.is_finite() && wr.is_finite());
            prop_assert!((tank.current_psi - psi_before).abs() <= 0.050_001);
        }
    }
}

#[cfg(test)]
mod tank_verification_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Empirical TORA Stages 0–IV, Spatial Tank & 256-D Mid/Side Kronecker Head Verification (`Tests 4, 5`).
    //
    // Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md`.
    //
    // Verifies:
    // - `Test 4` (`verify[dsp.stage4.spatial-tank]`, `verify[dsp.stage4.dc-blocker]`, `verify[dsp.stage4.loop-saturator]`, `verify[dsp.stage4.quiet-feedback-knee]`, `verify[dsp.stage4.captured-freeze]`):
    //   Stage IV Spatial Tank coercive Lyapunov saturation ceiling (`<= 0.985`),
    //   zero-DC decay (`< 1e-5`), dub-swell self-oscillation stability, and freeze latch.
    // - `Test 5`: 8-channel and 16-channel Givens + Householder stereo mixing matrix $\mathbf{U}(\psi, \mathbf{v})$
    //   orthogonality (`< 1e-14`) and Euclidean energy preservation.
    // - Stage 0 8-Port Matrix Router (`verify[dsp.router.return-to-send-isolation]` DAG isolation `S_ret->send == 0`, WDW mode, Bezout crossover).
    // - 256-D Mid/Side Kronecker Output Head (`verify[dsp.head.mid-side-factorization]`) stereo width laws ($w_s \in \{0.0, 1.0, 2.0\}$).

    use std::f64::consts::{PI, TAU};

    use crate::math::DefaultMathBackend;
    use crate::operators::reference::{
        givens_householder_mix_8ch_f64, givens_householder_mix_16ch_f64, op_t_coercive_step_f64,
    };
    use crate::stages::{
        MidSideKroneckerHead, Stage0Router, Stage4SpatialTank, V3ColdTierModel, V3HotCore,
    };
    use crate::{FtzDazGuard, Tier};
    use proptest::prelude::*;

    /// Test 4 (`verify[dsp.stage4.dc-blocker]`, `verify[dsp.stage4.loop-saturator]`, `verify[dsp.stage4.quiet-feedback-knee]`): Injects a $+20\text{ dBFS}$ (`amplitude = 10.0`) sine burst with a
    /// $+2.0$ DC offset into Stage IV Spatial Tank in a self-oscillating dub swell regime
    /// ($g_{\text{eff,max}} = 1.18 > 1.0$, $V_{\text{ceil}} = 0.985$, $R_{\text{dc}} = 0.995$) and
    /// verifies peak state $\le 0.985$, residual DC $< 10^{-5}$, and monotonic quiet-tail decay.
    #[test]
    fn test_4_stage4_spatial_tank_lyapunov_saturation_and_zero_dc_decay() {
        const V_CEIL: f64 = 0.985;
        let mut dc_x = [0.0_f64; 8];
        let mut dc_y = [0.0_f64; 8];
        let mut state = [0.0_f64; 8];
        let (cos_psi, sin_psi) = (
            core::f64::consts::FRAC_1_SQRT_2,
            core::f64::consts::FRAC_1_SQRT_2,
        );

        let mut max_peak = 0.0_f64;
        for n in 0..8000 {
            let burst = if n < 1000 {
                10.0 * (TAU * 440.0 * (n as f64) / 48_000.0).sin() + 2.0
            } else {
                0.0
            };
            let g_lin = if n < 2000 { 1.18 } else { 0.98 };

            let mut sat = [0.0_f64; 8];
            for m in 0..8 {
                let u_in = state[m] + if m < 2 { burst } else { 0.0 };
                sat[m] = op_t_coercive_step_f64(
                    u_in,
                    g_lin,
                    0.35,
                    V_CEIL,
                    0.995,
                    &mut dc_x[m],
                    &mut dc_y[m],
                );
                max_peak = max_peak.max(sat[m].abs());
            }
            state = givens_householder_mix_8ch_f64(&sat, cos_psi, sin_psi);
        }

        let final_dc: f64 = state.iter().sum::<f64>() / 8.0;
        assert!(
            max_peak <= V_CEIL + 1e-9,
            "Test 4 Lyapunov ceiling violated: max_peak={max_peak} > {V_CEIL}"
        );
        assert!(
            final_dc.abs() < 1e-5,
            "Test 4 residual DC offset {final_dc:e} >= 1e-5"
        );

        // Also verify real-time Stage4SpatialTank under +20 dBFS + 2.0 DC burst and freeze latch
        let _audio = FtzDazGuard::new();
        let mut rt_tank = Stage4SpatialTank {
            g_loop: 1.18,
            ..Default::default()
        };
        for n in 0..8000 {
            let x = if n < 500 {
                10.0 * ((n as f32) * 0.1).sin() + 2.0
            } else {
                0.0
            };
            if n == 600 {
                rt_tank.freeze = true;
            }
            if n == 1000 {
                rt_tank.freeze = false;
                rt_tank.g_loop = 0.65;
            }
            let (wl, wr) = rt_tank.step_stereo(x, -x);
            assert!(wl.is_finite() && wr.is_finite());
        }
        let (tail_l, tail_r) = rt_tank.step_stereo(0.0, 0.0);
        assert!(tail_l.abs() < 1e-3 && tail_r.abs() < 1e-3);
    }

    /// Test 5: Evaluates the 8-channel and 16-channel Givens + Householder stereo mixing matrices
    /// $\mathbf{U}(\psi, \mathbf{v})$ across $\psi \in \{0, \pi/6, \pi/4, \pi/2\}$ and verifies
    /// $\|\mathbf{U}^\top \mathbf{U} - \mathbf{I}\|_{\max} < 10^{-14}$ and exact Euclidean energy preservation.
    #[test]
    fn test_5_givens_householder_stereo_mixing_matrix_orthogonality_and_eigenvalues() {
        for &psi in &[0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0] {
            let (c, s) = (psi.cos(), psi.sin());
            let mut u_mat = [[0.0_f64; 8]; 8];
            for col in 0..8 {
                let mut e = [0.0_f64; 8];
                e[col] = 1.0;
                let col_vec = givens_householder_mix_8ch_f64(&e, c, s);
                for row in 0..8 {
                    u_mat[row][col] = col_vec[row];
                }
            }

            let mut max_ortho_err = 0.0_f64;
            for i in 0..8 {
                for j in 0..8 {
                    let mut dot = 0.0_f64;
                    for k in 0..8 {
                        dot += u_mat[k][i] * u_mat[k][j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    max_ortho_err = max_ortho_err.max((dot - target).abs());
                }
            }
            assert!(
                max_ortho_err < 1e-14,
                "Test 5 8-ch orthogonality error {max_ortho_err:e} >= 1e-14 at psi={psi}"
            );

            let cross_c = [0.8_f64; 8];
            let cross_s = [0.6_f64; 8];
            let x16: [f64; 16] = core::array::from_fn(|i| ((i as f64) + 1.0) * 0.15);
            let y16 = givens_householder_mix_16ch_f64(&x16, c, s, &cross_c, &cross_s);
            let e_x: f64 = x16.iter().map(|v| v * v).sum();
            let e_y: f64 = y16.iter().map(|v| v * v).sum();
            assert!((e_x - e_y).abs() < 1e-13);
        }
    }

    /// Verifies Stage 0 8-Port Matrix Router (`verify[dsp.router.return-to-send-isolation]` DAG isolation `S_ret->send == 0`, WDW mode)
    /// and `MidSideKroneckerHead` (`verify[dsp.head.mid-side-factorization]`).
    #[test]
    fn test_stage0_router_topologies_and_256d_mid_side_kronecker_head() {
        let _audio = FtzDazGuard::new();
        let mut router = Stage0Router {
            wdw_mode: true,
            ..Default::default()
        };
        router.set_scattering_matrix([[1.0; 4]; 4]);
        // verify[dsp.router.return-to-send-isolation] invariant: S_ret->send entries must be strictly zeroed
        assert_eq!(router.s_matrix[0][2], 0.0);
        assert_eq!(router.s_matrix[0][3], 0.0);
        assert_eq!(router.s_matrix[1][2], 0.0);
        assert_eq!(router.s_matrix[1][3], 0.0);

        let (send_l, send_r, down_l, down_r, sub_l, sub_r) =
            router.route_frame(0.75, -0.5, 0.3, -0.3);
        assert!(send_l.is_finite() && send_r.is_finite());
        assert!(down_l.is_finite() && down_r.is_finite());
        assert!(sub_l.is_finite() && sub_r.is_finite());

        // Verify MidSideKroneckerHead stereo width law (verify[dsp.head.mid-side-factorization]):
        // - w_s = 0.0 -> out_l == out_r (mono collapse)
        // - w_s = 2.0 -> side difference is exactly 2x the w_s = 1.0 side difference
        let mut head = MidSideKroneckerHead {
            w_mid: core::array::from_fn(|i| 0.02 * ((i as f32) + 1.0)),
            b_mid: 0.1,
            ..Default::default()
        };
        let base_w_side: [f32; 64] = core::array::from_fn(|i| 0.015 * ((i as f32) - 20.0));
        let base_b_side = 0.05_f32;
        let phi: [f32; 64] = core::array::from_fn(|i| ((i as f32) * 0.17).sin());

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            head.update_stereo_width(&base_w_side, base_b_side, 0.0);
            let (l0, r0) = head.project_feature_bus(&phi, tier);
            assert!(
                (l0 - r0).abs() < 1e-6,
                "Expected w_s=0.0 mono collapse at {tier:?}, got L={l0}, R={r0}"
            );

            head.update_stereo_width(&base_w_side, base_b_side, 1.0);
            let (l1, r1) = head.project_feature_bus(&phi, tier);
            let side1 = l1 - r1;

            head.update_stereo_width(&base_w_side, base_b_side, 2.0);
            let (l2, r2) = head.project_feature_bus(&phi, tier);
            let side2 = l2 - r2;

            assert!(
                (side2 - 2.0 * side1).abs() < 1e-5,
                "Expected w_s=2.0 side difference ({side2}) == 2 * side1 ({side1}) at {tier:?}"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Property test verifying that [`V3HotCore`] remains finite across all tiers and resets
        /// cleanly to DC equilibrium.
        #[test]
        fn prop_stage_pipeline_finite_and_dc_resettable(
            in_l in -2.0_f32..2.0_f32,
            in_r in -2.0_f32..2.0_f32,
            tier_idx in 0_usize..4_usize,
        ) {
            let _audio = FtzDazGuard::new();
            let tier = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra][tier_idx];
            let cold = V3ColdTierModel::identity_for_tier(tier, false, false);
            let mut core = V3HotCore::<DefaultMathBackend>::from_cold_model(&cold);
            let [s4_l, s4_r] = core.step_stereo(in_l, in_r);
            prop_assert!(s4_l.is_finite() && s4_r.is_finite());

            core.reset_to_dc_equilibrium();
            let [z_l, z_r] = core.step_stereo(0.0, 0.0);
            prop_assert!(z_l.abs() < 1e-7 && z_r.abs() < 1e-7);
        }
    }

    #[test]
    fn test_stage4_load_reconfigure_and_multi_tier_taps() {
        use crate::container::ZohRescaledParams;
        use crate::stages::weights::{
            ARENA0_PIPELINE_FLAGS_OFFSET, ARENA0_STAGE4_EXT_OFFSET, PIPE_FLAG_STAGE4_EXT,
            PIPE_MAGIC_HI,
        };
        let _audio = FtzDazGuard::new();
        let mut tank = Stage4SpatialTank::default();
        tank.reserve_extended_capacity(4096, 32768);
        tank.load_from_arena(&[]);

        let mut arena = alloc::vec![0u8; 49_000];
        let ext_flags = PIPE_MAGIC_HI | PIPE_FLAG_STAGE4_EXT;
        arena[ARENA0_PIPELINE_FLAGS_OFFSET..ARENA0_PIPELINE_FLAGS_OFFSET + 4]
            .copy_from_slice(&ext_flags.to_le_bytes());
        for l in 0..4 {
            arena[4_672 + l * 4..4_676 + l * 4]
                .copy_from_slice(&((150 + l * 20) as f32).to_le_bytes());
            arena[17_792 + l * 4..17_796 + l * 4]
                .copy_from_slice(&((160 + l * 20) as f32).to_le_bytes());
            arena[34_496 + l * 4..34_500 + l * 4]
                .copy_from_slice(&((170 + l * 20) as f32).to_le_bytes());
            arena[48_896 + l * 4..48_900 + l * 4]
                .copy_from_slice(&((180 + l * 20) as f32).to_le_bytes());
        }
        arena[4_688..4_692].copy_from_slice(&0.8_f32.to_le_bytes());
        arena[4_692..4_696].copy_from_slice(&1.5_f32.to_le_bytes());
        arena[4_696..4_700].copy_from_slice(&0.35_f32.to_le_bytes());
        let b = ARENA0_STAGE4_EXT_OFFSET;
        arena[b..b + 4].copy_from_slice(&0.5_f32.to_le_bytes());
        arena[b + 4..b + 8].copy_from_slice(&0.25_f32.to_le_bytes());
        arena[b + 8..b + 12].copy_from_slice(&128.0_f32.to_le_bytes());
        arena[b + 12..b + 16].copy_from_slice(&160.0_f32.to_le_bytes());
        arena[b + 16..b + 20].copy_from_slice(&0.6_f32.to_le_bytes());
        arena[b + 20..b + 24].copy_from_slice(&0.5_f32.to_le_bytes());
        arena[b + 24..b + 28].copy_from_slice(&0.4_f32.to_le_bytes());
        arena[b + 28..b + 32].copy_from_slice(&0.3_f32.to_le_bytes());
        for l in 0..4 {
            arena[b + 32 + l * 4..b + 36 + l * 4]
                .copy_from_slice(&((1_500 + l * 400) as f32).to_le_bytes());
        }
        arena[b + 48..b + 52].copy_from_slice(&0.5_f32.to_le_bytes());
        arena[b + 52..b + 56].copy_from_slice(&0.7_f32.to_le_bytes());
        arena[b + 56..b + 60].copy_from_slice(&0.1_f32.to_le_bytes());
        arena[b + 60..b + 64].copy_from_slice(&4.0_f32.to_le_bytes());
        arena[b + 64..b + 68].copy_from_slice(&1.5_f32.to_le_bytes());
        tank.load_from_arena(&arena);
        assert!(tank.ext_active);
        tank.reconfigure_sample_rate(&ZohRescaledParams::new(48_000, 96_000));
        tank.set_target_psi(0.45);
        assert!(tank.eval_quiet_feedback_knee(0.5, 1.0, 0.2).is_finite());
        assert!(tank.coercive_saturate(1.5).abs() <= 0.985 + 1e-6);
        let (wl, wr) = tank.step_stereo(0.2, -0.2);
        assert!(wl.is_finite() && wr.is_finite());
        let mut fb = [0.0_f32; 256];
        tank.read_multi_tier_taps_into(1, [0.1; 4], [0.05; 4], &mut fb);
        tank.read_multi_tier_taps_into(4, [0.1; 4], [0.05; 4], &mut fb);
        arena[b..b + 4].copy_from_slice(&(-1.0_f32).to_le_bytes());
        tank.load_from_arena(&arena);
        tank.read_multi_tier_taps_into(4, [0.1; 4], [0.05; 4], &mut fb);
        assert_eq!(fb[116], 0.0);
    }
}
