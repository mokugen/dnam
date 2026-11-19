//! Branch-free `Q0.32` quadrature LFO, Catmull-Rom delay readers, partitioned Schroeder all-pass
//! diffusion, and `0-ULP` frozen integer-lag hold (`stage4_tank/v3_fdn/lfo_interp.rs`).

use alloc::boxed::Box;
use dnam_kernels::step_schroeder_quad_vec4;

use super::{Stage4QuadParams, V3_TANK_CAP, V3_TANK_LINES, V3_TANK_MASK};
use crate::math::{
    FP16_MAX_FINITE_F32, IEEE754_F32_EXP_BIAS, MathBackend, SimdVec4,
    const_hermite5_quarter_sine_coeffs, flush_recursive_state_f32, flush_recursive_state_vec4,
    fp16::fp16_to_f32, newton6_reciprocal_f64,
};
use crate::operators::op_t::BallisticDuckingVca;
use crate::stages::topology::STAGE4_MAX_V_CEIL_BITS;

/// Partitioned cold-allocated power-of-two Schroeder all-pass diffuser bank (`4` parallel lanes).
///
/// Each active lane (`delay >= 1`) occupies an independent segment `[l * seg_cap .. (l + 1) * seg_cap]`
/// with `& seg_mask` wrap (`0 UDIV`/`IDIV`) and exact `0-ULP` write lock during `freeze`.
#[derive(Clone, Debug)]
pub(crate) struct PartitionedSchroederBank {
    pub(super) buf: Box<[f32]>,
    pub(super) seg_cap: usize,
    pub(super) seg_mask: usize,
    pub(super) delays: [usize; V3_TANK_LINES],
    pub(super) gains: [f32; V3_TANK_LINES],
    pub(super) wr: usize,
    pub(super) active_mask: [bool; V3_TANK_LINES],
    pub(super) any_active: bool,
}

impl PartitionedSchroederBank {
    #[must_use]
    pub(crate) fn from_params(
        delays: [usize; V3_TANK_LINES],
        gains: [f32; V3_TANK_LINES],
    ) -> Option<Self> {
        let max_d = delays.iter().copied().max().unwrap_or(0);
        if max_d == 0 {
            return None;
        }
        let seg_cap = (max_d + 4).saturating_mul(4).max(64).next_power_of_two();
        let mut clamped_g = [0.0_f32; V3_TANK_LINES];
        let mut active_mask = [false; V3_TANK_LINES];
        for l in 0..V3_TANK_LINES {
            clamped_g[l] = if gains[l].is_finite() {
                gains[l].clamp(-0.75, 0.75)
            } else {
                0.0
            };
            active_mask[l] = delays[l] > 0;
        }
        Some(Self {
            buf: alloc::vec![0.0_f32; seg_cap * V3_TANK_LINES].into_boxed_slice(),
            seg_cap,
            seg_mask: seg_cap - 1,
            delays,
            gains: clamped_g,
            wr: 0,
            active_mask,
            any_active: active_mask.iter().any(|&a| a),
        })
    }

    pub(crate) fn ensure_cap(
        &mut self,
        delays: [usize; V3_TANK_LINES],
        gains: [f32; V3_TANK_LINES],
    ) {
        let max_d = delays.iter().copied().max().unwrap_or(0);
        let needed_cap = if max_d == 0 {
            self.seg_cap
        } else {
            (max_d + 4).max(64).next_power_of_two()
        };
        if needed_cap > self.seg_cap {
            let new_cap = (max_d + 4).saturating_mul(4).max(64).next_power_of_two();
            self.buf = alloc::vec![0.0_f32; new_cap * V3_TANK_LINES].into_boxed_slice();
            self.seg_cap = new_cap;
            self.seg_mask = new_cap - 1;
            self.wr = 0;
        }
        self.apply_params(delays, gains);
    }

    pub(crate) fn apply_params(
        &mut self,
        delays: [usize; V3_TANK_LINES],
        gains: [f32; V3_TANK_LINES],
    ) {
        let max_d = self.seg_cap.saturating_sub(1);
        for l in 0..V3_TANK_LINES {
            self.active_mask[l] = delays[l] > 0;
            self.delays[l] = if delays[l] > 0 {
                delays[l].clamp(1, max_d)
            } else {
                0
            };
            self.gains[l] = if gains[l].is_finite() {
                gains[l].clamp(-0.75, 0.75)
            } else {
                0.0
            };
        }
        self.any_active = self.active_mask.iter().any(|&a| a);
    }

    pub(crate) fn reset(&mut self) {
        self.buf.fill(0.0);
        self.wr = 0;
    }

    #[inline(always)]
    pub(super) fn step_quad<B: MathBackend, const FREEZE: bool>(&mut self, x: B::Vec4) -> B::Vec4 {
        if !self.any_active {
            return x;
        }
        let wr = self.wr & self.seg_mask;
        let mut del = [0.0_f32; V3_TANK_LINES];
        for l in 0..V3_TANK_LINES {
            let r_idx = (wr + self.delays[l]) & self.seg_mask;
            del[l] = self.buf[l * self.seg_cap + r_idx];
        }
        let (y_v, w_v) =
            step_schroeder_quad_vec4::<B::Vec4>(x, B::Vec4::load(&del), B::Vec4::load(&self.gains));
        let x_arr = x.to_array();
        let y_arr = y_v.to_array();
        let mut out = [0.0_f32; V3_TANK_LINES];
        for l in 0..V3_TANK_LINES {
            out[l] = if self.active_mask[l] {
                flush_recursive_state_f32(y_arr[l])
            } else {
                x_arr[l]
            };
        }
        if !FREEZE {
            let w_arr = w_v.to_array();
            for l in 0..V3_TANK_LINES {
                if self.active_mask[l] {
                    self.buf[l * self.seg_cap + wr] = flush_recursive_state_f32(w_arr[l]);
                }
            }
            self.wr = wr.wrapping_sub(1) & self.seg_mask;
        }
        B::Vec4::load(&out)
    }
}

/// Smooth $C^1$-peak degree-5 odd triangle-Hermite sine on `Q0.32` phase (`0` `libm`, `0` branches).
#[inline(always)]
pub(super) fn q32_smooth_sin(p: u32) -> f32 {
    const SCALE_Q30: f32 = f32::from_bits((IEEE754_F32_EXP_BIAS - 30) << 23);
    const COEFFS: (f32, f32, f32) = const_hermite5_quarter_sine_coeffs();
    const C1: f32 = COEFFS.0;
    const C3: f32 = COEFFS.1;
    const C5: f32 = COEFFS.2;
    let z_abs = (p.wrapping_sub(0x4000_0000) as i32).unsigned_abs();
    let tri = 0x4000_0000_u32.wrapping_sub(z_abs) as i32;
    let u = (tri as f32) * SCALE_Q30;
    let u2 = u * u;
    (u * (C1 - u2 * (C3 - C5 * u2))).clamp(-1.0, 1.0)
}

/// Evaluates Left and Right modulation excursions `(mod_l, mod_r)` in samples from `Q0.32` phase.
#[inline(always)]
pub(super) fn lfo_offsets_from_phase(
    lfo_phase_q32: u32,
    depth_samples: f32,
    cos_stereo: f32,
    sin_stereo: f32,
) -> (f32, f32) {
    let s_phi = q32_smooth_sin(lfo_phase_q32);
    let c_phi = q32_smooth_sin(lfo_phase_q32.wrapping_add(0x4000_0000));
    let mod_l = depth_samples * s_phi;
    let mod_r = depth_samples * (s_phi * cos_stereo + c_phi * sin_stereo);
    (mod_l, mod_r)
}

/// Computes `(lfo_inc_q32, half_alpha_cubic)` on the cold path with zero hardware division.
#[inline]
pub(super) fn cold_lfo_inc_and_half_alpha(
    lfo_rate_hz: f32,
    lfo_depth_samples: f32,
    g_loop: f32,
    sample_rate_hz: f64,
) -> (u32, f32) {
    let inc_q32 = if lfo_rate_hz <= 0.0 || lfo_depth_samples <= 0.0 {
        0
    } else {
        let inv_sr = newton6_reciprocal_f64(sample_rate_hz.max(8_000.0));
        ((f64::from(lfo_rate_hz) * 4_294_967_296.0 * inv_sr + 0.5) as u64) as u32
    };
    let g_abs = f64::from(g_loop.abs()).max(0.5);
    let alpha_cubic = ((0.9992 * newton6_reciprocal_f64(g_abs) - 1.0) * 4.0).clamp(0.0, 1.0);
    (inc_q32, (0.5 * alpha_cubic) as f32)
}

/// Ring abstraction decoupling inline frame-major `[[u16; 4]; 1024]` L1D buffers from cold-allocated
/// power-of-two extended delay rings (`ext_lines` in `"fp16"` or `"f32"` format).
pub(crate) trait TankRing {
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4;
    fn read_frame_f32(&self, frame: usize, line: usize) -> f32;
    fn write_frame<B: MathBackend>(&mut self, wr: usize, val: B::Vec4) -> usize;
}

/// Frame-major inline `fp16` ring (`128` frames = `1,024 B` = `16` cache lines).
///
/// Active `Lite` delays `[53, 67, 79, 89]` touch only `(89 + 1) * 8 = 720` contiguous bytes.
pub(super) struct InlineFrameRing<'a>(pub(super) &'a mut [[u16; V3_TANK_LINES]; V3_TANK_CAP]);

impl TankRing for InlineFrameRing<'_> {
    #[inline(always)]
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4 {
        let quad = [
            self.0[base & V3_TANK_MASK][line],
            self.0[base.wrapping_add(1) & V3_TANK_MASK][line],
            self.0[base.wrapping_add(2) & V3_TANK_MASK][line],
            self.0[base.wrapping_add(3) & V3_TANK_MASK][line],
        ];
        B::fp16x4_to_vec4(&quad)
    }

    #[inline(always)]
    fn read_frame_f32(&self, frame: usize, line: usize) -> f32 {
        fp16_to_f32(self.0[frame & V3_TANK_MASK][line])
    }

    #[inline(always)]
    fn write_frame<B: MathBackend>(&mut self, wr: usize, val: B::Vec4) -> usize {
        let w = wr & V3_TANK_MASK;
        self.0[w] = B::vec4_to_fp16x4(val);
        w.wrapping_sub(1) & V3_TANK_MASK
    }
}

pub(super) struct ExtFrameRingFp16<'a> {
    pub(super) buf: &'a mut [u16],
    pub(super) mask: usize,
}

impl TankRing for ExtFrameRingFp16<'_> {
    #[inline(always)]
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4 {
        let m = self.mask;
        let q = [
            self.buf[((base & m) << 2) + line],
            self.buf[((base.wrapping_add(1) & m) << 2) + line],
            self.buf[((base.wrapping_add(2) & m) << 2) + line],
            self.buf[((base.wrapping_add(3) & m) << 2) + line],
        ];
        B::fp16x4_to_vec4(&q)
    }

    #[inline(always)]
    fn read_frame_f32(&self, frame: usize, line: usize) -> f32 {
        fp16_to_f32(self.buf[((frame & self.mask) << 2) + line])
    }

    #[inline(always)]
    fn write_frame<B: MathBackend>(&mut self, wr: usize, val: B::Vec4) -> usize {
        let w = wr & self.mask;
        let base = w << 2;
        let packed = B::vec4_to_fp16x4(val);
        if base + 3 < self.buf.len() {
            self.buf[base..base + 4].copy_from_slice(&packed);
        }
        w.wrapping_sub(1) & self.mask
    }
}

pub(super) struct ExtFrameRingF32<'a> {
    pub(super) buf: &'a mut [f32],
    pub(super) mask: usize,
}

impl TankRing for ExtFrameRingF32<'_> {
    #[inline(always)]
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4 {
        let m = self.mask;
        let arr = [
            self.buf[((base & m) << 2) + line],
            self.buf[((base.wrapping_add(1) & m) << 2) + line],
            self.buf[((base.wrapping_add(2) & m) << 2) + line],
            self.buf[((base.wrapping_add(3) & m) << 2) + line],
        ];
        B::Vec4::load(&arr)
    }

    #[inline(always)]
    fn read_frame_f32(&self, frame: usize, line: usize) -> f32 {
        self.buf[((frame & self.mask) << 2) + line]
    }

    #[inline(always)]
    fn write_frame<B: MathBackend>(&mut self, wr: usize, val: B::Vec4) -> usize {
        let w = wr & self.mask;
        let base = w << 2;
        if base + 3 < self.buf.len() {
            let arr = val.to_array();
            self.buf[base..base + 4].copy_from_slice(&arr);
        }
        w.wrapping_sub(1) & self.mask
    }
}

/// Branchless SIMD 4-tap convex-blended Catmull-Rom read across any [`TankRing`].
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Hot SIMD Catmull-Rom reader takes pre-hoisted constants across subblock iterations"
)]
pub(super) fn read_modulated_catmull_rom<B: MathBackend, R: TankRing>(
    ring: &R,
    wr: usize,
    base_delays_v: B::Vec4,
    mod_l: f32,
    mod_r: f32,
    delay_mod_samples: f32,
    min_tau_v: B::Vec4,
    max_tau_v: B::Vec4,
    half_alpha_v: B::Vec4,
) -> B::Vec4 {
    let mod_v = B::Vec4::from_array([
        mod_l + delay_mod_samples,
        mod_r + delay_mod_samples,
        -mod_l + delay_mod_samples,
        -mod_r + delay_mod_samples,
    ]);
    let tau_v = base_delays_v.add(mod_v).clamp(min_tau_v, max_tau_v);
    let [tau0, tau1, tau2, tau3] = tau_v.to_array();
    let i0 = tau0 as usize;
    let i1 = tau1 as usize;
    let i2 = tau2 as usize;
    let i3 = tau3 as usize;

    let int_v = B::Vec4::from_array([i0 as f32, i1 as f32, i2 as f32, i3 as f32]);
    let mu_v = tau_v.sub(int_v);

    let wr_m1 = wr.wrapping_sub(1);
    let v0 = ring.load_quad::<B>(0, wr_m1.wrapping_add(i0));
    let v1 = ring.load_quad::<B>(1, wr_m1.wrapping_add(i1));
    let v2 = ring.load_quad::<B>(2, wr_m1.wrapping_add(i2));
    let v3 = ring.load_quad::<B>(3, wr_m1.wrapping_add(i3));
    let [s_m1, s_0, s_1, s_2] = B::transpose_4x4(v0, v1, v2, v3);

    let one = B::Vec4::splat(1.0);
    let om = one.sub(mu_v);
    let lin = om.mul(s_0).fma(mu_v, s_1);
    let curv = om
        .mul(s_0.sub(s_m1).add(s_0.sub(s_1)))
        .fma(mu_v, s_1.sub(s_0).add(s_1.sub(s_2)));
    let w = mu_v.mul(om).mul(half_alpha_v);
    lin.fma(w, curv)
}

#[inline(always)]
fn read_frozen_integer_lag<B: MathBackend, R: TankRing>(
    ring: &R,
    wr: usize,
    delays: [usize; V3_TANK_LINES],
    freeze_off: &mut [u32; V3_TANK_LINES],
) -> B::Vec4 {
    let mut raw = [0.0_f32; V3_TANK_LINES];
    for l in 0..V3_TANK_LINES {
        let d = delays[l].max(1);
        let off = freeze_off[l] as usize;
        let lag = if off < d { d - off } else { d };
        raw[l] = ring.read_frame_f32(wr.wrapping_add(lag), l);
        let next = off + 1;
        freeze_off[l] = if next >= d { 0 } else { next as u32 };
    }
    B::Vec4::load(&raw)
}

#[derive(Clone, Copy)]
pub(crate) struct TankStepCtx {
    pub(crate) params: Stage4QuadParams,
    pub(crate) inv_sigma: f32,
    pub(crate) cos_psi: f32,
    pub(crate) sin_psi: f32,
    pub(crate) cos_st: f32,
    pub(crate) sin_st: f32,
    pub(crate) half_alpha_cubic: f32,
    pub(crate) delay_mod_samples: f32,
    pub(crate) lfo_inc: u32,
    pub(crate) max_tau: f32,
    pub(crate) freeze: bool,
    pub(crate) wet_dc_pole: f32,
    pub(crate) wet_dc_alpha: f32,
}

pub(crate) struct TankMutableRefs<'a> {
    pub(crate) write_ptr: &'a mut u32,
    pub(crate) freeze_off: &'a mut [u32; V3_TANK_LINES],
    pub(crate) dc_x_prev: &'a mut [f32; V3_TANK_LINES],
    pub(crate) dc_y_prev: &'a mut [f32; V3_TANK_LINES],
    pub(crate) wet_dc_x: &'a mut [f32; 2],
    pub(crate) wet_dc_y: &'a mut [f32; 2],
    pub(crate) lfo_phase_q32: &'a mut u32,
    pub(crate) vca_env: &'a mut f32,
    pub(crate) schroeder: Option<&'a mut PartitionedSchroederBank>,
}

pub(crate) struct SubblockBuffers<'a> {
    pub(crate) in_l: &'a [f32],
    pub(crate) in_r: &'a [f32],
    pub(crate) out_l: &'a mut [f32],
    pub(crate) out_r: &'a mut [f32],
    pub(crate) len: usize,
    pub(crate) accumulate_output: bool,
}

/// Evaluates Stage IV quiet-feedback knee, cubic loop saturator, unitary `U(psi, v) = H_4(1) R(psi)`
/// circulating feedback, and normalized `0.5 * (1 + r_dc)` loop DC blocker (`0` FDIV/FSQRT).
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Hoisted SIMD constants shared between stereo and subblock inner loops"
)]
fn step_loop_feedback<B: MathBackend>(
    read_v: B::Vec4,
    dc_x_p: B::Vec4,
    dc_y_p: B::Vec4,
    inv_sigma_v: B::Vec4,
    g_loop_v: B::Vec4,
    cos_psi_v: B::Vec4,
    sin_psi_v: B::Vec4,
    r_dc_v: B::Vec4,
    alpha_dc_v: B::Vec4,
    inj_g: f32,
    in_l: f32,
    in_r: f32,
) -> (B::Vec4, B::Vec4) {
    const V_CEIL: f32 = f32::from_bits(STAGE4_MAX_V_CEIL_BITS);
    const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;
    let one = B::Vec4::splat(1.0);
    let rv = read_v.mul(inv_sigma_v);
    let inv_dv = one.fma(rv, rv).rcp_ge1_nr2();
    let knee_u = read_v
        .mul(g_loop_v)
        .mul(B::Vec4::splat(0.85).fma(B::Vec4::splat(0.15), inv_dv));
    let v = knee_u
        .mul(B::Vec4::splat(TWO_OVER_3V))
        .max(B::Vec4::splat(-1.0))
        .min(one);
    let v_ceil_v = B::Vec4::splat(V_CEIL);
    let half_sq = B::Vec4::splat(0.5).mul(v).mul(v);
    let loop_v = v_ceil_v
        .mul(v)
        .mul(B::Vec4::splat(1.5).sub(half_sq))
        .max(B::Vec4::splat(-V_CEIL))
        .min(v_ceil_v);

    // U(psi, v) = H_4(1) * R(psi): pairwise 2x2 Givens before 4x4 Householder reflection.
    let rot_v = loop_v.mul(cos_psi_v).fma(loop_v.pair_swap_neg(), sin_psi_v);
    let h_offset = 0.5 * rot_v.horizontal_sum();
    let fb_v = rot_v
        .sub(B::Vec4::splat(h_offset))
        .add(B::Vec4::splat_pair(inj_g * in_l, inj_g * in_r));

    let fp16_lo = B::Vec4::splat(-FP16_MAX_FINITE_F32);
    let fp16_hi = B::Vec4::splat(FP16_MAX_FINITE_F32);
    let dc_y = flush_recursive_state_vec4(fb_v.sub(dc_x_p).mul(alpha_dc_v).fma(r_dc_v, dc_y_p))
        .max(fp16_lo)
        .min(fp16_hi);
    (fb_v, dc_y)
}

/// Applies Givens readout rotation `R(psi)`, non-recirculating `5 Hz` wet DC blocker, and
/// stereo-linked ducking VCA.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Hot readout + DC blocker + VCA inline path avoiding extra indirection"
)]
fn step_wet_dc_and_vca(
    raw_l: f32,
    raw_r: f32,
    cos_psi: f32,
    sin_psi: f32,
    wet_dc_x: &mut [f32; 2],
    wet_dc_y: &mut [f32; 2],
    wet_dc_pole: f32,
    wet_dc_alpha: f32,
    vca: &mut BallisticDuckingVca,
    in_l: f32,
    in_r: f32,
) -> (f32, f32) {
    let wet_l_rot = cos_psi * raw_l - sin_psi * raw_r;
    let wet_r_rot = sin_psi * raw_l + cos_psi * raw_r;
    let wet_l = flush_recursive_state_f32(
        wet_dc_alpha * (wet_l_rot - wet_dc_x[0]) + wet_dc_pole * wet_dc_y[0],
    );
    let wet_r = flush_recursive_state_f32(
        wet_dc_alpha * (wet_r_rot - wet_dc_x[1]) + wet_dc_pole * wet_dc_y[1],
    );
    wet_dc_x[0] = wet_l_rot;
    wet_dc_y[0] = wet_l;
    wet_dc_x[1] = wet_r_rot;
    wet_dc_y[1] = wet_r;
    vca.step_stereo(in_l, in_r, wet_l, wet_r)
}

#[inline(always)]
pub(crate) fn step_stereo_ring<B: MathBackend, R: TankRing>(
    ring: &mut R,
    ctx: TankStepCtx,
    st: &mut TankMutableRefs<'_>,
    in_l: f32,
    in_r: f32,
) -> [f32; 2] {
    let wr = *st.write_ptr as usize;
    let raw_read_v = if ctx.freeze {
        read_frozen_integer_lag::<B, R>(ring, wr, ctx.params.delay_lengths, st.freeze_off)
    } else {
        let (mod_l, mod_r) = lfo_offsets_from_phase(
            *st.lfo_phase_q32,
            ctx.params.lfo_depth_samples,
            ctx.cos_st,
            ctx.sin_st,
        );
        *st.lfo_phase_q32 = st.lfo_phase_q32.wrapping_add(ctx.lfo_inc);
        let base_delays_v = B::Vec4::from_array(ctx.params.delay_lengths.map(|x| x as f32));
        read_modulated_catmull_rom::<B, R>(
            ring,
            wr,
            base_delays_v,
            mod_l,
            mod_r,
            ctx.delay_mod_samples,
            B::Vec4::splat(4.0),
            B::Vec4::splat(ctx.max_tau),
            B::Vec4::splat(ctx.half_alpha_cubic),
        )
    };
    let read_v = if let Some(ref mut ap) = st.schroeder {
        if ctx.freeze {
            ap.step_quad::<B, true>(raw_read_v)
        } else {
            ap.step_quad::<B, false>(raw_read_v)
        }
    } else {
        raw_read_v
    };
    let weighted_f32 = read_v.mul(B::Vec4::load(&ctx.params.tap_gains)).to_array();

    if !ctx.freeze {
        let (fb_v, dc_y) = step_loop_feedback::<B>(
            read_v,
            B::Vec4::load(st.dc_x_prev),
            B::Vec4::load(st.dc_y_prev),
            B::Vec4::splat(ctx.inv_sigma),
            B::Vec4::splat(ctx.params.g_loop),
            B::Vec4::splat(ctx.cos_psi),
            B::Vec4::splat(ctx.sin_psi),
            B::Vec4::splat(ctx.params.r_dc),
            B::Vec4::splat(0.5 * (1.0 + ctx.params.r_dc)),
            ctx.params.inject_gain,
            in_l,
            in_r,
        );
        fb_v.store(st.dc_x_prev);
        dc_y.store(st.dc_y_prev);
        *st.write_ptr = ring.write_frame::<B>(wr, dc_y) as u32;
    }

    let raw_l = weighted_f32[0] + weighted_f32[2];
    let raw_r = weighted_f32[1] + weighted_f32[3];
    let mut vca = BallisticDuckingVca {
        env: *st.vca_env,
        alpha_att: ctx.params.vca_att,
        alpha_rel: ctx.params.vca_rel,
        threshold: ctx.params.vca_thresh,
        kappa: ctx.params.vca_kappa,
        min_gain: ctx.params.vca_min_gain,
    };
    let (duck_l, duck_r) = step_wet_dc_and_vca(
        raw_l,
        raw_r,
        ctx.cos_psi,
        ctx.sin_psi,
        st.wet_dc_x,
        st.wet_dc_y,
        ctx.wet_dc_pole,
        ctx.wet_dc_alpha,
        &mut vca,
        in_l,
        in_r,
    );
    *st.vca_env = vca.env;
    [
        ctx.params.dry_mix * in_l + ctx.params.wet_mix * duck_l,
        ctx.params.dry_mix * in_r + ctx.params.wet_mix * duck_r,
    ]
}

#[inline(always)]
pub(crate) fn step_subblock_ring<
    B: MathBackend,
    const MONO: bool,
    const FREEZE: bool,
    R: TankRing,
>(
    ring: &mut R,
    ctx: TankStepCtx,
    st: &mut TankMutableRefs<'_>,
    io: SubblockBuffers<'_>,
) {
    let mut wr = *st.write_ptr as usize;
    let base_delays_v = B::Vec4::from_array(ctx.params.delay_lengths.map(|x| x as f32));
    let tap_gains_v = B::Vec4::load(&ctx.params.tap_gains);
    let mut dc_x_p = B::Vec4::load(st.dc_x_prev);
    let mut dc_y_p = B::Vec4::load(st.dc_y_prev);
    let mut vca = BallisticDuckingVca {
        env: *st.vca_env,
        alpha_att: ctx.params.vca_att,
        alpha_rel: ctx.params.vca_rel,
        threshold: ctx.params.vca_thresh,
        kappa: ctx.params.vca_kappa,
        min_gain: ctx.params.vca_min_gain,
    };

    let inj_g = ctx.params.inject_gain;
    let cos_psi = ctx.cos_psi;
    let sin_psi = ctx.sin_psi;
    let cos_psi_v = B::Vec4::splat(cos_psi);
    let sin_psi_v = B::Vec4::splat(sin_psi);
    let dry_mix = ctx.params.dry_mix;
    let wet_mix = ctx.params.wet_mix;

    let inv_sigma_v = B::Vec4::splat(ctx.inv_sigma);
    let g_loop_v = B::Vec4::splat(ctx.params.g_loop);
    let r_dc_v = B::Vec4::splat(ctx.params.r_dc);
    let alpha_dc_v = B::Vec4::splat(0.5 * (1.0 + ctx.params.r_dc));
    let lfo_depth = ctx.params.lfo_depth_samples;
    let min_tau_v = B::Vec4::splat(4.0);
    let max_tau_v = B::Vec4::splat(ctx.max_tau);
    let cos_st = ctx.cos_st;
    let sin_st = ctx.sin_st;
    let mut lfo_p = *st.lfo_phase_q32;
    let lfo_inc = if FREEZE { 0 } else { ctx.lfo_inc };
    let half_alpha_v = B::Vec4::splat(ctx.half_alpha_cubic);

    let n = io
        .len
        .min(16)
        .min(io.in_l.len())
        .min(io.in_r.len())
        .min(io.out_l.len())
        .min(io.out_r.len());

    for i in 0..n {
        let in_l = io.in_l[i];
        let in_r = io.in_r[i];

        let raw_read_v = if FREEZE {
            read_frozen_integer_lag::<B, R>(ring, wr, ctx.params.delay_lengths, st.freeze_off)
        } else {
            let (mod_l, mod_r) = lfo_offsets_from_phase(lfo_p, lfo_depth, cos_st, sin_st);
            lfo_p = lfo_p.wrapping_add(lfo_inc);
            read_modulated_catmull_rom::<B, R>(
                ring,
                wr,
                base_delays_v,
                mod_l,
                mod_r,
                ctx.delay_mod_samples,
                min_tau_v,
                max_tau_v,
                half_alpha_v,
            )
        };
        let read_v = if let Some(ref mut ap) = st.schroeder {
            ap.step_quad::<B, FREEZE>(raw_read_v)
        } else {
            raw_read_v
        };
        let weighted_f32 = read_v.mul(tap_gains_v).to_array();

        if !FREEZE {
            let (fb_v, dc_y) = step_loop_feedback::<B>(
                read_v,
                dc_x_p,
                dc_y_p,
                inv_sigma_v,
                g_loop_v,
                cos_psi_v,
                sin_psi_v,
                r_dc_v,
                alpha_dc_v,
                inj_g,
                in_l,
                in_r,
            );
            dc_x_p = fb_v;
            dc_y_p = dc_y;
            wr = ring.write_frame::<B>(wr, dc_y);
        }

        let raw_l = weighted_f32[0] + weighted_f32[2];
        let raw_r = weighted_f32[1] + weighted_f32[3];
        let (duck_l, duck_r) = step_wet_dc_and_vca(
            raw_l,
            raw_r,
            cos_psi,
            sin_psi,
            st.wet_dc_x,
            st.wet_dc_y,
            ctx.wet_dc_pole,
            ctx.wet_dc_alpha,
            &mut vca,
            in_l,
            in_r,
        );
        let full_l = dry_mix * in_l + wet_mix * duck_l;
        let full_r = dry_mix * in_r + wet_mix * duck_r;
        let (yl, yr) = if MONO {
            let mid = 0.5 * (full_l + full_r);
            (mid, mid)
        } else {
            (full_l, full_r)
        };
        if io.accumulate_output {
            io.out_l[i] += yl;
            io.out_r[i] += yr;
        } else {
            io.out_l[i] = yl;
            io.out_r[i] = yr;
        }
    }

    if !FREEZE {
        dc_x_p.store(st.dc_x_prev);
        dc_y_p.store(st.dc_y_prev);
        *st.lfo_phase_q32 = lfo_p;
        *st.write_ptr = wr as u32;
    }
    *st.vca_env = vca.env;
}

#[cfg(test)]
mod tests {
    use super::super::V3_TANK_CAP;
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};

    fn write_test_frame_lag(
        frames: &mut [[u16; V3_TANK_LINES]; V3_TANK_CAP],
        wr: usize,
        lag: usize,
        vals: [u16; V3_TANK_LINES],
    ) {
        frames[(wr + lag) & V3_TANK_MASK] = vals;
    }

    #[test]
    fn test_q32_smooth_sin_exact_quadrature_and_catmull_rom_identity() {
        use crate::math::fp16::f32_to_fp16;
        assert_eq!(q32_smooth_sin(0).to_bits(), 0);
        assert_eq!(q32_smooth_sin(0x8000_0000).to_bits(), 0);
        assert!((q32_smooth_sin(0x4000_0000) - 1.0).abs() < 1e-6);
        assert!((q32_smooth_sin(0xC000_0000) + 1.0).abs() < 1e-6);
        for k in 0..64 {
            let s = q32_smooth_sin(k * 0x0400_0000);
            assert!((-1.0..=1.0).contains(&s));
        }

        let mut frames = [[0u16; V3_TANK_LINES]; V3_TANK_CAP];
        let base_delays = [8_usize, 12, 16, 20];
        let expected = [0.25_f32, -0.5, 0.75, -0.125];
        for l in 0..V3_TANK_LINES {
            let val = f32_to_fp16(expected[l]);
            write_test_frame_lag(&mut frames, 0, base_delays[l], [val; 4]);
        }
        let ring = InlineFrameRing(&mut frames);
        let simd = read_modulated_catmull_rom::<DefaultMathBackend, _>(
            &ring,
            0,
            <DefaultMathBackend as MathBackend>::Vec4::from_array(base_delays.map(|x| x as f32)),
            0.0,
            0.0,
            0.0,
            <DefaultMathBackend as MathBackend>::Vec4::splat(4.0),
            <DefaultMathBackend as MathBackend>::Vec4::splat((V3_TANK_CAP - 3) as f32),
            <DefaultMathBackend as MathBackend>::Vec4::splat(0.5),
        )
        .to_array();
        let scal = read_modulated_catmull_rom::<ScalarBackend, _>(
            &ring,
            0,
            <ScalarBackend as MathBackend>::Vec4::from_array(base_delays.map(|x| x as f32)),
            0.0,
            0.0,
            0.0,
            <ScalarBackend as MathBackend>::Vec4::splat(4.0),
            <ScalarBackend as MathBackend>::Vec4::splat((V3_TANK_CAP - 3) as f32),
            <ScalarBackend as MathBackend>::Vec4::splat(0.5),
        )
        .to_array();
        for l in 0..V3_TANK_LINES {
            assert_eq!(simd[l].to_bits(), expected[l].to_bits());
            assert_eq!(scal[l].to_bits(), expected[l].to_bits());
        }
    }

    #[test]
    fn test_schroeder_bank_allpass_unity_energy_and_freeze_hold() {
        let mut ap =
            PartitionedSchroederBank::from_params([19, 31, 47, 0], [0.65, -0.55, 0.5, 0.9])
                .expect("non-zero delays allocate");
        assert!((ap.gains[3] - 0.75).abs() < 1e-6);
        assert!(!ap.active_mask[3]);

        let mut e_in = 0.0_f64;
        let mut e_out = 0.0_f64;
        for n in 0..2048 {
            let x = if n < 128 {
                [(n as f32 * 0.13).sin(), (n as f32 * 0.21).cos(), 0.25, 0.5]
            } else {
                [0.0; 4]
            };
            let y = ap
                .step_quad::<DefaultMathBackend, false>(
                    <DefaultMathBackend as MathBackend>::Vec4::load(&x),
                )
                .to_array();
            assert_eq!(y[3].to_bits(), x[3].to_bits());
            e_in += f64::from(x[0]) * f64::from(x[0]) + f64::from(x[1]) * f64::from(x[1]);
            e_out += f64::from(y[0]) * f64::from(y[0]) + f64::from(y[1]) * f64::from(y[1]);
        }
        assert!((e_out - e_in).abs() < 1e-3);

        let before = ap.buf.clone();
        let _ = ap.step_quad::<DefaultMathBackend, true>(
            <DefaultMathBackend as MathBackend>::Vec4::splat(0.5),
        );
        assert_eq!(&*ap.buf, &*before);
    }
}
