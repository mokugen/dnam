//! Branch-free `Q0.32` quadrature LFO and convex-blended 4-tap Catmull-Rom fractional reads
//! (`stage4_tank/v3_fdn/lfo_interp.rs`).

use super::{V3_TANK_CAP, V3_TANK_LINES, V3_TANK_MASK, V3_TANK_RING_STRIDE, V3Stage4TankParams};
use crate::math::{
    FP16_MAX_FINITE_F32, IEEE754_F32_EXP_BIAS, MathBackend, SimdVec4,
    const_hermite5_quarter_sine_coeffs, flush_recursive_state_vec4, newton6_reciprocal_f64,
};
use crate::operators::op_t::BallisticDuckingVca;
use crate::stages::topology::STAGE4_MAX_V_CEIL_BITS;

/// Smooth $C^1$-peak degree-5 odd triangle-Hermite sine on `Q0.32` phase (`0` `libm`, `0` branches,
/// exact `+0.0` at `p == 0`, peak `±1.0` at `p ∈ {0x4000_0000, 0xC000_0000}`).
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
///
/// Sets `alpha_cubic ∈ [0, 1]` such that `|g_loop| * (1 + 0.25 * alpha_cubic) <= 0.9992`
/// (`alpha_cubic == 1.0` for `|g_loop| <= 0.799`), preserving full 4-tap Catmull-Rom fidelity for
/// chorus/rotary/BBD delays and strict per-sample $\ell_\infty$ BIBO stability up to `g_loop <= 0.9990`.
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

/// Ring storage abstraction decoupling inline L1D `[u16; 1028]` rings from cold-allocated
/// power-of-two extended delay buffers (`ext_lines`) without runtime allocation or branching inside
/// the inner sample loop.
pub(super) trait TankRing {
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4;
    fn write_quad(&mut self, wr: usize, packed: [u16; V3_TANK_LINES]) -> usize;
}

pub(super) struct InlineTankRing<'a>(
    pub(super) &'a mut [[u16; V3_TANK_RING_STRIDE]; V3_TANK_LINES],
);

impl TankRing for InlineTankRing<'_> {
    #[inline(always)]
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4 {
        let b = base & V3_TANK_MASK;
        let Some((quad, _)) = self.0[line][b..b + 4].split_first_chunk::<4>() else {
            return B::Vec4::zero();
        };
        B::fp16x4_to_vec4(quad)
    }

    #[inline(always)]
    fn write_quad(&mut self, wr: usize, packed: [u16; V3_TANK_LINES]) -> usize {
        let w = wr & V3_TANK_MASK;
        for l in 0..V3_TANK_LINES {
            self.0[l][w] = packed[l];
        }
        if w < 4 {
            for l in 0..V3_TANK_LINES {
                self.0[l][w + V3_TANK_CAP] = packed[l];
            }
        }
        w.wrapping_sub(1) & V3_TANK_MASK
    }
}

pub(super) struct ExtTankRing<'a> {
    pub(super) buf: &'a mut [u16],
    pub(super) cap: usize,
    pub(super) stride: usize,
    pub(super) mask: usize,
}

impl TankRing for ExtTankRing<'_> {
    #[inline(always)]
    fn load_quad<B: MathBackend>(&self, line: usize, base: usize) -> B::Vec4 {
        let b = base & self.mask;
        let start = line * self.stride + b;
        let Some(slice) = self.buf.get(start..start + 4) else {
            return B::Vec4::zero();
        };
        let Some((quad, _)) = slice.split_first_chunk::<4>() else {
            return B::Vec4::zero();
        };
        B::fp16x4_to_vec4(quad)
    }

    #[inline(always)]
    fn write_quad(&mut self, wr: usize, packed: [u16; V3_TANK_LINES]) -> usize {
        let w = wr & self.mask;
        for l in 0..V3_TANK_LINES {
            let base = l * self.stride;
            if base + w < self.buf.len() {
                self.buf[base + w] = packed[l];
            }
            if w < 4 && base + w + self.cap < self.buf.len() {
                self.buf[base + w + self.cap] = packed[l];
            }
        }
        w.wrapping_sub(1) & self.mask
    }
}

/// Branchless SIMD 4-tap convex-blended Catmull-Rom read across any [`TankRing`].
///
/// When `mu == 0.0` (e.g. `lfo_depth_samples == 0.0`), yields bit-exact `s_0` for exact zero-ULP
/// parity with static integer-tap reads.
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
    min_tau_v: B::Vec4,
    max_tau_v: B::Vec4,
    half_alpha_v: B::Vec4,
) -> B::Vec4 {
    let mod_v = B::Vec4::from_array([mod_l, mod_r, -mod_l, -mod_r]);
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

#[derive(Clone, Copy)]
pub(super) struct TankStepCtx {
    pub(super) params: V3Stage4TankParams,
    pub(super) inv_sigma: f32,
    pub(super) cos_psi: f32,
    pub(super) sin_psi: f32,
    pub(super) cos_st: f32,
    pub(super) sin_st: f32,
    pub(super) half_alpha_cubic: f32,
    pub(super) lfo_inc: u32,
    pub(super) max_tau: f32,
    pub(super) freeze: bool,
}

pub(super) struct TankMutableRefs<'a> {
    pub(super) write_ptr: &'a mut u32,
    pub(super) dc_x_prev: &'a mut [f32; V3_TANK_LINES],
    pub(super) dc_y_prev: &'a mut [f32; V3_TANK_LINES],
    pub(super) lfo_phase_q32: &'a mut u32,
    pub(super) vca_env: &'a mut f32,
}

pub(super) struct SubblockBuffers<'a> {
    pub(super) s2_dir: &'a [[f32; 4]; 16],
    pub(super) cab_out: &'a [[f32; 2]; 16],
    pub(super) clean_sub: &'a [[f32; 2]; 16],
    pub(super) out_l: &'a mut [f32],
    pub(super) out_r: &'a mut [f32],
    pub(super) len: usize,
}

#[inline(always)]
pub(super) fn step_stereo_ring<B: MathBackend, R: TankRing>(
    ring: &mut R,
    ctx: TankStepCtx,
    st: &mut TankMutableRefs<'_>,
    in_l: f32,
    in_r: f32,
) -> [f32; 2] {
    const V_CEIL: f32 = f32::from_bits(STAGE4_MAX_V_CEIL_BITS);
    const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;

    let wr = *st.write_ptr as usize;
    let (mod_l, mod_r) = lfo_offsets_from_phase(
        *st.lfo_phase_q32,
        ctx.params.lfo_depth_samples,
        ctx.cos_st,
        ctx.sin_st,
    );
    let half_alpha_v = if ctx.freeze {
        B::Vec4::zero()
    } else {
        *st.lfo_phase_q32 = st.lfo_phase_q32.wrapping_add(ctx.lfo_inc);
        B::Vec4::splat(ctx.half_alpha_cubic)
    };
    let base_delays_v = B::Vec4::from_array(ctx.params.delay_lengths.map(|x| x as f32));
    let read_v = read_modulated_catmull_rom::<B, R>(
        ring,
        wr,
        base_delays_v,
        mod_l,
        mod_r,
        B::Vec4::splat(4.0),
        B::Vec4::splat(ctx.max_tau),
        half_alpha_v,
    );
    let weighted_f32 = read_v.mul(B::Vec4::load(&ctx.params.tap_gains)).to_array();
    let fp16_lo = B::Vec4::splat(-FP16_MAX_FINITE_F32);
    let fp16_hi = B::Vec4::splat(FP16_MAX_FINITE_F32);

    let next_wr = if ctx.freeze {
        let h_offset = 0.5 * read_v.horizontal_sum();
        let circ = flush_recursive_state_vec4(read_v.sub(B::Vec4::splat(h_offset)))
            .max(fp16_lo)
            .min(fp16_hi);
        ring.write_quad(wr, B::vec4_to_fp16x4(circ))
    } else {
        let one = B::Vec4::splat(1.0);
        let rv = read_v.mul(B::Vec4::splat(ctx.inv_sigma));
        let inv_dv = one.fma(rv, rv).rcp_ge1_nr2();
        let knee_u = read_v
            .mul(B::Vec4::splat(ctx.params.g_loop))
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

        let h_offset = 0.5 * loop_v.horizontal_sum();
        let inj_g = ctx.params.inject_gain;
        let fb_v = loop_v
            .sub(B::Vec4::splat(h_offset))
            .add(B::Vec4::splat_pair(inj_g * in_l, inj_g * in_r));

        let dc_x_p = B::Vec4::load(st.dc_x_prev);
        let dc_y_p = B::Vec4::load(st.dc_y_prev);
        let dc_y = flush_recursive_state_vec4(
            fb_v.sub(dc_x_p)
                .fma(B::Vec4::splat(ctx.params.r_dc), dc_y_p),
        )
        .max(fp16_lo)
        .min(fp16_hi);
        fb_v.store(st.dc_x_prev);
        dc_y.store(st.dc_y_prev);
        ring.write_quad(wr, B::vec4_to_fp16x4(dc_y))
    };
    *st.write_ptr = next_wr as u32;

    let raw_l = weighted_f32[0] + weighted_f32[2];
    let raw_r = weighted_f32[1] + weighted_f32[3];
    let wet_l = ctx.cos_psi * raw_l - ctx.sin_psi * raw_r;
    let wet_r = ctx.sin_psi * raw_l + ctx.cos_psi * raw_r;
    let mut vca = BallisticDuckingVca {
        env: *st.vca_env,
        alpha_att: ctx.params.vca_att,
        alpha_rel: ctx.params.vca_rel,
        threshold: ctx.params.vca_thresh,
        kappa: ctx.params.vca_kappa,
        min_gain: ctx.params.vca_min_gain,
    };
    let (duck_l, duck_r) = vca.step_stereo(in_l, in_r, wet_l, wet_r);
    *st.vca_env = vca.env;
    [
        ctx.params.dry_mix * in_l + ctx.params.wet_mix * duck_l,
        ctx.params.dry_mix * in_r + ctx.params.wet_mix * duck_r,
    ]
}

#[inline(always)]
pub(super) fn step_subblock_ring<
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
    const V_CEIL: f32 = f32::from_bits(STAGE4_MAX_V_CEIL_BITS);
    const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;

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
    let dry_mix = ctx.params.dry_mix;
    let wet_mix = ctx.params.wet_mix;

    let one = B::Vec4::splat(1.0);
    let inv_sigma_v = B::Vec4::splat(ctx.inv_sigma);
    let g_loop_v = B::Vec4::splat(ctx.params.g_loop);
    let c085 = B::Vec4::splat(0.85);
    let c015 = B::Vec4::splat(0.15);
    let two_over_3v = B::Vec4::splat(TWO_OVER_3V);
    let neg_one = B::Vec4::splat(-1.0);
    let v_ceil_v = B::Vec4::splat(V_CEIL);
    let neg_v_ceil_v = B::Vec4::splat(-V_CEIL);
    let half = B::Vec4::splat(0.5);
    let c15 = B::Vec4::splat(1.5);
    let r_dc_v = B::Vec4::splat(ctx.params.r_dc);
    let fp16_lo = B::Vec4::splat(-FP16_MAX_FINITE_F32);
    let fp16_hi = B::Vec4::splat(FP16_MAX_FINITE_F32);
    let lfo_depth = ctx.params.lfo_depth_samples;
    let min_tau_v = B::Vec4::splat(4.0);
    let max_tau_v = B::Vec4::splat(ctx.max_tau);
    let cos_st = ctx.cos_st;
    let sin_st = ctx.sin_st;
    let mut lfo_p = *st.lfo_phase_q32;
    let lfo_inc = if FREEZE { 0 } else { ctx.lfo_inc };
    let half_alpha_v = if FREEZE {
        B::Vec4::zero()
    } else {
        B::Vec4::splat(ctx.half_alpha_cubic)
    };
    let n = io.len.min(16).min(io.out_l.len()).min(io.out_r.len());
    let out_l = &mut io.out_l[..n];
    let out_r = &mut io.out_r[..n];

    for i in 0..n {
        let in_l = io.s2_dir[i][2] + io.cab_out[i][0] + io.clean_sub[i][0];
        let in_r = if MONO {
            in_l
        } else {
            io.s2_dir[i][3] + io.cab_out[i][1] + io.clean_sub[i][1]
        };

        let (mod_l, mod_r) = lfo_offsets_from_phase(lfo_p, lfo_depth, cos_st, sin_st);
        lfo_p = lfo_p.wrapping_add(lfo_inc);
        let read_v = read_modulated_catmull_rom::<B, R>(
            ring,
            wr,
            base_delays_v,
            mod_l,
            mod_r,
            min_tau_v,
            max_tau_v,
            half_alpha_v,
        );
        let weighted_f32 = read_v.mul(tap_gains_v).to_array();

        if FREEZE {
            let h_offset = 0.5 * read_v.horizontal_sum();
            let circ = flush_recursive_state_vec4(read_v.sub(B::Vec4::splat(h_offset)))
                .max(fp16_lo)
                .min(fp16_hi);
            wr = ring.write_quad(wr, B::vec4_to_fp16x4(circ));
        } else {
            let rv = read_v.mul(inv_sigma_v);
            let inv_dv = one.fma(rv, rv).rcp_ge1_nr2();
            let knee_u = read_v.mul(g_loop_v).mul(c085.fma(c015, inv_dv));
            let v = knee_u.mul(two_over_3v).max(neg_one).min(one);
            let half_sq = half.mul(v).mul(v);
            let loop_v = v_ceil_v
                .mul(v)
                .mul(c15.sub(half_sq))
                .max(neg_v_ceil_v)
                .min(v_ceil_v);

            let h_offset = 0.5 * loop_v.horizontal_sum();
            let fb_v = loop_v
                .sub(B::Vec4::splat(h_offset))
                .add(B::Vec4::splat_pair(inj_g * in_l, inj_g * in_r));

            let dc_y = flush_recursive_state_vec4(fb_v.sub(dc_x_p).fma(r_dc_v, dc_y_p))
                .max(fp16_lo)
                .min(fp16_hi);
            dc_x_p = fb_v;
            dc_y_p = dc_y;
            wr = ring.write_quad(wr, B::vec4_to_fp16x4(dc_y));
        }

        let raw_l = weighted_f32[0] + weighted_f32[2];
        let raw_r = weighted_f32[1] + weighted_f32[3];
        let wet_l = cos_psi * raw_l - sin_psi * raw_r;
        let wet_r = sin_psi * raw_l + cos_psi * raw_r;
        let (duck_l, duck_r) = vca.step_stereo(in_l, in_r, wet_l, wet_r);
        let yl = dry_mix * in_l + wet_mix * duck_l;
        out_l[i] = yl;
        out_r[i] = if MONO {
            yl
        } else {
            dry_mix * in_r + wet_mix * duck_r
        };
    }

    if !FREEZE {
        dc_x_p.store(st.dc_x_prev);
        dc_y_p.store(st.dc_y_prev);
        *st.lfo_phase_q32 = lfo_p;
    }
    *st.write_ptr = wr as u32;
    *st.vca_env = vca.env;
}

#[cfg(test)]
mod tests {
    use super::super::V3_TANK_CAP;
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend, fp16::f32_to_fp16};

    fn write_test_lag(
        lines: &mut [[u16; V3_TANK_RING_STRIDE]; V3_TANK_LINES],
        wr: usize,
        lag: usize,
        vals: [u16; V3_TANK_LINES],
    ) {
        let idx = (wr + lag) & V3_TANK_MASK;
        for l in 0..V3_TANK_LINES {
            lines[l][idx] = vals[l];
            if idx < 4 {
                lines[l][idx + V3_TANK_CAP] = vals[l];
            }
        }
    }

    #[test]
    fn test_q32_smooth_sin_exact_quadrature_and_catmull_rom_identity() {
        assert_eq!(q32_smooth_sin(0).to_bits(), 0);
        assert_eq!(q32_smooth_sin(0x8000_0000).to_bits(), 0);
        assert!((q32_smooth_sin(0x4000_0000) - 1.0).abs() < 1e-6);
        assert!((q32_smooth_sin(0xC000_0000) + 1.0).abs() < 1e-6);
        for k in 0..64 {
            let p = k * 0x0400_0000;
            let s = q32_smooth_sin(p);
            assert!((-1.0..=1.0).contains(&s));
        }

        // Verify half_alpha_cubic contractivity across g_loop range
        let (inc_zero, ha_chorus) = cold_lfo_inc_and_half_alpha(0.0, 0.0, 0.72, 48_000.0);
        assert_eq!(inc_zero, 0);
        assert!((ha_chorus - 0.5).abs() < 1e-6);
        let (inc_shimmer, ha_shimmer) = cold_lfo_inc_and_half_alpha(1.2, 8.0, 0.99, 48_000.0);
        assert!(inc_shimmer > 0);
        assert!(0.99 * (1.0 + 0.5 * ha_shimmer) <= 0.9995);

        // Verify midpoint Catmull-Rom weights [-1/16, 9/16, 9/16, -1/16] on mirrored ring wrap boundary
        let mut lines = [[0u16; V3_TANK_RING_STRIDE]; V3_TANK_LINES];
        let wr = 1020_usize;
        write_test_lag(&mut lines, wr, 10, [f32_to_fp16(1.0); 4]);
        write_test_lag(&mut lines, wr, 11, [f32_to_fp16(1.0); 4]);
        let base_delays_simd = <DefaultMathBackend as MathBackend>::Vec4::splat(10.0);
        let base_delays_scal = <ScalarBackend as MathBackend>::Vec4::splat(10.0);
        let ring_mid = InlineTankRing(&mut lines);
        let v_mid = read_modulated_catmull_rom::<DefaultMathBackend, _>(
            &ring_mid,
            wr,
            base_delays_simd,
            0.5,
            0.5,
            <DefaultMathBackend as MathBackend>::Vec4::splat(4.0),
            <DefaultMathBackend as MathBackend>::Vec4::splat((V3_TANK_CAP - 3) as f32),
            <DefaultMathBackend as MathBackend>::Vec4::splat(0.5),
        );
        let v_scal = read_modulated_catmull_rom::<ScalarBackend, _>(
            &ring_mid,
            wr,
            base_delays_scal,
            0.5,
            0.5,
            <ScalarBackend as MathBackend>::Vec4::splat(4.0),
            <ScalarBackend as MathBackend>::Vec4::splat((V3_TANK_CAP - 3) as f32),
            <ScalarBackend as MathBackend>::Vec4::splat(0.5),
        );
        assert!((v_mid.to_array()[0] - 1.125).abs() < 1e-4);
        assert!((v_mid.to_array()[2] - 0.5).abs() < 1e-4);
        assert_eq!(v_mid.to_array(), v_scal.to_array());

        // Exact integer endpoint identity when mod == 0.0, including cross-wrap lag 4..1023
        let mut wrap_lines = [[0u16; V3_TANK_RING_STRIDE]; V3_TANK_LINES];
        write_test_lag(&mut wrap_lines, wr, 4, [f32_to_fp16(0.75); 4]);
        write_test_lag(&mut wrap_lines, wr, 11, [f32_to_fp16(1.0); 4]);
        let wrap_ring = InlineTankRing(&mut wrap_lines);
        let v_exact = read_modulated_catmull_rom::<DefaultMathBackend, _>(
            &wrap_ring,
            wr,
            <DefaultMathBackend as MathBackend>::Vec4::from_array([4.0, 11.0, 12.0, 1023.0]),
            0.0,
            0.0,
            <DefaultMathBackend as MathBackend>::Vec4::splat(4.0),
            <DefaultMathBackend as MathBackend>::Vec4::splat((V3_TANK_CAP - 1) as f32),
            <DefaultMathBackend as MathBackend>::Vec4::splat(0.5),
        );
        assert_eq!(v_exact.to_array(), [0.75, 1.0, 0.0, 0.0]);
    }
}
