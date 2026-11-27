//! SONA v4.0 `CoreQuad` nonlinear gain dynamics: $C^\infty$ bounded pre-ADAA gain `g_pre(v_eff^2)`
//! (12 dB/oct gate and B+ sag), post-ADAA rank-1 `SE-Bus` (`A.2a`), and bounded cross-lane `GLU`
//! (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.core.interblock-tiles]`).

use crate::simd::{G_PRE_MAX, KernelVec4, StereoVec8};

pub use crate::simd::{G_PRE_E_MAX, SE_BUS_ABS_MAX};

/// Upper gain bound `[0.0, 2.5]` for $C^\infty$ `g_pre(v_eff^2)` (`ref[dsp.op-n.polyadaa.clamp]`).
pub const MAX_G_PRE_GAIN: f32 = G_PRE_MAX;

/// Magnitude clamp `|v_eff| <= 10^3` ahead of squaring so `v_eff^2 <= 10^6 == G_PRE_E_MAX` never overflows.
const G_PRE_V_ABS_MAX: f32 = 1.0e3;

/// Upper bound on non-negative modulation products (`d_pre * e` and `se_d_bus * p_bus`) preventing
/// `f32` overflow under extreme parameters.
const MAX_NONNEG_MOD_SCALE: f32 = 1.0e12;

const _: () = assert!(MAX_G_PRE_GAIN == 2.5);
const _: () = assert!(G_PRE_V_ABS_MAX * G_PRE_V_ABS_MAX == G_PRE_E_MAX);
const _: () = assert!((MAX_G_PRE_GAIN * MAX_NONNEG_MOD_SCALE).is_finite());

#[inline(always)]
pub(super) fn clamp_range_safe_vec4<V: KernelVec4>(x: V, lo: f32, hi: f32, fallback: f32) -> V {
    let lo_v = V::splat(lo);
    let hi_v = V::splat(hi);
    let fb_v = V::splat(fallback);
    V::select_le(x.abs(), V::splat(f32::MAX), x, fb_v)
        .clamp(lo_v, hi_v)
        .add(V::zero())
}

#[inline(always)]
fn clamp_nonneg_mod_vec4<V: KernelVec4>(x: V) -> V {
    let zero = V::zero();
    let max_mod = V::splat(MAX_NONNEG_MOD_SCALE);
    V::select_le(x.abs(), V::splat(f32::MAX), x, zero)
        .clamp(zero, max_mod)
        .add(zero)
}

/// Evaluates $C^\infty$ bounded pre-ADAA gain `g_pre = (g0 + g_inf * d_pre * e) * rcp_ge1_nr2(1 + d_pre * e)`
/// with `e = min(v_eff^2, G_PRE_E_MAX)` and `g0, g_inf in [0.0, 2.5]` across 4 SIMD lanes
/// (`ref[dsp.op-n.polyadaa.clamp]`).
///
/// Guards `d_pre * e <= 0.0 -> g0` for `0-ULP` Zero-DC parity, realizes a 12 dB/oct noise gate
/// when `g0 < g_inf` or B+ sag compression when `g0 > g_inf`, and preserves `0-ULP` Mechanism A2
/// invariance.
#[inline(always)]
#[must_use]
pub fn eval_g_pre_vec4<V: KernelVec4>(v_eff: V, g0: V, g_inf: V, d_pre: V) -> V {
    let zero = V::zero();
    let one = V::splat(1.0);
    let g0_c = clamp_range_safe_vec4(g0, 0.0, MAX_G_PRE_GAIN, 0.0);
    let g_inf_c = clamp_range_safe_vec4(g_inf, 0.0, MAX_G_PRE_GAIN, 0.0);
    let d_pre_c = clamp_nonneg_mod_vec4(d_pre);

    let v_abs = v_eff
        .flush_recursive_state()
        .abs()
        .min(V::splat(G_PRE_V_ABS_MAX));
    let e = v_abs.mul(v_abs).min(V::splat(G_PRE_E_MAX));
    let de = d_pre_c.mul(e).min(V::splat(MAX_NONNEG_MOD_SCALE));

    let lo = g0_c.min(g_inf_c);
    let hi = g0_c.max(g_inf_c);
    let num = g0_c.fma(g_inf_c, de);
    let inv_den = one.add(de).rcp_ge1_nr2();
    let raw = num.mul(inv_den).clamp(lo, hi);
    V::select_le(de, zero, g0_c, raw).add(zero)
}

/// Evaluates [`eval_g_pre_vec4`] across an 8-lane `StereoVec8` register (`0-ULP` parity).
#[inline(always)]
#[must_use]
pub fn eval_g_pre_stereo8<S: StereoVec8>(
    v_eff: S,
    g0: S::Half,
    g_inf: S::Half,
    d_pre: S::Half,
) -> S {
    let (v_l, v_r) = v_eff.into_lr();
    S::from_lr(
        eval_g_pre_vec4(v_l, g0, g_inf, d_pre),
        eval_g_pre_vec4(v_r, g0, g_inf, d_pre),
    )
}

/// Evaluates post-ADAA rank-1 `SE-Bus` (`A.2a`) gain `g_se = rcp_ge1_nr2(1 + d_bus * p_bus)`
/// where `p_bus = sum_{j=0..3} w_sq[j] * min(|v_eff[j]|, SE_BUS_ABS_MAX)` across 4 SIMD lanes
/// (`ref[dsp.op-n.core.interblock-tiles]`).
///
/// Guarantees `g_se in (0.0, 1.0]`, exact `1.0` when `d_bus * p_bus <= 0.0`, and `0-ULP` Mechanism A2
/// invariance.
#[inline(always)]
#[must_use]
pub fn eval_se_bus_gain_vec4<V: KernelVec4>(v_eff: V, se_w_sq: V, se_d_bus: V) -> V {
    let zero = V::zero();
    let one = V::splat(1.0);
    let w_sq_c = clamp_nonneg_mod_vec4(se_w_sq);
    let d_bus_c = clamp_nonneg_mod_vec4(se_d_bus);

    let v_abs = v_eff
        .flush_recursive_state()
        .abs()
        .min(V::splat(SE_BUS_ABS_MAX));
    let p_bus = w_sq_c.dot(v_abs);
    let bus_term = d_bus_c
        .mul(V::splat(p_bus))
        .min(V::splat(MAX_NONNEG_MOD_SCALE));
    let raw = one.add(bus_term).rcp_ge1_nr2().clamp(zero, one);
    V::select_le(bus_term, zero, one, raw).add(zero)
}

/// Evaluates [`eval_se_bus_gain_vec4`] independently on Left and Right channels of a `StereoVec8`
/// register (`0-ULP` parity and `0-ULP` Mechanism A2 invariance).
#[inline(always)]
#[must_use]
pub fn eval_se_bus_gain_stereo8<S: StereoVec8>(v_eff: S, se_w_sq: S::Half, se_d_bus: S::Half) -> S {
    let zero = S::zero();
    let one = S::splat(1.0);
    let w_sq_c = clamp_nonneg_mod_vec4(se_w_sq);
    let d_bus_c = clamp_nonneg_mod_vec4(se_d_bus);

    let v_abs = v_eff
        .flush_recursive_state()
        .abs()
        .min(S::splat(SE_BUS_ABS_MAX));
    let (p_bus_l, p_bus_r) = S::broadcast_half(w_sq_c).mul(v_abs).horizontal_sum_lr();
    let bus_term = S::broadcast_half(d_bus_c)
        .mul(S::splat_lr(p_bus_l, p_bus_r))
        .min(S::splat(MAX_NONNEG_MOD_SCALE));
    let raw = one.add(bus_term).rcp_ge1_nr2().clamp(zero, one);
    S::select_le(bus_term, zero, one, raw).add(zero)
}

/// Evaluates bounded cross-lane `GLU` output `e_tilde` strictly on `bounded_odd = g_se * a_odd_wf`:
/// `e_tilde = out_gain * (1 - lin_mix) * (g_se * a_even + glu_mix * bounded_odd * rotl1(bounded_odd))`
/// (`|e_tilde| <= 20.0`, canonical `+0.0` at zero equilibrium, `ref[dsp.op-n.core.interblock-tiles]`).
#[inline(always)]
#[must_use]
pub fn eval_bounded_glu_stereo8<S: StereoVec8>(
    a_odd_wf: S,
    a_even: S,
    g_se: S,
    lin_mix: S::Half,
    glu_mix: S::Half,
    out_gain: S::Half,
) -> S {
    let lin_c = clamp_range_safe_vec4(lin_mix, 0.0, 1.0, 0.0);
    let glu_c = clamp_range_safe_vec4(glu_mix, -4.0, 4.0, 0.0);
    let out_c = clamp_range_safe_vec4(out_gain, -4.0, 4.0, 0.0);

    let bounded_odd = g_se.mul(a_odd_wf);
    let glu_term = bounded_odd.mul(bounded_odd.rotate_left_1());
    let inner = g_se.mul(a_even).fma(S::broadcast_half(glu_c), glu_term);
    let one_minus_lin = S::splat(1.0).sub(S::broadcast_half(lin_c));
    S::broadcast_half(out_c)
        .mul(one_minus_lin)
        .mul(inner)
        .flush_recursive_state()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.core.interblock-tiles]
    /// verify[dsp.op-n.core.stereo-simd]
    #[test]
    fn test_g_pre_gate_sag_se_bus_and_bounded_glu() {
        // 1. 12 dB/oct noise gate: g0 < g_inf rises monotonically with |v_eff|.
        let g0_gate = GpuVec4::splat(0.1);
        let g_inf_gate = GpuVec4::splat(2.0);
        let d_pre = GpuVec4::splat(4.0);
        let v_levels = GpuVec4::from_array([0.0, 0.25, 1.0, 10.0]);
        let g_gate = eval_g_pre_vec4(v_levels, g0_gate, g_inf_gate, d_pre).to_array();
        assert_eq!(g_gate[0], 0.1);
        assert!(g_gate[0] < g_gate[1] && g_gate[1] < g_gate[2] && g_gate[2] < g_gate[3]);
        assert!(g_gate[3] <= MAX_G_PRE_GAIN && g_gate[3] > 1.98);

        // 2. B+ sag drive: g0 > g_inf compresses smoothly downward with |v_eff|.
        let g0_sag = GpuVec4::splat(2.5);
        let g_inf_sag = GpuVec4::splat(0.5);
        let g_sag = eval_g_pre_vec4(v_levels, g0_sag, g_inf_sag, d_pre).to_array();
        assert_eq!(g_sag[0], 2.5);
        assert!(g_sag[0] > g_sag[1] && g_sag[1] > g_sag[2] && g_sag[2] > g_sag[3]);
        assert!(g_sag[3] >= 0.5 && g_sag[3] < 0.52);

        // 3. SE-Bus A.2a: g_se == 1.0 at zero, strictly in (0, 1] under signal, 0-ULP vec4/stereo8 parity.
        let w_sq = GpuVec4::from_array([1.0, 0.5, 0.25, 0.125]);
        let d_bus = GpuVec4::from_array([0.0, 0.5, 1.0, 4.0]);
        let se_zero = eval_se_bus_gain_vec4(GpuVec4::splat(-0.0), w_sq, d_bus).to_array();
        assert_eq!(se_zero, [1.0; 4]);

        let v_lr = InterleavedStereo8::from_lr(
            GpuVec4::from_array([0.5, -1.0, 2.0, -4.0]),
            GpuVec4::from_array([-1.5, 0.25, -0.5, 3.0]),
        );
        let se8 = eval_se_bus_gain_stereo8(v_lr, w_sq, d_bus);
        let se_l = eval_se_bus_gain_vec4(v_lr.left, w_sq, d_bus);
        let se_r = eval_se_bus_gain_vec4(v_lr.right, w_sq, d_bus);
        assert_eq!(
            se8.into_lr().0.to_array().map(f32::to_bits),
            se_l.to_array().map(f32::to_bits)
        );
        assert_eq!(
            se8.into_lr().1.to_array().map(f32::to_bits),
            se_r.to_array().map(f32::to_bits)
        );
        assert!(se_l.to_array()[0] == 1.0);
        assert!(se_l.to_array()[1] > se_l.to_array()[2] && se_l.to_array()[2] > se_l.to_array()[3]);
        assert!(se_l.to_array()[3] > 0.0);

        // 4. ±1e9 / NaN / ±Inf boundedness on g_pre, SE-Bus, and GLU (|e_tilde| <= 20.0).
        let ext_v = InterleavedStereo8::from_lr(
            GpuVec4::from_array([-1.0e9, 1.0e9, f32::NAN, f32::INFINITY]),
            GpuVec4::from_array([1.0e9, -1.0e9, f32::NEG_INFINITY, -0.0]),
        );
        let g_ext = eval_g_pre_stereo8(
            ext_v,
            GpuVec4::splat(3.0),
            GpuVec4::splat(f32::NAN),
            GpuVec4::splat(1.0e20),
        );
        let se_ext =
            eval_se_bus_gain_stereo8(ext_v, GpuVec4::splat(1.0e20), GpuVec4::splat(1.0e20));
        for lane in 0..4 {
            assert!((0.0..=MAX_G_PRE_GAIN).contains(&g_ext.into_lr().0.to_array()[lane]));
            assert!((0.0..=1.0).contains(&se_ext.into_lr().0.to_array()[lane]));
        }

        let glu_max = eval_bounded_glu_stereo8(
            InterleavedStereo8::<GpuVec4>::splat_lr(1.0, -1.0),
            InterleavedStereo8::<GpuVec4>::splat(1.0),
            InterleavedStereo8::<GpuVec4>::splat(1.0),
            GpuVec4::zero(),
            GpuVec4::splat(4.0),
            GpuVec4::splat(-4.0),
        );
        assert_eq!(glu_max.into_lr().0.to_array(), [-20.0; 4]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_dynamics_mechanism_a2_0ulp_and_bounds(
            vl in prop::array::uniform4(-1.0e9_f32..1.0e9_f32),
            vr in prop::array::uniform4(-1.0e9_f32..1.0e9_f32),
            g0 in prop::array::uniform4(0.0_f32..2.5_f32),
            g_inf in prop::array::uniform4(0.0_f32..2.5_f32),
            d_pre in prop::array::uniform4(0.0_f32..64.0_f32),
            w_sq in prop::array::uniform4(0.0_f32..16.0_f32),
            d_bus in prop::array::uniform4(0.0_f32..16.0_f32),
            signs in [any::<bool>(); 4],
        ) {
            let s_arr = signs.map(|b| if b { 1.0_f32 } else { -1.0_f32 });
            let sv = GpuVec4::from_array(s_arr);
            let v_orig = InterleavedStereo8::<GpuVec4>::load_lr(&vl, &vr);
            let v_flip = InterleavedStereo8::from_lr(v_orig.left.mul(sv), v_orig.right.mul(sv));

            let g0_v = GpuVec4::from_array(g0);
            let gi_v = GpuVec4::from_array(g_inf);
            let dp_v = GpuVec4::from_array(d_pre);
            let gp_o = eval_g_pre_stereo8(v_orig, g0_v, gi_v, dp_v);
            let gp_f = eval_g_pre_stereo8(v_flip, g0_v, gi_v, dp_v);
            prop_assert_eq!(
                gp_o.into_lr().0.to_array().map(f32::to_bits),
                gp_f.into_lr().0.to_array().map(f32::to_bits)
            );
            prop_assert_eq!(
                gp_o.into_lr().1.to_array().map(f32::to_bits),
                gp_f.into_lr().1.to_array().map(f32::to_bits)
            );

            let ws_v = GpuVec4::from_array(w_sq);
            let db_v = GpuVec4::from_array(d_bus);
            let se_o = eval_se_bus_gain_stereo8(v_orig, ws_v, db_v);
            let se_f = eval_se_bus_gain_stereo8(v_flip, ws_v, db_v);
            prop_assert_eq!(
                se_o.into_lr().0.to_array().map(f32::to_bits),
                se_f.into_lr().0.to_array().map(f32::to_bits)
            );
            prop_assert_eq!(
                se_o.into_lr().1.to_array().map(f32::to_bits),
                se_f.into_lr().1.to_array().map(f32::to_bits)
            );
        }
    }
}
