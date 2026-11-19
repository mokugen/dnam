//! `Op-T` SIMD Interpolation Backend, Granular Windowing & Ballistic Ducking VCA (`Family 3, 4, 6`,
//! `ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.op-t.window.partition-of-unity]`,
//! `ref[dsp.op-t.ducking.vca]`).

use crate::math::{SimdVec4, floor_f32, flush_recursive_state_f32};

/// Domain-specific SIMD backend trait for `Op-T` (Transport, Catmull-Rom & Cubic Lagrange Reader).
pub trait OpTBackend<V: SimdVec4> {
    /// Evaluates 4-tap Cubic Lagrange interpolation weights for 4 delay channels simultaneously.
    fn cubic_lagrange_weights_4ch(vmu: V) -> (V, V, V, V);
}

/// Exact compile-time reciprocal `1 / 6` for 4-tap Cubic Lagrange outer weights (`0 FDIV`,
/// `ref[dsp.op-t.reader.catmull-rom]`, `ref[repo.code.constant-eval-and-derivations]`).
const LAGRANGE_INV_6: f32 = 1.0 / 6.0;

/// Exact compile-time negated reciprocal `-1 / 6` for 4-tap Cubic Lagrange outer weights.
const LAGRANGE_NEG_INV_6: f32 = -LAGRANGE_INV_6;

/// Exact compile-time inner weight scale `1 / 2` for 4-tap Cubic Lagrange interpolation.
const LAGRANGE_HALF: f32 = 0.5;

/// Exact compile-time negated inner weight scale `-1 / 2` for 4-tap Cubic Lagrange interpolation.
const LAGRANGE_NEG_HALF: f32 = -LAGRANGE_HALF;

const _: () = assert!(LAGRANGE_INV_6 > 0.16 && LAGRANGE_INV_6 < 0.17);
const _: () = assert!((LAGRANGE_INV_6 + LAGRANGE_NEG_INV_6).to_bits() == 0);
const _: () = assert!((LAGRANGE_HALF + LAGRANGE_NEG_HALF).to_bits() == 0);

/// Safe `SimdVec4` implementation of [`OpTBackend::cubic_lagrange_weights_4ch`].
#[inline(always)]
pub fn cubic_lagrange_weights_4ch_vec4<V: SimdVec4>(vmu: V) -> (V, V, V, V) {
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    let inv_six = V::splat(LAGRANGE_INV_6);
    let neg_inv_six = V::splat(LAGRANGE_NEG_INV_6);
    let half = V::splat(LAGRANGE_HALF);
    let neg_half = V::splat(LAGRANGE_NEG_HALF);

    let mu_p1 = vmu.add(one);
    let mu_m1 = vmu.sub(one);
    let mu_m2 = vmu.sub(two);

    let c_m1 = vmu.mul(mu_m1).mul(mu_m2).mul(neg_inv_six);
    let c_0 = mu_p1.mul(mu_m1).mul(mu_m2).mul(half);
    let c_1 = mu_p1.mul(vmu).mul(mu_m2).mul(neg_half);
    let c_2 = mu_p1.mul(vmu).mul(mu_m1).mul(inv_six);
    (c_m1, c_0, c_1, c_2)
}

impl OpTBackend<crate::math::ScalarVec4> for crate::math::ScalarBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::ScalarVec4,
    ) -> (
        crate::math::ScalarVec4,
        crate::math::ScalarVec4,
        crate::math::ScalarVec4,
        crate::math::ScalarVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpTBackend<crate::math::NeonVec4> for crate::math::NeonBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::NeonVec4,
    ) -> (
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpTBackend<crate::math::Sve2Vec4> for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Sve2Vec4,
    ) -> (
        crate::math::Sve2Vec4,
        crate::math::Sve2Vec4,
        crate::math::Sve2Vec4,
        crate::math::Sve2Vec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpTBackend<crate::math::NeonVec4> for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::NeonVec4,
    ) -> (
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
        crate::math::NeonVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpTBackend<crate::math::Sse2Vec4> for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Sse2Vec4,
    ) -> (
        crate::math::Sse2Vec4,
        crate::math::Sse2Vec4,
        crate::math::Sse2Vec4,
        crate::math::Sse2Vec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpTBackend<crate::math::Sse42Vec4> for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Sse42Vec4,
    ) -> (
        crate::math::Sse42Vec4,
        crate::math::Sse42Vec4,
        crate::math::Sse42Vec4,
        crate::math::Sse42Vec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpTBackend<crate::math::Avx2FmaVec4> for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::Avx2FmaVec4,
    ) -> (
        crate::math::Avx2FmaVec4,
        crate::math::Avx2FmaVec4,
        crate::math::Avx2FmaVec4,
        crate::math::Avx2FmaVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

#[cfg(target_arch = "wasm32")]
impl OpTBackend<crate::math::WasmVec4> for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn cubic_lagrange_weights_4ch(
        vmu: crate::math::WasmVec4,
    ) -> (
        crate::math::WasmVec4,
        crate::math::WasmVec4,
        crate::math::WasmVec4,
        crate::math::WasmVec4,
    ) {
        cubic_lagrange_weights_4ch_vec4(vmu)
    }
}

/// Maximum delay line capacity for Op-T Lite buffer (`2,048` samples = `4,096` bytes fp16).
pub const OP_T_LITE_RING_CAPACITY: usize = 2048;
pub const OP_T_FULL_RING_CAPACITY: usize = 4096;
pub const OP_T_ULTRA_RING_CAPACITY: usize = 8192;

/// Co-prime Schroeder all-pass diffuser delays (`ref[dsp.op-t.diffuser.recurrence]`).
pub const SCHROEDER_COPRIME_DELAYS: [usize; 8] = [19, 31, 47, 73, 103, 139, 179, 223];

/// Stateless 4-tap Catmull-Rom cubic Hermite interpolation in Horner form delegated to
/// [`dnam_kernels::catmull_rom_hermite_f32`] (`ref[dsp.op-t.reader.catmull-rom]`).
#[inline(always)]
pub fn catmull_rom_hermite_cubic(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    dnam_kernels::catmull_rom_hermite_f32(s_m1, s_0, s_1, s_2, mu)
}

/// Computes partition-of-unity grain weights for Hermite-Hann grains (Mode M = 2, Family 3) via
/// [`dnam_kernels::hermite_hann_partition_of_unity_f32`].
///
/// Exact identity: `H(phi_A) + H(phi_B) == 1.000000000000`.
#[inline(always)]
pub fn hermite_hann_partition_of_unity(phi_a: f32) -> (f32, f32) {
    let [h_a, h_b] = dnam_kernels::hermite_hann_partition_of_unity_f32(phi_a);
    (h_a, h_b)
}

/// Computes partition-of-unity 3-phase raised-cosine weights (Mode M = 3, Family 6) without
/// `libm` `.fract()` or ad-hoc polynomial literals, delegating cosine evaluation to
/// [`dnam_kernels::algebraic_unit_rotor_f32`].
///
/// Exact identity: `W0 + W1 + W2 == 1.000000000000` with non-negative weights.
#[inline(always)]
pub fn raised_cosine_barberpole_3phase(phi: f32) -> [f32; 3] {
    const TWO_PI: f32 = core::f32::consts::TAU;
    const ONE_THIRD: f32 = 1.0 / 3.0;

    let wrap01 = |x: f32| -> f32 {
        let f = x - floor_f32(x);
        if f >= 1.0 { 0.0 } else { f.max(0.0) }
    };

    let phi0 = wrap01(phi);
    let phi1 = wrap01(phi + ONE_THIRD);

    let (c0, _) = dnam_kernels::algebraic_unit_rotor_f32(phi0 * TWO_PI);
    let (c1, _) = dnam_kernels::algebraic_unit_rotor_f32(phi1 * TWO_PI);

    let w0 = ONE_THIRD * (1.0 + c0).max(0.0);
    let w1 = ONE_THIRD * (1.0 + c1).max(0.0);
    let w2 = (1.0 - w0 - w1).max(0.0);
    [w0, w1, w2]
}

/// 1-pole ballistic envelope ducking VCA (Family 4, `ref[dsp.op-t.ducking.vca]`).
#[derive(Clone, Copy, Debug)]
pub struct BallisticDuckingVca {
    pub env: f32,
    pub alpha_att: f32,
    pub alpha_rel: f32,
    pub threshold: f32,
    pub kappa: f32,
    pub min_gain: f32,
}

impl Default for BallisticDuckingVca {
    fn default() -> Self {
        Self::new()
    }
}

impl BallisticDuckingVca {
    pub const fn new() -> Self {
        Self {
            env: 0.0,
            alpha_att: 0.99,
            alpha_rel: 0.9995,
            threshold: 0.1,
            kappa: 2.0,
            min_gain: 0.1,
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.env = 0.0;
    }

    #[inline(always)]
    fn clamp_param(value: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
        if value.is_finite() {
            value.clamp(lo, hi)
        } else {
            fallback
        }
    }

    #[inline(always)]
    fn eval_ducking_gain(&mut self, abs_sc: f32) -> f32 {
        let alpha_att = Self::clamp_param(self.alpha_att, 0.5, 0.9999, 0.99);
        let alpha_rel = Self::clamp_param(self.alpha_rel, 0.5, 0.99999, 0.9995);
        let thresh = Self::clamp_param(self.threshold, 0.0, 2.0, 0.1);
        let kappa = Self::clamp_param(self.kappa, 0.0, 8.0, 2.0);
        let min_gain = Self::clamp_param(self.min_gain, 0.05, 1.0, 0.1);

        let prev_env = flush_recursive_state_f32(self.env);
        let alpha = if abs_sc > prev_env {
            alpha_att
        } else {
            alpha_rel
        };
        let next_env = alpha * prev_env + (1.0 - alpha) * abs_sc;
        self.env = flush_recursive_state_f32(next_env);
        // Compute excess strictly from the flushed state `self.env` so `Inf` sidechains never yield `0.0 * Inf = NaN`.
        let excess = (self.env - thresh).max(0.0);
        (1.0 - kappa * excess).clamp(min_gain, 1.0)
    }

    /// Applies ducking to one wet sample. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, sidechain: f32, wet: f32) -> f32 {
        self.eval_ducking_gain(sidechain.abs()) * wet
    }

    /// Evaluates a stereo-linked ballistic ducking step once per stereo frame (`1x fs`),
    /// preserving the stereo image `(wet_l, wet_r)` with identical L/R ducking gain.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(
        &mut self,
        sidechain_l: f32,
        sidechain_r: f32,
        wet_l: f32,
        wet_r: f32,
    ) -> (f32, f32) {
        let sc_l_clean = flush_recursive_state_f32(sidechain_l);
        let sc_r_clean = flush_recursive_state_f32(sidechain_r);
        let g = self.eval_ducking_gain(0.5 * sc_l_clean.abs() + 0.5 * sc_r_clean.abs());
        (g * wet_l, g * wet_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::math::{DefaultMathBackend, MathBackend};
    use proptest::prelude::*;

    #[test]
    fn test_interpolation_windowing_and_lagrange() {
        let v = catmull_rom_hermite_cubic(1.0, 2.0, 3.0, 4.0, 0.5);
        assert!((v - 2.5).abs() < 1e-4);

        for phi in [-2.3_f32, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 3.7] {
            let (ha, hb) = hermite_hann_partition_of_unity(phi - floor_f32(phi));
            assert!((ha + hb - 1.0).abs() < 1e-6);

            let [w0, w1, w2] = raised_cosine_barberpole_3phase(phi);
            assert!(w0 >= -1e-6 && w1 >= -1e-6 && w2 >= -1e-6);
            assert!((w0 + w1 + w2 - 1.0).abs() < 1e-5);
        }

        let mu = <DefaultMathBackend as MathBackend>::Vec4::from_array([0.0, 0.25, 0.5, 1.0]);
        let (cm1, c0, c1, c2) = DefaultMathBackend::cubic_lagrange_weights_4ch(mu);
        let sum = cm1.add(c0).add(c1).add(c2).to_array();
        for s in sum {
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ballistic_ducking_vca() {
        let _audio = FtzDazGuard::new();
        let mut vca = BallisticDuckingVca::default();
        let out_low = vca.step(0.01, 1.0);
        assert!((out_low - 1.0).abs() < 0.05);

        for _ in 0..50 {
            let _ = vca.step(0.9, 1.0);
        }
        let out_ducked = vca.step(0.9, 1.0);
        assert!(out_ducked < 0.9);

        let mut vca_s = BallisticDuckingVca::default();
        let (dl, dr) = vca_s.step_stereo(1.0, 0.0, 0.8, 0.4);
        assert!((dl - 2.0 * dr).abs() < 1e-6);

        // Inf/NaN sidechain inputs (even with kappa = 0.0) must not yield 0.0 * Inf = NaN or latch NaN.
        let mut vca_nan = BallisticDuckingVca {
            kappa: 0.0,
            ..Default::default()
        };
        for bad_sc in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let y = vca_nan.step(bad_sc, 1.0);
            assert!(
                y.is_finite(),
                "mono step produced non-finite {y} on {bad_sc}"
            );
            assert!(vca_nan.env.is_finite());

            let (yl, yr) = vca_nan.step_stereo(bad_sc, 0.25, 0.75, -0.5);
            assert!(yl.is_finite() && yr.is_finite());
            assert!(vca_nan.env.is_finite());
        }
        vca_nan.kappa = 2.0;
        let rec = vca_nan.step(0.9, 1.0);
        assert!(rec.is_finite());
        assert!(vca_nan.env.is_finite() && vca_nan.env > 0.0);

        let mut vca_max = BallisticDuckingVca::default();
        let (max_l, max_r) = vca_max.step_stereo(f32::MAX, f32::MAX, 1.0, -0.5);
        assert!(max_l.is_finite() && max_r.is_finite());
        assert!(vca_max.env.is_finite() && vca_max.env > 0.0);
        assert!(max_l.abs() < 1.0 && max_r.abs() < 0.5);
    }

    #[test]
    fn test_catmull_rom_exact_dc_gain() {
        let _audio = FtzDazGuard::new();
        for step in 0..=100 {
            let mu = (step as f32) / 100.0;
            let val = catmull_rom_hermite_cubic(1.0, 1.0, 1.0, 1.0, mu);
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_catmull_rom_linear_reproduction_and_windows(
            a in -5.0f32..5.0f32,
            b in -5.0f32..5.0f32,
            mu in 0.0f32..1.0f32,
            phi in -10.0f32..10.0f32,
        ) {
            let ym1 = a - b;
            let y0 = a;
            let y1 = a + b;
            let y2 = a + 2.0 * b;
            let interp = catmull_rom_hermite_cubic(ym1, y0, y1, y2, mu);
            let exact = a + b * mu;
            prop_assert!((interp - exact).abs() < 1e-4);

            let (ha, hb) = hermite_hann_partition_of_unity(mu);
            prop_assert!((ha + hb - 1.0).abs() < 1e-6);
            prop_assert!(ha >= -1e-6 && hb >= -1e-6);

            let [w0, w1, w2] = raised_cosine_barberpole_3phase(phi);
            prop_assert!((w0 + w1 + w2 - 1.0).abs() < 1e-5);
            prop_assert!(w0 >= -1e-5 && w1 >= -1e-5 && w2 >= -1e-5);
        }
    }
}
