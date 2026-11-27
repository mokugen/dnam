//! `3`-tap orthogonal folded `TPT-SVF` tone basis (`SonaToneFolded3` on Blocks `1..3`,
//! `SonaToneFoldedBlock0` on Block `0`, eliminating `HP` and `AP` vector evaluations,
//! `ref[dsp.op-n.core.tpt-tone-basis]`, `ref[dsp.op-m.recurrence.symplectic]`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`,
//! and `spirv-unknown-naga-wgsl`-clean.

use crate::simd::{KernelVec4, StereoVec8};
use crate::sona_core::SonaToneWeights4;

use super::TptSvfCoeffs4;

/// Orthogonal `3`-tap folded `TPT-SVF` tone weights `(c_lp, c_bp, c_u)` for Blocks `1..3`
/// (`y_tone = c_lp * LP + c_bp * BP + c_u * u`, eliminating `HP` and `AP`,
/// `ref[dsp.op-n.core.tpt-tone-basis]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaToneFolded3<V: KernelVec4> {
    pub c_lp: V,
    pub c_bp: V,
    pub c_u: V,
}

impl<V: KernelVec4> SonaToneFolded3<V> {
    /// Folds legacy 5-tap [`SonaToneWeights4`] `(lp, bp, hp, ap, dir)` at fixed linear damping `k`:
    /// `c_lp = lp - hp`, `c_bp = bp - k * (hp + 2 * ap)`, `c_u = hp + ap + dir`.
    #[inline(always)]
    #[must_use]
    pub fn from_weights(w: &SonaToneWeights4<V>, k: V) -> Self {
        let hp_plus_2ap = w.hp.fma(V::splat(2.0), w.ap);
        Self {
            c_lp: w.lp.sub(w.hp),
            c_bp: w.bp.fms(k, hp_plus_2ap),
            c_u: w.hp.add(w.ap).add(w.dir),
        }
    }

    /// Selects pure low-pass output (`c_lp = 1.0`, remaining taps `0.0`).
    #[inline(always)]
    #[must_use]
    pub fn lp_only() -> Self {
        Self {
            c_lp: V::splat(1.0),
            c_bp: V::zero(),
            c_u: V::zero(),
        }
    }

    /// Selects unity direct dry bypass (`c_u = 1.0`, `c_lp = c_bp = 0.0`).
    #[inline(always)]
    #[must_use]
    pub fn direct_only() -> Self {
        Self {
            c_lp: V::zero(),
            c_bp: V::zero(),
            c_u: V::splat(1.0),
        }
    }

    /// Evaluates `c_lp * LP + c_bp * BP + c_u * u` across 4 `KernelVec4` lanes.
    #[inline(always)]
    #[must_use]
    pub fn eval_vec4(&self, lp: V, bp: V, u: V) -> V {
        lp.mul(self.c_lp).fma(bp, self.c_bp).fma(u, self.c_u)
    }

    /// Evaluates `c_lp * LP + c_bp * BP + c_u * u` across `StereoVec8` (`0-ULP` stereo parity).
    #[inline(always)]
    #[must_use]
    pub fn eval_stereo8<S: StereoVec8<Half = V>>(&self, lp: S, bp: S, u: S) -> S {
        lp.mul(S::broadcast_half(self.c_lp))
            .fma(bp, S::broadcast_half(self.c_bp))
            .fma(u, S::broadcast_half(self.c_u))
    }
}

/// Block `0` `4`-tap folded `TPT-SVF` tone weights `(c_lp, c_bp, c_kbp, c_u)` supporting
/// time-varying `k_eff[n]` in Predictor-Corrector `PHS` (`y_tone = c_lp * LP + (c_bp + c_kbp * k_eff) * BP + c_u * u`,
/// `ref[dsp.op-n.core.tpt-tone-basis]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaToneFoldedBlock0<V: KernelVec4> {
    pub c_lp: V,
    pub c_bp: V,
    pub c_kbp: V,
    pub c_u: V,
}

impl<V: KernelVec4> SonaToneFoldedBlock0<V> {
    /// Folds 5-tap [`SonaToneWeights4`] into Block `0` PHS form:
    /// `c_lp = lp - hp`, `c_bp = bp`, `c_kbp = -(hp + 2 * ap)`, `c_u = hp + ap + dir`.
    #[inline(always)]
    #[must_use]
    pub fn from_weights(w: &SonaToneWeights4<V>) -> Self {
        Self {
            c_lp: w.lp.sub(w.hp),
            c_bp: w.bp,
            c_kbp: w.hp.fma(V::splat(2.0), w.ap).neg(),
            c_u: w.hp.add(w.ap).add(w.dir),
        }
    }

    /// Lifts a Blocks `1..3` [`SonaToneFolded3`] into Block `0` form (`c_kbp = 0.0`).
    #[inline(always)]
    #[must_use]
    pub fn from_folded3(folded: &SonaToneFolded3<V>) -> Self {
        Self {
            c_lp: folded.c_lp,
            c_bp: folded.c_bp,
            c_kbp: V::zero(),
            c_u: folded.c_u,
        }
    }

    /// Collapses into [`SonaToneFolded3`] at fixed linear damping `k` (`c_bp + c_kbp * k`).
    #[inline(always)]
    #[must_use]
    pub fn to_folded3(&self, k: V) -> SonaToneFolded3<V> {
        SonaToneFolded3 {
            c_lp: self.c_lp,
            c_bp: self.c_bp.fma(self.c_kbp, k),
            c_u: self.c_u,
        }
    }

    /// Selects pure low-pass output (`c_lp = 1.0`).
    #[inline(always)]
    #[must_use]
    pub fn lp_only() -> Self {
        Self::from_folded3(&SonaToneFolded3::lp_only())
    }

    /// Selects unity direct dry bypass (`c_u = 1.0`).
    #[inline(always)]
    #[must_use]
    pub fn direct_only() -> Self {
        Self::from_folded3(&SonaToneFolded3::direct_only())
    }

    /// Evaluates `c_lp * LP + (c_bp + c_kbp * k_eff) * BP + c_u * u` across 4 `KernelVec4` lanes.
    #[inline(always)]
    #[must_use]
    pub fn eval_vec4(&self, lp: V, bp: V, k_eff: V, u: V) -> V {
        let eff_bp = self.c_bp.fma(self.c_kbp, k_eff);
        lp.mul(self.c_lp).fma(bp, eff_bp).fma(u, self.c_u)
    }

    /// Evaluates `c_lp * LP + (c_bp + c_kbp * k_eff) * BP + c_u * u` across `StereoVec8`.
    #[inline(always)]
    #[must_use]
    pub fn eval_stereo8<S: StereoVec8<Half = V>>(&self, lp: S, bp: S, k_eff: S, u: S) -> S {
        let eff_bp = S::broadcast_half(self.c_bp).fma(S::broadcast_half(self.c_kbp), k_eff);
        lp.mul(S::broadcast_half(self.c_lp))
            .fma(bp, eff_bp)
            .fma(u, S::broadcast_half(self.c_u))
    }
}

/// Advances 4 `TPT-SVF` modes and evaluates the 3-tap folded tone mix without computing `HP` or `AP`,
/// returning `(tone_out, bp)` (`ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
pub fn step_tpt_svf_folded3_vec4<V: KernelVec4>(
    u: V,
    s1: &mut V,
    s2: &mut V,
    coeffs: &TptSvfCoeffs4<V>,
    tone: &SonaToneFolded3<V>,
) -> (V, V) {
    let v3 = u.sub(*s2);
    let v1 = coeffs.a1.mul(*s1).fma(coeffs.a2, v3);
    let v2 = s2.fma(coeffs.a2, *s1).fma(coeffs.a3, v3);
    *s1 = v1.add(v1).sub(*s1).flush_recursive_state();
    *s2 = v2.add(v2).sub(*s2).flush_recursive_state();
    (tone.eval_vec4(v2, v1, u), v1)
}

/// Advances stereo `L+R` 4-mode `TPT-SVF` states and evaluates the 3-tap folded tone mix without
/// computing `HP` or `AP`, returning `(tone_out, bp)` (`0-ULP` stereo parity,
/// `ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
pub fn step_tpt_svf_folded3_stereo8<S: StereoVec8>(
    u: S,
    s1: &mut S,
    s2: &mut S,
    coeffs: &TptSvfCoeffs4<S::Half>,
    tone: &SonaToneFolded3<S::Half>,
) -> (S, S) {
    let a1 = S::broadcast_half(coeffs.a1);
    let a2 = S::broadcast_half(coeffs.a2);
    let a3 = S::broadcast_half(coeffs.a3);
    let v3 = u.sub(*s2);
    let v1 = a1.mul(*s1).fma(a2, v3);
    let v2 = s2.fma(a2, *s1).fma(a3, v3);
    *s1 = v1.add(v1).sub(*s1).flush_recursive_state();
    *s2 = v2.add(v2).sub(*s2).flush_recursive_state();
    (tone.eval_stereo8(v2, v1, u), v1)
}

#[cfg(test)]
mod tests {
    use super::super::{prewarp_tan_pi_f_over_fs_f32, step_tpt_svf_stereo8};
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    fn solve_lr4_mid_partial_fraction_f64(g1: f64, g2: f64) -> (f64, f64, f64, f64) {
        let k = core::f64::consts::SQRT_2;
        let mut m = [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [k * g2, 1.0, k * g1, 1.0, g2 * g2],
            [g2 * g2, k * g2, g1 * g1, k * g1, 0.0],
            [0.0, g2 * g2, 0.0, g1 * g1, 0.0],
        ];
        for col in 0..4 {
            let mut piv = col;
            for r in (col + 1)..4 {
                if m[r][col].abs() > m[piv][col].abs() {
                    piv = r;
                }
            }
            m.swap(col, piv);
            let div = m[col][col];
            for item in &mut m[col][col..5] {
                *item /= div;
            }
            for r in 0..4 {
                if r != col {
                    let f = m[r][col];
                    for j in col..5 {
                        m[r][j] -= f * m[col][j];
                    }
                }
            }
        }
        let (a1, b1, a2, b2) = (m[0][4], m[1][4], m[2][4], m[3][4]);
        (b1 / (g1 * g1), a1 / g1, b2 / (g2 * g2), a2 / g2)
    }

    /// verify[dsp.op-n.core.tpt-tone-basis]
    #[test]
    fn test_folded_tone_2block_lr4_mid_bandpass_parity_vs_f64_oracle() {
        let fs = 96_000.0_f64;
        let f_lo = 300.0_f64;
        let f_hi = 3_000.0_f64;
        let g1 = (core::f64::consts::PI * f_lo / fs).tan();
        let g2 = (core::f64::consts::PI * f_hi / fs).tan();
        let (c_lp1, c_bp1, c_lp2, c_bp2) = solve_lr4_mid_partial_fraction_f64(g1, g2);

        let g_vec = GpuVec4::from_array([g1 as f32, g2 as f32, 0.01, 0.01]);
        let k_vec = GpuVec4::splat(core::f32::consts::SQRT_2);
        let coeffs = TptSvfCoeffs4::from_gk(g_vec, k_vec);
        let tone = SonaToneFolded3 {
            c_lp: GpuVec4::from_array([c_lp1 as f32, c_lp2 as f32, 0.0, 0.0]),
            c_bp: GpuVec4::from_array([c_bp1 as f32, c_bp2 as f32, 0.0, 0.0]),
            c_u: GpuVec4::zero(),
        };

        for &freq_hz in &[80.0_f64, 300.0, 1_000.0, 3_000.0, 10_000.0] {
            let mut b0_s1 = GpuVec4::zero();
            let mut b0_s2 = GpuVec4::zero();
            let mut b1_s1 = GpuVec4::zero();
            let mut b1_s2 = GpuVec4::zero();

            // 4th-order f64 LR4 cascade oracle: HP2(f_lo)^2 * LP2(f_hi)^2
            let w_dig = core::f64::consts::TAU * freq_hz / fs;
            let k_bw = core::f64::consts::SQRT_2;
            let a1_1 = 1.0 / (1.0 + g1 * (g1 + k_bw));
            let a2_1 = g1 * a1_1;
            let a3_1 = g1 * a2_1;
            let a1_2 = 1.0 / (1.0 + g2 * (g2 + k_bw));
            let a2_2 = g2 * a1_2;
            let a3_2 = g2 * a2_2;
            let mut ref_s = [[0.0_f64; 2]; 4];

            let mut max_err = 0.0_f64;
            for n in 0..512 {
                let x = (w_dig * (n as f64)).sin();
                let (t0, _) = step_tpt_svf_folded3_vec4(
                    GpuVec4::splat(x as f32),
                    &mut b0_s1,
                    &mut b0_s2,
                    &coeffs,
                    &tone,
                );
                let arr0 = t0.to_array();
                let y_blk0 = arr0[0] + arr0[1];
                let (t1, _) = step_tpt_svf_folded3_vec4(
                    GpuVec4::splat(y_blk0),
                    &mut b1_s1,
                    &mut b1_s2,
                    &coeffs,
                    &tone,
                );
                let arr1 = t1.to_array();
                let y_lr4 = f64::from(arr1[0] + arr1[1]);

                // Cascade 2x HP2(g1) + 2x LP2(g2) in f64
                let mut u_r = x;
                for st in &mut ref_s[0..2] {
                    let v3 = u_r - st[1];
                    let v1 = a1_1 * st[0] + a2_1 * v3;
                    let v2 = st[1] + a2_1 * st[0] + a3_1 * v3;
                    st[0] = 2.0 * v1 - st[0];
                    st[1] = 2.0 * v2 - st[1];
                    u_r = u_r - k_bw * v1 - v2; // HP2
                }
                for st in &mut ref_s[2..4] {
                    let v3 = u_r - st[1];
                    let v1 = a1_2 * st[0] + a2_2 * v3;
                    let v2 = st[1] + a2_2 * st[0] + a3_2 * v3;
                    st[0] = 2.0 * v1 - st[0];
                    st[1] = 2.0 * v2 - st[1];
                    u_r = v2; // LP2
                }
                max_err = max_err.max((y_lr4 - u_r).abs());
            }
            assert!(max_err < 1.0e-3, "freq={freq_hz}: max_err={max_err}");
        }
    }

    proptest! {
        #[test]
        fn prop_folded3_and_block0_match_5tap_tone_weights(
            ratios in prop::array::uniform4(1.0e-3_f32..0.42_f32),
            k_arr in prop::array::uniform4(0.0_f32..8.0_f32),
            w_lp in prop::array::uniform4(-4.0_f32..4.0_f32),
            w_bp in prop::array::uniform4(-4.0_f32..4.0_f32),
            w_hp in prop::array::uniform4(-4.0_f32..4.0_f32),
            w_ap in prop::array::uniform4(-4.0_f32..4.0_f32),
            w_dir in prop::array::uniform4(-4.0_f32..4.0_f32),
            u_l in prop::collection::vec(-2.0_f32..2.0_f32, 8),
            u_r in prop::collection::vec(-2.0_f32..2.0_f32, 8),
        ) {
            let g = GpuVec4::from_array(ratios.map(prewarp_tan_pi_f_over_fs_f32));
            let k = GpuVec4::from_array(k_arr);
            let coeffs = TptSvfCoeffs4::from_gk(g, k);
            let w5 = SonaToneWeights4 {
                lp: GpuVec4::from_array(w_lp),
                bp: GpuVec4::from_array(w_bp),
                hp: GpuVec4::from_array(w_hp),
                ap: GpuVec4::from_array(w_ap),
                dir: GpuVec4::from_array(w_dir),
            };
            let f3 = SonaToneFolded3::from_weights(&w5, k);
            let fb0 = SonaToneFoldedBlock0::from_weights(&w5);
            prop_assert_eq!(fb0.to_folded3(k), f3);

            let mut s1_5 = InterleavedStereo8::<GpuVec4>::zero();
            let mut s2_5 = InterleavedStereo8::<GpuVec4>::zero();
            let mut s1_3 = InterleavedStereo8::<GpuVec4>::zero();
            let mut s2_3 = InterleavedStereo8::<GpuVec4>::zero();
            let mut s1_v = GpuVec4::zero();
            let mut s2_v = GpuVec4::zero();

            for (&xl, &xr) in u_l.iter().zip(u_r.iter()) {
                let u8 = InterleavedStereo8::splat_lr(xl, xr);
                let o5 = step_tpt_svf_stereo8(u8, &mut s1_5, &mut s2_5, &coeffs);
                let y5 = o5.lp.mul(InterleavedStereo8::broadcast_half(w5.lp))
                    .fma(o5.bp, InterleavedStereo8::broadcast_half(w5.bp))
                    .fma(o5.hp, InterleavedStereo8::broadcast_half(w5.hp))
                    .fma(o5.ap, InterleavedStereo8::broadcast_half(w5.ap))
                    .fma(u8, InterleavedStereo8::broadcast_half(w5.dir));
                let (y3, bp3) = step_tpt_svf_folded3_stereo8(u8, &mut s1_3, &mut s2_3, &coeffs, &f3);
                let (yv_l, bpv_l) = step_tpt_svf_folded3_vec4(GpuVec4::splat(xl), &mut s1_v, &mut s2_v, &coeffs, &f3);
                let yb0 = fb0.eval_stereo8(o5.lp, o5.bp, InterleavedStereo8::broadcast_half(k), u8);

                prop_assert_eq!(y3.into_lr().0.to_array().map(f32::to_bits), yv_l.to_array().map(f32::to_bits));
                prop_assert_eq!(bp3.into_lr().0.to_array().map(f32::to_bits), bpv_l.to_array().map(f32::to_bits));
                let (y5l, y5r) = y5.into_lr();
                let (y3l, y3r) = y3.into_lr();
                let (ybl, _) = yb0.into_lr();
                for lane in 0..4 {
                    prop_assert!((y5l.to_array()[lane] - y3l.to_array()[lane]).abs() < 2.0e-5);
                    prop_assert!((y5r.to_array()[lane] - y3r.to_array()[lane]).abs() < 2.0e-5);
                    prop_assert!((y5l.to_array()[lane] - ybl.to_array()[lane]).abs() < 2.0e-5);
                }
            }
        }
    }

    #[test]
    fn test_folded_tone_lp_only_direct_only_and_from_folded3() {
        let lp = GpuVec4::from_array([0.2, -0.4, 0.6, -0.8]);
        let bp = GpuVec4::from_array([0.1, 0.3, -0.5, 0.7]);
        let u = GpuVec4::from_array([0.9, -0.1, 0.25, -0.75]);
        let k = GpuVec4::splat(0.75);

        let f3_lp = SonaToneFolded3::<GpuVec4>::lp_only();
        let fb0_lp = SonaToneFoldedBlock0::<GpuVec4>::lp_only();
        assert_eq!(
            fb0_lp.eval_vec4(lp, bp, k, u).to_array().map(f32::to_bits),
            lp.to_array().map(f32::to_bits)
        );
        assert_eq!(SonaToneFoldedBlock0::from_folded3(&f3_lp), fb0_lp);

        let fb0_dir = SonaToneFoldedBlock0::<GpuVec4>::direct_only();
        assert_eq!(
            fb0_dir.eval_vec4(lp, bp, k, u).to_array().map(f32::to_bits),
            u.to_array().map(f32::to_bits)
        );
    }
}
