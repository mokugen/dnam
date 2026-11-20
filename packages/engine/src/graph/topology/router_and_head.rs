//! Stage 0 8-Port Router (WDW, Kaiser-Sinc Sub-Sample Alignment, 200 Hz TPT LR4 Crossover,
//! Stereo-Linked & Aux-Key Detector), Stage IV Blended Quiet-Feedback Knee & Outer-Clamped Loop
//! Saturator, and 256-D Feature Bus Slot Allocation.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_crossbar_topology.md` (`ref[dsp.router.wdw-contract]`, `ref[dsp.router.parallel-alignment]`, `ref[dsp.router.lr4-crossover]`, `ref[dsp.router.stereo-link-detector]`, `ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.tank.loop-saturator]`, `ref[dsp.tank.quiet-feedback-knee]`, `ref[dsp.head.direct-register-readout]`).

pub mod tank_and_laws;

use alloc::vec::Vec;

use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4, newton6_reciprocal_f64, sqrt_f64,
};
use crate::operators::op_t::catmull_rom_hermite_cubic;
use crate::schema::cold_exp_f64;

pub use tank_and_laws::{
    TANK_MAX_G_LIN_BITS, TANK_MAX_V_CEIL_BITS, cabinet_feature_bus_slot_ranges,
    eval_stage_modulation_laws, eval_tank_blended_feedback_knee,
    eval_tank_blended_feedback_knee_fast, eval_tank_loop_saturator, eval_tank_loop_saturator_fast,
    pack_cabinet_modal_block_features, validate_tensor_directory_slice,
};

/// Wet-Dry-Wet (`WDW`) center dry tap mode (`ref[dsp.router.wdw-contract]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WdwDryTapMode {
    /// Center dry on `Port Out 1` is the Mid contribution `m_c` of the crossbar + core + cabinet;
    /// `Ports Out 2/3` carry Spatial Tank contribution `m_w ± alpha * s_w`.
    PreSpatialCore,
    /// Center dry on `Port Out 1` is voltage-normalized Mid `0.5 * (x_L + x_R)` of the selected
    /// direct source; `Ports Out 2/3` carry the wet effect response.
    DirectDry,
}

/// Evaluates the WDW output ports `(port_out1_dry, port_out2_wet_l, port_out3_wet_r)`
/// (`ref[dsp.router.wdw-contract]`).
#[inline]
pub fn eval_wdw_routing_outputs(
    mode: WdwDryTapMode,
    core_mid: f32,
    wet_mid: f32,
    wet_side: f32,
    width_alpha: f32,
    direct_l: f32,
    direct_r: f32,
) -> (f32, f32, f32) {
    let wet_l = wet_mid + width_alpha * wet_side;
    let wet_r = wet_mid - width_alpha * wet_side;
    let dry = match mode {
        WdwDryTapMode::PreSpatialCore => core_mid,
        WdwDryTapMode::DirectDry => 0.5 * (direct_l + direct_r),
    };
    (dry, wet_l, wet_r)
}

/// Parallel-path sub-sample alignment mode (`ref[dsp.router.parallel-alignment]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelAlignmentMode {
    /// 64-tap Kaiser-derived windowed sinc (`beta = 8.6`, `G = 31` host samples).
    FullBand64Tap,
    /// 16-tap Kaiser-derived windowed sinc (`beta = 6.5`, `G = 7` host samples).
    LowLatency16Tap,
    /// Centered 4-tap Catmull-Rom cubic Hermite (`G = 1` host sample).
    MinLatency4Tap,
}

impl ParallelAlignmentMode {
    /// Returns the host-rate centering offset `G` in samples (`31`, `7`, or `1`).
    #[inline]
    pub const fn centering_offset_host_samples(self) -> usize {
        match self {
            Self::FullBand64Tap => 31,
            Self::LowLatency16Tap => 7,
            Self::MinLatency4Tap => 1,
        }
    }

    /// Returns `(base_offset_b, effective_delay_samples, combined_pdc_host_samples)` at oversampling
    /// factor `oversampling` (`1` or `4`, `ref[dsp.router.parallel-alignment]`, `ref[arch.bounce.oversampling]`).
    #[inline]
    pub const fn timing_at_oversampling(
        self,
        oversampling: usize,
        delta_samples_eff: f64,
    ) -> (usize, f64, usize) {
        let g = self.centering_offset_host_samples();
        let b = (oversampling - 1) * g;
        let d_eff = (oversampling * g) as f64 + delta_samples_eff;
        let pdc_host = if oversampling == 4 { g + 12 } else { g };
        (b, d_eff, pdc_host)
    }
}

const fn build_inv_m_sq_table() -> [f64; 29] {
    let mut out = [0.0_f64; 29];
    let mut m = 1usize;
    while m <= 28 {
        out[m] = 1.0 / ((m * m) as f64);
        m += 1;
    }
    out
}

const INV_M_SQ: [f64; 29] = build_inv_m_sq_table();
const INV_6_F64: f64 = 1.0 / 6.0;
const INV_120_F64: f64 = 1.0 / 120.0;
const INV_5040_F64: f64 = 1.0 / 5_040.0;
const INV_362880_F64: f64 = 1.0 / 362_880.0;
const INV_39916800_F64: f64 = 1.0 / 39_916_800.0;
const INV_24_F64: f64 = 1.0 / 24.0;
const INV_720_F64: f64 = 1.0 / 720.0;
const INV_40320_F64: f64 = 1.0 / 40_320.0;

fn bessel_i0_f64(z: f64) -> f64 {
    let half = 0.5 * z;
    let half2 = half * half;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for &inv_m2 in &INV_M_SQ[1..=28] {
        term *= half2 * inv_m2;
        sum += term;
    }
    sum
}

#[inline]
fn normalized_sinc_f64(x: f64) -> f64 {
    if x.abs() < 1.0e-12 {
        1.0
    } else {
        let pix = core::f64::consts::PI * x;
        let k = crate::math::round_f64(x) as i64;
        let rem = x - (k as f64);
        let sign = if (k & 1) != 0 { -1.0 } else { 1.0 };
        let u = core::f64::consts::PI * rem;
        let u2 = u * u;
        let sin_u = u
            * (1.0
                - u2 * (INV_6_F64
                    - u2 * (INV_120_F64
                        - u2 * (INV_5040_F64 - u2 * (INV_362880_F64 - u2 * INV_39916800_F64)))));
        let inv_pix = if pix >= 0.0 {
            newton6_reciprocal_f64(pix)
        } else {
            -newton6_reciprocal_f64(-pix)
        };
        (sign * sin_u) * inv_pix
    }
}

/// Zero-allocation stack writer for normalized zero-ended Kaiser-derived sinc weights $h_k(\mu)$
/// (`num_taps in {16, 64}`, `beta > 0`, `0 <= mu < 1`, `ref[dsp.router.parallel-alignment]`).
pub fn prepare_kaiser_sinc_alignment_weights_into(
    num_taps: usize,
    beta: f64,
    mu: f64,
    out: &mut [f64; 64],
) -> bool {
    if !matches!(num_taps, 16 | 64) || !(0.0..1.0).contains(&mu) || beta <= 0.0 {
        return false;
    }
    let g = ((num_taps - 2) / 2) as f64;
    let inv_r = 2.0 * newton6_reciprocal_f64(num_taps as f64);
    let denom_i0 = (bessel_i0_f64(beta) - 1.0).max(1.0e-15);
    let inv_denom_i0 = newton6_reciprocal_f64(denom_i0);
    let mut sum = 0.0_f64;
    for (k, slot) in out[..num_taps].iter_mut().enumerate() {
        let s = (k as f64) - g - mu;
        let ratio = s * inv_r;
        let w = if ratio.abs() >= 1.0 {
            0.0
        } else {
            let arg = beta * sqrt_f64((1.0 - ratio * ratio).max(0.0));
            ((bessel_i0_f64(arg) - 1.0) * inv_denom_i0).max(0.0)
        };
        let h = normalized_sinc_f64(s) * w;
        *slot = h;
        sum += h;
    }
    out[num_taps..].fill(0.0);
    let inv_sum = if sum >= 0.0 {
        newton6_reciprocal_f64(sum.max(1.0e-15))
    } else {
        -newton6_reciprocal_f64((-sum).max(1.0e-15))
    };
    for v in &mut out[..num_taps] {
        *v *= inv_sum;
    }
    true
}

/// Prepares normalized zero-ended Kaiser-derived sinc weights $h_k(\mu)$ for $K \in \{16, 64\}$
/// (`beta = 8.6` for `K = 64`, `beta = 6.5` for `K = 16`, `0 <= mu < 1`,
/// `ref[dsp.router.parallel-alignment]`).
pub fn prepare_kaiser_sinc_alignment_weights(
    num_taps: usize,
    beta: f64,
    mu: f64,
) -> Option<Vec<f64>> {
    let mut buf = [0.0_f64; 64];
    if !prepare_kaiser_sinc_alignment_weights_into(num_taps, beta, mu, &mut buf) {
        return None;
    }
    Some(buf[..num_taps].to_vec())
}

/// Evaluates `MinLatency4Tap` Catmull-Rom alignment for `[s_{-1}, s_0, s_1, s_2]` (`G = 1`,
/// `ref[dsp.router.parallel-alignment]`).
#[inline]
pub fn eval_min_latency_4tap_alignment(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    catmull_rom_hermite_cubic(s_m1, s_0, s_1, s_2, mu)
}

/// 4th-order Linkwitz–Riley (`LR4`) Clean/Drive Crossover implemented as two cascaded 2nd-order
/// Butterworth trapezoidal state-variable filter (`TPT SVF`) sections (`Q = 1/sqrt(2)`,
/// `ref[dsp.router.lr4-crossover]`).
#[derive(Clone, Copy, Debug)]
pub struct TptLr4Crossover {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k_damp: f32,
    pub lp_s1: [f32; 2],
    pub lp_s2: [f32; 2],
    pub hp_s1: [f32; 2],
    pub hp_s2: [f32; 2],
    pub sec2_ic1: [f32; 4],
    pub sec2_ic2: [f32; 4],
}

impl Default for TptLr4Crossover {
    fn default() -> Self {
        Self::new_at_rate(200.0, 48_000.0)
    }
}

impl TptLr4Crossover {
    /// Const-initialized `200 Hz` at `48 kHz` `TPT SVF` LR4 crossover (`ref[dsp.router.lr4-crossover]`).
    pub const fn new_48khz() -> Self {
        Self {
            a1: 0.981_658_2,
            a2: 0.012_850_65,
            a3: 0.000_168_23,
            k_damp: core::f32::consts::SQRT_2,
            lp_s1: [0.0; 2],
            lp_s2: [0.0; 2],
            hp_s1: [0.0; 2],
            hp_s2: [0.0; 2],
            sec2_ic1: [0.0; 4],
            sec2_ic2: [0.0; 4],
        }
    }

    /// Prepares `TPT SVF` coefficients in `f64` for cutoff `fc_hz` (default `200 Hz`) at
    /// `effective_rate_hz` (`ref[dsp.router.lr4-crossover]`, `0 FDIV`).
    pub fn new_at_rate(fc_hz: f64, effective_rate_hz: f64) -> Self {
        let inv_rate = newton6_reciprocal_f64(effective_rate_hz.max(1.0));
        let angle = (core::f64::consts::PI * fc_hz * inv_rate).clamp(1.0e-5, 1.4);
        let u2 = angle * angle;
        let sin_a = angle
            * (1.0
                - u2 * (INV_6_F64
                    - u2 * (INV_120_F64 - u2 * (INV_5040_F64 - u2 * INV_362880_F64))));
        let cos_a = 1.0 - u2 * (0.5 - u2 * (INV_24_F64 - u2 * (INV_720_F64 - u2 * INV_40320_F64)));
        let g = sin_a * newton6_reciprocal_f64(cos_a);
        let k = core::f64::consts::SQRT_2;
        let a1 = newton6_reciprocal_f64(1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        Self {
            a1: a1 as f32,
            a2: a2 as f32,
            a3: a3 as f32,
            k_damp: k as f32,
            lp_s1: [0.0; 2],
            lp_s2: [0.0; 2],
            hp_s1: [0.0; 2],
            hp_s2: [0.0; 2],
            sec2_ic1: [0.0; 4],
            sec2_ic2: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.lp_s1 = [0.0; 2];
        self.lp_s2 = [0.0; 2];
        self.hp_s1 = [0.0; 2];
        self.hp_s2 = [0.0; 2];
        self.sec2_ic1 = [0.0; 4];
        self.sec2_ic2 = [0.0; 4];
    }

    #[inline(always)]
    fn step_svf_section(
        a1: f32,
        a2: f32,
        a3: f32,
        k: f32,
        x: f32,
        s1: &mut f32,
        s2: &mut f32,
    ) -> (f32, f32) {
        let v3 = x - *s2;
        let v1 = a1 * (*s1) + a2 * v3;
        let v2 = *s2 + a2 * (*s1) + a3 * v3;
        let low = v2;
        let high = x - k * v1 - v2;
        *s1 = flush_recursive_state_f32(2.0 * v1 - *s1);
        *s2 = flush_recursive_state_f32(2.0 * v2 - *s2);
        (low, high)
    }

    /// Steps the `LR4` crossover by 1 sample without real-time division, returning
    /// `(low_clean_bypass, high_drive_path)` (`ref[dsp.router.lr4-crossover]`).
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let (l2_first, h2_first) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            x,
            &mut self.lp_s1[0],
            &mut self.lp_s2[0],
        );
        self.hp_s1[0] = self.lp_s1[0];
        self.hp_s2[0] = self.lp_s2[0];
        let (l4, _) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            l2_first,
            &mut self.lp_s1[1],
            &mut self.lp_s2[1],
        );
        let (_, h4) = Self::step_svf_section(
            self.a1,
            self.a2,
            self.a3,
            self.k_damp,
            h2_first,
            &mut self.hp_s1[1],
            &mut self.hp_s2[1],
        );
        (l4, h4)
    }

    /// Steps a stereo pair (`self` = Left, `right` = Right) of `LR4` crossovers simultaneously,
    /// returning `(low_l, high_l, low_r, high_r)` (`ref[dsp.router.lr4-crossover]`).
    #[inline(always)]
    pub fn step_stereo(&mut self, right: &mut Self, x_l: f32, x_r: f32) -> (f32, f32, f32, f32) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let (a1, a2, a3, k) = (self.a1, self.a2, self.a3, self.k_damp);
        let (l2_l, h2_l) =
            Self::step_svf_section(a1, a2, a3, k, x_l, &mut self.lp_s1[0], &mut self.lp_s2[0]);
        let (l2_r, h2_r) =
            Self::step_svf_section(a1, a2, a3, k, x_r, &mut right.lp_s1[0], &mut right.lp_s2[0]);

        let x_arr = [l2_l, h2_l, l2_r, h2_r];
        let x = V::load(&x_arr);
        let ic1 = V::load(&self.sec2_ic1);
        let ic2 = V::load(&self.sec2_ic2);
        let v3 = x.sub(ic2);
        let v1 = ic1.mul(V::splat(a1)).fma(v3, V::splat(a2));
        let v2 = ic2.fma(ic1, V::splat(a2)).fma(v3, V::splat(a3));
        let two = V::splat(2.0);
        flush_recursive_state_vec4(v1.mul(two).sub(ic1)).store(&mut self.sec2_ic1);
        flush_recursive_state_vec4(v2.mul(two).sub(ic2)).store(&mut self.sec2_ic2);

        let mut v1_a = [0.0f32; 4];
        let mut v2_a = [0.0f32; 4];
        v1.store(&mut v1_a);
        v2.store(&mut v2_a);
        (
            v2_a[0],
            h2_l - k * v1_a[1] - v2_a[1],
            v2_a[2],
            h2_r - k * v1_a[3] - v2_a[3],
        )
    }
}

/// Evaluates the programmable stereo-linked & aux-key sidechain detector levels `(e_det_l, e_det_r)`
/// with zero `-6.02 dB` hard-panned peak loss (`ref[dsp.router.stereo-link-detector]`).
#[inline(always)]
pub fn eval_stereo_linked_aux_detector(
    x_l: f32,
    x_r: f32,
    x_aux: f32,
    alpha_link: f32,
    beta_key: f32,
) -> (f32, f32) {
    let a = alpha_link.clamp(0.0, 1.0);
    let b = beta_key.clamp(0.0, 1.0);
    let abs_l = x_l.abs();
    let abs_r = x_r.abs();
    let e_link = abs_l.max(abs_r);
    let e_key = x_aux.abs();
    let one_minus_b = 1.0 - b;
    let one_minus_a = 1.0 - a;
    let det_l = one_minus_b * (one_minus_a * abs_l + a * e_link) + b * e_key;
    let det_r = one_minus_b * (one_minus_a * abs_r + a * e_link) + b * e_key;
    (det_l, det_r)
}

/// Canonical 4-band asymmetric ballistic envelope time constants `(tau_att_sec, tau_rel_sec)`
/// (`Low`, `Mid`, `High`, `Aux Key`, `ref[dsp.router.ballistic-envelope-bank]`).
pub const BALLISTIC_ENVELOPE_BANK_TAUS_SEC: [(f64, f64); 4] = [
    (0.0100, 0.1500),
    (0.0020, 0.0400),
    (0.0002, 0.0150),
    (0.0005, 0.0500),
];

/// Prepares cold-path 1-pole step coefficients `(alpha_att, alpha_rel)` for band `band_idx` (`0..4`)
/// at `rate_hz` (`ref[dsp.router.ballistic-envelope-bank]`, `ref[dsp.zoh.router-detector]`).
pub fn prepare_ballistic_band_alphas(band_idx: usize, rate_hz: f64) -> Option<(f32, f32)> {
    let &(tau_att, tau_rel) = BALLISTIC_ENVELOPE_BANK_TAUS_SEC.get(band_idx)?;
    if rate_hz <= 0.0 || !rate_hz.is_finite() {
        return None;
    }
    let a_att = (1.0 - cold_exp_f64(-newton6_reciprocal_f64(rate_hz * tau_att))) as f32;
    let a_rel = (1.0 - cold_exp_f64(-newton6_reciprocal_f64(rate_hz * tau_rel))) as f32;
    Some((a_att, a_rel))
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::graph::IoRouter;
    use proptest::prelude::*;

    /// verify[dsp.router.signal-flow]
    /// verify[dsp.router.ports]
    /// verify[dsp.router.wdw-contract]
    /// verify[dsp.router.return-to-send-isolation]
    /// verify[dsp.router.parallel-alignment]
    /// verify[dsp.router.lr4-crossover]
    /// verify[dsp.router.mid-side-input]
    /// verify[dsp.router.stereo-link-detector]
    /// verify[dsp.router.ballistic-envelope-bank]
    /// verify[dsp.router.modulation-bus]
    #[test]
    fn test_stage0_router_wdw_isolation_alignment_lr4_ms_and_detector() {
        let _guard = FtzDazGuard::new();

        let (dry_core, wet_l, wet_r) =
            eval_wdw_routing_outputs(WdwDryTapMode::PreSpatialCore, 0.4, 0.2, 0.1, 1.5, 0.8, 0.2);
        assert!((dry_core - 0.4).abs() < 1e-6);
        assert!((wet_l - 0.35).abs() < 1e-6);
        assert!((wet_r - 0.05).abs() < 1e-6);

        let (dry_dir, _, _) =
            eval_wdw_routing_outputs(WdwDryTapMode::DirectDry, 0.4, 0.2, 0.1, 1.5, 0.8, 0.2);
        assert!((dry_dir - 0.5).abs() < 1e-6);

        let mut router = IoRouter::new();
        router.set_scattering_matrix([[1.0; 4]; 4]);
        assert_eq!(router.s_matrix[0][2], 0.0);
        assert_eq!(router.s_matrix[0][3], 0.0);
        assert_eq!(router.s_matrix[1][2], 0.0);
        assert_eq!(router.s_matrix[1][3], 0.0);
        let (s_l1, s_r1, _, _, _, _) = router.route_frame(0.5, -0.25, 0.0, 0.0);
        let (s_l2, s_r2, _, _, _, _) = router.route_frame(0.5, -0.25, 10.0, -10.0);
        assert_eq!((s_l1, s_r1), (s_l2, s_r2));

        let (mid, side) = IoRouter::voltage_normalize_ms(0.75, 0.75);
        assert_eq!((mid, side), (0.75, 0.0));
        assert_eq!(IoRouter::voltage_reconstruct_ms(mid, side), (0.75, 0.75));

        assert_eq!(
            ParallelAlignmentMode::FullBand64Tap.centering_offset_host_samples(),
            31
        );
        assert_eq!(
            ParallelAlignmentMode::LowLatency16Tap.centering_offset_host_samples(),
            7
        );
        assert_eq!(
            ParallelAlignmentMode::MinLatency4Tap.centering_offset_host_samples(),
            1
        );
        let (b_4x, d_eff_4x, pdc_4x) =
            ParallelAlignmentMode::FullBand64Tap.timing_at_oversampling(4, 5.5);
        assert_eq!(b_4x, 93);
        assert!((d_eff_4x - 129.5).abs() < 1e-12);
        assert_eq!(pdc_4x, 43);

        for (k, beta) in [(64usize, 8.6_f64), (16, 6.5)] {
            let w = prepare_kaiser_sinc_alignment_weights(k, beta, 0.35).unwrap();
            assert_eq!(w.len(), k);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14);
        }
        assert!((eval_min_latency_4tap_alignment(1.0, 1.0, 1.0, 1.0, 0.4) - 1.0).abs() < 1e-6);

        let mut lr4 = TptLr4Crossover::new_at_rate(200.0, 48_000.0);
        let mut low_dc = 0.0f32;
        let mut high_dc = 0.0f32;
        for _ in 0..4_000 {
            let (l, h) = lr4.step(1.0);
            low_dc = l;
            high_dc = h;
        }
        assert!((low_dc - 1.0).abs() < 1e-3);
        assert!(high_dc.abs() < 1e-3);

        assert_eq!(
            eval_stereo_linked_aux_detector(-0.75, 0.0, 0.0, 1.0, 0.0),
            (0.75, 0.75)
        );
        assert_eq!(
            eval_stereo_linked_aux_detector(0.75, -0.5, 0.25, 1.0, 0.5),
            (0.5, 0.5)
        );
        for b in 0..4 {
            let (a_att, a_rel) = prepare_ballistic_band_alphas(b, 48_000.0).unwrap();
            assert!(a_att > a_rel && a_rel > 0.0 && a_att < 1.0);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.router.stereo-link-detector]
        /// verify[dsp.tank.loop-saturator]
        /// verify[dsp.tank.quiet-feedback-knee]
        /// verify[dsp.ctrl.dispatch.modulation-laws]
        #[test]
        fn prop_topology_detector_stage4_saturator_knee_and_stage_laws(
            x_l in -10.0_f32..10.0_f32,
            x_r in -10.0_f32..10.0_f32,
            x_aux in -10.0_f32..10.0_f32,
            alpha_link in 0.0_f32..=1.0_f32,
            beta_key in 0.0_f32..=1.0_f32,
            u_in in -5.0_f32..5.0_f32,
            g_lin in 0.0_f32..=0.9990_f32,
            v_ceil in 0.1_f32..=0.985_f32,
            theta_val in -20.0_f32..20.0_f32,
        ) {
            let (e_l, e_r) = eval_stereo_linked_aux_detector(x_l, x_r, x_aux, alpha_link, beta_key);
            prop_assert!(e_l >= 0.0 && e_r >= 0.0);

            let knee = eval_tank_blended_feedback_knee(u_in, g_lin, 0.25, 0.5).unwrap();
            prop_assert!(knee.abs() <= g_lin * u_in.abs() + 1e-6);

            let sat = eval_tank_loop_saturator(u_in, v_ceil).unwrap();
            prop_assert!(sat.abs() <= v_ceil);
            prop_assert!(sat.abs().to_bits() <= TANK_MAX_V_CEIL_BITS);

            let laws = super::eval_stage_modulation_laws(
                &[theta_val; 16],
                [1.0, 1.0],
                64.0,
                2048.0,
                149.0,
                4096.0,
                0.85,
                0.5,
                1.0,
            );
            prop_assert!((0.0..=2.0).contains(&laws[0]));
            prop_assert!((4.0..=2048.0).contains(&laws[2]));
            prop_assert!((-24.0..=24.0).contains(&laws[3]));
            prop_assert!((0.0..=1.0).contains(&laws[10]));
            prop_assert!(laws[13].to_bits() <= TANK_MAX_G_LIN_BITS);
            prop_assert!((-0.75..=0.75).contains(&laws[14]));
            prop_assert!((0.0..=2.0).contains(&laws[15]));
        }
    }
}
