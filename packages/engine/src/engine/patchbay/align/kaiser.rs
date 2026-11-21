//! Compile-time Kaiser-derived windowed-sinc alignment weights (`0 FDIV`, `0 FSQRT`, `0 libm`)
//! and alignment mode definitions (`ref[dsp.router.parallel-alignment]`).

#![forbid(unsafe_code)]

use dnam_kernels::catmull_rom_hermite_f32;
use dnam_vm::OFFLINE_BOUNCE_PDC_SAMPLES;
use dnam_vm::math::{
    F64_INT_EXACT_THRESHOLD, IEEE754_F64_EXP_ALL_ONES, IEEE754_F64_EXP_BIAS,
    IEEE754_F64_MANTISSA_BITS, IEEE754_F64_MAX_NORMAL_BIASED_EXP, IEEE754_F64_RSQRT_SEED,
    IEEE754_F64_SIGN_MASK, const_abs_f64, const_f64_pow2, const_sin_cos_f64,
};

/// Full-band Kaiser-sinc FIR tap count (`K = 64`), shape parameter (`beta = 8.6`), and host center (`G = 31`).
pub const KAISER_FULL_BAND_TAPS: usize = 64;
/// Full-band Kaiser window shape parameter (`beta = 8.6`).
pub const KAISER_FULL_BAND_BETA: f64 = 8.6;
/// Full-band host-rate centering offset (`G = 31` samples).
pub const KAISER_FULL_BAND_CENTER_G: usize = (KAISER_FULL_BAND_TAPS - 2) / 2;

/// Low-latency Kaiser-sinc FIR tap count (`K = 16`), shape parameter (`beta = 6.5`), and host center (`G = 7`).
pub const KAISER_LOW_LATENCY_TAPS: usize = 16;
/// Low-latency Kaiser window shape parameter (`beta = 6.5`).
pub const KAISER_LOW_LATENCY_BETA: f64 = 6.5;
/// Low-latency host-rate centering offset (`G = 7` samples).
pub const KAISER_LOW_LATENCY_CENTER_G: usize = (KAISER_LOW_LATENCY_TAPS - 2) / 2;

/// Minimum-latency 4-tap Catmull-Rom centering offset (`G = 1` host sample).
pub const MIN_LATENCY_CENTER_G: usize = 1;

const KAISER_BESSEL_TERMS: usize = 28;
const INV_2_F64: f64 = 0.5;

const fn build_inv_m_sq_table() -> [f64; KAISER_BESSEL_TERMS + 1] {
    let mut out = [0.0_f64; KAISER_BESSEL_TERMS + 1];
    let mut m = 1_usize;
    while m <= KAISER_BESSEL_TERMS {
        out[m] = 1.0 / ((m * m) as f64);
        m += 1;
    }
    out
}

const INV_M_SQ: [f64; KAISER_BESSEL_TERMS + 1] = build_inv_m_sq_table();

#[inline]
const fn const_floor_f64(x: f64) -> f64 {
    if x.is_nan() || const_abs_f64(x) >= F64_INT_EXACT_THRESHOLD {
        return x;
    }
    let i = x as i64;
    let f = i as f64;
    if x < f { f - 1.0 } else { f }
}

#[inline]
const fn const_round_f64(x: f64) -> f64 {
    if x >= 0.0 {
        const_floor_f64(x + 0.5)
    } else {
        -const_floor_f64(-x + 0.5)
    }
}

#[inline]
const fn const_rcp_nr6_f64(d: f64) -> f64 {
    let bits = d.to_bits();
    let sign = bits & IEEE754_F64_SIGN_MASK;
    let exp = (((bits >> IEEE754_F64_MANTISSA_BITS) & (IEEE754_F64_EXP_ALL_ONES as u64)) as i32)
        - IEEE754_F64_EXP_BIAS;
    let mut inv_exp = IEEE754_F64_EXP_BIAS - 1 - exp;
    if inv_exp < 1 {
        inv_exp = 1;
    } else if inv_exp > IEEE754_F64_MAX_NORMAL_BIASED_EXP {
        inv_exp = IEEE754_F64_MAX_NORMAL_BIASED_EXP;
    }
    let mut inv = f64::from_bits(sign | ((inv_exp as u64) << IEEE754_F64_MANTISSA_BITS));
    let mut i = 0;
    while i < 6 {
        inv *= 2.0 - d * inv;
        i += 1;
    }
    inv
}

#[inline]
const fn const_sqrt_nr_f64(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let (scaled_x, scale_out) = if x < f64::MIN_POSITIVE {
        (x * const_f64_pow2(54), const_f64_pow2(-27))
    } else {
        (x, 1.0)
    };
    let rsqrt_bits = IEEE754_F64_RSQRT_SEED.wrapping_sub(scaled_x.to_bits() >> 1);
    let mut z = f64::from_bits(rsqrt_bits);
    let half_x = 0.5 * scaled_x;
    let mut i = 0;
    while i < 6 {
        z *= 1.5 - half_x * z * z;
        i += 1;
    }
    let y = scaled_x * z;
    (y + 0.5 * (scaled_x - y * y) * z) * scale_out
}

#[inline]
const fn const_bessel_i0_f64(z: f64) -> f64 {
    let half = 0.5 * z;
    let half2 = half * half;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut m = 1_usize;
    while m <= KAISER_BESSEL_TERMS {
        term *= half2 * INV_M_SQ[m];
        sum += term;
        m += 1;
    }
    sum
}

#[inline]
const fn const_normalized_sinc_f64(x: f64) -> f64 {
    if const_abs_f64(x) < 1.0e-12 {
        return 1.0;
    }
    let pix = core::f64::consts::PI * x;
    let k = const_round_f64(x) as i64;
    let rem = x - (k as f64);
    let sign = if (k & 1) != 0 { -1.0 } else { 1.0 };
    let u = core::f64::consts::PI * rem;
    let (sin_u, _) = const_sin_cos_f64(u);
    let inv_pix = if pix >= 0.0 {
        const_rcp_nr6_f64(pix)
    } else {
        -const_rcp_nr6_f64(-pix)
    };
    (sign * sin_u) * inv_pix
}

const fn const_eval_kaiser_sinc_weights<const K: usize>(beta: f64, mu: f64) -> [f64; K] {
    let g = ((K - 2) / 2) as f64;
    let inv_r = 2.0 * const_rcp_nr6_f64(K as f64);
    let raw_denom = const_bessel_i0_f64(beta) - 1.0;
    let denom = if raw_denom < 1.0e-15 {
        1.0e-15
    } else {
        raw_denom
    };
    let inv_denom = const_rcp_nr6_f64(denom);

    let mut weights = [0.0_f64; K];
    let mut sum = 0.0_f64;
    let mut k = 0_usize;
    while k < K {
        let s = (k as f64) - g - mu;
        let ratio = s * inv_r;
        let w = if const_abs_f64(ratio) >= 1.0 {
            0.0
        } else {
            let under = 1.0 - ratio * ratio;
            let arg = beta * const_sqrt_nr_f64(if under < 0.0 { 0.0 } else { under });
            let raw_w = (const_bessel_i0_f64(arg) - 1.0) * inv_denom;
            if raw_w < 0.0 { 0.0 } else { raw_w }
        };
        let h = const_normalized_sinc_f64(s) * w;
        weights[k] = h;
        sum += h;
        k += 1;
    }

    let mag = const_abs_f64(sum);
    let safe_mag = if mag < 1.0e-15 { 1.0e-15 } else { mag };
    let inv_sum = if sum >= 0.0 {
        const_rcp_nr6_f64(safe_mag)
    } else {
        -const_rcp_nr6_f64(safe_mag)
    };
    k = 0;
    while k < K {
        weights[k] *= inv_sum;
        k += 1;
    }
    weights
}

const fn to_f32_array<const K: usize>(src: [f64; K]) -> [f32; K] {
    let mut out = [0.0_f32; K];
    let mut i = 0;
    while i < K {
        out[i] = src[i] as f32;
        i += 1;
    }
    out
}

const fn const_sum_f64(slice: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    let mut i = 0;
    while i < slice.len() {
        acc += slice[i];
        i += 1;
    }
    acc
}

/// Compile-time zero-offset (`mu = 0.0`) 64-tap Kaiser-sinc weights (`KAISER_FULL_BAND_CENTER_G == 31`).
pub const KAISER_64_ZERO_OFFSET_TAPS: [f32; KAISER_FULL_BAND_TAPS] = to_f32_array(
    const_eval_kaiser_sinc_weights::<KAISER_FULL_BAND_TAPS>(KAISER_FULL_BAND_BETA, 0.0),
);

/// Compile-time zero-offset (`mu = 0.0`) 16-tap Kaiser-sinc weights (`KAISER_LOW_LATENCY_CENTER_G == 7`).
pub const KAISER_16_ZERO_OFFSET_TAPS: [f32; KAISER_LOW_LATENCY_TAPS] = to_f32_array(
    const_eval_kaiser_sinc_weights::<KAISER_LOW_LATENCY_TAPS>(KAISER_LOW_LATENCY_BETA, 0.0),
);

const KAISER_64_HALF_F64: [f64; KAISER_FULL_BAND_TAPS] =
    const_eval_kaiser_sinc_weights::<KAISER_FULL_BAND_TAPS>(KAISER_FULL_BAND_BETA, INV_2_F64);
const KAISER_16_HALF_F64: [f64; KAISER_LOW_LATENCY_TAPS] =
    const_eval_kaiser_sinc_weights::<KAISER_LOW_LATENCY_TAPS>(KAISER_LOW_LATENCY_BETA, INV_2_F64);

/// Compile-time half-sample (`mu = 0.5`) 64-tap Kaiser-sinc weights (`beta = 8.6`).
pub const KAISER_64_HALF_SAMPLE_TAPS: [f32; KAISER_FULL_BAND_TAPS] =
    to_f32_array(KAISER_64_HALF_F64);

/// Compile-time half-sample (`mu = 0.5`) 16-tap Kaiser-sinc weights (`beta = 6.5`).
pub const KAISER_16_HALF_SAMPLE_TAPS: [f32; KAISER_LOW_LATENCY_TAPS] =
    to_f32_array(KAISER_16_HALF_F64);

const _: () = assert!(KAISER_FULL_BAND_CENTER_G == 31);
const _: () = assert!(KAISER_LOW_LATENCY_CENTER_G == 7);
const _: () = assert!(MIN_LATENCY_CENTER_G == 1);
const _: () = assert!(KAISER_64_ZERO_OFFSET_TAPS[KAISER_FULL_BAND_CENTER_G] == 1.0);
const _: () = assert!(KAISER_16_ZERO_OFFSET_TAPS[KAISER_LOW_LATENCY_CENTER_G] == 1.0);
const _: () = assert!(const_abs_f64(const_sum_f64(&KAISER_64_HALF_F64) - 1.0) < 1.0e-12);
const _: () = assert!(const_abs_f64(const_sum_f64(&KAISER_16_HALF_F64) - 1.0) < 1.0e-12);
const _: () = assert!(const_abs_f64(KAISER_64_HALF_F64[31] - KAISER_64_HALF_F64[32]) < 1.0e-14);
const _: () = assert!(const_abs_f64(KAISER_16_HALF_F64[7] - KAISER_16_HALF_F64[8]) < 1.0e-14);
// Endpoint sidelobes at s = ±31.5 are symmetric and nonzero negative values.
const _: () = assert!(KAISER_64_HALF_F64[0] < 0.0 && KAISER_64_HALF_F64[63] < 0.0);
const _: () = assert!(const_abs_f64(KAISER_64_HALF_F64[0] - KAISER_64_HALF_F64[63]) < 1.0e-14);

/// Parallel-path sub-sample alignment mode (`ref[dsp.router.parallel-alignment]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParallelAlignmentMode {
    /// 64-tap Kaiser-derived windowed sinc (`beta = 8.6`, `G = 31` host samples).
    #[default]
    FullBand64Tap,
    /// 16-tap Kaiser-derived windowed sinc (`beta = 6.5`, `G = 7` host samples).
    LowLatency16Tap,
    /// Centered 4-tap Catmull-Rom cubic Hermite (`G = 1` host sample).
    MinLatency4Tap,
}

/// Alias matching host-shell patchbay alignment naming.
pub type IoAlignmentMode = ParallelAlignmentMode;

const _: () = assert!(ParallelAlignmentMode::FullBand64Tap.centering_offset_host_samples() == 31);
const _: () = assert!(ParallelAlignmentMode::LowLatency16Tap.centering_offset_host_samples() == 7);
const _: () = assert!(ParallelAlignmentMode::MinLatency4Tap.centering_offset_host_samples() == 1);
const _: () = assert!(
    ParallelAlignmentMode::FullBand64Tap
        .timing_at_oversampling(4, 0.0)
        .0
        == 93
);
const _: () = assert!(
    ParallelAlignmentMode::FullBand64Tap
        .timing_at_oversampling(4, 0.0)
        .2
        == 43
);
const _: () = assert!(ParallelAlignmentMode::FullBand64Tap.required_history_samples(4) == 157);

impl ParallelAlignmentMode {
    /// Returns the host-rate centering offset `G` in samples (`31`, `7`, or `1`).
    #[inline]
    #[must_use]
    pub const fn centering_offset_host_samples(self) -> usize {
        match self {
            Self::FullBand64Tap => KAISER_FULL_BAND_CENTER_G,
            Self::LowLatency16Tap => KAISER_LOW_LATENCY_CENTER_G,
            Self::MinLatency4Tap => MIN_LATENCY_CENTER_G,
        }
    }

    /// Returns the active interpolation tap count `K` (`64`, `16`, or `4`).
    #[inline]
    #[must_use]
    pub const fn tap_count(self) -> usize {
        match self {
            Self::FullBand64Tap => KAISER_FULL_BAND_TAPS,
            Self::LowLatency16Tap => KAISER_LOW_LATENCY_TAPS,
            Self::MinLatency4Tap => 4,
        }
    }

    /// Returns the Kaiser shape parameter `beta` (`8.6`, `6.5`, or `0.0` for Catmull-Rom).
    #[inline]
    #[must_use]
    pub const fn kaiser_beta(self) -> f64 {
        match self {
            Self::FullBand64Tap => KAISER_FULL_BAND_BETA,
            Self::LowLatency16Tap => KAISER_LOW_LATENCY_BETA,
            Self::MinLatency4Tap => 0.0,
        }
    }

    /// Returns `(base_offset_b, effective_delay_samples, combined_pdc_host_samples)` at oversampling
    /// factor `oversampling` (`1` or `4`, `ref[dsp.router.parallel-alignment]`,
    /// `ref[arch.bounce.oversampling]`).
    #[inline]
    #[must_use]
    pub const fn timing_at_oversampling(
        self,
        oversampling: usize,
        delta_samples_eff: f64,
    ) -> (usize, f64, usize) {
        let m = if oversampling == 0 { 1 } else { oversampling };
        let g = self.centering_offset_host_samples();
        let b = (m - 1) * g;
        let d_eff = (m * g) as f64 + delta_samples_eff;
        let pdc_host = if m == 4 {
            g + OFFLINE_BOUNCE_PDC_SAMPLES
        } else {
            g
        };
        (b, d_eff, pdc_host)
    }

    /// Returns minimum per-channel history samples required (`B + K`; `157` at `4x` `FullBand64Tap`).
    #[inline]
    #[must_use]
    pub const fn required_history_samples(self, oversampling: usize) -> usize {
        let m = if oversampling == 0 { 1 } else { oversampling };
        let g = self.centering_offset_host_samples();
        (m - 1) * g + self.tap_count()
    }

    /// Returns combined host-sample PDC (`g` live, `g + 12` under `4x` offline bounce).
    #[inline]
    #[must_use]
    pub const fn pdc_host_samples(self, offline_bounce: bool) -> usize {
        let m = if offline_bounce { 4 } else { 1 };
        self.timing_at_oversampling(m, 0.0).2
    }
}

/// Zero-allocation stack writer for normalized zero-ended Kaiser-derived sinc weights $h_k(\mu)$
/// (`num_taps in {16, 64}`, `beta > 0`, `0 <= mu < 1`, `ref[dsp.router.parallel-alignment]`).
///
/// Executes with zero `FDIV`, zero `FSQRT`, zero `IDIV`, and zero `libm` calls.
pub fn prepare_kaiser_sinc_alignment_weights_into(
    num_taps: usize,
    beta: f64,
    mu: f64,
    out: &mut [f64; 64],
) -> bool {
    if !matches!(num_taps, 16 | 64) || !(0.0..1.0).contains(&mu) || !beta.is_finite() || beta <= 0.0
    {
        return false;
    }
    match num_taps {
        KAISER_FULL_BAND_TAPS => {
            let weights = const_eval_kaiser_sinc_weights::<KAISER_FULL_BAND_TAPS>(beta, mu);
            *out = weights;
            true
        }
        KAISER_LOW_LATENCY_TAPS => {
            let weights = const_eval_kaiser_sinc_weights::<KAISER_LOW_LATENCY_TAPS>(beta, mu);
            out[..KAISER_LOW_LATENCY_TAPS].copy_from_slice(&weights);
            out[KAISER_LOW_LATENCY_TAPS..].fill(0.0);
            true
        }
        _ => false,
    }
}

/// Evaluates `MinLatency4Tap` Catmull-Rom alignment for `[s_{-1}, s_0, s_1, s_2]` (`G = 1`,
/// `ref[dsp.router.parallel-alignment]`) via [`dnam_kernels::catmull_rom_hermite_f32`] (`0 FDIV`).
#[inline(always)]
#[must_use]
pub fn eval_min_latency_4tap_alignment(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    catmull_rom_hermite_f32(s_m1, s_0, s_1, s_2, mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[dsp.router.parallel-alignment]
    #[test]
    fn test_kaiser_weights_modes_and_4x_bounce_timing() {
        assert_eq!(
            ParallelAlignmentMode::FullBand64Tap.centering_offset_host_samples(),
            31
        );
        assert_eq!(
            IoAlignmentMode::LowLatency16Tap.centering_offset_host_samples(),
            7
        );
        assert_eq!(
            ParallelAlignmentMode::MinLatency4Tap.centering_offset_host_samples(),
            1
        );
        assert_eq!(
            ParallelAlignmentMode::MinLatency4Tap.pdc_host_samples(false),
            1
        );
        assert_eq!(
            ParallelAlignmentMode::FullBand64Tap.pdc_host_samples(true),
            43
        );
        assert_eq!(ParallelAlignmentMode::MinLatency4Tap.kaiser_beta(), 0.0);

        let (b_4x, d_eff_4x, pdc_4x) =
            ParallelAlignmentMode::FullBand64Tap.timing_at_oversampling(4, 5.5);
        assert_eq!(b_4x, 93);
        assert!((d_eff_4x - 129.5).abs() < 1.0e-12);
        assert_eq!(pdc_4x, 43);
        assert_eq!(
            ParallelAlignmentMode::FullBand64Tap.required_history_samples(4),
            157
        );

        let mut buf = [0.0_f64; 64];
        assert!(!prepare_kaiser_sinc_alignment_weights_into(
            8, 8.6, 0.5, &mut buf
        ));
        assert!(!prepare_kaiser_sinc_alignment_weights_into(
            64, -1.0, 0.5, &mut buf
        ));
        assert!(!prepare_kaiser_sinc_alignment_weights_into(
            64, 8.6, 1.0, &mut buf
        ));

        for (taps, beta, half_const) in [
            (64_usize, 8.6_f64, &KAISER_64_HALF_SAMPLE_TAPS[..]),
            (16, 6.5, &KAISER_16_HALF_SAMPLE_TAPS[..]),
        ] {
            assert!(prepare_kaiser_sinc_alignment_weights_into(
                taps, beta, 0.5, &mut buf
            ));
            let sum: f64 = buf[..taps].iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-13);
            for (k, &w) in buf[..taps].iter().enumerate() {
                assert!((w as f32 - half_const[k]).abs() < 1.0e-7);
            }
        }
        assert!((eval_min_latency_4tap_alignment(1.0, 1.0, 1.0, 1.0, 0.4) - 1.0).abs() < 1.0e-6);
    }
}
