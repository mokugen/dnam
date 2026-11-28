//! Master-mode half-band FIR filter taps (`HB63` Kaiser, `HB23` minimax Remez) and
//! compile-time frequency-response verification (`ref[arch.bounce.oversampling]`,
//! `ref[repo.code.constant-eval-and-derivations]`).
//!
//! # Tap Normalization Convention
//! - `HB*_ODD_TAPS_OUTER` stores `2 * h[2k + 1]` in **outer-to-inner** order (`m = R, R-2, ..., 1`)
//!   for 2x zero-insertion upsampling (`0.5` one-sided sum, `1.0` DC gain with center tap `1.0`
//!   scaled from `0.5 * 2 = 1.0`, matching [`super::HB15_ODD_TAPS_OUTER`]).
//! - `HB*_DN_ODD_TAPS_OUTER` stores `h[2k + 1] = 0.5 * HB*_ODD_TAPS_OUTER` (`0.25` one-sided sum,
//!   `1.0` DC gain for `0.5 * even + sum(dn_taps * (x_neg + x_pos))` downsampling, matching
//!   [`super::HB15_DN_ODD_TAPS_OUTER`]).

use core::f32::consts::{FRAC_PI_2, PI, TAU};

/// Half-order radius (`R = (63 - 1) / 2`) for the `N = 63` Kaiser half-band FIR (`1x <-> 2x`).
pub const HB63_HALF_ORDER: usize = 31;

/// Number of symmetric outer-to-inner odd tap pairs in `HB63` (`16` taps, `R = 31`).
pub const HB63_ODD_TAP_COUNT: usize = 16;

/// Kaiser shape parameter `beta = 10.2` for `N = 63` half-band (`>= 100 dB` stopband at `>= 30 kHz @ 96 kHz`).
pub const HB63_KAISER_BETA: f32 = 10.2;

/// Half-order radius (`R = (23 - 1) / 2`) for the `N = 23` equiripple half-band FIR (`2x <-> 4x`).
pub const HB23_HALF_ORDER: usize = 11;

/// Number of symmetric outer-to-inner odd tap pairs in `HB23` (`6` taps, `R = 11`).
pub const HB23_ODD_TAP_COUNT: usize = 6;

/// Compile-time `f32` square root via Newton-Raphson (only invoked in `const` evaluation).
const fn const_sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = if x < 1.0 { 0.5 * (1.0 + x) } else { x };
    let mut iter = 0_usize;
    while iter < 16 {
        y = 0.5 * (y + x / y);
        iter += 1;
    }
    y
}

/// Compile-time modified Bessel function of the first kind `I_0(x)` for `|x| <= 12.0`.
///
/// Uses the power series `I_0(x) = sum_{k=0}^{31} ((x/2)^2)^k / (k!)^2`; at `x = 10.2`,
/// `(x/2)^2 = 26.01` and the `k = 31` term is `< 1e-24` (far below `f32` precision).
const fn const_bessel_i0_f32(x: f32) -> f32 {
    let half_x_sq = (0.5 * x) * (0.5 * x);
    let mut sum = 1.0_f32;
    let mut term = 1.0_f32;
    let mut k = 1_u32;
    while k <= 32 {
        let kf = k as f32;
        term *= half_x_sq / (kf * kf);
        sum += term;
        k += 1;
    }
    sum
}

/// Compile-time `cos(x)` for `const fn` zero-phase half-band verification.
const fn const_cos_f32(mut x: f32) -> f32 {
    if x < 0.0 {
        x = -x;
    }
    while x > TAU {
        x -= TAU;
    }
    if x > PI {
        x = TAU - x;
    }
    let sign = if x > FRAC_PI_2 {
        x = PI - x;
        -1.0_f32
    } else {
        1.0_f32
    };
    let x2 = x * x;
    let mut term = 1.0_f32;
    let mut sum = 1.0_f32;
    let mut n = 1_u32;
    while n <= 10 {
        let two_n = (2 * n) as f32;
        term *= -x2 / ((two_n - 1.0) * two_n);
        sum += term;
        n += 1;
    }
    sign * sum
}

/// Derive exact `HB63` Kaiser (`N = 63, beta = 10.2, R = 31`) outer-to-inner odd upsampling taps.
///
/// For `m = 31 - 2 * i` (`m in {31, 29, ..., 1}`):
/// `w[m] = I_0(beta * sqrt(1 - (m / 31)^2)) / I_0(beta)` and
/// `h_ideal[m] = 2 * sin(pi * m / 2) / (pi * m)` where `sin(pi * m / 2) = +1` for `m % 4 == 1`, `-1` for `m % 4 == 3`.
/// Normalizes the 16 taps so their one-sided sum equals `0.5` (`1.0` DC gain).
pub const fn const_hb63_odd_taps_outer() -> [f32; HB63_ODD_TAP_COUNT] {
    let inv_i0_beta = 1.0 / const_bessel_i0_f32(HB63_KAISER_BETA);
    let inv_r = 1.0 / (HB63_HALF_ORDER as f32);
    let mut raw = [0.0_f32; HB63_ODD_TAP_COUNT];
    let mut raw_sum = 0.0_f32;
    let mut i = 0_usize;
    while i < HB63_ODD_TAP_COUNT {
        let m = (HB63_HALF_ORDER - 2 * i) as u32;
        let mf = m as f32;
        let ratio = mf * inv_r;
        let arg_sq = 1.0 - ratio * ratio;
        let win = if arg_sq <= 0.0 {
            inv_i0_beta
        } else {
            const_bessel_i0_f32(HB63_KAISER_BETA * const_sqrt_f32(arg_sq)) * inv_i0_beta
        };
        let sign = if (m & 2) == 0 { 1.0_f32 } else { -1.0_f32 };
        let tap = (2.0 * sign / (PI * mf)) * win;
        raw[i] = tap;
        raw_sum += tap;
        i += 1;
    }
    let scale = 0.5_f32 / raw_sum;
    let mut norm = [0.0_f32; HB63_ODD_TAP_COUNT];
    let mut j = 0_usize;
    while j < HB63_ODD_TAP_COUNT {
        norm[j] = raw[j] * scale;
        j += 1;
    }
    norm
}

/// `N = 63` Kaiser (`beta = 10.2, R = 31`) half-band outer-to-inner odd upsampling taps (`sum == 0.5`).
pub const HB63_ODD_TAPS_OUTER: [f32; HB63_ODD_TAP_COUNT] = const_hb63_odd_taps_outer();

const fn const_halve_taps_16(src: [f32; HB63_ODD_TAP_COUNT]) -> [f32; HB63_ODD_TAP_COUNT] {
    let mut out = [0.0_f32; HB63_ODD_TAP_COUNT];
    let mut i = 0_usize;
    while i < HB63_ODD_TAP_COUNT {
        out[i] = 0.5 * src[i];
        i += 1;
    }
    out
}

/// `N = 63` Kaiser half-band outer-to-inner odd downsampling taps (`0.5 * HB63_ODD_TAPS_OUTER`, `sum == 0.25`).
pub const HB63_DN_ODD_TAPS_OUTER: [f32; HB63_ODD_TAP_COUNT] =
    const_halve_taps_16(HB63_ODD_TAPS_OUTER);

/// Constrained equiripple Parks-McClellan / Remez `N = 23` half-band upsampling taps (`f_p = 0.125`,
/// stopband `[0.375, 0.5]`, `R = 11`, outer-to-inner `m = 11, 9, 7, 5, 3, 1`, `sum == 0.5`).
///
/// Derivation (`ref[arch.bounce.oversampling]`, `ref[repo.code.constant-eval-and-derivations]`):
/// - Chebyshev alternation over the 5 independent odd taps `(h_1, h_3, h_5, h_7, h_9)` with
///   exact half-band DC constraint `h_11 = 0.5 - (h_1 + h_3 + h_5 + h_7 + h_9)` (`H(0) = 1.0`,
///   `H(f) + H(0.5 - f) = 1.0` identically across `[0, 0.5]`).
/// - Solved offline via Parks-McClellan minimax Chebyshev optimization over `f in [0.375, 0.5]`
///   (`72..96 kHz` at `192 kHz`, equivalently `[0, 0.125]` passband `0..24 kHz` at `192 kHz`),
///   yielding `f32` stopband attenuation `~104.0 dB` (`>= 98 dB`) and passband ripple
///   `~0.000055 dB` (`<= 0.001 dB`).
pub const HB23_ODD_TAPS_OUTER: [f32; HB23_ODD_TAP_COUNT] = {
    let h_1 = 0.618_760_5_f32;
    let h_3 = -0.163_771_62_f32;
    let h_5 = 0.060_896_94_f32;
    let h_7 = -0.020_168_238_f32;
    let h_9 = 0.004_956_699_f32;
    let h_11 = 0.5_f32 - (h_1 + h_3 + h_5 + h_7 + h_9);
    [h_11, h_9, h_7, h_5, h_3, h_1]
};

const fn const_halve_taps_6(src: [f32; HB23_ODD_TAP_COUNT]) -> [f32; HB23_ODD_TAP_COUNT] {
    let mut out = [0.0_f32; HB23_ODD_TAP_COUNT];
    let mut i = 0_usize;
    while i < HB23_ODD_TAP_COUNT {
        out[i] = 0.5 * src[i];
        i += 1;
    }
    out
}

/// `N = 23` minimax half-band outer-to-inner odd downsampling taps (`0.5 * HB23_ODD_TAPS_OUTER`, `sum == 0.25`).
pub const HB23_DN_ODD_TAPS_OUTER: [f32; HB23_ODD_TAP_COUNT] =
    const_halve_taps_6(HB23_ODD_TAPS_OUTER);

/// Zero-phase frequency response `H(f_norm)` of `HB63_ODD_TAPS_OUTER` for normalized frequency
/// `f_norm = f / f_s_out in [0.0, 0.5]`.
pub const fn const_hb63_zero_phase_response_f32(f_norm: f32) -> f32 {
    let w = TAU * f_norm;
    let mut h = 0.5_f32;
    let mut i = 0_usize;
    while i < HB63_ODD_TAP_COUNT {
        let m = (HB63_HALF_ORDER - 2 * i) as f32;
        h += HB63_ODD_TAPS_OUTER[i] * const_cos_f32(w * m);
        i += 1;
    }
    h
}

/// Zero-phase frequency response `H(f_norm)` of `HB23_ODD_TAPS_OUTER` for normalized frequency
/// `f_norm = f / f_s_out in [0.0, 0.5]`.
pub const fn const_hb23_zero_phase_response_f32(f_norm: f32) -> f32 {
    let w = TAU * f_norm;
    let mut h = 0.5_f32;
    let mut i = 0_usize;
    while i < HB23_ODD_TAP_COUNT {
        let m = (HB23_HALF_ORDER - 2 * i) as f32;
        h += HB23_ODD_TAPS_OUTER[i] * const_cos_f32(w * m);
        i += 1;
    }
    h
}

const fn const_abs_local(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

const fn verify_hb63_bounds() -> bool {
    let mut sum_up = 0.0_f32;
    let mut sum_dn = 0.0_f32;
    let mut i = 0_usize;
    while i < HB63_ODD_TAP_COUNT {
        sum_up += HB63_ODD_TAPS_OUTER[i];
        sum_dn += HB63_DN_ODD_TAPS_OUTER[i];
        i += 1;
    }
    if const_abs_local(sum_up - 0.5) > 1.0e-6 {
        return false;
    }
    if const_abs_local(sum_dn - 0.25) > 1.0e-6 {
        return false;
    }
    if const_abs_local(const_hb63_zero_phase_response_f32(0.0) - 1.0) > 1.0e-6 {
        return false;
    }
    // Passband <= 20 kHz @ 96 kHz => f_norm <= 5/24; linear ripple <= 0.025 dB (10^(0.025/20)-1 = 0.002882).
    let mut k = 0_usize;
    while k <= 64 {
        let f_pb = (5.0 / 24.0) * (k as f32) / 64.0;
        if const_abs_local(const_hb63_zero_phase_response_f32(f_pb) - 1.0) > 0.0028 {
            return false;
        }
        // Stopband >= 30 kHz @ 96 kHz => f_norm in [0.3125, 0.5]; >= 100 dB => |H| <= 1e-5.
        let f_sb = 0.3125 + (0.5 - 0.3125) * (k as f32) / 64.0;
        if const_abs_local(const_hb63_zero_phase_response_f32(f_sb)) > 1.0e-5 {
            return false;
        }
        k += 1;
    }
    true
}

const fn verify_hb23_bounds() -> bool {
    let mut sum_up = 0.0_f32;
    let mut sum_dn = 0.0_f32;
    let mut i = 0_usize;
    while i < HB23_ODD_TAP_COUNT {
        sum_up += HB23_ODD_TAPS_OUTER[i];
        sum_dn += HB23_DN_ODD_TAPS_OUTER[i];
        i += 1;
    }
    if const_abs_local(sum_up - 0.5) > 1.0e-6 {
        return false;
    }
    if const_abs_local(sum_dn - 0.25) > 1.0e-6 {
        return false;
    }
    if const_abs_local(const_hb23_zero_phase_response_f32(0.0) - 1.0) > 1.0e-6 {
        return false;
    }
    // Passband <= 24 kHz @ 192 kHz => f_norm <= 0.125; ripple <= 0.001 dB (10^(0.001/20)-1 = 1.151e-4).
    let mut k = 0_usize;
    while k <= 64 {
        let f_pb = 0.125 * (k as f32) / 64.0;
        if const_abs_local(const_hb23_zero_phase_response_f32(f_pb) - 1.0) > 1.0e-4 {
            return false;
        }
        // Stopband >= 72 kHz @ 192 kHz => f_norm in [0.375, 0.5]; >= 98 dB => |H| <= 1.2589e-5.
        let f_sb = 0.375 + (0.5 - 0.375) * (k as f32) / 64.0;
        if const_abs_local(const_hb23_zero_phase_response_f32(f_sb)) > 1.25e-5 {
            return false;
        }
        k += 1;
    }
    true
}

const _: () = assert!(verify_hb63_bounds());
const _: () = assert!(verify_hb23_bounds());

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn hb63_and_hb23_meet_passband_ripple_and_stopband_db_specs() {
        let mut max_pb_err_63 = 0.0_f64;
        let mut max_sb_mag_63 = 0.0_f64;
        for step in 0..=1000 {
            let t = f64::from(step) / 1000.0;
            let f_pb = (5.0 / 24.0) * t;
            let h_pb = f64::from(const_hb63_zero_phase_response_f32(f_pb as f32));
            max_pb_err_63 = max_pb_err_63.max((h_pb - 1.0).abs());
            let f_sb = 0.3125 + (0.5 - 0.3125) * t;
            let h_sb = f64::from(const_hb63_zero_phase_response_f32(f_sb as f32)).abs();
            max_sb_mag_63 = max_sb_mag_63.max(h_sb);
        }
        let pb_db_63 = 20.0 * (1.0 + max_pb_err_63).log10();
        let sb_db_63 = -20.0 * max_sb_mag_63.log10();
        assert!(
            pb_db_63 <= 0.025,
            "HB63 passband ripple {pb_db_63:.6} dB > 0.025 dB"
        );
        assert!(sb_db_63 >= 100.0, "HB63 stopband {sb_db_63:.3} dB < 100 dB");

        let mut max_pb_err_23 = 0.0_f64;
        let mut max_sb_mag_23 = 0.0_f64;
        for step in 0..=1000 {
            let t = f64::from(step) / 1000.0;
            let f_pb = 0.125 * t;
            let h_pb = f64::from(const_hb23_zero_phase_response_f32(f_pb as f32));
            max_pb_err_23 = max_pb_err_23.max((h_pb - 1.0).abs());
            let f_sb = 0.375 + (0.5 - 0.375) * t;
            let h_sb = f64::from(const_hb23_zero_phase_response_f32(f_sb as f32)).abs();
            max_sb_mag_23 = max_sb_mag_23.max(h_sb);
        }
        let pb_db_23 = 20.0 * (1.0 + max_pb_err_23).log10();
        let sb_db_23 = -20.0 * max_sb_mag_23.log10();
        assert!(
            pb_db_23 <= 0.001,
            "HB23 passband ripple {pb_db_23:.6} dB > 0.001 dB"
        );
        assert!(sb_db_23 >= 98.0, "HB23 stopband {sb_db_23:.3} dB < 98 dB");
    }

    proptest! {
        #[test]
        fn hb63_and_hb23_halfband_symmetry_holds_f32(f_norm in 0.0_f32..=0.5_f32) {
            let sum63 = const_hb63_zero_phase_response_f32(f_norm)
                + const_hb63_zero_phase_response_f32(0.5 - f_norm);
            prop_assert!((sum63 - 1.0).abs() < 2.0e-6);

            let sum23 = const_hb23_zero_phase_response_f32(f_norm)
                + const_hb23_zero_phase_response_f32(0.5 - f_norm);
            prop_assert!((sum23 - 1.0).abs() < 2.0e-6);
        }
    }
}
