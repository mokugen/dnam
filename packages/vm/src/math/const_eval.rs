//! Compile-time mathematical evaluation, IEEE-754 seed derivations, and algebraic constant checks
//! (`ref[repo.code.constant-eval-and-derivations]`).

pub use dnam_kernels::simd::const_eval::*;

/// Robertson/Lomont minimax `binary64` global two-binade inverse-square-root seed
/// (`docs/specs/dsp/op_n_paralru_and_polyadaa.md` §1.2).
pub const IEEE754_F64_RSQRT_SEED: u64 = 0x5FE6_EB50_C7B5_37A9;

/// Compile-time `5 Hz` tank passband-normalized DC blocker pole at `48 kHz` (`0x3F7F_D51F`).
pub const TANK_DC_BLOCKER_48K_F32: f32 = const_dc_blocker_pole_f32(5.0, 48_000.0);

/// Compile-time largest positive `f32` bit pattern `<= 0.985` (`0x3F7C_28F5`).
pub const TANK_MAX_V_CEIL_BITS_DERIVED: u32 = const_max_f32_bits_le(0.985);

/// Compile-time largest positive `f32` bit pattern `<= 0.9990` (`0x3F7F_BE76`).
pub const TANK_MAX_G_LIN_BITS_DERIVED: u32 = const_max_f32_bits_le(0.9990);

/// Branch-free compile-time `|x|` for `f64`.
#[inline]
#[must_use]
pub const fn const_abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & (u64::MAX >> 1))
}

/// Compile-time `f64` square root via bit-seeded Newton–Raphson iteration (`< 1 ULP` for `x > 0`).
#[must_use]
pub const fn const_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let init_bits = (x.to_bits() >> 1) + (0x3FF0_0000_0000_0000_u64 >> 1);
    let mut y = f64::from_bits(init_bits);
    let mut i = 0;
    while i < 8 {
        y = 0.5 * (y + x / y);
        i += 1;
    }
    y
}

const fn build_inv_sin_steps() -> [f64; 11] {
    let mut out = [0.0_f64; 11];
    let mut n = 1usize;
    while n <= 10 {
        out[n] = 1.0 / ((2 * n * (2 * n + 1)) as f64);
        n += 1;
    }
    out
}

const fn build_inv_cos_steps() -> [f64; 11] {
    let mut out = [0.0_f64; 11];
    let mut m = 1usize;
    while m <= 10 {
        out[m] = 1.0 / (((2 * m - 1) * (2 * m)) as f64);
        m += 1;
    }
    out
}

const INV_SIN_STEP: [f64; 11] = build_inv_sin_steps();
const INV_COS_STEP: [f64; 11] = build_inv_cos_steps();

/// Compile-time `(sin(x), cos(x))` via quadrant reduction (`[-pi/4, pi/4]`) and degree-21/20
/// Maclaurin evaluation (`< 2e-16` error across standard ranges).
#[must_use]
pub const fn const_sin_cos_f64(x: f64) -> (f64, f64) {
    if !x.is_finite() || x > 1.0e15 || x < -1.0e15 {
        return (0.0, 1.0);
    }
    let k_real = x * core::f64::consts::FRAC_2_PI;
    let k = if k_real >= 0.0 {
        (k_real + 0.5) as i64
    } else {
        (k_real - 0.5) as i64
    };
    let r = x - (k as f64) * core::f64::consts::FRAC_PI_2;
    let r2 = r * r;

    let mut sin_term = r;
    let mut sin_sum = r;
    let mut n = 1usize;
    while n <= 10 {
        sin_term = -sin_term * r2 * INV_SIN_STEP[n];
        sin_sum += sin_term;
        n += 1;
    }

    let mut cos_term = 1.0_f64;
    let mut cos_sum = 1.0_f64;
    let mut m = 1usize;
    while m <= 10 {
        cos_term = -cos_term * r2 * INV_COS_STEP[m];
        cos_sum += cos_term;
        m += 1;
    }

    match k & 3 {
        0 => (sin_sum, cos_sum),
        1 => (cos_sum, -sin_sum),
        2 => (-sin_sum, -cos_sum),
        _ => (-cos_sum, sin_sum),
    }
}

/// Compile-time `sin(x)` in `f64`.
#[must_use]
pub const fn const_sin_f64(x: f64) -> f64 {
    const_sin_cos_f64(x).0
}

/// Compile-time `cos(x)` in `f64`.
#[must_use]
pub const fn const_cos_f64(x: f64) -> f64 {
    const_sin_cos_f64(x).1
}

/// Compile-time `tan(x)` in `f64`.
#[must_use]
pub const fn const_tan_f64(x: f64) -> f64 {
    let (s, c) = const_sin_cos_f64(x);
    s / c
}

/// Compile-time passband-normalized DC blocker pole `R = cos(w) / (1 + sin(w))` rounded to `f32`
/// (`ref[dsp.tank.dc-blocker]`).
#[must_use]
pub const fn const_dc_blocker_pole_f32(fc_hz: f64, fs_hz: f64) -> f32 {
    let omega = core::f64::consts::TAU * fc_hz / fs_hz;
    let (s, c) = const_sin_cos_f64(omega);
    (c / (1.0 + s)) as f32
}

/// Solves the exact closed-form $C^1$-clamped rational Padé-`[5/4]` $\tanh(x)$ linear system
/// at clamping radius `r` (`docs/specs/dsp/op_n_paralru_and_polyadaa.md` §1.2), returning `[n1, n2, d1, d2]`.
#[must_use]
pub const fn const_c1_clamped_tanh_coeffs(r: f64) -> [f64; 4] {
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;

    let a11 = 15.0 * r3 - 5.0 * r5 - 15.0 * r2;
    let a12 = 15.0 * (r5 - r4);
    let b1 = 15.0 - 15.0 * r + 5.0 * r3 - 2.0 * r5;

    let a21 = 9.0 * r2 - 5.0 * r4 - 6.0 * r;
    let a22 = 3.0 * (5.0 * r4 - 4.0 * r3);
    let b2 = 3.0 * r2 - 2.0 * r4 - 3.0;

    let det = a11 * a22 - a12 * a21;
    let num_d1 = b1 * a22 - b2 * a12;
    let num_d2 = a11 * b2 - a21 * b1;
    let d1 = num_d1 / det;
    let d2 = num_d2 / det;
    let n1 = (3.0 * num_d1 - det) / (3.0 * det);
    let n2 = (15.0 * num_d2 - 5.0 * num_d1 + 2.0 * det) / (15.0 * det);
    [n1, n2, d1, d2]
}

/// Compile-time coefficients `(c1, c3, c5)` of the degree-5 odd Hermite quarter-sine.
#[must_use]
pub const fn const_hermite5_quarter_sine_coeffs() -> (f32, f32, f32) {
    (
        core::f64::consts::FRAC_PI_2 as f32,
        (core::f64::consts::PI - 2.5) as f32,
        (core::f64::consts::FRAC_PI_2 - 1.5) as f32,
    )
}

/// Compile-time largest positive normal `f32` bit pattern strictly `<= bound`.
#[must_use]
pub const fn const_max_f32_bits_le(bound: f64) -> u32 {
    let candidate = bound as f32;
    let bits = candidate.to_bits();
    if (candidate as f64) > bound {
        bits - 1
    } else {
        bits
    }
}

const TANH_R4: [f64; 4] = const_c1_clamped_tanh_coeffs(4.0);
const _: () = assert!(IEEE754_F32_RCP_SEED == 0x7EF3_11C3);
const _: () = assert!(const_f32_rcp_chebyshev_seed() == IEEE754_F32_RCP_SEED);
const _: () = assert!(TANK_DC_BLOCKER_48K_F32.to_bits() == 0x3F7F_D51F);
const _: () = assert!(TANK_DC_BLOCKER_48K_F32.to_bits() == DC_BLOCKER_5HZ_48K_F32.to_bits());
const _: () = assert!(TANK_MAX_V_CEIL_BITS_DERIVED == 0x3F7C_28F5);
const _: () = assert!(TANK_MAX_G_LIN_BITS_DERIVED == 0x3F7F_BE76);
const _: () = assert!(TANK_MAX_G_LIN_BITS_DERIVED == MAX_LOOP_GAIN_F32.to_bits());
const _: () = assert!(const_abs_f64(TANH_R4[0] - 61.0 / 600.0) < 1.0e-15);
const _: () = assert!(const_abs_f64(TANH_R4[1] - 11.0 / 19_200.0) < 1.0e-15);
const _: () = assert!(const_abs_f64(TANH_R4[2] - 87.0 / 200.0) < 1.0e-15);
const _: () = assert!(const_abs_f64(TANH_R4[3] - 47.0 / 3_840.0) < 1.0e-15);

#[cfg(test)]
mod tests {
    use core::hint::black_box;

    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_const_eval_derivations_match_analytic_identities() {
        assert!((const_sqrt_f64(2.0) - core::f64::consts::SQRT_2).abs() < 1.0e-15);
        assert_eq!(const_sqrt_f64(0.0), 0.0);
        assert_eq!(const_sqrt_f64(-1.0), 0.0);
        assert_eq!(const_sqrt_f64(f64::INFINITY), 0.0);

        for k in -12..=12 {
            let x = (k as f64) * 0.37;
            let (s, c) = (const_sin_f64(x), const_cos_f64(x));
            assert!((s * s + c * c - 1.0).abs() < 1.0e-14);
            assert!((const_tan_f64(x) - s / c).abs() < 1.0e-14);
        }

        let [n1, n2, d1, d2] = const_c1_clamped_tanh_coeffs(4.0);
        let r = 4.0_f64;
        let r2 = r * r;
        let r4 = r2 * r2;
        let p5 = r * (1.0 + r2 * n1 + r4 * n2);
        let q4 = 1.0 + r2 * d1 + r4 * d2;
        let dp5 = 1.0 + 3.0 * r2 * n1 + 5.0 * r4 * n2;
        let dq4 = 2.0 * r * d1 + 4.0 * r2 * r * d2;
        assert!((p5 - q4).abs() < 1.0e-12);
        assert!((dp5 - dq4).abs() < 1.0e-12);

        let (c1, c3, c5) = const_hermite5_quarter_sine_coeffs();
        let p_1 = c1 - c3 + c5;
        let dp_1 = c1 - 3.0 * c3 + 5.0 * c5;
        let p_half = 0.5 * c1 - 0.125 * c3 + 0.03125 * c5;
        assert!((p_1 - 1.0).abs() < 1.0e-6);
        assert!(dp_1.abs() < 1.0e-5);
        assert!((p_half - core::f32::consts::FRAC_1_SQRT_2).abs() < 2.0e-3);
        assert_eq!(const_abs_f64(black_box(-0.0)).to_bits(), 0);
        assert_eq!(const_sin_cos_f64(black_box(f64::NAN)), (0.0, 1.0));
        assert_eq!(const_max_f32_bits_le(black_box(0.5)), 0x3F00_0000);
        assert_eq!(
            const_dc_blocker_pole_f32(black_box(5.0), black_box(48_000.0)).to_bits(),
            0x3F7F_D51F
        );
    }

    proptest! {
        #[test]
        fn prop_const_eval_trig_sqrt_and_f32_bound(
            angle in -100.0_f64..100.0_f64,
            radicand in 1.0e-12_f64..1.0e12_f64,
            bound in 1.0e-4_f64..1.0_f64,
        ) {
            let (s, c) = const_sin_cos_f64(black_box(angle));
            let (ref_s, ref_c) = angle.sin_cos();
            prop_assert!((s - ref_s).abs() < 1.0e-13);
            prop_assert!((c - ref_c).abs() < 1.0e-13);
            prop_assert_eq!(const_abs_f64(black_box(-radicand)), radicand);

            let root = const_sqrt_f64(black_box(radicand));
            let ref_root = radicand.sqrt();
            prop_assert!(((root - ref_root) / ref_root).abs() < 1.0e-14);

            let bits = const_max_f32_bits_le(black_box(bound));
            prop_assert!(f64::from(f32::from_bits(bits)) <= bound);
            prop_assert!(f64::from(f32::from_bits(bits + 1)) > bound);
        }
    }
}
