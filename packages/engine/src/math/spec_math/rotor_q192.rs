//! Quadrant-reduced degree-13/14 Horner contractive rotor with adaptive 1–2 ULP norm guard
//! and `Q0.192` fixed-point LFO phase accumulator (`ref[dsp.op-m.rotor.*]`).

use super::{round_f64_to_f32_bits, widen_f32_bits};

/// `pi` in IEEE-754 `f64` (`0x400921FB54442D18`, `ref[dsp.op-m.rotor.quadrant-horner]`).
pub const PI_F64_BITS: u64 = 0x4009_21FB_5444_2D18;
/// `pi / 2` in IEEE-754 `f64` (`0x3FF921FB54442D18`, `ref[dsp.op-m.rotor.quadrant-horner]`).
pub const HALF_PI_F64_BITS: u64 = 0x3FF9_21FB_5444_2D18;
/// `pi / 4` in IEEE-754 `f64` (`0x3FE921FB54442D18`, `ref[dsp.op-m.rotor.quadrant-horner]`).
pub const QUARTER_PI_F64_BITS: u64 = 0x3FE9_21FB_5444_2D18;
/// Largest admissible `f32` angle magnitude strictly below `pi` (`0x40490FDA`).
pub const MAX_ROTOR_ANGLE_F32_BITS: u32 = 0x4049_0FDA;

/// Applies the adaptive 1–2 ULP norm guard to `(c_bits, s_bits)` (`ref[dsp.op-m.rotor.ulp-guard]`).
#[inline(always)]
pub fn project_rotor_components_ulp_guard(
    mut c_bits: u32,
    mut s_bits: u32,
    active: bool,
) -> (u32, u32) {
    if !active {
        return (c_bits, s_bits);
    }
    let reduce_cosine = (c_bits & 0x7fff_ffff) >= (s_bits & 0x7fff_ffff);
    for _ in 0..2 {
        let c = widen_f32_bits(c_bits).unwrap_or(0.0);
        let s = widen_f32_bits(s_bits).unwrap_or(0.0);
        if c * c + s * s >= 1.0 {
            if reduce_cosine {
                c_bits -= 1;
            } else {
                s_bits -= 1;
            }
        }
    }
    (c_bits, s_bits)
}

/// Evaluates the quadrant-reduced degree-13/14 Horner contractive rotor on a signed `f32` bit
/// pattern `theta_bits` (`-pi < theta < pi`), returning `(cos_bits, sin_bits)` (`ref[dsp.op-m.rotor.contractive]`,
/// `ref[dsp.op-m.rotor.zero-hold]`, `ref[dsp.op-m.rotor.quadrant-horner]`, `ref[dsp.op-m.rotor.ulp-guard]`).
/// Returns `None` if `theta` is non-finite or `|theta| >= pi`.
#[inline(always)]
pub fn contractive_rotor_f32_bits(theta_bits: u32) -> Option<(u32, u32)> {
    let mag_bits = theta_bits & 0x7fff_ffff;
    if mag_bits == 0 {
        return Some((1.0_f32.to_bits(), 0.0_f32.to_bits()));
    }
    let theta = widen_f32_bits(theta_bits)?;
    let magnitude = theta.abs();
    let pi64 = f64::from_bits(PI_F64_BITS);
    let half_pi64 = f64::from_bits(HALF_PI_F64_BITS);
    let quarter_pi64 = f64::from_bits(QUARTER_PI_F64_BITS);
    if !theta.is_finite() || magnitude >= pi64 {
        return None;
    }

    let reflected = magnitude > half_pi64;
    let quadrant = if reflected {
        pi64 - magnitude
    } else {
        magnitude
    };
    let swapped = quadrant > quarter_pi64;
    let reduced = if swapped {
        half_pi64 - quadrant
    } else {
        quadrant
    };
    let sq = reduced * reduced;

    let sine = reduced
        * (1.0
            + sq * (-1.0 / 6.0
                + sq * (1.0 / 120.0
                    + sq * (-1.0 / 5040.0
                        + sq * (1.0 / 362_880.0
                            + sq * (-1.0 / 39_916_800.0 + sq * (1.0 / 6_227_020_800.0)))))));
    let cosine = 1.0
        + sq * (-0.5
            + sq * (1.0 / 24.0
                + sq * (-1.0 / 720.0
                    + sq * (1.0 / 40_320.0
                        + sq * (-1.0 / 3_628_800.0
                            + sq * (1.0 / 479_001_600.0 - sq * (1.0 / 87_178_291_200.0)))))));

    let quad_cos = if swapped { sine } else { cosine };
    let quad_sin = if swapped { cosine } else { sine };
    let res_cos_f64 = if reflected { -quad_cos } else { quad_cos };
    let res_sin_f64 = if (theta_bits & 0x8000_0000) != 0 {
        -quad_sin
    } else {
        quad_sin
    };

    let c_bits = round_f64_to_f32_bits(res_cos_f64)?;
    let s_bits = round_f64_to_f32_bits(res_sin_f64)?;
    Some(project_rotor_components_ulp_guard(c_bits, s_bits, true))
}

/// Evaluates `contractive_rotor_f32_bits` on `theta: f32`, returning `Some((cos, sin))` for `|theta| < pi`.
#[inline(always)]
pub fn contractive_rotor_f32(theta: f32) -> Option<(f32, f32)> {
    let (c_bits, s_bits) = contractive_rotor_f32_bits(theta.to_bits())?;
    Some((f32::from_bits(c_bits), f32::from_bits(s_bits)))
}

/// Steps an `f32` LFO state `(re, im)` with zero-increment hold (`ref[dsp.op-m.rotor.zero-hold]`):
/// when `theta == ±0.0`, `(re, im)` is returned unchanged bit-for-bit.
#[inline]
pub fn step_lfo_phasor_zero_hold(re: f32, im: f32, theta: f32) -> Option<(f32, f32)> {
    if (theta.to_bits() & 0x7fff_ffff) == 0 {
        return Some((re, im));
    }
    let (c, s) = contractive_rotor_f32(theta)?;
    let next_re = c * re - s * im;
    let next_im = s * re + c * im;
    let corr = 1.5 - 0.5 * (next_re * next_re + next_im * next_im);
    Some((next_re * corr, next_im * corr))
}

/// 256-bit constant $K = \lfloor 2^{256} / (2\pi) \rfloor$ in little-endian `u64` words (`ref[dsp.op-m.rotor.q0-192-lfo]`).
pub const INV_TAU_256_LE: [u64; 4] = [
    0x7f94_58ea_f7ae_f158,
    0x36d8_a566_4f10_e410,
    0x7f09_d5f4_7d4d_3770,
    0x28be_60db_9391_054a,
];

/// `Q0.192` fixed-point LFO phase accumulator (`3 × u64` little-endian words modulo $2^{192}$,
/// `ref[dsp.op-m.rotor.q0-192-lfo]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Q0_192Phase {
    /// Little-endian limbs: `words[0]` = bits `0..64`, `words[1]` = bits `64..128`, `words[2]` = bits `128..192`.
    pub words: [u64; 3],
}

impl Q0_192Phase {
    pub const ZERO: Self = Self { words: [0; 3] };

    /// Advances `self` modulo $2^{192}$ by a signed increment `(sign_negative, mag_words)`.
    #[inline]
    pub fn advance_signed(&mut self, negative: bool, mag: [u64; 3]) {
        if mag == [0, 0, 0] {
            return;
        }
        if !negative {
            let (w0, c0) = self.words[0].overflowing_add(mag[0]);
            let (w1a, c1a) = self.words[1].overflowing_add(mag[1]);
            let (w1, c1b) = w1a.overflowing_add(u64::from(c0));
            let w2 = self.words[2]
                .wrapping_add(mag[2])
                .wrapping_add(u64::from(c1a || c1b));
            self.words = [w0, w1, w2];
        } else {
            let (w0, b0) = self.words[0].overflowing_sub(mag[0]);
            let (w1a, b1a) = self.words[1].overflowing_sub(mag[1]);
            let (w1, b1b) = w1a.overflowing_sub(u64::from(b0));
            let w2 = self.words[2]
                .wrapping_sub(mag[2])
                .wrapping_sub(u64::from(b1a || b1b));
            self.words = [w0, w1, w2];
        }
    }

    /// Advances `self` by the increment corresponding to `theta_bits` (`f32` bit pattern in `(-pi, pi)`).
    #[inline]
    pub fn step_angle_bits(&mut self, theta_bits: u32) -> Option<()> {
        let (neg, mag) = q0_192_phase_increment_from_f32_bits(theta_bits)?;
        self.advance_signed(neg, mag);
        Some(())
    }

    /// Maps the top 32 bits of the `Q0.192` phase to a unit phasor `(cos, sin)` (`ref[dsp.op-m.rotor.q0-192-lfo]`).
    #[inline]
    pub fn to_phasor(self) -> (f32, f32) {
        let q32 = (self.words[2] >> 32) as u32;
        q32_phase_to_phasor(q32)
    }
}

/// Computes the signed `Q0.192` increment `(negative, [u64; 3])` for an `f32` angle bit pattern
/// `theta_bits` with `|theta| < pi` (`ref[dsp.op-m.rotor.q0-192-lfo]`).
pub fn q0_192_phase_increment_from_f32_bits(theta_bits: u32) -> Option<(bool, [u64; 3])> {
    let negative = (theta_bits >> 31) != 0;
    let magnitude = theta_bits & 0x7fff_ffff;
    if magnitude > MAX_ROTOR_ANGLE_F32_BITS {
        return None;
    }
    if magnitude == 0 {
        return Some((false, [0, 0, 0]));
    }
    let exponent = magnitude >> 23;
    let mut mantissa = (magnitude & 0x007f_ffff) as u64;
    let power: i32 = if exponent != 0 {
        mantissa |= 1 << 23;
        (exponent as i32) - 150
    } else {
        -149
    };

    // Multiply 24-bit `mantissa` by 256-bit `INV_TAU_256_LE` -> 5 x u64 little-endian limbs (`prod[0..5]`).
    let mut prod = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..4 {
        let cur = (mantissa as u128) * (INV_TAU_256_LE[i] as u128) + carry;
        prod[i] = cur as u64;
        carry = cur >> 64;
    }
    prod[4] = carry as u64;

    // Shift right by `shift = 64 - power` with ties-to-even rounding.
    let shift = (64 - power) as u32;
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;

    let mut out = [0u64; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let src = i + word_shift;
        if src < 5 {
            let lo = prod[src] >> bit_shift;
            let hi = if bit_shift > 0 && src + 1 < 5 {
                prod[src + 1] << (64 - bit_shift)
            } else {
                0
            };
            *slot = lo | hi;
        }
    }

    // Check rounding bit at `shift - 1` and sticky bits below `shift - 1`.
    let half_pos = shift - 1;
    let half_word = (half_pos / 64) as usize;
    let half_bit = half_pos % 64;
    let half_set = half_word < 5 && ((prod[half_word] >> half_bit) & 1) != 0;

    let mut sticky = false;
    for &w in prod.iter().take(half_word.min(5)) {
        if w != 0 {
            sticky = true;
            break;
        }
    }
    if !sticky && half_word < 5 && half_bit > 0 {
        let mask = (1u64 << half_bit) - 1;
        if (prod[half_word] & mask) != 0 {
            sticky = true;
        }
    }

    if half_set && (sticky || (out[0] & 1) != 0) {
        let (w0, c0) = out[0].overflowing_add(1);
        let (w1, c1) = out[1].overflowing_add(u64::from(c0));
        let w2 = out[2].wrapping_add(u64::from(c1));
        out = [w0, w1, w2];
    }

    Some((negative, out))
}

/// Maps a 32-bit `Q0.32` turn phase `q32` into a unit phasor `(cos, sin)` by splitting quadrant
/// `j = q32 >> 30` and residual `n = q32 & 0x3FFF_FFFF`, multiplying by `0x30C90FDB` ($2\pi/2^{32}$),
/// and applying the quadrant-reduced contractive rotor (`ref[dsp.op-m.rotor.q0-192-lfo]`, `ref[dsp.tempo.q64-q32-phase]`).
#[inline]
pub fn q32_phase_to_phasor(q32: u32) -> (f32, f32) {
    let quadrant = q32 >> 30;
    let remainder = q32 & 0x3fff_ffff;
    let scale = f32::from_bits(0x30c9_0fdb);
    let angle = (remainder as f32) * scale;
    let (c, s) = contractive_rotor_f32(angle).unwrap_or((1.0, 0.0));
    match quadrant {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::FtzDazGuard;

    /// verify[dsp.op-m.rotor.contractive]
    /// verify[dsp.op-m.rotor.zero-hold]
    /// verify[dsp.op-m.rotor.quadrant-horner]
    /// verify[dsp.op-m.rotor.ulp-guard]
    /// Verifies quadrant-reduced degree-13/14 Horner rotor accuracy (`< 1.6e-7`), strict contractive
    /// norm (`c^2 + s^2 < 1` for non-zero angles), signed symmetry, quadrant boundaries, and `±0.0` hold.
    #[test]
    fn test_contractive_rotor_zero_hold_quadrant_horner_and_ulp_guard() {
        let _guard = FtzDazGuard::new();

        // 1. Exact zero-increment hold on +0.0 and -0.0, plus non-zero step_lfo_phasor_zero_hold
        assert_eq!(
            contractive_rotor_f32_bits(0x0000_0000),
            Some((1.0_f32.to_bits(), 0.0_f32.to_bits()))
        );
        assert_eq!(
            contractive_rotor_f32_bits(0x8000_0000),
            Some((1.0_f32.to_bits(), 0.0_f32.to_bits()))
        );
        let held = (0.6_f32, 0.8_f32);
        assert_eq!(step_lfo_phasor_zero_hold(held.0, held.1, 0.0), Some(held));
        assert_eq!(step_lfo_phasor_zero_hold(held.0, held.1, -0.0), Some(held));
        let stepped = step_lfo_phasor_zero_hold(1.0, 0.0, 0.25).unwrap();
        assert!((stepped.0 - 0.25_f64.cos() as f32).abs() < 1e-6);
        assert!((stepped.1 - 0.25_f64.sin() as f32).abs() < 1e-6);
        assert!(step_lfo_phasor_zero_hold(1.0, 0.0, 4.0).is_none());

        // 2. Rejects |theta| >= pi or non-finite
        for bad in [
            core::f32::consts::PI,
            -core::f32::consts::PI,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert!(contractive_rotor_f32(bad).is_none());
        }

        // 3. Signed angles & subnormal/boundary angles
        for bits in [1_u32, 0x007f_ffff, 0x0080_0000, 0x3f80_0000, 0x4049_0fda] {
            let (cp, sp) = contractive_rotor_f32_bits(bits).unwrap();
            let (cn, sn) = contractive_rotor_f32_bits(bits | 0x8000_0000).unwrap();
            assert_eq!(cp, cn);
            assert_eq!(sp ^ 0x8000_0000, sn);
            let c = widen_f32_bits(cp).unwrap();
            let s = widen_f32_bits(sp).unwrap();
            assert!(c * c + s * s < 1.0);
        }

        // 4. Small cosine near pi/2 quadrant boundary
        let half_pi_bits = (core::f64::consts::FRAC_PI_2 as f32).to_bits();
        for bits in (half_pi_bits - 2)..=(half_pi_bits + 2) {
            let theta64 = widen_f32_bits(bits).unwrap();
            let (c_bits, _) = contractive_rotor_f32_bits(bits).unwrap();
            let c64 = widen_f32_bits(c_bits).unwrap();
            assert!((c64 - theta64.cos()).abs() < 1e-14);
        }

        // 5. ULP guard inactive bypass and two-step limit verification
        assert_eq!(
            project_rotor_components_ulp_guard(0x3f80_0000, 0, false),
            (0x3f80_0000, 0)
        );
        let small_bits = (-30.0_f64).exp2() as f32;
        let (cg, sg) =
            project_rotor_components_ulp_guard(1.0_f32.to_bits(), small_bits.to_bits(), true);
        let cg64 = widen_f32_bits(cg).unwrap();
        let sg64 = widen_f32_bits(sg).unwrap();
        assert!(cg64 < 1.0);
        assert!(cg64 * cg64 + sg64 * sg64 < 1.0);

        let above_sqrt_half = f32::from_bits((0.5_f64.sqrt() as f32).to_bits() + 1);
        let (c2, s2) = project_rotor_components_ulp_guard(
            above_sqrt_half.to_bits(),
            above_sqrt_half.to_bits(),
            true,
        );
        let c2_64 = widen_f32_bits(c2).unwrap();
        let s2_64 = widen_f32_bits(s2).unwrap();
        assert_eq!(c2, above_sqrt_half.to_bits() - 2);
        assert_eq!(s2, above_sqrt_half.to_bits());
        assert!(c2_64 * c2_64 + s2_64 * s2_64 < 1.0);

        // 6. Sweep across 200,000 angles in (0, pi)
        let last_f32 = f32::from_bits(0x4049_0fda);
        let steps = 200_000usize;
        let mut max_comp_err = 0.0_f64;
        let mut max_ang_err = 0.0_f64;
        let mut min_norm_sq = 1.0_f64;
        for i in 1..=steps {
            let theta = (last_f32 as f64 * (i as f64) / (steps as f64)) as f32;
            let (c, s) = contractive_rotor_f32(theta).unwrap();
            let c64 = c as f64;
            let s64 = s as f64;
            let t64 = theta as f64;
            max_comp_err = max_comp_err
                .max((c64 - t64.cos()).abs())
                .max((s64 - t64.sin()).abs());
            max_ang_err = max_ang_err.max((s64.atan2(c64) - t64).abs());
            let n2 = c64 * c64 + s64 * s64;
            assert!(n2 < 1.0, "Non-contractive norm at theta={theta}: n2={n2}");
            min_norm_sq = min_norm_sq.min(n2);
        }
        assert!(max_comp_err < 1.6e-7);
        assert!(max_ang_err < 2.3e-7);
        assert!(min_norm_sq > 0.999_999_8);
    }

    /// verify[dsp.op-m.rotor.q0-192-lfo]
    /// Verifies `Q0.192` fixed-point LFO phase increment and slow-motion retention at `2^-30 rad/tick`.
    #[test]
    fn test_q0_192_lfo_phase_accumulator_and_slow_motion() {
        let _guard = FtzDazGuard::new();
        // Slow LFO angle 2^-30 rad/tick over 300,000 ticks
        let theta = f32::from_bits((127 - 30) << 23);
        let mut phase = Q0_192Phase {
            words: [0, 0, 1u64 << 61], // pi / 4 initial phase (1 << 189)
        };
        let initial = phase;
        assert_eq!(
            q0_192_phase_increment_from_f32_bits(0),
            Some((false, [0, 0, 0]))
        );
        phase.step_angle_bits(0).unwrap();
        assert_eq!(phase, initial);

        let (neg, step_words) = q0_192_phase_increment_from_f32_bits(theta.to_bits()).unwrap();
        assert!(!neg);
        assert!(step_words[0] >= (1u64 << 40));
        let (neg_sign, neg_words) =
            q0_192_phase_increment_from_f32_bits(theta.to_bits() | 0x8000_0000).unwrap();
        assert!(neg_sign);
        assert_eq!(neg_words, step_words);

        // Exercise subnormal f32 angle increment and borrow across negative Q0.192 steps
        assert_eq!(
            q0_192_phase_increment_from_f32_bits(0x0000_0001),
            Some((false, [1_399_941_684_381, 0, 0]))
        );
        assert!(q0_192_phase_increment_from_f32_bits(MAX_ROTOR_ANGLE_F32_BITS + 1).is_none());
        let mut neg_phase = Q0_192Phase::ZERO;
        neg_phase.advance_signed(true, [1, 0, 0]);
        assert_eq!(neg_phase.words, [u64::MAX, u64::MAX, u64::MAX]);

        for _ in 0..300_000 {
            phase.step_angle_bits(theta.to_bits()).unwrap();
        }
        let (c, s) = phase.to_phasor();
        let actual_angle = (s as f64).atan2(c as f64);
        let expected_angle = core::f64::consts::FRAC_PI_4 + (theta as f64) * 300_000.0;
        assert!((actual_angle - expected_angle).abs() < 4e-7);

        // Verify q32_phase_to_phasor quadrant boundaries
        assert_eq!(q32_phase_to_phasor(0), (1.0, 0.0));
        assert_eq!(q32_phase_to_phasor(0x4000_0000), (0.0, 1.0));
        assert_eq!(q32_phase_to_phasor(0x8000_0000), (-1.0, 0.0));
        assert_eq!(q32_phase_to_phasor(0xc000_0000), (0.0, -1.0));
    }
}
