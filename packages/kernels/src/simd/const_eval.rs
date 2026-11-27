//! Compile-time mathematical evaluation, IEEE-754 seed derivations, and fixed-shape `SONA` tier
//! specifications (`ref[repo.code.constant-eval-and-derivations]`, `ref[sdk.rust.shared-nostd-kernels]`).
//!
//! All constants derive from pure `u32` fixed-point `const fn` routines (zero `f64`, `u64`, `i64`,
//! `u128`, `i128`, or `libm` dependencies) so this module compiles identically on `#![no_std]` CPU
//! targets and 32-bit `spirv-unknown-naga-wgsl` shaders.

mod fixed_point;

use fixed_point::{
    ONE_Q60, U64Limb, div_fixed_q60, mul_q60, mul_u32_wide, mul_u64_u32_div_u32, pi_q60,
};

/// IEEE-754 `binary32` exponent bias (`127`).
pub const IEEE754_F32_EXP_BIAS: u32 = 127;

/// IEEE-754 `binary32` explicit fraction width (`23` bits).
pub const IEEE754_F32_MANTISSA_BITS: u32 = 23;

/// IEEE-754 `binary16` (`fp16`) exponent bias (`15`).
pub const IEEE754_FP16_EXP_BIAS: u32 = 15;

/// IEEE-754 `binary16` (`fp16`) explicit fraction width (`10` bits).
pub const IEEE754_FP16_MANTISSA_BITS: u32 = 10;

/// IEEE-754 `binary64` exponent bias (`1023`, stored as 32-bit `i32` for 32-bit `spirv` compatibility).
pub const IEEE754_F64_EXP_BIAS: i32 = 1023;

/// IEEE-754 `binary64` explicit fraction width (`52` bits).
pub const IEEE754_F64_MANTISSA_BITS: u32 = 52;

/// Minimum unbiased normal exponent for IEEE-754 `binary32` (`1 - 127 = -126`).
pub const IEEE754_F32_MIN_NORMAL_EXP: i32 = 1 - (IEEE754_F32_EXP_BIAS as i32);

/// Minimum unbiased subnormal LSB exponent for IEEE-754 `binary32` (`-126 - 23 = -149`).
pub const IEEE754_F32_MIN_SUBNORMAL_EXP: i32 =
    IEEE754_F32_MIN_NORMAL_EXP - (IEEE754_F32_MANTISSA_BITS as i32);

/// Sum `E_bias + M_bits` (`127 + 23 = 150`) mapping biased normal exponents cleanly to scaled shifts.
pub const IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET: i32 =
    (IEEE754_F32_EXP_BIAS as i32) + (IEEE754_F32_MANTISSA_BITS as i32);

/// IEEE-754 `binary32` explicit 23-bit mantissa mask (`0x007F_FFFF`).
pub const IEEE754_F32_MANTISSA_MASK: u32 = (1_u32 << IEEE754_F32_MANTISSA_BITS) - 1;

/// IEEE-754 `binary32` implicit leading significand bit (`2^23 == 0x0080_0000`).
pub const IEEE754_F32_IMPLICIT_BIT: u32 = 1_u32 << IEEE754_F32_MANTISSA_BITS;

/// IEEE-754 `binary32` total significand precision (`24` bits including implicit bit).
pub const IEEE754_F32_SIGNIFICAND_BITS: u32 = IEEE754_F32_MANTISSA_BITS + 1;

/// IEEE-754 `binary32` sign bit mask (`0x8000_0000`).
pub const IEEE754_F32_SIGN_MASK: u32 = 1_u32 << 31;

/// IEEE-754 `binary32` magnitude bitmask stripping the sign bit (`0x7FFF_FFFF`).
pub const IEEE754_F32_ABS_MASK: u32 = !IEEE754_F32_SIGN_MASK;

/// IEEE-754 `binary32` all-ones biased exponent field (`0xFF`, `2 * 127 + 1`).
pub const IEEE754_F32_EXP_ALL_ONES: u32 = (2 * IEEE754_F32_EXP_BIAS) + 1;

/// IEEE-754 `binary32` exponent bitmask (`0x7F80_0000`).
pub const IEEE754_F32_EXP_MASK: u32 = IEEE754_F32_EXP_ALL_ONES << IEEE754_F32_MANTISSA_BITS;

/// IEEE-754 `binary32` exact `1.0` bit pattern (`0x3F80_0000`).
pub const IEEE754_F32_ONE_BITS: u32 = IEEE754_F32_EXP_BIAS << IEEE754_F32_MANTISSA_BITS;

/// Mantissa right-shift converting IEEE-754 `binary32` fractions into `binary16` (`23 - 10 == 13`).
pub const IEEE754_F32_TO_FP16_MANTISSA_SHIFT: u32 =
    IEEE754_F32_MANTISSA_BITS - IEEE754_FP16_MANTISSA_BITS;

/// IEEE-754 `binary16` (`fp16`) sign bit mask in lower 16 bits (`0x8000`).
pub const IEEE754_FP16_SIGN_MASK: u32 = 1_u32 << 15;

/// IEEE-754 `binary16` (`fp16`) magnitude mask in lower 16 bits (`0x7FFF`).
pub const IEEE754_FP16_ABS_MASK: u32 = IEEE754_FP16_SIGN_MASK - 1;

/// IEEE-754 `binary16` (`fp16`) explicit 10-bit mantissa mask (`0x03FF`).
pub const IEEE754_FP16_MANTISSA_MASK: u32 = (1_u32 << IEEE754_FP16_MANTISSA_BITS) - 1;

/// IEEE-754 `binary16` (`fp16`) all-ones biased exponent field (`0x1F == 31`).
pub const IEEE754_FP16_EXP_ALL_ONES: u32 = (2 * IEEE754_FP16_EXP_BIAS) + 1;

/// IEEE-754 `binary16` (`fp16`) infinity exponent bitmask (`0x7C00`).
pub const IEEE754_FP16_EXP_MASK: u32 = IEEE754_FP16_EXP_ALL_ONES << IEEE754_FP16_MANTISSA_BITS;

/// Maximum finite positive IEEE-754 `binary16` (`fp16`) bit pattern (`0x7BFF`).
pub const IEEE754_FP16_MAX_FINITE_BITS: u32 = IEEE754_FP16_EXP_MASK - 1;

/// Maximum normal biased exponent for IEEE-754 `binary64` (`2 * 1023 == 2046`, stored as `i32`).
pub const IEEE754_F64_MAX_NORMAL_BIASED_EXP: i32 = 2 * IEEE754_F64_EXP_BIAS;

/// IEEE-754 `binary64` all-ones 11-bit exponent field (`0x7FF`, stored as `u32` for `spirv`).
pub const IEEE754_F64_EXP_ALL_ONES: u32 = (2 * IEEE754_F64_EXP_BIAS + 1) as u32;

/// IEEE-754 `1.5 * 2^23` (`12_582_912.0`) magic integer-rounding bias (`0x4B40_0000`).
pub const BIASED_INT_MAGIC_F32: f32 = 1.5 * ((1_u32 << IEEE754_F32_MANTISSA_BITS) as f32);

/// Derives an IEEE-754 `binary32` exact power-of-two scalar `2^exp` (`exp in -126..=127`)
/// purely via exponent bit shifting (`ref[repo.code.constant-eval-and-derivations]`).
#[inline(always)]
#[must_use]
pub const fn const_f32_pow2(exp: i32) -> f32 {
    let biased = (IEEE754_F32_EXP_BIAS as i32) + exp;
    f32::from_bits((biased as u32) << IEEE754_F32_MANTISSA_BITS)
}

/// Exact-integer truncation boundary `2^23` (`8_388_608.0`) in IEEE-754 `binary32`.
pub const F32_INT_EXACT_THRESHOLD: f32 = const_f32_pow2(IEEE754_F32_MANTISSA_BITS as i32);

/// Maximum finite IEEE-754 `binary16` (`fp16`) magnitude `(2 - 2^-10) * 2^15 == 65,504.0`
/// (`0x477F_E000` in `binary32`, `0x7BFF` in `fp16`).
pub const FP16_FINITE_MAX_F32: f32 = (2.0 - const_f32_pow2(-(IEEE754_FP16_MANTISSA_BITS as i32)))
    * const_f32_pow2(IEEE754_FP16_EXP_BIAS as i32);

/// Branch-free compile-time `|x|` for `binary32` (`ref[sdk.rust.shared-nostd-kernels]`).
#[inline(always)]
#[must_use]
pub const fn const_abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & IEEE754_F32_ABS_MASK)
}

/// Compile-time Taylor inverse factorial `(-1)^neg / n!` evaluated strictly in 32-bit
/// `f32`/`u32` arithmetic (`ref[repo.code.constant-eval-and-derivations]`).
#[inline(always)]
#[must_use]
pub const fn const_taylor_inv_fact_f32(n: u32, negative: bool) -> f32 {
    let mut fact = 1.0_f32;
    let mut k = 2_u32;
    while k <= n {
        fact *= k as f32;
        k += 1;
    }
    let mag = 1.0_f32 / fact;
    if negative { -mag } else { mag }
}

/// Chebyshev minimax `binary32` positive reciprocal integer-sub seed
/// `((2 * E_bias - 1) << 23) + round((2 * sqrt(6) - 4) * 2^23) == 0x7EF3_11C3`
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
pub const IEEE754_F32_RCP_SEED: u32 = const_f32_rcp_chebyshev_seed();

/// Robertson/Lomont minimax `binary32` global two-binade inverse-square-root seed
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
pub const IEEE754_F32_RSQRT_SEED: u32 = 0x5F37_59DF;

/// Compile-time `5 Hz` passband-normalized DC blocker pole at `48 kHz` (`0x3F7F_D51F`,
/// `ref[dsp.tank.dc-blocker]`).
pub const DC_BLOCKER_5HZ_48K_F32: f32 = f32::from_bits(const_dc_blocker_pole_bits(5, 48_000));

/// Compile-time `10 Hz` passband-normalized DC blocker pole bit pattern at `48 kHz` (`0x3F7F_AA45`).
pub const DC_BLOCKER_10HZ_48K_BITS: u32 = const_dc_blocker_pole_bits(10, 48_000);

/// Largest `binary32` value `<= 0.9990` (`0x3F7F_BE76`, `ref[dsp.tank.quiet-feedback-knee]`).
pub const MAX_LOOP_GAIN_F32: f32 = f32::from_bits(const_max_f32_bits_le_ratio(999, 1_000));

/// Largest `binary32` value `<= 0.999995` (`0x3F7F_FFAC`, upper `ParaLRU` and ZOH pole bound).
pub const MAX_PARALRU_R_F32: f32 = f32::from_bits(const_max_f32_bits_le_ratio(999_995, 1_000_000));

/// Clamped pre-activation endpoint magnitude bound `2^12 = 4096.0` for `AlgADAA1(0.5)` (`MAX_ALG_R^3 < 1e11 << f32::MAX`, `ref[dsp.op-n.polyadaa.clamp]`).
pub const MAX_ALG_R: f32 = const_f32_pow2(12);

/// Upper gain bound `[0.0, 2.5]` for $C^\infty$ `g_pre(v_eff^2)` pre-ADAA scaling (`ref[dsp.op-n.polyadaa.clamp]`).
pub const G_PRE_MAX: f32 = 2.5;

/// Saturation energy bound `min(v_eff^2, 1.0e6)` for `g_pre(v_eff^2)` (`ref[dsp.op-n.polyadaa.clamp]`).
pub const G_PRE_E_MAX: f32 = 1.0e6;

/// Saturation magnitude bound `min(|v_eff|, 1.0e6)` for post-ADAA rank-1 `SE-Bus` (`ref[dsp.op-n.core.interblock-tiles]`).
pub const SE_BUS_ABS_MAX: f32 = 1.0e6;

/// Upper bound on `|2 * svf_g * bp|` (`1.0e18`) preventing `gamma_duhem * |flux|` overflow to `+Inf` (`ref[dsp.op-n.core.matryoshka-blocks]`).
pub const MAX_DUHEM_FLUX_ABS: f32 = 1.0e18;

/// Compile-time fixed-shape tier capacity profile locked at `Gate 0b` (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SonaTierSpec {
    /// Number of 4-lane `Vec4` quads in each of the 4 Matryoshka `SONA` core blocks.
    pub blocks: [usize; 4],
    /// Number of 4-mode `TPT-SVF` `Vec4` quads in the `SONA` core.
    pub svf_quads: usize,
    /// Number of cabinet early-reflection FIR taps per stereo channel.
    pub fir_taps: usize,
    /// Number of cabinet shared conjugate Kautz modal pole pairs.
    pub kautz_pairs: usize,
    /// Number of `Op-T` transport delay lines (`M` lines = `4 * quads`).
    pub opt_lines: usize,
    /// Core oversampling factor (`1` for `Lite`, `2` for `Full`/`Pro`/`Ultra` real-time).
    pub oversample_factor: usize,
}

impl SonaTierSpec {
    /// Constructs a compile-time `SONA` tier profile with `svf_quads` derived from `blocks`
    /// and `opt_lines = 4 * tank_quads`.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        blocks: [usize; 4],
        fir_taps: usize,
        kautz_pairs: usize,
        tank_quads: usize,
        oversample_factor: usize,
    ) -> Self {
        let svf_quads = blocks[0] + blocks[1] + blocks[2] + blocks[3];
        Self {
            blocks,
            svf_quads,
            fir_taps,
            kautz_pairs,
            opt_lines: 4 * tank_quads,
            oversample_factor,
        }
    }

    /// Returns the total number of 4-lane `Vec4` groups across all 4 Matryoshka `SONA` blocks.
    #[inline(always)]
    #[must_use]
    pub const fn total_groups(&self) -> usize {
        self.blocks[0] + self.blocks[1] + self.blocks[2] + self.blocks[3]
    }

    /// Returns the total number of 4-lane `Vec4` quads across all 4 Matryoshka `SONA` core blocks.
    #[inline(always)]
    #[must_use]
    pub const fn block_sum(&self) -> usize {
        self.total_groups()
    }

    /// Returns the total number of scalar `SONA` core lanes (`4 * self.total_groups()`).
    #[inline(always)]
    #[must_use]
    pub const fn lanes(&self) -> usize {
        4 * self.total_groups()
    }

    /// Returns the total number of scalar `SONA` core lanes (`4 * self.block_sum()`).
    #[inline(always)]
    #[must_use]
    pub const fn total_lanes(&self) -> usize {
        self.lanes()
    }

    /// Returns the number of tank `Op-T` transport delay quads (`opt_lines >> 2`).
    #[inline(always)]
    #[must_use]
    pub const fn tank_quads(&self) -> usize {
        self.opt_lines >> 2
    }
}

/// `Lite` compile-time tier capacity profile (`[1, 1, 1, 1]`, `16` lanes, `4` SVF quads, `32` FIR, `8` Kautz).
pub const LITE_SPEC: SonaTierSpec = SonaTierSpec::new([1, 1, 1, 1], 32, 8, 1, 1);

/// `Full` compile-time tier capacity profile (`[2, 2, 1, 1]`, `24` lanes @ `2x`, `6` SVF quads, `48` FIR, `12` Kautz).
pub const FULL_SPEC: SonaTierSpec = SonaTierSpec::new([2, 2, 1, 1], 48, 12, 2, 2);

/// `Pro` compile-time tier capacity profile (`[3, 2, 2, 1]`, `32` lanes @ `2x`, `8` SVF quads, `96` FIR, `24` Kautz).
pub const PRO_SPEC: SonaTierSpec = SonaTierSpec::new([3, 2, 2, 1], 96, 24, 3, 2);

/// `Ultra` compile-time tier capacity profile (`[4, 3, 2, 1]`, `40` lanes @ `2x`, `10` SVF quads, `128` FIR, `32` Kautz).
pub const ULTRA_SPEC: SonaTierSpec = SonaTierSpec::new([4, 3, 2, 1], 128, 32, 4, 2);

/// Compile-time Chebyshev minimax `binary32` positive reciprocal integer-sub seed
/// `((2 * E_bias - 1) << 23) + round((2 * sqrt(6) - 4) * 2^23) == 0x7EF3_11C3`
/// computed via pure `u32` fixed-point integer square root (`24 * 2^58`).
#[must_use]
pub const fn const_f32_rcp_chebyshev_seed() -> u32 {
    let target = U64Limb {
        hi: 24_u32 << 26,
        lo: 0,
    };
    let mut res: u32 = 0;
    let mut bit: u32 = 1_u32 << 31;
    while bit > 0 {
        let cand = res | bit;
        let sq = mul_u32_wide(cand, cand);
        if target.ge(sq) {
            res = cand;
        }
        bit >>= 1;
    }
    let four_q29 = 4_u32 << 29;
    let mant_q29 = res - four_q29;
    let mant_round = (mant_q29 + (1_u32 << 5)) >> 6;
    ((2 * IEEE754_F32_EXP_BIAS - 1) << IEEE754_F32_MANTISSA_BITS) + mant_round
}

/// Compile-time largest positive `binary32` bit pattern `<= num / den` for `0 < num <= den`.
#[must_use]
pub const fn const_max_f32_bits_le_ratio(num: u32, den: u32) -> u32 {
    if den == 0 || num >= den {
        return IEEE754_F32_ONE_BITS;
    }
    if num == 0 {
        return 0;
    }
    let den_64 = U64Limb::from_u32(den);
    let mut num_64 = U64Limb::from_u32(num);
    let mut exp: u32 = IEEE754_F32_EXP_BIAS - 1;
    while !num_64.shl1().ge(den_64) && exp > 1 {
        num_64 = num_64.shl1();
        exp -= 1;
    }
    let scaled = num_64.shl(IEEE754_F32_SIGNIFICAND_BITS);
    let (q, _) = scaled.div_rem(den_64);
    (exp << IEEE754_F32_MANTISSA_BITS) + (q.lo - IEEE754_F32_IMPLICIT_BIT)
}

/// Compile-time passband-normalized DC blocker pole `R = cos(w) / (1 + sin(w))` in `binary32` bits,
/// where `w = 2 * pi * fc_hz / fs_hz` (`ref[dsp.tank.dc-blocker]`).
#[must_use]
pub const fn const_dc_blocker_pole_bits(fc_hz: u32, fs_hz: u32) -> u32 {
    if fc_hz == 0 {
        return IEEE754_F32_ONE_BITS;
    }
    if fs_hz == 0 || 4 * fc_hz >= fs_hz {
        return 0;
    }
    let tau_q60 = pi_q60().shl1();
    let omega = mul_u64_u32_div_u32(tau_q60, fc_hz, fs_hz);
    let w2 = mul_q60(omega, omega);

    let mut sin_term = omega;
    let mut sin_sum = omega;
    let mut n: u32 = 1;
    while n <= 10 {
        sin_term = mul_q60(sin_term, w2);
        let denom = U64Limb::from_u32((2 * n) * (2 * n + 1));
        sin_term = sin_term.div_rem(denom).0;
        if (n & 1) == 1 {
            sin_sum = sin_sum.sub(sin_term);
        } else {
            sin_sum = sin_sum.add(sin_term);
        }
        n += 1;
    }

    let mut cos_term = ONE_Q60;
    let mut cos_sum = ONE_Q60;
    let mut m: u32 = 1;
    while m <= 10 {
        cos_term = mul_q60(cos_term, w2);
        let denom = U64Limb::from_u32((2 * m - 1) * (2 * m));
        cos_term = cos_term.div_rem(denom).0;
        if (m & 1) == 1 {
            cos_sum = cos_sum.sub(cos_term);
        } else {
            cos_sum = cos_sum.add(cos_term);
        }
        m += 1;
    }

    let r_q60 = div_fixed_q60(cos_sum, ONE_Q60.add(sin_sum));
    let half_q60 = U64Limb {
        hi: 1_u32 << 27,
        lo: 0,
    };
    let mut exp: u32 = IEEE754_F32_EXP_BIAS - 1;
    let mut val = r_q60;
    while !val.ge(half_q60) && exp > 1 {
        val = val.shl1();
        exp -= 1;
    }
    let mut sig = val.hi >> 4;
    let round_bit = (val.hi >> 3) & 1;
    let sticky = (val.hi & 0x7) != 0 || val.lo != 0;
    if round_bit != 0 && (sticky || (sig & 1) != 0) {
        sig += 1;
    }
    if sig >= (1_u32 << IEEE754_F32_SIGNIFICAND_BITS) {
        sig >>= 1;
        exp += 1;
    }
    (exp << IEEE754_F32_MANTISSA_BITS) | (sig & IEEE754_F32_MANTISSA_MASK)
}

const fn verify_tier_spec(
    spec: &SonaTierSpec,
    expected_groups: usize,
    expected_tank_quads: usize,
    expected_fir: usize,
    expected_kautz: usize,
    expected_os: usize,
) -> bool {
    spec.total_groups() == expected_groups
        && spec.block_sum() == expected_groups
        && spec.svf_quads == expected_groups
        && spec.svf_quads * 4 == spec.lanes()
        && spec.lanes() == spec.total_lanes()
        && (spec.opt_lines & 3) == 0
        && spec.tank_quads() == expected_tank_quads
        && spec.opt_lines == expected_tank_quads * 4
        && spec.fir_taps == expected_fir
        && spec.kautz_pairs == expected_kautz
        && spec.oversample_factor == expected_os
}

const fn is_matryoshka_monotonic(lo: &SonaTierSpec, hi: &SonaTierSpec) -> bool {
    lo.blocks[0] <= hi.blocks[0]
        && lo.blocks[1] <= hi.blocks[1]
        && lo.blocks[2] <= hi.blocks[2]
        && lo.blocks[3] <= hi.blocks[3]
        && lo.svf_quads <= hi.svf_quads
        && lo.fir_taps <= hi.fir_taps
        && lo.kautz_pairs <= hi.kautz_pairs
        && lo.opt_lines <= hi.opt_lines
        && lo.oversample_factor <= hi.oversample_factor
        && lo.total_lanes() < hi.total_lanes()
}

const _: () = assert!(IEEE754_F32_RCP_SEED == 0x7EF3_11C3);
const _: () = assert!(DC_BLOCKER_5HZ_48K_F32.to_bits() == 0x3F7F_D51F);
const _: () = assert!(DC_BLOCKER_10HZ_48K_BITS == 0x3F7F_AA45);
const _: () = assert!(MAX_LOOP_GAIN_F32.to_bits() == 0x3F7F_BE76);
const _: () = assert!(MAX_LOOP_GAIN_F32 <= 0.9990);
const _: () = assert!(MAX_PARALRU_R_F32.to_bits() == 0x3F7F_FFAC);
const _: () = assert!(MAX_PARALRU_R_F32 <= 0.999_995);
const _: () = assert!(LITE_SPEC.svf_quads * 4 == LITE_SPEC.lanes());
const _: () = assert!(FULL_SPEC.svf_quads * 4 == FULL_SPEC.lanes());
const _: () = assert!(PRO_SPEC.svf_quads * 4 == PRO_SPEC.lanes());
const _: () = assert!(ULTRA_SPEC.svf_quads * 4 == ULTRA_SPEC.lanes());
const _: () = assert!(verify_tier_spec(&LITE_SPEC, 4, 1, 32, 8, 1));
const _: () = assert!(verify_tier_spec(&FULL_SPEC, 6, 2, 48, 12, 2));
const _: () = assert!(verify_tier_spec(&PRO_SPEC, 8, 3, 96, 24, 2));
const _: () = assert!(verify_tier_spec(&ULTRA_SPEC, 10, 4, 128, 32, 2));
const _: () = assert!(is_matryoshka_monotonic(&LITE_SPEC, &FULL_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&FULL_SPEC, &PRO_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&PRO_SPEC, &ULTRA_SPEC));
const _: () = assert!(IEEE754_FP16_EXP_BIAS == 15);
const _: () = assert!(IEEE754_F64_EXP_BIAS == 1023);
const _: () = assert!(IEEE754_F32_MIN_NORMAL_EXP == -126);
const _: () = assert!(IEEE754_F32_MIN_SUBNORMAL_EXP == -149);
const _: () = assert!(IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET == 150);
const _: () = assert!(F32_INT_EXACT_THRESHOLD == 8_388_608.0);
const _: () = assert!(IEEE754_F32_ABS_MASK == 0x7FFF_FFFF);
const _: () = assert!(IEEE754_F32_SIGN_MASK == 0x8000_0000);
const _: () = assert!(IEEE754_F32_MANTISSA_MASK == 0x007F_FFFF);
const _: () = assert!(IEEE754_F32_IMPLICIT_BIT == 0x0080_0000);
const _: () = assert!(IEEE754_F32_SIGNIFICAND_BITS == 24);
const _: () = assert!(IEEE754_F32_EXP_ALL_ONES == 0xFF);
const _: () = assert!(IEEE754_F32_EXP_MASK == 0x7F80_0000);
const _: () = assert!(IEEE754_F32_ONE_BITS == 0x3F80_0000);
const _: () = assert!(IEEE754_F32_TO_FP16_MANTISSA_SHIFT == 13);
const _: () = assert!(IEEE754_FP16_SIGN_MASK == 0x8000);
const _: () = assert!(IEEE754_FP16_ABS_MASK == 0x7FFF);
const _: () = assert!(IEEE754_FP16_MANTISSA_MASK == 0x03FF);
const _: () = assert!(IEEE754_FP16_EXP_ALL_ONES == 0x1F);
const _: () = assert!(IEEE754_FP16_EXP_MASK == 0x7C00);
const _: () = assert!(IEEE754_FP16_MAX_FINITE_BITS == 0x7BFF);
const _: () = assert!(IEEE754_F64_MAX_NORMAL_BIASED_EXP == 2046);
const _: () = assert!(IEEE754_F64_EXP_ALL_ONES == 0x7FF);
const _: () = assert!(BIASED_INT_MAGIC_F32.to_bits() == 0x4B40_0000);
const _: () = assert!(FP16_FINITE_MAX_F32 == 65_504.0);
const _: () = assert!(FP16_FINITE_MAX_F32.to_bits() == 0x477F_E000);
const _: () = assert!(const_f32_pow2(-16) * const_f32_pow2(16) == 1.0);
const _: () = assert!(const_abs_f32(-0.0).to_bits() == 0);
const _: () = assert!(const_abs_f32(-2.5) == 2.5);
const _: () = assert!(const_taylor_inv_fact_f32(0, false) == 1.0);
const _: () = assert!(const_taylor_inv_fact_f32(2, true) == -0.5);
const _: () = assert!(const_abs_f32(const_taylor_inv_fact_f32(3, false) - 1.0 / 6.0) < 1.0e-7);
const _: () = assert!(MAX_ALG_R == 4_096.0);
const _: () = assert!(
    (MAX_ALG_R * MAX_ALG_R * MAX_ALG_R).is_finite() && MAX_ALG_R * MAX_ALG_R * MAX_ALG_R < 1.0e11
);
const _: () = assert!(G_PRE_MAX == 2.5 && G_PRE_MAX > 0.0);
const _: () = assert!(G_PRE_E_MAX == 1.0e6 && (64.0 * G_PRE_E_MAX).is_finite());
const _: () = assert!(SE_BUS_ABS_MAX == 1.0e6 && (4.0 * SE_BUS_ABS_MAX).is_finite());
const _: () = assert!(MAX_DUHEM_FLUX_ABS == 1.0e18 && (64.0 * MAX_DUHEM_FLUX_ABS).is_finite());

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn dc_pole_f64_oracle(fc_hz: u32, fs_hz: u32) -> u32 {
        let w = core::f64::consts::TAU * (fc_hz as f64) / (fs_hz as f64);
        let r = (w.cos() / (1.0 + w.sin())) as f32;
        r.to_bits()
    }

    /// verify[repo.code.constant-eval-and-derivations]
    #[test]
    fn test_const_eval_seeds_poles_bounds_and_tiers() {
        assert_eq!(IEEE754_F32_EXP_BIAS, 127);
        assert_eq!(IEEE754_F32_MANTISSA_BITS, 23);
        assert_eq!(IEEE754_FP16_EXP_BIAS, 15);
        assert_eq!(IEEE754_FP16_MANTISSA_BITS, 10);
        assert_eq!(IEEE754_F64_EXP_BIAS, 1023);
        assert_eq!(IEEE754_F64_MANTISSA_BITS, 52);
        assert_eq!(IEEE754_F32_MIN_NORMAL_EXP, -126);
        assert_eq!(IEEE754_F32_MIN_SUBNORMAL_EXP, -149);
        assert_eq!(IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET, 150);
        assert_eq!(F32_INT_EXACT_THRESHOLD, 8_388_608.0);
        assert_eq!(IEEE754_F32_ABS_MASK, 0x7FFF_FFFF);
        assert_eq!(IEEE754_F32_SIGN_MASK, 0x8000_0000);
        assert_eq!(IEEE754_F32_MANTISSA_MASK, 0x007F_FFFF);
        assert_eq!(IEEE754_F32_IMPLICIT_BIT, 0x0080_0000);
        assert_eq!(IEEE754_F32_SIGNIFICAND_BITS, 24);
        assert_eq!(IEEE754_F32_EXP_ALL_ONES, 0xFF);
        assert_eq!(IEEE754_F32_EXP_MASK, 0x7F80_0000);
        assert_eq!(IEEE754_F32_ONE_BITS, 0x3F80_0000);
        assert_eq!(IEEE754_F32_TO_FP16_MANTISSA_SHIFT, 13);
        assert_eq!(IEEE754_FP16_SIGN_MASK, 0x8000);
        assert_eq!(IEEE754_FP16_ABS_MASK, 0x7FFF);
        assert_eq!(IEEE754_FP16_MANTISSA_MASK, 0x03FF);
        assert_eq!(IEEE754_FP16_EXP_ALL_ONES, 0x1F);
        assert_eq!(IEEE754_FP16_EXP_MASK, 0x7C00);
        assert_eq!(IEEE754_FP16_MAX_FINITE_BITS, 0x7BFF);
        assert_eq!(IEEE754_F64_MAX_NORMAL_BIASED_EXP, 2046);
        assert_eq!(IEEE754_F64_EXP_ALL_ONES, 0x7FF);
        assert_eq!(BIASED_INT_MAGIC_F32.to_bits(), 0x4B40_0000);
        assert_eq!(FP16_FINITE_MAX_F32, 65_504.0);
        assert_eq!(const_f32_pow2(0), 1.0);
        assert_eq!(const_f32_pow2(-8), 1.0 / 256.0);
        assert_eq!(const_f32_pow2(15), 32_768.0);
        assert_eq!(const_abs_f32(-3.25), 3.25);
        assert_eq!(const_abs_f32(-0.0).to_bits(), 0);
        assert_eq!(const_taylor_inv_fact_f32(4, false), 1.0_f32 / 24.0_f32);
        assert_eq!(const_taylor_inv_fact_f32(2, true), -0.5_f32);
        assert_eq!(IEEE754_F32_RCP_SEED, 0x7EF3_11C3);
        assert_eq!(const_f32_rcp_chebyshev_seed(), 0x7EF3_11C3);
        assert_eq!(IEEE754_F32_RSQRT_SEED, 0x5F37_59DF);
        assert_eq!(DC_BLOCKER_5HZ_48K_F32.to_bits(), 0x3F7F_D51F);
        assert_eq!(DC_BLOCKER_10HZ_48K_BITS, 0x3F7F_AA45);
        assert_eq!(MAX_LOOP_GAIN_F32.to_bits(), 0x3F7F_BE76);
        assert_eq!(MAX_PARALRU_R_F32.to_bits(), 0x3F7F_FFAC);
        assert_eq!(MAX_ALG_R, 4_096.0);
        assert_eq!(G_PRE_MAX, 2.5);
        assert_eq!(G_PRE_E_MAX, 1.0e6);
        assert_eq!(SE_BUS_ABS_MAX, 1.0e6);
        assert_eq!(MAX_DUHEM_FLUX_ABS, 1.0e18);

        assert_eq!(
            DC_BLOCKER_5HZ_48K_F32.to_bits(),
            dc_pole_f64_oracle(5, 48_000)
        );
        assert_eq!(DC_BLOCKER_10HZ_48K_BITS, dc_pole_f64_oracle(10, 48_000));
        assert_eq!(const_dc_blocker_pole_bits(0, 48_000), 0x3F80_0000);
        assert_eq!(const_dc_blocker_pole_bits(12_000, 48_000), 0);
        assert_eq!(const_dc_blocker_pole_bits(5, 0), 0);
        assert_eq!(
            const_dc_blocker_pole_bits(10_000, 48_000),
            dc_pole_f64_oracle(10_000, 48_000)
        );

        assert_eq!(const_max_f32_bits_le_ratio(1, 0), 0x3F80_0000);
        assert_eq!(const_max_f32_bits_le_ratio(5, 4), 0x3F80_0000);
        assert_eq!(const_max_f32_bits_le_ratio(0, 4), 0);
        assert_eq!(const_max_f32_bits_le_ratio(1, 2), 0x3F00_0000);
        assert_eq!(const_max_f32_bits_le_ratio(1, 4), 0x3E80_0000);
        assert_eq!(const_max_f32_bits_le_ratio(1, 3), 0x3EAA_AAAA);

        assert_eq!(DC_BLOCKER_5HZ_48K_F32, 0.999_345_7_f32);
        assert!(verify_tier_spec(&LITE_SPEC, 4, 1, 32, 8, 1));
        assert!(verify_tier_spec(&FULL_SPEC, 6, 2, 48, 12, 2));
        assert!(verify_tier_spec(&PRO_SPEC, 8, 3, 96, 24, 2));
        assert!(verify_tier_spec(&ULTRA_SPEC, 10, 4, 128, 32, 2));
        assert_eq!(LITE_SPEC.svf_quads, 4);
        assert_eq!(FULL_SPEC.svf_quads, 6);
        assert_eq!(PRO_SPEC.svf_quads, 8);
        assert_eq!(ULTRA_SPEC.svf_quads, 10);
        assert_eq!(LITE_SPEC.total_groups(), 4);
        assert_eq!(FULL_SPEC.total_groups(), 6);
        assert_eq!(PRO_SPEC.total_groups(), 8);
        assert_eq!(ULTRA_SPEC.total_groups(), 10);
        assert_eq!(LITE_SPEC.tank_quads(), 1);
        assert_eq!(FULL_SPEC.tank_quads(), 2);
        assert_eq!(PRO_SPEC.tank_quads(), 3);
        assert_eq!(ULTRA_SPEC.tank_quads(), 4);
        assert_eq!(LITE_SPEC.lanes(), 16);
        assert_eq!(FULL_SPEC.lanes(), 24);
        assert_eq!(PRO_SPEC.lanes(), 32);
        assert_eq!(ULTRA_SPEC.lanes(), 40);
        assert!(!is_matryoshka_monotonic(&FULL_SPEC, &LITE_SPEC));
    }

    proptest! {
        #[test]
        fn prop_dc_blocker_pole_matches_f64_oracle(
            fc_hz in 1_u32..=11_000_u32,
            fs_hz in 44_100_u32..=384_000_u32,
        ) {
            prop_assert_eq!(
                const_dc_blocker_pole_bits(fc_hz, fs_hz),
                dc_pole_f64_oracle(fc_hz, fs_hz)
            );
        }

        #[test]
        fn prop_max_f32_bits_le_ratio_is_tight_lower_bound(
            den in 100_u32..=1_000_000_u32,
            frac in 1_u32..1_000_000_u32,
        ) {
            let num = (frac % (den - 1)) + 1;
            let bits = const_max_f32_bits_le_ratio(num, den);
            let val = f64::from(f32::from_bits(bits));
            let next = f64::from(f32::from_bits(bits + 1));
            let target = f64::from(num) / f64::from(den);
            prop_assert!(val <= target);
            prop_assert!(next > target);
        }
    }
}
