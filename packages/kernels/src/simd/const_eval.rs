//! Compile-time mathematical evaluation, IEEE-754 seed derivations, and fixed-shape `SONA` tier
//! specifications (`ref[repo.code.constant-eval-and-derivations]`, `ref[repo.code.shared-nostd-kernels]`).
//!
//! All constants derive from pure `u32` fixed-point `const fn` routines (zero `f64`, `u64`, `i64`,
//! `u128`, `i128`, or `libm` dependencies) so this module compiles identically on `#![no_std]` CPU
//! targets and 32-bit `spirv-unknown-naga-wgsl` shaders.

/// IEEE-754 `binary32` exponent bias (`127`).
pub const IEEE754_F32_EXP_BIAS: u32 = 127;

/// Chebyshev minimax `binary32` positive reciprocal integer-sub seed
/// `((2 * E_bias - 1) << 23) + round((2 * sqrt(6) - 4) * 2^23) == 0x7EF3_11C3`
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
pub const IEEE754_F32_RCP_SEED: u32 = const_f32_rcp_chebyshev_seed();

/// Robertson/Lomont minimax `binary32` global two-binade inverse-square-root seed
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
pub const IEEE754_F32_RSQRT_SEED: u32 = 0x5F37_59DF;

/// Compile-time `5 Hz` passband-normalized DC blocker pole at `48 kHz` (`0x3F7F_D51F`,
/// `ref[dsp.stage4.dc-blocker]`).
pub const DC_BLOCKER_5HZ_48K_F32: f32 = f32::from_bits(const_dc_blocker_pole_bits(5, 48_000));

/// Compile-time `10 Hz` passband-normalized DC blocker pole bit pattern at `48 kHz` (`0x3F7F_AA45`).
pub const DC_BLOCKER_10HZ_48K_BITS: u32 = const_dc_blocker_pole_bits(10, 48_000);

/// Largest `binary32` value `<= 0.9990` (`0x3F7F_BE76`, `ref[dsp.stage4.quiet-feedback-knee]`).
pub const MAX_LOOP_GAIN_F32: f32 = f32::from_bits(const_max_f32_bits_le_ratio(999, 1_000));

/// Largest `binary32` value `<= 0.999995` (`0x3F7F_FFAC`, upper `ParaLRU` and ZOH pole bound).
pub const MAX_PARALRU_R_F32: f32 = f32::from_bits(const_max_f32_bits_le_ratio(999_995, 1_000_000));

/// Compile-time fixed-shape tier capacity profile locked at `Gate 0b` (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SonaTierSpec {
    /// Number of 4-lane `Vec4` quads in each of the 4 Matryoshka `SONA` core blocks.
    pub blocks: [usize; 4],
    /// Number of 4-mode `TPT-SVF` `Vec4` quads in the `SONA` core.
    pub svf_quads: usize,
    /// Number of Stage III early-reflection FIR taps per stereo channel.
    pub fir_taps: usize,
    /// Number of Stage III shared conjugate Kautz modal pole pairs.
    pub kautz_pairs: usize,
    /// Number of `Op-T` transport delay lines (`M` lines = `4 * quads`).
    pub opt_lines: usize,
    /// Core oversampling factor (`1` for `Lite`, `2` for `Full`/`Pro`/`Ultra` real-time).
    pub oversample_factor: usize,
}

impl SonaTierSpec {
    /// Returns the total number of 4-lane `Vec4` quads across all 4 Matryoshka `SONA` core blocks.
    #[inline(always)]
    pub const fn block_sum(&self) -> usize {
        self.blocks[0] + self.blocks[1] + self.blocks[2] + self.blocks[3]
    }

    /// Returns the total number of scalar `SONA` core lanes (`4 * self.block_sum()`).
    #[inline(always)]
    pub const fn total_lanes(&self) -> usize {
        4 * self.block_sum()
    }
}

/// `Lite` compile-time tier capacity profile (`[1, 1, 1, 1]`, `16` lanes, `4` SVF, `32` FIR, `8` Kautz).
pub const LITE_SPEC: SonaTierSpec = SonaTierSpec {
    blocks: [1, 1, 1, 1],
    svf_quads: 1,
    fir_taps: 32,
    kautz_pairs: 8,
    opt_lines: 4,
    oversample_factor: 1,
};

/// `Full` compile-time tier capacity profile (`[2, 2, 1, 1]`, `24` lanes @ `2x`, `48` FIR, `12` Kautz).
pub const FULL_SPEC: SonaTierSpec = SonaTierSpec {
    blocks: [2, 2, 1, 1],
    svf_quads: 2,
    fir_taps: 48,
    kautz_pairs: 12,
    opt_lines: 8,
    oversample_factor: 2,
};

/// `Pro` compile-time tier capacity profile (`[3, 2, 2, 1]`, `32` lanes @ `2x`, `96` FIR, `24` Kautz).
pub const PRO_SPEC: SonaTierSpec = SonaTierSpec {
    blocks: [3, 2, 2, 1],
    svf_quads: 2,
    fir_taps: 96,
    kautz_pairs: 24,
    opt_lines: 12,
    oversample_factor: 2,
};

/// `Ultra` compile-time tier capacity profile (`[4, 3, 2, 1]`, `40` lanes @ `2x`, `128` FIR, `32` Kautz).
pub const ULTRA_SPEC: SonaTierSpec = SonaTierSpec {
    blocks: [4, 3, 2, 1],
    svf_quads: 3,
    fir_taps: 128,
    kautz_pairs: 32,
    opt_lines: 16,
    oversample_factor: 2,
};

// Pure `u32`-limb 64-bit unsigned fixed-point helpers (zero `u64`/`f64` for `spirv` cleanliness).

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct U64Limb {
    hi: u32,
    lo: u32,
}

impl U64Limb {
    const ZERO: Self = Self { hi: 0, lo: 0 };

    #[inline(always)]
    const fn from_u32(v: u32) -> Self {
        Self { hi: 0, lo: v }
    }

    #[inline(always)]
    const fn add(self, rhs: Self) -> Self {
        let (lo, carry) = self.lo.overflowing_add(rhs.lo);
        let hi = self.hi.wrapping_add(rhs.hi).wrapping_add(carry as u32);
        Self { hi, lo }
    }

    #[inline(always)]
    const fn sub(self, rhs: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let hi = self.hi.wrapping_sub(rhs.hi).wrapping_sub(borrow as u32);
        Self { hi, lo }
    }

    #[inline(always)]
    const fn ge(self, rhs: Self) -> bool {
        self.hi > rhs.hi || (self.hi == rhs.hi && self.lo >= rhs.lo)
    }

    #[inline(always)]
    const fn shl1(self) -> Self {
        Self {
            hi: (self.hi << 1) | (self.lo >> 31),
            lo: self.lo << 1,
        }
    }

    #[inline(always)]
    const fn shl(self, shift: u32) -> Self {
        if shift == 0 {
            self
        } else if shift < 32 {
            Self {
                hi: (self.hi << shift) | (self.lo >> (32 - shift)),
                lo: self.lo << shift,
            }
        } else if shift < 64 {
            Self {
                hi: self.lo << (shift - 32),
                lo: 0,
            }
        } else {
            Self::ZERO
        }
    }

    const fn div_rem(self, divisor: Self) -> (Self, Self) {
        let mut q = Self::ZERO;
        let mut r = Self::ZERO;
        let mut i: u32 = 64;
        while i > 0 {
            i -= 1;
            let bit = if i >= 32 {
                (self.hi >> (i - 32)) & 1
            } else {
                (self.lo >> i) & 1
            };
            r = r.shl1();
            r.lo |= bit;
            if r.ge(divisor) {
                r = r.sub(divisor);
                if i >= 32 {
                    q.hi |= 1_u32 << (i - 32);
                } else {
                    q.lo |= 1_u32 << i;
                }
            }
        }
        (q, r)
    }
}

#[inline(always)]
const fn mul_u32_wide(a: u32, b: u32) -> U64Limb {
    let a_lo = a & 0xFFFF;
    let a_hi = a >> 16;
    let b_lo = b & 0xFFFF;
    let b_hi = b >> 16;
    let p00 = a_lo * b_lo;
    let p01 = a_lo * b_hi;
    let p10 = a_hi * b_lo;
    let p11 = a_hi * b_hi;
    let t0 = U64Limb { hi: 0, lo: p00 };
    let t1 = U64Limb {
        hi: p01 >> 16,
        lo: p01 << 16,
    };
    let t2 = U64Limb {
        hi: p10 >> 16,
        lo: p10 << 16,
    };
    let t3 = U64Limb { hi: p11, lo: 0 };
    t0.add(t1).add(t2).add(t3)
}

const fn add_wide_at(acc: &mut [u32; 4], idx: usize, val: U64Limb) {
    let (s0, c0) = acc[idx].overflowing_add(val.lo);
    acc[idx] = s0;
    let (s1, c1) = acc[idx + 1].overflowing_add(val.hi);
    let (s1b, c2) = s1.overflowing_add(c0 as u32);
    acc[idx + 1] = s1b;
    let mut carry = (c1 as u32) + (c2 as u32);
    let mut k = idx + 2;
    while carry != 0 && k < 4 {
        let (sk, ck) = acc[k].overflowing_add(carry);
        acc[k] = sk;
        carry = ck as u32;
        k += 1;
    }
}

const fn mul_q60(a: U64Limb, b: U64Limb) -> U64Limb {
    let mut acc = [0_u32; 4];
    add_wide_at(&mut acc, 0, mul_u32_wide(a.lo, b.lo));
    add_wide_at(&mut acc, 1, mul_u32_wide(a.lo, b.hi));
    add_wide_at(&mut acc, 1, mul_u32_wide(a.hi, b.lo));
    add_wide_at(&mut acc, 2, mul_u32_wide(a.hi, b.hi));
    U64Limb {
        hi: (acc[2] >> 28) | (acc[3] << 4),
        lo: (acc[1] >> 28) | (acc[2] << 4),
    }
}

const fn div_fixed_q60(num: U64Limb, den: U64Limb) -> U64Limb {
    let mut q = U64Limb::ZERO;
    let mut r = num;
    let mut i: u32 = 60;
    while i > 0 {
        i -= 1;
        r = r.shl1();
        if r.ge(den) {
            r = r.sub(den);
            if i >= 32 {
                q.hi |= 1_u32 << (i - 32);
            } else {
                q.lo |= 1_u32 << i;
            }
        }
    }
    q
}

const fn mul_u64_u32_div_u32(a: U64Limb, num: u32, den: u32) -> U64Limb {
    let lo_prod = mul_u32_wide(a.lo, num);
    let hi_prod = mul_u32_wide(a.hi, num);
    let (mid, carry) = lo_prod.hi.overflowing_add(hi_prod.lo);
    let top = hi_prod.hi.wrapping_add(carry as u32);
    let words = [lo_prod.lo, mid, top];
    let div_limb = U64Limb::from_u32(den);
    let mut q = U64Limb::ZERO;
    let mut r = U64Limb::ZERO;
    let mut i: u32 = 96;
    while i > 0 {
        i -= 1;
        let word_idx = (i >> 5) as usize;
        let bit = (words[word_idx] >> (i & 31)) & 1;
        r = r.shl1();
        r.lo |= bit;
        if r.ge(div_limb) {
            r = r.sub(div_limb);
            if i < 32 {
                q.lo |= 1_u32 << i;
            } else if i < 64 {
                q.hi |= 1_u32 << (i - 32);
            }
        }
    }
    q
}

const ONE_Q60: U64Limb = U64Limb {
    hi: 1_u32 << 28,
    lo: 0,
};

const fn atan_inv_q60(x: u32) -> U64Limb {
    let x2 = U64Limb::from_u32(x * x);
    let mut term = ONE_Q60.div_rem(U64Limb::from_u32(x)).0;
    let mut sum = term;
    let mut k: u32 = 1;
    while k <= 16 {
        term = term.div_rem(x2).0;
        let delta = term.div_rem(U64Limb::from_u32(2 * k + 1)).0;
        if delta.hi == 0 && delta.lo == 0 {
            break;
        }
        if (k & 1) == 1 {
            sum = sum.sub(delta);
        } else {
            sum = sum.add(delta);
        }
        k += 1;
    }
    sum
}

const fn pi_q60() -> U64Limb {
    let atan5 = atan_inv_q60(5);
    let atan239 = atan_inv_q60(239);
    atan5.shl(4).sub(atan239.shl(2))
}

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
    ((2 * IEEE754_F32_EXP_BIAS - 1) << 23) + mant_round
}

/// Compile-time largest positive `binary32` bit pattern `<= num / den` for `0 < num <= den`.
#[must_use]
pub const fn const_max_f32_bits_le_ratio(num: u32, den: u32) -> u32 {
    if den == 0 || num >= den {
        return 0x3F80_0000;
    }
    if num == 0 {
        return 0;
    }
    let den_64 = U64Limb::from_u32(den);
    let mut num_64 = U64Limb::from_u32(num);
    let mut exp: u32 = 126;
    while !num_64.shl1().ge(den_64) && exp > 1 {
        num_64 = num_64.shl1();
        exp -= 1;
    }
    let scaled = num_64.shl(24);
    let (q, _) = scaled.div_rem(den_64);
    (exp << 23) + (q.lo - (1_u32 << 23))
}

/// Compile-time passband-normalized DC blocker pole `R = cos(w) / (1 + sin(w))` in `binary32` bits,
/// where `w = 2 * pi * fc_hz / fs_hz` (`ref[dsp.stage4.dc-blocker]`).
#[must_use]
pub const fn const_dc_blocker_pole_bits(fc_hz: u32, fs_hz: u32) -> u32 {
    if fc_hz == 0 {
        return 0x3F80_0000;
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
    let mut exp: u32 = 126;
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
    if sig >= (1_u32 << 24) {
        sig >>= 1;
        exp += 1;
    }
    (exp << 23) | (sig & 0x007F_FFFF)
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
const _: () = assert!(LITE_SPEC.block_sum() == 4 && LITE_SPEC.total_lanes() == 16);
const _: () = assert!(FULL_SPEC.block_sum() == 6 && FULL_SPEC.total_lanes() == 24);
const _: () = assert!(PRO_SPEC.block_sum() == 8 && PRO_SPEC.total_lanes() == 32);
const _: () = assert!(ULTRA_SPEC.block_sum() == 10 && ULTRA_SPEC.total_lanes() == 40);
const _: () = assert!(is_matryoshka_monotonic(&LITE_SPEC, &FULL_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&FULL_SPEC, &PRO_SPEC));
const _: () = assert!(is_matryoshka_monotonic(&PRO_SPEC, &ULTRA_SPEC));

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
        assert_eq!(IEEE754_F32_RCP_SEED, 0x7EF3_11C3);
        assert_eq!(const_f32_rcp_chebyshev_seed(), 0x7EF3_11C3);
        assert_eq!(IEEE754_F32_RSQRT_SEED, 0x5F37_59DF);
        assert_eq!(DC_BLOCKER_5HZ_48K_F32.to_bits(), 0x3F7F_D51F);
        assert_eq!(DC_BLOCKER_10HZ_48K_BITS, 0x3F7F_AA45);
        assert_eq!(MAX_LOOP_GAIN_F32.to_bits(), 0x3F7F_BE76);
        assert_eq!(MAX_PARALRU_R_F32.to_bits(), 0x3F7F_FFAC);

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

        let limb_a = U64Limb {
            hi: 1,
            lo: 0xFFFF_FFFF,
        };
        let limb_b = U64Limb { hi: 0, lo: 2 };
        let added = limb_a.add(limb_b);
        assert_eq!(added, U64Limb { hi: 2, lo: 1 });
        assert_eq!(added.sub(limb_b), limb_a);
        assert!(added.ge(limb_a));
        assert!(!limb_b.ge(limb_a));
        assert_eq!(U64Limb::from_u32(1).shl(0), U64Limb::from_u32(1));
        assert_eq!(U64Limb::from_u32(1).shl(32), U64Limb { hi: 1, lo: 0 });
        assert_eq!(U64Limb::from_u32(1).shl(64), U64Limb::ZERO);

        let mut wide = [u32::MAX, u32::MAX, u32::MAX, 0];
        add_wide_at(&mut wide, 0, U64Limb::from_u32(1));
        assert_eq!(wide, [0, 0, 0, 1]);

        assert_eq!(LITE_SPEC.total_lanes(), 16);
        assert_eq!(FULL_SPEC.total_lanes(), 24);
        assert_eq!(PRO_SPEC.total_lanes(), 32);
        assert_eq!(ULTRA_SPEC.total_lanes(), 40);
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
