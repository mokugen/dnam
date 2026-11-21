//! Pure `u32`-limb 64-bit unsigned fixed-point and `Q60` arithmetic helpers used strictly by
//! compile-time mathematical evaluations (`ref[repo.code.constant-eval-and-derivations]`,
//! `ref[sdk.rust.shared-nostd-kernels]`).
//!
//! Uses zero `u64`/`i64`/`f64`/`u128`/`i128` types in production code so `dnam-kernels` stays
//! 32-bit `spirv-unknown-naga-wgsl`-clean.

/// 64-bit unsigned integer represented as two 32-bit limbs (`hi * 2^32 + lo`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct U64Limb {
    pub(super) hi: u32,
    pub(super) lo: u32,
}

impl U64Limb {
    pub(super) const ZERO: Self = Self { hi: 0, lo: 0 };

    #[inline(always)]
    pub(super) const fn from_u32(v: u32) -> Self {
        Self { hi: 0, lo: v }
    }

    #[inline(always)]
    pub(super) const fn add(self, rhs: Self) -> Self {
        let (lo, carry) = self.lo.overflowing_add(rhs.lo);
        let hi = self.hi.wrapping_add(rhs.hi).wrapping_add(carry as u32);
        Self { hi, lo }
    }

    #[inline(always)]
    pub(super) const fn sub(self, rhs: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let hi = self.hi.wrapping_sub(rhs.hi).wrapping_sub(borrow as u32);
        Self { hi, lo }
    }

    #[inline(always)]
    pub(super) const fn ge(self, rhs: Self) -> bool {
        self.hi > rhs.hi || (self.hi == rhs.hi && self.lo >= rhs.lo)
    }

    #[inline(always)]
    pub(super) const fn shl1(self) -> Self {
        Self {
            hi: (self.hi << 1) | (self.lo >> 31),
            lo: self.lo << 1,
        }
    }

    #[inline(always)]
    pub(super) const fn shl(self, shift: u32) -> Self {
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

    pub(super) const fn div_rem(self, divisor: Self) -> (Self, Self) {
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
pub(super) const fn mul_u32_wide(a: u32, b: u32) -> U64Limb {
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

pub(super) const fn mul_q60(a: U64Limb, b: U64Limb) -> U64Limb {
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

pub(super) const fn div_fixed_q60(num: U64Limb, den: U64Limb) -> U64Limb {
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

pub(super) const fn mul_u64_u32_div_u32(a: U64Limb, num: u32, den: u32) -> U64Limb {
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

pub(super) const ONE_Q60: U64Limb = U64Limb {
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

/// Evaluates $\pi$ in `Q60` fixed point via Machin's formula (`4 * atan(1/5) - atan(1/239)`).
pub(super) const fn pi_q60() -> U64Limb {
    let atan5 = atan_inv_q60(5);
    let atan239 = atan_inv_q60(239);
    atan5.shl(4).sub(atan239.shl(2))
}

#[cfg(test)]
mod tests {
    use core::hint::black_box;

    use super::*;

    #[test]
    fn test_u64_limb_and_q60_fixed_point_runtime() {
        let limb_a = black_box(U64Limb {
            hi: 1,
            lo: 0xFFFF_FFFF,
        });
        let limb_b = black_box(U64Limb { hi: 0, lo: 2 });
        let added = limb_a.add(limb_b);
        assert_eq!(added, U64Limb { hi: 2, lo: 1 });
        assert_eq!(added.sub(limb_b), limb_a);
        assert!(added.ge(limb_a));
        assert!(!limb_b.ge(limb_a));
        assert_eq!(U64Limb::from_u32(1).shl(0), U64Limb::from_u32(1));
        assert_eq!(U64Limb::from_u32(1).shl(16), U64Limb::from_u32(1 << 16));
        assert_eq!(U64Limb::from_u32(1).shl(32), U64Limb { hi: 1, lo: 0 });
        assert_eq!(U64Limb::from_u32(1).shl(64), U64Limb::ZERO);

        let mut wide = [u32::MAX, u32::MAX, u32::MAX, 0];
        add_wide_at(&mut wide, 0, U64Limb::from_u32(1));
        assert_eq!(wide, [0, 0, 0, 1]);

        let half_q60 = ONE_Q60.div_rem(U64Limb::from_u32(2)).0;
        assert_eq!(mul_q60(black_box(ONE_Q60), half_q60), half_q60);
        assert_eq!(div_fixed_q60(black_box(half_q60), ONE_Q60), half_q60);
        assert_eq!(mul_u64_u32_div_u32(black_box(ONE_Q60), 3, 6), half_q60);
        let pi_hi = pi_q60().hi;
        assert!(pi_hi > 3 * ONE_Q60.hi);
    }
}
