//! Exact-Integer Sufficient-Statistic Accumulators (`9-word S` + `18-word D`,
//! `ref[dsp.ctrl.accum.exact-int]`, zero heap allocation, zero division).

use core::cmp::Ordering;

use crate::types::VmError;

/// Number of `u32` limbs in an exact-integer linear sum `S` (`9 × 4 B = 36 B`).
pub const EXACT_SUM_WORDS: usize = 9;
/// Number of `u32` limbs in an exact-integer pairwise interaction sum `D` (`18 × 4 B = 72 B`).
pub const EXACT_SQUARE_WORDS: usize = 18;
/// Maximum contributors supported in one fixed inline accumulator (`512`).
pub const MAX_EXACT_CONTRIBUTORS: usize = 512;

/// Fixed-width unsigned multi-word magnitude (`N` 32-bit little-endian limbs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactMagnitude<const N: usize> {
    /// Little-endian 32-bit limbs (`words[0]` is LSB).
    pub words: [u32; N],
}

impl<const N: usize> Default for ExactMagnitude<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> ExactMagnitude<N> {
    /// Returns an all-zero `N`-limb unsigned magnitude.
    #[inline]
    pub const fn zero() -> Self {
        Self { words: [0; N] }
    }

    /// Compares two unsigned magnitudes from most-significant limb downward.
    #[inline]
    pub fn compare(&self, other: &Self) -> Ordering {
        for i in (0..N).rev() {
            match self.words[i].cmp(&other.words[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// Adds `other` in place, returning `false` on overflow instead of panicking.
    #[inline]
    pub fn add_assign_checked(&mut self, other: &Self) -> bool {
        let mut carry = 0_u64;
        for i in 0..N {
            let total = u64::from(self.words[i]) + u64::from(other.words[i]) + carry;
            self.words[i] = total as u32;
            carry = total >> 32;
        }
        carry == 0
    }

    /// Subtracts `other <= self` in place, returning `false` if borrow escapes.
    #[inline]
    pub fn sub_assign_checked(&mut self, other: &Self) -> bool {
        let mut borrow = false;
        for i in 0..N {
            let (first, b1) = self.words[i].overflowing_sub(other.words[i]);
            let (res, b2) = first.overflowing_sub(u32::from(borrow));
            self.words[i] = res;
            borrow = b1 || b2;
        }
        !borrow
    }

    #[inline]
    fn bit(&self, pos: usize) -> bool {
        let word = pos >> 5;
        word < N && (self.words[word] & (1_u32 << (pos & 31))) != 0
    }

    #[inline]
    fn any_bits_below(&self, limit: usize) -> bool {
        let full = limit >> 5;
        if self.words[..full.min(N)].iter().any(|&w| w != 0) {
            return true;
        }
        let rem = limit & 31;
        full < N && rem != 0 && (self.words[full] & ((1_u32 << rem) - 1)) != 0
    }

    #[inline]
    fn highest_bit(&self) -> Option<usize> {
        for i in (0..N).rev() {
            let w = self.words[i];
            if w != 0 {
                return Some((i << 5) + 31 - (w.leading_zeros() as usize));
            }
        }
        None
    }

    #[inline]
    fn rounded_shift_24(&self, shift: usize) -> u32 {
        let word = shift >> 5;
        if word >= N {
            return 0;
        }
        let low = u64::from(self.words[word]);
        let high = self.words.get(word + 1).map_or(0, |&v| u64::from(v));
        let top = (((low | (high << 32)) >> (shift & 31)) & 0x00ff_ffff) as u32;
        if shift == 0 {
            return top;
        }
        let guard = self.bit(shift - 1);
        let sticky = self.any_bits_below(shift - 1);
        top + u32::from(guard && (sticky || (top & 1) != 0))
    }
}

/// Signed multi-word integer `(negative, ExactMagnitude<N>)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedExactInt<const N: usize> {
    /// Sign flag (`true` when strictly negative).
    pub negative: bool,
    /// Unsigned limb magnitude.
    pub magnitude: ExactMagnitude<N>,
}

impl<const N: usize> SignedExactInt<N> {
    /// Returns exact zero (`+0`).
    #[inline]
    pub const fn zero() -> Self {
        Self {
            negative: false,
            magnitude: ExactMagnitude::zero(),
        }
    }

    /// Accumulates `(negative, magnitude)` into `self`, returning `false` on limb overflow.
    pub fn add_signed_checked(&mut self, negative: bool, magnitude: ExactMagnitude<N>) -> bool {
        if magnitude.highest_bit().is_none() {
            return true;
        }
        if self.negative == negative {
            self.magnitude.add_assign_checked(&magnitude)
        } else {
            match self.magnitude.compare(&magnitude) {
                Ordering::Greater => self.magnitude.sub_assign_checked(&magnitude),
                Ordering::Equal => {
                    *self = Self::zero();
                    true
                }
                Ordering::Less => {
                    let mut diff = magnitude;
                    let ok = diff.sub_assign_checked(&self.magnitude);
                    self.magnitude = diff;
                    self.negative = negative;
                    ok
                }
            }
        }
    }
}

impl SignedExactInt<EXACT_SUM_WORDS> {
    /// Rounds a 9-word linear sum ($S \cdot 2^{-149}$) to IEEE-754 `f32` bits (`RNE`).
    pub fn to_rounded_f32_bits(&self) -> Option<u32> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Some(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 23 {
            return Some(sign | self.magnitude.words[0]);
        }
        let mut rounded = self.magnitude.rounded_shift_24(highest - 23);
        if rounded == (1 << 24) {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = (highest as i32) - 149;
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

impl SignedExactInt<EXACT_SQUARE_WORDS> {
    /// Rounds an 18-word interaction sum ($z_{\text{fm}} = D \cdot 2^{-299}$) to `f32` bits (`RNE`).
    pub fn to_interaction_f32_bits(&self) -> Option<u32> {
        let Some(mut highest) = self.magnitude.highest_bit() else {
            return Some(0);
        };
        let sign = u32::from(self.negative) << 31;
        if highest < 173 {
            return Some(sign | self.magnitude.rounded_shift_24(150));
        }
        let mut rounded = self.magnitude.rounded_shift_24(highest - 23);
        if rounded == (1 << 24) {
            rounded >>= 1;
            highest += 1;
        }
        let exponent = (highest as i32) - 299;
        if exponent > 127 {
            return None;
        }
        Some(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
    }
}

#[inline]
fn decode_f32_significand_shift(bits: u32) -> Option<(bool, u32, usize)> {
    let negative = (bits >> 31) != 0;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return None;
    }
    let (sig, shift) = if exponent == 0 {
        (fraction, 0usize)
    } else {
        ((1 << 23) | fraction, (exponent - 1) as usize)
    };
    Some((negative, sig, shift))
}

#[inline]
fn decode_f32_to_9word(bits: u32) -> Option<(bool, ExactMagnitude<EXACT_SUM_WORDS>)> {
    let (neg, sig, shift) = decode_f32_significand_shift(bits)?;
    let mut mag = ExactMagnitude::zero();
    let idx = shift >> 5;
    let shifted = u64::from(sig) << (shift & 31);
    mag.words[idx] = shifted as u32;
    mag.words[idx + 1] = (shifted >> 32) as u32;
    Some((neg, mag))
}

fn sparse_product_18word(
    value: &ExactMagnitude<EXACT_SUM_WORDS>,
    significand: u32,
    shift: usize,
) -> ExactMagnitude<EXACT_SQUARE_WORDS> {
    let mut prod = [0_u32; EXACT_SUM_WORDS + 1];
    let mut carry = 0_u64;
    for i in 0..EXACT_SUM_WORDS {
        let total = u64::from(value.words[i]) * u64::from(significand) + carry;
        prod[i] = total as u32;
        carry = total >> 32;
    }
    prod[EXACT_SUM_WORDS] = carry as u32;
    let mut out = ExactMagnitude::zero();
    let word_off = shift >> 5;
    let bit_off = shift & 31;
    for (i, &w) in prod.iter().enumerate() {
        let dst = i + word_off;
        if dst + 1 < EXACT_SQUARE_WORDS {
            let shifted = u64::from(w) << bit_off;
            out.words[dst] |= shifted as u32;
            out.words[dst + 1] |= (shifted >> 32) as u32;
        }
    }
    out
}

/// Exact-integer `9-word` linear sum accumulator for one latent coordinate (`ref[dsp.ctrl.accum.exact-int]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExactIntMainAccumulator {
    /// Running exact signed sum $S \cdot 2^{-149}$.
    pub sum: SignedExactInt<EXACT_SUM_WORDS>,
}

impl ExactIntMainAccumulator {
    /// Creates a zeroed `9-word` main sum accumulator.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            sum: SignedExactInt::zero(),
        }
    }

    /// Resets the running linear sum to exact zero.
    #[inline]
    pub fn reset(&mut self) {
        self.sum = SignedExactInt::zero();
    }

    /// Adds one `binary32` bit pattern cleanly into the `9-word` exact sum.
    #[inline]
    pub fn add_f32_bits_rt(&mut self, bits: u32) -> bool {
        let Some((neg, sig, _)) = decode_f32_significand_shift(bits) else {
            return false;
        };
        if sig == 0 {
            return true;
        }
        let Some((_, mag)) = decode_f32_to_9word(bits) else {
            return false;
        };
        self.sum.add_signed_checked(neg, mag)
    }

    /// Rounds the exact `9-word` sum to IEEE-754 `f32` (`RNE`).
    #[inline]
    pub fn to_f32(&self) -> f32 {
        self.sum
            .to_rounded_f32_bits()
            .map(f32::from_bits)
            .unwrap_or(0.0)
    }
}

/// Exact-integer sufficient-statistic accumulator for one FM factor (`9-word S` + `18-word D`,
/// zero heap allocation, zero `Drop` glue).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactIntFactorAccumulator {
    /// Active contributor slot count (`<= MAX_EXACT_CONTRIBUTORS`).
    pub count: usize,
    /// Last applied `binary32` contribution per slot (`b_old * 2^-149`).
    pub previous_bits: [u32; MAX_EXACT_CONTRIBUTORS],
    /// Exact `9-word` linear sum $S$.
    pub sum: SignedExactInt<EXACT_SUM_WORDS>,
    /// Exact `18-word` pairwise sum $D = 2 \sum_{i<j} b_i b_j$.
    pub twice_pairs: SignedExactInt<EXACT_SQUARE_WORDS>,
}

impl ExactIntFactorAccumulator {
    /// Constructs a zeroed accumulator for up to `count <= MAX_EXACT_CONTRIBUTORS` contributors.
    #[inline]
    pub fn new(count: usize) -> Self {
        Self {
            count: count.min(MAX_EXACT_CONTRIBUTORS),
            previous_bits: [0; MAX_EXACT_CONTRIBUTORS],
            sum: SignedExactInt::zero(),
            twice_pairs: SignedExactInt::zero(),
        }
    }

    /// Resets the accumulator in place with zero heap allocation.
    #[inline]
    pub fn reset(&mut self) {
        self.previous_bits[..self.count].fill(0);
        self.sum = SignedExactInt::zero();
        self.twice_pairs = SignedExactInt::zero();
    }

    /// Real-time $O(1)$ incremental update of contributor `index` to `new_bits`.
    #[inline]
    pub fn update_contributor_rt(&mut self, index: usize, new_bits: u32) -> bool {
        if index >= self.count {
            return false;
        }
        let Some((new_neg, new_sig, new_shift)) = decode_f32_significand_shift(new_bits) else {
            return false;
        };
        let old_bits = self.previous_bits[index];
        if old_bits == new_bits || ((old_bits & 0x7fff_ffff) == 0 && new_sig == 0) {
            self.previous_bits[index] = new_bits;
            return true;
        }
        let Some((old_neg, old_sig, old_shift)) = decode_f32_significand_shift(old_bits) else {
            return false;
        };
        if old_sig != 0
            && let Some((_, old_mag)) = decode_f32_to_9word(old_bits)
        {
            if !self.sum.add_signed_checked(!old_neg, old_mag) {
                return false;
            }
            if self.sum.magnitude.words != [0; EXACT_SUM_WORDS] {
                let rest_neg = self.sum.negative;
                if !self.twice_pairs.add_signed_checked(
                    !(old_neg ^ rest_neg),
                    sparse_product_18word(&self.sum.magnitude, old_sig, old_shift + 1),
                ) {
                    return false;
                }
            }
        }
        if new_sig != 0
            && let Some((_, new_mag)) = decode_f32_to_9word(new_bits)
        {
            if self.sum.magnitude.words != [0; EXACT_SUM_WORDS] {
                let rest_neg = self.sum.negative;
                if !self.twice_pairs.add_signed_checked(
                    new_neg ^ rest_neg,
                    sparse_product_18word(&self.sum.magnitude, new_sig, new_shift + 1),
                ) {
                    return false;
                }
            }
            if !self.sum.add_signed_checked(new_neg, new_mag) {
                return false;
            }
        }
        self.previous_bits[index] = new_bits;
        true
    }

    /// Checked update of contributor `index`, returning [`VmError::InvalidExactAccumulatorInput`] on invalid input.
    #[inline]
    pub fn update_contributor(&mut self, index: usize, new_bits: u32) -> Result<(), VmError> {
        if self.update_contributor_rt(index, new_bits) {
            Ok(())
        } else {
            Err(VmError::InvalidExactAccumulatorInput)
        }
    }

    /// Returns rounded IEEE-754 `f32` bits of $z_{\text{fm}} = D \cdot 2^{-299}$.
    #[inline]
    pub fn interaction_f32_bits(&self) -> Option<u32> {
        self.twice_pairs.to_interaction_f32_bits()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_exact_int_cancellation_and_order_invariance() {
        let mut a = ExactIntFactorAccumulator::new(4);
        let mut b = ExactIntFactorAccumulator::new(4);
        let pos = (1.0e18_f32).to_bits();
        let neg = (-1.0e18_f32).to_bits();
        let x2 = (0.5_f32).to_bits();
        let x3 = (0.25_f32).to_bits();
        a.update_contributor(0, pos).unwrap();
        a.update_contributor(1, neg).unwrap();
        a.update_contributor(2, x2).unwrap();
        a.update_contributor(3, x3).unwrap();

        b.update_contributor(3, x3).unwrap();
        b.update_contributor(2, x2).unwrap();
        b.update_contributor(1, neg).unwrap();
        b.update_contributor(0, pos).unwrap();

        assert_eq!(a.sum, b.sum);
        assert_eq!(a.twice_pairs, b.twice_pairs);
        assert_eq!(a.interaction_f32_bits(), b.interaction_f32_bits());

        // Sweeping back to zero returns exact 0-ULP +0.0!
        a.update_contributor(0, 0).unwrap();
        a.update_contributor(1, 0).unwrap();
        a.update_contributor(2, 0).unwrap();
        a.update_contributor(3, 0).unwrap();
        assert_eq!(a.interaction_f32_bits(), Some(0));

        let mut main_acc = ExactIntMainAccumulator::zero();
        assert!(main_acc.add_f32_bits_rt(pos));
        assert!(main_acc.add_f32_bits_rt(neg));
        assert!(main_acc.add_f32_bits_rt(x2));
        assert_eq!(main_acc.to_f32(), 0.5);

        // Out-of-bounds, NaN, Inf, overflow, reset, and subnormal rounding paths.
        assert_eq!(
            a.update_contributor(99, 0).unwrap_err(),
            VmError::InvalidExactAccumulatorInput
        );
        assert_eq!(
            a.update_contributor(0, f32::NAN.to_bits()).unwrap_err(),
            VmError::InvalidExactAccumulatorInput
        );
        assert_eq!(
            a.update_contributor(0, f32::INFINITY.to_bits())
                .unwrap_err(),
            VmError::InvalidExactAccumulatorInput
        );
        a.reset();
        assert_eq!(a.interaction_f32_bits(), Some(0));

        // Non-zero tiny values exercising low-exponent subnormal/tie rounding path (highest < 173).
        let tiny = f32::from_bits(1);
        a.update_contributor(0, tiny.to_bits()).unwrap();
        a.update_contributor(1, (1.0_f32).to_bits()).unwrap();
        assert_eq!(a.interaction_f32_bits(), Some(1));
        a.update_contributor(1, (0.25_f32).to_bits()).unwrap();
        assert_eq!(a.interaction_f32_bits(), Some(0));
        a.update_contributor(0, (1.0_f32).to_bits()).unwrap();
        a.update_contributor(1, (-0.5_f32).to_bits()).unwrap();
        assert_eq!(a.interaction_f32_bits(), Some((-0.5_f32).to_bits()));

        assert!(!main_acc.add_f32_bits_rt(f32::NAN.to_bits()));
        let mut neg_main = ExactIntMainAccumulator::zero();
        assert!(neg_main.add_f32_bits_rt((-3.0_f32).to_bits()));
        assert!(neg_main.add_f32_bits_rt((1.0_f32).to_bits()));
        assert_eq!(neg_main.to_f32(), -2.0);
        neg_main.reset();
        assert_eq!(neg_main.to_f32(), 0.0);
    }

    proptest! {
        #[test]
        fn prop_exact_int_permutation_and_zero_return_invariant(
            v in prop::array::uniform4(-1.0e12_f32..1.0e12_f32),
        ) {
            let mut fwd = ExactIntFactorAccumulator::new(4);
            let mut rev = ExactIntFactorAccumulator::new(4);
            for i in 0..4 {
                fwd.update_contributor(i, v[i].to_bits()).unwrap();
                rev.update_contributor(3 - i, v[3 - i].to_bits()).unwrap();
            }
            prop_assert_eq!(fwd.sum, rev.sum);
            prop_assert_eq!(fwd.twice_pairs, rev.twice_pairs);
            prop_assert_eq!(fwd.interaction_f32_bits(), rev.interaction_f32_bits());
            for i in 0..4 {
                fwd.update_contributor(i, 0).unwrap();
            }
            prop_assert_eq!(fwd.interaction_f32_bits(), Some(0));
        }
    }
}
