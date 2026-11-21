//! Exact 192-bit-storage Triple-Double IEEE-754 accumulator (`Wide192Accum`), `uplo='U'` packed
//! Gram indexing, canonical `DNAM-GRAM-INPUT-V3\0` hashing (`zerocopy` little-endian wire layout),
//! register-tiled contractions (`tiled`), and cross-only RHS contraction (`cross_only`)
//! (`ref[train.gram.wide192]`, `ref[train.gram.accuracy-contract]`).

pub mod cross_only;
pub mod tiled;

pub use cross_only::contract_dim_cross_only_wide192;
use sha2::{Digest, Sha256};
pub use tiled::{
    contract_256d_gram_tiled_wide192, contract_dim_gram_tiled_wide192,
    contract_dim_gram_tiled_wide192_with_backend,
};
use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::{F64 as LeF64, U32 as LeU32, U64 as LeU64};

/// Canonical frame batch size used for register-tiled Gram contractions (`ref[repo.code.constant-eval-and-derivations]`).
pub const WIDE_GRAM_BATCH_FRAMES: usize = 256;

const DEKKER_SPLIT_BITS: u32 = 27;
const DEKKER_SPLIT: f64 = ((1_u64 << DEKKER_SPLIT_BITS) + 1) as f64;

/// Minimum magnitude $2^{-450}$ where `a_lo * b_lo` remains strictly normal (`> 2^{-1022}`).
const DEKKER_SAFE_MIN: f64 = f64::from_bits(0x23D0_0000_0000_0000);
/// Maximum magnitude $2^{+450}$ where `DEKKER_SPLIT * a` cannot overflow `f64::MAX`.
const DEKKER_SAFE_MAX: f64 = f64::from_bits(0x5C10_0000_0000_0000);

const SUBNORMAL_PRESCALE_EXP: i32 = 64;
const SUBNORMAL_PRESCALE_F64: f64 = f64::from_bits(0x43F0_0000_0000_0000);
const IEEE_F64_EXP_BIAS: i32 = 1023;
const IEEE_F64_SIGN_MANT_MASK: u64 = 0x800f_ffff_ffff_ffff;
const IEEE_F64_ONE_EXP_MASK: u64 = (IEEE_F64_EXP_BIAS as u64) << 52;
const IEEE_F64_POW2_STEP: i32 = 512;

/// `f64` type tag (`2`) in the canonical `DNAM-GRAM-INPUT-V3\0` hash transcript.
const GRAM_HASH_F64_TYPE_TAG: u8 = 2;

const _: () = assert!(DEKKER_SPLIT == 134_217_729.0);
const _: () = assert!(WIDE_GRAM_BATCH_FRAMES.is_power_of_two());

/// Packed upper-triangular (`uplo = 'U'`) storage element count `dim * (dim + 1) / 2`.
#[inline]
#[must_use]
pub const fn uplo_u_packed_len(dim: usize) -> usize {
    dim * (dim + 1) / 2
}

/// Column-major packed upper-triangular index (`row <= col` canonicalized symmetrically):
/// `col * (col + 1) / 2 + row`.
#[inline]
#[must_use]
pub const fn uplo_u_index(row: usize, col: usize) -> usize {
    let (r, c) = if row <= col { (row, col) } else { (col, row) };
    (c * (c + 1)) / 2 + r
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline(always)]
fn two_prod_dekker_core(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let ca = DEKKER_SPLIT * a;
    let cb = DEKKER_SPLIT * b;
    let a_hi = ca - (ca - a);
    let a_lo = a - a_hi;
    let b_hi = cb - (cb - b);
    let b_lo = b - b_hi;
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[inline]
fn normalize_f64_mant_exp(x: f64) -> (f64, i32) {
    let (bits, bias_adj) = if x.abs() < f64::MIN_POSITIVE {
        (
            (x * SUBNORMAL_PRESCALE_F64).to_bits(),
            IEEE_F64_EXP_BIAS + SUBNORMAL_PRESCALE_EXP,
        )
    } else {
        (x.to_bits(), IEEE_F64_EXP_BIAS)
    };
    let biased_exp = ((bits >> 52) & 0x7ff) as i32;
    let mant = f64::from_bits((bits & IEEE_F64_SIGN_MANT_MASK) | IEEE_F64_ONE_EXP_MASK);
    (mant, biased_exp - bias_adj)
}

#[inline]
fn ldexp_exact_f64(mut v: f64, mut exp_shift: i32) -> f64 {
    if v == 0.0 || exp_shift == 0 {
        return v;
    }
    while exp_shift > IEEE_F64_POW2_STEP {
        v *= f64::from_bits(((IEEE_F64_EXP_BIAS + IEEE_F64_POW2_STEP) as u64) << 52);
        if !v.is_finite() {
            return v;
        }
        exp_shift -= IEEE_F64_POW2_STEP;
    }
    while exp_shift < -IEEE_F64_POW2_STEP {
        v *= f64::from_bits(((IEEE_F64_EXP_BIAS - IEEE_F64_POW2_STEP) as u64) << 52);
        if v == 0.0 {
            return 0.0;
        }
        exp_shift += IEEE_F64_POW2_STEP;
    }
    v * f64::from_bits(((IEEE_F64_EXP_BIAS + exp_shift) as u64) << 52)
}

#[cold]
#[inline(never)]
fn two_prod_dekker_extreme(a: f64, b: f64) -> (f64, f64) {
    let (am, ea) = normalize_f64_mant_exp(a);
    let (bm, eb) = normalize_f64_mant_exp(b);
    let (pm, em) = two_prod_dekker_core(am, bm);
    let e_tot = ea + eb;
    let hi = ldexp_exact_f64(pm, e_tot);
    if !hi.is_finite() {
        return (hi, 0.0);
    }
    (hi, ldexp_exact_f64(em, e_tot))
}

#[inline(always)]
fn two_prod_dekker(a: f64, b: f64) -> (f64, f64) {
    let aa = a.abs();
    let ab = b.abs();
    if (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&aa)
        && (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&ab)
    {
        two_prod_dekker_core(a, b)
    } else {
        two_prod_dekker_extreme(a, b)
    }
}

#[inline(always)]
fn round_to_odd_pair(hi: f64, lo: f64) -> f64 {
    if lo == 0.0 || hi == 0.0 || (hi.to_bits() & 1) != 0 {
        return hi;
    }
    let same_sign = (hi > 0.0) == (lo > 0.0);
    let adj_bits = if same_sign {
        hi.to_bits() + 1
    } else {
        hi.to_bits() - 1
    };
    f64::from_bits(adj_bits)
}

/// 192-bit storage (`[f64; 3]` Triple-Double, `~159`-bit significand) non-overlapping
/// Priest/Dekker expansion accumulator with Boldo-Melquiond round-to-odd finalization
/// (`ref[train.gram.wide192]`, `ref[train.gram.accuracy-contract]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wide192Accum {
    limbs: [f64; 3],
}

impl Wide192Accum {
    /// Zero-initialized `192`-bit accumulator (`+0.0`).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self { limbs: [0.0; 3] }
    }

    /// Constructs an accumulator initialized from a finite `f64` (non-finite or `-0.0` becomes `+0.0`).
    #[inline]
    #[must_use]
    pub fn from_f64(val: f64) -> Self {
        if !val.is_finite() || val == 0.0 {
            Self::zero()
        } else {
            Self {
                limbs: [val, 0.0, 0.0],
            }
        }
    }

    /// Returns the ordered `[hi, mid, lo]` expansion limbs.
    #[inline]
    #[must_use]
    pub const fn limbs(&self) -> [f64; 3] {
        self.limbs
    }

    /// Adds a finite `f64` scalar using Priest 3-limb renormalization (ignores non-finite inputs).
    #[inline(always)]
    pub fn add_f64(&mut self, val: f64) {
        if !val.is_finite() || val == 0.0 {
            return;
        }
        if self.limbs[0] == 0.0 {
            self.limbs[0] = val;
            return;
        }
        let (t0, e0) = two_sum(self.limbs[0], val);
        let (t1, e1) = two_sum(self.limbs[1], e0);
        if e1 == 0.0 && self.limbs[2] == 0.0 {
            let (r0, r1) = two_sum(t0, t1);
            self.limbs[0] = r0;
            self.limbs[1] = r1;
            return;
        }
        let (t2, sticky) = two_sum(self.limbs[2], e1);
        let t2_odd = round_to_odd_pair(t2, sticky);
        let (r0, u0) = two_sum(t0, t1);
        let (r1, r2) = two_sum(u0, t2_odd);
        let (s0, v0) = two_sum(r0, r1);
        let (s1, s2) = two_sum(v0, r2);
        self.limbs = [s0, s1, s2];
    }

    /// Adds the exact `106`-bit Dekker product `a * b` (ignores non-finite inputs).
    #[inline(always)]
    pub fn add_product(&mut self, a: f64, b: f64) {
        if !a.is_finite() || !b.is_finite() || a == 0.0 || b == 0.0 {
            return;
        }
        let (hi, lo) = two_prod_dekker(a, b);
        if hi.is_finite() {
            self.add_f64(hi);
            self.add_f64(lo);
        }
    }

    #[cold]
    #[inline(never)]
    fn add_three_way_extreme(&mut self, w: f64, a: f64, b: f64) {
        let ab = a * b;
        if ab.is_finite() && (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&ab.abs()) {
            self.add_product(w, ab);
            return;
        }
        let (mw, ew) = normalize_f64_mant_exp(w);
        let (ma, ea) = normalize_f64_mant_exp(a);
        let (mb, eb) = normalize_f64_mant_exp(b);
        let (ab_hi, ab_lo) = two_prod_dekker_core(ma, mb);
        let (t1, t2) = two_prod_dekker_core(mw, ab_hi);
        let (t3, _) = two_prod_dekker_core(mw, ab_lo);
        let e_tot = ew + ea + eb;
        let hi = ldexp_exact_f64(t1, e_tot);
        if hi.is_finite() {
            self.add_f64(hi);
            self.add_f64(ldexp_exact_f64(t2, e_tot));
            self.add_f64(ldexp_exact_f64(t3, e_tot));
        }
    }

    /// Adds the 3-way product `w * a * b` without premature intermediate underflow or overflow.
    #[inline(always)]
    pub fn add_triple_product(&mut self, w: f64, a: f64, b: f64) {
        if !w.is_finite() || !a.is_finite() || !b.is_finite() || w == 0.0 || a == 0.0 || b == 0.0 {
            return;
        }
        let wa = w * a;
        if wa.is_finite() && (DEKKER_SAFE_MIN..=DEKKER_SAFE_MAX).contains(&wa.abs()) {
            self.add_product(wa, b);
        } else {
            self.add_three_way_extreme(w, a, b);
        }
    }

    /// Merges another [`Wide192Accum`] into `self` from coarsest to finest limb.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.add_f64(other.limbs[0]);
        self.add_f64(other.limbs[1]);
        self.add_f64(other.limbs[2]);
    }

    /// Accumulates another [`Wide192Accum`] into `self` (alias for [`Self::merge`]).
    #[inline]
    pub fn add_wide(&mut self, other: &Self) {
        self.merge(other);
    }

    /// Collapses the `192`-bit expansion to the nearest IEEE-754 `f64` (`-0.0` canonicalized to
    /// `+0.0`) using Boldo-Melquiond round-to-odd on `limbs[1] + limbs[2]`.
    #[inline]
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        let (low_hi, low_lo) = two_sum(self.limbs[1], self.limbs[2]);
        let total = self.limbs[0] + round_to_odd_pair(low_hi, low_lo);
        if total == 0.0 { 0.0 } else { total }
    }
}

/// Accumulates a single weighted sample frame `(feats, target, weight)` into packed `uplo='U'`
/// Gram and 2-channel RHS `Wide192Accum` buffers (`ref[train.gram.wide192]`).
///
/// Ignores frames with non-positive/non-finite weights or any non-finite feature/target entry.
pub fn accumulate_uplo_u_wide192(
    gram: &mut [Wide192Accum],
    rhs: &mut [[Wide192Accum; 2]],
    feats: &[f64],
    target: [f64; 2],
    weight: f64,
) {
    let dim = feats.len().min(rhs.len());
    debug_assert!(gram.len() >= uplo_u_packed_len(dim));
    if !weight.is_finite()
        || weight <= 0.0
        || !target[0].is_finite()
        || !target[1].is_finite()
        || !feats[..dim].iter().all(|v| v.is_finite())
    {
        return;
    }
    for j in 0..dim {
        let fj = feats[j];
        if fj == 0.0 {
            continue;
        }
        for i in 0..=j {
            let fi = feats[i];
            if fi != 0.0 {
                gram[uplo_u_index(i, j)].add_triple_product(weight, fi, fj);
            }
        }
        if target[0] != 0.0 {
            rhs[j][0].add_triple_product(weight, fj, target[0]);
        }
        if target[1] != 0.0 {
            rhs[j][1].add_triple_product(weight, fj, target[1]);
        }
    }
}

#[inline(always)]
fn canon_f64(v: f64) -> LeF64 {
    LeF64::new(if v == 0.0 { 0.0 } else { v })
}

/// Computes the canonical `DNAM-GRAM-INPUT-V3\0` SHA-256 transcript over `f64` rows and weights
/// using `zerocopy::byteorder::little_endian` primitives (`ref[train.gram.accuracy-contract]`):
/// `"DNAM-GRAM-INPUT-V3\0" || u32le(feature_dim) || [2, 2] || ... || 0 || u64le(N)`.
#[must_use]
pub fn hash_gram_input_v3(feature_dim: usize, rows: &[&[f64]], weights: &[f64]) -> [u8; 32] {
    let n = rows.len().min(weights.len());
    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-GRAM-INPUT-V3\0");
    hasher.update(LeU32::new(feature_dim as u32).as_bytes());
    hasher.update([GRAM_HASH_F64_TYPE_TAG, GRAM_HASH_F64_TYPE_TAG]);
    for frame_idx in 0..n {
        hasher.update([1_u8]);
        hasher.update(LeU64::new(frame_idx as u64).as_bytes());
        hasher.update(canon_f64(weights[frame_idx]).as_bytes());
        let row = rows[frame_idx];
        let copy_cols = row.len().min(feature_dim);
        for &val in &row[..copy_cols] {
            hasher.update(canon_f64(val).as_bytes());
        }
        for _ in copy_cols..feature_dim {
            hasher.update(canon_f64(0.0).as_bytes());
        }
    }
    hasher.update([0_u8]);
    hasher.update(LeU64::new(n as u64).as_bytes());
    hasher.finalize().into()
}

/// Convenience wrapper hashing sampled `f32` frame rows (widened to `f64::from(x)`) into the
/// exact `DNAM-GRAM-INPUT-V3\0` transcript (`ref[train.gram.accuracy-contract]`).
#[must_use]
pub fn hash_gram_input_v3_f32(
    feature_dim: usize,
    features: &[f32],
    weights: &[f64],
    stride: usize,
) -> [u8; 32] {
    let step = stride.max(1);
    let avail = features
        .len()
        .checked_div(feature_dim)
        .map_or(0, |n| n.min(weights.len()));
    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-GRAM-INPUT-V3\0");
    hasher.update(LeU32::new(feature_dim as u32).as_bytes());
    hasher.update([GRAM_HASH_F64_TYPE_TAG, GRAM_HASH_F64_TYPE_TAG]);
    let mut sampled_idx = 0_u64;
    let mut frame_idx = 0_usize;
    while frame_idx < avail {
        hasher.update([1_u8]);
        hasher.update(LeU64::new(sampled_idx).as_bytes());
        hasher.update(canon_f64(weights[frame_idx]).as_bytes());
        let row = &features[frame_idx * feature_dim..(frame_idx + 1) * feature_dim];
        for &val in row {
            hasher.update(canon_f64(f64::from(val)).as_bytes());
        }
        sampled_idx += 1;
        frame_idx += step;
    }
    hasher.update([0_u8]);
    hasher.update(LeU64::new(sampled_idx).as_bytes());
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_wide192_cancellation_extreme_exp_tie_round_and_hash() {
        // verify[train.gram.wide192]
        // verify[train.gram.accuracy-contract]
        let mut wide = Wide192Accum::zero();
        for _ in 0..64 {
            wide.add_f64(1.0e8);
            wide.add_f64(2.0_f64.powi(-30));
            wide.add_f64(-1.0e8);
        }
        assert!((wide.to_f64() - 64.0 * 2.0_f64.powi(-30)).abs() <= 1.0e-22);

        let mut acc_hi = Wide192Accum::zero();
        let a_big = 2.0_f64.powi(996) * (1.0 + 2.0_f64.powi(-30));
        let b_tiny = 2.0_f64.powi(-996) * (1.0 - 2.0_f64.powi(-30));
        acc_hi.add_product(a_big, b_tiny);
        acc_hi.add_f64(-1.0);
        assert_eq!(acc_hi.to_f64().to_bits(), (-2.0_f64.powi(-60)).to_bits());

        let mut acc_sub = Wide192Accum::zero();
        acc_sub.add_f64(1.0);
        acc_sub.add_product(2.0_f64.powi(-1040), 2.0_f64.powi(-20));
        acc_sub.add_f64(-1.0);
        assert_eq!(acc_sub.to_f64().to_bits(), (2.0_f64.powi(-1060)).to_bits());

        let mut tie_up = Wide192Accum::zero();
        tie_up.add_f64(1.0);
        tie_up.add_f64(2.0_f64.powi(-53));
        tie_up.add_f64(2.0_f64.powi(-108));
        assert_eq!(
            tie_up.to_f64().to_bits(),
            (1.0 + 2.0_f64.powi(-52)).to_bits()
        );

        let mut tie_down = Wide192Accum::zero();
        tie_down.add_f64(1.0 + 2.0_f64.powi(-52));
        tie_down.add_f64(2.0_f64.powi(-53));
        tie_down.add_f64(-2.0_f64.powi(-108));
        assert_eq!(
            tie_down.to_f64().to_bits(),
            (1.0 + 2.0_f64.powi(-52)).to_bits()
        );

        let neg_zero = Wide192Accum::from_f64(-0.0);
        assert_eq!(neg_zero.to_f64().to_bits(), 0);

        let mut fb64 = [0.0_f64; 8];
        fb64[0] = f64::from(1.0e35_f32);
        fb64[3] = f64::from(1.0e-35_f32);
        let mut g = vec![Wide192Accum::zero(); uplo_u_packed_len(8)];
        let mut c = vec![[Wide192Accum::zero(); 2]; 8];
        accumulate_uplo_u_wide192(
            &mut g,
            &mut c,
            &fb64,
            [f64::from(1.0e35_f32), f64::from(1.0e35_f32)],
            1.0e-300,
        );
        let expected_lo = 1.0e-300_f64 * (f64::from(1.0e35_f32) * f64::from(1.0e-35_f32));
        assert!((g[uplo_u_index(0, 3)].to_f64() - expected_lo).abs() <= 1.0e-314);
        assert!((c[3][0].to_f64() - expected_lo).abs() <= 1.0e-314);
        assert!((c[3][1].to_f64() - expected_lo).abs() <= 1.0e-314);

        // Trigger add_three_way_extreme overflow-compensated normalization path (`wa` & `ab` > DEKKER_SAFE_MAX)
        let mut extreme = Wide192Accum::zero();
        extreme.add_triple_product(2.0_f64.powi(600), 2.0_f64.powi(600), 2.0_f64.powi(-700));
        assert!((extreme.to_f64() - 2.0_f64.powi(500)).abs() <= 2.0_f64.powi(450));

        // Short row zero-padding in hash_gram_input_v3 matches explicit trailing 0.0
        let h_short = hash_gram_input_v3(2, &[&[-0.0_f64]], &[1.0]);
        let h_full = hash_gram_input_v3(2, &[&[0.0_f64, 0.0_f64]], &[1.0]);
        assert_eq!(h_short, h_full);

        let r0 = [0.0_f64, 1.0];
        let r0_neg = [-0.0_f64, 1.0];
        let h_pos = hash_gram_input_v3(2, &[&r0], &[1.0]);
        let h_neg = hash_gram_input_v3(2, &[&r0_neg], &[1.0]);
        let h_f32 = hash_gram_input_v3_f32(2, &[-0.0_f32, 1.0], &[1.0], 1);
        assert_eq!(h_pos, h_neg);
        assert_eq!(h_pos, h_f32);
        assert_ne!(
            h_pos,
            hash_gram_input_v3_f32(2, &[-0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 1.0, 1.0], 2)
        );
    }

    #[test]
    fn test_wide192_nonfinite_rejection_merge_and_uplo_symmetry() {
        let mut a = Wide192Accum::from_f64(3.0);
        let b = Wide192Accum::from_f64(-1.0);
        a.merge(&b);
        assert_eq!(a.to_f64(), 2.0);
        assert_eq!(a.limbs()[0], 2.0);
        a.add_product(f64::INFINITY, 2.0);
        a.add_triple_product(1.0, f64::NAN, 2.0);
        assert_eq!(a.to_f64(), 2.0);

        let mut g = vec![Wide192Accum::zero(); uplo_u_packed_len(2)];
        let mut c = vec![[Wide192Accum::zero(); 2]; 2];
        accumulate_uplo_u_wide192(&mut g, &mut c, &[1.0, 2.0], [1.0, 0.0], -1.0);
        accumulate_uplo_u_wide192(&mut g, &mut c, &[f64::NAN, 2.0], [1.0, 0.0], 1.0);
        assert_eq!(g[0].to_f64(), 0.0);
        assert_eq!(uplo_u_index(2, 5), uplo_u_index(5, 2));
        assert_eq!(uplo_u_packed_len(4), 10);
    }

    proptest! {
        #[test]
        fn prop_uplo_u_symmetry_and_wide192_order_invariance(
            r in 0usize..64,
            c in 0usize..64,
            v in prop::collection::vec(-100.0_f64..100.0_f64, 1..24),
        ) {
            prop_assert_eq!(uplo_u_index(r, c), uplo_u_index(c, r));
            prop_assert!(uplo_u_index(r, c) < uplo_u_packed_len(r.max(c) + 1));

            let mut fwd = Wide192Accum::zero();
            let mut rev = Wide192Accum::zero();
            for &x in &v {
                fwd.add_f64(x);
            }
            for &x in v.iter().rev() {
                rev.add_f64(x);
            }
            prop_assert!((fwd.to_f64() - rev.to_f64()).abs() <= 1.0e-12);
        }
    }
}
