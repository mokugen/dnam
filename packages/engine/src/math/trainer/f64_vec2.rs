//! Double-precision (`f64`) 2-lane SIMD vector abstraction (`SimdF64Vec2`), portable
//! [`ScalarF64Vec2`] backend, and strict IEEE-754 non-fused `[4x4]`/`[8x4]` tile contractions.

/// 16-byte SIMD-aligned 2-lane `f64` scalar vector (`#[repr(C, align(16))]`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScalarF64Vec2(pub [f64; 2]);

/// Level-1 hardware abstraction: 128-bit 2-lane `f64` SIMD register
/// (`float64x2_t` / `__m128d` / `v128 f64x2`).
pub trait SimdF64Vec2: Copy + Clone + Send + Sync + 'static {
    /// Broadcasts `val` across both `f64` lanes.
    fn splat(val: f64) -> Self;

    /// Constructs a 2-lane `f64` vector from `[a, b]`.
    #[inline(always)]
    fn splat_pair(a: f64, b: f64) -> Self {
        Self::from_array([a, b])
    }

    /// Widens a pair of IEEE-754 `f32` values into `[f64::from(a), f64::from(b)]`.
    #[inline(always)]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        Self::from_array([f64::from(a), f64::from(b)])
    }

    /// Loads 2 `f64` lanes from `[f64; 2]`.
    fn load(src: &[f64; 2]) -> Self;
    /// Stores 2 `f64` lanes into `[f64; 2]`.
    fn store(self, dst: &mut [f64; 2]);
    /// Lane-wise addition (`self + rhs`).
    fn add(self, rhs: Self) -> Self;
    /// Lane-wise subtraction (`self - rhs`).
    fn sub(self, rhs: Self) -> Self;
    /// Lane-wise multiplication (`self * rhs`).
    fn mul(self, rhs: Self) -> Self;
    /// Fused multiply-add (`self + a * b`), using hardware FMA where available.
    fn fma(self, a: Self, b: Self) -> Self;

    /// Fused multiply-add with indexed scalar lane broadcast (`self + a * v[LANE]`, `LANE < 2`).
    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 2) };
        self.fma(a, Self::splat(v.to_array()[LANE]))
    }

    /// Strict non-fused multiply-add (`self + (a * b)`) with separate IEEE-754 rounding.
    #[inline(always)]
    fn mul_add_ieee(self, a: Self, b: Self) -> Self {
        self.add(a.mul(b))
    }

    /// Fused multiply-subtract (`self - a * b`).
    fn fms(self, a: Self, b: Self) -> Self;
    /// Lane-wise maximum (`max(self, rhs)`).
    fn max(self, rhs: Self) -> Self;
    /// Lane-wise minimum (`min(self, rhs)`).
    fn min(self, rhs: Self) -> Self;

    /// Branchless lane-wise clamp to `[lo, hi]`.
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Lane-wise absolute value (`|self|`).
    fn abs(self) -> Self;
    /// Horizontal sum across both `f64` lanes (`lane0 + lane1`).
    fn horizontal_sum(self) -> f64;

    /// Simultaneously reduces two 2-lane vectors into `(self.horizontal_sum(), rhs.horizontal_sum())`.
    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f64, f64) {
        (self.horizontal_sum(), rhs.horizontal_sum())
    }

    /// 2-lane `f64` dot product (`sum(self * rhs)`).
    #[inline(always)]
    fn dot(self, rhs: Self) -> f64 {
        self.mul(rhs).horizontal_sum()
    }

    /// Returns a 2-lane vector of zeros (`0.0`).
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Constructs a 2-lane vector from `[f64; 2]`.
    #[inline(always)]
    fn from_array(arr: [f64; 2]) -> Self {
        Self::load(&arr)
    }

    /// Converts this 2-lane vector into `[f64; 2]`.
    #[inline(always)]
    fn to_array(self) -> [f64; 2] {
        let mut dst = [0.0; 2];
        self.store(&mut dst);
        dst
    }
}

impl SimdF64Vec2 for ScalarF64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        Self([val; 2])
    }
    #[inline(always)]
    fn load(src: &[f64; 2]) -> Self {
        Self(*src)
    }
    #[inline(always)]
    fn store(self, dst: &mut [f64; 2]) {
        *dst = self.0;
    }
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1]])
    }
    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self([self.0[0] + a.0[0] * b.0[0], self.0[1] + a.0[1] * b.0[1]])
    }
    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self([self.0[0] - a.0[0] * b.0[0], self.0[1] - a.0[1] * b.0[1]])
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self([self.0[0].max(rhs.0[0]), self.0[1].max(rhs.0[1])])
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self([self.0[0].min(rhs.0[0]), self.0[1].min(rhs.0[1])])
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Self([self.0[0].abs(), self.0[1].abs()])
    }
    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        self.0[0] + self.0[1]
    }
}

#[inline(always)]
pub(super) fn load_vec2_pair<V: SimdF64Vec2>(chunk: &[f64; 4]) -> (V, V) {
    (
        V::load(&[chunk[0], chunk[1]]),
        V::load(&[chunk[2], chunk[3]]),
    )
}

#[inline(always)]
pub(super) fn load_vec2_quad<V: SimdF64Vec2>(chunk: &[f64; 8]) -> [V; 4] {
    [
        V::load(&[chunk[0], chunk[1]]),
        V::load(&[chunk[2], chunk[3]]),
        V::load(&[chunk[4], chunk[5]]),
        V::load(&[chunk[6], chunk[7]]),
    ]
}

#[inline(always)]
pub(super) fn store_vec2_quad<V: SimdF64Vec2>(dst: &mut [f64; 8], vals: [V; 4]) {
    let [a0, a1, a2, a3] = [
        vals[0].to_array(),
        vals[1].to_array(),
        vals[2].to_array(),
        vals[3].to_array(),
    ];
    *dst = [a0[0], a0[1], a1[0], a1[1], a2[0], a2[1], a3[0], a3[1]];
}

/// Evaluates an IEEE-754 non-fused `[4 x 4]` tile contraction using `SimdF64Vec2` in exact
/// chunk-pair summation order matching `wide_gram::contract_offdiag_tile_4x4`.
#[inline(always)]
pub(crate) fn contract_tile_4x4_vec2_ieee<V: SimdF64Vec2>(
    f_i: &[[f64; 4]],
    wf_j: &[[f64; 4]],
) -> [[f64; 4]; 4] {
    let m = f_i.len().min(wf_j.len());
    let (fi_pairs, fi_rem) = f_i[..m].as_chunks::<2>();
    let (wf_pairs, wf_rem) = wf_j[..m].as_chunks::<2>();
    let z = V::zero();
    let mut a = [[z, z]; 4];
    let mut b = [[z, z]; 4];
    for (&[fa, fb], &[wa, wb]) in fi_pairs.iter().zip(wf_pairs) {
        let (fa0, fa1) = load_vec2_pair::<V>(&fa);
        let (fb0, fb1) = load_vec2_pair::<V>(&fb);
        for lj in 0..4 {
            let w0 = V::splat(wa[lj]);
            let w1 = V::splat(wb[lj]);
            a[lj][0] = a[lj][0].mul_add_ieee(fa0, w0);
            a[lj][1] = a[lj][1].mul_add_ieee(fa1, w0);
            b[lj][0] = b[lj][0].mul_add_ieee(fb0, w1);
            b[lj][1] = b[lj][1].mul_add_ieee(fb1, w1);
        }
    }
    for (&fa, &wa) in fi_rem.iter().zip(wf_rem) {
        let (fa0, fa1) = load_vec2_pair::<V>(&fa);
        for lj in 0..4 {
            let w0 = V::splat(wa[lj]);
            a[lj][0] = a[lj][0].mul_add_ieee(fa0, w0);
            a[lj][1] = a[lj][1].mul_add_ieee(fa1, w0);
        }
    }
    let mut out = [[0.0_f64; 4]; 4];
    for (row, (a_row, b_row)) in out.iter_mut().zip(a.iter().zip(&b)) {
        let (lo, hi) = (
            a_row[0].add(b_row[0]).to_array(),
            a_row[1].add(b_row[1]).to_array(),
        );
        *row = [lo[0], lo[1], hi[0], hi[1]];
    }
    out
}

/// Evaluates two IEEE-754 non-fused `[4 x 4]` tile contractions sharing `wf_j` (`0-ULP` parity).
#[inline(always)]
pub(crate) fn contract_tile_8x4_vec2_ieee<V: SimdF64Vec2>(
    f_i0: &[[f64; 4]],
    f_i1: &[[f64; 4]],
    wf_j: &[[f64; 4]],
) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let m = f_i0.len().min(f_i1.len()).min(wf_j.len());
    let z = V::zero();
    let (mut a, mut b) = ([[z, z]; 4], [[z, z]; 4]);
    for ((&fa, &fb), &w) in f_i0[..m].iter().zip(&f_i1[..m]).zip(&wf_j[..m]) {
        let (fa0, fa1) = load_vec2_pair::<V>(&fa);
        let (fb0, fb1) = load_vec2_pair::<V>(&fb);
        for lj in 0..4 {
            let wv = V::splat(w[lj]);
            a[lj][0] = a[lj][0].mul_add_ieee(fa0, wv);
            a[lj][1] = a[lj][1].mul_add_ieee(fa1, wv);
            b[lj][0] = b[lj][0].mul_add_ieee(fb0, wv);
            b[lj][1] = b[lj][1].mul_add_ieee(fb1, wv);
        }
    }
    let (mut out_a, mut out_b) = ([[0.0_f64; 4]; 4], [[0.0_f64; 4]; 4]);
    for lj in 0..4 {
        let (al, ah) = (a[lj][0].to_array(), a[lj][1].to_array());
        let (bl, bh) = (b[lj][0].to_array(), b[lj][1].to_array());
        out_a[lj] = [al[0], al[1], ah[0], ah[1]];
        out_b[lj] = [bl[0], bl[1], bh[0], bh[1]];
    }
    (out_a, out_b)
}

#[cfg(test)]
pub(crate) use self::tests::verify_simd_f64_vec2;

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn verify_simd_f64_vec2<V: SimdF64Vec2>() {
        assert_eq!(V::zero().to_array(), [0.0, 0.0]);
        assert_eq!(V::splat(3.5).to_array(), [3.5, 3.5]);
        assert_eq!(V::splat_pair(-1.25, 4.75).to_array(), [-1.25, 4.75]);
        assert_eq!(V::from_f32_pair(1.5, -2.25).to_array(), [1.5, -2.25]);

        let (a, b, c) = (
            V::from_array([2.0, -5.0]),
            V::from_array([3.0, 4.0]),
            V::from_array([10.0, -2.0]),
        );
        assert_eq!(a.add(b).to_array(), [5.0, -1.0]);
        assert_eq!(a.sub(b).to_array(), [-1.0, -9.0]);
        assert_eq!(a.mul(b).to_array(), [6.0, -20.0]);
        assert!((c.fma(a, b).to_array()[0] - 16.0).abs() <= 1.0e-12);
        assert!((c.fma(a, b).to_array()[1] + 22.0).abs() <= 1.0e-12);
        assert!((c.fma_lane::<0>(a, b).to_array()[1] + 17.0).abs() <= 1.0e-12);
        assert!((c.fma_lane::<1>(a, b).to_array()[0] - 18.0).abs() <= 1.0e-12);
        assert_eq!(c.mul_add_ieee(a, b).to_array(), [16.0, -22.0]);
        assert!((c.fms(a, b).to_array()[0] - 4.0).abs() <= 1.0e-12);
        assert!((c.fms(a, b).to_array()[1] - 18.0).abs() <= 1.0e-12);
        assert_eq!(a.max(b).to_array(), [3.0, 4.0]);
        assert_eq!(a.min(b).to_array(), [2.0, -5.0]);
        assert_eq!(
            V::from_array([-10.0, 10.0])
                .clamp(V::splat(-3.0), V::splat(3.0))
                .to_array(),
            [-3.0, 3.0]
        );
        assert_eq!(a.abs().to_array(), [2.0, 5.0]);
        assert_eq!(a.horizontal_sum(), -3.0);
        assert_eq!(a.horizontal_sum_pair(b), (-3.0, 7.0));
        assert_eq!(a.dot(b), -14.0);
    }

    #[test]
    fn test_scalar_f64_vec2_roundtrip() {
        verify_simd_f64_vec2::<ScalarF64Vec2>();
    }
}
