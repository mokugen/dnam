//! Hardware `#[target_feature(enable = "avx2,fma")]` 256-bit `__m256` intrinsics for
//! [`Avx2Stereo8`], passing/returning only pointers and SSE2 `__m128` halves across boundaries
//! so non-`avx` callers never observe an `__m256` ABI mismatch.

use core::arch::x86_64::{
    __m128, _CMP_LE_OQ, _mm_cvtss_f32, _mm_loadu_ps, _mm_set1_ps, _mm_storeu_ps, _mm256_add_ps,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_ps, _mm256_castps_si256,
    _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_cmpgt_epi32,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_max_ps, _mm256_min_ps,
    _mm256_movemask_ps, _mm256_mul_ps, _mm256_or_si256, _mm256_permute_ps, _mm256_set_m128,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_slli_epi32, _mm256_srli_epi32,
    _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_ps,
};
use core::mem::MaybeUninit;

use super::Avx2Stereo8;
use crate::math::x86::sse2::{SCRUB_BIAS_SIGNED_I32, SCRUB_LIMIT_SIGNED_I32};
use crate::math::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED,
};

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn from_lr_hw(left: __m128, right: __m128, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_set_m128(right, left)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn into_lr_hw(v: &Avx2Stereo8) -> (__m128, __m128) {
    (_mm256_castps256_ps128(v.0), _mm256_extractf128_ps::<1>(v.0))
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn splat_hw(val: f32, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_set1_ps(val)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn zero_hw(out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_setzero_ps()));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn splat_lr_hw(left: f32, right: f32, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_set_m128(
        _mm_set1_ps(right),
        _mm_set1_ps(left),
    )));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn load_lr_hw(
    left: &[f32; 4],
    right: &[f32; 4],
    out: &mut MaybeUninit<Avx2Stereo8>,
) {
    // SAFETY: `[f32; 4]` holds 4 initialized `f32` values (`16 B`).
    let l = unsafe { _mm_loadu_ps(left.as_ptr()) };
    // SAFETY: `[f32; 4]` holds 4 initialized `f32` values (`16 B`).
    let r = unsafe { _mm_loadu_ps(right.as_ptr()) };
    out.write(Avx2Stereo8(_mm256_set_m128(r, l)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn store_lr_hw(v: &Avx2Stereo8, left: &mut [f32; 4], right: &mut [f32; 4]) {
    let l = _mm256_castps256_ps128(v.0);
    let r = _mm256_extractf128_ps::<1>(v.0);
    // SAFETY: `[f32; 4]` provides 4 writable `f32` lanes (`16 B`).
    unsafe {
        _mm_storeu_ps(left.as_mut_ptr(), l);
        _mm_storeu_ps(right.as_mut_ptr(), r);
    }
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn broadcast_half_hw(h: __m128, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_set_m128(h, h)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn add_hw(a: &Avx2Stereo8, b: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_add_ps(a.0, b.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn sub_hw(a: &Avx2Stereo8, b: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_sub_ps(a.0, b.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn mul_hw(a: &Avx2Stereo8, b: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_mul_ps(a.0, b.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn fma_hw(
    acc: &Avx2Stereo8,
    a: &Avx2Stereo8,
    b: &Avx2Stereo8,
    out: &mut MaybeUninit<Avx2Stereo8>,
) {
    out.write(Avx2Stereo8(_mm256_fmadd_ps(a.0, b.0, acc.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn fms_hw(
    acc: &Avx2Stereo8,
    a: &Avx2Stereo8,
    b: &Avx2Stereo8,
    out: &mut MaybeUninit<Avx2Stereo8>,
) {
    out.write(Avx2Stereo8(_mm256_fnmadd_ps(a.0, b.0, acc.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn max_hw(a: &Avx2Stereo8, b: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_max_ps(a.0, b.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn min_hw(a: &Avx2Stereo8, b: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_min_ps(a.0, b.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn abs_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_castsi256_ps(_mm256_and_si256(
        _mm256_castps_si256(a.0),
        _mm256_set1_epi32(IEEE754_F32_ABS_MASK as i32),
    ))));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn neg_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    out.write(Avx2Stereo8(_mm256_xor_ps(a.0, _mm256_set1_ps(-0.0))));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn select_le_hw(
    lhs: &Avx2Stereo8,
    rhs: &Avx2Stereo8,
    if_true: &Avx2Stereo8,
    if_false: &Avx2Stereo8,
    out: &mut MaybeUninit<Avx2Stereo8>,
) {
    let mask = _mm256_cmp_ps::<_CMP_LE_OQ>(lhs.0, rhs.0);
    out.write(Avx2Stereo8(_mm256_blendv_ps(if_false.0, if_true.0, mask)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn all_le_hw(lhs: &Avx2Stereo8, rhs: &Avx2Stereo8) -> bool {
    _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(lhs.0, rhs.0)) == 0xFF
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rotate_left_1_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    // `_mm256_permute_ps::<0x39>` rotates each 128-bit half `[0,1,2,3] -> [1,2,3,0]`.
    out.write(Avx2Stereo8(_mm256_permute_ps::<0x39>(a.0)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn flush_subnormal_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let bits = _mm256_castps_si256(a.0);
    let abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(IEEE754_F32_ABS_MASK as i32));
    let is_sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(IEEE754_F32_IMPLICIT_BIT as i32), abs_bits);
    let signed_zero = _mm256_and_si256(bits, _mm256_set1_epi32(i32::MIN));
    out.write(Avx2Stereo8(_mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(is_sub, signed_zero),
        _mm256_andnot_si256(is_sub, bits),
    ))));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn flush_recursive_state_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let bits = _mm256_castps_si256(a.0);
    let biased = _mm256_sub_epi32(
        _mm256_slli_epi32::<1>(bits),
        _mm256_set1_epi32(SCRUB_BIAS_SIGNED_I32),
    );
    let is_normal = _mm256_cmpgt_epi32(_mm256_set1_epi32(SCRUB_LIMIT_SIGNED_I32), biased);
    out.write(Avx2Stereo8(_mm256_castsi256_ps(_mm256_and_si256(
        bits, is_normal,
    ))));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rcp_ge1_nr2_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let one = _mm256_set1_ps(1.0);
    let d_safe = _mm256_max_ps(a.0, one);
    let y0 = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RCP_SEED as i32),
        _mm256_castps_si256(d_safe),
    ));
    let err0 = _mm256_fnmadd_ps(d_safe, y0, one);
    let y1 = _mm256_fmadd_ps(y0, err0, y0);
    let err1 = _mm256_fnmadd_ps(d_safe, y1, one);
    out.write(Avx2Stereo8(_mm256_fmadd_ps(y1, err1, y1)));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rcp_pos_nr2_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let d_safe = _mm256_max_ps(a.0, _mm256_set1_ps(1.0e-6));
    let one = _mm256_set1_ps(1.0);
    let y0 = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RCP_SEED as i32),
        _mm256_castps_si256(d_safe),
    ));
    let y1 = _mm256_fmadd_ps(y0, _mm256_fnmadd_ps(d_safe, y0, one), y0);
    out.write(Avx2Stereo8(_mm256_fmadd_ps(
        y1,
        _mm256_fnmadd_ps(d_safe, y1, one),
        y1,
    )));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rcp_pos_nr3_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let d_safe = _mm256_max_ps(a.0, _mm256_set1_ps(1.0e-6));
    let one = _mm256_set1_ps(1.0);
    let y0 = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RCP_SEED as i32),
        _mm256_castps_si256(d_safe),
    ));
    let y1 = _mm256_fmadd_ps(y0, _mm256_fnmadd_ps(d_safe, y0, one), y0);
    let y2 = _mm256_fmadd_ps(y1, _mm256_fnmadd_ps(d_safe, y1, one), y1);
    out.write(Avx2Stereo8(_mm256_fmadd_ps(
        y2,
        _mm256_fnmadd_ps(d_safe, y2, one),
        y2,
    )));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rsqrt_pos_nr2_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let d_safe = _mm256_max_ps(a.0, _mm256_set1_ps(1.0e-12));
    let half_d = _mm256_mul_ps(d_safe, _mm256_set1_ps(0.5));
    let three_halves = _mm256_set1_ps(1.5);
    let mut y = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RSQRT_SEED as i32),
        _mm256_srli_epi32::<1>(_mm256_castps_si256(d_safe)),
    ));
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    out.write(Avx2Stereo8(_mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    )));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn rsqrt_pos_nr3_hw(a: &Avx2Stereo8, out: &mut MaybeUninit<Avx2Stereo8>) {
    let d_safe = _mm256_max_ps(a.0, _mm256_set1_ps(1.0e-12));
    let half_d = _mm256_mul_ps(d_safe, _mm256_set1_ps(0.5));
    let three_halves = _mm256_set1_ps(1.5);
    let mut y = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RSQRT_SEED as i32),
        _mm256_srli_epi32::<1>(_mm256_castps_si256(d_safe)),
    ));
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    out.write(Avx2Stereo8(_mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    )));
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn horizontal_sum_lr_hw(a: &Avx2Stereo8) -> (f32, f32) {
    // Reduces each 128-bit stereo half independently in `(a0 + a1) + (a2 + a3)` tree order.
    let s1 = _mm256_add_ps(a.0, _mm256_permute_ps::<0xB1>(a.0));
    let s2 = _mm256_add_ps(s1, _mm256_permute_ps::<0x4E>(s1));
    (
        _mm_cvtss_f32(_mm256_castps256_ps128(s2)),
        _mm_cvtss_f32(_mm256_extractf128_ps::<1>(s2)),
    )
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn fma_mat4x4_shared_hw(
    acc: &Avx2Stereo8,
    cols: &[[f32; 4]; 4],
    x: &Avx2Stereo8,
    out: &mut MaybeUninit<Avx2Stereo8>,
) {
    // SAFETY: Each `cols[i]` is a valid 16-byte `[f32; 4]`.
    let (c0_128, c1_128, c2_128, c3_128) = unsafe {
        (
            _mm_loadu_ps(cols[0].as_ptr()),
            _mm_loadu_ps(cols[1].as_ptr()),
            _mm_loadu_ps(cols[2].as_ptr()),
            _mm_loadu_ps(cols[3].as_ptr()),
        )
    };
    let c0 = _mm256_set_m128(c0_128, c0_128);
    let c1 = _mm256_set_m128(c1_128, c1_128);
    let c2 = _mm256_set_m128(c2_128, c2_128);
    let c3 = _mm256_set_m128(c3_128, c3_128);

    let x0 = _mm256_permute_ps::<0x00>(x.0);
    let x1 = _mm256_permute_ps::<0x55>(x.0);
    let x2 = _mm256_permute_ps::<0xAA>(x.0);
    let x3 = _mm256_permute_ps::<0xFF>(x.0);

    let p01 = _mm256_fmadd_ps(c1, x1, _mm256_mul_ps(c0, x0));
    let p23 = _mm256_fmadd_ps(c3, x3, _mm256_mul_ps(c2, x2));
    out.write(Avx2Stereo8(_mm256_add_ps(acc.0, _mm256_add_ps(p01, p23))));
}
