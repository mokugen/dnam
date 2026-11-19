//! Wide 256-bit `AVX2 + FMA` stereo 8-lane vector register ([`Avx2Stereo8`], `lanes 0..3 = Left`,
//! `4..7 = Right`) implementing [`dnam_kernels::StereoVec8`] with exact fixed 4-lane pair
//! operations and reduction trees for `0.000000 ULP` parity against
//! [`dnam_kernels::InterleavedStereo8<Avx2FmaVec4>`].

use core::arch::x86_64::{
    __m128, __m256, _CMP_LE_OQ, _mm_cvtss_f32, _mm256_add_ps, _mm256_and_si256, _mm256_andnot_ps,
    _mm256_andnot_si256, _mm256_blendv_ps, _mm256_castps_si256, _mm256_castps256_ps128,
    _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_cmpeq_epi32, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_fnmadd_ps, _mm256_hadd_ps, _mm256_max_ps, _mm256_min_ps, _mm256_movemask_ps,
    _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256,
    _mm256_srli_epi32, _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_ps,
};
use dnam_kernels::{IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, InterleavedStereo8, StereoVec8};

use super::{Avx2FmaVec4, X86Avx2FmaBackend};

/// 32-byte aligned 8-lane `f32` stereo vector (`lanes 0..3 = Left`, `4..7 = Right`) backed by a
/// single 256-bit `AVX2 + FMA` register when supported, with automatic 128-bit SSE fallback.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Avx2Stereo8(pub [f32; 8]);

impl Avx2Stereo8 {
    /// Constructs an [`Avx2Stereo8`] directly from 8 `[L0..L3, R0..R3]` lanes.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(lanes: [f32; 8]) -> Self {
        Self(lanes)
    }

    /// Extracts all 8 `[L0..L3, R0..R3]` lanes as `[f32; 8]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 8] {
        self.0
    }

    #[inline(always)]
    fn as_interleaved(self) -> InterleavedStereo8<Avx2FmaVec4> {
        let (left, right) = self.into_lr();
        InterleavedStereo8 { left, right }
    }

    #[inline(always)]
    fn from_interleaved(v: InterleavedStereo8<Avx2FmaVec4>) -> Self {
        Self::from_lr(v.left, v.right)
    }
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_add(a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_add_ps(va, vb)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_sub(a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_sub_ps(va, vb)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_mul(a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_mul_ps(va, vb)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_fma(acc: Avx2Stereo8, a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let vacc: __m256 = zerocopy::transmute!(acc.0);
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_fmadd_ps(va, vb, vacc)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_fms(acc: Avx2Stereo8, a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let vacc: __m256 = zerocopy::transmute!(acc.0);
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_fnmadd_ps(va, vb, vacc)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_max(a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_max_ps(va, vb)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_min(a: Avx2Stereo8, b: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let vb: __m256 = zerocopy::transmute!(b.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_min_ps(va, vb)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_abs(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_andnot_ps(
        _mm256_set1_ps(-0.0),
        va
    )))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_neg(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    Avx2Stereo8(zerocopy::transmute!(_mm256_xor_ps(
        va,
        _mm256_set1_ps(-0.0)
    )))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_select_le(
    lhs: Avx2Stereo8,
    rhs: Avx2Stereo8,
    if_true: Avx2Stereo8,
    if_false: Avx2Stereo8,
) -> Avx2Stereo8 {
    let vl: __m256 = zerocopy::transmute!(lhs.0);
    let vr: __m256 = zerocopy::transmute!(rhs.0);
    let vt: __m256 = zerocopy::transmute!(if_true.0);
    let vf: __m256 = zerocopy::transmute!(if_false.0);
    let mask = _mm256_cmp_ps::<_CMP_LE_OQ>(vl, vr);
    Avx2Stereo8(zerocopy::transmute!(_mm256_blendv_ps(vf, vt, mask)))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_flush_subnormal(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let bits = _mm256_castps_si256(va);
    let exp = _mm256_and_si256(bits, _mm256_set1_epi32(0x7F80_0000));
    let is_zero_exp = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    let clear_mant = _mm256_and_si256(is_zero_exp, _mm256_set1_epi32(0x007F_FFFF));
    Avx2Stereo8(zerocopy::transmute!(_mm256_castsi256_ps(
        _mm256_andnot_si256(clear_mant, bits)
    )))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_all_le(lhs: Avx2Stereo8, rhs: Avx2Stereo8) -> bool {
    let va: __m256 = zerocopy::transmute!(lhs.0);
    let vb: __m256 = zerocopy::transmute!(rhs.0);
    _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(va, vb)) == 0xFF
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_flush_recursive_state(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let bits = _mm256_castps_si256(va);
    let exp_mask = _mm256_set1_epi32(0x7F80_0000);
    let exp = _mm256_and_si256(bits, exp_mask);
    let is_nonfinite = _mm256_cmpeq_epi32(exp, exp_mask);
    let is_sub = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
    let clear_mant = _mm256_and_si256(is_sub, _mm256_set1_epi32(0x007F_FFFF));
    let cleared = _mm256_andnot_si256(is_nonfinite, bits);
    Avx2Stereo8(zerocopy::transmute!(_mm256_castsi256_ps(
        _mm256_andnot_si256(clear_mant, cleared)
    )))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_rcp_pos_seed(d: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RCP_SEED as i32),
        _mm256_castps_si256(d),
    ))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_rsqrt_pos_seed(d: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(IEEE754_F32_RSQRT_SEED as i32),
        _mm256_srli_epi32::<1>(_mm256_castps_si256(d)),
    ))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_rcp_pos_nr2(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let d_safe = _mm256_max_ps(va, _mm256_set1_ps(1.0e-6));
    let one = _mm256_set1_ps(1.0);
    let y0 = avx2_s8_rcp_pos_seed(d_safe);
    let y1 = _mm256_fmadd_ps(y0, _mm256_fnmadd_ps(d_safe, y0, one), y0);
    let y2 = _mm256_fmadd_ps(y1, _mm256_fnmadd_ps(d_safe, y1, one), y1);
    Avx2Stereo8(zerocopy::transmute!(y2))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_rcp_pos_nr3(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let d_safe = _mm256_max_ps(va, _mm256_set1_ps(1.0e-6));
    let one = _mm256_set1_ps(1.0);
    let y0 = avx2_s8_rcp_pos_seed(d_safe);
    let y1 = _mm256_fmadd_ps(y0, _mm256_fnmadd_ps(d_safe, y0, one), y0);
    let y2 = _mm256_fmadd_ps(y1, _mm256_fnmadd_ps(d_safe, y1, one), y1);
    let y3 = _mm256_fmadd_ps(y2, _mm256_fnmadd_ps(d_safe, y2, one), y2);
    Avx2Stereo8(zerocopy::transmute!(y3))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_rsqrt_pos_nr2(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let d_safe = _mm256_max_ps(va, _mm256_set1_ps(1.0e-12));
    let half_d = _mm256_mul_ps(d_safe, _mm256_set1_ps(0.5));
    let three_halves = _mm256_set1_ps(1.5);
    let mut y = avx2_s8_rsqrt_pos_seed(d_safe);
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    Avx2Stereo8(zerocopy::transmute!(y))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_rsqrt_pos_nr3(a: Avx2Stereo8) -> Avx2Stereo8 {
    let va: __m256 = zerocopy::transmute!(a.0);
    let d_safe = _mm256_max_ps(va, _mm256_set1_ps(1.0e-12));
    let half_d = _mm256_mul_ps(d_safe, _mm256_set1_ps(0.5));
    let three_halves = _mm256_set1_ps(1.5);
    let mut y = avx2_s8_rsqrt_pos_seed(d_safe);
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    y = _mm256_mul_ps(
        y,
        _mm256_fnmadd_ps(half_d, _mm256_mul_ps(y, y), three_halves),
    );
    Avx2Stereo8(zerocopy::transmute!(y))
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_horizontal_sum_lr(a: Avx2Stereo8) -> (f32, f32) {
    let v: __m256 = zerocopy::transmute!(a.0);
    let pairs = _mm256_hadd_ps(v, v);
    let totals = _mm256_hadd_ps(pairs, pairs);
    (
        _mm_cvtss_f32(_mm256_castps256_ps128(totals)),
        _mm_cvtss_f32(_mm256_extractf128_ps::<1>(totals)),
    )
}

#[target_feature(enable = "avx,avx2,fma")]
#[inline]
fn avx2_s8_fma_mat4x4_shared(
    acc: Avx2Stereo8,
    cols: &[[f32; 4]; 4],
    x: Avx2Stereo8,
) -> Avx2Stereo8 {
    let acc_v: __m256 = zerocopy::transmute!(acc.0);
    let xv: __m256 = zerocopy::transmute!(x.0);
    let bc0: __m256 = zerocopy::transmute!([cols[0], cols[0]]);
    let bc1: __m256 = zerocopy::transmute!([cols[1], cols[1]]);
    let bc2: __m256 = zerocopy::transmute!([cols[2], cols[2]]);
    let bc3: __m256 = zerocopy::transmute!([cols[3], cols[3]]);
    let x0 = _mm256_permute_ps::<0x00>(xv);
    let x1 = _mm256_permute_ps::<0x55>(xv);
    let x2 = _mm256_permute_ps::<0xAA>(xv);
    let x3 = _mm256_permute_ps::<0xFF>(xv);
    let p01 = _mm256_fmadd_ps(bc1, x1, _mm256_mul_ps(bc0, x0));
    let p23 = _mm256_fmadd_ps(bc3, x3, _mm256_mul_ps(bc2, x2));
    Avx2Stereo8(zerocopy::transmute!(_mm256_add_ps(
        acc_v,
        _mm256_add_ps(p01, p23)
    )))
}

impl StereoVec8 for Avx2Stereo8 {
    type Half = Avx2FmaVec4;

    #[inline(always)]
    fn from_lr(left: Self::Half, right: Self::Half) -> Self {
        Self(zerocopy::transmute!([left.0, right.0]))
    }

    #[inline(always)]
    fn into_lr(self) -> (Self::Half, Self::Half) {
        let [left, right]: [__m128; 2] = zerocopy::transmute!(self.0);
        (Avx2FmaVec4(left), Avx2FmaVec4(right))
    }

    #[inline(always)]
    fn broadcast_half(half: Self::Half) -> Self {
        Self(zerocopy::transmute!([half.0, half.0]))
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_add(self, rhs) }
        } else {
            Self::from_interleaved(self.as_interleaved().add(rhs.as_interleaved()))
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_sub(self, rhs) }
        } else {
            Self::from_interleaved(self.as_interleaved().sub(rhs.as_interleaved()))
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_mul(self, rhs) }
        } else {
            Self::from_interleaved(self.as_interleaved().mul(rhs.as_interleaved()))
        }
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_fma(self, a, b) }
        } else {
            Self::from_interleaved(
                self.as_interleaved()
                    .fma(a.as_interleaved(), b.as_interleaved()),
            )
        }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_fms(self, a, b) }
        } else {
            Self::from_interleaved(
                self.as_interleaved()
                    .fms(a.as_interleaved(), b.as_interleaved()),
            )
        }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_max(self, rhs) }
        } else {
            Self::from_interleaved(self.as_interleaved().max(rhs.as_interleaved()))
        }
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_min(self, rhs) }
        } else {
            Self::from_interleaved(self.as_interleaved().min(rhs.as_interleaved()))
        }
    }

    #[inline]
    fn abs(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_abs(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().abs())
        }
    }

    #[inline]
    fn neg(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_neg(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().neg())
        }
    }

    #[inline]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_select_le(lhs, rhs, if_true, if_false) }
        } else {
            Self::from_interleaved(InterleavedStereo8::select_le(
                lhs.as_interleaved(),
                rhs.as_interleaved(),
                if_true.as_interleaved(),
                if_false.as_interleaved(),
            ))
        }
    }

    #[inline]
    fn all_le(self, rhs: Self) -> bool {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_all_le(self, rhs) }
        } else {
            self.as_interleaved().all_le(rhs.as_interleaved())
        }
    }

    #[inline]
    fn flush_subnormal(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_flush_subnormal(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().flush_subnormal())
        }
    }

    #[inline]
    fn flush_recursive_state(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_flush_recursive_state(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().flush_recursive_state())
        }
    }

    #[inline]
    fn rcp_pos_nr2(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_rcp_pos_nr2(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().rcp_pos_nr2())
        }
    }

    #[inline]
    fn rcp_pos_nr3(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_rcp_pos_nr3(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().rcp_pos_nr3())
        }
    }

    #[inline]
    fn rsqrt_pos_nr2(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_rsqrt_pos_nr2(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().rsqrt_pos_nr2())
        }
    }

    #[inline]
    fn rsqrt_pos_nr3(self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_rsqrt_pos_nr3(self) }
        } else {
            Self::from_interleaved(self.as_interleaved().rsqrt_pos_nr3())
        }
    }

    #[inline]
    fn horizontal_sum_lr(self) -> (f32, f32) {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_horizontal_sum_lr(self) }
        } else {
            self.as_interleaved().horizontal_sum_lr()
        }
    }

    #[inline]
    fn fma_mat4x4_shared(acc: Self, cols: &[[f32; 4]; 4], x: Self) -> Self {
        if X86Avx2FmaBackend::is_supported_cached() {
            // SAFETY: Guarded by `X86Avx2FmaBackend::is_supported_cached()`.
            unsafe { avx2_s8_fma_mat4x4_shared(acc, cols, x) }
        } else {
            Self::from_interleaved(InterleavedStereo8::fma_mat4x4_shared(
                acc.as_interleaved(),
                cols,
                x.as_interleaved(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        KernelVec4, SonaGroupParamsStereo4, SonaParaLruCoeffs4, SonaToneWeights4, TptSvfCoeffs4,
        prewarp_tan_pi_f_over_fs_f32, step_sona_group_stereo8,
    };

    use super::*;
    use crate::math::ScalarVec4;

    #[test]
    fn test_avx2_stereo8_zero_ulp_parity_with_interleaved() {
        crate::math::warm_simd_feature_caches();
        type I8Avx = InterleavedStereo8<Avx2FmaVec4>;
        type I8Scal = InterleavedStereo8<ScalarVec4>;

        let lhs_l = [0.25_f32, -1.5, 3.75, -0.125];
        let lhs_r = [-0.5_f32, 2.25, -4.0, 0.625];
        let rhs_l = [1.0_f32, 0.5, -2.5, 4.5];
        let rhs_r = [0.75_f32, -1.25, 3.0, -2.0];

        let avx_a = Avx2Stereo8::load_lr(&lhs_l, &lhs_r);
        let avx_b = Avx2Stereo8::load_lr(&rhs_l, &rhs_r);
        let ilv_a = I8Avx::load_lr(&lhs_l, &lhs_r);
        let ilv_b = I8Avx::load_lr(&rhs_l, &rhs_r);
        let scal_a = I8Scal::load_lr(&lhs_l, &lhs_r);
        let scal_b = I8Scal::load_lr(&rhs_l, &rhs_r);

        let check_0ulp = |got: Avx2Stereo8, exp: I8Avx| {
            let (gl, gr) = got.into_lr();
            let (el, er) = exp.into_lr();
            assert_eq!(
                KernelVec4::to_array(gl).map(f32::to_bits),
                KernelVec4::to_array(el).map(f32::to_bits)
            );
            assert_eq!(
                KernelVec4::to_array(gr).map(f32::to_bits),
                KernelVec4::to_array(er).map(f32::to_bits)
            );
        };

        check_0ulp(avx_a.add(avx_b), ilv_a.add(ilv_b));
        check_0ulp(avx_a.sub(avx_b), ilv_a.sub(ilv_b));
        check_0ulp(avx_a.mul(avx_b), ilv_a.mul(ilv_b));
        check_0ulp(avx_a.fma(avx_b, avx_a), ilv_a.fma(ilv_b, ilv_a));
        check_0ulp(avx_a.fms(avx_b, avx_a), ilv_a.fms(ilv_b, ilv_a));
        check_0ulp(avx_a.max(avx_b), ilv_a.max(ilv_b));
        check_0ulp(avx_a.min(avx_b), ilv_a.min(ilv_b));
        check_0ulp(avx_a.abs(), ilv_a.abs());
        check_0ulp(avx_a.neg(), ilv_a.neg());
        check_0ulp(
            Avx2Stereo8::select_le(avx_a, avx_b, avx_a, avx_b),
            I8Avx::select_le(ilv_a, ilv_b, ilv_a, ilv_b),
        );

        let pos = Avx2Stereo8::load_lr(&[0.25, 1.0, 2.5, 9.0], &[0.5, 2.0, 4.0, 16.0]);
        let ilv_pos = I8Avx::load_lr(&[0.25, 1.0, 2.5, 9.0], &[0.5, 2.0, 4.0, 16.0]);
        check_0ulp(pos.rcp_pos_nr2(), ilv_pos.rcp_pos_nr2());
        check_0ulp(pos.rcp_pos_nr3(), ilv_pos.rcp_pos_nr3());
        check_0ulp(pos.rsqrt_pos_nr2(), ilv_pos.rsqrt_pos_nr2());
        check_0ulp(pos.rsqrt_pos_nr3(), ilv_pos.rsqrt_pos_nr3());

        // Fixed 4-lane tree reduction parity with ScalarVec4 (`0.000000 ULP`).
        let (sl_avx, sr_avx) = avx_a.horizontal_sum_lr();
        let (sl_scal, sr_scal) = scal_a.horizontal_sum_lr();
        assert_eq!(sl_avx.to_bits(), sl_scal.to_bits());
        assert_eq!(sr_avx.to_bits(), sr_scal.to_bits());

        let cols = [
            [0.5, -0.25, 0.125, -0.5],
            [-0.5, 0.5, 0.25, 0.125],
            [0.25, 0.5, -0.5, 0.25],
            [-0.125, -0.25, 0.5, 0.5],
        ];
        check_0ulp(
            Avx2Stereo8::fma_mat4x4_shared(avx_a, &cols, avx_b),
            I8Avx::fma_mat4x4_shared(ilv_a, &cols, ilv_b),
        );

        // Unfused tree operations match ScalarVec4 at `0 ULP`.
        let (sal, sar) = avx_a.add(avx_b).into_lr();
        let (ssl, ssr) = scal_a.add(scal_b).into_lr();
        assert_eq!(
            KernelVec4::to_array(sal).map(f32::to_bits),
            KernelVec4::to_array(ssl).map(f32::to_bits)
        );
        assert_eq!(
            KernelVec4::to_array(sar).map(f32::to_bits),
            KernelVec4::to_array(ssr).map(f32::to_bits)
        );

        // Full `SONA` group kernel `0-ULP` parity against `InterleavedStereo8<Avx2FmaVec4>`.
        let g = <Avx2FmaVec4 as KernelVec4>::splat(prewarp_tan_pi_f_over_fs_f32(0.08));
        let k = <Avx2FmaVec4 as KernelVec4>::splat(0.75);
        let params = SonaGroupParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4::lp_only(),
            lru: SonaParaLruCoeffs4::from_clamped(
                <Avx2FmaVec4 as KernelVec4>::splat(0.95),
                <Avx2FmaVec4 as KernelVec4>::splat(0.4),
                <Avx2FmaVec4 as KernelVec4>::splat(0.5),
            ),
            rect_sign: <Avx2FmaVec4 as KernelVec4>::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: <Avx2FmaVec4 as KernelVec4>::splat(0.4),
            adaa_drive: <Avx2FmaVec4 as KernelVec4>::splat(2.0),
            lru_bias_drive: <Avx2FmaVec4 as KernelVec4>::splat(0.3),
            wavefolder_mix: <Avx2FmaVec4 as KernelVec4>::from_array([0.0, 0.25, 0.5, 1.0]),
        };
        let mut st_wide = dnam_kernels::SonaBlockStateStereo8::<Avx2Stereo8>::zero();
        let mut st_ilv = dnam_kernels::SonaBlockStateStereo8::<I8Avx>::zero();
        for n in 0..16 {
            let u_l = ((n as f32) * 0.27).sin() * 1.4;
            let u_r = ((n as f32) * 0.19).cos() * 1.1;
            let out_w =
                step_sona_group_stereo8(Avx2Stereo8::splat_lr(u_l, u_r), &mut st_wide, &params);
            let out_i = step_sona_group_stereo8(I8Avx::splat_lr(u_l, u_r), &mut st_ilv, &params);
            check_0ulp(out_w.lru_state, out_i.lru_state);
            check_0ulp(out_w.adaa_odd, out_i.adaa_odd);
            check_0ulp(out_w.adaa_even, out_i.adaa_even);
        }
    }
}
