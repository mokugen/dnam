//! Stage III backend trait (`Stage3Backend`) and SIMD `16`-tap stereo FIR dot product (`dot16_stereo_vec4`).

use crate::math::SimdVec4;

/// Domain-specific SIMD backend trait for Stage III `16`-tap stereo early reflection FIR kernels.
pub trait Stage3Backend {
    /// Evaluates two `16`-tap FIR inner products (`Left` and `Right`) simultaneously.
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32);
}

/// Safe `SimdVec4` implementation of [`Stage3Backend::dot16_stereo`].
#[inline(always)]
pub fn dot16_stereo_vec4<V: SimdVec4>(
    a_l: &[[f32; 4]; 4],
    b_l: &[[f32; 4]; 4],
    a_r: &[[f32; 4]; 4],
    b_r: &[[f32; 4]; 4],
) -> (f32, f32) {
    let al0 = V::load(&a_l[0])
        .mul(V::load(&b_l[0]))
        .fma(V::load(&a_l[1]), V::load(&b_l[1]));
    let al1 = V::load(&a_l[2])
        .mul(V::load(&b_l[2]))
        .fma(V::load(&a_l[3]), V::load(&b_l[3]));
    let ar0 = V::load(&a_r[0])
        .mul(V::load(&b_r[0]))
        .fma(V::load(&a_r[1]), V::load(&b_r[1]));
    let ar1 = V::load(&a_r[2])
        .mul(V::load(&b_r[2]))
        .fma(V::load(&a_r[3]), V::load(&b_r[3]));
    al0.add(al1).horizontal_sum_pair(ar0.add(ar1))
}

impl Stage3Backend for crate::math::ScalarBackend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        dot16_stereo_vec4::<crate::math::ScalarVec4>(a_l, b_l, a_r, b_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage3Backend for crate::math::NeonBackend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        dot16_stereo_vec4::<crate::math::NeonVec4>(a_l, b_l, a_r, b_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage3Backend for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        dot16_stereo_vec4::<crate::math::Sve2Vec4>(a_l, b_l, a_r, b_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage3Backend for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        if crate::math::ArmSve2Backend::is_supported_cached() {
            crate::math::ArmSve2Backend::dot16_stereo(a_l, b_l, a_r, b_r)
        } else {
            crate::math::NeonBackend::dot16_stereo(a_l, b_l, a_r, b_r)
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage3Backend for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        dot16_stereo_vec4::<crate::math::Sse2Vec4>(a_l, b_l, a_r, b_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage3Backend for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        dot16_stereo_vec4::<crate::math::Sse42Vec4>(a_l, b_l, a_r, b_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage3Backend for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        crate::math::x86::avx2::avx2_dot16_stereo(a_l, b_l, a_r, b_r)
            .unwrap_or_else(|| crate::math::X86Sse42Backend::dot16_stereo(a_l, b_l, a_r, b_r))
    }
}

#[cfg(target_arch = "wasm32")]
impl Stage3Backend for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn dot16_stereo(
        a_l: &[[f32; 4]; 4],
        b_l: &[[f32; 4]; 4],
        a_r: &[[f32; 4]; 4],
        b_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        dot16_stereo_vec4::<crate::math::WasmVec4>(a_l, b_l, a_r, b_r)
    }
}
