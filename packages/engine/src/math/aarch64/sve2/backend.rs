//! [`ArmSve2Backend`] detection, `MathBackend` / `Crc32cBackend` implementations, and
//! register-resident SVE2 `T1` tracker exponent scaling.

use core::arch::aarch64::{
    float32x4_t, vandq_u32, vbslq_f32, vceqq_u32, vdupq_n_u32, vreinterpretq_u32_f32,
};
use core::sync::atomic::{AtomicU8, Ordering};

use super::super::neon::aarch64_crc32c_update;
use super::{Sve2Stereo8, Sve2Vec4, from_zreg, to_zreg};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, ScalarBackend, SimdVec4};

/// ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) hardware backend for `aarch64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSve2Backend;

/// Cached `FEAT_SVE2` support state (`0 = uninitialized`, `1 = unsupported`, `2 = supported`).
static SVE2_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

impl ArmSve2Backend {
    /// Returns the cached `FEAT_SVE2` capability flag with a single relaxed atomic load and zero
    /// OS feature-detection calls (`sysctlbyname`, `getauxval`, or `detect_and_initialize`).
    #[inline(always)]
    pub fn is_supported_cached() -> bool {
        SVE2_SUPPORT_CACHE.load(Ordering::Relaxed) == 2
    }

    /// Queries the OS/CPU once off the hot path when [`SVE2_SUPPORT_CACHE`] is uninitialized.
    #[cold]
    fn detect_uncached() -> bool {
        #[cfg(any(test, feature = "std"))]
        {
            std::arch::is_aarch64_feature_detected!("sve2")
        }
        #[cfg(all(target_os = "linux", not(any(test, feature = "std"))))]
        {
            let (pfr0, zfr0): (u64, u64);
            // SAFETY: `ID_AA64PFR0_EL1` (`SVE` [35:32]) and `ID_AA64ZFR0_EL1` (`S3_0_C0_C4_4`,
            // `SVEver` [3:0]) are emulated at EL0 by Linux. Read `ZFR0` only when `SVE != 0`.
            unsafe {
                core::arch::asm!(
                    "mrs {p}, id_aa64pfr0_el1",
                    p = out(reg) pfr0,
                    options(nomem, nostack, preserves_flags),
                );
                if ((pfr0 >> 32) & 0xF) >= 1 {
                    core::arch::asm!(
                        "mrs {z}, S3_0_C0_C4_4",
                        z = out(reg) zfr0,
                        options(nomem, nostack, preserves_flags),
                    );
                } else {
                    zfr0 = 0;
                }
            }
            (zfr0 & 0xF) >= 1
        }
        #[cfg(all(not(target_os = "linux"), not(any(test, feature = "std"))))]
        {
            cfg!(target_feature = "sve2")
        }
    }

    /// Detects and caches at runtime whether the host `aarch64` CPU supports `FEAT_SVE2`.
    #[inline]
    pub fn is_supported() -> bool {
        let cached = SVE2_SUPPORT_CACHE.load(Ordering::Relaxed);
        if cached != 0 {
            return cached == 2;
        }
        let ok = Self::detect_uncached();
        SVE2_SUPPORT_CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
        ok
    }
}

/// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` in SVE2 `z0..z4`
/// registers when `FEAT_SVE2` is supported at runtime.
#[inline]
pub(crate) fn sve2_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> Option<(f32, f32, f32, f32)> {
    if !ArmSve2Backend::is_supported_cached() {
        return None;
    }
    let max_v = to_zreg([max_l, max_r, 1.0, 1.0]);
    let exp_mask: float32x4_t = zerocopy::transmute!([0x7f80_0000_u32; 4]);
    let cap_253: float32x4_t = zerocopy::transmute!([253_u32 << 23; 4]);
    let two54: float32x4_t = zerocopy::transmute!([254_u32 << 23; 4]);
    let (su_v, sd_v): (float32x4_t, float32x4_t);
    // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
    unsafe {
        core::arch::asm!(
            ".arch armv9-a+sve2",
            "ptrue p0.s, vl4",
            "and z4.d, z0.d, z1.d",
            "umin z4.s, p0/m, z4.s, z2.s",
            "sub z5.s, z3.s, z4.s",
            in("q0") max_v,
            in("q1") exp_mask,
            in("q2") cap_253,
            in("q3") two54,
            out("q4") su_v,
            out("q5") sd_v,
            out("p0") _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    let (sd, su) = (from_zreg(sd_v), from_zreg(su_v));
    Some((sd[0], su[0], sd[1], su[1]))
}

impl Crc32cBackend for ArmSve2Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        aarch64_crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSve2Backend {
    type Vec4 = Sve2Vec4;
    type Stereo8 = Sve2Stereo8;
    const NAME: &'static str = "ARMv9-A SVE2 (Scalable Vector Extension 2)";
    const KIND: BackendKind = BackendKind::ArmSve2;

    #[inline]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        if Self::is_supported_cached() {
            let clamped =
                Sve2Vec4::load(src).clamp(Sve2Vec4::splat(-65504.0), Sve2Vec4::splat(65504.0));
            let raw_v = to_zreg(*src);
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2` (`aarch64` NEON baseline).
            let inp = unsafe {
                let raw_bits = vreinterpretq_u32_f32(raw_v);
                let exp_mask = vdupq_n_u32(0x7F80_0000);
                let is_nonfinite = vceqq_u32(vandq_u32(raw_bits, exp_mask), exp_mask);
                vbslq_f32(is_nonfinite, raw_v, to_zreg(clamped.0))
            };
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "fcvt z0.h, p0/m, z0.s",
                    "uzp1 z0.h, z0.h, z0.h",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            let words: [u16; 8] = zerocopy::transmute!(out);
            [words[0], words[1], words[2], words[3]]
        } else {
            ScalarBackend::f32x4_to_fp16x4(src)
        }
    }

    #[inline]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        if Self::is_supported_cached() {
            let raw_in: [u16; 8] = [src[0], src[1], src[2], src[3], 0, 0, 0, 0];
            let inp: float32x4_t = zerocopy::transmute!(raw_in);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "uunpklo z0.s, z0.h",
                    "fcvt z0.s, p0/m, z0.h",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            from_zreg(out)
        } else {
            ScalarBackend::fp16x4_to_f32x4(src)
        }
    }

    #[inline]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        Self::rsqrt_nr2_pair_f32(v0, 1.0).0
    }

    #[inline]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        if Self::is_supported_cached() {
            let v = to_zreg([v0, v1, 1.0, 1.0]);
            let magic: float32x4_t = zerocopy::transmute!([0x5f37_59df_u32; 4]);
            let half = to_zreg([0.5_f32; 4]);
            let three_halves = to_zreg([1.5_f32; 4]);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\n",
                    "fmul z4.s, z0.s, z2.s\n",
                    "lsr z5.s, z0.s, #1\n",
                    "sub z5.s, z1.s, z5.s\n",
                    "fmul z6.s, z4.s, z5.s\n",
                    "fmul z6.s, z6.s, z5.s\n",
                    "fsub z7.s, z3.s, z6.s\n",
                    "fmul z5.s, z5.s, z7.s\n",
                    "fmul z6.s, z4.s, z5.s\n",
                    "fmul z6.s, z6.s, z5.s\n",
                    "fsub z7.s, z3.s, z6.s\n",
                    "fmul z5.s, z5.s, z7.s\n",
                    in("q0") v,
                    in("q1") magic,
                    in("q2") half,
                    in("q3") three_halves,
                    out("q4") _,
                    out("q5") out,
                    out("q6") _,
                    out("q7") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            let arr = from_zreg(out);
            (arr[0], arr[1])
        } else {
            ScalarBackend::rsqrt_nr2_pair_f32(v0, v1)
        }
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{InterleavedStereo8, StereoVec8};
    use proptest::prelude::*;

    use super::*;
    use crate::math::{ScalarVec4, warm_simd_feature_caches};

    #[test]
    fn test_sve2_backend_fp16_rsqrt_tracker_and_crc32c() {
        warm_simd_feature_caches();
        assert_eq!(
            ArmSve2Backend::is_supported(),
            ArmSve2Backend::is_supported_cached()
        );
        assert_eq!(ArmSve2Backend::KIND, BackendKind::ArmSve2);
        assert!(!ArmSve2Backend::NAME.is_empty());

        let tracked = sve2_tracker_exp_scales_f32(0.25, 4.0);
        if ArmSve2Backend::is_supported_cached() {
            let (sd_l, su_l, sd_r, su_r) = tracked.expect("SVE2 supported");
            assert!((sd_l * su_l - 1.0).abs() < 1.0e-6);
            assert!((sd_r * su_r - 1.0).abs() < 1.0e-6);
        } else {
            assert!(tracked.is_none());
            assert!(sve2_tracker_exp_scales_f32(0.0, 0.0).is_none());
        }

        assert_eq!(ArmSve2Backend::crc32c(b"123456789"), 0xE306_9283);
        let payload: [u8; 19] = core::array::from_fn(|i| (i as u8).wrapping_mul(13));
        for len in [0, 1, 7, 8, 9, 19] {
            assert_eq!(
                ArmSve2Backend::crc32c_update(0x1234_5678, &payload[..len]),
                ScalarBackend::crc32c_update(0x1234_5678, &payload[..len])
            );
        }

        let cases = [
            [0.5_f32, -1.5, 2.25, -0.0],
            [65504.0, -65504.0, 100_000.0, -100_000.0],
            [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0e-6],
        ];
        for src in cases {
            let hw = ArmSve2Backend::f32x4_to_fp16x4(&src);
            let sw = ScalarBackend::f32x4_to_fp16x4(&src);
            assert_eq!(hw, sw);
            assert_eq!(ArmSve2Backend::vec4_to_fp16x4(Sve2Vec4::load(&src)), hw);
            let rt_hw = ArmSve2Backend::fp16x4_to_f32x4(&hw);
            let rt_sw = ScalarBackend::fp16x4_to_f32x4(&hw);
            assert_eq!(rt_hw.map(f32::to_bits), rt_sw.map(f32::to_bits));
            assert_eq!(
                ArmSve2Backend::fp16x4_to_vec4(&hw)
                    .to_array()
                    .map(f32::to_bits),
                rt_sw.map(f32::to_bits)
            );
        }

        let r0 = ArmSve2Backend::rsqrt_nr2_f32(4.0);
        let (p0, p1) = ArmSve2Backend::rsqrt_nr2_pair_f32(4.0, 9.0);
        assert!((r0 - 0.5).abs() < 2.5e-3);
        assert!((p0 - 0.5).abs() < 2.5e-3);
        assert!((p1 - 1.0 / 3.0).abs() < 2.5e-3);
        if !ArmSve2Backend::is_supported_cached() {
            assert_eq!(r0.to_bits(), ScalarBackend::rsqrt_nr2_f32(4.0).to_bits());
            let (s0, s1) = ScalarBackend::rsqrt_nr2_pair_f32(4.0, 9.0);
            assert_eq!((p0.to_bits(), p1.to_bits()), (s0.to_bits(), s1.to_bits()));
        }

        let (c, s) = ArmSve2Backend::fast_unit_phasor(0.45);
        assert!((c * c + s * s - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn test_sve2_vec4_kernel_and_stereo8_parity() {
        warm_simd_feature_caches();
        let a_arr = [1.5_f32, -2.5, 3.5, -4.5];
        let b_arr = [2.0_f32, -4.0, 0.5, 8.0];
        let va = Sve2Vec4::from_array(a_arr);
        let vb = Sve2Vec4::load(&b_arr);
        let sa = ScalarVec4::from_array(a_arr);
        let sb = ScalarVec4::load(&b_arr);

        let mut stored = [0.0_f32; 4];
        va.store(&mut stored);
        assert_eq!(stored, a_arr);
        assert_eq!(Sve2Vec4::zero().to_array(), [0.0; 4]);
        assert_eq!(
            Sve2Vec4::splat_pair(1.25, -3.75).to_array(),
            ScalarVec4::splat_pair(1.25, -3.75).to_array()
        );
        assert_eq!(va.dup_low_pair().to_array(), sa.dup_low_pair().to_array());
        assert_eq!(va.dup_high_pair().to_array(), sa.dup_high_pair().to_array());
        assert_eq!(va.add(vb).to_array(), sa.add(sb).to_array());
        assert_eq!(va.sub(vb).to_array(), sa.sub(sb).to_array());
        assert_eq!(va.mul(vb).to_array(), sa.mul(sb).to_array());
        assert_eq!(va.fma(vb, vb).to_array(), sa.fma(sb, sb).to_array());
        assert_eq!(va.fms(vb, vb).to_array(), sa.fms(sb, sb).to_array());
        assert_eq!(
            va.mul_lane::<1>(vb).to_array(),
            sa.mul_lane::<1>(sb).to_array()
        );
        assert_eq!(
            va.mul_lane::<3>(vb).to_array(),
            sa.mul_lane::<3>(sb).to_array()
        );
        assert_eq!(
            va.fma_lane::<2>(vb, vb).to_array(),
            sa.fma_lane::<2>(sb, sb).to_array()
        );
        assert_eq!(va.max(vb).to_array(), sa.max(sb).to_array());
        assert_eq!(va.min(vb).to_array(), sa.min(sb).to_array());
        assert_eq!(va.abs().to_array(), sa.abs().to_array());
        assert_eq!(va.pair_swap_neg().to_array(), sa.pair_swap_neg().to_array());
        assert_eq!(va.horizontal_sum(), sa.horizontal_sum());
        assert_eq!(va.horizontal_sum_pair(vb), sa.horizontal_sum_pair(sb));
        assert_eq!(va.dot(vb), sa.dot(sb));

        let dirty = Sve2Vec4::from_array([f32::NAN, f32::INFINITY, f32::from_bits(1), -1.25]);
        let s_dirty = ScalarVec4::from_array([f32::NAN, f32::INFINITY, f32::from_bits(1), -1.25]);
        assert_eq!(
            dirty.scrub_recursive_state().to_array().map(f32::to_bits),
            s_dirty.scrub_recursive_state().to_array().map(f32::to_bits)
        );
        assert_eq!(
            <Sve2Vec4 as dnam_kernels::KernelVec4>::neg(va).to_array(),
            <ScalarVec4 as dnam_kernels::KernelVec4>::neg(sa).to_array()
        );
        assert_eq!(
            <Sve2Vec4 as dnam_kernels::KernelVec4>::select_le(va, vb, va, vb).to_array(),
            <ScalarVec4 as dnam_kernels::KernelVec4>::select_le(sa, sb, sa, sb).to_array()
        );
        assert_eq!(
            <Sve2Vec4 as dnam_kernels::KernelVec4>::all_le(va, vb),
            <ScalarVec4 as dnam_kernels::KernelVec4>::all_le(sa, sb)
        );
        assert_eq!(
            <Sve2Vec4 as dnam_kernels::KernelVec4>::flush_recursive_state(dirty)
                .to_array()
                .map(f32::to_bits),
            <ScalarVec4 as dnam_kernels::KernelVec4>::flush_recursive_state(s_dirty)
                .to_array()
                .map(f32::to_bits)
        );

        let rows = [
            Sve2Vec4::from_array([1.0, 2.0, 3.0, 4.0]),
            Sve2Vec4::from_array([5.0, 6.0, 7.0, 8.0]),
            Sve2Vec4::from_array([9.0, 10.0, 11.0, 12.0]),
            Sve2Vec4::from_array([13.0, 14.0, 15.0, 16.0]),
        ];
        let t_hw = ArmSve2Backend::transpose_4x4(rows[0], rows[1], rows[2], rows[3]);
        assert_eq!(t_hw[0].to_array(), [1.0, 5.0, 9.0, 13.0]);
        assert_eq!(t_hw[3].to_array(), [4.0, 8.0, 12.0, 16.0]);

        let st_a = Sve2Stereo8::load_lr(&a_arr, &b_arr);
        let st_b = Sve2Stereo8::splat_lr(0.5, -0.5);
        let sc_a = InterleavedStereo8::<ScalarVec4>::load_lr(&a_arr, &b_arr);
        let sc_b = InterleavedStereo8::<ScalarVec4>::splat_lr(0.5, -0.5);
        assert_eq!(st_a.horizontal_sum_lr(), sc_a.horizontal_sum_lr());
        assert_eq!(
            st_a.fma(st_b, st_b).horizontal_sum_lr(),
            sc_a.fma(sc_b, sc_b).horizontal_sum_lr()
        );
        let cols = [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.75, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(
            Sve2Stereo8::fma_mat4x4_shared(Sve2Stereo8::zero(), &cols, st_a).horizontal_sum_lr(),
            InterleavedStereo8::<ScalarVec4>::fma_mat4x4_shared(
                InterleavedStereo8::<ScalarVec4>::zero(),
                &cols,
                sc_a,
            )
            .horizontal_sum_lr()
        );
        let mv = ArmSve2Backend::fma_mat4x4_vec4(va, &cols, &a_arr);
        let smv = ScalarBackend::fma_mat4x4_vec4(sa, &cols, &a_arr);
        assert_eq!(mv.to_array(), smv.to_array());
        let (sl, sr) = ArmSve2Backend::fma_mat4x4_stereo(va, vb, &cols, va, vb);
        let (cl, cr) = ScalarBackend::fma_mat4x4_stereo(sa, sb, &cols, sa, sb);
        assert_eq!(sl.to_array(), cl.to_array());
        assert_eq!(sr.to_array(), cr.to_array());
        let pos = Sve2Vec4::from_array([1.0, 2.0, 4.0, 8.0]);
        let spos = ScalarVec4::from_array([1.0, 2.0, 4.0, 8.0]);
        let (pe, spe) = (
            pos.exp2_clamped5().to_array(),
            spos.exp2_clamped5().to_array(),
        );
        let (pr, spr) = (pos.rcp_pos_nr3().to_array(), spos.rcp_pos_nr3().to_array());
        let (ps, sps) = (
            pos.rsqrt_pos_nr3().to_array(),
            spos.rsqrt_pos_nr3().to_array(),
        );
        for i in 0..4 {
            assert!((pe[i] - spe[i]).abs() < 1.0e-5);
            assert!((pr[i] - spr[i]).abs() < 1.0e-6);
            assert!((ps[i] - sps[i]).abs() < 1.0e-6);
        }
    }

    proptest! {
        #[test]
        fn prop_sve2_backend_fp16_and_rsqrt_parity(
            xs in prop::array::uniform4(-1000.0_f32..1000.0_f32),
            pos in prop::array::uniform2(1.0e-3_f32..1.0e4_f32),
        ) {
            warm_simd_feature_caches();
            let hw = ArmSve2Backend::f32x4_to_fp16x4(&xs);
            prop_assert_eq!(hw, ScalarBackend::f32x4_to_fp16x4(&xs));
            prop_assert_eq!(
                ArmSve2Backend::fp16x4_to_f32x4(&hw).map(f32::to_bits),
                ScalarBackend::fp16x4_to_f32x4(&hw).map(f32::to_bits)
            );
            let (r0, r1) = ArmSve2Backend::rsqrt_nr2_pair_f32(pos[0], pos[1]);
            let ex0 = 1.0 / pos[0].sqrt();
            let ex1 = 1.0 / pos[1].sqrt();
            prop_assert!(((r0 - ex0) / ex0).abs() < 2.5e-3);
            prop_assert!(((r1 - ex1) / ex1).abs() < 2.5e-3);
        }
    }
}
