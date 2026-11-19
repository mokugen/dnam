//! Operating-envelope sanitization (`±100.0` certified input, `±1e9` safety ceiling, `fp16`
//! saturation) and telemetry flags (`ref[arch.fp.range-saturation]`).

/// Error returned when a non-finite or out-of-domain floating-point bit pattern is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidFloatInput;

/// Event bitflags for `overflow_or_saturation_flags` (`ref[arch.fp.range-saturation]`).
pub mod saturation_flags {
    /// Bit 0: A processed audio input exceeded the `±1e9` safety ceiling.
    pub const INPUT_SATURATED: u32 = 1 << 0;
    /// Bit 1: A finite delay write exceeded `±65,504.0` (`0x7BFF` / `0xFBFF`).
    pub const FP16_SATURATED: u32 = 1 << 1;
    /// Bit 2: An intermediate or recursive-state safety guard clamped a value to `±1e9`.
    pub const INTERNAL_SATURATED: u32 = 1 << 2;
    /// Bit 3: The processed output guard clamped a value to `±1e9`.
    pub const OUTPUT_SATURATED: u32 = 1 << 3;
    /// Bit 4: An input exceeded the model's certified input envelope (`>= ±100.0`).
    pub const OUTSIDE_CERTIFIED_INPUT: u32 = 1 << 4;
    /// Bit 5: A non-finite external sample, control, or parameter was detected and rejected.
    pub const INVALID_NONFINITE: u32 = 1 << 5;
}

/// Hard safety ceiling `S = 1e9` (`ref[arch.fp.range-saturation]`).
pub const SAFETY_CEILING_F32: f32 = 1.0e9;

/// Minimum certified unclipped input envelope `±100.0` (`+40 dBFS`, `ref[arch.fp.range-saturation]`).
pub const CERTIFIED_INPUT_ENVELOPE_MIN_F32: f32 = 100.0;

/// Maximum finite `fp16` magnitude `65,504.0` (`0x7BFF`, `ref[arch.fp.range-saturation]`).
pub const FP16_MAX_FINITE_F32: f32 = 65_504.0;

/// Sanitizes and clamps an input sample against `certified_limit >= 100.0` and `±1e9` (`ref[arch.fp.range-saturation]`).
#[inline]
pub fn sanitize_input_sample(x: f32, certified_limit: f32, flags: &mut u32) -> f32 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0.0;
    }
    let cert = certified_limit.max(CERTIFIED_INPUT_ENVELOPE_MIN_F32);
    if x.abs() > cert {
        *flags |= saturation_flags::OUTSIDE_CERTIFIED_INPUT;
    }
    if x.abs() > SAFETY_CEILING_F32 {
        *flags |= saturation_flags::INPUT_SATURATED;
        return x.clamp(-SAFETY_CEILING_F32, SAFETY_CEILING_F32);
    }
    x
}

/// Sanitizes and clamps an internal state value against `±1e9` (`ref[arch.fp.range-saturation]`).
#[inline]
pub fn sanitize_internal_sample(x: f32, flags: &mut u32) -> f32 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0.0;
    }
    if x.abs() > SAFETY_CEILING_F32 {
        *flags |= saturation_flags::INTERNAL_SATURATED;
        return x.clamp(-SAFETY_CEILING_F32, SAFETY_CEILING_F32);
    }
    x
}

/// Sanitizes and clamps an output sample against `±1e9` (`ref[arch.fp.range-saturation]`).
#[inline]
pub fn sanitize_output_sample(x: f32, flags: &mut u32) -> f32 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0.0;
    }
    if x.abs() > SAFETY_CEILING_F32 {
        *flags |= saturation_flags::OUTPUT_SATURATED;
        return x.clamp(-SAFETY_CEILING_F32, SAFETY_CEILING_F32);
    }
    x
}

/// Converts a finite `f32` sample to `fp16` bits while recording `FP16_SATURATED` if `|x| > 65504.0`
/// (`ref[arch.fp.range-saturation]`, `ref[dsp.op-t.ring.fp16-saturation]`).
#[inline]
pub fn f32_to_fp16_with_telemetry(x: f32, flags: &mut u32) -> u16 {
    if !x.is_finite() {
        *flags |= saturation_flags::INVALID_NONFINITE;
        return 0;
    }
    if x.abs() > FP16_MAX_FINITE_F32 {
        *flags |= saturation_flags::FP16_SATURATED;
    }
    crate::math::fp16::f32_to_fp16(x.clamp(-FP16_MAX_FINITE_F32, FP16_MAX_FINITE_F32))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::{
        DefaultMathBackend, MathBackend, ScalarBackend, SimdVec4, flush_subnormal_f32_bits,
    };
    use crate::operators::OpNBackend;
    use crate::{FtzDazGuard, Tier};

    /// verify[arch.fp.callback-env]
    #[test]
    fn test_fp_callback_environment_and_subnormal_flushing() {
        let _guard = FtzDazGuard::new();
        let one = std::hint::black_box(1.0_f32);
        let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
        assert_eq!((one + half_ulp).to_bits(), 1.0_f32.to_bits());

        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x0000_0001)).to_bits(),
            0
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x807f_ffff)).to_bits(),
            0x8000_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x0080_0000)).to_bits(),
            0x0080_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x7f80_0000)).to_bits(),
            0x7f80_0000
        );
        assert_eq!(
            flush_subnormal_f32_bits(f32::from_bits(0x7fc0_1234)).to_bits(),
            0x7fc0_1234
        );
    }

    /// verify[arch.fp.range-saturation]
    #[test]
    fn test_range_saturation_policy_and_all_six_flags() {
        assert_eq!(CERTIFIED_INPUT_ENVELOPE_MIN_F32, 100.0);
        assert_eq!(SAFETY_CEILING_F32, 1.0e9);
        assert_eq!(FP16_MAX_FINITE_F32, 65_504.0);

        let mut flags = 0u32;
        assert_eq!(sanitize_input_sample(100.0, 100.0, &mut flags), 100.0);
        assert_eq!(flags, 0);
        assert_eq!(sanitize_input_sample(100.5, 100.0, &mut flags), 100.5);
        assert_eq!(flags, saturation_flags::OUTSIDE_CERTIFIED_INPUT);

        flags = 0;
        assert_eq!(
            sanitize_input_sample(2.0e9, 100.0, &mut flags),
            SAFETY_CEILING_F32
        );
        assert_eq!(
            flags,
            saturation_flags::OUTSIDE_CERTIFIED_INPUT | saturation_flags::INPUT_SATURATED
        );

        flags = 0;
        assert_eq!(
            sanitize_internal_sample(-2.0e9, &mut flags),
            -SAFETY_CEILING_F32
        );
        assert_eq!(flags, saturation_flags::INTERNAL_SATURATED);

        flags = 0;
        assert_eq!(
            sanitize_output_sample(5.0e9, &mut flags),
            SAFETY_CEILING_F32
        );
        assert_eq!(flags, saturation_flags::OUTPUT_SATURATED);

        flags = 0;
        assert_eq!(f32_to_fp16_with_telemetry(65_504.0, &mut flags), 0x7bff);
        assert_eq!(flags, 0);
        assert_eq!(f32_to_fp16_with_telemetry(70_000.0, &mut flags), 0x7bff);
        assert_eq!(flags, saturation_flags::FP16_SATURATED);

        for (out, got_flags) in [
            (
                sanitize_input_sample(f32::NAN, 100.0, {
                    flags = 0;
                    &mut flags
                }),
                flags,
            ),
            (
                sanitize_internal_sample(f32::NAN, {
                    flags = 0;
                    &mut flags
                }),
                flags,
            ),
            (
                sanitize_output_sample(f32::NAN, {
                    flags = 0;
                    &mut flags
                }),
                flags,
            ),
        ] {
            assert_eq!(out, 0.0);
            assert_eq!(got_flags, saturation_flags::INVALID_NONFINITE);
        }
        flags = 0;
        assert_eq!(f32_to_fp16_with_telemetry(f32::NAN, &mut flags), 0);
        assert_eq!(flags, saturation_flags::INVALID_NONFINITE);
    }

    /// verify[arch.simd.lane-packing]
    #[test]
    fn test_simd_4lane_packing_and_tier_hierarchy() {
        assert_eq!((Tier::Lite.channels(), Tier::Lite.simd_groups()), (16, 4));
        assert_eq!((Tier::Full.channels(), Tier::Full.simd_groups()), (24, 6));
        assert_eq!((Tier::Pro.channels(), Tier::Pro.simd_groups()), (32, 8));
        assert_eq!(
            (Tier::Ultra.channels(), Tier::Ultra.simd_groups()),
            (40, 10)
        );

        let _guard = FtzDazGuard::new();
        let x = [-2.5_f32, -0.5, 0.0, 1.5];
        let s_out =
            ScalarBackend::tanh_pade5_4ch(<ScalarBackend as MathBackend>::Vec4::from_array(x))
                .to_array();
        let d_out = DefaultMathBackend::tanh_pade5_4ch(
            <DefaultMathBackend as MathBackend>::Vec4::from_array(x),
        )
        .to_array();
        for lane in 0..4 {
            assert!((s_out[lane] - d_out[lane]).abs() < 1e-6);
        }
    }

    proptest! {
        #[test]
        fn prop_sanitization_always_finite_and_bounded(raw_x in prop::num::f32::ANY) {
            let mut flags = 0u32;
            let s_in = sanitize_input_sample(raw_x, 100.0, &mut flags);
            prop_assert!(s_in.is_finite() && s_in.abs() <= SAFETY_CEILING_F32);
            let s_out = sanitize_output_sample(raw_x, &mut flags);
            prop_assert!(s_out.is_finite() && s_out.abs() <= SAFETY_CEILING_F32);
        }
    }
}
