//! Shared unit-test verification harness (`packages/engine/src/test_support.rs`).
//!
//! Provides:
//! 1. Exact rational (`i128` / `u128` + multi-word) arithmetic matching Python's `fractions.Fraction`.
//! 2. IEEE-754 `f32`/`f64` bit-exact conversions, ULP distance, outward rounding, and rank-space helpers.
//! 3. Canonical 15-rate sample-rate sweep table (`44.1 kHz`..`384 kHz`) and deterministic `SplitMix64` PRNG.

#![allow(dead_code)]

/// Canonical 15 sample rates across `44.1 kHz`..`384 kHz` (`H1` Hilbert anchor grid & ZOH sweep).
pub(crate) const CANONICAL_SAMPLE_RATES: [u32; 15] = [
    44_100, 48_000, 64_000, 88_200, 96_000, 128_000, 176_400, 192_000, 220_500, 240_000, 256_000,
    288_000, 320_000, 352_800, 384_000,
];

/// Deterministic `SplitMix64` PRNG for reproducible property and sweep tests.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    pub(crate) fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform `f32` in `[lo, hi]`.
    pub(crate) fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let t = self.next_unit_f64();
        (f64::from(lo) + (f64::from(hi) - f64::from(lo)) * t) as f32
    }
}

/// Greatest common divisor of two unsigned 128-bit integers.
pub(crate) const fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Exact rational number `num / den` (`den > 0`, reduced by `gcd`) backed by `i128` / `u128`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rat {
    pub(crate) num: i128,
    pub(crate) den: i128,
}

impl Rat {
    pub(crate) fn new(num: i128, den: i128) -> Self {
        assert!(den != 0, "Rat denominator must be non-zero");
        if num == 0 {
            return Self { num: 0, den: 1 };
        }
        let sign = if (num < 0) ^ (den < 0) { -1i128 } else { 1i128 };
        let g = gcd_u128(num.unsigned_abs(), den.unsigned_abs()) as i128;
        Self {
            num: sign * (num.unsigned_abs() as i128 / g),
            den: (den.unsigned_abs() as i128 / g),
        }
    }

    pub(crate) const fn from_int(n: i128) -> Self {
        Self { num: n, den: 1 }
    }

    pub(crate) fn add(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
    }

    pub(crate) fn sub(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }

    pub(crate) fn mul(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.num, self.den * rhs.den)
    }

    pub(crate) fn div(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den, self.den * rhs.num)
    }

    pub(crate) fn abs(self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den,
        }
    }

    pub(crate) fn floor(self) -> i128 {
        self.num.div_euclid(self.den)
    }

    pub(crate) fn ceil(self) -> i128 {
        -((-self.num).div_euclid(self.den))
    }

    /// Rounds `self` to the nearest integer with halfway ties to even (`RNE`).
    pub(crate) fn round_ties_to_even(self) -> i128 {
        let q = self.num.div_euclid(self.den);
        let r = self.num.rem_euclid(self.den);
        let twice_r = 2 * r;
        if twice_r > self.den || (twice_r == self.den && (q & 1) != 0) {
            q + 1
        } else {
            q
        }
    }

    /// Rounds `self >= 0` to `(integer, q32_fraction)` with ties to even on `self * 2^32`.
    pub(crate) fn round_q32(self) -> (u64, u32) {
        assert!(self.num >= 0, "round_q32 requires non-negative value");
        let scaled = Self::new(self.num << 32, self.den);
        let whole = scaled.round_ties_to_even() as u128;
        ((whole >> 32) as u64, (whole & 0xFFFF_FFFF) as u32)
    }

    pub(crate) fn to_f64(self) -> f64 {
        (self.num as f64) / (self.den as f64)
    }

    pub(crate) fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
}

/// Bitwise-exact widening from `f32` raw bits to `f64` (immune to FTZ/DAZ on subnormal inputs).
pub(crate) fn widen_f32_bits_to_f64(bits: u32) -> Option<f64> {
    let sign = u64::from(bits & 0x8000_0000) << 32;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    let wide_bits = match (exponent, fraction) {
        (0xff, _) => return None,
        (0, 0) => sign,
        (0, fraction) => {
            let leading = 31 - fraction.leading_zeros();
            let exponent64 = u64::from(leading + 874);
            let fraction64 = u64::from(fraction - (1 << leading)) << (52 - leading);
            sign | (exponent64 << 52) | fraction64
        }
        (exponent, fraction) => {
            sign | (u64::from(exponent + 896) << 52) | (u64::from(fraction) << 29)
        }
    };
    Some(f64::from_bits(wide_bits))
}

/// ULP distance between two finite `f32` values (treating `-0.0` and `+0.0` as distance 0).
pub(crate) fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    assert!(a.is_finite() && b.is_finite());
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let to_ordered = |x: f32| -> i32 {
        let bits = x.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    to_ordered(a).abs_diff(to_ordered(b))
}

/// ULP distance between two finite `f64` values (treating `-0.0` and `+0.0` as distance 0).
pub(crate) fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    assert!(a.is_finite() && b.is_finite());
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let to_ordered = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    to_ordered(a).abs_diff(to_ordered(b))
}

#[cfg(test)]
pub(crate) fn build_test_dnam_bin(
    max_tier: crate::types::Tier,
    ref_sr_hz: u32,
    header_flags: u16,
    controls: &[dnam_vm::bin::ControlClampDescriptorWire],
    tsyn: &[dnam_vm::bin::TempoSyncEntryWire],
    passthrough: bool,
) -> alloc::vec::Vec<u8> {
    use dnam_vm::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DnamBinBuildInput,
        encode_dnam_bin,
    };
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    let tier_mask: u16 = match max_tier {
        crate::types::Tier::Lite => 0x0001,
        crate::types::Tier::Full => 0x0003,
        crate::types::Tier::Pro => 0x0007,
        crate::types::Tier::Ultra => 0x000F,
    };
    a0.crossbar.cell_desc.tier_flags = U16::new(tier_mask);
    if passthrough {
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
    }
    let a1 = Arena1FullDeltaWire::new_zeroed();
    let a2 = Arena2ProDeltaWire::new_zeroed();
    let a3 = Arena3UltraDeltaWire::new_zeroed();
    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: ref_sr_hz,
        header_flags,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: 0.995,
        tail_time_ms: 0.0,
        ro_pole_48k: 0.0,
        switch_xfade_samples: 128,
        content_hash_prefix: [0; 8],
        controls,
        cp1_overflow: &[],
        tsyn,
        cext: None,
        arena0: &a0,
        arena1: (max_tier >= crate::types::Tier::Full).then_some(&a1),
        arena2: (max_tier >= crate::types::Tier::Pro).then_some(&a2),
        arena3: (max_tier >= crate::types::Tier::Ultra).then_some(&a3),
    })
    .expect("valid test dnam-bin")
}

#[cfg(test)]
pub(crate) fn build_test_dnam_container_with_arena0_and_dnbn_pol(
    header: &mut crate::schema::DnamJsonHeader,
    max_tier: crate::types::Tier,
    mut a0: dnam_vm::bin::Arena0LiteWire,
    dnbn_wet_pol_inv: bool,
) -> alloc::vec::Vec<u8> {
    use dnam_vm::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_AUTO_UNITY,
        DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, DNBN_FLAG_WET_POLARITY_INV,
        DnamBinBuildInput, encode_dnam_bin,
    };
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::U16;

    let descriptors = header
        .compile_clamp_descriptors()
        .expect("compile test descriptors");
    let wire_ctrls: alloc::vec::Vec<_> = descriptors.iter().map(|d| d.to_wire()).collect();
    let mut flags = 0u16;
    if header.io_config.inputs == 1 {
        flags |= DNBN_FLAG_SINGLE_INPUT;
    }
    if header.io_config.output_mode != "mono" {
        flags |= DNBN_FLAG_STEREO_OUT;
    }
    if dnbn_wet_pol_inv {
        flags |= DNBN_FLAG_WET_POLARITY_INV;
    }
    if header.io_config.leveling.auto_unity_enabled {
        flags |= DNBN_FLAG_AUTO_UNITY;
    }
    let tier_mask: u16 = match max_tier {
        crate::types::Tier::Lite => 0x0001,
        crate::types::Tier::Full => 0x0003,
        crate::types::Tier::Pro => 0x0007,
        crate::types::Tier::Ultra => 0x000F,
    };
    a0.crossbar.cell_desc.tier_flags = U16::new(tier_mask);
    let a1 = Arena1FullDeltaWire::new_zeroed();
    let a2 = Arena2ProDeltaWire::new_zeroed();
    let a3 = Arena3UltraDeltaWire::new_zeroed();
    let dbin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: header.sample_rate_train_hz,
        header_flags: flags,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: header.transport.dc_blocker_pole,
        tail_time_ms: header.io_config.tail_time_ms,
        ro_pole_48k: header.linear_readout.ro_pole_48k,
        switch_xfade_samples: header.transport.switch_xfade_samples,
        content_hash_prefix: [0; 8],
        controls: &wire_ctrls,
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: (max_tier >= crate::types::Tier::Full).then_some(&a1),
        arena2: (max_tier >= crate::types::Tier::Pro).then_some(&a2),
        arena3: (max_tier >= crate::types::Tier::Ultra).then_some(&a3),
    })
    .expect("encode test dnam-bin");
    crate::container::build_dnam_container_from_dnam_bin(header, &dbin)
        .expect("build test dnam container")
}

#[cfg(test)]
pub(crate) fn build_test_dnam_container_with_arena0(
    header: &mut crate::schema::DnamJsonHeader,
    max_tier: crate::types::Tier,
    a0: dnam_vm::bin::Arena0LiteWire,
) -> alloc::vec::Vec<u8> {
    let pol = header.io_config.capabilities.wet_polarity_inverted;
    build_test_dnam_container_with_arena0_and_dnbn_pol(header, max_tier, a0, pol)
}

#[cfg(test)]
pub(crate) fn build_test_dnam_container(
    header: &mut crate::schema::DnamJsonHeader,
    max_tier: crate::types::Tier,
    passthrough: bool,
) -> alloc::vec::Vec<u8> {
    use dnam_vm::bin::DnamBinBuildInput;
    use zerocopy::byteorder::little_endian::F32;

    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    if passthrough {
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
    }
    build_test_dnam_container_with_arena0(header, max_tier, a0)
}

#[cfg(test)]
pub(crate) fn build_spatial_test_dnam_container_with_dnbn_polarity(
    header: &mut crate::schema::DnamJsonHeader,
    max_tier: crate::types::Tier,
    dnbn_wet_pol_inv: bool,
) -> alloc::vec::Vec<u8> {
    use dnam_kernels::f32_to_fp16;
    use dnam_vm::bin::DnamBinBuildInput;
    use zerocopy::byteorder::little_endian::{F32, U16};

    header.io_config.tail_time_ms = header.io_config.tail_time_ms.max(120.0);
    if header.linear_readout.ro_pole_48k == 0.0 {
        header.linear_readout.ro_pole_48k = 0.99;
    }
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.core_quads[0].in_mid_weight_fp16 = [0.6, 0.3, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
    a0.core_quads[0].in_side_weight_fp16 =
        [0.25, -0.15, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
    a0.core_quads[0].svf_g = [0.28; 4].map(F32::new);
    a0.core_quads[0].svf_k_fp16 = [U16::new(f32_to_fp16(0.707)); 4];
    a0.core_quads[0].adaa_drive_fp16 = [U16::new(f32_to_fp16(1.2)); 4];
    a0.core_quads[0].tone_lp_fp16 = [U16::new(f32_to_fp16(0.4)); 4];
    a0.core_quads[0].tone_dir_fp16 = [U16::new(f32_to_fp16(0.8)); 4];
    a0.core_quads[0].lru_alpha_rel = [0.2; 4].map(F32::new);
    a0.core_quads[0].lru_alpha_atk = [0.9; 4].map(F32::new);
    a0.core_quads[0].lru_kappa_fp16 = [U16::new(f32_to_fp16(0.25)); 4];
    a0.conv_fir_lite.fir_l[0] = F32::new(0.75);
    a0.conv_fir_lite.fir_r[0] = F32::new(0.65);
    a0.conv_fir_lite.fir_l[1] = F32::new(0.15);
    a0.conv_fir_lite.fir_r[1] = F32::new(-0.1);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.8), F32::new(0.7)];
    a0.linear_readout_lite.w_ro_fp16[0] = U16::new(f32_to_fp16(0.4));
    a0.linear_readout_lite.w_ro_fp16[1] = U16::new(f32_to_fp16(0.3));
    a0.linear_readout_lite.w_ro_fp16[6] = U16::new(f32_to_fp16(0.5));
    a0.linear_readout_lite.w_ro_fp16[7] = U16::new(f32_to_fp16(0.45));
    a0.transport_lite.delay_samples = [4.0, 7.0, 11.0, 15.0].map(F32::new);
    a0.transport_lite.g_loop = F32::new(0.55);
    a0.transport_lite.input_gains_fp16 = [0.5, 0.35, -0.25, 0.3].map(|v| U16::new(f32_to_fp16(v)));
    a0.transport_lite.wet_tap_l_fp16 = [0.4, -0.25, 0.3, 0.15].map(|v| U16::new(f32_to_fp16(v)));
    a0.transport_lite.wet_tap_r_fp16 = [-0.25, 0.45, 0.15, 0.3].map(|v| U16::new(f32_to_fp16(v)));
    a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.75));
    build_test_dnam_container_with_arena0_and_dnbn_pol(header, max_tier, a0, dnbn_wet_pol_inv)
}

#[cfg(test)]
pub(crate) fn build_spatial_test_dnam_container(
    header: &mut crate::schema::DnamJsonHeader,
    max_tier: crate::types::Tier,
) -> alloc::vec::Vec<u8> {
    let pol = header.io_config.capabilities.wet_polarity_inverted;
    build_spatial_test_dnam_container_with_dnbn_polarity(header, max_tier, pol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitmix64_rat_and_ieee754_helpers() {
        let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
        let _u32_v = rng.next_u32();
        let u_f64 = rng.next_unit_f64();
        assert!((0.0..1.0).contains(&u_f64));
        let r_f32 = rng.next_f32_range(-2.0, 2.0);
        assert!((-2.0..=2.0).contains(&r_f32));

        let zero = Rat::new(0, 5);
        assert_eq!(zero, Rat::from_int(0));
        let a = Rat::new(-6, 8);
        let b = Rat::new(1, -2);
        assert_eq!(a.add(b), Rat::new(-5, 4));
        assert_eq!(a.sub(b), Rat::new(-1, 4));
        assert_eq!(a.mul(b), Rat::new(3, 8));
        assert_eq!(a.div(b), Rat::new(3, 2));
        assert_eq!(a.abs(), Rat::new(3, 4));
        assert_eq!(a.floor(), -1);
        assert_eq!(a.ceil(), 0);
        assert_eq!(Rat::new(5, 2).round_ties_to_even(), 2);
        assert_eq!(Rat::new(7, 2).round_ties_to_even(), 4);
        assert_eq!(Rat::new(3, 4).round_ties_to_even(), 1);
        assert_eq!(Rat::new(1, 4).round_ties_to_even(), 0);
        let (q_int, q_frac) = Rat::new(3, 2).round_q32();
        assert_eq!(q_int, 1);
        assert_eq!(q_frac, 0x8000_0000);
        assert!((Rat::new(1, 2).to_f64() - 0.5).abs() < 1e-12);
        assert!((Rat::new(1, 2).to_f32() - 0.5).abs() < 1e-6);

        assert_eq!(widen_f32_bits_to_f64(f32::NAN.to_bits()), None);
        assert_eq!(widen_f32_bits_to_f64(0), Some(0.0));
        assert_eq!(widen_f32_bits_to_f64(1), Some(f64::from(f32::from_bits(1))));
        assert_eq!(widen_f32_bits_to_f64(1.5_f32.to_bits()), Some(1.5));

        assert_eq!(ulp_distance_f32(0.0, -0.0), 0);
        assert_eq!(
            ulp_distance_f32(1.0, f32::from_bits(1.0_f32.to_bits() + 1)),
            1
        );
        assert_eq!(
            ulp_distance_f32(-1.0, f32::from_bits((-1.0_f32).to_bits() + 1)),
            1
        );
        assert_eq!(ulp_distance_f64(0.0, -0.0), 0);
        assert_eq!(
            ulp_distance_f64(1.0, f64::from_bits(1.0_f64.to_bits() + 1)),
            1
        );
        assert_eq!(
            ulp_distance_f64(-1.0, f64::from_bits((-1.0_f64).to_bits() + 1)),
            1
        );
    }
}
