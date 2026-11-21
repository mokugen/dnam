//! `T1` 160-bit midpoint DC blocker coefficient synthesis, passband filter, and SIMD exponent-split scaling.

use crate::math::flush_recursive_state_f32;

/// Canonical `5 Hz` `T1` DC blocker pole at `48 kHz` (`dnam_kernels::DC_BLOCKER_5HZ_48K_F32`,
/// `ref[dsp.op-m.tracker.t1-dc-blocker]`).
pub(crate) const T1_DC_BLOCKER_5HZ_48K_POLE: f32 = dnam_kernels::DC_BLOCKER_5HZ_48K_F32;

/// Canonical `5 Hz` `T1` DC blocker passband-normalized gain `0.5 * (1 + R)` at `48 kHz`
/// (`0x3F7F_EA90`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
pub(crate) const T1_DC_BLOCKER_5HZ_48K_GAIN: f32 = 0.5 * (1.0 + T1_DC_BLOCKER_5HZ_48K_POLE);

const _: () = assert!(T1_DC_BLOCKER_5HZ_48K_POLE.to_bits() == 0x3F7F_D51F);
const _: () = assert!(T1_DC_BLOCKER_5HZ_48K_GAIN.to_bits() == 0x3F7F_EA90);

/// Shared `T1` `5 Hz` 1-pole DC blocker step scrubbing `NaN`/`±Inf` and flushing subnormals on all
/// recursive state (`prev_x`, `prev_d`).
#[inline(always)]
pub(crate) fn step_t1_dc_blocker_state(
    pole: f32,
    gain: f32,
    prev_x: &mut f32,
    prev_d: &mut f32,
    x: f32,
) -> f32 {
    let x_clean = flush_recursive_state_f32(x);
    let prev_x_clean = flush_recursive_state_f32(*prev_x);
    let prev_d_clean = flush_recursive_state_f32(*prev_d);
    let d = flush_recursive_state_f32(gain * (x_clean - prev_x_clean) + pole * prev_d_clean);
    *prev_x = x_clean;
    *prev_d = d;
    d
}

/// Computes the `T1` tracker `5 Hz` DC blocker `(pole_bits, gain_bits)` at `rate_hz` (`44.1..=384 kHz`)
/// via [`dnam_kernels::compute_t1_dc_blocker_coeffs_bits`] using the unified 160-bit midpoint
/// binary search without `libm` calls (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[inline]
pub fn compute_t1_dc_blocker_coeffs_bits(rate_hz: f64) -> Option<(u32, u32)> {
    dnam_kernels::compute_t1_dc_blocker_coeffs_bits(rate_hz)
}

/// Tracker-only passband-normalized `5 Hz` one-pole DC blocker (`T1`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug)]
pub struct T1DcBlocker {
    pub pole: f32,
    pub gain: f32,
    pub prev_x: f32,
    pub prev_d: f32,
}

impl Default for T1DcBlocker {
    fn default() -> Self {
        Self::new_48khz()
    }
}

impl T1DcBlocker {
    /// Creates a `48 kHz` `T1` DC blocker initialized from [`dnam_kernels::DC_BLOCKER_5HZ_48K_F32`].
    pub const fn new_48khz() -> Self {
        Self {
            pole: T1_DC_BLOCKER_5HZ_48K_POLE,
            gain: T1_DC_BLOCKER_5HZ_48K_GAIN,
            prev_x: 0.0,
            prev_d: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_d = 0.0;
    }

    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate_hz)?;
        Some(Self {
            pole: f32::from_bits(p_bits),
            gain: f32::from_bits(g_bits),
            prev_x: 0.0,
            prev_d: 0.0,
        })
    }

    /// Advances the `T1` `5 Hz` DC blocker by one sample while scrubbing `NaN`/`±Inf` and flushing
    /// subnormals on all recursive state.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        step_t1_dc_blocker_state(self.pole, self.gain, &mut self.prev_x, &mut self.prev_d, x)
    }
}

/// Domain-specific SIMD backend trait for `Op-M` (Symplectic Modal, Hilbert & Sub-Harmonic Tracker).
pub trait OpMBackend {
    /// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` for normal positive
    /// maxima `(max_l, max_r)` inside SIMD registers without scalar `fmov` / `movd` transfers.
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32);
}

/// Portable scalar reference implementation of [`OpMBackend::tracker_exp_scales_f32`].
#[inline(always)]
pub fn tracker_exp_scales_scalar_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    const F32_EXP_SHIFT: u32 = 23;
    const F32_EXP_MASK: u32 = 0xFF;
    const EXP_SUM_BIAS: u32 = 2 * dnam_kernels::IEEE754_F32_EXP_BIAS;
    const MAX_BIASED_EXP: u32 = EXP_SUM_BIAS - 1;
    let cl_l = ((max_l.to_bits() >> F32_EXP_SHIFT) & F32_EXP_MASK).min(MAX_BIASED_EXP);
    let cl_r = ((max_r.to_bits() >> F32_EXP_SHIFT) & F32_EXP_MASK).min(MAX_BIASED_EXP);
    (
        f32::from_bits((EXP_SUM_BIAS - cl_l) << F32_EXP_SHIFT),
        f32::from_bits(cl_l << F32_EXP_SHIFT),
        f32::from_bits((EXP_SUM_BIAS - cl_r) << F32_EXP_SHIFT),
        f32::from_bits(cl_r << F32_EXP_SHIFT),
    )
}

impl OpMBackend for crate::math::ScalarBackend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        tracker_exp_scales_scalar_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpMBackend for crate::math::NeonBackend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::aarch64::neon::neon_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpMBackend for crate::math::ArmSve2Backend {
    #[inline]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::aarch64::sve2::sve2_tracker_exp_scales_f32(max_l, max_r)
            .unwrap_or_else(|| tracker_exp_scales_scalar_f32(max_l, max_r))
    }
}

#[cfg(target_arch = "aarch64")]
impl OpMBackend for crate::math::ArmSmeBackend {
    #[inline]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::aarch64::sve2::sve2_tracker_exp_scales_f32(max_l, max_r).unwrap_or_else(|| {
            crate::math::aarch64::neon::neon_tracker_exp_scales_f32(max_l, max_r)
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl OpMBackend for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::x86::sse2::sse2_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpMBackend for crate::math::X86Sse42Backend {
    #[inline]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::x86::sse42::sse42_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpMBackend for crate::math::X86Avx2FmaBackend {
    #[inline]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::x86::sse42::sse42_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "wasm32")]
impl OpMBackend for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        #[cfg(target_feature = "simd128")]
        {
            crate::math::wasm::wasm_tracker_exp_scales_f32(max_l, max_r)
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            tracker_exp_scales_scalar_f32(max_l, max_r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;

    #[test]
    fn test_t1_dc_blocker_edges_and_scalar_exp_scales() {
        assert!(compute_t1_dc_blocker_coeffs_bits(f64::NAN).is_none());
        let def = T1DcBlocker::default();
        assert_eq!(def.pole.to_bits(), 0x3f7f_d51f);
        assert_eq!(def.gain.to_bits(), 0x3f7f_ea90);
        assert_eq!(
            def.pole.to_bits(),
            dnam_kernels::DC_BLOCKER_5HZ_48K_F32.to_bits()
        );

        let mut dcb = T1DcBlocker {
            prev_x: f32::NAN,
            prev_d: f32::INFINITY,
            ..Default::default()
        };
        assert_eq!(dcb.step(f32::NAN).to_bits(), 0);
        assert!(dcb.prev_x.is_finite() && dcb.prev_d.is_finite());
        assert_eq!(dcb.step(f32::INFINITY).to_bits(), 0);
        assert!(dcb.step(-3.0e38).is_finite());
        assert_eq!(dcb.step(3.0e38).to_bits(), 0);
        assert!(dcb.step(0.5).is_finite());

        let (sd_l, su_l, sd_r, su_r) = tracker_exp_scales_scalar_f32(0.5, 16.0);
        assert_eq!(
            (sd_l, su_l, sd_r, su_r),
            ScalarBackend::tracker_exp_scales_f32(0.5, 16.0)
        );
        assert!((sd_l * su_l - 1.0).abs() < 1e-6);
        assert!((sd_r * su_r - 1.0).abs() < 1e-6);
    }
}
