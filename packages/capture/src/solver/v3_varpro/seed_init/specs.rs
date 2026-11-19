#![forbid(unsafe_code)]
//! Generic `96 kHz` `16`-Seed Table Across `4` Physical Circuit Archetypes.

/// Circuit archetype partition preserved across Phase 0a/0b Gabor homotopy screening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedArchetype {
    GeneralDrive,
    SeriesTubeSag,
    PinchBiasClipper,
    EnvResonantSvf,
}

impl SeedArchetype {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::GeneralDrive => 0,
            Self::SeriesTubeSag => 1,
            Self::PinchBiasClipper => 2,
            Self::EnvResonantSvf => 3,
        }
    }
}

pub(super) struct SeedSpec {
    pub(super) arch: SeedArchetype,
    pub(super) svf: [f64; 14],
    pub(super) feat: [f64; 7],
    pub(super) drv: [f64; 6],
    pub(super) route_pre_hp: bool,
    pub(super) taps: [f64; 7],
}

impl SeedSpec {
    const fn row(
        arch: SeedArchetype,
        svf: [f64; 14],
        feat: [f64; 7],
        drv: [f64; 6],
        route_pre_hp: bool,
        taps: [f64; 7],
    ) -> Self {
        Self {
            arch,
            svf,
            feat,
            drv,
            route_pre_hp,
            taps,
        }
    }
}

pub(super) const SAG_B2_PREV: [[f64; 4]; 4] = [
    [0.50, 0.22, 0.11, -0.37],
    [-0.50, -0.22, -0.11, 0.37],
    [0.68, 0.30, 0.15, -0.50],
    [0.32, 0.36, 0.18, -0.18],
];

pub(super) const ASYM_OFF: [f64; 4] = [-0.45, 0.30, -0.85, 0.55];

/// 4th-order Butterworth damping constants `2 * sin((2k + 1) * pi / 8)` derived at compile time.
pub(crate) const BUTTER4_TWO_R_0: f64 =
    2.0 * dnam_engine::math::const_sin_cos_f64(std::f64::consts::PI / 8.0).0;
pub(crate) const BUTTER4_TWO_R_1: f64 =
    2.0 * dnam_engine::math::const_sin_cos_f64(3.0 * std::f64::consts::PI / 8.0).0;
const _: () = assert!(
    BUTTER4_TWO_R_0 > 0.765
        && BUTTER4_TWO_R_0 < 0.766
        && BUTTER4_TWO_R_1 > 1.847
        && BUTTER4_TWO_R_1 < 1.849
);

const fn const_sqrt_f64(x: f64) -> f64 {
    let mut g = if x > 1.0 { x } else { 1.0 };
    let mut i = 0_u32;
    while i < 24 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

/// Solves exact unity-gain pre-emphasis `p` at `omega = 3*pi/4` (`0.375 * fs`) for
/// `dnam_engine::stages::stage2_gru::frontend::ZeroLagOversampler2x` (`up_quad = -1/8`).
const fn derive_os2x_quad_passband_pre_emph() -> f64 {
    let cos_3pi8 = 0.5 * const_sqrt_f64(2.0 - std::f64::consts::SQRT_2);
    let sin_3pi8 = 0.5 * const_sqrt_f64(2.0 + std::f64::consts::SQRT_2);
    // e^{j*3pi/8} * (3/8 + 3/4 * e^{-j*3pi/4} - 1/8 * e^{-j*3pi/2})
    //   = 3/8 * e^{j*3pi/8} + 3/4 * e^{-j*3pi/8} + 1/8 * e^{-j*pi/8}
    let re_x0 = (9.0 / 8.0) * cos_3pi8 + (1.0 / 8.0) * sin_3pi8;
    let im_x0 = -(3.0 / 8.0) * sin_3pi8 - (1.0 / 8.0) * cos_3pi8;
    let h_re = 0.5 * (1.0 + re_x0);
    let h_im = 0.5 * im_x0;
    let h_os_sq = h_re * h_re + h_im * h_im;
    // |1 - p + p * e^{-j*3pi/4}|^2 = 1 + (2 + sqrt(2)) * (p^2 - p) = 1 / h_os_sq
    let rhs = (1.0 / h_os_sq - 1.0) / (2.0 + std::f64::consts::SQRT_2);
    0.5 * (1.0 - const_sqrt_f64(1.0 + 4.0 * rhs))
}

/// Compile-time derived HF passband flatness pre-emphasis for `ZeroLagOversampler2x` (`-0.14602`).
pub(super) const OS2X_QUAD_PASSBAND_PRE_EMPH: f64 = derive_os2x_quad_passband_pre_emph();
const _: () =
    assert!(OS2X_QUAD_PASSBAND_PRE_EMPH > -0.165 && OS2X_QUAD_PASSBAND_PRE_EMPH < -0.140);

pub(super) const SEED_SPECS: [SeedSpec; 16] = [
    // GeneralDrive (4): poly 2x OS + Butterworth4 AI pre-HP overdrive + multi-band drive
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.035, 1.50, 0.0, 0.6, 0.920, 0.992, 0.72, 0.25, 0.90, 1.00, 0.10, 0.10, 0.002, 0.65,
        ],
        [0.012, 0.08, 0.006, 0.05, 0.005, 1.0e-4, 0.0],
        [1.65, 0.25, 0.25, 0.002, 0.05, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.88, 0.12, 0.04, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.66,
            1.00,
            0.0,
            0.5,
            0.950,
            0.995,
            0.78,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.046, 0.08, 0.006, 1.0e-4, 0.005, 1.0e-4, 0.0],
        [3.20, 0.0, 0.0, 0.05, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        true,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.66,
            1.00,
            0.0,
            0.5,
            0.950,
            0.995,
            0.78,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.036, 0.08, 0.006, 1.0e-4, 0.005, 1.0e-4, 0.0],
        [6.40, 0.0, 0.0, 0.05, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        true,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.045, 2.20, 0.0, 1.3, 0.860, 0.985, 0.70, 0.75, 0.68, 0.75, -0.20, 0.45, 0.002, 0.80,
        ],
        [0.060, 0.15, 0.015, 3.50, 0.012, 1.8e-4, 0.0],
        [2.95, 0.60, 0.55, 0.002, 0.08, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.45, 0.12, 0.35, 0.15, 0.08, 0.10, 0.0],
    ),
    // SeriesTubeSag (4): Butterworth4 AI -> RC diode -> commuted (LP_lp - LP_hp) -> V1B/FMV -> sagged power
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.64,
            1.00,
            0.0,
            0.5,
            0.980,
            0.9990,
            0.80,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.015, 0.05, 0.003, 0.0, 0.0024, 8.0e-6, 4.5],
        [4.2, 1.05, 0.85, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.15, -0.10, 0.0, 0.85, 0.22, 0.05],
    ),
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.68,
            1.00,
            0.0,
            0.5,
            0.980,
            0.9990,
            0.80,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.015, 0.06, 0.003, 0.0, 0.0030, 8.0e-6, 7.5],
        [6.4, 1.35, 1.00, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [-0.08, 0.80, -0.45, -0.02, 0.28, 0.44, 0.18],
    ),
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.70,
            1.00,
            0.0,
            0.5,
            0.985,
            0.9992,
            0.82,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.015, 0.07, 0.003, 0.0, 0.0032, 9.0e-6, 10.5],
        [8.4, 1.55, 1.00, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.06, 0.72, -0.20, -0.15, 0.22, 0.38, 0.14],
    ),
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.72,
            1.00,
            0.0,
            0.5,
            0.985,
            0.9992,
            0.82,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.015, 0.08, 0.004, 0.0, 0.0035, 1.1e-5, 14.0],
        [10.6, 1.70, 1.10, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.0, -0.08, 0.0, 1.00, 0.16, 0.0],
    ),
    // PinchBiasClipper (4): memoryless wide-slope pinch-bias clipper + causal HF group-delay SVF
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.55, 1.25, 0.0, 0.7, 0.940, 0.994, 0.76, 0.25, 1.40, 1.40, 0.0, 0.0, 0.002, 0.75,
        ],
        [0.012, 0.035, 0.0015, 280.0, 0.012, 3.0e-5, 0.0],
        [1.85, 0.75, 1.15, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.79,
            1.00,
            0.0,
            1.2,
            0.930,
            0.995,
            0.78,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.95,
        ],
        [0.016, 0.040, 0.0016, 635.0, 0.014, 2.5e-5, 0.0],
        [2.60, 0.90, 1.35, 0.01, 0.01, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.70,
            1.00,
            0.0,
            1.2,
            0.930,
            0.995,
            0.78,
            0.0,
            BUTTER4_TWO_R_1,
            BUTTER4_TWO_R_0,
            0.0,
            0.0,
            0.002,
            0.90,
        ],
        [0.016, 0.042, 0.0020, 680.0, 0.014, 2.5e-5, 0.0],
        [2.55, 0.92, 1.35, 0.02, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.48, 1.00, 0.0, 1.4, 0.950, 0.997, 0.84, 0.35, 1.80, 0.78, 0.0, 0.0, 0.002, 0.75,
        ],
        [0.014, 0.058, 0.0032, 6_500.0, 0.010, 1.5e-5, 0.0],
        [3.75, 1.15, 1.35, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.68, 0.32, 0.14, 0.0, 0.0, 0.24, 0.10],
    ),
    // EnvResonantSvf (4): LP / BP / HP / 2-sec calibrated to 96 kHz octave ceiling
    SeedSpec::row(
        SeedArchetype::EnvResonantSvf,
        [
            0.0102, 1.00, 8.40, 0.001, 0.984, 0.9994, 0.25, -0.42, 0.18, 0.70, 0.0, 0.0, 0.0040,
            0.096,
        ],
        [0.012, 0.08, 0.006, 0.20, 0.008, 3.0e-5, 0.0],
        [1.45, 0.0, 0.0, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 1.15, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::EnvResonantSvf,
        [
            0.0105, 1.00, 8.45, 0.001, 0.984, 0.9994, 0.25, -0.42, 0.148, 0.70, 0.0, 0.0, 0.0040,
            0.096,
        ],
        [0.014, 0.09, 0.007, 0.15, 0.008, 3.0e-5, 0.0],
        [1.45, 0.0, 0.0, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.0, 2.20, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::EnvResonantSvf,
        [
            0.0108, 1.00, 8.00, 0.001, 0.982, 0.9993, 0.25, -0.42, 0.19, 0.65, 0.0, 0.0, 0.0040,
            0.135,
        ],
        [0.012, 0.08, 0.006, 0.25, 0.008, 3.0e-5, 0.0],
        [1.45, 0.0, 0.0, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.0, 0.25, 1.05, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::EnvResonantSvf,
        [
            0.0095, 1.08, 8.60, 0.001, 0.985, 0.9994, 0.28, -0.36, 0.20, 0.30, 0.20, 0.0, 0.0040,
            0.115,
        ],
        [0.014, 0.09, 0.007, 0.30, 0.008, 3.0e-5, 0.0],
        [1.50, 0.0, 0.0, 0.002, 0.96, OS2X_QUAD_PASSBAND_PRE_EMPH],
        false,
        [0.0, 0.15, 0.35, 0.0, 0.85, 0.35, 0.0],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_seed_archetype_indices_and_spec_table() {
        assert_eq!(SeedArchetype::GeneralDrive.index(), 0);
        assert_eq!(SeedArchetype::SeriesTubeSag.index(), 1);
        assert_eq!(SeedArchetype::PinchBiasClipper.index(), 2);
        assert_eq!(SeedArchetype::EnvResonantSvf.index(), 3);
        assert_eq!(SEED_SPECS.len(), 16);

        let w = 0.75 * PI;
        let p = OS2X_QUAD_PASSBAND_PRE_EMPH;
        let h_pe_re = (1.0 - p) + p * w.cos();
        let h_pe_im = -p * w.sin();
        let h_pe_sq = h_pe_re * h_pe_re + h_pe_im * h_pe_im;

        let half_w = 0.5 * w;
        let x0_re = 0.375 + 0.75 * w.cos() - 0.125 * (2.0 * w).cos();
        let x0_im = -0.75 * w.sin() + 0.125 * (2.0 * w).sin();
        let s_re = 0.5 * (1.0 + x0_re * half_w.cos() - x0_im * half_w.sin());
        let s_im = 0.5 * (x0_re * half_w.sin() + x0_im * half_w.cos());
        let h_os_sq = s_re * s_re + s_im * s_im;
        assert!((h_pe_sq * h_os_sq - 1.0).abs() < 1.0e-12);
    }
}
