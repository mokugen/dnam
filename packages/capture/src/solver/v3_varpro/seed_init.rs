#![forbid(unsafe_code)]
//! Generic Multi-Timescale `96 kHz` Physical Seed Bank & Matryoshka Group Unlocking (`v3`).

mod unlock;

use dnam_engine::stages::v3_core::V3MatryoshkaModel;
use dnam_engine::stages::v3_core::cascade::{MAX_BLOCK_LANES, NUM_GRU_BLOCKS};

use super::model_f64::V3MatryoshkaOptF64;

pub use unlock::unlock_matryoshka_groups_after_prescreen;

/// Circuit archetype partition preserved across Phase 0a/0b Gabor homotopy screening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedArchetype {
    GeneralDrive,
    SeriesTubeSag,
    PinchBiasClipper,
    EnvResonantSvf,
}

impl SeedArchetype {
    pub(super) fn index(self) -> usize {
        match self {
            Self::GeneralDrive => 0,
            Self::SeriesTubeSag => 1,
            Self::PinchBiasClipper => 2,
            Self::EnvResonantSvf => 3,
        }
    }
}

struct SeedSpec {
    arch: SeedArchetype,
    svf: [f64; 14],
    feat: [f64; 7],
    drv: [f64; 6],
    route_pre_hp: bool,
    taps: [f64; 7],
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

const SAG_B2_PREV: [[f64; 4]; 4] = [
    [0.50, 0.22, 0.11, -0.37],
    [-0.50, -0.22, -0.11, 0.37],
    [0.68, 0.30, 0.15, -0.50],
    [0.32, 0.36, 0.18, -0.18],
];

const ASYM_OFF: [f64; 4] = [-0.45, 0.30, -0.85, 0.55];

const SEED_SPECS: [SeedSpec; 16] = [
    // GeneralDrive (4): poly 2x OS + Butterworth4 AI pre-HP overdrive + multi-band drive
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.035, 1.50, 0.0, 0.6, 0.920, 0.992, 0.72, 0.25, 0.90, 1.00, 0.10, 0.10, 0.002, 0.65,
        ],
        [0.012, 0.08, 0.006, 0.05, 0.005, 1.0e-4, 0.0],
        [1.65, 0.25, 0.25, 0.002, 0.05, 0.0],
        false,
        [0.88, 0.12, 0.04, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.66, 1.00, 0.0, 0.5, 0.950, 0.995, 0.78, 0.0, 1.848, 0.765, 0.0, 0.0, 0.002, 0.90,
        ],
        [0.046, 0.08, 0.006, 1.0e-4, 0.005, 1.0e-4, 0.0],
        [3.20, 0.0, 0.0, 0.05, 0.96, -0.18],
        true,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.66, 1.00, 0.0, 0.5, 0.950, 0.995, 0.78, 0.0, 1.848, 0.765, 0.0, 0.0, 0.002, 0.90,
        ],
        [0.036, 0.08, 0.006, 1.0e-4, 0.005, 1.0e-4, 0.0],
        [6.40, 0.0, 0.0, 0.05, 0.96, -0.18],
        true,
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::GeneralDrive,
        [
            0.045, 2.20, 0.0, 1.3, 0.860, 0.985, 0.70, 0.75, 0.68, 0.75, -0.20, 0.45, 0.002, 0.80,
        ],
        [0.060, 0.15, 0.015, 3.50, 0.012, 1.8e-4, 0.0],
        [2.95, 0.60, 0.55, 0.002, 0.08, 0.0],
        false,
        [0.45, 0.12, 0.35, 0.15, 0.08, 0.10, 0.0],
    ),
    // SeriesTubeSag (4): Butterworth4 AI -> RC diode -> commuted (LP_lp - LP_hp) -> V1B/FMV -> sagged power
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.64, 1.00, 0.0, 0.5, 0.980, 0.9990, 0.80, 0.0, 1.848, 0.765, 0.0, 0.0, 0.002, 0.90,
        ],
        [0.015, 0.05, 0.003, 0.0, 0.0024, 8.0e-6, 4.5],
        [4.2, 1.05, 0.85, 0.002, 0.96, -0.18],
        false,
        [0.0, 0.15, -0.10, 0.0, 0.85, 0.22, 0.05],
    ),
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.68, 1.00, 0.0, 0.5, 0.980, 0.9990, 0.80, 0.0, 1.848, 0.765, 0.0, 0.0, 0.002, 0.90,
        ],
        [0.015, 0.06, 0.003, 0.0, 0.0030, 8.0e-6, 7.5],
        [6.4, 1.35, 1.00, 0.002, 0.96, -0.18],
        false,
        [-0.08, 0.80, -0.45, -0.02, 0.28, 0.44, 0.18],
    ),
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.70, 1.00, 0.0, 0.5, 0.985, 0.9992, 0.82, 0.0, 1.848, 0.765, 0.0, 0.0, 0.002, 0.90,
        ],
        [0.015, 0.07, 0.003, 0.0, 0.0032, 9.0e-6, 10.5],
        [8.4, 1.55, 1.00, 0.002, 0.96, -0.18],
        false,
        [0.06, 0.72, -0.20, -0.15, 0.22, 0.38, 0.14],
    ),
    SeedSpec::row(
        SeedArchetype::SeriesTubeSag,
        [
            0.72, 1.00, 0.0, 0.5, 0.985, 0.9992, 0.82, 0.0, 1.848, 0.765, 0.0, 0.0, 0.002, 0.90,
        ],
        [0.015, 0.08, 0.004, 0.0, 0.0035, 1.1e-5, 14.0],
        [10.6, 1.70, 1.10, 0.002, 0.96, -0.18],
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
        [1.85, 0.75, 1.15, 0.002, 0.96, -0.17],
        false,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.68, 1.00, 0.0, 1.2, 0.930, 0.995, 0.78, 0.45, 1.80, 0.78, 0.0, 0.0, 0.002, 0.80,
        ],
        [0.016, 0.042, 0.0020, 650.0, 0.014, 2.5e-5, 0.0],
        [2.45, 0.90, 1.35, 0.002, 0.96, -0.17],
        false,
        [0.68, 0.32, 0.14, 0.0, 0.0, 0.24, 0.10],
    ),
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.62, 1.25, 0.0, 1.6, 0.910, 0.996, 0.80, 0.65, 1.40, 1.40, 0.0, 0.0, 0.002, 0.82,
        ],
        [0.020, 0.050, 0.0025, 1_800.0, 0.016, 2.0e-5, 0.0],
        [3.10, 1.05, 1.55, 0.002, 0.96, -0.17],
        false,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    SeedSpec::row(
        SeedArchetype::PinchBiasClipper,
        [
            0.48, 1.00, 0.0, 1.4, 0.950, 0.997, 0.84, 0.35, 1.80, 0.78, 0.0, 0.0, 0.002, 0.75,
        ],
        [0.014, 0.058, 0.0032, 6_500.0, 0.010, 1.5e-5, 0.0],
        [3.75, 1.15, 1.35, 0.002, 0.96, -0.17],
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
        [1.45, 0.0, 0.0, 0.002, 0.96, -0.18],
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
        [1.45, 0.0, 0.0, 0.002, 0.96, -0.18],
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
        [1.45, 0.0, 0.0, 0.002, 0.96, -0.18],
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
        [1.50, 0.0, 0.0, 0.002, 0.96, -0.18],
        false,
        [0.0, 0.15, 0.35, 0.0, 0.85, 0.35, 0.0],
    ),
];

fn init_sag_group0(m: &mut V3MatryoshkaOptF64, drive: f64, sag_fb: f64, asym: f64) {
    // Block 0: 2-lane exact Pade [5/4] thresholded RC diode (lanes 0..2) + two 1-pole LPs (lanes 2..4).
    let b0 = &mut m.blocks[0];
    b0.b_z[0] = -6.57;
    b0.b_r[0] = -6.4;
    b0.w_in_r[0][0] = 16.0 * drive;
    b0.w_rec_r[0][0] = -40.0;
    b0.w_rec_r[0][1] = 32.0;
    b0.w_in_n[0][0] = 0.00223 * drive;
    b0.w_rec_n[0][0] = -0.00558;
    b0.w_rec_n[0][1] = 0.00446;
    b0.b_n[0] = 0.0;
    b0.k_quad[0] = 0.0;

    b0.b_z[1] = -6.57;
    b0.b_r[1] = -6.4;
    b0.w_in_r[0][1] = 16.0 * drive;
    b0.w_rec_r[1][0] = -40.0;
    b0.w_rec_r[1][1] = 32.0;
    b0.w_in_n[0][1] = 8.0 * drive;
    b0.w_rec_n[1][0] = -20.0;
    b0.w_rec_n[1][1] = 16.0;
    b0.b_n[1] = 3.383;
    b0.k_quad[1] = 0.0;

    let kq = (0.18 * asym).clamp(-0.40, 0.40);
    for &(l, bz) in &[(2_usize, -1.044_f64), (3_usize, -4.772_f64)] {
        b0.b_z[l] = bz;
        b0.b_r[l] = bz;
        b0.w_in_n[0][l] = drive;
        b0.w_rec_n[l][0] = -2.5;
        b0.w_rec_n[l][1] = 2.0;
        b0.b_n[l] = 0.0;
        b0.k_quad[l] = kq;
    }

    // Block 1: zero-feedthrough partial-fraction (LP_lp - LP_hp) -> V1B + 3 FMV poles.
    let g_v1b = (0.44 * drive).clamp(1.8, 3.8);
    let b1_bz = [7.8, -4.39, -2.70, -1.28];
    let b1 = &mut m.blocks[1];
    for l in 0..4 {
        b1.w_prev_n[l][0..4].copy_from_slice(&[0.0, 0.0, g_v1b, -g_v1b]);
        b1.b_n[l] = (-0.12 * asym).clamp(-0.6, 0.0);
        b1.k_quad[l] = 0.0;
        b1.k_sag[l] = 0.0;
        b1.b_z[l] = b1_bz[l];
        b1.b_r[l] = b1_bz[l];
    }

    // Block 2: FMV recombination into sagged push-pull power stage.
    let g_pwr = 2.2;
    let b2_bz = [7.8, 7.8, 7.6, 3.6];
    let sag_scale = [1.05, 1.05, 1.35, 0.80];
    let b2 = &mut m.blocks[2];
    for l in 0..4 {
        for src in 0..4 {
            b2.w_prev_n[l][src] = g_pwr * SAG_B2_PREV[l][src];
        }
        b2.b_z[l] = b2_bz[l];
        b2.b_r[l] = b2_bz[l];
        b2.b_n[l] = 0.0;
        b2.k_sag[l] = (sag_fb * sag_scale[l]).clamp(0.0, 6.5);
    }
}

fn init_gated_group0(m: &mut V3MatryoshkaOptF64, drive: f64, asym: f64, signs: [f64; 4]) {
    const GATED_B0_SLOPES: [f64; 4] = [3.5, 8.2, 13.5, 22.0];
    const GATED_BN_SLOPES: [f64; 4] = [5.5, 11.0, 18.0, 29.0];
    let ds = drive / 2.2;
    let ar = asym / 1.35;
    for (l, &st) in signs.iter().enumerate() {
        let sign_even = if (l & 1) == 0 { 1.0 } else { -1.0 };
        m.blocks[0].w_in_n[2][l] = GATED_B0_SLOPES[l] * ds * st;
        m.blocks[0].w_in_n[0][l] = 0.08 * drive * st;
        m.blocks[0].b_z[l] = [6.8, 6.7, 6.6, 6.4][l];
        m.blocks[0].b_r[l] = m.blocks[0].b_z[l];
        m.blocks[0].b_n[l] = [0.28, 0.48, -0.25, 0.85][l] * ar;
        m.blocks[0].k_quad[l] = [0.06, -0.05, 0.04, 0.0][l];
        m.blocks[1].w_in_n[2][l] = GATED_BN_SLOPES[l] * ds * st;
        m.blocks[1].w_prev_n[l][l] = 0.65;
        m.blocks[1].w_prev_n[l][(l + 1) & 3] = 0.20 * st;
        m.blocks[1].b_z[l] = [6.8, 6.6, 6.4, 3.2][l];
        m.blocks[1].b_r[l] = m.blocks[1].b_z[l];
        m.blocks[1].b_n[l] = [-0.65, 0.65, -1.05, 1.15][l] * ar;
        m.blocks[2].w_in_n[2][l] = GATED_BN_SLOPES[l] * ds * st;
        m.blocks[2].w_prev_n[l][l] = 0.65;
        m.blocks[2].w_prev_n[l][(l + 3) & 3] = 0.20 * sign_even;
        m.blocks[2].b_z[l] = [6.8, 6.6, 6.4, 3.2][l];
        m.blocks[2].b_r[l] = m.blocks[2].b_z[l];
        m.blocks[2].b_n[l] = [-0.65, 0.65, -1.05, 1.15][l] * ar;
    }
}

fn init_svf_group0(m: &mut V3MatryoshkaOptF64, drive: f64, asym: f64, signs: [f64; 4]) {
    for (l, &st) in signs.iter().enumerate() {
        m.blocks[0].w_in_n[0][l] = [0.45, 0.95, 1.75, 2.85][l] * drive * st;
        m.blocks[0].b_z[l] = [7.4, 7.2, 7.0, 6.8][l];
        m.blocks[0].b_r[l] = m.blocks[0].b_z[l];
        m.blocks[0].b_n[l] = 0.20 * asym * ASYM_OFF[l];
        m.blocks[1].w_in_n[0][l] = [0.68, 1.32, 2.25, 3.60][l] * drive * st;
        m.blocks[1].w_prev_n[l][l] = 0.28;
        m.blocks[1].w_prev_n[l][(l + 1) & 3] = 0.10 * st;
        m.blocks[1].b_z[l] = [7.4, 7.2, 7.0, 6.8][l];
        m.blocks[1].b_r[l] = m.blocks[1].b_z[l];
        m.blocks[1].b_n[l] = 0.12 * asym * st;
        m.blocks[2].w_in_n[0][l] = [0.30, 1.15, 2.05, 4.20][l] * drive * st;
        m.blocks[2].w_prev_n[l][l] = 0.28;
        m.blocks[2].w_prev_n[l][(l + 3) & 3] = -0.08;
        m.blocks[2].b_z[l] = [7.4, 7.2, 7.0, 6.8][l];
        m.blocks[2].b_r[l] = m.blocks[2].b_z[l];
        m.blocks[2].b_n[l] = 0.0;
    }
}

fn init_pre_hp_group0(m: &mut V3MatryoshkaOptF64, drive: f64, asym: f64, signs: [f64; 4]) {
    for (l, &st) in signs.iter().enumerate() {
        m.blocks[0].w_in_n[0][l] = 0.0;
        m.blocks[0].w_in_n[1][l] = [0.45, 0.95, 1.85, 3.30][l] * drive * st;
        m.blocks[0].w_in_n[2][l] = 0.0;
        m.blocks[0].b_z[l] = [7.4, 7.2, 7.0, 6.8][l];
        m.blocks[0].b_r[l] = m.blocks[0].b_z[l];
        m.blocks[0].b_n[l] = asym * ASYM_OFF[l];
        m.blocks[0].k_quad[l] = 0.0;
        if l < 2 {
            m.blocks[1].w_in_n[1][l] = [0.68, 1.38][l] * drive * st;
            m.blocks[1].b_z[l] = 7.2;
            m.blocks[1].b_r[l] = 7.2;
            m.blocks[1].b_n[l] = 0.12 * asym * st;
        } else {
            m.blocks[1].w_prev_n[l][0..4].copy_from_slice(&[
                0.06 * st,
                -0.08 * st,
                0.08 * st,
                -0.05 * st,
            ]);
            m.blocks[1].b_z[l] = if l == 2 { -0.45 } else { -1.75 };
            m.blocks[1].b_r[l] = m.blocks[1].b_z[l];
            m.blocks[1].b_n[l] = 0.0;
        }
        m.blocks[2].w_in_n[0][l] = 0.0;
        m.blocks[2].w_in_n[1][l] = [0.28, 1.15, 2.45, 4.40][l] * drive * st;
        m.blocks[2].w_in_n[2][l] = 0.0;
        m.blocks[2].w_prev_n[l][2] = 0.25 * st;
        m.blocks[2].w_prev_n[l][3] = -0.25 * st;
        m.blocks[2].b_z[l] = [7.4, 7.2, 7.0, 6.8][l];
        m.blocks[2].b_r[l] = m.blocks[2].b_z[l];
        m.blocks[2].b_n[l] = 0.0;
        m.blocks[2].k_sag[l] = 0.0;
    }
}

fn init_general_group0(
    m: &mut V3MatryoshkaOptF64,
    drive: f64,
    sag_fb: f64,
    asym: f64,
    signs: [f64; 4],
) {
    const LANE_GAINS: [f64; 4] = [0.45, 0.95, 1.85, 3.30];
    for (l, &st) in signs.iter().enumerate() {
        let sign_even = if (l & 1) == 0 { 1.0 } else { -1.0 };
        let lg = LANE_GAINS[l] * drive;
        m.blocks[0].w_in_n[0][l] = 0.75 * lg * st;
        m.blocks[0].w_in_n[1][l] = 0.32 * lg * sign_even;
        m.blocks[0].w_in_n[2][l] = -0.18 * lg * st;
        m.blocks[0].b_z[l] = [3.6, 1.0, -1.8, -5.0][l];
        m.blocks[0].b_r[l] = m.blocks[0].b_z[l];
        m.blocks[0].b_n[l] = asym * ASYM_OFF[l];
        m.blocks[0].k_quad[l] = [0.04, -0.03, 0.02, 0.0][l] * asym;
        if l == 3 {
            m.blocks[0].w_rec_n[3][3] = 0.15;
        }
        m.blocks[1].w_in_n[0][l] = 0.35 * drive * st;
        m.blocks[1].w_in_n[2][l] = 0.18 * sag_fb;
        m.blocks[1].w_prev_n[l][l] = 0.55;
        m.blocks[1].w_prev_n[l][(l + 1) & 3] = 0.22 * st;
        m.blocks[1].b_z[l] = [3.5, 0.8, -2.0, -5.2][l];
        m.blocks[1].b_r[l] = m.blocks[1].b_z[l];
        m.blocks[1].b_n[l] = 0.12 * asym * st;
        m.blocks[2].w_in_n[0][l] = 0.20 * drive * st;
        m.blocks[2].w_in_n[2][l] = 0.22 * sag_fb;
        m.blocks[2].w_prev_n[l][l] = 0.60;
        m.blocks[2].w_prev_n[l][(l + 3) & 3] = -0.18;
        m.blocks[2].b_z[l] = [3.5, 0.8, -2.0, -5.2][l];
        m.blocks[2].b_r[l] = m.blocks[2].b_z[l];
        m.blocks[2].k_sag[l] = (0.20 * sag_fb).clamp(0.0, 4.0);
    }
}

fn init_group0_for_seed(m: &mut V3MatryoshkaOptF64, s: &SeedSpec) {
    let [drive, sag_fb, asym, _, os_dec_aa_mix, _] = s.drv;
    let signs = [1.0, -1.0, 1.0, -1.0];
    let is_gated = s.arch == SeedArchetype::PinchBiasClipper;
    let is_svf = s.arch == SeedArchetype::EnvResonantSvf;
    let is_sag = s.arch == SeedArchetype::SeriesTubeSag;

    if is_sag {
        init_sag_group0(m, drive, sag_fb, asym);
    } else if is_gated {
        init_gated_group0(m, drive, asym, signs);
    } else if is_svf {
        init_svf_group0(m, drive, asym, signs);
    } else if s.route_pre_hp {
        init_pre_hp_group0(m, drive, asym, signs);
    } else {
        init_general_group0(m, drive, sag_fb, asym, signs);
    }

    for b in 0..NUM_GRU_BLOCKS {
        m.blocks[b].enforce_lower_triangular();
    }
    if is_sag {
        m.kautz_poles[0] = (0.993, 0.007);
        m.kautz_poles[1] = (0.984, 0.030);
        m.kautz_poles[2] = (0.968, 0.104);
        m.kautz_poles[3] = (0.950, 0.176);
    } else {
        m.kautz_poles[0] = (0.995, 0.0);
        m.kautz_poles[1] = (0.76, 0.0);
        m.kautz_poles[2] = (0.93, 0.045);
        m.kautz_poles[3] = (0.86, 0.24);
    }
    m.readouts[0].svf_direct_taps_l = [0.0; 7];
    m.readouts[0].svf_direct_taps_r = [0.0; 7];
    m.readouts[0].direct_s2_u_gain = [0.0; 2];
    m.readouts[0].direct_s2_hp_gain = [0.0; 2];
    if is_gated || is_sag {
        m.readouts[0].pre_cab_u_gain = [0.0, 0.0];
        m.readouts[0].pre_cab_hp_gain = [0.0, 0.0];
    } else if is_svf {
        m.readouts[0].pre_cab_u_gain = [0.45, 0.45];
        m.readouts[0].pre_cab_hp_gain = [0.0, 0.0];
    } else if s.route_pre_hp {
        m.readouts[0].pre_cab_u_gain = [0.0, 0.0];
        m.readouts[0].pre_cab_hp_gain = [0.18, 0.18];
    } else {
        m.readouts[0].pre_cab_u_gain = [0.25, 0.25];
        m.readouts[0].pre_cab_hp_gain = [0.10, 0.10];
    }
    let fir0 = if os_dec_aa_mix >= 0.8 {
        [0.06, 0.44, 0.42, 0.08]
    } else {
        [0.18, 0.46, 0.30, 0.06]
    };
    m.readouts[0].fir_l[..4].copy_from_slice(&fir0);
    m.readouts[0].fir_r[..4].copy_from_slice(&fir0);
    if is_sag {
        m.readouts[0].pre_cab_w[0][0..4].fill(0.0);
        m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES..MAX_BLOCK_LANES + 4]
            .copy_from_slice(&[0.02, -0.02, 0.02, -0.02]);
        m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES..2 * MAX_BLOCK_LANES + 4]
            .copy_from_slice(&[0.35, -0.30, 0.22, 0.12]);
    } else {
        let (w0, w1, w2) = if is_gated {
            (0.12, 0.18, 0.28)
        } else if is_svf {
            (0.18, 0.12, 0.08)
        } else if s.route_pre_hp {
            (0.16, 0.12, 0.14)
        } else {
            (0.14, 0.18, 0.22)
        };
        for (l, &st) in signs.iter().enumerate() {
            m.readouts[0].pre_cab_w[0][l] = w0 * st;
            m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES + l] = w1 * st;
            m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES + l] = w2 * st;
        }
    }
    m.mirror_mono();
}

fn build_model_from_spec(s: &SeedSpec, mono_out: bool) -> (SeedArchetype, V3MatryoshkaOptF64) {
    let mut m =
        V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, mono_out));
    m.max_active_groups = 1;
    m.active_tier_weights = [1.0, 0.0, 0.0, 0.0];
    m.os_up_quad = -0.125;
    m.os_dec_c_prev1 = 0.15;
    m.os_dec_c0 = 0.50;
    m.os_up_pre_emph = s.drv[5];
    m.os_up_aa_mix = s.drv[3];
    m.os_dec_aa_mix = s.drv[4];
    m.svf_base_g = s.svf[0];
    m.svf_sec2_g_ratio = s.svf[1];
    m.svf_env_oct_sens = s.svf[2];
    m.svf_env_comp = s.svf[3];
    m.svf_atk_alpha = s.svf[4];
    m.svf_rel_alpha = s.svf[5];
    m.svf_det_lp_pole = s.svf[6];
    m.svf_det_hp_mix = s.svf[7];
    m.svf_damp_k1 = s.svf[8];
    m.svf_damp_k2 = s.svf[9];
    m.svf_sec2_bp_feed = s.svf[10];
    m.svf_sec2_hp_feed = s.svf[11];
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = s.svf[12];
    m.svf_g_max = s.svf[13];
    m.svf_stage2_feed_taps = s.taps;
    m.feat_in_hp_beta = s.feat[0];
    m.feat_fast_atk_beta = s.feat[1];
    m.feat_fast_rel_beta = s.feat[2];
    m.feat_pinch_k = s.feat[3];
    m.feat_slow_atk_beta = s.feat[4];
    m.feat_slow_rel_beta = s.feat[5];
    m.feat_slow_det_comp = s.feat[6];
    init_group0_for_seed(&mut m, s);
    (s.arch, m)
}

/// Builds `16` `96 kHz`-calibrated tagged seed models (`4` per [`SeedArchetype`]).
#[must_use]
pub fn build_matryoshka_seed_bank_tagged(
    mono_out: bool,
) -> Vec<(SeedArchetype, V3MatryoshkaOptF64)> {
    SEED_SPECS
        .iter()
        .map(|s| build_model_from_spec(s, mono_out))
        .collect()
}

/// Returns the flat `16`-model `96 kHz` black-box seed bank without archetype tags.
#[must_use]
pub fn build_matryoshka_seed_bank(mono_out: bool) -> Vec<V3MatryoshkaOptF64> {
    build_matryoshka_seed_bank_tagged(mono_out)
        .into_iter()
        .map(|(_, m)| m)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_bank_tagged_and_flat_archetype_counts() {
        let tagged = build_matryoshka_seed_bank_tagged(true);
        assert_eq!(tagged.len(), 16);
        let mut counts = [0_usize; 4];
        for (arch, m) in &tagged {
            counts[arch.index()] += 1;
            assert_eq!(m.max_active_groups, 1);
            assert!(m.mono_out);
            assert_eq!(m.svf_stereo_link, 0.0);
        }
        assert_eq!(counts, [4, 4, 4, 4]);
        assert_eq!(build_matryoshka_seed_bank(false).len(), 16);
    }
}
