#![forbid(unsafe_code)]
//! Multi-start physical initializers and closed-form `VarPro` prescreening for Stage I SVF + Stage II.

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::V3ColdTierModel;
use dnam_engine::stages::v3_core::cascade::{NUM_CASCADE_BLOCKS, tier_cascade_lanes};

use super::adjoint::{AdjointWorkspace, evaluate_loss_f64};
use super::basis::{select_omp_kautz_poles, solve_bezout_varpro_cholesky};
use super::model_f64::{V3OptModelF64, V3TrainDataset, V3TrainStem};

#[allow(clippy::too_many_arguments)]
fn configure_cascade_lanes(
    m: &mut V3OptModelF64,
    in_gains: [f64; 4],
    lp_poles: [f64; 4],
    lp_mix: f64,
    hp_mix: f64,
    static_bias: [f64; 4],
    env_bias_sens: [f64; 4],
    inter_scale: f64,
) {
    let c_cnt = tier_cascade_lanes(m.tier).min(4);
    for b in 0..NUM_CASCADE_BLOCKS {
        let block_scale = match b {
            0 => 1.0,
            1 => 0.55,
            _ => 0.35,
        };
        let bias_sign = if b.is_multiple_of(2) { 1.0 } else { -0.8 };
        for l in 0..c_cnt {
            m.blocks[b].in_u_gain[l] = in_gains[l] * block_scale;
            m.blocks[b].lp_pole[l] = lp_poles[l];
            m.blocks[b].lp_mix[l] = lp_mix;
            m.blocks[b].hp_mix[l] = hp_mix;
            m.blocks[b].static_bias[l] = static_bias[l] * bias_sign;
            m.blocks[b].env_bias_sens[l] = env_bias_sens[l] * bias_sign;
            m.blocks[b].skip_u_gain[l] = 0.20;
            m.blocks[b].prev_skip_gain[l] = if b > 0 { 0.18 } else { 0.0 };
            if b > 0 {
                for s in 0..c_cnt {
                    m.blocks[b].inter_mat[l][s] = if l == s {
                        inter_scale
                    } else {
                        -0.12 * inter_scale * if (l + s) % 2 == 0 { 1.0 } else { -1.0 }
                    };
                }
            }
        }
    }
    m.pre_cab_u_gain = [0.25, 0.25];
    for l in 0..c_cnt {
        let sign = if l % 2 == 0 { 1.0 } else { -0.85 };
        m.pre_cab_w[0][l] = 0.22 * sign;
        m.pre_cab_w[1][l] = 0.22 * sign;
    }
}

fn base_seed(mono_out: bool) -> V3OptModelF64 {
    let mut cold = V3ColdTierModel::identity_for_tier(Tier::Lite, false, mono_out);
    cold.svf.g_min = 2.0e-4;
    cold.svf.g_max = 1.40;
    for b in 0..NUM_CASCADE_BLOCKS {
        cold.blocks[b].atk_alpha = 0.65;
        cold.blocks[b].rel_alpha = 0.96;
    }
    V3OptModelF64::from_cold_model(&cold)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SeedKind {
    General,
    SeriesTube,
    EnvSvf,
}

fn configure_series_tube_amp(
    m: &mut V3OptModelF64,
    gain_scale: f64,
    sag_sens: f64,
    tone_balance: [f64; 4],
) {
    // Block 0: V1A mixed-sign asymmetric / foldback stages + grid conduction RC attack/release
    m.blocks[0].atk_alpha = 0.96;
    m.blocks[0].rel_alpha = 0.998;
    let g0 = [6.2, -4.4, 3.1, -2.0];
    let p0 = [0.25, 0.45, 0.68, 0.88];
    let b0 = [-0.45, -1.65, 0.35, 1.15];
    let e0 = [-0.40, 0.40, -0.25, 0.25];
    for l in 0..4 {
        m.blocks[0].in_u_gain[l] = g0[l] * gain_scale;
        m.blocks[0].lp_pole[l] = p0[l];
        m.blocks[0].lp_mix[l] = 0.85;
        m.blocks[0].hp_mix[l] = 0.35;
        m.blocks[0].static_bias[l] = b0[l];
        m.blocks[0].env_bias_sens[l] = e0[l] * gain_scale.min(1.5);
        m.blocks[0].skip_u_gain[l] = 0.08;
    }

    // Block 1: series-driven (in_u_gain ~ 0), interstage AC-coupling HP (0.982..0.988) + HF rolloff
    m.blocks[1].atk_alpha = 0.95;
    m.blocks[1].rel_alpha = 0.997;
    let p1 = [0.982, 0.988, 0.55, 0.72];
    let lpm1 = [0.0, 0.0, 1.0, 0.65];
    let hpm1 = [1.0, 1.0, 0.0, 0.55];
    let b1 = [-0.28, 0.24, -0.18, 0.14];
    let e1 = [0.15, -0.15, 0.10, -0.10];
    let w1 = [1.15, -0.85, 0.70, -0.55];
    for l in 0..4 {
        m.blocks[1].in_u_gain[l] = if l % 2 == 0 { 0.02 } else { -0.02 };
        m.blocks[1].lp_pole[l] = p1[l];
        m.blocks[1].lp_mix[l] = lpm1[l];
        m.blocks[1].hp_mix[l] = hpm1[l];
        m.blocks[1].static_bias[l] = b1[l];
        m.blocks[1].env_bias_sens[l] = e1[l];
        m.blocks[1].skip_u_gain[l] = 0.12;
        m.blocks[1].prev_skip_gain[l] = 0.05;
        for s in 0..4 {
            let sign = if (l + s) % 2 == 0 { 1.0 } else { -1.0 };
            m.blocks[1].inter_mat[l][s] = 0.55 * w1[s] * sign;
        }
    }

    // Block 2: series-driven parallel 3-band FMV tone stack poles + push-pull paired (+s, -s) sag
    m.blocks[2].atk_alpha = 0.992;
    m.blocks[2].rel_alpha = 0.9995;
    let p2 = [0.975, 0.88, 0.62, 0.48];
    let lpm2 = [1.0, 0.42, 0.35, 0.0];
    let hpm2 = [0.0, 0.65, 0.75, 1.0];
    let b2 = [0.22, -0.22, 0.14, -0.14];
    let e2 = [sag_sens, -sag_sens, 0.65 * sag_sens, -0.65 * sag_sens];
    for l in 0..4 {
        m.blocks[2].in_u_gain[l] = 0.0;
        m.blocks[2].lp_pole[l] = p2[l];
        m.blocks[2].lp_mix[l] = lpm2[l];
        m.blocks[2].hp_mix[l] = hpm2[l];
        m.blocks[2].static_bias[l] = b2[l];
        m.blocks[2].env_bias_sens[l] = e2[l];
        m.blocks[2].skip_u_gain[l] = 0.15 * tone_balance[l];
        m.blocks[2].prev_skip_gain[l] = 0.12;
        for s in 0..4 {
            let diag = if l == s { 0.85 } else { 0.25 };
            let sign = if (l + s) % 2 == 0 { 1.0 } else { -1.0 };
            m.blocks[2].inter_mat[l][s] = tone_balance[l] * diag * sign;
        }
    }
    m.pre_cab_u_gain = [0.08, 0.08];
    for l in 0..4 {
        let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
        m.pre_cab_w[0][l] = 0.35 * tone_balance[l] * sign;
        m.pre_cab_w[1][l] = 0.35 * tone_balance[l] * sign;
    }
}

fn configure_env_svf_saturator(m: &mut V3OptModelF64, drv_scale: f64) {
    // Broadband zero-phase Stage II Block 0 (lp_mix = 1.0, hp_mix = 1.0 => up == v)
    let g0 = [0.95, -0.68, 0.45, -0.28];
    let p0 = [0.35, 0.55, 0.72, 0.86];
    for l in 0..4 {
        m.blocks[0].in_u_gain[l] = g0[l] * drv_scale;
        m.blocks[0].lp_pole[l] = p0[l];
        m.blocks[0].lp_mix[l] = 1.0;
        m.blocks[0].hp_mix[l] = 1.0;
        m.blocks[0].static_bias[l] = 0.0;
        m.blocks[0].env_bias_sens[l] = 0.0;
        m.blocks[0].skip_u_gain[l] = 0.12;
    }
    for b in 1..NUM_CASCADE_BLOCKS {
        for l in 0..4 {
            m.blocks[b].in_u_gain[l] = 0.05 * g0[l];
            m.blocks[b].lp_pole[l] = p0[l];
            m.blocks[b].lp_mix[l] = 1.0;
            m.blocks[b].hp_mix[l] = 1.0;
            m.blocks[b].static_bias[l] = 0.0;
            m.blocks[b].env_bias_sens[l] = 0.0;
            m.blocks[b].skip_u_gain[l] = 0.10;
            m.blocks[b].prev_skip_gain[l] = if b == 1 { 0.40 } else { 0.30 };
            m.blocks[b].inter_mat[l][l] = 0.22;
        }
    }
    m.pre_cab_u_gain = [0.38, 0.38];
    for l in 0..4 {
        let s = if l % 2 == 0 { 0.36 } else { -0.24 };
        m.pre_cab_w[0][l] = s;
        m.pre_cab_w[1][l] = s;
    }
}

fn build_tagged_templates(mono_out: bool) -> Vec<(SeedKind, V3OptModelF64)> {
    let mut seeds = Vec::with_capacity(18);

    // 1. Near-LTI clean preamp
    let mut s1 = base_seed(mono_out);
    configure_cascade_lanes(
        &mut s1,
        [0.45, 0.35, 0.55, 0.25],
        [0.45, 0.65, 0.80, 0.90],
        0.80,
        0.35,
        [0.0; 4],
        [0.0; 4],
        0.25,
    );
    s1.pre_cab_u_gain = [0.85, 0.85];
    seeds.push((SeedKind::General, s1));

    // 2. Mild symmetric OD
    let mut s2 = base_seed(mono_out);
    configure_cascade_lanes(
        &mut s2,
        [1.8, 1.4, 2.2, 1.2],
        [0.55, 0.72, 0.86, 0.93],
        0.75,
        0.55,
        [0.0; 4],
        [-0.08, 0.06, -0.05, 0.04],
        0.42,
    );
    seeds.push((SeedKind::General, s2));

    // 3. Hot symmetric OD + pre-emphasis
    let mut s3 = base_seed(mono_out);
    s3.svf_base_g = 0.055;
    s3.svf_stage2_feed_taps = [0.65, -0.25, 0.38, 0.22, 0.0, 0.0, 0.0];
    configure_cascade_lanes(
        &mut s3,
        [3.6, 2.8, 4.4, 2.2],
        [0.50, 0.74, 0.88, 0.94],
        0.65,
        0.70,
        [-0.08, 0.06, -0.05, 0.04],
        [-0.12, 0.08, -0.10, 0.06],
        0.52,
    );
    seeds.push((SeedKind::General, s3));

    // 4. Bright edge-of-breakup preamp
    let mut s4 = base_seed(mono_out);
    s4.svf_base_g = 0.09;
    s4.svf_stage2_feed_taps = [0.60, -0.20, 0.30, 0.50, 0.10, 0.20, 0.20];
    configure_cascade_lanes(
        &mut s4,
        [1.4, 1.1, 1.7, 0.9],
        [0.40, 0.58, 0.75, 0.89],
        0.72,
        0.55,
        [-0.20, 0.15, -0.15, 0.10],
        [-0.10, 0.08, -0.06, 0.05],
        0.38,
    );
    seeds.push((SeedKind::General, s4));

    // 5. Series Multi-Stage Tube Amp: Mid-Gain Tweed Crunch + Cab
    let mut s5 = base_seed(mono_out);
    s5.svf_base_g = 0.042;
    s5.svf_sec2_g_ratio = 2.0;
    s5.svf_damp_k1 = 0.95;
    s5.svf_damp_k2 = 0.85;
    s5.svf_stage2_feed_taps = [0.55, 0.25, -0.12, 0.20, 0.10, -0.05, 0.15];
    configure_series_tube_amp(&mut s5, 0.75, 0.38, [0.95, 0.85, 0.95, 0.90]);
    seeds.push((SeedKind::SeriesTube, s5));

    // 6. Series Multi-Stage Tube Amp: FMV Scooped High-Gain + Push-Pull Sag + 4x12 Cab
    let mut s6 = base_seed(mono_out);
    s6.svf_base_g = 0.038;
    s6.svf_sec2_g_ratio = 2.4;
    s6.svf_damp_k1 = 0.85;
    s6.svf_damp_k2 = 0.72;
    s6.svf_stage2_feed_taps = [0.50, 0.35, -0.22, 0.32, 0.20, -0.15, 0.28];
    configure_series_tube_amp(&mut s6, 1.10, 0.55, [1.05, 0.68, 0.82, 1.05]);
    seeds.push((SeedKind::SeriesTube, s6));

    // 7. Series Multi-Stage Tube Amp: Cranked Power-Tube Sag + Warm Cab
    let mut s7 = base_seed(mono_out);
    s7.svf_base_g = 0.052;
    s7.svf_sec2_g_ratio = 1.85;
    s7.svf_damp_k1 = 1.05;
    s7.svf_damp_k2 = 0.90;
    s7.svf_stage2_feed_taps = [0.62, 0.20, -0.10, 0.18, 0.12, -0.08, 0.12];
    configure_series_tube_amp(&mut s7, 1.45, 0.78, [1.10, 0.92, 0.75, 0.85]);
    seeds.push((SeedKind::SeriesTube, s7));

    // 8. Moderate gated asymmetric fuzz
    let mut s8 = base_seed(mono_out);
    s8.blocks[0].gate_mix = 0.65;
    s8.blocks[0].gate_sharpness = 40.0;
    s8.blocks[0].atk_alpha = 0.55;
    s8.blocks[0].rel_alpha = 0.96;
    configure_cascade_lanes(
        &mut s8,
        [6.5, 4.5, 8.0, 3.4],
        [0.45, 0.68, 0.84, 0.93],
        0.60,
        0.65,
        [-0.95, 0.82, -0.72, 0.58],
        [-0.60, 0.50, -0.42, 0.35],
        0.75,
    );
    seeds.push((SeedKind::General, s8));

    // 9. Hard-gated chaotic asymmetric fuzz
    let mut s9 = base_seed(mono_out);
    s9.blocks[0].gate_mix = 0.92;
    s9.blocks[0].gate_sharpness = 160.0;
    s9.blocks[0].atk_alpha = 0.42;
    s9.blocks[0].rel_alpha = 0.935;
    s9.blocks[1].gate_mix = 0.45;
    s9.blocks[1].gate_sharpness = 85.0;
    configure_cascade_lanes(
        &mut s9,
        [9.8, 7.2, 11.5, 5.4],
        [0.40, 0.64, 0.82, 0.92],
        0.52,
        0.78,
        [-1.35, 1.15, -1.05, 0.88],
        [-0.85, 0.68, -0.58, 0.46],
        0.85,
    );
    seeds.push((SeedKind::General, s9));

    // 10. Thick dark gated fuzz
    let mut s10 = base_seed(mono_out);
    s10.blocks[0].gate_mix = 0.35;
    s10.blocks[0].gate_sharpness = 22.0;
    configure_cascade_lanes(
        &mut s10,
        [7.5, 5.5, 8.5, 4.0],
        [0.75, 0.85, 0.92, 0.96],
        0.95,
        0.25,
        [-1.10, 0.90, -0.80, 0.60],
        [-0.50, 0.40, -0.35, 0.25],
        0.70,
    );
    seeds.push((SeedKind::General, s10));

    // 11. Optocoupler high-Q upward BP EnvSVF (linear-in-octaves + smooth [g_min, g_max] rails)
    let mut s11 = base_seed(mono_out);
    s11.svf_base_g = 0.021;
    s11.svf_g_min = 0.015;
    s11.svf_g_max = 0.22;
    s11.svf_sec2_g_ratio = 1.02;
    s11.svf_env_oct_sens = 8.4;
    s11.svf_env_comp = 0.02;
    s11.svf_atk_alpha = 0.968;
    s11.svf_rel_alpha = 0.9988;
    s11.svf_det_lp_pole = 0.15;
    s11.svf_det_hp_mix = 0.0;
    s11.svf_damp_k1 = 0.15;
    s11.svf_damp_k2 = 0.65;
    s11.svf_stage2_feed_taps = [0.0, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0];
    configure_env_svf_saturator(&mut s11, 1.0);
    seeds.push((SeedKind::EnvSvf, s11));

    // 12. Optocoupler high-Q upward LP EnvSVF (linear-in-octaves + rail)
    let mut s12 = base_seed(mono_out);
    s12.svf_base_g = 0.020;
    s12.svf_g_min = 0.012;
    s12.svf_g_max = 0.30;
    s12.svf_sec2_g_ratio = 1.25;
    s12.svf_env_oct_sens = 6.5;
    s12.svf_env_comp = 0.08;
    s12.svf_atk_alpha = 0.95;
    s12.svf_rel_alpha = 0.997;
    s12.svf_det_lp_pole = 0.72;
    s12.svf_damp_k1 = 0.18;
    s12.svf_damp_k2 = 0.55;
    s12.svf_stage2_feed_taps = [0.0, 1.4, 0.15, 0.0, 0.0, 0.0, 0.0];
    configure_env_svf_saturator(&mut s12, 0.95);
    seeds.push((SeedKind::EnvSvf, s12));

    // 13. Optocoupler high-Q upward HP EnvSVF (linear-in-octaves + rail)
    let mut s13 = base_seed(mono_out);
    s13.svf_base_g = 0.024;
    s13.svf_g_min = 0.015;
    s13.svf_g_max = 0.32;
    s13.svf_sec2_g_ratio = 1.25;
    s13.svf_env_oct_sens = 6.5;
    s13.svf_env_comp = 0.08;
    s13.svf_atk_alpha = 0.95;
    s13.svf_rel_alpha = 0.997;
    s13.svf_det_lp_pole = 0.72;
    s13.svf_damp_k1 = 0.18;
    s13.svf_damp_k2 = 0.55;
    s13.svf_stage2_feed_taps = [0.0, 0.0, 0.15, 1.4, 0.0, 0.0, 0.0];
    configure_env_svf_saturator(&mut s13, 0.95);
    seeds.push((SeedKind::EnvSvf, s13));

    // 14. Mid-corner upward coupled EnvSVF (mild optocoupler compression)
    let mut s14 = base_seed(mono_out);
    s14.svf_base_g = 0.042;
    s14.svf_g_min = 0.022;
    s14.svf_g_max = 0.45;
    s14.svf_sec2_g_ratio = 1.30;
    s14.svf_env_oct_sens = 5.5;
    s14.svf_env_comp = 1.8;
    s14.svf_atk_alpha = 0.94;
    s14.svf_rel_alpha = 0.996;
    s14.svf_damp_k1 = 0.22;
    s14.svf_damp_k2 = 0.35;
    s14.svf_sec2_bp_feed = 0.30;
    s14.svf_stage2_feed_taps = [0.05, 0.45, 0.95, 0.10, 0.35, 0.65, 0.05];
    configure_env_svf_saturator(&mut s14, 1.15);
    seeds.push((SeedKind::EnvSvf, s14));

    // 15. Downward ducking resonant BP EnvSVF (~2.2 kHz)
    let mut s15 = base_seed(mono_out);
    s15.svf_base_g = 0.15;
    s15.svf_g_min = 0.018;
    s15.svf_g_max = 0.35;
    s15.svf_sec2_g_ratio = 0.90;
    s15.svf_env_oct_sens = -6.5;
    s15.svf_env_comp = 0.15;
    s15.svf_atk_alpha = 0.95;
    s15.svf_rel_alpha = 0.997;
    s15.svf_damp_k1 = 0.18;
    s15.svf_damp_k2 = 0.45;
    s15.svf_stage2_feed_taps = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
    configure_env_svf_saturator(&mut s15, 1.0);
    seeds.push((SeedKind::EnvSvf, s15));

    // 16. Fast dual-section wah/formant EnvSVF (tight ceiling + linear-in-octaves)
    let mut s16 = base_seed(mono_out);
    s16.svf_base_g = 0.018;
    s16.svf_g_min = 0.012;
    s16.svf_g_max = 0.24;
    s16.svf_sec2_g_ratio = 1.8;
    s16.svf_env_oct_sens = 9.5;
    s16.svf_env_comp = 0.05;
    s16.svf_atk_alpha = 0.88;
    s16.svf_rel_alpha = 0.994;
    s16.svf_damp_k1 = 0.22;
    s16.svf_damp_k2 = 0.32;
    s16.svf_sec2_bp_feed = 0.35;
    s16.svf_stage2_feed_taps = [0.0, 0.20, 1.4, 0.0, 0.15, 0.90, 0.0];
    configure_env_svf_saturator(&mut s16, 1.05);
    seeds.push((SeedKind::EnvSvf, s16));

    // 17. Low-corner responsive resonant BP EnvSVF
    let mut s17 = base_seed(mono_out);
    s17.svf_base_g = 0.014;
    s17.svf_g_min = 0.008;
    s17.svf_g_max = 0.36;
    s17.svf_sec2_g_ratio = 1.15;
    s17.svf_env_oct_sens = 11.0;
    s17.svf_env_comp = 0.05;
    s17.svf_atk_alpha = 0.90;
    s17.svf_rel_alpha = 0.9965;
    s17.svf_det_lp_pole = 0.78;
    s17.svf_damp_k1 = 0.14;
    s17.svf_damp_k2 = 0.45;
    s17.svf_stage2_feed_taps = [0.0, 0.10, 2.2, 0.0, 0.0, 0.30, 0.0];
    configure_env_svf_saturator(&mut s17, 1.0);
    seeds.push((SeedKind::EnvSvf, s17));

    // 18. Coupled dual-stage SVF + asymmetric preamp
    let mut s18 = base_seed(mono_out);
    s18.svf_base_g = 0.038;
    s18.svf_sec2_g_ratio = 2.0;
    s18.svf_env_oct_sens = 1.6;
    s18.svf_damp_k1 = 0.40;
    s18.svf_damp_k2 = 0.30;
    s18.svf_sec2_hp_feed = 0.20;
    s18.svf_stage2_feed_taps = [0.20, 0.20, 0.70, 0.20, 0.30, 0.80, 0.10];
    configure_cascade_lanes(
        &mut s18,
        [2.4, 1.8, 2.8, 1.4],
        [0.48, 0.66, 0.82, 0.91],
        0.75,
        0.45,
        [-0.30, 0.24, -0.20, 0.16],
        [-0.18, 0.14, -0.12, 0.09],
        0.50,
    );
    seeds.push((SeedKind::General, s18));

    seeds
}

/// Builds 18 physical black-box seed templates spanning LTI, OD, series tube sag, gated fuzz, and resonant SVF.
#[must_use]
pub fn build_lite_seed_templates(mono_out: bool) -> Vec<V3OptModelF64> {
    build_tagged_templates(mono_out)
        .into_iter()
        .map(|(_, m)| m)
        .collect()
}

/// Prescreens diverse `Lite` seed templates on a capped `4,500`-sample prefix using 24-band Gabor
/// envelope homotopy + guarded `VarPro` + 18/8 L-BFGS micro-polish across top diverse archetypes.
#[must_use]
pub fn prescreen_lite_seeds(dataset: &V3TrainDataset, mono_out: bool) -> Vec<V3OptModelF64> {
    let prefix_weighted: Vec<(V3TrainStem, f64)> = dataset
        .stems
        .iter()
        .zip(&dataset.stem_weights)
        .take(2)
        .map(|(s, &w)| {
            let n = s
                .in_l
                .len()
                .min(s.in_r.len())
                .min(s.tgt_l.len())
                .min(s.tgt_r.len())
                .min(4_500);
            (
                V3TrainStem {
                    in_l: s.in_l[..n].to_vec(),
                    in_r: s.in_r[..n].to_vec(),
                    tgt_l: s.tgt_l[..n].to_vec(),
                    tgt_r: s.tgt_r[..n].to_vec(),
                    warmup: s.warmup.min(n / 4),
                },
                w,
            )
        })
        .collect();
    let screen_ds = if prefix_weighted.is_empty() {
        dataset.clone()
    } else {
        V3TrainDataset::new_weighted(prefix_weighted, dataset.sample_rate_hz)
    };
    let screen_hom = screen_ds.with_gabor_weights(0.45, 0.03);
    let screen_pol = screen_ds.with_gabor_weights(0.0, 0.01);

    let mut ws = AdjointWorkspace::default();
    let mut scored: Vec<(f64, usize, SeedKind, V3OptModelF64)> = build_tagged_templates(mono_out)
        .into_iter()
        .enumerate()
        .map(|(idx, (kind, mut seed))| {
            if !matches!(kind, SeedKind::EnvSvf) {
                select_omp_kautz_poles(&mut seed, &screen_hom, 0, 4);
                solve_bezout_varpro_cholesky(&mut seed, &screen_hom);
            }
            let loss = evaluate_loss_f64(&seed, &screen_hom, &mut ws);
            let clean = if loss.is_finite() { loss } else { f64::MAX };
            (clean, idx, kind, seed)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut chosen_pos: Vec<usize> = Vec::with_capacity(4);
    for pos in 0..scored.len().min(2) {
        chosen_pos.push(pos);
    }
    for want in [SeedKind::SeriesTube, SeedKind::EnvSvf] {
        if chosen_pos.len() < 4
            && let Some(pos) = scored
                .iter()
                .position(|(_, _, k, _)| *k == want)
                .filter(|p| !chosen_pos.contains(p))
        {
            chosen_pos.push(pos);
        }
    }
    for pos in 0..scored.len() {
        if chosen_pos.len() >= 4 {
            break;
        }
        if !chosen_pos.contains(&pos) {
            chosen_pos.push(pos);
        }
    }

    let mut refined: Vec<(f64, V3OptModelF64)> = Vec::with_capacity(chosen_pos.len());
    for pos in chosen_pos {
        let kind = scored[pos].2;
        let mut mdl = scored[pos].3.clone();
        super::run_lbfgs(&mut mdl, &screen_hom, &mut ws, 18);
        let pre_vp = evaluate_loss_f64(&mdl, &screen_pol, &mut ws);
        let backup = mdl.clone();
        select_omp_kautz_poles(&mut mdl, &screen_pol, 0, 4);
        solve_bezout_varpro_cholesky(&mut mdl, &screen_pol);
        let post_vp = evaluate_loss_f64(&mdl, &screen_pol, &mut ws);
        let accept_vp = if matches!(kind, SeedKind::EnvSvf) {
            post_vp.is_finite() && post_vp < 0.90 * pre_vp
        } else {
            post_vp.is_finite() && post_vp <= pre_vp
        };
        if !accept_vp {
            mdl = backup;
        }
        super::run_lbfgs(&mut mdl, &screen_pol, &mut ws, 8);
        let loss = evaluate_loss_f64(&mdl, &screen_pol, &mut ws);
        let clean = if loss.is_finite() { loss } else { f64::MAX };
        refined.push((clean, mdl));
    }
    refined.sort_by(|a, b| a.0.total_cmp(&b.0));
    refined.into_iter().take(3).map(|(_, m)| m).collect()
}
