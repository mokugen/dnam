#![forbid(unsafe_code)]
//! Generic Data-Driven Specialty Archetype Templates (`v3`):
//! optical dual-release dynamics, Stage I LR4 2-band & 3-band crossover dynamics/drive,
//! and Butterworth4 series tube sag parameterized strictly from `(x, y)` spectral estimates.

use std::f64::consts::{PI, SQRT_2, TAU};

use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::stage2_gru::MAX_BLOCK_LANES;

use super::SeedArchetype;
use super::specs::{BUTTER4_TWO_R_0, BUTTER4_TWO_R_1, OS2X_QUAD_PASSBAND_PRE_EMPH};
use super::spectral_id::SpectralSplitEstimate;
use crate::solver::v3_varpro::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};

const fn const_sqrt_f64(x: f64) -> f64 {
    let mut g = if x > 1.0 { x } else { 1.0 };
    let mut i = 0_u32;
    while i < 24 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

/// Equal-gain frequency multiplier `1 / sqrt((sqrt(5) - 1) / 2)` where `|2*LP - LP^2| == |(1 - LP)^2|`.
const EQ_2POLE_RATIO: f64 = 1.0 / const_sqrt_f64(0.5 * (const_sqrt_f64(5.0) - 1.0));
const _: () = assert!(EQ_2POLE_RATIO > 1.2720 && EQ_2POLE_RATIO < 1.2721);

fn lpf_gate_bias_hz_rt(fc_hz: f64, fs_hz: f64) -> f64 {
    let f = fc_hz.clamp(1.0, 0.45 * fs_hz);
    let p = (-TAU * f / fs_hz).exp().clamp(1.0e-4, 0.99995);
    ((1.0 - p) / p).ln()
}

fn svf_prewarp_g_rt(fc_hz: f64, fs_hz: f64) -> f64 {
    (PI * fc_hz.clamp(15.0, 0.42 * fs_hz) / fs_hz).tan()
}

fn base_specialty_model(mono_out: bool, os_up_aa_mix: f64, os_dec_aa_mix: f64) -> V3MatryoshkaOptF64 {
    let mut m =
        V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::dual_rate_default(mono_out));
    m.tier_os2x = [true; 4];
    m.max_active_groups = 1;
    m.active_tier_weights = [1.0, 0.0, 0.0, 0.0];
    m.os_up_zoh_mix = 0.02;
    m.os_up_quad = -0.125;
    m.os_dec_c_prev1 = 0.15;
    m.os_dec_c0 = 0.50;
    m.os_up_pre_emph = OS2X_QUAD_PASSBAND_PRE_EMPH;
    m.os_up_aa_mix = os_up_aa_mix;
    m.os_dec_aa_mix = os_dec_aa_mix;
    m
}

fn build_optical_dynamics_seed(
    mono_out: bool,
    key_link: f64,
    det_thresh: f64,
    gr_slope: f64,
    makeup: f64,
    slow_beta: f64,
    emp_poles: &[(f64, f64); 4],
) -> V3MatryoshkaOptF64 {
    let mut m = base_specialty_model(mono_out, 0.002, 0.05);

    m.svf_base_g = 0.42;
    m.svf_sec2_g_ratio = 1.0;
    m.svf_env_oct_sens = 0.0;
    m.svf_env_comp = 0.001;
    m.svf_atk_alpha = 0.92;
    m.svf_rel_alpha = 0.992;
    m.svf_det_lp_pole = 0.72;
    m.svf_det_hp_mix = 0.0;
    m.svf_damp_k1 = SQRT_2;
    m.svf_damp_k2 = SQRT_2;
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = 0.002;
    m.svf_g_max = 0.65;
    m.svf_stage2_feed_taps = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    m.feat_in_hp_beta = 0.004;
    m.feat_fast_atk_beta = 1.04e-3;
    m.feat_fast_rel_beta = 1.74e-4;
    m.feat_pinch_k = 1.0e-4;
    m.feat_slow_atk_beta = 2.60e-4;
    m.feat_slow_rel_beta = slow_beta;
    m.feat_slow_det_comp = 0.02;
    m.feat_key_link = key_link;
    m.feat_det_thresh = det_thresh;

    const IN_GAINS: [f64; 4] = [0.28, -0.28, 0.36, -0.36];
    const BR_OFF: [f64; 4] = [2.20, 1.40, 0.65, -0.10];
    const SIGNS: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    const MK_D: [f64; 4] = [0.55, 0.95, 1.35, 1.80];
    const READOUT_W1: [f64; 4] = [0.24, -0.24, 0.20, -0.20];
    let fast_w = (-0.58 * gr_slope).clamp(-5.2, -0.5);
    let slow_w = (-0.32 * gr_slope).clamp(-4.2, -0.3);
    for l in 0..4 {
        m.blocks[0].b_z[l] = 5.5;
        m.blocks[0].b_r[l] = BR_OFF[l];
        m.blocks[0].w_in_n[0][l] = IN_GAINS[l];
        m.blocks[0].w_in_r[3][l] = fast_w;
        m.blocks[0].w_in_r[4][l] = slow_w;
        m.blocks[0].b_n[l] = 0.0;

        m.blocks[1].b_z[l] = 6.0;
        m.blocks[1].b_r[l] = 4.0;
        for s in 0..4 {
            m.blocks[1].w_prev_n[l][s] = MK_D[l] * makeup * SIGNS[l] * SIGNS[s];
        }
        m.blocks[1].b_n[l] = 0.0;

        m.blocks[2].b_z[l] = 7.6;
        m.blocks[2].b_r[l] = 7.6;
        m.blocks[2].w_prev_n[l][l] = 1.15 * SIGNS[l];
        m.blocks[2].b_n[l] = 0.0;
    }
    for b in 0..3 {
        m.blocks[b].enforce_lower_triangular();
    }
    m.readouts[0].pre_cab_u_gain = [0.02, 0.02];
    m.readouts[0].fir_l[..4].copy_from_slice(&[0.92, 0.08, 0.0, 0.0]);
    m.readouts[0].fir_r[..4].copy_from_slice(&[0.92, 0.08, 0.0, 0.0]);
    for l in 0..4 {
        m.readouts[0].pre_cab_w[0][l] = 0.05 * SIGNS[l];
        m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES + l] = READOUT_W1[l];
        m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES + l] = 0.60 * READOUT_W1[l];
    }
    m.kautz_poles[..4].copy_from_slice(emp_poles);
    m.mirror_mono();
    m
}

fn build_series_tube_sag_specialty_seed(
    mono_out: bool,
    drive: f64,
    sag_fb: f64,
    emp_poles: &[(f64, f64); 4],
) -> V3MatryoshkaOptF64 {
    let mut m = base_specialty_model(mono_out, 0.85, 0.96);

    m.svf_base_g = 0.68;
    m.svf_sec2_g_ratio = 1.0;
    m.svf_env_oct_sens = 0.0;
    m.svf_env_comp = 0.001;
    m.svf_atk_alpha = 0.985;
    m.svf_rel_alpha = 0.9992;
    m.svf_det_lp_pole = 0.80;
    m.svf_damp_k1 = BUTTER4_TWO_R_1;
    m.svf_damp_k2 = BUTTER4_TWO_R_0;
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = 0.002;
    m.svf_g_max = 0.92;
    m.svf_stage2_feed_taps = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    m.feat_in_hp_beta = 0.007;
    m.feat_fast_atk_beta = 0.04;
    m.feat_fast_rel_beta = 0.003;
    m.feat_pinch_k = 0.0;
    m.feat_slow_atk_beta = 0.003;
    m.feat_slow_rel_beta = 8.0e-6;
    m.feat_slow_det_comp = 7.5;
    m.feat_det_thresh = 0.002;

    super::init_sag_group0(&mut m, drive, sag_fb, 0.85);
    for b in 0..3 {
        m.blocks[b].enforce_lower_triangular();
    }
    m.readouts[0].pre_cab_u_gain = [0.0, 0.0];
    m.readouts[0].pre_cab_hp_gain = [0.0, 0.0];
    m.readouts[0].fir_l[..4].copy_from_slice(&[0.65, 0.25, 0.10, 0.0]);
    m.readouts[0].fir_r[..4].copy_from_slice(&[0.65, 0.25, 0.10, 0.0]);
    m.readouts[0].pre_cab_w[0][0..4].fill(0.0);
    m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES..MAX_BLOCK_LANES + 4]
        .copy_from_slice(&[0.04, -0.04, 0.04, -0.04]);
    m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES..2 * MAX_BLOCK_LANES + 4]
        .copy_from_slice(&[0.35, -0.30, 0.22, 0.12]);
    m.kautz_poles[..4].copy_from_slice(emp_poles);
    m.mirror_mono();
    m
}

fn build_two_band_lr4_seed(
    mono_out: bool,
    sr_hz: f64,
    f_split_hz: f64,
    lf_drive: f64,
    hf_drive: f64,
    emp_poles: &[(f64, f64); 4],
) -> V3MatryoshkaOptF64 {
    let mut m = base_specialty_model(mono_out, 0.002, 0.05);
    let svf_g = svf_prewarp_g_rt(f_split_hz, 2.0 * sr_hz);
    m.svf_base_g = svf_g;
    m.svf_sec2_g_ratio = 1.0;
    m.svf_env_oct_sens = 0.0;
    m.svf_env_comp = 0.001;
    m.svf_atk_alpha = 0.95;
    m.svf_rel_alpha = 0.995;
    m.svf_det_lp_pole = 0.78;
    m.svf_damp_k1 = SQRT_2;
    m.svf_damp_k2 = SQRT_2;
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = 0.002;
    m.svf_g_max = (svf_g * 3.0).clamp(0.08, 0.45);
    m.svf_feat_lp2_mix = 0.9995;
    m.svf_stage2_feed_taps = [0.0, 1.0, -SQRT_2, 1.0, 0.0, 0.0, 0.0];

    m.feat_in_hp_beta = 0.012;
    m.feat_fast_atk_beta = 0.05;
    m.feat_fast_rel_beta = 0.005;
    m.feat_pinch_k = 1.0e-4;
    m.feat_slow_atk_beta = 0.004;
    m.feat_slow_rel_beta = 8.0e-5;

    m.blocks[0].b_z[0] = -4.60;
    m.blocks[0].b_r[0] = -4.60;
    m.blocks[0].w_in_n[0][0] = 0.95 * lf_drive;
    m.blocks[0].w_in_n[1][0] = -0.95 * lf_drive;

    m.blocks[0].b_z[1] = 7.40;
    m.blocks[0].b_r[1] = 7.40;
    m.blocks[0].w_in_n[0][1] = 1.35 * lf_drive;
    m.blocks[0].w_in_n[1][1] = -1.35 * lf_drive;
    m.blocks[0].w_rec_n[1][0] = 0.45;
    m.blocks[0].b_n[1] = 0.10;

    for (l, sg) in [(2_usize, 1.0_f64), (3_usize, -1.85_f64)] {
        m.blocks[0].b_z[l] = 7.40;
        m.blocks[0].b_r[l] = 7.40;
        m.blocks[0].w_in_n[1][l] = sg * hf_drive;
        m.blocks[0].b_n[l] = if l == 3 { 0.14 } else { -0.06 };
    }

    for b in [1_usize, 2] {
        for l in 0..4 {
            m.blocks[b].b_z[l] = 7.4;
            m.blocks[b].b_r[l] = 7.4;
            let sign = if (l & 1) == 0 { 1.0 } else { -1.0 };
            if l < 2 {
                m.blocks[b].w_prev_n[l][0] = 0.55 * sign;
                m.blocks[b].w_prev_n[l][1] = 0.85 * sign;
                m.blocks[b].w_in_n[0][l] = 0.45 * lf_drive * sign;
                m.blocks[b].w_in_n[1][l] = -0.45 * lf_drive * sign;
            } else {
                m.blocks[b].w_prev_n[l][2] = 0.70 * sign;
                m.blocks[b].w_prev_n[l][3] = 0.50 * sign;
                m.blocks[b].w_in_n[1][l] = 0.65 * hf_drive * sign;
            }
        }
    }
    for b in 0..3 {
        m.blocks[b].enforce_lower_triangular();
    }
    m.readouts[0].pre_cab_u_gain = [0.18, 0.18];
    m.readouts[0].pre_cab_hp_gain = [-0.12, -0.12];
    m.readouts[0].fir_l[..4].copy_from_slice(&[0.90, 0.10, 0.0, 0.0]);
    m.readouts[0].fir_r[..4].copy_from_slice(&[0.90, 0.10, 0.0, 0.0]);
    for (l, &w) in [0.42, 0.35, 0.28, -0.22].iter().enumerate() {
        m.readouts[0].pre_cab_w[0][l] = w;
        m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES + l] = 0.65 * w;
        m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES + l] = 0.85 * w;
    }
    m.kautz_poles[..4].copy_from_slice(emp_poles);
    m.mirror_mono();
    m
}

fn build_multiband_vca_seed(
    mono_out: bool,
    sr_hz: f64,
    f_lo_hz: f64,
    f_hi_hz: f64,
    vca_slope: f64,
    emp_poles: &[(f64, f64); 4],
) -> V3MatryoshkaOptF64 {
    const LPF_SCALE: f64 = 0.65;
    const DIODE_WR: f64 = 48.0;
    const DIODE_WN: f64 = 1.5;
    const ENV_BZ: [f64; 3] = [-6.65, -6.25, -5.85];

    let mut m = base_specialty_model(mono_out, 0.85, 0.96);
    let svf_g = svf_prewarp_g_rt(f_lo_hz, 2.0 * sr_hz);
    let hi_bias = lpf_gate_bias_hz_rt(f_hi_hz * EQ_2POLE_RATIO, 2.0 * sr_hz);

    m.svf_base_g = svf_g;
    m.svf_sec2_g_ratio = 1.0;
    m.svf_env_oct_sens = 0.0;
    m.svf_env_comp = 0.001;
    m.svf_atk_alpha = 0.90;
    m.svf_rel_alpha = 0.990;
    m.svf_det_lp_pole = 0.72;
    m.svf_damp_k1 = SQRT_2;
    m.svf_damp_k2 = SQRT_2;
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = 0.002;
    m.svf_g_max = (svf_g * 3.0).clamp(0.08, 0.45);
    m.svf_feat_lp2_mix = 0.9995;
    m.svf_stage2_feed_taps = [0.0, 1.0, -SQRT_2, 1.0, 0.0, 0.0, 0.0];

    m.feat_in_hp_beta = 0.006;
    m.feat_fast_atk_beta = 1.0e-2;
    m.feat_fast_rel_beta = 9.0e-4;
    m.feat_pinch_k = 1.0e-4;
    m.feat_slow_atk_beta = 2.1e-3;
    m.feat_slow_rel_beta = 9.0e-5;
    m.feat_det_thresh = 0.0;

    // Block 0: lane 0 = upper band-split LPF on u_hp; lanes 1..3 = log-compressive diode envelopes
    m.blocks[0].b_z[0] = hi_bias;
    m.blocks[0].b_r[0] = hi_bias;
    m.blocks[0].w_in_n[1][0] = LPF_SCALE;
    m.blocks[0].b_n[0] = 0.0;

    for (b, &bz) in ENV_BZ.iter().enumerate() {
        let l = 1 + b;
        m.blocks[0].b_z[l] = bz;
        m.blocks[0].b_r[l] = -3.0;
        m.blocks[0].b_n[l] = 3.0;
    }
    m.blocks[0].w_in_r[0][1] = DIODE_WR;
    m.blocks[0].w_in_r[1][1] = -DIODE_WR;
    m.blocks[0].w_in_n[0][1] = DIODE_WN;
    m.blocks[0].w_in_n[1][1] = -DIODE_WN;

    m.blocks[0].w_rec_r[2][0] = DIODE_WR / LPF_SCALE;
    m.blocks[0].w_rec_n[2][0] = DIODE_WN / LPF_SCALE;

    m.blocks[0].w_in_r[1][3] = DIODE_WR;
    m.blocks[0].w_rec_r[3][0] = -DIODE_WR / LPF_SCALE;
    m.blocks[0].w_in_n[1][3] = DIODE_WN;
    m.blocks[0].w_rec_n[3][0] = -DIODE_WN / LPF_SCALE;

    // Block 1: envelope forwarders + equal-gain mid band `(2*LP - LP^2) u_hp`
    for b in 0..3 {
        m.blocks[1].b_z[b] = 7.5;
        m.blocks[1].b_r[b] = 7.5;
        m.blocks[1].w_prev_n[b][1 + b] = 0.95;
    }
    m.blocks[1].b_z[3] = hi_bias;
    m.blocks[1].b_r[3] = hi_bias;
    m.blocks[1].w_in_n[1][3] = 2.0 * LPF_SCALE;
    m.blocks[1].w_prev_n[3][0] = -1.0;

    // Block 2: 1x-ZOH-compatible (`|w_prev_r| < 6.0`) per-band VCAs + bus saturator
    let vca_w = (-vca_slope).clamp(-5.4, -2.5);
    for b in 0..3 {
        m.blocks[2].b_z[b] = 7.5;
        m.blocks[2].b_r[b] = 1.65;
        m.blocks[2].w_prev_r[b][b] = vca_w;
    }
    m.blocks[2].w_in_n[0][0] = 0.55;
    m.blocks[2].w_in_n[1][0] = -0.55;
    m.blocks[2].w_prev_n[1][3] = 0.55 / LPF_SCALE;
    m.blocks[2].w_in_n[1][2] = 0.55;
    m.blocks[2].w_prev_n[2][3] = -0.55 / LPF_SCALE;
    m.blocks[2].b_z[3] = 7.5;
    m.blocks[2].b_r[3] = 7.5;
    for s in 0..3 {
        m.blocks[2].w_rec_n[3][s] = 0.85;
    }
    for b in 0..3 {
        m.blocks[b].enforce_lower_triangular();
    }
    m.readouts[0].pre_cab_u_gain = [0.14, 0.14];
    m.readouts[0].pre_cab_hp_gain = [-0.02, -0.02];
    m.readouts[0].fir_l[..4].copy_from_slice(&[0.92, 0.08, 0.0, 0.0]);
    m.readouts[0].fir_r[..4].copy_from_slice(&[0.92, 0.08, 0.0, 0.0]);
    for l in 0..4 {
        m.readouts[0].pre_cab_w[0][l] = 0.03;
        m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES + l] = if l < 3 { 0.04 } else { 0.0 };
        m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES + l] = if l < 3 { 0.42 } else { 0.14 };
    }
    m.kautz_poles[..4].copy_from_slice(emp_poles);
    m.mirror_mono();
    m
}

fn build_multiband_drive_seed(
    mono_out: bool,
    sr_hz: f64,
    f_lo_hz: f64,
    f_hi_hz: f64,
    f_top_hz: f64,
    drive_scale: f64,
    emp_poles: &[(f64, f64); 4],
) -> V3MatryoshkaOptF64 {
    const LPF_SCALE: f64 = 0.65;
    let mut m = base_specialty_model(mono_out, 0.85, 0.96);
    let svf_g = svf_prewarp_g_rt(f_hi_hz, 2.0 * sr_hz);
    let lo_bias = lpf_gate_bias_hz_rt(f_lo_hz, 2.0 * sr_hz);
    let top_bias = lpf_gate_bias_hz_rt(f_top_hz, 2.0 * sr_hz);
    let dc_ac_bias = lpf_gate_bias_hz_rt(15.0, 2.0 * sr_hz);

    m.svf_base_g = svf_g;
    m.svf_sec2_g_ratio = 1.0;
    m.svf_env_oct_sens = 0.0;
    m.svf_env_comp = 0.001;
    m.svf_atk_alpha = 0.95;
    m.svf_rel_alpha = 0.995;
    m.svf_det_lp_pole = 0.78;
    m.svf_damp_k1 = SQRT_2;
    m.svf_damp_k2 = SQRT_2;
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = 0.002;
    m.svf_g_max = (svf_g * 3.0).clamp(0.08, 0.45);
    m.svf_feat_lp2_mix = 0.9995;
    m.svf_stage2_feed_taps = [0.0, 1.0, -SQRT_2, 1.0, 0.0, 0.0, 0.0];

    m.feat_in_hp_beta = 0.012;
    m.feat_fast_atk_beta = 0.018;
    m.feat_fast_rel_beta = 0.0018;
    m.feat_pinch_k = 1.0e-4;
    m.feat_slow_atk_beta = 0.003;
    m.feat_slow_rel_beta = 1.6e-4;
    m.feat_det_thresh = 0.0;

    m.blocks[0].b_z[0] = lo_bias;
    m.blocks[0].b_r[0] = lo_bias;
    m.blocks[0].w_in_n[0][0] = LPF_SCALE;
    m.blocks[0].w_in_n[1][0] = -LPF_SCALE;

    m.blocks[0].b_z[1] = lo_bias;
    m.blocks[0].b_r[1] = lo_bias;
    m.blocks[0].w_rec_n[1][0] = 1.0;

    m.blocks[0].b_z[2] = -5.6;
    m.blocks[0].b_r[2] = -3.0;
    m.blocks[0].b_n[2] = 3.0;
    m.blocks[0].w_rec_r[2][1] = 42.0;
    m.blocks[0].w_rec_n[2][1] = 2.0;

    m.blocks[0].b_z[3] = top_bias;
    m.blocks[0].b_r[3] = top_bias;
    m.blocks[0].w_in_n[1][3] = 2.4 * drive_scale;
    m.blocks[0].b_n[3] = 0.06;

    const SIGNS: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    m.blocks[1].b_z[0] = 7.4;
    m.blocks[1].b_r[0] = 1.8;
    m.blocks[1].w_prev_n[0][1] = 1.0 / LPF_SCALE;
    m.blocks[1].w_prev_r[0][2] = -4.2;

    for &(l, d_mid, bn, kq) in &[
        (1_usize, 1.6 * drive_scale, 0.10_f64, 0.10_f64),
        (2, -3.2 * drive_scale, 0.16, -0.10),
    ] {
        m.blocks[1].b_z[l] = 7.4;
        m.blocks[1].b_r[l] = 7.4;
        m.blocks[1].w_in_n[0][l] = d_mid;
        m.blocks[1].w_in_n[1][l] = -d_mid;
        m.blocks[1].w_prev_n[l][1] = -d_mid / LPF_SCALE;
        m.blocks[1].b_n[l] = bn;
        m.blocks[1].k_quad[l] = kq;
    }

    m.blocks[1].b_z[3] = dc_ac_bias;
    m.blocks[1].b_r[3] = dc_ac_bias;
    m.blocks[1].w_in_n[0][3] = 1.6;
    m.blocks[1].w_in_n[1][3] = -1.6;
    m.blocks[1].w_prev_n[3][1] = -1.6 / LPF_SCALE;
    m.blocks[1].b_n[3] = 0.08;

    for l in 0..4 {
        m.blocks[2].b_z[l] = 7.4;
        m.blocks[2].b_r[l] = 7.4;
        m.blocks[2].w_prev_n[l][l] = 0.85 * SIGNS[l];
    }
    m.blocks[2].w_prev_n[2][1] = -0.38;
    m.blocks[2].w_prev_n[3][2] = -0.45;
    for b in 0..3 {
        m.blocks[b].enforce_lower_triangular();
    }
    m.readouts[0].pre_cab_u_gain = [0.22, 0.22];
    m.readouts[0].pre_cab_hp_gain = [-0.04, -0.04];
    m.readouts[0].fir_l[..4].copy_from_slice(&[0.90, 0.10, 0.0, 0.0]);
    m.readouts[0].fir_r[..4].copy_from_slice(&[0.90, 0.10, 0.0, 0.0]);
    for (l, &w) in [0.0, 0.18, 0.42, 0.36].iter().enumerate() {
        m.readouts[0].pre_cab_w[0][l] = w;
        m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES + l] = 0.35 * SIGNS[l];
        m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES + l] = 0.28 * SIGNS[l];
    }
    m.kautz_poles[..4].copy_from_slice(emp_poles);
    m.mirror_mono();
    m
}

fn build_stage0_post_tanh_seed(
    mono_out: bool,
    drv: f64,
    emp_poles: &[(f64, f64); 4],
) -> V3MatryoshkaOptF64 {
    let mut m = base_specialty_model(mono_out, 0.002, 0.05);

    m.svf_base_g = 0.42;
    m.svf_sec2_g_ratio = 1.0;
    m.svf_env_oct_sens = 0.0;
    m.svf_env_comp = 0.001;
    m.svf_damp_k1 = SQRT_2;
    m.svf_damp_k2 = SQRT_2;
    m.svf_stereo_link = if mono_out { 0.0 } else { 0.5 };
    m.svf_g_min = 0.002;
    m.svf_g_max = 0.65;
    m.svf_stage2_feed_taps = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    m.feat_in_hp_beta = 0.008;
    m.feat_fast_atk_beta = 0.04;
    m.feat_fast_rel_beta = 0.004;
    m.feat_pinch_k = 1.0e-4;
    m.feat_slow_atk_beta = 0.004;
    m.feat_slow_rel_beta = 8.0e-5;

    const LANE_DRIVES: [f64; 4] = [0.55, -0.75, 1.05, -1.40];
    const SIGNS: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    for l in 0..4 {
        m.blocks[0].b_z[l] = 7.6;
        m.blocks[0].b_r[l] = 7.6;
        m.blocks[0].w_in_n[0][l] = drv * LANE_DRIVES[l];
        m.blocks[1].b_z[l] = 7.6;
        m.blocks[1].b_r[l] = 7.6;
        m.blocks[1].w_prev_n[l][l] = 1.0 * SIGNS[l];
        m.blocks[2].b_z[l] = 7.6;
        m.blocks[2].b_r[l] = 7.6;
        m.blocks[2].w_prev_n[l][l] = 1.0 * SIGNS[l];
    }
    for b in 0..3 {
        m.blocks[b].enforce_lower_triangular();
    }
    m.readouts[0].pre_cab_u_gain = [0.25, 0.25];
    m.readouts[0].fir_l[..4].copy_from_slice(&[0.92, 0.08, 0.0, 0.0]);
    m.readouts[0].fir_r[..4].copy_from_slice(&[0.92, 0.08, 0.0, 0.0]);
    for l in 0..4 {
        m.readouts[0].pre_cab_w[0][l] = 0.24 * SIGNS[l];
        m.readouts[0].pre_cab_w[0][MAX_BLOCK_LANES + l] = 0.14 * SIGNS[l];
        m.readouts[0].pre_cab_w[0][2 * MAX_BLOCK_LANES + l] = 0.10 * SIGNS[l];
    }
    m.kautz_poles[..4].copy_from_slice(emp_poles);
    m.mirror_mono();
    m
}

/// Builds `(x, y)` data-driven physical specialty templates (optical dynamics, 2-band & 3-band
/// crossover dynamics/drive, and Butterworth4 series sag) with zero hardcoded fixture constants.
#[must_use]
pub fn build_specialty_seeds_for_dataset(
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> Vec<(SeedArchetype, V3MatryoshkaOptF64)> {
    let sr_hz = dataset.sample_rate_hz.max(8_000.0);
    let est = SpectralSplitEstimate::from_dataset(dataset, mono_out);
    let has_distinct_aux_key = mono_out
        && dataset.stems.first().is_some_and(|s| {
            let n = s.usable_len();
            let mut l_energy = 0.0_f64;
            let mut diff_energy = 0.0_f64;
            for (&a, &b) in s.in_l[..n].iter().zip(&s.in_r[..n]) {
                let fa = f64::from(a);
                let d = fa - f64::from(b);
                l_energy += fa * fa;
                diff_energy += d * d;
            }
            diff_energy > 1.0e-4 * l_energy
        });

    let mut seeds = vec![
        (
            SeedArchetype::GeneralDrive,
            build_optical_dynamics_seed(mono_out, 0.0, 0.20, 6.5, 2.2, 6.94e-6, &est.emp_poles),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_optical_dynamics_seed(mono_out, 0.0, 0.12, 11.0, 3.6, 6.94e-6, &est.emp_poles),
        ),
        (
            SeedArchetype::SeriesTubeSag,
            build_series_tube_sag_specialty_seed(mono_out, 4.8, 1.30, &est.emp_poles),
        ),
        (
            SeedArchetype::SeriesTubeSag,
            build_series_tube_sag_specialty_seed(mono_out, 7.2, 1.55, &est.emp_poles),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_two_band_lr4_seed(mono_out, sr_hz, est.f_biamp_hz, 1.2, 3.8, &est.emp_poles),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_two_band_lr4_seed(mono_out, sr_hz, est.f_biamp_hz, 1.6, 6.8, &est.emp_poles),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_multiband_vca_seed(
                mono_out,
                sr_hz,
                est.f_lo_hz,
                est.f_hi_hz,
                4.2,
                &est.emp_poles,
            ),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_multiband_vca_seed(
                mono_out,
                sr_hz,
                est.f_lo_hz,
                est.f_hi_hz,
                5.2,
                &est.emp_poles,
            ),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_multiband_drive_seed(
                mono_out,
                sr_hz,
                est.f_lo_hz,
                est.f_hi_hz,
                est.f_top_hz,
                0.85,
                &est.emp_poles,
            ),
        ),
        (
            SeedArchetype::GeneralDrive,
            build_multiband_drive_seed(
                mono_out,
                sr_hz,
                est.f_lo_hz,
                est.f_hi_hz,
                est.f_top_hz,
                1.35,
                &est.emp_poles,
            ),
        ),
    ];

    if has_distinct_aux_key {
        for &(det_th, gr_slope, makeup, slow_beta) in &[
            (0.22_f64, 6.5_f64, 3.8_f64, 7.0e-6_f64),
            (0.16, 8.5, 4.2, 7.0e-6),
            (0.24, 4.5, 2.4, 1.2e-5),
            (0.14, 10.5, 1.6, 5.0e-6),
        ] {
            seeds.push((
                SeedArchetype::GeneralDrive,
                build_optical_dynamics_seed(
                    mono_out,
                    0.999,
                    det_th,
                    gr_slope,
                    makeup,
                    slow_beta,
                    &est.emp_poles,
                ),
            ));
        }
    } else if let Some(st0) = dataset.stems.first() {
        let n = st0.usable_len();
        if n > 0 {
            let (mut peak_in, mut sum_in, mut min_in, mut max_in) =
                (0.0_f64, 0.0_f64, f64::INFINITY, f64::NEG_INFINITY);
            for &v in &st0.in_l[..n] {
                let x = f64::from(v);
                peak_in = peak_in.max(x.abs());
                sum_in += x;
                min_in = min_in.min(x);
                max_in = max_in.max(x);
            }
            let mean_in = sum_in / (n as f64);
            let has_ac = min_in < -1.0e-3 && max_in > 1.0e-3;
            if has_ac && (peak_in > 1.25 || mean_in.abs() > 0.02) {
                seeds.push((
                    SeedArchetype::GeneralDrive,
                    build_stage0_post_tanh_seed(mono_out, 0.25, &est.emp_poles),
                ));
                seeds.push((
                    SeedArchetype::GeneralDrive,
                    build_stage0_post_tanh_seed(mono_out, 0.55, &est.emp_poles),
                ));
            }
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::v3_varpro::model_f64::V3TrainStem;

    #[test]
    fn test_specialty_seeds_emit_aux_key_only_when_dual_mono_inputs_differ() {
        let identical = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: vec![0.1; 128],
                in_r: vec![0.1; 128],
                tgt_l: vec![0.1; 128],
                tgt_r: vec![0.1; 128],
                warmup: 16,
            }],
            48_000.0,
        );
        let s_same = build_specialty_seeds_for_dataset(&identical, true);
        assert_eq!(s_same.len(), 10);
        assert!(
            s_same
                .iter()
                .all(|(_, m)| (m.os_up_pre_emph - OS2X_QUAD_PASSBAND_PRE_EMPH).abs() < 1.0e-12)
        );
        assert!((s_same[4].1.svf_feat_lp2_mix - 0.9995).abs() < 1e-9);

        let distinct = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: vec![0.1; 128],
                in_r: vec![0.4; 128],
                tgt_l: vec![0.05; 128],
                tgt_r: vec![0.05; 128],
                warmup: 16,
            }],
            48_000.0,
        );
        let s_aux = build_specialty_seeds_for_dataset(&distinct, true);
        assert_eq!(s_aux.len(), 14);
        assert!(s_aux[10..].iter().all(|(_, m)| m.feat_key_link > 0.99));
    }
}
