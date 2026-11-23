//! Phase 2A generic Sobol/log-spaced prescreening and Top-K diverse basin selection
//! (`ref[train.pipeline.backbone-trajectory-opt]`, `REQ-1.9`).

use core::f64::consts::PI;

use dnam_kernels::MIN_TPT_G;

use crate::error::TrainError;
use crate::gpu::abi::{PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms, TILE_FRAMES};
use crate::losses::{CompositeLossConfig, CompositeLossEvaluator, CompositeLossWeights};
use crate::runner::{PrescreenInput, TrainRunner};

use super::super::model::{CoreQuadF64, SonaModelF64, StereoTrainingPairs};
use super::Pass2OptConfig;
use super::eval::eval_model_pass2_audio;
use super::theta::TIER_QUADS;
use super::varpro_sync::{VarProSyncCtx, materialize_tier_view, refresh_tier_readouts_varpro};

/// Ranked Phase-2A initial candidate basin returned by [`prescreen_top_k_diverse`].
#[derive(Clone, Debug)]
pub(super) struct PrescreenCandidate {
    pub model: SonaModelF64,
    pub seed_index: usize,
    pub loss: f64,
}

#[inline]
fn van_der_corput(mut idx: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    while idx > 0 {
        f /= base as f64;
        r += f * ((idx % base) as f64);
        idx /= base;
    }
    r
}

/// Generates `1..=128` pure black-box Sobol + log-spaced generic initial parameter seeds across
/// audio bandwidth `[40 Hz, 12 kHz]` matching the 16-slot `prescreen_population` ABI
/// (`[fc_ratio, q_val, w_lp/bp/hp/dir, lru_*, rect_sign, sag_depth, adaa_drive, lru_bias_drive]`),
/// starting at zero-based Sobol sequence offset `seed_offset`.
#[must_use]
pub(super) fn generate_generic_sobol_seeds_with_offset(
    num_seeds: usize,
    sr_hz: f64,
    seed_offset: usize,
) -> Vec<f32> {
    let count = num_seeds.clamp(1, 128);
    let sr = sr_hz.max(8_000.0);
    let mut seeds = Vec::with_capacity(count * PRESCREEN_SEED_PARAM_STRIDE);
    for i in 0..count {
        let seq_idx = seed_offset.saturating_add(i).saturating_add(1);
        let u0 = van_der_corput(seq_idx, 2);
        let u1 = van_der_corput(seq_idx, 3);
        let u2 = van_der_corput(seq_idx, 5);
        let u3 = van_der_corput(seq_idx, 7);
        let fc_hz = 40.0 * (12_000.0 / 40.0_f64).powf(u0);
        let fc_ratio = (fc_hz / sr).clamp(1.0e-4, 0.45) as f32;
        let q_val = (0.5 + 1.5 * u1) as f32;
        let drive = (0.25 * (64.0_f64).powf(u2)) as f32;
        let rel = (0.85 + 0.145 * u3) as f32;
        let row: [f32; PRESCREEN_SEED_PARAM_STRIDE] = [
            fc_ratio,
            q_val,
            0.25,
            0.25 + 0.5 * (u1 as f32),
            0.15,
            0.5 - (u2 as f32),
            rel,
            0.25,
            0.1 * ((u3 - 0.5).abs() as f32),
            0.0,
            0.0,
            0.1 * (u1 as f32),
            if (seq_idx & 1) == 1 { 1.0 } else { -1.0 },
            0.05 * (u0 as f32),
            drive,
            0.0,
        ];
        seeds.extend_from_slice(&row);
    }
    seeds
}

/// Generates `1..=128` pure black-box Sobol + log-spaced generic initial parameter seeds across
/// audio bandwidth `[40 Hz, 12 kHz]` matching the 16-slot `prescreen_population` ABI.
#[must_use]
pub fn generate_generic_sobol_seeds(num_seeds: usize, sr_hz: f64) -> Vec<f32> {
    generate_generic_sobol_seeds_with_offset(num_seeds, sr_hz, 0)
}

struct OctaveQuadSpec {
    freqs_hz: [f64; 4],
    k_vals: [f64; 4],
    tones: [[f64; 4]; 4],
    lru_rel: [f64; 4],
    lru_atk: [f64; 4],
    lru_kap: [f64; 4],
    sag: [f64; 4],
    drive: [f64; 4],
    bias_drive: [f64; 4],
    wf_mix: [f64; 4],
    glu: [f64; 4],
    link: f64,
}

fn set_octave_quad(quad: &mut CoreQuadF64, sr_hz: f64, spec: &OctaveQuadSpec) {
    quad.svf_g = spec.freqs_hz.map(|fc| {
        (PI * (fc / sr_hz.max(8_000.0)).clamp(1.0e-4, 0.45))
            .tan()
            .clamp(f64::from(MIN_TPT_G), 64.0)
    });
    quad.svf_k = spec.k_vals;
    quad.svf_k_nl = [0.0; 4];
    quad.tone_lp = spec.tones[0];
    quad.tone_bp = spec.tones[1];
    quad.tone_hp = spec.tones[2];
    quad.tone_ap = [0.0; 4];
    quad.tone_dir = spec.tones[3];
    quad.lru_alpha_rel = spec.lru_rel;
    quad.lru_alpha_atk = spec.lru_atk;
    quad.lru_kappa = spec.lru_kap;
    quad.lru_duhem_gamma = [0.0; 4];
    quad.lru_signed_mix = [0.0; 4];
    quad.lru_stereo_link = [spec.link; 4];
    quad.sag_depth = spec.sag;
    quad.adaa_drive = spec.drive;
    quad.lru_bias_drive = spec.bias_drive;
    quad.wavefolder_mix = spec.wf_mix;
    quad.glu_mix = spec.glu;
    quad.in_mid_weight = [1.0; 4];
    quad.in_side_weight = [1.0; 4];
}

/// Populates `core_quads[1..10]` and causal tier inter-block tiles with a strictly
/// device-agnostic multi-scale `TPT-SVF` + Selective-Duhem `ParaLRU` + sag/ADAA basis so all
/// `40` lanes across `[Lite..Ultra]` offer diverse spectral, dynamic, and nonlinear bases
/// (`REQ-1.9`, `ref[dsp.op-n.core.matryoshka-blocks]`).
fn init_universal_multiscale_core_quads(model: &mut SonaModelF64, sr_hz: f64) {
    let bp_split = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let bp_only = [[0.0; 4], [1.0; 4], [0.0; 4], [0.0; 4]];
    let low_oct = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let hi_oct = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let broadband_asym = [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.25, 0.0],
        [0.75, 0.75, 0.75, 1.0],
    ];

    let specs = [
        OctaveQuadSpec {
            freqs_hz: [85.0, 220.0, 550.0, 1_400.0],
            k_vals: [1.414; 4],
            tones: bp_split,
            lru_rel: [0.996, 0.992, 0.985, 0.970],
            lru_atk: [0.65, 0.45, 0.25, 0.15],
            lru_kap: [0.4, 0.6, 0.8, 1.0],
            sag: [0.45, 0.65, 0.80, 0.90],
            drive: [0.8, 1.2, 1.8, 2.6],
            bias_drive: [0.0, 0.15, -0.15, 0.30],
            wf_mix: [0.0; 4],
            glu: [0.0; 4],
            link: 0.5,
        },
        OctaveQuadSpec {
            freqs_hz: [320.0, 900.0, 2_400.0, 6_200.0],
            k_vals: [1.414; 4],
            tones: bp_split,
            lru_rel: [0.994, 0.988, 0.975, 0.950],
            lru_atk: [0.80, 0.60, 0.35, 0.20],
            lru_kap: [0.5, 0.7, 0.9, 1.0],
            sag: [0.60, 0.75, 0.85, 0.92],
            drive: [1.2, 2.5, 5.0, 10.0],
            bias_drive: [0.10, -0.20, 0.25, -0.35],
            wf_mix: [0.0; 4],
            glu: [0.0; 4],
            link: 0.8,
        },
        OctaveQuadSpec {
            freqs_hz: [120.0, 600.0, 2_000.0, 7_500.0],
            k_vals: [1.0; 4],
            tones: broadband_asym,
            lru_rel: [0.995, 0.990, 0.980, 0.960],
            lru_atk: [0.50, 0.35, 0.20, 0.10],
            lru_kap: [0.3, 0.5, 0.7, 0.9],
            sag: [0.25, 0.50, 0.75, 0.90],
            drive: [1.5, 4.5, 12.0, 24.0],
            bias_drive: [0.25, -0.25, 0.50, -0.50],
            wf_mix: [0.0; 4],
            glu: [0.2, -0.2, 0.35, -0.35],
            link: 0.25,
        },
        OctaveQuadSpec {
            freqs_hz: [70.0, 140.0, 280.0, 560.0],
            k_vals: [0.87; 4],
            tones: low_oct,
            lru_rel: [0.9985, 0.997, 0.993, 0.986],
            lru_atk: [0.75, 0.55, 0.35, 0.20],
            lru_kap: [0.6, 0.75, 0.85, 1.0],
            sag: [0.55, 0.75, 0.88, 0.95],
            drive: [0.9, 1.6, 2.8, 4.5],
            bias_drive: [0.0, 0.2, -0.2, 0.3],
            wf_mix: [0.0; 4],
            glu: [0.0; 4],
            link: 1.0,
        },
        OctaveQuadSpec {
            freqs_hz: [1_120.0, 2_240.0, 4_480.0, 8_960.0],
            k_vals: [0.87; 4],
            tones: hi_oct,
            lru_rel: [0.9985, 0.997, 0.993, 0.986],
            lru_atk: [0.75, 0.55, 0.35, 0.20],
            lru_kap: [0.6, 0.75, 0.85, 1.0],
            sag: [0.55, 0.75, 0.88, 0.95],
            drive: [0.9, 1.8, 3.2, 5.2],
            bias_drive: [-0.15, 0.2, -0.25, 0.3],
            wf_mix: [0.0; 4],
            glu: [0.0; 4],
            link: 1.0,
        },
        OctaveQuadSpec {
            freqs_hz: [100.0, 350.0, 1_000.0, 3_000.0],
            k_vals: [0.62, 0.55, 0.55, 0.71],
            tones: bp_only,
            lru_rel: [0.995, 0.990, 0.980, 0.960],
            lru_atk: [0.60, 0.40, 0.25, 0.15],
            lru_kap: [0.4, 0.6, 0.8, 0.95],
            sag: [0.30, 0.65, 0.85, 0.94],
            drive: [1.2, 2.5, 5.0, 8.0],
            bias_drive: [0.2, -0.3, 0.45, -0.55],
            wf_mix: [0.0, 0.0, 0.15, 0.30],
            glu: [0.3, -0.3, 0.5, -0.5],
            link: 0.5,
        },
        OctaveQuadSpec {
            freqs_hz: [180.0, 700.0, 2_200.0, 6_800.0],
            k_vals: [1.18; 4],
            tones: bp_split,
            lru_rel: [0.997, 0.991, 0.978, 0.940],
            lru_atk: [0.70, 0.50, 0.30, 0.15],
            lru_kap: [0.5, 0.7, 0.85, 1.0],
            sag: [0.50, 0.72, 0.88, 0.95],
            drive: [1.0, 2.2, 4.2, 7.5],
            bias_drive: [0.15, -0.25, 0.35, -0.45],
            wf_mix: [0.0; 4],
            glu: [0.0; 4],
            link: 0.75,
        },
        OctaveQuadSpec {
            freqs_hz: [65.0, 190.0, 480.0, 1_300.0],
            k_vals: [0.83; 4],
            tones: low_oct,
            lru_rel: [0.9992, 0.996, 0.980, 0.920],
            lru_atk: [0.85, 0.50, 0.25, 0.10],
            lru_kap: [0.6, 0.8, 0.9, 1.0],
            sag: [0.70, 0.85, 0.93, 0.95],
            drive: [0.7, 2.2, 4.5, 9.0],
            bias_drive: [0.35, -0.40, 0.60, -0.65],
            wf_mix: [0.0, 0.1, 0.25, 0.45],
            glu: [0.2, -0.25, 0.35, -0.4],
            link: 1.0,
        },
        OctaveQuadSpec {
            freqs_hz: [2_800.0, 5_200.0, 8_200.0, 11_500.0],
            k_vals: [0.83; 4],
            tones: hi_oct,
            lru_rel: [0.9992, 0.996, 0.980, 0.920],
            lru_atk: [0.85, 0.50, 0.25, 0.10],
            lru_kap: [0.6, 0.8, 0.9, 1.0],
            sag: [0.70, 0.85, 0.93, 0.95],
            drive: [0.7, 2.2, 4.5, 9.0],
            bias_drive: [-0.35, 0.40, -0.60, 0.65],
            wf_mix: [0.0, 0.1, 0.25, 0.45],
            glu: [-0.2, 0.25, -0.35, 0.4],
            link: 1.0,
        },
    ];
    for (idx, spec) in specs.iter().enumerate() {
        set_octave_quad(&mut model.core_quads[idx + 1], sr_hz, spec);
    }
    if model.core_quads[3].rect_sign.iter().all(|&v| v > 0.0) {
        model.core_quads[3].rect_sign = [1.0, -1.0, 1.0, -1.0];
    }

    // Causal multi-stage diagonal coupling across `SONA_INTERBLOCK_TILES` preserving tier prefixes:
    // `(0,1)=0, (1,2)=1, (2,3)=2, (4,5)=5, (6,7)=18, (5,7)=21, (8,9)=28`.
    for (tile_idx, gain) in [
        (0_usize, 0.40_f64),
        (1, 0.50),
        (2, 0.55),
        (5, 0.45),
        (18, 0.50),
        (21, 0.30),
        (28, 0.55),
    ] {
        for lane in 0..4 {
            model.tiles[tile_idx].w_odd_4x4[lane][lane] = gain;
        }
    }
}

pub(super) fn unpack_prescreen_seed(
    model: &mut SonaModelF64,
    seed_table: &[f32],
    seed_idx: usize,
    sample_rate_hz: f64,
) {
    let s = &seed_table
        [seed_idx * PRESCREEN_SEED_PARAM_STRIDE..(seed_idx + 1) * PRESCREEN_SEED_PARAM_STRIDE];
    let fc_ratio = f64::from(s[0]).clamp(1.0e-4, 0.45);
    let q_val = f64::from(s[1]).max(0.1);
    let g_val = (PI * fc_ratio).tan().clamp(f64::from(MIN_TPT_G), 64.0);
    let k_val = (1.0 / q_val).clamp(0.0, 8.0);
    let d0 = f64::from(s[14]).clamp(0.25, 16.0);
    let q0 = &mut model.core_quads[0];
    q0.svf_g = [g_val; 4];
    q0.svf_k = [k_val; 4];
    q0.tone_lp = [f64::from(s[2]); 4];
    q0.tone_bp = [f64::from(s[3]); 4];
    q0.tone_hp = [f64::from(s[4]); 4];
    q0.tone_dir = [f64::from(s[5]); 4];
    q0.lru_alpha_rel = [f64::from(s[6]); 4];
    q0.lru_alpha_atk = [f64::from(s[7]); 4];
    q0.lru_kappa = [f64::from(s[8]); 4];
    q0.lru_duhem_gamma = [f64::from(s[9]); 4];
    q0.lru_signed_mix = [f64::from(s[10]); 4];
    q0.lru_stereo_link = [f64::from(s[11]); 4];
    if q0.rect_sign.iter().all(|&v| v >= 0.0) {
        q0.rect_sign = [if s[12] < 0.0 { -1.0 } else { 1.0 }; 4];
    }
    q0.sag_depth = [f64::from(s[13]); 4];
    q0.adaa_drive = [
        d0,
        (0.5 * d0).clamp(0.25, 30.0),
        (2.0 * d0).clamp(0.25, 30.0),
        (4.0 * d0).clamp(0.25, 30.0),
    ];
    q0.lru_bias_drive = [f64::from(s[15]); 4];
    init_universal_multiscale_core_quads(model, sample_rate_hz);
}

fn seed_feature_vec(seed_table: &[f32], s_idx: usize) -> [f64; 4] {
    let s =
        &seed_table[s_idx * PRESCREEN_SEED_PARAM_STRIDE..(s_idx + 1) * PRESCREEN_SEED_PARAM_STRIDE];
    let log_fc = (f64::from(s[0]).max(1.0e-4).ln() + 9.21) / 8.5;
    let q_norm = (f64::from(s[1]) - 0.5) / 1.5;
    let log_drive = (f64::from(s[14]).max(0.1).ln() + 1.4) / 4.2;
    let rect = if s[12] < 0.0 { -0.5 } else { 0.5 };
    [log_fc, q_norm, log_drive, rect]
}

#[inline]
pub(super) fn tier_index_for_quads(q_count: usize) -> usize {
    match q_count {
        0..=4 => 0,
        5..=6 => 1,
        7..=8 => 2,
        _ => 3,
    }
}

/// Clones `base_model` and unpacks `seed_idx` when `core_quads` (`adaa_drive`) and `w_ro` are inert.
pub(super) fn seed_pass1_inert_defaults(
    base_model: &SonaModelF64,
    seed_table_opt: Option<&[f32]>,
    seed_idx: usize,
    seed_spec: (usize, usize, f64),
) -> SonaModelF64 {
    let mut out = base_model.clone();
    let core_quads_inert = base_model
        .core_quads
        .iter()
        .all(|q| q.adaa_drive.iter().all(|&d| d.abs() < 1.0e-9));
    let w_ro_all_zero = base_model
        .w_ro
        .iter()
        .all(|row| row.iter().all(|&w| w.abs() < 1.0e-12));
    if core_quads_inert && w_ro_all_zero {
        let (num_seeds, seed_offset, sr_hz) = seed_spec;
        let n_seeds = num_seeds.clamp(1, 128);
        let idx = seed_idx.min(n_seeds - 1);
        if let Some(tbl) = seed_table_opt {
            unpack_prescreen_seed(&mut out, tbl, idx, sr_hz);
        } else {
            let tbl = generate_generic_sobol_seeds_with_offset(n_seeds, sr_hz, seed_offset);
            unpack_prescreen_seed(&mut out, &tbl, idx, sr_hz);
        }
    }
    out
}

/// Selects a deterministic, high-energy representative sub-window start index `[0..n - win_len]`
/// across 4 evenly spaced offsets so prescreening and Phase-2D QAT never evaluate only a quiet
/// leading or stopband segment.
#[must_use]
pub(super) fn pick_representative_window_start(
    target_l: &[f64],
    target_r: &[f64],
    win_len: usize,
) -> usize {
    let n = target_l.len().min(target_r.len());
    if win_len == 0 || win_len >= n {
        return 0;
    }
    let slack = n - win_len;
    let mut best_start = 0_usize;
    let mut best_energy = -1.0_f64;
    for k in 0..4 {
        let start = (k * slack) / 3;
        let end = start + win_len;
        let energy: f64 = target_l[start..end]
            .iter()
            .zip(&target_r[start..end])
            .map(|(&l, &r)| l * l + r * r)
            .sum();
        if energy.is_finite() && energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
    }
    best_start
}

/// Evaluates `config.num_prescreen_seeds` Sobol seeds via `runner.prescreen_population`, retains up
/// to `top_k` diverse basins (alongside any Pass-1 modal baseline), scores each with fast prefix
/// VarPro projection under `homotopy_at(0.0)`, and returns candidates sorted best-first.
pub(super) fn prescreen_top_k_diverse(
    runner: &TrainRunner,
    base_model: &SonaModelF64,
    pairs: StereoTrainingPairs<'_>,
    config: Pass2OptConfig,
    top_k: usize,
    noise_pair: Option<(&[f64], &[f64])>,
) -> Result<Vec<PrescreenCandidate>, TrainError> {
    let n = pairs.aligned_len();
    let n_seeds = config.num_prescreen_seeds.clamp(1, 128);
    let k_keep = top_k.clamp(1, 8).min(n_seeds);
    let seed_table = generate_generic_sobol_seeds_with_offset(
        n_seeds,
        config.sample_rate_hz,
        config.prescreen_seed_offset,
    );
    let in_pairs: Vec<[f32; 2]> = (0..n)
        .map(|t| [pairs.in_l[t] as f32, pairs.in_r[t] as f32])
        .collect();
    let tgt_pairs: Vec<[f32; 2]> = (0..n)
        .map(|t| [pairs.target_l[t] as f32, pairs.target_r[t] as f32])
        .collect();
    let scores = runner.prescreen_population(&PrescreenInput {
        uniforms: PrescreenUniforms {
            num_seeds: n_seeds as u32,
            num_frames: n as u32,
            num_quads: 1,
            num_stems: 1,
            sample_rate_hz: config.sample_rate_hz as f32,
            eps_floor: 1.0e-5,
            _pad0: 0,
            _pad1: 0,
        },
        input_audio: &in_pairs,
        target_audio: &tgt_pairs,
        seed_params: &seed_table,
    })?;

    let num_tiles = n.div_ceil(TILE_FRAMES).max(1);
    let mut ranked: Vec<(usize, f64)> = (0..n_seeds)
        .map(|s| {
            let mut sum = 0.0_f64;
            let mut cnt = 0_usize;
            for tile in 0..num_tiles {
                if let Some(&val) = scores.get(s * num_tiles + tile)
                    && val.is_finite()
                {
                    sum += f64::from(val);
                    cnt += 1;
                }
            }
            let avg = if cnt > 0 {
                sum / (cnt as f64)
            } else {
                f64::INFINITY
            };
            (s, avg)
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut chosen_seeds = Vec::with_capacity(k_keep);
    chosen_seeds.push(ranked[0].0);
    let pool_cap = (k_keep * 6).clamp(16, n_seeds);
    let best_raw = ranked[0].1.max(1.0e-6);
    while chosen_seeds.len() < k_keep {
        let mut best_pick = None;
        let mut best_obj = f64::INFINITY;
        for &(s_idx, raw_score) in ranked.iter().take(pool_cap) {
            if chosen_seeds.contains(&s_idx) {
                continue;
            }
            let f_cand = seed_feature_vec(&seed_table, s_idx);
            let min_dist = chosen_seeds
                .iter()
                .map(|&sel| {
                    let f_sel = seed_feature_vec(&seed_table, sel);
                    f_cand
                        .iter()
                        .zip(&f_sel)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            let rel_raw = if raw_score.is_finite() {
                raw_score / best_raw
            } else {
                1.0e6
            };
            let obj = rel_raw - 0.15 * min_dist;
            if obj < best_obj {
                best_obj = obj;
                best_pick = Some(s_idx);
            }
        }
        if let Some(pick) = best_pick {
            chosen_seeds.push(pick);
        } else {
            break;
        }
    }

    let eval_ti = if config.use_progressive_matryoshka && config.active_quads > 4 {
        0
    } else {
        tier_index_for_quads(config.active_quads.clamp(1, 10))
    };
    let target_ti = tier_index_for_quads(config.active_quads.clamp(1, 10));
    let sync_top = eval_ti == target_ti;
    let q_count = TIER_QUADS[eval_ti];
    let m = if config.use_progressive_matryoshka && config.active_quads > 4 {
        n.min(2_048)
    } else {
        n.min(4_096)
    };
    let win_start = pick_representative_window_start(&pairs.target_l[..n], &pairs.target_r[..n], m);
    let win_end = win_start + m;
    let prefix_warmup = config.warmup_frames.min(m / 4);
    let prefix_pairs = StereoTrainingPairs {
        in_l: &pairs.in_l[win_start..win_end],
        in_r: &pairs.in_r[win_start..win_end],
        target_l: &pairs.target_l[win_start..win_end],
        target_r: &pairs.target_r[win_start..win_end],
    };
    let noise_l = noise_pair.map(|(nl, _)| &nl[win_start..win_end]);
    let noise_r = noise_pair.map(|(_, nr)| &nr[win_start..win_end]);
    let prefix_loss = CompositeLossEvaluator::new_for_homotopy(
        prefix_pairs.target_l,
        prefix_pairs.target_r,
        noise_l,
        noise_r,
        CompositeLossConfig {
            sample_rate_hz: config.sample_rate_hz,
            warmup_frames: prefix_warmup,
            ..CompositeLossConfig::default()
        },
    )?;
    let w0 = CompositeLossWeights::homotopy_at(0.0);
    let has_pass1_modal = base_model.active_fir_taps > 0 || base_model.active_kautz_pairs > 0;
    let w_ro0_was_zero =
        base_model.w_ro[0][6].abs() < 1.0e-9 && base_model.w_ro[0][7].abs() < 1.0e-9;

    let ctx = VarProSyncCtx {
        runner,
        pairs: prefix_pairs,
        loss_eval: &prefix_loss,
        weights: w0,
        warmup: prefix_warmup,
    };

    let mut out = Vec::with_capacity(chosen_seeds.len() + 1);
    for &seed_index in &chosen_seeds {
        let mut cand = base_model.clone();
        unpack_prescreen_seed(&mut cand, &seed_table, seed_index, config.sample_rate_hz);
        let synced = refresh_tier_readouts_varpro(&ctx, &mut cand, eval_ti, sync_top)?;
        if !synced && w_ro0_was_zero {
            let init_w = if has_pass1_modal { 0.02 } else { 0.25 };
            cand.w_ro[0][6] = init_w;
            cand.w_ro[0][7] = init_w;
        }
        let view = materialize_tier_view(&cand, eval_ti);
        let buf =
            eval_model_pass2_audio(runner, &view, prefix_pairs.in_l, prefix_pairs.in_r, q_count)?;
        let loss = prefix_loss
            .eval_and_adjoint_with_weights(&buf.pred_l, &buf.pred_r, w0, None, None)?
            .total;
        out.push(PrescreenCandidate {
            model: cand,
            seed_index,
            loss: if loss.is_finite() {
                loss
            } else {
                f64::INFINITY
            },
        });
    }

    if has_pass1_modal {
        let seeded_base = seed_pass1_inert_defaults(
            base_model,
            Some(&seed_table),
            ranked[0].0,
            (n_seeds, config.prescreen_seed_offset, config.sample_rate_hz),
        );
        let mut baseline = seeded_base.clone();
        let view = materialize_tier_view(&baseline, eval_ti);
        let buf =
            eval_model_pass2_audio(runner, &view, prefix_pairs.in_l, prefix_pairs.in_r, q_count)?;
        let loss = prefix_loss
            .eval_and_adjoint_with_weights(&buf.pred_l, &buf.pred_r, w0, None, None)?
            .total;
        if loss.is_finite() {
            let _ = refresh_tier_readouts_varpro(&ctx, &mut baseline, eval_ti, sync_top)?;
            let b_view = materialize_tier_view(&baseline, eval_ti);
            let b_buf = eval_model_pass2_audio(
                runner,
                &b_view,
                prefix_pairs.in_l,
                prefix_pairs.in_r,
                q_count,
            )?;
            let b_loss = prefix_loss
                .eval_and_adjoint_with_weights(&b_buf.pred_l, &b_buf.pred_r, w0, None, None)?
                .total;
            let (keep_model, keep_loss) = if b_loss.is_finite() && b_loss < loss {
                (baseline, b_loss)
            } else {
                (seeded_base, loss)
            };
            out.push(PrescreenCandidate {
                model: keep_model,
                seed_index: ranked[0].0,
                loss: keep_loss,
            });
        }
    }

    out.sort_by(|a, b| a.loss.total_cmp(&b.loss));
    out.truncate(k_keep);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prescreen_top_k_diverse_selection_and_pass1_baseline_preservation() {
        let runner = TrainRunner::cpu();
        let n = 512_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 0.35 * ((t as f64) * 0.05).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| 0.30 * ((t as f64) * 0.07).cos()).collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.65 * v + 0.15 * v * v * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.55 * v - 0.12 * v * v * v).collect();

        let base = SonaModelF64::new(48_000.0);
        let cfg = Pass2OptConfig {
            num_prescreen_seeds: 32,
            warmup_frames: 32,
            active_quads: 4,
            ..Pass2OptConfig::default()
        };
        let cands = prescreen_top_k_diverse(
            &runner,
            &base,
            StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            cfg,
            4,
            None,
        )
        .unwrap();
        assert_eq!(cands.len(), 4);
        let mut distinct = cands.iter().map(|c| c.seed_index).collect::<Vec<_>>();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 4, "top-k candidates must be distinct seeds");
        assert!(cands.windows(2).all(|w| w[0].loss <= w[1].loss + 1.0e-12));

        // Exact Pass-1 FIR model must win/retain low loss cleanly.
        let mut fir_model = SonaModelF64::new(48_000.0);
        fir_model.direct_s2_u_gain = [0.0; 2];
        fir_model.active_fir_taps = 2;
        fir_model.fir_l[..2].copy_from_slice(&[0.6, -0.25]);
        fir_model.fir_r[..2].copy_from_slice(&[0.5, 0.2]);
        fir_model.pre_conv_u_gain = [1.0, 1.0];
        let mut fir_tgt_l = vec![0.0_f64; n];
        let mut fir_tgt_r = vec![0.0_f64; n];
        for t in 0..n {
            let xl_p = if t > 0 { in_l[t - 1] } else { 0.0 };
            let xr_p = if t > 0 { in_r[t - 1] } else { 0.0 };
            fir_tgt_l[t] = 0.6 * in_l[t] - 0.25 * xl_p;
            fir_tgt_r[t] = 0.5 * in_r[t] + 0.2 * xr_p;
        }
        let fir_cands = prescreen_top_k_diverse(
            &runner,
            &fir_model,
            StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &fir_tgt_l,
                target_r: &fir_tgt_r,
            },
            cfg,
            4,
            None,
        )
        .unwrap();
        assert!(fir_cands[0].loss < 0.05, "loss={}", fir_cands[0].loss);
        assert!(
            fir_cands[0]
                .model
                .core_quads
                .iter()
                .any(|q| q.adaa_drive.iter().any(|&d| d.abs() > 0.1)),
            "Pass-1 baseline must retain active non-zero CoreQuad drives"
        );

        let quiet_then_loud: Vec<f64> = (0..1_024)
            .map(|t| if (512..768).contains(&t) { 1.0 } else { 0.0 })
            .collect();
        let rep_start = pick_representative_window_start(&quiet_then_loud, &quiet_then_loud, 256);
        assert_ne!(rep_start, 0);
        assert_eq!(
            pick_representative_window_start(&quiet_then_loud, &quiet_then_loud, 1_024),
            0
        );
        assert_eq!(tier_index_for_quads(1), 0);
        assert_eq!(tier_index_for_quads(4), 0);
        assert_eq!(tier_index_for_quads(5), 1);
        assert_eq!(tier_index_for_quads(8), 2);
        assert_eq!(tier_index_for_quads(10), 3);
    }
}
