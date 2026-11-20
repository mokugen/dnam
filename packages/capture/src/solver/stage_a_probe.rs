#![forbid(unsafe_code)]
//! Stage A (`Physics Probe` & Anchor-Stem Lifting): Stage 0 Crossover Estimation, Deconvolved
//! `64`-Tap Early FIR, `Steiglitz–McBride` `16`-Pole Kautz Seeding, Stage IV Reflection Delay
//! Extraction, and Bounded-State Pre-Audit (`solver::stage_a_probe`).
//!
//! Canonical Reference: `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.physics-probe]`).

use dnam_engine::provenance::STAGE3_KAUTZ_BLOCKS_PER_TIER;
use dnam_engine::stages::stage3_cab::V3MatryoshkaModel;
use dnam_engine::stages::stage4_tank::v3_fdn::V3_TANK_MAX_DECLARED_DELAY;
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, ARENA_ULTRA_BYTES, PIPE_FLAG_V3_MATRYOSHKA, has_pipeline_flag,
    pack_v3_matryoshka_to_arena, unpack_v3_matryoshka_from_arena,
};

use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::schema::DnamProjManifest;

pub mod id;
pub mod refl;

use self::id::{identify_steiglitz_mcbride_and_linearity, solve_early_fir_64_stereo};
use self::refl::extract_stage4_reflection_delays;

/// Configuration for Stage A physics probing & anchor seeding (`r[train.pipeline.physics-probe]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageAConfig {
    /// Preserve an already-populated `v3` physical backbone (`true` for white-box oracles or
    /// Path-1 incremental vault edits; `false` to seed fresh physical priors).
    pub preserve_existing_backbone: bool,
    /// Number of coordinate Variable-Projection refinement passes (`1..=2`).
    pub vp_gn_steps: usize,
}

impl Default for StageAConfig {
    fn default() -> Self {
        Self {
            preserve_existing_backbone: true,
            vp_gn_steps: 2,
        }
    }
}

/// Telemetry emitted by [`run_stage_a_physics_probe`] (`r[train.pipeline.physics-probe]`).
#[derive(Debug, Clone, PartialEq)]
pub struct StageAReport {
    /// Wall-clock Stage A execution time in milliseconds.
    pub elapsed_ms: f32,
    /// True if an existing non-trivial `v3` backbone was preserved rather than re-synthesized.
    pub preserved_backbone: bool,
    /// Stage 0 clean-sub / TPT LR4 crossover frequency in Hz (`80..260 Hz`).
    pub crossover_hz: f32,
    /// Energy of the lifted `64`-tap Early FIR impulse response.
    pub early_fir_energy: f64,
    /// Number of lifted shared Kautz stereo pole pairs (`16` across `Lite..Ultra`).
    pub kautz_poles_lifted: usize,
    /// Optional Phase-2 `[CLAMP-RAIL]` prescription emitted if corner drive saturated.
    pub phase2_rail_warning: Option<String>,
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

/// Returns `true` if `arena` contains a populated `v3` Matryoshka physical backbone (distinct
/// from an uninitialized zero buffer or blank resting identity template).
#[must_use]
pub fn has_populated_v3_backbone(arena: &[u8]) -> bool {
    if arena.len() < ARENA_LITE_BYTES {
        return false;
    }
    if !has_pipeline_flag(arena, PIPE_FLAG_V3_MATRYOSHKA) {
        return false;
    }
    let model = unpack_v3_matryoshka_from_arena(arena, false);
    let gru_active = model
        .blocks
        .iter()
        .any(|blk| blk.w_in_n.iter().flatten().any(|&v| v.abs() > 1.0e-6));
    let fir_active = model.readouts[0].fir_l[1..16]
        .iter()
        .any(|&v| v.abs() > 1.0e-6);
    let kautz_active = model.readouts[0].kautz_c_l[..16]
        .iter()
        .any(|&(re, im)| re.abs() > 1.0e-6 || im.abs() > 1.0e-6);
    let s4_active = model.stage4[0].wet_mix.abs() > 1.0e-6;
    gru_active || fir_active || kautz_active || s4_active
}

fn estimate_crossover_hz(spectral_summary_4096: Option<&[u8; 4_096]>) -> f32 {
    let Some(spec) = spectral_summary_4096 else {
        return 140.0;
    };
    let bin_hz = 48_000.0_f64 / 4_096.0;
    let mut sub_e = 0.0_f64;
    let mut mid_e = 0.0_f64;
    for b in 2..64usize {
        let off = (b / 2) * 4;
        let m = f64::from(f32::from_le_bytes(spec[off..off + 4].try_into().unwrap()));
        let f_hz = (b as f64) * bin_hz;
        if f_hz < 180.0 {
            sub_e += m * m;
        } else {
            mid_e += m * m;
        }
    }
    let ratio = (sub_e / (sub_e + mid_e + 1.0e-12)).clamp(0.0, 1.0);
    (95.0 + 130.0 * ratio as f32).clamp(80.0, 260.0)
}

/// Runs Stage A (`Physics Probe` & Anchor Seeding) in `28,672 B` `v3` layout (`r[train.pipeline.physics-probe]`).
pub fn run_stage_a_physics_probe(
    manifest: &DnamProjManifest,
    arena: &mut [u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    spectral_summary_4096: Option<&[u8; 4_096]>,
    config: StageAConfig,
) -> StageAReport {
    let timer = now_ms_token();
    if arena.len() < ARENA_LITE_BYTES {
        return StageAReport {
            elapsed_ms: elapsed_ms_since(timer),
            preserved_backbone: false,
            crossover_hz: 140.0,
            early_fir_energy: 0.0,
            kautz_poles_lifted: 0,
            phase2_rail_warning: None,
        };
    }

    let mono_out = manifest.base.io_config.output_mode == "mono";
    let total_poles: usize = STAGE3_KAUTZ_BLOCKS_PER_TIER.iter().sum::<usize>() * 4;

    if config.preserve_existing_backbone && has_populated_v3_backbone(arena) {
        let existing = unpack_v3_matryoshka_from_arena(arena, mono_out);
        let fir_e: f64 = existing.readouts[0]
            .fir_l
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let (_, safe_scale, sat) = crate::audit::audit_phase2_gru_state_saturation(arena, 3.0);
        let warn = sat.then(|| {
            format!(
                "[CLAMP-RAIL] GRU corner drive saturated on preserved backbone; scale {safe_scale:.3}"
            )
        });
        return StageAReport {
            elapsed_ms: elapsed_ms_since(timer),
            preserved_backbone: true,
            crossover_hz: existing.stage0.xo_hz.clamp(80.0, 260.0),
            early_fir_energy: fir_e,
            kautz_poles_lifted: total_poles,
            phase2_rail_warning: warn,
        };
    }

    type ReflTierDelays = ([f32; 4], [f32; 4], [f32; 4], [f32; 8]);
    let n_fit = dry_lr.len().min(wet_lr.len()).min(12_000);
    let crossover_hz = estimate_crossover_hz(spectral_summary_4096);
    let (mut fir_l, mut fir_r) = ([0.0_f32; 64], [0.0_f32; 64]);
    let mut sm_poles = Vec::new();
    let mut refl_tier_delays: Option<ReflTierDelays> = None;
    let mut g_loop_hint = 0.0_f32;

    if n_fit >= 64 {
        let (fl, fr) = solve_early_fir_64_stereo(&dry_lr[..n_fit], &wet_lr[..n_fit]);
        fir_l = fl;
        fir_r = fr;

        let (_, poles, _) =
            identify_steiglitz_mcbride_and_linearity(&dry_lr[..n_fit], &wet_lr[..n_fit], 1.25_f32);
        sm_poles = poles;
        let (dl, df, dp, du, gl_est, _has_refl) =
            extract_stage4_reflection_delays(&dry_lr[..n_fit], &wet_lr[..n_fit]);
        refl_tier_delays = Some((dl, df, dp, du));
        g_loop_hint = gl_est.clamp(0.0, 0.9990);
    } else {
        fir_l[0] = 1.0;
        fir_r[0] = 1.0;
    }

    let mut model =
        if arena.len() >= ARENA_ULTRA_BYTES && has_pipeline_flag(arena, PIPE_FLAG_V3_MATRYOSHKA) {
            unpack_v3_matryoshka_from_arena(arena, mono_out)
        } else {
            V3MatryoshkaModel::identity(true, mono_out)
        };
    model.stage0.xo_hz = crossover_hz;
    for ro in &mut model.readouts {
        ro.fir_l = fir_l;
        ro.fir_r = fir_r;
    }
    for (idx, &(r, theta)) in sm_poles.iter().take(total_poles).enumerate() {
        model.kautz_poles[idx] = (
            r.clamp(0.25, 0.9995),
            theta.clamp(1.0e-3, std::f32::consts::PI - 1.0e-3),
        );
    }
    for (t_idx, s4) in model.stage4.iter_mut().enumerate() {
        if let Some((ref dl, ref df, ref dp, ref du)) = refl_tier_delays {
            let src: &[f32] = match t_idx {
                0 => dl,
                1 => df,
                2 => dp,
                _ => du,
            };
            for (l, &d) in src.iter().take(s4.delay_lengths.len()).enumerate() {
                s4.delay_lengths[l] = (d.round() as usize).clamp(4, V3_TANK_MAX_DECLARED_DELAY);
            }
        }
        s4.g_loop = g_loop_hint.clamp(-0.9990, 0.9990);
    }

    pack_v3_matryoshka_to_arena(&model, arena);
    zero_arena_b_head_dc_bias(arena);

    let fir_e: f64 = fir_l.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let (_, safe_scale, sat) = crate::audit::audit_phase2_gru_state_saturation(arena, 3.0);
    let warn = sat.then(|| {
        format!(
            "[CLAMP-RAIL] GRU corner drive saturated during Stage A probe; scale {safe_scale:.3}"
        )
    });

    StageAReport {
        elapsed_ms: elapsed_ms_since(timer),
        preserved_backbone: false,
        crossover_hz,
        early_fir_energy: fir_e,
        kautz_poles_lifted: total_poles,
        phase2_rail_warning: warn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_a_physics_probe_seeds_and_preserves_v3_arena() {
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let manifest = DnamProjManifest::default_reference();
        assert!(!has_populated_v3_backbone(&arena));

        let n = 1_024usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.25 * ((i as f32) * 0.08).sin();
                [s, s * 0.8]
            })
            .collect();
        let wet: Vec<[f32; 2]> = dry.iter().map(|&[l, r]| [l * 0.9, r * 0.9]).collect();

        let rep = run_stage_a_physics_probe(
            &manifest,
            &mut arena,
            &dry,
            &wet,
            None,
            StageAConfig {
                preserve_existing_backbone: false,
                vp_gn_steps: 1,
            },
        );
        assert!(!rep.preserved_backbone);
        assert_eq!(rep.kautz_poles_lifted, 16);
        assert!((80.0..=260.0).contains(&rep.crossover_hz));

        let mut model = unpack_v3_matryoshka_from_arena(&arena, false);
        model.blocks[0].w_in_n[0][0] = 0.42;
        pack_v3_matryoshka_to_arena(&model, &mut arena);
        assert!(has_populated_v3_backbone(&arena));

        let rep_pres = run_stage_a_physics_probe(
            &manifest,
            &mut arena,
            &dry,
            &wet,
            None,
            StageAConfig::default(),
        );
        assert!(rep_pres.preserved_backbone);
    }
}
