#![forbid(unsafe_code)]
//! Stage D Nonlinear Stage/Tank Control Basis Fitting (`solver::stage_d_basis`, `REQ-1.6`).

use std::collections::BTreeMap;

use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::control::{ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::{ControlKind, ControlSchemaDef, DnamError};
use dnam_engine::stages::weights::{
    ARENA0_STAGE1_EXT_OFFSET as S1_EXT, ARENA0_STAGE2_EXT_OFFSET as S2_EXT,
    ARENA0_STAGE4_EXT_OFFSET as S4_EXT, PIPE_FLAG_STAGE1_EXT, PIPE_FLAG_STAGE4_EXT,
    has_pipeline_flag,
};
use dnam_engine::{DnamEngine, FtzDazGuard, Tier};

use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::directory::verify_proj_stmc_capsule;
use crate::proj::schema::{DnamProjManifest, StemRegistryEntry};
use crate::proj::{STMC_FLAG_MODE_A_SWEEP, STMC_FLAG_STEREO, fp16_bytes_to_f32};

use super::bezout_gram::canonical_besr_warmup;
use super::stage_d::{
    interpolate_sweep_windows, resolve_stationary_range_value, resolve_stationary_switch_option,
};
use super::stages::sync_clamp_descriptors_to_arena;
use super::wide_gram::extract_rank4_svd_modes;
use super::{
    factorize_uplo_u_cholesky, solve_factorized_uplo_u_2ch, solve_uplo_u_cholesky_2ch,
    uplo_u_index, uplo_u_packed_len,
};

const W_CTRL_BASE: usize = 9_536;
const W_CTRL_ULTRA_BASE: usize = 60_672;

#[inline]
fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    if off + 4 <= buf.len() {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    } else {
        0.0
    }
}

#[inline]
fn write_f32_le(buf: &mut [u8], off: usize, val: f32) {
    if off + 4 <= buf.len() {
        let clean = if val.is_finite() && val != 0.0 {
            val
        } else {
            0.0
        };
        buf[off..off + 4].copy_from_slice(&clean.to_le_bytes());
    }
}

pub(crate) fn decode_coreset_lr(fp16_bytes: &[u8], channels: usize) -> Vec<[f32; 2]> {
    let samples = fp16_bytes_to_f32(fp16_bytes);
    if channels == 2 {
        samples.as_chunks::<2>().0.to_vec()
    } else {
        samples.into_iter().map(|s| [s, s]).collect()
    }
}

/// Resolves the canonical `StarCenter` physical control vector (`default` for `Range`, `default_index` for `Switch`).
#[must_use]
pub fn star_center_control_values(manifest: &DnamProjManifest) -> Vec<f32> {
    manifest
        .base
        .controls_schema
        .iter()
        .map(|ctrl| match ctrl.kind {
            ControlKind::Range => {
                let [lo, hi] = ctrl.declared_range.unwrap_or([0.0, 1.0]);
                ctrl.default
                    .as_f64()
                    .map_or(0.5 * (lo + hi), |v| v as f32)
                    .clamp(lo, hi)
            }
            ControlKind::Switch => {
                let opts = ctrl.declared_options.as_deref().unwrap_or(&[]);
                let def_idx = ctrl
                    .default
                    .as_str()
                    .and_then(|s| opts.iter().position(|o| o == s))
                    .or_else(|| ctrl.default.as_u64().map(|u| u as usize))
                    .unwrap_or(0);
                def_idx.min(opts.len().saturating_sub(1)) as f32
            }
        })
        .collect()
}

/// Zeroes dynamic control columns `0..8` of `W_ctrl`, all of `W_ctrl_ultra`, and `main_*[*][4..8]`,
/// preserving LFO/Envelope columns `8..16` populated by Stage A.
pub(super) fn zero_w_ctrl_columns_0_8(arena_64k: &mut [u8], cp1_slots: &mut [Cp1ContributorSlot]) {
    if arena_64k.len() >= W_CTRL_BASE + 1_024 {
        for r in 0..16 {
            let row_off = W_CTRL_BASE + r * 64;
            arena_64k[row_off..row_off + 32].fill(0);
        }
    }
    if arena_64k.len() >= W_CTRL_ULTRA_BASE + 256 {
        arena_64k[W_CTRL_ULTRA_BASE..W_CTRL_ULTRA_BASE + 256].fill(0);
    }
    for slot in cp1_slots {
        for lane in 0..3 {
            slot.main_base[lane][4..8].fill(0.0);
            slot.main_ultra[lane][4..8].fill(0.0);
        }
    }
}

fn detect_active_stage_rows(arena_64k: &[u8]) -> Vec<usize> {
    let ext1 = has_pipeline_flag(arena_64k, PIPE_FLAG_STAGE1_EXT);
    let ext4 = has_pipeline_flag(arena_64k, PIPE_FLAG_STAGE4_EXT);
    let s1_gate = ext1 && read_f32_le(arena_64k, S1_EXT + 36) > 1.0e-5;
    let s1_comp = ext1 && read_f32_le(arena_64k, S1_EXT + 8) > 1.0e-5;
    let s1_pitch = ext1 && read_f32_le(arena_64k, S1_EXT + 52) > 1.0e-5;
    let s1_rm = ext1 && read_f32_le(arena_64k, S1_EXT + 60) > 1.0e-5;
    let s1_ch = ext1 && read_f32_le(arena_64k, S1_EXT + 68) > 1.0e-5;
    let s1_svf = ext1 && read_f32_le(arena_64k, S1_EXT + 100) > 1.0e-5;
    let s2_nl = read_f32_le(arena_64k, 116).abs() > 1.0e-5
        || read_f32_le(arena_64k, 120).abs() > 1.0e-5
        || read_f32_le(arena_64k, S2_EXT + 20).abs() > 1.0e-5
        || read_f32_le(arena_64k, 4_096).abs() > 1.0e-5;
    let bbd_on = ext4
        && read_f32_le(arena_64k, S4_EXT + 4) > 1.0e-5
        && read_f32_le(arena_64k, S4_EXT + 8) > 1.0;
    let tank_on = (ext4 && read_f32_le(arena_64k, S4_EXT + 28) > 1.0e-5)
        || read_f32_le(arena_64k, 4_688).abs() > 1.0e-4;
    let kautz_on = read_f32_le(arena_64k, 4_352).abs() > 1.0e-5;
    let mut rows = Vec::new();
    if s1_gate {
        rows.push(0);
    }
    if s1_comp {
        rows.extend([1, 2]);
    }
    if s1_pitch {
        rows.push(3);
    }
    if s1_gate || s1_comp || s1_pitch || s1_rm || s1_ch || s1_svf || s2_nl || bbd_on || tank_on {
        rows.push(4);
    }
    if s2_nl {
        rows.extend([5, 6, 7, 8]);
    }
    if s1_svf || kautz_on {
        rows.extend([9, 10]);
    }
    if bbd_on {
        rows.push(12);
    }
    if bbd_on || tank_on || s1_rm {
        rows.push(13);
    }
    if s1_ch || s1_svf {
        rows.extend([14, 15]);
    }
    rows
}

fn clamp_theta_row(row: usize, v: f64, base_bbd_l: f64) -> f64 {
    match row {
        0..=2 => v.clamp(-2.5, 2.5),
        3 => v.clamp(-1.0, 1.0),
        4 => v.clamp(-0.90, 6.0),
        5 | 7 => v.clamp(-3.0, 8.0),
        6 => v.clamp(-0.5, 0.5),
        8 => v.clamp(-2.0, 6.0),
        9 => v.clamp(-0.78, 0.78),
        10 => v.clamp(-1.0, 1.0),
        11 => v.clamp(-0.95, 0.95),
        12 => v.clamp(
            -(base_bbd_l * 0.85).max(0.0),
            (40_000.0 - base_bbd_l).max(0.0),
        ),
        13 => v.clamp(-0.95, 2.5),
        14 | 15 => v.clamp(-0.90, 3.0),
        _ => 0.0,
    }
}

fn eval_slot_psi(
    schema: &[ControlSchemaDef],
    descriptors: &[ControlClampDescriptor],
    cp1_slots: &[Cp1ContributorSlot],
    ctrl_vals: &[f32],
    sw_opts: &[u8],
) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let (mut psi, mut lane_map) = (Vec::new(), Vec::new());
    for (si, slot) in cp1_slots.iter().enumerate() {
        let p = usize::from(slot.control_idx);
        if p >= schema.len() {
            continue;
        }
        let desc = &descriptors[p];
        if slot.ctrl_kind == 0 {
            let (_, u_norm, _) = desc.clamp_and_normalize(ctrl_vals[p], sw_opts);
            let lob = desc.eval_lobatto_basis(u_norm);
            for lk in 0..usize::from(slot.lanes).min(3) {
                let v = f64::from(lob[lk]);
                psi.push(v);
                lane_map.push((si, lk, v));
            }
        } else if slot.ctrl_kind == 1 {
            let def_opt = (desc.default_val.max(0.0) + 0.5) as u8;
            let v = if sw_opts[p] != def_opt && sw_opts[p] == slot.option_idx {
                1.0
            } else {
                0.0
            };
            psi.push(v);
            lane_map.push((si, 0, v));
        }
    }
    (psi, lane_map)
}

fn blind_echo_delay_delta(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    base_bbd_l: f64,
) -> Option<f64> {
    let n = dry_lr.len().min(wet_lr.len()).min(36_000);
    let max_lag = (n / 2).min(28_800);
    if base_bbd_l <= 32.0 || n < 1_024 || max_lag <= 136 {
        return None;
    }
    let corr_at = |lag: usize| -> f64 {
        let (mut num, mut de, mut we) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in lag..n {
            let (x, r) = (
                f64::from(dry_lr[i - lag][0]),
                f64::from(wet_lr[i][0] - dry_lr[i][0]),
            );
            num += x * r;
            de += x * x;
            we += r * r;
        }
        let d = (de * we).sqrt();
        if d > 1.0e-9 { num.abs() / d } else { 0.0 }
    };
    let (mut best_lag, mut best_c, mut lag) = (128usize, 0.0_f64, 128usize);
    while lag <= max_lag {
        let c = corr_at(lag);
        if c > best_c {
            best_c = c;
            best_lag = lag;
        }
        lag += 8;
    }
    for l in best_lag.saturating_sub(12).max(128)..=(best_lag + 12).min(max_lag) {
        let c = corr_at(l);
        if c > best_c {
            best_c = c;
            best_lag = l;
        }
    }
    if best_c > 0.12 && ((best_lag as f64) - base_bbd_l).abs() > 2.0 {
        Some((best_lag as f64) - base_bbd_l)
    } else {
        None
    }
}

struct StemData {
    ctrl_vals: Vec<f32>,
    psi: Vec<f64>,
    wet_lr: Vec<[f32; 2]>,
    phys_bbd_l: Option<f64>,
    probe_slot: Cp1ContributorSlot,
    a_scale: f64,
    eval_n: usize,
}

fn run_stem_probe(
    manifest: &DnamProjManifest,
    base_arena: &[u8],
    stem: &StemData,
    theta: &[f64; 16],
    dry_sub: &[[f32; 2]],
    eval_stereo: bool,
) -> Option<(Vec<[f32; 2]>, f64)> {
    let mut arena = base_arena.to_vec();
    for (r, &th) in theta.iter().enumerate() {
        let off = W_CTRL_BASE + r * 64;
        arena[off..off + 32].fill(0);
        write_f32_le(
            &mut arena,
            off + 16,
            ((th / stem.a_scale) - if r == 5 { 1.0 } else { 0.0 }) as f32,
        );
    }
    let slots = [stem.probe_slot];
    crate::solver::stages::write_inline_cp1_if_fits(&mut arena, &slots);
    let mut hdr = manifest.base.clone();
    let bytes = build_dnam_container_with_cp1(&mut hdr, &arena, &arena, false, &slots).ok()?;
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes(&bytes, Tier::Full, 48_000).ok()?;
    eng.reset_to_dc_equilibrium();
    for (p, &v) in stem.ctrl_vals.iter().enumerate() {
        eng.set_control_immediate(p, v);
    }
    let n = dry_sub.len().min(stem.wet_lr.len());
    let w = canonical_besr_warmup(n);
    let (mut pred, mut sse, mut denom) = (Vec::with_capacity(n), 0.0_f64, 0.0_f64);
    let _ftz = FtzDazGuard::new();
    for i in 0..n {
        let y = eng.step_frame_tora(dry_sub[i][0], dry_sub[i][1], !eval_stereo);
        pred.push(y);
        if i >= w {
            let dl = f64::from(stem.wet_lr[i][0]) - f64::from(y[0]);
            sse += dl * dl;
            denom += 1.0;
            if eval_stereo {
                let dr = f64::from(stem.wet_lr[i][1]) - f64::from(y[1]);
                sse += dr * dr;
                denom += 1.0;
            }
        }
    }
    Some((pred, sse / denom.max(1.0)))
}

fn eval_arena_mse(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    cp1_slots: &[Cp1ContributorSlot],
    stems: &[StemData],
    dry_lr: &[[f32; 2]],
    tier: Tier,
    eval_stereo: bool,
) -> Option<f64> {
    let mut hdr = manifest.base.clone();
    let bytes =
        build_dnam_container_with_cp1(&mut hdr, arena_64k, arena_64k, false, cp1_slots).ok()?;
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes(&bytes, tier, 48_000).ok()?;
    let (mut sse, mut cnt) = (0.0_f64, 0.0_f64);
    let _ftz = FtzDazGuard::new();
    for s in stems {
        eng.reset_to_dc_equilibrium();
        for (p, &v) in s.ctrl_vals.iter().enumerate() {
            eng.set_control_immediate(p, v);
        }
        let n = s.eval_n.min(dry_lr.len()).min(s.wet_lr.len());
        let w = canonical_besr_warmup(n);
        for i in 0..n {
            let y = eng.step_frame_tora(dry_lr[i][0], dry_lr[i][1], !eval_stereo);
            if i >= w {
                let dl = f64::from(s.wet_lr[i][0]) - f64::from(y[0]);
                sse += dl * dl;
                cnt += 1.0;
                if eval_stereo {
                    let dr = f64::from(s.wet_lr[i][1]) - f64::from(y[1]);
                    sse += dr * dr;
                    cnt += 1.0;
                }
            }
        }
    }
    (cnt > 0.0).then_some(sse / cnt)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fit_nonlinear_stage_ctrl_basis(
    manifest: &DnamProjManifest,
    arena_64k: &mut [u8],
    cp1_slots: &mut [Cp1ContributorSlot],
    descriptors: &[ControlClampDescriptor],
    off_center_stems: &[&StemRegistryEntry],
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    dry_lr: &[[f32; 2]],
    star_ctrls: &[f32],
    eval_stereo: bool,
    clear_dynamic_w_ctrl: bool,
) -> Result<bool, DnamError> {
    if !clear_dynamic_w_ctrl {
        return Ok(false);
    }
    zero_w_ctrl_columns_0_8(arena_64k, cp1_slots);
    let active_rows = detect_active_stage_rows(arena_64k);
    if off_center_stems.is_empty()
        || cp1_slots.is_empty()
        || dry_lr.len() < 128
        || arena_64k.len() < W_CTRL_BASE + 1_024
        || active_rows.is_empty()
    {
        return Ok(false);
    }

    let mut base_arena = arena_64k.to_vec();
    zero_arena_b_head_dc_bias(&mut base_arena);
    let _ = sync_clamp_descriptors_to_arena(&manifest.base, &mut base_arena)?;
    let base_bbd_l = f64::from(read_f32_le(&base_arena, S4_EXT + 8));
    let def_eval_n = dry_lr
        .len()
        .min(if base_bbd_l > 1_000.0 { 8_192 } else { 3_072 });

    let mut stems = Vec::new();
    for &stem in off_center_stems {
        let Some(stmc_b) = stmc_by_id.get(&stem.chunk_id) else {
            continue;
        };
        let stmc = verify_proj_stmc_capsule(stmc_b)?;
        let is_sweep = (stmc.flags & STMC_FLAG_MODE_A_SWEEP) != 0 && stmc.windows.len() >= 2;
        if stem.user_metadata.contains_key("freeze_transition_window") || is_sweep {
            continue;
        }
        let ch = if (stmc.flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let wet_lr = decode_coreset_lr(stmc.wet_coreset_fp16, ch);
        let eval_n = def_eval_n.min(wet_lr.len());
        if eval_n < 128 {
            continue;
        }
        let p_cnt = manifest.base.controls_schema.len();
        let (mut ctrl_vals, mut sw_opts) = (vec![0.0_f32; p_cnt], vec![0_u8; p_cnt]);
        for (p, (ctrl, desc)) in manifest
            .base
            .controls_schema
            .iter()
            .zip(descriptors)
            .enumerate()
        {
            match ctrl.kind {
                ControlKind::Range => {
                    ctrl_vals[p] =
                        resolve_stationary_range_value(ctrl, p, stem, &stmc.windows, star_ctrls[p])
                }
                ControlKind::Switch => {
                    let def_opt = (desc.default_val.max(0.0) + 0.5) as u8;
                    let opt =
                        resolve_stationary_switch_option(ctrl, p, stem, &stmc.windows, def_opt);
                    sw_opts[p] = opt;
                    ctrl_vals[p] = f32::from(opt);
                }
            }
        }
        let (psi, lane_map) = eval_slot_psi(
            &manifest.base.controls_schema,
            descriptors,
            cp1_slots,
            &ctrl_vals,
            &sw_opts,
        );
        let Some(&(si, lk, bv)) = lane_map
            .iter()
            .max_by(|a, b| a.2.abs().total_cmp(&b.2.abs()))
        else {
            continue;
        };
        if bv.abs() <= 1.0e-5 {
            continue;
        }
        let mut probe_slot = cp1_slots[si];
        probe_slot.main_base = [[0.0; 8]; 3];
        probe_slot.main_ultra = [[0.0; 8]; 3];
        probe_slot.v_base = [0.0; 4];
        probe_slot.v_ultra = [0.0; 4];
        probe_slot.emb = [0.0; 3];
        let g = (1.0 / bv).clamp(-9.0, 9.0);
        probe_slot.main_base[lk][4] = g as f32;
        let phys_bbd_l = stem
            .extracted_physics
            .as_ref()
            .and_then(|p| p.delays_ms.first().copied())
            .map(|ms| f64::from(ms) * 48.0);
        stems.push(StemData {
            ctrl_vals,
            psi,
            wet_lr,
            phys_bbd_l,
            probe_slot,
            a_scale: g * bv,
            eval_n,
        });
        if stems.len() >= 14 {
            break;
        }
    }
    let Some(before_lite) = (!stems.is_empty())
        .then(|| {
            eval_arena_mse(
                manifest,
                &base_arena,
                cp1_slots,
                &stems,
                dry_lr,
                Tier::Lite,
                eval_stereo,
            )
        })
        .flatten()
    else {
        return Ok(false);
    };
    let Some(before_ultra) = eval_arena_mse(
        manifest,
        &base_arena,
        cp1_slots,
        &stems,
        dry_lr,
        Tier::Ultra,
        eval_stereo,
    ) else {
        return Ok(false);
    };

    let (k_act, n_ch_eval) = (active_rows.len(), if eval_stereo { 2 } else { 1 });
    let mut theta_per_stem = vec![[0.0_f64; 16]; stems.len()];
    for (s_idx, s) in stems.iter().enumerate() {
        let mut theta = [0.0_f64; 16];
        let (d_sub, w_sub) = (&dry_lr[..s.eval_n], &s.wet_lr[..s.eval_n]);
        let warmup = canonical_besr_warmup(s.eval_n);
        let Some((mut cur_pred, mut cur_mse)) =
            run_stem_probe(manifest, &base_arena, s, &theta, d_sub, eval_stereo)
        else {
            continue;
        };
        let mut try_seed = |row: usize, val: f64| {
            let mut t_try = theta;
            t_try[row] = clamp_theta_row(row, val, base_bbd_l);
            if let Some((pt, mt)) =
                run_stem_probe(manifest, &base_arena, s, &t_try, d_sub, eval_stereo)
                && mt < cur_mse
            {
                theta = t_try;
                cur_pred = pt;
                cur_mse = mt;
            }
        };
        if active_rows.contains(&12)
            && let Some(dc) = s
                .phys_bbd_l
                .map(|d| d - base_bbd_l)
                .or_else(|| blind_echo_delay_delta(d_sub, w_sub, base_bbd_l))
        {
            try_seed(12, dc);
        }
        for &(r, probes) in &[
            (9usize, &[-0.5_f64, -0.25, 0.25, 0.5][..]),
            (15usize, &[-0.6_f64, -0.3, 0.3, 0.7][..]),
        ] {
            if active_rows.contains(&r) {
                for &pv in probes {
                    try_seed(r, pv);
                }
            }
        }
        for _iter in 0..2 {
            let mut jac = vec![Vec::<f64>::new(); k_act];
            for (ci, &r) in active_rows.iter().enumerate() {
                let eps = if r == 12 { 1.0 } else { 0.08 };
                let mut tp = theta;
                tp[r] = clamp_theta_row(r, theta[r] + eps, base_bbd_l);
                let ae = tp[r] - theta[r];
                if ae.abs() <= 1.0e-8 {
                    continue;
                }
                if let Some((pp, _)) =
                    run_stem_probe(manifest, &base_arena, s, &tp, d_sub, eval_stereo)
                {
                    let (inv_e, pr, cr) = (1.0 / ae, pp.as_slice(), cur_pred.as_slice());
                    jac[ci] = (warmup..s.eval_n)
                        .flat_map(|i| {
                            (0..n_ch_eval)
                                .map(move |c| (f64::from(pr[i][c]) - f64::from(cr[i][c])) * inv_e)
                        })
                        .collect();
                }
            }
            let (mut jtj, mut jtr, mut md) = (
                vec![0.0_f64; uplo_u_packed_len(k_act)],
                vec![[0.0_f64; 2]; k_act],
                0.0_f64,
            );
            for j in 0..k_act {
                if jac[j].is_empty() {
                    continue;
                }
                for i in 0..=j {
                    if jac[i].is_empty() {
                        continue;
                    }
                    let dot: f64 = jac[i].iter().zip(&jac[j]).map(|(&a, &b)| a * b).sum();
                    jtj[uplo_u_index(i, j)] = dot;
                    if i == j {
                        md += dot;
                    }
                }
                let mut rd = 0.0_f64;
                let mut ik = 0usize;
                for i in warmup..s.eval_n {
                    for c in 0..n_ch_eval {
                        rd += jac[j][ik] * (f64::from(w_sub[i][c]) - f64::from(cur_pred[i][c]));
                        ik += 1;
                    }
                }
                jtr[j] = [rd, 0.0];
            }
            let damp = (1.0e-3 * (md / (k_act as f64))).max(1.0e-5);
            let Ok(delta) = solve_uplo_u_cholesky_2ch(&jtj, &jtr, damp) else {
                break;
            };
            let mut improved = false;
            for step in [1.0_f64, 0.5, 0.2] {
                let mut t_try = theta;
                for (ci, &r) in active_rows.iter().enumerate() {
                    t_try[r] = clamp_theta_row(r, theta[r] + step * delta[ci][0], base_bbd_l);
                }
                if let Some((pt, mt)) =
                    run_stem_probe(manifest, &base_arena, s, &t_try, d_sub, eval_stereo)
                    && mt < cur_mse
                {
                    theta = t_try;
                    cur_pred = pt;
                    cur_mse = mt;
                    improved = true;
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        theta_per_stem[s_idx] = theta;
    }

    let l_slot = stems[0].psi.len();
    if l_slot == 0 {
        return Ok(false);
    }
    let (mut h_psi, mut rhs_pairs, mut tr_h) = (
        vec![0.0_f64; uplo_u_packed_len(l_slot)],
        vec![[0.0_f64; 2]; l_slot],
        0.0_f64,
    );
    for j in 0..l_slot {
        for i in 0..=j {
            let dot: f64 = stems.iter().map(|s| s.psi[i] * s.psi[j]).sum();
            h_psi[uplo_u_index(i, j)] = dot;
            if i == j {
                tr_h += dot;
            }
        }
    }
    let Ok(r_h) = factorize_uplo_u_cholesky(
        l_slot,
        &h_psi,
        (1.0e-4 * (tr_h / (l_slot as f64))).max(1.0e-6),
    ) else {
        return Ok(false);
    };
    let mut u_stage = vec![vec![0.0_f64; 512]; l_slot];
    for col in (0..16).step_by(2) {
        for j in 0..l_slot {
            rhs_pairs[j] = [
                stems
                    .iter()
                    .zip(&theta_per_stem)
                    .map(|(s, t)| s.psi[j] * t[col])
                    .sum(),
                stems
                    .iter()
                    .zip(&theta_per_stem)
                    .map(|(s, t)| s.psi[j] * t[col + 1])
                    .sum(),
            ];
        }
        if let Ok(sol) = solve_factorized_uplo_u_2ch(l_slot, &r_h, &rhs_pairs) {
            for j in 0..l_slot {
                u_stage[j][col] = sol[j][0];
                u_stage[j][col + 1] = sol[j][1];
            }
        }
    }
    let (modes, mut coords) = extract_rank4_svd_modes(&u_stage);
    let max_abs = coords
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if modes.is_empty() || max_abs <= 1.0e-7 {
        return Ok(false);
    }
    let scale = if max_abs > 0.35 { 0.35 / max_abs } else { 1.0 };
    let inv_scale = 1.0 / scale;
    for row in &mut coords {
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
    for r in 0..16 {
        for (k, mode) in modes.iter().enumerate() {
            let comp = if (r == 5 && k == 0) || (r == 8 && k == 1) || (r == 10 && k == 2) {
                1.0
            } else {
                0.0
            };
            write_f32_le(
                arena_64k,
                W_CTRL_BASE + r * 64 + 16 + k * 4,
                (mode[r] * inv_scale - comp) as f32,
            );
        }
    }
    let mut cur = 0usize;
    for slot in cp1_slots.iter_mut() {
        let lanes = if slot.ctrl_kind == 0 {
            usize::from(slot.lanes).min(3)
        } else {
            1
        };
        for lk in 0..lanes {
            if cur < coords.len() {
                for k in 0..4 {
                    slot.main_base[lk][4 + k] = coords[cur][k] as f32;
                }
                cur += 1;
            }
        }
    }
    let al = eval_arena_mse(
        manifest,
        arena_64k,
        cp1_slots,
        &stems,
        dry_lr,
        Tier::Lite,
        eval_stereo,
    );
    let au = eval_arena_mse(
        manifest,
        arena_64k,
        cp1_slots,
        &stems,
        dry_lr,
        Tier::Ultra,
        eval_stereo,
    );
    if matches!((al, au), (Some(l), Some(u)) if l < 0.995 * before_lite && u < 0.995 * before_ultra)
    {
        Ok(true)
    } else {
        zero_w_ctrl_columns_0_8(arena_64k, cp1_slots);
        Ok(false)
    }
}

/// Harvests per-stem modulated `Mode 0` predictions and `256-D` feature vectors + contracted Gram
/// via `Tier::Ultra` when `stage_d_basis` is active and excites `ctrl_basis_4`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn harvest_modulated_stem_mode0(
    manifest: &DnamProjManifest,
    arena_64k: &[u8],
    cp1_slots: &[Cp1ContributorSlot],
    descriptors: &[ControlClampDescriptor],
    dry_lr: &[[f32; 2]],
    sample_stride: usize,
    stride_scale: f64,
    star_emr_w: Option<&[f64]>,
    eval_stereo: bool,
    is_mode_a_sweep: bool,
    sweep_knots: &[Vec<f32>],
    stat_range: &[f32],
    stat_switch: &[u8],
) -> Result<Option<(Vec<[f32; 256]>, Vec<[f32; 2]>, Vec<f64>)>, DnamError> {
    let has_basis = cp1_slots.iter().any(|slot| {
        let p = usize::from(slot.control_idx);
        p < descriptors.len()
            && (0..usize::from(slot.lanes).min(3))
                .any(|l| (4..8).any(|k| slot.main_base[l][k].abs() > 1.0e-7))
            && if slot.ctrl_kind == 0 {
                !sweep_knots[p].is_empty()
                    || descriptors[p]
                        .clamp_and_normalize(stat_range[p], stat_switch)
                        .1
                        .abs()
                        > 1.0e-6
            } else {
                let def_opt = (descriptors[p].default_val.max(0.0) + 0.5) as u8;
                stat_switch[p] != def_opt && stat_switch[p] == slot.option_idx
            }
    });
    if !has_basis {
        return Ok(None);
    }

    let mut hdr = manifest.base.clone();
    let bytes = build_dnam_container_with_cp1(&mut hdr, arena_64k, arena_64k, false, cp1_slots)?;
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes(&bytes, Tier::Ultra, 48_000)?;
    eng.reset_to_dc_equilibrium();
    let p_cnt = manifest.base.controls_schema.len();
    for p in 0..p_cnt {
        let v = match manifest.base.controls_schema[p].kind {
            ControlKind::Range => stat_range[p],
            ControlKind::Switch => f32::from(stat_switch[p]),
        };
        eng.set_control_immediate(p, v);
    }
    let stride = sample_stride.max(1);
    let n_total = dry_lr.len();
    let m_sub = n_total.div_ceil(stride);
    let (mut feats, mut pred0, mut w_sub) = (
        Vec::with_capacity(m_sub),
        Vec::with_capacity(m_sub),
        Vec::with_capacity(m_sub),
    );
    let mut fb = [0.0_f32; 256];
    let mono_only = !eval_stereo;
    let sweep_ids: Vec<usize> = (0..p_cnt).filter(|&p| !sweep_knots[p].is_empty()).collect();
    {
        let _ftz = FtzDazGuard::new();
        for (n, &[il, ir]) in dry_lr.iter().enumerate() {
            if is_mode_a_sweep && n % 16 == 0 && !sweep_ids.is_empty() {
                let frac = if n_total > 1 {
                    (n as f32) / ((n_total - 1) as f32)
                } else {
                    0.0
                };
                for &p in &sweep_ids {
                    eng.set_control_immediate(p, interpolate_sweep_windows(&sweep_knots[p], frac));
                }
            }
            if n % stride == 0 {
                let mut y = eng.step_frame_with_feature_bus(il, ir, &mut fb);
                if mono_only {
                    y[1] = y[0];
                }
                feats.push(fb);
                pred0.push(y);
                w_sub
                    .push(stride_scale * star_emr_w.and_then(|w| w.get(n)).copied().unwrap_or(1.0));
            } else {
                let _ = eng.step_frame_tora(il, ir, mono_only);
            }
        }
    }
    let g_stride = if stride >= 4 { 1 } else { 4 / stride };
    let zero_t = vec![[0.0_f64; 2]; feats.len()];
    let (gram, _) =
        super::wide_gram::contract_256d_gram_tiled_wide192(&feats, &zero_t, &w_sub, g_stride);
    Ok(Some((feats, pred0, gram)))
}
