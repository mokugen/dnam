#![forbid(unsafe_code)]
//! Linearized `V3HotCore` control-target Jacobian solver and PCA-8+4 basis extraction
//! (`solver::stage_c_knobs::jacobian`).

use dnam_engine::Tier;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::stages::V3CoreInstance;
use dnam_engine::stages::weights::unpack_v3_tier_from_arena;

use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};

mod four_band;

use self::four_band::{FourBandJacobianCtx, has_per_band_wiener_collapse};

/// Waveform cross-correlation threshold below which a post-conditioned stem is phase-shifted.
const LOW_WAVE_COHERENCE_RHO_MAX: f64 = 0.55;
/// Unsigned waveform cross-correlation magnitude below which an active stem is heterodyne /
/// frequency-shifted (e.g., ring modulation or pitch division).
const HETERODYNE_UNCORRELATED_RHO_MAX: f64 = 0.15;
/// Minimum broadband octave level (`> -15 dB`) distinguishing active stems from `0%` volume mutes.
const HETERODYNE_MIN_ACTIVE_OCT: f32 = -2.5;
/// RMS octave band around unity within which a phase-shifted perturbation is treated as allpass.
const ALLPASS_DISPERSIVE_OCT_MAX: f32 = 0.35;
/// Tight octave half-width (`±0.25 oct ≈ ±1.5 dB`) for `post_cond` on incoherent/allpass stems.
const TIGHT_INCOHERENT_MARGIN_OCT: f32 = 0.25;
/// Maximum window length (`4,096` samples) for linearized control-target Jacobian evaluation.
const MAX_JACOBIAN_FIT_SAMPLES: usize = 4_096;
/// 4-band TPT partition-of-unity amplitude-RMS consistency weight (`sum_b (sqrt(E_wet,b) - sqrt(E_mdl,b))^2`)
/// preventing Wiener shrinkage without cross-band energy dumping on multiband/phase-shifted stems.
const FOUR_BAND_RMS_MATCH_WEIGHT: f64 = 2.5;
/// Broadband octave shortfall (`-0.55 oct ≈ -3.3 dB`) below which a Pass-1 waveform-MSE solve
/// suffered severe Wiener amplitude collapse and triggers Pass-2 4-band TPT RMS restoration.
const WIENER_COLLAPSE_OCT_FLOOR: f32 = -0.55;

const STAGE_TARGET_BOUNDS: [(f32, f32); 16] = [
    (-2.5, 2.5),  // 0: Stage I Env-SVF fc_oct
    (-0.8, 1.5),  // 1: Stage I Env-SVF k_delta
    (-0.75, 2.5), // 2: Block 0 FiLM gain
    (-0.75, 2.5), // 3: Block 1 FiLM gain
    (-0.75, 2.5), // 4: Block 2 FiLM gain
    (-2.0, 2.0),  // 5: Block 0 Zero-DC b_n shift
    (-2.0, 2.0),  // 6: Block 1 Zero-DC b_n shift
    (-2.0, 2.0),  // 7: Block 2 Zero-DC b_n shift
    (-0.6, 0.6),  // 8: Stage III Kautz formant oct shift
    (-4.0, 2.5),  // 9: Post-cond gain_oct / Tone SVF fc_oct
    (-2.2, 2.2),  // 10: Post-cond 250 Hz sub_oct / Tone SVF k_delta
    (-1.5, 1.5),  // 11: Stage IV Tank rotation psi
    (-2.2, 2.2),  // 12: Post-cond LM oct / Tone SVF LP
    (-2.2, 2.2),  // 13: Post-cond Pres oct / Tone SVF BP
    (-2.2, 2.2),  // 14: Post-cond Hi oct / Tone SVF HP
    (-0.6, 0.6),  // 15: Stage III Kautz radial-Q damp
];

const _: () = assert!(
    HETERODYNE_UNCORRELATED_RHO_MAX < LOW_WAVE_COHERENCE_RHO_MAX
        && WIENER_COLLAPSE_OCT_FLOOR < 0.0
        && WIENER_COLLAPSE_OCT_FLOOR > HETERODYNE_MIN_ACTIVE_OCT
);

/// Cached linearized control-bus target Jacobian evaluated at `StarCenter` across `Ultra` & `Lite`.
pub(super) struct StageTargetJacobian {
    pub(super) star_out_l: Vec<f32>,
    pub(super) star_out_r: Vec<f32>,
    pub(super) lite_star_l: Vec<f32>,
    pub(super) lite_star_r: Vec<f32>,
    pub(super) jac_l: Vec<Vec<f64>>,
    pub(super) jac_r: Vec<Vec<f64>>,
    pub(super) lite_jac_l: Vec<Vec<f64>>,
    pub(super) lite_jac_r: Vec<Vec<f64>>,
    pub(super) active_cols: [bool; 16],
    pub(super) gram_uplo: Vec<f64>,
    pub(super) n_fit: usize,
    pub(super) warmup: usize,
    pub(super) mono_out: bool,
}

pub(super) fn render_with_stage_targets(
    cold: &dnam_engine::stages::V3ColdTierModel,
    in_l: &[f32],
    in_r: &[f32],
    targets: &[f32; 16],
) -> (Vec<f32>, Vec<f32>) {
    let n = in_l.len();
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(cold);
    inst.reset_to_dc_equilibrium();
    inst.hot_core_mut().snap_control_bus_stage_targets(targets);
    let (mut ol, mut or_) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    inst.process_block_stereo(in_l, in_r, &mut ol, &mut or_);
    (ol, or_)
}

pub(super) fn build_stage_target_jacobian(
    arena: &[u8],
    mono_out: bool,
    dry_lr: &[[f32; 2]],
    n_fit: usize,
) -> StageTargetJacobian {
    let n = n_fit.min(dry_lr.len()).min(MAX_JACOBIAN_FIT_SAMPLES);
    let warmup = if n >= 4_096 {
        512
    } else if n > 128 {
        128.min(n / 4)
    } else {
        0
    };
    let in_l: Vec<f32> = dry_lr[..n].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr[..n]
        .iter()
        .map(|lr| if mono_out { lr[0] } else { lr[1] })
        .collect();
    let cold_ultra = unpack_v3_tier_from_arena(arena, Tier::Ultra, mono_out);
    let cold_lite = unpack_v3_tier_from_arena(arena, Tier::Lite, mono_out);
    let zero_targets = [0.0_f32; 16];
    let (star_out_l, star_out_r) =
        render_with_stage_targets(&cold_ultra, &in_l, &in_r, &zero_targets);
    let (lite_star_l, lite_star_r) =
        render_with_stage_targets(&cold_lite, &in_l, &in_r, &zero_targets);

    let mut jac_l = vec![vec![0.0_f64; n]; 16];
    let mut jac_r = vec![vec![0.0_f64; n]; 16];
    let mut lite_jac_l = vec![vec![0.0_f64; n]; 16];
    let mut lite_jac_r = vec![vec![0.0_f64; n]; 16];
    let mut active_cols = [false; 16];
    const EPS: f32 = 0.05;
    let inv_eps = 1.0_f64 / f64::from(EPS);

    for k in 0..16 {
        let mut pert = [0.0_f32; 16];
        pert[k] = EPS;
        let (pul, pur) = render_with_stage_targets(&cold_ultra, &in_l, &in_r, &pert);
        let (pll, plr) = render_with_stage_targets(&cold_lite, &in_l, &in_r, &pert);
        let mut energy_u = 0.0_f64;
        let mut energy_l = 0.0_f64;
        for t in warmup..n {
            let dl_u = (f64::from(pul[t]) - f64::from(star_out_l[t])) * inv_eps;
            let dr_u = if mono_out {
                dl_u
            } else {
                (f64::from(pur[t]) - f64::from(star_out_r[t])) * inv_eps
            };
            jac_l[k][t] = dl_u;
            jac_r[k][t] = dr_u;
            energy_u += dl_u * dl_u + if mono_out { 0.0 } else { dr_u * dr_u };

            let dl_l = (f64::from(pll[t]) - f64::from(lite_star_l[t])) * inv_eps;
            let dr_l = if mono_out {
                dl_l
            } else {
                (f64::from(plr[t]) - f64::from(lite_star_r[t])) * inv_eps
            };
            lite_jac_l[k][t] = dl_l;
            lite_jac_r[k][t] = dr_l;
            energy_l += dl_l * dl_l + if mono_out { 0.0 } else { dr_l * dr_l };
        }
        active_cols[k] = (0.5 * energy_u + 0.5 * energy_l) > 1.0e-10;
    }

    let mut gram_uplo = vec![0.0_f64; uplo_u_packed_len(16)];
    for j in 0..16 {
        if !active_cols[j] {
            gram_uplo[uplo_u_index(j, j)] = 1.0;
            continue;
        }
        for i in 0..=j {
            if !active_cols[i] {
                continue;
            }
            let mut dot_u = 0.0_f64;
            let mut dot_l = 0.0_f64;
            for t in warmup..n {
                dot_u += jac_l[i][t] * jac_l[j][t];
                dot_l += lite_jac_l[i][t] * lite_jac_l[j][t];
                if !mono_out {
                    dot_u += jac_r[i][t] * jac_r[j][t];
                    dot_l += lite_jac_r[i][t] * lite_jac_r[j][t];
                }
            }
            gram_uplo[uplo_u_index(i, j)] = 0.5 * dot_u + 0.5 * dot_l;
        }
    }

    StageTargetJacobian {
        star_out_l,
        star_out_r,
        lite_star_l,
        lite_star_r,
        jac_l,
        jac_r,
        lite_jac_l,
        lite_jac_r,
        active_cols,
        gram_uplo,
        n_fit: n,
        warmup,
        mono_out,
    }
}

struct GnSolveCtx<'a> {
    cold_ultra: &'a dnam_engine::stages::V3ColdTierModel,
    cold_lite: &'a dnam_engine::stages::V3ColdTierModel,
    eff_jac: &'a StageTargetJacobian,
    in_l: &'a [f32],
    in_r: &'a [f32],
    wet_slice: &'a [[f32; 2]],
    bounds: &'a [(f32, f32); 16],
    ridge: f64,
    warmup: usize,
    n: usize,
    mono: bool,
}

impl GnSolveCtx<'_> {
    fn eval_wave_err(&self, pl: &[f32], pr: &[f32]) -> f64 {
        let mut e = 0.0_f64;
        for t in self.warmup..self.n {
            let dl = f64::from(self.wet_slice[t][0]) - f64::from(pl[t]);
            e += dl * dl;
            if !self.mono {
                let dr = f64::from(self.wet_slice[t][1]) - f64::from(pr[t]);
                e += dr * dr;
            }
        }
        e
    }

    fn eval_energy(&self, pl: &[f32], pr: &[f32]) -> f64 {
        let mut e = 0.0_f64;
        for t in self.warmup..self.n {
            let vl = f64::from(pl[t]);
            e += vl * vl;
            if !self.mono {
                let vr = f64::from(pr[t]);
                e += vr * vr;
            }
        }
        e
    }

    fn run_gauss_newton_pass(
        &self,
        w_band: f64,
        band_ctx: Option<&FourBandJacobianCtx>,
    ) -> ([f32; 16], Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let eval_obj = |ul: &[f32], ur: &[f32], ll: &[f32], lr: &[f32], tgts: &[f32; 16]| -> f64 {
            let err_u = self.eval_wave_err(ul, ur);
            let err_l = self.eval_wave_err(ll, lr);
            let (pen_u, pen_l) = band_ctx.map_or((0.0, 0.0), |ctx| ctx.penalty(ul, ur, ll, lr));
            let norm_sq: f64 = tgts.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            0.25 * (err_u + w_band * pen_u)
                + 0.25 * (err_l + w_band * pen_l)
                + 0.5 * self.ridge * norm_sq
        };

        let mut best_targets = [0.0_f32; 16];
        let mut best_ul = self.eff_jac.star_out_l.clone();
        let mut best_ur = self.eff_jac.star_out_r.clone();
        let mut best_ll = self.eff_jac.lite_star_l.clone();
        let mut best_lr = self.eff_jac.lite_star_r.clone();
        let mut best_obj = eval_obj(&best_ul, &best_ur, &best_ll, &best_lr, &best_targets);

        for _step in 0..10 {
            let mut grad_out = [[0.0_f64; 16]; 2];
            let mut gram_step = self.eff_jac.gram_uplo.clone();
            if let Some(ctx) = band_ctx {
                ctx.accumulate_grad_and_gram(
                    w_band,
                    [&best_ul, &best_ur],
                    [&best_ll, &best_lr],
                    &self.eff_jac.active_cols,
                    &mut grad_out,
                    &mut gram_step,
                );
            }
            let mut rhs = [[0.0_f64; 2]; 16];
            for k in 0..16 {
                if !self.eff_jac.active_cols[k] {
                    continue;
                }
                let mut dot_u = 0.0_f64;
                let mut dot_l = 0.0_f64;
                for t in self.warmup..self.n {
                    let r_ul = f64::from(self.wet_slice[t][0]) - f64::from(best_ul[t]);
                    let r_ll = f64::from(self.wet_slice[t][0]) - f64::from(best_ll[t]);
                    dot_u += self.eff_jac.jac_l[k][t] * r_ul;
                    dot_l += self.eff_jac.lite_jac_l[k][t] * r_ll;
                    if !self.mono {
                        let r_ur = f64::from(self.wet_slice[t][1]) - f64::from(best_ur[t]);
                        let r_lr = f64::from(self.wet_slice[t][1]) - f64::from(best_lr[t]);
                        dot_u += self.eff_jac.jac_r[k][t] * r_ur;
                        dot_l += self.eff_jac.lite_jac_r[k][t] * r_lr;
                    }
                }
                rhs[k][0] = 0.5 * (dot_u + grad_out[0][k])
                    + 0.5 * (dot_l + grad_out[1][k])
                    - self.ridge * f64::from(best_targets[k]);
            }
            let Ok(sol) = solve_uplo_u_cholesky_2ch(&gram_step, &rhs, self.ridge) else {
                break;
            };
            let mut improved = false;
            for &alpha in &[1.0_f32, 0.5, 0.25, 0.125, 0.05] {
                let mut cand = best_targets;
                for k in 0..16 {
                    if self.eff_jac.active_cols[k] {
                        let (lo, hi) = self.bounds[k];
                        cand[k] = (cand[k] + alpha * (sol[k][0] as f32)).clamp(lo, hi);
                    }
                }
                let (cul, cur) =
                    render_with_stage_targets(self.cold_ultra, self.in_l, self.in_r, &cand);
                let (cll, clr) =
                    render_with_stage_targets(self.cold_lite, self.in_l, self.in_r, &cand);
                let obj = eval_obj(&cul, &cur, &cll, &clr, &cand);
                if obj + 1.0e-12 < best_obj {
                    best_obj = obj;
                    best_targets = cand;
                    best_ul = cul;
                    best_ur = cur;
                    best_ll = cll;
                    best_lr = clr;
                    improved = true;
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        (best_targets, best_ul, best_ur, best_ll, best_lr)
    }
}

pub(super) fn solve_point_stage_targets(
    arena: &[u8],
    dry_lr: &[[f32; 2]],
    wet_slice: &[[f32; 2]],
    offset: usize,
    jac: &StageTargetJacobian,
) -> [f32; 16] {
    let avail_dry = dry_lr.len().saturating_sub(offset);
    let n = jac.n_fit.min(wet_slice.len()).min(avail_dry);
    let warmup = jac.warmup.min(n);
    if n <= warmup + 8 {
        return [0.0; 16];
    }
    let mono = jac.mono_out;
    let local_jac;
    let eff_jac = if offset > 0 || n != jac.n_fit {
        local_jac = build_stage_target_jacobian(arena, mono, &dry_lr[offset..offset + n], n);
        &local_jac
    } else {
        jac
    };
    let in_l: Vec<f32> = dry_lr[offset..offset + n].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr[offset..offset + n]
        .iter()
        .map(|lr| if mono { lr[0] } else { lr[1] })
        .collect();
    let cold_ultra = unpack_v3_tier_from_arena(arena, Tier::Ultra, mono);
    let cold_lite = unpack_v3_tier_from_arena(arena, Tier::Lite, mono);

    let eval_err = |pl: &[f32], pr: &[f32]| -> f64 {
        let mut e = 0.0_f64;
        for t in warmup..n {
            let dl = f64::from(wet_slice[t][0]) - f64::from(pl[t]);
            e += dl * dl;
            if !mono {
                let dr = f64::from(wet_slice[t][1]) - f64::from(pr[t]);
                e += dr * dr;
            }
        }
        e
    };

    let init_wave_err = 0.5 * eval_err(&eff_jac.star_out_l, &eff_jac.star_out_r)
        + 0.5 * eval_err(&eff_jac.lite_star_l, &eff_jac.lite_star_r);
    if init_wave_err <= 1.0e-12 {
        return [0.0; 16];
    }

    let (mut e_wet, mut e_star_u, mut e_star_l) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut dot_wu, mut dot_wl) = (0.0_f64, 0.0_f64);
    for t in warmup..n {
        let wl = f64::from(wet_slice[t][0]);
        let sul = f64::from(eff_jac.star_out_l[t]);
        let sll = f64::from(eff_jac.lite_star_l[t]);
        e_wet += wl * wl;
        e_star_u += sul * sul;
        e_star_l += sll * sll;
        dot_wu += wl * sul;
        dot_wl += wl * sll;
        if !mono {
            let wr = f64::from(wet_slice[t][1]);
            let sur = f64::from(eff_jac.star_out_r[t]);
            let slr = f64::from(eff_jac.lite_star_r[t]);
            e_wet += wr * wr;
            e_star_u += sur * sur;
            e_star_l += slr * slr;
            dot_wu += wr * sur;
            dot_wl += wr * slr;
        }
    }
    let eps = 1.0e-12_f64;
    let rho_wave =
        0.5 * (dot_wu / ((e_wet * e_star_u).sqrt() + eps)) + 0.5 * (dot_wl / ((e_wet * e_star_l).sqrt() + eps));
    let e_star = 0.5 * e_star_u + 0.5 * e_star_l;
    let oct_tot = (0.5 * ((e_wet + eps) / (e_star + eps)).log2()) as f32;

    let mut bounds = STAGE_TARGET_BOUNDS;
    let post_cond_mode = cold_ultra.tone_svf.cab_inject_w.abs() <= 1.0e-7
        && cold_ultra.tone_svf.b1_inject_w.abs() <= 1.0e-7;
    let heterodyne_uncorrelated = post_cond_mode
        && rho_wave.abs() < HETERODYNE_UNCORRELATED_RHO_MAX
        && oct_tot > HETERODYNE_MIN_ACTIVE_OCT;
    let allpass_dispersive = post_cond_mode
        && rho_wave < LOW_WAVE_COHERENCE_RHO_MAX
        && oct_tot.abs() < ALLPASS_DISPERSIVE_OCT_MAX;
    if heterodyne_uncorrelated || allpass_dispersive {
        let lo9 = (oct_tot - TIGHT_INCOHERENT_MARGIN_OCT)
            .clamp(STAGE_TARGET_BOUNDS[9].0, STAGE_TARGET_BOUNDS[9].1);
        let hi9 = (oct_tot + TIGHT_INCOHERENT_MARGIN_OCT)
            .clamp(lo9, STAGE_TARGET_BOUNDS[9].1);
        bounds[9] = (lo9, hi9);
        for idx in [10_usize, 12, 13, 14] {
            bounds[idx] = (-TIGHT_INCOHERENT_MARGIN_OCT, TIGHT_INCOHERENT_MARGIN_OCT);
        }
    }

    let mut active_diag_sum = 0.0_f64;
    let mut active_diag_cnt = 0_usize;
    for j in 0..16 {
        if eff_jac.active_cols[j] {
            active_diag_sum += eff_jac.gram_uplo[uplo_u_index(j, j)];
            active_diag_cnt += 1;
        }
    }
    let mean_diag = active_diag_sum / (active_diag_cnt.max(1) as f64);
    let ridge = (2.0e-3 * mean_diag).max(1.0e-6);

    let gn_ctx = GnSolveCtx {
        cold_ultra: &cold_ultra,
        cold_lite: &cold_lite,
        eff_jac,
        in_l: &in_l,
        in_r: &in_r,
        wet_slice,
        bounds: &bounds,
        ridge,
        warmup,
        n,
        mono,
    };
    let (mut best_targets, mut best_ul, mut best_ur, mut best_ll, mut best_lr) =
        gn_ctx.run_gauss_newton_pass(0.0, None);
    let e_mdl_p1 =
        0.5 * gn_ctx.eval_energy(&best_ul, &best_ur) + 0.5 * gn_ctx.eval_energy(&best_ll, &best_lr);
    let oct_err_p1 = (0.5 * ((e_mdl_p1 + eps) / (e_wet + eps)).log2()) as f32;
    let need_band_recovery = post_cond_mode
        && rho_wave > -HETERODYNE_UNCORRELATED_RHO_MAX
        && oct_tot > HETERODYNE_MIN_ACTIVE_OCT
        && (oct_err_p1 < WIENER_COLLAPSE_OCT_FLOOR
            || has_per_band_wiener_collapse(
                wet_slice,
                eff_jac,
                &best_ul,
                &best_ur,
                &best_ll,
                &best_lr,
                WIENER_COLLAPSE_OCT_FLOOR,
            ));
    if need_band_recovery {
        let band_ctx = FourBandJacobianCtx::new(wet_slice, eff_jac, warmup, n);
        (best_targets, best_ul, best_ur, best_ll, best_lr) =
            gn_ctx.run_gauss_newton_pass(FOUR_BAND_RMS_MATCH_WEIGHT, Some(&band_ctx));
    }
    let final_wave_err = 0.5 * gn_ctx.eval_wave_err(&best_ul, &best_ur)
        + 0.5 * gn_ctx.eval_wave_err(&best_ll, &best_lr);
    if allpass_dispersive && final_wave_err > 0.85 * init_wave_err {
        return [0.0; 16];
    }
    best_targets
}

pub(super) fn top_target_basis_8_4(targets: &[[f32; 16]]) -> ([[f64; 8]; 16], [[f64; 4]; 16]) {
    let mut cov = vec![vec![0.0_f64; 16]; 16];
    for tgt in targets {
        for i in 0..16 {
            let ti = f64::from(tgt[i]);
            if ti != 0.0 {
                for j in i..16 {
                    cov[i][j] += ti * f64::from(tgt[j]);
                }
            }
        }
    }
    for i in 0..16 {
        for j in 0..i {
            cov[i][j] = cov[j][i];
        }
    }

    let mut eigvecs = [[0.0_f64; 16]; 16];
    for (i, row) in eigvecs.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..20 {
        for p in 0..16 {
            for q in (p + 1)..16 {
                let apq = cov[p][q];
                if apq.abs() <= 1.0e-14 {
                    continue;
                }
                let tau = (cov[q][q] - cov[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                cov[p][p] -= t * apq;
                cov[q][q] += t * apq;
                cov[p][q] = 0.0;
                cov[q][p] = 0.0;
                for r in 0..16 {
                    if r != p && r != q {
                        let arp = cov[r][p];
                        let arq = cov[r][q];
                        cov[r][p] = c * arp - s * arq;
                        cov[p][r] = cov[r][p];
                        cov[r][q] = s * arp + c * arq;
                        cov[q][r] = cov[r][q];
                    }
                    let vrp = eigvecs[r][p];
                    let vrq = eigvecs[r][q];
                    eigvecs[r][p] = c * vrp - s * vrq;
                    eigvecs[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..16).collect();
    order.sort_by(|&a, &b| cov[b][b].total_cmp(&cov[a][a]));

    let mut w_base = [[0.0_f64; 8]; 16];
    let mut w_ultra = [[0.0_f64; 4]; 16];
    for (col, &src_col) in order.iter().take(8).enumerate() {
        for row in 0..16 {
            w_base[row][col] = eigvecs[row][src_col];
        }
    }
    for (col, &src_col) in order.iter().skip(8).take(4).enumerate() {
        for row in 0..16 {
            w_ultra[row][col] = eigvecs[row][src_col];
        }
    }
    (w_base, w_ultra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::schema::DnamProjManifest;
    use crate::solver::stage_b_backbone::solve_stage_b_backbone;
    use dnam_engine::stages::weights::ARENA_ULTRA_BYTES;
    use std::f64::consts::TAU;

    #[test]
    fn test_top_target_basis_8_4_orthonormal_and_jacobian_solve() {
        let mut t1 = [0.0_f32; 16];
        t1[0] = 2.0;
        t1[3] = 1.0;
        let mut t2 = [0.0_f32; 16];
        t2[12] = -1.5;
        let (w_base, w_ultra) = top_target_basis_8_4(&[t1, t2]);
        for c in 0..8 {
            let norm_sq: f64 = (0..16).map(|r| w_base[r][c] * w_base[r][c]).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1.0e-10,
                "base col {c} norm={norm_sq}"
            );
            for c2 in (c + 1)..8 {
                let dot: f64 = (0..16).map(|r| w_base[r][c] * w_base[r][c2]).sum();
                assert!(dot.abs() < 1.0e-10);
            }
        }
        for c in 0..4 {
            let norm_sq: f64 = (0..16).map(|r| w_ultra[r][c] * w_ultra[r][c]).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1.0e-10,
                "ultra col {c} norm={norm_sq}"
            );
        }

        let manifest = DnamProjManifest::default_reference();
        let n = 512usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.25 * (TAU * 220.0 * (i as f64) / 48_000.0).sin() as f32;
                [s, s]
            })
            .collect();
        let wet_star: Vec<[f32; 2]> = dry.iter().map(|&[l, r]| [0.8 * l, 0.8 * r]).collect();
        let wet_gain: Vec<[f32; 2]> = dry.iter().map(|&[l, r]| [1.2 * l, 1.2 * r]).collect();
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let _ =
            solve_stage_b_backbone(&manifest, &mut arena, &dry, &wet_star, n, 2, false).unwrap();
        let jac = build_stage_target_jacobian(&arena, false, &dry, n);
        let solved = solve_point_stage_targets(&arena, &dry, &wet_gain, 0, &jac);
        assert!(solved.iter().all(|v| v.is_finite() && v.abs() <= 2.5));
    }

    #[test]
    fn test_kautz_formant_and_radial_q_target_cols_active() {
        use dnam_engine::stages::V3MatryoshkaModel;
        use dnam_engine::stages::weights::pack_v3_matryoshka_to_arena;

        let mut model = V3MatryoshkaModel::identity(false, true);
        model.kautz_poles[0] = (0.92, 0.35);
        for ro in &mut model.readouts {
            ro.pre_cab_u_gain = [1.0, 1.0];
            ro.kautz_c_l[0] = (0.45, -0.20);
            ro.kautz_c_r[0] = (0.45, -0.20);
        }
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        pack_v3_matryoshka_to_arena(&model, &mut arena);

        let n = 512_usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let s = (0.25 * (TAU * 220.0 * t).sin() + 0.15 * (TAU * 1_650.0 * t).cos()) as f32;
                [s, s]
            })
            .collect();
        let jac = build_stage_target_jacobian(&arena, true, &dry, n);
        assert!(jac.active_cols[8], "Kautz formant column 8 must be active");
        assert!(
            jac.active_cols[15],
            "Kautz radial-Q damping column 15 must be active"
        );

        let in_l: Vec<f32> = dry.iter().map(|lr| lr[0]).collect();
        let cold = unpack_v3_tier_from_arena(&arena, Tier::Ultra, true);
        let mut expected_targets = [0.0_f32; 16];
        expected_targets[15] = 0.45;
        let (wet_l, _) = render_with_stage_targets(&cold, &in_l, &in_l, &expected_targets);
        let wet_lr: Vec<[f32; 2]> = wet_l.iter().map(|&v| [v, v]).collect();
        let solved = solve_point_stage_targets(&arena, &dry, &wet_lr, 0, &jac);
        assert!(
            solved[15].abs() > 0.05,
            "expected non-zero radial-Q perturbation on target[15], got {}",
            solved[15]
        );
    }
}
