//! Smooth differentiable hoisting of `32`-slot `stage_targets` + `stage_latent_*` tables onto
//! [`SonaModelF64`] and exact reverse-mode chain-rule pullback from [`Pass2ParamGrads`], mirroring
//! `dnam-vm`'s `3 kHz` modulation laws (`ref[dsp.ctrl.dispatch.modulation-laws]`,
//! `ref[train.pipeline.control-projection-fit]`).

use core::f64::consts::{LN_2, PI};

use dnam_kernels::{
    MIN_TPT_G, OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, TANK_MAX_G_LIN_F32,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::{SONA_CORE_QUAD_BLOCK_AND_LOCAL, Tier};

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;
use crate::solver::model::{
    NUM_CORE_QUADS_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, SonaModelF64, SonaStageLatentModsF64,
};
use crate::solver::pass2_opt::eval::Pass2ParamGrads;
use crate::solver::pass2_opt::is_op_t_crossbar_coupled;

const STAGE_LATENT_GLOBAL_MAX_F64: f64 = 8.0;
const STAGE_LATENT_ADAA_MAX_F64: f64 = 1.5;
const STAGE_LATENT_LRU_MAX_F64: f64 = 2.0;
const STAGE_LATENT_CORE_MAX_F64: f64 = 4.0;
const KAUTZ_R_MIN_F64: f64 = 0.25;
const KAUTZ_R_MAX_F64: f64 = 0.9995;
const KAUTZ_THETA_MIN_F64: f64 = 1.0e-4;
const KAUTZ_THETA_MAX_F64: f64 = PI - 1.0e-4;
const FREEZE_MAX_F64: f64 = 0.999;
const DAMP_MAX_F64: f64 = 0.9995;

#[inline]
fn clean_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
fn active_opt_quads(q_count: usize) -> usize {
    match q_count {
        0..=4 => 1,
        5..=6 => 2,
        7..=8 => 3,
        _ => 4,
    }
}

/// Forward cache from [`hoist_stage_targets_onto_model`] pairing the modulated [`SonaModelF64`]
/// with open-interval chain-rule multipliers required by [`pullback_stage_targets_and_latent_grads`].
#[derive(Clone, Debug)]
pub struct HoistedStageTargets {
    /// Modulated host-rate `SonaModelF64` ready for `eval_model_pass2_audio` (`q_count = tier.simd_groups()`).
    pub model: SonaModelF64,
    /// Active tier used for latent dimension (`4` vs `8`) and `CoreQuad` count (`4/6/8/10`).
    pub tier: Tier,
    pub(super) global_open: [bool; 5],
    pub(super) dsvf_g_dg_oct: [[f64; 4]; NUM_CORE_QUADS_ULTRA],
    pub(super) svf_k_open: [[bool; 4]; NUM_CORE_QUADS_ULTRA],
    pub(super) wf_open: [[bool; 4]; NUM_CORE_QUADS_ULTRA],
    pub(super) core_adaa_deriv: [[f64; 4]; NUM_CORE_QUADS_ULTRA],
    pub(super) core_lru_sign: [[f64; 4]; NUM_CORE_QUADS_ULTRA],
    pub(super) block_adaa_open: [bool; 4],
    pub(super) block_lru_open: [bool; 4],
    pub(super) quad_adaa_open: [bool; NUM_CORE_QUADS_ULTRA],
    pub(super) quad_lru_open: [bool; NUM_CORE_QUADS_ULTRA],
    pub(super) kautz_dr_dkdamp: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    pub(super) kautz_dbg_dr_eff: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    pub(super) kautz_dth_dkoct: [f64; NUM_KAUTZ_PAIRS_ULTRA],
    pub(super) wet_scale: f64,
    pub(super) wet_scale_open: bool,
    pub(super) d_delay_dt12: [[f64; 4]; 4],
    pub(super) d_delay_dt17: [[f64; 4]; 4],
    pub(super) g_loop_open: [bool; 4],
    pub(super) fb_psi_open: [bool; 4],
    pub(super) freeze_open: [bool; 4],
    pub(super) ramp_rate_open: [bool; 4],
    pub(super) lfo_rate_open: [bool; 4],
    pub(super) d_lfo_depth_dt20: [[f64; 4]; 4],
    pub(super) damp_pole_open: [bool; 4],
    pub(super) tank_dry_mix_open: bool,
    pub(super) d_inj_dt23: [[f64; 4]; 4],
    pub(super) window_depth_open: [bool; 4],
    pub(super) allpass_g_open: [[bool; 4]; 4],
    pub(super) gamma_bc_open: bool,
    pub(super) zoh_hold_open: bool,
    pub(super) beta_q_open: bool,
    pub(super) d_phasor_dt29: [f64; 4],
    pub(super) vca_bias_open: bool,
}

/// Analytical objective gradients wrt `stage_targets[0..32]` and [`SonaStageLatentModsF64`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageTargetAndLatentGrads {
    /// Gradient wrt `stage_targets[0..32]` (`[31]` always `0.0`).
    pub d_stage_targets: [f64; STAGE_TARGET_SLOTS],
    /// Gradient wrt [`SonaStageLatentModsF64`] sensitivity tables.
    pub d_latent_mods: SonaStageLatentModsF64,
}

#[inline]
fn eval_global_dot(mods: &[[f32; 8]; 5], g: usize, z: &[f64; 8], dims: usize) -> f64 {
    let mut sum = 0.0_f64;
    for k in 0..dims.min(8) {
        let w = clean_f64(f64::from(mods[g][k]))
            .clamp(-STAGE_LATENT_GLOBAL_MAX_F64, STAGE_LATENT_GLOBAL_MAX_F64);
        sum += w * z[k];
    }
    sum
}

/// Applies all `31` active `stage_targets[0..31]` slots and `stage_latent_*` sensitivity tables
/// onto a clone of `base` in `tier`, caching exact chain-rule multipliers for pullback.
#[must_use]
pub fn hoist_stage_targets_onto_model(
    base: &SonaModelF64,
    stage_targets: &[f64; STAGE_TARGET_SLOTS],
    z_stage: &[f64; 8],
    tier: Tier,
) -> HoistedStageTargets {
    let mut model = base.clone();
    let mut t = [0.0_f64; STAGE_TARGET_SLOTS];
    for (dst, &v) in t[..STAGE_TARGET_SLOTS - 1]
        .iter_mut()
        .zip(&stage_targets[..STAGE_TARGET_SLOTS - 1])
    {
        *dst = clean_f64(v);
    }
    let z: [f64; 8] = core::array::from_fn(|k| clean_f64(z_stage[k]));
    let is_ultra = matches!(tier, Tier::Ultra);
    let dims = tier.stage_latent_dim();
    let q_count = tier.simd_groups().clamp(1, NUM_CORE_QUADS_ULTRA);
    let active_opt = active_opt_quads(q_count);
    let mods = &base.stage_latent_mods;

    let g_oct_raw = t[0] + eval_global_dot(&mods.global_stage_mods, 0, &z, dims);
    let k_del_raw = t[1] + eval_global_dot(&mods.global_stage_mods, 1, &z, dims);
    let k_oct_raw = t[10] + eval_global_dot(&mods.global_stage_mods, 2, &z, dims);
    let k_dmp_raw = t[11] + eval_global_dot(&mods.global_stage_mods, 3, &z, dims);
    let psi_d_raw = t[14] + eval_global_dot(&mods.global_stage_mods, 4, &z, dims);

    let g_oct_glob = g_oct_raw.clamp(-3.0, 3.0);
    let k_del_glob = k_del_raw.clamp(-0.8, 1.5);
    let k_oct = k_oct_raw.clamp(-1.5, 1.5);
    let k_damp = k_dmp_raw.clamp(-1.5, 1.5);
    let psi_delta = psi_d_raw.clamp(-PI, PI);
    let global_open = [
        g_oct_raw > -3.0 && g_oct_raw < 3.0,
        k_del_raw > -0.8 && k_del_raw < 1.5,
        k_oct_raw > -1.5 && k_oct_raw < 1.5,
        k_dmp_raw > -1.5 && k_dmp_raw < 1.5,
        psi_d_raw > -PI && psi_d_raw < PI,
    ];

    let mut quad_adaa_scale = [1.0_f64; NUM_CORE_QUADS_ULTRA];
    let mut quad_lru_bias = [0.0_f64; NUM_CORE_QUADS_ULTRA];
    let mut block_adaa_open = [false; 4];
    let mut block_lru_open = [false; 4];
    let mut quad_adaa_open = [false; NUM_CORE_QUADS_ULTRA];
    let mut quad_lru_open = [false; NUM_CORE_QUADS_ULTRA];

    if !is_ultra {
        let mut blk_adaa = [1.0_f64; 4];
        let mut blk_lru = [0.0_f64; 4];
        for b in 0..4 {
            let mut sum_a = 0.0_f64;
            let mut sum_l = 0.0_f64;
            for k in 0..4 {
                let wa = clean_f64(f64::from(mods.block_adaa_scale_lite[k][b]))
                    .clamp(-STAGE_LATENT_ADAA_MAX_F64, STAGE_LATENT_ADAA_MAX_F64);
                let wl = clean_f64(f64::from(mods.block_lru_bias_lite[k][b]))
                    .clamp(-STAGE_LATENT_LRU_MAX_F64, STAGE_LATENT_LRU_MAX_F64);
                sum_a += wa * z[k];
                sum_l += wl * z[k];
            }
            let th_a_raw = t[2 + b] + sum_a;
            let th_l_raw = t[6 + b] + sum_l;
            block_adaa_open[b] = th_a_raw > -1.5 && th_a_raw < 1.5;
            block_lru_open[b] = th_l_raw > -2.0 && th_l_raw < 2.0;
            blk_adaa[b] = th_a_raw.clamp(-1.5, 1.5).exp2();
            blk_lru[b] = th_l_raw.clamp(-2.0, 2.0);
        }
        for (q, &(blk, _)) in SONA_CORE_QUAD_BLOCK_AND_LOCAL.iter().enumerate() {
            quad_adaa_scale[q] = blk_adaa[blk];
            quad_lru_bias[q] = blk_lru[blk];
            quad_adaa_open[q] = block_adaa_open[blk];
            quad_lru_open[q] = block_lru_open[blk];
        }
    } else {
        for (q, &(blk, _)) in SONA_CORE_QUAD_BLOCK_AND_LOCAL.iter().enumerate() {
            let mut sum_a = 0.0_f64;
            let mut sum_l = 0.0_f64;
            for k in 0..8 {
                let wa = clean_f64(f64::from(mods.group_adaa_scale_ultra[k][q]))
                    .clamp(-STAGE_LATENT_ADAA_MAX_F64, STAGE_LATENT_ADAA_MAX_F64);
                let wl = clean_f64(f64::from(mods.group_lru_bias_ultra[k][q]))
                    .clamp(-STAGE_LATENT_LRU_MAX_F64, STAGE_LATENT_LRU_MAX_F64);
                sum_a += wa * z[k];
                sum_l += wl * z[k];
            }
            let th_a_raw = t[2 + blk] + sum_a;
            let th_l_raw = t[6 + blk] + sum_l;
            quad_adaa_open[q] = th_a_raw > -1.5 && th_a_raw < 1.5;
            quad_lru_open[q] = th_l_raw > -2.0 && th_l_raw < 2.0;
            quad_adaa_scale[q] = th_a_raw.clamp(-1.5, 1.5).exp2();
            quad_lru_bias[q] = th_l_raw.clamp(-2.0, 2.0);
        }
    }

    let z_nonzero = z.iter().any(|&v| v != 0.0);
    let core_rows = mods.core_mods_ultra.as_deref().filter(|_| is_ultra);
    let mut dsvf_g_dg_oct = [[0.0_f64; 4]; NUM_CORE_QUADS_ULTRA];
    let mut svf_k_open = [[false; 4]; NUM_CORE_QUADS_ULTRA];
    let mut wf_open = [[false; 4]; NUM_CORE_QUADS_ULTRA];
    let mut core_adaa_deriv = [[0.0_f64; 4]; NUM_CORE_QUADS_ULTRA];
    let mut core_lru_sign = [[0.0_f64; 4]; NUM_CORE_QUADS_ULTRA];

    let g_min = f64::from(MIN_TPT_G);
    for q in 0..q_count {
        let bq = &base.core_quads[q];
        let mq = &mut model.core_quads[q];
        for l in 0..4 {
            let lane_idx = q * 4 + l;
            let (mut dg, mut dk, mut dw) = (0.0_f64, 0.0_f64, 0.0_f64);
            if let Some(rows) = core_rows
                && z_nonzero
            {
                for k in 0..8 {
                    let zk = z[k];
                    if zk != 0.0 {
                        let w_dg = clean_f64(f64::from(rows[k][lane_idx]))
                            .clamp(-STAGE_LATENT_CORE_MAX_F64, STAGE_LATENT_CORE_MAX_F64);
                        let w_dk = clean_f64(f64::from(rows[k][40 + lane_idx]))
                            .clamp(-STAGE_LATENT_CORE_MAX_F64, STAGE_LATENT_CORE_MAX_F64);
                        let w_dw = clean_f64(f64::from(rows[k][80 + lane_idx]))
                            .clamp(-STAGE_LATENT_CORE_MAX_F64, STAGE_LATENT_CORE_MAX_F64);
                        dg += w_dg * zk;
                        dk += w_dk * zk;
                        dw += w_dw * zk;
                    }
                }
            }
            let g_oct_l_raw = g_oct_glob + dg;
            let g_oct_l = g_oct_l_raw.clamp(-3.0, 3.0);
            let g_scale_l = g_oct_l.exp2();
            let base_g = clean_f64(bq.svf_g[l]).clamp(g_min, 64.0);
            let g_raw = base_g * g_scale_l;
            mq.svf_g[l] = g_raw.clamp(g_min, 64.0);
            if g_oct_l_raw > -3.0 && g_oct_l_raw < 3.0 && g_raw > g_min && g_raw < 64.0 {
                dsvf_g_dg_oct[q][l] = base_g * g_scale_l * LN_2;
            }

            let base_k = clean_f64(bq.svf_k[l]).clamp(0.0, 8.0);
            let k_raw = base_k + k_del_glob + dk;
            mq.svf_k[l] = k_raw.clamp(0.0, 8.0);
            svf_k_open[q][l] = k_raw > 0.0 && k_raw < 8.0;

            let base_wf = clean_f64(bq.wavefolder_mix[l]).clamp(0.0, 1.0);
            let wf_raw = base_wf + dw;
            mq.wavefolder_mix[l] = wf_raw.clamp(0.0, 1.0);
            wf_open[q][l] = wf_raw > 0.0 && wf_raw < 1.0;

            let base_adaa = clean_f64(bq.adaa_drive[l]).clamp(-32.0, 32.0);
            let adaa_raw = base_adaa * quad_adaa_scale[q];
            mq.adaa_drive[l] = adaa_raw.clamp(-32.0, 32.0);
            if quad_adaa_open[q] && adaa_raw.abs() < 32.0 {
                core_adaa_deriv[q][l] = base_adaa * quad_adaa_scale[q] * LN_2;
            }

            let rect_s = if bq.rect_sign[l].is_finite() && bq.rect_sign[l] < 0.0 {
                -1.0
            } else {
                1.0
            };
            let base_bias = clean_f64(bq.lru_bias_drive[l]).clamp(-16.0, 16.0);
            let bias_raw = base_bias + rect_s * quad_lru_bias[q];
            mq.lru_bias_drive[l] = bias_raw.clamp(-16.0, 16.0);
            if quad_lru_open[q] && bias_raw.abs() < 16.0 {
                core_lru_sign[q][l] = rect_s;
            }
        }
    }

    let active_k = base.active_kautz_pairs.min(NUM_KAUTZ_PAIRS_ULTRA);
    let freq_scale_k = k_oct.exp2();
    let damp_scale_k = k_damp.exp2();
    let mut kautz_dr_dkdamp = [0.0_f64; NUM_KAUTZ_PAIRS_ULTRA];
    let mut kautz_dbg_dr_eff = [0.0_f64; NUM_KAUTZ_PAIRS_ULTRA];
    let mut kautz_dth_dkoct = [0.0_f64; NUM_KAUTZ_PAIRS_ULTRA];
    for m in 0..active_k {
        let bk = &base.kautz[m];
        let dr_pos = if bk.dr.is_finite() && bk.dr > 0.0 {
            bk.dr
        } else {
            0.0
        };
        let r_sum = if bk.r_low.is_finite() {
            bk.r_low + 0.5 * dr_pos
        } else {
            0.0
        };
        if r_sum <= 0.0 || !bk.b_gamma.is_finite() || bk.b_gamma <= 0.0 {
            continue;
        }
        let r_host = r_sum.clamp(KAUTZ_R_MIN_F64, KAUTZ_R_MAX_F64);
        let th_host = clean_f64(bk.theta).clamp(0.0, KAUTZ_THETA_MAX_F64);
        let r_damped = 1.0 - (1.0 - r_host) * damp_scale_k;
        let r_eff = r_damped.clamp(KAUTZ_R_MIN_F64, KAUTZ_R_MAX_F64);
        let r_open = r_damped > KAUTZ_R_MIN_F64 && r_damped < KAUTZ_R_MAX_F64;
        if global_open[3] && r_open {
            kautz_dr_dkdamp[m] = -(1.0 - r_host) * damp_scale_k * LN_2;
        }

        let inv_denom = 1.0 / (1.0 - r_host * r_host).max(1.0e-7);
        let omr2 = 1.0 - r_eff * r_eff;
        let ratio = (omr2.max(1.0e-7) * inv_denom).sqrt();
        let mk = &mut model.kautz[m];
        mk.r_low = r_eff;
        mk.dr = 0.0;
        mk.b_gamma = bk.b_gamma * ratio;
        if r_open && omr2 > 1.0e-7 && ratio > 0.0 {
            kautz_dbg_dr_eff[m] = bk.b_gamma * (-r_eff * inv_denom / ratio);
        }

        if th_host > 1.0e-6 {
            let th_unclamped = th_host * freq_scale_k;
            mk.theta = th_unclamped.clamp(KAUTZ_THETA_MIN_F64, KAUTZ_THETA_MAX_F64);
            if global_open[2]
                && th_unclamped > KAUTZ_THETA_MIN_F64
                && th_unclamped < KAUTZ_THETA_MAX_F64
            {
                kautz_dth_dkoct[m] = th_unclamped * LN_2;
            }
        }
    }

    // Slot 15 (`crossbar_wet_mod`) scales Op-P, Op-T crossbar gains, carrier basis, and Op-T wet output.
    let wet_raw = 1.0 + t[15].clamp(-1.0, 1.0);
    let wet_scale = wet_raw.clamp(0.0, 2.0);
    let wet_scale_open = t[15] > -1.0 && t[15] < 1.0 && wet_raw > 0.0 && wet_raw < 2.0;
    for (dst, &b_g) in model
        .crossbar_op_gains
        .iter_mut()
        .zip(&base.crossbar_op_gains)
    {
        *dst = (clean_f64(b_g) * wet_scale).clamp(-4.0, 4.0);
    }
    for i in 0..3 {
        for j in 0..4 {
            model.carrier_basis_w[i][j] =
                (clean_f64(base.carrier_basis_w[i][j]) * wet_scale).clamp(-8.0, 8.0);
        }
    }
    let base_op_t_wet = clean_f64(base.op_t_wet_scale).clamp(0.0, 2.0);
    model.op_t_wet_scale = (base_op_t_wet * wet_scale).clamp(0.0, 2.0);

    // Slots 12..=25 (`Op-T` + stage dry mix slot 22).
    let coupled = is_op_t_crossbar_coupled(&model, q_count);
    let delay_floor = if coupled {
        MIN_DELAY_CROSSBAR_SAMP
    } else {
        4.0
    };
    let delay_scale_oct = t[17].clamp(-3.0, 3.0);
    let delay_scale = delay_scale_oct.exp2();
    let delay_scale_open = t[17] > -3.0 && t[17] < 3.0;
    let lfo_scale_oct = t[20].clamp(-4.0, 4.0);
    let lfo_scale = lfo_scale_oct.exp2();
    let lfo_scale_open = t[20] > -4.0 && t[20] < 4.0;
    let inj_scale_oct = t[23].clamp(-4.0, 4.0);
    let inj_scale = inj_scale_oct.exp2();
    let inj_scale_open = t[23] > -4.0 && t[23] < 4.0;

    let mut d_delay_dt12 = [[0.0_f64; 4]; 4];
    let mut d_delay_dt17 = [[0.0_f64; 4]; 4];
    let mut g_loop_open = [false; 4];
    let mut fb_psi_open = [false; 4];
    let mut freeze_open = [false; 4];
    let mut ramp_rate_open = [false; 4];
    let mut lfo_rate_open = [false; 4];
    let mut d_lfo_depth_dt20 = [[0.0_f64; 4]; 4];
    let mut damp_pole_open = [false; 4];
    let mut d_inj_dt23 = [[0.0_f64; 4]; 4];
    let mut window_depth_open = [false; 4];
    let mut allpass_g_open = [[false; 4]; 4];

    for q in 0..active_opt {
        let bt = &base.transport_quads[q];
        let mt = &mut model.transport_quads[q];
        for l in 0..4 {
            let d_base = clean_f64(bt.delay_samples[l]).clamp(delay_floor, 48_000.0);
            let d_scaled = d_base * delay_scale;
            let d_raw = d_scaled + t[12];
            mt.delay_samples[l] = d_raw.clamp(delay_floor, 48_000.0);
            if d_raw > delay_floor && d_raw < 48_000.0 {
                d_delay_dt12[q][l] = 1.0;
                if delay_scale_open {
                    d_delay_dt17[q][l] = d_scaled * LN_2;
                }
            }

            let ld_base = clean_f64(bt.lfo_depths[l]).clamp(0.0, 512.0);
            let ld_raw = ld_base * lfo_scale;
            mt.lfo_depths[l] = ld_raw.clamp(0.0, 512.0);
            if lfo_scale_open && ld_raw > 0.0 && ld_raw < 512.0 {
                d_lfo_depth_dt20[q][l] = ld_raw * LN_2;
            }

            let ig_base = clean_f64(bt.input_gains[l]).clamp(-2.0, 2.0);
            let ig_raw = ig_base * inj_scale;
            mt.input_gains[l] = ig_raw.clamp(-2.0, 2.0);
            if inj_scale_open && ig_raw.abs() < 2.0 {
                d_inj_dt23[q][l] = ig_raw * LN_2;
            }

            let ap_raw = clean_f64(bt.allpass_g[l]).clamp(-0.75, 0.75) + t[25];
            mt.allpass_g[l] = ap_raw.clamp(-0.75, 0.75);
            allpass_g_open[q][l] = ap_raw.abs() < 0.75;
        }

        let g_max = f64::from(TANK_MAX_G_LIN_F32);
        let g_raw = clean_f64(bt.g_loop).clamp(0.0, g_max) + t[13];
        mt.g_loop = g_raw.clamp(0.0, g_max);
        g_loop_open[q] = g_raw > 0.0 && g_raw < g_max;

        let psi_raw = clean_f64(bt.fb_psi).clamp(-PI, PI) + psi_delta;
        mt.fb_psi = psi_raw.clamp(-PI, PI);
        fb_psi_open[q] = global_open[4] && psi_raw.abs() < PI;

        let frz_raw = clean_f64(bt.freeze_mix).clamp(0.0, 1.0) + t[16];
        mt.freeze_mix = frz_raw.clamp(0.0, FREEZE_MAX_F64);
        freeze_open[q] = frz_raw > 0.0 && frz_raw < FREEZE_MAX_F64;

        let rr_raw = clean_f64(bt.ramp_rate_hz).clamp(-16.0, 16.0) + t[18];
        mt.ramp_rate_hz = rr_raw.clamp(-16.0, 16.0);
        ramp_rate_open[q] = rr_raw.abs() < 16.0;

        let lr_raw = clean_f64(bt.lfo_rate_hz).clamp(0.0, 16.0) + t[19];
        mt.lfo_rate_hz = lr_raw.clamp(0.0, 16.0);
        lfo_rate_open[q] = lr_raw > 0.0 && lr_raw < 16.0;

        let dp_raw = clean_f64(bt.damp_pole).clamp(0.0, DAMP_MAX_F64) + t[21];
        mt.damp_pole = dp_raw.clamp(0.0, DAMP_MAX_F64);
        damp_pole_open[q] = dp_raw > 1.0e-7 && dp_raw < DAMP_MAX_F64;

        let wd_raw = clean_f64(bt.window_depth).clamp(0.0, 1.0) + t[24];
        mt.window_depth = wd_raw.clamp(0.0, 1.0);
        window_depth_open[q] = wd_raw > 0.0 && wd_raw < 1.0;
    }

    let dry_raw = clean_f64(base.tank_dry_mix).clamp(-2.0, 2.0) + t[22];
    model.tank_dry_mix = dry_raw.clamp(-2.0, 2.0);
    let tank_dry_mix_open = dry_raw.abs() < 2.0;

    // Slots 26..=28 (`Op-N[Pre]`).
    let gbc_raw = clean_f64(base.op_n_pre_gamma_bc).clamp(0.0, 1.0) + t[26];
    model.op_n_pre_gamma_bc = gbc_raw.clamp(0.0, 1.0);
    let gamma_bc_open = gbc_raw > 0.0 && gbc_raw < 1.0;

    let zoh_max = f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH);
    let zoh_raw = clean_f64(base.op_n_pre_zoh_hold_depth).clamp(0.0, zoh_max) + t[27];
    model.op_n_pre_zoh_hold_depth = zoh_raw.clamp(0.0, zoh_max);
    let zoh_hold_open = zoh_raw > 0.0 && zoh_raw < zoh_max;

    let beta_max = f64::from(OP_N_PRE_MAX_BETA_Q);
    let beta_raw = clean_f64(base.op_n_pre_beta_q).clamp(0.0, beta_max) + t[28];
    model.op_n_pre_beta_q = beta_raw.clamp(0.0, beta_max);
    let beta_q_open = beta_raw > 0.0 && beta_raw < beta_max;

    // Slot 29 (`op_p_phasor_rate_oct`).
    let sr = clean_f64(base.sample_rate_hz).clamp(8_000.0, 192_000.0);
    let ph_lim = 0.49 * sr;
    let ph_oct = t[29].clamp(-4.0, 4.0);
    let ph_scale = ph_oct.exp2();
    let ph_open = t[29] > -4.0 && t[29] < 4.0;
    let mut d_phasor_dt29 = [0.0_f64; 4];
    for l in 0..4 {
        let f_base = clean_f64(base.phasor_freq_hz[l]).clamp(-ph_lim, ph_lim);
        let f_raw = f_base * ph_scale;
        model.phasor_freq_hz[l] = f_raw.clamp(-ph_lim, ph_lim);
        if ph_open && f_raw.abs() < ph_lim {
            d_phasor_dt29[l] = f_raw * LN_2;
        }
    }

    // Slot 30 (`sense_vca_gain_delta`).
    let t30 = t[30].clamp(-1.0, 1.0);
    let vca_raw = clean_f64(base.sense_vca_bias).clamp(-1.0, 1.0) + t30;
    model.sense_vca_bias = vca_raw.clamp(-1.0, 1.0);
    let vca_bias_open = t[30] > -1.0 && t[30] < 1.0 && vca_raw > -1.0 && vca_raw < 1.0;

    HoistedStageTargets {
        model,
        tier,
        global_open,
        dsvf_g_dg_oct,
        svf_k_open,
        wf_open,
        core_adaa_deriv,
        core_lru_sign,
        block_adaa_open,
        block_lru_open,
        quad_adaa_open,
        quad_lru_open,
        kautz_dr_dkdamp,
        kautz_dbg_dr_eff,
        kautz_dth_dkoct,
        wet_scale,
        wet_scale_open,
        d_delay_dt12,
        d_delay_dt17,
        g_loop_open,
        fb_psi_open,
        freeze_open,
        ramp_rate_open,
        lfo_rate_open,
        d_lfo_depth_dt20,
        damp_pole_open,
        tank_dry_mix_open,
        d_inj_dt23,
        window_depth_open,
        allpass_g_open,
        gamma_bc_open,
        zoh_hold_open,
        beta_q_open,
        d_phasor_dt29,
        vca_bias_open,
    }
}

/// Pulls back reverse-mode [`Pass2ParamGrads`] through the exact [`HoistedStageTargets`] map into
/// gradients wrt `stage_targets[0..31]` (`[31] == 0.0`) and [`SonaStageLatentModsF64`].
#[must_use]
pub(crate) fn pullback_stage_targets_and_latent_grads(
    base: &SonaModelF64,
    hoisted: &HoistedStageTargets,
    z_stage: &[f64; 8],
    grads: &Pass2ParamGrads,
) -> StageTargetAndLatentGrads {
    let mut out = StageTargetAndLatentGrads::default();
    let z: [f64; 8] = core::array::from_fn(|k| clean_f64(z_stage[k]));
    let is_ultra = matches!(hoisted.tier, Tier::Ultra);
    let dims = hoisted.tier.stage_latent_dim().min(8);
    let q_count = hoisted.tier.simd_groups().min(grads.core_quads.len());
    let active_opt = active_opt_quads(q_count);

    let mut core_rows_grad = if is_ultra && base.stage_latent_mods.core_mods_ultra.is_some() {
        Some(Box::new([[0.0_f32; 120]; 8]))
    } else {
        None
    };

    let mut d_g_oct_global = 0.0_f64;
    let mut d_k_del_global = 0.0_f64;

    for q in 0..q_count {
        let gq = &grads.core_quads[q];
        let blk = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0;
        let (mut d_th_a_q, mut d_th_l_q) = (0.0_f64, 0.0_f64);
        for l in 0..4 {
            let d_g_oct_l = gq.svf_g[l] * hoisted.dsvf_g_dg_oct[q][l];
            d_g_oct_global += d_g_oct_l;

            let d_k_l = if hoisted.svf_k_open[q][l] {
                gq.svf_k[l]
            } else {
                0.0
            };
            d_k_del_global += d_k_l;

            let d_w_l = if hoisted.wf_open[q][l] {
                gq.wavefolder_mix[l]
            } else {
                0.0
            };

            if let Some(ref mut rows) = core_rows_grad {
                let lane_idx = q * 4 + l;
                for k in 0..8 {
                    let zk = z[k];
                    if zk != 0.0 {
                        rows[k][lane_idx] += (d_g_oct_l * zk) as f32;
                        rows[k][40 + lane_idx] += (d_k_l * zk) as f32;
                        rows[k][80 + lane_idx] += (d_w_l * zk) as f32;
                    }
                }
            }

            d_th_a_q += gq.adaa_drive[l] * hoisted.core_adaa_deriv[q][l];
            d_th_l_q += gq.lru_bias_drive[l] * hoisted.core_lru_sign[q][l];
        }

        if hoisted.quad_adaa_open[q] {
            out.d_stage_targets[2 + blk] += d_th_a_q;
            if is_ultra {
                for k in 0..8 {
                    out.d_latent_mods.group_adaa_scale_ultra[k][q] += (d_th_a_q * z[k]) as f32;
                }
            } else if hoisted.block_adaa_open[blk] {
                for k in 0..4 {
                    out.d_latent_mods.block_adaa_scale_lite[k][blk] += (d_th_a_q * z[k]) as f32;
                }
            }
        }
        if hoisted.quad_lru_open[q] {
            out.d_stage_targets[6 + blk] += d_th_l_q;
            if is_ultra {
                for k in 0..8 {
                    out.d_latent_mods.group_lru_bias_ultra[k][q] += (d_th_l_q * z[k]) as f32;
                }
            } else if hoisted.block_lru_open[blk] {
                for k in 0..4 {
                    out.d_latent_mods.block_lru_bias_lite[k][blk] += (d_th_l_q * z[k]) as f32;
                }
            }
        }
    }
    out.d_latent_mods.core_mods_ultra = core_rows_grad;

    if hoisted.global_open[0] {
        out.d_stage_targets[0] += d_g_oct_global;
        for k in 0..dims {
            out.d_latent_mods.global_stage_mods[0][k] += (d_g_oct_global * z[k]) as f32;
        }
    }
    if hoisted.global_open[1] {
        out.d_stage_targets[1] += d_k_del_global;
        for k in 0..dims {
            out.d_latent_mods.global_stage_mods[1][k] += (d_k_del_global * z[k]) as f32;
        }
    }

    // Kautz slots 10..=11.
    let (mut d_k_oct, mut d_k_damp) = (0.0_f64, 0.0_f64);
    for (m, dk) in grads.d_kautz.iter().enumerate() {
        if m >= NUM_KAUTZ_PAIRS_ULTRA {
            break;
        }
        d_k_oct += dk[1] * hoisted.kautz_dth_dkoct[m];
        let d_r_tot = dk[0] + dk[2] * hoisted.kautz_dbg_dr_eff[m];
        d_k_damp += d_r_tot * hoisted.kautz_dr_dkdamp[m];
    }
    if hoisted.global_open[2] {
        out.d_stage_targets[10] += d_k_oct;
        for k in 0..dims {
            out.d_latent_mods.global_stage_mods[2][k] += (d_k_oct * z[k]) as f32;
        }
    }
    if hoisted.global_open[3] {
        out.d_stage_targets[11] += d_k_damp;
        for k in 0..dims {
            out.d_latent_mods.global_stage_mods[3][k] += (d_k_damp * z[k]) as f32;
        }
    }

    // Slot 15 (`crossbar_wet_mod`).
    if hoisted.wet_scale_open {
        let mut d_wet = 0.0_f64;
        for c in 0..2 {
            let b_g = clean_f64(base.crossbar_op_gains[c]);
            if (b_g * hoisted.wet_scale).abs() < 4.0 {
                d_wet += grads.crossbar.d_crossbar_op_gains[c] * b_g;
            }
        }
        for i in 0..3 {
            for j in 0..4 {
                let b_w = clean_f64(base.carrier_basis_w[i][j]);
                if (b_w * hoisted.wet_scale).abs() < 8.0 {
                    d_wet += grads.crossbar.d_carrier_basis_w[i][j] * b_w;
                }
            }
        }
        let b_op_t_wet = clean_f64(base.op_t_wet_scale).clamp(0.0, 2.0);
        if b_op_t_wet * hoisted.wet_scale < 2.0 {
            d_wet += grads.d_op_t_wet_scale * b_op_t_wet;
        }
        out.d_stage_targets[15] += d_wet;
    }

    // Op-T slots 12..=25 (except 15, 22).
    let mut d_psi_tot = 0.0_f64;
    for q in 0..active_opt {
        let gt = &grads.op_t[q];
        for l in 0..4 {
            out.d_stage_targets[12] += gt.d_delay_samples[l] * hoisted.d_delay_dt12[q][l];
            out.d_stage_targets[17] += gt.d_delay_samples[l] * hoisted.d_delay_dt17[q][l];
            out.d_stage_targets[20] += gt.d_lfo_depths[l] * hoisted.d_lfo_depth_dt20[q][l];
            out.d_stage_targets[23] += gt.d_input_gains[l] * hoisted.d_inj_dt23[q][l];
            if hoisted.allpass_g_open[q][l] {
                out.d_stage_targets[25] += gt.d_allpass_g[l];
            }
        }
        if hoisted.g_loop_open[q] {
            out.d_stage_targets[13] += gt.d_g_loop;
        }
        if hoisted.fb_psi_open[q] {
            d_psi_tot += gt.d_fb_psi;
        }
        if hoisted.freeze_open[q] {
            out.d_stage_targets[16] += gt.d_freeze_mix;
        }
        if hoisted.ramp_rate_open[q] {
            out.d_stage_targets[18] += gt.d_ramp_rate_hz;
        }
        if hoisted.lfo_rate_open[q] {
            out.d_stage_targets[19] += gt.d_lfo_rate_hz;
        }
        if hoisted.damp_pole_open[q] {
            out.d_stage_targets[21] += gt.d_damp_pole;
        }
        if hoisted.window_depth_open[q] {
            out.d_stage_targets[24] += gt.d_window_depth;
        }
    }
    if hoisted.global_open[4] {
        out.d_stage_targets[14] += d_psi_tot;
        for k in 0..dims {
            out.d_latent_mods.global_stage_mods[4][k] += (d_psi_tot * z[k]) as f32;
        }
    }

    if hoisted.tank_dry_mix_open {
        out.d_stage_targets[22] += grads.d_tank_dry_mix;
    }
    if hoisted.gamma_bc_open {
        out.d_stage_targets[26] += grads.op_n_pre.d_gamma_bc;
    }
    if hoisted.zoh_hold_open {
        out.d_stage_targets[27] += grads.op_n_pre.d_zoh_hold_depth;
    }
    if hoisted.beta_q_open {
        out.d_stage_targets[28] += grads.op_n_pre.d_beta_q;
    }
    for l in 0..4 {
        out.d_stage_targets[29] += grads.crossbar.d_phasor_freq_hz[l] * hoisted.d_phasor_dt29[l];
    }
    if hoisted.vca_bias_open {
        out.d_stage_targets[30] += grads.sensory.d_vca_bias;
    }

    out.d_stage_targets[STAGE_TARGET_SLOTS - 1] = 0.0;
    out
}
