#![forbid(unsafe_code)]
//! Shared `Zero-Lag 2x/1x` + `Stage I Env-SVF` + `Stage II 4-Lane-Group Vectorized GRU` Core & Reverse Block.

use dnam_engine::Tier;
use dnam_engine::stages::stage1_svf::SVF_G_MAX;
use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_GROUPS_PER_BLOCK, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
    tier_block_lanes_array,
};

use super::gabor::GaborAdjointScratch;
use super::ltas_adj::Welch33Scratch;
use super::model_f64::V3MatryoshkaOptF64;
use super::primitives::{
    BLK_TAPE_OFF, ClampedCoreParams, GruCascadeSubTape, InputFeatSubTape, SvfChanTape,
    downsample_2x_fwd, eval_gru_group_nonlinear_fwd, exp2_oct_with_deriv, inv4_pos_f64,
    step_tpt_pair_fwd, tan_prewarp_with_deriv, upsample_2x_fwd,
};

/// Per-tier scratch workspace enabling conflict-free parallel tier readout & adjoint evaluation.
#[derive(Debug, Default)]
pub(crate) struct TierReadoutWorkspace {
    pub(crate) s2_tier: [Vec<f64>; 2],
    pub(crate) y_tier: [Vec<f64>; 2],
    pub(crate) dy_tier: [Vec<f64>; 2],
    pub(crate) ds2_tier: [Vec<f64>; 2],
    pub(crate) err_buf: Vec<f64>,
    pub(crate) bezout_scratch: [Vec<f64>; 3],
    pub(crate) s2_sub_tier: [Vec<f64>; 2],
    pub(crate) dir_sub_tier: [Vec<f64>; 2],
    pub(crate) s2_aa_diff: [Vec<f64>; 2],
    pub(crate) dir_aa_diff: [Vec<f64>; 2],
    pub(crate) kautz_re: [Vec<f64>; 2],
    pub(crate) kautz_im: [Vec<f64>; 2],
    pub(crate) ddir_sub: [Vec<f64>; 2],
    pub(crate) ds2_sub: [Vec<f64>; 2],
    pub(crate) gabor_scratch: GaborAdjointScratch,
    pub(crate) welch_scratch: Welch33Scratch,
}

/// Reusable workspace avoiding heap reallocations across L-BFGS iterations and VarPro calls.
#[derive(Debug, Default)]
pub struct AdjointWorkspace {
    pub(crate) tap_dec: [Vec<[f64; 7]>; 2],
    pub(crate) u_dec: [Vec<f64>; 2],
    pub(crate) hp_dec: [Vec<f64>; 2],
    pub(crate) h_dec: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
    pub(crate) sub_taps: [Vec<[f64; 7]>; 2],
    pub(crate) sub_u: [Vec<f64>; 2],
    pub(crate) sub_hp: [Vec<f64>; 2],
    pub(crate) sub_h: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
    pub(crate) d_sub_taps: [Vec<[f64; 7]>; 2],
    pub(crate) d_sub_u: [Vec<f64>; 2],
    pub(crate) d_sub_hp: [Vec<f64>; 2],
    pub(crate) d_sub_h: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
    pub(crate) svf_sub: [Vec<SvfChanTape>; 2],
    pub(crate) feat_sub: [Vec<InputFeatSubTape>; 2],
    pub(crate) gru_sub: [Vec<GruCascadeSubTape>; 2],
    pub(crate) d_det_sub: [Vec<f64>; 2],
    pub(crate) d_xl_sub: Vec<[f64; 2]>,
    pub(crate) d_xr_sub: Vec<[f64; 2]>,
    pub(crate) tier_ws: [std::sync::Mutex<TierReadoutWorkspace>; 4],
    pub(crate) stem_pool: Vec<std::sync::Mutex<AdjointWorkspace>>,
}

impl AdjointWorkspace {
    pub(crate) fn ensure_stem_pool(&mut self, count: usize) {
        while self.stem_pool.len() < count {
            self.stem_pool
                .push(std::sync::Mutex::new(AdjointWorkspace::default()));
        }
    }

    pub(crate) fn ensure_capacity(
        &mut self,
        n: usize,
        sub_steps: usize,
        active_ch: usize,
        tiers_needed: &[usize; 4],
        need_grad: bool,
        decimate_all_lanes: bool,
    ) {
        let total_sub = n * sub_steps;
        let ch_lim = active_ch.clamp(1, 2);
        for ch in 0..ch_lim {
            if decimate_all_lanes && self.tap_dec[ch].len() < n {
                self.tap_dec[ch].resize(n, [0.0; 7]);
                self.u_dec[ch].resize(n, 0.0);
                self.hp_dec[ch].resize(n, 0.0);
                self.h_dec[ch].resize(n, [0.0; MAX_TOTAL_GRU_LANES]);
            }
            if !decimate_all_lanes && self.sub_taps[ch].len() < total_sub {
                self.sub_taps[ch].resize(total_sub, [0.0; 7]);
                self.sub_u[ch].resize(total_sub, 0.0);
                self.sub_hp[ch].resize(total_sub, 0.0);
                self.sub_h[ch].resize(total_sub, [0.0; MAX_TOTAL_GRU_LANES]);
            }
            if need_grad && self.gru_sub[ch].len() < total_sub {
                self.gru_sub[ch].resize(total_sub, GruCascadeSubTape::default());
                self.d_sub_u[ch].resize(total_sub, 0.0);
                self.d_sub_hp[ch].resize(total_sub, 0.0);
                self.d_sub_h[ch].resize(total_sub, [0.0; MAX_TOTAL_GRU_LANES]);
            }
        }
        if need_grad {
            for ch in 0..2 {
                if self.svf_sub[ch].len() < total_sub {
                    self.svf_sub[ch].resize(total_sub, SvfChanTape::default());
                    self.feat_sub[ch].resize(total_sub, InputFeatSubTape::default());
                    self.d_det_sub[ch].resize(total_sub, 0.0);
                    self.d_sub_taps[ch].resize(total_sub, [0.0; 7]);
                }
            }
            if self.d_xl_sub.len() < n {
                self.d_xl_sub.resize(n, [0.0; 2]);
                self.d_xr_sub.resize(n, [0.0; 2]);
            }
        }
        if !decimate_all_lanes {
            for (tau, &k_t) in tiers_needed.iter().enumerate() {
                if k_t == 0 {
                    continue;
                }
                let mut tws = self.tier_ws[tau]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for ch in 0..ch_lim {
                    if tws.s2_tier[ch].len() < n {
                        tws.s2_tier[ch].resize(n, 0.0);
                        tws.y_tier[ch].resize(n, 0.0);
                    }
                    if need_grad {
                        if tws.s2_sub_tier[ch].len() < total_sub {
                            tws.s2_sub_tier[ch].resize(total_sub, 0.0);
                            tws.dir_sub_tier[ch].resize(total_sub, 0.0);
                        }
                        if tws.ddir_sub[ch].len() < total_sub {
                            tws.ddir_sub[ch].resize(total_sub, 0.0);
                            tws.ds2_sub[ch].resize(total_sub, 0.0);
                        }
                        if tws.dy_tier[ch].len() < n {
                            tws.dy_tier[ch].resize(n, 0.0);
                            tws.ds2_tier[ch].resize(n, 0.0);
                        }
                        if sub_steps == 2 && tws.s2_aa_diff[ch].len() < n {
                            tws.s2_aa_diff[ch].resize(n, 0.0);
                            tws.dir_aa_diff[ch].resize(n, 0.0);
                        }
                        let need_k = n * k_t;
                        if tws.kautz_re[ch].len() < need_k {
                            tws.kautz_re[ch].resize(need_k, 0.0);
                            tws.kautz_im[ch].resize(need_k, 0.0);
                        }
                    }
                }
                if tws.err_buf.len() < n {
                    tws.err_buf.resize(n, 0.0);
                }
            }
        }
    }
}

const ALL_TIERS_MASK: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

pub(crate) fn max_lanes_for_mask(
    tiers_mask: &[usize; 4],
    cap: [usize; NUM_GRU_BLOCKS],
) -> [usize; NUM_GRU_BLOCKS] {
    let mut out = [4_usize; NUM_GRU_BLOCKS];
    for (tau, &tier) in ALL_TIERS_MASK.iter().enumerate() {
        if tiers_mask[tau] > 0 {
            let c_t = tier_block_lanes_array(tier);
            for b in 0..NUM_GRU_BLOCKS {
                out[b] = out[b].max(c_t[b]).min(cap[b]);
            }
        }
    }
    out
}

#[inline(always)]
fn dot4(w: &[f64; MAX_BLOCK_LANES], v: &[f64; MAX_BLOCK_LANES], s: usize) -> f64 {
    (w[s] * v[s] + w[s + 2] * v[s + 2]) + (w[s + 1] * v[s + 1] + w[s + 3] * v[s + 3])
}

/// Evaluates the shared `Zero-Lag 2x/1x` -> `Stage I Env-SVF` -> `Stage II 3-Block GRU` core once per stem.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_shared_core(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    in_l: &[f32],
    in_r: &[f32],
    n: usize,
    c_blk: [usize; NUM_GRU_BLOCKS],
    active_ch: usize,
    record_tape: bool,
    decimate_all_lanes: bool,
    ws: &mut AdjointWorkspace,
) {
    let sub_steps = if m.os2x { 2 } else { 1 };
    let g_cnts: [usize; NUM_GRU_BLOCKS] =
        std::array::from_fn(|b| (c_blk[b] >> 2).clamp(1, MAX_GROUPS_PER_BLOCK[b]));
    let (mut up_raw_prev, mut up_p1, mut up_p2) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut up_hb = [[0.0_f64; 4]; 2];
    let (mut det_lp, mut env) = ([0.0_f64; 2], [0.0_f64; 2]);
    let (mut ic1, mut ic2) = ([[0.0_f64; 2]; 2], [[0.0_f64; 2]; 2]);
    let (mut u_lp, mut env_fast, mut env_slow) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut h_state = [[[0.0_f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS]; 2];
    let (mut dec_tap_p, mut dec_u_p, mut dec_hp_p) =
        ([[0.0_f64; 7]; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut dec_h_p = [[0.0_f64; MAX_TOTAL_GRU_LANES]; 2];
    let (mut dec_tap_aa, mut dec_u_aa, mut dec_hp_aa) =
        ([[[0.0_f64; 4]; 7]; 2], [[0.0_f64; 4]; 2], [[0.0_f64; 4]; 2]);
    let mut dec_h_aa = [[[0.0_f64; 4]; MAX_TOTAL_GRU_LANES]; 2];
    let half_link = 0.5 * cp.link.0;
    let g_hi = f64::from(SVF_G_MAX);
    let static_svf = cp.env_oct.0 == 0.0;
    let w1_static = cp.base_g.0.clamp(cp.g_min.0, cp.g_max.0);
    let g1_static = if static_svf {
        tan_prewarp_with_deriv(w1_static).0.clamp(1.0e-4, g_hi)
    } else {
        0.0
    };
    let g2_static = (g1_static * cp.sec2_r.0).clamp(1.0e-4, g_hi);

    let svf_ch = if !m.mono_out || cp.key_link.0 > 0.0 {
        2
    } else {
        1
    };
    let keep_in_r = svf_ch == 2;
    let m_lp2 = if cp.feat_lp2_mix.0 > 0.0 {
        cp.feat_lp2_mix.0
    } else {
        0.0
    };

    for t in 0..n {
        let xl = f64::from(in_l[t]);
        let xr = if !keep_in_r { xl } else { f64::from(in_r[t]) };
        let (xs_sub, d2_lr, uz_diff_lr, dxp_diff_lr, up_hb_diff_lr) = if m.os2x {
            let (l_pair, d2l, uzl, dxpl, hbl) = upsample_2x_fwd(
                xl,
                cp.up_quad.0,
                cp.up_zoh_mix.0,
                cp.up_pre_emph.0,
                cp.up_aa_mix.0,
                &mut up_raw_prev[0],
                &mut up_p1[0],
                &mut up_p2[0],
                &mut up_hb[0],
                record_tape,
            );
            let (r_pair, d2r, uzr, dxpr, hbr) = if !keep_in_r {
                (l_pair, d2l, uzl, dxpl, hbl)
            } else {
                upsample_2x_fwd(
                    xr,
                    cp.up_quad.0,
                    cp.up_zoh_mix.0,
                    cp.up_pre_emph.0,
                    cp.up_aa_mix.0,
                    &mut up_raw_prev[1],
                    &mut up_p1[1],
                    &mut up_p2[1],
                    &mut up_hb[1],
                    record_tape,
                )
            };
            (
                [[l_pair[0], r_pair[0]], [l_pair[1], r_pair[1]]],
                [d2l, d2r],
                [uzl, uzr],
                [dxpl, dxpr],
                [hbl, hbr],
            )
        } else {
            (
                [[xl, xr], [0.0, 0.0]],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [[0.0; 2], [0.0; 2]],
            )
        };

        let mut tap_sub = [[[0.0_f64; 7]; 2]; 2];
        let (mut u_sub, mut hp_sub) = ([[0.0_f64; 2]; 2], [[0.0_f64; 2]; 2]);
        let mut h_sub = [[[0.0_f64; MAX_TOTAL_GRU_LANES]; 2]; 2];

        for sub in 0..sub_steps {
            let sub_t = if m.os2x { t * 2 + sub } else { t };
            let xs = xs_sub[sub];
            let (mut dr, mut det_lp_prev, mut v_det_hp, mut raw_sign) =
                ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
            for ch in 0..svf_ch {
                det_lp_prev[ch] = det_lp[ch];
                det_lp[ch] = cp.det_lp.0 * det_lp[ch] + (1.0 - cp.det_lp.0) * xs[ch];
                let vh = xs[ch] + cp.det_hp_mix.0 * (xs[ch] - det_lp[ch]);
                v_det_hp[ch] = vh;
                dr[ch] = vh.abs();
                raw_sign[ch] = if vh > 0.0 {
                    1.0
                } else if vh < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
            if svf_ch == 1 {
                dr[1] = dr[0];
            }
            let diff = dr[1] - dr[0];
            let det = [dr[0] + half_link * diff, dr[1] - half_link * diff];

            let mut tap_ch_cur = [[0.0_f64; 7]; 2];
            for ch in 0..svf_ch {
                let env_prev = env[ch];
                let atk_taken = det[ch] > env_prev;
                let alpha = if atk_taken { cp.atk.0 } else { cp.rel.0 };
                env[ch] = alpha * env_prev + (1.0 - alpha) * det[ch];
                let rsq = if cp.env_comp.0 == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + cp.env_comp.0 * env[ch] * env[ch]).sqrt()
                };
                let e = env[ch] * rsq;
                let (oct_p, w1_uncl, g1, g2) = if static_svf {
                    (1.0, cp.base_g.0, g1_static, g2_static)
                } else {
                    let (op, _) = exp2_oct_with_deriv(cp.env_oct.0 * e);
                    let w1_u = cp.base_g.0 * op;
                    let w1_c = w1_u.clamp(cp.g_min.0, cp.g_max.0);
                    let (g1_tan, _) = tan_prewarp_with_deriv(w1_c);
                    let g1_val = g1_tan.clamp(1.0e-4, g_hi);
                    let g2_val = (g1_val * cp.sec2_r.0).clamp(1.0e-4, g_hi);
                    (op, w1_u, g1_val, g2_val)
                };

                let svf_out = step_tpt_pair_fwd(
                    xs[ch],
                    [g1, g2],
                    [cp.dk1.0, cp.dk2.0],
                    [cp.s2_bp.0, cp.s2_hp.0],
                    &mut ic1[ch],
                    &mut ic2[ch],
                );
                let taps = svf_out.taps;
                tap_ch_cur[ch] = taps;
                if ch < active_ch {
                    if decimate_all_lanes {
                        tap_sub[sub][ch] = taps;
                    } else {
                        ws.sub_taps[ch][sub_t] = taps;
                    }
                }
                if record_tape {
                    let tp = SvfChanTape {
                        taps,
                        det_lp_prev: det_lp_prev[ch],
                        det_lp: det_lp[ch],
                        v_det_hp: v_det_hp[ch],
                        det_raw_sign: raw_sign[ch],
                        env_prev,
                        atk_taken,
                        rsq3: rsq * rsq * rsq,
                        e,
                        w1_unclamped: w1_uncl,
                        oct_p,
                        g1,
                        g2,
                        s1_c1: svf_out.s1_c1,
                        s1_a1: svf_out.s1_a1,
                        s1_a2: svf_out.s1_a2,
                        s1_a3: svf_out.s1_a3,
                        s1_v1: svf_out.s1_v1,
                        s1_v3: svf_out.s1_v3,
                        s2_c1: svf_out.s2_c1,
                        s2_a1: svf_out.s2_a1,
                        s2_a2: svf_out.s2_a2,
                        s2_a3: svf_out.s2_a3,
                        s2_v1: svf_out.s2_v1,
                        s2_v3: svf_out.s2_v3,
                    };
                    ws.svf_sub[ch][sub_t] = tp;
                }
            }

            let mut u_s2_lr = [0.0_f64; 2];
            let mut lp2_lr = [0.0_f64; 2];
            for ch in 0..svf_ch {
                let mut u_s2 = 0.0_f64;
                for i in 0..7 {
                    u_s2 += m.svf_stage2_feed_taps[i] * tap_ch_cur[ch][i];
                }
                u_s2_lr[ch] = u_s2;
                lp2_lr[ch] = tap_ch_cur[ch][4];
            }
            if svf_ch == 1 {
                u_s2_lr[1] = u_s2_lr[0];
                lp2_lr[1] = lp2_lr[0];
            }
            let al = u_s2_lr[0].abs();
            let ar = u_s2_lr[1].abs();
            let abs_lr = [al, ar];
            let kl = cp.key_link.0;
            let (rl_mix, rr_mix) = if kl <= 0.0 {
                (al, ar)
            } else {
                ((1.0 - kl) * al + kl * ar, (1.0 - kl) * ar + kl * al)
            };
            let r_mix_lr = [rl_mix, rr_mix];
            let th = cp.det_thresh.0;

            for ch in 0..active_ch {
                let u_s2 = u_s2_lr[ch];
                let u_lp_prev = u_lp[ch];
                u_lp[ch] = u_lp_prev + cp.hp_beta.0 * (u_s2 - u_lp_prev);
                let eff_lp = if m_lp2 > 0.0 {
                    u_lp[ch] + m_lp2 * (lp2_lr[ch] - u_lp[ch])
                } else {
                    u_lp[ch]
                };
                let u_hp = u_s2 - eff_lp;
                let r_mix = r_mix_lr[ch];
                let r_mix_above_thresh = th <= 0.0 || r_mix > th;
                let rect = if th <= 0.0 {
                    r_mix
                } else {
                    (r_mix - th).max(0.0)
                };
                let fast_atk_taken = rect > env_fast[ch];
                let fb = if fast_atk_taken {
                    cp.fast_atk.0
                } else {
                    cp.fast_rel.0
                };
                let env_fast_prev = env_fast[ch];
                env_fast[ch] = env_fast_prev + fb * (rect - env_fast_prev);
                let slow_rsq = if cp.slow_det_comp.0 == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + cp.slow_det_comp.0 * rect * rect).sqrt()
                };
                let slow_rsq3 = slow_rsq * slow_rsq * slow_rsq;
                let rect_slow = rect * slow_rsq;
                let slow_atk_taken = rect_slow > env_slow[ch];
                let sb = if slow_atk_taken {
                    cp.slow_atk.0
                } else {
                    cp.slow_rel.0
                };
                let env_slow_prev = env_slow[ch];
                env_slow[ch] = env_slow_prev + sb * (rect_slow - env_slow_prev);
                let (u_pinch, pinch_inv) = if cp.pinch_k.0 > 0.0 {
                    let pinv = 1.0 / (1.0 + cp.pinch_k.0 * env_fast[ch] * env_fast[ch]);
                    (u_s2 * (1.0 - pinv), pinv)
                } else {
                    (0.0, 1.0)
                };
                let feats = [u_s2, u_hp, u_pinch, env_fast[ch], env_slow[ch]];
                if decimate_all_lanes {
                    u_sub[sub][ch] = u_s2;
                    hp_sub[sub][ch] = u_hp;
                } else {
                    ws.sub_u[ch][sub_t] = u_s2;
                    ws.sub_hp[ch][sub_t] = u_hp;
                }
                if record_tape {
                    let sign_u = if u_s2 > 0.0 {
                        1.0
                    } else if u_s2 < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    ws.feat_sub[ch][sub_t] = InputFeatSubTape {
                        feats,
                        lp2: lp2_lr[ch],
                        raw_abs_u: abs_lr[ch],
                        rect,
                        r_mix_above_thresh,
                        u_lp_prev,
                        env_fast_prev,
                        env_slow_prev,
                        sign_u,
                        fast_atk_taken,
                        slow_atk_taken,
                        pinch_inv,
                        rect_slow,
                        slow_rsq3,
                        up_d2: d2_lr[ch],
                        up_zoh_diff: uz_diff_lr[ch],
                        up_dxp_diff: dxp_diff_lr[ch],
                        up_hb_diff: up_hb_diff_lr[ch],
                    };
                    let gtp = &mut ws.gru_sub[ch][sub_t];
                    gtp.a_sig_mask_bits = [0; NUM_GRU_BLOCKS];
                    gtp.clamp_mask_bits = [0; NUM_GRU_BLOCKS];
                }

                let mut prev_h = [0.0_f64; MAX_BLOCK_LANES];
                for b in 0..NUM_GRU_BLOCKS {
                    let blk = &m.blocks[b];
                    let h_old = h_state[ch][b];
                    let mut h_new = [0.0_f64; MAX_BLOCK_LANES];
                    let t_off = BLK_TAPE_OFF[b];
                    for dst_g in 0..g_cnts[b] {
                        let d_base = dst_g << 2;
                        let mut az_g = [0.0_f64; 4];
                        let mut ar_g = [0.0_f64; 4];
                        let mut an_in_g = [0.0_f64; 4];
                        let mut rec_n_g = [0.0_f64; 4];
                        for dl in 0..4 {
                            let dst = d_base + dl;
                            az_g[dl] = cp.b_z[b][dst].0;
                            ar_g[dl] = cp.b_r[b][dst].0;
                        }
                        for f in 0..NUM_IN_FEATS_B0 {
                            let fv = feats[f];
                            let wz = &blk.w_in_z[f][d_base..d_base + 4];
                            let wr = &blk.w_in_r[f][d_base..d_base + 4];
                            let wn = &blk.w_in_n[f][d_base..d_base + 4];
                            for dl in 0..4 {
                                az_g[dl] += wz[dl] * fv;
                                ar_g[dl] += wr[dl] * fv;
                                an_in_g[dl] += wn[dl] * fv;
                            }
                        }
                        for dl in 0..4 {
                            let dst = d_base + dl;
                            if b > 0 {
                                for sg in 0..=dst_g {
                                    let s = sg << 2;
                                    az_g[dl] += dot4(&blk.w_prev_z[dst], &prev_h, s);
                                    ar_g[dl] += dot4(&blk.w_prev_r[dst], &prev_h, s);
                                    an_in_g[dl] += dot4(&blk.w_prev_n[dst], &prev_h, s);
                                }
                            }
                            for sg in 0..=dst_g {
                                let s = sg << 2;
                                az_g[dl] += dot4(&blk.w_rec_z[dst], &h_old, s);
                                ar_g[dl] += dot4(&blk.w_rec_r[dst], &h_old, s);
                                rec_n_g[dl] += dot4(&blk.w_rec_n[dst], &h_old, s);
                            }
                        }
                        let mut a_sig_raw_g = [0.0_f64; 4];
                        let mut a_sig_g = [0.0_f64; 4];
                        let mut g_sag_g = [0.0_f64; 4];
                        let mut an_g = [0.0_f64; 4];
                        for dl in 0..4 {
                            let dst = d_base + dl;
                            let a_sig_raw = an_in_g[dl] + rec_n_g[dl];
                            let a_sig = a_sig_raw.clamp(-12.0, 12.0);
                            let a_eff = a_sig + cp.k_quad[b][dst].0 * a_sig * a_sig;
                            let g_sag = 1.0 + cp.k_sag[b][dst].0 * feats[4];
                            a_sig_raw_g[dl] = a_sig_raw;
                            a_sig_g[dl] = a_sig;
                            g_sag_g[dl] = g_sag;
                            an_g[dl] = g_sag * a_eff + cp.b_n[b][dst].0;
                        }
                        let nl =
                            eval_gru_group_nonlinear_fwd(az_g, ar_g, g_sag_g, an_g, record_tape);
                        for dl in 0..4 {
                            let dst = d_base + dl;
                            let delta_tanh = nl.tan_n[dl] - cp.tanh_bn[b][dst];
                            let nv = nl.h_sag[dl] * delta_tanh;
                            let h_raw = (1.0 - nl.z[dl]) * h_old[dst] + nl.r[dl] * nv;
                            let h_cl = h_raw.clamp(-2.5, 2.5);
                            h_new[dst] = h_cl;
                            if decimate_all_lanes {
                                h_sub[sub][ch][b * MAX_BLOCK_LANES + dst] = h_cl;
                            } else {
                                ws.sub_h[ch][sub_t][b * MAX_BLOCK_LANES + dst] = h_cl;
                            }
                            if record_tape {
                                let tp = &mut ws.gru_sub[ch][sub_t];
                                let t_idx = t_off + dst;
                                tp.z[t_idx] = nl.z[dl];
                                tp.dz_daz[t_idx] = nl.dz[dl];
                                tp.r[t_idx] = nl.r[dl];
                                tp.dr_dar[t_idx] = nl.dr[dl];
                                tp.a_sig[t_idx] = a_sig_g[dl];
                                tp.delta_tanh[t_idx] = delta_tanh;
                                tp.dtanh_an[t_idx] = nl.dtan_n[dl];
                                if a_sig_raw_g[dl] > -12.0 && a_sig_raw_g[dl] < 12.0 {
                                    tp.a_sig_mask_bits[b] |= 1_u16 << dst;
                                }
                                if h_raw > -2.5 && h_raw < 2.5 {
                                    tp.clamp_mask_bits[b] |= 1_u16 << dst;
                                }
                            }
                        }
                    }
                    h_state[ch][b] = h_new;
                    prev_h = h_new;
                }
            }

            if record_tape && svf_ch == 2 && active_ch == 1 {
                let u_s2_r = u_s2_lr[1];
                let sign_u_r = if u_s2_r > 0.0 {
                    1.0
                } else if u_s2_r < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                ws.feat_sub[1][sub_t] = InputFeatSubTape {
                    feats: [u_s2_r, 0.0, 0.0, 0.0, 0.0],
                    lp2: lp2_lr[1],
                    raw_abs_u: abs_lr[1],
                    rect: 0.0,
                    r_mix_above_thresh: false,
                    u_lp_prev: 0.0,
                    env_fast_prev: 0.0,
                    env_slow_prev: 0.0,
                    sign_u: sign_u_r,
                    fast_atk_taken: false,
                    slow_atk_taken: false,
                    pinch_inv: 1.0,
                    rect_slow: 0.0,
                    slow_rsq3: 1.0,
                    up_d2: d2_lr[1],
                    up_zoh_diff: uz_diff_lr[1],
                    up_dxp_diff: dxp_diff_lr[1],
                    up_hb_diff: up_hb_diff_lr[1],
                };
            }
        }

        if decimate_all_lanes {
            for ch in 0..active_ch {
                if !m.os2x {
                    ws.tap_dec[ch][t] = tap_sub[0][ch];
                    ws.u_dec[ch][t] = u_sub[0][ch];
                    ws.hp_dec[ch][t] = hp_sub[0][ch];
                    ws.h_dec[ch][t] = h_sub[0][ch];
                } else {
                    for i in 0..7 {
                        let (v, _, _) = downsample_2x_fwd(
                            tap_sub[0][ch][i],
                            tap_sub[1][ch][i],
                            &mut dec_tap_p[ch][i],
                            &mut dec_tap_aa[ch][i],
                            cp.dec_c_prev1.0,
                            cp.dec_c0.0,
                            cp.dec_aa_mix.0,
                            false,
                        );
                        ws.tap_dec[ch][t][i] = v;
                    }
                    let (uv, _, _) = downsample_2x_fwd(
                        u_sub[0][ch],
                        u_sub[1][ch],
                        &mut dec_u_p[ch],
                        &mut dec_u_aa[ch],
                        cp.dec_c_prev1.0,
                        cp.dec_c0.0,
                        cp.dec_aa_mix.0,
                        false,
                    );
                    ws.u_dec[ch][t] = uv;
                    let (hv, _, _) = downsample_2x_fwd(
                        hp_sub[0][ch],
                        hp_sub[1][ch],
                        &mut dec_hp_p[ch],
                        &mut dec_hp_aa[ch],
                        cp.dec_c_prev1.0,
                        cp.dec_c0.0,
                        cp.dec_aa_mix.0,
                        false,
                    );
                    ws.hp_dec[ch][t] = hv;
                    for b in 0..NUM_GRU_BLOCKS {
                        let base = b * MAX_BLOCK_LANES;
                        for l in 0..c_blk[b] {
                            let idx = base + l;
                            let (val, _, _) = downsample_2x_fwd(
                                h_sub[0][ch][idx],
                                h_sub[1][ch][idx],
                                &mut dec_h_p[ch][idx],
                                &mut dec_h_aa[ch][idx],
                                cp.dec_c_prev1.0,
                                cp.dec_c0.0,
                                cp.dec_aa_mix.0,
                                false,
                            );
                            ws.h_dec[ch][t][idx] = val;
                        }
                    }
                }
            }
        }
    }
}

type GateTriplet = ([f64; 4], [f64; 4], [f64; 4]);
type BlockRows<'a> = (
    &'a [[f64; MAX_BLOCK_LANES]],
    &'a [[f64; MAX_BLOCK_LANES]],
    &'a [[f64; MAX_BLOCK_LANES]],
);
type BlockRowsMut<'a> = (
    &'a mut [[f64; MAX_BLOCK_LANES]],
    &'a mut [[f64; MAX_BLOCK_LANES]],
    &'a mut [[f64; MAX_BLOCK_LANES]],
);

#[inline(always)]
fn accum_4row_matvec_and_grad(
    n_src: usize,
    src: &[f64; MAX_BLOCK_LANES],
    d_gates: GateTriplet,
    w_rows: BlockRows<'_>,
    g_rows: BlockRowsMut<'_>,
    d_out: &mut [f64; MAX_BLOCK_LANES],
) {
    let (daz, dar, das) = d_gates;
    let (
        [wz0, wz1, wz2, wz3],
        [wr0, wr1, wr2, wr3],
        [wn0, wn1, wn2, wn3],
        [gz0, gz1, gz2, gz3],
        [gr0, gr1, gr2, gr3],
        [gn0, gn1, gn2, gn3],
    ) = (w_rows.0, w_rows.1, w_rows.2, g_rows.0, g_rows.1, g_rows.2)
    else {
        return;
    };
    for sk in 0..n_src {
        let sv = src[sk];
        gz0[sk] += sv * daz[0];
        gz1[sk] += sv * daz[1];
        gz2[sk] += sv * daz[2];
        gz3[sk] += sv * daz[3];
        gr0[sk] += sv * dar[0];
        gr1[sk] += sv * dar[1];
        gr2[sk] += sv * dar[2];
        gr3[sk] += sv * dar[3];
        gn0[sk] += sv * das[0];
        gn1[sk] += sv * das[1];
        gn2[sk] += sv * das[2];
        gn3[sk] += sv * das[3];
        let s0 = wz0[sk] * daz[0] + wr0[sk] * dar[0] + wn0[sk] * das[0];
        let s1 = wz1[sk] * daz[1] + wr1[sk] * dar[1] + wn1[sk] * das[1];
        let s2 = wz2[sk] * daz[2] + wr2[sk] * dar[2] + wn2[sk] * das[2];
        let s3 = wz3[sk] * daz[3] + wr3[sk] * dar[3] + wn3[sk] * das[3];
        d_out[sk] += (s0 + s2) + (s1 + s3);
    }
}

/// Reverse-mode adjoint for one lower-triangular 4-lane-grouped GRU block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_gru_block(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    b: usize,
    c_blk: usize,
    tp: &GruCascadeSubTape,
    h_old_block: Option<&[f64]>,
    prev_h: Option<&[f64]>,
    feats: &[f64; NUM_IN_FEATS_B0],
    dh_sub: &[f64; MAX_TOTAL_GRU_LANES],
    r_gru_h: &mut [f64; MAX_BLOCK_LANES],
    d_prev_h: &mut [f64; MAX_BLOCK_LANES],
    d_feats: &mut [f64; NUM_IN_FEATS_B0],
    grad: &mut V3MatryoshkaOptF64,
) {
    const ZERO_BLOCK: [f64; MAX_BLOCK_LANES] = [0.0; MAX_BLOCK_LANES];
    let h_old_slice = h_old_block.unwrap_or(&ZERO_BLOCK);
    let mut h_old = ZERO_BLOCK;
    h_old.copy_from_slice(&h_old_slice[..MAX_BLOCK_LANES]);
    let blk = &m.blocks[b];
    let gb = &mut grad.blocks[b];
    let base = b * MAX_BLOCK_LANES;
    let t_off = BLK_TAPE_OFF[b];
    let g_cnt = (c_blk >> 2).clamp(1, MAX_GROUPS_PER_BLOCK[b]);
    let mut next_r_h = [0.0_f64; MAX_BLOCK_LANES];
    let mut d_prev_out = [0.0_f64; MAX_BLOCK_LANES];
    let mut prev_buf = ZERO_BLOCK;
    if let Some(pv) = prev_h {
        prev_buf.copy_from_slice(&pv[..MAX_BLOCK_LANES]);
    }

    for dst_g in 0..g_cnt {
        let d_base = dst_g << 2;
        let (g_sags, h_sags) = if feats[4] == 0.0 {
            ([1.0_f64; 4], [1.0_f64; 4])
        } else {
            let gs = std::array::from_fn(|dl| 1.0 + cp.k_sag[b][d_base + dl].0 * feats[4]);
            (gs, inv4_pos_f64(gs))
        };
        let mut daz_g = [0.0_f64; 4];
        let mut dar_g = [0.0_f64; 4];
        let mut das_g = [0.0_f64; 4];
        for dl in 0..4 {
            let dst = d_base + dl;
            let t_idx = t_off + dst;
            let clamp_mask = if ((tp.clamp_mask_bits[b] >> dst) & 1) != 0 {
                1.0
            } else {
                0.0
            };
            let d_h = (dh_sub[base + dst] + r_gru_h[dst] + d_prev_h[dst]) * clamp_mask;
            next_r_h[dst] += (1.0 - tp.z[t_idx]) * d_h;
            let dz = -h_old[dst] * d_h;
            let g_sag = g_sags[dl];
            let h_sag = h_sags[dl];
            let a_sig = tp.a_sig[t_idx];
            let a_eff = a_sig + cp.k_quad[b][dst].0 * a_sig * a_sig;
            let nv = h_sag * tp.delta_tanh[t_idx];
            let dr_val = nv * d_h;
            let dn = tp.r[t_idx] * d_h;
            let daz = dz * tp.dz_daz[t_idx];
            let dar = dr_val * tp.dr_dar[t_idx];
            let d_h_sag = dn * tp.delta_tanh[t_idx];
            let d_delta = dn * h_sag;
            let dan = d_delta * tp.dtanh_an[t_idx];
            let d_a_eff = g_sag * dan;
            let d_g_sag = a_eff * dan - h_sag * h_sag * d_h_sag;
            gb.k_quad[dst] += cp.k_quad[b][dst].1 * a_sig * a_sig * d_a_eff;
            let a_sig_mask = if ((tp.a_sig_mask_bits[b] >> dst) & 1) != 0 {
                1.0
            } else {
                0.0
            };
            let d_a_sig = (1.0 + 2.0 * cp.k_quad[b][dst].0 * a_sig) * d_a_eff * a_sig_mask;

            gb.b_z[dst] += cp.b_z[b][dst].1 * daz;
            gb.b_r[dst] += cp.b_r[b][dst].1 * dar;
            gb.b_n[dst] += cp.b_n[b][dst].1 * (dan - cp.dtanh_bn[b][dst] * d_delta);
            gb.k_sag[dst] += cp.k_sag[b][dst].1 * feats[4] * d_g_sag;
            d_feats[4] += cp.k_sag[b][dst].0 * d_g_sag;

            daz_g[dl] = daz;
            dar_g[dl] = dar;
            das_g[dl] = d_a_sig;
        }
        for f in 0..NUM_IN_FEATS_B0 {
            let fv = feats[f];
            let gz = &mut gb.w_in_z[f][d_base..d_base + 4];
            let gr = &mut gb.w_in_r[f][d_base..d_base + 4];
            let gn = &mut gb.w_in_n[f][d_base..d_base + 4];
            let bz = &blk.w_in_z[f][d_base..d_base + 4];
            let br = &blk.w_in_r[f][d_base..d_base + 4];
            let bn = &blk.w_in_n[f][d_base..d_base + 4];
            let mut df = 0.0_f64;
            for dl in 0..4 {
                gz[dl] += fv * daz_g[dl];
                gr[dl] += fv * dar_g[dl];
                gn[dl] += fv * das_g[dl];
                df += bz[dl] * daz_g[dl] + br[dl] * dar_g[dl] + bn[dl] * das_g[dl];
            }
            d_feats[f] += df;
        }
        let n_src = (dst_g + 1) << 2;
        if prev_h.is_some() {
            accum_4row_matvec_and_grad(
                n_src,
                &prev_buf,
                (daz_g, dar_g, das_g),
                (
                    &blk.w_prev_z[d_base..d_base + 4],
                    &blk.w_prev_r[d_base..d_base + 4],
                    &blk.w_prev_n[d_base..d_base + 4],
                ),
                (
                    &mut gb.w_prev_z[d_base..d_base + 4],
                    &mut gb.w_prev_r[d_base..d_base + 4],
                    &mut gb.w_prev_n[d_base..d_base + 4],
                ),
                &mut d_prev_out,
            );
        }
        accum_4row_matvec_and_grad(
            n_src,
            &h_old,
            (daz_g, dar_g, das_g),
            (
                &blk.w_rec_z[d_base..d_base + 4],
                &blk.w_rec_r[d_base..d_base + 4],
                &blk.w_rec_n[d_base..d_base + 4],
            ),
            (
                &mut gb.w_rec_z[d_base..d_base + 4],
                &mut gb.w_rec_r[d_base..d_base + 4],
                &mut gb.w_rec_n[d_base..d_base + 4],
            ),
            &mut next_r_h,
        );
    }
    *r_gru_h = next_r_h;
    *d_prev_h = d_prev_out;
}
