//! Exact 44-parameter `SensoryQuad` (`Op-M/N[Sense]`) sidechain envelope follower + stereo VCA
//! forward evaluator and ZOH-coupled reverse-time BPTT adjoints matching `dnam_vm::graph::sensory`
//! (`ref[dsp.router.ballistic-envelope-bank]`, `ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::{FRAC_1_PI, PI};

use dnam_kernels::{MAX_PARALRU_R_F32, MIN_TPT_G, SOFT_RECT_EPS};

use super::super::model::SonaModelF64;

/// Maximum `TPT-SVF` tangent gain allowed in `SensoryQuad` (`dnam-vm/src/graph/sensory.rs`).
const SENSE_SVF_G_MAX: f64 = 64.0;
/// Maximum `TPT-SVF` damping allowed in `SensoryQuad`.
const SENSE_SVF_K_MAX: f64 = 8.0;
/// Maximum `SensoryQuad` stereo key input weight magnitude.
const SENSE_IN_LIMIT: f64 = 4.0;
/// Maximum `SensoryQuad` tone tap weight magnitude.
const SENSE_TONE_LIMIT: f64 = 16.0;
/// Maximum `SensoryQuad` VCA projection weight magnitude.
const SENSE_VCA_LIMIT: f64 = 512.0;

/// Exact `44`-parameter gradients for `SensoryQuad` matching [`SonaModelF64`]'s `sense_*` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SensoryQuadGrads {
    pub d_in_l: [f64; 4],
    pub d_in_r: [f64; 4],
    pub d_svf_g: [f64; 4],
    pub d_svf_k: [f64; 4],
    pub d_tone_lp: [f64; 4],
    pub d_tone_bp: [f64; 4],
    pub d_tone_hp: [f64; 4],
    pub d_lru_rel: [f64; 4],
    pub d_lru_atk: [f64; 4],
    pub d_lru_kappa: [f64; 4],
    pub d_vca_w: [f64; 4],
}

#[inline(always)]
fn clamp_finite(v: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

/// Exact `f64` analogue of `dnam_kernels::soft_rect_zero_dc_vec4` and its analytic derivative.
#[inline(always)]
fn soft_rect_zero_dc_f64(z: f64) -> (f64, f64) {
    if z == 0.0 {
        return (0.0, 0.5);
    }
    let eps = f64::from(SOFT_RECT_EPS);
    let eps2 = eps * eps;
    let z_abs = z.abs();
    let rad = (z * z + eps2).sqrt();
    let phi = if z > 0.0 {
        0.5 * (z_abs + (z_abs * z_abs) / (rad + eps))
    } else {
        -0.5 * eps * (z_abs / (rad + z_abs)) * (1.0 + z_abs / (rad + eps))
    };
    let phi_prime = 0.5 * (1.0 + z / rad);
    (phi, phi_prime)
}

#[inline]
fn zoh_rescale_svf_g(g_ref: f64, rho: f64) -> (f64, f64) {
    let g_min = f64::from(MIN_TPT_G);
    let g = clamp_finite(g_ref, g_min, SENSE_SVF_G_MAX, 0.25);
    if (rho - 1.0).abs() <= 1.0e-12 {
        let deriv = if g_ref.is_finite() && g_ref > g_min && g_ref < SENSE_SVF_G_MAX {
            1.0
        } else {
            0.0
        };
        return (g, deriv);
    }
    let r_raw = g.atan() * FRAC_1_PI * rho;
    let r_eff = r_raw.clamp(1.0e-5, 0.495);
    let g_eff = (PI * r_eff).tan();
    let deriv = if g_ref.is_finite()
        && g_ref > g_min
        && g_ref < SENSE_SVF_G_MAX
        && r_raw > 1.0e-5
        && r_raw < 0.495
    {
        rho * (1.0 + g_eff * g_eff) / (1.0 + g * g)
    } else {
        0.0
    };
    (g_eff, deriv)
}

#[inline]
fn zoh_rescale_pow_pole(p_ref: f64, max_pole: f64, rho: f64) -> (f64, f64) {
    let p = clamp_finite(p_ref, 0.0, max_pole, 0.0);
    if p <= 0.0 {
        return (0.0, 0.0);
    }
    if (rho - 1.0).abs() <= 1.0e-12 {
        let deriv = if p_ref.is_finite() && p_ref > 0.0 && p_ref < max_pole {
            1.0
        } else {
            0.0
        };
        return (p, deriv);
    }
    let p_eff = p.powf(rho).clamp(0.0, max_pole);
    let deriv = if p_ref.is_finite() && p_ref > 0.0 && p_ref < max_pole && p_eff < max_pole {
        rho * p.powf(rho - 1.0)
    } else {
        0.0
    };
    (p_eff, deriv)
}

#[derive(Clone, Copy)]
struct LaneCoeff {
    in_l: f64,
    in_r: f64,
    g_eff: f64,
    dg_eff_dg_ref: f64,
    k: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    tone_lp: f64,
    tone_bp: f64,
    tone_hp: f64,
    rel_eff: f64,
    drel_eff_drel_ref: f64,
    atk_eff: f64,
    datk_eff_datk_ref: f64,
    kappa: f64,
    vca_w: f64,
    in_l_open: bool,
    in_r_open: bool,
    k_open: bool,
    tlp_open: bool,
    tbp_open: bool,
    thp_open: bool,
    kap_open: bool,
    vca_open: bool,
}

impl LaneCoeff {
    fn from_model(m: &SonaModelF64, lane: usize, rho: f64) -> Self {
        let r_max = f64::from(MAX_PARALRU_R_F32);
        let (g_eff, dg_eff_dg_ref) = zoh_rescale_svf_g(m.sense_svf_g[lane], rho);
        let k = clamp_finite(
            m.sense_svf_k[lane],
            0.0,
            SENSE_SVF_K_MAX,
            core::f64::consts::FRAC_1_SQRT_2,
        );
        let a1 = 1.0 / (1.0 + g_eff * (g_eff + k));
        let a2 = g_eff * a1;
        let a3 = g_eff * a2;
        let (rel_eff, drel_eff_drel_ref) = zoh_rescale_pow_pole(m.sense_lru_rel[lane], r_max, rho);
        let (atk_eff, datk_eff_datk_ref) = zoh_rescale_pow_pole(m.sense_lru_atk[lane], 1.0, rho);
        Self {
            in_l: clamp_finite(m.sense_in_l[lane], -SENSE_IN_LIMIT, SENSE_IN_LIMIT, 0.0),
            in_r: clamp_finite(m.sense_in_r[lane], -SENSE_IN_LIMIT, SENSE_IN_LIMIT, 0.0),
            g_eff,
            dg_eff_dg_ref,
            k,
            a1,
            a2,
            a3,
            tone_lp: clamp_finite(
                m.sense_tone_lp[lane],
                -SENSE_TONE_LIMIT,
                SENSE_TONE_LIMIT,
                0.0,
            ),
            tone_bp: clamp_finite(
                m.sense_tone_bp[lane],
                -SENSE_TONE_LIMIT,
                SENSE_TONE_LIMIT,
                0.0,
            ),
            tone_hp: clamp_finite(
                m.sense_tone_hp[lane],
                -SENSE_TONE_LIMIT,
                SENSE_TONE_LIMIT,
                0.0,
            ),
            rel_eff,
            drel_eff_drel_ref,
            atk_eff,
            datk_eff_datk_ref,
            kappa: clamp_finite(m.sense_lru_kappa[lane], 0.0, 1.0, 0.0),
            vca_w: clamp_finite(m.sense_vca_w[lane], -SENSE_VCA_LIMIT, SENSE_VCA_LIMIT, 0.0),
            in_l_open: m.sense_in_l[lane].is_finite() && m.sense_in_l[lane].abs() < SENSE_IN_LIMIT,
            in_r_open: m.sense_in_r[lane].is_finite() && m.sense_in_r[lane].abs() < SENSE_IN_LIMIT,
            k_open: m.sense_svf_k[lane].is_finite()
                && m.sense_svf_k[lane] >= 0.0
                && m.sense_svf_k[lane] < SENSE_SVF_K_MAX,
            tlp_open: m.sense_tone_lp[lane].is_finite()
                && m.sense_tone_lp[lane].abs() < SENSE_TONE_LIMIT,
            tbp_open: m.sense_tone_bp[lane].is_finite()
                && m.sense_tone_bp[lane].abs() < SENSE_TONE_LIMIT,
            thp_open: m.sense_tone_hp[lane].is_finite()
                && m.sense_tone_hp[lane].abs() < SENSE_TONE_LIMIT,
            kap_open: m.sense_lru_kappa[lane].is_finite()
                && m.sense_lru_kappa[lane] >= 0.0
                && m.sense_lru_kappa[lane] < 1.0,
            vca_open: m.sense_vca_w[lane].is_finite()
                && m.sense_vca_w[lane].abs() < SENSE_VCA_LIMIT,
        }
    }
}

/// Evaluates `SensoryQuad` sidechain sensing + uniform stereo VCA gain over `(in_l, in_r)` and
/// optionally executes reverse-time BPTT into [`SensoryQuadGrads`] and `(grad_in_l, grad_in_r)`.
pub(crate) fn eval_sensory_quad_and_adjoint(
    model: &SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    d_out_opt: Option<(&[f64], &[f64])>,
    mut grad_in_opt: Option<(&mut [f64], &mut [f64])>,
) -> (Vec<f64>, Vec<f64>, SensoryQuadGrads) {
    let n = in_l.len().min(in_r.len());
    if n == 0 {
        return (Vec::new(), Vec::new(), SensoryQuadGrads::default());
    }
    let sr = if model.sample_rate_hz.is_finite() && model.sample_rate_hz > 0.0 {
        model.sample_rate_hz
    } else {
        48_000.0
    };
    let rho = (48_000.0 / sr).clamp(0.125, 8.0);
    let lanes: [LaneCoeff; 4] = core::array::from_fn(|l| LaneCoeff::from_model(model, l, rho));

    let is_bypass = lanes.iter().all(|c| c.vca_w == 0.0);
    let need_adj = d_out_opt.is_some();
    if is_bypass && !need_adj {
        return (
            in_l[..n].to_vec(),
            in_r[..n].to_vec(),
            SensoryQuadGrads::default(),
        );
    }

    let mut out_l = vec![0.0_f64; n];
    let mut out_r = vec![0.0_f64; n];
    let mut s1_arr = [0.0_f64; 4];
    let mut s2_arr = [0.0_f64; 4];
    let mut lru_v_arr = [0.0_f64; 4];

    // Forward state trace stored only when computing adjoints.
    let mut g_raw_hist = if need_adj {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    let mut u_s_hist = if need_adj {
        vec![[0.0_f64; 4]; n]
    } else {
        Vec::new()
    };
    let mut s1_prev_hist = if need_adj {
        vec![[0.0_f64; 4]; n]
    } else {
        Vec::new()
    };
    let mut s2_prev_hist = if need_adj {
        vec![[0.0_f64; 4]; n]
    } else {
        Vec::new()
    };
    let mut s_sense_hist = if need_adj {
        vec![[0.0_f64; 4]; n]
    } else {
        Vec::new()
    };
    let mut v_prev_hist = if need_adj {
        vec![[0.0_f64; 4]; n]
    } else {
        Vec::new()
    };
    let mut env_hist = if need_adj {
        vec![[0.0_f64; 4]; n]
    } else {
        Vec::new()
    };

    for t in 0..n {
        let xl = in_l[t];
        let xr = in_r[t];
        let mut vca_dot = 0.0_f64;
        for (l, c) in lanes.iter().enumerate() {
            let u_s = c.in_l * xl + c.in_r * xr;
            let (s1_prev, s2_prev, v_prev) = (s1_arr[l], s2_arr[l], lru_v_arr[l]);
            let v3 = u_s - s2_prev;
            let v1 = c.a1 * s1_prev + c.a2 * v3;
            let v2 = s2_prev + c.a2 * s1_prev + c.a3 * v3;
            s1_arr[l] = 2.0 * v1 - s1_prev;
            s2_arr[l] = 2.0 * v2 - s2_prev;
            let hp = u_s - c.k * v1 - v2;
            let s_sense = c.tone_lp * v2 + c.tone_bp * v1 + c.tone_hp * hp;
            let decayed = c.rel_eff * v_prev;
            let z_arg = s_sense.abs() - c.kappa * decayed;
            let (phi, _) = soft_rect_zero_dc_f64(z_arg);
            let e_new = decayed + (1.0 - c.atk_eff) * phi;
            lru_v_arr[l] = e_new;
            vca_dot += c.vca_w * e_new;

            if need_adj {
                u_s_hist[t][l] = u_s;
                s1_prev_hist[t][l] = s1_prev;
                s2_prev_hist[t][l] = s2_prev;
                s_sense_hist[t][l] = s_sense;
                v_prev_hist[t][l] = v_prev;
                env_hist[t][l] = e_new;
            }
        }
        let g_raw = 1.0 + vca_dot;
        let g_in = g_raw.clamp(0.0, 2.0);
        out_l[t] = g_in * xl;
        out_r[t] = g_in * xr;
        if need_adj {
            g_raw_hist.push(g_raw);
        }
    }

    let mut grads = SensoryQuadGrads::default();
    if let Some((gl, gr)) = d_out_opt {
        let mut adj_v_next = [0.0_f64; 4];
        let mut adj_s1_next = [0.0_f64; 4];
        let mut adj_s2_next = [0.0_f64; 4];
        let mut d_g_eff = [0.0_f64; 4];
        let mut d_rel_eff = [0.0_f64; 4];
        let mut d_atk_eff = [0.0_f64; 4];

        for t in (0..n).rev() {
            let xl = in_l[t];
            let xr = in_r[t];
            let g_raw = g_raw_hist[t];
            let g_in = g_raw.clamp(0.0, 2.0);
            let g_open = g_raw > 0.0 && g_raw < 2.0;
            let d_g_in = gl[t] * xl + gr[t] * xr;
            let d_g_raw = if g_open { d_g_in } else { 0.0 };

            let mut d_xl = g_in * gl[t];
            let mut d_xr = g_in * gr[t];

            for (l, c) in lanes.iter().enumerate() {
                if c.vca_open {
                    grads.d_vca_w[l] += d_g_raw * env_hist[t][l];
                }
                let lam = d_g_raw * c.vca_w + adj_v_next[l];
                let v_prev = v_prev_hist[t][l];
                let s_sense = s_sense_hist[t][l];
                let rect = s_sense.abs();
                let decayed = c.rel_eff * v_prev;
                let z_arg = rect - c.kappa * decayed;
                let (phi, phi_prime) = soft_rect_zero_dc_f64(z_arg);
                let one_m_atk = 1.0 - c.atk_eff;

                let dz_drel = -c.kappa * v_prev;
                d_rel_eff[l] += lam * (v_prev + one_m_atk * phi_prime * dz_drel);
                d_atk_eff[l] += lam * (-phi);
                if c.kap_open {
                    grads.d_lru_kappa[l] += lam * one_m_atk * phi_prime * (-decayed);
                }
                adj_v_next[l] = lam * c.rel_eff * (1.0 - one_m_atk * phi_prime * c.kappa);

                let d_rect = lam * one_m_atk * phi_prime;
                let rect_sign = if s_sense > 0.0 {
                    1.0
                } else if s_sense < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let d_s_sense = d_rect * rect_sign;

                let u_s = u_s_hist[t][l];
                let s1_prev = s1_prev_hist[t][l];
                let s2_prev = s2_prev_hist[t][l];
                let v3 = u_s - s2_prev;
                let v1 = c.a1 * s1_prev + c.a2 * v3;
                let v2 = s2_prev + c.a2 * s1_prev + c.a3 * v3;
                let hp = u_s - c.k * v1 - v2;

                if c.tlp_open {
                    grads.d_tone_lp[l] += d_s_sense * v2;
                }
                if c.tbp_open {
                    grads.d_tone_bp[l] += d_s_sense * v1;
                }
                if c.thp_open {
                    grads.d_tone_hp[l] += d_s_sense * hp;
                }

                let adj_lp = d_s_sense * c.tone_lp;
                let adj_bp = d_s_sense * c.tone_bp;
                let adj_hp = d_s_sense * c.tone_hp;

                let tot_v1 = adj_bp + 2.0 * adj_s1_next[l] - c.k * adj_hp;
                let tot_v2 = adj_lp + 2.0 * adj_s2_next[l] - adj_hp;

                let d_a1 = tot_v1 * s1_prev;
                let d_a2 = tot_v1 * v3 + tot_v2 * s1_prev;
                let d_a3 = tot_v2 * v3;

                let da1_dg = -c.a1 * c.a1 * (2.0 * c.g_eff + c.k);
                let da2_dg = c.a1 + c.g_eff * da1_dg;
                let da3_dg = c.a2 + c.g_eff * da2_dg;
                d_g_eff[l] += d_a1 * da1_dg + d_a2 * da2_dg + d_a3 * da3_dg;

                if c.k_open {
                    let da1_dk = -c.a1 * c.a2;
                    let da2_dk = -c.a2 * c.a2;
                    let da3_dk = -c.a2 * c.a3;
                    grads.d_svf_k[l] +=
                        (-adj_hp * v1) + d_a1 * da1_dk + d_a2 * da2_dk + d_a3 * da3_dk;
                }

                let d_u_s = adj_hp + c.a2 * tot_v1 + c.a3 * tot_v2;
                if c.in_l_open {
                    grads.d_in_l[l] += d_u_s * xl;
                }
                if c.in_r_open {
                    grads.d_in_r[l] += d_u_s * xr;
                }
                d_xl += d_u_s * c.in_l;
                d_xr += d_u_s * c.in_r;

                let next_adj_s1 = c.a1 * tot_v1 + c.a2 * tot_v2 - adj_s1_next[l];
                let next_adj_s2 = -c.a2 * tot_v1 + (1.0 - c.a3) * tot_v2 - adj_s2_next[l];
                adj_s1_next[l] = next_adj_s1;
                adj_s2_next[l] = next_adj_s2;
            }

            if let Some((ref mut g_il, ref mut g_ir)) = grad_in_opt {
                g_il[t] += d_xl;
                g_ir[t] += d_xr;
            }
        }

        for (l, c) in lanes.iter().enumerate() {
            grads.d_svf_g[l] = d_g_eff[l] * c.dg_eff_dg_ref;
            grads.d_lru_rel[l] = d_rel_eff[l] * c.drel_eff_drel_ref;
            grads.d_lru_atk[l] = d_atk_eff[l] * c.datk_eff_datk_ref;
        }
    }

    (out_l, out_r, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::f32_to_fp16;
    use dnam_vm::DefaultMathBackend;
    use dnam_vm::bin::CrossbarWire;
    use dnam_vm::graph::sensory::{SensoryQuadCold, SensoryQuadHot};
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn test_sensory_model(sr: f64) -> SonaModelF64 {
        let mut m = SonaModelF64::new(sr);
        m.sense_in_l = [0.6, -0.3, 0.45, 0.25];
        m.sense_in_r = [-0.25, 0.55, 0.35, -0.4];
        m.sense_svf_g = [0.05, 0.18, 0.45, 1.1];
        m.sense_svf_k = [0.65, 0.85, 0.50, 1.25];
        m.sense_tone_lp = [0.8, 0.3, -0.2, 0.4];
        m.sense_tone_bp = [0.25, 0.7, 0.35, -0.3];
        m.sense_tone_hp = [-0.15, 0.2, 0.65, 0.55];
        m.sense_lru_rel = [0.97, 0.94, 0.90, 0.82];
        m.sense_lru_atk = [0.15, 0.28, 0.42, 0.58];
        m.sense_lru_kappa = [0.12, 0.24, 0.36, 0.48];
        m.sense_vca_w = [0.45, -0.35, 0.28, -0.22];
        m
    }

    #[test]
    fn test_sensory_quad_44param_fd_zoh_and_vm_parity() {
        let n = 128_usize;
        let in_l: Vec<f64> = (0..n)
            .map(|t| 0.42 * ((t as f64) * 0.11).sin() + 0.18 * ((t as f64) * 0.31).cos())
            .collect();
        let in_r: Vec<f64> = (0..n)
            .map(|t| -0.36 * ((t as f64) * 0.17).cos() + 0.15 * ((t as f64) * 0.23).sin())
            .collect();
        let gl: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.19).cos()).collect();
        let gr: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.27).sin()).collect();
        let obj = |yl: &[f64], yr: &[f64]| -> f64 {
            yl.iter().zip(&gl).map(|(a, b)| a * b).sum::<f64>()
                + yr.iter().zip(&gr).map(|(a, b)| a * b).sum::<f64>()
        };

        // 1. VM forward parity check at 96 kHz (`rho = 0.5`).
        let mut m96 = test_sensory_model(96_000.0);
        // Quantize fp16-wire fields so VM wire hydration matches exactly.
        for l in 0..4 {
            let snap16 = |v: f64| f64::from(dnam_kernels::fp16_to_f32(f32_to_fp16(v as f32)));
            m96.sense_in_l[l] = snap16(m96.sense_in_l[l]);
            m96.sense_in_r[l] = snap16(m96.sense_in_r[l]);
            m96.sense_svf_k[l] = snap16(m96.sense_svf_k[l]);
            m96.sense_tone_lp[l] = snap16(m96.sense_tone_lp[l]);
            m96.sense_tone_bp[l] = snap16(m96.sense_tone_bp[l]);
            m96.sense_tone_hp[l] = snap16(m96.sense_tone_hp[l]);
            m96.sense_lru_kappa[l] = snap16(m96.sense_lru_kappa[l]);
            m96.sense_vca_w[l] = snap16(m96.sense_vca_w[l]);
        }
        let (sl, sr, _) = eval_sensory_quad_and_adjoint(&m96, &in_l, &in_r, None, None);
        let mut cb = CrossbarWire::new_zeroed();
        for l in 0..4 {
            cb.sense_svf_g[l] = F32::new(m96.sense_svf_g[l] as f32);
            cb.sense_lru_alpha_rel[l] = F32::new(m96.sense_lru_rel[l] as f32);
            cb.sense_lru_alpha_atk[l] = F32::new(m96.sense_lru_atk[l] as f32);
            cb.sense_svf_k_fp16[l] = U16::new(f32_to_fp16(m96.sense_svf_k[l] as f32));
            cb.sense_lru_kappa_fp16[l] = U16::new(f32_to_fp16(m96.sense_lru_kappa[l] as f32));
            cb.sense_in_l_fp16[l] = U16::new(f32_to_fp16(m96.sense_in_l[l] as f32));
            cb.sense_in_r_fp16[l] = U16::new(f32_to_fp16(m96.sense_in_r[l] as f32));
            cb.sense_tone_lp_fp16[l] = U16::new(f32_to_fp16(m96.sense_tone_lp[l] as f32));
            cb.sense_tone_bp_fp16[l] = U16::new(f32_to_fp16(m96.sense_tone_bp[l] as f32));
            cb.sense_tone_hp_fp16[l] = U16::new(f32_to_fp16(m96.sense_tone_hp[l] as f32));
            cb.sense_vca_w_fp16[l] = U16::new(f32_to_fp16(m96.sense_vca_w[l] as f32));
            cb.phasor_freq_hz[l] = F32::new(0.0);
        }
        let vm_cold = SensoryQuadCold::from_crossbar_wire(&cb, 0.5);
        let mut vm_hot = SensoryQuadHot::zero();
        for t in 0..n {
            let (vl, vr) = vm_cold.step_frame::<DefaultMathBackend>(
                &mut vm_hot,
                in_l[t] as f32,
                in_r[t] as f32,
                0.0,
            );
            assert!(
                (f64::from(vl) - sl[t]).abs() < 2.0e-5,
                "t={t} L vm={vl} train={}",
                sl[t]
            );
            assert!(
                (f64::from(vr) - sr[t]).abs() < 2.0e-5,
                "t={t} R vm={vr} train={}",
                sr[t]
            );
        }

        // 2. Non-zero d_vca_w when initialized to zero.
        let mut m_zero = test_sensory_model(48_000.0);
        m_zero.sense_vca_w = [0.0; 4];
        let (_, _, g_zero) =
            eval_sensory_quad_and_adjoint(&m_zero, &in_l, &in_r, Some((&gl, &gr)), None);
        assert!(g_zero.d_vca_w.iter().all(|&v| v.abs() > 1.0e-3));

        // 3. Complete 44-parameter central-FD parity at BOTH 48 kHz and 96 kHz.
        for &rate in &[48_000.0_f64, 96_000.0_f64] {
            let m = test_sensory_model(rate);
            let mut g_in_l = vec![0.0_f64; n];
            let mut g_in_r = vec![0.0_f64; n];
            let (_, _, g) = eval_sensory_quad_and_adjoint(
                &m,
                &in_l,
                &in_r,
                Some((&gl, &gr)),
                Some((&mut g_in_l, &mut g_in_r)),
            );
            let eps = 1.0e-6_f64;
            let fd_field =
                |setter: fn(&mut SonaModelF64, usize, f64), base: f64, l: usize| -> f64 {
                    let (mut mp, mut mm) = (m.clone(), m.clone());
                    setter(&mut mp, l, base + eps);
                    setter(&mut mm, l, base - eps);
                    let (pl, pr, _) = eval_sensory_quad_and_adjoint(&mp, &in_l, &in_r, None, None);
                    let (ml, mr, _) = eval_sensory_quad_and_adjoint(&mm, &in_l, &in_r, None, None);
                    (obj(&pl, &pr) - obj(&ml, &mr)) / (2.0 * eps)
                };

            for l in 0..4 {
                let checks = [
                    (
                        "in_l",
                        g.d_in_l[l],
                        fd_field(|x, i, v| x.sense_in_l[i] = v, m.sense_in_l[l], l),
                    ),
                    (
                        "in_r",
                        g.d_in_r[l],
                        fd_field(|x, i, v| x.sense_in_r[i] = v, m.sense_in_r[l], l),
                    ),
                    (
                        "svf_g",
                        g.d_svf_g[l],
                        fd_field(|x, i, v| x.sense_svf_g[i] = v, m.sense_svf_g[l], l),
                    ),
                    (
                        "svf_k",
                        g.d_svf_k[l],
                        fd_field(|x, i, v| x.sense_svf_k[i] = v, m.sense_svf_k[l], l),
                    ),
                    (
                        "tone_lp",
                        g.d_tone_lp[l],
                        fd_field(|x, i, v| x.sense_tone_lp[i] = v, m.sense_tone_lp[l], l),
                    ),
                    (
                        "tone_bp",
                        g.d_tone_bp[l],
                        fd_field(|x, i, v| x.sense_tone_bp[i] = v, m.sense_tone_bp[l], l),
                    ),
                    (
                        "tone_hp",
                        g.d_tone_hp[l],
                        fd_field(|x, i, v| x.sense_tone_hp[i] = v, m.sense_tone_hp[l], l),
                    ),
                    (
                        "lru_rel",
                        g.d_lru_rel[l],
                        fd_field(|x, i, v| x.sense_lru_rel[i] = v, m.sense_lru_rel[l], l),
                    ),
                    (
                        "lru_atk",
                        g.d_lru_atk[l],
                        fd_field(|x, i, v| x.sense_lru_atk[i] = v, m.sense_lru_atk[l], l),
                    ),
                    (
                        "lru_kappa",
                        g.d_lru_kappa[l],
                        fd_field(|x, i, v| x.sense_lru_kappa[i] = v, m.sense_lru_kappa[l], l),
                    ),
                    (
                        "vca_w",
                        g.d_vca_w[l],
                        fd_field(|x, i, v| x.sense_vca_w[i] = v, m.sense_vca_w[l], l),
                    ),
                ];
                for (name, ana, fd) in checks {
                    let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                    assert!(
                        (ana - fd).abs() < 1.0e-4 * scale,
                        "rate={rate} lane={l} {name}: ana={ana} fd={fd}"
                    );
                }
            }
            let (mut xlp, mut xlm) = (in_l.clone(), in_l.clone());
            xlp[39] += eps;
            xlm[39] -= eps;
            let (pl, pr, _) = eval_sensory_quad_and_adjoint(&m, &xlp, &in_r, None, None);
            let (ml, mr, _) = eval_sensory_quad_and_adjoint(&m, &xlm, &in_r, None, None);
            let fd_x = (obj(&pl, &pr) - obj(&ml, &mr)) / (2.0 * eps);
            let scale = g_in_l[39].abs().max(fd_x.abs()).max(1.0e-3);
            assert!(
                (g_in_l[39] - fd_x).abs() < 1.0e-4 * scale,
                "rate={rate} in_l[39]: ana={} fd={fd_x}",
                g_in_l[39]
            );
        }
    }
}
