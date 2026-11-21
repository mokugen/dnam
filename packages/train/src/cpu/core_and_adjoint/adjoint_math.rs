//! Scalar reverse-mode BPTT pullback formulas matching `tools/sona-shaders`'s `adjoint_scan_256`
//! (`common`, `pullback`, `pass2`, and `adaa_deriv`) verbatim (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod adaa_derivs;

use adaa_derivs::{eval_adaa_wf_derivs, scrub_endpoint};
use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, KernelVec4, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32,
    MIN_SAG_GAIN, ONE_THIRD_F32, SonaParaLruDuhemCoeffs4, TptAdjointElem, TptSvfCoeffs4,
    apply_stereo_link_vec4_pair, canonicalize_zero_f32_bits, soft_rect_zero_dc_f32,
    step_paralru_duhem_vec4, step_tpt_svf_vec4,
};

use crate::gpu::abi::FEATURE_VEC4_STRIDE;

const IEEE_F32_EXP_MASK: u32 = 0x7f80_0000;

#[inline(always)]
pub(super) const fn is_finite_f32(x: f32) -> bool {
    (x.to_bits() & IEEE_F32_EXP_MASK) != IEEE_F32_EXP_MASK
}

#[inline(always)]
pub(super) const fn flush_zero(x: f32) -> f32 {
    canonicalize_zero_f32_bits(x)
}

#[inline(always)]
pub(super) fn clamp_nonneg_safe(x: f32, hi: f32) -> f32 {
    if is_finite_f32(x) {
        if x <= 0.0 {
            0.0
        } else if x >= hi {
            hi
        } else {
            canonicalize_zero_f32_bits(x)
        }
    } else if x.to_bits() == IEEE_F32_EXP_MASK {
        hi
    } else {
        0.0
    }
}

#[derive(Copy, Clone)]
pub(super) struct LaneParams {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k: f32,
    pub two_k: f32,
    pub w_lp: f32,
    pub w_bp: f32,
    pub w_hp: f32,
    pub w_ap: f32,
    pub w_dir: f32,
    pub alpha_rel: f32,
    pub alpha_atk: f32,
    pub kappa: f32,
    pub gamma_duhem: f32,
    pub signed_mix: f32,
    pub stereo_link: f32,
    pub alpha_rel_open: bool,
    pub alpha_atk_open: bool,
    pub kappa_open: bool,
    pub gamma_open: bool,
    pub signed_mix_open: bool,
    pub stereo_link_open: bool,
    pub two_svf_g: f32,
    pub rect_sign: f32,
    pub sag_depth: f32,
    pub adaa_drive: f32,
    pub lru_bias_drive: f32,
    pub wf_mix: f32,
    pub wf_open: bool,
    pub svf_g: f32,
    pub glu_mix_all: [f32; 4],
}

impl LaneParams {
    pub(super) fn extract(
        p: &CoreQuadParamsStereo4<GpuVec4>,
        svf_g_arr: [f32; 4],
        c: usize,
    ) -> Self {
        let rel_raw = p.lru.alpha_rel.to_array()[c];
        let atk_raw = p.lru.alpha_atk.to_array()[c];
        let kap_raw = p.lru.kappa.to_array()[c];
        let gam_raw = p.lru.gamma_duhem.to_array()[c];
        let eta_raw = p.lru.signed_mix.to_array()[c];
        let link_raw = p.lru.stereo_link.to_array()[c];
        let wf_raw = p.wavefolder_mix.to_array()[c];
        Self {
            a1: p.svf.a1.to_array()[c],
            a2: p.svf.a2.to_array()[c],
            a3: p.svf.a3.to_array()[c],
            k: p.svf.k.to_array()[c],
            two_k: p.svf.two_k.to_array()[c],
            w_lp: p.tone.lp.to_array()[c],
            w_bp: p.tone.bp.to_array()[c],
            w_hp: p.tone.hp.to_array()[c],
            w_ap: p.tone.ap.to_array()[c],
            w_dir: p.tone.dir.to_array()[c],
            alpha_rel: clamp_nonneg_safe(rel_raw, MAX_PARALRU_R_F32),
            alpha_atk: clamp_nonneg_safe(atk_raw, 1.0),
            kappa: clamp_nonneg_safe(kap_raw, 1.0),
            gamma_duhem: clamp_nonneg_safe(gam_raw, MAX_PARALRU_DUHEM_GAMMA_F32),
            signed_mix: clamp_nonneg_safe(eta_raw, 1.0),
            stereo_link: clamp_nonneg_safe(link_raw, 1.0),
            alpha_rel_open: is_finite_f32(rel_raw) && (0.0..=MAX_PARALRU_R_F32).contains(&rel_raw),
            alpha_atk_open: is_finite_f32(atk_raw) && (0.0..=1.0).contains(&atk_raw),
            kappa_open: is_finite_f32(kap_raw) && (0.0..=1.0).contains(&kap_raw),
            gamma_open: is_finite_f32(gam_raw)
                && (0.0..=MAX_PARALRU_DUHEM_GAMMA_F32).contains(&gam_raw),
            signed_mix_open: is_finite_f32(eta_raw) && (0.0..=1.0).contains(&eta_raw),
            stereo_link_open: is_finite_f32(link_raw) && (0.0..=1.0).contains(&link_raw),
            two_svf_g: p.two_svf_g.to_array()[c],
            rect_sign: p.rect_sign.to_array()[c],
            sag_depth: p.sag_depth.to_array()[c],
            adaa_drive: p.adaa_drive.to_array()[c],
            lru_bias_drive: p.lru_bias_drive.to_array()[c],
            wf_mix: clamp_nonneg_safe(wf_raw, 1.0),
            wf_open: is_finite_f32(wf_raw) && (0.0..=1.0).contains(&wf_raw),
            svf_g: svf_g_arr[c],
            glu_mix_all: p.glu_mix.to_array(),
        }
    }

    pub(super) fn svf_coeffs(&self) -> TptSvfCoeffs4<GpuVec4> {
        TptSvfCoeffs4 {
            a1: GpuVec4::splat(self.a1),
            a2: GpuVec4::splat(self.a2),
            a3: GpuVec4::splat(self.a3),
            k: GpuVec4::splat(self.k),
            two_k: GpuVec4::splat(self.two_k),
        }
    }

    pub(super) fn lru_coeffs(&self) -> SonaParaLruDuhemCoeffs4<GpuVec4> {
        SonaParaLruDuhemCoeffs4 {
            alpha_rel: GpuVec4::splat(self.alpha_rel),
            alpha_atk: GpuVec4::splat(self.alpha_atk),
            kappa: GpuVec4::splat(self.kappa),
            gamma_duhem: GpuVec4::splat(self.gamma_duhem),
            signed_mix: GpuVec4::splat(self.signed_mix),
            stereo_link: GpuVec4::splat(self.stereo_link),
        }
    }
}

#[derive(Copy, Clone, Default)]
pub(super) struct FwdFrameState {
    pub u_l: f32,
    pub u_r: f32,
    pub s1_prev_l: f32,
    pub s2_prev_l: f32,
    pub s1_prev_r: f32,
    pub s2_prev_r: f32,
    pub taps_l: [f32; 4],
    pub taps_r: [f32; 4],
    pub tone_l: f32,
    pub tone_r: f32,
    pub v_prev_l: f32,
    pub v_prev_r: f32,
    pub v_raw_l: f32,
    pub v_raw_r: f32,
    pub v_eff_l: f32,
    pub v_eff_r: f32,
    pub sag_raw_l: f32,
    pub sag_raw_r: f32,
    pub sag_l: f32,
    pub sag_r: f32,
    pub r_prev_l: f32,
    pub r_prev_r: f32,
    pub r_curr_l: f32,
    pub r_curr_r: f32,
}

#[inline(always)]
pub(super) fn step_lane_forward(
    lp: &LaneParams,
    svf_c: &TptSvfCoeffs4<GpuVec4>,
    lru_c: &SonaParaLruDuhemCoeffs4<GpuVec4>,
    u_l: f32,
    u_r: f32,
    state: &mut [f32; 8],
) -> FwdFrameState {
    let [
        s1_prev_l,
        s2_prev_l,
        s1_prev_r,
        s2_prev_r,
        v_prev_l,
        v_prev_r,
        r_prev_l,
        r_prev_r,
    ] = *state;
    let mut s1_l = GpuVec4::splat(s1_prev_l);
    let mut s2_l = GpuVec4::splat(s2_prev_l);
    let svf_l = step_tpt_svf_vec4(GpuVec4::splat(u_l), &mut s1_l, &mut s2_l, svf_c);
    let mut s1_r = GpuVec4::splat(s1_prev_r);
    let mut s2_r = GpuVec4::splat(s2_prev_r);
    let svf_r = step_tpt_svf_vec4(GpuVec4::splat(u_r), &mut s1_r, &mut s2_r, svf_c);
    let taps_l = [
        svf_l.lp.to_array()[0],
        svf_l.bp.to_array()[0],
        svf_l.hp.to_array()[0],
        svf_l.ap.to_array()[0],
    ];
    let taps_r = [
        svf_r.lp.to_array()[0],
        svf_r.bp.to_array()[0],
        svf_r.hp.to_array()[0],
        svf_r.ap.to_array()[0],
    ];
    let tone_l = (((taps_l[0] * lp.w_lp + taps_l[1] * lp.w_bp) + taps_l[2] * lp.w_hp)
        + taps_l[3] * lp.w_ap)
        + u_l * lp.w_dir;
    let tone_r = (((taps_r[0] * lp.w_lp + taps_r[1] * lp.w_bp) + taps_r[2] * lp.w_hp)
        + taps_r[3] * lp.w_ap)
        + u_r * lp.w_dir;

    let mut vl = GpuVec4::splat(v_prev_l);
    let mut vr = GpuVec4::splat(v_prev_r);
    let raw_l = step_paralru_duhem_vec4(
        GpuVec4::splat(lp.rect_sign * tone_l),
        GpuVec4::splat(lp.two_svf_g * taps_l[1]),
        &mut vl,
        lru_c,
    );
    let raw_r = step_paralru_duhem_vec4(
        GpuVec4::splat(lp.rect_sign * tone_r),
        GpuVec4::splat(lp.two_svf_g * taps_r[1]),
        &mut vr,
        lru_c,
    );
    let (eff_l, eff_r) = apply_stereo_link_vec4_pair(raw_l, raw_r, lru_c.stereo_link);
    let (v_raw_l, v_raw_r) = (raw_l.to_array()[0], raw_r.to_array()[0]);
    let (v_eff_l, v_eff_r) = (eff_l.to_array()[0], eff_r.to_array()[0]);
    let sag_raw_l = 1.0 - lp.sag_depth * v_eff_l;
    let sag_raw_r = 1.0 - lp.sag_depth * v_eff_r;
    let sag_l = sag_raw_l.max(MIN_SAG_GAIN);
    let sag_r = sag_raw_r.max(MIN_SAG_GAIN);
    let r_curr_l = ((tone_l * lp.adaa_drive) * sag_l + lp.lru_bias_drive * v_eff_l) + 0.0;
    let r_curr_r = ((tone_r * lp.adaa_drive) * sag_r + lp.lru_bias_drive * v_eff_r) + 0.0;

    *state = [
        s1_l.to_array()[0],
        s2_l.to_array()[0],
        s1_r.to_array()[0],
        s2_r.to_array()[0],
        v_raw_l,
        v_raw_r,
        scrub_endpoint(r_curr_l),
        scrub_endpoint(r_curr_r),
    ];

    FwdFrameState {
        u_l,
        u_r,
        s1_prev_l,
        s2_prev_l,
        s1_prev_r,
        s2_prev_r,
        taps_l,
        taps_r,
        tone_l,
        tone_r,
        v_prev_l,
        v_prev_r,
        v_raw_l,
        v_raw_r,
        v_eff_l,
        v_eff_r,
        sag_raw_l,
        sag_raw_r,
        sag_l,
        sag_r,
        r_prev_l,
        r_prev_r,
        r_curr_l,
        r_curr_r,
    }
}

#[derive(Copy, Clone)]
pub(super) struct DuhemTerms {
    pub flux_abs: f32,
    pub rho: f32,
    pub rho_open: bool,
    pub d: f32,
    pub one_minus_eta: f32,
    pub phi_rect: f32,
    pub psi_1: f32,
    pub phi_eta_prime: f32,
    pub jac_div_rel: f32,
    pub jac: f32,
    pub sign_bp: f32,
    pub v_clean: f32,
    pub bp_clean: f32,
}

#[inline(always)]
fn eval_duhem_terms(u_rect: f32, bp: f32, v_prev: f32, lp: &LaneParams) -> DuhemTerms {
    let u_clean = canonicalize_zero_f32_bits(u_rect);
    let bp_clean = canonicalize_zero_f32_bits(bp);
    let v_clean = canonicalize_zero_f32_bits(v_prev);
    let flux_abs = canonicalize_zero_f32_bits(lp.two_svf_g * bp_clean).abs();
    let duhem_mod = if lp.gamma_duhem <= 0.0 {
        0.0
    } else {
        lp.gamma_duhem * flux_abs
    };
    let rho_raw = (1.0 - lp.alpha_atk) + duhem_mod;
    let rho = rho_raw.clamp(0.0, 1.0);
    let d = lp.alpha_rel * v_clean;
    let one_minus_eta = 1.0 - lp.signed_mix;
    let z = u_clean - lp.kappa * one_minus_eta * d;
    let (phi_rect, phi_rect_prime) = soft_rect_zero_dc_f32(z);
    let z_bar = z.clamp(-1.0, 1.0);
    let psi_1 = z_bar * (1.0 - ONE_THIRD_F32 * (z_bar * z_bar));
    let psi_1_prime = if z.abs() < 1.0 { 1.0 - z * z } else { 0.0 };
    let phi_eta_prime = if lp.signed_mix <= 0.0 {
        phi_rect_prime
    } else {
        one_minus_eta * phi_rect_prime + lp.signed_mix * psi_1_prime
    };
    let jac_div_rel = (1.0 - lp.signed_mix * rho) - rho * lp.kappa * one_minus_eta * phi_eta_prime;
    DuhemTerms {
        flux_abs,
        rho,
        rho_open: rho > 0.0 && rho < 1.0,
        d,
        one_minus_eta,
        phi_rect,
        psi_1,
        phi_eta_prime,
        jac_div_rel,
        jac: lp.alpha_rel * jac_div_rel,
        sign_bp: bp_clean.signum() * if bp_clean == 0.0 { 0.0 } else { 1.0 },
        v_clean,
        bp_clean,
    }
}

#[inline(always)]
fn duhem_channel_grad(
    lam: f32,
    t: &DuhemTerms,
    adj_tone_from_r: f32,
    tone: f32,
    u: f32,
    taps: [f32; 4],
    lp: &LaneParams,
) -> (f32, f32, f32, [f32; 14]) {
    let adj_u_rect = lam * t.rho * t.phi_eta_prime;
    let adj_tone = adj_tone_from_r + adj_u_rect * lp.rect_sign;
    let phi_eta = if lp.signed_mix <= 0.0 {
        t.phi_rect
    } else {
        t.one_minus_eta * t.phi_rect + lp.signed_mix * t.psi_1
    };
    let adj_rho = lam * (phi_eta - lp.signed_mix * t.d);
    let (mut g_atk, mut g_gam, mut g_two_g, mut adj_bp_duhem) = (0.0, 0.0, 0.0, 0.0);
    if t.rho_open {
        if lp.alpha_atk_open {
            g_atk = -adj_rho;
        }
        if lp.gamma_open {
            g_gam = adj_rho * t.flux_abs;
        }
        if lp.gamma_duhem > 0.0 {
            adj_bp_duhem = adj_rho * lp.two_svf_g * lp.gamma_duhem * t.sign_bp;
            g_two_g = adj_rho * lp.gamma_duhem * t.bp_clean.abs();
        }
    }
    let g_rel = if lp.alpha_rel_open {
        lam * t.v_clean * t.jac_div_rel
    } else {
        0.0
    };
    let g_kap = if lp.kappa_open {
        -lam * t.rho * t.one_minus_eta * t.d * t.phi_eta_prime
    } else {
        0.0
    };
    let g_eta = if lp.signed_mix_open {
        lam * t.rho * ((t.psi_1 - t.phi_rect) + lp.kappa * t.d * t.phi_eta_prime - t.d)
    } else {
        0.0
    };
    let adj_lp = adj_tone * lp.w_lp;
    let adj_bp = adj_tone * lp.w_bp + adj_bp_duhem;
    let adj_hp = adj_tone * lp.w_hp;
    let adj_ap = adj_tone * lp.w_ap;
    let adj_u_nonrec = adj_tone * lp.w_dir + adj_hp + adj_ap;
    let adj_v1 = adj_bp - lp.k * adj_hp - lp.two_k * adj_ap;
    let adj_v2 = adj_lp - adj_hp;
    let grads = [
        g_rel,
        g_atk,
        g_kap,
        g_gam,
        g_eta,
        g_two_g,
        adj_u_rect * tone,
        adj_tone * taps[0],
        adj_tone * taps[1],
        adj_tone * taps[2],
        adj_tone * taps[3],
        adj_tone * u,
        -adj_hp * taps[1],
        -adj_ap * taps[1],
    ];
    (adj_v1, adj_v2, adj_u_nonrec, grads)
}

#[derive(Copy, Clone, Default)]
pub(super) struct ReverseCarry {
    pub adj_r_next_l: f32,
    pub adj_r_next_r: f32,
    pub lam_after_l: f32,
    pub lam_after_r: f32,
    pub adj_s_out_l: [f32; 2],
    pub adj_s_out_r: [f32; 2],
}

#[derive(Copy, Clone, Default)]
pub(super) struct TileGradAccum {
    pub slots: [f32; 24],
}

/// Shared immutable trajectory slices and scalar coordinates for [`step_lane_reverse`].
pub(super) struct ReverseStepContext<'a> {
    pub d_loss_d_features: &'a [[f32; 4]],
    pub feature_outputs: &'a [[f32; 4]],
    pub q_global: usize,
    pub num_frames: usize,
    pub c: usize,
    pub tiny_eps: f32,
}

/// Executes one reverse BPTT frame at `(q_global, frame)` on lane `c`, returning `(du_l, du_r)`.
pub(super) fn step_lane_reverse(
    ctx: &ReverseStepContext<'_>,
    fwd: &FwdFrameState,
    lp: &LaneParams,
    frame: usize,
    carry: &mut ReverseCarry,
    acc: &mut TileGradAccum,
) -> (f32, f32) {
    let c = ctx.c;
    let feat_base = (ctx.q_global * ctx.num_frames + frame) * FEATURE_VEC4_STRIDE;
    let c_next = (c + 1) & 3;
    let c_prev = (c + 3) & 3;
    let (gm_c, gm_prev) = (lp.glu_mix_all[c], lp.glu_mix_all[c_prev]);
    let dl_veff_l = ctx.d_loss_d_features[feat_base][c];
    let dl_veff_r = ctx.d_loss_d_features[feat_base + 1][c];
    let dl_odd_l = ctx.d_loss_d_features[feat_base + 2][c];
    let dl_odd_r = ctx.d_loss_d_features[feat_base + 3][c];
    let dl_even_l = ctx.d_loss_d_features[feat_base + 4];
    let dl_even_r = ctx.d_loss_d_features[feat_base + 5];
    let fwd_odd_l = ctx.feature_outputs[feat_base + 2];
    let fwd_odd_r = ctx.feature_outputs[feat_base + 3];

    let adj_odd_wf_l = dl_odd_l
        + dl_even_l[c] * gm_c * fwd_odd_l[c_next]
        + dl_even_l[c_prev] * gm_prev * fwd_odd_l[c_prev];
    let adj_odd_wf_r = dl_odd_r
        + dl_even_r[c] * gm_c * fwd_odd_r[c_next]
        + dl_even_r[c_prev] * gm_prev * fwd_odd_r[c_prev];
    acc.slots[22] += dl_even_l[c] * fwd_odd_l[c] * fwd_odd_l[c_next]
        + dl_even_r[c] * fwd_odd_r[c] * fwd_odd_r[c_next];

    let dl = eval_adaa_wf_derivs(fwd.r_prev_l, fwd.r_curr_l, ctx.tiny_eps);
    let dr = eval_adaa_wf_derivs(fwd.r_prev_r, fwd.r_curr_r, ctx.tiny_eps);
    let omw = 1.0 - lp.wf_mix;
    let adj_r_l = (omw * dl.d_odd_db + lp.wf_mix * dl.d_wf_db) * adj_odd_wf_l
        + dl.d_even_db * dl_even_l[c]
        + carry.adj_r_next_l;
    let adj_r_r = (omw * dr.d_odd_db + lp.wf_mix * dr.d_wf_db) * adj_odd_wf_r
        + dr.d_even_db * dl_even_r[c]
        + carry.adj_r_next_r;
    carry.adj_r_next_l =
        (omw * dl.d_odd_da + lp.wf_mix * dl.d_wf_da) * adj_odd_wf_l + dl.d_even_da * dl_even_l[c];
    carry.adj_r_next_r =
        (omw * dr.d_odd_da + lp.wf_mix * dr.d_wf_da) * adj_odd_wf_r + dr.d_even_da * dl_even_r[c];
    if lp.wf_open {
        acc.slots[21] +=
            adj_odd_wf_l * (dl.wf_odd - dl.odd_poly) + adj_odd_wf_r * (dr.wf_odd - dr.odd_poly);
    }

    let sag_open_l = fwd.sag_raw_l > MIN_SAG_GAIN;
    let sag_open_r = fwd.sag_raw_r > MIN_SAG_GAIN;
    let adj_tone_r_l = adj_r_l * lp.adaa_drive * fwd.sag_l;
    let adj_tone_r_r = adj_r_r * lp.adaa_drive * fwd.sag_r;
    let adj_veff_l = dl_veff_l + adj_r_l * lp.lru_bias_drive
        - if sag_open_l {
            adj_r_l * fwd.tone_l * lp.adaa_drive * lp.sag_depth
        } else {
            0.0
        };
    let adj_veff_r = dl_veff_r + adj_r_r * lp.lru_bias_drive
        - if sag_open_r {
            adj_r_r * fwd.tone_r * lp.adaa_drive * lp.sag_depth
        } else {
            0.0
        };
    let half_lam = 0.5 * lp.stereo_link;
    let adj_v_raw_l = (1.0 - half_lam) * adj_veff_l + half_lam * adj_veff_r;
    let adj_v_raw_r = (1.0 - half_lam) * adj_veff_r + half_lam * adj_veff_l;

    acc.slots[18] += if sag_open_l {
        -adj_r_l * fwd.tone_l * lp.adaa_drive * fwd.v_eff_l
    } else {
        0.0
    } + if sag_open_r {
        -adj_r_r * fwd.tone_r * lp.adaa_drive * fwd.v_eff_r
    } else {
        0.0
    };
    acc.slots[19] += adj_r_l * fwd.tone_l * fwd.sag_l + adj_r_r * fwd.tone_r * fwd.sag_r;
    acc.slots[20] += adj_r_l * fwd.v_eff_l + adj_r_r * fwd.v_eff_r;
    if lp.stereo_link_open {
        let v_mid = 0.5 * (fwd.v_raw_l + fwd.v_raw_r);
        acc.slots[15] += adj_veff_l * (v_mid - fwd.v_raw_l) + adj_veff_r * (v_mid - fwd.v_raw_r);
    }

    let tl = eval_duhem_terms(lp.rect_sign * fwd.tone_l, fwd.taps_l[1], fwd.v_prev_l, lp);
    let tr = eval_duhem_terms(lp.rect_sign * fwd.tone_r, fwd.taps_r[1], fwd.v_prev_r, lp);
    let lam_l = adj_v_raw_l + carry.lam_after_l;
    let lam_r = adj_v_raw_r + carry.lam_after_r;
    carry.lam_after_l = tl.jac * lam_l;
    carry.lam_after_r = tr.jac * lam_r;

    let (adj_v1_l, adj_v2_l, adj_u_nonrec_l, gl) = duhem_channel_grad(
        lam_l,
        &tl,
        adj_tone_r_l,
        fwd.tone_l,
        fwd.u_l,
        fwd.taps_l,
        lp,
    );
    let (adj_v1_r, adj_v2_r, adj_u_nonrec_r, gr) = duhem_channel_grad(
        lam_r,
        &tr,
        adj_tone_r_r,
        fwd.tone_r,
        fwd.u_r,
        fwd.taps_r,
        lp,
    );
    let grad_map = [10, 11, 12, 13, 14, 16, 17, 5, 6, 7, 8, 9, 3, 4];
    for (idx, &slot) in grad_map.iter().enumerate() {
        acc.slots[slot] += gl[idx] + gr[idx];
    }

    let tot_v1_l = adj_v1_l + 2.0 * carry.adj_s_out_l[0];
    let tot_v2_l = adj_v2_l + 2.0 * carry.adj_s_out_l[1];
    let tot_v1_r = adj_v1_r + 2.0 * carry.adj_s_out_r[0];
    let tot_v2_r = adj_v2_r + 2.0 * carry.adj_s_out_r[1];
    let v3_l = fwd.u_l - fwd.s2_prev_l;
    let v3_r = fwd.u_r - fwd.s2_prev_r;
    acc.slots[0] += tot_v1_l * fwd.s1_prev_l + tot_v1_r * fwd.s1_prev_r;
    acc.slots[1] +=
        (tot_v1_l * v3_l + tot_v2_l * fwd.s1_prev_l) + (tot_v1_r * v3_r + tot_v2_r * fwd.s1_prev_r);
    acc.slots[2] += tot_v2_l * v3_l + tot_v2_r * v3_r;

    let du_l = adj_u_nonrec_l + lp.a2 * tot_v1_l + lp.a3 * tot_v2_l;
    let du_r = adj_u_nonrec_r + lp.a2 * tot_v1_r + lp.a3 * tot_v2_r;
    carry.adj_s_out_l = dnam_kernels::apply_tpt_adjoint(
        build_tpt_elem(lp.a1, lp.a2, lp.a3, adj_v1_l, adj_v2_l),
        carry.adj_s_out_l,
    );
    carry.adj_s_out_r = dnam_kernels::apply_tpt_adjoint(
        build_tpt_elem(lp.a1, lp.a2, lp.a3, adj_v1_r, adj_v2_r),
        carry.adj_s_out_r,
    );
    (du_l, du_r)
}

#[inline(always)]
fn build_tpt_elem(a1: f32, a2: f32, a3: f32, adj_v1: f32, adj_v2: f32) -> TptAdjointElem {
    TptAdjointElem {
        t00: (a1 + a1) - 1.0,
        t01: a2 + a2,
        t10: -(a2 + a2),
        t11: 1.0 - (a3 + a3),
        g0: a1 * adj_v1 + a2 * adj_v2,
        g1: -a2 * adj_v1 + (1.0 - a3) * adj_v2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::abi::FEATURE_VEC4_STRIDE;

    #[test]
    fn test_adjoint_clamp_duhem_and_closed_boundary_guards() {
        assert_eq!(clamp_nonneg_safe(-1.0, 2.0), 0.0);
        assert_eq!(clamp_nonneg_safe(5.0, 2.0), 2.0);
        assert_eq!(clamp_nonneg_safe(f32::INFINITY, 2.0), 2.0);
        assert_eq!(clamp_nonneg_safe(f32::NAN, 2.0), 0.0);
        assert_eq!(flush_zero(-0.0), 0.0);

        // Exercise closed clamp bounds (`!alpha_*_open`, `!wf_open`, `sag <= MIN_SAG_GAIN`, `gamma = 0`).
        let mut lp = LaneParams::extract(
            &CoreQuadParamsStereo4 {
                svf: TptSvfCoeffs4::from_gk(GpuVec4::splat(0.1), GpuVec4::splat(0.8)),
                tone: dnam_kernels::SonaToneWeights4::lp_only(),
                lru: SonaParaLruDuhemCoeffs4::from_clamped(
                    GpuVec4::splat(0.8),
                    GpuVec4::splat(1.0),
                    GpuVec4::splat(0.0),
                    GpuVec4::splat(0.0),
                    GpuVec4::splat(0.0),
                    GpuVec4::splat(0.0),
                ),
                two_svf_g: GpuVec4::splat(0.2),
                rect_sign: GpuVec4::splat(1.0),
                sag_depth: GpuVec4::splat(4.0),
                adaa_drive: GpuVec4::splat(1.0),
                lru_bias_drive: GpuVec4::splat(0.0),
                wavefolder_mix: GpuVec4::splat(-0.5),
                glu_mix: GpuVec4::splat(0.0),
            },
            [0.1; 4],
            0,
        );
        lp.alpha_rel_open = false;
        lp.alpha_atk_open = false;
        lp.kappa_open = false;
        lp.gamma_open = false;
        lp.signed_mix_open = false;
        lp.wf_open = false;
        lp.stereo_link_open = false;

        let t0 = eval_duhem_terms(0.0, 0.0, 0.2, &lp);
        assert_eq!(t0.flux_abs, 0.0);
        assert_eq!(t0.sign_bp, 0.0);
        assert!(!t0.rho_open);

        let mut fwd = FwdFrameState {
            sag_raw_l: 0.0,
            sag_raw_r: 0.0,
            sag_l: MIN_SAG_GAIN,
            sag_r: MIN_SAG_GAIN,
            tone_l: 0.25,
            tone_r: -0.25,
            ..FwdFrameState::default()
        };
        fwd.r_curr_l = 0.2;
        fwd.r_curr_r = -0.2;

        let dl_f = vec![[0.5_f32; 4]; FEATURE_VEC4_STRIDE];
        let feats = vec![[0.1_f32; 4]; FEATURE_VEC4_STRIDE];
        let mut carry = ReverseCarry::default();
        let mut acc = TileGradAccum::default();
        let ctx = ReverseStepContext {
            d_loss_d_features: &dl_f,
            feature_outputs: &feats,
            q_global: 0,
            num_frames: 1,
            c: 0,
            tiny_eps: 1.0e-6,
        };
        let (du_l, du_r) = step_lane_reverse(&ctx, &fwd, &lp, 0, &mut carry, &mut acc);
        assert!(du_l.is_finite() && du_r.is_finite());
        assert_eq!(acc.slots[10], 0.0);
        assert_eq!(acc.slots[15], 0.0);
        assert_eq!(acc.slots[18], 0.0);
        assert_eq!(acc.slots[21], 0.0);
    }
}
