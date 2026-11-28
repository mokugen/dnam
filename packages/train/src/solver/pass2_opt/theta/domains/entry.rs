//! Physical <-> optimization-coordinate adapters ([`ThetaEntry`]) mapping every
//! [`ParamKind`] between `SonaModelF64` fields, conditioned coordinate domains
//! (`log(fc / fs)` for `CoreSvfG`/`SenseSvfG`), wire quantization masks, and `Pass2ParamGrads`.

use core::f64::consts::{FRAC_1_PI, PI};

use dnam_kernels::MIN_TPT_G;

use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::super::super::model::SonaModelF64;
use super::super::super::eval::Pass2ParamGrads;
use super::ParamKind;

/// One parameter slot descriptor linking physical model storage, coordinate transform,
/// wire quantization flag, and adjoint gradient slot.
#[derive(Clone, Copy, Debug)]
pub(in crate::solver::pass2_opt) struct ThetaEntry {
    pub kind: ParamKind,
    pub bound: ParamBoundSpec,
    pub is_fp16_wire: bool,
}

impl ThetaEntry {
    #[inline]
    pub(super) fn f32(kind: ParamKind, bound: ParamBoundSpec) -> Self {
        Self {
            kind,
            bound,
            is_fp16_wire: false,
        }
    }

    #[inline]
    pub(super) fn fp16(kind: ParamKind, bound: ParamBoundSpec) -> Self {
        Self {
            kind,
            bound,
            is_fp16_wire: true,
        }
    }

    #[must_use]
    pub(in super::super) fn read(&self, m: &SonaModelF64) -> f64 {
        match self.kind {
            ParamKind::CoreSvfG(q, l) => m.core_quads[q].svf_g[l].atan() * FRAC_1_PI,
            ParamKind::CoreSvfK(q, l) => m.core_quads[q].svf_k[l],
            ParamKind::CoreSvfKNl(q, l) => m.core_quads[q].svf_k_nl[l],
            ParamKind::CoreToneLp(q, l) => m.core_quads[q].tone_lp[l],
            ParamKind::CoreToneBp(q, l) => m.core_quads[q].tone_bp[l],
            ParamKind::CoreToneHp(q, l) => m.core_quads[q].tone_hp[l],
            ParamKind::CoreToneAp(q, l) => m.core_quads[q].tone_ap[l],
            ParamKind::CoreToneDir(q, l) => m.core_quads[q].tone_dir[l],
            ParamKind::CoreLruRel(q, l) => m.core_quads[q].lru_alpha_rel[l],
            ParamKind::CoreLruAtk(q, l) => m.core_quads[q].lru_alpha_atk[l],
            ParamKind::CoreLruKap(q, l) => m.core_quads[q].lru_kappa[l],
            ParamKind::CoreLruDuhem(q, l) => m.core_quads[q].lru_duhem_gamma[l],
            ParamKind::CoreLruSignedMix(q, l) => m.core_quads[q].lru_signed_mix[l],
            ParamKind::CoreLruStereoLink(q, l) => m.core_quads[q].lru_stereo_link[l],
            ParamKind::CoreSagDepth(q, l) => m.core_quads[q].sag_depth[l],
            ParamKind::CoreAdaaDrive(q, l) => m.core_quads[q].adaa_drive[l],
            ParamKind::CoreLruBias(q, l) => m.core_quads[q].lru_bias_drive[l],
            ParamKind::CoreWavefolder(q, l) => m.core_quads[q].wavefolder_mix[l],
            ParamKind::CoreGluMix(q, l) => m.core_quads[q].glu_mix[l],
            ParamKind::CoreG0(q, l) => m.core_quads[q].g0[l],
            ParamKind::CoreGInf(q, l) => m.core_quads[q].g_inf[l],
            ParamKind::CoreDPre(q, l) => m.core_quads[q].d_pre[l],
            ParamKind::CoreSeWSq(q, l) => m.core_quads[q].se_w_sq[l],
            ParamKind::CoreSeDBus(q, l) => m.core_quads[q].se_d_bus[l],
            ParamKind::CoreLinMix(q, l) => m.core_quads[q].lin_mix[l],
            ParamKind::CoreOutGain(q, l) => m.core_quads[q].out_gain[l],
            ParamKind::CoreInMid(q, l) => m.core_quads[q].in_mid_weight[l],
            ParamKind::CoreInSide(q, l) => m.core_quads[q].in_side_weight[l],
            ParamKind::TileOdd(t, s, d) => m.tiles[t].w_odd_4x4[s][d],
            ParamKind::TileEven(t, s, d) => m.tiles[t].w_even_4x4[s][d],
            ParamKind::TileLru(t, s, d) => m.tiles[t].w_lru_4x4[s][d],
            ParamKind::SenseSvfG(l) => m.sense_svf_g[l].atan() * FRAC_1_PI,
            ParamKind::SenseLruRel(l) => m.sense_lru_rel[l],
            ParamKind::SenseLruAtk(l) => m.sense_lru_atk[l],
            ParamKind::SenseSvfK(l) => m.sense_svf_k[l],
            ParamKind::SenseToneLp(l) => m.sense_tone_lp[l],
            ParamKind::SenseToneBp(l) => m.sense_tone_bp[l],
            ParamKind::SenseToneHp(l) => m.sense_tone_hp[l],
            ParamKind::SenseLruKappa(l) => m.sense_lru_kappa[l],
            ParamKind::SenseVcaW(l) => m.sense_vca_w[l],
            ParamKind::SenseInL(l) => m.sense_in_l[l],
            ParamKind::SenseInR(l) => m.sense_in_r[l],
            ParamKind::PhasorFreqHz(l) => m.phasor_freq_hz[l],
            ParamKind::CrossbarOpGain(c) => m.crossbar_op_gains[c],
            ParamKind::CarrierBasisW(b, l) => m.carrier_basis_w[b][l],
            ParamKind::OpPToCore(q, l) => m.op_p_to_core[q][l],
            ParamKind::PhasorToCore(q, l) => m.phasor_to_core[q][l],
            ParamKind::PhasorStereoPhase(l) => m.phasor_stereo_phase[l],
            ParamKind::OpNPreGammaBc => m.op_n_pre_gamma_bc,
            ParamKind::OpNPreZohHoldDepth => m.op_n_pre_zoh_hold_depth,
            ParamKind::OpNPreBetaQ => m.op_n_pre_beta_q,
            ParamKind::RoPole48k => m.ro_pole_48k,
            ParamKind::KautzRLow(k) => {
                if m.kautz[k].r_low > 0.0 && m.kautz[k].b_gamma > 0.0 {
                    m.kautz[k].r_low
                } else {
                    0.0
                }
            }
            ParamKind::KautzTheta(k) => m.kautz[k].theta,
            ParamKind::OpTDelay(q, l) => m.transport_quads[q].delay_samples[l],
            ParamKind::OpTRampSpan(q, l) => m.transport_quads[q].ramp_spans_samples[l],
            ParamKind::OpTGLoop(q) => m.transport_quads[q].g_loop,
            ParamKind::OpTRDc(q) => m.transport_quads[q].r_dc,
            ParamKind::OpTDampPole(q) => m.transport_quads[q].damp_pole,
            ParamKind::OpTLfoRate(q) => m.transport_quads[q].lfo_rate_hz,
            ParamKind::OpTRampRate(q) => m.transport_quads[q].ramp_rate_hz,
            ParamKind::OpTLfoDepth(q, l) => m.transport_quads[q].lfo_depths[l],
            ParamKind::OpTInputGain(q, l) => m.transport_quads[q].input_gains[l],
            ParamKind::OpTAllpassG(q, l) => m.transport_quads[q].allpass_g[l],
            ParamKind::OpTFbPsi(q) => m.transport_quads[q].fb_psi,
            ParamKind::OpTSigmaKnee(q) => m.transport_quads[q].sigma_knee,
            ParamKind::OpTFreezeMix(q) => m.transport_quads[q].freeze_mix,
            ParamKind::OpTWindowDepth(q) => m.transport_quads[q].window_depth,
            ParamKind::OpTScatPsi(q) => m.transport_quads[q].scat_psi,
            ParamKind::OptToLr(i, c) => m.opt_to_lr[i][c],
            ParamKind::KautzBGamma(k) => m.kautz[k].b_gamma,
            ParamKind::OpTWetTapL(q, l) => m.transport_quads[q].wet_tap_l[l],
            ParamKind::OpTWetTapR(q, l) => m.transport_quads[q].wet_tap_r[l],
            ParamKind::TankDryMix => m.tank_dry_mix,
            ParamKind::WReadout(row, col) => m.w_ro[row][col],
            ParamKind::DirectS2U(c) => m.direct_s2_u_gain[c],
            ParamKind::DirectS2Hp(c) => m.direct_s2_hp_gain[c],
            ParamKind::PreConvU(c) => m.pre_conv_u_gain[c],
            ParamKind::PreConvHp(c) => m.pre_conv_hp_gain[c],
            ParamKind::KautzCReL(k) => m.kautz[k].c_re_l,
            ParamKind::KautzCImL(k) => m.kautz[k].c_im_l,
            ParamKind::KautzCReR(k) => m.kautz[k].c_re_r,
            ParamKind::KautzCImR(k) => m.kautz[k].c_im_r,
        }
    }

    pub(in super::super) fn write(&self, m: &mut SonaModelF64, raw: f64, kautz_mask: &[bool; 32]) {
        let clamped = if raw.is_finite() {
            self.bound.snap_zero_edge(self.bound.project_physical(raw))
        } else {
            self.bound.default_midpoint()
        };
        let g_lo = f64::from(MIN_TPT_G);
        match self.kind {
            ParamKind::CoreSvfG(q, l) => {
                m.core_quads[q].svf_g[l] = (PI * clamped).tan().clamp(g_lo, 64.0);
            }
            ParamKind::CoreSvfK(q, l) => m.core_quads[q].svf_k[l] = clamped,
            ParamKind::CoreSvfKNl(q, l) => m.core_quads[q].svf_k_nl[l] = clamped,
            ParamKind::CoreToneLp(q, l) => m.core_quads[q].tone_lp[l] = clamped,
            ParamKind::CoreToneBp(q, l) => m.core_quads[q].tone_bp[l] = clamped,
            ParamKind::CoreToneHp(q, l) => m.core_quads[q].tone_hp[l] = clamped,
            ParamKind::CoreToneAp(q, l) => m.core_quads[q].tone_ap[l] = clamped,
            ParamKind::CoreToneDir(q, l) => m.core_quads[q].tone_dir[l] = clamped,
            ParamKind::CoreLruRel(q, l) => m.core_quads[q].lru_alpha_rel[l] = clamped,
            ParamKind::CoreLruAtk(q, l) => m.core_quads[q].lru_alpha_atk[l] = clamped,
            ParamKind::CoreLruKap(q, l) => m.core_quads[q].lru_kappa[l] = clamped,
            ParamKind::CoreLruDuhem(q, l) => m.core_quads[q].lru_duhem_gamma[l] = clamped,
            ParamKind::CoreLruSignedMix(q, l) => m.core_quads[q].lru_signed_mix[l] = clamped,
            ParamKind::CoreLruStereoLink(q, l) => m.core_quads[q].lru_stereo_link[l] = clamped,
            ParamKind::CoreSagDepth(q, l) => m.core_quads[q].sag_depth[l] = clamped,
            ParamKind::CoreAdaaDrive(q, l) => m.core_quads[q].adaa_drive[l] = clamped,
            ParamKind::CoreLruBias(q, l) => m.core_quads[q].lru_bias_drive[l] = clamped,
            ParamKind::CoreWavefolder(q, l) => m.core_quads[q].wavefolder_mix[l] = clamped,
            ParamKind::CoreGluMix(q, l) => m.core_quads[q].glu_mix[l] = clamped,
            ParamKind::CoreG0(q, l) => m.core_quads[q].g0[l] = clamped,
            ParamKind::CoreGInf(q, l) => m.core_quads[q].g_inf[l] = clamped,
            ParamKind::CoreDPre(q, l) => m.core_quads[q].d_pre[l] = clamped,
            ParamKind::CoreSeWSq(q, l) => m.core_quads[q].se_w_sq[l] = clamped,
            ParamKind::CoreSeDBus(q, l) => m.core_quads[q].se_d_bus[l] = clamped,
            ParamKind::CoreLinMix(q, l) => m.core_quads[q].lin_mix[l] = clamped,
            ParamKind::CoreOutGain(q, l) => m.core_quads[q].out_gain[l] = clamped,
            ParamKind::CoreInMid(q, l) => m.core_quads[q].in_mid_weight[l] = clamped,
            ParamKind::CoreInSide(q, l) => m.core_quads[q].in_side_weight[l] = clamped,
            ParamKind::TileOdd(t, s, d) => m.tiles[t].w_odd_4x4[s][d] = clamped,
            ParamKind::TileEven(t, s, d) => m.tiles[t].w_even_4x4[s][d] = clamped,
            ParamKind::TileLru(t, s, d) => m.tiles[t].w_lru_4x4[s][d] = clamped,
            ParamKind::SenseSvfG(l) => {
                m.sense_svf_g[l] = (PI * clamped).tan().clamp(g_lo, 64.0);
            }
            ParamKind::SenseLruRel(l) => m.sense_lru_rel[l] = clamped,
            ParamKind::SenseLruAtk(l) => m.sense_lru_atk[l] = clamped,
            ParamKind::SenseSvfK(l) => m.sense_svf_k[l] = clamped,
            ParamKind::SenseToneLp(l) => m.sense_tone_lp[l] = clamped,
            ParamKind::SenseToneBp(l) => m.sense_tone_bp[l] = clamped,
            ParamKind::SenseToneHp(l) => m.sense_tone_hp[l] = clamped,
            ParamKind::SenseLruKappa(l) => m.sense_lru_kappa[l] = clamped,
            ParamKind::SenseVcaW(l) => m.sense_vca_w[l] = clamped,
            ParamKind::SenseInL(l) => m.sense_in_l[l] = clamped,
            ParamKind::SenseInR(l) => m.sense_in_r[l] = clamped,
            ParamKind::PhasorFreqHz(l) => m.phasor_freq_hz[l] = clamped,
            ParamKind::CrossbarOpGain(c) => m.crossbar_op_gains[c] = clamped,
            ParamKind::CarrierBasisW(b, l) => m.carrier_basis_w[b][l] = clamped,
            ParamKind::OpPToCore(q, l) => m.op_p_to_core[q][l] = clamped,
            ParamKind::PhasorToCore(q, l) => m.phasor_to_core[q][l] = clamped,
            ParamKind::PhasorStereoPhase(l) => m.phasor_stereo_phase[l] = clamped,
            ParamKind::OpNPreGammaBc => m.op_n_pre_gamma_bc = clamped,
            ParamKind::OpNPreZohHoldDepth => m.op_n_pre_zoh_hold_depth = clamped,
            ParamKind::OpNPreBetaQ => m.op_n_pre_beta_q = clamped,
            ParamKind::RoPole48k => m.ro_pole_48k = clamped,
            ParamKind::KautzRLow(k) => {
                if !kautz_mask[k] || m.kautz[k].r_low <= 0.0 || !m.kautz[k].r_low.is_finite() {
                    m.kautz[k].r_low = 0.0;
                    m.kautz[k].dr = 0.0;
                } else {
                    m.kautz[k].r_low = clamped;
                    m.kautz[k].dr = 0.0;
                }
            }
            ParamKind::KautzTheta(k) => m.kautz[k].theta = clamped,
            ParamKind::OpTDelay(q, l) => m.transport_quads[q].delay_samples[l] = clamped,
            ParamKind::OpTRampSpan(q, l) => m.transport_quads[q].ramp_spans_samples[l] = clamped,
            ParamKind::OpTGLoop(q) => m.transport_quads[q].g_loop = clamped,
            ParamKind::OpTRDc(q) => m.transport_quads[q].r_dc = clamped,
            ParamKind::OpTDampPole(q) => m.transport_quads[q].damp_pole = clamped,
            ParamKind::OpTLfoRate(q) => m.transport_quads[q].lfo_rate_hz = clamped,
            ParamKind::OpTRampRate(q) => m.transport_quads[q].ramp_rate_hz = clamped,
            ParamKind::OpTLfoDepth(q, l) => m.transport_quads[q].lfo_depths[l] = clamped,
            ParamKind::OpTInputGain(q, l) => m.transport_quads[q].input_gains[l] = clamped,
            ParamKind::OpTAllpassG(q, l) => m.transport_quads[q].allpass_g[l] = clamped,
            ParamKind::OpTFbPsi(q) => m.transport_quads[q].fb_psi = clamped,
            ParamKind::OpTSigmaKnee(q) => m.transport_quads[q].sigma_knee = clamped,
            ParamKind::OpTFreezeMix(q) => m.transport_quads[q].freeze_mix = clamped,
            ParamKind::OpTWindowDepth(q) => m.transport_quads[q].window_depth = clamped,
            ParamKind::OpTScatPsi(q) => m.transport_quads[q].scat_psi = clamped,
            ParamKind::OptToLr(i, c) => m.opt_to_lr[i][c] = clamped,
            ParamKind::KautzBGamma(k) => {
                if !kautz_mask[k] || m.kautz[k].r_low <= 0.0 {
                    m.kautz[k].b_gamma = 0.0;
                } else {
                    m.kautz[k].b_gamma = clamped;
                }
            }
            ParamKind::OpTWetTapL(q, l) => m.transport_quads[q].wet_tap_l[l] = clamped,
            ParamKind::OpTWetTapR(q, l) => m.transport_quads[q].wet_tap_r[l] = clamped,
            ParamKind::TankDryMix => m.tank_dry_mix = clamped,
            ParamKind::WReadout(row, col) => m.w_ro[row][col] = clamped,
            ParamKind::DirectS2U(c) => m.direct_s2_u_gain[c] = clamped,
            ParamKind::DirectS2Hp(c) => m.direct_s2_hp_gain[c] = clamped,
            ParamKind::PreConvU(c) => m.pre_conv_u_gain[c] = clamped,
            ParamKind::PreConvHp(c) => m.pre_conv_hp_gain[c] = clamped,
            ParamKind::KautzCReL(k) => m.kautz[k].c_re_l = clamped,
            ParamKind::KautzCImL(k) => m.kautz[k].c_im_l = clamped,
            ParamKind::KautzCReR(k) => m.kautz[k].c_re_r = clamped,
            ParamKind::KautzCImR(k) => m.kautz[k].c_im_r = clamped,
        }
    }

    #[must_use]
    pub(in super::super) fn read_grad(
        &self,
        m: &SonaModelF64,
        g: &Pass2ParamGrads,
        kautz_mask: &[bool; 32],
    ) -> f64 {
        let kz_grad = |k: usize, idx: usize| -> f64 {
            if kautz_mask[k] {
                g.d_kautz.get(k).map_or(0.0, |v| v[idx])
            } else {
                0.0
            }
        };
        match self.kind {
            ParamKind::CoreSvfG(q, l) => {
                let svf_g = m.core_quads[q].svf_g[l];
                let dg_dr = PI * (1.0 + svf_g * svf_g);
                g.core_quads.get(q).map_or(0.0, |c| c.svf_g[l]) * dg_dr
            }
            ParamKind::CoreSvfK(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.svf_k[l]),
            ParamKind::CoreSvfKNl(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.svf_k_nl[l]),
            ParamKind::CoreToneLp(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.tone_lp[l]),
            ParamKind::CoreToneBp(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.tone_bp[l]),
            ParamKind::CoreToneHp(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.tone_hp[l]),
            ParamKind::CoreToneAp(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.tone_ap[l]),
            ParamKind::CoreToneDir(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.tone_dir[l]),
            ParamKind::CoreLruRel(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.lru_alpha_rel[l]),
            ParamKind::CoreLruAtk(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.lru_alpha_atk[l]),
            ParamKind::CoreLruKap(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.lru_kappa[l]),
            ParamKind::CoreLruDuhem(q, l) => {
                g.core_quads.get(q).map_or(0.0, |c| c.lru_duhem_gamma[l])
            }
            ParamKind::CoreLruSignedMix(q, l) => {
                g.core_quads.get(q).map_or(0.0, |c| c.lru_signed_mix[l])
            }
            ParamKind::CoreLruStereoLink(q, l) => {
                g.core_quads.get(q).map_or(0.0, |c| c.lru_stereo_link[l])
            }
            ParamKind::CoreSagDepth(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.sag_depth[l]),
            ParamKind::CoreAdaaDrive(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.adaa_drive[l]),
            ParamKind::CoreLruBias(q, l) => {
                g.core_quads.get(q).map_or(0.0, |c| c.lru_bias_drive[l])
            }
            ParamKind::CoreWavefolder(q, l) => {
                g.core_quads.get(q).map_or(0.0, |c| c.wavefolder_mix[l])
            }
            ParamKind::CoreGluMix(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.glu_mix[l]),
            ParamKind::CoreG0(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.g0[l]),
            ParamKind::CoreGInf(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.g_inf[l]),
            ParamKind::CoreDPre(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.d_pre[l]),
            ParamKind::CoreSeWSq(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.se_w_sq[l]),
            ParamKind::CoreSeDBus(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.se_d_bus[l]),
            ParamKind::CoreLinMix(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.lin_mix[l]),
            ParamKind::CoreOutGain(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.out_gain[l]),
            ParamKind::CoreInMid(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.in_mid_weight[l]),
            ParamKind::CoreInSide(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.in_side_weight[l]),
            ParamKind::TileOdd(t, s, d) => g.tiles.get(t).map_or(0.0, |tl| tl.w_odd_4x4[s][d]),
            ParamKind::TileEven(t, s, d) => g.tiles.get(t).map_or(0.0, |tl| tl.w_even_4x4[s][d]),
            ParamKind::TileLru(t, s, d) => g.tiles.get(t).map_or(0.0, |tl| tl.w_lru_4x4[s][d]),
            ParamKind::SenseSvfG(l) => {
                let svf_g = m.sense_svf_g[l];
                let dg_dr = PI * (1.0 + svf_g * svf_g);
                g.sensory.d_svf_g[l] * dg_dr
            }
            ParamKind::SenseLruRel(l) => g.sensory.d_lru_rel[l],
            ParamKind::SenseLruAtk(l) => g.sensory.d_lru_atk[l],
            ParamKind::SenseSvfK(l) => g.sensory.d_svf_k[l],
            ParamKind::SenseToneLp(l) => g.sensory.d_tone_lp[l],
            ParamKind::SenseToneBp(l) => g.sensory.d_tone_bp[l],
            ParamKind::SenseToneHp(l) => g.sensory.d_tone_hp[l],
            ParamKind::SenseLruKappa(l) => g.sensory.d_lru_kappa[l],
            ParamKind::SenseVcaW(l) => g.sensory.d_vca_w[l],
            ParamKind::SenseInL(l) => g.sensory.d_in_l[l],
            ParamKind::SenseInR(l) => g.sensory.d_in_r[l],
            ParamKind::PhasorFreqHz(l) => g.crossbar.d_phasor_freq_hz[l],
            ParamKind::CrossbarOpGain(c) => g.crossbar.d_crossbar_op_gains[c],
            ParamKind::CarrierBasisW(b, l) => g.crossbar.d_carrier_basis_w[b][l],
            ParamKind::OpPToCore(q, l) => g.crossbar.d_op_p_to_core[q][l],
            ParamKind::PhasorToCore(q, l) => g.crossbar.d_phasor_to_core[q][l],
            ParamKind::PhasorStereoPhase(l) => g.crossbar.d_phasor_stereo_phase[l],
            ParamKind::OpNPreGammaBc => g.op_n_pre.d_gamma_bc,
            ParamKind::OpNPreZohHoldDepth => g.op_n_pre.d_zoh_hold_depth,
            ParamKind::OpNPreBetaQ => g.op_n_pre.d_beta_q,
            ParamKind::RoPole48k => g.d_ro_pole_48k,
            ParamKind::KautzRLow(k) => kz_grad(k, 0),
            ParamKind::KautzTheta(k) => kz_grad(k, 1),
            ParamKind::OpTDelay(q, l) => g.op_t[q].d_delay_samples[l],
            ParamKind::OpTRampSpan(q, l) => g.op_t[q].d_ramp_spans_samples[l],
            ParamKind::OpTGLoop(q) => g.op_t[q].d_g_loop,
            ParamKind::OpTRDc(q) => g.op_t[q].d_r_dc,
            ParamKind::OpTDampPole(q) => g.op_t[q].d_damp_pole,
            ParamKind::OpTLfoRate(q) => g.op_t[q].d_lfo_rate_hz,
            ParamKind::OpTRampRate(q) => g.op_t[q].d_ramp_rate_hz,
            ParamKind::OpTLfoDepth(q, l) => g.op_t[q].d_lfo_depths[l],
            ParamKind::OpTInputGain(q, l) => g.op_t[q].d_input_gains[l],
            ParamKind::OpTAllpassG(q, l) => g.op_t[q].d_allpass_g[l],
            ParamKind::OpTFbPsi(q) => g.op_t[q].d_fb_psi,
            ParamKind::OpTSigmaKnee(q) => g.op_t[q].d_sigma_knee,
            ParamKind::OpTFreezeMix(q) => g.op_t[q].d_freeze_mix,
            ParamKind::OpTWindowDepth(q) => g.op_t[q].d_window_depth,
            ParamKind::OpTScatPsi(q) => g.op_t[q].d_scat_psi,
            ParamKind::OptToLr(i, c) => g.crossbar.d_opt_to_lr[i][c],
            ParamKind::KautzBGamma(k) => kz_grad(k, 2),
            ParamKind::OpTWetTapL(q, l) => g.op_t[q].d_wet_tap_l[l],
            ParamKind::OpTWetTapR(q, l) => g.op_t[q].d_wet_tap_r[l],
            ParamKind::TankDryMix => g.d_tank_dry_mix,
            ParamKind::WReadout(row, col) => g.d_w_ro.get(row).map_or(0.0, |r| r[col]),
            ParamKind::DirectS2U(c) => g.d_direct_s2_u[c],
            ParamKind::DirectS2Hp(c) => g.d_direct_s2_hp[c],
            ParamKind::PreConvU(c) => g.d_pre_conv_u[c],
            ParamKind::PreConvHp(c) => g.d_pre_conv_hp[c],
            ParamKind::KautzCReL(k) => kz_grad(k, 3),
            ParamKind::KautzCImL(k) => kz_grad(k, 4),
            ParamKind::KautzCReR(k) => kz_grad(k, 5),
            ParamKind::KautzCImR(k) => kz_grad(k, 6),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::nonlinear::{
        append_v4_core_quad_extra_entries, svf_g_freq_ratio_log_box, v4_core_quad_bounds,
    };
    use super::*;

    #[test]
    fn test_svf_g_fc_fs_adapter_roundtrip_and_chain_rule_fd() {
        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].svf_g[0] = 0.65;
        model.sense_svf_g[1] = 1.45;
        let core_e = ThetaEntry::f32(ParamKind::CoreSvfG(0, 0), svf_g_freq_ratio_log_box());
        let sense_e = ThetaEntry::f32(ParamKind::SenseSvfG(1), svf_g_freq_ratio_log_box());

        let r_core = core_e.read(&model);
        let r_sense = sense_e.read(&model);
        let mut m_rt = model.clone();
        core_e.write(&mut m_rt, r_core, &[true; 32]);
        sense_e.write(&mut m_rt, r_sense, &[true; 32]);
        assert!((m_rt.core_quads[0].svf_g[0] - 0.65).abs() < 1.0e-12);
        assert!((m_rt.sense_svf_g[1] - 1.45).abs() < 1.0e-12);

        let mut grads = Pass2ParamGrads::with_dims(1, 0);
        grads.core_quads[0].svf_g[0] = 1.0;
        grads.sensory.d_svf_g[1] = 1.0;
        let dg_dr_core = core_e.read_grad(&model, &grads, &[true; 32]);
        let dg_dr_sense = sense_e.read_grad(&model, &grads, &[true; 32]);

        let eps = 1.0e-7_f64;
        let (mut cp, mut cm) = (model.clone(), model.clone());
        core_e.write(&mut cp, r_core + eps, &[true; 32]);
        core_e.write(&mut cm, r_core - eps, &[true; 32]);
        let fd_core = (cp.core_quads[0].svf_g[0] - cm.core_quads[0].svf_g[0]) / (2.0 * eps);
        assert!((dg_dr_core - fd_core).abs() < 1.0e-9 * fd_core.abs().max(1.0));

        sense_e.write(&mut cp, r_sense + eps, &[true; 32]);
        sense_e.write(&mut cm, r_sense - eps, &[true; 32]);
        let fd_sense = (cp.sense_svf_g[1] - cm.sense_svf_g[1]) / (2.0 * eps);
        assert!((dg_dr_sense - fd_sense).abs() < 1.0e-9 * fd_sense.abs().max(1.0));

        // Verify v4 CoreQuad helper round-trips and read_grad without mutating default v3 layout.
        let mut v4_entries = Vec::new();
        append_v4_core_quad_extra_entries(&mut v4_entries, 1);
        assert_eq!(v4_entries.len(), 28);
        let _ = v4_core_quad_bounds();
        let mut grads_v4 = Pass2ParamGrads::with_dims(1, 0);
        grads_v4.core_quads[0].g0[0] = 2.0;
        grads_v4.core_quads[0].out_gain[3] = -3.0;
        for e in &v4_entries {
            let v = e.read(&model);
            e.write(&mut m_rt, v, &[true; 32]);
            let _ = e.read_grad(&m_rt, &grads_v4, &[true; 32]);
        }
    }
}
