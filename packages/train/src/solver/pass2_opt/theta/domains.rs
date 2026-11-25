//! Declarative full-subspace parameter descriptors ([`ThetaEntry`], [`ParamKind`]) mapping
//! every `SonaModelF64` continuous & joint-readout parameter to its exact `ParamBoundSpec`
//! bijection, `.dnam-bin` wire precision mask (`binary16` vs `binary32`), and gradient slot
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

mod arena;
mod linear_subspace;
mod opt_subspace;

pub(super) use linear_subspace::append_linear_modal_transport_entries;
pub(super) use opt_subspace::active_opt_quad_spec;

use core::f64::consts::PI;

use dnam_kernels::{
    MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, MAX_SVF_K_NL, MIN_TPT_G, OP_N_PRE_MAX_BETA_Q,
    OP_N_PRE_MAX_ZOH_HOLD_DEPTH, TANK_MAX_G_LIN_F32,
};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;

use crate::control::stability::{MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP};
use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::super::model::SonaModelF64;
use super::super::eval::Pass2ParamGrads;

#[derive(Clone, Copy, Debug)]
pub(super) enum ParamKind {
    CoreSvfG(usize, usize),
    CoreSvfK(usize, usize),
    CoreSvfKNl(usize, usize),
    CoreToneLp(usize, usize),
    CoreToneBp(usize, usize),
    CoreToneHp(usize, usize),
    CoreToneAp(usize, usize),
    CoreToneDir(usize, usize),
    CoreLruRel(usize, usize),
    CoreLruAtk(usize, usize),
    CoreLruKap(usize, usize),
    CoreLruDuhem(usize, usize),
    CoreLruSignedMix(usize, usize),
    CoreLruStereoLink(usize, usize),
    CoreSagDepth(usize, usize),
    CoreAdaaDrive(usize, usize),
    CoreLruBias(usize, usize),
    CoreWavefolder(usize, usize),
    CoreGluMix(usize, usize),
    CoreInMid(usize, usize),
    CoreInSide(usize, usize),
    TileOdd(usize, usize, usize),
    TileEven(usize, usize, usize),
    TileLru(usize, usize, usize),
    SenseSvfG(usize),
    SenseLruRel(usize),
    SenseLruAtk(usize),
    SenseSvfK(usize),
    SenseToneLp(usize),
    SenseToneBp(usize),
    SenseToneHp(usize),
    SenseLruKappa(usize),
    SenseVcaW(usize),
    SenseInL(usize),
    SenseInR(usize),
    PhasorFreqHz(usize),
    CrossbarOpGain(usize),
    CarrierBasisW(usize, usize),
    OpPToCore(usize, usize),
    PhasorToCore(usize, usize),
    PhasorStereoPhase(usize),
    OpNPreGammaBc,
    OpNPreZohHoldDepth,
    OpNPreBetaQ,
    RoPole48k,
    KautzRLow(usize),
    KautzTheta(usize),
    OpTDelay(usize, usize),
    OpTRampSpan(usize, usize),
    OpTGLoop(usize),
    OpTRDc(usize),
    OpTDampPole(usize),
    OpTLfoRate(usize),
    OpTRampRate(usize),
    OpTLfoDepth(usize, usize),
    OpTInputGain(usize, usize),
    OpTAllpassG(usize, usize),
    OpTFbPsi(usize),
    OpTSigmaKnee(usize),
    OpTFreezeMix(usize),
    OpTWindowDepth(usize),
    OpTScatPsi(usize),
    OptToLr(usize, usize),
    KautzBGamma(usize),
    OpTWetTapL(usize, usize),
    OpTWetTapR(usize, usize),
    TankDryMix,
    // Joint Pass-2 linear readout slice (used prior to M21 periodic VarPro).
    WReadout(usize, usize),
    DirectS2U(usize),
    DirectS2Hp(usize),
    PreConvU(usize),
    PreConvHp(usize),
    KautzCReL(usize),
    KautzCImL(usize),
    KautzCReR(usize),
    KautzCImR(usize),
}

/// One parameter slot descriptor linking physical model storage, coordinate transform,
/// wire quantization flag, and adjoint gradient slot.
#[derive(Clone, Copy, Debug)]
pub(super) struct ThetaEntry {
    pub kind: ParamKind,
    pub bound: ParamBoundSpec,
    pub is_fp16_wire: bool,
}

impl ThetaEntry {
    #[inline]
    fn f32(kind: ParamKind, bound: ParamBoundSpec) -> Self {
        Self {
            kind,
            bound,
            is_fp16_wire: false,
        }
    }

    #[inline]
    fn fp16(kind: ParamKind, bound: ParamBoundSpec) -> Self {
        Self {
            kind,
            bound,
            is_fp16_wire: true,
        }
    }

    #[must_use]
    pub(super) fn read(&self, m: &SonaModelF64) -> f64 {
        match self.kind {
            ParamKind::CoreSvfG(q, l) => m.core_quads[q].svf_g[l],
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
            ParamKind::CoreInMid(q, l) => m.core_quads[q].in_mid_weight[l],
            ParamKind::CoreInSide(q, l) => m.core_quads[q].in_side_weight[l],
            ParamKind::TileOdd(t, s, d) => m.tiles[t].w_odd_4x4[s][d],
            ParamKind::TileEven(t, s, d) => m.tiles[t].w_even_4x4[s][d],
            ParamKind::TileLru(t, s, d) => m.tiles[t].w_lru_4x4[s][d],
            ParamKind::SenseSvfG(l) => m.sense_svf_g[l],
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

    pub(super) fn write(&self, m: &mut SonaModelF64, raw: f64, kautz_mask: &[bool; 32]) {
        let clamped = if raw.is_finite() {
            self.bound.snap_zero_edge(self.bound.project_physical(raw))
        } else {
            self.bound.default_midpoint()
        };
        match self.kind {
            ParamKind::CoreSvfG(q, l) => m.core_quads[q].svf_g[l] = clamped,
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
            ParamKind::CoreInMid(q, l) => m.core_quads[q].in_mid_weight[l] = clamped,
            ParamKind::CoreInSide(q, l) => m.core_quads[q].in_side_weight[l] = clamped,
            ParamKind::TileOdd(t, s, d) => m.tiles[t].w_odd_4x4[s][d] = clamped,
            ParamKind::TileEven(t, s, d) => m.tiles[t].w_even_4x4[s][d] = clamped,
            ParamKind::TileLru(t, s, d) => m.tiles[t].w_lru_4x4[s][d] = clamped,
            ParamKind::SenseSvfG(l) => m.sense_svf_g[l] = clamped,
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
    pub(super) fn read_grad(&self, g: &Pass2ParamGrads, kautz_mask: &[bool; 32]) -> f64 {
        let kz_grad = |k: usize, idx: usize| -> f64 {
            if kautz_mask[k] {
                g.d_kautz.get(k).map_or(0.0, |v| v[idx])
            } else {
                0.0
            }
        };
        match self.kind {
            ParamKind::CoreSvfG(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.svf_g[l]),
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
            ParamKind::CoreInMid(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.in_mid_weight[l]),
            ParamKind::CoreInSide(q, l) => g.core_quads.get(q).map_or(0.0, |c| c.in_side_weight[l]),
            ParamKind::TileOdd(t, s, d) => g.tiles.get(t).map_or(0.0, |tl| tl.w_odd_4x4[s][d]),
            ParamKind::TileEven(t, s, d) => g.tiles.get(t).map_or(0.0, |tl| tl.w_even_4x4[s][d]),
            ParamKind::TileLru(t, s, d) => g.tiles.get(t).map_or(0.0, |tl| tl.w_lru_4x4[s][d]),
            ParamKind::SenseSvfG(l) => g.sensory.d_svf_g[l],
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

#[derive(Clone, Copy, Debug)]
pub(super) struct NonlinearSubspaceSpec {
    pub q_count: usize,
    pub active_tiles: usize,
    pub include_stereo_phase: bool,
    pub k_count: usize,
    pub active_opt_quads: usize,
    pub opt_feedforward_only_mask: [bool; 4],
    pub include_extras: bool,
    pub sr_hz: f64,
    pub max_tau: f64,
    pub kautz_r_max: f64,
}

/// Appends the canonical full-subspace continuous nonlinear descriptors matching
/// `webgpu_and_varpro_solver.md` lines `93–118`, returning `zoh_hold_depth_idx`.
pub(super) fn append_nonlinear_entries(
    entries: &mut Vec<ThetaEntry>,
    spec: &NonlinearSubspaceSpec,
) -> Option<usize> {
    let NonlinearSubspaceSpec {
        q_count,
        active_tiles,
        include_stereo_phase,
        k_count,
        active_opt_quads,
        opt_feedforward_only_mask,
        include_extras,
        sr_hz,
        max_tau,
        kautz_r_max,
    } = *spec;
    let g_lo = f64::from(MIN_TPT_G);
    let r_hi = f64::from(MAX_PARALRU_R_F32);
    let svf_g_box = ParamBoundSpec::LogBox { lo: g_lo, hi: 64.0 };
    let rel_clog = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - r_hi,
        hi_c: 1.0,
    };
    let unit_box = ParamBoundSpec::Box { lo: 0.0, hi: 1.0 };
    let svf_k_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 1.0,
        limit: 8.0,
    };
    let k_nl_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 1.0,
        limit: f64::from(MAX_SVF_K_NL),
    };
    let tone_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 16.0,
    };
    let duhem_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 2.0,
        limit: f64::from(MAX_PARALRU_DUHEM_GAMMA_F32),
    };
    let sag_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 8.0,
    };
    let adaa_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 32.0,
    };
    let glu_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 4.0,
    };

    // 1. CoreQuads (21/lane on Block 0, 20/lane on Blocks 1..=3).
    for q in 0..q_count {
        let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0;
        for l in 0..4 {
            entries.push(ThetaEntry::f32(ParamKind::CoreSvfG(q, l), svf_g_box));
            entries.push(ThetaEntry::f32(ParamKind::CoreLruRel(q, l), rel_clog));
            entries.push(ThetaEntry::f32(ParamKind::CoreLruAtk(q, l), unit_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreSvfK(q, l), svf_k_box));
            if is_block0 {
                entries.push(ThetaEntry::fp16(ParamKind::CoreSvfKNl(q, l), k_nl_box));
            }
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneLp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneBp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneHp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneAp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneDir(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLruKap(q, l), unit_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLruDuhem(q, l), duhem_box));
            entries.push(ThetaEntry::fp16(
                ParamKind::CoreLruSignedMix(q, l),
                unit_box,
            ));
            entries.push(ThetaEntry::fp16(
                ParamKind::CoreLruStereoLink(q, l),
                unit_box,
            ));
            entries.push(ThetaEntry::fp16(ParamKind::CoreSagDepth(q, l), sag_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreAdaaDrive(q, l), adaa_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLruBias(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreWavefolder(q, l), unit_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreGluMix(q, l), glu_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreInMid(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreInSide(q, l), tone_box));
        }
    }

    // 2. InterBlockTiles (48/tile: 16 f32 w_odd + 32 fp16 w_even/w_lru).
    let w_tile_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 8.0,
    };
    for t in 0..active_tiles {
        for s in 0..4 {
            for d in 0..4 {
                entries.push(ThetaEntry::f32(ParamKind::TileOdd(t, s, d), w_tile_box));
                entries.push(ThetaEntry::fp16(ParamKind::TileEven(t, s, d), w_tile_box));
                entries.push(ThetaEntry::fp16(ParamKind::TileLru(t, s, d), w_tile_box));
            }
        }
    }

    // 3. SensoryQuad (44 = 12 f32 + 32 fp16).
    let vca_box = ParamBoundSpec::AsinhBoxSym {
        scale: 4.0,
        limit: 512.0,
    };
    let sin_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 4.0,
    };
    for l in 0..4 {
        entries.push(ThetaEntry::f32(ParamKind::SenseSvfG(l), svf_g_box));
        entries.push(ThetaEntry::f32(ParamKind::SenseLruRel(l), rel_clog));
        entries.push(ThetaEntry::f32(ParamKind::SenseLruAtk(l), unit_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseSvfK(l), svf_k_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseToneLp(l), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseToneBp(l), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseToneHp(l), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseLruKappa(l), unit_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseVcaW(l), vca_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseInL(l), sin_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseInR(l), sin_box));
    }

    // 4. Op-P & Phasor Crossbar (50/70/86/102 base + 8/16/24/32 opt_to_lr).
    let phasor_lim = 23_520.0_f64.min(0.49 * sr_hz.max(44_100.0));
    let freq_box = ParamBoundSpec::AsinhBoxSym {
        scale: 10.0,
        limit: phasor_lim,
    };
    let angle_box = ParamBoundSpec::Box { lo: -PI, hi: PI };
    for l in 0..4 {
        entries.push(ThetaEntry::f32(ParamKind::PhasorFreqHz(l), freq_box));
    }
    for c in 0..2 {
        entries.push(ThetaEntry::fp16(ParamKind::CrossbarOpGain(c), glu_box));
    }
    for b in 0..3 {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::CarrierBasisW(b, l), w_tile_box));
        }
    }
    for q in 0..q_count {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::OpPToCore(q, l), w_tile_box));
        }
    }
    for q in 0..q_count {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::PhasorToCore(q, l), w_tile_box));
        }
    }
    if include_stereo_phase {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::PhasorStereoPhase(l), angle_box));
        }
    }
    if include_extras {
        for q in 0..active_opt_quads {
            if opt_feedforward_only_mask[q] {
                continue;
            }
            for lane in 0..4 {
                let i = q * 4 + lane;
                for c in 0..2 {
                    // Hydrated binary16 across ALL tiers (f32le on Lite wire, fp16 on Full..Ultra wire).
                    entries.push(ThetaEntry::fp16(ParamKind::OptToLr(i, c), glu_box));
                }
            }
        }
    }

    // 5. Op-N[Pre] (3 f32).
    entries.push(ThetaEntry::f32(ParamKind::OpNPreGammaBc, unit_box));
    let zoh_hold_depth_idx = Some(entries.len());
    entries.push(ThetaEntry::f32(
        ParamKind::OpNPreZohHoldDepth,
        ParamBoundSpec::Box {
            lo: 0.0,
            hi: f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH),
        },
    ));
    entries.push(ThetaEntry::f32(
        ParamKind::OpNPreBetaQ,
        ParamBoundSpec::AsinhBoxNonNeg {
            scale: 1.0,
            limit: f64::from(OP_N_PRE_MAX_BETA_Q),
        },
    ));

    // 6. ModalConvolver Poles & Shelf (17/25/49/65 base + 8/12/24/32 b_gamma).
    entries.push(ThetaEntry::f32(
        ParamKind::RoPole48k,
        ParamBoundSpec::ComplementLog {
            lo_c: 1.0 - MAX_RO_POLE_48K_F64,
            hi_c: 1.0,
        },
    ));
    let r_hi = kautz_r_max.clamp(0.5, 0.9995);
    let kautz_r_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - r_hi,
        hi_c: 1.0 - 0.25,
    };
    let (th_min, th_max) = crate::solver::pass1_probe::kautz_fit::audible_kautz_theta_bounds(sr_hz);
    let kautz_theta_box = ParamBoundSpec::LogBox {
        lo: th_min,
        hi: th_max,
    };
    for k in 0..k_count {
        entries.push(ThetaEntry::f32(ParamKind::KautzRLow(k), kautz_r_box));
        entries.push(ThetaEntry::f32(ParamKind::KautzTheta(k), kautz_theta_box));
        if include_extras {
            entries.push(ThetaEntry::f32(
                ParamKind::KautzBGamma(k),
                ParamBoundSpec::LogBox {
                    lo: 1.0e-4,
                    hi: 8.0,
                },
            ));
        }
    }

    // 7. TransportQuads (30/60/90/120 base + 9/17/25/33 wet_tap & tank_dry_mix).
    let tau_max = max_tau.clamp(MIN_DELAY_CROSSBAR_SAMP + 1.0, 48_000.0);
    let span_max = (tau_max - MIN_DELAY_CROSSBAR_SAMP).max(1.0);
    let delay_box = ParamBoundSpec::LogBox {
        lo: MIN_DELAY_CROSSBAR_SAMP,
        hi: tau_max,
    };
    let span_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 32.0,
        limit: span_max,
    };
    let gloop_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - f64::from(TANK_MAX_G_LIN_F32),
        hi_c: 1.0,
    };
    let rdc_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - f64::from(dnam_kernels::MAX_PARALRU_R_F32),
        hi_c: 1.0 - 0.95,
    };
    let damp_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - 0.9995,
        hi_c: 1.0,
    };
    let lfo_rate_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 1.0,
        limit: 16.0,
    };
    let lfo_depth_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 4.0,
        limit: 512.0,
    };
    let tap_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 2.0,
    };
    let ap_box = ParamBoundSpec::Box {
        lo: -0.75,
        hi: 0.75,
    };
    let sigma_box = ParamBoundSpec::LogBox {
        lo: 1.0e-3,
        hi: 0.984_375,
    };
    let freeze_box = ParamBoundSpec::Box { lo: 0.0, hi: 0.999 };
    for q in 0..active_opt_quads {
        if opt_feedforward_only_mask[q] {
            for l in 0..4 {
                entries.push(ThetaEntry::f32(ParamKind::OpTDelay(q, l), delay_box));
                entries.push(ThetaEntry::fp16(ParamKind::OpTInputGain(q, l), tap_box));
                if include_extras {
                    entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapL(q, l), tap_box));
                    entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapR(q, l), tap_box));
                }
            }
            continue;
        }
        for l in 0..4 {
            entries.push(ThetaEntry::f32(ParamKind::OpTDelay(q, l), delay_box));
            entries.push(ThetaEntry::f32(ParamKind::OpTRampSpan(q, l), span_box));
            entries.push(ThetaEntry::fp16(
                ParamKind::OpTLfoDepth(q, l),
                lfo_depth_box,
            ));
            entries.push(ThetaEntry::fp16(ParamKind::OpTInputGain(q, l), tap_box));
            entries.push(ThetaEntry::fp16(ParamKind::OpTAllpassG(q, l), ap_box));
            if include_extras {
                entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapL(q, l), tap_box));
                entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapR(q, l), tap_box));
            }
        }
        entries.push(ThetaEntry::f32(ParamKind::OpTGLoop(q), gloop_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTRDc(q), rdc_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTDampPole(q), damp_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTLfoRate(q), lfo_rate_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTRampRate(q), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTFbPsi(q), angle_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTSigmaKnee(q), sigma_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTFreezeMix(q), freeze_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTWindowDepth(q), unit_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTScatPsi(q), angle_box));
    }
    if include_extras && active_opt_quads > 0 {
        entries.push(ThetaEntry::fp16(ParamKind::TankDryMix, tap_box));
    }
    zoh_hold_depth_idx
}

/// Appends bilinear pre-convolver coupling parameters (`w_ro[..][0..6]`, `pre_conv_u_gain`,
/// `pre_conv_hp_gain`) so nonlinear `CoreQuad` features feed into `ModalConvolver` natively
/// (`ref[train.pipeline.backbone-trajectory-opt]`).
pub(super) fn append_pre_conv_entries(entries: &mut Vec<ThetaEntry>, q_count: usize) {
    let pre_box = ParamBoundSpec::Box { lo: -4.0, hi: 4.0 };
    for row in 0..(q_count * 4) {
        for col in 0..6 {
            entries.push(ThetaEntry::fp16(ParamKind::WReadout(row, col), pre_box));
        }
    }
    for c in 0..2 {
        entries.push(ThetaEntry::fp16(ParamKind::PreConvU(c), pre_box));
        entries.push(ThetaEntry::fp16(ParamKind::PreConvHp(c), pre_box));
    }
}

/// Appends tier-invariant bilinear pre-convolver coupling limited to shared `Lite` lanes (`0..16`,
/// `w_ro[0..16][0..6]`) and `pre_conv_u_gain`, keeping upper-tier `pre_conv` buses aligned between
/// `Pass 3` and `SonaVm` without unmodeled `PreConvHp` leakage (`ref[train.pipeline.backbone-trajectory-opt]`).
pub(super) fn append_shared_pre_conv_entries(entries: &mut Vec<ThetaEntry>, q_count: usize) {
    let pre_box = ParamBoundSpec::Box { lo: -4.0, hi: 4.0 };
    let shared_rows = q_count.min(4) * 4;
    for row in 0..shared_rows {
        for col in 0..6 {
            entries.push(ThetaEntry::fp16(ParamKind::WReadout(row, col), pre_box));
        }
    }
    for c in 0..2 {
        entries.push(ThetaEntry::fp16(ParamKind::PreConvU(c), pre_box));
    }
}

/// Appends strictly linear post-cascade direct readout parameters (`w_ro[..][6..12]`,
/// `direct_s2_u_gain`, `direct_s2_hp_gain`, and Kautz modal output residues).
pub(super) fn append_direct_readout_entries(
    entries: &mut Vec<ThetaEntry>,
    q_count: usize,
    k_count: usize,
) {
    let ro_box = ParamBoundSpec::Box { lo: -8.0, hi: 8.0 };
    for row in 0..(q_count * 4) {
        for col in 6..12 {
            entries.push(ThetaEntry::fp16(ParamKind::WReadout(row, col), ro_box));
        }
    }
    for c in 0..2 {
        entries.push(ThetaEntry::fp16(ParamKind::DirectS2U(c), ro_box));
        entries.push(ThetaEntry::fp16(ParamKind::DirectS2Hp(c), ro_box));
    }
    for k in 0..k_count {
        entries.push(ThetaEntry::f32(ParamKind::KautzCReL(k), ro_box));
        entries.push(ThetaEntry::f32(ParamKind::KautzCImL(k), ro_box));
        entries.push(ThetaEntry::f32(ParamKind::KautzCReR(k), ro_box));
        entries.push(ThetaEntry::f32(ParamKind::KautzCImR(k), ro_box));
    }
}

/// Appends active linear readout parameters (`w_ro`, `direct_s2`, `pre_conv`, Kautz residues)
/// used during full-subspace FD parity verification.
pub(super) fn append_joint_readout_entries(
    entries: &mut Vec<ThetaEntry>,
    q_count: usize,
    k_count: usize,
) {
    append_pre_conv_entries(entries, q_count);
    append_direct_readout_entries(entries, q_count, k_count);
}

/// Appends tier-invariant `Lite`-shared `pre_conv` coupling together with active linear post-cascade
/// readouts for Phase-2D STE `QAT` and `Op-T`-coupled Pass-2 trajectories.
pub(super) fn append_qat_readout_entries(
    entries: &mut Vec<ThetaEntry>,
    q_count: usize,
    k_count: usize,
) {
    append_shared_pre_conv_entries(entries, q_count);
    append_direct_readout_entries(entries, q_count, k_count);
}
