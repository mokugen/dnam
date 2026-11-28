//! Declarative full-subspace parameter descriptors ([`ParamKind`], [`ThetaEntry`]) and
//! readout subspace append helpers (`ref[train.pipeline.backbone-trajectory-opt]`).

mod arena;
mod entry;
mod linear_subspace;
mod nonlinear;
mod opt_subspace;

pub(super) use entry::ThetaEntry;
pub(super) use linear_subspace::append_linear_modal_transport_entries;
pub(super) use nonlinear::{NonlinearSubspaceSpec, append_nonlinear_entries};
pub(super) use opt_subspace::active_opt_quad_spec;

use crate::linalg::lbfgs::ParamBoundSpec;

/// Scalar parameter identifier indexing `SonaModelF64` and `Pass2ParamGrads` slots.
#[derive(Clone, Copy, Debug)]
pub enum ParamKind {
    /// `CoreQuad[q].svf_g[lane]` (`log(fc / fs)` coordinate adapter).
    CoreSvfG(usize, usize),
    /// `CoreQuad[q].svf_k[lane]`.
    CoreSvfK(usize, usize),
    /// `CoreQuad[q].svf_k_nl[lane]` (Block 0 only).
    CoreSvfKNl(usize, usize),
    /// `CoreQuad[q].tone_lp[lane]`.
    CoreToneLp(usize, usize),
    /// `CoreQuad[q].tone_bp[lane]`.
    CoreToneBp(usize, usize),
    /// `CoreQuad[q].tone_hp[lane]`.
    CoreToneHp(usize, usize),
    /// `CoreQuad[q].tone_ap[lane]`.
    CoreToneAp(usize, usize),
    /// `CoreQuad[q].tone_dir[lane]`.
    CoreToneDir(usize, usize),
    /// `CoreQuad[q].lru_alpha_rel[lane]`.
    CoreLruRel(usize, usize),
    /// `CoreQuad[q].lru_alpha_atk[lane]`.
    CoreLruAtk(usize, usize),
    /// `CoreQuad[q].lru_kappa[lane]`.
    CoreLruKap(usize, usize),
    /// `CoreQuad[q].lru_duhem_gamma[lane]`.
    CoreLruDuhem(usize, usize),
    /// `CoreQuad[q].lru_signed_mix[lane]`.
    CoreLruSignedMix(usize, usize),
    /// `CoreQuad[q].lru_stereo_link[lane]`.
    CoreLruStereoLink(usize, usize),
    /// `CoreQuad[q].sag_depth[lane]`.
    CoreSagDepth(usize, usize),
    /// `CoreQuad[q].adaa_drive[lane]`.
    CoreAdaaDrive(usize, usize),
    /// `CoreQuad[q].lru_bias_drive[lane]`.
    CoreLruBias(usize, usize),
    /// `CoreQuad[q].wavefolder_mix[lane]`.
    CoreWavefolder(usize, usize),
    /// `CoreQuad[q].glu_mix[lane]`.
    CoreGluMix(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].g0[lane]` (gated until `C3.1`).
    CoreG0(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].g_inf[lane]` (gated until `C3.1`).
    CoreGInf(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].d_pre[lane]` (gated until `C3.1`).
    CoreDPre(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].se_w_sq[lane]` (gated until `C3.1`).
    CoreSeWSq(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].se_d_bus[lane]` (gated until `C3.1`).
    CoreSeDBus(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].lin_mix[lane]` (gated until `C3.1`).
    CoreLinMix(usize, usize),
    /// Prepared `SONA v4.0` `CoreQuad[q].out_gain[lane]` (gated until `C3.1`).
    CoreOutGain(usize, usize),
    /// `CoreQuad[q].in_mid_weight[lane]`.
    CoreInMid(usize, usize),
    /// `CoreQuad[q].in_side_weight[lane]`.
    CoreInSide(usize, usize),
    /// `InterBlockTile[tile].w_odd_4x4[src][dst]`.
    TileOdd(usize, usize, usize),
    /// `InterBlockTile[tile].w_even_4x4[src][dst]`.
    TileEven(usize, usize, usize),
    /// `InterBlockTile[tile].w_lru_4x4[src][dst]`.
    TileLru(usize, usize, usize),
    /// `SensoryQuad.sense_svf_g[lane]` (`log(fc / fs)` coordinate adapter).
    SenseSvfG(usize),
    /// `SensoryQuad.sense_lru_rel[lane]`.
    SenseLruRel(usize),
    /// `SensoryQuad.sense_lru_atk[lane]`.
    SenseLruAtk(usize),
    /// `SensoryQuad.sense_svf_k[lane]`.
    SenseSvfK(usize),
    /// `SensoryQuad.sense_tone_lp[lane]`.
    SenseToneLp(usize),
    /// `SensoryQuad.sense_tone_bp[lane]`.
    SenseToneBp(usize),
    /// `SensoryQuad.sense_tone_hp[lane]`.
    SenseToneHp(usize),
    /// `SensoryQuad.sense_lru_kappa[lane]`.
    SenseLruKappa(usize),
    /// `SensoryQuad.sense_vca_w[lane]`.
    SenseVcaW(usize),
    /// `SensoryQuad.sense_in_l[lane]`.
    SenseInL(usize),
    /// `SensoryQuad.sense_in_r[lane]`.
    SenseInR(usize),
    /// `Op-P` crossbar `phasor_freq_hz[lane]`.
    PhasorFreqHz(usize),
    /// `Op-P` crossbar `crossbar_op_gains[idx]`.
    CrossbarOpGain(usize),
    /// `Op-P` crossbar `carrier_basis_w[basis][lane]`.
    CarrierBasisW(usize, usize),
    /// `Op-P` crossbar `op_p_to_core[q][lane]`.
    OpPToCore(usize, usize),
    /// `Op-P` crossbar `phasor_to_core[q][lane]`.
    PhasorToCore(usize, usize),
    /// `Op-P` crossbar `phasor_stereo_phase[lane]`.
    PhasorStereoPhase(usize),
    /// `Op-N[Pre].gamma_bc`.
    OpNPreGammaBc,
    /// `Op-N[Pre].zoh_hold_depth`.
    OpNPreZohHoldDepth,
    /// `Op-N[Pre].beta_q`.
    OpNPreBetaQ,
    /// `LinearReadout` leak pole `ro_pole_48k`.
    RoPole48k,
    /// `Kautz[k].r_low`.
    KautzRLow(usize),
    /// `Kautz[k].theta`.
    KautzTheta(usize),
    /// `TransportQuad[q].delay_samples[lane]`.
    OpTDelay(usize, usize),
    /// `TransportQuad[q].ramp_spans_samples[lane]`.
    OpTRampSpan(usize, usize),
    /// `TransportQuad[q].g_loop`.
    OpTGLoop(usize),
    /// `TransportQuad[q].r_dc`.
    OpTRDc(usize),
    /// `TransportQuad[q].damp_pole`.
    OpTDampPole(usize),
    /// `TransportQuad[q].lfo_rate_hz`.
    OpTLfoRate(usize),
    /// `TransportQuad[q].ramp_rate_hz`.
    OpTRampRate(usize),
    /// `TransportQuad[q].lfo_depths[lane]`.
    OpTLfoDepth(usize, usize),
    /// `TransportQuad[q].input_gains[lane]`.
    OpTInputGain(usize, usize),
    /// `TransportQuad[q].allpass_g[lane]`.
    OpTAllpassG(usize, usize),
    /// `TransportQuad[q].fb_psi`.
    OpTFbPsi(usize),
    /// `TransportQuad[q].sigma_knee`.
    OpTSigmaKnee(usize),
    /// `TransportQuad[q].freeze_mix`.
    OpTFreezeMix(usize),
    /// `TransportQuad[q].window_depth`.
    OpTWindowDepth(usize),
    /// `TransportQuad[q].scat_psi`.
    OpTScatPsi(usize),
    /// `Crossbar` `opt_to_lr[lane][ch]`.
    OptToLr(usize, usize),
    /// `Kautz[k].b_gamma`.
    KautzBGamma(usize),
    /// `TransportQuad[q].wet_tap_l[lane]`.
    OpTWetTapL(usize, usize),
    /// `TransportQuad[q].wet_tap_r[lane]`.
    OpTWetTapR(usize, usize),
    /// `SonaModelF64.tank_dry_mix`.
    TankDryMix,
    /// `LinearReadout` `w_ro[row][col]`.
    WReadout(usize, usize),
    /// `SonaModelF64.direct_s2_u_gain[ch]`.
    DirectS2U(usize),
    /// `SonaModelF64.direct_s2_hp_gain[ch]`.
    DirectS2Hp(usize),
    /// `SonaModelF64.pre_conv_u_gain[ch]`.
    PreConvU(usize),
    /// `SonaModelF64.pre_conv_hp_gain[ch]`.
    PreConvHp(usize),
    /// `Kautz[k].c_re_l`.
    KautzCReL(usize),
    /// `Kautz[k].c_im_l`.
    KautzCImL(usize),
    /// `Kautz[k].c_re_r`.
    KautzCReR(usize),
    /// `Kautz[k].c_im_r`.
    KautzCImR(usize),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readout_builders_and_arena_ids() {
        let mut joint = Vec::new();
        append_joint_readout_entries(&mut joint, 6, 4);
        assert_eq!(joint.len(), 24 * 6 + 4 + 24 * 6 + 4 + 16);
        assert!(joint.iter().all(|e| e.kind.arena_id() == 4));
        assert!(joint.iter().all(|e| e.kind.is_readout_coupled()));

        let mut qat = Vec::new();
        append_qat_readout_entries(&mut qat, 6, 4);
        assert_eq!(qat.len(), 16 * 6 + 2 + 24 * 6 + 4 + 16);
    }
}
