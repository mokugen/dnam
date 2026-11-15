#![forbid(unsafe_code)]
//! Shared `f32` engine trajectory extraction and 2x oversampled decimation for readout polish.

use dnam_engine::math::SimdVec4;
use dnam_engine::stages::v3_core::V3MatryoshkaModel;
use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, V3GruBlockHot,
    V3GruChannelState, ZeroLagOversampler2x,
};
use dnam_engine::stages::v3_core::svf::V3Stage1SvfState;

use crate::solver::backends::TrainerBackend;

use super::super::basis::StemChanTraj;

pub(super) fn extract_f32_shared_decimated<B: TrainerBackend>(
    cold: &V3MatryoshkaModel,
    in_l: &[f32],
    in_r: &[f32],
) -> [StemChanTraj; 2] {
    let n = in_l.len().min(in_r.len());
    let active_ch = if cold.mono_out { 1 } else { 2 };
    let mut os_p = cold.os_params;
    os_p.clamp_in_place();
    let mut svf_base = cold.svf;
    svf_base.clamp_in_place();
    let mut feat_p = cold.feat_params;
    feat_p.clamp_in_place();
    let blocks_hot = [
        V3GruBlockHot::from_cold::<B>(&cold.blocks[0], MAX_GROUPS),
        V3GruBlockHot::from_cold::<B>(&cold.blocks[1], MAX_GROUPS),
        V3GruBlockHot::from_cold::<B>(&cold.blocks[2], MAX_GROUPS),
    ];

    let mut svf_state = V3Stage1SvfState::default();
    let (mut os_l, mut os_r) = (
        ZeroLagOversampler2x::default(),
        ZeroLagOversampler2x::default(),
    );
    let (mut dec_tap_p, mut dec_u_p, mut dec_hp_p) =
        ([[0.0_f32; 7]; 2], [0.0_f32; 2], [0.0_f32; 2]);
    let mut dec_h_p = [[0.0_f32; MAX_TOTAL_GRU_LANES]; 2];
    let (mut dec_tap_aa, mut dec_u_aa, mut dec_hp_aa) =
        ([[[0.0_f32; 4]; 7]; 2], [[0.0_f32; 4]; 2], [[0.0_f32; 4]; 2]);
    let mut dec_h_aa = [[[0.0_f32; 4]; MAX_TOTAL_GRU_LANES]; 2];
    let mut ch_states = [V3GruChannelState::default(), V3GruChannelState::default()];
    let mut out = [
        StemChanTraj {
            tap: vec![[0.0; 7]; n],
            u: vec![0.0; n],
            hp: vec![0.0; n],
            h: vec![[0.0; MAX_TOTAL_GRU_LANES]; n],
        },
        StemChanTraj {
            tap: vec![[0.0; 7]; n],
            u: vec![0.0; n],
            hp: vec![0.0; n],
            h: vec![[0.0; MAX_TOTAL_GRU_LANES]; n],
        },
    ];
    let sub_steps = if cold.os2x { 2 } else { 1 };
    let zero_w = [[0.0_f32; 4]; MAX_GROUPS];

    for t in 0..n {
        let xl = if in_l[t].is_finite() { in_l[t] } else { 0.0 };
        let xr = if cold.mono_out {
            xl
        } else if in_r[t].is_finite() {
            in_r[t]
        } else {
            0.0
        };
        let xs_sub = if cold.os2x {
            let lp = os_l.upsample(xl, &os_p);
            let rp = if cold.mono_out {
                lp
            } else {
                os_r.upsample(xr, &os_p)
            };
            [[lp[0], rp[0]], [lp[1], rp[1]]]
        } else {
            [[xl, xr], [0.0, 0.0]]
        };
        let mut tap_sub = [[[0.0_f32; 7]; 2]; 2];
        let (mut u_sub, mut hp_sub) = ([[0.0_f32; 2]; 2], [[0.0_f32; 2]; 2]);
        let mut h_sub = [[[0.0_f32; MAX_TOTAL_GRU_LANES]; 2]; 2];

        for sub in 0..sub_steps {
            let (u_s2, _, tap_ch) =
                svf_state.step_stereo_with_taps(&svf_base, xs_sub[sub][0], xs_sub[sub][1]);
            tap_sub[sub] = tap_ch;
            for (ch, cst) in ch_states.iter_mut().enumerate().take(active_ch) {
                let feats = cst.feat.step(&feat_p, u_s2[ch]);
                u_sub[sub][ch] = feats[0];
                hp_sub[sub][ch] = feats[1];
                let mut prev_h = [[0.0_f32; 4]; MAX_GROUPS];
                for b in 0..NUM_GRU_BLOCKS {
                    let (mut ca, mut da) = (B::Vec4::zero(), B::Vec4::zero());
                    blocks_hot[b].step_block::<B>(
                        b,
                        MAX_GROUPS,
                        &feats,
                        &prev_h,
                        &mut cst.h[b],
                        &mut ca,
                        &mut da,
                        &zero_w,
                        &zero_w,
                    );
                    prev_h = cst.h[b];
                    for g in 0..MAX_GROUPS {
                        for l in 0..4 {
                            h_sub[sub][ch][b * MAX_BLOCK_LANES + g * 4 + l] = prev_h[g][l];
                        }
                    }
                }
            }
        }
        for ch in 0..active_ch {
            if !cold.os2x {
                out[ch].tap[t] = tap_sub[0][ch].map(f64::from);
                out[ch].u[t] = f64::from(u_sub[0][ch]);
                out[ch].hp[t] = f64::from(hp_sub[0][ch]);
                out[ch].h[t] = h_sub[0][ch].map(f64::from);
            } else {
                for i in 0..7 {
                    out[ch].tap[t][i] = f64::from(ZeroLagOversampler2x::downsample(
                        tap_sub[0][ch][i],
                        tap_sub[1][ch][i],
                        &mut dec_tap_p[ch][i],
                        &mut dec_tap_aa[ch][i],
                        &os_p,
                    ));
                }
                out[ch].u[t] = f64::from(ZeroLagOversampler2x::downsample(
                    u_sub[0][ch],
                    u_sub[1][ch],
                    &mut dec_u_p[ch],
                    &mut dec_u_aa[ch],
                    &os_p,
                ));
                out[ch].hp[t] = f64::from(ZeroLagOversampler2x::downsample(
                    hp_sub[0][ch],
                    hp_sub[1][ch],
                    &mut dec_hp_p[ch],
                    &mut dec_hp_aa[ch],
                    &os_p,
                ));
                for idx in 0..MAX_TOTAL_GRU_LANES {
                    out[ch].h[t][idx] = f64::from(ZeroLagOversampler2x::downsample(
                        h_sub[0][ch][idx],
                        h_sub[1][ch][idx],
                        &mut dec_h_p[ch][idx],
                        &mut dec_h_aa[ch][idx],
                        &os_p,
                    ));
                }
            }
        }
    }
    out
}
