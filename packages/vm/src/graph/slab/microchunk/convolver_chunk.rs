//! Batched `4`-frame `ModalConvolver` FIR shift-by-4 + register-cached `KautzStereoBank4`
//! evaluation (`#![no_std]`, `#![forbid(unsafe_code)]`, `0-ULP` parity vs 4× `step_convolver_frame`).

#![forbid(unsafe_code)]

use dnam_kernels::step_kautz_quad_stereo8;

use crate::graph::GraphBackend;
use crate::graph::convolver::kautz::KautzStereoBank4;
use crate::graph::convolver::{
    CONVOLVER_KAUTZ_ONSET_DELAY_48K, CONVOLVER_SEG_TAPS, CONVOLVER_WORK_LEN,
};
use crate::math::flush_recursive_state_f32;
use crate::types::Tier;

use super::super::layout::SonaHotSlab;

#[inline(always)]
fn block16_to_4x4(b: &[f32; 16]) -> [[f32; 4]; 4] {
    let (c, _) = b.as_chunks::<4>();
    [c[0], c[1], c[2], c[3]]
}

#[inline(always)]
fn slice16_to_4x4(s: &[f32]) -> [[f32; 4]; 4] {
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}

#[inline(always)]
fn shift_hist_and_carry<const N: usize>(hist: &mut [f32; N], carry_in: f32) -> f32 {
    let carry_out = hist[N - 1];
    hist.copy_within(0..N - 1, 1);
    hist[0] = carry_in;
    carry_out
}

#[inline(always)]
fn accum_fir_segs<B: GraphBackend>(
    acc_l: &mut f32,
    acc_r: &mut f32,
    wl_segs: &[[[f32; 4]; 4]],
    wr_segs: &[[[f32; 4]; 4]],
    hl_blocks: &[[f32; 16]],
    hr_blocks: &[[f32; 16]],
) {
    for (((wl, wr), hl), hr) in wl_segs.iter().zip(wr_segs).zip(hl_blocks).zip(hr_blocks) {
        let xl = block16_to_4x4(hl);
        let xr = block16_to_4x4(hr);
        let (sl, sr) = B::dot16_stereo(wl, wr, &xl, &xr);
        *acc_l += sl;
        *acc_r += sr;
    }
}

#[inline(always)]
fn accum_fir_work_segs<B: GraphBackend>(
    acc_l: &mut f32,
    acc_r: &mut f32,
    wl_segs: &[[[f32; 4]; 4]],
    wr_segs: &[[[f32; 4]; 4]],
    work_l: &[f32],
    work_r: &[f32],
) {
    for (seg, (wl, wr)) in wl_segs.iter().zip(wr_segs).enumerate() {
        let off = seg * CONVOLVER_SEG_TAPS;
        let xl = slice16_to_4x4(&work_l[off..off + CONVOLVER_SEG_TAPS]);
        let xr = slice16_to_4x4(&work_r[off..off + CONVOLVER_SEG_TAPS]);
        let (sl, sr) = B::dot16_stereo(wl, wr, &xl, &xr);
        *acc_l += sl;
        *acc_r += sr;
    }
}

/// Steps one frame of `ModalConvolver` Early FIR + active `KautzStereoBank4`s (peel / 1-frame path).
#[inline(always)]
pub(super) fn step_convolver_frame<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    pre_l: f32,
    pre_r: f32,
) -> (f32, f32) {
    let tier = slab.lite.tier;
    let cold_ext = slab.cold_fir_ext();
    let (early, onset) = if cold_ext {
        slab.cold.as_mut_slice()[0]
            .convolver_fir_ext
            .step_frame(pre_l, pre_r)
    } else {
        let c_lite_l =
            shift_hist_and_carry(&mut slab.lite.fir_hist_l, flush_recursive_state_f32(pre_l));
        let c_lite_r =
            shift_hist_and_carry(&mut slab.lite.fir_hist_r, flush_recursive_state_f32(pre_r));
        let onset = (slab.lite.fir_hist_l[16], slab.lite.fir_hist_r[16]);

        let mut early_l = 0.0_f32;
        let mut early_r = 0.0_f32;
        accum_fir_segs::<B>(
            &mut early_l,
            &mut early_r,
            &slab.lite.fir_weights_l,
            &slab.lite.fir_weights_r,
            slab.lite.fir_hist_l.as_chunks::<16>().0,
            slab.lite.fir_hist_r.as_chunks::<16>().0,
        );
        if tier >= Tier::Full {
            let c_full_l = shift_hist_and_carry(&mut slab.full_delta.fir_hist_l_full, c_lite_l);
            let c_full_r = shift_hist_and_carry(&mut slab.full_delta.fir_hist_r_full, c_lite_r);
            accum_fir_segs::<B>(
                &mut early_l,
                &mut early_r,
                &slab.full_delta.fir_weights_l_full,
                &slab.full_delta.fir_weights_r_full,
                core::slice::from_ref(&slab.full_delta.fir_hist_l_full),
                core::slice::from_ref(&slab.full_delta.fir_hist_r_full),
            );
            if tier >= Tier::Pro {
                let c_pro_l = shift_hist_and_carry(&mut slab.pro_delta.fir_hist_l_pro, c_full_l);
                let c_pro_r = shift_hist_and_carry(&mut slab.pro_delta.fir_hist_r_pro, c_full_r);
                accum_fir_segs::<B>(
                    &mut early_l,
                    &mut early_r,
                    &slab.pro_delta.fir_weights_l_pro,
                    &slab.pro_delta.fir_weights_r_pro,
                    slab.pro_delta.fir_hist_l_pro.as_chunks::<16>().0,
                    slab.pro_delta.fir_hist_r_pro.as_chunks::<16>().0,
                );
                if tier == Tier::Ultra {
                    shift_hist_and_carry(&mut slab.ultra_delta.fir_hist_l_ultra, c_pro_l);
                    shift_hist_and_carry(&mut slab.ultra_delta.fir_hist_r_ultra, c_pro_r);
                    accum_fir_segs::<B>(
                        &mut early_l,
                        &mut early_r,
                        &slab.ultra_delta.fir_weights_l_ultra,
                        &slab.ultra_delta.fir_weights_r_ultra,
                        slab.ultra_delta.fir_hist_l_ultra.as_chunks::<16>().0,
                        slab.ultra_delta.fir_hist_r_ultra.as_chunks::<16>().0,
                    );
                }
            }
        }
        ((early_l, early_r), onset)
    };

    let active_banks = slab.cold().active_kautz_blocks;
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for b in 0..active_banks {
        let (yl, yr) = KautzStereoBank4::step_banks_stereo_inline::<B>(
            core::slice::from_mut(slab.kautz_bank_mut(b)),
            onset.0,
            onset.1,
        );
        sum_l += yl;
        sum_r += yr;
    }
    (early.0 + sum_l, early.1 + sum_r)
}

/// Steps 4 frames of `ModalConvolver` Early FIR (shifting hot `48 kHz` FIR histories once by `4`
/// against pre-packed slab weight tiles) and active `KautzStereoBank4`s (loaded/stored once per
/// bank across the 4 frames), with `0-ULP` parity against 4× [`step_convolver_frame`].
#[inline(always)]
pub(super) fn step_convolver_chunk4<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    pre_lr: [[f32; 2]; 4],
) -> [[f32; 2]; 4] {
    let tier = slab.lite.tier;
    let mut early_lr = [[0.0_f32; 2]; 4];
    let mut onset_lr = [[0.0_f32; 2]; 4];

    if slab.cold_fir_ext() {
        slab.cold_mut()
            .convolver_fir_ext
            .step_chunk::<4>(&pre_lr, &mut early_lr, &mut onset_lr);
    } else {
        let mut work_l = [0.0_f32; CONVOLVER_WORK_LEN];
        let mut work_r = [0.0_f32; CONVOLVER_WORK_LEN];

        for f in 0..4 {
            work_l[3 - f] = flush_recursive_state_f32(pre_lr[f][0]);
            work_r[3 - f] = flush_recursive_state_f32(pre_lr[f][1]);
        }
        work_l[4..36].copy_from_slice(&slab.lite.fir_hist_l);
        work_r[4..36].copy_from_slice(&slab.lite.fir_hist_r);
        if tier >= Tier::Full {
            work_l[36..52].copy_from_slice(&slab.full_delta.fir_hist_l_full);
            work_r[36..52].copy_from_slice(&slab.full_delta.fir_hist_r_full);
        }
        if tier >= Tier::Pro {
            work_l[52..100].copy_from_slice(&slab.pro_delta.fir_hist_l_pro);
            work_r[52..100].copy_from_slice(&slab.pro_delta.fir_hist_r_pro);
        }
        if tier == Tier::Ultra {
            work_l[100..132].copy_from_slice(&slab.ultra_delta.fir_hist_l_ultra);
            work_r[100..132].copy_from_slice(&slab.ultra_delta.fir_hist_r_ultra);
        }

        for f in 0..4 {
            let start = 3 - f;
            onset_lr[f] = [
                work_l[start + CONVOLVER_KAUTZ_ONSET_DELAY_48K],
                work_r[start + CONVOLVER_KAUTZ_ONSET_DELAY_48K],
            ];
            let mut early_l = 0.0_f32;
            let mut early_r = 0.0_f32;
            accum_fir_work_segs::<B>(
                &mut early_l,
                &mut early_r,
                &slab.lite.fir_weights_l,
                &slab.lite.fir_weights_r,
                &work_l[start..],
                &work_r[start..],
            );
            if tier >= Tier::Full {
                accum_fir_work_segs::<B>(
                    &mut early_l,
                    &mut early_r,
                    &slab.full_delta.fir_weights_l_full,
                    &slab.full_delta.fir_weights_r_full,
                    &work_l[start + 32..],
                    &work_r[start + 32..],
                );
                if tier >= Tier::Pro {
                    accum_fir_work_segs::<B>(
                        &mut early_l,
                        &mut early_r,
                        &slab.pro_delta.fir_weights_l_pro,
                        &slab.pro_delta.fir_weights_r_pro,
                        &work_l[start + 48..],
                        &work_r[start + 48..],
                    );
                    if tier == Tier::Ultra {
                        accum_fir_work_segs::<B>(
                            &mut early_l,
                            &mut early_r,
                            &slab.ultra_delta.fir_weights_l_ultra,
                            &slab.ultra_delta.fir_weights_r_ultra,
                            &work_l[start + 96..],
                            &work_r[start + 96..],
                        );
                    }
                }
            }
            early_lr[f] = [early_l, early_r];
        }

        slab.lite.fir_hist_l.copy_from_slice(&work_l[0..32]);
        slab.lite.fir_hist_r.copy_from_slice(&work_r[0..32]);
        if tier >= Tier::Full {
            slab.full_delta
                .fir_hist_l_full
                .copy_from_slice(&work_l[32..48]);
            slab.full_delta
                .fir_hist_r_full
                .copy_from_slice(&work_r[32..48]);
        }
        if tier >= Tier::Pro {
            slab.pro_delta
                .fir_hist_l_pro
                .copy_from_slice(&work_l[48..96]);
            slab.pro_delta
                .fir_hist_r_pro
                .copy_from_slice(&work_r[48..96]);
        }
        if tier == Tier::Ultra {
            slab.ultra_delta
                .fir_hist_l_ultra
                .copy_from_slice(&work_l[96..128]);
            slab.ultra_delta
                .fir_hist_r_ultra
                .copy_from_slice(&work_r[96..128]);
        }
    }

    let active_banks = slab.cold().active_kautz_blocks;
    let mut kautz_sum = [[0.0_f32; 2]; 4];
    for b in 0..active_banks {
        let bank = slab.kautz_bank_mut(b);
        let mut ks = bank.to_kernel_state::<B>();
        for f in 0..4 {
            let (yl, yr) =
                step_kautz_quad_stereo8::<B::Stereo8>(&mut ks, onset_lr[f][0], onset_lr[f][1]);
            kautz_sum[f][0] += yl;
            kautz_sum[f][1] += yr;
        }
        bank.store_kernel_state::<B>(&ks);
    }

    let mut out = [[0.0_f32; 2]; 4];
    for f in 0..4 {
        out[f] = [
            early_lr[f][0] + kautz_sum[f][0],
            early_lr[f][1] + kautz_sum[f][1],
        ];
    }
    out
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_STEREO_OUT,
        DnamBinBuildInput, DnamBinView, encode_dnam_bin_into,
    };
    use crate::graph::slab::hydrate::hydrate_slab_owned;
    use crate::math::{DefaultMathBackend, ScalarBackend};
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::F32;

    fn verify_convolver_chunk4_parity<B: GraphBackend>(tier: Tier, host_hz: f32) {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0x000F);
        for i in 0..32 {
            a0.conv_fir_lite.fir_l[i] = F32::new(((i as f32) + 1.0) * 0.004);
            a0.conv_fir_lite.fir_r[i] = F32::new(-((i as f32) + 2.0) * 0.003);
        }
        for b in 0..2 {
            for p in 0..4 {
                a0.conv_kautz_lite[b].r_low[p] = F32::new(0.80 - 0.05 * (p as f32));
                a0.conv_kautz_lite[b].dr[p] = F32::new(0.05);
                a0.conv_kautz_lite[b].theta[p] = F32::new(0.15 + 0.1 * (p as f32));
                a0.conv_kautz_lite[b].b_gamma[p] = F32::new(0.25);
                a0.conv_kautz_lite[b].c_re_l[p] = F32::new(0.4);
                a0.conv_kautz_lite[b].c_im_l[p] = F32::new(-0.15);
                a0.conv_kautz_lite[b].c_re_r[p] = F32::new(-0.3);
                a0.conv_kautz_lite[b].c_im_r[p] = F32::new(0.2);
            }
        }
        let mut a1 = Arena1FullDeltaWire::new_zeroed();
        for i in 0..16 {
            a1.conv_fir_full.fir_l[i] = F32::new(0.002 * ((i as f32) + 1.0));
            a1.conv_fir_full.fir_r[i] = F32::new(-0.0015 * ((i as f32) + 1.0));
        }
        let mut a2 = Arena2ProDeltaWire::new_zeroed();
        for i in 0..48 {
            a2.conv_fir_pro.fir_l[i] = F32::new(-0.001 * ((i as f32) + 1.0));
            a2.conv_fir_pro.fir_r[i] = F32::new(0.0012 * ((i as f32) + 1.0));
        }
        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        for i in 0..32 {
            a3.conv_fir_ultra.fir_l[i] = F32::new(0.0008 * ((i as f32) + 1.0));
            a3.conv_fir_ultra.fir_r[i] = F32::new(-0.0009 * ((i as f32) + 1.0));
        }
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        };
        let mut bin_buf = alloc::vec![0_u8; 32_768];
        let bin_len = encode_dnam_bin_into(&input, &mut bin_buf).unwrap();
        let view = DnamBinView::parse(&bin_buf[..bin_len]).unwrap();

        let mut slab_a = hydrate_slab_owned::<B>(&view, tier, host_hz, false).unwrap();
        let mut slab_b = hydrate_slab_owned::<B>(&view, tier, host_hz, false).unwrap();

        // Run 6 consecutive 4-frame chunks (>16 onset delay + full Ultra 128-tap depth).
        for chunk in 0..6 {
            let mut pre_lr = [[0.0_f32; 2]; 4];
            for f in 0..4 {
                let n = (chunk * 4 + f) as f32;
                pre_lr[f] = [0.35 * (n * 0.17 - 0.9), -0.25 * (n * 0.23 - 1.1)];
            }
            let chunk_out = step_convolver_chunk4(&mut slab_a, pre_lr);
            for (f, &[pl, pr]) in pre_lr.iter().enumerate() {
                let (fl, fr) = step_convolver_frame(&mut slab_b, pl, pr);
                assert_eq!(chunk_out[f][0].to_bits(), fl.to_bits());
                assert_eq!(chunk_out[f][1].to_bits(), fr.to_bits());
            }
        }
        assert_eq!(
            slab_a.lite.fir_hist_l.map(f32::to_bits),
            slab_b.lite.fir_hist_l.map(f32::to_bits)
        );
        assert_eq!(
            slab_a.ultra_delta.fir_hist_r_ultra.map(f32::to_bits),
            slab_b.ultra_delta.fir_hist_r_ultra.map(f32::to_bits)
        );
    }

    #[test]
    fn test_step_convolver_chunk4_zero_ulp_parity_all_tiers_and_high_sr() {
        for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            verify_convolver_chunk4_parity::<ScalarBackend>(tier, 48_000.0);
            verify_convolver_chunk4_parity::<DefaultMathBackend>(tier, 48_000.0);
        }
        verify_convolver_chunk4_parity::<DefaultMathBackend>(Tier::Pro, 96_000.0);
    }
}
