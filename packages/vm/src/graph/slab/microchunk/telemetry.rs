//! Block-end cold telemetry & crossfade state accumulator (`#![no_std]`, `#![forbid(unsafe_code)]`,
//! `0-ULP` parity vs per-frame `SonaColdTail` writes).

#![forbid(unsafe_code)]

use crate::graph::GraphBackend;
use crate::math::sanitize_output_sample;

use super::super::layout::{FLAG_XFADE_ACTIVE, SonaHotSlab, TIER_SWITCH_XFADE_SAMPLES};

const INV_XFADE_LEN_F32: f32 = 1.0 / (TIER_SWITCH_XFADE_SAMPLES as f32);

/// Stack-resident telemetry & crossfade state accumulated across a `process_block` / `step_frame`
/// invocation and flushed once to `SonaColdTail` at exit (`0-ULP`).
#[derive(Clone, Copy, Debug)]
pub(super) struct BlockTelemetry {
    pub eff_trim: f32,
    pub mono: bool,
    pub xfade_rem: u16,
    pub xfade_prev_out: [f32; 2],
    pub xfade_prev_core: [f32; 2],
    pub xfade_prev_wet: [f32; 2],
    pub last_in: [f32; 2],
    pub last_s2_dir: [f32; 4],
    pub last_op_p_iq_lr: [f32; 4],
    pub last_op_p_feat: [f32; 4],
    pub last_core_lr: [f32; 2],
    pub last_transport_wet_lr: [f32; 2],
    pub sat_flags: u32,
    pub xfade_cleared: bool,
    pub dirty: bool,
}

impl BlockTelemetry {
    /// Captures block-invariant flags/trim and active crossfade state once at block entry.
    #[inline(always)]
    pub(super) fn begin_block<B: GraphBackend>(slab: &SonaHotSlab<'_, B>) -> Self {
        let tier_idx = slab.lite.tier.index();
        let sign = if slab.wet_polarity_inv() { -1.0 } else { 1.0 };
        let cold = slab.cold();
        let eff_trim = sign * cold.tier_gain_trim[tier_idx];
        let xfade_rem = if slab.xfade_active() {
            cold.xfade_rem
        } else {
            0
        };
        let (xfade_prev_out, xfade_prev_core, xfade_prev_wet) = if xfade_rem > 0 {
            (
                cold.xfade_prev_out,
                cold.xfade_prev_core,
                cold.xfade_prev_wet,
            )
        } else {
            ([0.0; 2], [0.0; 2], [0.0; 2])
        };
        Self {
            eff_trim,
            mono: slab.mono_out(),
            xfade_rem,
            xfade_prev_out,
            xfade_prev_core,
            xfade_prev_wet,
            last_in: [0.0; 2],
            last_s2_dir: [0.0; 4],
            last_op_p_iq_lr: [0.0; 4],
            last_op_p_feat: [0.0; 4],
            last_core_lr: [0.0; 2],
            last_transport_wet_lr: [0.0; 2],
            sat_flags: 0,
            xfade_cleared: false,
            dirty: false,
        }
    }

    /// Flushes accumulated telemetry, crossfade state, and saturation flags to `SonaColdTail` once.
    #[inline(always)]
    pub(super) fn flush_into<B: GraphBackend>(self, slab: &mut SonaHotSlab<'_, B>) {
        if self.xfade_cleared {
            slab.set_flag(FLAG_XFADE_ACTIVE, false);
        }
        let cold = slab.cold_mut();
        if self.dirty {
            cold.xfade_rem = self.xfade_rem;
            cold.xfade_prev_out = self.xfade_prev_out;
            cold.xfade_prev_core = self.xfade_prev_core;
            cold.xfade_prev_wet = self.xfade_prev_wet;
            cold.last_in = self.last_in;
            cold.last_s2_dir = self.last_s2_dir;
            cold.last_op_p_iq_lr = self.last_op_p_iq_lr;
            cold.last_op_p_feat = self.last_op_p_feat;
            cold.last_core_lr = self.last_core_lr;
            cold.last_transport_wet_lr = self.last_transport_wet_lr;
        }
        if self.sat_flags != 0 {
            cold.saturation_flags |= self.sat_flags;
        }
    }
}

#[inline(always)]
fn fold_mono_lr(mono: bool, l: f32, r: f32) -> (f32, f32) {
    if mono {
        let m = 0.5 * (l + r);
        (m, m)
    } else {
        (l, r)
    }
}

#[inline(always)]
pub(super) fn finalize_frame_output_with_telem(
    tank_l: f32,
    tank_r: f32,
    raw_core_lr: [f32; 2],
    raw_wet_lr: [f32; 2],
    tel: &mut BlockTelemetry,
) -> (f32, f32) {
    let mono = tel.mono;
    let (mut out_l, mut out_r) = fold_mono_lr(mono, tank_l, tank_r);
    let (mut core_l, mut core_r) = fold_mono_lr(mono, raw_core_lr[0], raw_core_lr[1]);
    let (mut wet_l, mut wet_r) = fold_mono_lr(mono, raw_wet_lr[0], raw_wet_lr[1]);

    let rem = tel.xfade_rem;
    if rem > 0 {
        let a = f32::from(rem) * INV_XFADE_LEN_F32;
        let one_m_a = 1.0 - a;
        let prev = tel.xfade_prev_out;
        let (prev_l, prev_r) = fold_mono_lr(mono, prev[0], prev[1]);
        out_l = prev_l * a + out_l * one_m_a;
        out_r = prev_r * a + out_r * one_m_a;

        let prev_c = tel.xfade_prev_core;
        let (prev_cl, prev_cr) = fold_mono_lr(mono, prev_c[0], prev_c[1]);
        core_l = prev_cl * a + core_l * one_m_a;
        core_r = prev_cr * a + core_r * one_m_a;

        let prev_w = tel.xfade_prev_wet;
        let (prev_wl, prev_wr) = fold_mono_lr(mono, prev_w[0], prev_w[1]);
        wet_l = prev_wl * a + wet_l * one_m_a;
        wet_r = prev_wr * a + wet_r * one_m_a;

        let next_rem = rem - 1;
        tel.xfade_rem = next_rem;
        if next_rem == 0 {
            tel.xfade_cleared = true;
            tel.xfade_prev_out = [out_l, out_r];
            tel.xfade_prev_core = [core_l, core_r];
            tel.xfade_prev_wet = [wet_l, wet_r];
        }
    } else {
        tel.xfade_prev_out = [out_l, out_r];
        tel.xfade_prev_core = [core_l, core_r];
        tel.xfade_prev_wet = [wet_l, wet_r];
    }

    let eff_trim = tel.eff_trim;
    out_l *= eff_trim;
    out_r *= eff_trim;
    core_l *= eff_trim;
    core_r *= eff_trim;
    wet_l *= eff_trim;
    wet_r *= eff_trim;

    let clean_l = sanitize_output_sample(out_l, &mut tel.sat_flags);
    let clean_r = sanitize_output_sample(out_r, &mut tel.sat_flags);

    let mut part_flags = 0_u32;
    let clean_core_l = sanitize_output_sample(core_l, &mut part_flags);
    let clean_core_r = sanitize_output_sample(core_r, &mut part_flags);
    let clean_wet_l = sanitize_output_sample(wet_l, &mut part_flags);
    let clean_wet_r = sanitize_output_sample(wet_r, &mut part_flags);
    tel.last_core_lr = [clean_core_l, clean_core_r];
    tel.last_transport_wet_lr = [clean_wet_l, clean_wet_r];
    tel.dirty = true;
    (clean_l, clean_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::slab::layout::SonaColdTail;
    use crate::math::ScalarBackend;
    use crate::types::ColdBuf;

    #[test]
    fn test_block_telemetry_flush_preserves_cold_when_not_dirty_and_clears_xfade() {
        assert!((INV_XFADE_LEN_F32 * 48.0 - 1.0).abs() < 1.0e-7);

        let mut cold_slot = [SonaColdTail::<ScalarBackend>::empty()];
        cold_slot[0].last_in = [0.25, -0.5];
        cold_slot[0].xfade_rem = 2;
        cold_slot[0].xfade_prev_out = [0.5, -0.25];
        let mut slab =
            SonaHotSlab::<ScalarBackend>::empty_with_cold(ColdBuf::borrowed(&mut cold_slot));
        slab.set_flag(FLAG_XFADE_ACTIVE, true);

        // Non-dirty flush leaves last_in intact while recording saturation flags.
        let mut tel_noop = BlockTelemetry::begin_block(&slab);
        tel_noop.sat_flags |= 0x4;
        tel_noop.flush_into(&mut slab);
        assert_eq!(slab.cold().last_in, [0.25, -0.5]);
        assert_eq!(slab.cold().saturation_flags, 0x4);
        assert!(slab.xfade_active());

        // Advancing 2 frames clears FLAG_XFADE_ACTIVE upon flush.
        let mut tel = BlockTelemetry::begin_block(&slab);
        let _ = finalize_frame_output_with_telem(0.1, -0.1, [0.1, -0.1], [0.0, 0.0], &mut tel);
        assert!(!tel.xfade_cleared);
        let _ = finalize_frame_output_with_telem(0.2, -0.2, [0.2, -0.2], [0.0, 0.0], &mut tel);
        assert!(tel.xfade_cleared);
        assert_eq!(tel.xfade_rem, 0);
        tel.flush_into(&mut slab);
        assert!(!slab.xfade_active());
        assert_eq!(slab.cold().xfade_rem, 0);
    }
}
