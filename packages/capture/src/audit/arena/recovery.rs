#![forbid(unsafe_code)]
//! Bounded-state zero-input GRU contraction and `+180 dBFS` overdrive silence-recovery checks
//! (`dnam_capture::audit::arena::recovery`).

use dnam_engine::Tier;
use dnam_engine::math::{DefaultMathBackend, MathBackend, SimdVec4};
use dnam_engine::stages::stage2_gru::{MAX_GROUPS, NUM_IN_FEATS_B0, V3GruBlockCold, V3GruBlockHot};
use dnam_engine::stages::{V3CoreInstance, V3MatryoshkaModel};

fn state_max_abs_up_to_group(block_h: &[[f32; 4]; MAX_GROUPS], max_g: usize) -> f32 {
    let mut peak = 0.0_f32;
    for lane_grp in block_h.iter().take(max_g + 1) {
        for &v in lane_grp {
            peak = peak.max(v.abs());
        }
    }
    peak
}

fn gru_group_decays_to_zero(b_idx: usize, g: usize, hot: &V3GruBlockHot) -> bool {
    let zero_feats = [0.0_f32; NUM_IN_FEATS_B0];
    let zero_prev = [[0.0_f32; 4]; MAX_GROUPS];
    let zero_w = [[0.0_f32; 4]; MAX_GROUPS];
    let mut states = Vec::with_capacity(4);
    for sign in [-2.5_f32, 2.5_f32] {
        let mut st = [[0.0_f32; 4]; MAX_GROUPS];
        st[g] = [sign; 4];
        states.push(st);
    }
    for &sign0 in &[1.0_f32, -1.0_f32] {
        let mut st = [[0.0_f32; 4]; MAX_GROUPS];
        for step in 0..32 {
            let s = if (step & 1) == 0 { sign0 } else { -sign0 };
            let feats = [12.0 * s, 12.0 * s, 12.0 * s, 12.0, 12.0];
            let prev = [[2.5 * s; 4]; MAX_GROUPS];
            let (mut ca, mut da) = (
                <DefaultMathBackend as MathBackend>::Vec4::zero(),
                <DefaultMathBackend as MathBackend>::Vec4::zero(),
            );
            hot.step_block::<DefaultMathBackend>(
                b_idx,
                g + 1,
                &feats,
                &prev,
                &mut st,
                &mut ca,
                &mut da,
                &zero_w,
                &zero_w,
            );
        }
        states.push(st);
    }
    for mut block_h in states {
        let mut decayed = false;
        for _ in 0..57_344 {
            let (mut ca, mut da) = (
                <DefaultMathBackend as MathBackend>::Vec4::zero(),
                <DefaultMathBackend as MathBackend>::Vec4::zero(),
            );
            hot.step_block::<DefaultMathBackend>(
                b_idx,
                g + 1,
                &zero_feats,
                &zero_prev,
                &mut block_h,
                &mut ca,
                &mut da,
                &zero_w,
                &zero_w,
            );
            if state_max_abs_up_to_group(&block_h, g) < 1.0e-5 {
                decayed = true;
                break;
            }
        }
        if !decayed && state_max_abs_up_to_group(&block_h, g) >= 1.0e-4 {
            return false;
        }
    }
    true
}

pub(crate) fn ensure_gru_block_zero_input_contraction(
    b_idx: usize,
    blk: &mut V3GruBlockCold,
) -> bool {
    let mut dirty = false;
    for g in 0..MAX_GROUPS {
        for iter in 0..12 {
            let hot = V3GruBlockHot::from_cold::<DefaultMathBackend>(blk, g + 1);
            if gru_group_decays_to_zero(b_idx, g, &hot) {
                break;
            }
            dirty = true;
            let r_lo = g * 4;
            let r_hi = r_lo + 4;
            for r in r_lo..r_hi {
                for c in 0..r_hi {
                    blk.w_rec_z[r][c] *= 0.5;
                    blk.w_rec_r[r][c] *= 0.5;
                    blk.w_rec_n[r][c] *= 0.5;
                }
                if iter >= 6 && blk.b_z[r] < -5.5 {
                    blk.b_z[r] = -5.5;
                }
            }
            blk.clamp_in_place();
        }
    }
    dirty
}

fn tier_overdrive_silence_tail_peak(model: &V3MatryoshkaModel, tier: Tier) -> f32 {
    let cold = model.extract_tier(tier);
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&cold);
    inst.reset_to_dc_equilibrium();

    let mut od = [0.0_f32; 256];
    for (i, slot) in od.iter_mut().enumerate() {
        *slot = if (i & 1) == 0 { 100.0 } else { -100.0 };
    }
    let mut stress = [0.0_f32; 256];
    for (i, slot) in stress.iter_mut().enumerate() {
        *slot = match i % 6 {
            0 | 3 => 1.0e9,
            1 | 4 => -1.0e9,
            2 => 0.0,
            _ => 0.5,
        };
    }
    let zeros = [0.0_f32; 1_024];
    let (mut out_l, mut out_r) = ([0.0_f32; 1_024], [0.0_f32; 1_024]);
    inst.process_block_stereo(&od, &od, &mut out_l[..256], &mut out_r[..256]);
    inst.process_block_stereo(&stress, &stress, &mut out_l[..256], &mut out_r[..256]);
    for _ in 0..64 {
        inst.process_block_stereo(&zeros, &zeros, &mut out_l, &mut out_r);
    }
    let last_l = out_l[1_023];
    let last_r = out_r[1_023];
    if !last_l.is_finite() || !last_r.is_finite() {
        f32::INFINITY
    } else {
        last_l.abs().max(last_r.abs())
    }
}

fn needs_overdrive_repair(model: &V3MatryoshkaModel) -> bool {
    [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .any(|tier| tier_overdrive_silence_tail_peak(model, tier) >= 1.0e-4)
}

fn halve_block_latches(blk: &mut V3GruBlockCold, damp_inputs: bool) {
    for r in 0..blk.w_rec_z.len() {
        for c in 0..=r {
            blk.w_rec_z[r][c] *= 0.5;
            blk.w_rec_r[r][c] *= 0.5;
            blk.w_rec_n[r][c] *= 0.5;
            if damp_inputs {
                blk.w_prev_z[r][c] *= 0.5;
                blk.w_prev_r[r][c] *= 0.5;
                blk.w_prev_n[r][c] *= 0.5;
            }
        }
        if blk.b_z[r] < -5.0 {
            blk.b_z[r] = -5.0;
        }
        if damp_inputs {
            blk.w_in_z[4][r] *= 0.5;
            blk.w_in_r[4][r] *= 0.5;
            blk.w_in_n[4][r] *= 0.5;
        }
    }
    blk.clamp_in_place();
}

/// Ensures `+180 dBFS` overdrive + `NaN`/`Inf` stress bursts decay below `1e-4` after `65,536` zeros
/// without modifying models that already satisfy recovery (`0-ULP` untouched).
pub(crate) fn ensure_model_overdrive_silence_recovery(model: &mut V3MatryoshkaModel) -> bool {
    if !needs_overdrive_repair(model) {
        return false;
    }
    let is_optical =
        (model.mono_out && model.feat_params.key_link > 0.0) || model.feat_params.det_thresh > 0.0;
    if is_optical && model.feat_params.slow_rel_beta < 1.0e-4 {
        let mut cleared_add = false;
        let zero_block_slow_add = |blk: &mut V3GruBlockCold, dirty: &mut bool| {
            for r in 0..blk.w_in_n[0].len() {
                if blk.w_in_n[3][r] != 0.0 || blk.w_in_n[4][r] != 0.0 || blk.k_sag[r] != 0.0 {
                    blk.w_in_n[3][r] = 0.0;
                    blk.w_in_n[4][r] = 0.0;
                    blk.k_sag[r] = 0.0;
                    *dirty = true;
                }
            }
        };
        for blk in &mut model.blocks {
            zero_block_slow_add(blk, &mut cleared_add);
        }
        if let Some(ref mut cal) = model.rate_1x_cal {
            for blk in &mut cal.blocks {
                zero_block_slow_add(blk, &mut cleared_add);
            }
        }
        if cleared_add && !needs_overdrive_repair(model) {
            return true;
        }
    }
    for feat in std::iter::once(&mut model.feat_params).chain(
        model
            .rate_1x_cal
            .as_mut()
            .into_iter()
            .map(|c| &mut c.feat_params),
    ) {
        feat.slow_det_comp = feat.slow_det_comp.max(1.0);
        feat.slow_rel_beta = feat.slow_rel_beta.max(3.5e-4);
        feat.fast_rel_beta = feat.fast_rel_beta.max(5.0e-4);
        feat.clamp_in_place();
    }
    if !needs_overdrive_repair(model) {
        return true;
    }
    for iter in 0..12 {
        let damp_inputs = iter >= 4;
        for blk in &mut model.blocks {
            halve_block_latches(blk, damp_inputs);
        }
        if let Some(ref mut cal) = model.rate_1x_cal {
            for blk in &mut cal.blocks {
                halve_block_latches(blk, damp_inputs);
            }
        }
        if damp_inputs {
            model.stage0.op_p_sub_mix *= 0.5;
            model.stage0.clamp_in_place();
        }
        if !needs_overdrive_repair(model) {
            break;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overdrive_recovery_preserves_clean_and_repairs_slow_latch() {
        let mut clean = V3MatryoshkaModel::identity(false, false);
        assert!(!ensure_model_overdrive_silence_recovery(&mut clean));

        let mut latched = V3MatryoshkaModel::identity(true, false);
        latched.feat_params.slow_det_comp = 0.0;
        latched.feat_params.slow_rel_beta = 1.0e-6;
        latched.feat_params.slow_atk_beta = 0.5;
        latched.blocks[0].w_in_n[4][0] = 2.0;
        latched.readouts[0].pre_cab_w[0][0] = 0.5;
        latched.readouts[3].pre_cab_w[0][0] = 0.5;
        assert!(ensure_model_overdrive_silence_recovery(&mut latched));
        assert!(!needs_overdrive_repair(&latched));

        let mut uncontracted = dnam_engine::stages::stage2_gru::V3GruBlockCold::default();
        for r in 0..4 {
            uncontracted.b_z[r] = -7.5;
            uncontracted.b_n[r] = 3.2;
            for c in 0..=r {
                uncontracted.w_rec_z[r][c] = -6.0;
                uncontracted.w_rec_n[r][c] = 8.0;
            }
        }
        assert!(ensure_gru_block_zero_input_contraction(
            0,
            &mut uncontracted
        ));
    }
}
