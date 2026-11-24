//! Normative Role B masking for `default_knobs` (`383..391`) and active stage-latent sensitivity
//! weights (`391..431`, `431..463`, `5_619..5_779`) (`ref[sec.watermark.tardos-code]`).

use dnam_kernels::fp16_to_f32;

use crate::provenance::TARDOS_CASCADE_WEIGHTS;

use super::layout_7680::TARDOS_FP16_SPANS;
use super::{TardosCodebook7680, for_each_tardos_fp16_offset_up_to, read_u16_le};

/// First Role B Tardos coordinate of `stage_latent_mods_fp16[0..8]` (`default_knobs`, always masked).
pub const TARDOS_DEFAULT_KNOBS_START: usize = 383;
/// Exclusive end coordinate of `stage_latent_mods_fp16[0..8]`.
pub const TARDOS_DEFAULT_KNOBS_END: usize = 391;
/// First Role B Tardos coordinate of `stage_latent_mods_fp16[8..48]` (global 5x8 latent mods).
pub const TARDOS_GLOBAL_LATENT_MODS_START: usize = 391;
/// Exclusive end coordinate of `stage_latent_mods_fp16[8..48]`.
pub const TARDOS_GLOBAL_LATENT_MODS_END: usize = 431;
/// First Role B Tardos coordinate of `stage_latent_mods_fp16[48..80]` (Lite ADAA/LRU mods).
pub const TARDOS_LITE_LATENT_MODS_START: usize = 431;
/// Exclusive end coordinate of `stage_latent_mods_fp16[48..80]`.
pub const TARDOS_LITE_LATENT_MODS_END: usize = 463;
/// First Role B Tardos coordinate of `stage_latent_ctrl_ultra_fp16[0..160]` (Ultra ADAA/LRU mods).
pub const TARDOS_ULTRA_LATENT_MODS_START: usize = 5_619;
/// Exclusive end coordinate of `stage_latent_ctrl_ultra_fp16[0..160]`.
pub const TARDOS_ULTRA_LATENT_MODS_END: usize = 5_779;
/// Minimum active stage-latent weight magnitude triggering Role B Tardos masking (`>= 1e-4`).
pub const TARDOS_ACTIVE_LATENT_MASK_THRESH: f32 = 1.0e-4;

const _: () = assert!(TARDOS_DEFAULT_KNOBS_END - TARDOS_DEFAULT_KNOBS_START == 8);
const _: () = assert!(TARDOS_GLOBAL_LATENT_MODS_END - TARDOS_GLOBAL_LATENT_MODS_START == 40);
const _: () = assert!(TARDOS_LITE_LATENT_MODS_END - TARDOS_LITE_LATENT_MODS_START == 32);
const _: () = assert!(TARDOS_ULTRA_LATENT_MODS_END - TARDOS_ULTRA_LATENT_MODS_START == 160);
const _: () = assert!(TARDOS_FP16_SPANS[0].start_idx == TARDOS_DEFAULT_KNOBS_START);
const _: () = assert!(TARDOS_FP16_SPANS[0].end_idx == TARDOS_LITE_LATENT_MODS_END);
const _: () = assert!(TARDOS_FP16_SPANS[8].start_idx == TARDOS_ULTRA_LATENT_MODS_START);
const _: () = assert!(TARDOS_FP16_SPANS[8].end_idx == TARDOS_ULTRA_LATENT_MODS_END);

#[inline]
const fn is_variable_stage_latent_coord(idx: usize) -> bool {
    (idx >= TARDOS_GLOBAL_LATENT_MODS_START && idx < TARDOS_LITE_LATENT_MODS_END)
        || (idx >= TARDOS_ULTRA_LATENT_MODS_START && idx < TARDOS_ULTRA_LATENT_MODS_END)
}

#[inline]
fn mask_stage_latent_coords_with<F: FnMut(usize)>(master_arena: &[u8], mut apply_zero: F) {
    for idx in TARDOS_DEFAULT_KNOBS_START..TARDOS_DEFAULT_KNOBS_END {
        apply_zero(idx);
    }
    for_each_tardos_fp16_offset_up_to(master_arena.len(), |idx, byte_off| {
        if is_variable_stage_latent_coord(idx) {
            let w = fp16_to_f32(read_u16_le(master_arena, byte_off));
            if !w.is_finite() || w.abs() >= TARDOS_ACTIVE_LATENT_MASK_THRESH {
                apply_zero(idx);
            }
        }
    });
}

/// Masks Role B coordinates belonging to `default_knobs` (`383..391`) and active nonzero master
/// stage-latent sensitivity weights (`|w_i| >= 1e-4` in `391..463` and `5_619..5_779`) post
/// codebook generation (`chips`, `accusation_weights`, and optional `envelopes`).
pub fn mask_tardos_stage_latent_active_coords(
    master_arena: &[u8],
    codebook: &mut TardosCodebook7680,
    mut envelopes: Option<&mut [f32; TARDOS_CASCADE_WEIGHTS]>,
) -> usize {
    let cb_len = codebook.chips.len().min(codebook.accusation_weights.len());
    let mut masked = 0_usize;
    mask_stage_latent_coords_with(master_arena, |idx| {
        let mut changed = false;
        if idx < cb_len && (codebook.chips[idx] != 0.0 || codebook.accusation_weights[idx] != 0.0) {
            codebook.chips[idx] = 0.0;
            codebook.accusation_weights[idx] = 0.0;
            changed = true;
        }
        if let Some(env) = envelopes.as_deref_mut()
            && idx < env.len()
            && env[idx] != 0.0
        {
            env[idx] = 0.0;
            changed = true;
        }
        if changed {
            masked += 1;
        }
    });
    masked
}

#[cfg(test)]
mod tests {
    use dnam_kernels::f32_to_fp16;
    use dnam_vm::bin::ARENA_TIER_BYTES;

    use super::*;
    use crate::provenance::watermark::tardos::{
        apply_tardos_watermark_7680_envelopes, compute_emr_masked_tardos_envelopes,
        generate_tardos_code_7680, write_u16_le,
    };

    fn fp16_offset_for_idx(arena_len: usize, target_idx: usize) -> usize {
        let mut found = None;
        for_each_tardos_fp16_offset_up_to(arena_len, |idx, off| {
            if idx == target_idx {
                found = Some(off);
            }
        });
        found.expect("target index present in arena tier")
    }

    #[test]
    fn test_mask_tardos_stage_latent_active_coords_protects_knobs_and_active_weights() {
        let mut master = alloc::vec![0_u8; ARENA_TIER_BYTES[3]];
        let off_395 = fp16_offset_for_idx(master.len(), 395);
        let off_440 = fp16_offset_for_idx(master.len(), 440);
        let off_450 = fp16_offset_for_idx(master.len(), 450);
        let off_5630 = fp16_offset_for_idx(master.len(), 5_630);
        let off_5700 = fp16_offset_for_idx(master.len(), 5_700);
        let off_wro = fp16_offset_for_idx(master.len(), 500);

        write_u16_le(&mut master, off_395, f32_to_fp16(0.125));
        write_u16_le(&mut master, off_440, f32_to_fp16(-0.5));
        write_u16_le(&mut master, off_450, f32_to_fp16(5.0e-5)); // below 1e-4 threshold
        write_u16_le(&mut master, off_5630, f32_to_fp16(0.75));
        write_u16_le(&mut master, off_wro, f32_to_fp16(0.25));
        let _ = off_5700; // left at 0.0

        let mut cb = generate_tardos_code_7680(&[0x42; 32], "model_m23", "lic_m23");
        let gram = [1.0_f64; 256];
        let mut env = compute_emr_masked_tardos_envelopes(&gram, [1.0e-4, 1.0e-4]);
        let masked_count = mask_tardos_stage_latent_active_coords(&master, &mut cb, Some(&mut env));
        assert_eq!(masked_count, 8 + 3); // 383..391 + {395, 440, 5630}

        for idx in TARDOS_DEFAULT_KNOBS_START..TARDOS_DEFAULT_KNOBS_END {
            assert_eq!(cb.chips[idx], 0.0);
            assert_eq!(cb.accusation_weights[idx], 0.0);
            assert_eq!(env[idx], 0.0);
        }
        for active in [395_usize, 440, 5_630] {
            assert_eq!(cb.chips[active], 0.0);
            assert_eq!(cb.accusation_weights[active], 0.0);
            assert_eq!(env[active], 0.0);
        }
        for untouched in [450_usize, 5_700, 500] {
            assert_ne!(cb.chips[untouched], 0.0);
            assert_ne!(cb.accusation_weights[untouched], 0.0);
            assert!(env[untouched] >= 5.0e-5);
        }

        let mut buyer = master.clone();
        apply_tardos_watermark_7680_envelopes(&mut buyer, &cb.chips, &env).unwrap();
        for protected_idx in (383..391).chain([395, 440, 5_630]) {
            let off = fp16_offset_for_idx(master.len(), protected_idx);
            assert_eq!(
                read_u16_le(&master, off),
                read_u16_le(&buyer, off),
                "coord {protected_idx} must remain 0-ULP unperturbed"
            );
        }

        // Truncated Lite arena safely masks only Lite range without indexing Ultra.
        let mut cb_lite = generate_tardos_code_7680(&[0x42; 32], "model_m23", "lic_m23_lite");
        let lite_masked = mask_tardos_stage_latent_active_coords(
            &master[..ARENA_TIER_BYTES[0]],
            &mut cb_lite,
            None,
        );
        assert_eq!(lite_masked, 8 + 2);
        assert_ne!(cb_lite.chips[5_630], 0.0);
    }
}
