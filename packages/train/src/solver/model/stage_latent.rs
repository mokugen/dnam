//! Host `f64`/`f32` representation and bounded wire serialization helpers for `SONA v3.0`
//! stage-latent sensitivity tables (`stage_latent_mods_fp16`, `stage_latent_ctrl_ultra_fp16`, and
//! `SONA_META_CORE_MODS_PRESENT` `stage_latent_core_mods_fp16`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use dnam_vm::bin::{Arena0LiteWire, Arena3UltraDeltaWire, SONA_META_CORE_MODS_PRESENT};
use zerocopy::byteorder::little_endian::U16;

/// Maximum magnitude allowed for global stage latent mods (`stage_latent_mods_fp16[8..48]`).
pub const STAGE_LATENT_GLOBAL_MODS_MAX: f32 = 8.0;
/// Maximum magnitude allowed for ADAA drive scale latent mods (`[-1.5, 1.5]`).
pub const STAGE_LATENT_ADAA_SCALE_MAX: f32 = 1.5;
/// Maximum magnitude allowed for ParaLRU bias shift latent mods (`[-2.0, 2.0]`).
pub const STAGE_LATENT_LRU_BIAS_MAX: f32 = 2.0;
/// Maximum magnitude allowed for `ou_to_z_stage_ultra_fp16` (`[-4.0, 4.0]`).
pub const STAGE_LATENT_OU_ULTRA_MAX: f32 = 4.0;
/// Maximum magnitude allowed for `stage_latent_core_mods_fp16` active entries (`[-4.0, 4.0]`).
pub const STAGE_LATENT_CORE_MODS_MAX: f32 = 4.0;

/// Number of global stage sensitivity vectors stored in `stage_latent_mods_fp16[8..48]`.
pub const STAGE_LATENT_GLOBAL_MOD_COUNT: usize = 5;
/// Start offset of global stage sensitivity vectors inside `stage_latent_mods_fp16`.
pub const STAGE_LATENT_GLOBAL_MODS_START: usize = 8;
/// Exclusive end offset of global stage sensitivity vectors (`8 + 5 * 8 = 48`).
pub const STAGE_LATENT_GLOBAL_MODS_END: usize =
    STAGE_LATENT_GLOBAL_MODS_START + STAGE_LATENT_GLOBAL_MOD_COUNT * 8;
/// Start offset of Lite–Pro `[4][4]` block ADAA drive scale mods inside `stage_latent_mods_fp16`.
pub const STAGE_LATENT_LITE_ADAA_START: usize = STAGE_LATENT_GLOBAL_MODS_END;
/// Start offset of Lite–Pro `[4][4]` block ParaLRU bias mods inside `stage_latent_mods_fp16`.
pub const STAGE_LATENT_LITE_LRU_START: usize = STAGE_LATENT_LITE_ADAA_START + 16;
/// Start offset of Ultra `[8][10]` per-quad ADAA mods inside `stage_latent_ctrl_ultra_fp16`.
pub const STAGE_LATENT_ULTRA_ADAA_START: usize = 0;
/// Start offset of Ultra `[8][10]` per-quad ParaLRU bias mods inside `stage_latent_ctrl_ultra_fp16`.
pub const STAGE_LATENT_ULTRA_LRU_START: usize = STAGE_LATENT_ULTRA_ADAA_START + 8 * 10;
/// Start offset of Ultra `ou_to_z_stage_ultra_fp16[4]` inside `stage_latent_ctrl_ultra_fp16`.
pub const STAGE_LATENT_ULTRA_OU_START: usize = STAGE_LATENT_ULTRA_LRU_START + 8 * 10;
/// Active prefix length of `stage_latent_ctrl_ultra_fp16` (`[0..164]`).
pub const STAGE_LATENT_ULTRA_ACTIVE_END: usize = STAGE_LATENT_ULTRA_OU_START + 4;
/// Active column width per row in `stage_latent_core_mods_fp16` (`40 * 3 = 120`; `120..128` zeroed).
pub const STAGE_LATENT_CORE_MODS_COLS: usize = 120;
/// Row stride of `stage_latent_core_mods_fp16`.
pub const STAGE_LATENT_CORE_MODS_STRIDE: usize = 128;

const _: () = assert!(STAGE_LATENT_GLOBAL_MODS_END == 48);
const _: () = assert!(STAGE_LATENT_LITE_LRU_START + 16 == 80);
const _: () = assert!(STAGE_LATENT_ULTRA_ACTIVE_END == 164);

#[inline]
fn clamp_fp16_bits(val: f32, lo: f32, hi: f32) -> U16 {
    let clamped = if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        0.0
    };
    let clean = if clamped == 0.0 { 0.0 } else { clamped };
    U16::new(f32_to_fp16(clean))
}

#[inline]
fn decode_fp16_clamped(bits: U16, lo: f32, hi: f32) -> f32 {
    let v = fp16_to_f32(bits.get());
    if v.is_finite() && v != 0.0 {
        v.clamp(lo, hi)
    } else {
        0.0
    }
}

/// All trainable stage-latent sensitivity tables across `Lite..Ultra` (`ref[dsp.ctrl.dispatch.modulation-laws]`).
#[derive(Clone, Debug, PartialEq)]
pub struct SonaStageLatentModsF64 {
    /// Cached physical/normalized default knobs in `stage_latent_mods_fp16[0..8]`.
    pub default_knobs_fp16: [u16; 8],
    /// Global stage modulation sensitivities (`[w_svf_g_oct, w_svf_k_delta, w_kautz_oct,
    /// w_kautz_damp, w_tank_psi]` in `stage_latent_mods_fp16[8..48]`).
    pub global_stage_mods: [[f32; 8]; STAGE_LATENT_GLOBAL_MOD_COUNT],
    /// Lite–Pro (`D_s = 4`) block ADAA drive scale mods (`stage_latent_mods_fp16[48..64]`, `[k][b]`).
    pub block_adaa_scale_lite: [[f32; 4]; 4],
    /// Lite–Pro (`D_s = 4`) block ParaLRU bias shift mods (`stage_latent_mods_fp16[64..80]`, `[k][b]`).
    pub block_lru_bias_lite: [[f32; 4]; 4],
    /// Ultra (`D_s = 8`) per-quad ADAA drive scale mods (`stage_latent_ctrl_ultra_fp16[0..80]`).
    pub group_adaa_scale_ultra: [[f32; 10]; 8],
    /// Ultra (`D_s = 8`) per-quad ParaLRU bias shift mods (`stage_latent_ctrl_ultra_fp16[80..160]`).
    pub group_lru_bias_ultra: [[f32; 10]; 8],
    /// Ultra `eff_z_stage[4..8]` OU noise sensitivity (`stage_latent_ctrl_ultra_fp16[160..164]`).
    pub ou_to_z_stage_ultra: [f32; 4],
    /// Optional `Ultra` per-lane `[8][120]` core mods (`stage_latent_core_mods_fp16[0..1024]`,
    /// `[0..40] dg`, `[40..80] dk`, `[80..120] dw`; cols `120..128` strictly zero).
    pub core_mods_ultra: Option<Box<[[f32; STAGE_LATENT_CORE_MODS_COLS]; 8]>>,
}

impl Default for SonaStageLatentModsF64 {
    fn default() -> Self {
        Self {
            default_knobs_fp16: [0_u16; 8],
            global_stage_mods: [[0.0_f32; 8]; STAGE_LATENT_GLOBAL_MOD_COUNT],
            block_adaa_scale_lite: [[0.0_f32; 4]; 4],
            block_lru_bias_lite: [[0.0_f32; 4]; 4],
            group_adaa_scale_ultra: [[0.0_f32; 10]; 8],
            group_lru_bias_ultra: [[0.0_f32; 10]; 8],
            ou_to_z_stage_ultra: [0.0_f32; 4],
            core_mods_ultra: None,
        }
    }
}

impl SonaStageLatentModsF64 {
    /// Returns `true` when `core_mods_ultra` contains at least one nonzero quantized `fp16` weight.
    #[must_use]
    pub fn has_active_core_mods(&self) -> bool {
        let Some(rows) = self.core_mods_ultra.as_deref() else {
            return false;
        };
        rows.iter().flatten().any(|&v| {
            clamp_fp16_bits(v, -STAGE_LATENT_CORE_MODS_MAX, STAGE_LATENT_CORE_MODS_MAX).get() != 0
        })
    }

    /// Returns `true` when any Ultra control-mod or OU sensitivity (`[0..164]`) is nonzero after
    /// bounded `fp16` quantization.
    #[must_use]
    pub fn has_ultra_ctrl_mods(&self) -> bool {
        let any_adaa = self.group_adaa_scale_ultra.iter().flatten().any(|&v| {
            clamp_fp16_bits(v, -STAGE_LATENT_ADAA_SCALE_MAX, STAGE_LATENT_ADAA_SCALE_MAX).get() != 0
        });
        let any_lru = self.group_lru_bias_ultra.iter().flatten().any(|&v| {
            clamp_fp16_bits(v, -STAGE_LATENT_LRU_BIAS_MAX, STAGE_LATENT_LRU_BIAS_MAX).get() != 0
        });
        let any_ou = self.ou_to_z_stage_ultra.iter().any(|&v| {
            clamp_fp16_bits(v, -STAGE_LATENT_OU_ULTRA_MAX, STAGE_LATENT_OU_ULTRA_MAX).get() != 0
        });
        any_adaa || any_lru || any_ou
    }

    /// Clamps all continuous fields strictly onto normative VM limits (`[-8.0, 8.0]`,
    /// `[-1.5, 1.5]`, `[-2.0, 2.0]`, and `[-4.0, 4.0]`).
    pub fn clamp_bounds(&mut self) {
        let clamp_slice = |s: &mut [f32], lo: f32, hi: f32| {
            for v in s {
                *v = if v.is_finite() && *v != 0.0 {
                    v.clamp(lo, hi)
                } else {
                    0.0
                };
            }
        };
        for row in &mut self.global_stage_mods {
            clamp_slice(
                row,
                -STAGE_LATENT_GLOBAL_MODS_MAX,
                STAGE_LATENT_GLOBAL_MODS_MAX,
            );
        }
        for row in &mut self.block_adaa_scale_lite {
            clamp_slice(
                row,
                -STAGE_LATENT_ADAA_SCALE_MAX,
                STAGE_LATENT_ADAA_SCALE_MAX,
            );
        }
        for row in &mut self.block_lru_bias_lite {
            clamp_slice(row, -STAGE_LATENT_LRU_BIAS_MAX, STAGE_LATENT_LRU_BIAS_MAX);
        }
        for row in &mut self.group_adaa_scale_ultra {
            clamp_slice(
                row,
                -STAGE_LATENT_ADAA_SCALE_MAX,
                STAGE_LATENT_ADAA_SCALE_MAX,
            );
        }
        for row in &mut self.group_lru_bias_ultra {
            clamp_slice(row, -STAGE_LATENT_LRU_BIAS_MAX, STAGE_LATENT_LRU_BIAS_MAX);
        }
        clamp_slice(
            &mut self.ou_to_z_stage_ultra,
            -STAGE_LATENT_OU_ULTRA_MAX,
            STAGE_LATENT_OU_ULTRA_MAX,
        );
        if let Some(rows) = self.core_mods_ultra.as_deref_mut() {
            for row in rows.iter_mut() {
                clamp_slice(row, -STAGE_LATENT_CORE_MODS_MAX, STAGE_LATENT_CORE_MODS_MAX);
            }
        }
    }

    /// Writes `default_knobs_fp16[0..8]`, global sensitivities `[8..48]`, and Lite `[48..80]`
    /// into `a0.linear_readout_lite.stage_latent_mods_fp16`.
    pub fn pack_lite_into_wire(&self, stage_latent_mods: &mut [U16; 96]) {
        for (dst, &bits) in stage_latent_mods[..STAGE_LATENT_GLOBAL_MODS_START]
            .iter_mut()
            .zip(self.default_knobs_fp16.iter())
        {
            let v = fp16_to_f32(bits);
            *dst = U16::new(if v.is_finite() { bits } else { 0 });
        }
        for g in 0..STAGE_LATENT_GLOBAL_MOD_COUNT {
            for d in 0..8 {
                stage_latent_mods[STAGE_LATENT_GLOBAL_MODS_START + g * 8 + d] = clamp_fp16_bits(
                    self.global_stage_mods[g][d],
                    -STAGE_LATENT_GLOBAL_MODS_MAX,
                    STAGE_LATENT_GLOBAL_MODS_MAX,
                );
            }
        }
        for d in 0..4 {
            for b in 0..4 {
                stage_latent_mods[STAGE_LATENT_LITE_ADAA_START + d * 4 + b] = clamp_fp16_bits(
                    self.block_adaa_scale_lite[d][b],
                    -STAGE_LATENT_ADAA_SCALE_MAX,
                    STAGE_LATENT_ADAA_SCALE_MAX,
                );
                stage_latent_mods[STAGE_LATENT_LITE_LRU_START + d * 4 + b] = clamp_fp16_bits(
                    self.block_lru_bias_lite[d][b],
                    -STAGE_LATENT_LRU_BIAS_MAX,
                    STAGE_LATENT_LRU_BIAS_MAX,
                );
            }
        }
    }

    /// Writes `stage_latent_ctrl_ultra_fp16[0..164]` (preserving pre-existing `[0..164]` when
    /// `preserve_existing_ctrl_if_empty` is `true` and `self` is empty, always zeroing `[164..256]`)
    /// plus `stage_latent_core_mods_fp16[0..1024]` (zeroing trailing columns `120..128`) into `a3`.
    pub fn pack_ultra_into_arena3(
        &self,
        a3: &mut Arena3UltraDeltaWire,
        preserve_existing_ctrl_if_empty: bool,
    ) {
        let ctrl = &mut a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16;
        ctrl[STAGE_LATENT_ULTRA_ACTIVE_END..].fill(U16::new(0));
        if !preserve_existing_ctrl_if_empty || self.has_ultra_ctrl_mods() {
            for d in 0..8 {
                for q in 0..10 {
                    ctrl[STAGE_LATENT_ULTRA_ADAA_START + d * 10 + q] = clamp_fp16_bits(
                        self.group_adaa_scale_ultra[d][q],
                        -STAGE_LATENT_ADAA_SCALE_MAX,
                        STAGE_LATENT_ADAA_SCALE_MAX,
                    );
                    ctrl[STAGE_LATENT_ULTRA_LRU_START + d * 10 + q] = clamp_fp16_bits(
                        self.group_lru_bias_ultra[d][q],
                        -STAGE_LATENT_LRU_BIAS_MAX,
                        STAGE_LATENT_LRU_BIAS_MAX,
                    );
                }
            }
            for k in 0..4 {
                ctrl[STAGE_LATENT_ULTRA_OU_START + k] = clamp_fp16_bits(
                    self.ou_to_z_stage_ultra[k],
                    -STAGE_LATENT_OU_ULTRA_MAX,
                    STAGE_LATENT_OU_ULTRA_MAX,
                );
            }
        }

        a3.stage_latent_core_mods_fp16.fill(U16::new(0));
        if self.has_active_core_mods()
            && let Some(rows) = self.core_mods_ultra.as_deref()
        {
            for (d, row) in rows.iter().enumerate() {
                for (c, &w) in row.iter().enumerate() {
                    a3.stage_latent_core_mods_fp16[d * STAGE_LATENT_CORE_MODS_STRIDE + c] =
                        clamp_fp16_bits(w, -STAGE_LATENT_CORE_MODS_MAX, STAGE_LATENT_CORE_MODS_MAX);
                }
            }
        }
    }

    /// Decodes wire `stage_latent_mods_fp16`, optional `Arena3`, and `sona_meta_flags` into `Self`.
    #[must_use]
    pub fn decode_from_wires(a0: &Arena0LiteWire, a3: Option<&Arena3UltraDeltaWire>) -> Self {
        let mut out = Self::default();
        let lite_mods = &a0.linear_readout_lite.stage_latent_mods_fp16;
        for (dst, &bits) in out
            .default_knobs_fp16
            .iter_mut()
            .zip(lite_mods[..STAGE_LATENT_GLOBAL_MODS_START].iter())
        {
            let v = fp16_to_f32(bits.get());
            *dst = if v.is_finite() { bits.get() } else { 0 };
        }
        for g in 0..STAGE_LATENT_GLOBAL_MOD_COUNT {
            for d in 0..8 {
                out.global_stage_mods[g][d] = decode_fp16_clamped(
                    lite_mods[STAGE_LATENT_GLOBAL_MODS_START + g * 8 + d],
                    -STAGE_LATENT_GLOBAL_MODS_MAX,
                    STAGE_LATENT_GLOBAL_MODS_MAX,
                );
            }
        }
        for d in 0..4 {
            for b in 0..4 {
                out.block_adaa_scale_lite[d][b] = decode_fp16_clamped(
                    lite_mods[STAGE_LATENT_LITE_ADAA_START + d * 4 + b],
                    -STAGE_LATENT_ADAA_SCALE_MAX,
                    STAGE_LATENT_ADAA_SCALE_MAX,
                );
                out.block_lru_bias_lite[d][b] = decode_fp16_clamped(
                    lite_mods[STAGE_LATENT_LITE_LRU_START + d * 4 + b],
                    -STAGE_LATENT_LRU_BIAS_MAX,
                    STAGE_LATENT_LRU_BIAS_MAX,
                );
            }
        }
        if let Some(a3_w) = a3 {
            let u_mods = &a3_w.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16;
            for d in 0..8 {
                for q in 0..10 {
                    out.group_adaa_scale_ultra[d][q] = decode_fp16_clamped(
                        u_mods[STAGE_LATENT_ULTRA_ADAA_START + d * 10 + q],
                        -STAGE_LATENT_ADAA_SCALE_MAX,
                        STAGE_LATENT_ADAA_SCALE_MAX,
                    );
                    out.group_lru_bias_ultra[d][q] = decode_fp16_clamped(
                        u_mods[STAGE_LATENT_ULTRA_LRU_START + d * 10 + q],
                        -STAGE_LATENT_LRU_BIAS_MAX,
                        STAGE_LATENT_LRU_BIAS_MAX,
                    );
                }
            }
            for k in 0..4 {
                out.ou_to_z_stage_ultra[k] = decode_fp16_clamped(
                    u_mods[STAGE_LATENT_ULTRA_OU_START + k],
                    -STAGE_LATENT_OU_ULTRA_MAX,
                    STAGE_LATENT_OU_ULTRA_MAX,
                );
            }
            if (a0.crossbar.sona_meta_flags.get() & SONA_META_CORE_MODS_PRESENT) != 0 {
                let mut rows = Box::new([[0.0_f32; STAGE_LATENT_CORE_MODS_COLS]; 8]);
                let mut any_nz = false;
                for (d, row) in rows.iter_mut().enumerate() {
                    for (c, dst) in row.iter_mut().enumerate() {
                        let v = decode_fp16_clamped(
                            a3_w.stage_latent_core_mods_fp16[d * STAGE_LATENT_CORE_MODS_STRIDE + c],
                            -STAGE_LATENT_CORE_MODS_MAX,
                            STAGE_LATENT_CORE_MODS_MAX,
                        );
                        if v != 0.0 {
                            any_nz = true;
                        }
                        *dst = v;
                    }
                }
                if any_nz {
                    out.core_mods_ultra = Some(rows);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::FromZeros;

    use super::*;

    #[test]
    fn test_stage_latent_mods_bounds_and_wire_roundtrip() {
        let mut mods = SonaStageLatentModsF64::default();
        mods.default_knobs_fp16[0] = f32_to_fp16(0.5);
        mods.global_stage_mods[0][0] = 12.0;
        mods.global_stage_mods[3][7] = -1.25;
        mods.global_stage_mods[4][2] = 0.75;
        mods.block_adaa_scale_lite[1][2] = 9.0;
        mods.block_lru_bias_lite[3][1] = -9.0;
        mods.group_adaa_scale_ultra[6][9] = 1.25;
        mods.group_lru_bias_ultra[7][8] = -1.75;
        mods.ou_to_z_stage_ultra[2] = 10.0;
        let mut core = Box::new([[0.0_f32; 120]; 8]);
        core[5][119] = -8.0;
        mods.core_mods_ultra = Some(core);
        assert!(mods.has_active_core_mods());
        assert!(mods.has_ultra_ctrl_mods());
        mods.clamp_bounds();
        assert_eq!(mods.global_stage_mods[0][0], 8.0);
        assert_eq!(mods.block_adaa_scale_lite[1][2], 1.5);
        assert_eq!(mods.block_lru_bias_lite[3][1], -2.0);
        assert_eq!(mods.ou_to_z_stage_ultra[2], 4.0);
        assert_eq!(mods.core_mods_ultra.as_ref().unwrap()[5][119], -4.0);

        let mut a0 = Arena0LiteWire::new_zeroed();
        a0.crossbar.sona_meta_flags =
            zerocopy::byteorder::little_endian::U32::new(SONA_META_CORE_MODS_PRESENT);
        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        mods.pack_lite_into_wire(&mut a0.linear_readout_lite.stage_latent_mods_fp16);
        mods.pack_ultra_into_arena3(&mut a3, false);
        let decoded = SonaStageLatentModsF64::decode_from_wires(&a0, Some(&a3));
        assert_eq!(decoded, mods);

        // Preserve existing control_bus ultra ctrl weights when stage_latent_mods has empty ultra ctrl.
        let empty = SonaStageLatentModsF64::default();
        empty.pack_ultra_into_arena3(&mut a3, true);
        assert_ne!(
            a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[162].get(),
            0
        );
        assert!(a3.stage_latent_core_mods_fp16.iter().all(|w| w.get() == 0));
    }
}
