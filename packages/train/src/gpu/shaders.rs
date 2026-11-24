//! Embedded AOT-compiled WGSL compute shaders, zero-`serde` `manifest.json` verification
//! (`sha2::Sha256` + `<= 15,872 B` workgroup SRAM checks), and browser WebGPU `enable subgroups;`
//! helper (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use sha2::{Digest, Sha256};

use super::abi::{
    ADJOINT_CORE_SRAM_BYTES, FUSED_CORE_SRAM_BYTES, GABOR_BEZOUT_SRAM_BYTES, INNER_OPT_SRAM_BYTES,
    MAX_WORKGROUP_SRAM_BYTES, OZAKI_GRAM_SRAM_BYTES, PRESCREEN_SRAM_BYTES, TILE_MIX_SRAM_BYTES,
};
use crate::error::TrainError;

/// Canonical checked-in WGSL manifest JSON emitted by `tools/sona-shaders`.
pub const MANIFEST_JSON: &str = include_str!("../../../../tools/sona-shaders/wgsl/manifest.json");

const PORTABLE_PRESCREEN: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/prescreen_population.wgsl");
const PORTABLE_FUSED_CORE: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/fused_core_scan_256.wgsl");
const PORTABLE_ADJOINT: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/adjoint_scan_256.wgsl");
const PORTABLE_TILE_MIX: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/tile_mix_fwd_adj.wgsl");
const PORTABLE_GABOR: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/gabor_bezout_grad.wgsl");
const PORTABLE_OZAKI: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/ozaki_gram_rhs_i32.wgsl");
const PORTABLE_INNER_OPT: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/inner_opt_step.wgsl");

const SUBGROUPS_PRESCREEN: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/prescreen_population.wgsl");
const SUBGROUPS_FUSED_CORE: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/fused_core_scan_256.wgsl");
const SUBGROUPS_ADJOINT: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/adjoint_scan_256.wgsl");
const SUBGROUPS_TILE_MIX: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/tile_mix_fwd_adj.wgsl");
const SUBGROUPS_GABOR: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/gabor_bezout_grad.wgsl");
const SUBGROUPS_OZAKI: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/ozaki_gram_rhs_i32.wgsl");
const SUBGROUPS_INNER_OPT: &str =
    include_str!("../../../../tools/sona-shaders/wgsl/subgroups/inner_opt_step.wgsl");

/// Identifier for the 7 canonical 32-bit GPU/CPU compute kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelId {
    /// Seed population prescreening (`8,192 B` SRAM).
    PrescreenPopulation,
    /// Fused CoreQuad forward tile scan with checkpointing (`15,360 B` SRAM).
    FusedCoreScan256,
    /// 3-pass multi-tile reverse BPTT adjoint scan (`14,848 B` SRAM).
    AdjointScan256,
    /// Wave-level inter-block DAG forward mix, pullback, and weight grad reduction (`2,048 B` SRAM).
    TileMixFwdAdj,
    /// Multi-band Bézout FIR (`fft_size == 0`) or radix-2 Gabor FFT (`fft_size > 0`) loss & grad (`11,264 B` SRAM).
    GaborBezoutGrad,
    /// Integer `i32` Ozaki `(2^8, 2^16, 2^24)` Gram and RHS contraction (`5,120 B` SRAM).
    OzakiGramRhsI32,
    /// Multi-candidate `(1, num_candidates, 1)` strided AdamW / L-SSBroyden / STE-fp16 inner step (`2,560 B` SRAM).
    InnerOptStep,
}

impl KernelId {
    /// Ordered slice of all 7 canonical compute kernels.
    pub const ALL: [Self; 7] = [
        Self::PrescreenPopulation,
        Self::FusedCoreScan256,
        Self::AdjointScan256,
        Self::TileMixFwdAdj,
        Self::GaborBezoutGrad,
        Self::OzakiGramRhsI32,
        Self::InnerOptStep,
    ];

    /// WGSL compute entry point name matching `manifest.json`.
    #[inline]
    #[must_use]
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::PrescreenPopulation => "prescreen_population",
            Self::FusedCoreScan256 => "fused_core_scan_256",
            Self::AdjointScan256 => "adjoint_scan_256",
            Self::TileMixFwdAdj => "tile_mix_fwd_adj",
            Self::GaborBezoutGrad => "gabor_bezout_grad",
            Self::OzakiGramRhsI32 => "ozaki_gram_rhs_i32",
            Self::InnerOptStep => "inner_opt_step",
        }
    }

    /// Expected per-workgroup shared memory footprint in bytes (`<= 15,872 B`).
    #[inline]
    #[must_use]
    pub const fn expected_sram_bytes(self) -> usize {
        match self {
            Self::PrescreenPopulation => PRESCREEN_SRAM_BYTES as usize,
            Self::FusedCoreScan256 => FUSED_CORE_SRAM_BYTES,
            Self::AdjointScan256 => ADJOINT_CORE_SRAM_BYTES,
            Self::TileMixFwdAdj => TILE_MIX_SRAM_BYTES as usize,
            Self::GaborBezoutGrad => GABOR_BEZOUT_SRAM_BYTES as usize,
            Self::OzakiGramRhsI32 => OZAKI_GRAM_SRAM_BYTES as usize,
            Self::InnerOptStep => INNER_OPT_SRAM_BYTES as usize,
        }
    }
}

/// Target WGSL shader compilation variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderVariant {
    /// Strict baseline WebGPU shared-memory reductions without subgroup ops.
    Portable,
    /// Hardware subgroup reductions (requiring `wgpu::Features::SUBGROUP` / browser `subgroups`).
    Subgroups,
}

impl ShaderVariant {
    /// Human-readable variant identifier matching `manifest.json` keys.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Subgroups => "subgroups",
        }
    }
}

/// Returns the checked-in embedded WGSL source for `(id, variant)`.
#[inline]
#[must_use]
pub const fn embedded_wgsl(id: KernelId, variant: ShaderVariant) -> &'static str {
    match (variant, id) {
        (ShaderVariant::Portable, KernelId::PrescreenPopulation) => PORTABLE_PRESCREEN,
        (ShaderVariant::Portable, KernelId::FusedCoreScan256) => PORTABLE_FUSED_CORE,
        (ShaderVariant::Portable, KernelId::AdjointScan256) => PORTABLE_ADJOINT,
        (ShaderVariant::Portable, KernelId::TileMixFwdAdj) => PORTABLE_TILE_MIX,
        (ShaderVariant::Portable, KernelId::GaborBezoutGrad) => PORTABLE_GABOR,
        (ShaderVariant::Portable, KernelId::OzakiGramRhsI32) => PORTABLE_OZAKI,
        (ShaderVariant::Portable, KernelId::InnerOptStep) => PORTABLE_INNER_OPT,
        (ShaderVariant::Subgroups, KernelId::PrescreenPopulation) => SUBGROUPS_PRESCREEN,
        (ShaderVariant::Subgroups, KernelId::FusedCoreScan256) => SUBGROUPS_FUSED_CORE,
        (ShaderVariant::Subgroups, KernelId::AdjointScan256) => SUBGROUPS_ADJOINT,
        (ShaderVariant::Subgroups, KernelId::TileMixFwdAdj) => SUBGROUPS_TILE_MIX,
        (ShaderVariant::Subgroups, KernelId::GaborBezoutGrad) => SUBGROUPS_GABOR,
        (ShaderVariant::Subgroups, KernelId::OzakiGramRhsI32) => SUBGROUPS_OZAKI,
        (ShaderVariant::Subgroups, KernelId::InnerOptStep) => SUBGROUPS_INNER_OPT,
    }
}

/// Prepares WGSL source for browser `navigator.gpu` WebGPU execution by prepending `enable subgroups;`
/// only when targeting [`ShaderVariant::Subgroups`] and missing an explicit directive.
#[must_use]
pub fn prepare_for_browser_webgpu(wgsl: &str, variant: ShaderVariant) -> String {
    if matches!(variant, ShaderVariant::Subgroups)
        && wgsl.contains("subgroup")
        && !wgsl.contains("enable subgroups;")
    {
        let mut out = String::with_capacity("enable subgroups;\n".len() + wgsl.len());
        out.push_str("enable subgroups;\n");
        out.push_str(wgsl);
        out
    } else {
        wgsl.to_owned()
    }
}

fn extract_section<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let key_pos = json.find(&needle)?;
    let rest = &json[key_pos + needle.len()..];
    let arr_start = rest.find('[')?;
    let bytes = rest.as_bytes();
    let mut depth = 0_i32;
    let mut in_str = false;
    let mut escaped = false;
    for (idx, &b) in bytes.iter().enumerate().skip(arr_start) {
        if escaped {
            escaped = false;
            continue;
        }
        if in_str {
            if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[arr_start..=idx]);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_json_str_field<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\"");
    let pos = obj.find(&pat)?;
    let after = obj[pos + pat.len()..].trim_start();
    let after_colon = after.strip_prefix(':')?.trim_start();
    let body = after_colon.strip_prefix('"')?;
    let end = body.find('"')?;
    Some(&body[..end])
}

fn find_json_usize_field(obj: &str, key: &str) -> Option<usize> {
    let pat = format!("\"{key}\"");
    let pos = obj.find(&pat)?;
    let after = obj[pos + pat.len()..].trim_start();
    let after_colon = after.strip_prefix(':')?.trim_start();
    let digits: String = after_colon
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn decode_hex_32(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut out = [0_u8; 32];
    for i in 0..32 {
        let pair = core::str::from_utf8(&bytes[i * 2..i * 2 + 2]).ok()?;
        out[i] = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(out)
}

fn split_entry_objects(section: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = section.as_bytes();
    let mut depth = 0_i32;
    let mut start = None;
    for (idx, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0
                    && let Some(s) = start.take()
                {
                    out.push(&section[s..=idx]);
                }
            }
            _ => {}
        }
    }
    out
}

/// Verifies all 14 embedded `.wgsl` shader sources against the given `manifest_json` string
/// (`sha2::Sha256` digest equality and `workgroup_sram_bytes <= 15,872 B`).
pub fn verify_manifest_digests(manifest_json: &str) -> Result<(), TrainError> {
    for variant in [ShaderVariant::Portable, ShaderVariant::Subgroups] {
        let section =
            extract_section(manifest_json, variant.label()).ok_or(TrainError::KernelMismatch {
                context: "wgsl_manifest",
                detail: "missing portable or subgroups array section",
            })?;
        let entries = split_entry_objects(section);
        for id in KernelId::ALL {
            let entry_name = id.entry_point();
            let obj = entries
                .iter()
                .copied()
                .find(|e| find_json_str_field(e, "name") == Some(entry_name))
                .ok_or(TrainError::KernelMismatch {
                    context: entry_name,
                    detail: "missing kernel entry in wgsl manifest",
                })?;

            let sram = find_json_usize_field(obj, "workgroup_sram_bytes").ok_or(
                TrainError::KernelMismatch {
                    context: entry_name,
                    detail: "missing workgroup_sram_bytes in manifest",
                },
            )?;
            if sram > MAX_WORKGROUP_SRAM_BYTES || sram != id.expected_sram_bytes() {
                return Err(TrainError::KernelMismatch {
                    context: entry_name,
                    detail: "workgroup_sram_bytes violates manifest/ABI bound",
                });
            }

            let expected_hex =
                find_json_str_field(obj, "sha256").ok_or(TrainError::KernelMismatch {
                    context: entry_name,
                    detail: "missing sha256 field in manifest",
                })?;
            let expected = decode_hex_32(expected_hex).ok_or(TrainError::KernelMismatch {
                context: entry_name,
                detail: "invalid hex sha256 digest in manifest",
            })?;
            let actual: [u8; 32] = Sha256::digest(embedded_wgsl(id, variant).as_bytes()).into();
            if actual != expected {
                return Err(TrainError::ShaderManifestMismatch {
                    kernel: entry_name,
                    expected,
                    actual,
                });
            }
        }
    }
    Ok(())
}

/// Verifies all 14 embedded `.wgsl` shaders against the checked-in `tools/sona-shaders/wgsl/manifest.json`.
pub fn verify_embedded_shader_digests() -> Result<(), TrainError> {
    verify_manifest_digests(MANIFEST_JSON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_manifest_digests_and_browser_webgpu_prefix() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        verify_embedded_shader_digests().expect("checked-in WGSL manifest must verify cleanly");

        let sub_ozaki = embedded_wgsl(KernelId::OzakiGramRhsI32, ShaderVariant::Subgroups);
        let browser_sub = prepare_for_browser_webgpu(sub_ozaki, ShaderVariant::Subgroups);
        assert!(browser_sub.starts_with("enable subgroups;\n"));
        let browser_sub_idem = prepare_for_browser_webgpu(&browser_sub, ShaderVariant::Subgroups);
        assert_eq!(browser_sub, browser_sub_idem);

        let sub_core = embedded_wgsl(KernelId::FusedCoreScan256, ShaderVariant::Subgroups);
        assert_eq!(
            prepare_for_browser_webgpu(sub_core, ShaderVariant::Subgroups),
            sub_core
        );
        let port_wgsl = embedded_wgsl(KernelId::OzakiGramRhsI32, ShaderVariant::Portable);
        assert_eq!(
            prepare_for_browser_webgpu(port_wgsl, ShaderVariant::Portable),
            port_wgsl
        );

        assert_eq!(
            verify_manifest_digests("{}"),
            Err(TrainError::KernelMismatch {
                context: "wgsl_manifest",
                detail: "missing portable or subgroups array section",
            })
        );
        let sha_key = "\"sha256\": \"";
        let pre_sha_idx = MANIFEST_JSON
            .find(sha_key)
            .expect("manifest must contain prescreen_population sha256")
            + sha_key.len();
        let pre_sha6 = &MANIFEST_JSON[pre_sha_idx..pre_sha_idx + 6];
        let replacement_hex = if pre_sha6 == "000000" {
            "111111"
        } else {
            "000000"
        };
        let tampered = MANIFEST_JSON.replacen(pre_sha6, replacement_hex, 1);
        assert!(matches!(
            verify_manifest_digests(&tampered),
            Err(TrainError::ShaderManifestMismatch {
                kernel: "prescreen_population",
                ..
            })
        ));
        let bad_hex = MANIFEST_JSON.replacen(&format!("\"{pre_sha6}"), "\"zzzzzz", 1);
        assert!(matches!(
            verify_manifest_digests(&bad_hex),
            Err(TrainError::KernelMismatch {
                detail: "invalid hex sha256 digest in manifest",
                ..
            })
        ));
        let bad_sram = MANIFEST_JSON.replacen("8192", "99999", 1);
        assert!(matches!(
            verify_manifest_digests(&bad_sram),
            Err(TrainError::KernelMismatch {
                detail: "workgroup_sram_bytes violates manifest/ABI bound",
                ..
            })
        ));
        let no_sram = MANIFEST_JSON.replacen("\"workgroup_sram_bytes\": 8192", "\"other\": 1", 1);
        assert!(matches!(
            verify_manifest_digests(&no_sram),
            Err(TrainError::KernelMismatch {
                detail: "missing workgroup_sram_bytes in manifest",
                ..
            })
        ));
        let no_kernel = MANIFEST_JSON.replacen("prescreen_population", "unknown_kernel", 1);
        assert!(matches!(
            verify_manifest_digests(&no_kernel),
            Err(TrainError::KernelMismatch {
                detail: "missing kernel entry in wgsl manifest",
                ..
            })
        ));
    }
}
