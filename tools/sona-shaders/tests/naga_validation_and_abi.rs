#![forbid(unsafe_code)]

//! Integration tests verifying ABI layouts, SRAM ceilings, checked-in `.wgsl` artifacts,
//! `wgsl/manifest.json` SHA-256 digests, and Naga/WebGPU validation (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use proptest::prelude::*;
use sha2::{Digest, Sha256};
use sona_shaders::abi::{
    ADJOINT_CORE_SRAM_BYTES, AdjointScanUniforms, FEATURE_VEC4_STRIDE, FUSED_CORE_SRAM_BYTES,
    FusedScanUniforms, GABOR_BEZOUT_SRAM_BYTES, GaborBezoutUniforms, INNER_OPT_SRAM_BYTES,
    InnerOptUniforms, LBFGS_MAX_HISTORY_M, MAX_WORKGROUP_SRAM_BYTES, OZAKI_GRAM_SRAM_BYTES,
    OZAKI_WORDS_PER_PAIR, OzakiGramUniforms, PRESCREEN_SEED_PARAM_STRIDE, PRESCREEN_SRAM_BYTES,
    PrescreenUniforms, QUAD_PARAMS_VEC4_STRIDE, TILE_ADJ_BOUNDARY_VEC4_STRIDE,
    TILE_CKPT_VEC4_STRIDE, TILE_FRAMES, UNIFORM_STRUCT_BYTES, WORKGROUP_SIZE,
};
use sona_shaders::compiler::ShaderPaths;
use sona_shaders::manifest::{
    EXPECTED_KERNELS, KernelId, MANIFEST_SCHEMA_VERSION, SHADER_TARGET_TRIPLE,
    ShaderArtifactRecord, ShaderManifest, prepare_for_browser_webgpu,
};
use sona_shaders::validate::validate_kernel_wgsl;
use std::fs;
use zerocopy::{FromBytes, Immutable, IntoBytes};

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn expected_sram_bytes(id: KernelId) -> u32 {
    match id {
        KernelId::PrescreenPopulation => PRESCREEN_SRAM_BYTES,
        KernelId::FusedCoreScan256 => FUSED_CORE_SRAM_BYTES as u32,
        KernelId::AdjointScan256 => ADJOINT_CORE_SRAM_BYTES as u32,
        KernelId::GaborBezoutGrad => GABOR_BEZOUT_SRAM_BYTES,
        KernelId::OzakiGramRhsI32 => OZAKI_GRAM_SRAM_BYTES,
        KernelId::InnerOptStep => INNER_OPT_SRAM_BYTES,
    }
}

fn verify_artifact_records(
    paths: &ShaderPaths,
    records: &[ShaderArtifactRecord],
    uses_subgroups: bool,
) {
    assert_eq!(records.len(), EXPECTED_KERNELS.len());
    for (spec, rec) in EXPECTED_KERNELS.iter().zip(records.iter()) {
        assert_eq!(rec.name, spec.entry_point);
        let expected_rel = if uses_subgroups {
            format!("subgroups/{}.wgsl", spec.entry_point)
        } else {
            format!("{}.wgsl", spec.entry_point)
        };
        assert_eq!(rec.wgsl_path, expected_rel);
        let full_path = paths.wgsl_dir.join(&rec.wgsl_path);
        let wgsl = fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("missing {}: {e}", full_path.display()));
        assert!(
            !wgsl.contains("continuing") && !wgsl.contains("break if"),
            "{} must not emit continuing/break if constructs",
            rec.wgsl_path
        );
        for forbidden in ["f64", "u64", "i64"] {
            assert!(
                !wgsl.contains(forbidden),
                "{} contains forbidden 64-bit token '{forbidden}'",
                rec.wgsl_path
            );
        }
        assert!(
            wgsl.contains("@workgroup_size(256, 1, 1)"),
            "{} missing @workgroup_size(256, 1, 1)",
            rec.wgsl_path
        );
        assert_eq!(rec.sha256, sha256_hex(wgsl.as_bytes()));
        let validated = validate_kernel_wgsl(spec, &wgsl, uses_subgroups)
            .unwrap_or_else(|e| panic!("{} failed validation: {e}", rec.wgsl_path));
        assert_eq!(rec.workgroup_sram_bytes, validated.workgroup_sram_bytes);
        assert_eq!(rec.workgroup_sram_bytes, expected_sram_bytes(spec.id));
        assert_eq!(rec.bindings, validated.bindings);
        assert!(rec.workgroup_sram_bytes <= MAX_WORKGROUP_SRAM_BYTES as u32);
    }
}

#[test]
fn test_abi_uniform_sizes_sram_ceilings_and_browser_helper() {
    assert_eq!(WORKGROUP_SIZE, 256);
    assert_eq!(TILE_FRAMES, 256);
    assert_eq!(UNIFORM_STRUCT_BYTES, 32);
    assert_eq!(MAX_WORKGROUP_SRAM_BYTES, 15_872);
    assert_eq!(FUSED_CORE_SRAM_BYTES, 15_360);
    assert_eq!(ADJOINT_CORE_SRAM_BYTES, 14_848);
    assert_eq!(QUAD_PARAMS_VEC4_STRIDE, 24);
    assert_eq!(TILE_CKPT_VEC4_STRIDE, 8);
    assert_eq!(FEATURE_VEC4_STRIDE, 6);
    assert_eq!(TILE_ADJ_BOUNDARY_VEC4_STRIDE, 16);
    assert_eq!(PRESCREEN_SEED_PARAM_STRIDE, 16);
    assert_eq!(OZAKI_WORDS_PER_PAIR, 6);
    assert_eq!(LBFGS_MAX_HISTORY_M, 8);

    let raw = "@compute @workgroup_size(256, 1, 1)\nfn main() {}\n";
    assert_eq!(prepare_for_browser_webgpu(raw, false), raw);
    let enabled = prepare_for_browser_webgpu(raw, true);
    assert!(enabled.starts_with("enable subgroups;\n"));
    assert_eq!(prepare_for_browser_webgpu(&enabled, true), enabled);
}

#[test]
fn test_checked_in_wgsl_manifest_and_naga_validation() {
    let paths = ShaderPaths::workspace_default();
    let manifest_raw = fs::read_to_string(paths.manifest_path())
        .expect("wgsl/manifest.json must exist after sona-shaders-gen build");
    let manifest: ShaderManifest =
        serde_json::from_str(&manifest_raw).expect("wgsl/manifest.json must deserialize cleanly");
    assert_eq!(manifest.version, MANIFEST_SCHEMA_VERSION);
    assert_eq!(manifest.target, SHADER_TARGET_TRIPLE);

    verify_artifact_records(&paths, &manifest.portable, false);
    verify_artifact_records(&paths, &manifest.subgroups, true);
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
/// verify[sdk.rust.shared-nostd-kernels]
#[test]
fn test_wgpu_native_shader_module_creation_if_adapter_available() {
    let paths = ShaderPaths::workspace_default();
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
    let adapter = match adapter {
        Ok(a) => a,
        Err(err) => {
            if cfg!(target_os = "macos") {
                panic!("Metal GPU adapter must exist on macOS host: {err:?}");
            }
            return;
        }
    };
    let supports_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
    let required_features = if supports_subgroups {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sona-shaders-smoke"),
        required_features,
        ..wgpu::DeviceDescriptor::default()
    }))
    .expect("wgpu device request must succeed");

    for spec in &EXPECTED_KERNELS {
        let portable_wgsl = fs::read_to_string(paths.wgsl_file_path(spec.entry_point, false))
            .expect("portable wgsl must exist");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(spec.entry_point),
            source: wgpu::ShaderSource::Wgsl(portable_wgsl.into()),
        });
        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(spec.entry_point),
            layout: None,
            module: &module,
            entry_point: Some(spec.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if supports_subgroups {
            let sg_wgsl = fs::read_to_string(paths.wgsl_file_path(spec.entry_point, true))
                .expect("subgroup wgsl must exist");
            let sg_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(spec.entry_point),
                source: wgpu::ShaderSource::Wgsl(sg_wgsl.into()),
            });
            let _sg_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(spec.entry_point),
                layout: None,
                module: &sg_module,
                entry_point: Some(spec.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        }
    }
}

fn check_uniform_zerocopy_roundtrip<T>(val: &T) -> Result<(), TestCaseError>
where
    T: FromBytes + IntoBytes + Immutable,
{
    let bytes = val.as_bytes();
    prop_assert_eq!(bytes.len(), UNIFORM_STRUCT_BYTES);
    let parsed = T::read_from_bytes(bytes).unwrap();
    prop_assert_eq!(parsed.as_bytes(), bytes);
    Ok(())
}

proptest! {
    #[test]
    fn prop_uniform_zerocopy_roundtrip_32_bytes(
        a in any::<u32>(),
        b in any::<u32>(),
        c in any::<u32>(),
        d in any::<u32>(),
        e in any::<f32>(),
        f in any::<f32>(),
        g in any::<u32>(),
        h in any::<u32>(),
    ) {
        check_uniform_zerocopy_roundtrip(&PrescreenUniforms {
            num_seeds: a,
            num_frames: b,
            num_quads: c,
            num_stems: d,
            sample_rate_hz: e,
            eps_floor: f,
            _pad0: g,
            _pad1: h,
        })?;
        check_uniform_zerocopy_roundtrip(&FusedScanUniforms {
            num_frames: a,
            num_tiles: b,
            num_quads: c,
            num_stems: d,
            tile_start: g,
            soft_rect_eps: e,
            tiny_crossing_eps: f,
            block_idx: h,
        })?;
        check_uniform_zerocopy_roundtrip(&AdjointScanUniforms {
            num_frames: a,
            num_tiles: b,
            num_quads: c,
            num_stems: d,
            pass_mode: g,
            soft_rect_eps: e,
            tiny_crossing_eps: f,
            block_idx: h,
        })?;
        check_uniform_zerocopy_roundtrip(&GaborBezoutUniforms {
            num_frames: a,
            fft_size: b,
            hop_size: c,
            num_stems: d,
            w_lf: e,
            w_mf: f,
            w_hf: f32::from_bits(g),
            eps_noise: f32::from_bits(h),
        })?;
        check_uniform_zerocopy_roundtrip(&OzakiGramUniforms {
            num_frames: a,
            feature_dim: b,
            target_channels: c,
            chunk_len: d,
            num_chunks: g,
            _pad0: h,
            _pad1: e.to_bits(),
            _pad2: f.to_bits(),
        })?;
        check_uniform_zerocopy_roundtrip(&InnerOptUniforms {
            num_params: a,
            history_m: b,
            step_idx: c,
            line_search_iter: d,
            step_size: e,
            damping_lambda: f,
            min_bound_eps: f32::from_bits(g),
            _pad: h,
        })?;
    }
}
