#![forbid(unsafe_code)]

//! Offline AOT `cargo-gpu-install` + `spirv-builder` compiler driver and drift checker
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod loop_lowering;

use crate::manifest::{
    EXPECTED_KERNELS, ExpectedKernelSpec, MANIFEST_SCHEMA_VERSION, SHADER_TARGET_TRIPLE,
    ShaderArtifactRecord, ShaderManifest,
};
use crate::validate::validate_kernel_wgsl;
use anyhow::{Context as _, bail, ensure};
use cargo_gpu_install::install::{Install, InstalledBackend};
use sha2::{Digest, Sha256};
use spirv_builder::{Capability, CompileResult, ShaderPanicStrategy, SpirvMetadata};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Filesystem paths used for shader compilation, output emission, and drift checking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderPaths {
    pub tool_root: PathBuf,
    pub gpu_crate_dir: PathBuf,
    pub wgsl_dir: PathBuf,
    pub target_dir: PathBuf,
}

impl ShaderPaths {
    /// Constructs shader paths relative to `tools/sona-shaders`.
    #[must_use]
    pub fn from_tool_root(tool_root: PathBuf) -> Self {
        let gpu_crate_dir = tool_root.join("crates").join("sona-shaders-gpu");
        let wgsl_dir = tool_root.join("wgsl");
        let target_dir = tool_root.join("target").join("spirv-builder");
        Self {
            tool_root,
            gpu_crate_dir,
            wgsl_dir,
            target_dir,
        }
    }

    /// Returns the default workspace paths rooted at `CARGO_MANIFEST_DIR`.
    #[must_use]
    pub fn workspace_default() -> Self {
        Self::from_tool_root(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
    }

    /// Returns the output path of `wgsl/manifest.json`.
    #[must_use]
    pub fn manifest_path(&self) -> PathBuf {
        self.wgsl_dir.join("manifest.json")
    }

    /// Returns the checked-in `.wgsl` file path for a kernel and variant.
    #[must_use]
    pub fn wgsl_file_path(&self, entry_point: &str, uses_subgroups: bool) -> PathBuf {
        let dir = if uses_subgroups {
            self.wgsl_dir.join("subgroups")
        } else {
            self.wgsl_dir.clone()
        };
        dir.join(format!("{entry_point}.wgsl"))
    }
}

/// One compiled, normalized, and Naga-validated WGSL shader ready for emission or comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledKernel {
    pub entry_point: &'static str,
    pub relative_wgsl_path: String,
    pub wgsl: String,
    pub record: ShaderArtifactRecord,
}

/// Complete compiled output for both portable and subgroup variants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledShaderSuite {
    pub portable: Vec<CompiledKernel>,
    pub subgroups: Vec<CompiledKernel>,
    pub manifest: ShaderManifest,
    pub manifest_json: String,
}

/// Normalizes line endings and restores the canonical entry point name when Naga's identifier
/// sanitizer appends a trailing `_` to entry points ending in a digit (e.g. `*_256`, `*_i32`).
pub(crate) fn normalize_wgsl(raw: &str, entry_point: &str) -> String {
    let unix = raw.replace("\r\n", "\n");
    let mangled_decl = format!("\nfn {entry_point}_(");
    let canonical_decl = format!("\nfn {entry_point}(");
    let restored = unix.replacen(&mangled_decl, &canonical_decl, 1);
    let trimmed = restored.trim_end_matches('\n');
    format!("{trimmed}\n")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

fn install_backend(paths: &ShaderPaths) -> anyhow::Result<InstalledBackend> {
    let mut install = Install::from_shader_crate(paths.gpu_crate_dir.clone());
    install.auto_install_rust_toolchain = true;
    install
        .run()
        .context("installing/loading rustc_codegen_spirv backend via cargo-gpu-install")
}

fn find_module_path<'a>(
    modules: &'a BTreeMap<String, PathBuf>,
    entry_point: &str,
) -> Option<&'a PathBuf> {
    if let Some(exact) = modules.get(entry_point) {
        return Some(exact);
    }
    let suffix = format!("::{entry_point}");
    modules
        .iter()
        .find(|(k, _)| k.ends_with(&suffix))
        .map(|(_, path)| path)
}

fn compile_variant(
    backend: &InstalledBackend,
    paths: &ShaderPaths,
    uses_subgroups: bool,
) -> anyhow::Result<Vec<CompiledKernel>> {
    let mut builder = backend.to_spirv_builder(&paths.gpu_crate_dir, SHADER_TARGET_TRIPLE);
    builder = builder
        .release(true)
        .multimodule(true)
        .spirv_metadata(SpirvMetadata::NameVariables)
        .shader_panic_strategy(ShaderPanicStrategy::SilentExit)
        .target_dir_path(paths.target_dir.clone())
        .shader_crate_default_features(false);

    if uses_subgroups {
        builder = builder
            .shader_crate_features(["subgroups".to_owned()])
            .capability(Capability::GroupNonUniform)
            .capability(Capability::GroupNonUniformArithmetic)
            .capability(Capability::GroupNonUniformBallot)
            .capability(Capability::GroupNonUniformShuffle)
            .capability(Capability::GroupNonUniformShuffleRelative)
            .capability(Capability::GroupNonUniformVote);
    }

    let result: CompileResult = builder
        .build()
        .with_context(|| format!("compiling sona-shaders-gpu (subgroups={uses_subgroups})"))?;
    let modules = result.module.unwrap_multi();

    let mut kernels = Vec::with_capacity(EXPECTED_KERNELS.len());
    for spec in &EXPECTED_KERNELS {
        kernels.push(process_compiled_kernel(spec, modules, uses_subgroups)?);
    }
    Ok(kernels)
}

fn process_compiled_kernel(
    spec: &ExpectedKernelSpec,
    modules: &BTreeMap<String, PathBuf>,
    uses_subgroups: bool,
) -> anyhow::Result<CompiledKernel> {
    let src_path = find_module_path(modules, spec.entry_point).with_context(|| {
        let keys: Vec<&str> = modules.keys().map(String::as_str).collect();
        format!(
            "missing compiled entry point '{}' (available: {keys:?})",
            spec.entry_point
        )
    })?;
    let raw_wgsl = fs::read_to_string(src_path)
        .with_context(|| format!("reading compiled WGSL '{}'", src_path.display()))?;
    let normalized = normalize_wgsl(&raw_wgsl, spec.entry_point);
    let wgsl =
        loop_lowering::lower_wgsl_loop_continuing(&normalized, spec.entry_point, uses_subgroups)?;
    let validated = validate_kernel_wgsl(spec, &wgsl, uses_subgroups)?;
    let relative_wgsl_path = if uses_subgroups {
        format!("subgroups/{}.wgsl", spec.entry_point)
    } else {
        format!("{}.wgsl", spec.entry_point)
    };
    let record = ShaderArtifactRecord {
        name: spec.entry_point.to_owned(),
        wgsl_path: relative_wgsl_path.clone(),
        sha256: sha256_hex(wgsl.as_bytes()),
        workgroup_sram_bytes: validated.workgroup_sram_bytes,
        bindings: validated.bindings,
    };
    Ok(CompiledKernel {
        entry_point: spec.entry_point,
        relative_wgsl_path,
        wgsl,
        record,
    })
}

/// Compiles and validates both portable and subgroup shader suites in memory without writing
/// to checked-in `wgsl/` artifacts.
pub fn compile_and_validate_suite(paths: &ShaderPaths) -> anyhow::Result<CompiledShaderSuite> {
    ensure!(
        paths.gpu_crate_dir.is_dir(),
        "shader crate directory '{}' does not exist",
        paths.gpu_crate_dir.display()
    );
    let backend = install_backend(paths)?;
    let portable = compile_variant(&backend, paths, false)?;
    let subgroups = compile_variant(&backend, paths, true)?;
    let manifest = ShaderManifest {
        version: MANIFEST_SCHEMA_VERSION,
        target: SHADER_TARGET_TRIPLE.to_owned(),
        portable: portable.iter().map(|k| k.record.clone()).collect(),
        subgroups: subgroups.iter().map(|k| k.record.clone()).collect(),
    };
    let mut manifest_json =
        serde_json::to_string_pretty(&manifest).context("serializing ShaderManifest")?;
    manifest_json.push('\n');
    Ok(CompiledShaderSuite {
        portable,
        subgroups,
        manifest,
        manifest_json,
    })
}

/// Compiles, validates, and writes checked-in `.wgsl` files and `wgsl/manifest.json`.
pub fn build_shaders(paths: &ShaderPaths) -> anyhow::Result<ShaderManifest> {
    let suite = compile_and_validate_suite(paths)?;
    fs::create_dir_all(paths.wgsl_dir.join("subgroups"))
        .context("creating wgsl/subgroups output directory")?;

    for kernel in &suite.portable {
        let dst = paths.wgsl_file_path(kernel.entry_point, false);
        fs::write(&dst, kernel.wgsl.as_bytes())
            .with_context(|| format!("writing '{}'", dst.display()))?;
    }
    for kernel in &suite.subgroups {
        let dst = paths.wgsl_file_path(kernel.entry_point, true);
        fs::write(&dst, kernel.wgsl.as_bytes())
            .with_context(|| format!("writing '{}'", dst.display()))?;
    }
    let manifest_dst = paths.manifest_path();
    fs::write(&manifest_dst, suite.manifest_json.as_bytes())
        .with_context(|| format!("writing '{}'", manifest_dst.display()))?;
    Ok(suite.manifest)
}

fn compare_file_exact(path: &Path, expected: &str, drifts: &mut Vec<String>) {
    match fs::read_to_string(path) {
        Ok(actual) if actual == expected => {}
        Ok(_) => drifts.push(format!("modified: {}", path.display())),
        Err(err) => drifts.push(format!("missing/unreadable ({err}): {}", path.display())),
    }
}

/// Recompiles both variants in `target/spirv-builder` and verifies byte-for-byte parity
/// against checked-in `wgsl/` artifacts and `wgsl/manifest.json` without mutating disk files.
pub fn check_drift(paths: &ShaderPaths) -> anyhow::Result<()> {
    let suite = compile_and_validate_suite(paths)?;
    let mut drifts = Vec::new();
    for kernel in &suite.portable {
        compare_file_exact(
            &paths.wgsl_file_path(kernel.entry_point, false),
            &kernel.wgsl,
            &mut drifts,
        );
    }
    for kernel in &suite.subgroups {
        compare_file_exact(
            &paths.wgsl_file_path(kernel.entry_point, true),
            &kernel.wgsl,
            &mut drifts,
        );
    }
    compare_file_exact(&paths.manifest_path(), &suite.manifest_json, &mut drifts);
    if !drifts.is_empty() {
        bail!(
            "checked-in WGSL shader drift detected (run `sona-shaders-gen build`):\n  - {}",
            drifts.join("\n  - ")
        );
    }
    Ok(())
}
