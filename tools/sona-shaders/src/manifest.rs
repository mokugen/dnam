#![forbid(unsafe_code)]

//! Canonical 7-kernel registry, expected binding ABI specs, browser WebGPU preparation, and
//! JSON manifest types (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{UNIFORM_STRUCT_BYTES, WORKGROUP_SIZE};
use serde::{Deserialize, Serialize};

/// Current manifest schema version emitted by `sona-shaders-gen`.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Target triple passed to `spirv-builder`.
pub const SHADER_TARGET_TRIPLE: &str = "spirv-unknown-naga-wgsl";

/// Canonical identifiers of all seven 32-bit SONA v3.0 WebGPU compute kernels.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelId {
    PrescreenPopulation,
    FusedCoreScan256,
    AdjointScan256,
    TileMixFwdAdj,
    GaborBezoutGrad,
    OzakiGramRhsI32,
    InnerOptStep,
}

impl KernelId {
    /// Returns the WGSL entry point and file stem for this kernel.
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

    /// Looks up the canonical [`ExpectedKernelSpec`] for this kernel.
    #[must_use]
    pub const fn expected_spec(self) -> &'static ExpectedKernelSpec {
        match self {
            Self::PrescreenPopulation => &EXPECTED_KERNELS[0],
            Self::FusedCoreScan256 => &EXPECTED_KERNELS[1],
            Self::AdjointScan256 => &EXPECTED_KERNELS[2],
            Self::TileMixFwdAdj => &EXPECTED_KERNELS[3],
            Self::GaborBezoutGrad => &EXPECTED_KERNELS[4],
            Self::OzakiGramRhsI32 => &EXPECTED_KERNELS[5],
            Self::InnerOptStep => &EXPECTED_KERNELS[6],
        }
    }
}

/// Binding access class expected in WGSL `@group(0) @binding(N)`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingAccess {
    Uniform,
    StorageRead,
    StorageReadWrite,
}

/// Expected binding declaration for a compute kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BindingSpec {
    pub binding: u32,
    pub name: &'static str,
    pub access: BindingAccess,
}

/// Static ABI contract verified by [`crate::validate::validate_kernel_wgsl`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ExpectedKernelSpec {
    pub id: KernelId,
    pub entry_point: &'static str,
    pub workgroup_size: [u32; 3],
    pub uniform_bytes: u32,
    pub bindings: &'static [BindingSpec],
}

const PRESCREEN_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "input_audio",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 2,
        name: "target_audio",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "seed_params",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 4,
        name: "seed_losses",
        access: BindingAccess::StorageReadWrite,
    },
];

const FUSED_CORE_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "core_inputs",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 2,
        name: "quad_params",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "tile_checkpoints",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 4,
        name: "feature_outputs",
        access: BindingAccess::StorageReadWrite,
    },
];

const ADJOINT_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "core_inputs",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 2,
        name: "quad_params",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "tile_checkpoints",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 4,
        name: "d_loss_d_features",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 5,
        name: "d_loss_d_inputs",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 6,
        name: "param_grads",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 7,
        name: "tile_adj_boundaries",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 8,
        name: "feature_outputs",
        access: BindingAccess::StorageRead,
    },
];

const TILE_MIX_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "block_inputs_or_d_loss_d_inputs",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 2,
        name: "tile_descriptors",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "feature_outputs",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 4,
        name: "tile_weight_grads",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 5,
        name: "d_loss_d_features",
        access: BindingAccess::StorageReadWrite,
    },
];

const GABOR_BEZOUT_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "pred_audio",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 2,
        name: "target_audio",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "bezout_taps",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 4,
        name: "d_loss_d_pred",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 5,
        name: "band_losses",
        access: BindingAccess::StorageReadWrite,
    },
];

const OZAKI_GRAM_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "features",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 2,
        name: "targets",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "col_scales",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 4,
        name: "ozaki_buckets",
        access: BindingAccess::StorageReadWrite,
    },
];

const INNER_OPT_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        binding: 0,
        name: "uniforms",
        access: BindingAccess::Uniform,
    },
    BindingSpec {
        binding: 1,
        name: "theta_candidates",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 2,
        name: "grad_candidates",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 3,
        name: "param_bounds_and_ste_mask",
        access: BindingAccess::StorageRead,
    },
    BindingSpec {
        binding: 4,
        name: "adamw_m_v_or_lbfgs",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 5,
        name: "search_dir",
        access: BindingAccess::StorageReadWrite,
    },
    BindingSpec {
        binding: 6,
        name: "diag_precond",
        access: BindingAccess::StorageRead,
    },
];

/// Canonical specification table for all seven SONA v3.0 WebGPU compute kernels.
pub const EXPECTED_KERNELS: [ExpectedKernelSpec; 7] = [
    ExpectedKernelSpec {
        id: KernelId::PrescreenPopulation,
        entry_point: KernelId::PrescreenPopulation.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: PRESCREEN_BINDINGS,
    },
    ExpectedKernelSpec {
        id: KernelId::FusedCoreScan256,
        entry_point: KernelId::FusedCoreScan256.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: FUSED_CORE_BINDINGS,
    },
    ExpectedKernelSpec {
        id: KernelId::AdjointScan256,
        entry_point: KernelId::AdjointScan256.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: ADJOINT_BINDINGS,
    },
    ExpectedKernelSpec {
        id: KernelId::TileMixFwdAdj,
        entry_point: KernelId::TileMixFwdAdj.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: TILE_MIX_BINDINGS,
    },
    ExpectedKernelSpec {
        id: KernelId::GaborBezoutGrad,
        entry_point: KernelId::GaborBezoutGrad.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: GABOR_BEZOUT_BINDINGS,
    },
    ExpectedKernelSpec {
        id: KernelId::OzakiGramRhsI32,
        entry_point: KernelId::OzakiGramRhsI32.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: OZAKI_GRAM_BINDINGS,
    },
    ExpectedKernelSpec {
        id: KernelId::InnerOptStep,
        entry_point: KernelId::InnerOptStep.entry_point(),
        workgroup_size: [WORKGROUP_SIZE, 1, 1],
        uniform_bytes: UNIFORM_STRUCT_BYTES as u32,
        bindings: INNER_OPT_BINDINGS,
    },
];

const _: () = assert!(EXPECTED_KERNELS.len() == 7);

/// Binding record persisted into `wgsl/manifest.json`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BindingRecord {
    pub group: u32,
    pub binding: u32,
    pub access: BindingAccess,
}

/// Artifact metadata persisted for one `.wgsl` file inside `wgsl/manifest.json`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShaderArtifactRecord {
    pub name: String,
    pub wgsl_path: String,
    pub sha256: String,
    pub workgroup_sram_bytes: u32,
    pub bindings: Vec<BindingRecord>,
}

/// Top-level `wgsl/manifest.json` describing portable and subgroup shader builds.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShaderManifest {
    pub version: u32,
    pub target: String,
    pub portable: Vec<ShaderArtifactRecord>,
    pub subgroups: Vec<ShaderArtifactRecord>,
}

/// Prepends `enable subgroups;` when submitting a subgroup-enabled WGSL shader to browser
/// `navigator.gpu` (`Tint`), keeping checked-in artifacts directly parseable by `naga`.
#[must_use]
pub fn prepare_for_browser_webgpu(wgsl: &str, uses_subgroups: bool) -> String {
    if uses_subgroups && !wgsl.contains("enable subgroups;") {
        format!("enable subgroups;\n{wgsl}")
    } else {
        wgsl.to_owned()
    }
}
