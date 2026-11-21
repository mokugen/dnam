#![forbid(unsafe_code)]

//! Naga WGSL parser, validator, SRAM budget analyzer, 64-bit type guard, and ABI verifier
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::MAX_WORKGROUP_SRAM_BYTES;
use crate::manifest::{BindingAccess, BindingRecord, ExpectedKernelSpec};
use anyhow::{Context as _, bail, ensure};
use naga::proc::Layouter;
use naga::valid::{Capabilities, GlobalUse, ValidationFlags, Validator};
use naga::{
    AddressSpace, Block, Module, ResourceBinding, ShaderStage, Statement, StorageAccess, TypeInner,
};
use std::collections::BTreeMap;

/// Verified properties extracted from one validated WGSL module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedShader {
    pub workgroup_sram_bytes: u32,
    pub bindings: Vec<BindingRecord>,
}

fn check_no_64bit_types(module: &Module) -> anyhow::Result<()> {
    for (handle, ty) in module.types.iter() {
        let is_64bit = match ty.inner {
            TypeInner::Scalar(scalar)
            | TypeInner::Atomic(scalar)
            | TypeInner::Vector { scalar, .. }
            | TypeInner::Matrix { scalar, .. }
            | TypeInner::CooperativeMatrix { scalar, .. }
            | TypeInner::ValuePointer { scalar, .. } => scalar.width == 8,
            _ => false,
        };
        if is_64bit {
            let label = ty.name.as_deref().unwrap_or("<anonymous>");
            bail!(
                "forbidden 64-bit type {handle:?} ('{label}') in WGSL module: {:?}",
                ty.inner
            );
        }
    }
    Ok(())
}

fn check_block_no_loop_continuing(block: &Block, entry_point: &str) -> anyhow::Result<()> {
    for stmt in block.iter() {
        match stmt {
            Statement::Block(inner) => check_block_no_loop_continuing(inner, entry_point)?,
            Statement::If { accept, reject, .. } => {
                check_block_no_loop_continuing(accept, entry_point)?;
                check_block_no_loop_continuing(reject, entry_point)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    check_block_no_loop_continuing(&case.body, entry_point)?;
                }
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                ensure!(
                    continuing.is_empty() && break_if.is_none(),
                    "'{entry_point}' contains unlowered WGSL loop `continuing`/`break if` \
                     (forbidden by Metal MSL workaround)"
                );
                check_block_no_loop_continuing(body, entry_point)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_no_loop_continuing(module: &Module, entry_point: &str) -> anyhow::Result<()> {
    for (_, func) in module.functions.iter() {
        check_block_no_loop_continuing(&func.body, entry_point)?;
    }
    for ep in &module.entry_points {
        check_block_no_loop_continuing(&ep.function.body, entry_point)?;
    }
    Ok(())
}

fn compute_workgroup_sram_bytes(module: &Module, layouter: &Layouter) -> anyhow::Result<u32> {
    let mut total_bytes: u32 = 0;
    for (_, gv) in module.global_variables.iter() {
        if gv.space == AddressSpace::WorkGroup {
            let layout = layouter[gv.ty];
            let aligned = layout.alignment.round_up(total_bytes);
            total_bytes = aligned
                .checked_add(layout.size)
                .context("workgroup SRAM byte calculation overflowed u32")?;
        }
    }
    ensure!(
        total_bytes <= MAX_WORKGROUP_SRAM_BYTES as u32,
        "workgroup SRAM {total_bytes} B exceeds strict ceiling {MAX_WORKGROUP_SRAM_BYTES} B"
    );
    Ok(total_bytes)
}

fn map_storage_access(access: StorageAccess) -> anyhow::Result<BindingAccess> {
    let rw = StorageAccess::LOAD | StorageAccess::STORE;
    if access == StorageAccess::LOAD {
        Ok(BindingAccess::StorageRead)
    } else if access == rw {
        Ok(BindingAccess::StorageReadWrite)
    } else {
        bail!("unsupported storage buffer access flags: {access:?}");
    }
}

/// Parses, validates, and checks ABI/SRAM invariants for a single compute kernel WGSL string.
pub fn validate_kernel_wgsl(
    spec: &ExpectedKernelSpec,
    wgsl: &str,
    uses_subgroups: bool,
) -> anyhow::Result<ValidatedShader> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|err| {
        anyhow::anyhow!(
            "WGSL parse failure for '{}':\n{}",
            spec.entry_point,
            err.emit_to_string(wgsl)
        )
    })?;

    check_no_64bit_types(&module)?;
    check_no_loop_continuing(&module, spec.entry_point)?;

    let caps = if uses_subgroups {
        Capabilities::SUBGROUP | Capabilities::SUBGROUP_BARRIER
    } else {
        Capabilities::empty()
    };
    let mut validator = Validator::new(ValidationFlags::all(), caps);
    let info = validator.validate(&module).map_err(|err| {
        anyhow::anyhow!(
            "Naga validation failure for '{}':\n{}",
            spec.entry_point,
            err.emit_to_string(wgsl)
        )
    })?;

    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|err| anyhow::anyhow!("Layouter failed for '{}': {err}", spec.entry_point))?;

    let workgroup_sram_bytes = compute_workgroup_sram_bytes(&module, &layouter)?;

    ensure!(
        module.entry_points.len() == 1,
        "expected exactly 1 entry point in '{}', found {}",
        spec.entry_point,
        module.entry_points.len()
    );
    let ep = &module.entry_points[0];
    ensure!(
        ep.name == spec.entry_point,
        "entry point mismatch: expected '{}', found '{}'",
        spec.entry_point,
        ep.name
    );
    ensure!(
        ep.stage == ShaderStage::Compute,
        "entry point '{}' must be @compute, found {:?}",
        spec.entry_point,
        ep.stage
    );
    ensure!(
        ep.workgroup_size == spec.workgroup_size,
        "entry point '{}' workgroup_size {:?} != expected {:?}",
        spec.entry_point,
        ep.workgroup_size,
        spec.workgroup_size
    );

    let ep_info = info.get_entry_point(0);
    let mut actual_bindings: BTreeMap<u32, (BindingAccess, u32, GlobalUse)> = BTreeMap::new();
    for (handle, gv) in module.global_variables.iter() {
        let Some(ResourceBinding { group, binding }) = gv.binding else {
            continue;
        };
        ensure!(
            group == 0,
            "'{}' binding ({group}, {binding}) must use @group(0)",
            spec.entry_point
        );
        let access = match gv.space {
            AddressSpace::Uniform => BindingAccess::Uniform,
            AddressSpace::Storage { access } => map_storage_access(access)?,
            other => bail!(
                "'{}' binding ({group}, {binding}) uses unexpected address space {other:?}",
                spec.entry_point
            ),
        };
        let byte_size = layouter[gv.ty].size;
        let global_use = ep_info[handle];
        ensure!(
            actual_bindings
                .insert(binding, (access, byte_size, global_use))
                .is_none(),
            "duplicate @group(0) @binding({binding}) in '{}'",
            spec.entry_point
        );
    }

    ensure!(
        actual_bindings.len() == spec.bindings.len(),
        "'{}' binding count {} != expected {}",
        spec.entry_point,
        actual_bindings.len(),
        spec.bindings.len()
    );

    let mut binding_records = Vec::with_capacity(spec.bindings.len());
    for expected in spec.bindings {
        let &(actual_access, byte_size, global_use) =
            actual_bindings.get(&expected.binding).with_context(|| {
                format!(
                    "'{}' missing @group(0) @binding({}) ({})",
                    spec.entry_point, expected.binding, expected.name
                )
            })?;
        ensure!(
            actual_access == expected.access,
            "'{}' @binding({}) ({}) access {:?} != expected {:?}",
            spec.entry_point,
            expected.binding,
            expected.name,
            actual_access,
            expected.access
        );
        ensure!(
            !global_use.is_empty(),
            "'{}' @binding({}) ({}) is unused by entry point",
            spec.entry_point,
            expected.binding,
            expected.name
        );
        if expected.access == BindingAccess::Uniform {
            ensure!(
                byte_size == spec.uniform_bytes,
                "'{}' uniform @binding({}) size {byte_size} B != expected {} B",
                spec.entry_point,
                expected.binding,
                spec.uniform_bytes
            );
        }
        binding_records.push(BindingRecord {
            group: 0,
            binding: expected.binding,
            access: actual_access,
        });
    }

    Ok(ValidatedShader {
        workgroup_sram_bytes,
        bindings: binding_records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::KernelId;

    #[test]
    fn rejects_64bit_types_and_sram_overflow() {
        let spec = KernelId::PrescreenPopulation.expected_spec();
        let bad_64bit = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u64>;
@compute @workgroup_size(256, 1, 1)
fn prescreen_population() {
    buf[0] = 1lu;
}
"#;
        let err = validate_kernel_wgsl(spec, bad_64bit, false).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("64-bit") || msg.contains("WGSL parse") || msg.contains("Naga validation"),
            "unexpected error: {msg}"
        );

        let big_sram = r#"
struct Uniforms { a: u32, b: u32, c: u32, d: u32, e: f32, f: f32, g: u32, h: u32 }
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage> input_audio: array<vec2<f32>>;
@group(0) @binding(2) var<storage> target_audio: array<vec2<f32>>;
@group(0) @binding(3) var<storage> seed_params: array<f32>;
@group(0) @binding(4) var<storage, read_write> seed_losses: array<f32>;
var<workgroup> huge: array<f32, 4096>;
@compute @workgroup_size(256, 1, 1)
fn prescreen_population(@builtin(local_invocation_index) lid: u32) {
    huge[lid] = input_audio[0].x + target_audio[0].x + seed_params[0] + f32(uniforms.a);
    workgroupBarrier();
    seed_losses[lid] = huge[lid];
}
"#;
        let overflow_err = validate_kernel_wgsl(spec, big_sram, false).unwrap_err();
        assert!(
            overflow_err.to_string().contains("exceeds strict ceiling"),
            "expected SRAM ceiling error, got: {overflow_err}"
        );
    }

    #[test]
    fn check_no_64bit_types_rejects_f64_and_u64_ir_directly() {
        let mut f64_mod = Module::default();
        f64_mod.types.insert(
            naga::Type {
                name: Some("bad_f64".to_owned()),
                inner: TypeInner::Scalar(naga::Scalar::F64),
            },
            naga::Span::default(),
        );
        let err_f64 = check_no_64bit_types(&f64_mod).unwrap_err().to_string();
        assert!(
            err_f64.contains("forbidden 64-bit type"),
            "unexpected f64 error: {err_f64}"
        );

        let mut u64_mod = Module::default();
        u64_mod.types.insert(
            naga::Type {
                name: Some("bad_u64".to_owned()),
                inner: TypeInner::Scalar(naga::Scalar::U64),
            },
            naga::Span::default(),
        );
        let err_u64 = check_no_64bit_types(&u64_mod).unwrap_err().to_string();
        assert!(
            err_u64.contains("forbidden 64-bit type"),
            "unexpected u64 error: {err_u64}"
        );
    }
}
