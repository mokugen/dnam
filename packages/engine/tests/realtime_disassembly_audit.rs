//! Automated Release-Binary Disassembly Audit for Goal 6 (`REQ-6.2`, `REQ-6.3`).
//!
//! Disassembles `libdnam_engine.rlib`, `libdnam_vm-*.rlib`, and `libdnam_kernels-*.rlib`
//! release artifacts via `objdump -dr --demangle` and verifies:
//! 1. **Crate-Wide `libm` Ban (`REQ-6.2`):** Zero external relocations to `fmodf`, `fmod`,
//!    `sinf`, `cosf`, `tanf`, `expf`, `logf`, `powf`, `sqrtf`, or `f64` `libm` equivalents.
//! 2. **Real-Time Loop Instruction & Call Audit (`REQ-6.2`, `REQ-6.3`):** Across all Layer 1
//!    `DnamEngine`, `DynamicDnamEngine`, `HostShellPatchbay` (`ReturnAligner`, `ParallelAlignLine`),
//!    `PreviewGuardState`, `DnamHostHandle`, C-ABI `dnam_engine_*`, `SonaVm`, `ControlBus`,
//!    and `dnam_kernels` real-time entry points and their transitive call-graph closure:
//!    - Zero hardware division/sqrt (`fdiv`/`fsqrt`/`udiv`/`sdiv` or x86 equivalents).
//!    - Zero 128-bit software division (`__udivti3`/`__divti3`/`__umodti3`/`__modti3`),
//!      heap allocation (`__rust_alloc`/`__rust_dealloc`/`__rust_realloc`), or uncached OS
//!      feature syscalls (`sysctlbyname`, `getauxval`).

#![forbid(unsafe_code)]

#[path = "sona_test_support/objdump_parser.rs"]
mod objdump_parser;

use std::collections::{BTreeMap, BTreeSet};

use objdump_parser::{
    AliasMap, FnDisasm, clean_reloc_target, collect_symbol_aliases, disassemble_rlib_into,
    ensure_release_rlibs, merge_folded_symbol_aliases,
};

const RT_ROOT_EXACT_OR_SUBSTRS: &[&str] = &[
    "DnamEngine<",
    "::process_block",
    "process_stereo_vm_block",
    "DynamicDnamEngine>::",
    "DnamHostHandle>::process_block_chunked",
    "DnamHostHandle>::process_block_in_place_chunked",
    "DnamHostHandle>::process_mono_chunked",
    "DnamHostHandle>::render_offline_bounce_with_options",
    "ParallelAlignLine>::configure",
    "PreviewGuardState>::process_block_in_place",
    "PreviewGuardState>::reset",
    "ControlConsumer>::drain_into_tracking_bounce",
    "dnam_engine_process_block",
    "dnam_engine_process_mono",
    "dnam_engine_set_sample_rate",
    "dnam_engine_set_offline_bounce",
    "dnam_engine_set_maximize_tier_on_bounce",
    "dnam_engine_set_tier",
    "dnam_engine_set_max_supported_tier",
    "dnam_engine_set_control_value",
    "dnam_engine_set_control_immediate",
    "dnam_engine_get_latency_samples",
    "dnam_engine_get_telemetry",
    "dnam_engine_reset",
    "sona_vm_process_block_default",
    "sona_vm_step_frame_default",
    "sona_vm_step_control_tick_default",
    "step_frame_sona",
    "step_frame_4x_sona",
    "process_block_sona",
    "DynamicSonaVm>::",
    "ControlBus",
    "step_control_tick_3khz_default",
    "sona_graph_eval_core_quads_default",
    "sona_graph_convolver_chunk_default",
    "sona_graph_transport_top_default",
    "sona_graph_transport_end_default",
];

fn is_realtime_root_symbol(sym: &str) -> bool {
    if sym.contains("reload_model_in_place_owned")
        || sym.contains("drop_in_place")
        || sym.contains("drop_glue")
        || sym.contains("from_dnam_bin")
        || sym.contains("with_headroom")
        || sym.contains("open_with_backend")
    {
        return false;
    }
    let layer1_engine_method = (sym.contains("DnamEngine")
        || sym.contains("DynamicDnamEngine>::")
        || sym.contains("process_stereo_vm_block"))
        && (sym.contains("::process_block")
            || sym.contains("::process_mono")
            || sym.contains("::process_stereo")
            || sym.contains("process_stereo_vm_block")
            || sym.contains("::step_frame")
            || sym.contains("::set_tier")
            || sym.contains("::set_max_supported_tier")
            || sym.contains("::set_offline_bounce")
            || sym.contains("::set_maximize_tier_on_bounce")
            || sym.contains("::set_sample_rate_rt")
            || sym.contains("::reconfigure_for_sample_rate_rt")
            || sym.contains("::set_bypass")
            || sym.contains("::set_routing_mode")
            || sym.contains("::set_wdw_mode")
            || sym.contains("::set_wdw_enabled")
            || sym.contains("::configure_return_alignment")
            || sym.contains("::configure_parallel_alignment")
            || sym.contains("::set_quad_write_enable")
            || sym.contains("::set_op_t_write_enable")
            || sym.contains("::set_stereo_width")
            || sym.contains("::set_control_value")
            || sym.contains("::set_control_immediate")
            || sym.contains("::reset_to_dc_equilibrium")
            || sym.contains("::sync_tempo")
            || sym.contains("::register_tap_tempo")
            || sym.contains("::reload_model_in_place")
            || sym.contains("::telemetry_snapshot")
            || sym.contains("::render_offline_bounce"));
    let patchbay_or_guard = sym.contains("HostShellPatchbay>::route_")
        || sym.contains("HostShellPatchbay>::set_routing_mode")
        || sym.contains("HostShellPatchbay>::set_wdw_mode")
        || sym.contains("HostShellPatchbay>::set_wdw_enabled")
        || sym.contains("HostShellPatchbay>::reconfigure_sample_rate")
        || sym.contains("HostShellPatchbay>::configure_return_alignment")
        || sym.contains("HostShellPatchbay>::configure_parallel_alignment")
        || sym.contains("HostShellPatchbay>::reset")
        || sym.contains("ParallelAlignLine>::configure")
        || sym.contains("ParallelAlignLine>::step")
        || sym.contains("ReturnAligner>::step")
        || sym.contains("eval_wdw_routing_outputs")
        || sym.contains("PreviewGuardState>::process_")
        || sym.contains("PreviewGuardState>::reset")
        || sym.contains("ControlConsumer>::drain_into_tracking_bounce");
    let c_sym = sym.trim_start_matches('_');
    let host_or_ffi = sym.contains("DnamHostHandle>::process_")
        || sym.contains("DnamHostHandle>::render_offline_bounce")
        || sym.contains("DnamHostHandle>::set_bypass")
        || sym.contains("DnamHostHandle>::set_stereo_width")
        || sym.contains("DnamHostHandle>::set_offline_bounce")
        || sym.contains("DnamHostHandle>::set_tier")
        || sym.contains("DnamHostHandle>::set_max_supported_tier")
        || sym.contains("DnamHostHandle>::set_maximize_tier_on_bounce")
        || sym.contains("DnamHostHandle>::set_control_value")
        || sym.contains("DnamHostHandle>::set_control_immediate")
        || sym.contains("DnamHostHandle>::reset_to_dc_equilibrium")
        || sym.contains("DnamHostHandle>::sync_tempo")
        || matches!(
            c_sym,
            "dnam_engine_process_block"
                | "dnam_engine_process_mono"
                | "dnam_engine_set_sample_rate"
                | "dnam_engine_set_offline_bounce"
                | "dnam_engine_set_maximize_tier_on_bounce"
                | "dnam_engine_set_tier"
                | "dnam_engine_set_max_supported_tier"
                | "dnam_engine_set_control_value"
                | "dnam_engine_set_control_immediate"
                | "dnam_engine_get_latency_samples"
                | "dnam_engine_get_telemetry"
                | "dnam_engine_reset"
        );
    let vm_or_kernel = sym.contains("sona_vm_process_")
        || sym.contains("sona_vm_step_")
        || sym.contains("step_frame_sona")
        || sym.contains("step_frame_4x_sona")
        || sym.contains("process_block_sona")
        || sym.contains("sona_graph_")
        || (sym.contains("DynamicSonaVm>::")
            && (sym.contains("::process_")
                || sym.contains("::step_")
                || sym.contains("::set_")
                || sym.contains("::switch_tier")
                || sym.contains("::reload_from_view_in_place")
                || sym.contains("::reset")))
        || (sym.contains("ControlBus")
            && (sym.contains("::step_frame")
                || sym.contains("::tick_control_boundary")
                || sym.contains("::eval_")
                || sym.contains("::reset")))
        || sym.contains("step_control_tick_3khz");
    layer1_engine_method || patchbay_or_guard || host_or_ffi || vm_or_kernel
}

fn is_forbidden_rt_instruction(mnemonic: &str) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        matches!(mnemonic, "fdiv" | "fsqrt" | "udiv" | "sdiv")
    }
    #[cfg(target_arch = "x86_64")]
    {
        matches!(
            mnemonic,
            "divss"
                | "divps"
                | "divsd"
                | "divpd"
                | "vdivss"
                | "vdivps"
                | "vdivsd"
                | "vdivpd"
                | "sqrtss"
                | "sqrtps"
                | "sqrtsd"
                | "sqrtpd"
                | "vsqrtss"
                | "vsqrtps"
                | "vsqrtsd"
                | "vsqrtpd"
                | "idiv"
                | "idivl"
                | "idivq"
                | "div"
                | "divl"
                | "divq"
        )
    }
}

fn is_forbidden_crate_wide_reloc(target: &str) -> bool {
    matches!(
        clean_reloc_target(target),
        "fmodf"
            | "fmod"
            | "sinf"
            | "cosf"
            | "tanf"
            | "asinf"
            | "acosf"
            | "atanf"
            | "atan2f"
            | "sinhf"
            | "coshf"
            | "tanhf"
            | "expf"
            | "exp2f"
            | "logf"
            | "log2f"
            | "log10f"
            | "powf"
            | "sqrtf"
            | "cbrtf"
            | "hypotf"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "atan2"
            | "sinh"
            | "cosh"
            | "tanh"
            | "exp"
            | "exp2"
            | "log"
            | "log2"
            | "log10"
            | "pow"
            | "sqrt"
            | "cbrt"
            | "hypot"
    )
}

fn is_prewarmed_host_mutex_init(caller_demangled: &str, callee_name: &str) -> bool {
    let is_host_sync = caller_demangled.contains("dnam_engine::host::")
        || caller_demangled.contains("HostMutex")
        || caller_demangled.contains("DnamHostHandle")
        || caller_demangled.contains("EngineSwapAudioHandle")
        || caller_demangled.contains("RealtimeEngineSwap");
    let is_once_mutex = callee_name.contains("OnceBox")
        && (callee_name.contains("mutex") || callee_name.contains("Mutex"));
    is_host_sync && is_once_mutex
}

fn is_forbidden_rt_reloc(target: &str) -> bool {
    if is_forbidden_crate_wide_reloc(target) {
        return true;
    }
    let clean = clean_reloc_target(target);
    matches!(
        clean,
        "udivti3"
            | "divti3"
            | "umodti3"
            | "modti3"
            | "rust_alloc"
            | "rust_dealloc"
            | "rust_realloc"
            | "rust_alloc_zeroed"
            | "sysctlbyname"
            | "getauxval"
    ) || clean.ends_with("___rust_alloc")
        || clean.ends_with("___rust_dealloc")
        || clean.ends_with("___rust_realloc")
        || clean.ends_with("___rust_alloc_zeroed")
        || clean.contains("detect_and_initialize")
        || clean.contains("OnceBox")
}

/// verify[sdk.rust.realtime-disassembly-audit]
/// verify[arch.simd.zero-stall-lowering]
/// verify[dsp.op-m.rotor.angle-wrap-and-retune]
/// verify[dsp.op-t.guard.incremental-counters]
/// verify[dsp.op-t.zero-fdiv]
#[test]
fn test_release_disassembly_zero_pipeline_stalls_and_nondeterministic_calls() {
    let [engine_rlib, vm_rlib, kernels_rlib] = ensure_release_rlibs();

    let mut alias_groups: AliasMap = BTreeMap::new();
    let mut fns: BTreeMap<String, FnDisasm> = BTreeMap::new();
    let mut blocks = BTreeMap::new();
    for (tag, path) in [
        ("engine", &engine_rlib),
        ("vm", &vm_rlib),
        ("kernels", &kernels_rlib),
    ] {
        collect_symbol_aliases(tag, path, &mut alias_groups);
        disassemble_rlib_into(tag, path, &mut fns, &mut blocks);
    }
    merge_folded_symbol_aliases(&alias_groups, &blocks, &mut fns);

    let mut raw_to_scoped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for key in fns.keys() {
        if let Some((_, raw)) = key.split_once("::") {
            raw_to_scoped
                .entry(raw.to_string())
                .or_default()
                .push(key.clone());
        }
    }

    let mut rt_closure: BTreeSet<String> = fns
        .iter()
        .filter(|(_, info)| is_realtime_root_symbol(&info.demangled))
        .map(|(k, _)| k.clone())
        .collect();
    let root_count = rt_closure.len();
    let mut worklist: Vec<String> = rt_closure.iter().cloned().collect();
    while let Some(sym) = worklist.pop() {
        if let Some(info) = fns.get(&sym) {
            for callee in &info.callees {
                if let Some(targets) = raw_to_scoped.get(callee) {
                    for target_key in targets {
                        if let Some(callee_info) = fns.get(target_key) {
                            if is_prewarmed_host_mutex_init(&info.demangled, &callee_info.demangled)
                                || is_prewarmed_host_mutex_init(&info.demangled, target_key)
                            {
                                continue;
                            }
                            if rt_closure.insert(target_key.clone()) {
                                worklist.push(target_key.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    for fn_key in &rt_closure {
        if let Some(info) = fns.get(fn_key) {
            assert!(
                !info.instructions.is_empty(),
                "Audited RT closure symbol `{}` (`{fn_key}`) had 0 disassembled instructions",
                info.demangled
            );
        }
    }

    for pattern in RT_ROOT_EXACT_OR_SUBSTRS {
        let found = rt_closure.iter().any(|k| {
            fns.get(k)
                .is_some_and(|i| i.demangled.contains(pattern) && !i.instructions.is_empty())
        });
        assert!(
            found,
            "Expected non-empty RT root matching `{pattern}` across engine/vm/kernels rlibs"
        );
    }

    for required_sym in [
        "dnam_engine_process_block",
        "dnam_engine_process_mono",
        "dnam_engine_set_sample_rate",
        "dnam_engine_set_offline_bounce",
        "dnam_engine_set_maximize_tier_on_bounce",
        "dnam_engine_set_tier",
        "dnam_engine_set_max_supported_tier",
        "dnam_engine_set_control_value",
        "dnam_engine_set_control_immediate",
        "dnam_engine_get_latency_samples",
        "dnam_engine_get_telemetry",
        "dnam_engine_reset",
        "DnamHostHandle>::process_block_chunked",
        "DnamHostHandle>::process_block",
        "DnamHostHandle>::process_block_in_place",
        "DnamHostHandle>::process_stereo",
        "DnamHostHandle>::process_stereo_to_mono",
        "DnamHostHandle>::process_mono_to_stereo",
        "DnamHostHandle>::process_mono",
        "DnamHostHandle>::process_mono_in_place_chunked",
        "DnamHostHandle>::process_mono_in_place",
        "DnamHostHandle>::render_offline_bounce_with_options",
        "DnamHostHandle>::set_tier",
        "DnamHostHandle>::set_max_supported_tier",
        "DnamHostHandle>::set_offline_bounce",
        "DnamHostHandle>::set_maximize_tier_on_bounce",
        "DnamHostHandle>::set_control_value",
        "DnamHostHandle>::set_control_immediate",
        "DnamHostHandle>::set_bypass",
        "DnamHostHandle>::set_stereo_width",
        "DnamHostHandle>::reset_to_dc_equilibrium",
        "DnamHostHandle>::sync_tempo",
        "ControlConsumer>::drain_into_tracking_bounce",
        "DynamicDnamEngine>::process_block",
        "DynamicDnamEngine>::process_block_in_place",
        "DynamicDnamEngine>::process_stereo",
        "DynamicDnamEngine>::process_stereo_to_mono",
        "DynamicDnamEngine>::process_mono_to_stereo",
        "DynamicDnamEngine>::process_mono",
        "DynamicDnamEngine>::process_block_io",
        "DynamicDnamEngine>::process_block_4cm",
        "DynamicDnamEngine>::process_block_wdw",
        "DynamicDnamEngine>::process_block_sidechain",
        "DynamicDnamEngine>::step_frame_sona",
        "DynamicDnamEngine>::set_tier",
        "DynamicDnamEngine>::set_max_supported_tier",
        "DynamicDnamEngine>::set_offline_bounce",
        "DynamicDnamEngine>::set_maximize_tier_on_bounce",
        "DynamicDnamEngine>::set_sample_rate_rt",
        "DynamicDnamEngine>::set_control_value",
        "DynamicDnamEngine>::set_control_immediate",
        "DynamicDnamEngine>::reset_to_dc_equilibrium",
        "DynamicDnamEngine>::set_bypass",
        "DynamicDnamEngine>::set_stereo_width",
        "DynamicDnamEngine>::set_routing_mode",
        "DynamicDnamEngine>::set_wdw_mode",
        "DynamicDnamEngine>::set_wdw_enabled",
        "DynamicDnamEngine>::configure_return_alignment",
        "DynamicDnamEngine>::configure_parallel_alignment",
        "DynamicDnamEngine>::set_quad_write_enable",
        "DynamicDnamEngine>::set_op_t_write_enable",
        "DynamicDnamEngine>::sync_tempo",
        "DynamicDnamEngine>::register_tap_tempo",
        "DynamicDnamEngine>::reload_model_in_place",
        "DynamicDnamEngine>::telemetry_snapshot",
        "DynamicDnamEngine>::render_offline_bounce",
        "step_frame_sona",
        "step_frame_4port_sona",
        "process_stereo_vm_block",
        "set_bypass",
        "set_quad_write_enable",
        "set_op_t_write_enable",
        "set_stereo_width",
        "reload_model_in_place",
        "HostShellPatchbay>::set_routing_mode",
        "HostShellPatchbay>::reconfigure_sample_rate",
        "HostShellPatchbay>::configure_return_alignment",
        "HostShellPatchbay>::configure_parallel_alignment",
        "ParallelAlignLine>::configure",
        "PreviewGuardState>::process_block_in_place",
    ] {
        let found = rt_closure.iter().any(|k| {
            k.starts_with("engine::")
                && fns.get(k).is_some_and(|i| {
                    i.demangled.contains(required_sym) && !i.instructions.is_empty()
                })
        });
        assert!(
            found,
            "Expected non-empty Layer 1 (`engine::`) RT symbol `{required_sym}`"
        );
    }

    for required_vm_sym in [
        "sona_vm_process_block_default",
        "sona_vm_step_frame_default",
        "sona_vm_step_control_tick_default",
        "step_frame_sona",
        "step_frame_4x_sona",
        "process_block_sona",
        "sona_graph_eval_core_quads_default",
        "sona_graph_convolver_chunk_default",
        "sona_graph_transport_top_default",
        "sona_graph_transport_end_default",
        "sona_graph_sensory_step_default",
        "sona_graph_op_n_pre_step_default",
        "sona_graph_dispatch_transport_3khz_default",
        "step_control_tick_3khz_default",
    ] {
        let found = rt_closure.iter().any(|k| {
            k.starts_with("vm::")
                && fns.get(k).is_some_and(|i| {
                    i.demangled.contains(required_vm_sym) && !i.instructions.is_empty()
                })
        });
        assert!(
            found,
            "Expected non-empty Layer 0 (`vm::`) RT symbol `{required_vm_sym}`"
        );
    }

    let mut crate_reloc_violations = Vec::new();
    let mut rt_violations = Vec::new();
    for (fn_key, info) in &fns {
        let fn_name = &info.demangled;
        for (target, line) in &info.relocs {
            if is_forbidden_crate_wide_reloc(target) {
                crate_reloc_violations.push(format!("{fn_name}: {line}"));
            }
        }
        if rt_closure.contains(fn_key) {
            for (target, line) in &info.relocs {
                if is_forbidden_rt_reloc(target) && !is_prewarmed_host_mutex_init(fn_name, target) {
                    rt_violations.push(format!("{fn_name}: {line}"));
                }
            }
            for (mnemonic, line) in &info.instructions {
                if is_forbidden_rt_instruction(mnemonic) {
                    rt_violations.push(format!("{fn_name}: {line}"));
                }
            }
        }
    }

    assert!(
        root_count >= 20 && rt_closure.len() > root_count,
        "Expected >= 20 real-time roots and transitive callees, got roots={root_count}, closure={}",
        rt_closure.len()
    );
    assert!(
        crate_reloc_violations.is_empty(),
        "REQ-6.2 forbidden libm relocations:\n{}",
        crate_reloc_violations.join("\n")
    );
    assert!(
        rt_violations.is_empty(),
        "REQ-6.2/6.3 RT instruction or nondeterministic call violations:\n{}",
        rt_violations.join("\n")
    );
    eprintln!(
        "Disassembly audit verified: roots={root_count}, rt_closure={}, total_fns={}",
        rt_closure.len(),
        fns.len()
    );
}
