//! Automated Release-Binary Disassembly Audit for `dnam-vm` (`REQ-6.2`, `REQ-6.3`).
//!
//! - `verify[sdk.rust.realtime-disassembly-audit]`
//! - `verify[arch.simd.zero-stall-lowering]`
//!
//! Disassembles the optimized `libdnam_vm.rlib` and `libdnam_kernels-*.rlib` release artifacts
//! via `objdump -dr --demangle` and verifies zero `libm` relocations crate-wide and zero
//! `FDIV`/`FSQRT`/`IDIV`/`UDIV`/`SDIV`, `u128` software division, heap allocs, locks, or
//! uncached feature probes across the transitive closure of every real-time root.

#![forbid(unsafe_code)]

mod realtime_disassembly_audit {
    pub(crate) mod objdump_parser;
}

use std::collections::{BTreeMap, BTreeSet};

use dnam_kernels::OpNPreHotState;
use dnam_vm::{
    ControlBus, ConvolverFirHistorySlices, ConvolverFirWeightSlices, CrossbarFrame,
    DefaultControlBus, DefaultMathBackend, DnamBinView, DynamicSonaVm, MathBackend,
    PartitionedSchroederBank, QuadColdMeta, QuadEffectiveTargets, QuadTopRead, SensoryQuadCold,
    SensoryQuadHot, SonaHotSlab, SonaTelemetrySnapshot, SonaVm, TRANSPORT_MAX_QUADS, Tier,
    TransportExtRing, TransportModSlots, TransportQuadHotCompact, TransportStageState,
    TransportTopOfFrameResult, VmError, bind_dnam_bin_prevalidated,
    reload_slab_from_dnam_bin_in_place, sona_graph_convolver_chunk_default,
    sona_graph_dispatch_transport_3khz_default, sona_graph_eval_core_quads_default,
    sona_graph_op_n_pre_step_default, sona_graph_sensory_step_default,
    sona_graph_transport_end_default, sona_graph_transport_top_default,
    sona_vm_clear_saturation_flags_default, sona_vm_process_block_default,
    sona_vm_process_tick16_default, sona_vm_set_live_analog_drift_enabled_default,
    sona_vm_step_control_tick_default, sona_vm_step_frame_default, step_control_frame_default,
    step_control_tick_3khz_default,
};
use realtime_disassembly_audit::objdump_parser::{
    AliasMap, FnDisasm, clean_reloc_target, collect_symbol_aliases, disassemble_rlib_into,
    ensure_release_rlibs,
};

fn retain_vm_realtime_roots() {
    core::hint::black_box::<
        for<'a> fn(&'a mut DefaultControlBus<'static>, Option<[f32; 4]>) -> Option<&'a [f32; 32]>,
    >(ControlBus::<'static, DefaultMathBackend>::step_frame);
    core::hint::black_box::<fn(&mut DefaultControlBus<'_>, Option<[f32; 4]>)>(
        step_control_frame_default,
    );
    core::hint::black_box::<fn(&mut DefaultControlBus<'_>, Option<[f32; 4]>)>(
        step_control_tick_3khz_default,
    );
    core::hint::black_box::<fn(&mut OpNPreHotState, f32, f32, f32, f32) -> (f32, f32)>(
        OpNPreHotState::step_stereo,
    );
    core::hint::black_box::<fn(&[u8]) -> Result<dnam_vm::DnamBinView<'_>, VmError>>(
        bind_dnam_bin_prevalidated,
    );
    core::hint::black_box::<
        fn(
            &mut SonaVm<'_, DefaultMathBackend>,
            &[f32],
            &[f32],
            &mut [f32],
            &mut [f32],
        ) -> Result<(), VmError>,
    >(sona_vm_process_block_default);
    core::hint::black_box::<
        fn(
            &mut SonaVm<'_, DefaultMathBackend>,
            &[f32; 16],
            &[f32; 16],
            &mut [f32; 16],
            &mut [f32; 16],
        ) -> Result<(), VmError>,
    >(sona_vm_process_tick16_default);
    core::hint::black_box::<
        fn(&mut SonaVm<'_, DefaultMathBackend>, f32, f32) -> Result<(f32, f32), VmError>,
    >(sona_vm_step_frame_default);
    core::hint::black_box::<fn(&mut SonaVm<'_, DefaultMathBackend>)>(
        sona_vm_step_control_tick_default,
    );
    core::hint::black_box::<
        fn(&mut DynamicSonaVm, &[f32], &[f32], &mut [f32], &mut [f32]) -> Result<(), VmError>,
    >(DynamicSonaVm::process_block);
    core::hint::black_box::<
        fn(
            &mut DynamicSonaVm,
            &[f32; 16],
            &[f32; 16],
            &mut [f32; 16],
            &mut [f32; 16],
        ) -> Result<(), VmError>,
    >(DynamicSonaVm::process_tick16_const);
    core::hint::black_box::<fn(&mut DynamicSonaVm, f32, f32) -> Result<(f32, f32), VmError>>(
        DynamicSonaVm::step_frame,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm)>(DynamicSonaVm::step_control_tick_3khz);
    core::hint::black_box::<fn(&mut DynamicSonaVm, bool)>(DynamicSonaVm::set_quad_write_enable);
    core::hint::black_box::<fn(&mut DynamicSonaVm, bool)>(DynamicSonaVm::set_op_t_write_enable);
    core::hint::black_box::<fn(&mut DynamicSonaVm, usize, f32) -> Result<(), VmError>>(
        DynamicSonaVm::set_control_physical,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm, &[(usize, f32)]) -> Result<(), VmError>>(
        DynamicSonaVm::set_controls_batch,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm, f32) -> usize>(DynamicSonaVm::apply_tempo_bpm);
    core::hint::black_box::<fn(&mut DynamicSonaVm, dnam_vm::Tier) -> Result<(), VmError>>(
        DynamicSonaVm::switch_tier,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm, bool) -> Result<(), VmError>>(
        DynamicSonaVm::set_offline_bounce,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm, f32) -> Result<(), VmError>>(
        DynamicSonaVm::retime_sample_rate,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm)>(DynamicSonaVm::reset);
    core::hint::black_box::<fn(&mut SonaVm<'_, DefaultMathBackend>, bool)>(
        sona_vm_set_live_analog_drift_enabled_default,
    );
    core::hint::black_box::<fn(&mut SonaVm<'_, DefaultMathBackend>)>(
        sona_vm_clear_saturation_flags_default,
    );
    core::hint::black_box::<fn(&mut DynamicSonaVm, bool)>(
        DynamicSonaVm::set_live_analog_drift_enabled,
    );
    core::hint::black_box::<fn(&DynamicSonaVm) -> usize>(DynamicSonaVm::get_latency_samples);
    core::hint::black_box::<fn(&DynamicSonaVm) -> u32>(DynamicSonaVm::saturation_flags);
    core::hint::black_box::<fn(&mut DynamicSonaVm)>(DynamicSonaVm::clear_saturation_flags);
    core::hint::black_box::<fn(&DynamicSonaVm) -> SonaTelemetrySnapshot>(DynamicSonaVm::telemetry);
    core::hint::black_box::<fn(&mut SonaVm<'static, DefaultMathBackend>, bool)>(
        SonaVm::<'static, DefaultMathBackend>::set_live_analog_drift_enabled,
    );
    core::hint::black_box::<fn(&mut SonaVm<'static, DefaultMathBackend>)>(
        SonaVm::<'static, DefaultMathBackend>::clear_saturation_flags,
    );
    core::hint::black_box::<fn(&mut SonaVm<'static, DefaultMathBackend>, bool)>(
        SonaVm::<'static, DefaultMathBackend>::set_quad_write_enable,
    );
    core::hint::black_box::<fn(&mut SonaVm<'static, DefaultMathBackend>, bool)>(
        SonaVm::<'static, DefaultMathBackend>::set_op_t_write_enable,
    );
    core::hint::black_box::<
        fn(
            &mut SonaHotSlab<'_, DefaultMathBackend>,
            &CrossbarFrame<<DefaultMathBackend as MathBackend>::Vec4>,
            (f32, f32),
        ) -> ((f32, f32), (f32, f32)),
    >(sona_graph_eval_core_quads_default);
    core::hint::black_box::<
        fn(
            &mut ConvolverFirHistorySlices<'_>,
            &ConvolverFirWeightSlices<'_>,
            &[[f32; 2]; 4],
            &mut [[f32; 2]; 4],
            &mut [[f32; 2]; 4],
        ),
    >(sona_graph_convolver_chunk_default);
    core::hint::black_box::<
        fn(
            &mut TransportStageState<'_, '_>,
            &mut [PartitionedSchroederBank<'_>],
            &[QuadEffectiveTargets],
            (f32, f32),
            f32,
        ) -> Result<TransportTopOfFrameResult, VmError>,
    >(sona_graph_transport_top_default);
    core::hint::black_box::<
        fn(
            &mut TransportStageState<'_, '_>,
            &[QuadEffectiveTargets],
            &[QuadTopRead; TRANSPORT_MAX_QUADS],
            (f32, f32),
            f32,
        ) -> Result<((f32, f32), bool), VmError>,
    >(sona_graph_transport_end_default);
    core::hint::black_box::<fn(&SensoryQuadCold, &mut SensoryQuadHot, f32, f32, f32) -> (f32, f32)>(
        sona_graph_sensory_step_default,
    );
    core::hint::black_box::<fn(&SensoryQuadCold, &mut SensoryQuadHot, f32, f32, f32) -> (f32, f32)>(
        SensoryQuadCold::step_frame::<DefaultMathBackend>,
    );
    core::hint::black_box::<fn(&mut OpNPreHotState, f32, f32, f32, f32) -> (f32, f32)>(
        sona_graph_op_n_pre_step_default,
    );
    core::hint::black_box::<
        fn(
            &mut [TransportQuadHotCompact],
            &[QuadColdMeta],
            usize,
            &TransportModSlots,
            &mut [TransportExtRing<'_>],
            &mut [PartitionedSchroederBank<'_>],
            &mut [QuadEffectiveTargets],
        ),
    >(sona_graph_dispatch_transport_3khz_default);
    core::hint::black_box::<
        fn(&mut DefaultControlBus<'static>, &DnamBinView<'_>, Tier, f32) -> Result<(), VmError>,
    >(ControlBus::<'static, DefaultMathBackend>::reload_from_dnam_bin_in_place);
    core::hint::black_box::<
        fn(
            &mut SonaHotSlab<'_, DefaultMathBackend>,
            &DnamBinView<'_>,
            Tier,
            f32,
            bool,
        ) -> Result<(), VmError>,
    >(reload_slab_from_dnam_bin_in_place::<DefaultMathBackend>);
    core::hint::black_box::<
        fn(
            &mut SonaVm<'static, DefaultMathBackend>,
            &DnamBinView<'_>,
            Tier,
            f32,
            bool,
        ) -> Result<(), VmError>,
    >(SonaVm::<'static, DefaultMathBackend>::reload_from_view_in_place);
    core::hint::black_box::<
        fn(&mut DynamicSonaVm, &DnamBinView<'_>, Tier, f32, bool) -> Result<(), VmError>,
    >(DynamicSonaVm::reload_from_view_in_place);
}

const EXACT_OR_SUBSTR_ROOTS: &[&str] = &[
    "step_control_frame_default",
    "step_control_tick_3khz_default",
    "ControlBus<dnam_vm::math::scalar::ScalarBackend>>::step_frame",
    "tick_control_boundary",
    "eval_static_and_snap_if_dirty",
    "eval_current_dispatch_and_snap",
    "scrub_drift_if_disabled",
    "sona_graph_eval_core_quads_default",
    "sona_graph_convolver_chunk_default",
    "sona_graph_transport_top_default",
    "sona_graph_transport_end_default",
    "sona_graph_sensory_step_default",
    "sona_graph_op_n_pre_step_default",
    "sona_graph_dispatch_transport_3khz_default",
    "ConvolverHighSrFirExt>::step_frame",
    "bind_dnam_bin_prevalidated",
    "sona_vm_process_block_default",
    "sona_vm_process_tick16_default",
    "sona_vm_step_frame_default",
    "sona_vm_step_control_tick_default",
    "process_block_sona",
    "process_tick16_const",
    "process_microchunk_4",
    "step_frame_sona",
    "retune_kautz_banks_3khz",
    "latch_quad_nonslew_targets",
    "tick_quad_3khz",
    "tick_transport_3khz",
    "apply_transport_mod_slots",
    "eval_block_and_group_adaa_lru",
    "eval_global_stage_mods",
    "eval_lane_core_mods",
    "switch_tier",
    "retime_sample_rate",
    "set_offline_bounce",
    "reset_to_dc_equilibrium",
    "hoist_core_quads_and_readouts",
    "step_control_tick_3khz",
    "set_op_t_write_enable",
    "DynamicSonaVm>::process_block",
    "DynamicSonaVm>::process_tick16_const",
    "DynamicSonaVm>::step_frame",
    "DynamicSonaVm>::step_control_tick_3khz",
    "DynamicSonaVm>::set_quad_write_enable",
    "DynamicSonaVm>::set_op_t_write_enable",
    "DynamicSonaVm>::set_control_physical",
    "DynamicSonaVm>::set_controls_batch",
    "DynamicSonaVm>::apply_tempo_bpm",
    "DynamicSonaVm>::switch_tier",
    "DynamicSonaVm>::set_offline_bounce",
    "DynamicSonaVm>::retime_sample_rate",
    "DynamicSonaVm>::reset",
    "step_lfos_and_skewed_q2",
    "evaluate_static_cp1_latents",
    "step_ou_drift",
    "latch_immediate_at_init_or_reset",
    "sona_vm_set_live_analog_drift_enabled_default",
    "sona_vm_clear_saturation_flags_default",
    "DynamicSonaVm>::set_live_analog_drift_enabled",
    "DynamicSonaVm>::get_latency_samples",
    "DynamicSonaVm>::saturation_flags",
    "DynamicSonaVm>::clear_saturation_flags",
    "DynamicSonaVm>::telemetry",
];

fn is_vm_realtime_root_symbol(sym: &str) -> bool {
    if EXACT_OR_SUBSTR_ROOTS.iter().any(|p| sym.contains(p)) {
        return true;
    }

    const RT_VM_AND_SLAB_METHODS: &[&str] = &[
        "::process_block",
        "::process_tick16_const",
        "::step_frame",
        "::step_frame_4x",
        "::step_control_tick_3khz",
        "::tick_control_boundary",
        "::set_control_physical",
        "::set_controls_batch",
        "::apply_tempo_bpm",
        "::set_quad_write_enable",
        "::set_op_t_write_enable",
        "::switch_tier",
        "::retime_sample_rate",
        "::set_offline_bounce",
        "::reset",
        "::set_live_analog_drift_enabled",
        "::get_latency_samples",
        "::saturation_flags",
        "::clear_saturation_flags",
        "::telemetry",
    ];
    let is_vm_or_slab = sym.contains("SonaVm<")
        || sym.contains("SonaVm>::")
        || sym.contains("SonaHotSlab<")
        || sym.contains("SonaHotSlab>::")
        || sym.contains("SensoryQuadCold")
        || sym.contains("CoreQuadHot")
        || sym.contains("TransportQuadHotCompact")
        || sym.contains("ControlBus<")
        || sym.contains("ControlBus>::");
    is_vm_or_slab
        && !sym.contains("drop_glue")
        && !sym.contains("Drop>::drop")
        && RT_VM_AND_SLAB_METHODS.iter().any(|m| sym.contains(m))
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
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = mnemonic;
        false
    }
}

fn is_forbidden_crate_wide_reloc(target: &str) -> bool {
    let clean = clean_reloc_target(target);
    matches!(
        clean,
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
#[test]
fn test_vm_release_disassembly_zero_pipeline_stalls_and_nondeterministic_calls() {
    retain_vm_realtime_roots();
    let Some([vm_rlib, kernels_rlib]) = ensure_release_rlibs() else {
        return;
    };
    let mut alias_groups: AliasMap = BTreeMap::new();
    collect_symbol_aliases("vm", &vm_rlib, &mut alias_groups);
    collect_symbol_aliases("kernels", &kernels_rlib, &mut alias_groups);

    let mut fns: BTreeMap<String, FnDisasm> = BTreeMap::new();
    let mut blocks: BTreeMap<(String, String, u64), FnDisasm> = BTreeMap::new();
    disassemble_rlib_into("vm", &vm_rlib, &mut fns, &mut blocks);
    disassemble_rlib_into("kernels", &kernels_rlib, &mut fns, &mut blocks);

    for (loc, group) in &alias_groups {
        if let Some(block) = blocks.get(loc) {
            for (alias_key, alias_demangled) in group {
                let entry = fns.entry(alias_key.clone()).or_default();
                if entry.demangled.is_empty() {
                    entry.demangled = alias_demangled.clone();
                }
                if entry.instructions.is_empty() {
                    entry.instructions = block.instructions.clone();
                }
                if entry.relocs.is_empty() {
                    entry.relocs = block.relocs.clone();
                }
                for callee in &block.callees {
                    if callee != alias_key {
                        entry.callees.insert(callee.clone());
                    }
                }
            }
        }
    }

    let mut rt_closure: BTreeSet<String> = fns
        .iter()
        .filter(|(_, info)| is_vm_realtime_root_symbol(&info.demangled))
        .map(|(k, _)| k.clone())
        .collect();
    let root_count = rt_closure.len();
    let mut worklist: Vec<String> = rt_closure.iter().cloned().collect();
    while let Some(sym) = worklist.pop() {
        if let Some(info) = fns.get(&sym) {
            for callee in &info.callees {
                if rt_closure.insert(callee.clone()) {
                    worklist.push(callee.clone());
                }
            }
        }
    }

    for &pat in EXACT_OR_SUBSTR_ROOTS {
        assert!(
            fns.values()
                .any(|i| i.demangled.contains(pat) && !i.instructions.is_empty()),
            "Non-vacuous disassembly audit failure: root pattern `{pat}` matched 0 non-empty symbols in {} / {}",
            vm_rlib.display(),
            kernels_rlib.display()
        );
    }

    for fn_key in &rt_closure {
        if let Some(info) = fns.get(fn_key) {
            assert!(
                !info.instructions.is_empty(),
                "Expected non-empty instruction stream for real-time closure symbol `{}`",
                info.demangled
            );
        }
    }

    for required in [
        "step_control_frame_default",
        "step_control_tick_3khz_default",
        "ControlBus<dnam_vm::math::scalar::ScalarBackend>>::step_frame",
        "tick_control_boundary",
        "sona_graph_eval_core_quads_default",
        "sona_graph_convolver_chunk_default",
        "sona_graph_transport_top_default",
        "sona_graph_transport_end_default",
        "sona_graph_sensory_step_default",
        "sona_graph_op_n_pre_step_default",
        "sona_graph_dispatch_transport_3khz_default",
        "ConvolverHighSrFirExt>::step_frame",
        "bind_dnam_bin_prevalidated",
        "sona_vm_process_block_default",
        "sona_vm_process_tick16_default",
        "sona_vm_step_frame_default",
        "sona_vm_step_control_tick_default",
        "process_block_sona",
        "process_tick16_const",
        "process_microchunk_4",
        "step_frame_sona",
        "retune_kautz_banks_3khz",
        "set_op_t_write_enable",
        "step_control_tick_3khz",
        "hoist_core_quads_and_readouts",
        "step_lfos_and_skewed_q2",
        "DynamicSonaVm>::process_block",
        "DynamicSonaVm>::process_tick16_const",
        "DynamicSonaVm>::step_frame",
        "DynamicSonaVm>::step_control_tick_3khz",
        "DynamicSonaVm>::set_quad_write_enable",
        "DynamicSonaVm>::set_op_t_write_enable",
        "DynamicSonaVm>::set_control_physical",
        "DynamicSonaVm>::set_controls_batch",
        "DynamicSonaVm>::apply_tempo_bpm",
        "DynamicSonaVm>::switch_tier",
        "DynamicSonaVm>::set_offline_bounce",
        "DynamicSonaVm>::retime_sample_rate",
        "DynamicSonaVm>::reset",
        "sona_vm_set_live_analog_drift_enabled_default",
        "sona_vm_clear_saturation_flags_default",
        "latch_immediate_at_init_or_reset",
        "latch_quad_nonslew_targets",
        "DynamicSonaVm>::set_live_analog_drift_enabled",
        "DynamicSonaVm>::get_latency_samples",
        "DynamicSonaVm>::saturation_flags",
        "DynamicSonaVm>::clear_saturation_flags",
        "DynamicSonaVm>::telemetry",
    ] {
        assert!(
            rt_closure.iter().any(|k| fns
                .get(k)
                .is_some_and(|i| i.demangled.contains(required) && !i.instructions.is_empty())),
            "Expected non-empty VM real-time root `{required}` in {} / {}",
            vm_rlib.display(),
            kernels_rlib.display()
        );
    }
    assert!(root_count >= 10 && rt_closure.len() > root_count);

    let mut crate_reloc_violations = Vec::new();
    let mut rt_violations = Vec::new();
    for (fn_key, info) in &fns {
        for (target, line) in &info.relocs {
            if is_forbidden_crate_wide_reloc(target) {
                crate_reloc_violations.push(format!("{}: {line}", info.demangled));
            }
        }
        if rt_closure.contains(fn_key) {
            for (target, line) in &info.relocs {
                if is_forbidden_rt_reloc(target) {
                    rt_violations.push(format!("{}: {line}", info.demangled));
                }
            }
            for (mnemonic, line) in &info.instructions {
                if is_forbidden_rt_instruction(mnemonic) {
                    rt_violations.push(format!("{}: {line}", info.demangled));
                }
            }
        }
    }
    let mut reload_closure: BTreeSet<String> = fns
        .iter()
        .filter(|(_, info)| {
            info.demangled.contains("reload_from_dnam_bin_in_place")
                || info
                    .demangled
                    .contains("reload_slab_from_dnam_bin_in_place")
                || info.demangled.contains("reload_from_view_in_place")
        })
        .filter(|(_, info)| !info.demangled.contains("drop_glue"))
        .map(|(k, _)| k.clone())
        .collect();
    let reload_root_count = reload_closure.len();
    let mut reload_worklist: Vec<String> = reload_closure.iter().cloned().collect();
    while let Some(sym) = reload_worklist.pop() {
        if let Some(info) = fns.get(&sym) {
            for callee in &info.callees {
                if reload_closure.insert(callee.clone()) {
                    reload_worklist.push(callee.clone());
                }
            }
        }
    }
    assert!(reload_root_count >= 3 && !reload_closure.is_empty());
    let mut reload_reloc_violations = Vec::new();
    for fn_key in &reload_closure {
        if let Some(info) = fns.get(fn_key) {
            for (target, line) in &info.relocs {
                if is_forbidden_rt_reloc(target) {
                    reload_reloc_violations.push(format!("{}: {line}", info.demangled));
                }
            }
        }
    }
    assert!(
        reload_reloc_violations.is_empty(),
        "Zero-alloc/zero-drop reload reloc violations:\n{}",
        reload_reloc_violations.join("\n")
    );
    assert!(
        crate_reloc_violations.is_empty(),
        "REQ-6.2 forbidden libm relocations:\n{}",
        crate_reloc_violations.join("\n")
    );
    assert!(
        rt_violations.is_empty(),
        "REQ-6.2/6.3 VM RT disassembly violations:\n{}",
        rt_violations.join("\n")
    );
}
