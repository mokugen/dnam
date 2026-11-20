//! Automated Release-Binary Disassembly Audit for `dnam-vm` (`REQ-6.2`, `REQ-6.3`).
//!
//! - `verify[sdk.rust.realtime-disassembly-audit]`
//! - `verify[arch.simd.zero-stall-lowering]`
//!
//! Disassembles the optimized `libdnam_vm.rlib` release artifact via `objdump -dr --demangle`
//! and verifies:
//! 1. **Crate-Wide `libm` Ban (`REQ-6.2`):** Zero external relocations to `_fmodf`, `_fmod`,
//!    `sinf`, `cosf`, `tanf`, `expf`, `logf`, `powf`, `sqrtf`, or `f64` `libm` equivalents.
//! 2. **Real-Time Loop Instruction & Call Audit (`REQ-6.2`, `REQ-6.3`):** Across all compiled
//!    `step_control_frame_default`, `step_control_tick_3khz_default`, `bind_dnam_bin_prevalidated`,
//!    `ControlBus::*`, `evaluate_static_cp1_latents`, `step_lfos_and_harmonics`, `evaluate_w_ctrl_32`,
//!    and `ExactIntFactorAccumulator` transitive callees:
//!    - Zero hardware floating-point or integer division/sqrt instructions (`fdiv`, `fsqrt`,
//!      `udiv`, `sdiv` on `aarch64`; `div*`, `vdiv*`, `sqrt*`, `vsqrt*`, `idiv*` on `x86_64`).
//!    - Zero calls to 128-bit software division (`__udivti3`, `__divti3`, `__umodti3`, `__modti3`),
//!      heap allocation (`__rust_alloc*`, `__rust_dealloc`, `__rust_realloc`), or uncached OS
//!      feature syscalls (`sysctlbyname`, `getauxval`, `detect_and_initialize`).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use dnam_vm::{
    DefaultControlBus, bind_dnam_bin_prevalidated, step_control_frame_default,
    step_control_tick_3khz_default,
};

fn retain_vm_realtime_roots() {
    core::hint::black_box::<fn(&mut DefaultControlBus<'_>, Option<[f32; 4]>, f32)>(
        step_control_frame_default,
    );
    core::hint::black_box::<fn(&mut DefaultControlBus<'_>, Option<[f32; 4]>, f32)>(
        step_control_tick_3khz_default,
    );
    core::hint::black_box::<fn(&[u8]) -> Result<dnam_vm::DnamBinView<'_>, dnam_vm::VmError>>(
        bind_dnam_bin_prevalidated,
    );
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn current_target_triple() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        "x86_64-unknown-linux-gnu"
    }
}

fn ensure_release_rlib() -> PathBuf {
    let root = workspace_root();
    let triple = current_target_triple();
    let audit_target_dir = root.join("target").join("release-audit-vm");
    let status = Command::new("cargo")
        .current_dir(&root)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("LLVM_PROFILE_FILE")
        .env("CARGO_PROFILE_RELEASE_LTO", "false")
        .env("CARGO_PROFILE_RELEASE_STRIP", "none")
        .args([
            "build",
            "--package",
            "dnam-vm",
            "--release",
            "--target",
            triple,
            "--target-dir",
        ])
        .arg(&audit_target_dir)
        .status()
        .expect("failed to invoke cargo build --release for VM disassembly audit");
    assert!(status.success(), "cargo build --release failed");

    let built = audit_target_dir
        .join(triple)
        .join("release")
        .join("libdnam_vm.rlib");
    assert!(
        built.is_file(),
        "expected release rlib at {}",
        built.display()
    );
    built
}

fn clean_reloc_target(raw: &str) -> &str {
    let s = raw.trim_start_matches('_');
    let s = s.split('@').next().unwrap_or(s);
    let s = s.split('+').next().unwrap_or(s);
    s.split('-').next().unwrap_or(s)
}

fn is_vm_realtime_root_symbol(sym: &str) -> bool {
    const EXACT_OR_SUBSTR_ROOTS: &[&str] = &[
        "step_control_frame_default",
        "step_control_tick_3khz_default",
        "bind_dnam_bin_prevalidated",
        "step_lfos_and_harmonics",
        "update_env_states",
        "evaluate_static_cp1_latents",
        "evaluate_w_ctrl_32",
        "step_ou_drift",
        "prepare_micro_ramp_deltas",
        "advance_micro_ramp_frame",
        "q32_phase_to_phasor",
        "eval_svf_g_modulated",
        "eval_group_adaa_and_lru_effective",
        "apply_ultra_core_mods_lane",
        "update_op_t_we_hold_schmitt",
    ];
    EXACT_OR_SUBSTR_ROOTS.iter().any(|p| sym.contains(p))
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

#[derive(Default)]
struct FnDisasm {
    demangled: String,
    instructions: Vec<(String, String)>,
    relocs: Vec<(String, String)>,
    callees: BTreeSet<String>,
}

fn is_local_temp_label(sym: &str) -> bool {
    sym.starts_with("ltmp") || sym.starts_with(".L")
}

fn parse_objdump_object_header(line: &str) -> Option<String> {
    let ff = line.find("file format ")?;
    let colon = line[..ff].rfind(':')?;
    Some(line[..colon].trim().to_string())
}

fn parse_objdump_sym_table_line(line: &str) -> Option<(u64, &str, &str)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let addr_tok = trimmed.split_whitespace().next()?;
    if addr_tok.len() < 8 || !addr_tok.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let addr = u64::from_str_radix(addr_tok, 16).ok()?;
    let mut cursor = trimmed[addr_tok.len()..].trim_start();
    let mut is_func = false;
    let section = loop {
        let tok = cursor.split_whitespace().next()?;
        if tok.starts_with("__") || tok.starts_with('.') || tok.starts_with('*') {
            break tok;
        }
        if tok.contains('F') {
            is_func = true;
        }
        cursor = cursor[tok.len()..].trim_start();
    };
    if !is_func || (!section.starts_with("__TEXT,__text") && !section.starts_with(".text")) {
        return None;
    }
    let mut sym_tail = cursor[section.len()..].trim_start();
    if let Some(first_tok) = sym_tail.split_whitespace().next()
        && first_tok.len() >= 4
        && first_tok.chars().all(|c| c.is_ascii_hexdigit())
        && !sym_tail[first_tok.len()..].trim_start().is_empty()
    {
        sym_tail = sym_tail[first_tok.len()..].trim_start();
    }
    if sym_tail.is_empty() {
        return None;
    }
    Some((addr, section, sym_tail))
}

fn collect_symbol_aliases(rlib: &Path) -> BTreeMap<(String, String, u64), Vec<(String, String)>> {
    let sym_raw_out = Command::new("objdump")
        .arg("-t")
        .arg(rlib)
        .output()
        .expect("objdump -t");
    let sym_dem_out = Command::new("objdump")
        .args(["-t", "--demangle"])
        .arg(rlib)
        .output()
        .expect("objdump -t --demangle");
    assert!(sym_raw_out.status.success() && sym_dem_out.status.success());

    let mut aliases: BTreeMap<(String, String, u64), Vec<(String, String)>> = BTreeMap::new();
    let mut current_obj = String::new();
    for (raw_line, dem_line) in String::from_utf8_lossy(&sym_raw_out.stdout)
        .lines()
        .zip(String::from_utf8_lossy(&sym_dem_out.stdout).lines())
    {
        if let Some(obj) = parse_objdump_object_header(raw_line) {
            current_obj = obj;
            continue;
        }
        if current_obj.is_empty() {
            continue;
        }
        let Some((addr, sec, raw_sym)) = parse_objdump_sym_table_line(raw_line) else {
            continue;
        };
        let clean_key = clean_reloc_target(raw_sym);
        if is_local_temp_label(raw_sym) || clean_key.is_empty() || is_local_temp_label(clean_key) {
            continue;
        }
        let dem_sym = parse_objdump_sym_table_line(dem_line)
            .map(|(_, _, s)| s)
            .unwrap_or(raw_sym);
        let group = aliases
            .entry((current_obj.clone(), sec.to_string(), addr))
            .or_default();
        if !group.iter().any(|(k, _)| k == clean_key) {
            group.push((clean_key.to_string(), dem_sym.to_string()));
        }
    }
    aliases
}

/// verify[sdk.rust.realtime-disassembly-audit]
#[test]
fn test_vm_release_disassembly_zero_pipeline_stalls_and_nondeterministic_calls() {
    retain_vm_realtime_roots();
    let rlib = ensure_release_rlib();
    let alias_groups = collect_symbol_aliases(&rlib);
    let out_raw = Command::new("objdump")
        .args(["-dr"])
        .arg(&rlib)
        .output()
        .expect("objdump -dr");
    let out_dem = Command::new("objdump")
        .args(["-dr", "--demangle"])
        .arg(&rlib)
        .output()
        .expect("objdump -dr --demangle");
    assert!(out_raw.status.success() && out_dem.status.success());

    let demangled_headers: Vec<String> = String::from_utf8_lossy(&out_dem.stdout)
        .lines()
        .filter_map(|raw_line| {
            let line = raw_line.trim();
            if line.ends_with(">:")
                && let Some(open) = line.find('<')
            {
                Some(line[open + 1..line.len() - 2].to_string())
            } else {
                None
            }
        })
        .collect();

    let mut fns: BTreeMap<String, FnDisasm> = BTreeMap::new();
    let mut blocks: BTreeMap<(String, String, u64), FnDisasm> = BTreeMap::new();
    let (mut current_obj, mut current_section, mut current_fn) =
        (String::new(), String::new(), String::new());
    let mut current_block: Option<(String, String, u64)> = None;
    let mut header_idx = 0usize;

    for raw_line in String::from_utf8_lossy(&out_raw.stdout).lines() {
        let line = raw_line.trim();
        if let Some(obj) = parse_objdump_object_header(line) {
            current_obj = obj;
            current_section.clear();
            current_block = None;
            current_fn.clear();
            continue;
        }
        if let Some(sec_rest) = line.strip_prefix("Disassembly of section ") {
            current_section = sec_rest.trim_end_matches(':').trim().to_string();
            current_block = None;
            current_fn.clear();
            continue;
        }
        if line.ends_with(">:")
            && let Some(open) = line.find('<')
        {
            let addr_hex = line[..open].trim();
            let raw_sym = &line[open + 1..line.len() - 2];
            let clean_key = clean_reloc_target(raw_sym).to_string();
            let demangled = demangled_headers
                .get(header_idx)
                .cloned()
                .unwrap_or_else(|| clean_key.clone());
            header_idx += 1;

            current_block = (!current_obj.is_empty() && !current_section.is_empty())
                .then(|| u64::from_str_radix(addr_hex, 16).ok())
                .flatten()
                .map(|addr| (current_obj.clone(), current_section.clone(), addr));
            if let Some(loc) = &current_block {
                blocks.entry(loc.clone()).or_default().demangled = demangled.clone();
            }
            if !is_local_temp_label(raw_sym) && !is_local_temp_label(&clean_key) {
                current_fn = clean_key;
                fns.entry(current_fn.clone()).or_default().demangled = demangled;
            } else {
                current_fn.clear();
            }
            continue;
        }
        if current_fn.is_empty() && current_block.is_none() {
            continue;
        }
        if line.contains("RELOC") || line.contains("R_AARCH64") || line.contains("R_X86_64") {
            if let Some(target) = line.split_whitespace().last() {
                let callee_key = clean_reloc_target(target).to_string();
                if let Some(loc) = &current_block {
                    let blk = blocks.entry(loc.clone()).or_default();
                    blk.relocs.push((target.to_string(), line.to_string()));
                    if !is_local_temp_label(&callee_key) {
                        blk.callees.insert(callee_key.clone());
                    }
                }
                if !current_fn.is_empty() {
                    let entry = fns.entry(current_fn.clone()).or_default();
                    entry.relocs.push((target.to_string(), line.to_string()));
                    if callee_key != current_fn && !is_local_temp_label(&callee_key) {
                        entry.callees.insert(callee_key);
                    }
                }
            }
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let insn_col_idx = usize::from(parts[0].trim_end().ends_with(':')) + 1;
        if let Some(insn_col) = parts.get(insn_col_idx).map(|s| s.trim())
            && let Some(mnemonic) = insn_col.split_whitespace().next()
        {
            let full_insn = parts[insn_col_idx..].join(" ");
            let callee_base = if matches!(mnemonic, "bl" | "b" | "call" | "callq" | "jmp")
                && let (Some(open), Some(close)) = (full_insn.find('<'), full_insn.rfind('>'))
                && open + 1 < close
            {
                let base = clean_reloc_target(&full_insn[open + 1..close]);
                (!is_local_temp_label(base)).then(|| base.to_string())
            } else {
                None
            };
            if let Some(loc) = &current_block {
                let blk = blocks.entry(loc.clone()).or_default();
                blk.instructions
                    .push((mnemonic.to_string(), line.to_string()));
                if let Some(base) = &callee_base {
                    blk.callees.insert(base.clone());
                }
            }
            if !current_fn.is_empty() {
                let entry = fns.entry(current_fn.clone()).or_default();
                entry
                    .instructions
                    .push((mnemonic.to_string(), line.to_string()));
                if let Some(base) = callee_base
                    && base != current_fn
                {
                    entry.callees.insert(base);
                }
            }
        }
    }

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

    for required in [
        "step_control_frame_default",
        "step_control_tick_3khz_default",
        "bind_dnam_bin_prevalidated",
    ] {
        assert!(
            rt_closure
                .iter()
                .any(|k| fns.get(k).is_some_and(|i| i.demangled.contains(required))),
            "Expected VM real-time root `{required}` in {}",
            rlib.display()
        );
    }
    assert!(root_count >= 3 && rt_closure.len() > root_count);

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
