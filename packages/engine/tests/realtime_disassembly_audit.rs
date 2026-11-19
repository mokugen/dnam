//! Automated Release-Binary Disassembly Audit for Goal 6 (`REQ-6.2`, `REQ-6.3`).
//!
//! - `verify[repo.code.realtime-disassembly-audit]`
//! - `verify[arch.simd.zero-stall-lowering]`
//! - `verify[dsp.op-m.rotor.angle-wrap-and-retune]`
//! - `verify[dsp.op-t.guard.incremental-counters]`
//!
//! Disassembles the optimized `libdnam_engine.rlib` release artifact via `objdump -dr --demangle`
//! and verifies:
//! 1. **Crate-Wide `libm` Ban (`REQ-6.2`):** Zero external relocations to `_fmodf`, `_fmod`,
//!    `sinf`, `cosf`, `tanf`, `expf`, `logf`, `powf`, `sqrtf`, or `f64` `libm` equivalents.
//! 2. **Real-Time Loop Instruction & Call Audit (`REQ-6.2`, `REQ-6.3`):** Across all compiled
//!    `DnamEngine<*>::process_block*`, `process_mono*`, `sona_process_block*`, `step_sona_group_stereo8*`,
//!    `V3Stage0Hot::*`, `V3Stage3Hot::*`, `V3Stage4TankHot::*`, `SonaHotSlab::*`, `SonaInstance::*`,
//!    `step_control_tick_3khz*`, `DnamHostHandle::process_*`, and C-ABI `_dnam_engine_process_*` functions:
//!    - Zero hardware floating-point or integer division/sqrt instructions (`fdiv`, `fsqrt`,
//!      `udiv`, `sdiv` on `aarch64`; `divss`/`divps`/`divsd`/`divpd`/`vdiv*`,
//!      `sqrtss`/`sqrtps`/`sqrtsd`/`sqrtpd`/`vsqrt*`, `idiv*`, `divl`/`divq` on `x86_64`).
//!    - Zero calls to 128-bit software division (`__udivti3`, `__divti3`, `__umodti3`, `__modti3`),
//!      heap allocation (`__rust_alloc`, `__rust_dealloc`, `__rust_realloc`), or uncached OS
//!      feature syscalls (`sysctlbyname`, `getauxval`).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use dnam_engine::ffi::{
    DnamEngineHandle, dnam_engine_process_mono, dnam_engine_reset,
    dnam_engine_set_control_immediate, dnam_engine_set_control_value,
    dnam_engine_set_offline_bounce, dnam_engine_set_tier,
};
use dnam_engine::host::DnamHostHandle;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamAudioIo, DnamEngine, DynamicDnamEngine, Tier};

/// Instantiates generic `DnamEngine` and host/FFI real-time entry points and retains them
/// via [`core::hint::black_box`] so the linker/compiler cannot strip their monomorphizations.
fn retain_realtime_entry_points() {
    type Eng = DnamEngine<'static, DefaultMathBackend>;
    type Dyn = DynamicDnamEngine<'static>;
    type Handle = DnamHostHandle;
    type C4cmDyn = fn(
        &mut Dyn,
        &[f32],
        &[f32],
        &[f32],
        &[f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
    );
    type C4cmEng = fn(
        &mut Eng,
        &[f32],
        &[f32],
        &[f32],
        &[f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
    );
    type WdwDyn = fn(&mut Dyn, &[f32], &[f32], &mut [f32], &mut [f32], &mut [f32]);
    type WdwEng = fn(&mut Eng, &[f32], &[f32], &mut [f32], &mut [f32], &mut [f32]);
    type ScDyn = fn(&mut Dyn, &[f32], &[f32], &[f32], &[f32], &mut [f32], &mut [f32]);
    type ScEng = fn(&mut Eng, &[f32], &[f32], &[f32], &[f32], &mut [f32], &mut [f32]);

    core::hint::black_box::<fn(&mut Eng, bool)>(Eng::set_offline_bounce);
    core::hint::black_box::<fn(&Handle, &[f32], &[f32], &mut [f32], &mut [f32], Option<bool>)>(
        Handle::render_offline_bounce_with_options,
    );
    core::hint::black_box::<fn(&Handle, &[f32], &[f32], &mut [f32], &mut [f32])>(
        Handle::process_block,
    );
    core::hint::black_box::<fn(&Handle, &mut [f32], &mut [f32])>(Handle::process_block_in_place);
    core::hint::black_box::<fn(&Handle, &[f32], &mut [f32])>(Handle::process_mono);
    core::hint::black_box::<fn(&mut Dyn, DnamAudioIo<'_>)>(Dyn::process_block_io);
    core::hint::black_box::<C4cmDyn>(Dyn::process_block_4cm);
    core::hint::black_box::<WdwDyn>(Dyn::process_block_wdw);
    core::hint::black_box::<ScDyn>(Dyn::process_block_sidechain);
    core::hint::black_box::<fn(&mut Dyn, &'static [u8], Tier, u32) -> Result<(), DnamError>>(
        Dyn::reload_model_in_place,
    );
    core::hint::black_box::<fn(&mut Eng, DnamAudioIo<'_>)>(Eng::process_block_io);
    core::hint::black_box::<C4cmEng>(Eng::process_block_4cm);
    core::hint::black_box::<WdwEng>(Eng::process_block_wdw);
    core::hint::black_box::<ScEng>(Eng::process_block_sidechain);
    core::hint::black_box::<fn(&mut Eng, bool, bool)>(Eng::set_bypass);
    core::hint::black_box::<fn(&mut Eng, bool)>(Eng::set_freeze);
    core::hint::black_box::<fn(&mut Eng, f32)>(Eng::set_stereo_width);
    core::hint::black_box::<fn(&mut Eng, &'static [u8], Tier, u32) -> Result<(), DnamError>>(
        Eng::reload_model_in_place,
    );
    core::hint::black_box::<unsafe extern "C" fn(*mut DnamEngineHandle, bool)>(
        dnam_engine_set_offline_bounce,
    );
    core::hint::black_box::<unsafe extern "C" fn(*mut DnamEngineHandle, u8)>(dnam_engine_set_tier);
    core::hint::black_box::<unsafe extern "C" fn(*mut DnamEngineHandle, usize, f32)>(
        dnam_engine_set_control_value,
    );
    core::hint::black_box::<unsafe extern "C" fn(*mut DnamEngineHandle, usize, f32)>(
        dnam_engine_set_control_immediate,
    );
    core::hint::black_box::<unsafe extern "C" fn(*mut DnamEngineHandle, *const f32, *mut f32, usize)>(
        dnam_engine_process_mono,
    );
    core::hint::black_box::<unsafe extern "C" fn(*mut DnamEngineHandle)>(dnam_engine_reset);
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

fn rlib_matches_current_arch(rlib_path: &Path) -> bool {
    if !rlib_path.is_file() {
        return false;
    }
    let Ok(out) = Command::new("objdump").arg("-t").arg(rlib_path).output() else {
        return false;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    if !text.contains("SonaHotSlab")
        && !text.contains("SonaInstance")
        && !text.contains("process_block")
    {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        text.contains("arm64") || text.contains("aarch64")
    }
    #[cfg(target_arch = "x86_64")]
    {
        text.contains("x86-64") || text.contains("x86_64")
    }
}

fn ensure_release_rlib() -> PathBuf {
    let root = workspace_root();
    let triple = current_target_triple();
    let audit_target_dir = root.join("target").join("release-audit");
    let status = Command::new("cargo")
        .current_dir(&root)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("LLVM_PROFILE_FILE")
        .args([
            "build",
            "--package",
            "dnam-engine",
            "--release",
            "--target",
            triple,
            "--target-dir",
        ])
        .arg(&audit_target_dir)
        .status()
        .expect("failed to invoke cargo build --release for disassembly audit");
    assert!(status.success(), "cargo build --release failed");

    let built = audit_target_dir
        .join(triple)
        .join("release")
        .join("libdnam_engine.rlib");
    assert!(
        rlib_matches_current_arch(&built),
        "expected release rlib at {}",
        built.display()
    );
    built
}

fn is_realtime_root_symbol(sym: &str) -> bool {
    const RT_OWNER_PATTERNS: &[&str] = &[
        "DnamEngine",
        "DynamicDnamEngine",
        "DnamHostHandle",
        "ControlBus",
        "Stage0Router",
        "V3Stage0Hot",
        "V3Stage3Hot",
        "V3Stage4TankHot",
        "SonaHotSlab",
        "SonaInstance",
        "step_sona_group_stereo8",
        "sona_process_block_default",
        "sona_process_block_infallible_default",
        "dnam_engine_process_block",
        "dnam_engine_process_mono",
        "dnam_engine_set_offline_bounce",
        "dnam_engine_set_tier",
        "dnam_engine_set_control_value",
        "dnam_engine_set_control_immediate",
        "dnam_engine_reset",
    ];
    const RT_METHOD_PATTERNS: &[&str] = &[
        "::process_block",
        "::process_block_in_place",
        "::process_block_stereo",
        "::process_block_infallible",
        "::process_block_io",
        "::process_block_4cm",
        "::process_block_wdw",
        "::process_block_sidechain",
        "::step_4port_frame",
        "::step_stereo",
        "::step_subblock",
        "::step_subblock_fast",
        "::run_3khz_tick",
        "::step_blocks",
        "::reconfigure",
        "step_sona_group_stereo8",
        "sona_process_block_default",
        "sona_process_block_infallible_default",
        "::process_mono",
        "::process_mono_to_stereo",
        "::process_stereo",
        "::process_stereo_to_mono",
        "::step_control_tick_3khz",
        "::tick_3khz",
        "::tick_3khz_with_detectors",
        "::kautz_formant",
        "::set_offline_bounce",
        "::render_offline_bounce_with_options",
        "::set_bypass",
        "::set_freeze",
        "::set_stereo_width",
        "::reload_model_in_place",
        "dnam_engine_process_block",
        "dnam_engine_process_mono",
        "dnam_engine_set_offline_bounce",
        "dnam_engine_set_tier",
        "dnam_engine_set_control_value",
        "dnam_engine_set_control_immediate",
        "dnam_engine_reset",
    ];
    RT_OWNER_PATTERNS.iter().any(|p| sym.contains(p))
        && RT_METHOD_PATTERNS.iter().any(|m| sym.contains(m))
        && !sym.contains("reload_model_in_place_owned")
}

fn clean_reloc_target(raw: &str) -> &str {
    let s = raw.trim_start_matches('_');
    let s = s.split('@').next().unwrap_or(s);
    let s = s.split('+').next().unwrap_or(s);
    s.split('-').next().unwrap_or(s)
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
    for vis in [".hidden", ".protected", ".internal"] {
        if let Some(rest) = sym_tail.strip_prefix(vis)
            && rest.chars().next().is_some_and(char::is_whitespace)
        {
            sym_tail = rest.trim_start();
            break;
        }
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
        .expect("failed to execute objdump -t");
    assert!(
        sym_raw_out.status.success(),
        "objdump -t failed on {}",
        rlib.display()
    );
    let sym_dem_out = Command::new("objdump")
        .args(["-t", "--demangle"])
        .arg(rlib)
        .output()
        .expect("failed to execute objdump -t --demangle");
    assert!(
        sym_dem_out.status.success(),
        "objdump -t --demangle failed on {}",
        rlib.display()
    );

    let sym_raw_text = String::from_utf8_lossy(&sym_raw_out.stdout);
    let sym_dem_text = String::from_utf8_lossy(&sym_dem_out.stdout);
    let mut aliases: BTreeMap<(String, String, u64), Vec<(String, String)>> = BTreeMap::new();
    let mut current_obj = String::new();

    for (raw_line, dem_line) in sym_raw_text.lines().zip(sym_dem_text.lines()) {
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
        if is_local_temp_label(raw_sym) {
            continue;
        }
        let clean_key = clean_reloc_target(raw_sym);
        if clean_key.is_empty() || is_local_temp_label(clean_key) {
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

/// verify[dsp.op-t.zero-fdiv]
#[test]
fn test_release_disassembly_zero_pipeline_stalls_and_nondeterministic_calls() {
    retain_realtime_entry_points();
    let rlib = ensure_release_rlib();
    let alias_groups = collect_symbol_aliases(&rlib);
    let out_raw = Command::new("objdump")
        .args(["-dr"])
        .arg(&rlib)
        .output()
        .expect("failed to execute objdump -dr");
    assert!(
        out_raw.status.success(),
        "objdump -dr failed on {}",
        rlib.display()
    );
    let text_raw = String::from_utf8_lossy(&out_raw.stdout);

    let out_demangled = Command::new("objdump")
        .args(["-dr", "--demangle"])
        .arg(&rlib)
        .output()
        .expect("failed to execute objdump -dr --demangle");
    assert!(
        out_demangled.status.success(),
        "objdump -dr --demangle failed on {}",
        rlib.display()
    );
    let text_demangled = String::from_utf8_lossy(&out_demangled.stdout);

    let demangled_headers: Vec<String> = text_demangled
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
    let mut current_obj = String::new();
    let mut current_section = String::new();
    let mut current_block: Option<(String, String, u64)> = None;
    let mut current_fn = String::new();
    let mut header_idx = 0usize;

    for raw_line in text_raw.lines() {
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

            current_block = if !current_obj.is_empty() && !current_section.is_empty() {
                u64::from_str_radix(addr_hex, 16)
                    .ok()
                    .map(|addr| (current_obj.clone(), current_section.clone(), addr))
            } else {
                None
            };
            if let Some(loc) = &current_block {
                blocks.entry(loc.clone()).or_default().demangled = demangled.clone();
            }

            if !is_local_temp_label(raw_sym) && !is_local_temp_label(&clean_key) {
                current_fn = clean_key;
                let entry = fns.entry(current_fn.clone()).or_default();
                entry.demangled = demangled;
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
        if parts.len() > insn_col_idx {
            let insn_col = parts[insn_col_idx].trim();
            let mnemonic = insn_col.split_whitespace().next().unwrap_or("");
            if !mnemonic.is_empty() {
                let full_insn = parts[insn_col_idx..].join(" ");
                let callee_base = if matches!(mnemonic, "bl" | "b" | "call" | "callq" | "jmp")
                    && let (Some(open), Some(close)) = (full_insn.find('<'), full_insn.rfind('>'))
                    && open + 1 < close
                {
                    let callee = &full_insn[open + 1..close];
                    let base = clean_reloc_target(callee);
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
    }

    // Propagate disassembled instructions, relocations, and callees across folded co-addressed
    // Mach-O / ELF aliases so symbol folding or `<ltmp0>` headers never hide call-graph edges.
    for (loc, group) in &alias_groups {
        let mut merged_instructions = Vec::new();
        let mut merged_relocs = Vec::new();
        let mut merged_callees = BTreeSet::new();
        if let Some(block) = blocks.get(loc) {
            merged_instructions.extend(block.instructions.iter().cloned());
            merged_relocs.extend(block.relocs.iter().cloned());
            merged_callees.extend(block.callees.iter().cloned());
        } else {
            for (alias_key, _) in group {
                if let Some(existing) = fns.get(alias_key) {
                    merged_instructions.extend(existing.instructions.iter().cloned());
                    merged_relocs.extend(existing.relocs.iter().cloned());
                    merged_callees.extend(existing.callees.iter().cloned());
                }
            }
        }
        if merged_instructions.is_empty() && merged_relocs.is_empty() && merged_callees.is_empty() {
            continue;
        }
        for (alias_key, alias_demangled) in group {
            let entry = fns.entry(alias_key.clone()).or_default();
            if entry.demangled.is_empty() {
                entry.demangled = alias_demangled.clone();
            }
            if entry.instructions.is_empty() {
                entry.instructions = merged_instructions.clone();
            }
            if entry.relocs.is_empty() {
                entry.relocs = merged_relocs.clone();
            }
            for callee in &merged_callees {
                if callee != alias_key {
                    entry.callees.insert(callee.clone());
                }
            }
        }
    }

    // Compute transitive call-graph closure from all real-time roots
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
                if let Some(callee_info) = fns.get(callee) {
                    if is_prewarmed_host_mutex_init(&info.demangled, &callee_info.demangled)
                        || is_prewarmed_host_mutex_init(&info.demangled, callee)
                    {
                        continue;
                    }
                    if rt_closure.insert(callee.clone()) {
                        worklist.push(callee.clone());
                    }
                }
            }
        }
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

    let has_sona_root = rt_closure.iter().any(|k| {
        fns.get(k).is_some_and(|info| {
            info.demangled.contains("sona_process_block")
                || info.demangled.contains("SonaHotSlab")
                || info.demangled.contains("SonaInstance")
                || info.demangled.contains("step_sona_group_stereo8")
        })
    });
    for required_sym in [
        "dnam_engine_process_block",
        "dnam_engine_process_mono",
        "dnam_engine_set_offline_bounce",
        "dnam_engine_set_tier",
        "dnam_engine_set_control_value",
        "dnam_engine_set_control_immediate",
        "dnam_engine_reset",
        "set_offline_bounce",
        "render_offline_bounce_with_options",
        "set_bypass",
        "set_freeze",
        "set_stereo_width",
    ] {
        let found = rt_closure.iter().any(|k| {
            fns.get(k)
                .is_some_and(|info| info.demangled.contains(required_sym))
        });
        assert!(
            found,
            "Expected real-time root `{required_sym}` to be audited in {}",
            rlib.display()
        );
    }
    for (owner, method) in [
        ("DynamicDnamEngine", "::process_block_io"),
        ("DynamicDnamEngine", "::process_block_4cm"),
        ("DynamicDnamEngine", "::process_block_wdw"),
        ("DynamicDnamEngine", "::process_block_sidechain"),
        ("DynamicDnamEngine", "::reload_model_in_place"),
        ("DnamHostHandle", "::process_block"),
        ("DnamHostHandle", "::process_block_in_place"),
        ("DnamHostHandle", "::process_mono"),
    ] {
        let found = rt_closure.iter().any(|k| {
            fns.get(k).is_some_and(|info| {
                let d = &info.demangled;
                if !d.contains(owner) {
                    return false;
                }
                match (owner, method) {
                    ("DynamicDnamEngine", "::reload_model_in_place") => {
                        d.contains("::reload_model_in_place")
                            && !d.contains("reload_model_in_place_owned")
                    }
                    ("DnamHostHandle", "::process_block") => {
                        d.ends_with("::process_block") || d.contains("::process_block_chunked")
                    }
                    ("DnamHostHandle", "::process_block_in_place") => {
                        d.ends_with("::process_block_in_place")
                            || d.contains("::process_block_in_place_chunked")
                    }
                    ("DnamHostHandle", "::process_mono") => {
                        d.ends_with("::process_mono") || d.contains("::process_mono_chunked")
                    }
                    _ => d.contains(method),
                }
            })
        });
        assert!(
            found,
            "Expected real-time entry point `{owner}{method}` to be audited in {}",
            rlib.display()
        );
    }
    assert!(
        has_sona_root,
        "Expected SONA real-time roots to be present and audited in {}",
        rlib.display()
    );
    assert!(
        root_count >= 15 && rt_closure.len() > root_count,
        "Expected >= 15 real-time root functions and non-empty transitive callees in {}, found {} roots and {} total in closure",
        rlib.display(),
        root_count,
        rt_closure.len()
    );
    assert!(
        crate_reloc_violations.is_empty(),
        "REQ-6.2 violation: forbidden libm/fmod relocations in {}:\n{}",
        rlib.display(),
        crate_reloc_violations.join("\n")
    );
    assert!(
        rt_violations.is_empty(),
        "REQ-6.2 / REQ-6.3 violation: pipeline-stalling instructions or nondeterministic calls in {}:\n{}",
        rlib.display(),
        rt_violations.join("\n")
    );
}
