#![forbid(unsafe_code)]
#![allow(dead_code)]
//! Release `.rlib` builder and `objdump` disassembly/symbol-alias parser for the
//! `dnam-engine` + `dnam-vm` + `dnam-kernels` real-time disassembly audit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) type SectionAddrKey = (String, String, u64);
pub(crate) type SymPair = (String, String);
pub(crate) type AliasMap = BTreeMap<SectionAddrKey, Vec<SymPair>>;

#[derive(Default)]
pub(crate) struct FnDisasm {
    pub(crate) demangled: String,
    pub(crate) instructions: Vec<(String, String)>,
    pub(crate) relocs: Vec<(String, String)>,
    pub(crate) callees: BTreeSet<String>,
}

pub(crate) fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

pub(crate) fn current_target_triple() -> &'static str {
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

pub(crate) fn clean_reloc_target(raw: &str) -> &str {
    let s = raw.trim_start_matches('_');
    let s = s.split('@').next().unwrap_or(s);
    let s = s.split('+').next().unwrap_or(s);
    s.split('-').next().unwrap_or(s)
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
    while let Some(vis) = sym_tail.split_whitespace().next()
        && matches!(vis, ".hidden" | ".protected" | ".internal")
    {
        sym_tail = sym_tail[vis.len()..].trim_start();
    }
    if sym_tail.is_empty() {
        return None;
    }
    Some((addr, section, sym_tail))
}

fn extract_rlib_from_cargo_json(stdout: &[u8], prefix: &str) -> Option<PathBuf> {
    let text = String::from_utf8_lossy(stdout);
    let mut found: Option<PathBuf> = None;
    for line in text.lines() {
        if !line.contains("\"compiler-artifact\"") {
            continue;
        }
        for token in line.split('"') {
            let norm = token.replace("\\\\", "/");
            if !norm.ends_with(".rlib") {
                continue;
            }
            let p = PathBuf::from(&norm);
            let is_match = p
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".rlib"));
            if is_match && p.is_file() {
                found = Some(p);
            }
        }
    }
    found
}

fn find_rlib_in_deps(deps_dir: &Path, prefix: &str) -> Option<PathBuf> {
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(deps_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".rlib"))
                && p.is_file()
        })
        .filter_map(|p| p.metadata().and_then(|m| m.modified()).ok().map(|t| (t, p)))
        .collect();
    candidates.sort_by_key(|(t, _)| *t);
    candidates.pop().map(|(_, p)| p)
}

pub(crate) fn ensure_release_rlibs() -> [PathBuf; 3] {
    let root = workspace_root();
    let triple = current_target_triple();
    let audit_target_dir = root.join("target").join("release-audit");
    let out = Command::new("cargo")
        .current_dir(&root)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("LLVM_PROFILE_FILE")
        .env("CARGO_PROFILE_RELEASE_LTO", "false")
        .env("CARGO_PROFILE_RELEASE_STRIP", "none")
        .args([
            "build",
            "--package",
            "dnam-engine",
            "--release",
            "--message-format=json-render-diagnostics",
            "--target",
            triple,
            "--target-dir",
        ])
        .arg(&audit_target_dir)
        .output()
        .expect("failed to invoke cargo build --release for disassembly audit");
    assert!(
        out.status.success(),
        "cargo build --release failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let rel_dir = audit_target_dir.join(triple).join("release");
    let deps_dir = rel_dir.join("deps");
    let engine_rlib = extract_rlib_from_cargo_json(&out.stdout, "libdnam_engine")
        .unwrap_or_else(|| rel_dir.join("libdnam_engine.rlib"));
    let vm_rlib = extract_rlib_from_cargo_json(&out.stdout, "libdnam_vm-")
        .or_else(|| find_rlib_in_deps(&deps_dir, "libdnam_vm-"))
        .expect("expected libdnam_vm-*.rlib in cargo build output");
    let kernels_rlib = extract_rlib_from_cargo_json(&out.stdout, "libdnam_kernels-")
        .or_else(|| find_rlib_in_deps(&deps_dir, "libdnam_kernels-"))
        .expect("expected libdnam_kernels-*.rlib in cargo build output");

    for p in [&engine_rlib, &vm_rlib, &kernels_rlib] {
        assert!(p.is_file(), "expected release rlib at {}", p.display());
    }
    [engine_rlib, vm_rlib, kernels_rlib]
}

pub(crate) fn collect_symbol_aliases(rlib_tag: &str, rlib: &Path, aliases: &mut AliasMap) {
    let sym_raw_out = Command::new("objdump")
        .arg("-t")
        .arg(rlib)
        .output()
        .expect("failed to execute objdump -t");
    let sym_dem_out = Command::new("objdump")
        .args(["-t", "--demangle"])
        .arg(rlib)
        .output()
        .expect("failed to execute objdump -t --demangle");
    assert!(
        sym_raw_out.status.success() && sym_dem_out.status.success(),
        "objdump -t failed on {}",
        rlib.display()
    );

    let mut current_obj = String::new();
    for (raw_line, dem_line) in String::from_utf8_lossy(&sym_raw_out.stdout)
        .lines()
        .zip(String::from_utf8_lossy(&sym_dem_out.stdout).lines())
    {
        if let Some(obj) = parse_objdump_object_header(raw_line) {
            current_obj = format!("{rlib_tag}::{obj}");
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
}

pub(crate) fn disassemble_rlib_into(
    rlib_tag: &str,
    rlib: &Path,
    fns: &mut BTreeMap<String, FnDisasm>,
    blocks: &mut BTreeMap<SectionAddrKey, FnDisasm>,
) {
    let out_raw = Command::new("objdump")
        .args(["-dr"])
        .arg(rlib)
        .output()
        .expect("failed to execute objdump -dr");
    let out_dem = Command::new("objdump")
        .args(["-dr", "--demangle"])
        .arg(rlib)
        .output()
        .expect("failed to execute objdump -dr --demangle");
    assert!(
        out_raw.status.success() && out_dem.status.success(),
        "objdump -dr failed on {}",
        rlib.display()
    );

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

    let (mut current_obj, mut current_section, mut current_fn, mut current_fn_clean) =
        (String::new(), String::new(), String::new(), String::new());
    let mut current_block: Option<SectionAddrKey> = None;
    let mut header_idx = 0usize;

    for raw_line in String::from_utf8_lossy(&out_raw.stdout).lines() {
        let line = raw_line.trim();
        if let Some(obj) = parse_objdump_object_header(line) {
            current_obj = format!("{rlib_tag}::{obj}");
            current_section.clear();
            current_block = None;
            current_fn.clear();
            current_fn_clean.clear();
            continue;
        }
        if let Some(sec_rest) = line.strip_prefix("Disassembly of section ") {
            current_section = sec_rest.trim_end_matches(':').trim().to_string();
            current_block = None;
            current_fn.clear();
            current_fn_clean.clear();
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
                current_fn = format!("{rlib_tag}::{clean_key}");
                current_fn_clean = clean_key;
                fns.entry(current_fn.clone()).or_default().demangled = demangled;
            } else {
                current_fn.clear();
                current_fn_clean.clear();
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
                    if callee_key != current_fn_clean && !is_local_temp_label(&callee_key) {
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
                    && base != current_fn_clean
                {
                    entry.callees.insert(base);
                }
            }
        }
    }
}

pub(crate) fn merge_folded_symbol_aliases(
    alias_groups: &AliasMap,
    blocks: &BTreeMap<SectionAddrKey, FnDisasm>,
    fns: &mut BTreeMap<String, FnDisasm>,
) {
    for (loc, group) in alias_groups {
        let rlib_tag = loc.0.split("::").next().unwrap_or("");
        let mut merged_instructions = Vec::new();
        let mut merged_relocs = Vec::new();
        let mut merged_callees = BTreeSet::new();
        if let Some(block) = blocks.get(loc) {
            merged_instructions.extend(block.instructions.iter().cloned());
            merged_relocs.extend(block.relocs.iter().cloned());
            merged_callees.extend(block.callees.iter().cloned());
        } else {
            for (alias_key, _) in group {
                let scoped_key = format!("{rlib_tag}::{alias_key}");
                if let Some(existing) = fns.get(&scoped_key) {
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
            let scoped_key = format!("{rlib_tag}::{alias_key}");
            let entry = fns.entry(scoped_key).or_default();
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
}
