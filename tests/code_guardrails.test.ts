/**
 * Code Quality, Type Safety, Unsafe Isolation, Real-Time Instruction, Modular Decomposition,
 * and `zerocopy` Wire-Struct Guardrail Tests.
 *
 * verify[repo.code.type-safety-and-imports]
 * verify[repo.code.zerocopy-wire-structs]
 * verify[repo.code.modular-decomposition-and-tests]
 * verify[repo.code.realtime-disassembly-audit]
 * verify[arch.simd.zero-stall-lowering]
 * verify[repo.code.explicit-simd-and-coverage]
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
  advanceAttributedBlockTracker,
  createBraceTracker,
  isCfgTestAttributeLine,
  PACKAGES_DIR,
  REPO_ROOT,
  relPath,
  SCRIPTS_DIR,
  stripRustCommentsAndStrings,
  stripRustCommentsOnly,
  TESTS_DIR,
  walkFilesByExtensions,
} from "./test-support.js";

const STRICT_FORBID_ROOTS = [
  "packages/kernels/src/lib.rs",
  "packages/fixtures/src/lib.rs",
  "packages/capture/src/lib.rs",
  "packages/studio/rust/lib.rs",
  "packages/studio/rust/main.rs",
  "packages/studio/build.rs",
  "packages/js/build.rs",
];

function checkTsAndMjsTypeSafety(): void {
  const tsViolations: string[] = [];
  const forbiddenTsPatterns = [
    new RegExp(`${"@ts"}-ignore\\b`),
    new RegExp(`${"@ts"}-expect-error\\b`),
    new RegExp(`:\\s*a${"ny"}\\b`),
    new RegExp(`\\bas\\s+a${"ny"}\\b`),
    new RegExp(`<a${"ny"}>`),
  ];
  for (const root of [PACKAGES_DIR, TESTS_DIR, SCRIPTS_DIR]) {
    for (const tsPath of walkFilesByExtensions(root, new Set([".ts", ".tsx", ".mjs"]))) {
      const lines = fs.readFileSync(tsPath, "utf8").split("\n");
      lines.forEach((line, idx) => {
        const trimmed = line.trim();
        if (trimmed.startsWith("//") || trimmed.startsWith("*")) return;
        if (forbiddenTsPatterns.some((pat) => pat.test(line))) {
          tsViolations.push(`${relPath(tsPath)}:${idx + 1}: ${trimmed}`);
        }
      });
    }
  }
  expect(tsViolations).toEqual([]);
}

function checkRustUnsafeIsolation(): void {
  const forbidOrDenyRe = /#\s*!\s*\[\s*(?:forbid|deny)\s*\(\s*unsafe_code\s*\)\s*\]/;
  for (const rustLibRel of ["packages/engine/src/lib.rs", "packages/js/rust/lib.rs"]) {
    const libPath = path.join(REPO_ROOT, rustLibRel);
    expect(fs.existsSync(libPath), `${rustLibRel} must exist`).toBe(true);
    expect(forbidOrDenyRe.test(stripRustCommentsAndStrings(fs.readFileSync(libPath, "utf8")))).toBe(
      true,
    );
  }

  const strictForbidSet = new Set<string>(STRICT_FORBID_ROOTS);
  for (const safePkg of ["kernels", "fixtures", "capture", "studio"]) {
    for (const subDir of ["src/bin", "rust/bin", "tests", "benches", "examples"]) {
      for (const p of walkFilesByExtensions(
        path.join(PACKAGES_DIR, safePkg, subDir),
        new Set([".rs"]),
      )) {
        strictForbidSet.add(relPath(p));
      }
    }
  }
  const strictForbidRe = /#\s*!\s*\[\s*forbid\s*\(\s*unsafe_code\s*\)\s*\]/;
  for (const strictRel of [...strictForbidSet].sort()) {
    const libPath = path.join(REPO_ROOT, strictRel);
    expect(fs.existsSync(libPath), `${strictRel} must exist`).toBe(true);
    expect(
      strictForbidRe.test(stripRustCommentsAndStrings(fs.readFileSync(libPath, "utf8"))),
      `${strictRel} missing #![forbid(unsafe_code)]`,
    ).toBe(true);
  }

  const unsafeOverrideRe = /\b(?:allow|warn|expect)\s*\([^)]*\bunsafe_code\b/;
  const unsafeConstructRe =
    /\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)|(?<!r#)\bunsafe\s*\(|#\s*!?\s*\[\s*unsafe\s*\(/;
  const unsafeBlockRe = /\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)/;
  const unsafeViolations: string[] = [];

  for (const safePkg of ["kernels", "fixtures", "capture", "studio"]) {
    for (const rsFile of walkFilesByExtensions(
      path.join(PACKAGES_DIR, safePkg),
      new Set([".rs"]),
    )) {
      const stripped = stripRustCommentsAndStrings(fs.readFileSync(rsFile, "utf8"));
      if (unsafeOverrideRe.test(stripped) || unsafeConstructRe.test(stripped)) {
        unsafeViolations.push(`${relPath(rsFile)}: contains forbidden unsafe construct`);
      }
    }
  }
  for (const rsFile of walkFilesByExtensions(path.join(PACKAGES_DIR, "js"), new Set([".rs"]))) {
    const rel = relPath(rsFile);
    const stripped = stripRustCommentsAndStrings(fs.readFileSync(rsFile, "utf8"));
    const isWasmExport =
      rel === "packages/js/rust/wasm.rs" ||
      (rel.startsWith("packages/js/rust/wasm/") && rel !== "packages/js/rust/wasm/abi.rs");
    if (unsafeBlockRe.test(stripped)) {
      unsafeViolations.push(`${rel}: contains forbidden unsafe block or item`);
    }
    if (!isWasmExport && (unsafeOverrideRe.test(stripped) || unsafeConstructRe.test(stripped))) {
      unsafeViolations.push(`${rel}: contains forbidden unsafe_code override or #[unsafe(...)]`);
    }
  }
  expect(unsafeViolations).toEqual([]);
}

function checkPackageLineCeilings(): void {
  const violations: string[] = [];
  const targets: Array<{ dir: string; exts: Set<string> }> = [
    { dir: path.join(PACKAGES_DIR, "kernels"), exts: new Set([".rs"]) },
    { dir: path.join(PACKAGES_DIR, "engine"), exts: new Set([".rs"]) },
    { dir: path.join(PACKAGES_DIR, "fixtures"), exts: new Set([".rs"]) },
    { dir: path.join(PACKAGES_DIR, "capture"), exts: new Set([".rs"]) },
    { dir: path.join(PACKAGES_DIR, "ui"), exts: new Set([".ts", ".tsx"]) },
    { dir: path.join(PACKAGES_DIR, "js"), exts: new Set([".rs", ".ts", ".tsx"]) },
    { dir: path.join(PACKAGES_DIR, "studio"), exts: new Set([".rs", ".ts", ".tsx"]) },
    { dir: TESTS_DIR, exts: new Set([".ts"]) },
    { dir: SCRIPTS_DIR, exts: new Set([".mjs"]) },
  ];
  for (const { dir, exts } of targets) {
    for (const srcFile of walkFilesByExtensions(dir, exts)) {
      const content = fs.readFileSync(srcFile, "utf8");
      const lineCount = content.endsWith("\n")
        ? content.split("\n").length - 1
        : content.split("\n").length;
      if (lineCount >= 900) violations.push(`${relPath(srcFile)} (${lineCount} lines >= 900)`);
    }
  }
  expect(violations).toEqual([]);
}

function scanRustProdLines(
  rsPath: string,
  checkLine: (strippedLine: string, commentFreeLine: string, lineNo: number) => void,
): void {
  const raw = fs.readFileSync(rsPath, "utf8");
  const commentFreeLines = stripRustCommentsOnly(raw).split("\n");
  const strippedLines = stripRustCommentsAndStrings(raw).split("\n");
  const cfgTestTracker = createBraceTracker();
  for (let idx = 0; idx < strippedLines.length; idx++) {
    const sLine = strippedLines[idx] ?? "";
    const cfTrimmed = (commentFreeLines[idx] ?? "").trim();
    if (
      advanceAttributedBlockTracker(sLine, cfgTestTracker, () => isCfgTestAttributeLine(cfTrimmed))
    ) {
      continue;
    }
    checkLine(sLine, commentFreeLines[idx] ?? "", idx + 1);
  }
}

function scanKernelsBottomTestsAnd64Bit(rsPath: string, violations: string[]): void {
  const rel = relPath(rsPath);
  const raw = fs.readFileSync(rsPath, "utf8");
  const commentFreeLines = stripRustCommentsOnly(raw).split("\n");
  const strippedLines = stripRustCommentsAndStrings(raw).split("\n");
  const nonSpirvTracker = createBraceTracker();
  const cfgTestTracker = createBraceTracker();
  const forbiddenTypeRe = /\b(?:f64|u64|i64|u128|i128)\b/;

  let inActiveTestMod = false;
  let completedTestMod = false;
  for (let idx = 0; idx < strippedLines.length; idx++) {
    const sLine = strippedLines[idx] ?? "";
    const cfTrimmed = (commentFreeLines[idx] ?? "").trim();
    const trimmed = sLine.trim();
    const inCfgTest = advanceAttributedBlockTracker(sLine, cfgTestTracker, () =>
      isCfgTestAttributeLine(cfTrimmed),
    );
    if (inCfgTest) {
      if (/\bmod\s+[A-Za-z_][A-Za-z0-9_]*\b/.test(cfTrimmed)) {
        inActiveTestMod = true;
      }
      if (inActiveTestMod && !cfgTestTracker.awaitingOpenOrSemi && cfgTestTracker.depth <= 0) {
        completedTestMod = true;
        inActiveTestMod = false;
      }
      continue;
    }
    if (completedTestMod && trimmed.length > 0) {
      violations.push(`${rel}:${idx + 1}: production code after #[cfg(test)] mod tests`);
      completedTestMod = false;
    }
    if (
      advanceAttributedBlockTracker(sLine, nonSpirvTracker, () =>
        /^#\s*\[\s*cfg\s*\(\s*not\s*\(\s*target_arch\s*=\s*"spirv"\s*\)\s*\)\s*\]/.test(cfTrimmed),
      )
    ) {
      continue;
    }
    if (forbiddenTypeRe.test(sLine)) {
      violations.push(`${rel}:${idx + 1}: forbidden 64/128-bit type in GPU kernel: ${trimmed}`);
    }
  }
}

function checkRustModularFile(rsPath: string, violations: string[]): void {
  const sharedHelpers = new Set(["test_support.rs", "test_utils.rs", "testing.rs"]);
  const isolatedTestRe = /^(?:tests?|.*_tests|test_.*)\.rs$/;
  const externalModRe = /^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(?:tests|[a-z0-9_]+_tests)\s*;/m;
  const rel = relPath(rsPath).split(path.sep).join("/");
  const base = path.basename(rsPath);
  const content = fs.readFileSync(rsPath, "utf8");

  if (
    /^packages\/[^/]+\/(?:src|rust)\//.test(rel) &&
    !/^packages\/[^/]+\/(?:src|rust)\/bin\//.test(rel)
  ) {
    if (isolatedTestRe.test(base) && !sharedHelpers.has(base)) {
      violations.push(`${rel}: isolated unit-test file in src/rust`);
    }
    const declMatch = externalModRe.exec(content);
    if (declMatch) violations.push(`${rel}: external unit-test mod '${declMatch[0].trim()}'`);
  }
  if (/^packages\/[^/]+\/tests\//.test(rel) && base.endsWith("_coverage.rs")) {
    violations.push(`${rel}: grab-bag '*_coverage.rs' integration test file`);
  }
  if (/\\n\\(?:\r?\n)/.test(content)) {
    violations.push(`${rel}: literal '\\n\\\\' line-continuation golf`);
  }
}

function checkTsModularFile(tsPath: string, violations: string[]): void {
  const genericNames = new Set([
    "test.ts",
    "tests.ts",
    "test.tsx",
    "tests.tsx",
    "test.js",
    "tests.js",
    "test.mjs",
    "tests.mjs",
  ]);
  const rel = relPath(tsPath).split(path.sep).join("/");
  const base = path.basename(tsPath);
  const content = fs.readFileSync(tsPath, "utf8");
  const isSrc = /^packages\/[^/]+\/src\//.test(rel);
  const isPkgTest = /^packages\/[^/]+\/tests\//.test(rel);

  if (isSrc) {
    if (rel.includes("/__tests__/")) violations.push(`${rel}: isolated '__tests__/' directory`);
    if (genericNames.has(base)) violations.push(`${rel}: generic unpaired '${base}' in src/`);
    if (/\.spec\.(?:ts|tsx|js|mjs)$/.test(base)) violations.push(`${rel}: '*.spec.*' inside src/`);
  }
  if ((isSrc || isPkgTest) && /coverage[^/]*\.(?:test|spec)\.(?:ts|tsx|js|mjs)$/.test(base)) {
    violations.push(`${rel}: grab-bag '*coverage*' test file`);
  }
  if (/\\n\\(?:\r?\n)/.test(content)) {
    violations.push(`${rel}: literal '\\n\\\\' line-continuation golf`);
  }
}

describe("DNAM Code Quality, Real-Time & Modular Decomposition Guardrails", () => {
  it("test_type_safety_unsafe_isolation_and_line_ceilings", () => {
    checkTsAndMjsTypeSafety();
    checkRustUnsafeIsolation();
    checkPackageLineCeilings();
  });

  it("test_engine_realtime_no_libm_or_float_rem_euclid", () => {
    const engineSrcDir = path.join(PACKAGES_DIR, "engine", "src");
    const kernelsSrcDir = path.join(PACKAGES_DIR, "kernels", "src");
    const forbiddenCallRe =
      /\.(?:rem_euclid|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|exp|exp2|ln|log2|log10|powf|powi|sqrt|cbrt|hypot)\s*\(/;
    const violations: string[] = [];

    for (const srcDir of [engineSrcDir, kernelsSrcDir]) {
      for (const rsPath of walkFilesByExtensions(srcDir, new Set([".rs"]))) {
        const base = path.basename(rsPath);
        if (base === "test_support.rs") continue;
        scanRustProdLines(rsPath, (sLine, _raw, ln) => {
          if (forbiddenCallRe.test(sLine)) {
            violations.push(`${relPath(rsPath)}:${ln}: ${sLine.trim()}`);
          }
        });
      }
    }
    for (const rsPath of walkFilesByExtensions(kernelsSrcDir, new Set([".rs"]))) {
      scanKernelsBottomTestsAnd64Bit(rsPath, violations);
    }
    expect(violations).toEqual([]);
  });

  it("test_engine_no_coverage_off_or_simd_macros", () => {
    const engineSrcDir = path.join(PACKAGES_DIR, "engine", "src");
    const kernelsSrcDir = path.join(PACKAGES_DIR, "kernels", "src");
    const violations: string[] = [];

    for (const srcDir of [engineSrcDir, kernelsSrcDir]) {
      for (const rsPath of walkFilesByExtensions(srcDir, new Set([".rs"]))) {
        if (fs.readFileSync(rsPath, "utf8").includes("coverage(off)")) {
          violations.push(`${relPath(rsPath)}: contains forbidden coverage(off)`);
        }
      }
    }
    for (const simdDir of [path.join(engineSrcDir, "math"), path.join(kernelsSrcDir, "simd")]) {
      for (const rsPath of walkFilesByExtensions(simdDir, new Set([".rs"]))) {
        if (fs.readFileSync(rsPath, "utf8").includes("macro_rules!")) {
          violations.push(`${relPath(rsPath)}: contains forbidden macro_rules! in SIMD modules`);
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_zerocopy_wire_no_manual_endian_byte_slicing", () => {
    const rawLeRe = /(?:\.|::)(?:from_le_bytes|to_le_bytes)\s*\(/;
    const violations: string[] = [];
    for (const safeDir of [
      path.join(PACKAGES_DIR, "kernels", "src"),
      path.join(PACKAGES_DIR, "engine", "src", "container"),
      path.join(PACKAGES_DIR, "engine", "src", "control"),
      path.join(PACKAGES_DIR, "engine", "src", "provenance"),
      path.join(PACKAGES_DIR, "engine", "src", "stages"),
      path.join(PACKAGES_DIR, "engine", "src", "schema"),
      path.join(PACKAGES_DIR, "studio", "rust"),
    ]) {
      for (const rsPath of walkFilesByExtensions(safeDir, new Set([".rs"]))) {
        scanRustProdLines(rsPath, (sLine, _raw, ln) => {
          if (rawLeRe.test(sLine)) violations.push(`${relPath(rsPath)}:${ln}: ${sLine.trim()}`);
        });
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_modular_decomposition_and_no_barrels_or_isolated_tests", () => {
    const modRsFiles = walkFilesByExtensions(PACKAGES_DIR, new Set([".rs"]))
      .filter((p) => path.basename(p) === "mod.rs")
      .map(relPath);
    expect(modRsFiles).toEqual([]);

    const violations: string[] = [];
    for (const rsPath of walkFilesByExtensions(PACKAGES_DIR, new Set([".rs"]))) {
      checkRustModularFile(rsPath, violations);
    }
    for (const tsPath of walkFilesByExtensions(
      PACKAGES_DIR,
      new Set([".ts", ".tsx", ".js", ".mjs"]),
    )) {
      checkTsModularFile(tsPath, violations);
    }
    expect(violations).toEqual([]);

    const allowedJsBarrels = new Set([
      "packages/js/src/index.ts",
      "packages/js/src/dsp/index.ts",
      "packages/js/src/parameters/index.ts",
      "packages/js/src/provenance/index.ts",
      "packages/js/src/schema/index.ts",
      "packages/js/src/tempo/index.ts",
      "packages/js/src/web-audio/index.ts",
    ]);
    const barrelViolations: string[] = [];
    for (const pkg of ["ui", "studio", "js"]) {
      for (const file of walkFilesByExtensions(
        path.join(PACKAGES_DIR, pkg, "src"),
        new Set([".ts", ".tsx"]),
      )) {
        const base = path.basename(file);
        if (base !== "index.ts" && base !== "index.tsx") continue;
        const rel = relPath(file).split(path.sep).join("/");
        if (pkg !== "js" || !allowedJsBarrels.has(rel)) {
          barrelViolations.push(`${rel}: forbidden internal barrel index file`);
        }
      }
    }
    expect(barrelViolations).toEqual([]);

    for (const dirPath of [
      path.join(PACKAGES_DIR, "kernels", "src"),
      path.join(PACKAGES_DIR, "engine", "src", "math"),
      path.join(PACKAGES_DIR, "engine", "src", "operators"),
      path.join(PACKAGES_DIR, "engine", "src", "stages"),
      path.join(PACKAGES_DIR, "engine", "src", "control"),
      path.join(PACKAGES_DIR, "engine", "src", "container"),
      path.join(PACKAGES_DIR, "engine", "src", "provenance"),
      path.join(PACKAGES_DIR, "engine", "src", "schema"),
      path.join(PACKAGES_DIR, "engine", "tests"),
      path.join(PACKAGES_DIR, "capture"),
      path.join(PACKAGES_DIR, "fixtures"),
    ]) {
      const hasProptest = walkFilesByExtensions(dirPath, new Set([".rs"])).some((f) =>
        fs.readFileSync(f, "utf8").includes("proptest!"),
      );
      expect(hasProptest, `${relPath(dirPath)} missing proptest! suite`).toBe(true);
    }
  });
});
