/**
 * Monorepo Architecture, Crate Isolation, Black-Box Purity, SONA Tier Budget,
 * `.idea/*.iml`, and Shared `pnpm` Tooling Guardrail Tests.
 *
 * verify[repo.code.type-safety-and-imports]
 * verify[repo.code.constant-eval-and-derivations]
 * verify[repo.code.modular-decomposition-and-tests]
 * verify[repo.code.universal-black-box]
 * verify[arch.index.subsystems]
 * verify[arch.tiers.maximize-quality-within-budget]
 * verify[qual.tiers.floors-not-stopping-criteria]
 * verify[train.pipeline.webgpu-scans-and-ozaki]
 * verify[sdk.rust.safe-rust-and-boundaries]
 * verify[sdk.rust.shared-nostd-kernels]
 * verify[sdk.rust.zerocopy-wire-structs]
 * verify[sdk.rust.modular-layout-and-proptest]
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
  ADR_DIR,
  collectSpecRules,
  DOCS_DIR,
  PACKAGES_DIR,
  parseFrontmatter,
  parseTomlSections,
  REPO_ROOT,
  relPath,
  SCRIPTS_DIR,
  SPECS_DIR,
  stripRustCommentsAndStrings,
  TESTS_DIR,
  TOOLS_DIR,
  walkFilesByExtensions,
} from "./test-support.js";

function checkFixturesManifestStrictness(): void {
  const allowed = new Set(["", "package", "lib", "lints", "dependencies", "dev-dependencies"]);
  const cargo = fs.readFileSync(path.join(PACKAGES_DIR, "fixtures", "Cargo.toml"), "utf8");
  for (const sec of parseTomlSections(cargo)) {
    expect(allowed.has(sec.header), `Disallowed fixtures section [${sec.header}]`).toBe(true);
    expect(/(?:^|\n|\.)\s*target\s*(?:=|\.)/.test(sec.body)).toBe(false);
    expect(/(?:^|\n|\.)\s*(?:(?:build|dev)-)?dependencies\s*(?:=|\.)/.test(sec.body)).toBe(false);
    if (sec.header === "dependencies") {
      const depKeys = sec.body
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .map((line) => line.split(/[=.]/)[0]?.trim() ?? "");
      for (const key of depKeys) {
        expect(key === "nalgebra" || key === "realfft", `Unexpected fixtures dep '${key}'`).toBe(
          true,
        );
      }
    }
  }
  for (const bad of [
    "dnam-engine",
    "dnam-capture",
    "dnam-train",
    "dnam-trainer",
    "dnam-studio",
    "dnam-js",
    "dnam_js",
  ]) {
    expect(cargo).not.toContain(bad);
  }
}

function checkSeparateIdeaImlsForPackages(): void {
  const ideaDir = path.join(REPO_ROOT, ".idea");
  const modulesXml = fs.readFileSync(path.join(ideaDir, "modules.xml"), "utf8");
  const workspaceIml = fs.readFileSync(path.join(ideaDir, "dnam-workspace.iml"), "utf8");
  const imlFiles = fs
    .readdirSync(ideaDir)
    .filter((f) => f.endsWith(".iml"))
    .map((f) => ({ name: f, body: fs.readFileSync(path.join(ideaDir, f), "utf8") }));

  for (const { name, body } of imlFiles) {
    for (const m of body.matchAll(/<sourceFolder\s+url="file:\/\/\$MODULE_DIR\$\/([^"]+)"/g)) {
      const folderRel = m[1] ?? "";
      expect(
        fs.existsSync(path.join(REPO_ROOT, folderRel)),
        `.idea/${name} references nonexistent <sourceFolder> '${folderRel}'`,
      ).toBe(true);
    }
  }

  const pkgImls = imlFiles.filter((m) => m.name !== "dnam-workspace.iml");
  const checkRoot = (parentRel: string, pkgName: string): void => {
    const contentUrl = `file://$MODULE_DIR$/${parentRel}/${pkgName}`;
    expect(
      workspaceIml,
      `dnam-workspace.iml must not inline ${parentRel}/${pkgName}`,
    ).not.toContain(`/${parentRel}/${pkgName}`);
    const owner = pkgImls.find((m) => m.body.includes(`"${contentUrl}"`));
    expect(owner, `Missing dedicated .idea/*.iml with content url="${contentUrl}"`).toBeDefined();
    if (owner) expect(modulesXml).toContain(`$PROJECT_DIR$/.idea/${owner.name}`);
  };

  for (const pkgEntry of fs.readdirSync(PACKAGES_DIR, { withFileTypes: true })) {
    if (pkgEntry.isDirectory()) checkRoot("packages", pkgEntry.name);
  }
  const toolsDir = path.join(REPO_ROOT, "tools");
  if (fs.existsSync(toolsDir)) {
    for (const toolEntry of fs.readdirSync(toolsDir, { withFileTypes: true })) {
      const toolPath = path.join(toolsDir, toolEntry.name);
      if (
        toolEntry.isDirectory() &&
        (fs.existsSync(path.join(toolPath, "Cargo.toml")) ||
          fs.existsSync(path.join(toolPath, "package.json")))
      ) {
        checkRoot("tools", toolEntry.name);
      }
    }
  }
}

function checkCargoAndMoonBoundaries(): void {
  const kernelsCargo = fs.readFileSync(path.join(PACKAGES_DIR, "kernels", "Cargo.toml"), "utf8");
  const depsSec = parseTomlSections(kernelsCargo).find((s) => s.header === "dependencies");
  expect((depsSec?.body ?? "").trim()).toBe("");
  expect(fs.readFileSync(path.join(PACKAGES_DIR, "kernels", "moon.yml"), "utf8")).toMatch(
    /^dependsOn:\s*\[\]\s*$/m,
  );

  const vmCargo = fs.readFileSync(path.join(PACKAGES_DIR, "vm", "Cargo.toml"), "utf8");
  const vmDepsSec = parseTomlSections(vmCargo).find((s) => s.header === "dependencies");
  const vmDepKeys = (vmDepsSec?.body ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"))
    .map((line) => line.split(/[=.]/)[0]?.trim() ?? "")
    .sort();
  expect(vmDepKeys, "dnam-vm must depend strictly on dnam-kernels and zerocopy").toEqual([
    "dnam-kernels",
    "zerocopy",
  ]);
  for (const forbiddenVmDep of [
    "dnam-engine",
    "dnam-train",
    "dnam-capture",
    "dnam-fixtures",
    "serde",
    "serde_json",
    "ed25519-dalek",
    "x25519-dalek",
    "chacha20poly1305",
    "hpke",
    "wgpu",
    "libc",
  ]) {
    expect(vmCargo, `dnam-vm Cargo.toml must not reference ${forbiddenVmDep}`).not.toContain(
      forbiddenVmDep,
    );
  }
  const vmMoon = fs.readFileSync(path.join(PACKAGES_DIR, "vm", "moon.yml"), "utf8");
  expect(vmMoon).toMatch(/^dependsOn:\s*(?:\[\s*"kernels"\s*\]|\n\s*-\s*"?kernels"?\s*)$/m);
  expect(vmMoon).not.toContain("fixtures");
  expect(vmMoon).not.toContain("engine");
  expect(fs.existsSync(path.join(PACKAGES_DIR, "trainer"))).toBe(false);
  expect(
    fs.existsSync(path.join(DOCS_DIR, "specs", "containers", "checkpoint_and_cache_format.md")),
  ).toBe(false);
  expect(
    fs.existsSync(path.join(DOCS_DIR, "specs", "containers", "project_container_format.md")),
  ).toBe(true);

  const fixturesRefRe = /dnam[-_]fixtures|packages\/fixtures|\.\.\/fixtures/;
  const engineCargo = fs.readFileSync(path.join(PACKAGES_DIR, "engine", "Cargo.toml"), "utf8");
  expect(engineCargo).not.toMatch(fixturesRefRe);
  expect(engineCargo).not.toMatch(
    /dnam[-_](?:capture|train(?:er)?)|packages\/(?:capture|train(?:er)?)|\.\.\/(?:capture|train(?:er)?)/,
  );
  expect(fs.readFileSync(path.join(PACKAGES_DIR, "fixtures", "moon.yml"), "utf8")).toMatch(
    /^dependsOn:\s*\[\]\s*$/m,
  );
  const engineMoon = fs.readFileSync(path.join(PACKAGES_DIR, "engine", "moon.yml"), "utf8");
  expect(engineMoon).toMatch(
    /^dependsOn:\s*(?:\[\s*"kernels"\s*,\s*"vm"\s*\]|\[\s*"vm"\s*,\s*"kernels"\s*\]|\n\s*-\s*"?kernels"?\s*\n\s*-\s*"?vm"?\s*)$/m,
  );
  expect(engineMoon).not.toMatch(/\b(?:fixtures|capture|train(?:er)?)\b/);

  const trainCargo = fs.readFileSync(path.join(PACKAGES_DIR, "train", "Cargo.toml"), "utf8");
  const trainDepsSec = parseTomlSections(trainCargo).find((s) => s.header === "dependencies");
  const trainDepKeys = (trainDepsSec?.body ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"))
    .map((line) => line.split(/[=.]/)[0]?.trim() ?? "")
    .sort();
  expect(
    trainDepKeys,
    "dnam-train must depend strictly on dnam-kernels, dnam-vm, pollster, sha2, wgpu, and zerocopy",
  ).toEqual(["dnam-kernels", "dnam-vm", "pollster", "sha2", "wgpu", "zerocopy"]);
  for (const forbiddenTrainDep of [
    "dnam-engine",
    "dnam-capture",
    "dnam-fixtures",
    "sona-shaders",
    "serde",
    "serde_json",
    "ndarray",
    "nalgebra",
    "rustfft",
    "realfft",
    "clap",
    "cpal",
  ]) {
    expect(
      trainCargo,
      `dnam-train Cargo.toml must not reference ${forbiddenTrainDep}`,
    ).not.toContain(forbiddenTrainDep);
  }
  const trainMoon = fs.readFileSync(path.join(PACKAGES_DIR, "train", "moon.yml"), "utf8");
  expect(trainMoon).toMatch(
    /^dependsOn:\s*(?:\[\s*"kernels"\s*,\s*"vm"\s*\]|\[\s*"vm"\s*,\s*"kernels"\s*\]|\n\s*-\s*"?kernels"?\s*\n\s*-\s*"?vm"?\s*)$/m,
  );
  expect(trainMoon).not.toMatch(/\b(?:engine|capture|fixtures)\b/);
  expect(trainMoon.replaceAll("/tools/sona-shaders/wgsl/**/*", "")).not.toMatch(/\bsona-shaders\b/);

  const captureCargo = fs.readFileSync(path.join(PACKAGES_DIR, "capture", "Cargo.toml"), "utf8");
  let foundFixturesDev = false;
  let foundEngineDep = false;
  for (const sec of parseTomlSections(captureCargo)) {
    expect(sec.header).not.toBe("bin");
    if (fixturesRefRe.test(sec.header) || fixturesRefRe.test(sec.body)) {
      expect(sec.header).toBe("dev-dependencies");
      foundFixturesDev = true;
    }
    if (sec.header === "dependencies" && sec.body.includes("dnam-engine")) foundEngineDep = true;
  }
  expect(foundFixturesDev && foundEngineDep).toBe(true);
  expect(captureCargo).not.toMatch(/\b(?:clap|cpal|toml)\b/);
  expect(fs.readFileSync(path.join(PACKAGES_DIR, "studio", "Cargo.toml"), "utf8")).not.toMatch(
    /\b(?:clap|toml)\b/,
  );

  const captureMoon = fs.readFileSync(path.join(PACKAGES_DIR, "capture", "moon.yml"), "utf8");
  expect(captureMoon).toMatch(/^\s*-\s*"engine"\s*$/m);
  expect(captureMoon).toMatch(/-\s*id:\s*"fixtures"\s*\n\s*scope:\s*"development"/);

  for (const pkg of ["studio", "js"]) {
    expect(fs.readFileSync(path.join(PACKAGES_DIR, pkg, "Cargo.toml"), "utf8")).not.toMatch(
      fixturesRefRe,
    );
  }
  for (const pkg of ["studio", "js", "ui"]) {
    expect(fs.readFileSync(path.join(PACKAGES_DIR, pkg, "moon.yml"), "utf8")).not.toContain(
      "fixtures",
    );
  }
}

interface WgslManifestEntry {
  name?: string;
  wgsl_path?: string;
  sha256?: string;
  workgroup_sram_bytes?: number;
}

interface WgslManifestFile {
  version?: number;
  portable?: WgslManifestEntry[];
  subgroups?: WgslManifestEntry[];
}

function checkSonaShadersArtifacts(): void {
  const rootCargo = fs.readFileSync(path.join(REPO_ROOT, "Cargo.toml"), "utf8");
  const wsSec = parseTomlSections(rootCargo).find((s) => s.header === "workspace");
  expect(wsSec?.body ?? "").toMatch(/exclude\s*=\s*\[[^\]]*"tools\/sona-shaders"/);

  const shaderToolDir = path.join(TOOLS_DIR, "sona-shaders");
  expect(
    fs.existsSync(path.join(shaderToolDir, "Cargo.lock")),
    "tools/sona-shaders/Cargo.lock must be checked in",
  ).toBe(true);
  const topCargo = fs.readFileSync(path.join(shaderToolDir, "Cargo.toml"), "utf8");
  expect(topCargo, "tools/sona-shaders/Cargo.toml missing isolated [workspace]").toMatch(
    /^\s*\[workspace\]/m,
  );
  const gpuCargo = fs.readFileSync(
    path.join(shaderToolDir, "crates", "sona-shaders-gpu", "Cargo.toml"),
    "utf8",
  );
  expect(
    gpuCargo,
    "tools/sona-shaders/crates/sona-shaders-gpu/Cargo.toml missing isolated [workspace]",
  ).toMatch(/^\s*\[workspace\]/m);

  const forbiddenShaderDeps = /\b(?:dnam[-_]fixtures|ndarray|dialoguer|realfft|rustfft|nalgebra)\b/;
  for (const cargoRel of ["Cargo.toml", path.join("crates", "sona-shaders-gpu", "Cargo.toml")]) {
    const body = fs.readFileSync(path.join(shaderToolDir, cargoRel), "utf8");
    expect(body, `tools/sona-shaders/${cargoRel} missing edition = "2024"`).toContain(
      'edition = "2024"',
    );
    expect(body, `tools/sona-shaders/${cargoRel} missing publish = false`).toMatch(
      /publish\s*=\s*false/,
    );
    expect(body, `tools/sona-shaders/${cargoRel} missing LicenseRef-DNAM-1.0.0`).toContain(
      'license = "LicenseRef-DNAM-1.0.0"',
    );
    expect(body, `tools/sona-shaders/${cargoRel} has forbidden dep`).not.toMatch(
      forbiddenShaderDeps,
    );
  }

  const wgslDir = path.join(shaderToolDir, "wgsl");
  const manifest = JSON.parse(
    fs.readFileSync(path.join(wgslDir, "manifest.json"), "utf8"),
  ) as WgslManifestFile;
  expect(manifest.version).toBe(1);

  const expectedKernels = [
    "prescreen_population",
    "fused_core_scan_256",
    "adjoint_scan_256",
    "gabor_bezout_grad",
    "ozaki_gram_rhs_i32",
    "inner_opt_step",
  ].sort();

  for (const entries of [manifest.portable, manifest.subgroups]) {
    expect(Array.isArray(entries)).toBe(true);
    const list = entries ?? [];
    expect(list.map((entry) => entry.name ?? "").sort()).toEqual(expectedKernels);
    for (const entry of list) {
      const wgslRel = entry.wgsl_path ?? "";
      const wgslPath = path.join(wgslDir, wgslRel);
      expect(fs.existsSync(wgslPath), `Missing WGSL artifact ${relPath(wgslPath)}`).toBe(true);
      const wgslBytes = fs.readFileSync(wgslPath);
      const wgsl = wgslBytes.toString("utf8");
      const actualSha = crypto.createHash("sha256").update(wgslBytes).digest("hex");
      expect(entry.sha256, `${wgslRel} manifest sha256 mismatch`).toBe(actualSha);
      expect(
        entry.workgroup_sram_bytes,
        `${wgslRel} workgroup_sram_bytes must fit <= 15,872 B`,
      ).toBeGreaterThanOrEqual(0);
      expect(
        entry.workgroup_sram_bytes,
        `${wgslRel} workgroup_sram_bytes must fit <= 15,872 B`,
      ).toBeLessThanOrEqual(15_872);
      expect(wgsl.trim().length, `${wgslRel} must be non-empty`).toBeGreaterThan(0);
      expect(wgsl, `${wgslRel} missing @workgroup_size(256, 1, 1)`).toContain(
        "@workgroup_size(256, 1, 1)",
      );
      expect(wgsl, `${wgslRel} contains forbidden 64/128-bit WGSL type`).not.toMatch(
        /\b(?:f64|u64|i64|u128|i128)\b/,
      );
      expect(wgsl, `${wgslRel} contains forbidden continuing / break if`).not.toMatch(
        /\bcontinuing\b|\bbreak\s+if\b/,
      );
    }
  }
}

function collectTreeMatches(
  roots: readonly string[],
  exts: ReadonlySet<string>,
  pattern: RegExp,
  skipRel?: string,
): string[] {
  const hits: string[] = [];
  for (const rootDir of roots) {
    for (const fPath of walkFilesByExtensions(rootDir, exts)) {
      const rel = relPath(fPath);
      if (rel !== skipRel && pattern.test(fs.readFileSync(fPath, "utf8"))) hits.push(rel);
    }
  }
  return hits;
}

describe("DNAM Monorepo Boundaries, Black-Box Purity & Tooling Guardrails", () => {
  it("test_crate_manifest_boundaries_isolation_and_idea_imls", () => {
    checkFixturesManifestStrictness();
    checkCargoAndMoonBoundaries();
    checkSonaShadersArtifacts();
    checkSeparateIdeaImlsForPackages();

    for (const cargoPath of [
      path.join(REPO_ROOT, "Cargo.toml"),
      path.join(REPO_ROOT, "Cargo.lock"),
      ...["kernels", "vm", "engine", "train", "fixtures", "capture", "studio", "js"].map((pkg) =>
        path.join(PACKAGES_DIR, pkg, "Cargo.toml"),
      ),
    ]) {
      const text = fs.readFileSync(cargoPath, "utf8");
      expect(text, `${relPath(cargoPath)} must not depend on ndarray`).not.toContain("ndarray");
      expect(text, `${relPath(cargoPath)} must not depend on dialoguer`).not.toContain("dialoguer");
    }
    for (const pkg of ["kernels", "vm", "engine", "train", "capture", "studio", "js"]) {
      const text = fs.readFileSync(path.join(PACKAGES_DIR, pkg, "Cargo.toml"), "utf8");
      expect(text).not.toContain("realfft");
      expect(text).not.toContain("rustfft");
      expect(text).not.toContain("nalgebra");
    }
    for (const mod of ["metrics.rs", "whitebox.rs", "proj.rs", "align.rs"]) {
      expect(fs.existsSync(path.join(PACKAGES_DIR, "capture", "src", mod))).toBe(true);
    }

    const forbiddenFixturesRe =
      /\b(?:Tier|DistillationTarget|internal_nodes|DnwbHeader64B|DNWB|dnam_engine|dnam_capture|dnam_train|dnam_trainer)\b/;
    const fixturesLeaks: string[] = [];
    for (const rsFile of walkFilesByExtensions(
      path.join(PACKAGES_DIR, "fixtures"),
      new Set([".rs"]),
    )) {
      if (forbiddenFixturesRe.test(stripRustCommentsAndStrings(fs.readFileSync(rsFile, "utf8")))) {
        fixturesLeaks.push(relPath(rsFile));
      }
    }
    expect(fixturesLeaks).toEqual([]);

    const nonTestLeaks: string[] = [];
    const includeFixturesRe =
      /(?:#\s*\[\s*path\s*=|include(?:_str|_bytes)?\s*!)\s*\(?[^;\n]*fixtures/;
    for (const dir of [
      path.join(PACKAGES_DIR, "kernels"),
      path.join(PACKAGES_DIR, "vm"),
      path.join(PACKAGES_DIR, "train"),
      path.join(PACKAGES_DIR, "capture", "src"),
      path.join(PACKAGES_DIR, "engine"),
      path.join(PACKAGES_DIR, "studio"),
      path.join(PACKAGES_DIR, "js"),
    ]) {
      for (const rsFile of walkFilesByExtensions(dir, new Set([".rs"]))) {
        const raw = fs.readFileSync(rsFile, "utf8");
        const noComments = raw.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
        if (
          /\bdnam_fixtures\b/.test(stripRustCommentsAndStrings(raw)) ||
          includeFixturesRe.test(noComments)
        ) {
          nonTestLeaks.push(relPath(rsFile));
        }
      }
    }
    expect(fs.existsSync(path.join(PACKAGES_DIR, "capture", "build.rs"))).toBe(false);
    expect(nonTestLeaks).toEqual([]);
  });

  it("test_zero_retired_tokens_python_or_todo_placeholders", () => {
    const retiredRe = /(?:\.dnam-ckpt|\.dnam-cache|\.dnce\b|\bDNCK\b|\bDNCE\b|\bdnam[-_]trainer\b)/;
    const scanExts = new Set([".md", ".rs", ".ts", ".tsx", ".json", ".toml", ".yml"]);
    expect(
      collectTreeMatches(
        [PACKAGES_DIR, SPECS_DIR],
        scanExts,
        retiredRe,
        "docs/specs/architecture/project_and_documentation_quality.md",
      ),
    ).toEqual([]);
    expect(collectSpecRules().has("qual.companion.mr-iir")).toBe(true);

    const mrStftRe = /\bmr[-_]?stft\b/i;
    const mrStftHits = collectTreeMatches([DOCS_DIR, PACKAGES_DIR], scanExts, mrStftRe);
    for (const rootMd of ["AGENTS.md", "README.md", "CONTRIBUTING.md", "CLA.md"]) {
      const p = path.join(REPO_ROOT, rootMd);
      if (fs.existsSync(p) && mrStftRe.test(fs.readFileSync(p, "utf8")))
        mrStftHits.push(relPath(p));
    }
    expect(mrStftHits).toEqual([]);
    expect(walkFilesByExtensions(REPO_ROOT, new Set([".py"]))).toEqual([]);

    const todoRe = new RegExp(
      `\\b(?:${["TO" + "DO", "FIX" + "ME", "X" + "XX", "HA" + "CK"].join("|")})\\b`,
    );
    const todoViolations: string[] = [];
    const shaderToolDir = path.join(TOOLS_DIR, "sona-shaders");
    for (const root of [DOCS_DIR, PACKAGES_DIR, shaderToolDir, TESTS_DIR, SCRIPTS_DIR]) {
      for (const filePath of walkFilesByExtensions(
        root,
        new Set([".md", ".rs", ".ts", ".tsx", ".mjs"]),
      )) {
        const lines = fs.readFileSync(filePath, "utf8").split("\n");
        lines.forEach((line, idx) => {
          if (
            todoRe.test(line) &&
            !line.includes("`TODO`") &&
            !line.includes("`FIXME`") &&
            !line.includes("todo =")
          ) {
            todoViolations.push(`${relPath(filePath)}:${idx + 1}: ${line.trim().slice(0, 100)}`);
          }
        });
      }
    }
    expect(todoViolations).toEqual([]);
  });

  it("test_license_cla_and_constant_eval_rules", () => {
    const licenseText = fs.readFileSync(path.join(REPO_ROOT, "LICENSE"), "utf8");
    expect(licenseText).toContain("DNAM Source-Available & Branding License 1.0.0");
    expect(licenseText).toContain("Distribution & Hardware/Plugin Linking License");
    expect(fs.existsSync(path.join(REPO_ROOT, "CLA.md"))).toBe(true);
    expect(fs.readFileSync(path.join(REPO_ROOT, "Cargo.toml"), "utf8")).toContain(
      'license = "LicenseRef-DNAM-1.0.0"',
    );
    for (const pkgJsonRel of [
      "package.json",
      "packages/js/package.json",
      "packages/ui/package.json",
      "packages/studio/package.json",
    ]) {
      expect(fs.readFileSync(path.join(REPO_ROOT, pkgJsonRel), "utf8")).toContain(
        '"license": "LicenseRef-DNAM-1.0.0"',
      );
    }

    const constRuleText = fs.readFileSync(
      path.join(REPO_ROOT, ".agents", "rules", "constant-eval-and-magic-numbers.md"),
      "utf8",
    );
    expect(constRuleText).toContain("ref[repo.code.constant-eval-and-derivations]");
    for (const constEvalPath of [
      path.join(PACKAGES_DIR, "vm", "src", "math", "const_eval.rs"),
      path.join(PACKAGES_DIR, "kernels", "src", "simd", "const_eval.rs"),
    ]) {
      const constEvalSrc = fs.readFileSync(constEvalPath, "utf8");
      expect(constEvalSrc, `${relPath(constEvalPath)} missing pub const fn`).toContain(
        "pub const fn",
      );
      expect(constEvalSrc, `${relPath(constEvalPath)} missing compile-time assert`).toContain(
        "const _: () = assert!(",
      );
    }
    expect(
      fs.readFileSync(path.join(PACKAGES_DIR, "capture", "src", "solver", "progress.rs"), "utf8"),
    ).toContain("pub const PROGRESS_WIRE_BYTES: usize = PROGRESS_WIRE_WORDS * 4;");
    expect(
      fs.readFileSync(path.join(PACKAGES_DIR, "ui", "src", "host", "trainer-progress.ts"), "utf8"),
    ).toContain("export const PROGRESS_WIRE_BYTES = PROGRESS_WIRE_WORDS * 4;");
  });

  it("test_capture_trainer_blackbox_purity_and_sona_rules", () => {
    const blackboxRuleText = fs.readFileSync(
      path.join(REPO_ROOT, ".agents", "rules", "universal-black-box.md"),
      "utf8",
    );
    expect(blackboxRuleText).toContain("ref[arch.pillars.universal-black-box]");
    expect(blackboxRuleText).toContain("ref[repo.code.universal-black-box]");

    const forbidden = [
      "dnam_fixtures",
      "dnam-fixtures",
      "build_tube_amp_di_seed",
      "build_triband_entropia_seed",
      "DI_LOAD_LF_POLE",
      "unlock_triband_",
    ];
    const violations: string[] = [];
    for (const rsPath of walkFilesByExtensions(
      path.join(PACKAGES_DIR, "capture", "src"),
      new Set([".rs"]),
    )) {
      const content = fs.readFileSync(rsPath, "utf8");
      for (const tok of forbidden) {
        if (content.includes(tok)) violations.push(`${relPath(rsPath)}: forbidden '${tok}'`);
      }
    }
    expect(violations).toEqual([]);

    const zerocopyRuleText = fs.readFileSync(
      path.join(REPO_ROOT, ".agents", "rules", "zerocopy-wire-structs.md"),
      "utf8",
    );
    expect(zerocopyRuleText).toContain("ref[sdk.rust.zerocopy-wire-structs]");
    expect(zerocopyRuleText).toContain("ref[sdk.rust.safe-rust-and-boundaries]");
    expect(zerocopyRuleText).toContain("ref[repo.code.constant-eval-and-derivations]");

    const sonaSpecPath = path.join(SPECS_DIR, "architecture", "sona_architecture.md");
    expect(fs.existsSync(sonaSpecPath)).toBe(true);
    expect(fs.existsSync(path.join(SPECS_DIR, "architecture", "tora_architecture.md"))).toBe(false);
    expect(fs.readFileSync(sonaSpecPath, "utf8")).toContain(
      "r[arch.tiers.maximize-quality-within-budget]",
    );

    const adrFm = parseFrontmatter(
      fs.readFileSync(
        path.join(ADR_DIR, "0022-sona-state-space-operator-and-webgpu-architecture.md"),
        "utf8",
      ),
    );
    expect(adrFm?.["id"]).toBe("ADR-0022");
    expect(adrFm?.["status"]).toBe("accepted");

    const rules = collectSpecRules();
    for (const reqRule of [
      "arch.tiers.maximize-quality-within-budget",
      "qual.tiers.floors-not-stopping-criteria",
      "sdk.rust.zerocopy-wire-structs",
      "sdk.rust.shared-nostd-kernels",
      "train.pipeline.webgpu-scans-and-ozaki",
    ]) {
      expect(rules.has(reqRule), `Missing r[${reqRule}]`).toBe(true);
    }
  });

  it("test_rust_coverage_and_parallel_test_use_shared_pnpm_scripts", () => {
    for (const scriptName of ["rust-coverage.mjs", "rust-parallel-test.mjs"]) {
      const p = path.join(SCRIPTS_DIR, scriptName);
      expect(fs.existsSync(p), `Missing scripts/${scriptName}`).toBe(true);
      const body = fs.readFileSync(p, "utf8");
      expect(body.startsWith("// @ts-check\n")).toBe(true);
      expect(body.split(/\r?\n/).length).toBeLessThan(900);
    }

    const rootPkg = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "package.json"), "utf8")) as {
      scripts?: Record<string, string>;
    };
    expect(rootPkg.scripts?.["rust-coverage"]).toBe("node scripts/rust-coverage.mjs");
    expect(rootPkg.scripts?.["rust-test"]).toBe("node scripts/rust-parallel-test.mjs");

    for (const spec of [
      { pkg: "kernels", crate: "dnam-kernels", sources: "packages/kernels/src" },
      { pkg: "vm", crate: "dnam-vm", sources: "packages/vm/src" },
      { pkg: "train", crate: "dnam-train", sources: "packages/train/src" },
      { pkg: "engine", crate: "dnam-engine", sources: "packages/engine/src" },
      { pkg: "fixtures", crate: "dnam-fixtures", sources: "packages/fixtures/src" },
      { pkg: "capture", crate: "dnam-capture", sources: "packages/capture/src" },
      { pkg: "studio", crate: "dnam-studio", sources: "packages/studio/rust" },
    ]) {
      const body = fs.readFileSync(path.join(PACKAGES_DIR, spec.pkg, "moon.yml"), "utf8");
      expect(body).toContain('"rust-coverage"');
      expect(body).toContain(`"${spec.crate}"`);
      expect(body).toContain(`"${spec.sources}"`);
      expect(body).toContain('"/scripts/rust-coverage.mjs"');
    }

    const captureMoonBody = fs.readFileSync(path.join(PACKAGES_DIR, "capture", "moon.yml"), "utf8");
    expect(captureMoonBody).toContain('"rust-test"');
    expect(captureMoonBody).toContain('"/scripts/rust-parallel-test.mjs"');

    const moonFiles = [
      path.join(TESTS_DIR, "moon.yml"),
      ...fs
        .readdirSync(PACKAGES_DIR, { withFileTypes: true })
        .filter((e) => e.isDirectory())
        .map((e) => path.join(PACKAGES_DIR, e.name, "moon.yml")),
      ...fs
        .readdirSync(TOOLS_DIR, { withFileTypes: true })
        .filter((e) => e.isDirectory())
        .map((e) => path.join(TOOLS_DIR, e.name, "moon.yml")),
    ];
    for (const mf of moonFiles) {
      if (fs.existsSync(mf)) {
        expect(fs.readFileSync(mf, "utf8"), `${relPath(mf)} embeds python`).not.toMatch(
          /\bpython3?\b/,
        );
      }
    }
  });
});
