/**
 * Documentation Registry, Frontmatter, Requirements, ADRs, Spec Rule Identifiers,
 * and Traceability Guardrail Tests.
 *
 * verify[repo.docs.requirements-layer]
 * verify[repo.docs.spec-rules-and-adrs]
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
  ADR_DIR,
  ADR_TOKEN_RE,
  collectSourceRefTokens,
  collectSpecRules,
  collectTestVerifyTokens,
  computeRustCfgTestLineMask,
  DOCS_DIR,
  HEADING_RE,
  isGuardrailFile,
  MALFORMED_SPEC_MARKER_RE,
  MD_LINK_RE,
  normalizeSpecRelPath,
  PACKAGES_DIR,
  parseFrontmatter,
  parseSpecScopeToml,
  REPO_ROOT,
  REQ_FILE,
  REQ_TABLE_RE,
  REQ_TOKEN_RE,
  relPath,
  SPEC_RULE_DEF_RE,
  SPEC_VERIFY_TOKEN_RE,
  SPECS_DIR,
  type SpecScopeRuleEntry,
  stripCode,
  stripFencedCodeBlocks,
  stripRustCommentsAndStrings,
  stripRustCommentsOnly,
  TESTS_DIR,
  VALID_RULE_ID_RE,
  walkFilesByExtensions,
  walkMarkdownFiles,
} from "./test-support.js";

interface AdrRegistryMaps {
  seenAdrIds: Map<string, string>;
  adrStatuses: Map<string, string>;
  adrSupersededBy: Map<string, string>;
  adrFilenames: Map<string, string>;
}

function validateAdrFrontmatterFile(adrPath: string, maps: AdrRegistryMaps): void {
  const requiredKeys = "id title status supersedes superseded_by specs requirements".split(" ");
  const fm = parseFrontmatter(fs.readFileSync(adrPath, "utf8"));
  expect(fm, `${relPath(adrPath)} is missing YAML frontmatter`).not.toBeNull();
  if (!fm) return;
  for (const key of requiredKeys) {
    expect(key in fm, `${relPath(adrPath)} frontmatter is missing '${key}'`).toBe(true);
  }
  const adrId = String(fm["id"]);
  expect(adrId).toMatch(/^ADR-\d{4}$/);
  expect(path.basename(adrPath).startsWith(adrId.slice(4))).toBe(true);
  expect(maps.seenAdrIds.has(adrId)).toBe(false);
  maps.seenAdrIds.set(adrId, relPath(adrPath));
  maps.adrFilenames.set(adrId, path.basename(adrPath));
  const status = String(fm["status"]);
  maps.adrStatuses.set(adrId, status);
  expect(["proposed", "accepted", "superseded"].includes(status)).toBe(true);
  expect(Array.isArray(fm["specs"])).toBe(true);
  expect(Array.isArray(fm["requirements"])).toBe(true);
  if (status === "accepted") expect((fm["specs"] as string[]).length).toBeGreaterThan(0);
  if (status === "superseded") {
    const supBy = String(fm["superseded_by"]);
    expect(supBy).toMatch(/^ADR-\d{4}$/);
    maps.adrSupersededBy.set(adrId, supBy);
  }
}

function verifySupersededAdrsNotCited(
  supersededIds: readonly string[],
  adrFilenames: ReadonlyMap<string, string>,
): void {
  for (const docFile of [REQ_FILE, ...walkMarkdownFiles(SPECS_DIR)]) {
    const rawText = fs.readFileSync(docFile, "utf8");
    const cleanBody = stripCode(rawText);
    const fm = parseFrontmatter(rawText);
    const fmAdrs = Array.isArray(fm?.["adrs"]) ? fm["adrs"] : [];
    for (const supId of supersededIds) {
      const supFilename = adrFilenames.get(supId) ?? "";
      expect(fmAdrs.includes(supId), `${relPath(docFile)} cites superseded ${supId}`).toBe(false);
      expect(
        cleanBody.includes(supId) || (supFilename !== "" && rawText.includes(supFilename)),
        `${relPath(docFile)} cites superseded ${supId} (${supFilename})`,
      ).toBe(false);
    }
  }
}

const STANDALONE_SPEC_RULE_LINE_RE = /^\s*`r\[[a-z0-9.-]+\]`\s*$/;
const LEGACY_SPEC_STAGE_RE =
  /\b(?:Stage\s*(?:0|I{1,3}|IV|[0-4]|[A-E])(?:[–-](?:IV|[0-4]|[A-E]))?|Stages\s*(?:0|I{1,3}|IV|[0-4]|[A-E])[–-](?:IV|[0-4]|[A-E])|Step[-\s]?0|Four-Stage|Five-Stage|Gate\s+0b|Refinement\s+\d+)\b/i;
const RETIRED_SPEC_TOKEN_RE =
  /\b(?:rate_1x_cal|V3_META_[A-Z0-9_]+|V3_TANK_[A-Z0-9_]+|stage2_gru|V3PostStageConditioner|Stage[0-4][A-Za-z0-9_]*|stage[0-4]_[A-Za-z0-9_]*|s[0-4]_[a-z0-9_]*|eval_stage[0-4]_[A-Za-z0-9_]*|STAGE[0-4]_[A-Z0-9_]+|PIPE_FLAG_STAGE[0-4]_[A-Z0-9_]+|HB_STAGE[12]_TAPS|V3Stage[0-9A-Za-z_]*|Op[MNT]Backend|fir_rev_history|specialty\.rs|src\/stages\b|DNAM-GRAM-INPUT-v2|Windowed-BPTT|Live16Tap|IntegerDelayOnly)\b/;
const SPEC_CHAIN_ID_RE = /\bChain-\d+\b/;
const CHAIN_SPEC_ALLOWED_REL = "docs/specs/training/webgpu_and_varpro_solver.md";

function collectSpecHeadingHierarchyViolations(
  rel: string,
  clean: string,
  violations: string[],
): void {
  const headings = Array.from(clean.matchAll(HEADING_RE));
  if (headings.length === 0) {
    violations.push(`${rel}: spec has no Markdown headings`);
    return;
  }
  let prevLevel = 0;
  for (let idx = 0; idx < headings.length; idx++) {
    const match = headings[idx];
    if (!match) continue;
    const level = (match[1] ?? "").length;
    const title = (match[2] ?? "").trim();
    const lineNo = clean.slice(0, match.index ?? 0).split("\n").length;
    if (idx === 0) {
      if (level !== 1) {
        violations.push(`${rel}:${lineNo}: first heading must be H1 '#', found H${level}`);
      }
      prevLevel = level;
      continue;
    }
    if (level === 1) {
      violations.push(`${rel}:${lineNo}: duplicate H1 '#' heading '${title}'`);
    } else if (idx === 1 && level !== 2) {
      violations.push(
        `${rel}:${lineNo}: first section heading after H1 must be H2 '##', found H${level}`,
      );
    } else if (level >= 5) {
      violations.push(`${rel}:${lineNo}: H${level} heading '${title}' exceeds H4 ceiling`);
    } else if (level > prevLevel + 1) {
      violations.push(
        `${rel}:${lineNo}: skipped heading level H${prevLevel} -> H${level} '${title}'`,
      );
    }
    prevLevel = level;
  }
}

function collectNonSpecDocCleanlinessViolations(filePath: string, violations: string[]): void {
  const rel = relPath(filePath);
  const rawLines = fs.readFileSync(filePath, "utf8").split("\n");
  for (let idx = 0; idx < rawLines.length; idx++) {
    const rawLine = rawLines[idx] ?? "";
    const stageMatch = LEGACY_SPEC_STAGE_RE.exec(rawLine);
    if (stageMatch) {
      violations.push(`${rel}:${idx + 1}: legacy stage label '${stageMatch[0]}'`);
    }
    const retiredMatch = RETIRED_SPEC_TOKEN_RE.exec(rawLine);
    if (retiredMatch) {
      violations.push(`${rel}:${idx + 1}: retired token '${retiredMatch[0]}'`);
    }
  }
}

function collectSpecCleanlinessViolations(specPath: string, violations: string[]): void {
  const rel = relPath(specPath);
  const rawText = fs.readFileSync(specPath, "utf8");
  const clean = stripFencedCodeBlocks(rawText);
  collectSpecHeadingHierarchyViolations(rel, clean, violations);

  const cleanLines = clean.split("\n");
  for (let idx = 0; idx < cleanLines.length; idx++) {
    const line = cleanLines[idx] ?? "";
    if (line.match(SPEC_RULE_DEF_RE) !== null && !STANDALONE_SPEC_RULE_LINE_RE.test(line)) {
      violations.push(`${rel}:${idx + 1}: r[...] definition must stand alone on its own line`);
    }
  }

  const rawLines = rawText.split("\n");
  for (let idx = 0; idx < rawLines.length; idx++) {
    const rawLine = rawLines[idx] ?? "";
    const stageMatch = LEGACY_SPEC_STAGE_RE.exec(rawLine);
    if (stageMatch) {
      violations.push(`${rel}:${idx + 1}: legacy stage label '${stageMatch[0]}'`);
    }
    const retiredMatch = RETIRED_SPEC_TOKEN_RE.exec(rawLine);
    if (retiredMatch) {
      violations.push(`${rel}:${idx + 1}: retired token '${retiredMatch[0]}'`);
    }
    if (rel !== CHAIN_SPEC_ALLOWED_REL) {
      const chainMatch = SPEC_CHAIN_ID_RE.exec(rawLine);
      if (chainMatch) {
        violations.push(
          `${rel}:${idx + 1}: fixture ID '${chainMatch[0]}' belongs only in ${CHAIN_SPEC_ALLOWED_REL}`,
        );
      }
    }
  }
}

function checkSpecHeadingSections(specPath: string, clean: string, missing: string[]): void {
  const headings = Array.from(clean.matchAll(/^(#{2,4})\s+(.+?)\s*$/gm));
  for (let idx = 0; idx < headings.length; idx++) {
    const h = headings[idx];
    if (!h || h.index === undefined) continue;
    const level = (h[1] ?? "").length;
    const title = (h[2] ?? "").trim();
    const start = h.index + h[0].length;
    const nextH = headings[idx + 1];
    const end = nextH && nextH.index !== undefined ? nextH.index : clean.length;
    const nextLevel = nextH ? (nextH[1] ?? "").length : 0;
    const hasRule = SPEC_RULE_DEF_RE.test(clean.slice(start, end));
    SPEC_RULE_DEF_RE.lastIndex = 0;
    if (!(idx + 1 < headings.length && nextLevel > level) && !hasRule) {
      missing.push(`${relPath(specPath)}: leaf '${"#".repeat(level)} ${title}' has no r[...] rule`);
    }
  }
}

function verifySpecFileRulePrefixesAndLeafSections(specPath: string, missing: string[]): void {
  const text = fs.readFileSync(specPath, "utf8");
  const fm = parseFrontmatter(text);
  const prefixes = Array.isArray(fm?.["rule_prefixes"]) ? fm["rule_prefixes"] : [];
  const clean = stripFencedCodeBlocks(text);
  const fileRules = Array.from(clean.matchAll(SPEC_RULE_DEF_RE), (m) => m[1] ?? m[2] ?? "");
  expect(fileRules.length, `${relPath(specPath)} defines zero r[...] rules`).toBeGreaterThan(0);
  for (const ruleId of fileRules) {
    const fullToken = `r[${ruleId}]`;
    expect(
      prefixes.some((prefix) => fullToken.startsWith(prefix)),
      `${relPath(specPath)}: '${fullToken}' does not match frontmatter rule_prefixes`,
    ).toBe(true);
  }
  checkSpecHeadingSections(specPath, clean, missing);
}

function collectStrayRulesOutsideSpecs(templateRel: string): string[] {
  const stray: string[] = [];
  for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
    const rel = relPath(mdPath);
    if (
      mdPath.startsWith(SPECS_DIR) ||
      rel.startsWith(templateRel) ||
      rel === "docs/CONTRIBUTING.md"
    ) {
      continue;
    }
    const lines = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8")).split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      for (const m of (lines[idx] ?? "").matchAll(SPEC_RULE_DEF_RE)) {
        stray.push(`${rel}:${idx + 1} defines r[${m[1] ?? m[2] ?? ""}] outside docs/specs/`);
      }
    }
  }
  return stray;
}

function collectMalformedDocMarkers(templateRel: string): string[] {
  const malformed: string[] = [];
  for (const mdPath of walkMarkdownFiles(DOCS_DIR)) {
    const rel = relPath(mdPath);
    if (rel.startsWith(templateRel) || rel === "docs/CONTRIBUTING.md") continue;
    const lines = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8")).split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      for (const m of (lines[idx] ?? "").matchAll(MALFORMED_SPEC_MARKER_RE)) {
        const kind = m[1] ?? "";
        const inner = m[2] ?? "";
        const isPlaceholder =
          (inner.endsWith(".") && kind === "r") ||
          inner.includes("...") ||
          inner.includes("*") ||
          inner.includes("<") ||
          inner.includes(">");
        if (!isPlaceholder && !VALID_RULE_ID_RE.test(inner)) {
          malformed.push(`${rel}:${idx + 1}: malformed '${kind}[${inner}]'`);
        }
      }
    }
  }
  return malformed;
}

function collectVerifyTokensInDocs(templateRel: string): string[] {
  const verifyInDocs: string[] = [];
  for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
    const rel = relPath(mdPath);
    if (
      rel.startsWith(templateRel) ||
      rel === "docs/CONTRIBUTING.md" ||
      rel === "docs/specs/architecture/project_and_documentation_quality.md"
    ) {
      continue;
    }
    const lines = fs.readFileSync(mdPath, "utf8").split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      for (const m of (lines[idx] ?? "").matchAll(SPEC_VERIFY_TOKEN_RE)) {
        verifyInDocs.push(`${rel}:${idx + 1}: verify[${m[1] ?? ""}]`);
      }
    }
  }
  return verifyInDocs;
}

function collectLegacyTokenHits(): string[] {
  const hits: string[] = [];
  const legacyClRe = new RegExp(`\\b${"CL"}-\\d{2}\\b`);
  const legacyGRe = new RegExp(`\\b${"G-0"}\\d\\b`);
  for (const filePath of [REQ_FILE, ...walkMarkdownFiles(SPECS_DIR)]) {
    const lines = fs.readFileSync(filePath, "utf8").split("\n");
    lines.forEach((line, idx) => {
      const m = legacyClRe.exec(line);
      if (m) hits.push(`${relPath(filePath)}:${idx + 1}: ${m[0]}`);
    });
  }
  for (const root of [DOCS_DIR, PACKAGES_DIR, TESTS_DIR]) {
    for (const filePath of walkFilesByExtensions(root, new Set([".md", ".rs", ".ts", ".tsx"]))) {
      const lines = fs.readFileSync(filePath, "utf8").split("\n");
      lines.forEach((line, idx) => {
        const m = legacyGRe.exec(line);
        if (m) hits.push(`${relPath(filePath)}:${idx + 1}: ${m[0]}`);
      });
    }
  }
  return hits;
}

function collectDocToTestLinks(): string[] {
  const links: string[] = [];
  for (const mdPath of walkMarkdownFiles(DOCS_DIR)) {
    const rel = relPath(mdPath);
    if (!mdPath.startsWith(SPECS_DIR) && !mdPath.startsWith(ADR_DIR) && mdPath !== REQ_FILE) {
      continue;
    }
    const lines = stripCode(fs.readFileSync(mdPath, "utf8")).split("\n");
    lines.forEach((line, idx) => {
      for (const m of line.matchAll(MD_LINK_RE)) {
        const target = m[2] ?? "";
        if (
          target.includes("test_") ||
          target.includes("_spec_references") ||
          target.includes("_verification") ||
          target.includes("/tests/")
        ) {
          links.push(`${rel}:${idx + 1} -> ${target}`);
        }
      }
    });
  }
  return links;
}

function isStandaloneTestSourcePath(rel: string): boolean {
  const norm = rel.split(path.sep).join("/");
  return (
    /(?:^|\/)tests\//.test(norm) ||
    /\.(?:test|spec)\.tsx?$/.test(norm) ||
    /(?:^|\/)(?:test_support|test_utils|testing)\.(?:rs|ts)$/.test(norm)
  );
}

function checkRustCfgTestSpecParse(
  rel: string,
  src: string,
  patterns: readonly RegExp[],
  violations: string[],
): void {
  const braceLines = stripRustCommentsAndStrings(src).split("\n");
  const codeLines = stripRustCommentsOnly(src).split("\n");
  const cfgMask = computeRustCfgTestLineMask(braceLines);
  const rawLines = src.split("\n");
  for (let idx = 0; idx < codeLines.length; idx++) {
    if (!cfgMask[idx]) continue;
    const codeLine = codeLines[idx] ?? "";
    if (patterns.some((pat) => pat.test(codeLine))) {
      violations.push(`${rel}:${idx + 1}: ${(rawLines[idx] ?? "").trim()}`);
    }
  }
}

function scanStandaloneTsTestFile(
  rel: string,
  src: string,
  patterns: readonly RegExp[],
  violations: string[],
): void {
  const lines = src.split("\n");
  lines.forEach((line, idx) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("//") || trimmed.startsWith("*") || trimmed.startsWith("/*")) {
      return;
    }
    if (patterns.some((pat) => pat.test(line))) {
      violations.push(`${rel}:${idx + 1}: ${trimmed}`);
    }
  });
}

function scanStandaloneRsTestFile(
  rel: string,
  src: string,
  patterns: readonly RegExp[],
  violations: string[],
): void {
  const rawLines = src.split("\n");
  const codeLines = stripRustCommentsOnly(src).split("\n");
  codeLines.forEach((codeLine, idx) => {
    if (patterns.some((pat) => pat.test(codeLine))) {
      violations.push(`${rel}:${idx + 1}: ${(rawLines[idx] ?? "").trim()}`);
    }
  });
}

function collectSpecTextParseViolations(): string[] {
  const patterns = [
    new RegExp(`${"docs/specs"}/`),
    new RegExp(`${"docs/adr"}/`),
    new RegExp(`${"docs/requirements"}/`),
  ];
  const violations: string[] = [];
  for (const root of [TESTS_DIR, PACKAGES_DIR]) {
    for (const filePath of walkFilesByExtensions(root, new Set([".rs", ".ts", ".tsx"]))) {
      const rel = relPath(filePath);
      if (isGuardrailFile(filePath)) continue;
      const src = fs.readFileSync(filePath, "utf8");
      if (!isStandaloneTestSourcePath(rel)) {
        if (filePath.endsWith(".rs")) {
          checkRustCfgTestSpecParse(rel, src, patterns, violations);
        }
        continue;
      }
      if (filePath.endsWith(".rs")) {
        scanStandaloneRsTestFile(rel, src, patterns, violations);
      } else {
        scanStandaloneTsTestFile(rel, src, patterns, violations);
      }
    }
  }
  return violations;
}

function collectAdrStatusAndSpecMaps(): {
  adrStatuses: Map<string, string>;
  adrEnforcesSpecs: Map<string, string[]>;
} {
  const adrStatuses = new Map<string, string>();
  const adrEnforcesSpecs = new Map<string, string[]>();
  for (const entry of fs.readdirSync(ADR_DIR).sort()) {
    if (!entry.endsWith(".md") || entry === "README.md") continue;
    const fm = parseFrontmatter(fs.readFileSync(path.join(ADR_DIR, entry), "utf8"));
    if (!fm) continue;
    const aid = String(fm["id"]);
    const status = String(fm["status"]);
    expect(/^\d{4}-superseded-/.test(entry), `ADR filename '${entry}' vs status '${status}'`).toBe(
      status === "superseded",
    );
    adrStatuses.set(aid, status);
    const rawSpecs = Array.isArray(fm["specs"]) ? fm["specs"] : [];
    adrEnforcesSpecs.set(aid, rawSpecs.map(normalizeSpecRelPath));
  }
  return { adrStatuses, adrEnforcesSpecs };
}

function collectSpecCitationSets(supersededAdrs: ReadonlySet<string>): {
  specCitedReqs: Set<string>;
  specCitedAdrs: Set<string>;
  specFileSet: Set<string>;
} {
  const specCitedReqs = new Set<string>();
  const specCitedAdrs = new Set<string>();
  const specFileSet = new Set<string>();
  for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
    specFileSet.add(path.relative(DOCS_DIR, specPath).split(path.sep).join("/"));
    const text = fs.readFileSync(specPath, "utf8");
    const fm = parseFrontmatter(text);
    const reqs = fm?.["requirements"];
    if (Array.isArray(reqs)) for (const r of reqs) specCitedReqs.add(r);
    for (const m of text.matchAll(REQ_TOKEN_RE)) specCitedReqs.add(m[0]);
    const adrs = fm?.["adrs"];
    if (Array.isArray(adrs)) {
      for (const a of adrs) {
        expect(supersededAdrs.has(a), `${relPath(specPath)} cites superseded ${a}`).toBe(false);
        specCitedAdrs.add(a);
      }
    }
    for (const m of text.matchAll(ADR_TOKEN_RE)) specCitedAdrs.add(m[0]);
  }
  return { specCitedReqs, specCitedAdrs, specFileSet };
}

const RUST_INTERNAL_SYM_ALT = [
  "SonaHotSlab|SonaInstance|SonaVm|DnamEngine|DynamicDnamEngine|DnamHostHandle|KernelVec4|SimdVec4",
  "GpuVec4|StereoVec8|WasmSimd128Backend|WasmSimd128TrainerBackend|DnamJsonHeader",
  "DnamProjManifest|hoist_groups_and_readouts|process_tick16_const|step_frame_sona",
  "step_frame_4x_sona|step_frame_4port_sona|step_frame_4x_4port_sona|step_control_tick_3khz",
  "strip_container_to_tier|rcp_pos_quad_nr2|f32x4_to_fp16x4|fp16x4_to_f32x4|flush_subnormal",
].join("|");
const RUST_ATTR_RE_SRC = [
  "#\\s*!?\\s*\\[(?:forbid|deny|allow|warn|expect|cfg|cfg_attr|repr|derive",
  "|inline|must_use|unsafe|napi|path|no_std)\\b",
].join("");
const NON_SDK_SPEC_FORBIDDEN_RE = new RegExp(
  [
    "\\b(?:packages|tools|scripts)/",
    "(?:\\.\\./)+tests/|\\btests/(?:[A-Za-z0-9_*.-]+\\.(?:rs|ts|tsx|mjs|js))\\b",
    "\\btests/(?:integration|profile|e2e|setup|\\*)|`tests/",
    "\\.agents/|(?:\\b[A-Za-z0-9_-]+|\\*)\\.(?:rs|ts|tsx|mjs|cjs|toml)\\b",
    "\\b(?:package\\.json|tsconfig\\.json|biome\\.json|moon\\.yml|Cargo\\.lock)\\b",
    RUST_ATTR_RE_SRC,
    "\\b(?:proptest|macro_rules|include_str|include_bytes)!",
    "\\b(?:Result<|Vec<|Box<|Arc<|BTreeMap|BTreeSet|HashMap|HashSet|AtomicU64|AtomicUsize)\\b",
    "\\b(?:Float32Array|Float64Array|Uint8Array|Int32Array|SharedArrayBuffer)\\b",
    "\\b(?:dnam-kernels|dnam_kernels|dnam-vm|dnam_vm|dnam-train|dnam_train|dnam-engine|dnam_engine|dnam-capture|dnam_capture)\\b",
    "\\b(?:dnam-fixtures|dnam_fixtures|dnam-js|dnam_js|dnam-studio|dnam_studio|dnam-ui)\\b",
    "@dnam-project/[a-z-]+\\b|\\b(?:Rust|TypeScript|JavaScript|Biome|Vitest|Playwright|N-API)\\b",
    "\\b(?:AudioWorklet|Cargo|rustc|rustfmt|clippy|fast-check|zerocopy|serde|serde_json)\\b",
    "\\b(?:cargo-gpu|rlib|dnam\\.wasm)\\b|\\b[A-Z][A-Za-z0-9_]*Wire\\b",
    `\\b(?:${RUST_INTERNAL_SYM_ALT}|FtzDazGuard|TelemetryReader|ControlConsumer)\\b`,
    "\\b(?:FromBytes|IntoBytes|KnownLayout|from_le_bytes|to_le_bytes)\\b|`(?:Immutable|Unaligned)`",
    "\\b(?:r|ref|verify)\\[sdk\\.|\\b(?:rust_sdk|typescript_sdk)\\.md\\b",
  ].join("|"),
);
const TS_SDK_RUST_FORBIDDEN_RE = new RegExp(
  [
    "(?:\\b[A-Za-z0-9_-]+|\\*)\\.rs\\b|\\[profile\\.[a-z]+\\]|\\bCargo\\.(?:toml|lock)\\b",
    "\\bpackages/(?:engine|kernels|vm|train|capture|fixtures)\\b|\\bpackages/(?:js|studio)/rust\\b",
    "\\btools/sona-shaders\\b",
    RUST_ATTR_RE_SRC,
    "\\b(?:dnam-kernels|dnam_kernels|dnam-vm|dnam_vm|dnam-train|dnam_train|dnam-engine|dnam_engine|dnam-capture|dnam_capture)\\b",
    "\\b(?:dnam-fixtures|dnam_fixtures|dnam-js|dnam_js|serde|serde_json|zerocopy|rustc)\\b",
    `\\b(?:cargo-gpu|proptest!|macro_rules!|ControlBus|${RUST_INTERNAL_SYM_ALT})\\b`,
  ].join("|"),
);
const RUST_PATH_COLON_RE = /\b[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_]/;

function collectSpecLanguageLeaks(
  specPath: string,
  forbiddenRe: RegExp,
  isTsSdk: boolean,
  violations: string[],
): void {
  const rel = relPath(specPath);
  const lines = fs.readFileSync(specPath, "utf8").split("\n");
  for (let idx = 0; idx < lines.length; idx++) {
    let cleaned = (lines[idx] ?? "").replace(/(?:https?|mailto):\S+/g, "");
    cleaned = isTsSdk
      ? cleaned.replace(/\bref\[sdk\.rust\.[a-z0-9.-]+\]/g, "")
      : cleaned.replace(/\b(?:r|ref|verify)\[(?!sdk\.)[a-z0-9.-]+\]/g, "");
    const hit = forbiddenRe.exec(cleaned) ?? RUST_PATH_COLON_RE.exec(cleaned);
    if (hit) violations.push(`${rel}:${idx + 1}: forbidden token '${hit[0]}'`);
  }
}

function checkEngineScopeEntry(
  ruleId: string,
  entry: SpecScopeRuleEntry,
  actualVerifies: ReadonlyMap<string, Set<string>>,
  errors: string[],
): void {
  if (entry.scope !== "engine" && entry.scope !== "out-of-scope") {
    errors.push(`r[${ruleId}]: invalid scope '${entry.scope}'`);
    return;
  }
  if (entry.scope === "out-of-scope") {
    if (!(entry.reason ?? "").trim()) {
      errors.push(`r[${ruleId}]: scope='out-of-scope' requires non-empty reason`);
    }
    const found = actualVerifies.get(ruleId);
    if (found) {
      errors.push(
        `r[${ruleId}]: out-of-scope but verified in engine: ${Array.from(found).sort().join(", ")}`,
      );
    }
    return;
  }
  if ((actualVerifies.get(ruleId)?.size ?? 0) === 0) {
    errors.push(`r[${ruleId}]: scope='engine', but no verify[${ruleId}] in packages/engine`);
  }
}

describe("DNAM Documentation Registry, Rules, ADRs & Traceability Guardrails", () => {
  it("test_three_layer_docs_directory_structure", () => {
    expect(fs.existsSync(path.join(DOCS_DIR, "index.md"))).toBe(true);
    expect(fs.existsSync(path.join(DOCS_DIR, "CONTRIBUTING.md"))).toBe(true);
    expect(fs.existsSync(REQ_FILE)).toBe(true);
    expect(fs.existsSync(ADR_DIR) && fs.statSync(ADR_DIR).isDirectory()).toBe(true);
    expect(fs.existsSync(SPECS_DIR) && fs.statSync(SPECS_DIR).isDirectory()).toBe(true);
    expect(fs.existsSync(path.join(DOCS_DIR, "verification"))).toBe(false);

    const templatesDir = path.join(DOCS_DIR, "templates");
    expect(fs.existsSync(templatesDir) && fs.statSync(templatesDir).isDirectory()).toBe(true);
    for (const tplName of ["requirement_template.md", "spec_template.md", "adr_template.md"]) {
      const tplPath = path.join(templatesDir, tplName);
      expect(fs.existsSync(tplPath), `Missing docs/templates/${tplName}`).toBe(true);
      expect(parseFrontmatter(fs.readFileSync(tplPath, "utf8"))).not.toBeNull();
    }

    const actualDomains = new Set(
      fs
        .readdirSync(SPECS_DIR, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name),
    );
    expect(actualDomains).toEqual(
      new Set(["architecture", "dsp", "containers", "training", "security", "sdks"]),
    );
  });

  it("test_all_spec_files_have_valid_yaml_frontmatter", () => {
    const requiredKeys = "id title status version domain rule_prefixes requirements adrs".split(
      " ",
    );
    const specFiles = walkMarkdownFiles(SPECS_DIR);
    expect(specFiles.length).toBe(18);

    const validAdrIds = new Set(
      fs
        .readdirSync(ADR_DIR)
        .filter((n) => n.endsWith(".md") && n !== "README.md")
        .map((n) => parseFrontmatter(fs.readFileSync(path.join(ADR_DIR, n), "utf8"))?.["id"])
        .filter((id): id is string => typeof id === "string"),
    );

    const seenSpecIds = new Map<string, string>();
    for (const specPath of specFiles) {
      const fm = parseFrontmatter(fs.readFileSync(specPath, "utf8"));
      expect(fm, `${relPath(specPath)} is missing YAML frontmatter`).not.toBeNull();
      if (!fm) continue;
      expect("clamps" in fm, `${relPath(specPath)} has legacy 'clamps' key`).toBe(false);
      for (const key of requiredKeys) {
        expect(
          key in fm && fm[key] !== "" && (Array.isArray(fm[key]) ? fm[key].length > 0 : true),
          `${relPath(specPath)} missing '${key}'`,
        ).toBe(true);
      }
      const specId = String(fm["id"]);
      expect(specId.startsWith("spec.")).toBe(true);
      expect(seenSpecIds.has(specId)).toBe(false);
      seenSpecIds.set(specId, relPath(specPath));
      expect(fm["status"]).toBe("canonical");
      expect(Array.isArray(fm["rule_prefixes"])).toBe(true);
      expect(Array.isArray(fm["requirements"])).toBe(true);
      expect(Array.isArray(fm["adrs"])).toBe(true);
      if (Array.isArray(fm["adrs"])) {
        for (const adrRef of fm["adrs"]) {
          expect(validAdrIds.has(adrRef), `${relPath(specPath)} cites unknown ${adrRef}`).toBe(
            true,
          );
        }
      }
    }
  });

  it("test_all_adr_files_have_valid_yaml_frontmatter", () => {
    const adrFiles = fs
      .readdirSync(ADR_DIR)
      .filter((n) => n.endsWith(".md") && n !== "README.md")
      .sort()
      .map((n) => path.join(ADR_DIR, n));
    expect(adrFiles.length).toBeGreaterThanOrEqual(18);

    const maps: AdrRegistryMaps = {
      seenAdrIds: new Map(),
      adrStatuses: new Map(),
      adrSupersededBy: new Map(),
      adrFilenames: new Map(),
    };
    for (const adrPath of adrFiles) validateAdrFrontmatterFile(adrPath, maps);

    for (const [adrId, supBy] of maps.adrSupersededBy.entries()) {
      expect(maps.adrStatuses.has(supBy), `${adrId} superseded_by '${supBy}' missing`).toBe(true);
      expect(maps.adrStatuses.get(supBy) !== "superseded").toBe(true);
    }

    const supersededIds = Array.from(maps.adrStatuses.entries())
      .filter(([, st]) => st === "superseded")
      .map(([aid]) => aid);
    verifySupersededAdrsNotCited(supersededIds, maps.adrFilenames);
  });

  it("test_requirement_registry_and_agnostic_prose", () => {
    const text = fs.readFileSync(REQ_FILE, "utf8");
    const fm = parseFrontmatter(text);
    expect(fm?.["id"]).toBe("req.system-goals");
    expect(fm?.["status"]).toBe("canonical");
    expect(fm?.["layer"]).toBe("requirements");

    const reqIds = Array.from(text.matchAll(REQ_TABLE_RE), (m) => m[1] ?? "");
    expect(reqIds.length).toBe(41);
    expect(reqIds.length).toBe(new Set(reqIds).size);

    const expectedCounts = [9, 5, 3, 3, 6, 3, 4, 8];
    for (let g = 0; g < expectedCounts.length; g++) {
      const count = expectedCounts[g] ?? 0;
      for (let sub = 1; sub <= count; sub++) expect(reqIds).toContain(`REQ-${g + 1}.${sub}`);
    }

    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const reqList = parseFrontmatter(fs.readFileSync(specPath, "utf8"))?.["requirements"];
      if (Array.isArray(reqList)) {
        for (const reqRef of reqList) {
          expect(reqIds.includes(reqRef), `${relPath(specPath)} unknown req '${reqRef}'`).toBe(
            true,
          );
        }
      }
    }

    const proseOnly = stripFencedCodeBlocks(text).replace(/\]\([^)]+\)/g, "]()");
    const forbiddenReqTokens = [
      "max_supported_tier|user_metadata|royalty_splits|allowed_protection_profiles",
      "allowed_oem_vendors|output head|output-head|SIMD lane|Hann-windowed|JSON header",
      ".dnam|.dnam-proj|.dnam-ckpt|.dnam-cache|.dnam-seal|.dnce|mono_only|CL-0|CL-1|CL-2",
      "WebGPU|IEEE-754|float64|f32|f64|fp16|i32|u32|0-ULP|48 kHz MACs|../specs/",
    ]
      .join("|")
      .split("|");
    for (const field of forbiddenReqTokens) expect(proseOnly).not.toContain(field);
  });

  it("test_layer_separation_no_adr_narratives_in_canonical_specs", () => {
    const forbidden = [
      "rejected alternatives",
      "counterexamples and failure modes",
      "why alternatives lose",
      "candidate comparison",
    ];
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const clean = stripFencedCodeBlocks(fs.readFileSync(specPath, "utf8"));
      for (const match of clean.matchAll(HEADING_RE)) {
        const headingText = (match[2] ?? "").toLowerCase();
        for (const phrase of forbidden) expect(headingText).not.toContain(phrase);
      }
    }
  });

  it("test_spec_rule_identifiers_adr_structure_and_no_spec_text_parsing", () => {
    const rules = collectSpecRules();
    expect(rules.size).toBeGreaterThanOrEqual(150);
    expect(
      Array.from(rules.entries())
        .filter(([, locs]) => locs.length > 1)
        .map(([rid, locs]) => `r[${rid}]: ${locs.map(([p, ln]) => `${p}:${ln}`).join(", ")}`),
    ).toEqual([]);

    const missingLeafRules: string[] = [];
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      verifySpecFileRulePrefixesAndLeafSections(specPath, missingLeafRules);
    }
    expect(missingLeafRules).toEqual([]);

    const templateRel = relPath(path.join(DOCS_DIR, "templates"));
    expect(collectStrayRulesOutsideSpecs(templateRel)).toEqual([]);
    expect(collectMalformedDocMarkers(templateRel)).toEqual([]);
    expect(collectVerifyTokensInDocs(templateRel)).toEqual([]);
    expect(collectLegacyTokenHits()).toEqual([]);
    expect(collectDocToTestLinks()).toEqual([]);

    const unresolvedRefs = collectSourceRefTokens()
      .filter(([ruleId]) => !rules.has(ruleId))
      .map(([ruleId, p, ln]) => `${p}:${ln} -> ref[${ruleId}]`);
    expect(unresolvedRefs).toEqual([]);

    const verifyTokens = collectTestVerifyTokens();
    expect(verifyTokens.length).toBeGreaterThan(0);
    expect(
      verifyTokens
        .filter(([ruleId]) => !rules.has(ruleId))
        .map(([ruleId, p, ln]) => `${p}:${ln} -> verify[${ruleId}]`),
    ).toEqual([]);

    expect(collectSpecTextParseViolations()).toEqual([]);
  });

  it("test_specs_have_monotonic_headings_standalone_rules_and_zero_legacy_stage_tokens", () => {
    const violations: string[] = [];
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      collectSpecCleanlinessViolations(specPath, violations);
    }
    for (const docPath of [
      REQ_FILE,
      path.join(REPO_ROOT, "README.md"),
      path.join(DOCS_DIR, "index.md"),
      path.join(PACKAGES_DIR, "kernels", "README.md"),
    ]) {
      collectNonSpecDocCleanlinessViolations(docPath, violations);
    }
    expect(violations).toEqual([]);
  });

  it("test_spec_rules_verified_in_tests", () => {
    const specFirstPendingVerifyRules: ReadonlySet<string> = new Set([
      "dsp.op-m.tracker.t1-plus",
      "dsp.op-m.kautz.er-sparse-bank",
      "dsp.op-t.routing.per-quad-and-series",
      "dsp.op-t.loop.convex-shimmer-freeze",
      "dsp.ctrl.gra.rational-and-entmax",
      "dsp.ctrl.gra.bibo-convex-hull",
    ]);
    const rules = collectSpecRules();
    const verifiedIds = new Set(collectTestVerifyTokens().map(([ruleId]) => ruleId));
    for (const pendingId of specFirstPendingVerifyRules) {
      expect(rules.has(pendingId), `unknown pending rule ${pendingId}`).toBe(true);
      expect(
        verifiedIds.has(pendingId),
        `${pendingId} is now verified; remove from specFirstPendingVerifyRules`,
      ).toBe(false);
    }
    expect(
      Array.from(rules.keys())
        .filter((r) => !verifiedIds.has(r) && !specFirstPendingVerifyRules.has(r))
        .sort(),
    ).toEqual([]);
  });

  it("test_engine_spec_scope_manifest", () => {
    const scopePath = path.join(PACKAGES_DIR, "engine", "spec_scope.toml");
    expect(fs.existsSync(scopePath)).toBe(true);
    const rulesMap = parseSpecScopeToml(fs.readFileSync(scopePath, "utf8"));
    const canonicalIds = new Set(collectSpecRules().keys());
    const manifestIds = new Set(rulesMap.keys());
    expect(
      Array.from(canonicalIds)
        .filter((id) => !manifestIds.has(id))
        .sort(),
    ).toEqual([]);
    expect(
      Array.from(manifestIds)
        .filter((id) => !canonicalIds.has(id))
        .sort(),
    ).toEqual([]);

    const engineDir = path.join(PACKAGES_DIR, "engine");
    const actualVerifies = new Map<string, Set<string>>();
    for (const rsPath of walkFilesByExtensions(engineDir, new Set([".rs"]))) {
      const rel = path.relative(engineDir, rsPath).split(path.sep).join("/");
      for (const m of fs.readFileSync(rsPath, "utf8").matchAll(SPEC_VERIFY_TOKEN_RE)) {
        const rid = m[1] ?? "";
        const set = actualVerifies.get(rid) ?? new Set<string>();
        set.add(rel);
        actualVerifies.set(rid, set);
      }
    }

    const errors: string[] = [];
    for (const [ruleId, entry] of Array.from(rulesMap.entries()).sort(([a], [b]) =>
      a.localeCompare(b),
    )) {
      checkEngineScopeEntry(ruleId, entry, actualVerifies, errors);
    }
    expect(errors).toEqual([]);
  });

  it("test_bidirectional_req_and_adr_traceability_across_specs", () => {
    const reqText = fs.readFileSync(REQ_FILE, "utf8");
    const definedReqs = new Set(Array.from(reqText.matchAll(REQ_TABLE_RE), (m) => m[1] ?? ""));
    const { adrStatuses, adrEnforcesSpecs } = collectAdrStatusAndSpecMaps();
    const supersededAdrs = new Set(
      Array.from(adrStatuses.entries())
        .filter(([, st]) => st === "superseded")
        .map(([aid]) => aid),
    );
    const { specCitedReqs, specCitedAdrs, specFileSet } = collectSpecCitationSets(supersededAdrs);

    for (const [aid, enforcedList] of adrEnforcesSpecs.entries()) {
      for (const specRel of enforcedList) {
        expect(specFileSet.has(specRel), `${aid} cites missing '${specRel}'`).toBe(true);
        if (adrStatuses.get(aid) === "accepted") specCitedAdrs.add(aid);
      }
    }

    expect(
      Array.from(definedReqs)
        .filter((r) => !specCitedReqs.has(r))
        .sort(),
    ).toEqual([]);
    const accepted = Array.from(adrStatuses.entries())
      .filter(([, st]) => st === "accepted")
      .map(([aid]) => aid);
    expect(accepted.filter((a) => !specCitedAdrs.has(a)).sort()).toEqual([]);
  });

  it("test_non_sdk_specs_are_language_and_implementation_independent", () => {
    const nonSdkSpecs = walkMarkdownFiles(SPECS_DIR).filter(
      (p) => !relPath(p).split(path.sep).join("/").startsWith("docs/specs/sdks/"),
    );
    expect(nonSdkSpecs.length).toBe(16);
    const violations: string[] = [];
    for (const specPath of nonSdkSpecs) {
      collectSpecLanguageLeaks(specPath, NON_SDK_SPEC_FORBIDDEN_RE, false, violations);
    }
    const tsSdkSpec = path.join(SPECS_DIR, "sdks", "typescript_sdk.md");
    collectSpecLanguageLeaks(tsSdkSpec, TS_SDK_RUST_FORBIDDEN_RE, true, violations);
    expect(violations).toEqual([]);
  });
});
