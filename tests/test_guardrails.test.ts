/**
 * Repository-wide architectural, documentation, and specification guardrail tests.
 *
 * Enforces the invariants defined in:
 * - docs/specs/architecture/project_and_documentation_quality.md
 * - docs/CONTRIBUTING.md
 * - CONTRIBUTING.md
 * - AGENTS.md
 *
 * verify[repo.code.type-safety-and-imports]
 * verify[repo.code.zerocopy-wire-structs]
 * verify[repo.code.shared-nostd-kernels]
 * verify[repo.code.modular-decomposition-and-tests]
 * verify[repo.code.constant-eval-and-derivations]
 * verify[repo.code.universal-black-box]
 * verify[repo.docs.agents-md-budgets]
 * verify[repo.docs.requirements-layer]
 * verify[repo.docs.spec-rules-and-adrs]
 * verify[repo.docs.markdown-table-math]
 * verify[repo.docs.github-latex-and-mermaid]
 * verify[repo.docs.relative-links-and-anchors]
 * verify[arch.index.subsystems]
 * verify[arch.tiers.maximize-quality-within-budget]
 * verify[qual.tiers.floors-not-stopping-criteria]
 * verify[train.pipeline.webgpu-scans-and-ozaki]
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(THIS_DIR, "..");
const DOCS_DIR = path.join(REPO_ROOT, "docs");
const SPECS_DIR = path.join(DOCS_DIR, "specs");
const ADR_DIR = path.join(DOCS_DIR, "adr");
const REQ_FILE = path.join(DOCS_DIR, "requirements", "system_goals_and_requirements.md");
const PACKAGES_DIR = path.join(REPO_ROOT, "packages");
const TESTS_DIR = path.join(REPO_ROOT, "tests");

const EXCLUDED_DIRS = new Set([
  ".agents",
  ".git",
  ".hg",
  ".svn",
  ".venv",
  "venv",
  "node_modules",
  "target",
  "dist",
  "dist-app",
  "build",
  "coverage",
  ".moon",
  ".benchmarks",
  ".vitest",
  ".vitepress",
  "cosmos-export",
  "playwright-report",
  "test-results",
  "scratch",
]);

const FRONTMATTER_RE = /^---\n([\s\S]*?)\n---\n/;
const FENCED_BLOCK_RE = /^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1[ \t]*$/gm;
const INLINE_CODE_RE = /`+[^`\n]+`+/g;

const SPEC_RULE_DEF_RE =
  /`r\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]`|^[ \t]*r\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\][ \t]*$/gm;
const SPEC_REF_TOKEN_RE = /\bref\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]/g;
const SPEC_VERIFY_TOKEN_RE = /\bverify\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]/g;
const MALFORMED_SPEC_MARKER_RE = /(?:^|[^a-zA-Z0-9_])(r|ref|verify)\[([^\]]+)\]/gm;
const VALID_RULE_ID_RE = /^[a-z0-9-]+(?:\.[a-z0-9-]+){2,}$/;

const REQ_TABLE_RE = /^\|\s*\*\*`(REQ-\d+\.\d+)`\*\*\s*\|/gm;
const REQ_TOKEN_RE = /\bREQ-\d+\.\d+\b/g;
const ADR_TOKEN_RE = /\bADR-\d{4}\b/g;

const MD_LINK_RE = /(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/gm;
const DISPLAY_MATH_RE = /\$\$([\s\S]*?)\$\$/g;
const INLINE_MATH_RE = /(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)/g;
const BACKSLASH_PUNCT_ESCAPE_RE = /\\[^a-zA-Z0-9\s]/;
const FORBIDDEN_MATH_MACROS = ["\\operatorname", "\\multicolumn", "\\texttt", "\\text{--"] as const;
const TEXT_UNDERSCORE_RE = /\\(?:text|mathrm)\{[^}]*_[^}]*\}/;
const LEGACY_ID_RE = /\bCL-\d{2}\b/;

function walkMarkdownFiles(root: string): string[] {
  const results: string[] = [];
  function recurse(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (EXCLUDED_DIRS.has(entry.name)) {
        continue;
      }
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        recurse(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        results.push(fullPath);
      }
    }
  }
  recurse(root);
  return results.sort();
}

function walkFilesByExtensions(root: string, extensions: ReadonlySet<string>): string[] {
  const results: string[] = [];
  if (!fs.existsSync(root)) {
    return results;
  }
  function recurse(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (EXCLUDED_DIRS.has(entry.name)) {
        continue;
      }
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        recurse(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name);
        if (extensions.has(ext)) {
          results.push(fullPath);
        }
      }
    }
  }
  recurse(root);
  return results.sort();
}

function relPath(filePath: string): string {
  return path.relative(REPO_ROOT, filePath);
}

function stripFencedCodeBlocks(text: string): string {
  return text.replace(FENCED_BLOCK_RE, (match) => "\n".repeat((match.match(/\n/g) ?? []).length));
}

function stripCode(text: string): string {
  const noFences = stripFencedCodeBlocks(text);
  return noFences.replace(INLINE_CODE_RE, " ");
}

function parseFrontmatter(text: string): Record<string, string | string[]> | null {
  const match = FRONTMATTER_RE.exec(text);
  if (!match || match[1] === undefined) {
    return null;
  }
  const raw = match[1];
  const data: Record<string, string | string[]> = {};
  let currentListKey: string | null = null;
  for (const line of raw.split("\n")) {
    const stripped = line.trim();
    if (!stripped || stripped.startsWith("#")) {
      continue;
    }
    if (stripped.startsWith("- ") && currentListKey !== null) {
      const val = stripped
        .slice(2)
        .trim()
        .replace(/^['"]|['"]$/g, "");
      const existing = data[currentListKey];
      if (Array.isArray(existing)) {
        existing.push(val);
      }
    } else if (stripped.includes(":")) {
      const colonIdx = stripped.indexOf(":");
      const key = stripped.slice(0, colonIdx).trim();
      const val = stripped
        .slice(colonIdx + 1)
        .trim()
        .replace(/^['"]|['"]$/g, "");
      if (val === "" || val === "[]") {
        data[key] = [];
        currentListKey = val === "" ? key : null;
      } else {
        data[key] = val;
        currentListKey = null;
      }
    }
  }
  return data;
}

function githubSlugify(headingText: string): string {
  let text = headingText.trim();
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  text = text.replace(/`/g, "");
  text = text.toLowerCase();
  const kept: string[] = [];
  for (const ch of text) {
    if (/[\p{L}\p{N}_-]/u.test(ch)) {
      kept.push(ch);
    } else if (ch === " " || ch === "\t") {
      kept.push(" ");
    }
  }
  const joined = kept.join("");
  return joined.replace(/[ \t]/g, "-");
}

function extractHeadingAnchors(mdText: string): Set<string> {
  const clean = stripFencedCodeBlocks(mdText);
  const anchors = new Set<string>();
  const counts = new Map<string, number>();
  for (const match of clean.matchAll(HEADING_RE)) {
    const headingBody = match[2] ?? "";
    const base = githubSlugify(headingBody);
    if (!base) {
      continue;
    }
    if (!counts.has(base)) {
      counts.set(base, 0);
      anchors.add(base);
    } else {
      const nextCount = (counts.get(base) ?? 0) + 1;
      counts.set(base, nextCount);
      anchors.add(`${base}-${nextCount}`);
    }
  }
  return anchors;
}

function collectSpecRules(): Map<string, [string, number][]> {
  const rules = new Map<string, [string, number][]>();
  for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
    const text = fs.readFileSync(specPath, "utf8");
    const clean = stripFencedCodeBlocks(text);
    const lines = clean.split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      const line = lines[idx] ?? "";
      for (const m of line.matchAll(SPEC_RULE_DEF_RE)) {
        const ruleId = m[1] ?? m[2] ?? "";
        const list = rules.get(ruleId) ?? [];
        list.push([relPath(specPath), idx + 1]);
        rules.set(ruleId, list);
      }
    }
  }
  return rules;
}

function collectSourceRefTokens(): [string, string, number][] {
  const refs: [string, string, number][] = [];
  const exts = new Set([".rs", ".ts", ".tsx", ".mjs", ".md"]);
  const templateRel = relPath(path.join(DOCS_DIR, "templates"));
  for (const root of [PACKAGES_DIR, DOCS_DIR, TESTS_DIR]) {
    for (const filePath of walkFilesByExtensions(root, exts)) {
      const rel = relPath(filePath);
      if (path.basename(filePath) === "test_guardrails.test.ts" || rel.startsWith(templateRel)) {
        continue;
      }
      const text = fs.readFileSync(filePath, "utf8");
      const lines = text.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(SPEC_REF_TOKEN_RE)) {
          refs.push([m[1] ?? "", relPath(filePath), idx + 1]);
        }
      }
    }
  }
  return refs;
}

function collectTestVerifyTokens(): [string, string, number][] {
  const verifies: [string, string, number][] = [];
  const exts = new Set([".rs", ".ts", ".tsx"]);
  for (const root of [TESTS_DIR, PACKAGES_DIR]) {
    for (const filePath of walkFilesByExtensions(root, exts)) {
      const text = fs.readFileSync(filePath, "utf8");
      const lines = text.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        for (const m of line.matchAll(SPEC_VERIFY_TOKEN_RE)) {
          verifies.push([m[1] ?? "", relPath(filePath), idx + 1]);
        }
      }
    }
  }
  return verifies;
}

interface SpecScopeRuleEntry {
  scope: string;
  reason?: string;
}

function _parseSpecScopeToml(text: string): Map<string, SpecScopeRuleEntry> {
  const rules = new Map<string, SpecScopeRuleEntry>();
  let currentRuleId: string | null = null;
  let currentEntry: SpecScopeRuleEntry | null = null;

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const headerMatch = /^\[rules\."([^"]+)"\]$/.exec(line);
    if (headerMatch && headerMatch[1] !== undefined) {
      if (currentRuleId !== null && currentEntry !== null) {
        rules.set(currentRuleId, currentEntry);
      }
      currentRuleId = headerMatch[1];
      currentEntry = { scope: "" };
      continue;
    }
    if (currentEntry === null) {
      continue;
    }
    const scopeMatch = /^scope\s*=\s*"([^"]+)"/.exec(line);
    if (scopeMatch && scopeMatch[1] !== undefined) {
      currentEntry.scope = scopeMatch[1];
      continue;
    }
    const reasonMatch = /^reason\s*=\s*"([^"]+)"/.exec(line);
    if (reasonMatch && reasonMatch[1] !== undefined) {
      currentEntry.reason = reasonMatch[1];
    }
  }
  if (currentRuleId !== null && currentEntry !== null) {
    rules.set(currentRuleId, currentEntry);
  }
  return rules;
}

function normalizeSpecRelPath(rawSpecRef: string): string {
  const trimmed = rawSpecRef.trim();
  if (trimmed.startsWith("docs/")) {
    return trimmed.slice("docs/".length);
  }
  if (trimmed.startsWith("../")) {
    return trimmed.slice("../".length);
  }
  return trimmed;
}

function isEscapedBackslashRun(text: string, idx: number): boolean {
  let bs = 0;
  while (idx > bs && text[idx - 1 - bs] === "\\") {
    bs++;
  }
  return (bs & 1) === 1;
}

function stripTomlComment(line: string): string {
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === "'" && !inDouble) {
      inSingle = !inSingle;
    } else if (ch === '"' && !inSingle && !isEscapedBackslashRun(line, i)) {
      inDouble = !inDouble;
    } else if (ch === "#" && !inSingle && !inDouble) {
      return line.slice(0, i);
    }
  }
  return line;
}

function extractTomlTableHeader(clean: string): string {
  const isArrayTable = clean.startsWith("[[");
  const openLen = isArrayTable ? 2 : 1;
  expect(
    isArrayTable
      ? clean.endsWith("]]") && clean.length > 4
      : clean.endsWith("]") && clean.length > 2,
    `Malformed TOML table header: ${clean}`,
  ).toBe(true);
  const inner = clean.slice(openLen, clean.length - openLen);
  let inS = false;
  let inD = false;
  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i];
    if (ch === "'" && !inD) {
      inS = !inS;
    } else if (ch === '"' && !inS && !isEscapedBackslashRun(inner, i)) {
      inD = !inD;
    } else if (!inS && !inD && (ch === "[" || ch === "]")) {
      throw new Error(`Unquoted bracket inside TOML table header: ${clean}`);
    }
  }
  expect(!inS && !inD, `Unterminated quote inside TOML table header: ${clean}`).toBe(true);
  return inner.trim();
}

function updateTomlLineDepths(
  clean: string,
  bracketDepth: number,
  braceDepth: number,
): [number, number] {
  let inS = false;
  let inD = false;
  let bDepth = bracketDepth;
  let brDepth = braceDepth;
  for (let i = 0; i < clean.length; i++) {
    const ch = clean[i];
    if (ch === "'" && !inD) {
      inS = !inS;
    } else if (ch === '"' && !inS && !isEscapedBackslashRun(clean, i)) {
      inD = !inD;
    } else if (!inS && !inD) {
      if (ch === "[") bDepth++;
      else if (ch === "]") bDepth = Math.max(0, bDepth - 1);
      else if (ch === "{") brDepth++;
      else if (ch === "}") brDepth = Math.max(0, brDepth - 1);
    }
  }
  return [bDepth, brDepth];
}

function parseTomlSections(tomlText: string): Array<{ header: string; body: string }> {
  expect(tomlText, "Cargo.toml must not use multi-line triple-quoted strings").not.toMatch(
    /'''|"""/,
  );
  const sections: Array<{ header: string; body: string }> = [];
  let currentHeader = "";
  let currentLines: string[] = [];
  let bracketDepth = 0;
  let braceDepth = 0;
  for (const rawLine of tomlText.split("\n")) {
    const clean = stripTomlComment(rawLine).trim();
    if (!clean) {
      continue;
    }
    if (bracketDepth === 0 && braceDepth === 0 && clean.startsWith("[")) {
      sections.push({ header: currentHeader, body: currentLines.join("\n") });
      currentHeader = extractTomlTableHeader(clean);
      currentLines = [];
      continue;
    }
    currentLines.push(clean);
    [bracketDepth, braceDepth] = updateTomlLineDepths(clean, bracketDepth, braceDepth);
  }
  sections.push({ header: currentHeader, body: currentLines.join("\n") });
  return sections;
}

function skipRustBlockComment(src: string, start: number): [string, number] {
  let i = start + 2;
  let depth = 1;
  let newlines = "";
  while (i < src.length && depth > 0) {
    if (src[i] === "/" && src[i + 1] === "*") {
      depth++;
      i += 2;
    } else if (src[i] === "*" && src[i + 1] === "/") {
      depth--;
      i += 2;
    } else {
      if (src[i] === "\n") newlines += "\n";
      i++;
    }
  }
  return [`${newlines} `, i];
}

function matchRustRawStringEnd(src: string, i: number): [string, number] | null {
  const ch = src[i] ?? "";
  const next = src[i + 1] ?? "";
  const isRaw = ch === "r" && (next === '"' || next === "#");
  const isByteRaw = ch === "b" && next === "r" && (src[i + 2] === '"' || src[i + 2] === "#");
  if (!isRaw && !isByteRaw) {
    return null;
  }
  let hashes = 0;
  let j = isByteRaw ? i + 2 : i + 1;
  while (src[j] === "#") {
    hashes++;
    j++;
  }
  if (src[j] !== '"') {
    return null;
  }
  const closing = `"${"#".repeat(hashes)}`;
  const endIdx = src.indexOf(closing, j + 1);
  if (endIdx === -1) {
    return null;
  }
  const nlCount = (src.slice(j + 1, endIdx).match(/\n/g) ?? []).length;
  return [`""${"\n".repeat(nlCount)}`, endIdx + closing.length];
}

function matchEscapedRustCharEnd(src: string, q: number): number {
  const esc = src[q + 2] ?? "";
  if (esc === "x") {
    return src[q + 5] === "'" ? q + 6 : -1;
  }
  if (esc === "u" && src[q + 3] === "{") {
    const braceIdx = src.indexOf("}", q + 4);
    return braceIdx !== -1 && braceIdx <= q + 12 && src[braceIdx + 1] === "'" ? braceIdx + 2 : -1;
  }
  return esc !== "" && esc !== "\n" && esc !== "\r" && src[q + 3] === "'" ? q + 4 : -1;
}

function matchRustCharLiteralEnd(src: string, i: number): number {
  const q = src[i] === "b" && src[i + 1] === "'" ? i + 1 : i;
  const c1 = src[q + 1] ?? "";
  if (c1 === "\\") {
    return matchEscapedRustCharEnd(src, q);
  }
  if (c1 !== "" && c1 !== "'" && c1 !== "\n" && c1 !== "\r") {
    const cp = src.codePointAt(q + 1);
    const cpLen = cp !== undefined && cp > 0xffff ? 2 : 1;
    return src[q + 1 + cpLen] === "'" ? q + 2 + cpLen : -1;
  }
  return -1;
}

function skipRustQuotedString(src: string, start: number): [string, number] {
  let out = '"';
  let i = start + 1;
  while (i < src.length && src[i] !== '"') {
    if (src[i] === "\\") {
      if (src[i + 1] === "\n") out += "\n";
      i += 2;
    } else {
      if (src[i] === "\n") out += "\n";
      i++;
    }
  }
  out += '"';
  return [out, i < src.length ? i + 1 : i];
}

function tryConsumeRustComment(src: string, i: number): [string, number] | null {
  const ch = src[i] ?? "";
  const next = src[i + 1] ?? "";
  if (ch === "/" && next === "/") {
    let j = i;
    while (j < src.length && src[j] !== "\n") j++;
    return ["", j];
  }
  if (ch === "/" && next === "*") {
    return skipRustBlockComment(src, i);
  }
  return null;
}

function tryConsumeRustStringOrChar(src: string, i: number): [string, number] | null {
  const rawMatch = matchRustRawStringEnd(src, i);
  if (rawMatch) return rawMatch;
  const ch = src[i] ?? "";
  const next = src[i + 1] ?? "";
  if (ch === "b" && next === '"') return ["", i + 1];
  if (ch === "'" || (ch === "b" && next === "'")) {
    const charEnd = matchRustCharLiteralEnd(src, i);
    if (charEnd !== -1) return ["''", charEnd];
    if (ch === "b") return ["b'", i + 2];
  }
  if (ch === '"') return skipRustQuotedString(src, i);
  return null;
}

function stripRustCommentsAndStrings(src: string): string {
  let out = "";
  let i = 0;
  while (i < src.length) {
    const token = tryConsumeRustComment(src, i) ?? tryConsumeRustStringOrChar(src, i);
    if (token) {
      out += token[0];
      i = token[1];
      continue;
    }
    out += src[i] ?? "";
    i++;
  }
  return out;
}

describe("DNAM Monorepo Architectural & Documentation Guardrails", () => {
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
      expect(
        parseFrontmatter(fs.readFileSync(tplPath, "utf8")),
        `Template ${tplName} is missing YAML frontmatter`,
      ).not.toBeNull();
    }

    const expectedDomains = new Set([
      "architecture",
      "dsp",
      "containers",
      "training",
      "security",
      "sdks",
    ]);
    const actualDomains = new Set(
      fs
        .readdirSync(SPECS_DIR, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name),
    );
    expect(actualDomains).toEqual(expectedDomains);
  });

  it("test_all_spec_files_have_valid_yaml_frontmatter", () => {
    const requiredKeys = [
      "id",
      "title",
      "status",
      "version",
      "domain",
      "rule_prefixes",
      "requirements",
      "adrs",
    ];
    const specFiles = walkMarkdownFiles(SPECS_DIR);
    expect(specFiles.length).toBe(17);

    const seenSpecIds = new Map<string, string>();
    for (const specPath of specFiles) {
      const text = fs.readFileSync(specPath, "utf8");
      const fm = parseFrontmatter(text);
      expect(fm, `${relPath(specPath)} is missing YAML frontmatter`).not.toBeNull();
      if (!fm) {
        continue;
      }
      expect("clamps" in fm, `${relPath(specPath)} has legacy 'clamps' frontmatter key`).toBe(
        false,
      );
      for (const key of requiredKeys) {
        expect(
          key in fm && fm[key] !== "" && (Array.isArray(fm[key]) ? fm[key].length > 0 : true),
          `${relPath(specPath)} frontmatter is missing non-empty '${key}'`,
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
    }
  });

  interface AdrRegistryMaps {
    seenAdrIds: Map<string, string>;
    adrStatuses: Map<string, string>;
    adrSupersededBy: Map<string, string>;
    adrFilenames: Map<string, string>;
  }

  function validateAdrFrontmatterFile(adrPath: string, maps: AdrRegistryMaps): void {
    const requiredKeys = [
      "id",
      "title",
      "status",
      "supersedes",
      "superseded_by",
      "specs",
      "requirements",
    ];
    const text = fs.readFileSync(adrPath, "utf8");
    const fm = parseFrontmatter(text);
    expect(fm, `${relPath(adrPath)} is missing YAML frontmatter`).not.toBeNull();
    if (!fm) {
      return;
    }
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
    if (status === "accepted") {
      expect((fm["specs"] as string[]).length).toBeGreaterThan(0);
    }
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
        expect(
          fmAdrs.includes(supId),
          `${relPath(docFile)} frontmatter cites superseded ${supId}`,
        ).toBe(false);
        expect(
          cleanBody.includes(supId) || (supFilename !== "" && rawText.includes(supFilename)),
          `${relPath(docFile)} cites superseded ${supId} (${supFilename})`,
        ).toBe(false);
      }
    }
  }

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
    for (const adrPath of adrFiles) {
      validateAdrFrontmatterFile(adrPath, maps);
    }

    for (const [adrId, supBy] of maps.adrSupersededBy.entries()) {
      expect(maps.adrStatuses.has(supBy), `${adrId} superseded_by '${supBy}' does not exist`).toBe(
        true,
      );
      expect(
        maps.adrStatuses.get(supBy) !== "superseded",
        `${adrId} superseded_by '${supBy}' points to another superseded ADR`,
      ).toBe(true);
    }

    const supersededIds = Array.from(maps.adrStatuses.entries())
      .filter(([, st]) => st === "superseded")
      .map(([aid]) => aid);
    verifySupersededAdrsNotCited(supersededIds, maps.adrFilenames);
  });

  it("test_requirement_registry_and_agnostic_prose", () => {
    const text = fs.readFileSync(REQ_FILE, "utf8");
    const fm = parseFrontmatter(text);
    expect(fm).not.toBeNull();
    if (!fm) {
      return;
    }
    expect(fm["id"]).toBe("req.system-goals");
    expect(fm["status"]).toBe("canonical");
    expect(fm["layer"]).toBe("requirements");

    const reqIds = Array.from(text.matchAll(REQ_TABLE_RE), (m) => m[1] ?? "");
    expect(reqIds.length).toBe(41);
    expect(reqIds.length).toBe(new Set(reqIds).size);

    const expectedCounts: Record<number, number> = {
      1: 9,
      2: 5,
      3: 3,
      4: 3,
      5: 6,
      6: 3,
      7: 4,
      8: 8,
    };
    for (const [goalStr, count] of Object.entries(expectedCounts)) {
      const goal = Number(goalStr);
      for (let sub = 1; sub <= count; sub++) {
        expect(reqIds).toContain(`REQ-${goal}.${sub}`);
      }
    }

    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const specFm = parseFrontmatter(fs.readFileSync(specPath, "utf8"));
      const reqList = specFm?.["requirements"];
      if (Array.isArray(reqList)) {
        for (const reqRef of reqList) {
          expect(
            reqIds.includes(reqRef),
            `${relPath(specPath)} frontmatter cites unknown requirement '${reqRef}'`,
          ).toBe(true);
        }
      }
    }

    const clean = stripFencedCodeBlocks(text);
    const proseOnly = clean.replace(/\]\([^)]+\)/g, "]()");
    const forbiddenFieldNames = [
      "max_supported_tier",
      "user_metadata",
      "royalty_splits",
      "allowed_protection_profiles",
      "allowed_oem_vendors",
      "output head",
      "output-head",
      "SIMD lane",
      "Hann-windowed",
      "JSON header",
      ".dnam",
      ".dnam-proj",
      ".dnam-ckpt",
      ".dnam-cache",
      ".dnam-seal",
      ".dnce",
      "mono_only",
      "CL-0",
      "CL-1",
      "CL-2",
    ];
    for (const field of forbiddenFieldNames) {
      expect(proseOnly).not.toContain(field);
    }
  });

  it("test_layer_separation_no_adr_narratives_in_canonical_specs", () => {
    const forbiddenHeadings = [
      "rejected alternatives",
      "counterexamples and failure modes",
      "why alternatives lose",
      "candidate comparison",
    ];
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const clean = stripFencedCodeBlocks(fs.readFileSync(specPath, "utf8"));
      for (const match of clean.matchAll(HEADING_RE)) {
        const headingText = (match[2] ?? "").toLowerCase();
        for (const phrase of forbiddenHeadings) {
          expect(headingText).not.toContain(phrase);
        }
      }
    }
  });

  function checkSpecHeadingSections(
    specPath: string,
    clean: string,
    missingLeafRules: string[],
  ): void {
    const sectionHeadingRe = /^(#{2,4})\s+(.+?)\s*$/gm;
    const headings = Array.from(clean.matchAll(sectionHeadingRe));
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
        missingLeafRules.push(
          `${relPath(specPath)}: leaf section '${"#".repeat(level)} ${title}' has no r[...] rule`,
        );
      }
    }
  }

  function verifySpecFileRulePrefixesAndLeafSections(
    specPath: string,
    missingLeafRules: string[],
  ): void {
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
        `${relPath(specPath)}: rule '${fullToken}' does not match any frontmatter rule_prefixes`,
      ).toBe(true);
    }
    checkSpecHeadingSections(specPath, clean, missingLeafRules);
  }

  function collectStrayRulesOutsideSpecs(templateRel: string): string[] {
    const strayRules: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const rel = relPath(mdPath);
      if (
        mdPath.startsWith(SPECS_DIR) ||
        rel.startsWith(templateRel) ||
        rel === "docs/CONTRIBUTING.md"
      ) {
        continue;
      }
      const clean = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        for (const m of (lines[idx] ?? "").matchAll(SPEC_RULE_DEF_RE)) {
          strayRules.push(`${rel}:${idx + 1} defines r[${m[1] ?? m[2] ?? ""}] outside docs/specs/`);
        }
      }
    }
    return strayRules;
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

  function scanRegexHitsInFile(filePath: string, re: RegExp, hits: string[]): void {
    const lines = fs.readFileSync(filePath, "utf8").split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      const m = re.exec(lines[idx] ?? "");
      if (m) hits.push(`${relPath(filePath)}:${idx + 1}: ${m[0]}`);
    }
  }

  function collectLegacyTokenHits(): string[] {
    const legacyHits: string[] = [];
    for (const filePath of [REQ_FILE, ...walkMarkdownFiles(SPECS_DIR)]) {
      scanRegexHitsInFile(filePath, LEGACY_ID_RE, legacyHits);
    }
    const legacyGuardrailRe = /\bG-0\d\b/;
    for (const root of [DOCS_DIR, PACKAGES_DIR, TESTS_DIR]) {
      for (const filePath of walkFilesByExtensions(root, new Set([".md", ".rs", ".ts", ".tsx"]))) {
        if (path.basename(filePath) === "test_guardrails.test.ts") continue;
        scanRegexHitsInFile(filePath, legacyGuardrailRe, legacyHits);
      }
    }
    return legacyHits;
  }

  function collectDocToTestLinks(): string[] {
    const docToTestLinks: string[] = [];
    const docRoots = [
      ...walkMarkdownFiles(DOCS_DIR),
      path.join(REPO_ROOT, "README.md"),
      path.join(REPO_ROOT, "CONTRIBUTING.md"),
      path.join(REPO_ROOT, "AGENTS.md"),
    ];
    for (const mdPath of docRoots) {
      const rel = relPath(mdPath);
      if (rel === "docs/specs/architecture/project_and_documentation_quality.md") continue;
      const isSpecOrReq =
        mdPath.startsWith(SPECS_DIR) || mdPath.startsWith(ADR_DIR) || mdPath === REQ_FILE;
      if (!isSpecOrReq) continue;
      const lines = stripCode(fs.readFileSync(mdPath, "utf8")).split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        for (const m of (lines[idx] ?? "").matchAll(MD_LINK_RE)) {
          const target = m[2] ?? "";
          if (
            target.includes("test_") ||
            target.includes("_spec_references") ||
            target.includes("_verification") ||
            target.includes("/tests/")
          ) {
            docToTestLinks.push(`${rel}:${idx + 1} -> ${target}`);
          }
        }
      }
    }
    return docToTestLinks;
  }

  function scanFileForSpecTextParse(
    filePath: string,
    forbiddenPatterns: readonly RegExp[],
    violations: string[],
  ): void {
    const rel = relPath(filePath);
    if (path.basename(filePath) === "test_guardrails.test.ts") return;
    if (!rel.includes("test") && !rel.includes("tests")) return;
    const lines = fs.readFileSync(filePath, "utf8").split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      const line = lines[idx] ?? "";
      const stripped = line.trim();
      if (
        stripped.startsWith("#") ||
        stripped.startsWith("//") ||
        stripped.startsWith("*") ||
        stripped.startsWith("/*")
      ) {
        continue;
      }
      if (forbiddenPatterns.some((pat) => pat.test(line))) {
        violations.push(`${rel}:${idx + 1}: ${stripped}`);
      }
    }
  }

  function collectSpecTextParseViolations(): string[] {
    const exts = new Set([".rs", ".ts", ".tsx"]);
    const forbiddenPatterns = [/docs\/specs\//, /docs\/adr\//, /docs\/requirements\//];
    const violations: string[] = [];
    for (const root of [TESTS_DIR, PACKAGES_DIR]) {
      for (const filePath of walkFilesByExtensions(root, exts)) {
        scanFileForSpecTextParse(filePath, forbiddenPatterns, violations);
      }
    }
    return violations;
  }

  it("test_spec_rule_identifiers_adr_structure_and_no_spec_text_parsing", () => {
    const rules = collectSpecRules();
    expect(rules.size).toBeGreaterThanOrEqual(150);

    const duplicates = Array.from(rules.entries())
      .filter(([, locs]) => locs.length > 1)
      .map(
        ([rid, locs]) =>
          `r[${rid}] defined multiple times: ${locs.map(([p, ln]) => `${p}:${ln}`).join(", ")}`,
      );
    expect(duplicates).toEqual([]);

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
      .map(([ruleId, pathStr, lineno]) => `${pathStr}:${lineno} -> ref[${ruleId}]`);
    expect(unresolvedRefs).toEqual([]);

    const verifyTokens = collectTestVerifyTokens();
    expect(verifyTokens.length).toBeGreaterThan(0);
    const unresolvedVerifies = verifyTokens
      .filter(([ruleId]) => !rules.has(ruleId))
      .map(([ruleId, pathStr, lineno]) => `${pathStr}:${lineno} -> verify[${ruleId}]`);
    expect(unresolvedVerifies).toEqual([]);

    expect(collectSpecTextParseViolations()).toEqual([]);

    const modRsFiles = walkFilesByExtensions(PACKAGES_DIR, new Set([".rs"]))
      .filter((p) => path.basename(p) === "mod.rs")
      .map((p) => relPath(p));
    expect(modRsFiles).toEqual([]);

    const proptestDirs = [
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
    ];
    for (const dirPath of proptestDirs) {
      const rsFiles = walkFilesByExtensions(dirPath, new Set([".rs"]));
      const hasProptest = rsFiles.some((f) => fs.readFileSync(f, "utf8").includes("proptest!"));
      expect(hasProptest, `${relPath(dirPath)} must contain at least one proptest! suite`).toBe(
        true,
      );
    }
  });

  it("test_spec_rules_verified_in_tests", () => {
    const rules = collectSpecRules();
    const verifiedIds = new Set(collectTestVerifyTokens().map(([ruleId]) => ruleId));
    const unverified = Array.from(rules.keys())
      .filter((r) => !verifiedIds.has(r))
      .sort();
    expect(unverified).toEqual([]);
  });

  function collectActualEngineVerifies(engineDir: string): Map<string, Set<string>> {
    const actualEngineVerifies = new Map<string, Set<string>>();
    for (const rsPath of walkFilesByExtensions(engineDir, new Set([".rs"]))) {
      const relToEngine = path.relative(engineDir, rsPath).split(path.sep).join("/");
      const text = fs.readFileSync(rsPath, "utf8");
      for (const m of text.matchAll(SPEC_VERIFY_TOKEN_RE)) {
        const rid = m[1] ?? "";
        const set = actualEngineVerifies.get(rid) ?? new Set<string>();
        set.add(relToEngine);
        actualEngineVerifies.set(rid, set);
      }
    }
    return actualEngineVerifies;
  }

  function validateEngineSpecScopeEntry(
    ruleId: string,
    entry: SpecScopeRuleEntry,
    actualEngineVerifies: ReadonlyMap<string, Set<string>>,
    errors: string[],
  ): void {
    const scope = entry.scope;
    if (scope !== "engine" && scope !== "out-of-scope") {
      errors.push(`r[${ruleId}]: invalid scope '${scope}'`);
      return;
    }
    if (scope === "out-of-scope") {
      if (!(entry.reason ?? "").trim()) {
        errors.push(`r[${ruleId}]: scope='out-of-scope' requires a non-empty 'reason'`);
      }
      const found = actualEngineVerifies.get(ruleId);
      if (found) {
        errors.push(
          `r[${ruleId}]: marked 'out-of-scope' but verified in packages/engine: ${Array.from(found).sort().join(", ")}`,
        );
      }
      return;
    }
    if ((actualEngineVerifies.get(ruleId)?.size ?? 0) === 0) {
      errors.push(
        `r[${ruleId}]: scope='engine', but no 'verify[${ruleId}]' found in packages/engine/**/*.rs`,
      );
    }
  }

  it("test_engine_spec_scope_manifest", () => {
    const scopePath = path.join(PACKAGES_DIR, "engine", "spec_scope.toml");
    expect(fs.existsSync(scopePath)).toBe(true);

    const rulesMap = _parseSpecScopeToml(fs.readFileSync(scopePath, "utf8"));
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

    const actualEngineVerifies = collectActualEngineVerifies(path.join(PACKAGES_DIR, "engine"));
    const errors: string[] = [];
    for (const [ruleId, entry] of Array.from(rulesMap.entries()).sort(([a], [b]) =>
      a.localeCompare(b),
    )) {
      validateEngineSpecScopeEntry(ruleId, entry, actualEngineVerifies, errors);
    }
    expect(errors).toEqual([]);
  });

  it("test_progressive_disclosure_agents_md_byte_limits", () => {
    const allAgents = walkMarkdownFiles(REPO_ROOT).filter((p) => path.basename(p) === "AGENTS.md");
    const rootAgents = path.join(REPO_ROOT, "AGENTS.md");
    expect(allAgents.includes(rootAgents)).toBe(true);

    const rootSize = fs.statSync(rootAgents).size;
    expect(rootSize).toBeLessThan(5000);

    const scopedAgents = allAgents.filter((p) => p !== rootAgents);
    expect(scopedAgents.length).toBeGreaterThanOrEqual(19);

    for (const agentsPath of allAgents) {
      const size = fs.statSync(agentsPath).size;
      if (agentsPath !== rootAgents) {
        expect(size, `${relPath(agentsPath)} is ${size} B (must be < 2,500 B)`).toBeLessThan(2500);
      }
      const text = fs.readFileSync(agentsPath, "utf8");
      const clean = stripFencedCodeBlocks(text);
      expect(clean).not.toContain("$$");
      expect(clean.match(SPEC_RULE_DEF_RE)).toBeNull();
    }
  });

  it("test_no_inline_latex_math_in_markdown_tables", () => {
    const violations: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const clean = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8"));
      const lines = clean.split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        const line = lines[idx] ?? "";
        if (!line.trimStart().startsWith("|")) {
          continue;
        }
        const withoutInlineCode = line.replace(INLINE_CODE_RE, "");
        if (withoutInlineCode.includes("$")) {
          violations.push(`${relPath(mdPath)}:${idx + 1}: ${line.trim().slice(0, 100)}`);
        }
      }
    }
    expect(violations).toEqual([]);
  });

  function checkDocMathBody(violations: string[], rel: string, kind: string, body: string): void {
    if (BACKSLASH_PUNCT_ESCAPE_RE.test(body)) {
      violations.push(`${rel} (${kind} backslash escape): ${body.trim().slice(0, 80)}`);
    }
    for (const macro of FORBIDDEN_MATH_MACROS) {
      if (body.includes(macro)) {
        violations.push(`${rel} (${kind} forbidden macro '${macro}'): ${body.trim().slice(0, 80)}`);
      }
    }
    if (TEXT_UNDERSCORE_RE.test(body)) {
      violations.push(
        `${rel} (${kind} unescaped '_' inside \\text/\\mathrm): ${body.trim().slice(0, 80)}`,
      );
    }
    const openCount = (body.match(/\{/g) ?? []).length;
    const closeCount = (body.match(/\}/g) ?? []).length;
    if (openCount !== closeCount) {
      violations.push(`${rel} (${kind} unbalanced braces): ${body.trim().slice(0, 80)}`);
    }
  }

  function checkDocMermaidBlocks(violations: string[], rel: string, rawText: string): void {
    const unquotedMermaidLabelRe = /\b[A-Za-z0-9_]+\[(?!")([^\]]*[():][^\]]*)\]/;
    const bidirMermaidArrowRe = /<(?:--|==)\s+[^->=\n]+\s+(?:-->|==>)/;
    for (const m of rawText.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
      const block = m[1] ?? "";
      const strippedBlock = block.trimStart();
      if (strippedBlock.startsWith("sequenceDiagram") && block.includes(";")) {
        violations.push(`${rel}: semicolon inside Mermaid sequenceDiagram`);
      } else if (strippedBlock.startsWith("flowchart")) {
        if (unquotedMermaidLabelRe.test(block)) {
          violations.push(`${rel}: unquoted special chars in Mermaid flowchart node label`);
        }
        if (bidirMermaidArrowRe.test(block)) {
          violations.push(`${rel}: invalid bidirectional arrow in Mermaid flowchart`);
        }
      }
    }
  }

  it("test_valid_github_latex_math_across_docs", () => {
    const violations: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const rel = relPath(mdPath);
      const rawText = fs.readFileSync(mdPath, "utf8");
      checkDocMermaidBlocks(violations, rel, rawText);
      const clean = stripCode(rawText);
      for (const m of clean.matchAll(DISPLAY_MATH_RE)) {
        checkDocMathBody(violations, rel, "display math", m[1] ?? "");
      }
      for (const m of clean.replace(DISPLAY_MATH_RE, " ").matchAll(INLINE_MATH_RE)) {
        checkDocMathBody(violations, rel, "inline math", m[1] ?? "");
      }
    }
    expect(violations).toEqual([]);
  });

  function resolveRelativeLinkTarget(
    mdPath: string,
    rawTarget: string,
  ): { targetFile: string; anchor: string } | null {
    if (
      rawTarget.startsWith("http://") ||
      rawTarget.startsWith("https://") ||
      rawTarget.startsWith("mailto:")
    ) {
      return null;
    }
    if (rawTarget.startsWith("#")) {
      return { targetFile: mdPath, anchor: rawTarget.slice(1) };
    }
    const hashIdx = rawTarget.indexOf("#");
    if (hashIdx !== -1) {
      return {
        targetFile: path.resolve(path.dirname(mdPath), rawTarget.slice(0, hashIdx)),
        anchor: rawTarget.slice(hashIdx + 1),
      };
    }
    return { targetFile: path.resolve(path.dirname(mdPath), rawTarget), anchor: "" };
  }

  function verifyResolvedLinkTarget(
    mdPath: string,
    lineNo: number,
    rawTarget: string,
    anchorCache: Map<string, Set<string>>,
    brokenLinks: string[],
  ): void {
    const resolved = resolveRelativeLinkTarget(mdPath, rawTarget);
    if (!resolved) return;
    const { targetFile, anchor } = resolved;
    if (!fs.existsSync(targetFile)) {
      brokenLinks.push(`${relPath(mdPath)}:${lineNo} -> target '${rawTarget}' does not exist`);
      return;
    }
    if (!anchor || !targetFile.endsWith(".md") || !fs.statSync(targetFile).isFile()) return;
    let anchors = anchorCache.get(targetFile);
    if (!anchors) {
      anchors = extractHeadingAnchors(fs.readFileSync(targetFile, "utf8"));
      anchorCache.set(targetFile, anchors);
    }
    if (!anchors.has(anchor)) {
      brokenLinks.push(
        `${relPath(mdPath)}:${lineNo} -> anchor '#${anchor}' not found in ${relPath(targetFile)}`,
      );
    }
  }

  function checkMarkdownFileLinks(
    mdPath: string,
    anchorCache: Map<string, Set<string>>,
    brokenLinks: string[],
  ): void {
    const lines = stripCode(fs.readFileSync(mdPath, "utf8")).split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      for (const m of (lines[idx] ?? "").matchAll(MD_LINK_RE)) {
        verifyResolvedLinkTarget(mdPath, idx + 1, m[2] ?? "", anchorCache, brokenLinks);
      }
    }
  }

  it("test_relative_markdown_links_resolve", () => {
    const anchorCache = new Map<string, Set<string>>();
    const brokenLinks: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      checkMarkdownFileLinks(mdPath, anchorCache, brokenLinks);
    }
    expect(brokenLinks).toEqual([]);
  });

  it("test_docs_index_links_all_specs_and_adrs", () => {
    const indexText = fs.readFileSync(path.join(DOCS_DIR, "index.md"), "utf8");
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      const relFromDocs = path.relative(DOCS_DIR, specPath).split(path.sep).join("/");
      expect(indexText).toContain(relFromDocs);
    }
    for (const entry of fs.readdirSync(ADR_DIR).sort()) {
      if (!entry.endsWith(".md") || entry === "README.md") {
        continue;
      }
      const relFromDocs = `adr/${entry}`;
      expect(indexText).toContain(relFromDocs);
    }
  });

  it("test_root_readme_stays_focused_on_musician_and_oem_personas", () => {
    const readmeText = fs.readFileSync(path.join(REPO_ROOT, "README.md"), "utf8");
    expect(readmeText).toContain("Musicians");
    expect(readmeText).toContain("Tone Creators");
    expect(readmeText).toContain("docs/index.md");
    expect(readmeText).toContain("CONTRIBUTING.md");
    expect(readmeText).not.toContain("$$");
    expect(readmeText.match(SPEC_RULE_DEF_RE)).toBeNull();
  });

  function collectAdrStatusesAndEnforcedSpecs(): {
    adrStatuses: Map<string, string>;
    adrEnforcesSpecs: Map<string, string[]>;
  } {
    const adrStatuses = new Map<string, string>();
    const adrEnforcesSpecs = new Map<string, string[]>();
    for (const entry of fs.readdirSync(ADR_DIR).sort()) {
      if (!entry.endsWith(".md") || entry === "README.md") continue;
      const fm = parseFrontmatter(fs.readFileSync(path.join(ADR_DIR, entry), "utf8"));
      if (!fm) continue;
      const adrId = String(fm["id"]);
      adrStatuses.set(adrId, String(fm["status"]));
      const rawSpecs = Array.isArray(fm["specs"]) ? fm["specs"] : [];
      adrEnforcesSpecs.set(
        adrId,
        rawSpecs.map((s) => normalizeSpecRelPath(s)),
      );
    }
    return { adrStatuses, adrEnforcesSpecs };
  }

  function collectSpecReqAndAdrCitations(
    specPath: string,
    supersededAdrs: ReadonlySet<string>,
    specCitedReqs: Set<string>,
    specCitedAdrs: Set<string>,
  ): void {
    const text = fs.readFileSync(specPath, "utf8");
    const fm = parseFrontmatter(text);
    if (Array.isArray(fm?.["requirements"])) {
      for (const r of fm["requirements"]) specCitedReqs.add(r);
    }
    for (const m of text.matchAll(REQ_TOKEN_RE)) specCitedReqs.add(m[0]);
    if (Array.isArray(fm?.["adrs"])) {
      for (const a of fm["adrs"]) {
        expect(
          supersededAdrs.has(a),
          `${relPath(specPath)} frontmatter cites superseded ADR '${a}'`,
        ).toBe(false);
        specCitedAdrs.add(a);
      }
    }
    for (const m of text.matchAll(ADR_TOKEN_RE)) specCitedAdrs.add(m[0]);
  }

  it("test_bidirectional_req_and_adr_traceability_across_specs", () => {
    const reqText = fs.readFileSync(REQ_FILE, "utf8");
    const definedReqs = new Set(Array.from(reqText.matchAll(REQ_TABLE_RE), (m) => m[1] ?? ""));
    const { adrStatuses, adrEnforcesSpecs } = collectAdrStatusesAndEnforcedSpecs();
    const supersededAdrs = new Set(
      Array.from(adrStatuses.entries())
        .filter(([, st]) => st === "superseded")
        .map(([aid]) => aid),
    );

    const specCitedReqs = new Set<string>();
    const specCitedAdrs = new Set<string>();
    const specFileSet = new Set<string>();
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      specFileSet.add(path.relative(DOCS_DIR, specPath).split(path.sep).join("/"));
      collectSpecReqAndAdrCitations(specPath, supersededAdrs, specCitedReqs, specCitedAdrs);
    }

    for (const [aid, enforcedList] of adrEnforcesSpecs.entries()) {
      for (const specRel of enforcedList) {
        expect(
          specFileSet.has(specRel),
          `${aid} frontmatter specs cites non-existent '${specRel}'`,
        ).toBe(true);
        if (adrStatuses.get(aid) === "accepted") {
          specCitedAdrs.add(aid);
        }
      }
    }

    expect(
      Array.from(definedReqs)
        .filter((r) => !specCitedReqs.has(r))
        .sort(),
    ).toEqual([]);
    const acceptedAdrs = Array.from(adrStatuses.entries())
      .filter(([, st]) => st === "accepted")
      .map(([aid]) => aid);
    expect(acceptedAdrs.filter((a) => !specCitedAdrs.has(a)).sort()).toEqual([]);
  });

  function checkTsTypeSafety(): void {
    const tsViolations: string[] = [];
    const forbiddenTsPatterns = [
      /@ts-ignore\b/,
      /@ts-expect-error\b/,
      /:\s*any\b/,
      /\bas\s+any\b/,
      /<any>/,
    ];
    for (const root of [PACKAGES_DIR, TESTS_DIR]) {
      for (const tsPath of walkFilesByExtensions(root, new Set([".ts", ".tsx"]))) {
        if (path.basename(tsPath) === "test_guardrails.test.ts") continue;
        const lines = fs.readFileSync(tsPath, "utf8").split("\n");
        for (let idx = 0; idx < lines.length; idx++) {
          const line = lines[idx] ?? "";
          const trimmed = line.trim();
          if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;
          if (forbiddenTsPatterns.some((pat) => pat.test(line))) {
            tsViolations.push(`${relPath(tsPath)}:${idx + 1}: ${trimmed}`);
          }
        }
      }
    }
    expect(tsViolations).toEqual([]);
  }

  function checkRustCrateUnsafeAttributes(): void {
    const crateForbidOrDenyRe = /#\s*!\s*\[\s*(?:forbid|deny)\s*\(\s*unsafe_code\s*\)\s*\]/;
    for (const rustLibRel of ["packages/engine/src/lib.rs", "packages/js/rust/lib.rs"]) {
      const libPath = path.join(REPO_ROOT, rustLibRel);
      expect(fs.existsSync(libPath), `${rustLibRel} must exist`).toBe(true);
      const strippedLib = stripRustCommentsAndStrings(fs.readFileSync(libPath, "utf8"));
      expect(
        crateForbidOrDenyRe.test(strippedLib),
        `${rustLibRel} must declare #![forbid(unsafe_code)] or #![deny(unsafe_code)] in active code`,
      ).toBe(true);
    }

    const strictForbidRootSet = new Set<string>([
      "packages/fixtures/src/lib.rs",
      "packages/capture/src/lib.rs",
      "packages/studio/rust/lib.rs",
      "packages/studio/rust/main.rs",
      "packages/studio/build.rs",
      "packages/js/build.rs",
    ]);
    for (const safePkg of ["fixtures", "capture", "studio"]) {
      for (const subDir of ["src/bin", "rust/bin", "tests", "benches", "examples"]) {
        for (const p of walkFilesByExtensions(
          path.join(PACKAGES_DIR, safePkg, subDir),
          new Set([".rs"]),
        )) {
          strictForbidRootSet.add(relPath(p));
        }
      }
    }
    const crateStrictForbidRe = /#\s*!\s*\[\s*forbid\s*\(\s*unsafe_code\s*\)\s*\]/;
    for (const strictForbidLibRel of [...strictForbidRootSet].sort()) {
      const libPath = path.join(REPO_ROOT, strictForbidLibRel);
      expect(fs.existsSync(libPath), `${strictForbidLibRel} must exist`).toBe(true);
      const strippedRoot = stripRustCommentsAndStrings(fs.readFileSync(libPath, "utf8"));
      expect(
        crateStrictForbidRe.test(strippedRoot),
        `${strictForbidLibRel} must declare #![forbid(unsafe_code)] in active code`,
      ).toBe(true);
    }
  }

  function checkRustSafeOnlyCrates(): void {
    const unsafeLintOverrideRe = /\b(?:allow|warn|expect)\s*\([^)]*\bunsafe_code\b/;
    const unsafeConstructRe =
      /\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)|(?<!r#)\bunsafe\s*\(|#\s*!?\s*\[\s*unsafe\s*\(/;
    const unsafeBlockOrItemOnlyRe = /\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)/;
    const unsafeViolations: string[] = [];
    for (const safePkg of ["fixtures", "capture", "studio"]) {
      for (const rsFile of walkFilesByExtensions(
        path.join(PACKAGES_DIR, safePkg),
        new Set([".rs"]),
      )) {
        const stripped = stripRustCommentsAndStrings(fs.readFileSync(rsFile, "utf8"));
        if (unsafeLintOverrideRe.test(stripped) || unsafeConstructRe.test(stripped)) {
          unsafeViolations.push(`${relPath(rsFile)}: contains forbidden unsafe construct`);
        }
      }
    }
    for (const rsFile of walkFilesByExtensions(path.join(PACKAGES_DIR, "js"), new Set([".rs"]))) {
      const rel = relPath(rsFile);
      const stripped = stripRustCommentsAndStrings(fs.readFileSync(rsFile, "utf8"));
      const isWasmExportShim =
        rel === "packages/js/rust/wasm.rs" ||
        (rel.startsWith("packages/js/rust/wasm/") && rel !== "packages/js/rust/wasm/abi.rs");
      if (unsafeBlockOrItemOnlyRe.test(stripped)) {
        unsafeViolations.push(`${rel}: contains forbidden unsafe block or item`);
      }
      if (
        !isWasmExportShim &&
        (unsafeLintOverrideRe.test(stripped) || unsafeConstructRe.test(stripped))
      ) {
        unsafeViolations.push(`${rel}: contains forbidden unsafe_code override or #[unsafe(...)]`);
      }
    }
    expect(unsafeViolations).toEqual([]);
  }

  function checkPackageLineCeilings(): void {
    const lineCeilingViolations: string[] = [];
    const strictLinePackages: Array<{ pkg: string; exts: Set<string> }> = [
      { pkg: "fixtures", exts: new Set([".rs"]) },
      { pkg: "capture", exts: new Set([".rs"]) },
      { pkg: "ui", exts: new Set([".ts", ".tsx"]) },
      { pkg: "js", exts: new Set([".rs", ".ts", ".tsx"]) },
      { pkg: "studio", exts: new Set([".rs", ".ts", ".tsx"]) },
    ];
    for (const { pkg, exts } of strictLinePackages) {
      for (const srcFile of walkFilesByExtensions(path.join(PACKAGES_DIR, pkg), exts)) {
        const content = fs.readFileSync(srcFile, "utf8");
        const lineCount = content.endsWith("\n")
          ? content.split("\n").length - 1
          : content.split("\n").length;
        if (lineCount >= 900) {
          lineCeilingViolations.push(`${relPath(srcFile)} (${lineCount} lines >= 900)`);
        }
      }
    }
    expect(lineCeilingViolations).toEqual([]);
  }

  function checkFixturesManifestStrictness(): void {
    const allowedFixturesSections = new Set([
      "",
      "package",
      "lib",
      "lints",
      "dependencies",
      "dev-dependencies",
    ]);
    const fixturesCargo = fs.readFileSync(
      path.join(PACKAGES_DIR, "fixtures", "Cargo.toml"),
      "utf8",
    );
    for (const sec of parseTomlSections(fixturesCargo)) {
      expect(
        allowedFixturesSections.has(sec.header),
        `packages/fixtures/Cargo.toml must only contain [package], [lib], [lints], [dependencies], and [dev-dependencies], found [${sec.header}]`,
      ).toBe(true);
      expect(
        /(?:^|\n|\.)\s*target\s*(?:=|\.)/.test(sec.body),
        `packages/fixtures/Cargo.toml must not define target-specific tables/keys in [${sec.header}]`,
      ).toBe(false);
      expect(
        /(?:^|\n|\.)\s*(?:(?:build|dev)-)?dependencies\s*(?:=|\.)/.test(sec.body),
        `packages/fixtures/Cargo.toml must not define inline or dotted dependency tables in [${sec.header}]`,
      ).toBe(false);
      if (sec.header === "dependencies") {
        const depKeys = sec.body
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0)
          .map((line) => line.split(/[=.]/)[0]?.trim() ?? "");
        for (const key of depKeys) {
          expect(
            key === "nalgebra" || key === "realfft",
            `packages/fixtures/Cargo.toml [dependencies] may only contain simulation kernels (nalgebra, realfft), found ${key}`,
          ).toBe(true);
        }
      }
    }
    for (const badTok of [
      "dnam-engine",
      "dnam-capture",
      "dnam-trainer",
      "dnam-studio",
      "dnam-js",
      "dnam_js",
    ]) {
      expect(fixturesCargo).not.toContain(badTok);
    }
  }

  function checkFixturesAndCaptureCargoBoundaries(): void {
    checkFixturesManifestStrictness();
    expect(fs.existsSync(path.join(PACKAGES_DIR, "trainer"))).toBe(false);
    expect(
      fs.existsSync(path.join(DOCS_DIR, "specs", "containers", "checkpoint_and_cache_format.md")),
    ).toBe(false);
    expect(
      fs.existsSync(path.join(DOCS_DIR, "specs", "containers", "project_container_format.md")),
    ).toBe(true);

    const fixturesRefRegex = /dnam[-_]fixtures|packages\/fixtures|\.\.\/fixtures/;
    const engineCargo = fs.readFileSync(path.join(PACKAGES_DIR, "engine", "Cargo.toml"), "utf8");
    expect(engineCargo).not.toMatch(fixturesRefRegex);
    expect(engineCargo).not.toMatch(
      /dnam[-_](?:capture|trainer)|packages\/(?:capture|trainer)|\.\.\/(?:capture|trainer)/,
    );
    expect(fs.readFileSync(path.join(PACKAGES_DIR, "fixtures", "moon.yml"), "utf8")).toMatch(
      /^dependsOn:\s*\[\]\s*$/m,
    );
    const engineMoon = fs.readFileSync(path.join(PACKAGES_DIR, "engine", "moon.yml"), "utf8");
    expect(engineMoon).toMatch(/^dependsOn:\s*\[\]\s*$/m);
    expect(engineMoon).not.toContain("fixtures");

    const captureCargo = fs.readFileSync(path.join(PACKAGES_DIR, "capture", "Cargo.toml"), "utf8");
    let foundFixturesInCaptureDevDeps = false;
    let foundEngineInCaptureDeps = false;
    for (const sec of parseTomlSections(captureCargo)) {
      expect(
        sec.header,
        "packages/capture/Cargo.toml must remain a pure library without [[bin]] targets (ADR-0020)",
      ).not.toBe("bin");
      if (fixturesRefRegex.test(sec.header) || fixturesRefRegex.test(sec.body)) {
        expect(
          sec.header,
          `packages/capture/Cargo.toml must only reference dnam-fixtures in [dev-dependencies], found in [${sec.header}]`,
        ).toBe("dev-dependencies");
        foundFixturesInCaptureDevDeps = true;
      }
      if (sec.header === "dependencies" && sec.body.includes("dnam-engine")) {
        foundEngineInCaptureDeps = true;
      }
    }
    expect(foundFixturesInCaptureDevDeps).toBe(true);
    expect(foundEngineInCaptureDeps).toBe(true);
    expect(captureCargo).not.toMatch(/\b(?:clap|cpal|toml)\b/);
    expect(fs.readFileSync(path.join(PACKAGES_DIR, "studio", "Cargo.toml"), "utf8")).not.toMatch(
      /\b(?:clap|toml)\b/,
    );

    const captureMoon = fs.readFileSync(path.join(PACKAGES_DIR, "capture", "moon.yml"), "utf8");
    expect(captureMoon).toMatch(/^\s*-\s*"engine"\s*$/m);
    expect(captureMoon).toMatch(/-\s*id:\s*"fixtures"\s*\n\s*scope:\s*"development"/);

    for (const nonConsumerPkg of ["studio", "js"]) {
      const pkgCargo = fs.readFileSync(
        path.join(PACKAGES_DIR, nonConsumerPkg, "Cargo.toml"),
        "utf8",
      );
      expect(
        pkgCargo,
        `packages/${nonConsumerPkg}/Cargo.toml must not depend on dnam-fixtures`,
      ).not.toMatch(fixturesRefRegex);
    }
    for (const nonConsumerMoonPkg of ["studio", "js", "ui"]) {
      const pkgMoon = fs.readFileSync(
        path.join(PACKAGES_DIR, nonConsumerMoonPkg, "moon.yml"),
        "utf8",
      );
      expect(
        pkgMoon,
        `packages/${nonConsumerMoonPkg}/moon.yml must not depend on fixtures`,
      ).not.toContain("fixtures");
    }
  }

  function checkBannedDependenciesAndIsolation(): void {
    for (const cargoManifest of [
      path.join(REPO_ROOT, "Cargo.toml"),
      path.join(REPO_ROOT, "Cargo.lock"),
      ...["engine", "fixtures", "capture", "studio", "js"].map((pkg) =>
        path.join(PACKAGES_DIR, pkg, "Cargo.toml"),
      ),
    ]) {
      const manifestText = fs.readFileSync(cargoManifest, "utf8");
      expect(manifestText, `${relPath(cargoManifest)} must not depend on ndarray`).not.toContain(
        "ndarray",
      );
      expect(manifestText, `${relPath(cargoManifest)} must not depend on dialoguer`).not.toContain(
        "dialoguer",
      );
    }
    for (const runtimePkg of ["engine", "capture", "studio", "js"]) {
      const pkgManifest = path.join(PACKAGES_DIR, runtimePkg, "Cargo.toml");
      const manifestText = fs.readFileSync(pkgManifest, "utf8");
      expect(manifestText, `${relPath(pkgManifest)} must not depend on realfft`).not.toContain(
        "realfft",
      );
      expect(manifestText, `${relPath(pkgManifest)} must not depend on rustfft`).not.toContain(
        "rustfft",
      );
      expect(manifestText, `${relPath(pkgManifest)} must not depend on nalgebra`).not.toContain(
        "nalgebra",
      );
    }

    for (const mod of ["metrics.rs", "whitebox.rs", "proj.rs", "align.rs"]) {
      expect(fs.existsSync(path.join(PACKAGES_DIR, "capture", "src", mod))).toBe(true);
    }

    const forbiddenFixturesSymbolsRe =
      /\b(?:Tier|DistillationTarget|internal_nodes|DnwbHeader64B|DNWB|dnam_engine|dnam_capture|dnam_trainer)\b/;
    const fixturesLeakViolations: string[] = [];
    for (const rsFile of walkFilesByExtensions(
      path.join(PACKAGES_DIR, "fixtures"),
      new Set([".rs"]),
    )) {
      if (
        forbiddenFixturesSymbolsRe.test(
          stripRustCommentsAndStrings(fs.readFileSync(rsFile, "utf8")),
        )
      ) {
        fixturesLeakViolations.push(relPath(rsFile));
      }
    }
    expect(fixturesLeakViolations).toEqual([]);

    const nonTestFixturesViolations: string[] = [];
    const pathOrIncludeFixturesRe =
      /(?:#\s*\[\s*path\s*=|include(?:_str|_bytes)?\s*!)\s*\(?[^;\n]*fixtures/;
    for (const scanDir of [
      path.join(PACKAGES_DIR, "capture", "src"),
      path.join(PACKAGES_DIR, "engine"),
      path.join(PACKAGES_DIR, "studio"),
      path.join(PACKAGES_DIR, "js"),
    ]) {
      for (const rsFile of walkFilesByExtensions(scanDir, new Set([".rs"]))) {
        const raw = fs.readFileSync(rsFile, "utf8");
        const noComments = raw.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
        if (
          /\bdnam_fixtures\b/.test(stripRustCommentsAndStrings(raw)) ||
          pathOrIncludeFixturesRe.test(noComments)
        ) {
          nonTestFixturesViolations.push(relPath(rsFile));
        }
      }
    }
    const captureBuildRs = path.join(PACKAGES_DIR, "capture", "build.rs");
    if (fs.existsSync(captureBuildRs)) nonTestFixturesViolations.push(relPath(captureBuildRs));
    expect(nonTestFixturesViolations).toEqual([]);
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
        if (rel === skipRel) continue;
        if (pattern.test(fs.readFileSync(fPath, "utf8"))) hits.push(rel);
      }
    }
    return hits;
  }

  function checkZeroRetiredTokensAndPython(): void {
    const retiredCkptOrTrainerRe =
      /(?:\.dnam-ckpt|\.dnam-cache|\.dnce\b|\bDNCK\b|\bDNCE\b|\bdnam[-_]trainer\b)/;
    const scanExts = new Set([".md", ".rs", ".ts", ".tsx", ".json", ".toml", ".yml"]);
    expect(
      collectTreeMatches(
        [PACKAGES_DIR, SPECS_DIR],
        scanExts,
        retiredCkptOrTrainerRe,
        "docs/specs/architecture/project_and_documentation_quality.md",
      ),
    ).toEqual([]);

    expect(collectSpecRules().has("qual.companion.mr-iir")).toBe(true);
    const retiredMrStftRe = /\bmr[-_]?stft\b/i;
    const mrStftViolations = collectTreeMatches(
      [DOCS_DIR, PACKAGES_DIR],
      scanExts,
      retiredMrStftRe,
    );
    for (const rootMd of ["AGENTS.md", "README.md", "CONTRIBUTING.md", "CLA.md"]) {
      const p = path.join(REPO_ROOT, rootMd);
      if (fs.existsSync(p) && retiredMrStftRe.test(fs.readFileSync(p, "utf8")))
        mrStftViolations.push(relPath(p));
    }
    expect(mrStftViolations).toEqual([]);
    expect(walkFilesByExtensions(REPO_ROOT, new Set([".py"]))).toEqual([]);
  }

  it("test_type_safety_and_import_boundaries", () => {
    checkTsTypeSafety();
    checkRustCrateUnsafeAttributes();
    checkRustSafeOnlyCrates();
    checkPackageLineCeilings();
    checkFixturesAndCaptureCargoBoundaries();
    checkBannedDependenciesAndIsolation();
    checkZeroRetiredTokensAndPython();
  });

  function collectTodoViolationsInFile(
    filePath: string,
    todoRe: RegExp,
    violations: string[],
  ): void {
    const lines = fs.readFileSync(filePath, "utf8").split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      const line = lines[idx] ?? "";
      if (!todoRe.test(line)) continue;
      if (line.includes("`TODO`") || line.includes("`FIXME`") || line.includes("todo =")) continue;
      violations.push(`${relPath(filePath)}:${idx + 1}: ${line.trim().slice(0, 100)}`);
    }
  }

  it("test_zero_todo_placeholders_in_docs_and_source", () => {
    const todoRe = /\b(?:TODO|FIXME|XXX|HACK)\b/;
    const violations: string[] = [];
    for (const root of [DOCS_DIR, PACKAGES_DIR, TESTS_DIR]) {
      for (const filePath of walkFilesByExtensions(root, new Set([".md", ".rs", ".ts", ".tsx"]))) {
        if (path.basename(filePath) === "test_guardrails.test.ts") continue;
        collectTodoViolationsInFile(filePath, todoRe, violations);
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_license_and_cla_parity_across_manifests", () => {
    const licensePath = path.join(REPO_ROOT, "LICENSE");
    const claPath = path.join(REPO_ROOT, "CLA.md");
    expect(fs.existsSync(licensePath)).toBe(true);
    expect(fs.existsSync(claPath)).toBe(true);

    const licenseText = fs.readFileSync(licensePath, "utf8");
    expect(licenseText).toContain("DNAM Source-Available & Branding License 1.0.0");
    expect(licenseText).toContain("Distribution & Hardware/Plugin Linking License");

    const cargoToml = fs.readFileSync(path.join(REPO_ROOT, "Cargo.toml"), "utf8");
    expect(cargoToml).toContain('license = "LicenseRef-DNAM-1.0.0"');

    for (const pkgJsonRel of [
      "package.json",
      "packages/js/package.json",
      "packages/ui/package.json",
      "packages/studio/package.json",
    ]) {
      const pkgText = fs.readFileSync(path.join(REPO_ROOT, pkgJsonRel), "utf8");
      expect(pkgText).toContain('"license": "LicenseRef-DNAM-1.0.0"');
    }
  });

  function scanEngineFileForForbiddenCalls(
    rsPath: string,
    forbiddenCallRe: RegExp,
    violations: string[],
  ): void {
    const base = path.basename(rsPath);
    if (base === "test_support.rs" || base === "exact_int_verify.rs") return;
    const lines = fs.readFileSync(rsPath, "utf8").split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      const line = lines[idx] ?? "";
      const trimmed = line.trim();
      const nextTrimmed = (lines[idx + 1] ?? "").trim();
      if (
        trimmed.startsWith("#[cfg(test)]") &&
        (nextTrimmed.startsWith("mod ") || nextTrimmed.startsWith("pub mod "))
      ) {
        break;
      }
      if (!trimmed.startsWith("//") && !trimmed.startsWith("*") && forbiddenCallRe.test(line)) {
        violations.push(`${relPath(rsPath)}:${idx + 1}: ${trimmed}`);
      }
    }
  }

  // verify[repo.code.realtime-disassembly-audit]
  // verify[arch.simd.zero-stall-lowering]
  it("test_engine_realtime_no_libm_or_float_rem_euclid", () => {
    const engineSrcDir = path.join(PACKAGES_DIR, "engine", "src");
    const forbiddenCallRe =
      /\.(?:rem_euclid|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|exp|exp2|ln|log2|log10|powf|powi|sqrt|cbrt|hypot)\s*\(/;
    const violations: string[] = [];
    for (const rsPath of walkFilesByExtensions(engineSrcDir, new Set([".rs"]))) {
      scanEngineFileForForbiddenCalls(rsPath, forbiddenCallRe, violations);
    }
    expect(violations).toEqual([]);
  });

  // verify[repo.code.explicit-simd-and-coverage]
  it("test_engine_no_coverage_off_or_simd_macros", () => {
    const engineSrcDir = path.join(PACKAGES_DIR, "engine", "src");
    const mathSrcDir = path.join(engineSrcDir, "math");
    const violations: string[] = [];

    for (const rsPath of walkFilesByExtensions(engineSrcDir, new Set([".rs"]))) {
      const content = fs.readFileSync(rsPath, "utf8");
      if (content.includes("coverage(off)")) {
        violations.push(`${relPath(rsPath)}: contains forbidden coverage(off)`);
      }
    }
    for (const rsPath of walkFilesByExtensions(mathSrcDir, new Set([".rs"]))) {
      const content = fs.readFileSync(rsPath, "utf8");
      if (content.includes("macro_rules!")) {
        violations.push(`${relPath(rsPath)}: contains forbidden macro_rules! in math/`);
      }
    }
    expect(violations).toEqual([]);
  });

  function checkRustModularFile(rsPath: string, violations: string[]): void {
    const sharedTestHelperNames = new Set(["test_support.rs", "test_utils.rs", "testing.rs"]);
    const isolatedUnitTestFileRe = /^(?:tests?|.*_tests|test_.*)\.rs$/;
    const externalTestModDeclRe =
      /^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(?:tests|[a-z0-9_]+_tests)\s*;/m;
    const srcOrRustRe = /^packages\/[^/]+\/(?:src|rust)\//;
    const binSubdirRe = /^packages\/[^/]+\/(?:src|rust)\/bin\//;
    const pkgIntegrationTestsRe = /^packages\/[^/]+\/tests\//;
    const lineContinuationGolfRe = /\\n\\(?:\r?\n)/;

    const rel = relPath(rsPath).split(path.sep).join("/");
    const base = path.basename(rsPath);
    const content = fs.readFileSync(rsPath, "utf8");

    if (srcOrRustRe.test(rel) && !binSubdirRe.test(rel)) {
      if (isolatedUnitTestFileRe.test(base) && !sharedTestHelperNames.has(base)) {
        violations.push(
          `${rel}: isolated unit-test file in src/rust (co-locate #[cfg(test)] mod tests { ... } in the production module instead)`,
        );
      }
      const declMatch = externalTestModDeclRe.exec(content);
      if (declMatch) {
        violations.push(
          `${rel}: external file-backed unit-test module declaration '${declMatch[0].trim()}' (use inline #[cfg(test)] mod tests { ... })`,
        );
      }
    }
    if (pkgIntegrationTestsRe.test(rel) && base.endsWith("_coverage.rs")) {
      violations.push(
        `${rel}: grab-bag '*_coverage.rs' integration test file (move unit tests to co-located #[cfg(test)] mod tests or name by domain)`,
      );
    }
    if (lineContinuationGolfRe.test(content)) {
      violations.push(
        `${rel}: contains literal '\\n\\\\' line-continuation compression (write readable multi-line statements/assembly)`,
      );
    }
  }

  function checkTsModularFile(tsPath: string, violations: string[]): void {
    const tsSrcRe = /^packages\/[^/]+\/src\//;
    const tsPkgTestsRe = /^packages\/[^/]+\/tests\//;
    const genericTsTestNames = new Set([
      "test.ts",
      "tests.ts",
      "test.tsx",
      "tests.tsx",
      "test.js",
      "tests.js",
      "test.mjs",
      "tests.mjs",
    ]);
    const specTsRe = /\.spec\.(?:ts|tsx|js|mjs)$/;
    const coverageGrabBagTsRe = /coverage[^/]*\.(?:test|spec)\.(?:ts|tsx|js|mjs)$/;
    const lineContinuationGolfRe = /\\n\\(?:\r?\n)/;

    const rel = relPath(tsPath).split(path.sep).join("/");
    const base = path.basename(tsPath);
    const content = fs.readFileSync(tsPath, "utf8");

    if (tsSrcRe.test(rel)) {
      if (rel.includes("/__tests__/")) {
        violations.push(
          `${rel}: isolated '__tests__/' directory in src/ (co-locate '<name>.test.ts(x)' beside its source module instead)`,
        );
      }
      if (genericTsTestNames.has(base)) {
        violations.push(
          `${rel}: generic unpaired '${base}' file in src/ (name co-located unit tests '<module>.test.ts(x)')`,
        );
      }
      if (specTsRe.test(base)) {
        violations.push(
          `${rel}: '*.spec.ts(x)' inside src/ (reserve '*.spec.ts' for tests/e2e/ and use '<module>.test.ts(x)' in src/)`,
        );
      }
    }
    if ((tsSrcRe.test(rel) || tsPkgTestsRe.test(rel)) && coverageGrabBagTsRe.test(base)) {
      violations.push(
        `${rel}: grab-bag '*coverage*.test/spec' test file (co-locate unit tests beside source modules or name integration suites by domain)`,
      );
    }
    if (lineContinuationGolfRe.test(content)) {
      violations.push(
        `${rel}: contains literal '\\n\\\\' line-continuation compression (write readable multi-line statements)`,
      );
    }
  }

  // verify[repo.code.modular-decomposition-and-tests]
  it("test_modular_decomposition_no_isolated_src_unit_tests_or_code_golf", () => {
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
  });

  // verify[repo.code.constant-eval-and-derivations]
  it("test_constant_eval_and_magic_number_guardrails", () => {
    const ruleFile = path.join(REPO_ROOT, ".agents", "rules", "constant-eval-and-magic-numbers.md");
    expect(fs.existsSync(ruleFile)).toBe(true);
    expect(fs.readFileSync(ruleFile, "utf8")).toContain(
      "ref[repo.code.constant-eval-and-derivations]",
    );

    const constEvalRs = path.join(PACKAGES_DIR, "engine", "src", "math", "const_eval.rs");
    expect(fs.existsSync(constEvalRs)).toBe(true);
    const constEvalSrc = fs.readFileSync(constEvalRs, "utf8");
    expect(constEvalSrc).toContain("pub const fn");
    expect(constEvalSrc).toContain("const _: () = assert!(");

    const progressRs = fs.readFileSync(
      path.join(PACKAGES_DIR, "capture", "src", "solver", "progress.rs"),
      "utf8",
    );
    expect(progressRs).toContain("const WORD_MAGIC: usize = WORD_TIER_LTAS + 4;");
    expect(progressRs).toContain("pub const PROGRESS_WIRE_BYTES: usize = PROGRESS_WIRE_WORDS * 4;");

    const progressTs = fs.readFileSync(
      path.join(PACKAGES_DIR, "ui", "src", "host", "trainer-progress.ts"),
      "utf8",
    );
    expect(progressTs).toContain("export const WORD_MAGIC = WORD_TIER_LTAS + 4;");
    expect(progressTs).toContain("export const PROGRESS_WIRE_BYTES = PROGRESS_WIRE_WORDS * 4;");

    const barrelViolations: string[] = [];
    const allowedJsBarrels = new Set([
      "packages/js/src/index.ts",
      "packages/js/src/dsp/index.ts",
      "packages/js/src/parameters/index.ts",
      "packages/js/src/provenance/index.ts",
      "packages/js/src/schema/index.ts",
      "packages/js/src/tempo/index.ts",
      "packages/js/src/web-audio/index.ts",
    ]);
    for (const pkg of ["ui", "studio", "js"]) {
      for (const file of walkFilesByExtensions(
        path.join(PACKAGES_DIR, pkg, "src"),
        new Set([".ts", ".tsx"]),
      )) {
        const base = path.basename(file);
        if (base !== "index.ts" && base !== "index.tsx") {
          continue;
        }
        const rel = relPath(file).split(path.sep).join("/");
        if (pkg !== "js" || !allowedJsBarrels.has(rel)) {
          barrelViolations.push(`${rel}: forbidden internal barrel index file`);
        }
      }
    }
    expect(barrelViolations).toEqual([]);
  });

  // verify[arch.index.subsystems]
  // verify[repo.code.universal-black-box]
  it("test_capture_trainer_blackbox_purity_no_fixture_leakage", () => {
    const blackboxRuleFile = path.join(REPO_ROOT, ".agents", "rules", "universal-black-box.md");
    expect(fs.existsSync(blackboxRuleFile)).toBe(true);
    const blackboxRuleText = fs.readFileSync(blackboxRuleFile, "utf8");
    expect(blackboxRuleText).toContain("ref[arch.pillars.universal-black-box]");
    expect(blackboxRuleText).toContain("ref[repo.code.universal-black-box]");

    const captureSrcDir = path.join(PACKAGES_DIR, "capture", "src");
    const forbiddenTokens = [
      "dnam_fixtures",
      "dnam-fixtures",
      "build_tube_amp_di_seed",
      "build_triband_entropia_seed",
      "DI_LOAD_LF_POLE",
      "unlock_triband_",
    ];
    const violations: string[] = [];
    for (const rsPath of walkFilesByExtensions(captureSrcDir, new Set([".rs"]))) {
      const rel = relPath(rsPath).split(path.sep).join("/");
      const content = fs.readFileSync(rsPath, "utf8");
      for (const token of forbiddenTokens) {
        if (content.includes(token)) {
          violations.push(`${rel}: forbidden white-box token '${token}'`);
        }
      }
    }
    expect(violations).toEqual([]);
  });

  // verify[arch.tiers.maximize-quality-within-budget]
  // verify[qual.tiers.floors-not-stopping-criteria]
  // verify[repo.code.zerocopy-wire-structs]
  // verify[repo.code.shared-nostd-kernels]
  // verify[train.pipeline.webgpu-scans-and-ozaki]
  it("test_sona_zerocopy_and_quality_budget_rules", () => {
    const zerocopyRulePath = path.join(REPO_ROOT, ".agents", "rules", "zerocopy-wire-structs.md");
    expect(fs.existsSync(zerocopyRulePath), "Missing .agents/rules/zerocopy-wire-structs.md").toBe(
      true,
    );
    const zerocopyRuleText = fs.readFileSync(zerocopyRulePath, "utf8");
    expect(zerocopyRuleText).toContain("ref[repo.code.zerocopy-wire-structs]");
    expect(zerocopyRuleText).toContain("ref[repo.code.constant-eval-and-derivations]");

    const sonaSpecPath = path.join(SPECS_DIR, "architecture", "sona_architecture.md");
    const legacyToraSpecPath = path.join(SPECS_DIR, "architecture", "tora_architecture.md");
    expect(
      fs.existsSync(sonaSpecPath),
      "Missing docs/specs/architecture/sona_architecture.md",
    ).toBe(true);
    expect(fs.existsSync(legacyToraSpecPath)).toBe(false);
    const sonaSpecText = fs.readFileSync(sonaSpecPath, "utf8");
    expect(sonaSpecText).toContain("r[arch.tiers.maximize-quality-within-budget]");

    const adr0022Path = path.join(
      ADR_DIR,
      "0022-sona-state-space-operator-and-webgpu-architecture.md",
    );
    expect(fs.existsSync(adr0022Path), "Missing ADR-0022").toBe(true);
    const adr0022Fm = parseFrontmatter(fs.readFileSync(adr0022Path, "utf8"));
    expect(adr0022Fm?.["id"]).toBe("ADR-0022");
    expect(adr0022Fm?.["status"]).toBe("accepted");

    const rules = collectSpecRules();
    for (const reqRule of [
      "arch.tiers.maximize-quality-within-budget",
      "qual.tiers.floors-not-stopping-criteria",
      "repo.code.zerocopy-wire-structs",
      "repo.code.shared-nostd-kernels",
      "train.pipeline.webgpu-scans-and-ozaki",
    ]) {
      expect(rules.has(reqRule), `Missing spec rule definition r[${reqRule}]`).toBe(true);
    }
  });
});
