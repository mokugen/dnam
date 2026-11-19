/**
 * Shared filesystem, Markdown/YAML/TOML parsing, Rust lexical stripping, and spec-registry
 * helpers for DNAM workspace architectural & documentation guardrail suites.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(THIS_DIR, "..");
export const DOCS_DIR = path.join(REPO_ROOT, "docs");
export const SPECS_DIR = path.join(DOCS_DIR, "specs");
export const ADR_DIR = path.join(DOCS_DIR, "adr");
export const REQ_FILE = path.join(DOCS_DIR, "requirements", "system_goals_and_requirements.md");
export const PACKAGES_DIR = path.join(REPO_ROOT, "packages");
export const TESTS_DIR = path.join(REPO_ROOT, "tests");
export const SCRIPTS_DIR = path.join(REPO_ROOT, "scripts");

export const EXCLUDED_DIRS = new Set([
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

export const SPEC_RULE_DEF_RE =
  /`r\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]`|^[ \t]*r\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\][ \t]*$/gm;
export const SPEC_REF_TOKEN_RE = /\bref\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]/g;
export const SPEC_VERIFY_TOKEN_RE = /\bverify\[([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\]/g;
export const MALFORMED_SPEC_MARKER_RE = /(?:^|[^a-zA-Z0-9_])(r|ref|verify)\[([^\]]+)\]/gm;
export const VALID_RULE_ID_RE = /^[a-z0-9-]+(?:\.[a-z0-9-]+){2,}$/;

export const REQ_TABLE_RE = /^\|\s*\*\*`(REQ-\d+\.\d+)`\*\*\s*\|/gm;
export const REQ_TOKEN_RE = /\bREQ-\d+\.\d+\b/g;
export const ADR_TOKEN_RE = /\bADR-\d{4}\b/g;
export const MD_LINK_RE = /(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
export const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/gm;

const FRONTMATTER_RE = /^---\n([\s\S]*?)\n---\n/;
const FENCED_BLOCK_RE = /^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1[ \t]*$/gm;
const INLINE_CODE_RE = /`+[^`\n]+`+/g;

export const GUARDRAIL_TEST_FILENAMES = new Set([
  "test-support.ts",
  "docs_registry_guardrails.test.ts",
  "docs_formatting_guardrails.test.ts",
  "code_guardrails.test.ts",
  "architecture_guardrails.test.ts",
]);

export function relPath(filePath: string): string {
  return path.relative(REPO_ROOT, filePath);
}

export function isGuardrailFile(filePath: string): boolean {
  return (
    path.dirname(filePath) === TESTS_DIR && GUARDRAIL_TEST_FILENAMES.has(path.basename(filePath))
  );
}

export function walkMarkdownFiles(root: string): string[] {
  const results: string[] = [];
  function recurse(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (EXCLUDED_DIRS.has(entry.name)) continue;
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) recurse(fullPath);
      else if (entry.isFile() && entry.name.endsWith(".md")) results.push(fullPath);
    }
  }
  recurse(root);
  return results.sort();
}

export function walkFilesByExtensions(root: string, extensions: ReadonlySet<string>): string[] {
  const results: string[] = [];
  if (!fs.existsSync(root)) return results;
  function recurse(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (EXCLUDED_DIRS.has(entry.name)) continue;
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) recurse(fullPath);
      else if (entry.isFile() && extensions.has(path.extname(entry.name))) results.push(fullPath);
    }
  }
  recurse(root);
  return results.sort();
}

export function stripFencedCodeBlocks(text: string): string {
  return text.replace(FENCED_BLOCK_RE, (match) => "\n".repeat((match.match(/\n/g) ?? []).length));
}

export function stripCode(text: string): string {
  return stripFencedCodeBlocks(text).replace(INLINE_CODE_RE, " ");
}

export function parseFrontmatter(text: string): Record<string, string | string[]> | null {
  const match = FRONTMATTER_RE.exec(text);
  if (!match || match[1] === undefined) return null;
  const data: Record<string, string | string[]> = {};
  let currentListKey: string | null = null;
  for (const line of match[1].split("\n")) {
    const stripped = line.trim();
    if (!stripped || stripped.startsWith("#")) continue;
    if (stripped.startsWith("- ") && currentListKey !== null) {
      const existing = data[currentListKey];
      if (Array.isArray(existing)) {
        existing.push(
          stripped
            .slice(2)
            .trim()
            .replace(/^['"]|['"]$/g, ""),
        );
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

export function githubSlugify(headingText: string): string {
  let text = headingText
    .trim()
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`/g, "")
    .toLowerCase();
  const kept: string[] = [];
  for (const ch of text) {
    if (/[\p{L}\p{N}_-]/u.test(ch)) kept.push(ch);
    else if (ch === " " || ch === "\t") kept.push(" ");
  }
  text = kept.join("");
  return text.replace(/[ \t]/g, "-");
}

export function extractHeadingAnchors(mdText: string): Set<string> {
  const clean = stripFencedCodeBlocks(mdText);
  const anchors = new Set<string>();
  const counts = new Map<string, number>();
  for (const match of clean.matchAll(HEADING_RE)) {
    const base = githubSlugify(match[2] ?? "");
    if (!base) continue;
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

export function collectSpecRules(): Map<string, [string, number][]> {
  const rules = new Map<string, [string, number][]>();
  for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
    const lines = stripFencedCodeBlocks(fs.readFileSync(specPath, "utf8")).split("\n");
    for (let idx = 0; idx < lines.length; idx++) {
      for (const m of (lines[idx] ?? "").matchAll(SPEC_RULE_DEF_RE)) {
        const ruleId = m[1] ?? m[2] ?? "";
        const list = rules.get(ruleId) ?? [];
        list.push([relPath(specPath), idx + 1]);
        rules.set(ruleId, list);
      }
    }
  }
  return rules;
}

export function collectSourceRefTokens(): [string, string, number][] {
  const refs: [string, string, number][] = [];
  const exts = new Set([".rs", ".ts", ".tsx", ".mjs", ".md"]);
  const templateRel = relPath(path.join(DOCS_DIR, "templates"));
  for (const root of [PACKAGES_DIR, DOCS_DIR, TESTS_DIR]) {
    for (const filePath of walkFilesByExtensions(root, exts)) {
      const rel = relPath(filePath);
      if (isGuardrailFile(filePath) || rel.startsWith(templateRel)) continue;
      const lines = fs.readFileSync(filePath, "utf8").split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        for (const m of (lines[idx] ?? "").matchAll(SPEC_REF_TOKEN_RE)) {
          refs.push([m[1] ?? "", rel, idx + 1]);
        }
      }
    }
  }
  return refs;
}

export function collectTestVerifyTokens(): [string, string, number][] {
  const verifies: [string, string, number][] = [];
  for (const root of [TESTS_DIR, PACKAGES_DIR]) {
    for (const filePath of walkFilesByExtensions(root, new Set([".rs", ".ts", ".tsx"]))) {
      const lines = fs.readFileSync(filePath, "utf8").split("\n");
      for (let idx = 0; idx < lines.length; idx++) {
        for (const m of (lines[idx] ?? "").matchAll(SPEC_VERIFY_TOKEN_RE)) {
          verifies.push([m[1] ?? "", relPath(filePath), idx + 1]);
        }
      }
    }
  }
  return verifies;
}

export interface SpecScopeRuleEntry {
  scope: string;
  reason?: string;
}

export function parseSpecScopeToml(text: string): Map<string, SpecScopeRuleEntry> {
  const rules = new Map<string, SpecScopeRuleEntry>();
  let currentRuleId: string | null = null;
  let currentEntry: SpecScopeRuleEntry | null = null;
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const headerMatch = /^\[rules\."([^"]+)"\]$/.exec(line);
    if (headerMatch?.[1] !== undefined) {
      if (currentRuleId !== null && currentEntry !== null) rules.set(currentRuleId, currentEntry);
      currentRuleId = headerMatch[1];
      currentEntry = { scope: "" };
      continue;
    }
    if (currentEntry === null) continue;
    const scopeMatch = /^scope\s*=\s*"([^"]+)"/.exec(line);
    if (scopeMatch?.[1] !== undefined) {
      currentEntry.scope = scopeMatch[1];
      continue;
    }
    const reasonMatch = /^reason\s*=\s*"([^"]+)"/.exec(line);
    if (reasonMatch?.[1] !== undefined) currentEntry.reason = reasonMatch[1];
  }
  if (currentRuleId !== null && currentEntry !== null) rules.set(currentRuleId, currentEntry);
  return rules;
}

export function normalizeSpecRelPath(rawSpecRef: string): string {
  const trimmed = rawSpecRef.trim();
  if (trimmed.startsWith("docs/")) return trimmed.slice("docs/".length);
  if (trimmed.startsWith("../")) return trimmed.slice("../".length);
  return trimmed;
}

function isEscapedBackslashRun(text: string, idx: number): boolean {
  let bs = 0;
  while (idx > bs && text[idx - 1 - bs] === "\\") bs++;
  return (bs & 1) === 1;
}

function stripTomlComment(line: string): string {
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === "'" && !inDouble) inSingle = !inSingle;
    else if (ch === '"' && !inSingle && !isEscapedBackslashRun(line, i)) inDouble = !inDouble;
    else if (ch === "#" && !inSingle && !inDouble) return line.slice(0, i);
  }
  return line;
}

function extractTomlTableHeader(clean: string): string {
  const isArray = clean.startsWith("[[");
  const openLen = isArray ? 2 : 1;
  const valid = isArray
    ? clean.endsWith("]]") && clean.length > 4
    : clean.endsWith("]") && clean.length > 2;
  if (!valid) throw new Error(`Malformed TOML table header: ${clean}`);
  const inner = clean.slice(openLen, clean.length - openLen);
  let inS = false;
  let inD = false;
  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i];
    if (ch === "'" && !inD) inS = !inS;
    else if (ch === '"' && !inS && !isEscapedBackslashRun(inner, i)) inD = !inD;
    else if (!inS && !inD && (ch === "[" || ch === "]")) {
      throw new Error(`Unquoted bracket inside TOML table header: ${clean}`);
    }
  }
  if (inS || inD) throw new Error(`Unterminated quote inside TOML table header: ${clean}`);
  return inner.trim();
}

function updateTomlLineDepths(clean: string, bStart: number, brStart: number): [number, number] {
  let inS = false;
  let inD = false;
  let bDepth = bStart;
  let brDepth = brStart;
  for (let i = 0; i < clean.length; i++) {
    const ch = clean[i];
    if (ch === "'" && !inD) inS = !inS;
    else if (ch === '"' && !inS && !isEscapedBackslashRun(clean, i)) inD = !inD;
    else if (!inS && !inD) {
      if (ch === "[") bDepth++;
      else if (ch === "]") bDepth = Math.max(0, bDepth - 1);
      else if (ch === "{") brDepth++;
      else if (ch === "}") brDepth = Math.max(0, brDepth - 1);
    }
  }
  return [bDepth, brDepth];
}

export function parseTomlSections(tomlText: string): Array<{ header: string; body: string }> {
  if (/'''|"""/.test(tomlText)) {
    throw new Error("Cargo.toml must not use multi-line triple-quoted strings");
  }
  const sections: Array<{ header: string; body: string }> = [];
  let currentHeader = "";
  let currentLines: string[] = [];
  let bracketDepth = 0;
  let braceDepth = 0;
  for (const rawLine of tomlText.split("\n")) {
    const clean = stripTomlComment(rawLine).trim();
    if (!clean) continue;
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
  if (!isRaw && !isByteRaw) return null;
  let hashes = 0;
  let j = isByteRaw ? i + 2 : i + 1;
  while (src[j] === "#") {
    hashes++;
    j++;
  }
  if (src[j] !== '"') return null;
  const closing = `"${"#".repeat(hashes)}`;
  const endIdx = src.indexOf(closing, j + 1);
  if (endIdx === -1) return null;
  const nlCount = (src.slice(j + 1, endIdx).match(/\n/g) ?? []).length;
  return [`""${"\n".repeat(nlCount)}`, endIdx + closing.length];
}

function matchEscapedRustCharEnd(src: string, q: number): number {
  const esc = src[q + 2] ?? "";
  if (esc === "x") return src[q + 5] === "'" ? q + 6 : -1;
  if (esc === "u" && src[q + 3] === "{") {
    const braceIdx = src.indexOf("}", q + 4);
    return braceIdx !== -1 && braceIdx <= q + 12 && src[braceIdx + 1] === "'" ? braceIdx + 2 : -1;
  }
  return esc !== "" && esc !== "\n" && esc !== "\r" && src[q + 3] === "'" ? q + 4 : -1;
}

function matchRustCharLiteralEnd(src: string, i: number): number {
  const q = src[i] === "b" && src[i + 1] === "'" ? i + 1 : i;
  const c1 = src[q + 1] ?? "";
  if (c1 === "\\") return matchEscapedRustCharEnd(src, q);
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
  return [`${out}"`, i < src.length ? i + 1 : i];
}

function tryConsumeRustComment(src: string, i: number): [string, number] | null {
  if (src[i] === "/" && src[i + 1] === "/") {
    let j = i;
    while (j < src.length && src[j] !== "\n") j++;
    return ["", j];
  }
  if (src[i] === "/" && src[i + 1] === "*") return skipRustBlockComment(src, i);
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

export function stripRustCommentsAndStrings(src: string): string {
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

export function stripRustCommentsOnly(src: string): string {
  let out = "";
  let i = 0;
  while (i < src.length) {
    const commentToken = tryConsumeRustComment(src, i);
    if (commentToken) {
      out += commentToken[0];
      i = commentToken[1];
      continue;
    }
    const litToken = tryConsumeRustStringOrChar(src, i);
    if (litToken) {
      out += src.slice(i, litToken[1]);
      i = litToken[1];
      continue;
    }
    out += src[i] ?? "";
    i++;
  }
  return out;
}

export interface BraceBlockTracker {
  awaitingOpenOrSemi: boolean;
  depth: number;
}

export function createBraceTracker(): BraceBlockTracker {
  return { awaitingOpenOrSemi: false, depth: 0 };
}

/**
 * Tracks brace-delimited items (`#[cfg(test)]`, `#[cfg(not(target_arch = "spirv"))]`),
 * counting same-line braces/semicolons so single-line `#[cfg(test)] mod tests {` works accurately.
 */
export function advanceAttributedBlockTracker(
  strippedLine: string,
  state: BraceBlockTracker,
  trigger: () => boolean,
): boolean {
  const trimmed = strippedLine.trim();
  if (!state.awaitingOpenOrSemi && state.depth <= 0) {
    if (!trigger()) {
      return false;
    }
    state.awaitingOpenOrSemi = true;
  }
  const opens = (strippedLine.match(/\{/g) ?? []).length;
  const closes = (strippedLine.match(/\}/g) ?? []).length;
  state.depth += opens - closes;
  if (state.depth > 0) {
    state.awaitingOpenOrSemi = false;
  } else if (
    state.depth <= 0 &&
    (trimmed.endsWith(";") || (opens > 0 && closes > 0 && trimmed.endsWith("}")))
  ) {
    state.awaitingOpenOrSemi = false;
    state.depth = 0;
    return true;
  }
  if (state.depth < 0) {
    state.depth = 0;
  }
  return true;
}

export function isCfgTestAttributeLine(trimmed: string): boolean {
  return /^#\s*\[\s*cfg\s*\(\s*(?:test\b|all\s*\([^)]*\btest\b)/.test(trimmed);
}
