/**
 * Documentation Formatting, `AGENTS.md` Byte Budgets, Table Math, GitHub LaTeX/Mermaid,
 * Relative Links & Anchors, Index, and Root `README.md` Guardrail Tests.
 *
 * verify[repo.docs.agents-md-budgets]
 * verify[repo.docs.markdown-table-math]
 * verify[repo.docs.github-latex-and-mermaid]
 * verify[repo.docs.relative-links-and-anchors]
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
  ADR_DIR,
  DOCS_DIR,
  extractHeadingAnchors,
  MD_LINK_RE,
  PACKAGES_DIR,
  REPO_ROOT,
  REQ_FILE,
  relPath,
  SPEC_RULE_DEF_RE,
  SPECS_DIR,
  stripCode,
  stripFencedCodeBlocks,
  walkMarkdownFiles,
} from "./test-support.js";

const DISPLAY_MATH_RE = /\$\$([\s\S]*?)\$\$/g;
const INLINE_CODE_OR_DISPLAY_MATH_RE = /`+[^`\n]+`+|\$\$[\s\S]*?\$\$/g;
const INLINE_MATH_RE = /(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)/g;
const BACKSLASH_PUNCT_ESCAPE_RE = /\\[^a-zA-Z0-9\s]/;
const FORBIDDEN_MATH_MACROS = ["\\operatorname", "\\multicolumn", "\\texttt", "\\text{--"] as const;
const TEXT_UNDERSCORE_RE = /\\(?:text|mathrm)\{[^}]*_[^}]*\}/;
const INLINE_CODE_RE = /`+[^`\n]+`+/g;
const ALLOWED_INLINE_MATH_NEXT_RE = /[\s,.;:?!]/;
const FRONTMATTER_STRIP_RE = /^---\n[\s\S]*?\n---\n/;
const STANDALONE_RULE_RE = /^\s*`r\[[a-z0-9.-]+\]`\s*$/;
const LIST_ITEM_RE = /^\s*(?:[-*+]|\d+\.)\s+/;
const BANNED_NUMERIC_PILL_RE =
  /^(?:[<>≤≥~±+-]?\s*\d[\d,_./–—×x%\s-]*(?:\s*(?:B|KiB|MiB|GiB|KB|MB|GB|ns|us|µs|ms|s|Hz|kHz|MHz|dB|dBFS|LUFS|ULP|CL|bit|bits|D|ch|taps|samples|oct|ppm))?|v?\d+\.\d+(?:\.\d+)?|\d+-bit|0-ULP)$/i;
const BANNED_PROSE_WORD_PILL_RE =
  /^(?:Lite|Full|Pro|Ultra|Open|Arena\s+\d+(?:\.\.\d+)?|Quad\s+\d+|WebGPU|SHA-256|HMAC-SHA256|Ed25519|X25519|ChaCha20-Poly1305|HTTP\s+Range|SONA|VarPro|Schur|Kautz|FIR|ZOH|SVD|OMP|LTAS|BPTT)$/;

function stripFrontmatterPreserveLines(text: string): string {
  return text.replace(FRONTMATTER_STRIP_RE, (m) => "\n".repeat((m.match(/\n/g) ?? []).length));
}

function normalizeSpecLineForProseMetrics(line: string): string {
  return line.replace(/\]\([^)]*\)/g, "]");
}

function countInlineCodeSpans(line: string): number {
  return (line.match(/`[^`\n]+`/g) ?? []).length;
}

function isSpecProseParagraphBoundary(trimmed: string): boolean {
  return (
    trimmed === "" ||
    trimmed.startsWith("#") ||
    trimmed.startsWith("|") ||
    trimmed.startsWith(">") ||
    trimmed.startsWith("$$") ||
    /^---+$/.test(trimmed) ||
    STANDALONE_RULE_RE.test(trimmed) ||
    LIST_ITEM_RE.test(trimmed)
  );
}

function checkLinePills(rel: string, lineNo: number, line: string, violations: string[]): void {
  if (STANDALONE_RULE_RE.test(line.trim())) return;
  for (const m of line.matchAll(/`([^`\n]+)`/g)) {
    const pill = (m[1] ?? "").trim();
    if (/^0x[0-9a-fA-F_]+$/.test(pill)) continue;
    if (BANNED_NUMERIC_PILL_RE.test(pill) || BANNED_PROSE_WORD_PILL_RE.test(pill)) {
      violations.push(`${rel}:${lineNo}: banned prose/numeric pill '\`${pill}\`'`);
    }
  }
}

function stripDisplayMathPreservingInlineCode(fencedStripped: string): string {
  return fencedStripped.replace(INLINE_CODE_OR_DISPLAY_MATH_RE, (m) =>
    m.startsWith("`") ? m : "\n".repeat((m.match(/\n/g) ?? []).length),
  );
}

function collectDocPillViolations(docPath: string, violations: string[]): void {
  const rel = relPath(docPath);
  const rawBody = stripFrontmatterPreserveLines(fs.readFileSync(docPath, "utf8"));
  const stripped = stripDisplayMathPreservingInlineCode(stripFencedCodeBlocks(rawBody));
  const lines = stripped.split("\n");
  for (let idx = 0; idx < lines.length; idx++) {
    checkLinePills(rel, idx + 1, lines[idx] ?? "", violations);
  }
}

function collectSpecPillAndNestedParenViolations(
  rel: string,
  lines: readonly string[],
  violations: string[],
): void {
  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx] ?? "";
    const trimmed = line.trim();
    if (STANDALONE_RULE_RE.test(trimmed)) continue;
    checkLinePills(rel, idx + 1, line, violations);
    const strippedForParens = line
      .replace(INLINE_CODE_RE, " ")
      .replace(/\$\$[^$]+\$\$/g, " ")
      .replace(INLINE_MATH_RE, " ")
      .replace(/\]\([^)]*\)/g, "]");
    if (!trimmed.startsWith("|") && /\([^()\n]*\([^()\n]*\)/.test(strippedForParens)) {
      violations.push(`${rel}:${idx + 1}: nested parenthetical in prose/list`);
    }
  }
}

function checkSpecListItemLine(
  rel: string,
  lineNo: number,
  trimmed: string,
  violations: string[],
): void {
  const norm = normalizeSpecLineForProseMetrics(trimmed);
  if (norm.length > 240) {
    violations.push(`${rel}:${lineNo}: list item is ${norm.length} chars (> 240)`);
  }
  const pills = countInlineCodeSpans(norm);
  if (pills > 5) {
    violations.push(`${rel}:${lineNo}: list item has ${pills} code pills (> 5)`);
  }
}

function flushSpecProseParagraph(
  rel: string,
  paraStartLine: number,
  paraParts: string[],
  violations: string[],
): number {
  if (paraParts.length === 0) return 0;
  const joined = normalizeSpecLineForProseMetrics(paraParts.join(" "));
  if (joined.length > 650) {
    violations.push(`${rel}:${paraStartLine}: prose paragraph is ${joined.length} chars (> 650)`);
  }
  const pills = countInlineCodeSpans(joined);
  if (pills > 8) {
    violations.push(`${rel}:${paraStartLine}: prose paragraph has ${pills} code pills (> 8)`);
  }
  paraParts.length = 0;
  return 1;
}

function collectSpecProseStructureViolations(specPath: string, violations: string[]): void {
  const rel = relPath(specPath);
  const rawBody = stripFrontmatterPreserveLines(fs.readFileSync(specPath, "utf8"));
  const mathAndCodeStripped = stripDisplayMathPreservingInlineCode(stripFencedCodeBlocks(rawBody));
  const lines = mathAndCodeStripped.split("\n");
  collectSpecPillAndNestedParenViolations(rel, lines, violations);

  let proseParagraphCount = 0;
  let tableDataRowCount = 0;
  let listItemCount = 0;
  let paraStartLine = 0;
  const paraParts: string[] = [];

  for (let idx = 0; idx < lines.length; idx++) {
    const line = lines[idx] ?? "";
    const trimmed = line.trim();
    if (LIST_ITEM_RE.test(line)) {
      proseParagraphCount += flushSpecProseParagraph(rel, paraStartLine, paraParts, violations);
      listItemCount++;
      checkSpecListItemLine(rel, idx + 1, trimmed, violations);
    } else if (trimmed.startsWith("|")) {
      proseParagraphCount += flushSpecProseParagraph(rel, paraStartLine, paraParts, violations);
      if (!/^\|(?:\s*:?-+:?\s*\|)+\s*$/.test(trimmed)) tableDataRowCount++;
    } else if (isSpecProseParagraphBoundary(trimmed)) {
      proseParagraphCount += flushSpecProseParagraph(rel, paraStartLine, paraParts, violations);
    } else {
      if (paraParts.length === 0) paraStartLine = idx + 1;
      paraParts.push(trimmed);
    }
  }
  proseParagraphCount += flushSpecProseParagraph(rel, paraStartLine, paraParts, violations);

  if (proseParagraphCount < 6) {
    violations.push(`${rel}: only ${proseParagraphCount} prose paragraphs (< 6)`);
  }
  if (proseParagraphCount + tableDataRowCount <= listItemCount) {
    violations.push(
      `${rel}: prose+tables (${proseParagraphCount}+${tableDataRowCount}) must exceed list items (${listItemCount})`,
    );
  }
}

function checkInlineMathBoundaries(
  violations: string[],
  rel: string,
  inlineText: string,
  match: RegExpExecArray,
): void {
  const openIdx = match.index;
  const closeIdx = openIdx + match[0].length - 1;
  const lineNo = inlineText.slice(0, openIdx).split("\n").length;
  const snippet = match[0].slice(0, 60);
  if (openIdx > 0) {
    const prevCh = inlineText[openIdx - 1] ?? "";
    if (prevCh !== "" && !/\s/.test(prevCh)) {
      violations.push(`${rel}:${lineNo} (opening '$' preceded by '${prevCh}'): ${snippet}`);
    }
  }
  if (closeIdx + 1 < inlineText.length) {
    const nextCh = inlineText[closeIdx + 1] ?? "";
    if (nextCh !== "" && !ALLOWED_INLINE_MATH_NEXT_RE.test(nextCh)) {
      violations.push(`${rel}:${lineNo} (closing '$' followed by '${nextCh}'): ${snippet}`);
    }
  }
}

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
    violations.push(`${rel} (${kind} unescaped '_' inside text): ${body.trim().slice(0, 80)}`);
  }
  if ((body.match(/\{/g) ?? []).length !== (body.match(/\}/g) ?? []).length) {
    violations.push(`${rel} (${kind} unbalanced braces): ${body.trim().slice(0, 80)}`);
  }
}

function cleanMermaidSource(rawBlock: string): string {
  const withoutFrontmatter = rawBlock.replace(/^\s*---\r?\n[\s\S]*?\r?\n---\r?\n/, "");
  return withoutFrontmatter
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("%%"))
    .join("\n")
    .trimStart();
}

function checkDocMermaidBlocks(violations: string[], rel: string, rawText: string): void {
  const unquotedRe =
    /\b[A-Za-z0-9_]+(?:\s*\[(?!")([^\]]*[():;&][^\]]*)\]|\s*\{(?!")([^}]*[():;&][^}]*)\}|\s*\(\((?!")([^)]*[():;&][^)]*)\)\)|\s*\((?![("])([^)\n]*[():;&][^)\n]*)\))/;
  const edgeLabelRe = /(?:--+|==+|-\.-+)>?\|(?!")([^|\n]*[():;&][^|\n]*)\|/;
  const bidirRe = /<(?:--|==)\s+[^->=\n]+\s+(?:-->|==>)/;
  for (const m of rawText.matchAll(/```mermaid\r?\n([\s\S]*?)```/g)) {
    const cleaned = cleanMermaidSource(m[1] ?? "");
    if (
      (cleaned.startsWith("sequenceDiagram") || cleaned.startsWith("stateDiagram")) &&
      cleaned.includes(";")
    ) {
      violations.push(`${rel}: semicolon inside Mermaid sequenceDiagram/stateDiagram`);
    } else if (cleaned.startsWith("flowchart") || cleaned.startsWith("graph")) {
      const noQuoted = cleaned.replace(/"[^"\n]*"/g, '""');
      if (unquotedRe.test(noQuoted) || edgeLabelRe.test(noQuoted)) {
        violations.push(`${rel}: unquoted special chars in Mermaid flowchart/graph node label`);
      }
      if (bidirRe.test(noQuoted)) {
        violations.push(`${rel}: invalid bidirectional arrow in Mermaid flowchart/graph`);
      }
    }
  }
}

function verifyResolvedLinkTarget(
  mdPath: string,
  lineNo: number,
  rawTarget: string,
  anchorCache: Map<string, Set<string>>,
  brokenLinks: string[],
): void {
  if (/^(?:https?:|mailto:)/.test(rawTarget)) return;
  const hashIdx = rawTarget.indexOf("#");
  const targetFile = rawTarget.startsWith("#")
    ? mdPath
    : path.resolve(path.dirname(mdPath), hashIdx !== -1 ? rawTarget.slice(0, hashIdx) : rawTarget);
  const anchor = rawTarget.startsWith("#")
    ? rawTarget.slice(1)
    : hashIdx !== -1
      ? rawTarget.slice(hashIdx + 1)
      : "";
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

describe("DNAM Documentation Formatting, Links & Progressive Disclosure Guardrails", () => {
  it("test_progressive_disclosure_agents_md_byte_limits", () => {
    const allAgents = walkMarkdownFiles(REPO_ROOT).filter((p) => path.basename(p) === "AGENTS.md");
    const rootAgents = path.join(REPO_ROOT, "AGENTS.md");
    expect(allAgents.includes(rootAgents)).toBe(true);
    expect(fs.statSync(rootAgents).size).toBeLessThan(5000);
    expect(allAgents.filter((p) => p !== rootAgents).length).toBeGreaterThanOrEqual(19);

    for (const agentsPath of allAgents) {
      if (agentsPath !== rootAgents) {
        const size = fs.statSync(agentsPath).size;
        expect(size, `${relPath(agentsPath)} is ${size} B (>= 2,500)`).toBeLessThan(2500);
      }
      const clean = stripFencedCodeBlocks(fs.readFileSync(agentsPath, "utf8"));
      expect(clean).not.toContain("$$");
      expect(clean.match(SPEC_RULE_DEF_RE)).toBeNull();
    }
  });

  it("test_no_inline_latex_math_in_markdown_tables", () => {
    const violations: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const lines = stripFencedCodeBlocks(fs.readFileSync(mdPath, "utf8")).split("\n");
      lines.forEach((line, idx) => {
        if (line.trimStart().startsWith("|") && line.replace(INLINE_CODE_RE, "").includes("$")) {
          violations.push(`${relPath(mdPath)}:${idx + 1}: ${line.trim().slice(0, 100)}`);
        }
      });
    }
    expect(violations).toEqual([]);
  });

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
      const inlineText = clean.replace(
        DISPLAY_MATH_RE,
        (m) => ` ${"\n".repeat((m.match(/\n/g) ?? []).length)}`,
      );
      for (const m of inlineText.matchAll(INLINE_MATH_RE)) {
        checkDocMathBody(violations, rel, "inline math", m[1] ?? "");
        checkInlineMathBoundaries(violations, rel, inlineText, m);
      }
      const residual = inlineText.replace(INLINE_MATH_RE, " ").replace(/\\\$/g, " ");
      if (residual.includes("$")) {
        residual.split("\n").forEach((line, idx) => {
          if (line.includes("$")) {
            violations.push(`${rel}:${idx + 1}: unclosed or multi-line inline '$'`);
          }
        });
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_relative_markdown_links_resolve", () => {
    const anchorCache = new Map<string, Set<string>>();
    const brokenLinks: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const rel = relPath(mdPath);
      const isDocsMd = mdPath.startsWith(DOCS_DIR);
      const lines = stripCode(fs.readFileSync(mdPath, "utf8")).split("\n");
      lines.forEach((line, idx) => {
        for (const m of line.matchAll(MD_LINK_RE)) {
          const label = (m[1] ?? "").trim();
          const target = m[2] ?? "";
          if (isDocsMd && (label.startsWith("../") || label.startsWith("./"))) {
            brokenLinks.push(`${rel}:${idx + 1} -> raw relative path in label '${label}'`);
          }
          verifyResolvedLinkTarget(mdPath, idx + 1, target, anchorCache, brokenLinks);
        }
      });
    }
    expect(brokenLinks).toEqual([]);
  });

  it("test_specs_use_structured_prose_not_bullet_walls_or_pill_soup", () => {
    const violations: string[] = [];
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      collectSpecProseStructureViolations(specPath, violations);
    }
    for (const docPath of [
      REQ_FILE,
      path.join(DOCS_DIR, "index.md"),
      path.join(PACKAGES_DIR, "kernels", "README.md"),
    ]) {
      collectDocPillViolations(docPath, violations);
    }
    expect(violations).toEqual([]);
  });

  it("test_docs_index_links_all_specs_and_adrs", () => {
    const indexText = fs.readFileSync(path.join(DOCS_DIR, "index.md"), "utf8");
    for (const specPath of walkMarkdownFiles(SPECS_DIR)) {
      expect(indexText).toContain(path.relative(DOCS_DIR, specPath).split(path.sep).join("/"));
    }
    for (const entry of fs.readdirSync(ADR_DIR).sort()) {
      if (entry.endsWith(".md") && entry !== "README.md")
        expect(indexText).toContain(`adr/${entry}`);
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
});
