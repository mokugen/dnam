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
  REPO_ROOT,
  relPath,
  SPEC_RULE_DEF_RE,
  SPECS_DIR,
  stripCode,
  stripFencedCodeBlocks,
  walkMarkdownFiles,
} from "./test-support.js";

const DISPLAY_MATH_RE = /\$\$([\s\S]*?)\$\$/g;
const INLINE_MATH_RE = /(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)/g;
const BACKSLASH_PUNCT_ESCAPE_RE = /\\[^a-zA-Z0-9\s]/;
const FORBIDDEN_MATH_MACROS = ["\\operatorname", "\\multicolumn", "\\texttt", "\\text{--"] as const;
const TEXT_UNDERSCORE_RE = /\\(?:text|mathrm)\{[^}]*_[^}]*\}/;
const INLINE_CODE_RE = /`+[^`\n]+`+/g;

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

function checkDocMermaidBlocks(violations: string[], rel: string, rawText: string): void {
  const unquotedRe = /\b[A-Za-z0-9_]+\[(?!")([^\]]*[():][^\]]*)\]/;
  const bidirRe = /<(?:--|==)\s+[^->=\n]+\s+(?:-->|==>)/;
  for (const m of rawText.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
    const block = m[1] ?? "";
    const trimmed = block.trimStart();
    if (trimmed.startsWith("sequenceDiagram") && block.includes(";")) {
      violations.push(`${rel}: semicolon inside Mermaid sequenceDiagram`);
    } else if (trimmed.startsWith("flowchart")) {
      if (unquotedRe.test(block)) {
        violations.push(`${rel}: unquoted special chars in Mermaid flowchart node label`);
      }
      if (bidirRe.test(block)) {
        violations.push(`${rel}: invalid bidirectional arrow in Mermaid flowchart`);
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
      for (const m of clean.replace(DISPLAY_MATH_RE, " ").matchAll(INLINE_MATH_RE)) {
        checkDocMathBody(violations, rel, "inline math", m[1] ?? "");
      }
    }
    expect(violations).toEqual([]);
  });

  it("test_relative_markdown_links_resolve", () => {
    const anchorCache = new Map<string, Set<string>>();
    const brokenLinks: string[] = [];
    for (const mdPath of walkMarkdownFiles(REPO_ROOT)) {
      const lines = stripCode(fs.readFileSync(mdPath, "utf8")).split("\n");
      lines.forEach((line, idx) => {
        for (const m of line.matchAll(MD_LINK_RE)) {
          verifyResolvedLinkTarget(mdPath, idx + 1, m[2] ?? "", anchorCache, brokenLinks);
        }
      });
    }
    expect(brokenLinks).toEqual([]);
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
