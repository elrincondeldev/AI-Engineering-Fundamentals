#!/usr/bin/env node
// Parse every ```mermaid block in the course Markdown with the real Mermaid parser.
//
// Setup once:   npm ci --prefix tools
// Usage:        node tools/check_mermaid.mjs [files or folders...]   (default: whole repository)
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SKIP = new Set([".git", ".venv", "node_modules", "outputs", "__pycache__", ".pytest_cache"]);

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.DOMParser = dom.window.DOMParser;
globalThis.Element = dom.window.Element;
const { default: mermaid } = await import("mermaid");
mermaid.initialize({ startOnLoad: false });

function walk(path, out) {
  const st = statSync(path);
  if (st.isFile()) {
    if (path.endsWith(".md")) out.push(path);
    return out;
  }
  for (const name of readdirSync(path)) {
    if (SKIP.has(name)) continue;
    walk(join(path, name), out);
  }
  return out;
}

function blocks(text) {
  const lines = text.split("\n");
  const found = [];
  let fence = null;
  let start = 0;
  let body = [];
  let lang = "";
  lines.forEach((line, i) => {
    const m = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (fence === null) {
      if (m) { fence = m[1]; lang = m[2].trim().split(" ")[0].toLowerCase(); start = i + 1; body = []; }
    } else if (m && m[1][0] === fence[0] && m[1].length >= fence.length && !m[2].trim()) {
      if (lang === "mermaid") found.push({ start, src: body.join("\n") });
      fence = null;
    } else {
      body.push(line);
    }
  });
  return found;
}

const targets = process.argv.slice(2).map((p) => resolve(p));
const files = (targets.length ? targets : [REPO]).flatMap((p) => walk(p, []))
  .filter((f) => !relative(REPO, f).startsWith("datasets/kb/"));
let total = 0;
let failed = 0;
for (const file of files.sort()) {
  for (const { start, src } of blocks(readFileSync(file, "utf8"))) {
    total += 1;
    try {
      await mermaid.parse(src);
    } catch (err) {
      failed += 1;
      const msg = String(err?.message ?? err).split("\n").slice(0, 3).join(" | ");
      console.log(`❌ ${relative(REPO, file)}:${start}: ${msg}`);
    }
  }
}
console.log(`\n${total} mermaid diagram(s) checked, ${failed} failed.`);
process.exit(failed ? 1 : 0);
