#!/usr/bin/env node
/**
 * Advanced AST Fix Script (ts-morph based placeholder)
 * Scans for common easy-to-fix TypeScript issues and applies safe edits.
 * Currently stubs logic: add // TODO markers where a missing symbol is detected.
 */
import fs from 'fs';
import path from 'path';

const ROOT = path.resolve('sveltekit-frontend/src');
const ISSUE_PATTERNS = [
  { regex: /Cannot find name '([A-Z][A-Za-z0-9_]*)'/g, fix: (file, sym) => `// TODO: declare or import ${sym}\n` },
  { regex: /Property '([a-zA-Z0-9_]+)' does not exist on type 'unknown'/g, fix: (file, prop) => `// TODO: narrow unknown before accessing .${prop}\n` }
];

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    const st = fs.statSync(full);
    if (st.isDirectory()) walk(full).forEach(f=>out.push(f));
    else if (/\.(ts|svelte)$/.test(entry)) out.push(full);
  }
  return out;
}

function applyFixes(file) {
  const content = fs.readFileSync(file,'utf8');
  let modified = content;
  let added = '';
  for (const pat of ISSUE_PATTERNS) {
    let m;
    while ((m = pat.regex.exec(content))) {
      added += pat.fix(file, m[1]);
    }
  }
  if (added) {
    fs.writeFileSync(file, added + '\n' + modified);
    return true;
  }
  return false;
}

function main(){
  const files = walk(ROOT);
  let fixed = 0;
  for (const f of files) {
    if (applyFixes(f)) fixed++;
  }
  console.log(`🧠 Advanced AST placeholder fixes applied to ${fixed} files`);
}

main();
