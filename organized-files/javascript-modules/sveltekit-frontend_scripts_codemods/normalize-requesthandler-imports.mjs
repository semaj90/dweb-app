import fs from 'fs';
import path from 'path';

function findServerFiles(dir, results = []) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (full.includes('node_modules') || full.includes('.svelte-kit') || full.includes('logs')) continue;
    if (e.isDirectory()) findServerFiles(full, results);
    else if (e.isFile() && e.name === '+server.ts') results.push(full);
  }
  return results;
}

function removeRequestHandlerImports(text) {
  // remove any import lines that include RequestHandler
  return text.replace(/^[ \t]*import[^\n]*RequestHandler[^;]*;\s*$/gim, '');
}

function ensureCanonicalImport(text, canonicalImport) {
  // find the block of leading imports
  const lines = text.split(/\r?\n/);
  let insertIndex = 0;
  while (insertIndex < lines.length && /^\s*import[^;]*;\s*$/.test(lines[insertIndex])) insertIndex++;

  // if canonicalImport already present, do nothing
  if (text.includes(canonicalImport)) return text;

  // insert canonicalImport after last import line
  lines.splice(insertIndex, 0, canonicalImport);
  return lines.join('\n');
}

function makePatchHeader(oldPath, newPath) {
  return `--- ${oldPath}\n+++ ${newPath}\n`;
}

function makeSimpleDiff(oldText, newText) {
  const oldLines = oldText.split(/\r?\n/);
  const newLines = newText.split(/\r?\n/);
  let i = 0;
  while (i < oldLines.length && i < newLines.length && oldLines[i] === newLines[i]) i++;
  let j = oldLines.length - 1;
  let k = newLines.length - 1;
  while (j >= i && k >= i && oldLines[j] === newLines[k]) { j--; k--; }

  const before = oldLines.slice(Math.max(0, i - 2), i).join('\n');
  const after = oldLines.slice(j + 1, Math.min(oldLines.length, j + 1 + 2)).join('\n');

  const removed = oldLines.slice(i, j + 1);
  const added = newLines.slice(i, k + 1);

  let chunk = '';
  chunk += '@@\n';
  if (before) chunk += before + '\n';
  for (const r of removed) chunk += `-${r}\n`;
  for (const a of added) chunk += `+${a}\n`;
  if (after) chunk += after + '\n';
  return chunk;
}

async function main() {
  const root = process.cwd();
  const targetDir = path.join(root, 'src');
  const files = findServerFiles(targetDir);
  const patches = [];

  for (const f of files) {
    const rel = path.relative(root, f).replace(/\\/g, '/');
    const original = fs.readFileSync(f, 'utf8');
    let modified = original;

    // Remove duplicate RequestHandler import lines
    modified = removeRequestHandlerImports(modified);

    // Decide canonical import: prefer './$types' for route handlers
    const canonical = "import type { RequestHandler } from './$types';";

    // If file doesn't live under routes (rare), fallback to @sveltejs/kit
    // But our scan is under src (routes), so use './$types'
    modified = ensureCanonicalImport(modified, canonical);

    if (modified !== original) {
      const header = makePatchHeader(rel, rel);
      const diff = makeSimpleDiff(original, modified);
      patches.push(header + diff + '\n');
    }
  }

  if (patches.length === 0) {
    console.log('No changes detected by codemod.');
    return;
  }

  const outDir = path.join(root, 'logs');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `dry-run-requesthandler-normalize-codemod.patch`);
  fs.writeFileSync(outPath, patches.join('\n'));
  console.log('Dry-run patch written to', outPath);
}

main().catch(err => { console.error(err); process.exit(1); });
