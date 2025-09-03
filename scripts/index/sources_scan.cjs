#!/usr/bin/env node
/** JS wrapper of sources_scan.ts (no TypeScript runtime needed) */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Adjust root: current file at scripts/index; repo root is two levels up
const ROOT = path.resolve(__dirname, '../..');
const OUT_DIR = path.join(ROOT, '.rag-metrics');
const OUT_FILE = path.join(OUT_DIR, 'manifest.json');
// Debug logging for path resolution
if (process.env.DEBUG_SOURCE_SCAN) {
  console.log('[debug] ROOT:', ROOT);
  console.log('[debug] OUT_DIR:', OUT_DIR);
  console.log('[debug] OUT_FILE:', OUT_FILE);
}
const INCLUDE_EXT = { '.go':'go','.ts':'ts','.tsx':'ts','.js':'js','.svelte':'svelte','.md':'md','.mdx':'md' };
const EXCLUDE_DIRS = new Set(['node_modules','dist','build','.git','.svelte-kit','.vscode','bin','cache','embeddings','perf','logs']);
function shouldSkipDir(name){ return EXCLUDE_DIRS.has(name) || name.startsWith('.cache'); }
function walk(dir, acc){
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) { if (shouldSkipDir(entry.name)) continue; walk(path.join(dir, entry.name), acc); }
    else if (entry.isFile()) { const ext = path.extname(entry.name); if (INCLUDE_EXT[ext]) acc.push(path.join(dir, entry.name)); }
  }
}
function sha256(buf){ return crypto.createHash('sha256').update(buf).digest('hex'); }
function main(){
  const files=[]; walk(ROOT, files);
  const manifest = files.map(fp => {
    const rel = path.relative(ROOT, fp).replace(/\\/g,'/');
    const content = fs.readFileSync(fp);
    const hash = sha256(content);
    const idSource = rel + ':' + hash.slice(0,16);
    const id = crypto.createHash('sha1').update(idSource).digest('hex');
    return { id, relPath: rel, lang: INCLUDE_EXT[path.extname(fp)], bytes: content.length, sha256: hash };
  }).sort((a,b)=>a.relPath.localeCompare(b.relPath));
  if (!fs.existsSync(OUT_DIR)) {
    fs.mkdirSync(OUT_DIR, { recursive: true });
    if (process.env.DEBUG_SOURCE_SCAN) console.log('[debug] Created directory', OUT_DIR);
  }
  fs.writeFileSync(OUT_FILE, JSON.stringify({ generatedAt: new Date().toISOString(), count: manifest.length, entries: manifest }, null, 2));
  console.log(`Inventory complete: ${manifest.length} files -> ${path.relative(ROOT, OUT_FILE)}`);
  if (process.env.DEBUG_SOURCE_SCAN) {
    console.log('[debug] Wrote bytes:', fs.statSync(OUT_FILE).size);
  }
}
main();
