#!/usr/bin/env ts-node
/**
 * Source Inventory Scanner (T5)
 * Enumerates repository source files (Go, TS/TSX, Svelte, Markdown) excluding build artifacts.
 * Emits manifest at: .rag-metrics/manifest.json
 */
import * as fs from 'fs';
import * as path from 'path';
import crypto from 'crypto';

interface ManifestEntry {
  id: string; // stable hash of relPath + sha256 content
  relPath: string;
  lang: string;
  bytes: number;
  sha256: string;
}

const ROOT = path.resolve(__dirname, '../../..');
const OUT_DIR = path.join(ROOT, '.rag-metrics');
const OUT_FILE = path.join(OUT_DIR, 'manifest.json');

const INCLUDE_EXT: Record<string,string> = {
  '.go': 'go', '.ts': 'ts', '.tsx': 'ts', '.js': 'js', '.svelte': 'svelte', '.md': 'md', '.mdx': 'md'
};

const EXCLUDE_DIRS = new Set([
  'node_modules','dist','build','.git','.svelte-kit','.vscode','bin','cache','embeddings','perf','logs'
]);

function shouldSkipDir(name: string) {
  return EXCLUDE_DIRS.has(name) || name.startsWith('.cache');
}

function walk(dir: string, acc: string[]) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (shouldSkipDir(entry.name)) continue;
      walk(path.join(dir, entry.name), acc);
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name);
      if (INCLUDE_EXT[ext]) {
        acc.push(path.join(dir, entry.name));
      }
    }
  }
}

function sha256(buf: Buffer) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function main() {
  const files: string[] = [];
  walk(ROOT, files);
  const manifest: ManifestEntry[] = files.map(fp => {
    const rel = path.relative(ROOT, fp).replace(/\\/g,'/');
    const content = fs.readFileSync(fp);
    const hash = sha256(content);
    const idSource = rel + ':' + hash.slice(0,16);
    const id = crypto.createHash('sha1').update(idSource).digest('hex');
    return { id, relPath: rel, lang: INCLUDE_EXT[path.extname(fp)], bytes: content.length, sha256: hash };
  });
  manifest.sort((a,b)=>a.relPath.localeCompare(b.relPath));
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify({ generatedAt: new Date().toISOString(), count: manifest.length, entries: manifest }, null, 2));
  console.log(`Inventory complete: ${manifest.length} files -> ${path.relative(ROOT, OUT_FILE)}`);
}

main();
