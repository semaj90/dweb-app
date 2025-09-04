#!/usr/bin/env node
import fs from 'fs';
import path from 'path';

const ROOT = path.resolve(process.cwd(), 'src', 'routes');
const OUT = path.resolve(process.cwd(), 'routes.txt');

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  let files = [];
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name.startsWith('.') || e.name.startsWith('_')) continue;
      files = files.concat(walk(full));
    } else {
      files.push(full);
    }
  }
  return files;
}

function toRoute(file) {
  const rel = path.relative(ROOT, file).split(path.sep).join('/');
  // Only consider page and server route files
  if (!/(?:\+page\.(?:svelte|ts|server\.ts)|\+server\.ts)$/.test(rel)) return null;
  const cleaned = rel.replace(/(\/+page(\.svelte|\.ts|\.server\.ts)$)|(\/+server\.ts$)/, '');
  let route = '/' + cleaned;
  route = route.replace(/\/+/g, '/');
  route = route.replace(/index$/i, '');
  route = route.replace(/\/+$/, '');
  if (route === '') route = '/';
  return route;
}

try {
  if (!fs.existsSync(ROOT)) {
    console.error('Routes directory not found:', ROOT);
    process.exit(1);
  }
  const files = walk(ROOT);
  const routes = new Set();
  for (const f of files) {
    const r = toRoute(f);
    if (r) routes.add(r);
  }
  const sorted = Array.from(routes).sort();
  fs.writeFileSync(OUT, sorted.join('\n'));
  console.log(`Wrote ${sorted.length} routes to ${OUT}`);
} catch (err) {
  console.error(err);
  process.exit(1);
}
