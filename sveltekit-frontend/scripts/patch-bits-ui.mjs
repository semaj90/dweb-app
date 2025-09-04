#!/usr/bin/env node
import fs from 'fs';
import path from 'path';

const root = path.resolve(process.cwd(), 'node_modules', 'bits-ui', 'dist');

function walk(dir) {
  const results = [];
  const list = fs.readdirSync(dir, { withFileTypes: true });
  for (const d of list) {
    const full = path.join(dir, d.name);
    if (d.isDirectory()) results.push(...walk(full));
    else if (d.isFile() && /\.(js|mjs|svelte|cjs|ts)$/.test(d.name)) results.push(full);
  }
  return results;
}

if (!fs.existsSync(root)) {
  console.error(`bits-ui dist not found at: ${root}`);
  process.exit(1);
}

const files = walk(root);
let changed = 0;

const propsRegex = /\$props\.([A-Za-z0-9_$]+)\(\)/g;

for (const f of files) {
  let src = fs.readFileSync(f, 'utf8');
  if (!propsRegex.test(src)) continue;
  const backup = `${f}.orig.bak`;
  if (!fs.existsSync(backup)) fs.writeFileSync(backup, src, 'utf8');
  const replaced = src.replace(propsRegex, '$props().$1');
  fs.writeFileSync(f, replaced, 'utf8');
  console.log(`Patched: ${path.relative(process.cwd(), f)}`);
  changed++;
}

console.log(`\nDone. Files changed: ${changed}. Backups (*.orig.bak) created next to each modified file.`);
if (changed === 0) process.exitCode = 2;
