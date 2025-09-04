#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const glob = require('glob');

const root = process.cwd();
const pattern = 'src/**/*.svelte';

let patchedFiles = [];

const files = glob.sync(pattern, { cwd: root, absolute: true });
for (const fp of files) {
  let content = fs.readFileSync(fp, 'utf8');

  // Replace attribute names that contain a `|` modifier and are not already prefixed with `on`.
  // Example: ` submit|preventDefault={createUser}` -> ` onsubmit|preventDefault={createUser}`
  const before = content;

  content = content.replace(/(\s)([A-Za-z][\w-]*\|[^\s=>]+)=/g, (m, ws, attr) => {
    if (attr.startsWith('on')) return m; // already has on
    // don't accidentally prefix data- or aria- attributes (they won't contain '|') but keep guard
    if (/^(data-|aria-)/.test(attr)) return m;
    return ws + 'on' + attr + '=';
  });

  if (content !== before) {
    fs.writeFileSync(fp, content, 'utf8');
    patchedFiles.push(fp);
    console.log('Patched:', fp);
  }
}

console.log(`Fixed missing 'on' prefix in ${patchedFiles.length} files.`);
if (patchedFiles.length > 0) process.exit(0);
process.exit(0);
