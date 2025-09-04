#!/usr/bin/env node
import fs from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const glob = require('glob');

const root = process.cwd();
const pattern = 'src/**/*.svelte';

let patched = [];
for (const fp of glob.sync(pattern, { cwd: root, absolute: true })) {
  let s = fs.readFileSync(fp, 'utf8');
  const before = s;
  // Replace on<event>|modifier (no colon) with on:<event>|modifier
  s = s.replace(/(\s)on([a-zA-Z]+)\|/g, (m, ws, ev) => `${ws}on:${ev}|`);
  if (s !== before) {
    fs.writeFileSync(fp, s, 'utf8');
    patched.push(fp);
    console.log('Patched:', fp);
  }
}
console.log('Inserted missing colon for on| modifiers in', patched.length, 'files');
process.exit(0);
