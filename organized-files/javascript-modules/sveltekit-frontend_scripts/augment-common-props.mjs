#!/usr/bin/env node
/*
 Phase 1 – Common Props Augmentation Script
 -----------------------------------------
 Scans Svelte components in high-value directories, injects `import type { CommonProps }` and merges into existing prop interfaces or creates one.
 Safe, idempotent: skips if already includes CommonProps.
*/
import fs from 'node:fs';
import path from 'node:path';
// glob v11+ provides named exports; use sync for simplicity
import { globSync } from 'glob';

const ROOT = process.cwd();
const TARGET_GLOBS = [
  'src/lib/components/**/*.{svelte,ts}',
  'src/lib/ui/**/*.{svelte,ts}',
  'src/lib/vendor/bits-ui-fallback/**/*.{svelte,ts}'
];

function processFile(file){
  const code = fs.readFileSync(file,'utf8');
  if(/CommonProps/.test(code)) return false; // already done
  let updated = code;
  const importLine = "import type { CommonProps } from '$lib/types/common-props';";
  if(file.endsWith('.svelte')){
    // Insert import after first script tag open
    updated = updated.replace(/<script(.*?)>/, (m)=> m + '\n' + importLine + '\n');
    // Heuristic: find interface Props / export interface.*Props
    if(/interface\s+Props/.test(updated)){
      updated = updated.replace(/interface\s+Props\s*{/, 'interface Props extends CommonProps {');
    } else if(/export\s+let\s+/.test(updated)) {
      // Legacy export lets: we won't refactor here; just append comment for future codemod.
      updated += '\n<!-- TODO: migrate export lets to $props(); CommonProps assumed. -->\n';
    } else {
      // Append a Props interface for future typing
      updated += '\n<script lang="ts">\n' + importLine + '\ninterface Props extends CommonProps {}\n</script>\n';
    }
  } else if(file.endsWith('.ts')) {
    // Add import at top
    updated = importLine + '\n' + updated;
    if(/interface\s+\w+Props\s*{/.test(updated)){
      updated = updated.replace(/interface\s+(\w+Props)\s*{/, 'interface $1 extends CommonProps {');
    }
  }
  if(updated !== code){
    fs.writeFileSync(file, updated, 'utf8');
    return true;
  }
  return false;
}

let modified = 0;
for(const pattern of TARGET_GLOBS){
  const files = globSync(pattern, { cwd: ROOT, absolute:true, nodir:true });
  for(const f of files){
    try { if(processFile(f)) modified++; } catch(e){ console.error('[props] Failed', f, e.message); }
  }
}
console.log(`[props] Augmentation complete. Modified ${modified} files.`);
