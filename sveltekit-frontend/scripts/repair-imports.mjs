#!/usr/bin/env node
/**
 * Automated Import & Header Repair Script (v2)
 * Enhancements:
 *  - Scans ALL +server.ts files under src/routes
 *  - Repairs orphaned import fragments & bare default imports
 *  - Removes lines starting with stray commas or orphaned markers
 *  - Rejoins multi-line imports / symbol lists
 *  - Injects missing `import type { RequestHandler }` when POST/GET/etc use it
 *  - Normalizes spacing & collapses excessive blank lines
 *  - Idempotent (safe to re-run)
 */
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const ROUTES_DIR = path.join(ROOT, 'sveltekit-frontend', 'src', 'routes');

/** Recursively discover +server.ts files */
function discoverServerFiles(dir, acc = []){
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for(const e of entries){
    const full = path.join(dir, e.name);
    if(e.isDirectory()) discoverServerFiles(full, acc);
    else if(e.isFile() && e.name === '+server.ts') acc.push(full);
  }
  return acc;
}

function needsRequestHandlerImport(source){
  const uses = /:\s*RequestHandler\b/.test(source);
  const hasImport = /import\s+type\s+{[^}]*RequestHandler[^}]*}\s+from\s+'@sveltejs\/kit'/.test(source) || /import\s+{[^}]*RequestHandler[^}]*}\s+from\s+'@sveltejs\/kit'/.test(source);
  return uses && !hasImport;
}

function repairFile(filePath){
  const originalText = fs.readFileSync(filePath,'utf8');
  const original = originalText.split(/\r?\n/);
  const repaired = [];
  let importBuffer = [];
  const flushImportBuffer = ()=>{
    if(!importBuffer.length) return;
    let joined = importBuffer.join(' ')
      .replace(/\s+/g,' ') // collapse whitespace
      .replace(/,\s*}/g,' }')
      .replace(/import\s+\{/g,'import {');
    // ensure semicolon
    if(!/;\s*$/.test(joined)) joined += ';';
    repaired.push(joined.trim());
    importBuffer = [];
  };

  for(let i=0;i<original.length;i++){
    let line = original[i];

    // Remove orphan markers entirely
    if(/\/\/\s*Orphaned content:/.test(line)) continue;

    // Strip leading solitary commas or comma-prefixed lines
    if(/^\s*,\s*$/.test(line)) continue;
    if(/^\s*,\s*/.test(line)) line = line.replace(/^\s*,\s*/,'');

    // Convert bare default import patterns e.g. crypto from 'crypto';
    if(/^[A-Za-z_$][\w$]*\s+from\s+['"]/i.test(line.trim()) && !line.trim().startsWith('import ')){
      line = 'import ' + line.trim();
    }

    // Handle lines that start an import but are fragmented (no from yet and no semicolon)
    if(/^\s*import\b/.test(line) && !/from\s+['"].+['"]/.test(line) && !/;\s*$/.test(line)){
      importBuffer.push(line.trim());
      continue;
    }

    if(importBuffer.length){
      importBuffer.push(line.trim());
      if(/from\s+['"].+['"];?/.test(line)) flushImportBuffer();
      continue;
    }

    // General cleanup of stray leading commas before interface/type declarations
    line = line.replace(/^(\s*),\s*(interface|type)\b/,'$1$2');

    repaired.push(line);
  }
  flushImportBuffer();

  let output = repaired.join('\n');

  // Inject RequestHandler import if needed
  if(needsRequestHandlerImport(output)){
    // If there's already an import from '@sveltejs/kit', augment it
    const kitImportRegex = /import\s+{[^}]*}\s+from\s+'@sveltejs\/kit';?/;
    if(kitImportRegex.test(output)){
      output = output.replace(kitImportRegex, (m)=>{
        return m.replace('{', '{ type RequestHandler, ');
      });
    } else {
      output = `import type { RequestHandler } from '@sveltejs/kit';\n` + output;
    }
  }

  // Collapse >2 blank lines
  output = output.replace(/\n{3,}/g,'\n\n');

  if(output !== originalText){
    fs.writeFileSync(filePath, output, 'utf8');
    console.log('✔ Repaired', path.relative(ROOT, filePath));
  }
}

const targets = discoverServerFiles(ROUTES_DIR);
console.log(`Scanning ${targets.length} route handler files...`);
targets.forEach(fp => {
  try { repairFile(fp); } catch (e){ console.warn('⚠ Failed to repair', fp, e.message); }
});
console.log('Import/header repair complete.');
