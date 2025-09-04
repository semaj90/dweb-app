#!/usr/bin/env node
/**
 * Apply lightweight automated fixes for import & event directive issues.
 * Non-destructive: creates backup alongside file with .bak suffix.
 */
import { readFile, writeFile, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const SUG_FILE = '.vscode/svelte-fix-suggestions.json';
const DRY_RUN = process.argv.includes('--dry-run');

function normalizeImports(src){
  let out = src.replace(/import\s+type\s+type/g,'import type')
    .replace(/from\s+['"]([^'".][^'"/]*)['"]/g,(m,mod)=>`from '${mod}'`)
    .replace(/(import\s+[^;]+)(;?)(\r?\n)/g,'$1;$3');
  return out;
}

function migrateEvents(src){
  return src.replace(/on:click=/g,'onclick=');
}

function repairSyntax(src){
  const open = (src.match(/\{/g)||[]).length;
  const close = (src.match(/\}/g)||[]).length;
  if(open>close) return src + '\n'.repeat(open-close) + '}'.repeat(open-close);
  return src;
}

function bindingFix(src){
  // Ensure exported props exist for patterns like let { prop } = $props()
  const missingExports = [];
  const propDestruct = src.match(/let\s*\{([^}]+)\}\s*=\s*\$props\(\)/);
  if(propDestruct){
    const names = propDestruct[1].split(',').map(s=>s.trim()).filter(Boolean);
    for(const n of names){
      if(!new RegExp(`export\s+let\s+${n}\b`).test(src)) missingExports.push(n);
    }
  }
  if(missingExports.length){
    const insertionPoint = src.indexOf('<script');
    if(insertionPoint>=0){
      // insert after opening <script> tag
      return src.replace(/<script[^>]*>/, m => m + '\n' + missingExports.map(n=>`export let ${n};`).join('\n'));
    }
  }
  return src;
}

function removeUnusedCss(src){
  // Very naive: remove empty rules `.class { }`
  return src.replace(/\.[a-zA-Z0-9_-]+\s*\{\s*\}/g,'');
}

function applyAction(content, action){
  switch(action){
    case 'normalize-imports': return normalizeImports(content);
    case 'event-directive': return migrateEvents(content);
    case 'syntax-repair': return repairSyntax(content);
    case 'binding-fix': return bindingFix(content);
    case 'css-prune': return removeUnusedCss(content);
    default: return content;
  }
}

(async function(){
  if(!existsSync(SUG_FILE)){
    console.error('Suggestions file missing; run generate-suggestions first.');
    process.exit(2);
  }
  const { suggestions } = JSON.parse(await readFile(SUG_FILE,'utf8'));
  let applied = 0;
  for(const s of suggestions){
    if(!existsSync(s.file)) continue;
    try {
      const orig = await readFile(s.file,'utf8');
      const fixed = applyAction(orig, s.action);
      if(fixed !== orig){
        if(DRY_RUN){
          applied++;
        } else {
          await copyFile(s.file, s.file + '.bak');
          await writeFile(s.file, fixed, 'utf8');
          applied++;
        }
      }
    } catch(e){
      console.warn('Skip', s.file, e.message);
    }
  }
  if(DRY_RUN){
    console.log(`🔎 Dry-run: ${applied} files would be modified (no changes written)`);
  } else {
    console.log(`✅ Applied ${applied} fixes (backups created for changed files)`);
  }
})();
