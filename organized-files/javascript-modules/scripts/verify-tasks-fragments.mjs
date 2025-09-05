#!/usr/bin/env node
/**
 * Verify that .vscode/tasks.json (monolith) and split fragments round-trip cleanly.
 * Checks:
 *  1. Comment-aware parse of monolith succeeds.
 *  2. Split fragments exist & cover identical task count.
 *  3. Duplicate task labels are reported (fail if duplicates unless --allow-duplicates passed).
 *  Exits non-zero on failure so it can be used in pre-commit.
 */
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const monolithPath = path.join(root,'.vscode','tasks.json');
const fragDir = path.join(root,'.vscode','tasks');
const allowDuplicates = process.argv.includes('--allow-duplicates');

function stripComments(input){
  let out=''; let inStr=false; let esc=false;
  for(let i=0;i<input.length;i++){
    const ch = input[i];
    if(inStr){
      out += ch;
      if(esc){ esc=false; }
      else if(ch==='\\') esc=true;
      else if(ch==='"') inStr=false;
      continue;
    }
    if(ch==='"'){ inStr=true; out+=ch; continue; }
    if(ch==='/' && i+1<input.length){
      const nx=input[i+1];
      if(nx==='/' ){ i+=2; while(i<input.length && input[i] !== '\n') i++; continue; }
      if(nx==='*'){ i+=2; while(i+1<input.length && !(input[i]==='*' && input[i+1]==='/')) i++; i++; continue; }
    }
    out+=ch;
  }
  return out;
}

function fail(msg){
  console.error(`❌ tasks:verify: ${msg}`);
  process.exit(1);
}

if(!fs.existsSync(monolithPath)) fail('Missing .vscode/tasks.json');
if(!fs.existsSync(fragDir)) fail('Missing fragments directory .vscode/tasks');

const monoRaw = fs.readFileSync(monolithPath,'utf8');
let monoObj;
try { monoObj = JSON.parse(stripComments(monoRaw)); }
catch(e){ fail('Failed to parse monolith: '+e.message); }
if(!Array.isArray(monoObj.tasks)) fail('Monolith has no tasks array');

// Gather fragment tasks
const fragFiles = fs.readdirSync(fragDir).filter(f=>/^tasks-.*\.json$/.test(f));
if(fragFiles.length === 0) fail('No fragment files found');
let merged=[];
for(const f of fragFiles){
  const p = path.join(fragDir,f);
  try {
    const data = JSON.parse(fs.readFileSync(p,'utf8'));
    if(Array.isArray(data.tasks)) merged.push(...data.tasks);
  } catch(e){ fail(`Fragment ${f} is invalid JSON: ${e.message}`); }
}

// Count parity
if(merged.length !== monoObj.tasks.length){
  fail(`Task count mismatch: monolith=${monoObj.tasks.length} fragments=${merged.length}`);
}

// Duplicate label detection
const labelMap = new Map();
for(const t of merged){
  const lbl = t.label || '__undefined__';
  labelMap.set(lbl, (labelMap.get(lbl)||0)+1);
}
const duplicates = [...labelMap.entries()].filter(([,c])=>c>1);
if(duplicates.length && !allowDuplicates){
  console.error('Duplicate task labels detected:');
  for(const [lbl,c] of duplicates){ console.error(`  ${lbl} (${c})`); }
  fail('Duplicate labels must be resolved or pass --allow-duplicates');
}

console.log(`✅ tasks:verify passed (tasks=${merged.length}, fragments=${fragFiles.length}${duplicates.length?`, duplicates=${duplicates.length} (allowed)`:''})`);
process.exit(0);
