#!/usr/bin/env node
/** Merge fragments in .vscode/tasks into single .vscode/tasks.merged.json (non-destructive) */
import fs from 'node:fs';
import path from 'node:path';
const root = process.cwd();
const fragDir = path.join(root,'.vscode','tasks');
if(!fs.existsSync(fragDir)){
  console.error('No fragments directory .vscode/tasks');
  process.exit(1);
}
const files = fs.readdirSync(fragDir).filter(f=>/^tasks-.*\.json$/.test(f));
let version = '2.0.0';
const merged = [];
for(const f of files){
  const obj = JSON.parse(fs.readFileSync(path.join(fragDir,f),'utf8'));
  version = obj.version || version;
  if(Array.isArray(obj.tasks)) merged.push(...obj.tasks);
}
const out = { version, tasks: merged };
const outPath = path.join(root,'.vscode','tasks.merged.json');
fs.writeFileSync(outPath, JSON.stringify(out,null,2));
console.log(`Merged ${files.length} fragments (${merged.length} tasks) → ${outPath}`);

// Optional validation against original monolith if present
const monolithPath = path.join(root,'.vscode','tasks.json');
if(fs.existsSync(monolithPath)){
  try {
    const raw = fs.readFileSync(monolithPath,'utf8');
    // Reuse comment strip logic (duplicate minimal) to parse original
    function stripComments(input){
      let out=''; let inStr=false; let esc=false;
      for(let i=0;i<input.length;i++){
        const ch = input[i];
        if(inStr){
          out += ch;
            if(esc){ esc=false; }
            else if(ch==='\\'){ esc=true; }
            else if(ch==='"'){ inStr=false; }
          continue;
        }
        if(ch==='"'){ inStr=true; out+=ch; continue; }
        if(ch==='/' && i+1<input.length){
          const nx = input[i+1];
          if(nx==='/' ){
            i+=2; while(i<input.length && input[i] !== '\n'){ i++; } continue;
          } else if(nx==='*'){
            i+=2; while(i+1<input.length && !(input[i]==='*' && input[i+1]==='/')) i++; i++; continue;
          }
        }
        out+=ch;
      }
      return out;
    }
    const cleaned = stripComments(raw);
    const orig = JSON.parse(cleaned);
    const origTasks = Array.isArray(orig.tasks)?orig.tasks:[];
    const mismatch = origTasks.length !== merged.length;
    if(mismatch){
      console.warn(`⚠️ Validation: monolith has ${origTasks.length} tasks vs merged ${merged.length}`);
    } else {
      console.log('✅ Validation: task count matches original monolith');
    }
  } catch(e){
    console.warn('Validation skipped (could not parse original):', e.message);
  }
}
