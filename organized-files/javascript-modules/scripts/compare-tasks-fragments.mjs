import fs from 'node:fs';
import path from 'node:path';

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

const root = process.cwd();
const monolithPath = path.join(root,'.vscode','tasks.json');
const fragDir = path.join(root,'.vscode','tasks');

if(!fs.existsSync(monolithPath)) { console.error('.vscode/tasks.json missing'); process.exit(1); }
if(!fs.existsSync(fragDir)) { console.error('.vscode/tasks directory missing'); process.exit(1); }

const monoRaw = fs.readFileSync(monolithPath,'utf8');
let monoObj;
try { monoObj = JSON.parse(stripComments(monoRaw)); }
catch(e){ console.error('Failed to parse monolith:', e.message); process.exit(1); }
if(!Array.isArray(monoObj.tasks)) { console.error('Monolith has no tasks array'); process.exit(1); }

const fragFiles = fs.readdirSync(fragDir).filter(f=>/^tasks-.*\.json$/.test(f));
if(fragFiles.length===0) { console.error('No fragment files found'); process.exit(1); }

let mergedLabels=[];
for(const f of fragFiles){
  const p = path.join(fragDir,f);
  try{
    const data = JSON.parse(fs.readFileSync(p,'utf8'));
    if(Array.isArray(data.tasks)) mergedLabels.push(...data.tasks.map(t=>t.label).filter(Boolean));
  } catch(e){ console.error('Fragment',f,'invalid JSON:',e.message); }
}

const monoLabels = monoObj.tasks.map(t=>t.label).filter(Boolean);
const monoSet = new Set(monoLabels);
const fragSet = new Set(mergedLabels);
const inFragNotMono = [...fragSet].filter(l=>!monoSet.has(l)).sort();
const inMonoNotFrag = [...monoSet].filter(l=>!fragSet.has(l)).sort();
const counts = {};
mergedLabels.forEach(l=>counts[l]=(counts[l]||0)+1);
const duplicates = Object.entries(counts).filter(([,c])=>c>1).sort((a,b)=>b[1]-a[1]).slice(0,50);

console.log('monolith_count', monoLabels.length);
console.log('fragments_total_tasks', mergedLabels.length, 'fragment_files', fragFiles.length);
console.log('unique_frag_labels', fragSet.size, 'unique_mono_labels', monoSet.size);
console.log('labels_in_fragments_not_in_monolith_count', inFragNotMono.length);
if(inFragNotMono.length) console.log('--- labels in fragments not in monolith (sample up to 200) ---\n'+inFragNotMono.slice(0,200).join('\n')+'\n--- end ---');
console.log('labels_in_monolith_not_in_fragments_count', inMonoNotFrag.length);
if(inMonoNotFrag.length) console.log('--- labels in monolith not in fragments ---\n'+inMonoNotFrag.join('\n')+'\n--- end ---');
if(duplicates.length){ console.log('--- top duplicate labels in fragments (label:count) ---'); duplicates.forEach(d=>console.log(d[0]+':'+d[1])); }
