#!/usr/bin/env node
/** Search with optional filters: --ext=.ts,.svelte --path-contains=vector,ai */
const fs=require('fs'); const path=require('path'); const crypto=require('crypto');
const args=process.argv.slice(2); const query=args.filter(a=>!a.startsWith('--')).join(' ')||'contract';
const extArg=(args.find(a=>a.startsWith('--ext='))||'').split('=')[1];
const pathArg=(args.find(a=>a.startsWith('--path-contains='))||'').split('=')[1];
const exts=extArg? new Set(extArg.split(',').map(s=>s.trim())):null;
const substrings=pathArg? pathArg.split(',').map(s=>s.trim().toLowerCase()):[];
const DIR=path.resolve('.rag-metrics/embeddings');
const index=JSON.parse(fs.readFileSync(path.join(DIR,'embeddings.index.json'),'utf8'));
const meta=JSON.parse(fs.readFileSync(path.join(DIR,'embeddings.meta.json'),'utf8'));
const data=fs.readFileSync(path.join(DIR,index.dataFile));
const dim=index.dim; const mode=index.mode; const vectors=index.vectors;
function fakeEmbed(text){ const hash=crypto.createHash('sha256').update(text).digest(); const v=[]; for(let i=0;i<dim;i++) v.push(hash[i%hash.length]/255); return v; }
let q=fakeEmbed(query); let qn=0; for(const v of q) qn+=v*v; qn=Math.sqrt(qn)||1; q=q.map(v=>v/qn);
const results=[]; const limit=50;
function passesFilters(relPath){ if(exts){ const e=path.extname(relPath); if(!exts.has(e)) return false; } if(substrings.length){ const lower=relPath.toLowerCase(); for(const sub of substrings){ if(!lower.includes(sub)) return false; } } return true; }
if(mode==='f32'){
  for(let i=0;i<vectors;i++){
    const rel=meta[i].relPath; if(!passesFilters(rel)) continue;
    const base=i*dim*4; let score=0; for(let d=0;d<dim;d++){ const val=data.readFloatLE(base+d*4); score+=q[d]*val; }
    results.push({score,id:meta[i].id,relPath:rel});
  }
} else {
  for(let i=0;i<vectors;i++){
    const rel=meta[i].relPath; if(!passesFilters(rel)) continue;
    const base=i*dim; let score=0; for(let d=0;d<dim;d++){ const val=data.readInt8(base+d); score+=q[d]*val/127; }
    results.push({score,id:meta[i].id,relPath:rel});
  }
}
results.sort((a,b)=>b.score-a.score);
console.log(JSON.stringify({query, filters:{ext:[...(exts||[])], pathContains:substrings}, results:results.slice(0,10)},null,2));
