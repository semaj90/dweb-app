#!/usr/bin/env node
/** Search packed embeddings using cosine similarity */
const fs=require('fs'); const path=require('path'); const crypto=require('crypto');
const DIR=path.resolve('.rag-metrics/embeddings');
const index=JSON.parse(fs.readFileSync(path.join(DIR,'embeddings.index.json'),'utf8'));
const meta=JSON.parse(fs.readFileSync(path.join(DIR,'embeddings.meta.json'),'utf8'));
const data=fs.readFileSync(path.join(DIR,index.dataFile));
const dim=index.dim; const mode=index.mode; const vectors=index.vectors;
function fakeEmbed(text){ const hash=crypto.createHash('sha256').update(text).digest(); const v=[]; for(let i=0;i<dim;i++) v.push(hash[i%hash.length]/255); return v; }
function cosine(q, i){ let dot=0; for(let k=0;k<dim;k++) dot+=q[k]*i[k]; return dot; }
const query=process.argv.slice(2).join(' ')||'contract liability';
let q=fakeEmbed(query); // already 0..1
// Normalize q
let qn=0; for(const v of q) qn+=v*v; qn=Math.sqrt(qn)||1; q=q.map(v=>v/qn);
const results=[];
if(mode==='f32'){
  for(let i=0;i<vectors;i++){
    const base=i*dim*4; const vec=new Array(dim);
    let norm=0; for(let d=0;d<dim;d++){ const val=data.readFloatLE(base+d*4); norm+=val*val; vec[d]=val; }
    const score=cosine(q, vec); results.push({score,id:meta[i].id,relPath:meta[i].relPath});
  }
} else {
  for(let i=0;i<vectors;i++){
    const base=i*dim; let score=0; for(let d=0;d<dim;d++){ const val=data.readInt8(base+d)/127; score+=q[d]*val; } results.push({score,id:meta[i].id, relPath:meta[i].relPath});
  }
}
results.sort((a,b)=>b.score-a.score);
console.log(JSON.stringify({query, top:results.slice(0,10)},null,2));
