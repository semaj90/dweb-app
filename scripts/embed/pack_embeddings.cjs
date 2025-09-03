#!/usr/bin/env node
/** Pack embeddings.jsonl into binary float32 or int8 arrays + index */
const fs=require('fs'); const path=require('path');
const OPT_MODE=process.env.EMB_PACK_MODE||'f32'; // 'f32' | 'i8'
const DIR=path.resolve('.rag-metrics/embeddings');
const SRC=path.join(DIR,'embeddings.jsonl');
if(!fs.existsSync(SRC)) { console.error('Missing embeddings.jsonl'); process.exit(1);}
const lines=fs.readFileSync(SRC,'utf8').trim().split(/\n+/);
let vectors=0, dim=0; const meta=[];
// First pass: determine dim
for(const line of lines){ const rec=JSON.parse(line); if(!dim) dim=rec.embedding.length; }
const bufSize=OPT_MODE==='f32'? lines.length*dim*4 : lines.length*dim; // bytes
const buffer=OPT_MODE==='f32'? Buffer.allocUnsafe(bufSize):Buffer.alloc(bufSize);
let offset=0;
for(const line of lines){
  const rec=JSON.parse(line); const emb=rec.embedding; const id=rec.id; const relPath=rec.relPath; vectors++;
  // Normalize
  let norm=0; for(const v of emb) norm+=v*v; norm=Math.sqrt(norm)||1; const normed=emb.map(v=>v/norm);
  if(OPT_MODE==='f32'){
    for(const v of normed){ buffer.writeFloatLE(v, offset); offset+=4; }
  } else {
    for(const v of normed){ const q=Math.max(-1, Math.min(1, v)); buffer.writeInt8(Math.round(q*127), offset); offset+=1; }
  }
  meta.push({id, relPath});
}
fs.writeFileSync(path.join(DIR, OPT_MODE==='f32'?'embeddings.f32':'embeddings.i8'), buffer);
fs.writeFileSync(path.join(DIR,'embeddings.index.json'), JSON.stringify({
  mode:OPT_MODE,
  dim,
  vectors,
  bytePerVector: OPT_MODE==='f32'? dim*4: dim,
  dataFile: OPT_MODE==='f32'?'embeddings.f32':'embeddings.i8',
  metaFile: 'embeddings.meta.json'
},null,2));
fs.writeFileSync(path.join(DIR,'embeddings.meta.json'), JSON.stringify(meta));
console.log('Packed', vectors, 'vectors -> mode', OPT_MODE, 'dim', dim);
