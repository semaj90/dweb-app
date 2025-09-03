#!/usr/bin/env node
const fs=require('fs'); const path=require('path');
const {splitWithOverlap, estTokens, stableChunkId, sha256}=require('./chunk_utils.cjs');
const MANIFEST=path.resolve('.rag-metrics/manifest.json');
const OUT_DIR=path.resolve('.rag-metrics/chunks');
if(!fs.existsSync(MANIFEST)) { console.error('Manifest missing.'); process.exit(1);}
const manifest=JSON.parse(fs.readFileSync(MANIFEST,'utf8')).entries.filter(e=>e.lang==='ts' || e.lang==='js');
if(!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR,{recursive:true});
const out=fs.createWriteStream(path.join(OUT_DIR,'ts.jsonl'));
for(const file of manifest){
  const content=fs.readFileSync(file.relPath,'utf8');
  // try splitting on export / function boundaries
  const boundaryRegex=/^(export\s+.*|function\s+.*|class\s+.*)/m;
  let pieces=[]; let buffer='';
  const lines=content.split('\n');
  for(const line of lines){
    if(boundaryRegex.test(line) && buffer.length>1200){ pieces.push(buffer); buffer=''; }
    buffer+=line+'\n';
  }
  if(buffer.trim()) pieces.push(buffer);
  const assembled=[]; for(const p of pieces){ if(p.length>1600){ for(const sub of splitWithOverlap(p,1200,160)) assembled.push(sub);} else assembled.push(p);}
  assembled.forEach((c,i)=>{ const id=stableChunkId(file.relPath,i,c); out.write(JSON.stringify({id,relPath:file.relPath,lang:file.lang==='ts'?'ts':'js',chunkIndex:i,totalChunks:assembled.length,bytes:c.length,sha256:sha256(c),tokens:estTokens(c),content:c})+'\n'); });
}
out.end(()=>console.log('TS/JS chunking complete -> .rag-metrics/chunks/ts.jsonl'));
