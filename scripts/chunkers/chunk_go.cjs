#!/usr/bin/env node
const fs=require('fs'); const path=require('path');
const {splitWithOverlap, estTokens, stableChunkId, sha256}=require('./chunk_utils.cjs');
const MANIFEST=path.resolve('.rag-metrics/manifest.json');
const OUT_DIR=path.resolve('.rag-metrics/chunks');
if(!fs.existsSync(MANIFEST)) { console.error('Manifest missing, run sources scan first.'); process.exit(1);}
const manifest=JSON.parse(fs.readFileSync(MANIFEST,'utf8')).entries.filter(e=>e.lang==='go');
if(!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR,{recursive:true});
const out=fs.createWriteStream(path.join(OUT_DIR,'go.jsonl'));
for(const file of manifest){
  const content=fs.readFileSync(file.relPath,'utf8');
  // heuristic: split on double newline boundaries first for functions
  const rawBlocks=content.split(/\n{2,}/);
  let chunkIndex=0; const assembled=[];
  for(const block of rawBlocks){
    if(block.length>1600){
      for(const sub of splitWithOverlap(block,1100,150)) assembled.push(sub);
    } else if(block.trim()) {
      assembled.push(block);
    }
  }
  assembled.forEach((c,i)=>{
    const id=stableChunkId(file.relPath,i,c);
    out.write(JSON.stringify({id, relPath:file.relPath, lang:'go', chunkIndex:i, totalChunks:assembled.length, bytes:c.length, sha256:sha256(c), tokens:estTokens(c), content:c})+'\n');
  });
}
out.end(()=>console.log('Go chunking complete -> .rag-metrics/chunks/go.jsonl'));
