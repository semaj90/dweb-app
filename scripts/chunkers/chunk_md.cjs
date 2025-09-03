#!/usr/bin/env node
const fs=require('fs'); const path=require('path');
const {splitWithOverlap, estTokens, stableChunkId, sha256}=require('./chunk_utils.cjs');
const MANIFEST=path.resolve('.rag-metrics/manifest.json');
const OUT_DIR=path.resolve('.rag-metrics/chunks');
if(!fs.existsSync(MANIFEST)) { console.error('Manifest missing.'); process.exit(1);}
const manifest=JSON.parse(fs.readFileSync(MANIFEST,'utf8')).entries.filter(e=>e.lang==='md');
if(!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR,{recursive:true});
const out=fs.createWriteStream(path.join(OUT_DIR,'md.jsonl'));
for(const file of manifest){
  const content=fs.readFileSync(file.relPath,'utf8');
  const blocks=content.split(/\n(?=#)/); // split before headings
  const assembled=[];
  for(const b of blocks){ if(b.length>2000){ for(const sub of splitWithOverlap(b,1400,200)) assembled.push(sub); } else if(b.trim()) assembled.push(b); }
  assembled.forEach((c,i)=>{ const id=stableChunkId(file.relPath,i,c); out.write(JSON.stringify({id,relPath:file.relPath,lang:'md',chunkIndex:i,totalChunks:assembled.length,bytes:c.length,sha256:sha256(c),tokens:estTokens(c),content:c})+'\n'); });
}
out.end(()=>console.log('Markdown chunking complete -> .rag-metrics/chunks/md.jsonl'));
