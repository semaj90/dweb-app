#!/usr/bin/env node
const fs=require('fs'); const path=require('path');
const {splitWithOverlap, estTokens, stableChunkId, sha256}=require('./chunk_utils.cjs');
const MANIFEST=path.resolve('.rag-metrics/manifest.json');
const OUT_DIR=path.resolve('.rag-metrics/chunks');
if(!fs.existsSync(MANIFEST)) { console.error('Manifest missing.'); process.exit(1);}
const manifest=JSON.parse(fs.readFileSync(MANIFEST,'utf8')).entries.filter(e=>e.lang==='svelte');
if(!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR,{recursive:true});
const out=fs.createWriteStream(path.join(OUT_DIR,'svelte.jsonl'));
for(const file of manifest){
  const raw=fs.readFileSync(file.relPath,'utf8');
  // separate <script>, <style>, template
  const segments=[];
  const scriptMatches=[...raw.matchAll(/<script[\s\S]*?>[\s\S]*?<\/script>/g)];
  let consumed=new Set();
  for(const m of scriptMatches){ segments.push(m[0]); consumed.add(m.index); }
  const styleMatches=[...raw.matchAll(/<style[\s\S]*?>[\s\S]*?<\/style>/g)];
  for(const m of styleMatches){ segments.push(m[0]); consumed.add(m.index); }
  // Remaining template
  // naive fallback: remove extracted
  let template=raw;
  for(const seg of segments){ template=template.replace(seg,''); }
  segments.push(template);
  const assembled=[];
  for(const seg of segments){ if(seg.trim().length===0) continue; if(seg.length>1800){ for(const sub of splitWithOverlap(seg,1300,180)) assembled.push(sub);} else assembled.push(seg); }
  assembled.forEach((c,i)=>{ const id=stableChunkId(file.relPath,i,c); out.write(JSON.stringify({id,relPath:file.relPath,lang:'svelte',chunkIndex:i,totalChunks:assembled.length,bytes:c.length,sha256:sha256(c),tokens:estTokens(c),content:c})+'\n'); });
}
out.end(()=>console.log('Svelte chunking complete -> .rag-metrics/chunks/svelte.jsonl'));
