#!/usr/bin/env node
/** Simple cosine similarity search over placeholder embeddings */
const fs=require('fs'); const path=require('path');
const EMB_PATH=path.resolve('.rag-metrics/embeddings/embeddings.jsonl');
if(!fs.existsSync(EMB_PATH)) { console.error('Missing embeddings file'); process.exit(1);}
const queryText=process.argv.slice(2).join(' ')||'contract liability terms';
// Deterministic pseudo-embedding same as generation (mirrors fakeEmbed)
const crypto=require('crypto');
function fakeEmbed(text){ const hash=crypto.createHash('sha256').update(text).digest(); const v=[]; for(let i=0;i<64;i++) v.push(hash[i%hash.length]/255); return v; }
function cosine(a,b){ let dot=0,na=0,nb=0; for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; } return dot/(Math.sqrt(na)*Math.sqrt(nb) || 1); }
const qVec=fakeEmbed(queryText.slice(0,4000));
const topN=10; const heap=[]; // simple array sort
const stream=fs.createReadStream(EMB_PATH,{encoding:'utf8'});
let buf='';
stream.on('data',chunk=>{ buf+=chunk; let lines=buf.split('\n'); buf=lines.pop(); for(const line of lines){ if(!line.trim()) continue; const rec=JSON.parse(line); const score=cosine(qVec, rec.embedding); heap.push({score,id:rec.id,relPath:rec.relPath,chunkIndex:rec.chunkIndex}); }});
stream.on('end',()=>{ heap.sort((a,b)=>b.score-a.score); console.log(JSON.stringify({ query:queryText, results:heap.slice(0,topN) }, null,2)); });
