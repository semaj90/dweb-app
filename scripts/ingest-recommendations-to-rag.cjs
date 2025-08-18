#!/usr/bin/env node
// Ingest top aggregated recommendations into GPU indexer / RAG pipeline

const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');

const AGG_FILE = path.join('logs','recommendation-aggregate.json');
const INDEX_ENDPOINT = process.env.RAG_INDEX_ENDPOINT || 'http://localhost:8097/index';
const LIMIT = parseInt(process.env.RAG_INGEST_LIMIT || '20', 10);

function postJSON(url, body){
  return new Promise((resolve,reject)=>{
    const u = new URL(url);
    const data = JSON.stringify(body);
    const req = http.request({hostname:u.hostname, port:u.port, path:u.pathname, method:'POST', headers:{'Content-Type':'application/json','Content-Length':Buffer.byteLength(data)}}, res=>{
      let buf=''; res.on('data',d=>buf+=d); res.on('end',()=>{ try{ resolve(JSON.parse(buf)); }catch{ resolve({raw:buf}); } });
    });
    req.on('error',reject);
    req.write(data); req.end();
  });
}

async function main(){
  if(!fs.existsSync(AGG_FILE)){
    console.error('Aggregate file not found. Run npm run recommend:aggregate first.');
    process.exit(1);
  }
  const agg = JSON.parse(fs.readFileSync(AGG_FILE,'utf8'));
  const top = (agg.top_recommendations||[]).slice(0,LIMIT);
  if(!top.length){
    console.log('No top recommendations to ingest.');
    return;
  }
  let success=0, failed=0;
  for (const rec of top){
    const text = rec.text || rec.issue || JSON.stringify(rec);
    const id = 'rec_'+crypto.createHash('sha1').update(text).digest('hex').slice(0,16);
    const doc = {
      id,
      content: text,
      metadata: { source: 'autosolve-aggregate', count: rec.count || 1, generated_at: agg.generated_at }
    };
    try { await postJSON(INDEX_ENDPOINT, doc); success++; }
    catch(e){ failed++; console.warn('Failed indexing', id, e.message); }
  }
  console.log(`Ingestion complete. Success: ${success} Failed: ${failed}`);
}

if (require.main === module) main();
