#!/usr/bin/env node
/**
 * Lightweight Recommendation Service
 * Endpoint: POST /api/process-error-log
 * Body: { logFile, content, timestamp }
 * Returns: { recommendations: [{ issue, autoFixable, confidence, file, codeFix }] }
 */
import http from 'http';
import { randomUUID } from 'crypto';
import fs from 'fs/promises';
import path from 'path';

// Persistence of history is handled in daemon (error-processor-daemon.js) via recommendations-history.jsonl
let BASE = parseInt(process.env.RECOMMENDATION_PORT||'8099');
let PORT = BASE;

function analyze(content){
  const recs = [];
  // Simple heuristics
  if (/Cannot find name '([A-Za-z0-9_]+)'/g.test(content)) {
    const matches = [...content.matchAll(/Cannot find name '([A-Za-z0-9_]+)'/g)].slice(0,10);
    for (const m of matches){
      recs.push({
        id: randomUUID(),
        issue: `Missing symbol ${m[1]}`,
        autoFixable: false,
        confidence: 0.92,
        file: null,
        codeFix: `// import or declare ${m[1]}`
      });
    }
  }
  if (/is declared but its value is never read/g.test(content)) {
    const matches = [...content.matchAll(/'(.*?)' is declared but its value is never read/g)].slice(0,10);
    for (const m of matches){
      recs.push({
        id: randomUUID(),
        issue: `Unused identifier ${m[1]}`,
        autoFixable: true,
        confidence: 0.88,
        file: null,
        codeFix: `remove variable: ${m[1]}`
      });
    }
  }
  if (/Type '(.*?)' is not assignable to type '(.*?)'/g.test(content)) {
    recs.push({
      id: randomUUID(),
      issue: 'Type assignment mismatch(s)',
      autoFixable: false,
      confidence: 0.8,
      file: null,
      codeFix: 'add type assertion'
    });
  }
  return recs.slice(0,50);
}

async function writePortMetadata(port){
  try {
    const dir = path.resolve('logs');
    await fs.mkdir(dir,{recursive:true});
    const file = path.join(dir,'recommendation-service.json');
    await fs.writeFile(file, JSON.stringify({ port, timestamp: new Date().toISOString() }, null, 2));
  } catch(e){
    console.error('Failed to write recommendation-service port file:', e.message);
  }
}

const server = http.createServer((req,res)=>{
  if (req.method==='POST' && req.url==='/api/process-error-log') {
    let body='';
    req.on('data',d=> body+=d);
    req.on('end',()=>{
      try {
        const payload = JSON.parse(body||'{}');
        const recs = analyze(payload.content||'');
        res.writeHead(200,{ 'Content-Type':'application/json'});
        res.end(JSON.stringify({ recommendations: recs }));
      } catch(e){
        res.writeHead(400,{ 'Content-Type':'application/json'});
        res.end(JSON.stringify({ error: e.message }));
      }
    });
  } else if (req.method==='GET' && req.url==='/health') {
    res.writeHead(200,{ 'Content-Type':'application/json'});
    res.end(JSON.stringify({ status:'ok', port: PORT }));
  } else {
    res.writeHead(404);res.end();
  }
});

let attempt = 0;
server.on('error', err => {
    if (err.code === 'EADDRINUSE' && attempt < 10) {
        attempt++;
        PORT = BASE + attempt;
        console.log(`⚠️ Port in use, trying ${PORT}`);
        setTimeout(bind, 150);
    } else if (err.code === 'EADDRINUSE') {
        console.error('❌ Failed to bind any port');
        process.exit(1);
    } else {
        console.error('❌ Server error', err);
        process.exit(1);
    }
});

function bind() {
    server.listen(PORT, async () => {
        console.log(`🧠 Recommendation service listening on :${PORT}`);
        await writePortMetadata(PORT);
    });
}

bind();
