// worker.js - processes error events from Redis Stream, embeds, indexes, generates TODOs
import Redis from 'ioredis';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Client as PgClient } from 'pg';
import axios from 'axios';
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import Fastify from 'fastify';

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const QDRANT_URL = process.env.QDRANT_URL || 'http://127.0.0.1:6333';
const POSTGRES_DSN = process.env.DATABASE_URL || 'postgresql://postgres:password@127.0.0.1:5432/postgres';
const EMBEDDER_GRPC_ADDR = process.env.EMBEDDER_GRPC_ADDR || 'localhost:50051';
const GEMMA_API = process.env.GEMMA_API || 'http://localhost:8080/gemma';
const REPO_ROOT = process.env.REPO_ROOT || process.cwd();
const COLLECTION = process.env.QDRANT_COLLECTION || 'code_errors';

const redis = new Redis(REDIS_URL);
const qdrant = new QdrantClient({ url: QDRANT_URL });
let pg = null; let pgAvailable = true;
try {
  pg = new PgClient({ connectionString: POSTGRES_DSN });
  await pg.connect();
  console.log('Connected to Postgres');
} catch (e) {
  pgAvailable = false;
  console.warn('Postgres connection failed, falling back to file-based TODO store:', e.message);
}

// ensure collection exists (simple create if missing) with fallback when Qdrant unavailable
let qdrantAvailable = true;
async function ensureCollection(){
  try {
    await qdrant.getCollection(COLLECTION);
  } catch {
    try {
      await qdrant.createCollection({
        collection_name: COLLECTION,
        vectors: { size: 768, distance: 'Cosine' }
      });
      console.log('Created Qdrant collection', COLLECTION);
    } catch(e){
      qdrantAvailable = false;
      console.warn('Qdrant unavailable, vector operations disabled:', e.message);
    }
  }
}
await ensureCollection();

// --- Optional tokenizer load (package may be absent) ---
let tokenizer = null;
const tokenizerPath = path.join(process.cwd(), 'tokenizer.json');
if (fs.existsSync(tokenizerPath)) {
  try {
    const tokMod = await import('@huggingface/tokenizers').catch(()=>null);
    if (tokMod?.Tokenizer) {
      tokenizer = await tokMod.Tokenizer.fromFile(tokenizerPath);
      console.log('Loaded tokenizer from', tokenizerPath);
    } else {
      console.warn('huggingface tokenizers package not installed; using naive token count');
    }
  } catch(e){
    console.warn('Tokenizer load failed, falling back to naive token count:', e.message);
  }
} else {
  console.warn('Tokenizer file not found; using naive token count.');
}

const PROTO_PATH = path.join(process.cwd(), 'proto', 'embed.proto');
const packageDef = protoLoader.loadSync(PROTO_PATH, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const embedProto = grpc.loadPackageDefinition(packageDef).embed;
const embedClient = new embedProto.Embedder(EMBEDDER_GRPC_ADDR, grpc.credentials.createInsecure());

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

async function getFileContext(filePath, line, ctx=6){
  const full = path.resolve(REPO_ROOT, filePath);
  try {
    const lines = fs.readFileSync(full,'utf8').split(/\r?\n/);
    const start = Math.max(0, line-ctx-1);
    const end = Math.min(lines.length, line+ctx);
    return lines.slice(start,end).join('\n');
  } catch {
    try {
      const out = execSync(`git -C "${REPO_ROOT}" --no-pager show HEAD:"${filePath}"`, { encoding:'utf8' });
      const lines = out.split(/\r?\n/);
      const start = Math.max(0, line-ctx-1);
      const end = Math.min(lines.length, line+ctx);
      return lines.slice(start,end).join('\n');
    } catch(e){
      return '';
    }
  }
}

// --- Metrics state ---
const metrics = {
  startTime: Date.now(),
  embeddings: 0,
  embeddingErrors: 0,
  totalEmbeddingLatencyMs: 0,
  gemmaCalls: 0,
  gemmaFailures: 0,
  processedErrors: 0,
  lastErrorAt: null,
  highScoreDistributions: [] // collect top context high_score averages
};

function embedTextGrpc(text){
  const t0 = Date.now();
  return new Promise((resolve,reject)=>{
    embedClient.Embed({ text }, (err, resp)=>{
      const dt = Date.now() - t0;
      if (err){
        metrics.embeddingErrors++;
        return reject(err);
      }
      metrics.embeddings++;
      metrics.totalEmbeddingLatencyMs += dt;
      resolve(resp.vector);
    });
  });
}

async function upsertVector(id, vector, payload){
  if (!qdrantAvailable) return;
  try { await qdrant.upsert({ collection: COLLECTION, points:[{ id, vector, payload }] }); }
  catch(e){ console.warn('Qdrant upsert failed:', e.message); }
}

async function writeTodo({ errorId, title, desc, priority=5 }){
  if (pgAvailable && pg){
    try {
      const res = await pg.query(`INSERT INTO todos (error_id,title,body,priority,status,created_at) VALUES ($1,$2,$3,$4,'open',NOW()) RETURNING id`,[errorId,title,desc,priority]);
      return res.rows[0].id;
    } catch(e){
      console.warn('DB insert failed, using fallback store:', e.message);
    }
  }
  // Fallback: append to JSON file
  const fallbackFile = path.join(process.cwd(), 'todos-fallback.json');
  let data = [];
  try { data = JSON.parse(fs.readFileSync(fallbackFile,'utf8')); } catch { /* ignore */ }
  const id = data.length + 1;
  data.push({ id, errorId, title, body: desc, priority, status:'open', created_at: new Date().toISOString() });
  fs.writeFileSync(fallbackFile, JSON.stringify(data,null,2));
  return id;
}

async function callGemmaTodo(file,line,message,snippet,topContexts=[]){
  metrics.gemmaCalls++;
  // High score prompt enhancement: provide rationale & scoring components
  const contextSection = topContexts.map((c,i)=>{
    return `CTX${i}:: high_score=${c.high_score.toFixed(4)} (semantic=${(c.score||0).toFixed(4)}, recencyBoost=${c.recencyBoost?.toFixed(3)}, overlapBoost=${c.overlapBoost?.toFixed(3)})\n${(c.payload?.snippet||'').slice(0,600)}`;
  }).join('\n\n');
  const avgHigh = topContexts.length ? topContexts.reduce((a,b)=>a+b.high_score,0)/topContexts.length : 0;
  metrics.highScoreDistributions.push(avgHigh);
  if (metrics.highScoreDistributions.length>200) metrics.highScoreDistributions.shift();
  const prompt = [
    'You are a smart code assistant (Gemma-3).',
    'Strictly output compact JSON: {"title","description","steps":[...],"priority", "tags":[...]}.',
    'Scoring Guidance: high_score combines: 60% semantic similarity, 25% recencyBoost (1/ageHours), 15% overlapBoost (best-practice keyword density). Prefer contexts with higher high_score when forming remediation steps.',
    `Primary Error: ${message}`,
    `Location: ${file}:${line}`,
    'Current Code Snippet (focus on precise fix):\n'+snippet,
    'Top Ranked Related Contexts with Scores:\n'+contextSection,
    'Instructions: 1) Derive a concise actionable title. 2) Provide a multi-line description referencing ONLY relevant CTX indices. 3) steps[]: ordered concrete remediation actions (max 6). 4) priority: 1 (urgent) to 5 (low). 5) tags: include error type & any best-practice categories (performance, security, typing, cache, null, retry). Return JSON only.'
  ].join('\n\n');
  try {
    const r = await axios.post(GEMMA_API + '/generate', { model:'gemma-3', prompt, max_tokens:512, temperature:0.1 }, { timeout:60000 });
    return r.data;
  } catch(e){ metrics.gemmaFailures++; console.error('Gemma call failed', e.message); return null; }
}

async function processEntry(data){
  const { id, file, line, col, message, raw } = data;
  const errorId = id;
  const snippet = await getFileContext(file, Number(line||1));
  let tokenCount = 0;
  if (tokenizer){ try { tokenCount = tokenizer.encode(snippet).length; } catch{ tokenCount = snippet.split(/\s+/).length; } }
  else tokenCount = snippet.split(/\s+/).length;

  const processed = await redis.hget('error_meta:'+errorId,'processed');
  if (processed) return;

  let vector;
  try { vector = await embedTextGrpc(snippet); } catch(e){ console.error('embed error', e.message); return; }
  const payload = { file, line:Number(line||1), col:Number(col||0), message, snippet, tokenCount };
  await upsertVector(errorId, vector, payload);
  let topContexts = [];
  if (qdrantAvailable){
    try {
      const searchRes = await qdrant.search({ collection: COLLECTION, vector, limit:15, with_payload:true });
      topContexts = (searchRes?.result||[]).map(r=>({ id:r.id, payload:r.payload, score:r.score }));
    } catch(e){ console.warn('Qdrant search failed:', e.message); }
  }

  // --- high_score ranking: semantic (qdrant score) + recency + best_practice keyword overlap ---
  const BEST_PRACTICE_KEYWORDS = ['accessibility','performance','security','typing','null','error','retry','cache'];
  const now = Date.now();
  topContexts = topContexts.map(c=>{
    const ts = c.payload?.processed_at ? Date.parse(c.payload.processed_at) : now;
    const ageHours = Math.max(1, (now - ts)/3600000);
    const recencyBoost = 1 / ageHours; // more recent -> higher
    const text = (c.payload?.snippet||'').toLowerCase();
    let overlap = 0;
    for (const kw of BEST_PRACTICE_KEYWORDS){ if (text.includes(kw)) overlap++; }
    const overlapBoost = 1 + (overlap / BEST_PRACTICE_KEYWORDS.length);
    const semantic = c.score || 0;
    const finalScore = semantic * 0.6 + recencyBoost * 0.25 + overlapBoost * 0.15;
    return { ...c, high_score: finalScore, recencyBoost, overlapBoost };
  }).sort((a,b)=> b.high_score - a.high_score).slice(0,5);

  const gemmaOut = await callGemmaTodo(file,line,message,snippet,topContexts);
  let todoCandidate=null;
  if (gemmaOut && typeof gemmaOut==='object' && gemmaOut.text){ try { todoCandidate = JSON.parse(gemmaOut.text); } catch{}};
  if (!todoCandidate){
    todoCandidate = { title:`Fix: ${(message||'').slice(0,100)}`, description:`Error in ${file}:${line}\n${message}`, steps:['Investigate','Patch','Test'], priority:5, tags:['error'] };
  }
  const todoId = await writeTodo({ errorId, title:todoCandidate.title, desc: JSON.stringify(todoCandidate,null,2), priority: todoCandidate.priority||5 });
  metrics.processedErrors++;
  metrics.lastErrorAt = Date.now();
  await redis.hset('error_meta:'+errorId, { processed:1, processed_at:new Date().toISOString(), todo_id: todoId });
  console.log('Processed error', errorId, '-> TODO', todoId);
}

async function mainLoop(){
  let lastId = '0-0';
  while(true){
    try {
      const res = await redis.xread('BLOCK', 0, 'STREAMS', 'errors_stream', lastId).catch(err=>{
        if (/unknown command 'xread'/i.test(err.message)) return 'FALLBACK';
        throw err;
      });
      if (res === 'FALLBACK'){
        // Fallback: pop items from a list where producers LPUSH JSON
        const raw = await redis.rpop('errors_stream_list');
        if (!raw){ await sleep(1000); continue; }
        try {
          const data = JSON.parse(raw);
          await processEntry(data);
        } catch(e){ console.warn('Failed to parse fallback list item:', e.message); }
        continue;
      }
      if (!res) continue;
      const items = res[0][1];
      for (const item of items){
        const sid = item[0];
        const arr = item[1];
        const data = {};
        for (let i=0;i<arr.length;i+=2) data[arr[i]] = arr[i+1];
        await processEntry(data);
        lastId = sid;
      }
    } catch(e){ console.error('Loop error', e.message); await sleep(2000); }
  }
}

// --- Metrics HTTP server (/metrics) ---
const METRICS_PORT = Number(process.env.WORKER_METRICS_PORT || 9301);
const fastify = Fastify({ logger: false });
fastify.get('/metrics', async (_req, _rep)=>{
  const uptimeSec = (Date.now()-metrics.startTime)/1000;
  const avgEmbedMs = metrics.embeddings ? (metrics.totalEmbeddingLatencyMs/metrics.embeddings) : 0;
  const avgHighScore = metrics.highScoreDistributions.length ? (metrics.highScoreDistributions.reduce((a,b)=>a+b,0)/metrics.highScoreDistributions.length) : 0;
  return {
    service: 'worker',
    uptimeSec,
    processedErrors: metrics.processedErrors,
    embeddings: metrics.embeddings,
    embeddingErrors: metrics.embeddingErrors,
    avgEmbeddingLatencyMs: Number(avgEmbedMs.toFixed(2)),
    gemmaCalls: metrics.gemmaCalls,
    gemmaFailures: metrics.gemmaFailures,
    avgHighScore: Number(avgHighScore.toFixed(4)),
    timestamp: new Date().toISOString()
  };
});
fastify.listen({ port: METRICS_PORT, host: '0.0.0.0' }).then(()=>{
  console.log('Worker metrics endpoint listening on /metrics port', METRICS_PORT);
}).catch(e=> console.warn('Metrics server failed:', e.message));

mainLoop().catch(e=>{ console.error(e); process.exit(1); });
