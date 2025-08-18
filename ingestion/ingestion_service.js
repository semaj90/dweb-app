import Fastify from 'fastify';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import Redis from 'ioredis';

const fastify = Fastify({ logger: true });
const STORAGE_ROOT = process.env.RAG_STORAGE_DIR || path.join(process.cwd(),'ingest-store');
const BUCKET = process.env.RAG_BUCKET || 'legal-documents';
const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
const metrics = { start: Date.now(), chunks_received:0, docs_completed:0, queue_push_failures:0, queue_mode: process.env.RAG_QUEUE_MODE || 'list' };
const QUEUE_MODE = metrics.queue_mode; // list | stream
async function enqueueJob(job, log){
  if (QUEUE_MODE === 'stream'){
    try { await redis.xadd('rag_ingest_jobs_stream','*','job', JSON.stringify(job)); return; }
    catch(e){ log.warn('Stream enqueue failed, fallback to list: '+e.message); }
  }
  try { await redis.rpush('rag_ingest_jobs', JSON.stringify(job)); }
  catch(e){ metrics.queue_push_failures++; log.warn('Queue push failed: '+e.message); }
}
function ensureDir(p){ if (!fs.existsSync(p)) fs.mkdirSync(p,{recursive:true}); }
ensureDir(path.join(STORAGE_ROOT, BUCKET));

const docs = new Map();

fastify.post('/ingest/chunk', async (req, reply)=>{
  const { doc_id, chunk_index, total_chunks, content, mime_type='text/plain', hash, tags=[] } = req.body || {};
  if (!doc_id || typeof chunk_index!=='number' || typeof total_chunks!=='number' || !content){
    return reply.code(400).send({ error:'missing fields' });
  }
  let state = docs.get(doc_id);
  if (!state){ state = { total: total_chunks, received:0, chunks:new Map(), tags, created:Date.now() }; docs.set(doc_id,state); }
  if (state.chunks.has(chunk_index)){
    return reply.send({ status:'duplicate', doc_id, received: state.received, complete: state.received===state.total });
  }
  state.chunks.set(chunk_index, content);
  metrics.chunks_received++;
  state.received++;
  const complete = state.received === state.total;
  if (complete){
    const ordered = [...state.chunks.entries()].sort((a,b)=> a[0]-b[0]).map(e=>e[1]).join('\n');
    const folder = path.join(STORAGE_ROOT, BUCKET, doc_id);
    ensureDir(folder);
    fs.writeFileSync(path.join(folder,'raw.txt'), ordered, 'utf8');
    fs.writeFileSync(path.join(folder,'meta.json'), JSON.stringify({ doc_id, total_chunks, tags, mime_type, hash, stored_at:new Date().toISOString() },null,2));
    docs.delete(doc_id);
    metrics.docs_completed++;
    const job = { doc_id, mime_type, tags, hash, text_path: path.join(folder,'raw.txt') };
  await enqueueJob(job, fastify.log);
  }
  return reply.send({ status:'ok', doc_id, received: state.received, complete });
});

fastify.get('/ingest/status/:docId', async (req)=>{
  const docId = req.params.docId;
  const state = docs.get(docId);
  if (!state) return { doc_id: docId, status:'unknown' };
  return { doc_id: docId, received: state.received, total: state.total, complete: state.received===state.total };
});

fastify.get('/metrics', async ()=>{
  const uptimeSec = (Date.now()-metrics.start)/1000;
  let queueLen = -1; try { queueLen = await redis.llen('rag_ingest_jobs'); } catch {}
  return { service:'ingestion', uptimeSec, ...metrics, queue_len: queueLen, timestamp: new Date().toISOString() };
});

fastify.listen({ port: process.env.INGEST_PORT || 8600, host:'0.0.0.0' }).then(()=> fastify.log.info('Ingestion service listening')).catch(e=>{ console.error('Ingestion start failed', e); process.exit(1); });
