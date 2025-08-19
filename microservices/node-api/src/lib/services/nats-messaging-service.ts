// Enhanced NATS messaging service with:
// - Circuit breaker & retry
// - Pending publish queue
// - Subject whitelist & payload size limit
// - Prometheus metrics exporter
// - Health snapshot
// - Optional Redis recent message cache

import { connect, StringCodec } from 'nats';
import { db } from '../db/drizzle';
import { nats_messages, pipeline_logs } from '../db/schema';
// Dynamic loaders for TS modules (avoids direct .ts static import in plain JS runtime)
let runGPUWorker = async (payload) => ({ ok: true, skipped: true, reason: 'gpu-worker not loaded', payloadSize: JSON.stringify(payload||{}).length });
let runWasmWorker = async () => null;
let callGoLLM = async () => null;
try {
  const gpuMod = await import('./gpu-worker').catch(()=>null);
  if (gpuMod?.runGPUWorker) runGPUWorker = gpuMod.runGPUWorker;
  const wasmMod = await import('./wasm-worker').catch(()=>null);
  if (wasmMod?.runWasmExample) runWasmWorker = wasmMod.runWasmExample;
  const goMod = await import('./go-llama-client').catch(()=>null);
  if (goMod?.callGoLLM) callGoLLM = goMod.callGoLLM;
} catch {}
// logger.ts is TypeScript; when executed under Vite/adapter-node it's compiled. For direct node selftest we guard import.
let logger;
try { ({ logger } = await import('./logger')); } catch { logger = { info: console.log, error: console.error, warn: console.warn, debug: console.debug }; }

// Runtime config (duplicated minimal subset to avoid direct TS import during standalone node execution)
const ENV = process.env;
const NATS_URL = ENV.NATS_URL || 'nats://127.0.0.1:4222';
const SERVICE_NAME = ENV.NODE_API_SERVICE_NAME || 'node-api';
const MAX_PAYLOAD_KB = parseInt(ENV.NODE_API_MAX_PAYLOAD_KB || '256', 10);
const SUBJECT_WHITELIST = (ENV.NODE_API_SUBJECT_WHITELIST || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);
const REDIS_URL = ENV.REDIS_URL;

let nc = null;
let redisClient = null;
let redisInitAttempted = false;

export const NATS_SUBJECTS = {
  EVIDENCE_UPLOAD: 'evidence.upload',
  AI_RESPONSE: 'ai.response'
};
// Internal state & metrics
const sc = StringCodec();
const metrics = {
  publishTotal: 0,
  publishFailures: 0,
  subscriptions: new Set(),
  queueBacklog: 0,
  latency: { values: [], maxStored: 200 }, // rolling window (ms)
  lastError: null,
  lastPublishTs: null,
  lastConnectTs: null
};

// Circuit breaker state
const circuit = {
  failures: 0,
  open: false,
  halfOpen: false,
  threshold: parseInt(ENV.NATS_CB_THRESHOLD || '5', 10),
  coolDownMs: parseInt(ENV.NATS_CB_COOLDOWN_MS || '30000', 10),
  nextAttempt: 0
};

// Pending publishes while disconnected / open
const pendingQueue = [];
const MAX_QUEUE = parseInt(ENV.NATS_MAX_QUEUE || '500', 10);

function subjectAllowed(subject){
  return SUBJECT_WHITELIST.length === 0 || SUBJECT_WHITELIST.includes(subject);
}

async function ensureRedis(){
  if (redisClient || redisInitAttempted || !REDIS_URL) return redisClient;
  redisInitAttempted = true;
  try {
    const { default: Redis } = await import('ioredis');
    redisClient = new Redis(REDIS_URL, { lazyConnect: true });
    await redisClient.connect().catch(()=>{});
  } catch (err){
    logger.warn?.('[NATS] Redis init skipped:', err.message);
  }
  return redisClient;
}

function recordLatency(ms){
  const arr = metrics.latency.values;
  arr.push(ms);
  if (arr.length > metrics.latency.maxStored) arr.shift();
}

function openCircuit(err){
  if (circuit.open) return;
  circuit.open = true;
  circuit.halfOpen = false;
  circuit.nextAttempt = Date.now() + circuit.coolDownMs;
  metrics.lastError = err?.message || String(err);
  logger.error('[NATS] Circuit opened:', metrics.lastError);
}

function evaluateCircuitSuccess(){
  circuit.failures = 0;
  if (circuit.open || circuit.halfOpen){
    circuit.open = false;
    circuit.halfOpen = false;
    logger.info('[NATS] Circuit closed');
  }
}

function recordFailure(err){
  circuit.failures++;
  metrics.lastError = err?.message || String(err);
  if (circuit.failures >= circuit.threshold){
    openCircuit(err);
  }
}

export function healthSnapshot(){
  return {
    connected: !!nc,
    circuitOpen: circuit.open,
    circuitHalfOpen: circuit.halfOpen,
    failures: circuit.failures,
    queueBacklog: pendingQueue.length,
    lastError: metrics.lastError,
    lastPublishTs: metrics.lastPublishTs,
    lastConnectTs: metrics.lastConnectTs,
    publishTotal: metrics.publishTotal,
    publishFailures: metrics.publishFailures
  };
}

export function renderMetrics(){
  const lines = [];
  lines.push('# HELP nats_publish_total Total NATS publish attempts');
  lines.push('# TYPE nats_publish_total counter');
  lines.push(`nats_publish_total ${metrics.publishTotal}`);
  lines.push('# HELP nats_publish_failures_total Total failed NATS publish attempts');
  lines.push('# TYPE nats_publish_failures_total counter');
  lines.push(`nats_publish_failures_total ${metrics.publishFailures}`);
  lines.push('# HELP nats_subscriptions Current subscription count');
  lines.push('# TYPE nats_subscriptions gauge');
  lines.push(`nats_subscriptions ${metrics.subscriptions.size}`);
  lines.push('# HELP nats_queue_backlog Pending publish queue size');
  lines.push('# TYPE nats_queue_backlog gauge');
  lines.push(`nats_queue_backlog ${pendingQueue.length}`);
  // latency quantiles
  const lats = [...metrics.latency.values].sort((a,b)=>a-b);
  function quantile(q){ if(lats.length===0) return 0; const idx = Math.min(lats.length-1, Math.floor(q*(lats.length-1))); return lats[idx]; }
  const quantiles = [0.5,0.9,0.99];
  for (const q of quantiles){
    lines.push(`# HELP nats_message_latency_ms Latency quantiles for publish (rolling window)`);
    lines.push(`# TYPE nats_message_latency_ms summary`);
    lines.push(`nats_message_latency_ms{quantile="${q}"} ${quantile(q).toFixed(2)}`);
  }
  return lines.join('\n') + '\n';
}

export async function getNATSService(){
  if (nc) return nc;
  // Circuit gating
  if (circuit.open && Date.now() < circuit.nextAttempt){
    throw new Error('Circuit open - skipping new connection');
  }
  if (circuit.open && Date.now() >= circuit.nextAttempt){
    circuit.halfOpen = true; // try a half-open attempt
  }
  try {
    nc = await connect({ servers: [NATS_URL], name: SERVICE_NAME });
    metrics.lastConnectTs = Date.now();
    nc.closed().then(err => { if(err) logger.error('NATS closed with error:', err); nc = null; });
    evaluateCircuitSuccess();
    flushQueue();
  } catch (err){
    recordFailure(err);
    throw err;
  }
  return nc;
}

function enqueue(subject, payload){
  if (pendingQueue.length >= MAX_QUEUE){
    pendingQueue.shift(); // drop oldest
  }
  pendingQueue.push({ subject, payload, ts: Date.now(), attempts: 0 });
  metrics.queueBacklog = pendingQueue.length;
}

async function flushQueue(){
  if (!nc) return;
  for (let i=0; i<pendingQueue.length; i++){
    const item = pendingQueue[0];
    try {
      await internalPublish(item.subject, item.payload, true);
      pendingQueue.shift();
    } catch (err){
      // stop flushing on first error to prevent hot loop
      break;
    }
  }
  metrics.queueBacklog = pendingQueue.length;
}

function sizeOk(payload){
  try { const bytes = Buffer.byteLength(JSON.stringify(payload), 'utf8'); return bytes <= MAX_PAYLOAD_KB * 1024; } catch { return false; }
}

function genTraceId(){
  try { return crypto.randomUUID(); } catch { return Math.random().toString(36).slice(2); }
}

async function internalPublish(subject, payload, isQueueFlush=false){
  const start = Date.now();
  const conn = await getNATSService();
  const traceId = genTraceId();
  const enriched = { traceId, ts: Date.now(), ...payload };
  conn.publish(subject, sc.encode(JSON.stringify(enriched)));
  metrics.publishTotal++;
  metrics.lastPublishTs = Date.now();
  recordLatency(metrics.lastPublishTs - start);
  try {
    await db.insert(nats_messages).values({ subject, payload: JSON.stringify(enriched), created_at: new Date() });
  } catch (e){
    logger.error('Failed to log NATS message', e);
  }
  try {
    const rc = await ensureRedis();
    if (rc){
      await rc.lpush('recent:nats:messages', JSON.stringify({ subject, traceId, ts: Date.now() }));
      await rc.ltrim('recent:nats:messages', 0, 199);
    }
  } catch (e){ /* ignore */ }
  if (!isQueueFlush) logger.info?.('[NATS publish]', subject, { traceId });
}

export async function publishMessage(subject, payload){
  if (!subjectAllowed(subject)){
    metrics.publishFailures++;
    return { ok: false, error: 'Subject not allowed' };
  }
  if (!sizeOk(payload)){
    metrics.publishFailures++;
    return { ok: false, error: 'Payload too large' };
  }
  if (circuit.open){
    enqueue(subject, payload);
    return { ok: false, queued: true, error: 'Circuit open' };
  }
  try {
    await internalPublish(subject, payload);
    return { ok: true };
  } catch (err){
    metrics.publishFailures++;
    recordFailure(err);
    enqueue(subject, payload);
    return { ok: false, queued: true, error: err.message };
  }
}

export async function subscribe(subject, callback){
  const conn = await getNATSService();
  metrics.subscriptions.add(subject);
  const sub = conn.subscribe(subject);
  (async () => {
    for await (const m of sub){
      try {
        const parsed = JSON.parse(sc.decode(m.data));
        callback(parsed);
        // If this is an evidence upload, run AI/RAG pipeline chain
        if (subject === NATS_SUBJECTS.EVIDENCE_UPLOAD) {
          try {
            const base = parsed;
            const traceId = base.traceId || genTraceId();
            // GPU Stage
            let gpuResult = null;
            try { gpuResult = await runGPUWorker(base); } catch(e){ logger.warn('GPU worker failed', e.message); }
            await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'gpu', ok: !!gpuResult });
            // WASM Stage
            let wasmResult = null;
            try { wasmResult = await runWasmWorker(1,1); } catch(e){ logger.warn('WASM worker failed', e.message); }
            await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'wasm', ok: !!wasmResult });
            // Embedding Stage with hash + skip
            let embedding = null;
            let embeddingHash = null;
            try {
              const text = base?.data?.text || base?.text || base?.content || JSON.stringify(base).slice(0,2048);
              if (text) {
                // simple hash (FNV-1a like)
                let h = 2166136261; for (let i=0;i<text.length;i++){ h ^= text.charCodeAt(i); h = (h * 16777619) >>> 0; }
                embeddingHash = 'e:'+h.toString(16);
                // naive in-memory cache per process (could move to Redis)
                globalThis.__embeddingCache = globalThis.__embeddingCache || new Map();
                let redisTried = false;
                let redisHit = false;
                if (globalThis.__embeddingCache.has(embeddingHash)) {
                  embedding = globalThis.__embeddingCache.get(embeddingHash);
                  logger.debug?.('Embedding cache hit (memory)', embeddingHash);
                } else {
                  // Try Redis shared cache first
                  try {
                    const rc = await ensureRedis();
                    if (rc) {
                      redisTried = true;
                      const cached = await rc.get('embed:'+embeddingHash);
                      if (cached) {
                        embedding = JSON.parse(cached);
                        redisHit = true;
                        globalThis.__embeddingCache.set(embeddingHash, embedding);
                        logger.debug?.('Embedding cache hit (redis)', embeddingHash);
                      }
                    }
                  } catch(e){ logger.warn?.('Redis embedding lookup failed', e.message); }
                  if (!embedding) {
                    const embedEndpoint = process.env.EMBED_ENDPOINT || 'http://localhost:11434/api/embeddings';
                    const model = process.env.EMBED_MODEL || 'nomic-embed-text';
                    const er = await fetch(embedEndpoint, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model, input: text }) });
                    if (er.ok){ const ej = await er.json(); embedding = ej?.data?.[0]?.embedding || ej?.embedding || null; }
                    if (embedding) {
                      globalThis.__embeddingCache.set(embeddingHash, embedding);
                      // Write-through to Redis
                      try { const rc = await ensureRedis(); if (rc) await rc.set('embed:'+embeddingHash, JSON.stringify(embedding), 'EX', 60*60*24); } catch(e){ /* ignore */ }
                    }
                  }
                  await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'embedding_cache', redisTried, redisHit });
                }
              }
            } catch (e){ logger.warn('Embedding stage failed', e.message); await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'embedding', error: e.message }); }
            await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'embedding', ok: !!embedding });
            // Retrieval Stage
            let retrieval = null; let contextDocs = [];
            try {
              const queryText = base?.data?.query || base?.query || base?.text || 'contract law';
              const vsUrl = process.env.VECTOR_SEARCH_ENDPOINT || 'http://localhost:5173/api/ai/vector-search';
              const vr = await fetch(vsUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ query: queryText, model: 'claude', limit: 5 }) });
              if (vr.ok){ retrieval = await vr.json(); contextDocs = (retrieval.results||[]).map(r=>r.content||r.text||'').slice(0,5); }
            } catch(e){ logger.warn('Retrieval stage failed', e.message); await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'retrieval', error: e.message }); }
            await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'retrieval', k: retrieval?.results?.length || 0 });
            // Go LLM Stage with retry/backoff
            let llmResult = null;
            if (callGoLLM){
              const maxAttempts = 3;
              for (let attempt=1; attempt<=maxAttempts; attempt++){
                try {
                  llmResult = await callGoLLM({ gpuResult, wasmResult, context: contextDocs });
                  if (llmResult && !llmResult.error) break;
                } catch(e){ logger.warn(`Go LLM attempt ${attempt} failed`, e.message); if (attempt===maxAttempts){ await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'llm', error: e.message }); } }
                const backoff = 250 * Math.pow(2, attempt-1);
                await new Promise(r=>setTimeout(r, backoff));
              }
            }
            await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, stage: 'llm', ok: !!llmResult });
            // Persist pipeline log
            try {
              await db.insert(pipeline_logs).values({
                message_id: traceId,
                gpu: gpuResult ? JSON.stringify(gpuResult) : null,
                wasm: wasmResult ? JSON.stringify(wasmResult) : null,
                llm: llmResult ? JSON.stringify(llmResult) : null,
                embedding: embedding ? JSON.stringify(embedding) : null,
                embedding_hash: embeddingHash || null,
                retrieval: retrieval ? JSON.stringify(retrieval) : null,
                context: contextDocs.length ? JSON.stringify(contextDocs) : null,
                created_at: new Date()
              });
            } catch (e){ logger.error('Failed to log pipeline', e); }
            // Final publish
            await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { id: traceId, final: true, llmResult, gpuResult, wasmResult, context: contextDocs, embeddingPresent: !!embedding, retrievalCount: retrieval?.results?.length || 0 });
          } catch (pipelineErr){
            logger.error('Pipeline processing error', pipelineErr);
          }
        }
      } catch(err){
        logger.error('NATS parse error', err);
      }
    }
  })();
  logger.info?.('[NATS subscribe]', subject);
}

// Self-test: loopback publish & subscribe
if (process.argv.includes('--selftest')){
  (async () => {
    await subscribe(NATS_SUBJECTS.AI_RESPONSE, (msg) => logger.info('Loopback received', msg));
    await publishMessage(NATS_SUBJECTS.AI_RESPONSE, { test: true, ts: Date.now() });
    setTimeout(()=>{ console.log(renderMetrics()); process.exit(0); }, 1500);
  })();
}
