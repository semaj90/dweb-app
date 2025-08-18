// gateway.js - Fastify API gateway for errors -> Redis Stream
import Fastify from 'fastify';
import Redis from 'ioredis';
import crypto from 'crypto';
import fs from 'fs';
import S from 'fluent-json-schema';
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import axios from 'axios';
import { retrieve, buildPrompt, degradedFlags } from './rag/retrieval.js';
import { detectIntent } from './chat/intent.js';

const fastify = Fastify({ logger: true });
const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
// Optional embedder client
let embedClient = null;
try {
  const PROTO_PATH = new URL('./proto/embed.proto', import.meta.url).pathname;
  const packageDef = protoLoader.loadSync(PROTO_PATH, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
  const embedProto = grpc.loadPackageDefinition(packageDef).embed;
  embedClient = new embedProto.Embedder(process.env.EMBEDDER_GRPC_ADDR || 'localhost:50051', grpc.credentials.createInsecure());
} catch(e){ fastify.log.warn('Embedder client init failed: '+ e.message); }

const errorSchema = S.object()
  .prop('file', S.string().required())
  .prop('line', S.integer().required())
  .prop('col', S.integer())
  .prop('message', S.string().required())
  .prop('raw', S.string())
  .prop('source', S.string());

async function appendFile(file, data){ await fs.promises.appendFile(file, data); }

fastify.post('/v1/error', { schema: { body: errorSchema } }, async (req, reply) => {
  const body = req.body;
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2,9)}`,
    file: body.file,
    line: body.line,
    col: body.col || 0,
    message: body.message,
    raw: body.raw || '',
    source: body.source || 'checker',
    ts: new Date().toISOString()
  };
  const fp = crypto.createHash('sha256').update(JSON.stringify({file:entry.file,line:entry.line,message:entry.message})).digest('hex');
  const exists = await redis.sismember('errors_seen', fp);
  if (!exists){
    await redis.sadd('errors_seen', fp);
    await redis.xadd('errors_stream','*', 'id',entry.id,'file',entry.file,'line',String(entry.line),'col',String(entry.col),'message',entry.message,'raw',entry.raw,'source',entry.source,'ts',entry.ts);
  }
  await appendFile('errorlog.jsonl', JSON.stringify(entry)+'\n');
  return reply.code(201).send({ ok:true, id: entry.id, dedup: !!exists });
});

fastify.get('/health', async ()=>({ ok:true, redis: await redis.ping() }));

// RAG query endpoint
fastify.post('/rag/query', async (req, reply)=>{
  const { query, topK = 6 } = req.body || {};
  if (!query) return reply.code(400).send({ error:'query required' });
  const intent = await detectIntent(query);
  let vector = [];
  if (embedClient){
    vector = await new Promise(res=>{
      embedClient.Embed({ text: query }, (err, r)=>{ if (err||!r) return res([]); res(r.vector||[]); });
    });
  }
  if (!vector.length){
    vector = Array.from({length:768},()=>Math.random());
  }
  const ranked = (await retrieve({ queryEmbedding: vector })).slice(0, topK);
  const flags = degradedFlags();
  // Intent-specific instruction tailoring
  const intentInstructions = {
    summarize: 'Summarize key legal points clearly. Provide bullet list of clauses and a concise final paragraph. Cite sources as [C#].',
    risk: 'Identify potential legal risks, mitigation strategies, and cite supporting context chunks. Provide a short risk matrix.',
    definition: 'Provide precise legal definition, typical usage, and jurisdictional nuances if present. Cite sources.',
    default: 'Provide an accurate legal analysis, structured answer, and cite sources.'
  };
  const basePrompt = buildPrompt(query, ranked);
  const extra = intentInstructions[intent.intent] || intentInstructions.default;
  const prompt = basePrompt + '\n\nIntentMode: ' + intent.intent + '\n' + extra + (flags.neoAvailable ? '\nGraph enrichment active: incorporate entity relationships when forming answer.' : '');
  let answer = '';
  try {
    const r = await axios.post((process.env.OLLAMA_URL||'http://localhost:11434') + '/api/generate', { model:'gemma3-legal', prompt, stream:false });
    answer = r.data?.response || '';
  } catch(e){ answer = 'Degraded mode: language model unavailable.'; }
  return { query, intent, answer, sources: ranked.map((r,i)=>({ id:r.id, high_score:r.high_score, snippet:r.text?.slice(0,180) })), degraded: !embedClient, graph: { neoAvailable: flags.neoAvailable } };
});

fastify.listen({ port: process.env.PORT || 3000, host: '0.0.0.0' }).then(()=> fastify.log.info('gateway listening'));
