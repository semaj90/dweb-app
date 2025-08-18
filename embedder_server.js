// embedder_server.js - gRPC ONNX embedder skeleton
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import ort from 'onnxruntime-node';
import path from 'path';
import fs from 'fs';
// import { Tokenizer } from '@huggingface/tokenizers'; // Package not available
import Fastify from 'fastify';

const PROTO_PATH = path.join(process.cwd(), 'proto', 'embed.proto');
const packageDef = protoLoader.loadSync(PROTO_PATH, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const embedProto = grpc.loadPackageDefinition(packageDef).embed;

const MODEL_PATH = process.env.ONNX_MODEL_PATH || path.join(process.cwd(), 'models', 'code-embed.onnx');
const TOKENIZER_PATH = process.env.TOKENIZER_PATH || path.join(process.cwd(), 'tokenizer.json');
const MAX_SEQ_LEN = Number(process.env.MAX_SEQ_LEN || 512);
let EMBED_DIM = Number(process.env.EMBED_DIM || 768);
let session = null;
let tokenizer = null;
const metrics = { startTime: Date.now(), singleCalls:0, batchCalls:0, modelLatencyMs:0, fallbackCalls:0 };

async function initSession(){
  if (fs.existsSync(MODEL_PATH)) {
    console.log('Loading ONNX model', MODEL_PATH);
    try {
      session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['cuda','cpu'] });
      console.log('ONNX session ready');
    } catch(e){
      console.warn('Failed to load ONNX model:', e.message);
    }
  } else {
    console.warn('Model not found at', MODEL_PATH, 'embedding will return random vectors');
  }
  // Tokenizer functionality disabled due to missing @huggingface/tokenizers package
  console.warn('HuggingFace tokenizers package not available, using naive tokenization only');
}

function naiveTokenize(text){ return text.split(/\s+/).slice(0,MAX_SEQ_LEN).map(t=> t.length % 100); }

function buildBatchedTensors(texts){
  // Tokenize & pad
  const encs = texts.map(t=>{
    if (tokenizer){
      try { const e = tokenizer.encode(t); return e.getIds().slice(0,MAX_SEQ_LEN); } catch { /* ignore */ }
    }
    return naiveTokenize(t);
  });
  const maxLen = Math.min(MAX_SEQ_LEN, Math.max(...encs.map(e=>e.length), 1));
  const batch = encs.map(e=> e.length < maxLen ? [...e, ...Array(maxLen - e.length).fill(0)] : e.slice(0,maxLen));
  const flat = Int32Array.from(batch.flat());
  const inputIds = new ort.Tensor('int32', flat, [batch.length, maxLen]);
  // attention mask (1 for real tokens else 0)
  const maskData = Int32Array.from(batch.flatMap(row => row.map(v=> v===0 ? 0 : 1)));
  const attentionMask = new ort.Tensor('int32', maskData, [batch.length, maxLen]);
  return { inputIds, attentionMask, maxLen };
}

async function runModelSingle(text){
  if (!session){ metrics.fallbackCalls++; return Array.from({length:EMBED_DIM},()=> Math.random()); }
  const { inputIds, attentionMask } = buildBatchedTensors([text]);
  const feeds = { input_ids: inputIds };
  // Some models require attention_mask - include if present in session input names
  if (session.inputNames.includes('attention_mask')) feeds.attention_mask = attentionMask;
  const t0 = Date.now();
  const out = await session.run(feeds);
  const dt = Date.now()-t0; metrics.singleCalls++; metrics.modelLatencyMs += dt;
  const firstKey = Object.keys(out)[0];
  const raw = out[firstKey];
  // Assume shape [1, hidden] or [1, seq, hidden]; pool if needed
  let dataArr = Array.from(raw.data);
  if (raw.dims.length === 3){
    const [b, s, h] = raw.dims; // pool by mean over sequence
    const pooled = new Array(h).fill(0);
    for (let i=0;i<s;i++){
      for (let j=0;j<h;j++) pooled[j]+= dataArr[i*h + j];
    }
    for (let j=0;j<h;j++) pooled[j]/=s;
    dataArr = pooled;
  }
  if (raw.dims.length >= 2){
    const hidden = raw.dims[raw.dims.length-1];
    if (hidden && hidden !== EMBED_DIM){ console.warn('[embedder] Adjust EMBED_DIM', EMBED_DIM,'->', hidden); EMBED_DIM = hidden; }
  }
  return dataArr.slice(0,EMBED_DIM);
}

async function runModelBatch(texts){
  if (!session){ metrics.fallbackCalls+=texts.length; return texts.map(()=> Array.from({length:EMBED_DIM},()=> Math.random())); }
  const { inputIds, attentionMask } = buildBatchedTensors(texts);
  const feeds = { input_ids: inputIds };
  if (session.inputNames.includes('attention_mask')) feeds.attention_mask = attentionMask;
  const t0 = Date.now();
  const out = await session.run(feeds);
  const dt = Date.now()-t0; metrics.batchCalls++; metrics.modelLatencyMs += dt;
  const firstKey = Object.keys(out)[0];
  const tensor = out[firstKey];
  const dims = tensor.dims;
  const all = Array.from(tensor.data);
  const results = [];
  if (dims.length === 2){
    const [b,h] = dims;
  for (let i=0;i<b;i++){ if (h !== EMBED_DIM) EMBED_DIM = h; results.push(all.slice(i*h, (i+1)*h).slice(0,EMBED_DIM)); }
  } else if (dims.length === 3){
    const [b,s,h] = dims;
    for (let bi=0;bi<b;bi++){
      const offset = bi*s*h;
      const pooled = new Array(h).fill(0);
      for (let i=0;i<s;i++){
        for (let j=0;j<h;j++) pooled[j]+= all[offset + i*h + j];
      }
      for (let j=0;j<h;j++) pooled[j]/=s;
  if (h !== EMBED_DIM) EMBED_DIM = h;
  results.push(pooled.slice(0,EMBED_DIM));
    }
  } else {
    // fallback treat as flat per text
    const per = Math.floor(all.length / texts.length);
  for (let i=0;i<texts.length;i++) results.push(all.slice(i*per,(i+1)*per).slice(0,EMBED_DIM));
  }
  return results;
}

async function embed(call, cb){
  try {
    const text = call.request.text || '';
    const vector = await runModelSingle(text);
    cb(null, { id: call.request.id || '', vector, token_count: vector.length });
  } catch(e){ cb(e); }
}

async function batchEmbed(call, cb){
  try {
    const texts = call.request.texts || [];
    if (!texts.length) return cb(null, { results: [] });
    const vectors = await runModelBatch(texts);
    const results = vectors.map(v=> ({ id:'', vector: v, token_count: v.length }));
    cb(null, { results });
  } catch(e){ cb(e); }
}

async function main(){
  await initSession();
  const server = new grpc.Server();
  server.addService(embedProto.Embedder.service, { Embed: embed, BatchEmbed: batchEmbed });
  const addr = process.env.EMBEDDER_ADDR || '0.0.0.0:50051';
  server.bindAsync(addr, grpc.ServerCredentials.createInsecure(), (err)=>{
    if (err) throw err;
    server.start();
    console.log('gRPC embedder listening on', addr);
  });

  // metrics HTTP
  const METRICS_PORT = Number(process.env.EMBED_METRICS_PORT || 9300);
  const fastify = Fastify({ logger:false });
  fastify.get('/metrics', async ()=>{
    const uptimeSec = (Date.now()-metrics.startTime)/1000;
    const totalCalls = metrics.singleCalls + metrics.batchCalls;
    const avgLatency = totalCalls ? metrics.modelLatencyMs / totalCalls : 0;
    return {
      service:'embedder', uptimeSec,
      singleCalls: metrics.singleCalls,
      batchCalls: metrics.batchCalls,
      fallbackCalls: metrics.fallbackCalls,
  avgModelLatencyMs: Number(avgLatency.toFixed(2)),
  embedDim: EMBED_DIM,
      timestamp: new Date().toISOString()
    };
  });
  fastify.listen({ port: METRICS_PORT, host:'0.0.0.0' }).then(()=>{
    console.log('Embedder metrics endpoint on /metrics port', METRICS_PORT);
  }).catch(e=> console.warn('Embedder metrics server failed:', e.message));
}

main().catch(e=>{ console.error(e); process.exit(1); });
