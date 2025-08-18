// ingestion/grpc_ingestion_server.js - streaming ingestion server
import grpc from '@grpc/grpc-js';
import protoLoader from '@grpc/proto-loader';
import fs from 'fs';
import path from 'path';
import Redis from 'ioredis';

const PROTO_PATH = path.join(process.cwd(),'proto','ingest.proto');
const packageDef = protoLoader.loadSync(PROTO_PATH,{ keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const ingestProto = grpc.loadPackageDefinition(packageDef).ingest;

const STORAGE_ROOT = process.env.RAG_STORAGE_DIR || path.join(process.cwd(),'ingest-store');
const BUCKET = process.env.RAG_BUCKET || 'legal-documents';
fs.mkdirSync(path.join(STORAGE_ROOT, BUCKET), { recursive: true });
const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
const QUEUE_MODE = process.env.RAG_QUEUE_MODE || 'list';

async function enqueue(job){
  if (QUEUE_MODE === 'stream'){
    try { await redis.xadd('rag_ingest_jobs_stream','*','job', JSON.stringify(job)); return; } catch { /* fallback below */ }
  }
  await redis.rpush('rag_ingest_jobs', JSON.stringify(job));
}

function StreamChunks(call, cb){
  let docId=null; let total=0; let received=0; const chunks=[]; let meta={}; let tags=[]; let mime='text/plain'; let hash='';
  call.on('data', d=>{
    if (!docId) docId = d.doc_id;
    total = d.total_chunks || total;
    received++;
    if (d.mime_type) mime = d.mime_type;
    if (d.hash) hash = d.hash;
    if (d.tags && d.tags.length) tags = d.tags;
    if (d.meta) meta = { ...meta, ...d.meta };
    if (d.content) chunks.push(Buffer.from(d.content).toString('utf8'));
  });
  call.on('end', async ()=>{
    if (docId && chunks.length){
      const folder = path.join(STORAGE_ROOT, BUCKET, docId);
      fs.mkdirSync(folder,{recursive:true});
      fs.writeFileSync(path.join(folder,'raw.txt'), chunks.join('\n'),'utf8');
      fs.writeFileSync(path.join(folder,'meta.json'), JSON.stringify({ doc_id:docId, total_chunks:total, tags, mime, hash, meta, stored_at:new Date().toISOString() },null,2));
      await enqueue({ doc_id: docId, mime_type:mime, tags, hash, text_path: path.join(folder,'raw.txt') });
      cb(null, { doc_id: docId, received_chunks: received, complete:true, status:'ok', message:'stored' });
    } else {
      cb(null, { doc_id: docId||'', received_chunks: received, complete:false, status:'empty', message:'no chunks' });
    }
  });
  call.on('error', e=> console.warn('Ingestion stream error', e.message));
}

function main(){
  const server = new grpc.Server();
  server.addService(ingestProto.Ingestion.service, { StreamChunks });
  const addr = process.env.INGEST_GRPC_ADDR || '0.0.0.0:56051';
  server.bindAsync(addr, grpc.ServerCredentials.createInsecure(), err=>{
    if (err) throw err; server.start(); console.log('gRPC ingestion server listening on', addr); });
}
main();
