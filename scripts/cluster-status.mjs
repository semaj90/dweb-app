#!/usr/bin/env node
// Cluster status aggregator: collects service port status + NATS health snapshot.
import net from 'node:net';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ports = [
  { name: 'node-api', port: parseInt(process.env.NODE_API_PORT||'3000',10) },
  { name: 'gpu-worker', port: parseInt(process.env.GPU_WORKER_PORT||'8094',10) },
  { name: 'wasm-worker', port: parseInt(process.env.WASM_WORKER_PORT||'8095',10) },
  { name: 'go-llama', port: parseInt(process.env.GO_LLAMA_PORT||'8096',10) },
  { name: 'quic-gateway', port: parseInt(process.env.QUIC_GATEWAY_PORT||'8097',10) },
  { name: 'ws-fanout', port: 8080 }
];

function checkPort(p){
  return new Promise(res=>{
    const s = new net.Socket();
    let done=false;
    const finish = (ok)=>{ if(done) return; done=true; try{s.destroy();}catch{} res(ok); };
    s.once('connect', ()=> finish(true));
    s.once('error', ()=> finish(false));
    s.setTimeout(800, ()=> finish(false));
    s.connect(p, '127.0.0.1');
  });
}

async function fetchJson(url){
  try { const r = await fetch(url); if(!r.ok) return null; return await r.json(); } catch { return null; }
}

async function main(){
  const status = { ts: new Date().toISOString(), services: {}, nats: null };
  for (const { name, port } of ports){
    status.services[name] = { port, up: await checkPort(port) };
  }
  // Attempt to read NATS health via node-api endpoints (if exposed)
  try { status.nats = await fetchJson('http://localhost:5173/api/v1/nats/health'); } catch {}
  const outPath = resolve('.vscode/cluster-status.json');
  try { writeFileSync(outPath, JSON.stringify(status,null,2)); } catch {}
  console.log(JSON.stringify(status,null,2));
}

main();
