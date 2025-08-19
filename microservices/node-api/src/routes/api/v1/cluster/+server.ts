import type { RequestHandler } from '@sveltejs/kit';
import { healthSnapshot } from '$lib/services/nats-messaging-service';

async function portUp(port: number, timeout=300){
  return new Promise<boolean>(res=>{
    const net = require('node:net');
    const s = new net.Socket();
    let done=false; const finish=(ok:boolean)=>{ if(done) return; done=true; try{s.destroy();}catch{}; res(ok); };
    s.once('connect', ()=> finish(true));
    s.once('error', ()=> finish(false));
    s.setTimeout(timeout, ()=> finish(false));
    s.connect(port, '127.0.0.1');
  });
}

export const GET: RequestHandler = async () => {
  const ports = [
    { name:'node-api', port: parseInt(process.env.NODE_API_PORT||'3000',10) },
    { name:'gpu-worker', port: parseInt(process.env.GPU_WORKER_PORT||'8094',10) },
    { name:'wasm-worker', port: parseInt(process.env.WASM_WORKER_PORT||'8095',10) },
    { name:'go-llama', port: parseInt(process.env.GO_LLAMA_PORT||'8096',10) },
    { name:'quic-gateway', port: parseInt(process.env.QUIC_GATEWAY_PORT||'8097',10) },
    { name:'ws-fanout', port: 8080 }
  ];
  const statuses = {} as Record<string,{port:number,up:boolean}>;
  for (const p of ports){ statuses[p.name] = { port: p.port, up: await portUp(p.port) }; }
  const body = { ts: new Date().toISOString(), services: statuses, nats: healthSnapshot() };
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type':'application/json' } });
};
