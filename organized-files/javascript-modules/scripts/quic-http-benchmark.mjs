#!/usr/bin/env node
// QUIC vs HTTP latency benchmark (simplified)
import { performance } from 'node:perf_hooks';

const RUNS = parseInt(process.env.BENCH_RUNS || '25', 10);
const HTTP_PORT = parseInt(process.env.GO_LLAMA_PORT || '8096', 10);
const QUIC_PORT = parseInt(process.env.QUIC_GATEWAY_PORT || '8097', 10);
const PATH = process.env.BENCH_PATH || '/health';

async function timeFetch(url){
  const start = performance.now();
  let ok=false; let status=0;
  try { const r = await fetch(url, { cache: 'no-store' }); status=r.status; ok=r.ok; } catch {}
  return { ms: performance.now()-start, ok, status };
}
async function series(label, base){
  const results=[]; for (let i=0;i<RUNS;i++) results.push(await timeFetch(`${base}${PATH}`));
  const times = results.map(r=>r.ms).sort((a,b)=>a-b);
  const pct = p => times[Math.min(times.length-1, Math.floor(p*(times.length-1)))];
  return { label, runs: RUNS, success: results.filter(r=>r.ok).length, p50: pct(0.5), p90: pct(0.9), p99: pct(0.99), avg: times.reduce((a,b)=>a+b,0)/times.length };
}
(async ()=>{
  const httpRes = await series('http', `http://localhost:${HTTP_PORT}`);
  const quicRes = await series('quic', `http://localhost:${QUIC_PORT}`);
  const out = { ts: new Date().toISOString(), path: PATH, http: httpRes, quic: quicRes };
  console.log(JSON.stringify(out,null,2));
})();
