#!/usr/bin/env node
// Periodically poll backend QUIC stats endpoint (placeholder or external service)
// and POST samples to local Svelte endpoint which updates in-process metrics.
import fetch from 'node-fetch';

const INTERVAL_MS = process.env.QUIC_POLL_INTERVAL_MS ? parseInt(process.env.QUIC_POLL_INTERVAL_MS,10) : 5000;
const SOURCE = process.env.QUIC_SOURCE_URL || 'http://localhost:8090/quic/internal-stats';

async function cycle(){
  try {
    const res = await fetch(SOURCE, { timeout: 3000 });
    if(!res.ok) throw new Error('Bad status '+res.status);
    const data = await res.json();
    // Expect shape: { connections, streams, errors, latencySamples: [..], recentError:boolean }
    const payload = {
      total_connections: data.connections ?? 0,
      total_streams: data.streams ?? 0,
      total_errors: data.errors ?? 0,
      latencySamples: Array.isArray(data.latencySamples) ? data.latencySamples.slice(-10) : [],
      errorOccurred: !!data.recentError
    };
    await fetch('http://localhost:5173/api/v1/quic/push', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) });
    process.stdout.write(`QUIC poll ok: latencies=${payload.latencySamples.length}\n`);
  } catch(e){
    process.stderr.write('QUIC poll failed: '+ e.message + '\n');
  }
}

setInterval(cycle, INTERVAL_MS);
cycle();
