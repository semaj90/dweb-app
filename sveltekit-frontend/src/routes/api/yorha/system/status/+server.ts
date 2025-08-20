import type { RequestHandler } from '@sveltejs/kit';

let startTime = Date.now();
let requestCount = 0;

interface YoRHaSystemStatus {
  database: { connected: boolean; latency: number; activeConnections: number; queryCount: number };
  backend: { healthy: boolean; uptime: number; activeServices: number; cpuUsage: number; memoryUsage: number };
  frontend: { renderFPS: number; componentCount: number; activeComponents: number; webGPUEnabled: boolean };
}

function collectStatus(): YoRHaSystemStatus {
  const mem = process.memoryUsage();
  const rssMB = Math.round(mem.rss / 1024 / 1024);
  const cpuApprox = 5 + Math.random() * 20; // placeholder approximation
  return {
    database: {
      connected: true,
      latency: 5 + Math.random() * 10,
      activeConnections: 4 + Math.floor(Math.random() * 3),
      queryCount: requestCount
    },
    backend: {
      healthy: true,
      uptime: Math.floor((Date.now() - startTime) / 1000),
      activeServices: 5,
      cpuUsage: Number(cpuApprox.toFixed(2)),
      memoryUsage: rssMB
    },
    frontend: {
      renderFPS: 60,
      componentCount: 42,
      activeComponents: 30 + Math.floor(Math.random() * 5),
      webGPUEnabled: true
    }
  };
}

export const GET: RequestHandler = async () => {
  requestCount++;
  const status = collectStatus();
  return new Response(JSON.stringify(status), {
    status: 200,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }
  });
};
