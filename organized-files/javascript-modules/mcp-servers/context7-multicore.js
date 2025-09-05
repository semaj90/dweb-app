#!/usr/bin/env node
import cluster from 'cluster';
import os from 'os';
import http from 'http';
import { WebSocketServer } from 'ws';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const CONFIG = {
  port: Number(process.env.MCP_PORT || 40000),
  workers: Number(process.env.MCP_WORKERS) || Math.min(os.cpus().length, 8),
  enableMultiCore: process.env.MCP_MULTICORE !== 'false',
  scaleHighWater: Number(process.env.MCP_SCALE_HIGH || 0.75), // CPU util threshold to scale up
  scaleLowWater: Number(process.env.MCP_SCALE_LOW || 0.25),   // CPU util threshold to scale down
  minWorkers: 1,
  healthIntervalMs: 15000,
  metricsBroadcastMs: 5000
};

// Simple rolling metrics store (primary only)
const workerMetrics = new Map();

function sampleCpu() {
  const loads = os.loadavg();
  const one = loads[0] / os.cpus().length; // normalize
  return Number(one.toFixed(2));
}

function decideScaling() {
  if (!cluster.isPrimary) return;
  const cpu = sampleCpu();
  const live = Object.keys(cluster.workers).length;
  if (cpu > CONFIG.scaleHighWater && live < CONFIG.workers) {
    cluster.fork();
    console.error(`[Context7 Multi] 🔼 Scaling up → ${live + 1} (cpu=${cpu})`);
  } else if (cpu < CONFIG.scaleLowWater && live > CONFIG.minWorkers) {
    // Pick highest id to kill
    const ids = Object.keys(cluster.workers).map(Number).sort((a, b) => b - a);
    const target = ids[0];
    cluster.workers[target]?.disconnect();
    console.error(`[Context7 Multi] 🔽 Scaling down → ${live - 1} (cpu=${cpu})`);
  }
}

async function startWorker() {
  const started = Date.now();
  const mcp = new Server({ name: 'context7-multi', version: '1.1.0' }, { capabilities: { resources: {}, tools: {} } });
  const transport = new StdioServerTransport();
  await mcp.connect(transport);

  const httpServer = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({
        status: 'healthy',
        workerId: cluster.worker?.id,
        pid: process.pid,
        uptimeSec: Math.round((Date.now() - started) / 1000)
      }));
    }
    if (req.url === '/metrics') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({
        rss: process.memoryUsage().rss,
        heapUsed: process.memoryUsage().heapUsed,
        cpuLoad: sampleCpu(),
        workerId: cluster.worker?.id,
        pid: process.pid
      }));
    }
    res.statusCode = 404; res.end('Not Found');
  });
  const port = CONFIG.port + (cluster.worker?.id || 0) - 1;
  httpServer.listen(port, () => console.error(`[Context7 Multi] Worker ${cluster.worker?.id} HTTP :${port}`));

  // Periodic metric push to primary via IPC
  setInterval(() => {
    if (process.send) {
      process.send({
        type: 'metrics',
        data: {
          workerId: cluster.worker?.id,
          rss: process.memoryUsage().rss,
          heapUsed: process.memoryUsage().heapUsed,
          cpu: sampleCpu(),
          ts: Date.now()
        }
      });
    }
  }, CONFIG.metricsBroadcastMs).unref();

  // Graceful shutdown
  const shutdown = () => {
    console.error(`[Context7 Multi] Worker ${cluster.worker?.id} shutting down`);
    httpServer.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 3000).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

if (CONFIG.enableMultiCore && cluster.isPrimary) {
  console.error(`[Context7 Multi] Primary starting target=${CONFIG.workers} basePort=${CONFIG.port}`);
  for (let i = 0; i < CONFIG.workers; i++) cluster.fork();

  cluster.on('message', (worker, msg) => {
    if (msg?.type === 'metrics') {
      workerMetrics.set(worker.id, msg.data);
    }
  });

  cluster.on('exit', (worker) => {
    console.error(`[Context7 Multi] Worker ${worker.id} died, restarting...`);
    cluster.fork();
  });

  // Expose aggregate metrics HTTP on base port -1 (primary only)
  const primaryPort = CONFIG.port - 1;
  const primaryServer = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.url === '/health') {
      return res.end(JSON.stringify({
        status: 'primary',
        workers: Object.keys(cluster.workers).length,
        target: CONFIG.workers,
        cpu: sampleCpu(),
        timestamp: Date.now()
      }));
    }
    if (req.url === '/metrics') {
      return res.end(JSON.stringify({
        primary: true,
        cpu: sampleCpu(),
        workers: Array.from(workerMetrics.values())
      }));
    }
    res.statusCode = 404; res.end('Not Found');
  });
  primaryServer.listen(primaryPort, () => console.error(`[Context7 Multi] Primary metrics HTTP :${primaryPort}`));

  // Scaling loop
  setInterval(decideScaling, CONFIG.healthIntervalMs).unref();
} else {
  startWorker().catch((e) => { console.error('Worker error', e); process.exit(1); });
}

// VS Code settings hint (documentation): set environment in tasks.json
// Example:
// {
//   "label": "MCP: Context7 Multi-Core",
//   "command": "node",
//   "args": ["mcp-servers/context7-multicore.js"],
//   "options": {
//     "env": { "MCP_MULTICORE": "true", "MCP_WORKERS": "6", "MCP_PORT": "4100" }
//   }
// }
