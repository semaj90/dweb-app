#!/usr/bin/env node
import cluster from 'cluster';
import os from 'os';
import http from 'http';
import { WebSocketServer } from 'ws';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const CONFIG = {
  port: Number(process.env.MCP_PORT || 40000),
  workers: Math.min(os.cpus().length, 8),
  enableMultiCore: process.env.MCP_MULTICORE !== 'false'
};

async function startWorker() {
  const mcp = new Server({ name: 'context7-multi', version: '1.0.0' }, { capabilities: { resources: {}, tools: {} } });
  const transport = new StdioServerTransport();
  await mcp.connect(transport);
  // Lightweight HTTP for health/metrics per worker
  const httpServer = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ status: 'healthy', workerId: cluster.worker?.id, pid: process.pid }));
    }
    res.statusCode = 404; res.end('Not Found');
  });
  const port = CONFIG.port + (cluster.worker?.id || 0) - 1; // spread workers across ports starting at base
  httpServer.listen(port, () => console.error(`[Context7 Multi] Worker ${cluster.worker?.id} HTTP :${port}`));
}

if (CONFIG.enableMultiCore && cluster.isPrimary) {
  console.error(`[Context7 Multi] Primary starting ${CONFIG.workers} workers at base port ${CONFIG.port}`);
  for (let i = 0; i < CONFIG.workers; i++) cluster.fork();
  cluster.on('exit', (worker) => {
    console.error(`[Context7 Multi] Worker ${worker.id} died, restarting...`);
    cluster.fork();
  });
} else {
  startWorker().catch((e) => { console.error('Worker error', e); process.exit(1); });
}
