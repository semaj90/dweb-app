// GPU Worker with proper server implementation
import http from 'http';
import { Worker } from 'worker_threads';

export async function runGPUWorker(payload: unknown) {
  // TODO: integrate with shared webgpu pipeline or offload via worker_threads
  return { ok: true, mode: 'cpu-fallback', inputSize: JSON.stringify(payload).length };
}

// Start GPU worker server if this file is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const port = process.env.WORKER_PORT || 8094;
  
  const server = http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    
    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200);
      res.end(JSON.stringify({ 
        status: 'healthy', 
        type: 'gpu-worker',
        pid: process.pid,
        uptime: process.uptime(),
        memory: process.memoryUsage()
      }));
      return;
    }

    if (req.method === 'POST' && req.url === '/process') {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', async () => {
        try {
          const payload = JSON.parse(body);
          const result = await runGPUWorker(payload);
          res.writeHead(200);
          res.end(JSON.stringify(result));
        } catch (error) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'Invalid JSON payload' }));
        }
      });
      return;
    }

    res.writeHead(404);
    res.end(JSON.stringify({ error: 'Not found' }));
  });

  server.listen(port, () => {
    console.log(`[GPU-WORKER] Listening on port ${port}`);
    console.log(`[GPU-WORKER] Health check: http://localhost:${port}/health`);
  });

  process.on('SIGTERM', () => {
    console.log('[GPU-WORKER] Shutting down gracefully...');
    server.close(() => process.exit(0));
  });
}
