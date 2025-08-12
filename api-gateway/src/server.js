import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';

export function createServer() {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  // Health
  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, pid: process.pid, ts: new Date().toISOString() });
  });

  // Proxies to existing Go services
  const clusterBase = process.env.CLUSTER_BASE_URL || 'http://localhost:8090';
  const summarizerBase = process.env.SUMMARIZER_BASE_URL || 'http://localhost:8091';

  app.use('/api/cluster', createProxyMiddleware({
    target: clusterBase,
    changeOrigin: true,
    pathRewrite: { '^/api/cluster$': '/cluster', '^/api/cluster/health$': '/health' }
  }));

  app.use('/api/ai/summarize', createProxyMiddleware({
    target: summarizerBase,
    changeOrigin: true,
    pathRewrite: { '^/api/ai/summarize$': '/summarize', '^/api/ai/summarize/health$': '/health' }
  }));

  const port = parseInt(process.env.GATEWAY_PORT || '3000', 10);
  app.listen(port, () => {
    console.log(`[gateway] worker ${process.pid} listening on :${port}`);
  });
}
