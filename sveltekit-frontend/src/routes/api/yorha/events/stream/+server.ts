
import type { RequestHandler } from './$types.js';
import { URL } from "url";

const DEFAULT_INTERVAL = 3000;

export interface StreamMetricEnvelope { type: string; data: any; ts: string }

function collectMetrics() {
  return {
    cpu: 5 + Math.random() * 25,
    memMB: Math.round(process.memoryUsage().rss / 1024 / 1024),
    vectorQueriesPerMin: Math.floor(Math.random() * 20),
    gpuQueueDepth: Math.floor(Math.random() * 4),
    timestamp: Date.now()
  };
}

export const GET: RequestHandler = async ({ url }) => {
  const intervalMs = Number(url.searchParams.get('interval') || DEFAULT_INTERVAL);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      function send(obj: StreamMetricEnvelope) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
      }
      send({ type: 'sse_hello', data: { message: 'YoRHa stream online' }, ts: new Date().toISOString() });
      const interval = setInterval(() => {
        send({ type: 'system_metrics', data: collectMetrics(), ts: new Date().toISOString() });
      }, Math.max(1000, intervalMs));
      const keepAlive = setInterval(() => controller.enqueue(encoder.encode(': ping\n\n')), 25000);
      const abort = () => { clearInterval(interval); clearInterval(keepAlive); controller.close(); };
      (globalThis as any)._yoAbort = abort; // placeholder
    }
  });
  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Connection': 'keep-alive',
      'Transfer-Encoding': 'chunked'
    }
  });
};
