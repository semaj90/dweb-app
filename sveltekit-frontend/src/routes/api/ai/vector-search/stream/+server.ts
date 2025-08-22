
// Streaming (SSE) vector search endpoint.
// Usage: GET /api/ai/vector-search/stream?query=...&limit=8&threshold=0.2&model=...&mode=simple
// Sends events: result (per similarity row), meta (once), error (if any), done (end)

import { getEmbeddingRepository } from '$lib/server/embedding/embedding-repository';

export const GET = async ({ url }) => {
  const query = url.searchParams.get('query') || '';
  if (!query.trim()) {
    return new Response('query required', { status: 400 });
  }
  const limit = parseInt(url.searchParams.get('limit') || '8', 10);
  const thresholdParam = url.searchParams.get('threshold');
  const threshold = thresholdParam != null ? parseFloat(thresholdParam) : null;
  const model = url.searchParams.get('model') || undefined;
  const mode = (url.searchParams.get('mode') as 'simple' | 'enhanced') || 'simple';

  const encoder = new TextEncoder();
  const repo = getEmbeddingRepository();
  const started = Date.now();

  const stream = new ReadableStream({
    async start(controller) {
      function send(event: string, data: any) {
        const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
        controller.enqueue(encoder.encode(payload));
      }
      try {
        send('meta', { mode, limit, threshold, model, startedAt: new Date().toISOString() });
        const results = await repo.querySimilar(query, { limit: limit, model });
        let emitted = 0;
        for (const r of results) {
          if (threshold != null && r.score < threshold) continue;
          send('result', r);
          emitted++;
          // Slight delay for incremental UX (optional)
          await new Promise((res) => setTimeout(res, 5));
        }
        send('done', { total: emitted, elapsedMs: Date.now() - started });
      } catch (e: any) {
        send('error', { message: e?.message || String(e) });
        send('done', { total: 0, elapsedMs: Date.now() - started });
      } finally {
        controller.close();
      }
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive'
    }
  });
};
