import stream from "stream";
import type { RequestHandler } from './$types';

// SSE streaming endpoint with streamId + interrupt + optional summary
import { createStream, recordToken, removeStream, generateSummary, getStream, cachedSummary } from '$lib/server/ragStreamRegistry';
// Use process.env directly (avoids type issues if $env modules not generated)
const privateEnv: Record<string, string | undefined> = (process as any).env || {};
const EXTERNAL_SUMMARIZE = '/api/rag?action=summarize';

// Placeholder token generator simulating model output
// Real model pipeline placeholder: replace with integration to backend model / inference server
async function* modelPipeline(query: string, signal: AbortSignal) {
  // Strategy: if BACKEND_RAG_STREAM_ENDPOINT set, proxy tokens from there, else fallback simple echo
  const backend = privateEnv.BACKEND_RAG_STREAM_ENDPOINT;
  if (backend) {
    const resp = await fetch(backend, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
      signal,
    });
    if (resp.ok && resp.body) {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (line) {
            if (signal.aborted) return;
            yield line + ' ';
          }
        }
      }
      if (buf.trim()) yield buf.trim() + ' ';
      return;
    }
  }
  // Fallback minimal generation
  const fallback = (
    query + ' — providing structured legal analysis, obligations, liabilities, summary and risks.'
  ).split(/\s+/);
  for (const w of fallback) {
    if (signal.aborted) return;
    await new Promise((r) => setTimeout(r, 50));
    yield w + ' ';
  }
}

async function externalSummarize(text: string, fetchFn: typeof fetch): Promise<string | undefined> {
  try {
    const res = await fetchFn(EXTERNAL_SUMMARIZE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, length: 'short' }),
    });
    if (!res.ok) return undefined;
    const j = await res.json();
    return j.summary || undefined;
  } catch {
    return undefined;
  }
}

export const POST: RequestHandler = async ({ request, fetch }) => {
  try {
    const { query } = await request.json();
    if (!query) return new Response('query required', { status: 400 });

    const stream = createStream();

    const encoder = new TextEncoder();
    const headers = new Headers({
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'x-stream-id': stream.id,
    });

    const readable = new ReadableStream({
      async start(controller) {
        controller.enqueue(encoder.encode(`event: meta\ndata: {"streamId":"${stream.id}"}\n\n`));
        try {
          for await (const tok of modelPipeline(query, stream.controller.signal)) {
            recordToken(stream.id, tok);
            // If interrupted gracefully, stop early and emit summary
            const meta = getStream(stream.id);
            if (meta?.interrupted) break;
            controller.enqueue(encoder.encode(`event: token\ndata: ${tok}\n\n`));
          }
          // After token streaming, or early interruption, attempt a summary
          const meta = getStream(stream.id);
          const fullText = (meta?.tokens ?? []).join(' ').replace(/\s+/g, ' ').trim();
          try {
            let sum = await externalSummarize(fullText, fetch);
            if (!sum) sum = await cachedSummary(fullText, 3);
            if (!sum) sum = generateSummary(stream.id, 3);
            if (sum && meta) {
              meta.summarySent = true;
              controller.enqueue(
                encoder.encode(`event: summary\ndata: ${sum.replace(/\n/g, ' ')}\n\n`)
              );
            }
          } catch (summaryErr) {
            // Non-fatal; proceed to done
            controller.enqueue(
              encoder.encode(
                `event: error\ndata: summary_failed: ${
                  (summaryErr as any)?.message || 'unknown'
                }\n\n`
              )
            );
          }
          controller.enqueue(encoder.encode(`event: done\ndata: end\n\n`));
        } catch (e: any) {
          if (stream.controller.signal.aborted) {
            controller.enqueue(encoder.encode(`event: done\ndata: aborted\n\n`));
          } else {
            controller.enqueue(encoder.encode(`event: error\ndata: ${e.message}\n\n`));
          }
        } finally {
          removeStream(stream.id);
          controller.close();
        }
      },
    });

    return new Response(readable, { status: 200, headers });
  } catch (e: any) {
    return new Response(`error: ${e.message}`, { status: 500 });
  }
};

// removed unused recordedText helper (was causing noUnusedLocals error)
