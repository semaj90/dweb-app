import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { vectorSearchRequestSchema, vectorSearchResponseSchema } from '$lib/schemas/vector';
import { safeFetchJson } from '$lib/server/fetch-wrapper';

// Proxy to internal vector search microservice (assumed on 8095 or via env)
const VECTOR_BASE = process.env.VECTOR_SERVICE_URL || 'http://localhost:8095';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const payloadRaw = await request.json();
    const parse = vectorSearchRequestSchema.safeParse(payloadRaw);
    if (!parse.success) {
      return json({ error: 'Invalid request', issues: parse.error.issues }, { status: 400 });
    }
    const body = parse.data;
    const serviceUrl = `${VECTOR_BASE}/search`;
    const upstream = await safeFetchJson<any>(serviceUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      timeoutMs: 15000,
      retries: 1
    });

    if (!upstream.ok) {
      return json({ error: upstream.error || 'Vector search failed', status: upstream.status, details: upstream.data }, { status: 502 });
    }

    const validate = vectorSearchResponseSchema.safeParse(upstream.data);
    if (!validate.success) {
      return json({ error: 'Upstream response validation failed', issues: validate.error.issues }, { status: 502 });
    }

    return json({ results: validate.data.results, meta: { source: 'vector-service', latency: Date.now() } });
  } catch (err: any) {
    return json({ error: 'Unhandled vector search error', message: err.message }, { status: 500 });
  }
};
