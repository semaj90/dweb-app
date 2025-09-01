import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { ragRequestSchema, ragResponseSchema } from '$lib/schemas/vector';
import { safeFetchJson } from '$lib/server/fetch-wrapper';

const RAG_BASE = process.env.RAG_SERVICE_URL || 'http://localhost:8094';

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json().catch(() => ({}));
  const parsed = ragRequestSchema.safeParse(body);
  if (!parsed.success) {
    return json({ error: 'Invalid request', details: parsed.error.flatten() }, { status: 400 });
  }
  const upstream = await safeFetchJson<any>(`${RAG_BASE}/rag/enhanced`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(parsed.data)
  });
  if (!upstream.ok) {
    return json({ error: 'Upstream error', detail: upstream.error }, { status: upstream.status || 502 });
  }
  const validated = ragResponseSchema.safeParse(upstream.data);
  if (!validated.success) {
    return json({ error: 'Invalid upstream response' }, { status: 502 });
  }
  return json(validated.data);
};
