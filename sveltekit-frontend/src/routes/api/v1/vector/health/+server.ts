import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { healthResponseSchema } from '$lib/schemas/vector';
import { safeFetchJson } from '$lib/server/fetch-wrapper';

const VECTOR_BASE = process.env.VECTOR_SERVICE_URL || 'http://localhost:8095';

export const GET: RequestHandler = async () => {
  const upstream = await safeFetchJson<any>(`${VECTOR_BASE}/health`, { timeoutMs: 5000 });
  if (!upstream.ok) {
    return json({ status: 'unhealthy', error: upstream.error }, { status: 200 });
  }
  const parse = healthResponseSchema.safeParse(upstream.data);
  return json(parse.success ? parse.data : { status: 'degraded' });
};
