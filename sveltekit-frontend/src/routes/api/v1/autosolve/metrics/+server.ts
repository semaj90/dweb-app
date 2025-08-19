import type { RequestHandler } from '@sveltejs/kit';
import { getAutosolveMetrics } from '$lib/services/pipeline-metrics';

export const GET: RequestHandler = async () => {
  return new Response(JSON.stringify({ ok: true, autosolve: getAutosolveMetrics() }), { status: 200, headers: { 'content-type': 'application/json' } });
};
