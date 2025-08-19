import type { RequestHandler } from '@sveltejs/kit';
import { getPipelineHistogram, getDedupeMetrics } from '$lib/services/pipeline-metrics';

export const GET: RequestHandler = async () => {
  const hist = getPipelineHistogram();
  const dedupe = getDedupeMetrics();
  return new Response(JSON.stringify({ ok: true, pipeline: hist, dedupe }), { status: 200, headers: { 'content-type': 'application/json' } });
};
