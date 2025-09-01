
import type { RequestHandler } from './$types';

// Versioned NLP metrics endpoint aligning with FULL_STACK_INTEGRATION_COMPLETE.md docs
export const GET: RequestHandler = async () => {
  const body = renderNlpMetrics();
  return new Response(body, { status: 200, headers: { 'content-type': 'text/plain; version=0.0.4' } });
};
