import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { getMetricsSnapshot } from '$lib/server/logger';

export const GET: RequestHandler = async () => {
  return json({
    metrics: getMetricsSnapshot(),
    timestamp: new Date().toISOString()
  });
};
