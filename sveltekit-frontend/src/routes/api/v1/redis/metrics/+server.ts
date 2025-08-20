import type { RequestHandler } from '@sveltejs/kit';
import type { RequestHandler } from "@sveltejs/kit";
pollRedisHealth, getRedisMetrics

let lastPoll = 0;

export const GET: RequestHandler = async () => {
  const now = Date.now();
  if (now - lastPoll > 5000) { // poll at most every 5s
    lastPoll = now;
    try { await pollRedisHealth(); } catch {}
  }
  return new Response(JSON.stringify({ redis: getRedisMetrics() }), { headers: { 'Content-Type': 'application/json' } });
};
