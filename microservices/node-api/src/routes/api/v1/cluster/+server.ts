import type { RequestHandler } from '@sveltejs/kit';
import { readFileSync } from 'fs';

export const GET: RequestHandler = async () => {
  const ports = JSON.parse(readFileSync('cluster-status.json', 'utf-8'));
  return new Response(JSON.stringify({ ts: new Date().toISOString(), ports }), {
    status: 200,
    headers: { 'content-type': 'application/json' }
  });
};
