import type { RequestHandler } from "@sveltejs/kit";
// Orphaned content: import {

export const GET: RequestHandler = async () => {
  return new Response(JSON.stringify({ ok: true, autosolve: getAutosolveMetrics() }), { status: 200, headers: { 'content-type': 'application/json' } });
};
