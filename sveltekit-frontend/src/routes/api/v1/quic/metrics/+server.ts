import type { RequestHandler } from "@sveltejs/kit";
// Orphaned content: import {

export const GET: RequestHandler = async () => {
  return new Response(JSON.stringify({ ok: true, quic: getQUICMetrics() }), { status: 200, headers: { 'content-type': 'application/json' } });
};
