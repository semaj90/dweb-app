import type { RequestHandler } from "@sveltejs/kit";
// Orphaned content: import {

export const GET: RequestHandler = async () => {
  const body = renderNlpMetrics();
  return new Response(body, { status: 200, headers: { 'content-type': 'text/plain; version=0.0.4' } });
};
