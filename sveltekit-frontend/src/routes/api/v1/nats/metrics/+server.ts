import type { RequestHandler } from '@sveltejs/kit';
import { natsMessaging } from '$lib/services/nats-messaging-service';

// Expose in-process NATS client metrics (mock or real) as JSON
export const GET: RequestHandler = async () => {
  try {
    const metrics = (natsMessaging as any).getMetrics ? (natsMessaging as any).getMetrics() : null;
    return new Response(JSON.stringify({ ok: true, metrics }), { status: 200, headers: { 'content-type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ ok: false, error: e?.message || 'metrics_error' }), { status: 500, headers: { 'content-type': 'application/json' } });
  }
};
