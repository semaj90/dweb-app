import type { RequestHandler } from '@sveltejs/kit';
import { renderMetrics } from '$lib/services/nats-messaging-service';
export const GET: RequestHandler = async () => new Response(renderMetrics(), { status: 200, headers: { 'content-type': 'text/plain; version=0.0.4' } });
