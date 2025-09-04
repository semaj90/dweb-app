import type { RequestHandler } from '@sveltejs/kit';
import { healthCheck } from '$lib/db/drizzle';
import { healthSnapshot } from '$lib/services/nats-messaging-service';
export const GET: RequestHandler = async () => {
  const dbOk = await healthCheck().catch(()=>false);
  const nats = healthSnapshot();
  const ok = dbOk && !nats.circuitOpen;
  return new Response(JSON.stringify({ status: ok? 'ok':'degraded', dbOk, nats }), { status: ok?200:503, headers: { 'content-type':'application/json' } });
};
