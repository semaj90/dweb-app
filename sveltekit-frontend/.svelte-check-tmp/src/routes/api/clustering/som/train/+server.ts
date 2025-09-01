import type { RequestHandler } from './$types';

/**
 * SvelteKit 2 API Route: SOM Training
 * POST /api/clustering/som/train
 */

import { json } from "@sveltejs/kit";
import { LegalDocumentSOM } from "$lib/services/som-clustering";
import { Redis } from "ioredis";
import { legalDocuments } from "$lib/server/db/schema-postgres";
import { URL } from "url";
// Optional amqp for message queue integration

// Initialize Redis connection (safe minimal config for dev)
const redis = new Redis({
  port: parseInt(import.meta.env.REDIS_PORT || "6379"),
  host: import.meta.env.REDIS_HOST || '127.0.0.1'
});

export const POST: RequestHandler = async ({ request }) => {
  const body = await request.json().catch(() => ({}));
  // placeholder: enqueue training job
  await redis.rpush('som:train:queue', JSON.stringify(body));
  return json({ ok: true });
};