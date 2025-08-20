import type { RequestHandler } from '@sveltejs/kit';

// Simple in-memory placeholder; replace with persistent status store.
const evidenceStatus = new Map<string, any>();

export const GET: RequestHandler = async ({ params }) => {
  const id = params.id;
  const status = evidenceStatus.get(id) || { id, status: 'unknown' };
  return new Response(JSON.stringify(status), { status: 200, headers: { 'Content-Type': 'application/json' } });
};

export function _setEvidenceStatus(id: string, status: any) { evidenceStatus.set(id, status); }
