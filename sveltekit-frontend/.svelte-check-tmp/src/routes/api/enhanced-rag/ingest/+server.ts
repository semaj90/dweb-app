import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { apiClient } from '$lib/services/api-client';

/**
 * Orchestrating ingestion proxy.
 * Accepts: { ids: string[], mode?: 'full' | 'incremental' }
 * Forwards to backend Enhanced RAG pipeline (Go service) via internal API client.
 */
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const { ids, mode = 'incremental' } = await request.json();
    if (!Array.isArray(ids) || ids.length === 0) {
      return json({ success: false, error: 'ids[] required' }, { status: 400 });
    }
    // Optionally enforce auth (placeholder: if locals.user missing -> 401)
    if (!locals?.user) {
      // Soft auth: allow but tag anonymous; adjust if stricter needed
    }
    const resp = await apiClient.triggerEnhancedRagIngestion(ids, mode);
    return json({ success: true, queued: resp.data?.queued || ids.length, mode });
  } catch (e: any) {
    return json({ success: false, error: e.message || 'Ingestion failed' }, { status: 500 });
  }
};
