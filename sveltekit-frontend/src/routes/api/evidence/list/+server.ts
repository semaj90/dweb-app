import type { RequestHandler } from '@sveltejs/kit';
// Endpoint to list evidence including summaryType field
import type { RequestHandler } from "@sveltejs/kit";

import { evidence } from "$lib/server/db/schema-postgres-enhanced";

export const GET: RequestHandler = async () => {
  try {
    const rows = await db.select().from(evidence).orderBy(desc(evidence.createdAt)).limit(500);
    // Map to frontend shape (camelCase + summaryType passthrough)
    const mapped = rows.map(r => ({
      id: r.id,
      filename: r.fileName || r.title,
      status: 'ready', // placeholder until real status tracking
      type: r.evidenceType || 'document',
      size: r.fileSize || 0,
      uploadDate: r.createdAt,
      summary: r.summary || r.aiSummary || null,
      prosecutionScore: (r.aiAnalysis as any)?.prosecutionScore ?? null,
      tags: Array.isArray(r.tags) ? r.tags : [],
      summaryType: (r as any).summaryType || null,
    }));
    return new Response(JSON.stringify({ evidence: mapped }), { status: 200 });
  } catch (err) {
    console.error('Evidence list error', err);
    return new Response(JSON.stringify({ error: 'Failed to list evidence' }), { status: 500 });
  }
};
