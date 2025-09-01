import type { RequestHandler } from './$types';
import { db } from '$lib/db';
import { sql } from 'drizzle-orm';
import { buildSuccessResponse, buildErrorResponse } from '$lib/server/api/response';

const EMBEDDING_MODEL = 'nomic-embed-text';

async function getEmbedding(text: string): Promise<number[]> {
  const resp = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBEDDING_MODEL, prompt: text })
  });
  if (!resp.ok) {
    throw new Error(`Embedding service error: ${resp.status}`);
  }
  const data = await resp.json();
  if (!data?.embedding || !Array.isArray(data.embedding)) {
    throw new Error('Invalid embedding response');
  }
  return data.embedding as number[];
}

function validateQuery(q: unknown): string | null {
  if (typeof q !== 'string') return null;
  const trimmed = q.trim();
  if (!trimmed) return null;
  if (trimmed.length > 512) return trimmed.slice(0, 512); // enforce cap
  return trimmed;
}

async function performSearch(userId: string, query: string) {
  const embedding = await getEmbedding(query);
  // Using raw SQL due to pgvector operator requirements
  const result = await db.execute(sql`
    SELECT id, filename, content, summary,
           1 - (embedding <=> ${JSON.stringify(embedding)}::vector) as similarity
    FROM documents
    WHERE user_id = ${userId}
    ORDER BY embedding <=> ${JSON.stringify(embedding)}::vector
    LIMIT 10
  `);
  return { rows: result.rows, embeddingDimensions: embedding.length };
}

function respond(status: number, payload: any) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json' }
  });
}

export const GET: RequestHandler = async ({ url, locals }) => {
  const start = performance.now();
  const requestId = (locals as any).requestId || `req_${Date.now()}`;
  if (!locals.user) {
    return respond(401, buildErrorResponse('UNAUTHORIZED', 'Authentication required', { requestId, processingTimeMs: performance.now() - start }));
  }
  const qParam = url.searchParams.get('q');
  const query = validateQuery(qParam);
  if (!query) {
    return respond(400, buildErrorResponse('MISSING_QUERY', 'Query parameter q is required', { requestId, processingTimeMs: performance.now() - start }));
  }
  try {
    const searchStart = performance.now();
    const { rows, embeddingDimensions } = await performSearch(locals.user.id, query);
    return respond(200, buildSuccessResponse(rows, {
      requestId,
      processingTimeMs: performance.now() - start,
      searchLatencyMs: performance.now() - searchStart,
      model: EMBEDDING_MODEL,
      resultCount: rows.length,
      embeddingDimensions,
      engine: 'pgvector'
    }));
  } catch (e: any) {
    return respond(500, buildErrorResponse('SEARCH_ERROR', e?.message || 'Search failed', { requestId, processingTimeMs: performance.now() - start }));
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  const start = performance.now();
  const requestId = (locals as any).requestId || `req_${Date.now()}`;
  if (!locals.user) {
    return respond(401, buildErrorResponse('UNAUTHORIZED', 'Authentication required', { requestId, processingTimeMs: performance.now() - start }));
  }
  let body: any;
  try {
    body = await request.json();
  } catch {
    return respond(400, buildErrorResponse('INVALID_JSON', 'Request body must be valid JSON', { requestId, processingTimeMs: performance.now() - start }));
  }
  const query = validateQuery(body?.query ?? body?.q);
  if (!query) {
    return respond(400, buildErrorResponse('MISSING_QUERY', 'Body must include query (or q) string', { requestId, processingTimeMs: performance.now() - start }));
  }
  try {
    const searchStart = performance.now();
    const { rows, embeddingDimensions } = await performSearch(locals.user.id, query);
    return respond(200, buildSuccessResponse(rows, {
      requestId,
      processingTimeMs: performance.now() - start,
      searchLatencyMs: performance.now() - searchStart,
      model: EMBEDDING_MODEL,
      resultCount: rows.length,
      embeddingDimensions,
      engine: 'pgvector'
    }));
  } catch (e: any) {
    return respond(500, buildErrorResponse('SEARCH_ERROR', e?.message || 'Search failed', { requestId, processingTimeMs: performance.now() - start }));
  }
};
