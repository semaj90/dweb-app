import { json } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { evidence } from '$lib/server/db/schema-postgres-enhanced';
import { sql, ilike, and, or } from 'drizzle-orm';
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, useSemanticSearch, includeContext7, maxResults = 10 } = await request.json();

    if (!query?.trim()) {
      return json({ matches: [] }, { status: 200 });
    }

    // Basic text search in evidence
    const searchResults = await db
      .select({
        id: evidence.id,
        filename: evidence.fileName,
        summary: evidence.summary,
        content: evidence.aiSummary,
        tags: evidence.tags,
        prosecutionScore: evidence.aiAnalysis,
        similarity: sql<number>`1.0`, // Placeholder similarity
      })
      .from(evidence)
      .where(
        or(
          ilike(evidence.fileName, `%${query}%`),
          ilike(evidence.summary, `%${query}%`),
          ilike(evidence.aiSummary, `%${query}%`)
        )
      )
      .limit(maxResults);

    const matches = searchResults.map(row => ({
      id: row.id,
      filename: row.filename,
      content: row.summary || row.content,
      similarity: 0.8, // Mock similarity score
      tags: Array.isArray(row.tags) ? row.tags : [],
      prosecutionScore: typeof row.prosecutionScore === 'object' 
        ? (row.prosecutionScore as any)?.prosecutionScore || 0 
        : 0,
    }));

    return json({ matches, query, useSemanticSearch, includeContext7 }, { status: 200 });
  } catch (error: any) {
    console.error('Evidence search error:', error);
    return json({ 
      error: 'Search failed', 
      matches: [] 
    }, { status: 500 });
  }
};