import type { RequestHandler } from './$types';

/**
 * RAG Search API - Semantic search across processed documents
 */
import { db, documents, embeddings, searchSessions } from "$lib/server/database";

import { desc, eq, sql } from 'drizzle-orm';

// Generate embedding for search query
async function generateQueryEmbedding(query: string): Promise<any> {
  try {
    const embeddingResponse = await fetch('/api/ai/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: query, save: false })
    });

    if (embeddingResponse.ok) {
      const result = await embeddingResponse.json();
      return result.embedding;
    }
  } catch (error: any) {
    console.error('Failed to generate query embedding:', error);
  }
  return null;
}

// Perform vector similarity search
async function vectorSearch(queryEmbedding: number[], limit: number, threshold: number): Promise<any> {
  try {
    // Use PostgreSQL vector similarity search
    const similarResults = await db
      .select({
        id: embeddings.id,
        documentId: embeddings.documentId,
        content: embeddings.content,
        metadata: embeddings.metadata,
        filename: documents.filename,
        fullContent: documents.content,
        similarity: sql<number>`1 - (${embeddings.embedding} <=> ${queryEmbedding}::vector)`.as('similarity')
      })
      .from(embeddings)
      .innerJoin(documents, eq(embeddings.documentId, documents.id))
      .where(sql`1 - (${embeddings.embedding} <=> ${queryEmbedding}::vector) > ${threshold}`)
      .orderBy(desc(sql`1 - (${embeddings.embedding} <=> ${queryEmbedding}::vector)`))
      .limit(limit);

    return similarResults;
  } catch (error: any) {
    console.error('Vector search failed:', error);
    return [];
  }
}

// Perform text-based search
async function textSearch(query: string, limit: number): Promise<any> {
  try {
    const textResults = await db
      .select({
        id: documents.id,
        filename: documents.filename,
        content: documents.content,
        metadata: documents.metadata,
        createdAt: documents.createdAt
      })
      .from(documents)
      .where(sql`${documents.content} ILIKE ${`%${query}%`}`)
      .orderBy(desc(documents.createdAt))
      .limit(limit);

    return textResults.map(result => ({
      ...result,
      similarity: 0.8, // Fixed similarity for text search
      searchType: 'text'
    }));
  } catch (error: any) {
    console.error('Text search failed:', error);
    return [];
  }
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, searchType = 'hybrid', limit = 10, threshold = 0.7 } = await request.json();

    if (!query) {
      return json({ error: 'Query is required' }, { status: 400 });
    }

    console.log('RAG Search Request:', { query, searchType, limit, threshold });

    let results: any[] = [];
    const startTime = Date.now();

    if (searchType === 'semantic' || searchType === 'hybrid') {
      // Generate query embedding
      const queryEmbedding = await generateQueryEmbedding(query);

      if (queryEmbedding) {
        // Perform vector similarity search
        const vectorResults = await vectorSearch(queryEmbedding, limit, threshold);
        results = results.concat(vectorResults.map(result => ({
          id: result.documentId,
          filename: result.filename,
          content: result.content,
          fullContent: result.fullContent,
          similarity: result.similarity,
          metadata: result.metadata,
          searchType: 'semantic'
        })));

        // Save search session
        try {
          await db.insert(searchSessions).values({
            query,
            queryEmbedding,
            results: results,
            searchType,
            resultCount: results.length
          });
        } catch (sessionError) {
          console.error('Failed to save search session:', sessionError);
        }
      }
    }

    if (searchType === 'text' || searchType === 'hybrid') {
      // Perform text search
      const textResults = await textSearch(query, limit);
      results = results.concat(textResults);
    }

    // Remove duplicates and sort by similarity
    const uniqueResults = results
      .filter((result, index, self) =>
        index === self.findIndex(r => r.id === result.id)
      )
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);

    const processingTime = `${Date.now() - startTime}ms`;

    return json({
      success: true,
      query,
      results: uniqueResults,
      totalResults: uniqueResults.length,
      searchType,
      processingTime,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('RAG search error:', error);
    return json(
      {
        error: 'Search failed',
        details: error instanceof Error ? error.message : 'Unknown error',
        query: 'unknown'
      },
      { status: 500 }
    );
  }
};
