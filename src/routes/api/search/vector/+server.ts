import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { sql } from 'drizzle-orm';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const { query, collections = ['cases', 'evidence', 'reports', 'citations'], limit = 10, filters = {} } = await request.json();
    const userId = locals.user?.id || 'system';
    
    // Generate query embedding
    const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: query
      })
    });
    
    if (!embeddingResponse.ok) {
      return json({ error: 'Failed to generate embedding' }, { status: 500 });
    }
    
    const embResult = await embeddingResponse.json();
    const queryEmbedding = embResult.embedding;
    
    // Search in PostgreSQL with pgvector
    const pgResults = await searchWithPgVector(queryEmbedding, {
      userId,
      collections,
      limit,
      filters
    });
    
    // Search in Qdrant if available
    let qdrantResults = [];
    try {
      const qdrantResponse = await fetch('http://localhost:6333/collections/legal-ai/points/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: queryEmbedding,
          limit,
          filter: {
            must: [
              { key: 'user_id', match: { value: userId } },
              ...Object.entries(filters).map(([key, value]) => ({
                key,
                match: { value }
              }))
            ]
          }
        })
      });
      
      if (qdrantResponse.ok) {
        const qdrantData = await qdrantResponse.json();
        qdrantResults = qdrantData.result || [];
      }
    } catch (err) {
      console.log('Qdrant not available, using pgvector only');
    }
    
    // Merge and rank results
    const mergedResults = [...pgResults, ...qdrantResults]
      .sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
      .slice(0, limit);
    
    return json({
      success: true,
      query,
      results: mergedResults,
      sources: {
        pgvector: pgResults.length,
        qdrant: qdrantResults.length
      }
    });
  } catch (error) {
    console.error('Error performing vector search:', error);
    return json({ error: 'Failed to perform search' }, { status: 500 });
  }
};

async function searchWithPgVector(embedding: number[], options: any) {
  const { userId, collections, limit, filters } = options;
  const results = [];
  
  for (const collection of collections) {
    let embeddingColumn = '';
    let table = '';
    
    switch (collection) {
      case 'cases':
        table = 'cases';
        embeddingColumn = 'case_summary_embedding';
        break;
      case 'evidence':
        table = 'evidence';
        embeddingColumn = 'content_embedding';
        break;
      case 'reports':
        table = 'reports';
        embeddingColumn = 'content_embedding';
        break;
      case 'citations':
        table = 'citations';
        embeddingColumn = 'embedding';
        break;
      default:
        continue;
    }
    
    const query = `
      SELECT *, 
        1 - (${embeddingColumn} <-> $1::vector) as similarity
      FROM ${table}
      WHERE user_id = $2
        AND ${embeddingColumn} IS NOT NULL
        ${filters.caseId ? `AND case_id = $3` : ''}
      ORDER BY ${embeddingColumn} <-> $1::vector
      LIMIT $${filters.caseId ? 4 : 3}
    `;
    
    const params = [
      `[${embedding.join(',')}]`,
      userId,
      ...(filters.caseId ? [filters.caseId] : []),
      limit
    ];
    
    try {
      const result = await db.execute(sql.raw(query, params));
      results.push(...result.rows.map(row => ({
        ...row,
        collection,
        type: collection
      })));
    } catch (err) {
      console.error(`Error searching ${collection}:`, err);
    }
  }
  
  return results;
}