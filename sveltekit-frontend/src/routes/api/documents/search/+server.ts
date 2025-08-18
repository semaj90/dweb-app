// Real Document Search API with PostgreSQL, pgvector, and hybrid search
import {
    db,
    documents,
    embeddings,
    initializeDatabase,
    searchSessions,
} from '$lib/server/database';
import { error, json } from '@sveltejs/kit';
import { sql } from 'drizzle-orm';
import type { RequestHandler } from './$types';

// Redis client for caching search results
import { createRedisConnection } from '$lib/utils/redis-helper';
// ... other imports ...

const redis = createRedisConnection();

// Ensure database is initialized
let dbInitialized = false;

export const POST: RequestHandler = async ({ request }) => {
  try {
    console.log('[Search] Processing real search request...');

    // Initialize database if not already done
    if (!dbInitialized) {
      dbInitialized = await initializeDatabase();
      if (!dbInitialized) {
        console.warn('[Search] Database initialization failed, proceeding anyway');
      }
    }

    const body = await request.json();
    const {
      query,
      embedding,
      limit = 10,
      threshold = 0.7,
      searchType = 'hybrid',
      filters = {},
    } = body;

    if (!query && !embedding) {
      throw error(400, 'Query or embedding is required');
    }

    console.log(`[Search] Performing ${searchType} search for: "${query}"`);

    // Check cache for search results
    const cacheKey = `search:${searchType}:${Buffer.from(JSON.stringify({ query, filters, limit, threshold })).toString('base64').substring(0, 50)}`;
    try {
      const cached = await redis.get(cacheKey);
      if (cached) {
        console.log('[Search] Cache hit');
        const cachedResult = JSON.parse(cached);
        return json({ ...cachedResult, cached: true });
      }
    } catch (err) {
      console.warn('[Search] Redis cache unavailable:', err);
    }

    let results: any[] = [];
    let searchMethod = '';

    // Generate embedding for the query if not provided
    let queryEmbedding = embedding;
    if (query && !queryEmbedding) {
      try {
        console.log('[Search] Generating query embedding...');
        const embResponse = await fetch('/api/embeddings/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: query,
            model: 'nomic-embed-text',
          }),
        });

        if (embResponse.ok) {
          const embResult = await embResponse.json();
          queryEmbedding = embResult.embedding;
        }
      } catch (embError) {
        console.warn('[Search] Failed to generate query embedding:', embError);
      }
    }

    // Perform search based on type
    switch (searchType) {
      case 'vector':
        if (queryEmbedding) {
          results = await vectorSearch(queryEmbedding, limit, threshold, filters);
          searchMethod = 'Vector Similarity';
        } else {
          throw error(400, 'Embedding required for vector search');
        }
        break;

      case 'keyword':
        results = await keywordSearch(query, limit, filters);
        searchMethod = 'Full-text Search';
        break;

      case 'hybrid':
        results = await hybridSearch(query, queryEmbedding, limit, threshold, filters);
        searchMethod = 'Hybrid (Vector + Keyword)';
        break;

      case 'semantic':
        if (queryEmbedding) {
          results = await semanticSearch(query, queryEmbedding, limit, threshold, filters);
          searchMethod = 'Semantic + Context';
        } else {
          results = await keywordSearch(query, limit, filters);
          searchMethod = 'Keyword (fallback)';
        }
        break;

      default:
        throw error(400, 'Invalid search type');
    }

    // Log search session
    try {
      await db.insert(searchSessions).values({
        query: query || 'embedding-only',
        queryEmbedding: queryEmbedding,
        searchType,
        resultCount: results.length,
      });
    } catch (logError) {
      console.warn('[Search] Failed to log search session:', logError);
    }

    const finalResult = {
      success: true,
      results,
      count: results.length,
      searchType,
      searchMethod,
      query,
      cached: false,
      timestamp: new Date().toISOString(),
    };

    // Cache search results for 5 minutes
    try {
      await redis.setex(cacheKey, 300, JSON.stringify(finalResult));
      console.log('[Search] Results cached successfully');
    } catch (err) {
      console.warn('[Search] Failed to cache results:', err);
    }

    console.log(`[Search] Found ${results.length} results using ${searchMethod}`);
    return json(finalResult);
  } catch (err: any) {
    console.error('[Search] Error:', err);

    return json(
      {
        success: false,
        error: err.message || 'Search failed',
        details: err.stack,
      },
      { status: err.status || 500 }
    );
  }
};

// Vector similarity search with pgvector
async function vectorSearch(
  embedding: number[],
  limit: number,
  threshold: number,
  filters: any
): Promise<any[]> {
  try {
    console.log('[Search] Performing vector similarity search');

    // Build dynamic query with filters
    let whereClause = sql`1 - (e.embedding <=> ${JSON.stringify(embedding)}::vector) > ${threshold}`;

    if (filters.documentType) {
      whereClause = sql`${whereClause} AND d.metadata->>'documentType' = ${filters.documentType}`;
    }
    if (filters.jurisdiction) {
      whereClause = sql`${whereClause} AND d.metadata->>'jurisdiction' = ${filters.jurisdiction}`;
    }
    if (filters.dateFrom) {
      whereClause = sql`${whereClause} AND d.created_at >= ${filters.dateFrom}`;
    }
    if (filters.dateTo) {
      whereClause = sql`${whereClause} AND d.created_at <= ${filters.dateTo}`;
    }

    const query = sql`
      SELECT
        d.id,
        d.filename,
        d.content,
        d.metadata,
        d.created_at,
        d.legal_analysis,
        1 - (e.embedding <=> ${JSON.stringify(embedding)}::vector) AS similarity,
        'vector' as search_type
      FROM documents d
      JOIN legal_embeddings e ON d.id = e.document_id
      WHERE ${whereClause}
      ORDER BY similarity DESC
      LIMIT ${limit}
    `;

    const results = await db.execute(query);

    const rowArray: any[] = (results as any)?.rows || (Array.isArray(results) ? results : []);
    return rowArray.map((row) => ({
      id: row.id,
      filename: row.filename,
      title: row.filename,
      content: row.content,
      excerpt: row.content.substring(0, 200) + '...',
      metadata: row.metadata,
      similarity: parseFloat(row.similarity),
      createdAt: row.created_at,
      legalAnalysis: row.legal_analysis,
      searchType: 'vector',
    }));
  } catch (err) {
    console.error('[Search] Vector search error:', err);
    return [];
  }
}

// Full-text keyword search
async function keywordSearch(query: string, limit: number, filters: any): Promise<any[]> {
  try {
    console.log('[Search] Performing full-text search');

    // Build dynamic query with filters
    let whereClause = sql`to_tsvector('english', d.content) @@ plainto_tsquery('english', ${query})`;

    if (filters.documentType) {
      whereClause = sql`${whereClause} AND d.metadata->>'documentType' = ${filters.documentType}`;
    }
    if (filters.jurisdiction) {
      whereClause = sql`${whereClause} AND d.metadata->>'jurisdiction' = ${filters.jurisdiction}`;
    }

    const searchQuery = sql`
      SELECT
        d.id,
        d.filename,
        d.content,
        d.metadata,
        d.created_at,
        d.legal_analysis,
        ts_rank(
          to_tsvector('english', d.content),
          plainto_tsquery('english', ${query})
        ) AS rank,
        'keyword' as search_type
      FROM documents d
      WHERE ${whereClause}
      ORDER BY rank DESC
      LIMIT ${limit}
    `;

    const results = await db.execute(searchQuery);

    const rowArray: any[] = (results as any)?.rows || (Array.isArray(results) ? results : []);
    return rowArray.map((row) => ({
      id: row.id,
      filename: row.filename,
      title: row.filename,
      content: row.content,
      excerpt: extractExcerpt(row.content, query),
      metadata: row.metadata,
      similarity: parseFloat(row.rank),
      createdAt: row.created_at,
      legalAnalysis: row.legal_analysis,
      searchType: 'keyword',
    }));
  } catch (err) {
    console.error('[Search] Keyword search error:', err);
    return [];
  }
}

// Hybrid search combining vector and keyword
async function hybridSearch(
  query: string,
  embedding: number[] | null,
  limit: number,
  threshold: number,
  filters: any
): Promise<any[]> {
  console.log('[Search] Performing hybrid search');

  // Perform both searches in parallel
  const [vectorResults, keywordResults] = await Promise.all([
    embedding ? vectorSearch(embedding, limit * 2, threshold, filters) : Promise.resolve([]),
    keywordSearch(query, limit * 2, filters),
  ]);

  // Combine and deduplicate results
  const combinedResults = new Map();

  // Add vector results with higher weight
  vectorResults.forEach((result) => {
    combinedResults.set(result.id, {
      ...result,
      score: result.similarity * 0.7, // Vector weight
      sources: ['vector'],
    });
  });

  // Add/update with keyword results
  keywordResults.forEach((result) => {
    if (combinedResults.has(result.id)) {
      const existing = combinedResults.get(result.id);
      existing.score += result.similarity * 0.3; // Keyword weight
      existing.sources.push('keyword');
    } else {
      combinedResults.set(result.id, {
        ...result,
        score: result.similarity * 0.3,
        sources: ['keyword'],
      });
    }
  });

  // Sort by combined score and limit
  return Array.from(combinedResults.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((result) => ({
      ...result,
      similarity: result.score,
      searchType: 'hybrid',
      matchedBy: result.sources,
    }));
}

// Enhanced semantic search with context
async function semanticSearch(
  query: string,
  embedding: number[],
  limit: number,
  threshold: number,
  filters: any
): Promise<any[]> {
  console.log('[Search] Performing semantic search with context');

  // Get vector results first
  const vectorResults = await vectorSearch(embedding, limit * 3, threshold * 0.8, filters);

  // Enhance with semantic context analysis
  return vectorResults
    .map((result) => {
      const contextScore = calculateContextScore(query, result.content);
      const legalRelevance = calculateLegalRelevance(query, result.legalAnalysis);

      return {
        ...result,
        contextScore,
        legalRelevance,
        enhancedSimilarity: result.similarity * 0.6 + contextScore * 0.2 + legalRelevance * 0.2,
        searchType: 'semantic',
      };
    })
    .sort((a, b) => b.enhancedSimilarity - a.enhancedSimilarity)
    .slice(0, limit);
}

// Extract relevant excerpt from content based on query
function extractExcerpt(content: string, query: string): string {
  const words = query.toLowerCase().split(' ');
  const sentences = content.split(/[.!?]+/);

  // Find sentence containing query terms
  for (const sentence of sentences) {
    const lowerSentence = sentence.toLowerCase();
    if (words.some((word) => lowerSentence.includes(word))) {
      return sentence.trim().substring(0, 200) + '...';
    }
  }

  // Fallback to first 200 characters
  return content.substring(0, 200) + '...';
}

// Calculate context relevance score
function calculateContextScore(query: string, content: string): number {
  const queryWords = query.toLowerCase().split(' ');
  const contentWords = content.toLowerCase().split(' ');

  let matches = 0;
  for (const queryWord of queryWords) {
    if (contentWords.includes(queryWord)) {
      matches++;
    }
  }

  return matches / queryWords.length;
}

// Calculate legal relevance score
function calculateLegalRelevance(query: string, legalAnalysis: any): number {
  if (!legalAnalysis) return 0;

  const queryLower = query.toLowerCase();
  let relevanceScore = 0;

  // Check legal entities
  if (legalAnalysis.entities) {
    for (const entity of legalAnalysis.entities) {
      if (queryLower.includes(entity.text.toLowerCase())) {
        relevanceScore += entity.confidence || 0.5;
      }
    }
  }

  // Check legal concepts
  if (legalAnalysis.concepts) {
    for (const concept of legalAnalysis.concepts) {
      if (queryLower.includes(concept.toLowerCase())) {
        relevanceScore += 0.3;
      }
    }
  }

  return Math.min(relevanceScore, 1.0);
}

// Store document endpoint (renamed to avoid duplicate POST export)
// Store document function (internal use)
async function handleStoreDocument(request: Request, url: URL) {
  if (url.pathname.endsWith('/store')) {
    try {
      console.log('[Search] Storing document...');

      const body = await request.json();
      const { content, embedding, metadata = {}, filename, originalContent, legalAnalysis } = body;

      if (!content) {
        throw error(400, 'Content is required');
      }

      // Store document
      const documentResult = await db
        .insert(documents)
        .values({
          filename: filename || 'untitled',
          content,
          originalContent,
          metadata: JSON.stringify(metadata),
          legalAnalysis: legalAnalysis ? JSON.stringify(legalAnalysis) : null,
          confidence: metadata.confidence || null,
        })
        .returning();

      const documentId = documentResult[0].id;

      // Store embedding if provided
      if (embedding && embedding.length > 0) {
        await db.insert(embeddings).values({
          documentId,
          content,
          embedding: embedding,
          metadata: {
            ...metadata,
            stored_at: new Date().toISOString(),
          },
        });
      }

      console.log(`[Search] Document stored with ID: ${documentId}`);

      return json({
        success: true,
        documentId,
        message: 'Document stored successfully',
      });
    } catch (err: any) {
      console.error('[Search] Storage error:', err);

      return json(
        {
          success: false,
          error: err.message || 'Storage failed',
        },
        { status: err.status || 500 }
      );
    }
  }

  // Default to search
  return json({ error: 'Invalid operation' }, { status: 400 });
}

// Health check endpoint
export const GET: RequestHandler = async () => {
  try {
    // Test database connection
    let dbStatus = false;
    let dbInfo = '';
    try {
      const result = await sql`SELECT version()`;
      dbStatus = true;
      dbInfo = result[0]?.version || 'Connected';
    } catch (err) {
      dbInfo = err.message;
    }

    // Test Redis connection
    let redisStatus = false;
    try {
      const pong = await redis.ping();
      redisStatus = pong === 'PONG';
    } catch (err) {
      console.warn('[Search] Redis health check failed:', err);
    }

    // Count documents and embeddings
    let documentCount = 0;
    let embeddingCount = 0;
    try {
      const docResult = await db.execute(sql`SELECT COUNT(*) as count FROM documents`);
      const docRows: any[] = (docResult as any)?.rows || docResult || [];
      documentCount = parseInt(docRows[0]?.count || '0');

      const embResult = await db.execute(sql`SELECT COUNT(*) as count FROM legal_embeddings`);
      const embRows: any[] = (embResult as any)?.rows || embResult || [];
      embeddingCount = parseInt(embRows[0]?.count || '0');
    } catch (err) {
      console.warn('[Search] Failed to count documents:', err);
    }

    return json({
      status: dbStatus ? 'healthy' : 'unhealthy',
      service: 'Real Document Search & Storage',
      features: {
        vectorSearch: dbStatus,
        keywordSearch: dbStatus,
        hybridSearch: dbStatus,
        semanticSearch: dbStatus,
        caching: redisStatus,
        documentStorage: dbStatus,
      },
      database: {
        connected: dbStatus,
        info: dbInfo,
        documents: documentCount,
        embeddings: embeddingCount,
      },
      cache: {
        connected: redisStatus,
      },
      timestamp: new Date().toISOString(),
      version: '2.0.0',
    });
  } catch (err: any) {
    return json(
      {
        status: 'unhealthy',
        error: err.message,
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
};
