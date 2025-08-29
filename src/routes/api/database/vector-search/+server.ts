/**
 * pgvector Semantic Search API
 * High-performance vector similarity search using PostgreSQL pgvector
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Database connection helper
async function executeQuery(query: string, params: any[] = []): Promise<any[]> {
  try {
    const response = await fetch('http://localhost:8094/api/database/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        params,
        timeout: 30000
      })
    });

    if (!response.ok) {
      throw new Error(`Database query failed: ${response.status}`);
    }

    const result = await response.json();
    return result.rows || [];

  } catch (err: any) {
    console.error('Vector search query error:', err);
    throw err;
  }
}

/**
 * POST /api/database/vector-search - Perform semantic similarity search
 */
export const POST: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const body = await request.json();
    const {
      embedding,
      threshold = 0.7,
      maxResults = 20,
      userId,
      caseId,
      includeAnalysis = true,
      searchType = 'cosine' // cosine, euclidean, dot_product
    } = body;

    if (!embedding || !Array.isArray(embedding)) {
      throw error(400, 'Valid embedding vector is required');
    }

    console.log(`🔍 Performing pgvector search with ${embedding.length} dimensions`);

    const startTime = performance.now();
    
    // Convert embedding to pgvector format
    const vectorString = `[${embedding.join(',')}]`;

    // Build the similarity search query
    let similarityOp = '<=>';
    let orderDirection = 'ASC';

    switch (searchType) {
      case 'euclidean':
        similarityOp = '<->';
        break;
      case 'dot_product':
        similarityOp = '<#>';
        orderDirection = 'DESC';
        break;
      case 'cosine':
      default:
        similarityOp = '<=>';
        break;
    }

    // Main search query with document details
    let searchQuery = `
      WITH vector_search AS (
        SELECT 
          e.document_id,
          e.embedding ${similarityOp} $1::vector as similarity_score,
          e.model,
          e.dimensions
        FROM document_embeddings e
        WHERE e.embedding ${similarityOp} $1::vector ${searchType === 'dot_product' ? '>' : '<'} $2
        ORDER BY e.embedding ${similarityOp} $1::vector ${orderDirection}
        LIMIT $3
      )
      SELECT 
        vs.document_id,
        vs.similarity_score,
        vs.model,
        vs.dimensions,
        d.title,
        d.content,
        d.file_name,
        d.mime_type,
        d.file_size,
        d.user_id,
        d.case_id,
        d.processing_method,
        d.processing_time_ms,
        d.created_at,
        ${includeAnalysis ? 'd.ai_analysis,' : 'NULL as ai_analysis,'}
        ${includeAnalysis ? 'd.image_analysis' : 'NULL as image_analysis'}
      FROM vector_search vs
      JOIN legal_documents d ON vs.document_id = d.id
    `;

    const params = [vectorString, threshold, maxResults];

    // Add user filter if provided
    if (userId) {
      searchQuery += ` WHERE d.user_id = $${params.length + 1}`;
      params.push(userId);
    }

    // Add case filter if provided
    if (caseId) {
      const whereClause = userId ? ' AND' : ' WHERE';
      searchQuery += `${whereClause} d.case_id = $${params.length + 1}`;
      params.push(caseId);
    }

    searchQuery += ` ORDER BY vs.similarity_score ${orderDirection}`;

    const results = await executeQuery(searchQuery, params);

    const searchTime = performance.now() - startTime;

    // Process results
    const processedResults = results.map(row => ({
      documentId: row.document_id,
      similarity: searchType === 'dot_product' 
        ? row.similarity_score 
        : 1 - row.similarity_score, // Convert distance to similarity
      title: row.title,
      content: row.content ? row.content.substring(0, 500) + '...' : null,
      fileName: row.file_name,
      mimeType: row.mime_type,
      fileSize: row.file_size,
      userId: row.user_id,
      caseId: row.case_id,
      processingMethod: row.processing_method,
      processingTime: row.processing_time_ms,
      createdAt: row.created_at,
      analysis: row.ai_analysis ? JSON.parse(row.ai_analysis) : null,
      imageAnalysis: row.image_analysis ? JSON.parse(row.image_analysis) : null,
      embedding: {
        model: row.model,
        dimensions: row.dimensions
      }
    }));

    console.log(`✅ Vector search completed in ${searchTime.toFixed(2)}ms, found ${processedResults.length} results`);

    return json({
      success: true,
      results: processedResults,
      metadata: {
        searchType,
        threshold,
        maxResults,
        actualResults: processedResults.length,
        searchTime: Math.round(searchTime),
        vectorDimensions: embedding.length,
        filters: {
          userId: userId || null,
          caseId: caseId || null
        }
      },
      timestamp: Date.now()
    });

  } catch (err: any) {
    console.error('[Vector Search] Error:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Vector search failed',
      code: 'VECTOR_SEARCH_FAILED'
    });
  }
};

/**
 * GET /api/database/vector-search - Get search statistics and health
 */
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  try {
    const searchParams = url.searchParams;
    const info = searchParams.get('info');

    if (info === 'health') {
      // Check pgvector extension availability
      const healthQuery = `
        SELECT 
          extname,
          extversion
        FROM pg_extension 
        WHERE extname = 'vector';
      `;

      const healthResult = await executeQuery(healthQuery);
      const pgvectorAvailable = healthResult.length > 0;

      // Get embedding statistics
      const statsQuery = `
        SELECT 
          COUNT(*) as total_embeddings,
          COUNT(DISTINCT model) as unique_models,
          AVG(dimensions) as avg_dimensions,
          MIN(created_at) as oldest_embedding,
          MAX(created_at) as newest_embedding
        FROM document_embeddings;
      `;

      const stats = await executeQuery(statsQuery);

      return json({
        healthy: pgvectorAvailable,
        pgvector: {
          available: pgvectorAvailable,
          version: pgvectorAvailable ? healthResult[0].extversion : null
        },
        embeddings: stats[0] || {},
        searchCapabilities: {
          cosine: true,
          euclidean: true,
          dotProduct: true,
          maxDimensions: 16000 // pgvector limit
        },
        timestamp: Date.now()
      });
    }

    if (info === 'stats') {
      // Get detailed search statistics
      const recentSearches = `
        SELECT 
          DATE_TRUNC('hour', created_at) as hour,
          COUNT(*) as embedding_count
        FROM document_embeddings 
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY hour DESC;
      `;

      const modelStats = `
        SELECT 
          model,
          COUNT(*) as count,
          AVG(dimensions) as avg_dimensions
        FROM document_embeddings
        GROUP BY model
        ORDER BY count DESC;
      `;

      const [hourlyStats, modelBreakdown] = await Promise.all([
        executeQuery(recentSearches),
        executeQuery(modelStats)
      ]);

      return json({
        success: true,
        statistics: {
          hourlyEmbeddings: hourlyStats,
          modelBreakdown: modelBreakdown,
          capabilities: {
            maxVectorDimensions: 16000,
            supportedDistances: ['cosine', 'euclidean', 'dot_product'],
            indexTypes: ['ivfflat', 'hnsw']
          }
        },
        timestamp: Date.now()
      });
    }

    return json({
      service: 'pgvector Semantic Search',
      version: '1.0.0',
      description: 'High-performance vector similarity search using PostgreSQL pgvector extension',
      endpoints: [
        'POST / - Perform similarity search with embedding vector',
        'GET ?info=health - Check pgvector health and availability',
        'GET ?info=stats - Get detailed search statistics'
      ],
      features: [
        'Cosine similarity search',
        'Euclidean distance search', 
        'Dot product search',
        'User and case filtering',
        'Configurable similarity thresholds',
        'Analysis and metadata inclusion'
      ],
      timestamp: Date.now()
    });

  } catch (err: any) {
    console.error('[Vector Search] GET error:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Failed to get search info',
      code: 'VECTOR_SEARCH_INFO_FAILED'
    });
  }
};