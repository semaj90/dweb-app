/**
 * Gemma3-Loki Integration API - LLVM-Quality Legal AI with Advanced Caching
 * High-performance endpoint for legal document analysis with intelligent storage
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { gemma3LokiIntegration } from '$lib/services/gemma3-loki-integration';

/**
 * POST /api/ai/gemma3-loki - Comprehensive legal document analysis
 */
export const POST: RequestHandler = async ({ request, url }): Promise<any> => {
  const searchParams = url.searchParams;
  const action = searchParams.get('action') || 'analyze';

  try {
    const body = await request.json();

    switch (action) {
      case 'analyze':
        return await handleLegalAnalysis(body);

      case 'search':
        return await handleLegalSearch(body);

      case 'batch-analyze':
        return await handleBatchAnalysis(body);

      case 'performance':
        return await handlePerformanceStats();

      default:
        throw error(400, `Unknown action: ${action}`);
    }

  } catch (err: any) {
    console.error('[Gemma3-Loki API] Request failed:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Internal server error',
      code: 'GEMMA3_LOKI_REQUEST_FAILED'
    });
  }
};

/**
 * Handle single document analysis
 */
async function handleLegalAnalysis(body: {
  content: string;
  title: string;
  caseId?: string;
  analysisType?: 'comprehensive' | 'quick' | 'risk-focused' | 'legal-precedent';
  useCache?: boolean;
  storeResults?: boolean;
  userId?: string;
}): Promise<Response> {
  const { 
    content, 
    title, 
    caseId, 
    analysisType = 'comprehensive',
    useCache = true,
    storeResults = false,
    userId 
  } = body;

  if (!content?.trim()) {
    throw error(400, 'Content is required');
  }

  if (!title?.trim()) {
    throw error(400, 'Title is required');
  }

  const startTime = performance.now();

  try {
    const result = await gemma3LokiIntegration.analyzeLegalDocument({
      content,
      title,
      caseId,
      analysisType,
      useCache,
      storeResults,
      userId
    });

    const totalTime = performance.now() - startTime;

    return json({
      success: true,
      ...result,
      performance: {
        totalTime: Math.round(totalTime),
        method: result.analysis.method,
        cached: result.caching.cached,
        timestamp: Date.now()
      }
    });

  } catch (err: any) {
    console.error('[Gemma3-Loki] Analysis failed:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Analysis failed',
      code: 'LEGAL_ANALYSIS_FAILED'
    });
  }
}

/**
 * Handle legal content search
 */
async function handleLegalSearch(body: {
  query: string;
  caseId?: string;
  analysisType?: string;
  includeEmbeddings?: boolean;
  maxResults?: number;
}): Promise<Response> {
  const { 
    query, 
    caseId, 
    analysisType, 
    includeEmbeddings = true, 
    maxResults = 20 
  } = body;

  if (!query?.trim()) {
    throw error(400, 'Query is required');
  }

  const startTime = performance.now();

  try {
    const results = await gemma3LokiIntegration.searchLegalContent(query, {
      caseId,
      analysisType,
      includeEmbeddings,
      maxResults
    });

    const totalTime = performance.now() - startTime;

    return json({
      success: true,
      query,
      results,
      metadata: {
        totalResults: results.length,
        searchTime: Math.round(totalTime),
        includeEmbeddings,
        maxResults,
        timestamp: Date.now()
      }
    });

  } catch (err: any) {
    console.error('[Gemma3-Loki] Search failed:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Search failed',
      code: 'LEGAL_SEARCH_FAILED'
    });
  }
}

/**
 * Handle batch document analysis
 */
async function handleBatchAnalysis(body: {
  documents: Array<{
    content: string;
    title: string;
    caseId?: string;
    metadata?: any;
  }>;
  analysisType?: 'comprehensive' | 'quick' | 'risk-focused' | 'legal-precedent';
  useCache?: boolean;
  storeResults?: boolean;
  userId?: string;
  maxConcurrency?: number;
}): Promise<Response> {
  const { 
    documents, 
    analysisType = 'comprehensive',
    useCache = true,
    storeResults = false,
    userId,
    maxConcurrency = 5 
  } = body;

  if (!documents || !Array.isArray(documents) || documents.length === 0) {
    throw error(400, 'Documents array is required and must not be empty');
  }

  if (documents.length > 50) {
    throw error(400, 'Maximum 50 documents per batch');
  }

  const startTime = performance.now();
  const results = [];
  const errors = [];

  try {
    // Process documents in chunks to avoid overwhelming the system
    const chunks = [];
    for (let i = 0; i < documents.length; i += maxConcurrency) {
      chunks.push(documents.slice(i, i + maxConcurrency));
    }

    for (const chunk of chunks) {
      const chunkPromises = chunk.map(async (doc, index): Promise<any> => {
        try {
          const result = await gemma3LokiIntegration.analyzeLegalDocument({
            content: doc.content,
            title: doc.title,
            caseId: doc.caseId,
            analysisType,
            useCache,
            storeResults,
            userId
          });

          return { index: results.length + index, success: true, ...result };
        } catch (error: any) {
          const errorResult = {
            index: results.length + index,
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
            document: { title: doc.title, caseId: doc.caseId }
          };
          errors.push(errorResult);
          return errorResult;
        }
      });

      const chunkResults = await Promise.all(chunkPromises);
      results.push(...chunkResults);
    }

    const totalTime = performance.now() - startTime;
    const successfulResults = results.filter(r => r.success);
    const failedResults = results.filter(r => !r.success);

    return json({
      success: true,
      batchAnalysis: {
        total: documents.length,
        successful: successfulResults.length,
        failed: failedResults.length,
        results: successfulResults,
        errors: failedResults,
        performance: {
          totalTime: Math.round(totalTime),
          averageTimePerDocument: Math.round(totalTime / documents.length),
          concurrency: maxConcurrency,
          timestamp: Date.now()
        }
      }
    });

  } catch (err: any) {
    console.error('[Gemma3-Loki] Batch analysis failed:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Batch analysis failed',
      code: 'BATCH_ANALYSIS_FAILED'
    });
  }
}

/**
 * Handle performance statistics request
 */
async function handlePerformanceStats(): Promise<Response> {
  try {
    const stats = gemma3LokiIntegration.getPerformanceStats();

    return json({
      success: true,
      performance: stats,
      timestamp: Date.now()
    });

  } catch (err: any) {
    console.error('[Gemma3-Loki] Performance stats failed:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Failed to get performance stats',
      code: 'PERFORMANCE_STATS_FAILED'
    });
  }
}

/**
 * GET /api/ai/gemma3-loki - Service health and information
 */
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  const searchParams = url.searchParams;
  const info = searchParams.get('info');

  try {
    if (info === 'health') {
      const stats = gemma3LokiIntegration.getPerformanceStats();
      
      return json({
        healthy: true,
        service: 'Gemma3-Loki Integration',
        version: '1.0.0',
        components: {
          webassemblyEngine: {
            status: 'active',
            successRate: stats.webassemblySuccessRate
          },
          lokiCache: {
            status: 'active',
            entries: stats.cacheStats.totalEntries,
            hitRate: stats.cacheStats.hitRate
          },
          evidenceDatabase: {
            status: 'active',
            totalEvidence: stats.evidenceStats.total
          },
          searchIndex: {
            status: 'active',
            indexedItems: stats.searchStats.totalItems,
            embeddings: stats.searchStats.itemsWithEmbeddings
          }
        },
        performance: {
          totalRequests: stats.totalRequests,
          cacheHitRate: stats.cacheHitRate,
          averageProcessingTime: Math.round(stats.averageProcessingTime)
        },
        timestamp: Date.now()
      });
    }

    if (info === 'stats') {
      const stats = gemma3LokiIntegration.getPerformanceStats();
      return json({
        performance: stats,
        timestamp: Date.now()
      });
    }

    return json({
      service: 'Gemma3-Loki Legal AI Integration',
      version: '1.0.0',
      description: 'LLVM-Quality WebAssembly Gemma3 inference with intelligent Loki caching',
      endpoints: [
        'POST ?action=analyze - Analyze legal documents',
        'POST ?action=search - Search legal content',  
        'POST ?action=batch-analyze - Batch document analysis',
        'POST ?action=performance - Performance statistics',
        'GET ?info=health - Service health check',
        'GET ?info=stats - Detailed performance statistics'
      ],
      features: [
        'LLVM-quality WebAssembly inference',
        'Intelligent Loki database caching',
        'Evidence management integration',
        'Semantic search with embeddings',
        'Batch processing capabilities',
        'Performance monitoring',
        'Legal-specific analysis patterns'
      ],
      timestamp: Date.now()
    });

  } catch (err: any) {
    console.error('[Gemma3-Loki API] GET request failed:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Service unavailable',
      code: 'SERVICE_UNAVAILABLE'
    });
  }
};