// ======================================================================
// "DID YOU MEAN" API ENDPOINT - QUIC-Optimized Suggestions
// Ultra-low latency intelligent search suggestions with graph traversal
// ======================================================================

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { didYouMeanService, type DidYouMeanQuery } from '$lib/services/did-you-mean-quic-graph.js';
import { z } from 'zod';

// Validation schema for suggestion requests
const suggestionRequestSchema = z.object({
  query: z.string().min(1, 'Query cannot be empty').max(500, 'Query too long'),
  userIntent: z.enum(['search', 'legal_research', 'case_lookup', 'document_analysis']).optional().default('search'),
  context: z.object({
    caseId: z.string().optional(),
    jurisdiction: z.string().optional(),
    practiceArea: z.string().optional(),
    documentType: z.string().optional()
  }).optional(),
  options: z.object({
    maxSuggestions: z.number().min(1).max(20).optional().default(5),
    similarityThreshold: z.number().min(0).max(1).optional().default(0.3),
    includeTypos: z.boolean().optional().default(true),
    includeSemanticSuggestions: z.boolean().optional().default(true),
    graphDepth: z.number().min(1).max(5).optional().default(3)
  }).optional()
});

// GET /api/v1/suggestions?q=contract+law&intent=legal_research&maxSuggestions=10
export const GET: RequestHandler = async ({ url, request }) => {
  const startTime = performance.now();

  try {
    // Extract query parameters
    const query = url.searchParams.get('q') || url.searchParams.get('query');
    const intent = url.searchParams.get('intent') || 'search';
    const maxSuggestions = parseInt(url.searchParams.get('maxSuggestions') || '5');
    const threshold = parseFloat(url.searchParams.get('threshold') || '0.3');
    const includeTypos = url.searchParams.get('includeTypos') !== 'false';
    const caseId = url.searchParams.get('caseId');
    const practiceArea = url.searchParams.get('practiceArea');

    if (!query) {
      throw error(400, {
        message: 'Query parameter is required',
        code: 'MISSING_QUERY'
      });
    }

    // Build suggestion query
    const suggestionQuery: DidYouMeanQuery = {
      originalQuery: query,
      userIntent: intent as any,
      context: caseId || practiceArea ? {
        caseId: caseId || undefined,
        practiceArea: practiceArea || undefined
      } : undefined,
      options: {
        maxSuggestions,
        similarityThreshold: threshold,
        includeTypos,
        includeSemanticSuggestions: true
      }
    };

    // Generate suggestions
    const result = await didYouMeanService.generateSuggestions(suggestionQuery);
    const processingTime = performance.now() - startTime;

    // Add request metadata
    const response = {
      ...result,
      metadata: {
        requestTime: new Date().toISOString(),
        processingTimeMs: processingTime,
        streamStats: didYouMeanService.getStreamStats(),
        version: '1.0'
      }
    };

    return json(response, {
      status: 200,
      headers: {
        'X-Processing-Time': processingTime.toString(),
        'X-Suggestions-Count': result.suggestions.length.toString(),
        'X-QUIC-Streams': result.cacheInfo.quicStreamsUsed.toString(),
        'Cache-Control': 'public, max-age=300', // 5 minutes cache
        'Vary': 'Accept-Encoding'
      }
    });

  } catch (err: any) {
    const processingTime = performance.now() - startTime;

    if (err && typeof err === 'object' && 'status' in err) {
      throw err; // Re-throw SvelteKit errors
    }

    console.error('Suggestion generation failed:', err);
    throw error(500, {
      message: 'Failed to generate suggestions',
      code: 'SUGGESTION_ERROR',
      processingTimeMs: processingTime
    });
  }
};

// POST /api/v1/suggestions - Advanced suggestions with full context
export const POST: RequestHandler = async ({ request }) => {
  const startTime = performance.now();

  try {
    const body = await request.json();
    
    // Validate request body
    const validatedData = suggestionRequestSchema.parse(body);

    // Build suggestion query
    const suggestionQuery: DidYouMeanQuery = {
      originalQuery: validatedData.query,
      userIntent: validatedData.userIntent,
      context: validatedData.context,
      options: validatedData.options
    };

    // Generate suggestions with full context
    const result = await didYouMeanService.generateSuggestions(suggestionQuery);
    const processingTime = performance.now() - startTime;

    // Enhanced response with detailed metrics
    const response = {
      ...result,
      metadata: {
        requestTime: new Date().toISOString(),
        requestId: crypto.randomUUID(),
        processingTimeMs: processingTime,
        streamStats: didYouMeanService.getStreamStats(),
        version: '1.0',
        optimizations: {
          quicEnabled: result.cacheInfo.quicStreamsUsed > 0,
          graphTraversalUsed: (result.cacheInfo.graphTraversalTime || 0) > 0,
          cacheHitRatio: result.cacheInfo.cacheHits / 
                        (result.cacheInfo.cacheHits + result.cacheInfo.cacheMisses)
        }
      }
    };

    return json(response, {
      status: 200,
      headers: {
        'X-Processing-Time': processingTime.toString(),
        'X-Suggestions-Count': result.suggestions.length.toString(),
        'X-QUIC-Streams': result.cacheInfo.quicStreamsUsed.toString(),
        'X-Graph-Nodes': result.graphContext?.nodesTraversed?.toString() || '0',
        'X-Cache-Hit-Ratio': response.metadata.optimizations.cacheHitRatio.toFixed(3),
        'Cache-Control': 'public, max-age=300'
      }
    });

  } catch (err: any) {
    const processingTime = performance.now() - startTime;

    if (err.name === 'ZodError') {
      throw error(400, {
        message: 'Invalid request format',
        code: 'VALIDATION_ERROR',
        errors: err.errors,
        processingTimeMs: processingTime
      });
    }

    if (err && typeof err === 'object' && 'status' in err) {
      throw err;
    }

    console.error('Advanced suggestion generation failed:', err);
    throw error(500, {
      message: 'Failed to generate suggestions',
      code: 'SUGGESTION_ERROR',
      processingTimeMs: processingTime
    });
  }
};

// DELETE /api/v1/suggestions - Clear suggestion cache
export const DELETE: RequestHandler = async ({ request }) => {
  const startTime = performance.now();

  try {
    await didYouMeanService.clearCache();
    const processingTime = performance.now() - startTime;

    return json({
      success: true,
      message: 'Suggestion cache cleared',
      processingTimeMs: processingTime,
      timestamp: new Date().toISOString()
    });

  } catch (err: any) {
    const processingTime = performance.now() - startTime;

    console.error('Cache clear failed:', err);
    throw error(500, {
      message: 'Failed to clear cache',
      code: 'CACHE_CLEAR_ERROR',
      processingTimeMs: processingTime
    });
  }
};