/**
 * Semantic Search API Endpoint
 * GPU-accelerated semantic search using nomic-embed-text
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { gpuEmbeddingService } from '$lib/services/gpu-semantic-embedding-service';
import type { SemanticSearchRequest } from '$lib/services/gpu-semantic-embedding-service';

/**
 * POST /api/v1/embeddings/search
 * Perform semantic search with GPU acceleration
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const searchRequest: SemanticSearchRequest = await request.json();
    
    // Validate required fields
    if (!searchRequest.query) {
      return json(
        { error: 'Missing required field: query' },
        { status: 400 }
      );
    }
    
    if (!searchRequest.documents || !Array.isArray(searchRequest.documents)) {
      return json(
        { error: 'Missing or invalid field: documents (must be array)' },
        { status: 400 }
      );
    }

    if (searchRequest.documents.length === 0) {
      return json({
        success: true,
        query: searchRequest.query,
        results: [],
        metadata: {
          documentCount: 0,
          resultsFound: 0,
          processingTime: 0,
          threshold: searchRequest.threshold || 0.3,
          topK: searchRequest.topK || 10
        }
      });
    }

    // Perform semantic search
    const results = await gpuEmbeddingService.semanticSearch(searchRequest);
    
    return json({
      success: true,
      query: searchRequest.query,
      results: results.map(r => ({
        document: r.document,
        score: Math.round(r.score * 10000) / 10000, // Round to 4 decimal places
        index: r.index
        // Note: not including embeddings in response to reduce payload size
      })),
      metadata: {
        documentCount: searchRequest.documents.length,
        resultsFound: results.length,
        threshold: searchRequest.threshold || 0.3,
        topK: searchRequest.topK || 10,
        gpuUsed: searchRequest.useGPU !== false
      },
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('Semantic search API error:', error);
    return json(
      { 
        error: 'Failed to perform semantic search',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};

/**
 * GET /api/v1/embeddings/search
 * Get semantic search endpoint information
 */
export const GET: RequestHandler = async () => {
  return json({
    endpoint: 'POST /api/v1/embeddings/search',
    description: 'GPU-accelerated semantic search using nomic-embed-text embeddings',
    parameters: {
      query: { type: 'string', required: true, description: 'Search query text' },
      documents: { type: 'string[]', required: true, description: 'Array of documents to search' },
      threshold: { type: 'number', required: false, default: 0.3, description: 'Minimum similarity threshold' },
      topK: { type: 'number', required: false, default: 10, description: 'Maximum number of results' },
      useGPU: { type: 'boolean', required: false, default: true, description: 'Enable GPU acceleration' }
    },
    response: {
      success: 'boolean',
      query: 'string',
      results: [{
        document: 'string',
        score: 'number',
        index: 'number'
      }],
      metadata: {
        documentCount: 'number',
        resultsFound: 'number',
        threshold: 'number',
        topK: 'number',
        gpuUsed: 'boolean'
      }
    },
    examples: {
      request: {
        query: 'legal contract terms',
        documents: [
          'This contract shall be governed by applicable law',
          'The parties agree to binding arbitration',
          'Payment terms are net 30 days'
        ],
        threshold: 0.4,
        topK: 5,
        useGPU: true
      }
    },
    timestamp: Date.now()
  });
};