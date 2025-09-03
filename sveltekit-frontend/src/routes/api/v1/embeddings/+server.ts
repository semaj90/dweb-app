/**
 * GPU-Accelerated Embeddings API Endpoint
 * Provides access to nomic-embed-text GPU embeddings and semantic search
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { gpuEmbeddingService } from '$lib/services/gpu-semantic-embedding-service';
import type { 
  EmbeddingRequest, 
  SemanticSearchRequest 
} from '$lib/services/gpu-semantic-embedding-service';

/**
 * POST /api/v1/embeddings
 * Generate embeddings for text or array of texts
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const embeddingRequest: EmbeddingRequest = await request.json();
    
    if (!embeddingRequest.text) {
      return json(
        { error: 'Missing required field: text' },
        { status: 400 }
      );
    }

    const result = await gpuEmbeddingService.generateEmbeddings(embeddingRequest);
    
    return json({
      success: true,
      ...result,
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('Embeddings API error:', error);
    return json(
      { 
        error: 'Failed to generate embeddings',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};

/**
 * GET /api/v1/embeddings
 * Get service status and configuration
 */
export const GET: RequestHandler = async () => {
  try {
    const status = await gpuEmbeddingService.getStatus();
    
    return json({
      service: 'gpu-semantic-embedding',
      status,
      endpoints: {
        generate: 'POST /api/v1/embeddings',
        search: 'POST /api/v1/embeddings/search',
        rag: 'POST /api/v1/embeddings/rag',
        status: 'GET /api/v1/embeddings'
      },
      models: {
        default: 'nomic-embed-text:latest',
        dimensions: 384,
        supportsBatch: true
      },
      features: {
        gpuAcceleration: status.gpuAvailable,
        semanticSearch: true,
        ragIntegration: true,
        telemetryEnabled: true
      },
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('Embeddings status error:', error);
    return json(
      { 
        error: 'Failed to get service status',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};