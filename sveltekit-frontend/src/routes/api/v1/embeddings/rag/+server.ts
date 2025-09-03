/**
 * Enhanced RAG API Endpoint
 * GPU-accelerated RAG queries with semantic embeddings
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { gpuEmbeddingService } from '$lib/services/gpu-semantic-embedding-service';

interface RAGRequest {
  query: string;
  documents: string[];
  options?: {
    useGPU?: boolean;
    model?: string;
    contextLimit?: number;
    temperature?: number;
    threshold?: number;
  };
}

/**
 * POST /api/v1/embeddings/rag
 * Enhanced RAG query with GPU-accelerated embeddings
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    const ragRequest: RAGRequest = await request.json();
    
    // Validate required fields
    if (!ragRequest.query) {
      return json(
        { error: 'Missing required field: query' },
        { status: 400 }
      );
    }
    
    if (!ragRequest.documents || !Array.isArray(ragRequest.documents)) {
      return json(
        { error: 'Missing or invalid field: documents (must be array)' },
        { status: 400 }
      );
    }

    if (ragRequest.documents.length === 0) {
      return json({
        success: true,
        query: ragRequest.query,
        context: {
          similarDocs: [],
          embeddings: [],
          processingTime: 0,
          metadata: {
            model: 'nomic-embed-text:latest',
            gpuUsed: false,
            vectorDimensions: 384
          }
        },
        message: 'No documents provided for context'
      });
    }

    // Perform enhanced RAG query
    const context = await gpuEmbeddingService.enhancedRAGQuery(
      ragRequest.query,
      ragRequest.documents,
      ragRequest.options || {}
    );
    
    return json({
      success: true,
      query: ragRequest.query,
      context: {
        similarDocs: context.similarDocs.map(doc => ({
          document: doc.document.length > 500 
            ? doc.document.substring(0, 500) + '...' 
            : doc.document,
          score: Math.round(doc.score * 10000) / 10000,
          index: doc.index
        })),
        processingTime: context.processingTime,
        metadata: context.metadata
      },
      options: {
        model: ragRequest.options?.model || 'gemma3-legal:latest',
        contextLimit: ragRequest.options?.contextLimit || 5,
        temperature: ragRequest.options?.temperature || 0.7,
        threshold: ragRequest.options?.threshold || 0.4,
        useGPU: ragRequest.options?.useGPU !== false
      },
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('Enhanced RAG API error:', error);
    return json(
      { 
        error: 'Failed to process RAG query',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};

/**
 * GET /api/v1/embeddings/rag
 * Get enhanced RAG endpoint information
 */
export const GET: RequestHandler = async () => {
  return json({
    endpoint: 'POST /api/v1/embeddings/rag',
    description: 'Enhanced RAG queries with GPU-accelerated semantic embeddings and RAG service integration',
    features: [
      'GPU-accelerated embeddings with nomic-embed-text',
      'Semantic similarity search for context selection',
      'Integration with enhanced RAG service (port 8094)',
      'Configurable context limits and similarity thresholds',
      'Performance telemetry and monitoring'
    ],
    parameters: {
      query: { 
        type: 'string', 
        required: true, 
        description: 'The user query or question' 
      },
      documents: { 
        type: 'string[]', 
        required: true, 
        description: 'Array of documents to use as context' 
      },
      options: {
        type: 'object',
        required: false,
        properties: {
          useGPU: { type: 'boolean', default: true, description: 'Enable GPU acceleration' },
          model: { type: 'string', default: 'gemma3-legal:latest', description: 'RAG model to use' },
          contextLimit: { type: 'number', default: 5, description: 'Maximum documents in context' },
          temperature: { type: 'number', default: 0.7, description: 'Response creativity (0.0-1.0)' },
          threshold: { type: 'number', default: 0.4, description: 'Minimum similarity threshold' }
        }
      }
    },
    response: {
      success: 'boolean',
      query: 'string',
      context: {
        similarDocs: [{
          document: 'string',
          score: 'number',
          index: 'number'
        }],
        processingTime: 'number',
        metadata: {
          model: 'string',
          gpuUsed: 'boolean',
          vectorDimensions: 'number'
        }
      },
      options: 'object'
    },
    integration: {
      embeddingModel: 'nomic-embed-text:latest',
      ragService: 'http://localhost:8094/api/rag',
      gpuSupport: true,
      telemetryEnabled: true
    },
    examples: {
      request: {
        query: 'What are the key terms in this contract?',
        documents: [
          'This agreement shall be binding upon all parties...',
          'Payment terms are net 30 days from invoice date...',
          'Either party may terminate with 30 days notice...'
        ],
        options: {
          useGPU: true,
          model: 'gemma3-legal:latest',
          contextLimit: 3,
          temperature: 0.7,
          threshold: 0.4
        }
      }
    },
    timestamp: Date.now()
  });
};