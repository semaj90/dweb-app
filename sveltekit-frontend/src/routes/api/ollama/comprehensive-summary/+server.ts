/**
 * Comprehensive Ollama Summary API Endpoint
 * 
 * Unified API that integrates all Ollama services and provides
 * comprehensive document summarization with full feature set:
 * - LangChain + Ollama integration
 * - CUDA GPU acceleration
 * - Multi-model support (Gemma3, embeddings)
 * - Streaming responses
 * - Caching and performance optimization
 * - RAG (Retrieval-Augmented Generation)
 */

import { json, type RequestHandler } from '@sveltejs/kit';
import { 
  comprehensiveOllamaSummarizer, 
  type ComprehensiveSummaryRequest,
  type ComprehensiveSummaryResponse 
} from '$lib/services/comprehensive-ollama-summarizer';

// ============================================================================
// MAIN REQUEST HANDLER
// ============================================================================

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const body = await request.json() as ComprehensiveSummaryRequest;
    
    // Validate request
    if (!body.content || !body.type) {
      return json(
        { error: 'Missing required fields: content and type' },
        { status: 400 }
      );
    }

    // Check for streaming request
    const streaming = url.searchParams.get('stream') === 'true' || body.options?.streamResponse;
    
    if (streaming) {
      // Handle streaming response
      return handleStreamingRequest(body);
    } else {
      // Handle standard request
      return handleStandardRequest(body);
    }

  } catch (error) {
    console.error('Comprehensive summary API error:', error);
    return json(
      { 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};

// ============================================================================
// STREAMING RESPONSE HANDLER
// ============================================================================

async function handleStreamingRequest(
  request: ComprehensiveSummaryRequest
): Promise<Response> {
  try {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Initialize summarizer
          await comprehensiveOllamaSummarizer.initialize();

          // Process streaming summary
          const streamGenerator = comprehensiveOllamaSummarizer.generateStreamingSummary(request);
          
          for await (const partialResponse of streamGenerator) {
            const chunk = JSON.stringify({
              type: 'partial',
              data: partialResponse
            }) + '\n';
            
            controller.enqueue(encoder.encode(chunk));
          }

          // Send final completion marker
          const finalChunk = JSON.stringify({
            type: 'complete',
            data: { status: 'finished' }
          }) + '\n';
          
          controller.enqueue(encoder.encode(finalChunk));
          controller.close();

        } catch (error) {
          const errorChunk = JSON.stringify({
            type: 'error',
            data: { 
              error: error instanceof Error ? error.message : 'Stream processing failed'
            }
          }) + '\n';
          
          controller.enqueue(encoder.encode(errorChunk));
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
      }
    });

  } catch (error) {
    return json(
      { error: 'Failed to initialize streaming', details: String(error) },
      { status: 500 }
    );
  }
}

// ============================================================================
// STANDARD RESPONSE HANDLER
// ============================================================================

async function handleStandardRequest(
  request: ComprehensiveSummaryRequest
): Promise<Response> {
  try {
    const startTime = Date.now();
    
    // Initialize summarizer if needed
    await comprehensiveOllamaSummarizer.initialize();
    
    // Generate comprehensive summary
    const result = await comprehensiveOllamaSummarizer.generateComprehensiveSummary(request);
    
    // Add API metadata
    const response = {
      ...result,
      api: {
        version: '1.0.0',
        timestamp: new Date().toISOString(),
        requestId: generateRequestId(),
        totalProcessingTime: Date.now() - startTime
      }
    };

    return json(response, {
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=300', // Cache for 5 minutes
        'Access-Control-Allow-Origin': '*'
      }
    });

  } catch (error) {
    console.error('Standard request processing failed:', error);
    return json(
      { 
        error: 'Processing failed', 
        details: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

// ============================================================================
// HEALTH CHECK ENDPOINT
// ============================================================================

export const GET: RequestHandler = async ({ url }) => {
  const action = url.searchParams.get('action');

  try {
    switch (action) {
      case 'health':
        const health = await comprehensiveOllamaSummarizer.getHealth();
        return json(health);

      case 'stats':
        const stats = await comprehensiveOllamaSummarizer.getStats();
        return json(stats);

      case 'warmup':
        await comprehensiveOllamaSummarizer.warmup();
        return json({ status: 'warmed-up', timestamp: new Date().toISOString() });

      case 'models':
        const modelStats = await comprehensiveOllamaSummarizer.getStats();
        return json({
          loaded: modelStats.models.loaded,
          available: modelStats.models.available,
          gpu: modelStats.models.gpu
        });

      default:
        // Default health check
        const basicHealth = await comprehensiveOllamaSummarizer.getHealth();
        return json({
          status: basicHealth.status,
          services: basicHealth.services.length,
          timestamp: new Date().toISOString(),
          version: '1.0.0'
        });
    }

  } catch (error) {
    return json(
      { 
        error: 'Health check failed',
        details: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};

// ============================================================================
// CONFIGURATION ENDPOINT
// ============================================================================

export const PUT: RequestHandler = async ({ request }) => {
  try {
    const config = await request.json();
    
    // Update configuration
    comprehensiveOllamaSummarizer.updateConfig(config);
    
    return json({
      status: 'updated',
      message: 'Configuration updated successfully',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    return json(
      { 
        error: 'Configuration update failed',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
};

// ============================================================================
// OPTIONS HANDLER (CORS)
// ============================================================================

export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}

// ============================================================================
// TYPE EXPORTS FOR CLIENT USAGE
// ============================================================================

export type {
  ComprehensiveSummaryRequest,
  ComprehensiveSummaryResponse
} from '$lib/services/comprehensive-ollama-summarizer';