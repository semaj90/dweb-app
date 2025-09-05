

import { json, type RequestHandler } from '@sveltejs/kit';

// Define types locally
export interface OrchestrationRequest {
  query: string;
  documentType?: 'contract' | 'motion' | 'evidence' | 'correspondence' | 'brief';
  jurisdiction?: string;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  requiresMultiAgent: boolean;
  enableStreaming: boolean;
  context?: Record<string, any>;
}

// Service import helpers with fallbacks
async function getOrchestrator(): Promise<any> {
  try {
    // @ts-ignore - Dynamic import may not exist
    const module = await import('$lib/agents/orchestrator');
    return module.legalOrchestrator;
  } catch (error: any) {
    console.warn('Orchestrator module not available:', error);
    return {
      orchestrate: async (): Promise<any> => ({
        synthesizedConclusion: 'Service temporarily unavailable - orchestrator not loaded',
        primaryResponse: {
          agentName: 'fallback',
          response: 'Service temporarily unavailable - orchestrator not loaded',
          confidence: 0,
          tokenUsage: { prompt: 0, completion: 0, total: 0 }
        },
        confidence: 0,
        totalProcessingTime: 0,
        recommendations: []
      })
    };
  }
}

async function getCacheManager(): Promise<any> {
  try {
    // @ts-ignore - Dynamic import may not exist
    const module = await import('$lib/database/redis');
    return module.cacheManager;
  } catch (error: any) {
    console.warn('Redis module not available:', error);
    return {
      getCachedEmbeddings: async (): Promise<any> => null,
      cacheEmbeddings: async (): Promise<any> => {}
    };
  }
}

async function getQdrantManager(): Promise<any> {
  try {
    // @ts-ignore - Dynamic import may not exist
    const module = await import('$lib/database/qdrant');
    return module.qdrantManager;
  } catch (error: any) {
    console.warn('Qdrant module not available:', error);
    return {
      searchLegalDocuments: async (): Promise<any> => []
    };
  }
}

/**
 * Legal AI Chat API with Streaming Support
 * Handles legal document analysis, case research, and AI-powered legal assistance
 */

export const POST: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const {
      message,
      documentType,
      jurisdiction = 'federal',
      urgency = 'medium',
      enableStreaming = false,
      enableMultiAgent = false,
      context = {}
    } = await request.json();

    if (!message) {
      return json({ error: 'Message is required' }, { status: 400 });
    }

    // Prepare orchestration request
    const orchestrationRequest: OrchestrationRequest = {
      query: message,
      documentType,
      jurisdiction,
      urgency,
      requiresMultiAgent: enableMultiAgent,
      enableStreaming,
      context
    };

    if (enableStreaming) {
      return handleStreamingResponse(orchestrationRequest);
    } else {
      return handleStandardResponse(orchestrationRequest);
    }
  } catch (error: any) {
    console.error('AI chat API error:', error);
    return json(
      { error: 'Internal server error', details: (error as Error)?.message || 'Unknown error' },
      { status: 500 }
    );
  }
};

async function handleStandardResponse(request: OrchestrationRequest): Promise<any> {
  try {
    const orchestrator = await getOrchestrator();
    const result = await orchestrator.orchestrate(request);
    
    // Ensure we have a valid result
    if (!result || !result.primaryResponse) {
      throw new Error('Invalid orchestration result');
    }
    
    return json({
      response: result.synthesizedConclusion || 'No response generated',
      metadata: {
        primaryAgent: result.primaryResponse?.agentName || 'unknown',
        confidence: result.confidence || 0,
        processingTime: result.totalProcessingTime || 0,
        tokenUsage: result.primaryResponse?.tokenUsage || { prompt: 0, completion: 0, total: 0 },
        recommendations: result.recommendations || [],
        collaborativeAnalysis: result.collaborativeAnalysis?.map((a: any) => ({
          agent: a?.agentName || 'unknown',
          confidence: a?.confidence || 0,
          specialization: a?.metadata?.specialization || 'general'
        })) || []
      }
    });
  } catch (error: any) {
    throw new Error(`Standard response failed: ${(error as Error)?.message || 'Unknown error'}`);
  }
}

async function handleStreamingResponse(request: OrchestrationRequest): Promise<any> {
  const encoder = new TextEncoder();
  
  const readable = new ReadableStream({
    async start(controller): Promise<any> {
      try {
        // Send initial status
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'status',
            message: 'Initializing legal analysis...',
            timestamp: Date.now()
          })}\n\n`)
        );

        // Select and initialize agents
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'agent_selection',
            message: `Selected agents for ${request.documentType || 'general'} analysis`,
            urgency: request.urgency,
            timestamp: Date.now()
          })}\n\n`)
        );

        // Get orchestration result
        const orchestrator = await getOrchestrator();
        const result = await orchestrator.orchestrate(request);

        // Validate result
        if (!result || !result.primaryResponse) {
          throw new Error('Invalid orchestration result received');
        }

        // Stream primary response
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'primary_response',
            agent: result.primaryResponse?.agentName || 'unknown',
            response: result.primaryResponse?.response || 'No response generated',
            confidence: result.primaryResponse?.confidence || 0,
            timestamp: Date.now()
          })}\n\n`)
        );

        // Stream collaborative analysis if available
        if (result.collaborativeAnalysis) {
          for (const analysis of result.collaborativeAnalysis) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({
                type: 'collaborative_analysis',
                agent: (analysis as any).agentName,
                response: (analysis as any).response,
                confidence: (analysis as any).confidence,
                specialization: (analysis as any).metadata.specialization,
                timestamp: Date.now()
              })}\n\n`)
            );
          }
        }

        // Stream synthesis
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'synthesis',
            conclusion: result.synthesizedConclusion || 'No synthesis available',
            confidence: result.confidence || 0,
            timestamp: Date.now()
          })}\n\n`)
        );

        // Stream recommendations
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'recommendations',
            recommendations: result.recommendations || [],
            timestamp: Date.now()
          })}\n\n`)
        );

        // Final completion message
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'complete',
            totalProcessingTime: result.totalProcessingTime || 0,
            tokenUsage: result.primaryResponse?.tokenUsage || { prompt: 0, completion: 0, total: 0 },
            timestamp: Date.now()
          })}\n\n`)
        );

        controller.close();
      } catch (error: any) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'error',
            error: (error as any)?.message || 'Unknown error',
            timestamp: Date.now()
          })}\n\n`)
        );
        controller.close();
      }
    }
  });

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
    }
  });
}

// Vector search endpoint
export const GET: RequestHandler = async ({ url }): Promise<any> => {
  try {
    const query = url.searchParams.get('q');
    const documentType = url.searchParams.get('type');
    const jurisdiction = url.searchParams.get('jurisdiction');
    const limit = parseInt(url.searchParams.get('limit') || '10');

    if (!query) {
      return json({ error: 'Query parameter is required' }, { status: 400 });
    }

    // Generate embeddings for the query (this would use your embedding service)
    const embeddings = await generateQueryEmbeddings(query);

    // Search similar documents
    const qdrant = await getQdrantManager();
    const results = await qdrant.searchLegalDocuments(
      embeddings,
      {
        documentTypes: documentType ? [documentType] : undefined,
        jurisdictions: jurisdiction ? [jurisdiction] : undefined,
      },
      { limit }
    ).catch((error) => {
      console.warn('Vector search failed, returning empty results:', error);
      return []; // Return empty array if search fails
    });

    return json({
      query,
      results: results.map((r: any) => ({
        id: r.id,
        score: r.score,
        title: r.payload.title,
        documentType: r.payload.documentType,
        jurisdiction: r.payload.jurisdiction,
        excerpt: r.payload.content.substring(0, 300) + '...',
        metadata: r.payload.metadata
      }))
    });
  } catch (error: any) {
    console.error('Vector search error:', error);
    return json(
      { error: 'Search failed', details: (error as any)?.message || 'Unknown error' },
      { status: 500 }
    );
  }
};

// Utility function to generate embeddings (placeholder)
async function generateQueryEmbeddings(query: string): Promise<number[]> {
  try {
    // This would integrate with your embedding service (Ollama, OpenAI, etc.)
    // For now, return a mock embedding
    const cache = await getCacheManager();
    const cached = await cache.getCachedEmbeddings(query);
    
    if (cached) {
      return cached;
    }

    // Mock embedding generation - replace with actual embedding service
    const mockEmbedding = Array.from({ length: 384 }, () => Math.random() - 0.5);
    
    // Cache the embedding
    await cache.cacheEmbeddings(query, mockEmbedding);
    
    return mockEmbedding;
  } catch (error: any) {
    console.warn('Embedding generation failed, using fallback:', error);
    // Return fallback mock embedding if caching fails
    return Array.from({ length: 384 }, () => Math.random() - 0.5);
  }
}