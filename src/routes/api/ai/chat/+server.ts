

import { json } from '@sveltejs/kit';
import { legalOrchestrator, type OrchestrationRequest } from '$lib/agents/orchestrator.js';
import { cacheManager } from '$lib/database/redis.js';
import { qdrantManager } from '$lib/database/qdrant.js';
import type { RequestHandler } from './$types';

/**
 * Legal AI Chat API with Streaming Support
 * Handles legal document analysis, case research, and AI-powered legal assistance
 */

export const POST: RequestHandler = async ({ request }) => {
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
  } catch (error: unknown) {
    console.error('AI chat API error:', error);
    return json(
      { error: 'Internal server error', details: error?.message || 'Unknown error' },
      { status: 500 }
    );
  }
};

async function handleStandardResponse(request: OrchestrationRequest) {
  try {
    const result = await legalOrchestrator.orchestrate(request);
    
    return json({
      response: result.synthesizedConclusion,
      metadata: {
        primaryAgent: result.primaryResponse.agentName,
        confidence: result.confidence,
        processingTime: result.totalProcessingTime,
        tokenUsage: result.primaryResponse.tokenUsage,
        recommendations: result.recommendations,
        collaborativeAnalysis: result.collaborativeAnalysis?.map((a: unknown) => ({
          agent: a.agentName,
          confidence: a.confidence,
          specialization: a.metadata.specialization
        }))
      }
    });
  } catch (error: unknown) {
    throw new Error(`Standard response failed: ${error?.message || 'Unknown error'}`);
  }
}

async function handleStreamingResponse(request: OrchestrationRequest) {
  const encoder = new TextEncoder();
  
  const readable = new ReadableStream({
    async start(controller) {
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
        const result = await legalOrchestrator.orchestrate(request);

        // Stream primary response
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'primary_response',
            agent: result.primaryResponse.agentName,
            response: result.primaryResponse.response,
            confidence: result.primaryResponse.confidence,
            timestamp: Date.now()
          })}\n\n`)
        );

        // Stream collaborative analysis if available
        if (result.collaborativeAnalysis) {
          for (const analysis of result.collaborativeAnalysis) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({
                type: 'collaborative_analysis',
                agent: analysis.agentName,
                response: analysis.response,
                confidence: analysis.confidence,
                specialization: analysis.metadata.specialization,
                timestamp: Date.now()
              })}\n\n`)
            );
          }
        }

        // Stream synthesis
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'synthesis',
            conclusion: result.synthesizedConclusion,
            confidence: result.confidence,
            timestamp: Date.now()
          })}\n\n`)
        );

        // Stream recommendations
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'recommendations',
            recommendations: result.recommendations,
            timestamp: Date.now()
          })}\n\n`)
        );

        // Final completion message
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'complete',
            totalProcessingTime: result.totalProcessingTime,
            tokenUsage: result.primaryResponse.tokenUsage,
            timestamp: Date.now()
          })}\n\n`)
        );

        controller.close();
      } catch (error: unknown) {
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
export const GET: RequestHandler = async ({ url }) => {
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
    const results = await qdrantManager.searchLegalDocuments(
      embeddings,
      {
        documentTypes: documentType ? [documentType] : undefined,
        jurisdictions: jurisdiction ? [jurisdiction] : undefined,
      },
      { limit }
    );

    return json({
      query,
      results: results.map((r: unknown) => ({
        id: r.id,
        score: r.score,
        title: r.payload.title,
        documentType: r.payload.documentType,
        jurisdiction: r.payload.jurisdiction,
        excerpt: r.payload.content.substring(0, 300) + '...',
        metadata: r.payload.metadata
      }))
    });
  } catch (error: unknown) {
    console.error('Vector search error:', error);
    return json(
      { error: 'Search failed', details: (error as any)?.message || 'Unknown error' },
      { status: 500 }
    );
  }
};

// Utility function to generate embeddings (placeholder)
async function generateQueryEmbeddings(query: string): Promise<number[]> {
  // This would integrate with your embedding service (Ollama, OpenAI, etc.)
  // For now, return a mock embedding
  const cacheKey = `embeddings:${query}`;
  const cached = await cacheManager.getCachedEmbeddings(query);
  
  if (cached) {
    return cached;
  }

  // Mock embedding generation - replace with actual embedding service
  const mockEmbedding = Array.from({ length: 384 }, () => Math.random() - 0.5);
  
  // Cache the embedding
  await cacheManager.cacheEmbeddings(query, mockEmbedding);
  
  return mockEmbedding;
}