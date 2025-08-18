/**
 * Enhanced AI Chat API with Intent Detection
 * Integrates Gemma2B ONNX + Legal-BERT + RAG + Context7
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { semanticPipeline } from '$lib/ai/semantic-analysis-pipeline';
import { streamRag } from '$lib/ai/ragStreamClient';

interface ChatRequest {
  message: string;
  context: {
    userId: string;
    sessionId: string;
    caseId?: string;
    documentId?: string;
    userPreferences: {
      role: 'prosecutor' | 'detective' | 'paralegal' | 'attorney';
      experience: 'junior' | 'mid' | 'senior';
      jurisdiction: string;
      specialization: string[];
    };
  };
  options?: {
    enableRAG?: boolean;
    enableIntentDetection?: boolean;
    maxTokens?: number;
    temperature?: number;
  };
}

interface ChatResponse {
  response: string;
  intent: any;
  confidence: number;
  sources: string[];
  followUpQuestions: string[];
  processingTime: number;
  metadata: {
    model: string;
    tokensUsed: number;
    ragEnabled: boolean;
    intentDetected: boolean;
  };
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();
  
  try {
    const chatRequest: ChatRequest = await request.json();
    
    if (!chatRequest.message || !chatRequest.context) {
      return json({
        error: 'Invalid request: message and context required'
      }, { status: 400 });
    }

    console.log(`💬 AI Chat request from ${chatRequest.context.userId}`);

    // Initialize semantic pipeline if needed
    await semanticPipeline.initialize();

    // 1. Detect user intent
    let intent = null;
    if (chatRequest.options?.enableIntentDetection !== false) {
      intent = await semanticPipeline.detectUserIntent(
        chatRequest.message,
        chatRequest.context
      );
    }

    // 2. Retrieve relevant documents using RAG
    let relevantDocuments: any[] = [];
    if (chatRequest.options?.enableRAG !== false) {
      relevantDocuments = await retrieveRelevantDocuments(
        chatRequest.message,
        chatRequest.context
      );
    }

    // 3. Generate AI response
    const aiResponse = await semanticPipeline.generateChatResponse(
      chatRequest.message,
      intent!,
      chatRequest.context,
      relevantDocuments
    );

    // 4. Track user interaction
    await trackChatInteraction(chatRequest, intent, aiResponse);

    const processingTime = Date.now() - startTime;

    const response: ChatResponse = {
      response: aiResponse.response,
      intent,
      confidence: aiResponse.confidence,
      sources: aiResponse.sources,
      followUpQuestions: aiResponse.followUpQuestions,
      processingTime,
      metadata: {
        model: 'gemma3-legal',
        tokensUsed: estimateTokens(aiResponse.response),
        ragEnabled: chatRequest.options?.enableRAG !== false,
        intentDetected: chatRequest.options?.enableIntentDetection !== false
      }
    };

    console.log(`✅ AI Chat response generated in ${processingTime}ms`);
    return json(response);

  } catch (error) {
    console.error('❌ AI Chat request failed:', error);
    
    return json({
      error: 'AI Chat request failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      processingTime: Date.now() - startTime
    }, { status: 500 });
  }
};

// RAG document retrieval
async function retrieveRelevantDocuments(
  query: string,
  context: any
): Promise<any[]> {
  try {
    // Use existing RAG streaming client for document retrieval
    const ragResult = await new Promise<any[]>((resolve, reject) => {
      const documents: any[] = [];
      
      streamRag({
        query,
        contextIds: context.caseId ? [context.caseId] : [],
        intent: 'document_search',
        model: 'default',
        endpoint: '/api/rag/query',
        onToken: (token) => {
          // Collect tokens for document retrieval
        },
        onDone: () => {
          resolve(documents);
        },
        onError: (error) => {
          console.warn('RAG retrieval failed:', error);
          resolve([]); // Return empty array on error
        }
      });
    });

    return ragResult;

  } catch (error) {
    console.warn('Document retrieval failed:', error);
    return [];
  }
}

// Track chat interactions for analytics
async function trackChatInteraction(
  request: ChatRequest,
  intent: any,
  response: any
): Promise<void> {
  try {
    // This would integrate with your user activity tracking
    console.log(`📊 Tracking chat interaction for user ${request.context.userId}`);
    
    // Track conversation patterns for improving AI responses
    const interactionData = {
      userId: request.context.userId,
      sessionId: request.context.sessionId,
      message: request.message,
      intent: intent?.type,
      confidence: intent?.confidence,
      responseLength: response.response.length,
      sourcesCount: response.sources.length,
      timestamp: new Date().toISOString()
    };

    // Send to analytics service (implementation would depend on your analytics setup)
    // await analyticsService.trackChatInteraction(interactionData);
    
  } catch (error) {
    console.warn('Failed to track chat interaction:', error);
  }
}

// Token estimation utility
function estimateTokens(text: string): number {
  // Rough estimation: ~4 characters per token
  return Math.ceil(text.length / 4);
}

// Streaming endpoint for real-time chat
export const GET: RequestHandler = async ({ url }) => {
  const query = url.searchParams.get('query');
  const userId = url.searchParams.get('userId');
  const caseId = url.searchParams.get('caseId');

  if (!query || !userId) {
    return json({
      error: 'Query and userId parameters required'
    }, { status: 400 });
  }

  // Create Server-Sent Events stream
  const encoder = new TextEncoder();
  
  const stream = new ReadableStream({
    start(controller) {
      // Send initial connection event
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ type: 'connected', userId })}\n\n`)
      );
    },
    
    async pull(controller) {
      try {
        // Initialize context for streaming
        const context = {
          userId,
          sessionId: crypto.randomUUID(),
          caseId: caseId || undefined,
          userPreferences: {
            role: 'attorney' as const,
            experience: 'mid' as const,
            jurisdiction: 'federal',
            specialization: ['contract_law']
          }
        };

        // Detect intent
        const intent = await semanticPipeline.detectUserIntent(query, context);
        
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ 
            type: 'intent', 
            intent: intent.type,
            confidence: intent.confidence 
          })}\n\n`)
        );

        // Stream response tokens
        await streamRag({
          query,
          contextIds: caseId ? [caseId] : [],
          intent: intent.type,
          model: 'gemma3-legal',
          endpoint: '/api/rag/query/stream',
          onToken: (token) => {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: 'token', token })}\n\n`)
            );
          },
          onDone: () => {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
            );
            controller.close();
          },
          onError: (error) => {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ 
                type: 'error', 
                error: error.message 
              })}\n\n`)
            );
            controller.close();
          }
        });

      } catch (error) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ 
            type: 'error', 
            error: error instanceof Error ? error.message : 'Unknown error'
          })}\n\n`)
        );
        controller.close();
      }
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Stream-ID': crypto.randomUUID()
    }
  });
};