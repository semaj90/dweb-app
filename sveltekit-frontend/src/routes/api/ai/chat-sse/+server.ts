import type { RequestHandler } from './$types.js';
import { apiError, getRequestId, withErrorHandling } from '$lib/server/api/standard-response';
import { ollamaService } from '$lib/server/services/OllamaService.js';
import { logger } from '$lib/server/production-logger.js';
import { conversationService } from '$lib/server/services/conversation-service';

/**
 * Enhanced Server-Sent Events (SSE) Chat API
 * Provides real-time streaming responses for Legal AI Assistant
 * Compatible with SvelteKit 2 + Svelte 5 + Enhanced RAG
 */

export const POST: RequestHandler = withErrorHandling(async (event) => {
  const requestId = getRequestId(event);
  
  const {
    message,
    model = "gemma3-legal:latest", 
    temperature = 0.7,
    conversationId,
    userId = 'mock-user-id',
    caseId,
    useRAG = true
  } = await event.request.json();

  // Validate input
  if (!message?.trim()) {
    return apiError("Message is required", 400, 'INVALID_INPUT', undefined, requestId);
  }

  // Check Ollama health
  const isHealthy = await ollamaService.isHealthy();
  if (!isHealthy) {
    logger.error("Ollama service is not healthy");
    return apiError("AI service is currently unavailable", 503, 'SERVICE_UNAVAILABLE', undefined, requestId);
  }

  // Handle conversation creation/continuation
  let currentConversationId = conversationId;
  if (!currentConversationId) {
    const conversationTitle = message.length > 50 ? 
      message.substring(0, 47) + '...' : message;
    
    const newConversation = await conversationService.create({
      userId,
      title: conversationTitle,
      caseId,
      context: { model, temperature, useRAG }
    });
    currentConversationId = newConversation.id;
  }

  // Save user message
  await conversationService.addMessage({
    conversationId: currentConversationId,
    role: 'user',
    content: message,
    metadata: { requestId, useRAG }
  });

  // Enhanced system prompt with RAG context
  let systemPrompt = `You are an expert legal AI assistant. Provide accurate, professional legal information.

User question: ${message}`;

  // Add RAG context if enabled
  if (useRAG) {
    try {
      const ragResponse = await fetch('http://localhost:8094/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: message,
          limit: 5,
          threshold: 0.7
        })
      });

      if (ragResponse.ok) {
        const ragData = await ragResponse.json();
        if (ragData.results && ragData.results.length > 0) {
          const contextDocs = ragData.results
            .map((r: any) => `- ${r.content || r.text || 'Relevant legal information'}`)
            .join('\n');
          
          systemPrompt += `\n\nRelevant legal context from your knowledge base:\n${contextDocs}\n\nUse this context to provide more accurate and specific answers.`;
        }
      }
    } catch (ragError) {
      logger.warn('RAG service unavailable, proceeding without context', { requestId, ragError });
    }
  }

  // Create SSE stream
  return new Response(
    new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        
        // Helper function to send SSE data
        const sendSSEData = (data: any) => {
          const sseData = `data: ${JSON.stringify(data)}\n\n`;
          controller.enqueue(encoder.encode(sseData));
        };

        try {
          // Send initial connection confirmation
          sendSSEData({
            type: 'connection',
            conversationId: currentConversationId,
            requestId,
            timestamp: new Date().toISOString()
          });

          // Call Ollama streaming API
          const response = await fetch("http://localhost:11434/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model,
              prompt: systemPrompt,
              stream: true,
              options: {
                temperature,
                num_predict: 2048,
                top_k: 40,
                top_p: 0.9,
                repeat_penalty: 1.1
              }
            })
          });

          if (!response.ok) {
            throw new Error(`Ollama API error: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error("Failed to get response reader");
          }

          let assistantMessage = '';
          let tokenCount = 0;

          // Process streaming chunks
          while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
              break;
            }

            const chunk = new TextDecoder().decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
              try {
                const data = JSON.parse(line);
                
                if (data.response) {
                  assistantMessage += data.response;
                  tokenCount++;
                  
                  // Send token update
                  sendSSEData({
                    type: 'token',
                    content: data.response,
                    fullResponse: assistantMessage,
                    tokenCount,
                    timestamp: new Date().toISOString()
                  });
                }

                if (data.done) {
                  // Save assistant message to conversation
                  await conversationService.addMessage({
                    conversationId: currentConversationId,
                    role: 'assistant',
                    content: assistantMessage,
                    metadata: {
                      requestId,
                      model,
                      temperature,
                      tokenCount,
                      useRAG
                    }
                  });

                  // Send completion signal
                  sendSSEData({
                    type: 'complete',
                    fullResponse: assistantMessage,
                    tokenCount,
                    conversationId: currentConversationId,
                    timestamp: new Date().toISOString()
                  });
                  
                  break;
                }
              } catch (parseError) {
                logger.warn('Failed to parse Ollama response chunk', { 
                  requestId, 
                  chunk: line.substring(0, 100),
                  parseError 
                });
              }
            }
          }

        } catch (error) {
          logger.error('SSE streaming error', { requestId, error });
          
          // Send error to client
          sendSSEData({
            type: 'error',
            error: error instanceof Error ? error.message : 'Stream processing failed',
            timestamp: new Date().toISOString()
          });
        } finally {
          // Close the stream
          sendSSEData({
            type: 'close',
            timestamp: new Date().toISOString()
          });
          
          controller.close();
        }
      }
    }),
    {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      }
    }
  );
});

// Handle CORS preflight requests
export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  });
};