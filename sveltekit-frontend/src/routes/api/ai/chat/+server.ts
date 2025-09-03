

import { ensureError } from '$lib/utils/ensure-error';
import type { RequestHandler } from './$types.js';
import type { ChatRequest, ChatResponse } from '$lib/types/api.js';
import { apiSuccess, apiError, getRequestId, withErrorHandling } from '$lib/server/api/standard-response';

// Simplified Ollama API route for Legal AI Chat
// SvelteKit 2.0 + Svelte 5 + Direct Ollama integration

import { json, error } from "@sveltejs/kit";
import { ollamaService } from '../../../../lib/server/services/OllamaService.js';
import { logger } from '../../../../lib/server/production-logger.js';
import { conversationService } from '$lib/server/services/conversation-service';
const dev = import.meta.env.NODE_ENV === 'development';

export const POST: RequestHandler = withErrorHandling(async (event) => {
  const requestId = getRequestId(event);
  const startTime = Date.now();

  const {
    message,
    model = "gemma3-legal:latest",
    temperature = 0.7,
    stream = false,
    conversationId,
    userId = 'mock-user-id', // TODO: Get from auth session
    caseId
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

  // Handle conversation creation/retrieval
  let currentConversationId = conversationId;
  
  if (!currentConversationId) {
    // Create new conversation
    const conversationTitle = conversationService.generateConversationTitle(message);
    const newConversation = await conversationService.createConversation({
      userId,
      title: conversationTitle,
      caseId,
      context: { model, temperature }
    });
    currentConversationId = newConversation.id;
  }

  // Save user message to conversation
  await conversationService.addMessage({
    conversationId: currentConversationId,
    role: 'user',
    content: message,
    metadata: { requestId }
  });

  // Add legal AI system prompt
  const systemPrompt = `You are a legal AI assistant. User question: ${message}`;

  // Handle streaming vs non-streaming responses
  if (stream) {
    return handleStreamingResponse(model, systemPrompt, temperature, currentConversationId, requestId);
  } else {
    return handleNonStreamingResponse(
      model,
      systemPrompt,
      temperature,
      startTime,
      requestId,
      currentConversationId
    );
  }
});

async function handleStreamingResponse(
  model: string,
  prompt: string,
  temperature: number,
  conversationId: string,
  requestId: string
): Promise<Response> {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const response = await fetch("http://localhost:11434/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            prompt,
            stream: true,
            options: { temperature },
          }),
        });

        if (!response.ok) {
          throw new Error(`Ollama API error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split("\n").filter(Boolean);

          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.response) {
                controller.enqueue(encoder.encode(data.response));
              }
              if (data.done) {
                controller.close();
                return;
              }
            } catch (parseError) {
              // Skip invalid JSON lines
            }
          }
        }
      } catch (error: any) {
        logger.error("Streaming error", error);
        controller.error(error);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain",
      "Transfer-Encoding": "chunked",
    },
  });
}

async function handleNonStreamingResponse(
  model: string,
  prompt: string,
  temperature: number,
  startTime: number,
  requestId: string,
  conversationId: string
): Promise<Response> {
  const response = await ollamaService.generate(model, prompt, { temperature });
  const endTime = Date.now();
  const duration = endTime - startTime;

  // Generate intelligent suggestions (simplified, no external API calls)
  const suggestions = generateSimpleSuggestions(prompt, response);

  // Enhanced token counting
  const promptTokens = estimateTokens(prompt);
  const responseTokens = estimateTokens(response);
  const totalTokens = promptTokens + responseTokens;

  // Save assistant message to conversation
  await conversationService.addMessage({
    conversationId,
    role: 'assistant',
    content: response,
    model,
    tokenCount: totalTokens,
    processingTime: duration,
    metadata: { 
      requestId,
      promptTokens,
      responseTokens,
      tokensPerSecond: duration > 0 ? totalTokens / (duration / 1000) : 0
    }
  });

  const chatResponse: ChatResponse = {
    response,
    model,
    timestamp: new Date().toISOString(),
    performance: {
      duration,
      tokens: totalTokens,
      promptTokens,
      responseTokens,
      tokensPerSecond: duration > 0 ? totalTokens / (duration / 1000) : 0,
    },
    suggestions,
    relatedCases: [], // Simplified - no external case lookup
    conversationId, // Include conversation ID in response
  };

  return apiSuccess(chatResponse, 'Chat message processed successfully', requestId);
}

// Enhanced token estimation function
function estimateTokens(text: string): number {
  // More accurate token estimation
  // Roughly 1 token per 4 characters for English text
  // This is a simplified approximation - production would use tiktoken or similar
  return Math.ceil(text.length / 4);
}

function generateSimpleSuggestions(
  prompt: string,
  response: string
): string[] {
  // Simple rule-based suggestions based on keywords in the prompt
  const suggestions: string[] = [];
  const lowerPrompt = prompt.toLowerCase();

  if (lowerPrompt.includes('evidence') || lowerPrompt.includes('proof')) {
    suggestions.push("What makes evidence admissible in court?");
    suggestions.push("How do I establish chain of custody?");
    suggestions.push("What are the different types of evidence?");
  } else if (lowerPrompt.includes('contract') || lowerPrompt.includes('agreement')) {
    suggestions.push("What elements make a contract valid?");
    suggestions.push("How can a contract be breached?");
    suggestions.push("What are the remedies for contract violations?");
  } else if (lowerPrompt.includes('criminal') || lowerPrompt.includes('crime')) {
    suggestions.push("What are the elements of this crime?");
    suggestions.push("What defenses might the defendant raise?");
    suggestions.push("What evidence do I need to prove intent?");
  } else {
    // Default legal suggestions
    suggestions.push("What legal precedents apply to this situation?");
    suggestions.push("What additional evidence should I gather?");
    suggestions.push("Are there any constitutional issues to consider?");
  }

  return suggestions.slice(0, 3);
}

// Health check endpoint
export const GET: RequestHandler = withErrorHandling(async (event) => {
  const requestId = getRequestId(event);

  const isHealthy = await ollamaService.isHealthy();
  const models = await ollamaService.listModels();

  const healthData = {
    status: isHealthy ? "healthy" : "unhealthy",
    models: models.map((m) => ({
      name: m.name,
      size: m.size,
      family: m.details?.family || 'unknown',
    })),
    endpoints: [
      "POST /api/ai/chat - Send chat message",
      "GET /api/ai/chat - Health check",
    ],
  };

  return apiSuccess(
    healthData, 
    isHealthy ? 'Chat service is healthy' : 'Chat service has issues', 
    requestId
  );
});
