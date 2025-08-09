// @ts-nocheck
// Simplified Ollama API route for Legal AI Chat
// SvelteKit 2.0 + Svelte 5 + Direct Ollama integration

import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { ollamaService } from "$lib/server/services/OllamaService";
import { logger } from "$lib/server/logger";
import { dev } from "$app/environment";

export interface ChatRequest {
  message: string;
  model?: string;
  context?: string[];
  temperature?: number;
  stream?: boolean;
  caseId?: string;
  useRAG?: boolean;
}

export interface ChatResponse {
  response: string;
  model: string;
  context?: number[];
  timestamp: string;
  performance: {
    duration: number;
    tokens: number;
    promptTokens: number;
    responseTokens: number;
    tokensPerSecond: number;
  };
  suggestions?: string[];
  relatedCases?: string[];
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const {
      message,
      model = "gemma2:2b",
      temperature = 0.7,
      stream = false,
    }: ChatRequest = await request.json();

    // Validate input
    if (!message?.trim()) {
      throw error(400, { message: "Message is required" });
    }

    // Check Ollama health
    const isHealthy = await ollamaService.isHealthy();
    if (!isHealthy) {
      logger.error("Ollama service is not healthy");
      throw error(503, { message: "AI service is currently unavailable" });
    }

    // Add legal AI system prompt
    const systemPrompt = `You are a specialized legal AI assistant for prosecutors, detectives, and legal professionals.

Instructions:
- Provide accurate, well-reasoned legal analysis
- Cite relevant laws, precedents, and procedures when possible
- Consider ethical implications and due process
- Focus on factual, evidence-based responses
- Always note when additional professional legal research is recommended
- Keep responses concise but comprehensive

User question: ${message}

Please provide a helpful legal analysis:`;

    // Handle streaming vs non-streaming responses
    if (stream) {
      return handleStreamingResponse(model, systemPrompt, temperature);
    } else {
      return handleNonStreamingResponse(
        model,
        systemPrompt,
        temperature,
        startTime
      );
    }
  } catch (err) {
    logger.error("Chat API error", err);

    if (err instanceof Error && "status" in err) {
      throw err; // Re-throw SvelteKit errors
    }

    throw error(500, {
      message: dev ? err.message : "Internal server error",
    });
  }
};

async function handleStreamingResponse(
  model: string,
  prompt: string,
  temperature: number
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
      } catch (error) {
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
  startTime: number
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
  };

  return json(chatResponse);
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
export const GET: RequestHandler = async () => {
  try {
    const isHealthy = await ollamaService.isHealthy();
    const models = await ollamaService.listModels();

    return json({
      status: isHealthy ? "healthy" : "unhealthy",
      timestamp: new Date().toISOString(),
      models: models.map((m) => ({
        name: m.name,
        size: m.size,
        family: m.details?.family || 'unknown',
      })),
      endpoints: [
        "POST /api/ai/chat - Send chat message",
        "GET /api/ai/chat - Health check",
      ],
    });
  } catch (error) {
    logger.error("Health check failed", error);
    return json({ 
      status: "error", 
      error: (error as Error).message,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};
