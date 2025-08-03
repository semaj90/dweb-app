// Enhanced Ollama API route for Legal AI Chat
// SvelteKit 2.0 + Svelte 5 + Drizzle ORM integration

import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { ollamaService } from "$lib/server/services/OllamaService";
import { redisVectorService } from "$lib/services/redis-vector-service";
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

export const POST: RequestHandler = async ({ request, url, locals }) => {
  const startTime = Date.now();

  try {
    const {
      message,
      model = "gemma3-legal",
      context = [],
      temperature = 0.7,
      stream = false,
      caseId,
      useRAG = true,
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

    // Enhanced prompt with legal context
    let enhancedPrompt = message;

    if (useRAG && message.length > 10) {
      try {
        // Get embeddings for semantic search
        const embedding = await ollamaService.embeddings(
          "nomic-embed-text",
          message
        );

        // Search for related legal documents
        const relatedDocs = await redisVectorService.searchVectors(embedding, {
          limit: 3,
          threshold: 0.7,
          collection: "legal-documents",
        });

        if (relatedDocs.length > 0) {
          const contextDocs = relatedDocs
            .map((doc) => `[${doc.payload.title}]: ${doc.payload.summary}`)
            .join("\n");

          enhancedPrompt = `Context from legal documents:
${contextDocs}

User question: ${message}

Please provide a comprehensive legal analysis based on the context above.`;
        }
      } catch (ragError) {
        logger.warn("RAG enhancement failed, using original prompt", ragError);
      }
    }

    // Add legal AI system prompt
    const systemPrompt = `You are a specialized legal AI assistant for prosecutors.
- Provide accurate, well-reasoned legal analysis
- Cite relevant laws, precedents, and procedures
- Consider ethical implications and due process
- Focus on factual, evidence-based responses
- Always note when additional research is recommended

${enhancedPrompt}`;

    // Handle streaming vs non-streaming responses
    if (stream) {
      return handleStreamingResponse(model, systemPrompt, temperature);
    } else {
      return handleNonStreamingResponse(
        model,
        systemPrompt,
        temperature,
        startTime,
        caseId
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
  startTime: number,
  caseId?: string
): Promise<Response> {
  const response = await ollamaService.generate(model, prompt, { temperature });
  const endTime = Date.now();
  const duration = endTime - startTime;

  // Generate intelligent suggestions
  const suggestions = await generateSuggestions(prompt, response);

  // Find related cases if caseId is provided
  let relatedCases: string[] = [];
  if (caseId) {
    try {
      relatedCases = await findRelatedCases(caseId, prompt);
    } catch (error) {
      logger.warn("Failed to find related cases", error);
    }
  }

  // Cache the interaction for future RAG enhancement
  try {
    await redisVectorService.cacheEmbedding(
      `${prompt} ${response}`,
      await ollamaService.embeddings(
        "nomic-embed-text",
        `${prompt} ${response}`
      ),
      "chat-interactions"
    );
  } catch (cacheError) {
    logger.warn("Failed to cache interaction", cacheError);
  }

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
      tokensPerSecond: totalTokens / (duration / 1000),
    },
    suggestions,
    relatedCases,
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

async function generateSuggestions(
  prompt: string,
  response: string
): Promise<string[]> {
  try {
    const suggestionPrompt = `Based on this legal query and response, suggest 3 related questions a prosecutor might ask:

Query: ${prompt}
Response: ${response.substring(0, 200)}...

Provide only the questions, one per line:`;

    const suggestions = await ollamaService.generate(
      "gemma3:2b",
      suggestionPrompt,
      {
        temperature: 0.8,
      }
    );

    return suggestions
      .split("\n")
      .filter((line) => line.trim() && line.includes("?"))
      .slice(0, 3);
  } catch (error) {
    logger.warn("Failed to generate suggestions", error);
    return [];
  }
}

async function findRelatedCases(
  caseId: string,
  prompt: string
): Promise<string[]> {
  try {
    // Get embedding for the prompt
    const embedding = await ollamaService.embeddings(
      "nomic-embed-text",
      prompt
    );

    // Search for similar cases
    const similarCases = await redisVectorService.searchVectors(embedding, {
      limit: 5,
      threshold: 0.6,
      collection: "cases",
    });

    return similarCases
      .filter((c) => c.id !== caseId)
      .map((c) => c.payload.title || c.id)
      .slice(0, 3);
  } catch (error) {
    logger.warn("Failed to find related cases", error);
    return [];
  }
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
        family: m.details.family,
      })),
      endpoints: [
        "POST /api/ai/chat - Send chat message",
        "GET /api/ai/chat - Health check",
      ],
    });
  } catch (error) {
    logger.error("Health check failed", error);
    return json({ status: "error", error: error.message }, { status: 500 });
  }
};
