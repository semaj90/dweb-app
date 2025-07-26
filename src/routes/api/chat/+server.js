import { json } from "@sveltejs/kit";
import { OLLAMA_URL } from "$env/static/private";
import { db } from "$lib/server/db/index.js";
import { messages, chatSessions } from "$lib/server/db/schema.js";
import { ragService } from "$lib/server/rag.js";
import { embeddingService } from "$lib/server/embedding.js";

/**
 * Enhanced Chat API with full RAG (Retrieval-Augmented Generation) support
 * Handles streaming responses, context retrieval, and conversation management
 */

const OLLAMA_API_URL = OLLAMA_URL || "http://localhost:11434";
const DEFAULT_MODEL = "gemma2-legal";

export async function POST({ request }) {
  try {
    const { message, sessionId, caseId, options = {} } = await request.json();

    if (!message?.trim()) {
      return json({ error: "Message is required" }, { status: 400 });
    }

    // Validate or create session
    const session = await validateOrCreateSession(sessionId, caseId);

    // Store user message
    const userMessage = await storeMessage(session.id, "user", message);

    // Check if this should be a streaming response
    const shouldStream = options.stream !== false;

    if (shouldStream) {
      return handleStreamingResponse(message, session, userMessage, options);
    } else {
      return handleRegularResponse(message, session, userMessage, options);
    }
  } catch (error) {
    console.error("Chat API error:", error);
    return json(
      {
        error: "Internal server error",
        details: error.message,
      },
      { status: 500 }
    );
  }
}

/**
 * Handle streaming chat response with RAG
 */
async function handleStreamingResponse(message, session, userMessage, options) {
  const readable = new ReadableStream({
    async start(controller) {
      try {
        let fullResponse = "";
        const startTime = Date.now();

        // Send initial status
        controller.enqueue(
          new TextEncoder().encode(
            "data: " +
              JSON.stringify({
                type: "status",
                message: "Retrieving relevant context...",
              }) +
              "\n\n"
          )
        );

        // Retrieve context using RAG
        const contextResult = await ragService.retrieveContext(message, {
          caseId: session.caseId,
          maxChunks: options.maxContextChunks || 5,
          includeKnowledgeBase: options.includeKnowledgeBase !== false,
          includeDocuments: options.includeDocuments !== false,
        });

        // Send context info
        controller.enqueue(
          new TextEncoder().encode(
            "data: " +
              JSON.stringify({
                type: "context",
                sources: contextResult.sources,
                totalSources: contextResult.totalSources,
              }) +
              "\n\n"
          )
        );

        // Enhance prompt with context
        const enhancedPrompt = ragService.enhancePrompt(
          message,
          contextResult.context,
          {
            caseId: session.caseId,
          }
        );

        // Send generation status
        controller.enqueue(
          new TextEncoder().encode(
            "data: " +
              JSON.stringify({
                type: "status",
                message: "Generating response...",
              }) +
              "\n\n"
          )
        );

        // Call Ollama API for streaming generation
        const ollamaResponse = await fetch(`${OLLAMA_API_URL}/api/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: options.model || DEFAULT_MODEL,
            prompt: enhancedPrompt,
            stream: true,
            options: {
              temperature: options.temperature || 0.2,
              top_k: options.top_k || 30,
              top_p: options.top_p || 0.8,
              repeat_penalty: options.repeat_penalty || 1.15,
              num_ctx: options.num_ctx || 8192,
              num_predict: options.num_predict || 1024,
            },
          }),
        });

        if (!ollamaResponse.ok) {
          throw new Error(`Ollama API error: ${ollamaResponse.status}`);
        }

        // Process streaming response
        const reader = ollamaResponse.body?.getReader();
        if (!reader) {
          throw new Error("No response body from Ollama");
        }

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split("\n").filter((line) => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);

              if (data.response) {
                fullResponse += data.response;

                // Send token to client
                controller.enqueue(
                  new TextEncoder().encode(
                    "data: " +
                      JSON.stringify({
                        type: "token",
                        content: data.response,
                      }) +
                      "\n\n"
                  )
                );
              }

              if (data.done) {
                // Store assistant message
                const responseTime = Date.now() - startTime;
                await storeMessage(session.id, "assistant", fullResponse, {
                  contextSources: contextResult.sources,
                  responseTime,
                  tokensUsed: data.eval_count || 0,
                  model: options.model || DEFAULT_MODEL,
                });

                // Send completion
                controller.enqueue(
                  new TextEncoder().encode(
                    "data: " +
                      JSON.stringify({
                        type: "complete",
                        messageId: userMessage.id,
                        responseTime,
                        tokensUsed: data.eval_count || 0,
                        sources: contextResult.sources,
                      }) +
                      "\n\n"
                  )
                );

                controller.close();
                return;
              }
            } catch (parseError) {
              console.error("Error parsing Ollama response:", parseError);
            }
          }
        }
      } catch (error) {
        console.error("Streaming error:", error);

        controller.enqueue(
          new TextEncoder().encode(
            "data: " +
              JSON.stringify({
                type: "error",
                error: error.message,
              }) +
              "\n\n"
          )
        );

        controller.close();
      }
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

/**
 * Handle regular (non-streaming) chat response with RAG
 */
async function handleRegularResponse(message, session, userMessage, options) {
  try {
    const startTime = Date.now();

    // Retrieve context using RAG
    const contextResult = await ragService.retrieveContext(message, {
      caseId: session.caseId,
      maxChunks: options.maxContextChunks || 5,
      includeKnowledgeBase: options.includeKnowledgeBase !== false,
      includeDocuments: options.includeDocuments !== false,
    });

    // Enhance prompt with context
    const enhancedPrompt = ragService.enhancePrompt(
      message,
      contextResult.context,
      {
        caseId: session.caseId,
      }
    );

    // Call Ollama API for generation
    const ollamaResponse = await fetch(`${OLLAMA_API_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: options.model || DEFAULT_MODEL,
        prompt: enhancedPrompt,
        stream: false,
        options: {
          temperature: options.temperature || 0.2,
          top_k: options.top_k || 30,
          top_p: options.top_p || 0.8,
          repeat_penalty: options.repeat_penalty || 1.15,
          num_ctx: options.num_ctx || 8192,
          num_predict: options.num_predict || 1024,
        },
      }),
    });

    if (!ollamaResponse.ok) {
      throw new Error(`Ollama API error: ${ollamaResponse.status}`);
    }

    const data = await ollamaResponse.json();
    const responseTime = Date.now() - startTime;

    // Store assistant message
    await storeMessage(session.id, "assistant", data.response, {
      contextSources: contextResult.sources,
      responseTime,
      tokensUsed: data.eval_count || 0,
      model: options.model || DEFAULT_MODEL,
    });

    return json({
      response: data.response,
      messageId: userMessage.id,
      sessionId: session.id,
      sources: contextResult.sources,
      responseTime,
      tokensUsed: data.eval_count || 0,
      model: options.model || DEFAULT_MODEL,
    });
  } catch (error) {
    console.error("Chat generation error:", error);
    throw error;
  }
}

/**
 * Validate existing session or create new one
 */
async function validateOrCreateSession(sessionId, caseId) {
  try {
    if (sessionId) {
      const [existingSession] = await db
        .select()
        .from(chatSessions)
        .where(eq(chatSessions.id, sessionId))
        .limit(1);

      if (existingSession) {
        return existingSession;
      }
    }

    // Create new session
    const [newSession] = await db
      .insert(chatSessions)
      .values({
        title: "New Legal Consultation",
        userId: "00000000-0000-0000-0000-000000000000", // Default user - replace with auth
        caseId: caseId || null,
        metadata: {
          createdVia: "api",
          timestamp: new Date().toISOString(),
        },
      })
      .returning();

    return newSession;
  } catch (error) {
    console.error("Error managing session:", error);
    throw error;
  }
}

/**
 * Store message in database
 */
async function storeMessage(sessionId, role, content, metadata = {}) {
  try {
    const [message] = await db
      .insert(messages)
      .values({
        sessionId,
        role,
        content,
        metadata: {
          ...metadata,
          timestamp: new Date().toISOString(),
        },
        tokensUsed: metadata.tokensUsed || 0,
        responseTime: metadata.responseTime || 0,
      })
      .returning();

    return message;
  } catch (error) {
    console.error("Error storing message:", error);
    throw error;
  }
}

/**
 * Get available models from Ollama
 */
export async function GET({ url }) {
  try {
    const action = url.searchParams.get("action");

    if (action === "models") {
      const response = await fetch(`${OLLAMA_API_URL}/api/tags`);

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status}`);
      }

      const data = await response.json();
      return json({ models: data.models || [] });
    }

    if (action === "health") {
      // Check system health
      const ragHealth = await ragService.healthCheck();
      const embeddingHealth = await embeddingService.healthCheck();

      return json({
        status: "healthy",
        rag: ragHealth,
        embedding: embeddingHealth,
        ollama: {
          url: OLLAMA_API_URL,
          defaultModel: DEFAULT_MODEL,
        },
        timestamp: new Date().toISOString(),
      });
    }

    return json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Chat API GET error:", error);
    return json(
      {
        error: "Service unavailable",
        details: error.message,
      },
      { status: 503 }
    );
  }
}
