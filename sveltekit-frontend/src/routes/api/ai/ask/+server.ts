// SvelteKit API endpoint for AI RAG system with Gemma3 Local LLM Support
// Handles "Ask AI" requests with vector search and hybrid LLM integration
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { ollamaService } from "$lib/services/ollama-service";

// Fallback imports with error handling
let vectorSearch: any = null;
let aiService: any = null;
let cache: any = null;
let tauriLLM: any = null;

try {
  const vectorSearchModule = await import(
    "../../../../lib/server/search/vector-search.js"
  );
  vectorSearch = vectorSearchModule.vectorSearch;
} catch (error) {
  console.warn("Vector search not available:", error);
  vectorSearch = { search: async () => ({ results: [] }) };
}
try {
  const aiServiceModule = await import(
    "../../../../lib/services/ai-service.js"
  );
  aiService = aiServiceModule.aiService || aiServiceModule.default;
} catch (error) {
  console.warn("AI service not available:", error);
  aiService = { generateResponse: async () => "AI service not available" };
}
try {
  const cacheModule = await import("../../../../lib/server/cache/redis.js");
  cache = cacheModule.cache || { get: async () => null, set: async () => {} };
} catch (error) {
  console.warn("Cache not available:", error);
  cache = { get: async () => null, set: async () => {} };
}
try {
  const tauriModule = await import("../../../../lib/services/tauri-llm.js");
  tauriLLM = tauriModule.tauriLLM || tauriModule.default;
} catch (error) {
  console.warn("Tauri LLM not available:", error);
  tauriLLM = { isAvailable: () => false };
}
// Enhanced AI response interface with Gemma3 support
interface AIResponse {
  answer: string;
  sources: Array<{
    id: string;
    title: string;
    content: string;
    score: number;
    type: "case" | "evidence" | "document";
  }>;
  query: string;
  executionTime: number;
  fromCache: boolean;
  provider: "local" | "cloud" | "hybrid";
  model: string;
  confidence: number;
}
export const POST: RequestHandler = async ({ request, locals }) => {
  const startTime = Date.now();

  try {
    const {
      query,
      context = [],
      includeHistory = true,
      maxSources = 5,
      searchThreshold = 0.7,
      useCache = true,
    } = await request.json();

    if (!query || query.trim().length === 0) {
      return json(
        {
          success: false,
          error: "Query is required",
        },
        { status: 400 },
      );
    }
    // Check cache for identical queries
    const cacheKey = `ai_response:${JSON.stringify({ query, context, includeHistory })}`;
    if (useCache) {
      const cached = (await cache.get(cacheKey)) as AIResponse;
      if (cached) {
        return json({
          success: true,
          data: {
            ...cached,
            fromCache: true,
            executionTime: Date.now() - startTime,
          },
        });
      }
    }
    // Perform vector search to get relevant context
    let searchResults: any = { results: [] };

    try {
      if (vectorSearch && typeof vectorSearch === "function") {
        searchResults = await vectorSearch(query, {
          limit: maxSources * 2, // Get more to filter down
          threshold: searchThreshold,
          useCache: true,
          fallbackToQdrant: true,
          searchType: "hybrid",
        });
      } else {
        // Fallback: create mock results for testing Gemma3
        searchResults = {
          results: [
            {
              id: "mock-result-1",
              title: "Mock Legal Document",
              content: `Mock legal document content related to: ${query}. This is a placeholder result for testing Gemma3 integration.`,
              score: 0.85,
              type: "document",
            },
          ],
        };
      }
    } catch (searchError) {
      console.warn("Vector search failed, using fallback:", searchError);
      searchResults = { results: [] };
    }
    if (searchResults.results.length === 0) {
      return json({
        success: true,
        data: {
          answer:
            "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the relevant documents have been uploaded.",
          sources: [],
          query,
          executionTime: Date.now() - startTime,
          fromCache: false,
        },
      });
    }
    // Prepare context for LLM
    const relevantSources = searchResults.results.slice(0, maxSources);
    const contextText = relevantSources
      .map(
        (source: any) =>
          `Title: ${source.title}\nContent: ${source.content}\nType: ${source.type}`,
      )
      .join("\n\n---\n\n");

    // Generate AI response with Gemma3 Local LLM priority
    let aiAnswer: string;
    let provider: "local" | "cloud" | "hybrid";
    let model: string;
    let confidence: number;

    try {
      // Try Ollama Gemma3 first (web environment)
      if (ollamaService.checkAvailability()) {
        console.log("Using Ollama Gemma3 for inference");

        const systemPrompt = `You are a specialized legal AI assistant. Based on the provided context documents, answer the user's question accurately and professionally.

Context Documents:
${contextText}

Instructions:
- Provide clear, professional legal analysis
- Cite specific information from the context
- If information is insufficient, state this clearly
- Use appropriate legal terminology
- Be concise but thorough`;

        const response = await ollamaService.generate(query, {
          system: systemPrompt,
          temperature: 0.7,
          maxTokens: 512,
        });

        aiAnswer = response;
        provider = "local";
        model = "gemma2:2b";
        confidence = 0.85; // High confidence for local processing
      } else if (tauriLLM && tauriLLM.isAvailable()) {
        // Try Tauri LLM (desktop environment)
        await tauriLLM.initialize();

        console.log("Using Tauri Gemma3 local LLM for inference");

        const systemPrompt = `You are a specialized legal AI assistant. Based on the provided context documents, answer the user's question accurately and professionally.

Context Documents:
${contextText}

Instructions:
- Provide clear, professional legal analysis
- Cite specific information from the context
- If information is insufficient, state this clearly
- Use appropriate legal terminology
- Be concise but thorough`;

        aiAnswer = await tauriLLM.runInference(query, {
          temperature: 0.7,
          maxTokens: 512,
          systemPrompt: systemPrompt,
        });

        provider = "local";
        model = tauriLLM.getCurrentModels().chat || "gemma3-local";
        confidence = 0.85; // High confidence for local processing
      } else {
        // Fallback to AI service or template response
        try {
          if (aiService && typeof aiService.generateResponse === "function") {
            aiAnswer = await aiService.generateResponse(query, {
              provider: "auto",
              legalContext: true,
              context: relevantSources.map((s: any) => s.content),
              temperature: 0.7,
              maxTokens: 512,
            });
            provider = "cloud";
            model = "cloud-llm";
            confidence = 0.75;
          } else {
            // Template response when no AI service is available
            aiAnswer = generateFallbackResponse(query, relevantSources);
            provider = "hybrid";
            model = "template";
            confidence = 0.5;
          }
        } catch (cloudError) {
          console.warn("Cloud AI failed, using template response:", cloudError);
          aiAnswer = generateFallbackResponse(query, relevantSources);
          provider = "hybrid";
          model = "template";
          confidence = 0.5;
        }
      }
    } catch (error) {
      console.error("AI inference failed:", error);
      aiAnswer = generateFallbackResponse(query, relevantSources);
      provider = "hybrid";
      model = "fallback";
      confidence = 0.4;
    }
    const response: AIResponse = {
      answer: aiAnswer,
      sources: relevantSources.map((source: any) => ({
        id: source.id,
        title: source.title,
        content: source.content.substring(0, 200) + "...", // Truncate for UI
        score: source.score,
        type: source.type,
      })),
      query,
      executionTime: Date.now() - startTime,
      fromCache: false,
      provider,
      model,
      confidence,
    };

    // Cache the response
    if (useCache) {
      await cache.set(cacheKey, response, 10 * 60 * 1000); // 10 minutes
    }
    return json({
      success: true,
      data: response,
    });
  } catch (error) {
    console.error("AI endpoint error:", error);
    return json(
      {
        success: false,
        error: "Failed to process AI request",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Generate AI response using available LLM
async function generateAIResponse(
  query: string,
  context: string,
  conversationHistory: any[] = [],
): Promise<string> {
  // Try Ollama first (local LLM)
  try {
    const isAvailable = await ollamaService.initialize();
    if (isAvailable) {
      return generateOllamaResponse(query, context, conversationHistory);
    }
  } catch (error) {
    console.warn("Ollama not available:", error);
  }
  // Fallback to OpenAI if available
  if (env.OPENAI_API_KEY) {
    return generateOpenAIResponse(query, context, conversationHistory);
  }
  // Final fallback: template-based response
  return generateTemplateResponse(query, context);
}
// Ollama local LLM integration
async function generateOllamaResponse(
  query: string,
  context: string,
  history: any[],
): Promise<string> {
  try {
    const ollamaUrl = env.OLLAMA_URL || "http://localhost:11434";

    const prompt = `Context Information:
${context}

Previous conversation (if any):
${history.map((h) => `${h.role}: ${h.content}`).join("\n")}

User Question: ${query}

Instructions: Based on the provided context, answer the user's question accurately and concisely. If the context doesn't contain enough information to answer the question, say so clearly. Always cite specific sources when possible.

Answer:`;

    const response = await fetch(`${ollamaUrl}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: env.OLLAMA_MODEL || "llama2",
        prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 512,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }
    const data = await response.json();
    return data.response || "No response generated";
  } catch (error) {
    console.error("Ollama error:", error);
    throw error;
  }
}
// OpenAI GPT integration
async function generateOpenAIResponse(
  query: string,
  context: string,
  history: any[],
): Promise<string> {
  try {
    const messages = [
      {
        role: "system",
        content: `You are a helpful AI assistant for a legal case management system. Use the provided context to answer questions accurately and cite sources when possible. If you don't have enough information, say so clearly.`,
      },
      ...history.slice(-5), // Include last 5 conversation turns
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${query}`,
      },
    ];

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages,
        max_tokens: 512,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }
    const data = await response.json();
    return data.choices[0]?.message?.content || "No response generated";
  } catch (error) {
    console.error("OpenAI error:", error);
    throw error;
  }
}
// Template-based fallback response
function generateTemplateResponse(query: string, context: string): string {
  const sources = context.split("---").length;

  return `Based on ${sources} relevant document(s) in the database, I found the following information related to your query "${query}":

${context.substring(0, 300)}...

Please note: This is a summary based on available documents. For more detailed information, please review the source documents or contact the relevant personnel.`;
}
// Helper functions for Gemma3 integration

function generateFallbackResponse(query: string, sources: any[]): string {
  if (sources.length === 0) {
    return `I couldn't find any relevant information to answer your question: "${query}". Please try rephrasing your query or ensure the relevant documents have been uploaded to the system.`;
  }
  let response = `# Legal Analysis for: "${query}"\n\n`;

  response += `## Document Summary\n`;
  response += `I've identified ${sources.length} relevant documents from your case files:\n\n`;

  sources.slice(0, 3).forEach((source: any, index: number) => {
    response += `**Document ${index + 1}** (${source.type})\n`;
    response += `- Title: ${source.title}\n`;
    response += `- Relevance Score: ${(source.score * 100).toFixed(1)}%\n`;
    response += `- Content Preview: ${source.content.substring(0, 200)}...\n\n`;
  });

  response += `## Key Findings\n`;
  response += `Based on the available documents, please review the source materials for detailed information related to your query.\n\n`;

  response += `## Recommendations\n`;
  response += `- Review the source documents for comprehensive information\n`;
  response += `- Consider searching with more specific terms\n`;
  response += `- For AI-powered analysis, ensure Gemma3 local LLM is properly configured\n`;

  return response;
}
// Environment variable fallback
let env: any = {};
try {
  const envModule = await import("$env/dynamic/private");
  env = envModule.env || {};
} catch (error) {
  console.warn("Environment variables not available:", error);
  env = {};
}
// Autocomplete endpoint for citations
export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get("q") || "";
    const limit = parseInt(url.searchParams.get("limit") || "5");

    if (query.length < 2) {
      return json({ suggestions: [] });
    }
    // Quick search for autocomplete
    const searchResults = await vectorSearch(query, {
      limit,
      threshold: 0.5,
      useCache: true,
      searchType: "similarity",
    });

    const suggestions = searchResults.results.map((result) => ({
      id: result.id,
      title: result.title,
      type: result.type,
      score: result.score,
      preview: result.content.substring(0, 100) + "...",
    }));

    return json({ suggestions });
  } catch (error) {
    console.error("Autocomplete error:", error);
    return json({ suggestions: [] });
  }
};
