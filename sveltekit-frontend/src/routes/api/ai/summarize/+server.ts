import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";

// AI Summarization endpoint using local Ollama
const OLLAMA_BASE_URL = import.meta.env.OLLAMA_BASE_URL || "http://localhost:11434";
const DEFAULT_MODEL = import.meta.env.OLLAMA_MODEL || "gemma2:2b";

export const GET: RequestHandler = async () => {
  try {
    // Health check for Ollama service
    const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    const models = await res.json().catch(() => ({ models: [] }));
    
    return json({
      ok: true,
      status: "healthy",
      service: "ai-summarization",
      models: models.models?.map((m: any) => m.name) || [],
      endpoint: `${OLLAMA_BASE_URL}/api/generate`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return json({
      ok: false,
      error: "Ollama service unreachable",
      service: "ai-summarization",
      endpoint: `${OLLAMA_BASE_URL}/api/generate`,
      timestamp: new Date().toISOString()
    }, { status: 503 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { text, type = "legal", options = {} } = await request.json();
    
    if (!text || typeof text !== "string" || text.trim().length < 10) {
      return json({
        success: false,
        error: "Text is required and must be at least 10 characters long"
      }, { status: 400 });
    }

    const maxTokens = options.max_tokens || 500;
    const style = type === "legal" ? "legal document" : "document";
    
    const prompt = `Please provide a concise summary of the following ${style}. Focus on the key points, main arguments, and essential information. Limit your response to approximately ${maxTokens / 4} words.

Text to summarize:
${text.trim()}

Summary:`;

    const startTime = Date.now();
    
    const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.3, // Lower temperature for more focused summaries
          top_p: 0.9,
          max_tokens: maxTokens
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const result = await response.json();
    const duration = Date.now() - startTime;
    
    if (!result.response) {
      throw new Error("No response from AI model");
    }

    // Clean up the summary response
    const summary = result.response.trim();
    
    return json({
      success: true,
      summary: summary,
      model: DEFAULT_MODEL,
      type: type,
      originalLength: text.length,
      summaryLength: summary.length,
      compressionRatio: (summary.length / text.length * 100).toFixed(1) + "%",
      performance: {
        duration: duration,
        tokens: result.eval_count || 0,
        promptTokens: result.prompt_eval_count || 0,
        tokensPerSecond: result.eval_count ? (result.eval_count / (duration / 1000)).toFixed(2) : 0
      },
      timestamp: new Date().toISOString(),
      suggestions: [
        "Try summarizing different sections separately",
        "Ask for specific aspects to focus on",
        "Request different summary lengths"
      ]
    });

  } catch (error) {
    console.error("AI summarization error:", error);
    
    return json({
      success: false,
      error: "AI summarization service temporarily unavailable",
      details: import.meta.env.NODE_ENV === 'development' ? String(error) : undefined,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};
