/// <reference types="vite/client" />

import { json } from "@sveltejs/kit";
import type { RequestHandler } from './$types';


export const GET = (async (): Promise<any> => {
  try {
    // Check if service is available
    const isAvailable = await ollamaService.healthCheck();
    const models = ollamaService.getAvailableModels();
    const currentModel = ollamaService.getGemma3Model();

    // Get more detailed status
    let ollamaDetails = null;
    try {
      const response = await fetch("http://localhost:11434/api/version");
      if (response.ok) {
        ollamaDetails = await response.json();
      }
    } catch (error: any) {
      // Ollama not accessible
    }

    return json({
      status: "success",
      timestamp: new Date().toISOString(),
      ollama: {
        available: isAvailable,
        version: ollamaDetails?.version || "unknown",
        models: models,
        gemma3Model: currentModel,
        modelCount: models.length,
        serviceUrl: "http://localhost:11434",
      },
      sveltekit: {
        version: "5.x",
        mode: import.meta.env.NODE_ENV || "development",
        apiBase: "/api/ai",
      },
      integration: {
        chatEndpoint: "/api/ai/chat",
        testEndpoint: "/api/ai/test-ollama",
        ollamaDirectEndpoint: "/api/ai/ollama-gemma3",
      },
    });
  } catch (error: any) {
    return json(
      {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
});

export const POST = (async ({ request }): Promise<any> => {
  try {
    const { prompt = "What are the key elements of a valid contract?" } =
      await request.json();

    // Check if Ollama service is available
    const isAvailable = await ollamaService.healthCheck();
    if (!isAvailable) {
      return json(
        {
          status: "error",
          error: "Ollama service not available",
          suggestion: "Please ensure Ollama is running: ollama serve",
        },
        { status: 503 },
      );
    }

    const startTime = Date.now();

    try {
      const response = await ollamaService.generate(prompt, {
        system:
          "You are a specialized Legal AI Assistant. Provide concise, accurate legal information.",
        temperature: 0.1,
        maxTokens: 256,
        topP: 0.8,
        topK: 20,
        repeatPenalty: 1.05,
      });

      const executionTime = Date.now() - startTime;

      return json({
        status: "success",
        prompt,
        response,
        model: ollamaService.getGemma3Model(),
        execution: {
          timeMs: executionTime,
          tokensEstimate: Math.ceil(response.length / 4), // Rough estimate
          provider: "ollama",
        },
        timestamp: new Date().toISOString(),
      });
    } catch (generateError) {
      return json(
        {
          status: "error",
          error: "Model generation failed",
          details:
            generateError instanceof Error
              ? generateError.message
              : "Unknown generation error",
          suggestion: "Check if gemma3-legal model is imported: ollama list",
        },
        { status: 500 },
      );
    }
  } catch (error: any) {
    return json(
      {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    );
  }
});
