
import type { RequestHandler } from './$types';

// Test endpoint for Gemma3 local LLM integration
import { json } from "@sveltejs/kit";
import { tauriLLM } from "$lib/services/tauri-llm";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { prompt, options = {} } = await request.json();

    if (!prompt) {
      return json(
        {
          error: "Prompt is required",
        },
        { status: 400 },
      );
    }
    // Initialize Tauri LLM service
    await tauriLLM.initialize();

    if (!tauriLLM.isAvailable()) {
      return json({
        success: false,
        error: "Local LLM not available",
        fallback: true,
        message:
          "Gemma3 is not loaded. Please ensure the desktop app is running with local LLM support.",
      });
    }
    // Test Gemma3 inference
    const startTime = Date.now();

    const response = await tauriLLM.runInference(prompt, {
      temperature: options.temperature || 0.7,
      maxTokens: options.maxTokens || 512,
      systemPrompt:
        options.systemPrompt ||
        "You are a helpful legal AI assistant. Provide clear, accurate responses.",
    });

    const inferenceTime = Date.now() - startTime;

    return json({
      success: true,
      data: {
        response,
        metadata: {
          inferenceTime,
          model: tauriLLM.getCurrentModels().chat || "gemma3-local",
          provider: "local",
          prompt: prompt.substring(0, 100) + (prompt.length > 100 ? "..." : ""),
        },
      },
    });
  } catch (error: any) {
    console.error("Gemma3 test failed:", error);

    return json(
      {
        success: false,
        error: "Gemma3 inference failed",
        details: error instanceof Error ? error.message : "Unknown error",
        troubleshooting: [
          "Ensure the desktop application is running",
          "Check that Gemma3 model files are available",
          "Verify sufficient system memory (4GB+ recommended)",
          "Check Tauri backend logs for detailed errors",
        ],
      },
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async () => {
  try {
    await tauriLLM.initialize();

    const status = {
      available: tauriLLM.isAvailable(),
      models: tauriLLM.getAvailableModels(),
      currentModels: tauriLLM.getCurrentModels(),
      initialized: true,
    };

    return json({
      success: true,
      status,
      message: status.available
        ? "Gemma3 local LLM is ready"
        : "Local LLM not available - running in web mode",
    });
  } catch (error: any) {
    return json({
      success: false,
      status: {
        available: false,
        initialized: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
    });
  }
};
