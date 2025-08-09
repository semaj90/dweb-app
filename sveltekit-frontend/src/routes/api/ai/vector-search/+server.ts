import { enhancedAIPipeline } from "$lib/services/enhanced-ai-pipeline";
import type { EnhancedSearchOptions } from "$lib/types/ai-types";
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

// Enhanced POST endpoint using Go microservice or local fallback
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options = {} } = await request.json();

    if (!query || typeof query !== "string") {
      return json(
        {
          success: false,
          error: "Query parameter is required and must be a string",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    const searchOptions: EnhancedSearchOptions = {
      limit: options.limit || 5,
      useCache: options.useCache !== false,
      temperature: options.temperature || 0.1,
    };

    // Try enhanced AI pipeline with Go microservice fallback
    const results = await enhancedAIPipeline.performEnhancedSemanticSearch(
      query,
      searchOptions
    );

    return json({
      success: true,
      results,
      count: results.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    console.error("Vector search error:", error);
    return json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Internal server error during vector search",
        timestamp: new Date().toISOString(),
        fallback: true,
      },
      { status: 500 }
    );
  }
};

// GET endpoint for search status and health check
export const GET: RequestHandler = async () => {
  try {
    // Check microservice health
    const goHealthy = await enhancedAIPipeline.checkGoMicroserviceHealth();

    return json({
      status: "operational",
      services: {
        goMicroservice: goHealthy ? "healthy" : "unavailable",
        localPipeline: "available",
        fallbackEnabled: true,
      },
      capabilities: {
        vectorSearch: true,
        semanticAnalysis: true,
        documentIngestion: true,
        multiModelSupport: true,
      },
      supportedModels: ["claude", "gemini", "ollama"],
      embeddingModels: [
        "ollama-nomic-embed-text",
        "claude-fallback",
        "gemini-embed",
      ],
      cacheEnabled: true,
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    console.error("Vector search status API error:", error);
    return json(
      {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};
