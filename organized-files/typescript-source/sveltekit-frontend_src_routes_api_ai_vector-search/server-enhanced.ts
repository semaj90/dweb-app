// @ts-nocheck
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { enhancedAiPipeline } from "$lib/services/enhanced-ai-pipeline";
import type {
  EnhancedSearchOptions,
  SearchRequest,
  SearchResponse,
} from "$lib/types/ai-types";

// Enhanced POST endpoint using Go microservice or local fallback
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options = {} } = await request.json();

    if (!query || typeof query !== "string") {
      return json(
        { error: "Query is required and must be a string" },
        { status: 400 }
      );
    }

    // Initialize enhanced AI pipeline
    await enhancedAiPipeline.initialize();

    // Create search options with defaults
    const searchOptions: EnhancedSearchOptions = {
      limit: options.limit || 10,
      minSimilarity: options.minSimilarity || 0.6,
      documentType: options.documentType,
      practiceArea: options.practiceArea,
      jurisdiction: options.jurisdiction || "US",
      useCache: options.useCache !== false,
      useGPU: options.useGPU !== false,
      temperature: options.temperature || 0.1,
      ragMode: options.ragMode || "enhanced",
      includeContext: options.includeContext !== false,
      contextWindow: options.contextWindow || 4096,
    };

    // Perform enhanced semantic search (will use Go microservice if available)
    const results = await enhancedAiPipeline.semanticSearch(
      query,
      searchOptions
    );

    // Enhanced response with detailed metadata
    return json({
      success: true,
      results,
      query,
      total: results.length,
      timestamp: new Date().toISOString(),
      searchMetadata: {
        embeddingModel: "nomic-embed-text",
        vectorStore: "postgresql-pgvector",
        llmModel: "gemma3-legal",
        processingMode: "enhanced",
        usedGoMicroservice: enhancedAiPipeline.useGoMicroservice,
        searchOptions,
      },
      performance: {
        cached: results.some((r) => r.ragContext?.relatedDocuments?.length),
        gpu_accelerated: searchOptions.useGPU,
        rag_mode: searchOptions.ragMode,
      },
    });
  } catch (error) {
    console.error("Enhanced vector search API error:", error);

    return json(
      {
        success: false,
        error: error.message || "Internal server error during vector search",
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
    const goHealthy = await enhancedAiPipeline.checkGoMicroserviceHealth();

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
        legalAnalysis: true,
        cudaAcceleration: true,
        redisCache: true,
        gemma3Legal: true,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return json(
      {
        status: "error",
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};
