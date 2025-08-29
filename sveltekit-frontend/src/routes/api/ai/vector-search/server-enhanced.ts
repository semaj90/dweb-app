
import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";
import { vectorSearchService } from "$lib/services/real-vector-search-service";

export interface EnhancedSearchOptions {
  maxResults?: number;
  threshold?: number;
  collection?: string;
  includeMetadata?: boolean;
  filter?: Record<string, any>;
}

export interface SearchRequest {
  query: string;
  options?: EnhancedSearchOptions;
}

export interface SearchResponse {
  success: boolean;
  results?: any[];
  error?: string;
  queryTime?: number;
  model?: string;
}

// Enhanced POST endpoint using real vector search service
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { query, options = {} } = await request.json();

    if (!query || typeof query !== "string") {
      return json(
        { error: "Query is required and must be a string" },
        { status: 400 }
      );
    }

    // Create search options with defaults
    const searchOptions: EnhancedSearchOptions = {
      maxResults: options.maxResults || options.limit || 10,
      threshold: options.threshold || options.minSimilarity || 0.7,
      collection: options.collection || 'legal_documents',
      includeMetadata: options.includeMetadata !== false,
      filter: options.filter
    };

    // Perform real vector search
    const searchResponse = await vectorSearchService.search(query, searchOptions);

    if (!searchResponse.success) {
      return json(
        { 
          success: false, 
          error: "Vector search failed",
          timestamp: new Date().toISOString()
        },
        { status: 500 }
      );
    }

    // Enhanced response with detailed metadata
    return json({
      success: true,
      results: searchResponse.results,
      query,
      total: searchResponse.totalResults,
      queryTime: searchResponse.queryTime,
      timestamp: new Date().toISOString(),
      searchMetadata: {
        embeddingModel: searchResponse.model,
        vectorStore: "qdrant",
        searchOptions,
      },
      performance: {
        queryTimeMs: searchResponse.queryTime,
        resultsFound: searchResponse.totalResults,
      }
    });
  } catch (error: any) {
    console.error("Enhanced vector search API error:", error);

    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error during vector search",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// GET endpoint for search status and health check
export const GET: RequestHandler = async () => {
  try {
    // Check real vector search service health
    const healthStatus = await vectorSearchService.healthCheck();

    return json({
      status: "operational",
      services: {
        ollama: healthStatus.ollama ? "healthy" : "unavailable",
        qdrant: healthStatus.qdrant ? "healthy" : "unavailable",
        vectorSearch: healthStatus.overall ? "operational" : "degraded",
      },
      capabilities: {
        vectorSearch: healthStatus.overall,
        semanticAnalysis: healthStatus.ollama,
        legalAnalysis: healthStatus.overall,
        embeddings: healthStatus.ollama,
        vectorStorage: healthStatus.qdrant,
        realTimeSearch: healthStatus.overall,
      },
      metadata: {
        embeddingModel: "nomic-embed-text",
        vectorStore: "qdrant",
        searchEngine: "cosine_similarity",
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        status: "error",
        error: error instanceof Error ? error.message : "Health check failed",
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};
