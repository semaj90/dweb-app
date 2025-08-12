import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
// Prefer the unified server-side vector search which uses pgvector and/or Qdrant, with caching and fallbacks
import type { VectorSearchResult as ServerVectorSearchResult } from "$lib/server/search/vector-search";
import { vectorSearch } from "$lib/server/search/vector-search";
// Health checks
import { ollamaService } from "$lib/server/services/OllamaService";
import { isQdrantHealthy } from "$lib/server/vector/qdrant";
// Keep enhanced pipeline as a secondary fallback when available
import type { EnhancedSemanticSearchOptions } from "$lib/services/enhanced-ai-pipeline";
import { enhancedAIPipeline } from "$lib/services/enhanced-ai-pipeline";

// Enhanced POST endpoint using Go microservice or local fallback
export const POST: RequestHandler = async ({ request }) => {
  try {
    // Parse request body with better error handling
    let requestBody;
    try {
      const text = await request.text();
      if (!text || text.trim() === '') {
        return json(
          {
            success: false,
            error: "Request body is empty",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        );
      }
      requestBody = JSON.parse(text);
    } catch (parseError) {
      console.error("JSON parse error:", parseError);
      return json(
        {
          success: false,
          error: "Invalid JSON in request body",
          details: parseError instanceof Error ? parseError.message : "Unknown parse error",
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    const { query, options = {} } = requestBody;

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

    // Check if Go microservice is available (port 8084)
    let goServiceAvailable = false;
    try {
      const goHealthCheck = await fetch('http://localhost:8084/api/health', {
        method: 'GET',
        signal: AbortSignal.timeout(1000) // 1 second timeout
      });
      goServiceAvailable = goHealthCheck.ok;
    } catch {
      console.log("Go microservice not available, using fallback");
    }

    // If Go service is available and we're doing legal document search
    if (goServiceAvailable && (options.searchType === 'legal' || query.toLowerCase().includes('legal') || query.toLowerCase().includes('contract'))) {
      try {
        const goResponse = await fetch('http://localhost:8084/api/ai/summarize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            content: query,
            document_type: options.documentType || 'general',
            options: {
              style: 'search',
              max_length: 200,
              temperature: 0.1
            }
          }),
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });

        if (goResponse.ok) {
          const goResult = await goResponse.json();
          return json({
            success: true,
            results: [{
              content: goResult.summary?.executive_summary || '',
              metadata: {
                source: 'go-microservice',
                confidence: goResult.summary?.confidence || 0,
                key_findings: goResult.summary?.key_findings || []
              },
              score: goResult.summary?.confidence || 0.8
            }],
            count: 1,
            source: 'go-microservice-gpu',
            executionTimeMs: goResult.processing_time || 0,
            timestamp: new Date().toISOString(),
          });
        }
      } catch (goError) {
        console.warn("Go microservice request failed, falling back:", goError);
      }
    }

    // 1) Try unified vectorSearch (pgvector primary, Qdrant/Loki/Fuse fallbacks)
    let vs;
    try {
      vs = await vectorSearch(query, {
        limit: options.limit ?? 10,
        threshold: options.minSimilarity ?? 0.6,
        useCache: options.useCache !== false,
        fallbackToQdrant: options.fallbackToQdrant !== false,
        useFuzzySearch: options.useFuzzySearch === true,
        useLocalDb: options.useLocalDb === true,
        filters: options.filters || {},
        searchType: options.searchType || "hybrid",
      });
    } catch (vsError) {
      console.warn("Vector search failed:", vsError);
      // Create a fallback response
      vs = {
        results: [],
        source: 'error',
        executionTime: 0
      };
    }

    type CombinedResult = ServerVectorSearchResult | unknown;
    let results: CombinedResult[] = vs.results as ServerVectorSearchResult[];
    let used: string = vs.source;

    // 2) If nothing came back, fall back to enhanced pipeline (can use Go microservice if healthy)
    if (!results || results.length === 0) {
      try {
        const enhancedOptions: EnhancedSemanticSearchOptions = {
          limit: options.limit ?? 10,
          useCache: options.useCache !== false,
          minSimilarity: options.minSimilarity ?? 0.6,
          temperature: options.temperature ?? 0.1,
        };
        const enhancedResults =
          await enhancedAIPipeline.performEnhancedSemanticSearch(
            query,
            enhancedOptions
          );
        results = enhancedResults as unknown as CombinedResult[];
        used = "enhanced-pipeline";
      } catch (e) {
        // keep empty results on failure
        console.warn("Enhanced pipeline fallback failed:", e);
        
        // Final fallback: return a basic response
        results = [{
          content: `No results found for query: "${query}"`,
          metadata: { source: 'fallback' },
          score: 0
        }] as unknown as CombinedResult[];
        used = "fallback";
      }
    }

    return json({
      success: true,
      results,
      count: results?.length || 0,
      source: used,
      executionTimeMs: vs.executionTime || 0,
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
    // Check all services in parallel with timeouts
    const checkService = async (url: string, name: string): Promise<boolean> => {
      try {
        const response = await fetch(url, {
          method: 'GET',
          signal: AbortSignal.timeout(2000) // 2 second timeout
        });
        return response.ok;
      } catch {
        return false;
      }
    };

    const [goHealthy, qdrantHealthy, ollamaHealthy] = await Promise.all([
      checkService('http://localhost:8084/api/health', 'go-microservice'),
      isQdrantHealthy().catch(() => false),
      ollamaService.isHealthy().catch(() => false),
    ]);

    return json({
      status: "operational",
      services: {
        goMicroservice: goHealthy ? "healthy" : "unavailable",
        qdrant: qdrantHealthy ? "healthy" : "unavailable",
        ollama: ollamaHealthy ? "healthy" : "unavailable",
        localPipeline: "available",
        fallbackEnabled: true,
      },
      capabilities: {
        vectorSearch: true,
        semanticAnalysis: true,
        documentIngestion: true,
        multiModelSupport: true,
        gpuAcceleration: goHealthy,
      },
      supportedModels: ["claude", "gemini", "ollama", "gemma3-legal"],
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
