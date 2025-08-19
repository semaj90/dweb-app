import { json } from "@sveltejs/kit";
// Orphaned content: import type { RequestHandler
// Prefer the unified server-side vector search which uses pgvector and/or Qdrant, with caching and fallbacks
type { VectorSearchResult as ServerVectorSearchResult }, {
vectorSearch,
  getQueryEmbeddingLegal,
  searchLegalDocumentsText,
} from "$lib/server/search/vector-search";
// Orphaned content: // Health checks
import { ollamaService
import {
isQdrantHealthy } from "$lib/server/vector/qdrant";
// Keep enhanced pipeline as a secondary fallback when available
type { EnhancedSemanticSearchOptions }, {
enhancedAIPipeline } from "$lib/services/enhanced-ai-pipeline";
// Orphaned content: // Direct DB/vector access for legal_documents
import { db, sql

// Enhanced POST endpoint using Go microservice or local fallback
export const POST: RequestHandler = async ({ request }) => {
  try {
    // Parse request body with better error handling
    let requestBody;
    try {
      const text = await request.text();
      if (!text || text.trim() === "") {
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
          details:
            parseError instanceof Error
              ? parseError.message
              : "Unknown parse error",
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

    const GPU_GO_BASE = import.meta.env.GO_GPU_API_URL || "http://localhost:8084";
    const SUMM_BASE =
      import.meta.env.SUMMARIZER_BASE_URL || "http://localhost:8091";

    // Check if Go GPU API is available (8084), else try summarizer (8091)
    let goServiceAvailable = false;
    let summarizerAvailable = false;
    try {
      const goHealthCheck = await fetch(`${GPU_GO_BASE}/api/health`, {
        method: "GET",
        signal: AbortSignal.timeout(1000), // 1 second timeout
      });
      goServiceAvailable = goHealthCheck.ok;
    } catch {
      console.log("Go microservice not available, using fallback");
    }
    if (!goServiceAvailable) {
      try {
        const summHealth = await fetch(`${SUMM_BASE}/health`, {
          method: "GET",
          signal: AbortSignal.timeout(1000),
        });
        summarizerAvailable = summHealth.ok;
      } catch {
        summarizerAvailable = false;
      }
    }

    // If Go service is available and we're doing legal document search
    const looksLegal =
      options.searchType === "legal" ||
      query.toLowerCase().includes("legal") ||
      query.toLowerCase().includes("contract");
    if (goServiceAvailable && looksLegal) {
      try {
        const goResponse = await fetch(`${GPU_GO_BASE}/api/ai/summarize`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            content: query,
            document_type: options.documentType || "general",
            options: {
              style: "search",
              max_length: 200,
              temperature: 0.1,
            },
          }),
          signal: AbortSignal.timeout(5000), // 5 second timeout
        });

        if (goResponse.ok) {
          const goResult = await goResponse.json();
          return json({
            success: true,
            results: [
              {
                content: goResult.summary?.executive_summary || "",
                metadata: {
                  source: "go-microservice",
                  confidence: goResult.summary?.confidence || 0,
                  key_findings: goResult.summary?.key_findings || [],
                },
                score: goResult.summary?.confidence || 0.8,
              },
            ],
            count: 1,
            source: "go-microservice-gpu",
            executionTimeMs: goResult.processing_time || 0,
            timestamp: new Date().toISOString(),
          });
        }
      } catch (goError) {
        console.warn("Go microservice request failed, falling back:", goError);
      }
    }

    // Fall back to direct summarizer if available and query looks legal
    if (!goServiceAvailable && summarizerAvailable && looksLegal) {
      try {
        const summResp = await fetch(`${SUMM_BASE}/summarize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: query,
            format: "summary",
            maxTokens: 200,
          }),
          signal: AbortSignal.timeout(5000),
        });
        if (summResp.ok) {
          const data = await summResp.json();
          const content =
            data && typeof data.response === "string" ? data.response : "";
          return json({
            success: true,
            results: [
              {
                content,
                metadata: {
                  source: "summarizer-http",
                  model: data.model,
                  format: data.format,
                },
                score: 0.8,
              },
            ],
            count: 1,
            source: "summarizer-http",
            executionTimeMs: 0,
            timestamp: new Date().toISOString(),
          });
        }
      } catch (summErr) {
        console.warn("Summarizer fallback failed:", summErr);
      }
    }

    // Try direct pgvector search on legal_documents using Ollama embeddings (matches seeded data)
    // This runs before the generic unified search so we get true legal doc retrieval first.
    const legalResults = await (async () => {
      try {
        const embedding = await getQueryEmbeddingLegal(query);
        if (!Array.isArray(embedding) || embedding.length === 0) return [];

        // Convert to pgvector literal and query legal_documents
        const vectorString = `[${embedding.join(",")}]`;
        const limit = Math.min(Math.max(options.limit ?? 10, 1), 50);
        const threshold = options.minSimilarity ?? 0.6;

        const result = await db.execute(sql`
          SELECT
            id,
            title,
            content,
            1 - (embedding <=> ${vectorString}::vector) AS score
          FROM legal_documents
          WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> ${vectorString}::vector) > ${threshold}
          ORDER BY embedding <=> ${vectorString}::vector
          LIMIT ${limit}
        `);

        const mapped: ServerVectorSearchResult[] = (result as any[]).map(
          (r: any) => ({
            id: r.id,
            title: r.title || "",
            content: r.content || "",
            score:
              typeof r.score === "number"
                ? r.score
                : parseFloat(String(r.score ?? 0)),
            metadata: {},
            source: "pgvector",
            type: "document",
          })
        );

        return mapped;
      } catch (e) {
        // Silent fallthrough to generic pipeline
        return [] as ServerVectorSearchResult[];
      }
    })();

    if (legalResults.length > 0) {
      return json({
        success: true,
        results: legalResults,
        count: legalResults.length,
        source: "pgvector-legal",
        executionTimeMs: 0,
        timestamp: new Date().toISOString(),
      });
    }

    // If legal vector search failed, try text fallback directly on legal_documents
    try {
      const tf = await searchLegalDocumentsText(
        query,
        Math.min(options.limit ?? 10, 50)
      );
      if (tf.length > 0) {
        return json({
          success: true,
          results: tf,
          count: tf.length,
          source: "legal-text",
          executionTimeMs: 0,
          timestamp: new Date().toISOString(),
        });
      }
    } catch {}

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
        source: "error",
        executionTime: 0,
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
        results = [
          {
            content: `No results found for query: "${query}"`,
            metadata: { source: "fallback" },
            score: 0,
          },
        ] as unknown as CombinedResult[];
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
    const checkService = async (url: string): Promise<boolean> => {
      try {
        const response = await fetch(url, {
          method: "GET",
          signal: AbortSignal.timeout(2000), // 2 second timeout
        });
        return response.ok;
      } catch {
        return false;
      }
    };

    const GPU_GO_BASE = "http://localhost:8084"; // Default Go microservice endpoint

    const [goHealthy, qdrantHealthy, ollamaHealthy] = await Promise.all([
      checkService(`${GPU_GO_BASE}/api/health`),
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
