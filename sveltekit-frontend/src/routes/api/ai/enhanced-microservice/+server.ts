/// <reference types="vite/client" />
import type { RequestHandler } from './$types';

// Enhanced API Route for Go Microservice Integration
// Connects SvelteKit frontend with enhanced legal AI processing
// Supports local Gemma3-legal GGUF model with CUDA acceleration

import { json } from "@sveltejs/kit";
import { URL } from "url";

export interface GoMicroserviceConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  enableCache: boolean;
}

export interface ProcessDocumentRequest {
  content: string;
  document_type: string;
  practice_area?: string;
  jurisdiction: string;
  metadata?: Record<string, any>;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  filters?: Record<string, string>;
  use_rag?: boolean;
  include_context?: boolean;
}

const config: GoMicroserviceConfig = {
  baseUrl: import.meta.env.GO_MICROSERVICE_URL || "http://localhost:8080",
  timeout: parseInt(import.meta.env.GO_MICROSERVICE_TIMEOUT || "30000"),
  retries: 3,
  enableCache: true,
};

class GoMicroserviceClient {
  private baseUrl: string;
  private timeout: number;
  private retries: number;

  constructor(config: GoMicroserviceConfig) {
    this.baseUrl = config.baseUrl;
    this.timeout = config.timeout;
    this.retries = config.retries;
  }

  private async makeRequest(
    endpoint: string,
    method: string = "GET",
    body?: unknown
  ): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;

    for (let attempt = 1; attempt <= this.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const response = await fetch(url, {
          method,
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
      } catch (error: any) {
        console.warn(
          `Attempt ${attempt} failed for ${endpoint}:`,
          error.message
        );

        if (attempt === this.retries) {
          throw new Error(
            `Failed to connect to Go microservice after ${this.retries} attempts: ${error.message}`
          );
        }

        // Exponential backoff
        await new Promise((resolve) =>
          setTimeout(resolve, Math.pow(2, attempt) * 1000)
        );
      }
    }
  }

  async healthCheck(): Promise<any> {
    return this.makeRequest("/api/v1/health");
  }

  async processDocument(request: ProcessDocumentRequest): Promise<any> {
    return this.makeRequest("/api/v1/documents/process", "POST", request);
  }

  async searchDocuments(request: SearchRequest): Promise<any> {
    return this.makeRequest("/api/v1/search", "POST", request);
  }
}

// Initialize the client
const goClient = new GoMicroserviceClient(config);

// Health check endpoint
export const GET: RequestHandler = async ({ url }) => {
  const action = url.searchParams.get("action");

  try {
    if (action === "health") {
      const health = await goClient.healthCheck();

      return json({
        status: "success",
        timestamp: new Date().toISOString(),
        service: "sveltekit-api-gateway",
        microservice: health,
        config: {
          baseUrl: config.baseUrl,
          timeout: config.timeout,
          retries: config.retries,
          enableCache: config.enableCache,
        },
      });
    }

    return json(
      {
        status: "error",
        message: "Invalid action parameter",
        availableActions: ["health"],
      },
      { status: 400 }
    );
  } catch (error: any) {
    console.error("Health check failed:", error);

    return json(
      {
        status: "error",
        message: "Microservice unavailable",
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
};

// Document processing and search endpoints
export const POST: RequestHandler = async ({ request, url }) => {
  const action = url.searchParams.get("action");

  try {
    const body = await request.json();

    switch (action) {
      case "process-document": {
        // Validate required fields
        if (!body.content || !body.document_type || !body.jurisdiction) {
          return json(
            {
              status: "error",
              message:
                "Missing required fields: content, document_type, jurisdiction",
            },
            { status: 400 }
          );
        }

        const processRequest: ProcessDocumentRequest = {
          content: body.content,
          document_type: body.document_type,
          practice_area: body.practice_area,
          jurisdiction: body.jurisdiction,
          metadata: body.metadata || {},
        };

        const result = await goClient.processDocument(processRequest);

        return json({
          status: "success",
          timestamp: new Date().toISOString(),
          result: result,
          metadata: {
            processingTime: result.processing_time,
            chunksCreated: result.chunks?.length || 0,
            entitiesFound: result.legal_entities?.length || 0,
            riskScore: result.risk_assessment?.overall_score || 0,
          },
        });
      }

      case "search": {
        if (!body.query) {
          return json(
            {
              status: "error",
              message: "Query parameter is required",
            },
            { status: 400 }
          );
        }

        const searchRequest: SearchRequest = {
          query: body.query,
          limit: body.limit || 10,
          filters: body.filters || {},
          use_rag: body.use_rag || false,
          include_context: body.include_context || false,
        };

        const result = await goClient.searchDocuments(searchRequest);

        return json({
          status: "success",
          timestamp: new Date().toISOString(),
          result: result,
          metadata: {
            queryProcessingTime: result.processing_time,
            resultsFound: result.total_found,
            ragEnabled: searchRequest.use_rag,
            hasRAGContext: !!result.rag_context,
          },
        });
      }

      case "enhanced-rag": {
        if (!body.query) {
          return json(
            {
              status: "error",
              message: "Query parameter is required for enhanced RAG",
            },
            { status: 400 }
          );
        }

        // Enhanced RAG with multiple microservice calls
        const searchRequest: SearchRequest = {
          query: body.query,
          limit: body.limit || 20,
          filters: body.filters || {},
          use_rag: true,
          include_context: true,
        };

        const [searchResult, healthStatus] = await Promise.all([
          goClient.searchDocuments(searchRequest),
          goClient.healthCheck(),
        ]);

        // Enhanced response with microservice status
        return json({
          status: "success",
          timestamp: new Date().toISOString(),
          result: {
            ...searchResult,
            enhanced_rag: {
              cross_references:
                searchResult.rag_context?.cross_references || [],
              contextual_insights:
                searchResult.rag_context?.contextual_summary || "",
              related_documents:
                searchResult.rag_context?.related_documents || [],
              confidence_score: calculateConfidenceScore(searchResult.results),
              processing_pipeline: {
                embedding_model: "nomic-embed-text",
                llm_model: "gemma3-legal:8b",
                cuda_enabled: healthStatus.config?.enable_gpu || false,
                vector_similarity: true,
                semantic_analysis: true,
              },
            },
          },
          metadata: {
            queryProcessingTime: searchResult.processing_time,
            resultsFound: searchResult.total_found,
            enhancedRAG: true,
            microserviceStatus: healthStatus.status,
            processingMode: healthStatus.config?.enable_gpu
              ? "GPU-Accelerated"
              : "CPU-Only",
          },
        });
      }

      case "legal-analysis": {
        if (!body.content) {
          return json(
            {
              status: "error",
              message: "Content parameter is required for legal analysis",
            },
            { status: 400 }
          );
        }

        // Comprehensive legal analysis combining processing and search
        const processRequest: ProcessDocumentRequest = {
          content: body.content,
          document_type: body.document_type || "general",
          practice_area: body.practice_area,
          jurisdiction: body.jurisdiction || "federal",
          metadata: {
            analysis_type: "comprehensive",
            include_risk_assessment: true,
            include_entity_extraction: true,
            ...body.metadata,
          },
        };

        const processResult = await goClient.processDocument(processRequest);

        // Follow up with related document search
        const searchQueries = [
          ...processResult.keywords.slice(0, 3),
          ...processResult.legal_entities
            .map((entity: any) => entity.text)
            .slice(0, 2),
        ];

        const relatedSearches = await Promise.all(
          searchQueries.map((query: string) =>
            goClient
              .searchDocuments({
                query,
                limit: 5,
                use_rag: true,
                include_context: true,
              })
              .catch(() => ({ results: [], total_found: 0 }))
          )
        );

        return json({
          status: "success",
          timestamp: new Date().toISOString(),
          result: {
            document_analysis: processResult,
            related_research: {
              keyword_searches: relatedSearches,
              total_related_documents: relatedSearches.reduce(
                (sum, search) => sum + (search.total_found || 0),
                0
              ),
              cross_references: extractCrossReferences(relatedSearches),
            },
            comprehensive_insights: {
              risk_factors: processResult.risk_assessment?.risk_factors || [],
              legal_entities: processResult.legal_entities || [],
              key_themes: processResult.keywords || [],
              recommendations: generateRecommendations(
                processResult,
                relatedSearches
              ),
            },
          },
          metadata: {
            analysisType: "comprehensive",
            documentsProcessed: 1,
            relatedDocumentsFound: relatedSearches.reduce(
              (sum, search) => sum + (search.total_found || 0),
              0
            ),
            confidenceScore: calculateConfidenceScore([processResult]),
            processingPipeline: "gemma3-legal-enhanced",
          },
        });
      }

      default:
        return json(
          {
            status: "error",
            message: "Invalid action parameter",
            availableActions: [
              "process-document",
              "search",
              "enhanced-rag",
              "legal-analysis",
            ],
          },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error(`API action failed (${action}):`, error);

    return json(
      {
        status: "error",
        message: "Microservice request failed",
        error: error.message,
        timestamp: new Date().toISOString(),
        action: action,
      },
      { status: 500 }
    );
  }
};

// Helper functions
function calculateConfidenceScore(results: any[]): number {
  if (!results || results.length === 0) return 0;

  const scores = results.map((result) => {
    if (result.similarity) return result.similarity;
    if (result.risk_assessment?.confidence)
      return result.risk_assessment.confidence;
    return 0.5; // Default confidence
  });

  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

function extractCrossReferences(searches: any[]): string[] {
  const references = new Set<string>();

  searches.forEach((search) => {
    if (search.results) {
      search.results.forEach((result: any) => {
        if (result.document_id) {
          references.add(result.document_id);
        }
      });
    }
  });

  return Array.from(references);
}

function generateRecommendations(
  processResult: any,
  relatedSearches: any[]
): string[] {
  const recommendations: string[] = [];

  // Risk-based recommendations
  if (processResult.risk_assessment?.overall_score > 0.7) {
    recommendations.push(
      "High risk detected - recommend immediate legal review"
    );
  }

  // Entity-based recommendations
  if (processResult.legal_entities?.length > 5) {
    recommendations.push(
      "Multiple legal entities identified - consider entity relationship mapping"
    );
  }

  // Related document recommendations
  const totalRelated = relatedSearches.reduce(
    (sum, search) => sum + (search.total_found || 0),
    0
  );
  if (totalRelated > 20) {
    recommendations.push(
      "Extensive related documentation found - suggest comprehensive case analysis"
    );
  }

  // Default recommendations
  if (recommendations.length === 0) {
    recommendations.push(
      "Document analyzed successfully - consider cross-referencing with similar cases"
    );
  }

  return recommendations;
}
