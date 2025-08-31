
import type { RequestHandler } from './$types';

// API endpoint for vector search operations
import VectorService from "$lib/server/services/vector-service";

import { z } from "zod";

// Request validation schemas
const searchSchema = z.object({
  query: z.string().min(1).max(1000),
  documentType: z.enum(["case", "evidence", "note", "report"]).optional(),
  limit: z.number().min(1).max(50).default(10),
  threshold: z.number().min(0).max(1).default(0.7),
});

const storeDocumentSchema = z.object({
  documentId: z.string(),
  documentType: z.enum(["case", "evidence", "note", "report"]),
  text: z.string().min(1),
  metadata: z.record(z.any()).optional(),
});

const analyzeSchema = z.object({
  text: z.string().min(1),
  analysisType: z.enum([
    "summary",
    "key_points",
    "legal_issues",
    "recommendations",
  ]),
});

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    const action = body.action;

    switch (action) {
      case "search": {
        const params = searchSchema.parse(body);
        const results = await VectorService.semanticSearch(params.query, {
          type: params.documentType,
          limit: params.limit,
          threshold: params.threshold
        });

        return json({
          success: true,
          results,
          count: results.length,
        });
      }
      case "store": {
        const params = storeDocumentSchema.parse(body);
        const embeddings = await VectorService.storeDocument(
          params.documentId,
          params.documentType,
          params.text,
          params.metadata
        );

        return json({
          success: true,
          embeddingIds: embeddings,
          count: embeddings.length,
        });
      }
      case "analyze": {
        const params = analyzeSchema.parse(body);
        const analysis = await VectorService.analyzeDocument(
          params.text,
          params.analysisType
        );

        return json({
          success: true,
          analysis,
          analysisType: params.analysisType,
        });
      }
      case "similar": {
        const { documentId, limit = 5 } = body;
        const similar = await VectorService.findSimilarDocuments(
          documentId,
          limit
        );

        return json({
          success: true,
          similar,
          count: similar.length,
        });
      }
      case "test": {
        // Test endpoint to verify Ollama connection
        const testEmbedding = await VectorService.generateEmbedding(
          "This is a test legal document for vector embedding."
        );

        return json({
          success: true,
          message: "Vector service is working!",
          embeddingDimension: testEmbedding.length,
          model: "test-model",
        });
      }
      default:
        return json(
          { success: false, error: "Invalid action" },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error("Vector API error:", error);

    if (error instanceof z.ZodError) {
      return json(
        { success: false, error: "Validation error", details: error.errors },
        { status: 400 }
      );
    }
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async () => {
  // Health check endpoint
  try {
    const ollamaUrl = import.meta.env.OLLAMA_URL || "http://localhost:11434";
    const response = await fetch(`${ollamaUrl}/api/tags`);
    const data = await response.json();

    return json({
      success: true,
      status: "healthy",
      ollama: {
        connected: true,
        models: data.models?.map((m: any) => m.name) || [],
      },
      embedding: {
        model: import.meta.env.OLLAMA_EMBEDDING_MODEL || "nomic-embed-text",
        dimension: parseInt(import.meta.env.EMBEDDING_DIMENSION || "768"),
      },
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        status: "unhealthy",
        error: "Failed to connect to Ollama",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 503 }
    );
  }
};
