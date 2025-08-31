import { json } from "@sveltejs/kit";
import type { RequestHandler } from './$types';


// Embeddings API Endpoint
// Handles embedding generation and bulk embedding operations
import {
  generateBatchEmbeddings,
  generateEmbedding,
} from "$lib/services/enhanced-ollama-service";
import { fetchEmbedding } from "$lib/server/qdrant";
// import { syncDocumentEmbeddings

// Temporary fallback function
async function syncDocumentEmbeddings(
  type: string,
  limit: number,
  forceRegenerate: boolean,
): Promise<{ total: number; updated: number; errors: number }> {
  console.warn(`syncDocumentEmbeddings not implemented for type: ${type}`);
  return { total: 0, updated: 0, errors: 0 };
}

// Single embedding request
export interface EmbeddingRequest {
  text: string;
  model?: string;
}
// Batch embedding request
export interface BatchEmbeddingRequest {
  texts: string[];
  model?: string;
  batchSize?: number;
}
// Sync request for database documents
export interface SyncRequest {
  type?: "cases" | "evidence" | "documents" | "all";
  limit?: number;
  forceRegenerate?: boolean;
}
// Generate single embedding
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body: EmbeddingRequest = await request.json();
    const { text, model } = body;

    if (!text || text.trim().length === 0) {
      return json(
        {
          success: false,
          error: "Text is required",
        },
        { status: 400 },
      );
    }
    // Try primary embedding service first, fallback to Qdrant
    let embedding: number[];
    try {
      embedding = await generateEmbedding(text, { model: model as "local" | "openai" });
    } catch (primaryError) {
      console.warn('Primary embedding failed, trying Qdrant fallback:', primaryError);
      embedding = await fetchEmbedding(text);
      if (embedding.length === 0) {
        throw new Error('Both primary and fallback embedding methods failed');
      }
    }

    return json({
      success: true,
      data: {
        embedding,
        dimensions: embedding.length,
        text: text.substring(0, 100) + (text.length > 100 ? "..." : ""),
      },
    });
  } catch (error: any) {
    console.error("Embedding generation error:", error);
    return json(
      {
        success: false,
        error: "Failed to generate embedding",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Batch embedding generation
export const PUT: RequestHandler = async ({ request, locals }) => {
  try {
    const body: BatchEmbeddingRequest = await request.json();
    const { texts, model, batchSize = 10 } = body;

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return json(
        {
          success: false,
          error: "Texts array is required",
        },
        { status: 400 },
      );
    }
    if (texts.length > 100) {
      return json(
        {
          success: false,
          error: "Maximum 100 texts per batch",
        },
        { status: 400 },
      );
    }
    const embeddings = await generateBatchEmbeddings(texts, {
      model: model as "local" | "openai",
    });

    return json({
      success: true,
      data: {
        embeddings,
        count: embeddings.length,
        dimensions: embeddings[0]?.length || 0,
      },
    });
  } catch (error: any) {
    console.error("Batch embedding generation error:", error);
    return json(
      {
        success: false,
        error: "Failed to generate batch embeddings",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Sync embeddings for existing database documents
export const PATCH: RequestHandler = async ({ request, locals }) => {
  try {
    const body: SyncRequest = await request.json();
    const { type = "all", limit = 100, forceRegenerate = false } = body;

    // Check if user has permission for bulk operations
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        {
          success: false,
          error: "Admin privileges required for bulk operations",
        },
        { status: 403 },
      );
    }
    const startTime = Date.now();
    let syncResult;

    switch (type) {
      case "cases":
        syncResult = await syncDocumentEmbeddings(
          "cases",
          limit,
          forceRegenerate,
        );
        break;
      case "evidence":
        syncResult = await syncDocumentEmbeddings(
          "evidence",
          limit,
          forceRegenerate,
        );
        break;
      case "documents":
        syncResult = await syncDocumentEmbeddings(
          "documents",
          limit,
          forceRegenerate,
        );
        break;
      case "all":
        const casesResult = await syncDocumentEmbeddings(
          "cases",
          limit,
          forceRegenerate,
        );
        const evidenceResult = await syncDocumentEmbeddings(
          "evidence",
          limit,
          forceRegenerate,
        );
        const documentsResult = await syncDocumentEmbeddings(
          "documents",
          limit,
          forceRegenerate,
        );

        syncResult = {
          total:
            casesResult.total + evidenceResult.total + documentsResult.total,
          updated:
            casesResult.updated +
            evidenceResult.updated +
            documentsResult.updated,
          errors:
            casesResult.errors + evidenceResult.errors + documentsResult.errors,
          details: {
            cases: casesResult,
            evidence: evidenceResult,
            documents: documentsResult,
          },
        };
        break;
      default:
        return json(
          {
            success: false,
            error: "Invalid sync type. Use: cases, evidence, documents, or all",
          },
          { status: 400 },
        );
    }
    return json({
      success: true,
      data: {
        ...syncResult,
        executionTime: Date.now() - startTime,
        type,
      },
    });
  } catch (error: any) {
    console.error("Embedding sync error:", error);
    return json(
      {
        success: false,
        error: "Failed to sync embeddings",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// Get embedding status and statistics
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    // This could be expanded to show embedding statistics
    // For now, return basic info
    return json({
      success: true,
      data: {
        info: "Embeddings API is operational",
        endpoints: {
          "POST /api/embeddings": "Generate single embedding",
          "PUT /api/embeddings": "Generate batch embeddings",
          "PATCH /api/embeddings": "Sync database embeddings (admin only)",
          "GET /api/embeddings": "Get API status",
        },
        limits: {
          maxBatchSize: 100,
          maxTextLength: 8000,
        },
      },
    });
  } catch (error: any) {
    console.error("Embeddings status error:", error);
    return json(
      {
        success: false,
        error: "Failed to get embeddings status",
      },
      { status: 500 },
    );
  }
};
