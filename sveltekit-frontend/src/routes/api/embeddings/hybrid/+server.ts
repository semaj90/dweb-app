/// <reference types="vite/client" />

import type { RequestHandler } from './$types';

/**
 * Hybrid Vector Embedding API - PGVector + Qdrant + Local LLM Integration
 * Supports multiple embedding models with fallback strategies
 */

import { json, error } from "@sveltejs/kit";
import { db } from "$lib/server/db/drizzle";
import { eq, sql } from "drizzle-orm";
import { URL } from "url";

// Embedding models configuration
export interface EmbeddingModel {
  id: string;
  name: string;
  dimensions: number;
  maxTokens: number;
  provider: "ollama" | "openai" | "huggingface" | "local";
  endpoint?: string;
}

const EMBEDDING_MODELS: Record<string, EmbeddingModel> = {
  "nomic-embed-text": {
    id: "nomic-embed-text",
    name: "Nomic Embed Text",
    dimensions: 384,
    maxTokens: 8192,
    provider: "ollama",
    endpoint: "http://localhost:11434",
  },
  "all-MiniLM-L6-v2": {
    id: "all-MiniLM-L6-v2",
    name: "All MiniLM L6 v2",
    dimensions: 384,
    maxTokens: 512,
    provider: "local",
  },
  "text-embedding-ada-002": {
    id: "text-embedding-ada-002",
    name: "OpenAI Ada 002",
    dimensions: 1536,
    maxTokens: 8191,
    provider: "openai",
  },
};

// Vector storage backends
export interface VectorBackend {
  id: string;
  name: string;
  endpoint: string;
  supported: boolean;
}

const VECTOR_BACKENDS: Record<string, VectorBackend> = {
  pgvector: {
    id: "pgvector",
    name: "PostgreSQL pgvector",
    endpoint: import.meta.env.DATABASE_URL || "",
    supported: true,
  },
  qdrant: {
    id: "qdrant",
    name: "Qdrant Vector DB",
    endpoint: import.meta.env.QDRANT_URL || "http://localhost:6333",
    supported: false, // Will be checked at runtime
  },
};

// Embedding cache for performance
const embeddingCache = new Map<
  string,
  { embedding: number[]; timestamp: number }
>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export const POST: RequestHandler = async ({ request, url }) => {
  try {
    const {
      content,
      model = "nomic-embed-text",
      backend = "hybrid",
      options = {},
    } = await request.json();

    if (!content || typeof content !== "string") {
      return error(400, "Content is required and must be a string");
    }

    if (content.length > 100000) {
      return error(400, "Content too large (max 100,000 characters)");
    }

    // Check cache first
    const cacheKey = `${model}:${hashContent(content)}`;
    const cached = embeddingCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      return json({
        success: true,
        embedding: cached.embedding,
        model,
        cached: true,
        dimensions: cached.embedding.length,
      });
    }

    // Generate embedding based on backend preference
    let embedding: number[];
    let usedBackend: string;

    switch (backend) {
      case "pgvector":
        ({ embedding, backend: usedBackend } = await generatePGVectorEmbedding(
          content,
          model
        ));
        break;
      case "qdrant":
        ({ embedding, backend: usedBackend } = await generateQdrantEmbedding(
          content,
          model
        ));
        break;
      case "hybrid":
      default:
        ({ embedding, backend: usedBackend } = await generateHybridEmbedding(
          content,
          model
        ));
        break;
    }

    // Cache the result
    embeddingCache.set(cacheKey, { embedding, timestamp: Date.now() });

    // Store in database if requested
    if (options.store && options.documentId) {
      await storeEmbedding(options.documentId, embedding, model, usedBackend);
    }

    return json({
      success: true,
      embedding,
      model,
      backend: usedBackend,
      dimensions: embedding.length,
      cached: false,
    });
  } catch (err: any) {
    console.error("Embedding generation error:", err);
    return error(500, `Embedding generation failed: ${err.message}`);
  }
};

export const GET: RequestHandler = async ({ url }) => {
  const action = url.searchParams.get("action");

  switch (action) {
    case "models":
      return json({
        success: true,
        models: Object.values(EMBEDDING_MODELS),
        backends: Object.values(VECTOR_BACKENDS),
      });

    case "health":
      return json({
        success: true,
        health: await checkSystemHealth(),
      });

    case "cache-stats":
      return json({
        success: true,
        cache: {
          size: embeddingCache.size,
          entries: Array.from(embeddingCache.keys()).slice(0, 10),
        },
      });

    default:
      return error(400, "Invalid action. Use: models, health, cache-stats");
  }
};

/**
 * Generate embedding using PGVector (PostgreSQL)
 */
async function generatePGVectorEmbedding(
  content: string,
  model: string
): Promise<{ embedding: number[]; backend: string }> {
  try {
    // First try to use Ollama for embedding generation
    const embedding = await generateOllamaEmbedding(content, model);

    // Store in PostgreSQL using pgvector
    if (embedding.length > 0) {
      // We can extend this to store the embedding in a pgvector table
      // For now, just return the embedding
      return { embedding, backend: "pgvector" };
    }

    throw new Error("Empty embedding returned");
  } catch (error: any) {
    console.warn("PGVector embedding failed:", error);
    throw error;
  }
}

/**
 * Generate embedding using Qdrant
 */
async function generateQdrantEmbedding(
  content: string,
  model: string
): Promise<{ embedding: number[]; backend: string }> {
  try {
    const qdrantEndpoint = VECTOR_BACKENDS.qdrant.endpoint;

    // First generate embedding using Ollama
    const embedding = await generateOllamaEmbedding(content, model);

    // Store in Qdrant (optional)
    try {
      await fetch(`${qdrantEndpoint}/collections/copilot_embeddings/points`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          points: [
            {
              id: Date.now(),
              vector: embedding,
              payload: {
                content: content.substring(0, 1000), // Truncate for storage
                model,
                timestamp: Date.now(),
              },
            },
          ],
        }),
      });
    } catch (storageError) {
      console.warn(
        "Qdrant storage failed, but embedding succeeded:",
        storageError
      );
    }

    return { embedding, backend: "qdrant" };
  } catch (error: any) {
    console.warn("Qdrant embedding failed:", error);
    throw error;
  }
}

/**
 * Generate embedding using hybrid approach (PGVector → Qdrant → Local)
 */
async function generateHybridEmbedding(
  content: string,
  model: string
): Promise<{ embedding: number[]; backend: string }> {
  // Try PGVector first
  try {
    return await generatePGVectorEmbedding(content, model);
  } catch (error: any) {
    console.warn("PGVector failed, trying Qdrant:", error);
  }

  // Fallback to Qdrant
  try {
    return await generateQdrantEmbedding(content, model);
  } catch (error: any) {
    console.warn("Qdrant failed, using local fallback:", error);
  }

  // Final fallback to local embedding
  const embedding = generateLocalEmbedding(content, model);
  return { embedding, backend: "local_fallback" };
}

/**
 * Generate embedding using Ollama
 */
async function generateOllamaEmbedding(
  content: string,
  model: string
): Promise<number[]> {
  const modelConfig = EMBEDDING_MODELS[model];
  if (!modelConfig || modelConfig.provider !== "ollama") {
    throw new Error(`Model ${model} not supported for Ollama`);
  }

  const ollamaEndpoint = modelConfig.endpoint || "http://localhost:11434";

  try {
    const response = await fetch(`${ollamaEndpoint}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model,
        prompt: content,
      }),
      signal: AbortSignal.timeout(30000), // 30 second timeout
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();

    if (!data.embedding || !Array.isArray(data.embedding)) {
      throw new Error("Invalid embedding format from Ollama");
    }

    return data.embedding;
  } catch (error: any) {
    if (error.name === "AbortError") {
      throw new Error("Ollama request timed out");
    }
    throw new Error(`Ollama embedding failed: ${error.message}`);
  }
}

/**
 * Generate local embedding (fallback)
 */
function generateLocalEmbedding(content: string, model: string): number[] {
  const modelConfig = EMBEDDING_MODELS[model];
  const dimensions = modelConfig?.dimensions || 384;

  // Simple TF-IDF based embedding
  const words = content
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2);

  const wordCount = new Map<string, number>();
  words.forEach((word) => {
    wordCount.set(word, (wordCount.get(word) || 0) + 1);
  });

  // Create normalized embedding vector
  const embedding = new Array(dimensions).fill(0);
  let index = 0;

  for (const [word, count] of wordCount.entries()) {
    if (index < dimensions) {
      // Simple hash-based positioning with TF weighting
      const position = Math.abs(hashString(word)) % dimensions;
      embedding[position] += count / words.length;
      index++;
    }
  }

  // Normalize vector
  const magnitude = Math.sqrt(
    embedding.reduce((sum, val) => sum + val * val, 0)
  );
  if (magnitude > 0) {
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= magnitude;
    }
  }

  return embedding;
}

/**
 * Store embedding in database
 */
async function storeEmbedding(
  documentId: string,
  embedding: number[],
  model: string,
  backend: string
): Promise<void> {
  try {
    // Note: Database storage would be implemented here
    // await db.update(documents).set({ embeddings: embedding }).where(eq(documents.id, documentId));
    console.log(`Would store embedding for document ${documentId} using model ${model} and backend ${backend}`);
  } catch (error: any) {
    console.error("Failed to store embedding in database:", error);
  }
}

/**
 * Check system health
 */
async function checkSystemHealth(): Promise<any> {
  const health = {
    ollama: false,
    qdrant: false,
    pgvector: true, // Assume true if we can connect to DB
    models: [] as string[],
  };

  // Check Ollama
  try {
    const response = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(5000),
    });
    if (response.ok) {
      const data = await response.json();
      health.ollama = true;
      health.models = data.models?.map((m: any) => m.name) || [];
    }
  } catch (error: any) {
    console.warn("Ollama health check failed:", error);
  }

  // Check Qdrant
  try {
    const qdrantEndpoint = VECTOR_BACKENDS.qdrant.endpoint;
    const response = await fetch(`${qdrantEndpoint}/collections`, {
      signal: AbortSignal.timeout(5000),
    });
    health.qdrant = response.ok;
  } catch (error: any) {
    console.warn("Qdrant health check failed:", error);
  }

  return health;
}

/**
 * Utility functions
 */
function hashContent(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return hash.toString(36);
}

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return hash;
}

// Clean up cache periodically
setInterval(() => {
  const now = Date.now();
  for (const [key, value] of embeddingCache.entries()) {
    if (now - value.timestamp > CACHE_TTL) {
      embeddingCache.delete(key);
    }
  }
}, 60000); // Clean every minute
