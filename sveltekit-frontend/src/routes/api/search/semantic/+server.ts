/// <reference types="vite/client" />

import type { RequestHandler } from './$types';

/**
 * SvelteKit 2 API Route: Semantic Search with Clustering Integration
 * POST /api/search/semantic
 */

import { json } from "@sveltejs/kit";
import { LegalDocumentSOM } from "$lib/services/som-clustering";
import { Redis } from "ioredis";
import { db } from "$lib/server/db";
import { sql, desc, and, eq } from "drizzle-orm";

// Initialize connections
const redis = new Redis({
  host: import.meta.env.REDIS_HOST || "localhost",
  port: parseInt(import.meta.env.REDIS_PORT || "6379"),
});

const qdrant = new QdrantClient({
  url: import.meta.env.QDRANT_URL || "http://localhost:6333",
});

// Gemma3 embedding service
async function getGemma3Embedding(text: string): Promise<number[]> {
  try {
    const response = await fetch("http://localhost:11434/api/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma3-legal:latest",
        prompt: text,
      }),
    });

    const data = await response.json();
    return data.embedding || [];
  } catch (error: any) {
    console.error("Gemma3 embedding error:", error);
    throw new Error("Failed to generate embedding");
  }
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const {
      query,
      clusterId,
      useKnowledge = true,
      limit = 20,
      threshold = 0.7,
    } = await request.json();

    if (!query || typeof query !== "string") {
      return json(
        {
          success: false,
          error: "Query string is required",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 400 },
      );
    }

    // Generate embedding for the query
    const queryEmbedding = await getGemma3Embedding(query);

    if (queryEmbedding.length === 0) {
      return json(
        {
          success: false,
          error: "Failed to generate query embedding",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 500 },
      );
    }

    // Multi-source semantic search
    const searchResults: DocumentCluster[] = [];
    const clusterInfo: string[] = [];

    // 1. PostgreSQL pgvector search
    try {
      const pgResults = await db
        .select({
          id: documents.id,
          title: documents.title,
          content: documents.content,
          embedding: documents.embedding,
          keywords: documents.keywords,
          similarity: sql<number>`1 - (${documents.embedding} <=> ${JSON.stringify(queryEmbedding)})`,
        })
        .from(documents)
        .where(
          and(
            sql`${documents.embedding} <=> ${JSON.stringify(queryEmbedding)} < ${1 - threshold}`,
            clusterId
              ? eq(
                  documents.keywords,
                  sql`jsonb_build_array(${clusterId})`,
                )
              : sql`true`,
          ),
        )
        .orderBy(
          desc(
            sql`1 - (${documents.embedding} <=> ${JSON.stringify(queryEmbedding)})`,
          ),
        )
        .limit(limit);

      // Convert to DocumentCluster format
      for (const result of pgResults) {
        searchResults.push({
          id: result.id.toString(),
          centroid: result.embedding as number[] || [],
          documents: [result.id.toString()],
          size: 1,
          documentId: result.id,
          similarity: result.similarity,
          embedding: result.embedding as number[],
          metadata: {
            title: result.title,
            content: result.content?.substring(0, 500),
            source: "postgresql",
            keywords: result.keywords || [],
          },
        });
      }
    } catch (pgError) {
      console.warn("PostgreSQL search failed:", pgError);
    }

    // 2. Qdrant vector search (if available)
    try {
      const qdrantResults = await qdrant.search("legal_documents", {
        vector: queryEmbedding,
        limit: limit,
        score_threshold: threshold,
        with_payload: true,
        filter: clusterId
          ? {
              must: [{ key: "clusterId", match: { value: clusterId } }],
            }
          : undefined,
      });

      // Merge with existing results (avoid duplicates)
      const existingIds = new Set(searchResults.map((r) => r.documentId));

      for (const result of (qdrantResults as any) || []) {
        if (!existingIds.has(result.id.toString())) {
          searchResults.push({
            id: result.id.toString(),
            centroid: [],
            documents: [result.id.toString()],
            size: 1,
            documentId: result.id.toString(),
            similarity: result.score || 0,
            embedding: [], // Not needed for results
            metadata: {
              ...result.payload,
              source: "qdrant",
            },
          });
        }
      }
    } catch (qdrantError) {
      console.warn("Qdrant search failed:", qdrantError);
    }

    // 3. Cluster-aware reranking (if clustering is enabled)
    if (useKnowledge) {
      try {
        // Load SOM model for cluster analysis
        const som = await LegalDocumentSOM.loadFromRedis(redis);
        if (som) {
          const queryCluster = await som.cluster(queryEmbedding);
          clusterInfo.push(
            `SOM cluster: (${queryCluster.x}, ${queryCluster.y}) confidence: ${queryCluster.confidence.toFixed(3)}`,
          );

          // Boost results from similar SOM clusters
          for (const result of searchResults) {
            if (result.embedding.length > 0) {
              const resultCluster = await som.cluster(result.embedding);
              const clusterDistance = Math.sqrt(
                Math.pow(queryCluster.x - resultCluster.x, 2) +
                  Math.pow(queryCluster.y - resultCluster.y, 2),
              );

              // Boost similarity for nearby clusters
              if (clusterDistance <= 2) {
                result.similarity *= 1 + (2 - clusterDistance) * 0.1;
                result.metadata.somBoost = true;
                result.metadata.clusterDistance = clusterDistance;
              }
            }
          }
        }

        // Load K-Means model for cluster prediction
        const kmeans = await LegalKMeansClusterer.loadFromRedis(redis);
        if (kmeans) {
          const predictedCluster = await kmeans.predict(queryEmbedding);
          clusterInfo.push(`K-Means cluster: ${predictedCluster}`);

          // Boost results from the same K-Means cluster
          for (const result of searchResults) {
            if (result.metadata.kmeansCluster === predictedCluster) {
              result.similarity *= 1.15; // 15% boost for same cluster
              result.metadata.kmeansBoost = true;
            }
          }
        }
      } catch (clusterError) {
        console.warn("Clustering analysis failed:", clusterError);
      }
    }

    // 4. Legal context enhancement
    const enhancedResults = await enhanceLegalContext(searchResults, query);

    // 5. Final ranking and filtering
    const finalResults = enhancedResults
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map((result) => ({
        ...result,
        similarity: Math.min(result.similarity, 1), // Clamp to [0, 1]
        metadata: {
          ...result.metadata,
          relevanceScore: result.similarity,
          searchQuery: query,
          enhancedAt: new Date().toISOString(),
        },
      }));

    // Get cluster insights for the query
    const clusterInsights = await getClusterInsights(
      queryEmbedding,
      clusterInfo,
    );

    return json({
      success: true,
      data: {
        results: finalResults,
        clusters: clusterInfo,
        insights: clusterInsights,
        searchMetadata: {
          queryEmbeddingLength: queryEmbedding.length,
          totalResults: finalResults.length,
          sourcesUsed: ["postgresql", "qdrant"],
          clusteringEnabled: useKnowledge,
          threshold,
        },
      },
      metadata: {
        timestamp: new Date().toISOString(),
        processingTime: Date.now() - startTime,
        confidence: finalResults.length > 0 ? finalResults[0].similarity : 0,
      },
    });
  } catch (error: any) {
    console.error("Semantic search error:", error);

    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Search failed",
        metadata: {
          timestamp: new Date().toISOString(),
          processingTime: Date.now() - startTime,
        },
      },
      { status: 500 },
    );
  }
};

/**
 * Enhance search results with legal context using Gemma3
 */
async function enhanceLegalContext(
  results: DocumentCluster[],
  query: string,
): Promise<DocumentCluster[]> {
  const legalKeywords = [
    "contract",
    "liability",
    "compliance",
    "regulation",
    "statute",
    "precedent",
    "jurisdiction",
    "defendant",
    "plaintiff",
    "evidence",
    "testimony",
    "warrant",
    "subpoena",
    "discovery",
    "motion",
  ];

  const queryWords = query.toLowerCase().split(/\\s+/);
  const legalScore =
    queryWords.filter((word) =>
      legalKeywords.some((keyword) => word.includes(keyword)),
    ).length / queryWords.length;

  return results.map((result) => {
    const text = (result.metadata.content || "").toLowerCase();
    const textLegalScore =
      legalKeywords.filter((keyword) => text.includes(keyword)).length /
      legalKeywords.length;

    // Boost legal documents
    if (textLegalScore > 0.1 || legalScore > 0.1) {
      result.similarity *= 1 + (textLegalScore + legalScore) * 0.2;
      result.metadata.legalRelevance = textLegalScore;
      result.metadata.queryLegalScore = legalScore;
    }

    return result;
  });
}

/**
 * Generate cluster insights for the search query
 */
async function getClusterInsights(
  queryEmbedding: number[],
  clusterInfo: string[],
): Promise<string[]> {
  const insights: string[] = [];

  try {
    // Get related legal topics from clustering
    const cachedInsights = await redis.get(
      `insights:${JSON.stringify(queryEmbedding).slice(0, 50)}`,
    );
    if (cachedInsights) {
      return JSON.parse(cachedInsights);
    }

    // Generate insights based on cluster patterns
    if (clusterInfo.length > 0) {
      insights.push("Query mapped to trained document clusters");
      insights.push("Results enhanced with cluster-based relevance scoring");

      for (const info of clusterInfo) {
        insights.push(info);
      }
    }

    // Cache insights for 5 minutes
    await redis.setex(
      `insights:${JSON.stringify(queryEmbedding).slice(0, 50)}`,
      300,
      JSON.stringify(insights),
    );
  } catch (error: any) {
    console.warn("Failed to generate cluster insights:", error);
    insights.push("Cluster analysis unavailable");
  }

  return insights;
}

// GET endpoint for cluster status and available clusters
export const GET: RequestHandler = async () => {
  try {
    const clusterStatus = {
      som: {
        trained: false,
        accuracy: 0,
        lastTraining: null,
      },
      kmeans: {
        clusters: 0,
        silhouetteScore: 0,
        lastClustering: null,
      },
      queue: {
        pending: 0,
        processing: 0,
      },
    };

    // Check SOM status
    const somModel = await redis.get("som:model");
    if (somModel) {
      const somData = JSON.parse(somModel);
      clusterStatus.som.trained = somData.trained || false;
      clusterStatus.som.lastTraining = somData.savedAt;
    }

    // Check K-Means status
    const kmeansModel = await redis.get("kmeans:model");
    if (kmeansModel) {
      const kmeansData = JSON.parse(kmeansModel);
      clusterStatus.kmeans.clusters = kmeansData.centroids?.length || 0;
      clusterStatus.kmeans.silhouetteScore = kmeansData.silhouetteScore || 0;
      clusterStatus.kmeans.lastClustering = kmeansData.savedAt;
    }

    // Check queue status
    const queueKeys = await redis.keys("*:training:*");
    clusterStatus.queue.pending = queueKeys.filter((key) =>
      key.includes("queued"),
    ).length;
    clusterStatus.queue.processing = queueKeys.filter((key) =>
      key.includes("processing"),
    ).length;

    return json({
      success: true,
      data: clusterStatus,
      metadata: {
        timestamp: new Date().toISOString(),
        processingTime: 50,
      },
    });
  } catch (error: any) {
    console.error("Cluster status error:", error);

    return json(
      {
        success: false,
        error: "Failed to retrieve cluster status",
        metadata: {
          timestamp: new Date().toISOString(),
          processingTime: 0,
        },
      },
      { status: 500 },
    );
  }
};
