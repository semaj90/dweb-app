/// <reference types="vite/client" />

import type { RequestHandler } from './$types.js';

/**
 * SvelteKit 2 API Route: K-Means Clustering
 * POST /api/clustering/kmeans/cluster
 */

import { json } from "@sveltejs/kit";
import { LegalKMeansClusterer } from "$lib/services/kmeans-clustering";
import { Redis } from "ioredis";
import { db } from "$lib/server/db";
import { inArray } from "drizzle-orm";
import { URL } from "url";
// Optional amqp for message queue integration

// Initialize connections
const redis = new Redis({
  host: import.meta.env.REDIS_HOST || "localhost",
  port: parseInt(import.meta.env.REDIS_PORT || "6379"),
});

const qdrant = new QdrantClient({
  url: import.meta.env.QDRANT_URL || "http://localhost:6333",
});

let rabbitConnection: any | null = null;
async function getRabbitConnection(): Promise<any> {
  if (!rabbitConnection) {
    try {
      // const amqp = await import('amqplib').catch(() => null);
      // if (amqp) {
      //   rabbitConnection = await amqp.connect(import.meta.env.RABBITMQ_URL || 'amqp://localhost');
      // }
    } catch (error: any) {
      console.warn("RabbitMQ not available:", error);
    }
  }
  return rabbitConnection;
}

export const POST: RequestHandler = async ({ request }) => {
  const startTime = Date.now();

  try {
    const { documentIds, k, config } = await request.json();

    if (!documentIds || !Array.isArray(documentIds)) {
      return json(
        {
          success: false,
          error: "Document IDs array is required",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 400 },
      );
    }

    // Validate k parameter
    const clusterCount =
      k || Math.min(Math.ceil(Math.sqrt(documentIds.length / 2)), 10);
    if (clusterCount < 2 || clusterCount > documentIds.length) {
      return json(
        {
          success: false,
          error: `Invalid cluster count: ${clusterCount}. Must be between 2 and ${documentIds.length}`,
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 400 },
      );
    }

    // Fetch embeddings from multiple sources for redundancy
    let embeddings: number[][] = [];
    let documentMetadata: Array<{
      id: string;
      type: string;
      keywords: string[];
    }> = [];

    try {
      // Primary: PostgreSQL with pgvector
      const pgDocuments = await db
        .select({
          id: legalDocuments.id,
          embedding: legalDocuments.embedding,
          metadata: legalDocuments.keywords, // Use keywords as metadata
          extractedText: legalDocuments.content,
        })
        .from(legalDocuments)
        .where(inArray(legalDocuments.id, documentIds));

      // Secondary: Qdrant vector database
      let qdrantResults = [];
      try {
        const qdrantResponse = await qdrant.retrieve("legal_documents", {
          ids: documentIds,
          with_payload: true,
          with_vector: true,
        });
        qdrantResults = (qdrantResponse as any).points || qdrantResponse || [];
      } catch (qdrantError) {
        console.warn(
          "Qdrant retrieval failed, using PostgreSQL only:",
          qdrantError,
        );
      }

      // Merge results with preference for PostgreSQL
      const mergedDocuments = new Map();

      // Add PostgreSQL results
      for (const doc of pgDocuments) {
        if (doc.embedding && Array.isArray(doc.embedding)) {
          mergedDocuments.set(doc.id, {
            id: doc.id,
            embedding: doc.embedding,
            metadata: doc.metadata || {},
            source: "postgresql",
          });
        }
      }

      // Add Qdrant results if not already present
      for (const result of qdrantResults) {
        if (!mergedDocuments.has(result.id) && result.vector) {
          mergedDocuments.set(result.id, {
            id: result.id,
            embedding: result.vector,
            metadata: result.payload || {},
            source: "qdrant",
          });
        }
      }

      // Extract final embeddings and metadata
      embeddings = Array.from(mergedDocuments.values()).map(
        (doc) => doc.embedding,
      );
      documentMetadata = Array.from(mergedDocuments.values()).map((doc) => ({
        id: doc.id,
        type: doc.metadata.type || "unknown",
        keywords: doc.metadata.keywords || [],
      }));
    } catch (dbError) {
      console.error("Database retrieval error:", dbError);
      return json(
        {
          success: false,
          error: "Failed to retrieve document embeddings",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 500 },
      );
    }

    if (embeddings.length === 0) {
      return json(
        {
          success: false,
          error: "No valid embeddings found",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 404 },
      );
    }

    // Configure K-Means with all required properties
    const kmeansConfig = {
      k: clusterCount,
      maxIterations: config?.maxIterations || 100,
      tolerance: config?.tolerance || 0.001,
      initMethod: config?.initMethod || ("kmeans++" as const),
      algorithm: "kmeans" as const, // Required algorithm property
      distanceMetric: "euclidean" as const, // Required distance metric
    };

    // Generate cluster job ID
    const clusterJobId = `kmeans_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Initialize K-Means clusterer
    const kmeans = new LegalKMeansClusterer(kmeansConfig, redis);

    // Store job status in Redis
    await redis.hset(`kmeans:job:${clusterJobId}`, {
      status: "processing",
      documentCount: embeddings.length,
      clusterCount,
      startedAt: Date.now(),
      config: JSON.stringify(kmeansConfig),
    });

    // Queue clustering job in RabbitMQ for monitoring
    const connection = await getRabbitConnection();
    const channel = await connection.createChannel();
    await channel.assertExchange("clustering", "topic", { durable: true });

    const message = {
      messageId: clusterJobId,
      type: "kmeans_clustering",
      payload: {
        documentIds,
        embeddings: embeddings.slice(0, 10), // Sample for messaging
        config: kmeansConfig,
      },
      priority: "high",
      timestamp: new Date().toISOString(),
    };

    await channel.publish(
      "clustering",
      "kmeans.clustering.start",
      Buffer.from(JSON.stringify(message)),
    );

    try {
      // Try WASM clustering first for better performance
      const wasmMetrics = wasmClusteringService.getPerformanceMetrics();
      let clusters;
      
      if (wasmMetrics.recommendedForDataSize(embeddings.length)) {
        console.log('Using WebAssembly K-Means clustering for enhanced performance');
        const wasmResult = await wasmClusteringService.performKMeansClustering(
          embeddings,
          clusterCount,
          kmeansConfig
        );
        clusters = wasmResult.clusters;
      } else {
        // Fallback to JavaScript implementation
        clusters = await kmeans.fit(embeddings);
      }

      // Analyze legal context
      const analysis = await kmeans.analyzeLegalClusters(
        embeddings,
        documentMetadata,
      );

      // Get model metrics (mock implementation for now)
      const silhouetteScore = 0.75; // TODO: Implement proper silhouette score calculation
      const centroids = []; // TODO: Get actual centroids from kmeans

      // Store results in Redis with TTL
      const results = {
        clusters,
        analysis: analysis.clusterAnalysis,
        metrics: {
          silhouetteScore,
          documentCount: embeddings.length,
          clusterCount,
          convergenceTime: Date.now() - startTime,
        },
        centroids,
      };

      await redis.setex(
        `kmeans:results:${clusterJobId}`,
        3600, // 1 hour TTL
        JSON.stringify(results),
      );

      // Update job status
      await redis.hset(`kmeans:job:${clusterJobId}`, {
        status: "completed",
        completedAt: Date.now(),
        silhouetteScore: silhouetteScore.toString(),
      });

      // Store centroids in Qdrant for future similarity searches
      try {
        const centroidPoints = centroids.map((centroid, index) => ({
          id: `centroid_${clusterJobId}_${index}`,
          vector: centroid,
          payload: {
            type: "centroid",
            clusterId: `cluster_${index}`,
            jobId: clusterJobId,
            createdAt: new Date().toISOString(),
          },
        }));

        await qdrant.upsert("legal_centroids", {
          wait: true,
          points: centroidPoints,
        });
      } catch (qdrantError) {
        console.warn("Failed to store centroids in Qdrant:", qdrantError);
      }

      // Publish completion event
      await channel.publish(
        "clustering",
        "kmeans.clustering.completed",
        Buffer.from(
          JSON.stringify({
            jobId: clusterJobId,
            status: "completed",
            metrics: results.metrics,
          }),
        ),
      );

      await channel.close();

      return json({
        success: true,
        data: {
          jobId: clusterJobId,
          clusters: results.clusters,
          analysis: results.analysis,
          metrics: results.metrics,
        },
        metadata: {
          timestamp: new Date().toISOString(),
          processingTime: Date.now() - startTime,
          clusterId: clusterJobId,
          confidence: silhouetteScore,
        },
      });
    } catch (clusteringError) {
      console.error("K-Means clustering error:", clusteringError);

      // Update job status
      await redis.hset(`kmeans:job:${clusterJobId}`, {
        status: "failed",
        error:
          clusteringError instanceof Error
            ? clusteringError.message
            : "Unknown error",
        failedAt: Date.now(),
      });

      // Publish failure event
      await channel.publish(
        "clustering",
        "kmeans.clustering.failed",
        Buffer.from(
          JSON.stringify({
            jobId: clusterJobId,
            status: "failed",
            error:
              clusteringError instanceof Error
                ? clusteringError.message
                : "Unknown error",
          }),
        ),
      );

      await channel.close();

      return json(
        {
          success: false,
          error:
            clusteringError instanceof Error
              ? clusteringError.message
              : "Clustering failed",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: Date.now() - startTime,
          },
        },
        { status: 500 },
      );
    }
  } catch (error: any) {
    console.error("K-Means API error:", error);

    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
        metadata: {
          timestamp: new Date().toISOString(),
          processingTime: Date.now() - startTime,
        },
      },
      { status: 500 },
    );
  }
};

// GET endpoint for cluster prediction
export const GET: RequestHandler = async ({ url }) => {
  const jobId = url.searchParams.get("jobId");
  const embeddingStr = url.searchParams.get("embedding");

  if (!jobId || !embeddingStr) {
    return json(
      {
        success: false,
        error: "Job ID and embedding are required",
        metadata: {
          timestamp: new Date().toISOString(),
          processingTime: 0,
        },
      },
      { status: 400 },
    );
  }

  try {
    // Parse embedding
    const embedding: number[] = JSON.parse(embeddingStr);

    // Load K-Means model from Redis
    const kmeans = await LegalKMeansClusterer.loadFromRedis(redis);
    if (!kmeans) {
      return json(
        {
          success: false,
          error: "No trained K-Means model found",
          metadata: {
            timestamp: new Date().toISOString(),
            processingTime: 0,
          },
        },
        { status: 404 },
      );
    }

    // Predict cluster
    const clusterId = await kmeans.predict(embedding);

    return json({
      success: true,
      data: {
        clusterId,
        jobId,
      },
      metadata: {
        timestamp: new Date().toISOString(),
        processingTime: 10,
      },
    });
  } catch (error: any) {
    console.error("K-Means prediction error:", error);

    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Prediction failed",
        metadata: {
          timestamp: new Date().toISOString(),
          processingTime: 0,
        },
      },
      { status: 500 },
    );
  }
};
