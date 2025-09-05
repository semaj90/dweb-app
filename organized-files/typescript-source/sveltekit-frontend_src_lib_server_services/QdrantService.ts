// @ts-nocheck
// QdrantService.ts - Production Implementation
// Fixed: 384-dimensional vectors for nomic-embed-text

import { QdrantClient } from "@qdrant/js-client-rest";
import { env } from "../../env/dynamic/private";
import { logger } from "../logger";
import type {
  VectorSearchResult,
  DocumentVector,
  SearchOptions,
  CollectionInfo,
  BatchUpsertResult,
} from "$lib/types";

export class QdrantService {
  private client: QdrantClient;
  private readonly VECTOR_SIZE = 384; // Fixed for nomic-embed-text
  private readonly DEFAULT_COLLECTION = "legal_documents";
  private readonly COLLECTIONS = {
    documents: "legal_documents",
    cases: "case_embeddings",
    evidence: "evidence_vectors",
  };

  constructor() {
    const qdrantUrl = env.QDRANT_URL || "http://localhost:6333";
    this.client = new QdrantClient({
      url: qdrantUrl,
      timeout: 30000,
    });
    logger.info("QdrantService initialized", {
      url: qdrantUrl,
      vectorSize: this.VECTOR_SIZE,
    });
  }

  /**
   * Initialize collections with proper 384-dim configuration
   */
  async initializeCollections(): Promise<void> {
    try {
      for (const [key, collectionName] of Object.entries(this.COLLECTIONS)) {
        const exists = await this.collectionExists(collectionName);

        if (!exists) {
          await this.createCollection(collectionName, {
            vectorSize: this.VECTOR_SIZE,
            distance: "Cosine",
            onDisk: false,
            shardNumber: key === "documents" ? 2 : 1,
            replicationFactor: 1,
            optimizersConfig: {
              indexingThreshold: 20000,
              memmapThreshold: 50000,
              maxOptimizationThreads: 2,
            },
            hnswConfig: {
              m: 16,
              efConstruct: 200,
              fullScanThreshold: 10000,
              maxIndexingThreads: 4,
            },
          });
          logger.info(`Created collection: ${collectionName}`);
        } else {
          // Verify dimensions
          const info = await this.getCollectionInfo(collectionName);
          if (info.config?.params?.vectors?.size !== this.VECTOR_SIZE) {
            logger.error(
              `Collection ${collectionName} has wrong dimensions: ${info.config?.params?.vectors?.size}`
            );
            throw new Error(`Vector dimension mismatch in ${collectionName}`);
          }
        }
      }
    } catch (error) {
      logger.error("Failed to initialize collections", error);
      throw error;
    }
  }

  /**
   * Store document with vector embedding
   */
  async storeDocument(document: DocumentVector): Promise<string> {
    try {
      // Validate vector dimensions
      if (document.vector.length !== this.VECTOR_SIZE) {
        throw new Error(
          `Invalid vector size: expected ${this.VECTOR_SIZE}, got ${document.vector.length}`
        );
      }

      const point = {
        id: document.id || crypto.randomUUID(),
        vector: document.vector,
        payload: {
          content: document.content,
          title: document.title || "",
          type: document.type || "document",
          metadata: document.metadata || {},
          created_at: new Date().toISOString(),
          case_id: document.case_id,
          relevance_score: document.relevance_score || 1.0,
        },
      };

      await this.client.upsert(this.DEFAULT_COLLECTION, {
        points: [point],
        wait: true,
      });

      logger.info("Document stored in Qdrant", { id: point.id });
      return point.id.toString();
    } catch (error) {
      logger.error("Failed to store document", error);
      throw error;
    }
  }

  /**
   * Batch store multiple documents
   */
  async batchStoreDocuments(
    documents: DocumentVector[]
  ): Promise<BatchUpsertResult> {
    try {
      // Validate all vectors
      const invalidDocs = documents.filter(
        (d) => d.vector.length !== this.VECTOR_SIZE
      );
      if (invalidDocs.length > 0) {
        throw new Error(
          `${invalidDocs.length} documents have invalid vector dimensions`
        );
      }

      const points = documents.map((doc) => ({
        id: doc.id || crypto.randomUUID(),
        vector: doc.vector,
        payload: {
          content: doc.content,
          title: doc.title || "",
          type: doc.type || "document",
          metadata: doc.metadata || {},
          created_at: new Date().toISOString(),
          case_id: doc.case_id,
          relevance_score: doc.relevance_score || 1.0,
        },
      }));

      const result = await this.client.upsert(this.DEFAULT_COLLECTION, {
        points,
        wait: true,
      });

      logger.info(`Batch stored ${points.length} documents`);
      return {
        operation_id: Date.now(),
        status: "completed" as const,
        result: points.map((p) => p.id),
        successful: true,
      };
    } catch (error) {
      logger.error("Failed to batch store documents", error);
      throw error;
    }
  }

  /**
   * Search for similar documents
   */
  async searchSimilar(
    queryVector: number[],
    options: SearchOptions = {}
  ): Promise<VectorSearchResult[]> {
    try {
      // Validate query vector
      if (queryVector.length !== this.VECTOR_SIZE) {
        throw new Error(
          `Invalid query vector size: expected ${this.VECTOR_SIZE}, got ${queryVector.length}`
        );
      }

      const {
        limit = 10,
        threshold = 0.7,
        filter = {},
        collection = this.DEFAULT_COLLECTION,
        includePayload = true,
        includeVector = false,
      } = options;

      // Build Qdrant filter
      const qdrantFilter: any = {};
      if (filter.case_id) {
        qdrantFilter.must = [
          { key: "case_id", match: { value: filter.case_id } },
        ];
      }
      if (filter.type) {
        qdrantFilter.must = qdrantFilter.must || [];
        qdrantFilter.must.push({ key: "type", match: { value: filter.type } });
      }
      if (filter.date_range) {
        qdrantFilter.must = qdrantFilter.must || [];
        qdrantFilter.must.push({
          key: "created_at",
          range: {
            gte: filter.date_range.start,
            lte: filter.date_range.end,
          },
        });
      }

      const searchParams: any = {
        vector: queryVector,
        limit,
        with_payload: includePayload,
        with_vector: includeVector,
        score_threshold: threshold,
      };

      if (Object.keys(qdrantFilter).length > 0) {
        searchParams.filter = qdrantFilter;
      }

      const results = await this.client.search(collection, searchParams);

      return results.map((result) => ({
        id: result.id.toString(),
        score: result.score,
        payload: result.payload || {},
        vector:
          includeVector && Array.isArray(result.vector)
            ? (result.vector as number[])
            : undefined,
        similarity: result.score, // Cosine similarity
        content: result.payload ? String(result.payload.content || "") : "",
        title: result.payload ? String(result.payload.title || "") : "",
        type: result.payload
          ? String(result.payload.type || "document")
          : "document",
        metadata: result.payload?.metadata || {},
        case_id: result.payload ? String(result.payload.case_id || "") : "",
        created_at: result.payload?.created_at,
        relevance_score: result.payload
          ? Number(result.payload.relevance_score || result.score)
          : result.score,
      }));
    } catch (error) {
      logger.error("Failed to search similar documents", error);
      throw error;
    }
  }

  /**
   * Get document by ID
   */
  async getDocument(
    id: string,
    collection: string = this.DEFAULT_COLLECTION
  ): Promise<DocumentVector | null> {
    try {
      const results = await this.client.retrieve(collection, {
        ids: [id],
        with_payload: true,
        with_vector: true,
      });

      if (results.length === 0) {
        return null;
      }

      const point = results[0];
      return {
        id: point.id.toString(),
        vector: point.vector as number[],
        payload: point.payload || {},
        content: point.payload ? String(point.payload.content || "") : "",
        title: point.payload ? String(point.payload.title || "") : "",
        type: point.payload
          ? String(point.payload.type || "document")
          : "document",
        metadata: point.payload?.metadata || {},
        case_id: point.payload
          ? String(point.payload.case_id || "")
          : undefined,
        relevance_score: point.payload
          ? Number(point.payload.relevance_score || 0)
          : undefined,
      };
    } catch (error) {
      logger.error("Failed to get document", error);
      throw error;
    }
  }

  /**
   * Delete documents
   */
  async deleteDocuments(
    ids: string[],
    collection: string = this.DEFAULT_COLLECTION
  ): Promise<void> {
    try {
      await this.client.delete(collection, {
        points: ids,
        wait: true,
      });
      logger.info(`Deleted ${ids.length} documents from ${collection}`);
    } catch (error) {
      logger.error("Failed to delete documents", error);
      throw error;
    }
  }

  /**
   * Update document metadata
   */
  async updateDocumentMetadata(
    id: string,
    metadata: Record<string, any>,
    collection: string = this.DEFAULT_COLLECTION
  ): Promise<void> {
    try {
      await this.client.setPayload(collection, {
        points: [id],
        payload: { metadata },
        wait: true,
      });
      logger.info("Updated document metadata", { id });
    } catch (error) {
      logger.error("Failed to update metadata", error);
      throw error;
    }
  }

  /**
   * Get collection statistics
   */
  async getCollectionStats(
    collection: string = this.DEFAULT_COLLECTION
  ): Promise<any> {
    try {
      const info = await this.client.getCollection(collection);
      return {
        name: collection,
        vectorsCount: info.vectors_count,
        pointsCount: info.points_count,
        segmentsCount: info.segments_count,
        config: info.config,
        status: info.status,
        optimizersStatus: info.optimizer_status,
      };
    } catch (error) {
      logger.error("Failed to get collection stats", error);
      throw error;
    }
  }

  /**
   * Optimize collection for better search performance
   */
  async optimizeCollection(
    collection: string = this.DEFAULT_COLLECTION
  ): Promise<void> {
    try {
      await this.client.updateCollection(collection, {
        optimizers_config: {
          indexing_threshold: 20000,
          max_optimization_threads: 4,
        },
      });
      logger.info(`Optimized collection: ${collection}`);
    } catch (error) {
      logger.error("Failed to optimize collection", error);
      throw error;
    }
  }

  /**
   * Create a snapshot for backup
   */
  async createSnapshot(
    collection: string = this.DEFAULT_COLLECTION
  ): Promise<string> {
    try {
      const result = await this.client.createSnapshot(collection);
      logger.info(`Created snapshot for ${collection}`, result);
      return result.name;
    } catch (error) {
      logger.error("Failed to create snapshot", error);
      throw error;
    }
  }

  /**
   * Helper methods
   */
  private async collectionExists(name: string): Promise<boolean> {
    try {
      const collections = await this.client.getCollections();
      return collections.collections.some((c) => c.name === name);
    } catch (error) {
      logger.error("Failed to check collection existence", error);
      return false;
    }
  }

  private async createCollection(name: string, config: any): Promise<void> {
    await this.client.createCollection(name, {
      vectors: {
        size: config.vectorSize,
        distance: config.distance,
        on_disk: config.onDisk,
      },
      shard_number: config.shardNumber,
      replication_factor: config.replicationFactor,
      optimizers_config: config.optimizersConfig,
      hnsw_config: config.hnswConfig,
    });
  }

  private async getCollectionInfo(name: string): Promise<any> {
    return await this.client.getCollection(name);
  }

  /**
   * Cleanup and maintenance
   */
  async cleanup(): Promise<void> {
    // Cleanup resources if needed
    logger.info("QdrantService cleanup completed");
  }
}

// Export singleton instance
export const qdrantService = new QdrantService();
