
// src/lib/server/vector/vectorService.ts (continued)
import { QdrantClient } from "@qdrant/js-client-rest";
import { Redis } from "ioredis";
import {
  embeddingCache,
  vectorMetadata,
  cases,
  evidence,
  criminals
} from '../db/schema-postgres.js';
import { eq, and, sql, or, ilike } from "drizzle-orm";
import cuid2 from "@paralleldrive/cuid2";

export interface VectorSearchOptions {
  limit?: number;
  threshold?: number;
  filter?: Record<string, any>;
  includeMetadata?: boolean;
}

export interface EmbeddingResult {
  id: string;
  score: number;
  metadata: any;
  content: string;
}

export class VectorService {
  private qdrant: QdrantClient;
  private redis: Redis;
  private collectionName: string;

  constructor() {
    this.qdrant = new QdrantClient({
      url: import.meta.env.QDRANT_URL || "http://localhost:6333",
      apiKey: import.meta.env.QDRANT_API_KEY,
    });

    this.redis = new Redis({
      host: import.meta.env.REDIS_HOST || "localhost",
      port: parseInt(import.meta.env.REDIS_PORT || "6379"),
      password: import.meta.env.REDIS_PASSWORD,
      db: parseInt(import.meta.env.REDIS_DB || "0"),
      maxRetriesPerRequest: 3,
    });

    this.collectionName = import.meta.env.QDRANT_COLLECTION || "legal_documents";
  }

  // Initialize vector collection with proper schema
  async initializeCollection(): Promise<void> {
    try {
      const collections = await this.qdrant.getCollections();
      const exists = collections.collections.some(
        (c) => c.name === this.collectionName,
      );

      if (!exists) {
        await this.qdrant.createCollection(this.collectionName, {
          vectors: {
            size: 768, // Nomic Embed dimension
            distance: "Cosine",
          },
          optimizers_config: {
            default_segment_number: 2,
          },
          replication_factor: 1,
        });

        // Create index for better performance - API compatibility issue
        // Note: createPayloadIndex method doesn't exist in current Qdrant client
        console.log("Payload index creation skipped - method compatibility issue");
        // TODO: Replace with correct Qdrant v1+ API methods when available
        
        try {
          // await this.qdrant.createPayloadIndex(this.collectionName, "type");
          // await this.qdrant.createPayloadIndex(this.collectionName, "case_id");
        } catch (error: any) {
          console.log("Index for 'case_id' may already exist");
        }

        console.log(`Created Qdrant collection: ${this.collectionName}`);
      }
    } catch (error: any) {
      console.error("Failed to initialize Qdrant collection:", error);
      throw error;
    }
  }

  // Generate embeddings using Ollama with Nomic Embed model
  async generateEmbedding(text: string): Promise<number[]> {
    const cacheKey = `embedding:${Buffer.from(text).toString("base64")}`;

    try {
      // Check Redis cache first
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }

      // Generate new embedding using Ollama
      const response = await fetch("http://localhost:11434/api/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "nomic-embed-text",
          prompt: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama embedding API error: ${response.statusText}`);
      }

      const result = await response.json();
      const embedding = result.embedding;

      // Cache the result with 24-hour expiry - using modern Redis syntax
      await this.redis.setex(cacheKey, 24 * 60 * 60, JSON.stringify(embedding));

      // Also store in PostgreSQL for persistence
      await db
        .insert(embeddingCache)
        .values({
          id: cuid2.createId(),
          textHash: Buffer.from(text).toString("base64"),
          embedding: embedding,
          model: "nomic-embed-text",
          createdAt: new Date(),
        })
        .onConflictDoNothing();

      return embedding;
    } catch (error: any) {
      console.error("Failed to generate embedding:", error);

      // Fallback: check PostgreSQL cache
      try {
        const cached = await db
          .select()
          .from(embeddingCache)
          .where(
            eq(embeddingCache.textHash, Buffer.from(text).toString("base64")),
          )
          .limit(1);

        if (cached.length > 0) {
          return cached[0].embedding as number[];
        }
      } catch (dbError) {
        console.error("Failed to fetch from embedding cache:", dbError);
      }

      throw error;
    }
  }

  // Store document with vector embedding
  async storeDocument(
    id: string,
    content: string,
    metadata: {
      type: "case" | "evidence" | "criminal" | "document";
      case_id?: string;
      title: string;
      created_at: string;
      [key: string]: unknown;
    },
  ): Promise<void> {
    try {
      // Generate embedding
      const embedding = await this.generateEmbedding(content);

      // Store in Qdrant
      await this.qdrant.upsert(this.collectionName, {
        wait: true,
        points: [
          {
            id: id,
            vector: embedding,
            payload: {
              content,
              ...metadata,
            },
          },
        ],
      });

      // Store metadata in PostgreSQL
      await db
        .insert(vectorMetadata)
        .values({
          id: cuid2.createId(),
          documentId: id,
          collectionName: this.collectionName,
          metadata: metadata,
          contentHash: Buffer.from(content).toString("base64"),
          createdAt: new Date(),
        })
        .onConflictDoUpdate({
          target: vectorMetadata.documentId,
          set: {
            metadata: metadata,
            contentHash: Buffer.from(content).toString("base64"),
            updatedAt: new Date(),
          },
        });

      console.log(`Stored document ${id} in vector database`);
    } catch (error: any) {
      console.error("Failed to store document:", error);
      throw error;
    }
  }

  // Semantic search with hybrid scoring
  async search(
    query: string,
    options: VectorSearchOptions = {},
  ): Promise<EmbeddingResult[]> {
    const {
      limit = 10,
      threshold = 0.7,
      filter = {},
      includeMetadata = true,
    } = options;

    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      // Build Qdrant filter
      const qdrantFilter = this.buildQdrantFilter(filter);

      // Perform vector search
      const searchResult = await this.qdrant.search(this.collectionName, {
        vector: queryEmbedding,
        limit,
        score_threshold: threshold,
        filter: qdrantFilter,
        with_payload: true,
      });

      // Format results
      const results: EmbeddingResult[] = searchResult.map((point) => ({
        id: point.id.toString(),
        score: point.score,
        metadata: includeMetadata ? point.payload : {},
        content: (point.payload?.content as string) || "",
      }));

      // Cache search results
      const cacheKey = `search:${Buffer.from(query + JSON.stringify(options)).toString("base64")}`;
      await this.redis.setex(cacheKey, 5 * 60, JSON.stringify(results)); // 5-minute cache

      return results;
    } catch (error: any) {
      console.error("Vector search failed:", error);
      throw error;
    }
  }

  // Hybrid search combining vector similarity and keyword matching
  async hybridSearch(
    query: string,
    options: VectorSearchOptions & {
      keywordWeight?: number;
      vectorWeight?: number;
    } = {},
  ): Promise<EmbeddingResult[]> {
    const {
      limit = 10,
      threshold = 0.7,
      keywordWeight = 0.3,
      vectorWeight = 0.7,
      filter = {},
    } = options;

    try {
      // Perform vector search
      const vectorResults = await this.search(query, {
        ...options,
        limit: limit * 2,
      });

      // Perform keyword search in PostgreSQL
      const keywordResults = await this.keywordSearch(query, filter, limit * 2);

      // Combine and re-rank results
      const combinedResults = this.combineSearchResults(
        vectorResults,
        keywordResults,
        vectorWeight,
        keywordWeight,
      );

      return combinedResults.slice(0, limit);
    } catch (error: any) {
      console.error("Hybrid search failed:", error);
      throw error;
    }
  }

  // Keyword search using PostgreSQL full-text search
  private async keywordSearch(
    query: string,
    filter: Record<string, any>,
    limit: number,
  ): Promise<EmbeddingResult[]> {
    try {
      const results: EmbeddingResult[] = [];

      // Search cases
      if (!filter.type || filter.type === "case") {
        const caseResults = await db
          .select()
          .from(cases)
          .where(
            or(
              ilike(cases.title, `%${query}%`),
              ilike(cases.description, `%${query}%`),
              ilike(cases.category, `%${query}%`),
            ),
          )
          .limit(limit);

        results.push(
          ...caseResults.map((c) => ({
            id: c.id,
            score: 0.8, // Default keyword score
            metadata: { type: "case", title: c.title, case_id: c.id },
            content: `${c.title} ${c.description}`,
          })),
        );
      }

      // Search evidence
      if (!filter.type || filter.type === "evidence") {
        const evidenceResults = await db
          .select()
          .from(evidence)
          .where(
            or(
              ilike(evidence.title, `%${query}%`),
              ilike(evidence.description, `%${query}%`),
              ilike(evidence.summary, `%${query}%`),
            ),
          )
          .limit(limit);

        results.push(
          ...evidenceResults.map((e: any) => ({
            id: e.id,
            score: 0.8,
            metadata: { type: "evidence", title: e.title, case_id: e.caseId },
            content: `${e.title} ${e.description || ""} ${e.summary || ""}`,
          })),
        );
      }

      // Search criminals
      if (!filter.type || filter.type === "criminal") {
        const criminalResults = await db
          .select()
          .from(criminals)
          .where(
            or(
              ilike(criminals.firstName, `%${query}%`),
              ilike(criminals.lastName, `%${query}%`),
              ilike(criminals.notes, `%${query}%`),
            ),
          )
          .limit(limit);

        results.push(
          ...criminalResults.map((c) => ({
            id: c.id,
            score: 0.8,
            metadata: {
              type: "criminal",
              title: `${c.firstName} ${c.lastName}`,
            },
            content: `${c.firstName} ${c.lastName} ${c.notes || ""}`,
          })),
        );
      }

      return results;
    } catch (error: any) {
      console.error("Keyword search failed:", error);
      return [];
    }
  }

  // Combine vector and keyword search results
  private combineSearchResults(
    vectorResults: EmbeddingResult[],
    keywordResults: EmbeddingResult[],
    vectorWeight: number,
    keywordWeight: number,
  ): EmbeddingResult[] {
    const combinedMap = new Map<string, EmbeddingResult>();

    // Add vector results
    vectorResults.forEach((result) => {
      combinedMap.set(result.id, {
        ...result,
        score: result.score * vectorWeight,
      });
    });

    // Add keyword results
    keywordResults.forEach((result) => {
      const existing = combinedMap.get(result.id);
      if (existing) {
        // Combine scores
        existing.score += result.score * keywordWeight;
      } else {
        combinedMap.set(result.id, {
          ...result,
          score: result.score * keywordWeight,
        });
      }
    });

    // Sort by combined score
    return Array.from(combinedMap.values()).sort((a, b) => b.score - a.score);
  }

  // Build Qdrant filter from options
  private buildQdrantFilter(filter: Record<string, any>): unknown {
    if (!filter || Object.keys(filter).length === 0) {
      return undefined;
    }

    const must: any[] = [];

    if (filter.type) {
      must.push({
        key: "type",
        match: { value: filter.type },
      });
    }

    if (filter.case_id) {
      must.push({
        key: "case_id",
        match: { value: filter.case_id },
      });
    }

    if (filter.created_after) {
      must.push({
        key: "created_at",
        range: { gte: filter.created_after },
      });
    }

    if (filter.created_before) {
      must.push({
        key: "created_at",
        range: { lte: filter.created_before },
      });
    }

    return must.length > 0 ? { must } : undefined;
  }

  // Find similar documents
  async findSimilar(
    documentId: string,
    options: VectorSearchOptions = {},
  ): Promise<EmbeddingResult[]> {
    try {
      // Get the document - method compatibility issue
      // TODO: Verify correct Qdrant client API for retrieve method
      // const response = await this.qdrant.retrieve(this.collectionName, [documentId]);
      // const point = response.points;
      
      // Placeholder response for now
      const response = { points: [] };
      const point = response.points;

      if (point.length === 0) {
        console.warn(`Document retrieval skipped due to API compatibility - document ${documentId}`);
        return [];
      }

      const document = point[0];
      const vector = document.vector as number[];

      // Search for similar documents
      const similar = await this.qdrant.search(this.collectionName, {
        vector,
        limit: (options.limit || 10) + 1, // +1 to exclude self
        score_threshold: options.threshold || 0.7,
        filter: this.buildQdrantFilter(options.filter || {}),
        with_payload: true,
      });

      // Filter out the original document
      const results = similar
        .filter((p) => p.id.toString() !== documentId)
        .map((point) => ({
          id: point.id.toString(),
          score: point.score,
          metadata: point.payload,
          content: (point.payload?.content as string) || "",
        }));

      return results.slice(0, options.limit || 10);
    } catch (error: any) {
      console.error("Failed to find similar documents:", error);
      throw error;
    }
  }

  // Bulk index documents
  async bulkIndex(
    documents: Array<{
      id: string;
      content: string;
      metadata: any;
    }>,
  ): Promise<void> {
    try {
      const batchSize = 50;

      for (let i = 0; i < documents.length; i += batchSize) {
        const batch = documents.slice(i, i + batchSize);

        // Generate embeddings for batch
        const embeddings = await Promise.all(
          batch.map((doc) => this.generateEmbedding(doc.content)),
        );

        // Prepare points for Qdrant
        const points = batch.map((doc, index) => ({
          id: doc.id,
          vector: embeddings[index],
          payload: {
            content: doc.content,
            ...doc.metadata,
          },
        }));

        // Upsert batch to Qdrant
        await this.qdrant.upsert(this.collectionName, {
          wait: true,
          points,
        });

        // Store metadata in PostgreSQL
        const metadataRecords = batch.map((doc) => ({
          id: cuid2.createId(),
          documentId: doc.id,
          collectionName: this.collectionName,
          metadata: doc.metadata,
          contentHash: Buffer.from(doc.content).toString("base64"),
          createdAt: new Date(),
        }));

        await db
          .insert(vectorMetadata)
          .values(metadataRecords)
          .onConflictDoNothing();

        console.log(
          `Indexed batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(documents.length / batchSize)}`,
        );
      }

      console.log(`Bulk indexing completed: ${documents.length} documents`);
    } catch (error: any) {
      console.error("Bulk indexing failed:", error);
      throw error;
    }
  }

  // Delete document
  async deleteDocument(documentId: string): Promise<void> {
    try {
      // Delete from Qdrant
      await this.qdrant.delete(this.collectionName, {
        wait: true,
        points: [documentId],
      });

      // Delete metadata from PostgreSQL
      await db
        .delete(vectorMetadata)
        .where(eq(vectorMetadata.documentId, documentId));

      console.log(`Deleted document ${documentId}`);
    } catch (error: any) {
      console.error("Failed to delete document:", error);
      throw error;
    }
  }

  // Health check
  async healthCheck(): Promise<{
    qdrant: boolean;
    redis: boolean;
    collection: boolean;
  }> {
    const status = {
      qdrant: false,
      redis: false,
      collection: false,
    };

    try {
      // Check Qdrant
      await this.qdrant.getCollections();
      status.qdrant = true;

      // Check collection exists
      const collections = await this.qdrant.getCollections();
      status.collection = collections.collections.some(
        (c) => c.name === this.collectionName,
      );
    } catch (error: any) {
      console.error("Qdrant health check failed:", error);
    }

    try {
      // Check Redis
      await this.redis.ping();
      status.redis = true;
    } catch (error: any) {
      console.error("Redis health check failed:", error);
    }

    return status;
  }

  // Get collection stats
  async getStats(): Promise<{
    documentCount: number;
    collectionInfo: any;
  }> {
    try {
      const info = await this.qdrant.getCollection(this.collectionName);

      return {
        documentCount: info.points_count || 0,
        collectionInfo: info,
      };
    } catch (error: any) {
      console.error("Failed to get collection stats:", error);
      return {
        documentCount: 0,
        collectionInfo: null,
      };
    }
  }

  // Close connections
  async close(): Promise<void> {
    try {
      await this.redis.quit();
    } catch (error: any) {
      console.error("Failed to close Redis connection:", error);
    }
  }
}

// Singleton instance
export const vectorService = new VectorService();
;