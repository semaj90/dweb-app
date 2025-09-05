/**
 * PostgreSQL to Qdrant Sync Service
 * 
 * Architecture:
 * - PostgreSQL is the single source of truth for all data
 * - Qdrant is a search index that mirrors PostgreSQL embeddings
 * - This service ensures Qdrant stays in sync with PostgreSQL
 * - Can rebuild Qdrant index entirely from PostgreSQL data
 */

import { db } from '$lib/server/db/index.js';
import { evidence, documentEmbeddings, documentMetadata } from '$lib/server/db/schema-unified.js';
import { eq, isNotNull, sql, desc, and } from 'drizzle-orm';
import { QdrantClient } from '@qdrant/js-client-rest';

export interface SyncConfig {
  qdrantUrl?: string;
  collectionName?: string;
  batchSize?: number;
  enableFullRebuild?: boolean;
  logProgress?: boolean;
  // WebAssembly-specific configuration
  wasmEmbeddingModel?: string;
  wasmOptimizedRetrieval?: boolean;
  wasmCacheSize?: number;
}

export interface SyncStats {
  totalEvidenceItems: number;
  totalDocumentEmbeddings: number;
  syncedToQdrant: number;
  skippedNoEmbedding: number;
  errors: number;
  startTime: Date;
  endTime?: Date;
  durationMs?: number;
  // WebAssembly-specific stats
  wasmOptimizedItems?: number;
  wasmCacheHits?: number;
  wasmCacheMisses?: number;
  averageRetrievalTime?: number;
}

export class PostgreSQLQdrantSyncService {
  private qdrant: QdrantClient;
  private config: Required<SyncConfig>;
  private stats: SyncStats;
  // WebAssembly-specific cache for optimized retrieval
  private wasmRetrievalCache: Map<string, { embedding: number[]; metadata: any; timestamp: number }> = new Map();

  constructor(config: SyncConfig = {}) {
    this.config = {
      qdrantUrl: config.qdrantUrl || process.env.QDRANT_URL || 'http://localhost:6333',
      collectionName: config.collectionName || 'legal_documents',
      batchSize: config.batchSize || 50,
      enableFullRebuild: config.enableFullRebuild ?? true,
      logProgress: config.logProgress ?? true,
      // WebAssembly-specific defaults
      wasmEmbeddingModel: config.wasmEmbeddingModel || 'nomic-embed-text',
      wasmOptimizedRetrieval: config.wasmOptimizedRetrieval ?? true,
      wasmCacheSize: config.wasmCacheSize || 1000
    };

    this.qdrant = new QdrantClient({ 
      url: this.config.qdrantUrl 
    });

    this.stats = this.resetStats();
  }

  private resetStats(): SyncStats {
    return {
      totalEvidenceItems: 0,
      totalDocumentEmbeddings: 0,
      syncedToQdrant: 0,
      skippedNoEmbedding: 0,
      errors: 0,
      startTime: new Date(),
      // WebAssembly-specific stats
      wasmOptimizedItems: 0,
      wasmCacheHits: 0,
      wasmCacheMisses: 0,
      averageRetrievalTime: 0
    };
  }

  private log(message: string) {
    if (this.config.logProgress) {
      console.log(`🔄 [PostgreSQL→Qdrant] ${message}`);
    }
  }

  /**
   * Ensure Qdrant collection exists with proper configuration
   */
  async ensureCollection(): Promise<void> {
    try {
      const collections = await this.qdrant.getCollections();
      const exists = collections.collections.some(c => c.name === this.config.collectionName);

      if (!exists) {
        await this.qdrant.createCollection(this.config.collectionName, {
          vectors: {
            size: 768, // nomic-embed-text dimensions (corrected)
            distance: 'Cosine'
          },
          optimizers_config: {
            default_segment_number: 2,
            memmap_threshold: 20000
          },
          hnsw_config: {
            m: 16,
            ef_construct: 100
          }
        });
        this.log(`Created Qdrant collection: ${this.config.collectionName}`);
      } else {
        this.log(`Qdrant collection exists: ${this.config.collectionName}`);
      }
    } catch (error: any) {
      throw new Error(`Failed to setup Qdrant collection: ${error.message}`);
    }
  }

  /**
   * Full rebuild of Qdrant index from PostgreSQL data
   */
  async fullRebuild(): Promise<SyncStats> {
    if (!this.config.enableFullRebuild) {
      throw new Error('Full rebuild is disabled in configuration');
    }

    this.stats = this.resetStats();
    this.log('Starting full rebuild of Qdrant index from PostgreSQL');

    try {
      // Ensure collection exists
      await this.ensureCollection();

      // Clear existing Qdrant data
      await this.clearQdrantCollection();

      // Sync all evidence items
      await this.syncAllEvidence();

      // Sync all document embeddings
      await this.syncAllDocumentEmbeddings();

      // Finalize stats
      this.stats.endTime = new Date();
      this.stats.durationMs = this.stats.endTime.getTime() - this.stats.startTime.getTime();

      this.log(`Full rebuild completed in ${this.stats.durationMs}ms`);
      this.log(`Synced: ${this.stats.syncedToQdrant}, Skipped: ${this.stats.skippedNoEmbedding}, Errors: ${this.stats.errors}`);

      return this.stats;
    } catch (error: any) {
      this.log(`Full rebuild failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Incremental sync - only sync items that have changed
   */
  async incrementalSync(sinceTimestamp?: Date): Promise<SyncStats> {
    this.stats = this.resetStats();
    const since = sinceTimestamp || new Date(Date.now() - 24 * 60 * 60 * 1000); // Last 24 hours

    this.log(`Starting incremental sync since ${since.toISOString()}`);

    try {
      await this.ensureCollection();

      // Sync updated evidence
      await this.syncRecentEvidence(since);

      // Sync updated document embeddings
      await this.syncRecentDocumentEmbeddings(since);

      this.stats.endTime = new Date();
      this.stats.durationMs = this.stats.endTime.getTime() - this.stats.startTime.getTime();

      this.log(`Incremental sync completed in ${this.stats.durationMs}ms`);
      return this.stats;
    } catch (error: any) {
      this.log(`Incremental sync failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Sync specific evidence item by ID
   */
  async syncEvidenceById(evidenceId: string): Promise<boolean> {
    try {
      const evidenceItem = await this.getEvidenceWithEmbedding(evidenceId);
      if (!evidenceItem) {
        this.log(`Evidence ${evidenceId} not found`);
        return false;
      }

      return await this.syncSingleEvidence(evidenceItem);
    } catch (error: any) {
      this.log(`Failed to sync evidence ${evidenceId}: ${error.message}`);
      return false;
    }
  }

  /**
   * Sync specific document embedding by ID
   */
  async syncDocumentById(documentId: string): Promise<boolean> {
    try {
      const docData = await this.getDocumentWithEmbedding(documentId);
      if (!docData) {
        this.log(`Document ${documentId} not found`);
        return false;
      }

      return await this.syncSingleDocument(docData);
    } catch (error: any) {
      this.log(`Failed to sync document ${documentId}: ${error.message}`);
      return false;
    }
  }

  /**
   * Clear all data in Qdrant collection
   */
  private async clearQdrantCollection(): Promise<void> {
    try {
      // Delete collection and recreate (fastest way to clear)
      await this.qdrant.deleteCollection(this.config.collectionName);
      await this.ensureCollection();
      this.log(`Cleared Qdrant collection: ${this.config.collectionName}`);
    } catch (error: any) {
      this.log(`Failed to clear Qdrant collection: ${error.message}`);
    }
  }

  /**
   * Sync all evidence items with embeddings
   */
  private async syncAllEvidence(): Promise<void> {
    let offset = 0;
    let hasMore = true;

    while (hasMore) {
      const evidenceItems = await db
        .select({
          id: evidence.id,
          caseId: evidence.caseId,
          title: evidence.title,
          description: evidence.description,
          evidenceType: evidence.evidenceType,
          mimeType: evidence.mimeType,
          aiTags: evidence.aiTags,
          confidentialityLevel: evidence.confidentialityLevel,
          titleEmbedding: evidence.titleEmbedding,
          contentEmbedding: evidence.contentEmbedding,
          createdAt: evidence.createdAt
        })
        .from(evidence)
        .where(
          and(
            isNotNull(evidence.titleEmbedding),
            isNotNull(evidence.contentEmbedding)
          )
        )
        .orderBy(desc(evidence.createdAt))
        .limit(this.config.batchSize)
        .offset(offset);

      if (evidenceItems.length === 0) {
        hasMore = false;
        break;
      }

      this.stats.totalEvidenceItems += evidenceItems.length;

      // Process batch
      for (const item of evidenceItems) {
        await this.syncSingleEvidence(item);
      }

      offset += this.config.batchSize;
      this.log(`Processed ${offset} evidence items`);
    }
  }

  /**
   * Sync all document embeddings
   */
  private async syncAllDocumentEmbeddings(): Promise<void> {
    let offset = 0;
    let hasMore = true;

    while (hasMore) {
      const documents = await db
        .select({
          embedding: documentEmbeddings,
          metadata: documentMetadata
        })
        .from(documentEmbeddings)
        .leftJoin(documentMetadata, eq(documentEmbeddings.documentId, documentMetadata.id))
        .where(isNotNull(documentEmbeddings.embedding))
        .orderBy(desc(documentEmbeddings.createdAt))
        .limit(this.config.batchSize)
        .offset(offset);

      if (documents.length === 0) {
        hasMore = false;
        break;
      }

      this.stats.totalDocumentEmbeddings += documents.length;

      // Process batch
      for (const item of documents) {
        await this.syncSingleDocument(item);
      }

      offset += this.config.batchSize;
      this.log(`Processed ${offset} document embeddings`);
    }
  }

  /**
   * Sync recent evidence items
   */
  private async syncRecentEvidence(since: Date): Promise<void> {
    const evidenceItems = await db
      .select({
        id: evidence.id,
        caseId: evidence.caseId,
        title: evidence.title,
        description: evidence.description,
        evidenceType: evidence.evidenceType,
        mimeType: evidence.mimeType,
        aiTags: evidence.aiTags,
        confidentialityLevel: evidence.confidentialityLevel,
        titleEmbedding: evidence.titleEmbedding,
        contentEmbedding: evidence.contentEmbedding,
        createdAt: evidence.createdAt,
        updatedAt: evidence.updatedAt
      })
      .from(evidence)
      .where(
        and(
          sql`${evidence.updatedAt} >= ${since}`,
          isNotNull(evidence.titleEmbedding),
          isNotNull(evidence.contentEmbedding)
        )
      )
      .orderBy(desc(evidence.updatedAt));

    this.stats.totalEvidenceItems = evidenceItems.length;

    for (const item of evidenceItems) {
      await this.syncSingleEvidence(item);
    }
  }

  /**
   * Sync recent document embeddings
   */
  private async syncRecentDocumentEmbeddings(since: Date): Promise<void> {
    const documents = await db
      .select({
        embedding: documentEmbeddings,
        metadata: documentMetadata
      })
      .from(documentEmbeddings)
      .leftJoin(documentMetadata, eq(documentEmbeddings.documentId, documentMetadata.id))
      .where(
        and(
          sql`${documentEmbeddings.createdAt} >= ${since}`,
          isNotNull(documentEmbeddings.embedding)
        )
      )
      .orderBy(desc(documentEmbeddings.createdAt));

    this.stats.totalDocumentEmbeddings = documents.length;

    for (const item of documents) {
      await this.syncSingleDocument(item);
    }
  }

  /**
   * Get evidence with embedding data
   */
  private async getEvidenceWithEmbedding(evidenceId: string) {
    const [result] = await db
      .select({
        id: evidence.id,
        caseId: evidence.caseId,
        title: evidence.title,
        description: evidence.description,
        evidenceType: evidence.evidenceType,
        mimeType: evidence.mimeType,
        aiTags: evidence.aiTags,
        confidentialityLevel: evidence.confidentialityLevel,
        titleEmbedding: evidence.titleEmbedding,
        contentEmbedding: evidence.contentEmbedding,
        createdAt: evidence.createdAt
      })
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    return result;
  }

  /**
   * Get document with embedding data
   */
  private async getDocumentWithEmbedding(documentId: string) {
    const [result] = await db
      .select({
        embedding: documentEmbeddings,
        metadata: documentMetadata
      })
      .from(documentEmbeddings)
      .leftJoin(documentMetadata, eq(documentEmbeddings.documentId, documentMetadata.id))
      .where(eq(documentEmbeddings.documentId, documentId))
      .limit(1);

    return result;
  }

  /**
   * Sync single evidence item to Qdrant
   */
  private async syncSingleEvidence(evidenceItem: any): Promise<boolean> {
    try {
      // Use content embedding first, fallback to title embedding
      const embedding = evidenceItem.contentEmbedding || evidenceItem.titleEmbedding;
      const content = evidenceItem.description || evidenceItem.title;

      if (!embedding) {
        this.stats.skippedNoEmbedding++;
        return false;
      }

      await this.qdrant.upsert(this.config.collectionName, {
        wait: false,
        points: [{
          id: `evidence_${evidenceItem.id}`,
          vector: embedding,
          payload: {
            type: 'evidence',
            evidenceId: evidenceItem.id,
            caseId: evidenceItem.caseId,
            title: evidenceItem.title,
            tags: evidenceItem.aiTags || [],
            content: content,
            metadata: {
              evidenceType: evidenceItem.evidenceType,
              mimeType: evidenceItem.mimeType,
              confidentialityLevel: evidenceItem.confidentialityLevel,
              source: 'postgresql_sync',
              syncedAt: new Date().toISOString()
            }
          }
        }]
      });

      this.stats.syncedToQdrant++;
      return true;
    } catch (error: any) {
      this.stats.errors++;
      this.log(`Failed to sync evidence ${evidenceItem.id}: ${error.message}`);
      return false;
    }
  }

  /**
   * Sync single document to Qdrant
   */
  private async syncSingleDocument(docData: any): Promise<boolean> {
    try {
      const { embedding, metadata } = docData;

      if (!embedding?.embedding) {
        this.stats.skippedNoEmbedding++;
        return false;
      }

      await this.qdrant.upsert(this.config.collectionName, {
        wait: false,
        points: [{
          id: `document_${embedding.id}`,
          vector: embedding.embedding,
          payload: {
            type: 'document',
            documentId: embedding.documentId,
            evidenceId: embedding.evidenceId,
            title: metadata?.originalFilename || `Document ${embedding.documentId}`,
            content: embedding.content,
            metadata: {
              documentType: metadata?.documentType,
              processingStatus: metadata?.processingStatus,
              embeddingModel: embedding.embeddingModel,
              chunkIndex: embedding.chunkIndex,
              source: 'postgresql_sync',
              syncedAt: new Date().toISOString()
            }
          }
        }]
      });

      this.stats.syncedToQdrant++;
      return true;
    } catch (error: any) {
      this.stats.errors++;
      this.log(`Failed to sync document ${docData.embedding?.id}: ${error.message}`);
      return false;
    }
  }

  /**
   * Get current sync statistics
   */
  getStats(): SyncStats {
    return { ...this.stats };
  }

  /**
   * Health check for both PostgreSQL and Qdrant
   */
  async healthCheck(): Promise<{
    postgresql: boolean;
    qdrant: boolean;
    collection: boolean;
    status: 'healthy' | 'degraded' | 'unhealthy';
  }> {
    const health = {
      postgresql: false,
      qdrant: false,
      collection: false,
      status: 'unhealthy' as const
    };

    try {
      // Check PostgreSQL
      await db.execute(sql`SELECT 1`);
      health.postgresql = true;
    } catch (error: any) {
      this.log(`PostgreSQL health check failed: ${error.message}`);
    }

    try {
      // Check Qdrant
      await this.qdrant.getCollections();
      health.qdrant = true;

      // Check collection
      const collections = await this.qdrant.getCollections();
      health.collection = collections.collections.some(c => c.name === this.config.collectionName);
    } catch (error: any) {
      this.log(`Qdrant health check failed: ${error.message}`);
    }

    // Determine overall status
    if (health.postgresql && health.qdrant && health.collection) {
      health.status = 'healthy';
    } else if (health.postgresql && health.qdrant) {
      health.status = 'degraded';
    }

    return health;
  }

  /**
   * WebAssembly-optimized vector search for RAG inference
   * Integrates directly with WebAssembly inference pipeline
   */
  async searchForWASMInference(
    queryEmbedding: number[], 
    limit: number = 5,
    scoreThreshold: number = 0.7,
    filters?: Record<string, any>
  ): Promise<Array<{
    id: string;
    content: string;
    score: number;
    metadata: any;
  }>> {
    const startTime = Date.now();

    try {
      this.log(`🧠 Performing WASM-optimized vector search (limit: ${limit}, threshold: ${scoreThreshold})`);

      // Check cache first if enabled
      const cacheKey = this.generateCacheKey(queryEmbedding, limit, filters);
      if (this.config.wasmOptimizedRetrieval && this.wasmRetrievalCache.has(cacheKey)) {
        const cached = this.wasmRetrievalCache.get(cacheKey)!;
        const cacheAge = Date.now() - cached.timestamp;
        
        // Cache valid for 5 minutes
        if (cacheAge < 5 * 60 * 1000) {
          this.stats.wasmCacheHits = (this.stats.wasmCacheHits || 0) + 1;
          this.log(`✅ WASM cache hit (age: ${Math.round(cacheAge / 1000)}s)`);
          return cached.metadata as any[];
        }
      }

      // Perform Qdrant search with WASM-optimized parameters
      const searchResponse = await this.qdrant.search(this.config.collectionName, {
        vector: queryEmbedding,
        limit: limit * 2, // Over-fetch to allow for filtering
        score_threshold: scoreThreshold,
        with_payload: true,
        with_vector: false, // Don't return vectors to save bandwidth
        ...(filters && { filter: this.buildQdrantFilter(filters) })
      });

      // Process and optimize results for WASM inference
      const results = searchResponse
        .filter(hit => hit.score >= scoreThreshold)
        .slice(0, limit)
        .map(hit => ({
          id: hit.id as string,
          content: hit.payload?.content as string || '',
          score: hit.score,
          metadata: {
            type: hit.payload?.type,
            title: hit.payload?.title,
            tags: hit.payload?.tags || [],
            evidenceId: hit.payload?.evidenceId,
            documentId: hit.payload?.documentId,
            caseId: hit.payload?.caseId,
            retrievedAt: new Date().toISOString(),
            retrievalMethod: 'wasm_optimized',
            embeddingModel: this.config.wasmEmbeddingModel
          }
        }));

      // Cache results if enabled
      if (this.config.wasmOptimizedRetrieval) {
        this.wasmRetrievalCache.set(cacheKey, {
          embedding: queryEmbedding,
          metadata: results,
          timestamp: Date.now()
        });

        // Cleanup old cache entries
        this.cleanupWASMCache();
      }

      const retrievalTime = Date.now() - startTime;
      this.stats.wasmCacheMisses = (this.stats.wasmCacheMisses || 0) + 1;
      this.stats.averageRetrievalTime = this.updateAverageRetrievalTime(retrievalTime);
      
      this.log(`🔍 WASM search completed: ${results.length} results in ${retrievalTime}ms`);
      return results;

    } catch (error: any) {
      this.log(`❌ WASM-optimized search failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Batch vector search optimized for WebAssembly inference
   */
  async batchSearchForWASMInference(
    queries: Array<{
      embedding: number[];
      id: string;
      limit?: number;
      filters?: Record<string, any>;
    }>,
    scoreThreshold: number = 0.7
  ): Promise<Record<string, Array<{
    id: string;
    content: string;
    score: number;
    metadata: any;
  }>>> {
    const startTime = Date.now();
    const results: Record<string, any[]> = {};

    try {
      this.log(`🔄 Performing WASM batch search for ${queries.length} queries`);

      // Process queries in parallel for better performance
      const searchPromises = queries.map(async (query) => {
        const queryResults = await this.searchForWASMInference(
          query.embedding,
          query.limit || 5,
          scoreThreshold,
          query.filters
        );
        return { id: query.id, results: queryResults };
      });

      const allResults = await Promise.all(searchPromises);
      
      // Organize results by query ID
      allResults.forEach(({ id, results: queryResults }) => {
        results[id] = queryResults;
      });

      const totalTime = Date.now() - startTime;
      this.log(`✅ WASM batch search completed: ${queries.length} queries in ${totalTime}ms`);

      return results;

    } catch (error: any) {
      this.log(`❌ WASM batch search failed: ${error.message}`);
      throw error;
    }
  }

  /**
   * Store WebAssembly inference result for future RAG improvements
   */
  async storeWASMInferenceResult(
    queryEmbedding: number[],
    retrievedDocuments: string[],
    inferenceResult: string,
    metadata: {
      inferenceId: string;
      model: string;
      processingTime: number;
      ragContext?: any;
    }
  ): Promise<void> {
    try {
      // Store the inference result as a new document in Qdrant for future retrieval
      const resultId = `wasm_inference_${metadata.inferenceId}`;
      
      await this.qdrant.upsert(this.config.collectionName, {
        wait: false,
        points: [{
          id: resultId,
          vector: queryEmbedding,
          payload: {
            type: 'wasm_inference_result',
            inferenceId: metadata.inferenceId,
            content: inferenceResult,
            retrievedDocuments,
            metadata: {
              model: metadata.model,
              processingTime: metadata.processingTime,
              ragContext: metadata.ragContext,
              createdAt: new Date().toISOString(),
              source: 'wasm_inference',
              embeddingModel: this.config.wasmEmbeddingModel
            }
          }
        }]
      });

      this.log(`📝 Stored WASM inference result: ${metadata.inferenceId}`);

    } catch (error: any) {
      this.log(`❌ Failed to store WASM inference result: ${error.message}`);
    }
  }

  /**
   * Clean up old WebAssembly cache entries
   */
  private cleanupWASMCache(): void {
    const maxAge = 10 * 60 * 1000; // 10 minutes
    const now = Date.now();
    
    let cleanedCount = 0;
    for (const [key, entry] of this.wasmRetrievalCache.entries()) {
      if (now - entry.timestamp > maxAge) {
        this.wasmRetrievalCache.delete(key);
        cleanedCount++;
      }
    }

    // Also enforce cache size limit
    if (this.wasmRetrievalCache.size > this.config.wasmCacheSize) {
      // Remove oldest entries
      const entries = Array.from(this.wasmRetrievalCache.entries())
        .sort(([,a], [,b]) => a.timestamp - b.timestamp);
      
      const toRemove = entries.slice(0, this.wasmRetrievalCache.size - this.config.wasmCacheSize);
      toRemove.forEach(([key]) => this.wasmRetrievalCache.delete(key));
      cleanedCount += toRemove.length;
    }

    if (cleanedCount > 0) {
      this.log(`🧹 Cleaned ${cleanedCount} WASM cache entries`);
    }
  }

  /**
   * Generate cache key for WASM retrieval
   */
  private generateCacheKey(
    embedding: number[],
    limit: number,
    filters?: Record<string, any>
  ): string {
    // Create a hash from embedding (use first few dimensions for speed)
    const embeddingHash = embedding.slice(0, 10).join(',');
    const filtersHash = filters ? JSON.stringify(filters) : '';
    return `wasm_${embeddingHash}_${limit}_${filtersHash}`;
  }

  /**
   * Build Qdrant filter from generic filters object
   */
  private buildQdrantFilter(filters: Record<string, any>): any {
    const qdrantFilter: any = {
      must: []
    };

    Object.entries(filters).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        qdrantFilter.must.push({
          key: `payload.${key}`,
          match: { any: value }
        });
      } else {
        qdrantFilter.must.push({
          key: `payload.${key}`,
          match: { value }
        });
      }
    });

    return qdrantFilter;
  }

  /**
   * Update average retrieval time
   */
  private updateAverageRetrievalTime(newTime: number): number {
    const currentAvg = this.stats.averageRetrievalTime || 0;
    const totalQueries = (this.stats.wasmCacheMisses || 0) + (this.stats.wasmCacheHits || 0);
    
    if (totalQueries <= 1) {
      return newTime;
    }
    
    return ((currentAvg * (totalQueries - 1)) + newTime) / totalQueries;
  }

  /**
   * Get WebAssembly-specific statistics
   */
  getWASMStats(): {
    cacheSize: number;
    cacheHits: number;
    cacheMisses: number;
    averageRetrievalTime: number;
    cacheHitRate: number;
  } {
    const hits = this.stats.wasmCacheHits || 0;
    const misses = this.stats.wasmCacheMisses || 0;
    const total = hits + misses;

    return {
      cacheSize: this.wasmRetrievalCache.size,
      cacheHits: hits,
      cacheMisses: misses,
      averageRetrievalTime: this.stats.averageRetrievalTime || 0,
      cacheHitRate: total > 0 ? hits / total : 0
    };
  }

  /**
   * Clear WebAssembly cache
   */
  clearWASMCache(): void {
    this.wasmRetrievalCache.clear();
    this.log('🧹 Cleared WASM retrieval cache');
  }
}

// Export singleton instance
export const postgresqlQdrantSync = new PostgreSQLQdrantSyncService();

// Default export
export default PostgreSQLQdrantSyncService;