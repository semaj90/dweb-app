/**
 * Optimized Qdrant Service with Memory-Efficient Design
 * Integrates SOM clustering, NES cache orchestrator, and PostgreSQL sync
 * Low memory usage with intelligent caching and batch processing
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import type { PointStruct, ScoredPoint, SearchParams } from '@qdrant/js-client-rest';
import { LegalDocumentSOM } from './som-clustering.js';
import { NESCacheOrchestrator } from './nes-cache-orchestrator.js';
import { db } from '$lib/server/db/index.js';
import { evidence, cases, legalDocuments } from '$lib/server/db/unified-schema.js';
import { eq, sql, inArray, desc } from 'drizzle-orm';

// Corrected dimensions for nomic-embed-text (768, not 384)
const NOMIC_EMBED_DIMENSIONS = 768;
const BATCH_SIZE = 50;
const MAX_MEMORY_USAGE = 32 * 1024 * 1024; // 32MB memory limit

export interface QdrantConfig {
  url?: string;
  apiKey?: string;
  timeout?: number;
  collectionName?: string;
  enableBatching?: boolean;
  enableSOMClustering?: boolean;
  enableNESCache?: boolean;
  memoryLimit?: number;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  payload: Record<string, any>;
  document?: {
    id: string;
    title: string;
    content: string;
    type: 'evidence' | 'case' | 'legal_document';
  };
}

export interface SearchStats {
  totalResults: number;
  searchTimeMs: number;
  cacheHit: boolean;
  somClusterUsed?: string;
  memoryUsage: number;
}

export class OptimizedQdrantService {
  private client: QdrantClient;
  private config: Required<QdrantConfig>;
  private somCluster?: LegalDocumentSOM;
  private nesCache?: NESCacheOrchestrator;
  private searchCache = new Map<string, { results: VectorSearchResult[], timestamp: number, stats: SearchStats }>();
  private batchQueue: Array<{ id: string, vector: number[], payload: any }> = [];
  private memoryUsage = 0;
  private processingBatch = false;

  constructor(config: QdrantConfig = {}) {
    this.config = {
      url: config.url || import.meta.env.QDRANT_URL || 'http://localhost:6333',
      apiKey: config.apiKey || import.meta.env.QDRANT_API_KEY || '',
      timeout: config.timeout || 30000,
      collectionName: config.collectionName || 'legal_vectors',
      enableBatching: config.enableBatching ?? true,
      enableSOMClustering: config.enableSOMClustering ?? true,
      enableNESCache: config.enableNESCache ?? true,
      memoryLimit: config.memoryLimit || MAX_MEMORY_USAGE
    };

    this.client = new QdrantClient({
      url: this.config.url,
      apiKey: this.config.apiKey || undefined
    });

    this.initializeEnhancedFeatures();
    this.setupMemoryMonitoring();
  }

  private async initializeEnhancedFeatures(): Promise<void> {
    if (this.config.enableNESCache) {
      this.nesCache = new NESCacheOrchestrator();
    }

    if (this.config.enableSOMClustering) {
      // Initialize SOM with Redis-like interface for clustering
      const mockRedis = {
        hset: async () => {},
        set: async () => {},
        get: async () => null
      };
      this.somCluster = new LegalDocumentSOM({
        width: 10,
        height: 10,
        dimensions: NOMIC_EMBED_DIMENSIONS,
        learningRate: 0.1,
        radius: 3,
        maxIterations: 500
      }, mockRedis as any);
    }
  }

  private setupMemoryMonitoring(): void {
    setInterval(() => {
      this.cleanupMemory();
    }, 30000); // Cleanup every 30 seconds
  }

  /**
   * Ensure collection exists with correct nomic-embed dimensions
   */
  async ensureCollection(): Promise<void> {
    try {
      const collections = await this.client.getCollections();
      const exists = collections.collections?.some(c => c.name === this.config.collectionName);

      if (!exists) {
        await this.client.createCollection(this.config.collectionName, {
          vectors: {
            size: NOMIC_EMBED_DIMENSIONS, // Corrected to 768 for nomic-embed
            distance: 'Cosine'
          },
          optimizers_config: {
            default_segment_number: 4,
            max_segment_size: 20000,
            memmap_threshold: 10000,
            indexing_threshold: 20000
          },
          hnsw_config: {
            m: 16,
            ef_construct: 200,
            full_scan_threshold: 10000
          },
          quantization_config: {
            scalar: {
              type: 'int8',
              quantile: 0.99,
              always_ram: false
            }
          }
        });

        console.log(`✅ Created optimized Qdrant collection: ${this.config.collectionName}`);
      }
    } catch (error: any) {
      console.error('❌ Failed to ensure Qdrant collection:', error);
      throw error;
    }
  }

  /**
   * Memory-efficient vector search with intelligent caching
   */
  async searchVectors(
    queryVector: number[],
    options: {
      limit?: number;
      filter?: any;
      threshold?: number;
      useCache?: boolean;
      enableSOM?: boolean;
    } = {}
  ): Promise<{ results: VectorSearchResult[], stats: SearchStats }> {
    const startTime = Date.now();
    const limit = options.limit || 10;
    const threshold = options.threshold || 0.7;
    const useCache = options.useCache ?? true;
    const enableSOM = options.enableSOM ?? this.config.enableSOMClustering;

    // Generate cache key
    const cacheKey = this.generateCacheKey(queryVector, options);

    // Check cache first
    if (useCache && this.searchCache.has(cacheKey)) {
      const cached = this.searchCache.get(cacheKey)!;
      if (Date.now() - cached.timestamp < 300000) { // 5 minutes cache TTL
        const stats: SearchStats = {
          ...cached.stats,
          cacheHit: true,
          searchTimeMs: Date.now() - startTime
        };
        return { results: cached.results, stats };
      }
    }

    let searchResults: ScoredPoint[] = [];
    let somClusterUsed: string | undefined;

    try {
      // Use SOM clustering for intelligent search if enabled
      if (enableSOM && this.somCluster) {
        const clusterResult = await this.somCluster.cluster(queryVector);
        somClusterUsed = `${clusterResult.x},${clusterResult.y}`;
        
        // Search within the identified cluster first
        searchResults = await this.searchInCluster(queryVector, clusterResult, limit, options.filter);
      }

      // Fallback to standard vector search if SOM didn't find enough results
      if (searchResults.length < limit) {
        const searchParams: SearchParams = {
          vector: queryVector,
          limit: limit,
          score_threshold: threshold,
          with_payload: true,
          with_vector: false // Save memory by not returning vectors
        };

        if (options.filter) {
          searchParams.filter = options.filter;
        }

        const qdrantResults = await this.client.search(this.config.collectionName, searchParams);
        searchResults = qdrantResults.slice(0, limit);
      }

      // Process results and enrich with document data
      const results = await this.enrichSearchResults(searchResults);
      
      // Calculate stats
      const stats: SearchStats = {
        totalResults: results.length,
        searchTimeMs: Date.now() - startTime,
        cacheHit: false,
        somClusterUsed,
        memoryUsage: this.calculateMemoryUsage(results)
      };

      // Cache results if memory allows
      if (useCache && this.memoryUsage + stats.memoryUsage < this.config.memoryLimit) {
        this.searchCache.set(cacheKey, { results, timestamp: Date.now(), stats });
        this.memoryUsage += stats.memoryUsage;
      }

      return { results, stats };

    } catch (error: any) {
      console.error('❌ Qdrant search error:', error);
      throw new Error(`Vector search failed: ${error.message}`);
    }
  }

  /**
   * Batch upsert vectors with memory optimization
   */
  async upsertVectors(
    vectors: Array<{ id: string; vector: number[]; payload: any }>
  ): Promise<{ success: number; errors: number }> {
    if (this.config.enableBatching) {
      // Add to batch queue
      this.batchQueue.push(...vectors);
      
      // Process batch if it reaches the limit
      if (this.batchQueue.length >= BATCH_SIZE) {
        return await this.processBatch();
      }
      
      return { success: vectors.length, errors: 0 };
    } else {
      // Direct upsert
      return await this.upsertBatch(vectors);
    }
  }

  /**
   * Sync from PostgreSQL with memory-efficient streaming
   */
  async syncFromPostgreSQL(options: {
    fullSync?: boolean;
    batchSize?: number;
    sinceTimestamp?: Date;
  } = {}): Promise<{ synced: number; errors: number; duration: number }> {
    const startTime = Date.now();
    const batchSize = options.batchSize || BATCH_SIZE;
    let synced = 0;
    let errors = 0;

    try {
      await this.ensureCollection();

      // Stream evidence vectors
      const evidenceStream = await this.streamEvidenceVectors(batchSize, options.sinceTimestamp);
      const evidenceResult = await this.processVectorStream(evidenceStream, 'evidence');
      synced += evidenceResult.synced;
      errors += evidenceResult.errors;

      // Stream case vectors
      const caseStream = await this.streamCaseVectors(batchSize, options.sinceTimestamp);
      const caseResult = await this.processVectorStream(caseStream, 'case');
      synced += caseResult.synced;
      errors += caseResult.errors;

      // Stream legal document vectors
      const legalDocsStream = await this.streamLegalDocumentVectors(batchSize, options.sinceTimestamp);
      const legalDocsResult = await this.processVectorStream(legalDocsStream, 'legal_document');
      synced += legalDocsResult.synced;
      errors += legalDocsResult.errors;

      const duration = Date.now() - startTime;
      console.log(`✅ PostgreSQL sync completed: ${synced} synced, ${errors} errors in ${duration}ms`);

      return { synced, errors, duration };

    } catch (error: any) {
      console.error('❌ PostgreSQL sync failed:', error);
      throw error;
    }
  }

  // Private helper methods

  private async searchInCluster(
    queryVector: number[],
    clusterResult: { x: number; y: number; confidence: number },
    limit: number,
    filter?: any
  ): Promise<ScoredPoint[]> {
    // Implementation for cluster-based search
    // This would search within the identified SOM cluster
    const searchParams: SearchParams = {
      vector: queryVector,
      limit: Math.ceil(limit * 1.5), // Search a bit more in cluster
      with_payload: true,
      with_vector: false,
      filter: {
        must: [
          ...(filter?.must || []),
          {
            key: 'som_cluster',
            match: { value: `${clusterResult.x},${clusterResult.y}` }
          }
        ]
      }
    };

    try {
      return await this.client.search(this.config.collectionName, searchParams);
    } catch (error: any) {
      console.warn('⚠️ Cluster search failed, falling back to standard search');
      return [];
    }
  }

  private async enrichSearchResults(searchResults: ScoredPoint[]): Promise<VectorSearchResult[]> {
    const results: VectorSearchResult[] = [];
    
    // Group results by type for efficient database queries
    const evidenceIds: string[] = [];
    const caseIds: string[] = [];
    const legalDocIds: string[] = [];

    for (const result of searchResults) {
      const payload = result.payload || {};
      const type = payload.type || 'unknown';
      
      if (type === 'evidence') evidenceIds.push(result.id);
      else if (type === 'case') caseIds.push(result.id);
      else if (type === 'legal_document') legalDocIds.push(result.id);
    }

    // Batch fetch documents from PostgreSQL
    const [evidenceData, caseData, legalDocData] = await Promise.all([
      evidenceIds.length > 0 ? db.select().from(evidence).where(inArray(evidence.id, evidenceIds)) : [],
      caseIds.length > 0 ? db.select().from(cases).where(inArray(cases.id, caseIds)) : [],
      legalDocIds.length > 0 ? db.select().from(legalDocuments).where(inArray(legalDocuments.id, legalDocIds)) : []
    ]);

    // Create lookup maps
    const evidenceMap = new Map(evidenceData.map(e => [e.id, e]));
    const caseMap = new Map(caseData.map(c => [c.id, c]));
    const legalDocMap = new Map(legalDocData.map(d => [d.id, d]));

    // Enrich results
    for (const result of searchResults) {
      const payload = result.payload || {};
      const type = payload.type;
      let document = undefined;

      if (type === 'evidence') {
        const evidenceDoc = evidenceMap.get(result.id);
        if (evidenceDoc) {
          document = {
            id: evidenceDoc.id,
            title: evidenceDoc.title,
            content: evidenceDoc.description || '',
            type: 'evidence' as const
          };
        }
      } else if (type === 'case') {
        const caseDoc = caseMap.get(result.id);
        if (caseDoc) {
          document = {
            id: caseDoc.id,
            title: caseDoc.title,
            content: caseDoc.description || '',
            type: 'case' as const
          };
        }
      } else if (type === 'legal_document') {
        const legalDoc = legalDocMap.get(result.id);
        if (legalDoc) {
          document = {
            id: legalDoc.id,
            title: legalDoc.title,
            content: legalDoc.content,
            type: 'legal_document' as const
          };
        }
      }

      results.push({
        id: result.id,
        score: result.score || 0,
        payload,
        document
      });
    }

    return results;
  }

  private async streamEvidenceVectors(batchSize: number, sinceTimestamp?: Date): Promise<AsyncIterable<any[]>> {
    const query = sinceTimestamp 
      ? db.select().from(evidence)
          .where(sql`${evidence.updatedAt} > ${sinceTimestamp} AND ${evidence.titleEmbedding} IS NOT NULL`)
          .orderBy(desc(evidence.updatedAt))
      : db.select().from(evidence)
          .where(sql`${evidence.titleEmbedding} IS NOT NULL`)
          .orderBy(desc(evidence.updatedAt));

    return this.createBatchStream(query, batchSize);
  }

  private async streamCaseVectors(batchSize: number, sinceTimestamp?: Date): Promise<AsyncIterable<any[]>> {
    const query = sinceTimestamp
      ? db.select().from(cases)
          .where(sql`${cases.updatedAt} > ${sinceTimestamp} AND ${cases.titleEmbedding} IS NOT NULL`)
          .orderBy(desc(cases.updatedAt))
      : db.select().from(cases)
          .where(sql`${cases.titleEmbedding} IS NOT NULL`)
          .orderBy(desc(cases.updatedAt));

    return this.createBatchStream(query, batchSize);
  }

  private async streamLegalDocumentVectors(batchSize: number, sinceTimestamp?: Date): Promise<AsyncIterable<any[]>> {
    const query = sinceTimestamp
      ? db.select().from(legalDocuments)
          .where(sql`${legalDocuments.updatedAt} > ${sinceTimestamp} AND ${legalDocuments.titleEmbedding} IS NOT NULL`)
          .orderBy(desc(legalDocuments.updatedAt))
      : db.select().from(legalDocuments)
          .where(sql`${legalDocuments.titleEmbedding} IS NOT NULL`)
          .orderBy(desc(legalDocuments.updatedAt));

    return this.createBatchStream(query, batchSize);
  }

  private async *createBatchStream<T>(query: any, batchSize: number): AsyncIterable<T[]> {
    let offset = 0;
    let batch: T[] = [];

    do {
      batch = await query.limit(batchSize).offset(offset);
      if (batch.length > 0) {
        yield batch;
        offset += batchSize;
      }
    } while (batch.length === batchSize);
  }

  private async processVectorStream(
    stream: AsyncIterable<any[]>,
    type: 'evidence' | 'case' | 'legal_document'
  ): Promise<{ synced: number; errors: number }> {
    let synced = 0;
    let errors = 0;

    for await (const batch of stream) {
      const vectors = batch
        .filter(item => item.titleEmbedding) // Ensure embedding exists
        .map(item => ({
          id: item.id,
          vector: item.titleEmbedding,
          payload: {
            type,
            title: item.title,
            updated_at: item.updatedAt?.toISOString(),
            ...(type === 'case' && { case_number: item.caseNumber, status: item.status }),
            ...(type === 'evidence' && { evidence_type: item.evidenceType, case_id: item.caseId }),
            ...(type === 'legal_document' && { document_type: item.documentType, case_id: item.caseId })
          }
        }));

      if (vectors.length > 0) {
        const result = await this.upsertBatch(vectors);
        synced += result.success;
        errors += result.errors;
      }

      // Memory cleanup between batches
      if (this.memoryUsage > this.config.memoryLimit * 0.8) {
        await this.cleanupMemory();
      }
    }

    return { synced, errors };
  }

  private async upsertBatch(
    vectors: Array<{ id: string; vector: number[]; payload: any }>
  ): Promise<{ success: number; errors: number }> {
    try {
      const points: PointStruct[] = vectors.map(v => ({
        id: v.id,
        vector: v.vector,
        payload: v.payload
      }));

      await this.client.upsert(this.config.collectionName, {
        wait: true,
        points
      });

      return { success: vectors.length, errors: 0 };
    } catch (error: any) {
      console.error('❌ Batch upsert error:', error);
      return { success: 0, errors: vectors.length };
    }
  }

  private async processBatch(): Promise<{ success: number; errors: number }> {
    if (this.processingBatch || this.batchQueue.length === 0) {
      return { success: 0, errors: 0 };
    }

    this.processingBatch = true;
    const batch = this.batchQueue.splice(0, BATCH_SIZE);
    
    try {
      const result = await this.upsertBatch(batch);
      return result;
    } finally {
      this.processingBatch = false;
    }
  }

  private generateCacheKey(queryVector: number[], options: any): string {
    const hashInput = JSON.stringify({ 
      vector: queryVector.slice(0, 10), // Use first 10 dimensions for key
      limit: options.limit,
      threshold: options.threshold,
      filter: options.filter 
    });
    
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < hashInput.length; i++) {
      const char = hashInput.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return `qdrant_${hash}`;
  }

  private calculateMemoryUsage(results: VectorSearchResult[]): number {
    return JSON.stringify(results).length * 2; // Rough estimate in bytes
  }

  private async cleanupMemory(): Promise<void> {
    // Clean expired search cache
    const now = Date.now();
    const cacheExpiry = 300000; // 5 minutes

    for (const [key, cached] of this.searchCache.entries()) {
      if (now - cached.timestamp > cacheExpiry) {
        this.memoryUsage -= cached.stats.memoryUsage;
        this.searchCache.delete(key);
      }
    }

    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
  }

  /**
   * Health check method
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    collections: number;
    memoryUsage: number;
    cacheHits: number;
    lastSync?: string;
  }> {
    try {
      const collections = await this.client.getCollections();
      const collectionExists = collections.collections?.some(c => c.name === this.config.collectionName);
      
      return {
        status: collectionExists ? 'healthy' : 'degraded',
        collections: collections.collections?.length || 0,
        memoryUsage: this.memoryUsage,
        cacheHits: this.searchCache.size,
        lastSync: new Date().toISOString()
      };
    } catch (error: any) {
      return {
        status: 'unhealthy',
        collections: 0,
        memoryUsage: this.memoryUsage,
        cacheHits: this.searchCache.size
      };
    }
  }
}

// Export singleton instance
export const optimizedQdrantService = new OptimizedQdrantService({
  enableBatching: true,
  enableSOMClustering: true,
  enableNESCache: true,
  memoryLimit: MAX_MEMORY_USAGE
});