// Hybrid Vector Operations: PostgreSQL pgvector + Qdrant Integration
// Best practices implementation with fallback and performance optimization

import type { Database } from '$lib/server/db/index.js';
import type { SQL } from 'drizzle-orm';
import type { LegalDocuments } from '$lib/server/db/unified-schema.js';
import type { SimilarityResult } from '$lib/server/db/vector-operations.js';

// ===== INTERFACES =====

export interface HybridSearchOptions {
  threshold: number;
  limit: number;
  useQdrant?: boolean;
  usePgVector?: boolean;
  hybridWeights?: {
    pgvector: number;
    qdrant: number;
  };
  includeMetadata?: boolean;
}

export interface VectorSearchResult {
  id: string;
  content: string;
  title?: string;
  similarity: number;
  source: 'pgvector' | 'qdrant' | 'hybrid';
  metadata?: {
    keywords?: string[];
    topics?: string[];
    documentType?: string;
    category?: string;
    [key: string]: any;
  };
}

export interface QdrantPoint {
  id: string;
  vector: number[];
  payload: Record<string, any>;
}

// ===== QDRANT CLIENT =====

export class QdrantClient {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl = 'http://localhost:6333', apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async request(method: string, endpoint: string, data?: any): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['api-key'] = this.apiKey;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers,
      body: data ? JSON.stringify(data) : undefined,
    });

    if (!response.ok) {
      throw new Error(`Qdrant request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async search(
    collection: string,
    vector: number[],
    limit: number = 10,
    scoreThreshold?: number
  ): Promise<QdrantPoint[]> {
    const searchParams: any = {
      vector,
      limit,
      with_payload: true,
      with_vector: false
    };

    if (scoreThreshold !== undefined) {
      searchParams.score_threshold = scoreThreshold;
    }

    const response = await this.request('POST', `/collections/${collection}/points/search`, searchParams);
    
    return response.result?.map((hit: any) => ({
      id: hit.id,
      vector: hit.vector || [],
      payload: hit.payload || {},
      score: hit.score
    })) || [];
  }

  async upsert(collection: string, points: QdrantPoint[]): Promise<void> {
    await this.request('PUT', `/collections/${collection}/points`, {
      points: points.map(point => ({
        id: point.id,
        vector: point.vector,
        payload: point.payload
      }))
    });
  }

  async createCollection(
    collection: string,
    vectorSize: number,
    distance: 'Cosine' | 'Euclidean' | 'Dot' = 'Cosine'
  ): Promise<void> {
    try {
      await this.request('PUT', `/collections/${collection}`, {
        vectors: {
          size: vectorSize,
          distance
        }
      });
    } catch (error) {
      // Collection might already exist
      if (!error.message.includes('already exists')) {
        throw error;
      }
    }
  }

  async collectionInfo(collection: string): Promise<any> {
    return this.request('GET', `/collections/${collection}`);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/collections`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

// ===== HYBRID VECTOR SERVICE =====

export class HybridVectorService {
  private qdrantClient: QdrantClient;
  private defaultCollection = 'legal_documents';
  private vectorDimensions = 384; // nomic-embed-text dimensions

  constructor() {
    this.qdrantClient = new QdrantClient();
    this.initializeCollections();
  }

  private async initializeCollections() {
    try {
      const isHealthy = await this.qdrantClient.healthCheck();
      if (isHealthy) {
        await this.qdrantClient.createCollection(
          this.defaultCollection, 
          this.vectorDimensions, 
          'Cosine'
        );
      }
    } catch (error) {
      console.error('Failed to initialize Qdrant collections:', error);
    }
  }

  // ===== HYBRID SEARCH METHODS =====

  async hybridVectorSearch(
    queryEmbedding: number[],
    options: HybridSearchOptions = {
      threshold: 0.7,
      limit: 10,
      useQdrant: true,
      usePgVector: true,
      hybridWeights: { pgvector: 0.6, qdrant: 0.4 }
    }
  ): Promise<VectorSearchResult[]> {
    const results: VectorSearchResult[] = [];

    // Execute searches in parallel
    const searchPromises = [];

    if (options.usePgVector !== false) {
      searchPromises.push(this.searchPgVector(queryEmbedding, options));
    }

    if (options.useQdrant !== false) {
      searchPromises.push(this.searchQdrant(queryEmbedding, options));
    }

    const [pgResults, qdrantResults] = await Promise.allSettled(searchPromises);

    // Process pgvector results
    if (pgResults.status === 'fulfilled') {
      const weighted = pgResults.value.map(result => ({
        ...result,
        similarity: result.similarity * (options.hybridWeights?.pgvector || 0.6),
        source: 'pgvector' as const
      }));
      results.push(...weighted);
    }

    // Process Qdrant results
    if (qdrantResults.status === 'fulfilled') {
      const weighted = qdrantResults.value.map(result => ({
        ...result,
        similarity: result.similarity * (options.hybridWeights?.qdrant || 0.4),
        source: 'qdrant' as const
      }));
      results.push(...weighted);
    }

    // Merge and deduplicate results
    const mergedResults = this.mergeAndDeduplicateResults(results);

    // Sort by similarity and return top results
    return mergedResults
      .filter(result => result.similarity >= options.threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, options.limit);
  }

  private async searchPgVector(
    queryEmbedding: number[],
    options: HybridSearchOptions
  ): Promise<VectorSearchResult[]> {
    try {
      const vectorString = `[${queryEmbedding.join(',')}]`;
      
      const results = await db.execute(sql`
        SELECT 
          id,
          title,
          content,
          1 - (embedding <=> ${vectorString}::vector) as similarity,
          keywords,
          topics,
          metadata
        FROM legal_documents 
        WHERE embedding IS NOT NULL 
          AND 1 - (embedding <=> ${vectorString}::vector) > ${options.threshold}
        ORDER BY embedding <=> ${vectorString}::vector
        LIMIT ${Math.ceil(options.limit * 1.5)}
      `);

      return results.map((row: any) => ({
        id: row.id,
        content: row.content || '',
        title: row.title || '',
        similarity: parseFloat(row.similarity || '0'),
        source: 'pgvector' as const,
        metadata: {
          keywords: this.parseArrayField(row.keywords),
          topics: this.parseArrayField(row.topics),
          ...((typeof row.metadata === 'object' ? row.metadata : {}) || {})
        }
      }));
    } catch (error) {
      console.error('PgVector search failed:', error);
      return [];
    }
  }

  private async searchQdrant(
    queryEmbedding: number[],
    options: HybridSearchOptions
  ): Promise<VectorSearchResult[]> {
    try {
      const points = await this.qdrantClient.search(
        this.defaultCollection,
        queryEmbedding,
        Math.ceil(options.limit * 1.5),
        options.threshold
      );

      return points.map(point => ({
        id: point.id,
        content: point.payload.content || '',
        title: point.payload.title || '',
        similarity: point.score || 0,
        source: 'qdrant' as const,
        metadata: {
          keywords: point.payload.keywords || [],
          topics: point.payload.topics || [],
          documentType: point.payload.documentType,
          category: point.payload.category,
          ...point.payload
        }
      }));
    } catch (error) {
      console.error('Qdrant search failed:', error);
      return [];
    }
  }

  private mergeAndDeduplicateResults(results: VectorSearchResult[]): VectorSearchResult[] {
    const seen = new Map<string, VectorSearchResult>();

    for (const result of results) {
      if (seen.has(result.id)) {
        // If we have a duplicate, keep the one with higher similarity
        const existing = seen.get(result.id)!;
        if (result.similarity > existing.similarity) {
          seen.set(result.id, {
            ...result,
            source: 'hybrid' as const,
            similarity: Math.max(result.similarity, existing.similarity)
          });
        }
      } else {
        seen.set(result.id, result);
      }
    }

    return Array.from(seen.values());
  }

  private parseArrayField(field: any): string[] {
    if (Array.isArray(field)) {
      return field.map(String);
    }
    if (typeof field === 'string') {
      try {
        const parsed = JSON.parse(field);
        return Array.isArray(parsed) ? parsed.map(String) : [String(parsed)];
      } catch {
        return field.split(',').map(s => s.trim()).filter(Boolean);
      }
    }
    return [];
  }

  // ===== DATA SYNCHRONIZATION =====

  async syncToQdrant(documents: Array<{
    id: string;
    content: string;
    title?: string;
    embedding: number[];
    metadata?: Record<string, any>;
  }>): Promise<void> {
    try {
      const points: QdrantPoint[] = documents.map(doc => ({
        id: doc.id,
        vector: doc.embedding,
        payload: {
          content: doc.content,
          title: doc.title,
          ...doc.metadata
        }
      }));

      await this.qdrantClient.upsert(this.defaultCollection, points);
    } catch (error) {
      console.error('Failed to sync to Qdrant:', error);
      throw error;
    }
  }

  async syncFromPgVector(): Promise<void> {
    try {
      const results = await db.execute(sql`
        SELECT id, title, content, embedding, keywords, topics, metadata
        FROM legal_documents 
        WHERE embedding IS NOT NULL
        LIMIT 1000
      `);

      const documents = results.map((row: any) => ({
        id: row.id,
        content: row.content || '',
        title: row.title || '',
        embedding: this.parseEmbedding(row.embedding),
        metadata: {
          keywords: this.parseArrayField(row.keywords),
          topics: this.parseArrayField(row.topics),
          ...((typeof row.metadata === 'object' ? row.metadata : {}) || {})
        }
      }));

      await this.syncToQdrant(documents);
    } catch (error) {
      console.error('Failed to sync from PgVector:', error);
      throw error;
    }
  }

  private parseEmbedding(embedding: any): number[] {
    if (Array.isArray(embedding)) {
      return embedding.map(Number);
    }
    if (typeof embedding === 'string') {
      try {
        // Handle pgvector format: [1,2,3] or "1,2,3"
        const cleaned = embedding.replace(/^\[|\]$/g, '');
        return cleaned.split(',').map(s => parseFloat(s.trim()));
      } catch {
        return [];
      }
    }
    return [];
  }

  // ===== HEALTH & MONITORING =====

  async getSystemHealth(): Promise<{
    pgvector: boolean;
    qdrant: boolean;
    hybrid: boolean;
    collections: Record<string, any>;
  }> {
    const health = {
      pgvector: false,
      qdrant: false,
      hybrid: false,
      collections: {}
    };

    try {
      // Test pgvector
      await db.execute(sql`SELECT 1::vector;`);
      health.pgvector = true;
    } catch {
      // pgvector not available
    }

    try {
      // Test Qdrant
      health.qdrant = await this.qdrantClient.healthCheck();
      
      if (health.qdrant) {
        health.collections[this.defaultCollection] = await this.qdrantClient.collectionInfo(this.defaultCollection);
      }
    } catch {
      // Qdrant not available
    }

    health.hybrid = health.pgvector || health.qdrant;

    return health;
  }

  async getCollectionStats(): Promise<{
    pgvector: { count: number; avgSimilarity: number };
    qdrant: { count: number; vectorSize: number };
  }> {
    const stats = {
      pgvector: { count: 0, avgSimilarity: 0 },
      qdrant: { count: 0, vectorSize: 0 }
    };

    try {
      const pgResult = await db.execute(sql`
        SELECT COUNT(*) as count
        FROM legal_documents 
        WHERE embedding IS NOT NULL
      `);
      stats.pgvector.count = parseInt(pgResult[0]?.count || '0');
    } catch (error) {
      console.error('Failed to get pgvector stats:', error);
    }

    try {
      const qdrantInfo = await this.qdrantClient.collectionInfo(this.defaultCollection);
      stats.qdrant.count = qdrantInfo.result?.points_count || 0;
      stats.qdrant.vectorSize = qdrantInfo.result?.config?.params?.vectors?.size || 0;
    } catch (error) {
      console.error('Failed to get Qdrant stats:', error);
    }

    return stats;
  }
}

// ===== SINGLETON INSTANCE =====

export const hybridVectorService = new HybridVectorService();

// ===== CONVENIENCE FUNCTIONS =====

export async function hybridSearch(
  queryEmbedding: number[],
  options?: HybridSearchOptions
): Promise<VectorSearchResult[]> {
  return hybridVectorService.hybridVectorSearch(queryEmbedding, options);
}

export async function syncVectorData(): Promise<void> {
  return hybridVectorService.syncFromPgVector();
}

export async function getVectorSystemHealth() {
  return hybridVectorService.getSystemHealth();
}