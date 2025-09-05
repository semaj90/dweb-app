/**
 * Production PGVector Similarity Service
 * Optimized for legal document search with caching and batching
 */

import { db } from '$lib/server/db';
import { sql } from 'drizzle-orm';
import type { Redis } from 'ioredis';

interface SimilarityResult {
  id: string;
  document_id: string;
  text: string;
  distance: number;
  metadata?: Record<string, any>;
}

interface SimilarityOptions {
  limit?: number;
  threshold?: number;
  filter?: Record<string, any>;
  useCache?: boolean;
  cacheKey?: string;
  cacheTTL?: number;
}

export class PGVectorSimilarityService {
  private redis?: Redis;
  private readonly DEFAULT_CACHE_TTL = 300; // 5 minutes

  constructor(redis?: Redis) {
    this.redis = redis;
  }

  /**
   * High-performance similarity search with caching
   */
  async searchByEmbedding(
    queryVec: number[], 
    options: SimilarityOptions = {}
  ): Promise<SimilarityResult[]> {
    const {
      limit = 10,
      threshold = 0.8,
      filter = {},
      useCache = true,
      cacheKey,
      cacheTTL = this.DEFAULT_CACHE_TTL
    } = options;

    // Generate cache key if not provided
    const finalCacheKey = cacheKey || this.generateCacheKey(queryVec, options);

    // Try cache first
    if (useCache && this.redis) {
      const cached = await this.getCachedResults(finalCacheKey);
      if (cached) return cached;
    }

    // Execute similarity query
    const results = await this.executeSimilarityQuery(queryVec, limit, threshold, filter);

    // Cache results
    if (useCache && this.redis && results.length > 0) {
      await this.cacheResults(finalCacheKey, results, cacheTTL);
    }

    return results;
  }

  /**
   * Batch similarity search for multiple queries
   */
  async batchSearchByEmbedding(
    queries: Array<{ embedding: number[]; options?: SimilarityOptions }>
  ): Promise<SimilarityResult[][]> {
    // Execute all queries in parallel
    const promises = queries.map(({ embedding, options }) => 
      this.searchByEmbedding(embedding, options)
    );
    
    return await Promise.all(promises);
  }

  /**
   * Legal-specific similarity search with case/document filtering
   */
  async searchLegalDocuments(
    queryVec: number[],
    caseId?: string,
    documentType?: string,
    limit: number = 10
  ): Promise<SimilarityResult[]> {
    const filter = {
      ...(caseId && { case_id: caseId }),
      ...(documentType && { document_type: documentType })
    };

    return this.searchByEmbedding(queryVec, {
      limit,
      threshold: 0.7, // More permissive for legal context
      filter,
      cacheKey: `legal:${caseId}:${documentType}:${this.hashVector(queryVec)}`
    });
  }

  /**
   * Hybrid search: embedding + text similarity + metadata filters
   */
  async hybridSearch(
    queryVec: number[],
    textQuery?: string,
    metadata?: Record<string, any>,
    options: SimilarityOptions = {}
  ): Promise<SimilarityResult[]> {
    const { limit = 10, threshold = 0.8 } = options;

    // Build complex query with multiple similarity measures
    const query = sql`
      WITH vector_similarity AS (
        SELECT 
          id, document_id, text, metadata,
          vector <-> ${queryVec}::vector as vector_distance,
          ${textQuery ? 
            sql`similarity(text, ${textQuery}) as text_similarity` : 
            sql`0 as text_similarity`
          }
        FROM document_chunks
        WHERE vector IS NOT NULL
        ${metadata ? this.buildMetadataFilter(metadata) : sql``}
      ),
      ranked_results AS (
        SELECT *,
          -- Weighted hybrid score: 70% vector, 30% text
          (vector_distance * 0.7 + (1 - text_similarity) * 0.3) as hybrid_score
        FROM vector_similarity
        WHERE vector_distance < ${threshold}
      )
      SELECT id, document_id, text, hybrid_score as distance, metadata
      FROM ranked_results
      ORDER BY hybrid_score ASC
      LIMIT ${limit}
    `;

    const result = await db.execute(query);
    return result.rows as SimilarityResult[];
  }

  /**
   * Semantic clustering for legal document analysis
   */
  async findSemanticClusters(
    embeddings: number[][],
    clusterThreshold: number = 0.3
  ): Promise<Array<{ cluster_id: number; documents: SimilarityResult[] }>> {
    // Use DBSCAN-like clustering via SQL
    const query = sql`
      WITH document_similarities AS (
        SELECT 
          d1.id as doc1_id, d2.id as doc2_id,
          d1.vector <-> d2.vector as distance
        FROM document_chunks d1, document_chunks d2
        WHERE d1.id < d2.id 
          AND d1.vector IS NOT NULL 
          AND d2.vector IS NOT NULL
      ),
      clusters AS (
        -- Simplified clustering logic - in production use proper clustering
        SELECT 
          doc1_id,
          COUNT(*) as cluster_size,
          AVG(distance) as avg_distance
        FROM document_similarities
        WHERE distance < ${clusterThreshold}
        GROUP BY doc1_id
        HAVING COUNT(*) > 1
      )
      SELECT * FROM clusters ORDER BY cluster_size DESC
    `;

    const result = await db.execute(query);
    // Transform to cluster format (simplified)
    return result.rows.map((row: any, index) => ({
      cluster_id: index,
      documents: [] // Would need additional query to populate
    }));
  }

  /**
   * Performance monitoring and analytics
   */
  async getSearchAnalytics(timeWindow: string = '24h'): Promise<{
    total_searches: number;
    avg_results: number;
    cache_hit_rate: number;
    popular_queries: string[];
  }> {
    if (!this.redis) {
      return { total_searches: 0, avg_results: 0, cache_hit_rate: 0, popular_queries: [] };
    }

    const stats = await this.redis.hgetall('pgvector:stats');
    return {
      total_searches: parseInt(stats.total_searches || '0'),
      avg_results: parseFloat(stats.avg_results || '0'),
      cache_hit_rate: parseFloat(stats.cache_hit_rate || '0'),
      popular_queries: JSON.parse(stats.popular_queries || '[]')
    };
  }

  // Private helper methods

  private async executeSimilarityQuery(
    queryVec: number[],
    limit: number,
    threshold: number,
    filter: Record<string, any>
  ): Promise<SimilarityResult[]> {
    const query = sql`
      SELECT 
        id, 
        document_id, 
        text, 
        vector <-> ${queryVec}::vector as distance,
        metadata
      FROM document_chunks
      WHERE vector IS NOT NULL
        AND vector <-> ${queryVec}::vector < ${threshold}
        ${this.buildMetadataFilter(filter)}
      ORDER BY vector <-> ${queryVec}::vector
      LIMIT ${limit}
    `;

    const result = await db.execute(query);
    
    // Update analytics
    await this.updateAnalytics('search_executed', result.rows.length);
    
    return result.rows as SimilarityResult[];
  }

  private buildMetadataFilter(filter: Record<string, any>) {
    if (Object.keys(filter).length === 0) return sql``;
    
    const conditions = Object.entries(filter).map(([key, value]) => 
      sql`metadata->>${key} = ${value}`
    );
    
    return sql`AND ${sql.join(conditions, sql` AND `)}`;
  }

  private generateCacheKey(queryVec: number[], options: SimilarityOptions): string {
    const hash = this.hashVector(queryVec);
    const optionsHash = this.hashObject(options);
    return `pgvector:search:${hash}:${optionsHash}`;
  }

  private hashVector(vec: number[]): string {
    // Simple hash for caching - use crypto.subtle in production
    return Buffer.from(vec.join(',')).toString('base64').slice(0, 16);
  }

  private hashObject(obj: any): string {
    return Buffer.from(JSON.stringify(obj)).toString('base64').slice(0, 8);
  }

  private async getCachedResults(key: string): Promise<SimilarityResult[] | null> {
    if (!this.redis) return null;
    
    try {
      const cached = await this.redis.get(key);
      if (cached) {
        await this.updateAnalytics('cache_hit');
        return JSON.parse(cached);
      }
      await this.updateAnalytics('cache_miss');
      return null;
    } catch (error) {
      console.error('Cache read error:', error);
      return null;
    }
  }

  private async cacheResults(key: string, results: SimilarityResult[], ttl: number): Promise<void> {
    if (!this.redis) return;
    
    try {
      await this.redis.setex(key, ttl, JSON.stringify(results));
    } catch (error) {
      console.error('Cache write error:', error);
    }
  }

  private async updateAnalytics(event: string, value: number = 1): Promise<void> {
    if (!this.redis) return;
    
    try {
      await this.redis.hincrby('pgvector:stats', event, value);
    } catch (error) {
      console.error('Analytics update error:', error);
    }
  }
}

// Export singleton instance
export const pgvectorSimilarity = new PGVectorSimilarityService();

// Export factory for dependency injection
export function createPGVectorService(redis?: Redis): PGVectorSimilarityService {
  return new PGVectorSimilarityService(redis);
}