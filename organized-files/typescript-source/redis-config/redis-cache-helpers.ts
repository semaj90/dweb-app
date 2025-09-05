/**
 * Production Redis Cache Helpers
 * Optimized for legal AI inference pipeline with smart TTLs and compression
 */

import Redis from 'ioredis';
import { createHash } from 'crypto';
import { compress, decompress } from 'lz4';

// Configuration
const REDIS_CONFIG = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3,
  lazyConnect: true,
  keepAlive: 30000,
  family: 4,
  db: 0, // Use database 0 for general cache
};

// TTL Configuration (in seconds)
const TTL_CONFIG = {
  TOKENS: 300,           // 5 minutes - tokenization cache
  EMBEDDINGS: 1800,      // 30 minutes - embedding cache
  RESULTS: 600,          // 10 minutes - inference results
  SIMILARITY: 900,       // 15 minutes - similarity search results
  USER_SESSION: 3600,    // 1 hour - user session data
  DOCUMENT_CHUNKS: 7200, // 2 hours - document processing cache
  LEGAL_ANALYSIS: 1800,  // 30 minutes - legal analysis cache
  PREFETCH: 180,         // 3 minutes - prefetch cache
};

// Cache key prefixes
const CACHE_PREFIXES = {
  TOKENS: 'tokens:',
  EMBEDDINGS: 'embed:',
  RESULTS: 'result:',
  SIMILARITY: 'sim:',
  USER: 'user:',
  DOCUMENT: 'doc:',
  LEGAL: 'legal:',
  PREFETCH: 'prefetch:',
  STATS: 'stats:',
  HEALTH: 'health:',
};

export class RedisCacheManager {
  private redis: Redis;
  private compressionThreshold = 1024; // Compress data > 1KB
  private connected = false;

  constructor(config: Partial<typeof REDIS_CONFIG> = {}) {
    this.redis = new Redis({ ...REDIS_CONFIG, ...config });
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redis.on('connect', () => {
      console.log('Redis connected');
      this.connected = true;
    });

    this.redis.on('ready', () => {
      console.log('Redis ready');
    });

    this.redis.on('error', (error) => {
      console.error('Redis error:', error);
      this.connected = false;
    });

    this.redis.on('close', () => {
      console.log('Redis connection closed');
      this.connected = false;
    });
  }

  async connect(): Promise<boolean> {
    try {
      await this.redis.connect();
      return true;
    } catch (error) {
      console.error('Failed to connect to Redis:', error);
      return false;
    }
  }

  async disconnect(): Promise<void> {
    await this.redis.disconnect();
  }

  isConnected(): boolean {
    return this.connected;
  }

  // Token caching methods
  async cacheTokens(text: string, tokens: number[], model: string = 'default'): Promise<void> {
    const key = this.generateTokenCacheKey(text, model);
    await this.setCompressed(key, tokens, TTL_CONFIG.TOKENS);
    await this.incrementStats('cache:tokens:writes');
  }

  async getTokens(text: string, model: string = 'default'): Promise<number[] | null> {
    const key = this.generateTokenCacheKey(text, model);
    const result = await this.getCompressed<number[]>(key);
    
    if (result) {
      await this.incrementStats('cache:tokens:hits');
    } else {
      await this.incrementStats('cache:tokens:misses');
    }
    
    return result;
  }

  // Embedding caching methods
  async cacheEmbedding(
    text: string, 
    embedding: number[], 
    model: string = 'default',
    metadata: Record<string, any> = {}
  ): Promise<void> {
    const key = this.generateEmbeddingCacheKey(text, model);
    const data = {
      embedding,
      model,
      dimensions: embedding.length,
      timestamp: Date.now(),
      metadata
    };
    
    await this.setCompressed(key, data, TTL_CONFIG.EMBEDDINGS);
    await this.incrementStats('cache:embeddings:writes');
  }

  async getEmbedding(text: string, model: string = 'default'): Promise<{
    embedding: number[];
    model: string;
    dimensions: number;
    timestamp: number;
    metadata: Record<string, any>;
  } | null> {
    const key = this.generateEmbeddingCacheKey(text, model);
    const result = await this.getCompressed<any>(key);
    
    if (result) {
      await this.incrementStats('cache:embeddings:hits');
    } else {
      await this.incrementStats('cache:embeddings:misses');
    }
    
    return result;
  }

  // Batch embedding operations
  async batchCacheEmbeddings(
    items: Array<{ text: string; embedding: number[]; metadata?: Record<string, any> }>,
    model: string = 'default'
  ): Promise<void> {
    const pipeline = this.redis.pipeline();
    
    for (const item of items) {
      const key = this.generateEmbeddingCacheKey(item.text, model);
      const data = {
        embedding: item.embedding,
        model,
        dimensions: item.embedding.length,
        timestamp: Date.now(),
        metadata: item.metadata || {}
      };
      
      const serialized = await this.serialize(data);
      pipeline.setex(key, TTL_CONFIG.EMBEDDINGS, serialized);
    }
    
    await pipeline.exec();
    await this.incrementStats('cache:embeddings:batch_writes', items.length);
  }

  async batchGetEmbeddings(
    texts: string[], 
    model: string = 'default'
  ): Promise<Array<{ text: string; embedding: number[] | null }>> {
    const keys = texts.map(text => this.generateEmbeddingCacheKey(text, model));
    const pipeline = this.redis.pipeline();
    
    keys.forEach(key => pipeline.get(key));
    const results = await pipeline.exec();
    
    let hits = 0;
    const output = await Promise.all(
      texts.map(async (text, index) => {
        const result = results?.[index]?.[1];
        if (result) {
          hits++;
          const data = await this.deserialize(result as string);
          return { text, embedding: data?.embedding || null };
        }
        return { text, embedding: null };
      })
    );
    
    await this.incrementStats('cache:embeddings:batch_hits', hits);
    await this.incrementStats('cache:embeddings:batch_misses', texts.length - hits);
    
    return output;
  }

  // Inference result caching
  async cacheInferenceResult(
    prompt: string,
    result: any,
    model: string = 'default',
    temperature: number = 0.8
  ): Promise<void> {
    const key = this.generateResultCacheKey(prompt, model, temperature);
    const data = {
      ...result,
      cached_at: Date.now(),
      model,
      temperature
    };
    
    await this.setCompressed(key, data, TTL_CONFIG.RESULTS);
    await this.incrementStats('cache:results:writes');
  }

  async getInferenceResult(
    prompt: string,
    model: string = 'default',
    temperature: number = 0.8
  ): Promise<any> {
    const key = this.generateResultCacheKey(prompt, model, temperature);
    const result = await this.getCompressed<any>(key);
    
    if (result) {
      await this.incrementStats('cache:results:hits');
      // Update access time
      await this.redis.expire(key, TTL_CONFIG.RESULTS);
    } else {
      await this.incrementStats('cache:results:misses');
    }
    
    return result;
  }

  // Similarity search caching
  async cacheSimilarityResults(
    queryVector: number[],
    results: any[],
    options: Record<string, any> = {}
  ): Promise<void> {
    const key = this.generateSimilarityCacheKey(queryVector, options);
    const data = {
      results,
      query_hash: this.hashVector(queryVector),
      options,
      cached_at: Date.now(),
      result_count: results.length
    };
    
    await this.setCompressed(key, data, TTL_CONFIG.SIMILARITY);
    await this.incrementStats('cache:similarity:writes');
  }

  async getSimilarityResults(
    queryVector: number[],
    options: Record<string, any> = {}
  ): Promise<any[] | null> {
    const key = this.generateSimilarityCacheKey(queryVector, options);
    const cached = await this.getCompressed<any>(key);
    
    if (cached) {
      await this.incrementStats('cache:similarity:hits');
      return cached.results;
    } else {
      await this.incrementStats('cache:similarity:misses');
      return null;
    }
  }

  // Legal-specific caching methods
  async cacheLegalAnalysis(
    documentId: string,
    analysisType: string,
    analysis: any,
    caseId?: string
  ): Promise<void> {
    const key = `${CACHE_PREFIXES.LEGAL}${analysisType}:${documentId}${caseId ? `:${caseId}` : ''}`;
    const data = {
      ...analysis,
      document_id: documentId,
      analysis_type: analysisType,
      case_id: caseId,
      cached_at: Date.now()
    };
    
    await this.setCompressed(key, data, TTL_CONFIG.LEGAL_ANALYSIS);
    await this.incrementStats('cache:legal:writes');
  }

  async getLegalAnalysis(
    documentId: string,
    analysisType: string,
    caseId?: string
  ): Promise<any> {
    const key = `${CACHE_PREFIXES.LEGAL}${analysisType}:${documentId}${caseId ? `:${caseId}` : ''}`;
    const result = await this.getCompressed<any>(key);
    
    if (result) {
      await this.incrementStats('cache:legal:hits');
    } else {
      await this.incrementStats('cache:legal:misses');
    }
    
    return result;
  }

  // Prefetch management
  async cachePrefetch(
    prompt: string,
    result: any,
    priority: 'low' | 'medium' | 'high' = 'low'
  ): Promise<void> {
    const key = `${CACHE_PREFIXES.PREFETCH}${this.hashText(prompt)}:${priority}`;
    await this.setCompressed(key, result, TTL_CONFIG.PREFETCH);
    await this.incrementStats('cache:prefetch:writes');
  }

  async getPrefetch(prompt: string, priority: 'low' | 'medium' | 'high' = 'low'): Promise<any> {
    const key = `${CACHE_PREFIXES.PREFETCH}${this.hashText(prompt)}:${priority}`;
    const result = await this.getCompressed<any>(key);
    
    if (result) {
      await this.incrementStats('cache:prefetch:hits');
    } else {
      await this.incrementStats('cache:prefetch:misses');
    }
    
    return result;
  }

  // Cache management and statistics
  async clearCache(pattern: string = '*'): Promise<number> {
    const keys = await this.redis.keys(pattern);
    if (keys.length === 0) return 0;
    
    const deleted = await this.redis.del(...keys);
    await this.incrementStats('cache:clears', deleted);
    return deleted;
  }

  async getCacheStats(): Promise<Record<string, number>> {
    const keys = await this.redis.keys(`${CACHE_PREFIXES.STATS}*`);
    const pipeline = this.redis.pipeline();
    
    keys.forEach(key => pipeline.get(key));
    const results = await pipeline.exec();
    
    const stats: Record<string, number> = {};
    keys.forEach((key, index) => {
      const statName = key.replace(CACHE_PREFIXES.STATS, '');
      const value = results?.[index]?.[1];
      stats[statName] = parseInt(value as string) || 0;
    });
    
    return stats;
  }

  async getCacheHealth(): Promise<{
    connected: boolean;
    memory_usage: string;
    total_keys: number;
    cache_stats: Record<string, number>;
    uptime: number;
  }> {
    if (!this.connected) {
      return {
        connected: false,
        memory_usage: '0',
        total_keys: 0,
        cache_stats: {},
        uptime: 0
      };
    }
    
    const info = await this.redis.info('memory');
    const dbSize = await this.redis.dbsize();
    const stats = await this.getCacheStats();
    
    const memoryMatch = info.match(/used_memory_human:(.+)/);
    const memoryUsage = memoryMatch ? memoryMatch[1].trim() : 'unknown';
    
    return {
      connected: true,
      memory_usage: memoryUsage,
      total_keys: dbSize,
      cache_stats: stats,
      uptime: Date.now() // Simplified
    };
  }

  // Private helper methods
  private async setCompressed(key: string, data: any, ttl: number): Promise<void> {
    const serialized = await this.serialize(data);
    await this.redis.setex(key, ttl, serialized);
  }

  private async getCompressed<T>(key: string): Promise<T | null> {
    const data = await this.redis.get(key);
    if (!data) return null;
    
    return await this.deserialize<T>(data);
  }

  private async serialize(data: any): Promise<string> {
    const json = JSON.stringify(data);
    
    // Compress if data is large enough
    if (json.length > this.compressionThreshold) {
      try {
        const compressed = compress(Buffer.from(json, 'utf8'));
        return `lz4:${compressed.toString('base64')}`;
      } catch (error) {
        console.warn('Compression failed, storing uncompressed:', error);
        return json;
      }
    }
    
    return json;
  }

  private async deserialize<T>(data: string): Promise<T | null> {
    try {
      // Check if data is compressed
      if (data.startsWith('lz4:')) {
        const compressed = Buffer.from(data.slice(4), 'base64');
        const decompressed = decompress(compressed);
        return JSON.parse(decompressed.toString('utf8'));
      }
      
      return JSON.parse(data);
    } catch (error) {
      console.error('Deserialization failed:', error);
      return null;
    }
  }

  private generateTokenCacheKey(text: string, model: string): string {
    return `${CACHE_PREFIXES.TOKENS}${model}:${this.hashText(text)}`;
  }

  private generateEmbeddingCacheKey(text: string, model: string): string {
    return `${CACHE_PREFIXES.EMBEDDINGS}${model}:${this.hashText(text)}`;
  }

  private generateResultCacheKey(prompt: string, model: string, temperature: number): string {
    const hash = this.hashText(`${prompt}:${model}:${temperature}`);
    return `${CACHE_PREFIXES.RESULTS}${hash}`;
  }

  private generateSimilarityCacheKey(vector: number[], options: Record<string, any>): string {
    const vectorHash = this.hashVector(vector);
    const optionsHash = this.hashText(JSON.stringify(options));
    return `${CACHE_PREFIXES.SIMILARITY}${vectorHash}:${optionsHash}`;
  }

  private hashText(text: string): string {
    return createHash('sha256').update(text).digest('hex').slice(0, 16);
  }

  private hashVector(vector: number[]): string {
    const text = vector.map(n => n.toFixed(6)).join(',');
    return this.hashText(text);
  }

  private async incrementStats(key: string, increment: number = 1): Promise<void> {
    try {
      await this.redis.incrby(`${CACHE_PREFIXES.STATS}${key}`, increment);
    } catch (error) {
      // Fail silently for stats
    }
  }
}

// Export singleton instance
export const redisCacheManager = new RedisCacheManager();

// Initialization helper
export async function initializeRedisCache(config?: Partial<typeof REDIS_CONFIG>): Promise<boolean> {
  if (config) {
    // Create new instance with custom config
    const customCache = new RedisCacheManager(config);
    return await customCache.connect();
  }
  
  return await redisCacheManager.connect();
}

// Export types for TypeScript
export type CacheKey = keyof typeof CACHE_PREFIXES;
export type CacheStats = Record<string, number>;
export type CacheHealth = {
  connected: boolean;
  memory_usage: string;
  total_keys: number;
  cache_stats: CacheStats;
  uptime: number;
};

// Factory function for dependency injection
export function createRedisCacheManager(config?: Partial<typeof REDIS_CONFIG>): RedisCacheManager {
  return new RedisCacheManager(config);
}