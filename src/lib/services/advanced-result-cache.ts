/**
 * Advanced Result Cache with Memoization for Legal AI Platform
 * Implements SHA256-based result caching with LRU memory + Redis fallback
 * Optimized for deterministic legal AI computations and GPU operations
 */
import { createHash } from 'crypto';
import { LRUCache } from 'lru-cache';
import Redis from 'ioredis';
import { performance } from 'perf_hooks';
import { EventEmitter } from 'events';
import * as cbor from 'cbor';
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack';

export interface CacheConfig {
  maxMemoryItems: number;
  maxMemoryMB: number;
  redisPrefix: string;
  defaultTTL: number; // seconds
  compressionEnabled: boolean;
  sha256Salt: string;
}

export interface CacheEntry<T = any> {
  result: T;
  timestamp: number;
  ttl: number;
  metadata: {
    taskType: string;
    inputHash: string;
    computeTimeMs: number;
    hitCount: number;
    compressionRatio?: number;
    fromGPU?: boolean;
  };
}

export interface CacheStats {
  memory: {
    size: number;
    hits: number;
    misses: number;
    hitRate: number;
  };
  redis: {
    hits: number;
    misses: number;
    hitRate: number;
  };
  overall: {
    totalHits: number;
    totalMisses: number;
    hitRate: number;
    averageComputeTime: number;
    totalComputeTimeSaved: number;
  };
  taskTypes: Record<string, {
    hits: number;
    misses: number;
    averageComputeTime: number;
  }>;
}

export class AdvancedResultCache extends EventEmitter {
  private memoryCache: LRUCache<string, CacheEntry>;
  private redis: Redis;
  private stats: CacheStats;
  private config: CacheConfig;

  constructor(config: Partial<CacheConfig> = {}, redisUrl: string = 'redis://localhost:6379') {
    super();
    
    this.config = {
      maxMemoryItems: 1000,
      maxMemoryMB: 256,
      redisPrefix: 'legal_ai_cache',
      defaultTTL: 3600, // 1 hour
      compressionEnabled: true,
      sha256Salt: 'legal_ai_salt_2024',
      ...config
    };

    // Initialize LRU cache with size limits
    this.memoryCache = new LRUCache({
      max: this.config.maxMemoryItems,
      maxSize: this.config.maxMemoryMB * 1024 * 1024, // Convert to bytes
      sizeCalculation: (entry: CacheEntry) => {
        return JSON.stringify(entry.result).length;
      },
      ttl: this.config.defaultTTL * 1000, // Convert to milliseconds
      allowStale: false,
      updateAgeOnGet: true,
      updateAgeOnHas: true
    });

    // Initialize Redis
    this.redis = new Redis(redisUrl);

    // Initialize stats
    this.stats = {
      memory: { size: 0, hits: 0, misses: 0, hitRate: 0 },
      redis: { hits: 0, misses: 0, hitRate: 0 },
      overall: { totalHits: 0, totalMisses: 0, hitRate: 0, averageComputeTime: 0, totalComputeTimeSaved: 0 },
      taskTypes: {}
    };

    // Update stats periodically
    setInterval(() => this.updateStats(), 30000);
  }

  /**
   * Generate SHA256 hash for input with task type
   */
  generateInputHash(input: any, taskType: string): string {
    const inputString = typeof input === 'string' ? input : JSON.stringify(input, Object.keys(input).sort());
    const hashInput = `${taskType}:${inputString}:${this.config.sha256Salt}`;
    return createHash('sha256').update(hashInput).digest('hex');
  }

  /**
   * Get cached result with multi-tier fallback (Memory → Redis)
   */
  async get<T = any>(inputHash: string, taskType: string): Promise<CacheEntry<T> | null> {
    const startTime = performance.now();

    // Try memory cache first
    const memoryEntry = this.memoryCache.get(inputHash);
    if (memoryEntry) {
      memoryEntry.metadata.hitCount++;
      this.stats.memory.hits++;
      this.stats.overall.totalHits++;
      
      const retrievalTime = performance.now() - startTime;
      this.emit('cache:hit', {
        source: 'memory',
        inputHash,
        taskType,
        retrievalTime,
        savedComputeTime: memoryEntry.metadata.computeTimeMs
      });

      return memoryEntry as CacheEntry<T>;
    }

    this.stats.memory.misses++;

    // Try Redis cache
    try {
      const redisKey = `${this.config.redisPrefix}:${taskType}:${inputHash}`;
      const redisData = await this.redis.get(redisKey);
      
      if (redisData) {
        const entry = await this.deserializeEntry<T>(redisData);
        
        // Promote to memory cache
        this.memoryCache.set(inputHash, entry);
        
        entry.metadata.hitCount++;
        this.stats.redis.hits++;
        this.stats.overall.totalHits++;
        
        const retrievalTime = performance.now() - startTime;
        this.emit('cache:hit', {
          source: 'redis',
          inputHash,
          taskType,
          retrievalTime,
          savedComputeTime: entry.metadata.computeTimeMs
        });

        return entry;
      }
      
      this.stats.redis.misses++;
    } catch (error) {
      this.emit('cache:error', {
        operation: 'redis_get',
        inputHash,
        taskType,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }

    this.stats.overall.totalMisses++;
    
    this.emit('cache:miss', {
      inputHash,
      taskType,
      searchTime: performance.now() - startTime
    });

    return null;
  }

  /**
   * Store result in multi-tier cache (Memory + Redis)
   */
  async set<T = any>(
    inputHash: string, 
    taskType: string, 
    result: T, 
    computeTimeMs: number,
    options: { ttl?: number; fromGPU?: boolean; metadata?: any } = {}
  ): Promise<void> {
    const startTime = performance.now();
    
    const entry: CacheEntry<T> = {
      result,
      timestamp: Date.now(),
      ttl: options.ttl || this.config.defaultTTL,
      metadata: {
        taskType,
        inputHash,
        computeTimeMs,
        hitCount: 0,
        fromGPU: options.fromGPU || false,
        ...options.metadata
      }
    };

    // Store in memory cache
    this.memoryCache.set(inputHash, entry);

    // Store in Redis with compression
    try {
      const serializedEntry = await this.serializeEntry(entry);
      const redisKey = `${this.config.redisPrefix}:${taskType}:${inputHash}`;
      
      if (entry.ttl > 0) {
        await this.redis.setex(redisKey, entry.ttl, serializedEntry);
      } else {
        await this.redis.set(redisKey, serializedEntry);
      }
      
      // Calculate compression ratio if enabled
      if (this.config.compressionEnabled) {
        const originalSize = JSON.stringify(result).length;
        const compressedSize = serializedEntry.length;
        entry.metadata.compressionRatio = originalSize / compressedSize;
      }
      
    } catch (error) {
      this.emit('cache:error', {
        operation: 'redis_set',
        inputHash,
        taskType,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }

    const storeTime = performance.now() - startTime;
    this.emit('cache:store', {
      inputHash,
      taskType,
      storeTime,
      originalComputeTime: computeTimeMs
    });
  }

  /**
   * Compute with automatic caching - main interface for legal AI operations
   */
  async computeWithCache<TInput, TResult>(
    taskType: string,
    input: TInput,
    computeFn: (input: TInput) => Promise<TResult>,
    options: { 
      ttl?: number; 
      forceRecompute?: boolean;
      cacheMetadata?: any;
    } = {}
  ): Promise<{ result: TResult; fromCache: boolean; computeTime: number; cacheStats?: any }> {
    const inputHash = this.generateInputHash(input, taskType);
    const startTime = performance.now();

    // Check cache unless forced recompute
    if (!options.forceRecompute) {
      const cached = await this.get<TResult>(inputHash, taskType);
      if (cached) {
        return {
          result: cached.result,
          fromCache: true,
          computeTime: 0,
          cacheStats: {
            originalComputeTime: cached.metadata.computeTimeMs,
            hitCount: cached.metadata.hitCount,
            age: Date.now() - cached.timestamp
          }
        };
      }
    }

    // Compute result
    const computeStartTime = performance.now();
    const result = await computeFn(input);
    const computeTime = performance.now() - computeStartTime;

    // Cache the result
    await this.set(inputHash, taskType, result, computeTime, {
      ttl: options.ttl,
      metadata: options.cacheMetadata
    });

    // Update task type stats
    if (!this.stats.taskTypes[taskType]) {
      this.stats.taskTypes[taskType] = { hits: 0, misses: 0, averageComputeTime: 0 };
    }
    this.stats.taskTypes[taskType].misses++;
    this.stats.taskTypes[taskType].averageComputeTime = 
      (this.stats.taskTypes[taskType].averageComputeTime + computeTime) / 2;

    return {
      result,
      fromCache: false,
      computeTime,
      cacheStats: { originalComputeTime: computeTime }
    };
  }

  /**
   * Legal AI specific caching methods
   */

  // Document embedding with caching
  async cacheDocumentEmbedding(
    document: { content: string; metadata: any },
    embedFn: (doc: any) => Promise<number[]>
  ): Promise<{ result: number[]; fromCache: boolean; computeTime: number }> {
    return this.computeWithCache(
      'document_embedding',
      { content: document.content, type: document.metadata?.type },
      embedFn,
      { ttl: 86400 } // 24 hours - embeddings are stable
    );
  }

  // Legal entity extraction with caching
  async cacheLegalEntityExtraction(
    text: string,
    extractFn: (text: string) => Promise<any[]>
  ): Promise<{ result: any[]; fromCache: boolean; computeTime: number }> {
    return this.computeWithCache(
      'legal_entity_extraction',
      { text: text.trim(), length: text.length },
      extractFn,
      { ttl: 3600 } // 1 hour - entities may change with model updates
    );
  }

  // Vector similarity computation with caching
  async cacheVectorSimilarity(
    vectors: { query: number[]; candidates: number[][] },
    similarityFn: (vectors: any) => Promise<number[]>
  ): Promise<{ result: number[]; fromCache: boolean; computeTime: number }> {
    return this.computeWithCache(
      'vector_similarity',
      vectors,
      similarityFn,
      { 
        ttl: 7200, // 2 hours
        cacheMetadata: { fromGPU: true }
      }
    );
  }

  // Legal document classification with caching
  async cacheLegalDocumentClassification(
    document: { content: string; metadata: any },
    classifyFn: (doc: any) => Promise<{ category: string; confidence: number }>
  ): Promise<{ result: { category: string; confidence: number }; fromCache: boolean; computeTime: number }> {
    return this.computeWithCache(
      'legal_document_classification',
      { contentHash: this.generateInputHash(document.content, 'content'), metadata: document.metadata },
      classifyFn,
      { ttl: 21600 } // 6 hours - classifications are relatively stable
    );
  }

  // Enhanced RAG query with caching
  async cacheRAGQuery(
    query: { question: string; context: string[]; filters?: any },
    ragFn: (query: any) => Promise<{ answer: string; sources: any[]; confidence: number }>
  ): Promise<{ result: { answer: string; sources: any[]; confidence: number }; fromCache: boolean; computeTime: number }> {
    // Sort context for consistent hashing
    const normalizedQuery = {
      ...query,
      context: query.context.sort()
    };
    
    return this.computeWithCache(
      'enhanced_rag_query',
      normalizedQuery,
      ragFn,
      { ttl: 1800 } // 30 minutes - RAG results can change with context updates
    );
  }

  /**
   * Serialize cache entry with compression
   */
  private async serializeEntry<T>(entry: CacheEntry<T>): Promise<string> {
    if (!this.config.compressionEnabled) {
      return JSON.stringify(entry);
    }

    try {
      // Use MessagePack for better compression and speed
      const packed = msgpackEncode(entry);
      return Buffer.from(packed).toString('base64');
    } catch (error) {
      // Fallback to JSON
      return JSON.stringify(entry);
    }
  }

  /**
   * Deserialize cache entry with decompression
   */
  private async deserializeEntry<T>(data: string): Promise<CacheEntry<T>> {
    // Try MessagePack first
    try {
      const buffer = Buffer.from(data, 'base64');
      return msgpackDecode(buffer) as CacheEntry<T>;
    } catch (error) {
      // Fallback to JSON
      return JSON.parse(data);
    }
  }

  /**
   * Update cache statistics
   */
  private updateStats(): void {
    // Update memory stats
    this.stats.memory.size = this.memoryCache.size;
    this.stats.memory.hitRate = this.stats.memory.hits / (this.stats.memory.hits + this.stats.memory.misses) || 0;

    // Update Redis stats
    this.stats.redis.hitRate = this.stats.redis.hits / (this.stats.redis.hits + this.stats.redis.misses) || 0;

    // Update overall stats
    this.stats.overall.hitRate = this.stats.overall.totalHits / (this.stats.overall.totalHits + this.stats.overall.totalMisses) || 0;

    // Calculate compute time saved
    this.stats.overall.totalComputeTimeSaved = Array.from(this.memoryCache.values())
      .reduce((total, entry) => total + (entry.metadata.hitCount * entry.metadata.computeTimeMs), 0);

    this.emit('stats:updated', this.stats);
  }

  /**
   * Get comprehensive cache statistics
   */
  getStats(): CacheStats {
    this.updateStats();
    return { ...this.stats };
  }

  /**
   * Cache maintenance operations
   */
  async clearCache(options: { taskType?: string; olderThan?: number } = {}): Promise<number> {
    let cleared = 0;

    if (options.taskType) {
      // Clear specific task type from memory
      for (const [key, entry] of this.memoryCache.entries()) {
        if (entry.metadata.taskType === options.taskType) {
          this.memoryCache.delete(key);
          cleared++;
        }
      }

      // Clear from Redis
      const pattern = `${this.config.redisPrefix}:${options.taskType}:*`;
      const keys = await this.redis.keys(pattern);
      if (keys.length > 0) {
        await this.redis.del(...keys);
        cleared += keys.length;
      }
    } else if (options.olderThan) {
      const cutoff = Date.now() - options.olderThan;
      
      // Clear old entries from memory
      for (const [key, entry] of this.memoryCache.entries()) {
        if (entry.timestamp < cutoff) {
          this.memoryCache.delete(key);
          cleared++;
        }
      }
    } else {
      // Clear everything
      this.memoryCache.clear();
      const pattern = `${this.config.redisPrefix}:*`;
      const keys = await this.redis.keys(pattern);
      if (keys.length > 0) {
        await this.redis.del(...keys);
      }
      cleared = this.memoryCache.size + keys.length;
    }

    this.emit('cache:cleared', { cleared, options });
    return cleared;
  }

  /**
   * Preload frequently used computations
   */
  async preloadCache(preloadConfig: {
    taskType: string;
    inputs: any[];
    computeFn: (input: any) => Promise<any>;
    concurrency?: number;
  }): Promise<void> {
    const concurrency = preloadConfig.concurrency || 5;
    const chunks = [];
    
    for (let i = 0; i < preloadConfig.inputs.length; i += concurrency) {
      chunks.push(preloadConfig.inputs.slice(i, i + concurrency));
    }

    for (const chunk of chunks) {
      await Promise.all(
        chunk.map(input => 
          this.computeWithCache(preloadConfig.taskType, input, preloadConfig.computeFn)
        )
      );
    }

    this.emit('cache:preloaded', {
      taskType: preloadConfig.taskType,
      count: preloadConfig.inputs.length
    });
  }

  /**
   * Shutdown cache system
   */
  async shutdown(): Promise<void> {
    this.memoryCache.clear();
    await this.redis.quit();
    this.emit('cache:shutdown');
  }
}

// Global cache instance for legal AI platform
export const legalAIResultCache = new AdvancedResultCache({
  maxMemoryItems: 2000,
  maxMemoryMB: 512,
  redisPrefix: 'legal_ai_advanced_cache',
  defaultTTL: 3600,
  compressionEnabled: true,
  sha256Salt: 'legal_ai_secure_salt_2024'
});