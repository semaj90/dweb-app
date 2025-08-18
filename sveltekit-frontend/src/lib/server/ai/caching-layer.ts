// @ts-nocheck
// lib/server/ai/caching-layer.ts
// Advanced caching layer for AI synthesis results with Redis and LRU fallback

import { logger } from "./logger";
import { Redis } from 'ioredis';
import LRU from 'lru-cache';
import crypto from 'crypto';

interface CacheOptions {
  ttl?: number; // Time to live in seconds
  tags?: string[]; // Tags for cache invalidation
  priority?: number; // Cache priority (higher = more important)
}

interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  size: number;
  memoryUsage: number;
}

class CachingLayer {
  private redis: Redis | null = null;
  private lruCache: LRU<string, any>;
  private stats: CacheStats;
  private hotCache: Map<string, { data: any; hits: number; lastAccess: number }>;
  private cacheConfig = {
    maxMemory: 512 * 1024 * 1024, // 512MB max memory
    maxItems: 10000,
    hotCacheThreshold: 5, // Hits needed to promote to hot cache
    compressionThreshold: 1024 // Compress items larger than 1KB
  };

  constructor() {
    this.initializeCache();
    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      size: 0,
      memoryUsage: 0
    };
    this.hotCache = new Map();
  }

  private initializeCache(): void {
    // Try to connect to Redis
    try {
      this.redis = new Redis({
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD,
        db: parseInt(process.env.REDIS_DB || '0'),
        retryStrategy: (times) => {
          if (times > 3) {
            logger.warn('[CachingLayer] Redis connection failed, falling back to LRU cache');
            this.redis = null;
            return null;
          }
          return Math.min(times * 100, 3000);
        }
      });

      this.redis.on('connect', () => {
        logger.info('[CachingLayer] Redis connected successfully');
      });

      this.redis.on('error', (err) => {
        logger.error('[CachingLayer] Redis error:', err);
        this.redis = null;
      });
    } catch (error) {
      logger.warn('[CachingLayer] Redis initialization failed, using LRU cache only:', error);
      this.redis = null;
    }

    // Initialize LRU cache as fallback/primary
    this.lruCache = new LRU({
      max: this.cacheConfig.maxItems,
      maxSize: this.cacheConfig.maxMemory,
      sizeCalculation: (value) => {
        // Estimate memory size
        return JSON.stringify(value).length;
      },
      dispose: (value, key) => {
        this.stats.evictions++;
        logger.debug(`[CachingLayer] Evicted key: ${key}`);
      },
      ttl: 1000 * 60 * 60 // 1 hour default TTL
    });

    // Periodic cleanup and stats update
    setInterval(() => this.performMaintenance(), 60000); // Every minute
  }

  /**
   * Generate cache key from input parameters
   */
  generateKey(params: any): string {
    const normalized = JSON.stringify(params, Object.keys(params).sort());
    return `ai:synthesis:${crypto.createHash('sha256').update(normalized).digest('hex')}`;
  }

  /**
   * Get item from cache with multi-tier strategy
   */
  async get(key: string): Promise<any | null> {
    try {
      // Check hot cache first (ultra-fast)
      const hotItem = this.hotCache.get(key);
      if (hotItem) {
        hotItem.hits++;
        hotItem.lastAccess = Date.now();
        this.stats.hits++;
        logger.debug(`[CachingLayer] Hot cache hit for ${key}`);
        return hotItem.data;
      }

      // Check LRU cache (fast)
      const lruItem = this.lruCache.get(key);
      if (lruItem) {
        this.stats.hits++;
        
        // Promote to hot cache if accessed frequently
        this.promoteToHotCache(key, lruItem);
        
        logger.debug(`[CachingLayer] LRU cache hit for ${key}`);
        return lruItem;
      }

      // Check Redis if available (distributed)
      if (this.redis) {
        try {
          const redisValue = await this.redis.get(key);
          if (redisValue) {
            const data = JSON.parse(redisValue);
            
            // Populate LRU cache
            this.lruCache.set(key, data);
            
            this.stats.hits++;
            logger.debug(`[CachingLayer] Redis cache hit for ${key}`);
            return data;
          }
        } catch (error) {
          logger.warn(`[CachingLayer] Redis get failed for ${key}:`, error);
        }
      }

      this.stats.misses++;
      return null;

    } catch (error) {
      logger.error(`[CachingLayer] Get failed for ${key}:`, error);
      this.stats.misses++;
      return null;
    }
  }

  /**
   * Set item in cache with multi-tier strategy
   */
  async set(key: string, value: any, options: CacheOptions = {}): Promise<void> {
    try {
      const ttl = options.ttl || 3600; // Default 1 hour
      
      // Compress large values
      const dataToStore = this.shouldCompress(value) ? 
        await this.compress(value) : value;

      // Store in LRU cache
      this.lruCache.set(key, dataToStore, {
        ttl: ttl * 1000 // Convert to milliseconds
      });

      // Store in Redis if available
      if (this.redis) {
        try {
          const redisValue = JSON.stringify(dataToStore);
          const pipeline = this.redis.pipeline();
          
          pipeline.set(key, redisValue);
          pipeline.expire(key, ttl);
          
          // Add tags for invalidation
          if (options.tags) {
            for (const tag of options.tags) {
              pipeline.sadd(`tag:${tag}`, key);
              pipeline.expire(`tag:${tag}`, ttl);
            }
          }
          
          await pipeline.exec();
          logger.debug(`[CachingLayer] Stored ${key} in Redis with TTL ${ttl}s`);
        } catch (error) {
          logger.warn(`[CachingLayer] Redis set failed for ${key}:`, error);
        }
      }

      // Update stats
      this.stats.size++;
      this.stats.memoryUsage = this.lruCache.calculatedSize || 0;

      logger.debug(`[CachingLayer] Cached ${key} with TTL ${ttl}s`);

    } catch (error) {
      logger.error(`[CachingLayer] Set failed for ${key}:`, error);
    }
  }

  /**
   * Invalidate cache by tags
   */
  async invalidateByTags(tags: string[]): Promise<void> {
    try {
      if (this.redis) {
        for (const tag of tags) {
          const keys = await this.redis.smembers(`tag:${tag}`);
          
          if (keys.length > 0) {
            // Remove from Redis
            await this.redis.del(...keys);
            
            // Remove from LRU cache
            for (const key of keys) {
              this.lruCache.delete(key);
              this.hotCache.delete(key);
            }
            
            // Clean up tag set
            await this.redis.del(`tag:${tag}`);
            
            logger.info(`[CachingLayer] Invalidated ${keys.length} keys with tag ${tag}`);
          }
        }
      }
    } catch (error) {
      logger.error('[CachingLayer] Tag invalidation failed:', error);
    }
  }

  /**
   * Clear entire cache
   */
  async clear(): Promise<void> {
    try {
      this.lruCache.clear();
      this.hotCache.clear();
      
      if (this.redis) {
        await this.redis.flushdb();
      }
      
      this.stats.size = 0;
      this.stats.memoryUsage = 0;
      
      logger.info('[CachingLayer] Cache cleared');
    } catch (error) {
      logger.error('[CachingLayer] Clear failed:', error);
    }
  }

  /**
   * Warm up cache with frequently accessed items
   */
  async warmUp(items: Array<{ key: string; value: any; options?: CacheOptions }>): Promise<void> {
    logger.info(`[CachingLayer] Warming up cache with ${items.length} items`);
    
    const warmUpPromises = items.map(item =>
      this.set(item.key, item.value, item.options)
    );
    
    await Promise.all(warmUpPromises);
    
    logger.info('[CachingLayer] Cache warm-up completed');
  }

  /**
   * Get cache statistics
   */
  async getStats(): Promise<any> {
    const redisInfo = this.redis ? await this.getRedisStats() : null;
    
    return {
      hits: this.stats.hits,
      misses: this.stats.misses,
      hitRate: this.stats.hits / (this.stats.hits + this.stats.misses) || 0,
      evictions: this.stats.evictions,
      size: this.stats.size,
      memoryUsage: this.stats.memoryUsage,
      hotCacheSize: this.hotCache.size,
      lruCacheSize: this.lruCache.size,
      redisConnected: !!this.redis,
      redisStats: redisInfo
    };
  }

  // === PRIVATE HELPER METHODS ===

  private promoteToHotCache(key: string, data: any): void {
    // Track access count
    const accessCount = (this.lruCache as any).get(key + ':count') || 0;
    (this.lruCache as any).set(key + ':count', accessCount + 1);
    
    // Promote to hot cache if threshold met
    if (accessCount >= this.cacheConfig.hotCacheThreshold) {
      this.hotCache.set(key, {
        data,
        hits: accessCount,
        lastAccess: Date.now()
      });
      
      // Limit hot cache size
      if (this.hotCache.size > 100) {
        this.evictFromHotCache();
      }
    }
  }

  private evictFromHotCache(): void {
    // Find least recently accessed item
    let oldestKey = null;
    let oldestTime = Date.now();
    
    for (const [key, item] of this.hotCache) {
      if (item.lastAccess < oldestTime) {
        oldestTime = item.lastAccess;
        oldestKey = key;
      }
    }
    
    if (oldestKey) {
      this.hotCache.delete(oldestKey);
      logger.debug(`[CachingLayer] Evicted ${oldestKey} from hot cache`);
    }
  }

  private shouldCompress(value: any): boolean {
    const size = JSON.stringify(value).length;
    return size > this.cacheConfig.compressionThreshold;
  }

  private async compress(value: any): Promise<any> {
    // For now, just return the value
    // In production, use zlib or similar for compression
    return value;
  }

  private async getRedisStats(): Promise<any> {
    if (!this.redis) return null;
    
    try {
      const info = await this.redis.info('memory');
      const dbSize = await this.redis.dbsize();
      
      return {
        dbSize,
        memory: info
      };
    } catch (error) {
      return null;
    }
  }

  private async performMaintenance(): Promise<void> {
    // Clean up expired hot cache entries
    const now = Date.now();
    const maxAge = 30 * 60 * 1000; // 30 minutes
    
    for (const [key, item] of this.hotCache) {
      if (now - item.lastAccess > maxAge) {
        this.hotCache.delete(key);
      }
    }
    
    // Update memory usage stats
    this.stats.memoryUsage = this.lruCache.calculatedSize || 0;
    
    // Log stats periodically
    if (Math.random() < 0.1) { // 10% chance
      const stats = await this.getStats();
      logger.info('[CachingLayer] Cache stats:', stats);
    }
  }
}

// Export singleton instance
export const cachingLayer = new CachingLayer();

// Export types
export type { CacheOptions, CacheStats };

