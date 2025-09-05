// Multi-layer cache implementation
import { LRUCache } from 'lru-cache';
import type { Redis } from 'ioredis';

export interface CacheOptions {
  maxSize?: number;
  ttl?: number; // in milliseconds
  redis?: Redis;
}

export interface ExtendedCacheOptions {
  type?: string;
  ttl?: number;
  tags?: string[];
  priority?: 'low' | 'medium' | 'high';
}

export interface MultiLayerCache {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, options?: number | ExtendedCacheOptions): Promise<any>;
  delete(key: string): Promise<any>;
  clear(): Promise<any>;
}

class MultiLayerCacheImpl implements MultiLayerCache {
  private memoryCache: LRUCache<string, any>;
  private redis?: Redis;

  constructor(options: CacheOptions = {}) {
    this.memoryCache = new LRUCache({
      max: options.maxSize || 1000,
      ttl: options.ttl || 5 * 60 * 1000, // 5 minutes default
    });
    this.redis = options.redis;
  }

  async get<T>(key: string): Promise<T | null> {
    // Try memory cache first
    const memoryValue = this.memoryCache.get(key);
    if (memoryValue !== undefined) {
      return memoryValue as T;
    }

    // Try Redis if available
    if (this.redis) {
      try {
        const redisValue = await this.redis.get(key);
        if (redisValue) {
          const parsed = JSON.parse(redisValue) as T;
          // Store in memory cache for faster future access
          this.memoryCache.set(key, parsed);
          return parsed;
        }
      } catch (error: any) {
        console.warn('Redis cache get error:', error);
      }
    }

    return null;
  }

  async set<T>(key: string, value: T, options?: number | ExtendedCacheOptions): Promise<any> {
    // Handle both old number format and new options format
    let ttl: number | undefined;
    if (typeof options === 'number') {
      ttl = options;
    } else if (options && options.ttl) {
      ttl = options.ttl;
    }

    // Set in memory cache
    this.memoryCache.set(key, value, { ttl });

    // Set in Redis if available
    if (this.redis) {
      try {
        const serialized = JSON.stringify(value);
        if (ttl) {
          await this.redis.setex(key, Math.floor(ttl / 1000), serialized);
        } else {
          await this.redis.set(key, serialized);
        }
      } catch (error: any) {
        console.warn('Redis cache set error:', error);
      }
    }
  }

  async delete(key: string): Promise<any> {
    this.memoryCache.delete(key);

    if (this.redis) {
      try {
        await this.redis.del(key);
      } catch (error: any) {
        console.warn('Redis cache delete error:', error);
      }
    }
  }

  async clear(): Promise<any> {
    this.memoryCache.clear();

    if (this.redis) {
      try {
        await this.redis.flushall();
      } catch (error: any) {
        console.warn('Redis cache clear error:', error);
      }
    }
  }
}

// Default cache instance
export const multiLayerCache = new MultiLayerCacheImpl();

// Export the class for custom instances
export { MultiLayerCacheImpl };