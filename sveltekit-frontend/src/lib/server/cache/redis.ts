// @ts-nocheck
// Redis caching service for embeddings and search results
// Use process.env for server-side environment variables

// Simple in-memory cache fallback
const memoryCache = new Map<string, any>();
const MEMORY_CACHE_MAX_SIZE = 1000;
const CACHE_TTL = 60 * 60 * 1000; // 1 hour

interface CacheItem<T> {
  data: T;
  timestamp: number;
  ttl: number;
}
class CacheService {
  private redisClient: any = null;
  private useRedis = false;

  constructor() {
    this.initializeRedis();
  }
  private async initializeRedis() {
    if (!import.meta.env.REDIS_URL) {
      console.log("📝 Using memory cache (Redis not configured)");
      return;
    }
    try {
      // Import redis dynamically to avoid issues in development
      const { createClient } = await import("redis");

      this.redisClient = createClient({
        url: import.meta.env.REDIS_URL,
        socket: {
          connectTimeout: 5000,
          reconnectStrategy: (retries) => Math.min(retries * 50, 500),
        },
      });

      this.redisClient.on("error", (err: any) => {
        console.warn(
          "Redis connection error, falling back to memory cache:",
          err.message,
        );
        this.useRedis = false;
      });

      await this.redisClient.connect();
      this.useRedis = true;
      console.log("📝 Redis cache connected");
    } catch (error) {
      console.warn("Redis not available, using memory cache:", error);
      this.useRedis = false;
    }
  }
  async get<T>(key: string): Promise<T | null> {
    try {
      if (this.useRedis && this.redisClient) {
        const result = await this.redisClient.get(key);
        return result ? JSON.parse(result) : null;
      } else {
        return this.getFromMemory(key);
      }
    } catch (error) {
      console.warn("Cache get error:", error);
      return null;
    }
  }
  async set<T>(
    key: string,
    value: T,
    ttlMs: number = CACHE_TTL,
  ): Promise<void> {
    try {
      if (this.useRedis && this.redisClient) {
        await this.redisClient.setEx(
          key,
          Math.floor(ttlMs / 1000),
          JSON.stringify(value),
        );
      } else {
        this.setInMemory(key, value, ttlMs);
      }
    } catch (error) {
      console.warn("Cache set error:", error);
      // Fallback to memory cache
      this.setInMemory(key, value, ttlMs);
    }
  }
  async del(key: string): Promise<void> {
    try {
      if (this.useRedis && this.redisClient) {
        await this.redisClient.del(key);
      } else {
        memoryCache.delete(key);
      }
    } catch (error) {
      console.warn("Cache delete error:", error);
    }
  }
  private getFromMemory<T>(key: string): T | null {
    const item = memoryCache.get(key) as CacheItem<T> | undefined;
    if (!item) return null;

    const now = Date.now();
    if (now > item.timestamp + item.ttl) {
      memoryCache.delete(key);
      return null;
    }
    return item.data;
  }
  private setInMemory<T>(key: string, value: T, ttlMs: number): void {
    // Limit memory cache size
    if (memoryCache.size >= MEMORY_CACHE_MAX_SIZE) {
      const firstKey = memoryCache.keys().next().value;
      if (firstKey) {
        memoryCache.delete(firstKey);
      }
    }
    memoryCache.set(key, {
      data: value,
      timestamp: Date.now(),
      ttl: ttlMs,
    });
  }
  // Specialized methods for embeddings
  async getEmbedding(
    text: string,
    model: string = "openai",
  ): Promise<number[] | null> {
    const key = `embedding:${model}:${this.hashString(text)}`;
    return this.get<number[]>(key);
  }
  async setEmbedding(
    text: string,
    embedding: number[],
    model: string = "openai",
  ): Promise<void> {
    const key = `embedding:${model}:${this.hashString(text)}`;
    // Embeddings can be cached longer since they don't change
    await this.set(key, embedding, 24 * 60 * 60 * 1000); // 24 hours
  }
  // Specialized methods for search results
  async getSearchResults(
    query: string,
    searchType: string,
    filters: any = {},
  ): Promise<any[] | null> {
    const key = `search:${searchType}:${this.hashString(query)}:${this.hashString(JSON.stringify(filters))}`;
    return this.get<any[]>(key);
  }
  async setSearchResults(
    query: string,
    searchType: string,
    results: any[],
    filters: any = {},
  ): Promise<void> {
    const key = `search:${searchType}:${this.hashString(query)}:${this.hashString(JSON.stringify(filters))}`;
    // Search results cached for shorter time
    await this.set(key, results, 5 * 60 * 1000); // 5 minutes
  }
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }
  async close(): Promise<void> {
    if (this.redisClient) {
      await this.redisClient.quit();
    }
  }
}
// Singleton instance
export const cache = new CacheService();

// Helper functions for common cache operations
export const cacheEmbedding = (
  text: string,
  embedding: number[],
  model?: string,
) => cache.setEmbedding(text, embedding, model);

export const getCachedEmbedding = (text: string, model?: string) =>
  cache.getEmbedding(text, model);

export const cacheSearchResults = (
  query: string,
  searchType: string,
  results: any[],
  filters?: any,
) => cache.setSearchResults(query, searchType, results, filters);

export const getCachedSearchResults = (
  query: string,
  searchType: string,
  filters?: any,
) => cache.getSearchResults(query, searchType, filters);
