// @ts-nocheck
// Temporary stub for caching service to resolve TypeScript errors
export interface CacheOptions {
  ttl?: number;
  tags?: string[];
  priority?: 'low' | 'medium' | 'high';
  layer?: 'memory' | 'loki' | 'redis' | 'postgres' | 'all';
}

class StubCacheService {
  private cache = new Map<string, any>();
  
  async get<T>(key: string, options?: CacheOptions): Promise<T | null> {
    return this.cache.get(key) || null;
  }

  async set<T>(key: string, value: T, options?: CacheOptions): Promise<boolean> {
    this.cache.set(key, value);
    return true;
  }

  async delete(key: string): Promise<boolean> {
    return this.cache.delete(key);
  }

  async clear(): Promise<boolean> {
    this.cache.clear();
    return true;
  }

  async getStats() {
    return {
      requests: 0,
      hits: 0,
      misses: 0,
      errors: 0
    };
  }
}

export const cachingService = new StubCacheService();
export default cachingService;