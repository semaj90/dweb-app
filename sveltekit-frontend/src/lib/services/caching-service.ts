// Enhanced Caching Service Integration
// Integrates with existing comprehensive caching architecture + adds Redis support
// Provides simple interface for SvelteKit components

import { comprehensiveCache, ComprehensiveCacheManager } from './comprehensive-caching-architecture';
import { db } from '$lib/server/database';
import { vectorSimilarityCache } from '$lib/database/enhanced-schema';
import { eq, sql } from 'drizzle-orm';

// ============================================================================
// CACHE SERVICE INTERFACE
// ============================================================================

export interface CacheOptions {
  ttl?: number; // Time to live in milliseconds
  tags?: string[];
  priority?: 'low' | 'medium' | 'high';
  layer?: 'memory' | 'loki' | 'redis' | 'postgres' | 'all';
}

export interface SearchCacheOptions extends CacheOptions {
  similarity?: number;
  maxResults?: number;
  includeMetadata?: boolean;
}

// ============================================================================
// ENHANCED CACHING SERVICE
// ============================================================================

class EnhancedCachingService {
  private cache: ComprehensiveCacheManager;
  private stats = {
    requests: 0,
    hits: 0,
    misses: 0,
    errors: 0
  };

  constructor() {
    this.cache = comprehensiveCache;
  }

  // ============================================================================
  // BASIC CACHE OPERATIONS
  // ============================================================================

  async get<T>(key: string, options: CacheOptions = {}): Promise<T | null> {
    this.stats.requests++;
    
    try {
      const result = await this.cache.get<T>(key);
      if (result) {
        this.stats.hits++;
        return result;
      } else {
        this.stats.misses++;
        return null;
      }
    } catch (error) {
      this.stats.errors++;
      console.error('Cache get error:', error);
      return null;
    }
  }

  async set<T>(key: string, value: T, options: CacheOptions = {}): Promise<boolean> {
    try {
      const cacheOptions = {
        ttl: options.ttl,
        tags: options.tags,
        priority: this.convertPriority(options.priority)
      };

      return await this.cache.set(key, value, cacheOptions);
    } catch (error) {
      this.stats.errors++;
      console.error('Cache set error:', error);
      return false;
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      return await this.cache.delete(key);
    } catch (error) {
      this.stats.errors++;
      console.error('Cache delete error:', error);
      return false;
    }
  }

  async clear(): Promise<boolean> {
    try {
      return await this.cache.clear();
    } catch (error) {
      this.stats.errors++;
      console.error('Cache clear error:', error);
      return false;
    }
  }

  // ============================================================================
  // SPECIALIZED CACHE METHODS
  // ============================================================================

  async getWithFallback<T>(
    key: string, 
    fallbackFn: () => Promise<T>, 
    options: CacheOptions = {}
  ): Promise<T> {
    const cached = await this.get<T>(key, options);
    if (cached !== null) {
      return cached;
    }

    // Execute fallback and cache result
    const value = await fallbackFn();
    await this.set(key, value, options);
    return value;
  }

  async batchGet<T>(keys: string[], options: CacheOptions = {}): Promise<Map<string, T>> {
    const results = new Map<string, T>();
    
    // Use batch operation if available
    if (this.cache.batchGet) {
      return await this.cache.batchGet<T>(keys);
    }

    // Fallback to individual gets
    const promises = keys.map(async (key) => {
      const value = await this.get<T>(key, options);
      return { key, value };
    });

    const results_array = await Promise.all(promises);
    for (const { key, value } of results_array) {
      if (value !== null) {
        results.set(key, value);
      }
    }

    return results;
  }

  async batchSet<T>(items: Array<{ key: string; value: T; options?: CacheOptions }>): Promise<boolean[]> {
    // Use batch operation if available
    if (this.cache.batchSet) {
      const batchItems = items.map(item => ({
        key: item.key,
        value: item.value,
        options: {
          ttl: item.options?.ttl,
          tags: item.options?.tags,
          priority: this.convertPriority(item.options?.priority)
        }
      }));
      return await this.cache.batchSet(batchItems);
    }

    // Fallback to individual sets
    const promises = items.map(item => 
      this.set(item.key, item.value, item.options || {})
    );
    return await Promise.all(promises);
  }

  // ============================================================================
  // LEGAL AI SPECIFIC METHODS
  // ============================================================================

  async cacheSearchResults(query: string, results: any[], options: SearchCacheOptions = {}): Promise<void> {
    const cacheKey = `search:${this.hashQuery(query)}`;
    const cacheData = {
      query,
      results,
      timestamp: Date.now(),
      metadata: {
        resultCount: results.length,
        similarity: options.similarity,
        maxResults: options.maxResults
      }
    };

    await this.set(cacheKey, cacheData, {
      ttl: options.ttl || 600000, // 10 minutes default for search results
      tags: ['search', 'legal-ai', ...(options.tags || [])],
      priority: options.priority || 'medium'
    });
  }

  async getCachedSearchResults(query: string, options: SearchCacheOptions = {}): Promise<any[] | null> {
    const cacheKey = `search:${this.hashQuery(query)}`;
    const cached = await this.get<any>(cacheKey, options);
    
    if (cached && cached.results) {
      return cached.results;
    }
    
    return null;
  }

  async cacheDocumentAnalysis(documentId: string, analysis: any, options: CacheOptions = {}): Promise<void> {
    const cacheKey = `analysis:${documentId}`;
    await this.set(cacheKey, analysis, {
      ttl: options.ttl || 3600000, // 1 hour default for document analysis
      tags: ['analysis', 'document', documentId, ...(options.tags || [])],
      priority: options.priority || 'high'
    });
  }

  async getCachedDocumentAnalysis(documentId: string, options: CacheOptions = {}): Promise<any | null> {
    const cacheKey = `analysis:${documentId}`;
    return await this.get<any>(cacheKey, options);
  }

  async cacheVectorSimilarity(queryHash: string, results: any[], options: CacheOptions = {}): Promise<void> {
    const cacheKey = `vector:${queryHash}`;
    await this.set(cacheKey, results, {
      ttl: options.ttl || 1800000, // 30 minutes default for vector results
      tags: ['vector', 'similarity', ...(options.tags || [])],
      priority: options.priority || 'high'
    });
  }

  async getCachedVectorSimilarity(queryHash: string, options: CacheOptions = {}): Promise<any[] | null> {
    const cacheKey = `vector:${queryHash}`;
    return await this.get<any[]>(cacheKey, options);
  }

  // ============================================================================
  // CACHE INVALIDATION
  // ============================================================================

  async invalidateByTag(tag: string): Promise<number> {
    try {
      if (this.cache.invalidateByTag) {
        return await this.cache.invalidateByTag(tag);
      }
      
      // Fallback: clear all cache (less efficient)
      await this.cache.clear();
      return 1;
    } catch (error) {
      this.stats.errors++;
      console.error('Cache invalidation error:', error);
      return 0;
    }
  }

  async invalidateDocument(documentId: string): Promise<void> {
    await Promise.all([
      this.delete(`analysis:${documentId}`),
      this.invalidateByTag(documentId),
      this.invalidateByTag('document')
    ]);
  }

  async invalidateSearchCache(): Promise<void> {
    await this.invalidateByTag('search');
  }

  // ============================================================================
  // STATISTICS AND HEALTH
  // ============================================================================

  async getStats(): Promise<{
    service: typeof this.stats;
    layers?: any;
  }> {
    const layerStats = this.cache.getOverallStats ? await this.cache.getOverallStats() : null;
    
    return {
      service: {
        ...this.stats,
        hitRate: this.stats.requests > 0 ? this.stats.hits / this.stats.requests : 0,
        errorRate: this.stats.requests > 0 ? this.stats.errors / this.stats.requests : 0
      },
      layers: layerStats
    };
  }

  async healthCheck(): Promise<{
    healthy: boolean;
    service: boolean;
    layers?: any;
  }> {
    let serviceHealthy = true;
    
    try {
      // Test basic cache operations
      const testKey = '__health_check__';
      await this.set(testKey, 'test');
      const result = await this.get(testKey);
      await this.delete(testKey);
      
      serviceHealthy = result === 'test';
    } catch {
      serviceHealthy = false;
    }

    const layerHealth = this.cache.healthCheck ? await this.cache.healthCheck() : null;

    return {
      healthy: serviceHealthy && (layerHealth?.healthy ?? true),
      service: serviceHealthy,
      layers: layerHealth
    };
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private convertPriority(priority?: 'low' | 'medium' | 'high'): number {
    switch (priority) {
      case 'low': return 1;
      case 'medium': return 5;
      case 'high': return 10;
      default: return 5;
    }
  }

  private hashQuery(query: string): string {
    // Simple hash function for cache keys
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  resetStats(): void {
    this.stats = {
      requests: 0,
      hits: 0,
      misses: 0,
      errors: 0
    };
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const cachingService = new EnhancedCachingService();

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

export async function getCached<T>(key: string, options?: CacheOptions): Promise<T | null> {
  return cachingService.get<T>(key, options);
}

export async function setCached<T>(key: string, value: T, options?: CacheOptions): Promise<boolean> {
  return cachingService.set(key, value, options);
}

export async function getCachedWithFallback<T>(
  key: string, 
  fallbackFn: () => Promise<T>, 
  options?: CacheOptions
): Promise<T> {
  return cachingService.getWithFallback(key, fallbackFn, options);
}

export async function cacheSearchResults(query: string, results: any[], options?: SearchCacheOptions): Promise<void> {
  return cachingService.cacheSearchResults(query, results, options);
}

export async function getCachedSearchResults(query: string, options?: SearchCacheOptions): Promise<any[] | null> {
  return cachingService.getCachedSearchResults(query, options);
}

export async function cacheDocumentAnalysis(documentId: string, analysis: any, options?: CacheOptions): Promise<void> {
  return cachingService.cacheDocumentAnalysis(documentId, analysis, options);
}

export async function getCachedDocumentAnalysis(documentId: string, options?: CacheOptions): Promise<any | null> {
  return cachingService.getCachedDocumentAnalysis(documentId, options);
}

export async function invalidateDocument(documentId: string): Promise<void> {
  return cachingService.invalidateDocument(documentId);
}

export async function getCacheStats() {
  return cachingService.getStats();
}

export async function getCacheHealth() {
  return cachingService.healthCheck();
}

export default cachingService;