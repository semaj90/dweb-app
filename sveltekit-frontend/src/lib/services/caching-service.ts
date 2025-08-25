
// Enhanced Caching Service - Production Implementation
// Integrated with NES-Style Cache Orchestrator and Advanced Caching

import type { Writable } from "svelte/store";
import { writable } from "svelte/store";

// Import advanced cache manager for L1-L7 caching integration
import type { AdvancedCacheManager } from '../../lib/caching/advanced-cache-manager.js';

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
  private cache = new Map<string, any>();
  private stats = {
    requests: 0,
    hits: 0,
    misses: 0,
    errors: 0
  };
  private nesCacheOrchestrator?: any; // Dynamic import to avoid circular deps
  private advancedCacheManager?: AdvancedCacheManager; // L1-L7 cache integration

  constructor() {
    // Initialize with NES Cache Orchestrator integration
    this.initializeNESCacheOrchestrator();
    this.initializeAdvancedCacheManager();
  }

  private async initializeNESCacheOrchestrator() {
    try {
      // Lazy load the NES cache orchestrator to avoid circular dependencies
      const { NESCacheOrchestrator } = await import('./nes-cache-orchestrator.js');
      this.nesCacheOrchestrator = new NESCacheOrchestrator();
      await this.nesCacheOrchestrator.initialize();
    } catch (error) {
      console.warn('NES Cache Orchestrator not available, using fallback cache:', error);
    }
  }

  private async initializeAdvancedCacheManager() {
    try {
      // Lazy load the advanced cache manager
      const { AdvancedCacheManager } = await import('../../lib/caching/advanced-cache-manager.js');
      this.advancedCacheManager = new AdvancedCacheManager();
      await this.advancedCacheManager.initialize();
    } catch (error) {
      console.warn('Advanced Cache Manager not available, using fallback cache:', error);
    }
  }

  // ============================================================================
  // BASIC CACHE OPERATIONS
  // ============================================================================

  async get<T>(key: string, options: CacheOptions = {}): Promise<T | null> {
    this.stats.requests++;
    
    try {
      // Priority 1: Try NES Cache Orchestrator first (fastest, NES-inspired efficiency)
      if (this.nesCacheOrchestrator && options.layer !== 'memory') {
        const nesResult = await this.nesCacheOrchestrator.getFromOptimalTier(key, options);
        if (nesResult !== null) {
          this.stats.hits++;
          return nesResult;
        }
      }

      // Priority 2: Try Advanced Cache Manager (L1-L7 intelligent tiers)
      if (this.advancedCacheManager && options.layer !== 'memory') {
        const advancedResult = await this.advancedCacheManager.get(key, options);
        if (advancedResult !== null) {
          this.stats.hits++;
          return advancedResult;
        }
      }

      // Priority 3: Fallback to local cache
      const result = this.cache.get(key);
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
      // Priority 1: Store in NES Cache Orchestrator if available
      if (this.nesCacheOrchestrator && options.layer !== 'memory') {
        const nesSuccess = await this.nesCacheOrchestrator.setToOptimalTier(key, value, options);
        if (nesSuccess) {
          return true;
        }
      }

      // Priority 2: Store in Advanced Cache Manager (L1-L7 intelligent placement)
      if (this.advancedCacheManager && options.layer !== 'memory') {
        const advancedSuccess = await this.advancedCacheManager.set(key, value, options);
        if (advancedSuccess) {
          return true;
        }
      }

      // Priority 3: Fallback to local cache
      this.cache.set(key, value);
      return true;
    } catch (error) {
      this.stats.errors++;
      console.error('Cache set error:', error);
      return false;
    }
  }

  async delete(key: string): Promise<boolean> {
    try {
      return this.cache.delete(key);
    } catch (error) {
      this.stats.errors++;
      console.error('Cache delete error:', error);
      return false;
    }
  }

  async clear(): Promise<boolean> {
    try {
      this.cache.clear();
      return true;
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
    
    // Individual gets for simple implementation
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
    // Individual sets for simple implementation
    const promises = items.map((item) => this.set(item.key, item.value, item.options || {}));
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

  async getCachedSearchResults(query: string, options: SearchCacheOptions = {}): Promise<unknown[] | null> {
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

  async getCachedVectorSimilarity(queryHash: string, options: CacheOptions = {}): Promise<unknown[] | null> {
    const cacheKey = `vector:${queryHash}`;
    return await this.get<unknown[]>(cacheKey, options);
  }

  // ============================================================================
  // CACHE INVALIDATION
  // ============================================================================

  async invalidateByTag(tag: string): Promise<number> {
    try {
      // Simple implementation: clear all cache
      await this.clear();
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
    layers?: unknown;
  }> {
    const layerStats = null; // Simple implementation
    
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
    layers?: unknown;
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

    const layerHealth = null; // Simple implementation

    return {
      healthy: serviceHealthy,
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
// NES CACHE ORCHESTRATOR INTEGRATION HELPERS
// ============================================================================

export async function initializeNESCacheIntegration(): Promise<boolean> {
  try {
    // Force initialization of NES Cache Orchestrator if not already done
    await (cachingService as any).initializeNESCacheOrchestrator();
    return true;
  } catch (error) {
    console.error('Failed to initialize NES Cache integration:', error);
    return false;
  }
}

export function getNESCacheStats(): any {
  const nesOrchestrator = (cachingService as any).nesCacheOrchestrator;
  if (nesOrchestrator) {
    return {
      initialized: true,
      memoryUsage: nesOrchestrator.getMemoryUsage?.() || 'N/A',
      cacheHierarchy: nesOrchestrator.getCacheHierarchy?.() || 'N/A',
      performance: nesOrchestrator.getPerformanceMetrics?.() || 'N/A'
    };
  }
  return { initialized: false };
}

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

export async function getCachedSearchResults(query: string, options?: SearchCacheOptions): Promise<unknown[] | null> {
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