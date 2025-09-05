// Multi-Layer Caching System for NLP and Legal-BERT Operations
// Redis (server) + LokiJS (client) + Memory caching with XState management

import { browser } from '$app/environment';
import { createMachine, interpret } from 'xstate';
import type { SynthesizedAnalysis } from '../middleware/tfjs-synthesizer.js';
import type { LegalBERTAnalysis } from '../services/legal-bert-middleware.js';
import type { AdvancedExtractionResult } from '../services/langextract-tfjs.js';

// LokiJS dynamic import for client-side
let Loki: any = null;
if (browser) {
  import('lokijs').then(module => {
    Loki = module.default;
  });
}

export interface CacheConfig {
  enableRedisCache: boolean;
  enableLokiCache: boolean;
  enableMemoryCache: boolean;
  redisTTL: number;
  lokiTTL: number;
  memoryTTL: number;
  maxMemorySize: number;
  maxLokiSize: number;
  compressionEnabled: boolean;
  encryptionEnabled: boolean;
}

export interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: number;
  ttl: number;
  accessCount: number;
  size: number;
  compressed: boolean;
  encrypted: boolean;
  source: 'memory' | 'loki' | 'redis' | 'computation';
  hash: string;
}

export interface CacheStats {
  memory: {
    hits: number;
    misses: number;
    size: number;
    entries: number;
    hitRate: number;
  };
  loki: {
    hits: number;
    misses: number;
    size: number;
    entries: number;
    hitRate: number;
  };
  redis: {
    hits: number;
    misses: number;
    size: number;
    entries: number;
    hitRate: number;
  };
  overall: {
    totalHits: number;
    totalMisses: number;
    overallHitRate: number;
  };
}

export interface NLPCacheOperations {
  legalBERT: Map<string, CacheEntry<LegalBERTAnalysis>>;
  languageExtraction: Map<string, CacheEntry<AdvancedExtractionResult>>;
  synthesizedAnalysis: Map<string, CacheEntry<SynthesizedAnalysis>>;
  embeddings: Map<string, CacheEntry<Float32Array>>;
  summaries: Map<string, CacheEntry<string>>;
}

// XState machine for cache management
const cacheStateMachine = createMachine({
  id: 'multiLayerCache',
  initial: 'idle',
  context: {
    pendingOperations: 0,
    errors: [],
    lastCleanup: 0
  },
  states: {
    idle: {
      on: {
        GET: 'retrieving',
        SET: 'storing',
        CLEANUP: 'cleaning'
      }
    },
    retrieving: {
      invoke: {
        id: 'retrieve',
        src: 'retrieveData',
        onDone: {
          target: 'idle',
          actions: 'onRetrieveSuccess'
        },
        onError: {
          target: 'error',
          actions: 'onRetrieveError'
        }
      }
    },
    storing: {
      invoke: {
        id: 'store',
        src: 'storeData',
        onDone: {
          target: 'idle',
          actions: 'onStoreSuccess'
        },
        onError: {
          target: 'error',
          actions: 'onStoreError'
        }
      }
    },
    cleaning: {
      invoke: {
        id: 'cleanup',
        src: 'cleanupCache',
        onDone: {
          target: 'idle',
          actions: 'onCleanupSuccess'
        },
        onError: {
          target: 'error',
          actions: 'onCleanupError'
        }
      }
    },
    error: {
      on: {
        RETRY: 'idle'
      }
    }
  }
});

/**
 * Multi-Layer Cache Service for NLP Operations
 * Implements Redis + LokiJS + Memory caching with intelligent fallback
 */
export class MultiLayerCache {
  private config: CacheConfig;
  private memoryCache = new Map<string, CacheEntry<any>>();
  private lokiDB: any = null;
  private nlpCaches: NLPCacheOperations;
  private stats: CacheStats;
  private cacheActor: any;
  private initialized = false;

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = {
      enableRedisCache: true,
      enableLokiCache: browser,
      enableMemoryCache: true,
      redisTTL: 3600, // 1 hour
      lokiTTL: 1800, // 30 minutes
      memoryTTL: 900, // 15 minutes
      maxMemorySize: 50 * 1024 * 1024, // 50MB
      maxLokiSize: 100 * 1024 * 1024, // 100MB
      compressionEnabled: true,
      encryptionEnabled: false,
      ...config
    };

    this.nlpCaches = {
      legalBERT: new Map(),
      languageExtraction: new Map(),
      synthesizedAnalysis: new Map(),
      embeddings: new Map(),
      summaries: new Map()
    };

    this.stats = {
      memory: { hits: 0, misses: 0, size: 0, entries: 0, hitRate: 0 },
      loki: { hits: 0, misses: 0, size: 0, entries: 0, hitRate: 0 },
      redis: { hits: 0, misses: 0, size: 0, entries: 0, hitRate: 0 },
      overall: { totalHits: 0, totalMisses: 0, overallHitRate: 0 }
    };

    // Initialize XState cache manager
    this.cacheActor = interpret(cacheStateMachine.withConfig({
      services: {
        retrieveData: this.retrieveFromLayers.bind(this),
        storeData: this.storeInLayers.bind(this),
        cleanupCache: this.performCleanup.bind(this)
      },
      actions: {
        onRetrieveSuccess: (context, event) => {
          console.log('[Cache] Retrieve operation completed');
        },
        onStoreSuccess: (context, event) => {
          console.log('[Cache] Store operation completed');
        },
        onCleanupSuccess: (context, event) => {
          console.log('[Cache] Cleanup operation completed');
        },
        onRetrieveError: (context, event) => {
          console.error('[Cache] Retrieve operation failed:', event.data);
        },
        onStoreError: (context, event) => {
          console.error('[Cache] Store operation failed:', event.data);
        },
        onCleanupError: (context, event) => {
          console.error('[Cache] Cleanup operation failed:', event.data);
        }
      }
    }));
  }

  /**
   * Initialize multi-layer cache system
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    try {
      console.log('[Multi-Layer Cache] Initializing caching system...');

      // Start XState cache manager
      this.cacheActor.start();

      // Initialize LokiJS if in browser
      if (this.config.enableLokiCache && browser && Loki) {
        await this.initializeLokiDB();
      }

      // Test Redis connection if enabled
      if (this.config.enableRedisCache) {
        await this.testRedisConnection();
      }

      // Start periodic cleanup
      this.startPeriodicCleanup();

      this.initialized = true;
      console.log('[Multi-Layer Cache] Initialization complete');
      return true;

    } catch (error: any) {
      console.error('[Multi-Layer Cache] Initialization failed:', error);
      return false;
    }
  }

  /**
   * Get cached Legal-BERT analysis
   */
  async getLegalBERTAnalysis(textHash: string): Promise<LegalBERTAnalysis | null> {
    return await this.get('legalBERT', textHash);
  }

  /**
   * Cache Legal-BERT analysis
   */
  async setLegalBERTAnalysis(textHash: string, analysis: LegalBERTAnalysis): Promise<void> {
    await this.set('legalBERT', textHash, analysis, this.config.redisTTL);
  }

  /**
   * Get cached language extraction results
   */
  async getLanguageExtractionResults(textHash: string): Promise<AdvancedExtractionResult | null> {
    return await this.get('languageExtraction', textHash);
  }

  /**
   * Cache language extraction results
   */
  async setLanguageExtractionResults(textHash: string, results: AdvancedExtractionResult): Promise<void> {
    await this.set('languageExtraction', textHash, results, this.config.redisTTL);
  }

  /**
   * Get cached synthesized analysis
   */
  async getSynthesizedAnalysis(textHash: string): Promise<SynthesizedAnalysis | null> {
    return await this.get('synthesizedAnalysis', textHash);
  }

  /**
   * Cache synthesized analysis
   */
  async setSynthesizedAnalysis(textHash: string, analysis: SynthesizedAnalysis): Promise<void> {
    await this.set('synthesizedAnalysis', textHash, analysis, this.config.redisTTL);
  }

  /**
   * Get cached embeddings
   */
  async getEmbeddings(textHash: string): Promise<Float32Array | null> {
    return await this.get('embeddings', textHash);
  }

  /**
   * Cache embeddings
   */
  async setEmbeddings(textHash: string, embeddings: Float32Array): Promise<void> {
    await this.set('embeddings', textHash, embeddings, this.config.redisTTL * 2); // Longer TTL for embeddings
  }

  /**
   * Get cached summary
   */
  async getSummary(textHash: string): Promise<string | null> {
    return await this.get('summaries', textHash);
  }

  /**
   * Cache summary
   */
  async setSummary(textHash: string, summary: string): Promise<void> {
    await this.set('summaries', textHash, summary, this.config.redisTTL);
  }

  /**
   * Generic get method with multi-layer fallback
   */
  private async get<T>(cacheType: keyof NLPCacheOperations, key: string): Promise<T | null> {
    const fullKey = `${cacheType}:${key}`;

    try {
      // Layer 1: Memory cache (fastest)
      if (this.config.enableMemoryCache) {
        const memoryResult = await this.getFromMemory<T>(fullKey);
        if (memoryResult !== null) {
          this.stats.memory.hits++;
          this.updateOverallStats();
          return memoryResult;
        }
        this.stats.memory.misses++;
      }

      // Layer 2: LokiJS cache (fast, persistent across page reloads)
      if (this.config.enableLokiCache && this.lokiDB) {
        const lokiResult = await this.getFromLoki<T>(fullKey);
        if (lokiResult !== null) {
          // Store in memory for next time
          if (this.config.enableMemoryCache) {
            await this.setInMemory(fullKey, lokiResult, this.config.memoryTTL);
          }
          this.stats.loki.hits++;
          this.updateOverallStats();
          return lokiResult;
        }
        this.stats.loki.misses++;
      }

      // Layer 3: Redis cache (network call, but shared across users)
      if (this.config.enableRedisCache) {
        const redisResult = await this.getFromRedis<T>(fullKey);
        if (redisResult !== null) {
          // Store in faster layers for next time
          if (this.config.enableLokiCache && this.lokiDB) {
            await this.setInLoki(fullKey, redisResult, this.config.lokiTTL);
          }
          if (this.config.enableMemoryCache) {
            await this.setInMemory(fullKey, redisResult, this.config.memoryTTL);
          }
          this.stats.redis.hits++;
          this.updateOverallStats();
          return redisResult;
        }
        this.stats.redis.misses++;
      }

      this.updateOverallStats();
      return null;

    } catch (error: any) {
      console.error('[Multi-Layer Cache] Get operation failed:', error);
      return null;
    }
  }

  /**
   * Generic set method with multi-layer storage
   */
  private async set<T>(cacheType: keyof NLPCacheOperations, key: string, value: T, ttl: number): Promise<void> {
    const fullKey = `${cacheType}:${key}`;

    try {
      // Store in all enabled layers
      const storePromises: Promise<void>[] = [];

      if (this.config.enableMemoryCache) {
        storePromises.push(this.setInMemory(fullKey, value, Math.min(ttl, this.config.memoryTTL)));
      }

      if (this.config.enableLokiCache && this.lokiDB) {
        storePromises.push(this.setInLoki(fullKey, value, Math.min(ttl, this.config.lokiTTL)));
      }

      if (this.config.enableRedisCache) {
        storePromises.push(this.setInRedis(fullKey, value, ttl));
      }

      await Promise.allSettled(storePromises);

    } catch (error: any) {
      console.error('[Multi-Layer Cache] Set operation failed:', error);
    }
  }

  /**
   * Memory cache operations
   */
  private async getFromMemory<T>(key: string): Promise<T | null> {
    const entry = this.memoryCache.get(key);
    if (!entry) return null;

    if (this.isExpired(entry)) {
      this.memoryCache.delete(key);
      return null;
    }

    entry.accessCount++;
    return entry.value as T;
  }

  private async setInMemory<T>(key: string, value: T, ttl: number): Promise<void> {
    // Check memory size limit
    if (this.getMemorySize() > this.config.maxMemorySize) {
      await this.evictFromMemory();
    }

    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: Date.now(),
      ttl: ttl * 1000, // Convert to milliseconds
      accessCount: 1,
      size: this.estimateSize(value),
      compressed: false,
      encrypted: false,
      source: 'memory',
      hash: await this.generateHash(key)
    };

    this.memoryCache.set(key, entry);
    this.updateMemoryStats();
  }

  /**
   * LokiJS cache operations
   */
  private async getFromLoki<T>(key: string): Promise<T | null> {
    if (!this.lokiDB) return null;

    try {
      const collection = this.lokiDB.getCollection('nlp_cache') || this.lokiDB.addCollection('nlp_cache');
      const doc = collection.findOne({ key });

      if (!doc) return null;

      const entry = doc.entry as CacheEntry<T>;
      if (this.isExpired(entry)) {
        collection.remove(doc);
        this.lokiDB.saveDatabase();
        return null;
      }

      entry.accessCount++;
      collection.update(doc);
      this.lokiDB.saveDatabase();

      return entry.value;

    } catch (error: any) {
      console.error('[Multi-Layer Cache] LokiJS get failed:', error);
      return null;
    }
  }

  private async setInLoki<T>(key: string, value: T, ttl: number): Promise<void> {
    if (!this.lokiDB) return;

    try {
      const collection = this.lokiDB.getCollection('nlp_cache') || this.lokiDB.addCollection('nlp_cache');

      const entry: CacheEntry<T> = {
        key,
        value,
        timestamp: Date.now(),
        ttl: ttl * 1000,
        accessCount: 1,
        size: this.estimateSize(value),
        compressed: this.config.compressionEnabled,
        encrypted: this.config.encryptionEnabled,
        source: 'loki',
        hash: await this.generateHash(key)
      };

      // Remove existing entry if present
      const existing = collection.findOne({ key });
      if (existing) {
        collection.remove(existing);
      }

      collection.insert({ key, entry });
      this.lokiDB.saveDatabase();

    } catch (error: any) {
      console.error('[Multi-Layer Cache] LokiJS set failed:', error);
    }
  }

  /**
   * Redis cache operations (API calls to backend)
   */
  private async getFromRedis<T>(key: string): Promise<T | null> {
    try {
      const response = await fetch('/api/cache/get', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key })
      });

      if (!response.ok) {
        return null;
      }

      const data = await response.json();
      return data.value as T;

    } catch (error: any) {
      console.error('[Multi-Layer Cache] Redis get failed:', error);
      return null;
    }
  }

  private async setInRedis<T>(key: string, value: T, ttl: number): Promise<void> {
    try {
      await fetch('/api/cache/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, value, ttl })
      });

    } catch (error: any) {
      console.error('[Multi-Layer Cache] Redis set failed:', error);
    }
  }

  /**
   * Initialize LokiJS database
   */
  private async initializeLokiDB(): Promise<void> {
    if (!Loki) {
      console.warn('[Multi-Layer Cache] LokiJS not available');
      return;
    }

    return new Promise((resolve, reject) => {
      this.lokiDB = new Loki('nlp_cache.db', {
        autoload: true,
        autoloadCallback: () => {
          console.log('[Multi-Layer Cache] LokiJS database loaded');
          resolve();
        },
        autosave: true,
        autosaveInterval: 4000
      });
    });
  }

  /**
   * Test Redis connection
   */
  private async testRedisConnection(): Promise<void> {
    try {
      const response = await fetch('/api/cache/health');
      if (!response.ok) {
        throw new Error('Redis health check failed');
      }
      console.log('[Multi-Layer Cache] Redis connection verified');
    } catch (error: any) {
      console.warn('[Multi-Layer Cache] Redis not available:', error.message);
      this.config.enableRedisCache = false;
    }
  }

  /**
   * Generate text hash for cache keys
   */
  async generateTextHash(text: string): Promise<string> {
    if (browser && crypto.subtle) {
      const encoder = new TextEncoder();
      const data = encoder.encode(text);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } else {
      // Fallback hash function for environments without crypto.subtle
      let hash = 0;
      for (let i = 0; i < text.length; i++) {
        const char = text.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
      }
      return Math.abs(hash).toString(16);
    }
  }

  // Helper methods

  private async generateHash(text: string): Promise<string> {
    return await this.generateTextHash(text);
  }

  private isExpired(entry: CacheEntry<any>): boolean {
    return Date.now() - entry.timestamp > entry.ttl;
  }

  private estimateSize(value: any): number {
    try {
      return new Blob([JSON.stringify(value)]).size;
    } catch {
      return 0;
    }
  }

  private getMemorySize(): number {
    let totalSize = 0;
    for (const [, entry] of this.memoryCache) {
      totalSize += entry.size;
    }
    return totalSize;
  }

  private async evictFromMemory(): Promise<void> {
    // LRU eviction based on access count and age
    const entries = Array.from(this.memoryCache.entries());
    entries.sort(([, a], [, b]) => {
      const scoreA = a.accessCount / (Date.now() - a.timestamp);
      const scoreB = b.accessCount / (Date.now() - b.timestamp);
      return scoreA - scoreB;
    });

    // Remove bottom 25% of entries
    const toRemove = Math.ceil(entries.length * 0.25);
    for (let i = 0; i < toRemove; i++) {
      this.memoryCache.delete(entries[i][0]);
    }

    this.updateMemoryStats();
  }

  private updateMemoryStats(): void {
    this.stats.memory.entries = this.memoryCache.size;
    this.stats.memory.size = this.getMemorySize();
  }

  private updateOverallStats(): void {
    const { memory, loki, redis } = this.stats;
    const totalHits = memory.hits + loki.hits + redis.hits;
    const totalMisses = memory.misses + loki.misses + redis.misses;
    
    this.stats.overall = {
      totalHits,
      totalMisses,
      overallHitRate: totalHits + totalMisses > 0 ? totalHits / (totalHits + totalMisses) : 0
    };

    // Update individual hit rates
    memory.hitRate = memory.hits + memory.misses > 0 ? memory.hits / (memory.hits + memory.misses) : 0;
    loki.hitRate = loki.hits + loki.misses > 0 ? loki.hits / (loki.hits + loki.misses) : 0;
    redis.hitRate = redis.hits + redis.misses > 0 ? redis.hits / (redis.hits + redis.misses) : 0;
  }

  private startPeriodicCleanup(): void {
    setInterval(() => {
      this.cacheActor.send('CLEANUP');
    }, 300000); // Cleanup every 5 minutes
  }

  private async performCleanup(): Promise<void> {
    console.log('[Multi-Layer Cache] Performing periodic cleanup...');
    
    // Clean expired entries from memory cache
    for (const [key, entry] of this.memoryCache) {
      if (this.isExpired(entry)) {
        this.memoryCache.delete(key);
      }
    }

    // Clean expired entries from LokiJS
    if (this.lokiDB) {
      const collection = this.lokiDB.getCollection('nlp_cache');
      if (collection) {
        const expired = collection.find({
          'entry.timestamp': { '$lt': Date.now() - this.config.lokiTTL * 1000 }
        });
        collection.remove(expired);
        this.lokiDB.saveDatabase();
      }
    }

    this.updateMemoryStats();
  }

  private async retrieveFromLayers(context: any, event: any): Promise<any> {
    // This is called by XState machine
    return null; // Implementation would go here
  }

  private async storeInLayers(context: any, event: any): Promise<void> {
    // This is called by XState machine
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): CacheStats {
    this.updateOverallStats();
    return { ...this.stats };
  }

  /**
   * Clear all caches
   */
  async clearAll(): Promise<void> {
    // Clear memory cache
    this.memoryCache.clear();

    // Clear LokiJS cache
    if (this.lokiDB) {
      this.lokiDB.deleteDatabase();
    }

    // Clear Redis cache via API
    if (this.config.enableRedisCache) {
      try {
        await fetch('/api/cache/clear', { method: 'POST' });
      } catch (error) {
        console.error('[Multi-Layer Cache] Failed to clear Redis cache:', error);
      }
    }

    // Reset stats
    this.stats = {
      memory: { hits: 0, misses: 0, size: 0, entries: 0, hitRate: 0 },
      loki: { hits: 0, misses: 0, size: 0, entries: 0, hitRate: 0 },
      redis: { hits: 0, misses: 0, size: 0, entries: 0, hitRate: 0 },
      overall: { totalHits: 0, totalMisses: 0, overallHitRate: 0 }
    };

    console.log('[Multi-Layer Cache] All caches cleared');
  }

  /**
   * Get health status
   */
  getHealthStatus(): {
    initialized: boolean;
    memoryEnabled: boolean;
    lokiEnabled: boolean;
    redisEnabled: boolean;
    overallHitRate: number;
    totalEntries: number;
  } {
    return {
      initialized: this.initialized,
      memoryEnabled: this.config.enableMemoryCache,
      lokiEnabled: this.config.enableLokiCache && !!this.lokiDB,
      redisEnabled: this.config.enableRedisCache,
      overallHitRate: this.stats.overall.overallHitRate,
      totalEntries: this.stats.memory.entries + this.stats.loki.entries + this.stats.redis.entries
    };
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.cacheActor.stop();
    this.memoryCache.clear();
    
    if (this.lokiDB) {
      this.lokiDB.close();
    }

    this.initialized = false;
    console.log('[Multi-Layer Cache] Resources disposed');
  }
}

// Export singleton instance
export const multiLayerCache = new MultiLayerCache();