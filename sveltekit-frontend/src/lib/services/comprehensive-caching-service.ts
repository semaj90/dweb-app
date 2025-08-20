/**
 * Comprehensive Multi-Layer Caching Service
 * Clean rebuild after corruption: provides typed multi-layer caching with pluggable strategies.
 * Implemented layers: memory, IndexedDB (browser), LokiJS (optional), Redis (server), stubs for PostgreSQL & vector.
 */

import { writable, type Writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import { set as idbSet, get as idbGet, del as idbDel, clear as idbClear, keys as idbKeys } from 'idb-keyval';
// LokiJS is optional; wrap dynamic import & define minimal types to avoid build break if absent.
// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
type LokiDatabase = { getCollection: (name: string) => any; addCollection: (name: string, opts?: any) => any } | null;
// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
type LokiStatic = new (name: string, opts: any) => any;
let LokiRef: LokiStatic | null = null;

// Redis type (lightweight) – adapt path if a proper client wrapper exists.
// Avoid hard import if it causes client bundle issues; dynamic import used in initializeRedis.
export interface RedisLike {
  get(key: string): Promise<string | null>;
  setex(key: string, ttlSeconds: number, value: string): Promise<any>;
  del(key: string): Promise<any>;
}

// SIMD helper (optional) – safe wrapper. If unavailable operations will degrade gracefully.
const simdRedisClient: { healthCheck?: () => Promise<any>; cacheJSON?: (k: string, v: unknown) => Promise<any> } = {};

export interface CacheConfig {
  enableBrowserCache: boolean;
  enableIndexedDB: boolean;
  enableLokiJS: boolean;
  enableRedis: boolean;
  enablePostgreSQL: boolean;
  enableSIMD: boolean;
  enableVectorCache: boolean;
  defaultTTL: number; // ms
  maxMemoryUsage: number; // bytes
  evictionPolicy: 'lru' | 'lfu' | 'fifo' | 'ttl';
  compressionEnabled: boolean;
  encryptionEnabled: boolean;
}

export interface CacheEntry<T = any> {
  key: string;
  value: T;
  metadata: {
    size: number;
    ttl: number;
    createdAt: number;
    lastAccessed: number;
    accessCount: number;
    compressed: boolean;
    encrypted: boolean;
    layer: CacheLayer;
    tags: string[];
  };
}

export interface CacheStats {
  layers: Record<
    CacheLayer,
    {
      entries: number;
      size: number;
      hitRate: number;
      missRate: number;
      evictions: number;
    }
  >;
  overall: {
    totalEntries: number;
    totalSize: number;
    hitRate: number;
    missRate: number;
    layerDistribution: Record<CacheLayer, number>;
  };
}

export type CacheLayer =
  | "memory"
  | "indexeddb"
  | "lokijs"
  | "redis"
  | "postgresql"
  | "vector";

export interface CacheStrategy {
  name: string;
  layers: CacheLayer[];
  readOrder: CacheLayer[];
  writeOrder: CacheLayer[];
  evictionPolicy: "lru" | "lfu" | "fifo" | "ttl";
  ttl: number;
  maxSize: number;
}

class ComprehensiveCachingService {
  private static instance: ComprehensiveCachingService;
  private config: CacheConfig;
  private stats: Writable<CacheStats>;

  // Cache layers
  private memoryCache = new Map<string, CacheEntry>();
  private lokiDB: LokiDatabase = null;
  private lokiCollection: any | null = null; // Collection<CacheEntry>
  private redisClient: RedisLike | null = null;

  // Cache strategies
  private strategies = new Map<string, CacheStrategy>();
  private currentStrategy: CacheStrategy;

  // Performance tracking
  private hitCounts = new Map<CacheLayer, number>();
  private missCounts = new Map<CacheLayer, number>();
  private evictionCounts = new Map<CacheLayer, number>();

  private initialized = false;
  private simdEnabled = false;

  private constructor() {
    this.config = this.getDefaultConfig();
    this.stats = writable(this.getInitialStats());
    this.currentStrategy = this.getDefaultStrategy();
    this.initializeStrategies();
    this.initializeSIMD();
  }

  public static getInstance(): ComprehensiveCachingService {
    if (!ComprehensiveCachingService.instance) {
      ComprehensiveCachingService.instance = new ComprehensiveCachingService();
    }
    return ComprehensiveCachingService.instance;
  }

  private getDefaultConfig(): CacheConfig {
    return {
      enableBrowserCache: true,
      enableIndexedDB: true,
      enableLokiJS: true,
      enableRedis: true,
      enablePostgreSQL: true,
      enableSIMD: true,
      enableVectorCache: true,
      defaultTTL: 3600000, // 1 hour
      maxMemoryUsage: 512 * 1024 * 1024, // 512MB
      evictionPolicy: "lru",
      compressionEnabled: true,
      encryptionEnabled: false,
    };
  }

  private getInitialStats(): CacheStats {
    const layers: CacheLayer[] = [
      "memory",
      "indexeddb",
      "lokijs",
      "redis",
      "postgresql",
      "vector",
    ];
    const layerStats = layers.reduce(
      (acc, layer) => {
        acc[layer] = {
          entries: 0,
          size: 0,
          hitRate: 0,
          missRate: 0,
          evictions: 0,
        };
        return acc;
      },
      {} as Record<CacheLayer, any>
    );

    return {
      layers: layerStats,
      overall: {
        totalEntries: 0,
        totalSize: 0,
        hitRate: 0,
        missRate: 0,
        layerDistribution: layers.reduce(
          (acc, layer) => {
            acc[layer] = 0;
            return acc;
          },
          {} as Record<CacheLayer, number>
        ),
      },
    };
  }

  private getDefaultStrategy(): CacheStrategy {
    return {
      name: "multi-layer",
      layers: ["memory", "indexeddb", "lokijs", "redis", "postgresql"],
      readOrder: ["memory", "indexeddb", "lokijs", "redis", "postgresql"],
      writeOrder: ["memory", "indexeddb"],
      evictionPolicy: "lru",
      ttl: this.config.defaultTTL,
      maxSize: this.config.maxMemoryUsage,
    };
  }

  private initializeStrategies(): void {
    // Fast strategy - memory + indexeddb only
    this.strategies.set("fast", {
      name: "fast",
      layers: ["memory", "indexeddb"],
      readOrder: ["memory", "indexeddb"],
      writeOrder: ["memory", "indexeddb"],
      evictionPolicy: "lru",
      ttl: 1800000, // 30 minutes
      maxSize: 256 * 1024 * 1024, // 256MB
    });

    // Persistent strategy - all layers
    this.strategies.set("persistent", {
      name: "persistent",
      layers: ["memory", "indexeddb", "lokijs", "redis", "postgresql"],
      readOrder: ["memory", "indexeddb", "lokijs", "redis", "postgresql"],
      writeOrder: ["memory", "indexeddb", "lokijs", "redis", "postgresql"],
      evictionPolicy: "lfu",
      ttl: 86400000, // 24 hours
      maxSize: 1024 * 1024 * 1024, // 1GB
    });

    // Vector strategy - optimized for embeddings
    this.strategies.set("vector", {
      name: "vector",
      layers: ["memory", "vector", "postgresql"],
      readOrder: ["memory", "vector", "postgresql"],
      writeOrder: ["memory", "vector", "postgresql"],
      evictionPolicy: "lru",
      ttl: 7200000, // 2 hours
      maxSize: 512 * 1024 * 1024, // 512MB
    });

    // Session strategy - temporary data
    this.strategies.set("session", {
      name: "session",
      layers: ["memory"],
      readOrder: ["memory"],
      writeOrder: ["memory"],
      evictionPolicy: "fifo",
      ttl: 900000, // 15 minutes
      maxSize: 128 * 1024 * 1024, // 128MB
    });
  }

  public async initialize(): Promise<void> {
    try {
      // Initialize LokiJS
      if (this.config.enableLokiJS) {
        await this.initializeLokiJS();
      }

      // Initialize Redis connection
      if (this.config.enableRedis && !browser) {
        await this.initializeRedis();
      }

      // Initialize performance tracking
      this.initializePerformanceTracking();

      this.initialized = true;
      console.log("✅ Comprehensive Caching Service initialized");
    } catch (error) {
      console.error(
        "❌ Failed to initialize Comprehensive Caching Service:",
        error
      );
      throw error;
    }
  }

  private async initializeLokiJS(): Promise<void> {
    if (!this.config.enableLokiJS) return;
    try {
      if (!LokiRef) {
        // Dynamic import to keep client bundle lean if unused.
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const mod = await import('lokijs');
        LokiRef = mod.default || mod;
      }
      this.lokiDB = new (LokiRef as LokiStatic)('cache.db', {
        autoload: true,
        autoloadCallback: () => {
          this.lokiCollection = this.lokiDB!.getCollection('cache');
          if (!this.lokiCollection) {
            this.lokiCollection = this.lokiDB!.addCollection('cache', {
              indices: ['key'],
            });
          }
        },
        autosave: true,
        autosaveInterval: 10000,
      });
    } catch (e) {
      console.warn('LokiJS unavailable – continuing without it', e);
    }
  }

  private async initializeRedis(): Promise<void> {
    if (!this.config.enableRedis || browser) return;
    try {
      const mod = await import('$lib/server/redis');
      this.redisClient = (mod as any).REDIS_CONNECTION as RedisLike;
      console.log('✅ Redis connection initialized');
    } catch (error) {
      console.warn('⚠️ Redis connection failed, continuing without Redis cache');
    }
  }

  private async initializeSIMD(): Promise<void> {
    if (!this.config.enableSIMD) return;
    try {
      if (simdRedisClient.healthCheck) {
        await simdRedisClient.healthCheck();
        this.simdEnabled = true;
        console.log('✅ SIMD JSON processing enabled');
      }
    } catch {
      console.warn('⚠️ SIMD service unavailable, falling back');
    }
  }

  private initializePerformanceTracking(): void {
    const layers: CacheLayer[] = [
      "memory",
      "indexeddb",
      "lokijs",
      "redis",
      "postgresql",
      "vector",
    ];

    layers.forEach((layer) => {
      this.hitCounts.set(layer, 0);
      this.missCounts.set(layer, 0);
      this.evictionCounts.set(layer, 0);
    });

    // Update stats periodically
    setInterval(() => {
      this.updateStats();
    }, 30000); // Every 30 seconds
  }

  /**
   * Get value from cache using current strategy
   */
  public async get<T>(key: string, strategy?: string): Promise<T | null> {
    const cacheStrategy = strategy
      ? this.strategies.get(strategy)
      : this.currentStrategy;
    if (!cacheStrategy) {
      throw new Error(`Unknown cache strategy: ${strategy}`);
    }

    // Try each layer in read order
    for (const layer of cacheStrategy.readOrder) {
      try {
        const entry = await this.getFromLayer<T>(key, layer);
        if (entry && !this.isExpired(entry)) {
          // Update access metadata
          entry.metadata.lastAccessed = Date.now();
          entry.metadata.accessCount++;

          // Propagate to faster layers
          await this.propagateToFasterLayers(entry, layer, cacheStrategy);

          this.recordHit(layer);
          return entry.value;
        }
      } catch (error) {
        console.warn(`Cache layer ${layer} error:`, error);
      }
    }

    // Record miss for all layers in strategy
    cacheStrategy.readOrder.forEach((layer) => this.recordMiss(layer));
    return null;
  }

  /**
   * Set value in cache using current strategy
   */
  public async set<T>(
    key: string,
    value: T,
    options: {
      ttl?: number;
      tags?: string[];
      strategy?: string;
      compress?: boolean;
      encrypt?: boolean;
    } = {}
  ): Promise<void> {
    const cacheStrategy = options.strategy
      ? this.strategies.get(options.strategy)
      : this.currentStrategy;
    if (!cacheStrategy) {
      throw new Error(`Unknown cache strategy: ${options.strategy}`);
    }

    const entry: CacheEntry<T> = {
      key,
      value,
      metadata: {
        size: this.calculateSize(value),
        ttl: options.ttl || cacheStrategy.ttl,
        createdAt: Date.now(),
        lastAccessed: Date.now(),
        accessCount: 0,
        compressed: options.compress || this.config.compressionEnabled,
        encrypted: options.encrypt || this.config.encryptionEnabled,
        layer: "memory", // Will be updated per layer
        tags: options.tags || [],
      },
    };

    // Apply compression if enabled
    if (entry.metadata.compressed) {
      entry.value = (await this.compress(entry.value)) as T;
    }

    // Apply encryption if enabled
    if (entry.metadata.encrypted) {
      entry.value = (await this.encrypt(entry.value)) as T;
    }

    // Store in write order layers
    const promises = cacheStrategy.writeOrder.map(async (layer) => {
      try {
        const layerEntry = { ...entry };
        layerEntry.metadata.layer = layer;
        await this.setInLayer(layerEntry, layer);
      } catch (error) {
        console.warn(`Failed to set in layer ${layer}:`, error);
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Delete from all cache layers
   */
  public async delete(key: string): Promise<void> {
    const layers: CacheLayer[] = [
      "memory",
      "indexeddb",
      "lokijs",
      "redis",
      "postgresql",
      "vector",
    ];

    const promises = layers.map(async (layer) => {
      try {
        await this.deleteFromLayer(key, layer);
      } catch (error) {
        console.warn(`Failed to delete from layer ${layer}:`, error);
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Clear cache by tags
   */
  public async clearByTags(tags: string[]): Promise<number> {
    let clearedCount = 0;

    // Clear from memory cache
    for (const [key, entry] of this.memoryCache) {
      if (entry.metadata.tags.some((tag) => tags.includes(tag))) {
        // Flatten metadata or use direct property access in queries
        this.memoryCache.delete(key);
        clearedCount++;
      }
    }

    // Clear from LokiJS
    if (this.lokiCollection) {
      // Find all items first
      const allItems = this.lokiCollection.find({});
      const toRemove = allItems.filter((item: any) => {
        // Flatten metadata or use direct property access in queries
        return item.tags?.some((tag: string) => tags.includes(tag));
      });
      clearedCount += toRemove.length;

      // Remove each item by its Loki ID
      toRemove.forEach((item: any) => {
        this.lokiCollection!.remove(item);
      });
    }

    // Clear from other layers would be implemented here

    return clearedCount;
  }

  /**
   * Get cache statistics
   */
  public getStats(): CacheStats {
    this.updateStats();
    return get(this.stats);
  }

  /**
   * SIMD-accelerated cache operations for large JSON objects
   */
  public async getSIMD<T>(key: string): Promise<T | null> {
    if (!this.simdEnabled) {
      return this.get<T>(key);
    }

    try {
      // Try standard cache first
      const cached = await this.get<T>(key);
      if (cached) {
        return cached;
      }

      // If not in cache but SIMD is enabled, log for analytics
      console.log(`SIMD cache miss for key: ${key}`);
      return null;
    } catch (error) {
      console.warn(
        "SIMD get operation failed, falling back to standard cache:",
        error
      );
      return this.get<T>(key);
    }
  }

  /**
   * SIMD-accelerated cache set with ultra-fast JSON serialization
   */
  public async setSIMD<T>(
    key: string,
    value: T,
    ttl?: number,
    tags?: string[]
  ): Promise<void> {
    if (!this.simdEnabled) {
      return this.set(key, value, { ttl, tags });
    }
    try {
      const str = JSON.stringify(value);
      if (str.length > 1024 && simdRedisClient.cacheJSON) {
        await simdRedisClient.cacheJSON(`simd:${key}`, value);
        console.log(`SIMD cached large object: ${key} (${str.length} bytes)`);
      }
      return this.set(key, value, { ttl, tags });
    } catch (e) {
      console.warn('SIMD set failed – fallback', e);
      return this.set(key, value, { ttl, tags });
    }
  }

  /**
   * Switch cache strategy
   */
  public switchStrategy(strategyName: string): void {
    const strategy = this.strategies.get(strategyName);
    if (!strategy) {
      throw new Error(`Unknown strategy: ${strategyName}`);
    }

    this.currentStrategy = strategy;
    console.log(`✅ Switched to cache strategy: ${strategyName}`);
  }

  /**
   * Warm up cache with frequently accessed data
   */
  public async warmup(
    data: Array<{ key: string; value: any; options?: any }>
  ): Promise<void> {
    console.log(`🔥 Warming up cache with ${data.length} entries`);

    const promises = data.map(({ key, value, options }) =>
      this.set(key, value, { ...options, strategy: "fast" })
    );

    await Promise.allSettled(promises);
    console.log("✅ Cache warmup completed");
  }

  // Layer-specific implementations
  private async getFromLayer<T>(
    key: string,
    layer: CacheLayer
  ): Promise<CacheEntry<T> | null> {
    switch (layer) {
      case "memory":
        return (this.memoryCache.get(key) as CacheEntry<T>) || null;

      case 'indexeddb':
        if (!browser || !this.config.enableIndexedDB) return null;
        return ((await idbGet(key)) as CacheEntry<T>) || null;

      case "lokijs":
        if (!this.lokiCollection) return null;
        const lokiResult = this.lokiCollection.findOne({ key });
        return (lokiResult as CacheEntry<T>) || null;

      case 'redis':
        if (!this.redisClient) return null;
        try {
          const redisResult = await this.redisClient.get(key);
          return redisResult ? JSON.parse(redisResult) : null;
        } catch (e) {
          console.warn('Redis get failed', e); return null;
        }

      case "postgresql":
        // PostgreSQL cache implementation would go here
        return null;

      case "vector":
        // Vector cache implementation would go here
        return null;

      default:
        throw new Error(`Unknown cache layer: ${layer}`);
    }
  }

  private async setInLayer<T>(
    entry: CacheEntry<T>,
    layer: CacheLayer
  ): Promise<void> {
    switch (layer) {
      case "memory":
        this.memoryCache.set(entry.key, entry);
        this.enforceMemoryLimits();
        break;

      case 'indexeddb':
        if (browser && this.config.enableIndexedDB) {
          await idbSet(entry.key, entry);
        }
        break;

      case "lokijs":
        if (this.lokiCollection) {
          const existing = this.lokiCollection.findOne({ key: entry.key });
          if (existing) {
            Object.assign(existing, entry);
            this.lokiCollection.update(existing);
          } else {
            this.lokiCollection.insert(entry);
          }
        }
        break;

      case 'redis':
        if (this.redisClient) {
          try {
            await this.redisClient.setex(
              entry.key,
              Math.floor(entry.metadata.ttl / 1000),
              JSON.stringify(entry)
            );
          } catch (e) {
            console.warn('Redis set failed', e);
          }
        }
        break;

      case "postgresql":
        // PostgreSQL cache implementation would go here
        break;

      case "vector":
        // Vector cache implementation would go here
        break;

      default:
        throw new Error(`Unknown cache layer: ${layer}`);
    }
  }

  private async deleteFromLayer(key: string, layer: CacheLayer): Promise<void> {
    switch (layer) {
      case "memory":
        this.memoryCache.delete(key);
        break;

      case 'indexeddb':
        if (browser && this.config.enableIndexedDB) {
          await idbDel(key);
        }
        break;

      case "lokijs":
        if (this.lokiCollection) {
          this.lokiCollection.removeWhere({ key });
        }
        break;

      case 'redis':
        if (this.redisClient) {
          try { await this.redisClient.del(key); } catch (e) { console.warn('Redis delete failed', e); }
        }
        break;

      case "postgresql":
        // PostgreSQL cache implementation would go here
        break;

      case "vector":
        // Vector cache implementation would go here
        break;
    }
  }

  // Utility methods
  private async propagateToFasterLayers<T>(
    entry: CacheEntry<T>,
    currentLayer: CacheLayer,
    strategy: CacheStrategy
  ): Promise<void> {
    const currentIndex = strategy.readOrder.indexOf(currentLayer);
    const fasterLayers = strategy.readOrder.slice(0, currentIndex);

    const promises = fasterLayers.map((layer) => this.setInLayer(entry, layer));
    await Promise.allSettled(promises);
  }

  private isExpired(entry: CacheEntry): boolean {
    return Date.now() > entry.metadata.createdAt + entry.metadata.ttl;
  }

  private calculateSize(value: any): number {
    return JSON.stringify(value).length * 2; // Rough estimation in bytes
  }

  private async compress<T>(value: T): Promise<T> {
    // Compression implementation would go here
    // For now, return as-is
    return value;
  }

  private async encrypt<T>(value: T): Promise<T> {
    // Encryption implementation would go here
    // For now, return as-is
    return value;
  }

  private enforceMemoryLimits(): void {
    const memorySize = Array.from(this.memoryCache.values()).reduce(
      (total, entry) => total + entry.metadata.size,
      0
    );

    if (memorySize > this.currentStrategy.maxSize) {
      this.evictFromMemory();
    }
  }

  private evictFromMemory(): void {
    const entries = Array.from(this.memoryCache.entries());

    // Sort by eviction policy
    entries.sort(([, a], [, b]) => {
      switch (this.currentStrategy.evictionPolicy) {
        case "lru":
          return a.metadata.lastAccessed - b.metadata.lastAccessed;
        case "lfu":
          return a.metadata.accessCount - b.metadata.accessCount;
        case "fifo":
          return a.metadata.createdAt - b.metadata.createdAt;
        case "ttl":
          return (
            a.metadata.createdAt +
            a.metadata.ttl -
            (b.metadata.createdAt + b.metadata.ttl)
          );
        default:
          return 0;
      }
    });

    // Remove 25% of entries
    const toRemove = Math.ceil(entries.length * 0.25);
    for (let i = 0; i < toRemove; i++) {
      const [key] = entries[i];
      this.memoryCache.delete(key);
      this.recordEviction("memory");
    }
  }

  private recordHit(layer: CacheLayer): void {
    const current = this.hitCounts.get(layer) || 0;
    this.hitCounts.set(layer, current + 1);
  }

  private recordMiss(layer: CacheLayer): void {
    const current = this.missCounts.get(layer) || 0;
    this.missCounts.set(layer, current + 1);
  }

  private recordEviction(layer: CacheLayer): void {
    const current = this.evictionCounts.get(layer) || 0;
    this.evictionCounts.set(layer, current + 1);
  }

  private updateStats(): void {
    const layers: CacheLayer[] = [
      "memory",
      "indexeddb",
      "lokijs",
      "redis",
      "postgresql",
      "vector",
    ];

    const layerStats = layers.reduce(
      (acc, layer) => {
        const hits = this.hitCounts.get(layer) || 0;
        const misses = this.missCounts.get(layer) || 0;
        const total = hits + misses;

        acc[layer] = {
          entries: this.getLayerEntryCount(layer),
          size: this.getLayerSize(layer),
          hitRate: total > 0 ? hits / total : 0,
          missRate: total > 0 ? misses / total : 0,
          evictions: this.evictionCounts.get(layer) || 0,
        };

        return acc;
      },
      {} as Record<CacheLayer, any>
    );

    const totalEntries = Object.values(layerStats).reduce(
      (sum, stats) => sum + stats.entries,
      0
    );
    const totalHits = Array.from(this.hitCounts.values()).reduce(
      (sum, hits) => sum + hits,
      0
    );
    const totalMisses = Array.from(this.missCounts.values()).reduce(
      (sum, misses) => sum + misses,
      0
    );
    const totalRequests = totalHits + totalMisses;

    this.stats.set({
      layers: layerStats,
      overall: {
        totalEntries,
        totalSize: Object.values(layerStats).reduce(
          (sum, stats) => sum + stats.size,
          0
        ),
        hitRate: totalRequests > 0 ? totalHits / totalRequests : 0,
        missRate: totalRequests > 0 ? totalMisses / totalRequests : 0,
        layerDistribution: layers.reduce(
          (acc, layer) => {
            acc[layer] = layerStats[layer].entries;
            return acc;
          },
          {} as Record<CacheLayer, number>
        ),
      },
    });
  }

  private getLayerEntryCount(layer: CacheLayer): number {
    switch (layer) {
      case "memory":
        return this.memoryCache.size;
      case "lokijs":
        return this.lokiCollection?.count() || 0;
      default:
        return 0; // Would need actual implementation for other layers
    }
  }

  private getLayerSize(layer: CacheLayer): number {
    switch (layer) {
      case "memory":
        return Array.from(this.memoryCache.values()).reduce(
          (total, entry) => total + entry.metadata.size,
          0
        );
      default:
        return 0; // Would need actual implementation for other layers
    }
  }

  // Public getters
  public get isInitialized(): boolean {
    return this.initialized;
  }

  public get currentStrategyName(): string {
    return this.currentStrategy.name;
  }

  public getAvailableStrategies(): string[] {
    return Array.from(this.strategies.keys());
  }

  public get statsStore(): Writable<CacheStats> {
    return this.stats;
  }
}

// Export singleton instance
export const comprehensiveCachingService =
  ComprehensiveCachingService.getInstance();
export default comprehensiveCachingService;

// Types are already exported above
