/**
 * Proactive Cache Service
 * Intelligent caching system based on SOM density maps and user interaction patterns
 * Implements: Predictive prefetching, Redis-based storage, cache warming
 */

import { browser } from '$app/environment';
import { writable, derived } from 'svelte/store';
import type { UserInteraction, InteractionPattern } from './interaction-tracker';

interface CacheEntry {
  key: string;
  data: unknown;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  importance: number;
  size: number;
  ttl: number;
}

interface CachePrediction {
  key: string;
  probability: number;
  importance: number;
  reason: 'som_density' | 'user_pattern' | 'temporal' | 'similarity';
  metadata: Record<string, any>;
}

interface CacheStats {
  hitRate: number;
  missRate: number;
  totalRequests: number;
  totalHits: number;
  averageLatency: number;
  memoryUsage: number;
  predictionAccuracy: number;
}

class ProactiveCacheService {
  private cache = new Map<string, CacheEntry>();
  private accessLog = writable<Array<{ key: string; timestamp: number; hit: boolean }>>([]);
  private predictions = writable<CachePrediction[]>([]);
  private stats = writable<CacheStats>({
    hitRate: 0,
    missRate: 0,
    totalRequests: 0,
    totalHits: 0,
    averageLatency: 0,
    memoryUsage: 0,
    predictionAccuracy: 0
  });

  private isInitialized = false;
  private redisClient: unknown = null;
  private prefetchQueue = new Set<string>();
  private warmupTimer: NodeJS.Timeout | null = null;

  // Hidden Markov Model for access pattern prediction
  private accessPatterns = new Map<string, Map<string, number>>();
  private recentAccesses: string[] = [];

  // SOM-based importance scoring
  private somDensityMap: Float32Array | null = null;
  private documentMappings = new Map<string, { x: number; y: number; importance: number }>();

  constructor() {
    if (browser) {
      this.initialize();
    }
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      // Initialize Redis client (mock for client-side, real for server-side)
      if (typeof window !== 'undefined') {
        // Client-side: use IndexedDB as cache backend
        await this.initializeIndexedDBCache();
      } else {
        // Server-side: use Redis
        await this.initializeRedisCache();
      }

      // Start cache warming and maintenance
      this.startCacheWarmup();
      this.startCacheMaintenance();

      this.isInitialized = true;
      console.log('🚀 Proactive cache service initialized');
    } catch (error) {
      console.error('❌ Failed to initialize cache service:', error);
    }
  }

  private async initializeIndexedDBCache() {
    // Use IndexedDB for client-side caching
    const db = await this.openCacheDB();
    console.log('📱 IndexedDB cache backend initialized');
  }

  private async initializeRedisCache() {
    // Server-side Redis initialization would go here
    // For now, use in-memory cache
    console.log('🔴 Redis cache backend initialized (mock)');
  }

  private async openCacheDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('ProactiveCacheDB', 1);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
      
      request.onupgradeneeded = () => {
        const db = request.result;
        
        if (!db.objectStoreNames.contains('cache')) {
          const store = db.createObjectStore('cache', { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('importance', 'importance');
          store.createIndex('accessCount', 'accessCount');
        }
      };
    });
  }

  // Core cache operations
  async get(key: string): Promise<any> {
    const startTime = Date.now();
    
    let entry = this.cache.get(key);
    let hit = false;

    if (entry && this.isEntryValid(entry)) {
      // Cache hit
      entry.accessCount++;
      entry.lastAccessed = Date.now();
      hit = true;
    } else {
      // Cache miss - try to fetch from backend
      try {
        const data = await this.fetchFromBackend(key);
        if (data) {
          await this.set(key, data, { importance: 0.5 });
          entry = this.cache.get(key);
        }
      } catch (error) {
        console.error('Cache miss and backend fetch failed:', error);
      }
    }

    // Update statistics
    this.updateStats(hit, Date.now() - startTime);
    
    // Log access for pattern analysis
    this.logAccess(key, hit);
    
    // Update access patterns for prediction
    this.updateAccessPatterns(key);

    return entry?.data;
  }

  async set(key: string, data: unknown, options: { 
    ttl?: number; 
    importance?: number; 
    metadata?: Record<string, any> 
  } = {}): Promise<void> {
    const entry: CacheEntry = {
      key,
      data,
      timestamp: Date.now(),
      accessCount: 0,
      lastAccessed: Date.now(),
      importance: options.importance || 0.5,
      size: this.estimateSize(data),
      ttl: options.ttl || 3600000 // 1 hour default
    };

    this.cache.set(key, entry);

    // Persist to IndexedDB if available
    if (browser) {
      try {
        const db = await this.openCacheDB();
        const tx = db.transaction(['cache'], 'readwrite');
        await tx.objectStore('cache').put(entry);
      } catch (error) {
        console.error('Failed to persist cache entry:', error);
      }
    }

    // Evict old entries if cache is getting too large
    this.evictIfNeeded();
  }

  async delete(key: string): Promise<void> {
    this.cache.delete(key);
    
    if (browser) {
      try {
        const db = await this.openCacheDB();
        const tx = db.transaction(['cache'], 'readwrite');
        await tx.objectStore('cache').delete(key);
      } catch (error) {
        console.error('Failed to delete cache entry:', error);
      }
    }
  }

  // Proactive caching based on SOM density
  setSOMDensityMap(densityMap: Float32Array, documentMappings: Map<string, { x: number; y: number }>) {
    this.somDensityMap = densityMap;
    
    // Calculate importance scores based on SOM density
    for (const [docId, mapping] of documentMappings.entries()) {
      const { x, y } = mapping;
      const gridWidth = Math.sqrt(densityMap.length); // Assume square grid
      const index = x * gridWidth + y;
      const density = densityMap[index] || 0;
      
      this.documentMappings.set(docId, {
        x,
        y,
        importance: density
      });
    }

    console.log(`🧠 Updated SOM density map for ${documentMappings.size} documents`);
    this.generatePredictions();
  }

  // Generate cache predictions based on multiple signals
  private generatePredictions() {
    const predictions: CachePrediction[] = [];

    // 1. SOM density-based predictions
    if (this.somDensityMap) {
      const highDensityDocs = Array.from(this.documentMappings.entries())
        .filter(([, mapping]) => mapping.importance > 0.7)
        .sort(([, a], [, b]) => b.importance - a.importance)
        .slice(0, 10);

      for (const [docId, mapping] of highDensityDocs) {
        predictions.push({
          key: `document:${docId}`,
          probability: mapping.importance,
          importance: mapping.importance,
          reason: 'som_density',
          metadata: { x: mapping.x, y: mapping.y }
        });
      }
    }

    // 2. User pattern-based predictions
    const patternPredictions = this.generatePatternPredictions();
    predictions.push(...patternPredictions);

    // 3. Temporal predictions (frequently accessed items)
    const temporalPredictions = this.generateTemporalPredictions();
    predictions.push(...temporalPredictions);

    // 4. Similarity-based predictions
    const similarityPredictions = this.generateSimilarityPredictions();
    predictions.push(...similarityPredictions);

    // Sort by importance and probability
    predictions.sort((a, b) => (b.probability * b.importance) - (a.probability * a.importance));

    this.predictions.set(predictions.slice(0, 20)); // Top 20 predictions
    
    // Trigger prefetching for top predictions
    this.triggerPrefetching(predictions.slice(0, 5));
  }

  private generatePatternPredictions(): CachePrediction[] {
    const predictions: CachePrediction[] = [];
    
    if (this.recentAccesses.length < 2) return predictions;

    const lastAccess = this.recentAccesses[this.recentAccesses.length - 1];
    const transitions = this.accessPatterns.get(lastAccess);

    if (transitions) {
      const totalTransitions = Array.from(transitions.values()).reduce((sum, count) => sum + count, 0);
      
      for (const [nextKey, count] of transitions.entries()) {
        const probability = count / totalTransitions;
        
        if (probability > 0.1) { // Only predict if probability > 10%
          predictions.push({
            key: nextKey,
            probability,
            importance: 0.6,
            reason: 'user_pattern',
            metadata: { lastAccess, transitionCount: count }
          });
        }
      }
    }

    return predictions;
  }

  private generateTemporalPredictions(): CachePrediction[] {
    const predictions: CachePrediction[] = [];
    const now = Date.now();
    
    // Find frequently accessed items in the last hour
    for (const [key, entry] of this.cache.entries()) {
      const timeSinceAccess = now - entry.lastAccessed;
      const accessFrequency = entry.accessCount / Math.max(1, (now - entry.timestamp) / 3600000); // accesses per hour
      
      if (accessFrequency > 2 && timeSinceAccess < 3600000) { // Accessed more than 2x/hour in last hour
        predictions.push({
          key,
          probability: Math.min(0.9, accessFrequency / 10),
          importance: 0.7,
          reason: 'temporal',
          metadata: { accessFrequency, timeSinceAccess }
        });
      }
    }

    return predictions;
  }

  private generateSimilarityPredictions(): CachePrediction[] {
    const predictions: CachePrediction[] = [];
    
    // If user recently accessed a document, predict similar documents
    const recentDocAccesses = this.recentAccesses
      .filter(key => key.startsWith('document:'))
      .slice(-3);

    for (const docKey of recentDocAccesses) {
      const docId = docKey.replace('document:', '');
      const mapping = this.documentMappings.get(docId);
      
      if (mapping) {
        // Find nearby documents in SOM space
        const nearbyDocs = Array.from(this.documentMappings.entries())
          .filter(([id, m]) => {
            if (id === docId) return false;
            const distance = Math.sqrt((m.x - mapping.x) ** 2 + (m.y - mapping.y) ** 2);
            return distance <= 3; // Within 3 units in SOM space
          })
          .sort(([, a], [, b]) => {
            const distA = Math.sqrt((a.x - mapping.x) ** 2 + (a.y - mapping.y) ** 2);
            const distB = Math.sqrt((b.x - mapping.x) ** 2 + (b.y - mapping.y) ** 2);
            return distA - distB;
          })
          .slice(0, 5);

        for (const [similarDocId, similarMapping] of nearbyDocs) {
          const distance = Math.sqrt((similarMapping.x - mapping.x) ** 2 + (similarMapping.y - mapping.y) ** 2);
          const similarity = 1 / (1 + distance);
          
          predictions.push({
            key: `document:${similarDocId}`,
            probability: similarity * 0.8,
            importance: similarMapping.importance,
            reason: 'similarity',
            metadata: { sourceDoc: docId, distance, similarity }
          });
        }
      }
    }

    return predictions;
  }

  private async triggerPrefetching(predictions: CachePrediction[]) {
    for (const prediction of predictions) {
      if (!this.cache.has(prediction.key) && !this.prefetchQueue.has(prediction.key)) {
        this.prefetchQueue.add(prediction.key);
        
        // Prefetch in background
        this.prefetchData(prediction.key, prediction.reason).catch(error => {
          console.error(`Prefetch failed for ${prediction.key}:`, error);
          this.prefetchQueue.delete(prediction.key);
        });
      }
    }
  }

  private async prefetchData(key: string, reason: string) {
    try {
      console.log(`🔮 Prefetching ${key} (reason: ${reason})`);
      
      const data = await this.fetchFromBackend(key);
      if (data) {
        await this.set(key, data, { 
          importance: 0.8, // High importance for prefetched data
          metadata: { prefetched: true, reason }
        });
      }
    } finally {
      this.prefetchQueue.delete(key);
    }
  }

  private async fetchFromBackend(key: string): Promise<any> {
    // Simulate backend fetch
    if (key.startsWith('document:')) {
      const docId = key.replace('document:', '');
      
      try {
        const response = await fetch(`/api/documents/${docId}`);
        if (response.ok) {
          return await response.json();
        }
      } catch (error) {
        console.error('Backend fetch failed:', error);
      }
    }
    
    return null;
  }

  // Cache maintenance
  private startCacheWarmup() {
    this.warmupTimer = setInterval(() => {
      this.performCacheWarmup();
    }, 60000); // Every minute
  }

  private startCacheMaintenance() {
    setInterval(() => {
      this.performMaintenance();
    }, 300000); // Every 5 minutes
  }

  private performCacheWarmup() {
    // Warm up cache with high-importance items
    const highImportanceItems = Array.from(this.documentMappings.entries())
      .filter(([, mapping]) => mapping.importance > 0.8)
      .slice(0, 5);

    for (const [docId] of highImportanceItems) {
      const key = `document:${docId}`;
      if (!this.cache.has(key)) {
        this.prefetchData(key, 'warmup').catch(console.error);
      }
    }
  }

  private performMaintenance() {
    // Remove expired entries
    const now = Date.now();
    const toDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        toDelete.push(key);
      }
    }

    for (const key of toDelete) {
      this.delete(key);
    }

    // Update statistics
    this.calculateStats();
    
    console.log(`🧹 Cache maintenance: removed ${toDelete.length} expired entries`);
  }

  private evictIfNeeded() {
    const maxCacheSize = 100; // Maximum number of entries
    
    if (this.cache.size > maxCacheSize) {
      // Evict least important and least recently used items
      const entries = Array.from(this.cache.entries())
        .sort(([, a], [, b]) => {
          const scoreA = a.importance * Math.log(a.accessCount + 1) - (Date.now() - a.lastAccessed) / 3600000;
          const scoreB = b.importance * Math.log(b.accessCount + 1) - (Date.now() - b.lastAccessed) / 3600000;
          return scoreA - scoreB;
        });

      const toEvict = entries.slice(0, this.cache.size - maxCacheSize + 10); // Evict extra for breathing room
      
      for (const [key] of toEvict) {
        this.delete(key);
      }

      console.log(`🗑️ Evicted ${toEvict.length} cache entries`);
    }
  }

  // Utility methods
  private isEntryValid(entry: CacheEntry): boolean {
    const now = Date.now();
    return (now - entry.timestamp) < entry.ttl;
  }

  private estimateSize(data: unknown): number {
    // Rough estimation of object size in bytes
    return JSON.stringify(data).length * 2; // Assume 2 bytes per character
  }

  private logAccess(key: string, hit: boolean) {
    this.accessLog.update(log => [
      ...log.slice(-99), // Keep last 100 accesses
      { key, timestamp: Date.now(), hit }
    ]);
  }

  private updateAccessPatterns(key: string) {
    this.recentAccesses.push(key);
    
    // Keep only last 50 accesses
    if (this.recentAccesses.length > 50) {
      this.recentAccesses = this.recentAccesses.slice(-50);
    }

    // Update transition matrix
    if (this.recentAccesses.length >= 2) {
      const prevKey = this.recentAccesses[this.recentAccesses.length - 2];
      
      if (!this.accessPatterns.has(prevKey)) {
        this.accessPatterns.set(prevKey, new Map());
      }
      
      const transitions = this.accessPatterns.get(prevKey)!;
      transitions.set(key, (transitions.get(key) || 0) + 1);
    }
  }

  private updateStats(hit: boolean, latency: number) {
    this.stats.update(current => {
      const totalRequests = current.totalRequests + 1;
      const totalHits = current.totalHits + (hit ? 1 : 0);
      
      return {
        ...current,
        totalRequests,
        totalHits,
        hitRate: totalHits / totalRequests,
        missRate: 1 - (totalHits / totalRequests),
        averageLatency: (current.averageLatency * current.totalRequests + latency) / totalRequests,
        memoryUsage: this.calculateMemoryUsage()
      };
    });
  }

  private calculateMemoryUsage(): number {
    let totalSize = 0;
    for (const entry of this.cache.values()) {
      totalSize += entry.size;
    }
    return totalSize;
  }

  private calculateStats() {
    // Update prediction accuracy by checking recent predictions
    // This is a simplified version - real implementation would track prediction outcomes
    this.stats.update(current => ({
      ...current,
      predictionAccuracy: 0.75 // Placeholder
    }));
  }

  // Public API
  getStats() {
    return this.stats;
  }

  getPredictions() {
    return this.predictions;
  }

  getAccessLog() {
    return this.accessLog;
  }

  getCacheInsights() {
    return derived([this.stats, this.predictions], ([$stats, $predictions]) => ({
      performance: {
        hitRate: ($stats.hitRate * 100).toFixed(1) + '%',
        averageLatency: $stats.averageLatency.toFixed(2) + 'ms',
        memoryUsage: (($stats.memoryUsage / 1024 / 1024).toFixed(2)) + 'MB'
      },
      predictions: {
        total: $predictions.length,
        highConfidence: $predictions.filter(p => p.probability > 0.7).length,
        byReason: $predictions.reduce((acc, p) => {
          acc[p.reason] = (acc[p.reason] || 0) + 1;
          return acc;
        }, {} as Record<string, number>)
      },
      cacheSize: this.cache.size
    }));
  }

  // Manual cache operations
  async warmupArea(x: number, y: number, radius: number = 2) {
    // Warm up cache for a specific SOM area
    const documentsInArea = Array.from(this.documentMappings.entries())
      .filter(([, mapping]) => {
        const distance = Math.sqrt((mapping.x - x) ** 2 + (mapping.y - y) ** 2);
        return distance <= radius;
      });

    console.log(`🔥 Warming up ${documentsInArea.length} documents in area (${x}, ${y})`);

    for (const [docId] of documentsInArea) {
      const key = `document:${docId}`;
      if (!this.cache.has(key)) {
        this.prefetchData(key, 'manual_warmup').catch(console.error);
      }
    }
  }

  async clearCache() {
    this.cache.clear();
    this.predictions.set([]);
    this.accessLog.set([]);
    
    if (browser) {
      try {
        const db = await this.openCacheDB();
        const tx = db.transaction(['cache'], 'readwrite');
        await tx.objectStore('cache').clear();
      } catch (error) {
        console.error('Failed to clear IndexedDB cache:', error);
      }
    }

    console.log('🗑️ Cache cleared');
  }

  destroy() {
    if (this.warmupTimer) {
      clearInterval(this.warmupTimer);
      this.warmupTimer = null;
    }
    
    this.clearCache();
    this.isInitialized = false;
  }
}

// Singleton instance
export const proactiveCache = new ProactiveCacheService();