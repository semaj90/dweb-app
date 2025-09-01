/**
 * Loki.js + Redis High-Performance Caching - Phase 14
 * 
 * Hybrid caching architecture combining:
 * - Loki.js: Fast in-memory document database with MongoDB-like queries
 * - Redis: Distributed cache with pub/sub for real-time synchronization
 * - Legal AI context awareness with document type optimization
 * - NES memory architecture integration for overflow management
 */

import Loki from 'lokijs';
import { redis } from '$lib/server/cache/redis-service';
import { nesMemory, type LegalDocument } from '../memory/nes-memory-architecture.js';
import { EventEmitter } from 'events';
import type { Redis as IORedisClient } from 'ioredis';
import crypto from "crypto";

// Cache configuration optimized for legal AI workloads
const CACHE_CONFIG = {
  // Loki.js settings
  loki: {
    autosave: true,
    autosaveInterval: 5000, // 5 seconds
    autoload: true,
    throttledSaves: true,
    serializationMethod: 'pretty'
  },
  
  // Redis settings
  redis: {
    host: 'localhost',
    port: 6379,
    db: 0,
    keyPrefix: 'legal_ai:',
    ttl: {
      documents: 3600,      // 1 hour for documents
      searches: 1800,       // 30 minutes for search results
      analyses: 7200,       // 2 hours for AI analyses
      embeddings: 86400     // 24 hours for vector embeddings
    }
  },
  
  // Memory management
  memory: {
    maxLokiSize: 50 * 1024 * 1024,  // 50MB in-memory limit
    evictionThreshold: 0.85,         // Evict when 85% full
    compressionThreshold: 1024,      // Compress documents > 1KB
    nesIntegration: true             // Use NES memory for overflow
  }
} as const;

export interface CachedDocument extends LegalDocument {
  cacheTimestamp: number;
  accessCount: number;
  cacheLocation: 'loki' | 'redis' | 'nes';
  compressed: boolean;
  syncStatus: 'synced' | 'dirty' | 'pending';
}

export interface SearchResult {
  id: string;
  document: LegalDocument;
  score: number;
  matchType: 'exact' | 'fuzzy' | 'semantic';
  highlight?: string[];
}

export interface CacheStats {
  loki: {
    collections: number;
    documents: number;
    memoryUsage: number;
    queries: number;
    hits: number;
    misses: number;
  };
  redis: {
    connected: boolean;
    keys: number;
    memoryUsage: number;
    operations: number;
    hits: number;
    misses: number;
  };
  nes: {
    documentsStored: number;
    memoryUsage: number;
    bankSwitches: number;
  };
  overall: {
    hitRatio: number;
    avgResponseTime: number;
    totalDocuments: number;
    syncConflicts: number;
  };
}

export class LokiRedisCache extends EventEmitter {
  private loki: Loki | null = null;
  private redis: IORedisClient | null = null;
  private subscriber: IORedisClient | null = null;
  
  // Loki collections by document type
  private collections: Map<string, Collection<CachedDocument>> = new Map();
  
  // Performance tracking
  private stats = {
    loki: { collections: 0, documents: 0, memoryUsage: 0, queries: 0, hits: 0, misses: 0 },
    redis: { connected: false, keys: 0, memoryUsage: 0, operations: 0, hits: 0, misses: 0 },
    nes: { documentsStored: 0, memoryUsage: 0, bankSwitches: 0 },
    overall: { hitRatio: 0, avgResponseTime: 0, totalDocuments: 0, syncConflicts: 0 }
  };
  
  private responseTimeTracker: number[] = [];
  private isInitialized = false;

  async initialize(): Promise<void> {
    try {
      await Promise.all([
        this.initializeLoki(),
        this.initializeRedis()
      ]);
      
      await this.setupSynchronization();
      this.startPerformanceMonitoring();
      
      this.isInitialized = true;
      this.emit('initialized');
      
      console.log('✅ Loki.js + Redis cache initialized successfully');
    } catch (error: any) {
      console.error('❌ Cache initialization failed:', error);
      throw error;
    }
  }

  private async initializeLoki(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.loki = new Loki('legal_ai_cache.db', {
        ...CACHE_CONFIG.loki,
        autoloadCallback: (err) => {
          if (err) {
            reject(err);
            return;
          }
          
          // Initialize collections for different document types
          const documentTypes = ['contract', 'evidence', 'brief', 'citation', 'precedent'];
          
          for (const type of documentTypes) {
            const collectionName = `documents_${type}`;
            let collection = this.loki!.getCollection<CachedDocument>(collectionName);
            
            if (!collection) {
              collection = this.loki!.addCollection<CachedDocument>(collectionName, {
                indices: ['id', 'cacheTimestamp', 'type', 'priority', 'riskLevel'],
                unique: ['id']
              });
            }
            
            this.collections.set(type, collection as any);
          }
          
          // Search results collection
          const searchCollection = this.loki!.getCollection('search_results') || 
                                  this.loki!.addCollection('search_results', {
                                    indices: ['query', 'timestamp']
                                  });
          
          this.collections.set('searches', searchCollection as any);
          this.stats.loki.collections = this.collections.size;
          
          resolve();
        }
      });
    });
  }

  private async initializeRedis(): Promise<void> {
    try {
      // Use centralized Redis service
      this.redis = redis;
      
      // Use centralized Redis service for subscriber as well
      this.subscriber = redis;

      // Redis service handles connection management

      this.stats.redis.connected = true;
      
      console.log('✅ Redis clients connected');
    } catch (error: any) {
      console.error('❌ Redis connection failed:', error);
      throw error;
    }
  }

  private async setupSynchronization(): Promise<void> {
    if (!this.subscriber) return;

    // Subscribe to document change events
    await this.subscriber.psubscribe('legal_ai:document:*');
    this.subscriber.on('pmessage', (pattern: string, channel: string, message: string) => {
      this.handleRedisMessage(message, channel);
    });

    // Subscribe to search invalidation events
    await this.subscriber.subscribe('legal_ai:search:invalidate');
    this.subscriber.on('message', (channel: string, message: string) => {
      if (channel === 'legal_ai:search:invalidate') {
        this.invalidateSearchCache(JSON.parse(message) as any);
      }
    });
  }

  private handleRedisMessage(message: string, channel: string): void {
    try {
      const data = JSON.parse(message) as any;
      const documentId = data.documentId;
      const operation = data.operation; // 'update' | 'delete' | 'create'
      
      // Update local Loki cache based on Redis changes
      switch (operation) {
        case 'update':
          this.updateLocalDocument(documentId, data.document);
          break;
        case 'delete':
          this.removeLocalDocument(documentId);
          break;
        case 'create':
          this.addLocalDocument(data.document);
          break;
      }
      
      this.emit('documentSynced', { documentId, operation });
    } catch (error: any) {
      console.error('❌ Failed to handle Redis message:', error);
      this.stats.overall.syncConflicts++;
    }
  }

  async storeDocument(document: LegalDocument, data?: ArrayBuffer): Promise<void> {
    const startTime = Date.now();
    
    try {
      const cachedDoc: CachedDocument = {
        ...document,
        cacheTimestamp: Date.now(),
        accessCount: 1,
        cacheLocation: 'loki',
        compressed: false,
        syncStatus: 'synced'
      };

      // Store in Loki.js first (fastest access)
      await this.storeLokiDocument(cachedDoc);
      
      // Store in Redis for distribution
      await this.storeRedisDocument(cachedDoc, data);
      
      // If memory pressure, consider NES memory for cold storage
      if (this.shouldUseNESMemory(cachedDoc)) {
        await this.storeNESDocument(cachedDoc, data);
      }

      this.updateStats('store', Date.now() - startTime);
      this.emit('documentStored', { documentId: document.id });
      
    } catch (error: any) {
      console.error(`❌ Failed to store document ${document.id}:`, error);
      throw error;
    }
  }

  private async storeLokiDocument(document: CachedDocument): Promise<void> {
    const collection = this.collections.get(document.type);
    if (!collection) {
      throw new Error(`No collection found for document type: ${document.type}`);
    }

    // Check memory pressure
    if (this.stats.loki.memoryUsage > CACHE_CONFIG.memory.maxLokiSize * CACHE_CONFIG.memory.evictionThreshold) {
      await this.evictLokiDocuments();
    }

    // Remove existing document if it exists
    const existing = collection.findOne({ id: document.id });
    if (existing) {
      collection.remove(existing);
    }

    collection.insert(document);
    this.stats.loki.documents++;
  }

  private async storeRedisDocument(document: CachedDocument, data?: ArrayBuffer): Promise<void> {
    if (!this.redis) return;

    const key = `${CACHE_CONFIG.redis.keyPrefix}doc:${document.id}`;
    const value = JSON.stringify({
      document,
      data: data ? Array.from(new Uint8Array(data)) : null
    });

    await this.redis.setex(key, CACHE_CONFIG.redis.ttl.documents, value);
    this.stats.redis.operations++;

    // Publish change event
    await this.redis.publish(`legal_ai:document:${document.type}`, JSON.stringify({
      documentId: document.id,
      operation: 'create',
      document
    }));
  }

  private shouldUseNESMemory(document: CachedDocument): boolean {
    if (!CACHE_CONFIG.memory.nesIntegration) return false;
    
    // Use NES memory for large, low-priority, or old documents
    return (
      document.size > 10240 ||                           // > 10KB
      document.priority < 100 ||                         // Low priority
      document.riskLevel === 'low' ||                    // Low risk
      Date.now() - document.cacheTimestamp > 300000     // Older than 5 minutes
    );
  }

  private async storeNESDocument(document: CachedDocument, data?: ArrayBuffer): Promise<void> {
    if (!data) return;

    const success = await nesMemory.allocateDocument(document, data, {
      compress: document.size > CACHE_CONFIG.memory.compressionThreshold,
      preferredBank: this.selectNESBank(document)
    });

    if (success) {
      document.cacheLocation = 'nes';
      this.stats.nes.documentsStored++;
    }
  }

  private selectNESBank(document: CachedDocument): string {
    // Select NES bank based on document characteristics
    if (document.riskLevel === 'critical' || document.priority > 200) {
      return 'INTERNAL_RAM'; // Fastest access
    }
    
    if (document.type === 'contract' || document.type === 'evidence') {
      return 'CHR_ROM'; // Pattern storage
    }
    
    if (document.type === 'brief' || document.type === 'precedent') {
      return 'PRG_ROM'; // Logic storage
    }
    
    return 'SAVE_RAM'; // Persistent storage
  }

  async getDocument(documentId: string): Promise<CachedDocument | null> {
    const startTime = Date.now();
    
    try {
      // Try Loki.js first (fastest)
      let document = await this.getLokiDocument(documentId);
      if (document) {
        document.accessCount++;
        document.cacheTimestamp = Date.now();
        this.stats.loki.hits++;
        this.updateStats('get', Date.now() - startTime);
        return document;
      }

      // Try Redis second
      document = await this.getRedisDocument(documentId);
      if (document) {
        // Promote to Loki for faster future access
        await this.storeLokiDocument(document);
        this.stats.redis.hits++;
        this.updateStats('get', Date.now() - startTime);
        return document;
      }

      // Try NES memory last
      document = await this.getNESDocument(documentId);
      if (document) {
        // Promote to higher cache levels if frequently accessed
        if (document.accessCount > 5) {
          await this.storeLokiDocument(document);
        }
        this.stats.nes.memoryUsage = nesMemory.getMemoryStats().documentCount;
        this.updateStats('get', Date.now() - startTime);
        return document;
      }

      // Cache miss
      this.stats.loki.misses++;
      this.stats.redis.misses++;
      return null;

    } catch (error: any) {
      console.error(`❌ Failed to get document ${documentId}:`, error);
      return null;
    }
  }

  private async getLokiDocument(documentId: string): Promise<CachedDocument | null> {
    for (const collection of this.collections.values()) {
      const document = collection.findOne({ id: documentId });
      if (document) {
        this.stats.loki.queries++;
        return document;
      }
    }
    return null;
  }

  private async getRedisDocument(documentId: string): Promise<CachedDocument | null> {
    if (!this.redis) return null;

    try {
      const key = `${CACHE_CONFIG.redis.keyPrefix}doc:${documentId}`;
      const value = await this.redis.get(key);
      
      if (value && typeof value === 'string') {
        const parsed = JSON.parse(value) as any;
        this.stats.redis.operations++;
        return parsed.document as any;
      }
    } catch (error: any) {
      console.error(`❌ Redis get error for ${documentId}:`, error);
    }
    
    return null;
  }

  private async getNESDocument(documentId: string): Promise<CachedDocument | null> {
    const nesDoc = nesMemory.getDocument(documentId);
    if (nesDoc) {
      return {
        ...nesDoc,
        cacheTimestamp: Date.now(),
        accessCount: (nesDoc as any).accessCount || 1,
        cacheLocation: 'nes',
        compressed: nesDoc.compressed,
        syncStatus: 'synced'
      };
    }
    return null;
  }

  async searchDocuments(
    query: string,
    filters: {
      type?: string[];
      riskLevel?: string[];
      confidenceMin?: number;
      priorityMin?: number;
    } = {},
    options: {
      limit?: number;
      useSemanticSearch?: boolean;
      cacheResults?: boolean;
    } = {}
  ): Promise<SearchResult[]> {
    const startTime = Date.now();
    const { limit = 50, useSemanticSearch = false, cacheResults = true } = options;

    try {
      // Check search cache first
      const cacheKey = this.generateSearchCacheKey(query, filters, options);
      let results = await this.getCachedSearchResults(cacheKey);
      
      if (results) {
        this.updateStats('search', Date.now() - startTime);
        return results;
      }

      // Perform search across all collections
      results = [];
      
      for (const [type, collection] of this.collections.entries()) {
        if (type === 'searches') continue;
        
        if (filters.type && !filters.type.includes(type as any)) continue;

        // Build Loki.js query
        let lokiQuery: any = {};
        
        if (filters.riskLevel) {
          lokiQuery.riskLevel = { $in: filters.riskLevel };
        }
        
        if (filters.confidenceMin) {
          lokiQuery.confidenceLevel = { $gte: filters.confidenceMin };
        }
        
        if (filters.priorityMin) {
          lokiQuery.priority = { $gte: filters.priorityMin };
        }

        // Text search (simple contains for now, could be enhanced with full-text search)
        if (query) {
          lokiQuery.$or = [
            { id: { $contains: query } },
            { 'metadata.caseId': { $contains: query } },
            { 'metadata.jurisdiction': { $contains: query } }
          ];
        }

        const documents = collection.find(lokiQuery);
        
        for (const doc of documents) {
          results.push({
            id: doc.id,
            document: doc,
            score: this.calculateRelevanceScore(doc, query),
            matchType: 'fuzzy'
          });
        }
      }

      // Sort by relevance and limit
      results.sort((a, b) => b.score - a.score);
      results = results.slice(0, limit);

      // Cache results if enabled
      if (cacheResults) {
        await this.cacheSearchResults(cacheKey, results);
      }

      this.updateStats('search', Date.now() - startTime);
      return results;

    } catch (error: any) {
      console.error('❌ Search failed:', error);
      return [];
    }
  }

  private calculateRelevanceScore(document: CachedDocument, query: string): number {
    let score = 0;

    // Base score from document priority and confidence
    score += document.priority * 0.01;
    score += document.confidenceLevel * 100;

    // Risk level scoring
    switch (document.riskLevel) {
      case 'critical': score += 50; break;
      case 'high': score += 30; break;
      case 'medium': score += 15; break;
      case 'low': score += 5; break;
    }

    // Access frequency boost
    score += Math.min(document.accessCount * 2, 20);

    // Recency boost (newer documents get higher scores)
    const age = Date.now() - document.cacheTimestamp;
    const daysSinceCache = age / (1000 * 60 * 60 * 24);
    score += Math.max(0, 10 - daysSinceCache);

    return score;
  }

  private generateSearchCacheKey(query: string, filters: any, options: any): string {
    const hashInput = JSON.stringify({ query, filters, options });
    // Simple hash function (could use crypto.createHash for production)
    let hash = 0;
    for (let i = 0; i < hashInput.length; i++) {
      const char = hashInput.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `search:${Math.abs(hash)}`;
  }

  private async getCachedSearchResults(cacheKey: string): Promise<SearchResult[] | null> {
    if (!this.redis) return null;

    try {
      const cached = await this.redis.get(`${CACHE_CONFIG.redis.keyPrefix}${cacheKey}`);
      if (cached && typeof cached === 'string') {
        return JSON.parse(cached) as any;
      }
    } catch (error: any) {
      console.error('❌ Search cache retrieval failed:', error);
    }
    
    return null;
  }

  private async cacheSearchResults(cacheKey: string, results: SearchResult[]): Promise<void> {
    if (!this.redis) return;

    try {
      const key = `${CACHE_CONFIG.redis.keyPrefix}${cacheKey}`;
      await this.redis.setex(key, CACHE_CONFIG.redis.ttl.searches, JSON.stringify(results));
    } catch (error: any) {
      console.error('❌ Search cache storage failed:', error);
    }
  }

  private async evictLokiDocuments(): Promise<void> {
    // Find least recently used documents across all collections
    const candidates: Array<{ collection: Collection<CachedDocument>; document: CachedDocument }> = [];
    
    for (const collection of this.collections.values()) {
      const documents = collection.find();
      for (const doc of documents) {
        if (doc.riskLevel !== 'critical' && doc.priority < 200) {
          candidates.push({ collection, document: doc });
        }
      }
    }

    // Sort by access patterns (LRU + priority)
    candidates.sort((a, b) => {
      const scoreA = a.document.accessCount + (a.document.priority / 255) * 10;
      const scoreB = b.document.accessCount + (b.document.priority / 255) * 10;
      return scoreA - scoreB;
    });

    // Evict 25% of candidates
    const evictCount = Math.ceil(candidates.length * 0.25);
    for (let i = 0; i < evictCount && i < candidates.length; i++) {
      const { collection, document } = candidates[i];
      collection.remove(document);
      this.stats.loki.documents--;
      
      // Consider moving to NES memory instead of full eviction
      if (CACHE_CONFIG.memory.nesIntegration) {
        // Would need document data to store in NES, which we don't have here
        // In a real implementation, we'd need to track this separately
      }
    }

    console.log(`🗑️ Evicted ${evictCount} documents from Loki cache`);
  }

  private async invalidateSearchCache(criteria: any): Promise<void> {
    if (!this.redis) return;

    try {
      // Clear all search cache keys matching criteria
      const pattern = `${CACHE_CONFIG.redis.keyPrefix}search:*`;
      const keys = await this.redis.keys(pattern);
      
      if (keys.length > 0) {
        await this.redis.del(keys);
        console.log(`🗑️ Invalidated ${keys.length} search cache entries`);
      }
    } catch (error: any) {
      console.error('❌ Search cache invalidation failed:', error);
    }
  }

  private updateLocalDocument(documentId: string, document: CachedDocument): void {
    const collection = this.collections.get(document.type);
    if (collection) {
      const existing = collection.findOne({ id: documentId });
      if (existing) {
        // Update existing document
        Object.assign(existing, document, {
          syncStatus: 'synced',
          cacheTimestamp: Date.now()
        });
        collection.update(existing);
      } else {
        // Insert new document
        collection.insert({
          ...document,
          syncStatus: 'synced',
          cacheTimestamp: Date.now()
        });
      }
    }
  }

  private removeLocalDocument(documentId: string): void {
    for (const collection of this.collections.values()) {
      const document = collection.findOne({ id: documentId });
      if (document) {
        collection.remove(document);
        this.stats.loki.documents--;
        break;
      }
    }
  }

  private addLocalDocument(document: CachedDocument): void {
    const collection = this.collections.get(document.type);
    if (collection) {
      collection.insert({
        ...document,
        syncStatus: 'synced',
        cacheTimestamp: Date.now()
      });
      this.stats.loki.documents++;
    }
  }

  private updateStats(operation: string, responseTime: number): void {
    this.responseTimeTracker.push(responseTime);
    if (this.responseTimeTracker.length > 1000) {
      this.responseTimeTracker = this.responseTimeTracker.slice(-1000);
    }

    const total = this.stats.loki.hits + this.stats.loki.misses + this.stats.redis.hits + this.stats.redis.misses;
    const hits = this.stats.loki.hits + this.stats.redis.hits;
    
    this.stats.overall.hitRatio = total > 0 ? hits / total : 0;
    this.stats.overall.avgResponseTime = this.responseTimeTracker.reduce((a, b) => a + b, 0) / this.responseTimeTracker.length;
    this.stats.overall.totalDocuments = this.stats.loki.documents + this.stats.nes.documentsStored;
  }

  private startPerformanceMonitoring(): void {
    setInterval(() => {
      this.updateMemoryStats();
      this.emit('stats', this.getStats());
    }, 10000); // Every 10 seconds
  }

  private updateMemoryStats(): void {
    // Estimate Loki memory usage
    this.stats.loki.memoryUsage = this.stats.loki.documents * 2048; // Rough estimate
    
    // Get NES memory stats
    const nesStats = nesMemory.getMemoryStats();
    this.stats.nes = {
      documentsStored: nesStats.documentCount,
      memoryUsage: nesStats.usedRAM + nesStats.usedCHR + nesStats.usedPRG,
      bankSwitches: nesStats.bankSwitches
    };
  }

  getStats(): CacheStats {
    return { ...this.stats };
  }

  async clear(): Promise<void> {
    try {
      // Clear Loki collections
      for (const collection of this.collections.values()) {
        collection.clear();
      }
      
      // Clear Redis cache
      if (this.redis) {
        const pattern = `${CACHE_CONFIG.redis.keyPrefix}*`;
        const keys = await this.redis.keys(pattern);
        if (keys.length > 0) {
          await this.redis.del(keys);
        }
      }
      
      // Reset stats
      this.stats = {
        loki: { collections: this.collections.size, documents: 0, memoryUsage: 0, queries: 0, hits: 0, misses: 0 },
        redis: { connected: this.stats.redis.connected, keys: 0, memoryUsage: 0, operations: 0, hits: 0, misses: 0 },
        nes: { documentsStored: 0, memoryUsage: 0, bankSwitches: 0 },
        overall: { hitRatio: 0, avgResponseTime: 0, totalDocuments: 0, syncConflicts: 0 }
      };

      console.log('✅ Cache cleared successfully');
    } catch (error: any) {
      console.error('❌ Failed to clear cache:', error);
      throw error;
    }
  }

  async destroy(): Promise<void> {
    try {
      // Save Loki database
      if (this.loki) {
        await new Promise<void>((resolve, reject) => {
          this.loki!.saveDatabase((err) => {
            if (err) reject(err);
            else resolve();
          });
        });
      }
      
      // Close Redis connections
      if (this.redis) {
        await this.redis.quit();
      }
      if (this.subscriber) {
        await this.subscriber.quit();
      }
      
      this.loki = null;
      this.redis = null;
      this.subscriber = null;
      this.collections.clear();
      this.isInitialized = false;
      
      console.log('✅ Loki.js + Redis cache destroyed');
    } catch (error: any) {
      console.error('❌ Cache destruction failed:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const lokiRedisCache = new LokiRedisCache();
;
