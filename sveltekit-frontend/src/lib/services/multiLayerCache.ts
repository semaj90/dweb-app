import crypto from "crypto";
// @ts-nocheck
Loki from 'lokijs';
// Orphaned content: import Fuse from "fuse.js";
import {

import type { SearchResult } from './aiPipeline.js';

export interface CacheEntry {
  id: string;
  key: string;
  value: any;
  metadata: {
    type: 'query' | 'document' | 'embedding' | 'search' | 'recommendation';
    createdAt: Date;
    lastAccessed: Date;
    accessCount: number;
    ttl: number; // Time to live in seconds
    size: number; // Size in bytes
    userId?: string;
    tags?: string[];
  };
}

export interface CacheStats {
  totalEntries: number;
  totalSize: number;
  hitRate: number;
  evictionCount: number;
  avgAccessTime: number;
  layerStats: {
    memory: { entries: number; size: number; hitRate: number };
    persistent: { entries: number; size: number; hitRate: number };
    search: { entries: number; queries: number };
  };
}

export interface FuseSearchOptions {
  keys: string[];
  threshold?: number;
  limit?: number;
  includeScore?: boolean;
}

export class MultiLayerCache {
  private memoryDb: Loki;
  private persistentDb: Loki | null = null;
  private cacheCollection: any; // Collection<CacheEntry>
  private searchCollection: any; // Collection<any>
  private fuseInstances: Map<string, Fuse<any>> = new Map();
  
  // Cache statistics
  private stats = {
    hits: 0,
    misses: 0,
    evictions: 0,
    totalAccessTime: 0,
    accessCount: 0
  };

  // Configuration
  private readonly maxMemorySize = 50 * 1024 * 1024; // 50MB
  private readonly maxPersistentSize = 200 * 1024 * 1024; // 200MB
  private readonly defaultTTL = 3600; // 1 hour

  constructor() {
    // Initialize in-memory database
    this.memoryDb = new Loki('multiLayerCache.db', {
      env: browser ? 'BROWSER' : 'NODEJS'
    });

    // Create collections
    this.cacheCollection = this.memoryDb.addCollection<CacheEntry>('cache', {
      indices: ['key'],
      ttl: this.defaultTTL * 1000,
      ttlInterval: 60000 // Check every minute
    });

    this.searchCollection = this.memoryDb.addCollection('searchIndex', {
      indices: ['type', 'userId']
    });

    // Initialize persistent storage if in browser
    if (browser) {
      this.initPersistentStorage();
    }

    // Start cleanup interval
    this.startCleanupInterval();
  }

  /**
   * Initialize persistent storage using IndexedDB
   */
  private async initPersistentStorage() {
    try {
      this.persistentDb = new Loki('multiLayerCache.persistent.db', {
        adapter: new LokiIndexedAdapter('multiLayerCache'),
        autoload: true,
        autosave: true,
        autosaveInterval: 10000 // Save every 10 seconds
      });

      await new Promise((resolve) => {
        this.persistentDb!.loadDatabase({}, resolve);
      });
    } catch (error) {
      console.error('Failed to initialize persistent storage:', error);
    }
  }

  /**
   * Set a value in the cache with multi-layer storage
   */
  async set(
    key: string,
    value: any,
    options: {
      type: CacheEntry['metadata']['type'];
      ttl?: number;
      userId?: string;
      tags?: string[];
      persistent?: boolean;
    }
  ): Promise<void> {
    const startTime = Date.now();
    
    try {
      const size = this.calculateSize(value);
      const entry: CacheEntry = {
        id: crypto.randomUUID(),
        key,
        value,
        metadata: {
          type: options.type,
          createdAt: new Date(),
          lastAccessed: new Date(),
          accessCount: 0,
          ttl: options.ttl || this.defaultTTL,
          size,
          userId: options.userId,
          tags: options.tags
        }
      };

      // Check memory size and evict if necessary
      await this.evictIfNecessary(size);

      // Store in memory cache
      this.cacheCollection.insert(entry);

      // Store in persistent cache if requested and available
      if (options.persistent && this.persistentDb) {
        const persistentCollection = this.getPersistentCollection(options.type);
        persistentCollection.insert(entry);
      }

      // Update search index if it's searchable content
      if (options.type === 'document' || options.type === 'search') {
        this.updateSearchIndex(entry);
      }

      // Update access time stats
      this.stats.totalAccessTime += Date.now() - startTime;
      this.stats.accessCount++;
    } catch (error) {
      console.error('Cache set error:', error);
      throw error;
    }
  }

  /**
   * Get a value from the cache (checks all layers)
   */
  async get<T = any>(
    key: string,
    options?: { userId?: string }
  ): Promise<T | null> {
    const startTime = Date.now();

    try {
      // Check memory cache first
      let entry = this.cacheCollection.findOne({
        key,
        ...(options?.userId && { 'metadata.userId': options.userId })
      });

      if (entry) {
        this.stats.hits++;
        this.updateAccessMetadata(entry);
        return entry.value as T;
      }

      // Check persistent cache if available
      if (this.persistentDb) {
        const collections = ['query', 'document', 'embedding', 'search', 'recommendation'];
        for (const type of collections) {
          const collection = this.getPersistentCollection(type as any);
          entry = collection.findOne({
            key,
            ...(options?.userId && { 'metadata.userId': options.userId })
          });

          if (entry) {
            this.stats.hits++;
            // Promote to memory cache
            this.cacheCollection.insert(entry);
            this.updateAccessMetadata(entry);
            return entry.value as T;
          }
        }
      }

      this.stats.misses++;
      return null;
    } finally {
      this.stats.totalAccessTime += Date.now() - startTime;
      this.stats.accessCount++;
    }
  }

  /**
   * Perform fuzzy search using Fuse.js
   */
  async fuzzySearch<T = any>(
    collection: string,
    query: string,
    options: FuseSearchOptions
  ): Promise<Array<{ item: T; score?: number }>> {
    // Get or create Fuse instance for the collection
    const fuseKey = `${collection}-${JSON.stringify(options.keys)}`;
    let fuse = this.fuseInstances.get(fuseKey);

    if (!fuse) {
      // Get all documents from the collection
      const docs = this.searchCollection.find({ type: collection });
      
      if (docs.length === 0) {
        return [];
      }

      // Create Fuse instance with configuration
      fuse = new Fuse(docs, {
        keys: options.keys,
        threshold: options.threshold || 0.6,
        includeScore: options.includeScore !== false,
        minMatchCharLength: 2,
        findAllMatches: false,
        location: 0,
        distance: 100,
        useExtendedSearch: true
      });

      this.fuseInstances.set(fuseKey, fuse);
    }

    // Perform search
    const results = fuse.search(query).slice(0, options.limit || 10);

    return results.map((result: any) => ({
      item: result.item as T,
      score: result.score
    }));
  }

  /**
   * Search for documents with advanced filtering
   */
  async searchDocuments(
    query: string,
    filters?: {
      type?: string;
      userId?: string;
      tags?: string[];
      dateRange?: { start: Date; end: Date };
    }
  ): Promise<SearchResult[]> {
    // Build Loki query
    const lokiQuery: any = { type: 'document' };
    
    if (filters?.userId) {
      lokiQuery['metadata.userId'] = filters.userId;
    }

    if (filters?.dateRange) {
      lokiQuery['metadata.createdAt'] = {
        $gte: filters.dateRange.start,
        $lte: filters.dateRange.end
      };
    }

    // Get documents from search collection
    const documents = this.searchCollection.find(lokiQuery);

    // Use Fuse.js for fuzzy search on content
    const fuse = new Fuse(documents, {
      keys: ['content', 'title', 'summary'],
      threshold: 0.4,
      includeScore: true,
      minMatchCharLength: 3
    });

    const searchResults = fuse.search(query);

    // Filter by tags if specified
    let filteredResults = searchResults;
    if (filters?.tags && filters.tags.length > 0) {
      filteredResults = searchResults.filter((result: any) => {
        const docTags = result.item.metadata?.tags || [];
        return filters.tags!.some((tag: any) => docTags.includes(tag));
      });
    }

    // Convert to SearchResult format
    return filteredResults.map((result: any) => ({
      id: result.item.id,
      content: result.item.content,
      score: 1 - (result.score || 0), // Convert Fuse score to similarity
      metadata: result.item.metadata
    }));
  }

  /**
   * Invalidate cache entries
   */
  async invalidate(
    pattern: string | RegExp,
    options?: { type?: string; userId?: string }
  ): Promise<number> {
    const query: any = {};

    if (options?.type) {
      query['metadata.type'] = options.type;
    }

    if (options?.userId) {
      query['metadata.userId'] = options.userId;
    }

    // Find matching entries
    const entries = this.cacheCollection.find(query);
    let invalidatedCount = 0;

    for (const entry of entries) {
      const matches = pattern instanceof RegExp
        ? pattern.test(entry.key)
        : entry.key.includes(pattern);

      if (matches) {
        this.cacheCollection.remove(entry);
        invalidatedCount++;
      }
    }

    // Also invalidate in persistent storage
    if (this.persistentDb && options?.type) {
      const collection = this.getPersistentCollection(options.type as any);
      const persistentEntries = collection.find(query);
      
      for (const entry of persistentEntries) {
        const matches = pattern instanceof RegExp
          ? pattern.test(entry.key)
          : entry.key.includes(pattern);

        if (matches) {
          collection.remove(entry);
        }
      }
    }

    return invalidatedCount;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const memoryEntries = this.cacheCollection.count();
    const memorySize = this.cacheCollection.data.reduce(
      (sum, entry) => sum + entry.metadata.size,
      0
    );

    const searchEntries = this.searchCollection.count();
    
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? this.stats.hits / (this.stats.hits + this.stats.misses)
      : 0;

    const avgAccessTime = this.stats.accessCount > 0
      ? this.stats.totalAccessTime / this.stats.accessCount
      : 0;

    return {
      totalEntries: memoryEntries,
      totalSize: memorySize,
      hitRate,
      evictionCount: this.stats.evictions,
      avgAccessTime,
      layerStats: {
        memory: {
          entries: memoryEntries,
          size: memorySize,
          hitRate
        },
        persistent: {
          entries: 0, // Would need to count from persistent DB
          size: 0,
          hitRate: 0
        },
        search: {
          entries: searchEntries,
          queries: this.fuseInstances.size
        }
      }
    };
  }

  /**
   * Clear all cache entries
   */
  async clear(options?: { type?: string; userId?: string }): Promise<void> {
    if (!options) {
      // Clear everything
      this.cacheCollection.clear();
      this.searchCollection.clear();
      this.fuseInstances.clear();
      
      if (this.persistentDb) {
        // Clear all persistent collections
        const types = ['query', 'document', 'embedding', 'search', 'recommendation'];
        for (const type of types) {
          const collection = this.getPersistentCollection(type as any);
          collection.clear();
        }
      }
    } else {
      // Clear specific entries
      const query: any = {};
      if (options.type) query['metadata.type'] = options.type;
      if (options.userId) query['metadata.userId'] = options.userId;

      const entries = this.cacheCollection.find(query);
      for (const entry of entries) {
        this.cacheCollection.remove(entry);
      }

      // Clear from search index
      const searchEntries = this.searchCollection.find(query);
      for (const entry of searchEntries) {
        this.searchCollection.remove(entry);
      }
    }

    // Reset stats
    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      totalAccessTime: 0,
      accessCount: 0
    };
  }

  /**
   * Calculate size of a value
   */
  private calculateSize(value: any): number {
    if (typeof value === 'string') {
      return value.length * 2; // Approximate UTF-16 size
    }
    return JSON.stringify(value).length * 2;
  }

  /**
   * Evict entries if necessary
   */
  private async evictIfNecessary(requiredSize: number): Promise<void> {
    const currentSize = this.cacheCollection.data.reduce(
      (sum, entry) => sum + entry.metadata.size,
      0
    );

    if (currentSize + requiredSize <= this.maxMemorySize) {
      return;
    }

    // Sort by last accessed (LRU) - manual sort since simplesort has issues with nested properties
    const allEntries = this.cacheCollection.data;
    const sortedEntries = allEntries.sort((a, b) => {
      const aTime = new Date(a.metadata.lastAccessed).getTime();
      const bTime = new Date(b.metadata.lastAccessed).getTime();
      return aTime - bTime; // Ascending order (oldest first)
    });

    let freedSize = 0;
    for (const entry of sortedEntries) {
      if (currentSize - freedSize + requiredSize <= this.maxMemorySize) {
        break;
      }

      this.cacheCollection.remove(entry);
      freedSize += entry.metadata.size;
      this.stats.evictions++;
    }
  }

  /**
   * Update access metadata
   */
  private updateAccessMetadata(entry: CacheEntry): void {
    entry.metadata.lastAccessed = new Date();
    entry.metadata.accessCount++;
    this.cacheCollection.update(entry);
  }

  /**
   * Update search index
   */
  private updateSearchIndex(entry: CacheEntry): void {
    if (entry.metadata.type === 'document' || entry.metadata.type === 'search') {
      this.searchCollection.insert({
        id: entry.id,
        type: entry.metadata.type,
        userId: entry.metadata.userId,
        content: entry.value.content || entry.value,
        title: entry.value.title,
        summary: entry.value.summary,
        metadata: entry.metadata,
        tags: entry.metadata.tags
      });

      // Clear Fuse instances to force rebuild
      this.fuseInstances.clear();
    }
  }

  /**
   * Get persistent collection by type
   */
  private getPersistentCollection(type: string): any { // Collection<CacheEntry>
    if (!this.persistentDb) {
      throw new Error('Persistent storage not initialized');
    }

    let collection = this.persistentDb.getCollection<CacheEntry>(`cache_${type}`);
    if (!collection) {
      collection = this.persistentDb.addCollection<CacheEntry>(`cache_${type}`, {
        indices: ['key'],
        ttl: this.defaultTTL * 1000 * 10 // 10x TTL for persistent
      });
    }

    return collection;
  }

  /**
   * Start cleanup interval
   */
  private startCleanupInterval(): void {
    if (browser) {
      setInterval(() => {
        // Clean up expired entries (Loki handles TTL automatically)
        // Clean up orphaned Fuse instances
        if (this.fuseInstances.size > 10) {
          // Keep only the 10 most recently used
          const entries = Array.from(this.fuseInstances.entries());
          entries.slice(10).forEach(([key]) => {
            this.fuseInstances.delete(key);
          });
        }
      }, 60000); // Every minute
    }
  }
}

// IndexedDB adapter for Loki.js
class LokiIndexedAdapter {
  constructor(private dbName: string) {}

  async loadDatabase(dbname: string, callback: (data: any) => void): Promise<void> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction(['database'], 'readonly');
      const store = transaction.objectStore('database');
      const request = store.get(dbname);

      request.onsuccess = () => {
        callback(request.result?.data || null);
      };

      request.onerror = () => {
        callback(null);
      };
    } catch (error) {
      callback(null);
    }
  }

  async saveDatabase(dbname: string, dbstring: string, callback: () => void): Promise<void> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction(['database'], 'readwrite');
      const store = transaction.objectStore('database');
      
      store.put({ id: dbname, data: dbstring });
      
      transaction.oncomplete = () => {
        callback();
      };

      transaction.onerror = () => {
        console.error('Failed to save database');
        callback();
      };
    } catch (error) {
      console.error('Database save error:', error);
      callback();
    }
  }

  private openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains('database')) {
          db.createObjectStore('database', { keyPath: 'id' });
        }
      };
    });
  }
}

// Export singleton instance
export const multiLayerCache = new MultiLayerCache();