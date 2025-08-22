import type { AIAnalysisResult, LegalDocument } from "$lib/types/legal";
import crypto from "crypto";

// Legal AI Cache Service - Performance Optimization

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  accessCount: number;
  documentHash: string;
}

interface CacheMetrics {
  hits: number;
  misses: number;
  totalRequests: number;
  averageResponseTime: number;
}

export class LegalAICache {
  private cache = new Map<string, CacheEntry<AIAnalysisResult>>();
  private readonly TTL = 1000 * 60 * 30; // 30 minutes for legal analysis
  private readonly MAX_CACHE_SIZE = 100; // Maximum cached items
  private metrics: CacheMetrics = {
    hits: 0,
    misses: 0,
    totalRequests: 0,
    averageResponseTime: 0,
  };

  constructor() {
    // Clean up expired entries every 10 minutes
    setInterval(() => this.cleanup(), 1000 * 60 * 10);
  }

  /**
   * Generate a hash for document content to use as cache key
   */
  private async generateDocumentHash(document: LegalDocument): Promise<string> {
    const content = `${document.id}-${document.title}-${document.content}-${document.lastModified}`;
    const encoder = new TextEncoder();
    const data = encoder.encode(content);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  /**
   * Get cached analysis result if available and not expired
   */
  async getCachedAnalysis(
    document: LegalDocument,
  ): Promise<AIAnalysisResult | null> {
    const startTime = performance.now();
    this.metrics.totalRequests++;

    try {
      const documentHash = await this.generateDocumentHash(document);
      const cached = this.cache.get(documentHash);

      if (cached && Date.now() - cached.timestamp < this.TTL) {
        // Update access count and metrics
        cached.accessCount++;
        this.metrics.hits++;

        const responseTime = performance.now() - startTime;
        this.updateAverageResponseTime(responseTime);

        console.log(
          `Cache HIT for document ${document.id} (hash: ${documentHash})`,
        );
        return cached.data;
      }

      // Cache miss
      this.metrics.misses++;
      console.log(`Cache MISS for document ${document.id}`);
      return null;
    } catch (error) {
      console.error("Cache lookup failed:", error);
      this.metrics.misses++;
      return null;
    }
  }

  /**
   * Store analysis result in cache
   */
  async setCachedAnalysis(
    document: LegalDocument,
    result: AIAnalysisResult,
  ): Promise<void> {
    try {
      const documentHash = await this.generateDocumentHash(document);

      // Enforce cache size limit
      if (this.cache.size >= this.MAX_CACHE_SIZE) {
        this.evictLeastUsed();
      }

      const cacheEntry: CacheEntry<AIAnalysisResult> = {
        data: result,
        timestamp: Date.now(),
        accessCount: 1,
        documentHash,
      };

      this.cache.set(documentHash, cacheEntry);

      console.log(
        `Cached analysis for document ${document.id} (hash: ${documentHash})`,
      );

      // Store in IndexedDB for persistence
      await this.persistToIndexedDB(documentHash, cacheEntry);
    } catch (error) {
      console.error("Failed to cache analysis:", error);
    }
  }

  /**
   * Remove expired entries from cache
   */
  private cleanup(): void {
    const now = Date.now();
    let removedCount = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.TTL) {
        this.cache.delete(key);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      console.log(`Cache cleanup: removed ${removedCount} expired entries`);
    }
  }

  /**
   * Evict least recently used entry when cache is full
   */
  private evictLeastUsed(): void {
    let leastUsedKey = "";
    let leastAccessCount = Infinity;
    let oldestTimestamp = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (
        entry.accessCount < leastAccessCount ||
        (entry.accessCount === leastAccessCount &&
          entry.timestamp < oldestTimestamp)
      ) {
        leastUsedKey = key;
        leastAccessCount = entry.accessCount;
        oldestTimestamp = entry.timestamp;
      }
    }

    if (leastUsedKey) {
      this.cache.delete(leastUsedKey);
      console.log(`Evicted least used cache entry: ${leastUsedKey}`);
    }
  }

  /**
   * Update average response time metric
   */
  private updateAverageResponseTime(responseTime: number): void {
    const total = this.metrics.averageResponseTime * this.metrics.hits;
    this.metrics.averageResponseTime =
      (total + responseTime) / (this.metrics.hits + 1);
  }

  /**
   * Persist cache entry to IndexedDB for longer-term storage
   */
  private async persistToIndexedDB(
    key: string,
    entry: CacheEntry<AIAnalysisResult>,
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("LegalAICache", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(["cache"], "readwrite");
        const store = transaction.objectStore("cache");

        const putRequest = store.put({ key, ...entry });
        putRequest.onsuccess = () => resolve();
        putRequest.onerror = () => reject(putRequest.error);
      };

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains("cache")) {
          const store = db.createObjectStore("cache", { keyPath: "key" });
          store.createIndex("timestamp", "timestamp");
        }
      };
    });
  }

  /**
   * Load cache entries from IndexedDB on startup
   */
  async loadFromIndexedDB(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("LegalAICache", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(["cache"], "readonly");
        const store = transaction.objectStore("cache");

        const getAllRequest = store.getAll();
        getAllRequest.onsuccess = () => {
          const entries = getAllRequest.result;
          const now = Date.now();

          for (const entry of entries) {
            // Only load non-expired entries
            if (now - entry.timestamp < this.TTL) {
              this.cache.set(entry.key, {
                data: entry.data,
                timestamp: entry.timestamp,
                accessCount: entry.accessCount,
                documentHash: entry.documentHash,
              });
            }
          }

          console.log(`Loaded ${this.cache.size} cache entries from IndexedDB`);
          resolve();
        };
        getAllRequest.onerror = () => reject(getAllRequest.error);
      };
    });
  }

  /**
   * Clear all cache entries
   */
  clearCache(): void {
    this.cache.clear();
    this.metrics = {
      hits: 0,
      misses: 0,
      totalRequests: 0,
      averageResponseTime: 0,
    };
    console.log("Cache cleared");
  }

  /**
   * Get cache statistics
   */
  getMetrics(): CacheMetrics & { cacheSize: number; hitRate: string } {
    const hitRate =
      this.metrics.totalRequests > 0
        ? ((this.metrics.hits / this.metrics.totalRequests) * 100).toFixed(1)
        : "0.0";

    return {
      ...this.metrics,
      cacheSize: this.cache.size,
      hitRate: `${hitRate}%`,
    };
  }

  /**
   * Preload commonly accessed analyses
   */
  async preloadCommonAnalyses(documents: LegalDocument[]): Promise<void> {
    console.log(`Preloading ${documents.length} common legal analyses...`);

    // This would typically load from a background service
    // For now, we'll just prepare the cache structure
    for (const document of documents) {
      const hash = await this.generateDocumentHash(document);
      // Mark as preloaded but don't actually store analysis
      console.log(`Prepared cache slot for document ${document.id} (${hash})`);
    }
  }
}

// Singleton instance
export const legalAICache = new LegalAICache();

// Initialize cache from IndexedDB on module load
legalAICache.loadFromIndexedDB().catch(console.error);
