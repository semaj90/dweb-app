import stream from "stream";

/**
 * Browser Cache Manager for Neural Sprite JSON States
 * Multi-layer caching with compression and Service Worker integration
 */

export interface BrowserCacheConfig {
  cachePrefix: string;
  maxCacheSize: number; // bytes
  enableCompression: boolean;
  enableServiceWorkerIntegration: boolean;
}

export interface CachedSprite {
  id: string;
  data: any;
  compressed: boolean;
  timestamp: number;
  accessCount: number;
  size: number;
}

export class BrowserCacheManager {
  private config: BrowserCacheConfig;
  private memoryCache: Map<string, CachedSprite> = new Map();
  private currentCacheSize = 0;
  private serviceWorkerRegistration?: ServiceWorkerRegistration;

  constructor(config: BrowserCacheConfig) {
    this.config = config;
    this.initializeServiceWorker();
    this.loadPersistedCache();
  }

  private async initializeServiceWorker(): Promise<void> {
    if (
      !this.config.enableServiceWorkerIntegration ||
      !("serviceWorker" in navigator)
    ) {
      return;
    }

    try {
      // Register our sprite caching service worker
      this.serviceWorkerRegistration = await navigator.serviceWorker.register(
        "/workers/sprite-cache-sw.js",
      );

      console.log("Sprite cache service worker registered");

      // Listen for cache updates from service worker
      navigator.serviceWorker.addEventListener(
        "message",
        this.handleServiceWorkerMessage.bind(this),
      );
    } catch (error: any) {
      console.warn("Failed to register sprite cache service worker:", error);
    }
  }

  private handleServiceWorkerMessage(event: MessageEvent): void {
    const { type, data } = event.data;

    switch (type) {
      case "SPRITE_CACHED":
        console.log(`Sprite ${data.spriteId} cached in Service Worker`);
        break;
      case "CACHE_FULL":
        console.warn(
          "Service Worker sprite cache is full, evicting old sprites",
        );
        break;
    }
  }

  private loadPersistedCache(): void {
    try {
      // Load compressed cache index from localStorage
      const cacheIndex = localStorage.getItem(
        `${this.config.cachePrefix}index`,
      );
      if (!cacheIndex) return;

      const parsed = JSON.parse(cacheIndex);
      for (const [key, metadata] of Object.entries(parsed)) {
        // Only load metadata, actual data loaded on demand
        const sprite = metadata as CachedSprite;
        sprite.data = null; // Will be loaded from IndexedDB on access
        this.memoryCache.set(key, sprite);
      }
    } catch (error: any) {
      console.warn("Failed to load persisted sprite cache:", error);
    }
  }

  public async getSprite(spriteId: string): Promise<any | null> {
    const cacheKey = this.getCacheKey(spriteId);

    // 1. Check memory cache first (fastest)
    let cached = this.memoryCache.get(cacheKey);
    if (cached && cached.data) {
      cached.accessCount++;
      cached.timestamp = Date.now();
      return this.decompressData(cached.data, cached.compressed);
    }

    // 2. Check Service Worker cache
    if (this.serviceWorkerRegistration) {
      const swCached = await this.getFromServiceWorker(spriteId);
      if (swCached) {
        // Store in memory cache for next access
        await this.cacheSprite(swCached);
        return swCached;
      }
    }

    // 3. Check IndexedDB (persistent storage)
    const idbCached = await this.getFromIndexedDB(cacheKey);
    if (idbCached) {
      // Restore to memory cache
      cached = {
        id: spriteId,
        data: idbCached,
        compressed: true,
        timestamp: Date.now(),
        accessCount: 1,
        size: JSON.stringify(idbCached).length,
      };
      this.memoryCache.set(cacheKey, cached);
      return this.decompressData(idbCached, true);
    }

    return null;
  }

  public async cacheSprite(sprite: any): Promise<void> {
    const cacheKey = this.getCacheKey(sprite.id);
    const spriteData = { ...sprite };

    // Compress JSON data if enabled
    const compressed = this.config.enableCompression
      ? await this.compressData(spriteData)
      : spriteData;

    const size = JSON.stringify(compressed).length;

    // Check cache size limits
    if (this.currentCacheSize + size > this.config.maxCacheSize) {
      await this.evictLeastUsedSprites(size);
    }

    const cached: CachedSprite = {
      id: sprite.id,
      data: compressed,
      compressed: this.config.enableCompression,
      timestamp: Date.now(),
      accessCount: 1,
      size,
    };

    // Store in memory cache
    this.memoryCache.set(cacheKey, cached);
    this.currentCacheSize += size;

    // Store in IndexedDB for persistence
    await this.storeInIndexedDB(cacheKey, compressed);

    // Cache in Service Worker for cross-tab sharing
    if (this.serviceWorkerRegistration) {
      this.cacheInServiceWorker(sprite);
    }

    // Update persistent cache index
    this.updateCacheIndex();
  }

  private async getFromServiceWorker(spriteId: string): Promise<any | null> {
    if (!this.serviceWorkerRegistration?.active) {
      return null;
    }

    return new Promise((resolve) => {
      const messageChannel = new MessageChannel();

      messageChannel.port1.onmessage = (event: any) => {
        const { data } = event.data;
        resolve(data || null);
      };

      this.serviceWorkerRegistration!.active!.postMessage(
        {
          type: "GET_SPRITE",
          spriteId,
        },
        [messageChannel.port2],
      );

      // Timeout after 100ms
      setTimeout(() => resolve(null), 100);
    });
  }

  private cacheInServiceWorker(sprite: any): void {
    if (!this.serviceWorkerRegistration?.active) {
      return;
    }

    this.serviceWorkerRegistration.active.postMessage({
      type: "CACHE_SPRITE",
      sprite,
    });
  }

  private async getFromIndexedDB(key: string): Promise<any | null> {
    return new Promise((resolve) => {
      const request = indexedDB.open(`${this.config.cachePrefix}db`, 1);

      request.onerror = () => resolve(null);

      request.onsuccess = (event: any) => {
        const db = (event.target as IDBOpenDBRequest).result;
        const transaction = db.transaction(["sprites"], "readonly");
        const store = transaction.objectStore("sprites");
        const getRequest = store.get(key);

        getRequest.onsuccess = () => resolve(getRequest.result?.data || null);
        getRequest.onerror = () => resolve(null);
      };

      request.onupgradeneeded = (event: any) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains("sprites")) {
          db.createObjectStore("sprites", { keyPath: "key" });
        }
      };
    });
  }

  private async storeInIndexedDB(key: string, data: any): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(`${this.config.cachePrefix}db`, 1);

      request.onerror = () => reject(request.error);

      request.onsuccess = (event: any) => {
        const db = (event.target as IDBOpenDBRequest).result;
        const transaction = db.transaction(["sprites"], "readwrite");
        const store = transaction.objectStore("sprites");

        store.put({ key, data, timestamp: Date.now() });
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
      };

      request.onupgradeneeded = (event: any) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains("sprites")) {
          db.createObjectStore("sprites", { keyPath: "key" });
        }
      };
    });
  }

  private async compressData(data: any): Promise<string> {
    if (!this.config.enableCompression) {
      return data;
    }

    // Use CompressionStream API if available (Chrome 80+)
    if ("CompressionStream" in window) {
      try {
        const jsonString = JSON.stringify(data);
        const stream = new CompressionStream("gzip");
        const writer = stream.writable.getWriter();
        const reader = stream.readable.getReader();

        writer.write(new TextEncoder().encode(jsonString));
        writer.close();

        const chunks: Uint8Array[] = [];
        let done = false;

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            chunks.push(value);
          }
        }

        // Convert to base64 for storage
        const totalLength = chunks.reduce(
          (acc, chunk) => acc + chunk.length,
          0,
        );
        const compressed = new Uint8Array(totalLength);
        let offset = 0;
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          compressed.set(chunk, offset);
          offset += chunk.length;
        }

        return btoa(String.fromCharCode.apply(null, Array.from(compressed)));
      } catch (error: any) {
        console.warn("Compression failed, storing uncompressed:", error);
        return JSON.stringify(data);
      }
    }

    // Fallback: Simple JSON stringification with manual compression
    const jsonString = JSON.stringify(data);

    // Basic string compression (replace common patterns)
    return jsonString.replace(/\"([^\"]{1,10})\"/g, (match, key) => {
      // Replace common JSON keys with shorter versions
      const shortcuts: Record<string, string> = {
        jsonState: "j",
        metadata: "m",
        objects: "o",
        complexity: "c",
        triggers: "t",
        usageCount: "u",
        createdAt: "ca",
      };
      return shortcuts[key] ? `"${shortcuts[key]}"` : match;
    });
  }

  private decompressData(data: any, compressed: boolean): unknown {
    if (!compressed) {
      return data;
    }

    if (typeof data === "string" && data.length > 0) {
      try {
        // Try to parse as compressed data first
        if ("DecompressionStream" in window && !data.startsWith("{")) {
          // TODO: Implement gzip decompression
          // For now, fall back to JSON parsing
        }

        // Handle manually compressed JSON
        let jsonString = data;

        // Reverse the compression shortcuts
        const shortcuts: Record<string, string> = {
          '"j"': '"jsonState"',
          '"m"': '"metadata"',
          '"o"': '"objects"',
          '"c"': '"complexity"',
          '"t"': '"triggers"',
          '"u"': '"usageCount"',
          '"ca"': '"createdAt"',
        };

        for (const [short, long] of Object.entries(shortcuts)) {
          jsonString = jsonString.replace(new RegExp(short, "g"), long);
        }

        return JSON.parse(jsonString);
      } catch (error: any) {
        console.warn("Decompression failed:", error);
        return data;
      }
    }

    return data;
  }

  private async evictLeastUsedSprites(requiredSize: number): Promise<void> {
    // Sort by access count and timestamp (LRU)
    const entries = Array.from(this.memoryCache.entries());
    const sorted = entries.sort(([, a], [, b]) => {
      if (a.accessCount !== b.accessCount) {
        return a.accessCount - b.accessCount;
      }
      return a.timestamp - b.timestamp;
    });

    let freedSize = 0;
    const toRemove: string[] = [];

    for (const [key, sprite] of sorted) {
      if (freedSize >= requiredSize) {
        break;
      }

      toRemove.push(key);
      freedSize += sprite.size;
    }

    // Remove from memory cache
    for (const key of toRemove) {
      const sprite = this.memoryCache.get(key);
      if (sprite) {
        this.currentCacheSize -= sprite.size;
        this.memoryCache.delete(key);
      }
    }

    console.log(`Evicted ${toRemove.length} sprites, freed ${freedSize} bytes`);
  }

  private getCacheKey(spriteId: string): string {
    return `${this.config.cachePrefix}${spriteId}`;
  }

  private updateCacheIndex(): void {
    try {
      const index: Record<string, Omit<CachedSprite, "data">> = {};

      const cacheEntries = Array.from(this.memoryCache.entries());
      for (let i = 0; i < cacheEntries.length; i++) {
        const [key, sprite] = cacheEntries[i];
        index[key] = {
          id: sprite.id,
          compressed: sprite.compressed,
          timestamp: sprite.timestamp,
          accessCount: sprite.accessCount,
          size: sprite.size,
        };
      }

      localStorage.setItem(
        `${this.config.cachePrefix}index`,
        JSON.stringify(index),
      );
    } catch (error: any) {
      console.warn("Failed to update cache index:", error);
    }
  }

  public getCacheStats(): {
    memorySprites: number;
    totalSize: number;
    compressionRatio: number;
    hitRate: number;
  } {
    const cacheValues = Array.from(this.memoryCache.values());
    const totalAccess = cacheValues.reduce(
      (sum, sprite) => sum + sprite.accessCount,
      0,
    );
    const compressedSprites = cacheValues.filter((sprite) => sprite.compressed);

    return {
      memorySprites: this.memoryCache.size,
      totalSize: this.currentCacheSize,
      compressionRatio: compressedSprites.length / this.memoryCache.size,
      hitRate:
        totalAccess > 0
          ? (totalAccess - this.memoryCache.size) / totalAccess
          : 0,
    };
  }

  public async clearCache(): Promise<void> {
    this.memoryCache.clear();
    this.currentCacheSize = 0;

    // Clear IndexedDB
    try {
      const request = indexedDB.deleteDatabase(`${this.config.cachePrefix}db`);
      await new Promise((resolve) => {
        request.onsuccess = () => resolve(void 0);
        request.onerror = () => resolve(void 0);
      });
    } catch (error: any) {
      console.warn("Failed to clear IndexedDB cache:", error);
    }

    // Clear localStorage index
    localStorage.removeItem(`${this.config.cachePrefix}index`);

    // Clear Service Worker cache
    if (this.serviceWorkerRegistration?.active) {
      this.serviceWorkerRegistration.active.postMessage({
        type: "CLEAR_CACHE",
      });
    }
  }
}
