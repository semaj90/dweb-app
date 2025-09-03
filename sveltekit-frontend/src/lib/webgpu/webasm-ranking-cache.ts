/**
 * WebAssembly Client-Side Ranking Cache
 * High-performance vector ranking with service worker concurrency
 */

import type { GPUSearchMetrics } from './webgpu-rag-service';

export interface WASMRankingEntry {
  hash: string;
  summary: Float32Array;
  rankings: Uint16Array;
  confidence: number;
  timestamp: number;
  crc32: number;
}

export interface WASMCacheConfig {
  maxEntries: number;
  ttlSeconds: number;
  enableServiceWorker: boolean;
  wasmModulePath: string;
  redisBackend?: string;
}

export interface RankingRequest {
  id: string;
  vectors: Float32Array[];
  topK: number;
  threshold?: number;
  useCache?: boolean;
}

export interface RankingResponse {
  id: string;
  rankings: Array<{ index: number; score: number; }>;
  cached: boolean;
  processingTime: number;
  wasmTime?: number;
  serviceWorkerTime?: number;
}

export interface CacheMetrics {
  hits: number;
  misses: number;
  totalRequests: number;
  hitRatio: number;
  avgWasmTime: number;
  avgServiceWorkerTime: number;
  cacheSize: number;
  memoryUsage: number;
}

/**
 * WebAssembly Ranking Cache with Service Worker Integration
 */
export class WebASMRankingCache {
  private wasmModule: WebAssembly.Module | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;
  private serviceWorker: ServiceWorkerRegistration | null = null;
  private cache = new Map<string, WASMRankingEntry>();
  private pendingRequests = new Map<string, Promise<RankingResponse>>();
  private metrics: CacheMetrics = {
    hits: 0,
    misses: 0,
    totalRequests: 0,
    hitRatio: 0,
    avgWasmTime: 0,
    avgServiceWorkerTime: 0,
    cacheSize: 0,
    memoryUsage: 0
  };

  constructor(private config: WASMCacheConfig) {}

  async initialize(): Promise<boolean> {
    try {
      console.log('🚀 Initializing WebASM Ranking Cache...');

      // 1. Initialize WebAssembly module
      await this.initializeWASM();

      // 2. Initialize Service Worker for concurrency
      if (this.config.enableServiceWorker && 'serviceWorker' in navigator) {
        await this.initializeServiceWorker();
      }

      // 3. Warm up the system
      await this.warmup();

      console.log('✅ WebASM Ranking Cache initialized successfully');
      return true;

    } catch (error) {
      console.error('❌ Failed to initialize WebASM Ranking Cache:', error);
      return false;
    }
  }

  /**
   * High-performance ranking with WebAssembly and service worker concurrency
   */
  async rank(request: RankingRequest): Promise<RankingResponse> {
    const startTime = performance.now();
    this.metrics.totalRequests++;

    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(request);
      if (request.useCache !== false) {
        const cached = this.getCachedResult(cacheKey);
        if (cached) {
          this.metrics.hits++;
          this.updateMetrics();
          return {
            id: request.id,
            rankings: this.deserializeRankings(cached.rankings, cached.summary),
            cached: true,
            processingTime: performance.now() - startTime
          };
        }
      }

      this.metrics.misses++;

      // Check for pending request to avoid duplicate computation
      if (this.pendingRequests.has(cacheKey)) {
        return await this.pendingRequests.get(cacheKey)!;
      }

      // Create and cache the ranking promise
      const rankingPromise = this.performRanking(request, cacheKey, startTime);
      this.pendingRequests.set(cacheKey, rankingPromise);

      try {
        const result = await rankingPromise;
        this.updateMetrics();
        return result;
      } finally {
        this.pendingRequests.delete(cacheKey);
      }

    } catch (error) {
      console.error('Ranking failed:', error);
      throw error;
    }
  }

  /**
   * Batch ranking with WebAssembly parallel processing
   */
  async batchRank(requests: RankingRequest[]): Promise<RankingResponse[]> {
    const startTime = performance.now();

    try {
      console.log(`🔄 Processing batch of ${requests.length} ranking requests...`);

      // Use service worker for parallel processing if available
      if (this.serviceWorker && requests.length > 1) {
        return await this.batchRankWithServiceWorker(requests);
      }

      // Fallback to sequential WebAssembly processing
      const results: RankingResponse[] = [];
      for (const request of requests) {
        const result = await this.rank(request);
        results.push(result);
      }

      const totalTime = performance.now() - startTime;
      console.log(`✅ Batch ranking completed in ${totalTime.toFixed(2)}ms`);

      return results;

    } catch (error) {
      console.error('Batch ranking failed:', error);
      throw error;
    }
  }

  /**
   * QUIC integration - publish ranking results to server cache
   */
  async publishToQUICCache(hash: string, rankings: RankingResponse): Promise<boolean> {
    try {
      // Serialize ranking data for QUIC protocol
      const payload = this.serializeForQUIC(rankings);

      const response = await fetch('/api/quic/rankings/publish', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-Cache-Key': hash,
          'X-Rankings-Count': rankings.rankings.length.toString()
        },
        body: payload
      });

      if (response.ok) {
        console.log(`📤 Published rankings ${hash} to QUIC cache`);
        return true;
      } else {
        console.warn(`⚠️ Failed to publish to QUIC cache: ${response.statusText}`);
        return false;
      }

    } catch (error) {
      console.error('QUIC publish failed:', error);
      return false;
    }
  }

  /**
   * Decode QUIC cached rankings
   */
  async decodeFromQUICCache(hash: string): Promise<RankingResponse | null> {
    try {
      const response = await fetch(`/api/quic/rankings/decode/${hash}`);
      
      if (response.ok) {
        const buffer = await response.arrayBuffer();
        const decoded = this.deserializeFromQUIC(buffer);
        console.log(`📥 Decoded rankings ${hash} from QUIC cache`);
        return decoded;
      } else if (response.status !== 404) {
        console.warn(`⚠️ QUIC decode failed: ${response.statusText}`);
      }

      return null;

    } catch (error) {
      console.error('QUIC decode failed:', error);
      return null;
    }
  }

  // ============ Private Methods ============

  private async initializeWASM(): Promise<void> {
    try {
      // Load WebAssembly module for vector operations
      const wasmResponse = await fetch(this.config.wasmModulePath);
      const wasmBytes = await wasmResponse.arrayBuffer();
      
      this.wasmModule = await WebAssembly.compile(wasmBytes);
      this.wasmInstance = await WebAssembly.instantiate(this.wasmModule, {
        env: {
          memory: new WebAssembly.Memory({ initial: 256, maximum: 1024 }),
          __wbindgen_throw: (a: number, b: number) => {
            throw new Error(`WASM error: ${a}, ${b}`);
          }
        }
      });

      console.log('✅ WebAssembly module loaded successfully');

    } catch (error) {
      console.error('❌ Failed to load WebAssembly module:', error);
      throw error;
    }
  }

  private async initializeServiceWorker(): Promise<void> {
    try {
      this.serviceWorker = await navigator.serviceWorker.register(
        '/webasm-ranking-worker.js',
        { scope: '/webasm/' }
      );

      if (this.serviceWorker.installing) {
        await new Promise<void>((resolve) => {
          this.serviceWorker!.installing!.addEventListener('statechange', function() {
            if (this.state === 'activated') {
              resolve();
            }
          });
        });
      }

      console.log('✅ Service Worker registered for concurrent processing');

    } catch (error) {
      console.warn('⚠️ Service Worker registration failed:', error);
      // Continue without service worker
    }
  }

  private async performRanking(
    request: RankingRequest,
    cacheKey: string,
    startTime: number
  ): Promise<RankingResponse> {
    const wasmStartTime = performance.now();

    // Prepare data for WebAssembly
    const vectorData = this.prepareVectorData(request.vectors);
    
    // Call WebAssembly ranking function
    const rankings = await this.callWASMRanking(
      vectorData,
      request.topK,
      request.threshold || 0.0
    );

    const wasmTime = performance.now() - wasmStartTime;

    // Cache the result
    if (request.useCache !== false) {
      const cacheEntry = this.createCacheEntry(cacheKey, rankings, vectorData);
      this.setCachedResult(cacheKey, cacheEntry);
    }

    // Publish to QUIC cache in background
    if (rankings.length > 0) {
      const response: RankingResponse = {
        id: request.id,
        rankings,
        cached: false,
        processingTime: performance.now() - startTime,
        wasmTime
      };

      // Non-blocking QUIC publish
      this.publishToQUICCache(cacheKey, response).catch(console.warn);

      return response;
    }

    return {
      id: request.id,
      rankings,
      cached: false,
      processingTime: performance.now() - startTime,
      wasmTime
    };
  }

  private async batchRankWithServiceWorker(
    requests: RankingRequest[]
  ): Promise<RankingResponse[]> {
    const swStartTime = performance.now();

    return new Promise((resolve, reject) => {
      if (!this.serviceWorker?.active) {
        reject(new Error('Service Worker not active'));
        return;
      }

      const channel = new MessageChannel();
      
      channel.port1.onmessage = (event) => {
        const { type, data, error } = event.data;
        
        if (type === 'batch-ranking-complete') {
          const results = data.map((result: any) => ({
            ...result,
            serviceWorkerTime: performance.now() - swStartTime
          }));
          resolve(results);
        } else if (type === 'batch-ranking-error') {
          reject(new Error(error));
        }
      };

      // Send batch ranking request to service worker
      this.serviceWorker.active.postMessage({
        type: 'batch-ranking-request',
        data: {
          requests,
          wasmModule: this.wasmModule,
          config: this.config
        }
      }, [channel.port2]);
    });
  }

  private async callWASMRanking(
    vectorData: Float32Array,
    topK: number,
    threshold: number
  ): Promise<Array<{ index: number; score: number; }>> {
    if (!this.wasmInstance) {
      throw new Error('WASM instance not initialized');
    }

    try {
      // Allocate memory in WASM
      const vectorPtr = this.allocateWASMMemory(vectorData.byteLength);
      const resultPtr = this.allocateWASMMemory(topK * 8); // index + score pairs

      // Copy vector data to WASM memory
      const wasmMemory = new Uint8Array((this.wasmInstance.exports.memory as WebAssembly.Memory).buffer);
      const vectorBytes = new Uint8Array(vectorData.buffer);
      wasmMemory.set(vectorBytes, vectorPtr);

      // Call WASM ranking function
      const exports = this.wasmInstance.exports as any;
      const resultCount = exports.rank_vectors(
        vectorPtr,
        vectorData.length,
        resultPtr,
        topK,
        threshold
      );

      // Read results from WASM memory
      const results: Array<{ index: number; score: number; }> = [];
      const resultData = new Float32Array(
        wasmMemory.buffer,
        resultPtr,
        resultCount * 2
      );

      for (let i = 0; i < resultCount; i++) {
        results.push({
          index: Math.floor(resultData[i * 2]),
          score: resultData[i * 2 + 1]
        });
      }

      // Free WASM memory
      exports.free(vectorPtr);
      exports.free(resultPtr);

      return results.sort((a, b) => b.score - a.score);

    } catch (error) {
      console.error('WASM ranking failed:', error);
      throw error;
    }
  }

  private prepareVectorData(vectors: Float32Array[]): Float32Array {
    // Flatten vectors into single array for WASM processing
    const totalLength = vectors.reduce((sum, v) => sum + v.length, 0);
    const flatData = new Float32Array(totalLength + vectors.length); // +length for sizes

    let offset = 0;
    vectors.forEach((vector, index) => {
      flatData[offset] = vector.length; // Store vector size
      offset++;
      flatData.set(vector, offset);
      offset += vector.length;
    });

    return flatData;
  }

  private allocateWASMMemory(size: number): number {
    if (!this.wasmInstance) {
      throw new Error('WASM instance not initialized');
    }

    const exports = this.wasmInstance.exports as any;
    return exports.malloc(size);
  }

  private generateCacheKey(request: RankingRequest): string {
    // Generate hash based on vector data and parameters
    const data = new Uint32Array([
      request.topK,
      Math.floor((request.threshold || 0) * 10000),
      request.vectors.length
    ]);

    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      hash = ((hash << 5) - hash + data[i]) & 0xffffffff;
    }

    // Add vector data hash (simplified)
    for (const vector of request.vectors.slice(0, 3)) { // Sample first 3 vectors
      for (let i = 0; i < Math.min(vector.length, 10); i++) {
        hash = ((hash << 5) - hash + Math.floor(vector[i] * 10000)) & 0xffffffff;
      }
    }

    return `rank_${Math.abs(hash).toString(16)}`;
  }

  private createCacheEntry(
    key: string,
    rankings: Array<{ index: number; score: number; }>,
    vectorData: Float32Array
  ): WASMRankingEntry {
    // Create compact summary of vector data
    const summary = new Float32Array(Math.min(vectorData.length, 384));
    summary.set(vectorData.subarray(0, summary.length));

    // Pack rankings into compact format
    const rankingsArray = new Uint16Array(rankings.length * 2);
    rankings.forEach((rank, i) => {
      rankingsArray[i * 2] = rank.index;
      rankingsArray[i * 2 + 1] = Math.floor(rank.score * 10000);
    });

    return {
      hash: key,
      summary,
      rankings: rankingsArray,
      confidence: rankings.length > 0 ? rankings[0].score : 0,
      timestamp: Date.now(),
      crc32: this.calculateCRC32(rankingsArray.buffer)
    };
  }

  private getCachedResult(key: string): WASMRankingEntry | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check TTL
    if (Date.now() - entry.timestamp > this.config.ttlSeconds * 1000) {
      this.cache.delete(key);
      return null;
    }

    // Verify integrity
    const currentCRC = this.calculateCRC32(entry.rankings.buffer);
    if (currentCRC !== entry.crc32) {
      console.warn(`⚠️ Cache integrity check failed for ${key}`);
      this.cache.delete(key);
      return null;
    }

    return entry;
  }

  private setCachedResult(key: string, entry: WASMRankingEntry): void {
    // Implement LRU eviction
    if (this.cache.size >= this.config.maxEntries) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }

    this.cache.set(key, entry);
  }

  private deserializeRankings(
    rankings: Uint16Array,
    summary: Float32Array
  ): Array<{ index: number; score: number; }> {
    const results: Array<{ index: number; score: number; }> = [];
    
    for (let i = 0; i < rankings.length; i += 2) {
      results.push({
        index: rankings[i],
        score: rankings[i + 1] / 10000
      });
    }

    return results;
  }

  private serializeForQUIC(rankings: RankingResponse): ArrayBuffer {
    // Create compact binary format for QUIC protocol
    const header = new Uint32Array([
      rankings.rankings.length,
      Math.floor(rankings.processingTime * 100),
      rankings.cached ? 1 : 0,
      Date.now() >>> 0 // Timestamp lower 32 bits
    ]);

    const rankingData = new Float32Array(rankings.rankings.length * 2);
    rankings.rankings.forEach((rank, i) => {
      rankingData[i * 2] = rank.index;
      rankingData[i * 2 + 1] = rank.score;
    });

    const totalSize = header.byteLength + rankingData.byteLength;
    const buffer = new ArrayBuffer(totalSize);
    const view = new Uint8Array(buffer);

    view.set(new Uint8Array(header.buffer), 0);
    view.set(new Uint8Array(rankingData.buffer), header.byteLength);

    return buffer;
  }

  private deserializeFromQUIC(buffer: ArrayBuffer): RankingResponse {
    const headerView = new Uint32Array(buffer, 0, 4);
    const rankingsCount = headerView[0];
    const processingTime = headerView[1] / 100;
    const cached = headerView[2] === 1;

    const rankingDataView = new Float32Array(
      buffer,
      16, // Header size
      rankingsCount * 2
    );

    const rankings: Array<{ index: number; score: number; }> = [];
    for (let i = 0; i < rankingsCount; i++) {
      rankings.push({
        index: Math.floor(rankingDataView[i * 2]),
        score: rankingDataView[i * 2 + 1]
      });
    }

    return {
      id: 'quic-decoded',
      rankings,
      cached,
      processingTime
    };
  }

  private calculateCRC32(buffer: ArrayBuffer): number {
    // Simple CRC32 implementation
    const crcTable = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let j = 0; j < 8; j++) {
        c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      }
      crcTable[i] = c;
    }

    const data = new Uint8Array(buffer);
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) {
      crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
    }
    return crc ^ 0xFFFFFFFF;
  }

  private updateMetrics(): void {
    this.metrics.hitRatio = this.metrics.hits / this.metrics.totalRequests;
    this.metrics.cacheSize = this.cache.size;
    this.metrics.memoryUsage = this.calculateMemoryUsage();
  }

  private calculateMemoryUsage(): number {
    let bytes = 0;
    for (const entry of this.cache.values()) {
      bytes += entry.summary.byteLength;
      bytes += entry.rankings.byteLength;
      bytes += 32; // Approximate overhead
    }
    return bytes;
  }

  private async warmup(): Promise<void> {
    try {
      // Warm up WebAssembly with small test
      const testVectors = [
        new Float32Array([0.1, 0.2, 0.3, 0.4]),
        new Float32Array([0.5, 0.6, 0.7, 0.8])
      ];

      const testRequest: RankingRequest = {
        id: 'warmup',
        vectors: testVectors,
        topK: 2,
        threshold: 0.0,
        useCache: false
      };

      await this.rank(testRequest);
      console.log('🔥 WebASM Ranking Cache warmed up successfully');

    } catch (error) {
      console.warn('⚠️ Warmup failed:', error);
    }
  }

  // ============ Public API ============

  getMetrics(): CacheMetrics {
    return { ...this.metrics };
  }

  clearCache(): void {
    this.cache.clear();
    this.metrics.cacheSize = 0;
    this.metrics.memoryUsage = 0;
  }

  async getQUICMetrics(): Promise<any> {
    try {
      const response = await fetch('/api/quic/rankings/metrics');
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Failed to get QUIC metrics:', error);
    }
    return null;
  }
}

// Singleton instance with default configuration
export const webASMRankingCache = new WebASMRankingCache({
  maxEntries: 1000,
  ttlSeconds: 300, // 5 minutes
  enableServiceWorker: true,
  wasmModulePath: '/webasm/ranking-cache.wasm'
});

// Auto-initialize on client side
if (typeof window !== 'undefined') {
  webASMRankingCache.initialize().catch(console.error);
}

export default webASMRankingCache;