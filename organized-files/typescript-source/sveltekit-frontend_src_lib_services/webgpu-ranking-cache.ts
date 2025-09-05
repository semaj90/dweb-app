/**
 * WebGPU-accelerated ranking cache service
 * Integrates client-side WASM ranking cache with WebGPU AI engine
 * Provides concurrent vector search result caching with protocol fallback
 */

import { webgpuAI, type WebGPUCapabilities } from '$lib/webgpu/webgpu-ai-engine'

export interface RankingResult {
  docId: number;
  score: number;
  flags: number;
  summary: string;
  url: string;
}

export interface CachePublishOptions {
  compress?: boolean;
  useWebGPU?: boolean;
  priority?: 'high' | 'medium' | 'low';
  maxAge?: number;
}

export interface CacheMetrics {
  hits: number;
  misses: number;
  compressions: number;
  decompressions: number;
  totalBytesStored: number;
  averageCompressionRatio: number;
  cacheSize: number;
  totalSlots: number;
  utilizationPercent: number;
  averageHitRatio: number;
  webgpuAcceleration: boolean;
}

export interface RankingCacheResponse<T = any> {
  success: boolean;
  found?: boolean;
  cached?: boolean;
  key?: string;
  hash?: number;
  meta?: any;
  results?: RankingResult[];
  processingTime: number;
  protocol: 'webgpu' | 'wasm' | 'fallback';
  data?: T;
  error?: string;
}

class WebGPURankingCacheService {
  private worker: Worker | null = null;
  private workerReady = false;
  private initPromise: Promise<boolean> | null = null;
  private messageId = 0;
  private pendingMessages = new Map<number, { resolve: Function; reject: Function }>();
  private webgpuCapabilities: WebGPUCapabilities | null = null;

  constructor() {
    this.initPromise = this.initialize();
  }

  /**
   * Initialize WebGPU ranking cache with WebAssembly worker
   */
  private async initialize(): Promise<boolean> {
    try {
      // Initialize WebGPU engine first
      await webgpuAI.init();
      this.webgpuCapabilities = webgpuAI.getCapabilities().webgpu;

      // Create ranking cache worker
      this.worker = new Worker('/workers/ranking-cache-worker.js');
      
      this.worker.onmessage = (event) => {
        const { id, success, result, error } = event.data;
        
        const pending = this.pendingMessages.get(id);
        if (pending) {
          this.pendingMessages.delete(id);
          if (success) {
            pending.resolve(result);
          } else {
            pending.reject(new Error(error));
          }
        }
      };

      this.worker.onerror = (error) => {
        console.error('🔥 Ranking cache worker error:', error);
        // Reject all pending messages
        for (const [id, { reject }] of this.pendingMessages) {
          reject(new Error('Worker error'));
          this.pendingMessages.delete(id);
        }
      };

      // Initialize worker WASM module
      await this.sendMessage('init', {});
      this.workerReady = true;
      
      console.log('✅ WebGPU ranking cache service initialized');
      return true;

    } catch (error) {
      console.error('❌ Failed to initialize WebGPU ranking cache:', error);
      return false;
    }
  }

  /**
   * Send message to worker and wait for response
   */
  private sendMessage(action: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = ++this.messageId;
      this.pendingMessages.set(id, { resolve, reject });
      
      this.worker.postMessage({ id, action, data });
    });
  }

  /**
   * Wait for service to be ready
   */
  async waitForReady(timeoutMs = 10000): Promise<boolean> {
    if (!this.initPromise) {
      return false;
    }

    try {
      const timeoutPromise = new Promise<boolean>((_, reject) =>
        setTimeout(() => reject(new Error('Timeout')), timeoutMs)
      );

      await Promise.race([this.initPromise, timeoutPromise]);
      return this.workerReady;
    } catch {
      return false;
    }
  }

  /**
   * Publish ranking results to cache with WebGPU acceleration
   */
  async publishRankings(
    results: RankingResult[],
    options: CachePublishOptions = {}
  ): Promise<RankingCacheResponse> {
    const startTime = performance.now();
    
    if (!this.workerReady) {
      await this.waitForReady();
    }

    try {
      // Use WebGPU for preprocessing if available and enabled
      let processedResults = results;
      let protocol: 'webgpu' | 'wasm' | 'fallback' = 'wasm';

      if (options.useWebGPU !== false && this.webgpuCapabilities?.isSupported) {
        try {
          // WebGPU-accelerated score normalization and ranking
          const scores = new Float32Array(results.map(r => r.score));
          const processed = await webgpuAI.processDimensionalArray(
            scores,
            [results.length],
            new Float32Array([0.8, 0.6, 0.4, 0.2]) // Attention weights
          );

          // Update scores with WebGPU-processed values
          processedResults = results.map((result, i) => ({
            ...result,
            score: processed.result[i]
          }));

          protocol = 'webgpu';
        } catch (webgpuError) {
          console.warn('WebGPU acceleration failed, falling back to WASM:', webgpuError);
          protocol = 'wasm';
        }
      }

      // Publish to WASM cache worker
      const workerResult = await this.sendMessage('publish', {
        results: processedResults,
        options: {
          compress: options.compress !== false,
          priority: options.priority || 'medium',
          maxAge: options.maxAge
        }
      });

      const processingTime = performance.now() - startTime;

      return {
        success: true,
        cached: workerResult.cached,
        key: workerResult.key,
        hash: workerResult.hash,
        meta: workerResult.meta,
        processingTime: processingTime,
        protocol: protocol
      };

    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime: performance.now() - startTime,
        protocol: 'fallback'
      };
    }
  }

  /**
   * Fetch cached ranking results
   */
  async fetchRankings(key: string): Promise<RankingCacheResponse<RankingResult[]>> {
    const startTime = performance.now();
    
    if (!this.workerReady) {
      await this.waitForReady();
    }

    try {
      const workerResult = await this.sendMessage('fetch', { key });
      
      const processingTime = performance.now() - startTime;

      if (!workerResult.found) {
        return {
          success: true,
          found: false,
          processingTime: processingTime,
          protocol: 'wasm'
        };
      }

      return {
        success: true,
        found: true,
        key: workerResult.key,
        meta: workerResult.meta,
        results: workerResult.results,
        processingTime: processingTime,
        protocol: 'wasm'
      };

    } catch (error) {
      return {
        success: false,
        found: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        processingTime: performance.now() - startTime,
        protocol: 'fallback'
      };
    }
  }

  /**
   * Enhanced vector search with caching integration
   */
  async searchAndCache(
    query: string,
    context: any[] = [],
    options: {
      threshold?: number;
      limit?: number;
      useCache?: boolean;
      cacheOptions?: CachePublishOptions;
    } = {}
  ): Promise<RankingCacheResponse<RankingResult[]>> {
    const startTime = performance.now();
    
    try {
      // Generate cache key from query
      const cacheKey = this.generateCacheKey(query);
      
      // Try cache first if enabled
      if (options.useCache !== false) {
        const cachedResult = await this.fetchRankings(cacheKey);
        if (cachedResult.found) {
          return {
            ...cachedResult,
            processingTime: performance.now() - startTime,
            protocol: 'webgpu'
          };
        }
      }

      // Perform vector search with WebGPU acceleration
      let searchResults: RankingResult[] = [];
      
      if (this.webgpuCapabilities?.isSupported) {
        try {
          // Use WebGPU for vector similarity computation
          const queryVector = await this.textToVector(query);
          const contextVectors = await this.processContextVectors(context);
          
          const similarities = await webgpuAI.processDimensionalArray(
            queryVector,
            [queryVector.length],
            contextVectors
          );

          // Convert to ranking results
          searchResults = similarities.result.map((score, i) => ({
            docId: i + 1,
            score: Math.max(0, Math.min(1, score)), // Clamp to [0,1]
            flags: this.categorizeContent(context[i]?.content || ''),
            summary: this.extractSummary(context[i]?.content || ''),
            url: context[i]?.url || `doc://${i}`
          }))
          .filter(result => result.score >= (options.threshold || 0.5))
          .sort((a, b) => b.score - a.score)
          .slice(0, options.limit || 10);

        } catch (webgpuError) {
          console.warn('WebGPU search failed, using fallback:', webgpuError);
          searchResults = await this.performFallbackSearch(query, context, options);
        }
      } else {
        searchResults = await this.performFallbackSearch(query, context, options);
      }

      // Cache results if enabled
      if (options.useCache !== false && searchResults.length > 0) {
        await this.publishRankings(searchResults, {
          ...options.cacheOptions,
          useWebGPU: true
        });
      }

      return {
        success: true,
        found: true,
        results: searchResults,
        processingTime: performance.now() - startTime,
        protocol: this.webgpuCapabilities?.isSupported ? 'webgpu' : 'fallback'
      };

    } catch (error) {
      return {
        success: false,
        found: false,
        error: error instanceof Error ? error.message : 'Search failed',
        processingTime: performance.now() - startTime,
        protocol: 'fallback'
      };
    }
  }

  /**
   * Get comprehensive cache metrics
   */
  async getMetrics(): Promise<CacheMetrics> {
    if (!this.workerReady) {
      await this.waitForReady();
    }

    try {
      const workerMetrics = await this.sendMessage('metrics', {});
      return {
        ...workerMetrics,
        webgpuAcceleration: this.webgpuCapabilities?.isSupported || false
      };
    } catch (error) {
      return {
        hits: 0,
        misses: 0,
        compressions: 0,
        decompressions: 0,
        totalBytesStored: 0,
        averageCompressionRatio: 0,
        cacheSize: 0,
        totalSlots: 85,
        utilizationPercent: 0,
        averageHitRatio: 0,
        webgpuAcceleration: false
      };
    }
  }

  /**
   * Clear all cached rankings
   */
  async clearCache(): Promise<boolean> {
    if (!this.workerReady) {
      return false;
    }

    try {
      await this.sendMessage('clear', {});
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Debug information
   */
  async getDebugInfo(): Promise<any> {
    if (!this.workerReady) {
      return null;
    }

    try {
      const workerDebug = await this.sendMessage('debug', {});
      return {
        ...workerDebug,
        webgpuCapabilities: this.webgpuCapabilities,
        webgpuStats: webgpuAI.getCapabilities()
      };
    } catch {
      return null;
    }
  }

  /**
   * Helper methods
   */
  private generateCacheKey(query: string): string {
    // Generate single character cache key from query hash
    const hash = this.simpleHash(query);
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~!*()+@#=|?";
    return alphabet[hash % alphabet.length];
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  private async textToVector(text: string): Promise<Float32Array> {
    // Simplified text to vector conversion
    // In production, use proper embedding model
    const vector = new Float32Array(384); // nomic-embed-text dimensions
    const words = text.toLowerCase().split(/\s+/);
    
    for (let i = 0; i < Math.min(words.length, vector.length); i++) {
      const word = words[i];
      let hash = 0;
      for (let j = 0; j < word.length; j++) {
        hash = ((hash << 5) - hash) + word.charCodeAt(j);
        hash = hash & hash;
      }
      vector[i] = (hash % 2000) / 1000.0 - 1.0; // Normalize to [-1, 1]
    }
    
    // Normalize vector
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= magnitude;
      }
    }
    
    return vector;
  }

  private async processContextVectors(context: any[]): Promise<Float32Array> {
    // Create attention weights from context
    const weights = new Float32Array(Math.min(context.length, 8));
    for (let i = 0; i < weights.length; i++) {
      weights[i] = Math.exp(-i * 0.2); // Exponential decay
    }
    
    // Normalize weights
    const sum = weights.reduce((s, w) => s + w, 0);
    for (let i = 0; i < weights.length; i++) {
      weights[i] /= sum;
    }
    
    return weights;
  }

  private categorizeContent(content: string): number {
    let flags = 0;
    if (content.match(/contract|agreement|terms/i)) flags |= 0x01;
    if (content.match(/legal|court|judge/i)) flags |= 0x02;
    if (content.match(/case|precedent|citation/i)) flags |= 0x04;
    if (content.match(/regulation|statute|law/i)) flags |= 0x08;
    return flags;
  }

  private extractSummary(content: string): string {
    const sentences = content.split(/[.!?]+/);
    return sentences[0]?.substring(0, 160) + (content.length > 160 ? '...' : '');
  }

  private async performFallbackSearch(
    query: string, 
    context: any[], 
    options: any
  ): Promise<RankingResult[]> {
    // Simple text-based similarity search fallback
    const queryWords = query.toLowerCase().split(/\s+/);
    
    return context.map((item, i) => {
      const content = (item.content || '').toLowerCase();
      const contentWords = content.split(/\s+/);
      
      // Simple word overlap scoring
      const overlap = queryWords.filter(word => 
        contentWords.some(cword => cword.includes(word) || word.includes(cword))
      ).length;
      
      const score = overlap / Math.max(queryWords.length, 1);
      
      return {
        docId: i + 1,
        score: Math.min(score, 1.0),
        flags: this.categorizeContent(item.content || ''),
        summary: this.extractSummary(item.content || ''),
        url: item.url || `doc://${i}`
      };
    })
    .filter(result => result.score >= (options.threshold || 0.3))
    .sort((a, b) => b.score - a.score)
    .slice(0, options.limit || 10);
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.workerReady = false;
    this.pendingMessages.clear();
  }
}

// Export singleton instance
export const webgpuRankingCache = new WebGPURankingCacheService();

// Export for module cleanup
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    webgpuRankingCache.destroy();
  });
}